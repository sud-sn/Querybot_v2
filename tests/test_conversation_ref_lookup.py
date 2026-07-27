"""
store.get_conversation_ref_for_user — the lookup core/notify.py's Teams
delivery path uses to find a proactive-capable conversation_ref for a
portal user. Real DB integration test (not mocked) since it exercises an
actual query against pending_platform_user, following the pattern
established in tests/test_teams_tenant_mapping.py.
"""

import json
import sys
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ConversationRefLookupTests(unittest.TestCase):
    def setUp(self):
        import store
        store.init_db()
        self.store = store
        self.account_id = f"acct-convref-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")

    def tearDown(self):
        with self.store.get_db() as conn:
            conn.execute("DELETE FROM pending_platform_user WHERE account_id=?", (self.account_id,))
            conn.execute("DELETE FROM portal_user WHERE account_id=?", (self.account_id,))
            conn.execute("DELETE FROM client WHERE account_id=?", (self.account_id,))

    def _approved_pending(self, platform_user_id: str, conversation_ref: dict) -> tuple[int, dict]:
        is_new, pending = self.store.upsert_pending_user(
            self.account_id, "teams", platform_user_id, "Test User",
            json.dumps(conversation_ref),
        )
        approved = self.store.approve_pending_user(
            pending_id=pending["id"], account_id=self.account_id, group_id=None,
        )
        return approved["id"], approved

    def test_no_registration_returns_none(self):
        result = self.store.get_conversation_ref_for_user(self.account_id, 999999)
        self.assertIsNone(result)

    def test_approved_user_returns_conversation_ref(self):
        conv_ref = {"service_url": "https://smba.example/amer/", "conversation_id": "conv-abc"}
        portal_user_id, _ = self._approved_pending("teams-user-1", conv_ref)

        result = self.store.get_conversation_ref_for_user(self.account_id, portal_user_id)
        self.assertIsNotNone(result)
        self.assertEqual(result["platform_user_id"], "teams-user-1")
        self.assertEqual(json.loads(result["conversation_ref"]), conv_ref)
        self.assertEqual(result["status"], "approved")

    def test_pending_not_yet_approved_returns_none(self):
        is_new, pending = self.store.upsert_pending_user(
            self.account_id, "teams", "teams-user-2", "Test User",
            json.dumps({"service_url": "https://smba.example/", "conversation_id": "c"}),
        )
        # No approve_pending_user call — still status='pending', no portal_user_id link.
        result = self.store.get_conversation_ref_for_user(self.account_id, 1)
        self.assertIsNone(result)

    def test_rejected_user_returns_none(self):
        conv_ref = {"service_url": "https://smba.example/", "conversation_id": "conv-xyz"}
        is_new, pending = self.store.upsert_pending_user(
            self.account_id, "teams", "teams-user-3", "Test User", json.dumps(conv_ref),
        )
        self.store.reject_pending_user(pending["id"], self.account_id, reviewer_id="admin")
        # Rejected rows have no portal_user_id, so no user_id would ever match them —
        # confirm the lookup doesn't accidentally match on some other key.
        result = self.store.get_conversation_ref_for_user(self.account_id, 999999)
        self.assertIsNone(result)

    def test_scoped_to_account_id(self):
        other_account = f"acct-convref-other-{uuid.uuid4().hex[:8]}"
        self.store.upsert_client(other_account, "portal")
        try:
            conv_ref = {"service_url": "https://smba.example/", "conversation_id": "conv-1"}
            portal_user_id, _ = self._approved_pending("teams-user-4", conv_ref)

            result = self.store.get_conversation_ref_for_user(other_account, portal_user_id)
            self.assertIsNone(result)
        finally:
            with self.store.get_db() as conn:
                conn.execute("DELETE FROM pending_platform_user WHERE account_id=?", (other_account,))
                conn.execute("DELETE FROM client WHERE account_id=?", (other_account,))


if __name__ == "__main__":
    unittest.main()
