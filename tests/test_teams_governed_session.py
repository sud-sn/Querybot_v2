"""Focused tests for Teams governed session parity and isolation."""

from __future__ import annotations

import unittest
import uuid

from core.result_cache import result_cache
from gateway.teams_adapter import TeamsAdapter


def _adapter() -> TeamsAdapter:
    return TeamsAdapter({
        "app_id": "test-app",
        "app_password": "test-secret",
        "tenant_id": "test-tenant",
    })


class TeamsGovernedSessionTests(unittest.TestCase):
    def tearDown(self):
        for session_id in getattr(self, "session_ids", set()):
            result_cache.clear(session_id)

    def _bind(self, account: str, user: str) -> TeamsAdapter:
        adapter = _adapter()
        adapter.bind_session(account, user)
        if not hasattr(self, "session_ids"):
            self.session_ids = set()
        self.session_ids.add(adapter.session_id)
        return adapter

    def test_session_key_is_scoped_by_account_channel_and_user(self):
        first = self._bind("client-a", "user-1")
        second = self._bind("client-a", "user-2")
        third = self._bind("client-b", "user-1")
        self.assertEqual(first.session_id, "client-a:teams:user-1")
        self.assertEqual(len({first.session_id, second.session_id, third.session_id}), 3)

    def test_metadata_history_survives_new_adapter_for_same_identity(self):
        first = self._bind("client-a", "user-1")
        first.add_to_history(
            "Revenue for Acme",
            "SELECT * FROM sales WHERE customer_name = 'Acme'",
            ["REVENUE"],
            4,
        )
        second = self._bind("client-a", "user-1")
        history = second.get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["columns"], ["REVENUE"])
        self.assertEqual(history[0]["row_count"], 4)
        self.assertNotIn("Acme'", history[0]["sql"])
        self.assertNotIn("rows", history[0])
        first.clear_history()

    def test_history_is_not_shared_with_another_user(self):
        first = self._bind("client-a", "user-1")
        first.add_to_history("Question", "SELECT 1", ["VALUE"], 1)
        second = self._bind("client-a", "user-2")
        self.assertEqual(second.get_history(), [])
        first.clear_history()

    def test_result_cache_is_available_to_next_teams_request_for_same_user(self):
        first = self._bind("client-a", "user-1")
        first.cache_result(
            [{"WAREHOUSE": "A", "REVENUE": 10}],
            "Revenue by warehouse",
            "SELECT WAREHOUSE, REVENUE FROM result",
            question_id="question-1",
        )
        self.assertTrue(result_cache.has_result(first.session_id))
        second = self._bind("client-a", "user-1")
        self.assertTrue(result_cache.has_result(second.session_id))
        snapshot = result_cache.get_snapshot(second.session_id)
        self.assertEqual(snapshot["rows"][0]["REVENUE"], 10)

    def test_result_cache_is_not_shared_across_users_or_clients(self):
        source = self._bind("client-a", "user-1")
        source.cache_result([{"VALUE": 7}], "Q", "SELECT 7")
        other_user = self._bind("client-a", "user-2")
        other_client = self._bind("client-b", "user-1")
        self.assertFalse(result_cache.has_result(other_user.session_id))
        self.assertFalse(result_cache.has_result(other_client.session_id))


class PlatformUserTenantIsolationTests(unittest.TestCase):
    def setUp(self):
        import store

        store.init_db()
        self.store = store
        suffix = uuid.uuid4().hex[:10]
        self.account_a = f"teams-user-a-{suffix}"
        self.account_b = f"teams-user-b-{suffix}"
        self.external_id = f"teams-external-{suffix}"
        store.upsert_client(self.account_a, "portal")
        store.upsert_client(self.account_b, "portal")
        self.user_a, _ = store.create_user(
            self.account_a,
            "User A",
            f"user-a-{suffix}@example.test",
            role="analyst",
            password="Test-password-1",
        )
        self.user_b, _ = store.create_user(
            self.account_b,
            "User B",
            f"user-b-{suffix}@example.test",
            role="admin",
            password="Test-password-2",
        )
        store.link_zoom_user(self.user_a, self.external_id)
        store.link_zoom_user(self.user_b, self.external_id)

    def tearDown(self):
        with self.store.get_db() as conn:
            conn.execute("DELETE FROM portal_user WHERE id IN (?, ?)", (self.user_a, self.user_b))
            conn.execute(
                "DELETE FROM client WHERE account_id IN (?, ?)",
                (self.account_a, self.account_b),
            )

    def test_same_external_id_resolves_only_inside_requested_account(self):
        resolved_a = self.store.get_user_by_platform_id(self.account_a, self.external_id)
        resolved_b = self.store.get_user_by_platform_id(self.account_b, self.external_id)
        self.assertEqual(resolved_a["id"], self.user_a)
        self.assertEqual(resolved_b["id"], self.user_b)
        self.assertEqual(resolved_a["role"], "analyst")
        self.assertEqual(resolved_b["role"], "admin")


if __name__ == "__main__":
    unittest.main()
