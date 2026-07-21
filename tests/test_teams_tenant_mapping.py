"""Per-client Teams tenant mapping.

The Teams bot registration (app_id/app_password) is shared across every
Teams-enabled client - one bot, many workspaces. gateway/webhooks.py's
webhook_teams route used to route an inbound activity to a QueryBot client
purely by heuristic: "if there is exactly one client with a db_config
assigned, route there." That silently breaks the moment a second client is
configured (the exact failure the user hit - the bot replied "workspace
not registered" because two clients existed and the account_id needed for
a specific one wasn't the same as its Azure AD tenant ID).

This adds an explicit, deterministic mapping: client.teams_tenant_id, set
on the client's Settings tab (mirroring how db_config_id is assigned there),
enforced unique via a partial index so two clients can never silently steal
the same tenant. gateway/webhooks.py now tries this exact lookup FIRST,
before falling back to the old single-client heuristic for anyone who
hasn't set it yet.
"""
from __future__ import annotations

import asyncio
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _arun(coro):
    return asyncio.run(coro)


class TeamsTenantMappingStoreTests(unittest.TestCase):
    def setUp(self):
        import store
        store.init_db()
        self.store = store
        self.a1 = f"acct-tt-a-{uuid.uuid4().hex[:8]}"
        self.a2 = f"acct-tt-b-{uuid.uuid4().hex[:8]}"
        self.tenant = "79e7ae13-0669-4b20-93f8-308057aef35a"
        store.upsert_client(self.a1, "portal")
        store.upsert_client(self.a2, "portal")

    def tearDown(self):
        with self.store.get_db() as conn:
            conn.execute("DELETE FROM client WHERE account_id IN (?, ?)", (self.a1, self.a2))

    def test_fresh_client_has_no_mapping(self):
        self.assertEqual(self.store.get_client(self.a1)["teams_tenant_id"], "")
        self.assertIsNone(self.store.get_client_by_teams_tenant_id(self.tenant))

    def test_assign_and_look_up(self):
        self.store.update_client_meta(self.a1, teams_tenant_id=self.tenant)
        found = self.store.get_client_by_teams_tenant_id(self.tenant)
        self.assertIsNotNone(found)
        self.assertEqual(found["account_id"], self.a1)

    def test_duplicate_tenant_rejected(self):
        self.store.update_client_meta(self.a1, teams_tenant_id=self.tenant)
        with self.assertRaises(ValueError) as ctx:
            self.store.update_client_meta(self.a2, teams_tenant_id=self.tenant)
        self.assertIn("already assigned", str(ctx.exception))
        # a2 must be untouched by the failed attempt
        self.assertEqual(self.store.get_client(self.a2)["teams_tenant_id"], "")

    def test_both_clients_can_be_unmapped_simultaneously(self):
        # Empty string is the default for every client - the partial unique
        # index must not treat two blanks as a collision.
        self.store.update_client_meta(self.a1, teams_tenant_id="")
        self.store.update_client_meta(self.a2, teams_tenant_id="")
        self.assertEqual(self.store.get_client(self.a1)["teams_tenant_id"], "")
        self.assertEqual(self.store.get_client(self.a2)["teams_tenant_id"], "")

    def test_clearing_then_reassigning_to_a_different_client_works(self):
        self.store.update_client_meta(self.a1, teams_tenant_id=self.tenant)
        self.store.update_client_meta(self.a1, teams_tenant_id="")
        self.store.update_client_meta(self.a2, teams_tenant_id=self.tenant)
        self.assertEqual(self.store.get_client_by_teams_tenant_id(self.tenant)["account_id"], self.a2)

    def test_empty_lookup_returns_none(self):
        self.assertIsNone(self.store.get_client_by_teams_tenant_id(""))
        self.assertIsNone(self.store.get_client_by_teams_tenant_id("   "))


class ClientUpdateRouteTests(unittest.TestCase):
    """Exercises the real credential encrypt/decrypt round trip via
    save_platform()/get_platform(). store.crypto.KEY_FILE is a module-level
    constant bound once at import - some other test module (e.g.
    test_clarification_fixes.py) sets QUERYBOT_KEY_FILE at import time with
    no cleanup, and whichever test file's key file "wins" depends on pytest
    collection order, not on anything this class does. Pinning KEY_FILE to
    a private temp file for the duration of this class makes it hermetic
    against that ordering, matching this codebase's crypto module design
    (a plain file path, not a fixture) rather than depending on the
    ambient key surviving unchanged across the whole suite."""

    def setUp(self):
        import store
        import store.crypto as crypto
        import tempfile

        self._key_tmpdir = tempfile.mkdtemp(prefix="querybot_test_key_")
        # Different test files' sys.path insertions resolve "store" to more
        # than one module object in this suite (confirmed: id(this test's
        # `store`) != id(admin.routes.store)) - a pytest collection-order
        # artifact, not something introduced here. Patching store.crypto
        # via a fresh import in THIS module doesn't reach the KEY_FILE the
        # route handler actually reads. Patch the exact globals dict
        # admin.routes' own save_platform/get_platform calls resolve
        # against instead - correct regardless of which copy exists.
        import admin.routes as routes
        self._crypto_globals = routes.store.get_platform.__globals__["decrypt_json"].__globals__
        self._orig_key_file = self._crypto_globals["KEY_FILE"]
        self._crypto_globals["KEY_FILE"] = Path(self._key_tmpdir) / "test_key"
        crypto.KEY_FILE = self._crypto_globals["KEY_FILE"]

        store.init_db()
        self.store = store
        self.a1 = f"acct-ttroute-a-{uuid.uuid4().hex[:8]}"
        self.a2 = f"acct-ttroute-b-{uuid.uuid4().hex[:8]}"
        self.tenant = "11111111-2222-3333-4444-555555555555"
        store.upsert_client(self.a1, "portal")
        store.upsert_client(self.a2, "portal")
        self.platform_id = store.save_platform("teams", "Teams — Test Bot", {
            "app_id": "app", "app_password": "pw", "tenant_id": self.tenant,
        })
        # save_platform() itself requires tenant_id for "teams" (PLATFORM_FIELDS),
        # so this legacy/inconsistent-row case (defense-in-depth in client_update)
        # can only be constructed by writing the row directly, bypassing that
        # validation - simulating data from before tenant_id was required.
        from store.crypto import encrypt
        with store.get_db() as conn:
            cur = conn.execute(
                """INSERT INTO platform_config (platform_type, name, is_active, credentials_encrypted)
                   VALUES ('teams', 'Teams — No Tenant Yet', 1, ?)""",
                (encrypt({"app_id": "app2", "app_password": "pw2", "tenant_id": ""}),),
            )
            self.no_tenant_platform_id = cur.lastrowid

    def tearDown(self):
        with self.store.get_db() as conn:
            conn.execute("DELETE FROM client WHERE account_id IN (?, ?)", (self.a1, self.a2))
            conn.execute(
                "DELETE FROM platform_config WHERE id IN (?, ?)",
                (self.platform_id, self.no_tenant_platform_id),
            )
        import shutil
        self._crypto_globals["KEY_FILE"] = self._orig_key_file
        shutil.rmtree(self._key_tmpdir, ignore_errors=True)

    def _update(self, account_id, **kwargs):
        import admin.routes as routes
        from starlette.datastructures import FormData

        async def _form():
            return FormData([])

        req = type("R", (), {"form": staticmethod(_form)})()
        defaults = dict(
            client_name="", db_config_id="", llm_provider="", llm_model="",
            query_limit_monthly="", token_limit_monthly="", enable_llm_audit="",
            portal_only="", teams_platform_config_id="",
        )
        defaults.update(kwargs)
        with patch.object(routes, "_is_auth", return_value=True):
            return _arun(routes.client_update(req, account_id, **defaults))

    def test_assigning_platform_derives_tenant_id(self):
        resp = self._update(self.a1, teams_platform_config_id=str(self.platform_id))
        self.assertEqual(resp.status_code, 303)
        self.assertIn("saved=1", resp.headers["location"])
        client = self.store.get_client(self.a1)
        self.assertEqual(client["teams_tenant_id"], self.tenant)
        self.assertEqual(client["platform_config_id"], self.platform_id)

    def test_duplicate_assignment_redirects_with_error_not_500(self):
        self._update(self.a1, teams_platform_config_id=str(self.platform_id))
        resp = self._update(self.a2, teams_platform_config_id=str(self.platform_id))
        self.assertEqual(resp.status_code, 303)
        self.assertIn("error=", resp.headers["location"])
        self.assertEqual(self.store.get_client(self.a2)["teams_tenant_id"], "")

    def test_clearing_selection_unassigns(self):
        self._update(self.a1, teams_platform_config_id=str(self.platform_id))
        resp = self._update(self.a1, teams_platform_config_id="")
        self.assertEqual(resp.status_code, 303)
        self.assertIn("saved=1", resp.headers["location"])
        self.assertEqual(self.store.get_client(self.a1)["teams_tenant_id"], "")

    def test_platform_with_no_tenant_id_rejected(self):
        resp = self._update(self.a1, teams_platform_config_id=str(self.no_tenant_platform_id))
        self.assertEqual(resp.status_code, 303)
        self.assertIn("error=", resp.headers["location"])
        self.assertEqual(self.store.get_client(self.a1)["teams_tenant_id"], "")

    def test_unknown_platform_id_rejected(self):
        resp = self._update(self.a1, teams_platform_config_id="999999")
        self.assertIn("error=", resp.headers["location"])

    def test_non_teams_platform_rejected(self):
        zoom_id = self.store.save_platform("zoom", "Zoom bot", {
            "client_id": "c", "client_secret": "s", "bot_jid": "j", "webhook_secret": "w",
        })
        try:
            resp = self._update(self.a1, teams_platform_config_id=str(zoom_id))
            self.assertIn("error=", resp.headers["location"])
        finally:
            with self.store.get_db() as conn:
                conn.execute("DELETE FROM platform_config WHERE id=?", (zoom_id,))


class TeamsWebhookRoutingWiringTests(unittest.TestCase):
    """webhook_teams isn't practically unit-testable in isolation (real
    Bot Framework JWT verification, httpx calls) - static source assertions
    confirm the explicit mapping is tried BEFORE the old single-client
    heuristic, matching this codebase's established pattern for routes that
    are impractical to run end-to-end (see QuestionScrubWiringTests)."""

    def setUp(self):
        self.src = (ROOT / "gateway/webhooks.py").read_text(encoding="utf-8")

    def test_explicit_mapping_tried_before_heuristic_fallback(self):
        anchor = "async def webhook_teams("
        block = self.src[self.src.index(anchor):self.src.index(anchor) + 3500]
        mapped_pos = block.index("get_client_by_teams_tenant_id")
        heuristic_pos = block.index("configured  = [c for c in all_clients")
        self.assertLess(mapped_pos, heuristic_pos)

    def test_assigned_platform_is_loaded_before_global_fallback(self):
        anchor = "def _load_teams_adapter("
        block = self.src[self.src.index(anchor):self.src.index(anchor) + 1800]
        assigned_pos = block.index('store.get_platform(int(platform_id))')
        fallback_pos = block.index('return _load_adapter("teams")')
        self.assertLess(assigned_pos, fallback_pos)

    def test_teams_session_is_bound_after_client_mapping(self):
        anchor = "async def webhook_teams("
        block = self.src[self.src.index(anchor):self.src.index(anchor) + 4200]
        mapping_pos = block.index("get_client_by_teams_tenant_id")
        bind_pos = block.index("bind_session(event.account_id, event.user_id)")
        dispatch_pos = block.index("await dispatch(event.account_id")
        self.assertLess(mapping_pos, bind_pos)
        self.assertLess(bind_pos, dispatch_pos)

    def test_settings_tab_has_teams_integration_dropdown(self):
        src = (ROOT / "admin/templates/client_detail.html").read_text(encoding="utf-8")
        self.assertIn('name="teams_platform_config_id"', src)
        # Must be a <select>, not a free-text tenant ID input — the whole
        # point is picking from Admin -> Platforms, not typing a raw GUID.
        select_pos = src.index('name="teams_platform_config_id"')
        self.assertIn("<select", src[max(0, select_pos - 60):select_pos])

    def test_client_update_route_accepts_teams_platform_config_id(self):
        src = (ROOT / "admin/routes.py").read_text(encoding="utf-8")
        anchor = "async def client_update("
        block = src[src.index(anchor):src.index(anchor) + 800]
        self.assertIn("teams_platform_config_id", block)

    def test_tenant_id_derived_from_platform_not_taken_as_raw_input(self):
        src = (ROOT / "admin/routes.py").read_text(encoding="utf-8")
        anchor = "async def client_update("
        block = src[src.index(anchor):src.index(anchor) + 3000]
        self.assertIn("store.get_platform(int(teams_platform_config_id))", block)
        self.assertIn('credentials") or {}).get("tenant_id")', block)


if __name__ == "__main__":
    unittest.main()
