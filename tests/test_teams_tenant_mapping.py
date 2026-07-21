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
    def setUp(self):
        import store
        store.init_db()
        self.store = store
        self.a1 = f"acct-ttroute-a-{uuid.uuid4().hex[:8]}"
        self.a2 = f"acct-ttroute-b-{uuid.uuid4().hex[:8]}"
        self.tenant = "11111111-2222-3333-4444-555555555555"
        store.upsert_client(self.a1, "portal")
        store.upsert_client(self.a2, "portal")

    def tearDown(self):
        with self.store.get_db() as conn:
            conn.execute("DELETE FROM client WHERE account_id IN (?, ?)", (self.a1, self.a2))

    def _update(self, account_id, **kwargs):
        import admin.routes as routes
        from starlette.datastructures import FormData

        async def _form():
            return FormData([])

        req = type("R", (), {"form": staticmethod(_form)})()
        defaults = dict(
            client_name="", db_config_id="", llm_provider="", llm_model="",
            query_limit_monthly="", token_limit_monthly="", enable_llm_audit="",
            portal_only="", teams_tenant_id="",
        )
        defaults.update(kwargs)
        with patch.object(routes, "_is_auth", return_value=True):
            return _arun(routes.client_update(req, account_id, **defaults))

    def test_saving_teams_tenant_id_persists(self):
        resp = self._update(self.a1, teams_tenant_id=self.tenant)
        self.assertEqual(resp.status_code, 303)
        self.assertIn("saved=1", resp.headers["location"])
        self.assertEqual(self.store.get_client(self.a1)["teams_tenant_id"], self.tenant)

    def test_duplicate_tenant_redirects_with_error_not_500(self):
        self._update(self.a1, teams_tenant_id=self.tenant)
        resp = self._update(self.a2, teams_tenant_id=self.tenant)
        self.assertEqual(resp.status_code, 303)
        self.assertIn("error=", resp.headers["location"])
        self.assertEqual(self.store.get_client(self.a2)["teams_tenant_id"], "")

    def test_whitespace_only_tenant_id_saved_as_empty(self):
        resp = self._update(self.a1, teams_tenant_id="   ")
        self.assertEqual(resp.status_code, 303)
        self.assertIn("saved=1", resp.headers["location"])
        self.assertEqual(self.store.get_client(self.a1)["teams_tenant_id"], "")


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
        block = self.src[self.src.index(anchor):self.src.index(anchor) + 2500]
        mapped_pos = block.index("get_client_by_teams_tenant_id")
        heuristic_pos = block.index("configured  = [c for c in all_clients")
        self.assertLess(mapped_pos, heuristic_pos)

    def test_settings_tab_has_teams_tenant_field(self):
        src = (ROOT / "admin/templates/client_detail.html").read_text(encoding="utf-8")
        self.assertIn('name="teams_tenant_id"', src)

    def test_client_update_route_accepts_teams_tenant_id(self):
        src = (ROOT / "admin/routes.py").read_text(encoding="utf-8")
        anchor = "async def client_update("
        block = src[src.index(anchor):src.index(anchor) + 800]
        self.assertIn("teams_tenant_id", block)


if __name__ == "__main__":
    unittest.main()
