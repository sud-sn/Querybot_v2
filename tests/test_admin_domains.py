# -*- coding: utf-8 -*-
"""The admin surface that makes subject areas exist at all.

``core/domains.py`` and ``core/corroboration_run.py`` were built, tested and
wired into the pipeline, and none of it could ever run: ``store.save_domain``
had no caller outside its own module, so ``list_domains`` returned ``[]`` for
every tenant, the narrowing block was a no-op on every question, and the
corroboration gate never opened. The plan document recorded domains as having
"working, tested JSON endpoints". There was no endpoint.

So the tests that matter here are the ones that run write API to read API with
nothing handed between them by the test: the admin route saves, and then the
PIPELINE's own read path — ``store.list_domains`` into
``core.domains.narrow_scope`` — is asked what a question would do. A test that
passed a domain dict into ``narrow_scope`` itself would pass against exactly
the state this fixes.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import admin.routes as routes  # noqa: E402


def _request(query=None):
    request = MagicMock()
    request.query_params = query or {}
    request.session = {"admin_id": "admin_user_1"}
    return request


class _RealStore(unittest.TestCase):
    """A real database, because the point is that the write reaches the read."""

    KB_TABLES = ["SALES.ORDERS", "SALES.CUSTOMER", "FIN.GL", "FIN.AP", "OPS.STOCK"]

    def setUp(self):
        import store

        self._dir = tempfile.mkdtemp(prefix="qb-admin-dom-")
        self._saved = {k: os.environ.get(k) for k in ("DB_PATH", "QUERYBOT_DB_PATH")}
        path = os.path.join(self._dir, "d.db")
        os.environ["DB_PATH"] = path
        os.environ["QUERYBOT_DB_PATH"] = path
        store.init_db()
        self.account_id = f"acct-adm-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")
        self.client = {
            "account_id": self.account_id,
            "client_name": "Test Corp",
            "state": "READY",
            "state_data": '{"kb_tables": %s}' % str(self.KB_TABLES).replace("'", '"'),
        }

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self._dir, ignore_errors=True)

    def _save(self, **kwargs):
        payload = {"name": "Sales", "original_name": "", "description": "",
                   "synonyms": "", "tables": []}
        payload.update(kwargs)
        with patch.object(routes, "_is_auth", return_value=True):
            return asyncio.run(routes.domains_save(
                _request(), self.account_id, **payload))

    def _delete(self, name):
        with patch.object(routes, "_is_auth", return_value=True):
            return asyncio.run(routes.domains_delete(
                _request(), self.account_id, name=name))


class TestTheWriteReachesThePipelinesRead(_RealStore):

    def test_an_area_saved_in_the_admin_narrows_a_users_scope(self):
        """The whole chain, with nothing passed between the halves."""
        import store
        from core.domains import narrow_scope

        self._save(name="Sales", synonyms="bookings, orders",
                   tables=["SALES.ORDERS", "SALES.CUSTOMER"])

        # From here on, exactly what core/query_pipeline.py does.
        decision = narrow_scope(
            "what were bookings last month",
            store.list_domains(self.account_id),
            effective=set(self.KB_TABLES), allowed_tables=None,
        )
        self.assertTrue(decision.applied)
        self.assertEqual(decision.domain, "Sales")
        self.assertEqual(decision.effective, {"SALES.ORDERS", "SALES.CUSTOMER"})

    def test_with_no_area_saved_the_scope_is_untouched(self):
        # The state this fixes: list_domains returns nothing and the block is
        # a no-op. Proves the test above is not passing on the empty case.
        import store
        from core.domains import narrow_scope

        decision = narrow_scope(
            "what were bookings last month", store.list_domains(self.account_id),
            effective=set(self.KB_TABLES), allowed_tables=None,
        )
        self.assertFalse(decision.applied)
        self.assertEqual(decision.effective, set(self.KB_TABLES))

    def test_two_saved_areas_open_the_corroboration_gate(self):
        """The gate the pipeline reads before spending a second query."""
        import store
        from core.domains import narrow_scope

        self._save(name="Sales", synonyms="revenue", tables=["SALES.ORDERS"])
        self._save(name="Finance", synonyms="revenue", tables=["FIN.GL"])

        decision = narrow_scope(
            "revenue", store.list_domains(self.account_id),
            effective=set(self.KB_TABLES), allowed_tables=None,
        )
        self.assertTrue(decision.applied)
        self.assertTrue(decision.routing.should_corroborate)
        self.assertTrue(decision.secondary)
        self.assertFalse(decision.secondary & decision.effective)

    def test_the_table_names_the_route_writes_are_the_ones_routing_matches(self):
        """Bracketed and quoted refs, because that is what the route adds.

        The store already uppercases, so lower-case input proves nothing here.
        `[SALES].[ORDERS]` is the shape a schema picker produces and the shape
        the store would keep verbatim -- and a domain holding it matches no
        table anywhere, because every scope lookup is on the bare name. Every
        lookup misses and nothing raises.
        """
        import store
        from core.domains import narrow_scope

        self._save(name="Sales", tables=["[SALES].[ORDERS]", ' "sales"."customer" '])
        saved = store.get_domain(self.account_id, "Sales")
        self.assertEqual(saved["tables"], ["SALES.CUSTOMER", "SALES.ORDERS"])

        # And the names it wrote are the ones scoping actually matches.
        decision = narrow_scope(
            "sales", store.list_domains(self.account_id),
            effective=set(self.KB_TABLES), allowed_tables=None,
        )
        self.assertEqual(decision.effective, {"SALES.ORDERS", "SALES.CUSTOMER"})


class TestEditingAnArea(_RealStore):

    def test_saving_the_same_name_replaces_rather_than_duplicates(self):
        import store

        self._save(name="Sales", tables=["SALES.ORDERS"])
        self._save(name="Sales", original_name="Sales",
                   tables=["SALES.ORDERS", "SALES.CUSTOMER"], synonyms="bookings")
        areas = store.list_domains(self.account_id)
        self.assertEqual([a["name"] for a in areas], ["Sales"])
        self.assertEqual(len(areas[0]["tables"]), 2)

    def test_a_rename_does_not_leave_the_old_area_routing(self):
        """save_domain keys on the name, so a rename creates a second row.

        Left behind, the old area keeps matching its own synonyms and keeps
        winning questions — a routing decision nobody can see in the UI.
        """
        import store

        self._save(name="Sales", synonyms="bookings", tables=["SALES.ORDERS"])
        self._save(name="Commercial", original_name="Sales",
                   synonyms="bookings", tables=["SALES.ORDERS"])
        names = [a["name"] for a in store.list_domains(self.account_id)]
        self.assertEqual(names, ["Commercial"])

    def test_removing_an_area_stops_it_routing(self):
        import store
        from core.domains import narrow_scope

        self._save(name="Sales", synonyms="bookings", tables=["SALES.ORDERS"])
        self._delete("Sales")
        decision = narrow_scope(
            "what were bookings", store.list_domains(self.account_id),
            effective=set(self.KB_TABLES), allowed_tables=None,
        )
        self.assertFalse(decision.applied)

    def test_a_nameless_area_is_refused_rather_than_saved(self):
        """Refused by the route, not by the exception handler behind it.

        The store raises on a blank name too, so "nothing was written" holds
        either way -- what the route's own check buys is that a typo comes back
        as a short message instead of a logged error and a stack-trace string.
        """
        import store

        response = self._save(name="   ")
        self.assertEqual(store.list_domains(self.account_id), [])
        location = response.headers["location"]
        self.assertIn("error=", location)
        self.assertIn("A%20name%20is%20required", location)


class TestThePageAndItsPreview(_RealStore):

    def _page(self):
        with patch.object(routes, "_is_auth", return_value=True), \
                patch.object(routes.store, "get_client", return_value=self.client):
            return asyncio.run(routes.domains_page(_request(), self.account_id))

    def test_the_table_picker_offers_the_workspaces_own_tables(self):
        with patch.object(routes, "_is_auth", return_value=True), \
                patch.object(routes.store, "get_client", return_value=self.client), \
                patch.object(routes, "_resp", side_effect=lambda r, t, c: c) as resp:
            context = asyncio.run(routes.domains_page(_request(), self.account_id))
        resp.assert_called_once()
        self.assertEqual(context["table_choices"], sorted(self.KB_TABLES))

    def test_the_page_reports_tables_that_belong_to_no_area(self):
        self._save(name="Sales", tables=["SALES.ORDERS"])
        with patch.object(routes, "_is_auth", return_value=True), \
                patch.object(routes.store, "get_client", return_value=self.client), \
                patch.object(routes, "_resp", side_effect=lambda r, t, c: c):
            context = asyncio.run(routes.domains_page(_request(), self.account_id))
        self.assertNotIn("SALES.ORDERS", context["unassigned"])
        self.assertIn("FIN.GL", context["unassigned"])

    def test_the_page_reports_a_table_two_areas_share(self):
        # Two areas that share their facts will always agree, so a second
        # opinion between them confirms nothing.
        self._save(name="Sales", tables=["SALES.ORDERS", "SALES.CUSTOMER"])
        self._save(name="Finance", tables=["SALES.ORDERS", "FIN.GL"])
        with patch.object(routes, "_is_auth", return_value=True), \
                patch.object(routes.store, "get_client", return_value=self.client), \
                patch.object(routes, "_resp", side_effect=lambda r, t, c: c):
            context = asyncio.run(routes.domains_page(_request(), self.account_id))
        self.assertEqual(sorted(context["shared"]), ["SALES.ORDERS"])
        self.assertEqual(sorted(context["shared"]["SALES.ORDERS"]), ["Finance", "Sales"])

    def test_the_preview_answers_with_the_area_that_would_win(self):
        self._save(name="Sales", synonyms="bookings", tables=["SALES.ORDERS"])
        with patch.object(routes, "_is_auth", return_value=True):
            response = asyncio.run(routes.domains_route_preview(
                _request({"q": "what were bookings last month"}), self.account_id))
        import json

        payload = json.loads(response.body)
        self.assertEqual(payload["primary"], "Sales")
        self.assertFalse(payload["corroborates"])

    def test_the_preview_says_when_a_second_area_would_be_asked_too(self):
        self._save(name="Sales", synonyms="revenue", tables=["SALES.ORDERS"])
        self._save(name="Finance", synonyms="revenue", tables=["FIN.GL"])
        with patch.object(routes, "_is_auth", return_value=True):
            response = asyncio.run(routes.domains_route_preview(
                _request({"q": "revenue"}), self.account_id))
        import json

        payload = json.loads(response.body)
        self.assertTrue(payload["corroborates"])
        self.assertTrue(payload["secondary"])

    def test_the_preview_says_when_nothing_matched(self):
        self._save(name="Sales", synonyms="bookings", tables=["SALES.ORDERS"])
        with patch.object(routes, "_is_auth", return_value=True):
            response = asyncio.run(routes.domains_route_preview(
                _request({"q": "how many widgets shipped"}), self.account_id))
        import json

        payload = json.loads(response.body)
        self.assertEqual(payload["primary"], "")
        self.assertEqual(payload["reason"], "no_domain_matched")


class TestAuthorisation(_RealStore):

    def test_an_unauthenticated_save_writes_nothing(self):
        import store

        with patch.object(routes, "_is_auth", return_value=False):
            response = asyncio.run(routes.domains_save(
                _request(), self.account_id, name="Sales", tables=["SALES.ORDERS"]))
        self.assertEqual(store.list_domains(self.account_id), [])
        self.assertIn("/admin/login", response.headers["location"])

    def test_an_unauthenticated_delete_removes_nothing(self):
        import store

        self._save(name="Sales", tables=["SALES.ORDERS"])
        with patch.object(routes, "_is_auth", return_value=False):
            asyncio.run(routes.domains_delete(
                _request(), self.account_id, name="Sales"))
        self.assertEqual([a["name"] for a in store.list_domains(self.account_id)], ["Sales"])

    def test_an_unauthenticated_preview_is_refused(self):
        from fastapi import HTTPException

        with patch.object(routes, "_is_auth", return_value=False):
            with self.assertRaises(HTTPException):
                asyncio.run(routes.domains_route_preview(
                    _request({"q": "revenue"}), self.account_id))


class TestTheNavReachesIt(unittest.TestCase):
    """A page nobody can navigate to is the state this commit is fixing."""

    def test_the_workspace_nav_links_to_it_and_lights_up_on_it(self):
        nav = (Path(__file__).resolve().parents[1]
               / "admin" / "templates" / "_client_workspace_nav.html").read_text(encoding="utf-8")
        self.assertIn('{{ _b }}/domains', nav)
        self.assertIn("'domains'", nav)

    def test_the_route_the_nav_points_at_exists(self):
        paths = {getattr(r, "path", "") for r in routes.router.routes}
        self.assertIn("/admin/clients/{account_id}/domains", paths)
        self.assertIn("/admin/clients/{account_id}/domains/save", paths)
        self.assertIn("/admin/clients/{account_id}/domains/delete", paths)
        self.assertIn("/admin/api/clients/{account_id}/domains/route", paths)


if __name__ == "__main__":
    unittest.main()
