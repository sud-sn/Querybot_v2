# -*- coding: utf-8 -*-
"""The industry vocabulary was chosen two pages after the setup that needed it.

The setup wizard offered source systems only. An industry pack -- what the
business calls things, layered on the ERP's spellings -- could be added only
afterwards, under Advanced settings on the client page, after a Knowledge
Base had already been built without it. And the naming detector's verdict on
which source system the schema looked like appeared on the Model Health page,
two steps after the admin had chosen one blind.

The wizard now carries an industry-vocabulary step right after the source
system, adding packs and never replacing the source, and the source step
shows what the discovered schema's own naming says when there is one.

Every test executes the real handlers against the real store on a temporary
database of its own, and renders the real setup page.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

MANUFACTURING = "manufacturing"
DISTRIBUTION = "wholesale_distribution"


def _arun(coro):
    return asyncio.run(coro)


class _Base(unittest.TestCase):

    def setUp(self):
        import store
        self._tmp = tempfile.mkdtemp(prefix="qb_onboarding_")
        # Never the application's data/querybot.db: every test gets a database
        # of its own, the way tests/test_preflight_reads_health.py does. The
        # store opens each connection at the path the environment names then.
        self._saved_env = {k: os.environ.get(k) for k in ("DB_PATH", "QUERYBOT_DB_PATH")}
        os.environ["DB_PATH"] = os.environ["QUERYBOT_DB_PATH"] = os.path.join(self._tmp, "store.db")
        store.init_db()
        self.store = store
        self.account = f"acct-onb-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account, "portal")

    def tearDown(self):
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self._tmp, ignore_errors=True)

    def packs(self):
        return json.loads(self.store.get_client(self.account).get("erp_packs") or "[]")

    def save_industry(self, *pack_ids):
        import admin.routes as routes
        from starlette.datastructures import FormData

        async def _form():
            return FormData([("industry_packs", pid) for pid in pack_ids])

        request = MagicMock()
        request.form = _form
        request.query_params = {}
        with patch.object(routes, "_is_auth", return_value=True):
            return _arun(routes.admin_setup_industry_vocabulary(request, self.account))

    def save_source(self, pack_id):
        import admin.routes as routes
        request = MagicMock()
        request.query_params = {}
        with patch.object(routes, "_is_auth", return_value=True):
            return _arun(routes.admin_setup_source_system(request, self.account, source_pack=pack_id))

    def render_setup(self):
        import admin.routes as routes
        from starlette.requests import Request
        scope = {"type": "http", "method": "GET", "path": f"/admin/clients/{self.account}/setup",
                 "headers": [], "query_string": b"", "session": {}}
        # The page lists every saved warehouse connection, which DECRYPTS each
        # row. By the time a full-suite run reaches this module the process has
        # been through a dozen modules that repoint QUERYBOT_KEY_FILE at their
        # own mkdtemp and re-import store, so rows written by them no longer
        # decrypt here (tests/test_client_sources.py documents the same
        # condition). None of these clients has a connection, so the listing
        # is stubbed empty on the store object routes actually bound.
        with patch.object(routes, "_is_auth", return_value=True), \
                patch.object(routes.store, "list_db_configs", return_value=[]):
            resp = _arun(routes.client_setup_page(Request(scope), self.account))
        return resp.body.decode("utf-8")

    def discovered_schema(self, tables: dict[str, list[str]]):
        """Write a discovered schema the way discovery does: one markdown file
        per table, columns in backticks down the first cell."""
        for table, columns in tables.items():
            rows = "\n".join(f"| `{c}` | varchar | Yes |  |" for c in columns)
            (Path(self._tmp) / f"{table}.md").write_text(
                f"# {table}\n\n| Column | Type | Nullable | Notes |\n|---|---|---|---|\n{rows}\n", encoding="utf-8")
        client = self.store.get_client(self.account)
        state_data = json.loads(client.get("state_data") or "{}")
        state_data["schema_dir"] = self._tmp
        self.store.update_client_state(self.account, client.get("state") or "NEW", state_data=state_data)


class TestIndustryVocabularyIsAddedNeverSubstituted(_Base):

    def test_a_pack_is_added_beside_the_source_system(self):
        self.save_source("infor_m3")
        resp = self.save_industry(MANUFACTURING)
        self.assertEqual(resp.status_code, 303)
        self.assertIn("Manufacturing", resp.headers["location"])
        self.assertEqual(self.packs(), ["infor_m3", MANUFACTURING])

    def test_two_packs_keep_the_catalogue_order(self):
        self.save_source("infor_m3")
        self.save_industry(MANUFACTURING, DISTRIBUTION)
        self.assertEqual(self.packs()[0], "infor_m3")
        self.assertEqual(set(self.packs()[1:]), {MANUFACTURING, DISTRIBUTION})

    def test_saving_none_clears_only_the_industry_packs(self):
        self.save_source("infor_m3")
        self.save_industry(MANUFACTURING)
        resp = self.save_industry()
        self.assertIn("cleared", resp.headers["location"])
        self.assertEqual(self.packs(), ["infor_m3"])

    def test_the_source_system_step_still_keeps_them(self):
        self.save_industry(MANUFACTURING)
        self.save_source("infor_m3")
        self.assertEqual(self.packs(), ["infor_m3", MANUFACTURING])
        self.save_source("netsuite")
        self.assertEqual(self.packs(), ["netsuite", MANUFACTURING])

    def test_a_source_pack_or_an_unknown_id_in_the_form_is_ignored(self):
        """The form is a list of industry packs. A tampered request naming an
        ERP pack must not turn it into a second source system, and a pack id
        that does not exist must not be stored."""
        self.save_source("infor_m3")
        self.save_industry("netsuite", "not_a_pack", MANUFACTURING)
        self.assertEqual(self.packs(), ["infor_m3", MANUFACTURING])

    def test_the_message_says_when_a_rebuild_is_needed(self):
        self.discovered_schema({"OOHEAD": ["OAORNO", "OACUNO"]})
        resp = self.save_industry(MANUFACTURING)
        self.assertIn("rebuild", resp.headers["location"].lower())
        again = self.save_industry(MANUFACTURING)
        self.assertIn("No%20rebuild", again.headers["location"])


class TestTheWizardOffersIt(_Base):

    def test_the_step_lists_every_industry_pack_and_no_source_system(self):
        html = self.render_setup()
        step = html.split('id="industry-vocabulary"')[1].split("Regulated industry")[0]
        self.assertIn('value="manufacturing"', step)
        self.assertIn('value="wholesale_distribution"', step)
        self.assertIn('value="healthcare"', step)
        self.assertNotIn('value="infor_m3"', step)
        self.assertNotIn('value="netsuite"', step)

    def test_a_selected_pack_is_shown_checked_and_named(self):
        self.save_source("infor_m3")
        self.save_industry(MANUFACTURING)
        html = self.render_setup()
        step = html.split('id="industry-vocabulary"')[1].split("Regulated industry")[0]
        self.assertRegex(step, r'value="manufacturing"\s+checked')
        self.assertIn("Manufacturing", step.split("<form")[0])

    def test_the_source_step_still_offers_no_industry_pack(self):
        html = self.render_setup()
        source = html.split('<select id="setup_source_pack"')[1].split("</select>")[0]
        self.assertIn("Infor M3", source)
        self.assertNotIn("manufacturing", source)


class TestTheSchemaSpeaksBeforeTheAdminChooses(_Base):

    def test_nothing_is_recommended_before_discovery(self):
        self.assertNotIn('id="naming-recommendation"', self.render_setup())

    def test_an_m3_shaped_schema_recommends_infor_m3(self):
        self.discovered_schema({
            "OOHEAD": ["OACONO", "OAORNO", "OAORDT", "OACUNO"],
            "OOLINE": ["OBORNO", "OBPONR", "OBITNO", "OBORQA"],
        })
        html = self.render_setup()
        self.assertIn('id="naming-recommendation"', html)
        note = html.split('id="naming-recommendation"')[1].split("</p>")[0]
        self.assertIn("Infor M3", note)
        self.assertRegex(note, r"\d+% confidence")
        self.assertIn("Select it above", note)

    def test_once_selected_the_note_stops_asking(self):
        self.discovered_schema({
            "OOHEAD": ["OACONO", "OAORNO", "OAORDT", "OACUNO"],
            "OOLINE": ["OBORNO", "OBPONR", "OBITNO", "OBORQA"],
        })
        self.save_source("infor_m3")
        note = self.render_setup().split('id="naming-recommendation"')[1].split("</p>")[0]
        self.assertNotIn("Select it above", note)

    def test_a_broken_schema_directory_never_breaks_the_page(self):
        client = self.store.get_client(self.account)
        state_data = json.loads(client.get("state_data") or "{}")
        state_data["schema_dir"] = os.path.join(self._tmp, "does-not-exist")
        self.store.update_client_state(self.account, client.get("state") or "NEW", state_data=state_data)
        html = self.render_setup()
        self.assertIn('id="industry-vocabulary"', html)
        self.assertNotIn('id="naming-recommendation"', html)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
