"""
tests/test_draft_review.py

Gather → stage → accept, against a real database.

The drafters in core/model_drafts.py are pure and tested there. This is the
half that touches the store, and its failure modes are the ones a pure test
cannot see: an evidence source keyed differently at the write site and the read
site, a proposal applied against a row that moved after it was made, a queue
that grows a duplicate every time the report is re-run.

So there are no mocked store functions here. Every test writes to a real
SQLite database, a real value index file and a real query log, then calls the
real gather, the real stage, and the real HTTP accept handler.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sqlite3
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch


class _Request:
    cookies: dict = {}
    headers: dict = {}
    url = type("U", (), {"path": "/"})()


class RealWorkspace(unittest.TestCase):
    """One account with a schema, a graph, an index and a question log."""

    QUESTIONS = [
        "revenue for shipped orders last year",
        "total revenue for shipped orders",
        "how many shipped orders",
        "cancelled orders by region",
        "which pending orders are late",
        "gross margin for March 2024",
        "gross margin for April 2025",
        "gross margin last quarter",
    ]

    def setUp(self):
        import store

        import core.value_index as value_index

        self._dir = tempfile.mkdtemp(prefix="qb-drafts-")
        self._saved = {k: os.environ.get(k) for k in ("DB_PATH", "QUERYBOT_DB_PATH")}
        path = os.path.join(self._dir, "d.db")
        os.environ["DB_PATH"] = path
        os.environ["QUERYBOT_DB_PATH"] = path
        store.init_db()

        self.account_id = f"acct-draft-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")

        schema_dir = os.path.join(self._dir, "schema")
        os.makedirs(schema_dir)
        # Keyed by table FQN at the top level, which is how the product writes
        # it. A fixture nested by schema loads as a table with no columns, and
        # the drafter then reports a workspace with no gaps.
        (Path(schema_dir) / "_schema.json").write_text(json.dumps({
            "SALES.ORDERS": {"columns": [
                {"name": "INVOICE_DATE", "type": "date"},
                {"name": "STAT_CD", "type": "varchar"},
                {"name": "NET_AMOUNT", "type": "decimal"},
            ]},
        }), encoding="utf-8")
        store.update_client_state(self.account_id, "READY", {"schema_dir": schema_dir})
        store.save_entity(account_id=self.account_id, entity_name="Orders",
                          table_name="ORDERS", schema_name="SALES",
                          entity_type="fact")

        self._index_base = os.path.join(self._dir, "vi")
        self._write_value_index({"SALES.ORDERS": {
            "STAT_CD": ["Shipped", "Pending", "Cancelled"]}})
        self._real_sample = value_index.sample_values_by_column
        base = self._index_base
        patcher = patch.object(
            value_index, "sample_values_by_column",
            side_effect=lambda account, **kw: self._real_sample(
                account, base_dir=base, **kw))
        patcher.start()
        self.addCleanup(patcher.stop)

        self._log_questions(self.QUESTIONS)

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self._dir, ignore_errors=True)

    # ── fixtures written the way production writes them ───────────────────
    def _write_value_index(self, tables: dict, *, complete: bool = True):
        import core.value_index as value_index

        path = value_index._index_path(self.account_id, self._index_base)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        conn.executescript("""
            DROP TABLE IF EXISTS column_value;
            DROP TABLE IF EXISTS column_meta;
            CREATE TABLE column_value (
              table_fqn TEXT NOT NULL, column_name TEXT NOT NULL,
              business_name TEXT NOT NULL DEFAULT '',
              value TEXT NOT NULL, value_norm TEXT NOT NULL);
            CREATE TABLE column_meta (
              table_fqn TEXT NOT NULL, column_name TEXT NOT NULL,
              distinct_count INTEGER NOT NULL DEFAULT 0,
              complete INTEGER NOT NULL DEFAULT 1,
              PRIMARY KEY (table_fqn, column_name));
        """)
        for table_fqn, columns in tables.items():
            for column, values in columns.items():
                for value in values:
                    conn.execute("INSERT INTO column_value VALUES (?,?,?,?,?)",
                                 (table_fqn, column, "", value, value.casefold()))
                conn.execute("INSERT INTO column_meta VALUES (?,?,?,?)",
                             (table_fqn, column, len(values), 1 if complete else 0))
        conn.commit()
        conn.close()

    def _log_questions(self, questions):
        import store

        with store.get_db() as conn:
            for question in questions:
                conn.execute(
                    "INSERT INTO query_log (account_id, question, created_at) "
                    "VALUES (?,?,datetime('now'))", (self.account_id, question))

    def _accept(self, proposal_id):
        import admin.routes as routes

        with patch.object(routes, "_is_auth", return_value=True):
            return asyncio.run(routes.graph_accept_change_proposal(
                _Request(), self.account_id, proposal_id))

    def _properties(self):
        import store

        return {str(row["column_name"]): dict(row)
                for row in store.list_all_entity_properties(self.account_id)}

    def _pending(self):
        import store

        return {str(p["target_id"]): p for p in
                store.list_graph_change_proposals(self.account_id, "pending")}


class TestGatheringReadsTheWorkspace(RealWorkspace):

    def test_a_date_column_in_the_schema_is_drafted(self):
        from core.draft_review import gather

        drafts = {d.target: d for d in gather(self.account_id).date_roles}
        self.assertIn("Orders.INVOICE_DATE", drafts)
        self.assertEqual(drafts["Orders.INVOICE_DATE"].payload["role"], "date")

    def test_the_entity_name_reaches_the_draft_not_the_table_name(self):
        # The value index stores SALES.ORDERS and the graph calls it Orders.
        # A drafter that reports the table name produces a proposal whose
        # target no entity_properties row can ever match, and the accept then
        # writes a row against an entity that does not exist.
        from core.draft_review import gather

        drafts = gather(self.account_id)
        for draft in drafts.date_roles + drafts.column_vocabulary:
            self.assertEqual(draft.entity, "Orders", draft)

    def test_the_table_is_found_however_the_graph_spells_its_case(self):
        # The graph stores the table as the admin typed it and the value index
        # stores it as the warehouse reports it. A case-sensitive join between
        # the two finds nothing and the drafter reports a workspace with no
        # gaps, which is indistinguishable from a workspace with none.
        import store

        from core.draft_review import gather

        # Both directions: the graph mixed-cased against an upper-cased
        # index, and an upper-cased graph against a mixed-cased index. Folding
        # only one side leaves the other broken and looks fixed.
        for graph_case, index_case in ((("Sales", "Orders"), "SALES.ORDERS"),
                                       (("SALES", "ORDERS"), "Sales.Orders")):
            with self.subTest(graph=graph_case, index=index_case):
                store.save_entity(account_id=self.account_id,
                                  entity_name="Orders",
                                  table_name=graph_case[1],
                                  schema_name=graph_case[0],
                                  entity_type="fact")
                self._write_value_index({index_case: {
                    "STAT_CD": ["Shipped", "Pending", "Cancelled"]}})
                drafts = {d.target: d
                          for d in gather(self.account_id).column_vocabulary}
                self.assertIn("Orders.STAT_CD", drafts)

    def test_the_value_index_and_the_question_log_meet(self):
        from core.draft_review import gather

        drafts = {d.target: d for d in gather(self.account_id).column_vocabulary}
        self.assertIn("Orders.STAT_CD", drafts)
        self.assertIn("orders", drafts["Orders.STAT_CD"].payload["synonyms"])

    def test_an_incomplete_column_is_not_harvested(self):
        # A column truncated at the index's build cap holds a prefix, and
        # vocabulary derived from whichever values sorted first is not
        # vocabulary, it is an artefact of the cap.
        self._write_value_index(
            {"SALES.ORDERS": {"STAT_CD": ["Shipped", "Pending", "Cancelled"]}},
            complete=False)
        from core.draft_review import gather

        self.assertEqual(gather(self.account_id).column_vocabulary, ())

    def test_an_indexed_table_with_no_entity_is_skipped(self):
        self._write_value_index({"OTHER.THING": {"STAT_CD": ["Shipped", "Pending"]}})
        from core.draft_review import gather

        self.assertEqual(gather(self.account_id).column_vocabulary, ())

    def test_repeated_question_shapes_are_reported(self):
        from core.draft_review import gather

        shapes = {d.payload["shape"] for d in gather(self.account_id).metric_shapes}
        self.assertIn("gross margin", shapes)

    def test_a_workspace_with_nothing_in_it_reports_nothing_and_does_not_raise(self):
        import store

        from core.draft_review import gather

        empty = f"acct-empty-{uuid.uuid4().hex[:8]}"
        store.upsert_client(empty, "portal")
        result = gather(empty)
        self.assertEqual(result.total, 0)
        self.assertEqual(result.error, "")


class TestStagingIsIdempotent(RealWorkspace):

    def test_the_applicable_drafts_become_pending_proposals(self):
        from core.draft_review import gather, stage

        result = stage(self.account_id, gather(self.account_id).applicable)
        self.assertEqual(result["staged_count"], 2)
        self.assertEqual(set(self._pending()),
                         {"Orders.INVOICE_DATE", "Orders.STAT_CD"})

    def test_running_it_twice_does_not_duplicate_the_queue(self):
        from core.draft_review import gather, stage

        drafts = gather(self.account_id).applicable
        stage(self.account_id, drafts)
        again = stage(self.account_id, drafts)
        self.assertEqual(again["staged_count"], 0)
        self.assertEqual(again["skipped_count"], 2)
        self.assertEqual(len(self._pending()), 2)

    def test_a_rejected_target_is_not_proposed_again(self):
        import store

        from core.draft_review import gather, stage

        drafts = gather(self.account_id).applicable
        stage(self.account_id, drafts)
        for proposal_id in [p["id"] for p in self._pending().values()]:
            store.review_graph_change_proposal(self.account_id, proposal_id, "rejected")

        again = stage(self.account_id, drafts)
        # Rejection is an answer. Re-proposing it next Tuesday is the review
        # queue arguing with the reviewer.
        self.assertEqual(again["staged_count"], 0)
        self.assertEqual({s["because"] for s in again["skipped"]}, {"rejected"})

    def test_a_metric_shape_is_never_staged_as_an_applyable_change(self):
        from core.draft_review import stage
        from core.model_drafts import Draft

        shape = Draft(kind="metric_shape", entity="", column="",
                      payload={"shape": "gross margin", "occurrences": 9},
                      reason="", confidence=90)
        result = stage(self.account_id, [shape])
        self.assertEqual(result["staged_count"], 0)
        self.assertEqual(self._pending(), {})

    def test_the_proposal_carries_the_evidence_a_reviewer_needs(self):
        from core.draft_review import gather, stage

        stage(self.account_id, gather(self.account_id).applicable)
        proposal = self._pending()["Orders.INVOICE_DATE"]
        self.assertEqual(proposal["target_kind"], "property")
        self.assertEqual(proposal["action"], "set_date_role")
        self.assertIn("INVOICE_DATE", proposal["reason"])
        self.assertGreater(proposal["confidence_score"], 0)
        self.assertEqual(proposal["generated_by"], "model_drafts")


class TestAcceptingOneAppliesIt(RealWorkspace):

    def _stage_all(self):
        from core.draft_review import gather, stage

        stage(self.account_id, gather(self.account_id).applicable)
        return self._pending()

    def test_the_date_role_reaches_the_column_the_resolver_reads(self):
        pending = self._stage_all()
        response = self._accept(pending["Orders.INVOICE_DATE"]["id"])
        self.assertEqual(response.status_code, 200)

        saved = self._properties()["INVOICE_DATE"]
        self.assertEqual(saved["role"], "date")
        self.assertEqual(saved["display_name"], "Invoice Date")
        self.assertIn("billing date", saved["synonyms"])

    def test_an_accepted_row_is_confirmed_not_left_in_the_queue(self):
        pending = self._stage_all()
        self._accept(pending["Orders.INVOICE_DATE"]["id"])
        # A human just looked at the diff and said yes, which is what
        # confirmed means. Writing it back as 'suggested' would leave it in
        # the queue it was accepted out of, and the drafter would re-propose.
        self.assertEqual(self._properties()["INVOICE_DATE"]["status"], "confirmed")

    def test_an_accepted_proposal_leaves_the_pending_queue(self):
        pending = self._stage_all()
        self._accept(pending["Orders.INVOICE_DATE"]["id"])
        self.assertNotIn("Orders.INVOICE_DATE", self._pending())

    def test_accepting_the_same_proposal_twice_is_refused(self):
        import fastapi

        pending = self._stage_all()
        proposal_id = pending["Orders.INVOICE_DATE"]["id"]
        self._accept(proposal_id)
        with self.assertRaises(fastapi.HTTPException) as caught:
            self._accept(proposal_id)
        self.assertEqual(caught.exception.status_code, 404)

    def test_the_drafter_stops_proposing_once_it_is_accepted(self):
        from core.draft_review import gather

        pending = self._stage_all()
        self._accept(pending["Orders.INVOICE_DATE"]["id"])
        targets = {d.target for d in gather(self.account_id).applicable}
        self.assertNotIn("Orders.INVOICE_DATE", targets)

    def test_a_row_confirmed_by_hand_after_staging_refuses_the_accept(self):
        import store

        pending = self._stage_all()
        # The admin got there first and said something different.
        store.save_entity_property(
            account_id=self.account_id, entity_name="Orders",
            column_name="INVOICE_DATE", role="dimension",
            display_name="Whatever the admin decided", synonyms="",
            status="confirmed")

        response = self._accept(pending["Orders.INVOICE_DATE"]["id"])
        self.assertEqual(response.status_code, 409)
        saved = self._properties()["INVOICE_DATE"]
        self.assertEqual(saved["display_name"], "Whatever the admin decided")
        self.assertEqual(saved["role"], "dimension")

    def test_a_row_that_moved_after_staging_refuses_the_accept(self):
        import store

        pending = self._stage_all()
        store.save_entity_property(
            account_id=self.account_id, entity_name="Orders",
            column_name="INVOICE_DATE", role="dimension",
            display_name="Half-edited", synonyms="", status="suggested")

        response = self._accept(pending["Orders.INVOICE_DATE"]["id"])
        self.assertEqual(response.status_code, 409)
        self.assertIn(b"changed", response.body.lower())
        self.assertEqual(self._properties()["INVOICE_DATE"]["display_name"],
                         "Half-edited")

    def test_a_refused_accept_leaves_the_proposal_pending(self):
        import store

        pending = self._stage_all()
        store.save_entity_property(
            account_id=self.account_id, entity_name="Orders",
            column_name="INVOICE_DATE", role="dimension",
            display_name="Half-edited", synonyms="", status="suggested")
        self._accept(pending["Orders.INVOICE_DATE"]["id"])
        # Refusing is not deciding. The reviewer still has to look at it.
        self.assertIn("Orders.INVOICE_DATE", self._pending())

    def test_accepting_takes_a_graph_snapshot_first(self):
        import store

        before = len(store.list_graph_versions(self.account_id)) \
            if hasattr(store, "list_graph_versions") else None
        if before is None:
            self.skipTest("this build has no graph version listing")
        pending = self._stage_all()
        self._accept(pending["Orders.INVOICE_DATE"]["id"])
        self.assertGreater(len(store.list_graph_versions(self.account_id)), before)


class TestAnAcceptedWordReachesTheResolver(RealWorkspace):
    """The drafter's whole premise is "these are the words readers used". An
    accepted word that only helps pick the TABLE has not kept that promise.

    entity_properties.synonyms reaches core.graph_resolver, which decides which
    table a question is about. It does not reach direct_aliases, which is what
    resolves a measure NAME to a column — that comes from
    table_description.column_synonyms via core.vocab_packs. So the question
    that produced the proposal still failed after somebody accepted it.
    """

    def _aliases(self, column="STAT_CD"):
        """What the resolver would see, WITHOUT clearing the cache first.

        Clearing it here would do the product's job for it: the vocabulary
        cache key is built from file mtimes, so a term saved to the database
        changes nothing it watches and the stale vocabulary is served until the
        process restarts. That is the "I accepted it and nothing happened"
        failure, and a test that resets the cache cannot see it.
        """
        from core.vocab_packs import vocab_for_account

        return {t.casefold() for t in
                (vocab_for_account(self.account_id).direct_aliases or {}).get(column, [])}

    def _accept_vocabulary(self):
        from core.draft_review import gather, stage

        stage(self.account_id, gather(self.account_id).applicable)
        pending = self._pending()
        target = next(t for t in pending if t.endswith(".STAT_CD"))
        return self._accept(pending[target]["id"]), pending[target]

    def test_before_the_accept_the_word_resolves_to_nothing(self):
        # The fixture is real before the claim is made about it.
        self.assertEqual(self._aliases(), set())

    def test_the_accept_takes_effect_without_a_restart(self):
        # Warmed first, exactly as a running process has it warmed, and never
        # cleared by the test. Without the explicit invalidation inside the
        # accept, this reads the vocabulary from before it.
        self.assertEqual(self._aliases(), set())     # warms the cache
        self._accept_vocabulary()
        self.assertTrue(self._aliases())

    def test_the_accepted_word_becomes_a_way_to_reach_the_column(self):
        response, proposal = self._accept_vocabulary()
        self.assertEqual(response.status_code, 200)

        payload = proposal["payload"]
        if isinstance(payload, str):
            import json as _json
            payload = _json.loads(payload or "{}")
        proposed = {t.strip().casefold()
                    for t in str(payload.get("synonyms") or "").split(",")
                    if t.strip()}
        self.assertTrue(proposed)
        self.assertTrue(proposed <= self._aliases(),
                        f"{proposed - self._aliases()} never reached the resolver")

    def test_it_is_also_still_on_the_property(self):
        # Both readers, not one instead of the other: the graph resolver scores
        # a table on these too.
        self._accept_vocabulary()
        self.assertTrue(self._properties()["STAT_CD"]["synonyms"])

    def test_the_terms_are_filed_under_the_entitys_table(self):
        import store

        self._accept_vocabulary()
        stored = store.list_table_descriptions(self.account_id)
        self.assertEqual(list(stored), ["SALES.ORDERS"])
        self.assertIn("STAT_CD", stored["SALES.ORDERS"]["column_synonym_map"])

    def test_a_second_accept_does_not_drop_the_first(self):
        import store

        store.save_table_description(
            self.account_id, "SALES.ORDERS",
            column_synonyms={"NET_AMOUNT": ["takings"]})
        self._accept_vocabulary()
        stored = store.list_table_descriptions(self.account_id)["SALES.ORDERS"]
        self.assertEqual(stored["column_synonym_map"]["NET_AMOUNT"], ["takings"])
        self.assertIn("STAT_CD", stored["column_synonym_map"])

    def test_a_word_an_admin_typed_on_the_column_survives_the_accept(self):
        import store

        store.save_table_description(
            self.account_id, "SALES.ORDERS",
            column_synonyms={"STAT_CD": ["progress"]})
        self._accept_vocabulary()
        terms = store.list_table_descriptions(
            self.account_id)["SALES.ORDERS"]["column_synonym_map"]["STAT_CD"]
        self.assertEqual(terms[0], "progress")
        self.assertGreater(len(terms), 1)

    def test_the_bare_table_key_an_admin_already_used_is_reused(self):
        # Descriptions are keyed by whatever the admin selected, which may be
        # bare where the graph is qualified. A second key leaves two rows for
        # one table and the one they edit is not the one this wrote.
        import store

        store.save_table_description(self.account_id, "ORDERS",
                                     column_synonyms={"NET_AMOUNT": ["takings"]})
        self._accept_vocabulary()
        self.assertEqual(list(store.list_table_descriptions(self.account_id)),
                         ["ORDERS"])

    def test_publishing_survives_a_store_that_will_not_take_it(self):
        # The accept itself must still stand, and the admin must be told.
        import store

        from core.draft_review import publish_column_terms

        with patch.object(store, "save_table_description",
                          side_effect=RuntimeError("locked")):
            message = publish_column_terms(
                self.account_id, "Orders", "STAT_CD", "delivered")
        self.assertIn("could not be published", message)

    def test_an_entity_with_no_table_says_so_rather_than_failing_quietly(self):
        import store

        from core.draft_review import publish_column_terms

        store.save_entity(account_id=self.account_id, entity_name="Ghost",
                          table_name="", schema_name="", entity_type="fact")
        message = publish_column_terms(self.account_id, "Ghost", "X", "a word")
        self.assertIn("not mapped to a table", message)

    def test_a_proposal_with_no_words_publishes_nothing(self):
        # Asserted on the WRITE, because the store drops an empty term list on
        # the way back out — so "nothing was stored" cannot tell a skipped
        # write from a pointless one.
        import store

        from core.draft_review import publish_column_terms

        with patch.object(store, "save_table_description") as saved:
            self.assertEqual(
                publish_column_terms(self.account_id, "Orders", "STAT_CD", "  ,  "), "")
        saved.assert_not_called()
        self.assertEqual(store.list_table_descriptions(self.account_id), {})


class TestTheConflictRuleItself(unittest.TestCase):
    """property_conflict, directly — every branch, without HTTP in the way."""

    BEFORE = {"role": "dimension", "display_name": "", "synonyms": ""}

    def test_an_unchanged_suggested_row_may_be_applied(self):
        from core.draft_review import property_conflict

        current = dict(self.BEFORE, status="suggested")
        self.assertEqual(property_conflict(current, self.BEFORE), "")

    def test_a_missing_row_may_be_created(self):
        from core.draft_review import property_conflict

        self.assertEqual(property_conflict(None, {}), "")

    def test_a_row_that_vanished_under_a_change_proposal_is_refused(self):
        from core.draft_review import property_conflict

        conflict = property_conflict(None, {"role": "date", "display_name": "X"})
        self.assertIn("removed", conflict)

    def test_a_confirmed_row_is_refused_even_when_it_matches(self):
        from core.draft_review import property_conflict

        current = dict(self.BEFORE, status="confirmed")
        self.assertIn("confirmed", property_conflict(current, self.BEFORE))

    def test_a_row_with_no_status_counts_as_confirmed(self):
        from core.draft_review import property_conflict

        self.assertIn("confirmed", property_conflict(dict(self.BEFORE), self.BEFORE))

    def test_each_field_is_checked(self):
        from core.draft_review import property_conflict

        for field_name, moved in (("role", "date"),
                                  ("display_name", "Something"),
                                  ("synonyms", "a, b")):
            current = dict(self.BEFORE, status="suggested", **{field_name: moved})
            with self.subTest(field=field_name):
                self.assertIn("changed", property_conflict(current, self.BEFORE))

    def test_a_target_is_split_on_the_last_dot(self):
        from core.draft_review import split_target

        self.assertEqual(split_target("Orders.INVOICE_DATE"), ("Orders", "INVOICE_DATE"))
        self.assertEqual(split_target("Sales.Orders.INVOICE_DATE"),
                         ("Sales.Orders", "INVOICE_DATE"))
        self.assertEqual(split_target("noDot"), ("noDot", ""))
        self.assertEqual(split_target(""), ("", ""))


class TestTheReviewSurface(RealWorkspace):
    """
    The page and its three buttons, rendered and posted through the real
    routes. Every feature on this branch that shipped broken shipped with
    working internals and no reachable surface, so a queue nobody can open is
    the same as no queue.
    """

    def _page(self, notice=""):
        import admin.routes as routes

        with patch.object(routes, "_is_auth", return_value=True):
            response = asyncio.run(routes.drafts_page(
                _Request(), self.account_id, notice))
        return response

    def _rendered(self, response) -> str:
        """The page as the browser gets it.

        A TemplateResponse renders on construction, so .body is the real HTML
        -- which is what makes the negative assertions here worth anything: an
        empty string would satisfy every assertNotIn in this class.
        """
        html = bytes(response.body).decode("utf-8", "replace")
        assert len(html) > 2000, "the page did not render"
        return html

    def _post(self, handler, *args):
        import admin.routes as routes

        with patch.object(routes, "_is_auth", return_value=True):
            return asyncio.run(handler(_Request(), self.account_id, *args))

    def test_the_page_renders_a_diff_for_each_pending_proposal(self):
        import admin.routes as routes

        self._post(routes.drafts_refresh)
        html = self._rendered(self._page())
        self.assertIn("Orders.INVOICE_DATE", html)
        self.assertIn("Invoice Date", html)
        # The diff shows what it would become, and says the field is unset now
        # rather than showing an empty cell the reviewer has to interpret.
        self.assertIn("not set", html)

    def test_the_refresh_button_stages_and_says_how_many(self):
        import admin.routes as routes

        response = self._post(routes.drafts_refresh)
        self.assertEqual(response.status_code, 303)
        self.assertIn("Drafted%202", response.headers["location"])
        self.assertEqual(len(self._pending()), 2)

    def test_refreshing_twice_stages_nothing_more(self):
        import admin.routes as routes

        self._post(routes.drafts_refresh)
        response = self._post(routes.drafts_refresh)
        self.assertIn("Drafted%200", response.headers["location"])

    def test_the_accept_button_applies_it(self):
        import admin.routes as routes

        self._post(routes.drafts_refresh)
        proposal_id = self._pending()["Orders.INVOICE_DATE"]["id"]
        response = self._post(routes.drafts_accept, proposal_id)
        self.assertEqual(response.status_code, 303)
        self.assertEqual(self._properties()["INVOICE_DATE"]["role"], "date")

    def test_the_accept_button_reports_a_refusal_rather_than_claiming_success(self):
        import store

        import admin.routes as routes

        self._post(routes.drafts_refresh)
        proposal_id = self._pending()["Orders.INVOICE_DATE"]["id"]
        store.save_entity_property(
            account_id=self.account_id, entity_name="Orders",
            column_name="INVOICE_DATE", role="dimension",
            display_name="Half-edited", synonyms="", status="suggested")

        response = self._post(routes.drafts_accept, proposal_id)
        location = response.headers["location"]
        self.assertNotIn("Applied", location)
        self.assertIn("changed", location)
        self.assertIn("Orders.INVOICE_DATE", self._pending())

    def test_the_reject_button_takes_it_out_of_the_queue_for_good(self):
        import admin.routes as routes

        self._post(routes.drafts_refresh)
        proposal_id = self._pending()["Orders.INVOICE_DATE"]["id"]
        self._post(routes.drafts_reject, proposal_id)
        self.assertNotIn("Orders.INVOICE_DATE", self._pending())
        self._post(routes.drafts_refresh)
        self.assertNotIn("Orders.INVOICE_DATE", self._pending())

    def test_a_graph_proposal_is_not_shown_on_the_column_review_page(self):
        # graph_change_proposal is shared with Graph Chat. An entity or
        # relationship proposal rendered here would show a diff built for a
        # different shape of row, under an Accept that applies a graph change
        # from a screen about column meanings.
        import store

        import admin.routes as routes

        store.create_graph_change_proposal(
            self.account_id, action="set_entity_filter", target_kind="entity",
            target_id="Orders", before={"entity_filter": ""},
            payload={"entity_filter": "STATUS <> 'X'"}, generated_by="chat",
            reason="a Graph Chat proposal that belongs on the graph page")
        html = self._rendered(self._page())
        self.assertNotIn("STATUS &lt;&gt; &#39;X&#39;", html)
        self.assertNotIn("a Graph Chat proposal", html)

    def test_a_field_that_would_not_change_is_left_out_of_the_diff(self):
        # Showing somebody an unchanged row and asking them to approve it
        # spends the only thing a review queue has, which is attention.
        import store

        import admin.routes as routes

        store.create_graph_change_proposal(
            self.account_id, action="set_date_role", target_kind="property",
            target_id="Orders.INVOICE_DATE",
            before={"role": "date", "display_name": "", "synonyms": ""},
            payload={"role": "date", "display_name": "Invoice Date",
                     "synonyms": "invoice date"},
            generated_by="model_drafts", reason="unchanged role")
        html = self._rendered(self._page())
        rows = html[html.index("Orders.INVOICE_DATE"):]
        table = rows[rows.index("<tbody>"):rows.index("</tbody>")]
        self.assertIn("display name", table)
        self.assertNotIn("<td>role</td>", table)

    def test_the_shapes_are_listed_without_an_accept_button(self):
        html = self._rendered(self._page())
        self.assertIn("gross margin", html)
        # An Accept here would have to invent SQL for a governed measure.
        shapes_section = html[html.index("Measures people keep asking for"):]
        self.assertNotIn("/accept", shapes_section)
        self.assertIn("Author", shapes_section)

    def test_an_empty_workspace_renders_the_empty_state_not_a_crash(self):
        import store

        import admin.routes as routes

        empty = f"acct-empty-{uuid.uuid4().hex[:8]}"
        store.upsert_client(empty, "portal")
        with patch.object(routes, "_is_auth", return_value=True):
            response = asyncio.run(routes.drafts_page(_Request(), empty, ""))
        self.assertIn("Nothing drafted", self._rendered(response))

    def test_the_page_is_linked_from_the_workspace_nav(self):
        # Reachability is the point. Every feature on this branch that shipped
        # broken shipped with working internals and no way in.
        nav = (Path(__file__).resolve().parents[1] / "admin" / "templates"
               / "_client_workspace_nav.html").read_text(encoding="utf-8")
        self.assertIn("/drafts", nav)
        self.assertIn("'drafts'", nav)

    def test_signed_out_is_redirected_not_served(self):
        import admin.routes as routes

        with patch.object(routes, "_is_auth", return_value=False):
            page = asyncio.run(routes.drafts_page(_Request(), self.account_id, ""))
            refresh = asyncio.run(routes.drafts_refresh(_Request(), self.account_id))
            accept = asyncio.run(routes.drafts_accept(_Request(), self.account_id, 1))
            reject = asyncio.run(routes.drafts_reject(_Request(), self.account_id, 1))
        for response in (page, refresh, accept, reject):
            self.assertEqual(response.status_code, 303)
            self.assertEqual(response.headers["location"], "/admin/login")

    def test_another_tenants_proposal_is_not_shown_here(self):
        import store

        import admin.routes as routes

        self._post(routes.drafts_refresh)
        other = f"acct-other-{uuid.uuid4().hex[:8]}"
        store.upsert_client(other, "portal")
        with patch.object(routes, "_is_auth", return_value=True):
            response = asyncio.run(routes.drafts_page(_Request(), other, ""))
        self.assertNotIn("Orders.INVOICE_DATE", self._rendered(response))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
