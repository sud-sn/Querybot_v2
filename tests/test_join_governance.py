"""
tests/test_join_governance.py

A join that can never be used is refused when it is saved, not silently at
every question afterwards.

core.join_planner has always known that a fact-to-fact edge is prohibited and
that a many-to-many needs a bridge. It just ran at QUERY time, where a refusal
becomes "no path" and the admin who saved the edge never hears about it. The
canvas returned 200 with status='confirmed' for an edge the planner would
refuse forever — and confirmed means schema discovery will never overwrite it
either, so the workspace carries a dead edge until somebody notices by hand.

Two classes, handled differently on purpose, and most of this file is about
keeping them apart:

  - structurally impossible → refuse the write
  - unverifiable → store it and flag it, because a column missing from the
    DISCOVERED schema may be a real column behind a stale discovery

Every test runs the real check or the real HTTP route against a real database.
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
from unittest.mock import patch

from core.join_governance import (
    REFUSE_INADMISSIBLE,
    REFUSE_NO_COLUMNS,
    REFUSE_SELF_JOIN,
    REFUSE_UNKNOWN_ENTITY,
    WARN_COLUMN_MISSING,
    WARN_NO_SCHEMA,
    check_join,
    join_pairs,
)

ENTITIES = {
    "Sales": {"entity_name": "Sales", "table_name": "F_SALES",
              "schema_name": "DBO", "entity_type": "fact"},
    "Stock": {"entity_name": "Stock", "table_name": "F_STOCK",
              "schema_name": "DBO", "entity_type": "fact"},
    "Customer": {"entity_name": "Customer", "table_name": "DIM_CUSTOMER",
                 "schema_name": "DBO", "entity_type": "dimension"},
    "Region": {"entity_name": "Region", "table_name": "DIM_REGION",
               "schema_name": "DBO", "entity_type": "dimension"},
    "OrderLine": {"entity_name": "OrderLine", "table_name": "BR_ORDER_LINE",
                  "schema_name": "DBO", "entity_type": "bridge"},
}

SCHEMA = {
    "DBO.F_SALES": {"CUSTOMER_ID": "int", "ITEM_ID": "int", "REGION_ID": "int"},
    "DBO.F_STOCK": {"ITEM_ID": "int"},
    "DBO.DIM_CUSTOMER": {"CUSTOMER_ID": "int", "REGION_ID": "int"},
    "DBO.DIM_REGION": {"REGION_ID": "int"},
    "DBO.BR_ORDER_LINE": {"CUSTOMER_ID": "int", "ITEM_ID": "int"},
}


def verdict(**relationship):
    base = {"from_entity": "Sales", "to_entity": "Customer",
            "from_column": "CUSTOMER_ID", "to_column": "CUSTOMER_ID"}
    base.update(relationship)
    return check_join(base, ENTITIES, SCHEMA)


class TestTheImpossibleIsRefused(unittest.TestCase):

    def test_a_fact_to_fact_join_is_refused(self):
        result = verdict(to_entity="Stock", from_column="ITEM_ID",
                         to_column="ITEM_ID")
        self.assertTrue(result.refuses)
        self.assertEqual(result.code, REFUSE_INADMISSIBLE)
        self.assertIn("fact-to-fact", result.reason)

    def test_a_many_to_many_without_a_bridge_is_refused(self):
        self.assertTrue(verdict(relationship_type="many_to_many").refuses)

    def test_a_many_to_many_through_a_bridge_is_allowed(self):
        result = verdict(to_entity="OrderLine", relationship_type="many_to_many")
        self.assertFalse(result.refuses)

    def test_a_dimension_to_fact_direction_is_refused(self):
        result = verdict(from_entity="Customer", to_entity="Sales")
        self.assertTrue(result.refuses)
        self.assertIn("direction", result.reason)

    def test_an_unknown_entity_is_refused(self):
        for side in ({"from_entity": "Ghost"}, {"to_entity": "Ghost"}):
            with self.subTest(**side):
                result = verdict(**side)
                self.assertTrue(result.refuses)
                self.assertEqual(result.code, REFUSE_UNKNOWN_ENTITY)
                self.assertIn("Ghost", result.reason)

    def test_a_blank_entity_is_refused_without_saying_none(self):
        result = verdict(to_entity="")
        self.assertTrue(result.refuses)
        self.assertIn("(blank)", result.reason)

    def test_a_join_with_no_columns_is_refused(self):
        result = verdict(from_column="", to_column="")
        self.assertTrue(result.refuses)
        self.assertEqual(result.code, REFUSE_NO_COLUMNS)

    def test_one_column_alone_is_not_a_join(self):
        self.assertTrue(verdict(to_column="").refuses)
        self.assertTrue(verdict(from_column="").refuses)

    def test_a_table_joined_to_itself_on_the_same_column_is_refused(self):
        # Matches every row to itself: no rows, no columns, no meaning, and
        # the traversal search visits the table twice.
        result = check_join(
            {"from_entity": "Customer", "to_entity": "Customer",
             "from_column": "REGION_ID", "to_column": "REGION_ID"},
            ENTITIES, SCHEMA)
        self.assertTrue(result.refuses)
        self.assertEqual(result.code, REFUSE_SELF_JOIN)

    def test_a_table_joined_to_itself_on_different_columns_is_allowed(self):
        # A genuine self-reference (parent → child in the same table) is a
        # real modelling pattern and must survive.
        result = check_join(
            {"from_entity": "Customer", "to_entity": "Customer",
             "from_column": "REGION_ID", "to_column": "CUSTOMER_ID"},
            ENTITIES, SCHEMA)
        self.assertFalse(result.refuses)


class TestEveryRefusalSaysWhatToDo(unittest.TestCase):
    """A refusal that only names a rule leaves the admin where they were."""

    def test_the_fact_to_fact_refusal_names_the_alternative(self):
        reason = verdict(to_entity="Stock", from_column="ITEM_ID",
                         to_column="ITEM_ID").reason
        self.assertIn("shared dimension", reason)

    def test_the_many_to_many_refusal_names_the_alternative(self):
        self.assertIn("bridge", verdict(relationship_type="many_to_many").reason)

    def test_the_direction_refusal_says_which_way_round(self):
        reason = verdict(from_entity="Customer", to_entity="Sales").reason
        self.assertIn("Swap", reason)
        self.assertIn("foreign key", reason)

    def test_every_remedy_key_can_actually_be_found(self):
        # The lookup lower-cases the planner's reason, so a key carrying
        # upper-case can never match — which is exactly what happened to the
        # direction remedy first time round.
        from core.join_governance import _REMEDIES

        for key in _REMEDIES:
            self.assertEqual(key, key.lower(), key)

    def test_every_planner_refusal_reason_has_a_remedy(self):
        # Walks the planner's own reasons rather than a list typed here, so a
        # new rule added there shows up as a missing remedy instead of as a
        # refusal with no advice.
        import inspect
        import re

        from core.join_planner import relationship_is_admissible
        from core.join_governance import _REMEDIES

        source = inspect.getsource(relationship_is_admissible)
        reasons = re.findall(r'return False, "([^"]+)"', source)
        self.assertGreaterEqual(len(reasons), 4, reasons)
        for reason in reasons:
            if reason.startswith("unsupported "):
                continue    # formatted with a role, not a fixed sentence
            self.assertIn(reason.lower(), _REMEDIES, reason)


class TestTheUnverifiableIsFlaggedNotRefused(unittest.TestCase):

    def test_a_column_the_schema_does_not_know_is_stored_and_flagged(self):
        result = verdict(to_column="CUST_NO")
        self.assertTrue(result.ok)
        self.assertTrue(result.flags)
        self.assertEqual(result.code, WARN_COLUMN_MISSING)
        self.assertEqual(result.missing_columns, ("Customer.CUST_NO",))

    def test_both_sides_are_reported(self):
        result = verdict(from_column="NOPE1", to_column="NOPE2")
        self.assertEqual(set(result.missing_columns),
                         {"Sales.NOPE1", "Customer.NOPE2"})

    def test_no_schema_at_all_is_a_flag_not_a_refusal(self):
        result = check_join(
            {"from_entity": "Sales", "to_entity": "Customer",
             "from_column": "ANYTHING", "to_column": "ANYTHING"},
            ENTITIES, None)
        self.assertTrue(result.ok)
        self.assertEqual(result.code, WARN_NO_SCHEMA)

    def test_a_table_missing_from_the_schema_is_a_flag_not_a_refusal(self):
        entities = dict(ENTITIES)
        entities["New"] = {"entity_name": "New", "table_name": "DIM_NEW",
                           "schema_name": "DBO", "entity_type": "dimension"}
        result = check_join(
            {"from_entity": "Sales", "to_entity": "New",
             "from_column": "CUSTOMER_ID", "to_column": "ID"},
            entities, SCHEMA)
        self.assertTrue(result.ok)
        self.assertEqual(result.code, WARN_NO_SCHEMA)

    def test_a_structurally_impossible_edge_is_refused_even_with_no_schema(self):
        # The structural rules need no schema, so losing one must not turn a
        # refusal into a pass.
        result = check_join(
            {"from_entity": "Sales", "to_entity": "Stock",
             "from_column": "ITEM_ID", "to_column": "ITEM_ID"},
            ENTITIES, None)
        self.assertTrue(result.refuses)

    def test_column_matching_ignores_case(self):
        self.assertFalse(verdict(from_column="customer_id",
                                 to_column="Customer_Id").flags)


class TestCompositeJoins(unittest.TestCase):

    def test_every_condition_pair_is_checked(self):
        result = verdict(
            to_entity="OrderLine",
            join_conditions=[{"from_col": "CUSTOMER_ID", "to_col": "CUSTOMER_ID"},
                             {"from_col": "ITEM_ID", "to_col": "MISSING"}])
        self.assertEqual(result.missing_columns, ("OrderLine.MISSING",))

    def test_conditions_arrive_as_json_or_as_a_list(self):
        conditions = [{"from_col": "ITEM_ID", "to_col": "ITEM_ID"}]
        as_list = join_pairs({"from_column": "A", "to_column": "B",
                              "join_conditions": conditions})
        as_json = join_pairs({"from_column": "A", "to_column": "B",
                              "join_conditions": json.dumps(conditions)})
        self.assertEqual(as_list, as_json)
        self.assertEqual(as_list, [("A", "B"), ("ITEM_ID", "ITEM_ID")])

    def test_malformed_conditions_do_not_crash_the_check(self):
        for broken in ("not json", "{", None, [None, 3, {}], {"x": 1}):
            with self.subTest(broken=broken):
                pairs = join_pairs({"from_column": "A", "to_column": "B",
                                    "join_conditions": broken})
                self.assertEqual(pairs, [("A", "B")])

    def test_a_composite_join_with_only_conditions_is_a_join(self):
        pairs = join_pairs({"join_conditions": [
            {"from_col": "A", "to_col": "B"}]})
        self.assertEqual(pairs, [("A", "B")])


# ── The routes ───────────────────────────────────────────────────────────────

class _Request:
    cookies: dict = {}
    headers: dict = {}
    url = type("U", (), {"path": "/"})()

    def __init__(self, body=None):
        self._body = body or {}

    async def json(self):
        return self._body


class RealWorkspace(unittest.TestCase):

    def setUp(self):
        import store

        self._dir = tempfile.mkdtemp(prefix="qb-joins-")
        self._saved = {k: os.environ.get(k) for k in ("DB_PATH", "QUERYBOT_DB_PATH")}
        path = os.path.join(self._dir, "d.db")
        os.environ["DB_PATH"] = path
        os.environ["QUERYBOT_DB_PATH"] = path
        store.init_db()
        self.account_id = f"acct-join-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")

        schema_dir = os.path.join(self._dir, "schema")
        os.makedirs(schema_dir)
        (Path(schema_dir) / "_schema.json").write_text(json.dumps({
            fqn: {"columns": [{"name": c, "type": t} for c, t in cols.items()]}
            for fqn, cols in SCHEMA.items()
        }), encoding="utf-8")
        store.update_client_state(self.account_id, "READY",
                                  {"schema_dir": schema_dir})
        for entity in ENTITIES.values():
            store.save_entity(account_id=self.account_id,
                              entity_name=entity["entity_name"],
                              table_name=entity["table_name"],
                              schema_name=entity["schema_name"],
                              entity_type=entity["entity_type"])

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self._dir, ignore_errors=True)

    def _save(self, **body):
        import admin.routes as routes

        payload = {"from_entity": "Sales", "to_entity": "Customer",
                   "from_column": "CUSTOMER_ID", "to_column": "CUSTOMER_ID"}
        payload.update(body)
        with patch.object(routes, "_is_auth", return_value=True):
            return asyncio.run(routes.graph_api_rel_upsert(
                _Request(payload), self.account_id))

    def _bulk(self, updates, deletes=()):
        import admin.routes as routes

        with patch.object(routes, "_is_auth", return_value=True):
            return asyncio.run(routes.graph_api_rel_bulk(
                _Request({"updates": updates, "deletes": list(deletes)}),
                self.account_id))

    def _rows(self):
        import store

        return store.list_relationships(self.account_id, active_only=False)


class TestTheCanvasRoute(RealWorkspace):

    def test_the_edge_that_started_this_is_now_refused(self):
        # Fact→fact, many-to-many, on a column that does not exist. This
        # returned 200 with status='confirmed'.
        response = self._save(to_entity="Stock", from_column="ITEM_ID",
                              to_column="NONEXISTENT",
                              relationship_type="many_to_many")
        self.assertEqual(response.status_code, 422)
        body = json.loads(response.body)
        self.assertEqual(body["status"], "invalid")
        self.assertEqual(body["code"], REFUSE_INADMISSIBLE)
        self.assertEqual(self._rows(), [])

    def test_a_good_edge_still_saves(self):
        response = self._save()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self._rows()), 1)

    def test_an_unverifiable_column_saves_and_is_flagged(self):
        response = self._save(to_column="CUST_NO")
        self.assertEqual(response.status_code, 200)
        row = self._rows()[0]
        self.assertEqual(row["validation_status"], "broken")

    def test_a_verified_edge_is_not_flagged_broken(self):
        self._save()
        self.assertNotEqual(self._rows()[0]["validation_status"], "broken")

    def test_the_refusal_reaches_the_caller_with_something_to_act_on(self):
        body = json.loads(self._save(from_entity="Customer",
                                     to_entity="Sales").body)
        self.assertIn("Swap", body["message"])
        self.assertTrue(body["code"])

    def test_nothing_is_written_when_the_edge_is_refused(self):
        # Not merely "no relationship": the semantic-model sync and the
        # recompile below it must not run for an edge that was never stored.
        import admin.routes as routes

        with patch.object(routes, "_after_semantic_approval") as recompile:
            self._save(to_entity="Stock", from_column="ITEM_ID",
                       to_column="ITEM_ID")
        recompile.assert_not_called()


class TestTheBulkRoute(RealWorkspace):

    GOOD = {"from_entity": "Sales", "to_entity": "Customer",
            "from_column": "CUSTOMER_ID", "to_column": "CUSTOMER_ID"}
    ALSO_GOOD = {"from_entity": "Sales", "to_entity": "Region",
                 "from_column": "REGION_ID", "to_column": "REGION_ID"}
    BAD = {"from_entity": "Sales", "to_entity": "Stock",
           "from_column": "ITEM_ID", "to_column": "ITEM_ID"}

    def test_one_bad_row_does_not_lose_the_good_ones(self):
        # An editor that refuses the whole batch because row 14 is a
        # fact-to-fact join makes the admin find the needle themselves.
        response = self._bulk([self.GOOD, self.BAD, self.ALSO_GOOD])
        body = json.loads(response.body)
        self.assertEqual(body["updated"], 2)
        self.assertEqual(len(body["rejected"]), 1)
        self.assertEqual(len(self._rows()), 2)

    def test_the_rejected_row_says_which_one_and_why(self):
        body = json.loads(self._bulk([self.BAD]).body)
        rejected = body["rejected"][0]
        self.assertIn("Sales", rejected["target"])
        self.assertIn("Stock", rejected["target"])
        self.assertEqual(rejected["code"], REFUSE_INADMISSIBLE)
        self.assertIn("fact-to-fact", rejected["message"])

    def test_a_batch_of_nothing_but_bad_rows_changes_nothing(self):
        import admin.routes as routes

        with patch.object(routes, "_after_semantic_approval") as recompile:
            body = json.loads(self._bulk([self.BAD]).body)
        self.assertEqual(body["updated"], 0)
        self.assertEqual(self._rows(), [])
        recompile.assert_not_called()

    def test_deletes_still_happen_alongside_a_rejection(self):
        import store

        rel_id = store.save_relationship(
            account_id=self.account_id, from_entity="Sales",
            to_entity="Region", from_column="REGION_ID", to_column="REGION_ID")
        body = json.loads(self._bulk([self.BAD], deletes=[rel_id]).body)
        self.assertEqual(body["deleted"], 1)
        self.assertEqual(len(body["rejected"]), 1)
        self.assertEqual(self._rows(), [])


class TestGraphHealthFindsTheOnesAlreadySaved(RealWorkspace):
    """
    J1 refuses these at the moment of saving. This is how the ones saved
    before it get found.

    A workspace whose only fact tables were wired directly to each other, on a
    column neither of them has, scored 94/100 and was told its entities had no
    field properties. Seven health checks, none of which asked whether a join
    could be traversed.
    """

    def _health(self):
        from core.graph_health import check_graph_health

        return check_graph_health(self.account_id)

    def _codes(self, report):
        return {issue.code for issue in report.issues}

    def _store_edge(self, **kwargs):
        import store

        payload = {"account_id": self.account_id, "from_entity": "Sales",
                   "to_entity": "Customer", "from_column": "CUSTOMER_ID",
                   "to_column": "CUSTOMER_ID"}
        payload.update(kwargs)
        return store.save_relationship(**payload)

    def test_an_unusable_join_is_an_error(self):
        self._store_edge(to_entity="Stock", from_column="ITEM_ID",
                         to_column="ITEM_ID")
        report = self._health()
        self.assertIn("RELATIONSHIP_UNUSABLE", self._codes(report))

    def test_it_says_which_edge_and_why_without_repeating_itself(self):
        self._store_edge(to_entity="Stock", from_column="ITEM_ID",
                         to_column="ITEM_ID")
        message = next(i.message for i in self._health().issues
                       if i.code == "RELATIONSHIP_UNUSABLE")
        self.assertIn("Sales", message)
        self.assertIn("Stock", message)
        self.assertIn("fact-to-fact", message)
        self.assertEqual(message.lower().count("cannot be used"), 1)

    def test_a_missing_column_is_an_error(self):
        self._store_edge(to_column="MISSING_COL")
        report = self._health()
        self.assertIn("RELATIONSHIP_COLUMN_MISSING", self._codes(report))
        message = next(i.message for i in report.issues
                       if i.code == "RELATIONSHIP_COLUMN_MISSING")
        self.assertIn("MISSING_COL", message)

    def test_a_broken_validation_is_an_error(self):
        import store

        rel_id = self._store_edge()
        store.update_relationship_validation(self.account_id, rel_id, "broken")
        report = self._health()
        self.assertIn("RELATIONSHIP_VALIDATION_BROKEN", self._codes(report))
        # Severity, not just presence. A measured failure demoted to a warning
        # costs 3 points instead of 10 and sits below the fold of a list
        # sorted by severity.
        severity = next(i.severity for i in report.issues
                        if i.code == "RELATIONSHIP_VALIDATION_BROKEN")
        self.assertEqual(severity, "error")

    def test_a_row_from_before_the_column_existed_is_not_called_broken(self):
        # SQLite hands back None for a legacy row, and the default decides
        # what that means. Defaulting to "broken" would mark every join in
        # every workspace that upgraded into this version.
        import store

        rel_id = self._store_edge()
        with store.get_db() as conn:
            conn.execute("UPDATE entity_relationships SET validation_status=NULL "
                         "WHERE id=?", (rel_id,))
        codes = self._codes(self._health())
        self.assertNotIn("RELATIONSHIP_VALIDATION_BROKEN", codes)
        self.assertNotIn("RELATIONSHIP_VALIDATION_WARNING", codes)

    def test_a_validation_warning_is_a_warning(self):
        import store

        rel_id = self._store_edge()
        store.update_relationship_validation(self.account_id, rel_id, "warning")
        report = self._health()
        self.assertIn("RELATIONSHIP_VALIDATION_WARNING", self._codes(report))
        severity = next(i.severity for i in report.issues
                        if i.code == "RELATIONSHIP_VALIDATION_WARNING")
        self.assertEqual(severity, "warning")

    def test_an_unprobed_join_is_not_reported_as_broken(self):
        # Nobody has run the live probe, which is the only thing that sets
        # these. Unmeasured is a different thing to tell somebody than broken,
        # and a different thing to fix.
        self._store_edge()
        codes = self._codes(self._health())
        self.assertNotIn("RELATIONSHIP_VALIDATION_BROKEN", codes)
        self.assertNotIn("RELATIONSHIP_VALIDATION_WARNING", codes)

    def test_a_healthy_join_raises_nothing(self):
        self._store_edge()
        codes = self._codes(self._health())
        self.assertFalse({c for c in codes if c.startswith("RELATIONSHIP_")},
                         codes)

    def test_the_score_actually_moves(self):
        clean = self._health().score
        self._store_edge(to_entity="Stock", from_column="ITEM_ID",
                         to_column="ITEM_ID")
        self.assertLess(self._health().score, clean)

    def test_an_orphaned_edge_is_reported_once_not_twice(self):
        # Check 4 already names an edge whose entity is gone; running the
        # join check on it as well would report the same row under two codes
        # and cost the score twice.
        self._store_edge(to_entity="Ghost")
        codes = [i.code for i in self._health().issues]
        self.assertIn("ORPHANED_RELATIONSHIP", codes)
        self.assertNotIn("RELATIONSHIP_UNUSABLE", codes)

    def test_an_inactive_edge_is_not_reported(self):
        import store

        rel_id = self._store_edge(to_entity="Stock", from_column="ITEM_ID",
                                  to_column="ITEM_ID")
        with store.get_db() as conn:
            conn.execute("UPDATE entity_relationships SET is_active=0 WHERE id=?",
                         (rel_id,))
        self.assertNotIn("RELATIONSHIP_UNUSABLE", self._codes(self._health()))

    def test_one_bad_row_does_not_kill_the_whole_report(self):
        # A health report that dies on one row tells an admin nothing about
        # the other two hundred.
        import core.graph_health as health

        self._store_edge()
        with patch.object(health, "check_join",
                          side_effect=RuntimeError("boom")):
            report = self._health()
        self.assertTrue(report.issues)
        self.assertGreater(report.score, 0)

    def test_health_and_the_save_path_agree(self):
        # One vocabulary: a workspace cannot be told an edge is fine by one
        # and impossible by the other.
        import json as _json

        response = self._save(to_entity="Stock", from_column="ITEM_ID",
                              to_column="ITEM_ID")
        self.assertEqual(response.status_code, 422)
        refusal = _json.loads(response.body)["message"]

        self._store_edge(to_entity="Stock", from_column="ITEM_ID",
                         to_column="ITEM_ID")
        message = next(i.message for i in self._health().issues
                       if i.code == "RELATIONSHIP_UNUSABLE")
        self.assertIn(refusal, message)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
