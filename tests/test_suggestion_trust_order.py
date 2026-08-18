"""
tests/test_suggestion_trust_order.py

A suggestion is a promise. The user clicks a button this module supplied, so
offering one that cannot be answered is worse than offering nothing.

Two defects made the portal do exactly that:

1. get_suggestions ran its LEAST trustworthy source first. The module docstring
   said validated examples, then metrics, then "Stage 2 cache as metadata only,
   not as raw user-facing prompts" — but the code ran Stage 2 first. Those are
   questions the LLM wrote while generating the KB; they have never executed.

2. Nothing checked the entity graph. A duplicate entity minted for a table that
   already had one ("Customer" beside "Cus Dms") carries no relationships, and
   any question naming it is refused with "Missing governed path". The same
   pairing appears for Warehouse, Region and every other dimension a Suggest run
   duplicated, so whole families of offered questions dead-ended.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_tmpdir = tempfile.mkdtemp(prefix="querybot_suggestions_")
os.environ.setdefault("DB_PATH", str(Path(_tmpdir) / "test_querybot.db"))
os.environ.setdefault("QUERYBOT_KEY_FILE", str(Path(_tmpdir) / "test_key"))

import core.suggestions as suggestions  # noqa: E402

SCHEMA = "EMDW_DMART"


def _entity(name, table, entity_type="dimension"):
    return {
        "entity_name": name, "table_name": table, "schema_name": SCHEMA,
        "entity_type": entity_type, "status": "confirmed",
    }


def _graph_with_twins():
    """The live shape: connected entity + jointless duplicate on one table."""
    return {
        "entities": [
            _entity("CUS_ORD_IVC_FCT", "CUS_ORD_IVC_FCT", "fact"),
            _entity("Cus Dms", "CUS_DMS"),
            _entity("Customer", "CUS_DMS"),      # the duplicate, no edges
            _entity("Whs Dms", "WHS_DMS"),
            _entity("Warehouse", "WHS_DMS"),     # same defect, another table
        ],
        "relationships": [
            {"id": 1, "from_entity": "CUS_ORD_IVC_FCT", "to_entity": "Cus Dms",
             "from_column": "CUS_DMS_KEY", "to_column": "CUS_DMS_KEY",
             "status": "confirmed"},
            {"id": 2, "from_entity": "CUS_ORD_IVC_FCT", "to_entity": "Whs Dms",
             "from_column": "WHS_DMS_KEY", "to_column": "WHS_DMS_KEY",
             "status": "confirmed"},
        ],
        "properties": [],
    }


class TestJointlessTwinsAreNotOffered(unittest.TestCase):

    def _check(self, graph):
        with patch("store.get_full_graph", return_value=graph):
            return suggestions._graph_reachability_check("acct")

    def test_a_question_resolving_to_a_jointless_twin_is_withheld(self):
        check = self._check(_graph_with_twins())
        self.assertFalse(check("What is my revenue by each customer, top 5?"))

    def test_the_defect_is_not_customer_specific(self):
        """Every dimension a Suggest run duplicated has the same problem."""
        check = self._check(_graph_with_twins())
        self.assertFalse(check("Show total revenue by warehouse"))

    def test_questions_that_avoid_the_twins_are_still_offered(self):
        check = self._check(_graph_with_twins())
        self.assertTrue(check("What is my total revenue for the last 2 days?"))

    def test_a_healthy_graph_withholds_nothing(self):
        graph = _graph_with_twins()
        graph["entities"] = [
            entity for entity in graph["entities"]
            if entity["entity_name"] not in {"Customer", "Warehouse"}
        ]
        check = self._check(graph)
        for question in ("revenue by customer", "revenue by warehouse", "revenue"):
            with self.subTest(question=question):
                self.assertTrue(check(question))

    def test_a_lone_disconnected_entity_is_not_treated_as_a_twin(self):
        """Only a duplicate BESIDE a connected sibling is the known defect."""
        graph = _graph_with_twins()
        graph["entities"] = [
            entity for entity in graph["entities"]
            if entity["entity_name"] != "Cus Dms"
        ]
        check = self._check(graph)
        self.assertTrue(check("revenue by customer"))

    def test_it_fails_open(self):
        """This may only ever remove questions proven to dead-end."""
        for graph in ({}, {"entities": [], "relationships": []}, None):
            with self.subTest(graph=graph):
                with patch("store.get_full_graph", return_value=graph):
                    self.assertTrue(
                        suggestions._graph_reachability_check("acct")("anything"),
                    )

    def test_it_fails_open_when_the_graph_cannot_be_read(self):
        with patch("store.get_full_graph", side_effect=RuntimeError("db down")):
            self.assertTrue(
                suggestions._graph_reachability_check("acct")("anything"),
            )


class TestTheGateActuallyRuns(unittest.TestCase):
    """The gate above was correct and had never executed.

    95ace2d added the reordered tiers and the reachability check BELOW a
    `return suggestions` that already existed, leaving 80 lines — including the
    only call to _graph_reachability_check — statically unreachable. Every
    assertion that was supposed to protect this searched the SOURCE TEXT of
    get_suggestions for tier comments, so it matched the dead block and passed
    while the feature was inert. These tests go through get_suggestions.
    """

    QUESTION = "What is my revenue by each customer, top 5?"

    def _run(self, *, validated=(), cached=(), metrics=(), n=6, graph=None):
        with tempfile.TemporaryDirectory() as kb_dir:
            if cached:
                (Path(kb_dir) / suggestions._CACHE_FILENAME).write_text(
                    __import__("json").dumps(list(cached)), encoding="utf-8")
            with patch("store.get_full_graph",
                       return_value=_graph_with_twins() if graph is None else graph),                     patch("store.get_validated_examples", return_value=list(validated)),                     patch("store.list_metrics", return_value=list(metrics)):
                return suggestions.get_suggestions("acct", kb_dir, None, n=n)

    def test_get_suggestions_withholds_a_question_the_graph_cannot_reach(self):
        offered = self._run(validated=[
            {"question": self.QUESTION, "table_name": f"{SCHEMA}.CUS_DMS",
             "sql_query": f"SELECT 1 FROM {SCHEMA}.CUS_DMS"},
        ])
        self.assertNotIn(
            self.QUESTION, [s["question"] for s in offered],
            "a question that dead-ends on 'Missing governed path' was offered",
        )

    def test_the_same_question_is_offered_once_the_graph_is_healthy(self):
        healthy = _graph_with_twins()
        healthy["entities"] = [
            e for e in healthy["entities"]
            if e["entity_name"] not in {"Customer", "Warehouse"}
        ]
        with tempfile.TemporaryDirectory() as kb_dir:
            with patch("store.get_full_graph", return_value=healthy),                     patch("store.get_validated_examples", return_value=[
                        {"question": self.QUESTION, "table_name": f"{SCHEMA}.CUS_DMS",
                         "sql_query": f"SELECT 1 FROM {SCHEMA}.CUS_DMS"}]),                     patch("store.list_metrics", return_value=[]):
                offered = suggestions.get_suggestions("acct", kb_dir, None)
        self.assertIn(self.QUESTION, [s["question"] for s in offered])

    def test_the_gate_is_reached_on_every_tier_that_offers_authored_text(self):
        calls = []
        real = suggestions._graph_reachability_check

        def counting(account_id):
            calls.append(account_id)
            return real(account_id)

        with patch.object(suggestions, "_graph_reachability_check", counting):
            self._run(
                validated=[{"question": "revenue by customer",
                            "table_name": f"{SCHEMA}.CUS_DMS", "sql_query": ""}],
                cached=[{"question": "revenue by warehouse",
                         "table": "WHS_DMS", "fqn": f"{SCHEMA}.WHS_DMS"}],
            )
        self.assertTrue(calls, "the reachability gate was never invoked")

    def test_the_stage_two_cache_still_reaches_the_user_when_it_is_safe(self):
        # Tier 3 was also mis-scoped: it compared a bare table name against a
        # set of fully-qualified names, so it emitted nothing even when live.
        healthy = _graph_with_twins()
        healthy["entities"] = [
            e for e in healthy["entities"]
            if e["entity_name"] not in {"Customer", "Warehouse"}
        ]
        offered = self._run(graph=healthy, cached=[
            {"question": "How many warehouses are there?",
             "table": "WHS_DMS", "fqn": f"{SCHEMA}.WHS_DMS"},
        ])
        self.assertIn("How many warehouses are there?",
                      [s["question"] for s in offered])

    def test_validated_examples_outrank_the_invented_cache(self):
        offered = self._run(
            validated=[{"question": "proven question", "table_name": f"{SCHEMA}.WHS_DMS",
                        "sql_query": ""}],
            cached=[{"question": "invented question", "table": "WHS_DMS",
                     "fqn": f"{SCHEMA}.WHS_DMS"}],
            n=1,
        )
        self.assertEqual([s["question"] for s in offered], ["proven question"])


class TestFailOpenFiltersButDoesNotPromote(unittest.TestCase):
    """Failing open is right when filtering a trusted source and wrong when
    promoting an untrusted one, so the predicate reports which it did."""

    def test_a_readable_graph_vouches_for_what_it_checked(self):
        with patch("store.get_full_graph", return_value=_graph_with_twins()):
            self.assertTrue(suggestions._graph_reachability_check("acct").verified)

    def test_an_unusable_graph_vouches_for_nothing(self):
        for graph in ({}, {"entities": [], "relationships": []}, None):
            with self.subTest(graph=graph):
                with patch("store.get_full_graph", return_value=graph):
                    check = suggestions._graph_reachability_check("acct")
                self.assertTrue(check("anything"), "must not withhold blindly")
                self.assertFalse(check.verified, "must not vouch blindly")

    def test_the_invented_cache_is_withheld_when_nothing_vouches_for_it(self):
        with tempfile.TemporaryDirectory() as kb_dir:
            (Path(kb_dir) / suggestions._CACHE_FILENAME).write_text(
                __import__("json").dumps(
                    [{"question": "invented question", "table": "WHS_DMS",
                      "fqn": f"{SCHEMA}.WHS_DMS"}]), encoding="utf-8")
            with patch("store.get_full_graph", return_value={"entities": []}),                     patch("store.get_validated_examples", return_value=[]),                     patch("store.list_metrics", return_value=[]):
                offered = suggestions.get_suggestions("acct", kb_dir, None)
        self.assertEqual(offered, [])


class TestNoUnreachableCode(unittest.TestCase):
    """A structural guard. The defect above was invisible to every behavioural
    test because the code simply never ran; this catches the shape directly."""

    def test_get_suggestions_has_no_statements_after_its_return(self):
        import ast
        tree = ast.parse((ROOT / "core" / "suggestions.py").read_text(encoding="utf-8"))
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "get_suggestions")
        seen_return = False
        dead = []
        for stmt in fn.body:
            if seen_return:
                dead.append(f"line {stmt.lineno}: {type(stmt).__name__}")
            if isinstance(stmt, ast.Return):
                seen_return = True
        self.assertEqual(dead, [], "unreachable statements after the return")


if __name__ == "__main__":
    unittest.main()
