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


class TestTrustOrder(unittest.TestCase):
    """The proven sources must be consulted before the invented ones."""

    def test_validated_examples_run_before_the_generated_cache(self):
        source = (ROOT / "core" / "suggestions.py").read_text(encoding="utf-8")
        body = source.split("def get_suggestions(", 1)[1]
        validated = body.find("Tier 1: Validated examples")
        metrics = body.find("Tier 2: Metric registry")
        generated = body.find("Tier 3: Stage 2 query patterns")
        self.assertGreater(validated, 0)
        self.assertLess(validated, metrics, "validated examples must come first")
        self.assertLess(
            metrics, generated,
            "questions the LLM invented must be the last resort, not the first",
        )

    def test_the_generated_tier_is_graph_filtered(self):
        source = (ROOT / "core" / "suggestions.py").read_text(encoding="utf-8")
        generated = source.split("Tier 3: Stage 2 query patterns", 1)[1]
        self.assertIn("_graph_reachability_check(account_id)", generated)
        self.assertIn("if not _reachable(", generated)


if __name__ == "__main__":
    unittest.main()
