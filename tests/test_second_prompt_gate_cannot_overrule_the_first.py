"""
tests/test_second_prompt_gate_cannot_overrule_the_first.py

Catalogue check C4 — "show customers with no invoices" must be an anti-join and
must NOT silently become an inner join.

Two gates decide which advanced SQL rules reach the model, and they had
separate phrase lists for the same concepts:

  gate 1  core.sql_prompt_rules.GATED_RULES  — decides what to EMIT
  gate 2  core.llm._compiled_sql_rule_features — decides what to STRIP after

Gate 1's _ANTI_JOIN_WORDS carries "with no\b" and correctly emitted the
ANTI-JOIN RULE for C4's exact wording. Gate 2's private regex listed "have no"
but not "with no", so it deleted the block gate 1 had just decided the question
needed. The model then had no instruction to write NOT EXISTS / LEFT JOIN ...
IS NULL, and an inner join returns precisely the customers the user did not ask
for — a silently inverted answer, not an error.

A second pass may narrow where the first expressed no opinion. It must never
overrule an include.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_tmpdir = tempfile.mkdtemp(prefix="querybot_gates_")
os.environ.setdefault("DB_PATH", str(Path(_tmpdir) / "test_querybot.db"))
os.environ.setdefault("QUERYBOT_KEY_FILE", str(Path(_tmpdir) / "test_key"))

import core.sql_prompt_rules as gate1  # noqa: E402
from core.llm import (  # noqa: E402
    _GATE1_RULE_FOR_FEATURE,
    _OPTIONAL_SQL_RULE_MARKERS,
    _filter_sql_rules_for_compiled_plan,
)

# A compiled plan with no signal of its own, so gate 2's own feature detection
# contributes nothing and the first gate's decision is the only thing left.
REQUEST_PLAN = {
    "status": "compiled", "dimensions": [], "measures": [], "metrics": [],
    "joins": [], "source_facts": ["EMDW_DMART.CUS_ORD_IVC_FCT"],
    "temporal_operations": [], "filters": [], "question": "",
}
PLAN = {"analytical_request_plan": REQUEST_PLAN}

ANTI_JOIN_MARKER = _OPTIONAL_SQL_RULE_MARKERS["anti_join"]


def _prompt_with_all_blocks() -> str:
    parts = ["You are a SQL generator.\n\n"]
    for marker in _OPTIONAL_SQL_RULE_MARKERS.values():
        parts.append(f"{marker} body text for this rule.\n\n")
    parts.append("Knowledge Base — available tables and their business context:\nx")
    return "".join(parts)


def _ctx(question: str):
    return gate1.RuleContext(
        question=question, table_context="", semantic_plan=PLAN, graph_context={},
    )


class TestGateTwoNeverStripsWhatGateOneIncluded(unittest.TestCase):

    C4 = "show customers with no invoices"

    def test_gate_one_really_does_want_the_rule_for_this_wording(self):
        self.assertTrue(gate1.rule_applies("anti_join", _ctx(self.C4)))

    def test_the_anti_join_rule_survives_the_second_pass(self):
        kept = _filter_sql_rules_for_compiled_plan(
            _prompt_with_all_blocks(), PLAN, _ctx(self.C4),
        )
        self.assertIn(
            ANTI_JOIN_MARKER, kept,
            "the second gate deleted the anti-join rule the first gate emitted, "
            "so nothing tells the model not to write an inner join",
        )

    def test_every_drifted_phrasing_survives(self):
        for question in (
            "show customers with no invoices",
            "show customers with no orders",
            "suppliers lacking receipts",
            "orders not yet shipped",
            "customers who haven't ordered",
            "items with no sales",
        ):
            with self.subTest(question=question):
                kept = _filter_sql_rules_for_compiled_plan(
                    _prompt_with_all_blocks(), PLAN, _ctx(question),
                )
                self.assertIn(ANTI_JOIN_MARKER, kept)

    def test_the_invariant_holds_across_every_mapped_rule(self):
        probes = {
            "anti_join": "customers with no invoices",
            "correlation": "plot revenue against margin",
            "benchmark": "warehouses that outperform",
            "moving_average": "rolling sum of revenue",
            "event_interval": "how long between order and receipt",
        }
        for feature, question in probes.items():
            rule_id = _GATE1_RULE_FOR_FEATURE[feature]
            with self.subTest(feature=feature):
                if not gate1.rule_applies(rule_id, _ctx(question)):
                    self.skipTest(f"gate 1 does not want {rule_id} here")
                kept = _filter_sql_rules_for_compiled_plan(
                    _prompt_with_all_blocks(), PLAN, _ctx(question),
                )
                self.assertIn(
                    _OPTIONAL_SQL_RULE_MARKERS[feature], kept,
                    f"second gate overruled the first for {rule_id}",
                )

    def test_it_still_strips_rules_neither_gate_wants(self):
        # The filter must keep doing its job — an unrelated question should not
        # drag the anti-join rule along.
        kept = _filter_sql_rules_for_compiled_plan(
            _prompt_with_all_blocks(), PLAN, _ctx("what is my total revenue"),
        )
        self.assertNotIn(ANTI_JOIN_MARKER, kept)

    def test_every_mapped_feature_names_a_real_rule_on_both_sides(self):
        for feature, rule_id in _GATE1_RULE_FOR_FEATURE.items():
            with self.subTest(feature=feature):
                self.assertIn(feature, _OPTIONAL_SQL_RULE_MARKERS)
                self.assertIn(rule_id, gate1.GATED_RULES)

    def test_omitting_the_context_keeps_the_old_behaviour(self):
        # Callers that pass no context must not crash or silently keep all rules.
        kept = _filter_sql_rules_for_compiled_plan(_prompt_with_all_blocks(), PLAN)
        self.assertNotIn(ANTI_JOIN_MARKER, kept)


if __name__ == "__main__":
    unittest.main(verbosity=2)
