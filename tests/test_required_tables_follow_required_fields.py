"""
tests/test_required_tables_follow_required_fields.py

One defect, three symptoms, all observed on the EMCO-shaped mart.

required_semantic_tables promotes BOTH endpoints of every non-optional join
into required_tables. build_runtime_semantic_plan emitted a join for every
dimension that merely SCORED, so a dimension nobody asked to see dragged its
table into the plan:

  B1  "total amount of confirmed purchase orders by profit center"
      pulled CUS_DMS in through the many-to-many bridge PFT_CTR_CUS_DAT.
      Every purchase-order row was counted once per customer in that profit
      centre; the total came out ~7.5x too large. Valid SQL, real declared
      foreign keys, correct grouping label, no validator error.

  C4  "show customers with no invoices" required CUS_SEG_DMS. The anti-join
      correctly did not join customer segment, so the answer was refused:
      graph_plan_mismatch.

  B2  "revenue by customer, top 5" required CUS_SEG_DMS and CUS_TYP_DMS.
      Harmless (both many-to-one) but the same mechanism.

An earlier attempt gated the join on `claims_display` at emission time. That
flag is only ever reconsidered for date dimensions inside a time window, so it
was always true for an ordinary dimension and the fix was inert.

Emission time is also the wrong place: requirement is TRANSITIVE. "revenue by
customer type" needs fact -> CUS_DMS -> CUS_TYP_DMS, and the first hop hosts no
required field of its own. The decision has to be made once every field and
join is known, by walking out from the source tables of the required fields.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_tmpdir = tempfile.mkdtemp(prefix="querybot_reqtables_")
os.environ.setdefault("DB_PATH", str(Path(_tmpdir) / "test_querybot.db"))
os.environ.setdefault("QUERYBOT_KEY_FILE", str(Path(_tmpdir) / "test_key"))

from core.semantic_model import (  # noqa: E402
    _demote_joins_not_needed_to_reach_a_required_field as demote,
    build_runtime_semantic_plan,
)
from core.semantic_plan_utils import required_semantic_tables  # noqa: E402

FCT = "EMDW_DMART.CUS_ORD_IVC_FCT"
PCH = "EMDW_DMART.PCH_ORD_RCT_FCT"
CUS = "EMDW_DMART.CUS_DMS"
SEG = "EMDW_DMART.CUS_SEG_DMS"
TYP = "EMDW_DMART.CUS_TYP_DMS"
PFT = "EMDW_DMART.PFT_CTR_DMS"
BRIDGE = "EMDW_DMART.PFT_CTR_CUS_DAT"


def _field(table, column, source_table, enforcement="required"):
    return {"table": table, "column": column, "source_table": source_table,
            "enforcement": enforcement, "role": "display_dimension",
            "display_required": enforcement != "optional"}


def _join(a, b):
    return {"from": a, "to": b, "enforcement": "required"}


def _enforcement(joins):
    return {(j["from"], j["to"]): j["enforcement"] for j in joins}


class TestOnlyJoinsThatReachTheAnswerAreRequired(unittest.TestCase):

    def test_c4_an_unasked_dimension_no_longer_blocks_the_anti_join(self):
        fields = [_field(CUS, "CUS_NM", FCT)]
        joins = [_join(FCT, CUS), _join(CUS, SEG), _join(CUS, TYP)]
        demote(fields, joins)
        got = _enforcement(joins)
        self.assertEqual(got[(FCT, CUS)], "required")
        self.assertEqual(got[(CUS, SEG)], "optional")
        self.assertEqual(got[(CUS, TYP)], "optional")
        required = required_semantic_tables({"fields": fields, "joins": joins})
        self.assertNotIn(SEG, required)
        self.assertNotIn(TYP, required)

    def test_b1_the_fan_out_bridge_is_not_required_by_a_profit_centre_question(self):
        fields = [_field(PFT, "PFT_CTR_NM", PCH)]
        joins = [_join(PCH, PFT), _join(BRIDGE, PFT), _join(BRIDGE, CUS)]
        demote(fields, joins)
        required = required_semantic_tables({"fields": fields, "joins": joins})
        self.assertIn(PFT, required)
        self.assertNotIn(CUS, required, "the many-to-many bridge reached customers again")
        self.assertNotIn(BRIDGE, required)

    def test_a_transitive_chain_survives(self):
        # "revenue by customer type": the first hop hosts no field of its own
        # but is the only way to reach the one that does.
        fields = [_field(TYP, "CUS_TYP_NM", CUS)]
        joins = [_join(FCT, CUS), _join(CUS, TYP), _join(CUS, SEG)]
        demote(fields, joins)
        got = _enforcement(joins)
        self.assertEqual(got[(CUS, TYP)], "required")
        self.assertEqual(got[(CUS, SEG)], "optional")
        required = required_semantic_tables({"fields": fields, "joins": joins})
        self.assertIn(TYP, required)
        self.assertIn(CUS, required, "the intermediate hop was dropped, breaking the chain")

    def test_two_required_fields_keep_both_branches(self):
        fields = [_field(TYP, "CUS_TYP_NM", CUS), _field(SEG, "CUS_SEG_NM", CUS)]
        joins = [_join(FCT, CUS), _join(CUS, TYP), _join(CUS, SEG)]
        demote(fields, joins)
        for edge, enforcement in _enforcement(joins).items():
            with self.subTest(edge=edge):
                self.assertEqual(enforcement, "required")

    def test_a_field_on_the_fact_itself_needs_no_joins(self):
        fields = [_field(FCT, "SOP_CUS_IVC_LIN_AMT", FCT)]
        joins = [_join(FCT, CUS), _join(CUS, SEG)]
        demote(fields, joins)
        required = required_semantic_tables({"fields": fields, "joins": joins})
        self.assertEqual(required, {FCT})


class TestItNeverMakesThePlanWorse(unittest.TestCase):

    def test_an_already_optional_join_is_left_alone(self):
        fields = [_field(CUS, "CUS_NM", FCT)]
        joins = [_join(FCT, CUS), {"from": CUS, "to": SEG, "enforcement": "optional"}]
        self.assertEqual(demote(fields, joins), [])

    def test_no_required_fields_changes_nothing(self):
        fields = [_field(CUS, "CUS_NM", FCT, enforcement="optional")]
        joins = [_join(FCT, CUS)]
        self.assertEqual(demote(fields, joins), [])
        self.assertEqual(joins[0]["enforcement"], "required")

    def test_it_works_even_when_fields_carry_no_source_table(self):
        # The fact is recoverable from the join structure alone: it is a join
        # source that is never a join target.
        fields = [{"table": CUS, "column": "CUS_NM", "enforcement": "required"}]
        joins = [_join(FCT, CUS), _join(CUS, SEG)]
        demote(fields, joins)
        got = _enforcement(joins)
        self.assertEqual(got[(FCT, CUS)], "required")
        self.assertEqual(got[(CUS, SEG)], "optional")

    def test_with_no_derivable_root_it_changes_nothing(self):
        # Every table is both a source and a target, so nothing anchors the
        # walk. Guessing would be worse than leaving the plan alone.
        fields = [{"table": CUS, "column": "CUS_NM", "enforcement": "required"}]
        joins = [_join(FCT, CUS), _join(CUS, FCT)]
        self.assertEqual(demote(fields, joins), [])
        self.assertTrue(all(j["enforcement"] == "required" for j in joins))

    def test_an_unreachable_required_field_does_not_crash(self):
        fields = [_field("EMDW_DMART.NOWHERE", "X", FCT)]
        joins = [_join(FCT, CUS)]
        demote(fields, joins)   # must simply return
        self.assertIn(joins[0]["enforcement"], {"required", "optional"})


class TestItRunsInsideTheRealPlanBuilder(unittest.TestCase):
    """The helper being correct proves nothing if the plan never calls it.

    A previous fix in this exact area was gated on a flag that could not be
    false on the path that mattered, so it was inert. Tests that call the
    helper directly cannot see that. These go through the public builder.
    """

    MODEL = {
        "tables": [{
            "qualified_name": FCT,
            "schema": "EMDW_DMART",
            "fields": [],
            "dimensions": [
                {"name": "Customer", "source_key": "CUS_DMS_KEY",
                 "display_table": CUS, "display_column": "CUS_NM",
                 "display_key": "CUS_DMS_KEY", "confidence": 90},
                {"name": "Customer Segment", "source_key": "CUS_SEG_DMS_KEY",
                 "display_table": SEG, "display_column": "CUS_SEG_NM",
                 "display_key": "CUS_SEG_DMS_KEY", "confidence": 90},
            ],
        }],
    }

    def _plan(self, question):
        return build_runtime_semantic_plan(
            "", question=question, selected_schema="EMDW_DMART", model=self.MODEL,
        )

    def test_a_customer_question_does_not_require_customer_segment(self):
        plan = self._plan("show customers with no invoices")
        self.assertIn(CUS, plan["required_tables"])
        self.assertNotIn(
            SEG, plan["required_tables"],
            "an unasked dimension is still forced into the plan, which is what "
            "refused the anti-join with graph_plan_mismatch",
        )

    def test_asking_for_the_segment_still_requires_it(self):
        plan = self._plan("revenue by customer segment")
        self.assertIn(SEG, plan["required_tables"])

    def test_the_builder_actually_invokes_the_closure(self):
        import core.semantic_model as sm
        calls = []
        real = sm._demote_joins_not_needed_to_reach_a_required_field

        def counting(fields, joins):
            calls.append(len(joins))
            return real(fields, joins)

        original = sm._demote_joins_not_needed_to_reach_a_required_field
        sm._demote_joins_not_needed_to_reach_a_required_field = counting
        try:
            self._plan("show customers with no invoices")
        finally:
            sm._demote_joins_not_needed_to_reach_a_required_field = original
        self.assertTrue(calls, "build_runtime_semantic_plan never called the closure")


if __name__ == "__main__":
    unittest.main(verbosity=2)
