"""
tests/test_measure_fact_and_timeout_handling.py

Two production defects from the same family: a heuristic overriding governed
evidence, and a failure path that made things worse instead of explaining them.

1. Business-source arbitration outranked the compiled plan's required measure.

   Live: "what is the total amount of confirmed purchase orders by profit
   center". Arbitration scored CUS_ORD_IVC_FCT (customer invoices) 33 on an
   approved-metric binding, while the compiled plan required
   PCH_ORD_RCT_FCT.PCH_ORD_LIN_CAD_AMT. The plan then declared CUS_ORD_IVC_FCT
   the measure fact, the model generated CORRECT SQL over PCH_ORD_RCT_FCT, and
   source_fact_mismatch rejected the right answer.

   A source the user NAMED or CONFIRMED still wins — that is a governed
   decision. An inferred one does not.

2. A statement timeout was treated as a repairable SQL defect.

   Live: a governed revenue query timed out at 120 s, the LLM retry ran another
   120 s and then failed validation on the entity-graph join plan. Four minutes
   to be told the wrong thing. A timeout means the SQL was valid and the
   database was slow; rewriting it cannot make the database faster.
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

_tmpdir = tempfile.mkdtemp(prefix="querybot_measure_fact_")
os.environ.setdefault("DB_PATH", str(Path(_tmpdir) / "test_querybot.db"))
os.environ.setdefault("QUERYBOT_KEY_FILE", str(Path(_tmpdir) / "test_key"))

from core.analytical_request_plan import compile_analytical_request_plan  # noqa: E402
from core.failure_messages import (  # noqa: E402
    build_query_timeout_guidance,
    is_query_timeout,
)
from core.semantic_model import _scope_plan_to_single_fact  # noqa: E402

PURCHASE_FACT = "EMDW_DMART.PCH_ORD_RCT_FCT"
INVOICE_FACT = "EMDW_DMART.CUS_ORD_IVC_FCT"
INVENTORY_FACT = "EMDW_DMART.ITM_BAL_PRD_FCT"
PROFIT_CENTRE = "EMDW_DMART.PFT_CTR_DMS"

TABLES = [
    {"qualified_name": PURCHASE_FACT, "type": "fact"},
    {"qualified_name": INVOICE_FACT, "type": "fact"},
    {"qualified_name": INVENTORY_FACT, "type": "fact"},
    {"qualified_name": PROFIT_CENTRE, "type": "dimension"},
]


def _emco_fields():
    """The merged plan exactly as production logged it."""
    return [
        # The LLM field planner's guess: "purchase" -> an inventory quantity.
        {"term": "purchase", "table": f"EMCODW_DEV.{INVENTORY_FACT}",
         "column": "PCH_QTY", "role": "measure", "enforcement": None},
        # The structured semantic model's governed binding.
        {"term": "purchase order amount", "table": PURCHASE_FACT,
         "column": "PCH_ORD_LIN_CAD_AMT", "role": "measure",
         "enforcement": "required"},
        {"term": "Profit Center", "table": PROFIT_CENTRE,
         "column": "PFT_CTR_NM", "role": "display_dimension",
         "enforcement": "required"},
    ]


def _emco_joins():
    return [{
        "from": PURCHASE_FACT, "to": PROFIT_CENTRE,
        "conditions": [["PFT_CTR_KEY", "PFT_CTR_KEY"]],
        "enforcement": "required",
    }]


# ══════════════════════════════════════════════════════════════════════════════
# 1  The measure decides the measure fact
# ══════════════════════════════════════════════════════════════════════════════
class TestFactAnchorFollowsTheMeasure(unittest.TestCase):

    def test_inferred_source_does_not_override_the_required_measure(self):
        fields, joins = _emco_fields(), _emco_joins()
        anchor = _scope_plan_to_single_fact(
            fields, joins, TABLES,
            preferred_fact_tables={INVOICE_FACT},
            preferred_source_reason="approved metric source binding",
        )
        self.assertEqual(anchor, PURCHASE_FACT)
        # The governed measure stays enforceable...
        governed = next(f for f in fields if f["column"] == "PCH_ORD_LIN_CAD_AMT")
        self.assertNotEqual(governed["enforcement"], "optional")
        # ...the weak guess is demoted...
        guess = next(f for f in fields if f["column"] == "PCH_QTY")
        self.assertEqual(guess["enforcement"], "optional")
        # ...and the join the answer needs is still required.
        self.assertEqual(joins[0]["enforcement"], "required")

    def test_a_governed_binding_outranks_an_unset_one(self):
        """enforcement="required" is the semantic model; unset is a suggestion."""
        fields, joins = _emco_fields(), _emco_joins()
        anchor = _scope_plan_to_single_fact(fields, joins, TABLES)
        self.assertEqual(anchor, PURCHASE_FACT)

    def test_an_explicitly_named_source_still_wins(self):
        """The user naming the source is a governed decision, not a guess."""
        fields, joins = _emco_fields(), _emco_joins()
        anchor = _scope_plan_to_single_fact(
            fields, joins, TABLES,
            preferred_fact_tables={INVOICE_FACT},
            preferred_source_reason="explicit tenant source terminology",
        )
        self.assertEqual(anchor, INVOICE_FACT)

    def test_a_user_confirmed_source_still_wins(self):
        fields, joins = _emco_fields(), _emco_joins()
        anchor = _scope_plan_to_single_fact(
            fields, joins, TABLES,
            preferred_fact_tables={INVOICE_FACT},
            preferred_source_reason="user-confirmed governed source",
        )
        self.assertEqual(anchor, INVOICE_FACT)

    def test_a_corroborated_source_wins_however_it_was_chosen(self):
        """When the preferred fact carries the measure there is no conflict."""
        fields, joins = _emco_fields(), _emco_joins()
        anchor = _scope_plan_to_single_fact(
            fields, joins, TABLES,
            preferred_fact_tables={PURCHASE_FACT},
            preferred_source_reason="tenant semantic model and terminology evidence",
        )
        self.assertEqual(anchor, PURCHASE_FACT)


# ══════════════════════════════════════════════════════════════════════════════
# 2  The request plan the validator enforces must agree
# ══════════════════════════════════════════════════════════════════════════════
class TestRequestPlanMeasureFact(unittest.TestCase):
    """This plan is what source_fact_mismatch checks the SQL against."""

    def _plan(self, *, reason, fact_anchor=""):
        semantic = {
            "enabled": True,
            "fields": _emco_fields(),
            "joins": _emco_joins(),
            "fact_anchor": fact_anchor,
            "source_scope": {
                "status": "selected",
                "selected_fact": INVOICE_FACT,
                "reason": reason,
            },
        }
        return compile_analytical_request_plan(
            "What is the total amount of confirmed purchase orders by profit center?",
            semantic,
        )

    def test_inferred_source_is_overridden_by_the_measure(self):
        plan = self._plan(reason="approved metric source binding")
        self.assertEqual(plan["source_fact"], PURCHASE_FACT)
        self.assertEqual(plan["source_facts"], [PURCHASE_FACT])

    def test_the_governed_fact_anchor_is_honoured(self):
        plan = self._plan(
            reason="approved metric source binding", fact_anchor=PURCHASE_FACT,
        )
        self.assertEqual(plan["source_fact"], PURCHASE_FACT)

    def test_a_user_confirmed_source_is_never_overridden(self):
        plan = self._plan(reason="user-confirmed governed source")
        self.assertEqual(plan["source_fact"], INVOICE_FACT)

    def test_an_agreeing_source_is_left_alone(self):
        semantic = {
            "enabled": True,
            "fields": _emco_fields(),
            "joins": _emco_joins(),
            "source_scope": {
                "status": "selected", "selected_fact": PURCHASE_FACT,
                "reason": "approved metric source binding",
            },
        }
        plan = compile_analytical_request_plan("purchase order amount", semantic)
        self.assertEqual(plan["source_fact"], PURCHASE_FACT)

    def test_the_dimension_survives_as_a_dimension(self):
        """The profit-centre grouping must not be lost in the re-anchoring."""
        plan = self._plan(reason="approved metric source binding")
        dimension_columns = {
            str(item.get("column") or "") for item in plan.get("dimensions") or []
        }
        self.assertIn("PFT_CTR_NM", dimension_columns)


# ══════════════════════════════════════════════════════════════════════════════
# 3  A timeout is not a repairable defect
# ══════════════════════════════════════════════════════════════════════════════
class TestQueryTimeoutHandling(unittest.TestCase):

    def test_statement_timeouts_are_recognized(self):
        for text in (
            "('HYT00', '[HYT00] [Microsoft][ODBC Driver 18 for SQL Server]"
            "Query timeout expired (0) (SQLExecDirectW)')",
            "Query timeout expired",
            "The request timed out",
            "query timeout",
        ):
            with self.subTest(text=text[:40]):
                self.assertTrue(is_query_timeout(text))

    def test_other_failures_are_not_timeouts(self):
        for text in (
            "[HYT00] Login timeout expired",   # connectivity — worth one retry
            "Invalid column name 'FOO'",
            "communication link failure",
            "deadlock victim",
            "divide by zero",
            "", None,
        ):
            with self.subTest(text=str(text)[:40]):
                self.assertFalse(is_query_timeout(text))

    def test_guidance_names_the_column_that_needs_an_index(self):
        guidance = build_query_timeout_guidance(
            {"temporal_policies": [{
                "fact_table": INVOICE_FACT, "fact_column": "CUS_IVC_DT_DMS_KEY",
            }]},
            timeout_seconds=120,
        )
        self.assertIn("CUS_IVC_DT_DMS_KEY", guidance["reason"])
        self.assertIn(INVOICE_FACT, guidance["next_step"])
        self.assertIn("index", guidance["next_step"].lower())
        self.assertIn("120", guidance["reason"])

    def test_guidance_degrades_gracefully_without_a_plan(self):
        for plan in ({}, None, {"temporal_policies": []},
                     {"temporal_policies": [{"fact_table": ""}]}):
            with self.subTest(plan=plan):
                guidance = build_query_timeout_guidance(plan, timeout_seconds=120)
                self.assertTrue(guidance["reason"].strip())
                self.assertTrue(guidance["next_step"].strip())

    def test_the_pipeline_does_not_repair_a_timeout(self):
        source = (ROOT / "core" / "query_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("is_query_timeout(exec_error)", source)
        block = source.split("_timed_out = exec_error is not None", 1)[1].split(
            "if retryable:", 1,
        )[0]
        self.assertIn("retryable = False", block)
        self.assertIn("execution_timeout_no_repair", block)

    def test_the_pipeline_surfaces_actionable_timeout_guidance(self):
        source = (ROOT / "core" / "query_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("build_query_timeout_guidance(", source)
        self.assertIn('_rca["most_likely_reason"] = _timeout_guidance["reason"]', source)


if __name__ == "__main__":
    unittest.main()
