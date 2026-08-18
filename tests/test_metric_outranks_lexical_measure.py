"""
tests/test_metric_outranks_lexical_measure.py

Catalogue check B1 — "what is the total amount of confirmed purchase orders by
profit center" must resolve to PCH_ORD_RCT_FCT and answer.

It resolved to the right fact and still failed, because two validators demanded
contradictory columns. Observed on the test mart, trace #760:

  attempt 1  SUM(PCH_ORD_LIN_CAD_AMT), wrong join   -> graph_plan_mismatch
  attempt 2  joins fixed, metric column kept        -> field_plan_mismatch:
             "did not use required semantic field purchase order:
              PCH_ORD_RCT_FCT.PCH_ORD_QTY"
  attempt 3  obeys, SUM(PCH_ORD_QTY)                -> metric_formula_mismatch:
             "Missing required formula column(s): PCH_ORD_LIN_CAD_AMT"

Both errors were individually correct. The PLAN was self-contradictory: the
field plan is built from schema names before metric matching runs, so the term
"purchase order" matched the QUANTITY column and hard-required it, while the
registry required the AMOUNT for the approved metric "Purchase Order Amount".

The registry is admin-approved and the field match is lexical, so the registry
wins. Demote rather than drop — the column stays a hint and only its power to
reject correct SQL is withdrawn.
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

_tmpdir = tempfile.mkdtemp(prefix="querybot_metricfield_")
os.environ.setdefault("DB_PATH", str(Path(_tmpdir) / "test_querybot.db"))
os.environ.setdefault("QUERYBOT_KEY_FILE", str(Path(_tmpdir) / "test_key"))

from core.semantic_planner import (  # noqa: E402
    demote_measures_governed_by_a_metric,
    metric_formula_columns,
)

PCH = "EMDW_DMART.PCH_ORD_RCT_FCT"
IVC = "EMDW_DMART.CUS_ORD_IVC_FCT"

PO_METRIC = {
    "name": "Purchase Order Amount",
    "formula_type": "expression",
    "sql_template": "SUM(PCH_ORD_LIN_CAD_AMT)",
    "base_table": PCH,
    "_resolved_source_tables": [PCH],
}


def _field(table, column, role="measure", enforcement="required"):
    return {"term": "purchase order", "table": table, "column": column,
            "role": role, "enforcement": enforcement}


def _enforcement(fields, column):
    return next(f["enforcement"] for f in fields if f["column"] == column)


class TestTheRegistryOutranksALexicalMatch(unittest.TestCase):

    def test_the_quantity_column_stops_being_hard_required(self):
        fields = [_field(PCH, "PCH_ORD_QTY")]
        demoted = demote_measures_governed_by_a_metric(fields, [PO_METRIC])
        self.assertEqual(demoted, [f"{PCH}.PCH_ORD_QTY"])
        self.assertEqual(_enforcement(fields, "PCH_ORD_QTY"), "optional")

    def test_the_metrics_own_column_is_never_demoted(self):
        fields = [_field(PCH, "PCH_ORD_LIN_CAD_AMT")]
        demote_measures_governed_by_a_metric(fields, [PO_METRIC])
        self.assertEqual(_enforcement(fields, "PCH_ORD_LIN_CAD_AMT"), "required")

    def test_a_measure_on_a_different_fact_is_untouched(self):
        fields = [_field(IVC, "SOP_CUS_IVC_LIN_AMT")]
        demote_measures_governed_by_a_metric(fields, [PO_METRIC])
        self.assertEqual(_enforcement(fields, "SOP_CUS_IVC_LIN_AMT"), "required")

    def test_dimensions_and_attributes_are_untouched(self):
        for role in ("dimension", "attribute", "date_key", "display_dimension"):
            with self.subTest(role=role):
                fields = [_field(PCH, "PFT_CTR_DMS_KEY", role=role)]
                demote_measures_governed_by_a_metric(fields, [PO_METRIC])
                self.assertEqual(_enforcement(fields, "PFT_CTR_DMS_KEY"), "required")

    def test_no_metrics_means_no_change(self):
        fields = [_field(PCH, "PCH_ORD_QTY")]
        for metrics in ([], None, [{"formula_type": "query", "base_table": PCH}]):
            with self.subTest(metrics=metrics):
                self.assertEqual(demote_measures_governed_by_a_metric(fields, metrics), [])
                self.assertEqual(_enforcement(fields, "PCH_ORD_QTY"), "required")

    def test_an_already_optional_field_is_not_reported_as_demoted(self):
        fields = [_field(PCH, "PCH_ORD_QTY", enforcement="optional")]
        self.assertEqual(demote_measures_governed_by_a_metric(fields, [PO_METRIC]), [])

    def test_the_deadlock_from_trace_760_cannot_recur(self):
        # Both requirements present at once is the exact contradiction.
        fields = [_field(PCH, "PCH_ORD_QTY"), _field(PCH, "PCH_ORD_LIN_CAD_AMT")]
        demote_measures_governed_by_a_metric(fields, [PO_METRIC])
        required = {f["column"] for f in fields if f["enforcement"] != "optional"}
        self.assertEqual(
            required, {"PCH_ORD_LIN_CAD_AMT"},
            "the plan still requires two different measures from one fact",
        )


class TestFormulaColumnExtraction(unittest.TestCase):

    def test_it_reads_the_column_out_of_the_formula(self):
        self.assertIn("PCH_ORD_LIN_CAD_AMT", metric_formula_columns(PO_METRIC))

    def test_sql_keywords_are_not_mistaken_for_columns(self):
        columns = metric_formula_columns({
            "sql_template": "SUM(NET_AMT) * 100.0 / NULLIF(SUM(GROSS_AMT), 0)",
        })
        self.assertEqual(columns, {"NET_AMT", "GROSS_AMT"})

    def test_declared_required_columns_are_honoured(self):
        columns = metric_formula_columns({
            "sql_template": "", "required_columns": f"{PCH}.RCT_QTY, RJT_QTY",
        })
        self.assertEqual(columns, {"RCT_QTY", "RJT_QTY"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
