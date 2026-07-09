"""
tests/test_defect_fixes.py

Regression tests for the tester's validation defect report:

  A. Validator flags bare columns that exist on NO base table (defect 6:
     SQL referenced YR directly on the fact; YR lives on DT_DMS) — with
     candidate_tables so the retry knows which table to join, and without
     false positives for CTE/select aliases or partial column metadata.
  B. SQL prompt: calendar columns live on the date dimension (defect 6b).
  C. Derived-metric gap guard (defect 3: "open order quantity" answered by
     summing the raw ordered column with HIGH confidence).
  D. Field-plan repair swaps a sibling measure for the required measure
     (defect 4: plan required CUS_IVC_LIN_AMT, SQL summed
     SOP_CUS_IVC_LIN_AMT — used to dead-end at "could not build a trusted
     query").
  E. "Did you mean" suggestions on terminal failures (defects 7/8: rephrase
     dead-end with no guidance).
  F. WS error paths degrade to a visible assistant_error instead of silence
     (defect 5 hardening).
"""
import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Isolated DB — suggestion tests write metrics/terms through the store layer.
_tmp_db = os.path.join(tempfile.mkdtemp(), "test_defects.db")
os.environ["QUERYBOT_DB_PATH"] = _tmp_db
for mod in list(sys.modules.keys()):
    if mod.startswith("store"):
        del sys.modules[mod]
import store.db as db_mod
db_mod.init_db()

import store
from core.answer_confidence import build_answer_confidence
from core.failure_messages import suggest_closest_terms, translate_failure
from core.metric_semantics import detect_derived_metric_gap
from core.pipeline_helpers import attempt_field_plan_repair
from core.validator import validate_sql_detailed


_FACT = "EMDW_DMART.CUS_ORD_IVC_FCT"
_DT = "EMDW_DMART.DT_DMS"
_CUS = "EMDW_DMART.CUS_DMS"
_COLS = {
    _FACT: {"CUS_IVC_LIN_AMT": "decimal", "SOP_CUS_IVC_LIN_AMT": "decimal",
            "CUS_IVC_DT_DMS_KEY": "int", "CUS_DMS_KEY": "int"},
    _DT: {"DT_DMS_KEY": "int", "YR": "int", "DT_DSC": "varchar"},
    _CUS: {"CUS_DMS_KEY": "int", "CUS_NM": "varchar"},
}


class BareColumnOnNoTableTests(unittest.TestCase):
    """Item A — defect 6a."""

    def test_bare_calendar_column_on_fact_flagged_with_candidate_tables(self):
        sql = ("SELECT SUM(f.CUS_IVC_LIN_AMT) AS AMT FROM [EMDW_DMART].[CUS_ORD_IVC_FCT] f "
               "JOIN [EMDW_DMART].[CUS_DMS] c ON f.CUS_DMS_KEY = c.CUS_DMS_KEY WHERE YR = 2025")
        r = validate_sql_detailed(sql, set(_COLS), "azure_sql", None, _COLS, None)
        self.assertFalse(r.ok)
        self.assertEqual(r.code, "unknown_column")
        err = next(e for e in r.errors if e["column"] == "YR")
        self.assertIn(_DT, err["candidate_tables"])
        self.assertIn("join that table", r.reason)

    def test_aliased_calendar_column_on_fact_flagged(self):
        sql = "SELECT SUM(f.CUS_IVC_LIN_AMT) FROM [EMDW_DMART].[CUS_ORD_IVC_FCT] f WHERE f.YR = 2025"
        r = validate_sql_detailed(sql, set(_COLS), "azure_sql", None, _COLS, None)
        self.assertFalse(r.ok)
        self.assertEqual(r.code, "unknown_column")

    def test_proper_date_dimension_join_passes(self):
        sql = ("SELECT COUNT(*) AS MATCHED, COUNT(f.CUS_IVC_LIN_AMT) AS NONNULL, "
               "COALESCE(SUM(f.CUS_IVC_LIN_AMT), 0) AS AMT "
               "FROM [EMDW_DMART].[CUS_ORD_IVC_FCT] f "
               "JOIN [EMDW_DMART].[DT_DMS] d ON f.CUS_IVC_DT_DMS_KEY = d.DT_DMS_KEY "
               "WHERE d.YR = 2025")
        r = validate_sql_detailed(sql, set(_COLS), "azure_sql", None, _COLS, None)
        self.assertTrue(r.ok, r.reason)

    def test_cte_and_select_aliases_not_false_positives(self):
        sql = ("WITH x AS (SELECT f.CUS_IVC_LIN_AMT AS AMT FROM [EMDW_DMART].[CUS_ORD_IVC_FCT] f "
               "JOIN [EMDW_DMART].[DT_DMS] d ON f.CUS_IVC_DT_DMS_KEY = d.DT_DMS_KEY) "
               "SELECT AMT FROM x")
        r = validate_sql_detailed(sql, set(_COLS), "azure_sql", None, _COLS, None)
        self.assertTrue(r.ok, r.reason)

    def test_partial_column_metadata_stays_lenient(self):
        # CUS_DMS has no column metadata: YR could legitimately live there,
        # so no unknown_column may fire (single-table attribution included).
        partial = {_FACT: _COLS[_FACT]}
        sql = ("SELECT SUM(f.CUS_IVC_LIN_AMT) AS AMT FROM [EMDW_DMART].[CUS_ORD_IVC_FCT] f "
               "JOIN [EMDW_DMART].[CUS_DMS] c ON f.CUS_DMS_KEY = c.CUS_DMS_KEY WHERE YR = 2025")
        r = validate_sql_detailed(sql, set(_COLS), "azure_sql", None, partial, None)
        self.assertNotEqual(r.code, "unknown_column")

    def test_datepart_units_not_flagged(self):
        sql = ("SELECT DATEPART(YEAR, TRY_CONVERT(date, CONVERT(varchar(8), f.CUS_IVC_DT_DMS_KEY), 112)) AS Y, "
               "COUNT(*) AS MATCHED, COUNT(f.CUS_IVC_LIN_AMT) AS NONNULL, "
               "COALESCE(SUM(f.CUS_IVC_LIN_AMT), 0) AS AMT "
               "FROM [EMDW_DMART].[CUS_ORD_IVC_FCT] f "
               "JOIN [EMDW_DMART].[DT_DMS] d ON f.CUS_IVC_DT_DMS_KEY = d.DT_DMS_KEY "
               "WHERE f.CUS_IVC_DT_DMS_KEY > 0 "
               "GROUP BY DATEPART(YEAR, TRY_CONVERT(date, CONVERT(varchar(8), f.CUS_IVC_DT_DMS_KEY), 112))")
        r = validate_sql_detailed(sql, set(_COLS), "azure_sql", None, _COLS, None)
        self.assertNotEqual(r.code, "unknown_column", r.reason)


class DateDimensionPromptRuleTests(unittest.TestCase):
    """Item B — defect 6b."""

    def test_rule_present_when_date_key_in_context(self):
        from core.llm import build_sql_system_prompt
        p = build_sql_system_prompt("azure_sql", "CUS_ORD_IVC_FCT: CUS_IVC_DT_DMS_KEY int, CUS_IVC_LIN_AMT decimal")
        self.assertIn("CALENDAR COLUMNS LIVE ON THE DATE DIMENSION", p)
        self.assertIn("Never referencing YR" if False else "NEVER on fact tables", p)
        self.assertIn("key / 10000", p)
        self.assertIn("must come from the date DIMENSION table", p)

    def test_rule_absent_without_date_key(self):
        from core.llm import build_sql_system_prompt
        p = build_sql_system_prompt("azure_sql", "T: A int, B varchar")
        self.assertNotIn("CALENDAR COLUMNS LIVE ON THE DATE DIMENSION", p)


class DerivedMetricGapTests(unittest.TestCase):
    """Item C — defect 3."""

    def test_open_order_quantity_fires(self):
        self.assertEqual(
            detect_derived_metric_gap("what is the open order quantity by purchase order date"),
            "open order quantity",
        )

    def test_formula_sources_suppress_gap(self):
        q = "what is the open order quantity"
        self.assertEqual(detect_derived_metric_gap(q, has_metric_formula=True), "")
        self.assertEqual(detect_derived_metric_gap(q, has_term_expression=True), "")

    def test_plain_questions_do_not_fire(self):
        for q in (
            "total purchase amount by purchase order date",
            "revenue by customer",
            "top 5 customers by sales",
            "which items have the lowest margin",
            "how many orders were shipped",
        ):
            self.assertEqual(detect_derived_metric_gap(q), "", q)

    def test_other_derived_phrasings_fire(self):
        self.assertEqual(detect_derived_metric_gap("show net sales value by month"), "net sales value")
        self.assertEqual(
            detect_derived_metric_gap("what is the outstanding balance by customer"),
            "outstanding balance",
        )

    def test_confidence_penalty_and_warning(self):
        with_gap = build_answer_confidence(
            validation_code="ok", row_count=30, derived_metric_gap="open order quantity",
        )
        without = build_answer_confidence(validation_code="ok", row_count=30)
        self.assertLess(with_gap["score"], without["score"])
        self.assertNotEqual(with_gap["level"], "high")
        self.assertTrue(any("open order quantity" in w for w in with_gap["warnings"]))
        self.assertTrue(any("Metric Registry or Business Terms" in w for w in with_gap["warnings"]))


class RequiredMeasureSwapRepairTests(unittest.TestCase):
    """Item D — defect 4 residual."""

    PLAN = {
        "enabled": True,
        "fields": [{
            "term": "sales amount", "table": _FACT, "column": "CUS_IVC_LIN_AMT",
            "role": "measure", "display_required": False, "confidence": 90,
            "source": "semantic_planner", "enforcement": "required",
        }],
        "joins": [],
    }

    def test_sibling_measure_swapped_deterministically(self):
        sql = ("SELECT COUNT(*) AS MATCHED, COUNT(f.SOP_CUS_IVC_LIN_AMT) AS NONNULL, "
               "COALESCE(SUM(f.SOP_CUS_IVC_LIN_AMT), 0) AS SALES_AMOUNT "
               "FROM [EMDW_DMART].[CUS_ORD_IVC_FCT] f WHERE f.CUS_IVC_DT_DMS_KEY > 0")
        ctx = {"semantic_plan": self.PLAN}
        r = validate_sql_detailed(sql, {_FACT}, "azure_sql", None, {_FACT: _COLS[_FACT]}, ctx)
        self.assertFalse(r.ok)
        self.assertEqual(r.code, "field_plan_mismatch")
        repaired = attempt_field_plan_repair(sql, "azure_sql", {_FACT}, None, {_FACT: _COLS[_FACT]}, ctx)
        self.assertTrue(repaired, "expected deterministic repair, got dead-end")
        self.assertNotIn("SOP_CUS_IVC_LIN_AMT", repaired.upper())
        self.assertIn("CUS_IVC_LIN_AMT", repaired.upper())
        recheck = validate_sql_detailed(repaired, {_FACT}, "azure_sql", None, {_FACT: _COLS[_FACT]}, ctx)
        self.assertTrue(recheck.ok, recheck.reason)

    def test_ambiguous_two_candidates_no_swap(self):
        cols = {_FACT: dict(_COLS[_FACT], OTHER_AMT="decimal")}
        sql = ("SELECT COUNT(*) AS M, COALESCE(SUM(f.SOP_CUS_IVC_LIN_AMT), 0) AS A, "
               "COALESCE(SUM(f.OTHER_AMT), 0) AS B "
               "FROM [EMDW_DMART].[CUS_ORD_IVC_FCT] f WHERE f.CUS_IVC_DT_DMS_KEY > 0")
        repaired = attempt_field_plan_repair(
            sql, "azure_sql", {_FACT}, None, cols, {"semantic_plan": self.PLAN},
        )
        self.assertEqual(repaired, "")


class ClosestTermSuggestionTests(unittest.TestCase):
    """Item E — defects 7/8."""

    @classmethod
    def setUpClass(cls):
        store.upsert_client("acct_defects", "portal")
        store.save_metric("acct_defects", {
            "name": "Purchase Order Quantity", "sql_template": "SELECT 1",
            "synonyms": "purchase quantity, po quantity", "description": "",
        })
        cls.kb = tempfile.mkdtemp()
        Path(cls.kb, "_semantic_model.json").write_text(json.dumps({
            "tables": [{"table": "F", "qualified_name": "S.F", "fields": [{
                "column": "CUS_ORD_QTY", "status": "approved",
                "approved_meaning": "Customer ordered quantity",
                "business_candidates": ["customer ordered quantity"],
            }]}],
        }), encoding="utf-8")

    def test_defect_question_gets_close_suggestions(self):
        s = suggest_closest_terms(
            "what is the total previous year customer ordered quantity",
            "acct_defects", self.kb,
        )
        self.assertTrue(s)
        self.assertEqual(s[0], "Customer ordered quantity")

    def test_unrelated_question_gets_none(self):
        self.assertEqual(
            suggest_closest_terms("completely unrelated gibberish zzz", "acct_defects", self.kb),
            [],
        )

    def test_translate_failure_appends_suggestions(self):
        rca = translate_failure(
            kind="validation", code="unknown_column", reason="x",
            suggestions=["Customer ordered quantity"],
        )
        self.assertIn("Closest known terms in your data: Customer ordered quantity.",
                      rca["suggested_next_step"])

    def test_translate_failure_unchanged_without_suggestions(self):
        rca = translate_failure(kind="validation", code="unknown_column", reason="x")
        self.assertNotIn("Closest known terms", rca["suggested_next_step"])

    def test_cannot_generate_path_wired_in_pipeline(self):
        src = (ROOT / "core" / "query_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("suggest_closest_terms(question, account_id, state.get(\"kb_dir\", \"\"))", src)
        self.assertIn("Closest known terms in your data:", src)
        self.assertIn("suggestions=_suggest", src)


class WsFallbackErrorTests(unittest.TestCase):
    """Item F — defect 5 hardening."""

    def test_send_assistant_response_sends_error_frame_on_failure(self):
        from gateway.web_adapter import WebAdapter

        class FakeWS:
            def __init__(self):
                self.sent = []
                self.fail_first = True

            async def send_json(self, payload):
                if self.fail_first:
                    self.fail_first = False
                    raise TypeError("Object of type Decimal is not JSON serializable")
                self.sent.append(payload)

        ws = FakeWS()
        adapter = WebAdapter(ws, "acct", "user1")
        asyncio.run(adapter.send_assistant_response(None, {"type": "assistant_response"}))
        self.assertEqual(len(ws.sent), 1)
        self.assertEqual(ws.sent[0]["type"], "assistant_error")
        self.assertIn("Something went wrong", ws.sent[0]["content"])

    def test_ws_bg_task_handler_sends_error_frame(self):
        src = (ROOT / "gateway" / "webhooks.py").read_text(encoding="utf-8")
        anchor = src.index('log.error("WS bg task error: %s", e)')
        tail = src[anchor:anchor + 900]
        self.assertIn('"type": "assistant_error"', tail)
        self.assertIn("Something went wrong while preparing your answer", tail)


if __name__ == "__main__":
    unittest.main()
