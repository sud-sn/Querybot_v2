import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.graph_resolver import detect_entities, resolve_for_question
from core.llm import build_sql_system_prompt
from core.query_semantics import analyze_query_intent, detect_top_n_intent
from core.semantic_planner import build_semantic_field_plan
from core.validator import has_identity_filter, normalize_generated_sql, validate_sql, validate_sql_detailed
from core.answer_confidence import build_answer_confidence
from core.answer_rca import build_business_rca, extract_sql_tables
from core.query_router import should_route_to_result_cache, build_duckdb_system_prompt
from core.response_builder import build_assistant_response, detect_null_metric_issue


KNOWN = {
    "CHATBOT_DB.PROFITABILITY.CUS_ORD_IVC_FCT",
    "PROFITABILITY.CUS_ORD_IVC_FCT",
    "CUS_ORD_IVC_FCT",
    "CHATBOT_DB.PROFITABILITY.ITM_BAL_PRD_FCT",
    "PROFITABILITY.ITM_BAL_PRD_FCT",
    "ITM_BAL_PRD_FCT",
    "CHATBOT_DB.PROFITABILITY.FIFO_BI_SAL_MGP_EXT",
    "PROFITABILITY.FIFO_BI_SAL_MGP_EXT",
    "FIFO_BI_SAL_MGP_EXT",
    "CHATBOT_DB.PROFITABILITY.FIFOBISALMGPEXT",
    "PROFITABILITY.FIFOBISALMGPEXT",
    "FIFOBISALMGPEXT",
    "CHATBOT_DB.PROFITABILITY.OOLINE",
    "PROFITABILITY.OOLINE",
    "OOLINE",
    "CHATBOT_DB.PROFITABILITY.DIM_DIVISION",
    "PROFITABILITY.DIM_DIVISION",
    "DIM_DIVISION",
    "CHATBOT_DB.PROFITABILITY.WHS_DMS",
    "PROFITABILITY.WHS_DMS",
    "WHS_DMS",
    "CHATBOT_DB.PHARMACY.DIMPATIENT",
    "PHARMACY.DIMPATIENT",
    "DIMPATIENT",
}

COLUMNS = {
    "PROFITABILITY.CUS_ORD_IVC_FCT": {
        "ITM_GRP_DMS_KEY": "int",
        "CUS_IVC_DT_DMS_KEY": "int",
        "CUS_IVC_LIN_AMT": "decimal",
        "CUS_ORD_NUM": "varchar",
        "CUS_ORD_LIN_NUM": "int",
        "CUS_ORD_LIN_SFX": "int",
        "WHS_DMS_KEY": "int",
        "CUS_DMS_KEY": "int",
        "SOP_CUS_IVC_LIN_AMT": "decimal",
        "SOP_CUS_IVC_LIN_CST_AMT": "decimal",
        "SOP_CUS_LIN_GRS_PFT_AMT": "decimal",
    },
    "CUS_ORD_IVC_FCT": {
        "ITM_GRP_DMS_KEY": "int",
        "CUS_IVC_DT_DMS_KEY": "int",
        "CUS_IVC_LIN_AMT": "decimal",
        "CUS_ORD_NUM": "varchar",
        "CUS_ORD_LIN_NUM": "int",
        "CUS_ORD_LIN_SFX": "int",
        "WHS_DMS_KEY": "int",
        "CUS_DMS_KEY": "int",
        "SOP_CUS_IVC_LIN_AMT": "decimal",
        "SOP_CUS_IVC_LIN_CST_AMT": "decimal",
        "SOP_CUS_LIN_GRS_PFT_AMT": "decimal",
    },
    "PROFITABILITY.WHS_DMS": {
        "WHS_DMS_KEY": "int",
        "WHS_CD": "varchar",
        "WHS_DSC": "varchar",
    },
    "WHS_DMS": {
        "WHS_DMS_KEY": "int",
        "WHS_CD": "varchar",
        "WHS_DSC": "varchar",
    },
    "PROFITABILITY.ITM_BAL_PRD_FCT": {
        "NUM_OF_RCT": "int",
        "ITM_DMS_KEY": "int",
    },
    "ITM_BAL_PRD_FCT": {
        "NUM_OF_RCT": "int",
        "ITM_DMS_KEY": "int",
    },
    "PROFITABILITY.FIFO_BI_SAL_MGP_EXT": {
        "PCLA": "decimal",
        "FL_DT_TS": "datetime",
        "ORNO": "varchar",
    },
    "FIFO_BI_SAL_MGP_EXT": {
        "PCLA": "decimal",
        "FL_DT_TS": "datetime",
        "ORNO": "varchar",
    },
    "PROFITABILITY.FIFOBISALMGPEXT": {
        "PCLA": "decimal",
        "ORNO": "varchar",
        "PONR": "int",
        "POSX": "int",
    },
    "FIFOBISALMGPEXT": {
        "PCLA": "decimal",
        "ORNO": "varchar",
        "PONR": "int",
        "POSX": "int",
    },
    "PROFITABILITY.OOLINE": {
        "ORNO": "varchar",
        "PONR": "int",
        "POSX": "int",
        "DIVI": "varchar",
    },
    "OOLINE": {
        "ORNO": "varchar",
        "PONR": "int",
        "POSX": "int",
        "DIVI": "varchar",
    },
    "PROFITABILITY.DIM_DIVISION": {
        "DIVI": "varchar",
        "DIVISION_NAME": "varchar",
    },
    "DIM_DIVISION": {
        "DIVI": "varchar",
        "DIVISION_NAME": "varchar",
    },
    "CHATBOT_DB.PHARMACY.DIMPATIENT": {
        "AGE": "int",
        "PATIENT_ID": "varchar",
    },
    "PHARMACY.DIMPATIENT": {
        "AGE": "int",
        "PATIENT_ID": "varchar",
    },
}


class StrictColumnValidationTests(unittest.TestCase):
    def test_rejects_missing_underscore_column(self):
        ok, msg, code = validate_sql(
            "SELECT cus.ITMGRPDMS_KEY FROM [PROFITABILITY].[CUS_ORD_IVC_FCT] cus",
            KNOWN,
            "azure_sql",
            None,
            COLUMNS,
        )
        self.assertFalse(ok)
        self.assertEqual(code, "unknown_column")
        self.assertIn("ITM_GRP_DMS_KEY", msg)

    def test_rejects_numofrct_and_suggests_num_of_rct(self):
        result = validate_sql_detailed(
            "SELECT itm.NUMOFRCT FROM [PROFITABILITY].[ITM_BAL_PRD_FCT] itm",
            KNOWN,
            "azure_sql",
            None,
            COLUMNS,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "unknown_column")
        self.assertIn("NUM_OF_RCT", result.errors[0]["suggestions"])

    def test_wrong_table_column_points_to_candidate_table(self):
        result = validate_sql_detailed(
            "SELECT itm.DIVI, SUM(itm.NUM_OF_RCT) FROM [PROFITABILITY].[ITM_BAL_PRD_FCT] itm GROUP BY itm.DIVI",
            KNOWN,
            "azure_sql",
            None,
            COLUMNS,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "unknown_column")
        self.assertIn("Exact column exists on", result.reason)
        self.assertTrue(any("DIM_DIVISION" in t for t in result.errors[0]["candidate_tables"]))

    def test_rejects_yoy_hallucinated_year_column(self):
        ok, _, code = validate_sql(
            "SELECT CAST(YEA4 AS INT) AS Year, SUM(PCLA) FROM [PROFITABILITY].[FIFO_BI_SAL_MGP_EXT] GROUP BY CAST(YEA4 AS INT)",
            KNOWN,
            "azure_sql",
            None,
            COLUMNS,
        )
        self.assertFalse(ok)
        self.assertEqual(code, "unknown_column")

    def test_rejects_format_on_numeric_date_key(self):
        ok, msg, code = validate_sql(
            "SELECT FORMAT(cus.CUS_IVC_DT_DMS_KEY, 'yyyy-MM') AS PERIOD FROM [PROFITABILITY].[CUS_ORD_IVC_FCT] cus",
            KNOWN,
            "azure_sql",
            None,
            COLUMNS,
        )
        self.assertFalse(ok)
        self.assertEqual(code, "date_key_format")
        self.assertIn("Convert YYYYMMDD", msg)

    def test_normalizes_misplaced_sql_server_datepart_cast(self):
        sql = (
            "WITH base AS ("
            "SELECT CAST(YEAR(TRY_CONVERT(DATE, CONVERT(VARCHAR(8), CUS_IVC_DT_DMS_KEY), 112))) AS INT) AS YR, "
            "SUM(SOP_CUS_IVC_LIN_AMT) AS Revenue "
            "FROM [PROFITABILITY].[CUS_ORD_IVC_FCT] "
            "WHERE CUS_IVC_DT_DMS_KEY > 0 "
            "GROUP BY CAST(YEAR(TRY_CONVERT(DATE, CONVERT(VARCHAR(8), CUS_IVC_DT_DMS_KEY), 112))) AS INT)"
            ") SELECT * FROM base"
        )
        normalized = normalize_generated_sql(sql, "azure_sql")
        self.assertNotIn("))) AS INT)", normalized)
        self.assertIn(
            "CAST(YEAR(TRY_CONVERT(date, CONVERT(varchar(8), CUS_IVC_DT_DMS_KEY), 112)) AS INT)",
            normalized,
        )
        result = validate_sql_detailed(
            normalized,
            KNOWN,
            "azure_sql",
            None,
            COLUMNS,
        )
        self.assertTrue(result.ok, result.reason)

    def test_rejects_filtered_sum_without_null_diagnostics(self):
        result = validate_sql_detailed(
            "SELECT SUM(CUS_IVC_LIN_AMT) AS Revenue "
            "FROM [PROFITABILITY].[CUS_ORD_IVC_FCT] "
            "WHERE CUS_DMS_KEY = 1055930",
            KNOWN,
            "azure_sql",
            None,
            COLUMNS,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "null_aggregate_diagnostic")
        self.assertTrue(result.errors[0]["requires_matched_count"])
        self.assertTrue(result.errors[0]["requires_non_null_count"])
        self.assertTrue(result.errors[0]["requires_null_safe_sum"])

    def test_plain_time_bounded_aggregate_does_not_require_null_diagnostics(self):
        # Regression: "what is my sales for the last 7 days" — a WHERE clause
        # that only range-filters a date column has no "does this entity
        # exist" ambiguity the way "revenue for customer 123" does. Forcing
        # MatchedRows/NonNullMetricRows onto it broke the single-value answer
        # shape and cost an unnecessary repair retry on every plain
        # time-bounded question.
        result = validate_sql_detailed(
            "SELECT SUM(CUS_IVC_LIN_AMT) AS TOTAL_SALES "
            "FROM [PROFITABILITY].[CUS_ORD_IVC_FCT] "
            "WHERE CUS_IVC_DT_DMS_KEY >= 20260101",
            KNOWN, "azure_sql", None, COLUMNS,
        )
        self.assertTrue(result.ok, result.reason)

    def test_exact_date_equality_does_not_require_null_diagnostics(self):
        result = validate_sql_detailed(
            "SELECT SUM(CUS_IVC_LIN_AMT) AS TOTAL_SALES "
            "FROM [PROFITABILITY].[CUS_ORD_IVC_FCT] "
            "WHERE CUS_IVC_DT_DMS_KEY = 20260101",
            KNOWN, "azure_sql", None, COLUMNS,
        )
        self.assertTrue(result.ok, result.reason)

    def test_identity_filter_combined_with_date_range_still_requires_diagnostics(self):
        # The entity-lookup ambiguity is still present even alongside a date
        # filter — must still be rejected without diagnostics.
        result = validate_sql_detailed(
            "SELECT SUM(CUS_IVC_LIN_AMT) AS Revenue "
            "FROM [PROFITABILITY].[CUS_ORD_IVC_FCT] "
            "WHERE CUS_DMS_KEY = 1055930 AND CUS_IVC_DT_DMS_KEY >= 20260101",
            KNOWN, "azure_sql", None, COLUMNS,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "null_aggregate_diagnostic")

    def test_has_identity_filter_public_helper(self):
        # core/examples.py reuses this exact classification to filter stale
        # few-shot examples — must agree with the validator's own judgment.
        self.assertFalse(has_identity_filter(
            "SELECT SUM(x) AS T FROM TBL WHERE SALE_DT_DMS_KEY >= 20260101"))
        self.assertTrue(has_identity_filter(
            "SELECT SUM(x) AS T FROM TBL WHERE CUS_DMS_KEY = 123"))
        self.assertFalse(has_identity_filter("SELECT SUM(x) AS T FROM TBL"))
        self.assertFalse(has_identity_filter(""))
        self.assertFalse(has_identity_filter("not valid sql ((("))

    def test_accepts_filtered_sum_with_null_diagnostics(self):
        result = validate_sql_detailed(
            "SELECT COUNT_BIG(*) AS [MatchedRows], "
            "COUNT(CUS_IVC_LIN_AMT) AS [NonNullRevenueRows], "
            "COALESCE(SUM(CUS_IVC_LIN_AMT), 0) AS [Revenue] "
            "FROM [PROFITABILITY].[CUS_ORD_IVC_FCT] "
            "WHERE CUS_DMS_KEY = 1055930",
            KNOWN,
            "azure_sql",
            None,
            COLUMNS,
        )
        self.assertTrue(result.ok, result.reason)

    def test_rejects_revenue_query_that_ignores_approved_formula_metric(self):
        metric = {
            "name": "Revenue",
            "synonyms": "total revenue,sales revenue",
            "formula_type": "expression",
            "sql_template": "SUM(SOP_CUS_IVC_LIN_AMT)",
            "required_columns": "SOP_CUS_IVC_LIN_AMT",
        }
        result = validate_sql_detailed(
            "SELECT COUNT_BIG(*) AS [MatchedRows], "
            "COUNT(CUS_IVC_LIN_AMT) AS [NonNullRevenueRows], "
            "COALESCE(SUM(CUS_IVC_LIN_AMT), 0) AS [Revenue] "
            "FROM [PROFITABILITY].[CUS_ORD_IVC_FCT] "
            "WHERE CUS_DMS_KEY = 1035573",
            KNOWN,
            "azure_sql",
            None,
            COLUMNS,
            {"question": "what is the revenue for the customer 1035573", "metric_formulas": [metric]},
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "metric_formula_mismatch")
        self.assertIn("SOP_CUS_IVC_LIN_AMT", result.reason)

    def test_accepts_revenue_query_that_uses_approved_formula_metric(self):
        metric = {
            "name": "Revenue",
            "synonyms": "total revenue,sales revenue",
            "formula_type": "expression",
            "sql_template": "SUM(SOP_CUS_IVC_LIN_AMT)",
            "required_columns": "SOP_CUS_IVC_LIN_AMT",
        }
        result = validate_sql_detailed(
            "SELECT COUNT_BIG(*) AS [MatchedRows], "
            "COUNT(SOP_CUS_IVC_LIN_AMT) AS [NonNullRevenueRows], "
            "COALESCE(SUM(SOP_CUS_IVC_LIN_AMT), 0) AS [Revenue] "
            "FROM [PROFITABILITY].[CUS_ORD_IVC_FCT] "
            "WHERE CUS_DMS_KEY = 1035573",
            KNOWN,
            "azure_sql",
            None,
            COLUMNS,
            {"question": "what is the revenue for the customer 1035573", "metric_formulas": [metric]},
        )
        self.assertTrue(result.ok, result.reason)

    def test_rejects_missing_fifo_single_table_null_filter(self):
        sql = (
            "SELECT DISTINCT fif.ORNO AS OrderNumber, fif.POSX AS OrderLineNumber, 0 AS PCLA "
            "FROM [PROFITABILITY].[FIFOBISALMGPEXT] fif "
            "WHERE fif.PCLA IS NULL"
        )
        ok, msg, code = validate_sql(
            sql,
            KNOWN,
            "azure_sql",
            None,
            COLUMNS,
            {"intent": {"wants_missing_records": True}},
        )
        self.assertFalse(ok)
        self.assertEqual(code, "anti_join_shape")
        self.assertIn("source table", msg)

    def test_accepts_missing_fifo_left_join_shape(self):
        sql = (
            "SELECT oo.ORNO AS OrderNumber, oo.PONR AS OrderLineNumber, "
            "COALESCE(fif.PCLA, 0) AS PCLA "
            "FROM [PROFITABILITY].[OOLINE] oo "
            "LEFT JOIN [PROFITABILITY].[FIFOBISALMGPEXT] fif "
            "ON oo.ORNO = fif.ORNO AND oo.PONR = fif.PONR AND oo.POSX = fif.POSX "
            "WHERE fif.ORNO IS NULL"
        )
        ok, msg, code = validate_sql(
            sql,
            KNOWN,
            "azure_sql",
            None,
            COLUMNS,
            {"intent": {"wants_missing_records": True}},
        )
        self.assertTrue(ok, msg)
        self.assertEqual(code, "ok")

    def test_semantic_plan_maps_division_question_to_correct_tables(self):
        plan = build_semantic_field_plan(
            "For each division, what percentage of total invoice line amount comes from each item group?",
            COLUMNS,
        )
        self.assertTrue(plan["enabled"])
        fields = {(f["term"], f["column"], f["table"]) for f in plan["fields"]}
        self.assertTrue(any(col == "DIVI" and table.endswith("OOLINE") for _, col, table in fields))
        self.assertTrue(any(col == "ITM_GRP_DMS_KEY" and table.endswith("CUS_ORD_IVC_FCT") for _, col, table in fields))
        self.assertTrue(any(col == "CUS_IVC_LIN_AMT" and table.endswith("CUS_ORD_IVC_FCT") for _, col, table in fields))
        self.assertTrue(plan["joins"])
        self.assertFalse(any(col == "AGE" for _, col, _ in fields))

    def test_semantic_plan_ignores_day_inside_duration_idiom(self):
        # Regression: a bare "DAY" column's alias pluralizes to "days",
        # which falsely matched inside "avg days to pay" — a duration
        # metric idiom, not a request to group by calendar day. This
        # incorrectly forced a DT_DMS join/field requirement onto queries
        # for a saved "Avg Days To Pay" row-calculated metric, blocking
        # otherwise-correct SQL. Confirmed against a real production query.
        table_columns = {
            "EMDW_DMART.CUS_ORD_IVC_FCT": {"CUS_DMS_KEY": "bigint", "SOP_CUS_IVC_LIN_AMT": "decimal"},
            "EMDW_DMART.CUS_DMS": {"CUS_DMS_KEY": "bigint", "CUS_NM": "varchar"},
            "EMDW_DMART.DT_DMS": {"DT_DMS_KEY": "bigint", "DAY": "int"},
        }
        plan = build_semantic_field_plan(
            "what is the avg days to pay by top 10 customers by revenue?",
            table_columns,
        )
        columns = {f["column"] for f in plan.get("fields", [])}
        self.assertNotIn("DAY", columns)

    def test_semantic_plan_ignores_day_in_between_since_until_idioms(self):
        # Regression: the original duration-idiom guard only caught "days
        # to X" — a differently-phrased duration question ("number of days
        # present between payment") still falsely required the DAY column,
        # blocking otherwise-correct SQL a second time in production.
        table_columns = {
            "EMDW_DMART.CUS_ORD_IVC_FCT": {"CUS_DMS_KEY": "bigint", "PAY_DT_DMS_KEY": "bigint"},
            "EMDW_DMART.CUS_DMS": {"CUS_DMS_KEY": "bigint", "CUS_NM": "varchar"},
            "EMDW_DMART.DT_DMS": {"DT_DMS_KEY": "bigint", "DAY": "int"},
        }
        for question in (
            "what is the number of days present between payment by each customer",
            "days between due date and payment date",
            "days since last order",
            "how many days until delivery",
        ):
            plan = build_semantic_field_plan(question, table_columns)
            columns = {f["column"] for f in plan.get("fields", [])}
            self.assertNotIn("DAY", columns, f"false match for: {question!r}")

    def test_subsumed_generic_term_is_dropped(self):
        # Production repro: "Show the number of patients grouped by their
        # insurance coverage type." — the bare word "type" matched
        # DIM_INGREDIENT.TYPE and became a second REQUIRED field alongside
        # DIM_PATIENT.Coverage_Type, so the validator rejected correct SQL
        # with field_plan_mismatch ("SQL did not use required semantic
        # field type: ...DIM_INGREDIENT.TYPE"). A term whose every
        # occurrence sits inside a longer matched term is not an
        # independent mention.
        table_columns = {
            "CHATBOT_DB.PHARMACY.DIM_PATIENT": {
                "PATIENT_ID": "int", "COVERAGE_TYPE": "varchar",
            },
            "CHATBOT_DB.PHARMACY.DIM_INGREDIENT": {
                "INGREDIENT_ID": "int", "TYPE": "varchar",
            },
        }
        plan = build_semantic_field_plan(
            "Show the number of patients grouped by their insurance coverage type.",
            table_columns,
        )
        self.assertTrue(plan["enabled"])
        fields = {(f["table"], f["column"]) for f in plan["fields"]}
        self.assertIn(("CHATBOT_DB.PHARMACY.DIM_PATIENT", "COVERAGE_TYPE"), fields)
        self.assertNotIn(("CHATBOT_DB.PHARMACY.DIM_INGREDIENT", "TYPE"), fields)

    def test_subsumed_term_dropped_for_plural_phrasing_too(self):
        table_columns = {
            "CHATBOT_DB.PHARMACY.DIM_PATIENT": {"COVERAGE_TYPE": "varchar"},
            "CHATBOT_DB.PHARMACY.DIM_INGREDIENT": {"TYPE": "varchar"},
        }
        plan = build_semantic_field_plan(
            "patient count by coverage types", table_columns,
        )
        fields = {(f["table"], f["column"]) for f in plan.get("fields", [])}
        self.assertNotIn(("CHATBOT_DB.PHARMACY.DIM_INGREDIENT", "TYPE"), fields)

    def test_independent_mention_of_shorter_term_is_kept(self):
        # "state and state code": the first "state" stands alone, so BOTH
        # terms must survive — the subsumption drop only fires when every
        # occurrence of the shorter term is inside a longer one.
        table_columns = {
            "CHATBOT_DB.PHARMACY.DIM_PATIENT": {
                "STATE": "varchar", "STATE_CODE": "varchar",
            },
        }
        plan = build_semantic_field_plan(
            "patients by state and state code", table_columns,
        )
        columns = {f["column"] for f in plan.get("fields", [])}
        self.assertIn("STATE_CODE", columns)
        self.assertIn("STATE", columns)

    def test_ambiguous_cross_table_term_is_demoted_to_optional(self):
        # Production repro: "find the doctors state who are top 5 in
        # revenue" — STATE exists in both DIM_PATIENT and DIM_PRESCRIBER
        # with identical scores; the arbitrary pick (DIM_PATIENT) became a
        # hard requirement and rejected correct SQL that used
        # DIM_PRESCRIBER.STATE. Ties must demote to a hint.
        table_columns = {
            "CHATBOT_DB.PHARMACY.DIM_PATIENT": {
                "PATIENT_ID": "int", "STATE": "varchar",
            },
            "CHATBOT_DB.PHARMACY.DIM_PRESCRIBER": {
                "PRESCRIBER_ID": "int", "STATE": "varchar",
            },
            "CHATBOT_DB.PHARMACY.FACT_PRESCRIPTION_FILL": {
                "PRESCRIBER_ID": "int", "TOTAL_CHARGE_USD": "decimal",
            },
        }
        plan = build_semantic_field_plan(
            "find the doctors state who are top 5 in revenue", table_columns,
        )
        state_fields = [f for f in plan.get("fields", []) if f["column"] == "STATE"]
        self.assertTrue(state_fields)
        for f in state_fields:
            self.assertEqual(f.get("enforcement"), "optional", f)

    def test_correct_sql_passes_despite_ambiguous_state_plan(self):
        # End-to-end: the exact SQL the LLM generated in production (using
        # DIM_PRESCRIBER.STATE) must now pass validation even though the
        # plan's arbitrary tie-pick suggested DIM_PATIENT.STATE.
        table_columns = {
            "CHATBOT_DB.PHARMACY.DIM_PATIENT": {
                "PATIENT_ID": "int", "STATE": "varchar",
            },
            "CHATBOT_DB.PHARMACY.DIM_PRESCRIBER": {
                "PRESCRIBER_ID": "int", "STATE": "varchar",
            },
            "CHATBOT_DB.PHARMACY.FACT_PRESCRIPTION_FILL": {
                "PRESCRIBER_ID": "int", "TOTAL_CHARGE_USD": "decimal",
            },
        }
        question = "find the doctors state who are top 5 in revenue"
        plan = build_semantic_field_plan(question, table_columns)
        known = {
            "CHATBOT_DB.PHARMACY.DIM_PATIENT", "PHARMACY.DIM_PATIENT", "DIM_PATIENT",
            "CHATBOT_DB.PHARMACY.DIM_PRESCRIBER", "PHARMACY.DIM_PRESCRIBER", "DIM_PRESCRIBER",
            "CHATBOT_DB.PHARMACY.FACT_PRESCRIPTION_FILL", "PHARMACY.FACT_PRESCRIPTION_FILL",
            "FACT_PRESCRIPTION_FILL",
        }
        sql = (
            "SELECT TOP 5 dp.State AS Doctor_State, "
            "SUM(fpf.Total_Charge_USD) AS Total_Revenue "
            "FROM PHARMACY.FACT_Prescription_Fill fpf "
            "JOIN PHARMACY.DIM_Prescriber dp ON fpf.Prescriber_ID = dp.Prescriber_ID "
            "GROUP BY dp.State ORDER BY Total_Revenue DESC;"
        )
        ok, msg, code = validate_sql(
            sql, known, "azure_sql", None, table_columns,
            {"semantic_plan": plan},
        )
        self.assertTrue(ok, msg)
        self.assertEqual(code, "ok")

    def test_semantic_plan_still_matches_day_for_calendar_grouping(self):
        # The guard must not block legitimate "group by day" questions.
        table_columns = {
            "EMDW_DMART.CUS_ORD_IVC_FCT": {"CUS_DMS_KEY": "bigint", "SOP_CUS_IVC_LIN_AMT": "decimal"},
            "EMDW_DMART.DT_DMS": {"DT_DMS_KEY": "bigint", "DAY": "int"},
        }
        plan = build_semantic_field_plan("show revenue by day", table_columns)
        columns = {f["column"] for f in plan.get("fields", [])}
        self.assertIn("DAY", columns)

    def test_semantic_plan_respects_selected_schema(self):
        plan = build_semantic_field_plan(
            "For each division, what percentage of total invoice line amount comes from each item group?",
            COLUMNS,
            selected_schema="PROFITABILITY",
        )
        self.assertTrue(plan["enabled"])
        self.assertTrue(all(".PHARMACY." not in f["table"] for f in plan["fields"]))
        self.assertFalse(any(f["column"] == "AGE" for f in plan["fields"]))

    def test_semantic_plan_expands_allowed_fqn_scope(self):
        plan = build_semantic_field_plan(
            "For each division, what percentage of total invoice line amount comes from each item group?",
            COLUMNS,
            allowed_tables={
                "CHATBOT_DB.PROFITABILITY.CUS_ORD_IVC_FCT",
                "CHATBOT_DB.PROFITABILITY.OOLINE",
            },
            selected_schema="PROFITABILITY",
        )
        self.assertTrue(plan["enabled"])
        self.assertTrue(any(f["column"] == "DIVI" and f["table"].endswith("OOLINE") for f in plan["fields"]))
        self.assertTrue(any(f["column"] == "CUS_IVC_LIN_AMT" and f["table"].endswith("CUS_ORD_IVC_FCT") for f in plan["fields"]))

    def test_semantic_plan_validator_rejects_ignored_field_source(self):
        plan = build_semantic_field_plan(
            "For each division, what percentage of total invoice line amount comes from each item group?",
            COLUMNS,
        )
        sql = (
            "SELECT o.DIVI, SUM(c.CUS_IVC_LIN_AMT) AS TotalInvoiceLineAmount "
            "FROM [PROFITABILITY].[CUS_ORD_IVC_FCT] c "
            "JOIN [PROFITABILITY].[OOLINE] o "
            "ON c.CUS_ORD_NUM = o.ORNO AND c.CUS_ORD_LIN_NUM = o.PONR AND c.CUS_ORD_LIN_SFX = o.POSX "
            "GROUP BY o.DIVI"
        )
        result = validate_sql_detailed(
            sql,
            KNOWN,
            "azure_sql",
            None,
            COLUMNS,
            {"semantic_plan": plan},
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "field_plan_mismatch")
        self.assertIn("ITM_GRP_DMS_KEY", result.reason)

    def test_semantic_plan_validator_accepts_division_item_group_invoice_share_shape(self):
        plan = build_semantic_field_plan(
            "For each division, what percentage of total invoice line amount comes from each item group?",
            COLUMNS,
        )
        sql = (
            "SELECT o.DIVI AS Division, c.ITM_GRP_DMS_KEY AS ItemGroup, "
            "SUM(c.CUS_IVC_LIN_AMT) AS TotalInvoiceLineAmount, "
            "SUM(c.CUS_IVC_LIN_AMT) * 100.0 / NULLIF(SUM(SUM(c.CUS_IVC_LIN_AMT)) "
            "OVER (PARTITION BY o.DIVI), 0) AS PercentageOfDivisionTotal "
            "FROM [PROFITABILITY].[CUS_ORD_IVC_FCT] c "
            "JOIN [PROFITABILITY].[OOLINE] o "
            "ON c.CUS_ORD_NUM = o.ORNO AND c.CUS_ORD_LIN_NUM = o.PONR AND c.CUS_ORD_LIN_SFX = o.POSX "
            "GROUP BY o.DIVI, c.ITM_GRP_DMS_KEY"
        )
        result = validate_sql_detailed(
            sql,
            KNOWN,
            "azure_sql",
            None,
            COLUMNS,
            {"semantic_plan": plan},
        )
        self.assertTrue(result.ok, result.reason)

    def test_semantic_plan_prefers_warehouse_description_over_key(self):
        plan = build_semantic_field_plan(
            "Which warehouses have high invoice revenue but low gross profit percentage?",
            COLUMNS,
        )
        self.assertTrue(plan["enabled"])
        fields = {(f["term"], f["column"], f["table"], f.get("role")) for f in plan["fields"]}
        self.assertTrue(any(col == "WHS_DSC" and table.endswith("WHS_DMS") and role == "display_dimension"
                            for _, col, table, role in fields))
        self.assertFalse(any(col == "WHS_DMS_KEY" and role == "dimension" for _, col, _, role in fields))
        self.assertTrue(any(
            any(left == "WHS_DMS_KEY" and right == "WHS_DMS_KEY" for left, right in edge.get("conditions", []))
            for edge in plan["joins"]
        ))

    def test_semantic_plan_allows_warehouse_key_when_user_asks_for_key(self):
        plan = build_semantic_field_plan(
            "Show invoice revenue by warehouse key",
            COLUMNS,
        )
        self.assertTrue(plan["enabled"])
        self.assertTrue(any(f["column"] == "WHS_DMS_KEY" for f in plan["fields"]))
        self.assertFalse(any(f["column"] == "WHS_DSC" for f in plan["fields"]))

    def test_semantic_plan_validator_rejects_warehouse_key_as_display_label(self):
        plan = build_semantic_field_plan(
            "Which warehouses have high invoice revenue but low gross profit percentage?",
            COLUMNS,
        )
        sql = (
            "SELECT c.WHS_DMS_KEY AS Warehouse, "
            "SUM(c.SOP_CUS_IVC_LIN_AMT) AS TotalRevenue, "
            "SUM(c.SOP_CUS_LIN_GRS_PFT_AMT) AS GrossProfit "
            "FROM [PROFITABILITY].[CUS_ORD_IVC_FCT] c "
            "GROUP BY c.WHS_DMS_KEY"
        )
        result = validate_sql_detailed(
            sql,
            KNOWN,
            "azure_sql",
            None,
            COLUMNS,
            {"semantic_plan": plan},
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "field_plan_mismatch")
        self.assertIn("WHS_DSC", result.reason)

    def test_semantic_plan_validator_accepts_warehouse_description_shape(self):
        plan = build_semantic_field_plan(
            "Which warehouses have high invoice revenue but low gross profit percentage?",
            COLUMNS,
        )
        sql = (
            "SELECT w.WHS_DSC AS Warehouse, "
            "SUM(c.SOP_CUS_IVC_LIN_AMT) AS TotalRevenue, "
            "SUM(c.SOP_CUS_LIN_GRS_PFT_AMT) AS GrossProfit "
            "FROM [PROFITABILITY].[CUS_ORD_IVC_FCT] c "
            "JOIN [PROFITABILITY].[WHS_DMS] w ON c.WHS_DMS_KEY = w.WHS_DMS_KEY "
            "GROUP BY w.WHS_DSC"
        )
        result = validate_sql_detailed(
            sql,
            KNOWN,
            "azure_sql",
            None,
            COLUMNS,
            {"semantic_plan": plan},
        )
        self.assertTrue(result.ok, result.reason)

    def test_display_dimension_join_uses_only_source_key_not_all_shared_dms_keys(self):
        # Regression: when the dimension table (WHS_DMS) also holds OTHER _DMS_KEY
        # columns shared with the fact table (e.g. FCY_DMS_KEY), _join_edges used to
        # include all of them as join conditions.  The validator then required every
        # condition to appear in the SQL — the LLM only wrote WHS_DMS_KEY = WHS_DMS_KEY
        # and failed.  The plan must pin to exactly one condition: source_key_column.
        columns_with_extra_fk = dict(COLUMNS)
        columns_with_extra_fk["PROFITABILITY.WHS_DMS"] = {
            "WHS_DMS_KEY": "int",
            "WHS_CD": "varchar",
            "WHS_DSC": "varchar",
            "FCY_DMS_KEY": "int",   # WHS_DMS also stores a factory FK — this shared
                                    # column must NOT appear as an extra join condition.
        }
        plan = build_semantic_field_plan(
            "Which warehouses have high invoice revenue but low gross profit percentage?",
            columns_with_extra_fk,
        )
        self.assertTrue(plan["enabled"])
        # There must be exactly one join and it must use only WHS_DMS_KEY.
        whs_joins = [e for e in plan["joins"] if "WHS_DMS" in e.get("to", "").upper()]
        self.assertEqual(len(whs_joins), 1)
        conditions = whs_joins[0]["conditions"]
        self.assertEqual(conditions, [("WHS_DMS_KEY", "WHS_DMS_KEY")],
                         f"Expected only WHS_DMS_KEY condition, got: {conditions}")
        # The SQL with only WHS_DMS_KEY join must pass validation.
        sql = (
            "SELECT w.WHS_DSC AS Warehouse, "
            "SUM(c.SOP_CUS_IVC_LIN_AMT) AS TotalRevenue, "
            "SUM(c.SOP_CUS_LIN_GRS_PFT_AMT) AS GrossProfit "
            "FROM [PROFITABILITY].[CUS_ORD_IVC_FCT] c "
            "JOIN [PROFITABILITY].[WHS_DMS] w ON c.WHS_DMS_KEY = w.WHS_DMS_KEY "
            "GROUP BY w.WHS_DSC"
        )
        result = validate_sql_detailed(
            sql,
            KNOWN,
            "azure_sql",
            None,
            columns_with_extra_fk,
            {"semantic_plan": plan},
        )
        self.assertTrue(result.ok, result.reason)


class IntentAndGraphReliabilityTests(unittest.TestCase):
    def test_missing_record_phrases_are_detected(self):
        for question in (
            "customers with no orders",
            "show rows with no matching invoice",
            "items without receipts",
            "products never sold",
            "customers who haven't placed orders this year",
            "suppliers without invoices last month",
            "which invoices are missing",
        ):
            self.assertTrue(analyze_query_intent(question)["wants_missing_records"], question)

    def test_exclusion_phrasing_does_not_trigger_missing_records(self):
        # wants_missing_records HARD-enforces LEFT JOIN + IS NULL in the
        # validator, so exclusion/NULL-filter phrasing must not set it.
        for question in (
            "what is the total sales without tax",
            "total sales, don't include cancelled orders",
            "list invoice amounts with missing due dates",
            "show revenue excluding returns, never mind the discounts",
            "what is the number of days present between payments by each customer",
        ):
            self.assertFalse(analyze_query_intent(question)["wants_missing_records"], question)

    def test_ddl_check_ignores_string_literals_and_comments(self):
        cols = {"EMDW_DMART.AUDIT_LOG": {"ACTION_TYPE": "varchar"}}
        ok, _, code = validate_sql(
            "SELECT COUNT(*) FROM EMDW_DMART.AUDIT_LOG WHERE ACTION_TYPE = 'DELETE'",
            {"EMDW_DMART.AUDIT_LOG"}, "azure_sql", table_columns=cols,
        )
        self.assertTrue(ok, code)
        ok2, _, code2 = validate_sql(
            "SELECT 1 -- UPDATE nothing\nFROM EMDW_DMART.AUDIT_LOG",
            {"EMDW_DMART.AUDIT_LOG"}, "azure_sql", table_columns=cols,
        )
        self.assertTrue(ok2, code2)
        ok3, _, code3 = validate_sql(
            "DELETE FROM EMDW_DMART.AUDIT_LOG", {"EMDW_DMART.AUDIT_LOG"}, "azure_sql",
        )
        self.assertFalse(ok3)
        self.assertEqual(code3, "ddl")

    def test_date_part_dimension_fields_are_optional(self):
        # "by month"/"in year N" may be answered by bucketing the fact table's
        # YYYYMMDD key directly (per the DATE-KEY RULE) — DT_DMS.MONTH/YEAR
        # must be a hint, not a hard requirement.
        cols = {
            "EMDW_DMART.DT_DMS": {"DT_DMS_KEY": "int", "YEAR": "int", "MONTH": "int", "DAY": "int"},
            "EMDW_DMART.CUS_ORD_IVC_FCT": {
                "CUS_IVC_LIN_AMT": "decimal", "CUS_ORD_DT_DMS_KEY": "int", "CUS_DMS_KEY": "int",
            },
        }
        plan = build_semantic_field_plan("total sales by month", cols, None)
        month_fields = [f for f in plan["fields"] if f["column"] == "MONTH"]
        self.assertTrue(month_fields)
        self.assertEqual(month_fields[0].get("enforcement"), "optional")
        for edge in plan.get("joins") or []:
            if edge["to"].endswith("DT_DMS") or edge["from"].endswith("DT_DMS"):
                self.assertEqual(edge.get("enforcement"), "optional")
        sql = (
            "SELECT FORMAT(TRY_CONVERT(date, CONVERT(varchar(8), CUS_ORD_DT_DMS_KEY), 112), 'yyyy-MM') AS PERIOD, "
            "SUM(CUS_IVC_LIN_AMT) AS TOTAL FROM EMDW_DMART.CUS_ORD_IVC_FCT WHERE CUS_ORD_DT_DMS_KEY > 0 "
            "GROUP BY FORMAT(TRY_CONVERT(date, CONVERT(varchar(8), CUS_ORD_DT_DMS_KEY), 112), 'yyyy-MM')"
        )
        ok, reason, _ = validate_sql(
            sql, set(cols), "azure_sql", table_columns=cols,
            semantic_context={"semantic_plan": plan},
        )
        self.assertTrue(ok, reason)

    def _graph(self):
        return {
            "entities": [
                {"entity_name": "Order", "table_name": "FACT_ORDER", "schema_name": "dbo", "entity_type": "fact", "display_name": "Orders"},
                {"entity_name": "Product", "table_name": "DIM_PRODUCT", "schema_name": "dbo", "entity_type": "dimension", "display_name": "Product"},
                {"entity_name": "Customer", "table_name": "DIM_CUSTOMER", "schema_name": "dbo", "entity_type": "dimension", "display_name": "Customer"},
            ],
            "relationships": [
                {"from_entity": "Order", "to_entity": "Product", "from_column": "ProductID", "to_column": "ProductID", "join_type": "INNER"},
                {"from_entity": "Order", "to_entity": "Customer", "from_column": "CustomerID", "to_column": "CustomerID", "join_type": "INNER"},
            ],
            "properties": [
                {"entity_name": "Product", "column_name": "ProductName", "display_name": "Item Description", "synonyms": "item, items, products"},
            ],
        }

    def test_plural_and_property_matching_detects_product(self):
        found = detect_entities("compare sales by products and item description", self._graph())
        self.assertIn("Product", found)

    def test_metric_formula_table_forces_fact_entity(self):
        found = detect_entities(
            "compare gross margin year over year by product",
            self._graph(),
            required_tables={"FACT_ORDER"},
        )
        self.assertIn("Order", found)
        self.assertIn("Product", found)

    def test_entity_filter_with_between_survives_and_split(self):
        from core.graph_resolver import build_join_skeleton
        entities_map = {
            "Fact": {
                "table_name": "FNN_FCT", "schema_name": "EMDW_DMART",
                "entity_filter": "PAY_DT_DMS_KEY BETWEEN 20240101 AND 20241231",
            },
            "Customer": {"table_name": "CUS_DMS", "schema_name": "EMDW_DMART"},
        }
        path = [{
            "from_entity": "Fact", "to_entity": "Customer",
            "from_column": "CUS_DMS_KEY", "to_column": "CUS_DMS_KEY",
            "join_type": "INNER", "_direction": "forward",
        }]
        skeleton = build_join_skeleton(path, entities_map, "Fact", "azure_sql")
        self.assertIn("BETWEEN 20240101 AND 20241231", skeleton)
        self.assertNotIn("fac.20241231", skeleton)

    def test_split_and_conditions_respects_between_and_parens(self):
        from core.graph_resolver import _split_and_conditions
        self.assertEqual(
            _split_and_conditions("A BETWEEN 1 AND 5 AND B = 2"),
            ["A BETWEEN 1 AND 5", "B = 2"],
        )
        self.assertEqual(
            _split_and_conditions("(X = 1 OR Y = 2) AND Z BETWEEN 3 AND 4"),
            ["(X = 1 OR Y = 2)", "Z BETWEEN 3 AND 4"],
        )

    def test_antijoin_resolver_uses_left_join(self):
        result = resolve_for_question(
            "customers with no orders",
            "acct",
            "azure_sql",
            graph=self._graph(),
            intent={"wants_missing_records": True},
        )
        self.assertTrue(result["enabled"])
        self.assertTrue(result["anti_join"])
        self.assertIn("LEFT", result["join_skeleton"])

    def test_graph_prompt_allows_skeleton_inside_cte(self):
        prompt = build_sql_system_prompt(
            "azure_sql",
            "KB",
            graph_context={
                "enabled": True,
                "join_skeleton": "FROM [dbo].[FACT_ORDER] ord INNER JOIN [dbo].[DIM_PRODUCT] pro ON ord.[ProductID] = pro.[ProductID]",
                "detected": ["Order", "Product"],
            },
        )
        self.assertIn("inside the base CTE", prompt)

    def test_antijoin_prompt_keeps_left_join(self):
        prompt = build_sql_system_prompt(
            "azure_sql",
            "KB",
            graph_context={
                "enabled": True,
                "anti_join": True,
                "join_skeleton": "FROM [dbo].[DIM_CUSTOMER] cus LEFT JOIN [dbo].[FACT_ORDER] ord ON cus.[CustomerID] = ord.[CustomerID]",
                "detected": ["Customer", "Order"],
            },
        )
        self.assertIn("ANTI-JOIN GRAPH MODE", prompt)
        self.assertIn("Do not convert these joins back to INNER JOIN", prompt)

    def test_date_key_rule_forbids_dateadd_on_raw_key(self):
        prompt = build_sql_system_prompt(
            "azure_sql",
            "Table CUS_ORD_IVC_FCT has column CUS_ORD_DT_DMS_KEY",
        )
        self.assertIn("AZURE SQL DATE-KEY RULE", prompt)
        self.assertIn("DATEADD()", prompt)
        self.assertIn("OVERRIDES the CRITICAL TIME RULE", prompt)
        self.assertIn(
            "TRY_CONVERT(date, CONVERT(varchar(8), alias.DATE_KEY_COL), 112) >= "
            "DATEADD(month, -1,",
            prompt,
        )
        self.assertIn("Azure SQL date-key pattern", prompt)
        self.assertIn("WITH dated AS", prompt)
        self.assertIn("GROUP BY YR", prompt)
        self.assertIn("Never write CAST(YEAR(TRY_CONVERT", prompt)

    def test_date_key_rule_absent_without_dms_key_column(self):
        prompt = build_sql_system_prompt("azure_sql", "Table ORDERS has column ORDER_DATE")
        self.assertNotIn("AZURE SQL DATE-KEY RULE", prompt)

    def test_avg_interval_rule_present_for_all_dialects(self):
        for db_type in ("azure_sql", "oracle", "snowflake"):
            prompt = build_sql_system_prompt(db_type, "Table FNN_FCT")
            self.assertIn(
                "AVG INTERVAL BETWEEN EVENTS RULE", prompt, f"missing for {db_type}"
            )
            self.assertIn(
                "LAG(EVENT_DATE) OVER (PARTITION BY group_col ORDER BY EVENT_DATE)",
                prompt,
                f"missing nested-subquery pattern for {db_type}",
            )

    def test_avg_interval_rule_forbids_flat_window_plus_group_by(self):
        prompt = build_sql_system_prompt("azure_sql", "Table FNN_FCT")
        self.assertIn("CANNOT be combined with an outer aggregate/GROUP BY", prompt)
        self.assertIn("not contained in either an aggregate function or the GROUP BY", prompt)

    def test_avg_interval_rule_requires_conversion_inside_inner_subquery(self):
        prompt = build_sql_system_prompt("azure_sql", "Table FNN_FCT has column PAY_DT_DMS_KEY")
        self.assertIn(
            "convert it with TRY_CONVERT inside the INNER subquery", prompt
        )

    def test_period_comparison_prompt_uses_staged_native_date_pattern(self):
        prompt = build_sql_system_prompt(
            "azure_sql",
            "PHARMA_LAB.F_RX_FILL joins PHARMA_LAB.D_DATE; CALENDAR_DATE date",
        )
        self.assertIn("WITH period_totals AS", prompt)
        self.assertIn("period_comparison AS", prompt)
        self.assertIn("Never write LAG(SUM(...))", prompt)
        self.assertIn("Never convert an already-native date", prompt)
        self.assertIn("DATE ROLE IS NOT STATUS", prompt)


class FieldPlanRepairTests(unittest.TestCase):
    """Deterministic display-field repair — no LLM retry for mechanical fixes."""

    _COLS = {
        "EMDW_DMART.CUS_DMS": {"CUS_DMS_KEY": "int", "CUS_NM": "varchar", "CUS_ID": "varchar"},
        "EMDW_DMART.FNN_FCT": {"CUS_DMS_KEY": "int", "PAY_DT_DMS_KEY": "int", "PAY_AMT": "decimal"},
    }

    def _plan(self):
        return {
            "enabled": True,
            "fields": [{
                "term": "customer", "table": "EMDW_DMART.CUS_DMS", "column": "CUS_NM",
                "role": "display_dimension", "display_required": True,
                "source_key_column": "CUS_DMS_KEY", "source_key_table": "EMDW_DMART.FNN_FCT",
            }],
            "joins": [{
                "from": "EMDW_DMART.FNN_FCT", "to": "EMDW_DMART.CUS_DMS",
                "conditions": [("CUS_DMS_KEY", "CUS_DMS_KEY")],
            }],
        }

    def _repair(self, sql):
        from core.pipeline_helpers import attempt_field_plan_repair
        return attempt_field_plan_repair(
            sql, "azure_sql", set(self._COLS), None, self._COLS,
            {"semantic_plan": self._plan()},
        )

    def test_repairs_key_grouping_by_adding_join_and_display_column(self):
        sql = (
            "SELECT CUS_DMS_KEY, SUM(PAY_AMT) AS TOTAL FROM EMDW_DMART.FNN_FCT "
            "WHERE PAY_DT_DMS_KEY > 0 GROUP BY CUS_DMS_KEY"
        )
        fixed = self._repair(sql)
        self.assertTrue(fixed)
        self.assertIn("CUS_NM", fixed)
        self.assertIn("JOIN EMDW_DMART.CUS_DMS", fixed)
        ok, reason, _ = validate_sql(
            fixed, set(self._COLS), "azure_sql", table_columns=self._COLS,
            semantic_context={"semantic_plan": self._plan()},
        )
        self.assertTrue(ok, reason)

    def test_repairs_when_dim_already_joined(self):
        sql = (
            "SELECT f.CUS_DMS_KEY, SUM(f.PAY_AMT) AS TOTAL FROM EMDW_DMART.FNN_FCT f "
            "JOIN EMDW_DMART.CUS_DMS c ON f.CUS_DMS_KEY = c.CUS_DMS_KEY GROUP BY f.CUS_DMS_KEY"
        )
        fixed = self._repair(sql)
        self.assertTrue(fixed)
        self.assertIn("c.CUS_NM", fixed)
        # The JOIN ON condition must keep the surrogate key untouched.
        self.assertIn("f.CUS_DMS_KEY = c.CUS_DMS_KEY", fixed)

    def test_bails_on_valid_sql(self):
        sql = (
            "SELECT c.CUS_NM, SUM(f.PAY_AMT) AS TOTAL FROM EMDW_DMART.FNN_FCT f "
            "JOIN EMDW_DMART.CUS_DMS c ON f.CUS_DMS_KEY = c.CUS_DMS_KEY GROUP BY c.CUS_NM"
        )
        self.assertEqual(self._repair(sql), "")

    def test_bails_when_key_not_projected(self):
        # Adding a display column here would change the query grain.
        sql = "SELECT SUM(PAY_AMT) AS TOTAL FROM EMDW_DMART.FNN_FCT"
        self.assertEqual(self._repair(sql), "")

    def test_pipeline_wires_repair_before_llm_retry(self):
        import inspect
        import core.query_pipeline as qp
        src = inspect.getsource(qp)
        self.assertIn("attempt_field_plan_repair(", src)
        # Repair must run before the retryable/LLM-retry block.
        self.assertLess(
            src.index("attempt_field_plan_repair("),
            src.index("retryable = ("),
        )


class DiagnosticRenderingReliabilityTests(unittest.TestCase):
    def test_zero_row_message_fences_sql_with_underscored_columns(self):
        from core.pipeline_helpers import _build_zero_row_message

        sql = (
            "SELECT c.CUS_ORD_NUM, c.CUS_ORD_LIN_NUM "
            "FROM [profitability].[CUS_ORD_IVC_FCT] c "
            "JOIN [profitability].[OOLINE] o ON c.CUS_ORD_NUM = o.ORNO"
        )
        message = _build_zero_row_message("test", sql, {}, "ok", 0)
        self.assertIn("```sql", message)
        self.assertIn("c.CUS_ORD_NUM", message)
        self.assertIn("c.CUS_ORD_LIN_NUM", message)

    def test_chat_formatter_preserves_sql_code_blocks_before_markdown(self):
        template = (ROOT / "portal" / "templates" / "portal_chat.html").read_text(encoding="utf-8")
        self.assertIn("const codeBlocks = [];", template)
        self.assertIn("return `@@CODEBLOCK${idx}@@`;", template)
        self.assertNotIn("h = h.replace(/_([^_\\n]+)_/g", template)


class BusinessConfidenceRcaTests(unittest.TestCase):
    def test_zero_row_empty_table_lowers_confidence(self):
        confidence = build_answer_confidence(
            validation_code="ok",
            row_count=0,
            retry_count=0,
            has_semantic_plan=True,
            has_graph_context=True,
            tables_used=["PROFITABILITY.CUS_ORD_IVC_FCT", "PROFITABILITY.OOLINE"],
            empty_tables=["PROFITABILITY.OOLINE"],
        )
        self.assertEqual(confidence["level"], "low")
        self.assertTrue(any("no records" in w for w in confidence["warnings"]))

    def test_zero_row_rca_names_empty_source_table(self):
        rca = build_business_rca(
            question="division share by item group",
            row_count=0,
            tables_used=["PROFITABILITY.CUS_ORD_IVC_FCT", "PROFITABILITY.OOLINE"],
            empty_tables=["PROFITABILITY.OOLINE"],
            validation_code="ok",
            retry_count=0,
        )
        self.assertIn("no records", rca["most_likely_reason"])
        self.assertIn("PROFITABILITY.OOLINE", rca["most_likely_reason"])

    def test_successful_query_gets_high_confidence_summary(self):
        confidence = build_answer_confidence(
            validation_code="ok",
            row_count=42,
            retry_count=0,
            has_semantic_plan=True,
            has_graph_context=False,
            tables_used=["PROFITABILITY.CUS_ORD_IVC_FCT"],
        )
        self.assertEqual(confidence["level"], "high")
        self.assertGreaterEqual(confidence["score"], 80)

    def test_null_metric_issue_lowers_confidence(self):
        confidence = build_answer_confidence(
            validation_code="ok",
            row_count=1,
            retry_count=0,
            tables_used=["PROFITABILITY.CUS_ORD_IVC_FCT"],
            null_metric_issue=True,
        )
        self.assertEqual(confidence["level"], "medium")
        self.assertTrue(any("metric values were null" in w for w in confidence["warnings"]))

    def test_null_metric_issue_gets_business_readable_answer(self):
        rows = [{"MatchedRows": 64, "NonNullRevenueRows": 0, "Revenue": 0}]
        self.assertIsNotNone(detect_null_metric_issue(rows))
        payload = build_assistant_response(
            question="what is the revenue for customer 1055930",
            rows=rows,
            sql="SELECT COUNT_BIG(*) AS [MatchedRows], COUNT(CUS_IVC_LIN_AMT) AS [NonNullRevenueRows], COALESCE(SUM(CUS_IVC_LIN_AMT), 0) AS [Revenue] FROM [PROFITABILITY].[CUS_ORD_IVC_FCT] WHERE CUS_DMS_KEY = 1055930",
            duration_ms=20,
        )
        self.assertIn("all matched values are missing", payload["answer"]["headline"])
        self.assertIn("64 matching records", payload["answer"]["comparison"])
        self.assertIn("64 records matched", payload["insight_summary"])

    def test_extract_sql_tables_preserves_schema_names(self):
        sql = (
            "SELECT o.DIVI, c.ITM_GRP_DMS_KEY "
            "FROM [profitability].[CUS_ORD_IVC_FCT] c "
            "JOIN [profitability].[OOLINE] o ON c.CUS_ORD_NUM = o.ORNO"
        )
        tables = extract_sql_tables(sql, "azure_sql")
        self.assertIn("PROFITABILITY.CUS_ORD_IVC_FCT", tables)
        self.assertIn("PROFITABILITY.OOLINE", tables)


class TopNSemanticsTests(unittest.TestCase):
    KNOWN = {
        "PHARMACY.FACT_PRESCRIPTION_FILL",
        "FACT_PRESCRIPTION_FILL",
        "PHARMACY.DIM_PRESCRIBER",
        "DIM_PRESCRIBER",
    }

    def _validate(self, sql: str, question: str):
        intent = detect_top_n_intent(question)
        self.assertIsNotNone(intent)
        return validate_sql_detailed(
            sql,
            self.KNOWN,
            "azure_sql",
            semantic_context={"top_n": intent.to_dict()},
        )

    def test_detects_awkward_top_two_wording(self):
        intent = detect_top_n_intent(
            "find the doctors who have prescribed more prescriptions the top 2"
        )
        self.assertIsNotNone(intent)
        self.assertEqual(intent.limit, 2)
        self.assertEqual(intent.direction, "descending")
        self.assertEqual(intent.tie_policy, "exactly_n")

    def test_detects_word_number_and_explicit_ties(self):
        intent = detect_top_n_intent("show top two doctors and include all ties")
        self.assertIsNotNone(intent)
        self.assertEqual(intent.limit, 2)
        self.assertEqual(intent.tie_policy, "include_ties")

    def test_accepts_final_top_two(self):
        sql = """
        WITH PrescriberCounts AS (
            SELECT dp.Prescriber_ID, dp.Full_Name AS Doctor_Name,
                   COUNT(fpf.Rx_Fill_ID) AS Total_Prescriptions
            FROM PHARMACY.FACT_Prescription_Fill fpf
            JOIN PHARMACY.DIM_Prescriber dp
              ON fpf.Prescriber_ID = dp.Prescriber_ID
            GROUP BY dp.Prescriber_ID, dp.Full_Name
        )
        SELECT TOP (2) Doctor_Name, Total_Prescriptions
        FROM PrescriberCounts
        ORDER BY Total_Prescriptions DESC, Doctor_Name
        """
        result = self._validate(sql, "show the top 2 doctors by prescriptions")
        self.assertTrue(result.ok, result.reason)

    def test_rejects_top_n_with_reversed_order(self):
        sql = """
        SELECT TOP (2) dp.Full_Name AS Doctor_Name,
               COUNT(fpf.Rx_Fill_ID) AS Total_Prescriptions
        FROM PHARMACY.FACT_Prescription_Fill fpf
        JOIN PHARMACY.DIM_Prescriber dp
          ON fpf.Prescriber_ID = dp.Prescriber_ID
        GROUP BY dp.Full_Name
        ORDER BY Total_Prescriptions ASC, Doctor_Name
        """
        result = self._validate(sql, "show the top 2 doctors by prescriptions")
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "top_n_shape")

    def test_accepts_bottom_n_with_ascending_order(self):
        sql = """
        SELECT TOP (2) dp.Full_Name AS Doctor_Name,
               COUNT(fpf.Rx_Fill_ID) AS Total_Prescriptions
        FROM PHARMACY.FACT_Prescription_Fill fpf
        JOIN PHARMACY.DIM_Prescriber dp
          ON fpf.Prescriber_ID = dp.Prescriber_ID
        GROUP BY dp.Full_Name
        ORDER BY Total_Prescriptions ASC, Doctor_Name
        """
        result = self._validate(sql, "show the bottom 2 doctors by prescriptions")
        self.assertTrue(result.ok, result.reason)

    def test_rejects_top_n_converted_to_threshold(self):
        sql = """
        WITH PrescriberCounts AS (
            SELECT dp.Prescriber_ID, dp.Full_Name AS Doctor_Name,
                   COUNT(fpf.Rx_Fill_ID) AS Total_Prescriptions
            FROM PHARMACY.FACT_Prescription_Fill fpf
            JOIN PHARMACY.DIM_Prescriber dp
              ON fpf.Prescriber_ID = dp.Prescriber_ID
            GROUP BY dp.Prescriber_ID, dp.Full_Name
        ), Top2 AS (
            SELECT TOP 2 Total_Prescriptions
            FROM PrescriberCounts
            ORDER BY Total_Prescriptions DESC
        )
        SELECT pc.Doctor_Name, pc.Total_Prescriptions
        FROM PrescriberCounts pc
        WHERE pc.Total_Prescriptions > (
            SELECT MIN(Total_Prescriptions) FROM Top2
        )
        """
        result = self._validate(
            sql,
            "find the doctors who have prescribed more prescriptions the top 2",
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "top_n_shape")

    def test_exact_top_n_rejects_rank_tie_expansion(self):
        sql = """
        WITH Ranked AS (
            SELECT dp.Full_Name AS Doctor_Name,
                   RANK() OVER (ORDER BY COUNT(fpf.Rx_Fill_ID) DESC) AS rnk
            FROM PHARMACY.FACT_Prescription_Fill fpf
            JOIN PHARMACY.DIM_Prescriber dp
              ON fpf.Prescriber_ID = dp.Prescriber_ID
            GROUP BY dp.Full_Name
        )
        SELECT Doctor_Name FROM Ranked WHERE rnk <= 2
        """
        result = self._validate(sql, "show the top 2 doctors")
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "top_n_shape")

    def test_explicit_ties_accepts_rank(self):
        sql = """
        WITH Ranked AS (
            SELECT dp.Full_Name AS Doctor_Name,
                   RANK() OVER (ORDER BY COUNT(fpf.Rx_Fill_ID) DESC) AS rnk
            FROM PHARMACY.FACT_Prescription_Fill fpf
            JOIN PHARMACY.DIM_Prescriber dp
              ON fpf.Prescriber_ID = dp.Prescriber_ID
            GROUP BY dp.Full_Name
        )
        SELECT Doctor_Name FROM Ranked WHERE rnk <= 2
        """
        result = self._validate(sql, "show the top 2 doctors including all ties")
        self.assertTrue(result.ok, result.reason)


class ProductionSqlGuardTests(unittest.TestCase):
    KNOWN = {"S.T", "T", "S.U", "U"}

    def _validate(self, sql: str, db_type: str = "azure_sql", question: str = "show the results"):
        return validate_sql_detailed(
            sql,
            self.KNOWN,
            db_type,
            semantic_context={"production_sql": True, "question": question},
        )

    def test_rejects_limit_for_azure_sql(self):
        result = self._validate("SELECT A FROM S.T LIMIT 10")
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "dialect_mismatch")

    def test_rejects_top_for_oracle(self):
        result = self._validate("SELECT TOP 10 A FROM S.T", "oracle")
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "dialect_mismatch")

    def test_rejects_tsql_try_convert_for_snowflake(self):
        result = self._validate("SELECT TRY_CONVERT(date, A) FROM S.T", "snowflake")
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "dialect_mismatch")

    def test_rejects_projection_wildcards_but_allows_count_star(self):
        wildcard = self._validate("SELECT t.* FROM S.T t")
        self.assertFalse(wildcard.ok)
        self.assertEqual(wildcard.code, "production_shape")

        count = self._validate("SELECT COUNT(*) AS TotalRows FROM S.T")
        self.assertTrue(count.ok, count.reason)

    def test_rejects_join_without_condition(self):
        result = self._validate("SELECT t.A FROM S.T t JOIN S.U u")
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "production_shape")
        self.assertEqual(result.errors[0]["code"], "missing_join_condition")

    def test_cross_join_requires_explicit_user_intent(self):
        rejected = self._validate("SELECT t.A FROM S.T t CROSS JOIN S.U u")
        self.assertFalse(rejected.ok)
        self.assertEqual(rejected.code, "production_shape")

        allowed = self._validate(
            "SELECT t.A FROM S.T t CROSS JOIN S.U u",
            question="show all possible combinations from T and U",
        )
        self.assertTrue(allowed.ok, allowed.reason)

    def test_allows_cross_join_to_inline_scalar_aggregate(self):
        result = self._validate(
            "SELECT t.A, b.Benchmark "
            "FROM S.T t CROSS JOIN (SELECT AVG(u.A) AS Benchmark FROM S.U u) b"
        )
        self.assertTrue(result.ok, result.reason)

    def test_allows_cross_join_to_scalar_aggregate_cte(self):
        result = self._validate(
            "WITH benchmark AS (SELECT AVG(u.A) AS Benchmark FROM S.U u) "
            "SELECT t.A, b.Benchmark FROM S.T t CROSS JOIN benchmark b"
        )
        self.assertTrue(result.ok, result.reason)

    def test_rejects_cross_join_to_grouped_benchmark(self):
        result = self._validate(
            "SELECT t.A, b.Benchmark "
            "FROM S.T t CROSS JOIN "
            "(SELECT u.A, AVG(u.A) AS Benchmark FROM S.U u GROUP BY u.A) b"
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.errors[0]["code"], "cartesian_join")

    def test_rejects_cross_join_to_windowed_multirow_relation(self):
        result = self._validate(
            "SELECT t.A, b.Benchmark "
            "FROM S.T t CROSS JOIN "
            "(SELECT AVG(u.A) OVER () AS Benchmark FROM S.U u) b"
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.errors[0]["code"], "cartesian_join")

    def test_rejects_multiple_statements(self):
        result = self._validate("SELECT A FROM S.T; SELECT A FROM S.U")
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "multi_statement")

    def test_rejects_select_into_write(self):
        result = self._validate("SELECT A INTO S.NEW_TABLE FROM S.T")
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "ddl")


class PeriodComparisonSqlTests(unittest.TestCase):
    TABLES = {
        "CHATBOT_DB.PHARMA_LAB.F_RX_FILL": {
            "BOOKED_DATE_ID": "int",
            "NET_REVENUE_AMT": "decimal",
            "FILL_STATUS": "varchar",
        },
        "CHATBOT_DB.PHARMA_LAB.D_DATE": {
            "DATE_ID": "int",
            "CALENDAR_DATE": "date",
            "CALENDAR_YEAR": "int",
        },
    }
    QUESTION = "What was net revenue by booked month in 2025, and how did it change month over month?"

    def _validate(self, sql: str):
        return validate_sql_detailed(
            sql,
            set(self.TABLES),
            "azure_sql",
            table_columns=self.TABLES,
            semantic_context={"production_sql": True, "question": self.QUESTION},
        )

    def test_parse_error_preserves_exact_location_and_token(self):
        malformed = """
        SELECT ROUND((SUM(f.NET_REVENUE_AMT) -
            LAG(SUM(f.NET_REVENUE_AMT)) OVER (ORDER BY d.CALENDAR_DATE) * 100.0 /
            NULLIF(LAG(SUM(f.NET_REVENUE_AMT)) OVER (ORDER BY d.CALENDAR_DATE), 0), 2) AS PCT_CHANGE
        FROM CHATBOT_DB.PHARMA_LAB.F_RX_FILL f
        JOIN CHATBOT_DB.PHARMA_LAB.D_DATE d ON f.BOOKED_DATE_ID = d.DATE_ID
        GROUP BY d.CALENDAR_DATE
        """
        result = self._validate(malformed)
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "parse")
        self.assertTrue(result.errors[0]["line"])
        self.assertEqual(result.errors[0]["token"].upper(), "AS")
        self.assertIn("Expecting )", result.reason)

    def test_rejects_lag_wrapped_aggregate_for_period_comparison(self):
        fragile = """
        SELECT d.CALENDAR_DATE AS PERIOD,
               SUM(f.NET_REVENUE_AMT) AS METRIC,
               LAG(SUM(f.NET_REVENUE_AMT)) OVER (ORDER BY d.CALENDAR_DATE) AS PREV_METRIC
        FROM CHATBOT_DB.PHARMA_LAB.F_RX_FILL f
        JOIN CHATBOT_DB.PHARMA_LAB.D_DATE d ON f.BOOKED_DATE_ID = d.DATE_ID
        GROUP BY d.CALENDAR_DATE
        """
        result = self._validate(fragile)
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "period_comparison_shape")

    def test_accepts_staged_booked_date_month_comparison(self):
        staged = """
        WITH period_totals AS (
            SELECT DATEFROMPARTS(YEAR(d.CALENDAR_DATE), MONTH(d.CALENDAR_DATE), 1) AS PERIOD,
                   SUM(f.NET_REVENUE_AMT) AS METRIC
            FROM CHATBOT_DB.PHARMA_LAB.F_RX_FILL f
            JOIN CHATBOT_DB.PHARMA_LAB.D_DATE d ON f.BOOKED_DATE_ID = d.DATE_ID
            WHERE d.CALENDAR_DATE >= DATEFROMPARTS(2025, 1, 1)
              AND d.CALENDAR_DATE < DATEFROMPARTS(2026, 1, 1)
            GROUP BY DATEFROMPARTS(YEAR(d.CALENDAR_DATE), MONTH(d.CALENDAR_DATE), 1)
        ), period_comparison AS (
            SELECT PERIOD, METRIC,
                   LAG(METRIC) OVER (ORDER BY PERIOD) AS PREV_METRIC
            FROM period_totals
        )
        SELECT PERIOD, METRIC, PREV_METRIC,
               METRIC - PREV_METRIC AS DIFF,
               ROUND((METRIC - PREV_METRIC) * 100.0 / NULLIF(PREV_METRIC, 0), 2) AS PCT_CHANGE
        FROM period_comparison
        ORDER BY PERIOD
        """
        result = self._validate(staged)
        self.assertTrue(result.ok, result.reason)


class ResultTransformationRoutingTests(unittest.TestCase):
    def test_main_chat_routes_flag_followup_to_cached_result(self):
        self.assertTrue(
            should_route_to_result_cache(
                "for each unique warehouse id flag them Warehouse A and display the revenue",
                True,
                ["Warehouse", "TotalRevenue"],
            )
        )

    def test_duckdb_prompt_prefers_chr_for_letter_flags(self):
        prompt = build_duckdb_system_prompt(
            [{"name": "Warehouse", "type": "BIGINT"}, {"name": "TotalRevenue", "type": "DOUBLE"}],
        )
        self.assertIn("chr(CAST(64 +", prompt)
        self.assertIn("return the original useful columns PLUS the new computed column", prompt)


if __name__ == "__main__":
    unittest.main()
