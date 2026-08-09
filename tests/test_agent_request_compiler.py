import unittest

from core.analytical_request_plan import compile_analytical_request_plan
from core.contextual_dates import build_contextual_date_plan, detect_temporal_window
from core.semantic_planner import build_semantic_field_plan, format_semantic_field_plan
from core.source_resolution import resolve_source_scope, source_clarification_options
from core.validator import validate_sql_detailed
from core.vocab_packs import MergedVocab
from core.response_builder import _build_decision_signal
from core.result_verifier import verify_result_shape


def _model(schema="ACME_OPS", m3_name="M3_MITBAL"):
    return {
        "tables": [
            {
                "qualified_name": f"{schema}.F_STOCK_DAY",
                "schema": schema,
                "type": "fact",
                "entity": "stock snapshot",
                "grain": "one row per product warehouse day",
                "fact_type": "periodic snapshot",
                "fields": [
                    {"column": "STOCK_VALUE", "role": "measure", "expanded_name": "Stock Value",
                     "business_candidates": ["inventory value", "stock"]},
                ],
                "measures": [],
            },
            {
                "qualified_name": f"{schema}.F_STOCK_MONTH",
                "schema": schema,
                "type": "fact",
                "entity": "stock snapshot",
                "grain": "one row per product warehouse month",
                "fact_type": "periodic snapshot",
                "fields": [
                    {"column": "ENDING_STOCK_VALUE", "role": "measure", "expanded_name": "Ending Stock Value",
                     "business_candidates": ["inventory value", "month end inventory"]},
                ],
                "measures": [],
            },
            {
                "qualified_name": f"{schema}.{m3_name}",
                "schema": schema,
                "type": "fact",
                "entity": "item balance",
                "grain": "one row per item warehouse period",
                "fact_type": "periodic snapshot",
                "fields": [
                    {"column": "MLAVAL", "role": "measure", "expanded_name": "Inventory Value",
                     "business_candidates": ["inventory value", "item balance value"]},
                ],
                "measures": [],
            },
        ]
    }


class DynamicSourceResolutionTests(unittest.TestCase):
    def test_requested_grain_selects_daily_fact_without_dataset_names(self):
        result = resolve_source_scope(
            "show daily inventory by warehouse", _model(schema="CLIENT_X")
        )
        self.assertEqual(result["status"], "selected")
        self.assertEqual(result["selected_fact"], "CLIENT_X.F_STOCK_DAY")

    def test_month_end_selects_monthly_fact(self):
        result = resolve_source_scope(
            "show month inventory by warehouse", _model(schema="TENANT_BLUE")
        )
        self.assertEqual(result["status"], "selected")
        self.assertEqual(result["selected_fact"], "TENANT_BLUE.F_STOCK_MONTH")

    def test_pack_source_alias_and_measure_can_be_separated_in_sentence(self):
        vocab = MergedVocab(table_dict={
            "M3_MITBAL": {
                "label": "M3 Item Balance",
                "synonyms": ["M3 balance data", "item balance data"],
                "type": "fact",
            }
        })
        result = resolve_source_scope(
            "using the M3 balance data, give me inventory value by warehouse",
            _model(schema="ERP_LAKE"), vocab=vocab,
        )
        self.assertEqual(result["status"], "selected")
        self.assertEqual(result["selected_fact"], "ERP_LAKE.M3_MITBAL")

    def test_bare_business_concept_is_derived_from_measure_suffix(self):
        columns = {
            "CLIENT_A.F_DAILY": {"INVENTORY_VALUE": "decimal", "WAREHOUSE_SK": "int"},
            "CLIENT_A.F_OTHER": {"ORDER_AMOUNT": "decimal"},
        }
        plan = build_semantic_field_plan(
            "show inventory", columns,
            fact_tables={"CLIENT_A.F_DAILY", "CLIENT_A.F_OTHER"},
            preferred_fact_tables={"CLIENT_A.F_DAILY"},
        )
        self.assertTrue(plan["enabled"])
        self.assertEqual(plan["fields"][0]["column"], "INVENTORY_VALUE")

    def test_ambiguous_sources_offer_business_labels_not_fqns(self):
        result = resolve_source_scope("show inventory by warehouse", _model(schema="TENANT_Z"))
        self.assertEqual(result["status"], "ambiguous")
        options = source_clarification_options(result)
        self.assertGreaterEqual(len(options), 2)
        self.assertTrue(all("TENANT_Z." not in option["label"] for option in options))
        self.assertTrue(all(option["value"].startswith("TENANT_Z.") for option in options))


class ExecutableRequestPlanTests(unittest.TestCase):
    def test_compiler_preserves_single_fact_and_output_shape(self):
        semantic = {
            "enabled": True,
            "source_scope": {"selected_fact": "S.F_SALES"},
            "fields": [{"term": "revenue", "table": "S.F_SALES", "column": "NET_AMOUNT", "role": "measure"}],
            "joins": [],
            "temporal_policies": [{"kind": "latest_n_observed", "amount": 2, "unit": "day"}],
        }
        plan = compile_analytical_request_plan("last 2 data days", semantic)
        self.assertEqual(plan["source_fact"], "S.F_SALES")
        self.assertEqual(plan["output_shape"], "time_series")
        self.assertIn("no_direct_fact_to_fact_join", plan["prohibitions"])

    def test_compound_request_compiles_one_isolated_subplan_per_fact(self):
        semantic = {
            "enabled": True,
            "source_scope": {"selected_fact": "OPS.F_SALES"},
            "fields": [
                {"term": "revenue", "table": "OPS.F_SALES", "column": "NET_AMOUNT", "role": "measure"},
                {"term": "inventory", "table": "OPS.F_INVENTORY", "column": "STOCK_VALUE", "role": "measure"},
                {"term": "warehouse", "table": "OPS.D_WAREHOUSE", "column": "WAREHOUSE_NAME", "role": "dimension"},
            ],
        }
        graph = {
            "entities": [
                {"entity_name": "Sales", "schema_name": "OPS", "table_name": "F_SALES"},
                {"entity_name": "Inventory", "schema_name": "OPS", "table_name": "F_INVENTORY"},
            ],
            "join_plan": {
                "status": "requires_isolated_aggregation",
                "common_dimensions": ["Warehouse"],
                "isolated_fact_plans": [
                    {"fact_entity": "Sales"},
                    {"fact_entity": "Inventory"},
                ],
            },
        }

        plan = compile_analytical_request_plan(
            "compare monthly revenue with daily inventory",
            semantic,
            graph_context=graph,
        )

        self.assertEqual(plan["source_facts"], ["OPS.F_SALES", "OPS.F_INVENTORY"])
        self.assertEqual(len(plan["subrequests"]), 2)
        self.assertTrue(all(item["aggregate_before_join"] for item in plan["subrequests"]))
        self.assertEqual(plan["combination"]["join_inputs"], "aggregated_subplans_only")

    def test_compound_source_contract_requires_every_subplan_fact(self):
        semantic = {
            "analytical_request_plan": {
                "source_fact": "OPS.F_SALES",
                "source_facts": ["OPS.F_SALES", "OPS.F_INVENTORY"],
            },
        }
        result = validate_sql_detailed(
            "SELECT SUM(NET_AMOUNT) FROM OPS.F_SALES",
            {"OPS.F_SALES", "OPS.F_INVENTORY"},
            "mssql",
            None,
            {"OPS.F_SALES": {"NET_AMOUNT": "decimal"}, "OPS.F_INVENTORY": {"STOCK_VALUE": "decimal"}},
            semantic,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "source_fact_mismatch")
        self.assertEqual(result.errors[0]["missing_facts"], ["OPS.F_INVENTORY"])

    def test_latest_data_days_is_not_a_calendar_window(self):
        window = detect_temporal_window("revenue for the last 2 data days")
        self.assertEqual(window["kind"], "latest_n_observed")
        self.assertEqual(window["anchor_policy"], "observed_periods")
        self.assertEqual(window["amount"], 2)

    def test_prompt_demands_distinct_observed_periods(self):
        binding = {
            "fact_table": "S.F_SALES", "fact_column": "INVOICE_DATE_SK",
            "dimension_table": "S.D_DATE", "dimension_key": "DATE_SK",
            "date_value_column": "FULL_DATE", "date_key_type": "surrogate_fk",
            "context_name": "Invoice Date", "governance_status": "approved",
        }
        plan = build_contextual_date_plan(binding, "revenue for the last 2 data days")
        text = format_semantic_field_plan(plan, "azure_sql")
        self.assertIn("latest 2 DISTINCT observed day", text)
        self.assertIn("do not turn this into MAX(date)-N", text)

    def test_source_only_plan_still_reaches_the_sql_prompt(self):
        text = format_semantic_field_plan({
            "enabled": False,
            "fields": [],
            "source_scope": {"selected_fact": "S.F_INVENTORY_DAILY"},
            "analytical_request_plan": {
                "status": "compiled", "source_fact": "S.F_INVENTORY_DAILY",
                "measures": [], "dimensions": [], "output_shape": "table",
            },
        })
        self.assertIn("S.F_INVENTORY_DAILY is the single fact", text)
        self.assertIn("Executable analytical request plan", text)

    def test_validator_accepts_exact_observed_period_shape(self):
        policy = {
            "kind": "latest_n_observed", "amount": 2, "unit": "day",
            "anchor_policy": "observed_periods", "fact_table": "S.F_SALES",
            "fact_column": "INVOICE_DATE_SK", "date_table": "S.D_DATE",
            "date_column": "FULL_DATE", "dimension_table": "S.D_DATE",
            "dimension_key": "DATE_SK", "date_key_type": "surrogate_fk",
        }
        semantic = {
            "semantic_plan": {"enabled": True, "fields": [], "joins": [], "temporal_policies": [policy]},
            "analytical_request_plan": {"source_fact": "S.F_SALES"},
        }
        sql = (
            "WITH observed_days AS ("
            "SELECT DISTINCT TOP (2) d.FULL_DATE FROM S.F_SALES f "
            "JOIN S.D_DATE d ON f.INVOICE_DATE_SK=d.DATE_SK "
            "ORDER BY d.FULL_DATE DESC) "
            "SELECT d.FULL_DATE, SUM(f.NET_AMOUNT) AS REVENUE FROM S.F_SALES f "
            "JOIN S.D_DATE d ON f.INVOICE_DATE_SK=d.DATE_SK "
            "JOIN observed_days o ON o.FULL_DATE=d.FULL_DATE "
            "GROUP BY d.FULL_DATE ORDER BY d.FULL_DATE"
        )
        result = validate_sql_detailed(
            sql, {"S.F_SALES", "S.D_DATE"}, "mssql", None,
            {"S.F_SALES": {"INVOICE_DATE_SK": "int", "NET_AMOUNT": "decimal"},
             "S.D_DATE": {"DATE_SK": "int", "FULL_DATE": "date"}},
            semantic,
        )
        self.assertTrue(result.ok, result.reason)

    def test_validator_rejects_scalar_calendar_arithmetic_for_data_days(self):
        policy = {
            "kind": "latest_n_observed", "amount": 2, "unit": "day",
            "anchor_policy": "observed_periods", "fact_table": "S.F_SALES",
            "fact_column": "INVOICE_DATE_SK", "date_table": "S.D_DATE",
            "date_column": "FULL_DATE", "dimension_table": "S.D_DATE",
            "dimension_key": "DATE_SK", "date_key_type": "surrogate_fk",
        }
        sql = (
            "SELECT SUM(f.NET_AMOUNT) FROM S.F_SALES f JOIN S.D_DATE d "
            "ON f.INVOICE_DATE_SK=d.DATE_SK WHERE d.FULL_DATE >= "
            "DATEADD(day,-2,(SELECT MAX(d2.FULL_DATE) FROM S.D_DATE d2 "
            "JOIN S.F_SALES f2 ON f2.INVOICE_DATE_SK=d2.DATE_SK))"
        )
        result = validate_sql_detailed(
            sql, {"S.F_SALES", "S.D_DATE"}, "mssql", None,
            {"S.F_SALES": {"INVOICE_DATE_SK": "int", "NET_AMOUNT": "decimal"},
             "S.D_DATE": {"DATE_SK": "int", "FULL_DATE": "date"}},
            {"semantic_plan": {"enabled": True, "fields": [], "joins": [], "temporal_policies": [policy]}},
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "observed_period_shape")

    def test_two_points_do_not_claim_a_sustained_trend(self):
        signal = _build_decision_signal(
            {"mode": "time_series", "row_count": 2},
            {"mode": "time_series", "row_count": 2, "time_series": {
                "direction": "decreasing", "overall_pct_change": -35,
                "longest_decline_streak": 1,
            }},
            [],
        )
        self.assertEqual(signal, {})

    def test_post_execution_verifier_rejects_too_many_observed_periods(self):
        report = verify_result_shape(
            [
                {"BUSINESS_DATE": "2026-01-01", "REVENUE": 10},
                {"BUSINESS_DATE": "2026-01-02", "REVENUE": 12},
                {"BUSINESS_DATE": "2026-01-03", "REVENUE": 14},
            ],
            request_plan={"temporal_operations": [{
                "kind": "latest_n_observed", "amount": 2, "unit": "day",
            }]},
        )
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any("latest 2 observed" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
