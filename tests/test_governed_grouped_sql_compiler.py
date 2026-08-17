import unittest

from core.pipeline_helpers import compile_governed_temporal_metric_sql
from core.validator import validate_sql_detailed


TABLE_COLUMNS = {
    "OPS.F_SALES": {
        "SALES_LINE_SK": "bigint",
        "ORDER_NUMBER": "varchar",
        "INVOICE_DATE_SK": "int",
        "WAREHOUSE_SK": "int",
        "CUSTOMER_SK": "int",
        "NET_REVENUE_AMOUNT": "decimal",
        "GROSS_MARGIN_AMOUNT": "decimal",
    },
    "OPS.D_DATE": {"DATE_SK": "int", "FULL_DATE": "date"},
    "OPS.D_WAREHOUSE": {"WAREHOUSE_SK": "int", "WAREHOUSE_NAME": "varchar"},
    "OPS.D_CUSTOMER": {"CUSTOMER_SK": "int", "CUSTOMER_NAME": "varchar"},
}


def _date_policy():
    return {
        "kind": "last_n",
        "amount": 2,
        "unit": "month",
        "requested_grain": "month",
        "fact_table": "OPS.F_SALES",
        "fact_column": "INVOICE_DATE_SK",
        "dimension_table": "OPS.D_DATE",
        "dimension_key": "DATE_SK",
        "date_column": "FULL_DATE",
        "date_key_type": "surrogate_fk",
        "role_alias": "invoice_date",
    }


def _joins():
    return [
        {
            "from": "OPS.F_SALES",
            "to": "OPS.D_DATE",
            "conditions": [["INVOICE_DATE_SK", "DATE_SK"]],
            "enforcement": "required",
        },
        {
            "from": "OPS.F_SALES",
            "to": "OPS.D_WAREHOUSE",
            "conditions": [["WAREHOUSE_SK", "WAREHOUSE_SK"]],
            "enforcement": "required",
        },
        {
            "from": "OPS.F_SALES",
            "to": "OPS.D_CUSTOMER",
            "conditions": [["CUSTOMER_SK", "CUSTOMER_SK"]],
            "enforcement": "required",
        },
    ]


class GovernedGroupedSqlCompilerTests(unittest.TestCase):
    def _compile(self, context):
        return compile_governed_temporal_metric_sql(
            "azure_sql",
            set(TABLE_COLUMNS),
            set(TABLE_COLUMNS),
            TABLE_COLUMNS,
            context,
        )

    def test_top_warehouses_by_revenue_uses_exact_metric_fact_and_join(self):
        context = {
            "question": "top 10 warehouses by revenue in the last 2 months",
            "top_n": {"limit": 10, "direction": "descending", "tie_policy": "exactly_n"},
            "metric_formulas": [{
                "name": "Revenue",
                "formula_type": "expression",
                "sql_template": "SUM(NET_REVENUE_AMOUNT)",
                "base_table": "OPS.F_SALES",
            }],
            "semantic_plan": {
                "fields": [
                    {
                        "term": "Revenue", "table": "OPS.F_SALES",
                        "column": "NET_REVENUE_AMOUNT", "role": "measure",
                    },
                    {
                        "term": "Warehouse", "table": "OPS.D_WAREHOUSE",
                        "column": "WAREHOUSE_NAME", "role": "display_dimension",
                        "display_required": True,
                    },
                ],
                "joins": _joins(),
                "temporal_policies": [_date_policy()],
            },
            "analytical_request_plan": {
                "status": "compiled",
                "intent": "ranking",
                "source_fact": "OPS.F_SALES",
                "source_facts": ["OPS.F_SALES"],
                "top_n": 10,
                "output_shape": "table",
            },
        }

        sql = self._compile(context)

        self.assertTrue(sql)
        self.assertIn("TOP (10)", sql)
        self.assertIn("SUM(NET_REVENUE_AMOUNT) AS REVENUE", sql)
        self.assertIn("business_dimension.[WAREHOUSE_NAME] AS WAREHOUSE", sql)
        self.assertIn(
            "fact_rows.[WAREHOUSE_SK] = business_dimension.[WAREHOUSE_SK]", sql,
        )
        self.assertIn("fact_rows.[INVOICE_DATE_SK] = invoice_date.[DATE_SK]", sql)
        self.assertIn("ORDER BY REVENUE DESC", sql)
        self.assertNotIn("F_INVENTORY", sql)
        result = validate_sql_detailed(
            sql, set(TABLE_COLUMNS), "azure_sql", set(TABLE_COLUMNS), TABLE_COLUMNS, context,
        )
        self.assertTrue(result.ok, result.reason)

    def test_total_orders_by_customer_counts_exact_business_identifier(self):
        context = {
            "question": "top 10 customers by total orders in the last 2 months",
            "top_n": {"limit": 10, "direction": "descending", "tie_policy": "exactly_n"},
            "metric_formulas": [],
            "semantic_plan": {
                "fields": [{
                    "term": "Customer", "table": "OPS.D_CUSTOMER",
                    "column": "CUSTOMER_NAME", "role": "display_dimension",
                    "display_required": True,
                }],
                "joins": _joins(),
                "temporal_policies": [_date_policy()],
            },
            "analytical_request_plan": {
                "status": "compiled",
                "intent": "ranking",
                "source_fact": "OPS.F_SALES",
                "source_facts": ["OPS.F_SALES"],
                "top_n": 10,
                "output_shape": "table",
                "derived_measure": {
                    "semantics": "count_distinct_business_identifier",
                    "business_entity": "order",
                    "target_table": "OPS.F_SALES",
                    "target_column": "ORDER_NUMBER",
                },
            },
        }

        sql = self._compile(context)

        self.assertTrue(sql)
        self.assertIn("COUNT(DISTINCT fact_rows.[ORDER_NUMBER]) AS ORDER_COUNT", sql)
        self.assertIn("business_dimension.[CUSTOMER_NAME] AS CUSTOMER", sql)
        self.assertNotIn("NET_REVENUE_AMOUNT", sql)
        result = validate_sql_detailed(
            sql, set(TABLE_COLUMNS), "azure_sql", set(TABLE_COLUMNS), TABLE_COLUMNS, context,
        )
        self.assertTrue(result.ok, result.reason)

    def test_same_fact_multi_metric_trend_is_compiled_without_fact_fanout(self):
        context = {
            "question": "show revenue and gross margin trend for the last 2 months",
            "metric_formulas": [
                {
                    "name": "Revenue", "formula_type": "expression",
                    "sql_template": "SUM(NET_REVENUE_AMOUNT)", "base_table": "OPS.F_SALES",
                },
                {
                    "name": "Gross Margin", "formula_type": "expression",
                    "sql_template": "SUM(GROSS_MARGIN_AMOUNT)", "base_table": "OPS.F_SALES",
                },
            ],
            "semantic_plan": {
                "fields": [], "joins": _joins(),
                "temporal_policies": [_date_policy()],
            },
            "analytical_request_plan": {
                "status": "compiled", "intent": "trend",
                "source_fact": "OPS.F_SALES", "source_facts": ["OPS.F_SALES"],
                "output_shape": "time_series",
            },
        }

        sql = self._compile(context)

        self.assertTrue(sql)
        self.assertIn("SUM(NET_REVENUE_AMOUNT) AS REVENUE", sql)
        self.assertIn("SUM(GROSS_MARGIN_AMOUNT) AS GROSS_MARGIN", sql)
        self.assertEqual(sql.upper().count("OPS].[F_SALES"), 2)  # anchor and result only
        self.assertNotIn("JOIN [OPS].[F_SALES]", sql.upper())

    def test_reduced_orders_compiles_two_equal_governed_windows(self):
        context = {
            "question": "which customers have reduced orders in the last 2 months",
            "metric_formulas": [],
            "semantic_plan": {
                "fields": [{
                    "term": "Customer", "table": "OPS.D_CUSTOMER",
                    "column": "CUSTOMER_NAME", "role": "display_dimension",
                    "display_required": True,
                }],
                "joins": _joins(),
                "temporal_policies": [_date_policy()],
            },
            "analytical_request_plan": {
                "status": "compiled",
                "intent": "entity_lookup",
                "source_fact": "OPS.F_SALES",
                "source_facts": ["OPS.F_SALES"],
                "derived_measure": {
                    "semantics": "count_distinct_business_identifier",
                    "business_entity": "order",
                    "target_table": "OPS.F_SALES",
                    "target_column": "ORDER_NUMBER",
                },
                "analytical_recipe": {
                    "kind": "period_over_period_entity_change",
                    "direction": "decrease",
                },
            },
        }

        sql = self._compile(context)

        self.assertTrue(sql)
        self.assertIn("COUNT(DISTINCT CASE", sql)
        self.assertIn("DATEADD(month, -2, anchor.max_business_date)", sql)
        self.assertIn("DATEADD(month, -4, anchor.max_business_date)", sql)
        self.assertIn("CURRENT_PERIOD_COUNT < PRIOR_PERIOD_COUNT", sql)
        self.assertIn("PERCENTAGE_CHANGE", sql)
        self.assertNotIn("NET_REVENUE_AMOUNT", sql)
        result = validate_sql_detailed(
            sql, set(TABLE_COLUMNS), "azure_sql", set(TABLE_COLUMNS), TABLE_COLUMNS, context,
        )
        self.assertTrue(result.ok, result.reason)

    def test_missing_dimension_join_refuses_to_guess(self):
        context = {
            "question": "top 10 warehouses by revenue",
            "metric_formulas": [{
                "name": "Revenue", "formula_type": "expression",
                "sql_template": "SUM(NET_REVENUE_AMOUNT)", "base_table": "OPS.F_SALES",
            }],
            "semantic_plan": {
                "fields": [{
                    "term": "Warehouse", "table": "OPS.D_WAREHOUSE",
                    "column": "WAREHOUSE_NAME", "role": "display_dimension",
                    "display_required": True,
                }],
                "joins": [],
                "temporal_policies": [],
            },
            "analytical_request_plan": {
                "status": "compiled", "intent": "ranking",
                "source_fact": "OPS.F_SALES", "source_facts": ["OPS.F_SALES"],
                "top_n": 10,
            },
        }
        self.assertEqual(self._compile(context), "")

    def test_same_bare_table_name_in_another_schema_is_not_accepted(self):
        context = {
            "question": "top warehouses by revenue",
            "metric_formulas": [{
                "name": "Revenue", "formula_type": "expression",
                "sql_template": "SUM(NET_REVENUE_AMOUNT)", "base_table": "OTHER.F_SALES",
            }],
            "semantic_plan": {
                "fields": [{
                    "term": "Warehouse", "table": "OPS.D_WAREHOUSE",
                    "column": "WAREHOUSE_NAME", "role": "display_dimension",
                    "display_required": True,
                }],
                "joins": _joins(),
                "temporal_policies": [],
            },
            "analytical_request_plan": {
                "status": "compiled", "intent": "ranking",
                "source_fact": "OPS.F_SALES", "source_facts": ["OPS.F_SALES"],
                "top_n": 10,
            },
        }

        self.assertEqual(self._compile(context), "")

    def test_existing_scalar_metric_path_remains_unchanged(self):
        context = {
            "question": "total revenue for the last 2 months",
            "metric_formulas": [{
                "name": "Revenue", "formula_type": "expression",
                "sql_template": "SUM(NET_REVENUE_AMOUNT)", "base_table": "OPS.F_SALES",
            }],
            "semantic_plan": {"fields": [], "joins": _joins(), "temporal_policies": [_date_policy()]},
            "analytical_request_plan": {
                "status": "compiled", "intent": "metric_query",
                "source_fact": "OPS.F_SALES", "source_facts": ["OPS.F_SALES"],
                "output_shape": "kpi",
            },
        }
        sql = self._compile(context)
        self.assertTrue(sql)
        self.assertNotIn("TOP (", sql)
        self.assertNotIn("GROUP BY", sql)
        self.assertIn("SUM(NET_REVENUE_AMOUNT) AS REVENUE", sql)

    def test_latest_two_observed_data_days_are_selected_from_fact_rows(self):
        policy = _date_policy()
        policy.update({
            "kind": "latest_n_observed",
            "amount": 2,
            "unit": "day",
            "requested_grain": "day",
            "anchor_policy": "observed_periods",
        })
        context = {
            "question": "show revenue for the latest 2 data days",
            "metric_formulas": [{
                "name": "Revenue", "formula_type": "expression",
                "sql_template": "SUM(NET_REVENUE_AMOUNT)", "base_table": "OPS.F_SALES",
            }],
            "semantic_plan": {
                "fields": [], "joins": _joins(), "temporal_policies": [policy],
            },
            "analytical_request_plan": {
                "status": "compiled", "intent": "trend",
                "source_fact": "OPS.F_SALES", "source_facts": ["OPS.F_SALES"],
                "output_shape": "time_series",
                "temporal_operations": [policy],
            },
        }

        sql = self._compile(context)

        self.assertTrue(sql)
        self.assertIn("SELECT DISTINCT TOP (2) invoice_date.[FULL_DATE]", sql)
        self.assertIn("FROM [OPS].[F_SALES] AS fact_rows", sql)
        self.assertIn(
            "invoice_date.[FULL_DATE] IN (SELECT observed_business_date FROM observed_periods)",
            sql,
        )
        self.assertIn("GROUP BY CAST(invoice_date.[FULL_DATE] AS date)", sql)
        self.assertIn("ORDER BY PERIOD", sql)
        self.assertNotIn("DATEADD(day, -2", sql)
        result = validate_sql_detailed(
            sql, set(TABLE_COLUMNS), "azure_sql", set(TABLE_COLUMNS), TABLE_COLUMNS, context,
        )
        self.assertTrue(result.ok, result.reason)


if __name__ == "__main__":
    unittest.main()
