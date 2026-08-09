import unittest

import sqlglot

from core.join_planner import (
    choose_fact_anchor,
    compile_join_plan,
    relationship_is_admissible,
)
from core.table_role_classifier import classify_schema_tables, classify_table
from core.validator import _join_plan_contract_errors


class TestEvidenceBasedTableRoles(unittest.TestCase):
    def test_cryptic_erp_table_is_fact_from_structure_not_name(self):
        meta = {
            "pk_columns": ["ROW_KEY"],
            "columns": [
                {"name": "ROW_KEY", "type": "bigint"},
                {"name": "CUS_KEY", "type": "int"},
                {"name": "PRD_KEY", "type": "int"},
                {"name": "NET_AMT", "type": "decimal"},
                {"name": "ORDER_QTY", "type": "decimal"},
            ],
        }
        fks = [
            {"parent_table": "OOHEAD", "parent_col": "CUS_KEY", "ref_table": "OCUSMA"},
            {"parent_table": "OOHEAD", "parent_col": "PRD_KEY", "ref_table": "MITMAS"},
        ]
        result = classify_table("OOHEAD", meta, declared_fks=fks)
        self.assertEqual(result.role, "fact")
        self.assertGreaterEqual(result.confidence, 70)

    def test_numeric_lookup_is_not_misclassified_as_fact(self):
        meta = {
            "pk_columns": ["CUSTOMER_KEY"],
            "columns": [
                {"name": "CUSTOMER_KEY", "type": "bigint"},
                {"name": "CUSTOMER_CODE", "type": "varchar"},
                {"name": "CUSTOMER_NAME", "type": "varchar"},
                {"name": "CREDIT_LIMIT", "type": "decimal"},
            ],
        }
        result = classify_table("OCUSMA", meta, declared_fks=[
            {"parent_table": "OOHEAD", "ref_table": "OCUSMA"},
        ])
        self.assertEqual(result.role, "dimension")

    def test_bridge_requires_structural_evidence(self):
        meta = {
            "pk_columns": ["PRODUCT_KEY", "CATEGORY_KEY"],
            "columns": [
                {"name": "PRODUCT_KEY", "type": "int"},
                {"name": "CATEGORY_KEY", "type": "int"},
            ],
        }
        fks = [
            {"parent_table": "XPCAT", "parent_col": "PRODUCT_KEY", "ref_table": "PRODUCT"},
            {"parent_table": "XPCAT", "parent_col": "CATEGORY_KEY", "ref_table": "CATEGORY"},
        ]
        result = classify_table("XPCAT", meta, declared_fks=fks)
        self.assertEqual(result.role, "bridge")

    def test_calendar_table_is_date_dimension(self):
        result = classify_table("CALENDAR", {
            "pk_columns": ["DATE_KEY"],
            "columns": [
                {"name": "DATE_KEY", "type": "int"},
                {"name": "FULL_DATE", "type": "date"},
                {"name": "YEAR_NUMBER", "type": "int"},
                {"name": "MONTH_NUMBER", "type": "int"},
                {"name": "DAY_NUMBER", "type": "int"},
            ],
        })
        self.assertEqual(result.role, "date_dimension")

    def test_schema_classifier_uses_fk_direction(self):
        schema = {
            "dbo.TX": {
                "pk_columns": ["ID"],
                "columns": [
                    {"name": "ID", "type": "bigint"},
                    {"name": "CUSTOMER_KEY", "type": "int"},
                    {"name": "SALES_AMT", "type": "decimal"},
                    {"name": "SALES_QTY", "type": "decimal"},
                ],
            },
            "dbo.CUSTOMER": {
                "pk_columns": ["CUSTOMER_KEY"],
                "columns": [
                    {"name": "CUSTOMER_KEY", "type": "int"},
                    {"name": "CUSTOMER_NAME", "type": "varchar"},
                ],
            },
            "__db_fk_constraints__": [{
                "parent_table": "TX", "parent_col": "CUSTOMER_KEY",
                "ref_table": "CUSTOMER", "ref_col": "CUSTOMER_KEY",
            }],
        }
        roles = classify_schema_tables(schema)
        self.assertEqual(roles["dbo.TX"].role, "fact")
        self.assertEqual(roles["dbo.CUSTOMER"].role, "dimension")


class TestJoinPlanCompilation(unittest.TestCase):
    def setUp(self):
        self.graph = {
            "entities": [
                {"entity_name": "Sales", "table_name": "FACT_SALES", "entity_type": "fact", "status": "confirmed"},
                {"entity_name": "Inventory", "table_name": "FACT_INVENTORY", "entity_type": "fact", "status": "confirmed"},
                {"entity_name": "Warehouse", "table_name": "DIM_WAREHOUSE", "entity_type": "dimension", "status": "confirmed"},
                {"entity_name": "Region", "table_name": "DIM_REGION", "entity_type": "dimension", "status": "confirmed"},
                {"entity_name": "ProductCategory", "table_name": "BRIDGE_PRODUCT_CATEGORY", "entity_type": "bridge", "status": "confirmed"},
            ],
            "relationships": [],
        }

    def test_metric_source_deterministically_selects_anchor(self):
        anchor = choose_fact_anchor(
            self.graph["entities"], ["Inventory", "Sales"],
            metric_formula_tables=["FACT_SALES"],
        )
        self.assertEqual(anchor, "Sales")

    def test_direct_fact_to_fact_is_blocked(self):
        entities = {e["entity_name"]: e for e in self.graph["entities"]}
        allowed, reason = relationship_is_admissible({
            "from_entity": "Sales", "to_entity": "Inventory",
            "relationship_type": "many_to_one",
        }, entities)
        self.assertFalse(allowed)
        self.assertIn("fact-to-fact", reason)

    def test_governed_fact_existence_anti_join_is_narrow_exception(self):
        path = [{
            "id": 9,
            "from_entity": "Sales",
            "to_entity": "Inventory",
            "relationship_type": "one_to_many",
            "_direction": "forward",
        }]
        normal = compile_join_plan(self.graph, ["Sales", "Inventory"], path)
        anti = compile_join_plan(
            self.graph, ["Sales", "Inventory"], path, anti_join=True,
        )
        self.assertEqual(normal["status"], "blocked")
        self.assertEqual(anti["status"], "selected")
        self.assertIn("existence", anti["warnings"][0])

    def test_snowflake_path_is_selected(self):
        path = [
            {"id": 1, "from_entity": "Sales", "to_entity": "Warehouse", "relationship_type": "many_to_one"},
            {"id": 2, "from_entity": "Warehouse", "to_entity": "Region", "relationship_type": "many_to_one"},
        ]
        plan = compile_join_plan(self.graph, ["Sales", "Warehouse", "Region"], path)
        self.assertEqual(plan["status"], "selected")
        self.assertEqual(plan["anchor_fact"], "Sales")
        self.assertEqual(plan["required_edge_ids"], [1, 2])

    def test_shared_dimension_multi_fact_requires_isolation(self):
        path = [
            {"id": 1, "from_entity": "Sales", "to_entity": "Warehouse", "relationship_type": "many_to_one"},
            {"id": 2, "from_entity": "Inventory", "to_entity": "Warehouse", "relationship_type": "many_to_one"},
        ]
        plan = compile_join_plan(
            self.graph, ["Sales", "Inventory", "Warehouse"], path,
            metric_formula_tables=["FACT_SALES"],
        )
        self.assertEqual(plan["status"], "requires_isolated_aggregation")
        self.assertEqual(plan["common_dimensions"], ["Warehouse"])
        self.assertFalse(plan["raw_fact_to_fact_allowed"])
        self.assertEqual(len(plan["isolated_fact_plans"]), 2)

    def test_bridge_allocation_is_discovered_from_confirmed_business_metadata(self):
        graph = dict(self.graph)
        graph["properties"] = [{
            "entity_name": "ProductCategory",
            "column_name": "CATEGORY_SHARE",
            "display_name": "Revenue allocation percentage",
            "status": "confirmed",
        }]
        path = [{
            "id": 3, "from_entity": "Sales", "to_entity": "ProductCategory",
            "relationship_type": "many_to_one",
        }]
        plan = compile_join_plan(graph, ["Sales", "ProductCategory"], path)
        self.assertEqual(plan["bridge_allocations"][0]["column"], "CATEGORY_SHARE")


class TestJoinPlanSqlContract(unittest.TestCase):
    def setUp(self):
        self.context = {
            "enabled": True,
            "entities": [
                {"entity_name": "Sales", "table_name": "FACT_SALES", "entity_type": "fact"},
                {"entity_name": "Inventory", "table_name": "FACT_INVENTORY", "entity_type": "fact"},
                {"entity_name": "Warehouse", "table_name": "DIM_WAREHOUSE", "entity_type": "dimension"},
            ],
            "join_plan": {
                "status": "requires_isolated_aggregation",
                "fact_entities": ["Sales", "Inventory"],
            },
        }

    def _errors(self, sql):
        tree = sqlglot.parse_one(sql, read="tsql")
        return _join_plan_contract_errors(tree, self.context)

    def test_raw_multi_fact_select_is_rejected(self):
        errors = self._errors("""
            SELECT SUM(s.REVENUE_AMOUNT), SUM(i.ON_HAND_VALUE)
            FROM dbo.FACT_SALES s
            JOIN dbo.DIM_WAREHOUSE w ON s.WAREHOUSE_KEY = w.WAREHOUSE_KEY
            JOIN dbo.FACT_INVENTORY i ON i.WAREHOUSE_KEY = w.WAREHOUSE_KEY
        """)
        self.assertEqual(errors[0]["code"], "raw_fact_to_fact_join")

    def test_isolated_fact_ctes_are_accepted(self):
        errors = self._errors("""
            WITH sales_agg AS (
                SELECT WAREHOUSE_KEY, SUM(REVENUE_AMOUNT) AS REVENUE
                FROM dbo.FACT_SALES GROUP BY WAREHOUSE_KEY
            ), inventory_agg AS (
                SELECT WAREHOUSE_KEY, SUM(ON_HAND_VALUE) AS INVENTORY_VALUE
                FROM dbo.FACT_INVENTORY GROUP BY WAREHOUSE_KEY
            )
            SELECT s.WAREHOUSE_KEY, s.REVENUE, i.INVENTORY_VALUE
            FROM sales_agg s JOIN inventory_agg i ON s.WAREHOUSE_KEY = i.WAREHOUSE_KEY
        """)
        self.assertEqual(errors, [])

    def test_unresolved_plan_rejects_any_sql(self):
        context = {"join_plan": {"status": "blocked", "reason": "missing relationship"}}
        tree = sqlglot.parse_one("SELECT 1", read="tsql")
        errors = _join_plan_contract_errors(tree, context)
        self.assertEqual(errors[0]["code"], "join_plan_unresolved")

    def test_additive_bridge_measure_requires_governed_allocation(self):
        context = {
            "entities": [
                {"entity_name": "Sales", "table_name": "FACT_SALES", "entity_type": "fact"},
                {"entity_name": "CategoryBridge", "table_name": "BRIDGE_CATEGORY", "entity_type": "bridge"},
            ],
            "join_plan": {
                "status": "selected",
                "fact_entities": ["Sales"],
                "bridge_entities": ["CategoryBridge"],
                "bridge_allocations": [{"entity": "CategoryBridge", "table": "BRIDGE_CATEGORY", "column": "ALLOCATION_PCT"}],
            },
        }
        missing = _join_plan_contract_errors(sqlglot.parse_one(
            "SELECT SUM(s.REVENUE) FROM FACT_SALES s JOIN BRIDGE_CATEGORY b ON s.PRODUCT_KEY=b.PRODUCT_KEY",
            read="tsql",
        ), context)
        self.assertEqual(missing[0]["code"], "bridge_allocation_missing")

        accepted = _join_plan_contract_errors(sqlglot.parse_one(
            "SELECT SUM(s.REVENUE * b.ALLOCATION_PCT) FROM FACT_SALES s JOIN BRIDGE_CATEGORY b ON s.PRODUCT_KEY=b.PRODUCT_KEY",
            read="tsql",
        ), context)
        self.assertEqual(accepted, [])

    def test_bridge_without_allocation_mapping_fails_closed_for_sum(self):
        context = {
            "entities": [
                {"entity_name": "Sales", "table_name": "FACT_SALES", "entity_type": "fact"},
                {"entity_name": "CategoryBridge", "table_name": "BRIDGE_CATEGORY", "entity_type": "bridge"},
            ],
            "join_plan": {
                "status": "selected", "fact_entities": ["Sales"],
                "bridge_entities": ["CategoryBridge"], "bridge_allocations": [],
            },
        }
        errors = _join_plan_contract_errors(sqlglot.parse_one(
            "SELECT SUM(s.REVENUE) FROM FACT_SALES s JOIN BRIDGE_CATEGORY b ON s.PRODUCT_KEY=b.PRODUCT_KEY",
            read="tsql",
        ), context)
        self.assertEqual(errors[0]["code"], "bridge_allocation_unresolved")


if __name__ == "__main__":
    unittest.main()
