import unittest

from core.metric_scope import resolve_metric_scope


class MetricScopeTests(unittest.TestCase):
    def setUp(self):
        self.columns = {
            "PROFITABILITY.CUS_ORD_IVC_FCT": {
                "DEL_IVC_REC_IND": "int",
                "SOP_CUS_IVC_LIN_AMT": "decimal",
                "WHS_DMS_KEY": "int",
            },
            "PHARMACY.FACT_PRESCRIPTION_FILL": {
                "Total_Charge_USD": "decimal",
                "PRESCRIBER_KEY": "int",
            },
            "PHARMACY.DIM_PRESCRIBER": {
                "PRESCRIBER_KEY": "int",
                "PRESCRIBER_NAME": "varchar",
            },
        }
        # Metrics with FQN-prefixed required_columns (old path via all_columns)
        self.metrics = [
            {
                "name": "Revenue",
                "synonyms": "sales amount",
                "formula_type": "expression",
                "sql_template": "SUM(CASE WHEN DEL_IVC_REC_IND = 0 THEN SOP_CUS_IVC_LIN_AMT ELSE 0 END)",
                "required_columns": "DEL_IVC_REC_IND,SOP_CUS_IVC_LIN_AMT",
                "base_entity": "Warehouse",          # links to PROFITABILITY entity
            },
            {
                "name": "Total Revenue USD",
                "synonyms": "total charge usd,total revenue",
                "formula_type": "expression",
                "sql_template": "SUM(Total_Charge_USD)",
                "required_columns": "Total_Charge_USD",
                "base_entity": "Prescription Fill",  # links to PHARMACY entity
            },
        ]
        # Metrics without base_entity (bare base_table only — tests fallback path)
        self.metrics_no_entity = [
            {
                "name": "Revenue",
                "synonyms": "sales amount",
                "formula_type": "expression",
                "sql_template": "SUM(CASE WHEN DEL_IVC_REC_IND = 0 THEN SOP_CUS_IVC_LIN_AMT ELSE 0 END)",
                "required_columns": "DEL_IVC_REC_IND,SOP_CUS_IVC_LIN_AMT",
                "base_table": "FACT_PRESCRIPTION_FILL",  # bare table — no schema extractable
            },
            {
                "name": "Total Revenue USD",
                "synonyms": "total charge usd,total revenue",
                "formula_type": "expression",
                "sql_template": "SUM(Total_Charge_USD)",
                "required_columns": "Total_Charge_USD",
                "base_table": "FACT_PRESCRIPTION_FILL",
            },
        ]
        self.entity_schema_map = {
            "Warehouse": "PROFITABILITY",
            "Prescription Fill": "PHARMACY",
            "DIM_Prescriber": "PHARMACY",
        }

    # ── Primary path: base_entity → entity_schema_map ─────────────────────────

    def test_entity_map_filters_metric_for_pharmacy(self):
        """base_entity→entity_schema_map correctly picks the PHARMACY metric."""
        graph = {
            "entities": [
                {"entity_name": "DIM_Prescriber", "schema_name": "PHARMACY", "table_name": "DIM_PRESCRIBER"}
            ]
        }
        result = resolve_metric_scope(
            self.metrics,
            "what is my total revenue usd by each prescriber",
            self.columns,
            graph_context={"enabled": True, "detected": ["DIM_Prescriber"], "anchor": "DIM_Prescriber"},
            graph=graph,
            entity_schema_map=self.entity_schema_map,
        )
        self.assertFalse(result.ambiguous)
        self.assertEqual([m["name"] for m in result.metrics], ["Total Revenue USD"])

    def test_entity_map_filters_metric_for_profitability(self):
        """base_entity→entity_schema_map correctly picks the PROFITABILITY metric."""
        graph = {
            "entities": [
                {"entity_name": "Warehouse", "schema_name": "PROFITABILITY", "table_name": "CUS_ORD_IVC_FCT"}
            ]
        }
        result = resolve_metric_scope(
            self.metrics,
            "what is the total revenue by warehouse",
            self.columns,
            graph_context={"enabled": True, "detected": ["Warehouse"], "anchor": "Warehouse"},
            graph=graph,
            entity_schema_map=self.entity_schema_map,
        )
        self.assertFalse(result.ambiguous)
        self.assertEqual([m["name"] for m in result.metrics], ["Revenue"])

    # ── Fallback path: graph entity schema (no entity_schema_map) ─────────────

    def test_dimension_schema_filters_metric_for_pharmacy(self):
        """Graph-entity schema path (no entity_schema_map) via required_columns."""
        graph = {
            "entities": [
                {"entity_name": "DIM_Prescriber", "schema_name": "PHARMACY", "table_name": "DIM_PRESCRIBER"}
            ]
        }
        result = resolve_metric_scope(
            self.metrics,
            "what is my total revenue usd by each prescriber",
            self.columns,
            graph_context={"enabled": True, "detected": ["DIM_Prescriber"], "anchor": "DIM_Prescriber"},
            graph=graph,
        )
        self.assertFalse(result.ambiguous)
        self.assertEqual([m["name"] for m in result.metrics], ["Total Revenue USD"])

    def test_dimension_schema_filters_metric_for_profitability(self):
        """Graph-entity schema path (no entity_schema_map) via required_columns."""
        graph = {
            "entities": [
                {"entity_name": "Warehouse", "schema_name": "PROFITABILITY", "table_name": "CUS_ORD_IVC_FCT"}
            ]
        }
        result = resolve_metric_scope(
            self.metrics,
            "what is the total revenue by warehouse",
            self.columns,
            graph_context={"enabled": True, "detected": ["Warehouse"], "anchor": "Warehouse"},
            graph=graph,
        )
        self.assertFalse(result.ambiguous)
        self.assertEqual([m["name"] for m in result.metrics], ["Revenue"])

    # ── Ambiguity ──────────────────────────────────────────────────────────────

    def test_bare_revenue_across_schemas_is_ambiguous(self):
        """No context + two cross-schema metrics → ambiguous, ask user."""
        result = resolve_metric_scope(
            self.metrics,
            "what is my revenue",
            self.columns,
            entity_schema_map=self.entity_schema_map,
        )
        self.assertTrue(result.ambiguous)
        self.assertIn("Revenue", result.options)
        self.assertIn("Total Revenue USD", result.options)

    def test_selected_schema_resolves_without_graph(self):
        """Explicit schema selection resolves even without graph context."""
        result = resolve_metric_scope(
            self.metrics,
            "what is total revenue",
            self.columns,
            selected_schema="PHARMACY",
            entity_schema_map=self.entity_schema_map,
        )
        self.assertFalse(result.ambiguous)
        self.assertEqual([m["name"] for m in result.metrics], ["Total Revenue USD"])


if __name__ == "__main__":
    unittest.main()
