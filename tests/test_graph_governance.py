import json
import tempfile
import unittest
from pathlib import Path

from core.graph_resolver import build_join_skeleton, find_join_path
from core.relationship_validator import build_profile_sql
from core.schema import _build_join_map, build_entity_graph_from_schema
from core.validator import validate_sql_detailed


class TestGraphContractValidation(unittest.TestCase):
    def setUp(self):
        self.known = {"DBO.FACT_SALES", "DBO.DIM_PRODUCT"}
        self.columns = {
            "DBO.FACT_SALES": {
                "PRODUCT_ID": "int", "COMPANY_ID": "int", "AMOUNT": "decimal"
            },
            "DBO.DIM_PRODUCT": {
                "PRODUCT_ID": "int", "COMPANY_ID": "int", "PRODUCT_NAME": "varchar"
            },
        }
        self.graph = {
            "enabled": True,
            "resolved_edges": [{
                "id": 42,
                "relationship_key": "DBFK:FK_SALES_PRODUCT",
                "from_schema": "dbo",
                "from_table": "FACT_SALES",
                "to_schema": "dbo",
                "to_table": "DIM_PRODUCT",
                "conditions": [["PRODUCT_ID", "PRODUCT_ID"], ["COMPANY_ID", "COMPANY_ID"]],
                "join_type": "LEFT",
            }],
        }

    def _validate(self, sql):
        return validate_sql_detailed(
            sql,
            self.known,
            "azure_sql",
            self.known,
            self.columns,
            {"graph_context": self.graph},
        )

    def test_exact_composite_left_join_passes(self):
        result = self._validate("""
            SELECT p.PRODUCT_NAME, SUM(f.AMOUNT) AS TOTAL_AMOUNT
            FROM dbo.FACT_SALES f
            LEFT JOIN dbo.DIM_PRODUCT p
              ON f.PRODUCT_ID = p.PRODUCT_ID
             AND f.COMPANY_ID = p.COMPANY_ID
            GROUP BY p.PRODUCT_NAME
        """)
        self.assertTrue(result.ok, result.reason)

    def test_missing_composite_condition_is_blocked(self):
        result = self._validate("""
            SELECT p.PRODUCT_NAME, SUM(f.AMOUNT) AS TOTAL_AMOUNT
            FROM dbo.FACT_SALES f
            LEFT JOIN dbo.DIM_PRODUCT p ON f.PRODUCT_ID = p.PRODUCT_ID
            GROUP BY p.PRODUCT_NAME
        """)
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "graph_plan_mismatch")
        self.assertEqual(result.errors[0]["edge_id"], 42)

    def test_optional_relationship_cannot_be_changed_to_inner(self):
        result = self._validate("""
            SELECT p.PRODUCT_NAME, SUM(f.AMOUNT) AS TOTAL_AMOUNT
            FROM dbo.FACT_SALES f
            INNER JOIN dbo.DIM_PRODUCT p
              ON f.PRODUCT_ID = p.PRODUCT_ID
             AND f.COMPANY_ID = p.COMPANY_ID
            GROUP BY p.PRODUCT_NAME
        """)
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "graph_plan_mismatch")
        self.assertEqual(result.errors[0]["code"], "graph_join_type_mismatch")

    def test_same_table_names_in_another_schema_do_not_satisfy_plan(self):
        known = self.known | {"PHARMACY.FACT_SALES", "PHARMACY.DIM_PRODUCT"}
        columns = dict(self.columns)
        columns["PHARMACY.FACT_SALES"] = self.columns["DBO.FACT_SALES"]
        columns["PHARMACY.DIM_PRODUCT"] = self.columns["DBO.DIM_PRODUCT"]
        result = validate_sql_detailed(
            """
            SELECT p.PRODUCT_NAME, SUM(f.AMOUNT) AS TOTAL_AMOUNT
            FROM pharmacy.FACT_SALES f
            LEFT JOIN pharmacy.DIM_PRODUCT p
              ON f.PRODUCT_ID = p.PRODUCT_ID AND f.COMPANY_ID = p.COMPANY_ID
            GROUP BY p.PRODUCT_NAME
            """,
            known,
            "azure_sql",
            known,
            columns,
            {"graph_context": self.graph},
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "graph_plan_mismatch")


class TestGovernedGraphResolution(unittest.TestCase):
    def test_weighted_path_prefers_governed_edges(self):
        graph = {
            "entities": [
                {"entity_name": "Fact", "entity_type": "fact"},
                {"entity_name": "Bridge", "entity_type": "bridge"},
                {"entity_name": "Customer", "entity_type": "dimension"},
            ],
            "relationships": [
                {
                    "id": 1, "from_entity": "Fact", "to_entity": "Customer",
                    "from_column": "CUSTOMER_ID", "to_column": "CUSTOMER_ID",
                    "generated_by": "llm", "status": "suggested",
                    "validation_status": "warning", "confidence_score": 50,
                },
                {
                    "id": 2, "from_entity": "Fact", "to_entity": "Bridge",
                    "from_column": "BRIDGE_ID", "to_column": "BRIDGE_ID",
                    "generated_by": "db_fk", "source_enforced": 1,
                    "status": "confirmed", "validation_status": "valid",
                    "confidence_score": 100,
                },
                {
                    "id": 3, "from_entity": "Bridge", "to_entity": "Customer",
                    "from_column": "CUSTOMER_ID", "to_column": "CUSTOMER_ID",
                    "generated_by": "db_fk", "source_enforced": 1,
                    "status": "confirmed", "validation_status": "valid",
                    "confidence_score": 100,
                },
            ],
        }
        path = find_join_path(["Fact", "Customer"], graph)
        self.assertEqual([edge["id"] for edge in path], [2, 3])

    def test_composite_join_is_rendered_in_skeleton(self):
        path = [{
            "from_entity": "Fact", "to_entity": "Product",
            "from_column": "PRODUCT_ID", "to_column": "PRODUCT_ID",
            "join_conditions": json.dumps([
                {"from_col": "COMPANY_ID", "to_col": "COMPANY_ID"}
            ]),
            "join_type": "LEFT", "_direction": "forward",
        }]
        entities = {
            "Fact": {"table_name": "FACT_SALES", "schema_name": "dbo"},
            "Product": {"table_name": "DIM_PRODUCT", "schema_name": "dbo"},
        }
        skeleton = build_join_skeleton(path, entities, "Fact", "azure_sql")
        self.assertIn("[PRODUCT_ID]", skeleton)
        self.assertIn("[COMPANY_ID]", skeleton)
        self.assertRegex(skeleton, r"LEFT\s+JOIN")


class TestConstraintImport(unittest.TestCase):
    def test_composite_database_fk_becomes_one_edge(self):
        with tempfile.TemporaryDirectory() as tmp:
            schema = {
                "DBO.FACT_SALES": {
                    "columns": [
                        {"name": "PRODUCT_ID", "type": "int", "nullable": False},
                        {"name": "COMPANY_ID", "type": "int", "nullable": False},
                        {"name": "AMOUNT", "type": "decimal", "nullable": True},
                    ],
                    "pk_columns": [],
                },
                "DBO.DIM_PRODUCT": {
                    "columns": [
                        {"name": "PRODUCT_ID", "type": "int", "nullable": False},
                        {"name": "COMPANY_ID", "type": "int", "nullable": False},
                    ],
                    "pk_columns": ["PRODUCT_ID", "COMPANY_ID"],
                },
                "__db_fk_constraints__": [
                    {
                        "source": "azure_sql", "constraint_name": "FK_SALES_PRODUCT",
                        "parent_schema": "DBO", "parent_table": "FACT_SALES",
                        "parent_col": "PRODUCT_ID", "ref_schema": "DBO",
                        "ref_table": "DIM_PRODUCT", "ref_col": "PRODUCT_ID",
                        "ordinal": 1, "enforced": True,
                    },
                    {
                        "source": "azure_sql", "constraint_name": "FK_SALES_PRODUCT",
                        "parent_schema": "DBO", "parent_table": "FACT_SALES",
                        "parent_col": "COMPANY_ID", "ref_schema": "DBO",
                        "ref_table": "DIM_PRODUCT", "ref_col": "COMPANY_ID",
                        "ordinal": 2, "enforced": True,
                    },
                ],
            }
            Path(tmp, "_schema.json").write_text(json.dumps(schema), encoding="utf-8")
            graph = build_entity_graph_from_schema(tmp)

        db_edges = [r for r in graph["relationships"] if r.get("generated_by") == "db_fk"]
        self.assertEqual(len(db_edges), 1)
        self.assertEqual(db_edges[0]["join_type"], "INNER")
        self.assertEqual(len(db_edges[0]["join_conditions"]), 1)

    def test_profile_sql_checks_composite_match_quality(self):
        rel = {
            "from_column": "PRODUCT_ID", "to_column": "PRODUCT_ID",
            "join_conditions": [{"from_col": "COMPANY_ID", "to_col": "COMPANY_ID"}],
        }
        fact = {"schema_name": "dbo", "table_name": "FACT_SALES"}
        dim = {"schema_name": "dbo", "table_name": "DIM_PRODUCT"}
        sql = build_profile_sql("azure_sql", rel, fact, dim)
        self.assertIn("EXISTS", sql)
        self.assertIn("[PRODUCT_ID]", sql)
        self.assertIn("[COMPANY_ID]", sql)
        self.assertIn("orphan_rows", sql)

    def test_kb_join_map_keeps_composite_constraint_atomic(self):
        master = {
            "DBO.FACT_SALES": {
                "columns": [
                    {"name": "PRODUCT_ID", "nullable": False},
                    {"name": "COMPANY_ID", "nullable": False},
                ],
                "schema": "DBO",
            },
            "DBO.DIM_PRODUCT": {
                "columns": [
                    {"name": "PRODUCT_ID", "nullable": False},
                    {"name": "COMPANY_ID", "nullable": False},
                ],
                "schema": "DBO",
            },
            "__db_fk_constraints__": [
                {
                    "source": "azure_sql", "constraint_name": "FK_SALES_PRODUCT",
                    "parent_schema": "DBO", "parent_table": "FACT_SALES",
                    "parent_col": "PRODUCT_ID", "ref_schema": "DBO",
                    "ref_table": "DIM_PRODUCT", "ref_col": "PRODUCT_ID",
                    "ordinal": 1, "enforced": True,
                },
                {
                    "source": "azure_sql", "constraint_name": "FK_SALES_PRODUCT",
                    "parent_schema": "DBO", "parent_table": "FACT_SALES",
                    "parent_col": "COMPANY_ID", "ref_schema": "DBO",
                    "ref_table": "DIM_PRODUCT", "ref_col": "COMPANY_ID",
                    "ordinal": 2, "enforced": True,
                },
            ],
        }
        join_map = _build_join_map(master)
        self.assertEqual(join_map.count("DB-enforced FK"), 1)
        self.assertIn("[FACT_SALES].[PRODUCT_ID] = [DIM_PRODUCT].[PRODUCT_ID] AND", join_map)
        self.assertIn("[FACT_SALES].[COMPANY_ID] = [DIM_PRODUCT].[COMPANY_ID]", join_map)


if __name__ == "__main__":
    unittest.main()
