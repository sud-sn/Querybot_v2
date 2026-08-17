import json
import tempfile
import unittest
from pathlib import Path

from core.graph_resolver import (
    build_join_skeleton,
    find_join_path,
    resolve_for_question,
)
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

    def test_direct_fact_to_terminal_dimension_cannot_replace_snowflake_path(self):
        known = {"DBO.F_SALES", "DBO.D_WAREHOUSE", "DBO.D_REGION"}
        columns = {
            "DBO.F_SALES": {"WAREHOUSE_SK": "int", "REGION_SK": "int", "AMOUNT": "decimal"},
            "DBO.D_WAREHOUSE": {"WAREHOUSE_SK": "int", "REGION_SK": "int"},
            "DBO.D_REGION": {"REGION_SK": "int", "REGION_NAME": "varchar"},
        }
        graph = {
            "enabled": True,
            "resolved_edges": [
                {
                    "id": 51, "from_schema": "dbo", "from_table": "F_SALES",
                    "to_schema": "dbo", "to_table": "D_WAREHOUSE",
                    "conditions": [["WAREHOUSE_SK", "WAREHOUSE_SK"]],
                    "join_type": "INNER",
                },
                {
                    "id": 52, "from_schema": "dbo", "from_table": "D_WAREHOUSE",
                    "to_schema": "dbo", "to_table": "D_REGION",
                    "conditions": [["REGION_SK", "REGION_SK"]],
                    "join_type": "INNER",
                },
            ],
        }
        result = validate_sql_detailed(
            """
            SELECT r.REGION_NAME, SUM(f.AMOUNT) AS TOTAL_AMOUNT
            FROM dbo.F_SALES f
            JOIN dbo.D_REGION r ON f.REGION_SK = r.REGION_SK
            GROUP BY r.REGION_NAME
            """,
            known, "azure_sql", known, columns, {"graph_context": graph},
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "graph_plan_mismatch")
        missing_edge_ids = {
            error.get("edge_id") for error in result.errors
            if error.get("code") == "graph_join_missing"
        }
        self.assertEqual(missing_edge_ids, {51, 52})


class TestFanoutRiskEdgeExemption(unittest.TestCase):
    """core.llm's fan-out guard tells the model to pre-aggregate a flagged
    fact table into a CTE before joining, rather than joining it directly —
    so the literal table.column = table.column edge legitimately never
    appears in the outer query for that table. The graph-plan validator must
    not reject that restructuring as a plan violation."""

    def setUp(self):
        self.known = {"PHARMA_LAB.F_RX_FILL", "PHARMA_LAB.D_PHARMACY"}
        self.columns = {
            "PHARMA_LAB.F_RX_FILL": {"PHARMACY_ID": "int", "NET_REVENUE_AMT": "decimal"},
            "PHARMA_LAB.D_PHARMACY": {"PHARMACY_ID": "int", "PHARMACY_NAME": "varchar"},
        }
        self.graph = {
            "enabled": True,
            "resolved_edges": [{
                "id": 7,
                "relationship_key": "DBFK:FK_RX_FILL_PHARMACY",
                "from_entity": "F_RX_FILL",
                "to_entity": "D_PHARMACY",
                "from_schema": "PHARMA_LAB", "from_table": "F_RX_FILL",
                "to_schema": "PHARMA_LAB", "to_table": "D_PHARMACY",
                "conditions": [["PHARMACY_ID", "PHARMACY_ID"]],
                "join_type": "INNER",
            }],
        }
        self.cte_sql = """
            WITH rev AS (
                SELECT PHARMACY_ID, SUM(NET_REVENUE_AMT) AS TOTAL_REVENUE
                FROM PHARMA_LAB.F_RX_FILL
                GROUP BY PHARMACY_ID
            )
            SELECT dph.PHARMACY_NAME, rev.TOTAL_REVENUE
            FROM PHARMA_LAB.D_PHARMACY dph
            JOIN rev ON dph.PHARMACY_ID = rev.PHARMACY_ID
        """

    def _validate(self, sql, fanout_risk_facts=None):
        graph = dict(self.graph)
        if fanout_risk_facts is not None:
            graph["fanout_risk_facts"] = fanout_risk_facts
        return validate_sql_detailed(
            sql, self.known, "azure_sql", self.known, self.columns,
            {"graph_context": graph},
        )

    def test_cte_restructuring_blocked_without_exemption(self):
        # Baseline: without the fanout_risk_facts flag, the CTE-based join
        # legitimately doesn't contain the literal edge and is (correctly,
        # pre-fix) rejected — proves the test fixture reproduces the bug.
        result = self._validate(self.cte_sql)
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "graph_plan_mismatch")

    def test_cte_restructuring_passes_when_flagged_as_fanout_risk(self):
        result = self._validate(self.cte_sql, fanout_risk_facts=["F_RX_FILL"])
        self.assertTrue(result.ok, result.reason)

    def test_unflagged_table_is_still_enforced(self):
        # Flagging an unrelated entity must not accidentally exempt this edge.
        result = self._validate(self.cte_sql, fanout_risk_facts=["SOME_OTHER_FACT"])
        self.assertFalse(result.ok)

    def test_direct_join_still_passes_when_flagged(self):
        # Flagging a table doesn't forbid the model from still joining it
        # directly if it chooses to — the exemption only widens what's
        # accepted, it doesn't require the CTE shape.
        result = self._validate(
            """
            SELECT dph.PHARMACY_NAME, SUM(f.NET_REVENUE_AMT) AS TOTAL_REVENUE
            FROM PHARMA_LAB.F_RX_FILL f
            INNER JOIN PHARMA_LAB.D_PHARMACY dph ON f.PHARMACY_ID = dph.PHARMACY_ID
            GROUP BY dph.PHARMACY_NAME
            """,
            fanout_risk_facts=["F_RX_FILL"],
        )
        self.assertTrue(result.ok, result.reason)


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
    def test_trusted_database_fk_and_endpoints_are_immediately_confirmed(self):
        with tempfile.TemporaryDirectory() as tmp:
            schema = {
                "DBO.F_SALES": {
                    "columns": [
                        {"name": "CUSTOMER_SK", "type": "int", "nullable": False},
                        {"name": "AMOUNT", "type": "decimal", "nullable": True},
                    ],
                    "pk_columns": [],
                },
                "DBO.D_CUSTOMER": {
                    "columns": [
                        {"name": "CUSTOMER_SK", "type": "int", "nullable": False},
                    ],
                    "pk_columns": ["CUSTOMER_SK"],
                },
                "__db_fk_constraints__": [{
                    "source": "azure_sql", "constraint_name": "FK_SALES_CUSTOMER",
                    "parent_schema": "DBO", "parent_table": "F_SALES",
                    "parent_col": "CUSTOMER_SK", "ref_schema": "DBO",
                    "ref_table": "D_CUSTOMER", "ref_col": "CUSTOMER_SK",
                    "ordinal": 1, "enforced": True,
                }],
            }
            Path(tmp, "_schema.json").write_text(json.dumps(schema), encoding="utf-8")
            graph = build_entity_graph_from_schema(tmp)

        entities = {entity["entity_name"]: entity for entity in graph["entities"]}
        edge = next(rel for rel in graph["relationships"] if rel.get("source_enforced"))
        self.assertEqual(edge["status"], "confirmed")
        self.assertEqual(entities[edge["from_entity"]]["status"], "confirmed")
        self.assertEqual(entities[edge["to_entity"]]["status"], "confirmed")
        self.assertEqual(entities[edge["from_entity"]]["generated_by"], "db_schema")

    def test_untrusted_database_fk_remains_review_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            schema = {
                "DBO.F_SALES": {
                    "columns": [{"name": "CUSTOMER_SK", "type": "int", "nullable": False}],
                    "pk_columns": [],
                },
                "DBO.D_CUSTOMER": {
                    "columns": [{"name": "CUSTOMER_SK", "type": "int", "nullable": False}],
                    "pk_columns": ["CUSTOMER_SK"],
                },
                "__db_fk_constraints__": [{
                    "source": "azure_sql", "constraint_name": "FK_SALES_CUSTOMER",
                    "parent_schema": "DBO", "parent_table": "F_SALES",
                    "parent_col": "CUSTOMER_SK", "ref_schema": "DBO",
                    "ref_table": "D_CUSTOMER", "ref_col": "CUSTOMER_SK",
                    "ordinal": 1, "enforced": False,
                }],
            }
            Path(tmp, "_schema.json").write_text(json.dumps(schema), encoding="utf-8")
            graph = build_entity_graph_from_schema(tmp)

        edge = next(rel for rel in graph["relationships"] if rel.get("generated_by") == "db_declared_fk")
        self.assertEqual(edge["status"], "suggested")
        self.assertFalse(edge["source_enforced"])

    def test_two_role_playing_fks_to_same_date_dimension_remain_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            schema = {
                "DBO.FACT_REVENUE": {
                    "columns": [
                        {"name": "BOOKED_DT_ID", "type": "int", "nullable": False},
                        {"name": "ORDER_DT_ID", "type": "int", "nullable": False},
                        {"name": "AMOUNT", "type": "decimal", "nullable": True},
                    ],
                    "pk_columns": [],
                },
                "DBO.DIM_DATE": {
                    "columns": [
                        {"name": "DATE_ID", "type": "int", "nullable": False},
                        {"name": "FULL_DATE", "type": "date", "nullable": False},
                    ],
                    "pk_columns": ["DATE_ID"],
                },
                "__db_fk_constraints__": [
                    {
                        "source": "azure_sql", "constraint_name": "FK_REVENUE_BOOKED_DATE",
                        "parent_schema": "DBO", "parent_table": "FACT_REVENUE",
                        "parent_col": "BOOKED_DT_ID", "ref_schema": "DBO",
                        "ref_table": "DIM_DATE", "ref_col": "DATE_ID",
                        "ordinal": 1, "enforced": True,
                    },
                    {
                        "source": "azure_sql", "constraint_name": "FK_REVENUE_ORDER_DATE",
                        "parent_schema": "DBO", "parent_table": "FACT_REVENUE",
                        "parent_col": "ORDER_DT_ID", "ref_schema": "DBO",
                        "ref_table": "DIM_DATE", "ref_col": "DATE_ID",
                        "ordinal": 1, "enforced": True,
                    },
                ],
            }
            Path(tmp, "_schema.json").write_text(json.dumps(schema), encoding="utf-8")
            graph = build_entity_graph_from_schema(tmp)

        edges = [
            edge for edge in graph["relationships"]
            if edge.get("generated_by") == "db_fk"
        ]
        self.assertEqual(len(edges), 2)
        self.assertEqual(
            {edge["from_column"] for edge in edges},
            {"BOOKED_DT_ID", "ORDER_DT_ID"},
        )
        self.assertEqual({edge["to_column"] for edge in edges}, {"DATE_ID"})

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
        self.assertIn("target_distinct_keys", sql)
        self.assertIn("target_duplicate_keys", sql)
        self.assertIn("HAVING COUNT_BIG(1) > 1", sql)

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


class TestProductionGraphPlanning(unittest.TestCase):
    @staticmethod
    def _entity(name, kind, table=None):
        return {
            "entity_name": name,
            "table_name": table or name,
            "schema_name": "dbo",
            "entity_type": kind,
            "display_name": name.replace("_", " "),
            "status": "confirmed",
        }

    @staticmethod
    def _edge(edge_id, source, target, source_col, target_col, label=""):
        return {
            "id": edge_id,
            "from_entity": source,
            "to_entity": target,
            "from_column": source_col,
            "to_column": target_col,
            "relationship_key": f"DBFK:{edge_id}",
            "relationship_type": "many_to_one",
            "join_type": "INNER",
            "label": label,
            "generated_by": "db_fk",
            "source_enforced": 1,
            "status": "confirmed",
            "validation_status": "valid",
            "confidence_score": 100,
        }

    def test_snowflake_path_uses_intermediate_dimension(self):
        graph = {
            "entities": [
                self._entity("F_SALES", "fact"),
                self._entity("D_WAREHOUSE", "dimension"),
                self._entity("D_REGION", "dimension"),
            ],
            "relationships": [
                self._edge(1, "F_SALES", "D_WAREHOUSE", "WAREHOUSE_SK", "WAREHOUSE_SK"),
                self._edge(2, "D_WAREHOUSE", "D_REGION", "REGION_SK", "REGION_SK"),
            ],
            "properties": [],
        }
        result = resolve_for_question(
            "show revenue by region",
            "tenant", "azure_sql", graph=graph,
            required_entities={"F_SALES", "D_REGION"},
            metric_formula_tables={"F_SALES"},
        )
        self.assertTrue(result["enabled"])
        self.assertEqual(result["edge_ids"], [1, 2])
        self.assertIn("[WAREHOUSE_SK]", result["join_skeleton"])
        self.assertIn("[REGION_SK]", result["join_skeleton"])

    def test_unreachable_requested_entity_blocks_partial_plan(self):
        graph = {
            "entities": [
                self._entity("F_SALES", "fact"),
                self._entity("D_WAREHOUSE", "dimension"),
                self._entity("D_REGION", "dimension"),
            ],
            "relationships": [
                self._edge(1, "F_SALES", "D_WAREHOUSE", "WAREHOUSE_SK", "WAREHOUSE_SK"),
            ],
            "properties": [],
        }
        result = resolve_for_question(
            "show revenue by region",
            "tenant", "azure_sql", graph=graph,
            required_entities={"F_SALES", "D_REGION"},
            metric_formula_tables={"F_SALES"},
        )
        self.assertTrue(result["enabled"])
        self.assertEqual(result["planning_status"], "blocked")
        self.assertEqual(result["join_skeleton"], "")
        self.assertIn("D_REGION", result["missing_entities"])

    def test_equal_governed_paths_are_ranked_and_disclosed_not_escalated(self):
        graph = {
            "entities": [
                self._entity("F_SALES", "fact"),
                self._entity("B_DIRECT", "bridge"),
                self._entity("B_CHANNEL", "bridge"),
                self._entity("D_CUSTOMER", "dimension"),
            ],
            "relationships": [
                self._edge(1, "F_SALES", "B_DIRECT", "DIRECT_SK", "DIRECT_SK", "Direct sales"),
                self._edge(2, "B_DIRECT", "D_CUSTOMER", "CUSTOMER_SK", "CUSTOMER_SK", "Direct customer"),
                self._edge(3, "F_SALES", "B_CHANNEL", "CHANNEL_SK", "CHANNEL_SK", "Channel sales"),
                self._edge(4, "B_CHANNEL", "D_CUSTOMER", "CUSTOMER_SK", "CUSTOMER_SK", "Channel customer"),
            ],
            "properties": [],
        }
        result = resolve_for_question(
            "show revenue by customer",
            "tenant", "azure_sql", graph=graph,
            required_entities={"F_SALES", "D_CUSTOMER"},
            metric_formula_tables={"F_SALES"},
        )
        # A near-tie is not an absence of governance. The pathfinder already
        # ranks by risk weight and breaks ties stably, so the answer is
        # deterministic and repeatable — and a business user cannot be expected
        # to choose a physical join path anyway. Asking also dead-ended: the
        # question was posed per ambiguous target, so answering it resolved only
        # the first and the next target failed with the same message.
        #
        # The contract is now "resolve, but never silently": the query proceeds,
        # and what was chosen and what it was chosen over are both reported so
        # the caller can disclose it and the user can redirect.
        self.assertEqual(result["planning_status"], "selected")
        self.assertTrue(result["join_skeleton"])
        ranked = result.get("ranked_relationship") or {}
        self.assertTrue(ranked.get("chosen"), "the chosen relationship must be named")
        self.assertTrue(
            ranked.get("alternatives"),
            "the path not taken must be named so the user can ask for it",
        )
        self.assertEqual(ranked.get("target"), "D_CUSTOMER")

    def test_duplicate_rows_for_same_physical_edge_do_not_create_ambiguity(self):
        graph = {
            "entities": [
                self._entity("F_SALES", "fact"),
                self._entity("D_WAREHOUSE", "dimension"),
            ],
            "relationships": [
                self._edge(
                    101, "F_SALES", "D_WAREHOUSE",
                    "WAREHOUSE_SK", "WAREHOUSE_SK", "Warehouse",
                ),
                self._edge(
                    202, "F_SALES", "D_WAREHOUSE",
                    "WAREHOUSE_SK", "WAREHOUSE_SK", "Warehouse",
                ),
            ],
            "properties": [],
        }

        result = resolve_for_question(
            "show revenue by warehouse",
            "tenant", "azure_sql", graph=graph,
            required_entities={"F_SALES", "D_WAREHOUSE"},
            metric_formula_tables={"F_SALES"},
        )

        self.assertEqual(result["planning_status"], "selected")
        self.assertEqual(len(result["edge_ids"]), 1)

    def test_unrequested_audit_path_loses_to_business_relationship(self):
        graph = {
            "entities": [
                self._entity("F_SALES", "fact"),
                self._entity("B_BUSINESS", "bridge"),
                self._entity("B_AUDIT", "bridge"),
                self._entity("D_WAREHOUSE", "dimension"),
            ],
            "relationships": [
                self._edge(1, "F_SALES", "B_BUSINESS", "WH_SK", "WH_SK", "Warehouse assignment"),
                self._edge(2, "B_BUSINESS", "D_WAREHOUSE", "WH_SK", "WH_SK", "Warehouse"),
                self._edge(3, "F_SALES", "B_AUDIT", "MODIFIED_SK", "MODIFIED_SK", "Last Modified Date"),
                self._edge(4, "B_AUDIT", "D_WAREHOUSE", "WH_SK", "WH_SK", "Last Modified Date"),
            ],
            "properties": [],
        }

        result = resolve_for_question(
            "show revenue by warehouse",
            "tenant", "azure_sql", graph=graph,
            required_entities={"F_SALES", "D_WAREHOUSE"},
            metric_formula_tables={"F_SALES"},
        )

        self.assertEqual(result["planning_status"], "selected")
        self.assertEqual(result["edge_ids"], [1, 2])

    def test_selected_edge_ids_resume_true_path_ambiguity(self):
        graph = {
            "entities": [
                self._entity("F_SALES", "fact"),
                self._entity("B_DIRECT", "bridge"),
                self._entity("B_CHANNEL", "bridge"),
                self._entity("D_CUSTOMER", "dimension"),
            ],
            "relationships": [
                self._edge(1, "F_SALES", "B_DIRECT", "DIRECT_SK", "DIRECT_SK", "Direct sales"),
                self._edge(2, "B_DIRECT", "D_CUSTOMER", "CUSTOMER_SK", "CUSTOMER_SK", "Direct customer"),
                self._edge(3, "F_SALES", "B_CHANNEL", "CHANNEL_SK", "CHANNEL_SK", "Channel sales"),
                self._edge(4, "B_CHANNEL", "D_CUSTOMER", "CUSTOMER_SK", "CUSTOMER_SK", "Channel customer"),
            ],
            "properties": [],
        }

        result = resolve_for_question(
            "show revenue by customer",
            "tenant", "azure_sql", graph=graph,
            required_entities={"F_SALES", "D_CUSTOMER"},
            metric_formula_tables={"F_SALES"},
            selected_edge_ids=[3, 4],
        )

        self.assertEqual(result["planning_status"], "selected")
        self.assertEqual(result["edge_ids"], [3, 4])

    def test_explicit_business_role_selects_the_matching_governed_path(self):
        graph = {
            "entities": [
                self._entity("F_SALES", "fact"),
                self._entity("B_DIRECT", "bridge"),
                self._entity("B_CHANNEL", "bridge"),
                self._entity("D_CUSTOMER", "dimension"),
            ],
            "relationships": [
                self._edge(1, "F_SALES", "B_DIRECT", "DIRECT_SK", "DIRECT_SK", "Direct sales"),
                self._edge(2, "B_DIRECT", "D_CUSTOMER", "CUSTOMER_SK", "CUSTOMER_SK", "Direct customer"),
                self._edge(3, "F_SALES", "B_CHANNEL", "CHANNEL_SK", "CHANNEL_SK", "Channel sales"),
                self._edge(4, "B_CHANNEL", "D_CUSTOMER", "CUSTOMER_SK", "CUSTOMER_SK", "Channel customer"),
            ],
            "properties": [],
        }
        result = resolve_for_question(
            "show revenue by customer using channel sales",
            "tenant", "azure_sql", graph=graph,
            required_entities={"F_SALES", "D_CUSTOMER"},
            metric_formula_tables={"F_SALES"},
        )
        self.assertEqual(result["planning_status"], "selected")
        self.assertEqual(result["edge_ids"], [3, 4])


if __name__ == "__main__":
    unittest.main()
