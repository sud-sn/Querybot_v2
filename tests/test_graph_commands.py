import json
import unittest

from core.graph_commands import (
    build_graph_command_input,
    compile_graph_command_response,
    compile_graph_commands_response,
    parse_explicit_graph_commands,
    parse_graph_command,
)


class GraphCommandCompileTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "DBO.DIM_CUSTOMER": ["CUSTOMER_ID", "CUSTOMER_NAME", "REGION"],
            "DBO.F_ORDERS": ["ORDER_ID", "CUSTOMER_ID", "ORDER_DATE", "TOTAL"],
        }

    def _raw(self, **plan):
        return json.dumps(plan)

    def test_register_entity_happy_path(self):
        command, error = compile_graph_command_response(
            self._raw(action="register_entity", table_name="DBO.DIM_CUSTOMER", confidence=0.95),
            self.manifest,
        )
        self.assertEqual(error, "")
        self.assertEqual(command.action, "register_entity")
        self.assertEqual(command.table_name, "DBO.DIM_CUSTOMER")
        self.assertEqual(command.entity_name, "DIM_CUSTOMER")
        self.assertEqual(command.confidence, 0.95)

    def test_register_entity_rejects_hallucinated_table(self):
        command, error = compile_graph_command_response(
            self._raw(action="register_entity", table_name="DBO.NOT_A_REAL_TABLE"),
            self.manifest,
        )
        self.assertIsNone(command)
        self.assertIn("not found", error)

    def test_create_join_happy_path(self):
        command, error = compile_graph_command_response(
            self._raw(
                action="create_join",
                from_table="DBO.F_ORDERS",
                to_table="DBO.DIM_CUSTOMER",
                from_column="CUSTOMER_ID",
                to_column="CUSTOMER_ID",
                join_type="left",
                confidence=0.9,
            ),
            self.manifest,
        )
        self.assertEqual(error, "")
        self.assertEqual(command.action, "create_join")
        self.assertEqual(command.from_table, "DBO.F_ORDERS")
        self.assertEqual(command.to_table, "DBO.DIM_CUSTOMER")
        self.assertEqual(command.from_column, "CUSTOMER_ID")
        self.assertEqual(command.to_column, "CUSTOMER_ID")
        self.assertEqual(command.join_type, "LEFT")
        self.assertEqual(command.relationship_type, "many_to_one")

    def test_create_join_rejects_hallucinated_column(self):
        command, error = compile_graph_command_response(
            self._raw(
                action="create_join",
                from_table="DBO.F_ORDERS",
                to_table="DBO.DIM_CUSTOMER",
                from_column="NOT_A_REAL_COLUMN",
                to_column="CUSTOMER_ID",
            ),
            self.manifest,
        )
        self.assertIsNone(command)
        self.assertIn("not found", error)

    def test_create_join_rejects_hallucinated_table(self):
        command, error = compile_graph_command_response(
            self._raw(
                action="create_join",
                from_table="DBO.NOT_A_REAL_TABLE",
                to_table="DBO.DIM_CUSTOMER",
                from_column="CUSTOMER_ID",
                to_column="CUSTOMER_ID",
            ),
            self.manifest,
        )
        self.assertIsNone(command)
        self.assertIn("not found", error)

    def test_batch_compiles_governed_join_and_filter_operations(self):
        commands, error = compile_graph_commands_response(json.dumps({
            "operations": [
                {
                    "action": "update_join",
                    "from_table": "DBO.F_ORDERS",
                    "to_table": "DBO.DIM_CUSTOMER",
                    "from_column": "CUSTOMER_ID",
                    "to_column": "CUSTOMER_ID",
                    "join_type": "INNER",
                    "confidence": 0.91,
                },
                {
                    "action": "set_entity_filter",
                    "table_name": "DBO.DIM_CUSTOMER",
                    "filter_column": "REGION",
                    "filter_operator": "=",
                    "filter_value": "North",
                    "confidence": 0.94,
                },
            ]
        }), self.manifest)
        self.assertEqual(error, "")
        self.assertEqual([item.action for item in commands], ["update_join", "set_entity_filter"])
        self.assertEqual(commands[0].join_type, "INNER")
        self.assertEqual(commands[1].where_clause, "REGION = 'North'")

    def test_structured_filter_escapes_literals_and_rejects_raw_sql(self):
        command, error = compile_graph_command_response(self._raw(
            action="set_entity_filter",
            table_name="DBO.DIM_CUSTOMER",
            filter_column="REGION",
            filter_operator="=",
            filter_value="O'Reilly",
        ), self.manifest)
        self.assertEqual(error, "")
        self.assertEqual(command.where_clause, "REGION = 'O''Reilly'")

        command, error = compile_graph_command_response(self._raw(
            action="set_entity_filter",
            table_name="DBO.DIM_CUSTOMER",
            where_clause="1=1; DROP TABLE customers",
        ), self.manifest)
        self.assertIsNone(command)
        self.assertIn("filter column", error)

    def test_batch_fails_closed_when_any_operation_is_invalid(self):
        commands, error = compile_graph_commands_response(json.dumps({
            "operations": [
                {"action": "register_entity", "table_name": "DBO.DIM_CUSTOMER"},
                {"action": "delete_join", "from_table": "DBO.NOT_REAL", "to_table": "DBO.F_ORDERS"},
            ]
        }), self.manifest)
        self.assertEqual(commands, [])
        self.assertIn("not found", error)

    def test_invalid_join_type_falls_back_to_left(self):
        command, error = compile_graph_command_response(
            self._raw(
                action="create_join",
                from_table="DBO.F_ORDERS",
                to_table="DBO.DIM_CUSTOMER",
                from_column="CUSTOMER_ID",
                to_column="CUSTOMER_ID",
                join_type="CROSS",
            ),
            self.manifest,
        )
        self.assertEqual(error, "")
        self.assertEqual(command.join_type, "LEFT")

    def test_unsupported_action_rejected(self):
        command, error = compile_graph_command_response(
            self._raw(action="unsupported"),
            self.manifest,
        )
        self.assertIsNone(command)
        self.assertIn("could not be matched", error)

    def test_unknown_action_rejected(self):
        command, error = compile_graph_command_response(
            self._raw(action="delete_everything"),
            self.manifest,
        )
        self.assertIsNone(command)
        self.assertIn("unsupported action", error)

    def test_unsupported_fields_rejected(self):
        command, error = compile_graph_command_response(
            self._raw(action="register_entity", table_name="DBO.DIM_CUSTOMER", raw_sql="DROP TABLE x"),
            self.manifest,
        )
        self.assertIsNone(command)
        self.assertIn("unsupported fields", error)

    def test_malformed_json_rejected(self):
        command, error = compile_graph_command_response("not json", self.manifest)
        self.assertIsNone(command)
        self.assertIn("valid JSON", error)

    def test_confidence_defaults_low_when_missing(self):
        command, error = compile_graph_command_response(
            self._raw(action="register_entity", table_name="DBO.DIM_CUSTOMER"),
            self.manifest,
        )
        self.assertEqual(error, "")
        self.assertEqual(command.confidence, 0.0)

    def test_confidence_clamped_to_range(self):
        command, _ = compile_graph_command_response(
            self._raw(action="register_entity", table_name="DBO.DIM_CUSTOMER", confidence=5.0),
            self.manifest,
        )
        self.assertEqual(command.confidence, 1.0)

    def test_explicit_mapping_accepts_unambiguous_schema_suffix(self):
        manifest = {
            "CHATBOT_DB.PHARMA_LAB.BR_RX_DIAGNOSIS": ["RX_ORDER_ID", "DIAGNOSIS_ID"],
        }
        commands, error = parse_explicit_graph_commands(
            "Change PHARMA_LAB.BR_RX_DIAGNOSIS from dimension to bridge",
            manifest,
        )
        self.assertEqual(error, "")
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0].table_name, "CHATBOT_DB.PHARMA_LAB.BR_RX_DIAGNOSIS")
        self.assertEqual(commands[0].entity_name, "BR_RX_DIAGNOSIS")

    def test_llm_plan_accepts_unambiguous_schema_suffix(self):
        manifest = {
            "CHATBOT_DB.PHARMA_LAB.F_RX_FILL": ["RX_ORDER_ID"],
            "CHATBOT_DB.PHARMA_LAB.F_RX_ORDER": ["RX_ORDER_ID"],
        }
        command, error = compile_graph_command_response(self._raw(
            action="update_join",
            from_table="PHARMA_LAB.F_RX_FILL",
            to_table="PHARMA_LAB.F_RX_ORDER",
            from_column="RX_ORDER_ID",
            to_column="RX_ORDER_ID",
            join_type="INNER",
        ), manifest)
        self.assertEqual(error, "")
        self.assertEqual(command.from_table, "CHATBOT_DB.PHARMA_LAB.F_RX_FILL")
        self.assertEqual(command.to_table, "CHATBOT_DB.PHARMA_LAB.F_RX_ORDER")

    def test_llm_plan_rejects_ambiguous_bare_table_suffix(self):
        manifest = {
            "CHATBOT_DB.PHARMA_LAB.F_RX_FILL": ["RX_ORDER_ID"],
            "CHATBOT_DB.ARCHIVE.F_RX_FILL": ["RX_ORDER_ID"],
            "CHATBOT_DB.PHARMA_LAB.F_RX_ORDER": ["RX_ORDER_ID"],
        }
        command, error = compile_graph_command_response(self._raw(
            action="update_join",
            from_table="F_RX_FILL",
            to_table="F_RX_ORDER",
            from_column="RX_ORDER_ID",
            to_column="RX_ORDER_ID",
        ), manifest)
        self.assertIsNone(command)
        self.assertIn("not found", error)


class GraphCommandPromptTests(unittest.TestCase):
    def test_prompt_never_contains_row_data_only_schema_manifest(self):
        manifest = {"DBO.DIM_CUSTOMER": ["CUSTOMER_ID", "CUSTOMER_NAME"]}
        built = build_graph_command_input("map the customer table", manifest)
        combined = built.system_prompt + built.user_prompt
        self.assertIn("DIM_CUSTOMER", combined)
        self.assertIn("CUSTOMER_ID", combined)
        # No row-shaped content (e.g. a sample value) should ever appear --
        # only table/column names drawn from the manifest.
        self.assertNotIn("row_count", combined)
        self.assertNotIn("sample_values", combined)


class ParseGraphCommandAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_parse_uses_complete_and_compiles_result(self):
        manifest = {"DBO.DIM_CUSTOMER": ["CUSTOMER_ID", "CUSTOMER_NAME"]}
        captured = {}

        async def complete(**kwargs):
            captured.update(kwargs)
            return json.dumps({"action": "register_entity", "table_name": "DBO.DIM_CUSTOMER"}), 10, 10

        command, error = await parse_graph_command("map the customer table", manifest, complete)
        self.assertEqual(error, "")
        self.assertEqual(command.action, "register_entity")
        self.assertIn("system", captured)
        self.assertIn("user", captured)

    async def test_parse_handles_complete_failure_gracefully(self):
        async def complete(**kwargs):
            raise RuntimeError("boom")

        command, error = await parse_graph_command("map the customer table", {}, complete)
        self.assertIsNone(command)
        self.assertIn("unavailable", error)


if __name__ == "__main__":
    unittest.main()
