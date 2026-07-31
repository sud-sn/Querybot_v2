import json
import unittest

from core.graph_commands import (
    build_graph_command_input,
    compile_graph_command_response,
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
