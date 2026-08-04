"""Regression tests for governed workspace onboarding answers."""

from __future__ import annotations

import unittest
from unittest.mock import patch


MODEL = {
    "business_description": "A governed retail workspace.",
    "tables": [
        {
            "fqn": "DB.SALES.F_ORDERS",
            "schema": "SALES",
            "table": "F_ORDERS",
            "entity": "Orders",
            "type": "fact",
            "grain": "one row per order",
            "fields": [{"column": "PRIVATE_COLUMN"}],
        },
        {
            "fqn": "DB.SECRET.D_PAYROLL",
            "schema": "SECRET",
            "table": "D_PAYROLL",
            "entity": "Payroll",
            "type": "dimension",
            "grain": "one row per employee",
        },
    ],
    "relationships": [
        {"from": "DB.SALES.F_ORDERS", "to": "DB.SALES.D_CUSTOMER"},
        {"from": "DB.SALES.F_ORDERS", "to": "DB.SECRET.D_PAYROLL"},
    ],
    "date_roles": [
        {"fact_table": "DB.SALES.F_ORDERS", "dimension_table": "DB.SALES.D_DATE"},
        {"fact_table": "DB.SECRET.D_PAYROLL", "dimension_table": "DB.SECRET.D_DATE"},
    ],
}


class WorkspaceGuideTests(unittest.TestCase):
    def setUp(self):
        self.user = {"id": 7, "role": "user", "name": "Delivery Manager"}
        self.patchers = [
            patch("core.workspace_guide.store.get_client", return_value={
                "business_desc": "We manage retail orders and customer performance."
            }),
            patch("core.workspace_guide.get_state", return_value={
                "state": "READY", "kb_dir": "Z:/missing/kb", "schema_dir": "Z:/missing/schema"
            }),
            patch("core.workspace_guide.load_semantic_model", return_value=MODEL),
            patch("core.workspace_guide.store.get_allowed_tables", return_value={
                "DB.SALES.F_ORDERS", "DB.SALES.D_CUSTOMER", "DB.SALES.D_DATE"
            }),
            patch("core.workspace_guide.store.list_metrics", return_value=[
                {"name": "Revenue", "description": "Recognized revenue", "base_table": "DB.SALES.F_ORDERS", "is_active": 1},
                {"name": "Salary", "description": "Payroll", "base_table": "DB.SECRET.D_PAYROLL", "is_active": 1},
                {"name": "Unbound", "description": "No governed table", "base_table": "", "is_active": 1},
            ]),
            patch("core.workspace_guide.store.list_terms", return_value=[
                {"term": "customer", "definition": "A buying organization", "tables_involved": "DB.SALES.D_CUSTOMER"},
                {"term": "salary", "definition": "Employee salary", "tables_involved": "DB.SECRET.D_PAYROLL"},
            ]),
            patch("core.workspace_guide.store.list_dashboards", return_value=[
                {"id": 1, "name": "Sales overview", "chart_count": 4}
            ]),
            patch("core.workspace_guide._safe_examples", return_value=[
                "Show monthly revenue?", "Which customers placed the most orders?"
            ]),
            patch("core.workspace_guide.get_portal_base", return_value="https://querybot.example"),
        ]
        for patcher in self.patchers:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.patchers):
            patcher.stop()

    def test_build_is_acl_scoped_across_all_metadata(self):
        from core.workspace_guide import build_workspace_guide

        guide = build_workspace_guide("client-1", self.user)
        self.assertEqual([table["name"] for table in guide["tables"]], ["Orders"])
        self.assertEqual([metric["name"] for metric in guide["metrics"]], ["Revenue"])
        self.assertEqual([term["term"] for term in guide["terms"]], ["customer"])
        self.assertEqual(guide["relationship_count"], 1)
        self.assertEqual(guide["date_role_count"], 1)
        self.assertEqual(len(guide["dashboards"]), 1)

    def test_table_answer_has_business_meaning_but_no_fields(self):
        from core.workspace_guide import render_workspace_guide

        text, examples = render_workspace_guide("table_meanings", "client-1", self.user)
        self.assertIn("Orders", text)
        self.assertIn("one row per order", text)
        self.assertNotIn("Payroll", text)
        self.assertNotIn("PRIVATE_COLUMN", text)
        self.assertEqual(examples[0], "Show monthly revenue?")

    def test_capability_answer_describes_dynamic_dashboard_and_semantics(self):
        from core.workspace_guide import render_workspace_guide

        text, _ = render_workspace_guide("capability_overview", "client-1", self.user)
        self.assertIn("1 named dashboard", text)
        self.assertIn("1 governed metric", text)
        self.assertIn("1 business term", text)
        self.assertIn("1 relationship", text)
        self.assertIn("1 date role", text)
        self.assertIn("https://querybot.example/portal/dashboard", text)
        self.assertNotIn("Salary", text)

    def test_button_renderer_does_not_duplicate_examples_in_message(self):
        from core.workspace_guide import render_workspace_guide

        text, examples = render_workspace_guide(
            "question_examples", "client-1", self.user, include_examples=False
        )
        self.assertEqual(len(examples), 2)
        self.assertNotIn("Show monthly revenue", text)
        self.assertNotIn("Validated questions", text)

    def test_missing_curated_table_description_uses_safe_grain_fallback(self):
        from core.workspace_guide import build_workspace_guide

        guide = build_workspace_guide("client-1", self.user)
        meaning = guide["tables"][0]["meaning"]
        self.assertEqual(meaning, "Represents orders at one row per order.")

    def test_restricted_user_without_table_access_sees_no_inventory(self):
        from core.workspace_guide import render_workspace_guide

        with patch("core.workspace_guide.store.get_allowed_tables", return_value=set()):
            text, _ = render_workspace_guide("data_inventory", "client-1", self.user)
        self.assertIn("No queryable tables are assigned", text)
        self.assertNotIn("Orders", text)
        self.assertNotIn("Payroll", text)


if __name__ == "__main__":
    unittest.main()
