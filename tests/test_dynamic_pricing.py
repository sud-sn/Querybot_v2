"""
tests/test_dynamic_pricing.py

The columns API's flattening and the metric editor's field suggestions.

This file also held the tests of the one-rate-pair-per-model price table on each
workspace's Billing page (llm_pricing, save_pricing, get_all_pricing, the gpt-4o
fallback for unknown models). That table is gone: prices are set once on Admin ->
System and every call is costed when it is made. Those tests live in
test_every_ai_call_costs_what_it_cost.py and test_ai_prices_are_set_once_on_system.py.
"""
import os
import unittest

from metrics_template import metrics_template


class TestColumnsAPIHelpers(unittest.TestCase):
    """Unit test the column-flattening logic used by the columns API."""

    def _flatten_tree(self, tree):
        """Mirror the flattening logic in admin_columns_api."""
        columns = []
        for _db, schemas in tree.items():
            for _schema, objs in schemas.items():
                for tbl_name, tbl_info in objs.get("tables", {}).items():
                    for col in tbl_info.get("columns", []):
                        col_name = col.get("name") or col.get("column_name", "")
                        col_type = col.get("type") or col.get("data_type", "")
                        if col_name:
                            columns.append({
                                "table":  tbl_name,
                                "column": col_name,
                                "type":   col_type,
                                "fqn":    f"{tbl_name}.{col_name}",
                            })
        columns.sort(key=lambda c: (c["table"].lower(), c["column"].lower()))
        return columns

    def test_basic_flatten(self):
        tree = {
            "CHATBOT_DB": {
                "HR": {
                    "tables": {
                        "EMPLOYEE": {
                            "columns": [
                                {"name": "EmployeeID",   "type": "int"},
                                {"name": "EmployeeName", "type": "nvarchar"},
                                {"name": "Nationality",  "type": "nvarchar"},
                            ]
                        }
                    }
                }
            }
        }
        cols = self._flatten_tree(tree)
        self.assertEqual(len(cols), 3)
        names = [c["column"] for c in cols]
        self.assertIn("EmployeeID", names)
        self.assertIn("Nationality", names)

    def test_sorted_output(self):
        tree = {
            "DB": {
                "SCH": {
                    "tables": {
                        "ZEBRA": {"columns": [{"name": "ZCol", "type": "int"}]},
                        "ALPHA": {"columns": [{"name": "ACol", "type": "int"}]},
                    }
                }
            }
        }
        cols = self._flatten_tree(tree)
        self.assertEqual(cols[0]["table"], "ALPHA")
        self.assertEqual(cols[1]["table"], "ZEBRA")

    def test_empty_tree(self):
        self.assertEqual(self._flatten_tree({}), [])

    def test_fqn_format(self):
        tree = {"DB": {"SCH": {"tables": {"TBL": {"columns": [{"name": "Col1", "type": "int"}]}}}}}
        cols = self._flatten_tree(tree)
        self.assertEqual(cols[0]["fqn"], "TBL.Col1")

    def test_column_name_fallback_key(self):
        # Some adapters use 'column_name' instead of 'name'
        tree = {"DB": {"SCH": {"tables": {"TBL": {"columns": [{"column_name": "OldKey", "data_type": "varchar"}]}}}}}
        cols = self._flatten_tree(tree)
        self.assertEqual(cols[0]["column"], "OldKey")
        self.assertEqual(cols[0]["type"],   "varchar")

    def test_skips_empty_column_names(self):
        tree = {"DB": {"SCH": {"tables": {"TBL": {"columns": [
            {"name": "",    "type": "int"},
            {"name": "Good","type": "int"},
        ]}}}}}
        cols = self._flatten_tree(tree)
        self.assertEqual(len(cols), 1)
        self.assertEqual(cols[0]["column"], "Good")

    def test_multiple_schemas_and_tables(self):
        tree = {
            "DB": {
                "HR":      {"tables": {"EMPLOYEE": {"columns": [{"name": "EmpID",  "type": "int"}]}}},
                "FINANCE": {"tables": {"INVOICE":  {"columns": [{"name": "InvAmt", "type": "decimal"}]}}},
            }
        }
        cols = self._flatten_tree(tree)
        self.assertEqual(len(cols), 2)
        tables = {c["table"] for c in cols}
        self.assertIn("EMPLOYEE", tables)
        self.assertIn("INVOICE", tables)


class TestArchitectureGuards(unittest.TestCase):
    """The metric editor suggests columns as they are typed, from the columns API."""

    def test_metrics_template_has_col_suggest(self):
        tmpl = metrics_template()
        self.assertIn("col-suggest", tmpl)
        self.assertIn("_allColumns", tmpl)
        self.assertIn("api/columns", tmpl)

    def test_metrics_template_has_field_browser(self):
        # The inline "colBrowser*" sidebar was replaced by the Qlik-style New
        # Metric modal's field browser (commit ed951e4) — assert on the
        # current mc-fields-* implementation instead of the removed markup.
        tmpl = metrics_template()
        self.assertIn("mc-fields-list", tmpl)
        self.assertIn("mc-fields-search", tmpl)
        self.assertIn("insertColumnFromMcDialog", tmpl)

    def test_metrics_template_cursor_in_parens(self):
        """Snippet buttons now park cursor inside parens."""
        tmpl = metrics_template()
        self.assertIn("inner = text.match", tmpl,
                      "insertAtCursor should detect empty parens and park cursor inside them")

    def test_admin_routes_has_columns_api(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "admin", "routes.py"), encoding="utf-8") as f:
            src = f.read()
        self.assertIn("api/columns", src)
        self.assertIn("admin_columns_api", src)



if __name__ == "__main__":
    unittest.main()
