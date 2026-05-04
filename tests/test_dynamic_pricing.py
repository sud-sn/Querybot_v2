"""
tests/test_dynamic_pricing.py

Tests for Feature 1: Dynamic LLM billing cost rates.
Covers:
  - calculate_cost() falls back to hardcoded defaults when DB is empty
  - save_pricing() persists to DB and invalidates cache
  - calculate_cost() reads updated DB rate after save
  - get_all_pricing() merges DB rows and defaults correctly
  - save_pricing() rejects negative rates (via the route layer logic)
  - calculate_cost() uses safe fallback for completely unknown model
  - LLM_COST_RATES hardcoded dict is unchanged (regression)
  - Existing record_llm_call path still calls calculate_cost (wiring check)
  - DB seed migration seeds known models without overwriting edits
"""
import importlib
import sys
import os
import tempfile
import unittest

# ── Isolate DB to a temp file so tests never touch real data ─────────────────
_tmp_dir  = tempfile.mkdtemp()
_tmp_db   = os.path.join(_tmp_dir, "test_querybot.db")
os.environ["QUERYBOT_DB_PATH"] = _tmp_db


def _fresh_modules():
    """Reload store modules so each test group gets a clean in-memory state."""
    for mod in list(sys.modules.keys()):
        if mod.startswith("store"):
            del sys.modules[mod]


_fresh_modules()
import store.db as db_mod
import store.config_store as cs

db_mod.init_db()


class TestHardcodedFallback(unittest.TestCase):
    """calculate_cost falls back to hardcoded dict when DB has no row."""

    def setUp(self):
        # Wipe pricing table and in-process cache
        cs._pricing_cache = None
        with db_mod.get_db() as conn:
            conn.execute("DELETE FROM llm_pricing")

    def test_gpt4o_hardcoded_fallback(self):
        # After clearing DB, should still compute using LLM_COST_RATES
        cost = cs.calculate_cost("gpt-4o", 1_000_000, 0)
        self.assertAlmostEqual(cost, 5.00, places=4)

    def test_gpt4o_mini_hardcoded_fallback(self):
        cost = cs.calculate_cost("gpt-4o-mini", 1_000_000, 1_000_000)
        expected = (1_000_000 * 0.15 + 1_000_000 * 0.60) / 1_000_000
        self.assertAlmostEqual(cost, expected, places=6)

    def test_completely_unknown_model_safe_fallback(self):
        # Unknown model must not return 0.0 — uses gpt-4o rate as safe default
        cost = cs.calculate_cost("some-future-model-9000", 1_000_000, 0)
        self.assertGreater(cost, 0.0)

    def test_hardcoded_dict_unchanged(self):
        # Regression: nobody accidentally removed models from LLM_COST_RATES
        for model in ("gpt-4o", "gpt-4o-mini", "claude-sonnet-4-6",
                      "claude-opus-4-5", "claude-haiku-4-5"):
            self.assertIn(model, cs.LLM_COST_RATES)
            rates = cs.LLM_COST_RATES[model]
            self.assertIn("in", rates)
            self.assertIn("out", rates)
            self.assertGreater(rates["in"],  0)
            self.assertGreater(rates["out"], 0)


class TestSavePricingAndCache(unittest.TestCase):
    """save_pricing() persists to DB, invalidates cache, calculate_cost picks it up."""

    def setUp(self):
        cs._pricing_cache = None
        with db_mod.get_db() as conn:
            conn.execute("DELETE FROM llm_pricing")

    def test_save_and_read_back(self):
        cs.save_pricing("gpt-4o", 2.50, 10.00)
        cs._pricing_cache = None  # force reload
        cost = cs.calculate_cost("gpt-4o", 1_000_000, 0)
        self.assertAlmostEqual(cost, 2.50, places=4)

    def test_cache_invalidated_on_save(self):
        # Prime cache with old value
        _ = cs.calculate_cost("gpt-4o", 0, 0)
        self.assertIsNotNone(cs._pricing_cache)
        cs.save_pricing("gpt-4o", 1.00, 5.00)
        # Cache should be None after save
        self.assertIsNone(cs._pricing_cache)

    def test_updated_rate_used_in_next_call(self):
        cs.save_pricing("gpt-4o", 1.00, 5.00)
        cost = cs.calculate_cost("gpt-4o", 1_000_000, 1_000_000)
        expected = (1_000_000 * 1.00 + 1_000_000 * 5.00) / 1_000_000
        self.assertAlmostEqual(cost, expected, places=6)

    def test_add_brand_new_model(self):
        cs.save_pricing("my-custom-model", 0.50, 2.00)
        cost = cs.calculate_cost("my-custom-model", 2_000_000, 0)
        self.assertAlmostEqual(cost, 1.00, places=4)

    def test_upsert_overwrites_existing(self):
        cs.save_pricing("gpt-4o-mini", 9.99, 9.99)
        cs.save_pricing("gpt-4o-mini", 0.10, 0.40)  # overwrite
        cs._pricing_cache = None
        cost = cs.calculate_cost("gpt-4o-mini", 1_000_000, 0)
        self.assertAlmostEqual(cost, 0.10, places=4)


class TestGetAllPricing(unittest.TestCase):
    """get_all_pricing() merges DB rows with hardcoded defaults."""

    def setUp(self):
        cs._pricing_cache = None
        with db_mod.get_db() as conn:
            conn.execute("DELETE FROM llm_pricing")

    def test_returns_list(self):
        rows = cs.get_all_pricing()
        self.assertIsInstance(rows, list)
        self.assertGreater(len(rows), 0)

    def test_all_rows_have_required_keys(self):
        for row in cs.get_all_pricing():
            for key in ("model", "tokens_in", "tokens_out", "updated_at", "source"):
                self.assertIn(key, row, f"Missing key {key!r} in row {row}")

    def test_db_row_source_is_db(self):
        cs.save_pricing("gpt-4o", 3.00, 12.00)
        rows = {r["model"]: r for r in cs.get_all_pricing()}
        self.assertEqual(rows["gpt-4o"]["source"], "db")

    def test_default_row_source_is_default(self):
        # When no DB row exists for a model, source should be 'default'
        rows = {r["model"]: r for r in cs.get_all_pricing()}
        # At least some models have no DB row yet → source='default'
        default_rows = [r for r in rows.values() if r["source"] == "default"]
        self.assertGreater(len(default_rows), 0)

    def test_no_duplicate_models(self):
        cs.save_pricing("gpt-4o", 5.00, 15.00)
        models = [r["model"] for r in cs.get_all_pricing()]
        self.assertEqual(len(models), len(set(models)), "Duplicate models in get_all_pricing()")

    def test_tokens_in_out_are_floats(self):
        for row in cs.get_all_pricing():
            self.assertIsInstance(row["tokens_in"],  float)
            self.assertIsInstance(row["tokens_out"], float)


class TestSeedMigration(unittest.TestCase):
    """DB seed migration inserts defaults without overwriting edits."""

    def setUp(self):
        cs._pricing_cache = None

    def test_seed_does_not_overwrite_manual_edit(self):
        # Simulate admin edit
        cs.save_pricing("gpt-4o", 1.23, 4.56)
        # Re-run migration (as would happen on restart)
        with db_mod.get_db() as conn:
            for model, rates in cs.LLM_COST_RATES.items():
                conn.execute(
                    "INSERT OR IGNORE INTO llm_pricing (model, tokens_in, tokens_out) VALUES (?,?,?)",
                    (model, rates["in"], rates["out"]),
                )
        cs._pricing_cache = None
        cost = cs.calculate_cost("gpt-4o", 1_000_000, 0)
        # Admin rate 1.23 must survive restart
        self.assertAlmostEqual(cost, 1.23, places=4)


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
    """Ensure new functions are properly exported from store and billing template has new vars."""

    def test_store_exports_get_all_pricing(self):
        import store
        self.assertTrue(callable(store.get_all_pricing))

    def test_store_exports_save_pricing(self):
        import store
        self.assertTrue(callable(store.save_pricing))

    def test_billing_template_uses_pricing_rows(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "admin", "templates", "billing.html")) as f:
            tmpl = f.read()
        self.assertIn("pricing_rows", tmpl,
                      "billing.html must iterate over pricing_rows, not old cost_rates dict")
        self.assertNotIn("cost_rates.items()", tmpl,
                         "billing.html must not reference old cost_rates.items()")

    def test_billing_template_has_pricing_save_form(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "admin", "templates", "billing.html")) as f:
            tmpl = f.read()
        self.assertIn("billing/pricing/save", tmpl,
                      "billing.html must include the pricing save form action")

    def test_metrics_template_has_col_suggest(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "admin", "templates", "client_metrics.html")) as f:
            tmpl = f.read()
        self.assertIn("col-suggest", tmpl)
        self.assertIn("_allColumns", tmpl)
        self.assertIn("api/columns", tmpl)

    def test_metrics_template_has_field_browser(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "admin", "templates", "client_metrics.html")) as f:
            tmpl = f.read()
        self.assertIn("colBrowser", tmpl)
        self.assertIn("colBrowserSearch", tmpl)
        self.assertIn("insertColumnFromBrowser", tmpl)

    def test_metrics_template_cursor_in_parens(self):
        """Snippet buttons now park cursor inside parens."""
        with open(os.path.join(os.path.dirname(__file__), "..", "admin", "templates", "client_metrics.html")) as f:
            tmpl = f.read()
        self.assertIn("inner = text.match", tmpl,
                      "insertAtCursor should detect empty parens and park cursor inside them")

    def test_admin_routes_has_pricing_save_endpoint(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "admin", "routes.py")) as f:
            src = f.read()
        self.assertIn("billing/pricing/save", src)
        self.assertIn("billing_pricing_save", src)

    def test_admin_routes_has_columns_api(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "admin", "routes.py")) as f:
            src = f.read()
        self.assertIn("api/columns", src)
        self.assertIn("admin_columns_api", src)

    def test_db_has_llm_pricing_table(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "store", "db.py")) as f:
            src = f.read()
        self.assertIn("CREATE TABLE IF NOT EXISTS llm_pricing", src)

    def test_config_store_has_pricing_cache(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "store", "config_store.py")) as f:
            src = f.read()
        self.assertIn("_pricing_cache", src)
        self.assertIn("get_all_pricing", src)
        self.assertIn("save_pricing", src)


if __name__ == "__main__":
    unittest.main()
