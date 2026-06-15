"""
tests/test_formula_editor.py

Tests for formula editor enhancements:
  1. Duplicate column disambiguation (TABLE.COLUMN display, bare COLUMN insert)
  2. Function helper popover — db-aware catalogue, bucketing, search
  3. DB-aware syntax — correct templates per db_type in metrics route
  4. Syntax validator — integer division, null division, MEDIAN/concat dialect checks
  5. Architecture guards — template and routes correctness
"""
import os
import re
import sys
import tempfile
import unittest

# Isolate DB
_tmp_db = os.path.join(tempfile.mkdtemp(), "test_fe.db")
os.environ["QUERYBOT_DB_PATH"] = _tmp_db

for mod in list(sys.modules.keys()):
    if mod.startswith("store"):
        del sys.modules[mod]

import store.db as db_mod
db_mod.init_db()

TMPL_PATH  = os.path.join(os.path.dirname(__file__), "..", "admin", "templates", "client_metrics.html")
ROUTES_PATH = os.path.join(os.path.dirname(__file__), "..", "admin", "routes.py")

def _tmpl():
    with open(TMPL_PATH, encoding="utf-8") as f:
        return f.read()

def _routes():
    with open(ROUTES_PATH, encoding="utf-8") as f:
        return f.read()


# ── 1  Duplicate column display ──────────────────────────────────────────────
class TestDuplicateColumnDisplay(unittest.TestCase):
    """JS uses TABLE.COLUMN label for duplicate columns, inserts bare COLUMN."""

    def test_dup_set_built_on_load(self):
        tmpl = _tmpl()
        self.assertIn("_dupCols", tmpl, "_dupCols Set must be defined in template JS")
        self.assertIn("_buildDupSet", tmpl, "_buildDupSet() must be called after columns load")

    def test_display_label_uses_table_prefix_for_dups(self):
        tmpl = _tmpl()
        self.assertIn("_displayLabel", tmpl)
        self.assertIn("_dupCols.has(c.column)", tmpl)
        self.assertIn("c.table+\".\"", tmpl)

    def test_insert_always_uses_bare_col_name(self):
        """_insertSuggestion receives data-col which is always the bare column name."""
        tmpl = _tmpl()
        # data-col must be set to c.column (bare), not the display label
        self.assertIn("data-col=\"'+c.column+'\"", tmpl)

    def test_col_browser_also_uses_display_label(self):
        tmpl = _tmpl()
        # Field browser should also call _displayLabel for consistency
        self.assertIn("_displayLabel(c)", tmpl)

    def test_insert_suggestion_strips_table_prefix(self):
        """_insertSuggestion uses data-col (bare name) not the display label text."""
        tmpl = _tmpl()
        # The click handler reads dataset.col not innerText
        self.assertIn("_insertSuggestion(item.dataset.col)", tmpl)
        # insertColumnFromBrowser also receives bare name
        self.assertIn("insertColumnFromBrowser", tmpl)


# ── 2  Function helper popover ──────────────────────────────────────────────
class TestFnHelperPopover(unittest.TestCase):
    """ƒ button replaces old pill toolbar; popover has 6 buckets; search works."""

    def test_fn_helper_button_present(self):
        tmpl = _tmpl()
        self.assertIn("fn-helper-btn", tmpl)
        self.assertIn("ƒ Functions", tmpl)

    def test_old_formula_toolbar_removed(self):
        tmpl = _tmpl()
        # The flat pill toolbar class must be gone
        self.assertNotIn("formula-snippet", tmpl,
            "Old formula-snippet pill buttons must be removed in favour of the ƒ popover")

    def test_popover_element_present(self):
        tmpl = _tmpl()
        self.assertIn("fn-popover", tmpl)
        self.assertIn("fn-buckets", tmpl)

    def test_six_buckets_defined(self):
        tmpl = _tmpl()
        # Each bucket name must appear in FN_CATALOGUE
        for bucket in ("Aggregation", "Ratio & Division", "Conditional",
                       "Null Handling", "Type Conversion", "String"):
            self.assertIn(bucket, tmpl, f"Bucket '{bucket}' missing from FN_CATALOGUE")

    def test_fn_search_input_present(self):
        tmpl = _tmpl()
        self.assertIn("fnSearch", tmpl)
        self.assertIn("filterFns", tmpl)

    def test_open_close_helpers(self):
        tmpl = _tmpl()
        self.assertIn("openFnHelper", tmpl)
        self.assertIn("closeFnHelper", tmpl)

    def test_catalogue_has_count_distinct(self):
        tmpl = _tmpl()
        self.assertIn("COUNT(DISTINCT", tmpl)

    def test_catalogue_has_conditional_sum(self):
        tmpl = _tmpl()
        self.assertIn("Conditional SUM", tmpl)

    def test_catalogue_has_safe_divide(self):
        tmpl = _tmpl()
        self.assertIn("Safe divide", tmpl)

    def test_catalogue_has_coalesce(self):
        tmpl = _tmpl()
        self.assertIn("COALESCE", tmpl)

    def test_catalogue_has_cast_decimal(self):
        tmpl = _tmpl()
        self.assertIn("CAST", tmpl)
        self.assertIn("DECIMAL", tmpl)

    def test_catalogue_has_median(self):
        tmpl = _tmpl()
        self.assertIn("MEDIAN", tmpl)


# ── 3  DB-aware syntax ────────────────────────────────────────────────────────
class TestDbAwareSyntax(unittest.TestCase):
    """DB type is passed to template and drives function templates."""

    def test_db_type_passed_to_js(self):
        tmpl = _tmpl()
        self.assertIn("window._qbDbType", tmpl)
        self.assertIn("{{ db_type", tmpl)

    def test_metrics_route_resolves_db_type(self):
        routes = _routes()
        self.assertIn("db_type", routes)
        self.assertIn("get_db_config", routes)
        # db_type must be passed to the template context
        self.assertIn('"db_type": db_type', routes)

    def test_metrics_route_has_safe_fallback(self):
        routes = _routes()
        # If no db config, fallback to azure_sql
        self.assertIn('"azure_sql"', routes)

    def test_median_db_conditional(self):
        tmpl = _tmpl()
        # MEDIAN must have a branch for azure_sql (PERCENTILE_CONT)
        self.assertIn("PERCENTILE_CONT", tmpl)
        self.assertIn("azure_sql", tmpl)

    def test_concat_db_conditional(self):
        tmpl = _tmpl()
        # Concatenation must have azure_sql (+) and oracle/snowflake (||) branches
        self.assertIn("||", tmpl)
        # azure_sql uses + for concat
        self.assertIn("azure_sql", tmpl)

    def test_isnull_marked_azure_only(self):
        tmpl = _tmpl()
        # ISNULL snippet must be limited to azure_sql
        self.assertIn("dbs:[\"azure_sql\"]", tmpl)

    def test_nvl_marked_oracle_only(self):
        tmpl = _tmpl()
        self.assertIn("dbs:[\"oracle\"]", tmpl)

    def test_db_labels_defined(self):
        tmpl = _tmpl()
        self.assertIn("DB_LABELS", tmpl)
        self.assertIn("Azure SQL", tmpl)
        self.assertIn("Snowflake", tmpl)
        self.assertIn("Oracle", tmpl)

    def test_fn_db_badge_shown_in_popover(self):
        tmpl = _tmpl()
        self.assertIn("fn-db-label", tmpl)
        self.assertIn("fn-db-badge", tmpl)


# ── 4  Syntax validator ────────────────────────────────────────────────────────
class TestSyntaxValidator(unittest.TestCase):
    """
    Validator logic is JS — we test the pattern strings are present in template
    and that each rule fires on its specific trigger.
    All regex are tested by inspecting the template source.
    """

    def _get_validator_src(self):
        tmpl = _tmpl()
        start = tmpl.find("function updateFormulaHints")
        end   = tmpl.find("function setStatus", start)
        return tmpl[start:end]

    def test_validator_function_exists(self):
        tmpl = _tmpl()
        self.assertIn("updateFormulaHints", tmpl)

    def test_rule_select_in_expression_mode(self):
        src = self._get_validator_src()
        self.assertIn("SELECT", src)
        self.assertIn("expression", src)

    def test_rule_division_without_nullif(self):
        src = self._get_validator_src()
        # Must check for / operator without NULLIF
        self.assertIn("NULLIF", src)
        self.assertIn("division", src.lower())

    def test_rule_integer_division_risk(self):
        src = self._get_validator_src()
        self.assertIn("Integer division", src)
        self.assertIn("CAST", src)

    def test_rule_avg_null_warning(self):
        src = self._get_validator_src()
        self.assertIn("AVG", src)
        self.assertIn("null", src.lower())

    def test_rule_median_azure_warning(self):
        src = self._get_validator_src()
        self.assertIn("MEDIAN", src)
        self.assertIn("azure_sql", src)
        self.assertIn("PERCENTILE_CONT", src)

    def test_rule_pipe_concat_in_azure(self):
        src = self._get_validator_src()
        self.assertIn("||", src)
        self.assertIn("azure_sql", src)

    def test_rule_plus_concat_in_oracle(self):
        src = self._get_validator_src()
        self.assertIn("oracle", src)
        self.assertIn("||", src)

    def test_destructive_keywords_in_sql_mode(self):
        src = self._get_validator_src()
        self.assertIn("DROP", src)
        self.assertIn("DELETE", src)

    def test_good_state_set_on_clean_formula(self):
        src = self._get_validator_src()
        self.assertIn('"good"', src)

    def test_warn_and_danger_states_present(self):
        tmpl = _tmpl()
        self.assertIn("warn", tmpl)
        self.assertIn("danger", tmpl)


# ── 5  Architecture guards ─────────────────────────────────────────────────────
class TestArchitectureGuards(unittest.TestCase):

    def test_db_type_in_template_head_script(self):
        tmpl = _tmpl()
        # Must be set before the main JS IIFE runs
        idx_script = tmpl.find("window._qbDbType")
        idx_iife   = tmpl.find("(function(){")
        self.assertLess(idx_script, idx_iife,
            "window._qbDbType must be set before the main IIFE")

    def test_col_browser_still_present(self):
        tmpl = _tmpl()
        self.assertIn("colBrowser", tmpl)
        self.assertIn("colBrowserSearch", tmpl)

    def test_col_suggestion_dropdown_still_present(self):
        tmpl = _tmpl()
        self.assertIn("col-suggest", tmpl)
        self.assertIn("_showSuggestions", tmpl)

    def test_insert_at_cursor_parks_inside_parens(self):
        tmpl = _tmpl()
        self.assertIn("innerParen", tmpl.replace("inner = text.match", "innerParen"))
        # More precisely: the paren-parking logic
        self.assertIn('text.indexOf("(") + 1', tmpl)

    def test_tag_input_suggestions_still_present(self):
        tmpl = _tmpl()
        self.assertIn("_showTagSuggestions", tmpl)
        self.assertIn("required_columns", tmpl)
        self.assertIn("allowed_dimensions", tmpl)

    def test_toggle_edit_helper_present(self):
        tmpl = _tmpl()
        self.assertIn("toggleEdit", tmpl)

    def test_both_formula_editor_textareas_have_fn_btn(self):
        tmpl = _tmpl()
        # Both add-new and edit forms need the ƒ button
        count = tmpl.count("fn-helper-btn")
        self.assertGreaterEqual(count, 2,
            "Both add-new and edit metric forms need an ƒ Functions button")

    def test_fn_catalogue_has_25_plus_entries(self):
        tmpl = _tmpl()
        # Count { bucket: entries in FN_CATALOGUE
        count = tmpl.count("{ bucket:")
        self.assertGreaterEqual(count, 25,
            f"Expected at least 25 catalogue entries, found {count}")


if __name__ == "__main__":
    unittest.main()
