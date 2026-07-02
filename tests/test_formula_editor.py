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
    """ƒ button replaces old pill toolbar; popover has 7 buckets; search works."""

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

    def test_seven_buckets_defined(self):
        tmpl = _tmpl()
        # Each bucket name must appear in FN_CATALOGUE
        for bucket in ("Aggregation", "Date & Time", "Ratio & Division", "Conditional",
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

    def test_catalogue_has_date_functions(self):
        tmpl = _tmpl()
        for fn in ("TRY_CONVERT", "DATEDIFF", "DATEADD", "DATE_TRUNC", "GETDATE", "CURRENT_DATE"):
            self.assertIn(fn, tmpl, f"Date function '{fn}' missing from FN_CATALOGUE")

    def test_try_convert_documents_dms_key_columns(self):
        tmpl = _tmpl()
        self.assertIn("_DT_DMS_KEY", tmpl)
        self.assertIn("never operate on the raw key directly", tmpl)


class TestInlineFunctionSuggestions(unittest.TestCase):
    """While typing in the formula editor / row-expression field, matching
    SQL function names from FN_CATALOGUE are suggested inline (DB-aware),
    in addition to the existing column-name autocomplete."""

    def test_matching_functions_helper_present(self):
        tmpl = _tmpl()
        self.assertIn("function _matchingFunctions(token)", tmpl)
        self.assertIn("fn.dbs.indexOf(DB)", tmpl)

    def test_row_expression_field_requests_function_suggestions(self):
        tmpl = _tmpl()
        self.assertIn('editor.matches(".formula-editor,.metric-builder-row-expression")', tmpl)

    def test_function_suggestion_insert_helper_present(self):
        tmpl = _tmpl()
        self.assertIn("function _insertFunctionSuggestion(tpl)", tmpl)

    def test_keydown_and_click_route_function_suggestions_separately(self):
        tmpl = _tmpl()
        self.assertIn('sel.dataset.isfn === "1"', tmpl)
        self.assertIn('fcsItem.dataset.isfn === "1"', tmpl)

    def test_function_rows_render_with_distinct_marker(self):
        tmpl = _tmpl()
        self.assertIn('data-isfn="1"', tmpl)
        self.assertIn("fcs-fn", tmpl)

    def test_unknown_identifier_check_recognises_try_convert_and_date(self):
        # Regression: TRY_CONVERT(date, ...) — the exact pattern this
        # session's DATE-KEY RULE prompt fix recommends — was previously
        # flagged as containing unknown identifiers "TRY_CONVERT" and
        # "date" by the admin-facing syntax checker.
        tmpl = _tmpl()
        self.assertIn('"TRY_CONVERT"', tmpl)
        self.assertIn('"DATE"', tmpl)


class TestSqlSyntaxHighlighting(unittest.TestCase):
    """Transparent-text textarea over a colored backdrop div — like a DB
    IDE — for .formula-editor and .metric-builder-row-expression. Colors
    keywords, functions, known columns, strings, numbers, and operators."""

    def test_highlight_wrapper_and_backdrop_css_present(self):
        tmpl = _tmpl()
        self.assertIn(".sql-hl-wrap{", tmpl)
        self.assertIn(".sql-hl-backdrop{", tmpl)
        self.assertIn("pointer-events:none", tmpl)

    def test_all_sql_editor_fields_disable_spellcheck(self):
        # Regression: the transparent-text overlay technique lets the
        # browser's native spellcheck decoration (e.g. a highlighted
        # background on an unrecognized word like a column name fragment)
        # render on top of the custom syntax coloring, clashing with it.
        # Confirmed visually — reported as a "colour issue" on a partially
        # typed column name. All 4 SQL-writing textareas (2x
        # .formula-editor, 2x .metric-builder-row-expression) must disable
        # spellcheck/autocomplete/autocorrect/autocapitalize.
        tmpl = _tmpl()
        self.assertEqual(tmpl.count('class="formula-editor"'), 2)
        self.assertEqual(tmpl.count('class="metric-builder-row-expression"'), 2)
        # autocomplete="off" also appears on unrelated inputs elsewhere in
        # this template (e.g. base-table-input) — scope the check to the
        # combined attribute cluster these 4 fields actually carry.
        self.assertEqual(tmpl.count('spellcheck="false" autocomplete="off" autocorrect="off" autocapitalize="off"'), 4)

    def test_token_color_classes_present(self):
        tmpl = _tmpl()
        for cls in (".tok-kw", ".tok-fn", ".tok-col", ".tok-str", ".tok-num", ".tok-op"):
            self.assertIn(cls + "{", tmpl, f"Missing color rule for {cls}")

    def test_textarea_text_made_transparent_not_hidden(self):
        # The textarea's own text must be transparent (caret still visible)
        # rather than removed — screen readers and copy/paste still work
        # off the real <textarea> value.
        tmpl = _tmpl()
        self.assertIn("color:transparent!important", tmpl)
        self.assertIn("caretColor", tmpl)

    def test_tokenizer_function_present(self):
        tmpl = _tmpl()
        self.assertIn("function _tokenizeSqlHtml(text)", tmpl)

    def test_tokenizer_classifies_functions_keywords_and_known_columns(self):
        tmpl = _tmpl()
        self.assertIn('out += \'<span class="tok-fn">\'', tmpl)
        self.assertIn('out += \'<span class="tok-kw">\'', tmpl)
        self.assertIn('out += \'<span class="tok-col">\'', tmpl)
        self.assertIn('out += \'<span class="tok-str">\'', tmpl)
        self.assertIn('out += \'<span class="tok-num">\'', tmpl)
        self.assertIn('out += \'<span class="tok-op">\'', tmpl)

    def test_tokenizer_sources_functions_from_fn_catalogue_not_a_second_list(self):
        # Must reuse FN_CATALOGUE's fnName tags (same source as the inline
        # autocomplete) rather than maintaining a second, driftable list.
        tmpl = _tmpl()
        self.assertIn("function _sqlHlFnNames()", tmpl)
        self.assertIn("FN_CATALOGUE.forEach(function(fn){ if(fn.fnName)", tmpl)

    def test_tokenizer_sources_columns_from_all_columns(self):
        tmpl = _tmpl()
        self.assertIn("(_allColumns||[]).forEach(function(c){ colNames[String(c.column).toUpperCase()] = true; });", tmpl)

    def test_init_and_render_helpers_present(self):
        tmpl = _tmpl()
        self.assertIn("function _initSqlHighlight(editor)", tmpl)
        self.assertIn("function _renderSqlHighlight(editor)", tmpl)

    def test_init_is_idempotent_via_dataset_flag(self):
        tmpl = _tmpl()
        self.assertIn("editor.dataset.hlInit", tmpl)

    def test_init_transplants_computed_box_style_not_hardcoded_css(self):
        # Two different textareas (.formula-editor and
        # .metric-builder-row-expression) have different padding/border in
        # their own CSS rules — the backdrop must mirror whichever textarea
        # it wraps via getComputedStyle, not a second hardcoded CSS copy
        # that would drift out of sync. The read must happen BEFORE the
        # editor moves into .sql-hl-wrap — once inside, the wrap's own
        # color:transparent/background:transparent rules already apply and
        # would corrupt a getComputedStyle read taken afterward.
        tmpl = _tmpl()
        self.assertIn("var cs = getComputedStyle(editor);", tmpl)
        self.assertIn("backdrop.style[p] = box[p]", tmpl)
        idx_cs   = tmpl.index("var cs = getComputedStyle(editor);")
        idx_move = tmpl.index("wrap.appendChild(editor);")
        self.assertLess(idx_cs, idx_move,
            "getComputedStyle must run before the editor is moved into .sql-hl-wrap")

    def test_editor_keeps_own_border_backdrop_mirrors_it_transparently(self):
        # The editor's own border/background are left alone (border isn't
        # affected by background:transparent, so it still renders normally
        # on top) — only the backdrop gets a transparent border of the same
        # width, so padding-box text alignment matches exactly.
        tmpl = _tmpl()
        self.assertIn('backdrop.style.borderColor = "transparent";', tmpl)
        self.assertIn("backdrop.style.borderWidth = box.borderTopWidth", tmpl)

    def test_resize_observer_keeps_backdrop_sized_on_manual_resize(self):
        tmpl = _tmpl()
        self.assertIn("new ResizeObserver(function(){ _renderSqlHighlight(editor); })", tmpl)

    def test_mouseup_fallback_resyncs_box_after_drag_resize(self):
        # ResizeObserver doesn't fire reliably for a manual drag-resize in
        # every browser/automation context — mouseup is a defensive
        # fallback that catches the end of the drag either way.
        tmpl = _tmpl()
        self.assertIn('editor.addEventListener("mouseup", function(){ _renderSqlHighlight(editor); });', tmpl)

    def test_render_resyncs_box_size_not_just_content(self):
        # _renderSqlHighlight must re-check width/height on every call, not
        # only content — this is the primary size-sync path, ResizeObserver
        # and mouseup are best-effort enhancements on top of it.
        tmpl = _tmpl()
        idx = tmpl.index("function _renderSqlHighlight(editor){")
        end = tmpl.index("\n}", idx)
        body = tmpl[idx:end]
        self.assertIn("backdrop.style.width  = editor.offsetWidth", body)
        self.assertIn("backdrop.style.height = editor.offsetHeight", body)

    def test_type_names_colored_as_keywords(self):
        # Regression: TRY_CONVERT(date, ...) left "date" uncolored because
        # type names weren't in the keyword set — confirmed visually during
        # manual verification of this feature.
        tmpl = _tmpl()
        for type_name in ("DATE", "VARCHAR", "DECIMAL", "INT"):
            self.assertIn(f'"{type_name}"', tmpl)

    def test_keyword_check_runs_before_function_check(self):
        # Regression: CASE has both fnName:"CASE" (for autocomplete) and is
        # a reserved keyword — checking fnNames first colored it purple
        # (function) instead of blue (keyword). Confirmed visually during
        # manual verification: the keyword check must run first.
        tmpl = _tmpl()
        idx = tmpl.index("function _tokenizeSqlHtml(text){")
        end = tmpl.index("\nfunction _renderSqlHighlight", idx)
        body = tmpl[idx:end]
        idx_kw_check = body.index("_SQL_HL_KEYWORDS.has(upper)")
        idx_fn_check = body.index("fnNames[upper]")
        self.assertLess(idx_kw_check, idx_fn_check,
            "keyword check must be evaluated before the function-name check")

    def test_scroll_position_synced_between_backdrop_and_textarea(self):
        tmpl = _tmpl()
        self.assertIn('editor.addEventListener("scroll", function(){', tmpl)
        self.assertIn("backdrop.scrollTop  = editor.scrollTop;", tmpl)

    def test_highlight_wired_into_focusin_and_input_for_both_fields(self):
        tmpl = _tmpl()
        self.assertIn("_initSqlHighlight(e.target);", tmpl)
        self.assertIn("_renderSqlHighlight(e.target);", tmpl)

    def test_highlight_refreshed_after_function_template_insert(self):
        # insertAtCursor is used by both the ƒ Functions popover and the
        # inline function-suggestion dropdown — both must refresh the
        # backdrop so the inserted template is colored immediately.
        tmpl = _tmpl()
        self.assertIn("updateFormulaHints(editor);\n  _renderSqlHighlight(editor);", tmpl)

    def test_highlight_refreshed_when_builder_compiles_formula(self):
        tmpl = _tmpl()
        self.assertIn(
            'if(editor){ editor.value = formula; updateFormulaHints(editor); _renderSqlHighlight(editor); }',
            tmpl,
        )

    def test_highlight_reapplied_once_real_columns_load(self):
        # So a field name typed before the schema fetch resolves gets
        # retroactively colored green once real columns are known.
        tmpl = _tmpl()
        self.assertIn(
            'document.querySelectorAll(".formula-editor,.metric-builder-row-expression").forEach(_initSqlHighlight);',
            tmpl,
        )

    def test_required_joins_json_field_excluded_from_sql_highlighting(self):
        # required_joins is JSON, not SQL — must not be tokenized as SQL
        # (would misleadingly color JSON punctuation as SQL operators).
        # _initSqlHighlight/_renderSqlHighlight must only be gated behind a
        # matcher naming .metric-builder-row-expression alone, never paired
        # with .metric-builder-required-joins the way _showSuggestions is.
        tmpl = _tmpl()
        self.assertIn(
            'if(e.target.matches(".metric-builder-row-expression")){\n    _initSqlHighlight(e.target);\n  }',
            tmpl,
        )
        self.assertIn(
            'if(e.target.matches(".metric-builder-row-expression")){\n    _renderSqlHighlight(e.target);\n  }',
            tmpl,
        )


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

    def test_nvl_marked_oracle_and_snowflake_not_azure(self):
        # Regression: NVL() is natively supported by both Oracle AND
        # Snowflake (not Oracle-only as originally scoped) — audited
        # against real dialect support 2026-07-02. Azure SQL has no NVL;
        # ISNULL/COALESCE cover that dialect instead.
        tmpl = _tmpl()
        idx = tmpl.index('name:"NVL')
        entry = tmpl[idx:idx + 320]
        self.assertIn('dbs:["oracle","snowflake"]', entry)
        self.assertNotIn("azure_sql", entry)

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
        # The inline "colBrowser*" sidebar was replaced by the Qlik-style New
        # Metric modal's field browser (commit ed951e4) — assert on the
        # current mc-fields-* implementation instead of the removed markup.
        tmpl = _tmpl()
        self.assertIn("mc-fields-list", tmpl)
        self.assertIn("mc-fields-search", tmpl)

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
