"""
tests/test_metric_wizard.py

Guided create-wizard on the Metric Registry page (admin/templates/client_metrics.html).
Template-marker convention matching tests/test_formula_editor.py and
tests/test_metric_builder.py. The wizard is a pure DOM restructure of the
existing create dialog — these tests pin the new structural markers without
duplicating the exhaustive JS-behavior assertions already covered elsewhere.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Isolate DB for the usage-tracking tests further down this file.
_tmp_db = os.path.join(tempfile.mkdtemp(), "test_mw.db")
os.environ["QUERYBOT_DB_PATH"] = _tmp_db
for mod in list(sys.modules.keys()):
    if mod.startswith("store"):
        del sys.modules[mod]
import store.db as db_mod
db_mod.init_db()
import store


class MetricWizardStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = (ROOT / "admin" / "templates" / "client_metrics.html").read_text(encoding="utf-8")

    def test_three_wizard_steps_only_in_create_dialog(self):
        # Exactly 3 — proves the edit form (inline expand) was not wrapped
        # into the wizard; only #mc-form gained step wrappers.
        self.assertEqual(self.template.count('class="mc-step"'), 3)
        for n in (1, 2, 3):
            self.assertIn(f'<div class="mc-step" data-step="{n}">', self.template)

    def test_stepper_bar_present(self):
        self.assertIn("mc-wizard-bar", self.template)
        self.assertIn("wizard-step-circle", self.template)
        self.assertIn("wizard-step-label", self.template)
        self.assertIn('data-goto="1"', self.template)
        self.assertIn('data-goto="2"', self.template)
        self.assertIn('data-goto="3"', self.template)

    def test_footer_has_back_next_and_gated_submit(self):
        self.assertIn('id="mc-back"', self.template)
        self.assertIn('id="mc-next"', self.template)
        self.assertIn(':not([data-step="3"]) .mc-footer [type="submit"]{display:none}', self.template)

    def test_mode_cards_cover_all_four_calculation_kinds(self):
        self.assertGreaterEqual(self.template.count("mc-mode-card"), 4)
        for mode in ("aggregate", "date_gap", "expression", "query"):
            self.assertIn(f'data-mode="{mode}"', self.template)

    def test_select_mode_card_writes_legacy_controls_via_change_event(self):
        # Mode cards must drive the *existing* handler matrix (dispatchEvent),
        # never duplicate the compile/sync logic themselves.
        self.assertIn("window.selectMcModeCard", self.template)
        self.assertIn(".metric-builder-mode", self.template)
        self.assertIn('dispatchEvent(new Event("change"', self.template)

    def test_no_sql_modes_never_show_the_formula_editor(self):
        self.assertIn('[data-mc-mode="aggregate"] .mc-formula-group', self.template)
        self.assertIn('[data-mc-mode="date_gap"] .mc-formula-group', self.template)

    def test_navigation_functions_present(self):
        for fn in ("_mcGoToStep", "_mcValidateStep", "_mcUpdateStepper", "_mcSetStep"):
            self.assertIn(fn, self.template)
        # Step 2 gating must reuse the real compiler, not a re-implementation.
        self.assertIn("_syncMetricBuilder(form)", self.template)

    def test_review_step_hosts_lineage_and_readonly_formula(self):
        self.assertIn("data-mc-review", self.template)
        self.assertIn("data-mc-review-formula", self.template)
        self.assertIn("_ensureMetricUxPanels", self.template)
        # Patched to target the review host when present.
        self.assertIn('form.querySelector("[data-mc-review]")', self.template)

    def test_ai_import_lands_on_review_step(self):
        ai_idx = self.template.index("window.aiImportMetric")
        next_ai_fn = self.template.index("function _checkParens")
        ai_block = self.template[ai_idx:next_ai_fn]
        self.assertIn("_mcGoToStep(3)", ai_block)

    def test_clone_marks_all_steps_visited(self):
        clone_idx = self.template.index("window.cloneMetric")
        export_idx = self.template.index("window.exportMetricsJSON")
        clone_block = self.template[clone_idx:export_idx]
        self.assertIn("_mcVisited = {1:true, 2:true, 3:true}", clone_block)
        self.assertIn("_mcSelectCardForConfig", clone_block)

    def test_enter_key_guard_present(self):
        self.assertIn('e.key !== "Enter"', self.template)

    def test_save_gating_revalidates_before_submit(self):
        self.assertIn('e.target.id !== "mc-form"', self.template)
        self.assertIn("_mcValidateStep(1)", self.template)
        self.assertIn("_mcValidateStep(2)", self.template)


class MetricListRevampTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = (ROOT / "admin" / "templates" / "client_metrics.html").read_text(encoding="utf-8")

    def test_row_carries_usage_and_status_data_attrs(self):
        self.assertIn('data-usage="{{ m.usage_count or 0 }}"', self.template)
        self.assertIn("status-pill ok", self.template)
        self.assertIn("status-pill warning", self.template)
        self.assertIn("status-pill error", self.template)
        self.assertIn("mr-usage", self.template)

    def test_toolbar_has_draft_chip_and_usage_sort(self):
        self.assertIn('data-filter="draft"', self.template)
        self.assertIn('value="usage-desc"', self.template)

    def test_usage_sort_falls_back_to_name(self):
        self.assertIn('sort === "usage-desc"', self.template)
        self.assertIn("localeCompare(b.dataset.name)", self.template)

    def test_needs_attention_stat_tile(self):
        self.assertIn("Needs attention", self.template)
        self.assertIn("attn_ns.count", self.template)

    def test_empty_states_use_shared_macro(self):
        self.assertIn("empty_state", self.template)
        self.assertIn('id="metricsEmpty"', self.template)


class MetricEditFormCleanupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = (ROOT / "admin" / "templates" / "client_metrics.html").read_text(encoding="utf-8")

    def test_section_dividers_present(self):
        for label in ("Identity", "Calculation", "Metadata", "Advanced"):
            self.assertIn(f'mr-edit-sect">{label}', self.template)

    def test_advanced_fields_wrapped_in_details(self):
        idx = self.template.index('mr-edit-sect">Advanced')
        tail = self.template[idx:idx + 1500]
        self.assertIn('<details class="metric-advanced"', tail)
        self.assertIn('name="required_columns"', tail)
        self.assertIn('name="base_table"', tail)

    def test_edit_form_still_has_one_builder_and_one_formula_editor(self):
        # Guardrail against accidental duplication while restructuring.
        self.assertEqual(self.template.count('class="formula-editor"'), 2)  # create + edit
        self.assertEqual(self.template.count('class="metric-builder-date-gap"'), 2)


class MetricUsageTrackingTests(unittest.TestCase):
    def setUp(self):
        self.account_id = "acct_usage_test_" + self._testMethodName
        store.save_metric(self.account_id, {
            "name": "total_revenue", "synonyms": "revenue", "sql_template": "SUM(Revenue)",
            "formula_type": "expression", "result_format": "currency",
        })

    def test_increment_bumps_usage_count(self):
        before = store.list_metrics(self.account_id, active_only=False)[0]
        self.assertEqual(before.get("usage_count") or 0, 0)
        store.increment_metric_usage(self.account_id, ["total_revenue"])
        after = store.list_metrics(self.account_id, active_only=False)[0]
        self.assertEqual(after["usage_count"], 1)
        store.increment_metric_usage(self.account_id, ["total_revenue"])
        after2 = store.list_metrics(self.account_id, active_only=False)[0]
        self.assertEqual(after2["usage_count"], 2)

    def test_empty_names_is_a_noop(self):
        store.increment_metric_usage(self.account_id, [])
        m = store.list_metrics(self.account_id, active_only=False)[0]
        self.assertEqual(m.get("usage_count") or 0, 0)

    def test_unknown_name_does_not_raise(self):
        store.increment_metric_usage(self.account_id, ["no_such_metric"])
        m = store.list_metrics(self.account_id, active_only=False)[0]
        self.assertEqual(m.get("usage_count") or 0, 0)

    def test_query_pipeline_wires_usage_increment(self):
        src = (ROOT / "core" / "query_pipeline.py").read_text(encoding="utf-8")
        idx = src.index("_matched_metrics = _metric_scope.metrics")
        following = src[idx:idx + 400]
        self.assertIn("increment_metric_usage", following)


if __name__ == "__main__":
    unittest.main()
