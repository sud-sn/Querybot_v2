"""
tests/test_query_duration_breakdown.py

Query Duration Breakdown (Snowflake-style phase bars on the admin Traces page):
  1. compute_duration_breakdown bucket aggregation (pure function)
  2. get_answer_trace_by_question_id store lookup (isolated DB)
  3. Template markers: duration bars on client_traces.html, audit-table link
     on client_detail.html
  4. Wiring guards: query_pipeline.py instrumented call sites pass duration_ms=
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Isolate DB — the store-lookup test writes real answer_trace rows and must
# never touch the real dev DB (established convention, see test_value_resolver.py).
_tmp_db = os.path.join(tempfile.mkdtemp(), "test_qdb.db")
os.environ["QUERYBOT_DB_PATH"] = _tmp_db
for mod in list(sys.modules.keys()):
    if mod.startswith("store"):
        del sys.modules[mod]
import store.db as db_mod
db_mod.init_db()

import store
from core.pipeline_trace import compute_duration_breakdown


class ComputeDurationBreakdownTests(unittest.TestCase):
    def test_single_run_sums_by_bucket(self):
        steps = [
            {"step_name": "retrieve_kb", "duration_ms": 1500},
            {"step_name": "retrieve_examples", "duration_ms": 1000},
            {"step_name": "llm_generate_sql", "duration_ms": 14000},
            {"step_name": "validate_sql", "duration_ms": 100},
            {"step_name": "execute_sql", "duration_ms": 27000},
        ]
        rows = compute_duration_breakdown(steps, total_ms=44000)
        by_label = {r["label"]: r for r in rows}
        self.assertEqual(by_label["KB retrieval"]["duration_ms"], 2500)
        self.assertEqual(by_label["SQL generation"]["duration_ms"], 14000)
        self.assertEqual(by_label["Validation"]["duration_ms"], 100)
        self.assertEqual(by_label["Execution"]["duration_ms"], 27000)
        self.assertEqual(by_label["Other"]["duration_ms"], 400)
        self.assertEqual(by_label["Total"]["duration_ms"], 44000)
        self.assertEqual(by_label["Total"]["pct"], 100.0)

    def test_retry_duplicated_step_names_accumulate(self):
        steps = [
            {"step_name": "llm_generate_sql", "duration_ms": 5000},
            {"step_name": "validate_sql", "duration_ms": 50, "status": "error"},
            {"step_name": "execute_sql", "duration_ms": 200, "status": "error"},
            # Retry path re-emits the same step names.
            {"step_name": "llm_generate_sql", "duration_ms": 6000},
            {"step_name": "validate_sql", "duration_ms": 60},
            {"step_name": "execute_sql", "duration_ms": 8000},
        ]
        rows = compute_duration_breakdown(steps, total_ms=19310)
        by_label = {r["label"]: r for r in rows}
        self.assertEqual(by_label["SQL generation"]["duration_ms"], 11000)
        self.assertEqual(by_label["Validation"]["duration_ms"], 110)
        self.assertEqual(by_label["Execution"]["duration_ms"], 8200)

    def test_unmapped_steps_fall_into_other(self):
        steps = [
            {"step_name": "receive_question", "duration_ms": 5},
            {"step_name": "route", "duration_ms": 3},
            {"step_name": "llm_generate_sql", "duration_ms": 1000},
        ]
        rows = compute_duration_breakdown(steps, total_ms=1100)
        by_label = {r["label"]: r for r in rows}
        self.assertEqual(by_label["KB retrieval"]["duration_ms"], 0)
        self.assertEqual(by_label["SQL generation"]["duration_ms"], 1000)
        self.assertEqual(by_label["Other"]["duration_ms"], 100)

    def test_field_plan_repair_folds_into_validation(self):
        steps = [
            {"step_name": "validate_sql", "duration_ms": 100, "status": "error"},
            {"step_name": "field_plan_repair", "duration_ms": 40},
            {"step_name": "validate_sql", "duration_ms": 80},
        ]
        rows = compute_duration_breakdown(steps, total_ms=220)
        by_label = {r["label"]: r for r in rows}
        self.assertEqual(by_label["Validation"]["duration_ms"], 220)
        self.assertEqual(by_label["Other"]["duration_ms"], 0)

    def test_other_clamped_at_zero_when_buckets_exceed_total(self):
        # Rounding/measurement drift could push bucket sum above the stored total.
        steps = [
            {"step_name": "llm_generate_sql", "duration_ms": 5000},
            {"step_name": "execute_sql", "duration_ms": 6000},
        ]
        rows = compute_duration_breakdown(steps, total_ms=1000)
        by_label = {r["label"]: r for r in rows}
        self.assertEqual(by_label["Other"]["duration_ms"], 0)
        self.assertGreaterEqual(by_label["Other"]["pct"], 0.0)

    def test_empty_steps_no_crash(self):
        rows = compute_duration_breakdown([], total_ms=0)
        by_label = {r["label"]: r for r in rows}
        self.assertEqual(by_label["Total"]["duration_ms"], 0)
        for label in ("KB retrieval", "SQL generation", "Validation", "Execution", "Other"):
            self.assertEqual(by_label[label]["duration_ms"], 0)

    def test_missing_duration_ms_defaults_to_zero(self):
        steps = [{"step_name": "llm_generate_sql"}, {"step_name": "execute_sql", "duration_ms": None}]
        rows = compute_duration_breakdown(steps, total_ms=0)
        by_label = {r["label"]: r for r in rows}
        self.assertEqual(by_label["SQL generation"]["duration_ms"], 0)
        self.assertEqual(by_label["Execution"]["duration_ms"], 0)

    def test_bucket_order_is_stable(self):
        rows = compute_duration_breakdown([], total_ms=0)
        labels = [r["label"] for r in rows]
        self.assertEqual(
            labels,
            ["KB retrieval", "SQL generation", "Validation", "Execution", "Other", "Total"],
        )


class GetAnswerTraceByQuestionIdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for acct in ("acct-1", "acct-2", "acct-3", "acct-4", "other-acct"):
            store.upsert_client(acct, "portal")

    def test_returns_none_for_unknown_question(self):
        self.assertIsNone(store.get_answer_trace_by_question_id("acct-1", "does-not-exist"))

    def test_returns_none_for_empty_question_id(self):
        self.assertIsNone(store.get_answer_trace_by_question_id("acct-1", ""))

    def test_found_by_question_id(self):
        trace_id = store.create_answer_trace(
            account_id="acct-2", question_id="q-abc", question_text="how many orders",
        )
        found = store.get_answer_trace_by_question_id("acct-2", "q-abc")
        self.assertIsNotNone(found)
        self.assertEqual(found["id"], trace_id)
        self.assertEqual(found["question_id"], "q-abc")

    def test_wrong_account_id_does_not_match(self):
        store.create_answer_trace(
            account_id="acct-3", question_id="q-xyz", question_text="revenue this month",
        )
        self.assertIsNone(store.get_answer_trace_by_question_id("other-acct", "q-xyz"))

    def test_picks_most_recent_when_duplicates_exist(self):
        store.create_answer_trace(
            account_id="acct-4", question_id="q-dup", question_text="first ask",
        )
        newest_id = store.create_answer_trace(
            account_id="acct-4", question_id="q-dup", question_text="re-asked",
        )
        found = store.get_answer_trace_by_question_id("acct-4", "q-dup")
        self.assertEqual(found["id"], newest_id)


class TemplateMarkerTests(unittest.TestCase):
    def setUp(self):
        self.traces_html = (ROOT / "admin" / "templates" / "client_traces.html").read_text(encoding="utf-8")
        self.detail_html = (ROOT / "admin" / "templates" / "client_detail.html").read_text(encoding="utf-8")

    def test_duration_bar_css_present(self):
        for cls in (".duration-row", ".duration-wrap", ".duration-fill", ".duration-val", ".duration-total"):
            self.assertIn(cls, self.traces_html)

    def test_duration_panel_markup_present(self):
        self.assertIn("duration_breakdown", self.traces_html)
        self.assertIn('class="duration-row', self.traces_html)
        self.assertIn("Query duration", self.traces_html)

    def test_audit_table_links_to_traces_by_question_id(self):
        self.assertIn("/traces?question_id=", self.detail_html)
        self.assertIn("View query duration breakdown", self.detail_html)


class QueryPipelineWiringGuardTests(unittest.TestCase):
    def setUp(self):
        self.src = (ROOT / "core" / "query_pipeline.py").read_text(encoding="utf-8")

    def test_instrumented_call_sites_pass_duration_ms(self):
        # Every phase this feature instruments must actually measure and pass
        # duration_ms=, not rely on the store-layer default of 0.
        markers = [
            'duration_ms=int((time.time() - _kb_phase_t0) * 1000)',
            'duration_ms=int((time.time() - _examples_t0) * 1000)',
            'duration_ms=int((time.time() - _llm_gen_t0) * 1000)',
            'duration_ms=int((time.time() - _repair_t0) * 1000)',
            'duration_ms=_validate_ms',
            'duration_ms=int((time.time() - _exec_t0) * 1000)',
            'duration_ms=int((time.time() - _retry_llm_t0) * 1000)',
            'duration_ms=int((time.time() - _retry_validate_t0) * 1000)',
            'duration_ms=int((time.time() - _retry_exec_t0) * 1000)',
        ]
        for marker in markers:
            self.assertIn(marker, self.src, f"missing wiring: {marker}")

    def test_retry_path_has_trace_steps(self):
        # Before this feature the retry path had zero _trace_step calls.
        self.assertIn('_trace_step(trace_id, "llm_generate_sql", output_summary={"retry": True}', self.src)
        self.assertIn('"retry": True', self.src)


if __name__ == "__main__":
    unittest.main()
