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
        #
        # Validation, execution and the deterministic repairs moved into
        # core.sql_attempt, so their markers are asserted behaviourally below
        # rather than as source text -- which is stronger: a marker present in
        # a phase that no longer runs would still pass this list.
        markers = [
            'duration_ms=int((time.time() - _kb_phase_t0) * 1000)',
            'duration_ms=int((time.time() - _examples_t0) * 1000)',
            'duration_ms=int((time.time() - _llm_gen_t0) * 1000)',
            'duration_ms=int((time.time() - _retry_llm_t0) * 1000)',
            'duration_ms=int((time.time() - _retry_validate_t0) * 1000)',
            'duration_ms=int((time.time() - _retry_exec_t0) * 1000)',
        ]
        for marker in markers:
            self.assertIn(marker, self.src, f"missing wiring: {marker}")
        # The extracted phases still reach the trace with a measured duration.
        for marker in ('duration_ms=attempt.validate_ms',
                       'duration_ms=_attempt.execute_ms',
                       'duration_ms=duration_ms'):
            self.assertIn(marker, self.src, f"missing wiring: {marker}")

    def test_validation_measures_its_own_duration(self):
        # A slow validator, so the assertion discriminates: >= 0 passes for a
        # hardcoded zero, which is the exact defect this whole test class
        # exists to catch.
        import time as _time
        from unittest.mock import patch

        from core.sql_attempt import ValidationScope, validate_with_repairs

        def _slow_validate(*args, **kwargs):
            _time.sleep(0.01)
            return (True, "OK", "ok")

        scope = ValidationScope(known_tables={"S.T"}, db_type="azure_sql")
        with patch("core.validator.validate_sql", _slow_validate):
            attempt = validate_with_repairs("SELECT 1", scope)
        self.assertGreaterEqual(attempt.validate_ms, 10)

    def test_a_repair_reports_how_long_it_took(self):
        # A repair step recorded at the store layer's default of zero is a
        # phase that vanishes from the duration breakdown -- which is exactly
        # what the extraction dropped on its first pass.
        import time as _time
        from unittest.mock import patch

        from core.sql_attempt import ValidationScope, validate_with_repairs

        recorded = []

        def _slow_repair(*args, **kwargs):
            _time.sleep(0.01)
            return "SELECT 2"

        scope = ValidationScope(known_tables={"S.T"}, db_type="azure_sql")
        with (
            patch("core.validator.validate_sql",
                  return_value=(False, "planned", "field_plan_mismatch")),
            patch("core.pipeline_helpers.attempt_governed_temporal_metric_repair",
                  return_value=""),
            patch("core.pipeline_helpers.attempt_field_plan_repair", _slow_repair),
        ):
            validate_with_repairs(
                "SELECT 1", scope,
                on_repair=lambda *a: recorded.append(a))

        self.assertEqual(len(recorded), 1)
        kind, before, after, metadata, duration_ms = recorded[0]
        self.assertEqual(kind, "field_plan_repair")
        self.assertEqual(after, "SELECT 2")
        self.assertGreaterEqual(duration_ms, 10)

    def test_execution_measures_its_own_duration(self):
        import asyncio
        import time as _time
        from types import SimpleNamespace

        from core.sql_attempt import Attempt, ValidationScope, execute

        def _slow_executor(sql, semantic):
            _time.sleep(0.01)
            return SimpleNamespace(rows=[{"A": 1}], sql=sql, truncated=False)

        attempt = Attempt(sql="SELECT 1", ok=True)
        done = asyncio.run(execute(
            attempt, executor=_slow_executor, semantic_context=None,
            timeout=5, timeout_message="timed out",
        ))
        self.assertGreaterEqual(done.execute_ms, 10)
        self.assertEqual(done.rows, [{"A": 1}])

    def test_retry_path_has_trace_steps(self):
        # Before this feature the retry path had zero _trace_step calls.
        self.assertIn('_trace_step(trace_id, "llm_generate_sql", output_summary={"retry": True}', self.src)
        self.assertIn('"retry": True', self.src)


if __name__ == "__main__":
    unittest.main()
