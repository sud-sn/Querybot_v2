"""
tests/test_remaining_ledger_findings.py

The last 23 findings from the fault audit. None had been reported from the
field, which is the point: every one of them degrades an answer without
producing an error, so nobody would report them until the number was wrong
enough to notice.

Four themes:

  Silent failure   — a guard fails, is logged below the default level, and its
                     absence costs the answer nothing, so a failed check scores
                     the same as a passed one.
  Stale dates      — a bound that a config change silently removes, a freshness
                     read taken and discarded, and the wall clock permitted
                     wherever governance happens to be absent.
  Frozen values    — distinct values, PII verdicts, join percentages and schema
                     trees measured once and then presented as present tense.
  Suggestions      — chips promoted on evidence that does not mean what its
                     name says, and failures that never reach the ranker.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest


# ── Silent failure ───────────────────────────────────────────────────────────


class TestAFailedCheckCostsSomething:
    def test_unverifiable_result_shape_is_not_scored_as_verified(self):
        """Verification contributes +5 when it passes and used to contribute
        nothing when it could not run — so the stated defence against a
        schema-valid but business-wrong answer reported no issues precisely
        because it never looked."""
        from core.answer_confidence import build_answer_confidence

        unavailable = build_answer_confidence(
            validation_code="ok", row_count=5,
            result_verification={"status": "unavailable"},
        )
        passed = build_answer_confidence(
            validation_code="ok", row_count=5,
            result_verification={"status": "pass"},
        )
        assert unavailable["score"] < passed["score"]
        assert any("shape of the question" in w for w in unavailable["warnings"])

    def test_unscored_retrieval_is_not_scored_as_relevant(self):
        """With no cross-encoder the relevance floor cannot run, so
        weak_retrieval can never be raised — the penalty and the filter go
        together, silently, for the whole process lifetime."""
        from core.answer_confidence import build_answer_confidence

        unscored = build_answer_confidence(
            validation_code="ok", row_count=5, retrieval_unscored=True,
        )
        scored = build_answer_confidence(validation_code="ok", row_count=5)
        assert unscored["score"] < scored["score"]
        assert any("could not be scored" in w for w in unscored["warnings"])

    def test_the_retriever_records_that_the_floor_did_not_run(self):
        import core.vector_store as vs

        retriever = vs.QdrantKBRetriever.__new__(vs.QdrantKBRetriever)
        retriever.last_retrieval_weak = False
        retriever.last_retrieval_unscored = False
        retriever._apply_relevance_floor([
            {"fqn": "DW.F_SALES", "content": "x"},
            {"fqn": "DW.D_CUST", "content": "y"},
        ])
        assert retriever.last_retrieval_unscored is True

    def test_a_scored_retrieval_is_not_flagged_unscored(self):
        import core.vector_store as vs

        retriever = vs.QdrantKBRetriever.__new__(vs.QdrantKBRetriever)
        retriever.last_retrieval_weak = False
        retriever.last_retrieval_unscored = False
        retriever._apply_relevance_floor([
            {"fqn": "DW.F_SALES", "content": "x", "_rerank_score": 0.9},
        ])
        assert retriever.last_retrieval_unscored is False

    def test_the_portal_suggestion_panel_reports_its_own_failure(self):
        """A bare `except Exception: return []` with no log at all: an empty
        panel was indistinguishable from a workspace with no suggestions."""
        import inspect

        import portal.routes as routes

        source = inspect.getsource(routes._build_chat_suggestions)
        assert "log.error(" in source
        assert "exc_info=True" in source

    def test_plan_reuse_does_not_depend_on_unrelated_traffic(self):
        """The question filter ran in Python over the 100 most recent rows, so
        the same question reused its plan on a quiet workspace and silently
        re-generated on a busy one."""
        import inspect

        import store.trace_store as trace_store

        source = inspect.getsource(trace_store.find_reusable_validated_sql_plan)
        assert "qb_normalized_question(q.question) = ?" in source
        assert "create_function" in source


# ── Stale dates ──────────────────────────────────────────────────────────────


class TestTheStalenessBoundCannotBeSwitchedOffByAccident:
    def test_a_long_ttl_does_not_outlive_the_max_age(self, monkeypatch):
        """Raising the probe interval for performance is reasonable. It used to
        disable the only staleness check there is, because a live in-memory hit
        returns before the stored anchor's age is ever examined."""
        import core.date_anchor as da

        monkeypatch.setenv("QUERYBOT_DATE_ANCHOR_TTL_SECONDS", "86400")
        monkeypatch.setenv("QUERYBOT_DATE_ANCHOR_MAX_AGE_SECONDS", "3600")
        policy = {"fact_table": "DW.F", "fact_column": "D", "date_column": "D",
                  "anchor_policy": "latest_available"}

        da.clear_cache()
        stamp = (datetime.utcnow() - timedelta(seconds=3000)).isoformat(sep=" ")
        da.remember_anchor("acct", policy, {"value": "2025-04-17", "resolved_at": stamp})
        entry = da._cache.get(da.anchor_key("acct", policy))
        assert entry is not None
        remaining = entry["expires_at"] - __import__("time").time()
        assert remaining <= 601, (
            f"cached for {remaining:.0f}s — the 1h staleness bound was ignored"
        )
        da.clear_cache()

    def test_an_already_stale_anchor_is_not_cached_at_all(self, monkeypatch):
        import core.date_anchor as da

        monkeypatch.setenv("QUERYBOT_DATE_ANCHOR_TTL_SECONDS", "86400")
        monkeypatch.setenv("QUERYBOT_DATE_ANCHOR_MAX_AGE_SECONDS", "3600")
        policy = {"fact_table": "DW.F", "fact_column": "D", "date_column": "D",
                  "anchor_policy": "latest_available"}

        da.clear_cache()
        stamp = (datetime.utcnow() - timedelta(seconds=7200)).isoformat(sep=" ")
        da.remember_anchor("acct", policy, {"value": "2025-04-17", "resolved_at": stamp})
        assert da.cached_anchor("acct", policy) == {}
        da.clear_cache()


class TestTheWallClockIsNotPermittedWhereGovernanceIsAbsent:
    KNOWN = {"DB.dbo.SALES_FCT"}
    COLS = {"DB.dbo.SALES_FCT": {"INV_DT": "date", "AMT": "decimal"}}
    CLOCK = ("SELECT SUM(AMT) FROM DB.dbo.SALES_FCT "
             "WHERE INV_DT = CAST(GETDATE() AS DATE)")

    def _validate(self, sql, context):
        from core.validator import validate_sql_detailed

        return validate_sql_detailed(sql, self.KNOWN, "azure_sql", None, self.COLS, context)

    def test_a_relative_question_with_no_date_role_refuses_the_clock(self):
        """No approved date role means no policy to enforce — and the early
        return turned that absence of governance into permission. Oracle
        rejected GETDATE as a dialect error; Azure SQL accepted it, so the same
        question was governed on one warehouse and not on another."""
        result = self._validate(self.CLOCK, {
            "question": "what is my revenue today",
            "semantic_plan": {"temporal_policies": []},
        })
        assert result.ok is False
        assert result.code == "temporal_anchor_ungoverned"

    def test_a_question_with_no_relative_period_is_unaffected(self):
        result = self._validate(self.CLOCK, {
            "question": "list all customers",
            "semantic_plan": {"temporal_policies": []},
        })
        assert result.ok is True

    def test_clock_free_sql_passes(self):
        result = self._validate(
            "SELECT SUM(AMT) FROM DB.dbo.SALES_FCT",
            {"question": "what is my revenue today", "semantic_plan": {"temporal_policies": []}},
        )
        assert result.ok is True


class TestTheFreshnessReadIsUsedRatherThanDiscarded:
    def test_the_coverage_message_states_the_date_the_data_runs_through(self):
        from core.date_coverage import CoverageGap

        assert "observed_through" in CoverageGap.__dataclass_fields__

    def test_a_period_grained_source_is_no_longer_skipped_silently(self):
        """The anchor read was skipped entirely for exactly the windows most
        likely to be stale — which month the data actually runs through is what
        a 'this month' question needs to know."""
        import inspect

        import core.date_coverage as dc

        source = inspect.getsource(dc.check_date_coverage)
        assert "counts_days" in source
        assert "if not counts_days:" in source

    def test_a_failed_freshness_probe_is_not_a_confirmed_fresh_answer(self):
        import inspect

        import core.date_coverage as dc

        source = inspect.getsource(dc.check_date_coverage)
        assert 'log.debug("Date coverage check skipped' not in source
        assert "log.warning(" in source


# ── Frozen values ────────────────────────────────────────────────────────────


class TestASnapshotSaysWhenItWasTaken:
    def test_the_schema_document_is_dated(self):
        from core.schema import _observed_at_note

        note = _observed_at_note()
        assert "Schema observed:" in note
        assert "snapshot, not a live view" in note

    def test_the_kb_writer_is_not_told_the_value_list_is_closed(self):
        from pathlib import Path

        source = Path("core/llm.py").read_text(encoding="utf-8")
        assert "never as the complete or only" in source

    def test_a_join_percentage_is_reported_with_its_measurement_date(self):
        from core.join_coverage import _coverage_message

        fresh = _coverage_message(
            "Sales", "Customer", 23.4,
            (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S"),
        )
        assert "measured 2 days ago" in fresh

    def test_an_old_join_percentage_is_reported_in_the_past_tense(self):
        from core.join_coverage import _coverage_message

        old = _coverage_message(
            "Sales", "Customer", 23.4,
            (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d %H:%M:%S"),
        )
        assert "excluded about 23%" in old
        assert "Re-profile" in old

    def test_an_undated_join_percentage_says_so(self):
        from core.join_coverage import _coverage_message

        assert "undated" in _coverage_message("Sales", "Customer", 23.4, "")

    def test_the_masking_verdict_records_its_own_provenance(self):
        from pathlib import Path

        source = Path("core/schema.py").read_text(encoding="utf-8")
        assert '"mask_observed_at"' in source
        assert '"mask_sample_rows"' in source

    def test_a_thin_pii_sample_is_reported(self):
        from pathlib import Path

        source = Path("core/schema.py").read_text(encoding="utf-8")
        assert "_MASK_SCAN_MIN_ROWS" in source

    def test_the_schema_tree_invalidator_has_production_callers(self):
        """bust_cache() was written for exactly this and had none — an admin
        who added a table and re-ran Discover kept seeing yesterday's list."""
        from pathlib import Path

        source = Path("admin/routes.py").read_text(encoding="utf-8")
        assert source.count("bust_cache as _bust_tree") == 2

    def test_the_distinct_value_cache_notices_a_rediscovery(self):
        """Keyed on the directory PATH, which a re-Discover does not change."""
        import inspect

        import core.clarification as clarification

        source = inspect.getsource(clarification._schema_distinct_value_candidates)
        assert "_schema_dir_fingerprint(schema_dir)" in source

    def test_a_fingerprint_moves_when_a_schema_file_is_rewritten(self):
        import time

        from core.clarification import _schema_dir_fingerprint

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "_schema.json"
            target.write_text("{}", encoding="utf-8")
            first = _schema_dir_fingerprint(tmp)
            time.sleep(0.01)
            os.utime(target, (first + 10, first + 10))
            assert _schema_dir_fingerprint(tmp) != first


class TestAnExampleDoesNotCarryLastSpringsDate:
    def test_date_literals_are_replaced(self):
        from core.examples import scrub_example_date_literals as scrub

        assert scrub("SELECT 1 FROM F WHERE D = '2025-04-17'").endswith("'<date>'")
        assert "'<date>'" in scrub("SELECT 1 FROM F WHERE D BETWEEN '2024-01-01' AND '2024-12-31'")
        assert scrub("SELECT 1 FROM F WHERE DK = 20250417").endswith("<date_key>")
        assert scrub("SELECT 1 FROM F WHERE P = 202504").endswith("<date_key>")

    def test_non_date_values_are_untouched(self):
        from core.examples import scrub_example_date_literals as scrub

        assert scrub("SELECT 1 FROM F WHERE STATUS = 'Denied'").endswith("'Denied'")
        assert scrub("SELECT 1 FROM F WHERE ORDER_NO = 12345678").endswith("12345678")

    def test_the_prompt_no_longer_calls_them_currently_tested(self):
        from core.examples import format_examples_for_prompt

        block = format_examples_for_prompt([
            {"question": "revenue today", "sql": "SELECT 1 FROM F WHERE D = '2025-04-17'"},
        ])
        assert "'<date>'" in block
        assert "at some point in the past" in block
        assert "never from an example" in block


# ── Suggestions ──────────────────────────────────────────────────────────────


class TestASuggestionIsCheckedAgainstWhatWillPlanIt:
    def test_the_gate_reads_the_subgraph_the_pipeline_plans_on(self):
        """The gate read every relationship including unreviewed ones; the
        pipeline narrows to confirmed. A question joinable only through a
        suggested edge passed the gate and was refused after the click."""
        import inspect

        import core.suggestions as suggestions

        source = inspect.getsource(suggestions._graph_reachability_check)
        assert "_confirmed_subgraph" in source
        assert "_client_allows_suggested" in source
        assert '"broken"' in source

    def test_proven_examples_outrank_merely_compiled_ones(self):
        """kb_stage2 SQL was COMPILE-checked, which proves the columns resolve
        and nothing about whether the question returns anything. query_log
        examples actually came back with rows."""
        import inspect

        import core.suggestions as suggestions

        source = inspect.getsource(suggestions.get_suggestions)
        assert 'if str(ex.get("source") or "") == "query_log" else 1' in source

    def test_a_date_question_needs_an_approved_date_role(self):
        from core.suggestions import _date_scope_check

        with tempfile.TemporaryDirectory() as tmp:
            check = _date_scope_check(tmp)
            assert check("what was our revenue last month") is False
            assert check("how many customers do we have") is True

    def test_the_date_gate_fails_open_when_it_cannot_run_at_all(self, monkeypatch):
        """No approved date role is a real answer — withhold. The check itself
        being unavailable is not, and must never withhold on a guess."""
        import core.semantic_model as sm
        from core.suggestions import _date_scope_check

        def _boom(*_a, **_k):
            raise RuntimeError("model unreadable")

        monkeypatch.setattr(sm, "find_default_date_roles", _boom)
        assert _date_scope_check("any-dir")("revenue last month") is True

    def test_a_workspace_with_a_date_role_offers_date_questions(self):
        import core.semantic_model as sm
        from core.suggestions import _date_scope_check

        original = sm.find_default_date_roles
        sm.find_default_date_roles = lambda *_a, **_k: [{"fact_table": "DW.F"}]
        try:
            assert _date_scope_check("any-dir")("revenue last month") is True
        finally:
            sm.find_default_date_roles = original

    def test_stage_two_questions_are_pruned_by_their_validation_result(self):
        """The cache is written from every Q:/SQL: pair BEFORE validation runs,
        and nothing revisited it with the results."""
        from core.suggestions import _CACHE_FILENAME, prune_suggestion_cache

        import json as _json
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / _CACHE_FILENAME).write_text(_json.dumps([
                {"question": "Revenue by region", "fqn": "DW.F_SALES"},
                {"question": "This one did not compile", "fqn": "DW.F_SALES"},
            ]), encoding="utf-8")
            removed = prune_suggestion_cache(tmp, {"Revenue by region"})
            assert removed == 1
            kept = _json.loads((Path(tmp) / _CACHE_FILENAME).read_text(encoding="utf-8"))
            assert [e["question"] for e in kept] == ["Revenue by region"]

    def test_pruning_with_no_validated_set_changes_nothing(self):
        from core.suggestions import prune_suggestion_cache

        with tempfile.TemporaryDirectory() as tmp:
            assert prune_suggestion_cache(tmp, set()) == 0

    def test_the_pruner_is_called_after_validation(self):
        from pathlib import Path

        source = Path("core/dispatcher.py").read_text(encoding="utf-8")
        assert "prune_suggestion_cache(" in source

    def test_the_portal_filler_is_gated_and_acl_checked(self):
        """It ran after get_suggestions() and was gated on nothing, and its own
        phrase filter is skipped entirely when the glossary is empty — the
        state a new workspace is in."""
        import inspect

        import portal.routes as routes

        source = inspect.getsource(routes._build_chat_suggestions)
        assert "_graph_reachability_check" in source
        assert "if not reachable(q):" in source
        assert hasattr(routes, "_metrics_within_acl")


class TestTheRankerLearnsWhatActuallyHappened:
    HTML = Path("portal/templates/portal_chat.html")

    def test_a_text_refusal_is_not_credited_as_a_success(self):
        """A refusal arrives on the same branch as a text answer. Crediting it
        ranked the chips that dead-ended above the ones that worked."""
        source = self.HTML.read_text(encoding="utf-8")
        assert "if (msg.trust && msg.trust.sql) {" in source
        assert "_genieEvent('successful',  _pendingSuggestion);" not in source

    def test_a_failure_reaches_the_ranker(self):
        """Recording nothing meant failures were invisible while every success
        accumulated, so a chip that fails for everyone kept its position."""
        source = self.HTML.read_text(encoding="utf-8")
        assert "if (_pendingSuggestion) _genieEvent('executed', _pendingSuggestion);" in source


class TestAClickedQuestionIsPlannedLikeATypedOne:
    def test_the_table_hint_is_a_naming_hint_not_a_table_choice(self):
        from pathlib import Path

        source = Path("core/query_pipeline.py").read_text(encoding="utf-8")
        assert "TABLE NAME FORMAT:" in source
        assert "does NOT restrict which tables" in source
        assert "SCHEMA HINT: This question is about the table" not in source


class TestFollowUpChipsAreActuallyGrounded:
    def test_signals_are_computed_where_the_rows_are(self):
        """compute_signals() needs rows, and rows stop at the PII boundary
        before the chip generator — so the statistical tier was unreachable and
        every chip came from the model, ungrounded, while the comments claimed
        otherwise."""
        from core.result_renderer import _result_signals

        rows = [
            {"REGION": "North", "REVENUE": 100.0},
            {"REGION": "South", "REVENUE": 105.0},
            {"REGION": "West", "REVENUE": 4200.0},
        ]
        assert [s["type"] for s in _result_signals(rows)]

    def test_signal_detection_never_raises(self):
        from core.result_renderer import _result_signals

        assert _result_signals([]) == []
        assert _result_signals(None) == []

    def test_the_generator_accepts_and_uses_them(self):
        import inspect

        import core.insight as insight

        assert "signals" in inspect.signature(
            insight.generate_followup_suggestions
        ).parameters
        source = inspect.getsource(insight.generate_followup_suggestions)
        assert "template_suggestions" in source

    def test_they_are_passed_from_the_renderer(self):
        from pathlib import Path

        source = Path("core/result_renderer.py").read_text(encoding="utf-8")
        assert "signals=_result_signals(rows)," in source


# ── Replayed rows ────────────────────────────────────────────────────────────


class TestARestoredResultIsNotSilentlyYesterdays:
    def test_the_restore_is_age_bounded(self):
        import gateway.webhooks as webhooks

        assert webhooks._RESTORE_MAX_AGE_HOURS > 0
        source = __import__("inspect").getsource(webhooks._restore_durable_thread_result)
        assert "_RESTORE_MAX_AGE_HOURS" in source

    @pytest.mark.parametrize("stamp,expected", [
        ("", None),
        ("not a date", None),
    ])
    def test_an_unreadable_timestamp_does_not_block_the_restore(self, stamp, expected):
        """None means "cannot tell" and leaves existing behaviour alone; only a
        readable, genuinely old timestamp withholds."""
        from gateway.webhooks import _trace_age_hours

        assert _trace_age_hours({"created_at": stamp}) is expected

    def test_a_day_old_result_reads_as_old(self):
        from gateway.webhooks import _trace_age_hours

        stamp = (datetime.now() - timedelta(hours=30)).strftime("%Y-%m-%d %H:%M:%S")
        age = _trace_age_hours({"created_at": stamp})
        assert age is not None and age > 24
