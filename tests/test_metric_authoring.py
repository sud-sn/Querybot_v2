"""
tests/test_metric_authoring.py

Phase 4: a user describes a calculation, it answers their question now, and a
request goes to an admin to make it shared.

The safety argument is that the model never writes SQL. It fills structured
slots and references columns by tokens this process issued; the formula is
compiled locally. So most of these tests are about what the model CANNOT do.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

_tmp_db = os.path.join(tempfile.mkdtemp(), "test_metric_authoring.db")
os.environ["QUERYBOT_DB_PATH"] = _tmp_db
for _mod in list(sys.modules):
    if _mod.startswith("store"):
        del sys.modules[_mod]

import store  # noqa: E402

from core.metric_authoring import (  # noqa: E402
    build_metric_plan_input,
    compile_metric_plan_response,
)

store.init_db()

ROOT = Path(__file__).resolve().parents[1]

MANIFEST = {
    "EMDW_DMART.CUS_ORD_IVC_FCT": ["IVC_AMT", "CUS_DMS_KEY", "INVOICE_DT_DMS_KEY"],
    "EMDW_DMART.CUS_DMS": ["CUS_NO", "CUS_NM", "STATUS_CD"],
}


@pytest.fixture
def plan_input():
    return build_metric_plan_input("revenue per active customer", MANIFEST, ["Net Revenue"])


def _plan(**overrides):
    plan = {
        "operation": "define_metric", "name": "Revenue Per Active Customer",
        "mode": "ratio", "result_format": "currency", "base_table_ref": "TABLE_REF_1",
        "numerator": {"aggregation": "SUM", "measure_ref": "COL_REF_1"},
        "denominator": {"aggregation": "COUNT_DISTINCT", "measure_ref": "COL_REF_4"},
        "confidence": 0.9,
    }
    plan.update(overrides)
    return json.dumps(plan)


# ── The model composes; it does not write SQL ────────────────────────────────


class TestTheModelFillsSlotsAndNeverWritesSql:
    def test_a_ratio_compiles_from_structured_choices(self, plan_input):
        draft, error = compile_metric_plan_response(_plan(), plan_input)
        assert error == ""
        assert draft.sql_template == (
            "SUM(IVC_AMT) * 1.0 / NULLIF(COUNT(DISTINCT CUS_NO), 0)"
        )
        assert draft.required_columns == "IVC_AMT, CUS_NO"
        assert draft.base_table == "EMDW_DMART.CUS_ORD_IVC_FCT"

    def test_the_tables_it_touches_are_recorded(self, plan_input):
        """These drive the ACL re-check on every later read."""
        draft, _ = compile_metric_plan_response(_plan(), plan_input)
        assert set(draft.source_tables) == {
            "EMDW_DMART.CUS_ORD_IVC_FCT", "EMDW_DMART.CUS_DMS",
        }

    def test_there_is_no_plan_key_that_can_carry_sql(self):
        """The existing AI-import route accepts a row_expression string from the
        model. There is deliberately no equivalent here — nothing to sanitise
        because nothing can arrive."""
        from core.metric_authoring import _PLAN_KEYS

        for suspicious in ("sql", "sql_template", "formula", "row_expression", "expression"):
            assert suspicious not in _PLAN_KEYS


class TestWhatTheModelCannotDo:
    @pytest.mark.parametrize("bad_ref", ["SECRET_SALARY", "COL_REF_99", "", None, "DROP TABLE"])
    def test_a_column_it_invents_is_refused(self, plan_input, bad_ref):
        draft, error = compile_metric_plan_response(
            _plan(mode="aggregate", aggregation="SUM", measure_ref=bad_ref,
                  numerator=None, denominator=None).replace('"numerator": null, ', "")
                .replace('"denominator": null, ', ""),
            plan_input,
        )
        assert draft is None and "not available to you" in error

    def test_an_unknown_plan_field_is_refused(self, plan_input):
        draft, error = compile_metric_plan_response(
            json.dumps({"operation": "define_metric", "name": "X", "raw_sql": "SELECT 1"}),
            plan_input,
        )
        assert draft is None and "unsupported fields" in error

    def test_an_unsupported_aggregation_is_refused(self, plan_input):
        draft, error = compile_metric_plan_response(
            _plan(numerator={"aggregation": "EXEC", "measure_ref": "COL_REF_1"}), plan_input,
        )
        assert draft is None and "not a supported" in error

    def test_an_unsupported_filter_operator_is_refused(self, plan_input):
        draft, error = compile_metric_plan_response(
            _plan(numerator={
                "aggregation": "SUM", "measure_ref": "COL_REF_1",
                "filters": [{"field_ref": "COL_REF_6", "operator": "; DROP", "value": "x"}],
            }), plan_input,
        )
        assert draft is None and "filter operator" in error

    def test_a_filter_value_is_escaped_not_executed(self, plan_input):
        """A value is a literal. Quotes in it are doubled, so it filters on a
        strange-looking string rather than changing the statement."""
        draft, _ = compile_metric_plan_response(
            _plan(numerator={
                "aggregation": "SUM", "measure_ref": "COL_REF_1",
                "filters": [{"field_ref": "COL_REF_6", "operator": "equals",
                             "value": "' OR 1=1--"}],
            }), plan_input,
        )
        assert "''' OR 1=1--'" in draft.sql_template

    def test_a_composed_metric_cannot_reference_another_metric(self, plan_input):
        """${Name} resolves against list_metrics, which cannot see a session
        draft — it would raise inside a try/except and ship the unresolved
        literal into the prompt."""
        from core.metric_authoring import compile_metric_plan_response as compile_

        # Reach the guard directly: no legitimate plan can produce ${...},
        # which is exactly why the guard is cheap to keep.
        from core.metric_authoring import MetricDraft  # noqa: F401
        source = (ROOT / "core" / "metric_authoring.py").read_text(encoding="utf-8")
        assert 'if "${" in compiled.formula' in source

    def test_declining_is_a_normal_outcome(self, plan_input):
        draft, error = compile_metric_plan_response(
            json.dumps({"operation": "unsupported"}), plan_input,
        )
        assert draft is None and "could not be built" in error

    @pytest.mark.parametrize("raw", ["not json", "", "[]", "null", "```json\n{bad}\n```"])
    def test_malformed_output_is_refused(self, plan_input, raw):
        draft, error = compile_metric_plan_response(raw, plan_input)
        assert draft is None and error


class TestConfidenceFailsClosed:
    def test_a_missing_confidence_is_zero_not_certain(self, plan_input):
        plan = json.loads(_plan())
        plan.pop("confidence")
        draft, _ = compile_metric_plan_response(json.dumps(plan), plan_input)
        assert draft.confidence == 0.0

    @pytest.mark.parametrize("value", ["high", None, float("nan"), -3, 99])
    def test_malformed_confidence_never_reads_as_high(self, plan_input, value):
        draft, _ = compile_metric_plan_response(_plan(confidence=value), plan_input)
        assert 0.0 <= draft.confidence <= 1.0

    def test_the_gate_is_above_the_fail_closed_default(self):
        """0.0 must not pass the handler's threshold, or failing closed would
        fail open."""
        webhooks = (ROOT / "gateway" / "webhooks.py").read_text(encoding="utf-8")
        assert "draft.confidence < 0.6" in webhooks


# ── The prompt carries names, never values ───────────────────────────────────


class TestTheEgressBoundary:
    def test_the_prompt_contains_no_row_data(self, plan_input):
        """Table and column NAMES only. A sample value here would need an entry
        in llm_audit's _VALUE_BEARING_MARKERS, and the module docstring says so."""
        blob = plan_input.system_prompt + plan_input.user_prompt
        assert "IVC_AMT" in blob and "CUS_ORD_IVC_FCT" in blob
        for value_ish in ("Nova Scotia", "ACTIVE'", "£", "$"):
            assert value_ish not in blob

    def test_the_audit_component_is_named(self):
        webhooks = (ROOT / "gateway" / "webhooks.py").read_text(encoding="utf-8")
        assert 'component="metric_authoring_chat"' in webhooks


# ── Wiring the two tracks ────────────────────────────────────────────────────


class TestTheTwoTracks:
    WEBHOOKS = (ROOT / "gateway" / "webhooks.py").read_text(encoding="utf-8")

    def test_the_acl_filter_precedes_the_prompt(self):
        """A table this user cannot see must never be offered to the model as an
        option. Same ordering the report builder enforces."""
        handler = self.WEBHOOKS[self.WEBHOOKS.index("async def _run_metric_authoring_chat"):]
        handler = handler[: handler.index("async def _run_report_builder_chat")]
        assert handler.index("get_allowed_tables") < handler.index("parse_metric_plan")

    def test_the_question_is_still_answered(self):
        """Track (a). Without this the user describes a calculation and gets a
        card instead of an answer."""
        handler = self.WEBHOOKS[self.WEBHOOKS.index("async def _run_metric_authoring_chat"):]
        handler = handler[: handler.index("async def _run_report_builder_chat")]
        assert "_run_main_question(text, table_hint, schema_hint)" in handler

    def test_every_gate_falls_through_rather_than_erroring(self):
        """A false intent match must cost one planner call, not an error page."""
        handler = self.WEBHOOKS[self.WEBHOOKS.index("async def _run_metric_authoring_chat"):]
        handler = handler[: handler.index("async def _run_report_builder_chat")]
        assert handler.count("_fall_through(") >= 7

    def test_the_chat_handler_never_writes_a_metric(self):
        """The governance rule, asserted where it can be broken.

        Comments are stripped first: the handler explains in prose why it does
        NOT call save_metric, and a naive substring search reads that
        explanation as a violation."""
        handler = self.WEBHOOKS[self.WEBHOOKS.index("async def _run_metric_authoring_chat"):]
        handler = handler[: handler.index("async def _run_report_builder_chat")]
        code = " ".join(line.split("#", 1)[0] for line in handler.splitlines())
        assert "save_metric(" not in code
        assert "_after_semantic_approval" not in code

    def test_promotion_creates_a_proposal_and_nothing_else(self):
        frame = self.WEBHOOKS[self.WEBHOOKS.index('if msg_type == "metric_promotion_request"'):]
        frame = frame[: frame.index('if msg_type == "clarification_response"')]
        assert "create_metric_proposal" in frame
        assert "save_metric" not in frame

    def test_the_intent_gate_sits_after_reports_and_before_result_commands(self):
        """A report is the more specific ask, so it wins; and defining a
        calculation has nothing to do with a cached result, so it comes first.

        `parse_result_command` is compared at its DISPATCH site, not its import
        at the top of the file."""
        report_at = self.WEBHOOKS.index("_REPORT_BUILDER_INTENT_RE.search(text)")
        metric_at = self.WEBHOOKS.index("_METRIC_AUTHOR_INTENT_RE.search(text)")
        command_at = self.WEBHOOKS.index("result_command = parse_result_command(text)")
        assert report_at < metric_at < command_at


class TestTheIntentRegex:
    @pytest.mark.parametrize("text", [
        "define a metric for revenue per active customer",
        "create a new metric called gross margin",
        "set up a kpi for orders per warehouse",
        "take IVC_AMT from the invoice fact and compute revenue per active customer",
        "use CUS_NO and calculate the average order value",
    ])
    def test_it_catches_an_authoring_request(self, text):
        from gateway.webhooks import _METRIC_AUTHOR_INTENT_RE

        assert _METRIC_AUTHOR_INTENT_RE.search(text)

    @pytest.mark.parametrize("text", [
        "what is my revenue last month",
        "show me the top 10 customers",
        "how many customers do we have",
        "build me a report with net revenue",
        "which items have the highest on hand quantity",
    ])
    def test_it_leaves_ordinary_questions_alone(self, text):
        from gateway.webhooks import _METRIC_AUTHOR_INTENT_RE

        assert not _METRIC_AUTHOR_INTENT_RE.search(text)


# ── The pipeline injection ───────────────────────────────────────────────────


class TestThePipelineSeesTheDraft:
    PIPELINE = (ROOT / "core" / "query_pipeline.py").read_text(encoding="utf-8")

    def test_drafts_are_prepended_before_both_metric_scope_passes(self):
        """The early pass feeds authoritative_fact_tables into source
        resolution, which is what anchors the query on the fact the user named.
        Injecting after it would lose that."""
        inject_at = self.PIPELINE.index("active_session_metrics(")
        first_scope = self.PIPELINE.index("resolve_metric_scope(")
        assert inject_at < first_scope

    def test_a_pinned_draft_narrows_the_scope_rather_than_only_silencing_it(self):
        """The user already said which definition they meant -- by defining it.

        Suppressing the clarification alone was worse than asking. The rival
        registry metrics stayed in scope, their source tables unioned into the
        graph resolution, and a two-table ratio pulled ten entities in: the live
        answer was "the confirmed entity graph cannot reach SUP_DMS" on a
        question about customers. The draft has to REPLACE the rivals."""
        idx = self.PIPELINE.index("if _metric_scope.ambiguous and _adhoc_metrics:")
        block = self.PIPELINE[idx:idx + 600]
        assert "dataclasses.replace" in block
        assert "metrics=[]" in block
        assert "ambiguous=False" in block

    def test_a_draft_never_counts_as_registry_usage(self):
        idx = self.PIPELINE.index("store.increment_metric_usage(account_id")
        assert 'not m.get("_adhoc")' in self.PIPELINE[idx:idx + 300]

    def test_the_validator_enforces_a_pinned_draft_on_follow_ups(self):
        """metric_formula_mismatch is gated on the metric being mentioned by
        name. A follow-up does not repeat the name, so without this the formula
        is shown to the model and silently unenforced."""
        validator = (ROOT / "core" / "validator.py").read_text(encoding="utf-8")
        idx = validator.index("def _metric_mentioned(")
        assert '_pinned_thread_metric' in validator[idx:idx + 700]


class TestAMetricMaySpanAFactAndADimension:
    """Found on the live warehouse. "Revenue per active customer" composed
    correctly, then validation rejected it:

        Required column 'ACT_FLG' does not exist in table 'CUS_ORD_IVC_FCT'

    which is true and beside the point — the active flag lives on the customer
    master, as it should. validate_metric checks required_columns against
    base_table alone, which is right for a single-table metric and wrong for the
    ratio shape this whole feature exists to compose.
    """

    SCHEMA = {
        "CHATBOT_DB.EMDW_DMART.CUS_ORD_IVC_FCT": ["IVC_AMT", "CUS_DMS_KEY"],
        "CHATBOT_DB.EMDW_DMART.CUS_DMS": ["CUS_NO", "ACT_FLG"],
    }

    def _draft(self):
        plan_input = build_metric_plan_input("revenue per active customer", {
            "EMDW_DMART.CUS_ORD_IVC_FCT": ["IVC_AMT", "CUS_DMS_KEY"],
            "EMDW_DMART.CUS_DMS": ["CUS_NO", "ACT_FLG"],
        })
        draft, error = compile_metric_plan_response(json.dumps({
            "operation": "define_metric", "name": "Revenue Per Active Customer",
            "mode": "ratio", "base_table_ref": "TABLE_REF_1",
            "numerator": {"aggregation": "SUM", "measure_ref": "COL_REF_1"},
            "denominator": {
                "aggregation": "COUNT_DISTINCT", "measure_ref": "COL_REF_3",
                "filters": [{"field_ref": "COL_REF_4", "operator": "equals", "value": "Y"}],
            },
            "confidence": 0.9,
        }), plan_input)
        assert error == ""
        return draft

    def test_the_cross_table_ratio_now_validates(self):
        from core.metric_authoring import schema_columns_for_draft
        from core.metric_validator import validate_metric

        draft = self._draft()
        widened = schema_columns_for_draft(draft, self.SCHEMA)
        assert validate_metric(draft.as_metric(), db_type="azure_sql",
                               schema_columns=widened).valid

    def test_without_widening_it_was_rejected(self):
        """Pin the failure so the fix cannot be quietly undone."""
        from core.metric_validator import validate_metric

        result = validate_metric(
            self._draft().as_metric(), db_type="azure_sql", schema_columns=self.SCHEMA,
        )
        assert not result.valid
        assert any("ACT_FLG" in e for e in result.errors)

    def test_widening_is_a_no_op_for_a_single_table_metric(self):
        from core.metric_authoring import schema_columns_for_draft

        plan_input = build_metric_plan_input(
            "total revenue", {"EMDW_DMART.CUS_ORD_IVC_FCT": ["IVC_AMT"]},
        )
        draft, _ = compile_metric_plan_response(json.dumps({
            "operation": "define_metric", "name": "Total Revenue", "mode": "aggregate",
            "aggregation": "SUM", "measure_ref": "COL_REF_1",
            "base_table_ref": "TABLE_REF_1", "confidence": 0.9,
        }), plan_input)
        assert schema_columns_for_draft(draft, self.SCHEMA) == self.SCHEMA

    def test_widening_cannot_admit_a_table_the_user_never_saw(self):
        """The union is bounded by the draft's own source_tables, every one of
        which came from a COL_REF binding — so it can only ever contain tables
        the ACL already allowed into the manifest."""
        from core.metric_authoring import schema_columns_for_draft

        draft = self._draft()
        widened = schema_columns_for_draft(draft, {
            **self.SCHEMA, "CHATBOT_DB.EMDW_DMART.PAYROLL": ["SALARY"],
        })
        assert "SALARY" not in widened[draft.base_table]


class TestADraftsTablesAreKnownNotInferred:
    """The live failure: a two-table ratio resolved to ten entities, graph
    planning blocked, and the answer was "the confirmed entity graph cannot
    reach SUP_DMS" — a question about customers refused over a supplier table
    nobody mentioned.

    metric_source_tables INFERS a metric's tables, and one of its rules is "any
    table whose columns intersect required_columns". A draft requiring
    CUS_DMS_KEY therefore claims every fact and dimension carrying that key.
    That inference is the right default for a registry metric someone typed by
    hand. For a draft it is strictly worse than the truth, because every table
    came from a COL_REF binding this process issued.
    """

    ALL_COLUMNS = {
        "EMDW_DMART.CUS_ORD_IVC_FCT": {"IVC_GRS_AMT": "d", "CUS_DMS_KEY": "i"},
        "EMDW_DMART.CUS_DMS": {"CUS_DMS_KEY": "i", "ACT_FLG": "c"},
        "EMDW_DMART.CUS_TYP_DMS": {"CUS_DMS_KEY": "i"},
        "EMDW_DMART.SUP_DMS": {"CUS_DMS_KEY": "i"},
        "EMDW_DMART.WHS_DMS": {"CUS_DMS_KEY": "i"},
    }

    def _draft_metric(self):
        return {
            "name": "Revenue Per Active Customer",
            "sql_template": (
                "SUM(IVC_GRS_AMT) * 1.0 / NULLIF("
                "COUNT(DISTINCT CASE WHEN ACT_FLG = 'Y' THEN CUS_DMS_KEY END), 0)"
            ),
            "required_columns": "IVC_GRS_AMT, CUS_DMS_KEY, ACT_FLG",
            "base_table": "EMDW_DMART.CUS_ORD_IVC_FCT",
            "_adhoc": True,
            "_source_tables": ["EMDW_DMART.CUS_ORD_IVC_FCT", "EMDW_DMART.CUS_DMS"],
        }

    def test_inference_really_does_over_reach(self):
        """Pin the cause, so the fix cannot be undone as an apparent tidy-up."""
        from core.metric_scope import metric_source_tables

        inferred = metric_source_tables(self._draft_metric(), self.ALL_COLUMNS)
        assert any("SUP_DMS" in table for table in inferred)
        assert len(inferred) > 2

    def test_the_pipeline_prefers_the_bindings_over_inference(self):
        pipeline = (ROOT / "core" / "query_pipeline.py").read_text(encoding="utf-8")
        idx = pipeline.index("_metric_formula_tables = set()")
        block = pipeline[idx:idx + 1400]
        assert '_metric.get("_adhoc")' in block
        assert '_metric["_source_tables"]' in block

    def test_a_registry_metric_still_uses_inference(self):
        """Inference is right for a metric someone typed by hand — it has no
        bindings to fall back on."""
        pipeline = (ROOT / "core" / "query_pipeline.py").read_text(encoding="utf-8")
        idx = pipeline.index("_metric_formula_tables = set()")
        assert "metric_source_tables(_metric, all_columns)" in pipeline[idx:idx + 1400]


class TestAFormulaThatBindsIsNotAFormulaThatMatches:
    """The value-guess gap, found live.

    The model is shown column NAMES and never their contents, so it guesses the
    values it filters on. Across three consecutive live attempts "active" became
    ACT_FLG = 'Y', then 'true', then 1. All three are valid SQL, all three bind,
    all three pass the dry run — and at most one is right. The wrong ones return
    a confident number computed over zero matching rows.
    """

    CONFIG = json.dumps({
        "enabled": True, "mode": "ratio",
        "numerator": {"aggregation": "SUM", "measure": "IVC_GRS_AMT",
                      "filters": [{"field": "CNL_FLG", "operator": "equals", "value": "N"}]},
        "denominator": {"aggregation": "COUNT", "measure": "CUS_DMS_KEY",
                        "filters": [{"field": "ACT_FLG", "operator": "equals", "value": "1"}]},
    })

    def test_filters_are_found_in_every_builder_mode(self):
        from core.metric_dryrun import _filters_from_config

        assert [f["field"] for f in _filters_from_config(self.CONFIG)] == ["CNL_FLG", "ACT_FLG"]
        aggregate = json.dumps({
            "enabled": True, "mode": "aggregate", "aggregation": "SUM", "measure": "AMT",
            "filters": [{"field": "STATUS_CD", "operator": "equals", "value": "X"}],
        })
        assert [f["field"] for f in _filters_from_config(aggregate)] == ["STATUS_CD"]

    @pytest.mark.parametrize("config", ["", "{}", "not json", None])
    def test_a_metric_with_no_filters_has_nothing_to_check(self, config):
        from core.metric_dryrun import _filters_from_config

        assert _filters_from_config(config) == []

    def test_a_zero_match_filter_is_reported_by_column_not_by_value(self):
        """Column names may be shown to a portal user; the probe's view of the
        data may not — a probe does not pass through the compliance boundary."""
        import asyncio
        from unittest.mock import patch

        from core import metric_dryrun

        class _Cur:
            def __init__(self, outer):
                self.outer = outer

            def execute(self, sql):
                # ACT_FLG = '1' matches nothing; CNL_FLG = 'N' matches plenty.
                self.outer.n = 0 if "ACT_FLG" in sql else 4212

            def fetchone(self):
                return (self.outer.n,)

        class _Conn:
            n = 0

            def cursor(self):
                return _Cur(self)

            def close(self):
                pass

        master = {
            "DW.CUS_ORD_IVC_FCT": {"columns": [{"name": "IVC_GRS_AMT"}, {"name": "CNL_FLG"}]},
            "DW.CUS_DMS": {"columns": [{"name": "CUS_DMS_KEY"}, {"name": "ACT_FLG"}]},
        }
        with patch.object(metric_dryrun.store, "get_client", return_value={"db_config_id": 1}), \
             patch.object(metric_dryrun.store, "get_db_config",
                          return_value={"db_type": "azure_sql", "credentials": {}}), \
             patch.object(metric_dryrun, "_load_schema_master", return_value=master), \
             patch("core.schema._az_connect", return_value=_Conn()):
            empty = asyncio.run(metric_dryrun.check_filter_matches(
                "acct", metric_builder_config=self.CONFIG,
            ))

        assert empty == ("ACT_FLG",), "only the filter matching nothing should be reported"
        assert not any(v in str(empty) for v in ("'1'", "true", "Y"))

    def test_a_probe_that_cannot_run_is_not_evidence_of_an_empty_filter(self):
        """Failing closed here would refuse every metric whenever the database
        is briefly unreachable."""
        import asyncio
        from unittest.mock import patch

        from core import metric_dryrun

        with patch.object(metric_dryrun.store, "get_client", return_value={"db_config_id": 1}), \
             patch.object(metric_dryrun.store, "get_db_config",
                          return_value={"db_type": "azure_sql", "credentials": {}}), \
             patch.object(metric_dryrun, "_load_schema_master", return_value={
                 "DW.CUS_DMS": {"columns": [{"name": "ACT_FLG"}]}}), \
             patch("core.schema._az_connect", side_effect=RuntimeError("network down")):
            empty = asyncio.run(metric_dryrun.check_filter_matches(
                "acct", metric_builder_config=self.CONFIG,
            ))
        assert empty == ()

    def test_the_handler_names_the_column_and_asks(self):
        """Falling through silently would leave the user with an ordinary answer
        and no idea their definition was dropped."""
        webhooks = (ROOT / "gateway" / "webhooks.py").read_text(encoding="utf-8")
        handler = webhooks[webhooks.index("async def _run_metric_authoring_chat"):]
        handler = handler[: handler.index("async def _run_report_builder_chat")]
        assert "check_filter_matches" in handler
        # The sentence itself lives in the catalogue now, so the handler is
        # checked for the id and the SHIPPED string is checked for what it has
        # to say -- in both languages, because a translation that dropped the
        # column name would be the same silent failure in French.
        assert '_t("reply.metric.empty_filter"' in handler
        assert "columns=columns" in handler
        from core import i18n
        for lang in i18n.SUPPORTED_LANGUAGES:
            said = i18n.t("reply.metric.empty_filter", lang=lang,
                          columns="ACT_FLG", example="ACT_FLG is Y")
            assert "ACT_FLG" in said, lang
            assert "?" not in said.split("\n")[0], lang
        assert "matches no rows" in i18n.t(
            "reply.metric.empty_filter", lang="en", columns="X", example="Y")
        assert "_fall_through(f\"filter matched no rows" in handler


class TestTheFilterCheckIsObservable:
    """A check that silently found nothing and a check that silently never ran
    produce the same log and the same outcome. On the live warehouse the check
    correctly did not fire — and there was no way to tell that from the log,
    which is the same "did this code even execute" problem that has produced
    four separate wrong diagnoses this week."""

    def test_it_logs_what_it_probed_not_only_what_failed(self, caplog):
        import asyncio
        import logging
        from unittest.mock import patch

        from core import metric_dryrun

        class _Conn:
            def cursor(self):
                class _Cur:
                    def execute(self, sql): pass
                    def fetchone(self): return (17,)
                return _Cur()

            def close(self): pass

        config = json.dumps({
            "enabled": True, "mode": "aggregate", "aggregation": "SUM", "measure": "AMT",
            "filters": [{"field": "ACT_FLG", "operator": "equals", "value": "1"}],
        })
        with caplog.at_level(logging.INFO, logger="querybot.metric_dryrun"), \
             patch.object(metric_dryrun.store, "get_client", return_value={"db_config_id": 1}), \
             patch.object(metric_dryrun.store, "get_db_config",
                          return_value={"db_type": "azure_sql", "credentials": {}}), \
             patch.object(metric_dryrun, "_load_schema_master", return_value={
                 "DW.CUS_DMS": {"columns": [{"name": "ACT_FLG"}]}}), \
             patch("core.schema._az_connect", return_value=_Conn()):
            empty = asyncio.run(metric_dryrun.check_filter_matches(
                "acct", metric_builder_config=config,
            ))

        assert empty == ()
        assert "ACT_FLG" in caplog.text
        assert "matched no rows: none" in caplog.text


class TestTheDryRunGateRejectsAnythingNotProven:
    """"Not an error" is not "verified".

    DryRunOutcome has three statuses. The gate was written as
    `if outcome.status == "error"`, so "skipped" -- returned when there is no
    database configured, no discovered tables, or no such account, i.e. exactly
    the cases where NOTHING was probed -- passed through as though the formula
    had been proven against the live warehouse. The dry run is the last gate
    before a composed metric answers a real question, and it was accepting
    silence as proof.

    The condition is lifted out of the shipped source and evaluated, rather than
    restated here. A restatement would pass whatever the file says.
    """

    def _gate_condition(self) -> str:
        from pathlib import Path

        src = (Path(__file__).resolve().parents[1] / "gateway" / "webhooks.py").read_text(
            encoding="utf-8",
        )
        marker = "outcome.status"
        idx = src.index("outcome = await dry_run_metric_formula(")
        window = src[idx:idx + 1400]
        line = next(
            ln.strip() for ln in window.splitlines()
            if ln.strip().startswith("if ") and marker in ln
        )
        assert line.endswith(":"), line
        return line[len("if "):-1]

    @pytest.mark.parametrize("status,should_fall_through", [
        ("ok", False),
        ("error", True),
        ("skipped", True),
    ])
    def test_only_a_proven_dry_run_passes_the_gate(self, status, should_fall_through):
        from core.metric_dryrun import DryRunOutcome

        outcome = DryRunOutcome(status=status, detail="probe detail")
        assert bool(eval(self._gate_condition(), {}, {"outcome": outcome})) is should_fall_through

    def test_every_status_the_outcome_can_carry_is_covered(self):
        """If a fourth status is added, this test has to be revisited rather
        than silently letting the new one through."""
        import re
        from pathlib import Path

        src = (Path(__file__).resolve().parents[1] / "core" / "metric_dryrun.py").read_text(
            encoding="utf-8",
        )
        declared = re.search(r'status: str\s*#\s*(.+)', src).group(1)
        found = set(re.findall(r'"(\w+)"', declared))
        assert found == {"ok", "error", "skipped"}, f"statuses changed: {found}"
