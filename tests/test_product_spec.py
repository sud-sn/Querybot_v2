"""
tests/test_product_spec.py

THE SPECIFICATION. What this product promises a user, stated as executable
sentences, end to end.

Every other test module verifies a unit. This one verifies a PROMISE, and it
deliberately crosses module boundaries to do it -- because every defect found
in this work sat in the seam between two modules that were each individually
correct:

  * a forecast gate that ran on a name belonging to a different module
  * a chart policy enforced in one of the two places that needed it
  * an access-control rule applied on the read someone remembered
  * an accept route calling a store function with the wrong arity
  * a dry run whose "skipped" was read as "passed"

None of those were visible from inside a single module. Each is pinned here as
a promise instead, so the seam is what gets tested.

Read this file to learn what the product guarantees. If a promise here is
wrong, the product is wrong -- not the test.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_tmp_db = os.path.join(tempfile.mkdtemp(), "test_product_spec.db")
os.environ["QUERYBOT_DB_PATH"] = _tmp_db
for _mod in list(sys.modules):
    if _mod.startswith("store"):
        del sys.modules[_mod]

import store  # noqa: E402

from core.forecast import (  # noqa: E402
    compute_forecast, detect_forecast_intent, extract_forecast_periods,
)
from core.forecast_gate import assess_fit, evaluate_forecast_request  # noqa: E402
from core.forecast_models import fit_series, statsmodels_available  # noqa: E402
from core.result_renderer import _format_value, _rows_to_table  # noqa: E402

store.init_db()


# ── Fixtures shaped like the real thing ──────────────────────────────────────

def months(values, *, start_year=2025, start_month=1, col="REVENUE"):
    """A monthly series in the shape the pipeline actually produces."""
    out = []
    y, m = start_year, start_month
    for v in values:
        out.append({"PERIOD": f"{y}-{m:02d}", col: v})
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


# Eighteen months of warehouse revenue: flat, which is what real revenue
# usually is, and the case every threshold in the gate had to be judged against.
EIGHTEEN_MONTHS = [
    7379419.76, 6867159.02, 7548077.20, 7307544.65, 7523942.04, 7381519.76,
    7670171.04, 7710319.55, 7444167.07, 7639739.52, 7332990.06, 7540560.84,
    7489655.63, 6903766.55, 7639510.50, 7367087.16, 7590724.34, 7439558.42,
]


@pytest.fixture
def account():
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Spec Ltd")
    return account_id


# ═════════════════════════════════════════════════════════════════════════════
# PROMISE 1 — A forecast appears only when the series can support one, and
#             every refusal says what would make it possible.
# ═════════════════════════════════════════════════════════════════════════════

class TestAForecastIsEarned:
    def test_a_healthy_monthly_series_is_projected(self):
        decision = evaluate_forecast_request(months(EIGHTEEN_MONTHS), horizon=3)
        assert decision.allowed
        assert decision.grain == "month"
        assert decision.n_points == 18

    @pytest.mark.parametrize("rows,kwargs,reason", [
        (months([100, 110, 120]), {}, "too_short"),
        (months([100] * 8), {}, "constant_series"),
        (months(list(range(1, 9))), {"truncated": True}, "truncated_result"),
        (months(list(range(1, 9))), {"policy_allows_derived_visual": False}, "policy_blocked"),
        (months(list(range(1, 9)))[::-1], {}, "unordered_series"),
    ])
    def test_a_series_that_cannot_support_one_is_refused_by_name(self, rows, kwargs, reason):
        decision = evaluate_forecast_request(rows, **kwargs)
        assert not decision.allowed
        assert decision.reason_code == reason

    def test_every_reachable_refusal_is_visible_to_the_user(self):
        """A refusal the user cannot see is indistinguishable from the product
        ignoring them. no_rows is the one deliberate silence: an empty result
        already explains itself."""
        cases = [
            months([100, 110, 120]),
            months([100] * 8),
            [{"PERIOD": f"2026-{i + 1:02d}", "STATUS": "open"} for i in range(8)],
            [{"CUSTOMER": n, "REVENUE": i} for i, n in enumerate("abcdef")],
        ]
        for rows in cases:
            decision = evaluate_forecast_request(rows)
            assert not decision.allowed
            assert decision.caveat, f"{decision.reason_code} refuses in silence"
            assert decision.caveat.startswith("I did not project")

    def test_a_masked_value_stops_the_projection_entirely(self):
        """Compliance masking runs BEFORE analytics. Fitting over the cells that
        happened to survive would be fitting an unknown, unreported subset."""
        rows = months(list(range(1, 13)))
        rows[4]["REVENUE"] = "[REDACTED]"
        assert evaluate_forecast_request(rows).reason_code == "masked_series"

    def test_a_flat_but_predictable_series_is_NOT_refused(self):
        """The live case, and the reason poor_fit needs two conditions. Real
        revenue has an R-squared near zero because there is no trend to
        explain -- but "about the same again" is an accurate projection, and
        refusing it would refuse a correct answer."""
        decision = evaluate_forecast_request(months(EIGHTEEN_MONTHS))
        fit = fit_series(EIGHTEEN_MONTHS, decision.horizon, model=decision.model)
        assert fit.r2 < 0.5, "no linear trend, as expected"
        assert fit.backtest_mape < 5.0, "yet highly predictable"
        assert assess_fit(decision, fit.r2, fit.backtest_mape).allowed

    def test_a_series_that_is_both_unexplained_and_unpredictable_is_refused(self):
        noise = [100, 900, 150, 40, 800, 90, 700, 30, 850, 60, 780, 45]
        decision = evaluate_forecast_request(months(noise))
        fit = fit_series(noise, decision.horizon, model=decision.model)
        assert assess_fit(decision, fit.r2, fit.backtest_mape).reason_code == "poor_fit"


# ═════════════════════════════════════════════════════════════════════════════
# PROMISE 2 — A projection always carries an interval, and the interval is
#             honest about how often it is right.
# ═════════════════════════════════════════════════════════════════════════════

class TestAProjectionSaysHowUncertainItIs:
    def test_every_projected_row_carries_a_band(self):
        rows = compute_forecast(months(EIGHTEEN_MONTHS), "PERIOD", "REVENUE", 3)
        projected = [r for r in rows if r["is_forecast"]]
        assert len(projected) == 3
        for r in projected:
            assert r["forecast_low"] < r["forecast_value"] < r["forecast_high"]

    def test_a_measured_period_never_carries_one(self):
        rows = compute_forecast(months(EIGHTEEN_MONTHS), "PERIOD", "REVENUE", 3)
        for r in rows:
            if not r["is_forecast"]:
                assert r["forecast_low"] is None and r["forecast_high"] is None

    @pytest.mark.parametrize("n,h", [(12, 1), (18, 3)])
    def test_the_band_covers_the_truth_about_95_percent_of_the_time(self, n, h):
        """Checked against reality, not against a restatement of the formula:
        draw series from a known process and count how often the truth lands
        inside. A normal approximation instead of the Student t lands near 85%
        here and fails."""
        rng = random.Random(4242)
        hits, trials = 0, 250
        for _ in range(trials):
            series = [100 + 4 * i + rng.gauss(0, 25) for i in range(n + h)]
            fit = fit_series(series[:n], h, model="ols")
            if fit.lower[h - 1] <= series[n + h - 1] <= fit.upper[h - 1]:
                hits += 1
        assert 0.90 <= hits / trials <= 0.99, f"covered {hits / trials:.1%}"

    def test_the_interval_is_labelled_with_the_confidence_it_actually_carries(self):
        rows = compute_forecast(months(EIGHTEEN_MONTHS), "PERIOD", "REVENUE", 3)
        assert rows[0]["__forecast_meta"]["interval_confidence"] == 0.95

    def test_a_missing_library_degrades_the_model_and_never_the_answer(self):
        """statsmodels is declared, but a host where its wheel failed must still
        answer -- with a simpler model, not an error."""
        fit = fit_series([100 + i * 4 for i in range(30)], 3,
                         model="sarimax", seasonal_period=12)
        assert fit is not None and fit.predictions and fit.lower and fit.upper
        if statsmodels_available():
            assert fit.model == "sarimax"
        else:
            assert fit.model == "ols" and fit.fell_back_from == "sarimax"

    def test_the_projection_is_fitted_on_the_calendar_not_the_row_number(self):
        rows = [{"PERIOD": p, "REVENUE": v} for p, v in [
            ("2026-01", 100), ("2026-02", 110), ("2026-03", 120),
            ("2026-09", 180), ("2026-10", 190), ("2026-11", 200),
        ]]
        out = compute_forecast(rows, "PERIOD", "REVENUE", 1)
        assert out[0]["__forecast_meta"]["fitted_on"] == "period_dates"
        assert out[0]["__trend_slope"] == pytest.approx(10.0, abs=0.5)

    def test_a_null_tail_period_does_not_shift_the_labels(self):
        """The current month often exists in the date dimension before any
        value lands against it."""
        rows = months([10, 12, 11, 13, 12, 14], start_year=2026)
        rows.append({"PERIOD": "2026-07", "REVENUE": None})
        out = compute_forecast(rows, "PERIOD", "REVENUE", 2)
        assert [r["PERIOD"] for r in out if r["is_forecast"]] == ["2026-07", "2026-08"]


# ═════════════════════════════════════════════════════════════════════════════
# PROMISE 3 — The pipeline's analytics actually run, and a failure is loud.
# ═════════════════════════════════════════════════════════════════════════════

class TestTheAnalyticsAreReachable:
    def test_every_name_the_query_pipeline_reads_resolves(self):
        """Two live outages came from a name that did not exist on the path
        that read it: a global belonging to another module, then a local
        assigned only under an unrelated branch. Both were swallowed."""
        import builtins
        import dis
        import types

        import core.query_pipeline as qp

        seen, missing = set(), []

        def walk(code):
            if id(code) in seen:
                return
            seen.add(id(code))
            for ins in dis.get_instructions(code):
                if ins.opname in {"LOAD_GLOBAL", "LOAD_NAME"}:
                    if not hasattr(qp, ins.argval) and not hasattr(builtins, ins.argval):
                        missing.append(ins.argval)
            for const in code.co_consts:
                if isinstance(const, types.CodeType):
                    walk(const)

        walk(qp._handle_query_impl.__code__)
        assert sorted(set(missing)) == []

    def test_a_programming_error_in_post_processing_is_logged_loudly(self):
        from pathlib import Path

        src = (Path(__file__).resolve().parents[1] / "core" / "query_pipeline.py").read_text(
            encoding="utf-8")
        assert 'log.debug("Post-processing analytics skipped' not in src, (
            "an analytic the user asked for failing to run is never routine"
        )

    def test_the_forecast_intent_still_reaches_the_pipeline(self):
        for q in ("forecast my revenue for the next 3 months",
                  "project revenue for the next 4 months",
                  "what will my monthly revenue be over the coming 3 months"):
            assert detect_forecast_intent(q)
        assert extract_forecast_periods("forecast revenue for the next 4 months") == 4


# ═════════════════════════════════════════════════════════════════════════════
# PROMISE 4 — One rule governs every derived view of the same values.
# ═════════════════════════════════════════════════════════════════════════════

class TestOneGateGovernsCharts_And_Forecasts:
    def test_both_surfaces_call_the_same_policy_function(self):
        import inspect

        import core.query_pipeline as qp
        import core.result_renderer as rr

        for module in (qp, rr):
            assert "aggregate_only_gate_passes" in inspect.getsource(module)

    def test_a_regulated_tenant_fails_closed_when_policy_evaluation_breaks(self):
        from core.chart_policy import aggregate_only_gate_passes
        import core.compliance.policy_engine as pe

        broken = dict(portal_user=None, event=None, sql="not sql", db_type="nope")
        with patch.object(pe, "is_regulated", lambda a: True):
            assert not aggregate_only_gate_passes(account_id="x", **broken)
        with patch.object(pe, "is_regulated", lambda a: False):
            assert aggregate_only_gate_passes(account_id="x", **broken)


# ═════════════════════════════════════════════════════════════════════════════
# PROMISE 5 — A number is rendered as the thing it is.
# ═════════════════════════════════════════════════════════════════════════════

class TestNumbersReadAsWhatTheyAre:
    @pytest.mark.parametrize("value,column,rendered", [
        (2025, "PERIOD", "2025"),            # a year is a name, not an amount
        (2026, "FISCAL_YEAR", "2026"),
        (202601, "MONTH", "202601"),
        (20260117, "INVOICE_DATE", "20260117"),
        (2025, "ORDER_COUNT", "2,025"),      # a count keeps its separators
        (12345, "PERIOD", "12,345"),         # not a valid period
        (900001, "CUSTOMER_PERIOD_KEY", "900,001"),   # a surrogate key, not a period
    ])
    def test_a_period_is_a_label_and_a_quantity_is_a_quantity(self, value, column, rendered):
        assert _format_value(value, column) == rendered

    def test_the_chat_reply_never_shows_the_analytics_own_scratch_columns(self):
        """A forecast adds seven marker columns, one of which is a dict. They
        were printed into the text a customer reads."""
        rows = compute_forecast(months([10, 12, 11, 13, 12, 14], start_year=2026),
                                "PERIOD", "REVENUE", 3)
        table = _rows_to_table(rows)
        for marker in ("__forecast_meta", "__trend_slope", "is_forecast",
                       "forecast_low", "forecast_high", "'model':"):
            assert marker not in table, f"{marker} leaked into the chat reply"

    def test_a_projection_is_labelled_as_one_in_channels_with_no_chart(self):
        """Teams and the REST API get the table and nothing else, so hiding the
        marker columns would otherwise erase the only thing distinguishing a
        projection from a measurement."""
        rows = compute_forecast(months([10, 12, 11, 13, 12, 14], start_year=2026),
                                "PERIOD", "REVENUE", 3)
        assert "projected, not measured" in _rows_to_table(rows)


# ═════════════════════════════════════════════════════════════════════════════
# PROMISE 6 — A composed metric is a request, never a fact.
# ═════════════════════════════════════════════════════════════════════════════

SCHEMA = {
    "DW.SALES_FACT": {"AMOUNT", "DISCOUNT", "CUS_NO"},
    "DW.CUSTOMER_DIM": {"CUS_NO", "ACT_FLG"},
}


def _compose(account_id, message):
    from admin import routes
    from core.metric_dryrun import DryRunOutcome

    request = MagicMock()
    request.json = AsyncMock(return_value={"message": message, "history": []})
    patches = [
        patch.object(routes, "_is_auth", return_value=True),
        patch.object(routes, "_load_metric_schema_columns", return_value=dict(SCHEMA)),
        patch("core.metric_dryrun.dry_run_metric_formula", AsyncMock(
            return_value=DryRunOutcome(status="ok", detail="bound",
                                       probe_kind="single_table",
                                       tables_probed=("DW.SALES_FACT",), value=99))),
    ]
    for p in patches:
        p.start()
    try:
        resp = asyncio.run(routes.metric_authoring_chat(request, account_id))
    finally:
        for p in reversed(patches):
            p.stop()
    return json.loads(bytes(resp.body))


class TestComposingAMetricNeverPublishesIt:
    def test_a_described_calculation_produces_a_proposal_and_no_metric(self, account):
        body = _compose(account, "Net Revenue = SUM(AMOUNT) - SUM(DISCOUNT) FROM DW.SALES_FACT")
        assert body["status"] == "ok"
        assert store.list_metrics(account, active_only=False) == []
        assert len(store.list_metric_proposals(account, status="pending")) == 1

    def test_the_semantic_contract_is_never_recompiled_by_a_chat_handler(self, account):
        """Recompiling changes answers to questions nobody has asked yet. Only
        a human accept route may do it."""
        from admin import routes

        with patch.object(routes, "_after_semantic_approval") as recompile:
            _compose(account, "Net Revenue = SUM(AMOUNT) FROM DW.SALES_FACT")
        recompile.assert_not_called()

    def test_accepting_is_what_creates_the_metric(self, account):
        from admin import routes

        body = _compose(account, "Net Revenue = SUM(AMOUNT) FROM DW.SALES_FACT")
        with patch.object(routes, "_is_auth", return_value=True), \
                patch.object(routes, "_after_semantic_approval") as recompile, \
                patch.object(routes, "_notify_metric_proposal_reviewed", create=True):
            resp = asyncio.run(routes.metric_proposal_accept(
                MagicMock(), account, body["proposal_id"]))
        assert resp.status_code == 200
        recompile.assert_called_once()
        assert [m["name"] for m in store.list_metrics(account, active_only=False)] == ["Net Revenue"]

    def test_an_explicit_definition_costs_no_model_call(self, account):
        """No LLM is patched in _compose: reaching one raises rather than
        passing quietly."""
        assert _compose(account, "X = SUM(AMOUNT) FROM DW.SALES_FACT")["source"] == "pasted"

    @pytest.mark.parametrize("text,expected_in_reply", [
        ("Bad = SUM(NOPE) FROM DW.SALES_FACT", "NOPE"),
        ("Bad = SUM(AMOUNT) FROM DW.NOT_YOURS", "NOT_YOURS"),
        ("Evil = SUM(AMOUNT); DROP TABLE X FROM DW.SALES_FACT", "will not accept"),
    ])
    def test_a_definition_that_cannot_be_trusted_is_refused_by_name(
        self, account, text, expected_in_reply,
    ):
        body = _compose(account, text)
        assert body["status"] == "clarify"
        assert expected_in_reply in body["reply"]
        assert store.list_metric_proposals(account) == []

    def test_the_review_queue_is_reachable(self, account):
        """It shipped with no caller for a release: proposals accumulated where
        no administrator could see them."""
        from admin import routes

        _compose(account, "Net Revenue = SUM(AMOUNT) FROM DW.SALES_FACT")
        request = MagicMock()
        request.query_params = {}
        with patch.object(routes, "_is_auth", return_value=True):
            page = asyncio.run(routes.metrics_page(request, account)).body.decode("utf-8", "replace")
        assert 'id="proposals"' in page
        assert "Net Revenue" in page


# ═════════════════════════════════════════════════════════════════════════════
# PROMISE 7 — A revoked grant takes effect on every path, not the remembered one.
# ═════════════════════════════════════════════════════════════════════════════

class TestAccessControlIsNotOptional:
    def _draft(self, account_id):
        return store.save_session_metric_draft(
            account_id, "thread-1", 7,
            {"name": "R", "sql_template": "SUM(AMOUNT)", "formula_type": "expression",
             "required_columns": "AMOUNT", "base_table": "DW.SALES_FACT"},
            source_tables=["DW.SALES_FACT"], validation={"valid": True},
            dryrun={"status": "ok"}, confidence=0.9, source_question="revenue",
        )

    def test_a_revoked_table_hides_the_draft_from_the_thread(self, account):
        self._draft(account)
        assert store.active_session_metrics(account, "thread-1", ["DW.SALES_FACT"])
        assert store.active_session_metrics(account, "thread-1", ["DW.OTHER"]) == []

    def test_and_from_the_path_that_makes_it_permanent(self, account):
        """Promotion is the one path that turns a draft into a durable artifact,
        and it was the one path with no re-check."""
        draft_id = self._draft(account)
        assert store.get_session_metric_draft(account, draft_id, allowed_tables=["DW.SALES_FACT"])
        assert store.get_session_metric_draft(account, draft_id, allowed_tables=["DW.OTHER"]) is None

    def test_a_proposal_for_another_tenants_metric_is_not_found(self, account):
        from admin import routes

        other = f"acct{os.urandom(4).hex()}"
        store.upsert_client(other, "Other Ltd")
        store.save_metric(other, {"name": "Theirs", "sql_template": "SUM(A)",
                                  "formula_type": "expression", "required_columns": "A",
                                  "base_table": "DW.T"}, db_type="azure_sql")
        live = [m for m in store.list_metrics(other, active_only=False)][0]
        pid = store.create_metric_proposal(
            account, action="update_metric", target_metric_id=int(live["id"]),
            before=dict(live), payload={**dict(live), "sql_template": "SUM(HIJACKED)"})
        with patch.object(routes, "_is_auth", return_value=True), \
                patch.object(routes, "_after_semantic_approval"), \
                patch.object(routes, "_notify_metric_proposal_reviewed", create=True):
            resp = asyncio.run(routes.metric_proposal_accept(MagicMock(), account, pid))
        assert resp.status_code == 404
        assert store.get_metric(int(live["id"]))["sql_template"] == "SUM(A)"


# ═════════════════════════════════════════════════════════════════════════════
# PROMISE 8 — The page tells you what it is doing while it does it.
# ═════════════════════════════════════════════════════════════════════════════

class TestThePageNarratesItsWork:
    """The chat surface is a template, so these read it -- but they read it for
    WIRING, which is the thing that was broken: state that was declared and
    never consumed, and rules written at a specificity that never applied."""

    @staticmethod
    def _template():
        from pathlib import Path

        return (Path(__file__).resolve().parents[1] / "portal" / "templates"
                / "portal_chat.html").read_text(encoding="utf-8")

    def test_the_stage_trail_state_is_consumed_not_merely_declared(self):
        src = self._template()
        assert "_completedStages.push" in src, "stages must accumulate"
        assert 'id="answerProgressSteps"' in src, "the trail needs somewhere to render"

    def test_the_composer_announces_that_it_is_working(self):
        src = self._template()
        assert "Processing your request" in src
        assert "_composerPlaceholder" in src, (
            "the placeholder must restore from a stash: it is schema-dependent"
        )

    def test_the_stop_control_is_saturated_at_rest_not_on_hover(self):
        """It halts a running query; it should not look inert until touched."""
        src = self._template()
        stop = src[src.index(".send-btn.is-stop{"):]
        stop = stop[:stop.index("}") + 1]
        assert "var(--danger)" in stop and "linear-gradient" not in stop

    def test_follow_up_suggestions_are_offered_for_a_single_column_answer(self):
        """A KPI is exactly the result that gives a user no column to pivot on
        themselves, and it was the one excluded."""
        from pathlib import Path

        src = (Path(__file__).resolve().parents[1] / "core" / "result_renderer.py").read_text(
            encoding="utf-8")
        assert "len(rows[0]) >= 1" in src
        assert "len(rows[0]) >= 2" not in src


# ═════════════════════════════════════════════════════════════════════════════
# PROMISE 9 — The product wears one mark.
# ═════════════════════════════════════════════════════════════════════════════

class TestOneVisualIdentity:
    def test_the_favicon_and_the_animated_component_draw_the_same_glyph(self):
        import re
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        svg = (root / "static" / "img" / "logo-mark.svg").read_text(encoding="utf-8")
        macro = (root / "portal" / "templates" / "macros.html").read_text(encoding="utf-8")
        macro = macro.split("{% macro brand_motion", 1)[1].split("{%- endmacro %}", 1)[0]

        a = re.search(r'<path[^>]*d="(M31\.32[^"]+)"', svg)
        b = re.search(r'class="qb-brand-motion__bowl" d="([^"]+)"', macro)
        assert a and b and a.group(1).strip() == b.group(1).strip()

    def test_the_admin_and_portal_marks_have_not_drifted(self):
        import re
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        marks = []
        for where in ("admin", "portal"):
            macro = (root / where / "templates" / "macros.html").read_text(encoding="utf-8")
            macro = macro.split("{% macro brand_motion", 1)[1].split("{%- endmacro %}", 1)[0]
            svg = macro[macro.index("<svg"):macro.index("</svg>")]
            marks.append(re.sub(r"\s+", " ", svg).strip())
        assert marks[0] == marks[1]

    def test_the_resting_mark_never_loops(self):
        """It is the avatar on every assistant message; forty of them animating
        out of sync down a conversation is noise, not life."""
        from pathlib import Path

        svg = (Path(__file__).resolve().parents[1] / "static" / "img"
               / "logo-mark.svg").read_text(encoding="utf-8")
        assert "infinite" not in svg
        assert "prefers-reduced-motion" in svg


# ═════════════════════════════════════════════════════════════════════════════
# PROMISE 5b — The two renderers agree about what a number is.
# ═════════════════════════════════════════════════════════════════════════════

class TestTheServerAndTheBrowserFormatAlike:
    """Found live, after the server fix was verified through /api/ask and
    declared done: the portal draws its OWN tables in JavaScript, so the same
    year rendered "2025" in Teams and "2,025" on screen. One rule, two
    implementations, and only one of them had been fixed.

    These execute BOTH and require identical verdicts, so the next person to
    change either has to change both.
    """

    CASES = [
        (2025, "PERIOD"), (2026, "YEAR"), (202601, "MONTH"),
        (20260117, "INVOICE_DATE"), (2025, "FISCAL_YEAR"),
        (2025, "ORDER_COUNT"),              # a count keeps its separators
        (12345, "PERIOD"),                  # not a period at all
        (900001, "CUSTOMER_PERIOD_KEY"),    # a surrogate key, not a month
        (1200, "DAYS_LATE"), (45000, "AMOUNT"),
        (202613, "MONTH"),                  # month 13 does not exist
        (2025, ""),                         # no column, no opinion
    ]

    def test_both_implementations_return_the_same_verdict(self):
        dukpy = pytest.importorskip("dukpy")
        from pathlib import Path

        from core.result_renderer import _is_period_label

        src = (Path(__file__).resolve().parents[1] / "portal" / "templates"
               / "portal_chat.html").read_text(encoding="utf-8")
        js = src[src.index("const _PERIOD_COLUMN_WORDS"):src.index("function _formatDisplayValue")]
        payload = json.dumps([[str(v), c] for v, c in self.CASES])
        client = json.loads(dukpy.evaljs(js + f"\nJSON.stringify({payload}.map(p => _isPeriodLabel(p[0], p[1])));"))

        disagreements = [
            (v, c, _is_period_label(v, c), cl)
            for (v, c), cl in zip(self.CASES, client)
            if _is_period_label(v, c) != cl
        ]
        assert not disagreements, f"server/browser disagree: {disagreements}"

    def test_a_year_is_never_given_thousands_separators(self):
        from core.result_renderer import _format_value

        assert _format_value(2025, "PERIOD") == "2025"
        assert _format_value(2025, "ORDER_COUNT") == "2,025"
