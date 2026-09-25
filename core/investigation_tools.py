"""
core/investigation_tools.py

The tools an investigation's planner calls, each by name with typed inputs.

The loop (core/investigation_planner.py) had one tool: word a question and ask
it. Every analysis this product computes -- a forecast, the values in a series
that stand out, a correlation, a distribution, a cohort matrix, a funnel, each
item's share of a total -- was reachable only if the planner happened to word a
question the pipeline's keyword detection turned into it, and then only as one
more warehouse query.

Now the planner names a tool and gives its inputs. A tool that works on a
result takes a step of this same run -- never a result id, never another run's
result -- reads that step's cached rows and computes on them: no SQL, no second
warehouse query. What it computes is kept as a result of its own, derived from
the step it read, so a later step can work on it in turn. What it found goes
back to the planner as a brief of labels and figures -- the evidence the
summary is checked against.

`query` and `drill` are the tools that need new data. Both ask through the
governed pipeline, exactly as a reader's own question does.

Every input is checked before anything runs: an unknown tool, a step this run
did not take or that found nothing, a column its result does not have, a count
out of range. The step fails with the reason, and the loop goes on. The
forecast keeps the pipeline's own gates: the policy's word on a derived visual
of the result, whether the series can carry a forecast, and whether the fit
describes it.

The analysis sandbox is not offered. Whether a planner may run code over a
tenant's results is the tenant's decision, and it has not been made.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

from core.investigation import ToolResult

log = logging.getLogger("querybot.investigation_tools")

# How many items a brief names: enough to reason from, few enough that the
# planner reads figures rather than a table.
_BRIEF_ITEMS = 6

# The forecast models core.forecast_models fits, each named for a reader.
_MODELS = {"ols", "ets", "sarimax"}


@dataclass(frozen=True)
class ToolInput:
    name: str
    kind: str                 # "text", "step", "column" or "count"
    help: str
    required: bool = True
    low: int = 1              # a count's range, and its default
    high: int = 1
    default: int = 0
    of: str = "step"          # a column belongs to this step input's result


@dataclass(frozen=True)
class Tool:
    name: str
    summary: str
    inputs: tuple[ToolInput, ...]
    run: Callable[["ToolCall"], Awaitable[ToolResult]]
    # Computes on the rows of the steps it reads. `drill` reads only a
    # step's question and asks it again.
    reads_rows: bool = True


@dataclass
class ToolCall:
    """One checked call: its inputs, and the snapshot of each step it reads."""
    tool: Tool
    account_id: str
    portal_user: dict
    run_id: str
    session_id: str
    lang: str | None
    values: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, dict] = field(default_factory=dict)
    label: str = ""


class ToolCallError(ValueError):
    """An input the tool cannot run with; the message is the reader's."""


def _t(key: str, lang: str | None, **kwargs) -> str:
    from core.i18n import t

    return t(key, lang=lang, **kwargs)


def _fmt(value: Any, lang: str | None, digits: int | None = None) -> str:
    """A figure written the way the reader's language writes it: a brief is
    what the planner cites, and what the reader sees if the planner's
    summary is not used -- "1,234" is a thousand to one reader and one and
    a bit to another."""
    from core.i18n import format_decimal

    return "—" if value is None else format_decimal(value, digits, lang=lang)


def _pct(value: Any, lang: str | None) -> str:
    from core.i18n import format_percent

    return "—" if value is None else format_percent(value, 1, lang=lang)


def _column(rows: list[dict], wanted: str) -> str:
    """The result's own spelling of `wanted`, or ""."""
    names = {str(name).upper(): str(name) for name in (rows[0].keys() if rows else [])}
    return names.get(str(wanted or "").strip().upper(), "")


def _db_type(call: ToolCall, snapshot: dict) -> str:
    import store

    config_id = (snapshot.get("metadata") or {}).get("db_config_id")
    if not config_id:
        config_id = (store.get_client(call.account_id) or {}).get("db_config_id")
    if config_id:
        try:
            return str((store.get_db_config(config_id) or {}).get("db_type") or "azure_sql")
        except Exception as exc:  # noqa: BLE001 - the dialect is a hint to the policy check
            log.warning("Investigation tool could not read the connection %s: %s", config_id, exc)
    return "azure_sql"


def _keep(call: ToolCall, rows: list[dict], source: str = "step") -> str:
    """Keep what a tool computed as a result of its own, derived from the step it read."""
    from core.result_cache import result_cache

    if not rows:
        return ""
    derived = result_cache.derive_snapshot(
        call.session_id, str(call.sources[source].get("result_id") or ""), rows,
        question=call.label, operation=f"investigation:{call.tool.name}",
    )
    return str(derived.get("result_id") or "")


def _ok(call: ToolCall, brief: str, rows: list[dict], result_id: str) -> ToolResult:
    return ToolResult(ok=True, kind=call.tool.name, question=call.label, brief=brief,
                      result_id=result_id, row_count=len(rows),
                      columns=tuple(str(name) for name in (rows[0].keys() if rows else [])))


def _failed(call: ToolCall, error: str) -> ToolResult:
    return ToolResult(ok=False, kind=call.tool.name, question=call.label, error=error)


def _rows(call: ToolCall, source: str = "step") -> list[dict]:
    return [dict(row) for row in call.sources[source].get("rows") or []]


def _value_column(call: ToolCall, rows: list[dict], infer: Callable[[list[dict]], str]) -> str:
    return call.values.get("column") or infer(rows)


def _naming_columns(rows: list[dict], value: str) -> list[str]:
    return [str(name) for name in (rows[0].keys() if rows else [])
            if name != value and not str(name).startswith("_")]


def _label_column(rows: list[dict], value: str) -> str:
    """The column that names each row: the one label infer_label_col trusts,
    or the only column besides the value. "" when a second dimension would
    make it a guess."""
    from core.contribution_analysis import infer_label_col

    label = infer_label_col(rows, value)
    if label:
        return label
    others = _naming_columns(rows, value)
    return others[0] if len(others) == 1 else ""


# ── The tools that need new data: through the governed pipeline ─────────────


async def _query(call: ToolCall) -> ToolResult:
    from core.investigation import run_query_tool

    return await run_query_tool(
        account_id=call.account_id, portal_user=call.portal_user, run_id=call.run_id,
        question=call.values["question"],
    )


async def _drill(call: ToolCall) -> ToolResult:
    """The step's own question, broken down by one more dimension -- asked
    as a question, so its SQL is written, checked and run like any other."""
    from core.investigation import run_query_tool

    asked = str(call.sources["step"].get("question") or "").strip()
    result = await run_query_tool(
        account_id=call.account_id, portal_user=call.portal_user, run_id=call.run_id,
        question=f"{asked} by {call.values['dimension']}",
    )
    return ToolResult(ok=result.ok, kind="drill", question=call.label, brief=result.brief,
                      result_id=result.result_id, row_count=result.row_count, error=result.error,
                      columns=result.columns)


# ── The tools that compute on a result this run already has ─────────────────


async def _compare(call: ToolCall) -> ToolResult:
    """Two results side by side, label by label: what each holds and the change."""
    from core.contribution_analysis import infer_numeric_col

    first, second = _rows(call, "step"), _rows(call, "other_step")
    value = call.values.get("column") or infer_numeric_col(first)
    label = _label_column(first, value)
    other_value = _column(second, value) or infer_numeric_col(second)
    other_label = _column(second, label) or _label_column(second, other_value)
    if not (value and label and other_value and other_label):
        return _failed(call, _t("investigation.tool.error.nothing_to_compare", call.lang))
    before = {str(row.get(label)): row.get(value) for row in first}
    after = {str(row.get(other_label)): row.get(other_value) for row in second}
    # A result with a row per label per month matched on the label alone
    # would keep one month of each and call it the label's value.
    for step, rows_of, labels in (("step", first, before), ("other_step", second, after)):
        if len(labels) < len(rows_of):
            return _failed(call, _t("investigation.tool.error.not_one_row_per_label", call.lang,
                                    step=call.values[step], label=label))
    rows: list[dict[str, Any]] = []
    for key in list(before) + [k for k in after if k not in before]:
        a, b = _number(before.get(key)), _number(after.get(key))
        change = (b - a) if a is not None and b is not None else None
        pct = (change / a * 100.0) if change is not None and a else None
        rows.append({label: key, "first": a, "second": b, "change": change, "change_pct": pct})
    if not any(row["change"] is not None for row in rows):
        return _failed(call, _t("investigation.tool.error.nothing_to_compare", call.lang))
    rows.sort(key=lambda row: -abs(row["change"] or 0.0))
    items = "; ".join(
        _t("investigation.tool.brief.compare_item", call.lang, label=row[label], first=_fmt(row["first"], call.lang),
           second=_fmt(row["second"], call.lang), change=_fmt(row["change"], call.lang),
           pct=_pct(row["change_pct"], call.lang))
        for row in rows[:_BRIEF_ITEMS] if row["change"] is not None
    )
    brief = _t("investigation.tool.brief.compare", call.lang, value=value,
               first=call.values["step"], second=call.values["other_step"], items=items)
    return _ok(call, brief, rows, _keep(call, rows))


def _number(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _move(value: float, lang: str | None) -> str:
    """A move as prose states one: its size, and its direction in words.

    Not as a signed figure. A summary says "sales fell 13", and the synthesis
    check (core.investigation_planner.verify_synthesis) looks for each figure
    the summary gives in the evidence: a brief that recorded -13 made the
    correct sentence read as an invented figure.
    """
    if abs(value) < 1e-9:
        return _t("investigation.tool.move.none", lang)
    key = "investigation.tool.move.up" if value > 0 else "investigation.tool.move.down"
    return _t(key, lang, amount=_fmt(abs(value), lang))


def _two_steps(call: ToolCall, *, besides: str = "") -> tuple[list[dict], list[dict], str, str, str] | None:
    """Both steps' rows, the member column, and the value column on each side.

    The second step's rows come back keyed by the first step's member column,
    so a member named REGION in one result and REGION_NM in the other is still
    one member. `besides` is a column the value is not: the quantity, for a
    price-volume-mix split, which would otherwise be inferred as the value.
    """
    from core.contribution_analysis import infer_numeric_col

    def infer(rows: list[dict]) -> str:
        return infer_numeric_col([{k: v for k, v in row.items() if k != besides} for row in rows])

    first, second = _rows(call, "step"), _rows(call, "other_step")
    value = call.values.get("column") or infer(first)
    label = _label_column(first, value)
    other_value = _column(second, value) or infer(second)
    other_label = _column(second, label) or _label_column(second, other_value)
    if not (value and label and other_value and other_label):
        return None
    if other_label != label:
        second = [{**row, label: row.get(other_label)} for row in second]
    return first, second, label, value, other_value


def _bridge_refusal(call: ToolCall, refused: Any, label: str) -> ToolResult:
    if refused.reason == "not_one_row_per_member":
        return _failed(call, _t("investigation.tool.error.not_one_row_per_label", call.lang,
                                step=call.values["step"], label=label))
    if refused.reason == "non_additive":
        return _failed(call, _t("investigation.tool.error.bridge_non_additive", call.lang,
                                column=refused.detail.get("column", "")))
    if refused.reason == "no_priced_members":
        return _failed(call, _t("investigation.tool.error.no_priced_members", call.lang,
                                quantity=refused.detail.get("quantity", "")))
    return _failed(call, _t("investigation.tool.error.nothing_to_compare", call.lang))


async def _bridge(call: ToolCall) -> ToolResult:
    """From one step's total to the other's: each member's move, and the rest."""
    from core.variance_bridge import BridgeRefused, variance_bridge

    found_steps = _two_steps(call)
    if found_steps is None:
        return _failed(call, _t("investigation.tool.error.nothing_to_compare", call.lang))
    first, second, label, value, other_value = found_steps
    try:
        found = variance_bridge(first, second, member=label, value=value, other_value=other_value)
    except BridgeRefused as refused:
        return _bridge_refusal(call, refused, label)
    step, other = call.values["step"], call.values["other_step"]
    rows: list[dict[str, Any]] = [
        {label: _t("investigation.tool.bridge.start", call.lang, step=step), "kind": "start", "amount": found.start}]
    rows += [{label: member, "kind": "member", "amount": move} for member, move in found.steps]
    if found.rest_members:
        rows.append({label: _t("investigation.tool.bridge.rest", call.lang, count=found.rest_members),
                     "kind": "rest", "amount": found.rest})
    rows.append({label: _t("investigation.tool.bridge.end", call.lang, step=other), "kind": "end", "amount": found.end})
    items = "; ".join(
        _t("investigation.tool.brief.bridge_item", call.lang, label=member, change=_move(move, call.lang))
        for member, move in found.steps
    ) or _t("investigation.tool.brief.none", call.lang)
    rest = (_t("investigation.tool.brief.bridge_rest", call.lang, count=found.rest_members,
               change=_move(found.rest, call.lang)) if found.rest_members else "")
    share = found.explained_share
    brief = _t("investigation.tool.brief.bridge", call.lang, value=value, first=step, second=other,
               start=_fmt(found.start, call.lang), end=_fmt(found.end, call.lang),
               change=_move(found.change, call.lang), items=items, rest=rest,
               share=_pct(share * 100.0 if share is not None else None, call.lang))
    return _ok(call, brief, rows, _keep(call, rows))


async def _price_volume_mix(call: ToolCall) -> ToolResult:
    """The move between two steps split into volume, mix, price, new and lost."""
    from core.variance_bridge import BridgeRefused, price_volume_mix

    quantity = call.values["quantity"]
    found_steps = _two_steps(call, besides=quantity)
    if found_steps is None:
        return _failed(call, _t("investigation.tool.error.nothing_to_compare", call.lang))
    first, second, label, value, other_value = found_steps
    other_quantity = _column(second, quantity)
    if not other_quantity:
        return _failed(call, _t("investigation.tool.error.no_column", call.lang, column=quantity,
                                step=call.values["other_step"]))
    if other_value != value or other_quantity != quantity:
        second = [{**row, value: row.get(other_value), quantity: row.get(other_quantity)} for row in second]
    try:
        found = price_volume_mix(first, second, member=label, quantity=quantity, value=value)
    except BridgeRefused as refused:
        return _bridge_refusal(call, refused, label)
    parts = found.parts()
    rows: list[dict[str, Any]] = [
        {"part": _t(f"investigation.tool.pvm.{name}", call.lang), "kind": name, "amount": amount}
        for name, amount in parts.items() if name in {"volume", "mix", "price"} or amount
    ]

    def drivers(pairs: tuple[tuple[str, float], ...]) -> str:
        return ", ".join(f"{member} ({_move(effect, call.lang)})" for member, effect in pairs[:3]) \
            or _t("investigation.tool.brief.none", call.lang)

    brief = _t("investigation.tool.brief.price_volume_mix", call.lang, value=value, quantity=quantity,
               first=call.values["step"], second=call.values["other_step"],
               start=_fmt(found.start, call.lang), end=_fmt(found.end, call.lang),
               change=_move(found.change, call.lang),
               **{name: _move(amount, call.lang) for name, amount in parts.items()},
               price_drivers=drivers(found.price_drivers), mix_drivers=drivers(found.mix_drivers))
    return _ok(call, brief, rows, _keep(call, rows))


async def _forecast(call: ToolCall) -> ToolResult:
    from core.chart_policy import aggregate_only_gate_passes
    from core.forecast import compute_forecast
    from core.forecast_gate import assess_fit, evaluate_forecast_request

    snapshot, rows = call.sources["step"], _rows(call)
    # A cut result never reaches here: _check refuses it for every tool
    # that reads rows, so the gate's own truncation rule is left at its default.
    decision = evaluate_forecast_request(
        rows,
        question=str(snapshot.get("question") or ""),
        horizon=call.values["periods"],
        policy_allows_derived_visual=aggregate_only_gate_passes(
            account_id=call.account_id, portal_user=call.portal_user,
            event=SimpleNamespace(platform="portal"), sql=str(snapshot.get("sql") or ""),
            db_type=_db_type(call, snapshot), what="Forecast",
        ),
    )
    projected: list[dict] = []
    meta: dict = {}
    if decision.allowed:
        projected = compute_forecast(
            rows, decision.period_col, decision.value_col, decision.horizon,
            model=decision.model, seasonal_period=decision.seasonal_period,
        )
        meta = (projected[0] if projected else {}).get("__forecast_meta") or {}
        decision = assess_fit(decision, meta.get("r2"), meta.get("backtest_mape"))
    if not decision.allowed:
        return _failed(call, decision.caveat or _t(
            "investigation.tool.error.forecast_refused", call.lang, reason=decision.reason_code))
    points = "; ".join(
        _t("investigation.tool.brief.forecast_point", call.lang, period=row.get(decision.period_col),
           value=_fmt(row.get("forecast_value"), call.lang), low=_fmt(row.get("forecast_low"), call.lang),
           high=_fmt(row.get("forecast_high"), call.lang))
        for row in projected if row.get("is_forecast")
    )
    model = str(meta.get("model") or decision.model)
    brief = _t("investigation.tool.brief.forecast", call.lang, value=decision.value_col,
               period=decision.period_col,
               model=_t(f"investigation.tool.model.{model}", call.lang) if model in _MODELS else model,
               r2=_fmt(float(meta.get("r2") or 0.0), call.lang, 2), points=points)
    return _ok(call, brief, projected, _keep(call, projected))


async def _anomalies(call: ToolCall) -> ToolResult:
    from core.anomaly_detection import detect_anomalies, infer_value_col

    rows = _rows(call)
    value = _value_column(call, rows, infer_value_col)
    if not value:
        return _failed(call, _t("investigation.tool.error.no_value_column", call.lang))
    found = detect_anomalies(rows, value)
    if found.mean_val is None:
        # Fewer than four values: the detector flags nothing, which is not
        # the same as finding nothing that stands out.
        return _failed(call, _t("investigation.tool.error.too_few_for_anomalies", call.lang))
    naming = _naming_columns(rows, value)
    items = "; ".join(
        _t("investigation.tool.brief.item", call.lang, label=" · ".join(str(row.get(name)) for name in naming),
           value=_fmt(row.get(value), call.lang))
        for row in found.flagged[:_BRIEF_ITEMS]
    ) or _t("investigation.tool.brief.none", call.lang)
    brief = _t("investigation.tool.brief.anomalies", call.lang, flagged=_fmt(found.flagged_rows, call.lang),
               total=_fmt(found.total_rows, call.lang), value=value,
               method=_t(f"investigation.tool.method.{found.method}", call.lang,
                         threshold=_fmt(found.threshold, call.lang)),
               items=items)
    return _ok(call, brief, found.rows, _keep(call, found.rows))


async def _correlate(call: ToolCall) -> ToolResult:
    from core.correlation_analysis import annotate_rows_with_correlation, compute_correlation, infer_corr_cols

    rows = _rows(call)
    inferred_x, inferred_y = infer_corr_cols(rows, str(call.sources["step"].get("question") or ""))
    x, y = call.values.get("column") or inferred_x, call.values.get("other_column") or inferred_y
    if not x or not y or x == y:
        return _failed(call, _t("investigation.tool.error.two_columns", call.lang))
    found = compute_correlation(rows, x, y)
    if found.pearson_r is None:
        return _failed(call, _t("investigation.tool.error.too_few_pairs", call.lang))
    strength = re.sub(r"\W+", "_", str(found.interpretation or "")).strip("_") or "negligible"
    brief = _t("investigation.tool.brief.correlate", call.lang, x=x, y=y, r=_fmt(found.pearson_r, call.lang, 2),
               strength=_t(f"investigation.tool.correlation.{strength}", call.lang), n=_fmt(found.n, call.lang))
    derived = annotate_rows_with_correlation(rows, found)
    return _ok(call, brief, derived, _keep(call, derived))


async def _distribution(call: ToolCall) -> ToolResult:
    from core.distribution_analysis import compute_histogram, infer_histogram_col

    rows = _rows(call)
    value = _value_column(call, rows, infer_histogram_col)
    if not value:
        return _failed(call, _t("investigation.tool.error.no_value_column", call.lang))
    bins = compute_histogram(rows, value)
    if not bins or "bin_label" not in bins[0]:
        return _failed(call, _t("investigation.tool.error.too_few_values", call.lang))
    items = "; ".join(
        _t("investigation.tool.brief.bin", call.lang, bin=row.get("bin_label"), count=_fmt(row.get("count"), call.lang),
           pct=_pct(row.get("frequency_pct"), call.lang))
        for row in bins
    )
    brief = _t("investigation.tool.brief.distribution", call.lang, value=value, n=_fmt(len(rows), call.lang),
               items=items)
    return _ok(call, brief, bins, _keep(call, bins))


async def _cohort(call: ToolCall) -> ToolResult:
    from core.cohort_analysis import build_cohort_summary, compute_cohort_matrix, infer_cohort_cols

    rows = _rows(call)
    cohort, period, value = infer_cohort_cols(rows)
    if not (cohort and period and value):
        return _failed(call, _t("investigation.tool.error.not_a_cohort", call.lang))
    matrix = compute_cohort_matrix(rows, cohort, period, value)
    if not matrix or "cohort" not in matrix[0]:
        return _failed(call, _t("investigation.tool.error.not_a_cohort", call.lang))
    summary = build_cohort_summary(matrix)
    brief = _t("investigation.tool.brief.cohort", call.lang, cohorts=summary.cohort_count,
               periods=summary.period_count, retention=_pct(summary.overall_avg_retention, call.lang),
               best=summary.best_cohort, worst=summary.worst_cohort)
    return _ok(call, brief, matrix, _keep(call, matrix))


async def _funnel(call: ToolCall) -> ToolResult:
    from core.funnel_analysis import build_funnel_summary, compute_funnel, infer_funnel_cols

    rows = _rows(call)
    stage, count = infer_funnel_cols(rows)
    if not (stage and count):
        return _failed(call, _t("investigation.tool.error.not_a_funnel", call.lang))
    summary = build_funnel_summary(rows, stage, count)
    enriched = compute_funnel(rows, stage, count)
    brief = _t("investigation.tool.brief.funnel", call.lang, stages=summary.stage_count,
               top=_fmt(summary.top_of_funnel, call.lang), bottom=_fmt(summary.bottom_of_funnel, call.lang),
               overall=_pct(summary.overall_conversion_pct, call.lang), stage=summary.biggest_drop_stage,
               drop=_pct(summary.biggest_drop_pct, call.lang))
    return _ok(call, brief, enriched, _keep(call, enriched))


async def _contribution(call: ToolCall) -> ToolResult:
    from core.contribution_analysis import compute_contribution, infer_numeric_col

    rows = _rows(call)
    value = _value_column(call, rows, infer_numeric_col)
    if not value:
        return _failed(call, _t("investigation.tool.error.no_value_column", call.lang))
    label = _label_column(rows, value)
    if not label:
        return _failed(call, _t("investigation.tool.error.no_label", call.lang, step=call.values["step"]))
    shares = compute_contribution(rows, value, label)
    # Withheld, as the pipeline withholds them, for a measure that does not
    # add up across the rows (a balance, a percentage) or a zero total.
    if all(row.get("contribution_pct") is None for row in shares):
        return _failed(call, _t("investigation.tool.error.no_shares", call.lang, value=value))
    items = "; ".join(
        _t("investigation.tool.brief.share", call.lang, label=row.get(label), value=_fmt(row.get(value), call.lang),
           pct=_pct(row.get("contribution_pct"), call.lang))
        for row in shares[:_BRIEF_ITEMS]
    )
    top = sum(float(row.get("contribution_pct") or 0.0) for row in shares[:3])
    brief = _t("investigation.tool.brief.contribution", call.lang, value=value, label=label,
               items=items, top=_pct(top, call.lang))
    return _ok(call, brief, shares, _keep(call, shares))


def _step(help_key: str) -> ToolInput:
    return ToolInput("step", "step", help_key)


REGISTRY: dict[str, Tool] = {tool.name: tool for tool in (
    Tool("query", "investigation.tool.summary.query",
         (ToolInput("question", "text", "investigation.tool.input.question"),), _query,
         reads_rows=False),
    Tool("drill", "investigation.tool.summary.drill",
         (_step("investigation.tool.input.step"),
          ToolInput("dimension", "text", "investigation.tool.input.dimension")), _drill,
         reads_rows=False),
    Tool("compare", "investigation.tool.summary.compare",
         (_step("investigation.tool.input.step"),
          ToolInput("other_step", "step", "investigation.tool.input.other_step"),
          ToolInput("column", "column", "investigation.tool.input.column", required=False)), _compare),
    Tool("bridge", "investigation.tool.summary.bridge",
         (_step("investigation.tool.input.step"),
          ToolInput("other_step", "step", "investigation.tool.input.other_step"),
          ToolInput("column", "column", "investigation.tool.input.column", required=False)), _bridge),
    Tool("price_volume_mix", "investigation.tool.summary.price_volume_mix",
         (_step("investigation.tool.input.step"),
          ToolInput("other_step", "step", "investigation.tool.input.other_step"),
          ToolInput("quantity", "column", "investigation.tool.input.quantity"),
          ToolInput("column", "column", "investigation.tool.input.column", required=False)),
         _price_volume_mix),
    Tool("forecast", "investigation.tool.summary.forecast",
         (_step("investigation.tool.input.step"),
          ToolInput("periods", "count", "investigation.tool.input.periods", required=False,
                    low=1, high=12, default=3)), _forecast),
    Tool("anomalies", "investigation.tool.summary.anomalies",
         (_step("investigation.tool.input.step"),
          ToolInput("column", "column", "investigation.tool.input.column", required=False)), _anomalies),
    Tool("correlate", "investigation.tool.summary.correlate",
         (_step("investigation.tool.input.step"),
          ToolInput("column", "column", "investigation.tool.input.column", required=False),
          ToolInput("other_column", "column", "investigation.tool.input.other_column", required=False)),
         _correlate),
    Tool("distribution", "investigation.tool.summary.distribution",
         (_step("investigation.tool.input.step"),
          ToolInput("column", "column", "investigation.tool.input.column", required=False)), _distribution),
    Tool("cohort", "investigation.tool.summary.cohort", (_step("investigation.tool.input.step"),), _cohort),
    Tool("funnel", "investigation.tool.summary.funnel", (_step("investigation.tool.input.step"),), _funnel),
    Tool("contribution", "investigation.tool.summary.contribution",
         (_step("investigation.tool.input.step"),
          ToolInput("column", "column", "investigation.tool.input.column", required=False)), _contribution),
)}


def describe_tools(lang: str | None = None) -> str:
    """The catalogue as the planner reads it: each tool, what it does, its inputs."""
    lines = []
    for tool in REGISTRY.values():
        inputs = ", ".join(
            f"{spec.name}{'' if spec.required else '?'} ({_t(spec.help, lang)})" for spec in tool.inputs
        )
        lines.append(f"- {tool.name}: {_t(tool.summary, lang)} Inputs: {inputs}.")
    return "\n".join(lines)


def _check(tool: Tool, inputs: dict, steps: list, session_id: str, lang: str | None) -> tuple[dict, dict]:
    """The call's inputs as the tool runs them, and the snapshot of each step it reads."""
    from core.result_cache import result_cache

    values: dict[str, Any] = {}
    sources: dict[str, dict] = {}
    by_index = {int(step.index): step for step in steps}
    for spec in tool.inputs:
        raw: Any = inputs.get(spec.name)
        if raw in (None, ""):
            if spec.required:
                raise ToolCallError(_t("investigation.tool.error.missing_input", lang, name=spec.name))
            if spec.kind == "count":
                values[spec.name] = spec.default
            continue
        if spec.kind == "text":
            values[spec.name] = re.sub(r"\s+", " ", str(raw)).strip()[:400]
        elif spec.kind == "count":
            try:
                count = int(raw)
            except (TypeError, ValueError):
                count = spec.low - 1
            if not spec.low <= count <= spec.high:
                raise ToolCallError(_t("investigation.tool.error.count_range", lang, name=spec.name,
                                       low=spec.low, high=spec.high))
            values[spec.name] = count
        elif spec.kind == "step":
            try:
                index = int(raw)
            except (TypeError, ValueError):
                index = 0
            step = by_index.get(index)
            if step is None:
                raise ToolCallError(_t("investigation.tool.error.no_such_step", lang, step=raw))
            if not step.result.ok or not step.result.result_id:
                raise ToolCallError(_t("investigation.tool.error.step_found_nothing", lang, step=index))
            snapshot = result_cache.get_snapshot(session_id, step.result.result_id)
            if not snapshot.get("rows"):
                raise ToolCallError(_t("investigation.tool.error.expired", lang, step=index))
            # The pipeline refuses row-level statistics over the head of a
            # result cut at its row cap; so does every tool that reads rows.
            if tool.reads_rows and (snapshot.get("metadata") or {}).get("rows_truncated"):
                raise ToolCallError(_t("investigation.tool.error.truncated", lang, step=index))
            values[spec.name] = index
            sources[spec.name] = snapshot
    for spec in tool.inputs:
        if spec.kind == "column" and spec.name in inputs and inputs[spec.name] not in (None, ""):
            rows = (sources.get(spec.of) or {}).get("rows") or []
            column = _column(rows, str(inputs[spec.name]))
            if not column:
                raise ToolCallError(_t("investigation.tool.error.no_column", lang,
                                       column=str(inputs[spec.name])[:60], step=values.get(spec.of, "?")))
            values[spec.name] = column
    unknown = set(inputs) - {spec.name for spec in tool.inputs}
    if unknown:
        raise ToolCallError(_t("investigation.tool.error.unknown_input", lang, name=sorted(unknown)[0]))
    return values, sources


def _label(tool: Tool, values: dict, lang: str | None) -> str:
    if tool.name == "query":
        return str(values.get("question") or "")
    return _t(f"investigation.tool.label.{tool.name}", lang, **{
        key: value for key, value in values.items()
    })


async def run_tool(
    name: str,
    inputs: dict | None,
    *,
    account_id: str,
    portal_user: dict,
    run_id: str,
    steps: list,
    lang: str | None = None,
) -> ToolResult:
    """Run one tool call of this investigation. Never raises: a call that
    cannot run is a failed step with the reason, and the loop goes on."""
    from core.investigation import investigation_session_id

    tool = REGISTRY.get(str(name or "").strip().lower())
    if tool is None:
        return ToolResult(ok=False, kind="tool", question=str(name or "")[:40],
                          error=_t("investigation.tool.error.unknown_tool", lang, name=str(name)[:40]))
    session_id = investigation_session_id(account_id, portal_user, run_id)
    call = ToolCall(tool, account_id, portal_user, run_id, session_id, lang)
    try:
        call.values, call.sources = _check(tool, dict(inputs or {}), steps, session_id, lang)
    except ToolCallError as exc:
        return ToolResult(ok=False, kind=tool.name, question=tool.name, error=str(exc))
    call.label = _label(tool, call.values, lang)
    try:
        return await tool.run(call)
    except Exception as exc:  # noqa: BLE001 - one failed step must not end the run
        log.warning("Investigation tool %s failed: %s", tool.name, exc, exc_info=True)
        return _failed(call, _t("investigation.tool.error.failed", lang))
