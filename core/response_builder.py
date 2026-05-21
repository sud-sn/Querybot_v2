from __future__ import annotations

import logging
import math
import re
from statistics import mean, median, stdev
from typing import Any

log = logging.getLogger("querybot.response_builder")

_PREVIEW_ROW_CAP = 200


def _format_number(value: Any) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(num):
        return str(value)
    if abs(num) >= 1000:
        return f"{num:,.0f}" if num.is_integer() else f"{num:,.2f}"
    return f"{num:.0f}" if num.is_integer() else f"{num:.2f}"


def _numeric_cols(rows: list[dict]) -> list[str]:
    cols: list[str] = []
    if not rows:
        return cols
    for h in rows[0].keys():
        ok = True
        seen = False
        for r in rows:
            v = r.get(h)
            if v is None or v == "":
                continue
            seen = True
            try:
                float(str(v).replace(",", ""))
            except (TypeError, ValueError):
                ok = False
                break
        if ok and seen:
            cols.append(h)
    return cols


def _text_cols(rows: list[dict], numeric_cols: list[str]) -> list[str]:
    return [h for h in (rows[0].keys() if rows else []) if h not in numeric_cols]


def _looks_temporal(values: list[str]) -> bool:
    sample = " ".join(v.lower() for v in values[:8] if v)
    # Full month names and long tokens — safe for substring match
    substr_tokens = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "week", "month", "quarter", "year", "date",
    ]
    # Short abbreviations — need word boundary to avoid false positives
    wb_tokens = [
        "jan", "feb", "mar", "apr", "jun", "jul", "aug",
        "sep", "oct", "nov", "dec",
    ]
    return (
        bool(re.search(r"\b\d{4}[-/]\d{1,2}([-/]\d{1,2})?\b", sample))
        or any(tok in sample for tok in substr_tokens)
        or any(re.search(r"\b" + tok + r"\b", sample) for tok in wb_tokens)
    )


def _safe_pct_change(first: float, last: float) -> float | None:
    if first == 0:
        return None
    return ((last - first) / abs(first)) * 100.0


def _to_float(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _best_label(question: str, label_col: str, value_col: str) -> str:
    q = question.strip().rstrip("?")
    if len(q.split()) >= 4:
        return q
    return f"{value_col.replace('_', ' ').title()} by {label_col.replace('_', ' ').title()}"


def _extract_limit(sql: str) -> int | None:
    if not sql:
        return None
    patterns = [
        r"\btop\s*\(\s*(\d+)\s*\)",
        r"\btop\s+(\d+)\b",
        r"\blimit\s+(\d+)\b",
        r"\bfetch\s+first\s+(\d+)\s+rows?\s+only\b",
        r"\bfetch\s+next\s+(\d+)\s+rows?\s+only\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, sql, re.I)
        if match:
            try:
                return int(match.group(1))
            except (TypeError, ValueError):
                return None
    return None


def infer_result_scope(
    rows: list[dict],
    question: str,
    sql: str = "",
    *,
    mode: str = "table",
) -> dict[str, Any]:
    row_count = len(rows)
    lower_sql = (sql or "").lower()
    explicit_limit = _extract_limit(sql)
    preview_cap_hit = row_count >= _PREVIEW_ROW_CAP and explicit_limit is None and row_count > 0
    filtered_subset = " where " in f" {lower_sql} "
    was_limited = explicit_limit is not None or preview_cap_hit

    scope: dict[str, Any] = {
        "kind": mode,
        "question": question,
        "row_count": row_count,
        "limit_value": explicit_limit,
        "was_limited": was_limited,
        "is_preview": preview_cap_hit,
        "filtered_subset": filtered_subset,
        "is_top_n": False,
        "n": None,
        "is_complete_distribution": False,
        "is_complete_series": False,
    }

    if mode == "ranking":
        if explicit_limit is not None:
            scope["is_top_n"] = True
            scope["n"] = explicit_limit
        scope["is_complete_distribution"] = not was_limited
    elif mode == "time_series":
        scope["is_complete_series"] = not was_limited
        if explicit_limit is not None:
            scope["n"] = explicit_limit

    badge = "Returned result"
    note = "This reflects the rows returned by the query."
    if scope["is_top_n"]:
        n = scope["n"] or row_count
        if n == 1:
            badge = "Top result only"
            note = "This result is based on the top-ranked row only, not the full distribution."
        else:
            badge = f"Top {n} only"
            note = f"This result is based only on the top {n} returned rows."
    elif mode == "ranking" and scope["is_complete_distribution"]:
        badge = "Full distribution"
        note = "This result reflects the full returned distribution."
    elif mode == "time_series" and scope["is_complete_series"]:
        badge = "Full series"
        note = "This result reflects the full returned time series."
    elif scope["is_preview"]:
        badge = "Preview"
        note = "This result is a preview because the returned rows are capped for display."
    elif filtered_subset:
        badge = "Filtered subset"
        note = "This result reflects a filtered subset defined by the query conditions."

    scope["badge"] = badge
    scope["note"] = note
    scope["analysis_note"] = (
        "Interpret this as a returned slice rather than a complete picture."
        if was_limited and mode in {"ranking", "time_series"}
        else note
    )
    return scope


def build_answer(
    rows: list[dict],
    question: str,
    result_scope: dict | None = None,
) -> dict:
    scope = result_scope or infer_result_scope(rows, question)
    if not rows:
        return {
            "headline": "No matching data was found for this question.",
            "short_value": "0 rows",
            "comparison": "Try adjusting the filters or time range.",
            "scope_badge": scope.get("badge", ""),
            "scope_note": scope.get("note", ""),
        }

    numeric_cols = _numeric_cols(rows)
    text_cols = _text_cols(rows, numeric_cols)

    if len(rows) == 1 and len(rows[0]) == 1:
        col = next(iter(rows[0].keys()))
        val = rows[0][col]
        return {
            "headline": f"{col.replace('_', ' ').title()}: {_format_number(val)}",
            "short_value": _format_number(val),
            "comparison": scope.get("badge") or "Single-value result",
            "scope_badge": scope.get("badge", ""),
            "scope_note": scope.get("note", ""),
        }

    if numeric_cols and text_cols:
        label_col = text_cols[0]
        value_col = numeric_cols[0]
        ordered = sorted(rows, key=lambda r: _to_float(r.get(value_col)) or 0.0, reverse=True)
        labels = [str(r.get(label_col, "")) for r in rows]
        if _looks_temporal(labels):
            first = rows[0]
            last = rows[-1]
            first_val = _to_float(first.get(value_col)) or 0.0
            last_val = _to_float(last.get(value_col)) or 0.0
            direction = "up" if last_val > first_val else "down" if last_val < first_val else "flat"
            headline = f"{str(last.get(label_col, 'Latest period'))} closed at {_format_number(last_val)}."
            comparison = scope.get("badge") or f"Trend is {direction} versus {_format_number(first_val)} at the start"
            return {
                "headline": headline,
                "short_value": _format_number(last_val),
                "comparison": comparison,
                "scope_badge": scope.get("badge", ""),
                "scope_note": scope.get("note", ""),
            }
        best = ordered[0]
        best_label = str(best.get(label_col, 'Top result'))
        best_value = _to_float(best.get(value_col)) or 0.0
        comparison = scope.get("badge") or f"Across {len(rows)} results"
        if scope.get("is_top_n") and (scope.get("n") or 0) == 1:
            headline = f"Top-ranked result: {best_label} at {_format_number(best_value)}."
            comparison = "This card shows only the leading row"
        else:
            headline = f"{best_label} leads at {_format_number(best_value)}."
        if len(ordered) > 1 and not scope.get("is_top_n"):
            second = ordered[1]
            second_value = _to_float(second.get(value_col)) or 0.0
            delta = best_value - second_value
            comparison = f"{_format_number(delta)} above the next result"
        return {
            "headline": headline,
            "short_value": _format_number(best_value),
            "comparison": comparison,
            "scope_badge": scope.get("badge", ""),
            "scope_note": scope.get("note", ""),
        }

    if numeric_cols:
        col = numeric_cols[0]
        values = [_to_float(r.get(col)) or 0.0 for r in rows]
        return {
            "headline": f"Returned {len(rows)} rows for {question.strip().rstrip('?') or 'this query'}.",
            "short_value": _format_number(values[0]),
            "comparison": scope.get("badge") or f"Range {_format_number(min(values))} to {_format_number(max(values))}",
            "scope_badge": scope.get("badge", ""),
            "scope_note": scope.get("note", ""),
        }

    # Pure text result — e.g. a list of names. Show a preview in the chip.
    first_col = list(rows[0].keys())[0]
    preview_items = [str(r.get(first_col, "")) for r in rows[:3] if r.get(first_col)]
    preview = ", ".join(preview_items)
    if len(rows) > 3:
        preview += f", +{len(rows) - 3} more"
    return {
        "headline": f"Found {len(rows)} result{'s' if len(rows) != 1 else ''} for: {question.strip().rstrip('?') or 'your query'}",
        "short_value": f"{len(rows)} rows",
        "comparison": scope.get("badge") or preview or "Review the records below",
        "scope_badge": scope.get("badge", ""),
        "scope_note": scope.get("note", ""),
    }


def summarize_result_context(rows: list[dict], question: str, sql: str = "") -> dict:
    numeric_cols = _numeric_cols(rows)
    text_cols = _text_cols(rows, numeric_cols)
    ctx: dict[str, Any] = {
        "question": question,
        "row_count": len(rows),
        "numeric_cols": numeric_cols,
        "text_cols": text_cols,
        "mode": "table",
        "chartable": False,
    }
    if not rows:
        ctx["mode"] = "empty"
        ctx["result_scope"] = infer_result_scope(rows, question, sql, mode="empty")
        return ctx

    if numeric_cols and text_cols:
        label_col = text_cols[0]
        value_col = numeric_cols[0]
        labels = [str(r.get(label_col, "")) for r in rows]
        values = [_to_float(r.get(value_col)) or 0.0 for r in rows]
        ctx.update({
            "label_col": label_col,
            "value_col": value_col,
            "labels": labels,
            "values": values,
            "min_value": min(values),
            "max_value": max(values),
            "avg_value": mean(values),
            "median_value": median(values),
            "chartable": True,
        })
        ordered = sorted(rows, key=lambda r: _to_float(r.get(value_col)) or 0.0, reverse=True)
        ctx["top_items"] = [
            {"label": str(r.get(label_col, "")), "value": _to_float(r.get(value_col)) or 0.0}
            for r in ordered[:5]
        ]
        if _looks_temporal(labels):
            first, last = values[0], values[-1]
            pct = _safe_pct_change(first, last)
            diffs = [values[i] - values[i - 1] for i in range(1, len(values))]
            ctx.update({
                "mode": "time_series",
                "first_label": labels[0],
                "last_label": labels[-1],
                "first_value": first,
                "last_value": last,
                "pct_change": pct,
                "avg_step_change": mean(diffs) if diffs else 0.0,
                "volatility": mean(abs(d) for d in diffs) if diffs else 0.0,
                "comparison_stats": {
                    "first_period": labels[0],
                    "first_value": first,
                    "last_period": labels[-1],
                    "last_value": last,
                    "absolute_change": round(last - first, 2),
                    "pct_change": round(pct, 2) if pct is not None else None,
                },
            })
        else:
            ctx["mode"] = "ranking"
            total = sum(values)
            top_items = ctx.get("top_items") or []
            leader = top_items[0] if top_items else None
            runner_up = top_items[1] if len(top_items) > 1 else None
            ctx["distribution_stats"] = {
                "category_count": len(set(labels)),
                "spread": round(max(values) - min(values), 2) if values else 0.0,
                "median_value": round(median(values), 2) if values else 0.0,
                "top_3_share_pct": round(sum(item["value"] for item in top_items[:3]) / total * 100, 1) if total > 0 and top_items else None,
                "std_dev": round(stdev(values), 2) if len(values) >= 3 else None,
            }
            comparison_stats = {}
            if leader:
                comparison_stats.update({
                    "leader": leader["label"],
                    "leader_value": leader["value"],
                    "leader_share_pct": round(leader["value"] / total * 100, 1) if total > 0 else None,
                })
            if leader and runner_up:
                comparison_stats.update({
                    "runner_up": runner_up["label"],
                    "runner_up_value": runner_up["value"],
                    "gap": round(leader["value"] - runner_up["value"], 2),
                })
            ctx["comparison_stats"] = comparison_stats
        ctx["result_scope"] = infer_result_scope(rows, question, sql, mode=ctx["mode"])
        return ctx

    if numeric_cols:
        value_col = numeric_cols[0]
        values = [_to_float(r.get(value_col)) or 0.0 for r in rows]
        ctx.update({
            "mode": "numeric_table",
            "value_col": value_col,
            "values": values,
            "min_value": min(values),
            "max_value": max(values),
            "avg_value": mean(values),
            "median_value": median(values),
            "distribution_stats": {
                "spread": round(max(values) - min(values), 2),
                "std_dev": round(stdev(values), 2) if len(values) >= 3 else None,
            },
        })
        ctx["result_scope"] = infer_result_scope(rows, question, sql, mode="numeric_table")
        return ctx

    ctx["mode"] = "text_table"
    ctx["result_scope"] = infer_result_scope(rows, question, sql, mode="text_table")
    return ctx


def _dynamic_actions(ctx: dict) -> list[dict]:
    """
    Action buttons for the result card.

    "Why this pattern?" has been removed — the Insight Layer (inline
    summary sentence + anomaly callouts + follow-up suggestions) surfaces
    this analysis automatically without requiring a button click.
    """
    mode = ctx.get("mode")
    actions: list[dict] = []
    if mode == "time_series":
        actions = [
            {"id": "explain", "label": "Explain result"},
            {"id": "analyze", "label": "Analyze trend"},
            {"id": "compare", "label": "Compare periods"},
        ]
        if ctx.get("row_count", 0) >= 3:
            actions.append({"id": "predict", "label": "Predict next period"})
    elif mode == "ranking":
        actions = [
            {"id": "explain", "label": "Explain result"},
            {"id": "analyze", "label": "Analyze ranking"},
            {"id": "compare", "label": "Compare top results"},
        ]
    elif mode == "numeric_table":
        actions = [
            {"id": "explain", "label": "Explain result"},
            {"id": "analyze", "label": "Analyze spread"},
        ]
    else:
        actions = [{"id": "explain", "label": "Explain result"}]
    return actions


# ── Insight Layer helpers — pure statistics, no LLM call ─────────────────────

def _build_insight_summary(rows: list[dict], ctx: dict, brief: dict) -> str:
    """
    Generate a one-sentence plain-English summary from the data brief.

    Purely stat-driven — no LLM call, no latency added.
    Returns empty string when there is not enough structure to say anything useful.
    """
    mode = ctx.get("mode", "table")
    row_count = len(rows)

    if mode == "single_value":
        col = (brief.get("value_column") or "").replace("_", " ").title()
        val = brief.get("value", "")
        return f"{col}: {_format_number(val)}." if col else ""

    if mode == "time_series":
        ts = brief.get("time_series") or {}
        direction = ts.get("direction", "stable")
        pct = ts.get("overall_pct_change")
        first = ts.get("first_period", "")
        last_ = ts.get("last_period", "")
        value_col = (ctx.get("value_col") or "").replace("_", " ").title()
        dir_word = {"increasing": "up", "decreasing": "down", "stable": "flat"}.get(direction, direction)
        if pct is not None:
            base = f"{value_col} trended {dir_word} {abs(pct):.1f}% from {first} to {last_}."
        else:
            base = f"{value_col} remained {dir_word} between {first} and {last_}."
        peak = ts.get("peak") or {}
        if peak and direction in ("increasing", "decreasing"):
            base += f" Peak: {_format_number(peak.get('value', 0))} in {peak.get('period', '')}."
        return base

    if mode == "ranking":
        cat = brief.get("category_breakdown") or {}
        top5 = cat.get("top_5") or []
        if top5:
            leader = top5[0]
            leader_share = cat.get("leader_share_pct")
            label_col = (cat.get("label_column") or "").replace("_", " ").lower()
            count = cat.get("category_count", row_count)
            share_str = f" ({leader_share}% of total)" if leader_share else ""
            return (
                f"{leader['label']} leads at {_format_number(leader['value'])}{share_str}"
                f" across {count} {label_col or 'entries'}."
            )

    if mode == "numeric_table":
        value_col = (ctx.get("value_col") or "").replace("_", " ").title()
        mn = ctx.get("min_value", 0)
        mx = ctx.get("max_value", 0)
        avg = ctx.get("avg_value", 0)
        return (
            f"{row_count} records — {value_col} ranges "
            f"{_format_number(mn)} to {_format_number(mx)}, avg {_format_number(avg)}."
        )

    return ""


def _build_anomaly_callouts(brief: dict) -> list[dict]:
    """
    Detect notable statistical patterns from the data brief.

    Returns a list of up to 3 callout dicts:
      {"type": str, "icon": str, "message": str, "severity": "warning"|"success"|"info"}

    Severity → UI colour:
      warning  = amber   (drops, streaks)
      success  = green   (gains)
      info     = blue    (concentration, outliers)
    """
    callouts: list[dict] = []
    mode = brief.get("mode", "table")

    if mode == "time_series":
        ts = brief.get("time_series") or {}
        drop = ts.get("biggest_period_drop") or {}
        gain = ts.get("biggest_period_gain") or {}
        streak = ts.get("longest_decline_streak", 0)

        if drop.get("pct_change") is not None and drop["pct_change"] < -10:
            callouts.append({
                "type": "drop", "icon": "↓",
                "message": (
                    f"Biggest drop: {drop['from_period']} → {drop['to_period']} "
                    f"({drop['pct_change']:.1f}%)"
                ),
                "severity": "warning",
            })
        if gain.get("pct_change") is not None and gain["pct_change"] > 10:
            callouts.append({
                "type": "gain", "icon": "↑",
                "message": (
                    f"Biggest gain: {gain['from_period']} → {gain['to_period']} "
                    f"(+{gain['pct_change']:.1f}%)"
                ),
                "severity": "success",
            })
        if streak >= 3:
            callouts.append({
                "type": "streak", "icon": "⚠",
                "message": f"{streak} consecutive periods of decline",
                "severity": "warning",
            })

    elif mode == "ranking":
        cat = brief.get("category_breakdown") or {}
        for _col, stats in (brief.get("numeric_summaries") or {}).items():
            conc = stats.get("top_3_concentration_pct")
            if conc and conc >= 80:
                callouts.append({
                    "type": "concentration", "icon": "◉",
                    "message": f"Top 3 entries account for {conc}% of total — highly concentrated",
                    "severity": "info",
                })
                break
        leader_share = cat.get("leader_share_pct")
        top5 = cat.get("top_5") or []
        if leader_share and leader_share >= 50 and top5 and len(callouts) < 2:
            callouts.append({
                "type": "dominance", "icon": "★",
                "message": f"{top5[0]['label']} holds {leader_share}% of the total",
                "severity": "info",
            })

    # Outlier detection across numeric columns (all modes)
    if len(callouts) < 3:
        for col, stats in (brief.get("numeric_summaries") or {}).items():
            std = stats.get("std_dev")
            mean_v = stats.get("mean")
            mx_v = stats.get("max")
            if std and mean_v and std > 0 and mx_v and mx_v > mean_v + 2.5 * std:
                callouts.append({
                    "type": "outlier", "icon": "◆",
                    "message": (
                        f"Outlier in {col.replace('_', ' ')}: "
                        f"max {_format_number(mx_v)} vs avg {_format_number(mean_v)}"
                    ),
                    "severity": "info",
                })
                break

    return callouts[:3]


def _why_it_matters(ctx: dict) -> str:
    mode = ctx.get("mode")
    if mode == "time_series":
        pct = ctx.get("pct_change")
        if pct is None:
            return "The direction is visible, but the starting point is too close to zero for a stable percentage comparison."
        direction = "higher" if pct > 0 else "lower" if pct < 0 else "flat"
        return f"This leaves the latest period {abs(pct):.1f}% {direction} than the starting period, which is useful for judging whether performance is improving or deteriorating over time."
    if mode == "ranking":
        top_items = ctx.get("top_items") or []
        if len(top_items) >= 2:
            gap = (top_items[0]["value"] - top_items[1]["value"])
            return f"The leading category is ahead by {_format_number(gap)}, so performance is concentrated rather than evenly distributed across categories."
        return "This identifies the leading category directly, which helps focus follow-up analysis on where performance is strongest or weakest."
    if mode == "numeric_table":
        return "The spread between the minimum and maximum values shows whether the result is tightly grouped or highly variable."
    if mode == "empty":
        return "No impact can be inferred because the result set is empty under the current filters."
    return "This result is best used as a starting point for a more targeted follow-up question."


def build_analysis_response(action: str, contract: dict) -> dict:
    """
    Synchronous fallback for action button clicks when LLM insight is unavailable.
    
    The preferred path is the async generate_analysis_response() below, which
    uses the LLM insight engine. This function is kept as a zero-latency
    fallback that works without an LLM call.
    """
    mode = contract.get("mode")
    scope = contract.get("result_scope") or {}
    title = "Analysis"
    body = ""
    bullets: list[str] = []
    secondary = scope.get("analysis_note", "")

    if action == "explain":
        title = "Result explanation"
        if mode == "time_series":
            last_value = contract.get("last_value", 0.0)
            body = f"This result shows {scope.get('badge', 'the returned series').lower()}. The latest returned period is {contract.get('last_label', 'the latest period')} at {_format_number(last_value)}."
            pct = contract.get("pct_change")
            if pct is not None:
                direction = "up" if pct > 0 else "down" if pct < 0 else "flat"
                bullets.append(f"Overall direction across the returned series: {direction} ({abs(pct):.1f}%)")
        elif mode == "ranking":
            top_items = contract.get("top_items") or []
            if top_items:
                body = f"This result shows {scope.get('badge', 'the returned ranking').lower()}. {top_items[0]['label']} ranks first at {_format_number(top_items[0]['value'])}."
                if len(top_items) > 1 and not scope.get("is_top_n"):
                    body += f" The next highest returned result is {top_items[1]['label']} at {_format_number(top_items[1]['value'])}."
        elif mode == "numeric_table":
            body = f"The result contains {contract.get('row_count', 0)} numeric rows with values ranging from {_format_number(contract.get('min_value', 0.0))} to {_format_number(contract.get('max_value', 0.0))}."
        else:
            body = "This result is already concise and does not require deeper interpretation without an additional breakdown."

    elif action == "analyze":
        title = "Detailed analysis"
        if mode == "time_series":
            body = f"The returned time series varies between {_format_number(contract.get('min_value', 0.0))} and {_format_number(contract.get('max_value', 0.0))}, with an average of {_format_number(contract.get('avg_value', 0.0))}."
            bullets = [
                f"Average step change: {_format_number(contract.get('avg_step_change', 0.0))}",
                f"Observed volatility per step: {_format_number(contract.get('volatility', 0.0))}",
            ]
        elif mode == "ranking":
            stats = contract.get("distribution_stats") or {}
            if stats.get("top_3_share_pct") is not None:
                body = f"The ranking is concentrated: the top three returned categories account for {stats['top_3_share_pct']:.1f}% of the total."
            else:
                body = "The ranking pattern should be read as a distribution, not just a winner."
            bullets = [
                f"Category count in returned result: {stats.get('category_count', contract.get('row_count', 0))}",
                f"Spread from highest to lowest returned value: {_format_number(stats.get('spread', 0.0))}",
            ]
            if stats.get("std_dev") is not None:
                bullets.append(f"Standard deviation across returned values: {_format_number(stats['std_dev'])}")
        elif mode == "numeric_table":
            body = f"The numeric values average {_format_number(contract.get('avg_value', 0.0))} across {contract.get('row_count', 0)} rows."
            bullets = [
                f"Spread: {_format_number((contract.get('distribution_stats') or {}).get('spread', 0.0))}",
                f"Median: {_format_number(contract.get('median_value', 0.0))}",
            ]
        else:
            body = "There is not enough structure in this result for a richer analysis without a more specific breakdown."

    elif action == "compare":
        title = "Comparison view"
        if mode == "time_series":
            cmp = contract.get("comparison_stats") or {}
            body = f"{cmp.get('last_period', 'Latest period')} is {_format_number(cmp.get('last_value', 0.0))} versus {_format_number(cmp.get('first_value', 0.0))} in {cmp.get('first_period', 'the first period')}."
            if cmp.get("pct_change") is not None:
                bullets.append(f"Percent change across returned periods: {abs(cmp['pct_change']):.1f}%")
        elif mode == "ranking":
            cmp = contract.get("comparison_stats") or {}
            if cmp.get("leader") and cmp.get("runner_up"):
                body = f"{cmp['leader']} is ahead of {cmp['runner_up']} by {_format_number(cmp.get('gap', 0.0))}."
                if cmp.get("leader_share_pct") is not None:
                    bullets.append(f"Leader share of returned total: {cmp['leader_share_pct']:.1f}%")
            elif cmp.get("leader"):
                body = f"{cmp['leader']} is the only comparable returned category, so there is no runner-up to compare."
            else:
                body = "There is not enough comparable structure in this result for a comparison."
        else:
            body = "This result does not yet have enough comparable structure for a useful comparison card."

    elif action == "why":
        title = "Business framing"
        body = _why_it_matters(contract)
        bullets = [
            "This framing is based on the returned result shape, not on inferred root causes.",
        ]

    elif action == "predict":
        title = "Forecast"
        if mode == "time_series" and contract.get("row_count", 0) >= 3:
            vals = contract.get("values") or []
            labels = contract.get("labels") or []
            xs = list(range(len(vals)))
            n = len(xs)
            mean_x = sum(xs) / n
            mean_y = sum(vals) / n
            denom = sum((x - mean_x) ** 2 for x in xs) or 1.0
            slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, vals)) / denom
            intercept = mean_y - slope * mean_x
            next_x = n
            forecast = intercept + slope * next_x
            forecast = max(forecast, 0.0)
            vol = contract.get("volatility", 0.0)
            conf = "low" if vol > abs(slope) * 3 else "medium" if vol > abs(slope) else "moderate"
            body = f"A simple trend projection puts the next period near {_format_number(forecast)}."
            secondary = f"This is a {conf}-confidence directional estimate based only on the returned series, not a full forecasting model."
            bullets = [
                f"Last observed period: {labels[-1]} at {_format_number(vals[-1])}",
                f"Average step change used in projection: {_format_number(contract.get('avg_step_change', 0.0))}",
            ]
        else:
            body = "Prediction is only available when the result contains a clear time series with at least three periods."

    else:
        title = "Analysis"
        body = "This follow-up action is not supported for the current result."

    return {
        "type": "assistant_analysis",
        "action": action,
        "title": title,
        "body": body,
        "secondary": secondary,
        "bullets": bullets,
        "source_question": contract.get("question", ""),
        "mode": mode,
        "result_scope": scope,
    }


async def generate_analysis_response(
    action: str,
    rows: list[dict],
    question: str,
    provider: str,
    model: str,
    api_key: str,
    follow_up: str = "",
    original_sql: str = "",
    db_cfg: dict | None = None,
    context: str = "",
    known_tables: set[str] | None = None,
    **extra_kwargs,
) -> dict:
    """
    Async LLM-powered analysis — the preferred path for action buttons
    and "why" follow-up questions.
    
    Falls back to the synchronous build_analysis_response() if the LLM
    call fails.
    """
    from core.insight import (
        generate_insight,
        generate_drilldown_insight,
        is_insight_question,
    )

    try:
        # "why" questions with drill-down capability
        if action == "why" and db_cfg and original_sql and context:
            return await generate_drilldown_insight(
                rows=rows,
                question=question,
                follow_up=follow_up,
                original_sql=original_sql,
                db_cfg=db_cfg,
                context=context,
                provider=provider,
                model=model,
                api_key=api_key,
                known_tables=known_tables,
                business_context=context,
                **extra_kwargs,
            )

        # Standard action buttons (explain, analyze, compare, predict)
        return await generate_insight(
            rows=rows,
            question=question,
            action=action,
            follow_up=follow_up,
            provider=provider,
            model=model,
            api_key=api_key,
            business_context=context,
            original_sql=original_sql,
            **extra_kwargs,
        )

    except Exception as e:
        log.error("Dynamic analysis failed, falling back to static: %s", e)
        # Fall back to synchronous/static analysis
        ctx = summarize_result_context(rows, question, sql=original_sql)
        return build_analysis_response(action, ctx)


def build_assistant_response(*, question: str, rows: list[dict], sql: str, duration_ms: int, chart: dict | None = None, data_source: str | None = None) -> dict:
    from core.insight import compute_data_brief
    ctx = summarize_result_context(rows, question, sql=sql)
    answer = build_answer(rows, question, ctx.get("result_scope"))
    brief = compute_data_brief(
        rows,
        question,
        result_scope=ctx.get("result_scope"),
        context=ctx,
    )

    # ── Insight Layer — pure-stats, zero-latency ─────────────────────────────
    # Generate a summary sentence and anomaly callouts from the data brief.
    # These are computed entirely from statistics — no LLM call, no extra latency.
    insight_summary  = _build_insight_summary(rows, ctx, brief)
    anomaly_callouts = _build_anomaly_callouts(brief)

    # Include the actual row data (bounded) so the frontend can render a table.
    # Frontend is the ONLY consumer of raw rows — LLM insight path never sees these.
    # We cap at 200 rows to keep WebSocket payload reasonable; full set is already
    # limited by run_query(max_rows=200).
    headers: list[str] = []
    display_rows: list[dict] = []
    if rows:
        headers = list(rows[0].keys())
        # Send formatted string values for reliable frontend display
        for r in rows[:_PREVIEW_ROW_CAP]:
            display_rows.append({h: _safe_cell(r.get(h)) for h in headers})

    return {
        "type": "assistant_response",
        "question": question,
        "answer": answer,
        "chart": chart,
        "insight_summary": insight_summary,
        "anomaly_callouts": anomaly_callouts,
        "summary": {"executive_summary": ""},
        "next_actions": _dynamic_actions(ctx),
        "analysis_contract": ctx,
        "data_brief": brief,
        "result_scope": ctx.get("result_scope", {}),
        "data": {
            "headers": headers,
            "rows": display_rows,
            "total_rows": len(rows),
            "truncated": len(rows) > _PREVIEW_ROW_CAP,
        },
        "trust": {
            "sql": sql,
            "row_count": len(rows),
            "duration_label": f"{duration_ms}ms" if duration_ms < 1000 else f"{duration_ms/1000:.1f}s",
            "data_source": data_source or "",
            "scope_badge": ctx.get("result_scope", {}).get("badge", ""),
        },
    }


def _safe_cell(val: Any) -> str:
    """Format a cell value for frontend display. Returns a string."""
    if val is None:
        return ""
    if isinstance(val, float):
        if val != val or val in (float("inf"), float("-inf")):
            return ""
        if val.is_integer():
            return f"{int(val):,}"
        return f"{val:,.4f}".rstrip("0").rstrip(".")
    if isinstance(val, int):
        return f"{val:,}" if abs(val) >= 1000 else str(val)
    return str(val)
