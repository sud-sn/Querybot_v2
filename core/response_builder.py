from __future__ import annotations

import logging
import math
import re
from statistics import mean, median, stdev
from typing import Any

log = logging.getLogger("querybot.response_builder")

_PREVIEW_ROW_CAP = 200
_RESULT_FORMATS = {"number", "currency", "percentage", "date", "text"}

_CURRENCY_NAME_RE = re.compile(
    r"\b(revenue|amount|cost|price|total|sales|charge|fee|payment|spend|"
    r"value|income|profit|loss|margin|earning|billing|invoice|budget|"
    r"gross|net|balance|credit|debit|cash|dollar|usd|gbp|eur|salary|"
    r"wage|commission|rebate|discount|tax|surcharge|reimbursement)\b",
    re.IGNORECASE,
)
_PERCENT_NAME_RE = re.compile(r"\b(percent|percentage|pct|rate|ratio|share)\b", re.IGNORECASE)
_DATE_NAME_RE = re.compile(r"\b(date|period|year|month|quarter|week|day)\b", re.IGNORECASE)
_VALUE_TOKENS = {
    "amount", "avg", "average", "balance", "charge", "cost", "count",
    "earning", "fee", "gross", "income", "invoice", "loss", "margin",
    "net", "payment", "pct", "percent", "percentage", "price", "profit",
    "quantity", "rate", "ratio", "revenue", "sales", "share", "spend",
    "sum", "tax", "total", "value",
}
_DIMENSION_TOKENS = {
    "code", "date", "day", "description", "flag", "id", "identifier", "item",
    "key", "month", "name", "num", "number", "period", "product", "rank",
    "warehouse", "week", "year",
}
_FORMAT_STOP_TOKENS = {
    "a", "an", "and", "as", "by", "for", "from", "in", "is", "my", "of",
    "on", "per", "show", "the", "to", "total", "what", "with",
}


def _format_number(value: Any, fmt: str | None = None) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(num):
        return str(value)
    fmt = _normalise_result_format(fmt)
    if fmt == "currency":
        return f"${num:,.2f}"
    if fmt == "percentage":
        return f"{num:,.2f}%".replace(".00%", "%")
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


def _normalise_result_format(value: Any) -> str:
    fmt = str(value or "number").strip().lower()
    return fmt if fmt in _RESULT_FORMATS else "number"


def _normalise_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _term_tokens(value: Any) -> set[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(value or ""))
    text = text.replace("_", " ").replace("-", " ")
    return {
        tok.lower()
        for tok in re.findall(r"[A-Za-z0-9]+", text)
        if tok and tok.lower() not in _FORMAT_STOP_TOKENS
    }


def _metric_tokens(metric: dict) -> set[str]:
    raw = " ".join(
        str(metric.get(k) or "")
        for k in ("name", "synonyms", "description", "required_columns")
    )
    tokens = _term_tokens(raw)
    return {t for t in tokens if t not in _FORMAT_STOP_TOKENS}


def _is_dimension_like_column(column: str) -> bool:
    tokens = _term_tokens(column)
    if not tokens:
        return False
    if tokens & _VALUE_TOKENS:
        return False
    return bool(tokens & _DIMENSION_TOKENS)


def _format_matches_column_name(fmt: str, column: str) -> bool:
    if fmt == "currency":
        return bool(_CURRENCY_NAME_RE.search(column))
    if fmt == "percentage":
        return bool(_PERCENT_NAME_RE.search(column))
    if fmt == "date":
        return bool(_DATE_NAME_RE.search(column))
    return False


def _columns_for_metric_format(
    rows: list[dict],
    metric: dict,
    *,
    strict: bool = False,
) -> list[str]:
    if not rows:
        return []

    fmt = _normalise_result_format(metric.get("result_format"))
    if fmt == "number" and not strict:
        return []

    headers = list(rows[0].keys())
    numeric_cols = set(_numeric_cols(rows))
    text_cols = {h for h in headers if h not in numeric_cols}
    metric_terms = _metric_tokens(metric)

    if fmt in {"currency", "percentage", "number"}:
        candidates = [h for h in headers if h in numeric_cols]
    elif fmt == "date":
        candidates = [h for h in headers if h not in numeric_cols or _format_matches_column_name(fmt, h)]
    else:
        candidates = [h for h in headers if h in text_cols]

    scored: list[tuple[int, str]] = []
    for header in candidates:
        header_terms = _term_tokens(header)
        term_match = bool(metric_terms and (header_terms & metric_terms))
        format_name_match = _format_matches_column_name(fmt, header)
        value_name_match = bool(header_terms & _VALUE_TOKENS)
        score = 0
        if term_match:
            score += 5
        if format_name_match:
            score += 4
        if value_name_match and (strict or term_match or format_name_match):
            score += 2
        if _is_dimension_like_column(header) and not format_name_match:
            score -= 4
        if score > 0:
            scored.append((score, header))

    if scored:
        scored.sort(key=lambda item: (-item[0], headers.index(item[1])))
        return [h for _, h in scored]

    value_candidates = [h for h in candidates if not _is_dimension_like_column(h)]
    if strict and len(value_candidates) == 1:
        return value_candidates
    if strict and value_candidates:
        return value_candidates
    return []


def build_column_formats(
    rows: list[dict],
    display_context: dict | None = None,
    explicit_formats: dict | None = None,
) -> dict[str, str]:
    """
    Build a header -> display-format map for the frontend.

    Metric result_format should drive presentation only. SQL remains numeric/date
    friendly so sorting, charting, CSV export, and result-chat calculations keep
    working.
    """
    if not rows:
        return {}

    headers = list(rows[0].keys())
    by_norm = {_normalise_key(h): h for h in headers}
    formats: dict[str, str] = {}

    for raw_col, raw_fmt in (explicit_formats or {}).items():
        header = by_norm.get(_normalise_key(raw_col))
        fmt = _normalise_result_format(raw_fmt)
        # Allow explicit "number" through — it lets callers override currency
        # heuristics for columns that happen to have monetary-sounding names.
        if header:
            formats[header] = fmt

    ctx = display_context or {}
    metrics = ctx.get("metrics") if isinstance(ctx, dict) else []
    if isinstance(metrics, dict):
        metrics = [metrics]
    if not isinstance(metrics, list):
        metrics = []
    strict = (ctx.get("format_scope") if isinstance(ctx, dict) else "") == "metric_registry"

    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        fmt = _normalise_result_format(metric.get("result_format"))
        if fmt == "number" and not strict:
            continue
        for header in _columns_for_metric_format(rows, metric, strict=strict):
            formats.setdefault(header, fmt)

    return formats


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


def _to_float_z(value: Any) -> float:
    """Like _to_float but returns 0.0 for None/unparseable.

    Use this instead of ``_to_float(v) or 0.0`` because the ``or`` idiom
    silently zeroes legitimate negative values (e.g. -500.0 is falsy).
    """
    v = _to_float(value)
    return v if v is not None else 0.0


def _display_label(column: str) -> str:
    return re.sub(r"\s+", " ", str(column or "").replace("_", " ")).strip().title()


def _find_header_by_norm(headers: list[str], norm: str) -> str:
    if not norm:
        return ""
    for header in headers:
        if _normalise_key(header) == norm:
            return header
    for header in headers:
        h_norm = _normalise_key(header)
        if h_norm.endswith(norm) or norm.endswith(h_norm):
            return header
    return ""


def detect_null_metric_issue(rows: list[dict]) -> dict[str, Any] | None:
    """
    Detect diagnostic rows where records matched, but a requested metric was
    NULL/missing for every matched record.
    """
    if len(rows) != 1 or not rows[0]:
        return None
    row = rows[0]
    headers = list(row.keys())
    matched_header = next(
        (
            h for h in headers
            if _normalise_key(h) in {"matchedrows", "rowcount", "matchcount", "matchedrecords"}
        ),
        "",
    )
    matched_rows = _to_float(row.get(matched_header)) if matched_header else None
    if matched_rows is None or matched_rows <= 0:
        return None

    issues: list[dict[str, Any]] = []
    for header in headers:
        norm = _normalise_key(header)
        if not (norm.startswith("nonnull") and norm.endswith("rows")):
            continue
        non_null_rows = _to_float(row.get(header))
        if non_null_rows is None or non_null_rows > 0:
            continue
        metric_norm = norm[len("nonnull"):-len("rows")]
        metric_header = _find_header_by_norm(headers, metric_norm)
        if not metric_header:
            continue
        metric_value = row.get(metric_header)
        metric_num = _to_float(metric_value)
        if metric_value not in (None, "") and metric_num not in (0, 0.0):
            continue
        issues.append({
            "metric_column": metric_header,
            "non_null_column": header,
            "matched_rows": int(matched_rows),
            "non_null_rows": int(non_null_rows),
            "value": metric_value,
        })

    if not issues:
        return None
    return {
        "matched_rows": int(matched_rows),
        "matched_column": matched_header,
        "issues": issues,
    }


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
    column_formats: dict | None = None,
) -> dict:
    scope = result_scope or infer_result_scope(rows, question)
    column_formats = column_formats or {}
    if not rows:
        return {
            "headline": "No matching data was found for this question.",
            "short_value": "0 rows",
            "comparison": "Try adjusting the filters or time range.",
            "scope_badge": scope.get("badge", ""),
            "scope_note": scope.get("note", ""),
        }

    null_issue = detect_null_metric_issue(rows)
    if null_issue:
        issue = null_issue["issues"][0]
        metric_col = issue["metric_column"]
        fmt = column_formats.get(metric_col)
        value = _format_number(_to_float(rows[0].get(metric_col)) or 0, fmt)
        metric_label = _display_label(metric_col)
        matched = null_issue["matched_rows"]
        return {
            "headline": f"{metric_label}: {value} because all matched values are missing.",
            "short_value": value,
            "comparison": f"{matched} matching records, 0 non-null {metric_label} values",
            "scope_badge": "Missing metric values",
            "scope_note": (
                f"The filter matched {matched} records, but the requested metric column "
                f"had no non-null values in those records."
            ),
        }

    numeric_cols = _numeric_cols(rows)
    text_cols = _text_cols(rows, numeric_cols)

    if len(rows) == 1 and len(rows[0]) == 1:
        col = next(iter(rows[0].keys()))
        val = rows[0][col]
        fmt = column_formats.get(col)
        return {
            "headline": f"{col.replace('_', ' ').title()}: {_format_number(val, fmt)}",
            "short_value": _format_number(val, fmt),
            "comparison": scope.get("badge") or "Single-value result",
            "scope_badge": scope.get("badge", ""),
            "scope_note": scope.get("note", ""),
        }

    if numeric_cols and text_cols:
        label_col = text_cols[0]
        value_col = numeric_cols[0]
        value_fmt = column_formats.get(value_col)
        ordered = sorted(rows, key=lambda r: _to_float_z(r.get(value_col)), reverse=True)
        labels = [str(r.get(label_col, "")) for r in rows]
        if _looks_temporal(labels):
            first = rows[0]
            last = rows[-1]
            first_val = _to_float_z(first.get(value_col))
            last_val = _to_float_z(last.get(value_col))
            direction = "up" if last_val > first_val else "down" if last_val < first_val else "flat"
            headline = f"{str(last.get(label_col, 'Latest period'))} closed at {_format_number(last_val, value_fmt)}."
            comparison = scope.get("badge") or f"Trend is {direction} versus {_format_number(first_val, value_fmt)} at the start"
            return {
                "headline": headline,
                "short_value": _format_number(last_val, value_fmt),
                "comparison": comparison,
                "scope_badge": scope.get("badge", ""),
                "scope_note": scope.get("note", ""),
            }
        best = ordered[0]
        best_label = str(best.get(label_col, 'Top result'))
        best_value = _to_float_z(best.get(value_col))
        comparison = scope.get("badge") or f"Across {len(rows)} results"
        if scope.get("is_top_n") and (scope.get("n") or 0) == 1:
            headline = f"Top-ranked result: {best_label} at {_format_number(best_value, value_fmt)}."
            comparison = "This card shows only the leading row"
        else:
            headline = f"{best_label} leads at {_format_number(best_value, value_fmt)}."
        if len(ordered) > 1 and not scope.get("is_top_n"):
            second = ordered[1]
            second_value = _to_float_z(second.get(value_col))
            delta = best_value - second_value
            comparison = f"{_format_number(delta, value_fmt)} above the next result"
        return {
            "headline": headline,
            "short_value": _format_number(best_value, value_fmt),
            "comparison": comparison,
            "scope_badge": scope.get("badge", ""),
            "scope_note": scope.get("note", ""),
        }

    if numeric_cols:
        col = numeric_cols[0]
        value_fmt = column_formats.get(col)
        values = [_to_float_z(r.get(col)) for r in rows]
        return {
            "headline": f"Returned {len(rows)} rows for {question.strip().rstrip('?') or 'this query'}.",
            "short_value": _format_number(values[0], value_fmt),
            "comparison": scope.get("badge") or f"Range {_format_number(min(values), value_fmt)} to {_format_number(max(values), value_fmt)}",
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

    # Single scalar result (one row, one column) — set mode so _build_insight_summary
    # can produce a meaningful sentence instead of falling through to return "".
    if len(rows) == 1 and len(rows[0]) == 1:
        col = next(iter(rows[0].keys()))
        ctx.update({
            "mode": "single_value",
            "value_column": col,
            "value": rows[0][col],
            "chartable": False,
        })
        ctx["result_scope"] = infer_result_scope(rows, question, sql, mode="single_value")
        return ctx

    if numeric_cols and text_cols:
        label_col = text_cols[0]
        value_col = numeric_cols[0]
        labels = [str(r.get(label_col, "")) for r in rows]
        values = [_to_float_z(r.get(value_col)) for r in rows]
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
        ordered = sorted(rows, key=lambda r: _to_float_z(r.get(value_col)), reverse=True)
        ctx["top_items"] = [
            {"label": str(r.get(label_col, "")), "value": _to_float_z(r.get(value_col))}
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
        values = [_to_float_z(r.get(value_col)) for r in rows]
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


_CHIP_THRESHOLD = 68  # minimum confidence to surface a chip


def compute_chip_eligibility(
    ctx: dict,
    brief: dict | None = None,
    semantic_plan: dict | None = None,
) -> list[dict]:
    """
    Signal-based chip eligibility.  Replaces the old mode-only ``_dynamic_actions``.

    Every chip is scored against actual data-brief signals — not just the result
    *mode*.  Chips that score below ``_CHIP_THRESHOLD`` are silently omitted so
    the user only sees actions the data can actually support.

    Returns a list of ``{id, label, confidence, pre_context}`` dicts ordered by
    a fixed display priority (explain → analyze → compare → … → decide).
    The ``pre_context`` string is a one-liner explaining *why* the chip is
    relevant (shown as a hover tooltip / subtitle on the button).
    """
    brief = brief or {}
    mode       = ctx.get("mode", "table")
    row_count  = ctx.get("row_count", 0)
    ts         = brief.get("time_series") or {}
    cat        = brief.get("category_breakdown") or {}
    dist       = ctx.get("distribution_stats") or {}
    cmp_stats  = ctx.get("comparison_stats") or {}

    chips: list[dict] = []

    def _add(id_: str, label: str, confidence: int, pre_context: str = "") -> None:
        if confidence >= _CHIP_THRESHOLD:
            chips.append({
                "id": id_,
                "label": label,
                "confidence": confidence,
                "pre_context": pre_context,
            })

    # ── time_series chips ────────────────────────────────────────────────────
    if mode == "time_series":
        direction    = ts.get("direction") or "stable"
        period_count = ts.get("period_count") or row_count
        pct_change   = ts.get("overall_pct_change")
        if pct_change is None:
            pct_change = ctx.get("pct_change") or 0.0

        # compare_period: only meaningful when overall change is non-trivial
        if period_count >= 2 and pct_change is not None and abs(pct_change) >= 3.0:
            sign = "+" if pct_change > 0 else ""
            _add(
                "compare", "Compare periods",
                82 if abs(pct_change) >= 10 else 73,
                f"{sign}{pct_change:.1f}% overall change",
            )

        # diagnose: root-cause chip for significant movement
        if pct_change is not None and abs(pct_change) >= 5.0:
            _change_word = "drop" if pct_change < 0 else "rise"
            _add(
                "diagnose", f"Why the {_change_word}?",
                88 if abs(pct_change) >= 10 else 80,
                f"{abs(pct_change):.1f}% {_change_word} — identify what drove this",
            )

        # compare_prior: available when the semantic model knows the date role
        if semantic_plan and semantic_plan.get("enabled"):
            has_date_role = any(
                f.get("role") == "date_dimension"
                for f in (semantic_plan.get("fields") or [])
            )
            if has_date_role:
                _add(
                    "compare_prior", "vs prior period", 70,
                    "Fetch the same metric for the previous cycle",
                )

    # ── ranking chips ────────────────────────────────────────────────────────
    elif mode == "ranking":
        # contribution: % share breakdown useful for ranking results
        leader      = cmp_stats.get("leader") or "top item"
        leader_share = cmp_stats.get("leader_share_pct")
        if leader_share is not None and row_count >= 2:
            _add(
                "contribution", "Show % contribution", 78,
                f"{leader} holds {leader_share:.0f}% of total",
            )

    # ── drill_dim — "Break down by X" chips ─────────────────────────────────
    # Show at most 2 dimensions that are available in the semantic model but
    # not already present in the current result.
    if semantic_plan and semantic_plan.get("enabled") and row_count >= 1:
        result_cols_upper = {
            c.upper()
            for c in (ctx.get("numeric_cols") or []) + (ctx.get("text_cols") or [])
        }
        drill_count = 0
        for dim in (semantic_plan.get("available_dimensions") or []):
            if drill_count >= 2:
                break
            dc = (dim.get("display_column") or "").upper()
            name = (dim.get("name") or "").strip()
            if not dc or not name:
                continue
            if dc in result_cols_upper:
                continue  # already in the result — skip
            conf = 75 if dim.get("status") == "approved" else 68
            _add(
                f"drill_dim:{name}",
                f"Break down by {name}",
                conf,
                f"Add {name} dimension to this result",
            )
            drill_count += 1

    # ── download_csv — available for any non-empty result ────────────────────
    if row_count >= 1 and mode != "empty":
        _add(
            "download_csv", "Download CSV", 85,
            f"{row_count} row{'s' if row_count != 1 else ''} ready to export",
        )

    # Fixed display order. drill_dim chips slot between contribution and download.
    _fixed = {
        "compare": 0, "diagnose": 1, "compare_prior": 2,
        "contribution": 3,
        "download_csv": 90,
    }
    chips.sort(key=lambda c: (
        _fixed.get(c["id"], 50 if c["id"].startswith("drill_dim:") else 99),
        c["id"],
    ))
    return chips


def _dynamic_actions(ctx: dict) -> list[dict]:
    """Deprecated — delegates to ``compute_chip_eligibility``.

    Kept for backward compatibility with any call sites that haven't been
    updated.  No ``brief`` or ``semantic_plan`` context is available here so
    only mode-level signals are used.
    """
    return compute_chip_eligibility(ctx)


# ── Insight Layer helpers — pure statistics, no LLM call ─────────────────────

def _build_insight_summary(rows: list[dict], ctx: dict, brief: dict) -> str:
    """
    Generate a one-sentence plain-English summary from the data brief.

    Purely stat-driven — no LLM call, no latency added.
    Returns empty string when there is not enough structure to say anything useful.
    """
    mode = ctx.get("mode", "table")
    row_count = len(rows)

    null_issue = detect_null_metric_issue(rows)
    if null_issue:
        issue = null_issue["issues"][0]
        metric = _display_label(issue["metric_column"])
        return (
            f"{null_issue['matched_rows']} records matched, but {metric} is missing "
            "for every matched row."
        )

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


def _build_decision_signal(ctx: dict, brief: dict, anomaly_callouts: list[dict]) -> dict:
    """
    Deterministic 'so-what' line — zero LLM, zero latency.

    Turns the existing statistical brief + anomaly callouts into one
    decision-oriented sentence the user can act on, plus a tone for UI colour.

    Returns:
        {"line": str, "tone": "watch"|"positive"|"neutral", "basis": str}
        or {} when there is nothing decision-relevant to say.
    """
    mode = brief.get("mode") or ctx.get("mode", "table")

    if mode == "ranking":
        cat = brief.get("category_breakdown") or {}
        leader_share = cat.get("leader_share_pct")
        # concentration from numeric summaries (top-3)
        conc = None
        for _c, stats in (brief.get("numeric_summaries") or {}).items():
            if stats.get("top_3_concentration_pct") is not None:
                conc = stats["top_3_concentration_pct"]
                break
        top5 = cat.get("top_5") or []
        leader = top5[0]["label"] if top5 else ""
        if conc is not None and conc >= 80:
            return {
                "line": f"Top entries drive {conc:.0f}% of the total — concentration risk if any one is lost.",
                "tone": "watch", "basis": "concentration",
            }
        if leader_share is not None and leader_share >= 50:
            return {
                "line": f"{leader} alone holds {leader_share:.0f}% of the total — a single point of dependency.",
                "tone": "watch", "basis": "dominance",
            }
        if leader_share is not None:
            return {
                "line": f"Volume is spread across the field — no single entry exceeds {max(leader_share,1):.0f}%; broadly diversified.",
                "tone": "positive", "basis": "spread",
            }

    if mode == "time_series":
        ts = brief.get("time_series") or {}
        direction = ts.get("direction", "stable")
        pct = ts.get("overall_pct_change")
        streak = ts.get("longest_decline_streak", 0)
        if direction == "decreasing" and (streak >= 3 or (pct is not None and pct <= -10)):
            return {
                "line": f"Sustained downward trend ({pct:+.0f}% overall) — worth investigating before it compounds." if pct is not None
                        else "Sustained downward trend — worth investigating before it compounds.",
                "tone": "watch", "basis": "decline",
            }
        if direction == "increasing" and pct is not None and pct >= 10:
            return {
                "line": f"Momentum is building (+{pct:.0f}% overall) — confirm it is sustainable, not a one-off spike.",
                "tone": "positive", "basis": "growth",
            }
        if direction == "stable":
            return {
                "line": "Metric is holding steady over the period — no urgent action indicated.",
                "tone": "neutral", "basis": "stable",
            }

    if mode == "numeric_table":
        outliers = [c for c in anomaly_callouts if c.get("type") == "outlier"]
        if outliers:
            return {
                "line": "One or more values sit well above normal — review for data quality or a genuine signal before acting.",
                "tone": "watch", "basis": "outlier",
            }

    if mode == "single_value":
        # Restate with directional framing only if a comparison exists.
        comp = ctx.get("comparison") or brief.get("comparison")
        if comp:
            return {"line": f"{comp} — factor this into the decision.", "tone": "neutral", "basis": "single"}

    return {}


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

    elif action == "decide":
        title = "Recommended next step"
        # Deterministic advisory fallback (no LLM). Reuse the decision-signal
        # rules so the static path still gives a useful, safe recommendation.
        signal = _build_decision_signal(contract, contract, [])
        if signal.get("line"):
            body = signal["line"]
        else:
            body = ("This result is a useful starting point. Before acting, "
                    "confirm the figures against a second cut of the data.")
        bullets = [
            "Finding: based only on the returned result, not external context.",
            "Caveat: this is an advisory observation, not a directive.",
        ]
        secondary = scope.get("note", "Based on the returned rows.")

    else:
        title = "Analysis"
        body = "This follow-up action is not supported for the current result."

    next_step = ""
    if action == "decide":
        next_step = "Re-run with a narrower filter or a second time window to verify before acting."

    return {
        "type": "assistant_analysis",
        "action": action,
        "title": title,
        "body": body,
        "secondary": secondary,
        "bullets": bullets,
        "next_step": next_step,
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
    query_executor=None,
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
                query_executor=query_executor,
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


def build_assistant_response(
    *,
    question: str,
    rows: list[dict],
    sql: str,
    duration_ms: int,
    chart: dict | None = None,
    data_source: str | None = None,
    confidence: dict | None = None,
    display_context: dict | None = None,
    column_formats: dict | None = None,
    semantic_plan: dict | None = None,
    question_id: str = "",
) -> dict:
    from core.insight import compute_data_brief
    ctx = summarize_result_context(rows, question, sql=sql)
    resolved_column_formats = build_column_formats(
        rows,
        display_context=display_context,
        explicit_formats=column_formats,
    )
    answer = build_answer(
        rows,
        question,
        ctx.get("result_scope"),
        column_formats=resolved_column_formats,
    )
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
    decision_signal  = _build_decision_signal(ctx, brief, anomaly_callouts)

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
        "decision_signal": decision_signal,
        "summary": {"executive_summary": ""},
        "next_actions": compute_chip_eligibility(ctx, brief=brief, semantic_plan=semantic_plan),
        "analysis_contract": ctx,
        "data_brief": brief,
        "result_scope": ctx.get("result_scope", {}),
        "data": {
            "headers": headers,
            "rows": display_rows,
            "total_rows": len(rows),
            "truncated": len(rows) > _PREVIEW_ROW_CAP,
            "column_formats": resolved_column_formats,
            "currency_columns": [
                col for col, fmt in resolved_column_formats.items()
                if fmt == "currency"
            ],
        },
        "trust": {
            "sql": sql,
            "row_count": len(rows),
            "duration_label": f"{duration_ms}ms" if duration_ms < 1000 else f"{duration_ms/1000:.1f}s",
            "data_source": data_source or "",
            "scope_badge": ctx.get("result_scope", {}).get("badge", ""),
            "confidence": confidence or {},
            "question_id": question_id,   # public key for feedback API (B3)
        },
        "confidence": confidence or {},
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
