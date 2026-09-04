from __future__ import annotations

import logging
import json
import math
import re
from datetime import date, datetime
from statistics import mean, median, stdev
from typing import Any

from core.display_formats import normalize_display_format
from core.clarification import extract_original_question
from core.temporal_columns import infer_series_grain, parse_period_label

log = logging.getLogger("querybot.response_builder")

_PREVIEW_ROW_CAP = 200
_RESULT_FORMATS = {"number", "currency", "percentage", "date", "text"}

_TEXT_ONLY_RESPONSE_KEYS = {
    "content", "text", "headline", "short_value", "insight_summary",
    "label", "title", "message", "reason", "description", "next_step",
    "executive_summary", "scope_badge", "duration_label", "data_source",
    "question", "question_id", "sql",
}


def sanitize_response_text_fields(value: Any, *, parent_key: str = "") -> Any:
    """Keep structured objects out of response fields consumed as text.

    Browser coercion of an accidental object produces ``[object Object]``.
    Enforce the response contract at the final payload boundary while leaving
    legitimate structured fields (chart, data, confidence, diagnostics) intact.
    """
    if parent_key in _TEXT_ONLY_RESPONSE_KEYS:
        if isinstance(value, list):
            scalar_items = [
                str(item) for item in value
                if isinstance(item, (str, int, float, bool))
            ]
            return " · ".join(scalar_items)
        if isinstance(value, dict):
            return ""
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            return str(value)
        return ""
    if isinstance(value, dict):
        return {
            key: sanitize_response_text_fields(item, parent_key=str(key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_response_text_fields(item) for item in value]
    return value

_CURRENCY_NAME_RE = re.compile(
    r"\b(revenue|amount|cost|price|total|sales|charge|fee|payment|spend|"
    r"value|income|profit|loss|margin|earning|billing|invoice|budget|"
    r"gross|net|balance|credit|debit|cash|dollar|usd|gbp|eur|salary|"
    r"wage|commission|rebate|discount|tax|surcharge|reimbursement)\b",
    re.IGNORECASE,
)
_PERCENT_NAME_RE = re.compile(r"\b(percent|percentage|pct|rate|ratio|share)\b", re.IGNORECASE)
_DATE_NAME_RE = re.compile(
    r"\b(date|dt|period|prd|yyyymm|yyyymmdd|year|month|quarter|week|day)\b",
    re.IGNORECASE,
)
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


_CURRENCY_SYMBOLS = {
    "USD": "$", "INR": "₹", "EUR": "€", "GBP": "£",
    "CAD": "CA$", "AUD": "A$", "JPY": "¥",
}


def _format_number(
    value: Any,
    fmt: str | None = None,
    display_format: dict | None = None,
) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(num):
        return str(value)
    spec = normalize_display_format(display_format)
    fmt = _normalise_result_format(spec.get("type") or fmt)
    grouping = spec.get("grouping", True)
    group_flag = "," if grouping else ""
    digits = spec.get("fraction_digits")
    if spec.get("style") == "compact" and abs(num) >= 1000:
        for divisor, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
            if abs(num) >= divisor:
                compact_digits = 1 if digits is None else digits
                compact = f"{num / divisor:.{compact_digits}f}".rstrip("0").rstrip(".")
                return f"{compact}{suffix}"
    if fmt == "currency":
        digits = 2 if digits is None else digits
        code = str(spec.get("currency_code") or "USD")
        symbol = _CURRENCY_SYMBOLS.get(code, f"{code} ")
        absolute = f"{abs(num):{group_flag}.{digits}f}"
        if num < 0 and spec.get("accounting"):
            return f"({symbol}{absolute})"
        return f"{'-' if num < 0 else ''}{symbol}{absolute}"
    if fmt == "percentage":
        if spec.get("scale") == "fraction":
            num *= 100
        digits = 2 if digits is None else digits
        rendered = f"{num:{group_flag}.{digits}f}"
        return f"{rendered}%"
    if digits is not None:
        return f"{num:{group_flag}.{digits}f}"
    if abs(num) >= 1000:
        return f"{num:,.0f}" if num.is_integer() else f"{num:,.2f}"
    return f"{num:.0f}" if num.is_integer() else f"{num:.2f}"


def _format_display_value(
    value: Any,
    fmt: str | None = None,
    display_format: dict | None = None,
) -> str:
    spec = normalize_display_format(display_format)
    if spec.get("type") == "date" or _normalise_result_format(fmt) == "date":
        parsed: date | None = None
        if isinstance(value, datetime):
            parsed = value.date()
        elif isinstance(value, date):
            parsed = value
        else:
            text = str(value or "").strip()
            match = re.match(r"^(\d{4})[-/](\d{1,2})(?:[-/](\d{1,2}))?", text)
            if not match:
                match = re.fullmatch(r"(\d{4})(\d{2})(\d{2})?", text)
            if match:
                try:
                    candidate = date(
                        int(match.group(1)),
                        int(match.group(2)),
                        int(match.group(3) or 1),
                    )
                    parsed = candidate if 1900 <= candidate.year <= 2199 else None
                except ValueError:
                    parsed = None
        if parsed:
            style = spec.get("style") or "iso"
            if style == "month_year_short":
                return parsed.strftime("%b-%y")
            if style == "month_year_long":
                return parsed.strftime("%B %Y")
            if style == "day_month_year":
                return parsed.strftime("%d-%m-%Y")
            if style == "month_day_year":
                return parsed.strftime("%m-%d-%Y")
            if style == "day_month_name_year":
                return parsed.strftime("%d-%b-%Y")
            if style == "year":
                return parsed.strftime("%Y")
            if style == "month_name":
                return parsed.strftime("%B")
            return parsed.strftime("%Y-%m")
    return _format_number(value, fmt, spec)


def narrative_period_labels(labels: list) -> list[str]:
    """Format a series of period labels the way the table and KPI show them.

    Periods reach the user through THREE paths, not two: the rendered table,
    the KPI headline, and the sentences written about the series. The first two
    go through the display formatter via column_formats; narration did not, so
    the same answer said "2026-06 closed at $7.4M" in its headline and
    "trended flat from 2026-01-01 to 2026-06-01" three lines below it.

    Same bucket-shape rule as build_column_formats, and for the same reason: a
    month bucket is always the first day of its period, so day == 1 across the
    whole series distinguishes a bucket from a real date. Invoices due on the
    15th and month-end balance dates both step ~30 days and must be left alone.

    Returns the labels unchanged unless every one of them is a month or quarter
    bucket -- narration is prose, and a half-formatted series reads worse than
    an unformatted one.
    """
    raw = [str(label) if label is not None else "" for label in labels]
    if len(raw) < 2:
        return raw
    parsed = [parse_period_label(label) for label in raw]
    if any(value is None or value.day != 1 for value in parsed):
        return raw
    ordered = sorted(set(parsed))
    if len(ordered) < 2:
        return raw
    grain, confidence = infer_series_grain(ordered)
    if confidence < 0.8:
        return raw
    if grain == "month" or (
        grain == "quarter" and all(value.month in (1, 4, 7, 10) for value in ordered)
    ):
        return [value.strftime("%Y-%m") for value in parsed]
    return raw


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
    payload = {
        tok.lower()
        for tok in re.findall(r"[A-Za-z0-9]+", text)
        if tok and tok.lower() not in _FORMAT_STOP_TOKENS
    }
    return payload


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

    # Encoded ERP periods such as 202601/20260131 are semantically dates even
    # when the database type is INT. Apply a display-only date contract when
    # both the column name and sampled values support that interpretation.
    for header in headers:
        if header in formats or not _DATE_NAME_RE.search(header.replace("_", " ")):
            continue
        values = [
            row.get(header) for row in rows[:20]
            if row.get(header) not in (None, "")
        ]
        if values and all(_parse_compact_date_value(value) is not None for value in values):
            formats[header] = "date"
            continue

        # The branch above only ever recognised a COMPACT ERP integer
        # (202601 / 20260131 -- the regex admits no separators), so a genuine
        # DATE column was never marked. It reaches the browser as the ISO
        # string "2026-01-01", nothing declares it a date, and the portal
        # prints it verbatim: a month bucket displayed as the first of the
        # month, which reads as a single day's figure.
        #
        # Marked only at MONTH or QUARTER grain, and this restriction is
        # load-bearing rather than cautious. The portal's date renderer falls
        # through to `YYYY-MM` for any style it does not recognise
        # (portal_chat.html), so declaring a DAILY column a date would collapse
        # every day of a month onto one label and silently merge rows in the
        # reader's eyes. A day-grain column already displays correctly.
        #
        # The test is the BUCKET SHAPE, not the cadence. A governed month or
        # quarter bucket is always the FIRST DAY of its period -- every builder
        # emits DATEFROMPARTS(YEAR(x), MONTH(x), 1), see
        # core.contextual_dates.format_period_bucket_expression -- so day == 1
        # is a shape the server itself created and can recognise.
        #
        # Cadence alone cannot tell a bucket from a real day, and the
        # difference is destructive: invoices due on the 15th of each month,
        # and month-END balance dates, both step ~30 days, and relabelling
        # either one "2026-01" erases the exact thing the reader needs. Both
        # were declared dates by a cadence test.
        #
        # Checked over EVERY rendered row rather than a sample, because the
        # format is applied to every row: a page of month buckets followed by
        # a daily tail would otherwise collapse the tail onto shared labels
        # and silently merge rows on screen.
        column_values = [row.get(header) for row in rows if row.get(header) not in (None, "")]
        if len(column_values) != len(rows):
            continue
        parsed = [parse_period_label(value) for value in column_values]
        if any(value is None or value.day != 1 for value in parsed):
            continue
        # Sorted and de-duplicated so a DESC result and a breakdown that
        # repeats each period per category both read as one clean series.
        ordered = sorted(set(parsed))
        if len(ordered) < 2:
            continue
        grain, confidence = infer_series_grain(ordered)
        if confidence < 0.8:
            continue
        # Year grain is excluded deliberately: the shared date renderer has no
        # style that prints a year as a year by default, so 2026-01-01 would
        # display as "2026-01" -- no better than the day stamp it replaced.
        if grain == "month" or (
            grain == "quarter" and all(value.month in (1, 4, 7, 10) for value in ordered)
        ):
            formats[header] = "date"

    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        fmt = _normalise_result_format(metric.get("result_format"))
        if fmt == "number" and not strict:
            continue
        for header in _columns_for_metric_format(rows, metric, strict=strict):
            formats.setdefault(header, fmt)

    return formats


def _parse_compact_date_value(value: Any) -> date | None:
    text = str(value or "").strip()
    match = re.fullmatch(r"(\d{4})(\d{2})(\d{2})?", text)
    if not match:
        return None
    try:
        parsed = date(
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3) or 1),
        )
    except ValueError:
        return None
    return parsed if 1900 <= parsed.year <= 2199 else None


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


def _temporal_sort_value(value: Any) -> tuple[int, int, int, int] | None:
    """Return a sortable date/period key without changing displayed values."""
    if isinstance(value, datetime):
        return value.year, value.month, value.day, 0
    if isinstance(value, date):
        return value.year, value.month, value.day, 0
    text = str(value or "").strip()
    if not text:
        return None
    match = re.match(r"^((?:19|20)\d{2})[-/]?(\d{2})(?:[-/]?(\d{2}))?$", text)
    if match:
        year, month, day = (int(part or 1) for part in match.groups())
        if 1 <= month <= 12 and 1 <= day <= 31:
            return year, month, day, 0
    match = re.match(r"^(?:Q([1-4])\s*[-/]?\s*((?:19|20)\d{2})|((?:19|20)\d{2})\s*[-/]?\s*Q([1-4]))$", text, re.I)
    if match:
        quarter = int(match.group(1) or match.group(4))
        year = int(match.group(2) or match.group(3))
        return year, ((quarter - 1) * 3) + 1, 1, quarter
    for fmt in ("%B %Y", "%b %Y", "%b-%y", "%B-%y", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.year, parsed.month, parsed.day, 0
        except ValueError:
            continue
    return None


def _chronological_analysis_rows(rows: list[dict]) -> list[dict]:
    """Sort a copy for temporal analysis while preserving table display order."""
    copied = list(rows)
    if len(copied) < 2:
        return copied
    numeric_cols = _numeric_cols(copied)
    for column in _text_cols(copied, numeric_cols):
        values = [str(row.get(column, "")) for row in copied]
        if not _looks_temporal(values):
            continue
        keys = [_temporal_sort_value(row.get(column)) for row in copied]
        if all(key is not None for key in keys):
            return [
                row
                for _key, _index, row in sorted(
                    zip(keys, range(len(copied)), copied),
                    key=lambda item: (item[0], item[1]),
                )
            ]
    return copied


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


_MATCHED_ROWS_HEADER_KEYS = {"matchedrows", "rowcount", "matchcount", "matchedrecords"}


def _find_matched_rows_header(headers: list[str]) -> str:
    """The header name of a diagnostic match-count column (e.g. MatchedRows),
    if this row shape carries one. Shared by detect_null_metric_issue and
    detect_zero_match_result -- the two checks are mutually exclusive by
    construction (matched_rows > 0 vs <= 0), never both true for the same row."""
    return next(
        (h for h in headers if _normalise_key(h) in _MATCHED_ROWS_HEADER_KEYS),
        "",
    )


def result_diagnostic_headers(rows: list[dict]) -> list[str]:
    """Return support columns used to validate aggregate result quality.

    These values remain available to answer/confidence logic but are not
    business measures and therefore should not be rendered as table columns
    or KPI values.
    """
    if len(rows) != 1 or not rows[0]:
        return []
    output: list[str] = []
    for header in rows[0].keys():
        norm = _normalise_key(header)
        if norm in _MATCHED_ROWS_HEADER_KEYS or (
            norm.startswith("nonnull") and norm.endswith("rows")
        ):
            output.append(header)
    return output


def visible_result_rows(rows: list[dict]) -> list[dict]:
    hidden = set(result_diagnostic_headers(rows))
    if not hidden:
        return list(rows)
    return [
        {header: value for header, value in row.items() if header not in hidden}
        for row in rows
    ]


def _result_diagnostics(rows: list[dict]) -> dict[str, Any]:
    headers = result_diagnostic_headers(rows)
    if not headers:
        return {}
    row = rows[0]
    matched = _find_matched_rows_header(headers)
    return {
        "matched_rows": int(_to_float(row.get(matched)) or 0) if matched else None,
        "non_null_counts": {
            header: int(_to_float(row.get(header)) or 0)
            for header in headers
            if _normalise_key(header).startswith("nonnull")
        },
        "hidden_columns": headers,
    }


def _build_kpi_payload(
    rows: list[dict],
    column_formats: dict[str, str],
    display_formats: dict[str, dict],
    *,
    zero_match: bool,
    null_issue: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if zero_match or len(rows) != 1 or len(rows[0]) != 1:
        return None
    column = next(iter(rows[0]))
    value = rows[0].get(column)
    scalar_missing = _is_missing_scalar(value)
    return {
        "label": _display_label(column),
        "value": _safe_cell(value),
        "format": column_formats.get(column, "number"),
        "display_format": dict(display_formats.get(column) or {}),
        "state": "missing" if (null_issue or scalar_missing) else "ready",
        "note": (
            "Matching records were found, but this metric has no non-null values."
            if null_issue
            else "No data was returned for the requested period or filters."
            if scalar_missing
            else "Single-value result"
        ),
    }


_COMPARISON_PREFIXES = (
    ("CURRENT_", "PREVIOUS_"),
    ("CURRENT_", "PRIOR_"),
    ("THIS_", "LAST_"),
)
_PCT_CHANGE_COLUMNS = ("PCT_CHANGE", "PERCENT_CHANGE", "PCT_DIFF", "CHANGE_PCT")


def _numeric_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return None


def _is_missing_scalar(value: Any) -> bool:
    """Return True when a scalar database result carries no usable value."""
    if value is None:
        return True
    if isinstance(value, float) and not math.isfinite(value):
        return True
    return False


def _single_missing_scalar(rows: list[dict]) -> tuple[str, Any] | None:
    if len(rows) != 1 or len(rows[0]) != 1:
        return None
    column = next(iter(rows[0]))
    value = rows[0].get(column)
    return (column, value) if _is_missing_scalar(value) else None


def _missing_scalar_copy(column: str, question: str) -> dict[str, str]:
    """Build calm, business-facing copy for successful NULL aggregates.

    SQL aggregates such as SUM() return one physical row containing NULL when
    the requested period has no matching facts.  That is an empty analytical
    result, not a value called ``None`` and not a query failure.
    """
    metric = _display_label(column)
    metric_lower = metric[:1].lower() + metric[1:] if metric else "metric"
    temporal = bool(re.search(
        r"\b(today|yesterday|tomorrow|day|week|month|quarter|q[1-4]|year|"
        r"fiscal|calendar|period|date|latest|last|current|previous|prior)\b",
        str(question or ""),
        re.IGNORECASE,
    ))
    target = "the requested period" if temporal else "the current filters"
    return {
        "headline": f"No {metric_lower} data was found for {target}.",
        "short_value": "No data",
        "comparison": "The query completed successfully, but no metric value was returned.",
        "scope_badge": "No data",
        "scope_note": f"There are no matching {metric_lower} values for {target}.",
    }


def _period_comparison_from_rows(rows: list[dict]) -> dict | None:
    """Recognise a single-row current-vs-previous comparison.

    The SQL for a period comparison returns ONE wide row of paired columns
    (CURRENT_x / PREVIOUS_x, plus a difference and percentage), not a series.
    Narrating it as a trend produced "trended flat 0.0% from 2026-03 to
    2026-03" over data that actually read 2026-03 $500 vs 2026-02 $400, +25%.

    Column-naming convention only -- no tenant vocabulary. Requires a numeric
    pair to call it a comparison; a non-numeric pair (e.g. CURRENT_MONTH /
    PREVIOUS_MONTH) supplies the period labels instead.
    """
    if not rows or len(rows) != 1:
        return None
    row = rows[0]
    if not isinstance(row, dict):
        return None
    upper = {str(k).upper(): k for k in row}

    numeric_pair = None
    label_pair = None
    for cur_prefix, prev_prefix in _COMPARISON_PREFIXES:
        for key_u, key in upper.items():
            if not key_u.startswith(cur_prefix):
                continue
            suffix = key_u[len(cur_prefix):]
            prev_u = f"{prev_prefix}{suffix}"
            if prev_u not in upper:
                continue
            prev_key = upper[prev_u]
            cur_val = _numeric_or_none(row.get(key))
            prev_val = _numeric_or_none(row.get(prev_key))
            is_period_label = bool(re.search(
                r"(?:^|_)(?:DATE|DT|DAY|WEEK|MONTH|QUARTER|YEAR|PERIOD|PRD|YYYYMM|YYYYMMDD)(?:_|$)",
                suffix,
            ))
            if is_period_label and label_pair is None:
                label_pair = (row.get(key), row.get(prev_key))
            elif cur_val is not None and prev_val is not None:
                if numeric_pair is None:
                    numeric_pair = (key, prev_key, cur_val, prev_val)
            elif label_pair is None:
                label_pair = (row.get(key), row.get(prev_key))

    if not numeric_pair:
        return None
    measure_col, previous_col, current_value, previous_value = numeric_pair

    pct = None
    for candidate in _PCT_CHANGE_COLUMNS:
        if candidate in upper:
            pct = _numeric_or_none(row.get(upper[candidate]))
            if pct is not None:
                break
    if pct is None and previous_value:
        pct = (current_value - previous_value) * 100.0 / previous_value

    current_period, previous_period = ("the current period", "the previous period")
    if label_pair and label_pair[0] is not None and label_pair[1] is not None:
        current_period, previous_period = str(label_pair[0]), str(label_pair[1])

    return {
        "measure_column": measure_col,
        "previous_column": previous_col,
        "current_value": row.get(measure_col),
        "previous_value": row.get(previous_col),
        "current_period": current_period,
        "previous_period": previous_period,
        "pct_change": pct,
    }


def _plural(phrase: str) -> str:
    """"revenue category" -> "revenue categories". Enough English for a count
    sentence; a phrase that already reads as plural is left alone."""
    words = str(phrase or "").split()
    if not words or words[-1].endswith("s"):
        return phrase
    last = words[-1]
    if len(last) > 1 and last.endswith("y") and last[-2] not in "aeiou":
        last = last[:-1] + "ies"
    elif last.endswith(("x", "z", "ch", "sh")):
        last += "es"
    else:
        last += "s"
    return " ".join(words[:-1] + [last])


def _safe_category_label(label: Any, label_column: str) -> str:
    """The category's own name, or "" when the value redactor replaced it.

    core.insight._display_label is imported under an alias here and in every
    other caller: this module already defines a one-argument _display_label
    column prettifier used in five places, and an unaliased import would shadow
    it and raise TypeError on a path with no protection around it.
    """
    from core.insight import _display_label as _redact_value_label

    text = _redact_value_label(str(label or ""), label_column)
    return "" if not text or text == "redacted segment" else text


def _measure_prefix(column: str, label: str) -> str:
    """"NET_AMOUNT_2025" with the label "2025" -> "NET_AMOUNT".

    Empty when the column IS the period label, which is what makes the caller
    say "Total" rather than name a measure it cannot see.
    """
    from core.multi_period import period_alias_suffix

    suffix = period_alias_suffix(label)
    name = str(column or "")
    if suffix and name.upper().endswith("_" + suffix):
        return name[: -(len(suffix) + 1)]
    return "" if name.upper() == suffix else name


def _period_pair_facts(rows: list[dict], plan_labels: list[str] | None) -> dict | None:
    """The arithmetic behind every period-comparison sentence, or None.

    None means "this is not a named-period comparison" and every caller falls
    straight through to the behaviour it had before. The gate is strict on
    purpose: at least two of the plan's OWN period aliases must be present as
    result columns, and every label must parse as a real calendar period.

    Column matching is delegated to core/multi_period.py rather than repeated
    here. The hint, the post-processor and this all have to agree on which
    columns are the periods; two matchers would drift, and the loose one would
    start reading ERP columns like P_QTY as a period.
    """
    if not rows or not plan_labels or len(plan_labels) < 2:
        return None
    try:
        from core.multi_period import (
            PeriodPlan, period_columns_for_plan, period_alias_suffix, period_parts,
        )
    except Exception:
        return None

    labels = [str(label) for label in plan_labels]
    if not all(period_parts(label) for label in labels):
        return None
    aliases = [period_alias_suffix(label) for label in labels]
    found = period_columns_for_plan(
        rows,
        PeriodPlan(labels=labels, aliases=aliases, predicates=[""] * len(labels),
                   grain="", date_field={}),
    )
    if len(found) < 2:
        return None

    present = [(label, found[alias]) for label, alias in zip(labels, aliases)
               if alias in found]
    (oldest_label, oldest_col), (newest_label, newest_col) = present[0], present[-1]

    numeric_cols = _numeric_cols(rows)
    text_cols = _text_cols(rows, numeric_cols)
    label_col = text_cols[0] if text_cols else ""

    movers: list[dict] = []
    oldest_total = newest_total = 0.0
    for row in rows:
        before, after = _to_float(row.get(oldest_col)), _to_float(row.get(newest_col))
        if before is None or after is None:
            continue          # masked or missing; excluded from every total
        oldest_total += before
        newest_total += after
        movers.append({
            "label": str(row.get(label_col, "")) if label_col else "",
            "change": after - before,
            "pct": ((after - before) * 100.0 / abs(before)) if before else None,
        })
    if not movers:
        return None

    net = sum(mover["change"] for mover in movers)
    gross = sum(abs(mover["change"]) for mover in movers)
    grew = sum(1 for mover in movers if mover["change"] > 0)
    shrank = sum(1 for mover in movers if mover["change"] < 0)
    risers = [mover for mover in movers if mover["change"] > 0]
    fallers = [mover for mover in movers if mover["change"] < 0]
    top_riser = max(risers, key=lambda m: m["change"]) if risers else None
    top_faller = min(fallers, key=lambda m: m["change"]) if fallers else None

    return {
        "labels": [label for label, _ in present],
        "oldest_label": oldest_label, "newest_label": newest_label,
        "oldest_column": oldest_col, "newest_column": newest_col,
        "label_column": label_col,
        "oldest_total": oldest_total, "newest_total": newest_total,
        "total_pct": (net * 100.0 / abs(oldest_total)) if oldest_total else None,
        "net": net,
        # Below this the gains and losses have cancelled and a share of the net
        # is not a number worth printing -- the same floor annotate_period_change
        # applies to SHARE_OF_CHANGE_PCT.
        "share_holds": gross > 0 and abs(net) >= 0.05 * gross,
        "row_count": len(movers), "grew": grew, "shrank": shrank,
        "flat": len(movers) - grew - shrank,
        "top_riser": top_riser, "top_faller": top_faller,
    }


def _period_comparison_summary(
    rows: list[dict],
    plan_labels: list[str] | None,
    column_formats: dict | None = None,
    display_formats: dict | None = None,
) -> str:
    """The note under the card for a named-period comparison, or "".

    "Across 12 revenue categories, 7 grew and 5 shrank between 2024 and 2025.
    Pumps added the most (+400,000, 46% of the total increase); Valves fell the
    most (-180,000)."

    Category labels go through core.insight's value redactor, imported under an
    alias: this module already defines a one-argument _display_label column
    prettifier used in five places, and an unaliased import would shadow it and
    raise TypeError on an unprotected path.
    """
    facts = _period_pair_facts(rows, plan_labels)
    if not facts:
        return ""
    column_formats = column_formats or {}
    display_formats = display_formats or {}

    def money(value: float) -> str:
        return _format_display_value(
            value,
            column_formats.get(facts["newest_column"]),
            display_formats.get(facts["newest_column"]),
        )

    def signed(value: float) -> str:
        return ("+" if value > 0 else "") + money(value)

    def named(mover: dict) -> str:
        return _safe_category_label(mover["label"], facts["label_column"])

    counted = (_plural(_display_label(facts["label_column"]).lower())
               if facts["label_column"] else "groups")
    opening = (
        f"Across {facts['row_count']} {counted}, {facts['grew']} grew and "
        f"{facts['shrank']} shrank between {facts['oldest_label']} and "
        f"{facts['newest_label']}."
    )

    clauses: list[str] = []
    riser, faller = facts["top_riser"], facts["top_faller"]
    if riser:
        share = ""
        if facts["share_holds"] and facts["net"]:
            share = f", {abs(riser['change'] * 100.0 / facts['net']):.0f}% of the net change"
        who = named(riser)
        clauses.append(
            f"{who} added the most ({signed(riser['change'])}{share})" if who
            else f"the largest increase was {signed(riser['change'])}{share}"
        )
    if faller:
        who = named(faller)
        clauses.append(
            f"{who} fell the most ({signed(faller['change'])})" if who
            else f"the largest decrease was {signed(faller['change'])}"
        )
    if not clauses:
        return f"{opening} No category moved between the two periods."
    detail = "; ".join(clauses)
    return f"{opening} {detail[0].upper()}{detail[1:]}."


def detect_zero_match_result(rows: list[dict]) -> bool:
    """
    True for a single-row diagnostic aggregate whose match-count column is
    itself zero (or negative) -- e.g. [{"MatchedRows": 0, "Revenue": 0}].

    A query like `SELECT COUNT(*) AS MatchedRows, SUM(x) AS Total FROM ...
    WHERE <date filter>` always returns exactly one physical row even when
    nothing matched, so the ordinary "not rows" empty-result check never
    fires -- the answer layer would otherwise present the zero as if it
    were a real, successful single-value answer ("Returned 1 rows").

    Deliberately narrower than "all numeric columns are zero/null": that
    would misfire on a legitimately-zero real answer (e.g. actual $0 profit
    this month). Only fires when the row carries one of the same explicit
    match-count column names detect_null_metric_issue already trusts.
    """
    if len(rows) != 1 or not rows[0]:
        return False
    row = rows[0]
    matched_header = _find_matched_rows_header(list(row.keys()))
    if not matched_header:
        return False
    matched_rows = _to_float(row.get(matched_header))
    return matched_rows is not None and matched_rows <= 0


def detect_null_metric_issue(rows: list[dict]) -> dict[str, Any] | None:
    """
    Detect diagnostic rows where records matched, but a requested metric was
    NULL/missing for every matched record.
    """
    if len(rows) != 1 or not rows[0]:
        return None
    row = rows[0]
    headers = list(row.keys())
    matched_header = _find_matched_rows_header(headers)
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
    display_formats: dict | None = None,
    period_labels: list[str] | None = None,
) -> dict:
    scope = result_scope or infer_result_scope(rows, question)
    column_formats = column_formats or {}
    display_formats = display_formats or {}

    def format_value(value: Any, column: str) -> str:
        return _format_display_value(
            value,
            column_formats.get(column),
            display_formats.get(column),
        )
    if not rows or detect_zero_match_result(rows):
        return {
            "headline": "No matching data was found for this question.",
            "short_value": "0 rows",
            "comparison": "Try adjusting the filters or time range.",
            "scope_badge": scope.get("badge", ""),
            "scope_note": scope.get("note", ""),
        }

    missing_scalar = _single_missing_scalar(rows)
    if missing_scalar:
        return _missing_scalar_copy(missing_scalar[0], question)

    null_issue = detect_null_metric_issue(rows)
    if null_issue:
        issue = null_issue["issues"][0]
        metric_col = issue["metric_column"]
        fmt = column_formats.get(metric_col)
        value = format_value(_to_float(rows[0].get(metric_col)) or 0, metric_col)
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
            "headline": f"{col.replace('_', ' ').title()}: {format_value(val, col)}",
            "short_value": format_value(val, col),
            "comparison": scope.get("badge") or "Single-value result",
            "scope_badge": scope.get("badge", ""),
            "scope_note": scope.get("note", ""),
        }

    if numeric_cols and text_cols:
        # A named-period comparison, before the ranking path. Fixing the SQL
        # alone does not fix the answer: build_answer took numeric_cols[0] --
        # the OLDEST period column on a widened result -- and opened the card
        # with "Pumps leads at 3,800,000." and a cross-category gap chip that
        # reads exactly like a year-over-year delta. The target question is not
        # causal, so no narration runs and these deterministic sentences ARE
        # the answer the reader gets.
        _pair = _period_pair_facts(rows, period_labels)
        if _pair:
            newest_col = _pair["newest_column"]
            measure = _display_label(
                _measure_prefix(newest_col, _pair["newest_label"])) or "Total"
            pct = _pair["total_pct"]
            direction = ("rose" if _pair["net"] > 0
                         else "fell" if _pair["net"] < 0 else "was flat")
            movement = (f"{direction} {abs(pct):.1f}%" if pct is not None
                        else direction)
            headline = (f"{measure} {movement} from {_pair['oldest_label']} "
                        f"to {_pair['newest_label']}")
            riser = _pair["top_riser"] or _pair["top_faller"]
            if riser:
                mover = _safe_category_label(riser["label"], _pair["label_column"])
                change = ("+" if riser["change"] > 0 else "") + format_value(
                    riser["change"], newest_col)
                headline += (f"; {mover} moved the most, {change}" if mover
                             else f"; the largest single move was {change}")
            comparison = (f"{pct:+.1f}% versus {_pair['oldest_label']}"
                          if pct is not None
                          else f"compared with {_pair['oldest_label']}")
            return {
                "headline": headline + ".",
                "short_value": format_value(_pair["newest_total"], newest_col),
                "comparison": comparison,
                "scope_badge": scope.get("badge", ""),
                "scope_note": scope.get("note", ""),
            }

        label_col = text_cols[0]
        value_col = numeric_cols[0]
        value_fmt = column_formats.get(value_col)
        ordered = sorted(rows, key=lambda r: _to_float_z(r.get(value_col)), reverse=True)
        labels = [str(r.get(label_col, "")) for r in rows]
        if _looks_temporal(labels) and scope.get("kind") != "ranking":
            first = rows[0]
            last = rows[-1]
            first_val = _to_float_z(first.get(value_col))
            last_val = _to_float_z(last.get(value_col))
            direction = "up" if last_val > first_val else "down" if last_val < first_val else "flat"
            last_label = format_value(last.get(label_col, 'Latest period'), label_col)
            headline = f"{last_label} closed at {format_value(last_val, value_col)}."
            comparison = scope.get("badge") or f"Trend is {direction} versus {format_value(first_val, value_col)} at the start"
            return {
                "headline": headline,
                "short_value": format_value(last_val, value_col),
                "comparison": comparison,
                "scope_badge": scope.get("badge", ""),
                "scope_note": scope.get("note", ""),
            }
        best = ordered[0]
        best_label = str(best.get(label_col, 'Top result'))
        best_value = _to_float_z(best.get(value_col))
        comparison = scope.get("badge") or f"Across {len(rows)} results"
        if scope.get("is_top_n") and (scope.get("n") or 0) == 1:
            headline = f"Top-ranked result: {best_label} at {format_value(best_value, value_col)}."
            comparison = "This card shows only the leading row"
        else:
            headline = f"{best_label} leads at {format_value(best_value, value_col)}."
        if len(ordered) > 1 and not scope.get("is_top_n"):
            second = ordered[1]
            second_value = _to_float_z(second.get(value_col))
            delta = best_value - second_value
            comparison = f"{format_value(delta, value_col)} above the next result"
        return {
            "headline": headline,
            "short_value": format_value(best_value, value_col),
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
            "short_value": format_value(values[0], col),
            "comparison": scope.get("badge") or f"Range {format_value(min(values), col)} to {format_value(max(values), col)}",
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
        raw_val = rows[0][col]
        # Every other branch below normalizes DB values through _to_float/
        # _to_float_z before putting them in ctx; this one didn't, so a raw
        # decimal.Decimal (returned by pyodbc/Azure SQL for SUM() on a
        # numeric/decimal column) rode straight into analysis_contract and
        # broke ws.send_json's JSON encoder — the query succeeded but the
        # user got total silence with no error surfaced anywhere.
        safe_val = _to_float(raw_val)
        if safe_val is None:
            safe_val = "" if raw_val is None else str(raw_val)
        ctx.update({
            "mode": "single_value",
            "value_column": col,
            "value": safe_val,
            "chartable": False,
        })
        ctx["result_scope"] = infer_result_scope(rows, question, sql, mode="single_value")
        return ctx

    if numeric_cols and text_cols:
        label_col = text_cols[0]
        value_col = numeric_cols[0]
        # Formatted here, at the one place the series is read, so every
        # sentence written about it downstream inherits the same labels the
        # table and the KPI show.
        labels = narrative_period_labels([r.get(label_col, "") for r in rows])
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
    *,
    sql: str = "",
    db_type: str = "",
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
            if sql:
                from core.drill_dimension import build_deterministic_drill_sql
                if not build_deterministic_drill_sql(sql, dim, db_type or "azure_sql"):
                    continue
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

def _build_insight_summary(
    rows: list[dict],
    ctx: dict,
    brief: dict,
    column_formats: dict | None = None,
    display_formats: dict | None = None,
) -> str:
    """
    Generate a one-sentence plain-English summary from the data brief.

    Purely stat-driven — no LLM call, no latency added.
    Returns empty string when there is not enough structure to say anything useful.
    """
    mode = ctx.get("mode", "table")
    row_count = len(rows)
    column_formats = column_formats or {}
    display_formats = display_formats or {}

    def format_value(value: Any, column: str = "") -> str:
        return _format_display_value(
            value,
            column_formats.get(column),
            display_formats.get(column),
        )

    if detect_zero_match_result(rows):
        return "No matching data was found for this question."

    missing_scalar = _single_missing_scalar(rows)
    if missing_scalar:
        return _missing_scalar_copy(missing_scalar[0], ctx.get("question", ""))["headline"]

    null_issue = detect_null_metric_issue(rows)
    if null_issue:
        issue = null_issue["issues"][0]
        metric = _display_label(issue["metric_column"])
        return (
            f"{null_issue['matched_rows']} records matched, but {metric} is missing "
            "for every matched row."
        )

    if mode == "single_value":
        raw_col = brief.get("value_column") or ""
        col = raw_col.replace("_", " ").title()
        val = brief.get("value", "")
        return f"{col}: {format_value(val, raw_col)}." if col else ""

    # A comparison of periods the USER named, which arrives as one row per
    # category with a column per period. Deliberately ahead of the single-wide-
    # row check below: that one recognises CURRENT_x/PREVIOUS_x pairs from the
    # compare_prior chip and has no idea what 2024 and 2025 are.
    _period_note = _period_comparison_summary(
        rows, ctx.get("period_labels"), column_formats, display_formats)
    if _period_note:
        return _period_note

    # A period comparison arrives as ONE wide row (current/previous pairs), not
    # a series. Classified as time_series it narrated "trended flat 0.0% from
    # 2026-03 to 2026-03" -- first and last period of a single-row series are
    # the same cell -- while the table correctly showed 2026-03 $500.00 against
    # 2026-02 $400.00, +25%. Correct data with a contradicting summary is worse
    # than an error: the reader gets no signal to distrust it.
    _cmp = _period_comparison_from_rows(rows)
    if _cmp:
        measure = _display_label(_cmp["measure_column"])
        cur = format_value(_cmp["current_value"], _cmp["measure_column"])
        prev = format_value(_cmp["previous_value"], _cmp["previous_column"])
        sentence = (
            f"{measure} was {cur} in {_cmp['current_period']} "
            f"versus {prev} in {_cmp['previous_period']}"
        )
        pct = _cmp.get("pct_change")
        if pct is None:
            return sentence + "."
        if pct > 0:
            return sentence + f" - up {abs(pct):.1f}%."
        if pct < 0:
            return sentence + f" - down {abs(pct):.1f}%."
        return sentence + " - unchanged."

    if mode == "time_series":
        ts = brief.get("time_series") or {}
        observation_count = int(
            ts.get("observation_count")
            or brief.get("row_count")
            or ctx.get("row_count")
            or 0
        )
        # Two endpoints support a comparison, not a trend claim. Avoid
        # presenting one interval as sustained momentum or decline.
        if observation_count == 1:
            raw_value_col = ctx.get("value_col") or ""
            value_col = raw_value_col.replace("_", " ").title() or "Value"
            return (
                f"{value_col} was "
                f"{format_value(ts.get('first_value'), raw_value_col)} "
                f"in {ts.get('first_period', 'the returned period')}."
            )
        if observation_count == 2:
            raw_value_col = ctx.get("value_col") or ""
            value_col = raw_value_col.replace("_", " ").title() or "Value"
            first_value = ts.get("first_value")
            last_value = ts.get("last_value")
            sentence = (
                f"{value_col} changed from {format_value(first_value, raw_value_col)} "
                f"in {ts.get('first_period', 'the first period')} to "
                f"{format_value(last_value, raw_value_col)} "
                f"in {ts.get('last_period', 'the second period')}"
            )
            pct = ts.get("overall_pct_change")
            if pct is None:
                return sentence + "."
            direction = "up" if pct > 0 else "down" if pct < 0 else "unchanged"
            if direction == "unchanged":
                return sentence + " - unchanged."
            return sentence + f" - {direction} {abs(pct):.1f}%."
        direction = ts.get("direction", "stable")
        pct = ts.get("overall_pct_change")
        first = ts.get("first_period", "")
        last_ = ts.get("last_period", "")
        raw_value_col = ctx.get("value_col") or ""
        value_col = raw_value_col.replace("_", " ").title()
        dir_word = {"increasing": "up", "decreasing": "down", "stable": "flat"}.get(direction, direction)
        if pct is not None:
            base = f"{value_col} trended {dir_word} {abs(pct):.1f}% from {first} to {last_}."
        else:
            base = f"{value_col} remained {dir_word} between {first} and {last_}."
        peak = ts.get("peak") or {}
        if peak and direction in ("increasing", "decreasing"):
            base += f" Peak: {format_value(peak.get('value', 0), raw_value_col)} in {peak.get('period', '')}."
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
                f"{leader['label']} leads at {format_value(leader['value'], ctx.get('value_col') or '')}{share_str}"
                f" across {count} {label_col or 'entries'}."
            )

    if mode == "numeric_table":
        value_col = (ctx.get("value_col") or "").replace("_", " ").title()
        mn = ctx.get("min_value", 0)
        mx = ctx.get("max_value", 0)
        avg = ctx.get("avg_value", 0)
        return (
            f"{row_count} records — {value_col} ranges "
            f"{format_value(mn, ctx.get('value_col') or '')} to "
            f"{format_value(mx, ctx.get('value_col') or '')}, avg "
            f"{format_value(avg, ctx.get('value_col') or '')}."
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
        if int(ts.get("period_count") or brief.get("row_count") or 0) < 3:
            # One point has no interval; two points have one comparison. Do
            # not label that single interval an anomaly or a sustained move.
            return []
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

    # A named-period comparison is ranked by MOVEMENT, and every sentence below
    # is about a level: "X alone holds 62% of the total" computed over the
    # oldest period's column would read as a claim about today.
    if ctx.get("period_labels"):
        return {}

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
        period_count = int(ts.get("period_count") or brief.get("row_count") or 0)
        if period_count < 3:
            return {}
        direction = ts.get("direction", "stable")
        pct = ts.get("overall_pct_change")
        streak = ts.get("longest_decline_streak", 0)
        if direction == "decreasing" and streak >= 3:
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


def _regulated_analysis_fallback(action: str) -> dict:
    """Static, non-LLM response for regulated tenants — see
    core.compliance.policy_engine.result_llm_features_allowed."""
    return {
        "type": "assistant_analysis",
        "action": action,
        "title": "Not available for this workspace",
        "body": (
            "This workspace is configured for a regulated industry. To keep "
            "protected data from ever reaching the AI model, the assistant "
            "only writes SQL queries here — it doesn't generate follow-up "
            "analysis, explanations, or comparisons from results."
        ),
        "bullets": [],
    }


async def generate_analysis_response(
    action: str,
    rows: list[dict],
    question: str,
    provider: str,
    model: str,
    api_key: str,
    account_id: str,
    follow_up: str = "",
    original_sql: str = "",
    db_cfg: dict | None = None,
    context: str = "",
    known_tables: set[str] | None = None,
    query_executor=None,
    # Explicit, NOT via extra_kwargs: generate_drilldown_insight forwards
    # **extra_kwargs straight into llm_complete, so an unexpected key there
    # raises TypeError rather than being ignored.
    grounding: dict | None = None,
    **extra_kwargs,
) -> dict:
    """
    Async LLM-powered analysis — the preferred path for action buttons
    and "why" follow-up questions.

    Falls back to the synchronous build_analysis_response() if the LLM
    call fails. Regulated tenants get the static _regulated_analysis_fallback
    unconditionally instead — the LLM never sees `rows` for them.
    """
    from core.compliance.policy_engine import result_llm_features_allowed
    if not result_llm_features_allowed(account_id):
        from core.llm_audit import record_llm_blocked
        record_llm_blocked(
            "analysis",
            f"action={action!r} blocked — regulated tenant, LLM never received result rows.",
        )
        return _regulated_analysis_fallback(action)

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
            grounding=grounding,
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
    display_formats: dict | None = None,
    semantic_plan: dict | None = None,
    question_id: str = "",
) -> dict:
    from core.insight import compute_data_brief
    from core.clarification import extract_display_question
    display_question = extract_display_question(question).strip() or question
    display_chart = dict(chart) if isinstance(chart, dict) else chart
    if isinstance(display_chart, dict):
        if "title" in display_chart:
            display_chart["title"] = display_question
        if "question" in display_chart:
            display_chart["question"] = display_question
    raw_rows = list(rows)
    zero_match = detect_zero_match_result(raw_rows)
    null_issue = detect_null_metric_issue(raw_rows)
    visible_rows = visible_result_rows(raw_rows)
    analysis_rows = _chronological_analysis_rows(visible_rows)
    ctx = summarize_result_context(analysis_rows, display_question, sql=sql)
    # The periods the user named, published by the pipeline once the widened
    # result actually arrived. Empty for every other answer in the product, and
    # every branch that reads it falls through to its previous behaviour.
    _period_labels = [
        str(label) for label in
        (((display_context or {}).get("period_comparison") or {}).get("labels") or [])
    ]
    if _period_labels:
        ctx["period_labels"] = _period_labels
    result_operation = str((display_context or {}).get("result_operation") or "")
    if (result_operation in {"keep_top", "sort", "contribution"}
            and ctx.get("mode") == "time_series" and not _period_labels):
        # Period labels sorted by a measure are a ranking, not a chronology.
        # Treating them as a series creates false trend claims from sort order.
        # Not applied to a named-period comparison: its periods are COLUMNS, so
        # the downgrade would relabel a change result as a leaderboard.
        ctx["mode"] = "ranking"
        ctx["result_scope"] = infer_result_scope(visible_rows, display_question, sql, mode="ranking")
    resolved_column_formats = build_column_formats(
        visible_rows,
        display_context=display_context,
        explicit_formats=column_formats,
    )
    headers: list[str] = list(visible_rows[0].keys()) if visible_rows else []
    resolved_display_formats = {
        header: dict(spec)
        for header, spec in (display_formats or {}).items()
        if header in headers and isinstance(spec, dict)
    }
    answer_rows = raw_rows if (zero_match or null_issue) else analysis_rows
    answer = build_answer(
        answer_rows,
        display_question,
        ctx.get("result_scope"),
        column_formats=resolved_column_formats,
        display_formats=resolved_display_formats,
        period_labels=_period_labels,
    )
    brief = compute_data_brief(
        analysis_rows,
        display_question,
        result_scope=ctx.get("result_scope"),
        context=ctx,
    )

    # ── Insight Layer — pure-stats, zero-latency ─────────────────────────────
    # Generate a summary sentence and anomaly callouts from the data brief.
    # These are computed entirely from statistics — no LLM call, no extra latency.
    insight_summary = _build_insight_summary(
        raw_rows if (zero_match or null_issue) else analysis_rows,
        ctx,
        brief,
        resolved_column_formats,
        resolved_display_formats,
    )
    anomaly_callouts = _build_anomaly_callouts(brief)
    decision_signal  = _build_decision_signal(ctx, brief, anomaly_callouts)

    # Include the actual row data (bounded) so the frontend can render a table.
    # Frontend is the ONLY consumer of raw rows — LLM insight path never sees these.
    # We cap at 200 rows to keep WebSocket payload reasonable; full set is already
    # limited by run_query(max_rows=200).
    display_rows: list[dict] = []
    if visible_rows:
        # Send formatted string values for reliable frontend display
        for r in visible_rows[:_PREVIEW_ROW_CAP]:
            display_rows.append({h: _safe_cell(r.get(h)) for h in headers})
    kpi = _build_kpi_payload(
        visible_rows,
        resolved_column_formats,
        resolved_display_formats,
        zero_match=zero_match,
        null_issue=null_issue,
    )

    payload = {
        "type": "assistant_response",
        "question": display_question,
        "answer": answer,
        "chart": display_chart,
        "kpi": kpi,
        "insight_summary": insight_summary,
        "anomaly_callouts": anomaly_callouts,
        "decision_signal": decision_signal,
        "summary": {"executive_summary": ""},
        "next_actions": compute_chip_eligibility(
            ctx,
            brief=brief,
            semantic_plan=semantic_plan,
            sql=sql,
            db_type=data_source or "azure_sql",
        ),
        "analysis_contract": ctx,
        "data_brief": brief,
        "result_scope": ctx.get("result_scope", {}),
        "data": {
            "headers": headers,
            "rows": display_rows,
            "total_rows": len(visible_rows),
            "truncated": len(visible_rows) > _PREVIEW_ROW_CAP,
            "column_formats": resolved_column_formats,
            "display_formats": resolved_display_formats,
            "diagnostics": _result_diagnostics(raw_rows),
            "currency_columns": [
                col for col, fmt in resolved_column_formats.items()
                if fmt == "currency"
            ],
        },
        "trust": {
            "sql": sql,
            "row_count": len(raw_rows),
            "duration_label": f"{duration_ms}ms" if duration_ms < 1000 else f"{duration_ms/1000:.1f}s",
            "data_source": data_source or "",
            "scope_badge": ctx.get("result_scope", {}).get("badge", ""),
            "confidence": confidence or {},
            "question_id": question_id,   # public key for feedback API (B3)
            "date_context": list((semantic_plan or {}).get("date_disclosures") or []),
        },
        "confidence": confidence or {},
    }
    return sanitize_response_text_fields(payload)


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
    if isinstance(val, (dict, list, tuple)):
        try:
            return json.dumps(val, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return ""
    return str(val)
