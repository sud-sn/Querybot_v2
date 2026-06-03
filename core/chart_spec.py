"""
Result-aware chart specification.

This module turns a returned row set into a small visualization contract that
the frontend can render safely. It is deliberately deterministic: no LLM, no DB
calls, and no dependence on raw schema metadata.
"""

from __future__ import annotations

import re
from typing import Any


_ID_SUFFIX_RE = re.compile(
    r"(?i)(^|_)(id|key|code|num|no|nbr|nr|ref|pk|fk|seq|idx|index|rank|number)$"
)
_TEMPORAL_NAME_RE = re.compile(r"(?i)(date|day|week|month|quarter|year|period|dt|time)")
_TEMPORAL_VALUE_RE = re.compile(
    r"(?i)(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|q[1-4]|"
    r"\b20\d{2}[-/]\d{1,2}|\b20\d{6}\b|\b20\d{2}\b)"
)
_CURRENCY_RE = re.compile(
    r"(?i)(revenue|sales|amount|amt|charge|cost|cogs|price|margin|profit|usd|value|balance|total)"
)
_PERCENT_RE = re.compile(r"(?i)(percent|percentage|pct|rate|ratio|share|margin_pct)")
_COUNT_RE = re.compile(r"(?i)(count|cnt|rows|quantity|qty|volume|units)")

_TREND_RE = re.compile(
    r"\b(trend|over time|monthly|weekly|daily|yearly|by month|by week|by year|"
    r"by quarter|evolution|progression|growth|history|timeline|mom|yoy)\b",
    re.IGNORECASE,
)
_SHARE_RE = re.compile(
    r"\b(share|proportion|breakdown|distribution|percent|percentage|contribution|"
    r"split|composition|mix|part of total|of total)\b",
    re.IGNORECASE,
)
_SCATTER_RE = re.compile(
    r"\b(correlat|vs\.?|versus|scatter|relationship between|compare .{1,30} with|"
    r"related to|association between|show .{1,30} vs|x vs y)\b",
    re.IGNORECASE,
)
_RANKING_RE = re.compile(
    r"\b(top|bottom|highest|lowest|rank|ranking|largest|smallest|best|worst|leader)\b",
    re.IGNORECASE,
)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s or "").lower())


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        raw = str(value).strip()
        if not raw:
            return None
        raw = raw.replace("$", "").replace(",", "").replace("%", "")
        n = float(raw)
        return None if n != n else n
    except (TypeError, ValueError):
        return None


def _values(rows: list[dict], col: str) -> list[Any]:
    return [r.get(col) for r in rows if r.get(col) is not None]


def _numeric_values(rows: list[dict], col: str) -> list[float]:
    return [n for n in (_to_float(v) for v in _values(rows, col)) if n is not None]


def _is_numeric_col(rows: list[dict], col: str) -> bool:
    vals = _values(rows, col)
    if not vals:
        return False
    numeric = _numeric_values(rows, col)
    return len(numeric) >= max(1, len(vals) // 2)


def _is_date_key_name(col: str) -> bool:
    return bool(re.search(r"(?i)(^|_)dt(_|$)|date|_dt_dms_key$", col or ""))


def _looks_temporal_values(values: list[Any]) -> bool:
    sample = " ".join(str(v) for v in values[:10] if v is not None)
    if _TEMPORAL_VALUE_RE.search(sample):
        return True
    numeric = [_to_float(v) for v in values[:10]]
    numeric = [v for v in numeric if v is not None]
    if not numeric:
        return False
    # Integer YYYYMMDD keys, common in warehouse schemas.
    return all(19000101 <= int(v) <= 21001231 for v in numeric if float(v).is_integer())


def _looks_identifier(rows: list[dict], col: str) -> bool:
    # Business metric names should never be demoted to identifiers just because
    # a small demo result happens to contain unique whole numbers.
    if _CURRENCY_RE.search(col or "") or _PERCENT_RE.search(col or "") or _COUNT_RE.search(col or ""):
        return False
    if _ID_SUFFIX_RE.search(col or ""):
        return True
    numeric = _numeric_values(rows, col)
    vals = _values(rows, col)
    if len(numeric) < 2 or len(numeric) != len(vals):
        return False
    all_int = all(float(v).is_integer() for v in numeric)
    if not all_int:
        return False
    unique_ratio = len(set(int(v) for v in numeric)) / max(len(numeric), 1)
    return unique_ratio > 0.8


def _display_name(col: str) -> str:
    spaced = re.sub(r"[_\s]+", " ", str(col or "")).strip()
    return " ".join(part.capitalize() if not part.isupper() else part for part in spaced.split())


def _format_for_column(col: str, explicit_formats: dict[str, str]) -> str:
    key = _norm(col)
    explicit = explicit_formats.get(key)
    if explicit in {"currency", "percentage", "date", "text", "number"}:
        return explicit
    if _PERCENT_RE.search(col):
        return "percentage"
    if _CURRENCY_RE.search(col):
        return "currency"
    if _TEMPORAL_NAME_RE.search(col):
        return "date"
    return "number" if _COUNT_RE.search(col) else "number"


def _column_roles(rows: list[dict], column_formats: dict | None = None) -> dict[str, dict]:
    explicit_formats = {_norm(k): str(v).lower() for k, v in (column_formats or {}).items()}
    roles: dict[str, dict] = {}
    for col in rows[0].keys():
        vals = _values(rows, col)
        numeric = _is_numeric_col(rows, col)
        temporal = (
            _format_for_column(col, explicit_formats) == "date"
            or _TEMPORAL_NAME_RE.search(col or "") is not None
            or _looks_temporal_values(vals)
        )
        identifier = _looks_identifier(rows, col) and not temporal
        if temporal:
            role = "temporal"
            dtype = "temporal"
        elif numeric and not identifier:
            role = "measure"
            dtype = "numeric"
        elif identifier:
            role = "identifier"
            dtype = "categorical"
        else:
            role = "dimension"
            dtype = "categorical"
        roles[col] = {
            "column": col,
            "label": _display_name(col),
            "role": role,
            "type": dtype,
            "format": _format_for_column(col, explicit_formats) if role == "measure" else ("date" if temporal else "text"),
            "unique_count": len(set(str(v) for v in vals)),
            "non_null_count": len(vals),
            "is_technical_id": identifier,
        }
    return roles


def _first(cols: list[str]) -> str | None:
    return cols[0] if cols else None


def infer_chart_spec(
    rows: list[dict],
    question: str = "",
    column_formats: dict | None = None,
    title: str = "Results",
) -> dict | None:
    """
    Build a deterministic chart spec for returned rows.

    The spec is intentionally frontend-friendly and backward-compatible with
    the existing ECharts payload contract.
    """
    if not rows:
        return None
    headers = list(rows[0].keys())
    if not headers:
        return None

    roles = _column_roles(rows, column_formats)
    measures = [c for c in headers if roles[c]["role"] == "measure"]
    temporals = [c for c in headers if roles[c]["role"] == "temporal"]
    dimensions = [c for c in headers if roles[c]["role"] in {"dimension", "identifier"}]

    q = question or ""
    trend_q = bool(_TREND_RE.search(q))
    share_q = bool(_SHARE_RE.search(q))
    scatter_q = bool(_SCATTER_RE.search(q))
    ranking_q = bool(_RANKING_RE.search(q))

    warnings: list[str] = []
    intent = "table"
    recommended = "table"
    allowed = ["table"]
    x_col: str | None = None
    y_cols: list[str] = []
    series_col: str | None = None

    if not measures:
        warnings.append("No numeric measure column was found, so a table is safer than a chart.")
    elif len(rows) == 1:
        intent = "kpi"
        recommended = "kpi"
        allowed = ["kpi", "table"]
        x_col = _first(dimensions)
        y_cols = measures[:4]
    elif scatter_q and len(measures) >= 2:
        intent = "correlation"
        recommended = "scatter"
        allowed = ["scatter", "table"]
        x_col = _first(dimensions)
        y_cols = measures[:2]
    elif temporals and measures and (trend_q or _first(temporals)):
        intent = "trend"
        recommended = "area" if len(rows) <= 36 else "line"
        allowed = ["line", "area", "bar", "table"]
        x_col = _first(temporals)
        y_cols = measures[:4]
        if dimensions:
            series_col = _first([c for c in dimensions if c != x_col])
    elif share_q and measures and dimensions and len(rows) <= 6 and len(measures) == 1:
        intent = "composition"
        recommended = "donut"
        allowed = ["donut", "bar", "table"]
        x_col = _first(dimensions)
        y_cols = measures[:1]
    elif dimensions and measures:
        intent = "ranking" if ranking_q or len(rows) <= 50 else "breakdown"
        recommended = "bar"
        allowed = ["bar", "table"]
        if len(rows) <= 10 and share_q and len(measures) == 1:
            allowed.insert(1, "donut")
        if len(measures) >= 2:
            allowed.append("scatter")
        x_col = _first(dimensions)
        y_cols = measures[:4]
    elif len(measures) >= 2:
        intent = "correlation" if scatter_q else "measure_comparison"
        recommended = "scatter" if scatter_q else "bar"
        allowed = ["scatter", "bar", "table"]
        x_col = headers[0]
        y_cols = measures[:2] if scatter_q else measures[:4]

    if x_col and roles.get(x_col, {}).get("is_technical_id"):
        warnings.append(
            f"{x_col} looks like a technical identifier; prefer a semantic display column when available."
        )
    if recommended in {"pie", "donut"} and len(rows) > 6:
        warnings.append("Too many categories for a readable pie/donut chart; use bar or table.")
    if len(rows) > 50 and recommended == "bar":
        warnings.append("Large categorical result; a top-N filter or table view may be more readable.")

    renderable_types = [t for t in allowed if t not in {"table", "kpi"}]
    confidence = 0.92
    if warnings:
        confidence -= min(0.25, 0.08 * len(warnings))
    if recommended == "table":
        confidence = min(confidence, 0.72)

    return {
        "title": title,
        "intent": intent,
        "recommended_type": recommended,
        "allowed_types": allowed,
        "renderable_types": renderable_types,
        "x": roles.get(x_col) if x_col else None,
        "y": [roles[c] for c in y_cols],
        "series": roles.get(series_col) if series_col else None,
        "column_roles": roles,
        "warnings": warnings,
        "confidence": round(max(0.0, min(confidence, 1.0)), 2),
    }
