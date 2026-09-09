"""
Result-aware chart specification.

This module turns a returned row set into a small visualization contract that
the frontend can render safely. It is deliberately deterministic: no LLM, no DB
calls, and no dependence on raw schema metadata.
"""

from __future__ import annotations

import re
from typing import Any

from core.i18n import enum_label, t as _t
from core.schema_enrichment import display_label


_ID_SUFFIX_RE = re.compile(
    r"(?i)(^|_)(id|key|code|num|no|nbr|nr|ref|pk|fk|seq|idx|index|rank|number)$"
)
# Temporal detection by NAME matches whole tokens, not substrings — plain
# substring matching classified CONSOLIDATED_SALES ("date"), WIDTH ("dt") and
# OVERTIME_COST ("time") as temporal, which removed them from the measure list
# and suppressed the chart entirely.
_TEMPORAL_NAME_TOKENS = frozenset({
    "date", "day", "week", "month", "quarter", "year", "period", "dt", "time",
})
# Temporal detection by VALUE requires the whole value to look like a date —
# substring matching flipped dimension columns to temporal whenever any single
# value contained a month fragment (MARtin, NOVak, DECker, JANssen ...).
_DATEISH_VALUE_RES = [
    re.compile(r"^\d{4}[-/.]\d{1,2}([-/.]\d{1,2})?([T ].+)?$"),   # 2026-01[-05][ 10:30]
    re.compile(r"^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}$"),             # 05/01/2026
    re.compile(r"^(19|20)\d{2}$"),                                # bare year
    re.compile(r"^(19|20)\d{6}$"),                                # YYYYMMDD key
    re.compile(r"(?i)^q[1-4]([ \-/]*\d{2,4})?$"),                 # Q1, Q1 2026
    re.compile(r"(?i)^\d{4}[ \-]?q[1-4]$"),                       # 2026-Q1
    re.compile(                                                    # Jan, March 2026, Aug-26
        r"(?i)^(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|"
        r"aug(ust)?|sep(t(ember)?)?|oct(ober)?|nov(ember)?|dec(ember)?)"
        r"([ ,\-/']*\d{1,4})?$"
    ),
]
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
_DERIVED_METRIC_RE = re.compile(
    r"\b(buildup|build\s*up|gap|difference|diff|delta|variance|var|"
    r"leakage|impact|shortfall|surplus|deficit|excess|change|growth|"
    r"margin|percentage|percent|pct|rate|ratio|score)\b",
    re.IGNORECASE,
)
# An explicit "in a pie chart" / "as a bar graph" style request names the
# chart type directly, which should win over the generic intent heuristics
# below (those only ever infer "donut", never literal "pie").
_EXPLICIT_CHART_RE = re.compile(
    r"\b(pie|donut|doughnut|bar|column|line|area|scatter)\s*(chart|graph|plot)\b",
    re.IGNORECASE,
)
_EXPLICIT_CHART_MAP = {
    "pie": "pie",
    "donut": "donut",
    "doughnut": "donut",
    "bar": "bar",
    "column": "bar",
    "line": "line",
    "area": "area",
    "scatter": "scatter",
}

_AUXILIARY_SHARE_MEASURE_RE = re.compile(
    r"(?i)(?:contribution|share|percent|percentage|pct)(?:_of_total)?$|"
    r"(?:contribution|share|percent|percentage|pct)_"
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


def _name_tokens(col: str) -> list[str]:
    tokens: list[str] = []
    for part in re.split(r"[^a-zA-Z0-9]+", str(col or "")):
        tokens.extend(re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", part))
    return [t.lower() for t in tokens if t]


def _is_temporal_name(col: str) -> bool:
    return bool(_TEMPORAL_NAME_TOKENS & set(_name_tokens(col)))


def _value_is_dateish(value: Any) -> bool:
    s = str(value).strip()
    if not s:
        return False
    return any(rx.match(s) for rx in _DATEISH_VALUE_RES)


def _looks_temporal_values(values: list[Any]) -> bool:
    # A real date column is uniform: EVERY sampled value must look like a
    # date, not just one. (One month-fragment match in a joined sample used
    # to flip whole dimension columns to temporal.)
    sample = [v for v in values[:10] if v is not None and str(v).strip()]
    if not sample:
        return False
    if all(_value_is_dateish(v) for v in sample):
        return True
    numeric = [_to_float(v) for v in sample]
    numeric = [v for v in numeric if v is not None]
    if not numeric or len(numeric) != len(sample):
        return False
    # Integer YYYYMMDD keys, common in warehouse schemas. Fractional values
    # can never be date keys — without this guard, an all-decimal currency
    # column (every value carrying cents) left the filtered generator empty
    # and all([]) vacuously classified it as temporal, killing the chart.
    if not all(float(v).is_integer() for v in numeric):
        return False
    return all(19000101 <= int(v) <= 21001231 for v in numeric)


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
    """The business name for a column, for the labels a chart carries.

    This was a plain underscore-strip that left an all-caps column all-caps, so
    the chart's axis title, legend and tooltip said "WHS NM" and "BAL VAL AMT"
    beside prose that the L4 work had already taught to say "Warehouse Name"
    and "Balance Value Amount". The same reader, the same answer card, two
    spellings of the same column.

    The label written here reaches the browser as payload["column_roles"][col]
    ["label"], which both templates read before their own prettifier.

    Imported lazily: this module's contract is that it does no I/O and imports
    nothing heavy, and the expansion is only ever needed when a label is being
    built. display_label carries its own fallback -- it returns the plain
    title-cased spelling whenever the vocabulary has no opinion or raises -- so
    there is nothing to catch here, and a second copy of that fallback would
    only be a second thing to keep in step.

    One deliberate difference from the transform this replaces: it title-cases,
    so an all-caps token the vocabulary cannot expand now reads "Sku" rather
    than "SKU". That is the price of the chart and the prose saying the same
    thing, and the prose already said "Sku".
    """
    from core.schema_enrichment import display_label

    return display_label(col)


def _terms(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", str(text or "").lower()) if t]


def _measure_score(col: str, question: str) -> tuple[int, int]:
    """
    Rank measures by how directly they answer the user's question.

    SQL often returns component measures before the derived business answer:
    purchase quantity, sales quantity, inventory buildup. The chart should
    still treat inventory buildup as the primary measure when the question
    asks about buildup/gap/leakage/variance style analysis.
    """
    q_norm = _norm(question)
    q_terms = set(_terms(question))
    c_norm = _norm(col)
    c_terms = _terms(col)
    c_term_set = set(c_terms)

    score = 0
    if c_norm and c_norm in q_norm:
        score += 120

    meaningful_terms = [t for t in c_terms if t not in {"total", "sum", "avg", "average"}]
    if meaningful_terms and all(t in q_terms for t in meaningful_terms):
        score += 70

    score += len(c_term_set & q_terms) * 18

    if _DERIVED_METRIC_RE.search(col or "") and _DERIVED_METRIC_RE.search(question or ""):
        score += 45

    if any(t in c_term_set for t in {
        "buildup", "gap", "difference", "delta", "variance", "leakage",
        "shortfall", "surplus", "deficit", "excess",
    }):
        score += 18

    if c_terms and c_terms[0] in {"total", "sum"}:
        score -= 5

    return score, -len(c_terms)


def _rank_measures_for_question(measures: list[str], question: str) -> list[str]:
    indexed = list(enumerate(measures))
    indexed.sort(key=lambda item: (_measure_score(item[1], question), -item[0]), reverse=True)
    return [col for _, col in indexed]


def _default_series(measures: list[str], roles: dict[str, dict]) -> list[str]:
    """The measures a bar chart should draw by default.

    A rate and a money column cannot share a y-axis: on a scale of millions a
    percentage is a flat line along the bottom. Executed on the period-
    comparison shape, the axis came back as two money columns plus CHANGE_PCT
    and SHARE_OF_CHANGE_PCT.

    Percent-formatted measures are dropped only once at least two others
    remain, so a result whose only measures are rates still charts. They stay
    in `allowed` and in the table -- they simply stop being drawn against the
    wrong scale. A general improvement for any result mixing a money column and
    a rate column, not a multi-period special case.
    """
    plain = [col for col in measures
             if (roles.get(col) or {}).get("format") != "percentage"]
    return plain if len(plain) >= 2 else measures


def _composition_measures(measures: list[str]) -> list[str]:
    """Prefer the business value over an auxiliary calculated share column."""
    primary = [
        col for col in measures
        if not _AUXILIARY_SHARE_MEASURE_RE.search(str(col or ""))
    ]
    return primary or measures


def _pie_incompatibility(rows: list[dict], measure: str | None) -> str:
    """Return why a pie is misleading, or an empty string when it is safe."""
    if not measure:
        return _t("ui.chart.warn.pie_needs_measure")
    values = _numeric_values(rows, measure)
    if not values:
        return _t("ui.chart.warn.pie_needs_values")
    if any(value < 0 for value in values):
        return _t("ui.chart.warn.pie_negative")
    if sum(values) <= 0:
        return _t("ui.chart.warn.pie_nonpositive_total")
    return ""


def _primary_dimension(dimensions: list[str], roles: dict[str, dict]) -> str | None:
    if not dimensions:
        return None
    for col in dimensions:
        meta = roles.get(col, {})
        if meta.get("role") == "dimension" and not meta.get("is_technical_id"):
            return col
    return dimensions[0]


# The most series a grouped chart may draw. Eight is the length of the validated
# palette (static/js/chart-palettes.js), so a ninth group either wraps the
# colours -- two series drawn identically -- or is silently dropped by the chat
# page's seriesCap while the dashboard, which has no cap at all, keeps drawing
# it. A grid wider than the palette is a table.
_SERIES_CAP = 8


def _labels_of(rows: list[dict], col: str) -> list[str]:
    return [("" if row.get(col) is None else str(row.get(col))) for row in rows]


def _series_dimension(
    rows: list[dict], x_col: str | None, roles: dict[str, dict],
    headers: list[str],
) -> str | None:
    """The column that splits rows sharing an x label into separate series.

    "Revenue by warehouse for the last three months" returns a row per
    warehouse PER MONTH. Plotted as one series the axis carries each warehouse
    once per month and the line walks between rows belonging to different
    months -- a single jagged series that is really three flat ones
    interleaved. What the reader needs is a series per month, which is a chart
    both pages can already draw: they build one series per key in the row dict.
    All that was missing is knowing WHICH column splits the rows.

    Returns None whenever a grouped chart would be a guess, and the caller then
    leaves the result exactly as it draws today. The rules, in order:

    * x labels already distinct -- nothing to split, and this is the gate that
      makes every one-dimensional result byte-identical to before.
    * the candidate must be categorical and not the x column itself.
    * between 2 and _SERIES_CAP distinct values. One value splits nothing; more
      than the palette cannot be drawn honestly.
    * every (x, group) cell must appear once. A repeated cell means a THIRD
      dimension, and no two-axis chart can show three.
    A column functionally dependent on x -- a region that is fixed per warehouse
    -- falls out of the cell rule rather than needing one of its own: it cannot
    produce a distinct cell per row when x already repeats.
    * the grid must be at least half full, so a nearly-empty cross product does
      not become eight series of mostly gaps.
    * no group VALUE may collide with a column name, because the pivot uses
      those values as row keys.

    More than one candidate qualifying means the result has more dimensions
    than a chart has axes, and picking one of them would be arbitrary. None.
    """
    if not x_col or not rows:
        return None
    x_labels = _labels_of(rows, x_col)
    distinct_x = set(x_labels)
    if len(distinct_x) == len(x_labels):
        return None

    header_names = {str(h) for h in headers}
    qualifying: list[str] = []
    for col in headers:
        if col == x_col:
            continue
        if roles.get(col, {}).get("role") not in {"dimension", "identifier", "temporal"}:
            continue
        values = _labels_of(rows, col)
        groups = set(values)
        if not 2 <= len(groups) <= _SERIES_CAP:
            continue
        if groups & header_names:
            continue
        pairs = set(zip(x_labels, values))
        if len(pairs) != len(rows):
            continue
        if len(rows) * 2 < len(distinct_x) * len(groups):
            continue
        qualifying.append(col)

    return qualifying[0] if len(qualifying) == 1 else None


def _format_for_column(col: str, explicit_formats: dict[str, str]) -> str:
    key = _norm(col)
    explicit = explicit_formats.get(key)
    if explicit in {"currency", "percentage", "date", "text", "number"}:
        return explicit
    if _PERCENT_RE.search(col):
        return "percentage"
    if _CURRENCY_RE.search(col):
        return "currency"
    if _is_temporal_name(col):
        return "date"
    return "number" if _COUNT_RE.search(col) else "number"


def _column_roles(rows: list[dict], column_formats: dict | None = None) -> dict[str, dict]:
    explicit_formats = {_norm(k): str(v).lower() for k, v in (column_formats or {}).items()}
    roles: dict[str, dict] = {}
    for col in rows[0].keys():
        vals = _values(rows, col)
        numeric = _is_numeric_col(rows, col)
        explicit_format = explicit_formats.get(_norm(col))
        explicit_measure = explicit_format in {"currency", "percentage", "number"}
        # A column the admin explicitly formatted as a measure is never
        # temporal; a column NAMED like a measure (revenue/profit/count/...)
        # is never value-sniffed into temporal either — only an explicit date
        # format or a temporal name token can make it one.
        looks_measure_name = bool(
            _CURRENCY_RE.search(col or "")
            or _PERCENT_RE.search(col or "")
            or _COUNT_RE.search(col or "")
        )
        temporal = (
            explicit_format == "date"
            or (not explicit_measure and (
                _is_temporal_name(col)
                or (not looks_measure_name and _looks_temporal_values(vals))
            ))
        )
        identifier = _looks_identifier(rows, col) and not temporal and not (numeric and explicit_measure)
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


# ── Structural chart types ───────────────────────────────────────────────────
# Some result shapes are not inferred from column roles at all: a post-processor
# has already annotated the rows and the chart type follows from that annotation.
# A forecast marks its projected rows, a boxplot carries its five-number summary,
# a histogram carries its bins.
#
# These never reached the browser. `detect_chart_type` identified them correctly,
# but `infer_chart_spec` only ever offered {table, kpi, line, area, bar, pie,
# donut, scatter}, so `build_chart_payload`'s
#     effective_type = requested if requested in allowed else recommended_type
# silently downgraded every one of them to bar or line -- and the projection
# branch then stripped the very marker columns the ECharts branch needed. The
# forecast branch in portal_chat.html has never once run in production.
_STRUCTURAL_MARKERS: tuple[tuple[str, str], ...] = (
    ("is_forecast", "forecast"),
    ("bp_data", "boxplot"),
    ("funnel_pct", "funnel"),
)

# Columns a post-processor added for the renderer, not for the reader. They must
# be kept in the payload and kept OUT of role inference: `forecast_value` is
# numeric on projected rows and None on historical ones, and `_values` drops
# None, so it would otherwise be classified a measure and drawn as a series.
_STRUCTURAL_META_COLUMNS = frozenset({
    "is_forecast", "forecast_value", "forecast_low", "forecast_high",
    "__trend_slope", "__trend_r2", "__forecast_model", "__forecast_meta",
    "bp_data", "funnel_pct", "bin_min", "bin_max", "frequency_pct",
})

# Switched on one at a time. Each of these ECharts branches has never executed,
# so enabling all four at once would ship four untested renderers in one commit.
_STRUCTURAL_ENABLED = {"forecast"}


def structural_chart_type(rows: list[dict]) -> str:
    """The chart type implied by a post-processor's annotations, or "".

    Single source of truth: `detect_chart_type` used to keep its own copy of
    this marker list.
    """
    if not rows:
        return ""
    first = rows[0]
    for marker, kind in _STRUCTURAL_MARKERS:
        if first.get(marker) is not None:
            return kind
    # Histogram is the one type with no single distinguishing marker.
    if "bin_label" in first and "count" in first:
        return "histogram"
    return ""


def _structural_spec(rows: list[dict], kind: str, title: str) -> dict:
    """A spec that names a structural type as the only thing worth rendering."""
    headers = [h for h in rows[0].keys() if h not in _STRUCTURAL_META_COLUMNS]
    roles = _column_roles([
        {k: v for k, v in row.items() if k not in _STRUCTURAL_META_COLUMNS}
        for row in rows
    ], None) if headers else {}
    x_col = next((c for c in headers if roles.get(c, {}).get("role") == "temporal"), None)
    if x_col is None:
        x_col = next((c for c in headers if roles.get(c, {}).get("role") != "measure"), None)
    y_cols = [c for c in headers if roles.get(c, {}).get("role") == "measure"]
    return {
        "title": title,
        "intent": kind,
        "recommended_type": kind,
        "allowed_types": [kind, "table"],
        "renderable_types": [kind],
        "x": roles.get(x_col) if x_col else None,
        "y": [roles[c] for c in y_cols if c in roles],
        "series": None,
        "column_roles": roles,
        "warnings": [],
        "confidence": 0.95,
    }


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

    # A structurally-marked result is not a role-inference problem: the shape is
    # already decided, and offering "bar" here is what made these unreachable.
    _structural = structural_chart_type(rows)
    if _structural and _structural in _STRUCTURAL_ENABLED:
        return _structural_spec(rows, _structural, title)

    headers = list(rows[0].keys())
    if not headers:
        return None

    roles = _column_roles(rows, column_formats)
    measures = [c for c in headers if roles[c]["role"] == "measure"]
    measures = _rank_measures_for_question(measures, question or title)
    composition_measures = _composition_measures(measures)
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
        warnings.append(_t("ui.chart.warn.no_measure"))
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
        x_col = _primary_dimension(dimensions, roles)
        y_cols = measures[:2]
    elif temporals and measures and trend_q:
        intent = "trend"
        recommended = "area" if len(rows) <= 36 else "line"
        allowed = ["line", "area", "bar", "table"]
        x_col = _first(temporals)
        y_cols = measures[:4]
    elif share_q and composition_measures and dimensions and len(rows) <= 6 and len(composition_measures) == 1:
        x_col = _primary_dimension(dimensions, roles)
        y_cols = composition_measures[:1]
        pie_problem = _pie_incompatibility(rows, y_cols[0])
        if pie_problem:
            intent = "breakdown"
            recommended = "bar"
            allowed = ["bar", "table"]
            warnings.append(pie_problem)
        else:
            intent = "composition"
            recommended = "pie"
            allowed = ["pie", "donut", "bar", "table"]
    elif dimensions and measures:
        intent = "ranking" if ranking_q or len(rows) <= 50 else "breakdown"
        recommended = "bar"
        allowed = ["bar", "table"]
        if (
            len(rows) <= 10
            and share_q
            and len(composition_measures) == 1
            and not _pie_incompatibility(rows, composition_measures[0])
        ):
            allowed[1:1] = ["pie", "donut"]
        if len(measures) >= 2:
            allowed.append("scatter")
        x_col = _primary_dimension(dimensions, roles)
        y_cols = _default_series(measures, roles)[:4]
    elif temporals and measures:
        # Time axis + measures is chartable even without trend wording in the
        # question ("show revenue for each of the last 3 months" used to fall
        # through to table because it says neither "trend" nor "by month").
        intent = "trend"
        recommended = "area" if len(rows) <= 36 else "line"
        allowed = ["line", "area", "bar", "table"]
        x_col = _first(temporals)
        y_cols = measures[:4]
    elif len(measures) >= 2:
        intent = "correlation" if scatter_q else "measure_comparison"
        recommended = "scatter" if scatter_q else "bar"
        allowed = ["scatter", "bar", "table"]
        x_col = headers[0]
        y_cols = measures[:2] if scatter_q else measures[:4]

    requested_type = None
    explicit_match = _EXPLICIT_CHART_RE.search(q)
    if explicit_match:
        requested_type = _EXPLICIT_CHART_MAP.get(explicit_match.group(1).lower())

    if requested_type and measures:
        if requested_type in {"pie", "donut"} and x_col:
            requested_measure = _first(composition_measures)
            pie_problem = _pie_incompatibility(rows, requested_measure)
            if pie_problem:
                warnings.append(pie_problem)
            else:
                intent = "composition"
                recommended = requested_type
                y_cols = [requested_measure]
                allowed = [requested_type] + [t for t in allowed if t != requested_type]
        elif requested_type == "scatter":
            if len(measures) >= 2:
                intent = "correlation"
                recommended = "scatter"
                x_col = x_col or _primary_dimension(dimensions, roles)
                y_cols = measures[:2]
                allowed = ["scatter"] + [t for t in allowed if t != "scatter"]
            else:
                warnings.append(_t("ui.chart.warn.scatter_needs_two"))
        elif requested_type in {"bar", "line", "area"} and x_col:
            recommended = requested_type
            allowed = [requested_type] + [t for t in allowed if t != requested_type]
        else:
            warnings.append(_t("ui.chart.warn.poor_fit",
                                requested=enum_label("charttype", requested_type),
                                recommended=enum_label("charttype", recommended)))
        if "table" not in allowed:
            allowed.append("table")

    # ── A second dimension becomes a series, not a longer axis ───────────────
    # Until now `series` was cut into the contract and never wired up: it was
    # assigned in the trend branch alone, and read by nothing -- not by
    # build_chart_payload, not by either template. So a result grouped by two
    # things was drawn as one series walking between rows that belong to
    # different groups.
    #
    # This is the only place that knows every column's role AND has the final
    # chart type in hand, so it is where the choice belongs.
    if recommended in {"bar", "line", "area"} and x_col:
        series_col = _series_dimension(rows, x_col, roles, headers)
        if series_col:
            # The series slot now belongs to the group, so only one measure can
            # be drawn: two dimensions of variation cannot share it.
            if len(y_cols) == 2:
                warnings.append(_t(
                    "ui.chart.warn.grouped_two_measures",
                    series=roles[series_col]["label"],
                    drawn=roles[y_cols[0]]["label"],
                    other=roles[y_cols[1]]["label"],
                ))
            elif len(y_cols) > 2:
                warnings.append(_t(
                    "ui.chart.warn.grouped_many_measures",
                    series=roles[series_col]["label"],
                    drawn=roles[y_cols[0]]["label"],
                ))
            y_cols = y_cols[:1]
            # A scatter needs two measures. Leaving it offered after the
            # truncation ships a button that draws an empty chart.
            allowed = [t for t in allowed if t != "scatter"]
    elif recommended in {"pie", "donut"} and x_col and _labels_of(rows, x_col):
        # A pie is a part-to-whole claim. The renderer totals the rows sharing
        # a slice name -- it has to, or one category is drawn twice -- and a
        # total is only a whole when the measure adds up. Commit 4452268 fixed
        # the drawing and said the choice of chart belongs upstream; this is
        # upstream.
        labels = _labels_of(rows, x_col)
        if len(set(labels)) != len(labels) and y_cols:
            from core.analysis_contract import measure_class_for_column

            if measure_class_for_column(y_cols[0]) != "additive":
                intent = "breakdown"
                recommended = "bar"
                allowed = ["bar"] + [t for t in allowed
                                     if t not in {"pie", "donut", "bar"}]
                warnings.append(_t(
                    "ui.chart.warn.not_totallable",
                    measure=roles[y_cols[0]]["label"],
                ))
                if "table" not in allowed:
                    allowed.append("table")

    if x_col and roles.get(x_col, {}).get("is_technical_id"):
        warnings.append(_t("ui.chart.warn.technical_identifier",
                            column=display_label(x_col)))
    if recommended in {"pie", "donut"} and len(rows) > 6:
        warnings.append(_t("ui.chart.warn.too_many_slices"))
    if len(rows) > 50 and recommended == "bar":
        warnings.append(_t("ui.chart.warn.large_result"))

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
