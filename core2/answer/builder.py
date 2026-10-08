"""From rows to the answer the portal shows (DESIGN §10).

Everything here is read from the plan and the logical query: which column is the
period, which a member, which a measure and how it is formatted, what the window
was, which periods are partial. Nothing is guessed from column names or question
words. The payload has the shape the web portal renders today (the same frame
the current pipeline sends), so the portal needs no change to show it.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Any, Callable

from core2.bootstrap import names
from core2.compile.compiler import Compiled, OutColumn
from core2.resolve.resolver import Condition, Logical
from core2.resolve.time import Range, label as period_label

log = logging.getLogger("querybot.core2")
_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December"]
_PERIOD_STYLE = {"day": "iso_date", "week": "iso_date", "month": "month_year_long", "quarter": "quarter",
                 "year": "year"}
_TABLE_FORMAT = {"currency": "currency", "percent": "percentage", "count": "number", "integer": "number",
                 "number": "number"}
PREVIEW_ROWS = 200


# ── values ─────────────────────────────────────────────────────────────────


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _gaps(logical: Logical, cols: "_Columns", raw: list[dict], truncated: bool) -> str:
    """Periods of a series with no row at all (a month never loaded): named, so a gap is not read as a zero.

    Not when rows were left out on purpose: a top N, a filter on the totals, a cut list.
    """
    periods = cols.of("period")
    if not logical.expected or not periods or not raw or logical.limit or logical.having or truncated:
        return ""
    present = {_day(r[periods[0].name]) for r in raw}
    missing = [d for d in logical.expected if d not in present]
    if not missing:
        return ""
    grain = periods[0].grain or "month"

    def name(d: dt.date) -> str:
        return period_label(d, grain, fiscal_start=logical.fiscal_start)

    # Missing periods side by side read as one stretch: "from Aug 2022 to Sep 2025".
    place = {d: i for i, d in enumerate(logical.expected)}
    runs: list[list[dt.date]] = []
    for d in missing:
        if runs and place[d] == place[runs[-1][-1]] + 1:
            runs[-1].append(d)
        else:
            runs.append([d])
    if len(runs) == 1 and len(runs[0]) > 1:
        return f"No data from {name(runs[0][0])} to {name(runs[0][-1])}: shown as a gap, not as zero."
    said = [name(r[0]) if len(r) == 1 else f"{name(r[0])} to {name(r[-1])}" for r in runs]
    text = said[0] if len(said) == 1 else f"{', '.join(said[:-1])} and {said[-1]}" if len(said) <= 3 \
        else f"{', '.join(said[:3])} and {len(said) - 3} more stretches"
    return f"No data for {text}: shown as a gap, not as zero."


def _day(value: Any) -> dt.date | None:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        try:
            return dt.date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


# Unit codes a reader takes for a size: "19,000 M" (metres) read as 19 billion beside "$1.2M".
_LIKE_A_SIZE = {"k", "m", "b", "t", "g", "mm", "bn"}


def fmt(value: Any, format_: str, *, unit: str | None = None) -> str:
    """A value as a sentence writes it: $1.2M, 38.0%, 1,234, 19,000 EA, 19,000 (unit M)."""
    number = _number(value)
    if number is None:
        return "no value" if value is None else str(value)
    if format_ == "currency":
        sign = "-" if number < 0 else ""
        number = abs(number)
        if number >= 1e9:
            return f"{sign}${number / 1e9:,.2f}B"
        if number >= 1e6:
            return f"{sign}${number / 1e6:,.2f}M"
        return f"{sign}${number:,.2f}"
    if format_ in ("percent", "percentage"):
        return f"{number:,.1f}%"
    if format_ == "days":
        whole = float(number).is_integer()
        text = f"{number:,.0f}" if whole else (f"{number:,.1f}" if abs(number) >= 1 else f"{number:,.2f}")
        return f"{text} day{'' if abs(number) == 1 else 's'}"
    if format_ in ("count", "integer") or float(number).is_integer():
        text = f"{number:,.0f}"
    else:
        text = f"{number:,.2f}"
    if unit and unit.strip().casefold() in _LIKE_A_SIZE:
        return f"{text} (unit {unit.strip()})"
    return f"{text} {unit}" if unit else text


_OP_TEXT = {"gt": "above", "gte": "at least", "lt": "below", "lte": "at most"}


def _listed(values: list, joiner: str) -> str:
    shown = [str(v) for v in values[:3]]
    if len(values) > 3:
        return f"{', '.join(shown)} {joiner} {len(values) - 3} more"
    return shown[0] if len(shown) == 1 else f"{', '.join(shown[:-1])} {joiner} {shown[-1]}"


def _compared(op: str, values: list[str]) -> str:
    """A comparison in words: "above 10", "between 7 and 14", "not 3"."""
    if op == "between" and len(values) == 2:
        return f"between {values[0]} and {values[1]}"
    if op in _OP_TEXT and values:
        return f"{_OP_TEXT[op]} {values[0]}"
    if op in ("ne", "not_in"):
        return f"not {_listed(values, 'or')}"
    return _listed(values, "or") if values else ""


def condition_words(c: Condition) -> str:
    """A condition as the answer's sentence names it: "for customer type Wholesale"."""
    if c.kind == "activity":
        return f"with {'an' if c.label[:1] in 'aeiou' else 'a'} {c.label}"
    if c.kind == "by":
        return f"by {c.label}"
    if c.kind == "snapshot":
        return "at the last snapshot"
    if c.kind == "date":
        return f"with {c.label} {span_words(c.dates)}" if c.dates is not None else ""
    if c.kind == "days":
        return f"where {c.label} is {_compared(c.op, [fmt(v, 'number') for v in c.values])} days"
    if c.kind == "total":
        # A threshold as written: "$250,000", not "$250,000.00".
        amounts = [f"${float(v):,.0f}" if c.format == "currency" and _number(v) is not None
                   and float(v).is_integer() else fmt(v, c.format or "number") for v in c.values]
        return f"with {c.label} {_compared(c.op, amounts)}"
    if c.op in ("eq", "in"):
        return f"for {c.label} {_listed(c.values, 'or')}"
    if c.op in ("ne", "not_in"):
        return f"excluding {c.label} {_listed(c.values, 'and')}"
    if c.op in ("contains", "starts_with"):
        return f"where {c.label} {c.op.replace('_', ' ')} \u201c{c.values[0] if c.values else ''}\u201d"
    if c.op == "is_null":
        return f"where {c.label} is empty"
    if c.op == "not_null":
        return f"where {c.label} is filled"
    return f"where {c.label} is {_compared(c.op, [str(v) for v in c.values])}"


def scope_words(logical: Logical) -> tuple[str, str]:
    """What the answer counts, around its period: ("with a customer order invoice", "in Q2 2026 by order
    date, for customer type Wholesale"). The first goes after the measure, the second is the period and
    every condition that narrowed the rows, so the sentence never reads as the whole data's figure."""
    before = " ".join(condition_words(c) for c in logical.conditions if c.kind == "activity")
    span = span_words(logical.window)
    by = " ".join(condition_words(c) for c in logical.conditions if c.kind == "by")
    rest = [w for w in (condition_words(c) for c in logical.conditions if c.kind not in ("activity", "by")) if w]
    after = " ".join(w for w in (span, by) if w)
    if rest:
        after = f"{after}, {', '.join(rest)}" if after else ", ".join(rest)
    return before, after


def scoped_label(logical: Logical, label: str) -> str:
    """A measure's name with every condition the answer applied: "Gross profit for profit centre X"."""
    order = {"activity": 0, "by": 1}
    words = [w for w in (condition_words(c) for c in sorted(logical.conditions, key=lambda c: order.get(c.kind, 2)))
             if w]
    return " ".join([label, *words])


def conditions_tail(logical: Logical) -> str:
    """The conditions to repeat in a next question about the same answer: ", for customer type Wholesale"."""
    words = [w for w in (condition_words(c) for c in logical.conditions) if w]
    return f", {', '.join(words)}" if words else ""


def span_words(rng: Range) -> str:
    """A window in words: "in 2025", "in March 2026", "from 1 Jun 2026 to 7 Jun 2026", "since ..."."""
    start, end = rng.start, rng.end
    if start is None and end is None:
        return ""
    if start and end:
        last = end - dt.timedelta(days=1)
        if last == start:
            return f"on {start.day} {start:%b %Y}"
        if start.month == 1 and start.day == 1 and end.month == 1 and end.day == 1 and end.year == start.year + 1:
            return f"in {start.year}"
        if start.day == 1 and end.day == 1 and last.year == start.year and last.month == start.month:
            return f"in {_MONTHS[start.month - 1]} {start.year}"
        if start.day == 1 and end.day == 1 and start.month in (1, 4, 7, 10) and last.year == start.year \
                and last.month == start.month + 2:
            return f"in Q{(start.month - 1) // 3 + 1} {start.year}"
        if start.day == 1 and end.day == 1:
            return f"from {start:%b %Y} to {last:%b %Y}"
        return f"from {start.day} {start:%b %Y} to {last.day} {last:%b %Y}"
    if start:
        return f"since {start.day} {start:%b %Y}"
    assert end is not None
    last = end - dt.timedelta(days=1)
    return f"up to {last.day} {last:%b %Y}"


def versus(rng: Range) -> str:
    """How a comparison names what it is compared with: "on April 2026", "against 17 Aug 2026"."""
    words = span_words(rng)
    if words.startswith("in "):
        return "on " + words[3:]
    return "against " + words.removeprefix("on ").removeprefix("from ")


def _period_badge(rng: Range) -> str:
    """A window as a short label: "April 2026", "Jan–Jun 2026", "Mar 2025 – Feb 2026", "since 1 Jan 2025"."""
    words = span_words(rng)
    for lead in ("in ", "on "):
        if words.startswith(lead):
            return words[len(lead):]
    if words.startswith("from "):
        start, _, end = words[5:].partition(" to ")
        first, last = start.split(" "), end.split(" ")
        if len(first) == len(last) == 2 and first[1] == last[1]:
            return f"{first[0]}–{last[0]} {first[1]}"
        return f"{start} – {end}"
    return words


def answer_badges(logical: Logical) -> list[dict[str, str]]:
    """What the answer counted, as short labels above its sentence: the period (and what it is compared
    with) and the date it was counted by. The sentence and the notes say the same at length."""
    badges: list[dict[str, str]] = []
    period = _period_badge(logical.window)
    if period and logical.compare is not None:
        prior = _period_badge(logical.compare)
        period = f"{period} vs {prior}" if prior else period
    if period:
        badges.append({"kind": "period", "text": period})
    roles = list(dict.fromkeys(p.date.role.name.lower() for p in logical.parts if p.date is not None))
    if roles:
        badges.append({"kind": "date", "text": "by " + " and ".join(roles)})
    return badges


# ── the table ──────────────────────────────────────────────────────────────


class _Columns:
    """Result columns matched to what the compiler said they are (case-insensitively).

    The grouping that keeps a quantity's units apart has the role "unit": it is a
    column of the table, never a member the sentence or the chart ranks.
    """

    def __init__(self, compiled: Compiled, names: list[str], unit: str | None = None):
        by_name = {c.name.casefold(): c for c in compiled.columns}
        self.index = {c.name: i for i, name in enumerate(names) for c in [by_name.get(name.casefold())] if c}
        self.columns = [replace(c, role="unit") if unit and c.name == unit else c
                        for c in compiled.columns if c.name in self.index]

    def of(self, role: str) -> list[OutColumn]:
        return [c for c in self.columns if c.role == role]


def _cell(column: OutColumn, value: Any, logical: Logical) -> Any:
    """A cell as the portal shows it: periods as their first day, time attributes named, fractions as %."""
    if column.role == "period":
        day = _day(value)
        if day is None:
            return ""
        if column.grain and column.grain.startswith("fiscal"):
            return period_label(day, column.grain, fiscal_start=logical.fiscal_start)
        return day.isoformat()
    if column.role == "time":
        number = _number(value)
        if number is None:
            return ""
        n = int(number)
        if column.name.startswith("day_of_week") and 1 <= n <= 7:
            return _DAYS[n - 1]
        if column.name.startswith("month_of_year") and 1 <= n <= 12:
            return _MONTHS[n - 1]
        if column.name.startswith("quarter_of_year"):
            return f"Q{n}"
        if column.name.startswith("is_weekend"):
            return "Weekend" if n else "Weekday"
        return str(n)
    if column.role in ("attribute", "member_code", "unit"):
        if value is None:
            return "Unknown"
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    number = _number(value)
    if number is None:
        return None
    if column.role in ("share", "pct_change"):
        return round(number * 100, 4)
    return round(number, 6) if not float(number).is_integer() else int(number)


def display_value(column: OutColumn, value: Any, logical: Logical) -> Any:
    """A cell as the portal shows it (for answers built outside :func:`build_answer`)."""
    return _cell(column, value, logical)


def _table_format(column: OutColumn) -> str:
    if column.role == "period":
        return "text" if column.grain and column.grain.startswith("fiscal") else "date"
    if column.role in ("attribute", "member_code", "time", "unit"):
        return "text"
    if column.role in ("share", "pct_change"):
        return "percentage"
    return _TABLE_FORMAT.get(column.format, "number")


@dataclass
class _Units:
    """The unit of each row when a quantity is kept apart by unit (issue E4), and the unit most rows are in."""

    column: OutColumn | None = None
    main: str | None = None
    count: int = 0

    def key(self, row: dict) -> str:
        value = row.get(self.column.name) if self.column else None
        return "Unknown" if value in (None, "") else str(value)

    def of(self, row: dict) -> str | None:
        """The unit to write after a value; none for a row without one."""
        if self.column is None:
            return None
        unit = self.key(row)
        return None if unit == "Unknown" else unit

    @property
    def mixed(self) -> bool:
        return self.count > 1

    def main_rows(self, rows: list[dict]) -> list[dict]:
        return [r for r in rows if self.key(r) == self.main] if self.mixed else rows


def _units(cols: _Columns, raw: list[dict], order: list[str]) -> _Units:
    """Which unit the sentence and the chart use: the one most of the data is in (from its profile),
    else the one most rows of the answer are in."""
    column = next(iter(cols.of("unit")), None)
    if column is None:
        return _Units()
    units = _Units(column)
    counts: dict[str, int] = {}
    totals: dict[str, float] = {}
    measure = next(iter(cols.of("measure")), None)
    for r in raw:
        key = units.key(r)
        counts[key] = counts.get(key, 0) + 1
        totals[key] = totals.get(key, 0.0) + abs((_number(r[measure.name]) if measure else 0.0) or 0.0)
    units.count = len(counts)
    rank = {u: i for i, u in enumerate(order)}
    units.main = min(counts, key=lambda u: (rank.get(u, len(rank)), u == "Unknown", -counts[u], -totals[u])) \
        if counts else None
    return units


# ── the answer ─────────────────────────────────────────────────────────────


def build_answer(question: str, logical: Logical, compiled: Compiled, columns: list[str], rows: list[tuple],
                 *, duration_ms: float = 0.0, data_source: str = "", question_id: str = "",
                 model_version: int = 0, truncated: bool = False, chart_type: str | None = None) -> dict[str, Any]:
    if len(rows) > logical.max_rows:
        truncated, rows = True, rows[:logical.max_rows]
    cols = _Columns(compiled, columns, unit=logical.unit_group)
    shown = cols.columns
    records = [{c.name: _cell(c, row[cols.index[c.name]], logical) for c in shown} for row in rows]
    raw = [{c.name: row[cols.index[c.name]] for c in shown} for row in rows]
    formats = {c.name: _table_format(c) for c in shown}
    labels = {c.name: _header(c) for c in shown}
    display: dict[str, dict] = {c.name: _digits(c) for c in shown
                                if c.role in ("measure", "prior", "change") and _digits(c)}
    for c in cols.of("period"):
        if c.grain and not c.grain.startswith("fiscal"):
            display[c.name] = {"type": "date", "style": _PERIOD_STYLE.get(c.grain, "iso_date")}

    partial = {d.isoformat() for d in logical.partial}
    caveats: list[str] = []
    if partial and cols.of("period"):
        grain = cols.of("period")[0].grain or "month"
        days = sorted(dt.date.fromisoformat(p) for p in partial)
        # A period that starts after the data ends has no data yet: said so, not called partly covered.
        ahead = [d for d in days if logical.data_last is not None and d > logical.data_last]
        part = [d for d in days if d not in ahead]
        if part:
            named = ", ".join(period_label(d, grain, fiscal_start=logical.fiscal_start) for d in part)
            caveats.append(f"{named} {'is' if len(part) == 1 else 'are'} only partly covered by the data: "
                           "not compared as whole periods.")
        if ahead and logical.data_last is not None:
            first = period_label(ahead[0], grain, fiscal_start=logical.fiscal_start)
            last = period_label(ahead[-1], grain, fiscal_start=logical.fiscal_start)
            span = first if len(ahead) == 1 else f"{first} to {last}"
            caveats.append(f"The data ends on {logical.data_last.day} {logical.data_last:%b %Y}: {span} "
                           f"{'has' if len(ahead) == 1 else 'have'} no data yet.")
    if truncated:
        caveats.append(f"Showing the first {logical.max_rows:,} rows.")
    gaps = _gaps(logical, cols, raw, truncated)
    if gaps:
        caveats.append(gaps)

    units = _units(cols, raw, logical.unit_order)
    headline = _headline(logical, cols, raw, records, partial, units)
    short_value, comparison = _lead(logical, cols, raw, units)
    drawn = [r for r, plain in zip(records, raw) if not units.mixed or units.key(plain) == units.main]
    chart = _chart(logical, cols, drawn, formats, labels, display)
    if chart is not None and units.main and units.main != "Unknown":
        chart["title"] = f"{chart['title']} ({units.main})"
        if units.mixed:
            chart["chart_warnings"] = [*chart["chart_warnings"], f"Only {units.main} is drawn: the other "
                                       f"{units.count - 1} units are in the table."]
    if units.mixed and (cols.of("attribute") or cols.of("time") or cols.of("period")):
        caveats.append(f"{cols.of('measure')[0].label} is counted in {units.count} units; the sentence and the "
                       f"chart use {units.main}, the unit of most rows. Every unit is in the table.")
    notes = list(logical.notes)
    if chart is not None and chart_type == "table":
        chart = None
    elif chart is not None and chart_type and chart_type != chart["chart_type"]:
        temporal = ((chart.get("chart_spec") or {}).get("x") or {}).get("role") == "temporal"
        allowed = [t for t in chart.get("allowed_types", []) if not (temporal and t == "pie")]
        if chart_type in allowed:
            chart["chart_type"] = chart["recommended_type"] = chart_type
            if chart_type not in chart["renderable_types"]:
                chart["renderable_types"] = [chart_type, *chart["renderable_types"]]
        else:
            notes.append(f"A {chart_type} chart does not fit this answer; it is shown as a "
                         f"{chart['chart_type']} chart.")
    from core2.answer.insights import summarize

    try:
        insights = summarize(logical, cols, raw, units, truncated=truncated)
    except Exception as exc:  # noqa: BLE001 - the findings are extra; the answer stands without them
        log.warning("core2 could not work out findings for %r: %s", question, exc)
        insights = []
    return frame(question, headline=headline, short_value=short_value, comparison=comparison, caveats=caveats,
                 insights=insights, badges=answer_badges(logical),
                 chart=chart, kpi=_kpi(logical, cols, raw, formats, units),
                 suggestions=[], headers=[c.name for c in shown], labels=labels, records=records,
                 formats=formats, display=display, sql=compiled.sql, row_count=len(rows), duration_ms=duration_ms,
                 data_source=data_source, question_id=question_id, notes=notes,
                 model_version=model_version)


def frame(question: str, *, headline: str, short_value: str = "", comparison: str = "", caveats: list[str],
          insights: list[str] | None = None, badges: list[dict[str, str]] | None = None,
          chart: dict | None, kpi: dict | None, suggestions: list[str], headers: list[str], labels: dict[str, str],
          records: list[dict], formats: dict[str, str], display: dict[str, dict] | None = None, sql: str,
          row_count: int, duration_ms: float, data_source: str, question_id: str, notes: list[str],
          model_version: int) -> dict[str, Any]:
    """The frame the web portal renders (the same shape today's pipeline sends)."""
    return {
        "type": "assistant_response",
        "engine": "core2",
        "question": question,
        # The portal's answer card: the value leads when there is one, the sentence otherwise.
        "answer": {"headline": headline, "short_value": short_value, "comparison": comparison,
                   "scope_badge": "", "scope_note": caveats[0] if caveats else "",
                   "badges": list(badges or [])},
        "result_scope": {"badge": "", "note": ""},
        "chart": chart,
        "kpi": kpi,
        "insight_summary": "",
        "anomaly_callouts": [],
        "key_insights": list(insights or []),     # what the rows show beyond the first sentence
        "coverage_caveats": caveats,
        "follow_up_suggestions": [s if isinstance(s, dict) else {"label": s, "question": s} for s in suggestions],
        "data": {
            "headers": headers,
            "header_labels": labels,
            "rows": records[:PREVIEW_ROWS],
            "total_rows": len(records),
            "truncated": len(records) > PREVIEW_ROWS,
            "column_formats": formats,
            "display_formats": display or {},
            "currency_columns": [n for n, f in formats.items() if f == "currency"],
        },
        "trust": {
            "sql": sql,
            "row_count": row_count,
            "duration_label": f"{duration_ms:.0f}ms" if duration_ms < 1000 else f"{duration_ms / 1000:.1f}s",
            "data_source": data_source,
            "scope_badge": "",
            "confidence": {},
            "question_id": question_id,
            "date_context": notes,
            "engine": "core2",
            "model_version": model_version,
        },
        "confidence": {},
        # Every row of the answer, for the full export; the portal bridge keeps it on the
        # server (the answer trace) and never sends it over the socket.
        "export_rows": records,
    }


def bar_chart(title: str, x: str, x_label: str, ys: list[tuple[str, str, str]], rows: list[dict],
              *, intent: str) -> dict[str, Any]:
    """A bar chart of ``rows``: ``x`` the category, each of ``ys`` (column, label, table format) a bar."""
    roles: dict[str, dict] = {x: {"column": x, "label": x_label, "role": "dimension", "format": "text"}}
    for column, label, format_ in ys:
        roles[column] = {"column": column, "label": label, "role": "measure", "format": format_}
    return {"title": title, "chart_type": "bar", "x_key": x, "y_keys": [y for y, _, _ in ys],
            "rows": [{x: "" if r.get(x) is None else str(r[x]), **{y: _number(r.get(y)) for y, _, _ in ys}}
                     for r in rows],
            "x_style": "", "column_roles": roles, "column_formats": {k: v["format"] for k, v in roles.items()},
            "renderable_types": ["bar", "line", "area"], "allowed_types": ["bar", "line", "area"],
            "recommended_type": "bar", "chart_spec": {"x": {"column": x, "role": "dimension"}, "column_roles": roles},
            "intent": intent, "grouped_by": None, "forecast_meta": None, "chart_warnings": []}


def _header(c: OutColumn) -> str:
    if c.role == "prior":
        return f"{c.label} (before)"
    if c.role == "change":
        return f"{c.label}: change"
    if c.role == "pct_change":
        return f"{c.label}: change %"
    if c.role == "share":
        return f"Share of {c.label.lower()}"
    if c.role == "period":
        return {"week": "Week of", "day": "Day"}.get(c.grain or "", "Period")
    return c.label


def _thing(label: str) -> str:
    """What a grouping counts, in the singular: "Warehouse name" -> "warehouse", "GL account" -> "GL account"."""
    words = [w if w.isupper() and len(w) > 1 else w.lower() for w in label.split()]   # "GL accounts"
    while len(words) > 1 and words[-1] in ("name", "description", "desc", "code", "label", "title"):
        words.pop()
    return " ".join(words)


def _noun(label: str) -> str:
    """What a grouping counts, in the plural: "Warehouse name" -> "warehouses", "Day of week" -> "days of week"."""
    words = _thing(label).split()
    if len(words) > 2 and words[1] == "of":
        return " ".join([names.plural(words[0]), *words[1:]])
    return names.plural(" ".join(words))


def _ranked_periods(logical: Logical, cols: _Columns) -> bool:
    """Periods ordered by a measure ("which month had the most"), not by time."""
    periods = cols.of("period")
    return bool(periods and logical.sort and logical.compare is None and logical.sort[0][0] != periods[0].name)


def _lower(label: str) -> str:
    """A name inside a sentence: "Net amount" -> "net amount"; "GMV" stays "GMV"."""
    return label if label[1:2].isupper() else label[:1].lower() + label[1:]


def _moved(by: float, ratio: float | None, format_: str, unit: str | None) -> str:
    """ "up $1,200 (+4.0%)", "down 3 (-2.1%)", or "unchanged" when nothing moved at the precision shown."""
    shown = fmt(abs(by), format_, unit=unit)
    if shown == fmt(0, format_, unit=unit):
        return "unchanged"
    return f"{'up' if by >= 0 else 'down'} {shown}" + (f" ({ratio * 100:+.1f}%)" if ratio is not None else "")


def _signed(number: float, format_: str, unit: str | None = None) -> str:
    return ("+" if number > 0 else "") + fmt(number, format_, unit=unit)


def _headline(logical: Logical, cols: _Columns, raw: list[dict], shown: list[dict], partial: set[str],
              units: _Units) -> str:
    measures = cols.of("measure")
    before, span = scope_words(logical)
    if not raw:
        return f"No rows match{(' ' + span) if span else ''}."
    if logical.intent == "list" or not measures:
        groups = cols.of("attribute")
        if not groups:
            return f"{len(raw):,} rows{(' ' + span) if span else ''}."
        listed = [str(r[groups[0].name]) for r in shown[:8] if r[groups[0].name] not in (None, "")]
        more = f", and {len(raw) - 8:,} more" if len(raw) > 8 else ""
        return f"{len(raw):,} {_noun(groups[0].label)}{(' ' + span) if span else ''}: {', '.join(listed)}{more}."
    m = measures[0]
    periods, members = cols.of("period"), cols.of("attribute") + cols.of("time")
    value = lambda r, c=m: fmt(r[c.name], c.format, unit=units.of(r))   # noqa: E731
    lead = f"{m.label}{(' ' + before) if before else ''}{(' ' + span) if span else ''}"
    if logical.compare is not None:
        change = next((c for c in cols.columns if c.role == "change" and c.measure == m.measure), None)
        pct = next((c for c in cols.columns if c.role == "pct_change" and c.measure == m.measure), None)
        prior_span = span_words(logical.compare).removeprefix("in ").removeprefix("from ")
        against = versus(logical.compare)
        prior = next((c for c in cols.columns if c.role == "prior" and c.measure == m.measure), None)

        def moved(r: dict, with_unit: bool = True) -> str:
            return _moved(_number(r[change.name]) or 0.0, _number(r[pct.name]) if pct else None,   # type: ignore[union-attr]
                          m.format, units.of(r) if with_unit else None)

        if not members and not periods and change is not None:
            if units.column is not None and len(raw) > 1:
                ordered = sorted(raw, key=lambda r: -abs(_number(r[change.name]) or 0.0))
                listed = [f"{units.of(r) or 'with no unit'} {moved(r, with_unit=False)}" for r in ordered[:3]]
                extra = len(raw) - 3
                tail = f"; {extra} more unit{'s' if extra > 1 else ''} in the table" if extra > 0 else ""
                return f"{lead} against {prior_span}: {'; '.join(listed)}{tail}."
            r = raw[0]
            if prior is not None and _number(r[prior.name]) is None:
                return (f"{lead}: {value(r)}. There is no {_lower(m.label)} "
                        f"{span_words(logical.compare) or 'in the period before'} to compare it with.")
            return f"{lead}: {value(r)}, {moved(r)} {against}."
        if members and change is not None:
            changes = [_number(r[change.name]) or 0.0 for r in raw]
            order = sorted(range(len(raw)), key=lambda i: -changes[i])
            hi, lo = order[0], order[-1]
            who = lambda i: str(shown[i][members[0].name])   # noqa: E731
            by = lambda i: _signed(changes[i], m.format, units.of(raw[i]))   # noqa: E731
            many = f"{len(raw)} {_noun(members[0].label)}{' shown' if logical.limit else ''}"
            if changes[hi] > 0 and changes[lo] < 0:
                said = f"{who(hi)} rose most ({by(hi)}); {who(lo)} fell most ({by(lo)})"
            elif len(raw) == 1:
                said = f"{who(hi)} {'rose' if changes[hi] > 0 else 'fell' if changes[hi] < 0 else 'held'}" + (
                    f" ({by(hi)})" if changes[hi] else "")
            elif changes[hi] > 0:
                said = (f"{'all' if changes[lo] > 0 else 'none of the'} {many} "
                        f"{'rose' if changes[lo] > 0 else 'fell'}; {who(hi)} rose most ({by(hi)})")
            elif changes[lo] < 0:
                said = (f"{'all' if changes[hi] < 0 else 'none of the'} {many} "
                        f"{'fell' if changes[hi] < 0 else 'rose'}; {who(lo)} fell most ({by(lo)})")
            else:
                said = f"no change for any of the {many}"
            return f"{m.label} {span} against {prior_span}: {said}."
    if periods and not members:
        grain = periods[0].grain or "month"

        def name(r: dict) -> str:
            """A period inside a sentence: "the week of 4 May 2026", "Aug 2025"."""
            text = period_label(_day(r[periods[0].name]) or dt.date.min, grain, fiscal_start=logical.fiscal_start)
            return "the w" + text[1:] if grain == "week" else text

        rows = units.main_rows(raw)
        if _ranked_periods(logical, cols):
            ranked_by = next((c for c in measures if c.name == logical.sort[0][0]), m)
            word = "highest" if logical.sort[0][1] else "lowest"
            top = rows[0]
            leader = name(top)
            text = (f"{leader[:1].upper() + leader[1:]} had the {word} {_lower(ranked_by.label)}"
                    f"{(' ' + span) if span else ''}: "
                    f"{value(top, ranked_by)}")
            if top[periods[0].name] is not None and str(_day(top[periods[0].name])) in partial:
                text += " (a period the data only partly covers)"
            if len(rows) > 1:
                text += f", then {name(rows[1])} ({value(rows[1], ranked_by)})"
            return text + "."
        complete = [r for r in rows if _day(r[periods[0].name]) and _day(r[periods[0].name]).isoformat()  # type: ignore[union-attr]
                    not in partial] or rows
        complete = sorted(complete, key=lambda r: _day(r[periods[0].name]) or dt.date.min)   # newest-first too
        complete = [r for r in complete if _number(r[m.name]) is not None] or complete     # a period with a value
        first, last = complete[0], complete[-1]
        peak = max(complete, key=lambda r: _number(r[m.name]) or float("-inf"))
        if len(complete) > 2 and len({_number(r[m.name]) for r in complete}) == 1:
            # Flat: "60 in Jan 2026 (the highest), 60 in Jun 2026" named a peak there is none of.
            unit = grain.replace("fiscal_", "fiscal ")
            return f"{lead}, by {unit}: {value(first)} in every {unit}" + (
                "." if span else f", {name(first)} to {name(last)}.")
        # The highest is named once: beside the first or last period when it is one of them, and not
        # at all between two periods (one of two is plainly the higher).
        highest = " (the highest)" if len(complete) > 2 else ""
        text = f"{lead}, by {grain.replace('fiscal_', 'fiscal ')}: {value(first)} in {name(first)}"
        if peak is first:
            text += highest
        if len(complete) > 1:
            text += f", {value(last)} in {name(last)}"
            if peak is last:
                text += highest
            elif peak is not first:
                text += f"; highest {value(peak)} in {name(peak)}"
        return text + "."
    if members and not periods:
        g = members[0]
        # Ranked lowest first ("the 5 items with the least gross profit"), the answer names the lowest:
        # "leads with" the largest read as the opposite of what was asked.
        lowest = bool(logical.sort) and not logical.sort[0][1] and logical.sort[0][0] in {c.name for c in measures}
        if lowest:
            ranked_by = next(c for c in measures if c.name == logical.sort[0][0])
            low = min(range(len(raw)), key=lambda i: _number(raw[i][ranked_by.name]) or float("inf"))
            count_of = len({str(r[g.name]) for r in shown})
            among = f" of the {count_of:,} {_noun(g.label)} shown" if count_of > 1 and g.role != "time" else ""
            return f"{lead}: {shown[low][g.name]} is lowest{among}, at {value(raw[low], ranked_by)}."
        best = max(range(len(raw)), key=lambda i: _number(raw[i][m.name]) or float("-inf"))
        top = raw[best]
        leader = str(shown[best][g.name])
        total = sum(_number(r[m.name]) or 0.0 for r in raw)
        share = ""
        # The share the query worked out is of everything; the rows shown may be the top 10 of many, and a
        # share of their sum said "11%" beside a share column saying 3%.
        worked_out = next((c for c in cols.of("share") if c.measure == (m.measure or None)), None)
        if not units.mixed and worked_out is not None and _number(top.get(worked_out.name)) is not None:
            portion = _number(top[worked_out.name]) or 0.0      # a fraction, as the query returns it
            share = f" ({portion:.0%} of the total)" if 0 < portion < 1 else ""
        elif not units.mixed and (logical.share or (m.format in ("currency", "number", "integer", "count") and total
                                                    and len(raw) > 1 and not logical.limit)):
            portion = (_number(top[m.name]) or 0.0) / total if total else 0.0
            share = f" ({portion:.0%} of the total)" if 0 < portion < 1 else ""
        count_of = len({str(r[g.name]) for r in shown})
        # Days of the week and weekends are always the same few: counting them says nothing ("across 2 is weekends").
        count = f" across {count_of:,} {_noun(g.label)}" if count_of > 1 and g.role != "time" else ""
        # A tie is not a lead: five groups of 12 members each read "<the first group> leads with 12".
        tied = [i for i in range(len(raw)) if _number(raw[i][m.name]) == _number(top[m.name])]
        if len(tied) > 1 and not units.mixed:
            if len(tied) == len(raw):
                return f"{lead}: each of the {len(raw):,} {_noun(g.label)} has {value(top)}."
            named = [str(shown[i][g.name]) for i in tied]
            leaders = (", ".join(named[:-1]) + f" and {named[-1]}" if len(named) <= 3
                       else ", ".join(named[:3]) + f" and {len(named) - 3} more")
            return f"{lead}: {leaders} lead with {value(top)} each{share.replace(' of the total)', ' of the total each)')}{count}."
        # The other measures asked for, for the same leader: "and $456K gross profit".
        others = [_valued(fmt(top[c.name], c.format, unit=units.of(top)), c) for c in measures[1:] if c.name in top]
        joined = ", ".join(others[:-1]) + f" and {others[-1]}" if len(others) > 1 else "".join(others)
        also = f", and {joined}{',' if count else ''}" if others else ""
        return f"{lead}: {leader} leads with {value(top)}{share}{also}{count}."
    if not members and not periods:
        if units.column is not None and len(raw) > 1:
            ordered = sorted(raw, key=lambda r: -(_number(r[m.name]) or 0.0))
            listed = [value(r) if units.of(r) else f"{value(r)} with no unit" for r in ordered[:4]]
            extra = len(raw) - 4
            tail = f" and {extra} more unit{'s' if extra > 1 else ''}" if extra > 0 else ""
            return f"{lead}: {', '.join(listed)}{tail}."
        r = raw[0]
        others = [_valued(fmt(r[c.name], c.format, unit=units.of(r)), c) if c.format == "days"
                  else f"{c.label} {fmt(r[c.name], c.format, unit=units.of(r))}" for c in measures[1:]]
        tail = f"; {', '.join(others)}" if others else ""
        return f"{lead}: {value(r)}{tail}."
    if periods and members:
        said = _led_by_period(logical, periods[0], members[0], m, raw, shown, partial, units, value)
        if said:
            every = (periods[0].grain or "month").replace("fiscal_", "fiscal ")
            return f"{lead}, by {every} and {_thing(members[0].label)}: {said}."
    return f"{lead}: {len(raw):,} rows."


def _led_by_period(logical: Logical, period: OutColumn, member: OutColumn, m: OutColumn, raw: list[dict],
                   shown: list[dict], partial: set[str], units: _Units, value: Callable[[dict], str]) -> str:
    """Who led a series broken down by a member: "North led in every month from Jan 2026 to May 2026,
    among the 5 regions", or the leaders of the first and last whole periods. Never a total over the
    periods: a share, an average or a distinct count does not add up across them."""
    grain = period.grain or "month"
    rows = [i for i in range(len(raw)) if (not units.mixed or units.key(raw[i]) == units.main)
            and _day(raw[i][period.name]) is not None and _number(raw[i][m.name]) is not None]
    if not rows:
        return ""
    days = sorted({d for d in (_day(raw[i][period.name]) for i in rows) if d is not None})
    whole = [d for d in days if d.isoformat() not in partial] or days

    def leader(day: dt.date) -> int:
        return max((i for i in rows if _day(raw[i][period.name]) == day),
                   key=lambda i: _number(raw[i][m.name]) or 0.0)

    def name(day: dt.date) -> str:
        text = period_label(day, grain, fiscal_start=logical.fiscal_start)
        return "the w" + text[1:] if grain == "week" else text

    leaders = [leader(d) for d in whole]
    who = [str(shown[i][member.name]) for i in leaders]
    count = len({str(shown[i][member.name]) for i in rows})
    among = f", among the {count:,} {_noun(member.label)}" if count > 1 else ""
    if len(whole) == 1:
        return f"{who[0]} led in {name(whole[0])} with {value(raw[leaders[0]])}{among}"
    if len(set(who)) == 1:
        unit = grain.replace("fiscal_", "fiscal ")
        return f"{who[0]} led in every {unit} from {name(whole[0])} to {name(whole[-1])}{among}"
    return (f"{who[0]} led in {name(whole[0])} ({value(raw[leaders[0]])}); "
            f"{who[-1]} in {name(whole[-1])} ({value(raw[leaders[-1]])})")


def _valued(value: str, column: OutColumn) -> str:
    """A value beside what it is: "$7.73M gross profit"; "9.5 days from order to invoice", never "days days"."""
    label = _lower(column.label)
    if column.format == "days" and label.startswith("days "):
        return f"{value} {label[5:]}"
    return f"{value} {label}"


def _lead(logical: Logical, cols: _Columns, raw: list[dict], units: _Units) -> tuple[str, str]:
    """The card's lead value and what it is compared with, when the answer is one number."""
    measures = cols.of("measure")
    if len(raw) != 1 or cols.of("period") or cols.of("attribute") or cols.of("time") or not measures:
        return "", ""
    m, r = measures[0], raw[0]
    value = fmt(r[m.name], m.format, unit=units.of(r))
    if logical.compare is not None:
        change = next((c for c in cols.columns if c.role == "change" and c.measure == m.measure), None)
        pct = next((c for c in cols.columns if c.role == "pct_change" and c.measure == m.measure), None)
        if change is not None:
            prior = next((c for c in cols.columns if c.role == "prior" and c.measure == m.measure), None)
            if prior is not None and _number(r[prior.name]) is None:
                return value, f"nothing {span_words(logical.compare)} to compare with"
            ratio = _number(r[pct.name]) if pct is not None else None
            return value, f"{_moved(_number(r[change.name]) or 0.0, ratio, m.format, units.of(r))} {versus(logical.compare)}"
    return value, scope_words(logical)[1]


def _digits(column: OutColumn) -> dict:
    """How many decimals the portal shows for a measure: none for a count, a tenth for days."""
    if column.format in ("count", "integer"):
        return {"fraction_digits": 0}
    if column.format == "days":
        return {"fraction_digits": 1}
    return {}


def _kpi(logical: Logical, cols: _Columns, raw: list[dict], formats: dict[str, str], units: _Units) -> dict | None:
    measures = cols.of("measure")
    if len(raw) != 1 or cols.of("period") or cols.of("attribute") or len(measures) != 1 or logical.compare:
        return None
    m = measures[0]
    value = raw[0][m.name]
    unit = units.of(raw[0])
    return {"label": f"{m.label} ({unit})" if unit else m.label,
            "value": value if _number(value) is None else _number(value),
            "format": formats[m.name], "display_format": _digits(m),
            "state": "missing" if _number(value) is None else "ready", "note": scope_words(logical)[1]}


def _chart(logical: Logical, cols: _Columns, records: list[dict], formats: dict[str, str],
           labels: dict[str, str], display: dict[str, dict]) -> dict | None:
    measures = cols.of("measure")
    periods, members = cols.of("period"), cols.of("attribute") + cols.of("time")
    if not measures or len(records) < 2 or logical.intent == "list":
        return None
    m = measures[0]
    ranked = _ranked_periods(logical, cols) and not members
    compare: dict | None = None
    if logical.compare is not None and members:
        prior = next((c for c in cols.columns if c.role == "prior" and c.measure == m.measure), None)
        x, ys, kind = members[0], [c.name for c in (prior, m) if c is not None], "bar"
        if prior is not None:
            # The same members at two times: a dot per period on one line per member, the change beside it.
            kind = "dumbbell"
            compare = {"prior": prior.name, "current": m.name, "prior_label": _period_badge(logical.compare),
                       "current_label": _period_badge(logical.window)}
    elif periods and ranked:
        # A ranking of periods is drawn as ranked bars, named as periods ("Aug 2025"), never as a time line.
        x, ys, kind = periods[0], [m.name], "bar"
    elif periods:
        x, ys, kind = periods[0], [c.name for c in measures], "line"
        if members:   # one line per member
            return _pivot(logical, periods[0], members[0], m, records, formats, labels, display)
    elif members:
        # Every measure asked for is drawn: one of a unit shares a plot, another unit gets a panel.
        x, ys = members[0], [c.name for c in measures]
        kind = "pie" if logical.share and len(records) <= 6 and len(ys) == 1 else "bar"
    else:
        return None
    temporal = x.role == "period" and not ranked
    roles = {c.name: {"column": c.name, "label": labels[c.name],
                      "role": "temporal" if c.name == x.name and temporal else (
                          "measure" if c.name in ys else "dimension"),
                      "format": "percentage" if formats.get(c.name) == "percentage" else (
                          "text" if c.name == x.name and ranked else formats.get(c.name))}
             for c in cols.columns if c.name in ys or c.name == x.name}

    def x_value(r: dict) -> str:
        if ranked and _day(r[x.name]) is not None:
            return period_label(_day(r[x.name]) or dt.date.min, x.grain or "month", fiscal_start=logical.fiscal_start)
        return "" if r[x.name] is None else str(r[x.name])

    rows = [{x.name: x_value(r), **{y: _number(r[y]) for y in ys}} for r in records]
    if temporal:
        rows.sort(key=lambda r: str(r[x.name]))     # a time axis runs forward, whatever order the table is in
    facets: list[list[str]] = []
    if len(ys) > 1 and kind == "bar" and compare is None:
        by_unit: dict[str, list[str]] = {}
        for y in ys:
            by_unit.setdefault(str(roles[y]["format"]), []).append(y)
        facets = list(by_unit.values()) if len(by_unit) > 1 else []
    # The shapes that fit: a category is never a line, a ranking of periods is only bars.
    if kind == "dumbbell":
        renderable, allowed = ["dumbbell", "bar"], ["dumbbell", "bar"]
    elif kind == "pie":
        renderable, allowed = ["pie", "bar"], ["pie", "bar"]
    elif temporal:
        renderable, allowed = ["line", "area", "bar"], ["line", "area", "bar"]
    else:
        renderable, allowed = ["bar"], ["bar", "pie"] if len(ys) == 1 and not ranked else ["bar"]
    return {"title": labels[m.name] if len(ys) == 1 or compare else _listed([labels[ys[0]], *(_lower(labels[y]) for y in ys[1:])], "and"),
            "chart_type": kind, "x_key": x.name, "y_keys": ys, "rows": rows,
            "x_style": "" if ranked else (display.get(x.name) or {}).get("style", ""),
            "column_roles": roles, "column_formats": {k: v["format"] for k, v in roles.items()},
            "renderable_types": renderable, "allowed_types": allowed, "recommended_type": kind,
            "compare": compare, "facets": facets,
            "chart_spec": {"x": {"column": x.name, "role": roles[x.name]["role"]}, "column_roles": roles},
            "intent": logical.intent, "grouped_by": None, "forecast_meta": None, "chart_warnings": []}


def _pivot(logical: Logical, period: OutColumn, member: OutColumn, m: OutColumn, records: list[dict],
           formats: dict[str, str], labels: dict[str, str], display: dict[str, dict]) -> dict:
    totals: dict[str, float] = {}
    for r in records:
        key = "Unknown" if r[member.name] is None else str(r[member.name])
        totals[key] = totals.get(key, 0.0) + (_number(r[m.name]) or 0.0)
    series = [k for k, _ in sorted(totals.items(), key=lambda kv: -kv[1])][:8]
    by_period: dict[str, dict] = {}
    for r in records:
        key = "Unknown" if r[member.name] is None else str(r[member.name])
        if key not in series:
            continue
        row = by_period.setdefault(str(r[period.name]), {period.name: str(r[period.name])})
        row[key] = _number(r[m.name])
    rows = [by_period[k] for k in sorted(by_period)]
    roles: dict[str, dict] = {period.name: {"column": period.name, "label": labels[period.name], "role": "temporal"}}
    for s in series:
        roles[s] = {"column": s, "label": s, "role": "measure", "format": formats[m.name]}
    return {"title": f"{labels[m.name]} by {_thing(labels[member.name])}", "chart_type": "line",
            "x_key": period.name, "y_keys": series, "rows": rows,
            "x_style": (display.get(period.name) or {}).get("style", ""), "column_roles": roles,
            "column_formats": {s: formats[m.name] for s in series}, "renderable_types": ["line", "bar", "area"],
            "allowed_types": ["line", "bar", "area"], "recommended_type": "line",
            "chart_spec": {"x": {"column": period.name, "role": "temporal"}, "column_roles": roles},
            "intent": logical.intent, "grouped_by": member.name, "grouped_measure": m.name,
            "forecast_meta": None, "chart_warnings": [] if len(totals) <= 8 else [
                f"Showing the 8 largest of {len(totals)} {labels[member.name].lower()} values."]}
