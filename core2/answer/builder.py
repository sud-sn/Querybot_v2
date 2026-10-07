"""From rows to the answer the portal shows (DESIGN §10).

Everything here is read from the plan and the logical query: which column is the
period, which a member, which a measure and how it is formatted, what the window
was, which periods are partial. Nothing is guessed from column names or question
words. The payload has the shape the web portal renders today (the same frame
the current pipeline sends), so the portal needs no change to show it.
"""

from __future__ import annotations

import datetime as dt
import math
from decimal import Decimal
from typing import Any

from core2.bootstrap import names
from core2.compile.compiler import Compiled, OutColumn
from core2.resolve.resolver import Logical
from core2.resolve.time import Range, label as period_label

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


def fmt(value: Any, format_: str, *, unit: str | None = None) -> str:
    """A value as a sentence writes it: $1.2M, 38.0%, 1,234."""
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
    if format_ in ("count", "integer") or float(number).is_integer():
        text = f"{number:,.0f}"
    else:
        text = f"{number:,.2f}"
    return f"{text} {unit}" if unit else text


def _span(rng: Range) -> str:
    """A window in words: "in 2025", "in March 2026", "from 1 Jun 2026 to 7 Jun 2026", "since ..."."""
    start, end = rng.start, rng.end
    if start is None and end is None:
        return ""
    if start and end:
        last = end - dt.timedelta(days=1)
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


# ── the table ──────────────────────────────────────────────────────────────


class _Columns:
    """Result columns matched to what the compiler said they are (case-insensitively)."""

    def __init__(self, compiled: Compiled, names: list[str]):
        by_name = {c.name.casefold(): c for c in compiled.columns}
        self.index = {c.name: i for i, name in enumerate(names) for c in [by_name.get(name.casefold())] if c}
        self.columns = [c for c in compiled.columns if c.name in self.index]

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
    if column.role in ("attribute", "member_code"):
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


def _table_format(column: OutColumn) -> str:
    if column.role == "period":
        return "text" if column.grain and column.grain.startswith("fiscal") else "date"
    if column.role in ("attribute", "member_code", "time"):
        return "text"
    if column.role in ("share", "pct_change"):
        return "percentage"
    return _TABLE_FORMAT.get(column.format, "number")


# ── the answer ─────────────────────────────────────────────────────────────


def build_answer(question: str, logical: Logical, compiled: Compiled, columns: list[str], rows: list[tuple],
                 *, duration_ms: float = 0.0, data_source: str = "", question_id: str = "",
                 model_version: int = 0, truncated: bool = False) -> dict[str, Any]:
    cols = _Columns(compiled, columns)
    shown = cols.columns
    records = [{c.name: _cell(c, row[cols.index[c.name]], logical) for c in shown} for row in rows]
    raw = [{c.name: row[cols.index[c.name]] for c in shown} for row in rows]
    formats = {c.name: _table_format(c) for c in shown}
    labels = {c.name: _header(c) for c in shown}
    display: dict[str, dict] = {}
    for c in cols.of("period"):
        if c.grain and not c.grain.startswith("fiscal"):
            display[c.name] = {"type": "date", "style": _PERIOD_STYLE.get(c.grain, "iso_date")}

    partial = {d.isoformat() for d in logical.partial}
    caveats: list[str] = []
    if partial and cols.of("period"):
        grain = cols.of("period")[0].grain or "month"
        named = ", ".join(period_label(dt.date.fromisoformat(p), grain, fiscal_start=logical.fiscal_start)
                          for p in sorted(partial))
        caveats.append(f"{named} {'is' if len(partial) == 1 else 'are'} only partly covered by the data: "
                       "not compared as whole periods.")
    if truncated or len(rows) > logical.max_rows:
        caveats.append(f"Showing the first {logical.max_rows:,} rows.")

    headline = _headline(logical, cols, raw, records, partial)
    payload: dict[str, Any] = {
        "type": "assistant_response",
        "engine": "core2",
        "question": question,
        "answer": headline,
        "chart": _chart(logical, cols, records, formats, labels, display),
        "kpi": _kpi(logical, cols, raw, formats),
        "insight_summary": "",
        "anomaly_callouts": [],
        "coverage_caveats": caveats,
        "follow_up_suggestions": _next(logical, cols),
        "data": {
            "headers": [c.name for c in shown],
            "header_labels": labels,
            "rows": records[:PREVIEW_ROWS],
            "total_rows": len(records),
            "truncated": len(records) > PREVIEW_ROWS,
            "column_formats": formats,
            "display_formats": display,
            "currency_columns": [n for n, f in formats.items() if f == "currency"],
        },
        "trust": {
            "sql": compiled.sql,
            "row_count": len(rows),
            "duration_label": f"{duration_ms:.0f}ms" if duration_ms < 1000 else f"{duration_ms / 1000:.1f}s",
            "data_source": data_source,
            "scope_badge": "",
            "confidence": {},
            "question_id": question_id,
            "date_context": list(logical.notes),
            "engine": "core2",
            "model_version": model_version,
        },
        "confidence": {},
    }
    return payload


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


def _noun(label: str) -> str:
    """What a grouping counts, in the plural: "Warehouse name" -> "warehouses", "Day of week" -> "days of week"."""
    words = label.lower().split()
    while len(words) > 1 and words[-1] in ("name", "description", "desc", "code", "label", "title"):
        words.pop()
    if len(words) > 2 and words[1] == "of":
        return " ".join([names.plural(words[0]), *words[1:]])
    return names.plural(" ".join(words))


def _headline(logical: Logical, cols: _Columns, raw: list[dict], shown: list[dict], partial: set[str]) -> str:
    measures = cols.of("measure")
    span = _span(logical.window)
    if not raw:
        return f"No rows match{(' ' + span) if span else ''}."
    if logical.intent == "list" or not measures:
        groups = cols.of("attribute")
        if not groups:
            return f"{len(raw):,} rows."
        listed = [str(r[groups[0].name]) for r in shown[:8] if r[groups[0].name] not in (None, "")]
        more = f", and {len(raw) - 8:,} more" if len(raw) > 8 else ""
        return f"{len(raw):,} {_noun(groups[0].label)}: {', '.join(listed)}{more}."
    m = measures[0]
    periods, members = cols.of("period"), cols.of("attribute") + cols.of("time")
    value = lambda r, c=m: fmt(r[c.name], c.format)   # noqa: E731
    lead = f"{m.label}{(' ' + span) if span else ''}"
    if logical.compare is not None:
        change = next((c for c in cols.columns if c.role == "change" and c.measure == m.measure), None)
        pct = next((c for c in cols.columns if c.role == "pct_change" and c.measure == m.measure), None)
        prior_span = _span(logical.compare)
        if not members and not periods and change is not None:
            r = raw[0]
            moved = _number(r[change.name]) or 0.0
            ratio = _number(r[pct.name]) if pct else None
            pct_text = f" ({ratio * 100:+.1f}%)" if ratio is not None else ""
            return (f"{lead}: {value(r)}, {'up' if moved >= 0 else 'down'} {fmt(abs(moved), m.format)}{pct_text} "
                    f"on {prior_span.removeprefix('in ')}.")
        if members and change is not None:
            order = sorted(range(len(raw)), key=lambda i: -(_number(raw[i][change.name]) or 0.0))
            top, bottom = raw[order[0]], raw[order[-1]]
            named = {id(raw[i]): str(shown[i][members[0].name]) for i in range(len(raw))}
            who = lambda r: named[id(r)]  # noqa: E731
            parts = [f"{who(top)} rose most ({fmt(_number(top[change.name]), m.format)})"]
            if (_number(bottom[change.name]) or 0) < 0:
                parts.append(f"{who(bottom)} fell most ({fmt(_number(bottom[change.name]), m.format)})")
            return f"{m.label} {span} against {prior_span.removeprefix('in ')}: {'; '.join(parts)}."
    if periods and not members:
        complete = [r for r in raw if _day(r[periods[0].name]) and _day(r[periods[0].name]).isoformat()  # type: ignore[union-attr]
                    not in partial] or raw
        grain = periods[0].grain or "month"
        name = lambda r: period_label(_day(r[periods[0].name]) or dt.date.min, grain,  # noqa: E731
                                      fiscal_start=logical.fiscal_start)
        first, last = complete[0], complete[-1]
        peak = max(complete, key=lambda r: _number(r[m.name]) or float("-inf"))
        text = f"{lead}, by {grain.replace('fiscal_', 'fiscal ')}: {value(first)} in {name(first)}"
        if len(complete) > 1:
            text += f", {value(last)} in {name(last)}; highest {value(peak)} in {name(peak)}"
        return text + "."
    if members and not periods:
        g = members[0]
        best = max(range(len(raw)), key=lambda i: _number(raw[i][m.name]) or float("-inf"))
        top = raw[best]
        leader = str(shown[best][g.name])
        total = sum(_number(r[m.name]) or 0.0 for r in raw)
        share = ""
        if logical.share or (m.format in ("currency", "number", "integer", "count") and total and len(raw) > 1
                             and not logical.limit):
            portion = (_number(top[m.name]) or 0.0) / total if total else 0.0
            share = f" ({portion:.0%} of the total)" if 0 < portion < 1 else ""
        count = f" across {len(raw):,} {_noun(g.label)}" if len(raw) > 1 else ""
        return f"{lead}: {leader} leads with {value(top)}{share}{count}."
    if not members and not periods:
        r = raw[0]
        others = [f"{c.label} {fmt(r[c.name], c.format)}" for c in measures[1:]]
        tail = f"; {', '.join(others)}" if others else ""
        return f"{lead}: {value(r)}{tail}."
    return f"{lead}: {len(raw):,} rows."


def _kpi(logical: Logical, cols: _Columns, raw: list[dict], formats: dict[str, str]) -> dict | None:
    measures = cols.of("measure")
    if len(raw) != 1 or cols.of("period") or cols.of("attribute") or len(measures) != 1 or logical.compare:
        return None
    m = measures[0]
    value = raw[0][m.name]
    return {"label": m.label, "value": value if _number(value) is None else _number(value),
            "format": formats[m.name], "display_format": {},
            "state": "missing" if _number(value) is None else "ready", "note": _span(logical.window)}


def _chart(logical: Logical, cols: _Columns, records: list[dict], formats: dict[str, str],
           labels: dict[str, str], display: dict[str, dict]) -> dict | None:
    measures = cols.of("measure")
    periods, members = cols.of("period"), cols.of("attribute") + cols.of("time")
    if not measures or len(records) < 2 or logical.intent == "list":
        return None
    m = measures[0]
    if logical.compare is not None and members:
        prior = next((c for c in cols.columns if c.role == "prior" and c.measure == m.measure), None)
        x, ys, kind = members[0], [c.name for c in (prior, m) if c is not None], "bar"
    elif periods:
        x, ys, kind = periods[0], [c.name for c in measures], "line"
        if members:   # one line per member
            return _pivot(logical, periods[0], members[0], m, records, formats, labels, display)
    elif members:
        x, ys = members[0], [m.name]
        kind = "pie" if logical.share and len(records) <= 6 else "bar"
    else:
        return None
    roles = {c.name: {"column": c.name, "label": labels[c.name],
                      "role": "temporal" if c.role == "period" else ("measure" if c.name in ys else "dimension"),
                      "format": "percentage" if formats.get(c.name) == "percentage" else formats.get(c.name)}
             for c in cols.columns if c.name in ys or c.name == x.name}
    rows = [{x.name: "" if r[x.name] is None else str(r[x.name]), **{y: _number(r[y]) for y in ys}} for r in records]
    return {"title": labels[m.name], "chart_type": kind, "x_key": x.name, "y_keys": ys, "rows": rows,
            "x_style": (display.get(x.name) or {}).get("style", ""),
            "column_roles": roles, "column_formats": {k: v for k, v in formats.items() if k in roles},
            "renderable_types": ["bar", "line", "area"] if kind != "pie" else ["pie", "bar"],
            "allowed_types": ["bar", "line", "area", "pie"], "recommended_type": kind,
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
    return {"title": f"{labels[m.name]} by {labels[member.name].lower()}", "chart_type": "line",
            "x_key": period.name, "y_keys": series, "rows": rows,
            "x_style": (display.get(period.name) or {}).get("style", ""), "column_roles": roles,
            "column_formats": {s: formats[m.name] for s in series}, "renderable_types": ["line", "bar", "area"],
            "allowed_types": ["line", "bar", "area"], "recommended_type": "line",
            "chart_spec": {"x": {"column": period.name, "role": "temporal"}, "column_roles": roles},
            "intent": logical.intent, "grouped_by": member.name, "grouped_measure": m.name,
            "forecast_meta": None, "chart_warnings": [] if len(totals) <= 8 else [
                f"Showing the 8 largest of {len(totals)} {labels[member.name].lower()} values."]}


def _next(logical: Logical, cols: _Columns) -> list[str]:
    """A few next questions that follow from this plan (the planner turns them into plans)."""
    measures = cols.of("measure")
    if not measures:
        return []
    m = measures[0].label.lower()
    out: list[str] = []
    if cols.of("attribute") and not cols.of("period"):
        out.append(f"{measures[0].label} by month for the top {cols.of('attribute')[0].label.lower()}")
    if cols.of("period") and not cols.of("attribute"):
        out.append(f"Break {m} down by its largest group")
    if logical.compare is None and logical.window.start and logical.window.end:
        out.append(f"Compare {m} with the previous period")
    return out[:3]
