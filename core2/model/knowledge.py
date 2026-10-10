"""What each table and field means, and which dates each table is counted by, for the admin to see and change.

The Knowledge base page holds a table's "what one row is" and, per field, its name, what it
means, the words people also use, the names its codes go by ("C" is Cancelled) and whether
it is shown, hidden or sensitive. The Dates page holds each table's dates: the default one,
their names and words, a date the build did not find (checked against the data first) and
the fiscal year. Each change is saved as an admin decision (core2.model.overrides), so it
holds from the next answer and through the next Learn.

The checks run read-only queries written through the compiler's own helpers: aggregates
only, no row leaves the warehouse.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any

from sqlglot import exp

from core2 import ids
from core2.model.schema import Column, DateRole, Join, SemanticModel
from core2.model.view import KIND_WORDS
from core2.warehouse import dialect as D

MAX_NAMED = 30            # a field whose values can be named has at most this many
SHOWN = ("shown", "hidden", "sensitive")
PERSONAL = ("none", "name", "detail")       # not people's data; a person's name; a personal detail
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December")
_KIND_ORDER = {"fact": 0, "snapshot": 1, "dimension": 2, "bridge": 3, "calendar": 4, "other": 5}
_GROUP_WORDS = {**KIND_WORDS, "calendar": "Dates"}
_DATE_KIND_WORDS = {"event": "", "snapshot": "the period a row is for", "planned": "planned or requested",
                    "due": "a due date", "validity": "when a row was valid",
                    "audit": "when the row was written: never a default"}
_NOT_DATES = ("measure", "key", "foreign_key", "label", "flag", "text")


class KnowledgeError(ValueError):
    """Why a change cannot be saved, in words."""


def _words(synonyms: dict[str, list[str]]) -> list[str]:
    return list(dict.fromkeys(w for ws in synonyms.values() for w in ws))


def _clean(text: Any, limit: int) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()[:limit]


def clean_words(values: Any) -> list[str]:
    """The words people also use, as the page sends them: trimmed, each once, at most twelve."""
    items = values if isinstance(values, list) else []
    return list(dict.fromkeys(w for w in (_clean(s, 40) for s in items) if w))[:12]


def shown(column: Column) -> str:
    if column.sensitivity != "none":
        return "sensitive"
    return "hidden" if column.hidden else "shown"


# ── the Knowledge base page ──────────────────────────────────────────────────

def tables(model: SemanticModel) -> list[dict[str, Any]]:
    """Every table, grouped by kind, with how many fields it has and how many still need a meaning."""
    out = []
    for t in sorted(model.tables.values(), key=lambda t: (_KIND_ORDER.get(t.kind, 9), t.business_name.lower())):
        columns = [model.columns[k] for k in t.columns if k in model.columns]
        out.append({"key": t.key, "name": t.business_name or t.name, "physical": t.name, "kind": t.kind,
                    "group": _GROUP_WORDS.get(t.kind, "Other tables"), "fields": len(columns),
                    "unexplained": sum(1 for c in columns if not c.description and not c.hidden),
                    "hidden": t.hidden})
    return out


def namable(column: Column) -> list[dict[str, Any]]:
    """The values of a field whose codes can be named: text, few values, all of them known, and shown."""
    p = column.profile
    if column.data_type != "text" or column.sensitivity != "none" or not column.values_allowed or p is None \
            or not p.top or p.distinct > MAX_NAMED or p.distinct > len(p.top):
        return []
    values = sorted((t for t in p.top if t.value is not None), key=lambda t: str(t.value))
    return [{"code": str(t.value), "name": column.value_names.get(str(t.value), ""), "rows": t.count} for t in values]


def table_view(model: SemanticModel, table_key: str) -> dict[str, Any]:
    t = model.tables[table_key]
    dated = {r.column for r in model.date_roles.values()}
    fields = []
    for key in t.columns:
        column = model.columns.get(key)
        if column is None:
            continue
        values = namable(column)
        fields.append({
            "key": column.key, "name": column.business_name or column.name, "physical": column.name,
            "description": column.description, "synonyms": _words(column.synonyms), "shown": shown(column),
            "personal": column.personal,
            "values": values, "namable": bool(values),
            "kind": "date" if column.key in dated or column.data_type in ("date", "timestamp") else column.data_type,
        })
    return {"key": t.key, "name": t.business_name or t.name, "physical": t.name, "rows": t.row_count,
            "kind": t.kind, "kind_words": _GROUP_WORDS.get(t.kind, "Other tables").lower(), "grain": t.grain_text,
            "fields": fields}


def field_changes(model: SemanticModel, column_key: str, data: dict[str, Any]) -> dict[str, Any]:
    """The decisions a field edit makes: only what differs from the field as it is now."""
    column = model.columns.get(column_key)
    if column is None:
        raise KnowledgeError("This field is not in the data any more.")
    changes: dict[str, Any] = {}
    if "name" in data:
        name = _clean(data.get("name"), 60)
        if not name:
            raise KnowledgeError("Give the field a name.")
        if name != column.business_name:
            changes["business_name"] = name
    if "description" in data:
        description = _clean(data.get("description"), 300)
        if description != column.description:
            changes["description"] = description
    if "synonyms" in data:
        words = clean_words(data.get("synonyms"))
        if words != _words(column.synonyms):
            changes["synonyms"] = {"en": words} if words else {}
    if "shown" in data:
        choice = str(data.get("shown") or "")
        if choice not in SHOWN:
            raise KnowledgeError("Choose Shown, Hidden or Sensitive.")
        if choice != shown(column):
            changes["hidden"] = choice == "hidden"
            changes["sensitivity"] = "confidential" if choice == "sensitive" else "none"
    if "personal" in data:
        reading = str(data.get("personal") or "")
        if reading not in PERSONAL:
            raise KnowledgeError("Choose whether it holds people's names, their details, or neither.")
        if reading != column.personal:
            changes["personal"] = reading
    if "value_names" in data:
        codes = {v["code"] for v in namable(column)}
        given = data.get("value_names") or {}
        if not isinstance(given, dict):
            raise KnowledgeError("Name the values as code and name.")
        names = {str(code): _clean(name, 60) for code, name in given.items() if _clean(name, 60)}
        stray = sorted(set(names) - codes)
        if stray:
            raise KnowledgeError(f"{', '.join(stray[:3])} is not one of this field's values.")
        if len(set(n.casefold() for n in names.values())) < len(names):
            raise KnowledgeError("Two values have the same name: readers could not tell them apart.")
        if names != column.value_names:
            changes["value_names"] = names
    return changes


def grain_change(model: SemanticModel, table_key: str, text: Any) -> str | None:
    """The table's new "what one row is", or None when it is unchanged."""
    if table_key not in model.tables:
        raise KnowledgeError("This table is not in the data any more.")
    grain = _clean(text, 200)
    if not grain:
        raise KnowledgeError("Say what one row is, for example \"one product on one customer order\".")
    return None if grain == model.tables[table_key].grain_text else grain


# ── the Dates page ───────────────────────────────────────────────────────────

def _month(day: dt.date | None) -> str:
    return f"{day:%b %Y}" if day else ""


def _coverage(role: DateRole, table_rows: int) -> str:
    if role.granularity == "month" and role.kind == "snapshot":
        return "one row per period"
    return f"{role.coverage:.0%} of rows" if table_rows else ""


def candidates(model: SemanticModel, table_key: str) -> list[dict[str, Any]]:
    """Columns of a table that could be a date the build did not find: date-typed, or numbers or codes that
    may key the calendar or read as year-month-day."""
    dated = {r.column for r in model.date_roles.values()}
    linked = {c for j in model.joins.values() if not j.to_calendar and j.trust != "rejected" for c in j.from_columns}
    text_keys = any(model.columns[c.key_column].data_type == "text" for c in model.calendars.values()
                    if c.key_column in model.columns)
    out = []
    for key in model.tables[table_key].columns:
        column = model.columns.get(key)
        if column is None or key in dated or key in linked or column.role in _NOT_DATES or column.hidden:
            continue
        if column.data_type in ("date", "timestamp") or (column.data_type == "text" and text_keys) or (
                column.data_type in ("integer", "decimal") and _date_shaped(column)):
            out.append({"key": key, "physical": column.name, "name": column.business_name or column.name})
    return out


def _date_shaped(column: Column) -> bool:
    """A number that may be a date: read as year-month(-day), or as large as one (a line number is not)."""
    p = column.profile
    return p is None or p.pattern in ("yyyymmdd", "yyyymm") or (p.max_num or 0) >= 190001


def dates_view(model: SemanticModel) -> dict[str, Any]:
    out = []
    for t in sorted(model.tables.values(), key=lambda t: (_KIND_ORDER.get(t.kind, 9), t.business_name.lower())):
        if t.kind == "calendar" or t.hidden:
            continue
        roles = sorted((r for r in model.date_roles.values() if r.table == t.key),
                       key=lambda r: (not r.is_default, r.kind == "audit", -r.coverage, r.name.lower()))
        if not roles and t.kind not in ("fact", "snapshot"):
            continue
        monthly = bool(roles) and all(r.granularity == "month" for r in roles if r.kind != "audit")
        out.append({
            "key": t.key, "name": t.business_name or t.name, "monthly": monthly,
            "roles": [{"key": r.key, "name": r.name, "physical": model.columns[r.column].name,
                       "synonyms": _words(r.synonyms), "default": r.is_default, "kind": r.kind,
                       "kind_words": _DATE_KIND_WORDS.get(r.kind, ""), "may_default": r.kind != "audit",
                       "coverage": _coverage(r, t.row_count), "first": _month(r.first), "last": _month(r.last)}
                      for r in roles],
            "candidates": candidates(model, t.key),
        })
    fiscal = model.settings.fiscal_year_start_month
    calendars = []
    for c in model.calendars.values():
        table = model.tables.get(c.table)
        days = ((c.last_date - c.first_date).days + 1) if c.first_date and c.last_date else 0
        calendars.append({"name": table.business_name if table else c.table, "physical": table.name if table else "",
                          "grain": c.grain, "days": days, "first": c.first_date.year if c.first_date else None,
                          "last": c.last_date.year if c.last_date else None,
                          "week_start": model.settings.week_start.title()})
    return {"tables": out, "fiscal": fiscal or 1, "calendars": calendars, "months": MONTHS}


def fiscal_words(month: int, today: dt.date) -> str:
    """How a fiscal year reads with this start: "FY2026 runs from April 2025 to March 2026"."""
    if month == 1:
        return "The fiscal year is the calendar year."
    year = today.year + (1 if today.month >= month else 0)
    return (f"FY{year} runs from {MONTHS[month - 1]} {year - 1} to {MONTHS[(month - 2) % 12]} {year}. "
            "Answers say “fiscal” when they use it.")


def date_changes(model: SemanticModel, role_key: str, data: dict[str, Any]) -> dict[str, Any]:
    role = model.date_roles.get(role_key)
    if role is None:
        raise KnowledgeError("This date is not in the data any more.")
    changes: dict[str, Any] = {}
    if "name" in data:
        name = _clean(data.get("name"), 60)
        if not name:
            raise KnowledgeError("Give the date a name.")
        if name != role.name:
            changes["name"] = name
    if "synonyms" in data:
        words = clean_words(data.get("synonyms"))
        if words != _words(role.synonyms):
            changes["synonyms"] = {"en": words} if words else {}
    return changes


def _day(value: object, granularity: str) -> dt.date | None:
    from core2.bootstrap.dates import _parse_day

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = str(int(value))
    return _parse_day(value, granularity)


def _calendar_for(model: SemanticModel, column: Column):
    """The calendar a column may key, by the family of its key's type."""
    family = "text" if column.data_type == "text" else "number" if column.data_type in ("integer", "decimal") else ""
    for c in model.calendars.values():
        key = model.columns.get(c.key_column or "")
        if key is not None and family and ("text" if key.data_type == "text" else "number") == family:
            return c
    return None


def check_date(model: SemanticModel, column_key: str, warehouse: Any) -> dict[str, Any]:
    """Whether a column can be a date: it keys the calendar, is a date, or reads as year-month-day numbers;
    how many rows have one and the first and last."""
    from core2.model import links

    column = model.columns.get(column_key)
    if column is None or column.key not in {c["key"] for c in candidates(model, column.table)}:
        raise KnowledgeError("Choose a column of this table that is not a date yet.")
    d = warehouse.dialect
    table = model.tables[column.table]
    col = links._col(model, "t", column.key, d)
    out: dict[str, Any] = {"ok": False, "column": column.key, "words": "", "problem": ""}
    try:
        calendar = _calendar_for(model, column)
        if calendar is not None:
            spec = links.LinkSpec(column.table, calendar.table, [(column.key, calendar.key_column or "")])
            checked = links.check_link(model, spec, warehouse)
            if checked.get("problem"):
                out["problem"] = checked["problem"]
                return out
            if checked.get("with_key") and checked["match"] >= 0.5 and not checked.get("twice"):
                key = links._col(model, "c", calendar.key_column or "", d)
                shown = links._col(model, "c", calendar.date_column, d) if calendar.date_column else key
                q = exp.select(exp.Count(this=exp.Star()).as_(D.ident("n", d)),
                               exp.Min(this=shown.copy()).as_(D.ident("lo", d)),
                               exp.Max(this=shown.copy()).as_(D.ident("hi", d))).from_(
                    links._table(model, column.table, "t", d)).join(
                    links._table(model, calendar.table, "c", d), on=exp.EQ(this=col, expression=key), join_type="inner")
                if calendar.placeholders:
                    q = q.where(exp.not_(exp.In(this=key.copy(), expressions=[
                        exp.Literal.string(str(v)) if isinstance(v, str) else exp.Literal.number(v)
                        for v in calendar.placeholders])))
                row = warehouse.query(D.render(q, d), max_rows=2).rows[0]
                grain = "month" if calendar.grain == "month" else "day"
                rows = checked["rows"] or table.row_count or 0
                out.update(ok=True, mode="calendar", calendar=calendar.table, granularity=grain, rows=rows,
                           coverage=round(int(row[0] or 0) / rows, 4) if rows else 0.0, match=checked["match"],
                           first=_day(row[1], grain), last=_day(row[2], grain), checked=checked)
            elif column.data_type == "text":
                out["problem"] = (f"Only {checked['match']:.0%} of its values are days of the calendar: "
                                  "it does not read as a date.")
                return out
        if not out["ok"] and column.data_type in ("date", "timestamp"):
            q = exp.select(exp.Count(this=exp.Star()).as_(D.ident("n", d)),
                           exp.Count(this=col.copy()).as_(D.ident("filled", d)),
                           exp.Min(this=col.copy()).as_(D.ident("lo", d)),
                           exp.Max(this=col.copy()).as_(D.ident("hi", d))).from_(
                links._table(model, column.table, "t", d))
            row = warehouse.query(D.render(q, d), max_rows=2).rows[0]
            rows = int(row[0] or 0)
            grain = "day" if column.data_type == "date" else "timestamp"
            out.update(ok=True, mode=column.data_type, granularity=grain, rows=rows,
                       coverage=round(int(row[1] or 0) / rows, 4) if rows else 0.0,
                       first=_day(row[2], "day"), last=_day(row[3], "day"))
        elif not out["ok"] and column.data_type in ("integer", "decimal"):
            q = exp.select(exp.Count(this=exp.Star()).as_(D.ident("n", d)), exp.Max(this=col.copy()).as_(
                D.ident("top", d))).from_(links._table(model, column.table, "t", d))
            rows, top = warehouse.query(D.render(q, d), max_rows=2).rows[0]
            low, high, grain = (19010101, 20991231, "day") if (top or 0) >= 10_000_000 else (190101, 209912, "month")
            inside = exp.Between(this=col.copy(), low=exp.Literal.number(low), high=exp.Literal.number(high))
            q = exp.select(links._count_when(inside).as_(D.ident("filled", d)),
                           exp.Min(this=exp.Case(ifs=[exp.If(this=inside.copy(), true=col.copy())])).as_(
                               D.ident("lo", d)),
                           exp.Max(this=exp.Case(ifs=[exp.If(this=inside.copy(), true=col.copy())])).as_(
                               D.ident("hi", d))).from_(links._table(model, column.table, "t", d))
            filled, lo, hi = warehouse.query(D.render(q, d), max_rows=2).rows[0]
            first, last = _day(lo, grain), _day(hi, grain)
            rows = int(rows or 0)
            if first is None or last is None or not filled:
                out["problem"] = ("Its values neither join the calendar nor read as year-month-day numbers: "
                                  "it is not a date QueryBot can count by.")
                return out
            out.update(ok=True, mode="yyyymmdd" if grain == "day" else "yyyymm", granularity=grain, rows=rows,
                       coverage=round(int(filled) / rows, 4) if rows else 0.0, first=first, last=last)
    except Exception as exc:  # noqa: BLE001 - the warehouse's refusal is the admin's to read
        out["problem"] = f"The data refused the check: {str(exc)[:300]}"
        return out
    if not out["ok"]:
        out["problem"] = "It is not a date QueryBot can count by."
        return out
    said = {"calendar": "Joins the calendar", "date": "A date", "timestamp": "A date and time",
            "yyyymmdd": "Reads as year-month-day numbers", "yyyymm": "Reads as year-month numbers"}[out["mode"]]
    span = f" · values from {_month(out['first'])} to {_month(out['last'])}" if out["first"] and out["last"] else ""
    out["words"] = f"{said} · {out['coverage']:.0%} of rows have one{span}"
    return out


def new_date(model: SemanticModel, column_key: str, name: Any, checked: dict[str, Any]) -> tuple[DateRole, Join | None]:
    """The date an admin adds, from its check; with the link to the calendar when it keys one and none exists."""
    from core2.model import links

    column = model.columns[column_key]
    name = _clean(name, 60)
    if not name:
        raise KnowledgeError("Give the date a name.")
    if not checked.get("ok"):
        raise KnowledgeError(checked.get("problem") or "Check the date first.")
    join: Join | None = None
    join_key = None
    if checked["mode"] == "calendar":
        calendar = model.calendars[checked["calendar"]]
        spec = links.LinkSpec(column.table, calendar.table, [(column.key, calendar.key_column or "")])
        existing = next((j for j in model.joins.values() if links.same_link(j, spec)), None)
        if existing is not None:
            join_key = existing.key
        else:
            join = links.as_join(model, spec, checked.get("checked"))
            join_key = join.key
    role = DateRole(key=column.key, table=column.table, column=column.key, name=name, slug=ids.slug(name),
                    calendar=checked.get("calendar"), calendar_join=join_key,
                    granularity=checked["granularity"], kind="event", coverage=checked["coverage"],
                    first=checked.get("first"), last=checked.get("last"), provenance="admin", status="approved",
                    confidence=1.0)
    return role, join
