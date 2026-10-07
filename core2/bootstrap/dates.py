"""Date roles: every date a table holds, what kind of date it is, and the default.

A table's dates are its date and timestamp columns, its keys into a calendar,
and its yyyymmdd / yyyymm numbers. Each is scored on evidence, recorded in the
role so an admin can read why:

* whether it is part of the table's grain (a snapshot's period);
* how complete it is (empty and placeholder values do not count);
* whether its values cluster at load times, many rows sharing one timestamp,
  which is what a row-write stamp looks like, and what its name says
  (loaded, updated, modified...): an audit stamp is never a default;
* whether its name says planned, requested, promised or due;
* whether it names the event the table records ("invoice" date on an invoice
  table; the table name's last word is its head noun);
* whether the table is a periodic snapshot (every period holds about the same
  number of rows, or all dates are month ends).

The default is the best business date. When the top two are close it is still
chosen, and the close call is offered for review.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from sqlglot import exp

from core2.bootstrap import names
from core2.bootstrap.calendar import CalendarFinding
from core2.bootstrap.inventory import Inventory
from core2.bootstrap.joins import JoinFinding
from core2.bootstrap.keys import TableKeys, _duplicates
from core2.bootstrap.profiler import TableProfile
from core2.model.schema import Evidence
from core2.warehouse import dialect as D
from core2.warehouse.runner import Warehouse

AUDIT_WORDS = _AUDIT = {"loaded", "load", "ld", "etl", "updated", "upd", "update", "modified", "mod", "inserted", "ins",
          "entered",
          "batch", "sync", "synced", "ingest", "ingested", "extract", "extracted", "audit", "written", "refresh",
          "refreshed", "processed", "staged", "stg"}
_PLANNED = {"planned", "plan", "pln", "promised", "requested", "rqs", "req", "expected", "exp", "target",
            "scheduled", "sched", "forecast", "estimated", "est"}
_DUE = {"due", "deadline", "expiry", "expires", "expiration", "maturity"}
_CANCEL = {"cancel", "cancelled", "canceled", "cnl", "void", "voided", "reversed", "rejected"}
MARGIN = 0.1


@dataclass
class DateCandidate:
    table: str
    column: str
    via_calendar: JoinFinding | None
    granularity: str                # day | month | year | timestamp
    kind: str = "event"
    score: float = 0.0
    coverage: float = 1.0
    placeholder_share: float = 0.0
    whole_year_share: float = 0.0
    periodic: bool = False
    first: dt.date | None = None
    last: dt.date | None = None
    is_default: bool = False
    evidence: list[Evidence] = field(default_factory=list)

    def add(self, kind: str, weight: float, detail: str, **data: object) -> None:
        self.score += weight
        self.evidence.append(Evidence(kind=kind, weight=weight, detail=detail, data=dict(data)))


def _parse_day(value: object, granularity: str) -> dt.date | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    text = str(value).strip()
    try:
        if granularity == "month" and len(text) == 6 and text.isdigit():
            return dt.date(int(text[:4]), int(text[4:]), 1)
        if len(text) == 8 and text.isdigit():
            return dt.date(int(text[:4]), int(text[4:6]), int(text[6:]))
        return dt.date.fromisoformat(text[:10])
    except ValueError:
        return None


def _periodic(warehouse: Warehouse, inventory: Inventory, table: str, column: str, granularity: str,
              rows: int, entity_columns: list[str]) -> tuple[bool, str]:
    """Is ``column`` the period of a snapshot: one row per thing per period, things recurring?

    The things a row is about are found by adding the table's links one at a
    time, the link to the largest table first, until (period + links)
    identifies every row. A snapshot then holds each thing about once per
    period; an event table's month-end or evenly spread dates do not make it a
    snapshot, because its rows are not one per thing and period.
    """
    if not entity_columns:
        return False, ""
    t = inventory.tables[table]
    d = warehouse.dialect
    source = D.table_sql(t.database, t.schema, t.name, d)
    c = exp.column(D.ident(column, d))
    periods_q = exp.select(exp.Count(this=exp.Distinct(expressions=[c.copy()]))).from_(exp.to_table("__SRC__"))
    periods = int(warehouse.query(periods_q.sql(dialect=d).replace("__SRC__", source)).rows[0][0] or 0)
    if periods < 3:
        return False, ""
    chosen: list[str] = []
    for link in entity_columns[:6]:
        chosen.append(link)
        if not _duplicates(warehouse, t, [column, *chosen]):
            break
    else:
        return False, ""   # several rows per thing and period: events, not a snapshot
    things_q = exp.select(exp.Count(this=exp.Literal.number(1))).from_(
        exp.select(*[exp.column(D.ident(e, d)) for e in chosen]).distinct().from_(exp.to_table("__SRC__")).subquery("e"))
    things = int(warehouse.query(things_q.sql(dialect=d).replace("__SRC__", source)).rows[0][0] or 0)
    recurrence = rows / things if things else 0.0
    if recurrence >= max(3.0, 0.5 * periods):
        return True, (f"each of {things:,} combinations of {', '.join(chosen)} recurs in about "
                      f"{recurrence:.0f} of {periods} periods")
    # A short history cannot show things recurring; the names can still say the
    # table holds balances (stock on hand, a balance table), one row per thing
    # and period.
    words = set(names.tokens(t.name)) | {w for col in t.columns for w in names.tokens(col.name)}
    if words & names.LEVEL_WORDS and not names.opaque(t.name):
        return True, (f"one row per {', '.join(chosen)} and period, and its names say it holds balances "
                      f"({', '.join(sorted(words & names.LEVEL_WORDS)[:3])})")
    return False, ""


def find_date_roles(warehouse: Warehouse, inventory: Inventory, profiles: dict[str, TableProfile],
                    keys: dict[str, TableKeys], calendars: dict[str, CalendarFinding],
                    joins: list[JoinFinding]) -> dict[str, list[DateCandidate]]:
    """Date roles per table key, the default marked."""
    to_calendar = {(j.from_table, j.from_column): j for j in joins if j.to_calendar and j.trust != "rejected"}
    # What a snapshot row is about: its links, the largest tables first.
    entity_columns: dict[str, list[str]] = {}
    for j in sorted(joins, key=lambda j: (-profiles[j.to_table].rows, j.from_column)):
        if not j.to_calendar and j.trust != "rejected":
            entity_columns.setdefault(j.from_table, []).append(j.from_column)
    out: dict[str, list[DateCandidate]] = {}
    for key, table in inventory.tables.items():
        if key in calendars:
            continue
        profile = profiles[key]
        candidates: list[DateCandidate] = []
        for column in table.columns:
            p = profile.columns[column.name]
            join = to_calendar.get((key, column.name))
            if join:
                granularity = "month" if calendars[join.to_table].grain == "month" else "day"
            elif column.data_type == "date":
                granularity = "day"
            elif column.data_type == "timestamp":
                granularity = "timestamp"
            elif p.pattern == "yyyymmdd":
                granularity = "day"
            elif p.pattern == "yyyymm":
                granularity = "month"
            else:
                continue
            c = DateCandidate(table=key, column=column.name, via_calendar=join, granularity=granularity)
            filled = p.non_null - (join.placeholder_rows if join else 0)
            c.coverage = filled / profile.rows if profile.rows else 0.0
            c.placeholder_share = (join.placeholder_rows / profile.rows) if join and profile.rows else 0.0
            c.whole_year_share = (p.whole_year_rows or 0) / profile.rows if profile.rows else 0.0
            low, high = (p.date_min, p.date_max) if p.date_min else (p.min, p.max)
            c.first, c.last = _parse_day(low, granularity), _parse_day(high, granularity)
            if c.first and c.first.year <= 1901:
                c.first = None   # a sentinel, not a date
            candidates.append(c)
        if not candidates:
            continue

        table_words = names.core_table(table.name)
        head = table_words[-1] if table_words else ""
        grain = set(keys[key].primary_key) if len(keys[key].primary_key) > 1 else set()
        for c in candidates:
            p = profile.columns[c.column]
            words = [w for w in names.core_column(c.column) if w not in ("date", "dt", "day", "time", "ts", "timestamp",
                                                                         "key", "period", "per", "month")]
            readable = (c.via_calendar.role if c.via_calendar and c.via_calendar.role else names.readable(c.column))
            c.add("base", 0.5, "a date of this table")
            if c.coverage < 0.999:
                c.add("coverage", -0.4 * (1 - c.coverage), f"{c.coverage:.0%} of rows have a real date")
            if c.column in grain:
                c.add("grain", 0.3, f"{readable} is part of what identifies a row")
            if c.granularity == "timestamp" and p.distinct and profile.rows / p.distinct >= 50 and (p.time_share or 0) > 0:
                c.kind = "audit"
                c.add("load_clustering", -1.0, f"only {p.distinct:,} distinct timestamps across {profile.rows:,} rows: "
                      "rows are stamped when they are loaded")
            if p.distinct == 1:
                c.add("constant", -0.6, f"{readable} has the same value on every row")
            word_set = set(names.tokens(c.column))
            if word_set & _AUDIT and not names.opaque(c.column):
                c.kind = "audit"
                c.add("audit_name", -1.0, f"the name ({readable}) says when the row was written")
            elif word_set & _DUE:
                c.kind = "due"
                c.add("due_name", -0.25, f"the name ({readable}) says it is a due date")
            elif word_set & _PLANNED:
                c.kind = "planned"
                c.add("planned_name", -0.25, f"the name ({readable}) says it is planned or requested")
            elif word_set & _CANCEL:
                c.add("cancel_name", -0.3, f"the name ({readable}) says it applies only to cancelled rows")
            meaningful = not names.opaque(c.column) and not names.opaque(table.name)
            if meaningful and words and table_words and any(names.same_word(w, t) for w in words for t in table_words):
                c.add("names_event", 0.25, f"{readable} is the event the table records")
                if head and any(names.same_word(w, head) for w in words):
                    c.add("names_head", 0.1, f"it names the table's main subject ({head})")
            if c.kind == "event" and c.granularity in ("day", "month"):   # a planned or due date is never a snapshot
                periodic, why = _periodic(warehouse, inventory, key, c.column, c.granularity, profile.rows,
                                          entity_columns.get(key, [])) \
                    if profile.rows >= 50 and p.distinct and profile.rows / p.distinct >= 5 else (False, "")
                if periodic:
                    c.periodic = True
                    c.kind = "snapshot"
                    c.add("snapshot", 0.35, why)

        _written_later(warehouse, inventory, key, candidates)
        _load_times(warehouse, inventory, key, candidates)
        business = [c for c in candidates if c.kind in ("event", "snapshot")]
        ranked = sorted(business or [], key=lambda c: (-c.score, [x.name for x in table.columns].index(c.column)))
        if ranked:
            ranked[0].is_default = True
        out[key] = candidates
    return out


def _months(value: exp.Expression, candidate: DateCandidate, data_type: str, dialect: str) -> exp.Expression:
    """A date as a month count (year * 12 + month): from a yyyymm or yyyymmdd number, or a date."""
    if candidate.granularity == "month":
        return D.add(D.mul(exp.Floor(this=D.div(value, 100)), 12), D.mod(value.copy(), 100))
    if data_type in ("integer", "decimal"):
        return D.add(D.mul(exp.Floor(this=D.div(value, 10000)), 12),
                     D.mod(exp.Floor(this=D.div(value.copy(), 100)), 100))
    return D.add(D.mul(D.date_part(value, "year", dialect), 12), D.date_part(value.copy(), "month", dialect))


def _written_later(warehouse: Warehouse, inventory: Inventory, key: str, candidates: list[DateCandidate]) -> None:
    """A timestamp that trails the table's period or business date by months is when rows were written."""
    anchors = sorted((c for c in candidates if c.kind in ("event", "snapshot") and c.granularity in ("day", "month")
                      and (c.periodic or c.granularity == "month")), key=lambda c: -c.score)
    stamps = [c for c in candidates if c.granularity == "timestamp" and c.kind != "audit"]
    if not anchors or not stamps:
        return
    anchor = anchors[0]
    if anchor.granularity == "day" and anchor.via_calendar is not None:
        return   # a calendar key would need the calendar's date: the period or a real date column is enough
    d = warehouse.dialect
    t = inventory.tables[key]
    a = exp.column(D.ident(anchor.column, d))
    anchor_months = _months(a, anchor, t.type_of(anchor.column), d)
    valid = exp.GT(this=D.mod(a.copy(), 100), expression=exp.Literal.number(0)) if anchor.granularity == "month" \
        else exp.not_(exp.Is(this=a.copy(), expression=exp.Null()))
    for stamp in stamps:
        s = exp.column(D.ident(stamp.column, d))
        lag = D.sub(D.add(D.mul(D.date_part(s, "year", d), 12), D.date_part(s.copy(), "month", d)), anchor_months.copy())
        late = exp.Sum(this=exp.Case(ifs=[exp.If(this=exp.GTE(this=lag, expression=exp.Literal.number(2)),
                                                  true=exp.Literal.number(1))], default=exp.Literal.number(0)))
        query = exp.select(exp.Count(this=exp.Literal.number(1)).as_("n"), late.as_("late")).from_(
            exp.to_table("__SRC__")).where(exp.and_(valid.copy(), exp.not_(exp.Is(this=s.copy(), expression=exp.Null()))))
        n, late_rows = warehouse.query(query.sql(dialect=d).replace(
            "__SRC__", D.table_sql(t.database, t.schema, t.name, d), 1)).rows[0]
        if n and int(late_rows or 0) / int(n) >= 0.3:
            stamp.kind = "audit"
            stamp.add("written_later", -1.0, f"{int(late_rows) / int(n):.0%} of rows were written two months or more "
                      f"after the {names.readable(anchor.column).lower()} they belong to: it records when rows were "
                      "written, not a business date")


def _load_times(warehouse: Warehouse, inventory: Inventory, key: str, candidates: list[DateCandidate]) -> None:
    """A timestamp written at the same few times of day, beside another date, is when a batch loaded the rows.

    Business events happen around the clock or across working hours; a nightly load
    stamps every row at 03:00. A table's only date stays a business date whatever
    its times (a shift that always starts at 06:00 is still when the shift started).
    """
    if not any(c.kind in ("event", "snapshot") and c.granularity != "timestamp" for c in candidates):
        return
    d = warehouse.dialect
    t = inventory.tables[key]
    for stamp in candidates:
        if stamp.granularity != "timestamp":
            continue                # an audit stamp by its name is still checked: the evidence is shown
        s = exp.column(D.ident(stamp.column, d))
        minute_of_day = D.add(D.mul(D.date_part(s, "hour", d), 60), D.date_part(s.copy(), "minute", d))
        query = exp.select(exp.Count(this=exp.Literal.number(1)).as_("n"),
                           exp.Count(this=exp.Distinct(expressions=[minute_of_day])).as_("times")).from_(
            exp.to_table("__SRC__")).where(exp.not_(exp.Is(this=s.copy(), expression=exp.Null())))
        n, times = warehouse.query(query.sql(dialect=d).replace(
            "__SRC__", D.table_sql(t.database, t.schema, t.name, d), 1)).rows[0]
        if int(n or 0) >= 200 and 0 < int(times or 0) <= 3:
            stamp.add("load_clustering", -1.0 if stamp.kind != "audit" else 0.0,
                      f"all {int(n):,} rows were written at {int(times)} time"
                      f"{'s' if int(times) > 1 else ''} of day: rows are stamped when a batch loads them")
            stamp.kind = "audit"


def close_call(candidates: list[DateCandidate]) -> DateCandidate | None:
    """The runner-up when it is within the margin of the default."""
    business = sorted((c for c in candidates if c.kind in ("event", "snapshot")), key=lambda c: -c.score)
    if len(business) >= 2 and business[0].score - business[1].score < MARGIN:
        return business[1]
    return None
