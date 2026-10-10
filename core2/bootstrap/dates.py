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

from functools import partial

import datetime as dt
from dataclasses import dataclass, field

from sqlglot import exp

from core2.bootstrap import names
from core2.bootstrap.calendar import CalendarFinding
from core2.bootstrap.inventory import Inventory, InvTable
from core2.bootstrap.joins import JoinFinding
from core2.bootstrap.keys import TableKeys, _duplicates
from core2.bootstrap.profiler import TableProfile
from core2.model.schema import Evidence
from core2.bootstrap.journal import attempt
from core2.warehouse import dialect as D
from core2.warehouse.runner import Warehouse

AUDIT_WORDS = _AUDIT = {"loaded", "load", "ld", "etl", "updated", "upd", "updt", "update", "modified", "mod", "inserted",
          "ins", "insrt", "entered",
          "batch", "sync", "synced", "ingest", "ingested", "extract", "extracted", "audit", "written", "refresh",
          "refreshed", "processed", "staged", "stg"}
# Words that stamp a row only on a timestamp: a prescription's WRITTEN_DATE is when the doctor wrote it.
_STAMP_ONLY = {"written", "entered", "processed", "batch"}
# A row written moments after the event it records: its stamp, when the name says it was made then.
_MADE = {"created", "crt", "crtd", "inserted", "insrt", "recorded", "logged", "captured"}
_PLANNED = {"planned", "plan", "pln", "promised", "prms", "requested", "rqs", "req", "expected", "exp", "target",
            "scheduled", "sched", "schd", "forecast", "estimated", "est", "estm"}
_DUE = {"due", "deadline", "maturity", "required", "rqrd", "needed", "need"}
# The end of a row's validity (an expiry, a beyond-use date), or either end of a contract's term.
_VALIDITY = {"valid", "validity", "effective", "eff", "expiry", "expires", "expiration", "expire", "expr", "exprt",
             "beyond", "bynd", "until", "thru"}
_TERM = {"contract", "cntrct", "cntr", "coverage", "policy", "lease", "warranty", "license", "licence", "membership",
         "agreement", "term"}
# The date the books count by: a journal is dated by its posting, though its document is dated first.
_POSTING = {"posting", "posted", "post", "pst", "booked", "booking", "accounting", "acctg", "gl", "ledger"}
_TERM_ENDS = {"start", "strt", "begin", "from", "end", "to", "finish", "stop"}
_STARTS = {"start", "strt", "begin", "from", "effective", "eff"}
# Steps a row goes through after the event it records: never the row's own date when an earlier one exists.
_PROGRESS = {"acknowledged", "ack", "ackd", "acknowledgement", "resolved", "rslv", "cleared", "clr", "closed",
             "completed", "cmpl", "done", "finished", "paid", "settled", "shipped", "shp", "delivered", "dlv",
             "approved", "apprvd", "fulfilled", "received", "rcv", "rcvd", "responded", "escalated"}
_CANCEL = {"cancel", "cancelled", "canceled", "cnl", "void", "voided", "reversed", "rejected"}
_PERSONAL = {"birth", "brth", "bth", "dob", "born", "birthday"}
# The first, last or original time something happened to what a row describes (a customer's first
# invoice, an item's last receipt): a fact about that thing, repeated on its rows, not the row's event.
_MILESTONE = {"first", "fst", "frst", "last", "lst", "latest", "earliest", "original", "orig", "initial", "init",
              "previous", "prev", "prior", "next", "nxt"}
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
              rows: int, entity_columns: list[str], key_rest: list[str] | None = None) -> tuple[bool, str]:
    """Is ``column`` the period of a snapshot: one row per thing per period, things recurring?

    The things a row is about are found by adding the table's links one at a
    time, the link to the largest table first, until (period + links)
    identifies every row. A snapshot then holds each thing about once per
    period; an event table's month-end or evenly spread dates do not make it a
    snapshot, because its rows are not one per thing and period.
    """
    if not entity_columns and not key_rest:
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
        # The links alone do not tell a row's thing (stock per ingredient and lot): the rest of the
        # table's own key, beside the period, may.
        if not key_rest or _duplicates(warehouse, t, [column, *key_rest]):
            return False, ""   # several rows per thing and period: events, not a snapshot
        chosen = list(key_rest)
    things_q = exp.select(exp.Count(this=exp.Literal.number(1))).from_(
        exp.select(*[exp.column(D.ident(e, d)) for e in chosen]).distinct().from_(exp.to_table("__SRC__")).subquery("e"))
    things = int(warehouse.query(things_q.sql(dialect=d).replace("__SRC__", source)).rows[0][0] or 0)
    recurrence = rows / things if things else 0.0
    # A growing base (subscriptions that start each month) holds each thing in fewer than half the
    # periods; a third still tells a snapshot from events, which seldom repeat per thing and period.
    if recurrence >= max(3.0, 0.3 * periods):
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
            placeholders = join.placeholder_rows if join else (p.placeholder_rows or 0)
            filled = p.non_null - placeholders
            c.coverage = filled / profile.rows if profile.rows else 0.0
            c.placeholder_share = placeholders / profile.rows if profile.rows else 0.0
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
                if c.granularity == "timestamp" and profile.rows >= 2 and (p.time_share or 0) > 0:
                    c.kind = "audit"
                    c.add("one_stamp", -0.5, f"every row carries the same moment ({p.min}): when they were loaded")
            word_set = set(names.tokens(c.column))
            if word_set & _AUDIT and not names.opaque(c.column) and (
                    word_set & (_AUDIT - _STAMP_ONLY) or c.granularity == "timestamp"):
                c.kind = "audit"
                c.add("audit_name", -1.0, f"the name ({readable}) says when the row was written")
            elif word_set & _VALIDITY or word_set & _TERM and word_set & _TERM_ENDS:
                c.kind = "validity"
                starts = bool(word_set & _STARTS)
                c.add("validity_name", 0.0 if starts else -0.25,
                      f"the name ({readable}) says when a term {'starts' if starts else 'ends'}: a validity date")
            elif word_set & _DUE:
                c.kind = "due"
                c.add("due_name", -0.25, f"the name ({readable}) says it is a due date")
            elif word_set & _PLANNED:
                c.kind = "planned"
                c.add("planned_name", -0.25, f"the name ({readable}) says it is planned or requested")
            elif word_set & _CANCEL:
                c.add("cancel_name", -0.3, f"the name ({readable}) says it applies only to cancelled rows")
            elif word_set & _PERSONAL:
                c.add("personal_name", -0.5, f"the name ({readable}) is a date in a person's life, not an event "
                      "the table records")
            elif word_set & _MILESTONE:
                c.add("milestone_name", -0.4, f"the name ({readable}) is the first, last or original time "
                      "something happened to what the row describes, not the event the row records")
            meaningful = not names.opaque(c.column) and not names.opaque(table.name)
            if meaningful and words and table_words and any(names.same_word(w, t) for w in words for t in table_words):
                c.add("names_event", 0.25, f"{readable} is the event the table records")
                if head and any(names.same_word(w, head) for w in words):
                    c.add("names_head", 0.1, f"it names the table's main subject ({head})")
            if c.kind == "event" and c.granularity in ("day", "month"):   # a planned or due date is never a snapshot
                periodic, why = attempt(
                    warehouse, f"the snapshot check on {table.name}.{c.column}",
                    partial(_periodic, warehouse, inventory, key, c.column, c.granularity, profile.rows,
                            entity_columns.get(key, []),
                            [k for k in keys[key].primary_key if k != c.column] if c.column in keys[key].primary_key
                            else None), (False, "")) \
                    if profile.rows >= 50 and p.distinct and profile.rows / p.distinct >= 5 else (False, "")
                if periodic:
                    c.periodic = True
                    c.kind = "snapshot"
                    c.add("snapshot", 0.35, why)

        attempt(warehouse, f"the load-date check on {table.name}",
                partial(_written_later, warehouse, inventory, key, candidates), None)
        attempt(warehouse, f"the load-time check on {table.name}",
                partial(_load_times, warehouse, inventory, key, candidates), None)
        _old_dates(candidates)
        attempt(warehouse, f"the row-stamp check on {table.name}",
                partial(_made_with, warehouse, inventory, key, candidates), None)
        for c in candidates:
            if c.kind == "event" and set(names.tokens(c.column)) & _POSTING and not names.opaque(c.column):
                c.add("posting_name", 0.3, f"the name ({names.readable(c.column)}) is the date the books count by")
        attempt(warehouse, f"the date-order check on {table.name}",
                partial(_first_of, warehouse, inventory, key, candidates), None)
        business = [c for c in candidates if _business(c)]
        ranked = sorted(business or [], key=lambda c: (-c.score, [x.name for x in table.columns].index(c.column)))
        if ranked:
            ranked[0].is_default = True
        out[key] = candidates
    return out


def _months(value: exp.Expression, candidate: DateCandidate, data_type: str, dialect: str) -> exp.Expression:
    """A date as a month count (year * 12 + month): from a yyyymm or yyyymmdd number, or a date."""
    if candidate.granularity == "month":
        return D.add(D.mul(D.floor_div(value, 100), 12), D.mod(value.copy(), 100))
    if data_type in ("integer", "decimal"):
        return D.add(D.mul(D.floor_div(value, 10000), 12),
                     D.mod(D.floor_div(value.copy(), 100), 100))
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


def _sample(t: InvTable, dialect: str) -> str:
    """The first rows of a table, enough to see how two dates of a row relate without reading all of it."""
    return D.aliased(D.first_rows(D.table_sql(t.database, t.schema, t.name, dialect), dialect, rows=200_000),
                     "s", dialect)


def _business(c: DateCandidate) -> bool:
    """A date a table may be dated by: its events, its snapshot's period, or the start of a term. Never a
    date in a person's life (a birth): a table whose only date is one has no default."""
    if any(e.kind in ("personal_name", "old_dates") for e in c.evidence):
        return False
    return c.kind in ("event", "snapshot") or c.kind == "validity" and bool(set(names.tokens(c.column)) & _STARTS)


def _old_dates(candidates: list[DateCandidate]) -> None:
    """A date whose values end decades before the table's other dates is a date in a life (a birth), not an
    event the table records: an employee is dated by the hire, not the birthday."""
    lasts = [c.last for c in candidates if c.last and c.kind in ("event", "snapshot")]
    if len(lasts) < 2:
        return
    latest = max(lasts)
    for c in candidates:
        if c.kind == "event" and c.last and c.last.year < latest.year - 10:
            c.add("old_dates", -0.4, f"its dates end in {c.last.year}, {latest.year - c.last.year} years before the "
                  "table's others: a date in the life of what a row describes, not the event it records")


def _made_with(warehouse: Warehouse, inventory: Inventory, key: str, candidates: list[DateCandidate]) -> None:
    """A timestamp named for when the row was made (CREATED_AT) that follows another time of the same rows
    within a day records the row being written for that event, not a second event."""
    anchors = [c for c in candidates if c.kind == "event" and c.granularity == "timestamp"
               and not set(names.tokens(c.column)) & _MADE]
    stamps = [c for c in candidates if c.kind == "event" and c.granularity == "timestamp"
              and set(names.tokens(c.column)) & _MADE and not names.opaque(c.column)]
    if not anchors or not stamps:
        return
    d = warehouse.dialect
    t = inventory.tables[key]
    for stamp in stamps:
        for anchor in anchors:
            a, s = exp.column(D.ident(anchor.column, d)), exp.column(D.ident(stamp.column, d))
            close = exp.and_(exp.GTE(this=s.copy(), expression=a.copy()),
                             exp.LTE(this=D.days_between(a.copy(), s.copy(), d), expression=exp.Literal.number(1)))
            query = exp.select(exp.Count(this=exp.Literal.number(1)).as_("n"), exp.Sum(this=exp.Case(
                ifs=[exp.If(this=close, true=exp.Literal.number(1))], default=exp.Literal.number(0))).as_("near")
            ).from_(exp.to_table("__SRC__")).where(exp.and_(exp.not_(exp.Is(this=a.copy(), expression=exp.Null())),
                                                            exp.not_(exp.Is(this=s.copy(), expression=exp.Null()))))
            n, near = warehouse.query(query.sql(dialect=d).replace("__SRC__", _sample(t, d), 1)).rows[0]
            if n and int(near or 0) / int(n) >= 0.95:
                stamp.kind = "audit"
                stamp.add("made_with", -1.0, f"{int(near) / int(n):.0%} of rows were made within a day after their "
                          f"{names.readable(anchor.column).lower()}: it records the row being written")
                break


def _first_of(warehouse: Warehouse, inventory: Inventory, key: str, candidates: list[DateCandidate]) -> None:
    """Dates that follow one another on every row (raised, acknowledged, cleared; written, received) date
    the row by the first: the event the row records starting. What comes after is its progress."""
    events = [c for c in candidates if c.kind == "event"]
    if len(events) < 2:
        return
    d = warehouse.dialect
    t = inventory.tables[key]

    def comparable(c: DateCandidate) -> str:
        """Dates with dates, keys with keys of the same kind: a timestamp is never compared with 20260315."""
        kind = t.type_of(c.column)
        return "date" if c.via_calendar is None and kind in ("date", "timestamp") else f"{kind}:{c.via_calendar is None}"

    def precedes(a: DateCandidate, b: DateCandidate) -> bool:
        x, y = exp.column(D.ident(a.column, d)), exp.column(D.ident(b.column, d))

        def count(condition: exp.Expression) -> exp.Expression:
            return exp.Sum(this=exp.Case(ifs=[exp.If(this=condition, true=exp.Literal.number(1))],
                                         default=exp.Literal.number(0)))

        query = exp.select(exp.Count(this=exp.Literal.number(1)), count(exp.LTE(this=x.copy(), expression=y.copy())),
                           count(exp.LT(this=x.copy(), expression=y.copy()))).from_(exp.to_table("__SRC__")).where(
            exp.and_(exp.not_(exp.Is(this=x.copy(), expression=exp.Null())),
                     exp.not_(exp.Is(this=y.copy(), expression=exp.Null()))))
        n, before, strictly = warehouse.query(query.sql(dialect=d).replace("__SRC__", _sample(t, d), 1)).rows[0]
        n = int(n or 0)
        return n >= 20 and int(before or 0) / n >= 0.98 and int(strictly or 0) / n >= 0.2

    if len({comparable(c) for c in events}) > 1:
        return      # a date key and a timestamp are not compared: the first of some would not be the first of all
    for c in events:
        others = [o for o in events if o is not c]
        if all(precedes(c, o) for o in others):
            # What follows and is often still empty (acknowledged, resolved, paid) is the row's progress: the
            # first date is the row's own. Two dates every row has (a document's and its posting) are a
            # convention the order cannot settle: a nudge, and the admin is asked.
            progress = all(o.coverage < 0.99 or set(names.tokens(o.column)) & _PROGRESS for o in others)
            c.add("first_event", 0.15 if progress else 0.05,
                  "it comes before " + ", ".join(names.readable(o.column).lower() for o in others)
                  + " on every row" + (": the event each row starts with" if progress else ""))
            return


def close_call(candidates: list[DateCandidate]) -> DateCandidate | None:
    """The runner-up when it is within the margin of the default."""
    business = sorted((c for c in candidates if _business(c)), key=lambda c: -c.score)
    if len(business) >= 2 and business[0].score - business[1].score < MARGIN:
        return business[1]
    return None
