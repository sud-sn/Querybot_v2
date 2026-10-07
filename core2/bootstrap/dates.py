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

_AUDIT = {"loaded", "load", "ld", "etl", "updated", "upd", "update", "modified", "mod", "inserted", "ins",
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
    """Is ``column`` the period of a snapshot: do the same things recur in every period?

    A snapshot holds each item (or employee, account...) once per period, so the
    number of rows per item is about the number of periods. An event table's
    month-end or evenly spread dates do not make it a snapshot: its items do not
    recur period after period.
    """
    if not entity_columns:
        return False, ""
    t = inventory.tables[table]
    d = warehouse.dialect
    source = D.table_sql(t.database, t.schema, t.name, d)
    c = exp.column(D.ident(column, d))
    periods_q = exp.select(exp.Count(this=exp.Distinct(expressions=[c.copy()]))).from_(exp.to_table("__SRC__"))
    entities = exp.select(*[exp.column(D.ident(e, d)) for e in entity_columns]).distinct().from_(exp.to_table("__SRC__"))
    entities_q = exp.select(exp.Count(this=exp.Literal.number(1))).from_(entities.subquery("e"))
    periods = int(warehouse.query(periods_q.sql(dialect=d).replace("__SRC__", source)).rows[0][0] or 0)
    things = int(warehouse.query(entities_q.sql(dialect=d).replace("__SRC__", source)).rows[0][0] or 0)
    if periods < 3 or not things:
        return False, ""
    if _duplicates(warehouse, t, [column, *entity_columns]):
        return False, ""   # several rows per item and period: events, not a snapshot
    recurrence = rows / things
    if recurrence >= max(3.0, 0.5 * periods):
        return True, (f"each of {things:,} combinations of {', '.join(entity_columns)} recurs in about "
                      f"{recurrence:.0f} of {periods} periods")
    return False, ""


def find_date_roles(warehouse: Warehouse, inventory: Inventory, profiles: dict[str, TableProfile],
                    keys: dict[str, TableKeys], calendars: dict[str, CalendarFinding],
                    joins: list[JoinFinding]) -> dict[str, list[DateCandidate]]:
    """Date roles per table key, the default marked."""
    to_calendar = {(j.from_table, j.from_column): j for j in joins if j.to_calendar and j.trust != "rejected"}
    entity_columns: dict[str, list[str]] = {}
    for j in joins:
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
                granularity = "day"
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
            c.first, c.last = _parse_day(p.min, granularity), _parse_day(p.max, granularity)
            if c.first and c.first.year <= 1901 and join:
                c.first = None   # the minimum is the placeholder row's sentinel
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
            readable = names.readable(c.column)
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
            if c.kind != "audit" and c.granularity in ("day", "month"):
                periodic, why = _periodic(warehouse, inventory, key, c.column, c.granularity, profile.rows,
                                          entity_columns.get(key, [])) \
                    if profile.rows >= 50 and p.distinct and profile.rows / p.distinct >= 5 else (False, "")
                if periodic:
                    c.periodic = True
                    c.kind = "snapshot"
                    c.add("snapshot", 0.35, why)

        business = [c for c in candidates if c.kind in ("event", "snapshot")]
        ranked = sorted(business or [], key=lambda c: (-c.score, [x.name for x in table.columns].index(c.column)))
        if ranked:
            ranked[0].is_default = True
        out[key] = candidates
    return out


def close_call(candidates: list[DateCandidate]) -> DateCandidate | None:
    """The runner-up when it is within the margin of the default."""
    business = sorted((c for c in candidates if c.kind in ("event", "snapshot")), key=lambda c: -c.score)
    if len(business) >= 2 and business[0].score - business[1].score < MARGIN:
        return business[1]
    return None
