"""Tables that keep each member once per version: a customer's segment, as it changed.

A history table (a "slowly changing dimension") holds a row for each version of a member: the member's
own number repeats, and one row of each is its current version -- marked by a flag (IS_CURRENT = 1), or
by a validity period still open (VALID_TO empty, or a far-off date such as 9999-12-31). Read as an
ordinary list, every version is a member: a customer with three versions is three customers, counted
three times, and a link on the customer number finds three rows for each order.

Each claim is checked against the data before it is believed: the number repeats, and the rows the
marker keeps hold each number once -- and nearly every number has one. A table that only looks the part
(a status flag on a list whose numbers never repeat) is left as it is.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from sqlglot import exp

from core2.bootstrap import names
from core2.bootstrap.calendar import CalendarFinding
from core2.bootstrap.inventory import Inventory, InvTable
from core2.bootstrap.journal import attempt, journal_of
from core2.bootstrap.keys import TableKeys
from core2.bootstrap.profiler import TableProfile
from core2.model.schema import Evidence
from core2.warehouse import dialect as D
from core2.warehouse.runner import Warehouse

# Words a current-version flag is named with (IS_CURRENT, CURRENT_FLAG, CURR_IND, IS_LATEST).
_FLAG_WORDS = {"current", "curr", "latest"}
# The values that say "this is the current version".
_TRUE = {"1", "true", "t", "y", "yes", "current", "c"}
# Words of a validity period's end and start (VALID_TO, EFFECTIVE_END, END_DATE; VALID_FROM, EFF_START).
_END_WORDS = {"to", "end", "until", "thru", "through", "expiry", "expires", "expiration", "expire"}
_START_WORDS = {"from", "start", "effective", "eff", "begin", "begins"}
# An open period kept as a date: 9999-12-31, 9000-01-01, 2999-12-31.
FAR_OFF = "2900-01-01"
# Of the member numbers, the share that must have a current row.
_COVERED = 0.95


@dataclass
class VersionsFinding:
    table: str
    business_key: str                                   # the column naming the member, whatever its version
    current: list[tuple[str, str, list]]                # (column, op, values): the rows each member's current
    valid_from: str | None = None
    valid_to: str | None = None
    members: int = 0
    evidence: list[Evidence] = field(default_factory=list)


def find_versions(warehouse: Warehouse, inventory: Inventory, profiles: dict[str, TableProfile],
                  keys: dict[str, TableKeys], calendars: dict[str, CalendarFinding]) -> dict[str, VersionsFinding]:
    """Each table that keeps its members once per version, with what marks the current one."""
    out: dict[str, VersionsFinding] = {}
    for key, table in inventory.tables.items():
        profile = profiles.get(key)
        if key in calendars or profile is None or profile.rows < 4:
            continue
        markers = _markers(warehouse, table, profile)
        if not markers:
            continue
        found = None
        for candidate in _member_numbers(table, profile, keys[key], markers):
            for marker in markers:
                checked = attempt(warehouse, f"the versions check on {table.name} ({candidate})",
                                  lambda c=candidate, m=marker: _holds(warehouse, table, c, m, profile), None)
                if checked:
                    found = VersionsFinding(key, candidate, marker["current"], marker.get("from"), marker.get("to"),
                                            members=checked)
                    break
            if found:
                break
        if found is None:
            continue
        said = " and ".join(f"{c} {op} {', '.join(map(str, v)) or ''}".strip() for c, op, v in found.current)
        found.evidence.append(Evidence(kind="versions", weight=1.0, detail=(
            f"{table.name} keeps each {found.business_key} once per version ({profile.rows:,} rows for "
            f"{found.members:,} of them); the current one is where {said}")))
        out[key] = found
        journal_of(warehouse).step(f"{table.name} keeps each {found.business_key} once per version")
    return out


def _markers(warehouse: Warehouse, table: InvTable, profile: TableProfile) -> list[dict]:
    """What could mark the current version: a flag, or the end of a validity period."""
    out = []
    for column in table.columns:
        words = set(names.tokens(column.name))
        p = profile.columns[column.name]
        if words & _FLAG_WORDS and p.distinct == 2:
            value = attempt(warehouse, f"the values of {table.name}.{column.name}",
                            lambda c=column.name: _true_value(warehouse, table, c), None)
            if value is not None:
                out.append({"current": [(column.name, "eq", [value])]})
    dates = [c for c in table.columns if c.data_type in ("date", "timestamp")]
    starts = [c.name for c in dates if set(names.tokens(c.name)) & _START_WORDS]
    for column in dates:
        if not set(names.tokens(column.name)) & _END_WORDS or not starts:
            continue
        start = next((s for s in starts if s != column.name), None)
        if start is None:
            continue
        out.append({"current": [(column.name, "is_null", [])], "from": start, "to": column.name})
        out.append({"current": [(column.name, "gte", [FAR_OFF])], "from": start, "to": column.name})
    return out


def _true_value(warehouse: Warehouse, table: InvTable, column: str):
    d = warehouse.dialect
    col = exp.column(D.ident(column, d))
    sql = exp.select(col.copy()).distinct().from_(exp.to_table("__T__")).sql(dialect=d).replace(
        "__T__", D.table_sql(table.database, table.schema, table.name, d), 1)
    for (value,) in warehouse.query(sql).rows:
        if value is not None and str(value).strip().casefold() in _TRUE:
            return value
    return None


def _member_numbers(table: InvTable, profile: TableProfile, tkeys: TableKeys, markers: list[dict]) -> list[str]:
    """Columns that could name the member across its versions: a number or code that repeats."""
    marked = {c for m in markers for c, _, _ in m["current"]} | {m.get("from") for m in markers}
    own = set(names.core_table(table.name))
    out = []
    for column in table.columns:
        p = profile.columns[column.name]
        if column.name in marked or column.data_type not in ("integer", "text") or p.distinct < 2:
            continue
        if p.distinct >= profile.rows or column.name in tkeys.unique_columns:
            continue          # never repeats: each row is its own
        if column.data_type == "text" and (p.avg_len or 0) > 24:
            continue          # a name or a description, not a number
        words = names.tokens(column.name)
        if not words or words[-1] not in names.KEY_SUFFIXES:
            continue
        out.append(column.name)
    # The column named for its table first (CUSTOMER_ID on CUSTOMER_VERSIONS), then the one with most members.
    return sorted(out, key=lambda c: (not any(names.same_word(w, o) for w in names.tokens(c) for o in own),
                                      -profile.columns[c].distinct))


def _holds(warehouse: Warehouse, table: InvTable, column: str, marker: dict, profile: TableProfile) -> int | None:
    """Members counted when the marker keeps each member's number once and nearly every member has one."""
    d = warehouse.dialect
    col = exp.column(D.ident(column, d))
    kept = [exp.not_(exp.Is(this=col.copy(), expression=exp.Null()))]
    kept += [condition_sql(name, op, values, d) for name, op, values in marker["current"]]
    current = (exp.select(exp.Count(this=exp.Star()).as_("n"),
                          exp.Count(this=exp.Distinct(expressions=[col.copy()])).as_("k"))
               .from_(exp.to_table("__T__")).where(exp.and_(*kept)))
    every = exp.select(exp.Count(this=exp.Distinct(expressions=[col.copy()]))).from_(exp.to_table("__T__"))
    name = D.table_sql(table.database, table.schema, table.name, d)
    rows, keys_kept = warehouse.query(current.sql(dialect=d).replace("__T__", name, 1)).rows[0]
    members = warehouse.query(every.sql(dialect=d).replace("__T__", name, 1)).rows[0][0]
    rows, keys_kept, members = int(rows or 0), int(keys_kept or 0), int(members or 0)
    if not members or members >= profile.rows or rows != keys_kept or keys_kept < _COVERED * members:
        return None
    return members


def condition_sql(column: str, op: str, values: list, dialect: str) -> exp.Expr:
    """A current-version condition as a warehouse reads it (a flag's value, an open period's end), its values
    written as the compiler writes them (core2/compile/compiler.py, ``literal``)."""
    c = exp.column(D.ident(column, dialect))
    if op == "is_null":
        return exp.Is(this=c, expression=exp.Null())
    if op == "gte":
        return exp.GTE(this=c, expression=D.date_literal(dt.date.fromisoformat(str(values[0])[:10]), dialect))
    value = values[0]
    if isinstance(value, bool):
        literal = exp.Literal.number(int(value)) if dialect in ("tsql", "oracle") else exp.Boolean(this=value)
    elif isinstance(value, (int, float)):
        literal = exp.Literal.number(value)
    else:
        literal = exp.Literal.string(str(value))
    return exp.EQ(this=c, expression=literal)
