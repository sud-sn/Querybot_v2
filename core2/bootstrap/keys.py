"""Which columns identify a row: the primary key and other unique combinations.

A declared key is checked, not trusted. Without one, the smallest unique column
set is searched among key-like columns: single columns first (from exact distinct
counts, verified when the count was estimated), then pairs and triples whose
distinct counts could cover the rows, verified with a duplicate-group count.
Alternate keys (a document number plus a line number) are looked for too: they
tell a line number from a measure, and from a reference to a small table.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from sqlglot import exp

from core2.bootstrap import names
from core2.bootstrap.inventory import InvTable
from core2.bootstrap.profiler import TableProfile
from core2.model.schema import Evidence
from core2.bootstrap.journal import attempt
from core2.warehouse import dialect as D
from core2.warehouse.runner import Warehouse

MAX_COMPOSITE_CHECKS = 10


@dataclass
class TableKeys:
    primary_key: list[str] = field(default_factory=list)
    unique_columns: list[str] = field(default_factory=list)
    alternate_keys: list[list[str]] = field(default_factory=list)
    # Columns numbering each document's rows (a line in each booking, a fill of each prescription), in the
    # key or beside it: they count within the document and point at no table.
    line_numbers: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)


def _duplicates(warehouse: Warehouse, table: InvTable, columns: list[str]) -> int:
    """Rows sharing their values in ``columns``; when the check is refused, counted as having some."""
    columns = list(dict.fromkeys(columns))   # a column named twice is grouped once: Azure SQL refuses it twice (8156)
    return attempt(warehouse, f"the key check on {table.name} ({', '.join(columns)})",
                   lambda: _count_duplicates(warehouse, table, columns), 1)


def _count_duplicates(warehouse: Warehouse, table: InvTable, columns: list[str]) -> int:
    d = warehouse.dialect
    cols = [exp.column(D.ident(c, d)) for c in columns]
    inner = (exp.select(*[c.copy() for c in cols]).from_(exp.to_table("__SRC__"))
             .group_by(*[c.copy() for c in cols]).having(exp.GT(this=exp.Count(this=exp.Literal.number(1)),
                                                                expression=exp.Literal.number(1))))
    query = exp.select(exp.Count(this=exp.Literal.number(1))).from_(inner.subquery("x"))
    sql = query.sql(dialect=d).replace("__SRC__", D.table_sql(table.database, table.schema, table.name, d))
    return int(warehouse.query(sql).rows[0][0] or 0)


# A number that runs within a document, named so (a fill number seen from the fifth fill on starts at 5).
_SEQUENCE_WORDS = {"no", "nbr", "num", "number", "seq", "sequence", "line", "ln", "version", "ver", "renewal",
                   "rnwl", "fill", "installment", "instalment", "revision", "rev", "step"}


def _restarts(warehouse: Warehouse, table: InvTable, doc: str, seq: str, *, values: int = 0,
              gapless: bool = False) -> bool:
    """Is ``seq`` a running number within each ``doc``, as a line number is in each order or a fill number
    in each prescription? It starts again at 1 in (nearly) every doc, or it runs without a gap in (nearly)
    every doc of several rows, wherever it starts (a prescription's fills seen from its fifth on).

    A department number beside a nearly unique name is unique with it by chance: it starts at 1
    only for the names in department 1, and the names with several rows are too few to say it runs.
    Nor does a column whose every value sits in most docs (``values`` of them): every item 1..N on
    each day of a daily balance is a list of items, not each day's lines. ``gapless`` asks for the
    run without a gap even where it starts at 1: playlist 1 holds nearly every track, and a track's
    playlists are still not its lines.
    """
    d = warehouse.dialect
    s = exp.column(D.ident(seq, d))
    inner = exp.select(exp.Min(this=s.copy()).as_("m"), exp.Max(this=s.copy()).as_("x"),
                       exp.Count(this=exp.Literal.number(1)).as_("n")).from_(exp.to_table("__SRC__")).group_by(
        exp.column(D.ident(doc, d)))

    def when(condition: exp.Expr, value: exp.Expression) -> exp.Expression:
        return exp.Sum(this=exp.Case(ifs=[exp.If(this=condition, true=value)], default=exp.Literal.number(0)))

    several = exp.GT(this=exp.column("n"), expression=exp.Literal.number(1))
    runs = exp.and_(several.copy(), exp.EQ(this=D.add(D.sub(exp.column("x"), exp.column("m")), 1),
                                          expression=exp.column("n")))
    one = exp.Literal.number(1)
    every = exp.EQ(this=exp.column("n"), expression=exp.Literal.number(max(values, 2)))
    query = exp.select(when(exp.EQ(this=exp.column("m"), expression=one.copy()), one.copy()),
                       exp.Count(this=one.copy()), when(several, one.copy()), when(runs, one.copy()),
                       when(several.copy(), exp.column("n")), exp.Sum(this=exp.column("n")),
                       when(every, one.copy())).from_(inner.subquery("g"))
    sql = query.sql(dialect=d).replace("__SRC__", D.table_sql(table.database, table.schema, table.name, d))
    got = attempt(warehouse, f"the line-number check on {table.name} ({doc}, {seq})",
                  lambda: warehouse.query(sql).rows[0], (0, 0, 0, 0, 0, 0, 0))
    started, groups, multi, running, multi_rows, rows, full = (int(v or 0) for v in got)
    if values and full >= 0.5 * max(groups, 1):
        return False
    if groups and started >= 0.9 * groups and not gapless:
        return True
    return multi > 0 and running >= 0.9 * multi and multi_rows >= 0.3 * max(rows, 1)


def _numbers_lines(table: InvTable, profile: TableProfile, column: str) -> bool:
    """Could ``column`` number the lines of a document: a small whole number from 0 or 1, or named so."""
    p = profile.columns[column]
    return table.type_of(column) == "integer" \
        and (p.min_num in (0, 1) or bool(set(names.tokens(column)) & _SEQUENCE_WORDS)) \
        and 1 < p.distinct <= 1000 and (p.max_num or 0) <= 10000


def _lines_in_key(warehouse: Warehouse, table: InvTable, profile: TableProfile, result: TableKeys) -> None:
    """A key of several columns, one a document: the column numbering each document's rows. The key's own
    (a booking's line number) or one beside it (a prescription's fill number, where the fill's date was
    found unique with the prescription first). The number runs without a gap in each document, and the
    two identify the row on their own: a period, a warehouse and an item key a balance, and the
    warehouse numbers nothing within the period."""
    columns = profile.columns
    key = result.primary_key
    for doc in key:
        if table.type_of(doc) not in ("text", "integer"):
            continue
        for seq in [*key, *(c.name for c in table.columns if c.name not in key)]:
            if seq == doc or not _numbers_lines(table, profile, seq) or columns[seq].distinct >= columns[doc].distinct:
                continue
            if sorted(key) != sorted([doc, seq]) and (columns[doc].distinct * columns[seq].distinct < profile.rows
                                                      or _duplicates(warehouse, table, [doc, seq])):
                continue
            if _restarts(warehouse, table, doc, seq, values=columns[seq].distinct, gapless=True):
                result.line_numbers = [seq]
                if seq not in key:
                    result.alternate_keys = [[doc, seq]]
                return


def series_key(warehouse: Warehouse, table: InvTable, profile: TableProfile, stamp: str) -> str | None:
    """What each reading of a timestamped table is of: the column that, with the time, identifies a row (an
    interface's counters every 15 minutes), when no link says so. Fewest values first: the device, before a
    reading that happens to be unique at each time."""
    rows = profile.rows
    if profile.columns[stamp].distinct > rows / 3:
        return None    # one row at each time, or nearly (orders): nothing is read again and again
    found = sorted((p.distinct, c.name) for c in table.columns
                   if c.name != stamp and table.type_of(c.name) in ("integer", "text")
                   and (p := profile.columns[c.name]).non_null == rows and 3 <= p.distinct <= rows / 10
                   and not (p.pattern or "").startswith("flag"))
    for _, name in found[:3]:
        if not _duplicates(warehouse, table, [name, stamp]):
            return name
    return None


def _key_like(table: InvTable, profile: TableProfile) -> list[str]:
    out = []
    for c in table.columns:
        p = profile.columns[c.name]
        if p.non_null < profile.rows or p.distinct < 1:
            continue
        if c.data_type in ("integer", "date", "boolean") or p.pattern in ("yyyymmdd", "yyyymm"):
            out.append(c.name)
        elif c.data_type == "text" and (p.max_len or 0) <= 64:
            out.append(c.name)
        elif c.data_type == "decimal" and p.integer_share == 1.0:
            out.append(c.name)
    return out


def infer_keys(warehouse: Warehouse, table: InvTable, profile: TableProfile) -> TableKeys:
    result = TableKeys()
    rows = profile.rows
    if rows == 0:
        result.primary_key = list(table.primary_key)
        return result

    # Single unique columns.
    for c in table.columns:
        p = profile.columns[c.name]
        if p.non_null != rows or p.distinct < rows * (0.97 if p.distinct_is_approx else 1.0):
            continue
        if p.distinct_is_approx and _duplicates(warehouse, table, [c.name]):
            continue
        if c.data_type in ("float",):
            continue
        result.unique_columns.append(c.name)

    declared = list(table.primary_key)
    if declared:
        if len(declared) == 1 and declared[0] in result.unique_columns or \
                len(declared) > 1 and not _duplicates(warehouse, table, declared):
            result.primary_key = declared
            result.evidence.append(Evidence(kind="declared_key", detail=f"declared key {', '.join(declared)} is unique",
                                            weight=1.0))
        else:
            result.evidence.append(Evidence(kind="declared_key_broken",
                                            detail=f"declared key {', '.join(declared)} has duplicates", weight=-1.0))

    if not result.primary_key and result.unique_columns:
        # The first unique column in table order, preferring whole-number and code columns.
        ranked = sorted(result.unique_columns, key=lambda n: (
            0 if table.type_of(n) in ("integer", "text") else 1,
            [c.name for c in table.columns].index(n)))
        result.primary_key = [ranked[0]]
        result.evidence.append(Evidence(kind="uniqueness", detail=f"{ranked[0]} is unique and never empty",
                                        weight=0.9, data={"rows": rows}))

    if result.primary_key:
        # A document number and a line sequence beside a surrogate key: the line
        # number is then an identifier, not a quantity.
        unique = set(result.unique_columns) | set(result.primary_key)
        docs = [c.name for c in table.columns if c.name not in unique
                and rows / 1000 < profile.columns[c.name].distinct < rows
                and table.type_of(c.name) in ("text", "integer")]
        seqs = [c.name for c in table.columns if c.name not in unique and _numbers_lines(table, profile, c.name)]
        found: list[list[str]] = []
        for doc, seq in itertools.product(docs, seqs):
            if doc == seq or len(found) >= 1:
                continue
            if profile.columns[doc].distinct * profile.columns[seq].distinct < rows:
                continue
            if not _duplicates(warehouse, table, [doc, seq]) and _restarts(warehouse, table, doc, seq):
                found.append([doc, seq])
        result.alternate_keys = found
        result.line_numbers = [alt[1] for alt in found if profile.columns[alt[1]].distinct < profile.columns[alt[0]].distinct]
        if len(result.primary_key) > 1 and not found:
            _lines_in_key(warehouse, table, profile, result)
        return result

    # No single-column key: the smallest combination that identifies each row.
    candidates = [c for c in _key_like(table, profile)
                  if profile.columns[c].distinct > 1
                  and not (table.type_of(c) == "text" and profile.columns[c].distinct <= 20)
                  and table.type_of(c) != "decimal"]
    checks = 0
    found = []
    for size in (2, 3):
        combos = []
        for combo in itertools.combinations(candidates, size):
            product = 1
            for name in combo:
                product *= max(profile.columns[name].distinct, 1)
            if product < rows:
                continue
            dated = sum(1 for n in combo if profile.columns[n].pattern in ("yyyymmdd", "yyyymm")
                        or table.type_of(n) == "date")
            combos.append((-dated, product, combo))
        for _, _, combo in sorted(combos):
            if checks >= MAX_COMPOSITE_CHECKS or found:
                break
            checks += 1
            if not _duplicates(warehouse, table, list(combo)):
                found.append(list(combo))
        if found:
            break
    if found:
        result.primary_key = found[0]
        result.evidence.append(Evidence(kind="uniqueness", weight=0.8,
                                        detail=f"{', '.join(found[0])} together identify each row"))
        _lines_in_key(warehouse, table, profile, result)
    return result
