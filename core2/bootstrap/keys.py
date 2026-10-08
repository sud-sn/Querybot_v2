"""Which columns identify a row: the primary key and other unique combinations.

A declared key is checked, not trusted. Without one, the smallest unique column
set is searched among key-like columns: single columns first (from exact distinct
counts, verified when the count was estimated), then pairs and triples whose
distinct counts could cover the rows, verified with a duplicate-group count.
Alternate keys (a document number plus a line number) are looked for too: they
tell a line number from a measure.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from sqlglot import exp

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


def _restarts(warehouse: Warehouse, table: InvTable, doc: str, seq: str) -> bool:
    """Does ``seq`` start again at 1 in (nearly) every ``doc``, as a line number does in each order?

    A department number beside a nearly unique name is unique with it by chance, and starts at 1
    only for the names that happen to be in department 1.
    """
    d = warehouse.dialect
    first = exp.Min(this=exp.column(D.ident(seq, d)))
    inner = exp.select(first.as_("m")).from_(exp.to_table("__SRC__")).group_by(exp.column(D.ident(doc, d)))
    starts = exp.Sum(this=exp.Case(ifs=[exp.If(this=exp.EQ(this=exp.column("m"), expression=exp.Literal.number(1)),
                                                true=exp.Literal.number(1))], default=exp.Literal.number(0)))
    query = exp.select(starts, exp.Count(this=exp.Literal.number(1))).from_(inner.subquery("x"))
    sql = query.sql(dialect=d).replace("__SRC__", D.table_sql(table.database, table.schema, table.name, d))
    started, groups = attempt(warehouse, f"the line-number check on {table.name} ({doc}, {seq})",
                              lambda: warehouse.query(sql).rows[0], (0, 0))
    return bool(groups) and int(started or 0) >= 0.9 * int(groups)


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
        seqs = [c.name for c in table.columns if c.name not in unique
                and table.type_of(c.name) == "integer"
                and profile.columns[c.name].min_num == 1 and 1 < profile.columns[c.name].distinct <= 1000
                and (profile.columns[c.name].max_num or 0) <= 10000]
        found: list[list[str]] = []
        for doc, seq in itertools.product(docs, seqs):
            if doc == seq or len(found) >= 1:
                continue
            if profile.columns[doc].distinct * profile.columns[seq].distinct < rows:
                continue
            if not _duplicates(warehouse, table, [doc, seq]) and _restarts(warehouse, table, doc, seq):
                found.append([doc, seq])
        result.alternate_keys = found
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
    return result
