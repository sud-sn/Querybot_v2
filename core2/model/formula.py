"""Formulas brought over from today's metrics, read before the new core uses them.

Today's metrics are often a formula over their table's columns
(``SUM(ON_HND_QTY * ITM_CST)``, ``SUM(ON_HND_QTY) - COALESCE(SUM(ALC_ON_HND_QTY), 0)``).
Only what a measure needs is accepted: the aggregates SUM, AVG, COUNT (with
DISTINCT), MIN and MAX over columns of the metric's own table, arithmetic,
COALESCE, NULLIF, CAST, numbers and strings, and CASE WHEN with comparisons.
Anything else (a subquery, another table, any other function, a column outside
an aggregate) is refused with its reason. What is accepted is kept as a parsed
form with the warehouse's own column names, and the compiler writes the SQL
from it: a formula's text is never pasted into a query.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

AGGREGATES = (exp.Sum, exp.Avg, exp.Count, exp.Min, exp.Max)
_ALLOWED = (*AGGREGATES, exp.Add, exp.Sub, exp.Mul, exp.Div, exp.Paren, exp.Literal, exp.Column, exp.Identifier,
            exp.Neg, exp.Distinct, exp.Star, exp.Coalesce, exp.Nullif, exp.Case, exp.If, exp.EQ, exp.NEQ, exp.GT,
            exp.GTE, exp.LT, exp.LTE, exp.And, exp.Or, exp.Not, exp.Is, exp.Null, exp.In, exp.Boolean, exp.Cast,
            exp.DataType, exp.DataTypeParam, exp.Tuple)
_READS = (None, "tsql", "snowflake", "oracle")


class FormulaError(ValueError):
    """Why a formula is not used: said to the admin, in words."""


@dataclass
class Formula:
    tree: exp.Expr                                    # columns are bare, as the warehouse spells them
    columns: list[str] = field(default_factory=list)  # their column keys, each once
    additive: bool = False                            # adds up across rows (sums and counts, added or subtracted)

    @property
    def sql(self) -> str:
        return self.tree.sql()


def parse(text: str, column_key: Callable[[str], str | None], spelled: Callable[[str], str],
          table_names: set[str]) -> Formula:
    """Read ``text`` as a measure over one table.

    ``column_key(name)`` finds a column of the metric's table by name (None when it has none),
    ``spelled(key)`` gives the warehouse's spelling of that column, and ``table_names`` are the
    names a column may be qualified with (the table's own, case-insensitive).
    """
    first: FormulaError | None = None
    for read in _READS:      # the generic reading first; a warehouse's own spelling ([COL]) if that fails
        try:
            tree = sqlglot.parse_one(str(text or "").strip().rstrip(";"), read=read)
        except sqlglot.errors.ParseError:
            continue
        try:
            return _checked(tree, column_key, spelled, table_names)
        except FormulaError as exc:
            first = first or exc
    raise first or FormulaError("it cannot be read as a formula")


def _checked(tree: exp.Expr, column_key: Callable[[str], str | None], spelled: Callable[[str], str],
             table_names: set[str]) -> Formula:
    for node in tree.walk():
        if not isinstance(node, _ALLOWED):
            raise FormulaError(f"it uses {node.key.upper()}, which a measure does not")
    if not any(isinstance(n, AGGREGATES) for n in tree.walk()):
        raise FormulaError("it adds nothing up (no SUM, COUNT, AVG, MIN or MAX)")
    keys: list[str] = []
    for column in list(tree.find_all(exp.Column)):
        if column.find_ancestor(*AGGREGATES) is None:
            raise FormulaError(f"{column.name} is used outside a SUM, COUNT, AVG, MIN or MAX")
        if column.table and column.table.casefold() not in table_names:
            raise FormulaError(f"it reads {column.table}, another table")
        key = column_key(column.name)
        if key is None:
            raise FormulaError(f"its table has no column {column.name}")
        column.replace(exp.column(spelled(key)))
        if key not in keys:
            keys.append(key)
    return Formula(tree, keys, additive(tree))


def parse_stored(sql: str) -> exp.Expr:
    """A formula as stored by an import (already checked, in the generic spelling)."""
    return sqlglot.parse_one(sql)


def additive(tree: exp.Expr) -> bool:
    """Sums and counts, added or subtracted (``SUM(a * b)`` is additive; a ratio or an average is not)."""
    for node in tree.walk():
        if isinstance(node, (exp.Avg, exp.Min, exp.Max)):
            return False
        if isinstance(node, exp.Count) and isinstance(node.this, exp.Distinct):
            return False
        if isinstance(node, (exp.Div, exp.Mul)) and node.find_ancestor(*AGGREGATES) is None:
            return False
    return True
