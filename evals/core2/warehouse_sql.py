"""A test warehouse that speaks Azure SQL, Snowflake or Oracle SQL, and Azure SQL's own rules.

The new core is tested on DuckDB, which accepts what Azure SQL refuses: the
remainder of a float, a column named twice in a derived table, an int sum past
2,147,483,647, a division by zero, a sample of a view, text compared across two
collations. Each refusal stopped a Learn on a real database the first time it
was met. Here every query is (1) checked against those rules, (2) carried back
to DuckDB by sqlglot, with the few idioms it does not translate rewritten, and
(3) run, so the result can be compared with what DuckDB itself gives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from core2.bootstrap.inventory import Inventory
from core2.warehouse.runner import DuckDBWarehouse, QueryResult

DB_TYPES = {"tsql": "azure_sql", "snowflake": "snowflake", "oracle": "oracle"}
_FLOATS = {"FLOAT", "DOUBLE", "REAL"}
_WIDE = {"BIGINT", "DECIMAL", "FLOAT", "DOUBLE", "NUMERIC", "MONEY"}
_STRINGS = {"CHAR", "VARCHAR", "NCHAR", "NVARCHAR"}
_UNITS = {"MM": "month", "Q": "quarter", "IW": "week", "YYYY": "year", "DD": "day"}
_PREDICATES = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.And, exp.Or, exp.Not, exp.Is, exp.In,
               exp.Between, exp.Like, exp.ILike)


@dataclass
class Types:
    """What the warehouse says each column is, by bare name; a name typed differently in two tables is "mixed"."""

    data: dict[str, str] = field(default_factory=dict)
    raw: dict[str, str] = field(default_factory=dict)
    views: set[str] = field(default_factory=set)

    @classmethod
    def of(cls, inventory: Inventory, views: set[str] | None = None) -> Types:
        out = cls(views={v.upper() for v in views or set()})
        for table in inventory.tables.values():
            for column in table.columns:
                name = column.name.upper()
                raw = column.raw_type.strip().upper().split("(")[0].strip()
                out.data[name] = column.data_type if out.data.get(name, column.data_type) == column.data_type \
                    else "mixed"
                out.raw[name] = raw if out.raw.get(name, raw) == raw else "MIXED"
        return out

    def of_column(self, column: exp.Column) -> str:
        return self.data.get(column.name.upper(), "")


def _under_cast(node: exp.Expression) -> bool:
    parent = node.parent
    while isinstance(parent, exp.Paren):
        parent = parent.parent
    return isinstance(parent, (exp.Cast, exp.TryCast))


def _unwrap(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _selects_of(node: exp.Expression) -> list[exp.Select]:
    if isinstance(node, exp.Select):
        return [node]
    if isinstance(node, exp.SetOperation):
        return _selects_of(node.this) + _selects_of(node.expression)
    return []


def azure_sql_refuses(tree: exp.Expression, types: Types) -> list[str]:
    """What Azure SQL would refuse in this query, though DuckDB runs it: its own rules, by error number."""
    refused: list[str] = []

    def say(number: int, what: str, node: exp.Expression) -> None:
        refused.append(f"{number} {what}: {node.sql(dialect='tsql')[:160]}")

    for mod in tree.find_all(exp.Mod):
        for side in (mod.this, mod.expression):
            if any(c.to.this.name in _FLOATS for c in side.find_all(exp.Cast)) or \
                    any(types.of_column(c) == "float" for c in side.find_all(exp.Column)):
                say(402, "the remainder of a float", mod)
    for total in tree.find_all(exp.Sum):
        inner = _unwrap(total.this)
        if total.find_ancestor(exp.Window) or (isinstance(inner, exp.Cast) and inner.to.this.name in _WIDE):
            continue
        values = [inner] if not isinstance(inner, exp.Case) else [i.args["true"] for i in inner.args.get("ifs") or []]
        if any(isinstance(_unwrap(v), exp.Column) and types.of_column(_unwrap(v)) == "integer" for v in values):
            say(8115, "an int sum that overflows past 2,147,483,647", total)
    for mul in tree.find_all(exp.Mul):
        sides = [_unwrap(mul.this), _unwrap(mul.expression)]
        if any(isinstance(s, exp.Literal) and s.is_number and abs(float(s.name)) >= 100 for s in sides) and \
                any(isinstance(s, exp.Column) and types.of_column(s) == "integer" for s in sides):
            say(8115, "an int product that can overflow", mul)
    for div in tree.find_all(exp.Div):
        den = _unwrap(div.expression)
        if isinstance(den, (exp.Cast, exp.TryCast)):
            den = _unwrap(den.this)
        if not (isinstance(den, exp.Nullif) or isinstance(den, exp.Literal) and den.is_number and float(den.name) != 0):
            say(8134, "a division that can divide by zero", div)
    for node in [*tree.find_all(exp.Subquery), *[cte.this for cte in tree.find_all(exp.CTE)]]:
        body = node.this if isinstance(node, exp.Subquery) else node
        # A derived table (in FROM or JOIN) or a CTE names its columns; a scalar or IN subquery need not.
        named = not isinstance(node, exp.Subquery) or isinstance(node.parent, (exp.From, exp.Join))
        for select in _selects_of(body):
            if select.args.get("order") and not (select.args.get("limit") or select.args.get("offset")):
                say(1033, "ORDER BY in a derived table without TOP", select)
            if not named:
                continue
            names = []
            for e in select.expressions:
                if isinstance(e, exp.Star) or isinstance(e, exp.Column) and isinstance(e.this, exp.Star):
                    continue
                if not e.alias_or_name:
                    say(8155, "a derived-table column with no name", e)
                names.append(e.alias_or_name.casefold())
            if any(names.count(n) > 1 for n in names if n):
                say(8156, "a derived-table column named twice", select)
        if isinstance(node, exp.Subquery) and isinstance(node.parent, (exp.From, exp.Join)) and not node.alias:
            say(102, "a derived table with no alias", node)
    for count in tree.find_all(exp.Count):
        if isinstance(count.this, exp.Distinct) and len(count.this.expressions) > 1:
            say(102, "COUNT(DISTINCT) of several columns", count)
    for agg in tree.find_all(exp.Sum, exp.Avg, exp.Min, exp.Max):
        if isinstance(_unwrap(agg.this), exp.Column) and types.raw.get(_unwrap(agg.this).name.upper()) == "BIT":
            say(8117, "an aggregate of a bit column", agg)
    for column in tree.find_all(exp.Column):
        if types.raw.get(column.name.upper()) in ("TEXT", "NTEXT") and not _under_cast(column) \
                and not (isinstance(column.parent, exp.Count)):
            say(306, "a text or ntext column compared or sorted", column)
    for select in tree.find_all(exp.Select):
        aliases = {e.alias.casefold() for e in select.expressions if isinstance(e, exp.Alias)}
        for e in select.expressions:
            if isinstance(_unwrap(e.this if isinstance(e, exp.Alias) else e), _PREDICATES):
                say(102, "a true/false value in the select list", e)
        group = select.args.get("group")
        for g in (group.expressions if group else []):
            if isinstance(g, exp.Column) and not g.table and g.name.casefold() in aliases \
                    and g.name.upper() not in types.data:
                say(207, "GROUP BY a select alias", g)
    for agg in tree.find_all(exp.AggFunc):
        if isinstance(_unwrap(agg.this) if isinstance(agg.this, exp.Expression) else None, _PREDICATES):
            say(102, "a true/false value inside an aggregate", agg)
    for eq in tree.find_all(exp.EQ):
        a, b = _unwrap(eq.this), _unwrap(eq.expression)
        if isinstance(a, exp.Column) and isinstance(b, exp.Column) and a.table and b.table and a.table != b.table \
                and a.name.casefold() != b.name.casefold() \
                and types.of_column(a) == "text" and types.of_column(b) == "text":
            say(468, "text compared across two tables without a collation", eq)
    for collate in tree.find_all(exp.Collate):
        inner = _unwrap(collate.this)
        if isinstance(inner, exp.Column) and types.raw.get(inner.name.upper()) not in _STRINGS:
            say(447, "COLLATE on a column that is not text", collate)
    text = tree.sql(dialect="tsql").upper()
    if "NULLS FIRST" in text or "NULLS LAST" in text:
        refused.append("102 NULLS FIRST/LAST in ORDER BY")
    for sample in tree.find_all(exp.TableSample):
        table = sample.find_ancestor(exp.Table) or sample.parent
        name = table.name.upper() if isinstance(table, exp.Table) else ""
        if name in types.views:
            say(494, "TABLESAMPLE on a view", sample)
    return refused


def to_duckdb(tree: exp.Expression, dialect: str) -> str:
    """The query as DuckDB reads it, where sqlglot leaves a warehouse's own idiom untranslated."""
    def fix(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Collate):
            return node.this.copy()
        if isinstance(node, exp.TableSample):
            node.set("method", exp.var("BERNOULLI"))   # DuckDB's default samples whole blocks: none of a small table
        if isinstance(node, exp.DataType) and any(e.name.upper() == "MAX" for e in node.expressions):
            node.set("expressions", [])               # NVARCHAR(MAX): DuckDB's text has no length
        if dialect == "tsql":
            if isinstance(node, exp.Parameter) and "DATEFIRST" in node.sql().upper():
                return exp.Literal.number(7)        # Azure SQL's default: Sunday is day 1
            if isinstance(node, exp.Extract) and node.this.name.upper() == "DAYOFWEEK":
                return exp.Paren(this=exp.Add(this=node.copy(), expression=exp.Literal.number(1)))
        if dialect == "oracle":
            if isinstance(node, exp.Anonymous) and node.name.upper() == "TRUNC" and len(node.expressions) == 1:
                return exp.DateTrunc(this=node.expressions[0].copy(), unit=exp.Literal.string("day"))
            if isinstance(node, exp.DateTrunc) and isinstance(node.args.get("unit"), exp.Literal):
                unit = _UNITS.get(node.args["unit"].name.upper())
                if unit:
                    node.set("unit", exp.Literal.string(unit))
        return node

    return tree.transform(fix).sql(dialect="duckdb")


class SpeakingWarehouse(DuckDBWarehouse):
    """A DuckDB test warehouse that takes another warehouse's SQL, and refuses what Azure SQL refuses."""

    def __init__(self, con, dialect: str, types: Types):
        super().__init__(con)
        self.dialect, self.db_type = dialect, DB_TYPES[dialect]
        self.types, self.refused = types, []

    def query(self, sql: str, *, max_rows: int | None = None) -> QueryResult:
        tree = sqlglot.parse_one(sql, read=self.dialect)
        if self.dialect == "tsql":
            found = azure_sql_refuses(tree, self.types)
            stopped = [p for p in found if p.startswith("494 ")]
            self.refused += [p for p in found if not p.startswith("494 ")]
            if stopped:   # Azure SQL stops this query, as it would: the code must cope
                raise RuntimeError(f"[42000] TABLESAMPLE cannot be applied to views ({stopped[0]})")
        return DuckDBWarehouse.query(self, to_duckdb(tree, self.dialect), max_rows=max_rows)
