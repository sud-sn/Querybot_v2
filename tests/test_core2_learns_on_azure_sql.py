"""Learn this database runs on Azure SQL, and on every warehouse, as it does on DuckDB.

The first Learn on a real Azure SQL database stopped at its first table:
"The data types float and int are incompatible in the modulo operator (402)".
Reading a yyyymmdd key's month divided by 100, rounded down and took the
remainder; sqlglot writes that division for Azure SQL as CAST(key AS FLOAT) / 100,
and Azure SQL refuses the remainder of a float. DuckDB, where every test ran,
has no such rule, and only the answer queries were ever written in the other
dialects -- never the queries Learn sends.

Here the whole of Learn runs in each warehouse's SQL: every query is checked
against the rules Azure SQL enforces and DuckDB does not, carried back by
sqlglot and run on the test warehouse, and the model must be the one DuckDB
learns. Integer sums are summed as bigints on Azure SQL (an int sum stops
past 2,147,483,647), and a column the database refuses to read is left out
and named instead of stopping the whole Learn.
"""

from __future__ import annotations

import logging

import pytest
import sqlglot
from sqlglot import exp

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import InvColumn, from_duckdb
from core2.bootstrap.profiler import _readable
from core2.compile.compiler import compile_query
from core2.resolve.resolver import Context, resolve
from core2.warehouse import dialect as D
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.framework import materialize

logging.getLogger("sqlglot").setLevel(logging.ERROR)

DB_TYPES = {"tsql": "azure_sql", "snowflake": "snowflake", "oracle": "oracle"}
# Oracle's TO_CHAR formats and interval arithmetic do not carry back to DuckDB:
# these are the domains whose every Learn query does.
CARRIED_BACK = {"tsql": domains.available(), "snowflake": domains.available(), "oracle": ["hr", "subscriptions"]}


def _to_duckdb(tree: exp.Expression, dialect: str) -> str:
    """The query as DuckDB reads it, where sqlglot leaves a warehouse's own idiom untranslated."""
    units = {"MM": "month", "Q": "quarter", "IW": "week", "YYYY": "year", "DD": "day"}

    def fix(node: exp.Expression) -> exp.Expression:
        if dialect == "tsql":
            if isinstance(node, exp.Parameter) and "DATEFIRST" in node.sql().upper():
                return exp.Literal.number(7)        # Azure SQL's default: Sunday is day 1
            if isinstance(node, exp.Extract) and node.this.name.upper() == "DAYOFWEEK":
                return exp.Paren(this=exp.Add(this=node.copy(), expression=exp.Literal.number(1)))
        if dialect == "oracle":
            if isinstance(node, exp.Anonymous) and node.name.upper() == "TRUNC" and len(node.expressions) == 1:
                return exp.DateTrunc(this=node.expressions[0].copy(), unit=exp.Literal.string("day"))
            if isinstance(node, exp.DateTrunc) and isinstance(node.args.get("unit"), exp.Literal):
                unit = units.get(node.args["unit"].name.upper())
                if unit:
                    node.set("unit", exp.Literal.string(unit))
        return node

    return tree.transform(fix).sql(dialect="duckdb")


def azure_sql_refuses(tree: exp.Expression, types: dict[str, str]) -> list[str]:
    """What Azure SQL would refuse in this query and DuckDB runs: its own type rules."""
    refused = []
    for mod in tree.find_all(exp.Mod):
        for side in (mod.this, mod.expression):
            if any(cast.to.this.name in ("FLOAT", "DOUBLE", "REAL") for cast in side.find_all(exp.Cast)) or \
                    any(types.get(column.name.upper()) == "float" for column in side.find_all(exp.Column)):
                refused.append(f"the remainder of a float (402): {mod.sql(dialect='tsql')}")
    for total in tree.find_all(exp.Sum):
        inner = total.this
        if total.find_ancestor(exp.Window) or isinstance(inner, exp.Case) or \
                (isinstance(inner, exp.Cast) and inner.to.this.name in ("BIGINT", "DECIMAL", "FLOAT", "DOUBLE")):
            continue
        if any(types.get(column.name.upper()) == "integer" for column in inner.find_all(exp.Column)):
            refused.append(f"an int sum that overflows past 2,147,483,647 (8115): {total.sql(dialect='tsql')}")
    return refused


class Warehouse(DuckDBWarehouse):
    """A test warehouse that speaks another warehouse's SQL."""

    def __init__(self, con, dialect: str, types: dict[str, str]):
        super().__init__(con)
        self.dialect, self.db_type = dialect, DB_TYPES[dialect]
        self.types, self.refused = types, []

    def query(self, sql, *, max_rows=None):
        tree = sqlglot.parse_one(sql, read=self.dialect)
        if self.dialect == "tsql":
            self.refused += azure_sql_refuses(tree, self.types)
        return DuckDBWarehouse.query(self, _to_duckdb(tree, self.dialect), max_rows=max_rows)


def _built(name):
    built = materialize(domains.build(name), "warehouse")
    inventory = from_duckdb(DuckDBWarehouse(built.con), declared_fks=built.declared_fks)
    types = {c.name.upper(): t.type_of(c.name) for t in inventory.tables.values() for c in t.columns}
    return built, inventory, types


def _learned(model):
    return sorted(model.tables), sorted(model.measures), sorted(model.date_roles), \
        sorted((m.key, m.additivity) for m in model.measures.values())


@pytest.mark.parametrize("dialect, name", [(d, n) for d, names in CARRIED_BACK.items() for n in names])
def test_learn_runs_in_the_warehouses_own_sql_and_learns_what_duckdb_learns(dialect, name):
    built, inventory, types = _built(name)
    on_duckdb = build_model(DuckDBWarehouse(built.con), inventory, options=BuildOptions(workers=1))
    warehouse = Warehouse(built.con, dialect, types)
    model = build_model(warehouse, inventory, options=BuildOptions(workers=1))
    assert not warehouse.refused, "\n".join(warehouse.refused)
    assert _learned(model) == _learned(on_duckdb)
    assert len(warehouse.log) > 20


@pytest.mark.parametrize("granularity, key, months", [("day", 20260131, 2026 * 12 + 1), ("month", 202602, 2026 * 12 + 2)])
def test_a_date_keys_month_is_read_without_a_float(granularity, key, months):
    """A date key as a month count -- how far a load stamp trails it -- is the same number, float-free."""
    import duckdb

    from core2.bootstrap.dates import DateCandidate, _months

    candidate = DateCandidate(table="t", column="DT_KEY", via_calendar=None, granularity=granularity)
    count = _months(exp.column("DT_KEY"), candidate, "integer", "tsql")
    tree = sqlglot.parse_one(f"SELECT {count.sql(dialect='tsql')} FROM t", read="tsql")
    assert not azure_sql_refuses(tree, {"DT_KEY": "integer"})
    duck = _months(exp.column("DT_KEY"), candidate, "integer", "duckdb").sql(dialect="duckdb")
    assert duckdb.connect().execute(f"SELECT {duck} FROM (SELECT {key} AS DT_KEY)").fetchone()[0] == months


def _sql(name, case_id, dialect):
    from evals.core2.compile_eval import _Words, golden, learn

    domain = domains.build(name)
    built, model = learn(domain, "warehouse")
    case = next(c for c in golden(name)["questions"] if c["id"] == case_id)
    import datetime as dt

    logical = resolve(_Words(model, built).plan(case["plan"]), model,
                      Context(today=dt.date.fromisoformat(str(golden(name)["today"]))))
    return compile_query(logical, model, dialect).sql


def test_an_integer_quantity_is_summed_as_a_bigint_on_azure_sql_only():
    ordered = _sql("purchasing", "purchasing-008", "tsql")   # SUM of ordered_qty, an INTEGER
    assert "CAST(" in ordered and "AS BIGINT)" in ordered
    assert "BIGINT" not in _sql("purchasing", "purchasing-008", "snowflake")
    assert "BIGINT" not in _sql("purchasing", "purchasing-001", "tsql")   # line_amount is a DECIMAL


class _Refusing(DuckDBWarehouse):
    """A warehouse that will not read one column, as Azure SQL will not sort an ntext."""

    def __init__(self, con, column):
        super().__init__(con)
        self.column = column

    def query(self, sql, *, max_rows=None):
        if self.column.upper() in sql.upper():
            raise RuntimeError(f"[42000] The ntext data type cannot be selected as DISTINCT ({self.column})")
        return super().query(sql, max_rows=max_rows)


def test_a_column_the_database_will_not_read_is_left_out_and_named():
    built, inventory, _types = _built("retail")
    table, column = built.c("customers.segment")
    model = build_model(_Refusing(built.con, column), inventory, options=BuildOptions(workers=1))
    note = next(n for n in model.notes if f"{table}.{column}" in n)
    assert "was left out" in note and "ntext" in note
    # Everything else is learned as before.
    on_duckdb = build_model(DuckDBWarehouse(built.con), inventory, options=BuildOptions(workers=1))
    assert sorted(model.measures) == sorted(on_duckdb.measures)
    assert sorted(model.tables) == sorted(on_duckdb.tables)


def test_a_table_the_database_will_not_read_at_all_still_stops_learn():
    built, inventory, _types = _built("retail")
    table = built.t("customers")

    class Down(DuckDBWarehouse):
        def query(self, sql, *, max_rows=None):
            if table.upper() in sql.upper():
                raise RuntimeError("connection lost")
            return super().query(sql, max_rows=max_rows)

    from core2.bootstrap.profiler import profile_table

    import dataclasses

    key = next(k for k, t in inventory.tables.items() if t.name == table)
    known = dataclasses.replace(inventory.tables[key], row_count=40)   # no row count asked first
    with pytest.raises(RuntimeError, match="connection lost"):
        profile_table(Down(built.con), known)
    with pytest.raises(RuntimeError, match="connection lost"):
        build_model(Down(built.con), inventory, options=BuildOptions(workers=1))


@pytest.mark.parametrize("raw, cast", [("NTEXT", "NVARCHAR(MAX)"), ("TEXT", "NVARCHAR(MAX)"),
                                       ("UNIQUEIDENTIFIER", "CHAR(36)"), ("NVARCHAR(40)", None)])
def test_old_azure_sql_types_are_read_as_text_it_can_compare(raw, cast):
    column = InvColumn(name="NOTE_TXT", raw_type=raw, data_type="text")
    sql = _readable(column, "tsql").sql(dialect="tsql")
    assert sql == (f"CAST(NOTE_TXT AS {cast})" if cast else "NOTE_TXT")
    assert _readable(column, "snowflake").sql(dialect="snowflake") == "NOTE_TXT"
