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
from evals.core2.warehouse_sql import SpeakingWarehouse, Types, azure_sql_refuses

logging.getLogger("sqlglot").setLevel(logging.ERROR)

# Oracle's TO_CHAR formats and interval arithmetic do not carry back to DuckDB:
# these are the domains whose every Learn query does.
CARRIED_BACK = {"tsql": domains.available(), "snowflake": domains.available(), "oracle": ["hr", "subscriptions"]}


def _built(name):
    built = materialize(domains.build(name), "warehouse")
    inventory = from_duckdb(DuckDBWarehouse(built.con), declared_fks=built.declared_fks)
    return built, inventory, Types.of(inventory)


def _learned(model):
    return sorted(model.tables), sorted(model.measures), sorted(model.date_roles), \
        sorted((m.key, m.additivity) for m in model.measures.values())


@pytest.mark.parametrize("dialect, name", [(d, n) for d, names in CARRIED_BACK.items() for n in names])
def test_learn_runs_in_the_warehouses_own_sql_and_learns_what_duckdb_learns(dialect, name):
    built, inventory, types = _built(name)
    on_duckdb = build_model(DuckDBWarehouse(built.con), inventory, options=BuildOptions(workers=1))
    warehouse = SpeakingWarehouse(built.con, dialect, types)
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
    assert not azure_sql_refuses(tree, Types(data={"DT_KEY": "integer"}))
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


def _refusing_table(con, table):
    class Refusing(DuckDBWarehouse):
        def query(self, sql, *, max_rows=None):
            if table.upper() in sql.upper():
                raise RuntimeError(f"[42000] The SELECT permission was denied on the object '{table}' (229)")
            return super().query(sql, max_rows=max_rows)

    return Refusing(con)


def test_a_table_the_database_will_not_read_is_left_out_and_the_rest_is_learned():
    import dataclasses

    from core2.bootstrap.profiler import profile_table

    built, inventory, _types = _built("retail")
    table = built.t("customers")
    key = next(k for k, t in inventory.tables.items() if t.name == table)
    known = dataclasses.replace(inventory.tables[key], row_count=40)   # no row count asked first
    with pytest.raises(RuntimeError, match="permission was denied"):
        profile_table(_refusing_table(built.con, table), known)   # the table, not one of its columns

    lines: list[str] = []
    model = build_model(_refusing_table(built.con, table), inventory,
                        options=BuildOptions(workers=1, progress=lines.append))
    assert table not in {t.name for t in model.tables.values()}
    assert [n for n in model.notes if table in n] == \
        [f"The table {table} was left out: the database refused it ([42000] The SELECT permission was denied "
         f"on the object '{table}' (229))."]
    assert any(line.startswith(f"Left out the table {table}") for line in lines)
    assert len(model.tables) == len(inventory.tables) - 1 and model.measures


def test_a_database_that_stops_answering_stops_learn():
    class Gone(DuckDBWarehouse):
        calls = 0

        def query(self, sql, *, max_rows=None):
            Gone.calls += 1
            if Gone.calls > 3:
                raise RuntimeError("[08S01] Communication link failure")
            return super().query(sql, max_rows=max_rows)

    built, inventory, _types = _built("retail")
    lines: list[str] = []
    with pytest.raises(RuntimeError, match=r"^\[08S01\] Communication link failure$"):
        build_model(Gone(built.con), inventory, options=BuildOptions(workers=1, progress=lines.append))
    # A database that stopped answering refused nothing: nothing is said to be left out.
    assert not [line for line in lines if line.startswith("Left out")]


def test_a_database_that_refuses_every_table_says_so():
    class Locked(DuckDBWarehouse):
        def query(self, sql, *, max_rows=None):
            if sql.strip() != "SELECT 1":
                raise RuntimeError("[42000] The SELECT permission was denied (229)")
            return super().query(sql, max_rows=max_rows)

    built, inventory, _types = _built("retail")
    with pytest.raises(RuntimeError, match="The database refused every table"):
        build_model(Locked(built.con), inventory, options=BuildOptions(workers=1))


@pytest.mark.parametrize("raw, cast", [("NTEXT", "NVARCHAR(MAX)"), ("TEXT", "NVARCHAR(MAX)"),
                                       ("UNIQUEIDENTIFIER", "CHAR(36)"), ("NVARCHAR(40)", None)])
def test_old_azure_sql_types_are_read_as_text_it_can_compare(raw, cast):
    column = InvColumn(name="NOTE_TXT", raw_type=raw, data_type="text")
    sql = _readable(column, "tsql").sql(dialect="tsql")
    assert sql == (f"CAST(NOTE_TXT AS {cast})" if cast else "NOTE_TXT")
    assert _readable(column, "snowflake").sql(dialect="snowflake") == "NOTE_TXT"


@pytest.mark.parametrize("raws, collated", [(("nvarchar(20)", "VARCHAR(10)"), True), (("char(3)", "nchar(3)"), True),
                                            (("uniqueidentifier", "uniqueidentifier"), False),
                                            (("nvarchar(36)", "uniqueidentifier"), False)])
def test_text_keys_are_compared_in_the_databases_collation_where_they_have_one(raws, collated):
    sql = D.same_text(exp.column("A", table="f"), exp.column("K", table="t"), "tsql", raws).sql(dialect="tsql")
    assert sql == ("f.A COLLATE DATABASE_DEFAULT = t.K COLLATE DATABASE_DEFAULT" if collated else "f.A = t.K")
    assert D.same_text(exp.column("A"), exp.column("K"), "snowflake", raws).sql(dialect="snowflake") == "A = K"


def test_every_golden_answer_query_is_one_azure_sql_runs():
    """The answers' own SQL, for every golden question, meets the same rules."""
    import datetime as dt

    from evals.core2.compile_eval import Untranslatable, _Words, golden, learn

    refused, total = [], 0
    for name in domains.available():
        if not golden(name):
            continue
        cases = golden(name)
        built, model = learn(domains.build(name), "warehouse")
        types = Types.of(from_duckdb(DuckDBWarehouse(built.con), declared_fks=built.declared_fks))
        words = _Words(model, built)
        for case in cases["questions"]:
            if not case.get("plan"):
                continue
            try:
                logical = resolve(words.plan(case["plan"]), model,
                                  Context(today=dt.date.fromisoformat(str(cases["today"]))))
            except Untranslatable:
                continue
            total += 1
            sql = compile_query(logical, model, "tsql").sql
            refused += [f"{case['id']}: {r}" for r in azure_sql_refuses(sqlglot.parse_one(sql, read="tsql"), types)]
    assert total > 100 and not refused, "\n".join(refused)
