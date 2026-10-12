"""A date key's month is read in every database's SQL, whatever SQL the server wrote before.

Learn reads a yyyymmdd key's month as FLOOR(key / 100) % 100 (PostgreSQL casts the floor back to a
whole number). sqlglot keeps each database's table of how to write each expression from the first SQL
it writes for that database; a server that wrote any SQL before the new core's dialect module was
loaded -- the question pipeline checking a first question -- kept a table without it, and every key
read was written FLOOR_DIV(key, 100), which no database has. Learn then left out every table with a
date key ("Scalar Function with name floor_div does not exist"). Each case runs in a fresh process so
that the SQL written first really is first.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = """
import sqlglot
for name in ("duckdb", "tsql", "snowflake", "oracle", "postgres"):
    sqlglot.parse_one("SELECT FLOOR(a / 100) FROM t").sql(dialect=name)   # the SQL written first
from sqlglot import exp
from core2.warehouse import dialect
for name in ("duckdb", "tsql", "snowflake", "oracle", "postgres"):
    print(name, dialect.floor_div(exp.column("k"), 100).sql(dialect=name))
"""


@pytest.fixture(scope="module")
def written() -> dict[str, str]:
    done = subprocess.run([sys.executable, "-c", SCRIPT], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    return dict(line.split(" ", 1) for line in done.stdout.strip().splitlines())


@pytest.mark.parametrize("name", ["duckdb", "tsql", "snowflake", "oracle"])
def test_the_month_of_a_key_is_a_floor_in_each_database(written, name):
    assert written[name] == "FLOOR(k / 100)"


def test_postgres_casts_the_floor_back_to_a_whole_number(written):
    assert written["postgres"] == "CAST(FLOOR(k / 100) AS BIGINT)"


def test_it_runs_on_duckdb_after_other_sql_was_written():
    import duckdb
    import sqlglot
    from sqlglot import exp

    sqlglot.parse_one("SELECT 1").sql(dialect="duckdb")
    from core2.warehouse import dialect

    sql = exp.select(exp.Mod(this=dialect.floor_div(exp.Literal.number(20260131), 100),
                             expression=exp.Literal.number(100))).sql(dialect="duckdb")
    assert duckdb.connect().execute(sql).fetchone()[0] == 1
