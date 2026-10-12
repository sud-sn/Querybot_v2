"""PostgreSQL is a database QueryBot answers from, like Azure SQL, Snowflake and Oracle.

Run against a real PostgreSQL server (tests/postgres_warehouse.py): the synthetic retail warehouse is
copied into a schema of its own, its table names in capitals and small letters as created, its keys
declared. Then QueryBot does what it does with any database: tests the connection, discovers the
tables, learns them, and answers the golden questions, each answer compared with the reference rows.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from tests import postgres_warehouse as pg


@pytest.fixture(scope="module")
def creds():
    return pg.credentials()


# Every synthetic warehouse with names in small letters, and retail with capitals (FactOrderLines).
WAREHOUSES = [("retail", "descriptive"), ("retail", "pascal"), ("inventory", "descriptive"),
              ("finance", "descriptive"), ("purchasing", "descriptive"), ("hr", "descriptive"),
              ("subscriptions", "descriptive")]


@pytest.fixture(scope="module", params=WAREHOUSES, ids=["-".join(w) for w in WAREHOUSES])
def retail(request, creds):
    """A synthetic warehouse in PostgreSQL, in a schema of its own."""
    from evals.core2 import domains
    from evals.core2.framework import materialize

    name, style = request.param
    built = materialize(domains.build(name), style)
    with pg.copied(built, creds) as schema:
        yield built, {**creds, "schemas": schema}, schema


def test_the_connection_test_names_the_database_and_what_the_login_may_change(creds):
    from core.schema import test_connection

    found = test_connection(creds, "postgres")
    assert found["database"] == creds["database"] and found["user"] == creds["user"]
    assert found["version"].startswith("PostgreSQL ")
    assert "write_permissions" in found          # the test login may create tables: the admin is told


def test_a_query_that_would_change_data_is_refused_by_the_server_itself(retail):
    import psycopg2

    from core.schema import run_query

    built, creds, schema = retail
    table = next(iter(built.tables.values()))
    with pytest.raises(psycopg2.Error, match="read-only transaction"):
        run_query(creds, "postgres", f'DELETE FROM "{schema}"."{table}"')


def test_a_query_past_its_time_limit_is_stopped_and_said_so(creds):
    from core.schema import run_query
    from core2.bootstrap.journal import timed_out

    with pytest.raises(Exception) as stopped:
        run_query({**creds, "query_timeout_seconds": 1}, "postgres", "SELECT pg_sleep(3)")
    assert timed_out(stopped.value)


def test_discovery_reads_every_table_its_columns_keys_and_links(retail, tmp_path):
    from core.schema import discover_and_write

    built, creds, schema = retail
    count = discover_and_write(creds, "postgres", str(tmp_path))
    found = json.loads((tmp_path / "_schema.json").read_text())
    tables = {k: v for k, v in found.items() if not k.startswith("__")}
    assert count == len(built.tables) == len(tables)
    by_name = {v["schema"] + "." + k.split(".")[-1]: v for k, v in tables.items()}
    for physical in built.tables.values():
        entry = by_name[f"{schema}.{physical}"]
        expected = built.con.execute(f'SELECT COUNT(*) FROM "{physical}"').fetchone()[0]
        assert entry["row_count"] == expected
        assert [c["name"] for c in entry["columns"]] == [
            r[0] for r in built.con.execute("SELECT column_name FROM information_schema.columns "
                                            "WHERE table_name = ? ORDER BY ordinal_position", [physical]).fetchall()]
        assert entry["pk_columns"] == built.declared_pks.get(physical, [])
    declared = {(f["parent_table"], f["parent_col"], f["ref_table"], f["ref_col"]) for f in built.declared_fks}
    read = {(f["parent_table"], f["parent_col"], f["ref_table"], f["ref_col"])
            for f in found.get("__db_fk_constraints__", [])}
    assert declared and read == declared
    # The KB tells the AI the name to write: quoted where PostgreSQL would otherwise fold it.
    some = next(p for p in built.tables.values())
    text = next(tmp_path.glob("*.md")).read_text()
    assert "**SQL table name:**" in text
    if any(ch.isupper() for ch in some):
        assert f'"{schema}"' not in text and '."' in text   # schema in small letters bare, the table quoted


@pytest.fixture(scope="module")
def learned(retail, tmp_path_factory):
    """Learn on the PostgreSQL copy, from what discovery wrote, as production runs it."""
    from core.schema import discover_and_write
    from core2.bootstrap.build import BuildOptions, build_model
    from core2.bootstrap.inventory import from_schema_json
    from core2.warehouse.querybot import QueryBotWarehouse

    built, creds, _schema = retail
    folder = tmp_path_factory.mktemp("pg_schema")
    discover_and_write(creds, "postgres", str(folder))
    inventory = from_schema_json(json.loads((folder / "_schema.json").read_text()), "postgres")
    lines: list[str] = []
    with QueryBotWarehouse("postgres", creds) as warehouse:
        model = build_model(warehouse, inventory, client_id="pg-test",
                            options=BuildOptions(workers=2, progress=lines.append))
    return built, creds, model, lines


def test_learn_on_postgres_learns_what_it_learns_on_any_warehouse(learned):
    from evals.core2.compile_eval import learn

    built, _creds, model, lines = learned
    _, reference = learn(built.domain, built.style)
    assert {t.name for t in model.tables.values()} == {t.name for t in reference.tables.values()}

    def links(m):
        return {(m.tables[j.from_table].name, m.tables[j.to_table].name) for j in m.joins.values()}

    assert links(model) == links(reference)
    def measures(m):
        return {(m.tables[x.table].name if x.table in m.tables else "", x.slug, x.kind, x.additivity)
                for x in m.measures.values()}

    assert measures(model) == measures(reference)
    assert not [line for line in lines if line.startswith("Left out ")], lines


def test_every_golden_question_gives_the_reference_rows_on_postgres(learned):
    from core2.compile.compiler import compile_query
    from core2.resolve.resolver import Context, resolve
    from core2.warehouse.querybot import QueryBotWarehouse
    from core2.warehouse.runner import DuckDBWarehouse
    from evals.core2.compile_eval import Untranslatable, _Words, golden, same_rows

    built, creds, model, _lines = learned
    cases = golden(built.domain.name)
    today = dt.date.fromisoformat(str(cases["today"]))
    words = _Words(model, built)
    from evals.core2.framework import materialize

    reference = DuckDBWarehouse(materialize(built.domain, "descriptive").con)   # the reference SQL's own names
    ran, wrong = 0, []
    with QueryBotWarehouse("postgres", creds) as warehouse:
        for case in cases["questions"]:
            if not case.get("plan") or not case.get("reference_sql"):
                continue
            try:
                logical = resolve(words.plan(case["plan"]), model, Context(today=today))
            except Untranslatable:
                continue
            sql = compile_query(logical, model, "postgres").sql
            got = warehouse.query(sql)
            ran += 1
            expected = reference.query(case["reference_sql"])
            diff = same_rows(expected.columns, expected.rows, got.columns, got.rows,
                             order_matters=bool((case.get("expect") or {}).get("order_matters")))
            if diff:
                wrong.append(f"{case['id']}: {diff}\n{sql}")
    assert ran >= 5
    assert not wrong, "\n\n".join(wrong)
