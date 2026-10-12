"""A real PostgreSQL server for the tests that need one, and the synthetic warehouses copied into it.

Set QUERYBOT_TEST_POSTGRES_URL (postgresql://user:password@host:port/database) to a server the tests may
create schemas in; CI starts one. Without it, or when the server does not answer, those tests are skipped.
Each copy goes into a schema of its own, dropped when the test module ends.
"""

from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from urllib.parse import unquote, urlparse

import pytest

URL_SETTING = "QUERYBOT_TEST_POSTGRES_URL"

# DuckDB's type names as PostgreSQL's.
_TYPES = {"DOUBLE": "double precision", "FLOAT": "real", "HUGEINT": "numeric(38,0)", "UBIGINT": "numeric(20,0)",
          "TINYINT": "smallint", "UTINYINT": "smallint", "USMALLINT": "integer", "UINTEGER": "bigint",
          "TIMESTAMP WITH TIME ZONE": "timestamp with time zone", "BLOB": "bytea"}


def credentials() -> dict:
    """The server's settings in the shape QueryBot keeps them for a PostgreSQL connection."""
    url = os.getenv(URL_SETTING, "").strip()
    if not url:
        pytest.skip(f"{URL_SETTING} is not set: no PostgreSQL server to test against")
    parsed = urlparse(url)
    creds = {"host": parsed.hostname or "localhost", "port": parsed.port or 5432,
             "database": (parsed.path or "/postgres").lstrip("/") or "postgres",
             "user": unquote(parsed.username or "postgres"), "password": unquote(parsed.password or ""),
             "sslmode": "disable"}
    try:
        import psycopg2

        psycopg2.connect(host=creds["host"], port=creds["port"], dbname=creds["database"], user=creds["user"],
                         password=creds["password"], connect_timeout=5).close()
    except Exception as exc:  # noqa: BLE001 - no server: the tests that need one are skipped, not failed
        pytest.skip(f"PostgreSQL at {creds['host']}:{creds['port']} does not answer: {exc}")
    return creds


def _connect(creds: dict):
    import psycopg2

    conn = psycopg2.connect(host=creds["host"], port=creds["port"], dbname=creds["database"], user=creds["user"],
                            password=creds["password"])
    conn.autocommit = True
    return conn


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _pg_type(duck: str) -> str:
    duck = duck.upper()
    return _TYPES.get(duck, duck.lower())


@contextmanager
def copied(built, creds: dict, *, foreign_keys: bool = True):
    """The synthetic warehouse ``built`` (evals.core2.framework.materialize) as a PostgreSQL schema.

    Table and column names are kept exactly, capitals included; the primary keys and the declared
    foreign keys become real constraints, so discovery reads them as it would on any server.
    Yields the schema's name.
    """
    from psycopg2.extras import execute_values

    schema = f"qb_test_{uuid.uuid4().hex[:10]}"
    conn = _connect(creds)
    cur = conn.cursor()
    cur.execute(f"CREATE SCHEMA {_q(schema)}")
    try:
        for physical in built.tables.values():
            cols = built.con.execute(
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = ? "
                "ORDER BY ordinal_position", [physical]).fetchall()
            defs = [f"{_q(c)} {_pg_type(t)}" for c, t in cols]
            pk = built.declared_pks.get(physical)
            if pk:
                defs.append("PRIMARY KEY (" + ", ".join(_q(c) for c in pk) + ")")
            cur.execute(f"CREATE TABLE {_q(schema)}.{_q(physical)} ({', '.join(defs)})")
            rows = built.con.execute(f'SELECT * FROM "{physical}"').fetchall()
            if rows:
                execute_values(cur, f"INSERT INTO {_q(schema)}.{_q(physical)} VALUES %s", rows, page_size=5000)
        if foreign_keys:
            grouped: dict[tuple[str, str], list[dict]] = {}
            for fk in built.declared_fks:
                grouped.setdefault((fk["parent_table"], fk["constraint_name"]), []).append(fk)
            for (parent, _name), pairs in grouped.items():
                pairs.sort(key=lambda p: p["ordinal"])
                ref = pairs[0]["ref_table"]
                cur.execute(
                    f"ALTER TABLE {_q(schema)}.{_q(parent)} ADD FOREIGN KEY "
                    f"({', '.join(_q(p['parent_col']) for p in pairs)}) REFERENCES {_q(schema)}.{_q(ref)} "
                    f"({', '.join(_q(p['ref_col']) for p in pairs)}) NOT VALID")
        cur.execute(f"ANALYZE")
        yield schema
    finally:
        cur.execute(f"DROP SCHEMA {_q(schema)} CASCADE")
        conn.close()
