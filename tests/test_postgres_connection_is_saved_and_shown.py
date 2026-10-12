"""A PostgreSQL connection is made on the Databases page, kept, shown, and edited like the others.

The store refused one: db_config's CHECK listed three types, and an existing store keeps the table it
was created with. A store made before is now widened when QueryBot starts, without losing the
workspaces that point at a connection (the table is rebuilt with SQLite's foreign keys off, or they
would have been emptied). A QueryBot that keeps its own store on PostgreSQL could not start on a new
database either: a literal % in a statement with parameters, and the id of a new row, both failed there.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.test_snowflake_signs_in_as_a_service_user import _request, admin  # noqa: F401 - the fixture

FORM = {"db_type": "postgres", "name": "Analytics", "pg_host": "db.example.com", "pg_port": "6432",
        "pg_database": "analytics", "pg_schemas": " sales,finance , ", "pg_user": "querybot_reader",
        "pg_password": "s3cret", "pg_sslmode": "require"}


def _save(routes, fields: dict):
    with patch.object(routes, "_is_auth", return_value=True):
        return asyncio.run(routes.database_save(_request("/admin/databases/save", fields), routes.BackgroundTasks()))


def _page(routes) -> str:
    import html

    with patch.object(routes, "_is_auth", return_value=True):
        page = asyncio.run(routes.databases_page(_request("/admin/databases", method="GET"))).body.decode()
    return html.unescape(page)


def test_the_form_offers_postgres_and_reads_its_fields(admin):  # noqa: F811
    from admin.routes import _db_credentials_from_form

    assert '<option value="postgres">PostgreSQL</option>' in _page(admin)
    db_type, name, _id, creds = _db_credentials_from_form(FORM)
    assert (db_type, name) == ("postgres", "Analytics")
    assert {k: creds[k] for k in ("host", "port", "database", "user", "password", "sslmode", "schemas")} == {
        "host": "db.example.com", "port": 6432, "database": "analytics", "user": "querybot_reader",
        "password": "s3cret", "sslmode": "require", "schemas": "sales, finance"}
    _, _, _, odd = _db_credentials_from_form({**FORM, "pg_port": "", "pg_sslmode": "whatever"})
    assert odd["port"] == 5432 and odd["sslmode"] == "prefer"


def test_a_saved_connection_is_listed_and_edited_without_its_password(admin):  # noqa: F811
    response = _save(admin, FORM)
    assert response.status_code == 303 and "error" not in response.headers["location"]
    saved = admin.store.list_db_configs()
    assert [(d["db_type"], d["label"], d["dialect"]) for d in saved] == [("postgres", "PostgreSQL", "postgres")]
    page = _page(admin)
    assert "db.example.com:6432 / analytics / sales, finance" in page
    row = page.split('data-db-type="postgres"', 1)[1].split(">", 1)[0]
    edit = json.loads(row.split("data-db-creds='", 1)[1].split("'", 1)[0])
    assert edit == {"host": "db.example.com", "port": "6432", "database": "analytics", "schemas": "sales, finance",
                    "user": "querybot_reader", "sslmode": "require"}
    assert "s3cret" not in page

    # Saved again with the password left blank: the stored one is kept.
    _save(admin, {**FORM, "db_id": str(saved[0]["id"]), "pg_password": ""})
    assert admin.store.get_db_config(saved[0]["id"])["credentials"]["password"] == "s3cret"


def test_a_store_made_before_postgres_takes_one_and_keeps_its_workspaces(tmp_path, monkeypatch):
    path = tmp_path / "querybot.db"
    monkeypatch.setenv("QUERYBOT_DB_PATH", str(path))
    monkeypatch.setenv("DB_PATH", str(path))
    import store
    import store.crypto

    monkeypatch.setattr(store.crypto, "KEY_FILE", tmp_path / ".key")
    store.init_db()
    # The table as a store created before PostgreSQL was added has it.
    con = sqlite3.connect(path)
    old = con.execute("SELECT sql FROM sqlite_master WHERE name = 'db_config'").fetchone()[0]
    assert ",'postgres'" in old
    con.execute("PRAGMA foreign_keys=OFF")
    con.execute("DROP TABLE db_config")
    con.execute(old.replace(",'postgres'", ""))
    con.commit()
    con.close()
    azure = store.save_db_config("azure_sql", "Warehouse", {"server": "s", "user": "u", "password": "p"})
    store.upsert_client("acct-before", "web")
    store.update_client_meta("acct-before", db_config_id=azure)
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        store.save_db_config("postgres", "Analytics", {"host": "h", "database": "d", "user": "u", "password": "p"})

    store.init_db()                 # QueryBot starts after the update
    pg = store.save_db_config("postgres", "Analytics", {"host": "h", "database": "d", "user": "u", "password": "p"})
    assert store.get_db_config(pg)["db_type"] == "postgres"
    assert store.get_client("acct-before")["db_config_id"] == azure
    assert store.get_db_config(azure)["credentials"]["server"] == "s"
    store.init_db()                 # and again: nothing more to do


def test_querybot_starts_on_a_new_postgres_store_and_saves_a_connection(tmp_path):
    from tests import postgres_warehouse as pg

    creds = pg.credentials()
    import psycopg2

    name = f"qb_store_{uuid.uuid4().hex[:10]}"
    admin_conn = psycopg2.connect(host=creds["host"], port=creds["port"], dbname=creds["database"],
                                  user=creds["user"], password=creds["password"])
    admin_conn.autocommit = True
    admin_conn.cursor().execute(f'CREATE DATABASE "{name}" ENCODING \'UTF8\' TEMPLATE template0')
    try:
        url = f"postgresql://{creds['user']}:{creds['password']}@{creds['host']}:{creds['port']}/{name}"
        script = textwrap.dedent("""
            import logging, sys
            from pathlib import Path
            failures = []
            class Catch(logging.Handler):
                def emit(self, record):
                    if record.levelno >= logging.ERROR:
                        failures.append(record.getMessage())
            logging.getLogger().addHandler(Catch())
            import store, store.crypto
            store.crypto.KEY_FILE = Path(sys.argv[1]) / ".key"
            store.init_db()
            a = store.save_db_config("azure_sql", "Warehouse", {"server": "s", "user": "u", "password": "p"})
            b = store.save_db_config("postgres", "Analytics", {"host": "h", "database": "d", "user": "u", "password": "p"})
            store.upsert_client("acct-pg", "web")
            store.update_client_meta("acct-pg", db_config_id=b)
            assert isinstance(a, int) and isinstance(b, int) and a != b, (a, b)
            assert store.get_client("acct-pg")["db_config_id"] == b
            assert store.get_db_config(b)["label"] == "PostgreSQL"
            assert not failures, failures
            print("ok")
        """)
        env = {**os.environ, "DATABASE_URL": url, "QUERYBOT_DB_PATH": str(tmp_path / "unused.db")}
        done = subprocess.run([sys.executable, "-c", script, str(tmp_path)], env=env, capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parents[1], timeout=600)
        assert done.returncode == 0 and done.stdout.strip().endswith("ok"), done.stderr[-3000:]
    finally:
        admin_conn.cursor().execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        admin_conn.close()
