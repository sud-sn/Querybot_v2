"""
Browse schemas lists every Snowflake table and view the connection's role can see.

It found none on a working connection. The listing left out the PUBLIC schema, where most accounts keep their
tables; it read INFORMATION_SCHEMA, which needs a warehouse the role can use; a database it could not read was
skipped without a word; and a database name it did not find fell back to the account's first database. The page
then said "No schemas or tables were visible with these credentials." and nothing else.

It now lists with SHOW TABLES and SHOW VIEWS (no warehouse needed), keeps PUBLIC, leaves out only Snowflake's
own shared databases, finds the named database whatever its case, and when nothing can be listed says why: the
role in use, the database it could not see or read, and the grant to ask the Snowflake admin for.

The real discovery, error message and route; the Snowflake account is a stand-in.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch
from urllib.parse import urlencode

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from starlette.requests import Request

from core.schema_discovery import NothingVisible, _discover_snowflake


class _Account:
    """A Snowflake account as its SHOW commands and CURRENT_ROLE() answer one role, with no warehouse in use."""

    def __init__(self, objects=(), role="QUERYBOT_READER", hidden=(), unreadable=()):
        self.objects = list(objects)          # (database, schema, name, "table" | "view")
        self.role = role
        self.hidden = set(hidden)             # databases SHOW DATABASES lists but whose objects cannot be read
        self.unreadable = set(unreadable)
        self.ran: list[str] = []
        self.closed = False
        self._rows: list[dict] = []

    def databases(self):
        seen = []
        for database, *_ in self.objects:
            if database not in seen:
                seen.append(database)
        return seen + [d for d in self.unreadable if d not in seen]

    def cursor(self, *_):
        return self

    def execute(self, sql: str):
        self.ran.append(sql)
        text = " ".join(sql.split())
        if text.upper().startswith("SELECT CURRENT_ROLE()"):
            self._rows = [{"ROLE": self.role}]
        elif text.upper() == "SHOW DATABASES":
            self._rows = [{"name": d, "created_on": None} for d in self.databases()]
        elif text.upper().startswith(("SHOW TABLES IN DATABASE ", "SHOW VIEWS IN DATABASE ")):
            database = text.split(" IN DATABASE ", 1)[1].strip()[1:-1].replace('""', '"')
            if database in self.unreadable:
                raise RuntimeError(f"002003 (02000): SQL compilation error:\nDatabase '{database}' does not exist "
                                   "or not authorized.")
            kind = "table" if text.upper().startswith("SHOW TABLES") else "view"
            self._rows = [{"name": name, "schema_name": schema, "database_name": db, "kind": kind.upper()}
                          for db, schema, name, k in self.objects if db == database and k == kind]
        else:
            raise RuntimeError("000606 (57P03): No active warehouse selected in the current session.  Select an "
                               "active warehouse with the 'use warehouse' command.")

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def close(self):
        self.closed = True


def _browse(account: _Account, **cfg) -> dict:
    with patch("core.schema._sf_connect", return_value=account):
        return _discover_snowflake({"account": "org-acct", "user": "QUERYBOT_SVC", **cfg})


SALES = [
    ("SALES_DB", "PUBLIC", "ORDERS", "table"),
    ("SALES_DB", "PUBLIC", "CUSTOMERS", "table"),
    ("SALES_DB", "PUBLIC", "V_MONTHLY", "view"),
    ("SALES_DB", "FINANCE", "LEDGER", "table"),
    ("SALES_DB", "INFORMATION_SCHEMA", "TABLES", "view"),
]


class TestWhatIsListed:

    def test_the_tables_and_views_of_public_are_listed(self):
        assert _browse(_Account(SALES), database="SALES_DB") == {"SALES_DB": {
            "PUBLIC": {"tables": ["CUSTOMERS", "ORDERS"], "views": ["V_MONTHLY"]},
            "FINANCE": {"tables": ["LEDGER"], "views": []},
        }}

    def test_listing_needs_no_warehouse(self):
        account = _Account(SALES)
        _browse(account, database="SALES_DB")
        assert all(sql.split()[0].upper() in ("SHOW", "SELECT") for sql in account.ran)
        assert not any("INFORMATION_SCHEMA" in sql.upper() or "USE " in sql.upper() for sql in account.ran)
        assert account.closed

    def test_the_named_database_is_found_whatever_its_case(self):
        tree = _browse(_Account(SALES + [("OTHER_DB", "PUBLIC", "T", "table")]), database="sales_db")
        assert list(tree) == ["SALES_DB"]

    def test_with_no_database_named_snowflakes_own_databases_are_left_out(self):
        account = _Account(SALES + [("SNOWFLAKE", "ACCOUNT_USAGE", "QUERY_HISTORY", "view"),
                                    ("SNOWFLAKE_SAMPLE_DATA", "TPCH_SF1", "ORDERS", "table"),
                                    ("MARKETING", "PUBLIC", "CAMPAIGNS", "table")])
        assert set(_browse(account)) == {"SALES_DB", "MARKETING"}
        assert not any("SNOWFLAKE" in sql.split(" IN DATABASE ")[-1] for sql in account.ran if " IN DATABASE " in sql)

    def test_a_database_that_cannot_be_read_does_not_hide_the_others(self):
        tree = _browse(_Account(SALES, unreadable={"ARCHIVE"}))
        assert list(tree) == ["SALES_DB"]

    def test_a_quote_in_a_database_name_is_kept_inside_its_quotes(self):
        account = _Account([('ODD"DB', "PUBLIC", "T", "table")])
        _browse(account)
        assert 'SHOW TABLES IN DATABASE "ODD""DB"' in account.ran


class TestWhenNothingIsListed:

    def test_a_named_database_the_role_cannot_see_is_said_with_the_grant_to_ask_for(self):
        with pytest.raises(NothingVisible) as said:
            _browse(_Account(SALES), database="DW")
        assert 'cannot see a database named "DW"' in str(said.value)
        assert "GRANT USAGE ON DATABASE DW TO ROLE QUERYBOT_READER" in str(said.value)

    def test_it_never_falls_back_to_another_database(self):
        # Before, a name not found listed the account's first database instead.
        with pytest.raises(NothingVisible):
            _browse(_Account(SALES), database="DW")

    def test_a_database_with_nothing_the_role_may_read_names_the_role_and_the_grants(self):
        with pytest.raises(NothingVisible) as said:
            _browse(_Account([], unreadable={"DW"}), database="DW")
        message = str(said.value)
        assert "the role QUERYBOT_READER sees no tables or views in the database \"DW\"" in message
        assert "Database 'DW' does not exist or not authorized." in message
        assert "GRANT SELECT ON ALL TABLES IN DATABASE DW TO ROLE QUERYBOT_READER" in message

    def test_an_account_with_only_snowflakes_own_databases_says_so(self):
        with pytest.raises(NothingVisible, match="sees no tables or views in any database it can see"):
            _browse(_Account([("SNOWFLAKE_SAMPLE_DATA", "TPCH_SF1", "ORDERS", "table")]))


# ── What the admin sees ─────────────────────────────────────────────────────────────────────────────────────────

def _request(path: str, fields: dict) -> Request:
    body = urlencode(fields).encode()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request({
        "type": "http", "method": "POST", "path": path, "root_path": "",
        "scheme": "http", "query_string": b"", "server": ("testserver", 80), "client": ("127.0.0.1", 1),
        "headers": [(b"content-type", b"application/x-www-form-urlencoded"),
                    (b"content-length", str(len(body)).encode())],
    }, receive)


@pytest.fixture
def admin(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    import store.crypto

    monkeypatch.setattr(store.crypto, "KEY_FILE", tmp_path / ".key")
    from admin import routes

    routes.store.init_db()
    return routes


def _key() -> str:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption()).decode()


FORM = {"db_type": "snowflake", "name": "Snowflake DW", "sf_account": "org-acct", "sf_user": "QUERYBOT_SVC",
        "sf_warehouse": "WH", "sf_database": "SALES_DB", "sf_role": "QUERYBOT_READER",
        "sf_auth_method": "keypair"}


def _browse_from_the_form(admin, account: _Account, **fields) -> dict:
    with patch.object(admin, "_is_auth", return_value=True), \
            patch("core.schema._sf_connect", return_value=account):
        response = asyncio.run(admin.database_discover_schema(
            _request("/admin/databases/discover-schema", {**FORM, "sf_private_key": _key(), **fields})))
    return json.loads(response.body)


class TestTheAdminConsole:

    def test_browse_schemas_shows_the_tables_of_public(self, admin):
        answer = _browse_from_the_form(admin, _Account(SALES))
        assert answer["status"] == "ok"
        assert answer["tree"]["SALES_DB"]["PUBLIC"]["tables"] == ["CUSTOMERS", "ORDERS"]

    def test_an_empty_listing_shows_why_as_it_is(self, admin):
        answer = _browse_from_the_form(admin, _Account(SALES), sf_database="DW")
        assert answer["status"] == "error"
        assert answer["message"].startswith('The role QUERYBOT_READER cannot see a database named "DW".')
        assert "Raw error" not in answer["message"]


# ── The tables picked are then read where they are ──────────────────────────────────────────────────────────────
#
# Browse schemas offers every database the role sees. Reading the picked tables (Save, then the knowledge base and
# Learn) looked only in the connection's own database, so a table picked in another one was silently left out.

import re


class _Warehouse:
    """A Snowflake account answering discovery's catalogue and sample queries for the session's database."""

    def __init__(self, tables: dict, current: str):
        self.tables = tables                  # {(database, schema, table): {column: [values]}}
        self.current = current
        self.ran: list[str] = []

    def cursor(self, kind=None):
        return _WarehouseCursor(self, as_dicts=kind is not None)

    def close(self):
        pass

    def _info(self, sql: str) -> str:
        """The database an INFORMATION_SCHEMA query reads: the one it names, else the session's."""
        named = re.search(r'"((?:[^"]|"")+)"\.INFORMATION_SCHEMA\.', sql, re.I)
        db = named.group(1).replace('""', '"') if named else self.current
        if db not in {d for d, _, _ in self.tables}:
            raise RuntimeError(f"002003 (02000): SQL compilation error: Database '{db}' does not exist or not "
                               "authorized.")
        return db

    def answer(self, sql: str) -> list[dict]:
        self.ran.append(sql)
        text = " ".join(sql.split())
        if text.startswith("SELECT CURRENT_DATABASE()"):
            return [{"DB": self.current}]
        if text == "SHOW DATABASES":
            return [{"name": d} for d in sorted({d for d, _, _ in self.tables})]
        if "INFORMATION_SCHEMA.TABLES" in text.upper():
            db = self._info(text)
            return [{"TABLE_CATALOG": d, "TABLE_SCHEMA": s, "TABLE_NAME": t, "TABLE_TYPE": "BASE TABLE",
                     "ROW_COUNT": 3, "COMMENT": None} for d, s, t in self.tables if d == db]
        if "INFORMATION_SCHEMA.COLUMNS" in text.upper():
            db = self._info(text)
            schema, table = re.search(r"TABLE_SCHEMA = '([^']*)' AND TABLE_NAME = '([^']*)'", text).groups()
            columns = self.tables.get((db, schema, table), {})
            return [{"COLUMN_NAME": c, "DATA_TYPE": "TEXT", "IS_NULLABLE": "YES", "CHARACTER_MAXIMUM_LENGTH": None,
                     "NUMERIC_PRECISION": None, "COMMENT": None} for c in columns]
        if "INFORMATION_SCHEMA." in text.upper():
            self._info(text)
            return []                          # no declared keys
        source = re.search(r'FROM ((?:"(?:[^"]|"")+"\.?)+)', text)
        parts = tuple(p.replace('""', '"') for p in re.findall(r'"((?:[^"]|"")+)"', source.group(1)))
        key = parts if len(parts) == 3 else (self.current, *parts)
        columns = self.tables[key]
        if text.startswith("SELECT DISTINCT"):
            column = re.search(r'SELECT DISTINCT "([^"]+)"', text).group(1)
            return [{column: v} for v in sorted(set(columns[column]))]
        return [dict(zip(columns, row)) for row in zip(*columns.values())]


class _WarehouseCursor:
    def __init__(self, warehouse: _Warehouse, as_dicts: bool):
        self.warehouse, self.as_dicts, self.rows = warehouse, as_dicts, []

    def execute(self, sql, *_):
        self.rows = self.warehouse.answer(sql)

    def fetchall(self):
        return self.rows if self.as_dicts else [tuple(r.values()) for r in self.rows]

    def fetchone(self):
        rows = self.fetchall()
        return rows[0] if rows else None


TWO_DATABASES = {
    ("SALES_DB", "PUBLIC", "ORDERS"): {"ORDER_ID": ["1", "2", "3"], "REGION": ["North", "South", "North"]},
    ("FINANCE_DB", "GL", "LEDGER"): {"ENTRY_ID": ["7", "8", "9"], "ACCOUNT": ["Cash", "Sales", "Cash"]},
}


def _read_picked(tmp_path, warehouse: _Warehouse, picked: set[str] | None, **cfg) -> dict:
    from core import schema

    with patch("core.schema._sf_connect", return_value=warehouse):
        count = schema.discover_and_write({"account": "org-acct", "user": "QUERYBOT_SVC", **cfg}, "snowflake",
                                          str(tmp_path), allowed_tables=picked, masking_config={})
    master = json.loads((tmp_path / "_schema.json").read_text())
    assert count == len([k for k in master if not k.startswith("__")])
    return master


class TestThePickedTablesAreRead:

    def test_a_table_picked_in_another_database_is_read_there(self, tmp_path):
        warehouse = _Warehouse(TWO_DATABASES, current="SALES_DB")
        master = _read_picked(tmp_path, warehouse, {"SALES_DB.PUBLIC.ORDERS", "FINANCE_DB.GL.LEDGER"},
                              database="SALES_DB")
        assert set(master) >= {"ORDERS", "LEDGER"}
        assert master["LEDGER"]["database"] == "FINANCE_DB" and master["LEDGER"]["schema"] == "GL"
        assert [c["name"] for c in master["LEDGER"]["columns"]] == ["ENTRY_ID", "ACCOUNT"]
        assert master["LEDGER"]["row_count_sent"] == 3
        assert any('FROM "FINANCE_DB"."GL"."LEDGER"' in sql for sql in warehouse.ran)

    def test_a_database_name_typed_in_lower_case_is_read_as_the_session_spells_it(self, tmp_path):
        warehouse = _Warehouse(TWO_DATABASES, current="SALES_DB")
        master = _read_picked(tmp_path, warehouse, {"PUBLIC.ORDERS"}, database="sales_db")
        assert "ORDERS" in master and master["ORDERS"]["database"] == "SALES_DB"

    def test_with_nothing_picked_the_connections_own_database_is_read(self, tmp_path):
        warehouse = _Warehouse(TWO_DATABASES, current="SALES_DB")
        master = _read_picked(tmp_path, warehouse, None, database="SALES_DB")
        assert "ORDERS" in master and "LEDGER" not in master

    def test_no_database_at_all_is_said(self, tmp_path):
        from core import schema

        warehouse = _Warehouse(TWO_DATABASES, current="")
        with patch("core.schema._sf_connect", return_value=warehouse), \
                pytest.raises(RuntimeError, match="Enter the Database on the connection"):
            schema.discover_and_write({"account": "org-acct"}, "snowflake", str(tmp_path), allowed_tables=None)
