"""
A warehouse login that can write is named when the connection is tested.

QueryBot only reads the warehouse: its validator refuses anything else. The
login it connects with was never looked at, so a login in db_owner or
db_datawriter tested "Connection successful." like a read-only one, and the
setup checklist's "use a read-only login" rested on the admin remembering.

Test connection now reads the rights the login holds beyond reading -- on the
database and on each of its schemas, a role's included -- and says what they
are, in amber. The connection still works: a tenant may run such a login
behind controls of its own. Rights that cannot be read are not guessed at.

The real route and connection test; the database connection is a stand-in.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch
from urllib.parse import urlencode

import pytest
from starlette.requests import Request


class _Cursor:
    def __init__(self, granted: list[str], *, unreadable: bool = False):
        self.granted, self.unreadable = granted, unreadable

    def execute(self, sql, params=()):
        if self.unreadable and "fn_my_permissions" in sql:
            raise RuntimeError("The server principal is not able to access the database under this context.")

    def fetchone(self):
        return ("SALES_DW", "dbo", "reader_login")

    def fetchall(self):
        return [(name,) for name in self.granted]


class _Connection:
    def __init__(self, cursor: _Cursor):
        self._cursor, self.closed = cursor, False

    def cursor(self):
        return self._cursor

    def close(self):
        self.closed = True


def _tested(granted: list[str], *, unreadable: bool = False) -> dict:
    from core import schema

    connection = _Connection(_Cursor(granted, unreadable=unreadable))
    with patch.object(schema, "_az_connect", return_value=connection):
        details = schema.test_connection(
            {"server": "s", "database": "SALES_DW", "user": "reader_login", "password": "p"}, "azure_sql")
    assert connection.closed
    return details


class TestTheConnectionTest:

    def test_a_login_that_can_write_is_named(self):
        details = _tested(["INSERT", "UPDATE", "DELETE", "INSERT"])
        assert details["write_permissions"] == ["DELETE", "INSERT", "UPDATE"]
        assert details["user"] == "reader_login"

    def test_a_read_only_login_is_not(self):
        assert "write_permissions" not in _tested([])

    def test_rights_it_cannot_read_are_not_guessed(self, caplog):
        details = _tested(["INSERT"], unreadable=True)
        assert "write_permissions" not in details and details["database"] == "SALES_DW"
        assert "Could not read the login's permissions" in caplog.text


@pytest.fixture
def admin(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    monkeypatch.setenv("QUERYBOT_KEY_FILE", str(tmp_path / ".key"))
    from admin import routes

    routes.store.init_db()
    return routes


def _test_button(routes, details: dict) -> dict:
    body = urlencode({"db_type": "azure_sql", "name": "Warehouse", "az_server": "dw.example.net",
                      "az_database": "SALES_DW", "az_user": "reader_login", "az_password": "a-password"}).encode()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request({
        "type": "http", "method": "POST", "path": "/admin/databases/test", "root_path": "",
        "scheme": "http", "query_string": b"", "server": ("testserver", 80), "client": ("127.0.0.1", 1),
        "headers": [(b"content-type", b"application/x-www-form-urlencoded"),
                    (b"content-length", str(len(body)).encode())],
    }, receive)
    with patch.object(routes, "_is_auth", return_value=True), \
            patch("core.schema.test_connection", return_value=details):
        return json.loads(asyncio.run(routes.database_test(request)).body)


class TestTheAdminIsTold:

    def test_what_the_login_can_do(self, admin):
        answer = _test_button(admin, {"database": "SALES_DW", "user": "reader_login",
                                      "write_permissions": ["DELETE", "INSERT"]})
        assert answer["status"] == "ok"
        assert "(DELETE, INSERT)" in answer["warning"] and "read-only login" in answer["warning"]
        assert answer["details"] == {"database": "SALES_DW", "user": "reader_login"}

    def test_nothing_of_a_read_only_login(self, admin):
        answer = _test_button(admin, {"database": "SALES_DW", "user": "reader_login"})
        assert answer["status"] == "ok" and "warning" not in answer
