"""
A database that cannot answer says so, and one waking up is waited for.

* Azure SQL pauses an idle serverless database. The first connection after
  that fails with error 40613, "is not currently available. Please retry the
  connection later." -- reported by the ODBC driver under the generic SQLSTATE
  HY000, with the number at the end of the message. The connection retried
  only on SQLSTATEs, so it never waited for the database to wake.
* "list the available items" then failed at the database, and the listing
  fell through to the ordinary path, whose reply for a question with no
  measure was that the semantic layer could not resolve a dataset or a
  measure: a misdiagnosis that sends the reader rephrasing a question that
  was fine.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

import pytest

from tests import answer_harness as harness

_ASLEEP = ("[HY000] [Microsoft][ODBC Driver 18 for SQL Server][SQL Server]Database 'db' on server "
           "'srv.example.net' is not currently available.  Please retry the connection later.  If the "
           "problem persists, contact customer support, and provide them the session tracing ID of "
           "'{00000000-0000-0000-0000-000000000000}'. (40613) (SQLDriverConnect)")
_LOGIN_REFUSED = ("[28000] [Microsoft][ODBC Driver 18 for SQL Server][SQL Server]Login failed for user "
                  "'reader'. (18456) (SQLDriverConnect)")


class _DriverError(Exception):
    """What pyodbc raises: a SQLSTATE and the driver's message."""


def _driver(*outcomes):
    """A stand-in pyodbc whose connect() meets each outcome in turn."""
    calls = []

    def connect(conn_str, timeout=0):
        outcome = outcomes[len(calls)]
        calls.append(conn_str)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return types.SimpleNamespace(connect=connect, Error=_DriverError), calls


_CFG = {"server": "srv.example.net", "database": "db", "user": "reader", "password": "pw"}


class TestADatabaseWakingUpIsWaitedFor:

    def test_a_paused_database_is_retried(self):
        from core.schema import _az_connect

        connection = object()
        pyodbc, calls = _driver(_DriverError("HY000", _ASLEEP), connection)
        with patch.dict(sys.modules, {"pyodbc": pyodbc}), patch("time.sleep") as slept:
            assert _az_connect(_CFG) is connection
        assert len(calls) == 2 and slept.call_count == 1

    def test_a_refused_login_is_not(self):
        from core.schema import _az_connect

        pyodbc, calls = _driver(_DriverError("28000", _LOGIN_REFUSED), object())
        with patch.dict(sys.modules, {"pyodbc": pyodbc}), patch("time.sleep") as slept, \
                pytest.raises(_DriverError):
            _az_connect(_CFG)
        assert len(calls) == 1 and not slept.called

    def test_another_error_under_hy000_is_not(self):
        from core.schema import _az_connect

        # A number in the slot, but not one of Azure's transient ones.
        pyodbc, calls = _driver(_DriverError("HY000", "[HY000] Something else went wrong. (18456) (SQLDriverConnect)"),
                                object())
        with patch.dict(sys.modules, {"pyodbc": pyodbc}), patch("time.sleep"), pytest.raises(_DriverError):
            _az_connect(_CFG)
        assert len(calls) == 1


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("database-cannot-answer")) as built:
        yield built


def _listing_fails(warehouse, error):
    """The warehouse, failing the listing's own query with ``error``."""
    real = warehouse.query

    def query(sql, max_rows=200):
        if "SELECT DISTINCT TOP 201" in sql:
            raise error
        return real(sql, max_rows)

    return patch.object(warehouse, "query", side_effect=query)


class TestAListTheDatabaseCannotAnswer:

    def test_says_the_database_is_unavailable(self, warehouse):
        with _listing_fails(warehouse, _DriverError("HY000", _ASLEEP)):
            answer = harness.ask(warehouse, "list the available items")

        (reply,) = [str(payload) for _kind, payload in answer["replies"]]
        assert "Most likely reason:" in reply and "paused" in reply
        assert "SELECT DISTINCT TOP 201" in reply
        assert "semantic layer" not in reply.lower()
        assert not answer["model_wrote_sql"]

    def test_a_statement_timeout_is_told_as_one(self, warehouse):
        # Not "check that the database is running and reachable": a query
        # that ran to its time limit reached it.
        error = _DriverError("HYT00", "[HYT00] [Microsoft][ODBC Driver 18 for SQL Server]Query timeout expired "
                                      "(0) (SQLExecDirectW)")
        with _listing_fails(warehouse, error):
            answer = harness.ask(warehouse, "list the available items")

        (reply,) = [str(payload) for _kind, payload in answer["replies"]]
        assert "The database stopped the query after" in reply
        assert "did not respond in time" not in reply

    def test_a_list_query_the_database_rejects_is_still_asked_another_way(self, warehouse):
        # Not the database's fault but the query's: the ordinary path may
        # read it differently, so it runs -- and its reply is no failure card.
        error = _DriverError("42S22", "[42S22] [Microsoft][ODBC Driver 18 for SQL Server][SQL Server]"
                                      "Invalid column name 'ITM_NM'. (207) (SQLExecDirectW)")
        with _listing_fails(warehouse, error):
            answer = harness.ask(warehouse, "list the available items")

        assert answer["replies"]
        assert not any("Most likely reason:" in str(payload) for _kind, payload in answer["replies"])


class TestTheDatabaseUnavailable:

    @pytest.mark.parametrize("raw, unavailable", [
        (_ASLEEP, True), (_LOGIN_REFUSED, True), ("Login timeout expired (HYT00)", True),
        ("[08S01] Communication link failure", True), ("Query timeout expired", True),
        ("[42000] Cannot open server 'srv' requested by the login. Client with IP address '10.0.0.1' is not "
         "allowed to access the server. (40615) (SQLDriverConnect)", True),
        ("250001 (08001): Failed to connect to DB: x.snowflakecomputing.com:443. Incorrect username or password "
         "was specified.", True),
        ("ORA-12541: TNS:no listener", True), ("ORA-12170: TNS:Connect timeout occurred", True),
        ("ORA-01017: invalid username/password; logon denied", True), ("DPY-6005: cannot connect to database", True),
        ("ORA-01034: ORACLE not available", True), ("DPY-4024: call timeout of 300000 ms exceeded", True),
        ("000630 (57014): Statement reached its statement or warehouse timeout of 300 second(s) and was canceled.",
         True),
        ("000606 (57P03): No active warehouse selected in the current session.", True),
        ("Invalid column name 'X'. (207)", False), ("Incorrect syntax near 'FROM'.", False),
        ("ORA-00942: table or view does not exist", False), ("Something nobody has a pattern for.", False),
        # A login's SQLSTATE as a value from the data is no refused login.
        ("[22018] Conversion failed when converting the nvarchar value '28000-1234' to data type int. (245)", False),
    ])
    def test_is_told_from_a_query_at_fault(self, raw, unavailable):
        from core.failure_messages import is_database_unavailable

        assert is_database_unavailable(raw) is unavailable
