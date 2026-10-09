"""A question asked while an Azure SQL database is paused waits for it, and says so plainly when it cannot.

On the first server test of the new core, four questions in a day "stopped with an error" and went
to today's pipeline, whose fallback answer was unrelated to the question. Each was Azure SQL's
40613, "Database ... is not currently available. Please retry the connection later": a serverless
database resuming from a pause, or a failover. The new core tried its connection once, while
reading member names before planning, and the error went up as a crash. Today's pipeline tries
four times over about a minute.

The new core now tries as today's pipeline does. A database still unavailable after that is told
to the reader in a sentence; a member-name read that fails any other way costs only the names.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import types
from unittest.mock import patch

import pytest

from core2.plan.values import MemberIndex
from core2.service import UNAVAILABLE, Services, Session, answer_question

PAUSED = ("HY000", "[HY000] [Microsoft][ODBC Driver 18 for SQL Server][SQL Server]Database 'db' on server "
                   "'server.example.test' is not currently available.  Please retry the connection later. (40613) "
                   "(SQLDriverConnect)")


class _DriverError(Exception):
    pass


def _driver(failures: int) -> tuple[types.ModuleType, list[int]]:
    """A stand-in ODBC driver whose first ``failures`` connections meet a paused database."""
    tries: list[int] = []

    class Connection:
        timeout = 0

        def close(self):
            pass

    def connect(conn_str, timeout=0):
        tries.append(1)
        if len(tries) <= failures:
            raise _DriverError(*PAUSED)
        return Connection()

    module = types.ModuleType("pyodbc")
    module.Error = _DriverError
    module.connect = connect
    return module, tries


CREDENTIALS = {"server": "server.example.test", "database": "db", "user": "u", "password": "p"}


def test_the_new_core_waits_for_a_paused_database_as_todays_pipeline_does():
    from core2.warehouse.querybot import QueryBotWarehouse

    driver, tries = _driver(failures=2)
    with patch.dict(sys.modules, {"pyodbc": driver}), patch("time.sleep") as slept:
        conn = QueryBotWarehouse("azure_sql", CREDENTIALS)._connection()
    assert conn is not None and len(tries) == 3
    assert [c.args[0] for c in slept.call_args_list] == [5, 20]       # the waits today's pipeline makes


def test_a_database_still_paused_after_a_minute_is_an_error_not_a_hang():
    from core2.warehouse.querybot import QueryBotWarehouse

    driver, tries = _driver(failures=99)
    with patch.dict(sys.modules, {"pyodbc": driver}), patch("time.sleep"):
        with pytest.raises(_DriverError):
            QueryBotWarehouse("azure_sql", CREDENTIALS)._connection()
    assert len(tries) == 4


# ── what the reader is told ─────────────────────────────────────────────────


@pytest.fixture(scope="module")
def retail():
    from evals.core2 import domains
    from evals.core2.compile_eval import learn

    return learn(domains.build("retail"), "descriptive")


class _Paused:
    dialect = "duckdb"

    def query(self, sql, *, max_rows=None):
        raise _DriverError(*PAUSED)


def test_a_query_that_meets_a_paused_database_says_so(retail):
    _, model = retail
    plan = {"kind": "query", "intent": "value", "measures": ["net_amount"]}
    services = Services(model=model, warehouse=_Paused(), complete=lambda s, t: json.dumps(plan),
                        index=MemberIndex(), today=dt.date(2026, 6, 15))
    payload = answer_question("net sales", services, Session())
    assert payload["answer"]["headline"] == UNAVAILABLE and not payload.get("unsupported")
    assert "refused" not in payload["answer"]["headline"]


def _portal(retail, names_fail_with: Exception):
    """The portal's answer with the workspace's setup stood in for, and member names failing as given."""
    from core2 import service

    _, model = retail
    plan = {"kind": "query", "intent": "value", "measures": ["net_amount"]}
    built, _ = retail

    def planner(*args, **kwargs):
        return lambda stable, tail: json.dumps(plan)

    def names(*args, **kwargs):
        raise names_fail_with

    from core2.warehouse.runner import DuckDBWarehouse

    with patch("store.get_client", return_value={"db_config_id": 7}), \
            patch("store.get_db_config", return_value={"db_type": "duckdb", "credentials": {}}), \
            patch("store.get_client_state", return_value={}), \
            patch("core.schema.load_known_tables", return_value=[]), \
            patch("core.value_index.value_index_enabled", return_value=True), \
            patch("core2.bootstrap.service.keep_decisions_current"), \
            patch("core2.bootstrap.service.answering_model", return_value=model), \
            patch("core2.bootstrap.ai.workspace_planner", planner), \
            patch("core2.warehouse.governed.GovernedWarehouse", lambda *a, **k: DuckDBWarehouse(built.con)), \
            patch.object(service, "question_scrubber", return_value=None), \
            patch.object(service, "_member_index", names):
        return service.portal_answer("acct-paused", "net sales", None, session_key="s-paused")


def test_member_names_unread_because_the_database_is_paused_are_told_not_crashed(retail):
    payload = _portal(retail, _DriverError(*PAUSED))
    assert payload["answer"]["headline"] == UNAVAILABLE and payload.get("data") is None


def test_member_names_unread_for_any_other_reason_cost_only_the_names(retail):
    payload = _portal(retail, RuntimeError("the names table was renamed"))
    assert payload.get("data", {}).get("rows"), payload["answer"]["headline"]
