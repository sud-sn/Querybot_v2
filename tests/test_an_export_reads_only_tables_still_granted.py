"""The CSV export of a kept answer reads only tables the reader still has.

An answer's rows are kept with its trace, released under the reader's access when the question was asked. The
thread's history checks again, each time it is opened, that the reader still has every table the answer's query
read, and leaves out an answer that read one taken from them since. The CSV export of the same answer did not:
a table removed from the reader's group stayed readable through the export of an answer kept from before. The
export now checks as the history does -- the same rule, the same answers left out -- and an answer whose rows no
query it recorded can account for is not exported to anyone whose tables are limited.

Every test drives the real routes; invented data only.
"""

from __future__ import annotations

import csv
import io
import os

import pytest

import store

SQL = "SELECT REGION, SUM(REVENUE) AS REVENUE FROM DW.SALES GROUP BY REGION"
JOINED = "SELECT s.REGION, SUM(s.REVENUE) AS REVENUE FROM DW.SALES s JOIN DW.STORES t ON s.STORE_ID=t.ID GROUP BY s.REGION"
ROWS = [{"REGION": "North", "REVENUE": 900}, {"REGION": "South", "REVENUE": 400}]
THREAD = "thread-export"
NO_LONGER = "This answer read a table you no longer have access to."
UNCHECKED = "These rows could not be checked against your access today."


def _client(user_id: int):
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    import portal.routes as pr

    app = FastAPI()
    app.include_router(pr.router)
    client = TestClient(app)
    client.cookies.set(pr._COOKIE, pr._sign_session_value(user_id))
    return client


class Reader:
    """A reader in a group granted ``tables``, and the answers they were given."""

    def __init__(self, tables=("DW.SALES",), role="analyst"):
        self.account_id = f"acct{os.urandom(4).hex()}"
        store.upsert_client(self.account_id, "Test Ltd")
        self.group_id = store.create_group(self.account_id, f"grp{os.urandom(3).hex()}")
        store.set_group_tables(self.group_id, self.account_id, list(tables))
        self.id, _ = store.create_user(self.account_id, "Ada", f"{os.urandom(4).hex()}@x.com", group_id=self.group_id,
                                       role=role, password="a-password-they-chose")

    def answer(self, sql: str | None = SQL, *, route: str = "", logged: str = "") -> int:
        """An answer kept for this reader: its SQL on the trace (``sql``), or only in the query log (``logged``),
        as a follow-up answered from the result on screen keeps it."""
        question_id = f"q{os.urandom(4).hex()}"
        trace_id = store.create_answer_trace(account_id=self.account_id, question_id=question_id,
                                             question_text="revenue by region", portal_user_id=self.id,
                                             session_id=f"s1:thread:{THREAD}", **({"route": route} if route else {}))
        store.update_answer_trace(trace_id, generated_sql=sql or "", result_rows=ROWS, db_type="azure_sql")
        store.finish_answer_trace(trace_id, status="success", row_count=len(ROWS))
        if logged:
            store.log_query(self.account_id, "top two", logged, row_count=2, portal_user_id=self.id,
                            question_id=question_id)
        return trace_id

    def export(self, trace_id: int):
        return _client(self.id).get(f"/portal/api/export-csv?trace_id={trace_id}")

    def history(self) -> list[dict]:
        response = _client(self.id).get(f"/portal/api/history/{THREAD}")
        assert response.status_code == 200, response.text
        return response.json()["turns"]


def _csv(response) -> list[dict]:
    return list(csv.DictReader(io.StringIO(response.text)))


def _exports(account_id: str) -> int:
    with store.get_db() as conn:
        return conn.execute("SELECT COUNT(*) AS n FROM export_event WHERE account_id=?", (account_id,)).fetchone()["n"]


def test_a_reader_who_still_has_the_table_exports_the_answer():
    reader = Reader()
    response = reader.export(reader.answer())
    assert response.status_code == 200, response.text
    assert _csv(response) == [{"REGION": "North", "REVENUE": "900"}, {"REGION": "South", "REVENUE": "400"}]
    assert _exports(reader.account_id) == 1


@pytest.mark.parametrize("change", ["the table taken from their group", "moved to a group without it",
                                    "their group deleted"])
def test_a_table_taken_from_the_reader_is_not_read_again_through_an_export(change):
    reader = Reader()
    trace_id = reader.answer()
    if change == "the table taken from their group":
        store.set_group_tables(reader.group_id, reader.account_id, ["DW.RETURNS"])
    elif change == "moved to a group without it":
        other = store.create_group(reader.account_id, "Returns desk")
        store.set_group_tables(other, reader.account_id, ["DW.RETURNS"])
        store.update_user(reader.id, group_id=other)
    else:
        store.delete_group(reader.group_id)
    response = reader.export(trace_id)
    assert response.status_code == 403 and NO_LONGER in response.json()["error"]
    assert reader.history() == [], "the history and the export disagree"
    assert _exports(reader.account_id) == 0, "a refused export was recorded as made"


@pytest.mark.parametrize("change", ["granted to them by name", "made an admin"])
def test_access_given_back_exports_again(change):
    reader = Reader()
    trace_id = reader.answer()
    store.set_group_tables(reader.group_id, reader.account_id, [])
    assert reader.export(trace_id).status_code == 403
    if change == "granted to them by name":
        store.set_user_extra_tables(reader.id, reader.account_id, ["DW.SALES"])
    else:
        store.update_user(reader.id, role="admin")
    assert reader.export(trace_id).status_code == 200
    assert len(reader.history()) == 1


def test_an_answer_that_joined_two_tables_needs_both():
    reader = Reader(tables=("DW.SALES", "DW.STORES"))
    trace_id = reader.answer(JOINED)
    assert reader.export(trace_id).status_code == 200
    store.set_group_tables(reader.group_id, reader.account_id, ["DW.SALES"])
    response = reader.export(trace_id)
    assert response.status_code == 403 and NO_LONGER in response.json()["error"]


def test_an_answer_that_read_no_table_exports():
    reader = Reader(tables=())
    assert reader.export(reader.answer("SELECT 2 AS total")).status_code == 200


def test_a_follow_up_kept_with_its_sql_only_in_the_query_log_is_checked_by_it():
    reader = Reader()
    trace_id = reader.answer(None, logged=SQL)
    assert reader.export(trace_id).status_code == 200
    store.set_group_tables(reader.group_id, reader.account_id, ["DW.RETURNS"])
    response = reader.export(trace_id)
    assert response.status_code == 403 and NO_LONGER in response.json()["error"]


@pytest.mark.parametrize("role,status", [("analyst", 403), ("admin", 200)])
def test_rows_no_recorded_query_accounts_for(role, status):
    reader = Reader(role=role)
    response = reader.export(reader.answer(None))
    assert response.status_code == status, response.text
    if status == 403:
        assert UNCHECKED in response.json()["error"]


def test_a_new_core_answer_follows_the_same_rule():
    reader = Reader()
    trace_id = reader.answer(route="core2")
    assert reader.export(trace_id).status_code == 200
    store.set_group_tables(reader.group_id, reader.account_id, [])
    assert reader.export(trace_id).status_code == 403


@pytest.mark.parametrize("regulated,status", [(True, 403), (False, 200)])
def test_a_query_that_cannot_be_read_is_decided_as_the_history_decides_it(regulated, status):
    from unittest.mock import patch

    import core.compliance.policy_engine as policy_engine

    reader = Reader()
    trace_id = reader.answer("SELEC REGION FRM DW.SALES ((")
    with patch.object(policy_engine, "is_regulated", return_value=regulated):
        assert reader.export(trace_id).status_code == status
        assert len(reader.history()) == (0 if regulated else 1)


def test_another_readers_answer_is_still_not_theirs_to_export():
    owner, other = Reader(), Reader()
    response = other.export(owner.answer())
    assert response.status_code == 404
