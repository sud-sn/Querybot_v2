"""
tests/test_thread_rebuild_language.py

The thread rebuild -- GET /portal/api/history/{thread_id} -- which is what
turns an already-delivered answer into French when the reader flips the
switcher.

The mechanism is not a translation pass. The endpoint rebuilds every turn from
the durable rows in answer_trace through build_assistant_response, so an answer
comes back in whatever language the reader is in NOW. The chat page calls it on
every load, and the switcher's post lands the reader back on that page, so the
flip needs no client code at all.

That makes this endpoint a deliberate, user-triggered re-read of governed rows,
which is why the grant check is tested here rather than assumed. Every test
drives the real route through a TestClient.
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

_tmp_db = os.path.join(tempfile.mkdtemp(), "test_thread_rebuild.db")
os.environ["QUERYBOT_DB_PATH"] = _tmp_db
for _mod in list(sys.modules):
    if _mod.startswith("store"):
        del sys.modules[_mod]

import store  # noqa: E402

store.init_db()

THREAD = "thread-abc"
SQL = "SELECT REGION, REVENUE FROM DW.SALES"
# A list, not a JSON string: update_answer_trace json-encodes result_rows
# itself, so handing it a string stores a double-encoded one and the rebuild
# reads back zero rows.
ROWS = [{"REGION": "North", "REVENUE": 900}, {"REGION": "South", "REVENUE": 400}]


def _client():
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    import portal.routes as pr

    app = FastAPI()
    app.include_router(pr.router)
    return TestClient(app), pr


def _signed_in(lang="en", group_tables=("DW.SALES",)):
    """A user, a group with a grant, and one successful trace in a thread."""
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    group_id = store.create_group(account_id, f"grp{os.urandom(3).hex()}")
    # set_group_tables is (group_id, account_id, tables) -- the account id
    # sits in the middle, which is easy to get wrong and fails silently.
    store.set_group_tables(group_id, account_id, list(group_tables))
    user_id, _ = store.create_user(
        account_id, "Ada", f"{os.urandom(4).hex()}@x.com", group_id=group_id)
    if lang != "en":
        store.set_user_language(user_id, lang)

    trace_id = store.create_answer_trace(
        account_id=account_id,
        question_id=f"q{os.urandom(4).hex()}",
        question_text="revenue by region",
        portal_user_id=user_id,
        session_id=f"s1:thread:{THREAD}",
    )
    store.update_answer_trace(
        trace_id, generated_sql=SQL, result_rows=ROWS,
        query_duration_ms=42, db_type="azure_sql",
    )
    store.finish_answer_trace(trace_id, status="success", row_count=2)

    client, pr = _client()
    client.cookies.set(pr._COOKIE, pr._sign_session_value(user_id))
    return client, pr, user_id, (group_id, account_id)


def _turns(client):
    response = client.get(f"/portal/api/history/{THREAD}")
    assert response.status_code == 200, response.text
    return response.json()["turns"]


# ══════════════════════════════════════════════════════════════════════════════
# The flip
# ══════════════════════════════════════════════════════════════════════════════

class TestTheAnswerComesBackInTheReadersLanguage:

    def test_an_english_reader_gets_the_english_card(self):
        client, _, _, _ = _signed_in(lang="en")
        assert _turns(client)[0]["payload"]["answer"]["headline"] == \
            "North leads at 900."

    def test_a_french_reader_gets_the_same_answer_in_french(self):
        """Same trace, same rows, same SQL -- only the reader changed."""
        client, _, _, _ = _signed_in(lang="fr")
        assert _turns(client)[0]["payload"]["answer"]["headline"] == \
            "North arrive en tête avec 900."

    def test_flipping_the_preference_flips_the_delivered_answer(self):
        """The whole point of the switcher: an answer given in English before
        the flip reads French after it, with nothing re-run against the
        database."""
        client, _, user_id, _ = _signed_in(lang="en")
        before = _turns(client)[0]["payload"]["answer"]["headline"]
        store.set_user_language(user_id, "fr")
        after = _turns(client)[0]["payload"]["answer"]["headline"]
        store.set_user_language(user_id, "en")
        back = _turns(client)[0]["payload"]["answer"]["headline"]
        assert before == "North leads at 900."
        assert after == "North arrive en tête avec 900."
        assert back == before

    def test_the_row_data_is_never_translated(self):
        client, _, _, _ = _signed_in(lang="fr")
        payload = _turns(client)[0]["payload"]
        assert payload["data"]["headers"] == ["REGION", "REVENUE"]
        assert payload["data"]["rows"][0]["REGION"] == "North"

    def test_the_activation_does_not_leak_to_the_next_request(self):
        """The token is released in a finally. Without it a French reader's
        request would leave the worker in French for whoever it serves next."""
        from core import i18n
        client, _, _, _ = _signed_in(lang="fr")
        _turns(client)
        assert i18n.get_active_language() == "en"


# ══════════════════════════════════════════════════════════════════════════════
# The grant check
# ══════════════════════════════════════════════════════════════════════════════

class TestARevokedGrantStopsTheReplay:

    def test_a_granted_table_replays(self):
        client, _, _, _ = _signed_in()
        assert len(_turns(client)) == 1

    def test_a_revoked_table_does_not(self):
        """The rows were authorised when the question was asked. Replaying
        them after the grant is gone -- and the switcher re-fetches this
        endpoint every time the reader changes language -- would make thread
        history a way to keep reading a table the workspace took away."""
        client, _, _, (group_id, account_id) = _signed_in()
        assert len(_turns(client)) == 1
        store.set_group_tables(group_id, account_id, ["DW.OTHER"])
        assert _turns(client) == []

    def test_a_user_with_no_grants_at_all_gets_nothing_back(self):
        client, _, _, (group_id, account_id) = _signed_in()
        store.set_group_tables(group_id, account_id, [])
        assert _turns(client) == []

    def test_an_admin_is_unrestricted(self):
        """get_allowed_tables returns None for an admin, which is what
        unrestricted means everywhere else in the product."""
        client, _, user_id, (group_id, account_id) = _signed_in()
        store.set_group_tables(group_id, account_id, [])
        store.update_user(user_id, role="admin")
        assert len(_turns(client)) == 1

    def test_the_grant_is_matched_case_insensitively(self):
        """Grants are stored uppercase; a lowercase one must not read as a
        revocation and silently empty someone's history."""
        client, _, _, (group_id, account_id) = _signed_in()
        store.set_group_tables(group_id, account_id, ["dw.sales"])
        assert len(_turns(client)) == 1

    def test_unparseable_sql_closes_for_a_regulated_tenant(self):
        """The ambiguous case: we cannot tell which tables it reads. Resolved
        the same way portal_export_csv in the same file resolves its own."""
        from unittest.mock import patch
        import portal.routes as pr
        import core.compliance.policy_engine as engine
        with patch.object(engine, "is_regulated", return_value=True):
            assert pr._trace_tables_still_granted(
                "this is not sql", "azure_sql", {"DW.SALES"}, "acct") is False

    def test_unparseable_sql_stays_open_for_an_unregulated_one(self):
        """Matching the export precedent exactly rather than being stricter in
        one place and not the other -- an inconsistency is how the next reader
        concludes one of them is wrong."""
        from unittest.mock import patch
        import portal.routes as pr
        import core.compliance.policy_engine as engine
        with patch.object(engine, "is_regulated", return_value=False):
            assert pr._trace_tables_still_granted(
                "this is not sql", "azure_sql", {"DW.SALES"}, "acct") is True

    def test_an_empty_query_is_never_granted(self):
        import portal.routes as pr
        assert pr._trace_tables_still_granted("", "azure_sql", {"DW.SALES"}, "a") is False

    def test_a_query_that_reads_no_table_has_nothing_to_authorise(self):
        """"SELECT 2 AS total" is not the ambiguous case -- it parses, and it
        reads nothing. Refusing it would drop turns for no gain."""
        import portal.routes as pr
        assert pr._trace_tables_still_granted(
            "SELECT 2 AS total", "azure_sql", set(), "acct") is True

    def test_a_second_table_the_user_lacks_blocks_the_whole_turn(self):
        """A join reads every table in it. Partial access is no access."""
        import portal.routes as pr
        assert pr._trace_tables_still_granted(
            "SELECT a FROM DW.SALES JOIN DW.SECRET ON 1=1",
            "azure_sql", {"DW.SALES"}, "acct") is False
