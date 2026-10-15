"""Standing conditions in the portal: the chip row above the chat box, and the reader's own on the Profile page.

A reader sets one by saying it in the chat (tests/test_standing_conditions_hold_until_removed.py). Here: the chips a
conversation shows (the reader's own from their profile, then the conversation's), the chip's x taking one off that
conversation only, and the Profile page listing the reader's own and removing one for good. Another reader's, or
another workspace's, are never listed. Invented data only.
"""

from __future__ import annotations

import os
from contextlib import ExitStack
from unittest.mock import patch

import pytest

from core2.plan.ir import Filter, Window
from core2.standing import Standing

LAST_2 = Standing("p2m", "the last 2 months", "always", window=Window(kind="last", unit="month", n=2))
DOWNTOWN = Standing("dtn", "Store name is Downtown Store", "conversation",
                    filters=[Filter(field="store.name", op="eq", values=["Downtown Store"])])


@pytest.fixture
def site(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    import store
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from portal import routes as portal_routes

    store.init_db()
    account = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account, "portal")
    reader = {"id": 7, "account_id": account, "name": "Riley Reader", "email": "riley@example.test",
              "role": "analyst"}
    app = FastAPI()
    app.include_router(portal_routes.router)
    with ExitStack() as stack:
        stack.enter_context(patch.object(portal_routes, "_get_portal_user", lambda *a, **k: reader))
        yield TestClient(app), account, reader


def test_the_chips_show_the_readers_own_then_the_conversations(site):
    client, account, reader = site
    from core2 import service
    from store import profile_store as P

    P.keep_standing(7, account, LAST_2.data())
    P.keep_standing(8, account, Standing("other", "Segment is not Online", "always").data())   # another reader
    P.keep_standing(7, "elsewhere", Standing("away", "the last 6 months", "always").data())     # another workspace
    service._session(f"{account}:7:thread:t1").standing.append(DOWNTOWN)

    got = client.get("/portal/api/chat/standing", params={"thread_id": "t1"}).json()
    assert got["standing"] == [
        {"id": "p2m", "words": "the last 2 months", "words_fr": "the last 2 months", "scope": "always"},
        {"id": "dtn", "words": "Store name is Downtown Store", "words_fr": "Store name is Downtown Store",
         "scope": "conversation"}]
    # Another conversation has only the reader's own.
    assert [s["id"] for s in client.get("/portal/api/chat/standing", params={"thread_id": "t2"}).json()["standing"]] \
        == ["p2m"]


def test_the_chips_x_takes_one_off_this_conversation_only(site):
    client, account, reader = site
    from core2 import service
    from store import profile_store as P

    P.keep_standing(7, account, LAST_2.data())
    service._session(f"{account}:7:thread:t3").standing.append(DOWNTOWN)
    left = client.post("/portal/api/chat/standing/off", json={"thread_id": "t3", "id": "p2m"}).json()
    assert [s["id"] for s in left["standing"]] == ["dtn"]
    left = client.post("/portal/api/chat/standing/off", json={"thread_id": "t3", "id": "dtn"}).json()
    assert left["standing"] == []
    # The profile keeps it: other conversations still start with it.
    assert [s["id"] for s in P.standing_conditions(7, account)] == ["p2m"]
    assert [s["id"] for s in client.get("/portal/api/chat/standing", params={"thread_id": "t4"}).json()["standing"]] \
        == ["p2m"]


def test_the_profile_page_lists_them_and_removes_one_for_good(site):
    client, account, reader = site
    from store import profile_store as P

    P.keep_standing(7, account, LAST_2.data())
    page = client.get("/portal/settings", params={"tab": "profile"}).text
    assert "Standing conditions" in page and "the last 2 months" in page
    assert client.get("/portal/api/profile/standing").json()["standing"] == [
        {"id": "p2m", "words": "the last 2 months", "words_fr": "the last 2 months"}]
    assert client.post("/portal/api/profile/standing/remove", json={"id": "p2m"}).json() == {"ok": True}
    assert P.standing_conditions(7, account) == []
    assert client.post("/portal/api/profile/standing/remove", json={"id": "p2m"}).status_code == 404
    assert "None yet" in client.get("/portal/settings", params={"tab": "profile"}).text


def test_a_new_period_replaces_the_one_kept(site):
    _, account, _ = site
    from store import profile_store as P

    P.keep_standing(7, account, LAST_2.data())
    P.keep_standing(7, account, Standing("p6m", "the last 6 months", "always",
                                         window=Window(kind="last", unit="month", n=6)).data())
    P.keep_standing(7, account, DOWNTOWN.data())
    assert [s["words"] for s in P.standing_conditions(7, account)] == ["the last 6 months",
                                                                       "Store name is Downtown Store"]
