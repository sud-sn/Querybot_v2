"""The Share dialog finds people and groups of the reader's own workspace by name, and nothing else.

A reader types two letters and sees up to eight people (with their group) and groups (with their size).
The search is the only reader-facing list of people, so what it may never return is the point:
another workspace's people, someone deactivated, a Teams or Slack account that cannot sign in, the
reader themself, an email, a password field. % and _ are letters, not wildcards, and the search is paced
so a script cannot walk the workspace's names.
"""

from __future__ import annotations

import uuid

import pytest

import store


@pytest.fixture()
def office():
    store.init_db()
    account, other = f"people-{uuid.uuid4().hex[:10]}", f"people-other-{uuid.uuid4().hex[:8]}"
    store.upsert_client(account, "portal")
    store.upsert_client(other, "portal")
    sales = store.create_group(account, "Sales team")
    riley_group = store.create_group(account, "Rivers office")

    def person(name, group=None, account_id=account, password="a-password-they-chose"):
        return store.create_user(account_id, name, f"{uuid.uuid4().hex[:8]}@example.com", group_id=group,
                                 password=password)[0]

    ids = {
        "me": person("Morgan Reader"),
        "riley": person("Riley Stone", sales),
        "rita": person("Rita Rivers", riley_group),
        "percent": person("Ana 100% Sure"),
        "under": person("Bo_Underscore"),
        "gone": person("Riley Gone"),
        "elsewhere": person("Riley Elsewhere", None, other),
        "teams": person("Riley Teams"),
    }
    store.update_user(ids["gone"], is_active=0)
    with store.get_db() as conn:                      # an account approved from Teams: it cannot sign in here
        conn.execute("UPDATE portal_user SET password_hash='!' WHERE id=?", (ids["teams"],))
    for n in range(12):
        person(f"Rae Number {n:02d}")
    store.create_group(account, "Raven office")      # with the people above, more than eight "ra" matches
    store.create_group(account, "Rapid response")
    try:
        yield {"account": account, "other": other, "sales": sales, "rivers": riley_group, **ids}
    finally:
        with store.get_db() as conn:
            for table in ("portal_user", "user_group", "client"):
                conn.execute(f"DELETE FROM {table} WHERE account_id IN (?, ?)", (account, other))


def _names(found):
    return [f["name"] for f in found]


def test_two_letters_find_people_with_their_group_and_groups_with_their_size(office):
    found = store.search_people_and_groups(office["account"], "ri", exclude_user_id=office["me"])
    assert {"type": "user", "id": office["riley"], "name": "Riley Stone", "group": "Sales team"} in found
    assert {"type": "group", "id": office["rivers"], "name": "Rivers office", "members": 1} in found
    for item in found:
        assert set(item) <= {"type", "id", "name", "group", "members"}, item


def test_one_letter_finds_nothing_and_no_more_than_eight_come_back(office):
    assert store.search_people_and_groups(office["account"], "r") == []
    assert store.search_people_and_groups(office["account"], "  ") == []
    assert len(store.search_people_and_groups(office["account"], "ra")) == 8           # of 14 that match


def test_names_that_start_with_the_letters_come_first(office):
    found = _names(store.search_people_and_groups(office["account"], "riv"))
    assert found[0] == "Rivers office" and "Rita Rivers" in found


def test_never_someone_who_cannot_open_a_dashboard_here_nor_the_reader(office):
    found = _names(store.search_people_and_groups(office["account"], "riley", exclude_user_id=office["me"]))
    assert found == ["Riley Stone"]                   # not the deactivated, the other workspace's, the Teams one
    assert "Morgan Reader" not in _names(store.search_people_and_groups(office["account"], "morgan",
                                                                        exclude_user_id=office["me"]))


def test_percent_and_underscore_are_letters_not_wildcards(office):
    assert _names(store.search_people_and_groups(office["account"], "0%")) == ["Ana 100% Sure"]
    assert _names(store.search_people_and_groups(office["account"], "o_u")) == ["Bo_Underscore"]
    assert store.search_people_and_groups(office["account"], "%%") == []
    assert store.search_people_and_groups(office["account"], "__") == []


# ── over HTTP ─────────────────────────────────────────────────────────────


@pytest.fixture()
def http():
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    import portal.routes as routes

    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app), routes


def test_the_search_answers_a_signed_in_reader_with_names_only(office, http):
    client, routes = http
    client.cookies.set(routes._COOKIE, routes._sign_session_value(office["me"]))
    reply = client.get("/portal/api/people", params={"q": "ri"})
    assert reply.status_code == 200 and reply.json()["ok"]
    assert "@" not in reply.text and "password" not in reply.text and "email" not in reply.text
    assert "Morgan Reader" not in reply.text and "Riley Elsewhere" not in reply.text
    client.cookies.clear()
    assert client.get("/portal/api/people", params={"q": "ri"}).status_code == 401


def test_a_script_walking_the_names_is_paced(office, http, monkeypatch):
    client, routes = http
    monkeypatch.setattr(routes, "_PEOPLE_SEARCH", routes._Paced(most=3, seconds=60))
    client.cookies.set(routes._COOKIE, routes._sign_session_value(office["me"]))
    codes = [client.get("/portal/api/people", params={"q": f"r{c}"}).status_code for c in "aeiou"]
    assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]
    slowed = client.get("/portal/api/people", params={"q": "ra"})
    assert int(slowed.headers["Retry-After"]) >= 1
    client.cookies.set(routes._COOKIE, routes._sign_session_value(office["riley"]))
    assert client.get("/portal/api/people", params={"q": "ra"}).status_code == 200   # each reader is paced alone
