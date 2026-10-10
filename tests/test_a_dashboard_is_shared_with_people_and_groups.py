"""A dashboard is shared with the workspace, with people by name, or with a group; and it is always live.

Until now a dashboard was its owner's, or the whole team's -- and the team saw it only while it was
"published": adding a chart or dragging a tile made it a draft, and it vanished from every teammate's
portal until someone published it again. Now:

- the owner shares with the workspace, with a person, or with a group (its members as they are when the
  dashboard is opened: someone who leaves the group stops seeing it);
- everyone it is shared with sees its current version (version history still lets the owner roll back);
- removing a share, or keeping a dashboard to oneself again, also ends the follow of anyone who lost
  access -- nothing is sent to someone who can no longer open it;
- rolling back changes tiles and layout, never who may open it;
- only a person or group of the same workspace can be added, never an email; an inactive person cannot.
"""

from __future__ import annotations

import uuid

import pytest

import store


@pytest.fixture()
def team():
    store.init_db()
    account = f"share-{uuid.uuid4().hex[:10]}"
    other = f"share-other-{uuid.uuid4().hex[:8]}"
    store.upsert_client(account, "portal")
    store.upsert_client(other, "portal")
    sales = store.create_group(account, "Sales team")
    finance = store.create_group(account, "Finance")
    stranger_group = store.create_group(other, "Sales team")

    def person(name, group=None, account_id=account):
        return store.create_user(account_id, name, f"{uuid.uuid4().hex[:8]}@test.com", group_id=group,
                                 password="a-password-they-chose")[0]

    people = {
        "owner": person("Ada Owner"),
        "riley": person("Riley Sales", sales),
        "sam": person("Sam Sales", sales),
        "fin": person("Fin Finance", finance),
        "solo": person("Solo Nobody"),
        "stranger": person("Stranger Elsewhere", stranger_group, other),
    }
    dashboard = store.create_dashboard(account, people["owner"], "thread", "Sales overview")
    try:
        yield {"account": account, "other": other, "sales": sales, "finance": finance,
               "stranger_group": stranger_group, "dashboard": dashboard["id"], **people}
    finally:
        with store.get_db() as conn:
            for table in ("dashboard_share", "dashboard_subscription", "pinned_chart", "dashboard_data_source",
                          "dashboard_artifact_version", "dashboard_artifact", "portal_user", "user_group", "client"):
                conn.execute(f"DELETE FROM {table} WHERE account_id IN (?, ?)", (account, other))


def _sees(t, who) -> bool:
    return store.get_dashboard_for_view(t["dashboard"], t[who], t["account"]) is not None


def _listed(t, who) -> bool:
    return any(d["id"] == t["dashboard"] for d in store.list_dashboards(t["account"], t[who]))


# ── who may open it ───────────────────────────────────────────────────────


def test_a_dashboard_is_its_owners_until_shared(team):
    assert _sees(team, "owner") and not any(_sees(team, w) for w in ("riley", "fin", "solo", "stranger"))


def test_shared_with_a_person_that_person_opens_it_read_only_and_no_one_else(team):
    assert store.add_dashboard_share(team["dashboard"], team["owner"], team["account"], "user", team["solo"])
    seen = store.get_dashboard_for_view(team["dashboard"], team["solo"], team["account"])
    assert seen and not seen["can_edit"] and seen["owner_name"] == "Ada Owner"
    assert _listed(team, "solo")
    assert not any(_sees(team, w) for w in ("riley", "fin"))


def test_shared_with_a_group_its_members_open_it_while_they_are_members(team):
    store.add_dashboard_share(team["dashboard"], team["owner"], team["account"], "group", team["sales"])
    assert _sees(team, "riley") and _sees(team, "sam") and _listed(team, "riley")
    assert not _sees(team, "fin") and not _sees(team, "solo")
    store.update_user(team["riley"], group_id=team["finance"])        # Riley moves to Finance
    assert not _sees(team, "riley") and not _listed(team, "riley")
    assert _sees(team, "sam")


def test_shared_with_the_workspace_everyone_in_it_opens_it_and_no_one_outside(team):
    store.share_dashboard(team["dashboard"], team["owner"], team["account"], "team")
    assert all(_sees(team, w) for w in ("riley", "fin", "solo"))
    assert store.get_dashboard_for_view(team["dashboard"], team["stranger"], team["account"]) is None
    assert store.get_dashboard_for_view(team["dashboard"], team["stranger"], team["other"]) is None


def test_a_shared_dashboard_stays_open_while_its_owner_edits_it(team):
    """Always live: an edit no longer makes it a draft hidden from everyone it is shared with."""
    store.share_dashboard(team["dashboard"], team["owner"], team["account"], "team")
    store.add_dashboard_share(team["dashboard"], team["owner"], team["account"], "user", team["solo"])
    store.update_dashboard_controls(team["dashboard"], team["owner"], team["account"], tabs=["Overview", "Detail"])
    assert store.get_dashboard(team["dashboard"], team["owner"], team["account"])["status"] == "draft"
    assert _sees(team, "fin") and _sees(team, "solo") and _listed(team, "fin")


# ── who it may be shared with ─────────────────────────────────────────────


def test_only_a_person_or_group_of_the_same_workspace_can_be_added(team):
    d, o, a = team["dashboard"], team["owner"], team["account"]
    assert store.add_dashboard_share(d, o, a, "user", team["stranger"]) is None
    assert store.add_dashboard_share(d, o, a, "group", team["stranger_group"]) is None
    store.update_user(team["solo"], is_active=0)
    assert store.add_dashboard_share(d, o, a, "user", team["solo"]) is None
    with pytest.raises(ValueError):
        store.add_dashboard_share(d, o, a, "email", 1)
    assert store.list_dashboard_shares(d, o, a) == []


def test_only_its_owner_shares_it(team):
    assert store.add_dashboard_share(team["dashboard"], team["riley"], team["account"], "user", team["solo"]) is None
    assert not store.remove_dashboard_share(team["dashboard"], team["riley"], team["account"], "user", team["solo"])
    assert store.list_dashboard_shares(team["dashboard"], team["riley"], team["account"]) == []


def test_sharing_twice_or_with_the_owner_changes_nothing(team):
    d, o, a = team["dashboard"], team["owner"], team["account"]
    store.add_dashboard_share(d, o, a, "user", team["solo"])
    store.add_dashboard_share(d, o, a, "user", team["solo"])
    store.add_dashboard_share(d, o, a, "user", o)
    assert [(s["subject_type"], s["subject_id"]) for s in store.list_dashboard_shares(d, o, a)] == [("user", team["solo"])]


def test_the_shares_name_people_with_their_group_and_groups_with_their_size_never_an_email(team):
    d, o, a = team["dashboard"], team["owner"], team["account"]
    store.add_dashboard_share(d, o, a, "user", team["fin"])
    store.add_dashboard_share(d, o, a, "group", team["sales"])
    shares = store.list_dashboard_shares(d, o, a)
    assert shares == [
        {"subject_type": "user", "subject_id": team["fin"], "name": "Fin Finance", "group_name": "Finance"},
        {"subject_type": "group", "subject_id": team["sales"], "name": "Sales team", "members": 2},
    ]
    assert not any("email" in s or "password_hash" in s for s in shares)


def test_the_audience_counts_each_person_once_and_never_the_owner(team):
    d, o, a = team["dashboard"], team["owner"], team["account"]
    assert store.dashboard_audience(d, a) == 0
    store.add_dashboard_share(d, o, a, "group", team["sales"])
    store.add_dashboard_share(d, o, a, "user", team["riley"])          # also in Sales: counted once
    assert store.dashboard_audience(d, a) == 2
    store.update_user(team["sam"], is_active=0)
    assert store.dashboard_audience(d, a) == 1


# ── follows end with access ───────────────────────────────────────────────


def test_removing_a_share_ends_the_follow_of_whoever_lost_access(team):
    d, o, a = team["dashboard"], team["owner"], team["account"]
    store.add_dashboard_share(d, o, a, "group", team["sales"])
    store.add_dashboard_share(d, o, a, "user", team["riley"])
    for who in ("riley", "sam"):
        store.subscribe_dashboard(d, team[who], a, cadence="weekly")
    store.subscribe_dashboard(d, o, a)
    assert store.remove_dashboard_share(d, o, a, "group", team["sales"])
    assert not _sees(team, "sam") and store.get_dashboard_subscription(d, team["sam"], a) is None
    # Riley still sees it, by name: the follow stays.
    assert _sees(team, "riley") and store.get_dashboard_subscription(d, team["riley"], a) is not None
    assert store.get_dashboard_subscription(d, o, a) is not None          # the owner's own follow stays


def test_keeping_it_to_oneself_again_ends_the_follows_of_the_workspace(team):
    d, o, a = team["dashboard"], team["owner"], team["account"]
    store.share_dashboard(d, o, a, "team")
    store.add_dashboard_share(d, o, a, "user", team["solo"])
    for who in ("fin", "solo"):
        store.subscribe_dashboard(d, team[who], a)
    store.share_dashboard(d, o, a, "personal")
    assert store.get_dashboard_subscription(d, team["fin"], a) is None
    assert store.get_dashboard_subscription(d, team["solo"], a) is not None   # still shared with Solo by name


def test_turning_the_workspace_off_through_the_chat_controls_ends_follows_too(team):
    d, o, a = team["dashboard"], team["owner"], team["account"]
    store.update_dashboard_controls(d, o, a, visibility="team")
    store.subscribe_dashboard(d, team["fin"], a)
    store.update_dashboard_controls(d, o, a, visibility="personal")
    assert store.get_dashboard_subscription(d, team["fin"], a) is None


# ── version history ───────────────────────────────────────────────────────


def test_rolling_back_never_changes_who_may_open_it(team):
    d, o, a = team["dashboard"], team["owner"], team["account"]
    store.share_dashboard(d, o, a, "team")
    team_version = store.get_dashboard(d, o, a)["version"]
    store.share_dashboard(d, o, a, "personal")
    store.add_dashboard_share(d, o, a, "user", team["solo"])
    store.rollback_dashboard(d, o, a, team_version)
    assert store.get_dashboard(d, o, a)["visibility"] == "personal"
    assert not _sees(team, "fin") and _sees(team, "solo")
    store.share_dashboard(d, o, a, "team")
    store.rollback_dashboard(d, o, a, 1)                                   # back before any sharing
    assert store.get_dashboard(d, o, a)["visibility"] == "team" and _sees(team, "fin")


# ── over HTTP ─────────────────────────────────────────────────────────────


@pytest.fixture()
def client():
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    import portal.routes as routes

    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app), routes


def _as(client, routes, user_id):
    client.cookies.set(routes._COOKIE, routes._sign_session_value(user_id))
    return client


def test_the_owner_shares_and_unshares_over_http_and_reads_names_only(team, client):
    http, routes = client
    _as(http, routes, team["owner"])
    base = f"/portal/api/dashboard/{team['dashboard']}/shares"
    added = http.post(base, json={"subject_type": "group", "subject_id": team["sales"]})
    assert added.status_code == 200, added.text
    assert added.json()["shares"] == [{"subject_type": "group", "subject_id": team["sales"], "name": "Sales team",
                                       "members": 2}]
    assert http.post(base, json={"workspace": True}).json()["workspace"] is True
    assert http.post(base, json={"subject_type": "user", "subject_id": team["stranger"]}).status_code == 404
    assert http.post(base, json={"subject_type": "email", "subject_id": 1}).status_code == 400
    removed = http.delete(f"{base}/group/{team['sales']}")
    assert removed.status_code == 200 and removed.json()["shares"] == []
    assert "@" not in http.get(base).text


def test_someone_else_cannot_read_or_change_the_shares(team, client):
    http, routes = client
    store.add_dashboard_share(team["dashboard"], team["owner"], team["account"], "group", team["sales"])
    _as(http, routes, team["riley"])                       # shared with Riley's group, but not Riley's dashboard
    base = f"/portal/api/dashboard/{team['dashboard']}/shares"
    assert http.get(base).status_code == 404
    assert http.post(base, json={"subject_type": "user", "subject_id": team["solo"]}).status_code == 404
    assert http.delete(f"{base}/group/{team['sales']}").status_code == 404
    assert _sees(team, "riley")
    http.cookies.clear()
    assert http.get(base).status_code == 401


def test_the_person_shared_with_opens_the_page_and_reads_who_shared_it(team, client):
    http, routes = client
    store.add_dashboard_share(team["dashboard"], team["owner"], team["account"], "user", team["solo"])
    page = _as(http, routes, team["solo"]).get(f"/portal/dashboard?dashboard_id={team['dashboard']}")
    assert page.status_code == 200 and "Sales overview" in page.text and "Shared by Ada Owner" in page.text
    other = _as(http, routes, team["fin"]).get(f"/portal/dashboard?dashboard_id={team['dashboard']}")
    assert "Shared by Ada Owner" not in other.text
