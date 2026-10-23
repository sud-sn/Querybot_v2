"""A group gathers people, and what a group is given reaches each of its members.

An admin makes a group (a region's sales team, say), puts people in it, and writes row rules for it on Row access:
each member sees the rows the group's rules allow. A person was in one group at most, and a group was a list of
tables with no members of its own to add: putting someone in their region's team took them out of their first
group and its tables. Now a person belongs to every group they are put in, and each group's row rules, tables,
shared dashboards, scheduled emails and alerts reach each of its members. A group has its own page: its members,
people added and taken out, and its row rules.

A rule for a group names the group by its number, as Row access writes it: a group whose name is that number is
another group. A rule that reads "their groups" reads the names of every group the person is in.

Invented data only.
"""

from __future__ import annotations

import re
import uuid
from unittest.mock import patch

import pytest

import store
from tests.test_a_dashboards_kept_rows_follow_access import EVERYONE, _keep, _kept, _workspace
from tests.test_a_row_rule_reaches_every_table_that_links_to_it import _rule, shop  # noqa: F401 - a fixture


def _names(user: dict) -> list[str]:
    return [g["name"] for g in user["groups"]]


# ── Row rules ────────────────────────────────────────────────────────────────

def test_a_person_in_two_groups_sees_the_rows_of_each(shop):  # noqa: F811
    gold, silver = (shop.store.create_group(shop.account, name) for name in ("Gold team", "Silver team"))
    shop.store.save_row_rule(shop.account, _rule("group", gold, values=["Gold"]))
    shop.store.save_row_rule(shop.account, _rule("group", silver, values=["Silver"]))
    ada = shop.person("Ada", group=gold)
    assert shop.total(ada) == shop.truth("c.segment = 'Gold'")
    assert shop.store.add_group_members(silver, shop.account, [ada["id"]]) == 1
    ada = shop.store.get_user(ada["id"])
    assert _names(ada) == ["Gold team", "Silver team"]
    assert shop.total(ada) == shop.truth("c.segment IN ('Gold', 'Silver')")


def test_a_person_taken_out_of_a_group_loses_its_rows_at_once(shop):  # noqa: F811
    gold, silver = (shop.store.create_group(shop.account, name) for name in ("Gold team", "Silver team"))
    shop.store.save_row_rule(shop.account, _rule("group", gold, values=["Gold"]))
    shop.store.save_row_rule(shop.account, _rule("group", silver, values=["Silver"]))
    ada = shop.person("Ada", group=gold)
    shop.store.add_group_members(silver, shop.account, [ada["id"]])
    assert shop.store.remove_group_member(gold, shop.account, ada["id"])
    ada = shop.store.get_user(ada["id"])
    assert _names(ada) == ["Silver team"] and ada["group_id"] == silver
    assert shop.total(ada) == shop.truth("c.segment = 'Silver'")
    assert shop.store.remove_group_member(silver, shop.account, ada["id"])
    ada = shop.store.get_user(ada["id"])
    assert ada["groups"] == [] and ada["group_id"] is None
    assert shop.total(ada) == shop.truth(), "no rule names her now"


def test_a_group_named_with_another_groups_number_is_not_that_group(shop):  # noqa: F811
    gold = shop.store.create_group(shop.account, "Gold team")
    namesake = shop.store.create_group(shop.account, str(gold))
    shop.store.save_row_rule(shop.account, _rule("group", gold, values=["Gold"]))
    ben = shop.person("Ben", group=namesake)
    assert shop.total(ben) == shop.truth(), "a rule for the Gold team bound a member of the group named its number"
    assert shop.total(shop.person("Ada", group=gold)) == shop.truth("c.segment = 'Gold'")


def test_a_rule_on_their_groups_reads_the_name_of_every_group_they_are_in(shop):  # noqa: F811
    gold, silver = (shop.store.create_group(shop.account, name) for name in ("Gold", "Silver"))
    shop.store.replace_row_policies(shop.account, 1, [{
        "name": "Their segments", "subject_type": "role", "subject_id": "analyst", "table_fqn": "CUSTOMERS",
        "condition": {"field": "segment", "operator": "IN", "value_source": "user.groups"}}])
    ada = shop.person("Ada", group=gold)
    shop.store.add_group_members(silver, shop.account, [ada["id"]])
    assert shop.total(shop.store.get_user(ada["id"])) == shop.truth("c.segment IN ('Gold', 'Silver')")


def test_what_each_person_sees_says_every_group_of_theirs(shop):  # noqa: F811
    from admin import row_access

    gold, silver = (shop.store.create_group(shop.account, name) for name in ("Gold team", "Silver team"))
    shop.store.save_row_rule(shop.account, _rule("group", gold, values=["Gold"]))
    shop.store.save_row_rule(shop.account, _rule("group", silver, values=["Silver"]))
    ada = shop.person("Ada", group=gold)
    shop.store.add_group_members(silver, shop.account, [ada["id"]])
    people = shop.store.list_users(shop.account)
    seen = next(s for s in row_access.summary(shop.store.list_row_rules(shop.account), people, [])
                if s["id"] == ada["id"])
    assert seen["lines"] == ["Customers › segment: only Gold (Gold team); or only Silver (Silver team)"]


# ── Tables, shares, schedules and alerts ─────────────────────────────────────

@pytest.fixture
def team():
    account = f"groups-{uuid.uuid4().hex[:10]}"
    store.upsert_client(account, "portal")
    sales, east = store.create_group(account, "Sales"), store.create_group(account, "East")
    store.set_group_tables(sales, account, ["SALES"])
    store.set_group_tables(east, account, ["STOCK"])
    owner = store.create_user(account, "Owen Owner", f"owen@{account}.example.com", group_id=sales,
                              password="a-long-password-1")[0]
    ada = store.create_user(account, "Ada Analyst", f"ada@{account}.example.com", group_id=sales,
                            password="a-long-password-1")[0]
    return account, sales, east, owner, ada


def test_a_persons_tables_are_those_of_every_group_they_are_in(team):
    account, sales, east, _, ada = team
    assert store.get_allowed_tables(store.get_user(ada)) == {"SALES"}
    store.add_group_members(east, account, [ada])
    assert store.get_allowed_tables(store.get_user(ada)) == {"SALES", "STOCK"}


def test_what_is_shared_with_a_second_group_reaches_its_member(team):
    account, _, east, owner, ada = team
    board = store.create_dashboard(account, owner, "t", "East stock")
    store.add_dashboard_share(board["id"], owner, account, "group", east)
    assert store.get_dashboard_for_view(board["id"], ada, account) is None
    assert store.add_group_members(east, account, [ada]) == 1
    assert store.get_dashboard_for_view(board["id"], ada, account) is not None
    assert board["id"] in {d["id"] for d in store.list_dashboards(account, ada)}
    assert store.dashboard_audience(board["id"], account) == 1
    assert [r["id"] for r in store.share_recipients(account, "group", east, owner)] == [ada]
    assert store.list_dashboard_shares(board["id"], owner, account)[0]["members"] == 1

    from store import schedule_store, tile_alert_store

    schedule = schedule_store.create_schedule(account, board["id"], owner, next_run_at="2026-11-01 08:00:00",
                                              recipients=[("group", east)], include_owner=0)
    assert [p["id"] for p in schedule_store.schedule_people(schedule_store.get_schedule(schedule))] == [ada]
    assert schedule_store.schedule_recipients(schedule)[0]["members"] == 1
    alert = tile_alert_store.create_alert(account, board["id"], 7, owner, condition="above", threshold=10,
                                          check_minutes=60, by_email=False, next_check_at="2026-11-01 08:00:00",
                                          recipients=[("group", east)])
    assert ada in {p["id"] for p in tile_alert_store.alert_people(tile_alert_store.get_alert(alert))}
    assert [a["id"] for a in tile_alert_store.alerts_for_tile(board["id"], 7, ada, account)] == [alert]
    assert tile_alert_store.tile_marks(board["id"], ada, account) == {7: {"on": True, "firing": False}}

    from core import profile

    kinds = {(n["kind"], n["dashboard_id"]) for n in profile.notifications(store.get_user(ada), "en")}
    assert ("schedule-for-me", board["id"]) in kinds

    store.remove_group_member(east, account, ada)
    assert store.get_dashboard_for_view(board["id"], ada, account) is None
    assert schedule_store.schedule_people(schedule_store.get_schedule(schedule)) == []


def test_a_group_counts_and_lists_its_members(team):
    account, sales, east, owner, ada = team
    store.add_group_members(east, account, [ada, owner])
    groups = {g["name"]: g for g in store.list_groups(account)}
    assert groups["Sales"]["member_count"] == 2 and groups["East"]["member_count"] == 2
    assert [m["name"] for m in store.group_members(east)] == ["Ada Analyst", "Owen Owner"]
    found = store.search_people_and_groups(account, "ada")
    assert found[0]["group"] == "Sales, East"
    assert next(g for g in store.search_people_and_groups(account, "east") if g["type"] == "group")["members"] == 2


def test_only_people_of_the_groups_own_workspace_are_added(team):
    account, _, east, _, ada = team
    elsewhere = f"groups-{uuid.uuid4().hex[:10]}"
    store.upsert_client(elsewhere, "portal")
    stranger = store.create_user(elsewhere, "Sam Stranger", f"sam@{elsewhere}.example.com",
                                 password="a-long-password-1")[0]
    with pytest.raises(ValueError):
        store.add_group_members(east, account, [stranger])
    assert store.group_members(east) == []
    assert store.add_group_members(east, account, [ada, ada]) == 1
    assert store.add_group_members(east, account, [ada]) == 0, "already a member"


def test_a_group_is_named_once_in_a_workspace(team):
    account, sales, _, _, _ = team
    with pytest.raises(ValueError):
        store.create_group(account, " sales ")
    with pytest.raises(ValueError):
        store.update_group(sales, "East", "")
    store.update_group(sales, "Sales and marketing", "Every sales table")
    assert store.get_group(sales)["name"] == "Sales and marketing"


def test_a_group_deleted_takes_its_rows_tables_and_members_with_it(team):
    account, sales, east, _, ada = team
    store.add_group_members(east, account, [ada])
    store.delete_group(east)
    ada_now = store.get_user(ada)
    assert _names(ada_now) == ["Sales"] and store.get_allowed_tables(ada_now) == {"SALES"}
    store.delete_group(sales)
    ada_now = store.get_user(ada)
    assert ada_now["groups"] == [] and ada_now["group_id"] is None


def test_someone_put_in_a_group_before_is_its_member(team):
    """A person whose group was set before groups had members of their own is a member of it."""
    account, sales, east, owner, _ = team
    with store.get_db() as conn:
        cur = conn.execute("INSERT INTO portal_user (account_id, group_id, name, email, password_hash, role) "
                           "VALUES (?,?,?,?,?,?)", (account, east, "Old Timer", f"old@{account}.example.com",
                                                    "x", "analyst"))
        old = int(cur.lastrowid)
        conn.execute("DELETE FROM user_group_member WHERE user_id=?", (old,))
    store.init_db()
    assert _names(store.get_user(old)) == ["East"]
    assert _names(store.get_user(owner)) == ["Sales"]


def test_a_person_is_never_put_in_another_workspaces_group(team):
    account, _, east, _, ada = team
    elsewhere = f"groups-{uuid.uuid4().hex[:10]}"
    store.upsert_client(elsewhere, "portal")
    theirs = store.create_group(elsewhere, "Theirs")
    store.set_group_tables(theirs, elsewhere, ["PAYROLL"])
    with pytest.raises(ValueError):
        store.create_user(account, "Una Other", f"una@{account}.example.com", group_id=theirs,
                          password="a-long-password-1")
    with pytest.raises(ValueError):
        store.update_user(ada, group_id=theirs)
    with pytest.raises(ValueError):
        store.set_user_groups(ada, account, [east, theirs])
    _, pending = store.upsert_pending_user(account, "teams", f"u-{uuid.uuid4().hex[:6]}", "Pat Pending", "conv-1")
    with pytest.raises(ValueError):
        store.approve_pending_user(pending_id=int(pending["id"]), account_id=account, group_id=theirs)
    assert store.get_allowed_tables(store.get_user(ada)) == {"SALES"}
    assert store.group_members(theirs) == []


def test_kept_rows_go_when_someone_joins_or_leaves_a_group():
    here = _workspace()
    for owner in here.owners.values():
        _keep(owner.source)
    store.add_group_members(here.west, here.account_id, [here.owners["first"].id])
    assert _kept(here) == EVERYONE - {"first"}
    _keep(here.owners["first"].source)
    store.set_group_tables(here.west, here.account_id, ["SALES"])
    assert _kept(here) == EVERYONE - {"first", "third"}, "the west group's members: first is one of them now"
    for owner in here.owners.values():
        _keep(owner.source)
    store.remove_group_member(here.east, here.account_id, here.owners["second"].id)
    assert _kept(here) == EVERYONE - {"second"}
    # A rule reading "their groups" reads the group's name: renamed, what its members may see changed.
    for owner in here.owners.values():
        _keep(owner.source)
    store.update_group(here.west, "West region", "")
    assert _kept(here) == EVERYONE - {"first", "third"}
    for owner in here.owners.values():
        _keep(owner.source)
    store.update_group(here.west, "West region", "Only what it says has changed")
    assert _kept(here) == EVERYONE, "a new description changes no one's rows"


def test_an_answer_kept_in_one_workspace_never_reads_another_workspaces_person(team):
    """Whether a kept answer was given before someone's access changed reads that person's own record: another
    workspace's person with the same number, put in a group since, leaves it as it was."""
    from core.compliance import kept_answers

    _, _, east, _, ada = team
    store.add_group_members(east, team[0], [ada])
    trace = {"id": 1, "created_at": "2000-01-01 00:00:00"}
    assert kept_answers.narrowed_since(team[0], {"id": ada}, trace), "their own access changed since"
    elsewhere = f"groups-{uuid.uuid4().hex[:10]}"
    store.upsert_client(elsewhere, "portal")
    assert not kept_answers.narrowed_since(elsewhere, {"id": ada, "account_id": elsewhere}, trace)


# ── The admin pages ──────────────────────────────────────────────────────────

@pytest.fixture
def admin(team):
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from admin import routes, row_access

    app = FastAPI()
    app.include_router(routes.router)
    with patch.object(routes, "_is_auth", return_value=True), \
            patch.object(row_access, "_is_auth", return_value=True):
        yield TestClient(app), team


def test_a_group_is_made_and_opens_on_its_own_page(admin):
    client, (account, *_rest) = admin
    made = client.post(f"/admin/clients/{account}/groups/create", data={"name": "North team",
                                                                       "description": "The north region"},
                       follow_redirects=False)
    assert made.status_code == 303
    gid = int(re.search(r"/groups/(\d+)", made.headers["location"]).group(1))
    assert store.get_group(gid)["name"] == "North team"
    page = client.get(f"/admin/clients/{account}/groups/{gid}")
    assert page.status_code == 200 and "North team" in page.text and "Add people" in page.text
    again = client.post(f"/admin/clients/{account}/groups/create", data={"name": "north TEAM"},
                        follow_redirects=False)
    assert "error=" in again.headers["location"]


def test_people_are_added_to_and_taken_out_of_a_group_on_its_page(admin):
    client, (account, _, east, owner, ada) = admin
    base = f"/admin/clients/{account}/groups/{east}"
    added = client.post(base + "/members", json={"user_ids": [ada, owner]}).json()
    assert added["ok"] and added["added"] == 2
    assert [m["name"] for m in store.group_members(east)] == ["Ada Analyst", "Owen Owner"]
    page = client.get(base).text
    assert "Ada Analyst" in page and "Owen Owner" in page
    assert client.post(base + f"/members/{ada}/remove").json()["ok"]
    assert [m["name"] for m in store.group_members(east)] == ["Owen Owner"]
    events = [e["action"] for e in store.list_user_events(account)]
    assert events.count("group_changed") == 3


@pytest.fixture(scope="module")
def retail():
    from evals.core2 import domains
    from evals.core2.compile_eval import learn

    return learn(domains.build("retail"), "descriptive")[1]


def test_a_group_page_shows_its_row_rules_and_adds_one_for_it(admin, retail):
    client, (account, _, east, _, _) = admin
    store.save_core2_model(account, None, retail.model_dump_json())
    store.save_row_rule(account, _rule("group", east, field="REGION", table="STORES", values=["East"]))
    store.save_row_rule(account, _rule("everyone", field="REGION", table="STORES", values=["West"]))
    page = client.get(f"/admin/clients/{account}/groups/{east}").text
    assert "only East" in page and "only West" not in page
    assert f"/row-access?who=group&amp;whom={east}" in page
    rows = client.get(f"/admin/clients/{account}/row-access?who=group&whom={east}").text
    assert '"open": {"who": "group", "whom": "' + str(east) + '"}' in rows


def test_a_group_is_renamed_on_its_page(admin):
    client, (account, sales, east, _, _) = admin
    base = f"/admin/clients/{account}/groups/{east}"
    assert client.post(base + "/rename", json={"name": "East region", "description": "Stores in the east"}).json()["ok"]
    assert store.get_group(east)["name"] == "East region"
    refused = client.post(base + "/rename", json={"name": "Sales"})
    assert refused.status_code == 400 and "already" in refused.json()["error"]
    assert client.post(base + "/rename", json={"name": "  "}).status_code == 400


def test_a_group_or_a_person_of_another_workspace_is_refused(admin):
    client, (account, _, east, _, _) = admin
    elsewhere = f"groups-{uuid.uuid4().hex[:10]}"
    store.upsert_client(elsewhere, "portal")
    theirs = store.create_group(elsewhere, "Theirs")
    stranger = store.create_user(elsewhere, "Sam Stranger", f"sam@{elsewhere}.example.com",
                                 password="a-long-password-1")[0]
    assert client.get(f"/admin/clients/{account}/groups/{theirs}").status_code == 404
    assert client.post(f"/admin/clients/{account}/groups/{theirs}/members", json={"user_ids": []}).status_code == 404
    refused = client.post(f"/admin/clients/{account}/groups/{east}/members", json={"user_ids": [stranger]})
    assert refused.status_code == 400 and store.group_members(east) == []


def test_a_persons_groups_are_chosen_on_the_people_page(admin):
    client, (account, sales, east, _, ada) = admin
    done = client.post(f"/admin/clients/{account}/users/{ada}/groups",
                       data={"group_ids": [str(sales), str(east)]}, follow_redirects=False)
    assert done.status_code == 303
    assert _names(store.get_user(ada)) == ["Sales", "East"]
    client.post(f"/admin/clients/{account}/users/{ada}/groups", data={"group_ids": [str(east)]},
                follow_redirects=False)
    assert _names(store.get_user(ada)) == ["East"]
    client.post(f"/admin/clients/{account}/users/{ada}/groups", data={}, follow_redirects=False)
    assert store.get_user(ada)["groups"] == []
    page = client.get(f"/admin/clients/{account}/users").text
    assert "Ada Analyst" in page


def test_a_new_person_is_put_in_every_group_ticked(admin):
    client, (account, sales, east, _, _) = admin
    made = client.post(f"/admin/clients/{account}/users/create", data={
        "name": "Nia New", "email": f"nia@{account}.example.com", "group_ids": [str(east), str(sales)],
        "role": "analyst", "portal_role": "creator"}, follow_redirects=False)
    assert made.status_code == 303 and "error" not in made.headers["location"]
    nia = store.get_user_by_email(account, f"nia@{account}.example.com")
    assert _names(nia) == ["East", "Sales"] and nia["group_id"] == east
    page = client.get(f"/admin/clients/{account}/users").text
    assert f'data-groups="{east} {sales}"' in page
