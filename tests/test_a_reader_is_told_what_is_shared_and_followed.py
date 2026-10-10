"""A reader is told when a dashboard is shared with them, and at the pace they follow one.

What they are told is kept (their Notifications page lists it, newest first, and reading the page marks it
read) and, when a page of theirs is open, shown at once with a count on the sidebar. Kept first: a reader who
is away is told when they come back. The scheduler runs in a worker thread, and the live copy is handed to the
server's loop, where the sockets live.

- A share with a person tells that person; with a group, its active members; never the owner, never twice for
  the same share, never anyone for "everyone in the workspace". In the reader's own language, with a link.
- The Dashboards page lists the reader's own dashboards, then those shared with them, naming each owner.
- A follow sends, once a day, week or month, the dashboard's first number tiles as the follower sees them. A
  follower who lost access stops following and is sent nothing; an update that fails is tried again.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import threading
import uuid

import pytest

import store


@pytest.fixture()
def team():
    store.init_db()
    account = f"told-{uuid.uuid4().hex[:10]}"
    store.upsert_client(account, "portal")
    sales = store.create_group(account, "Sales team")

    def person(name, group=None):
        return store.create_user(account, name, f"{uuid.uuid4().hex[:8]}@test.com", group_id=group,
                                 password="a-password-they-chose")[0]

    people = {"owner": person("Ada Owner"), "riley": person("Riley Sales", sales), "sam": person("Sam Sales", sales),
              "solo": person("Solo Nobody")}
    store.set_user_language(people["sam"], "fr")
    dashboard = store.create_dashboard(account, people["owner"], "thread", "Sales overview")
    try:
        yield {"account": account, "sales": sales, "dashboard": dashboard["id"], **people}
    finally:
        with store.get_db() as conn:
            for table in ("portal_notice", "dashboard_share", "dashboard_subscription", "pinned_chart",
                          "dashboard_artifact_version", "dashboard_artifact", "portal_user", "user_group", "client"):
                conn.execute(f"DELETE FROM {table} WHERE account_id=?", (account,))


def _notices(t, who):
    return store.list_notices(t["account"], t[who])


class _Page:
    """An open portal page: its notification socket, which -- as a real one -- is written to only on the loop
    that accepted it."""

    def __init__(self):
        self.received = []
        self.loop = None

    async def accept(self):
        self.loop = asyncio.get_running_loop()

    async def send_json(self, payload):
        if asyncio.get_running_loop() is not self.loop:
            raise RuntimeError("a socket is written to from the loop that accepted it")
        self.received.append(payload)


@pytest.fixture()
def hub(monkeypatch):
    """The server's notification hub, on a loop of its own in another thread, as under uvicorn."""
    import core.portal_notifications as notifications

    fresh = notifications.PortalNotificationHub()
    monkeypatch.setattr(notifications, "portal_notification_hub", fresh)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    def open_page(account_id, user_id):
        page = _Page()
        asyncio.run_coroutine_threadsafe(fresh.connect(page, account_id=account_id, user_id=user_id),
                                         loop).result(5)
        return page

    try:
        yield open_page
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        loop.close()


# ── kept notices ──────────────────────────────────────────────────────────


def test_a_notice_is_kept_listed_newest_first_and_read_once_listed(team):
    first = store.add_notice(team["account"], team["riley"], "dashboard_shared", "First", "", "/portal/dashboard")
    second = store.add_notice(team["account"], team["riley"], "dashboard_shared", "Second", "body",
                              "https://elsewhere.example.com/x")
    assert [n["id"] for n in _notices(team, "riley")] == [second, first]
    assert _notices(team, "riley")[0]["link"] == ""            # never a link out of the portal
    assert store.unread_notice_count(team["account"], team["riley"]) == 2
    assert store.unread_notice_count(team["account"], team["sam"]) == 0
    assert store.mark_notices_read(team["account"], team["riley"]) == 2
    assert store.unread_notice_count(team["account"], team["riley"]) == 0


def test_each_reader_keeps_only_their_latest_notices(team, monkeypatch):
    # The module add_notice itself reads (another test may have imported the store afresh).
    monkeypatch.setitem(store.add_notice.__globals__, "KEPT", 3)
    for n in range(5):
        store.add_notice(team["account"], team["riley"], "k", f"N{n}")
    store.add_notice(team["account"], team["sam"], "k", "Sam's")
    assert [n["title"] for n in _notices(team, "riley")] == ["N4", "N3", "N2"]
    assert [n["title"] for n in _notices(team, "sam")] == ["Sam's"]


def test_a_scheduled_job_tells_a_reader_on_their_open_page_from_its_own_thread(team, hub):
    from core.notices import tell

    page = hub(team["account"], team["riley"])
    other = hub(team["account"], team["sam"])
    notice_id = tell(team["account"], team["riley"], "dashboard_follow", "Your daily update", "Sales: 4",
                     "/portal/dashboard?dashboard_id=1")
    (payload,) = page.received
    assert payload == {"type": "notice", "id": notice_id, "kind": "dashboard_follow", "title": "Your daily update",
                       "body": "Sales: 4", "link": "/portal/dashboard?dashboard_id=1", "unread": 1}
    assert other.received == []
    assert [n["id"] for n in _notices(team, "riley")] == [notice_id]


def test_a_reader_with_no_page_open_is_told_when_they_come_back(team, hub):
    from core.notices import tell

    tell(team["account"], team["riley"], "dashboard_follow", "Your daily update")
    assert [n["title"] for n in _notices(team, "riley")] == ["Your daily update"]


def test_a_route_tells_a_reader_on_the_servers_own_loop(team, monkeypatch):
    import core.portal_notifications as notifications
    from core.notices import tell_async

    fresh = notifications.PortalNotificationHub()
    monkeypatch.setattr(notifications, "portal_notification_hub", fresh)

    async def scene():
        page = _Page()
        await fresh.connect(page, account_id=team["account"], user_id=team["riley"])
        await tell_async(team["account"], team["riley"], "dashboard_shared", "Shared")
        return page

    page = asyncio.run(scene())
    assert [p["title"] for p in page.received] == ["Shared"]


# ── over HTTP ─────────────────────────────────────────────────────────────


@pytest.fixture()
def http(monkeypatch):
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    import core.alert_engine
    import portal.routes as routes

    monkeypatch.setattr(core.alert_engine, "list_alerts", lambda: [])
    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)

    def as_(user_id):
        client.cookies.set(routes._COOKIE, routes._sign_session_value(user_id))
        return client

    return as_


def _share(client, t, body):
    reply = client.post(f"/portal/api/dashboard/{t['dashboard']}/shares", json=body)
    assert reply.status_code == 200, reply.text
    return reply


def test_a_person_shared_with_is_told_with_a_link_and_no_one_else(team, http):
    _share(http(team["owner"]), team, {"subject_type": "user", "subject_id": team["riley"]})
    (told,) = _notices(team, "riley")
    assert told["title"] == "Ada Owner shared “Sales overview” with you" and told["body"] == ""
    assert told["link"] == f"/portal/dashboard?dashboard_id={team['dashboard']}"
    assert not any(_notices(team, w) for w in ("owner", "sam", "solo"))
    _share(http(team["owner"]), team, {"subject_type": "user", "subject_id": team["riley"]})
    assert len(_notices(team, "riley")) == 1                     # the same share again tells no one


def test_a_group_shared_with_tells_each_member_in_their_language(team, http):
    _share(http(team["owner"]), team, {"subject_type": "group", "subject_id": team["sales"]})
    (riley,) = _notices(team, "riley")
    (sam,) = _notices(team, "sam")
    assert (riley["title"], riley["body"]) == ("Ada Owner shared “Sales overview” with you", "Shared with Sales team.")
    assert (sam["title"], sam["body"]) == ("Ada Owner a partagé « Sales overview » avec vous",
                                           "Partagé avec Sales team.")
    assert not _notices(team, "owner") and not _notices(team, "solo")


def test_a_group_share_never_tells_its_owner_nor_a_member_who_cannot_sign_in(team, http):
    store.update_user(team["owner"], group_id=team["sales"])
    store.update_user(team["sam"], is_active=0)
    _share(http(team["owner"]), team, {"subject_type": "group", "subject_id": team["sales"]})
    assert [n["title"] for n in _notices(team, "riley")] == ["Ada Owner shared “Sales overview” with you"]
    assert not _notices(team, "owner") and not _notices(team, "sam")


def test_sharing_with_the_whole_workspace_tells_no_one(team, http):
    _share(http(team["owner"]), team, {"workspace": True})
    assert not any(_notices(team, w) for w in ("riley", "sam", "solo"))


def test_the_unread_count_is_the_readers_own(team, http):
    _share(http(team["owner"]), team, {"subject_type": "user", "subject_id": team["riley"]})
    assert http(team["riley"]).get("/portal/api/notices/unread").json() == {"ok": True, "unread": 1}
    assert http(team["sam"]).get("/portal/api/notices/unread").json() == {"ok": True, "unread": 0}
    client = http(team["riley"])
    client.cookies.clear()
    assert client.get("/portal/api/notices/unread").status_code == 401


def test_the_notifications_page_lists_what_the_reader_was_told_and_reads_it(team, http):
    _share(http(team["owner"]), team, {"subject_type": "user", "subject_id": team["riley"]})
    page = http(team["riley"]).get("/portal/notifications")
    assert page.status_code == 200
    assert "Ada Owner shared “Sales overview” with you" in page.text
    assert f'href="/portal/dashboard?dashboard_id={team["dashboard"]}"' in page.text
    assert store.unread_notice_count(team["account"], team["riley"]) == 0
    assert "Ada Owner shared" not in http(team["sam"]).get("/portal/notifications").text


def test_the_dashboards_page_lists_the_readers_own_then_those_shared_with_them(team, http):
    store.create_dashboard(team["account"], team["riley"], "thread", "Riley's own board")
    _share(http(team["owner"]), team, {"subject_type": "group", "subject_id": team["sales"]})
    text = http(team["riley"]).get("/portal/dashboard").text
    heading = ">Shared with you</h2>"                       # the page's own words also carry the phrase
    mine, shared = text.index(">Riley&#39;s own board<"), text.index(heading)
    assert mine < shared < text.index(">Sales overview<") and ">Shared by Ada Owner" in text
    assert heading not in http(team["solo"]).get("/portal/dashboard").text


# ── follows delivered ─────────────────────────────────────────────────────


def _follow(t, who, cadence="daily"):
    row = store.subscribe_dashboard(t["dashboard"], t[who], t["account"], cadence=cadence)
    return dt.datetime.fromisoformat(str(row["created_at"])[:19])


def _tile(t, title, kind="kpi"):
    return store.pin_chart(t["owner"], t["account"], title, f"What is {title}?", "SELECT 1", kind, None,
                           dashboard_id=t["dashboard"])


@pytest.fixture()
def only_here(team, monkeypatch):
    """Follows of this workspace only: other tests leave follows of their own in the same database, and a run
    sends every one that is due."""
    import core.dashboard_follow as follow

    real = follow.store.list_active_follows           # the store the follows are read from
    monkeypatch.setattr(follow.store, "list_active_follows",
                        lambda: [f for f in real() if f["account_id"] == team["account"]])


@pytest.fixture()
def drawn(only_here, monkeypatch):
    """The tiles as drawn for a reader: the number each one shows, or an error; and whom each was drawn for."""
    import portal.routes as routes
    from core.i18n import get_active_language

    state = {"values": {}, "drawn_for": []}

    def draw(chart, db_cfg, user, **_):
        state["drawn_for"].append((chart["title"], int(user["id"]), get_active_language()))
        value = state["values"].get(chart["title"])
        if value is None:
            return {**chart, "error": "This chart could not be refreshed.", "kpi": None}
        return {**chart, "error": None, "kpi": {"value": 1}, "kpi_display": value}

    monkeypatch.setattr(routes, "_refresh_chart", draw)
    return state


def test_a_daily_follow_is_sent_a_day_after_with_the_headline_numbers(team, drawn):
    from core.dashboard_follow import run_due_follows

    store.add_dashboard_share(team["dashboard"], team["owner"], team["account"], "user", team["riley"])
    for title in ("Revenue", "Orders", "Sales by region", "Margin", "Returns"):
        _tile(team, title, "bar" if title == "Sales by region" else "kpi")
    drawn["values"] = {"Revenue": "1.2M", "Orders": "3,400", "Margin": "12%", "Returns": "88"}
    since = _follow(team, "riley")
    assert run_due_follows(since + dt.timedelta(hours=23)) == 0 and not _notices(team, "riley")
    assert run_due_follows(since + dt.timedelta(days=1, minutes=1)) == 1
    (told,) = _notices(team, "riley")
    assert told["title"] == "“Sales overview”: your daily update"
    assert told["body"] == "Revenue: 1.2M · Orders: 3,400 · Margin: 12%"          # the first three number tiles
    assert told["link"] == f"/portal/dashboard?dashboard_id={team['dashboard']}"
    assert {who for _, who, _ in drawn["drawn_for"]} == {team["riley"]}            # as the follower sees them
    assert run_due_follows(since + dt.timedelta(days=1, minutes=2)) == 0          # once a day
    assert run_due_follows(since + dt.timedelta(days=2, minutes=2)) == 1
    assert len(_notices(team, "riley")) == 2


def test_a_weekly_follow_waits_a_week_and_speaks_the_followers_language(team, drawn):
    from core.dashboard_follow import run_due_follows

    store.add_dashboard_share(team["dashboard"], team["owner"], team["account"], "group", team["sales"])
    _tile(team, "Revenue")
    drawn["values"] = {"Revenue": "1,2 M"}
    since = _follow(team, "sam", "weekly")
    assert run_due_follows(since + dt.timedelta(days=6)) == 0
    assert run_due_follows(since + dt.timedelta(days=7, minutes=1)) == 1
    (told,) = _notices(team, "sam")
    assert told["title"] == "« Sales overview » : votre point hebdomadaire" and told["body"] == "Revenue: 1,2 M"
    assert drawn["drawn_for"] == [("Revenue", team["sam"], "fr")]


def test_a_tile_that_cannot_be_drawn_is_left_out_and_none_leaves_a_plain_update(team, drawn):
    from core.dashboard_follow import run_due_follows

    store.add_dashboard_share(team["dashboard"], team["owner"], team["account"], "user", team["riley"])
    _tile(team, "Revenue")
    since = _follow(team, "riley")
    assert run_due_follows(since + dt.timedelta(days=1, minutes=1)) == 1
    (told,) = _notices(team, "riley")
    assert told["body"] == "Open it to see where things stand."


def test_a_follower_who_lost_access_stops_following_and_is_sent_nothing(team, drawn):
    from core.dashboard_follow import run_due_follows

    store.add_dashboard_share(team["dashboard"], team["owner"], team["account"], "group", team["sales"])
    since = _follow(team, "riley")
    store.update_user(team["riley"], group_id=store.create_group(team["account"], "Elsewhere"))
    assert run_due_follows(since + dt.timedelta(days=1, minutes=1)) == 0
    assert not _notices(team, "riley") and drawn["drawn_for"] == []
    assert store.get_dashboard_subscription(team["dashboard"], team["riley"], team["account"]) is None


def test_a_deactivated_follower_is_sent_nothing(team, drawn):
    from core.dashboard_follow import run_due_follows

    store.add_dashboard_share(team["dashboard"], team["owner"], team["account"], "user", team["riley"])
    since = _follow(team, "riley")
    store.update_user(team["riley"], is_active=0)
    assert run_due_follows(since + dt.timedelta(days=1, minutes=1)) == 0 and not _notices(team, "riley")


def test_the_owner_following_their_own_dashboard_is_sent_it(team, drawn):
    from core.dashboard_follow import run_due_follows

    since = _follow(team, "owner", "monthly")
    assert run_due_follows(since + dt.timedelta(days=29)) == 0
    assert run_due_follows(since + dt.timedelta(days=30, minutes=1)) == 1
    assert _notices(team, "owner")[0]["title"] == "“Sales overview”: your monthly update"


def test_an_update_that_fails_is_tried_again_at_the_next_run(team, drawn, monkeypatch):
    import core.notices
    from core.dashboard_follow import run_due_follows

    store.add_dashboard_share(team["dashboard"], team["owner"], team["account"], "user", team["riley"])
    since = _follow(team, "riley")
    real = core.notices.tell

    def broken(*args, **kwargs):
        raise RuntimeError("the store is locked")

    monkeypatch.setattr(core.notices, "tell", broken)
    assert run_due_follows(since + dt.timedelta(days=1, minutes=1)) == 0
    monkeypatch.setattr(core.notices, "tell", real)
    assert run_due_follows(since + dt.timedelta(days=1, minutes=2)) == 1
    assert len(_notices(team, "riley")) == 1


def test_two_schedulers_never_send_the_same_update_twice(team):
    _follow(team, "owner")
    (follow,) = [f for f in store.list_active_follows() if f["account_id"] == team["account"]]
    assert store.claim_follow(follow["id"], None, "2030-01-01 00:00:00")
    assert not store.claim_follow(follow["id"], None, "2030-01-01 00:00:00")
    assert store.claim_follow(follow["id"], "2030-01-01 00:00:00", "2030-01-02 00:00:00")


def test_the_scheduler_runs_the_follows(monkeypatch):
    import core.dashboard_follow
    import core.notification_scheduler as scheduler

    ran = []
    for name in ("core.alert_engine.run_due_alert_checks", "core.report_engine.run_due_report_digests",
                 "core.dashboard_refresh.run_due_dashboard_refreshes"):
        monkeypatch.setattr(name, lambda: None)
    monkeypatch.setattr(core.dashboard_follow, "run_due_follows", lambda: ran.append(True))
    scheduler.run_due_notifications_once()
    assert ran == [True]


# ── the number a follower is sent is drawn by the dashboard's own tile code ──


def test_a_number_tile_is_drawn_for_the_follower_under_their_own_access(team, only_here, monkeypatch):
    """No stand-in for the tile: only the warehouse. The update's number is the tile's own, drawn with the
    follower's tables."""
    from types import SimpleNamespace

    from core.compliance.sql_guard import analyze_sql
    from core.dashboard_follow import run_due_follows

    ran = []

    def governed(credentials, db_type, sql, *args, **kwargs):
        ran.append(kwargs.get("allowed_tables"))
        return SimpleNamespace(rows=[{"REVENUE": 1234567.0}], sql=sql, analysis=analyze_sql(sql, db_type))

    monkeypatch.setattr("core.compliance.governed_query.execute_governed_query", governed)
    db_id = store.save_db_config("azure_sql", f"Warehouse {team['account']}", {
        "server": "dw.example.net", "database": "DW", "user": "reader", "password": "a-password"})
    store.pin_chart(team["owner"], team["account"], "Revenue", "What is revenue?",
                    "SELECT SUM(AMOUNT) AS REVENUE FROM SALES", "kpi", db_id, dashboard_id=team["dashboard"])
    store.set_group_tables(team["sales"], team["account"], ["SALES"])
    store.add_dashboard_share(team["dashboard"], team["owner"], team["account"], "group", team["sales"])
    since = _follow(team, "riley")
    assert run_due_follows(since + dt.timedelta(days=1, minutes=1)) == 1
    (told,) = _notices(team, "riley")
    assert told["body"] == "Revenue: 1,234,567"
    assert ran == [store.get_allowed_tables(store.get_user(team["riley"]))]
