"""
A dashboard refresh that fails is told, tried again later, and never shown as
fresh.

A scheduled refresh keeps a dashboard source's rows, encrypted, for its owner.
When one failed, the failure stamped the cache's refreshed_at: rows a day old
were shown as "Protected cache" of a moment ago. A source that had never been
refreshed recorded nothing at all, and was due again at the scheduler's next
tick -- every minute, for as long as the warehouse refused it. Nobody was told:
the failure went to the log.

Now a failure keeps the time of the last refresh that worked, and the rows,
and the owner's chart goes on showing them, saying they are from before the
failures. Each failure in a row waits twice as long as the one before it,
from five minutes up to the dashboard's own cadence. The owner is told once
per dashboard, in their language, when a second refresh in a row has failed
-- since when, what the dashboard shows, and when it is tried again -- and a
notice that reached no one is sent at a later failure. Every time is UTC, and
says so. A dashboard whose charts all refresh again clears all of it.

A scratch store holds the owner, the dashboard, its sources and their cache;
the warehouse and the delivery of the notice are the only stand-ins.
"""

from __future__ import annotations

import os
import sqlite3
import time
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import core.dashboard_refresh as refresh
import store

ROWS = [{"WAREHOUSE": "NORTH DEPOT", "STOCK_ON_HAND": 840.0}]
YESTERDAY = "2026-09-27 06:00:00"


def _now() -> datetime:
    """Now in UTC, the clock every time the cache keeps is on."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


@pytest.fixture
def dashboard(monkeypatch):
    """An owner, a daily dashboard with one source, and a warehouse that
    answers until the test says it is down."""
    account_id = f"refresh-fails-{uuid.uuid4().hex[:10]}"
    store.upsert_client(account_id, "portal")
    owner_id, _ = store.create_user(account_id, "Owner", f"{account_id}@example.com", password="a-long-password-1")
    board = store.create_dashboard(account_id, owner_id, "t", "Stock by warehouse")
    store.update_dashboard_controls(board["id"], owner_id, account_id, refresh_schedule="daily")
    db_id = store.save_db_config("azure_sql", f"Warehouse {account_id}", {
        "server": "dw.example.net", "database": "DW", "user": "reader", "password": "a-password"})
    source = store.create_data_source(
        board["id"], owner_id, account_id, name="Stock", question="Stock on hand by warehouse",
        sql_query="SELECT WAREHOUSE, SUM(STOCK_ON_HAND) AS STOCK_ON_HAND FROM STOCK GROUP BY WAREHOUSE",
        db_config_id=db_id)
    # "up_for": the queries that still answer while the warehouse is down.
    warehouse = {"down": False, "up_for": set(), "ttl": 86400}

    def governed(*args, **kwargs):
        if warehouse["down"] and args[2] not in warehouse["up_for"]:
            raise ConnectionError("The warehouse refused the connection.")
        return SimpleNamespace(rows=list(ROWS), sql=args[2],
                               decision=SimpleNamespace(cache_ttl_seconds=warehouse["ttl"]))

    # A notice reaches the owner when they have a portal page or a Teams chat
    # open; "online" says whether they do.
    owner = {"online": True}
    told: list[tuple[str, int, str]] = []

    async def deliver(account, user_id, message, chart=None):
        if owner["online"]:
            told.append((account, user_id, message))
        return owner["online"]

    monkeypatch.setattr("core.compliance.governed_query.execute_governed_query", governed)
    monkeypatch.setattr("core.notify.send_proactive_notification", deliver)
    return SimpleNamespace(account_id=account_id, owner_id=owner_id, board=board, source=source,
                           warehouse=warehouse, owner=owner, told=told)


def _cache(d) -> dict:
    with store.get_db() as conn:
        row = conn.execute("SELECT * FROM dashboard_source_cache WHERE source_id=?", (d.source["id"],)).fetchone()
    return dict(row) if row else {}


def _refreshed_yesterday(d) -> None:
    """The last refresh that worked, a day ago, and still served."""
    assert refresh.refresh_dashboard_source(_due(d)) is True
    with store.get_db() as conn:
        conn.execute("UPDATE dashboard_source_cache SET refreshed_at=? WHERE source_id=?",
                     (YESTERDAY, d.source["id"]))


def _due(d, at: datetime | None = None) -> dict | None:
    """The source as the scheduler is handed it, if it is due."""
    return next((s for s in store.list_due_dashboard_sources(at) if s["id"] == d.source["id"]), None)


def _fail(d, source_id: int | None = None) -> None:
    """A source's next attempt, whenever it comes, and the warehouse down."""
    at = _now() + timedelta(days=2)
    source = next(s for s in store.list_due_dashboard_sources(at) if s["id"] == (source_id or d.source["id"]))
    assert refresh.refresh_dashboard_source(source) is False


def _refresh_everything(d) -> None:
    """The next attempt of every source of the dashboard."""
    for source in store.list_due_dashboard_sources(_now() + timedelta(days=2)):
        if source["dashboard_id"] == d.board["id"]:
            refresh.refresh_dashboard_source(source)


def _owners_chart(d) -> dict:
    """The owner's chart, as their dashboard page renders it."""
    from portal.routes import _refresh_chart
    from tests.dashboard_render import CHART

    return _refresh_chart({**CHART, "data_source_id": d.source["id"], "user_id": d.owner_id, "chart_type": "bar",
                           "sql_query": d.source["sql_query"]},
                          store.get_db_config(d.source["db_config_id"]), store.get_user(d.owner_id))


def _source(d) -> dict:
    """The source with its schedule, due or not."""
    return {**d.source, "refresh_schedule": "daily", "dashboard_name": d.board["name"]}


class TestAFailure:

    def test_is_not_a_refresh(self, dashboard):
        _refreshed_yesterday(dashboard)
        dashboard.warehouse["down"] = True
        with store.get_db() as conn:
            conn.execute("UPDATE dashboard_source_cache SET expires_at='2000-01-01 00:00:00' WHERE source_id=?",
                         (dashboard.source["id"],))
        assert refresh.refresh_dashboard_source(_due(dashboard)) is False
        cache = _cache(dashboard)
        assert cache["refreshed_at"] == YESTERDAY
        assert (cache["status"], cache["failure_count"]) == ("error", 1)

    def test_keeps_the_rows_and_the_chart_says_they_are_from_before_it(self, dashboard):
        from tests.dashboard_render import render, visible

        _refreshed_yesterday(dashboard)
        dashboard.warehouse["down"] = True
        assert refresh.refresh_dashboard_source(_source(dashboard)) is False
        chart = _owners_chart(dashboard)
        assert (chart["from_cache"], chart["row_count"]) == (True, 1)
        page = visible(render(charts=[chart], lang="fr"))
        assert "ces données sont celles du 2026-09-27 06:00 UTC." in page
        since = _cache(dashboard)["failed_at"][:16]
        assert f"L'actualisation planifiée échoue depuis le {since} UTC : ces données" in page

    def test_the_chart_keeps_the_rows_once_they_expire(self, dashboard):
        """A source whose governed decision sets no cache life keeps its rows
        ten minutes, and the scheduler tries again only once they expire: the
        owner's chart then showed the warehouse's error, while the notice said
        the dashboard still showed the data."""
        dashboard.warehouse["ttl"] = 0
        assert refresh.refresh_dashboard_source(_due(dashboard)) is True
        with store.get_db() as conn:
            conn.execute("UPDATE dashboard_source_cache SET expires_at='2000-01-01 00:00:00' WHERE source_id=?",
                         (dashboard.source["id"],))
        dashboard.warehouse["down"] = True
        assert refresh.refresh_dashboard_source(_due(dashboard)) is False
        chart = _owners_chart(dashboard)
        assert (chart["from_cache"], chart["row_count"], chart["refresh_failed_at"]) == (
            True, 1, _cache(dashboard)["failed_at"])

    def test_a_first_one_is_kept_and_waits(self, dashboard):
        dashboard.warehouse["down"] = True
        now = _now()
        assert refresh.refresh_dashboard_source(_due(dashboard)) is False
        assert _due(dashboard, now) is None
        assert _due(dashboard, now + timedelta(minutes=6)) is not None
        cache = _cache(dashboard)
        assert (cache["failure_count"], cache["refreshed_at"], cache["row_count"]) == (1, None, 0)
        # Never served as rows.
        assert store.get_source_cache(dashboard.source["id"], dashboard.owner_id, dashboard.account_id,
                                      allow_stale=True) is None


class TestTheNextAttempt:

    @pytest.mark.parametrize("schedule,waits", [
        ("daily", [5, 10, 20, 40, 80, 160, 320, 640, 1280, 1440, 1440]),
        ("hourly", [5, 10, 20, 40, 60, 60]),
        ("weekly", [5, 10, 20]),
    ])
    def test_waits_twice_as_long_each_time_up_to_the_cadence(self, dashboard, schedule, waits):
        start = datetime(2026, 9, 28, 6, 0, 0)
        seen = []
        for attempt in range(len(waits)):
            now = start + timedelta(days=attempt)
            failure = store.mark_source_cache_error({**dashboard.source, "refresh_schedule": schedule}, "down", now=now)
            seen.append((datetime.strptime(failure["next_attempt_at"], "%Y-%m-%d %H:%M:%S") - now) / timedelta(minutes=1))
        assert seen == waits
        assert failure["failed_at"] == "2026-09-28 06:00:00"

    def test_a_long_run_of_failures_still_waits_the_cadence(self, dashboard):
        with store.get_db() as conn:
            conn.execute(
                """INSERT INTO dashboard_source_cache (source_id, dashboard_id, account_id, user_id,
                       rows_encrypted, expires_at, failure_count) VALUES (?, ?, ?, ?, '', '2000-01-01', 5000)""",
                (dashboard.source["id"], dashboard.board["id"], dashboard.account_id, dashboard.owner_id))
        now = datetime(2026, 9, 28, 6, 0, 0)
        failure = store.mark_source_cache_error(_source(dashboard), "down", now=now)
        assert failure["next_attempt_at"] == "2026-09-29 06:00:00"


class TestTheOwner:

    def test_is_told_once_in_their_language(self, dashboard):
        store.set_user_language(dashboard.owner_id, "fr")
        _refreshed_yesterday(dashboard)
        dashboard.warehouse["down"] = True
        _fail(dashboard)
        assert dashboard.told == []
        _fail(dashboard)
        ((account, user_id, message),) = dashboard.told
        cache = _cache(dashboard)
        assert (account, user_id) == (dashboard.account_id, dashboard.owner_id)
        assert message == (
            f"Votre tableau de bord « Stock by warehouse » n'a pas pu être actualisé : 2 tentatives de suite ont "
            f"échoué depuis le {cache['failed_at'][:16]} UTC. Il affiche toujours les données du 2026-09-27 06:00 "
            f"UTC. La prochaine tentative aura lieu le {cache['next_attempt_at'][:16]} UTC.")
        _fail(dashboard)
        assert len(dashboard.told) == 1

    def test_is_told_again_after_a_refresh_that_worked(self, dashboard):
        dashboard.warehouse["down"] = True
        _fail(dashboard)
        _fail(dashboard)
        assert "it has no data yet" in dashboard.told[0][2]
        dashboard.warehouse["down"] = False
        assert refresh.refresh_dashboard_source(_due(dashboard, _now() + timedelta(days=2))) is True
        cache = _cache(dashboard)
        assert (cache["failure_count"], cache["failed_at"], cache["next_attempt_at"], cache["owner_notified_at"],
                cache["status"]) == (0, None, None, None, "ready")
        dashboard.warehouse["down"] = True
        _fail(dashboard)
        _fail(dashboard)
        assert len(dashboard.told) == 2

    def test_an_owner_no_longer_active_is_not(self, dashboard):
        with store.get_db() as conn:
            conn.execute("UPDATE portal_user SET is_active=0 WHERE id=?", (dashboard.owner_id,))
        _fail(dashboard)
        _fail(dashboard)
        assert dashboard.told == []
        assert _cache(dashboard)["failure_count"] == 2

    def test_one_away_is_told_at_a_later_failure(self, dashboard):
        """A nightly refresh fails while the owner has no page open: the
        notice was marked as told when the failure was counted, before it was
        sent, and reached no one."""
        dashboard.owner["online"] = False
        dashboard.warehouse["down"] = True
        _fail(dashboard)
        _fail(dashboard)
        assert (dashboard.told, _cache(dashboard)["owner_notified_at"]) == ([], None)
        dashboard.owner["online"] = True
        _fail(dashboard)
        ((_, _, message),) = dashboard.told
        assert "3 attempts in a row" in message

    def test_one_claim_of_the_notice(self, dashboard):
        """Two refreshes failing at once claim the notice once."""
        store.mark_source_cache_error(_source(dashboard), "down")
        assert store.claim_owner_notice(_source(dashboard)) is not None
        assert store.claim_owner_notice(_source(dashboard)) is None


class TestADashboardOfSeveralCharts:
    """Its owner was sent one notice per chart, each the same."""

    @pytest.fixture
    def charts(self, dashboard):
        for name in ("Receipts", "Returns"):
            store.create_data_source(
                dashboard.board["id"], dashboard.owner_id, dashboard.account_id, name=name, question=name,
                sql_query=f"SELECT COUNT(*) AS {name.upper()} FROM {name.upper()}",
                db_config_id=dashboard.source["db_config_id"])
        return dashboard

    def test_is_told_once(self, charts):
        charts.warehouse["down"] = True
        for _ in range(3):
            _refresh_everything(charts)
        assert len(charts.told) == 1

    def test_is_told_again_only_once_every_chart_is_back(self, charts):
        charts.warehouse["down"] = True
        _refresh_everything(charts)
        _refresh_everything(charts)
        # One chart answers again; the others are still failing.
        charts.warehouse["up_for"] = {charts.source["sql_query"]}
        _refresh_everything(charts)
        charts.warehouse["up_for"] = set()
        _refresh_everything(charts)
        _refresh_everything(charts)
        assert len(charts.told) == 1
        charts.warehouse["down"] = False
        _refresh_everything(charts)
        charts.warehouse["down"] = True
        _refresh_everything(charts)
        _refresh_everything(charts)
        assert len(charts.told) == 2


@pytest.fixture
def east_of_utc():
    """A server whose clock is two hours ahead of UTC in summer."""
    before = os.environ.get("TZ")
    os.environ["TZ"] = "Europe/Paris"
    time.tzset()
    yield
    if before is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = before
    time.tzset()


class TestOneClock:
    """A refresh was stamped in UTC and its failures and next attempt in the
    server's time, so on a server not on UTC the notice put the failures hours
    away from the refresh, and the scheduler compared the two."""

    def test_the_failures_and_the_refresh_they_follow(self, dashboard, east_of_utc):
        assert refresh.refresh_dashboard_source(_due(dashboard)) is True
        dashboard.warehouse["down"] = True
        _fail(dashboard)
        _fail(dashboard)
        cache = _cache(dashboard)
        failed, refreshed = (datetime.strptime(cache[key], "%Y-%m-%d %H:%M:%S") for key in ("failed_at", "refreshed_at"))
        assert timedelta(0) <= failed - refreshed < timedelta(minutes=1)
        ((_, _, message),) = dashboard.told
        assert f"since {cache['failed_at'][:16]} UTC. It still shows the data from {cache['refreshed_at'][:16]} UTC" in message

    def test_an_hourly_source_refreshed_a_moment_ago_is_not_due(self, dashboard, east_of_utc):
        store.update_dashboard_controls(dashboard.board["id"], dashboard.owner_id, dashboard.account_id,
                                        refresh_schedule="hourly")
        dashboard.warehouse["ttl"] = 7200
        assert refresh.refresh_dashboard_source({**_source(dashboard), "refresh_schedule": "hourly"}) is True
        assert _due(dashboard) is None


class TestTheScheduler:

    def test_a_dashboard_deleted_during_its_refresh_stops_no_other(self, dashboard, monkeypatch):
        other = store.create_dashboard(dashboard.account_id, dashboard.owner_id, "t", "Receipts")
        store.update_dashboard_controls(other["id"], dashboard.owner_id, dashboard.account_id, refresh_schedule="daily")
        store.create_data_source(other["id"], dashboard.owner_id, dashboard.account_id, name="Receipts",
                                 question="Receipts", sql_query="SELECT COUNT(*) AS RECEIPTS FROM RECEIPTS",
                                 db_config_id=dashboard.source["db_config_id"])
        attempted = []

        def governed(*args, **kwargs):
            attempted.append(args[2])
            if len(attempted) == 1:
                with store.get_db() as conn:
                    conn.execute("DELETE FROM dashboard_artifact WHERE id=?", (dashboard.board["id"],))
            raise ConnectionError("The query timed out.")

        everyones = store.list_due_dashboard_sources
        monkeypatch.setattr("core.compliance.governed_query.execute_governed_query", governed)
        monkeypatch.setattr(store, "list_due_dashboard_sources", lambda now=None: [
            source for source in everyones(now) if source["account_id"] == dashboard.account_id])
        assert refresh.run_due_dashboard_refreshes() == {"due": 2, "refreshed": 0, "failed": 2}
        assert len(attempted) == 2


def test_a_cache_kept_before_this_release_is_upgraded(tmp_path, monkeypatch):
    """The four columns reach a store whose cache table is from before them."""
    path = tmp_path / "before.db"
    with sqlite3.connect(path) as conn:
        conn.execute("""CREATE TABLE dashboard_source_cache (
            source_id INTEGER PRIMARY KEY, dashboard_id INTEGER NOT NULL, account_id TEXT NOT NULL,
            user_id INTEGER NOT NULL, rows_encrypted TEXT NOT NULL, row_count INTEGER NOT NULL DEFAULT 0,
            policy_version INTEGER NOT NULL DEFAULT 0, contract_version TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'ready', error_message TEXT NOT NULL DEFAULT '',
            refreshed_at TEXT DEFAULT (datetime('now')), expires_at TEXT NOT NULL)""")
    monkeypatch.setenv("QUERYBOT_DB_PATH", str(path))
    monkeypatch.setenv("DB_PATH", str(path))
    store.init_db()
    with sqlite3.connect(path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(dashboard_source_cache)")}
    assert {"failed_at", "failure_count", "next_attempt_at", "owner_notified_at"} <= columns
