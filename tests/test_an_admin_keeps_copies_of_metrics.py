"""An admin keeps copies of metrics on Data > Precomputed copies; Solviz makes them on the workspace's connection.

    from the page       a copy is added by metric, fields, date, day or month and schedule; one that cannot be kept
                        (a balance, a field of people's names) is refused with why, and nothing is kept
    made at once        it is made in the background, on the workspace's own connection, into a file of its own;
                        a copy that would hold too many rows, or reach a table holding a key twice, is not made,
                        and the page says why
    answered from       once made, the portal's answers read it (and say so); the answers check never does
    made again          the minute scheduler makes each copy again on its schedule, and retries a failed one later
    taken out           its file is deleted with it
    workspace admins    it keeps the workspace's figures on this server: a data steward does not reach it

Invented data only (the retail business of evals/core2).
"""

from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

import store
from core2 import copies
from core2 import precompute as PC
from core2.warehouse.runner import DuckDBWarehouse
from tests.test_core2_answers_a_question_end_to_end import _domain


@pytest.fixture(autouse=True)
def _kept_here(tmp_path, monkeypatch):
    monkeypatch.setenv("QUERYBOT_COPIES_DIR", str(tmp_path / "copies"))


def _workspace() -> tuple[str, int]:
    """A workspace on its own (invented) database: (its id, its database's id)."""
    account = f"copies-{uuid.uuid4().hex[:10]}"
    store.upsert_client(account, "portal")
    db_id = store.save_db_config("azure_sql", "Retail warehouse", {"server": "retail", "database": "d", "user": "u",
                                                                    "password": "p"})
    store.update_client_meta(account, client_name="Retail Ltd", db_config_id=db_id)
    return account, db_id


def _key(model, slug: str) -> str:
    return next(k for k, m in model.measures.items() if m.slug == slug)


class _Connection:
    """The workspace's database as Learn reaches it (core2/warehouse/querybot.py), here the invented retail data."""

    def __init__(self, warehouse):
        self.warehouse = warehouse

    def __call__(self, *_, **__):
        return self

    def __enter__(self):
        return self.warehouse

    def __exit__(self, *_):
        return False


def _make(account: str, copy_id: int, model, warehouse) -> dict:
    with patch("core2.bootstrap.service.answering_model", return_value=model), \
            patch("store.get_db_config", return_value={"db_type": "duckdb", "credentials": {}}), \
            patch("core2.warehouse.querybot.QueryBotWarehouse", _Connection(warehouse)):
        return copies.make(account, copy_id)


# ── Made, and answered from ──────────────────────────────────────────────────

def test_a_copy_is_made_on_the_workspaces_connection_and_then_answers():
    _, built, model, _ = _domain("retail")
    account, db = _workspace()
    copy_id = store.add_copy(account, db, metric=_key(model, "net_amount"), fields=["region.name"], grain="month",
                             every_hours=6, by="Ada Admin")
    made = _make(account, copy_id, model, DuckDBWarehouse(built.con))
    assert made["ok"], made
    row = store.get_copy(account, copy_id)
    assert row["status"] == "ready" and row["rows_kept"] > 0 and Path(row["file"]).is_file() and row["fingerprint"]
    ready = copies.ready(account, db, model)
    assert [r.id for r in ready] == [copy_id]
    assert ready[0].name.endswith(", by month") and " by " in ready[0].name
    # Made again: a new file in its place; the one before stays for an answer already reading it, until the next.
    first = row["file"]
    assert _make(account, copy_id, model, DuckDBWarehouse(built.con))["ok"]
    again = store.get_copy(account, copy_id)
    assert again["file"] != first and Path(again["file"]).is_file() and Path(first).exists()
    assert _make(account, copy_id, model, DuckDBWarehouse(built.con))["ok"]
    third = store.get_copy(account, copy_id)["file"]
    assert not Path(first).exists() and Path(again["file"]).exists() and Path(third).is_file()
    assert copies.remove(account, copy_id) and PC.files_of(account, copy_id) == []


def test_a_copy_taken_out_while_it_is_made_leaves_nothing_on_the_server():
    _, built, model, _ = _domain("retail")
    account, db = _workspace()
    copy_id = store.add_copy(account, db, metric=_key(model, "net_amount"), fields=["region.name"], grain="month")
    written = PC.write

    def write_then_taken_out(path, defn, rows):
        size = written(path, defn, rows)
        store.remove_copy(account, copy_id)            # the admin pressed Take out meanwhile
        return size

    with patch.object(PC, "write", side_effect=write_then_taken_out):
        made = _make(account, copy_id, model, DuckDBWarehouse(built.con))
    assert not made["ok"] and PC.files_of(account, copy_id) == []


def test_a_copy_of_a_database_the_workspace_left_is_not_kept_on_the_server():
    _, built, model, _ = _domain("retail")
    account, db = _workspace()
    copy_id = store.add_copy(account, db, metric=_key(model, "net_amount"), fields=["region.name"], grain="month")
    assert _make(account, copy_id, model, DuckDBWarehouse(built.con))["ok"]
    assert PC.files_of(account, copy_id)
    other = store.save_db_config("azure_sql", "Another warehouse", {"server": "other", "database": "d", "user": "u",
                                                                     "password": "p"})
    store.update_client_meta(account, db_config_id=other)
    with patch.object(copies, "start"):
        copies.run_due_copies()
    assert PC.files_of(account, copy_id) == []


def test_a_copy_too_large_or_through_a_repeating_link_is_not_made_and_says_why(monkeypatch):
    _, built, model, _ = _domain("retail")
    account, db = _workspace()
    monkeypatch.setenv("QUERYBOT_COPY_MAX_ROWS", "1000")
    big = store.add_copy(account, db, metric=_key(model, "net_amount"), fields=["product.name", "customer.name"])
    made = _make(account, big, model, DuckDBWarehouse(built.con))
    row = store.get_copy(account, big)
    assert not made["ok"] and row["status"] == "failed" and "more than 1,000 rows" in row["error"]
    assert copies.ready(account, db, model) == []

    class Twice:
        """A database whose link checks find a key held twice."""

        dialect, db_type = "duckdb", "duckdb"

        def query(self, sql, *, max_rows=None):
            from core2.warehouse.runner import QueryResult

            twice = "HAVING" in sql.upper()
            return QueryResult(["k"], [(1,)] if twice else [], False, 0.0)

    linked = store.add_copy(account, db, metric=_key(model, "net_amount"), fields=["region.name"])
    made = _make(account, linked, model, Twice())
    assert not made["ok"] and "twice" in store.get_copy(account, linked)["error"]


def test_a_failed_making_keeps_the_copy_made_before():
    _, built, model, _ = _domain("retail")
    account, db = _workspace()
    copy_id = store.add_copy(account, db, metric=_key(model, "net_amount"), fields=["region.name"], grain="month")
    assert _make(account, copy_id, model, DuckDBWarehouse(built.con))["ok"]

    class Down:
        dialect, db_type = "duckdb", "duckdb"

        def query(self, sql, *, max_rows=None):
            raise ConnectionError("the database is paused")

    assert not _make(account, copy_id, model, Down())["ok"]
    row = store.get_copy(account, copy_id)
    assert row["status"] == "ready" and "could not be read" in row["error"] and Path(row["file"]).is_file()
    assert [r.id for r in copies.ready(account, db, model)] == [copy_id]


# ── Made again on its schedule ───────────────────────────────────────────────

def test_the_scheduler_makes_each_copy_again_when_due():
    now = dt.datetime(2026, 6, 15, 12, 0)
    at = lambda hours: (now - dt.timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")   # noqa: E731
    assert copies.due({"status": "new", "every_hours": 6}, now)
    assert not copies.due({"status": "ready", "every_hours": 6, "made_at": at(2), "tried_at": at(2)}, now)
    assert copies.due({"status": "ready", "every_hours": 6, "made_at": at(7), "tried_at": at(7)}, now)
    assert not copies.due({"status": "failed", "every_hours": 6, "tried_at": at(0.2)}, now), "retried at once"
    assert copies.due({"status": "failed", "every_hours": 6, "tried_at": at(1.5)}, now)
    assert not copies.due({"status": "making", "every_hours": 6, "started_at": at(0.5)}, now)
    assert copies.due({"status": "making", "every_hours": 6, "started_at": at(3)}, now), "a making that stopped"


def test_the_minute_scheduler_starts_the_due_copies_of_each_workspaces_own_database():
    account, db = _workspace()
    current = store.add_copy(account, db, metric="m", fields=[])
    store.add_copy(account, db + 1000, metric="m", fields=[])            # of a database the workspace left
    started = []
    with patch.object(copies, "start", side_effect=lambda a, c: started.append((a, c))):
        copies.run_due_copies()
    assert (account, current) in started and all(a != account or c == current for a, c in started)

    from core import notification_scheduler

    with patch("core2.copies.run_due_copies") as run:
        notification_scheduler.run_due_notifications_once()
    assert run.called


def test_the_answers_check_never_reads_a_copy():
    """Asked afresh or its plan replayed, the check compares the client's figures with the data as it is now."""
    from core2 import answers_check

    asked = []

    def services(*_, **kwargs):
        asked.append(kwargs.get("use_precomputed", True))
        return None, {"type": "assistant_response"}

    question = {"id": 1, "question": "Net sales in 2025"}
    with patch("core2.service._portal_services", side_effect=services):
        answers_check._ask("acct", question, 1)
        answers_check._replay("acct", question, {"kind": "query", "intent": "value", "measures": ["net_amount"]})
    assert asked == [False, False]


# ── The admin's page ─────────────────────────────────────────────────────────

@pytest.fixture
def admin():
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from admin import core2_copies, routes

    app = FastAPI()
    app.include_router(routes.router)
    with patch.object(routes, "_is_auth", return_value=True), patch.object(core2_copies, "_is_auth", return_value=True):
        yield TestClient(app)


@pytest.fixture
def learned():
    _, _, model, _ = _domain("retail")
    with patch("core2.bootstrap.service.answering_model", return_value=model), \
            patch("store.get_db_config", return_value={"db_type": "duckdb", "credentials": {}}):
        yield model


def test_an_admin_keeps_a_copy_and_it_is_made_at_once(admin, learned):
    account, db = _workspace()
    with patch.object(copies, "start") as start:
        saved = admin.post(f"/admin/clients/{account}/copies", follow_redirects=False, data={
            "metric": _key(learned, "net_amount"), "fields": ["region.name", "store.name"], "grain": "month",
            "every": "3"})
    assert saved.status_code == 303 and "saved=added" in saved.headers["location"]
    [row] = store.list_copies(account, db)
    assert row["fields"] == ["region.name", "store.name"] and row["grain"] == "month" and row["every_hours"] == 3
    start.assert_called_once_with(account, row["id"])
    page = admin.get(f"/admin/clients/{account}/copies")
    assert page.status_code == 200 and "Precomputed copies" in page.text and "Waiting" in page.text
    name = copies.copy_name(learned, row)
    assert name.endswith(", by month") and " and " in name and name in page.text


def test_a_name_on_the_page_is_shown_as_text_never_run(admin, learned):
    """A metric or field name (set by a data steward, or read from the warehouse) never runs as script."""
    account, _ = _workspace()
    named = [{"key": "m", "name": "</script><script>alert(1)</script>", "fields": [], "dates": [],
              "default_date": ""}]
    with patch.object(copies, "choices", return_value=named):
        page = admin.get(f"/admin/clients/{account}/copies")
    assert page.status_code == 200 and "<script>alert(1)" not in page.text


def test_a_copy_that_cannot_be_kept_is_refused_with_why(admin, learned):
    account, db = _workspace()
    with patch.object(copies, "start") as start:
        refused = admin.post(f"/admin/clients/{account}/copies", follow_redirects=False, data={
            "metric": _key(learned, "net_amount"), "fields": ["no.such_field"], "grain": "day", "every": "6"})
    assert "error=" in refused.headers["location"] and store.list_copies(account, db) == [] and not start.called
    with patch.object(copies, "start"):
        odd = admin.post(f"/admin/clients/{account}/copies", follow_redirects=False, data={
            "metric": _key(learned, "net_amount"), "grain": "day", "every": "5"})
    assert "error=" in odd.headers["location"] and store.list_copies(account, db) == []


def test_an_admin_makes_it_now_changes_its_schedule_and_takes_it_out(admin, learned, tmp_path):
    account, db = _workspace()
    copy_id = store.add_copy(account, db, metric=_key(learned, "net_amount"), fields=[])
    kept = tmp_path / "copies" / "kept.duckdb"
    kept.parent.mkdir(parents=True, exist_ok=True)
    kept.write_bytes(b"x")
    store.copy_made(account, copy_id, rows=1, took_ms=1, size_bytes=1, file=str(kept), fingerprint="f",
                    model_version=1)
    with patch.object(copies, "start") as start:
        assert "saved=making" in admin.post(f"/admin/clients/{account}/copies/{copy_id}/make",
                                            follow_redirects=False).headers["location"]
    start.assert_called_once_with(account, copy_id)
    admin.post(f"/admin/clients/{account}/copies/{copy_id}/every", data={"every": "12"}, follow_redirects=False)
    assert store.get_copy(account, copy_id)["every_hours"] == 12
    gone = admin.post(f"/admin/clients/{account}/copies/{copy_id}/remove", follow_redirects=False)
    assert "saved=removed" in gone.headers["location"]
    assert store.get_copy(account, copy_id) is None and not kept.exists()


def test_a_data_steward_does_not_reach_the_copies():
    from admin.access import AdminIdentity, decide

    steward = AdminIdentity(admin_id=5, email="s@example.com", name="Sam", role=store.DATA_STEWARD,
                            workspaces=frozenset({"acct"}))
    keeper = AdminIdentity(admin_id=6, email="w@example.com", name="Wren", role=store.WORKSPACE_ADMIN,
                           workspaces=frozenset({"acct"}))
    assert not decide(steward, "/clients/acct/copies").allowed
    assert decide(keeper, "/clients/acct/copies").allowed
    assert not decide(keeper, "/clients/other/copies").allowed


# ── What nobody waits for reads the database ─────────────────────────────────

def test_scheduled_emails_alerts_and_background_jobs_read_the_database_never_a_copy():
    from core import dashboard_schedule, notification_scheduler, tile_alerts
    from core2 import service

    with patch.object(copies, "ready", return_value=["a copy"]):
        assert isinstance(service._with_copies("the database", "acct", 1, None), PC.Precomputed)
        with copies.live():
            assert service._with_copies("the database", "acct", 1, None) == "the database"
    seen = []
    with patch("core.alert_engine.run_due_alert_checks", side_effect=lambda: seen.append(("jobs", copies.live_only()))):
        notification_scheduler.run_due_notifications_once()
    with patch("portal.routes._refresh_chart", side_effect=lambda *a, **k: seen.append(("alert", copies.live_only()))):
        tile_alerts._draw({"db_config_id": None}, {"lang": "en"})

    async def deliver_all(*_, **__):
        seen.append(("email", copies.live_only()))
        return {"queued": 0, "skipped": 0, "failed": 0, "why": {}, "email_ids": []}

    with patch.object(dashboard_schedule, "_owner_problem", return_value=""), \
            patch.object(dashboard_schedule, "_deliver_all", side_effect=deliver_all), \
            patch.object(dashboard_schedule.schedule_store, "schedule_people", return_value=[]), \
            patch.object(dashboard_schedule.schedule_store, "finish_run"), \
            patch.object(dashboard_schedule.store, "get_user", return_value={"id": 1}), \
            patch.object(dashboard_schedule.store, "get_dashboard", return_value={"id": 1}):
        dashboard_schedule.deliver({"id": 1, "owner_id": 1, "dashboard_id": 1, "account_id": "acct"}, 1, "")
    assert seen == [("jobs", True), ("alert", True), ("email", True)] and not copies.live_only()


def test_a_dashboard_tile_read_from_a_copy_says_when_the_copy_was_made(monkeypatch):
    import core2.service as service
    from core.i18n import t
    from portal import routes

    made = {"name": "Net amount by Store, by month", "made_at": "2026-10-24 04:10", "every_hours": 6}

    def tile(trust):
        payload = {"data": {"rows": [{"store": "Eastview", "net_amount": 847.7}]}, "chart": None,
                   "answer": {"badges": []}, "reading": {}, "trust": trust}
        monkeypatch.setattr(service, "portal_replay", lambda *a, **k: payload)
        return routes._refresh_core2_tile({"id": 1, "question": "Net sales by store", "chart_type": "table"}, {},
                                          {"account_id": "acct", "id": 1, "lang": "en"},
                                          {"measures": ["net_amount"], "group_by": ["store"]})

    assert tile({"row_count": 1, "copy": made})["copy_made"] == "2026-10-24 04:10"
    assert "copy_made" not in tile({"row_count": 1})
    assert t("ui.dash.from_copy", lang="en", when="2026-10-24 04:10") == "From the copy made 2026-10-24 04:10 UTC"
    assert "04:10" in t("ui.dash.from_copy", lang="fr", when="2026-10-24 04:10")
