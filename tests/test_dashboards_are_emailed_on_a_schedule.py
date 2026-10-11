"""
A dashboard emailed on a schedule arrives as the dashboard, drawn for each person as they may see it.

    when        every day, every weekday, weekly on a day, monthly on a date or the last day, at a wall-clock time
                in a time zone (summer time included); a time a clock change skips goes an hour later
    saving      its owner, a Creator, from the dashboard; people and groups of the workspace; saving shares the
                dashboard with them; never an email address shown; Askers and other people's dashboards refused
    the email   numbers and tables as text, charts as pictures carried in the email, the PDF attached (one page a
                tab); each person's own draw, in their language; a note from the owner; a link that stops it
    once        run once per time it was due, even with two schedulers or a restart halfway; a server that was
                down sends the missed run once, late
    skipped     someone who can no longer open the dashboard or stopped this email, named in the run
    paused      the owner can no longer send it, or the dashboard is gone: paused, saying why
    test        "Send me a test" goes to the owner alone, marked a test, at once
    no browser  without Chromium the email still goes with every number and table, and says why
    transport   pictures in the body over SMTP (multipart/related, Content-ID) and Graph (isInline); a message
                over 3 MB on Graph goes as a draft with an upload session, the token never sent to the upload

Invented data only (the retail domain); no AI.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import email as email_lib
import io
import json
import os
from email import policy
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from starlette.testclient import TestClient

from tests.test_a_dashboard_reads_as_the_plan import BY_REGION, TOP, TWO, _pin
from tests.test_a_new_core_answer_can_be_pinned import _replay, retail  # noqa: F401
from tests.test_email_goes_out_and_is_logged import _Inbox, _smtp_server

CHROMIUM = "/opt/pw-browsers/chromium"
UTC = dt.timezone.utc


def _browser_ready() -> bool:
    try:
        import playwright  # noqa: F401
    except ImportError:
        return False
    return Path(os.getenv("QUERYBOT_CHROMIUM_PATH") or CHROMIUM).exists()


needs_browser = pytest.mark.skipif(not _browser_ready(), reason="Playwright and Chromium are not on this machine")


@pytest.fixture
def app(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    monkeypatch.setenv("PORTAL_BASE_URL", "https://bi.example.com")
    if Path(CHROMIUM).exists() and not os.getenv("QUERYBOT_CHROMIUM_PATH"):
        monkeypatch.setenv("QUERYBOT_CHROMIUM_PATH", CHROMIUM)
    import main
    from core.mail import transport

    main.store.init_db()
    transport.forget_tokens()
    return main


@pytest.fixture
def inbox():
    return _Inbox()


@pytest.fixture
def smtp(app, inbox):
    server = _smtp_server(inbox)
    from core.mail import settings

    settings.save({"provider": "smtp", "from_address": "reports@example.com", "from_name": "Example BI",
                   "smtp_host": "127.0.0.1", "smtp_port": str(server.server_address[1]), "smtp_security": "none"})
    yield server
    server.shutdown()
    server.server_close()


def _owner(store) -> dict:
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Retail Ltd")
    db_id = store.save_db_config("azure_sql", "invented", {"server": "invented", "user": "u", "password": "p"})
    store.update_client_meta(account_id, db_config_id=db_id, chat_ui_enabled=1)
    store.save_compliance_profile(account_id, mode="standard")
    user_id, _ = store.create_user(account_id, "Sam Owner", f"sam-{os.urandom(2).hex()}@example.com",
                                   password="owner-pass-123", portal_role="creator")
    return store.get_user(user_id)


def _person(store, owner: dict, name: str, *, group_id=None, lang="en", portal_role="viewer") -> dict:
    slug = name.lower().replace(" ", ".")
    user_id, _ = store.create_user(owner["account_id"], name, f"{slug}-{os.urandom(2).hex()}@example.com",
                                   group_id=group_id, password="reader-pass-123", portal_role=portal_role)
    if lang != "en":
        store.set_user_language(user_id, lang)
    return store.get_user(user_id)


def _board(store, retail, owner: dict) -> dict:  # noqa: F811
    tiles = _pin(store, retail, owner, TWO, "net and gross in april", new_dashboard_name="Sales pulse")
    board_id = tiles[0]["dashboard_id"]
    for plan, question in ((BY_REGION, "net by region in april"), (TOP, "top products in april")):
        _pin(store, retail, owner, plan, question, dashboard_id=board_id)
    return store.get_dashboard(board_id, owner["id"], owner["account_id"])


def _per_reader(retail, west_only: set[int] = frozenset()):  # noqa: F811
    """The tiles drawn as each reader may see them: those in ``west_only`` see the West region alone (as a row
    rule would have it)."""
    plain = _replay(retail)

    def replay(account_id, plan_data, portal_user, *, question=""):
        if portal_user and int(portal_user.get("id") or 0) in west_only:
            plan_data = {**plan_data, "filters": list(plan_data.get("filters") or []) +
                         [{"field": "region.name", "op": "eq", "values": ["West"]}]}
        return plain(account_id, plan_data, portal_user, question=question)
    return replay


def _schedule(app, owner, board, *, people=(), groups=(), **fields) -> dict:
    from core import dashboard_schedule
    from store import schedule_store

    payload = {"cadence": "weekly", "weekday": 0, "send_time": "08:00", "timezone": "Europe/London",
               "recipients": [{"type": "user", "id": p["id"]} for p in people] +
                             [{"type": "group", "id": g} for g in groups], **fields}
    read = dashboard_schedule.read_form(payload, board, owner)
    assert isinstance(read, dict), read
    return schedule_store.get_schedule(dashboard_schedule.save(owner, board, read))


def _due_now(schedule: dict, minutes_ago: int = 1) -> str:
    from store.db import get_db

    stamp = (dt.datetime.now(UTC).replace(tzinfo=None) - dt.timedelta(minutes=minutes_ago)).strftime("%Y-%m-%d %H:%M:%S")
    with get_db() as conn:
        conn.execute("UPDATE dashboard_schedule SET next_run_at=? WHERE id=?", (stamp, int(schedule["id"])))
    return stamp


def _sent(kind="schedule") -> list[dict]:
    from store import email_store

    return email_store.list_emails(kind=kind, limit=200)


def _for(person: dict) -> list[dict]:
    return [m for m in _sent() if int(m.get("user_id") or 0) == int(person["id"])]


def _run(retail, west_only=frozenset(), *, now=None):  # noqa: F811
    from core.dashboard_schedule import run_due_schedules

    with patch("core2.service.portal_replay", _per_reader(retail, set(west_only))):
        return run_due_schedules(now)


# ── When ─────────────────────────────────────────────────────────────────────

class TestWhen:

    def _next(self, after: str, **schedule) -> str:
        from core.dashboard_schedule import next_run

        found = next_run({"send_time": "08:00", "timezone": "Europe/London", **schedule},
                         dt.datetime.fromisoformat(after))
        return found.strftime("%Y-%m-%d %H:%M") if found else ""

    def test_weekly_keeps_wall_clock_time_in_summer_and_winter(self):
        assert self._next("2026-07-05 12:00", cadence="weekly", weekday=0) == "2026-07-06 07:00"    # BST
        assert self._next("2026-11-01 12:00", cadence="weekly", weekday=0) == "2026-11-02 08:00"    # GMT

    def test_the_time_itself_is_not_sent_again(self):
        assert self._next("2026-07-06 07:00", cadence="weekly", weekday=0) == "2026-07-13 07:00"

    def test_weekdays_skip_the_weekend(self):
        assert self._next("2026-07-10 08:00", cadence="weekdays") == "2026-07-13 07:00"             # Fri -> Mon

    def test_monthly_on_a_date_and_on_the_last_day(self):
        assert self._next("2026-01-20 00:00", cadence="monthly", month_day=15) == "2026-02-15 08:00"
        assert self._next("2026-02-01 00:00", cadence="monthly", month_day=0) == "2026-02-28 08:00"
        assert self._next("2028-02-01 00:00", cadence="monthly", month_day=0) == "2028-02-29 08:00"

    def test_a_time_a_clock_change_skips_goes_an_hour_later(self):
        # 02:30 does not happen in New York on 8 March 2026: sent at 03:30 EDT, 07:30 UTC.
        assert self._next("2026-03-07 12:00", cadence="daily", send_time="02:30",
                          timezone="America/New_York") == "2026-03-08 07:30"

    def test_it_is_said_in_the_readers_language(self):
        from core.dashboard_schedule import describe

        schedule = {"cadence": "weekly", "weekday": 0, "send_time": "08:00", "timezone": "Europe/Paris"}
        assert describe(schedule, "en") == "every Monday at 08:00 (Europe/Paris)"
        assert describe(schedule, "fr") == "chaque lundi à 08:00 (Europe/Paris)"
        assert describe({**schedule, "cadence": "monthly", "month_day": 0}, "en").startswith("on the last day")


# ── Saving ───────────────────────────────────────────────────────────────────

def _signed_in(app, user: dict) -> TestClient:
    client = TestClient(app.app, base_url="https://testserver")
    response = client.post("/portal/login", data={"account_id": user["account_id"], "email": user["email"],
                                                  "password": "owner-pass-123" if user["name"] == "Sam Owner"
                                                  else "reader-pass-123"}, follow_redirects=False)
    assert response.status_code == 303, response.text[:300]
    return client


class TestSaving:

    def test_the_owner_saves_one_and_it_is_shared_with_whoever_it_goes_to(self, app, retail):  # noqa: F811
        store = app.store
        owner = _owner(store)
        board = _board(store, retail, owner)
        ann = _person(store, owner, "Ann Viewer")
        team = store.create_group(owner["account_id"], "Buyers")
        bob = _person(store, owner, "Bob Buyer", group_id=team)
        client = _signed_in(app, owner)
        reply = client.post(f"/portal/api/dashboard/{board['id']}/schedules", json={
            "cadence": "weekly", "weekday": 0, "send_time": "08:00", "timezone": "Europe/London",
            "recipients": [{"type": "user", "id": ann["id"]}, {"type": "group", "id": team}],
            "view": {"period": "", "members": {}}, "subject": "Monday numbers", "message": "For the review."})
        assert reply.status_code == 200, reply.text
        data = reply.json()
        (saved,) = data["schedules"]
        assert saved["when"] == "every Monday at 08:00 (Europe/London)" and saved["active"] and saved["next"]
        assert {(r["type"], r["name"]) for r in saved["recipients"]} == {("user", "Ann Viewer"), ("group", "Buyers")}
        assert "@" not in json.dumps(saved["recipients"]), "a person is a name and a group, never an email"
        for person in (ann, bob):
            assert store.get_dashboard_for_view(board["id"], person["id"], owner["account_id"]), person["name"]
        assert any("shared" in n["title"].lower() for n in store.list_notices(owner["account_id"], ann["id"]))

    def test_what_is_wrong_is_said(self, app, retail):  # noqa: F811
        store = app.store
        owner = _owner(store)
        board = _board(store, retail, owner)
        client = _signed_in(app, owner)
        url = f"/portal/api/dashboard/{board['id']}/schedules"
        base = {"cadence": "daily", "send_time": "08:00", "timezone": "Europe/London", "include_owner": True}
        assert client.post(url, json={**base, "send_time": "25:00"}).json()["error"] == "Choose a time."
        assert client.post(url, json={**base, "timezone": "Mars/Olympus"}).json()["error"] == "Choose a time zone."
        assert client.post(url, json={**base, "include_owner": False}).json()["error"] == "Choose who it goes to."

    def test_a_subject_is_one_line(self, app, retail):  # noqa: F811
        store = app.store
        owner = _owner(store)
        board = _board(store, retail, owner)
        schedule = _schedule(app, owner, board, subject="Monday\r\nBcc: everyone@example.com")
        assert schedule["subject"] == "Monday Bcc: everyone@example.com", "a header line break never gets through"

    def test_someone_of_another_workspace_is_never_a_recipient(self, app, retail):  # noqa: F811
        store = app.store
        owner = _owner(store)
        board = _board(store, retail, owner)
        stranger = _person(store, _owner(store), "Stranger")
        schedule = _schedule(app, owner, board, people=[stranger])
        from store import schedule_store

        assert schedule_store.schedule_recipients(schedule["id"]) == []
        assert not store.get_dashboard_for_view(board["id"], stranger["id"], stranger["account_id"])

    def test_an_asker_cannot_and_nobody_schedules_a_dashboard_not_theirs(self, app, retail):  # noqa: F811
        store = app.store
        owner = _owner(store)
        board = _board(store, retail, owner)
        asker = _person(store, owner, "Ada Asker", portal_role="asker")
        creator = _person(store, owner, "Cy Creator", portal_role="creator")
        store.add_dashboard_share(board["id"], owner["id"], owner["account_id"], "user", asker["id"])
        store.add_dashboard_share(board["id"], owner["id"], owner["account_id"], "user", creator["id"])
        url = f"/portal/api/dashboard/{board['id']}/schedules"
        body = {"cadence": "daily", "send_time": "08:00", "timezone": "UTC"}
        assert _signed_in(app, asker).post(url, json=body).status_code == 403
        assert _signed_in(app, creator).post(url, json=body).status_code == 404
        page = _signed_in(app, creator).get(f"/portal/dashboard?dashboard_id={board['id']}").text
        assert 'id="dashScheduleDialog"' not in page
        assert 'id="dashScheduleDialog"' in _signed_in(app, owner).get(
            f"/portal/dashboard?dashboard_id={board['id']}").text


# ── Sending ──────────────────────────────────────────────────────────────────

class TestSending:

    @needs_browser
    def test_each_person_gets_the_dashboard_itself_as_they_may_see_it(self, app, retail, smtp, inbox):  # noqa: F811
        from pypdf import PdfReader

        store = app.store
        owner = _owner(store)
        board = _board(store, retail, owner)
        ann = _person(store, owner, "Ann Viewer")
        wes = _person(store, owner, "Wes West")
        schedule = _schedule(app, owner, board, people=[ann, wes], message="For Monday's review.")
        _due_now(schedule)
        assert _run(retail, west_only={wes["id"]}) == 1
        mine, wests = _for(ann), _for(wes)
        assert len(mine) == 1 and len(wests) == 1 and len(_for(owner)) == 1, "the owner asked for a copy"
        ann_mail, wes_mail = mine[0], wests[0]
        assert "Sales pulse" in ann_mail["subject"]
        html = ann_mail["body_html"]
        assert "Net amount" in html and "Gross amount" in html and "Onions Large" in html, "numbers and rows as text"
        assert "For Monday&#x27;s review." in html and "A note from Sam Owner" in html
        assert ann_mail["body_html"] != wes_mail["body_html"]
        assert "Central" not in wes_mail["body_text"] or "West" in wes_mail["body_text"]
        from store import email_store

        files = {a[0]: a for a in email_store.email_attachments(ann_mail)}
        pictures = [a for a in files.values() if len(a) > 3]
        assert pictures and all(f"cid:{a[3]}" in html for a in pictures), "every chart shown in the body"
        assert all(a[2][:8] == b"\x89PNG\r\n\x1a\n" for a in pictures)
        (pdf,) = [a for a in files.values() if a[1] == "application/pdf"]
        assert len(PdfReader(io.BytesIO(pdf[2])).pages) == 1, "a page for its one tab"
        assert "A PDF of the dashboard is attached." in html
        assert ann_mail["unsubscribe_url"].startswith("https://bi.example.com/portal/unsubscribe/")
        from store import schedule_store

        (run,) = schedule_store.list_runs(schedule["id"])
        assert run["status"] == "sent" and run["queued"] == 3 and run["skipped"] == 0
        from core.mail import outbox

        outbox.deliver_due()
        (received,) = inbox.to(ann["email"])
        related = next(p for p in received.walk() if p.get_content_type() == "multipart/related")
        assert sum(1 for p in related.iter_parts() if p.get_content_type() == "image/png") == len(pictures)
        assert [a.get_filename() for a in received.iter_attachments()] == [pdf[0]]

    def test_with_email_not_set_up_the_run_says_it_reached_nobody(self, app, retail):  # noqa: F811
        from store import schedule_store

        store = app.store
        owner = _owner(store)
        board = _board(store, retail, owner)
        schedule = _schedule(app, owner, board, attach_pdf=False)
        _due_now(schedule)
        with patch("core.dashboard_snapshot.Drawer.__aenter__", _no_browser):
            _run(retail)
        (run,) = schedule_store.list_runs(schedule["id"])
        assert run["status"] == "failed" and "because email is not set up" in run["detail"]
        assert _for(owner)[0]["status"] == "skipped", "kept, and sent once email is set up and it is sent again"

    def test_a_run_goes_once_and_the_next_one_is_next_week(self, app, retail):  # noqa: F811
        from core.dashboard_schedule import run_due_schedules
        from store import schedule_store

        store = app.store
        owner = _owner(store)
        board = _board(store, retail, owner)
        ann = _person(store, owner, "Ann Viewer")
        schedule = _schedule(app, owner, board, people=[ann], attach_pdf=False)
        due = _due_now(schedule)
        with patch("core.dashboard_snapshot.Drawer.__aenter__", _no_browser):
            assert _run(retail) == 1
            assert _run(retail) == 0
        assert len(_for(ann)) == 1
        after = schedule_store.get_schedule(schedule["id"])
        assert after["next_run_at"] > due and dt.datetime.fromisoformat(after["next_run_at"]).weekday() == 0
        assert schedule_store.claim_run(schedule["id"], due) is None, "a second scheduler finds it claimed"
        assert run_due_schedules is not None

    def test_a_run_cut_short_by_a_restart_reaches_only_those_it_had_not(self, app, retail, smtp):  # noqa: F811
        from store import schedule_store
        from store.db import get_db

        store = app.store
        owner = _owner(store)
        board = _board(store, retail, owner)
        ann, bea = _person(store, owner, "Ann Viewer"), _person(store, owner, "Bea Viewer")
        schedule = _schedule(app, owner, board, people=[ann, bea], include_owner=False, attach_pdf=False)
        due = "2026-10-05 07:00:00"
        run_id = schedule_store.claim_run(schedule["id"], due)
        from core.mail import outbox

        outbox.queue(kind="schedule", to=ann["email"], subject="already", text="x", user_id=ann["id"],
                     dedupe_key=f"schedule:{schedule['id']}:{due}:{ann['id']}")
        with get_db() as conn:
            conn.execute("UPDATE dashboard_schedule_run SET started_at=datetime('now','-2 hours') WHERE id=?", (run_id,))
            conn.execute("UPDATE dashboard_schedule SET next_run_at=datetime('now','+6 days') WHERE id=?",
                         (schedule["id"],))
        with patch("core.dashboard_snapshot.Drawer.__aenter__", _no_browser):
            _run(retail)
        assert len(_for(ann)) == 1 and len(_for(bea)) == 1
        run = schedule_store.get_run(run_id)
        assert run["status"] == "partial" and run["queued"] == 1 and "sent already" in run["detail"]

    def test_a_run_missed_while_the_server_was_down_goes_once_late(self, app, retail):  # noqa: F811
        from store import schedule_store

        store = app.store
        owner = _owner(store)
        board = _board(store, retail, owner)
        schedule = _schedule(app, owner, board, attach_pdf=False)
        _due_now(schedule, minutes_ago=60 * 24 * 21)
        with patch("core.dashboard_snapshot.Drawer.__aenter__", _no_browser):
            assert _run(retail) == 1
            assert _run(retail) == 0
        assert len(_for(owner)) == 1
        upcoming = schedule_store.get_schedule(schedule["id"])["next_run_at"]
        assert upcoming > dt.datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    def test_who_lost_access_or_stopped_it_is_skipped_and_counted(self, app, retail, smtp):  # noqa: F811
        from core.mail import links
        from store import email_store, schedule_store

        store = app.store
        owner = _owner(store)
        board = _board(store, retail, owner)
        ann, bea, cal = (_person(store, owner, n) for n in ("Ann Viewer", "Bea Viewer", "Cal Viewer"))
        schedule = _schedule(app, owner, board, people=[ann, bea, cal], include_owner=False, attach_pdf=False)
        store.remove_dashboard_share(board["id"], owner["id"], owner["account_id"], "user", bea["id"])
        found = links.read(links.unsubscribe_token(cal["email"], "schedule", str(schedule["id"])))
        email_store.opt_out(found["a"], found["k"], found["r"])
        store.update_user(_person(store, owner, "Dee Gone")["id"], is_active=0)
        _due_now(schedule)
        with patch("core.dashboard_snapshot.Drawer.__aenter__", _no_browser):
            _run(retail)
        assert len(_for(ann)) == 1 and not _for(bea)
        assert _for(cal) and _for(cal)[0]["status"] == "skipped", "kept in the delivery log, not sent"
        (run,) = schedule_store.list_runs(schedule["id"])
        assert run["status"] == "partial" and run["queued"] == 1 and run["skipped"] == 2
        assert "who can no longer open the dashboard" in run["detail"] and "who stopped this email" in run["detail"]

    def test_the_stop_link_names_the_dashboard_and_stops_it(self, app, retail, smtp):  # noqa: F811
        store = app.store
        owner = _owner(store)
        board = _board(store, retail, owner)
        ann = _person(store, owner, "Ann Viewer")
        schedule = _schedule(app, owner, board, people=[ann], include_owner=False, attach_pdf=False)
        _due_now(schedule)
        with patch("core.dashboard_snapshot.Drawer.__aenter__", _no_browser):
            _run(retail)
        link = _for(ann)[0]["unsubscribe_url"].replace("https://bi.example.com", "")
        client = TestClient(app.app, base_url="https://testserver")
        assert "“Sales pulse” by email" in client.get(link).text
        client.post(link)
        _due_now(schedule, minutes_ago=2)
        with patch("core.dashboard_snapshot.Drawer.__aenter__", _no_browser):
            _run(retail)
        assert [m["status"] for m in _for(ann)] == ["skipped", "queued"], "newest first: the second one stopped"

    def test_an_owner_who_can_no_longer_send_it_pauses_it_saying_why(self, app, retail):  # noqa: F811
        from store import schedule_store
        from store.db import get_db

        store = app.store
        owner = _owner(store)
        board = _board(store, retail, owner)
        schedule = _schedule(app, owner, board)
        store.update_user(owner["id"], portal_role="asker")
        _due_now(schedule)
        _run(retail)
        paused = schedule_store.get_schedule(schedule["id"])
        assert not paused["is_active"] and "no longer a Creator" in paused["paused_reason"] and not _sent()
        store.update_user(owner["id"], portal_role="creator")
        other = _schedule(app, owner, board)
        with get_db() as conn:
            conn.execute("DELETE FROM dashboard_artifact WHERE id=?", (board["id"],))
        _due_now(other)
        _run(retail)
        assert schedule_store.get_schedule(other["id"])["paused_reason"] == "The dashboard was deleted."

    def test_a_test_goes_to_the_owner_alone_at_once(self, app, retail, smtp, inbox):  # noqa: F811
        from store import schedule_store

        store = app.store
        owner = _owner(store)
        board = _board(store, retail, owner)
        ann = _person(store, owner, "Ann Viewer")
        schedule = _schedule(app, owner, board, people=[ann], attach_pdf=False)
        client = _signed_in(app, owner)
        with patch("core.dashboard_snapshot.Drawer.__aenter__", _no_browser), \
                patch("core2.service.portal_replay", _per_reader(retail)):
            reply = client.post(f"/portal/api/dashboard/{board['id']}/schedules/{schedule['id']}/test").json()
        assert reply["sent"]["email"]["status"] == "sent" and reply["sent"]["email"]["to"] == owner["email"]
        (message,) = inbox.messages
        assert str(message["To"]) == owner["email"] and str(message["Subject"]).startswith("Test: ")
        assert "only you received it" in message.get_body(("html",)).get_content()
        assert "List-Unsubscribe" not in message and not _for(ann)
        assert schedule_store.get_schedule(schedule["id"])["last_status"] == "", "a test is not the schedule's run"

    def test_without_chromium_the_numbers_and_tables_still_go(self, app, retail, monkeypatch):  # noqa: F811
        store = app.store
        owner = _owner(store)
        board = _board(store, retail, owner)
        schedule = _schedule(app, owner, board)
        monkeypatch.setenv("QUERYBOT_CHROMIUM_PATH", "/nowhere/chromium")
        _due_now(schedule)
        _run(retail)
        (mail,) = _for(owner)
        assert "$185.9K" in mail["body_html"] and "Onions Large" in mail["body_html"]
        assert "Chromium is not installed" in mail["body_html"] and "This chart is on the dashboard." in mail["body_html"]
        assert mail["attachments"] == [], "no pictures and no PDF without a browser"

    def test_a_french_reader_gets_it_in_french(self, app, retail):  # noqa: F811
        store = app.store
        owner = _owner(store)
        board = _board(store, retail, owner)
        flo = _person(store, owner, "Flo Lecteur", lang="fr")
        schedule = _schedule(app, owner, board, people=[flo], include_owner=False)
        _due_now(schedule)
        with patch("core.dashboard_snapshot.Drawer.__aenter__", _no_browser):
            _run(retail)
        (mail,) = _for(flo)
        assert "Établi le" in mail["body_html"] and "Ouvrir le tableau de bord" in mail["body_html"]
        assert "chaque lundi à 08:00 (Europe/London)" in mail["body_html"]
        assert "Ne plus recevoir ces courriels" in mail["body_html"]


async def _no_browser(self):
    self.problem = "snapshot.no_chromium"
    return self


# ── Pictures in the body, over SMTP and Graph ────────────────────────────────

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64


def _message(**kw):
    from core.mail.transport import Outgoing

    return Outgoing(to="ann@example.com", subject="Sales pulse", text="text", html='<img src="cid:tile-1@querybot">',
                    attachments=[("chart-1.png", "image/png", PNG, "tile-1@querybot"),
                                 ("Sales pulse.pdf", "application/pdf", b"%PDF-1.7 x")], **kw)


class TestTransport:

    def test_smtp_carries_the_pictures_in_the_body_and_the_pdf_attached(self, app, smtp, inbox):
        from core.mail import settings, transport

        transport.send(settings.load(), _message())
        (message,) = inbox.messages
        related = next(part for part in message.walk() if part.get_content_type() == "multipart/related")
        picture = next(part for part in related.iter_parts() if part.get_content_type() == "image/png")
        assert picture["Content-ID"] == "<tile-1@querybot>" and picture.get_content_disposition() == "inline"
        assert [a.get_filename() for a in message.iter_attachments()] == ["Sales pulse.pdf"]

    def _graph(self, app):
        from core.mail import settings

        settings.save({"provider": "graph", "from_address": "reports@example.com", "from_name": "Example BI",
                       "graph_tenant_id": "tenant-1", "graph_client_id": "client-1", "graph_client_secret": "s3cret"})
        return settings.load()

    def test_graph_marks_the_pictures_inline(self, app):
        from core.mail import transport

        calls = []

        def microsoft(request):
            calls.append(request)
            if "login.microsoftonline.com" in str(request.url):
                return httpx.Response(200, json={"access_token": "t", "expires_in": 3600})
            return httpx.Response(202, headers={"request-id": "r1"})

        transport.send(self._graph(app), _message(), http=httpx.Client(transport=httpx.MockTransport(microsoft)))
        body = json.loads(calls[-1].content)
        picture, pdf = body["message"]["attachments"]
        assert picture["isInline"] is True and picture["contentId"] == "tile-1@querybot"
        assert "isInline" not in pdf

    def test_graph_over_3_mb_goes_as_a_draft_with_an_upload_session(self, app):
        from core.mail import transport

        calls = []
        big = os.urandom(transport.GRAPH_INLINE_BYTES + 1000)

        def microsoft(request):
            calls.append(request)
            url = str(request.url)
            if "login.microsoftonline.com" in url:
                return httpx.Response(200, json={"access_token": "t", "expires_in": 3600})
            if url.endswith("/messages") and request.method == "POST":
                return httpx.Response(201, json={"id": "draft-1"})
            if url.endswith("/createUploadSession"):
                return httpx.Response(201, json={"uploadUrl": "https://upload.example.com/session-1"})
            if url.startswith("https://upload.example.com/"):
                return httpx.Response(200)
            return httpx.Response(202, headers={"request-id": "r2"})

        message = _message()
        message.attachments[1] = ("Sales pulse.pdf", "application/pdf", big)
        transport.send(self._graph(app), message, http=httpx.Client(transport=httpx.MockTransport(microsoft)))
        steps = [(c.method, str(c.url).split("/v1.0/users/reports@example.com")[-1]) for c in calls
                 if "login" not in str(c.url)]
        assert steps[0] == ("POST", "/messages")
        assert ("POST", "/messages/draft-1/attachments") in steps, "the small picture added to the draft"
        assert ("POST", "/messages/draft-1/attachments/createUploadSession") in steps
        uploads = [c for c in calls if str(c.url).startswith("https://upload.example.com/")]
        assert len(uploads) == -(-len(big) // transport.GRAPH_CHUNK)
        assert all("authorization" not in {k.lower() for k in c.headers} for c in uploads), "no token to the upload"
        assert uploads[-1].headers["Content-Range"].endswith(f"/{len(big)}")
        assert steps[-1] == ("POST", "/messages/draft-1/send")

    def test_graph_without_mail_readwrite_says_so(self, app):
        from core.mail import transport

        def microsoft(request):
            if "login.microsoftonline.com" in str(request.url):
                return httpx.Response(200, json={"access_token": "t", "expires_in": 3600})
            return httpx.Response(403, json={"error": {"code": "ErrorAccessDenied", "message": "Access is denied."}})

        message = _message()
        message.attachments[1] = ("Sales pulse.pdf", "application/pdf", b"x" * (transport.GRAPH_INLINE_BYTES + 1))
        with pytest.raises(transport.Permanent, match="Mail.ReadWrite"):
            transport.send(self._graph(app), message, http=httpx.Client(transport=httpx.MockTransport(microsoft)))


# ── The page drawn for an email ──────────────────────────────────────────────

def test_the_page_drawn_for_an_email_has_no_controls_and_no_live_socket(app, retail):  # noqa: F811
    from core.dashboard_snapshot import _tab_page

    store = app.store
    owner = _owner(store)
    board = _board(store, retail, owner)
    with patch("core2.service.portal_replay", _per_reader(retail)):
        html, context = asyncio.run(_tab_page(board["id"], owner, "Overview", {}, "en", "Drawn now for Sam Owner"))
    assert '<body class="is-snapshot">' in html and "Drawn now for Sam Owner" in html
    assert "/portal/ws/notifications" not in html
    assert [c["title"] for c in context["charts"]][:2] == ["Net amount", "Gross amount"]


def test_the_scheduler_sends_schedules_before_the_outbox_goes():
    source = Path(__file__).resolve().parents[1].joinpath("core/notification_scheduler.py").read_text()
    assert source.index("run_due_schedules()") < source.index("deliver_due()")
