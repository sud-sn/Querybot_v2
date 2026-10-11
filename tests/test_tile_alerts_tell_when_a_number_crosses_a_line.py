"""
An alert on a dashboard's number tile tells whoever it names, once, when the number crosses the line, and once when
it comes back -- each person as their own access draws the tile.

    the line        above or below a value; a rise, a fall or a move of more than N% on the period before
    who             anyone who can open the dashboard sets one for themselves; its owner, who may share it, for people
                    and groups too (saving shares the dashboard with them)
    once            fired when crossed, not again while it stays crossed, cleared when it comes back; two schedulers
                    never check one alert at once
    each their own  a person whose row rules show them less is told about their own number, never the whole
    how             a notice in the app; an email too when asked for; a link in it, or "Stop telling me", stops it
    the tile        its bell is on for whoever an alert tells, red while it is crossed for them; the dialog lists what
                    it told them lately
    paused          the tile is gone from the dashboard: paused, saying why

Invented data only (the retail domain); no AI.
"""

from __future__ import annotations

import datetime as dt
import os
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from tests.test_a_dashboard_reads_as_the_plan import BY_REGION, TOTAL, _pin
from tests.test_a_new_core_answer_can_be_pinned import retail  # noqa: F401
from tests.test_dashboards_are_emailed_on_a_schedule import _owner, _per_reader, _person, smtp, inbox  # noqa: F401


@pytest.fixture
def app(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    monkeypatch.setenv("PORTAL_BASE_URL", "https://bi.example.com")
    import main

    main.store.init_db()
    return main


def _board(store, retail, owner):  # noqa: F811
    """April's net amount ($185.9K, down 4.6% on March) and net amount by region."""
    tiles = _pin(store, retail, owner, TOTAL, "net amount in april", new_dashboard_name="Sales pulse")
    _pin(store, retail, owner, BY_REGION, "net amount by region in april", dashboard_id=tiles[0]["dashboard_id"])
    board = store.get_dashboard(tiles[0]["dashboard_id"], owner["id"], owner["account_id"])
    return board, tiles[0]


def _signed_in(app, user: dict) -> TestClient:
    client = TestClient(app.app, base_url="https://testserver")
    password = "owner-pass-123" if user["name"] == "Sam Owner" else "reader-pass-123"
    assert client.post("/portal/login", data={"account_id": user["account_id"], "email": user["email"],
                                              "password": password}, follow_redirects=False).status_code == 303
    return client


def _alert(app, retail, user, board, tile, **body):  # noqa: F811
    client = _signed_in(app, user)
    reply = client.post(f"/portal/api/dashboard/{board['id']}/tiles/{tile['id']}/alerts",
                        json={"condition": "above", "threshold": 150000, "every": 60, "format": "currency", **body})
    assert reply.status_code == 200, reply.text
    return reply.json(), client


def _check(retail, west_only=frozenset()):  # noqa: F811
    from core.tile_alerts import run_due_tile_alerts
    from store.db import get_db

    with get_db() as conn:
        conn.execute("UPDATE tile_alert SET next_check_at=datetime('now','-1 minute') WHERE is_active=1")
    with patch("core2.service.portal_replay", _per_reader(retail, set(west_only))):
        return run_due_tile_alerts()


def _notices(store, user):
    return [n for n in store.list_notices(user["account_id"], user["id"]) if n["kind"] == "tile_alert"]


def _threshold(value):
    from store.db import get_db

    with get_db() as conn:
        conn.execute("UPDATE tile_alert SET threshold=?", (float(value),))


# ── Reading a tile ───────────────────────────────────────────────────────────

class TestTheLine:

    def _read(self, value=185900.0, pct="-4.6%", direction="down"):
        from core.tile_alerts import reading

        return reading({"kpi": {"value": value, "format": "currency",
                                "change": {"direction": direction, "pct": pct, "vs": "March 2026"}},
                        "kpi_display": "$185.9K"})

    def test_above_and_below_a_value(self):
        from core.tile_alerts import crossed

        assert crossed("above", 150000, self._read()) is True
        assert crossed("above", 200000, self._read()) is False
        assert crossed("below", 200000, self._read()) is True

    def test_a_rise_a_fall_and_a_move_on_the_period_before(self):
        from core.tile_alerts import crossed

        assert crossed("fall", 4, self._read()) is True and crossed("fall", 5, self._read()) is False
        assert crossed("rise", 4, self._read()) is False
        assert crossed("move", 4, self._read()) is True
        assert crossed("move", 1, self._read(pct="", direction="flat")) is False

    def test_a_tile_that_cannot_say_is_never_crossed(self):
        from core.tile_alerts import crossed, reading

        assert crossed("fall", 1, self._read(pct=None, direction="")) is None
        assert crossed("above", 1, reading({"error": "Not available to you."})) is None


# ── Setting one ──────────────────────────────────────────────────────────────

class TestSettingOne:

    def test_a_viewer_sets_one_for_themselves_only(self, app, retail):  # noqa: F811
        store = app.store
        owner = _owner(store)
        board, tile = _board(store, retail, owner)
        viv = _person(store, owner, "Viv Viewer")
        ann = _person(store, owner, "Ann Viewer")
        store.add_dashboard_share(board["id"], owner["id"], owner["account_id"], "user", viv["id"])
        data, _ = _alert(app, retail, viv, board, tile, recipients=[{"type": "user", "id": ann["id"]}])
        (mine,) = data["alerts"]
        assert mine["mine"] and mine["recipients"] == [] and not data["may_tell_others"]
        assert mine["when"] == "is above $150,000.00"
        assert not store.get_dashboard_for_view(board["id"], ann["id"], owner["account_id"]), "nobody else told"

    def test_the_owner_tells_a_group_and_the_dashboard_is_shared_with_it(self, app, retail):  # noqa: F811
        store = app.store
        owner = _owner(store)
        board, tile = _board(store, retail, owner)
        team = store.create_group(owner["account_id"], "Buyers")
        bob = _person(store, owner, "Bob Buyer", group_id=team)
        data, _ = _alert(app, retail, owner, board, tile, recipients=[{"type": "group", "id": team}])
        assert data["may_tell_others"] and data["alerts"][0]["recipients"] == [{"type": "group", "id": team,
                                                                                "name": "Buyers"}]
        assert store.get_dashboard_for_view(board["id"], bob["id"], owner["account_id"])

    def test_what_is_wrong_is_said(self, app, retail):  # noqa: F811
        store = app.store
        owner = _owner(store)
        board, tile = _board(store, retail, owner)
        client = _signed_in(app, owner)
        url = f"/portal/api/dashboard/{board['id']}/tiles/{tile['id']}/alerts"
        assert client.post(url, json={"condition": "above", "threshold": "lots"}).json()["error"] == "Give a number."
        assert client.post(url, json={"condition": "fall", "threshold": -5}).json()["error"] == \
            "Give a percentage above zero."

    def test_nobody_sets_one_on_a_dashboard_they_cannot_open(self, app, retail):  # noqa: F811
        store = app.store
        owner = _owner(store)
        board, tile = _board(store, retail, owner)
        stranger = _person(store, owner, "Sly Stranger")
        reply = _signed_in(app, stranger).post(f"/portal/api/dashboard/{board['id']}/tiles/{tile['id']}/alerts",
                                               json={"condition": "above", "threshold": 1})
        assert reply.status_code == 404


# ── Telling ──────────────────────────────────────────────────────────────────

class TestTelling:

    def test_it_fires_once_and_comes_back_once(self, app, retail, smtp):  # noqa: F811
        store = app.store
        owner = _owner(store)
        board, tile = _board(store, retail, owner)
        _alert(app, retail, owner, board, tile, by_email=True)
        _check(retail)
        (fired,) = _notices(store, owner)
        assert fired["title"] == "“Net amount” is above $150,000.00: $185.9K"
        assert "Sales pulse" in fired["body"] and fired["link"] == f"/portal/dashboard?dashboard_id={board['id']}"
        _check(retail)
        assert len(_notices(store, owner)) == 1, "not again while it stays above"
        _threshold(200000)
        _check(retail)
        back = _notices(store, owner)[0]
        assert back["title"] == "“Net amount” is back: $185.9K"
        from store import email_store

        mails = email_store.list_emails(kind="alert")
        assert [m["subject"] for m in mails] == [back["title"], fired["title"]]
        assert all(m["unsubscribe_url"].startswith("https://bi.example.com/portal/unsubscribe/") for m in mails)

    def test_each_person_is_told_about_their_own_number(self, app, retail):  # noqa: F811
        store = app.store
        owner = _owner(store)
        board, tile = _board(store, retail, owner)
        wes = _person(store, owner, "Wes West")
        _alert(app, retail, owner, board, tile, recipients=[{"type": "user", "id": wes["id"]}])
        _check(retail, west_only={wes["id"]})
        assert len(_notices(store, owner)) == 1
        assert _notices(store, wes) == [], "West alone is under the line: Wes is not told the whole"
        _threshold(20000)
        _check(retail, west_only={wes["id"]})
        (told,) = _notices(store, wes)
        assert "$185.9K" not in told["title"] and "$28.1K" in told["title"], "his own number, West's"

    def test_a_change_condition_reads_the_tiles_own_comparison(self, app, retail):  # noqa: F811
        store = app.store
        owner = _owner(store)
        board, tile = _board(store, retail, owner)
        _alert(app, retail, owner, board, tile, condition="fall", threshold=4)
        _check(retail)
        (told,) = _notices(store, owner)
        assert told["title"] == "“Net amount” fell -4.6% vs March 2026"
        assert "falls more than 4%" in told["body"]

    def test_two_schedulers_never_check_one_alert_twice(self, app, retail):  # noqa: F811
        from store import tile_alert_store

        store = app.store
        owner = _owner(store)
        board, tile = _board(store, retail, owner)
        _alert(app, retail, owner, board, tile)
        (alert,) = tile_alert_store.due_alerts("9999-01-01 00:00:00")
        assert tile_alert_store.claim_check(alert["id"], alert["next_check_at"], "2099-01-01 00:00:00")
        assert not tile_alert_store.claim_check(alert["id"], alert["next_check_at"], "2099-01-01 00:00:00")

    def test_stop_telling_me_stops_it_for_that_person_alone(self, app, retail):  # noqa: F811
        store = app.store
        owner = _owner(store)
        board, tile = _board(store, retail, owner)
        ann = _person(store, owner, "Ann Viewer")
        data, _ = _alert(app, retail, owner, board, tile, recipients=[{"type": "user", "id": ann["id"]}])
        alert_id = data["alerts"][0]["id"]
        reply = _signed_in(app, ann).post(f"/portal/api/dashboard/{board['id']}/tiles/{tile['id']}/alerts/"
                                          f"{alert_id}/stop")
        assert reply.status_code == 200 and reply.json()["alerts"][0]["stopped"]
        _check(retail)
        assert _notices(store, ann) == [] and len(_notices(store, owner)) == 1

    def test_the_tiles_bell_shows_it_and_turns_red_while_crossed(self, app, retail):  # noqa: F811
        store = app.store
        owner = _owner(store)
        board, tile = _board(store, retail, owner)
        _, client = _alert(app, retail, owner, board, tile)

        def page():
            with patch("core2.service.portal_replay", _per_reader(retail)):
                return client.get(f"/portal/dashboard?dashboard_id={board['id']}").text
        drawn = page()
        assert f'data-alert-tile="{tile["id"]}"' in drawn and "dash-alert-button is-on" in drawn
        assert "dash-alert-button is-on is-firing" not in drawn
        _check(retail)
        assert "dash-alert-button is-on is-firing" in page()
        events = client.get(f"/portal/api/dashboard/{board['id']}/tiles/{tile['id']}/alerts").json()["events"]
        assert [e["kind"] for e in events] == ["fired"] and events[0]["display"] == "$185.9K"

    def test_a_tile_taken_off_the_dashboard_pauses_its_alerts(self, app, retail):  # noqa: F811
        from store import tile_alert_store

        store = app.store
        owner = _owner(store)
        board, tile = _board(store, retail, owner)
        data, _ = _alert(app, retail, owner, board, tile)
        store.remove_chart_from_dashboard(board["id"], tile["id"], owner["id"], owner["account_id"])
        _check(retail)
        alert = tile_alert_store.get_alert(data["alerts"][0]["id"])
        assert not alert["is_active"] and "no longer on a dashboard" in alert["paused_reason"]
        assert _notices(store, owner) == []

    def test_a_french_reader_is_told_in_french(self, app, retail):  # noqa: F811
        store = app.store
        owner = _owner(store)
        store.set_user_language(owner["id"], "fr")
        owner = store.get_user(owner["id"])
        board, tile = _board(store, retail, owner)
        _alert(app, retail, owner, board, tile)
        _check(retail)
        (told,) = _notices(store, owner)
        title = " ".join(told["title"].replace("\u202f", " ").replace("\xa0", " ").split())
        assert title == "« Net amount » dépasse 150 000,00 $ : 185,9k $", "numbers written the French way"


def test_the_scheduler_checks_alerts_before_the_outbox_goes():
    from pathlib import Path

    source = Path(__file__).resolve().parents[1].joinpath("core/notification_scheduler.py").read_text()
    assert source.index("run_due_tile_alerts()") < source.index("deliver_due()")
