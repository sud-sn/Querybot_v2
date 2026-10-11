"""
The reader's own page (portal Settings): Profile, Security, Charts and Notifications.

    profile        name, job title, language, time zone and start page, each checked; signing in lands on the start
                   page while the reader may still open it, else where their workspace sends everyone
    photo          whatever image is uploaded is re-drawn as a 256-pixel square PNG, so nothing of the file is kept;
                   shown to people of the reader's own workspace only
    security       the password changed from the tab; "sign out of other devices" ends every session but this one;
                   each sign-in listed with its browser and address
    notifications  every follow, scheduled email and alert that tells the reader, each with on and off: one they set
                   up pauses for everyone it tells, one someone else set up stops for them alone
    schedules      a new one starts in the reader's own time zone

Invented people and data only (the retail domain).
"""

from __future__ import annotations

import io
import os
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from tests.test_a_dashboard_reads_as_the_plan import BY_REGION, TOTAL, _pin
from tests.test_a_new_core_answer_can_be_pinned import retail  # noqa: F401
from tests.test_dashboards_are_emailed_on_a_schedule import _owner, _per_reader, _person, _schedule

CHROME_WINDOWS = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/141.0.0.0 Safari/537.36")
SAFARI_IPHONE = ("Mozilla/5.0 (iPhone; CPU iPhone OS 18_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) "
                 "Version/18.5 Mobile/15E148 Safari/604.1")


@pytest.fixture
def app(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    import main

    main.store.init_db()
    return main


def _password(user: dict) -> str:
    return "owner-pass-123" if user["name"].startswith("Sam") else "reader-pass-123"


def _sign_in(app, user: dict, agent: str = CHROME_WINDOWS):
    client = TestClient(app.app, base_url="https://testserver", headers={"User-Agent": agent})
    response = client.post("/portal/login", data={"account_id": user["account_id"], "email": user["email"],
                                                  "password": _password(user)}, follow_redirects=False)
    assert response.status_code == 303, response.text
    return client, response.headers["location"]


def _signed_in(app, user: dict, agent: str = CHROME_WINDOWS) -> TestClient:
    return _sign_in(app, user, agent)[0]


def _profile_form(user: dict, **changes) -> dict:
    return {"name": user["name"], "job_title": "", "timezone": "", "start_page": "", "lang": "en", **changes}


def _image(size=(300, 200), kind="JPEG", **save) -> bytes:
    from PIL import Image

    picture = Image.new("RGB", size, (30, 120, 200))
    for x in range(size[0] // 2):
        picture.putpixel((x, size[1] // 2), (250, 250, 250))
    out = io.BytesIO()
    picture.save(out, format=kind, **save)
    return out.getvalue()


def _board(store, retail, owner):  # noqa: F811
    tiles = _pin(store, retail, owner, TOTAL, "net amount in april", new_dashboard_name="Sales pulse")
    _pin(store, retail, owner, BY_REGION, "net amount by region in april", dashboard_id=tiles[0]["dashboard_id"])
    return store.get_dashboard(tiles[0]["dashboard_id"], owner["id"], owner["account_id"]), tiles[0]


class TestProfile:

    def test_what_the_reader_says_about_themselves_is_kept(self, app):
        store = app.store
        sam = _owner(store)
        client = _signed_in(app, sam)
        page = client.get("/portal/settings")
        assert page.status_code == 200
        for tab in ("Profile", "Security", "Charts", "Notifications"):
            assert f'>{tab}</button>' in page.text
        reply = client.post("/portal/settings/profile", data=_profile_form(
            sam, name="  Sam   Owner-Lee ", job_title="Finance manager", timezone="Europe/Paris",
            start_page="dashboards", lang="fr"), follow_redirects=False)
        assert reply.headers["location"] == "/portal/settings?tab=profile&saved=profile"
        from store import profile_store

        assert store.get_user(sam["id"])["name"] == "Sam Owner-Lee"
        assert store.get_user(sam["id"])["lang"] == "fr" and client.cookies.get("qb_lang") in (None, "fr")
        assert profile_store.get_profile(sam["id"]) == {"job_title": "Finance manager", "timezone": "Europe/Paris",
                                                        "start_page": "dashboards", "photo_rev": 0}
        page = client.get("/portal/settings?tab=profile&saved=profile").text
        assert "Profil enregistré." in page and 'value="Finance manager"' in page
        assert '<option value="Europe/Paris" selected>Europe/Paris</option>' in page
        events = [e["action"] for e in store.list_user_events(sam["account_id"])]
        assert "renamed" in events

    @pytest.mark.parametrize("change, error", [
        ({"name": "   "}, "Your name cannot be empty."),
        ({"name": "x" * 121}, "Your name is too long"),
        ({"timezone": "Mars/Olympus_Mons"}, "Pick a time zone from the list."),
        ({"start_page": "dashboard:999999"}, "Pick a start page from the list."),
        ({"start_page": "https://elsewhere.example.com"}, "Pick a start page from the list."),
    ])
    def test_what_is_wrong_is_said_and_nothing_is_kept(self, app, change, error):
        store = app.store
        sam = _owner(store)
        client = _signed_in(app, sam)
        reply = client.post("/portal/settings/profile", data=_profile_form(sam, **change))
        assert reply.status_code == 200 and error in reply.text
        from store import profile_store

        assert store.get_user(sam["id"])["name"] == sam["name"]
        assert profile_store.get_profile(sam["id"])["timezone"] == ""

    def test_an_error_in_the_address_is_shown_only_when_it_is_ours(self, app):
        sam = _owner(app.store)
        page = _signed_in(app, sam).get("/portal/settings?tab=profile&error=<script>alert(1)</script>").text
        assert "<script>alert(1)</script>" not in page and 'role="alert"' not in page.split("qb-tabs")[0]


class TestStartPage:

    def test_signing_in_lands_on_the_page_the_reader_chose(self, app, retail):  # noqa: F811
        store = app.store
        sam = _owner(store)
        assert _sign_in(app, sam)[1] == "/portal/chat", "the workspace's own: the chat"
        board, _ = _board(store, retail, sam)
        client = _signed_in(app, sam)
        client.post("/portal/settings/profile", data=_profile_form(sam, start_page=f"dashboard:{board['id']}"))
        assert _sign_in(app, sam)[1] == f"/portal/dashboard?dashboard_id={board['id']}"
        client.post("/portal/settings/profile", data=_profile_form(sam, start_page="dashboards"))
        assert _sign_in(app, sam)[1] == "/portal/dashboard"
        assert client.get("/portal/home", follow_redirects=False).headers["location"] == "/portal/dashboard"

    def test_a_dashboard_no_longer_shared_sends_them_where_everyone_goes(self, app, retail):  # noqa: F811
        store = app.store
        sam = _owner(store)
        board, _ = _board(store, retail, sam)
        viv = _person(store, sam, "Viv Viewer")
        store.add_dashboard_share(board["id"], sam["id"], sam["account_id"], "user", viv["id"])
        client = _signed_in(app, viv)
        page = client.get("/portal/settings").text
        assert f'value="dashboard:{board["id"]}"' in page and 'value="chat"' not in page, "a Viewer does not ask"
        client.post("/portal/settings/profile", data=_profile_form(viv, start_page=f"dashboard:{board['id']}"))
        assert _sign_in(app, viv)[1] == f"/portal/dashboard?dashboard_id={board['id']}"
        store.remove_dashboard_share(board["id"], sam["id"], sam["account_id"], "user", viv["id"])
        assert _sign_in(app, viv)[1] == "/portal/dashboard"


class TestPhoto:

    def test_the_photo_is_drawn_again_as_a_square_and_nothing_of_the_file_is_kept(self, app):
        from PIL import Image

        store = app.store
        sam = _owner(store)
        client = _signed_in(app, sam)
        exif = Image.Exif()
        exif[0x010F] = "Invented Camera Co"          # Make
        exif[0x0132] = "2026:10:01 09:30:00"         # DateTime
        upload = _image(exif=exif.tobytes())
        reply = client.post("/portal/settings/photo", files={"photo": ("me.jpg", upload, "image/jpeg")},
                            headers={"Accept": "application/json"})
        assert reply.status_code == 200 and reply.json() == {"ok": True, "url": f"/portal/people/{sam['id']}/photo?v=1"}
        served = client.get(reply.json()["url"])
        assert served.status_code == 200 and served.headers["content-type"] == "image/png"
        picture = Image.open(io.BytesIO(served.content))
        assert picture.format == "PNG" and picture.size == (256, 256)
        assert "exif" not in picture.info and b"Invented Camera" not in served.content
        assert served.headers["cache-control"].startswith("private")
        page = client.get("/portal/settings").text
        assert f'<img src="/portal/people/{sam["id"]}/photo?v=1" alt="">' in page, "in the sidebar and on the tab"

    def test_only_the_readers_own_workspace_sees_it(self, app):
        store = app.store
        sam = _owner(store)
        _signed_in(app, sam).post("/portal/settings/photo", files={"photo": ("me.png", _image(kind="PNG"), "image/png")})
        url = f"/portal/people/{sam['id']}/photo?v=1"
        colleague = _person(store, sam, "Cal Colleague")
        assert _signed_in(app, colleague).get(url).status_code == 200
        stranger = _owner(store)
        assert _signed_in(app, stranger).get(url).status_code == 404, "another workspace: as if there were none"
        assert TestClient(app.app, base_url="https://testserver").get(url).status_code == 401

    @pytest.mark.parametrize("name, data, error", [
        ("notes.png", b"just some text, not a picture", "is not a photo we can use"),
        ("drawing.svg", b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>',
         "is not a photo we can use"),
        ("empty.png", b"", "Choose a photo first."),
    ])
    def test_what_is_not_a_photo_is_refused(self, app, name, data, error):
        sam = _owner(app.store)
        client = _signed_in(app, sam)
        reply = client.post("/portal/settings/photo", files={"photo": (name, data, "image/png")},
                            headers={"Accept": "application/json"})
        assert reply.status_code == 400 and error in reply.json()["error"]
        assert client.get(f"/portal/people/{sam['id']}/photo?v=1").status_code == 404

    def test_a_picture_too_large_to_draw_is_refused_before_it_is_drawn(self, app):
        from PIL import Image

        sam = _owner(app.store)
        out = io.BytesIO()
        Image.new("1", (8000, 6000)).save(out, format="PNG")
        reply = _signed_in(app, sam).post("/portal/settings/photo", files={"photo": ("big.png", out.getvalue(), "image/png")},
                                          headers={"Accept": "application/json"})
        assert reply.status_code == 400 and "too large" in reply.json()["error"]

    def test_removing_it_brings_back_the_initial(self, app):
        sam = _owner(app.store)
        client = _signed_in(app, sam)
        client.post("/portal/settings/photo", files={"photo": ("me.png", _image(kind="PNG"), "image/png")})
        reply = client.post("/portal/settings/photo/delete", follow_redirects=False)
        assert reply.headers["location"] == "/portal/settings?tab=profile&saved=photo_removed"
        assert client.get(f"/portal/people/{sam['id']}/photo?v=1").status_code == 404
        page = client.get("/portal/settings").text
        assert "/photo?v=" not in page and ">S</a>" in page

    def test_a_deleted_person_takes_their_photo_and_sign_ins_with_them(self, app):
        from store import profile_store

        store = app.store
        sam = _owner(store)
        cal = _person(store, sam, "Cal Colleague")
        client = _signed_in(app, cal)
        client.post("/portal/settings/photo", files={"photo": ("me.png", _image(kind="PNG"), "image/png")})
        client.post("/portal/settings/profile", data=_profile_form(cal, job_title="Buyer"))
        assert profile_store.photo_path(cal["id"]).is_file() and profile_store.recent_sign_ins(cal["id"])
        store.delete_user(cal["id"])
        assert not profile_store.photo_path(cal["id"]).exists()
        assert profile_store.recent_sign_ins(cal["id"]) == [] and profile_store.get_profile(cal["id"])["job_title"] == ""
        _signed_in(app, sam).post("/portal/settings/photo", files={"photo": ("me.png", _image(kind="PNG"), "image/png")})
        store.delete_client(sam["account_id"])
        assert not profile_store.photo_path(sam["id"]).exists()

    def test_without_javascript_the_form_still_sends_it(self, app):
        sam = _owner(app.store)
        reply = _signed_in(app, sam).post("/portal/settings/photo", files={"photo": ("me.gif", _image(kind="GIF"), "image/gif")},
                                          follow_redirects=False)
        assert reply.headers["location"] == "/portal/settings?tab=profile&saved=photo"


class TestSecurity:

    def test_sign_out_of_other_devices_keeps_this_one(self, app):
        store = app.store
        sam = _owner(store)
        laptop = _signed_in(app, sam, CHROME_WINDOWS)
        phone = _signed_in(app, sam, SAFARI_IPHONE)
        assert phone.get("/portal/settings", follow_redirects=False).status_code == 200
        reply = laptop.post("/portal/settings/sign-out-others", follow_redirects=False)
        assert reply.headers["location"] == "/portal/settings?tab=security&saved=signed_out"
        assert laptop.get("/portal/settings", follow_redirects=False).status_code == 200
        assert phone.get("/portal/settings", follow_redirects=False).headers["location"] == "/portal/login"
        assert "signed_out_elsewhere" in [e["action"] for e in store.list_user_events(sam["account_id"])]

    def test_each_sign_in_is_listed_with_its_browser_and_address(self, app):
        sam = _owner(app.store)
        _signed_in(app, sam, SAFARI_IPHONE)
        client = _signed_in(app, sam, CHROME_WINDOWS)
        page = client.get("/portal/settings?tab=security").text
        table = page.split('class="qb-table settings-sign-ins"')[1].split("</table>")[0]
        assert table.index("Chrome on Windows") < table.index("Safari on iPhone"), "newest first"
        assert "testclient" in table

    def test_the_password_is_changed_from_the_tab(self, app):
        store = app.store
        sam = _owner(store)
        client = _signed_in(app, sam)
        other = _signed_in(app, sam, SAFARI_IPHONE)
        wrong = client.post("/portal/change-password", data={"current_pw": "not-it", "new_pw": "brand-new-pass-1",
                                                             "confirm_pw": "brand-new-pass-1", "back": "settings"},
                            follow_redirects=False)
        assert wrong.headers["location"] == "/portal/settings?tab=security&error=ui.auth.error.current_incorrect"
        from core.i18n import t

        assert t("ui.auth.error.current_incorrect", lang="en") in client.get(wrong.headers["location"]).text
        done = client.post("/portal/change-password", data={"current_pw": "owner-pass-123", "new_pw": "brand-new-pass-1",
                                                            "confirm_pw": "brand-new-pass-1", "back": "settings"},
                           follow_redirects=False)
        assert done.headers["location"] == "/portal/settings?tab=security&saved=password"
        assert client.get("/portal/settings", follow_redirects=False).status_code == 200
        assert other.get("/portal/settings", follow_redirects=False).status_code == 303
        assert store.verify_password(store.get_user(sam["id"]), "brand-new-pass-1")

    def test_a_temporary_password_still_goes_to_its_own_page(self, app):
        store = app.store
        sam = _owner(store)
        store.change_password(sam["id"], "temporary-pass-1", is_temp=True)
        client = TestClient(app.app, base_url="https://testserver")
        reply = client.post("/portal/login", data={"account_id": sam["account_id"], "email": sam["email"],
                                                   "password": "temporary-pass-1"}, follow_redirects=False)
        assert reply.headers["location"] == "/portal/change-password"
        assert client.get("/portal/settings", follow_redirects=False).headers["location"] == "/portal/change-password"
        done = client.post("/portal/change-password", data={"new_pw": "my-own-pass-12", "confirm_pw": "my-own-pass-12",
                                                            "back": "settings"}, follow_redirects=False)
        assert done.headers["location"] != "/portal/settings?tab=security&saved=password", "set, then landed"


class TestNotifications:

    def _setup(self, app, retail):  # noqa: F811
        store = app.store
        sam = _owner(store)
        board, tile = _board(store, retail, sam)
        rae = _person(store, sam, "Rae Reader")
        schedule = _schedule(app, sam, board, people=[rae])
        with patch("core2.service.portal_replay", _per_reader(retail)):
            reply = _signed_in(app, sam).post(f"/portal/api/dashboard/{board['id']}/tiles/{tile['id']}/alerts", json={
                "condition": "above", "threshold": 150000, "every": 60, "format": "currency",
                "recipients": [{"type": "user", "id": rae["id"]}]})
        assert reply.status_code == 200, reply.text
        alert_id = reply.json()["alerts"][0]["id"]
        store.subscribe_dashboard(board["id"], rae["id"], sam["account_id"], cadence="weekly", channel="email")
        return store, sam, rae, board, schedule, alert_id

    @staticmethod
    def _rows(page: str) -> dict:
        import re

        rows = {}
        for kind, item, body in re.findall(r'<li class="settings-n-row" data-kind="([a-z-]+)" data-id="(\d+)">(.*?)</li>',
                                           page, re.S):
            rows[(kind, int(item))] = {"on": 'aria-checked="true"' in body,
                                       "markup": body, "text": " ".join(re.sub(r"<[^>]+>", " ", body).split())}
        return rows

    def test_everything_that_tells_the_reader_is_in_one_list(self, app, retail):  # noqa: F811
        store, sam, rae, board, schedule, alert_id = self._setup(app, retail)
        mine = self._rows(_signed_in(app, sam).get("/portal/settings?tab=notifications").text)
        assert set(mine) == {("schedule", schedule["id"]), ("alert", alert_id)}
        assert "Every Monday at 08:00" in mine[("schedule", schedule["id"])]["text"]
        assert "Set up by you" in mine[("alert", alert_id)]["text"]
        hers = self._rows(_signed_in(app, rae).get("/portal/settings?tab=notifications").text)
        assert set(hers) == {("follow", board["id"]), ("schedule-for-me", schedule["id"]), ("alert-for-me", alert_id)}
        assert all(row["on"] for row in hers.values())
        assert "Weekly update, in the app and by email" in hers[("follow", board["id"])]["text"]
        assert "From Sam Owner" in hers[("alert-for-me", alert_id)]["text"]
        assert "Sales pulse · Net amount" in hers[("alert-for-me", alert_id)]["text"]
        assert "Tells when the number is above $150,000.00" in hers[("alert-for-me", alert_id)]["text"]

    def test_one_made_by_someone_else_stops_for_them_alone(self, app, retail):  # noqa: F811
        from store import email_store, schedule_store, tile_alert_store

        store, sam, rae, board, schedule, alert_id = self._setup(app, retail)
        client = _signed_in(app, rae)
        for kind, item in (("schedule-for-me", schedule["id"]), ("alert-for-me", alert_id)):
            reply = client.post(f"/portal/settings/notifications/{kind}/{item}", data={"on": "0"}, follow_redirects=False)
            assert reply.headers["location"] == "/portal/settings?tab=notifications&saved=notifications"
        assert email_store.opted_out(rae["email"], "schedule", str(schedule["id"]))
        assert email_store.opted_out(rae["email"], "alert", str(alert_id))
        assert schedule_store.get_schedule(schedule["id"])["is_active"] == 1, "still going to everyone else"
        assert tile_alert_store.get_alert(alert_id)["is_active"] == 1
        rows = self._rows(client.get("/portal/settings?tab=notifications").text)
        assert not rows[("schedule-for-me", schedule["id"])]["on"] and not rows[("alert-for-me", alert_id)]["on"]
        client.post(f"/portal/settings/notifications/schedule-for-me/{schedule['id']}", data={"on": "1"})
        assert not email_store.opted_out(rae["email"], "schedule", str(schedule["id"]))

    def test_a_follow_pauses_and_starts_again(self, app, retail):  # noqa: F811
        store, sam, rae, board, schedule, alert_id = self._setup(app, retail)
        client = _signed_in(app, rae)
        client.post(f"/portal/settings/notifications/follow/{board['id']}", data={"on": "0"})
        assert store.get_dashboard_subscription(board["id"], rae["id"], sam["account_id"])["status"] == "paused"
        assert all(f["user_id"] != rae["id"] for f in store.list_active_follows())
        client.post(f"/portal/settings/notifications/follow/{board['id']}", data={"on": "1"})
        follow = store.get_dashboard_subscription(board["id"], rae["id"], sam["account_id"])
        assert (follow["status"], follow["cadence"], follow["channel"]) == ("active", "weekly", "email")

    def test_ones_own_pause_for_everyone_and_the_others_see_why(self, app, retail):  # noqa: F811
        from store import schedule_store, tile_alert_store

        store, sam, rae, board, schedule, alert_id = self._setup(app, retail)
        client = _signed_in(app, sam)
        client.post(f"/portal/settings/notifications/schedule/{schedule['id']}", data={"on": "0"})
        client.post(f"/portal/settings/notifications/alert/{alert_id}", data={"on": "0"})
        assert schedule_store.get_schedule(schedule["id"])["is_active"] == 0
        assert tile_alert_store.get_alert(alert_id)["is_active"] == 0
        hers = self._rows(_signed_in(app, rae).get("/portal/settings?tab=notifications").text)
        assert "Paused by the person who set it up." in hers[("schedule-for-me", schedule["id"])]["text"]
        client.post(f"/portal/settings/notifications/schedule/{schedule['id']}", data={"on": "1"})
        client.post(f"/portal/settings/notifications/alert/{alert_id}", data={"on": "1"})
        resumed = schedule_store.get_schedule(schedule["id"])
        assert resumed["is_active"] == 1 and resumed["next_run_at"]
        assert tile_alert_store.get_alert(alert_id)["is_active"] == 1

    def test_nobody_switches_what_is_not_theirs(self, app, retail):  # noqa: F811
        from store import schedule_store, tile_alert_store

        store, sam, rae, board, schedule, alert_id = self._setup(app, retail)
        client = _signed_in(app, rae)
        for kind, item in (("schedule", schedule["id"]), ("alert", alert_id), ("follow", 999999),
                           ("everything", schedule["id"])):
            reply = client.post(f"/portal/settings/notifications/{kind}/{item}", data={"on": "0"}, follow_redirects=False)
            assert reply.headers["location"] == ("/portal/settings?tab=notifications"
                                                 "&error=ui.profile.n.error.not_found")
        assert schedule_store.get_schedule(schedule["id"])["is_active"] == 1
        assert tile_alert_store.get_alert(alert_id)["is_active"] == 1
        stranger = _owner(store)
        reply = _signed_in(app, stranger).post(f"/portal/settings/notifications/schedule-for-me/{schedule['id']}",
                                               data={"on": "0"}, follow_redirects=False)
        assert "error=ui.profile.n.error.not_found" in reply.headers["location"]

    def test_a_reader_whose_role_no_longer_sends_schedules_cannot_turn_one_back_on(self, app, retail):  # noqa: F811
        from store import schedule_store

        store, sam, rae, board, schedule, alert_id = self._setup(app, retail)
        client = _signed_in(app, sam)
        client.post(f"/portal/settings/notifications/schedule/{schedule['id']}", data={"on": "0"})
        store.update_user(sam["id"], portal_role="asker")
        reply = client.post(f"/portal/settings/notifications/schedule/{schedule['id']}", data={"on": "1"},
                            follow_redirects=False)
        assert reply.headers["location"].endswith("error=ui.profile.n.error.role")
        assert schedule_store.get_schedule(schedule["id"])["is_active"] == 0
        page = client.get("/portal/settings?tab=notifications").text
        assert "disabled title=\"Your role no longer lets you send scheduled emails" in page


class TestEveryReader:

    def test_a_viewer_has_the_whole_page(self, app):
        store = app.store
        sam = _owner(store)
        viv = _person(store, sam, "Viv Viewer")
        page = _signed_in(app, viv).get("/portal/settings", follow_redirects=False)
        assert page.status_code == 200 and 'id="panel-notifications"' in page.text

    def test_a_french_reader_reads_it_in_french(self, app):
        store = app.store
        sam = _owner(store)
        flo = _person(store, sam, "Flo Lecteur", lang="fr")
        page = _signed_in(app, flo).get("/portal/settings").text
        assert ">Sécurité</button>" in page and "Connexions récentes" in page and "Fuseau horaire" in page
        assert "ui.profile." not in page.split("i18n_catalogue")[0].split("<script")[0]

    def test_a_new_schedule_starts_in_the_readers_own_time_zone(self, app, retail):  # noqa: F811
        store = app.store
        sam = _owner(store)
        board, _ = _board(store, retail, sam)
        client = _signed_in(app, sam)
        client.post("/portal/settings/profile", data=_profile_form(sam, timezone="Asia/Kolkata"))
        with patch("core2.service.portal_replay", _per_reader(retail)):
            page = client.get(f"/portal/dashboard?dashboard_id={board['id']}").text
        assert 'data-zone="Asia/Kolkata"' in page and "const here = dialog.dataset.zone ||" in page


def test_devices_are_read_from_their_browsers():
    from core.profile import device

    assert device(CHROME_WINDOWS) == ("Chrome", "Windows")
    assert device(SAFARI_IPHONE) == ("Safari", "iPhone")
    assert device("Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6) Gecko/20100101 Firefox/131.0") == ("Firefox", "macOS")
    assert device("Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 Chrome/141.0 Safari/537.36 Edg/141.0") == \
        ("Edge", "Windows")
    assert device("curl/8.5.0") == ("", "")


def test_time_zones_are_the_real_names_only():
    from core.profile import valid_zone, zones

    assert zones()[0] == "UTC" and "Europe/London" in zones() and "America/Argentina/Buenos_Aires" in zones()
    assert not valid_zone("US/Eastern") and not valid_zone("Etc/GMT+5") and not valid_zone("../../etc/passwd")
    assert os.sep not in "".join(z for z in zones() if z.startswith(".."))
