"""
A reader does in the portal what their portal role allows (core/portal_roles.py, portal/access.py).

    viewer   opens the dashboards shared with them, filters and follows them; no chat
    asker    also asks, keeps answers on dashboards of their own, suggests, exports
    creator  also has dashboards made from a request, changes them by talking, shares and publishes

Every surface asks the same table: the pages and their APIs (one middleware in front of /portal), the chat
socket, a dashboard request or a change by talking inside the chat (refused before any AI is asked), and a
question from a chat platform. A refusal says why in the reader's language. A role never widens which tables
or rows a reader sees, and every reader made before roles is a creator.

The real application (main.app) on a scratch store; no AI is called.
"""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

PASSWORD = "a-readers-password"


@pytest.fixture
def app(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    import main

    main.store.init_db()
    return main


@pytest.fixture
def workspace(app):
    account_id = f"acct{os.urandom(4).hex()}"
    app.store.upsert_client(account_id, "Retail Ltd")
    app.store.update_client_meta(account_id, chat_ui_enabled=1)
    return account_id


def _reader(app, account_id, role, lang="en"):
    email = f"{role}-{os.urandom(3).hex()}@example.com"
    user_id, _ = app.store.create_user(account_id, role.title(), email, password=PASSWORD, portal_role=role)
    if lang != "en":
        app.store.set_user_language(user_id, lang)
    return user_id, email


def _signed_in(app, account_id, email):
    client = TestClient(app.app)
    response = client.post("/portal/login", follow_redirects=False,
                           data={"account_id": account_id, "email": email, "password": PASSWORD})
    assert response.status_code == 303, response.text
    return client, response.headers["location"]


def _refused(response) -> bool:
    """Refused by the role gate: a 403 that names the action, or a page sent back to the reader's home."""
    if response.status_code == 403:
        return response.json().get("error") == "not_allowed"
    return response.status_code == 303 and response.headers.get("location") in ("/portal/dashboard", "/portal/chat")


# What each role may not do: (method, path, needs). "{d}" is a dashboard id.
SURFACES = [
    ("GET", "/portal/chat", "ask"),
    ("GET", "/portal/kb", "ask"),
    ("GET", "/portal/api/history", "ask"),
    ("GET", "/portal/api/suggest?q=sal", "ask"),
    ("GET", "/portal/api/query-limit-status", "ask"),
    ("POST", "/portal/api/answers/q1/feedback", "ask"),
    ("GET", "/portal/api/export-csv?question_id=q1", "export"),
    ("POST", "/portal/api/requests", "suggest"),
    ("GET", "/portal/pin-confirm", "save_answer"),
    ("POST", "/portal/api/pin-chart", "save_answer"),
    ("POST", "/portal/api/dashboards", "own_dashboards"),
    ("POST", "/portal/api/update-chart-layout", "own_dashboards"),
    ("POST", "/portal/api/dashboard/{d}/tidy", "own_dashboards"),
    ("GET", "/portal/api/people?q=a", "share"),
    ("GET", "/portal/api/dashboard/{d}/shares", "share"),
    ("POST", "/portal/api/dashboard/{d}/shares", "share"),
    ("POST", "/portal/dashboard/{d}/share", "share"),
    ("POST", "/portal/dashboard/{d}/publish", "share"),
    ("GET", "/portal/reports/new", "reports"),
]
NEEDS = {"ask": "asker", "export": "asker", "suggest": "asker", "save_answer": "asker", "own_dashboards": "asker",
         "share": "creator", "reports": "creator"}
RANK = {"viewer": 1, "asker": 2, "creator": 3}


def _call(client, method, path):
    if method == "GET":
        return client.get(path, follow_redirects=False)
    return client.post(path, json={}, follow_redirects=False)


class TestEachSurfaceAgainstEachRole:

    @pytest.mark.parametrize("role", ["viewer", "asker", "creator"])
    def test_a_reader_reaches_exactly_what_their_role_allows(self, app, workspace, role):
        _uid, email = _reader(app, workspace, role)
        client, _home = _signed_in(app, workspace, email)
        board = app.store.create_dashboard(workspace, _uid, "t", "Mine")
        for method, path, action in SURFACES:
            response = _call(client, method, path.replace("{d}", str(board["id"])))
            allowed = RANK[role] >= RANK[NEEDS[action]]
            assert _refused(response) != allowed, (role, method, path, response.status_code)

    @pytest.mark.parametrize("role", ["viewer", "asker", "creator"])
    def test_every_reader_keeps_their_dashboards_settings_and_follows(self, app, workspace, role):
        _uid, email = _reader(app, workspace, role)
        client, _home = _signed_in(app, workspace, email)
        for path in ("/portal/dashboard", "/portal/settings", "/portal/notifications", "/portal/change-password"):
            response = client.get(path, follow_redirects=False)
            assert response.status_code == 200, (role, path, response.status_code)

    def test_a_refusal_says_why_in_the_readers_language(self, app, workspace):
        _uid, email = _reader(app, workspace, "asker", lang="fr")
        client, _ = _signed_in(app, workspace, email)
        refused = client.get("/portal/api/people?q=a")
        assert refused.status_code == 403 and refused.json()["action"] == "share"
        assert "Créateurs" in refused.json()["detail"] or "Créateur" in refused.json()["detail"]
        _uid, email = _reader(app, workspace, "viewer")
        client, _ = _signed_in(app, workspace, email)
        assert "Asker or Creator role" in client.get("/portal/api/history").json()["detail"]


class TestAViewer:

    @pytest.fixture
    def viewer(self, app, workspace):
        owner_id, _ = _reader(app, workspace, "creator")
        viewer_id, email = _reader(app, workspace, "viewer")
        board = app.store.create_dashboard(workspace, owner_id, "t", "Sales overview")
        app.store.add_dashboard_share(board["id"], owner_id, workspace, "user", viewer_id)
        client, home = _signed_in(app, workspace, email)
        return client, home, board

    def test_lands_on_their_dashboards_without_a_chat(self, viewer):
        client, home, board = viewer
        assert home == "/portal/dashboard"
        assert client.get("/portal/home", follow_redirects=False).headers["location"] == "/portal/dashboard"
        page = client.get("/portal/dashboard").text
        assert 'href="/portal/chat"' not in page and 'href="/portal/kb"' not in page
        assert "Viewer" in page

    def test_opens_a_dashboard_shared_with_them_to_read(self, viewer):
        client, _home, board = viewer
        page = client.get(f"/portal/dashboard?dashboard_id={board['id']}").text
        assert "Sales overview" in page
        assert 'id="dashShareButton"' not in page and 'id="dashShareDialog"' not in page
        assert 'id="dashEditToggle"' not in page
        assert "/portal/chat?thread=" not in page                    # no "Chat with this dashboard"
        assert f"/portal/dashboard/{board['id']}/subscribe" in page  # follows it

    def test_follows_a_dashboard(self, app, viewer):
        client, _home, board = viewer
        response = client.post(f"/portal/dashboard/{board['id']}/subscribe", data={"cadence": "weekly",
                                                                                   "channel": "in_app"},
                               follow_redirects=False)
        assert not _refused(response)

    def test_the_chat_socket_closes(self, app, viewer, workspace):
        client, _home, _board = viewer
        with pytest.raises(WebSocketDisconnect) as closed:
            with client.websocket_connect(f"/ws/chat/{workspace}") as socket:
                socket.receive_json()
        assert closed.value.code == 4403

    def test_a_chat_platform_question_is_refused_before_any_engine(self, app, workspace, monkeypatch):
        from fastapi import BackgroundTasks

        from core import dispatcher
        from gateway import PlatformEvent

        viewer_id, _ = _reader(app, workspace, "viewer")
        app.store.link_zoom_user(viewer_id, "teams-user-1")      # the linked chat identity, any platform
        monkeypatch.setattr(dispatcher, "get_state", lambda account_id: {"state": "READY"})
        monkeypatch.setattr(dispatcher, "_enqueue_query", lambda *a, **k: pytest.fail("the question ran"))
        said = []

        class Adapter:
            async def send_message(self, event, text, **_):
                said.append(text)

        event = PlatformEvent(account_id=workspace, user_id="teams-user-1", channel_id="c",
                              text="total sales by month", platform="teams")
        asyncio.run(dispatcher.dispatch(workspace, event, Adapter(), BackgroundTasks()))
        assert said and "dashboards shared with you" in said[-1].lower()


class TestAnAsker:

    def test_asks_and_keeps_answers_but_does_not_share(self, app, workspace):
        _uid, email = _reader(app, workspace, "asker")
        client, home = _signed_in(app, workspace, email)
        assert home == "/portal/chat"
        assert client.get("/portal/chat").status_code == 200
        board = app.store.create_dashboard(workspace, _uid, "t", "My answers")
        page = client.get(f"/portal/dashboard?dashboard_id={board['id']}").text
        assert 'id="dashShareButton"' not in page and 'id="dashShareDialog"' not in page

    def test_a_dashboard_request_in_the_chat_is_refused_before_the_ai(self, app, workspace, monkeypatch):
        import gateway.core2_bridge as bridge

        user_id, _ = _reader(app, workspace, "asker")
        user = app.store.get_user(user_id)
        monkeypatch.setattr(bridge, "_over_limit", lambda *a, **k: "")

        async def never(*a, **k):
            raise AssertionError("the new core was asked")

        monkeypatch.setattr(bridge, "_answer", never)
        sent = []

        class Socket:
            async def send_json(self, payload):
                sent.append(payload)

        adapter = SimpleNamespace(thread_id="th", send_lock=asyncio.Lock(), session_id="s", selected_dashboard_id=0)
        asyncio.run(bridge.answer_instead("core2", adapter, Socket(), workspace, "Create an executive dashboard", user))
        assert "dashboard made from a request" in sent[-1]["answer"]["headline"].lower() or \
               "creator" in sent[-1]["answer"]["headline"].lower()

    def test_a_change_by_talking_is_refused(self, app, workspace, monkeypatch):
        import gateway.core2_bridge as bridge

        user_id, _ = _reader(app, workspace, "asker")
        user = app.store.get_user(user_id)
        board = app.store.create_dashboard(workspace, user_id, "th", "Mine")
        sent = []

        class Socket:
            async def send_json(self, payload):
                sent.append(payload)

        adapter = SimpleNamespace(thread_id="th", send_lock=asyncio.Lock(), session_id="s",
                                  selected_dashboard_id=board["id"])
        handled = asyncio.run(bridge._talked(adapter, Socket(), workspace, "rename this dashboard to Weekly", user,
                                             "q1", "key", 0.0))
        assert handled and sent[-1]["kind"] == "dashboard_talk"
        assert app.store.get_dashboard(board["id"], user_id, workspace)["name"] == "Mine"

    def test_the_new_core_refuses_a_dashboard_request_too(self, app, workspace, monkeypatch):
        from core2 import service

        user_id, _ = _reader(app, workspace, "asker")
        monkeypatch.setattr(service, "_portal_services", lambda *a, **k: pytest.fail("services were built"))
        frame = service.portal_answer(workspace, "build me a sales dashboard", app.store.get_user(user_id),
                                      session_key="k")
        assert frame["answer"]["headline"]


class TestACreator:

    def test_does_everything_a_reader_can(self, app, workspace):
        _uid, email = _reader(app, workspace, "creator")
        client, home = _signed_in(app, workspace, email)
        assert home == "/portal/chat"
        board = app.store.create_dashboard(workspace, _uid, "t", "Mine")
        page = client.get(f"/portal/dashboard?dashboard_id={board['id']}").text
        assert 'id="dashShareButton"' in page and "Creator" in page

    def test_a_reader_made_before_roles_is_a_creator(self, app, workspace):
        from core.portal_roles import can, portal_role

        user_id, _ = app.store.create_user(workspace, "Old", "old@example.com", password=PASSWORD)
        user = app.store.get_user(user_id)
        assert portal_role(user) == "creator" and can(user, "share") and can(user, "create_dashboard")
        assert portal_role({"id": 1}) == "creator"          # a row read without the column


class TestTheRoleTable:

    def test_every_action_has_a_least_role(self):
        from core.portal_roles import ACTIONS, PORTAL_ROLES, can

        for action, least in ACTIONS.items():
            for role in PORTAL_ROLES:
                assert can({"portal_role": role}, action) == (RANK[role] >= RANK[least]), (action, role)
        with pytest.raises(ValueError):
            can({"portal_role": "creator"}, "fly")

    def test_no_reader_is_no_one(self):
        from core.portal_roles import can

        assert not can(None, "view_dashboards")

    def test_an_unknown_role_is_refused_by_the_store(self, app, workspace):
        with pytest.raises(ValueError):
            app.store.create_user(workspace, "X", "x@example.com", password=PASSWORD, portal_role="owner")
