"""
Admins sign in as themselves, each with a role, and the server refuses what a role does not reach.

The admin console had one shared password: whoever knew it could do everything, and the log named nobody.
Now (admin/access.py, admin/admin_accounts.py, store/admin_store.py):

  * an upgraded server is copied before it changes, and its shared password still opens the console once, as
    the emergency sign-in, which asks for a named Super admin account first;
  * a named admin signs in with email and password; wrong passwords wait, as the shared one did;
  * an invitation is a one-time link to choose a password;
  * a Super admin opens everything; a Workspace admin the people pages of their workspaces and all a Data
    steward does there; a Data steward the metrics, dates, relationships, knowledge base and requests of theirs.
    Every route the console has is checked against that, so a new route is closed until a rule opens it;
  * the last active Super admin cannot be demoted, deactivated or removed; a role change ends the admin's
    sessions; emergency sign-in can be turned off by a named Super admin and back on from the server;
  * every sign-in, refusal and change is logged with the admin's email.

The real application (main.app) on a scratch store; no AI is called.
"""

from __future__ import annotations

import io
import os
import re
import sqlite3
from pathlib import Path

from unittest.mock import patch

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

EMERGENCY = "the-emergency-password"
GOOD = "a-long-admin-password"


@pytest.fixture
def app(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    import main
    from admin import credentials

    main.store.init_db()
    credentials.set_password(EMERGENCY)
    return main


def _client(app) -> TestClient:
    return TestClient(app.app)


def _workspace(store, name="Retail") -> str:
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, name)
    return account_id


def _admin(store, role, workspaces=(), email=None, password=GOOD, name="Grace"):
    email = email or f"{os.urandom(4).hex()}@example.com"
    admin_id = store.create_admin(email, name, role, password=password, workspaces=list(workspaces))
    return admin_id, email


def _sign_in(client, email, password=GOOD):
    return client.post("/admin/login", data={"email": email, "password": password}, follow_redirects=False)


def _emergency(client, password=EMERGENCY):
    return client.post("/admin/login", data={"password": password}, follow_redirects=False)


# ── The upgrade ──────────────────────────────────────────────────────────────

class TestAnUpgradedServer:

    def test_is_copied_once_before_roles_change_it(self, tmp_path, monkeypatch):
        path = tmp_path / "querybot.db"
        monkeypatch.setenv("QUERYBOT_DB_PATH", str(path))
        monkeypatch.setenv("DB_PATH", str(path))
        import store

        store.init_db()
        account_id = _workspace(store)
        store.create_user(account_id, "Ada", "ada@example.com", password="readers-password", role="admin")
        store.create_user(account_id, "Bob", "bob@example.com", password="readers-password")
        # As a server before roles: no admin tables, no portal_role column.
        with sqlite3.connect(str(path)) as conn:
            for table in ("admin_event", "admin_workspace", "admin_account"):
                conn.execute(f"DROP TABLE {table}")
            conn.execute("ALTER TABLE portal_user DROP COLUMN portal_role")
        store.init_db()
        copies = sorted(tmp_path.glob("querybot.db.before-roles-*.bak"))
        assert len(copies) == 1
        with sqlite3.connect(str(copies[0])) as kept:
            tables = {r[0] for r in kept.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            assert "portal_user" in tables and "admin_account" not in tables
            assert kept.execute("SELECT COUNT(*) FROM portal_user").fetchone()[0] == 2
        # Every reader made before roles keeps all they could do: a creator. Who sees every table is unchanged.
        users = {u["email"]: u for u in store.list_users(account_id)}
        assert {u["portal_role"] for u in users.values()} == {"creator"}
        assert users["ada@example.com"]["role"] == "admin" and users["bob@example.com"]["role"] == "analyst"
        store.init_db()
        assert len(list(tmp_path.glob("querybot.db.before-roles-*.bak"))) == 1

    def test_a_new_server_has_nothing_to_copy(self, tmp_path, monkeypatch):
        path = tmp_path / "fresh.db"
        monkeypatch.setenv("QUERYBOT_DB_PATH", str(path))
        monkeypatch.setenv("DB_PATH", str(path))
        import store

        store.init_db()
        assert not list(tmp_path.glob("*.bak"))

    def test_the_shared_password_opens_once_to_make_a_super_admin(self, app):
        client = _client(app)
        response = _emergency(client)
        assert response.status_code == 303 and response.headers["location"] == "/admin/me/setup"
        page = client.get("/admin/me/setup")
        assert page.status_code == 200 and "Make your Super admin account" in page.text
        # The console opens meanwhile, under a banner that says how to stop needing the shared password.
        home = client.get("/admin")
        assert home.status_code == 200
        assert "emergency admin password" in home.text and 'href="/admin/me/setup"' in home.text
        short = client.post("/admin/me/setup", data={"name": "Ada", "email": "ada@example.com",
                                                     "password": "short", "confirm_password": "short"})
        assert "at least 12 characters" in short.text
        made = client.post("/admin/me/setup", follow_redirects=False, data={
            "name": "Ada Lovelace", "email": "Ada@Example.com", "password": GOOD, "confirm_password": GOOD})
        assert made.status_code == 303
        admin = app.store.get_admin_by_email("ada@example.com")
        assert admin["role"] == "super_admin" and admin["created_by"] == "emergency sign-in"
        home = client.get("/admin")
        assert "Ada Lovelace" in home.text and "Super admin" in home.text
        assert "emergency admin password" not in home.text
        # From now on the shared password lands on the console itself.
        assert _emergency(_client(app)).headers["location"] == "/admin"
        # The setup page is for the emergency sign-in only.
        assert client.get("/admin/me/setup", follow_redirects=False).headers["location"] == "/admin/me"

    def test_the_login_page_asks_for_the_shared_password_until_there_are_named_admins(self, app):
        page = _client(app).get("/admin/login").text
        assert 'name="email"' not in page and "Emergency admin password" in page
        _admin(app.store, "super_admin")
        page = _client(app).get("/admin/login").text
        assert 'name="email"' in page and "/admin/login?emergency=1" in page
        assert 'name="email"' not in _client(app).get("/admin/login?emergency=1").text


# ── Signing in ───────────────────────────────────────────────────────────────

class TestSigningIn:

    def test_with_email_and_password(self, app):
        _id, email = _admin(app.store, "super_admin")
        client = _client(app)
        wrong = _sign_in(client, email, "not-the-password")
        assert wrong.status_code == 200 and "do not match an active admin" in wrong.text
        assert "set-cookie" not in wrong.headers
        right = _sign_in(client, email.upper(), GOOD)
        assert right.status_code == 303 and right.headers["location"] == "/admin"
        assert client.get("/admin").status_code == 200
        assert app.store.get_admin_by_email(email)["last_sign_in_at"]

    def test_an_unknown_email_reads_like_a_wrong_password(self, app):
        response = _sign_in(_client(app), "nobody@example.com", GOOD)
        assert "do not match an active admin" in response.text

    def test_a_deactivated_admin_cannot_sign_in(self, app):
        admin_id, email = _admin(app.store, "super_admin")
        _admin(app.store, "super_admin")
        app.store.update_admin(admin_id, is_active=False)
        assert "do not match" in _sign_in(_client(app), email).text

    def test_guesses_wait_after_the_sixth(self, app):
        _id, email = _admin(app.store, "super_admin")
        client = _client(app)
        for _ in range(5):
            assert _sign_in(client, email, "a-guess").status_code == 200
        assert _sign_in(client, email, "a-guess").status_code == 429
        waited = _sign_in(client, email, GOOD)
        assert waited.status_code == 429 and "Too many attempts" in waited.text
        # Another admin's sign-in is counted on its own.
        _other, other = _admin(app.store, "super_admin")
        assert _sign_in(client, other).status_code == 303

    def test_signing_out_ends_the_session_in_this_browser(self, app):
        _id, email = _admin(app.store, "super_admin")
        client = _client(app)
        _sign_in(client, email)
        client.get("/admin/logout")
        assert client.get("/admin", follow_redirects=False).headers["location"] == "/admin/login"

    def test_a_password_change_signs_out_other_sessions(self, app):
        _id, email = _admin(app.store, "super_admin")
        here, there = _client(app), _client(app)
        _sign_in(here, email), _sign_in(there, email)
        new = "another-long-password"
        response = here.post("/admin/me/password", follow_redirects=False,
                             data={"current_password": GOOD, "password": new, "confirm_password": new})
        assert response.headers["location"] == "/admin/me?saved=password"
        assert here.get("/admin").status_code == 200
        assert there.get("/admin", follow_redirects=False).headers["location"] == "/admin/login"
        assert _sign_in(_client(app), email, new).status_code == 303

    def test_the_wrong_current_password_changes_nothing(self, app):
        _id, email = _admin(app.store, "super_admin")
        client = _client(app)
        _sign_in(client, email)
        response = client.post("/admin/me/password", follow_redirects=False, data={
            "current_password": "wrong-password", "password": "x" * 14, "confirm_password": "x" * 14})
        assert "error=" in response.headers["location"]
        assert _sign_in(_client(app), email, GOOD).status_code == 303


def test_a_server_set_up_with_named_admins_only_signs_them_in(tmp_path, monkeypatch):
    """No shared password was ever set (python -m admin.create_admin on a new server): not a first run."""
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    import main
    from admin import create_admin

    main.store.init_db()
    assert create_admin.main(["root@example.com", "Root", "--stdin"], stdin=io.StringIO(GOOD + "\n")) == 0
    client = TestClient(main.app)
    assert _sign_in(client, "root@example.com").headers["location"] == "/admin"
    assert client.get("/admin", follow_redirects=False).status_code == 200
    assert client.get("/admin/setup", follow_redirects=False).headers["location"] == "/admin"
    assert "no emergency admin password" in _emergency(_client(main), "anything").text
    assert "emergency=1" not in client.get("/admin/login").text


class TestInvitations:

    def _invite(self, app, client, role="data_steward", workspaces=()):
        email = f"{os.urandom(4).hex()}@example.com"
        page = client.post("/admin/admins/invite", data={"email": email, "name": "", "role": role,
                                                          "workspaces": list(workspaces)})
        link = re.search(r'value="(http://testserver/admin/invite/[^"]+)"', page.text)
        return email, page, (link.group(1) if link else None)

    def test_a_link_chooses_a_password_once(self, app):
        account_id = _workspace(app.store)
        _id, boss = _admin(app.store, "super_admin")
        client = _client(app)
        _sign_in(client, boss)
        email, page, link = self._invite(app, client, workspaces=[account_id])
        assert link and "It works once, for 72 hours" in page.text
        invited = app.store.get_admin_by_email(email)
        assert invited["role"] == "data_steward" and not invited["password_hash"]
        assert app.store.admin_workspaces(int(invited["id"])) == [account_id]
        # Waiting for a password: cannot sign in with anything.
        assert "do not match" in _sign_in(_client(app), email, "any-password-at-all").text
        theirs = _client(app)
        path = link.replace("http://testserver", "")
        assert "You are a <b>Data steward</b>" in theirs.get(path).text
        short = theirs.post(path, data={"name": "Ed", "password": "short", "confirm_password": "short"})
        assert "at least 12 characters" in short.text
        done = theirs.post(path, follow_redirects=False, data={"name": "Ed", "password": GOOD, "confirm_password": GOOD})
        assert done.status_code == 303 and done.headers["location"] == "/admin"
        assert theirs.get("/admin").status_code == 200
        assert "This link does not work" in _client(app).get(path).text
        assert _sign_in(_client(app), email).status_code == 303
        assert app.store.get_admin_by_email(email)["name"] == "Ed"

    def test_a_workspace_is_needed_below_super_admin(self, app):
        _id, boss = _admin(app.store, "super_admin")
        client = _client(app)
        _sign_in(client, boss)
        _email, page, link = self._invite(app, client)
        assert link is None and "Choose the workspaces this admin looks after" in page.text

    def test_an_expired_link_does_not_work(self, app):
        admin_id, _email = _admin(app.store, "data_steward", password=None)
        token = app.store.issue_admin_invite(admin_id, hours=-1)
        assert "This link does not work" in _client(app).get(f"/admin/invite/{token}").text

    def test_a_new_link_replaces_the_old_one(self, app):
        admin_id, _email = _admin(app.store, "data_steward", password=None)
        first = app.store.issue_admin_invite(admin_id)
        second = app.store.issue_admin_invite(admin_id)
        assert app.store.admin_for_invite(first) is None and app.store.admin_for_invite(second)


# ── What each role opens ─────────────────────────────────────────────────────

# The pages under /admin/clients/<id>/ a data steward works on (the plan's table), and what a workspace admin
# adds. Anything else about a workspace is a super admin's.
STEWARD_AREAS = {"", "learned", "answers-check", "knowledge", "relationships", "dates", "measures", "metrics", "requests",
                 "date-roles", "date-contexts", "kb", "glossary", "meanings", "drafts", "learning-queue",
                 "semantic-feedback", "model-health", "mapping", "domains", "graph", "evals", "traces", "queries",
                 "readiness", "health-score", "schema", "schema-tree", "api", "reports", "value-index",
                 "synonym-proposals"}
WORKSPACE_AREAS = {"users", "groups", "row-access", "pending-users", "admins"}
SUPER_ONLY = {"setup", "compliance", "egress", "egress-log", "billing", "kb-tables", "toggle-chat-ui", "reset",
              "update", "advanced", "settings", "diagnostics"}
OPEN_TO_ALL = {"/", "/clients", "/me", "/me/setup", "/me/password", "/me/name", "/ping", "/api/status",
               "/ws/notifications", "/api/semantic-feedback/pending-count"}


def _routes():
    from admin import routes

    seen = set()
    for route in routes.router.routes:
        path = route.path[len("/admin"):] or "/"
        if path not in seen:
            seen.add(path)
            yield path


def _filled(path: str, account_id: str) -> str:
    return re.sub(r"\{[^}]+\}", "1", path.replace("{account_id}", account_id))


def _area(path: str) -> str | None:
    found = re.match(r"^/(?:api/)?clients/\{account_id\}(?:/([^/]+))?", path)
    return (found.group(1) or "") if found else None


class TestEveryRouteAgainstEachRole:
    """Every route the console registers, as each role, for the admin's own workspace and another one."""

    def _who(self, role):
        from admin.access import AdminIdentity

        return AdminIdentity(7, "x@example.com", "X", role, frozenset({"mine"}) if role != "super_admin" else frozenset())

    def test_a_super_admin_opens_every_route(self):
        from admin.access import decide

        boss = self._who("super_admin")
        assert all(decide(boss, _filled(p, "any")).allowed for p in _routes())

    @pytest.mark.parametrize("role", ["data_steward", "workspace_admin"])
    def test_no_role_below_super_opens_another_workspace(self, role):
        from admin.access import decide

        who = self._who(role)
        for path in _routes():
            if "{account_id}" in path:
                assert not decide(who, _filled(path, "theirs")).allowed, path

    def test_a_data_steward_opens_only_the_data_pages_of_their_workspace(self):
        from admin.access import decide

        steward = self._who("data_steward")
        opened = {p for p in _routes() if decide(steward, _filled(p, "mine")).allowed}
        for path in opened:
            area = _area(path)
            assert (area in STEWARD_AREAS) if area is not None else path in OPEN_TO_ALL, path
        for path in _routes():
            area = _area(path)
            if area in SUPER_ONLY or area in WORKSPACE_AREAS:
                assert path not in opened, path
        for must in ("/clients/{account_id}/learned", "/clients/{account_id}/relationships",
                     "/clients/{account_id}/dates", "/clients/{account_id}/measures", "/clients/{account_id}/requests",
                     "/clients/{account_id}/knowledge", "/clients/{account_id}"):
            assert must in opened, must

    def test_a_workspace_admin_adds_the_people_pages(self):
        from admin.access import decide

        steward, keeper = self._who("data_steward"), self._who("workspace_admin")
        for path in _routes():
            filled = _filled(path, "mine")
            if decide(steward, filled).allowed:
                assert decide(keeper, filled).allowed, path
            area = _area(path)
            if decide(keeper, filled).allowed and not decide(steward, filled).allowed:
                assert area in WORKSPACE_AREAS, path
            if area in SUPER_ONLY:
                assert not decide(keeper, filled).allowed, path
        for must in ("/clients/{account_id}/users", "/clients/{account_id}/groups", "/clients/{account_id}/row-access",
                     "/clients/{account_id}/pending-users", "/clients/{account_id}/admins"):
            assert decide(keeper, _filled(must, "mine")).allowed, must

    @pytest.mark.parametrize("path", ["/databases", "/platforms", "/system", "/admins", "/clients/new",
                                      "/clients/create", "/system/password", "/api/admin-audit.csv"])
    def test_the_servers_own_pages_are_super_admins(self, path):
        from admin.access import decide

        for role in ("data_steward", "workspace_admin"):
            assert not decide(self._who(role), path).allowed

    def test_a_route_no_rule_names_is_closed_below_super(self):
        from admin.access import decide

        assert not decide(self._who("workspace_admin"), "/a-page-added-tomorrow").allowed
        assert not decide(self._who("workspace_admin"), "/clients/mine/a-page-added-tomorrow").allowed


class TestTheServerRefuses:

    @pytest.fixture
    def setting(self, app):
        mine, theirs = _workspace(app.store, "Mine"), _workspace(app.store, "Theirs")
        _sid, steward = _admin(app.store, "data_steward", [mine], name="Stella Steward")
        client = _client(app)
        assert _sign_in(client, steward).status_code == 303
        return app, client, mine, theirs, steward

    def test_pages_a_steward_does_not_reach(self, setting):
        app, client, mine, theirs, _email = setting
        for path in ("/admin/databases", "/admin/system", "/admin/admins", f"/admin/clients/{mine}/setup",
                     f"/admin/clients/{mine}/users", f"/admin/clients/{mine}/compliance",
                     f"/admin/clients/{theirs}", f"/admin/clients/{theirs}/learned"):
            response = client.get(path, follow_redirects=False)
            assert response.status_code == 403, path
            assert "This page is not open to you" in response.text and "Data steward" in response.text
        api = client.get(f"/admin/api/clients/{theirs}/readiness", headers={"accept": "application/json"})
        assert api.status_code == 403 and api.json()["error"] == "forbidden"
        posted = client.post(f"/admin/clients/{mine}/users/create", data={"name": "X", "email": "x@example.com"},
                             follow_redirects=False)
        assert posted.status_code == 403
        assert app.store.get_user_by_email(mine, "x@example.com") is None

    def test_pages_a_steward_does_reach(self, setting):
        _app, client, mine, _theirs, _email = setting
        assert client.get(f"/admin/clients/{mine}", follow_redirects=False).status_code == 200
        assert client.get("/admin/me").status_code == 200
        assert client.get("/admin").status_code == 200

    def test_a_steward_sees_only_their_workspaces_and_links(self, setting):
        _app, client, mine, theirs, _email = setting
        home = client.get("/admin").text
        assert f"/admin/clients/{mine}" in home and f"/admin/clients/{theirs}" not in home
        assert 'href="/admin/databases"' not in home and 'href="/admin/system"' not in home
        assert 'href="/admin/admins"' not in home
        assert "Stella Steward" in home and "Data steward" in home
        listing = client.get("/admin/clients").text
        assert f"/admin/clients/{mine}" in listing and f"/admin/clients/{theirs}" not in listing
        overview = client.get(f"/admin/clients/{mine}").text
        nav = overview[overview.index('class="client-workspace-primary"'):]
        nav = nav[:nav.index("</nav>")]
        assert f"/admin/clients/{mine}/learned" in nav            # Data opens on what a steward can reach
        for hidden in ("/setup", "/users", "/compliance", "/settings"):
            assert f"/admin/clients/{mine}{hidden}" not in nav, hidden

    def test_refusals_and_changes_are_logged_with_the_admins_email(self, setting):
        app, client, mine, theirs, email = setting
        client.get(f"/admin/clients/{theirs}/learned")
        client.post("/admin/me/name", data={"name": "Stella S"})
        events = app.store.list_admin_events(limit=10)
        refused = [e for e in events if e["action"] == "refused"]
        assert refused and refused[0]["admin_email"] == email and refused[0]["account_id"] == theirs
        changed = [e for e in events if e["action"] == "change" and e["path"] == "/admin/me/name"]
        assert changed and changed[0]["admin_email"] == email and changed[0]["status"] == 303
        assert any(e["action"] == "sign_in" and e["admin_email"] == email for e in events)

    def test_a_role_change_ends_the_admins_sessions(self, setting):
        app, client, mine, _theirs, email = setting
        admin = app.store.get_admin_by_email(email)
        app.store.update_admin(int(admin["id"]), role="workspace_admin")
        assert client.get("/admin", follow_redirects=False).headers["location"] == "/admin/login"
        assert _sign_in(client, email).status_code == 303
        assert client.get(f"/admin/clients/{mine}/users", follow_redirects=False).status_code == 200

    def test_the_notification_socket_still_opens(self, setting):
        _app, client, _mine, _theirs, _email = setting
        with client.websocket_connect("/admin/ws/notifications") as socket:
            assert socket.receive_json()["type"] == "semantic_feedback_pending"

    def test_live_notices_name_only_their_workspaces(self, setting):
        app, client, mine, theirs, _email = setting
        from core.admin_notifications import scoped

        summary = {"total": 5, "clients": [{"account_id": mine, "pending": 2}, {"account_id": theirs, "pending": 3}]}
        with patch("admin.routes.semantic_feedback_summary", return_value=summary):
            counted = client.get("/admin/api/semantic-feedback/pending-count").json()
            with client.websocket_connect("/admin/ws/notifications") as socket:
                snapshot = socket.receive_json()["summary"]
        for told in (counted, snapshot):
            assert told["total"] == 2 and [c["account_id"] for c in told["clients"]] == [mine]
        opens = {mine}.__contains__
        assert scoped({"type": "graph_ready", "account_id": theirs}, opens) is None
        assert scoped({"type": "graph_ready", "account_id": mine}, opens) == {"type": "graph_ready", "account_id": mine}
        assert scoped({"type": "x", "summary": summary}, None)["summary"]["total"] == 5

    def test_another_workspaces_socket_or_page_closes(self, app):
        mine = _workspace(app.store)
        _sid, steward = _admin(app.store, "data_steward", [mine])
        client = _client(app)
        _sign_in(client, steward)
        # A route a steward never reaches, as a socket: closed before it opens.
        from admin.access import decide, AdminIdentity

        assert not decide(AdminIdentity(1, "", "", "data_steward", frozenset({mine})), "/clients/other/learned").allowed


class TestManagingAdmins:

    def test_the_only_active_super_admin_stays(self, app):
        boss_id, boss = _admin(app.store, "super_admin")
        client = _client(app)
        _sign_in(client, boss)
        for data in ({"role": "data_steward", "is_active": "1", "workspaces": [_workspace(app.store)]},
                     {"role": "super_admin", "is_active": "0"}):
            response = client.post(f"/admin/admins/{boss_id}/update", data=data, follow_redirects=False)
            assert "only+active+super+admin" in response.headers["location"].replace("%20", "+")
        assert "only" in client.post(f"/admin/admins/{boss_id}/delete", follow_redirects=False).headers["location"]
        assert app.store.get_admin(boss_id)["role"] == "super_admin" and app.store.get_admin(boss_id)["is_active"]

    def test_a_super_admin_changes_role_and_workspaces(self, app):
        a, b = _workspace(app.store), _workspace(app.store)
        _boss_id, boss = _admin(app.store, "super_admin")
        other_id, _other = _admin(app.store, "data_steward", [a])
        client = _client(app)
        _sign_in(client, boss)
        client.post(f"/admin/admins/{other_id}/update", data={"role": "workspace_admin", "is_active": "1",
                                                              "workspaces": [a, b]})
        changed = app.store.get_admin(other_id)
        assert changed["role"] == "workspace_admin" and app.store.admin_workspaces(other_id) == sorted([a, b])
        detail = [e["detail"] for e in app.store.list_admin_events(limit=5) if e["action"] == "admin_changed"][0]
        assert "Data steward to Workspace admin" in detail
        page = client.get("/admin/admins").text
        assert "Workspace admin" in page and "What admins did" in page

    def test_emergency_sign_in_turns_off_and_back_on_from_the_server(self, app):
        _boss_id, boss = _admin(app.store, "super_admin")
        old = _client(app)
        _emergency(old)
        assert old.get("/admin").status_code == 200
        # An emergency session cannot turn emergency sign-in off.
        refused = old.post("/admin/admins/emergency", data={"state": "off"}, follow_redirects=False)
        assert "Sign+in+with+your+own" in refused.headers["location"].replace("%20", "+")
        client = _client(app)
        _sign_in(client, boss)
        client.post("/admin/admins/emergency", data={"state": "off"})
        assert old.get("/admin", follow_redirects=False).headers["location"] == "/admin/login"
        assert "turned off" in _emergency(_client(app)).text
        assert "Use the emergency admin password" not in _client(app).get("/admin/login").text
        from admin import reset_password

        assert reset_password.main(["--stdin"], stdin=io.StringIO("a-new-emergency-password\n")) == 0
        assert _emergency(_client(app), "a-new-emergency-password").status_code == 303

    def test_a_workspace_admin_adds_and_removes_stewards_in_their_workspace_only(self, app):
        mine, theirs = _workspace(app.store), _workspace(app.store)
        _kid, keeper = _admin(app.store, "workspace_admin", [mine])
        peer_id, peer = _admin(app.store, "workspace_admin", [theirs])
        client = _client(app)
        _sign_in(client, keeper)
        page = client.post(f"/admin/clients/{mine}/admins/add", data={"email": "new@example.com", "role": "data_steward"})
        assert "/admin/invite/" in page.text
        new = app.store.get_admin_by_email("new@example.com")
        assert new["role"] == "data_steward" and app.store.admin_workspaces(int(new["id"])) == [mine]
        assert "cannot grant" in client.post(f"/admin/clients/{mine}/admins/add",
                                             data={"email": "x@example.com", "role": "workspace_admin"}).text
        assert "only a super admin" in client.post(f"/admin/clients/{mine}/admins/add",
                                                   data={"email": peer, "role": "data_steward"}).text
        assert client.post(f"/admin/clients/{theirs}/admins/add", data={"email": "y@example.com"},
                           follow_redirects=False).status_code == 403
        client.post(f"/admin/clients/{mine}/admins/{new['id']}/remove")
        assert app.store.admin_workspaces(int(new["id"])) == []
        assert app.store.admin_workspaces(peer_id) == [theirs]
        listing = client.get(f"/admin/clients/{mine}/admins").text
        assert keeper in listing and peer not in listing

    def test_the_server_command_makes_a_super_admin(self, app):
        from admin import create_admin

        assert create_admin.main(["root@example.com", "Root", "--stdin"], stdin=io.StringIO(GOOD + "\n")) == 0
        assert app.store.get_admin_by_email("root@example.com")["role"] == "super_admin"
        assert _sign_in(_client(app), "root@example.com").status_code == 303
        assert create_admin.main(["x@example.com", "--role", "data_steward", "--stdin"],
                                 stdin=io.StringIO(GOOD + "\n")) == 1


class TestTheUsersPageNamesWhoDidIt:

    def test_a_reader_made_by_a_named_admin_is_logged_with_their_email(self, app):
        mine = _workspace(app.store)
        _kid, keeper = _admin(app.store, "workspace_admin", [mine])
        client = _client(app)
        _sign_in(client, keeper)
        client.post(f"/admin/clients/{mine}/users/create", data={
            "name": "Viv", "email": "viv@example.com", "portal_role": "viewer", "role": "analyst"})
        user = app.store.get_user_by_email(mine, "viv@example.com")
        assert user["portal_role"] == "viewer"
        client.post(f"/admin/clients/{mine}/users/{user['id']}/update", data={"portal_role": "asker"})
        assert app.store.get_user(user["id"])["portal_role"] == "asker"
        events = app.store.list_user_events(mine)
        assert events[0]["action"] == "role_changed" and events[0]["detail"] == "Viewer to Asker"
        assert {e["actor"] for e in events} == {keeper}
        page = client.get(f"/admin/clients/{mine}/users").text
        assert 'data-portal-role="asker"' in page and "What they can do" in page
