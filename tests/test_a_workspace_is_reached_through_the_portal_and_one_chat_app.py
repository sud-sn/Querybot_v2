"""A workspace is created with no chat platform; choosing one on its Settings page asks for what that app needs.

New workspace asked for a chat platform (or "Web portal only") before anything else, and the platform's
credentials were typed on another page (Platforms), then picked from a list (Teams only). A Slack or Zoom workspace
got its messages only when its own ID happened to be the chat's team ID. Now:

* a workspace is created with a name, an ID and (optionally) its database: people reach it through the web portal;
* its Settings page says how people reach it: the portal's sign-in address, always; and a chat app, chosen there,
  shows the credentials that app needs, the ID whose messages come to this workspace, and the address to paste
  into the app;
* the credentials are kept encrypted, a secret is never shown back, and left empty it is kept;
* Slack and Zoom messages go to the workspace that names their team or account, with that workspace's credentials;
* the rest of the Settings form never takes the chat app away.

A throwaway database and key; invented credentials.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

TEAMS = {"chat_app": "teams", "app_id": "app-1", "app_password": "teams-secret-1",
         "tenant_id": "11111111-2222-3333-4444-555555555555"}
SLACK = {"chat_app": "slack", "bot_token": "xoxb-invented", "signing_secret": "slack-secret-1", "app_id": "A01",
         "team_id": "T0INVENTED"}
ZOOM = {"chat_app": "zoom", "client_id": "zc", "client_secret": "zoom-secret-1", "bot_jid": "bot@xmpp.zoom.us",
        "webhook_secret": "zoom-token-1", "account_id": "ZOOMACCT1"}


@pytest.fixture()
def admin(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    import admin.routes as routes

    crypto = routes.store.get_platform.__globals__["decrypt_json"].__globals__
    monkeypatch.setitem(crypto, "KEY_FILE", Path(tmp_path) / "key")
    routes.store.init_db()
    app = FastAPI()
    app.include_router(routes.router)
    from admin import channels

    with patch.object(routes, "_is_auth", return_value=True), patch.object(channels, "_is_auth", return_value=True):
        yield routes.store, TestClient(app)


def _create(client, name="Acme Pharmacy", account=None) -> str:
    account = account or f"acme-{uuid.uuid4().hex[:6]}"
    response = client.post("/admin/clients/create", data={"client_name": name, "account_id": account},
                           follow_redirects=False)
    assert response.status_code == 303 and "/setup" in response.headers["location"], response.text
    return account


def _save(client, account, form) -> str:
    response = client.post(f"/admin/clients/{account}/channel", data=form, follow_redirects=False)
    assert response.status_code == 303
    return response.headers["location"]


# ── creating a workspace ────────────────────────────────────────────────────


def test_new_workspace_asks_for_no_chat_platform(admin):
    store, client = admin
    page = client.get("/admin/clients/new").text
    assert 'name="db_config_id"' in page and "Add a database connection" in page
    assert 'name="platform_type"' not in page and 'name="portal_only"' not in page
    account = _create(client)
    row = store.get_client(account)
    assert row["portal_only"] == 1 and row["chat_ui_enabled"] == 1 and not row["platform_config_id"]


def test_a_form_that_still_names_a_platform_creates_a_portal_workspace(admin):
    store, client = admin
    account = f"old-{uuid.uuid4().hex[:6]}"
    response = client.post("/admin/clients/create", data={"client_name": "Old form", "account_id": account,
                                                           "platform_type": "slack"}, follow_redirects=False)
    assert response.status_code == 303
    assert store.get_client(account)["portal_only"] == 1


# ── the Settings page ───────────────────────────────────────────────────────


def test_settings_show_the_portal_sign_in_and_each_apps_fields(admin):
    _store, client = admin
    account = _create(client)
    page = client.get(f"/admin/clients/{account}/settings").text
    assert 'id="channels"' in page and f"/portal/login?workspace={account}" in page
    for kind, fields in (("teams", TEAMS), ("slack", SLACK), ("zoom", ZOOM)):
        assert f'<option value="{kind}"' in page and f"/webhook/{kind}" in page
        for name in fields:
            if name != "chat_app":
                assert f'id="ch_{kind}_{name}"' in page, (kind, name)
    # Nothing chosen: every app's fields are hidden and never posted.
    assert page.count('class="ch-fields"') == 3 and page.count("hidden disabled") == 3


def test_teams_is_saved_encrypted_and_its_secret_never_shown_back(admin):
    store, client = admin
    account = _create(client)
    assert "saved=channel" in _save(client, account, TEAMS)
    row = store.get_client(account)
    assert row["platform_type"] == "teams" and row["teams_tenant_id"] == TEAMS["tenant_id"] and row["portal_only"] == 0
    platform = store.get_platform(row["platform_config_id"])
    assert platform["credentials"] == {k: v for k, v in TEAMS.items() if k != "chat_app"}
    with store.get_db() as conn:
        stored = conn.execute("SELECT credentials_encrypted FROM platform_config WHERE id=?",
                              (row["platform_config_id"],)).fetchone()[0]
    assert "teams-secret-1" not in str(stored)
    page = client.get(f"/admin/clients/{account}/settings").text
    assert "teams-secret-1" not in page and "Saved: leave empty to keep it" in page
    assert f'value="{TEAMS["app_id"]}"' in page and '<option value="teams" selected' in page


def test_a_secret_left_empty_is_kept_and_a_value_changed_is_saved(admin):
    store, client = admin
    account = _create(client)
    _save(client, account, TEAMS)
    _save(client, account, {**TEAMS, "app_password": "", "app_id": "app-2"})
    platform = store.get_platform(store.get_client(account)["platform_config_id"])
    assert platform["credentials"]["app_password"] == "teams-secret-1" and platform["credentials"]["app_id"] == "app-2"
    assert len(store.list_platforms("teams")) == 1, "the workspace's own app is updated, never a second one made"


def test_a_missing_credential_is_named_and_nothing_is_saved(admin):
    store, client = admin
    account = _create(client)
    location = _save(client, account, {**SLACK, "signing_secret": ""})
    assert "channel_error=" in location and "Signing%20secret" in location and "channel=slack" in location
    assert not store.get_client(account)["platform_config_id"] and not store.list_platforms("slack")
    page = client.get(location.replace("http://testserver", "")).text
    assert "Slack needs: Signing secret." in page and '<option value="slack" selected' in page


def test_one_slack_workspace_sends_to_one_workspace(admin):
    _store, client = admin
    first, second = _create(client, "First"), _create(client, "Second")
    _save(client, first, SLACK)
    location = _save(client, second, SLACK)
    assert "channel_error=" in location and "First" in location


def test_no_chat_app_takes_it_away_and_the_settings_form_never_does(admin):
    store, client = admin
    account = _create(client)
    _save(client, account, ZOOM)
    client.post(f"/admin/clients/{account}/update", data={"client_name": "Renamed", "analysis_mode": "always"},
                follow_redirects=False)
    row = store.get_client(account)
    assert row["client_name"] == "Renamed" and row["chat_team_id"] == "ZOOMACCT1" and row["platform_config_id"]
    _save(client, account, {"chat_app": ""})
    row = store.get_client(account)
    assert not row["platform_config_id"] and row["chat_team_id"] == "" and row["portal_only"] == 1
    assert '<span class="badge badge-gray">None</span>' in client.get(f"/admin/clients/{account}/settings").text


def test_a_teams_setup_another_workspace_uses_is_never_overwritten(admin):
    store, client = admin
    shared = store.save_platform("teams", "Shared bot", {"app_id": "shared", "app_password": "pw",
                                                         "tenant_id": "aaaa"})
    first, second = _create(client, "First"), _create(client, "Second")
    store.update_client_meta(first, platform_type="teams", platform_config_id=shared, teams_tenant_id="aaaa")
    store.update_client_meta(second, platform_type="teams", platform_config_id=shared)
    _save(client, second, {**TEAMS, "tenant_id": "bbbb"})
    assert store.get_platform(shared)["credentials"]["app_id"] == "shared"
    assert store.get_client(second)["platform_config_id"] != shared


# ── where a message goes ────────────────────────────────────────────────────


def test_slack_and_zoom_messages_go_to_the_workspace_that_names_them(admin):
    store, client = admin
    from gateway import PlatformEvent
    from gateway import webhooks

    slack_ws, zoom_ws = _create(client, "Slack one"), _create(client, "Zoom one")
    _save(client, slack_ws, SLACK)
    _save(client, zoom_ws, ZOOM)
    adapter = webhooks._load_chat_adapter("slack", json.dumps({"team_id": "T0INVENTED"}).encode())
    assert adapter._signing_secret == "slack-secret-1", "checked with the credentials of the workspace it is for"
    event = PlatformEvent(account_id="T0INVENTED", user_id="U1", channel_id="C1", text="hi", platform="slack")
    webhooks._to_workspace("slack", event)
    assert event.account_id == slack_ws
    zoom = PlatformEvent(account_id="ZOOMACCT1", user_id="u", channel_id="j", text="hi", platform="zoom",
                         raw={"payload": {"accountId": "ZOOMACCT1"}})
    webhooks._to_workspace("zoom", zoom)
    assert zoom.account_id == zoom_ws
    from gateway.zoom_adapter import _zoom_account

    assert _zoom_account(zoom) == "ZOOMACCT1", "a reply goes to the Zoom account the message came from"


def test_a_workspace_whose_own_id_is_the_team_keeps_its_messages(admin):
    store, client = admin
    from gateway import PlatformEvent
    from gateway import webhooks

    account = _create(client, "Old style", account="T0OLDSTYLE")
    event = PlatformEvent(account_id="T0OLDSTYLE", user_id="U1", channel_id="C1", text="hi", platform="slack")
    webhooks._to_workspace("slack", event)
    assert event.account_id == account


def test_the_overview_says_how_people_reach_it(admin):
    _store, client = admin
    account = _create(client)
    _save(client, account, SLACK)
    page = client.get(f"/admin/clients/{account}").text
    assert "Reached through" in page and "Web portal" in page and ">Slack<" in page
