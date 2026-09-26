"""
A platform save that fails never writes a secret back into the page.

When a Teams, Slack or Zoom platform could not be saved -- a required field
left empty -- the page was rendered again with every credential filled back
into the form: the secrets just typed, and, for an edit, the stored ones the
save had fallen back to for fields sent masked. They sat in the page source
of an admin page, and wherever that page was cached. The list below the form
shows the same credentials masked, which is the point of it.

Now what was typed comes back except the secrets, and the page says they
must be entered again.

The real route, store and template, with a scratch database.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch
from urllib.parse import urlencode

import pytest
from starlette.requests import Request

TYPED_SECRETS = {"app_password": "typed-app-password-1", "client_secret": "typed-client-secret-2",
                 "webhook_secret": "typed-webhook-secret-3", "bot_token": "xoxb-typed-token-4",
                 "signing_secret": "typed-signing-secret-5"}


@pytest.fixture
def admin(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    monkeypatch.setenv("QUERYBOT_KEY_FILE", str(tmp_path / ".key"))
    from admin import routes

    routes.store.init_db()
    return routes


def _save(routes, fields):
    body = urlencode(fields).encode()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request({
        "type": "http", "method": "POST", "path": "/admin/platforms/save", "root_path": "",
        "scheme": "http", "query_string": b"", "server": ("testserver", 80), "client": ("127.0.0.1", 1),
        "headers": [(b"content-type", b"application/x-www-form-urlencoded"),
                    (b"content-length", str(len(body)).encode())],
    }, receive)
    with patch.object(routes, "_is_auth", return_value=True):
        return asyncio.run(routes.platform_save(request))


class TestAFailedSave:

    def test_keeps_what_was_typed_but_not_a_teams_password(self, admin):
        page = _save(admin, {"platform_type": "teams", "name": "Teams bot", "app_id": "app-id-6",
                             "app_password": TYPED_SECRETS["app_password"], "tenant_id": ""})
        html = page.body.decode()
        assert "Missing required fields for teams: tenant_id" in html
        assert 'value="Teams bot"' in html and 'value="app-id-6"' in html
        assert TYPED_SECRETS["app_password"] not in html
        assert "Secrets are never shown back on a page" in html

    def test_nor_zoom_secrets(self, admin):
        page = _save(admin, {"platform_type": "zoom", "name": "Zoom bot", "client_id": "client-id-7",
                             "client_secret": TYPED_SECRETS["client_secret"], "bot_jid": "",
                             "webhook_secret": TYPED_SECRETS["webhook_secret"]})
        html = page.body.decode()
        assert 'value="client-id-7"' in html
        assert TYPED_SECRETS["client_secret"] not in html
        assert TYPED_SECRETS["webhook_secret"] not in html

    def test_nor_slack_secrets(self, admin):
        page = _save(admin, {"platform_type": "slack", "name": "Slack bot", "app_id": "",
                             "bot_token": TYPED_SECRETS["bot_token"],
                             "signing_secret": TYPED_SECRETS["signing_secret"]})
        html = page.body.decode()
        assert TYPED_SECRETS["bot_token"] not in html
        assert TYPED_SECRETS["signing_secret"] not in html

    def test_nor_a_stored_secret_an_edit_fell_back_to(self, admin):
        """A record saved before a field became required fails today's check
        on edit; the fields sent masked were read back from storage."""
        # Written through the store module the route itself reads. By the time a
        # full run gets here other tests have re-imported store under keys of
        # their own (tests/test_client_sources.py), and a row encrypted by
        # whatever `import store` resolves to now need not decrypt in the
        # route's copy.
        route_store = admin.store.get_platform.__globals__
        with route_store["get_db"]() as conn:
            platform_id = conn.execute(
                "INSERT INTO platform_config (platform_type, name, is_active, credentials_encrypted) "
                "VALUES ('teams', 'Old bot', 1, ?)",
                (route_store["encrypt"]({"app_id": "app-id-8", "app_password": "stored-app-password-9"}),)
            ).lastrowid
        page = _save(admin, {"platform_id": str(platform_id), "platform_type": "teams", "name": "Old bot",
                             "app_id": "", "app_password": "••••••••", "tenant_id": ""})
        html = page.body.decode()
        assert "tenant_id" in html
        assert "stored-app-password-9" not in html


class TestWhatIsUnchanged:

    def test_a_save_that_fails_with_no_secret_typed_says_nothing_of_secrets(self, admin):
        html = _save(admin, {"platform_type": "teams", "name": "Teams bot", "app_id": "app-id-6",
                             "app_password": "", "tenant_id": ""}).body.decode()
        assert "Secrets are never shown back on a page" not in html

    def test_a_complete_save_stores_every_credential(self, admin):
        response = _save(admin, {"platform_type": "teams", "name": "Teams bot", "app_id": "app-id-6",
                                 "app_password": TYPED_SECRETS["app_password"], "tenant_id": "common"})
        assert response.headers["location"] == "/admin/platforms?saved=1"
        [stored] = admin.store.list_platforms("teams")
        assert stored["credentials"] == {"app_id": "app-id-6", "app_password": TYPED_SECRETS["app_password"],
                                         "tenant_id": "common"}
