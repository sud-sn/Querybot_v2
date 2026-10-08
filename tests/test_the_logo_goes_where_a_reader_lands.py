"""The logo goes where a reader lands after signing in.

Signing in lands a reader in the chat when their workspace has it on, on the
dashboards otherwise; the logo in the menu still went to the dashboards. It now
goes through /portal/home, which follows the same rule.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest

from tests.portal_render import render


@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    from portal import routes

    routes.store.init_db()
    return routes.store


def _home(store, *, chat: bool, signed_in: bool = True):
    from portal import routes

    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    store.update_client_meta(account_id, chat_ui_enabled=1 if chat else 0)
    user = {"id": 1, "account_id": account_id, "role": "analyst"} if signed_in else None
    with patch.object(routes, "_get_portal_user", return_value=user):
        return asyncio.run(routes.portal_home(MagicMock()))


def test_with_the_chat_on_the_logo_goes_to_the_chat(fresh_store):
    response = _home(fresh_store, chat=True)
    assert response.status_code == 303 and response.headers["location"] == "/portal/chat"


def test_with_the_chat_off_it_goes_to_the_dashboards(fresh_store):
    assert _home(fresh_store, chat=False).headers["location"] == "/portal/dashboard"


def test_signed_out_it_goes_to_the_sign_in(fresh_store):
    assert _home(fresh_store, chat=True, signed_in=False).headers["location"] == "/portal/login"


def test_both_logos_link_home():
    html = render("portal_kb.html", path="/portal/kb", user={"id": 1, "name": "Ada", "account_id": "a"},
                  semantic_tables=[], visible_tables=[], schemas=[], pending_count=0)
    assert html.count('<a href="/portal/home" class="portal-brand"') == 2
    assert 'href="/portal/dashboard" class="portal-brand"' not in html
