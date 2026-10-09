"""The admin shell is the approved one, on every page.

One sidebar: the places (Dashboard, Workspaces, Databases, Chat platforms, System),
the workspaces with the open one marked, and who is signed in with the way out. A
workspace page carries its header across the top: where it is (Workspaces / the
workspace / the area, and the page above this one), its name and whether it answers,
the way into its reader chat, the six areas and the open area's own menu. A trail of
one (a top-level page's own title) is not drawn.
"""

from __future__ import annotations

import os
import re
from unittest.mock import patch

import pytest


@pytest.fixture
def admin(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    import store
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from admin import core2_metrics, core2_relationships, core2_routes, routes

    store.init_db()
    account = f"acct{os.urandom(4).hex()}"
    for account_id, name in ((account, "Retail (invented)"), ("acct-zz-other", "Zebra stores (invented)")):
        store.upsert_client(account_id, "portal")
        store.update_client_meta(account_id, client_name=name)
    app = FastAPI()
    app.include_router(routes.router)
    with patch.object(routes, "_is_auth", return_value=True), \
            patch.object(core2_routes, "_is_auth", return_value=True), \
            patch.object(core2_relationships, "_is_auth", return_value=True), \
            patch.object(core2_metrics, "_is_auth", return_value=True):
        yield TestClient(app), account, store


def _sidebar(html: str) -> str:
    return html[html.index('<aside class="sidebar"'):html.index("</aside>")]


def _head(html: str) -> str:
    return html[html.index('<section class="client-workspace-nav ws-head"'):html.index("</section>", html.index("ws-head"))]


def test_the_sidebar_holds_the_places_the_workspaces_and_the_way_out(admin):
    client, account, store = admin
    side = _sidebar(client.get(f"/admin/clients/{account}/measures").text)
    labels = re.findall(r'class="sidebar-link[^"]*"[^>]*>\s*<span class="sidebar-link-icon">.*?</span>\s*([^<]+?)\s*(?:<span|</a>)', side, re.S)
    assert labels == ["Dashboard", "Workspaces", "Databases", "Chat platforms", "System"]
    assert re.search(r'href="/admin/clients" class="sidebar-link active" aria-current="true"', side)
    names = re.findall(r'class="sidebar-workspace[^"]*"[^>]*title="([^"]+)"', side)
    assert names == ["Retail (invented)", "Zebra stores (invented)"]
    assert re.search(rf'href="/admin/clients/{account}" class="sidebar-workspace active"\s+aria-current="true"', side)
    assert 'href="/admin/logout"' in side and "Administrator" in side


def test_the_sidebar_lists_a_few_workspaces_and_always_the_open_one(admin):
    client, account, store = admin
    for i in range(10):
        store.upsert_client(f"acct-a{i:02d}", "portal")
        store.update_client_meta(f"acct-a{i:02d}", client_name=f"Aardvark {i:02d}")
    side = _sidebar(client.get(f"/admin/clients/{account}/measures").text)
    names = re.findall(r'class="sidebar-workspace[^"]*"[^>]*title="([^"]+)"', side)
    assert names[:8] == [f"Aardvark {i:02d}" for i in range(8)] and names[8] == "Retail (invented)"
    assert "All 12 workspaces" in side


def _nav(path: str, state: str = "READY") -> str:
    from jinja2 import Environment, FileSystemLoader

    from core.static_assets import asset_url

    env = Environment(loader=FileSystemLoader("admin/templates"), autoescape=True)
    env.globals["asset"] = asset_url
    request = type("R", (), {"url": type("U", (), {"path": f"/admin/clients/demo{path}"})()})()
    return env.get_template("_client_workspace_nav.html").render(
        request=request, client={"account_id": "demo", "client_name": "Demo", "state": state})


def test_a_workspace_page_says_where_it_is_and_whether_the_workspace_answers(admin):
    client, account, store = admin
    html = client.get(f"/admin/clients/{account}/measures").text
    head = _head(html)
    crumbs = re.findall(r'<a href="([^"]+)">([^<]+)</a>', head[:head.index("ws-title")])
    assert crumbs == [("/admin/clients", "Workspaces"), (f"/admin/clients/{account}", "Retail (invented)"),
                      (f"/admin/clients/{account}/setup", "Data")]
    assert f'href="/portal/login?workspace={account}"' in head and 'target="_blank"' in head
    menu = html[html.index('class="client-workspace-secondary ws-menu"'):]
    assert re.findall(r'>([^<>]+?)</a>', menu[:menu.index("</nav>")]) == [
        "Setup", "What QueryBot learned", "Knowledge base", "Relationships", "Dates", "Metrics",
        "Requests"]
    # The header says where the page is: the page's own trail is not drawn twice.
    assert 'class="breadcrumb"' not in html
    # A page below one of the menu's pages names that page too.
    nav = _nav("/measures/edit")
    trail = nav[:nav.index("ws-title")]
    assert re.findall(r'>([^<>]+)</a>', trail) == ["Workspaces", "Demo", "Data", "Metrics"]


def test_the_status_reads_in_words():
    for state, words in [("READY", "Answering"), ("KB_BUILDING", "Building the knowledge base"), ("ERROR", "Needs attention"),
                         ("NEW", "Being set up")]:
        assert re.search(rf'class="ws-status [a-z]+">{words}<', _nav("", state)), state


def test_a_top_level_page_has_no_workspace_header_and_no_trail_of_one(admin):
    client, account, store = admin
    html = client.get("/admin/clients").text
    assert 'class="client-workspace-nav ws-head"' not in html and 'class="breadcrumb"' not in html
    assert "<h1>Workspaces</h1>" in html
    assert re.search(r'href="/admin/clients" class="sidebar-link active" aria-current="page"', _sidebar(html))
