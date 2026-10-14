"""The admin console in the approved design: the portal's window, its sidebar panel, Light, Dark or Auto.

The console was a full-bleed page with a dark-edged sidebar, small headings, cards with a grey head and a shadow,
square buttons, uppercase table heads and number labels, and no dark theme. It now sits in the portal's window,
which is the whole browser window (a rounded window inset on a coloured page, capped at 1600px, left bands round it
and cut tables off at 100% zoom): the sidebar a light panel inside it (the places as pills, the workspaces,
Appearance, who is signed in), the page scrolling on the window and using its whole width. A page opens with its
name in the display face; a card is on the card radius without a grey head or a shadow; buttons are pills; a table's
head and a number's label are sentence case; the theme is the one chosen on the device, as in the portal.

Two things the new frame broke, and that stay fixed: the area's menu (Setup, Metrics, Relationships...) was squeezed
to a sliver above a long page once the main column scrolled inside the window; the workspace overview's two columns
ran off a phone's screen.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
ADMIN_CSS = (ROOT / "static/css/admin.css").read_text(encoding="utf-8")


def _rule(css: str, selector: str) -> str:
    """The declarations of the last rule whose selector list holds `selector` (the later rule wins), outside any
    @media block: the desk's own rules."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    css = re.sub(r"@media[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}", "", css)
    found = None
    for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        if selector in [s.strip() for s in match.group(1).split(",")]:
            found = match.group(2)
    assert found is not None, f"no rule for {selector}"
    return found


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
    store.upsert_client(account, "portal")
    store.update_client_meta(account, client_name="Retail (invented)")
    app = FastAPI()
    app.include_router(routes.router)
    with patch.object(routes, "_is_auth", return_value=True), \
            patch.object(core2_routes, "_is_auth", return_value=True), \
            patch.object(core2_relationships, "_is_auth", return_value=True), \
            patch.object(core2_metrics, "_is_auth", return_value=True):
        yield TestClient(app), account


PORTAL_CSS = (ROOT / "static/css/portal.css").read_text(encoding="utf-8")
HOME_CSS = (ROOT / "static/css/home.css").read_text(encoding="utf-8")
DASHBOARD_CSS = (ROOT / "static/css/dashboard.css").read_text(encoding="utf-8")


@pytest.mark.parametrize("css,shell,body", [(ADMIN_CSS, ".shell", "body:has(.shell)"),
                                            (PORTAL_CSS, ".portal-app-shell", "body:has(.portal-app-shell)")])
def test_the_console_and_the_portal_fill_the_browser_window(css, shell, body):
    """At any screen size and any zoom: no margin round the window, no cap on its width, no rounded frame, and the
    height of what the browser shows (100dvh, which follows a phone's browser bar)."""
    rule = _rule(css, shell)
    for part in ("width: 100%", "height: 100dvh", "margin: 0", "var(--window)", "display: grid"):
        assert part in rule, part
    for gone in ("max-width", "border-radius", "100vw", "- 32px"):
        assert gone not in rule, gone
    assert "var(--window)" in _rule(css, body)


@pytest.mark.parametrize("css,selector", [(ADMIN_CSS, ".page"), (HOME_CSS, ".home"),
                                          (DASHBOARD_CSS, ".dashboard-page")])
def test_a_page_uses_the_whole_width_it_is_given(css, selector):
    rule = _rule(css, selector)
    assert not re.search(r"max-width:\s*\d", rule), rule


def test_the_console_sits_in_the_portal_s_window():
    shell = _rule(ADMIN_CSS, ".shell")
    assert "var(--window)" in shell and "display: grid" in shell
    side = _rule(ADMIN_CSS, ".shell > .sidebar")
    assert "border-radius: 24px" in side and "position: relative" in side
    main = _rule(ADMIN_CSS, ".shell-main")
    assert "overflow-y: auto" in main, "the page scrolls on the window, the sidebar stays"
    active = _rule(ADMIN_CSS, ".sidebar-link.active")
    assert "var(--sidebar-active)" in active and "var(--shadow)" in active, "the open place is a raised pill"


def test_what_sits_above_a_long_page_keeps_its_height():
    assert "flex-shrink: 0" in _rule(ADMIN_CSS, ".shell-main > :not(.page)")
    assert "flex: 1 0 auto" in _rule(ADMIN_CSS, "body:not(.page-bleed) .shell-main > .page")


def test_pages_cards_buttons_and_tables_are_the_design_s():
    heading = _rule(ADMIN_CSS, ".shell .page-header h1")
    assert "var(--font-display)" in heading and "clamp(" in heading
    card = _rule(ADMIN_CSS, ".shell .card")
    assert "var(--radius-card)" in card and "box-shadow: none" in card
    assert "background: none" in _rule(ADMIN_CSS, ".shell .card-header")
    assert "text-transform: none" in _rule(ADMIN_CSS, ".shell .card-header h2")
    assert "var(--radius-pill)" in _rule(ADMIN_CSS, ".shell .btn")
    assert "text-transform: none" in _rule(ADMIN_CSS, ".shell table th")
    label = _rule(ADMIN_CSS, ".shell .metric-label")
    assert "text-transform: none" in label
    assert "var(--font-display)" in _rule(ADMIN_CSS, ".shell .metric-value")
    assert "var(--font-display)" in _rule(ADMIN_CSS, ".ws-name")


def test_a_page_is_drawn_in_the_theme_chosen_on_the_device(admin):
    client, account = admin
    html = client.get(f"/admin/clients/{account}/measures").text
    head = html[:html.index("</head>")]
    assert "css/tokens-dark.css" in head
    theme = head.index("localStorage.getItem('qb.theme')")
    assert theme < head.index("css/base.css"), "set before the first paint: a dark page never flashes white"
    side = html[html.index('<aside class="sidebar"'):html.index("</aside>")]
    choices = re.findall(r'data-theme-choice="(\w+)"', side)
    assert choices == ["light", "dark", "auto"]
    assert 'role="group" aria-label="Appearance"' in side
    script = html[html.index("Appearance: the choice is kept on this device"):]
    assert "localStorage.setItem('qb.theme', theme)" in script and "qb-theme-change" in script


def test_the_workspace_overview_is_one_column_on_a_phone():
    source = (ROOT / "admin/templates/client_detail.html").read_text(encoding="utf-8")
    assert "@media(max-width:720px){.overview-grid{grid-template-columns:minmax(0,1fr)}}" in source
    assert ".overview-card h2{font-family:var(--font-display)" in source
    assert "text-transform:uppercase" not in source[source.index(".overview-card h2"):source.index(".overview-card h2") + 200]


def test_the_sign_in_card_is_the_portal_s():
    source = (ROOT / "admin/templates/_auth_styles.html").read_text(encoding="utf-8")
    assert re.search(r"\.auth-card \{[^}]*var\(--radius-card\)", source)
    assert re.search(r"\.auth-card \.btn \{[^}]*var\(--radius-pill\)", source)
    assert re.search(r"\.auth-card \.card-header h1 \{[^}]*var\(--font-display\)", source)
