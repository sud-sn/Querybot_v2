"""
A static file's URL changes when the file does, and only then.

The version on every stylesheet, script and image link was written by hand
(?v=20260926-ui-8, ?v=20260926-lucide-1, ?v=6.1.0, ?v=20260818-bubble-1) and
bumped by whoever remembered, in every template that linked the file. base.css
once shipped with none, so returning browsers kept a stale copy through every
deploy; gridstack, cytoscape and dagre never had one. Now templates write
{{ asset('css/base.css') }}, which renders the file's URL with the first 12 hex
of its SHA-256 (core/static_assets.py).

These call the helper on a file they edit, render both consoles' pages through
their own Jinja environments and check every link against the bytes it names,
and read the templates and static files for a version written by hand.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core import static_assets
from tests.portal_render import render

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


class TestTheHelper:

    @pytest.fixture
    def static(self, tmp_path, monkeypatch):
        monkeypatch.setattr(static_assets, "STATIC_DIR", tmp_path)
        (tmp_path / "css").mkdir()
        return tmp_path

    def test_the_version_is_the_files_content(self, static):
        (static / "css" / "a.css").write_text("body { color: red; }")
        assert static_assets.asset_url("css/a.css") == f"/static/css/a.css?v={_hash(static / 'css' / 'a.css')}"

    def test_an_edited_file_gets_a_new_url_and_an_untouched_one_keeps_its_own(self, static):
        edited, untouched = static / "css" / "a.css", static / "css" / "b.css"
        edited.write_text("body { color: red; }")
        untouched.write_text("p { margin: 0; }")
        before = static_assets.asset_url("css/a.css"), static_assets.asset_url("css/b.css")
        edited.write_text("body { color: navy; }")
        after = static_assets.asset_url("css/a.css"), static_assets.asset_url("css/b.css")
        assert after[0] != before[0] and after[0].endswith(_hash(edited))
        assert after[1] == before[1]

    def test_two_files_with_the_same_bytes_share_a_version_and_not_a_url(self, static):
        (static / "css" / "a.css").write_text("x")
        (static / "css" / "b.css").write_text("x")
        a, b = static_assets.asset_url("css/a.css"), static_assets.asset_url("css/b.css")
        assert a != b and a.split("?")[1] == b.split("?")[1]

    def test_a_missing_file_still_links_and_says_so(self, static, caplog):
        with caplog.at_level(logging.WARNING, logger="querybot.static_assets"):
            assert static_assets.asset_url("css/gone.css") == "/static/css/gone.css"
        assert "css/gone.css" in caplog.text


# ── Every link a page is served ─────────────────────────────────────────────

def _admin(template: str, path: str = "/admin/", **context) -> str:
    import admin.routes as routes

    request = MagicMock()
    request.url.path = path
    return routes.templates.env.get_template(template).render(request=request, **context)


def _links(html: str) -> list[str]:
    """Every /static/ URL an attribute of the markup loads (a script's comment
    naming a file is not a link)."""
    return re.findall(r'(?:href|src|data-sprite)="(/static/[A-Za-z0-9_./-]+(?:\?v=[\w-]*)?)[#"]', html)


PAGES = {
    "admin shell": lambda: _admin("base.html"),
    "admin sign-in": lambda: _admin("login.html", "/admin/login"),
    "relationships": lambda: _admin("client_graph.html", "/admin/clients/acct/graph",
                                    client={"account_id": "acct", "client_name": "Acme"}, entities=[],
                                    relationships=[], health={}, pending_count=0, entity_columns={}),
    "portal shell": lambda: render("portal_notifications.html"),
    "portal dashboard": lambda: __import__("tests.test_dashboard_page", fromlist=["_render"])._render(),
    "portal chat": lambda: render("portal_chat.html", path="/portal/chat"),
    "portal sign-in": lambda: render("portal_login.html", path="/portal/login"),
}


@pytest.mark.parametrize("page", sorted(PAGES))
def test_every_link_carries_the_hash_of_the_file_it_names(page):
    html = PAGES[page]()
    links = _links(html)
    assert links, page
    for link in links:
        path, _, version = link.partition("?v=")
        file = ROOT / path.lstrip("/")
        assert file.is_file(), (page, link)
        if path.startswith("/static/fonts/"):
            # Linked bare: fonts.css fetches them by that URL, and a preload
            # is only used when its URL is the one the stylesheet fetches.
            assert not version, (page, link)
            assert path in (STATIC / "css" / "fonts.css").read_text(encoding="utf-8")
            continue
        assert version == _hash(file), (page, link)


def test_the_script_built_mark_is_the_shells():
    html = render("portal_notifications.html")
    src = re.search(r'window\.QB_MARK_SRC = "([^"]+)";', html).group(1)
    assert src == f"/static/img/logo-mark.svg?v={_hash(STATIC / 'img' / 'logo-mark.svg')}"


# ── No version written by hand ──────────────────────────────────────────────

def _templates():
    return list((ROOT / "admin" / "templates").rglob("*.html")) + list((ROOT / "portal" / "templates").rglob("*.html"))


def test_no_template_writes_a_static_url_itself():
    by_hand = []
    for page in _templates():
        text = re.sub(r"\{#[\s\S]*?#\}", "", page.read_text(encoding="utf-8"))
        text = re.sub(r"^\s*//.*$", "", text, flags=re.M)             # a script comment naming a file
        for m in re.finditer(r'/static/[A-Za-z0-9_./-]+', text):
            if not m.group(0).startswith("/static/fonts/"):
                by_hand.append(f"{page.name}: {m.group(0)}")
    assert not by_hand, by_hand


def test_no_static_file_names_a_version_of_another():
    written = []
    for file in list(STATIC.glob("js/*.js")) + list(STATIC.glob("css/*.css")):
        written += [f"{file.name}: {v}" for v in re.findall(r"/static/[\w./-]+\?v=[\w.-]+", file.read_text(encoding="utf-8"))]
    assert not written, written
