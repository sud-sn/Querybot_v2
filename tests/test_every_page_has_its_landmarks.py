"""
Every page has one main landmark, a way to skip to it, and a level-one heading.

An axe audit of nineteen admin, portal and sign-in views found no <main> on any
portal page or on either sign-in page -- the portal's content sat in
<div class="main">, the admin sign-in in <div class="auth-page"> -- so a screen
reader's "go to main content" went nowhere and whole pages sat outside every
landmark. Neither console had a skip link, so a keyboard user tabbed through
the sidebar, twenty-odd links, on every page. The sign-in pages, the client
overview, the relationships page and five more had no level-one heading, and
the shared empty state titled itself <h3> directly under a page's <h1>.

These render the shells, signed in and out, through the consoles' own Jinja
environments and read them as a browser would: the first control a keyboard
reaches, the landmark it points at, the headings in order.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.portal_render import render

ROOT = Path(__file__).resolve().parents[1]
USER = {"id": 1, "name": "Ada Lovelace", "account_id": "a", "role": "analyst", "group_name": "Analysts"}
CLIENT = {"account_id": "acct", "client_name": "Acme", "state": "READY", "chat_ui_enabled": 1}


class _Page(HTMLParser):
    """The body's focusable elements in order, its landmarks and its headings."""

    FOCUSABLE = {"a", "button", "input", "select", "textarea"}

    def __init__(self, html: str):
        super().__init__()
        self.in_body = False
        self.focusable: list[dict] = []
        self.mains: list[dict] = []
        self.headings: list[int] = []
        self._link: dict | None = None
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "body":
            self.in_body = True
        if not self.in_body:
            return
        if tag == "main":
            self.mains.append(a)
        if re.fullmatch(r"h[1-6]", tag):
            self.headings.append(int(tag[1]))
        if (tag in self.FOCUSABLE and a.get("type") != "hidden" and (tag != "a" or "href" in a)) \
                or a.get("tabindex", "-1") not in ("-1",):
            a["tag"], a["text"] = tag, ""
            self.focusable.append(a)
            if tag == "a":
                self._link = a

    def handle_endtag(self, tag):
        if tag == "a":
            self._link = None

    def handle_data(self, data):
        if self._link is not None:
            self._link["text"] += data


def _admin(template: str, path: str, **context) -> str:
    """Rendered through the console's own environment; a value the test does
    not supply renders empty, as tests/portal_render.py does for the portal."""
    from jinja2 import ChainableUndefined

    import admin.routes as routes

    request = MagicMock()
    request.url.path = path
    request.query_params = {}
    env = routes.templates.env
    previous, env.undefined = env.undefined, ChainableUndefined
    try:
        return env.get_template(template).render(request=request, **context)
    finally:
        env.undefined = previous


SHELLS = {
    "admin console": lambda: _admin("base.html", "/admin/"),
    "admin sign-in": lambda: _admin("login.html", "/admin/login"),
    "portal signed in": lambda: render("portal_notifications.html", path="/portal/notifications", user=USER,
                                       alerts=[], reports=[], subscriptions={}, saved=None, error=None),
    "portal sign-in": lambda: render("portal_login.html", path="/portal/login"),
    "portal chat": lambda: render("portal_chat.html", path="/portal/chat", user=USER, enabled=True),
}


@pytest.mark.parametrize("shell", sorted(SHELLS))
class TestTheKeyboardCanSkipToTheContent:

    def test_there_is_one_main_landmark_and_it_can_take_focus(self, shell):
        page = _Page(SHELLS[shell]())
        assert len(page.mains) == 1, page.mains
        assert page.mains[0].get("id") == "main-content"
        assert page.mains[0].get("tabindex") == "-1"

    def test_the_first_thing_a_tab_reaches_skips_to_it(self, shell):
        first = _Page(SHELLS[shell]()).focusable[0]
        assert first["tag"] == "a" and first["href"] == "#main-content", first
        assert first["text"].strip() == "Skip to content"


def test_a_french_reader_skips_in_french():
    page = _Page(render("portal_login.html", path="/portal/login", lang="fr"))
    assert page.focusable[0]["text"].strip() == "Aller au contenu"


# ── Headings ─────────────────────────────────────────────────────────────────

PAGES = {
    "admin sign-in": SHELLS["admin sign-in"],
    "portal sign-in": SHELLS["portal sign-in"],
    "portal chat": SHELLS["portal chat"],
    "client overview": lambda: _admin(
        "client_detail.html", "/admin/clients/acct", client=CLIENT, erp_packs_available=[], client_erp_packs=[],
        queries=[], llm_calls=[], audit_component=None, audit_status=None, stats={}, all_dbs=[], query_models=[],
        saved=None, cost_rates={}, monthly_count=0, limit_pct=0, failed_queries=[], top_questions=[],
        token_status={"limit": 0, "remaining": 0, "limit_pct": 0, "unlimited": True}, kb_file_count=0,
        schema_file_count=0, semantic_pending_count=0, db_name="", system_model="",
        health_score={"score": 80, "max": 100, "pct": 80, "grade": "B", "color": "green", "components": []},
        analysis_subtasks=[], latest_sql_eval=None, latest_sql_pass_rate=None, teams_platforms=[],
        active_tab="overview"),
    "relationships": lambda: _admin(
        "client_graph.html", "/admin/clients/acct/graph", client=CLIENT, entities=[], relationships=[],
        health={}, pending_count=0, entity_columns={}),
}


@pytest.mark.parametrize("name", sorted(PAGES))
def test_each_page_has_one_level_one_heading(name):
    headings = _Page(PAGES[name]()).headings
    assert headings.count(1) == 1, headings


def _page_templates():
    for page in sorted(list((ROOT / "admin" / "templates").rglob("*.html")) +
                       list((ROOT / "portal" / "templates").rglob("*.html"))):
        text = page.read_text(encoding="utf-8")
        if page.name not in ("base.html", "portal_base.html") and \
                re.search(r"\{%\s*block (?:content|auth_content)\s*%\}", text):
            yield page, text


def _with_includes(page: Path, text: str) -> str:
    folder = ROOT / ("admin" if "admin" in page.parts else "portal") / "templates"
    for name in re.findall(r'\{%\s*include\s+"([^"]+)"', text):
        text += (folder / name).read_text(encoding="utf-8")
    return text


def test_every_page_template_titles_itself_with_a_level_one_heading():
    untitled = [page.name for page, text in _page_templates()
                if not re.search(r"<h1\b|page_header\(", _with_includes(page, text))]
    assert not untitled, untitled


@pytest.mark.parametrize("console", ["admin", "portal"])
def test_the_empty_state_is_titled_one_level_below_the_page(console):
    import admin.routes
    import portal.routes

    env = (admin.routes if console == "admin" else portal.routes).templates.env
    html = env.from_string('{% from "macros.html" import empty_state %}{{ empty_state("inbox", "Nothing yet") }}').render()
    assert _Page("<body>" + html).headings == [2]
