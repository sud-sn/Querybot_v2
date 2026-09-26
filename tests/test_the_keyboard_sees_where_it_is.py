"""
The keyboard can see where it is, and never lands somewhere unseen.

Tabbing through fourteen admin, portal and sign-in pages found seventeen
controls that looked the same focused as not: the portal sidebar's links,
its language and settings buttons, the active navigation item, a database
type button, the chat's suggestion chips. The focus ring was a box-shadow,
and any component that drew its own shadow -- a flat sidebar link, the active
item's accent bar -- replaced it. The ring was also --primary at 26%, about
1.5:1 against white, and navy on the navy sidebar. On the chat page the closed
analysis pane was aria-hidden but its tabs and buttons stayed in the tab
order, the history list was a second "complementary" landmark inside the
sidebar's, and the header's language switch was a second landmark with the
sidebar's name.

Now links and buttons show a 2px outline in --focus-ring-color, which a
component's shadow cannot erase, and the sidebars set it to their accent. The
closed pane is inert. These compute the ring's contrast from tokens.css, read
the stylesheets for any rule that could erase it, run the chat page's own pane
functions, and render the chat page for its landmarks.
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path

import dukpy
import pytest

from tests.js_fakedom import FAKE_DOM
from tests.js_lift import function as lift
from tests.portal_render import render

ROOT = Path(__file__).resolve().parents[1]
CSS = ROOT / "static" / "css"
CHAT = (ROOT / "portal" / "templates" / "portal_chat.html").read_text(encoding="utf-8")


def _strip(css: str) -> str:
    return re.sub(r"/\*[\s\S]*?\*/", "", css)


def _tokens() -> dict[str, str]:
    return {k: v.upper() for k, v in re.findall(r"(--[\w-]+):\s*(#[0-9a-fA-F]{6})\b", _strip((CSS / "tokens.css").read_text(encoding="utf-8")))}


def _lum(hexcolour: str) -> float:
    h = hexcolour.lstrip("#")
    channels = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    r, g, b = (c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _ratio(a: str, b: str) -> float:
    high, low = sorted((_lum(a), _lum(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _rules(css: str):
    return [(" ".join(sel.split()), body) for sel, body in re.findall(r"([^{}]+)\{([^{}]*)\}", _strip(css))]


BASE = (CSS / "base.css").read_text(encoding="utf-8")


def _ring_rule() -> str:
    """The rule that draws focus on a link or a button."""
    for selector, body in _rules(BASE):
        parts = {p.strip() for p in selector.split(",")}
        if {"a:focus-visible", "button:focus-visible"} <= parts:
            return body
    raise AssertionError("no focus rule for links and buttons")


class TestTheRing:

    def test_it_is_an_outline_a_component_s_shadow_cannot_replace(self):
        body = _ring_rule()
        assert re.search(r"outline\s*:\s*2px solid var\(--focus-ring-color\)", body), body
        assert "box-shadow" not in body

    def test_it_reads_on_every_light_surface(self):
        tokens = _tokens()
        for ground in ("--surface-raised", "--surface", "--surface-alt", "--paper", "--paper-sunken", "--paper-recessed"):
            assert _ratio(tokens["--focus-ring-color"], tokens[ground]) >= 3, ground

    @pytest.mark.parametrize("sidebar", [".sidebar", ".client-sidebar", ".portal-sidebar"])
    def test_on_a_navy_sidebar_it_is_a_colour_that_reads_there(self, sidebar):
        tokens = _tokens()
        colours = [re.search(r"--focus-ring-color\s*:\s*var\((--[\w-]+)\)", body)
                   for selector, body in _rules(BASE) if sidebar in [p.strip() for p in selector.split(",")]]
        colour = next(m.group(1) for m in colours if m)
        for ground in ("--sidebar-bg", "--sidebar-surface", "--sidebar-active"):
            assert _ratio(tokens[colour], tokens[ground]) >= 3, (sidebar, ground)


def _all_css():
    for sheet in sorted(CSS.glob("*.css")):
        if sheet.name not in ("tokens.css", "fonts.css"):
            yield sheet.name, sheet.read_text(encoding="utf-8")
    for page in sorted(list((ROOT / "admin" / "templates").rglob("*.html")) +
                       list((ROOT / "portal" / "templates").rglob("*.html"))):
        yield page.name, "\n".join(re.findall(r"<style[^>]*>([\s\S]*?)</style>", page.read_text(encoding="utf-8")))


# Fields draw their own focus (a border and a halo) and clear the outline in
# their base rule; these are they. None may be added: a link or a button that
# clears its outline outside :focus has no ring at all.
FIELDS_WITHOUT_OUTLINE = {
    "base.css": 'input:not([type="checkbox"], [type="radio"], [type="hidden"], [type="file"], [type="range"], [type="color"])',
    "dashboard.css": ".chart-card-title[contenteditable=true]",
    "client_graph.html": "#sb-search|.jc-cond-row input|.jc-select|.jc-where|.jc-col-sel|.filter-ta|.eg-inp",
    "_styles.html": ".fn-search|.metrics-toolbar input[type=text]|.mc-fields-search",
    "portal_chat.html": ".data-table-filter|.rc-input|.chat-input|.dashboard-picker-body|.hp-search-input",
}


def test_nothing_but_a_field_clears_its_outline_outside_focus():
    cleared = []
    for where, css in _all_css():
        for selector, body in _rules(css):
            if re.search(r"(?:^|;|\s)outline\s*:\s*(?:none|0)\b", body) and ":focus" not in selector:
                allowed = FIELDS_WITHOUT_OUTLINE.get(where, "")
                if not any(field in selector for field in allowed.split("|") if field):
                    cleared.append(f"{where}: {selector[:80]}")
    assert not cleared, cleared


# ── The chat page's closed pane ──────────────────────────────────────────────

def _pane(script: str):
    harness = "\n".join([
        FAKE_DOM,
        "var workspace = make('div', {}); workspace.className = 'chat-workspace';",
        "var pane = make('aside', {'id': 'artifactPane', 'aria-hidden': 'true', 'inert': ''}, workspace);",
        "make('button', {'type': 'button'}, pane);",
        "var _activeArtifactPayload = {data: {result_id: 'r1'}};",
        "function _setArtifactUrl(id) {}",
        "window.dispatchEvent = function () {}; function Event(t) { this.type = t; }",
        "function setTimeout(f) {}",
        lift(CHAT, "function _artifactPaneOpen(open)"),
        lift(CHAT, "function closeArtifactPane()"),
        lift(CHAT, "function openArtifactPane()"),
        script,
        "JSON.stringify([pane.getAttribute('aria-hidden'), pane.hasAttribute('inert'), workspace.classList.contains('artifact-open')]);",
    ])
    return json.loads(dukpy.evaljs(harness))


class TestTheClosedPaneIsOutOfReach:

    def test_the_page_starts_with_it_inert(self):
        tag = re.search(r'<aside class="artifact-pane" id="artifactPane"[^>]*>', CHAT).group(0)
        assert 'aria-hidden="true"' in tag and re.search(r"\sinert(?:\s|>|=)", tag)

    def test_opening_it_lets_the_keyboard_in(self):
        assert _pane("openArtifactPane();") == ["false", False, True]

    def test_closing_it_takes_its_controls_out_of_the_tab_order(self):
        assert _pane("openArtifactPane(); closeArtifactPane();") == ["true", True, False]


# ── The chat page's landmarks ────────────────────────────────────────────────

class _Landmarks(HTMLParser):
    IMPLICIT = {"nav": "navigation", "aside": "complementary", "main": "main", "header": "banner"}

    def __init__(self, html: str):
        super().__init__()
        self.found: list[tuple[str, str, tuple[str, ...]]] = []
        self._stack: list[tuple[str, str | None]] = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        if tag in ("input", "img", "br", "meta", "link", "hr", "source", "use", "path"):
            return
        a = dict(attrs)
        role = a.get("role") or self.IMPLICIT.get(tag)
        # A <header> inside sectioning content or main is that section's
        # header, not the page's banner.
        if tag == "header" and not a.get("role") and any(
                t in ("article", "aside", "main", "nav", "section") for t, _ in self._stack):
            role = None
        if tag == "form" and a.get("aria-label"):
            role = "form"
        landmark = role if role in ("navigation", "complementary", "main", "banner", "form", "region", "search") else None
        if landmark:
            outer = tuple(r for _, r in self._stack if r)
            self.found.append((landmark, a.get("aria-label", ""), outer))
        self._stack.append((tag, landmark))

    def handle_endtag(self, tag):
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i][0] == tag:
                del self._stack[i:]
                return


USER = {"id": 1, "name": "Ada Lovelace", "account_id": "a", "role": "analyst", "group_name": "Analysts"}


def test_the_chat_page_has_no_two_landmarks_a_reader_cannot_tell_apart():
    marks = _Landmarks(render("portal_chat.html", path="/portal/chat", user=USER, enabled=True)).found
    named = [(role, label) for role, label, _ in marks if role != "main"]
    assert len(named) == len(set(named)), named


def test_no_complementary_region_sits_inside_another_landmark():
    marks = _Landmarks(render("portal_chat.html", path="/portal/chat", user=USER, enabled=True)).found
    nested = [(role, label, outer) for role, label, outer in marks
              if role == "complementary" and any(o in ("complementary", "main") for o in outer)
              and label != _artifact_label()]
    assert not nested, nested


def _artifact_label() -> str:
    from core import i18n

    return i18n.t("ui.chat.artifact_label", lang="en")


# ── Where the reader is ──────────────────────────────────────────────────────
# The navigation marked the current page with a colour (class "active") and
# nothing a screen reader announces. Now the page is aria-current="page", and
# a section or sidebar entry that contains it is aria-current="true" -- one
# "current page", never three.

class _Current(HTMLParser):
    def __init__(self, html: str):
        super().__init__()
        self.marked: list[tuple[str, str]] = []
        self._open: dict | None = None
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "a" and a.get("aria-current"):
            self._open = {"current": a["aria-current"], "text": ""}

    def handle_data(self, data):
        if self._open is not None:
            self._open["text"] += data

    def handle_endtag(self, tag):
        if tag == "a" and self._open is not None:
            self.marked.append((" ".join(self._open["text"].split()), self._open["current"]))
            self._open = None


def _admin_page(path: str, **context) -> str:
    from unittest.mock import MagicMock

    import admin.routes as routes

    request = MagicMock()
    request.url.path = path
    request.query_params = {}
    return routes.templates.env.get_template("base.html").render(request=request, **context)


def _is(marked, text: str, current: str) -> bool:
    """A link whose visible text starts with `text` (a count badge may follow)."""
    return any(t.startswith(text) and c == current for t, c in marked)


def test_the_admin_list_page_is_the_current_page():
    marked = _Current(_admin_page("/admin/clients")).marked
    assert len(marked) == 1 and _is(marked, "Clients", "page"), marked


def test_a_workspace_page_is_the_current_page_inside_its_section():
    client = {"account_id": "acct", "client_name": "Acme", "state": "READY"}
    marked = _Current(_admin_page("/admin/clients/acct/date-roles", client=client)).marked
    assert ("Dates", "page") in marked
    assert _is(marked, "Data & Model", "true") and _is(marked, "Clients", "true"), marked
    assert [m for m in marked if m[1] == "page"] == [("Dates", "page")]


def test_a_workspace_section_without_tools_is_itself_the_page():
    client = {"account_id": "acct", "client_name": "Acme", "state": "READY"}
    marked = _Current(_admin_page("/admin/clients/acct/compliance", client=client)).marked
    assert [m for m in marked if m[1] == "page"] == [("Compliance", "page")]


def test_the_portal_marks_the_page_the_reader_is_on():
    html = render("portal_notifications.html", path="/portal/notifications", user=USER,
                  alerts=[], reports=[], subscriptions={}, saved=None, error=None)
    pages = [m for m in _Current(html).marked if m[1] == "page"]
    assert len(pages) == 1 and pages[0][0].startswith("Notifications"), pages
