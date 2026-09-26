"""
The quiet states and the icon-only controls, done one way.

The access-request and learning queues said "the queue is clear" in a
grey line of their own, while every other empty list used the shared empty
state (an icon, a title, a sentence, an action). The icon-only buttons on
the users and clients pages -- reset password, deactivate, delete, KB setup --
had a tooltip drawn in CSS and no accessible name, so a screen reader read
"button". And beyond the three sign-in pages, nine more "show" buttons on the
users and system pages ran a toggleReveal of their own from inline handlers.

These render the two queues empty, read every template for icon-only
controls without a name, and run the shared reveal for a secret that is not
a password.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock

import dukpy
import pytest

from tests.js_fakedom import FAKE_DOM

ROOT = Path(__file__).resolve().parents[1]


def _render(template: str, **ctx) -> str:
    import admin.routes as routes

    request = MagicMock()
    request.url.path = "/admin/clients/acct/x"
    return routes.templates.env.get_template(template).render(
        request=request, client={"account_id": "acct", "client_name": "Acme"}, saved=None, **ctx)


def _empty_state(html: str) -> tuple[str, str]:
    """The shared empty state's title and icon name."""
    start = html.find('<div class="empty-state">')
    assert start >= 0, "no shared empty state on the page"
    block = html[start:html.index("</h3>", start)]
    title = re.search(r'<h3 class="empty-state-title">([^<]*)', block).group(1)
    icon = re.search(r'#([a-z-]+)"', block).group(1)
    return title, icon


class _Counts(dict):
    """A stats mapping the page reads by name: a count it was not given is 0."""

    def __missing__(self, key):
        return 0


class TestAnEmptyQueueSaysSoTheSharedWay:

    @pytest.mark.parametrize("status,title", [("pending", "The queue is clear"),
                                              ("approved", "No approved platform users yet"),
                                              ("rejected", "No rejected requests")])
    def test_access_requests(self, status, title):
        html = _render("client_pending_users.html", pending_users=[], groups=[], status_filter=status,
                       counts=_Counts(), pending_count=0)
        got_title, icon = _empty_state(html)
        assert got_title == title
        assert icon in re.findall(r'<symbol id="([a-z-]+)"',
                                  (ROOT / "static" / "icons" / "qb-icons.svg").read_text(encoding="utf-8"))

    @pytest.mark.parametrize("status,title", [("pending_review", "The queue is clear"),
                                              ("rejected", "Nothing matches this filter")])
    def test_learning_candidates(self, status, title):
        stats = _Counts()
        html = _render("client_learning_queue.html", candidates=[], status_filter=status, stats=stats,
                       flag_enabled=True, governed_count=0)
        assert _empty_state(html)[0] == title


def _templates():
    return list((ROOT / "admin" / "templates").rglob("*.html")) + list((ROOT / "portal" / "templates").rglob("*.html"))


def _controls(text: str):
    """Every <button> and <a> in a template, as (opening tag, inner markup)."""
    for m in re.finditer(r"<(button|a)\b([^>]*)>([\s\S]*?)</\1>", text):
        yield m.group(2), m.group(3)


def _visible_text(inner: str) -> str:
    inner = re.sub(r"\{\{\s*ic\([^}]*\}\}", "", inner)          # an icon from the sprite
    inner = re.sub(r"<svg[\s\S]*?</svg>", "", inner)
    inner = re.sub(r"<[^>]+>", "", inner)
    inner = re.sub(r"\{#[\s\S]*?#\}", "", inner)
    return inner.strip()


def test_every_tooltip_control_has_a_name():
    unnamed = []
    for page in _templates():
        for attrs, inner in _controls(page.read_text(encoding="utf-8")):
            if "qb-tip" in attrs and not _visible_text(inner) and "aria-label=" not in attrs:
                unnamed.append(f"{page.name}: {attrs.strip()[:80]}")
    assert not unnamed, unnamed


def test_every_icon_button_has_a_name():
    unnamed = []
    for page in _templates():
        for attrs, inner in _controls(page.read_text(encoding="utf-8")):
            classes = re.search(r'class="([^"]*)"', attrs)
            if classes and "btn-icon" in classes.group(1).split() and not _visible_text(inner) \
                    and "aria-label=" not in attrs:
                unnamed.append(f"{page.name}: {attrs.strip()[:80]}")
    assert not unnamed, unnamed


def test_every_show_button_is_the_shared_one_for_a_field_on_its_page():
    for page in _templates():
        text = page.read_text(encoding="utf-8")
        assert "toggleReveal" not in text, page.name
        ids = set(re.findall(r'\bid="([^"]+)"', text))
        for attrs, _ in _controls(text):
            if "input-reveal-btn" in attrs:
                names = re.search(r'data-qb-reveal="([^"]+)"', attrs)
                assert names, f"{page.name}: {attrs.strip()[:80]}"
                assert set(names.group(1).split()) <= ids, (page.name, names.group(1))
                assert "onclick" not in attrs, page.name


def test_a_show_button_can_say_what_it_shows():
    ui = (ROOT / "static" / "js" / "qb-ui.js").read_text(encoding="utf-8")
    out = json.loads(dukpy.evaljs(FAKE_DOM + ui + """
      var key = make('input', {'id': 'api_key'}); key.type = 'password';
      var btn = make('button', {'data-qb-reveal': 'api_key', 'data-label-show': 'Show key', 'data-label-hide': 'Hide key'});
      function press() { (docListeners.click || []).forEach(function (f) { f({target: btn}); }); }
      press(); var shown = [key.type, btn.getAttribute('aria-label')];
      press(); JSON.stringify([shown, [key.type, btn.getAttribute('aria-label')]]);
    """))
    assert out == [["text", "Hide key"], ["password", "Show key"]]
