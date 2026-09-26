"""
Every field says what it is for, and every column heading says something.

Two hundred labels across the templates sat beside their field with no for/id
-- <label>Storage type</label><select name="date_key_type"> -- so a screen
reader announced "combo box" and nothing else, and clicking the label did
nothing. Fifty-two more fields had no label at all, only a placeholder, which
disappears as soon as someone types: every search box, the chat composer, the
glossary and report edit rows, the setup page's descriptions, the users page's
per-row group picker. Three tables had an empty heading over their actions
column, and a link inside a sentence on the setup page differed from the text
around it by colour alone (1.06:1).

Now static/js/qb-ui.js ties each label that neither points at nor wraps a
field to the field that follows it, fields with no visible label carry an
aria-label (translated in the portal), and the action columns are named for a
screen reader. These run the real script against a stand-in DOM, render the
portal in French, and read every template for a field or heading with no name.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import dukpy
import pytest

from tests.js_fakedom import FAKE_DOM
from tests.portal_render import render

ROOT = Path(__file__).resolve().parents[1]
UI = (ROOT / "static" / "js" / "qb-ui.js").read_text(encoding="utf-8")


def _run(scenario: str):
    return json.loads(dukpy.evaljs(FAKE_DOM + scenario + UI + "\nJSON.stringify(out());"))


class TestALabelNamesTheFieldBesideIt:

    def test_a_label_before_a_select_is_tied_to_it_when_the_page_loads(self):
        got = _run("""
          var group = make('div', {'class': 'form-group'});
          var lab = make('label', {}, group); lab.textContent = 'Storage type';
          var sel = make('select', {'name': 'date_key_type'}, group);
          function out() { return [lab.getAttribute('for'), sel.id]; }
        """)
        assert got[0] and got[0] == got[1]

    def test_a_field_inside_the_wrapper_after_the_label_is_the_one_named(self):
        got = _run("""
          var lab = make('label', {}); lab.textContent = 'Password';
          var wrap = make('div', {'class': 'input-reveal-wrap'});
          var pw = make('input', {'type': 'password', 'name': 'pw'}, wrap);
          make('button', {'type': 'button'}, wrap);
          function out() { return [lab.getAttribute('for'), pw.id]; }
        """)
        assert got[0] and got[0] == got[1]

    def test_a_field_keeps_the_id_it_has(self):
        got = _run("""
          var lab = make('label', {}); var input = make('input', {'id': 'db_host', 'type': 'text'});
          function out() { return [lab.getAttribute('for'), input.id]; }
        """)
        assert got == ["db_host", "db_host"]

    def test_two_fields_get_two_ids(self):
        got = _run("""
          var a = make('label', {}); var x = make('input', {'type': 'text'});
          var b = make('label', {}); var y = make('input', {'type': 'text'});
          function out() { return [a.getAttribute('for'), b.getAttribute('for'), x.id, y.id]; }
        """)
        assert got[0] == got[2] and got[1] == got[3] and got[2] != got[3]

    @pytest.mark.parametrize("shape,build", [
        ("a label that wraps its field", """
          var lab = make('label', {}); var inner = make('input', {'type': 'checkbox'}, lab);
          var after = make('input', {'type': 'text'});"""),
        ("a label that already points somewhere", """
          var lab = make('label', {'for': 'elsewhere'}); var after = make('input', {'type': 'text'});"""),
        ("a label before a hidden field", """
          var lab = make('label', {}); var after = make('input', {'type': 'hidden', 'name': 'k'});"""),
        ("a label before something that is not a field", """
          var lab = make('label', {}); var after = make('p', {});"""),
    ])
    def test_a_label_is_left_alone_when_it_is_not_a_field_s_missing_name(self, shape, build):
        got = _run(build + """
          var before = lab.getAttribute('for');
          function out() { return [before, lab.getAttribute('for'), after.id || null]; }
        """)
        assert got[1] == got[0] and got[2] is None, shape

    def test_the_searchable_select_is_named_by_the_tied_label(self):
        got = _run("""
          var lab = make('label', {}); lab.textContent = 'Metric';
          var sel = make('select', {'name': 'metric_id', 'data-qb-select': ''});
          sel.appendChild(new OptionEl());
          function out() {
            var input = document.body.find(function (e) { return e.getAttribute('role') === 'combobox'; });
            return [lab.getAttribute('for'), input && input.id, input && input.getAttribute('aria-labelledby'), lab.id];
          }
        """)
        assert got[0] == got[1]              # clicking the label reaches what is typed in
        assert got[2] and got[2] == got[3]   # and a screen reader reads the label


# ── Every template ───────────────────────────────────────────────────────────

def _templates():
    return sorted(list((ROOT / "admin" / "templates").rglob("*.html")) +
                  list((ROOT / "portal" / "templates").rglob("*.html")))


def _markup(page: Path) -> str:
    """The template's markup: scripts and comments blanked, offsets kept."""
    text = page.read_text(encoding="utf-8")
    for pattern in (r"<script[\s\S]*?</script>", r"\{#[\s\S]*?#\}"):
        text = re.sub(pattern, lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)
    return text


def _unnamed(page: Path) -> list[str]:
    s = _markup(page)
    pointed_at = set(re.findall(r'<label[^>]*\bfor="([^"]+)"', s))
    found = []
    for m in re.finditer(r"<(select|textarea|input)\b([^>]*)>", s):
        tag, attrs = m.group(1), m.group(2)
        kind = re.search(r'\btype="([^"]+)"', attrs)
        if tag == "input" and kind and kind.group(1) in ("hidden", "submit", "button", "reset", "image"):
            continue
        if re.search(r"(?:^|\s)hidden(?:\s|$|=)", attrs) or re.search(r"\baria-label(?:ledby)?=", attrs):
            continue
        own_id = re.search(r'\bid="([^"]+)"', attrs)
        if own_id and own_id.group(1) in pointed_at:
            continue
        before = s[:m.start()]
        opened, closed = before.rfind("<label"), before.rfind("</label>")
        if opened > closed:                                          # wrapped by its label
            continue
        if closed >= 0 and re.fullmatch(r"</label>\s*(?:<(?:div|span)\b[^>]*>\s*)?", before[closed:]):
            continue                                                 # qbTieLabels ties it
        found.append(f"{page.relative_to(ROOT)}:{s.count(chr(10), 0, m.start()) + 1} <{tag}>")
    return found


def test_no_template_has_a_field_without_a_name():
    unnamed = [hit for page in _templates() for hit in _unnamed(page)]
    assert not unnamed, unnamed


def test_no_table_has_a_column_heading_with_nothing_in_it():
    empty = []
    for page in _templates():
        for m in re.finditer(r"<th\b[^>]*>([\s\S]*?)</th>", _markup(page)):
            if not re.sub(r"<[^>]+>", "", m.group(1)).strip() and "aria-label=" not in m.group(1):
                empty.append(f"{page.name}: {m.group(0)[:60]}")
    assert not empty, empty


def test_a_link_inside_a_sentence_is_underlined():
    css = (ROOT / "static" / "css" / "base.css").read_text(encoding="utf-8")
    rule = re.search(r"([^{}]*\bp a:not\(\.btn\)[^{}]*)\{([^}]*)\}", css)
    assert rule and "text-decoration: underline" in rule.group(2)


# ── The portal names them in the reader's language ───────────────────────────

USER = {"id": 1, "name": "Ada Lovelace", "account_id": "a", "role": "analyst", "group_name": "Analysts"}


@pytest.mark.parametrize("lang,expected", [("en", "Your question"), ("fr", "Votre question")])
def test_the_chat_composer_is_named_in_the_reader_s_language(lang, expected):
    html = render("portal_chat.html", path="/portal/chat", lang=lang, user=USER, enabled=True)
    tag = next(t for t in re.findall(r"<textarea\b[^>]*>", html) if 'id="input"' in t)
    assert f'aria-label="{expected}"' in tag


@pytest.mark.parametrize("lang,expected", [("en", "How often"), ("fr", "Fréquence")])
def test_a_report_subscription_says_what_each_picker_sets(lang, expected):
    html = render("portal_notifications.html", path="/portal/notifications", lang=lang, user=USER,
                  alerts=[], reports=[{"id": 7, "name": "Weekly sales", "description": "", "metrics": []}],
                  subscriptions={}, saved=None, error=None)
    tag = next(t for t in re.findall(r"<select\b[^>]*>", html) if 'name="cadence"' in t)
    assert f'aria-label="{expected}"' in tag
