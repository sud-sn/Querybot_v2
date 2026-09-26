"""
One field, for both consoles.

Fields were styled only inside a .form-group; everywhere else each page wrote
its own input, select and textarea: sixty rules across nineteen pages, with
borders on the decorative rung (1.3:1, a field you cannot find), and focus
rings in rgba(13,115,119) -- the retired teal -- on the review queues and the
portal's knowledge page. The searchable select's box (.form-input) had no rule
at all. Every textarea in a form was set in monospace. Three sign-in and
password pages each carried their own "show password" button, three ways.

Now base.css holds the one field: .form-input anywhere, and every field in a
.form-group, with hover, focus, invalid, disabled and read-only states, a
compact and a monospace size, a native select drawn with the icon set's
chevron, an icon-led search box, and one password reveal (qb-ui.js). These
read the stylesheet as data, compute its contrast from tokens.css, and run the
reveal in dukpy.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import unquote

import dukpy
import pytest

from tests.js_fakedom import FAKE_DOM

ROOT = Path(__file__).resolve().parents[1]
CSS = ROOT / "static" / "css"
UI_JS = ROOT / "static" / "js" / "qb-ui.js"


def _strip(css: str) -> str:
    return re.sub(r"/\*[\s\S]*?\*/", "", css)


def _tokens() -> dict[str, str]:
    css = _strip((CSS / "tokens.css").read_text(encoding="utf-8"))
    return {k: v.upper() for k, v in re.findall(r"(--[\w-]+):\s*(#[0-9a-fA-F]{6})\b", css)}


def _field_block() -> str:
    css = (CSS / "base.css").read_text(encoding="utf-8")
    start = css.index("/* ── Fields")
    return css[start:css.index(".form-grid {", start)]


def _split(selector: str) -> list[str]:
    """A selector list at its top-level commas (not those inside :not())."""
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(selector):
        depth += ch == "("
        depth -= ch == ")"
        if ch == "," and depth == 0:
            parts.append(selector[start:i])
            start = i + 1
    parts.append(selector[start:])
    return [" ".join(p.split()) for p in parts]


def _rules(block: str) -> list[tuple[list[str], str]]:
    return [(_split(selector), body) for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", _strip(block))]


def _lum(hexcolour: str) -> float:
    h = hexcolour.lstrip("#")
    channels = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    r, g, b = (c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _ratio(a: str, b: str) -> float:
    high, low = sorted((_lum(a), _lum(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


# The four ways a field is one: the class anywhere, or any field in a form group.
FORMS = {
    "class": ".form-input",
    "input": '.form-group input:not([type="checkbox"], [type="radio"], [type="hidden"], [type="file"], [type="range"], '
             '[type="color"])',
    "select": ".form-group select",
    "textarea": ".form-group textarea",
}


def _prop(state: str, form: str, prop: str) -> str:
    """The value a rule of the field block sets on `form` in `state`
    ('' = at rest), with var() read from tokens.css."""
    selector = FORMS[form] + state
    tokens = _tokens()
    for selectors, body in _rules(_field_block()):
        if selector in selectors:
            m = re.search(rf"(?:^|[;\s]){re.escape(prop)}\s*:\s*([^;]+);", body)
            if m:
                value = m.group(1).strip()
                ref = re.fullmatch(r"var\((--[\w-]+)\)", value)
                if ref and ref.group(1) in tokens:
                    return tokens[ref.group(1)]
                ref = re.match(r"1px solid var\((--[\w-]+)\)", value)
                return tokens[ref.group(1)] if ref else value
    raise AssertionError(f"no {prop} for {selector}")


class TestEveryFieldHasEveryState:

    @pytest.mark.parametrize("form", sorted(FORMS))
    def test_at_rest_its_text_and_its_edge_read(self, form):
        ground = _prop("", form, "background")
        assert _ratio(_prop("", form, "color"), ground) >= 4.5
        # The edge is how the reader finds the field: 3:1 against the
        # field and against the page it sits on.
        edge = _prop("", form, "border")
        assert _ratio(edge, ground) >= 3.0 and _ratio(edge, _tokens()["--paper"]) >= 3.0

    @pytest.mark.parametrize("form", sorted(FORMS))
    def test_hover_darkens_the_edge(self, form):
        rest, hover = _prop("", form, "border"), _prop(":hover", form, "border-color")
        assert hover != rest and _lum(hover) < _lum(rest)
        assert _ratio(hover, _prop("", form, "background")) >= 3.0

    @pytest.mark.parametrize("form", sorted(FORMS))
    def test_its_hint_text_reads(self, form):
        assert _ratio(_prop("::placeholder", form, "color"), _prop("", form, "background")) >= 4.5

    @pytest.mark.parametrize("form", sorted(FORMS))
    def test_focus_is_the_focus_ring_and_the_edge_turns_primary(self, form):
        assert _prop(":focus", form, "border-color") == _tokens()["--primary"]
        assert _prop(":focus", form, "box-shadow") == "var(--focus-ring)"

    @pytest.mark.parametrize("form", sorted(FORMS))
    @pytest.mark.parametrize("state", ['[aria-invalid="true"]', ":user-invalid"])
    def test_invalid_turns_the_edge_danger(self, form, state):
        edge = _prop(state, form, "border-color")
        assert edge == _tokens()["--danger"] and _ratio(edge, _prop("", form, "background")) >= 3.0

    def test_user_invalid_has_a_rule_of_its_own(self):
        # A browser that does not know :user-invalid drops the whole selector
        # list it appears in -- with aria-invalid in it, that too.
        for selectors, _ in _rules(_field_block()):
            if any(":user-invalid" in s for s in selectors):
                assert all(":user-invalid" in s for s in selectors), selectors

    @pytest.mark.parametrize("form", sorted(FORMS))
    def test_disabled_and_read_only_look_it(self, form):
        assert _prop(":disabled", form, "background") == _tokens()["--state-disabled"]
        assert _prop(":disabled", form, "cursor") == "not-allowed"
        assert _prop("[readonly]", form, "background") == _tokens()["--surface-alt"]

    def test_the_block_is_drawn_from_tokens(self):
        block = re.sub(r"url\(\"data:[^\"]*\"\)", "", _strip(_field_block()))
        literals = re.findall(r"#[0-9a-fA-F]{3,8}\b|\brgba?\(|\b(?:white|black)\b(?!-)", block)
        assert not literals, literals


def test_a_selects_chevron_is_the_icon_sets_in_the_muted_ink():
    block = _field_block()
    uri = re.search(r'url\("data:image/svg\+xml,([^"]*)"\)', block)
    assert uri, "no chevron"
    svg = unquote(uri.group(1))
    sprite = (ROOT / "static" / "icons" / "qb-icons.svg").read_text(encoding="utf-8")
    chevron = re.search(r'<symbol id="chevron-down"[^>]*>(.*?)</symbol>', sprite).group(1)
    assert re.findall(r"""\bd=["']([^"']+)""", chevron) == re.findall(r"""\bd=["']([^"']+)""", svg)
    stroke = re.search(r"stroke='(#[0-9A-Fa-f]{6})'", svg).group(1).upper()
    assert stroke == _tokens()["--text-muted"]


def test_a_textarea_is_set_in_the_ui_face_unless_asked():
    for selectors, body in _rules(_field_block()):
        if ".form-group textarea" in selectors:
            assert "font-mono" not in body, selectors
    mono = [body for selectors, body in _rules(_field_block()) if ".form-input.form-input-mono" in selectors]
    assert mono and "var(--font-mono)" in mono[0]


# ── Nothing else draws a field ─────────────────────────────────────────────

def _styles():
    for page in list((ROOT / "admin" / "templates").rglob("*.html")) + \
            list((ROOT / "portal" / "templates").rglob("*.html")):
        yield page, "\n".join(re.findall(r"<style[^>]*>([\s\S]*?)</style>", page.read_text(encoding="utf-8")))
    for sheet in CSS.glob("*.css"):
        if sheet.name != "base.css":
            yield sheet, sheet.read_text(encoding="utf-8")


_FIELD_SELECTOR = re.compile(r"(?:^|[\s>+~])(?:input|select|textarea)\b"
                             r"|\.[\w-]*?(?:^|-|\.)?(?:input|select|textarea|search)(?:-[\w-]*)?(?![\w-])")

# Fields that are their own control (a chat composer, a formula editor) or on
# a page a later phase redoes. None may be added.
SPECIAL_FIELDS = {
    "_styles.html": {".fn-search", ".mc-chat-composer input", ".mc-chat-composer input:focus", ".mc-fields-search",
                     ".metric-ai-import textarea", ".metric-builder input", ".metric-builder input:focus",
                     ".metric-builder select", ".metric-builder select:focus", ".metric-builder textarea",
                     ".metric-builder textarea:focus", ".metrics-toolbar input[type=text]",
                     ".metrics-toolbar input[type=text]:focus", ".metrics-toolbar select", ".sql-hl-wrap textarea",
                     ".time-col-select", ".time-col-select:focus"},                       # metrics, phase 6
    "chat_workspace.css": {".chat-workspace-main .chat-input-bar", ".clarification-input-row textarea",
                           ".clarification-input-row textarea:focus", ".portal-sidebar .hp-search-wrap",
                           ".portal-sidebar .hp-search-wrap:focus-within"},                 # chat composer, phase 6
    "portal_chat.html": {".chat-input", ".chat-input-bar", ".chat-input-bar:focus-within", ".chat-input-bar:hover",
                         ".chat-input:focus-visible", ".dashboard-picker-body input:focus",
                         ".dashboard-picker-body input:not([type=checkbox]):not([type=radio])",
                         ".dashboard-picker-body input[aria-invalid=true]", ".dashboard-picker-body select",
                         ".dashboard-picker-body select:focus", ".hp-search-clear", ".hp-search-input",
                         ".hp-search-wrap", ".rc-input", ".rc-input-row", ".rc-input:focus"},  # chat 6, picker 5
    "client_graph.html": {".jc-cond-row input", ".jc-cond-row input:focus", ".jc-select", ".jc-select:focus"},  # phase 4
    "client_setup.html": {"#tableDescriptions input[type=text]", "#tableDescriptions textarea"},  # phase 6
    "dashboard.css": {".dashboard-filter input"},                                              # dashboards, phase 5
}


def _field_rules():
    for source, css in _styles():
        for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", _strip(css)):
            if not re.search(r"(?:^|[;\s])(?:background(?:-color)?|border(?:-color)?|box-shadow)\s*:", " " + body):
                continue
            for one in _split(selector):
                if _FIELD_SELECTOR.search(one):
                    yield source.name, one


def test_no_page_draws_a_field_of_its_own():
    extra: dict[str, list[str]] = {}
    for name, selector in _field_rules():
        if selector not in SPECIAL_FIELDS.get(name, set()):
            extra.setdefault(name, []).append(selector)
    assert not extra, extra


def test_the_retired_teal_is_gone():
    found = []
    sources = [p for p in (ROOT / "admin" / "templates").rglob("*.html")] + \
        [p for p in (ROOT / "portal" / "templates").rglob("*.html")] + list(CSS.glob("*.css"))
    for source in sources:
        text = _strip(source.read_text(encoding="utf-8"))
        for hit in re.findall(r"rgba?\(\s*13\s*,\s*115\s*,\s*119|#0d7377\b", text, re.I):
            found.append(f"{source.name}: {hit}")
    assert not found, found


# ── Show a password ────────────────────────────────────────────────────────

_REVEAL = """
var a = make('input', {'id': 'pw-new'}); a.type = 'password';
var b = make('input', {'id': 'pw-confirm'}); b.type = 'password';
var btn = make('button', {'data-qb-reveal': 'pw-new pw-confirm', 'aria-pressed': 'false', 'aria-label': 'Show password'});
function press(target) {
  var ev = {target: target || btn, preventDefault: function () {}, stopPropagation: function () {}};
  (docListeners.click || []).forEach(function (f) { f(ev); });
}
function state() {
  return [a.type, b.type, btn.getAttribute('aria-pressed'), btn.getAttribute('aria-label'), btn.innerHTML];
}
"""


def _reveal(script: str, *, french: bool = False) -> object:
    catalogue = ""
    if french:
        catalogue = ("window.qbT = function (k) { return ({'ui.auth.show_password': 'Afficher le mot de passe', "
                     "'ui.auth.hide_password': 'Masquer le mot de passe'})[k] || k; };\n")
    return json.loads(dukpy.evaljs(FAKE_DOM + catalogue + UI_JS.read_text(encoding="utf-8") + "\n" + _REVEAL + script))


class TestShowAPassword:

    def test_it_shows_every_field_it_names_and_says_so(self):
        out = _reveal("press(); JSON.stringify(state());")
        assert out == ["text", "text", "true", "Hide password", '<svg data-icon="eye-off"></svg>']

    def test_pressing_again_hides_them(self):
        out = _reveal("press(); press(); JSON.stringify(state());")
        assert out == ["password", "password", "false", "Show password", '<svg data-icon="eye"></svg>']

    def test_it_speaks_the_readers_language(self):
        out = _reveal("press(); JSON.stringify(state()[3]);", french=True)
        assert out == "Masquer le mot de passe"

    def test_a_click_elsewhere_changes_nothing(self):
        out = _reveal("press(make('button', {})); JSON.stringify(state().slice(0, 3));")
        assert out == ["password", "password", "false"]

    def test_a_button_naming_no_field_on_the_page_changes_nothing(self):
        out = _reveal("btn.setAttribute('data-qb-reveal', 'nope'); press(); JSON.stringify(state().slice(0, 3));")
        assert out == ["password", "password", "false"]

    @pytest.mark.parametrize("page", ["admin/templates/login.html", "portal/templates/portal_login.html",
                                      "portal/templates/portal_change_password.html"])
    def test_each_password_page_uses_it_for_fields_it_has(self, page):
        text = (ROOT / page).read_text(encoding="utf-8")
        buttons = re.findall(r'data-qb-reveal="([^"]+)"', text)
        assert buttons, page
        ids = set(re.findall(r'\bid="([^"]+)"', text))
        for names in buttons:
            assert set(names.split()) <= ids, (page, names)
        # and keeps no reveal of its own
        assert ".type = " not in text and "input-reveal-btn{" not in text.replace(" ", "")
