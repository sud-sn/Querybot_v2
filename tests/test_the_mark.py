"""The mark: one drawing, one per screen, and it moves only while an answer is worked out.

The chat screen wore the logo four times -- the sidebar, a tile beside every
answer, a mascot inside the question box, and a separate "thinking" mark -- and
the mark was two drawings: a gradient tile for pages and a two-bar variant for
the browser tab. It tilted on hover, popped in on the sign-in pages, and turned
red and shook on an error.

Now (the logo step, L1, redrawn for the Solviz name):

- One drawing: two corner brackets (the frame a question is asked in) and a dot
  at the centre (the answer found in it), on a 32-unit grid.
  static/img/logo-mark.svg is the favicon and any <img>; the brand_motion macro
  draws the same geometry inline. Flat: the brackets in one colour, the dot in
  turmeric, no tile and no gradient.
- One per screen: the sidebar lockup. No mark beside answers, none in the
  question box. The working beat beside a pending answer is the one place it
  moves, and it goes when the answer arrives.
- Nothing else moves: no hover tilt, no intro, no colour or shake on error --
  a failure is said in words. Under reduced motion nothing moves at all.
- The wordmark, "solvız", is one macro in both consoles; its i carries the
  mark's dot, and a screen reader hears "Solviz".

These read the files and render the real templates.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest


ROOT = Path(__file__).resolve().parents[1]
MARK = ROOT / "static" / "img" / "logo-mark.svg"
CSS = ROOT / "static" / "css" / "brand-motion.css"
MACROS = (ROOT / "portal" / "templates" / "macros.html", ROOT / "admin" / "templates" / "macros.html")
WORKING = ("loading", "thinking", "resolving", "querying", "answering")


def _macro(path: Path, name: str = "brand_motion") -> str:
    source = path.read_text(encoding="utf-8")
    return source.split("{% macro " + name, 1)[1].split("{%- endmacro %}", 1)[0]


def _brackets(svg: str) -> list[str]:
    return re.findall(r'<path[^>]*\bd="([^"]+)"', svg)


def _dot(svg: str) -> tuple[float, float, float]:
    found = re.search(r'<circle[^>]*\bcx="([\d.]+)"\s+cy="([\d.]+)"\s+r="([\d.]+)"', svg)
    return tuple(float(v) for v in found.groups())


def _stroke(svg: str) -> float:
    return float(re.search(r'stroke-width="([\d.]+)"', svg).group(1))


def _css() -> str:
    return re.sub(r"/\*[\s\S]*?\*/", "", CSS.read_text(encoding="utf-8"))


def _rules(css: str) -> list[tuple[str, str]]:
    """(selector, body) for every rule outside @keyframes and @media."""
    css = re.sub(r"@keyframes[^{]+\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}", "", css)
    css = re.sub(r"@media[^{]+\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}", "", css)
    return [(s.strip(), b) for s, b in re.findall(r"([^{}]+)\{([^{}]*)\}", css)]


# ── one drawing ─────────────────────────────────────────────────────────────


class TestOneDrawing:

    def test_the_file_and_the_component_draw_the_same_mark(self):
        svg = MARK.read_text(encoding="utf-8")
        for macro in MACROS:
            drawn = _macro(macro)
            assert _brackets(drawn) == _brackets(svg), macro
            assert _dot(drawn) == _dot(svg), macro
            assert _stroke(drawn) == _stroke(svg), macro
            assert 'viewBox="0 0 32 32"' in drawn and 'viewBox="0 0 32 32"' in svg

    def test_both_consoles_carry_the_same_component(self):
        for name in ("brand_motion", "wordmark"):
            portal, admin = (re.sub(r"\s+", " ", _macro(m, name)) for m in MACROS)
            assert portal == admin, name

    def test_two_brackets_frame_one_dot_at_the_centre(self):
        svg = MARK.read_text(encoding="utf-8")
        first, second = _brackets(svg)
        assert _dot(svg)[:2] == (16.0, 16.0)
        # Each bracket is the other turned half way round the centre.
        (x1, y1), (x2, y2) = (tuple(float(v) for v in re.match(r"M([\d.]+) ([\d.]+)", d).groups())
                              for d in (first, second))
        assert (x1 + x2, y1 + y2) == (32.0, 32.0)

    def test_the_mark_holds_at_favicon_size(self):
        svg = MARK.read_text(encoding="utf-8")
        px = 16 / 32
        assert _stroke(svg) * px >= 1.5, "a bracket under 1.5px blurs away"
        cx, cy, r = _dot(svg)
        assert 2 * r * px >= 3, "and so does a dot under 3px"
        # The nearest bracket end stays clear of the dot.
        ends = [(14, 6.5), (6.5, 14), (18, 25.5), (25.5, 18)]
        clear = min(math.dist((cx, cy), end) for end in ends) - r - _stroke(svg) / 2
        assert clear * px >= 1, "the dot runs into a bracket"

    def test_the_mark_is_flat(self):
        for svg in [MARK.read_text(encoding="utf-8")] + [_macro(m) for m in MACROS]:
            assert "linearGradient" not in svg and "<rect" not in svg, "no tile, no gradient"

    def test_there_is_no_second_small_drawing(self):
        assert not (ROOT / "static" / "img" / "logo-mark-sm.svg").exists()


class TestTheBrowserTab:

    def test_the_favicon_turns_light_in_a_dark_browser(self):
        svg = MARK.read_text(encoding="utf-8")
        light, dark = svg.split("prefers-color-scheme: dark", 1)
        for part, prop in (("qb-glyph", "stroke"), ("qb-dot", "fill")):
            pattern = r"\." + part + r"\s*\{\s*" + prop + r":\s*(#[0-9A-Fa-f]{6})"
            assert re.search(pattern, light).group(1).upper() != re.search(pattern, dark).group(1).upper(), part

    def test_it_never_moves(self):
        svg = MARK.read_text(encoding="utf-8")
        assert "animation" not in svg and "<animate" not in svg

    def test_it_is_named(self):
        assert 'aria-label="Solviz"' in MARK.read_text(encoding="utf-8")

    @pytest.mark.parametrize("page", ["portal/templates/portal_base.html", "admin/templates/base.html"])
    def test_both_consoles_use_it(self, page):
        head = (ROOT / page).read_text(encoding="utf-8").split("</head>", 1)[0]
        assert re.search(r'rel="icon"[^>]*href="\{\{ asset\(\'img/logo-mark\.svg\'\) \}\}"', head), page


# ── it moves only while an answer is worked out ─────────────────────────────


class TestOnlyTheWorkingWaveMoves:

    def test_every_animation_belongs_to_a_working_state(self):
        for selector, body in _rules(_css()):
            if "animation" in body and "animation: none" not in body:
                for part in selector.split(","):
                    assert any(f'[data-state="{s}"]' in part for s in WORKING), part

    def test_nothing_answers_the_pointer(self):
        assert ":hover" not in _css() and ":active" not in _css()

    def test_an_error_is_not_said_by_the_mark(self):
        css = _css()
        assert '[data-state="error"]' not in css and '[data-state="success"]' not in css
        assert "shake" not in css and "--danger" not in css

    def test_nothing_plays_on_arrival(self):
        assert "--intro" not in _css()
        for page in ("portal/templates/portal_login.html", "admin/templates/login.html"):
            assert "qb-brand-motion--intro" not in (ROOT / page).read_text(encoding="utf-8"), page

    def test_reduced_motion_stops_everything(self):
        reduced = _css().split("prefers-reduced-motion: reduce", 1)[1]
        assert "animation: none !important" in reduced
        assert "transition: none !important" in reduced


# ── one per screen ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def chat_page() -> str:
    from tests.chat_render import render as render_chat

    return render_chat()


class TestOnePerScreen:

    def test_no_answer_carries_the_mark(self):
        source = (ROOT / "portal" / "templates" / "portal_chat.html").read_text(encoding="utf-8")
        assert "QB_MARK_SRC" not in source
        assert "QB_MARK_SRC" not in (ROOT / "portal" / "templates" / "portal_base.html").read_text(encoding="utf-8")

    def test_the_question_box_carries_no_mark(self, chat_page):
        composer = chat_page.split('class="chat-composer"', 1)[1].split("</textarea>", 1)[0]
        assert "qb-brand-motion" not in composer

    def test_the_chat_screen_shows_the_mark_once(self, chat_page):
        # The sidebar lockup, the phone bar's (one of the two is hidden at any
        # width), and the hidden template the working wave is cloned from.
        shown = re.findall(r'class="qb-brand-motion qb-brand-motion--\w+', chat_page)
        assert len(shown) == 3, shown
        assert 'id="processingBrand"' in chat_page

    def test_the_marks_on_a_page_carry_no_ids_to_clash(self, chat_page):
        for svg in re.findall(r'<svg class="qb-brand-motion__mark"[\s\S]*?</svg>', chat_page):
            assert " id=" not in svg

    def test_the_sidebar_mark_takes_the_sidebar_accent(self, chat_page):
        assert re.search(r'<div\s+class="qb-brand-motion qb-brand-motion--sm portal-brand-mark"', chat_page)
        css = (ROOT / "static" / "css" / "portal.css").read_text(encoding="utf-8")
        assert re.search(r"\.portal-sidebar[^{]*\.portal-brand-mark[^{]*\{[^}]*--qb-mark-glyph:\s*var\(--sidebar-accent\)"
                         r"[^}]*--qb-mark-dot:\s*var\(--brand-dot\)", css)


def test_the_admin_sidebar_carries_the_same_lockup_with_a_tag():
    import admin.routes as routes

    request = MagicMock()
    request.url.path = "/admin/"
    html = routes.templates.env.get_template("base.html").render(request=request)
    brand = html.split('class="sidebar-brand"', 1)[1].split("</nav>", 1)[0].split('<nav', 1)[0]
    assert "qb-brand-motion__mark" in brand
    assert 'class="qb-wordmark sidebar-brand-text" role="img" aria-label="Solviz"' in brand
    assert re.search(r">\s*Admin\s*<", brand)
    assert "<img" not in brand
    # In the sidebar the mark takes the sidebar's colours. Two classes: the mark's own
    # stylesheet sets the light-ground colours and loads after the admin one.
    css = (ROOT / "static" / "css" / "admin.css").read_text(encoding="utf-8")
    assert re.search(r"\.sidebar-brand \.sidebar-brand-mark\s*\{[^}]*--qb-mark-glyph:\s*var\(--sidebar-accent\)", css)


# ── the wordmark ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("page", [
    "portal/templates/portal_base.html", "portal/templates/portal_login.html",
    "portal/templates/portal_unsubscribe.html", "admin/templates/base.html", "admin/templates/login.html",
    "admin/templates/admin_invite.html", "admin/templates/admin_account_setup.html"])
def test_every_lockup_spells_the_name_with_the_wordmark(page):
    source = (ROOT / page).read_text(encoding="utf-8")
    assert "wordmark(" in source and "Query<span>Bot</span>" not in source, page


def test_the_wordmark_is_heard_as_one_word_and_its_dot_is_the_marks():
    drawn = _macro(MACROS[0], "wordmark")
    assert 'role="img" aria-label="Solviz"' in drawn
    assert 'solv<span class="qb-wordmark__i">\u0131</span>z' in drawn    # a dotless i, given the mark's dot
    css = _css()
    dot = re.search(r"\.qb-wordmark__i::after\s*\{([^}]*)\}", css).group(1)
    assert "var(--qb-mark-dot" in dot and "border-radius: 50%" in dot


@pytest.mark.parametrize("page", ["admin/templates/client_setup.html", "admin/templates/databases.html"])
def test_a_setup_wait_says_its_step_without_the_mark(page):
    source = (ROOT / page).read_text(encoding="utf-8")
    assert "brand_motion(" not in source and "qb-brand-loader-block" not in source
