"""The mark: one drawing, one per screen, and it moves only while an answer is worked out.

The chat screen wore the logo four times -- the sidebar, a tile beside every
answer, a mascot inside the question box, and a separate "thinking" mark -- and
the mark was two drawings: a gradient tile for pages and a two-bar variant for
the browser tab. It tilted on hover, popped in on the sign-in pages, and turned
red and shook on an error.

Now (the approved logo step, L1):

- One drawing, a speech bubble with three rising bars cut out of it, on the
  24-unit icon grid. static/img/logo-mark.svg is the favicon and any <img>; the
  brand_motion macro draws the same geometry inline. The bars are holes, cut
  with a mask, so they show whatever the mark sits on.
- One per screen: the sidebar lockup. No mark beside answers, none in the
  question box. The working wave beside a pending answer is the one place it
  moves, and it goes when the answer arrives.
- Nothing else moves: no hover tilt, no intro, no colour or shake on error --
  a failure is said in words. Under reduced motion nothing moves at all.

These read the files and render the real templates.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest


ROOT = Path(__file__).resolve().parents[1]
MARK = ROOT / "static" / "img" / "logo-mark.svg"
CSS = ROOT / "static" / "css" / "brand-motion.css"
MACROS = (ROOT / "portal" / "templates" / "macros.html", ROOT / "admin" / "templates" / "macros.html")
WORKING = ("loading", "thinking", "resolving", "querying", "answering")


def _macro(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    return source.split("{% macro brand_motion", 1)[1].split("{%- endmacro %}", 1)[0]


def _bubble(svg: str) -> str:
    return re.search(r'<path[^>]*\bd="([^"]+)"', svg).group(1)


def _bars(svg: str) -> list[tuple[float, float, float, float]]:
    found = re.findall(r'<rect[^>]*?x="([\d.]+)"\s+y="([\d.]+)"\s+width="([\d.]+)"\s+height="([\d.]+)"', svg)
    return [tuple(float(v) for v in bar) for bar in found]


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
            assert _bubble(drawn) == _bubble(svg), macro
            assert _bars(drawn) == _bars(svg), macro

    def test_both_consoles_carry_the_same_component(self):
        portal, admin = (re.sub(r"\s+", " ", _macro(m)) for m in MACROS)
        assert portal == admin

    def test_three_bars_rise_on_one_baseline(self):
        bars = _bars(MARK.read_text(encoding="utf-8"))
        assert len(bars) == 3
        assert [h for *_, h in bars] == sorted(h for *_, h in bars)
        assert len({round(y + h, 3) for _, y, _, h in bars}) == 1, "one baseline"

    def test_the_bars_stay_apart_at_favicon_size(self):
        bars = _bars(MARK.read_text(encoding="utf-8"))
        px = 16 / 24
        for (x1, _, w1, _), (x2, *_rest) in zip(bars, bars[1:]):
            assert w1 * px >= 1.5, "a bar under 1.5px blurs away"
            assert (x2 - x1 - w1) * px >= 0.9, "and so does the gap between two"

    def test_the_bars_are_cut_through_not_painted(self):
        # A painted bar has to match whatever the mark sits on; a hole cannot be wrong.
        for svg in [MARK.read_text(encoding="utf-8")] + [_macro(m) for m in MACROS]:
            assert "<mask" in svg and 'mask="url(#' in svg
            assert "linearGradient" not in svg, "the mark is one flat colour"

    def test_there_is_no_second_small_drawing(self):
        assert not (ROOT / "static" / "img" / "logo-mark-sm.svg").exists()


class TestTheBrowserTab:

    def test_the_favicon_turns_light_in_a_dark_browser(self):
        svg = MARK.read_text(encoding="utf-8")
        light = re.search(r"\.qb-glyph\s*\{\s*fill:\s*(#[0-9A-Fa-f]{6})", svg).group(1)
        dark = re.search(r"prefers-color-scheme:\s*dark\)\s*\{\s*\.qb-glyph\s*\{\s*fill:\s*(#[0-9A-Fa-f]{6})",
                         svg).group(1)
        assert light.upper() != dark.upper()

    def test_it_never_moves(self):
        svg = MARK.read_text(encoding="utf-8")
        assert "animation" not in svg and "<animate" not in svg

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

    def test_every_mark_on_a_page_cuts_with_its_own_mask(self, chat_page):
        ids = re.findall(r'<mask id="([^"]+)"', chat_page)
        assert ids and len(ids) == len(set(ids)), ids

    def test_the_sidebar_mark_takes_the_sidebar_accent(self, chat_page):
        assert re.search(r'<div\s+class="qb-brand-motion qb-brand-motion--sm portal-brand-mark"', chat_page)
        css = (ROOT / "static" / "css" / "portal.css").read_text(encoding="utf-8")
        assert re.search(r"\.portal-sidebar[^{]*\.portal-brand-mark[^{]*\{[^}]*--qb-mark-glyph:\s*var\(--sidebar-accent\)",
                         css)


def test_the_admin_sidebar_carries_the_same_lockup_with_a_tag():
    import admin.routes as routes

    request = MagicMock()
    request.url.path = "/admin/"
    html = routes.templates.env.get_template("base.html").render(request=request)
    brand = html.split('class="sidebar-brand"', 1)[1].split("</nav>", 1)[0].split('<nav', 1)[0]
    assert "qb-brand-motion__mark" in brand
    assert "Query<span>Bot</span>" in brand
    assert re.search(r">\s*Admin\s*<", brand)
    assert "<img" not in brand


@pytest.mark.parametrize("page", ["admin/templates/client_setup.html", "admin/templates/databases.html"])
def test_a_setup_wait_says_its_step_without_the_mark(page):
    source = (ROOT / page).read_text(encoding="utf-8")
    assert "brand_motion(" not in source and "qb-brand-loader-block" not in source
