"""The mark: one drawing, one per screen, and it moves only to say what the product is doing.

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
- It moves to say what the product is doing, and for nothing else (the motion
  step, M1): beside a pending answer each stage has its own movement (the frame
  breathes while the question is read, turns while the measures are found, the
  dot searches the frame while the query runs, the frame closes on it while the
  answer is drawn); the sidebar's and the sign-in page's marks draw themselves in
  once a visit; over the brand the frame opens and the dot hops. No colour or
  shake on error -- a failure is said in words. Under reduced motion nothing
  moves at all, and the favicon never does.
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


# ── it moves only to say what the product is doing ──────────────────────────


INTRO = ".qb-brand-motion--intro"
HOVER = '[data-brand-hover]:hover .qb-brand-motion[data-state="idle"]'


def _animation_names(css: str, state: str) -> set[str]:
    names = set()
    for selector, body in _rules(css):
        if f'[data-state="{state}"]' in selector and ":hover" not in selector:
            found = re.search(r"animation:\s*([\w-]+)", body)
            if found:
                names.add(found.group(1))
    return names


class TestTheMotionSaysWhatIsHappening:

    def test_every_animation_is_a_working_stage_the_arrival_or_the_pointer(self):
        for selector, body in _rules(_css()):
            if "animation" in body and "animation: none" not in body:
                for part in selector.split(","):
                    assert (any(f'[data-state="{s}"]' in part for s in WORKING) or part.strip().startswith(INTRO)
                            or part.strip().startswith(HOVER)), part

    def test_each_working_stage_moves_its_own_way(self):
        css = _css()
        moves = {state: frozenset(_animation_names(css, state)) for state in WORKING}
        for state, names in moves.items():
            assert names, f"{state} does not move"
        assert len(set(moves.values())) == len(WORKING), moves

    def test_the_chat_names_a_movement_for_every_stage_the_server_reports(self):
        source = (ROOT / "portal" / "templates" / "portal_chat.html").read_text(encoding="utf-8")
        table = source.split("const BRAND_STAGE_STATES = {", 1)[1].split("};", 1)[0]
        named = dict(re.findall(r"(\w+):\s*'(\w+)'", table))
        assert set(named.values()) <= set(WORKING) | {"idle"}, named
        pipeline = (ROOT / "core" / "query_pipeline.py").read_text(encoding="utf-8")
        reported = set(re.findall(r'_send_live_stage\(\s*\w+,\s*\w+,\s*"(\w+)"', pipeline))
        assert {"executing_query", "formatting_results"} <= reported, "the stages are still read from the pipeline"
        assert reported <= set(named), reported - set(named)

    def test_only_the_brand_answers_the_pointer_and_only_when_still(self):
        css = _css()
        for selector, _ in _rules(css):
            if ":hover" in selector:
                for part in selector.split(","):
                    assert part.strip().startswith(HOVER), part
        assert ":active" not in css and ":focus" not in css
        for page in ("portal/templates/portal_base.html", "admin/templates/base.html"):
            source = (ROOT / page).read_text(encoding="utf-8")
            assert "data-brand-hover" in source, page

    def test_an_error_is_not_said_by_the_mark(self):
        css = _css()
        assert '[data-state="error"]' not in css and '[data-state="success"]' not in css
        assert "shake" not in css and "--danger" not in css

    @pytest.mark.parametrize("page", ["portal/templates/portal_base.html", "admin/templates/base.html"])
    def test_the_sidebar_draws_itself_in_once_a_visit(self, page):
        source = (ROOT / page).read_text(encoding="utf-8")
        script = source[source.index("The brand draws itself in once a visit"):]
        script = script[:script.index("</script>")]
        assert "sessionStorage.getItem('qb.brandIntro')" in script and "sessionStorage.setItem('qb.brandIntro'" in script
        assert "classList.add('qb-brand-motion--intro')" in script
        assert "if (!mark.getClientRects().length) return;" in script, "only a mark on screen, which can end its arrival"
        # Once it has arrived, the class goes, so the pointer's hop never replays the arrival.
        assert "classList.remove('qb-brand-motion--intro')" in script and "'qb-intro-dot'" in script
        assert "data-brand-intro" in source.split("The brand draws itself in once a visit", 1)[0]

    @pytest.mark.parametrize("page", ["portal/templates/portal_login.html", "admin/templates/login.html",
                                      "admin/templates/admin_invite.html", "admin/templates/admin_account_setup.html"])
    def test_a_sign_in_page_s_mark_arrives_and_turns_while_it_signs_in(self, page):
        source = (ROOT / page).read_text(encoding="utf-8")
        assert "extra_class='qb-brand-motion--intro'" in source, page
        assert "dataset.state = 'loading'" in source or "dataset.state='loading'" in source, page

    def test_the_arrival_leaves_no_transform_behind(self):
        for selector, body in _rules(_css()):
            if selector.startswith(INTRO):
                assert "backwards" in body and "both" not in body and "forwards" not in body, selector

    def test_reduced_motion_stops_everything(self):
        reduced = _css().split("prefers-reduced-motion: reduce", 1)[1]
        assert "animation: none !important" in reduced
        assert "transition: none !important" in reduced
        assert "transform: none !important" in reduced, "the pointer's nudge rests too"


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
