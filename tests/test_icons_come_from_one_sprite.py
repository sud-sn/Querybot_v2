"""
Every icon comes from one sprite, drawn one way, by name.

Before: two Heroicons macros that disagreed about which icons they held (an
unknown name drew an empty box, silently), about forty hand-written <svg>
icons in templates and page scripts at five stroke widths, and characters
standing in for icons -- "×", "⋮⋮", "&#8635;", "&#8594;" turned on its side.

Now tools/build_icon_sprite.py builds static/icons/qb-icons.svg from Lucide,
the ic() macro and qbIcon() both draw a <use> of one of its symbols, and a
page carries no icon drawing of its own. These run the real macro (Jinja) and
the real helper (dukpy) and read the sprite and the pages as data.
"""

from __future__ import annotations

import importlib.util
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import dukpy
import pytest
from jinja2 import Environment, FileSystemLoader

from core.static_assets import asset_url

ROOT = Path(__file__).resolve().parents[1]
SPRITE = ROOT / "static" / "icons" / "qb-icons.svg"
HELPER = ROOT / "static" / "js" / "qb-icons.js"
SHELLS = {"admin": ROOT / "admin" / "templates", "portal": ROOT / "portal" / "templates"}


def _builder():
    spec = importlib.util.spec_from_file_location("build_icon_sprite", ROOT / "tools" / "build_icon_sprite.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _symbols() -> dict[str, ET.Element]:
    root = ET.parse(SPRITE).getroot()
    return {el.get("id"): el for el in root if el.tag.endswith("symbol")}


def _macro(shell: str):
    env = Environment(loader=FileSystemLoader(str(SHELLS[shell])), autoescape=True)
    env.globals["asset"] = asset_url     # as admin/routes.py and portal/routes.py register it
    return env.get_template("icons.html").module.ic


def _shell_sprite() -> str:
    """The sprite URL the admin shell writes on the helper's own script tag."""
    from unittest.mock import MagicMock

    import admin.routes as routes

    request = MagicMock()
    request.url.path = "/admin/"
    html = routes.templates.env.get_template("base.html").render(request=request)
    return re.search(r'<script src="/static/js/qb-icons\.js\?v=\w+" data-sprite="([^"]+)"', html).group(1)


def _helper(call: str, sprite: str | None = None) -> str:
    """Run qb-icons.js as the browser does: document.currentScript is its tag."""
    tag = "null" if sprite is None else (
        "{getAttribute: function (n) { return n === 'data-sprite' ? %s : null; }}" % json.dumps(sprite))
    return dukpy.evaljs("var window = {document: {currentScript: %s}};\n" % tag
                        + HELPER.read_text(encoding="utf-8") + f"\n{call}")


def _pages() -> list[Path]:
    pages = [p for d in SHELLS.values() for p in d.rglob("*.html") if p.name != "icons.html"]
    return sorted(pages) + sorted((ROOT / "static" / "js").glob("*.js"))


def _sprite_url(markup: str) -> str:
    return re.search(r'<use href="([^"#]+)#', markup).group(1)


class TestTheSprite:

    def test_it_holds_exactly_the_builders_icons(self):
        assert set(_symbols()) == set(_builder().ICONS)

    def test_each_symbol_is_a_24_unit_drawing(self):
        for name, symbol in _symbols().items():
            assert symbol.get("viewBox") == "0 0 24 24", name
            assert len(symbol) >= 1, name

    def test_it_ships_the_lucide_licence(self):
        licence = (SPRITE.parent / "LICENSE-Lucide.txt").read_text(encoding="utf-8")
        assert licence.startswith("ISC License")
        assert "Lucide" in licence


class TestTheMacro:

    @pytest.mark.parametrize("shell", sorted(SHELLS))
    def test_it_draws_a_symbol_of_the_sprite(self, shell):
        markup = str(_macro(shell)("chat", 16, "text-success"))
        assert 'class="qb-icon text-success"' in markup
        assert 'width="16" height="16"' in markup
        assert 'aria-hidden="true"' in markup
        assert markup.count("<use ") == 1 and markup.endswith("#chat\"/></svg>")
        assert "<path" not in markup

    @pytest.mark.parametrize("shell", sorted(SHELLS))
    def test_its_sprite_is_the_shipped_file(self, shell):
        url = _sprite_url(str(_macro(shell)("chat")))
        assert (ROOT / url.split("?")[0].lstrip("/")).resolve() == SPRITE.resolve()

    def test_admin_and_portal_draw_the_same_icon_alike(self):
        assert str(_macro("admin")("refresh", 14)) == str(_macro("portal")("refresh", 14))


class TestTheHelper:

    def test_it_draws_what_the_macro_draws(self):
        from_script = _helper("window.qbIcon('refresh', 14, 'text-muted')", _shell_sprite())
        from_template = str(_macro("portal")("refresh", 14, "text-muted"))
        assert from_script == from_template

    def test_without_its_tag_it_still_draws_from_the_sprite(self):
        assert _sprite_url(_helper("window.qbIcon('refresh')")) == "/static/icons/qb-icons.svg"

    def test_it_defaults_to_16px(self):
        assert 'width="16" height="16"' in _helper("window.qbIcon('x')")

    @pytest.mark.parametrize("name", ['x"><script>alert(1)</script>', "../x", "X", ""])
    def test_a_name_that_is_not_a_plain_name_draws_nothing(self, name):
        assert _helper(f"window.qbIcon({name!r})") == ""

    def test_a_class_with_markup_in_it_is_dropped(self):
        markup = _helper("window.qbIcon('x', 12, 'a\" onclick=\"b')")
        assert 'class="qb-icon"' in markup and "onclick" not in markup


def _names_called() -> dict[str, set[str]]:
    called: dict[str, set[str]] = {}
    for page in _pages():
        text = page.read_text(encoding="utf-8")
        for name in re.findall(r"""\bic\(\s*["']([^"']+)["']""", text) + \
                re.findall(r"""\bqbIcon\(\s*["']([^"']+)["']""", text):
            called.setdefault(name, set()).add(page.name)
    return called


class TestEveryPage:

    def test_every_icon_a_page_names_is_in_the_sprite(self):
        called = _names_called()
        assert len(called) >= 30
        missing = {name: sorted(pages) for name, pages in called.items() if name not in _symbols()}
        assert not missing, missing

    def test_no_page_draws_an_icon_of_its_own(self):
        # An <svg> in a page is the brand mark or a data graphic, marked as
        # such; an icon is ic() or qbIcon(), never path data in the page.
        own = []
        for page in _pages():
            if page == HELPER:
                continue
            for tag in re.findall(r"<svg\b[^>]*>", page.read_text(encoding="utf-8")):
                if not re.search(r'class="(qb-brand-motion__mark|qb-graphic)\b', tag):
                    own.append(f"{page.name}: {tag[:80]}")
        assert not own, own

    def test_the_shells_load_the_helper_before_any_page_script(self):
        for shell, base in (("admin", "base.html"), ("portal", "portal_base.html")):
            page = (SHELLS[shell] / base).read_text(encoding="utf-8")
            loads = re.search(r'<script src="\{\{ asset\(\'(js/qb-icons\.js)\'\) \}\}"[^>]*></script>', page)
            assert loads, shell
            assert (ROOT / "static" / loads.group(1)).is_file()
            head = page.split("</head>", 1)[0]
            assert loads.group(0) in head and head.index(loads.group(0)) < head.index("{% block head %}")


# Characters that stood in for an icon in the shells and the portal. Page-level
# admin screens that Phase 4 and 6 rebuild still carry some; these are the
# ones whose pages this change covers.
_GLYPHS = ("⋮⋮", "&#8635;", "&#8594;</span>", "&#8505;", "&#8249;</span>", ">×<", ">&times;<", ">✕<", "content:'+'")
_COVERED = [SHELLS["portal"] / name for name in ("portal_base.html", "portal_chat.html", "portal_dashboard.html")] + \
           [SHELLS["admin"] / "base.html"]


@pytest.mark.parametrize("page", _COVERED, ids=lambda p: p.name)
def test_no_character_stands_in_for_an_icon(page):
    text = page.read_text(encoding="utf-8")
    assert not [g for g in _GLYPHS if g in text]
