"""
The brand reads as one product: the navy mark in both consoles, and no colour
that still speaks for the green palette.

The palette moved to Navy (tests/test_every_token_pair_is_readable.py), but the
colour also lived in places the tokens do not reach: the mark's own gradient
(an <img> cannot read a custom property), the fallbacks a script or stylesheet
uses when a token cannot be read (a chart paints to a canvas and needs a
concrete colour), and gradients written out by hand. Each kept the green, and
the admin console showed no mark at all.

These read the mark files, stylesheets and scripts as data, compare each
fallback with the token it stands in for, and render the real admin shell.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
MARKS = [ROOT / "static" / "img" / "logo-mark.svg", ROOT / "static" / "img" / "logo-mark-sm.svg"]


def _tokens() -> dict[str, str]:
    css = re.sub(r"/\*[\s\S]*?\*/", "", (ROOT / "static" / "css" / "tokens.css").read_text(encoding="utf-8"))
    return {k: v.upper() for k, v in re.findall(r"(--[\w-]+):\s*(#[0-9a-fA-F]{6})\b", css)}


def _rgb(hexcolour: str) -> tuple[int, int, int]:
    h = hexcolour.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _lum(hexcolour: str) -> float:
    channels = [c / 255 for c in _rgb(hexcolour)]
    r, g, b = (c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _ratio(a: str, b: str) -> float:
    high, low = sorted((_lum(a), _lum(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _stops(mark: Path) -> dict[str, list[str]]:
    svg = mark.read_text(encoding="utf-8")
    gradients = re.findall(r'<linearGradient id="(\w+)"[^>]*>([\s\S]*?)</linearGradient>', svg)
    found = {gid: re.findall(r'stop-color="(#[0-9A-Fa-f]{6})"', body) for gid, body in gradients}
    return {"tile": next(v for k, v in found.items() if "Tile" in k),
            "glyph": next(v for k, v in found.items() if "Glyph" in k)}


class TestTheMark:

    @pytest.mark.parametrize("mark", MARKS, ids=lambda p: p.name)
    def test_its_tile_is_navy(self, mark):
        for stop in _stops(mark)["tile"]:
            r, g, b = _rgb(stop)
            assert b > g > r and _lum(stop) < 0.05, stop

    @pytest.mark.parametrize("mark", MARKS, ids=lambda p: p.name)
    def test_its_glyph_is_azure(self, mark):
        for stop in _stops(mark)["glyph"]:
            r, g, b = _rgb(stop)
            assert b > g > r, stop

    @pytest.mark.parametrize("mark", MARKS, ids=lambda p: p.name)
    def test_its_glyph_stands_off_its_tile(self, mark):
        # The bars are cut through the glyph to the tile: they read only if
        # the two differ by the 3:1 WCAG asks of a graphic, at every stop.
        stops = _stops(mark)
        for glyph in stops["glyph"]:
            for tile in stops["tile"]:
                assert _ratio(glyph, tile) >= 3.0, (glyph, tile)

    def test_the_reduced_mark_is_the_same_mark(self):
        assert _stops(MARKS[0]) == _stops(MARKS[1])

    @pytest.mark.parametrize("mark", MARKS, ids=lambda p: p.name)
    def test_every_page_links_it_by_its_content(self, mark):
        # A recoloured mark reaches a browser that cached the old one only
        # when every reference changes with it: each is asset(), whose version
        # is the file's hash (tests/test_asset_urls_follow_the_file.py).
        linked, by_hand = 0, []
        for page in list((ROOT / "admin" / "templates").rglob("*.html")) + \
                list((ROOT / "portal" / "templates").rglob("*.html")):
            text = page.read_text(encoding="utf-8")
            linked += text.count(f"asset('img/{mark.name}')")
            by_hand += [page.name for _ in re.finditer(rf"/static/img/{re.escape(mark.name)}", text)]
        assert linked and not by_hand, by_hand


def test_the_admin_console_carries_the_mark():
    import admin.routes as routes

    request = MagicMock()
    request.url.path = "/admin/"
    html = routes.templates.env.get_template("base.html").render(request=request)
    brand = re.search(r'<div class="sidebar-brand">[\s\S]*?</div>\s*</div>', html).group(0)
    source = re.search(r'<img src="(/static/img/logo-mark\.svg)\?v=[\w.-]+"', brand)
    assert source, brand
    assert (ROOT / source.group(1).lstrip("/")).is_file()


# The chart theme's keys and the tokens they are read from (chart-palettes.js,
# QB_CHART_THEME), for the fallbacks qb-charts.js writes as `theme.key || '#hex'`.
_THEME_TOKENS = {"surface": "--surface", "tooltipBg": "--surface-raised", "ink": "--text-strong",
                 "axis": "--text-muted", "split": "--line-subtle", "axisLine": "--line",
                 "good": "--success", "bad": "--danger"}


def _fallbacks() -> list[tuple[str, str, str]]:
    found = []
    for source in ("static/js/chart-palettes.js", "static/js/qb-charts.js", "static/css/brand-motion.css"):
        text = (ROOT / source).read_text(encoding="utf-8")
        found += [(source, token, hexcolour) for token, hexcolour in
                  re.findall(r"token\('(--[\w-]+)',\s*'(#[0-9a-fA-F]{6})'\)", text)]
        found += [(source, token, hexcolour) for token, hexcolour in
                  re.findall(r"var\((--[\w-]+),\s*(#[0-9a-fA-F]{6})\)", text)]
        found += [(source, _THEME_TOKENS[key], hexcolour) for key, hexcolour in
                  re.findall(r"theme\.(\w+) \|\| '(#[0-9a-fA-F]{6})'", text) if key in _THEME_TOKENS]
    return found


def test_every_brand_and_chart_fallback_is_its_tokens_colour():
    tokens = _tokens()
    fallbacks = _fallbacks()
    # The chart scripts paint to a canvas, which cannot read a custom
    # property, so they keep concrete fallbacks; stylesheets need none
    # (tests/test_colour_comes_from_tokens.py).
    assert len(fallbacks) >= 15
    stale = [f"{source}: {token} falls back to {colour}, the token is {tokens[token]}"
             for source, token, colour in fallbacks if token in tokens and colour.upper() != tokens[token]]
    assert not stale, stale


def test_no_gradient_runs_the_brand_into_green():
    # The old palette's signature: a blue-to-green (or green-to-primary) bar.
    # A decorative gradient is --brand-gradient, defined once in tokens.css.
    mixed = []
    for page in list((ROOT / "static" / "css").glob("*.css")) + \
            list((ROOT / "admin" / "templates").rglob("*.html")) + list((ROOT / "portal" / "templates").rglob("*.html")):
        for gradient in re.findall(r"linear-gradient\((?:[^()]|\([^()]*(?:\([^()]*\))?[^()]*\))*\)", page.read_text(encoding="utf-8")):
            names = set(re.findall(r"var\(--([\w-]+)", gradient))
            if names & {"green", "success"} and names & {"blue", "primary", "accent-500", "accent-600"}:
                mixed.append(f"{page.name}: {gradient[:90]}")
    assert not mixed, mixed
