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

import pytest

ROOT = Path(__file__).resolve().parents[1]


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
