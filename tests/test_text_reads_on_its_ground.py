"""
Text reads on the ground it sits on: 4.5:1, from tokens, never dimmed.

An axe audit of 32 admin, portal and sign-in views found 133 pieces of text
under 4.5:1. The admin sidebar's group headings were the brand's light blue at
60% opacity on navy, 2.62:1, on every admin page. Sixty-seven rules drew text
in --gray-500, a step of the grey ramp at 4.26:1 on a card and 3.97:1 on the
page, where --text-muted (6.25:1) is the token for secondary text. And ten
rules faded text with opacity -- an alert's body, a setup progress line, the
compliance pills, a provider link, three chat captions -- which lowers its
contrast by an amount no token records.

These compute every ratio from tokens.css: the text tokens on each light
surface, the sidebar's inks on navy, and each rule that colours text with a
grey, wherever it is written.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CSS = ROOT / "static" / "css"


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


LIGHT_GROUNDS = ["--surface-raised", "--surface", "--surface-alt", "--paper", "--paper-sunken", "--paper-recessed"]
TEXT = ["--text-strong", "--text", "--text-soft", "--text-secondary", "--text-muted"]
SIDEBAR_GROUNDS = ["--sidebar-bg", "--sidebar-surface", "--sidebar-active"]
SIDEBAR_TEXT = ["--sidebar-text", "--sidebar-muted", "--sidebar-accent", "--shell-text", "--shell-muted", "--shell-accent"]


@pytest.mark.parametrize("ink", TEXT)
def test_every_text_token_reads_on_every_light_surface(ink):
    tokens = _tokens()
    for ground in LIGHT_GROUNDS:
        assert _ratio(tokens[ink], tokens[ground]) >= 4.5, (ink, ground)


@pytest.mark.parametrize("ink", SIDEBAR_TEXT)
def test_every_sidebar_ink_reads_on_the_navy(ink):
    tokens = _tokens()
    for ground in SIDEBAR_GROUNDS:
        assert _ratio(tokens[ink], tokens[ground]) >= 4.5, (ink, ground)


def _faint_greys() -> set[str]:
    """Neutral tokens a reader cannot read on a card: under 4.5:1 on --surface.
    --text-inverse is for dark grounds and is not among them."""
    tokens = _tokens()
    return {name for name, value in tokens.items()
            if re.fullmatch(r"--(?:gray-\d+|text(?:-[a-z]+)?)", name) and name != "--text-inverse"
            and _ratio(value, tokens["--surface"]) < 4.5}


def _sources():
    """(where, css, is_markup): every stylesheet, page <style>, style="" and script."""
    for sheet in sorted(CSS.glob("*.css")):
        if sheet.name not in ("tokens.css", "fonts.css"):
            yield sheet.name, sheet.read_text(encoding="utf-8"), False
    for page in sorted(list((ROOT / "admin" / "templates").rglob("*.html")) +
                       list((ROOT / "portal" / "templates").rglob("*.html"))):
        text = page.read_text(encoding="utf-8")
        yield page.name, "\n".join(re.findall(r"<style[^>]*>([\s\S]*?)</style>", text)), False
        yield page.name, re.sub(r"<style[^>]*>[\s\S]*?</style>", "", text), True
    for script in sorted((ROOT / "static" / "js").glob("*.js")):
        yield script.name, script.read_text(encoding="utf-8"), True


# A control that is disabled, and a placeholder, are exempt from the ratio.
_EXEMPT = re.compile(r":disabled|\[disabled\]|aria-disabled|\.disabled|placeholder")


def _grey_text(where: str, css: str, is_markup: bool, faint: set[str]) -> list[str]:
    chunks = [("", css)] if is_markup else re.findall(r"([^{}]+)\{([^{}]*)\}", _strip(css))
    found = []
    for selector, body in chunks:
        if selector and _EXEMPT.search(selector):
            continue
        for token in re.findall(r"(?<![\w-])color\s*:\s*var\((--[\w-]+)\)", body):
            if token in faint:
                found.append(f"{where}: {' '.join(selector.split())[:60]} color {token}")
    return found


def test_no_text_is_drawn_in_a_grey_a_reader_cannot_read():
    faint = _faint_greys()
    assert "--gray-500" in faint and "--text-muted" not in faint     # the premise, from the data
    offenders = [hit for where, css, is_markup in _sources() for hit in _grey_text(where, css, is_markup, faint)]
    assert not offenders, offenders


@pytest.mark.parametrize("css,is_markup,flagged", [
    (".hint { color: var(--gray-500); }", False, True),
    ('<span style="color:var(--gray-400)">', True, True),
    (".note { color: var(--text-muted); }", False, False),
    (".btn:disabled { color: var(--text-faint); }", False, False),
    ("input::placeholder { color: var(--gray-400); }", False, False),
])
def test_the_rule_flags_grey_text_and_spares_a_disabled_control(css, is_markup, flagged):
    assert bool(_grey_text("sample", css, is_markup, _faint_greys())) is flagged


def _rule(css: str, selector: str) -> str:
    return re.search(r"(?:^|\})\s*" + re.escape(selector) + r"\s*\{([^}]*)\}", _strip(css)).group(1)


def test_the_admin_sidebar_s_group_headings_read_on_the_navy():
    body = _rule((CSS / "base.css").read_text(encoding="utf-8"), ".sidebar-group-label")
    ink = re.search(r"(?<![\w-])color\s*:\s*var\((--[\w-]+)\)", body).group(1)
    assert "opacity" not in body
    assert _ratio(_tokens()[ink], _tokens()["--sidebar-bg"]) >= 4.5, ink


# Opacity is for what says "not now" -- a disabled, inactive or finished
# item -- for icons and marks (a sort arrow, a check drawn in ::before), and
# for animation frames. Text that has to be read takes its colour from a
# text token instead.
_MAY_FADE = re.compile(r"disabled|inactive|done|placeholder|icon|dot|tick|chev|kind|skeleton|loading|busy|"
                       r"::?before|::?after|:hover|drag|ghost|^\s*(?:from|to|\d+%(?:\s*,\s*\d+%)*)\s*$")


def _faded(where: str, css: str) -> list[str]:
    found = []
    for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", _strip(css)):
        m = re.search(r"(?:^|;|\s)opacity\s*:\s*(0?\.\d+)\b", body)
        if m and 0 < float(m.group(1)) < 1 and not _MAY_FADE.search(" ".join(selector.split())):
            found.append(f"{where}: {' '.join(selector.split())[:70]} opacity {m.group(1)}")
    return found


def test_no_text_is_faded_with_opacity():
    faded = [hit for where, css, is_markup in _sources() if not is_markup for hit in _faded(where, css)]
    assert not faded, faded


@pytest.mark.parametrize("css,flagged", [
    (".alert-text { opacity: .85; }", True),
    (".caption { color: var(--text-muted); opacity: .7; }", True),
    (".btn:disabled { opacity: .5; }", False),
    (".row-inactive { opacity: .5; }", False),
    (".sidebar-link-icon { opacity: .72; }", False),
    ("@keyframes pulse { 50% { opacity: .4; } }", False),
])
def test_the_rule_flags_faded_text_and_spares_states_icons_and_frames(css, flagged):
    assert bool(_faded("sample", css)) is flagged
