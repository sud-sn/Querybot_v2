"""
Every colour comes from tokens.css.

The Navy palette lives in tokens.css, but 251 colours were still written by
hand across 25 stylesheets and pages: the old palette's greens (#16a34a,
#38a169, #10b981, #86efac...) on the metrics, system, readiness and chat
pages, Tailwind slate greys on the navy sidebar (#64748B muted text at about
3:1), black and slate shadows at a dozen alphas, and var(--token, #hex)
fallbacks whose hex had drifted from the token. A colour written by hand does
not move when the palette does.

Now every stylesheet, page <style> block, style="" attribute and script-built
style string names a token (or mixes one); the sidebar's text and rules use
the shell's own inks. These read the sources as data.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CSS = ROOT / "static" / "css"
_LITERAL = re.compile(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{8}\b|(?<![\w-])#[0-9a-fA-F]{3,4}\b(?![\w-])|rgba?\([^)]*\)"
                      r"|hsla?\([^)]*\)")
_NAMED = re.compile(r":[^;{}]*?(?<![\w-])(white|black|red|green|blue|orange|gray|grey|yellow|silver)(?![\w-])")


def _clean(text: str) -> str:
    text = re.sub(r"/\*[\s\S]*?\*/", "", text)
    return re.sub(r'url\("data:[^"]*"\)', "", text)   # the select chevron: held to a token elsewhere


def _page_colour_sources(page: Path, scripts: bool) -> str:
    text = page.read_text(encoding="utf-8")
    parts = re.findall(r"<style[^>]*>([\s\S]*?)</style>", text)
    parts += re.findall(r'style="([^"]*)"', text)
    if scripts:
        parts += re.findall(r"<script[^>]*>([\s\S]*?)</script>", text)   # style strings built in script
    return _clean("\n".join(parts))


def _sources(scripts: bool = True):
    for sheet in sorted(CSS.glob("*.css")):
        if sheet.name != "tokens.css":
            yield sheet.name, _clean(sheet.read_text(encoding="utf-8"))
    for page in sorted(list((ROOT / "admin" / "templates").rglob("*.html")) +
                       list((ROOT / "portal" / "templates").rglob("*.html"))):
        yield page.name, _page_colour_sources(page, scripts)


def test_no_colour_is_written_by_hand():
    found = [f"{name}: {hit}" for name, text in _sources() for hit in _LITERAL.findall(text)]
    assert not found, found


def test_no_named_colour_stands_in_for_a_token():
    # Stylesheets and style attributes only: in a script, "green" is a tone's
    # class name more often than a colour.
    found = [f"{name}: {m.group(1)}" for name, text in _sources(scripts=False) for m in _NAMED.finditer(text)]
    assert not found, found


def test_no_stylesheet_carries_a_fallback_for_a_token():
    # tokens.css loads first on every page; a fallback beside a token is a
    # second copy of the colour that nothing keeps in step.
    found = []
    for name, text in _sources():
        found += [f"{name}: {m}" for m in re.findall(r"var\(--[\w-]+\s*,\s*[^)]*\)", text)
                  if re.search(r"#|rgb|hsl", m)]
    assert not found, found


def _tokens() -> dict[str, str]:
    css = _clean((CSS / "tokens.css").read_text(encoding="utf-8"))
    return {k: v.upper() for k, v in re.findall(r"(--[\w-]+):\s*(#[0-9a-fA-F]{6})\b", css)}


def _ratio(a: str, b: str) -> float:
    def lum(h: str) -> float:
        c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        r, g, bl = (x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c)
        return 0.2126 * r + 0.7152 * g + 0.0722 * bl
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


@pytest.mark.parametrize("sheet,selector", [
    ("portal.css", ".portal-nav a"),
    ("chat_workspace.css", ".portal-thread-workspace small"),
    ("chat_workspace.css", ".portal-thread-heading"),
    ("base.css", ".sidebar-group-label"),
])
def test_text_on_the_navy_shell_reads(sheet, selector):
    tokens = _tokens()
    css = _clean((CSS / sheet).read_text(encoding="utf-8"))
    body = next((b for s, b in re.findall(r"([^{}]+)\{([^{}]*)\}", css)
                 if selector in [x.strip() for x in s.split(",")] and re.search(r"(?:^|[;\s])color:", " " + b)), None)
    assert body is not None, selector
    ink = re.search(r"(?:^|[;\s])color:\s*var\((--[\w-]+)\)", " " + body)
    assert ink and ink.group(1) in tokens, (selector, body)
    assert _ratio(tokens[ink.group(1)], tokens["--shell"]) >= 4.5, (selector, ink.group(1))
