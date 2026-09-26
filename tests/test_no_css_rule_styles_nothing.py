"""
No stylesheet rule styles nothing.

The stylesheets carried 158 rules for classes that no page, script or route
emits any more: the client sidebar two redesigns back, spacing utilities no
template used, sixty-three chat-workspace rules for a rail and a history
search since rebuilt, the portal's bottom navigation. Each was a rule someone
reads, edits and tests against for nothing, and a class name that anyone
reusing it by accident inherits styles nobody meant.

These read every rule in static/css and every place a class can be named --
templates, scripts, the icon sprite and the Python that writes markup -- and
fail on a rule whose every selector needs a class that appears nowhere. A
class assembled at runtime ('btn-' + tone, `status-${x}`,
qb-brand-motion--{{ size }}, f"pill-{state}", {{ tone }}-badge) counts as used
for every class that starts or ends with the fixed part. The bundled chart,
grid and graph libraries add classes of their own that a sheet may style, so
the names they spell out count too; what they assemble is theirs, not ours.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CSS = ROOT / "static" / "css"

# Where the fixed part of a class meets the part filled in when it runs.
_BUILT_AFTER = re.compile(r"([A-Za-z][\w-]*-)(?=\{\{|\{%|\{[A-Za-z_(]|\{\}|%s|%\(|['\"]\s*[+~]|\$\{|`)")
_BUILT_BEFORE = re.compile(r"(?:\}\}|\}|%s|['\"]\s*[+~]\s*[\w.()\[\]'\"]+\s*[+~]\s*['\"])(-[A-Za-z][\w-]*)")


def _sources():
    """Every file of the product's that can put a class on a page. Tests are not the product."""
    for folder in ("admin", "portal"):
        yield from (ROOT / folder / "templates").rglob("*.html")
    yield from (ROOT / "static" / "js").rglob("*.js")
    yield from (ROOT / "static").rglob("*.svg")
    for here, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if not (d.startswith((".", "_")) or "venv" in d or d in ("tests", "node_modules", "site-packages"))]
        yield from (Path(here) / name for name in files if name.endswith(".py"))


def _libraries():
    return sorted((ROOT / "static" / "vendor").glob("*.js"))


class Names:
    """The words a set of sources names, and the fixed parts of the classes they
    assemble; a library's words count, its assembled names do not."""

    def __init__(self, texts, libraries=()):
        self.words, self.starts, self.ends = set(), set(), set()
        for text in texts:
            self.words.update(re.findall(r"[A-Za-z][\w-]*", text))
            self.starts.update(_BUILT_AFTER.findall(text))
            self.ends.update(_BUILT_BEFORE.findall(text))
        for text in libraries:
            self.words.update(re.findall(r"[A-Za-z][\w-]*", text))

    def name(self, cls: str) -> bool:
        return cls in self.words or any(cls.startswith(s) for s in self.starts) or any(cls.endswith(e) for e in self.ends)


def _blank_comments(css: str) -> str:
    return re.sub(r"/\*[\s\S]*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), css)


def dead_rules(css: str, names: Names) -> list[str]:
    """Selectors whose every comma-separated part needs a class nothing names.
    A rule nested in @media or @supports is read like any other; keyframe
    steps and selectors with no class are never dead by this measure."""
    dead = []
    for m in re.finditer(r"([^{}@;]+)\{([^{}]*)\}", _blank_comments(css)):
        selector = " ".join(m.group(1).split())
        if not re.search(r"\.[A-Za-z]", selector):
            continue
        parts = [part.strip() for part in selector.split(",")]
        if all(any(not names.name(cls) for cls in re.findall(r"\.([A-Za-z][\w-]*)", part)) for part in parts):
            dead.append(selector)
    return dead


def test_every_stylesheet_rule_styles_something_the_product_names():
    names = Names((path.read_text(encoding="utf-8", errors="ignore") for path in _sources()),
                  (path.read_text(encoding="utf-8", errors="ignore") for path in _libraries()))
    assert "btn-primary" in names.words and "sidebar-group-label" in names.words   # the sources were read
    assert "grid-stack-placeholder" in names.words                                 # and the libraries
    found = [f"{sheet.name}: {selector[:90]}"
             for sheet in sorted(CSS.glob("*.css")) if sheet.name not in ("tokens.css", "fonts.css")
             for selector in dead_rules(sheet.read_text(encoding="utf-8"), names)]
    assert not found, found


@pytest.mark.parametrize("css,source,dead", [
    (".orphan { color: red; }", '<div class="card">', [".orphan"]),
    (".card { color: red; }", '<div class="card">', []),
    (".card .orphan { color: red; }", '<div class="card">', [".card .orphan"]),
    (".card, .orphan { color: red; }", '<div class="card">', []),
    (".orphan:hover, .orphan::after { color: red; }", "", [".orphan:hover, .orphan::after"]),
    ("@media (max-width: 600px) { .orphan { display: none; } }", "", [".orphan"]),
    ("/* .orphan { color: red; } */ a:hover { color: red; }", "", []),
    ("@keyframes spin { from { opacity: 0; } 50% { opacity: .5; } to { opacity: 1; } }", "", []),
    (".status-ok { color: green; }", "el.className = 'status-' + tone;", []),
    (".status-ok { color: green; }", "return `<span class=\"status-${tone}\">`;", []),
    (".qb-brand-motion--lg { width: 4rem; }", '<span class="qb-brand-motion--{{ size }}">', []),
    (".pill-ok { color: green; }", 'html = f\'<span class="pill-{state}">\'', []),
    (".pill-ok { color: green; }", "{{ 'pill-' ~ state }}", []),
    (".ok-badge { color: green; }", '<span class="{{ tone }}-badge">', []),
    (".status-ok { color: green; }", "status = 'ok'", [".status-ok"]),
])
def test_the_rule_finds_a_rule_nothing_names_and_spares_a_class_built_at_runtime(css, source, dead):
    assert dead_rules(css, Names([source])) == dead


@pytest.mark.parametrize("css,library,dead", [
    (".grid-stack-placeholder { border: 0; }", "el.classList.add('grid-stack-placeholder')", []),
    (".text-md { font-size: 1rem; }", "t.className = 'text-' + size;", [".text-md"]),
])
def test_a_library_s_own_classes_count_and_what_it_assembles_does_not(css, library, dead):
    assert dead_rules(css, Names([], [library])) == dead
