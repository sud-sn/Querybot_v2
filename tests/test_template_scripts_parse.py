# -*- coding: utf-8 -*-
"""Every inline <script> in every template has to parse — tests/chat_js.py cannot see this.

A JavaScript SyntaxError is not a broken feature, it is a broken page: the
browser discards the whole <script> element, so one bad character four
thousand lines down takes the websocket, the message rendering, the charts and
the history with it. The page still serves its server-rendered shell, so it
looks alive and does nothing.

This shipped. ``portal/templates/portal_chat.html`` carried

    tabs.push('<button ...>${t('ui.chat.card.visual')}</button>');

where the interpolation needed backticks — the inner quote closed the string
and the rest of the argument list was bare identifiers. The entire 4,822-line
block had not parsed since the page's i18n retrofit.

Nothing caught it because ``tests/chat_js.py`` lifts individual functions out
of the source and runs those. That is the right way to test what a function
does and it is structurally blind to the file around it: every one of the 580
language assertions passed against a page that could not load.

So this parses each block whole. ``new Function(body)`` compiles without
executing, which is the point — the page needs a DOM and a socket to run, and
needs neither to be syntactically valid.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

dukpy = pytest.importorskip(
    "dukpy", reason="dukpy compiles the page's own JavaScript; without it nothing here can run",
)

# Inline scripts only: a src= tag has no body to parse here.
_SCRIPT = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", re.S | re.I)

TEMPLATE_DIRS = ("portal/templates", "admin/templates")


def _blocks(path: Path):
    """Each inline script body, with Jinja neutralised, and the line it starts on."""
    source = path.read_text(encoding="utf-8")
    for match in _SCRIPT.finditer(source):
        body = match.group(1)
        if not body.strip():
            continue
        line = source[: match.start(1)].count("\n") + 1
        # An expression becomes a literal and a statement becomes nothing, so
        # what is checked is the JavaScript the template author wrote rather
        # than the template engine's syntax.
        body = re.sub(r"\{\{.*?\}\}", "null", body, flags=re.S)
        body = re.sub(r"\{%.*?%\}", "", body, flags=re.S)
        yield line, body


def _parse_error(body: str) -> str:
    """"" when the body compiles, else the engine's message."""
    try:
        dukpy.evaljs("new Function(dukpy['body']); 'ok';", body=body)
    except Exception as exc:  # noqa: BLE001 — any failure to compile is the finding
        return str(exc).split("\n")[0]
    return ""


def _templates() -> list[Path]:
    found: list[Path] = []
    for directory in TEMPLATE_DIRS:
        found.extend(sorted((ROOT / directory).glob("*.html")))
    return found


class TestEveryInlineScriptCompiles(unittest.TestCase):

    def test_the_templates_are_actually_being_read(self):
        # A glob that matches nothing would make every assertion below vacuous.
        templates = _templates()
        self.assertGreaterEqual(len(templates), 40, "template glob found almost nothing")
        self.assertTrue(any(p.name == "portal_chat.html" for p in templates))

    def test_the_chat_page_has_one_large_block_to_check(self):
        # The defect lived in a 4,800-line block. If the extractor silently
        # returned a fragment, this file would pass while checking nothing.
        chat = ROOT / "portal" / "templates" / "portal_chat.html"
        biggest = max(len(body.splitlines()) for _line, body in _blocks(chat))
        self.assertGreater(biggest, 3000, "the chat page's main script was not extracted")

    def test_every_inline_script_in_every_template_parses(self):
        failures = []
        for path in _templates():
            for line, body in _blocks(path):
                error = _parse_error(body)
                if error:
                    failures.append(f"{path.relative_to(ROOT)}:{line} — {error}")
        self.assertEqual(failures, [], "\n".join(failures))

    def test_a_broken_interpolation_is_detected(self):
        # The exact shape that shipped: ${...} inside a single-quoted string,
        # where the inner quote closes it early. Proves the check discriminates
        # rather than passing on everything handed to it.
        broken = (
            "tabs.push('<button data-artifact-tab=\"visual\">"
            "${t('ui.chat.card.visual')}</button>');"
        )
        self.assertTrue(_parse_error(broken))
        fixed = broken.replace("push('", "push(`").replace("');", "`);")
        self.assertEqual(_parse_error(fixed), "")

    def test_valid_modern_syntax_is_not_reported(self):
        # The engine has to understand what the pages actually use, or this
        # module becomes a source of false failures nobody trusts.
        for snippet in (
            "const f = (x) => `a${x}b`;",
            "let a = [...[1, 2]]; const {b} = {b: 1};",
            "const v = ({}).a?.b ?? 'd';",
            "async function go() { await Promise.resolve(1); }",
            "for (const [k, v] of Object.entries({a: 1})) { void k; void v; }",
        ):
            self.assertEqual(_parse_error(snippet), "", snippet)


if __name__ == "__main__":
    unittest.main()
