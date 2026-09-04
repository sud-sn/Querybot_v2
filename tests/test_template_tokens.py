"""
tests/test_template_tokens.py

867 font sizes moved from literals into var() references, in a surface nothing
was checking.

`test_production_ui.test_every_referenced_token_is_actually_defined` guards
exactly this class of bug — a var() with no definition computes to nothing,
silently — but it scans six stylesheets and no templates. Verified by injecting
`var(--font-nonexistent)` into portal_login.html and watching it pass.

That mattered the moment the sweep ran: every template carries its own inline
<style>, which is where the drift lived and where the tokens now live, and a
mistyped token there degrades to no rule at all. The text does not turn red, it
just quietly renders at whatever it inherits.

So this file covers templates, and adds the two properties the sweep depends on
that the stylesheet guard has no reason to check: that a template referencing a
token can actually SEE tokens.css, and that the sweep did not silently invent a
rung.
"""

import os
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TEMPLATES = sorted(
    p for folder in ("portal", "admin")
    for p in (ROOT / folder / "templates").glob("*.html")
)

VAR_REF = re.compile(r"var\(\s*(--[a-z0-9-]+)\s*[,)]")
VAR_NO_FALLBACK = re.compile(r"var\(\s*(--[a-z0-9-]+)\s*\)")
DEFINITION = re.compile(r"(--[a-z0-9-]+)\s*:")
EXTENDS = re.compile(r"""{%-?\s*extends\s+["']([^"']+)["']""")


def _read(path):
    return path.read_text(encoding="utf-8", errors="replace")


def _defined_tokens() -> set[str]:
    """Everything on :root, plus per-component properties each sheet defines."""
    tokens = set()
    for name in ("tokens", "base", "admin", "portal", "chat_workspace",
                 "production", "brand-motion", "fonts"):
        path = ROOT / "static" / "css" / f"{name}.css"
        if path.exists():
            tokens |= set(DEFINITION.findall(_read(path)))
    return tokens


def _is_partial(path: Path) -> bool:
    """A macro or include library, never rendered on its own.

    Its markup is emitted INSIDE a page that loads the tokens, so asking
    whether the partial itself reaches tokens.css is the wrong question — it
    has no <head> of its own to put a link in.
    """
    src = _read(path)
    return "{% macro" in src and not EXTENDS.search(src) and "<html" not in src.lower()


def _reaches_tokens_css(path: Path, depth: int = 0) -> bool:
    """A template sees the tokens if it links tokens.css or extends one that does."""
    src = _read(path)
    if "tokens.css" in src:
        return True
    if depth > 4:
        return False
    match = EXTENDS.search(src)
    if not match:
        return False
    parent = match.group(1)
    for candidate in (path.parent / parent,
                      ROOT / "portal" / "templates" / parent,
                      ROOT / "admin" / "templates" / parent):
        if candidate.exists():
            return _reaches_tokens_css(candidate, depth + 1)
    return False


class EveryTokenATemplateUsesIsDefined(unittest.TestCase):
    """The check the stylesheet guard does not extend to this surface."""

    def test_no_template_references_an_undefined_token(self):
        defined = _defined_tokens()
        self.assertIn("--font-sm", defined, "token file not found or unparsed")
        missing = {}
        for path in TEMPLATES:
            src = _read(path)
            # A template may define its own scoped properties inline.
            local = set(DEFINITION.findall(src))
            unknown = {
                ref for ref in VAR_NO_FALLBACK.findall(src)
                if ref not in defined and ref not in local
            }
            if unknown:
                missing[path.name] = sorted(unknown)
        self.assertEqual(missing, {}, f"undefined tokens in templates: {missing}")

    def test_the_scan_actually_finds_references(self):
        """Guards the guard — an empty scan would pass the test above."""
        total = sum(len(VAR_REF.findall(_read(p))) for p in TEMPLATES)
        self.assertGreater(total, 500, "the var() scan is not matching anything")


class EveryTemplateUsingATokenCanSeeOne(unittest.TestCase):
    """A var() in a template that never loads tokens.css computes to nothing."""

    def test_templates_that_reference_tokens_reach_the_token_file(self):
        orphans = [
            p.name for p in TEMPLATES
            if VAR_REF.search(_read(p))
            and not _is_partial(p)
            and not _reaches_tokens_css(p)
        ]
        self.assertEqual(orphans, [], f"templates using var() without tokens.css: {orphans}")

    def test_the_partial_exemption_is_narrow(self):
        """A page that forgot to extend a base must not slip through it."""
        pages = [p.name for p in TEMPLATES if not _is_partial(p)]
        self.assertIn("portal_chat.html", pages)
        self.assertIn("client_detail.html", pages)


class TheSweepStayedOnTheScale(unittest.TestCase):
    """It replaced literals with rungs; it must not have invented one."""

    FONT_TOKENS = {
        "--font-2xs", "--font-xs", "--font-sm", "--font-base", "--font-md",
        "--font-lg", "--font-xl", "--font-2xl", "--font-3xl",
    }

    def test_font_size_tokens_used_in_templates_are_real_rungs(self):
        used = set()
        for path in TEMPLATES:
            used |= set(re.findall(r"font-size:\s*var\(\s*(--[a-z0-9-]+)\s*\)", _read(path)))
        self.assertTrue(used, "no tokenised font sizes found")
        self.assertEqual(
            sorted(used - self.FONT_TOKENS), [],
            "font-size is using a token that is not part of the type scale",
        )

    def test_the_caption_rung_carries_its_share(self):
        """--font-2xs was added for a 10px tier used ~124 times; if the sweep
        did not actually use it, the rung was added for nothing."""
        count = sum(
            len(re.findall(r"font-size:\s*var\(--font-2xs\)", _read(p)))
            for p in TEMPLATES
        )
        self.assertGreater(count, 50)


if __name__ == "__main__":
    unittest.main()
