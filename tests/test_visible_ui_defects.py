"""
tests/test_visible_ui_defects.py

Three defects a client could see, and one token that unblocks the rest.

BOLD. `formatBotText` rewrote `*x*` to <strong> and had no rule for `**x**`,
so the single-asterisk pattern matched the inner pair and left the outer two
behind: "**10.61%**" rendered as "*10.61%*" with the asterisks visible. Worse,
a line mixing both constructs had the text BETWEEN them bolded, because the
regex paired the second `*` of the first marker with the first `*` of the
second. Models emit `**` constantly, so this was on ordinary answers.

ICONS. Two icon macros exist — portal and admin — and each was missing an icon
the other has. `ic()` renders an empty <svg> for an unknown name, silently, so
a missing entry is a blank box rather than an error. The portal one was the
password-reveal button on the login screen, the second page a new client sees.

THE MARK. The logo is drawn on a 40-unit grid and rendered at 16-32px. At the
16px favicon its three bars were 1.36px wide separated by 0.60px — sub-pixel.
A reduced two-bar variant now serves that size.

THE 10px RUNG. 963 hardcoded font-sizes across 34 templates cannot be swept
onto the scale while the scale has no rung at 10px, because the product uses a
10px caption tier ~124 times and "snap to nearest" would enlarge every one.
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def _icons_defined(rel):
    return set(re.findall(r"name == '([a-z0-9_-]+)'", _read(rel)))


def _icons_called(folder):
    called = set()
    for path in (ROOT / folder).glob("*.html"):
        if path.name == "icons.html":
            continue
        called |= set(re.findall(
            r'ic\(\s*["\']([a-z0-9_-]+)["\']',
            path.read_text(encoding="utf-8", errors="replace"),
        ))
    return called


class EveryIconCalledIsDrawn(unittest.TestCase):
    """`ic()` on an unknown name emits an empty <svg> — a blank box, no error."""

    def test_portal(self):
        missing = sorted(_icons_called("portal/templates")
                         - _icons_defined("portal/templates/icons.html"))
        self.assertEqual(missing, [], f"portal icons render blank: {missing}")

    def test_admin(self):
        missing = sorted(_icons_called("admin/templates")
                         - _icons_defined("admin/templates/icons.html"))
        self.assertEqual(missing, [], f"admin icons render blank: {missing}")

    def test_the_two_that_were_blank_are_specifically_covered(self):
        """Guards the fix: `eye` on the login page, `check` on client setup."""
        self.assertIn("eye", _icons_defined("portal/templates/icons.html"))
        self.assertIn("check", _icons_defined("admin/templates/icons.html"))


class MarkdownEmphasisSurvivesRendering(unittest.TestCase):
    """The JS lives in a template, so assert the rules exist and are ordered."""

    def setUp(self):
        self.chat = _read("portal/templates/portal_chat.html")

    def test_a_double_asterisk_rule_exists(self):
        self.assertIn(r"h = h.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');",
                      self.chat)

    def test_it_runs_before_the_single_asterisk_rule(self):
        """Reversed, the single rule eats the inner pair and strands the outer."""
        double = self.chat.index(r"/\*\*([^*\n]+)\*\*/g")
        single = self.chat.index(r"/\*([^*\n]+)\*/g")
        self.assertLess(double, single)


class TheEmphasisRulesBehave(unittest.TestCase):
    """The regexes themselves, executed. Python and JS agree on this syntax."""

    @staticmethod
    def _render(text):
        text = re.sub(r'\*\*([^*\n]+)\*\*', r'<strong>\1</strong>', text)
        return re.sub(r'\*([^*\n]+)\*', r'<em>\1</em>', text)

    def test_bold_leaves_no_stray_asterisks(self):
        out = self._render("Controlled compounds are **10.61%** of revenue")
        self.assertEqual(out, "Controlled compounds are <strong>10.61%</strong> of revenue")
        self.assertNotIn("*", out)

    def test_a_line_mixing_both_does_not_bold_the_text_between_them(self):
        """The nastiest form of the old bug."""
        out = self._render("**bold** and *italic* together")
        self.assertEqual(out, "<strong>bold</strong> and <em>italic</em> together")

    def test_plain_text_is_untouched(self):
        self.assertEqual(self._render("no markup at all"), "no markup at all")


class TheReducedMark(unittest.TestCase):
    """Drawn on 40 units, rendered at 16. The bars have to survive that."""

    MARK = "static/img/logo-mark-sm.svg"

    def setUp(self):
        self.svg = _read(self.MARK)
        self.bars = [
            (float(x), float(w))
            for x, w in re.findall(r'<rect x="([\d.]+)"[^>]*width="([\d.]+)"[^>]*fill="url\(#qbTileSm\)"',
                                   self.svg)
        ]

    def test_it_exists_and_has_two_bars(self):
        self.assertEqual(len(self.bars), 2, "two bars, not three — that is the point")

    def test_the_bars_resolve_at_favicon_size(self):
        scale = 16 / 40
        (x1, w1), (x2, _w2) = self.bars
        self.assertGreaterEqual(w1 * scale, 1.5, "a bar under ~1.5px at 16px blurs away")
        self.assertGreaterEqual((x2 - x1 - w1) * scale, 0.9, "the gap must survive too")

    def test_it_shares_the_optical_centre_of_the_full_mark(self):
        """Or swapping at a breakpoint visibly shifts the mark."""
        (x1, _w1), (x2, w2) = self.bars
        self.assertAlmostEqual((x1 + x2 + w2) / 2, 20.0, places=2)

    def test_both_favicons_point_at_it(self):
        for page in ("portal/templates/portal_base.html", "admin/templates/base.html"):
            with self.subTest(page=page):
                head = _read(page)[:2000]
                self.assertIn("logo-mark-sm.svg", head)


class TheScaleHasACaptionRung(unittest.TestCase):
    """Without it the template sweep cannot start."""

    def test_font_2xs_is_defined(self):
        self.assertIn("--font-2xs:", _read("static/css/tokens.css"))

    def test_it_is_the_10px_tier_the_product_actually_uses(self):
        match = re.search(r"--font-2xs:\s*(\d+)px", _read("static/css/tokens.css"))
        self.assertIsNotNone(match)
        self.assertEqual(int(match.group(1)), 10)


if __name__ == "__main__":
    unittest.main()
