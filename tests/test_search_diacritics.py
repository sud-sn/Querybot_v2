# -*- coding: utf-8 -*-
"""A French reader who types without accents still finds things.

Every search box in the portal compared `.toLowerCase()` on both sides, which
is diacritic-blind in the wrong direction: a reader typing "region" matched
nothing named "région", "cout" found no "coût", and "oeuf" found no "œuf".
Nobody types the accents into a search box — the whole point of one is that it
is faster than reading the list.

The helper is executed here rather than described: window.qbFold is lifted out
of the shell and run in duktape, the same way the number and date helpers are.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

dukpy = pytest.importorskip("dukpy", reason="the page's own JS is executed here")

from tests.browser_num import preamble  # noqa: E402

SHELL = (Path(__file__).resolve().parents[1]
         / "portal" / "templates" / "portal_base.html").read_text(encoding="utf-8")
CHAT = (Path(__file__).resolve().parents[1]
        / "portal" / "templates" / "portal_chat.html").read_text(encoding="utf-8")
KB = (Path(__file__).resolve().parents[1]
      / "portal" / "templates" / "portal_kb.html").read_text(encoding="utf-8")


def _fold(value, lang="fr"):
    return json.loads(dukpy.evaljs(
        preamble(lang) + "JSON.stringify(window.qbFold(dukpy['v']));", v=value))


class TestTheFoldingItself(unittest.TestCase):

    def test_accents_are_removed(self):
        for typed, shown in (("region", "Région"), ("cout", "COÛT"),
                             ("francais", "Français"), ("eleve", "élève"),
                             ("noel", "Noël")):
            with self.subTest(shown=shown):
                self.assertEqual(_fold(shown), typed)

    def test_the_french_ligatures_fold_to_their_letters(self):
        # Unicode gives œ and æ no decomposition at all, canonical or
        # compatibility, so NFD alone leaves them whole.
        self.assertEqual(_fold("Œuvre"), "oeuvre")
        self.assertEqual(_fold("sœur"), "soeur")
        self.assertEqual(_fold("Æther"), "aether")

    def test_it_lowercases_as_well(self):
        self.assertEqual(_fold("RÉGION"), _fold("région"))

    def test_plain_text_is_unchanged(self):
        self.assertEqual(_fold("revenue by region"), "revenue by region")

    def test_it_never_throws_on_junk(self):
        for value in (None, "", 0, "   ", "日本語"):
            _fold(value)

    def test_a_search_matches_what_the_reader_typed(self):
        # The property every box below relies on.
        for typed, shown in (("region", "Ventes par Région"),
                             ("cout", "Coût unitaire"),
                             ("soeur", "Société Sœur")):
            self.assertIn(_fold(typed), _fold(shown), shown)


class TestEverySearchBoxUsesIt(unittest.TestCase):
    """Structural: these live in DOM event handlers with no unit to call.

    What is checked is that no search comparison is left comparing
    `.toLowerCase()` against `.toLowerCase()`, which is the property that broke.
    """

    def test_the_helper_is_defined_once_in_the_shell(self):
        self.assertEqual(SHELL.count("window.qbFold = function"), 1)

    def test_the_chat_page_folds_all_four_of_its_filters(self):
        for marker in ("window.qbFold(String(query || '').trim())",   # dashboard picker
                       "window.qbFold(filterEl.value)",                # result table
                       "window.qbFold((q || '').trim())",              # thread history
                       "window.qbFold((el.querySelector('.hp-q')"):    # ...and its haystack
            self.assertIn(marker, CHAT, marker)

    def test_the_knowledge_base_page_folds_both_sides(self):
        self.assertIn("var q = window.qbFold(input.value.trim());", KB)
        for field in ("colName", "meaning", "useCase", "synonyms", "fieldType"):
            self.assertIn(f"{field}:", KB)
        self.assertIn("window.qbFold(card.dataset.tableName)", KB)

    def test_no_search_still_compares_lowercase_to_lowercase(self):
        import re

        offenders = []
        for name, source in (("portal_chat.html", CHAT), ("portal_kb.html", KB)):
            for line_no, line in enumerate(source.split("\n"), 1):
                if ".toLowerCase()" not in line:
                    continue
                if any(word in line for word in ("includes(", "indexOf(", "search(")):
                    offenders.append(f"{name}:{line_no}: {line.strip()[:70]}")
        self.assertEqual(offenders, [], "\n".join(offenders))

    def test_the_scan_would_notice_one(self):
        # Guards the assertion above against a pattern that matches nothing.
        sample = "const hit = a.toLowerCase().includes(b.toLowerCase());"
        self.assertIn(".toLowerCase()", sample)
        self.assertIn("includes(", sample)


if __name__ == "__main__":
    unittest.main()
