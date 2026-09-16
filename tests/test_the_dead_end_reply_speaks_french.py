"""
tests/test_the_dead_end_reply_speaks_french.py

The last thing a reader sees when no SQL could be written for their question,
in both languages.

It was the one hard-coded English block left on a user-facing path in
core/query_pipeline.py — sitting four lines below a clarification that had
always been translated, which is why nobody noticed. The relevance gate added
alongside this (core.clarification.has_ambiguity_signal) makes it the reply
rather than the fallback whenever the glossary has nothing to ask about, so a
French reader now reaches it far more often than before.

It also could not be tested where it stood: reaching that line needs a
warehouse, a knowledge base, a socket and a model. So the message moved into
core.failure_messages.cannot_generate_message, which is a function these tests
execute directly, and the branch now just calls it with the reader's language.

The closest-terms line reuses fail.v.suggestions rather than carrying a second
spelling of the same sentence — the validator refusals already say it, in both
languages.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.failure_messages import cannot_generate_message  # noqa: E402

CLOSEST = ["purchase order quantity", "scrap quantity"]


class TheDeadEndReplyIsTranslated(unittest.TestCase):

    def test_neither_language_is_missing_and_they_are_not_the_same(self):
        english = cannot_generate_message(CLOSEST, lang="en")
        french = cannot_generate_message(CLOSEST, lang="fr")
        self.assertTrue(english.strip())
        self.assertTrue(french.strip())
        self.assertNotEqual(english, french)
        # A catalogue id rendering as itself means the key is missing.
        self.assertNotIn("terminal.cannot_generate", french)
        self.assertNotIn("fail.v.suggestions", french)

    def test_not_one_line_of_it_stays_in_english(self):
        """Line by line. This is a six-line message, and one English line
        among five translated ones still reads as broken."""
        def _lines(text):
            return {line.strip() for line in text.splitlines() if line.strip()}

        shared = _lines(cannot_generate_message(CLOSEST, lang="en")) & _lines(
            cannot_generate_message(CLOSEST, lang="fr")
        )
        self.assertEqual(shared, set(), "English lines survive in the French reply")

    def test_the_readers_own_vocabulary_rides_through_untranslated(self):
        """The closest terms are the tenant's field and metric names. A
        translated one matches nothing in their data."""
        for lang in ("en", "fr"):
            with self.subTest(lang=lang):
                text = cannot_generate_message(CLOSEST, lang=lang)
                for term in CLOSEST:
                    self.assertIn(term, text)

    def test_with_nothing_close_the_line_is_dropped_not_left_empty(self):
        """suggest_closest_terms returns [] often enough that an empty
        "Closest known terms in your data: ." would be the common case, not
        the rare one — so the whole sentence has to go, not just its list."""
        from core.i18n import t as _t

        for lang in ("en", "fr"):
            with self.subTest(lang=lang):
                bare = cannot_generate_message([], lang=lang)
                with_terms = cannot_generate_message(CLOSEST, lang=lang)
                # The sentence that introduces the list, in this language,
                # taken from the catalogue rather than spelled out here.
                lead = _t("fail.v.suggestions", lang=lang, terms="\x00").split("\x00")[0].strip()
                self.assertTrue(lead)
                self.assertIn(lead, with_terms)
                self.assertNotIn(lead, bare)
                self.assertNotIn("  \n", bare)
                self.assertNotIn("\n\n\n", bare)
                # The rest of the advice survives losing the suggestions.
                self.assertTrue(len(bare.splitlines()) >= 5)
                self.assertLess(len(bare), len(with_terms))

    def test_none_is_treated_as_no_suggestions_rather_than_crashing(self):
        """The caller passes suggest_closest_terms' output straight through,
        and that function is documented as never raising — so this one must
        not either."""
        self.assertTrue(cannot_generate_message(None, lang="fr").strip())
        self.assertTrue(cannot_generate_message(lang=None).strip())

    def test_it_still_tells_the_reader_what_to_do(self):
        """A translated dead end that stopped being actionable would pass
        every test above."""
        english = cannot_generate_message(CLOSEST, lang="en")
        self.assertIn("Try rephrasing", english)
        self.assertIn("Metric Registry", english)
        self.assertEqual(english.count("  • "), 3)

        french = cannot_generate_message(CLOSEST, lang="fr")
        self.assertIn("reformuler", french)
        self.assertIn("registre des indicateurs", french)
        self.assertEqual(french.count("  • "), 3)


if __name__ == "__main__":
    unittest.main()
