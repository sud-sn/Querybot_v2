"""
tests/test_compound_question_answering.py

A two-part question got half an answer, or none.

`detect_compound_question` splits on five literal joiners — ";", "and also",
"as well as", "and then", "plus". None of them is how people actually bundle
two asks. "Are controlled compounds priced at a premium, and how much of our
revenue depends on them?" joins with a bare "and" after a comma and was
invisible to the detector, so it went to the pipeline as ONE question and one
SQL statement answered whichever half it happened to latch onto.

And when the detector DID fire, the reply was "which should I run first?" and
no answer at all: the reader retypes, and the half they did not pick is
discarded — there is no remainder queue anywhere in the repo.

Both halves of that had to change together. Improving detection alone would
have routed MORE questions into a dead end, which is a regression however good
the detection is.
"""

import unittest
from pathlib import Path

from core.conversational import detect_compound_question


class TheShapePeopleActuallyUse(unittest.TestCase):
    """A bare "and" after a comma, joining two questions."""

    def test_the_two_part_analytical_question_is_detected(self):
        split = detect_compound_question(
            "are controlled compounds priced at a premium, "
            "and how much of our revenue depends on them?"
        )
        self.assertIsNotNone(split)
        left, right = split
        self.assertIn("priced at a premium", left)
        self.assertIn("how much of our revenue", right)

    def test_it_survives_a_long_preamble(self):
        split = detect_compound_question(
            "Break down revenue and pricing by DEA schedule - are they priced "
            "at a premium, and how much of our revenue depends on them?"
        )
        self.assertIsNotNone(split)
        self.assertIn("how much of our revenue", split[1])

    def test_two_plain_questions(self):
        split = detect_compound_question(
            "what is our revenue, and what is our margin?"
        )
        self.assertEqual(split, ("what is our revenue", "what is our margin"))


class OneIntentIsNotTwoQuestions(unittest.TestCase):
    """A false positive hijacks a valid query, so the comma is load-bearing."""

    def test_a_conjunction_inside_one_ask_is_left_alone(self):
        self.assertIsNone(detect_compound_question("total revenue and tax by region"))

    def test_a_trailing_grouping_clause_is_not_a_second_question(self):
        self.assertIsNone(detect_compound_question("revenue by region, and by product"))

    def test_a_comma_and_that_joins_two_nouns_is_not_split(self):
        """The right half has to READ like a question, not just follow a comma."""
        self.assertIsNone(detect_compound_question("sales for Q3, and the margin trend"))

    def test_the_existing_joiners_still_behave(self):
        self.assertEqual(
            detect_compound_question("show revenue by region and also top 10 customers"),
            ("show revenue by region", "top 10 customers"),
        )

    def test_a_short_half_is_still_rejected(self):
        self.assertIsNone(detect_compound_question("revenue by region, and how?"))


class TheFirstHalfIsAnswered(unittest.TestCase):
    """Detection without this is a regression: more questions, same dead end."""

    def test_the_handler_enqueues_the_first_half_and_does_not_just_ask(self):
        import inspect
        import core.dispatcher as dispatcher

        source = inspect.getsource(dispatcher)
        marker = "# ── Compound question"
        self.assertIn(marker, source)
        block = source[source.index(marker):]
        block = block[:block.index("_enqueue_query(bg, account_id, event, adapter, text")]

        self.assertIn("_enqueue_query(", block,
                      "the first half must actually run")
        self.assertIn("_q1", block)
        self.assertNotIn("Which should I run first", block,
                         "the dead-end prompt must be gone")


class OneResultSetIsNotTwoQuestions(unittest.TestCase):
    """The comma-and joiner was broad enough to tear a single analytical
    question in half.

    "Compare 2025 against 2024 by revenue category which grew, which shrank,
    and what did each contribute to the overall change?" splits on ", and" and
    the right half opens with "what", so every earlier gate passed it. But
    "each" ranges over the rows the LEFT half produces. Run alone it asks about
    nothing, and the reader gets a contribution answer with no categories in it.

    The distinction that matters is not "does the right half refer back" -- it
    is whether it refers to a NAMED THING or to the left half's RESULT SET.
    """

    def test_a_set_quantifier_keeps_the_question_whole(self):
        self.assertIsNone(detect_compound_question(
            "Compare 2025 against 2024 by revenue category which grew, "
            "which shrank, and what did each contribute to the overall change?"
        ))

    def test_ordinary_anaphora_still_splits(self):
        """Guards the guard. "them" names controlled compounds, which the left
        half introduced and the session context carries forward -- a separate
        query about them is well defined, so blocking it would cost a real
        split."""
        self.assertIsNotNone(detect_compound_question(
            "are controlled compounds priced at a premium, "
            "and how much of our revenue depends on them?"
        ))

    def test_the_other_set_quantifiers(self):
        for question in (
            "show sales by region, and how did both compare",
            "top 10 customers, and what did the rest contribute",
            "revenue by month, and list them respectively",
            "show margin by product, and which beat the same last year",
        ):
            with self.subTest(question=question):
                self.assertIsNone(detect_compound_question(question))

    def test_the_quantifiers_are_whole_words(self):
        """"each" inside "reach", "both" inside "bothered" -- a substring match
        would silently suppress splits that have no back-reference at all."""
        split = detect_compound_question(
            "what is our revenue, and which regions reach target"
        )
        self.assertIsNotNone(split)


class NoSourceFileCarriesAMangledEscape(unittest.TestCase):
    """A shell heredoc on this machine collapses a doubled backslash before
    Python sees it, so an intended word boundary is written to the file as a
    literal backspace (0x08). The regex still compiles and still runs -- it
    just can never match.

    It has done this three times: the guard above, a comment in core/llm.py,
    and test_production_ui's hardcoded-white check, which was dead from the day
    it was written and passed every run because its offender list was always
    empty. Nothing else in the suite can see this class of damage.
    """

    def test_no_python_file_contains_a_literal_backspace(self):
        root = Path(__file__).resolve().parents[1]
        skip = {".git", "__pycache__", ".venv", "venv", "node_modules", "build", "dist"}
        offenders = []
        for path in root.rglob("*.py"):
            if skip & set(path.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if chr(8) in text:
                line = next(n for n, l in enumerate(text.splitlines(), 1)
                            if chr(8) in l)
                offenders.append(f"{path.relative_to(root)}:{line}")
        self.assertEqual(
            offenders, [],
            "literal backspace where a word boundary was intended: "
            + ", ".join(offenders),
        )

if __name__ == "__main__":
    unittest.main()
