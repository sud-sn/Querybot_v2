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


if __name__ == "__main__":
    unittest.main()
