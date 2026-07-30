"""
core/query_router.py::should_attempt_cache_followup -- widens the narrow
regex gate (should_route_to_result_cache) so the metadata-only LLM planner
gets a second opinion on phrasings its fixed trigger-word vocabulary
misses, without paying LLM latency on a fresh session, an obviously-new
question, or a short-but-genuinely-new question that merely happens to be
short (word count/starter-word alone can't distinguish "drill into North"
from "Profit by warehouse" -- both are three words, neither starts with a
starter word). A positive signal is required instead: a deictic reference,
a cached column mention, or a cached row VALUE mention.
"""

import unittest

from core.query_router import (
    _question_references_cached_value,
    should_attempt_cache_followup,
    should_route_to_result_cache,
)

_PHARMACY_ROWS = [
    {"PHARMACY_NAME": "North", "TOTAL_FILLS": 120},
    {"PHARMACY_NAME": "South", "TOTAL_FILLS": 90},
]
_PHARMACY_COLS = ["PHARMACY_NAME", "TOTAL_FILLS"]


class QuestionReferencesCachedValueTests(unittest.TestCase):

    def test_no_rows_returns_false(self):
        self.assertFalse(_question_references_cached_value("drill into North", None))
        self.assertFalse(_question_references_cached_value("drill into North", []))

    def test_matching_value_is_found(self):
        self.assertTrue(_question_references_cached_value("drill into North", _PHARMACY_ROWS))

    def test_unrelated_text_is_not_found(self):
        self.assertFalse(_question_references_cached_value("profit by warehouse", _PHARMACY_ROWS))

    def test_case_and_punctuation_insensitive(self):
        self.assertTrue(_question_references_cached_value("what about NORTH?", _PHARMACY_ROWS))

    def test_non_scalar_values_are_skipped_without_erroring(self):
        rows = [{"PHARMACY_NAME": "North", "TAGS": ["a", "b"], "META": {"x": 1}}]
        self.assertTrue(_question_references_cached_value("drill into North", rows))


class ShouldAttemptCacheFollowupTests(unittest.TestCase):

    def test_no_cached_result_never_attempts_followup(self):
        # Must never fire on a fresh session, regardless of phrasing --
        # this is what keeps a brand-new conversation at zero extra latency.
        self.assertFalse(should_attempt_cache_followup("drill into North", False))
        self.assertFalse(should_attempt_cache_followup("what is total revenue", False))

    def test_regex_match_still_routes_directly(self):
        # Anything the existing regex gate already catches must be
        # unaffected -- this function is additive, not a replacement.
        question = "from these results, show only the top 5"
        self.assertTrue(should_route_to_result_cache(question, True))
        self.assertTrue(should_attempt_cache_followup(question, True))

    def test_short_novel_phrasing_the_regex_gate_misses_gets_a_second_opinion(self):
        # The live-reported gap: "drill into X" matches none of
        # should_route_to_result_cache's trigger words. Requires the
        # positive signal (here, "North" as an actual cached value).
        question = "drill into North"
        self.assertFalse(should_route_to_result_cache(question, True))
        self.assertTrue(
            should_attempt_cache_followup(question, True, cached_rows=_PHARMACY_ROWS)
        )

    def test_novel_phrasing_without_any_positive_signal_does_not_get_a_second_opinion(self):
        # Without column/value/deictic evidence, a short novel phrase is
        # indistinguishable from a short new question -- must not attempt.
        question = "drill into North"
        self.assertFalse(should_attempt_cache_followup(question, True))

    def test_obviously_new_long_question_does_not_get_a_second_opinion(self):
        # A fully-formed new business question must still go straight to
        # fresh SQL generation even with a cached result active -- the
        # widened gate is a narrow middle ground, not "always ask the LLM."
        question = "what is the total revenue by region for the last quarter across all warehouses"
        self.assertFalse(should_route_to_result_cache(question, True))
        self.assertFalse(
            should_attempt_cache_followup(question, True, cached_rows=_PHARMACY_ROWS)
        )

    def test_short_elliptical_value_reference_gets_a_second_opinion(self):
        question = "South only"
        self.assertFalse(should_route_to_result_cache(question, True))
        self.assertTrue(
            should_attempt_cache_followup(question, True, cached_rows=_PHARMACY_ROWS)
        )

    def test_starter_worded_question_treated_as_new_even_when_short(self):
        # "show" is a new-query starter word; even a short message
        # beginning with it should not get the second-opinion treatment.
        question = "show pharmacy counts"
        self.assertFalse(
            should_attempt_cache_followup(question, True, cached_rows=_PHARMACY_ROWS)
        )

    def test_cached_col_names_forwarded_to_regex_gate(self):
        question = "what is the average of TOTAL_REVENUE"
        self.assertTrue(
            should_attempt_cache_followup(question, True, cached_col_names=["TOTAL_REVENUE"])
        )

    def test_deictic_reference_gets_a_second_opinion_with_no_column_or_value_match(self):
        # "what" is itself a new-query starter word, so this deliberately
        # avoids it -- the point is the deictic word alone ("that") is
        # sufficient signal even with no column/value match at all.
        question = "just that one"
        self.assertFalse(should_route_to_result_cache(question, True))
        self.assertTrue(should_attempt_cache_followup(question, True))

    # -- Reviewer-reported gap: short new questions mistaken for follow-ups --

    def test_short_new_question_sharing_no_cached_vocabulary_is_not_attempted(self):
        # "Revenue yesterday?" -- genuinely short and non-starter-worded,
        # exactly like a real follow-up, but names nothing in the active
        # cached result (a pharmacy-fills result with no revenue/date
        # column and no matching cached value).
        question = "Revenue yesterday?"
        self.assertFalse(should_route_to_result_cache(question, True, cached_col_names=_PHARMACY_COLS))
        self.assertFalse(
            should_attempt_cache_followup(
                question, True, cached_col_names=_PHARMACY_COLS, cached_rows=_PHARMACY_ROWS,
            )
        )

    def test_short_new_question_naming_an_unrelated_dimension_is_not_attempted(self):
        # "Profit by warehouse" -- same length and shape as "drill into
        # North", but "warehouse" is neither a cached column nor a cached
        # value for this result.
        question = "Profit by warehouse"
        self.assertFalse(should_route_to_result_cache(question, True, cached_col_names=_PHARMACY_COLS))
        self.assertFalse(
            should_attempt_cache_followup(
                question, True, cached_col_names=_PHARMACY_COLS, cached_rows=_PHARMACY_ROWS,
            )
        )


if __name__ == "__main__":
    unittest.main()
