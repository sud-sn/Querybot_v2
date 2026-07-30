"""
core/query_router.py::should_attempt_cache_followup -- widens the narrow
regex gate (should_route_to_result_cache) so the metadata-only LLM planner
gets a second opinion on phrasings its fixed trigger-word vocabulary
misses, without paying LLM latency on a fresh session or an obviously-new
question.
"""

import unittest

from core.query_router import should_attempt_cache_followup, should_route_to_result_cache


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
        # should_route_to_result_cache's trigger words.
        question = "drill into North"
        self.assertFalse(should_route_to_result_cache(question, True))
        self.assertTrue(should_attempt_cache_followup(question, True))

    def test_obviously_new_long_question_does_not_get_a_second_opinion(self):
        # A fully-formed new business question must still go straight to
        # fresh SQL generation even with a cached result active -- the
        # widened gate is a narrow middle ground, not "always ask the LLM."
        question = "what is the total revenue by region for the last quarter across all warehouses"
        self.assertFalse(should_route_to_result_cache(question, True))
        self.assertFalse(should_attempt_cache_followup(question, True))

    def test_short_two_word_message_gets_a_second_opinion(self):
        question = "East only"
        self.assertFalse(should_route_to_result_cache(question, True))
        self.assertTrue(should_attempt_cache_followup(question, True))

    def test_starter_worded_question_treated_as_new_even_when_short(self):
        # "show" is a new-query starter word; even a short message
        # beginning with it should not get the second-opinion treatment.
        question = "show pharmacy counts"
        self.assertFalse(should_attempt_cache_followup(question, True))

    def test_cached_col_names_forwarded_to_regex_gate(self):
        question = "what is the average of TOTAL_REVENUE"
        self.assertTrue(
            should_attempt_cache_followup(question, True, cached_col_names=["TOTAL_REVENUE"])
        )


if __name__ == "__main__":
    unittest.main()
