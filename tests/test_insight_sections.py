"""
tests/test_insight_sections.py

The narrative parser threw away everything it did not recognise.

`parse_insight_response` was an if/elif chain over four labels with no else, so
any line not opening with HEADLINE:/BODY:/DETAIL:/NEXT: or a bullet was dropped
without a trace. Two consequences, both invisible from the outside:

  - a BODY written across several lines kept only its first line, so a
    three-sentence explanation arrived as one sentence;
  - there was no way to answer a question with two parts. A reader asking
    "are they priced at a premium, and how much revenue depends on them?"
    got one flat body, because the contract had nowhere to put the second
    answer.

Continuation lines now attach to the open block, and `SECTION: <title>` opens a
titled one. The four original keys keep their exact old meaning, which is what
lets every existing renderer ignore `sections` and lose nothing.
"""

import unittest

from core.insight import parse_insight_response


class ContinuationLinesSurvive(unittest.TestCase):

    def test_a_multi_line_body_keeps_every_line(self):
        parsed = parse_insight_response(
            "HEADLINE: Controlled substances are 10.6% of revenue\n"
            "BODY: Pricing is flat across schedules.\n"
            "CIII prices marginally below non-controlled.\n"
            "So the dependency is volume mix, not a pricing lever."
        )
        self.assertIn("Pricing is flat", parsed["body"])
        self.assertIn("CIII prices marginally below", parsed["body"])
        self.assertIn("volume mix", parsed["body"])

    def test_the_old_behaviour_would_have_kept_only_the_first(self):
        """Guards the fix: the later lines are the ones that used to vanish."""
        parsed = parse_insight_response("BODY: first\nsecond\nthird")
        self.assertEqual(parsed["body"], "first second third")

    def test_blank_lines_do_not_become_content(self):
        parsed = parse_insight_response("HEADLINE: h\nBODY: one\n\n\ntwo")
        self.assertEqual(parsed["body"], "one two")


class TitledSections(unittest.TestCase):

    TWO_PART = (
        "HEADLINE: 10.6% of revenue, at flat pricing\n"
        "SECTION: Are controlled compounds priced at a premium?\n"
        "BODY: Essentially no.\n"
        "- CIII prices 0.07% below non-controlled\n"
        "- CV carries a 1.31% premium\n"
        "SECTION: How much revenue depends on them?\n"
        "BODY: CIII and CV together are 10.61% of YTD revenue.\n"
        "- Tracks their 10.59% share of Rx volume\n"
        "NEXT: Break the CIII share down by prescriber"
    )

    def test_each_part_of_the_question_gets_its_own_titled_block(self):
        parsed = parse_insight_response(self.TWO_PART)
        titles = [s["title"] for s in parsed["sections"]]
        self.assertEqual(titles, [
            "Are controlled compounds priced at a premium?",
            "How much revenue depends on them?",
        ])

    def test_body_and_bullets_attach_to_the_section_they_follow(self):
        parsed = parse_insight_response(self.TWO_PART)
        first, second = parsed["sections"]
        self.assertEqual(first["body"], "Essentially no.")
        self.assertEqual(len(first["bullets"]), 2)
        self.assertIn("10.61% of YTD revenue", second["body"])
        self.assertEqual(second["bullets"], ["Tracks their 10.59% share of Rx volume"])

    def test_the_headline_and_next_step_stay_at_the_top_level(self):
        """One NEXT per answer: a list of them is a menu, not a suggestion."""
        parsed = parse_insight_response(self.TWO_PART)
        self.assertEqual(parsed["headline"], "10.6% of revenue, at flat pricing")
        self.assertEqual(parsed["next_step"], "Break the CIII share down by prescriber")

    def test_section_bullets_do_not_leak_into_the_top_level_list(self):
        """Or every renderer that shows `bullets` would print them twice."""
        parsed = parse_insight_response(self.TWO_PART)
        self.assertEqual(parsed["bullets"], [])


class TheOldContractIsUntouched(unittest.TestCase):
    """Every existing renderer reads the four original keys and must not shift."""

    def test_a_single_part_answer_parses_exactly_as_before(self):
        parsed = parse_insight_response(
            "HEADLINE: Revenue rose 12%\n"
            "BODY: Driven by the EMEA region.\n"
            "DETAIL:\n- EMEA up 30%\n- APAC flat\n"
            "NEXT: Break EMEA down by country"
        )
        self.assertEqual(parsed["headline"], "Revenue rose 12%")
        self.assertEqual(parsed["body"], "Driven by the EMEA region.")
        self.assertEqual(parsed["bullets"], ["EMEA up 30%", "APAC flat"])
        self.assertEqual(parsed["next_step"], "Break EMEA down by country")

    def test_a_single_part_answer_carries_no_sections(self):
        parsed = parse_insight_response("HEADLINE: h\nBODY: b")
        self.assertEqual(parsed["sections"], [])

    def test_the_unparseable_fallback_still_fires(self):
        parsed = parse_insight_response("just some prose with no labels at all")
        self.assertEqual(parsed["body"], "just some prose with no labels at all")
        self.assertTrue(parsed["headline"])

    def test_the_fallback_does_not_clobber_a_sections_only_answer(self):
        """No headline, no top-level body — but the answer is not empty."""
        parsed = parse_insight_response("SECTION: Part one\nBODY: an answer")
        self.assertEqual(parsed["body"], "")
        self.assertEqual(len(parsed["sections"]), 1)
        self.assertEqual(parsed["sections"][0]["body"], "an answer")


class TheContractIsAdvertisedToTheModel(unittest.TestCase):
    """A format the parser accepts but the prompt never mentions is dead."""

    def test_the_prompt_offers_sections_for_multi_part_questions(self):
        from core.insight import build_insight_prompt_from_contract

        system, _user = build_insight_prompt_from_contract({
            "action": "analyze", "question": "a and b?", "mode": "table",
        })
        self.assertIn("SECTION:", system)
        self.assertIn("TWO OR MORE distinct things", system)

    def test_the_prompt_says_a_multi_line_body_is_kept(self):
        from core.insight import build_insight_prompt_from_contract

        system, _user = build_insight_prompt_from_contract({
            "action": "analyze", "question": "q", "mode": "table",
        })
        self.assertIn("BODY may run to several lines", system)


if __name__ == "__main__":
    unittest.main()
