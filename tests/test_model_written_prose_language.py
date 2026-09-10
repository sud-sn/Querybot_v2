"""
tests/test_model_written_prose_language.py

The reply language, for the sentences the MODEL writes rather than the ones the
catalogue holds.

Two mechanisms carry the reader's language through this product, and only one
of them is the message catalogue. Everything QueryBot writes itself — cards,
refusals, clarification copy, chart captions, table headers, number and date
formatting — comes from core/i18n.py. But the narrative under an answer, the
"what can you do?" reply, the clarification question when the glossary is too
sparse to build a menu, and the follow-up chips the model fills in are all
written by the LLM, and for those the language is a PROMPT RULE. Translating a
generated English paragraph afterwards would need a second model call and would
still be a translation of an analysis rather than an analysis.

Two prompts carried that rule. Three did not, and each produced English prose
inside an otherwise French answer:

  core/dispatcher.py        _generate_analyst_reply — the answer to "what can
                            you do?" and to anything off-topic, sent to the
                            reader verbatim.
  core/clarification.py     both ambiguity classifiers, whose JSON "question"
                            value IS the clarification card's question.
  core/insight.py           the Tier-2 follow-up gap-fill.

What these tests can and cannot establish: they check that the instruction
reaches the prompt the provider receives. Whether a given model obeys it is a
live-workspace question, not a suite one.
"""
from __future__ import annotations

import unittest

from core import i18n


FRENCH_MARK = "LANGUE :"


class TestTheRuleItself(unittest.TestCase):
    """core/i18n.prompt_language_rule — one home, three shapes."""

    def test_english_asks_for_nothing(self):
        """Every prompt in this product is written in English and tuned that
        way. Adding "answer in English" to one is a change for no gain."""
        for shape in ("narrative", "prose", "json"):
            with self.subTest(shape=shape):
                self.assertEqual(i18n.prompt_language_rule("en", shape=shape), "")

    def test_french_asks_in_french(self):
        for shape in ("narrative", "prose", "json"):
            with self.subTest(shape=shape):
                rule = i18n.prompt_language_rule("fr", shape=shape)
                self.assertTrue(rule.startswith(FRENCH_MARK))
                self.assertIn("français", rule)

    def test_each_shape_protects_what_its_caller_parses(self):
        """A translated label is a response that parses as unlabelled prose."""
        narrative = i18n.prompt_language_rule("fr", shape="narrative")
        self.assertIn("HEADLINE:", narrative)
        self.assertIn("NEXT:", narrative)

        json_rule = i18n.prompt_language_rule("fr", shape="json")
        self.assertIn("JSON", json_rule)
        self.assertIn("clés", json_rule)

    def test_every_shape_protects_the_customers_own_words(self):
        """Translating "Marge brute" into "Gross margin" makes the answer stop
        matching the table under it."""
        for shape in ("narrative", "prose", "json"):
            with self.subTest(shape=shape):
                rule = i18n.prompt_language_rule("fr", shape=shape)
                self.assertIn("colonnes", rule)

    def test_an_unknown_shape_falls_back_rather_than_raising(self):
        self.assertEqual(i18n.prompt_language_rule("fr", shape="nonsense"),
                         i18n.prompt_language_rule("fr", shape="narrative"))

    def test_it_follows_the_active_reader_when_not_told(self):
        token = i18n.activate_language("fr")
        try:
            self.assertTrue(i18n.prompt_language_rule().startswith(FRENCH_MARK))
        finally:
            i18n.deactivate_language(token)
        self.assertEqual(i18n.prompt_language_rule(), "")


def _prompt(lang, build):
    token = i18n.activate_language(lang)
    try:
        return build()
    finally:
        i18n.deactivate_language(token)


class TestEveryModelWrittenSurfaceCarriesIt(unittest.TestCase):

    BRIEF = {
        "mode": "ranking", "row_count": 3,
        "columns": {"WHS_NM": "text", "REVENUE_AMT": "numeric"},
        "category_breakdown": {"label_column": "WHS_NM", "value_column": "REVENUE_AMT",
                               "top_5": [{"label": "Halifax", "value": 900.0}],
                               "category_count": 3},
        "numeric_summaries": {},
    }

    def _answer_narrative(self, lang):
        from core.insight import build_insight_prompt_from_contract

        return _prompt(lang, lambda: build_insight_prompt_from_contract(
            {"action": "explain", "question": "revenue by warehouse",
             "data_brief": self.BRIEF})[0])

    def _prior_period(self, lang):
        from core.period_comparison import build_period_comparison_narrative_prompt

        return _prompt(lang, lambda: build_period_comparison_narrative_prompt(
            self.BRIEF, self.BRIEF, "q", "2026-01", "2025-12")[0])

    def _conversational(self, lang):
        import core.dispatcher as dispatcher

        captured = {}

        async def _spy(system, user, *a, **kw):
            captured["system"] = system
            return ("QueryBot can help with…", 1, 1)

        from unittest.mock import patch

        def _run():
            import asyncio

            with patch.object(dispatcher, "llm_complete", new=_spy), \
                 patch.object(dispatcher, "resolve_provider",
                              return_value=("openai", "m", "k", {})), \
                 patch.object(dispatcher, "_looks_like_data_request",
                              return_value=False), \
                 patch.object(dispatcher, "_build_analyst_context",
                              return_value="SALES"):
                asyncio.run(dispatcher._generate_analyst_reply(
                    "what can you do?", "acct", {}))
            return captured.get("system", "")

        return _prompt(lang, _run)

    def _ambiguity(self, lang, constrained):
        """Drive the real classifier and return the system prompt it sent.

        The first version of these two asserted that "prompt_language_rule"
        appeared in the function's SOURCE -- the exact shape this sweep spent
        its time removing. It passes with the call present and the result
        discarded, and it cannot tell an English prompt from a French one.
        """
        import asyncio
        from unittest.mock import AsyncMock, patch

        import core.clarification as clarification

        seen = {}

        def _respond(system, user, *a, **kw):
            seen["system"] = system
            return ('{"status":"AMBIGUOUS","question":"Quel indicateur ?",'
                    '"option_ids":["t1","t2"]}', 1, 1)

        terms = [{"id": "t1", "term": "gross margin", "definition": "d1"},
                 {"id": "t2", "term": "net margin", "definition": "d2"}]

        def _run():
            with patch("core.llm.llm_complete", new=AsyncMock(side_effect=_respond)):
                if constrained:
                    asyncio.run(clarification._llm_ambiguity_check_constrained(
                        "acct", "show me margin", "ctx", "openai", "m", "k", {},
                        candidate_terms=terms))
                else:
                    asyncio.run(clarification._llm_ambiguity_check(
                        "show me margin", "ctx", "openai", "m", "k", {}))
            return seen.get("system", "")

        return _prompt(lang, _run)

    def test_the_answer_narrative_carries_it(self):
        self.assertIn(FRENCH_MARK, self._answer_narrative("fr"))
        self.assertNotIn(FRENCH_MARK, self._answer_narrative("en"))

    def test_the_prior_period_narrative_carries_it(self):
        self.assertIn(FRENCH_MARK, self._prior_period("fr"))
        self.assertNotIn(FRENCH_MARK, self._prior_period("en"))

    def test_the_conversational_reply_carries_it(self):
        """The answer to "what can you do?" — the most likely first thing a
        new reader ever asks."""
        self.assertIn(FRENCH_MARK, self._conversational("fr"))
        self.assertNotIn(FRENCH_MARK, self._conversational("en"))

    def test_and_it_still_protects_the_routing_sentinel(self):
        """A French reply that translated PROCEED_TO_QUERY would stop matching
        the check that routes a real data question into the SQL pipeline."""
        system = self._conversational("fr")
        self.assertIn("PROCEED_TO_QUERY", system)
        self.assertIn("that exact token, in", system)

    def test_both_ambiguity_classifiers_carry_it(self):
        """The JSON "question" value IS the clarification card's question, so
        it follows the reader like the card around it."""
        for constrained in (True, False):
            with self.subTest(constrained=constrained):
                self.assertIn(FRENCH_MARK, self._ambiguity("fr", constrained))
                self.assertNotIn(FRENCH_MARK, self._ambiguity("en", constrained))

    def test_and_the_json_contract_is_protected_in_the_instruction(self):
        """A translated "AMBIGUOUS" or a translated key is a reply the parser
        reads as CLEAR, which silently turns off clarification."""
        prompt = self._ambiguity("fr", constrained=True)
        rule = prompt[prompt.index(FRENCH_MARK):]
        self.assertIn("JSON", rule)
        self.assertIn("anglais", rule)

    def test_the_classifier_still_returns_its_question(self):
        """The control: a prompt change that broke parsing would satisfy every
        assertion above."""
        import asyncio
        from unittest.mock import AsyncMock, patch

        import core.clarification as clarification

        async def _respond(*a, **kw):
            return ('{"status": "AMBIGUOUS", "question": "Quel indicateur ?"}', 1, 1)

        token = i18n.activate_language("fr")
        try:
            with patch("core.llm.llm_complete", new=AsyncMock(side_effect=_respond)):
                ambiguous, question, _ = asyncio.run(
                    clarification._llm_ambiguity_check(
                        "show me margin", "ctx", "openai", "m", "k", {}))
        finally:
            i18n.deactivate_language(token)
        self.assertTrue(ambiguous)
        self.assertEqual(question, "Quel indicateur ?")


class TestTheFollowUpGapFill(unittest.TestCase):
    """Tier 2. Tier 1's chips carry a French label and an English question
    because one catalogue id renders twice; the model's chips cannot do that,
    so it is asked for both halves."""

    BRIEF = {"row_count": 6,
             "columns": {"WHS_NM": "text", "REVENUE_AMT": "numeric"},
             "category_breakdown": {"label_column": "WHS_NM",
                                    "value_column": "REVENUE_AMT",
                                    "top_5": [{"label": "Halifax", "value": 100.0}],
                                    "category_count": 3}}

    def _run(self, lang, reply):
        import asyncio
        from unittest.mock import AsyncMock, patch

        import core.compliance.policy_engine as policy_engine
        import core.insight as insight
        import core.llm as llm

        seen = {}

        def _respond(system, user, *a, **kw):
            seen["system"] = system
            return (reply, 1, 1)

        token = i18n.activate_language(lang)
        try:
            with patch.object(llm, "llm_complete", new=AsyncMock(side_effect=_respond)), \
                 patch.object(llm, "resolve_provider",
                              return_value=("openai", "m", "k", {})), \
                 patch.object(policy_engine, "is_regulated", return_value=False):
                chips = asyncio.run(insight.generate_followup_suggestions(
                    brief=self.BRIEF, question="revenue by warehouse",
                    result_scope={}, db_cfg={}, account_id="acct", signals=[]))
        finally:
            i18n.deactivate_language(token)
        return seen.get("system", ""), chips

    PAIRED = ('[{"question": "Who leads on revenue?", '
              '"label": "Qui est en tête sur le chiffre d\'affaires ?"}]')
    PLAIN = '["Who leads on revenue?"]'

    def test_a_french_reader_makes_it_ask_for_both_halves(self):
        system, _ = self._run("fr", self.PAIRED)
        self.assertIn("Return an array of OBJECTS", system)
        self.assertIn("French", system)

    def test_an_english_reader_leaves_the_tuned_prompt_alone(self):
        """This prompt is tuned. English is the language it was tuned in, so
        the extra instruction is added only when it is needed."""
        system, _ = self._run("en", self.PLAIN)
        self.assertNotIn("Return an array of OBJECTS", system)

    def test_the_label_is_french_and_the_question_english(self):
        _, chips = self._run("fr", self.PAIRED)
        self.assertEqual(len(chips), 1)
        self.assertEqual(chips[0]["question"], "Who leads on revenue?")
        self.assertIn("chiffre d'affaires", chips[0]["label"])

    def test_a_model_that_ignores_the_instruction_still_produces_chips(self):
        """Parsing strictly would turn a wording drift into an empty chip row.
        English chips are the old behaviour and better than none."""
        _, chips = self._run("fr", self.PLAIN)
        self.assertEqual(len(chips), 1)
        self.assertEqual(chips[0]["question"], "Who leads on revenue?")
        self.assertEqual(chips[0]["label"], "Who leads on revenue?")

    def test_an_object_reply_to_an_english_reader_also_works(self):
        _, chips = self._run("en", self.PAIRED)
        self.assertEqual(chips[0]["question"], "Who leads on revenue?")

    def test_an_object_with_only_a_label_still_yields_a_question(self):
        _, chips = self._run("fr", '[{"label": "Qui mène ?"}]')
        self.assertEqual(chips[0]["question"], "Qui mène ?")

    def test_junk_still_costs_the_chips_and_not_the_answer(self):
        _, chips = self._run("fr", "not json at all")
        self.assertEqual(chips, [])


class TestTheClarificationFallbacksAreTranslated(unittest.TestCase):
    """Three English literals stood behind the model's own text — the strings
    used when it returns AMBIGUOUS with no question, when the reply is in the
    legacy text shape, and when the glossary alone shows several readings."""

    IDS = ("reply.clarify.which_one", "reply.clarify.need_more_context",
           "reply.clarify.several_readings")

    def test_each_resolves_in_both_languages(self):
        for msg_id in self.IDS:
            for lang in i18n.SUPPORTED_LANGUAGES:
                with self.subTest(msg_id=msg_id, lang=lang):
                    resolved = i18n.lookup(msg_id, lang)
                    self.assertTrue(resolved.strip())
                    self.assertNotEqual(resolved, msg_id)

    def test_and_the_french_is_not_a_copy_of_the_english(self):
        for msg_id in self.IDS:
            with self.subTest(msg_id=msg_id):
                self.assertNotEqual(i18n.lookup(msg_id, "en"),
                                    i18n.lookup(msg_id, "fr"))


if __name__ == "__main__":
    unittest.main()
