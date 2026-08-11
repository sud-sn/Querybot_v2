import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from core import dispatcher
from core.query_pipeline import _graph_entities_for_verified_values


class DataRequestRoutingTests(unittest.TestCase):
    def test_regulated_record_requests_are_data_shaped(self):
        questions = (
            "List patients with prescriptions and include their MRN, date of birth, diagnosis, and payment member identifier.",
            "Show prescription instructions and diagnosis for each patient.",
            "Show the top five prescribers by picked-up prescription count.",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertTrue(dispatcher._looks_like_data_request(question))

    def test_obvious_general_request_is_not_data_shaped(self):
        self.assertFalse(dispatcher._looks_like_data_request("Tell me a joke about summer."))

    def test_order_change_and_order_count_questions_are_data_requests(self):
        questions = (
            "which customers have reduced orders recently",
            "what is the total orders placed by each customer show top 10 in the last 2 months",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertTrue(dispatcher._looks_like_data_request(question))

    def test_data_shaped_request_bypasses_llm_classifier(self):
        # _generate_analyst_reply replaces _classify_is_data_question.
        # A data-shaped request must return None (fall through to pipeline)
        # without touching the LLM — the fast-path check _looks_like_data_request
        # should short-circuit it before any llm_complete call.
        with patch.object(dispatcher, "llm_complete") as complete:
            result = asyncio.run(
                dispatcher._generate_analyst_reply(
                    "List patients with prescriptions and include their diagnosis.",
                    "test-account",
                    {},
                )
            )
        self.assertIsNone(result)   # None → fall through to SQL pipeline
        complete.assert_not_called()

    def test_meta_question_reaches_llm_and_returns_its_reply(self):
        # A non-data-shaped question (e.g. "who are you") must NOT be
        # fast-pathed — it has to reach the llm_audit_scope + llm_complete
        # call. This exercises that branch for real (the prior regression
        # test above only ever exercised the fast-path skip, so it never
        # touched llm_audit_scope at all and could not have caught a
        # signature mismatch there — which is exactly what shipped and
        # made every meta/off-topic question silently fall through to SQL
        # generation instead of getting an analyst reply).
        with patch.object(
            dispatcher, "llm_complete",
            new=AsyncMock(return_value=("I'm QueryBot, your data analyst.", 10, 5)),
        ) as complete, patch.object(
            dispatcher, "resolve_provider",
            return_value=("openai", "gpt-4o-mini", "sk-test", {}),
        ):
            result = asyncio.run(
                dispatcher._generate_analyst_reply("who are you?", "test-account", {})
            )
        complete.assert_called_once()
        self.assertEqual(result, "I'm QueryBot, your data analyst.")

    def test_internal_query_handoff_marker_never_leaks_when_wrapped_in_prose(self):
        wrapped = (
            "It looks like a data request.\n\n"
            "Replying with: *PROCEED_TO_QUERY*"
        )
        with patch.object(
            dispatcher, "llm_complete",
            new=AsyncMock(return_value=(wrapped, 10, 5)),
        ), patch.object(
            dispatcher, "resolve_provider",
            return_value=("openai", "gpt-4o-mini", "sk-test", {}),
        ):
            result = asyncio.run(
                dispatcher._generate_analyst_reply(
                    "provide the date differently", "test-account", {},
                )
            )
        self.assertIsNone(result)

    def test_query_offer_and_confirmation_are_recognized_deterministically(self):
        reply = (
            "I can compare recent order volumes with history. "
            "If you'd like to proceed with retrieving this data, let me know."
        )
        self.assertTrue(dispatcher._analyst_reply_offers_query(reply))
        for text in ("yes", "yes proceed with the retrieving", "go ahead", "run it"):
            self.assertTrue(dispatcher._is_query_confirmation(text), text)
        self.assertFalse(dispatcher._is_query_confirmation("use invoice date instead"))

    def test_analyst_offer_resume_uses_the_original_question(self):
        source = (dispatcher.__file__ and open(dispatcher.__file__, encoding="utf-8").read())
        start = source.index('cmeta.get("source") == "analyst_query_offer"')
        block = source[start:start + 2600]
        self.assertIn('pending["original_q"]', block)
        self.assertIn("is_clarification=True", block)
        self.assertLess(block.index('pending["original_q"]'), block.index("is_clarification=True"))

    def test_count_target_choice_is_attached_to_replanned_event(self):
        source = open(dispatcher.__file__, encoding="utf-8").read()
        self.assertIn(
            'cmeta.get("source") in {"metric_date_context", "count_target"}',
            source,
        )
        self.assertIn('raw["_clarification_selected_option"] = dict(match)', source)


class VerifiedValueGraphTests(unittest.TestCase):
    def test_verified_status_forces_owning_fact_entity(self):
        graph = {
            "entities": [
                {
                    "entity_name": "Prescription Order",
                    "schema_name": "PHARMA_LAB",
                    "table_name": "F_RX_ORDER",
                },
                {
                    "entity_name": "Prescription Fill",
                    "schema_name": "PHARMA_LAB",
                    "table_name": "F_RX_FILL",
                },
            ]
        }
        resolved = {
            "verified": [
                {
                    "table_fqn": "CHATBOT_DB.PHARMA_LAB.F_RX_FILL",
                    "column": "FILL_STATUS",
                    "value": "PICKED_UP",
                }
            ]
        }
        self.assertEqual(
            _graph_entities_for_verified_values(resolved, graph),
            {"Prescription Fill"},
        )


if __name__ == "__main__":
    unittest.main()
