import asyncio
import unittest
from unittest.mock import patch

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

    def test_data_shaped_request_bypasses_llm_classifier(self):
        with patch.object(dispatcher, "llm_complete") as complete:
            allowed = asyncio.run(
                dispatcher._classify_is_data_question(
                    "List patients with prescriptions and include their diagnosis.",
                    {},
                )
            )
        self.assertTrue(allowed)
        complete.assert_not_called()


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
