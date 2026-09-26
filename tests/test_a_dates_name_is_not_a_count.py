"""
A date's name is not a count of its event.

"Budget by shipment date for 2025" asks for the budget, dated by shipment.
The event-count detector read "by shipment" as the count cue of "customers
by orders", and the question became a count of shipments: in the planner,
which sends it to the count-target resolver, and in the SQL prompt, which
told the model "BUSINESS EVENT COUNT DETECTED: interpret 'shipment count'
as COUNT(DISTINCT ...)". The same for any event date: "budget by order
month".

Now an event word followed by a date word is read as the date's name
(core/analytical_intent.py). Plural event words still count: "number of
shipments by order date" counts shipments.

The real detector, planner, prompt hints and canonicaliser.
"""

from __future__ import annotations

import pytest

from core.analytical_intent import detect_business_event_count, plan_analytical_intent
from core.query_semantics import build_generic_query_hints
from core.question_normalizer import canonical_question

BUDGET = {"name": "Budget", "base_table": "MART.BDG_FCT", "formula_type": "expression",
          "sql_template": "SUM(BDG_AMT)"}


class TestADateNamedByItsEvent:

    @pytest.mark.parametrize("question", [
        "budget by shipment date for 2025",
        "budget by order month",
        "budget by invoice quarter for 2025",
    ])
    def test_is_not_a_count(self, question):
        assert detect_business_event_count(question) == ""

    def test_nor_to_the_planner(self):
        plan = plan_analytical_intent("budget by shipment date for 2025", metrics=[BUDGET])
        assert plan.counted_entity == ""
        assert "shipment count" not in plan.business_concepts

    def test_nor_in_the_prompt(self):
        assert "BUSINESS EVENT COUNT DETECTED" not in build_generic_query_hints("budget by shipment date for 2025")

    def test_nor_in_french(self):
        assert detect_business_event_count(canonical_question("budget par date de commande", "fr")) == ""


class TestACountStillCounts:

    @pytest.mark.parametrize("question, event", [
        ("number of shipments by order date", "shipment"),
        ("how many orders by order date", "order"),
        ("customers ranked by orders", "order"),
        ("how many invoices this invoice year", "invoice"),
        ("total invoices month to date", "invoice"),
    ])
    def test_beside_a_date_named_by_an_event(self, question, event):
        assert detect_business_event_count(question) == event

    def test_in_the_prompt(self):
        assert "BUSINESS EVENT COUNT DETECTED" in build_generic_query_hints("number of shipments by order date")
