"""
A measure's own name is no period.

"What is the average yearly income of our customers?" came back as a series:
"2023-01-01 closed at $77,500.00" -- the average grouped by the year of the
customers' first purchase. On a warehouse whose customer table keeps more
than one date, the reader was asked which date to use instead. "Yearly" is
the attribute's name -- the customer table's yearly income -- and no year the
question asks about; but the question was read for time words whole, and
"yearly" is one.

The names of two words or more of the measures a question's plan binds are
taken out before its time words are read. "Average yearly income by gender"
is one figure a gender, "sales by year" is still by year, and "yearly income
by year of first purchase" still asks for a year.

tests/star_harness.py keeps four customers, each with a yearly income and
the date of their first purchase.
"""

from __future__ import annotations

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("measure-name")) as built:
        yield built


def _average_income() -> float:
    incomes = [customer[4] for customer in star.CUSTOMERS.values()]
    return sum(incomes) / len(incomes)


class TestTheProductAnswers:

    def test_one_average_not_one_a_year(self, warehouse):
        answer = star.ask(warehouse, "What is the average yearly income of our customers?")
        assert answer["model_wrote_sql"] is False
        assert [list(row.values()) for row in answer["rows"]] == [[_average_income()]]

    def test_sales_by_year_is_still_by_year(self, warehouse):
        answer = star.ask(warehouse, "Sales by year")
        assert sorted(row["PERIOD"] for row in answer["rows"]) == [2024, 2025]


def _measure(term: str, enforcement: str | None = None) -> dict:
    return {"term": term, "role": "measure", "enforcement": enforcement}


class TestTheRule:

    @pytest.mark.parametrize("question,fields,read", [
        ("what is the average yearly income of our customers", [_measure("yearly income")],
         "what is the average   of our customers"),
        ("average Yearly Incomes by gender", [_measure("yearly income")], "average   by gender"),
        # A name of one word is the question's own word.
        ("sales by year", [_measure("sales")], "sales by year"),
        # A name the plan set aside is no measure's.
        ("average yearly income", [_measure("yearly income", "optional")], "average yearly income"),
        # A name whose head is a period is a period, whatever the plan calls it.
        ("sales in fiscal year 2025", [_measure("fiscal year")], "sales in fiscal year 2025"),
        # Nor is a breakdown's.
        ("sales by customer segment", [{"term": "customer segment", "role": "dimension"}], "sales by customer segment"),
    ])
    def test_the_question_less_its_measures_names(self, question, fields, read):
        from core.semantic_model import without_measure_field_names

        assert without_measure_field_names(question, fields) == read

    def test_its_time_words_are_then_none(self):
        from core.date_roles import question_has_temporal_intent
        from core.semantic_model import without_measure_field_names

        question = "what is the average yearly income of our customers"
        assert question_has_temporal_intent(question) is True
        assert question_has_temporal_intent(without_measure_field_names(question, [_measure("yearly income")])) is False
