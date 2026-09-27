"""
A count that says "late" is not every row.

"How many orders were shipped late?" was answered: "Number of Orders: 48" --
every order the warehouse keeps, late or not. The governed compilers write a
measure, its breakdowns and its period; whether a row kept its time -- shipped
late, delivered on time, paid after its due date, overdue -- is a condition
none of them writes, and nothing stopped them answering without it. French
"expédiées en retard" arrived untranslated and was answered the same way.

A question that says whether rows kept their time is now left to the planner,
as one that names a member or a condition on a value already is -- unless a
metric the question matched says it ("Late Orders" is the admin's own
definition of late, and is answered). "Late 2024" and "early January" are
periods, not conditions. French "en retard", "en avance", "à temps", "en
souffrance" and "après leur ... date" are read as late, early, on time,
overdue and after their ... date -- and "commandes en souffrance" stays the
back orders it names.

tests/star_harness.py keeps two years of orders, each with a due date and a
ship date.
"""

from __future__ import annotations

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("late")) as built:
        yield built


class TestTheProductAnswers:

    @pytest.mark.parametrize("question,lang", [
        ("How many orders were shipped late?", "en"),
        ("Combien de commandes ont été expédiées en retard ?", "fr"),
        ("How many orders were delivered on time in 2025?", "en"),
    ])
    def test_every_order_is_not_the_answer(self, warehouse, question, lang):
        answer = star.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is True
        assert answer["rows"] == []

    def test_a_count_with_no_condition_is_still_answered(self, warehouse):
        answer = star.ask(warehouse, "How many orders in 2025?")
        assert answer["model_wrote_sql"] is False
        assert [list(row.values()) for row in answer["rows"]] == [
            [len({line[0] for line in star.orders() if line[2].year == 2025})]]


_ORDERS = {"name": "Number of Orders", "synonyms": "orders, order count", "sql_template": "COUNT(DISTINCT OrderNo)"}


class TestTheRule:

    @pytest.mark.parametrize("question", [
        "how many orders were shipped late",
        "orders shipped early in 2025",
        "overdue invoices by customer",
        "deliveries delayed last month",
        "invoices past due",
        "orders delivered on time",
        "orders behind schedule",
        "how many orders shipped after their due date",
        "invoices paid before its payment due date",
    ])
    def test_a_condition_on_time_kept(self, question):
        from core.pipeline_helpers import _left_to_the_planner

        assert _left_to_the_planner({"question": question, "metric_formulas": [_ORDERS]}) == (
            "states whether rows kept their time")

    @pytest.mark.parametrize("question", [
        "sales in late 2024",
        "orders placed in early january",
        "sales early next year",
        "the latest orders",
        "sales over time",
        "orders by due date",
    ])
    def test_a_period_is_not_a_condition(self, question):
        from core.pipeline_helpers import _left_to_the_planner

        assert _left_to_the_planner({"question": question, "metric_formulas": [_ORDERS]}) == ""

    @pytest.mark.parametrize("metric", [
        {"name": "Late Orders", "synonyms": ""},
        {"name": "Shipped Orders", "synonyms": "orders shipped late, late shipments"},
    ])
    def test_a_metric_that_says_it(self, metric):
        from core.pipeline_helpers import _left_to_the_planner

        assert _left_to_the_planner({"question": "how many orders were shipped late", "metric_formulas": [metric]}) == ""

    @pytest.mark.parametrize("question,read", [
        ("Combien de commandes ont été expédiées en retard ?", "how many orders have been shipped late ?"),
        ("Commandes livrées en avance", "orders delivered early"),
        ("Commandes expédiées à temps", "orders shipped on time"),
        ("Factures en souffrance", "invoices overdue"),
        ("Commandes expédiées après leur date d'échéance", "orders shipped after their due date"),
        # Orders "en souffrance" are back orders.
        ("Commandes en souffrance par entrepôt", "back orders by warehouse"),
        ("Une commande en souffrance", "a back order"),
    ])
    def test_in_french(self, question, read):
        from core.question_normalizer import canonical_question

        assert canonical_question(question, "fr") == read
