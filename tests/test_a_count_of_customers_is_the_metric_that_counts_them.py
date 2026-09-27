"""
A count of customers is the metric that counts them.

"How many customers placed an order in 2025?" was refused. Its words match
three registered metrics -- Number of Buying Customers, Number of Orders and
Order Quantity -- and with three in scope the plan was left open; the join
graph, reading "customers", required the customer table as well. Yet the
question counts customers: the order is only what they did, and one metric,
COUNT(DISTINCT CustomerKey), counts exactly that.

Where a count question's subject is what one matched metric's formula counts
-- its key named for the subject, by its words or its entity prefix -- that
metric is the answer, and the member it counts is no join the answer needs.
"How many customers do we have" is still every customer, counted on their
own table, and "how many orders did customers place" still counts orders.
French "ont passé", "ont acheté" and "ont commandé" are read as placed,
bought and ordered.

tests/star_harness.py keeps two years of orders from four customers and the
metric Number of Buying Customers.
"""

from __future__ import annotations

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("counted-customers")) as built:
        yield built


def _buying_customers(year: int) -> int:
    return len({line[5] for line in star.orders() if line[2].year == year})


class TestTheProductAnswers:

    @pytest.mark.parametrize("question,lang", [
        ("How many customers placed an order in 2025?", "en"),
        ("How many customers bought something in 2025?", "en"),
        ("Combien de clients ont acheté en 2025 ?", "fr"),
    ])
    def test_the_customers_who_bought(self, warehouse, question, lang):
        answer = star.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is False
        assert "DIMCUSTOMER" not in answer["sql"].upper()
        assert [list(row.values()) for row in answer["rows"]] == [[_buying_customers(2025)]]

    def test_a_count_of_orders_is_still_of_orders(self, warehouse):
        answer = star.ask(warehouse, "How many orders in 2025?")
        assert [list(row.values()) for row in answer["rows"]] == [
            [len({line[0] for line in star.orders() if line[2].year == 2025})]]


_METRICS = [
    {"name": "Number of Buying Customers", "sql_template": "COUNT(DISTINCT CustomerKey)"},
    {"name": "Number of Orders", "sql_template": "COUNT(DISTINCT SalesOrderNumber)"},
    {"name": "Order Quantity", "sql_template": "SUM(OrderQuantity)"},
]


class TestTheRule:

    @pytest.mark.parametrize("question,chosen", [
        ("how many customers placed an order in 2025", ["Number of Buying Customers"]),
        ("number of customers who ordered in 2025", ["Number of Buying Customers"]),
        ("how many customers bought something", ["Number of Buying Customers"]),
        # Orders are what is counted; the customers placed them.
        ("how many orders did customers place", [m["name"] for m in _METRICS]),
        # Every customer, not the buying ones: the population's own count.
        ("how many customers do we have", [m["name"] for m in _METRICS]),
        ("total order quantity in 2025", [m["name"] for m in _METRICS]),
    ])
    def test_the_metric_that_counts_the_subject(self, question, chosen):
        from core.analytical_request_plan import metrics_counting_the_subject

        assert [m["name"] for m in metrics_counting_the_subject(question, _METRICS)] == chosen

    def test_a_key_named_by_its_entity_prefix(self):
        from core.analytical_request_plan import metrics_counting_the_subject

        metrics = [{"name": "Active Customers", "sql_template": "COUNT(DISTINCT CUS_DMS_KEY)"},
                   {"name": "Orders", "sql_template": "COUNT(DISTINCT ORD_NO)"}]
        assert [m["name"] for m in metrics_counting_the_subject("how many customers placed an order", metrics)] == [
            "Active Customers"]

    def test_two_that_count_it_are_not_chosen_between(self):
        from core.analytical_request_plan import metrics_counting_the_subject

        metrics = [{"name": "Buying Customers", "sql_template": "COUNT(DISTINCT CustomerKey)"},
                   {"name": "Returning Customers", "sql_template": "COUNT(DISTINCT CustomerKey)"}]
        assert metrics_counting_the_subject("how many customers placed an order", metrics) == metrics

    def test_the_counted_member_on_its_own_table_is_set_aside(self):
        from core.analytical_request_plan import COUNTED_MEMBERS, demote_counted_members

        plan = {"fields": [{"term": "customer", "table": "dbo.DimCustomer", "column": "CustomerKey",
                            "role": "dimension", "enforcement": "required"}]}
        demote_counted_members(plan, {"intent": "metric_query", "dimensions": []}, [_METRICS[0]])
        assert plan["fields"][0]["demotion_reason"] == COUNTED_MEMBERS

    @pytest.mark.parametrize("question,read", [
        ("Combien de clients ont acheté en 2025 ?", "how many customers bought en 2025 ?"),
        ("Combien de clients ont commandé en 2025 ?", "how many customers ordered en 2025 ?"),
        ("Combien de clients ont passé une commande ?", "how many customers placed an order ?"),
    ])
    def test_in_french(self, question, read):
        from core.question_normalizer import canonical_question

        assert canonical_question(question, "fr") == read
