"""
An average of an attribute is not a count of its members.

"What is the average yearly income of our customers?" was refused, on two
warehouses. The product averages an attribute a member's own table keeps --
the customer table's yearly income -- over its members, and does so for
"average yearly income by gender"; but "customers" matched the metric Number
of Buying Customers, and any metric in scope turned that reading off. The
question then carried a count of customers on the sales table it never asked
for, and the plan was left between two tables and refused.

A metric that counts the members a question names is not asked for where the
question asks an average, a minimum or a maximum it does not compute, and no
count: the attribute is averaged on its own table, and a metric reading any
other table is set aside. "How many customers placed an order" and "average
sales per month" keep their metrics.

tests/star_harness.py keeps four customers with a yearly income and the
metric Number of Buying Customers on the sales table.
"""

from __future__ import annotations

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("attribute-average")) as built:
        yield built


class TestTheProductAnswers:

    def test_the_average_yearly_income_of_the_customers(self, warehouse):
        answer = star.ask(warehouse, "What is the average yearly income of our customers?")
        incomes = [customer[4] for customer in star.CUSTOMERS.values()]
        assert answer["model_wrote_sql"] is False
        assert "FACTINTERNETSALES" not in answer["sql"].upper()
        assert [list(row.values()) for row in answer["rows"]] == [[sum(incomes) / len(incomes)]]

    def test_a_count_of_the_customers_is_still_the_metrics(self, warehouse):
        answer = star.ask(warehouse, "How many customers placed an order in 2025?")
        assert "FACTINTERNETSALES" in answer["sql"].upper()
        assert [list(row.values()) for row in answer["rows"]] == [
            [len({line[5] for line in star.orders() if line[2].year == 2025})]]


_BUYERS = {"name": "Number of Buying Customers", "synonyms": "buying customers, active customers",
           "sql_template": "COUNT(DISTINCT CustomerKey)",
           "_resolved_source_tables": ["DBO.FACTINTERNETSALES"]}
_SALES = {"name": "Sales Amount", "synonyms": "sales", "sql_template": "SUM(SalesAmount)",
          "_resolved_source_tables": ["DBO.FACTINTERNETSALES"]}
_ON_THE_TABLE = {"name": "Customers On File", "sql_template": "COUNT(CustomerKey)",
                 "_resolved_source_tables": ["DBO.DIMCUSTOMER"]}


class TestTheRule:

    @pytest.mark.parametrize("question,metrics,asked", [
        ("what is the average yearly income of our customers", [_BUYERS], []),
        ("what is the maximum yearly income of our customers", [_BUYERS], []),
        # Nothing is averaged: the metric is what is asked for.
        ("buying customers by country", [_BUYERS], [_BUYERS]),
        # A count asked is the metric's.
        ("how many customers do we have", [_BUYERS], [_BUYERS]),
        ("average number of customers per month", [_BUYERS], [_BUYERS]),
        # A metric that is no count is asked for, averaged.
        ("average sales per customer", [_SALES], [_SALES]),
        ("average yearly income of customers and their sales", [_BUYERS, _SALES], [_BUYERS, _SALES]),
    ])
    def test_the_metrics_asked_for(self, question, metrics, asked):
        from core.analytical_request_plan import metrics_asked_for

        assert metrics_asked_for(question, metrics) == asked

    def test_on_the_attributes_table_only(self):
        from core.analytical_request_plan import ATTRIBUTE_SOURCE, metrics_on_the_attributes_table

        scope = {"selected_fact": "dbo.DimCustomer", "source_kind": "master", "reason": ATTRIBUTE_SOURCE}
        assert metrics_on_the_attributes_table(scope, [_BUYERS, _ON_THE_TABLE]) == [_ON_THE_TABLE]

    def test_another_source_keeps_them(self):
        from core.analytical_request_plan import metrics_on_the_attributes_table

        scope = {"selected_fact": "dbo.DimCustomer", "source_kind": "master", "reason": "governed population master table"}
        assert metrics_on_the_attributes_table(scope, [_BUYERS]) == [_BUYERS]
