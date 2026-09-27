"""
A person is broken down by who they are.

"Top 10 customers by sales" is among the first things a sales team asks, and
the product could not answer it. A customer keeps no single name: a first
name and a last name, neither of which tells two customers apart -- two
customers here are called Alex -- so no column was fit to group by (release
D4 made sure the first name was never taken for one), and a member with no
label was no breakdown at all. "Sales by customer", "which customer had the
highest sales", French "les 3 meilleurs clients" -- each was declined, and
the model's attempt refused.

A member no single column names is now a breakdown where the question asks
by its whole name -- "by customer", "for each customer", "which customer",
"top 3 customers", "the 3 best customers" -- and never where the name only
begins another ("by customer gender" asks for the gender). It is grouped by
its identity: the key the source system gives it (CustomerAlternateKey),
never a person's official number, or else the dimension's own key; and shown
by the parts of its name, first to last, beside that identity. Two customers
of one name are two rows.

tests/star_harness.py keeps four customers, two of them called Alex, each
with a code of their own, a first and a last name.
"""

from __future__ import annotations

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("people")) as built:
        yield built


def _sales_by_customer(year: int) -> list[tuple[str, str, float]]:
    totals: dict = {}
    for line in star.orders():
        if line[2].year == year:
            totals[line[5]] = totals.get(line[5], 0) + line[7] * star.PRODUCTS[line[4]][4]
    return sorted(
        ((f"{star.CUSTOMERS[key][1]} {star.CUSTOMERS[key][2]}", star.CUSTOMERS[key][0], total)
         for key, total in totals.items()),
        key=lambda row: -row[2],
    )


def _rows(answer: dict) -> list[tuple]:
    return [(row["CUSTOMER"], row["CUSTOMER_ID"], row["SALES_AMOUNT"]) for row in answer["rows"]]


class TestTheProductAnswers:

    @pytest.mark.parametrize("question,lang", [
        ("Top 3 customers by sales in 2025", "en"),
        ("Les 3 meilleurs clients par ventes en 2025", "fr"),
    ])
    def test_the_top_customers(self, warehouse, question, lang):
        answer = star.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is False
        assert _rows(answer) == _sales_by_customer(2025)[:3]

    def test_sales_by_customer(self, warehouse):
        answer = star.ask(warehouse, "Sales by customer in 2025")
        assert sorted(_rows(answer)) == sorted(_sales_by_customer(2025))

    def test_which_customer(self, warehouse):
        answer = star.ask(warehouse, "Which customer had the highest sales in 2025?")
        assert _rows(answer)[0] == _sales_by_customer(2025)[0]

    def test_two_customers_of_one_first_name_are_two(self, warehouse):
        answer = star.ask(warehouse, "Sales by customer in 2025")
        alexes = [row for row in _rows(answer) if row[0].startswith("Alex ")]
        assert len(alexes) == 2 and len({row[1] for row in alexes}) == 2

    def test_by_an_attribute_of_the_customer_is_by_the_attribute(self, warehouse):
        answer = star.ask(warehouse, "Sales by customer gender")
        totals: dict = {}
        for line in star.orders():
            gender = star.CUSTOMERS[line[5]][3]
            totals[gender] = totals.get(gender, 0) + line[7] * star.PRODUCTS[line[4]][4]
        assert {row["GENDER"]: row["SALES_AMOUNT"] for row in answer["rows"]} == totals


class TestTheRule:

    @pytest.mark.parametrize("entity,columns,key,identity", [
        ("Customer", ["CustomerKey", "CustomerAlternateKey", "FirstName", "LastName", "Gender"], "CustomerKey",
         ("CustomerAlternateKey", ["FirstName", "LastName"])),
        # A person's official number is no label: the dimension's own key is.
        ("Employee", ["EmployeeKey", "EmployeeNationalIDAlternateKey", "LastName", "MiddleName", "FirstName"],
         "EmployeeKey", ("EmployeeKey", ["FirstName", "MiddleName", "LastName"])),
        ("Supplier", ["SUPPLIER_SK", "SUPPLIER_BK", "SUPPLIER_CITY"], "SUPPLIER_SK", ("SUPPLIER_BK", [])),
        # Another member's key is not this one's.
        ("Customer", ["CustomerKey", "GeographyAlternateKey", "Gender"], "CustomerKey", ("", [])),
        ("Geography", ["GeographyKey", "City"], "GeographyKey", ("", [])),
    ])
    def test_the_identity_of_a_member(self, entity, columns, key, identity):
        from core.semantic_model import member_identity

        assert member_identity(entity, columns, key) == identity

    @pytest.mark.parametrize("question,asked", [
        ("top 3 customers by sales in 2025", True),
        ("the 3 best customers by sales", True),
        ("sales by customer in 2025", True),
        ("sales for each customer", True),
        ("which customer had the highest sales", True),
        ("sales by customer gender", False),
        ("sales by gender of customer", False),
        ("how many customers placed an order", False),
        ("customer sales in 2025", False),
    ])
    def test_asked_by_the_member(self, question, asked):
        from core.semantic_model import _asks_by_the_member

        columns = ["CustomerKey", "CustomerAlternateKey", "FirstName", "LastName", "Gender"]
        assert _asks_by_the_member(question, "Customer", set(), columns) is asked

    @pytest.mark.parametrize("dialect,expression", [
        ("azure_sql", "CONCAT_WS(' ', c.FirstName, c.LastName)"),
        ("snowflake", "ARRAY_TO_STRING(ARRAY_CONSTRUCT_COMPACT(c.FirstName, c.LastName), ' ')"),
        ("oracle", "TRIM(REGEXP_REPLACE(c.FirstName || ' ' || c.LastName, ' +', ' '))"),
        ("mysql", "CONCAT_WS(' ', c.FirstName, c.LastName)"),
    ])
    def test_a_name_joined_in_each_dialect(self, dialect, expression):
        from core.pipeline_helpers import joined_name

        assert joined_name(["c.FirstName", "c.LastName"], dialect) == expression
        assert joined_name([], dialect) == ""
