"""
Two measures a question names are both asked for.

"Sales and product cost by year" came back as the sales of each product,
year by year: no cost, and a breakdown nobody asked for. "Product" was read
inside "product cost" as the products to break the sales down by, and the
cost -- a column the sales table keeps, TotalProductCost, not a registered
metric -- was never bound: the planner reads a measure by its whole name,
and nobody says "total". Named in full, "sales and total product cost by
year", the cost was bound and then set aside, since the metric Sales Amount
governs the sales table and a second measure on it was taken for a rival
reading of the first; the join graph still joined the products. "Total
product cost by year" asked which dataset to read: the product inventory
tables were as likely, by the "product" in the measure's name.

A measure is asked for by its name without its "total", and a table, a
dimension or the members a question is about are not named by a word said
only inside a measure's name. A measure the question names in two words or
more that no matched metric shares is a second measure, not a rival: both
are asked for. One word is not: the "order" of "customers who placed an
order" names no order quantity. A measure named in words of its own is
strong evidence of the dataset keeping it.
Where no registered metric computes a measure, the model writes the query,
and is given both measures, the year, and no product.

tests/star_harness.py keeps each sale's product cost beside its amount, and
two stock tables keyed by product.
"""

from __future__ import annotations

import re

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("two-measures")) as built:
        yield built


def _measures_asked(answer: dict) -> set[str]:
    """The measure columns the prompt the model was given asks for."""
    assert answer["model_wrote_sql"] is True
    line = next(line for line in answer["prompts"][0].splitlines() if line.startswith("- Measures:"))
    return {name.strip().split(".")[-1] for name in line.split(":", 1)[1].split(",")}


def _joins_products(answer: dict) -> bool:
    prompt = answer["prompts"][0]
    return bool(re.search(r"JOIN \[dbo\]\.\[DimProduct\]", prompt)) or "Required business grain: product" in prompt


class TestTheProductAnswers:

    @pytest.mark.parametrize("question", [
        "Sales and product cost by year",
        "Sales and total product cost by year",
    ])
    def test_both_measures_by_year_and_no_product(self, warehouse, question):
        answer = star.ask(warehouse, question)
        assert answer["rows"] == []
        assert _measures_asked(answer) == {"SALESAMOUNT", "TOTALPRODUCTCOST"}
        assert not _joins_products(answer)

    @pytest.mark.parametrize("question", ["Total product cost by year", "Product cost by category in 2025"])
    def test_the_cost_is_read_where_it_is_kept(self, warehouse, question):
        answer = star.ask(warehouse, question)
        assert "which source" not in str(answer["replies"]).lower()
        assert _measures_asked(answer) == {"TOTALPRODUCTCOST"}

    def test_two_metrics_are_still_both_answered(self, warehouse):
        answer = star.ask(warehouse, "Total sales and order quantity by category in 2025")
        assert answer["model_wrote_sql"] is False
        assert set(answer["rows"][0]) == {"PRODUCT_CATEGORY", "ORDER_QUANTITY", "SALES_AMOUNT"}

    def test_the_customers_who_placed_an_order_are_still_counted(self, warehouse):
        answer = star.ask(warehouse, "How many customers placed an order in 2025?")
        assert answer["model_wrote_sql"] is False
        assert [list(row.values()) for row in answer["rows"]] == [
            [len({line[5] for line in star.orders() if line[2].year == 2025})]]

    def test_sales_by_product_is_still_by_product(self, warehouse):
        answer = star.ask(warehouse, "Sales by product in 2025")
        assert answer["model_wrote_sql"] is False
        assert len(answer["rows"]) == len(star.PRODUCTS)


def _measure(column: str, *, expanded: str = "") -> dict:
    return {"column": column, "role": "measure", "expanded_name": expanded}


def _fact(name: str, entity: str, *measures: dict) -> dict:
    return {"type": "fact", "qualified_name": f"dbo.{name}", "schema": "DBO", "entity": entity,
            "fields": list(measures)}


_MODEL = {"tables": [
    _fact("FactInternetSales", "internet sales", _measure("SalesAmount", expanded="sales amount"),
          _measure("TotalProductCost", expanded="total product cost"), _measure("Freight", expanded="freight")),
    _fact("FactProductInventory", "product inventory", _measure("UnitCost", expanded="unit cost")),
    _fact("FactProductInventoryMonthly", "product inventory", _measure("UnitCost", expanded="unit cost")),
    _fact("FactSalesReturns", "returns", _measure("ReturnedSalesAmount", expanded="sales amount")),
    _fact("FactFreightInvoices", "freight invoices", _measure("InvoicedCharge", expanded="invoiced charge")),
]}
_SALES = {"name": "Sales Amount", "synonyms": "sales, revenue", "sql_template": "SUM(SalesAmount)",
          "base_table": "dbo.FactInternetSales"}
_ORDERS = {"name": "Purchase Order Amount", "synonyms": "purchase order value", "sql_template": "SUM(CAD_AMT)",
           "base_table": "M3.PCH_ORD_FCT"}
_TOTAL_SALES = {"name": "Total Sales", "synonyms": "net sales", "sql_template": "SUM(SalesAmount)",
                "base_table": "dbo.FactInternetSales"}
_BUYERS = {"name": "Number of Buying Customers", "synonyms": "buying customers",
           "sql_template": "COUNT(DISTINCT CustomerKey)", "base_table": "dbo.FactInternetSales"}


def _field(term: str, table: str, column: str) -> dict:
    return {"term": term, "table": table, "column": column, "role": "measure", "enforcement": None}


class TestTheRule:

    @pytest.mark.parametrize("column,alias,said", [
        ("TotalProductCost", "product cost", True),
        ("TOTAL_PRODUCT_COST", "product cost", True),
        # "Cost" alone is any cost's.
        ("TotalCost", "cost", False),
    ])
    def test_a_total_is_said_without_its_total(self, column, alias, said):
        from core.semantic_planner import _aliases_for_column

        assert (alias in _aliases_for_column(column)) is said

    @pytest.mark.parametrize("field,metric,demoted", [
        # Named in words of its own: a second measure.
        (_field("product cost", "dbo.FactInternetSales", "TOTALPRODUCTCOST"), _SALES, False),
        (_field("total product cost", "dbo.FactInternetSales", "TOTALPRODUCTCOST"), _SALES, False),
        # Named in the metric's own words: a rival reading of it.
        (_field("purchase order", "M3.PCH_ORD_FCT", "PCH_ORD_QTY"), _ORDERS, True),
        (_field("purchase value", "M3.PCH_ORD_FCT", "PCH_VAL"), _ORDERS, True),
        # A "total" both say is no word they share.
        (_field("total product cost", "dbo.FactInternetSales", "TOTALPRODUCTCOST"), _TOTAL_SALES, False),
        # A word that says only "how much" is the metric's.
        (_field("amount", "dbo.FactInternetSales", "TOTALPRODUCTCOST"), _TOTAL_SALES, True),
        # One word is no measure of its own: "customers who placed an order".
        (_field("order", "dbo.FactInternetSales", "ORDERQUANTITY"), _BUYERS, True),
    ])
    def test_a_measure_beside_a_metric(self, field, metric, demoted):
        from core.semantic_planner import demote_measures_governed_by_a_metric

        assert bool(demote_measures_governed_by_a_metric([field], [metric])) is demoted
        assert (field["enforcement"] == "optional") is demoted

    @pytest.mark.parametrize("question,selected", [
        ("total product cost by year", "DBO.FACTINTERNETSALES"),
        ("product cost by year", "DBO.FACTINTERNETSALES"),
        # A measure two datasets keep is still theirs to choose between.
        ("unit cost by product", ""),
        # A measure of one word is any measure's: no dataset of its own.
        ("freight by year", ""),
        # A dataset named by a measure's own name is still named, where
        # another keeps a measure of that name too.
        ("sales by year", "DBO.FACTINTERNETSALES"),
        ("sales amount by year", "DBO.FACTINTERNETSALES"),
    ])
    def test_the_dataset_that_keeps_the_measure(self, question, selected):
        from core.source_resolution import resolve_source_scope

        assert resolve_source_scope(question, _MODEL)["selected_fact"] == selected

    @pytest.mark.parametrize("question,tables", [
        ("total product cost by year", ["DBO.FACTINTERNETSALES"]),
        # A dataset named for a measure's one word is still one to offer.
        ("freight by year", ["DBO.FACTFREIGHTINVOICES", "DBO.FACTINTERNETSALES"]),
    ])
    def test_a_table_named_only_inside_a_measure_is_not_named(self, question, tables):
        from core.source_resolution import resolve_source_scope

        candidates = resolve_source_scope(question, _MODEL)["candidates"]
        assert [candidate["table"] for candidate in candidates] == tables

    @pytest.mark.parametrize("question,grain,asked", [
        ("sales and product cost by year", "product", ""),
        ("sales and total product costs by year", "product", ""),
        ("top 5 products by product cost", "product", "product"),
        ("sales by product", "product", "product"),
    ])
    def test_the_members_a_question_is_about(self, question, grain, asked):
        from core.analytical_request_plan import grain_asked_for

        plan = {"fields": [{"term": "product cost", "role": "measure"}, {"term": "total product cost", "role": "measure"}]}
        assert grain_asked_for(question, grain, plan) == asked
