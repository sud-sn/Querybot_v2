"""
A word that narrows the question is never dropped.

"How many customers are female?" was answered "Number of Buying Customers:
4" -- every customer who bought. "Sales of red products" came back as every
product's sales, and "how many products did we sell that are red?" as every
product sold. The governed compilers write a measure, its breakdowns and its
period; a word that says which of the things counted the question means was
dropped without a sign, and the answer looked like one to the question asked.
French "clients actifs" was answered as every customer.

A word that stands before a thing the question names ("red products",
"female customers", "new customers"), after it in French ("clients
actifs"), or that the thing is said to be ("customers are female", "that are
red", "qui sont actifs") is now read: where no matched metric, field or the
table a field is read on is named by it, no governed compiler answers and the
question is left to the planner, which reads the warehouse's own values. The
words that narrow nothing -- "how many", "our top 5", "each", "last",
"sold", "customer gender" -- are read as they were.

tests/star_harness.py keeps customers with a gender and no "active" flag,
and products with no colour.
"""

from __future__ import annotations

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("narrowed")) as built:
        yield built


class TestTheProductAnswers:

    @pytest.mark.parametrize("question,lang", [
        ("How many customers are female?", "en"),
        ("Sales of red products", "en"),
        ("How many products did we sell that are red?", "en"),
        ("Combien de clients actifs avons-nous ?", "fr"),
    ])
    def test_no_total_for_a_narrowed_question(self, warehouse, question, lang):
        answer = star.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is True
        assert answer["rows"] == []

    @pytest.mark.parametrize("question,lang", [
        ("Sales by customer gender", "en"),
        ("How many customers placed an order in 2025?", "en"),
        ("Les 3 meilleurs clients par ventes en 2025", "fr"),
    ])
    def test_a_question_nothing_narrows_is_answered(self, warehouse, question, lang):
        answer = star.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is False
        assert answer["rows"]


_CUSTOMERS = {"fields": [{"term": "customer", "table": "dbo.DimCustomer", "column": "CustomerKey"}]}
_PRODUCTS = {"fields": [{"term": "Product", "table": "dbo.DimProduct", "column": "EnglishProductName"}]}
_SALES = [{"name": "Sales Amount", "synonyms": "sales, revenue"}]


def _context(question: str, plan: dict | None = None, metrics: list | None = None, canonical: str = "") -> dict:
    return {"question": question, "canonical_question": canonical or question,
            "semantic_plan": plan or {}, "metric_formulas": metrics or []}


class TestTheRule:

    @pytest.mark.parametrize("context,word", [
        (_context("How many customers are female?", _CUSTOMERS), "female"),
        (_context("customers who are married", _CUSTOMERS), "married"),
        # The entity a count counts, when no field names it.
        ({**_context("How many customers are female?"),
          "analytical_request_plan": {"derived_measure": {"business_entity": "customer"}}}, "female"),
        (_context("Sales of red products", _PRODUCTS, _SALES), "red"),
        (_context("How many products did we sell that are red?", _PRODUCTS), "red"),
        (_context("How many new customers did we have in 2025?", _CUSTOMERS), "new"),
        (_context("How many orders were cancelled?", metrics=[{"name": "Number of Orders"}]), "cancelled"),
        (_context("Combien de clients actifs avons-nous ?", _CUSTOMERS,
                  canonical="how many customers actifs do we have ?"), "actifs"),
        (_context("Nombre de clients qui sont actifs", _CUSTOMERS,
                  canonical="count of customers qui sont actifs"), "actifs"),
    ])
    def test_the_word_that_narrows(self, context, word):
        from core.pipeline_helpers import _left_to_the_planner, _unread_qualifier

        assert _unread_qualifier(context) == word
        assert _left_to_the_planner(context) == f"narrows what it asks by a word nothing reads ({word!r})"

    @pytest.mark.parametrize("context", [
        _context("How many customers are there?", _CUSTOMERS),
        _context("Top 5 customers by sales in 2025", _CUSTOMERS, _SALES),
        _context("How many customers placed an order in each month of 2025?", _CUSTOMERS),
        _context("Which products are selling best?", _PRODUCTS, _SALES),
        _context("How many orders were shipped in 2025?", metrics=[{"name": "Number of Orders"}]),
        # A metric the question matched is named by its words.
        _context("Net sales of our products by month", _PRODUCTS, [{"name": "Net Sales"}]),
        # The table a field is read on names whose it is.
        _context("Sales by customer gender", {"fields": [{"term": "gender", "table": "dbo.DimCustomer"}]}, _SALES),
        _context("Sales by customer gender", {"fields": [{"term": "gender", "table": "SALESDW.DBO.DIMCUSTOMER"}]}, _SALES),
        # ... or by the name the vocabulary reads in its code.
        _context("Sales by customer gender", {"fields": [{"term": "gender", "table": "SAMPLEDW.CUS_DMS"}]}, _SALES),
        # The word before a date names which date.
        _context("Net sales by order date and shipment date for 2025",
                 {"fields": [{"term": "order date", "table": "dbo.DimDate"}]}, [{"name": "Net Sales"}]),
        # A field's own name, whole.
        _context("Sales by freight carrier", {"fields": [{"term": "freight carrier", "table": "dbo.FactOrders"}]}, _SALES),
        # A metric's French synonym, read as the question was.
        _context("Unités vendues par mois", metrics=[{"name": "Units Sold", "synonyms": "unités vendues"}],
                 canonical="units sold by month"),
        # French rankings, ordinals and "current" narrow nothing.
        _context("Les 10 articles ayant la plus grande valeur de stock",
                 {"fields": [{"term": "item", "table": "dbo.Item"}]}, [{"name": "Inventory Value"}],
                 canonical="the 10 items ayant the plus grande value of inventory"),
        _context("Unités vendues au premier trimestre 2022", metrics=[{"name": "Units Sold"}],
                 canonical="unites vendues to premier quarter 2022"),
        _context("Les stocks actuelles par entrepôt", metrics=[{"name": "Inventory"}],
                 canonical="the inventory actuelles by warehouse"),
        # A word the canonicaliser wrote in English is not one it left in French.
        _context("Ventes budget 2025", metrics=_SALES, canonical="sales budget 2025"),
        _context("Les 3 meilleurs clients par ventes en 2025", _CUSTOMERS, _SALES,
                 canonical="the 3 best customers by sales en 2025"),
        _context("Ventes des produits par catégorie en 2025", _PRODUCTS, _SALES,
                 canonical="sales of products by category en 2025"),
    ])
    def test_a_word_that_narrows_nothing(self, context):
        from core.pipeline_helpers import _unread_qualifier

        assert _unread_qualifier(context) == ""
