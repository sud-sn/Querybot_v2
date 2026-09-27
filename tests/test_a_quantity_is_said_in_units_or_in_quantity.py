"""
A quantity is said in units or in quantity.

"Top 5 products by units sold in 2025" was answered. "Top 5 products by
quantity sold in 2025" was asked "What measure should I use to rank them?":
the metric is said "units sold" and "order quantity", and a ranking is read
for a metric only where the question says one of its names whole. Scored
word by word, "quantity sold" shared "sold" with "units sold" and both words
with "products sold" -- the count of products on at least one order, one
each. French "quantité vendue par catégorie" was refused, and "top 5 des
produits par quantité vendue" asked the same question: "vendue" was left in
French.

A count of goods is said in units or in quantity, "qty" among them: a
metric's name is found in the question whichever the two say it in, and
only in the plural -- a unit price is no quantity. French "vendu", in any
agreement, is "sold", and "unités" is units; where the reader's French
names no metric and would be asked which, its English reading is read for
one. "Top 3 products" still asks which measure.

tests/star_harness.py keeps two years of order lines, each with a quantity,
and the metric Order Quantity, said "units sold".
"""

from __future__ import annotations

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("quantity-words")) as built:
        yield built


def _quantity_by(key) -> dict:
    """2025's units sold, by what ``key`` reads off a product."""
    totals: dict = {}
    for line in star.orders():
        if line[2].year == 2025:
            name = key(star.PRODUCTS[line[4]])
            totals[name] = totals.get(name, 0) + line[7]
    return totals


def _category(lang: int):
    return lambda product: star.CATEGORIES[star.SUBCATEGORIES[product[3]][2]][lang]


class TestTheProductAnswers:

    @pytest.mark.parametrize("question,lang,name", [
        ("Top 5 products by quantity sold in 2025", "en", 1),
        ("Top 5 des produits par quantité vendue en 2025", "fr", 2),
    ])
    def test_the_top_products_by_quantity_sold(self, warehouse, question, lang, name):
        answer = star.ask(warehouse, question, lang)
        sold = _quantity_by(lambda product: product[name])
        assert answer["model_wrote_sql"] is False
        assert [row["ORDER_QUANTITY"] for row in answer["rows"]] == sorted(sold.values(), reverse=True)[:5]
        assert all(sold[row["PRODUCT"]] == row["ORDER_QUANTITY"] for row in answer["rows"])

    @pytest.mark.parametrize("question,lang,name", [
        ("Quantity sold by category in 2025", "en", 0),
        ("Quantité vendue par catégorie en 2025", "fr", 1),
    ])
    def test_the_quantity_sold_by_category(self, warehouse, question, lang, name):
        answer = star.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is False
        assert {row["PRODUCT_CATEGORY"]: row["ORDER_QUANTITY"] for row in answer["rows"]} == _quantity_by(
            _category(name))

    def test_a_ranking_by_no_measure_still_asks(self, warehouse):
        answer = star.ask(warehouse, "Top 3 products")
        assert answer["executed"] == []
        assert "What measure should I use to rank them?" in str(answer["replies"])


_ORDER_QUANTITY = next(metric for metric in star.METRICS if metric["name"] == "Order Quantity")
_PRODUCTS_SOLD = next(metric for metric in star.METRICS if metric["name"] == "Number of Products Sold")


class TestTheRule:

    @pytest.mark.parametrize("text,read", [
        ("units sold", "quantity sold"),
        ("Qty on hand", "quantity on hand"),
        ("quantities received", "quantity received"),
        ("quantity sold", "quantity sold"),
        # A unit price is no quantity.
        ("unit price", "unit price"),
    ])
    def test_one_word_for_a_quantity(self, text, read):
        from core.word_forms import with_one_quantity_word

        assert with_one_quantity_word(text) == read

    @pytest.mark.parametrize("question,matched", [
        ("top 5 products by quantity sold", ["Order Quantity"]),
        ("top 5 products by qty sold", ["Order Quantity"]),
        ("unit price by product", []),
    ])
    def test_the_metric_the_question_names(self, question, matched):
        from core.analytical_intent import _matched_catalog_names

        assert _matched_catalog_names(question, [_ORDER_QUANTITY]) == matched

    def test_the_quantity_outranks_the_products_counted(self):
        from core.metric_scope import _phrase_score

        question = "top 5 products by quantity sold in 2025"
        assert _phrase_score(_ORDER_QUANTITY, question) > _phrase_score(_PRODUCTS_SOLD, question)

    @pytest.mark.parametrize("question,read", [
        ("Quantité vendue par catégorie", "quantity sold by category"),
        ("Quantités vendues par mois", "quantity sold by month"),
        ("Unités vendues par produit", "units sold by product"),
        ("Nombre de produits vendus", "count of products sold"),
    ])
    def test_french_in_english(self, question, read):
        from core.question_normalizer import canonical_question

        assert canonical_question(question, "fr") == read
