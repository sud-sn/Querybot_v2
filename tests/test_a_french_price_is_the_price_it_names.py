"""
A French price is the price it names.

"Les 5 meilleurs produits par prix de vente" -- the five best products by
selling price -- came back as the five products with the most sales: "prix
de vente" was read word by word, "price of sales", and "sales" matched the
Sales Amount metric. "Prix catalogue moyen par catégorie" was refused, where
"average list price by category" is answered: "prix catalogue" came out
"price catalogue" and named no column. English "the 5 best products by list
price" was refused too, where "top 5 products by list price" is answered: a
ranking of members by their own attribute was read for "top", "highest",
"most", never for "best" or "worst".

A French price is read whole: "prix de vente" is the selling price, "prix
catalogue" the list price, "prix unitaire" the unit price, "prix d'achat"
the purchase price and "coût standard" the standard cost. "Best" and
"worst" rank as "top" and "bottom" do, and a French ranking ranks the
members its English names.

tests/star_harness.py keeps eight products, each with a list price and a
standard cost.
"""

from __future__ import annotations

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("french-price")) as built:
        yield built


def _by_list_price(name: int) -> list[tuple[str, float]]:
    """(name, list price), dearest first."""
    return sorted(((product[name], product[4]) for product in star.PRODUCTS.values()), key=lambda item: -item[1])


def _average_by_category(value: int) -> dict[str, float]:
    kept: dict[str, list[float]] = {}
    for product in star.PRODUCTS.values():
        category = star.CATEGORIES[star.SUBCATEGORIES[product[3]][2]][1]
        kept.setdefault(category, []).append(product[value])
    return {category: sum(values) / len(values) for category, values in kept.items()}


class TestTheProductAnswers:

    @pytest.mark.parametrize("question,lang,name,rows", [
        ("Les 5 meilleurs produits par prix catalogue", "fr", 2, slice(0, 5)),
        ("The 5 best products by list price", "en", 1, slice(0, 5)),
        ("The 3 worst products by list price", "en", 1, slice(-1, -4, -1)),
    ])
    def test_the_products_ranked_by_their_list_price(self, warehouse, question, lang, name, rows):
        answer = star.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is False
        assert [(row["PRODUCT"], row["LIST_PRICE"]) for row in answer["rows"]] == _by_list_price(name)[rows]

    def test_a_selling_price_is_no_sales_amount(self, warehouse):
        answer = star.ask(warehouse, "Les 5 meilleurs produits par prix de vente", "fr")
        assert all("SALES_AMOUNT" not in row for row in answer["rows"])
        assert "SALESAMOUNT" not in str(answer["executed"]).upper()

    def test_the_average_list_price_by_category(self, warehouse):
        answer = star.ask(warehouse, "Prix catalogue moyen par catégorie", "fr")
        assert answer["model_wrote_sql"] is False
        assert {row["PRODUCT_CATEGORY"]: row["AVERAGE_LIST_PRICE"] for row in answer["rows"]} == (
            _average_by_category(4))

    def test_the_average_standard_cost(self, warehouse):
        answer = star.ask(warehouse, "Coût standard moyen de nos produits", "fr")
        costs = [product[5] for product in star.PRODUCTS.values()]
        assert answer["model_wrote_sql"] is False
        assert [list(row.values()) for row in answer["rows"]] == [[sum(costs) / len(costs)]]


class TestTheRule:

    @pytest.mark.parametrize("question,read", [
        ("Produits par prix de vente", "products by selling price"),
        ("Prix de vente unitaire moyen", "unit selling price average"),
        ("Prix catalogue par produit", "list price by product"),
        ("Prix de catalogue par produit", "list price by product"),
        ("Prix unitaire par produit", "unit price by product"),
        ("Prix unitaires par produit", "unit price by product"),
        ("Prix d'achat par fournisseur", "purchase price by supplier"),
        ("Prix d’achat par fournisseur", "purchase price by supplier"),
        ("Prix d achat par fournisseur", "purchase price by supplier"),
        ("Coût standard par produit", "standard cost by product"),
        # A price on its own is a price.
        ("Prix moyen par produit", "price average by product"),
    ])
    def test_in_english(self, question, read):
        from core.question_normalizer import canonical_question

        assert canonical_question(question, "fr") == read

    @pytest.mark.parametrize("question,ranked", [
        ("the 5 best products by list price", True),
        ("the 3 worst products by list price", True),
        ("the average list price of our products", False),
    ])
    def test_best_and_worst_rank(self, question, ranked):
        from core.analytical_request_plan import _RANKED_BY

        assert bool(_RANKED_BY.search(question)) is ranked
