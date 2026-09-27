"""
A dimension said only inside a longer name is not asked for.

"Sales amount by product category" was never answered on a made-up retailer's
warehouse, in English or French, though its product has a category through its
subcategory and the compiler walks that path. The product is a dimension of its
own, called "Product", and the question says "product": the plan was asked to
group by the product and its category at once, and the compiler, which groups
by one dimension, declined. So did "by product subcategory", and every question
by a snowflaked dimension whose name begins with its parent's.

A dimension's name said only as part of a longer dimension name the question
says in full -- "product category", "product categories", or "category of
products", a French "catégorie de produits" read in English -- is that longer
name's. So is a dimension's name said only inside a measure's: "reseller sales
by business type" asks for no reseller, and "buying customers by product
category" for no customer. A product said on its own as well is still asked
for. And a French "sous-catégorie" is read as the subcategory, whose name is
longer than the category's -- not as the category.

tests/star_harness.py keeps a product with a subcategory and a category, and a
count of buying customers.
"""

from __future__ import annotations

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("longer-names")) as built:
        yield built


def _sales_by(level: str, lang: str = "en") -> dict:
    """Sales amount by the product's category or subcategory, by name."""
    totals: dict = {}
    for line in star.orders():
        _code, _en, _fr, subcategory, price, _cost = star.PRODUCTS[line[4]]
        english, french, category = star.SUBCATEGORIES[subcategory]
        name = (english, french) if level == "subcategory" else star.CATEGORIES[category]
        key = name[1] if lang == "fr" else name[0]
        totals[key] = totals.get(key, 0) + line[7] * price
    return totals


def _answered(answer: dict) -> dict:
    return {row["PRODUCT_CATEGORY" if "PRODUCT_CATEGORY" in row else "PRODUCT_SUBCATEGORY"]: row["SALES_AMOUNT"]
            for row in answer["rows"]}


class TestTheProductAnswers:

    @pytest.mark.parametrize("question,lang", [
        ("Sales amount by product category", "en"),
        ("Sales amount by product categories", "en"),
        ("Montant des ventes par catégorie de produits", "fr"),
    ])
    def test_by_category(self, warehouse, question, lang):
        answer = star.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is False
        assert "EnglishProductName" not in answer["sql"]
        assert _answered(answer) == _sales_by("category", lang)

    @pytest.mark.parametrize("question,lang", [
        ("Sales amount by product subcategory", "en"),
        ("Montant des ventes par sous-catégorie de produits", "fr"),
    ])
    def test_by_subcategory(self, warehouse, question, lang):
        answer = star.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is False
        assert _answered(answer) == _sales_by("subcategory", lang)

    def test_a_measure_named_for_the_products(self, warehouse):
        answer = star.ask(warehouse, "Products sold by customer gender")
        assert answer["model_wrote_sql"] is False
        assert "DimProduct" not in answer["sql"]
        sold: dict = {}
        for line in star.orders():
            sold.setdefault(star.CUSTOMERS[line[5]][3], set()).add(line[4])
        assert {row["GENDER"]: row["NUMBER_OF_PRODUCTS_SOLD"] for row in answer["rows"]} == {
            gender: len(products) for gender, products in sold.items()}

    def test_a_measure_named_for_the_customers(self, warehouse):
        answer = star.ask(warehouse, "Buying customers by product category")
        assert answer["model_wrote_sql"] is False
        assert "DimCustomer" not in answer["sql"]
        buyers: dict = {}
        for line in star.orders():
            category = star.CATEGORIES[star.SUBCATEGORIES[star.PRODUCTS[line[4]][3]][2]][0]
            buyers.setdefault(category, set()).add(line[5])
        assert {row["PRODUCT_CATEGORY"]: row["NUMBER_OF_BUYING_CUSTOMERS"] for row in answer["rows"]} == {
            category: len(customers) for category, customers in buyers.items()}


class TestTheWords:

    @pytest.mark.parametrize("french,english", [
        ("Ventes par sous-catégorie de produits", "sales by subcategory of products"),
        ("Ventes par sous-catégories", "sales by subcategories"),
        # "Sous" alone is "under", and a quoted value is the reader's own.
        ("Ventes sous le seuil", "sales sous the seuil"),
        ("Ventes par 'sous-marque'", "sales by 'sous-marque'"),
    ])
    def test_a_french_sub_compound(self, french, english):
        from core.question_normalizer import canonicalise

        assert canonicalise(french) == english

    @pytest.mark.parametrize("question,left", [
        ("sales by product category", False),
        ("sales by product categories", False),
        ("sales by category of products", False),
        ("sales by category of the product", False),
        ("sales by product", True),
        ("sales by product and product category", True),
        ("product sales by category", True),
    ])
    def test_the_product(self, question, left):
        from core.semantic_model import _without_longer_names

        said = _without_longer_names(question, "Product", {"product", "product category", "customer"})
        assert ("product" in said.lower()) is left

    @pytest.mark.parametrize("question,left", [
        ("reseller sales by business type", False),
        ("sales to resellers by business type", False),
        ("reseller sales by reseller", True),
        ("sales by reseller", True),
    ])
    def test_a_measures_name(self, question, left):
        from core.semantic_model import _without_longer_names

        said = _without_longer_names(question, "Reseller", {"reseller sales", "sales to resellers", "sales"})
        assert ("reseller" in said.lower()) is left

    def test_a_name_is_never_its_own_longer_name(self):
        from core.semantic_model import _without_longer_names

        question = "sales by product category"
        assert _without_longer_names(question, "Product Category", {"product category", "product"}) == question

    def test_every_word_of_the_name_is_in_the_longer_one(self):
        from core.semantic_model import _without_longer_names

        # A territory's region holds the territory; the customer's region does not.
        assert "territory" not in _without_longer_names(
            "sales by region of sales territory", "Sales Territory", {"sales territory region"})
        assert "customer" in _without_longer_names(
            "sales by customer region", "Customer", {"sales territory region"})
        # Sharing a word is not holding the name.
        question = "sales by product subcategory name"
        assert _without_longer_names(question, "Product Category", {"product subcategory name"}) == question

    @pytest.mark.parametrize("name,said", [
        ("product category", "Product Categories"),
        ("sales territory region", "region of sales territories"),
        ("box type", "box types"),
        ("working day", "working days"),
    ])
    def test_said_in_full(self, name, said):
        import re

        from core.semantic_model import _said_in_full

        assert re.search(_said_in_full(name), said, re.IGNORECASE)


_METRICS = [{"name": "Number of Buying Customers", "synonyms": "buying customers, active customers"},
            {"name": "Sales Amount", "synonyms": "sales, revenue"}]


class TestTheJoinGraph:

    def test_a_measures_name_is_left_out(self):
        from core.semantic_model import without_measure_names

        assert "customer" not in without_measure_names("Buying customers by product category", _METRICS).lower()
        assert "customer" in without_measure_names("Sales by customer", _METRICS).lower()
        # A one-word name leaves the question as it was.
        assert without_measure_names("Sales by customer", _METRICS) == "Sales by customer"

    @staticmethod
    def _graph() -> dict:
        return {"entities": [
            {"entity_name": name, "table_name": name, "schema_name": "dbo", "entity_type": kind}
            for name, kind in (("FactInternetSales", "fact"), ("DimCustomer", "dimension"),
                               ("DimProduct", "dimension"))]}

    def test_a_table_named_only_by_the_measure_is_not_required(self):
        from core.semantic_resolution import build_planner_alignment

        aligned = build_planner_alignment(
            graph=self._graph(), graph_ctx={"detected": ["DimCustomer", "DimProduct"]}, semantic_plan={},
            metric_formula_tables={"dbo.FactInternetSales"}, named_only_by_a_measure={"DimCustomer"})
        assert aligned["required_entities"] == ["DimProduct", "FactInternetSales"]
        assert aligned["dropped_measure_word_entities"] == ["DimCustomer"]

    def test_a_table_the_plan_needs_is_kept(self):
        from core.semantic_resolution import build_planner_alignment

        plan = {"fields": [{"table": "dbo.DimCustomer", "column": "Gender", "role": "attribute",
                            "enforcement": "required"}]}
        aligned = build_planner_alignment(
            graph=self._graph(), graph_ctx={"detected": ["DimCustomer"]}, semantic_plan=plan,
            metric_formula_tables={"dbo.FactInternetSales"}, named_only_by_a_measure={"DimCustomer"})
        assert "DimCustomer" in aligned["required_entities"]
        assert aligned["dropped_measure_word_entities"] == []
