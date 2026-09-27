"""
A label's French twin is found whatever the case of its name.

A warehouse built for two languages keeps a label twice, and a French reader is
shown the French one (core/label_language.py). The twins were found only in
underscore names -- ITM_GRP_DSC and ITM_GRP_FR_DSC -- by cutting a name at its
underscores. A camel-case warehouse keeps them as EnglishProductName and
FrenchProductName, which have none, so on a made-up retailer's warehouse "les 3
meilleurs produits par montant des ventes" was answered to a French reader with
the products' English names, and the French ones were never read.

A name is now read as its words, whatever its case: FrenchProductName is the
product's name in French, beside EnglishProductName, and ProductNameFr beside
ProductName. Underscore twins are found as before.

tests/star_harness.py keeps every product's name in English and French.
"""

from __future__ import annotations

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("label-twins")) as built:
        yield built


def _best(n: int) -> list[tuple[int, float]]:
    totals: dict = {}
    for line in star.orders():
        totals[line[4]] = totals.get(line[4], 0) + line[7] * star.PRODUCTS[line[4]][4]
    return sorted(totals.items(), key=lambda item: -item[1])[:n]


class TestTheProductAnswers:

    @pytest.mark.parametrize("question,lang,name", [
        ("Les 3 meilleurs produits par montant des ventes", "fr", 2),
        ("Top 3 products by sales amount", "en", 1),
        ("Top 3 products by sales amount in French", "en", 2),
    ])
    def test_the_products_are_named_in_the_readers_language(self, warehouse, question, lang, name):
        answer = star.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is False
        assert [(row["PRODUCT"], row["SALES_AMOUNT"]) for row in answer["rows"]] == [
            (star.PRODUCTS[key][name], total) for key, total in _best(3)]


class TestTheTwins:

    @pytest.mark.parametrize("columns,twins", [
        (["ProductKey", "EnglishProductName", "FrenchProductName"],
         [("EnglishProductName", "FrenchProductName", "fr")]),
        (["ProductName", "ProductNameFr", "ProductKey"], [("ProductName", "ProductNameFr", "fr")]),
        (["EnglishMonthName", "FrenchMonthName", "SpanishMonthName"],
         [("EnglishMonthName", "FrenchMonthName", "fr")]),
        (["ITM_GRP_DSC", "ITM_GRP_FR_DSC", "PRC_FR_DT"], [("ITM_GRP_DSC", "ITM_GRP_FR_DSC", "fr")]),
        (["item_nm", "item_fr_nm"], [("item_nm", "item_fr_nm", "fr")]),
        # Not labels: a date "from", and a French name with no English one.
        (["FromDate", "FrenchProductName", "ProductKey"], []),
    ])
    def test_found(self, columns, twins):
        from core.label_language import language_twins

        assert [(t["base"], t["twin"], t["language"]) for t in language_twins(columns)] == twins
