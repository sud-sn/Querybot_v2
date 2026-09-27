"""
A French ranking is a ranking.

"Top 3 products" is asked which measure to rank them by. "Les 3 meilleurs
produits" was answered: three products, each with a "number of products
sold" of 1 -- the count of the product itself, ranked by nothing. The
question's intent was read in French, where "meilleurs" names no ranking,
so no measure was asked for; the word "produits" then matched the metric
that counts products, and the governed compiler ranked the products by it.

A French question whose French reads no more than a measure is read for a
ranking in its English as well: "les 3 meilleurs produits" is "the 3 best
products", and a ranking that names no measure in either language is asked
which. One that names its measure in French -- "par ventes", "par clients
acheteurs" -- is not asked.

tests/star_harness.py keeps eight products and the metrics Sales Amount
and Number of Buying Customers, named in French too.
"""

from __future__ import annotations

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("french-ranking")) as built:
        yield built


def _asks_which_measure(answer: dict) -> bool:
    from core.i18n import t

    replies = str(answer["replies"])
    return any(t("clar.metric.ranking", lang=lang) in replies for lang in ("en", "fr"))


class TestTheProductAnswers:

    @pytest.mark.parametrize("question,lang", [
        ("Les 3 meilleurs produits", "fr"),
        ("Top 3 products", "en"),
    ])
    def test_a_ranking_by_no_measure_asks_which(self, warehouse, question, lang):
        answer = star.ask(warehouse, question, lang)
        assert answer["executed"] == []
        assert _asks_which_measure(answer)

    def test_a_ranking_by_a_measure_named_in_french_is_answered(self, warehouse):
        answer = star.ask(warehouse, "Les 3 meilleurs produits par ventes en 2025", "fr")
        sales: dict[str, float] = {}
        for line in star.orders():
            if line[2].year == 2025:
                name = star.PRODUCTS[line[4]][2]
                sales[name] = sales.get(name, 0) + line[7] * star.PRODUCTS[line[4]][4]
        assert answer["model_wrote_sql"] is False
        assert [(row["PRODUCT"], row["SALES_AMOUNT"]) for row in answer["rows"]] == sorted(
            sales.items(), key=lambda item: -item[1])[:3]

    def test_a_measure_only_the_french_names_is_not_asked_for(self, warehouse):
        answer = star.ask(warehouse, "Les 3 meilleurs produits par clients acheteurs en 2025", "fr")
        assert not _asks_which_measure(answer)
