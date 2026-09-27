"""
An accented word is folded, not shredded.

"Average unit price in 2025" was refused: "The confirmed relationships do
not connect Date to the rest of the question." The unit price is on the
sales lines; the question was taken for one about Order Quantity and Units
in Stock as well, and the stock is dated by another table's date. Both
metrics have a French synonym, "unités vendues" and "unités en stock", and
the metric matcher kept only the letters a to z of a phrase: "unités" came
out "unit s", and its "unit" was one word of evidence for every question
that said "unit price", "unit cost" or "per unit".

A metric's phrases and the question are read without their accents,
"unités" as "unites": a French synonym is matched by the French it is
written in, and lends no English word to another question. "Unit price"
names no count of units; the French reader who asks for "unités vendues"
is answered by the units sold, as before.

tests/star_harness.py keeps each sale's unit price, and the metrics Order
Quantity and Units in Stock with their French synonyms.
"""

from __future__ import annotations

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("accents")) as built:
        yield built


class TestTheProductAnswers:

    def test_a_unit_price_is_asked_of_the_sales(self, warehouse):
        answer = star.ask(warehouse, "Average unit price in 2025")
        assert "trusted join plan" not in str(answer["replies"])
        assert answer["prompts"] and "JOIN [dbo].[DimDate] ord ON fac.[OrderDateKey]" in answer["prompts"][0]
        assert "FactProductInventory]" not in answer["prompts"][0].split("## Entity graph", 1)[-1].split("\n\n")[1]

    def test_units_sold_in_french_are_still_the_units_sold(self, warehouse):
        answer = star.ask(warehouse, "Unités vendues par catégorie en 2025", "fr")
        sold: dict[str, int] = {}
        for line in star.orders():
            if line[2].year == 2025:
                category = star.CATEGORIES[star.SUBCATEGORIES[star.PRODUCTS[line[4]][3]][2]][1]
                sold[category] = sold.get(category, 0) + line[7]
        assert answer["model_wrote_sql"] is False
        assert {row["PRODUCT_CATEGORY"]: row["ORDER_QUANTITY"] for row in answer["rows"]} == sold


_ORDER_QUANTITY = next(metric for metric in star.METRICS if metric["name"] == "Order Quantity")
_IN_STOCK = next(metric for metric in star.METRICS if metric["name"] == "Units in Stock")


class TestTheRule:

    @pytest.mark.parametrize("text,read", [
        ("Unités vendues", "unites vendues"),
        ("Quantité commandée", "quantite commandee"),
        ("unit price", "unit price"),
    ])
    def test_the_words_without_their_accents(self, text, read):
        from core.metric_scope import _norm

        assert _norm(text) == read

    @pytest.mark.parametrize("metric", [_ORDER_QUANTITY, _IN_STOCK])
    def test_a_unit_price_names_no_count_of_units(self, metric):
        from core.metric_scope import _phrase_score

        assert _phrase_score(metric, "average unit price in 2025") == 0

    def test_a_french_synonym_is_matched_in_french(self):
        from core.metric_scope import _phrase_score

        assert _phrase_score(_ORDER_QUANTITY, "units sold by category", reader_question="Unités vendues par catégorie") > 100
