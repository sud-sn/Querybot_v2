"""
French names the parties and places a warehouse keeps.

A French question is read in English before its words are matched to the
warehouse's names, through a lexicon of the words every warehouse shares. It
kept a handful of entity nouns -- customers, products, countries, warehouses --
and none of the attributes those tables are asked by: "Ventes par sexe du
client" reached the planner as "sales by sexe of the customer", bound nothing,
and was not answered, where "Sales by customer gender" was. Nor the reseller,
the territory or the sales representative a sales warehouse keeps; and a
French compound puts its head noun first -- "type d'entreprise" is the
business type -- so a word-by-word reading cannot find the English name.

The lexicon now reads "sexe" as gender, "état civil" as marital status, "type
d'entreprise" as business type, and the reseller, territory and sales
representative, in either number. Trade nouns stay in the industry packs.

tests/star_harness.py keeps each customer's gender.
"""

from __future__ import annotations

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("french-parties")) as built:
        yield built


def _by_gender(value) -> dict[str, float]:
    totals: dict[str, float] = {}
    for line in star.orders():
        gender = star.CUSTOMERS[line[5]][3]
        totals[gender] = totals.get(gender, 0) + value(line)
    return totals


class TestTheProductAnswers:

    def test_sales_by_the_customers_gender(self, warehouse):
        answer = star.ask(warehouse, "Ventes par sexe du client", "fr")
        assert answer["model_wrote_sql"] is False
        assert {row["GENDER"]: row["SALES_AMOUNT"] for row in answer["rows"]} == _by_gender(
            lambda line: line[7] * star.PRODUCTS[line[4]][4])

    def test_buying_customers_by_gender(self, warehouse):
        answer = star.ask(warehouse, "Nombre de clients acheteurs par sexe", "fr")
        buyers: dict[str, set[int]] = {}
        for line in star.orders():
            buyers.setdefault(star.CUSTOMERS[line[5]][3], set()).add(line[5])
        assert {row["GENDER"]: row["NUMBER_OF_BUYING_CUSTOMERS"] for row in answer["rows"]} == {
            gender: len(customers) for gender, customers in buyers.items()}


class TestTheLexicon:

    @pytest.mark.parametrize("question,read", [
        ("Ventes par sexe du client", "sales by gender of customer"),
        ("Clients par état civil", "customers by marital status"),
        ("Ventes aux revendeurs par type d'entreprise", "sales to resellers by business type"),
        ("Ventes par type d’entreprise", "sales by business type"),
        ("Revendeurs par types d'entreprise", "resellers by business types"),
        ("Ventes par territoire", "sales by territory"),
        ("Quotas des territoires", "quotas of territories"),
        ("Ventes par représentant commercial", "sales by sales representative"),
        ("Représentants commerciaux par région", "sales representatives by region"),
        ("Ventes du revendeur", "sales of reseller"),
    ])
    def test_a_french_name_read_in_english(self, question, read):
        from core.question_normalizer import canonical_question

        assert canonical_question(question, "fr") == read
