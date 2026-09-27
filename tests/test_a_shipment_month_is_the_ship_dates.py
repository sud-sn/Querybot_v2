"""
A shipment month is the ship date's month.

"Sales by ship month in 2025" was answered by the ship date. "Sales by
shipment month in 2025" was answered too -- by the order date, the sales'
default, without a word: the date was named "Shipping Date", and its event
was read as ship, shipped, shipping, never as the shipment. "Sales by month
of shipment" and French "ventes par mois d'expédition" were refused, and
"ventes par mois de commande" as well, where "order month" was answered.

An event's date now answers to the noun made of its verb -- the shipment,
the payment, the delivery -- before its period or after it ("month of
shipment"), and a period followed by its event is a period, not a
breakdown. French names the date's period before its event; "mois de
commande", "semaine d'expédition", "trimestre de facturation" are read in
English order, and in the singular -- "month of orders" named the orders
counted as surely as their month. "Nombre de commandes par mois" still
counts orders.

tests/star_harness.py keeps two years of orders, each shipped a few days
after it was placed -- across a month's end for the second of each month.
"""

from __future__ import annotations

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("shipment-month")) as built:
        yield built


def _sales_by_month(day_of: int) -> dict[int, float]:
    """Sales in 2025 by the month of the line's order (2) or ship (3) day."""
    totals: dict[int, float] = {}
    for line in star.orders():
        day = line[day_of]
        if day.year == 2025:
            totals[day.month] = totals.get(day.month, 0) + line[7] * star.PRODUCTS[line[4]][4]
    return totals


def _answered(answer: dict) -> dict[int, float]:
    assert answer["model_wrote_sql"] is False
    assert [column for column in answer["rows"][0] if column != "PERIOD"] == ["SALES_AMOUNT"]
    return {row["PERIOD"].month: row["SALES_AMOUNT"] for row in answer["rows"]}


class TestTheProductAnswers:

    @pytest.mark.parametrize("question,lang", [
        ("Sales by shipment month in 2025", "en"),
        ("Sales by month of shipment in 2025", "en"),
        ("Ventes par mois d'expédition en 2025", "fr"),
    ])
    def test_by_the_ship_date(self, warehouse, question, lang):
        assert _answered(star.ask(warehouse, question, lang)) == _sales_by_month(3)

    def test_the_order_month_in_french(self, warehouse):
        assert _answered(star.ask(warehouse, "Ventes par mois de commande en 2025", "fr")) == _sales_by_month(2)

    def test_orders_counted_by_month_are_still_counted(self, warehouse):
        answer = star.ask(warehouse, "Nombre de commandes par mois en 2025", "fr")
        assert sum(row["NUMBER_OF_ORDERS"] for row in answer["rows"]) == len(
            {line[0] for line in star.orders() if line[2].year == 2025})


_SHIPPING = {"name": "Shipping Date", "business_role": "shipping_date", "status": "approved", "confidence": 100,
             "synonyms": []}


class TestTheRule:

    @pytest.mark.parametrize("stem,noun", [
        ("ship", "shipment"), ("shipping", "shipments"), ("pay", "payment"), ("deliver", "delivery"),
    ])
    def test_the_noun_of_an_events_verb(self, stem, noun):
        from core.contextual_dates import _event_word_forms

        assert noun in _event_word_forms(stem)

    @pytest.mark.parametrize("question,phrase", [
        ("sales by shipment month in 2025", "shipment month"),
        ("sales by month of shipment in 2025", "month of shipment"),
        ("sales by month shipment en 2025", "month shipment"),
    ])
    def test_the_date_it_names(self, question, phrase):
        from core.contextual_dates import _explicit_role_matches

        assert [role["_matched_phrase"] for role in _explicit_role_matches(question, [_SHIPPING])] == [phrase]

    @pytest.mark.parametrize("question,breakdowns", [
        ("sales by month of shipment in 2025", 0),
        ("sales by shipment month in 2025", 0),
        ("sales by product in 2025", 1),
    ])
    def test_a_period_before_its_event_is_no_breakdown(self, question, breakdowns):
        from core.pipeline_helpers import _requested_breakdowns

        assert _requested_breakdowns(question) == breakdowns

    @pytest.mark.parametrize("question,read", [
        ("Ventes par mois de commande en 2025", "sales by order month en 2025"),
        ("Ventes par mois d'expédition", "sales by shipment month"),
        ("Ventes par trimestre de facturation", "sales by billing quarter"),
        ("Ventes par année de livraison", "sales by delivery year"),
        ("Nombre de commandes par mois", "count of orders by month"),
    ])
    def test_in_english_order(self, question, read):
        from core.question_normalizer import canonical_question

        assert canonical_question(question, "fr") == read
