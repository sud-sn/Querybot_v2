"""
A question can rank the periods.

"Which month of 2025 had the highest sales?" asks for the months of 2025,
ordered by their sales. The governed compiler ranked members of a dimension
and grouped periods in time order, but a ranking of periods was neither: it
read "which month" as a breakdown by a member nothing bound and declined, and
the date reader did not see 2025 in "month of 2025" at all. The question went
to the model. Where the ranked months did come back, the answer read them as a
series and led with the last one: "2025-12 closed at $2,645.00".

"Which (day | week | month | quarter | year)" now names the grain of a
question, "month of 2025" names 2025, and the compiler writes the governed
date's buckets ordered by the measure, from the end the question asks for.
The answer names the leader, as the series names its periods. "Sales by month
in 2025" is still a series in time order, and "which product" still ranks the
products.

tests/star_harness.py keeps two years of order lines, dated by their Order Date.
"""

from __future__ import annotations

import datetime as dt

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("ranked-periods")) as built:
        yield built


def _sales(bucket, year: int = 2025) -> dict:
    totals: dict = {}
    for line in star.orders():
        if line[2].year == year:
            key = bucket(line[2])
            totals[key] = totals.get(key, 0) + line[7] * star.PRODUCTS[line[4]][4]
    return totals


def _month(day: dt.date) -> dt.date:
    return dt.date(day.year, day.month, 1)


def _quarter(day: dt.date) -> dt.date:
    return dt.date(day.year, 3 * ((day.month - 1) // 3) + 1, 1)


_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December")


def _named(month: dt.date) -> str:
    """A month as the answer names it: "October 2025"."""
    return f"{_MONTHS[month.month - 1]} {month.year}"


def _headline(answer: dict) -> str:
    return next(payload["answer"]["headline"] for kind, payload in answer["replies"] if kind == "answer")


class TestTheProductAnswers:

    def test_the_month_with_the_highest_sales(self, warehouse):
        answer = star.ask(warehouse, "Which month of 2025 had the highest sales?")
        assert answer["model_wrote_sql"] is False
        expected = sorted(_sales(_month).items(), key=lambda pair: -pair[1])
        assert [(row["PERIOD"], row["SALES_AMOUNT"]) for row in answer["rows"]] == expected
        assert _headline(answer).startswith(_named(expected[0][0]) + " ")

    def test_beside_placeholder_members(self, warehouse, monkeypatch):
        # A warehouse that keeps an "unknown" product leaves it out of a
        # ranking of products; a ranking of months ranks no product.
        import core.unknown_members as unknown_members

        monkeypatch.setattr(unknown_members, "unknown_member_policies", lambda account_id: [{
            "kind": "unknown_members", "table": "dbo.DimProduct", "entities": ["DimProduct"],
            "key_column": "ProductKey", "keys": ["-1"], "member_kinds": {"-1": "unknown"}, "member_text": {},
            "references": [{"table": "dbo.FactInternetSales", "column": "ProductKey"}]}])
        answer = star.ask(warehouse, "In 2025, which month had the highest sales?")
        assert [(row["PERIOD"], row["SALES_AMOUNT"]) for row in answer["rows"]] == sorted(
            _sales(_month).items(), key=lambda pair: -pair[1])

    def test_in_french(self, warehouse):
        answer = star.ask(warehouse, "Quel mois de 2025 a eu les ventes les plus élevées ?", "fr")
        assert [(row["PERIOD"], row["SALES_AMOUNT"]) for row in answer["rows"]] == sorted(
            _sales(_month).items(), key=lambda pair: -pair[1])

    def test_the_quarter_with_the_lowest_sales(self, warehouse):
        answer = star.ask(warehouse, "In 2025, which quarter had the lowest sales?")
        assert [(row["PERIOD"], row["SALES_AMOUNT"]) for row in answer["rows"]] == sorted(
            _sales(_quarter).items(), key=lambda pair: pair[1])

    def test_a_series_is_still_in_time_order(self, warehouse):
        answer = star.ask(warehouse, "Sales by month in 2025")
        assert [(row["PERIOD"], row["SALES_AMOUNT"]) for row in answer["rows"]] == sorted(_sales(_month).items())
        assert _headline(answer).startswith("December 2025 ")

    def test_members_are_still_ranked(self, warehouse):
        answer = star.ask(warehouse, "Which product had the highest sales in 2025?")
        assert "PERIOD" not in answer["rows"][0] and answer["rows"][0]["PRODUCT"]


class TestTheReading:

    @pytest.mark.parametrize("question,grain", [
        ("which month of 2025 had the highest sales", "month"),
        ("what quarter had the lowest sales in 2025", "quarter"),
        ("which week of 2025 had the most orders", "week"),
        ("which product had the highest sales", ""),
        ("sales by month", ""),
    ])
    def test_the_period_a_question_ranks(self, question, grain):
        from core.contextual_dates import ranked_period_grain

        assert ranked_period_grain(question) == grain

    @pytest.mark.parametrize("question", [
        "which month of 2025 had the highest sales",
        "which quarter of 2025 a eu the sales the lowest",
    ])
    def test_the_year_a_period_is_of(self, question):
        from core.contextual_dates import stated_period

        period = stated_period(question, "calendar")
        assert (period["start"], period["end"]) == ("2025-01-01", "2026-01-01")

    def test_the_grain_it_asks_for(self):
        from core.contextual_dates import requested_temporal_grain

        assert requested_temporal_grain("which month of 2025 had the highest sales") == "month"
        assert requested_temporal_grain("sales by quarter in 2025") == "quarter"
