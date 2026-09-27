"""
A date named by its event's word is that date.

"Sales by ship date in 2025" names the date the orders shipped on. The
warehouse calls that date the Shipping Date, and its names and synonyms say
"shipping", not "ship": the reader's "ship" was read only as the event's word,
a weaker sign than a date named outright. A calendar role called just "Date"
-- the inventory snapshot's -- was named outright by the question's last
word, won, and the reader was asked "Date is not a date of Sales Amount. Which
of its dates should I use?" of a date they had named.

A form of the event's word before the grain -- "ship date", "shipped month";
the French "date d'expédition" reads as "ship date" -- names the date as its
own word does, and it is the longer name for the same words, so a role named
"Date" alone gives way to it.

tests/star_harness.py keeps two years of order lines, each shipped a few days
after it was placed (the second order of a month across the month's end), and
a daily inventory snapshot whose date role is called "Date".
"""

from __future__ import annotations

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("event-word-date")) as built:
        yield built


def _sales(year: int, *, shipped: bool) -> float:
    return sum(line[7] * star.PRODUCTS[line[4]][4] for line in star.orders()
               if (line[3] if shipped else line[2]).year == year)


class TestTheProductAnswers:

    def test_sales_by_ship_date(self, warehouse):
        answer = star.ask(warehouse, "Sales by ship date in 2025")
        assert answer["model_wrote_sql"] is False
        assert "SHIPDATEKEY" in answer["sql"].upper()
        assert [list(row.values()) for row in answer["rows"]] == [[_sales(2025, shipped=True)]]

    def test_in_french(self, warehouse):
        answer = star.ask(warehouse, "Ventes par date d'expédition en 2025", "fr")
        assert [list(row.values()) for row in answer["rows"]] == [[_sales(2025, shipped=True)]]

    def test_the_order_date_is_its_own(self, warehouse):
        answer = star.ask(warehouse, "Sales by order date in 2025")
        assert [list(row.values()) for row in answer["rows"]] == [[_sales(2025, shipped=False)]]


def _role(name: str, table: str, column: str, synonyms: list[str]) -> dict:
    return {"name": name, "fact_table": table, "fact_column": column, "status": "approved",
            "synonyms": synonyms, "date_key_type": "surrogate_fk", "dimension_table": "dbo.DimDate",
            "dimension_key": "DateKey", "date_value_column": "FullDate"}


_ROLES = [
    _role("Date", "dbo.FactStock", "DateKey", []),
    _role("Shipping Date", "dbo.FactSales", "ShipDateKey", ["shipping date", "shipping month"]),
    _role("Posting Date", "dbo.FactLedger", "PostingDateKey", ["posting date"]),
]


class TestTheRule:

    @pytest.mark.parametrize("question,named", [
        ("sales by ship date in 2025", ["Shipping Date"]),
        ("sales by shipped month", ["Shipping Date"]),
        ("sales by shipped date", ["Shipping Date"]),
        ("amounts by post date", ["Posting Date"]),
        ("stock by date", ["Date"]),
        ("sales by shipping date", ["Shipping Date"]),
    ])
    def test_the_date_a_question_names(self, question, named):
        from core.contextual_dates import find_explicit_date_roles

        assert [role["name"] for role in find_explicit_date_roles(question, _ROLES)] == named
