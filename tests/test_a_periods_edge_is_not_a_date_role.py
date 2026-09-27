"""
A period's edge is not a date role.

A question names the close or the opening of a period in words of their own --
"stock at the end of 2024", "orders at the start of 2025" -- and a date role
named for the same words was taken to be asked for: a product's End Date, a
promotion's Start Date. On a made-up retailer's warehouse "units in stock at the
end of 2024" was asked "End Date is not a date of Units in Stock. Which of its
dates should I use?", and "sales amount at the end of 2025" which of two product
dates it meant.

"End of", "start of" and "close of" are a period's own words, never an
event's, and so are "month end", "year-end" and "quarter-end" -- which were
read, in French word order, as "mois de fin": End Date's month. Nor is a
measure's own name a date's: "month-end units in stock" was the End Date's
month too. A date role named outright ("by end date", "by end month") or by
its event ("products that ended in 2024") is found as before.

tests/star_harness.py keeps when each product was first and last sold.
"""

from __future__ import annotations

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("period-edge")) as built:
        yield built


class TestTheProductAnswers:

    def test_sales_at_the_end_of_a_year(self, warehouse):
        answer = star.ask(warehouse, "Sales amount at the end of 2025")
        assert answer["model_wrote_sql"] is False
        assert [list(row.values()) for row in answer["rows"]] == [[
            sum(line[7] * star.PRODUCTS[line[4]][4] for line in star.orders() if line[2].year == 2025)]]


def _role(name: str, table: str) -> dict:
    return {"name": name, "business_role": name.lower().replace(" ", "_"), "status": "approved",
            "fact_table": table, "fact_column": name.replace(" ", ""), "date_key_type": "timestamp",
            "confidence": 98, "synonyms": []}


_ROLES = [_role("End Date", "dbo.DimProduct"), _role("Start Date", "dbo.DimPromotion"),
          _role("Close Date", "dbo.DimAccount")]


class TestTheWords:

    @pytest.mark.parametrize("question,named", [
        ("Units in stock at the end of 2024", []),
        ("Orders at the start of 2025", []),
        ("Stock at the close of last month", []),
        ("Stock at year-end 2024", []),
        ("Units in stock at quarter-end", []),
        ("Sales by end month", ["End Date"]),
        ("Products that ended in 2024", ["End Date"]),
        ("Accounts by close date", ["Close Date"]),
        ("Sales by end date in 2024", ["End Date"]),
        ("Promotions by start date", ["Start Date"]),
    ])
    def test_the_roles_named(self, question, named):
        from core.contextual_dates import _explicit_role_matches

        assert [role["name"] for role in _explicit_role_matches(question, _ROLES)] == named

    def test_a_measure_named_for_its_months_end(self):
        from core.contextual_dates import _explicit_role_matches

        assert _explicit_role_matches("Month-end units in stock at the end of 2024", _ROLES,
                                      measure_phrases=["month end units in stock"]) == []
