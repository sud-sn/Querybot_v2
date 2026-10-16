"""A comparison of this year so far shows last year in full beside it, with the totals an analyst would add.

"What is my revenue comparison?" compares 1 January to 14 June 2026 with the same days of 2025, month by month.
The ask is "last year by month next to this year to date": last year's July to December were not shown, and
nothing said what all of 2025 came to, what the year so far adds up to against the same days, or what the rest of
2026 would need to match 2025. The difference and the % were only in the table, with no total row.

Now the answer adds, from one more query: last year's months after this year's last one (drawn and listed as
last year's only), all of last year, the whole of the month in progress last year, and what the rest of this year
needs. The year so far against the same days is a totals row and four figures under the lead, from the rows
themselves. A comparison of whole periods (the last 12 months against the 12 before) has nothing more to add, and
a figure that does not add up over months (an average) is not totalled.

Invented retail data whose last day is 14 Jun 2026, asked on 15 Oct 2026; every figure checked against SQL
written by hand.
"""

from __future__ import annotations

import json

import pytest

from core2.answer.facts import amount
from core2.service import Services, Session, answer_question
from tests.test_the_product_looks_before_it_answers import LINES, LIVE, TODAY
from tests.test_the_product_looks_before_it_answers import retail as retail  # noqa: F401 - fixture

COMPARE = {"kind": "query", "intent": "compare", "measures": ["net_amount"]}


def _ask(retail, question: str, plan: dict) -> dict:
    _built, model, warehouse, index = retail
    answers = [json.dumps(plan)]
    services = Services(model=model, warehouse=warehouse, complete=lambda s, t: answers.pop(0), index=index,
                        today=TODAY)
    return answer_question(question, services, Session())


def _one(retail, sql: str) -> float:
    return float(retail[0].con.execute(sql).fetchone()[0])


def _sum(retail, start: str, end: str) -> float:
    return _one(retail, f"SELECT SUM(net_amount) FROM ({LINES}) l WHERE day >= DATE '{start}' AND day < DATE '{end}' "
                        f"AND {LIVE}")


def _by_month_2025(retail) -> dict[int, float]:
    rows = retail[0].con.execute(f"""SELECT month(day), SUM(net_amount) FROM ({LINES}) l
        WHERE day >= DATE '2025-01-01' AND day < DATE '2026-01-01' AND {LIVE} GROUP BY 1""").fetchall()
    return {m: float(v) for m, v in rows}


def test_last_years_months_after_this_years_last_are_shown_as_last_years_only(retail):
    got = _ask(retail, "What is my revenue comparison?", COMPARE)
    months = _by_month_2025(retail)
    rest = (got.get("data") or {}).get("reference_rows") or []
    # July to December 2025, under the months of 2026 they line up with, with no figure for 2026.
    assert [r["period"][:7] for r in rest] == [f"2026-{m:02d}" for m in range(7, 13)], rest
    for r, m in zip(rest, range(7, 13)):
        assert r["net_amount"] is None and r["net_amount_prior"] == pytest.approx(months[m])
    drawn = [r for r in got["chart"]["rows"] if str(r.get("period", ""))[:7] >= "2026-07"]
    assert len(drawn) == 6 and all(r.get("net_amount") is None for r in drawn), drawn


def test_the_year_so_far_and_last_year_in_full_are_totals_rows(retail):
    got = _ask(retail, "What is my revenue comparison?", COMPARE)
    now, before = _sum(retail, "2026-01-01", "2026-06-15"), _sum(retail, "2025-01-01", "2025-06-15")
    full = _sum(retail, "2025-01-01", "2026-01-01")
    totals = got["data"]["totals"]
    so_far, whole = totals[0], totals[-1]
    assert so_far["label"] == "1 Jan – 14 Jun, both years"
    assert so_far["values"]["net_amount"] == pytest.approx(now)
    assert so_far["values"]["net_amount_prior"] == pytest.approx(before)
    assert so_far["values"]["net_amount_change"] == pytest.approx(now - before)
    assert so_far["values"]["net_amount_pct_change"] == pytest.approx((now / before - 1) * 100)
    assert whole["label"] == "2025 in full" and whole["label_fr"] == "2025 en entier"
    assert whole["values"]["net_amount_prior"] == pytest.approx(full) and whole["values"].get("net_amount") is None


def test_four_figures_under_the_lead_and_last_year_in_full(retail):
    got = _ask(retail, "What is my revenue comparison?", COMPARE)
    now, before = _sum(retail, "2026-01-01", "2026-06-15"), _sum(retail, "2025-01-01", "2025-06-15")
    full = _sum(retail, "2025-01-01", "2026-01-01")
    tiles = {t["key"]: t for t in got["compared"]["tiles"]}
    assert list(tiles) == ["now", "before", "change", "pct", "full"]
    assert tiles["now"]["label"] == "1 Jan – 14 Jun 2026" and tiles["now"]["value"] == pytest.approx(now)
    assert tiles["before"]["label"] == "Same days of 2025" and tiles["before"]["value"] == pytest.approx(before)
    assert tiles["change"]["value"] == pytest.approx(now - before)
    assert tiles["pct"]["value"] == pytest.approx(now / before - 1) and tiles["pct"]["format"] == "percent"
    assert tiles["full"]["label"] == "2025 in full" and tiles["full"]["value"] == pytest.approx(full)


def test_what_the_rest_of_the_year_needs_and_the_whole_month_last_year(retail):
    got = _ask(retail, "What is my revenue comparison?", COMPARE)
    now, full = _sum(retail, "2026-01-01", "2026-06-15"), _sum(retail, "2025-01-01", "2026-01-01")
    june = _sum(retail, "2025-06-01", "2025-07-01")
    facts = {f["kind"]: f for f in got["reading"]["facts"]}
    needs = facts["rest_of_year"]
    assert needs["en"] == (f"To match 2025's {amount(full, 'currency')}, the rest of 2026 needs "
                           f"{amount(full - now, 'currency')}."), needs["en"]
    assert needs["figures"] == [pytest.approx(full), pytest.approx(full - now)]
    whole = facts["whole_month"]
    assert whole["en"] == f"All of June 2025 came to {amount(june, 'currency')}.", whole["en"]
    assert whole["fr"] == f"Juin 2025 entier : {amount(june, 'currency', lang='fr')}.", whole["fr"]


def test_whole_periods_compared_have_nothing_more_to_add(retail):
    got = _ask(retail, "Net amount for the last 12 months against the 12 before, by month",
               {**COMPARE, "time": {"window": {"kind": "between", "start": "2025-06-01", "end": "2026-05-31"},
                                    "grain": "month", "compare": {"kind": "previous_period"}}})
    assert got.get("data", {}).get("rows")
    assert not (got.get("data") or {}).get("reference_rows")
    assert all("in full" not in t["label"] for t in (got.get("data") or {}).get("totals") or [])
    assert "rest_of_year" not in {f["kind"] for f in got["reading"]["facts"]}
