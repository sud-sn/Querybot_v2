"""A follow-up that takes the comparison off the answer keeps the report, never asks for the comparison again.

Found with a real model (gpt-4o) on a sample workbook: "net sales by store this year vs last year", then "drop the
vs last year column". The AI planned the follow-up right -- this year so far by store, no comparison -- and the
words "vs last year" in it were read in code as a request to compare with last year, so the answer came back the
same comparison, with "drop the vs last year column" said to be left out. Asked once more ("now hide the comparison
with last year"), the AI took "last year" for the period.

What a command takes off the answer ("drop", "remove", "hide", "take out", "without") asks for nothing: the report
stays this year so far by store, biggest first; a comparison asked beside the command ("drop the share and compare
with last year") stays a comparison.

Synthetic data only.
"""

from __future__ import annotations

import pytest

from core2.plan import asks
from core2.service import Session
from tests import shop_data
from tests.shop_data import ask, headline, rows

THIS_YEAR = {"window": {"kind": "to_date", "unit": "year"}}
COMPARED = {**THIS_YEAR, "compare": {"kind": "same_period_last_year"}}
FIRST = {"asked": [{"words": "net sales", "kind": "figure", "plan": "net_amount"},
                   {"words": "by store", "kind": "field", "plan": "store"},
                   {"words": "this year", "kind": "period", "plan": "time"},
                   {"words": "vs last year", "kind": "comparison", "plan": "compare"}],
         "intent": "compare", "measures": ["net_amount"], "group_by": ["store"], "time": COMPARED}


@pytest.fixture(scope="module")
def shop():
    return shop_data.build()


def _this_year_by_store(shop) -> list[tuple[str, float]]:
    return [(name, float(net)) for name, net in shop[0].execute(
        "SELECT s.store_name, SUM(o.net_amount) FROM order_lines o JOIN stores s USING (store_id) "
        "WHERE o.status <> 'Cancelled' AND o.order_date BETWEEN DATE '2026-01-01' AND DATE '2026-08-24' "
        "GROUP BY 1 ORDER BY 2 DESC").fetchall()]


def _compared_then(shop, question: str, plan: dict) -> dict:
    session = Session()
    first = ask(shop, "net sales by store this year vs last year", FIRST, session)
    assert any(k.endswith("_prior") for k in rows(first)[0]), "the answer above compares"
    return ask(shop, question, plan, session)


def _is_this_year_by_store(shop, payload: dict) -> None:
    got = [(r["store_name"], float(r["net_amount"])) for r in rows(payload)]
    assert got == pytest.approx(_this_year_by_store(shop)), "this year so far by store, biggest first"
    assert not any(k.endswith(("_prior", "_change")) for k in rows(payload)[0]), rows(payload)[0]
    assert "against" not in headline(payload), headline(payload)
    assert not payload.get("left_out"), payload.get("left_out")


# As the AI planned it: the comparison taken off.
TAKEN_OFF = {"asked": [{"words": "drop the vs last year column", "kind": "comparison", "plan": None}],
             "intent": "value", "measures": ["net_amount"], "group_by": ["store"], "time": THIS_YEAR,
             "follow_up": "refine"}


def test_the_comparison_dropped_is_the_report_without_it(shop):
    _is_this_year_by_store(shop, _compared_then(shop, "drop the vs last year column", TAKEN_OFF))


@pytest.mark.parametrize("question", ["remove the vs last year column", "drop the comparison with last year",
                                      "hide last year", "take out last year's column", "without the vs last year"])
def test_the_comparison_taken_off_by_a_model_that_kept_it(shop, question):
    # A model that copied the comparison from the plan above: taken off in code.
    kept = {**TAKEN_OFF, "asked": [{"words": question, "kind": "comparison", "plan": None}], "intent": "compare",
            "time": COMPARED}
    _is_this_year_by_store(shop, _compared_then(shop, question, kept))


def test_words_taking_the_comparison_off_never_set_the_period(shop):
    session = Session()
    ask(shop, "net sales by store this year vs last year", FIRST, session)
    ask(shop, "drop the vs last year column", TAKEN_OFF, session)
    # As the AI planned it: "with last year" read as the period.
    last_year = {"intent": "value", "measures": ["net_amount"], "group_by": ["store"],
                 "time": {"window": {"kind": "previous", "unit": "year"}}, "follow_up": "refine"}
    _is_this_year_by_store(shop, ask(shop, "now hide the comparison with last year", last_year, session))


@pytest.mark.parametrize("question", [
    "drop the vs last year column", "remove the vs last year column", "drop the comparison with last year",
    "take out last year's column", "hide the year before", "without the vs last year", "drop the change vs last year",
    "drop the yoy column", "drop last year", "enlève la colonne vs l'an dernier", "please hide the comparison",
    "can you remove the vs last year column?", "could you hide the comparison please", "drop the vs. last year column",
    "masquer l'année précédente"])
def test_a_command_taking_the_comparison_off_is_no_comparison_asked(question):
    assert asks.ask_of(question) is None, question


@pytest.mark.parametrize("question", [
    "drop the vs last year column", "remove the vs last year column", "hide last year", "without the vs last year",
    "enlève la colonne vs l'an dernier", "can you remove the vs last year column?", "could you hide the comparison please",
    "drop the vs. last year column", "masquer l'année précédente"])
def test_it_takes_the_comparison_off(question):
    assert asks.takes_off_comparison(question), question


@pytest.mark.parametrize("question", [
    # A comparison asked beside the command stays one.
    "drop the share and compare with last year", "Drop the margin and add units sold this year vs last year",
    "hide the share, then compare with last year", "remove returns and compare with last year",
    # Falls, and comparisons asked outright.
    "net sales by store vs last year", "drop by store vs last year", "revenue drop vs last year by region",
    "stores whose sales drop compared to last year", "how are sales doing vs last year?",
    "sales by store without returns vs last year",
    # A word of the business, never a command ("lose" customers, "take out" orders, "remove" cancelled ones).
    "and which customers did we gain and lose vs last year?", "now just remove cancelled orders vs last year",
    "dine in and take out sales vs last year", "Take out orders this year vs last year"])
def test_a_comparison_asked_stays_one(question):
    ask_ = asks.ask_of(question)
    assert ask_ is not None and ask_.kind == "compare" and ask_.said, question
    assert not asks.takes_off_comparison(question), question


@pytest.mark.parametrize("question", ["drop the % change column", "drop the change column", "drop the share column",
                                      "drop the % change vs last year"])
def test_dropping_the_change_alone_keeps_both_years(question):
    assert not asks.takes_off_comparison(question), question


def test_the_one_ranked_by_its_growth_stays_when_the_comparison_goes(shop):
    session = Session()
    grew = ask(shop, "which customer grew the most vs last year",
               {"intent": "compare", "measures": ["net_amount"], "group_by": ["customer.name"], "time": COMPARED,
                "sort": [{"by": "pct_change", "desc": True}], "limit": 1}, session)
    (top,) = [r["customer_name"] for r in rows(grew)]
    # As a model that kept the comparison wrote it: taken off in code, the ranking by growth with it.
    dropped = ask(shop, "drop the vs last year column",
                  {"intent": "compare", "measures": ["net_amount"], "group_by": ["customer.name"], "time": COMPARED,
                   "sort": [{"by": "pct_change", "desc": True}], "limit": 1, "follow_up": "refine"}, session)
    assert [r["customer_name"] for r in rows(dropped)] == [top], "the one that grew most, not the biggest"
    assert not any(k.endswith(("_prior", "_change")) for k in rows(dropped)[0]), rows(dropped)[0]
    biggest = max(_this_year_by_customer(shop), key=lambda kv: kv[1])[0]
    assert biggest != top, "the data must tell the two orders apart"


def _this_year_by_customer(shop) -> list[tuple[str, float]]:
    return [(name, float(net)) for name, net in shop[0].execute(
        "SELECT c.customer_name, SUM(o.net_amount) FROM order_lines o JOIN customers c USING (customer_id) "
        "WHERE o.status <> 'Cancelled' AND o.order_date BETWEEN DATE '2026-01-01' AND DATE '2026-08-24' "
        "GROUP BY c.customer_id, 1").fetchall()]
