"""What a question names in so many words, the plan keeps, whatever the AI wrote. Found by gpt-4o on the agentic
conversation books:

* "How many ingredient lots did we receive in Q1 2026, by supplier?" was counted as ingredients: the AI took the
  measure whose name is inside the one the question names ("Number of ingredients" in "Number of ingredient lots").
  The figure the question names in full is the one counted.
* "Nombre de réclamations à l'agence "Midtown" par catégorie en 2025", asked after a month-by-month answer, came back
  month by month and by category: the AI kept the months from above, though the question names its own split and
  its own period and asks for no series. A breakdown asking no series keeps no grain, kept or added.

* "daily spend on "Rent" in march 2026" came back Monday to Sunday: the AI grouped by the day of the week with no
  grain. "Daily" and "by day" are each day, unless the question names the week's days.

Invented retail data; the plans are written as gpt-4o wrote them.
"""

from __future__ import annotations

import datetime as dt

import pytest

from core2.plan.ir import Plan
from core2.plan.planner import settle
from evals.core2 import domains
from evals.core2.compile_eval import learn
from tests.test_the_product_looks_before_it_answers import MAY as MAY_ASKED
from tests.test_the_product_looks_before_it_answers import Chat, _days, _net, retail  # noqa: F401 - fixture

YEAR = {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}
MAY = {"kind": "between", "start": "2026-05-01", "end": "2026-05-31"}


@pytest.fixture(scope="module")
def model():
    return learn(domains.build("retail"), "descriptive")[1]


def _plan(**kw) -> Plan:
    return Plan.model_validate({"kind": "query", **kw})


# ── the figure the question names in full ─────────────────────────────────────


def test_the_figure_named_in_full_is_counted_never_one_inside_its_name(model):
    asked = "How many order lines did we have in May 2026, by store?"
    plan = _plan(intent="breakdown", measures=["number_of_orders"], group_by=["store.name"], time={"window": MAY},
                 sort=[{"by": "number_of_orders", "desc": True}])
    settled = settle(plan, model, asked, None)
    assert settled.measures == ["number_of_order_lines"]
    assert [s.by for s in settled.sort] == ["number_of_order_lines"]


@pytest.mark.parametrize("asked", ["How many orders did we take in May 2026, by store?",
                                   "Number of orders and order lines in May 2026"])
def test_a_figure_the_question_names_itself_stays(model, asked):
    plan = _plan(intent="breakdown", measures=["number_of_orders"], group_by=["store.name"], time={"window": MAY})
    assert settle(plan, model, asked, None).measures == ["number_of_orders"]


def test_a_figure_the_question_does_not_name_is_left_to_the_ai(model):
    # "Sales" names no measure by its own words: the AI's reading stands.
    plan = _plan(intent="value", measures=["net_amount"], time={"window": MAY})
    assert settle(plan, model, "What were sales in May 2026?", None).measures == ["net_amount"]


# ── no months a breakdown does not ask for ───────────────────────────────────


def _monthly() -> Plan:
    return _plan(intent="trend", measures=["net_amount"], time={"window": YEAR, "grain": "month"})


@pytest.mark.parametrize("asked", ['Net sales of "Riverside Store" by category in 2025',
                                   'Ventes nettes du magasin "Riverside Store" par catégorie en 2025'])
def test_a_breakdown_naming_its_own_split_and_period_keeps_no_months_from_above(model, asked):
    plan = _plan(intent="breakdown", measures=["net_amount"], group_by=["category.name"],
                 filters=[{"field": "store.name", "op": "eq", "values": ["Riverside Store"]}],
                 time={"window": YEAR, "grain": "month"}, follow_up="refine")
    assert settle(plan, model, asked, _monthly()).time.grain is None


def test_a_breakdown_asked_afresh_gets_no_months_it_did_not_ask_for(model):
    plan = _plan(intent="breakdown", measures=["net_amount"], group_by=["category.name"],
                 time={"window": YEAR, "grain": "month"})
    assert settle(plan, model, "Net sales by category in 2025", None).time.grain is None


@pytest.mark.parametrize("asked", ["Net sales by category by month in 2025", "Monthly net sales by category in 2025",
                                   "Ventes nettes par catégorie et par mois en 2025",
                                   "Net sales by category in each month of 2025"])
def test_a_breakdown_asking_for_months_keeps_them(model, asked):
    plan = _plan(intent="breakdown", measures=["net_amount"], group_by=["category.name"],
                 time={"window": YEAR, "grain": "month"})
    assert settle(plan, model, asked, None).time.grain == "month"


SIX_MONTHS = {"kind": "between", "start": "2026-04-01", "end": "2026-09-30"}


@pytest.mark.parametrize("asked, grain", [("Net sales by category over the last 2 quarters", "quarter"),
                                          ("Net sales by category in the past 6 months", "month"),
                                          ("Ventes nettes par catégorie sur les 6 derniers mois", "month"),
                                          ("Ventes nettes par catégorie au cours des 2 derniers trimestres",
                                           "quarter")])
def test_a_period_said_in_months_or_quarters_asks_for_no_series(model, asked, grain):
    # "REV journals by division over the last 2 quarters" came back quarter by quarter: "the last 2 quarters" is
    # how long the period is, not a split by quarter.
    plan = _plan(intent="breakdown", measures=["net_amount"], group_by=["category.name"],
                 time={"window": SIX_MONTHS, "grain": grain})
    assert settle(plan, model, asked, None).time.grain is None


@pytest.mark.parametrize("asked", ["Net sales by category for each of the last 6 months",
                                   "Net sales by category month by month over the last 6 months",
                                   "Ventes nettes par catégorie chaque mois sur les 6 derniers mois"])
def test_a_series_asked_inside_a_period_said_in_months_keeps_its_months(model, asked):
    plan = _plan(intent="breakdown", measures=["net_amount"], group_by=["category.name"],
                 time={"window": SIX_MONTHS, "grain": "month"})
    assert settle(plan, model, asked, None).time.grain == "month"


def test_a_split_of_the_series_on_screen_naming_no_period_is_left_as_it_was_read(model):
    # "And by category?" after a monthly series names no period of its own: the series on screen, split.
    plan = _plan(intent="breakdown", measures=["net_amount"], group_by=["category.name"],
                 time={"window": YEAR, "grain": "month"}, follow_up="refine")
    assert settle(plan, model, "and by category?", _monthly()).time.grain == "month"


@pytest.fixture(scope="module")
def inventory():
    return learn(domains.build("inventory"), "descriptive")[1]


def test_the_figure_named_in_full_passes_the_check_of_how_many(inventory):
    # "How many item groups" is no "how many items": the check of what "how many" counts lets the figure the
    # question names in full stand (it turned "how many ingredient lots" back to the ingredients).
    from core2.plan.planner import check

    groups = _plan(intent="value", measures=["number_of_item_groups"])
    assert check(groups, inventory, dt.date(2026, 10, 15), "How many item groups do we have?")[0] == []
    # "How many items" still asks for items, never another figure of them.
    other = next(m.slug for m in inventory.measures.values() if not m.slug.startswith("number_of_"))
    assert check(_plan(intent="value", measures=[other]), inventory, dt.date(2026, 10, 15),
                 "How many items do we have?")[0]



# ── "daily" is each day ───────────────────────────────────────────────────────


def test_daily_with_no_grain_from_the_ai_is_each_day(retail):  # noqa: F811
    # "daily spend on "Rent" in march 2026" came back as Monday to Sunday ("Tuesday leads"): the AI grouped by the
    # day of the week and set no grain at all, so the rule for "by day" with a daily grain never saw it.
    got = Chat(retail).ask('daily net sales of "Toothpaste Small" in May 2026', {
        **_net(product="Toothpaste Small"), "intent": "breakdown", "group_by": ["time:day_of_week"], "time": MAY_ASKED})
    assert [str(r["period"])[:10] for r in got["export_rows"]] == _days(retail, "Toothpaste Small")
    # Asked by weekday, it stays by weekday.
    weekday = Chat(retail).ask("Daily net sales by day of the week in May 2026", {
        "measures": ["net_amount"], "intent": "breakdown", "group_by": ["time:day_of_week"], "time": MAY_ASKED})
    assert "day_of_week" in weekday["export_rows"][0]


# ── a bar the question never sets ────────────────────────────────────────────


def _at_least_one(**kw) -> Plan:
    return _plan(intent="count", measures=["number_of_order_lines"], time={"window": MAY},
                 per={"of": "store", "summary": "count", "op": "gte", "values": [1]}, **kw)


@pytest.mark.parametrize("asked", ["How many order lines did we have in May 2026, by store?",
                                   "Number of order lines per store in May 2026"])
def test_a_figure_by_members_is_never_the_members_over_a_bar_it_never_sets(model, asked):
    # "How many ingredient lots did we receive in Q1 2026, by supplier?" was planned as the suppliers with at least
    # one lot: one number, the count of suppliers, where the question asks each supplier's lots.
    settled = settle(_at_least_one(), model, asked, None)
    assert settled.per is None and settled.intent == "breakdown"
    assert settled.measures == ["number_of_order_lines"] and settled.group_by == ["store"]
    # The AI reached the stores through each customer's home store; the order lines name their own store.
    assert settle(_at_least_one(via={"store": "customer"}), model, asked, None).via == {}


@pytest.mark.parametrize("asked", ["How many stores had at least one order line in May 2026?",
                                   "How many stores had order lines in May 2026, by region?",
                                   "Number of stores with 1 or more order lines in May 2026"])
def test_members_over_a_bar_the_question_sets_stay_counted(model, asked):
    assert settle(_at_least_one(), model, asked, None).per is not None
