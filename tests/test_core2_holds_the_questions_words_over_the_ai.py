"""What a question's own words settle, the new core holds to, whatever the AI writes.

Found by asking a real model (gpt-4o) the workbooks' threads, step by step, beside the plans written by hand:

* "Fiscal year 2026" was planned as this fiscal year: the AI is now told which dates each fiscal year has.
* A follow-up that names no period lost the one of the answer above ("show the returns by warehouse" after a
  Q1 question; "why did it grow?" compared other months), and "due that month" took the current month.
* A follow-up about "those" members filtered on no value at all, or on a stand-in ("first_one",
  "warehouse_with_max_increase"): the reader got "a in filter needs a value" or "no warehouse called first_one".
* "Between 27 May and 17 August" compared a period with itself ("no change for any warehouse").
* "In each snapshot" grouped by a date instead of a series, and was refused.
* "On the latest snapshot" was read as the last day, a partial load.
* "Which channel grew the most" kept the month-by-month grain of the trend before it.
* "Lost the most" was sorted lowest first.
* A question back written beside "kind" was refused twice, and the reader got "could not work out how".

Each is checked here on the synthetic warehouses, with a scripted AI. Invented data only.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.ir import Plan
from core2.plan.planner import Turn, check, fiscal_years_text, parse, plan_question, question_tail, settle
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 10, 10)
Q1 = {"kind": "between", "start": "2026-01-01", "end": "2026-03-31"}
MAY = {"kind": "between", "start": "2026-05-01", "end": "2026-05-31"}


@pytest.fixture(scope="module")
def retail():
    _, model = learn(domains.build("retail"), "descriptive")
    region = next(slug for slug in model.attributes if slug.startswith("region."))
    return model, region


@pytest.fixture(scope="module")
def hr():
    _, model = learn(domains.build("hr"), "descriptive")
    return model


def _plan(**kw) -> Plan:
    return Plan.model_validate({"kind": "query", **kw})


class Scripted:
    """The planner's AI, answering from a list; what it was asked is kept."""

    def __init__(self, *replies):
        self.replies, self.asked = [json.dumps(r) for r in replies], []

    def __call__(self, stable, tail):
        self.asked.append(tail)
        return self.replies.pop(0)


# ── fiscal years ─────────────────────────────────────────────────────────────


def test_the_ai_is_told_the_dates_of_each_fiscal_year(retail):
    model, _ = retail
    text = fiscal_years_text(model, TODAY)
    # The retail warehouse's year starts in April and is named by the year it ends in.
    assert "starts on 1 April" in text and "named by the year it ends in" in text
    assert "This fiscal year: FY2027 = 2026-04-01 to 2027-03-31" in text
    assert "last fiscal year: FY2026 = 2025-04-01 to 2026-03-31" in text
    tail = question_tail("Net sales in fiscal year 2026", today=TODAY, history=[], matches=[], model=model)
    assert "FY2026 = 2025-04-01 to 2026-03-31" in tail


def test_a_calendar_year_needs_no_fiscal_line(hr):
    assert fiscal_years_text(hr, TODAY) == ""


def test_before_the_fiscal_year_starts_this_fiscal_year_began_the_year_before(retail):
    model, _ = retail
    assert "This fiscal year: FY2026 = 2025-04-01 to 2026-03-31" in fiscal_years_text(model, dt.date(2026, 2, 1))


# ── plans that cannot be meant go back to the AI ─────────────────────────────


def test_a_filter_with_no_value_is_sent_back(retail):
    model, region = retail
    problems, _ = check(_plan(measures=["net_amount"], group_by=[region],
                              filters=[{"field": region, "op": "in", "values": []}]), model, TODAY, "how did they do?")
    assert problems and "has no value" in problems[0] and "keep the previous plan's group_by" in problems[0]


def test_a_stand_in_for_a_member_is_sent_back_but_a_name_the_reader_wrote_is_not(retail):
    model, region = retail
    for stand_in in ("first_one", "warehouse_with_max_increase", "first_supplier",
                     "<top_warehouse_from_previous_answer>", "{the_first_one}"):
        problems, _ = check(_plan(measures=["net_amount"], filters=[{"field": region, "op": "eq",
                                                                      "values": [stand_in]}]),
                            model, TODAY, "break the first one down")
        assert problems and f'"{stand_in}"' in problems[0] and "clarify" in problems[0], stand_in
    # A code with underscores that points at nothing, or one the reader typed, is a name.
    for value, question in (("in_transit", "orders in transit"), ("first_one", "sales for first_one")):
        problems, _ = check(_plan(measures=["net_amount"], filters=[{"field": region, "op": "eq",
                                                                      "values": [value]}]), model, TODAY, question)
        assert not any("stands for" in p for p in problems), (value, problems)


def test_a_period_compared_with_itself_is_sent_back(retail):
    model, _ = retail
    same = {"kind": "between", "start": "2026-05-27", "end": "2026-08-17"}
    problems, _ = check(_plan(intent="compare", measures=["net_amount"],
                              time={"window": same, "compare": {"kind": "window", "window": same}}), model, TODAY)
    assert problems and "the later one" in problems[0]
    fine, _ = check(_plan(intent="compare", measures=["net_amount"],
                          time={"window": {"kind": "between", "start": "2026-08-17", "end": "2026-08-17"},
                                "compare": {"kind": "window", "window": {"kind": "between", "start": "2026-05-27",
                                                                          "end": "2026-05-27"}}}), model, TODAY)
    assert not fine


def test_a_date_as_a_grouping_is_sent_back_as_a_series(retail):
    model, _ = retail
    problems, _ = check(_plan(measures=["number_of_orders"], group_by=["order_date"]), model, TODAY)
    assert problems and "order_date is a date" in problems[0] and "intent trend" in problems[0]


def test_the_repair_round_is_told_why_and_its_plan_is_used(retail):
    model, region = retail
    ai = Scripted({"kind": "query", "intent": "breakdown", "measures": ["net_amount"], "group_by": ["category.name"],
                   "filters": [{"field": region, "op": "eq", "values": ["first_one"]}]},
                  {"kind": "clarify", "clarify": {"about": "member", "question": "Which region do you mean?"}})
    outcome = plan_question(model, "Break the first one down by category", ai, today=TODAY)
    assert outcome.plan.kind == "clarify" and outcome.repaired
    assert '"first_one" is not a value of' in ai.asked[1]


def test_a_question_back_written_beside_kind_is_read(retail):
    plan = parse('```json {"kind": "clarify", "about": "follow_up", "question": "What does \'those\' refer to?", '
                 '"options": ["The rows", "The groups"]} ```')
    assert plan.kind == "clarify" and plan.clarify.question == "What does 'those' refer to?"
    assert plan.clarify.about == "follow_up" and plan.clarify.options == ["The rows", "The groups"]
    odd = parse('{"kind": "clarify", "about": "people", "question": "Which one?"}')
    assert odd.clarify.about == "other"


# ── what the question's words settle ─────────────────────────────────────────


def test_a_follow_up_that_names_no_period_keeps_the_one_above(retail):
    model, region = retail
    previous = _plan(intent="breakdown", measures=["number_of_returns"], time={"window": Q1})
    dropped = _plan(intent="breakdown", measures=["number_of_returns"], group_by=[region], follow_up="refine")
    kept = settle(dropped, model, "Show the returns by region", previous)
    assert kept.time.window.model_dump(mode="json", exclude_defaults=True) == Q1


def test_a_why_about_the_answer_keeps_both_of_its_periods(retail):
    model, _ = retail
    compare = {"kind": "same_period_last_year"}
    previous = _plan(intent="value", measures=["net_amount"], time={"window": MAY, "compare": compare})
    bare = _plan(intent="drivers", measures=["net_amount"], follow_up="refine")
    for question in ("Why did it grow?", "Pourquoi cette différence ?"):
        kept = settle(bare, model, question, previous)
        assert kept.time.window == previous.time.window and kept.time.compare == previous.time.compare, question
    # A period the AI chose stands: "why did it grow?" after "which month grew the most" is that month.
    june = {"kind": "between", "start": "2026-06-01", "end": "2026-06-30"}
    chosen = _plan(intent="drivers", measures=["net_amount"], follow_up="refine", time={"window": june})
    kept = settle(chosen, model, "Why did it grow?", previous)
    assert kept.time.window == chosen.time.window and kept.time.compare == previous.time.compare


def test_that_month_is_the_month_above_and_a_named_period_is_the_readers(retail):
    model, _ = retail
    previous = _plan(intent="value", measures=["net_amount"], time={"window": MAY, "date": "ship_date"})
    now = _plan(intent="rank", measures=["net_amount"], follow_up="refine",
                time={"window": {"kind": "this", "unit": "month"}})
    kept = settle(now, model, "Which 10 customers bought the most that month?", previous)
    assert kept.time.window == previous.time.window and kept.time.date == "ship_date"
    named = _plan(intent="value", measures=["net_amount"], follow_up="refine",
                  time={"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}})
    assert settle(named, model, "And in 2025?", previous).time.window == named.time.window


def test_a_new_question_carries_nothing(retail):
    model, _ = retail
    previous = _plan(intent="value", measures=["net_amount"], time={"window": Q1})
    new = _plan(intent="value", measures=["number_of_orders"], follow_up="new")
    assert settle(new, model, "How many orders do we have?", previous) is new


def test_a_route_the_reader_picked_is_not_asked_again(retail):
    model, region = retail
    previous = _plan(intent="breakdown", measures=["net_amount"], group_by=[region], via={region: "billing"})
    now = _plan(intent="rank", measures=["net_amount"], group_by=[region], follow_up="refine")
    assert settle(now, model, "Which region sold the most?", previous).via == {region: "billing"}


def test_the_latest_snapshot_of_a_stock_is_no_window(hr):
    last_day = {"kind": "last", "unit": "day", "n": 1}
    stock = _plan(intent="value", measures=["headcount"], time={"window": last_day})
    assert settle(stock, hr, "What is our headcount on the latest snapshot?", None).time.window.kind == "all"
    flow = _plan(intent="value", measures=["gross_pay"], time={"window": last_day})
    assert settle(flow, hr, "Gross pay over the last day", None).time.window.kind == "last"


def test_a_comparison_asked_no_series_drops_a_grain_kept_from_a_trend(retail):
    model, region = retail
    grown = _plan(intent="compare", measures=["number_of_orders"], group_by=[region],
                  time={"grain": "month", "window": {"kind": "this", "unit": "year"},
                        "compare": {"kind": "same_period_last_year"}}, sort=[{"by": "change"}])
    assert settle(grown, model, "Which region grew the most compared with the same period last year?",
                  None).time.grain is None
    monthly = settle(grown, model, "Compare monthly orders by region with last year", None)
    assert monthly.time.grain == "month"
    assert settle(grown, model, "Which month grew the most against last year?", None).time.grain == "month"
    # A ranking of periods is its grain: "which 3 months sold the most".
    months = _plan(intent="rank", measures=["net_amount"], sort=[{"by": "net_amount"}], limit=3,
                   time={"grain": "month", "window": Q1})
    assert settle(months, model, "Which 3 were the best?", None).time.grain == "month"


def test_the_most_is_highest_first_and_the_least_lowest_first(retail):
    model, region = retail
    asc = _plan(intent="rank", measures=["net_amount"], group_by=[region], sort=[{"by": "net_amount",
                                                                                  "desc": False}], limit=3)
    assert settle(asc, model, "Which 3 regions lost the most net sales to returns?", None).sort[0].desc is True
    desc = asc.model_copy(update={"sort": [asc.sort[0].model_copy(update={"desc": True})]})
    assert settle(desc, model, "Which 3 regions sold the least?", None).sort[0].desc is False
    # "at least" says nothing of the order, and a change keeps the direction the AI chose.
    assert settle(asc, model, "Regions with at least 5 orders", None) is asc
    change = _plan(intent="compare", measures=["net_amount"], group_by=[region], sort=[{"by": "change",
                                                                                       "desc": False}],
                   time={"window": MAY, "compare": {"kind": "previous_period"}})
    assert settle(change, model, "Which region fell the most?", None) is change


def test_the_planner_keeps_the_period_above_on_a_follow_up(retail):
    """End to end through the planner: the AI's plan for "now by category" left Q1 out; the answer keeps it."""
    model, region = retail
    previous = Turn("Net sales by region in Q1", _plan(intent="breakdown", measures=["net_amount"],
                                                       group_by=[region], time={"window": Q1}))
    ai = Scripted({"kind": "query", "intent": "breakdown", "measures": ["net_amount"], "group_by": ["category.name"],
                   "follow_up": "refine"})
    outcome = plan_question(model, "Now by category", ai, today=TODAY, history=[previous], reading="refine")
    assert outcome.plan.time.window.model_dump(mode="json", exclude_defaults=True) == Q1
    assert outcome.plan.group_by == ["category.name"]


def test_a_follow_up_naming_its_own_day_in_french_keeps_it(retail):
    """"Et au 27 mai ?" names a day: the answer is for 27 May, never the 17 August of the answer above."""
    model, region = retail
    aug17 = {"kind": "between", "start": "2026-08-17", "end": "2026-08-17"}
    may27 = {"kind": "between", "start": "2026-05-27", "end": "2026-05-27"}
    previous = _plan(intent="breakdown", measures=["net_amount"], group_by=[region], time={"window": aug17})
    asked = _plan(intent="breakdown", measures=["net_amount"], group_by=[region], follow_up="refine",
                  time={"window": may27})
    for question in ("Et au 27 mai ?", "And on 27/05?", "Y en mayo de 2026?"):
        assert settle(asked, model, question, previous).time.window == asked.time.window, question


# ── what a plan plainly means, filled in before it is checked ────────────────


def test_a_ranking_sorted_by_a_measure_it_leaves_out_shows_it(retail):
    from core2.plan.planner import fill_in

    model, region = retail
    plan = _plan(intent="rank", measures=["net_amount"], group_by=[region], sort=[{"by": "number_of_orders"}],
                 limit=10)
    done = fill_in(plan, model, None, True)
    assert done.measures == ["net_amount", "number_of_orders"]
    assert not check(done, model, TODAY)[0]
    by_change = _plan(intent="compare", measures=["net_amount"], group_by=[region], sort=[{"by": "change"}],
                      time={"window": MAY, "compare": {"kind": "previous_period"}})
    assert fill_in(by_change, model, None, True) is by_change


def test_a_stock_compared_between_two_dates_written_as_one_span_is_the_later_date_against_the_earlier(hr):
    from core2.plan.planner import fill_in

    span = {"kind": "between", "start": "2026-05-27", "end": "2026-08-17"}
    day = {"kind": "between", "start": "2026-05-27", "end": "2026-05-27"}
    for compared in (span, day):           # against itself, or against a day inside it
        plan = _plan(intent="compare", measures=["headcount"],
                     time={"window": span, "compare": {"kind": "window", "window": compared}})
        done = fill_in(plan, hr, None, True)
        assert done.time.window.model_dump(mode="json", exclude_defaults=True) == {
            "kind": "between", "start": "2026-08-17", "end": "2026-08-17"}, compared
        assert done.time.compare.window.model_dump(mode="json", exclude_defaults=True) == day, compared
        assert not check(done, hr, TODAY)[0]
    # A flow over the same span is sent back with the dates to use, never guessed.
    flow = _plan(intent="compare", measures=["gross_pay"],
                 time={"window": span, "compare": {"kind": "window", "window": day}})
    assert fill_in(flow, hr, None, True) is flow
    problems, _ = check(flow, hr, TODAY)
    assert problems and "overlaps" in problems[0] and '"start": "2026-08-17", "end": "2026-08-17"' in problems[0]


def test_a_plan_that_says_nothing_of_the_answer_above_follows_it_on_the_same_measure(retail):
    from core2.plan.planner import fill_in

    model, region = retail
    previous = _plan(intent="value", measures=["number_of_returns"], time={"window": Q1})
    plain = _plan(intent="breakdown", measures=["number_of_returns"], group_by=[region])
    assert fill_in(plain, model, previous, False).follow_up == "refine"
    assert fill_in(plain, model, previous, True).follow_up == "new"       # the AI said so: it stands
    other = _plan(intent="value", measures=["net_amount"])
    assert fill_in(other, model, previous, False).follow_up == "new"


def test_a_follow_up_the_ai_did_not_mark_keeps_the_period_end_to_end(retail):
    model, region = retail
    previous = Turn("How many returns in Q1 2026?", _plan(intent="value", measures=["number_of_returns"],
                                                          time={"window": Q1}))
    ai = Scripted({"kind": "query", "intent": "breakdown", "measures": ["number_of_returns"], "group_by": [region]})
    outcome = plan_question(model, "Show the returns by region", ai, today=TODAY, history=[previous])
    assert outcome.plan.follow_up == "refine"
    assert outcome.plan.time.window.model_dump(mode="json", exclude_defaults=True) == Q1


def test_a_measure_on_screen_split_another_way_with_no_period_follows_the_answer(retail):
    """"Show the returns by warehouse" after the returns of Q1 was read as a question of its own, and answered for
    all time. A measure of the answer on screen, split another way, naming no period, follows it; one that names
    its own period, or a measure the answer did not have, is a new question."""
    from core2.plan.followup import read_turn

    model, region = retail
    q1 = _plan(intent="value", measures=["number_of_orders", "number_of_returns"], time={"window": Q1})
    assert read_turn("Show the returns by region", previous=q1, model=model).kind == "refine"
    assert read_turn("Show the returns by region in 2025", previous=q1, model=model).kind == "new"
    assert read_turn("Show the net sales by region", previous=q1, model=model).kind != "refine"
    all_time = _plan(intent="value", measures=["number_of_returns"])
    assert read_turn("Show the returns by region", previous=all_time, model=model).kind == "new"


def test_plain_quarters_are_the_calendars_where_the_fiscal_year_is_not(retail):
    """The workspace's fiscal year starting in April made "Q2 2026" read as fiscal Q2: the AI is told the fiscal
    dates are only for a period the question calls fiscal."""
    model, _ = retail
    text = fiscal_years_text(model, TODAY)
    assert text.startswith("FISCAL YEARS (only for a period the question calls fiscal")
    assert "Q2 2026" in text and "is the calendar's" in text


def test_slips_of_form_in_a_duration_are_forgiven():
    plan = parse(json.dumps({"kind": "query", "intent": "value", "measures": ["number_of_orders"],
                             "durations": [{"name": "Days late", "start": "requested_date", "end": "delivered_date",
                                            "agg": "count"}],
                             "filters": [{"field": "Days late", "op": "gt", "values": [0], "measure": False}]}))
    assert plan.durations[0].agg == "avg" and plan.durations[0].measure is False
    assert plan.filters[0].field == "Days late" and plan.filters[0].values == [0]


def test_that_month_points_at_the_answer_above(retail):
    from core2.plan.followup import read_turn

    model, _ = retail
    previous = _plan(intent="value", measures=["net_amount"], time={"window": MAY})
    assert read_turn("Which 10 customers have the most due that month?", previous=previous, model=model).kind \
        == "refine"


def test_a_ranking_of_members_drops_a_grain_kept_from_a_trend(retail):
    model, region = retail
    ranked = _plan(intent="rank", measures=["number_of_orders"], group_by=[region], sort=[{"by": "number_of_orders"}],
                   time={"grain": "month", "window": Q1})
    assert settle(ranked, model, "Which region orders the most?", None).time.grain is None
    assert settle(ranked, model, "Which region orders the most each month?", None).time.grain == "month"


def test_a_share_of_the_total_written_as_a_share_of_members_is_read_as_a_share(retail):
    """"Which 5 items brought the most, and what share of the total is that?": a real model wrote a share of members
    with no condition for them to meet, twice, and the reader was told it could not be worked out."""
    model, region = retail
    plan = parse(json.dumps({"kind": "query", "intent": "rank", "measures": ["net_amount"], "group_by": [region],
                             "sort": [{"by": "net_amount", "desc": True}], "limit": 5,
                             "per": {"of": None, "summary": "share"}}))
    assert plan.intent == "share" and plan.per is None and not check(plan, model, TODAY)[0]
    counted = parse(json.dumps({"kind": "query", "intent": "value", "measures": ["number_of_orders"],
                                "per": {"of": "customer", "summary": "share", "op": "gte", "values": [3]}}))
    assert counted.per is not None and counted.per.summary == "share"      # a real share of members stands
