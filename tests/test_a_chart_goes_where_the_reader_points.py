"""A chart is drawn the way the reader picks, opens where it is, and a click on it goes somewhere useful.

From the reader's test of the new core:

* A pie had no donut, and a chart could only be switched between itself and its table: the
  answer card now offers every shape the answer can honestly be drawn as (a pie and a donut
  only for parts of a whole: one measure over every member, a handful of them, none below
  zero), and "as a donut" is a shape a question can ask for.
* A click on a bar sent "Break this down for X" and left the rest to the planner. A new-core
  chart now carries what a click opens: for a member, its month-by-month trend, the member by
  the next grouping and why it changed; for a period, that period by a grouping and why the
  measure changed in it; on a line per member, both for the member of the line clicked.
* "No data for Aug 2022 to Nov 2022 ...: shown as a gap" was untrue: the months with no row
  were left out of the chart, so the line ran straight across three years. They are now
  slots with no value, and the line breaks.
* The axis wrote the year under a label only when the category before it was another year;
  on a crowded axis that category is often not shown, so "Jun, Dec, Nov" read Dec 2022 and
  Nov 2025. The axis now chooses which labels it shows, and the year goes under the first
  label SHOWN in each year.
* Expand opened the chart beside the chat. It now grows the answer where it is.

The payloads come from the real service on the invented retail warehouse; the page's own
JavaScript runs in duktape.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn
from tests.test_chart_annotation_language import _build

TODAY = dt.date(2026, 6, 15)
H1 = {"kind": "between", "start": "2026-01-01", "end": "2026-06-30"}
SHARE = {"intent": "share", "measures": ["net_amount"], "group_by": ["region.name"], "time": {"window": H1}}
BY_STORE = {"intent": "breakdown", "measures": ["net_amount"], "group_by": ["store"], "time": {"window": H1}}
TOP_10 = {"intent": "rank", "measures": ["net_amount"], "group_by": ["customer"], "limit": 10, "time": {"window": H1}}
MONTHLY = {"intent": "trend", "measures": ["net_amount"], "time": {"grain": "month", "window": H1}}
PER_REGION = {**MONTHLY, "group_by": ["region.name"]}


@pytest.fixture(scope="module")
def retail():
    built, model = learn(domains.build("retail"), "descriptive")
    return built.con, model


@pytest.fixture(scope="module")
def no_march(retail):
    """The same warehouse, learned whole, then March 2026 never loaded."""
    built, model = learn(domains.build("retail"), "descriptive")
    built.con.execute("DELETE FROM main.order_lines WHERE order_date_key BETWEEN 20260301 AND 20260331")
    return built.con, model


def _answer(warehouse, plan: dict) -> dict:
    con, model = warehouse
    answer = json.dumps({"kind": "query", **plan})
    services = Services(model=model, warehouse=DuckDBWarehouse(con), complete=lambda s, t: answer,
                        index=MemberIndex(), today=TODAY)
    return answer_question("q", services, Session())


# ── the shapes offered ──────────────────────────────────────────────────────


def test_parts_of_a_whole_are_offered_as_a_pie_or_a_donut(retail):
    chart = _answer(retail, SHARE)["chart"]
    assert chart["chart_type"] == "pie"
    assert chart["renderable_types"] == ["pie", "donut", "bar"]


def test_shares_of_a_handful_drawn_as_bars_can_be_a_pie_or_a_donut(retail):
    chart = _answer(retail, {**SHARE, "group_by": ["category.name"]})["chart"]     # eight categories and Unknown
    assert chart["share_key"] and chart["chart_type"] == "bar"
    assert chart["renderable_types"] == ["bar", "pie", "donut"]


def test_the_top_few_of_many_are_never_a_pie(retail):
    """Their slices would not add up to the whole they are shares of."""
    chart = _answer(retail, {**SHARE, "group_by": ["category.name"], "limit": 5})["chart"]
    assert chart["chart_type"] == "bar" and chart["renderable_types"] == ["bar"]


def test_too_many_members_or_a_top_n_are_offered_as_bars_only(retail):
    assert _answer(retail, BY_STORE)["chart"]["renderable_types"] == ["bar"]       # twelve stores
    assert _answer(retail, {**SHARE, "group_by": ["store"]})["chart"]["renderable_types"] == ["bar"]
    assert _answer(retail, TOP_10)["chart"]["renderable_types"] == ["bar"]          # ten of many customers


def test_a_whys_changes_by_member_are_offered_as_bars_only(retail):
    chart = _answer(retail, {"intent": "drivers", "measures": ["net_amount"],
                             "time": {"window": {"kind": "between", "start": "2026-04-01", "end": "2026-04-30"}}})["chart"]
    assert chart["chart_type"] == "bar" and chart["renderable_types"] == ["bar"]


def test_a_question_can_ask_for_a_donut(retail):
    chart = _answer(retail, {**SHARE, "chart": "donut"})["chart"]
    assert chart["chart_type"] == "donut"


def test_a_donut_of_a_series_in_time_is_refused_and_said(retail):
    payload = _answer(retail, {**MONTHLY, "chart": "donut"})
    assert payload["chart"]["chart_type"] == "line"
    assert "A donut chart does not fit this answer" in json.dumps(payload)


# ── what a click opens ──────────────────────────────────────────────────────


def _drill(warehouse, plan: dict) -> dict:
    return _answer(warehouse, plan)["chart"]["drill"]


def test_a_member_opens_its_trend_its_next_grouping_and_why_it_changed(retail):
    drill = _drill(retail, BY_STORE)
    assert drill["on"] == "member" and drill["series"] == ""
    assert [i["question"] for i in drill["items"]] == [
        "Net amount by month for {member} from Jan 2026 to Jun 2026",
        "Net amount for {member} by product from Jan 2026 to Jun 2026",
        "Why did net amount change for {member} from Jan 2026 to Jun 2026?",
    ]


def test_a_member_is_never_broken_down_by_its_own_grouping(retail):
    for plan in (BY_STORE, {**BY_STORE, "group_by": ["store.name"]}, SHARE, TOP_10):
        drill = _drill(retail, plan)
        words = {"store": "by store", "store.name": "by store", "region.name": "by region", "customer": "by customer"}
        assert all(words[plan["group_by"][0]] not in i["question"] for i in drill["items"])


def test_a_period_opens_its_breakdown_and_why_the_measure_changed_in_it(retail):
    drill = _drill(retail, MONTHLY)
    assert drill == {"on": "period", "series": "", "items": [
        {"label": "{period} by store", "question": "Net amount by store in {period}"},
        {"label": "Why changed in {period}", "question": "Why did net amount change in {period}?"}]}


def test_a_line_per_member_opens_for_the_member_of_the_line_clicked(retail):
    drill = _drill(retail, PER_REGION)
    assert drill["on"] == "period" and drill["series"] == "member"
    assert all("{member}" in i["question"] and "{period}" in i["question"] for i in drill["items"])


def test_a_click_keeps_the_answers_conditions(retail):
    plan = {**BY_STORE, "filters": [{"field": "customer.segment", "op": "eq", "values": ["Retail"]}]}
    for item in _drill(retail, plan)["items"]:
        assert "Retail" in item["question"], item


def test_a_forecast_opens_nothing(retail):
    forecast = _answer(retail, {**MONTHLY, "intent": "forecast"})
    assert not (forecast.get("chart") or {}).get("drill")


# ── the page fills it in ────────────────────────────────────────────────────


def _choices(payload: dict, params: dict) -> list[dict]:
    from tests.chat_js import run

    return run(f"JSON.stringify(_drillChoices({json.dumps(payload)}, {json.dumps(params)}));",
               functions=["function _drillChoices(payload, params)"],
               preamble="var window = this; window.QBCharts = {periodLabel: function (raw) {"
                        " return raw === '2026-03-01' ? 'Mar 2026' : raw; }};")


def test_the_page_fills_in_the_member_clicked(retail):
    chart = _answer(retail, BY_STORE)["chart"]
    member = chart["rows"][0][chart["x_key"]]
    choices = _choices(chart, {"name": member, "value": 1.0})
    assert choices[0] == {"label": f"{member} by month",
                          "question": f"Net amount by month for {member} from Jan 2026 to Jun 2026"}


def test_the_page_names_the_period_clicked_as_the_chart_does(retail):
    chart = _answer(retail, MONTHLY)["chart"]
    choices = _choices(chart, {"name": "2026-03-01", "value": 1.0})
    assert [c["question"] for c in choices] == ["Net amount by store in Mar 2026",
                                                "Why did net amount change in Mar 2026?"]


def test_the_page_names_the_member_of_the_line_clicked(retail):
    chart = _answer(retail, PER_REGION)["chart"]
    choices = _choices(chart, {"name": "2026-03-01", "seriesName": "North", "value": 1.0})
    assert choices[1]["question"] == "Why did net amount change for North in Mar 2026?"
    assert _choices(chart, {"name": "2026-03-01", "seriesName": "Forecast", "value": 1.0}) == []


def test_a_gap_or_a_bucket_that_is_no_member_opens_nothing(retail):
    chart = _answer(retail, BY_STORE)["chart"]
    member = chart["rows"][0][chart["x_key"]]
    assert _choices(chart, {"name": member, "value": None}) == []
    assert _choices(chart, {"name": "Other (3)", "value": 12.0}) == []


# ── gaps ────────────────────────────────────────────────────────────────────


def test_a_month_with_no_row_is_a_gap_in_the_line_not_left_out(no_march):
    payload = _answer(no_march, MONTHLY)
    assert "shown as a gap, not as zero" in json.dumps(payload["coverage_caveats"])
    rows = payload["chart"]["rows"]
    assert [r["period"][:10] for r in rows] == [f"2026-0{m}-01" for m in range(1, 7)]
    assert rows[2]["net_amount"] is None and all(r["net_amount"] is not None for r in rows if r is not rows[2])


def test_gap_slots_keep_the_series_in_time_order():
    from core2.answer.builder import _with_gaps

    rows = [{"p": "2025-12-01", "v": 1.0}, {"p": "2026-01-01", "v": 2.0}, {"p": "2026-03-01", "v": 3.0}]
    expected = [dt.date(2026, 1, 1), dt.date(2026, 2, 1), dt.date(2026, 3, 1)]
    assert [(r["p"], r["v"]) for r in _with_gaps(rows, "p", ["v"], expected)] == [
        ("2025-12-01", 1.0), ("2026-01-01", 2.0), ("2026-02-01", None), ("2026-03-01", 3.0)]


def test_a_series_with_every_month_is_drawn_as_it_is(retail):
    rows = _answer(retail, MONTHLY)["chart"]["rows"]
    assert len(rows) == 6 and all(r["net_amount"] is not None for r in rows)


# ── the axis ────────────────────────────────────────────────────────────────


def _months(first: tuple[int, int], count: int) -> list[str]:
    year, month = first
    out = []
    for _ in range(count):
        out.append(f"{year}-{month:02d}-01")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def _axis(periods: list[str]) -> tuple[int, list[str]]:
    """The axis's step, and its labels as ECharts asks for them: only the shown ones, left to right."""
    payload = {"chart_type": "line", "x_key": "period", "y_keys": ["v"], "x_style": "month_year_long",
               "chart_spec": {"x": {"column": "period", "role": "temporal"}},
               "column_roles": {"period": {"column": "period", "role": "temporal"}},
               "rows": [{"period": p, "v": 1.0} for p in periods]}
    expression = ("var a = opt.xAxis.axisLabel, every = a.interval + 1, out = [];"
                  f"var periods = {json.dumps(periods)};"
                  "for (var i = 0; i < periods.length; i += every) out.push(a.formatter(periods[i], i));"
                  "JSON.stringify([a.interval, out])")
    return json.loads(_build("portal_chat.html", "en", payload, expression))


def test_the_year_goes_under_the_first_label_shown_in_each_year():
    step, shown = _axis(_months((2023, 11), 36))          # three years: every fifth month is shown
    assert step == 4
    assert shown[:4] == ["Nov\n2023", "Apr\n2024", "Sep", "Feb\n2025"]
    years = [label.split("\n")[1] for label in shown if "\n" in label]
    assert years == ["2023", "2024", "2025", "2026"], "every year on the axis is named once"


def test_a_short_axis_shows_every_label_and_the_year_where_it_changes():
    step, shown = _axis(["2022-06-01", "2022-12-01", "2025-10-01", "2025-11-01", "2025-12-01", "2026-01-01"])
    assert step == 0
    assert shown == ["Jun\n2022", "Dec", "Oct\n2025", "Nov", "Dec", "Jan\n2026"]


# ── the answer card ─────────────────────────────────────────────────────────


def _card(chart: dict) -> str:
    from tests.test_the_answer_card import card

    return card({"engine": "core2", "question": "q", "answer": {"headline": "Net amount by region."},
                 "chart": chart, "data": {"headers": ["region_name"], "rows": [{"region_name": "North"}]}})


def test_the_card_offers_the_shapes_the_answer_fits_and_marks_the_one_drawn(retail):
    from tests.test_the_answer_card import Elements

    markup = _card(_answer(retail, SHARE)["chart"])
    group = Elements(markup).first("answer-types")
    assert group is not None
    buttons = [e for e in Elements(markup).found if "data-type" in e["attrs"]]
    assert [(b["attrs"]["data-type"], b["attrs"]["aria-pressed"]) for b in buttons] == [
        ("bar", "false"), ("pie", "true"), ("donut", "false")]


def test_a_chart_of_one_shape_has_no_shape_buttons(retail):
    from tests.test_the_answer_card import Elements

    markup = _card(_answer(retail, BY_STORE)["chart"])
    assert Elements(markup).first("answer-types") is None


def test_expand_grows_the_answer_in_place_and_opens_nothing_beside_it(retail):
    from tests.test_the_answer_card import Elements

    markup = _card(_answer(retail, SHARE)["chart"])
    expand = next(e for e in Elements(markup).found if "data-expand" in e["attrs"])
    assert expand["attrs"]["aria-pressed"] == "false"
    assert "data-open-artifact" not in markup
