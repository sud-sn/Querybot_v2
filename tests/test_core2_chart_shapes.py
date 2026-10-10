"""A new-core chart takes the shape its answer needs, and offers only shapes that fit.

Three answers were drawn in a shape that hid what they said:

- "Net amount by store, April against March" drew two bars per store, side by
  side: eight bars to read for four changes. It is now a dumbbell -- one line
  per store from its March value to its April value, March a muted dot, April
  the series colour, and the change written past the further dot.
- "Net amount and cost by store" drew only the first measure: the second was
  in the table and nowhere on the chart. Every measure asked for is drawn, and
  measures in different units get a panel each, never a second axis.
- A breakdown by store offered Line and Area beside Bar, as if stores ran in
  time. A category is never a line; only a time axis offers one.

The payloads come from the real service on the invented retail warehouse; the
drawing runs through the real renderer, as the chat page loads it.
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
APRIL = {"kind": "between", "start": "2026-04-01", "end": "2026-04-30"}
H1 = {"kind": "between", "start": "2026-01-01", "end": "2026-06-30"}


@pytest.fixture(scope="module")
def retail():
    built, model = learn(domains.build("retail"), "descriptive")
    return built.con, model


def _chart(retail, plan: dict) -> dict:
    con, model = retail
    answers = [json.dumps({"kind": "query", **plan})]
    services = Services(model=model, warehouse=DuckDBWarehouse(con), complete=lambda s, t: answers.pop(0),
                        index=MemberIndex(), today=TODAY)
    return answer_question("q", services, Session())["chart"]


# ── the same members at two times ───────────────────────────────────────────


COMPARE = {"intent": "compare", "measures": ["net_amount"], "group_by": ["store"],
           "time": {"window": APRIL, "compare": {"kind": "previous_period"}}}


def test_a_comparison_by_member_is_a_dumbbell_naming_both_periods(retail):
    chart = _chart(retail, COMPARE)
    assert chart["chart_type"] == "dumbbell"
    cmp = chart["compare"]
    assert (cmp["prior_label"], cmp["current_label"]) == ("March 2026", "April 2026")
    assert {cmp["prior"], cmp["current"]} == set(chart["y_keys"])
    for row in chart["rows"]:
        assert isinstance(row[cmp["prior"]], float) and isinstance(row[cmp["current"]], float)


def test_a_comparison_can_be_switched_to_bars_and_nothing_else(retail):
    chart = _chart(retail, COMPARE)
    assert chart["renderable_types"] == ["dumbbell", "bar"]


# ── several measures ────────────────────────────────────────────────────────


def test_every_measure_asked_for_is_drawn(retail):
    chart = _chart(retail, {"intent": "breakdown", "measures": ["net_amount", "cost_amount"],
                            "group_by": ["store"], "time": {"window": H1}})
    assert set(chart["y_keys"]) == {"net_amount", "cost_amount"}
    assert chart["title"] == "Net amount and cost amount by store"
    assert chart["facets"] == [], "two amounts of money share one axis"


def test_measures_in_two_units_get_a_panel_each(retail):
    chart = _chart(retail, {"intent": "breakdown", "measures": ["net_amount", "margin_percent"],
                            "group_by": ["region.name"], "time": {"window": H1}})
    assert len(chart["facets"]) == 2
    assert sorted(len(group) for group in chart["facets"]) == [1, 1]
    assert {y for group in chart["facets"] for y in group} == set(chart["y_keys"])


# ── the shapes offered ──────────────────────────────────────────────────────


def test_a_breakdown_by_category_is_never_offered_as_a_line(retail):
    chart = _chart(retail, {"intent": "breakdown", "measures": ["net_amount"], "group_by": ["store"],
                            "time": {"window": H1}})
    assert chart["chart_type"] == "bar"
    assert "line" not in chart["renderable_types"] and "area" not in chart["renderable_types"]


def test_a_series_in_time_is_offered_as_a_line_an_area_or_bars(retail):
    chart = _chart(retail, {"intent": "trend", "measures": ["net_amount"],
                            "time": {"grain": "month", "window": H1}})
    assert chart["renderable_types"] == ["line", "area", "bar"]


# ── what the reader sees ────────────────────────────────────────────────────


DUMBBELL = {"chart_type": "dumbbell", "x_key": "store", "y_keys": ["sales_prior", "sales"],
            "rows": [{"store": "Airport", "sales_prior": 100.0, "sales": 150.0},
                     {"store": "Harbour", "sales_prior": 200.0, "sales": 120.0}],
            "compare": {"prior": "sales_prior", "current": "sales",
                        "prior_label": "March 2026", "current_label": "April 2026"},
            "column_formats": {"sales_prior": "number", "sales": "number"}}


def _drawn(expression: str):
    return json.loads(_build("portal_chat.html", "en", DUMBBELL, expression))


def test_the_dumbbell_names_both_periods_in_its_legend():
    assert _drawn("JSON.stringify(opt.legend.data)") == ["March 2026", "April 2026"]


def test_the_dumbbell_draws_a_line_per_member_and_a_dot_per_period():
    kinds = _drawn("JSON.stringify(opt.series.map(function (s) { return [s.type, s.name || ''] }))")
    lines = [k for k in kinds if k[0] == "line"]
    dots = [k[1] for k in kinds if k[0] == "scatter" and k[1]]
    assert len(lines) == 2, "one line per store"
    assert dots == ["March 2026", "April 2026"]


def test_the_earlier_period_is_muted_and_the_current_one_carries_the_colour():
    prior, current = _drawn(
        "var d = opt.series.filter(function (s) { return s.type === 'scatter' && s.name });"
        "JSON.stringify([d[0].itemStyle.color, d[1].itemStyle.color])")
    palette = _drawn("JSON.stringify(opt.color || [])")
    assert prior != current
    assert prior not in palette, "the earlier period is ink, not a series colour"


def test_the_change_is_written_with_its_sign():
    said = _drawn(
        "var s = opt.series[opt.series.length - 1];"
        "JSON.stringify([0, 1].map(function (i) { return s.label.formatter({dataIndex: i}) }))")
    assert said[0].startswith("+") and "50" in said[0]
    assert said[1].startswith("−") and "80" in said[1]


def test_the_chat_page_offers_the_dumbbell_it_was_sent():
    offered = _drawn("JSON.stringify(QBCharts.offeredTypes({renderable_types: ['dumbbell', 'bar']},"
                     " ['dumbbell', 'bar', 'line', 'area', 'pie', 'donut', 'scatter']))")
    assert offered == ["dumbbell", "bar"]


def test_a_comparison_too_long_to_draw_keeps_the_members_that_moved_most():
    # 25 stores: the largest stores barely moved, the small ones moved a lot.
    rows = [{"store": f"Big {i}", "sales_prior": 10_000.0 + i, "sales": 10_001.0 + i} for i in range(15)]
    rows += [{"store": f"Small {i}", "sales_prior": 100.0, "sales": 900.0 + i} for i in range(10)]
    payload = {**DUMBBELL, "rows": rows}
    kept = json.loads(_build("portal_chat.html", "en", payload, "JSON.stringify(opt.yAxis.data)"))
    assert len(kept) == 20
    assert all(f"Small {i}" in kept for i in range(10)), "every store that moved a lot is drawn"
    order = [r["store"] for r in rows]
    assert kept == sorted(kept, key=order.index), "the members keep the order they came in"
