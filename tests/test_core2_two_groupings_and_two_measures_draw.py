"""Two groupings, or two measures, are drawn as what they say.

"Net amount by store and segment" drew one bar per row: Old Town Store three times, once per segment, side by
side with no way to tell which bar was which segment. Two measures over sixty products drew two panels of
sixty bars each, when the question was how one goes with the other.

* two groupings and no time: a bar per member of the one with more members, its parts the other's -- stacked
  when the measure adds up, side by side when it does not; past six parts, a grid of both, darker for more;
* two measures over many members, or a dozen in two different units: a dot per member, the furthest named;
* the reader can ask for each ("stacked", "as a heatmap", "as a scatter").

Invented data only.
"""

from __future__ import annotations

import json

import pytest

from core2.plan.followup import _DISPLAY
from tests.test_chart_annotation_language import _build
from tests.test_core2_charts_read_as_their_answer import H1, _all_rows, _ask
from tests.test_a_new_core_answer_can_be_pinned import retail  # noqa: F401

BY_STORE_AND_SEGMENT = {"intent": "breakdown", "measures": ["net_amount"], "group_by": ["store", "customer.segment"],
                        "time": {"window": H1}}


def _cells(retail, payload: dict, across: str, down: str, measure: str) -> dict:  # noqa: F811
    def name(v):        # a row with no member is "Unknown", as the answer shows it
        return "Unknown" if v in (None, "") else str(v)
    return {(name(r[across]), name(r[down])): r[measure] for r in _all_rows(retail, payload)}


def test_a_measure_that_adds_up_is_a_bar_per_store_of_its_segments(retail):  # noqa: F811
    payload = _ask(retail, BY_STORE_AND_SEGMENT)
    chart = payload["chart"]
    assert chart["chart_type"] == "stacked" and chart["title"] == "Net amount by store and segment"
    assert sorted(chart["y_keys"]) == ["Online", "Retail", "Wholesale"], "the segments are the parts"
    store, segment = chart["x_key"], chart["grouped_by"]
    cells = _cells(retail, payload, store, segment, "net_amount")
    for row in chart["rows"]:
        for part in chart["y_keys"]:
            assert row[part] == pytest.approx(cells.get((row[store], part), 0.0))
    totals = [sum(row[k] for k in chart["y_keys"]) for row in chart["rows"]]
    assert totals == sorted(totals, reverse=True), "the largest store first"
    assert chart["renderable_types"] == ["stacked", "bar", "heatmap"]


def test_an_average_is_never_stacked(retail):  # noqa: F811
    chart = _ask(retail, {**BY_STORE_AND_SEGMENT, "measures": ["unit_price"]})["chart"]
    assert chart["chart_type"] == "bar" and "stacked" not in chart["renderable_types"]


def test_more_than_six_parts_are_a_grid(retail):  # noqa: F811
    payload = _ask(retail, {**BY_STORE_AND_SEGMENT, "group_by": ["store", "category.name"]})
    chart = payload["chart"]
    assert chart["chart_type"] == "heatmap" and chart["renderable_types"] == ["heatmap"]
    assert len(chart["y_keys"]) > 6 and chart["grouped_measure"] == "net_amount"
    cells = _cells(retail, payload, chart["x_key"], chart["grouped_by"], "net_amount")
    row = chart["rows"][0]
    assert all(row[c] == pytest.approx(cells.get((row[chart["x_key"]], c), 0.0)) for c in chart["y_keys"])
    drawn = json.loads(_build("portal_chat.html", "en", chart, "JSON.stringify(opt.series[0].type)"))
    assert drawn == "heatmap"


def test_three_groupings_are_a_table(retail):  # noqa: F811
    payload = _ask(retail, {**BY_STORE_AND_SEGMENT, "group_by": ["store", "customer.segment", "category.name"]})
    assert payload["chart"] is None and payload["data"]["total_rows"] > 0


def test_the_reader_can_ask_for_the_grid(retail):  # noqa: F811
    chart = _ask(retail, {**BY_STORE_AND_SEGMENT, "chart": "heatmap"})["chart"]
    assert chart["chart_type"] == "heatmap"


def test_stacked_bars_are_drawn_piled_with_their_total_at_the_end(retail):  # noqa: F811
    chart = _ask(retail, BY_STORE_AND_SEGMENT)["chart"]
    series = json.loads(_build("portal_chat.html", "en", chart, """JSON.stringify(opt.series.map(s => ({
        stack: s.stack, gap: s.itemStyle.borderWidth, label: !!(s.label && s.label.show)})))"""))
    assert all(s["stack"] == "total" and s["gap"] == 1 for s in series)
    assert [s["label"] for s in series] == [False, False, True], "one total per bar, at its end"
    total = json.loads(_build("portal_chat.html", "en", chart,
                              "JSON.stringify(opt.series[2].label.formatter({dataIndex: 0}))"))
    first = chart["rows"][0]
    assert total == json.loads(_build("portal_chat.html", "en", chart, "JSON.stringify(QBCharts.formatValue("
                                      f"{sum(first[k] for k in chart['y_keys'])}, 'currency', true))"))


# ── two measures ────────────────────────────────────────────────────────────


def test_two_measures_over_many_members_are_a_dot_each(retail):  # noqa: F811
    chart = _ask(retail, {"intent": "breakdown", "measures": ["net_amount", "quantity"],
                          "group_by": ["product.name"], "time": {"window": H1}})["chart"]
    assert chart["chart_type"] == "scatter" and chart["renderable_types"] == ["scatter", "bar"]
    assert chart["y_keys"] == ["net_amount", "quantity"] and len(chart["rows"]) > 12 and chart["other"] is None
    named = json.loads(_build("portal_chat.html", "en", chart,
                              "JSON.stringify(opt.series[0].data.filter(d => d.label && d.label.show).length)"))
    assert named == 3, "the three furthest out are named; the rest in the tooltip"


def test_a_dozen_in_two_units_are_a_dot_each(retail):  # noqa: F811
    chart = _ask(retail, {"intent": "breakdown", "measures": ["net_amount", "margin_percent"],
                          "group_by": ["store"], "time": {"window": H1}})["chart"]
    assert chart["chart_type"] == "scatter"


def test_two_amounts_of_a_few_members_stay_bars_and_offer_the_dots(retail):  # noqa: F811
    chart = _ask(retail, {"intent": "breakdown", "measures": ["net_amount", "cost_amount"],
                          "group_by": ["store"], "time": {"window": H1}})["chart"]
    assert chart["chart_type"] == "bar" and "scatter" in chart["renderable_types"]


def test_one_measure_is_never_a_scatter(retail):  # noqa: F811
    payload = _ask(retail, {"intent": "breakdown", "measures": ["net_amount"], "group_by": ["store"],
                            "time": {"window": H1}, "chart": "scatter"})
    assert payload["chart"]["chart_type"] == "bar"
    assert any("scatter chart does not fit" in n for n in payload["trust"]["date_context"])


@pytest.mark.parametrize("words", ["show that stacked", "as a heatmap", "as a scatter", "stacked bars"])
def test_asking_for_a_shape_changes_the_answer_on_screen(words):
    assert _DISPLAY.search(words)


# ── what the answer says of them ────────────────────────────────────────────


def test_the_sentence_names_the_store_with_the_most_in_all_never_one_pair(retail):  # noqa: F811
    payload = _ask(retail, {**BY_STORE_AND_SEGMENT, "group_by": ["store", "category.name"]})
    totals: dict = {}
    for r in _all_rows(retail, payload):
        totals[r["store_name"]] = totals.get(r["store_name"], 0.0) + r["net_amount"]
    leader = max(totals, key=totals.get)
    headline = payload["answer"]["headline"]
    assert f"{leader} leads with ${totals[leader]:,.2f}" in headline, "its rows with no category are in its total"
    assert "across 12 stores; its largest category is" in headline
    assert payload["key_insights"][0].startswith("The top 3 of the 12 stores"), "stores, never store-category pairs"


def test_an_average_by_two_groupings_names_its_highest_pair(retail):  # noqa: F811
    payload = _ask(retail, {**BY_STORE_AND_SEGMENT, "measures": ["unit_price"]})
    assert "are highest, at" in payload["answer"]["headline"] and "of 36 pairs of stores and segments" in \
        payload["answer"]["headline"]
    assert payload["key_insights"] == [], "an average per pair adds up to nothing to rank"


def test_the_units_note_names_the_measure_counted_in_units(retail):  # noqa: F811
    payload = _ask(retail, {"intent": "breakdown", "measures": ["net_amount", "quantity"],
                            "group_by": ["product.name"], "time": {"window": H1}})
    assert payload["coverage_caveats"][0].startswith("Quantity is counted in"), "net amount has a currency"


DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def test_days_of_the_week_run_monday_to_sunday(retail):  # noqa: F811
    one = _ask(retail, {"intent": "breakdown", "measures": ["net_amount"], "group_by": ["time:day_of_week"],
                        "time": {"window": H1}})["chart"]
    assert [r[one["x_key"]] for r in one["rows"]] == DAYS, "a week in its order, never largest first"
    payload = _ask(retail, {"intent": "breakdown", "measures": ["net_amount"],
                            "group_by": ["region.name", "time:day_of_week"], "time": {"window": H1}})
    two = payload["chart"]
    assert [r[two["x_key"]] for r in two["rows"]] == DAYS and two["chart_type"] == "stacked"
    assert " across 7 " not in payload["answer"]["headline"], "the seven days are not counted"
    assert payload["key_insights"][0].startswith("The top 3 of the 7 days"), "days, never day-region pairs"
