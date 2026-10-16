"""A chart's form follows what its data shows, and says why (core2/answer/shape_of_data.py).

One test for each row of the plan's table, on charts drawn by the answer builder from invented retail data or
written as the builder writes them, and the dashboard's own check of each tile: not flat, not one member, not the
same as another tile.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.answer.shape_of_data import AVERAGE_KEY, measure, settle, tile_score, weak_tile
from core2.plan.values import build_index
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 10, 15)


def _series(values: list[float | None], *, grain: str = "month", start: str = "2025-01-01", kind: str = "line",
            key: str = "net_amount") -> dict:
    first = dt.date.fromisoformat(start)
    rows = []
    for i, v in enumerate(values):
        month = (first.month - 1 + i) % 12 + 1
        year = first.year + (first.month - 1 + i) // 12
        rows.append({"period": dt.date(year, month, 1).isoformat(), key: v})
    roles = {"period": {"column": "period", "label": "Month", "role": "temporal"},
             key: {"column": key, "label": "Net amount", "role": "measure", "format": "currency"}}
    return {"chart_type": kind, "recommended_type": kind, "x_key": "period", "y_keys": [key], "rows": rows,
            "other": None, "column_roles": roles, "column_formats": {key: "currency"},
            "renderable_types": ["line", "area", "bar"], "allowed_types": ["line", "area", "bar"],
            "chart_spec": {"x": {"column": "period", "role": "temporal"}, "column_roles": roles},
            "chart_warnings": [], "compare": None}


def _members(values: dict[str, float], *, kind: str = "bar", parts: bool = True, other: dict | None = None) -> dict:
    rows = [{"store": k, "net_amount": v} for k, v in values.items()]
    roles = {"store": {"column": "store", "label": "Store", "role": "dimension"},
             "net_amount": {"column": "net_amount", "label": "Net amount", "role": "measure", "format": "currency"}}
    shapes = ["bar", "pie", "donut"] if parts else ["bar"]
    return {"chart_type": kind, "recommended_type": kind, "x_key": "store", "y_keys": ["net_amount"], "rows": rows,
            "other": other, "column_roles": roles, "column_formats": {"net_amount": "currency"},
            "renderable_types": list(shapes), "allowed_types": ["bar", "pie", "donut"],
            "chart_spec": {"x": {"column": "store", "role": "dimension"}, "column_roles": roles},
            "chart_warnings": [], "compare": None, "x_order": ""}


# ── one period only ──────────────────────────────────────────────────────────


def test_one_period_only_is_the_number_never_a_one_point_line():
    assert settle(_series([None, 1200.0, None])) is None


def test_lines_of_members_over_one_period_are_their_bars_side_by_side():
    rows = [{"period": "2026-05-01", "Riverside Store": 300.0, "Old Town Store": 500.0, "Airport Store": 200.0}]
    roles = {"period": {"column": "period", "label": "Month", "role": "temporal"},
             **{s: {"column": s, "label": s, "role": "measure", "format": "currency"} for s in list(rows[0])[1:]}}
    chart = {"chart_type": "line", "x_key": "period", "y_keys": list(rows[0])[1:], "rows": rows, "other": None,
             "column_roles": roles, "grouped_by": "store", "grouped_measure": "net_amount", "member_label": "Store",
             "measure_label": "Net amount", "renderable_types": ["line", "bar", "area"],
             "allowed_types": ["line", "bar", "area"], "chart_spec": {"x": {"column": "period", "role": "temporal"}},
             "chart_warnings": []}
    drawn = settle(chart, things="stores")
    assert drawn["chart_type"] == "bar" and drawn["x_key"] == "store"
    assert [r["store"] for r in drawn["rows"]] == ["Old Town Store", "Riverside Store", "Airport Store"]
    assert drawn["why"]["en"].startswith("Bars: one month only")


# ── few periods, or gaps ─────────────────────────────────────────────────────


def test_fewer_than_four_periods_are_bars():
    drawn = settle(_series([100.0, 140.0, 120.0]))
    assert drawn["chart_type"] == "bar" and "line" in drawn["allowed_types"]
    assert drawn["why"]["en"] == "Bars: 3 months only, too few for a line."
    assert drawn["why"]["fr"] == "Des barres : 3 mois seulement, trop peu pour une courbe."


def test_a_series_with_missing_months_is_bars_with_the_gaps_left_empty():
    drawn = settle(_series([100.0, None, None, 130.0, 120.0, 150.0]))
    assert drawn["chart_type"] == "bar"
    assert [r["net_amount"] for r in drawn["rows"]][1:3] == [None, None]
    assert drawn["why"]["en"] == "Bars: 2 of the 6 months have no data, left empty."


def test_a_long_steady_series_is_a_line_and_says_which_way_it_goes():
    drawn = settle(_series([100.0 + 5 * i for i in range(24)]))
    assert drawn["chart_type"] == "line" and AVERAGE_KEY not in drawn["y_keys"]
    assert drawn["why"]["en"] == "A line: 24 months, rising."
    assert drawn["why"]["fr"] == "Une courbe : 24 mois, en hausse."


# ── parts of a whole ─────────────────────────────────────────────────────────


def test_a_few_parts_of_a_whole_none_dominant_are_a_donut():
    drawn = settle(_members({"Riverside Store": 400.0, "Old Town Store": 300.0, "Airport Store": 200.0,
                             "Hillcrest Store": 100.0}), things="stores")
    assert drawn["chart_type"] == "donut"
    assert drawn["why"]["en"] == "A donut: 4 parts of the whole, the largest 40%."


def test_seven_parts_stay_bars():
    drawn = settle(_members({f"Store {i}": 100.0 + i for i in range(7)}), things="stores")
    assert drawn["chart_type"] == "bar"


def test_parts_whose_largest_is_most_of_the_whole_stay_bars():
    drawn = settle(_members({"Riverside Store": 800.0, "Old Town Store": 150.0, "Airport Store": 50.0}),
                   things="stores")
    assert drawn["chart_type"] == "bar"


def test_one_member_over_ninety_percent_is_bars_naming_it_never_a_pie():
    drawn = settle(_members({"Riverside Store": 950.0, "Old Town Store": 30.0, "Airport Store": 20.0}, kind="pie"),
                   things="stores")
    assert drawn["chart_type"] == "bar" and "pie" not in drawn["allowed_types"]
    assert drawn["why"]["en"] == "Bars: Riverside Store is 95% of the total."


# ── many members ─────────────────────────────────────────────────────────────


def test_more_than_twelve_members_that_do_not_add_up_are_the_largest_ten():
    chart = _members({f"Product {i:02d}": float(i) for i in range(1, 41)}, parts=False)
    drawn = settle(chart, adds=False, things="products")
    assert len(drawn["rows"]) == 10 and drawn["rows"][0]["store"] == "Product 40"
    assert drawn["why"]["en"] == "Bars: the 10 largest of 40 products."
    assert any("every one is in the table" in w for w in drawn["chart_warnings"])


def test_the_builder_folds_many_members_that_add_up_into_other(retail):
    got = _ask(retail, "Net sales by product in 2025", {"intent": "breakdown", "measures": ["net_amount"],
                                                          "group_by": ["product.name"], "time": YEAR})
    chart = got["chart"]
    assert chart["chart_type"] == "bar" and len(chart["rows"]) <= 11
    assert chart.get("other") or len(chart["rows"]) <= 12
    assert chart["why"]["en"].startswith("Bars:")


# ── above and below zero ─────────────────────────────────────────────────────


def test_values_above_and_below_zero_are_bars_from_zero_never_a_donut_or_area():
    drawn = settle(_members({"Riverside Store": 400.0, "Old Town Store": -120.0, "Airport Store": 200.0}, kind="pie"),
                   things="stores")
    assert drawn["chart_type"] == "bar" and not {"pie", "donut", "area"} & set(drawn["allowed_types"])
    series = settle(_series([100.0, -50.0, 80.0, -20.0, 60.0, 40.0]))
    assert series["chart_type"] == "bar" and "area" not in series["allowed_types"]
    assert series["why"]["en"] == "Bars from zero: some months are below zero."


# ── a series that swings ─────────────────────────────────────────────────────


def test_a_series_that_swings_is_drawn_with_its_three_period_average():
    values = [100.0, 180.0, 90.0, 200.0, 80.0, 190.0, 100.0, 210.0, 95.0, 205.0, 85.0, 195.0]
    drawn = settle(_series(values))
    assert drawn["chart_type"] == "line" and drawn["y_keys"] == ["net_amount", AVERAGE_KEY]
    averages = [r[AVERAGE_KEY] for r in drawn["rows"]]
    assert averages[:2] == [None, None] and averages[2] == pytest.approx((100 + 180 + 90) / 3)
    assert drawn["average"]["label"] == "3-month average"
    assert drawn["why"]["en"] == "A line with its 3-month average: 12 months that swing widely."
    assert measure(drawn).points == 12        # the average is never counted as a figure of the series


def test_a_steady_series_with_one_one_off_is_a_plain_line():
    # Near 177K a month and one month at 1.1M: the series does not swing, one month stands out (the analyst note
    # says so and offers it without the one-off). An average would smear that month over three.
    values = [164.5, 157.7, 167.8, 185.7, 168.2, 177.2, 180.6, 1110.0, 180.5, 169.2, 219.1, 215.3]
    drawn = settle(_series(values))
    assert drawn["chart_type"] == "line" and drawn["y_keys"] == ["net_amount"] and "average" not in drawn
    assert drawn["why"]["en"].startswith("A line: 12 months")
    # Nor does it make the year read as falling when the one-off is early: near 180K at both ends is steady.
    early = [177.2, 180.6, 1110.0, 180.5, 169.2, 180.0, 175.0, 168.0, 177.0, 194.9, 180.0, 180.1]
    assert settle(_series(early))["why"]["en"] == "A line: 12 months, steady."


# ── two groupings ────────────────────────────────────────────────────────────


def test_two_groupings_with_most_pairs_filled_are_a_grid():
    series = [f"Segment {i}" for i in range(8)]
    rows = [{"store": f"Store {j}", **{s: 10.0 + i + j for i, s in enumerate(series)}} for j in range(6)]
    chart = {"chart_type": "stacked", "x_key": "store", "y_keys": series, "rows": rows, "grouped_by": "segment",
             "grouped_measure": "net_amount", "renderable_types": ["stacked", "bar", "heatmap"],
             "allowed_types": ["stacked", "bar", "heatmap"], "chart_spec": {"x": {"column": "store",
                                                                                  "role": "dimension"}},
             "chart_warnings": []}
    drawn = settle(chart, things="stores")
    assert drawn["chart_type"] == "heatmap"
    assert drawn["why"]["en"] == "A grid: 6 by 8, most pairs filled."


def test_two_groupings_with_few_colours_stay_bars():
    rows = [{"store": f"Store {j}", "A": 1.0, "B": 2.0} for j in range(6)]
    chart = {"chart_type": "stacked", "x_key": "store", "y_keys": ["A", "B"], "rows": rows, "grouped_by": "segment",
             "renderable_types": ["stacked", "bar", "heatmap"], "allowed_types": ["stacked", "bar", "heatmap"],
             "chart_spec": {"x": {"column": "store", "role": "dimension"}}, "chart_warnings": []}
    assert settle(chart, things="stores")["chart_type"] == "stacked"


# ── many lines ───────────────────────────────────────────────────────────────


def test_more_than_six_lines_are_the_largest_five_and_other(retail):
    got = _ask(retail, "Monthly net sales by product in 2025", {
        "intent": "trend", "measures": ["net_amount"], "group_by": ["product.name"],
        "time": {**YEAR, "grain": "month"}})
    chart = got["chart"]
    assert chart["chart_type"] == "line" and len(chart["y_keys"]) <= 6
    assert chart["why"]["en"].startswith("Lines:")


# ── through the answer: the note says why, the reader's pick still wins ─────


YEAR = {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}


@pytest.fixture(scope="module")
def retail():
    built, model = learn(domains.build("retail"), "descriptive")
    warehouse = DuckDBWarehouse(built.con)

    def fetch(slug: str) -> list[object]:
        column = model.columns[model.attributes[slug].column]
        table = model.tables[column.table]
        return [r[0] for r in warehouse.query(f'SELECT DISTINCT "{column.name}" FROM "{table.name}"').rows]

    return model, warehouse, build_index(model, fetch)


def _ask(retail, question: str, plan: dict, *, chart_type: str | None = None) -> dict:
    model, warehouse, index = retail
    services = Services(model=model, warehouse=warehouse, index=index, today=TODAY,
                        complete=lambda system, tail: json.dumps({"kind": "query", **plan}))
    kwargs = {"chart_type": chart_type} if chart_type else {}
    return answer_question(question, services, Session(), **kwargs)


def test_a_monthly_answer_says_what_decided_its_chart(retail):
    got = _ask(retail, "Net sales by month in 2025", {"intent": "trend", "measures": ["net_amount"],
                                                       "time": {**YEAR, "grain": "month"}})
    assert got["chart"]["why"]["en"].startswith(("A line: 12 months", "A line with its 3-month average"))
    assert got["chart"]["why"]["fr"]


def test_a_breakdown_of_every_member_of_a_few_is_a_donut_a_top_three_or_an_average_stays_bars(retail):
    plan = {"intent": "breakdown", "measures": ["net_amount"], "group_by": ["region.name"], "time": YEAR}
    whole = _ask(retail, "Net sales by region in 2025", plan)["chart"]
    assert whole["chart_type"] == "donut" and whole["why"]["en"].startswith("A donut: 5 parts of the whole")
    top = _ask(retail, "Top 3 regions by net sales in 2025",
               {**plan, "limit": 3, "sort": [{"by": "net_amount", "desc": True}]})["chart"]
    assert top["chart_type"] == "bar"
    average = _ask(retail, "Average unit price by region in 2025", {**plan, "measures": ["unit_price"]})["chart"]
    assert average["chart_type"] == "bar"


# ── dashboard tiles: worth their place ───────────────────────────────────────


def _payload(rows: list[dict], headers: list[str], chart: dict | None = None) -> dict:
    return {"data": {"rows": rows, "headers": headers, "header_labels": {h: h.title() for h in headers}},
            "chart": chart or {}}


def test_a_trend_that_never_moves_is_left_out_and_said():
    chart = _series([500.0] * 6)
    rows = [{"period": r["period"], "net_amount": r["net_amount"]} for r in chart["rows"]]
    why = weak_tile("trend", _payload(rows, ["period", "net_amount"], chart), [])
    assert why.startswith("The same in every period")


def test_a_breakdown_of_one_member_is_left_out_and_said():
    why = weak_tile("breakdown", _payload([{"store": "Riverside Store", "net_amount": 10.0}], ["store", "net_amount"]),
                    [])
    assert why == "Only one store has any, so there is nothing to compare."


def test_a_tile_showing_the_same_as_another_is_left_out_and_said():
    rows = [{"store": "A", "net_amount": 1.0}, {"store": "B", "net_amount": 2.0}]
    first = _payload(rows, ["store", "net_amount"])
    assert weak_tile("breakdown", _payload(list(rows), ["store", "net_amount"]), [("Sales by store", first)]) \
        == "It shows the same as “Sales by store”."
    assert weak_tile("breakdown", first, []) == ""


def test_a_tile_that_shows_more_scores_higher():
    flat = {"data": {"rows": [{}] * 3}, "chart": {"shape": {"swing": 0.0}}}
    moving = {"data": {"rows": [{}] * 12}, "chart": {"shape": {"swing": 0.4}}}
    assert tile_score("trend", moving) > tile_score("trend", flat)


def test_a_dashboard_leaves_out_its_weak_tiles_and_lists_them_as_not_built():
    from core2 import dashboards

    rows = [{"store": "A", "net_amount": 1.0}, {"store": "B", "net_amount": 2.0}]
    flat = _series([500.0] * 6)
    tiles = [dashboards.Tile("Net amount by store", "breakdown", "q1", payload=_payload(rows, ["store", "net_amount"])),
             dashboards.Tile("Sales by store", "breakdown", "q2", payload=_payload(list(rows), ["store", "net_amount"])),
             dashboards.Tile("Monthly fees", "trend", "q3", payload=_payload(
                 [{"period": r["period"], "net_amount": r["net_amount"]} for r in flat["rows"]],
                 ["period", "net_amount"], flat)),
             dashboards.Tile("Refunds", "number", "q4", error="The data has nothing for it, in any period.")]
    dashboards.scored(tiles)
    draft = dashboards.Draft(name="Sales", request="a sales dashboard", tiles=tiles)
    assert [t["title"] for t in dashboards.not_built(draft)] == ["Sales by store", "Monthly fees", "Refunds"]
    assert dashboards.not_built(draft)[0]["error"] == "It shows the same as “Net amount by store”."
    assert [t["left_out"] for t in dashboards.not_built(draft)] == [True, True, False]
    assert not tiles[0].error and tiles[0].score > 0


def test_a_tile_repeated_on_its_own_tab_is_kept():
    # An area's tab repeats the overview's headline ("Delinquent loans" on Overview and on Risk): each tab is read on
    # its own, so a repeat is only a repeat on the same tab.
    from core2 import dashboards

    rows = [{"store": "A", "net_amount": 1.0}, {"store": "B", "net_amount": 2.0}]
    tiles = [dashboards.Tile("Net amount by store", "breakdown", "q1", tab="Overview",
                             payload=_payload(rows, ["store", "net_amount"])),
             dashboards.Tile("Net amount by store", "breakdown", "q1", tab="Stores",
                             payload=_payload(list(rows), ["store", "net_amount"]))]
    dashboards.scored(tiles)
    assert not tiles[0].error and not tiles[1].error


def test_a_dashboard_whose_every_tile_is_weak_keeps_its_strongest():
    from core2 import dashboards

    tiles = [dashboards.Tile("Only store", "breakdown", "q", payload=_payload([{"store": "A", "net_amount": 1.0}],
                                                                              ["store", "net_amount"]))]
    dashboards.scored(tiles)
    assert not tiles[0].error


# ── the largest ten say how many the rest hold ───────────────────────────────


def test_the_rest_of_the_largest_ten_never_counts_unknown_as_a_member():
    # 30 products and the rows with no product: the warning says 20 others and Unknown, as the analyst note
    # counts 30 products ("Unknown" is rows with no member, never one of them).
    from core2.answer.builder import _fold

    rows = [{"product": f"Product {i}", "net": 1000.0 - i} for i in range(30)] + [{"product": "Unknown", "net": 5.0}]
    kept, other, said = _fold(rows, "product", ["net"], 10, None, format_="currency", things="products")
    assert other is None and len(kept) == 10
    assert said.startswith("Showing the 10 largest: the other 20 products and Unknown add up to $19.6K")
    small = [{"product": f"Product {i}", "net": 1000.0 if i == 0 else 100.0} for i in range(14)] \
        + [{"product": None, "net": 1.0}]
    kept, other, _ = _fold(small, "product", ["net"], 10, None, format_="currency", things="products")
    assert other == {"label": "Other (4)", "count": 4}


def test_bars_cut_to_the_largest_ten_say_how_many_they_are_taken_from():
    rows = [{"product": f"Product {i}", "net": 1000.0 - i} for i in range(10)]
    chart = {"chart_type": "bar", "recommended_type": "bar", "x_key": "product", "y_keys": ["net"], "rows": rows,
             "other": None, "shown_of": 110, "whole": False, "renderable_types": ["bar"], "allowed_types": ["bar"],
             "chart_spec": {"x": {"column": "product", "role": "dimension"}}, "chart_warnings": []}
    drawn = settle(chart, things="products")
    assert drawn["chart_type"] == "bar" and len(drawn["rows"]) == 10
    assert drawn["why"] == {"en": "Bars: the 10 largest of 110 products.", "fr": "Des barres : les 10 plus grands sur 110."}


def test_the_card_lists_tiles_left_out_apart_from_tiles_that_failed():
    # A tile left out for showing nothing new was listed as one that "could not be built".
    from tests.test_core2_an_executive_dashboard_is_built_by_its_business_recipe import MADE, _card

    same = {"title": "Closing value", "question": "Closing value", "error": "It shows the same as “On hand value”.",
            "definable": False, "left_out": True}
    failed = {"title": "Margin by brand", "question": "Margin by brand", "error": "Nothing in this data measures it.",
              "definable": True, "left_out": False}
    card = _card({**MADE, "not_built": [same, failed]})
    assert "1 could not be built:" in card and "Margin by brand" in card
    assert "1 left out, as they would show nothing new:" in card and "It shows the same as “On hand value”." in card
    assert "could not be built" not in _card({**MADE, "not_built": [same]})
