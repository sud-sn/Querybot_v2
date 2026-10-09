"""An answer reads cleanly whatever its size: tiles, tables, summaries and charts, on a desk or a phone.

A sweep of answers on an inventory workspace found the figures drawn but hard to read:
- a stock value of $7,356,042.88 overflowed its tile and was cut off ("$7,356,042.8…");
- a metric's percent change was formatted as money ("-$99.83") because the column carried the
  metric's name and so its currency format;
- a single row was summarised as a range ("1 record, ranges from ... to ..., avg ...");
- when the new core handed a question back to today's pipeline the reader was not told why;
- chart labels mixed "-$549.25" with "$4.6K", and on a phone the names beside a ranking took
  180px of a 340px card, leaving the bars as slivers.

Invented names and synthetic figures only.
"""

from __future__ import annotations

import json

import pytest


# ── tiles that fit, and a percentage that reads as one ───────────────────────

def _tiles(msg: dict) -> list[dict]:
    from tests.chat_js import run
    from tests.test_the_answer_card import FUNCTIONS, PREAMBLE

    return run(f"JSON.stringify(_answerTiles({json.dumps(msg)}))", functions=FUNCTIONS,
               consts=["_NUMERIC_FORMATS", "_QB_CURRENCY_SYMBOL"], preamble=PREAMBLE)


def test_a_long_amount_is_drawn_short_with_the_exact_amount_under_it():
    msg = {"data": {"headers": ["INVENTORY_VALUE", "INVENTORY_VALUE_PRIOR", "INVENTORY_VALUE_CHANGE", "ITEMS"],
                    "column_formats": {"INVENTORY_VALUE": "currency", "INVENTORY_VALUE_PRIOR": "currency",
                                       "INVENTORY_VALUE_CHANGE": "currency", "ITEMS": "number"},
                    "rows": [{"INVENTORY_VALUE": 7356042.88, "INVENTORY_VALUE_PRIOR": 412500.5,
                              "INVENTORY_VALUE_CHANGE": -4399219.03, "ITEMS": 1250000}]}}
    tiles = _tiles(msg)
    assert [(t["value"], t["exact"]) for t in tiles] == [
        ("$7.36M", "$7,356,042.88"), ("$412,500.50", ""), ("-$4.4M", "-$4,399,219.03"), ("1.25M", "1,250,000")]


def test_a_metrics_percent_change_is_a_percentage_not_its_money():
    from core.response_builder import build_column_formats

    rows = [{"INVENTORY_VALUE": 7356.04, "INVENTORY_VALUE_PRIOR": 4406575.07, "INVENTORY_VALUE_CHANGE": -4399219.03,
             "INVENTORY_VALUE_PERCENT_CHANGE": -99.83, "MARGIN_PCT": 12.5, "REGION": "East"}]
    formats = build_column_formats(rows, {"metrics": [{"name": "Inventory value", "result_format": "currency",
                                                       "metric_name": "inventory_value"}]})
    assert formats["INVENTORY_VALUE"] == formats["INVENTORY_VALUE_CHANGE"] == "currency"
    assert formats["INVENTORY_VALUE_PERCENT_CHANGE"] == formats["MARGIN_PCT"] == "percentage"
    tile = _tiles({"data": {"headers": ["INVENTORY_VALUE_CHANGE", "INVENTORY_VALUE_PERCENT_CHANGE"],
                            "column_formats": formats, "rows": [rows[0]]}})
    assert tile[1]["value"] == "-99.83%"


def test_one_row_has_no_range_to_tell():
    from core.response_builder import _listing_summary

    one = [{"WAREHOUSE": "North", "INVENTORY_VALUE": 7356.04}]
    ctx = {"numeric_cols": ["INVENTORY_VALUE"]}
    assert _listing_summary(one, ctx, {"INVENTORY_VALUE": "currency"}, lambda v, c: f"${v:,.2f}") == ""
    two = one + [{"WAREHOUSE": "South", "INVENTORY_VALUE": 100.0}]
    assert "2 records" in _listing_summary(two, ctx, {"INVENTORY_VALUE": "currency"}, lambda v, c: f"${v:,.2f}")


def test_one_row_of_figures_is_answered_by_its_figure_not_by_a_row_count():
    # The answer the inventory workspace drew: one row, a metric against its earlier value.
    from core.response_builder import build_assistant_response

    rows = [{"INVENTORY_VALUE": 7356042.88, "INVENTORY_VALUE_PRIOR": 4406575.07,
             "INVENTORY_VALUE_CHANGE": 2949467.81, "INVENTORY_VALUE_PERCENT_CHANGE": 66.93}]
    payload = build_assistant_response(
        question="What is our inventory value compared with the last snapshot?", rows=rows, sql="SELECT 1",
        duration_ms=5, display_context={"metrics": [{"name": "Inventory value", "result_format": "currency",
                                                     "metric_name": "inventory_value"}]})
    assert payload["answer"]["headline"] == "Inventory Value: $7,356,042.88."
    assert payload["insight_summary"] == ""
    assert payload["data"]["column_formats"]["INVENTORY_VALUE_PERCENT_CHANGE"] == "percentage"


def test_one_record_still_says_what_it_comes_to():
    from core.response_builder import _listing_summary

    one = [{"INVOICE": "INV01", "NET_SLS_AMT": 125.0}]
    said = _listing_summary(one, {"numeric_cols": ["NET_SLS_AMT"]}, {"NET_SLS_AMT": "currency"},
                            lambda v, c: f"${v:,.2f}")
    assert said == "1 record — Net Sls Amount totals $125.00."


def test_when_the_new_core_cannot_answer_the_reader_is_told_who_does():
    from gateway.core2_bridge import _handed_back

    assert _handed_back("timeout", None) == ("The new core did not answer this within 120 seconds. "
                                             "Today's pipeline answers it instead.")
    said = _handed_back("unsupported", {"answer": {"headline": "I cannot answer that from this data: no measure "
                                                                "counts returns."}})
    assert said == ("The new core cannot answer this from the data (no measure counts returns). "
                    "Today's pipeline answers it instead.")


# ── charts: whole hundreds beside thousands, and room for the bars on a phone ─

def _drawn(payload: dict, expression: str, layout: dict | None = None):
    from tests.test_chart_annotation_language import _build

    built = f"QBCharts.buildOption({json.dumps(payload)}, {json.dumps(layout)})" if layout else "opt"
    return json.loads(_build("portal_chat.html", "en", payload, f"(function () {{ var o = {built}; return JSON.stringify({expression}); }})()"))


RANKED = {"rows": [{"ITEM": "Anchor bolt kit, galvanised, 12 mm", "VALUE": 4600.0},
                   {"ITEM": "Hinge, stainless, heavy duty", "VALUE": 2310.4},
                   {"ITEM": "Washer", "VALUE": -549.25}],
          "x_key": "ITEM", "y_keys": ["VALUE"], "chart_type": "bar",
          "column_formats": {"VALUE": "currency"}}


def test_a_label_in_the_hundreds_has_no_cents_beside_one_in_thousands():
    labels = _drawn(RANKED, "[0, 1, 2].map(i => o.series[0].label.formatter({value: o.series[0].data[i].value"
                            " !== undefined ? o.series[0].data[i].value : o.series[0].data[i], dataIndex: i}))")
    assert labels == ["$4.6K", "$2.3K", "-$549"], labels


@pytest.mark.parametrize("value,expected", [(549.25, "549"), (100, "100"), (99.5, "99.5"), (12.25, "12.25"),
                                            (4600, "4.6K"), (0.004, "0.004")])
def test_the_compact_ladder(value, expected):
    assert _drawn(RANKED, f"QBCharts.compactNumber({value})") == expected


@pytest.mark.parametrize("width,names", [(340, 113), (240, 96), (600, 180), (1200, 180)])
def test_the_names_beside_a_ranking_take_at_most_a_third_of_a_narrow_card(width, names):
    drawn = _drawn(RANKED, "{names: o.yAxis.axisLabel.width, horizontal: o.yAxis.type}", {"width": width, "height": 360})
    assert drawn == {"names": names, "horizontal": "category"}


def test_without_a_size_the_names_keep_their_desk_width():
    assert _drawn(RANKED, "o.yAxis.axisLabel.width") == 180
