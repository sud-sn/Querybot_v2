"""
A result reads as what it holds: its chart, its headline, its top rows.

An inventory warehouse keeps a quantity per unit of measure, and most of its
answers come back a row per thing PER UNIT -- feet are not added to eaches.
Each piece of the answer card that met such a result read it as something
else:

* "units sold by month" is a row per month per unit, and a unit is rarely in
  every month: the grid was too sparse for a series, and the chart became a
  table. "month-end stock by month" lost its chart too, because the measure's
  own name, MONTH_END_STOCK_ON_HAND, carries "month" and was read as the time
  axis. "stock on hand for the North Depot warehouse" was drawn along the one
  warehouse it was filtered to, as one row of a heatmap.
* "stock on hand by warehouse" was headed "Returned 19 rows for ...", "units
  sold last 6 months" -- one row, in no unit -- "Unknown leads at 939,315.50",
  and a blank unit labelled "Unknown" for the reader was listed as a unit
  called UNKNOWN.
* "just the top 5" of a breakdown kept the first five rows it was listed in.
* a group whose key matched no row of its dimension was headed "None leads".
"""

from __future__ import annotations

import pytest

from core import i18n


# ── The chart ────────────────────────────────────────────────────────────────

_UNITS = ["EA", "FT", "ME", "LN", "RL", "PK", "BX", "CS", "KG", "LB", "M2"]


def _sold_by_month(units: list[str], months: int = 12) -> list[dict]:
    """A row per month per unit; each unit but the first sold in a few months."""
    rows = []
    for month in range(1, months + 1):
        for index, unit in enumerate(units):
            if index == 0 or (month + index) % 4 == 0:
                rows.append({"PERIOD": f"2026-{month:02d}-01", "UNT_OF_MSR": unit,
                             "UNITS_SOLD": float(1000 - index * 50 + month)})
    return rows


class TestATrendKeptInSeveralUnits:

    def test_is_a_line_per_unit(self):
        from core.chart_spec import infer_chart_spec

        spec = infer_chart_spec(_sold_by_month(_UNITS[:5]), "units sold by month")

        assert spec["recommended_type"] == "line"
        assert spec["x"]["column"] == "PERIOD"
        assert spec["series"]["column"] == "UNT_OF_MSR"
        assert not spec["series"].get("members")

    def test_draws_the_largest_units_when_there_are_more_than_the_palette(self):
        from core.chart import build_chart_payload
        from core.chart_spec import infer_chart_spec

        rows = _sold_by_month(_UNITS)
        spec = infer_chart_spec(rows, "units sold by month")

        assert spec["recommended_type"] == "line"
        assert spec["series"]["members"] == _UNITS[:8]
        assert spec["warnings"] == [
            "A line per unit of measure: the 8 largest of 11 are drawn; the table has every unit."]

        payload = build_chart_payload(rows, "line", "units sold by month", "units sold by month")
        drawn = [key for key in payload["rows"][0] if key != "PERIOD"]
        assert sorted(drawn) == sorted(_UNITS[:8])
        assert len(payload["rows"]) == 12

    def test_a_breakdown_that_is_no_trend_is_not_drawn_per_unit(self):
        from core.chart_spec import infer_chart_spec

        rows = [{"WAREHOUSE": warehouse, "UNT_OF_MSR": unit, "STOCK_ON_HAND": 10.0}
                for warehouse in ("A", "B", "C") for unit in _UNITS[:9]]
        spec = infer_chart_spec(rows, "stock on hand by warehouse")

        assert (spec.get("series") or {}).get("members") is None

    def test_a_measure_named_for_the_month_end_is_a_measure(self):
        from core.chart_spec import infer_chart_spec

        rows = [{"PERIOD": f"2026-{month:02d}-01", "MONTH_END_STOCK_ON_HAND": 31624854.78 + month}
                for month in range(1, 7)]
        spec = infer_chart_spec(rows, "month-end stock by month")

        assert spec["recommended_type"] in {"line", "area"}
        assert [y["column"] for y in spec["y"]] == ["MONTH_END_STOCK_ON_HAND"]

    @pytest.mark.parametrize("column, values", [
        ("SALES_MONTH", ["2026-01-01", "2026-02-01", "2026-03-01"]),
        ("INVOICE_MONTH", [1, 2, 3]),
        ("WEEK", [1, 2, 53]),
        ("YEAR", [2024, 2025, 2026]),
        ("SALES_WEEK", list(range(1, 13))),
        ("SHIP_YEAR", [2023, 2024, 2025]),
        ("SHIP_MONTH", list(range(23, 29))),
        ("SALES_WEEK_NUM", list(range(1, 13))),
    ])
    def test_a_period_is_still_a_period(self, column, values):
        from core.chart_spec import _column_roles

        rows = [{column: value, "SALES": 10.0} for value in values]
        assert _column_roles(rows)[column]["role"] == "temporal"

    def test_one_member_filtered_to_is_not_the_axis(self):
        from core.chart_spec import infer_chart_spec

        rows = [{"WAREHOUSE": "NORTH DEPOT", "UNT_OF_MSR": unit, "STOCK_ON_HAND": float(10 + index)}
                for index, unit in enumerate(_UNITS[:9])]
        spec = infer_chart_spec(rows, "stock on hand for the North Depot warehouse")

        assert spec["recommended_type"] == "bar"
        assert spec["x"]["column"] == "UNT_OF_MSR"


# ── The headline ─────────────────────────────────────────────────────────────

_BY_WAREHOUSE = [
    {"WAREHOUSE": "NORTH DEPOT", "UNT_OF_MSR": "FT", "STOCK_ON_HAND": 900.0},
    {"WAREHOUSE": "NORTH DEPOT", "UNT_OF_MSR": "EA", "STOCK_ON_HAND": 160.0},
    {"WAREHOUSE": "SOUTH DEPOT", "UNT_OF_MSR": "FT", "STOCK_ON_HAND": 350.0},
    {"WAREHOUSE": "SOUTH DEPOT", "UNT_OF_MSR": "EA", "STOCK_ON_HAND": 60.0},
]


def _headline(rows: list[dict], question: str, lang: str = "en", mode: str = "ranking") -> str:
    from core.response_builder import build_answer, infer_result_scope

    token = i18n.activate_language(lang)
    try:
        return build_answer(rows, question, infer_result_scope(rows, question, mode=mode))["headline"]
    finally:
        i18n.deactivate_language(token)


class TestTheHeadline:

    def test_a_quantity_by_warehouse_and_unit_is_ranked_in_its_main_unit(self):
        assert _headline(_BY_WAREHOUSE, "stock on hand by warehouse") == "In FT, NORTH DEPOT leads at 900 FT."

    def test_the_lowest(self):
        assert (_headline(_BY_WAREHOUSE, "which warehouses have the lowest stock?")
                == "In FT, SOUTH DEPOT is lowest at 350 FT.")

    def test_in_french(self):
        assert (_headline(_BY_WAREHOUSE, "stock en main par entrepôt", "fr")
                == "En FT, NORTH DEPOT arrive en tête avec 900 FT.")

    def test_nothing_leads_where_every_one_is_at_nothing(self):
        rows = [{**row, "STOCK_ON_HAND": 0.0} for row in _BY_WAREHOUSE]

        assert _headline(rows, "available quantity by warehouse") == "Stock On Hand is 0 in every unit of measure."

    def test_the_unit_a_leader_is_said_in_holds_some(self):
        rows = [{"WAREHOUSE": w, "UNT_OF_MSR": "BG", "STOCK_ON_HAND": 0.0} for w in ("A DEPOT", "B DEPOT")] + [
            {"WAREHOUSE": "A DEPOT", "UNT_OF_MSR": "EA", "STOCK_ON_HAND": 10.0},
            {"WAREHOUSE": "B DEPOT", "UNT_OF_MSR": "EA", "STOCK_ON_HAND": 4.0}]

        assert _headline(rows, "stock on hand by warehouse") == "In EA, A DEPOT leads at 10 EA."

    def test_the_lowest_is_where_a_zero_is(self):
        rows = [{"WAREHOUSE": "SOUTH DEPOT", "UNT_OF_MSR": "FT", "ALLOCATED_QUANTITY": 0.0},
                {"WAREHOUSE": "NORTH DEPOT", "UNT_OF_MSR": "FT", "ALLOCATED_QUANTITY": 100.0},
                {"WAREHOUSE": "SOUTH DEPOT", "UNT_OF_MSR": "EA", "ALLOCATED_QUANTITY": 60.0},
                {"WAREHOUSE": "NORTH DEPOT", "UNT_OF_MSR": "EA", "ALLOCATED_QUANTITY": 20.0}]

        assert (_headline(rows, "which warehouse has the lowest allocated quantity?")
                == "In FT, SOUTH DEPOT is lowest at 0 FT.")

    def test_a_tie_leads_nothing(self):
        rows = [{"WAREHOUSE": w, "UNT_OF_MSR": "EA", "STOCK_ON_HAND": 1.0} for w in ("W1", "W2", "W3")] + [
            {"WAREHOUSE": "W1", "UNT_OF_MSR": "FT", "STOCK_ON_HAND": 50000.0},
            {"WAREHOUSE": "W2", "UNT_OF_MSR": "FT", "STOCK_ON_HAND": 40000.0}]

        assert " leads at " not in _headline(rows, "stock on hand by warehouse")

    def test_money_is_not_said_in_a_unit(self):
        rows = [{"WAREHOUSE": row["WAREHOUSE"], "UNT_OF_MSR": row["UNT_OF_MSR"],
                 "INVENTORY_VALUE": row["STOCK_ON_HAND"] * 2.5} for row in _BY_WAREHOUSE]

        assert " FT" not in _headline(rows, "inventory value by warehouse")

    def test_one_units_total_is_not_a_leader(self):
        rows = [{"UNT_OF_MSR": None, "UNITS_SOLD": 939315.5}]

        assert _headline(rows, "units sold last 6 months") == "Units Sold by unit of measure: 939,315.50 with no unit."

    def test_a_unit_labelled_unknown_is_no_unit(self):
        from core.units_of_measure import unit_of

        assert unit_of("Unknown") == unit_of("Inconnu") == unit_of(None) == ""
        assert unit_of("ea") == "EA"

    def test_the_member_filtered_to_is_not_what_the_headline_speaks_about(self):
        rows = [{"WAREHOUSE": "NORTH DEPOT", "UNT_OF_MSR": "FT", "STOCK_ON_HAND": 900.0},
                {"WAREHOUSE": "NORTH DEPOT", "UNT_OF_MSR": "EA", "STOCK_ON_HAND": 160.0}]

        assert (_headline(rows, "stock on hand for the north depot warehouse")
                == "Stock On Hand by unit of measure: 900 FT and 160 EA.")

    def test_a_group_with_no_label_is_unknown(self):
        rows = [{"ITEM_GROUP": None, "NUMBER_OF_RECEIPTS": 583.0},
                {"ITEM_GROUP": "FITTINGS", "NUMBER_OF_RECEIPTS": 69.0}]

        assert _headline(rows, "number of receipts by item group") == "Unknown leads at 583."


class TestBlankLabels:

    def test_an_empty_group_label_is_named(self):
        from core.unknown_members import grouped_columns, label_blank_members

        sql = ("SELECT g.ITM_GRP_DSC AS ITEM_GROUP, CAST(d.DMS_DT AS varchar(10)) AS PERIOD, COUNT(*) AS RECEIPTS "
               "FROM F f LEFT JOIN G g ON f.G = g.G LEFT JOIN D d ON f.D = d.D "
               "GROUP BY g.ITM_GRP_DSC, CAST(d.DMS_DT AS varchar(10))")
        rows = [{"ITEM_GROUP": None, "PERIOD": "2026-01-01", "RECEIPTS": 5},
                {"ITEM_GROUP": "FITTINGS", "PERIOD": None, "RECEIPTS": None}]
        named, changed = label_blank_members(rows, "Unknown", grouped_columns(sql))

        assert grouped_columns(sql) == {"ITEM_GROUP", "PERIOD"}
        assert changed == 1
        assert [row["ITEM_GROUP"] for row in named] == ["Unknown", "FITTINGS"]
        # A date and a measure keep their blanks.
        assert named[1]["PERIOD"] is None and named[1]["RECEIPTS"] is None

    def test_a_column_the_query_does_not_group_by_keeps_its_blanks(self):
        from core.unknown_members import grouped_columns, label_blank_members

        rows = [{"ORDER_ID": "1", "CANCEL_REASON": None}, {"ORDER_ID": "2", "CANCEL_REASON": "late"}]

        assert label_blank_members(rows, "Unknown", grouped_columns(
            "SELECT TOP 200 ORDER_ID, CANCEL_REASON FROM SALES.ORDERS")) == (rows, 0)


# ── The top rows ─────────────────────────────────────────────────────────────

class TestTheTopRows:

    def _keep_top(self, rows: list[dict], text: str = "just the top 2") -> list[dict]:
        from core.result_cache import ResultCache
        from core.result_commands import execute_result_command, parse_result_command

        cache = ResultCache(max_sessions=2)
        cache.store("s", rows, question="by warehouse", result_id="source-1")
        outcome = execute_result_command("s", parse_result_command(text), cache=cache, source_result_id="source-1")
        assert outcome.ok, outcome.message
        return outcome.snapshot["rows"]

    def test_are_the_largest(self):
        rows = [{"WAREHOUSE": "A", "INVENTORY_VALUE": 10.0}, {"WAREHOUSE": "B", "INVENTORY_VALUE": 30.0},
                {"WAREHOUSE": "C", "INVENTORY_VALUE": 20.0}]

        assert [row["WAREHOUSE"] for row in self._keep_top(rows)] == ["B", "C"]

    def test_of_a_series_are_its_first_periods(self):
        rows = [{"PERIOD": "2026-01-01", "UNITS_SOLD": 10.0}, {"PERIOD": "2026-02-01", "UNITS_SOLD": 30.0},
                {"PERIOD": "2026-03-01", "UNITS_SOLD": 20.0}]

        assert [row["PERIOD"] for row in self._keep_top(rows)] == ["2026-01-01", "2026-02-01"]

    def test_of_a_quantity_in_several_units_are_not_ranked_across_them(self):
        assert [row["UNT_OF_MSR"] for row in self._keep_top(list(_BY_WAREHOUSE))] == ["FT", "EA"]

    def test_the_first_rows_are_the_first_rows(self):
        rows = [{"WAREHOUSE": "A", "INVENTORY_VALUE": 10.0}, {"WAREHOUSE": "B", "INVENTORY_VALUE": 30.0},
                {"WAREHOUSE": "C", "INVENTORY_VALUE": 20.0}]

        assert [row["WAREHOUSE"] for row in self._keep_top(rows, "show the first 2 rows")] == ["A", "B"]

    def test_of_several_measures_keep_the_rows_order(self):
        rows = [{"CUSTOMER": "A", "ORDER_COUNT": 1, "REVENUE": 30.0}, {"CUSTOMER": "B", "ORDER_COUNT": 2, "REVENUE": 20.0},
                {"CUSTOMER": "D", "ORDER_COUNT": 9, "REVENUE": 1.0}]

        assert [row["CUSTOMER"] for row in self._keep_top(rows)] == ["A", "B"]

    @pytest.mark.parametrize("column, values", [
        ("FISCAL_YEAR", [2024, 2025, 2023]), ("MONTH", ["January", "February", "March"]), ("PRD_KEY", [202401, 202402, 202403]),
    ])
    def test_of_a_series_in_any_period_are_its_first_periods(self, column, values):
        rows = [{column: value, "UNITS_SOLD": float(10 * (index % 2 + 1))} for index, value in enumerate(values)]

        assert [row[column] for row in self._keep_top(rows)] == values[:2]
