"""A chart, and the dashboard tile pinned from it, read as the answer they come from.

Found reading the new core's charts and dashboards (October 2026):

* every tile was called by its measure alone: a dashboard of four tiles read "Net amount" four times;
* a grouping whose members are amounts (0 to 7 transit days) was drawn largest first, so its spread could
  not be seen; past 20 such members the chart kept the 20 largest and lost the order altogether;
* past 20 members the chart kept the 20 largest and dropped the rest of the whole without a trace: the
  bars of 241 customers did not add up to the total the sentence gave;
* a line per member stopped at eight lines, a tangle, and the rest were gone;
* a KPI said "$185,933" and nothing of whether that is good: not the period before, not the trend;
* several numbers of one answer were pinned as a one-row table.

Invented data only.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from tests.test_a_new_core_answer_can_be_pinned import _reader, _replay, fresh_store, retail  # noqa: F401
from tests.test_chart_annotation_language import _build
from tests.test_dashboard_page import _chart as _tile, _render

TODAY = dt.date(2026, 6, 15)
APRIL = {"kind": "between", "start": "2026-04-01", "end": "2026-04-30"}
MARCH = {"kind": "between", "start": "2026-03-01", "end": "2026-03-31"}
H1 = {"kind": "between", "start": "2026-01-01", "end": "2026-06-30"}
TRANSIT = {"Courier": 0, "Overnight": 1, "Two Day": 2, "Ground": 3, "Saver": 4, "Economy": 5, "Freight": 6,
           "Slow Boat": 7}


def _ask(retail, plan: dict) -> dict:  # noqa: F811
    con, model = retail
    answers = [json.dumps({"kind": "query", **plan})]
    services = Services(model=model, warehouse=DuckDBWarehouse(con), complete=lambda s, t: answers.pop(0),
                        index=MemberIndex(), today=TODAY)
    return answer_question("q", services, Session())


@pytest.fixture(scope="module")
def shipping():
    """Ship methods that each take their own number of days, and the shipments sent by them."""
    rng = np.random.default_rng(11)
    con = duckdb.connect()
    methods = pd.DataFrame({"ship_method_id": range(1, 9), "service_name": list(TRANSIT),
                            "typical_transit_days": list(TRANSIT.values())})
    con.register("_m", methods)
    con.execute("CREATE TABLE ship_methods AS SELECT * FROM _m")
    # One clinic sends most: its smallest few added up are less than it, so they can be one bar beside it.
    weights = np.array([40, 12, 10, 9, 8, 6, 5, 4, 3, 2, 1.5, 1, 0.8, 0.7, 0.5])
    clinics = pd.DataFrame({"clinic_id": range(1, 16), "clinic_name": [f"Clinic {c}" for c in "ABCDEFGHIJKLMNO"]})
    con.register("_c", clinics)
    con.execute("CREATE TABLE clinics AS SELECT * FROM _c")
    n = 2400
    # The fastest are the most used: largest first would read 1, 0, 2 ..., never 0 to 7.
    method = rng.choice(np.arange(1, 9), n, p=[0.1, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05, 0.02])
    shipments = pd.DataFrame({
        "shipment_id": np.arange(1, n + 1), "ship_method_id": method,
        "clinic_id": rng.choice(np.arange(1, 16), n, p=weights / weights.sum()),
        "ship_date": (pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 360, n), unit="D")).date,
        "shipping_cost": np.round(rng.uniform(4, 60, n), 2)})
    con.register("_s", shipments)
    con.execute("CREATE TABLE shipments AS SELECT shipment_id, ship_method_id, clinic_id, CAST(ship_date AS DATE) "
                "AS ship_date, shipping_cost FROM _s")
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="t", options=BuildOptions(workers=1))
    days = next(a.slug for a in model.attributes.values() if model.columns[a.column].name == "typical_transit_days")
    return con, model, days


def _ship(shipping, plan: dict) -> dict:
    con, model, _ = shipping
    return _ask((con, model), plan)["chart"]


def _all_rows(retail, payload: dict) -> list[dict]:  # noqa: F811
    """Every row of the answer (its table shows the first 200), from its own query."""
    con, _ = retail
    result = con.execute(payload["trust"]["sql"])
    names = [d[0] for d in result.description]
    return [{k: float(v) if isinstance(v, (int, float)) or type(v).__name__ == "Decimal" else v
             for k, v in zip(names, row)} for row in result.fetchall()]


def _labels(chart: dict) -> list[str]:
    return [str(r[chart["x_key"]]) for r in chart["rows"]]


# ── what a chart is called ──────────────────────────────────────────────────


def test_a_chart_is_named_by_what_it_counts_and_by_what(retail):  # noqa: F811
    assert _ask(retail, {"intent": "breakdown", "measures": ["net_amount"], "group_by": ["store"],
                         "time": {"window": H1}})["chart"]["title"] == "Net amount by store"
    assert _ask(retail, {"intent": "trend", "measures": ["net_amount"],
                         "time": {"grain": "month", "window": H1}})["chart"]["title"] == "Net amount by month"
    assert _ask(retail, {"intent": "trend", "measures": ["net_amount"], "group_by": ["store"],
                         "time": {"grain": "month", "window": H1}})["chart"]["title"] == \
        "Net amount by store and month"


def test_a_title_names_the_conditions_and_never_the_period(retail):  # noqa: F811
    chart = _ask(retail, {"intent": "breakdown", "measures": ["net_amount"], "group_by": ["store"],
                          "filters": [{"field": "customer.segment", "op": "eq", "values": ["Retail"]}],
                          "time": {"window": H1}})["chart"]
    # The period moves on with a window like "last month" while a pinned title stays: the subtitle says it.
    assert chart["title"] == "Net amount by store, for segment Retail"


def test_a_kpi_carries_the_title_its_tile_is_pinned_under(retail):  # noqa: F811
    payload = _ask(retail, {"intent": "value", "measures": ["net_amount"], "time": {"window": APRIL},
                            "filters": [{"field": "customer.segment", "op": "eq", "values": ["Retail"]}]})
    assert payload["kpi"]["title"] == "Net amount, for segment Retail"


# ── members that are amounts ────────────────────────────────────────────────


def test_members_that_are_amounts_are_drawn_in_their_own_order(shipping):
    _, model, days = shipping
    payload = _ask(shipping[:2], {"intent": "breakdown", "measures": ["number_of_shipments"], "group_by": [days]})
    chart = payload["chart"]
    assert "8 typical transit days." in payload["answer"]["headline"], "never \"dayses\""
    assert model.attributes[days].kind == "number"
    assert _labels(chart) == [str(d) for d in range(8)], "0 to 7 days, the spread, not a ranking"
    assert chart["x_order"] == "number"
    counts = [r["number_of_shipments"] for r in chart["rows"]]
    assert counts != sorted(counts, reverse=True), "the invented data is not already in number order by size"


def test_a_ranking_the_reader_asked_for_is_kept(shipping):
    _, _, days = shipping
    chart = _ship(shipping, {"intent": "rank", "measures": ["number_of_shipments"], "group_by": [days],
                             "sort": [{"by": "number_of_shipments", "desc": True}]})
    counts = [r["number_of_shipments"] for r in chart["rows"]]
    assert counts == sorted(counts, reverse=True) and chart["x_order"] == ""


def test_many_amounts_are_drawn_as_equal_ranges_that_add_up(retail):  # noqa: F811
    payload = _ask(retail, {"intent": "breakdown", "measures": ["number_of_order_lines"],
                            "group_by": ["product.list_price"]})
    chart, rows = payload["chart"], payload["data"]["rows"]
    labels = _labels(chart)
    assert len(rows) > 20 and len(chart["rows"]) <= 12
    ranges = [tuple(float(v) for v in label.split("–")) for label in labels if label != "Unknown"]
    assert all(b - a == ranges[0][1] - ranges[0][0] for a, b in ranges), "every range as wide as the others"
    assert all(ranges[i][1] == ranges[i + 1][0] for i in range(len(ranges) - 1)), "no gap between ranges"
    drawn = sum(r["number_of_order_lines"] for r in chart["rows"])
    assert drawn == sum(r["number_of_order_lines"] for r in rows), "the ranges add up to the whole"
    assert any("ranges of" in w for w in chart["chart_warnings"])
    assert not chart.get("drill"), "a range is no member a click could open"


# ── the rest of the whole ───────────────────────────────────────────────────


Y2025 = {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}


def test_many_members_are_the_ten_largest_and_the_rest_as_one_bar(shipping):
    con, _, _ = shipping
    payload = _ask(shipping[:2], {"intent": "breakdown", "measures": ["number_of_shipments"],
                                  "group_by": ["clinic.name"], "time": Y2025})
    chart, rows = payload["chart"], payload["data"]["rows"]
    x = chart["x_key"]
    assert len(rows) == 15 and len(chart["rows"]) == 11
    other = chart["rows"][-1]
    assert other[x] == "Other (5)" and chart["other"] == {"label": "Other (5)", "count": 5}
    largest = sorted(rows, key=lambda r: -r["number_of_shipments"])
    assert {r[x] for r in chart["rows"][:-1]} == {r[x] for r in largest[:10]}
    assert other["number_of_shipments"] == sum(r["number_of_shipments"] for r in largest[10:])
    assert sum(r["number_of_shipments"] for r in chart["rows"]) == sum(r["number_of_shipments"] for r in rows)
    assert not chart["chart_warnings"]


def test_a_rest_that_would_dwarf_the_bars_is_said_under_them(retail):  # noqa: F811
    payload = _ask(retail, {"intent": "breakdown", "measures": ["net_amount"], "group_by": ["customer.name"],
                            "time": {"window": H1}})
    chart, rows = payload["chart"], _all_rows(retail, payload)
    largest = sorted(rows, key=lambda r: -r["net_amount"])
    rest = sum(r["net_amount"] for r in largest[10:])
    assert len(chart["rows"]) == 10 and chart["other"] is None, "230 customers added up: twenty times the largest"
    share = rest / sum(r["net_amount"] for r in rows) * 100
    assert chart["chart_warnings"] == [f"Showing the 10 largest: the other {len(rows) - 10} customers add up to "
                                       f"${rest:,.2f}, {share:.0f}% of the total. Every one is in the table."]


def test_an_average_is_never_added_up_into_one_bar(retail):  # noqa: F811
    payload = _ask(retail, {"intent": "breakdown", "measures": ["unit_price"], "group_by": ["customer.name"],
                            "time": {"window": H1}})
    assert payload["chart"]["other"] is None and len(payload["chart"]["rows"]) == payload["data"]["total_rows"]


def test_a_line_per_member_keeps_five_and_adds_up_the_rest(shipping):
    payload = _ask(shipping[:2], {"intent": "trend", "measures": ["number_of_shipments"], "group_by": ["clinic.name"],
                                  "time": {"grain": "month", **Y2025}})
    chart, rows = payload["chart"], payload["data"]["rows"]
    assert len(chart["y_keys"]) == 6 and chart["y_keys"][-1] == "Other (10)" == chart["other"]["label"]
    kept = set(chart["y_keys"][:-1])
    clinic = next(h for h in payload["data"]["headers"] if h not in ("period", "number_of_shipments"))
    for point in chart["rows"]:
        rest = sum(r["number_of_shipments"] for r in rows if r["period"] == point["period"] and r[clinic] not in kept)
        assert point["Other (10)"] == rest


def test_lines_the_rest_would_flatten_are_the_six_largest(retail):  # noqa: F811
    chart = _ask(retail, {"intent": "trend", "measures": ["net_amount"], "group_by": ["store"],
                          "time": {"grain": "month", "window": H1}})["chart"]
    assert len(chart["y_keys"]) == 6 and chart["other"] is None
    assert chart["chart_warnings"] == ["Showing the 6 largest of 12 stores: every one is in the table."]


def test_the_rest_is_drawn_in_a_neutral_ink_and_opens_nothing(shipping):
    chart = _ask(shipping[:2], {"intent": "breakdown", "measures": ["number_of_shipments"],
                                "group_by": ["clinic.name"], "time": Y2025})["chart"]
    bars = json.loads(_build("portal_chat.html", "en", chart, "JSON.stringify(opt.series[0].data)"))
    hues = json.loads(_build("portal_chat.html", "en", chart, "JSON.stringify(opt.color)"))
    assert isinstance(bars[-1], dict) and bars[-1]["itemStyle"]["color"] not in hues, "not a series hue"
    assert not any(isinstance(b, dict) and b.get("itemStyle", {}).get("color") == bars[-1]["itemStyle"]["color"]
                   for b in bars[:-1])
    assert chart.get("drill"), "the clinics' own bars open"
    page = (Path(__file__).resolve().parents[1] / "portal" / "templates" / "portal_chat.html").read_text()
    assert "if (other && (raw === other || String((params && params.seriesName) || '') === other)) return [];" in page


def test_a_capped_chart_of_amounts_keeps_their_order():
    rows = [{"days": str(d), "n": float((d * 7) % 13 + 1)} for d in range(30)]
    payload = {"rows": rows, "x_key": "days", "y_keys": ["n"], "chart_type": "bar", "x_order": "number"}
    shown = json.loads(_build("portal_chat.html", "en", payload,
                              "JSON.stringify((opt.xAxis.data || opt.yAxis.data).map(Number))"))
    assert len(shown) == 20 and shown == sorted(shown)


# ── a KPI against the period before, and its trend ─────────────────────────


def test_a_kpi_says_how_it_moved_against_the_period_before(retail):  # noqa: F811
    april = _ask(retail, {"intent": "value", "measures": ["net_amount"], "time": {"window": APRIL}})["kpi"]
    march = _ask(retail, {"intent": "value", "measures": ["net_amount"], "time": {"window": MARCH}})["kpi"]
    change = april["change"]
    by = april["value"] - march["value"]
    assert change["direction"] == ("up" if by > 0 else "down") and change["vs"] == "March 2026"
    assert change["change"] == ("+" if by > 0 else "-") + f"${abs(by):,.2f}"
    assert change["pct"] == f"{by / march['value'] * 100:+.1f}%" and change["before"] == f"${march['value']:,.2f}"


def test_a_percentage_moves_by_points(retail):  # noqa: F811
    change = _ask(retail, {"intent": "value", "measures": ["margin_percent"], "time": {"window": APRIL}})["kpi"]["change"]
    assert change["pct"] == "" and (change["change"] == "unchanged" or change["change"].endswith(" pts"))


def test_a_kpi_draws_the_twelve_months_up_to_its_own(retail):  # noqa: F811
    trend = _ask(retail, {"intent": "value", "measures": ["net_amount"], "time": {"window": APRIL}})["kpi"]["trend"]
    assert trend["grain"] == "month" and trend["span"] == "May 2025 – Apr 2026" and not trend["tail"]
    points = re.findall(r"[ML](\d+\.\d),(\d+\.\d)", trend["path"])
    assert len(points) == 12 and float(points[0][0]) < float(points[-1][0])
    assert all(0 <= float(y) <= 32 for _, y in points)


def test_a_month_not_over_yet_is_drawn_dashed(retail):  # noqa: F811
    trend = _ask(retail, {"intent": "value", "measures": ["net_amount"],
                          "time": {"window": {"kind": "this", "unit": "month"}}})["kpi"]["trend"]
    assert trend["span"].endswith("Jun 2026") and trend["tail"], "June to the 15th is no whole month"


def test_a_total_over_all_the_data_has_neither(retail):  # noqa: F811
    kpi = _ask(retail, {"intent": "value", "measures": ["net_amount"]})["kpi"]
    assert "change" not in kpi and "trend" not in kpi


def test_several_numbers_of_one_answer_are_a_group(retail):  # noqa: F811
    kpi = _ask(retail, {"intent": "value", "measures": ["net_amount", "cost_amount"],
                        "time": {"window": APRIL}})["kpi"]
    assert [g["label"] for g in kpi["group"]] == ["Net amount", "Cost amount"]
    assert kpi["title"] == "Net amount and cost amount"


# ── on the dashboard ────────────────────────────────────────────────────────


def _pin(fresh_store, retail, plan: dict, title: str) -> dict:  # noqa: F811
    """The bridge's token, the pin API, the stored chart, its tile: end to end, under ``title``."""
    import asyncio
    from unittest.mock import MagicMock, patch

    from gateway import core2_bridge
    from portal import routes

    user = _reader(fresh_store)
    token = core2_bridge._pin(user["account_id"], user, "a question", _ask(retail, plan))
    request = MagicMock()
    request.json = MagicMock(return_value=asyncio.sleep(0, result={
        "token": token, "title": title, "new_dashboard_name": "Mine"}))
    with patch.object(routes, "_get_portal_user", return_value=user):
        assert json.loads(asyncio.run(routes.pin_chart_api(request)).body)["ok"] is True
    chart = next(c for c in fresh_store.list_pinned_charts(user["id"]) if c["title"] == title)
    with patch("core2.service.portal_replay", _replay(retail)):
        return routes._refresh_chart(chart, {"db_type": "duckdb"}, user)


BY_STORE = {"intent": "breakdown", "measures": ["net_amount"], "group_by": ["store"], "time": {"window": APRIL}}


def test_a_tile_pinned_under_its_measure_alone_reads_as_its_answer(fresh_store, retail):  # noqa: F811
    assert _pin(fresh_store, retail, BY_STORE, "Net amount")["title"] == "Net amount by store"


def test_a_title_the_reader_wrote_stays(fresh_store, retail):  # noqa: F811
    assert _pin(fresh_store, retail, BY_STORE, "Store sales")["title"] == "Store sales"


def test_a_kpi_tile_shows_its_change_and_trend(fresh_store, retail):  # noqa: F811
    tile = _pin(fresh_store, retail, {"intent": "value", "measures": ["net_amount"], "time": {"window": APRIL}},
                "Net amount")
    assert tile["kpi"]["change"]["vs"] == "March 2026" and tile["kpi"]["trend"]["path"]
    markup = _render([_tile(**{k: tile[k] for k in ("kpi", "kpi_display")}, chart_type="kpi")])
    assert "dash-kpi-change" in markup and "vs March 2026" in markup and ("▼" in markup or "▲" in markup)
    assert 'class="qb-graphic dash-kpi-spark"' in markup and tile["kpi"]["trend"]["path"] in markup


def test_several_numbers_are_one_tile_side_by_side(fresh_store, retail):  # noqa: F811
    tile = _pin(fresh_store, retail, {"intent": "value", "measures": ["net_amount", "cost_amount"],
                                      "time": {"window": APRIL}}, "Two numbers")
    assert [g["label"] for g in tile["kpi_group"]] == ["Net amount", "Cost amount"]
    assert all(g["display"].startswith("$") for g in tile["kpi_group"])
    markup = _render([_tile(kpi=tile["kpi"], kpi_display=tile["kpi_display"], kpi_group=tile["kpi_group"],
                            chart_type="kpi")])
    assert markup.count("dash-kpi-cell") == 2


def test_a_tile_of_several_numbers_gets_a_share_of_its_row_for_each():
    from store.dashboard_store import _packed

    def kpi(i, measures):
        return {"id": i, "chart_type": "kpi", "display_config": json.dumps({"core2_plan": {"measures": measures}})}
    rects = _packed([kpi(1, ["a"]), kpi(2, ["a", "b", "c"]), kpi(3, ["a", "b"])])
    assert rects[0] == (0, 0, 3, 2) and rects[1] == (3, 0, 9, 2), "one number and three share a row of four"
    assert rects[2] == (0, 2, 12, 2), "the next two numbers start a row of their own"
