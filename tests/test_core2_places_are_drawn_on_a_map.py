"""A grouping whose members are places is drawn on a map.

Asked for: geographic analysis of a business -- by state, by ZIP code. A breakdown by state was a bar per state,
fifty of them, the renderer keeping the largest twenty.

* members that are US states (written out or as codes), Canadian provinces or countries: a map, each region
  shaded for its value, one hue light to dark; a member the map has no region for is named under it;
* ZIP codes: a dot per ZIP code on the states, as large as its value; a ZIP code kept as a number ("2101") is
  "02101"; five-digit codes that are mostly no US ZIP code (German, French postal codes) are no map;
* amounts below zero stay bars, the map on offer: one hue light to dark would draw a loss as a small gain;
* a map whose outline cannot be fetched is drawn as its bars, never "drawing" for ever;
* the grouping's own name decides a code that could be something else: "CA" is California under "State",
  nothing under "Channel code";
* a top 5 stays a ranking, with the map on offer; the reader can ask for it ("on a map");
* the map files are bundled (public domain outlines; ZIP points under their MIT licence), fetched only when a
  map is drawn, and the United States are drawn with Alaska and Hawaii as insets.

Invented data only.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.plan.followup import _DISPLAY
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from tests.test_chart_annotation_language import _build

ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / "static" / "js" / "qb-charts.js"
STATES = ["CA", "NY", "TX", "WA", "FL", "IL", "OH", "GA", "NC", "MI", "AZ", "CO"]
COUNTRIES = ["United States", "Canada", "Mexico", "UK", "Germany", "France", "Japan", "Brazil", "Atlantis"]
ZIPS = ["10001", "94105", "60601", "73301", "98101", "33101", "30301", "80202"]
AS_NUMBERS = [2108, 1002, 6103, 10001, 94105, 60601]          # New England's ZIP codes lose their leading 0
NOT_US = ["00100", "00200", "00300", "00400", "00500", "00600"]
Y2025 = {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}


@pytest.fixture(scope="module")
def shop():
    rng = np.random.default_rng(5)
    con = duckdb.connect()
    n = 60
    customers = pd.DataFrame({"customer_id": range(1, n + 1), "customer_name": [f"Customer {i}" for i in range(1, n + 1)],
                              "state": [STATES[i % len(STATES)] for i in range(n)],
                              "country": [COUNTRIES[i % len(COUNTRIES)] for i in range(n)],
                              "zip_code": [ZIPS[i % len(ZIPS)] for i in range(n)],
                              "ship_zip": [AS_NUMBERS[i % len(AS_NUMBERS)] for i in range(n)],
                              "postal_code": [NOT_US[i % len(NOT_US)] for i in range(n)],
                              "channel_code": rng.choice(["CA", "NY", "TX", "WA"], n)})
    con.register("_c", customers)
    con.execute("CREATE TABLE customers AS SELECT * FROM _c")
    m = 3000
    orders = pd.DataFrame({"order_id": range(1, m + 1), "customer_id": rng.integers(1, n + 1, m),
                           "order_date": (pd.Timestamp("2025-01-01")
                                          + pd.to_timedelta(rng.integers(0, 360, m), unit="D")).date,
                           "amount": np.round(rng.uniform(10, 500, m), 2)})
    orders["margin"] = np.round(orders["amount"] - 300, 2)
    con.register("_o", orders)
    con.execute("CREATE TABLE orders AS SELECT order_id, customer_id, CAST(order_date AS DATE) AS order_date, amount, "
                "margin FROM _o")
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="t", options=BuildOptions(workers=1))
    return con, model


def _chart(shop, plan: dict) -> dict:
    con, model = shop
    answers = [json.dumps({"kind": "query", "intent": "breakdown", "measures": ["amount"], "time": Y2025, **plan})]
    services = Services(model=model, warehouse=DuckDBWarehouse(con), complete=lambda s, t: answers.pop(0),
                        index=MemberIndex(), today=__import__("datetime").date(2026, 6, 15))
    return answer_question("q", services, Session())["chart"]


def test_states_are_a_map_of_the_states_named_as_the_map_names_them(shop):
    chart = _chart(shop, {"group_by": ["customer.state"]})
    assert chart["chart_type"] == "map" and chart["renderable_types"] == ["map", "bar"]
    assert chart["geo"]["map"] == "us_state" and chart["geo"]["file"] == "us-states.json"
    assert chart["geo"]["regions"]["CA"] == "California" and chart["geo"]["regions"]["NY"] == "New York"
    assert chart["title"] == "Amount by state"


def test_a_code_is_a_place_only_under_a_name_that_says_so(shop):
    chart = _chart(shop, {"group_by": ["customer.channel_code"]})
    assert chart["chart_type"] == "bar" and "map" not in chart["renderable_types"]


def test_countries_are_a_world_map_and_one_it_has_not_is_said(shop):
    chart = _chart(shop, {"group_by": ["customer.country"]})
    assert chart["geo"]["map"] == "country" and chart["geo"]["regions"]["UK"] == "United Kingdom"
    assert chart["chart_warnings"] == ["Atlantis is not on the map: every one is in the table."]


def test_zip_codes_are_dots_on_the_states(shop):
    chart = _chart(shop, {"group_by": ["customer.zip_code"]})
    assert chart["chart_type"] == "map" and chart["geo"]["map"] == "us_zip"
    zips = {line.split(",")[0] for line in (ROOT / "static/geo/us-zip.csv").read_text().splitlines()[1:]}
    assert {r[chart["x_key"]] for r in chart["rows"]} <= zips


def test_a_zip_code_kept_as_a_number_is_its_five_digits(shop):
    chart = _chart(shop, {"group_by": ["customer.ship_zip"]})
    assert chart["chart_type"] == "map" and chart["geo"]["map"] == "us_zip" and not chart["chart_warnings"]
    placed = _drawn(chart, "QBCharts._geo.maps.us_zip = true; QBCharts._geo.zips = {'02108': [-71.06, 42.36]};",
                    "o.series[0].data.map(d => d.name)")
    assert placed == ["02108"]


def test_five_digit_codes_that_are_no_us_zip_code_are_no_map(shop):
    chart = _chart(shop, {"group_by": ["customer.postal_code"]})
    assert chart["chart_type"] == "bar" and "map" not in chart["renderable_types"]


def test_a_zip_code_with_no_point_is_named_under_the_map():
    from core2.answer.geo import off_map_zips, place_kind, zip5

    assert zip5("2108") == "02108" and zip5("02108-1234") == "02108" and zip5("SW1A") == "SW1A"
    assert off_map_zips(["10001", "00100", "Unknown"]) == ["00100"]
    assert place_kind("Zip code", "zip_code", ["10001", "94105", "60601", "30301", "00100"]) == "us_zip", "one in five is off"
    assert place_kind("Zip code", "zip_code", ["10001", "94105", "00100", "00200"]) is None


def test_losses_stay_bars_with_the_map_on_offer(shop):
    chart = _chart(shop, {"group_by": ["customer.state"], "measures": ["margin"]})
    assert any(r[chart["y_keys"][0]] < 0 for r in chart["rows"]), "the fixture has states that lost"
    assert chart["chart_type"] == "bar" and "map" in chart["renderable_types"]


def test_a_top_five_stays_a_ranking_with_the_map_on_offer(shop):
    chart = _chart(shop, {"group_by": ["customer.state"], "sort": [{"by": "amount", "desc": True}], "limit": 5,
                          "intent": "rank"})
    assert chart["chart_type"] == "bar" and "map" in chart["renderable_types"]
    assert _chart(shop, {"group_by": ["customer.state"], "chart": "map"})["chart_type"] == "map"
    assert _DISPLAY.search("show that on a map")


# ── drawn ───────────────────────────────────────────────────────────────────


def _drawn(chart: dict, ready: str, expression: str):
    return json.loads(_build("portal_chat.html", "en", chart,
                             f"{ready} var o = QBCharts.buildOption({json.dumps(chart)}); JSON.stringify({expression})"))


def test_the_map_says_it_is_coming_until_its_outline_is_here(shop):
    chart = _chart(shop, {"group_by": ["customer.state"]})
    assert _drawn(chart, "", "o.graphic.style.text") == "Drawing the map…"


def test_a_map_whose_outline_cannot_be_fetched_is_its_bars(shop):
    chart = _chart(shop, {"group_by": ["customer.state"]})
    assert _drawn(chart, "QBCharts._geo.failed.us_state = true;", "o.series[0].type") == "bar"


@pytest.mark.skipif(shutil.which("node") is None, reason="node runs the renderer's fetch")
@pytest.mark.parametrize("answer", ["Promise.resolve({ok: false, status: 404})", "Promise.reject(new Error('offline'))"])
def test_an_outline_that_does_not_come_marks_the_map_failed(shop, answer):
    chart = _chart(shop, {"group_by": ["customer.state"]})
    script = (f"global.window = global; window.fetch = () => {answer}; require({json.dumps(str(RENDERER))});"
              f"window.QBCharts.loadGeo({json.dumps(chart)}).then(() => "
              "console.log(JSON.stringify(window.QBCharts._geo.failed)));")
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=60, check=True).stdout
    assert json.loads(out) == {"us_state": True}


def test_a_region_is_shaded_for_its_value_on_the_projection_with_insets(shop):
    chart = _chart(shop, {"group_by": ["customer.state"]})
    series = _drawn(chart, "QBCharts._geo.maps.us_state = true;",
                    "[o.series[0].type, o.series[0].map, o.series[0].data.map(d => d.name).sort(), "
                    "typeof o.series[0].projection.project, o.visualMap.inRange.color.length]")
    assert series[:2] == ["map", "qb-us_state"]
    assert "California" in series[2] and "CA" not in series[2]
    assert series[3] == "function" and series[4] >= 5
    hawaii, alaska, sf = _drawn(chart, "", "[[-157.8, 21.3], [-149.9, 61.2], [-122.4, 37.8]].map("
                                           "p => QBCharts.albersUsa.project(p).map(Math.round))")
    assert hawaii[1] > sf[1] and alaska[1] > sf[1] and alaska[0] < sf[0] + 100, "both insets below the coast"


def test_a_zip_code_is_a_dot_as_large_as_its_value(shop):
    chart = _chart(shop, {"group_by": ["customer.zip_code"]})
    dots = _drawn(chart, "QBCharts._geo.maps.us_zip = true; QBCharts._geo.zips = "
                         "{'10001': [-74.0, 40.75], '94105': [-122.39, 37.79]};",
                  "[o.series[0].type, o.series[0].coordinateSystem, o.series[0].data.map(d => d.name).sort()]")
    assert dots == ["scatter", "geo", ["10001", "94105"]], "a ZIP code with no point is left off"


def test_the_map_files_are_bundled_with_their_sources():
    geo = ROOT / "static" / "geo"
    states = json.loads((geo / "us-states.json").read_text(encoding="utf-8"))
    assert len(states["features"]) == 51, "fifty states and DC"
    notes = (geo / "LICENSES.md").read_text(encoding="utf-8")
    assert "Natural Earth" in notes and "public domain" in notes.lower() and "MIT License" in notes
    assert sum(p.stat().st_size for p in geo.iterdir()) < 2_000_000
