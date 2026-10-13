"""A click on a member of a dashboard tile filters the dashboard to it.

Asked for (October 2026, the interactions after the look): clicking a bar -- "North" -- filters every other tile
to North, as choosing North in the filter bar does. The tile clicked keeps all its members, the chosen ones
standing out and the others stepping back, so another can be picked; a member clicked again clears the filter;
Ctrl, Cmd or Shift adds a member to those chosen (or takes it out). "Other (n)" is the rest, not a member.

* the server: a tile grouped by one field (not by time) says which field a click on it filters by, and which of
  its members the filter chose;
* the renderer (static/js/qb-charts.js): what a click asks for (pickOf), the members not chosen stepping back, and
  a tooltip that says what a click does;
* the bar (frontend/src/dashboard/view.js): the members a click leaves chosen.

Invented data only.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.test_a_dashboard_has_one_date_picker_and_its_owners_filters import MONTHLY, _page
from tests.test_a_dashboard_reads_as_the_plan import BY_REGION, TOTAL, _pin
from tests.test_a_new_core_answer_can_be_pinned import _reader, fresh_store, retail  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


# ── the server ───────────────────────────────────────────────────────────────

def test_a_tile_grouped_by_one_field_says_what_a_click_filters_by_and_keeps_every_member(fresh_store, retail):  # noqa: F811
    user = _reader(fresh_store)
    tiles = _pin(fresh_store, retail, user, BY_REGION, "net amount by region in april", new_dashboard_name="Mine")
    board = tiles[0]["dashboard_id"]
    _pin(fresh_store, retail, user, TOTAL, "net amount in april", dashboard_id=board)
    _pin(fresh_store, retail, user, MONTHLY, "net amount by month", dashboard_id=board)

    def tiles(pairs):
        page = _page(fresh_store, user, board, retail, [("period", "own"), *pairs])[1]
        return {c["title"]: c for c in page["charts"]}

    drawn = tiles([])
    regions = json.loads(drawn["Net amount by region"]["chart_json"])
    assert regions["pick"] == {"field": "region.name", "label": "Region", "selected": []}
    assert "pick" not in json.loads(drawn["Net amount by month"]["chart_json"]), "a chart over time is not filtered by a click"

    north = tiles([("f.region.name", "North")])
    picked = json.loads(north["Net amount by region"]["chart_json"])
    assert picked["pick"]["selected"] == ["North"]
    assert {r["region_name"] for r in picked["rows"]} == {r["region_name"] for r in regions["rows"]}, \
        "the tile clicked keeps every member: another can still be picked"
    assert north["Net amount"]["kpi"]["value"] == pytest.approx(
        next(r["net_amount"] for r in regions["rows"] if r["region_name"] == "North")), "the other tiles follow"


# ── the renderer ─────────────────────────────────────────────────────────────

pytest.importorskip("dukpy", reason="the renderer is JavaScript; only executing it shows what it draws")
from tests.test_chart_annotation_language import _build  # noqa: E402

PICK = {"field": "region.name", "label": "Region", "selected": []}
REGIONS = {"chart_type": "bar", "x_key": "region", "y_keys": ["amt"], "chart_id": 7, "pick": PICK,
           "column_formats": {"amt": "currency"},
           "rows": [{"region": r, "amt": v} for r, v in (("North", 90.0), ("South", 60.0), ("Other (3)", 20.0))],
           "other": {"label": "Other (3)", "count": 3}}


def run(payload, expression):
    return json.loads(_build("portal_dashboard.html", "en", payload, f"JSON.stringify({expression})"))


def test_a_click_on_a_member_asks_for_it_and_a_held_key_adds_it():
    asked = run(REGIONS, "QBCharts.pickOf({componentType: 'series', name: 'North', event: {event: {}}}, "
                         + json.dumps(REGIONS) + ")")
    assert asked == {"field": "region.name", "label": "Region", "member": "North", "add": False, "chart": 7}
    held = run(REGIONS, "QBCharts.pickOf({componentType: 'series', name: 'South', event: {event: {metaKey: true}}}, "
                        + json.dumps(REGIONS) + ")")
    assert held["add"] is True and held["member"] == "South"


def test_the_rest_of_the_members_a_tile_no_click_filters_and_a_title_ask_for_nothing():
    for params, payload in (({"componentType": "series", "name": "Other (3)"}, REGIONS),
                            ({"componentType": "series", "name": "North"}, {**REGIONS, "pick": None}),
                            ({"componentType": "title", "name": "North"}, REGIONS)):
        assert run(payload, f"QBCharts.pickOf({json.dumps(params)}, {json.dumps(payload)})") is None


def test_a_dumbbells_dot_and_a_maps_region_name_their_member():
    dot = {"componentType": "series", "seriesType": "scatter", "value": [12.0, "North"], "name": ""}
    region = {"componentType": "series", "name": "Ontario", "data": {"name": "Ontario", "member": "ON"}}
    assert run(REGIONS, f"QBCharts.pickOf({json.dumps(dot)}, {json.dumps(REGIONS)}).member") == "North"
    assert run(REGIONS, f"QBCharts.pickOf({json.dumps(region)}, {json.dumps(REGIONS)}).member") == "ON"


def test_the_members_not_chosen_step_back_and_the_tooltip_says_what_a_click_does():
    chosen = {**REGIONS, "pick": {**PICK, "selected": ["North"]}}
    data = run(chosen, "opt.series[0].data.map(function (d) { return d && d.itemStyle ? d.itemStyle.opacity || 1 : 1; })")
    assert data == [1, 0.3, 0.3], "North chosen; South and the rest step back"
    tip = "opt.tooltip.formatter({name: %s, value: 1, dataIndex: 0, color: '#111'})"
    assert "Click to clear this filter" in run(chosen, tip % "'North'")
    assert "Click to filter the dashboard to South" in run(chosen, tip % "'South'")
    assert "Click to" not in run(chosen, tip % "'Other (3)'")
    assert "Click to" not in run({**REGIONS, "pick": None}, tip % "'North'"), "a chart in the chat says nothing"
    assert run(REGIONS, "opt.series[0].data.filter(function (d) { return d && d.itemStyle && d.itemStyle.opacity; }).length") == 0


def test_a_slice_and_a_stacked_bar_not_chosen_step_back_too():
    pie = {**REGIONS, "chart_type": "donut", "rows": REGIONS["rows"][:2], "other": None,
           "pick": {**PICK, "selected": ["South"]}}
    assert run(pie, "opt.series[0].data.map(function (d) { return d.itemStyle.opacity || 1; })") == [0.3, 1]
    stacked = {"chart_type": "stacked", "x_key": "region", "y_keys": ["Retail", "Online"], "pick": {**PICK, "selected": ["West"]},
               "rows": [{"region": r, "Retail": 1.0, "Online": 2.0} for r in ("North", "West")]}
    assert run(stacked, "opt.series.map(function (s) { return s.data.map(function (d) { return typeof d === 'object' ? d.itemStyle.opacity : 1; }); })") \
        == [[0.3, 1], [0.3, 1]]


def test_french_says_it_in_french():
    tip = json.loads(_build("portal_dashboard.html", "fr", {**REGIONS, "pick": {**PICK, "selected": ["North"]}},
                            "JSON.stringify(opt.tooltip.formatter({name: 'South', value: 1, dataIndex: 1, color: '#111'}))"))
    assert "Cliquez pour filtrer le tableau de bord sur South" in tip


# ── the bar ──────────────────────────────────────────────────────────────────

@pytest.mark.skipif(shutil.which("node") is None, reason="Node runs the bar's own modules")
def test_the_members_a_click_leaves_chosen():
    module = (ROOT / "frontend" / "src" / "dashboard" / "view.js").as_uri()
    cases = [([], "North", False), (["North"], "North", False), (["North", "West"], "North", False),
             (["North"], "West", True), (["North", "West"], "West", True)]
    script = (f"import {{ picked }} from '{module}';"
              f"console.log(JSON.stringify({json.dumps(cases)}.map(([now, m, add]) => picked(now, m, add))))")
    run_ = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, timeout=60)
    assert run_.returncode == 0, run_.stderr
    assert json.loads(run_.stdout) == [["North"], [], ["North"], ["North", "West"], ["North"]]


def test_the_bar_listens_for_a_click_on_a_tile():
    bar = (ROOT / "frontend" / "src" / "dashboard" / "FilterBar.vue").read_text(encoding="utf-8")
    assert "addEventListener('qb-pick'" in bar and "removeEventListener('qb-pick'" in bar
