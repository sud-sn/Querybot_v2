"""A dashboard tile drills into a member, with the way back up above it.

Asked for (October 2026, the interactions): drill down on a tile -- Region, then North by store, then a store by
product -- with a breadcrumb to go back up. A right-click on a member of a tile opens its menu: filter the
dashboard to it (as a click does), or break it down by another field, a level further on that tile. The path is in
the address (d.<tile>=<field>:<member> a level, by.<tile>=<field>), so a drilled tile is drawn by the server, kept
by a refresh and Back, and shared as it is seen.

* the new core says what a member of an answer can be broken down by, as fields (core2.answer.suggestions);
* the address's path, read and written, and a tile's plan a level down (portal/dashboard_view.py);
* the page: the tile drawn a level down for the member drilled through (the dashboard's own filter on that field
  giving way to it), its path back up above it, what the next level can be broken down by, three levels at most;
* the renderer's tooltip, and the bar's next and previous levels (frontend/src/dashboard/view.js).

Invented data only.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from starlette.datastructures import QueryParams

from portal import dashboard_view as dv
from tests.test_a_dashboard_has_one_date_picker_and_its_owners_filters import MONTHLY, _page
from tests.test_a_dashboard_reads_as_the_plan import BY_REGION, TOTAL, _pin
from tests.test_a_new_core_answer_can_be_pinned import _answer, _reader, fresh_store, retail  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


# ── the address and the plan ─────────────────────────────────────────────────

def test_the_path_is_read_from_the_address_in_order_and_written_back():
    params = QueryParams([("d.7", "region.name:North"), ("d.7", "store:Eastview: Main"), ("by.7", "product"),
                          ("d.9", "region.name:West")])
    drills = dv.read_drills(params)
    assert drills == {7: {"path": [{"field": "region.name", "member": "North"},
                                   {"field": "store", "member": "Eastview: Main"}], "by": "product"}}, \
        "a member may hold a colon; a tile with no field to break down by is not drilled"
    assert dv.read_drills(QueryParams(dv.drill_address(drills))) == drills


def test_a_path_holds_three_levels_and_no_field_that_is_not_one():
    deep = [("d.3", f"f{i}:m{i}") for i in range(5)] + [("by.3", "x")]
    assert len(dv.read_drills(QueryParams(deep))[3]["path"]) == dv.DRILL_DEPTH == 3
    assert dv.read_drills(QueryParams([("d.3", "region name:North"), ("by.3", "store")])) == {}
    assert dv.read_drills(QueryParams([("d.x", "region.name:North"), ("by.x", "store")])) == {}
    assert dv.read_drills(QueryParams([("d.3", "region.name:North"), ("by.3", "store; drop")])) == {}


def test_a_tiles_plan_a_level_down_keeps_its_measure_period_and_order():
    plan = {**BY_REGION, "chart": "pie", "sort": [{"by": "net_amount", "desc": True}], "limit": 5}
    down = dv.drill_plan(plan, {"path": [{"field": "region.name", "member": "North"}], "by": "store"})
    assert down["group_by"] == ["store"] and "chart" not in down
    assert down["filters"] == [{"field": "region.name", "op": "in", "values": ["North"]}]
    assert (down["measures"], down["time"], down["sort"], down["limit"]) == (
        plan["measures"], plan["time"], plan["sort"], plan["limit"])
    assert plan["group_by"] == ["region.name"] and "filters" not in plan, "the pinned plan is left as it was"


def test_only_a_measure_for_each_member_of_one_field_is_drilled_into():
    assert dv.drillable(BY_REGION)
    assert not dv.drillable(TOTAL) and not dv.drillable(MONTHLY)
    assert not dv.drillable({**BY_REGION, "group_by": ["region.name", "store"]})
    assert not dv.drillable({**BY_REGION, "pareto": {"of": "store"}})


def _offers(retail, field):  # noqa: F811
    chart = _answer(retail, {**BY_REGION, "group_by": [field]})["chart"]
    return [o["field"] for o in chart.get("breakdowns") or []], chart.get("drill")


def test_a_member_is_never_offered_a_field_it_holds_one_value_of(retail):  # noqa: F811
    # Found in the portal: a store offered "by region", a product "by category", a customer "by segment" -- each
    # one bar, the store's own region. The sales rows reach those through the member; a customer's purchases by
    # store stay, each sale naming its own store.
    store, _ = _offers(retail, "store.name")
    assert store and "region" not in store and "region.name" not in store
    product, _ = _offers(retail, "product.name")
    assert product and "category" not in product and "category.name" not in product
    customer, drill = _offers(retail, "customer.name")
    assert "store" in customer and not {"customer.segment", "customer.city"} & set(customer)
    assert not any(" by Segment" in i["label"] or " by City" in i["label"] for i in (drill or {}).get("items", [])), \
        "the chat's click questions too"
    region, _ = _offers(retail, "region.name")
    assert "store" in region, "a region's stores are its parts"
    segment, _ = _offers(retail, "customer.segment")
    assert segment, "a segment is no member: its customers' other fields split it"


def test_nor_one_the_members_drilled_through_hold_one_value_of(retail):  # noqa: F811
    # Found in the portal: Eastview Store by product offered "region", then "product category" under a product --
    # the third level a table of one row.
    stores = _answer(retail, {**BY_REGION, "group_by": ["store.name"]})["chart"]
    store = stores["rows"][0][stores["x_key"]]
    one = {**BY_REGION, "group_by": ["customer.name"], "filters": [{"field": "store.name", "op": "in", "values": [store]}]}
    chart = _answer(retail, one)["chart"]
    offered = [o["field"] for o in chart.get("breakdowns") or []]
    assert offered and "region" not in offered and "product" in offered
    assert not any("by Region" in i["label"] for i in (chart.get("drill") or {}).get("items", []))
    several = {**one, "filters": [{"field": "store.name", "op": "in", "values": [store, "Elsewhere"]}]}
    assert "region" in [o["field"] for o in _answer(retail, several)["chart"].get("breakdowns") or []], \
        "two stores may lie in two regions"


# ── the page ─────────────────────────────────────────────────────────────────

def _board(store, retail):  # noqa: F811
    user = _reader(store)
    tiles = _pin(store, retail, user, BY_REGION, "net amount by region in april", new_dashboard_name="Mine")
    board = tiles[0]["dashboard_id"]
    _pin(store, retail, user, TOTAL, "net amount in april", dashboard_id=board)
    return user, board, tiles[0]["id"]


def _tiles(store, user, board, retail, pairs=()):  # noqa: F811
    page = _page(store, user, board, retail, [("period", "own"), *pairs])[1]
    return page, {c["title"].split(" for ")[0]: c for c in page["charts"]}


def test_a_member_drilled_into_draws_the_tile_a_level_down_with_its_way_back(fresh_store, retail):  # noqa: F811
    user, board, tile = _board(fresh_store, retail)
    _, top = _tiles(fresh_store, user, board, retail)
    regions = json.loads(top["Net amount by region"]["chart_json"])
    offers = regions["drill_down"]["offers"]
    assert offers and all(o["field"] != "region.name" for o in offers), "what North can be broken down by"
    assert regions["drill_down"]["depth"] == 0
    by = offers[0]["field"]
    north = next(r["net_amount"] for r in regions["rows"] if r["region_name"] == "North")

    page, drilled = _tiles(fresh_store, user, board, retail, [(f"d.{tile}", "region.name:North"), (f"by.{tile}", by)])
    card = next(c for c in page["charts"] if c["id"] == tile)
    shape = json.loads(card["chart_json"])
    value = next(k for k in shape["y_keys"])
    assert sum(r[value] for r in shape["rows"] if r[shape["x_key"]] != (shape.get("other") or {}).get("label")) \
        + sum(r[value] for r in shape["rows"] if r[shape["x_key"]] == (shape.get("other") or {}).get("label")) \
        == pytest.approx(north), "North's own total, broken down"
    assert card["drill"]["top"] == "Region" and [s["member"] for s in card["drill"]["path"]] == ["North"]
    assert "North" not in card["title"] and card["title"].startswith("Net amount by"), \
        "the path is drawn above the tile; its title says what is drawn"
    assert page["dash_filter_state"]["drills"] == {str(tile): {"path": [{"field": "region.name", "member": "North"}],
                                                               "by": by}}
    assert shape["pick"]["field"] == by, "a click on a member of the level drawn filters the dashboard to it"
    assert all(o["field"] not in ("region.name", by) for o in (shape.get("drill_down") or {}).get("offers", [])), \
        "the fields the path went through are not offered again"
    others = [c for c in page["charts"] if c["id"] != tile]
    assert all("drill" not in c for c in others), "the dashboard's other tiles are as they were"


def test_the_dashboards_own_filter_on_the_field_drilled_through_gives_way(fresh_store, retail):  # noqa: F811
    user, board, tile = _board(fresh_store, retail)
    _, top = _tiles(fresh_store, user, board, retail)
    by = json.loads(top["Net amount by region"]["chart_json"])["drill_down"]["offers"][0]["field"]
    pairs = [(f"d.{tile}", "region.name:North"), (f"by.{tile}", by)]
    alone = _tiles(fresh_store, user, board, retail, pairs)[0]
    west = _tiles(fresh_store, user, board, retail, [*pairs, ("f.region.name", "West")])[0]
    card = lambda page: next(c for c in page["charts"] if c["id"] == tile)  # noqa: E731
    assert json.loads(card(west)["chart_json"])["rows"] == json.loads(card(alone)["chart_json"])["rows"]


def test_the_dashboards_filter_bar_is_the_same_while_a_tile_is_drilled(fresh_store, retail):  # noqa: F811
    # Found in the portal: drilling the store tile took "Store" out of the bar (it offers four fields, from the
    # tiles in order) and slid in the next tile's "Product". The bar is the dashboard's, never one tile's level.
    import asyncio
    from unittest.mock import MagicMock, patch

    from portal import routes
    from tests.test_a_new_core_answer_can_be_pinned import _replay

    user, board, tile = _board(fresh_store, retail)
    top = _page(fresh_store, user, board, retail, [("period", "own")])[1]
    by = json.loads(next(c for c in top["charts"] if c["id"] == tile)["chart_json"])["drill_down"]["offers"][0]["field"]
    drilled = _page(fresh_store, user, board, retail, [("period", "own"), (f"d.{tile}", "region.name:North"),
                                                       (f"by.{tile}", by)])[1]
    fields = lambda page: [f["field"] for f in page["dash_filter_state"]["filters"]]  # noqa: E731
    assert fields(drilled) == fields(top) == ["region.name"]
    region = next(f for f in drilled["dash_filter_state"]["filters"] if f["field"] == "region.name")
    assert region["values"] is None and region["label"] == "Region", "its members read when it is opened"

    def values(field):
        with patch("core2.service.portal_replay", _replay(retail)), \
                patch.object(routes, "_get_portal_user", return_value=user):
            response = asyncio.run(routes.dashboard_filter_values_api(MagicMock(), board, field=field, q=""))
        return response.status_code, json.loads(response.body)

    status, found = values("region.name")
    assert status == 200 and "North" in found["values"], "a field a tile is grouped by is the dashboard's own"
    assert values("store.name")[0] == 404, "never a field the dashboard does not show"


def test_a_tile_drilled_three_levels_down_offers_no_fourth_and_a_trend_none(fresh_store, retail):  # noqa: F811
    user, board, tile = _board(fresh_store, retail)
    _pin(fresh_store, retail, user, MONTHLY, "net amount by month", dashboard_id=board)
    _, top = _tiles(fresh_store, user, board, retail)
    assert "drill_down" not in json.loads(top["Net amount by month"]["chart_json"])
    regions = json.loads(top["Net amount by region"]["chart_json"])
    fields = [o["field"] for o in regions["drill_down"]["offers"]]
    page = _page(fresh_store, user, board, retail, [("period", "own"), (f"d.{tile}", "region.name:North"),
                                                    *[(f"d.{tile}", f"{f}:Anything") for f in fields[:2]],
                                                    (f"by.{tile}", fields[2] if len(fields) > 2 else fields[0])])[1]
    card = next(c for c in page["charts"] if c["id"] == tile)
    assert len(card["drill"]["path"]) == 3
    assert "drill_down" not in json.loads(card["chart_json"] or "{}"), "three levels at most"


def test_a_drilled_title_leaves_out_the_members_drilled_through():
    from portal.routes import _without_path

    path = [{"field": "store", "member": "Eastview Store"}, {"field": "product", "member": "Laundry Pods"}]
    assert _without_path("Net Amount by customer, for store Eastview Store, for product Laundry Pods", path) \
        == "Net Amount by customer"
    assert _without_path('Top 5 customers by net amount, for store "Eastview Store", in April', path[:1]) \
        == "Top 5 customers by net amount, in April"
    assert _without_path("Net Amount by customer", path) == "Net Amount by customer"


def test_the_way_back_up_is_drawn_above_the_tile(fresh_store, retail):  # noqa: F811
    user, board, tile = _board(fresh_store, retail)
    _, top = _tiles(fresh_store, user, board, retail)
    by = json.loads(top["Net amount by region"]["chart_json"])["drill_down"]["offers"][0]["field"]
    html = _page(fresh_store, user, board, retail, [("period", "own"), (f"d.{tile}", "region.name:North"),
                                                    (f"by.{tile}", by)])[0]
    assert 'class="tile-drill"' in html and f'data-drill-tile="{tile}" data-drill-keep="0"' in html
    assert 'aria-current="true">North</span>' in html


# ── the renderer and the bar ─────────────────────────────────────────────────

pytest.importorskip("dukpy", reason="the renderer is JavaScript; only executing it shows what it draws")
from tests.test_chart_annotation_language import _build  # noqa: E402

REGIONS = {"chart_type": "bar", "x_key": "region", "y_keys": ["amt"], "chart_id": 7,
           "pick": {"field": "region.name", "label": "Region", "selected": []},
           "drill_down": {"offers": [{"field": "store", "label": "store"}], "depth": 0, "path": []},
           "rows": [{"region": r, "amt": v} for r, v in (("North", 90.0), ("South", 60.0))]}


def test_the_tooltip_says_a_right_click_breaks_a_member_down():
    tip = "JSON.stringify(opt.tooltip.formatter({name: 'North', value: 1, dataIndex: 0, color: '#111'}))"
    assert "Right-click to break North down" in json.loads(_build("portal_dashboard.html", "en", REGIONS, tip))
    assert "Right-click" not in json.loads(_build("portal_dashboard.html", "en", {**REGIONS, "drill_down": None}, tip))
    assert "Clic droit pour détailler North" in json.loads(_build("portal_dashboard.html", "fr", REGIONS, tip))


@pytest.mark.skipif(shutil.which("node") is None, reason="Node runs the bar's own modules")
def test_the_bar_goes_a_level_down_and_back_up_and_writes_it_in_the_address():
    module = (ROOT / "frontend" / "src" / "dashboard" / "view.js").as_uri()
    script = f"""import {{ address, drilledInto, drilledUp }} from '{module}';
      const one = drilledInto(undefined, {{field: 'region.name', member: 'North'}}, 'store');
      const two = drilledInto(one, {{field: 'store', member: 'Eastview'}}, 'product');
      const state = {{dashboard_id: 4, tab: 'Overview', picker: {{period: 'this_month', default: 'this_month'}},
                     filters: [], drills: {{'12': two}}}};
      console.log(JSON.stringify([one, two, drilledUp(two, 1), drilledUp(two, 0), address(state)]))"""
    run = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    one, two, up, top, url = json.loads(run.stdout)
    assert one == {"path": [{"field": "region.name", "member": "North"}], "by": "store"}
    assert two["path"][-1] == {"field": "store", "member": "Eastview"} and two["by"] == "product"
    assert up == one and top is None
    from urllib.parse import parse_qsl, urlsplit
    assert dv.read_drills(QueryParams(urlsplit(url).query))[12] == two
    assert [k for k, _ in parse_qsl(urlsplit(url).query)].count("d.12") == 2


def test_the_bar_answers_the_tiles_menu_and_its_way_back_up():
    bar = (ROOT / "frontend" / "src" / "dashboard" / "FilterBar.vue").read_text(encoding="utf-8")
    assert "addEventListener('qb-menu'" in bar and "[data-drill-keep]" in bar and "<TileMenu" in bar
