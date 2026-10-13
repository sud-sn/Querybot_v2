"""A dashboard tile is edited in its own panel: its name, kind, colours, and how many members it shows in what order.

Asked for (October 2026, the interactions): a chart editor on dashboard tiles. In Edit, a tile's settings button
opens a panel at the window's edge (frontend/src/dashboard/ChartEditor.vue) with a preview of the tile's own rows;
Save keeps the change as a version of the dashboard and draws it again in place.

* how many a chart of members shows (top 5, 10, 20 or all) and in what order (largest first, smallest first, by
  name) are its plan's, so every refresh, schedule and email draws it so; only those values, only on a chart of
  members, only by whoever may change the dashboard;
* the colours are one of the sets a reader may choose;
* the page tells the panel what the tile may change; a tile drilled into is edited at its top;
* the preview orders and cuts the tile's own rows as asked.

Invented data only.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.test_a_dashboard_has_one_date_picker_and_its_owners_filters import _owner, _page
from tests.test_a_dashboard_reads_as_the_plan import BY_REGION, _pin
from tests.test_a_new_core_answer_can_be_pinned import _reader, _replay, fresh_store, retail  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


def _board(store, retail):  # noqa: F811
    user = _reader(store)
    tiles = _pin(store, retail, user, BY_REGION, "net amount by region in april", new_dashboard_name="Mine")
    return user, tiles[0]["dashboard_id"], tiles[0]["id"]


def _save(user, board, tile, retail, **changes):  # noqa: F811
    from portal import routes

    request = MagicMock()

    async def body():
        return {"dashboard_id": board, "chart_id": tile, **changes}
    request.json = body
    with patch("core2.service.portal_replay", _replay(retail)), patch.object(routes, "_get_portal_user", return_value=user):
        response = asyncio.run(routes.update_chart_api(request))
    return response.status_code, json.loads(response.body)


def _plan(store, user, board, tile):
    chart = next(c for c in store.list_dashboard_charts_for_view(board, user["id"], user["account_id"]) if c["id"] == tile)
    return json.loads(chart["display_config"])["core2_plan"]


def _drawn(store, user, board, tile, retail):  # noqa: F811
    page = _page(store, user, board, retail, [("period", "own")])[1]
    card = next(c for c in page["charts"] if c["id"] == tile)
    return card, json.loads(card["chart_json"])


def test_how_many_and_in_what_order_are_kept_with_the_tile_s_plan(fresh_store, retail):  # noqa: F811
    user, board, tile = _board(fresh_store, retail)
    card, shape = _drawn(fresh_store, user, board, tile, retail)
    assert card["edit"] == {"limit": "", "order": "high"}, "every member, largest first"
    regions = [r[shape["x_key"]] for r in shape["rows"]]

    assert _save(user, board, tile, retail, order="name") == (200, {"ok": True})
    assert _plan(fresh_store, user, board, tile)["sort"] == [{"by": "region.name", "desc": False}]
    card, shape = _drawn(fresh_store, user, board, tile, retail)
    assert [r[shape["x_key"]] for r in shape["rows"]] == sorted(regions) and card["edit"]["order"] == "name"

    assert _save(user, board, tile, retail, order="low", limit="5")[0] == 200
    plan = _plan(fresh_store, user, board, tile)
    assert plan["limit"] == 5 and plan["sort"] == [{"by": "net_amount", "desc": False}]
    card, shape = _drawn(fresh_store, user, board, tile, retail)
    values = [r["net_amount"] for r in shape["rows"]]
    assert values == sorted(values) and card["edit"] == {"limit": 5, "order": "low"}

    assert _save(user, board, tile, retail, limit=None)[0] == 200
    assert "limit" not in _plan(fresh_store, user, board, tile), "all of them again"
    versions = fresh_store.list_dashboard_versions(board, user["id"], user["account_id"])
    assert sum(v["change_summary"] == "Visual updated" for v in versions) >= 3, "each a version of the dashboard"


def test_only_what_the_panel_offers_is_kept(fresh_store, retail):  # noqa: F811
    user, board, tile = _board(fresh_store, retail)
    before = _plan(fresh_store, user, board, tile)
    for changes in ({"limit": "7"}, {"limit": "100000"}, {"order": "random"}, {"color_palette": "sunset"}):
        assert _save(user, board, tile, retail, **changes)[0] == 400, changes
    assert _plan(fresh_store, user, board, tile) == before
    assert _save(user, board, tile, retail, color_palette="ocean")[0] == 200
    assert _save(_owner(fresh_store), board, tile, retail, order="name")[0] in (400, 403), "its owner alone"
    assert _plan(fresh_store, user, board, tile).get("sort", []) == before.get("sort", [])


def test_a_number_has_no_order_to_change(fresh_store, retail):  # noqa: F811
    from tests.test_a_dashboard_reads_as_the_plan import TOTAL

    user, board, _ = _board(fresh_store, retail)
    number = _pin(fresh_store, retail, user, TOTAL, "net amount in april", dashboard_id=board)[0]["id"]
    assert _save(user, board, number, retail, order="low")[0] == 400


def test_the_page_tells_the_panel_what_the_tile_may_change(fresh_store, retail):  # noqa: F811
    user, board, tile = _board(fresh_store, retail)
    html = _page(fresh_store, user, board, retail, [("period", "own")])[0]
    assert f'data-edit-tile="{tile}"' in html and 'data-edit-order="high"' in html and 'data-edit-palette="' in html
    drilled = _page(fresh_store, user, board, retail, [("period", "own"), (f"d.{tile}", "region.name:North"),
                                                      (f"by.{tile}", "store")])[0]
    assert f'data-edit-tile="{tile}"' not in drilled, "a tile drilled into is edited at its top"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node runs the bar's own modules")
def test_the_preview_orders_and_cuts_the_tile_s_own_rows():
    module = (ROOT / "frontend" / "src" / "dashboard" / "view.js").as_uri()
    payload = {"x_key": "s", "y_keys": ["v"], "other": {"label": "Other (4)"},
               "rows": [{"s": "Cedar", "v": 5}, {"s": "Ash", "v": 9}, {"s": "Birch", "v": 7}, {"s": "Other (4)", "v": 3}]}
    script = f"""import {{ arranged }} from '{module}';
      const p = {json.dumps(payload)};
      const names = (a) => a.rows.map((r) => r.s);
      console.log(JSON.stringify([names(arranged(p, 'high', '')), names(arranged(p, 'low', '')),
                                  names(arranged(p, 'name', '2')), arranged(p, 'name', '2').other]))"""
    run = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout) == [["Ash", "Birch", "Cedar", "Other (4)"], ["Cedar", "Birch", "Ash", "Other (4)"],
                                      ["Ash", "Birch"], None]


def test_the_bar_opens_the_panel_from_the_tile_s_button():
    bar = (ROOT / "frontend" / "src" / "dashboard" / "FilterBar.vue").read_text(encoding="utf-8")
    assert "[data-edit-tile]" in bar and "<ChartEditor" in bar
    panel = (ROOT / "frontend" / "src" / "dashboard" / "ChartEditor.vue").read_text(encoding="utf-8")
    assert "/portal/api/update-chart" in panel and 'role="dialog"' in panel and "Escape" in panel
