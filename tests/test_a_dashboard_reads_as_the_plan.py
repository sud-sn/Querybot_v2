"""A dashboard reads as the plan's: Edit, Share and a menu for the rest; filters over every tile; a card per number.

Asked for: the dashboard of the approved design -- an Edit button and a Share button, everything else
(subscribing, chat, tidy, details) in one menu beside them; a Period filter and filters on the members the
tiles are grouped by; KPI cards of one number each, never a tile of several; no tile named by its question;
and the same tile format for dashboards the product builds itself later.

* one way onto a dashboard (store.add_answer): an answer of several numbers becomes a card per number,
  and a tile pinned before that, of several numbers, is split the first time anyone opens its dashboard;
* the grid's rows are half as tall as they were, so a card is ~120px: a dashboard laid out on the old rows
  is brought onto the new ones once, and a version saved on the old rows comes back on the new;
* the Period replaces the window of a tile that counts one span and shades it on a tile over time; a member
  filter narrows every tile but the one grouped by its field; a tile that cannot take a filter is drawn
  without it and says so; a filter never renames a tile;
* Share keeps a dashboard to its owner or shares it with the team (published at once), only its owner;
* a card's number is short ($543.1K), its whole the tooltip;
* a chart over time names its lowest point (never the last, unfinished one) and shades the chosen period.

Invented data only.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.js_lift import function as lift
from tests.test_a_new_core_answer_can_be_pinned import _reader, _replay, fresh_store, retail  # noqa: F401
from tests.test_core2_charts_read_as_their_answer import _ask
from tests.test_dashboard_page import TestThePublishEndpoint, _chart, _render, _visible

ROOT = Path(__file__).resolve().parents[1]
APRIL = {"kind": "between", "start": "2026-04-01", "end": "2026-04-30"}
TOTAL = {"intent": "value", "measures": ["net_amount"], "time": {"window": APRIL}}
TWO = {"intent": "value", "measures": ["net_amount", "gross_amount"], "time": {"window": APRIL}}
BY_REGION = {"intent": "breakdown", "measures": ["net_amount"], "group_by": ["region.name"], "time": {"window": APRIL}}
TREND = {"intent": "trend", "measures": ["net_amount"],
         "time": {"window": {"kind": "between", "start": "2025-07-01", "end": "2026-06-30"}, "grain": "month"}}
TOP = {"intent": "rank", "measures": ["net_amount", "quantity"], "group_by": ["product.name"],
       "sort": [{"by": "net_amount", "desc": True}], "limit": 8, "time": {"window": APRIL}, "chart": "table"}


def _request(body: dict) -> MagicMock:
    request = MagicMock()
    request.json = MagicMock(return_value=asyncio.sleep(0, result=body))
    return request


def _pin(store, retail, user: dict, plan: dict, question: str, **body) -> list[dict]:  # noqa: F811
    """Pinned through the bridge's token and the pin API; the reader's tiles, in order."""
    from gateway import core2_bridge
    from portal import routes

    token = core2_bridge._pin(user["account_id"], user, question, _ask(retail, plan))
    with patch.object(routes, "_get_portal_user", return_value=user):
        response = asyncio.run(routes.pin_chart_api(_request({"token": token, **body})))
    assert json.loads(response.body)["ok"] is True, response.body
    return store.list_pinned_charts(user["id"])


def _draw(retail, chart: dict, user: dict, view: dict | None = None, replay=None) -> dict:  # noqa: F811
    from portal import routes

    with patch("core2.service.portal_replay", replay or _replay(retail)):
        return routes._refresh_chart(chart, {"db_type": "duckdb"}, user, view=view)


def _teammate(store, owner: dict) -> dict:
    user_id, _ = store.create_user(owner["account_id"], "Teammate", f"t{os.urandom(3).hex()}@example.com",
                                   role="analyst", password="teammate-pass-123")
    return {"id": user_id, "account_id": owner["account_id"], "role": "analyst"}


# ── one number a card ───────────────────────────────────────────────────────


def test_an_answer_of_several_numbers_is_a_card_each_named_for_its_number(fresh_store, retail):  # noqa: F811
    user = _reader(fresh_store)
    tiles = _pin(fresh_store, retail, user, TWO, "net amount and gross amount in april", new_dashboard_name="Mine")
    assert [(t["title"], t["chart_type"]) for t in tiles] == [("Net amount", "kpi"), ("Gross amount", "kpi")]
    assert [json.loads(t["display_config"])["core2_plan"]["measures"] for t in tiles] == [["net_amount"],
                                                                                         ["gross_amount"]]
    source = tiles[0]["data_source_id"]
    assert source and tiles[1]["data_source_id"] == source, "one answer, one query to refresh"
    dashboard_id = tiles[0]["dashboard_id"]
    assert fresh_store.remove_chart_from_dashboard(dashboard_id, tiles[0]["id"], user["id"], user["account_id"])
    assert fresh_store.get_data_source(source, user["id"], user["account_id"]), "still the other card's"
    assert fresh_store.remove_chart_from_dashboard(dashboard_id, tiles[1]["id"], user["id"], user["account_id"])
    assert not fresh_store.get_data_source(source, user["id"], user["account_id"])


def _old_group_tile(store, user: dict) -> int:
    """A tile of two numbers as one was pinned before each number had a card of its own."""
    dashboard = store.create_dashboard(user["account_id"], user["id"], "t", "Sales", visibility="team")
    chart_id = store.pin_chart(user["id"], user["account_id"], "Net and gross", "net and gross in april",
                               "SELECT 1", "kpi", None,
                               display_config={"core2_plan": TWO, "titles": ["Net amount", "Gross amount"]})
    store.add_chart_to_dashboard(dashboard["id"], chart_id, user["id"], user["account_id"])
    store.publish_dashboard(dashboard["id"], user["id"], user["account_id"])
    return int(dashboard["id"])


def test_an_older_tile_of_several_numbers_is_split_for_whoever_opens_it_first(fresh_store):  # noqa: F811
    owner = _reader(fresh_store)
    teammate = _teammate(fresh_store, owner)
    dashboard_id = _old_group_tile(fresh_store, owner)
    seen = fresh_store.list_dashboard_charts_for_view(dashboard_id, teammate["id"], owner["account_id"])
    assert [(c["title"], c["chart_type"]) for c in seen] == [("Net amount", "kpi"), ("Gross amount", "kpi")]
    again = fresh_store.list_dashboard_charts_for_view(dashboard_id, owner["id"], owner["account_id"])
    assert [c["id"] for c in again] == [c["id"] for c in seen], "split once, never twice"
    versions = fresh_store.list_dashboard_versions(dashboard_id, owner["id"], owner["account_id"])
    assert any("Each number on a tile of its own" in str(v.get("change_summary")) for v in versions)


def test_numbers_worked_out_from_each_other_stay_one_tile():
    from store.dashboard_store import number_titles

    assert number_titles({"core2_plan": {**TWO, "derived": [{"name": "margin"}]}}) == []
    assert number_titles({"core2_plan": TOTAL}) == []
    assert number_titles({"core2_plan": TWO}) == ["Net amount", "Gross amount"]


def test_a_card_is_its_number_and_never_a_row_of_several():
    group = {"label": "Net amount", "value": 5, "group": [{"label": "Net amount", "value": 5},
                                                          {"label": "Gross amount", "value": 6}]}
    cells = [{"label": "Net amount", "display": "5"}, {"label": "Gross amount", "display": "6"}]
    markup = _visible(_render([_chart(title="Net amount", chart_type="kpi", kpi=group, kpi_display="5",
                                      kpi_group=cells)]))
    assert "dash-kpi-group" not in markup and "dash-kpi-cell" not in markup
    assert markup.count('class="dash-kpi-value"') == 1


# ── the grid ────────────────────────────────────────────────────────────────


def _rect(store, dashboard_id: int, owner_id: int) -> tuple[int, int]:
    chart = store.list_dashboard_charts(dashboard_id, owner_id)[0]
    return chart["grid_y"], chart["grid_h"]


def test_a_dashboard_on_the_old_rows_comes_onto_todays_once(fresh_store):  # noqa: F811
    from store.db import get_db

    owner = _reader(fresh_store)
    dashboard = fresh_store.create_dashboard(owner["account_id"], owner["id"], "t", "Ops")
    chart_id = fresh_store.pin_chart(owner["id"], owner["account_id"], "Orders by region", "q", "SELECT 1", "bar", None)
    fresh_store.add_chart_to_dashboard(dashboard["id"], chart_id, owner["id"], owner["account_id"])
    fresh_store.update_dashboard_layouts(dashboard["id"], owner["id"], owner["account_id"],
                                         [{"chart_id": chart_id, "x": 0, "y": 3, "w": 6, "h": 5}])
    with get_db() as conn:
        conn.execute("UPDATE dashboard_artifact SET grid_scale=1 WHERE id=?", (dashboard["id"],))
    for _ in range(2):
        fresh_store.list_dashboard_charts_for_view(dashboard["id"], owner["id"], owner["account_id"])
        assert _rect(fresh_store, dashboard["id"], owner["id"]) == (6, 10), "where it was, as tall as it was"


def test_a_version_saved_on_the_old_rows_comes_back_on_todays(fresh_store):  # noqa: F811
    from store.db import get_db

    owner = _reader(fresh_store)
    dashboard = fresh_store.create_dashboard(owner["account_id"], owner["id"], "t", "Ops")
    chart_id = fresh_store.pin_chart(owner["id"], owner["account_id"], "Orders by region", "q", "SELECT 1", "bar", None)
    fresh_store.add_chart_to_dashboard(dashboard["id"], chart_id, owner["id"], owner["account_id"])
    with get_db() as conn:
        conn.execute("UPDATE dashboard_artifact SET grid_scale=1 WHERE id=?", (dashboard["id"],))
        conn.execute("UPDATE pinned_chart SET grid_y=2, grid_h=5 WHERE id=?", (chart_id,))
    saved = fresh_store.share_dashboard(dashboard["id"], owner["id"], owner["account_id"], "team")["version"]
    with get_db() as conn:
        conn.execute("UPDATE dashboard_artifact SET grid_scale=2 WHERE id=?", (dashboard["id"],))
        conn.execute("UPDATE pinned_chart SET grid_y=0, grid_h=3 WHERE id=?", (chart_id,))
    assert fresh_store.rollback_dashboard(dashboard["id"], owner["id"], owner["account_id"], saved)
    assert _rect(fresh_store, dashboard["id"], owner["id"]) == (4, 10)


# ── the filters ─────────────────────────────────────────────────────────────


def test_the_period_replaces_the_span_a_tile_counts_and_leaves_a_trend_its_range():
    from portal.routes import DASH_PERIODS, _view_plan

    compared = {**TOTAL, "time": {**TOTAL["time"], "compare": {"kind": "previous_period"}}}
    viewed, narrowed = _view_plan(compared, {"period": "last_quarter"})
    assert viewed["time"] == {"window": DASH_PERIODS["last_quarter"], "compare": {"kind": "previous_period"}}
    assert narrowed == [] and compared["time"]["window"] == APRIL, "the stored plan is left as it was"
    assert _view_plan(TREND, {"period": "last_quarter"})[0] == TREND
    listing = {"intent": "list", "group_by": ["store"]}
    assert _view_plan(listing, {})[0] == listing, "no filter, no change -- not even an empty time"


def test_a_member_filter_narrows_every_tile_but_the_one_grouped_by_its_field():
    from portal.routes import _view_plan

    view = {"members": {"region.name": "North", "store": "Downtown Store"}}
    viewed, narrowed = _view_plan(BY_REGION, view)
    assert narrowed == ["store"]
    assert viewed["filters"] == [{"field": "store", "op": "in", "values": ["Downtown Store"]}]
    assert _view_plan(TOTAL, view)[1] == ["region.name", "store"]


def test_a_tile_a_filter_narrows_keeps_its_name_and_counts_what_the_filter_leaves(fresh_store, retail):  # noqa: F811
    user = _reader(fresh_store)
    chart = _pin(fresh_store, retail, user, TOTAL, "net amount in april", new_dashboard_name="Mine")[0]
    assert chart["title"] == "Net amount" and not chart["title_set"]
    whole = _draw(retail, chart, user)
    north = _draw(retail, chart, user, {"period": "", "members": {"region.name": "North"}})
    assert north["title"] == "Net amount", "the filter bar says North; the tile keeps its name"
    assert 0 < north["kpi"]["value"] < whole["kpi"]["value"]
    assert not north.get("filter_warnings")


def test_a_filter_a_tile_cannot_take_is_left_off_and_said(fresh_store, retail):  # noqa: F811
    user = _reader(fresh_store)
    chart = _pin(fresh_store, retail, user, TOTAL, "net amount in april", new_dashboard_name="Mine")[0]
    plain = _replay(retail)

    def refuses_filters(account_id, plan, portal_user, *, question=""):
        if plan.get("filters") or plan["time"]["window"] != APRIL:
            return {"data": None, "answer": {"headline": "That cannot be counted here."}}
        return plain(account_id, plan, portal_user, question=question)

    drawn = _draw(retail, chart, user, {"period": "last_quarter", "members": {"region.name": "North"}},
                  replay=refuses_filters)
    assert drawn["filter_warnings"] == ["The Region filter does not apply to this tile.",
                                        "The Period filter does not apply to this tile."]
    assert drawn["kpi"]["value"] == _draw(retail, chart, user)["kpi"]["value"] and drawn["title"] == "Net amount"


def test_a_trend_shades_the_period_and_offers_no_member_filter(fresh_store, retail):  # noqa: F811
    user = _reader(fresh_store)
    chart = _pin(fresh_store, retail, user, TREND, "net amount by month", new_dashboard_name="Mine")[0]
    drawn = _draw(retail, chart, user, {"period": "last_quarter", "members": {}})
    shape = json.loads(drawn["chart_json"])
    assert shape["highlight"] == {"start": "2026-01-01", "end": "2026-04-01"}, "data to June: last quarter is Q1"
    assert len(shape["rows"]) == 12, "its own twelve months kept"
    assert "highlight" not in json.loads(_draw(retail, chart, user)["chart_json"])
    assert not drawn.get("grouping")


def test_a_breakdown_offers_its_members_as_a_filter_named_as_the_reader_thinks_of_it(fresh_store, retail):  # noqa: F811
    user = _reader(fresh_store)
    chart = _pin(fresh_store, retail, user, BY_REGION, "net amount by region in april", new_dashboard_name="Mine")[0]
    grouping = _draw(retail, chart, user)["grouping"]
    assert grouping["field"] == "region.name" and grouping["label"] == "Region"
    assert "North" in grouping["values"] and grouping["values"] == sorted(grouping["values"])


# ── names ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("plan, question, name", [
    (TOP, "top products by net amount and quantity in april", "Net amount and quantity by product"),
    ({"intent": "list", "group_by": ["store"]}, "list the stores", "Stores"),
])
def test_a_tile_added_without_a_name_takes_its_answers_never_its_question(fresh_store, retail, plan,  # noqa: F811
                                                                          question, name):
    user = _reader(fresh_store)
    chart = _pin(fresh_store, retail, user, plan, question, new_dashboard_name="Mine")[0]
    assert chart["title"] == name and chart["title"] != question


def test_a_tile_named_by_its_question_before_reads_as_its_answer(fresh_store, retail):  # noqa: F811
    from store.db import get_db

    user = _reader(fresh_store)
    question = "which were the top products by net amount and quantity in april this year"
    chart = _pin(fresh_store, retail, user, TOP, question, new_dashboard_name="Mine")[0]
    with get_db() as conn:     # as a tile was named before answers had names: its question, cut at 50
        conn.execute("UPDATE pinned_chart SET title=? WHERE id=?", (question[:50], chart["id"]))
    chart = fresh_store.list_pinned_charts(user["id"])[0]
    assert _draw(retail, chart, user)["title"] == "Net amount and quantity by product"


# ── Share, and the menu for the rest ────────────────────────────────────────


def test_share_shares_with_the_team_at_once_and_takes_it_back(fresh_store):  # noqa: F811
    owner = _reader(fresh_store)
    teammate = _teammate(fresh_store, owner)
    dashboard = fresh_store.create_dashboard(owner["account_id"], owner["id"], "t", "Sales")
    shared = fresh_store.share_dashboard(dashboard["id"], owner["id"], owner["account_id"], "team")
    assert (shared["visibility"], shared["status"]) == ("team", "published")
    assert fresh_store.get_dashboard_for_view(dashboard["id"], teammate["id"], owner["account_id"])
    assert fresh_store.share_dashboard(dashboard["id"], teammate["id"], owner["account_id"], "personal") is None
    kept = fresh_store.share_dashboard(dashboard["id"], owner["id"], owner["account_id"], "personal")
    assert kept["visibility"] == "personal"
    assert not fresh_store.get_dashboard_for_view(dashboard["id"], teammate["id"], owner["account_id"])
    with pytest.raises(ValueError):
        fresh_store.share_dashboard(dashboard["id"], owner["id"], owner["account_id"], "everyone")


class TestTheShareEndpoint:
    _client = TestThePublishEndpoint._client
    _signed_in = TestThePublishEndpoint._signed_in

    def test_the_owner_shares_and_nobody_else_can(self):
        client, store, account_id, user_id = self._signed_in()
        dashboard = store.create_dashboard(account_id, user_id, "t", "Ops")
        response = client.post(f"/portal/dashboard/{int(dashboard['id'])}/share", data={"visibility": "team"},
                               follow_redirects=False)
        assert response.status_code == 303 and response.headers["location"].endswith("&shared=1")
        assert store.get_dashboard(dashboard["id"], user_id, account_id)["visibility"] == "team"
        other = store.create_dashboard(account_id, store.create_user(
            account_id, "Other", f"{os.urandom(4).hex()}@x.com", password="another-password-1")[0], "t", "Theirs")
        response = client.post(f"/portal/dashboard/{int(other['id'])}/share", data={"visibility": "team"},
                               follow_redirects=False)
        assert response.status_code == 404
        response = client.post(f"/portal/dashboard/{int(dashboard['id'])}/share", data={"visibility": "public"},
                               follow_redirects=False)
        assert response.status_code == 400


def test_the_header_is_edit_share_and_a_menu_for_the_rest():
    import re

    markup = _render([_chart(chart_json='{"type":"bar"}')])
    start = markup.index('class="dash-header-actions"')
    actions = markup[start:markup.index("</header>", start)]
    assert re.search(r'<button[^>]*id="dashEditToggle"', actions)
    assert re.search(r'<button[^>]*class="btn btn-primary[^"]*"[^>]*id="dashShareButton"', actions)
    assert re.search(r'<button[^>]*id="dashMoreButton"[^>]*aria-haspopup="menu"[^>]*aria-controls="dashMoreMenu"',
                     actions)
    menu = actions[actions.index('id="dashMoreMenu"'):]
    assert menu.startswith('id="dashMoreMenu" role="menu"') and " hidden>" in menu[:120]
    for item in ('/subscribe"', 'name="cadence" value="weekly"', "/portal/chat?thread=", 'id="dashTidy"',
                 "dashAboutDialog"):
        assert item in menu, item
    dialog = markup[markup.index('id="dashShareDialog"'):]
    assert 'action="/portal/dashboard/5/share"' in dialog
    assert 'value="personal"' in dialog and 'value="team"' in dialog and 'id="dashShareLink"' in dialog


def test_a_reader_who_cannot_edit_sees_neither_edit_nor_share():
    markup = _render([_chart(chart_json='{"type":"bar"}')], can_edit=False)
    assert 'id="dashEditToggle"' not in markup and 'id="dashShareButton"' not in markup


# ── a card's number ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("value, fmt, shown", [
    (543107.86, "currency", "$543.1K"), (-8950.0, "currency", "-$8.9K"), (912.4, "currency", "$912.40"),
    (2551, "integer", "2,551"), (1_250_000, "integer", "1.2M"),
])
def test_a_cards_number_is_short_and_its_whole_is_its_tooltip(value, fmt, shown):
    from portal.routes import _kpi_compact

    assert _kpi_compact(value, fmt) == shown


# ── the chart over time ─────────────────────────────────────────────────────


QB_CHARTS = (ROOT / "static" / "js" / "qb-charts.js").read_text(encoding="utf-8")


def _node(script: str):
    if shutil.which("node") is None:
        pytest.skip("node runs the chart's own code")
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=60, check=True).stdout
    return json.loads(out)


def _low(values: list) -> list:
    labels = [f"m{i}" for i in range(len(values))]
    return _node("function t(id, v) { return 'Low ' + v.value; }\n" + lift(QB_CHARTS, "function lowPoint(")
                 + f"\nconst out = lowPoint({json.dumps(values)}, {json.dumps(labels)}, v => '$' + v,"
                 " {surface: '#fff', ink2: '#333', font: 'x'});"
                 "\nconsole.log(JSON.stringify(out.map(p => ({at: p.coord[0], align: p.label.align,"
                 " text: p.label.formatter(), position: p.label.position}))));")


def test_the_lowest_point_is_named_above_it_and_never_the_unfinished_last():
    assert _low([5, 3, 4, 6, 7, 6, 8, 1]) == [{"at": "m1", "align": "left", "text": "Low $3", "position": "top"}]
    assert _low([5, 6, 4, 2, 7, 6, 8, 9]) == [{"at": "m3", "align": "center", "text": "Low $2", "position": "top"}]
    assert _low([5, 6, 7, 8, 7, 6, 3, 9]) == [{"at": "m6", "align": "right", "text": "Low $3", "position": "top"}]
    assert _low([5, 3, 4]) == [], "three points have no low worth naming"


def test_the_chosen_period_is_shaded_from_its_first_period_to_its_last():
    rows = [{"month": f"2026-0{m}-01"} for m in range(1, 7)]
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    band = _node(lift(QB_CHARTS, "function tileBand(")
                 + f"\nconsole.log(JSON.stringify(tileBand({{highlight: {{start: '2026-01-01', end: '2026-04-01'}}}},"
                 f" {json.dumps(rows)}, 'month', {json.dumps(labels)}, {{}})));")
    assert band["data"] == [[{"xAxis": "Jan"}, {"xAxis": "Mar"}]] and band["silent"] is True
    none = _node(lift(QB_CHARTS, "function tileBand(")
                 + f"\nconsole.log(JSON.stringify(tileBand({{highlight: {{start: '2027-01-01', end: '2027-04-01'}}}},"
                 f" {json.dumps(rows)}, 'month', {json.dumps(labels)}, {{}})));")
    assert none is None
