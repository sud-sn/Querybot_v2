"""A dashboard tile is its name, and the reader chooses it.

Asked for: no question on a dashboard tile, only the chart's name; a box to name a KPI or a chart when it
is added to a dashboard (the chart's own name when none is given); the name editable on the dashboard too;
and a dashboard that looks refined, as the approved design did.

* a tile shows its name and the period it counts, never the question it came from, a type badge or
  "Live governed refresh";
* the add dialog holds the chart's own name, ready to change; a name typed there, or given on the
  dashboard (Edit, the pencil or the name), is kept as written: a name that happens to be the measure
  alone ("Net amount") is never replaced by the answer's own title ("Net amount by store"), as an
  untouched tile pinned under the measure alone still is;
* renaming: Enter or leaving the box saves, Escape keeps the old name, an empty box changes nothing, and
  a name the server would not save goes back to the old one;
* a number tile, as the plan's cards: its number, its change coloured with its arrow and "vs" the period
  before; the measure's own name only when the tile was given another (its trend line stays in the chat,
  where the wash under it still closes on its foot);
* tiles keep the grid's gap between them on every side.

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

from core2.answer.builder import sparkline
from tests.js_lift import function as lift
from tests.test_a_new_core_answer_can_be_pinned import _reader, _replay, fresh_store, retail  # noqa: F401
from tests.test_core2_charts_read_as_their_answer import BY_STORE, _ask
from tests.test_dashboard_page import _chart, _render, _visible

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = (ROOT / "portal" / "templates" / "portal_dashboard.html").read_text(encoding="utf-8")


def _json_request(body: dict) -> MagicMock:
    request = MagicMock()
    request.json = MagicMock(return_value=asyncio.sleep(0, result=body))
    return request


def _pinned(fresh_store, retail, title: str, **extra):  # noqa: F811
    """Pinned through the real bridge token and pin API, under ``title``; the stored chart and its reader."""
    from gateway import core2_bridge
    from portal import routes

    user = _reader(fresh_store)
    token = core2_bridge._pin(user["account_id"], user, "net amount by store in april", _ask(retail, BY_STORE))
    with patch.object(routes, "_get_portal_user", return_value=user):
        response = asyncio.run(routes.pin_chart_api(_json_request(
            {"token": token, "title": title, "new_dashboard_name": "Mine", **extra})))
    body = json.loads(response.body)
    assert body["ok"] is True
    chart = next(c for c in fresh_store.list_pinned_charts(user["id"]) if c["title"] == title)
    return chart, user, body["dashboard"]["id"]


def _drawn(fresh_store, retail, chart_id: int, user: dict) -> dict:  # noqa: F811
    from portal import routes

    chart = next(c for c in fresh_store.list_pinned_charts(user["id"]) if c["id"] == chart_id)
    with patch("core2.service.portal_replay", _replay(retail)):
        return routes._refresh_chart(chart, {"db_type": "duckdb"}, user)


# ── the name ────────────────────────────────────────────────────────────────


def test_a_tile_shows_its_name_and_period_never_its_question(fresh_store, retail):  # noqa: F811
    chart, user, _ = _pinned(fresh_store, retail, "Net amount by store")
    tile = _drawn(fresh_store, retail, chart["id"], user)
    markup = _visible(_render([_chart(**{**tile, "question": "net amount by store in april"})]))
    assert ">Net amount by store<" in markup and "April 2026" in markup
    assert "net amount by store in april" not in markup
    assert "Live governed" not in markup and ">BAR<" not in markup


def test_a_name_typed_when_adding_is_kept_even_when_it_is_the_measure_alone(fresh_store, retail):  # noqa: F811
    chart, user, _ = _pinned(fresh_store, retail, "Net amount", named=True)
    assert chart["title_set"] == 1
    assert _drawn(fresh_store, retail, chart["id"], user)["title"] == "Net amount"


def test_an_untouched_tile_under_its_measure_alone_still_reads_as_its_answer(fresh_store, retail):  # noqa: F811
    chart, user, _ = _pinned(fresh_store, retail, "Net amount")
    assert chart["title_set"] == 0
    assert _drawn(fresh_store, retail, chart["id"], user)["title"] == "Net amount by store"


def test_a_tile_renamed_on_the_dashboard_keeps_its_name(fresh_store, retail):  # noqa: F811
    from portal import routes

    chart, user, dashboard_id = _pinned(fresh_store, retail, "Net amount by store")
    with patch.object(routes, "_get_portal_user", return_value=user):
        saved = asyncio.run(routes.update_chart_api(_json_request(
            {"dashboard_id": dashboard_id, "chart_id": chart["id"], "title": "  Net amount  "})))
        assert json.loads(saved.body)["ok"] is True
        blank = asyncio.run(routes.update_chart_api(_json_request(
            {"dashboard_id": dashboard_id, "chart_id": chart["id"], "title": "   "})))
    assert blank.status_code == 403, "an empty name changes nothing"
    assert _drawn(fresh_store, retail, chart["id"], user)["title"] == "Net amount"


# ── renaming on the page ────────────────────────────────────────────────────


RENAME = """
function node(extra) { return Object.assign({hidden: false, dataset: {}, handlers: {}, removed: false,
  addEventListener: function (ev, fn) { this.handlers[ev] = fn; }, focus: function () {}, select: function () {},
  remove: function () { this.removed = true; }, setAttribute: function () {}}, extra || {}); }
var input = node({value: ''});
var title = node({textContent: 'Net amount by store', title: 'Net amount by store', after: function () {}});
var document = {createElement: function () { return input; }};
var saved = [], toasts = [];
var window = {qbToast: {notify: function (m) { toasts.push(m.body); }}};
function t(id) { return id; }
function _updateChart(id, fields) { saved.push([id, fields]); return Promise.resolve(OK); }
"""


def _rename(steps: str, ok: bool = True) -> dict:
    if shutil.which("node") is None:
        pytest.skip("node runs the page's own rename")
    script = (RENAME.replace("OK", "true" if ok else "false") + lift(DASHBOARD, "function startRenameChart(el, chartId)")
              + "\nstartRenameChart(title, 7);\n" + steps
              + "\nsetTimeout(function () { console.log(JSON.stringify({name: title.textContent, hidden: title.hidden,"
                " box: input.removed, saved: saved, toasts: toasts})); }, 0);")
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=60, check=True).stdout
    return json.loads(out)


def test_enter_saves_the_name_typed():
    out = _rename("input.value = '  Store sales  '; input.handlers.keydown({key: 'Enter', preventDefault: function () {}});")
    assert out == {"name": "Store sales", "hidden": False, "box": True, "saved": [[7, {"title": "Store sales"}]],
                   "toasts": []}


def test_escape_keeps_the_old_name_and_saves_nothing():
    out = _rename("input.value = 'Other'; input.handlers.keydown({key: 'Escape', preventDefault: function () {},"
                  " stopPropagation: function () {}}); input.handlers.blur();")
    assert out["name"] == "Net amount by store" and out["saved"] == [] and out["box"] is True


def test_an_emptied_box_changes_nothing():
    out = _rename("input.value = '   '; input.handlers.blur();")
    assert out["name"] == "Net amount by store" and out["saved"] == []


def test_a_name_the_server_would_not_keep_goes_back():
    out = _rename("input.value = 'Store sales'; input.handlers.blur();", ok=False)
    assert out["name"] == "Net amount by store" and out["toasts"] == ["ui.dash.rename_failed"]


# ── a number tile ───────────────────────────────────────────────────────────


KPI = {"label": "Net amount", "value": 185933.03,
       "change": {"direction": "down", "pct": "-4.6%", "change": "-$8,950.00", "vs": "March 2026",
                  "before": "$194,883.03"},
       "trend": {**sparkline([150.0, 170.0, 160.0, 190.0, 186.0]), "span": "Dec 2025 – Apr 2026"}}


def test_a_number_tile_is_its_number_and_its_change():
    markup = _visible(_render([_chart(title="Net amount", chart_type="kpi", kpi=KPI, kpi_display="$185.9K",
                                      kpi_full="$185,933.03")]))
    assert 'class="dash-kpi-value" title="$185,933.03">$185.9K<' in markup
    assert "dash-kpi-change is-down" in markup and 'class="dash-kpi-delta"' in markup
    assert "4.6%" in markup and "-4.6%" not in markup, "the arrow says which way: the change is written plain"
    assert "vs March 2026" in markup and "dash-kpi-spark" not in markup
    assert 'class="dash-kpi-label"' not in markup, "the tile's name is the number's name"
    assert "Single-value result" not in markup


def test_a_number_tile_given_its_own_name_says_its_measure_too():
    markup = _visible(_render([_chart(title="Q2 sales", chart_type="kpi", kpi=KPI, kpi_display="$185,933.03")]))
    assert '<div class="dash-kpi-label">Net amount</div>' in markup


def test_the_wash_under_a_trend_closes_on_its_foot_and_never_across_a_gap():
    drawn = sparkline([1.0, 3.0, 2.0, 5.0])
    assert drawn["area"] == drawn["path"] + " L118.0,32 L2.0,32 Z"
    assert sparkline([1.0, None, 2.0, 5.0, 4.0])["area"] == ""


def test_tiles_keep_the_grids_gap_on_every_side():
    """The grid places a tile's content by its insets; a height of 100% as well pushed each tile's bottom
    edge into the gap below it, and rows of tiles touched (seen in the browser, not testable without one)."""
    css = (ROOT / "static" / "css" / "dashboard.css").read_text(encoding="utf-8")
    assert ".chart-grid.is-gridstack .chart-card-content { height: auto; }" in css


# ── the pin link's own page ─────────────────────────────────────────────────


def _token(fresh_store, retail):  # noqa: F811
    from gateway import core2_bridge

    user = _reader(fresh_store)
    return core2_bridge._pin(user["account_id"], user, "net amount by store in april", _ask(retail, BY_STORE)), user


def test_the_pin_page_starts_from_the_charts_own_name_never_the_question(fresh_store, retail):  # noqa: F811
    from portal import routes

    token, user = _token(fresh_store, retail)
    with patch.object(routes, "_get_portal_user", return_value=user), \
            patch.object(routes, "_resp", lambda request, name, context: context):
        context = asyncio.run(routes.pin_confirm_page(MagicMock(), token=token))
    assert context["name"] == "Net amount by store"
    from tests.portal_render import render, visible

    markup = visible(render("portal_pin_confirm.html", lang="en", path="/portal/pin-confirm", user=None, token=token,
                            question=context["question"], name=context["name"], sql="", dashboards=[], error=""))
    assert 'value="Net amount by store"' in markup


@pytest.mark.parametrize("typed, kept", [("Net amount by store", 0), ("Net amount", 1)])
def test_the_pin_page_keeps_a_name_of_the_readers_own(fresh_store, retail, typed, kept):  # noqa: F811
    from portal import routes

    token, user = _token(fresh_store, retail)
    with patch.object(routes, "_get_portal_user", return_value=user):
        response = asyncio.run(routes.pin_confirm_submit(MagicMock(), token=token, title=typed, dashboard_id="",
                                                         new_dashboard_name="Mine"))
    assert response.status_code == 303
    chart = next(c for c in fresh_store.list_pinned_charts(user["id"]) if c["title"] == typed)
    assert chart["title_set"] == kept
    assert _drawn(fresh_store, retail, chart["id"], user)["title"] == "Net amount by store" if not kept else typed
