"""A dashboard the reader describes in the chat is designed, built, previewed and saved as one version.

"Build me a dashboard of sales by store and month" is recognised before any plan is asked for. One AI call
designs the tiles from the workspace's names (never its values): what was asked for, and a few more offered as
chips, never added. Each tile is a question of its own, planned and answered as a new conversation would be,
several at once, and the reader's card fills in as each lands, on a time of its own. A tile that cannot be built
says why and offers to define what is missing; the others are built. The reader previews the draft (kept on the
server behind a token) laid out as the dashboard will be, renames it, takes tiles out, adds the offered ones, and
saves it as a new dashboard or onto one of theirs, as one version. Each tile's figures are the chat answer's, its
tile is drawn again from its plan with no AI, and the dashboard's filters change each tile's SQL. A trend in two
units gets a panel each, and a metric says which way is good news.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import re
import uuid
from types import SimpleNamespace

import pytest

import store
from core2 import dashboards
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
YEAR = {"window": {"kind": "this", "unit": "year"}}
PLANS = {
    "Net amount this year": {"measures": ["net_amount"], "time": YEAR},
    "Number of orders this year": {"measures": ["number_of_orders"], "time": YEAR},
    "Net amount by month for the last 12 months": {"intent": "trend", "measures": ["net_amount"], "time": {
        "grain": "month", "window": {"kind": "last", "unit": "month", "n": 12}}},
    "Net amount by store this year": {"intent": "breakdown", "measures": ["net_amount"], "group_by": ["store.name"],
                                      "time": YEAR},
    "Net amount by segment this year": {"intent": "breakdown", "measures": ["net_amount"],
                                        "group_by": ["customer.segment"], "time": YEAR},
    "Top 10 customers by net amount this year": {"intent": "rank", "measures": ["net_amount"],
                                                 "group_by": ["customer"], "limit": 10, "time": YEAR,
                                                 "sort": [{"by": "net_amount", "desc": True}]},
    "Unicorns sold this year": {"measures": ["unicorns_sold"], "time": YEAR},
    "Margin percent this year": {"measures": ["margin_percent"], "time": YEAR},
}
DESIGN = {"name": "Sales overview", "tiles": [
    {"title": "Net amount", "kind": "number", "question": "Net amount this year"},
    {"title": "Orders", "kind": "number", "question": "Number of orders this year"},
    {"title": "Net amount by month", "kind": "trend", "question": "Net amount by month for the last 12 months"},
    {"title": "Net amount by store", "kind": "breakdown", "question": "Net amount by store this year"},
    {"title": "Net amount by segment", "kind": "breakdown", "question": "Net amount by segment this year"},
    {"title": "Top customers", "kind": "table", "question": "Top 10 customers by net amount this year"},
    {"title": "Unicorns", "kind": "number", "question": "Unicorns sold this year"}], "extras": [
    {"title": "Margin %", "kind": "number", "question": "Margin percent this year"},
    {"title": "Net amount again", "kind": "number", "question": "Net amount this year"}]}     # asked for already


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")


class AI:
    """The workspace's AI, scripted: the design for a dashboard request, each tile's plan by its question."""

    def __init__(self, design=DESIGN):
        self.design, self.asked, self.prompts = design, [], []

    def __call__(self, stable: str, tail: str) -> str:
        self.asked.append(tail)
        if tail.startswith("REQUEST:"):
            self.prompts.append(stable)
            return json.dumps(self.design)
        question = re.search(r"^QUESTION: (.*)$", tail, re.MULTILINE).group(1).strip()
        return json.dumps({"kind": "query", **PLANS[question]})


def _services(retail, ai) -> Services:
    built, model = retail
    return Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=ai, index=MemberIndex(), today=TODAY)


@pytest.mark.parametrize("words, about", [
    ("Build me a dashboard of sales by store and month", "sales by store and month"),
    ("can you make a pharmacy fills dashboard?", "pharmacy fills"), ("Create a dashboard", ""),
    ("I need a new dashboard showing net amount", "net amount"),
    ("show me the dashboard", None), ("what is on my dashboard", None), ("net amount by store", None),
    ("build me a sales dashboard please", "sales"), ("create dashboard for margin by region", "margin by region"),
    ("Crée-moi un tableau de bord des ventes par magasin", "ventes par magasin"),
    # about a dashboard the reader has: never a new one
    ("make the dashboard chart a bar", None), ("create a new tab on the dashboard", None),
    ("can you make my dashboard refresh daily", None), ("set up the dashboard filters", None),
    ("give me the dashboard link", None), ("I want my dashboard to show margin", None)])
def test_a_request_for_a_dashboard_is_recognised_before_any_plan(words, about):
    assert dashboards.asked_for(words) == about


def test_one_design_call_sees_names_never_values_and_each_tile_is_its_own_question(retail):
    built, model = retail
    ai = AI()
    payload = answer_question("Build me a dashboard of sales by store and month", _services(retail, ai), Session())
    assert payload["type"] == "dashboard_draft"
    assert len(ai.prompts) == 1                                      # one design call
    member = built.con.execute("SELECT customer_name FROM customers LIMIT 1").fetchone()[0]
    assert member not in ai.prompts[0] and "net_amount" in ai.prompts[0]
    tile_calls = [t for t in ai.asked if not t.startswith("REQUEST:")]
    assert all("PREVIOUS QUESTION" not in t for t in tile_calls)     # each a new conversation
    draft = payload["draft"]
    assert draft["name"] == "Sales overview" and draft["built"] == 6 and draft["failed"] == 1
    failed = next(t for t in draft["tiles"] if t["error"])
    assert failed["title"] == "Unicorns" and failed["error"].startswith("I could not work out how to answer that")
    assert failed["definable"] is True                              # nothing measures it: the reader may define it
    assert [t["title"] for t in payload["draft_tiles"]] == [t["title"] for t in DESIGN["tiles"][:6]]
    assert all(t["plan"] and t["layout"] for t in payload["draft_tiles"])
    # What the request did not ask for is offered, never built; one that repeats a tile is not offered at all.
    assert draft["extras"] == [{"title": "Margin %", "kind": "number", "question": "Margin percent this year"}]
    assert not any("Margin percent" in t for t in ai.asked)


def test_the_card_fills_in_as_each_tile_lands(retail):
    seen: list[dict] = []
    services = _services(retail, AI())
    services.progress = seen.append
    payload = answer_question("Build me a dashboard of sales", services, Session())
    assert payload["type"] == "dashboard_draft" and len(seen) == 8            # once designed, then once per tile
    first, last = seen[0]["draft"], seen[-1]["draft"]
    assert all(c["type"] == "dashboard_building" for c in seen)
    assert first["pending"] == 7 and all(t["pending"] and t["layout"] for t in first["tiles"])    # placed already
    pending = [c["draft"]["pending"] for c in seen]
    assert pending == sorted(pending, reverse=True) and last["pending"] == 0
    assert (last["built"], last["failed"]) == (6, 1)


def test_a_failing_tile_offers_to_define_only_what_is_missing():
    def refused(**extra):
        return lambda question: {"answer": {"headline": "No."}, "data": None, **extra}

    assert dashboards.answered(dashboards.Tile("A", "number", "q"), refused(unsupported=True)).definable
    assert dashboards.answered(dashboards.Tile("A", "number", "q"), refused(clarify={"about": "other"})).definable
    db = dashboards.answered(dashboards.Tile("A", "number", "q"), refused())      # the database was down: not that
    assert db.error == "No." and not db.definable


def test_each_tile_is_the_chat_answer_and_is_drawn_again_from_its_plan_with_no_ai(retail):
    from core2.service import _compute
    from core2.resolve.resolver import Context

    built, model = retail
    payload = answer_question("Build me a dashboard of sales", _services(retail, AI()), Session())
    silent = _services(retail, lambda s, t: pytest.fail("a tile was redrawn through the AI"))
    for tile in payload["draft_tiles"]:
        chat = answer_question(tile["question"], _services(retail, AI()), Session())
        from core2.plan.ir import Plan

        redrawn = _compute("", Plan.model_validate(tile["plan"]), silent, Context(today=TODAY), question_id="",
                           started=0.0)
        assert redrawn["export_rows"] == chat["export_rows"], tile["title"]


def test_the_layout_puts_numbers_in_a_row_trends_across_and_breakdowns_side_by_side(retail):
    payload = answer_question("Build me a dashboard of sales", _services(retail, AI()), Session())
    places = {t["title"]: t["layout"] for t in payload["draft_tiles"]}
    assert places["Net amount"] == {"grid_x": 0, "grid_y": 0, "grid_w": 6, "grid_h": 3, "tab": 0}
    assert places["Orders"]["grid_x"] == 6 and places["Orders"]["grid_y"] == 0
    assert places["Net amount by month"] == {"grid_x": 0, "grid_y": 3, "grid_w": 12, "grid_h": 10, "tab": 0}
    assert (places["Net amount by store"]["grid_w"], places["Net amount by segment"]["grid_x"]) == (6, 6)
    assert places["Top customers"]["grid_w"] == 12
    many = [dashboards.Tile(f"t{i}", "breakdown", "q") for i in range(15)]
    dashboards.lay_out(many)
    assert [t.layout["tab"] for t in many] == [0] * 12 + [1] * 3


def test_a_design_that_names_nothing_the_data_has_is_said(retail):
    payload = answer_question("Build me a dashboard of unicorns", _services(retail, AI({"name": "x", "tiles": []})),
                              Session())
    assert payload["type"] == "assistant_response" and "could not design that dashboard" in payload["answer"]["headline"]


# ── kept, previewed and saved ──────────────────────────────────────────────


@pytest.fixture()
def reader(retail):
    store.init_db()
    account = f"described-{uuid.uuid4().hex[:10]}"
    store.upsert_client(account, "portal")
    owner = store.create_user(account, "Ada Owner", f"{uuid.uuid4().hex[:8]}@test.com",
                              password="a-password-they-chose")[0]
    other = store.create_user(account, "Bo Other", f"{uuid.uuid4().hex[:8]}@test.com",
                              password="a-password-they-chose")[0]
    payload = answer_question("Build me a dashboard of sales", _services(retail, AI()), Session())
    yield account, owner, other, payload
    with store.get_db() as conn:
        for table in ("pinned_chart", "dashboard_artifact_version", "dashboard_artifact", "core2_dashboard_draft",
                      "portal_user", "client"):
            conn.execute(f"DELETE FROM {table} WHERE account_id=?", (account,))


def _client(user_id):
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    import portal.routes as routes

    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)
    client.cookies.set(routes._COOKIE, routes._sign_session_value(user_id))
    return client


def test_the_draft_is_renamed_trimmed_and_saved_as_one_version(reader):
    from gateway.core2_bridge import _drafted

    account, owner, other, payload = reader
    sent = _drafted(account, {"id": owner, "account_id": account}, dict(payload), "q-1")
    assert "draft_tiles" not in sent and sent["token"] and sent["draft"]["saveable"] is True
    token = sent["token"]
    me, them = _client(owner), _client(other)
    assert them.post(f"/portal/api/core2/dashboard-draft/{token}/rename", json={"name": "x"}).status_code == 404
    assert me.post(f"/portal/api/core2/dashboard-draft/{token}/rename", json={"name": "Store sales"}).json()["ok"]
    removed = me.post(f"/portal/api/core2/dashboard-draft/{token}/remove", json={"tile": 1}).json()
    assert (removed["ok"], removed["removed"], removed["left"]) == (True, "Orders", 5)
    assert removed["layouts"][0] == {"grid_x": 0, "grid_y": 0, "grid_w": 12, "grid_h": 3, "tab": 0}   # closes up
    saved = me.post(f"/portal/api/core2/dashboard-draft/{token}/save", json={"dashboard_id": "new"}).json()
    assert saved["ok"] and saved["tiles"] == 5 and saved["dashboard"]["name"] == "Store sales"
    board = saved["dashboard"]["id"]
    versions = store.list_dashboard_versions(board, owner, account)
    assert [v["change_summary"] for v in versions] == ["Built from your request"]
    charts = store.list_dashboard_charts(board, owner)
    assert [c["title"] for c in sorted(charts, key=lambda c: c["position"])] == [
        "Net amount", "Net amount by month", "Net amount by store", "Net amount by segment", "Top customers"]
    assert all(json.loads(c["display_config"])["core2_plan"] for c in charts)
    assert me.post(f"/portal/api/core2/dashboard-draft/{token}/save", json={}).status_code == 404   # used once


def test_a_draft_saved_onto_a_dashboard_of_theirs_goes_below_what_it_has(reader):
    from gateway.core2_bridge import _drafted

    account, owner, other, payload = reader
    mine = store.create_dashboard(account, owner, "thread", "Mine")
    store.pin_chart(owner, account, "Already here", "q", "SELECT 1", "kpi", None, dashboard_id=mine["id"])
    token = _drafted(account, {"id": owner}, dict(payload), "q-1")["token"]
    theirs = store.create_dashboard(account, other, "thread", "Theirs")
    me = _client(owner)
    assert me.post(f"/portal/api/core2/dashboard-draft/{token}/save",
                   json={"dashboard_id": theirs["id"]}).status_code == 403
    saved = me.post(f"/portal/api/core2/dashboard-draft/{token}/save", json={"dashboard_id": mine["id"]}).json()
    assert saved["ok"] and saved["dashboard"]["id"] == mine["id"]
    charts = store.list_dashboard_charts(mine["id"], owner)
    first = next(c for c in charts if c["title"] == "Already here")
    added = [c for c in charts if c["title"] != "Already here"]
    assert len(added) == 6 and min(c["grid_y"] for c in added) >= first["grid_y"] + first["grid_h"]
    assert store.list_dashboard_versions(mine["id"], owner, account)[0]["change_summary"] == "Built from your request"


def test_the_dashboards_filters_change_each_tiles_sql(retail):
    from core2.compile.compiler import compile_query
    from core2.plan.ir import Plan
    from core2.resolve.resolver import Context, resolve
    from portal.routes import _view_plan

    built, model = retail
    payload = answer_question("Build me a dashboard of sales", _services(retail, AI()), Session())
    for tile in payload["draft_tiles"]:
        viewed, _ = _view_plan(tile["plan"], {"period": "last_year", "members": {"region.name": "North"}})
        before = compile_query(resolve(Plan.model_validate(tile["plan"]), model, Context(today=TODAY)), model,
                               "duckdb").sql
        after = compile_query(resolve(Plan.model_validate(viewed), model, Context(today=TODAY)), model, "duckdb").sql
        assert after != before and "'North'" in after, tile["title"]


def test_an_offered_tile_is_added_built_as_the_rest_and_laid_out_with_them(reader, retail, monkeypatch):
    from gateway.core2_bridge import _drafted

    account, owner, other, payload = reader
    store.set_query_engine(account, "core2")
    asked: list[str] = []

    async def tile_answer(account_id, question, portal_user, token):
        asked.append(question)
        return "answered", answer_question(question, _services(retail, AI()), Session())

    monkeypatch.setattr("gateway.core2_bridge.tile_answer", tile_answer)
    token = _drafted(account, {"id": owner}, dict(payload), "q-1")["token"]
    me, them = _client(owner), _client(other)
    assert them.post(f"/portal/api/core2/dashboard-draft/{token}/add", json={"extra": 0}).status_code == 404
    added = me.post(f"/portal/api/core2/dashboard-draft/{token}/add", json={"extra": 0}).json()
    assert asked == ["Margin percent this year"] and added["ok"]
    assert added["tile"]["title"] == "Margin %" and added["tile"]["value"].endswith("%") and added["extras"] == []
    kept = store.get_dashboard_draft(token, account, owner)
    assert kept["tiles"][-1]["title"] == "Margin %" and kept["tiles"][-1]["plan"]["measures"] == ["margin_percent"]
    numbers = [t["layout"] for t in kept["tiles"] if t["kind"] == "number"]
    assert [(n["grid_x"], n["grid_w"], n["grid_y"]) for n in numbers] == [(0, 4, 0), (4, 4, 0), (8, 4, 0)]
    assert added["layouts"] == [t["layout"] for t in kept["tiles"]]
    assert me.post(f"/portal/api/core2/dashboard-draft/{token}/add", json={"extra": 0}).status_code == 400  # once
    # An offered tile the data cannot answer says why, offers to define it, and stays offered.
    kept["extras"] = [{"title": "Unicorns", "kind": "number", "question": "Unicorns sold this year"}]
    store.update_dashboard_draft(token, account, owner, kept)
    refused = me.post(f"/portal/api/core2/dashboard-draft/{token}/add", json={"extra": 0})
    assert refused.status_code == 422 and refused.json()["definable"] is True
    assert len(store.get_dashboard_draft(token, account, owner)["extras"]) == 1
    store.set_query_engine(account, "legacy")
    assert me.post(f"/portal/api/core2/dashboard-draft/{token}/add", json={"extra": 0}).status_code == 400


def _socket(sent: list):
    class Socket:
        async def send_json(self, payload):
            sent.append(payload)

    return SimpleNamespace(thread_id="t", send_lock=asyncio.Lock(), session_id="s"), Socket()


def test_the_chat_sends_the_card_as_it_fills_in_on_a_time_of_its_own(reader, retail, monkeypatch):
    import time

    import gateway.core2_bridge as bridge

    account, owner, other, _ = reader

    def portal_answer(account_id, question, portal_user, *, session_key, question_id="", progress=None):
        time.sleep(0.3)                    # longer than a question may take here, as a dashboard may
        services = _services(retail, AI())
        services.progress = progress
        return answer_question(question, services, Session())

    monkeypatch.setattr("core2.service.portal_answer", portal_answer)
    monkeypatch.setattr(bridge, "_over_limit", lambda *a, **k: "")
    monkeypatch.setattr(bridge, "_keep", lambda *a, **k: 1)
    monkeypatch.setattr(bridge, "TIMEOUT_SECONDS", 0.1)
    user = {"id": owner, "account_id": account, "lang": "en"}
    sent: list[dict] = []
    adapter, socket = _socket(sent)
    asyncio.run(bridge.answer_instead("core2", adapter, socket, account, "Build me a dashboard of sales", user))
    kinds = [p["type"] for p in sent]
    assert kinds == ["dashboard_building"] * 8 + ["dashboard_draft"]
    assert len({p["question_id"] for p in sent}) == 1                 # one card, filled in, then finished
    final = sent[-1]
    assert final["token"] and "draft_tiles" not in final
    assert store.get_dashboard_draft(final["token"], account, owner)["extras"][0]["title"] == "Margin %"
    # The same wait on a question is past its time.
    plain: list[dict] = []
    adapter, socket = _socket(plain)
    asyncio.run(bridge.answer_instead("core2", adapter, socket, account, "Net amount this year", user))
    assert plain[-1]["kind"] == "could_not" and "within 0 seconds" in plain[-1]["answer"]["headline"]


_CARD = dict(functions=["function dashboardDraftHtml(msg)", "function _draftDrawing(tile)", "function _draftPlace(tile)",
                        "function escHtml(str)"], consts=["_DRAFT_ICONS", "_DRAFT_DEFAULT"],
             preamble="function qbIcon(n){return '<svg></svg>';}")
_TILES = [
    {"id": 0, "title": "Net amount", "kind": "number", "value": "$1.20M", "error": "",
     "layout": {"grid_x": 0, "grid_y": 0, "grid_w": 12, "grid_h": 3, "tab": 0}},
    {"id": 1, "title": "Net amount by month", "kind": "trend", "error": "", "headline": "Rising",
     "layout": {"grid_x": 0, "grid_y": 3, "grid_w": 12, "grid_h": 10, "tab": 0},
     "chart": {"chart_type": "line", "rows": [{"x": "2026-01-01", "y": 10}, {"x": "2026-02-01", "y": 30}]}},
    {"id": 2, "title": "By store", "kind": "breakdown", "error": "", "headline": "Airport leads",
     "layout": {"grid_x": 0, "grid_y": 13, "grid_w": 6, "grid_h": 10, "tab": 0},
     "chart": {"chart_type": "bar", "rows": [{"x": "Airport", "y": 200}, {"x": "Harbour", "y": -50}]}},
    {"id": 3, "title": "Top customers", "kind": "table", "error": "",
     "layout": {"grid_x": 0, "grid_y": 0, "grid_w": 12, "grid_h": 10, "tab": 1},
     "chart": {"chart_type": "table", "columns": ["Customer", "Net amount"], "rows": [["Blue & Co", "$9.00"]]}},
    {"id": 4, "title": "Fills", "kind": "number", "error": "There is no measure called fills.", "definable": True}]


def _card(msg: dict) -> str:
    pytest.importorskip("dukpy")
    from tests.chat_js import run

    return run(f"JSON.stringify(dashboardDraftHtml({json.dumps(msg)}));", lang="en", **_CARD)


def test_the_card_is_laid_out_as_the_dashboard_with_each_tile_drawn_small():
    markup = _card({"type": "dashboard_draft", "token": "tok", "dashboards": [{"id": 7, "name": "Mine"}],
                    "draft": {"name": "Sales overview", "built": 4, "saveable": True, "tiles": _TILES,
                              "extras": [{"title": "Margin %", "kind": "number", "question": "Margin percent"}]}})
    assert 'value="Sales overview"' in markup and "Built from your request: 4 tiles · 1 could not be built" in markup
    assert 'style="grid-column:1 / span 12;grid-row:4 / span 10"' in markup             # where it will be
    assert "<polyline" in markup and 'aria-label="Rising"' in markup                     # the trend, as a line
    assert "width:100%" in markup and "is-negative" in markup and "width:25%" in markup  # the bars, to scale
    assert "<td>Blue &amp; Co</td>" in markup and "Tab 2" in markup                      # the table, on its tab
    assert [int(n) for n in re.findall(r'data-draft-remove="(\d+)"', markup)] == [0, 1, 2, 3]
    assert "There is no measure called fills." in markup and 'data-draft-define="Fills"' in markup
    assert 'data-draft-add="0"' in markup and "Margin %" in markup and '<option value="7">Mine</option>' in markup


def test_the_card_says_what_is_still_being_built_and_offers_nothing_until_done():
    tiles = [dict(t, pending=True, chart=None) if t["id"] in (2, 3) else t for t in _TILES[:4]]
    markup = _card({"type": "dashboard_building", "draft": {"name": "Sales overview", "tiles": tiles,
                                                            "extras": [{"title": "Margin %", "question": "q"}]}})
    assert "Building your dashboard: 2 of 4 tiles ready" in markup and markup.count("Building…") == 2
    assert 'aria-busy="true"' in markup and "data-draft-" not in markup and "dash-draft-name" not in markup
    saved = _card({"type": "dashboard_draft", "token": "tok", "saved": "/portal/dashboard?dashboard_id=4",
                   "draft": {"name": "Sales overview", "built": 4, "saveable": True, "tiles": _TILES[:4]}})
    assert 'href="/portal/dashboard?dashboard_id=4"' in saved and "data-draft-" not in saved


def test_a_trend_in_two_units_is_drawn_in_a_panel_each(retail):
    plan = {"intent": "trend", "measures": ["net_amount", "margin_percent"],
            "time": {"grain": "month", "window": {"kind": "last", "unit": "month", "n": 12}}}
    answers = [json.dumps({"kind": "query", **plan})]
    built, model = retail
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=lambda s, t: answers.pop(0),
                        index=MemberIndex(), today=TODAY)
    chart = answer_question("Net amount and margin by month", services, Session())["chart"]
    assert chart["chart_type"] == "line" and chart["facets"] == [["net_amount"], ["margin_percent"]]


@pytest.mark.parametrize("better, before, now, good", [
    ("higher", 100, 120, True), ("higher", 120, 100, False), ("lower", 100, 120, False), ("lower", 120, 100, True),
    ("neither", 100, 120, None)])
def test_a_metric_says_which_way_is_good_news(better, before, now, good):
    from core2.answer.builder import against

    assert against(now, before, "number", "May", better=better)["good"] is good


def test_the_admin_sets_which_way_is_good_and_the_chat_colours_the_change(retail):
    from core2.model.authoring import editor_state
    from core2.model.overrides import apply_overrides

    built, model = retail
    model = model.model_copy(deep=True)
    cost = model.measures[next(k for k, m in model.measures.items() if m.slug == "cost_amount")]
    assert editor_state(model, cost)["better"] == "higher"
    assert apply_overrides(model, [{"object_key": f"measure:{cost.key}", "field": "better", "value": "lower"}]) == []
    assert model.measures[cost.key].better == "lower" and editor_state(model, model.measures[cost.key])["better"] == "lower"
    pytest.importorskip("dukpy")
    from tests.chat_js import run

    def line(change):
        return run(f"JSON.stringify(_kpiChangeHtml({json.dumps(change)}));", lang="en",
                   functions=["function _kpiChangeHtml(change)", "function escHtml(str)"])

    rising = {"direction": "up", "change": "+$20.00", "pct": "+20.0%", "vs": "May", "before": "$100.00"}
    assert 'class="kpi-change is-bad"' in line({**rising, "good": False})
    assert 'class="kpi-change is-good"' in line({**rising, "good": True})
    assert 'class="kpi-change"' in line({**rising, "good": None}) and 'class="kpi-change"' in line(rising)


def test_every_icon_the_card_draws_is_in_the_sprite():
    """The card picks a tile's icon from a table, which the page-wide sprite check cannot see."""
    import pathlib

    page = pathlib.Path("portal/templates/portal_chat.html").read_text(encoding="utf-8")
    table = re.search(r"const _DRAFT_ICONS = \{([^}]*)\}", page).group(1)
    names = re.findall(r"'([a-z0-9-]+)'", table)
    sprite = set(re.findall(r'id="([a-z0-9-]+)"', pathlib.Path("static/icons/qb-icons.svg").read_text(encoding="utf-8")))
    assert names and not set(names) - sprite, set(names) - sprite


def test_no_reader_string_is_named_twice():
    """Two strings under one name: the second silently replaces the first, wherever it was used."""
    import ast
    import collections
    import pathlib

    tree = ast.parse(pathlib.Path("core/i18n.py").read_text(encoding="utf-8"))
    twice = [k for node in ast.walk(tree) if isinstance(node, ast.Dict)
             for k, n in collections.Counter(key.value for key in node.keys
                                             if isinstance(key, ast.Constant) and isinstance(key.value, str)).items()
             if n > 1]
    assert twice == []
