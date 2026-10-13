"""A dashboard the reader asks for in the chat is designed, built and saved as theirs at once, on named tabs.

"Create an executive dashboard", "create a revenue dashboard" or "I need net amount by month, top 10 customers
and orders" is recognised before any plan is asked for. One AI call designs the tiles from the workspace's names
(never its values): the tiles a list names, a subject's headline numbers, trend, breakdowns and top 10, or the
whole business's figures by area; a subject's design too thin to be a dashboard is asked for again, once. Each
tile is a question of its own, planned and answered as a new conversation would be, several at once, on a time of
its own, while the chat says how many are ready (never what they show). A tile that cannot be built says why and
offers to define what is missing; the others are saved at once as a new dashboard of the reader's, one version,
on the design's tabs ("Overview" first, then a tab per area), and the chat says its name, its tabs and links to
it. Each tile's figures are the chat answer's, its tile is drawn again from its plan with no AI, and the
dashboard's filters change each tile's SQL. A trend in two units gets a panel each, and a metric says which way
is good news.
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
    {"title": "Net amount", "kind": "number", "question": "Net amount this year", "tab": "Overview"},
    {"title": "Orders", "kind": "number", "question": "Number of orders this year", "tab": "Overview"},
    {"title": "Net amount by month", "kind": "trend", "question": "Net amount by month for the last 12 months",
     "tab": "Overview"},
    {"title": "Net amount by store", "kind": "breakdown", "question": "Net amount by store this year",
     "tab": "Stores"},
    {"title": "Net amount by segment", "kind": "breakdown", "question": "Net amount by segment this year",
     "tab": "Customers"},
    {"title": "Top customers", "kind": "table", "question": "Top 10 customers by net amount this year",
     "tab": "customers"},
    {"title": "Unicorns", "kind": "number", "question": "Unicorns sold this year", "tab": "Overview"}]}


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")


class AI:
    """The workspace's AI, scripted: the design for a dashboard request (the next of several, when it is asked
    again), each tile's plan by its question."""

    def __init__(self, *designs):
        self.designs, self.asked, self.prompts = list(designs or [DESIGN]), [], []

    def __call__(self, stable: str, tail: str) -> str:
        self.asked.append(tail)
        if tail.startswith("REQUEST:"):
            self.prompts.append(stable)
            return json.dumps(self.designs.pop(0) if len(self.designs) > 1 else self.designs[0])
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
    ("Create an executive dashboard", "executive"), ("Prepare a CFO dashboard", "CFO"),
    ("Generate a revenue dashboard for 2026", "revenue 2026"), ("Show me a dashboard", ""),
    ("I'd like a dashboard of fills by pharmacy", "fills by pharmacy"),
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
    assert payload["type"] == "dashboard_built"
    assert len(ai.prompts) == 1                                      # one design call
    assert ai.asked[0] == "REQUEST: Build me a dashboard of sales by store and month"     # the whole request
    member = built.con.execute("SELECT customer_name FROM customers LIMIT 1").fetchone()[0]
    assert member not in ai.prompts[0] and "net_amount" in ai.prompts[0]
    tile_calls = [t for t in ai.asked if not t.startswith("REQUEST:")]
    assert all("PREVIOUS QUESTION" not in t for t in tile_calls)     # each a new conversation
    assert payload["name"] == "Sales overview" and len(payload["dashboard_tiles"]) == 6
    (failed,) = payload["not_built"]
    assert failed["title"] == "Unicorns" and failed["error"].startswith("I could not work out how to answer that")
    assert failed["definable"] is True                              # nothing measures it: the reader may define it
    assert [t["title"] for t in payload["dashboard_tiles"]] == [t["title"] for t in DESIGN["tiles"][:6]]
    assert all(t["plan"] and t["layout"] for t in payload["dashboard_tiles"])
    assert "export_rows" not in json.dumps(payload["dashboard_tiles"])        # its plans, never its rows


THIN = {"name": "Revenue", "tiles": [{"title": "Net amount", "kind": "number", "question": "Net amount this year"}]}


def test_a_subjects_dashboard_of_one_tile_is_asked_for_again_once(retail):
    """"Create a revenue dashboard" came back from a real model as one number: a subject is never one tile."""
    ai = AI(THIN, DESIGN)
    payload = answer_question("Create a revenue dashboard", _services(retail, ai), Session())
    designs = [t for t in ai.asked if t.startswith("REQUEST:")]
    assert len(designs) == 2 and "is not a dashboard" in designs[1] and "YOUR PREVIOUS ANSWER" in designs[1]
    assert len(payload["dashboard_tiles"]) == 6
    # Asked again, still thin: the one tile it has is built, never a third call.
    ai = AI(THIN)
    payload = answer_question("Create a revenue dashboard", _services(retail, ai), Session())
    assert len(ai.prompts) == 2 and [t["title"] for t in payload["dashboard_tiles"]] == ["Net amount"]


def test_a_list_of_what_the_reader_needs_is_built_as_named(retail):
    """"I need a dashboard with net amount, orders": one tile each, never asked to be more."""
    ai = AI({"name": "Mine", "tiles": [
        {"title": "Net amount", "kind": "number", "question": "Net amount this year"},
        {"title": "Orders", "kind": "number", "question": "Number of orders this year"}]})
    payload = answer_question("I need a dashboard with net amount, orders", _services(retail, ai), Session())
    assert len(ai.prompts) == 1 and [t["title"] for t in payload["dashboard_tiles"]] == ["Net amount", "Orders"]
    assert payload["tabs"] == ["Overview"] and {t["tab"] for t in payload["dashboard_tiles"]} == {"Overview"}


def test_the_design_is_asked_for_by_kind_and_on_tabs():
    for words in ("executive", "subject", "whole business", "list of what the reader needs", "Overview",
                  "its own tab", "at most 6 tabs and at most 16 tiles"):
        assert words in dashboards.RULES, words


def test_the_chat_is_told_how_far_it_is_never_what_the_tiles_show(retail):
    seen: list[dict] = []
    services = _services(retail, AI())
    services.progress = seen.append
    payload = answer_question("Build me a dashboard of sales", services, Session())
    assert payload["type"] == "dashboard_built" and len(seen) == 8            # once designed, then once per tile
    assert all(c["type"] == "dashboard_building" and set(c) == {"type", "engine", "question", "building"}
               for c in seen)
    first, last = seen[0]["building"], seen[-1]["building"]
    assert first == {"name": "Sales overview", "count": 7, "tabs": ["Overview", "Stores", "Customers"],
                     "built": 0, "failed": 0, "pending": 7}
    pending = [c["building"]["pending"] for c in seen]
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
    for tile in payload["dashboard_tiles"]:
        chat = answer_question(tile["question"], _services(retail, AI()), Session())
        from core2.plan.ir import Plan

        redrawn = _compute("", Plan.model_validate(tile["plan"]), silent, Context(today=TODAY), question_id="",
                           started=0.0)
        assert redrawn["export_rows"] == chat["export_rows"], tile["title"]


def test_the_layout_puts_numbers_in_a_row_trends_across_and_breakdowns_side_by_side(retail):
    payload = answer_question("Build me a dashboard of sales", _services(retail, AI()), Session())
    assert payload["tabs"] == ["Overview", "Stores", "Customers"]        # "customers" is the Customers tab
    places = {t["title"]: t["layout"] for t in payload["dashboard_tiles"]}
    assert places["Net amount"] == {"grid_x": 0, "grid_y": 0, "grid_w": 6, "grid_h": 3, "tab": 0}
    assert places["Orders"]["grid_x"] == 6 and places["Orders"]["grid_y"] == 0
    assert places["Net amount by month"] == {"grid_x": 0, "grid_y": 3, "grid_w": 12, "grid_h": 10, "tab": 0}
    # A breakdown alone in its row spans it, never half a row beside a gap.
    assert places["Net amount by store"] == {"grid_x": 0, "grid_y": 0, "grid_w": 12, "grid_h": 10, "tab": 1}
    assert places["Net amount by segment"] == {"grid_x": 0, "grid_y": 0, "grid_w": 12, "grid_h": 10, "tab": 2}
    pair = [dashboards.Tile(f"b{i}", "breakdown", "q") for i in range(3)]
    dashboards.lay_out(pair)
    assert [(t.layout["grid_x"], t.layout["grid_w"]) for t in pair] == [(0, 6), (6, 6), (0, 12)]
    assert places["Top customers"] == {"grid_x": 0, "grid_y": 10, "grid_w": 12, "grid_h": 10, "tab": 2}
    assert {t["title"]: t["tab"] for t in payload["dashboard_tiles"]}["Top customers"] == "Customers"


def test_tabs_go_overview_first_at_most_six_and_twelve_tiles_a_tab():
    tiles = [dashboards.Tile("a", "breakdown", "q", tab="Revenue"), dashboards.Tile("b", "number", "q"),
             dashboards.Tile("c", "trend", "q", tab="overview")]
    assert dashboards.lay_out(tiles) == ["overview", "Revenue"]          # a tile with none goes on the first
    assert [(t.tab, t.layout["tab"]) for t in tiles] == [("Revenue", 1), ("overview", 0), ("overview", 0)]
    many = [dashboards.Tile(f"t{i}", "breakdown", "q") for i in range(15)]
    assert dashboards.lay_out(many) == ["Overview", "Overview 2"]
    assert [t.layout["tab"] for t in many] == [0] * 12 + [1] * 3
    areas = [dashboards.Tile(f"t{i}", "number", "q", tab=f"Area {i}") for i in range(8)]
    assert dashboards.lay_out(areas) == [f"Area {i}" for i in range(6)]
    assert [t.tab for t in areas[6:]] == ["Area 0", "Area 0"]            # past the sixth: on the first


def test_a_design_of_more_than_sixteen_tiles_keeps_sixteen(retail):
    design = {"name": "Big", "tiles": [{"title": f"t{i}", "kind": "number", "question": "Net amount this year"}
                                       for i in range(20)]}
    name, tiles = dashboards.design(retail[1], "Create an executive dashboard", AI(design))
    assert len(tiles) == dashboards.MOST_ALL == 16


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
        for table in ("pinned_chart", "dashboard_artifact_version", "dashboard_artifact", "portal_user", "client"):
            conn.execute(f"DELETE FROM {table} WHERE account_id=?", (account,))


def test_it_is_saved_at_once_as_a_new_dashboard_on_its_tabs(reader):
    from gateway.core2_bridge import _created

    account, owner, other, payload = reader
    sent = _created(account, {"id": owner, "account_id": account}, dict(payload), "q-1", "thread-9")
    assert sent["type"] == "dashboard_created" and sent["question_id"] == "q-1" and "dashboard_tiles" not in sent
    board = sent["dashboard"]
    assert board["name"] == "Sales overview" and board["url"] == f"/portal/dashboard?dashboard_id={board['id']}"
    assert sent["tiles"] == 6
    assert [(t["name"], t["tiles"]) for t in sent["tabs"]] == [("Overview", 3), ("Stores", 1), ("Customers", 2)]
    assert sent["tabs"][1]["url"] == f"{board['url']}&tab=Stores"
    assert [t["title"] for t in sent["not_built"]] == ["Unicorns"] and sent["not_built"][0]["definable"]
    saved = store.get_dashboard(board["id"], owner, account)
    assert json.loads(saved["tabs_json"]) == ["Overview", "Stores", "Customers"]
    assert saved["thread_id"] == "thread-9"
    assert [v["change_summary"] for v in store.list_dashboard_versions(board["id"], owner, account)] == [
        "Built from your request"]
    charts = store.list_dashboard_charts(board["id"], owner)
    assert {c["title"]: c["dashboard_tab"] for c in charts} == {
        "Net amount": "Overview", "Orders": "Overview", "Net amount by month": "Overview",
        "Net amount by store": "Stores", "Net amount by segment": "Customers", "Top customers": "Customers"}
    assert all(json.loads(c["display_config"])["core2_plan"] for c in charts)
    assert store.get_dashboard(board["id"], other, account) is None               # theirs alone
    # Asked for again: a second dashboard, named so the two are told apart.
    again = _created(account, {"id": owner, "account_id": account}, dict(payload), "q-2")
    assert again["dashboard"]["name"] == "Sales overview 2" and again["dashboard"]["id"] != board["id"]


def test_a_tile_on_a_tab_the_dashboard_has_goes_below_what_it_has(reader):
    account, owner, other, payload = reader
    mine = store.create_dashboard(account, owner, "thread", "Mine")
    store.pin_chart(owner, account, "Already here", "q", "SELECT 1", "kpi", None, dashboard_id=mine["id"])
    tiles = [dict(t) for t in payload["dashboard_tiles"]]
    store.save_built_dashboard(account, owner, name="x", tiles=tiles, db_config_id=None, dashboard_id=mine["id"],
                               tab_names=payload["tabs"])
    charts = store.list_dashboard_charts(mine["id"], owner)
    first = next(c for c in charts if c["title"] == "Already here")
    overview = [c for c in charts if c["dashboard_tab"] == "Overview" and c["title"] != "Already here"]
    assert len(overview) == 3 and min(c["grid_y"] for c in overview) >= first["grid_y"] + first["grid_h"]
    assert json.loads(store.get_dashboard(mine["id"], owner, account)["tabs_json"]) == [
        "Overview", "Stores", "Customers"]
    assert store.save_built_dashboard(account, other, name="x", tiles=tiles, db_config_id=None,
                                      dashboard_id=mine["id"]) is None                # not theirs to change


def test_a_dashboard_needs_a_reader_to_be_saved_as():
    from gateway.core2_bridge import _created

    sent = _created("acct", None, {"dashboard_tiles": [{"title": "x"}]}, "q-1")
    assert sent["type"] == "message" and sent["state"] == "failed" and "Sign in" in sent["content"]


def test_the_dashboards_filters_change_each_tiles_sql(retail):
    from core2.compile.compiler import compile_query
    from core2.plan.ir import Plan
    from core2.resolve.resolver import Context, resolve
    from portal.routes import _view_plan

    built, model = retail
    payload = answer_question("Build me a dashboard of sales", _services(retail, AI()), Session())
    for tile in payload["dashboard_tiles"]:
        viewed, _ = _view_plan(tile["plan"], {"period": "last_year", "members": {"region.name": "North"}})
        before = compile_query(resolve(Plan.model_validate(tile["plan"]), model, Context(today=TODAY)), model,
                               "duckdb").sql
        after = compile_query(resolve(Plan.model_validate(viewed), model, Context(today=TODAY)), model, "duckdb").sql
        assert after != before and "'North'" in after, tile["title"]


def _socket(sent: list):
    class Socket:
        async def send_json(self, payload):
            sent.append(payload)

    return SimpleNamespace(thread_id="t", send_lock=asyncio.Lock(), session_id="s"), Socket()


def test_the_chat_says_how_far_it_is_then_where_it_is_on_a_time_of_its_own(reader, retail, monkeypatch):
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
    assert kinds == ["dashboard_building"] * 8 + ["dashboard_created"]
    assert len({p["question_id"] for p in sent}) == 1                 # one card: how far it is, then where it is
    final = sent[-1]
    assert "dashboard_tiles" not in final and store.get_dashboard(final["dashboard"]["id"], owner, account)
    assert len(store.list_dashboard_charts(final["dashboard"]["id"], owner)) == 6
    # The same wait on a question is past its time.
    plain: list[dict] = []
    adapter, socket = _socket(plain)
    asyncio.run(bridge.answer_instead("core2", adapter, socket, account, "Net amount this year", user))
    assert plain[-1]["kind"] == "could_not" and "within 0 seconds" in plain[-1]["answer"]["headline"]


_CARD = dict(functions=["function dashboardMadeHtml(msg)", "function escHtml(str)"],
             preamble="function qbIcon(n){return '<svg></svg>';}")
MADE = {"type": "dashboard_created", "question_id": "q-1", "tiles": 6,
        "dashboard": {"id": 4, "name": "Sales & margin", "url": "/portal/dashboard?dashboard_id=4"},
        "tabs": [{"name": "Overview", "tiles": 3, "url": "/portal/dashboard?dashboard_id=4&tab=Overview"},
                 {"name": "Stores", "tiles": 1, "url": "/portal/dashboard?dashboard_id=4&tab=Stores"},
                 {"name": "Customers", "tiles": 2, "url": "/portal/dashboard?dashboard_id=4&tab=Customers"}],
        "not_built": [{"title": "Fills", "error": "There is no measure called fills.", "definable": True}]}


def _card(msg: dict) -> str:
    pytest.importorskip("dukpy")
    from tests.chat_js import run

    return run(f"JSON.stringify(dashboardMadeHtml({json.dumps(msg)}));", lang="en", **_CARD)


def test_the_card_names_the_dashboard_its_tabs_and_links_to_it():
    markup = _card(MADE)
    assert "Dashboard created" in markup and "“Sales &amp; margin” is ready in your dashboards." in markup
    assert "6 tiles on 3 tabs, built from your request." in markup
    assert 'href="/portal/dashboard?dashboard_id=4&amp;tab=Stores">Stores<span class="dash-made-count">1</span>' in markup
    assert "There is no measure called fills." in markup and 'data-dash-define="Fills"' in markup
    assert 'class="btn btn-primary btn-sm" href="/portal/dashboard?dashboard_id=4"' in markup and "Open dashboard" in markup
    assert "<svg" in markup and "dash-draft" not in markup                    # nothing of the dashboard drawn
    one = _card({**MADE, "tiles": 1, "tabs": MADE["tabs"][:1], "not_built": []})
    assert "1 tile, built from your request." in one and "dash-made-tabs" not in one and "could not" not in one


def test_while_it_is_built_the_card_is_one_line_of_how_far_it_is():
    markup = _card({"type": "dashboard_building", "building": {"name": "Sales overview", "count": 7, "built": 2,
                                                               "failed": 1, "pending": 4}})
    assert "Building your dashboard" in markup and "3 of 7 tiles ready" in markup and 'aria-busy="true"' in markup
    assert 'aria-valuenow="3"' in markup and "width:43%" in markup and "href" not in markup
    designing = _card({"type": "dashboard_building", "building": {}})
    assert "Choosing what it shows…" in designing


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
    # Learn reads it from its name (a cost falling is good news); the admin has the last word.
    assert editor_state(model, cost)["better"] == "lower"
    assert apply_overrides(model, [{"object_key": f"measure:{cost.key}", "field": "better", "value": "higher"}]) == []
    assert model.measures[cost.key].better == "higher" and editor_state(model, model.measures[cost.key])["better"] == "higher"
    pytest.importorskip("dukpy")
    from tests.chat_js import run

    def line(change):
        return run(f"JSON.stringify(_kpiChangeHtml({json.dumps(change)}));", lang="en",
                   functions=["function _kpiChangeHtml(change)", "function escHtml(str)"])

    rising = {"direction": "up", "change": "+$20.00", "pct": "+20.0%", "vs": "May", "before": "$100.00"}
    assert 'class="kpi-change is-bad"' in line({**rising, "good": False})
    assert 'class="kpi-change is-good"' in line({**rising, "good": True})
    assert 'class="kpi-change"' in line({**rising, "good": None}) and 'class="kpi-change"' in line(rising)


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
