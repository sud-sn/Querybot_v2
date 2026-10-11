"""A dashboard is changed by talking: "remove the Returns tab", "make the trends quarterly", "undo".

In the new core's chat a change to a dashboard was answered as a question ("I cannot answer that from this data"):
today's dashboard commands are never reached there. Now the words are read in code against the dashboard the
conversation is on (the one the chat was opened from, the one it just made or changed, or its thread's last), and
each change is one version of it, saying what it was, with Undo beside it. Undo puts the version before back,
tiles taken off included (a version now keeps what each tile asks). Words that only might be about the dashboard
("show net amount by month", "show that as a pie") are a question unless they name what it has; words about a
tab or tile it lacks are told what it has; words that point at several tiles ask which. A dashboard shared to
read is its owner's to change. The card of each change is kept in the thread.

Invented data only.
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
from core2.dashboard_talk import Board, Change, Refused, Tile, Which, period, read
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
M = {"window": {"kind": "last", "unit": "month", "n": 12}}
TILES = [Tile(1, "Net amount", "Overview", "kpi", {"measures": ["net_amount"], "time": M}),
         Tile(2, "Net amount by month", "Overview", "line", {"measures": ["net_amount"], "time": {**M, "grain": "month"}}),
         Tile(3, "Net amount by category", "Overview", "bar",
              {"measures": ["net_amount"], "group_by": ["category.name"], "time": M}),
         Tile(4, "Number of orders by month", "Orders", "line",
              {"measures": ["number_of_orders"], "time": {**M, "grain": "month"}}),
         Tile(5, "Top 10 customers", "Orders", "table",
              {"measures": ["net_amount"], "group_by": ["customer"], "limit": 10, "time": M}),
         Tile(6, "Refund amount by month", "Returns", "line",
              {"measures": ["refund_amount"], "time": {**M, "grain": "month"}})]
BOARD = Board(9, "Sales executive dashboard", ["Overview", "Orders", "Returns"], TILES)


# ── reading the words ───────────────────────────────────────────────────────


@pytest.mark.parametrize("words, on_it, said", [
    ("remove the Returns tab", False, "Removed the Returns tab and its 1 tile"),
    ("Can you remove the returns tab please?", False, "Removed the Returns tab and its 1 tile"),
    ("undo", True, "Undid the last change"),
    ("Undo the last change on the dashboard", False, "Undid the last change"),
    ("Annuler la dernière modification du tableau de bord", False, "Undid the last change"),
    ("make the trends quarterly", True, "Made “Net amount by month”, “Number of orders by month” and "
                                        "“Refund amount by month” quarterly"),
    ("show the trends by week", True, "Made “Net amount by month”, “Number of orders by month” and "
                                      "“Refund amount by month” weekly"),
    ("make the trend on the Orders tab weekly", False, "Made “Number of orders by month” weekly"),
    ("make Net amount by category a pie", True, "Drew “Net amount by category” as a pie"),
    ("rename the dashboard to Board pack", False, "Renamed the dashboard to Board pack"),
    ("call it Board pack", True, "Renamed the dashboard to Board pack"),
    ("rename the Orders tab to Sales", False, "Renamed the Orders tab to Sales"),
    ("remove Top 10 customers", True, "Removed “Top 10 customers”"),
    ("move Net amount by category to the Orders tab", False, "Moved “Net amount by category” to the Orders tab"),
    ("show the dashboard for 2025", False, "Set every tile to 2025"),
    ("set the period to the last 6 months", True, "Set every tile to the last 6 months"),
    ("add a tab called Margin", False, "Added a Margin tab"),
    ("add net amount by region to the Orders tab", False, "Added net amount by region")])
def test_a_change_is_read_against_the_dashboard(words, on_it, said):
    change = read(words, BOARD, on_it=on_it)
    assert isinstance(change, Change) and change.said == said


@pytest.mark.parametrize("words, on_it", [
    ("show net amount by month", True), ("show that as a pie", True), ("remove cancelled orders", True),
    ("what is net amount by region", True), ("undo", False), ("make the trend quarterly", False),
    ("net amount by store this year", True), ("add net amount by region", True),
    ("show credit card sales by month", False), ("show credit card sales by month", True)])
def test_words_that_only_might_be_about_it_are_a_question(words, on_it):
    assert read(words, BOARD, on_it=on_it) is None


def test_a_change_says_what_it_does_to_each_tile():
    change = read("make the trends quarterly", BOARD, on_it=True)
    assert change.tiles[2] == {"display_config": {"core2_plan": {
        "measures": ["net_amount"], "time": {**M, "grain": "quarter"}}}, "title": "Net amount by quarter"}
    tab = read("remove the Returns tab", BOARD, on_it=False)
    assert tab.tabs == ["Overview", "Orders"] and tab.remove == [6]
    renamed = read("rename the Orders tab to Sales", BOARD, on_it=False)
    assert renamed.tabs == ["Overview", "Sales", "Returns"] and renamed.tiles == {
        4: {"dashboard_tab": "Sales", "in_place": True}, 5: {"dashboard_tab": "Sales", "in_place": True}}
    assert read("remove the tile “Top 10 customers”", BOARD, on_it=False).remove == [5]
    every = read("show the dashboard for 2025", BOARD, on_it=False)
    assert every.tiles[1]["display_config"]["core2_plan"]["time"]["window"] == {
        "kind": "between", "start": "2025-01-01", "end": "2025-12-31"}


def test_words_that_point_at_several_tiles_ask_which_and_a_missing_one_is_told_what_there_is():
    which = read("make the trend quarterly", BOARD, on_it=True)
    assert isinstance(which, Which) and which.question == "Which trend do you mean?"
    assert which.options == ["Make Net amount by month quarterly", "Make Number of orders by month quarterly",
                             "Make Refund amount by month quarterly"]
    missing = read("remove the Shipping tab", BOARD, on_it=False)
    assert isinstance(missing, Refused) and missing.text == (
        "Sales executive dashboard has no tab called “Shipping”. Its tabs are Overview, Orders and Returns.")
    only = Board(1, "Mine", ["Overview"], TILES[:1])
    assert isinstance(read("remove the Overview tab", only, on_it=False), Refused)


@pytest.mark.parametrize("words, window", [
    ("2025", {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}),
    ("this year", {"kind": "this", "unit": "year"}), ("last quarter", {"kind": "previous", "unit": "quarter"}),
    ("the last 12 months", {"kind": "last", "unit": "month", "n": 12}), ("year to date", {"kind": "to_date", "unit": "year"}),
    ("all time", {"kind": "all"}), ("someday", None)])
def test_a_period_in_the_readers_words(words, window):
    assert period(words) == window


# ── one version per change, and undo ────────────────────────────────────────


@pytest.fixture()
def owner():
    store.init_db()
    account = f"talk-{uuid.uuid4().hex[:10]}"
    store.upsert_client(account, "portal")
    user = store.create_user(account, "Ada Owner", f"{uuid.uuid4().hex[:8]}@test.com",
                             password="a-password-they-chose")[0]
    other = store.create_user(account, "Bo Other", f"{uuid.uuid4().hex[:8]}@test.com",
                              password="a-password-they-chose")[0]
    yield account, user, other
    with store.get_db() as conn:
        for table in ("pinned_chart", "dashboard_artifact_version", "dashboard_share", "dashboard_artifact",
                      "answer_trace", "portal_user", "client"):
            try:
                conn.execute(f"DELETE FROM {table} WHERE account_id=?", (account,))
            except Exception:     # noqa: BLE001 - a table this store has no account column on
                pass


def _built(account: str, user: int) -> dict:
    tiles = [{"title": t.title, "question": t.title, "chart_type": t.chart_type, "tab": t.tab,
              "layout": {"grid_x": 0, "grid_y": i * 10, "grid_w": 12, "grid_h": 10},
              "display_config": {"core2_plan": t.plan, "title": t.title}} for i, t in enumerate(TILES)]
    return store.save_built_dashboard(account, user, name="Sales executive dashboard", tiles=tiles,
                                      db_config_id=None, tab_names=["Overview", "Orders", "Returns"])


def _state(board_id: int, user: int, account: str) -> tuple:
    board = store.get_dashboard(board_id, user, account)
    charts = store.list_dashboard_charts(board_id, user)
    return board["name"], json.loads(board["tabs_json"]), sorted((c["title"], c["dashboard_tab"]) for c in charts)


def test_each_change_is_one_version_and_undo_puts_the_one_before_back(owner):
    account, user, _ = owner
    board = _built(account, user)
    first = _state(board["id"], user, account)
    ids = {c["title"]: c["id"] for c in store.list_dashboard_charts(board["id"], user)}
    store.change_dashboard(board["id"], user, account, summary="Removed the Returns tab and its 1 tile",
                           tabs=["Overview", "Orders"], remove=[ids["Refund amount by month"]])
    plan = {"measures": ["net_amount"], "time": {**M, "grain": "quarter"}}
    store.change_dashboard(board["id"], user, account, summary="Made “Net amount by month” quarterly",
                           tiles={ids["Net amount by month"]: {"display_config": {"core2_plan": plan},
                                                              "title": "Net amount by quarter"}})
    second = _state(board["id"], user, account)
    assert second[1] == ["Overview", "Orders"] and ("Net amount by quarter", "Overview") in second[2]
    assert [v["change_summary"] for v in store.list_dashboard_versions(board["id"], user, account)] == [
        "Made “Net amount by month” quarterly", "Removed the Returns tab and its 1 tile", "Built from your request"]
    _, what = store.undo_dashboard(board["id"], user, account)
    assert what == "Made “Net amount by month” quarterly"
    assert ("Net amount by month", "Overview") in _state(board["id"], user, account)[2]
    _, what = store.undo_dashboard(board["id"], user, account)        # and the one before: the tab comes back
    assert what == "Removed the Returns tab and its 1 tile"
    assert _state(board["id"], user, account) == first
    redrawn = next(c for c in store.list_dashboard_charts(board["id"], user) if c["title"] == "Refund amount by month")
    assert json.loads(redrawn["display_config"])["core2_plan"] == TILES[5].plan and redrawn["question"]
    assert store.undo_dashboard(board["id"], user, account) is None    # as it was made: nothing more to undo


def test_a_tab_renamed_keeps_its_tiles_where_they_are(owner):
    account, user, _ = owner
    board = _built(account, user)
    before = {c["id"]: (c["grid_x"], c["grid_y"]) for c in store.list_dashboard_charts(board["id"], user)}
    change = read("rename the Orders tab to Sales", Board(board["id"], board["name"], ["Overview", "Orders", "Returns"], [
        Tile(c["id"], c["title"], c["dashboard_tab"], c["chart_type"]) for c in store.list_dashboard_charts(
            board["id"], user)]), on_it=False)
    store.change_dashboard(board["id"], user, account, summary=change.said, tabs=change.tabs, tiles=change.tiles)
    after = store.list_dashboard_charts(board["id"], user)
    assert {c["id"]: (c["grid_x"], c["grid_y"]) for c in after} == before
    assert sorted(c["title"] for c in after if c["dashboard_tab"] == "Sales") == ["Number of orders by month",
                                                                                  "Top 10 customers"]


def test_a_change_to_a_dashboard_not_theirs_or_of_a_tile_not_on_it_is_not_made(owner):
    account, user, other = owner
    board = _built(account, user)
    assert store.change_dashboard(board["id"], other, account, summary="x", name="Mine now") is None
    assert store.change_dashboard(board["id"], user, account, summary="x", remove=[999999]) is None
    assert store.get_dashboard(board["id"], user, account)["name"] == "Sales executive dashboard"


# ── through the chat ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")


def _ai(stable: str, tail: str) -> str:
    asked = re.search(r"^QUESTION: (.*)$", tail, re.MULTILINE)
    if asked and "region" in asked.group(1):
        return json.dumps({"kind": "query", "intent": "breakdown", "measures": ["net_amount"],
                           "group_by": ["region.name"], "time": M})
    raise AssertionError(f"the AI was asked: {tail[:80]}")


class _Chat:
    """The new core's chat on one thread: each request through the bridge, the frames it sent."""

    def __init__(self, account: str, user: int, retail, monkeypatch, *, thread: str = "th-1", selected: int = 0):
        import gateway.core2_bridge as bridge

        self.bridge, self.account, self.user = bridge, account, {"id": user, "account_id": account, "lang": "en"}
        self.asked: list[str] = []

        def portal_answer(account_id, question, portal_user, *, session_key, question_id="", progress=None):
            self.asked.append(question)
            built, model = retail
            services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=_ai,
                                index=MemberIndex(), today=TODAY)
            services.progress = progress
            return answer_question(question, services, Session())

        monkeypatch.setattr("core2.service.portal_answer", portal_answer)
        monkeypatch.setattr(bridge, "_over_limit", lambda *a, **k: "")
        self.adapter = SimpleNamespace(thread_id=thread, send_lock=asyncio.Lock(),
                                       session_id=f"{account}:{user}:thread:{thread}", selected_dashboard_id=selected)

    def say(self, words: str) -> dict:
        sent: list[dict] = []

        class Socket:
            async def send_json(self, payload):
                sent.append(payload)

        asyncio.run(self.bridge.answer_instead("core2", self.adapter, Socket(), self.account, words, self.user))
        return sent[-1]


def test_a_dashboard_made_in_the_chat_is_changed_and_undone_there(owner, retail, monkeypatch):
    account, user, _ = owner
    chat = _Chat(account, user, retail, monkeypatch)
    made = chat.say("Create an executive dashboard")
    board_id = made["dashboard"]["id"]
    tabs = json.loads(store.get_dashboard(board_id, user, account)["tabs_json"])
    assert tabs == ["Overview", "Orders", "Returns", "Sales target"]
    returns = [c for c in store.list_dashboard_charts(board_id, user) if c["dashboard_tab"] == "Returns"]

    changed = chat.say("remove the Returns tab")
    assert changed["type"] == "dashboard_changed" and changed["dashboard"]["id"] == board_id
    assert changed["change"] == f"Removed the Returns tab and its {len(returns)} tiles" and changed["undo"]
    assert json.loads(store.get_dashboard(board_id, user, account)["tabs_json"]) == ["Overview", "Orders",
                                                                                     "Sales target"]
    # Right after, the trends need no "on the dashboard": the conversation is on it.
    changed = chat.say("make the trends quarterly")
    trends = [c for c in store.list_dashboard_charts(board_id, user)
              if json.loads(c["display_config"])["core2_plan"].get("time", {}).get("grain")]
    assert trends and all(json.loads(c["display_config"])["core2_plan"]["time"]["grain"] == "quarter" for c in trends
                          if "lifecycle" not in json.loads(c["display_config"])["core2_plan"])
    assert "Net amount by quarter" in [c["title"] for c in trends]
    assert changed["version"] == store.get_dashboard(board_id, user, account)["version"]

    undone = chat.say("undo")
    assert undone["type"] == "dashboard_changed" and undone["change"].startswith("Undid: Made")
    assert "Net amount by month" in [c["title"] for c in store.list_dashboard_charts(board_id, user)]
    chat.say("undo")
    assert len([c for c in store.list_dashboard_charts(board_id, user) if c["dashboard_tab"] == "Returns"]) == len(
        returns)                                                      # the tab's tiles came back with it
    assert chat.asked == ["Create an executive dashboard"]         # not one change went to the new core

    # A question after the dashboard is a question, and then the trends are words of a question again.
    chat.say("net amount by region")
    assert chat.asked[-1] == "net amount by region"


def test_a_tile_is_added_from_the_answer_to_the_question_it_names(owner, retail, monkeypatch):
    account, user, _ = owner
    chat = _Chat(account, user, retail, monkeypatch)
    board_id = chat.say("Create an executive dashboard")["dashboard"]["id"]
    added = chat.say("add net amount by region to the Orders tab")
    assert added["type"] == "dashboard_changed" and added["change"] == "Added “Net amount by region” to the Orders tab"
    tile = next(c for c in store.list_dashboard_charts(board_id, user) if c["title"] == "Net amount by region")
    assert tile["dashboard_tab"] == "Orders"
    assert json.loads(tile["display_config"])["core2_plan"]["group_by"] == ["region.name"]
    assert store.list_dashboard_versions(board_id, user, account)[0]["change_summary"] == "Added Net amount by region"


def test_a_change_with_no_dashboard_or_to_one_shared_to_read_is_said(owner, retail, monkeypatch):
    account, user, other = owner
    chat = _Chat(account, user, retail, monkeypatch, thread="th-empty")
    said = chat.say("remove the Shipping tab")
    assert said["answer"]["headline"].startswith("This conversation has no dashboard to change.")
    board = _built(account, user)
    store.add_dashboard_share(board["id"], user, account, "user", other)
    theirs = _Chat(account, other, retail, monkeypatch, thread="th-2", selected=board["id"])
    said = theirs.say("remove the Returns tab")
    assert said["answer"]["headline"] == "Sales executive dashboard is shared with you to read: its owner can change it."
    assert json.loads(store.get_dashboard(board["id"], user, account)["tabs_json"]) == ["Overview", "Orders",
                                                                                        "Returns"]
    asked = chat.say("make the trend on the Overview tab quarterly please")     # no dashboard in this thread...
    assert asked["answer"]["headline"].startswith("This conversation has no dashboard")
    mine = _Chat(account, user, retail, monkeypatch, thread="th-3", selected=board["id"])
    which = mine.say("make the trend quarterly")                   # ...opened from it, the trend is one of three
    assert which["answer"]["headline"] == "Which trend do you mean?"
    assert [c["question"] for c in which["follow_up_suggestions"]][0] == "Make Net amount by month quarterly"


def test_the_card_of_each_change_is_kept_in_its_thread(owner, retail, monkeypatch):
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    import portal.routes as pr

    account, user, _ = owner
    chat = _Chat(account, user, retail, monkeypatch, thread="th-kept")
    chat.say("Create an executive dashboard")
    changed = chat.say("rename the dashboard to Board pack")
    assert changed["dashboard"]["name"] == "Board pack"
    app = FastAPI()
    app.include_router(pr.router)
    client = TestClient(app)
    client.cookies.set(pr._COOKIE, pr._sign_session_value(user))
    turns = client.get("/portal/api/history/th-kept").json()["turns"]
    assert [t["payload"]["type"] for t in turns] == ["dashboard_created", "dashboard_changed"]
    assert turns[1]["payload"]["change"] == "Renamed the dashboard to Board pack"


_CARD = dict(functions=["function dashboardChangedHtml(msg)", "function escHtml(str)"],
             preamble="function qbIcon(n){return '<svg></svg>';}")


def test_the_card_says_what_changed_the_version_and_offers_undo():
    pytest.importorskip("dukpy")
    from tests.chat_js import run, source

    msg = {"type": "dashboard_changed", "change": "Removed the Returns tab and its 2 tiles", "version": 3,
           "undo": True, "dashboard": {"id": 4, "name": "Sales executive dashboard", "url": "/portal/dashboard?dashboard_id=4"}}
    markup = run(f"JSON.stringify(dashboardChangedHtml({json.dumps(msg)}));", lang="en", **_CARD)
    assert "Dashboard changed" in markup and "Removed the Returns tab and its 2 tiles" in markup
    assert "Version 3 of “Sales executive dashboard”: its history lists each change." in markup
    assert 'data-dash-undo="1"' in markup and ">Undo<" in markup and 'href="/portal/dashboard?dashboard_id=4"' in markup
    page = source()
    assert "sendChartDrill(t('ui.chat.dashchanged.undo_words'))" in page
    assert "msg.type === 'dashboard_created' || msg.type === 'dashboard_changed'" in page
