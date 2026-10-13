"""A dashboard is asked for in the reader's own words, and the catalogue of recipes behind it is never shown.

"New dashboard" opened a picker of 24 recipes ("Start from a recipe": ready, partly, not for this data). Readers
are not to see the catalogue: they say what they need in the chat, and the recipe their words name is built in
code, unseen. So:

- "New dashboard" opens the chat, asking the reader to describe the dashboard; the picker, its route and the "Your
  request also fits these dashboards" buttons are gone.
- A request is read however it is said: "Sales team dashboard" (no verb), "can I get a dashboard of returns by
  courier", "help me build a dashboard for finance", "let's create a CFO dashboard", "tableau de bord des ventes".
  Words about dashboards that ask for none ("how many dashboards do I have", "open sales dashboard", "the revenue
  dashboard is broken") stay questions.
- "Create a dashboard using this", "turn this into a dashboard": a dashboard around the answer on screen, its
  figure's number, trend, breakdowns and top lists over the answer's period and under its filters, in code, the
  answer itself one of its tiles. Nothing on screen is said.
- "Add this chart to my dashboard", "pin this", "save that to Board pack", "put it on the Orders tab", "add this to
  a new dashboard": the answer on screen goes on the dashboard the words name, else the one the conversation is
  on, else the reader's only one; with several and none named, the chat asks which, a button each.
- A dashboard may be named in a change: "add net amount by region to the Board pack dashboard" changes that one.

Invented data only.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import pathlib
import re
import uuid
from types import SimpleNamespace

import pytest

import store
from core2 import dashboard_talk as talk
from core2.dashboards import asked_for, language
from core2.plan.ir import Plan
from core2.plan.values import MemberIndex
from core2.service import Services, Session, _answer_question, _session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
ROOT = pathlib.Path(__file__).resolve().parents[1]
YEAR = {"window": {"kind": "to_date", "unit": "year"}}


# ── reading the words ───────────────────────────────────────────────────────


@pytest.mark.parametrize("words, about", [
    ("create a sales dashboard", "sales"),
    ("Create a dashboard for the sales team", "the sales team"),
    ("Sales team dashboard", "Sales team"),
    ("a dashboard for my store managers", "my store managers"),
    ("CFO dashboard", "CFO"),
    ("executive dashboard please", "executive"),
    ("a new dashboard", ""),
    ("dashboard for 2025", "2025"),
    ("can I get a dashboard of returns by courier", "returns by courier"),
    ("help me build a dashboard for finance", "finance"),
    ("let's create a CFO dashboard", "CFO"),
    ("we need a dashboard showing net sales by store and month", "net sales by store and month"),
    ("create a dashboard for this year", "this year"),
    ("tableau de bord des ventes", "ventes"),
    ("create a dashboard using this", "this"),
    ("make a dashboard from this chart", "this"),
    ("build a dashboard with this", "this"),
    ("make a dashboard out of this", "this"),
    ("turn this into a dashboard", "this"),
    ("make this a dashboard", "this"),
    ("Crée-moi un tableau de bord à partir de ça", "this"),
    ("transforme ça en tableau de bord", "this")])
def test_a_request_for_a_dashboard_is_read_however_it_is_said(words, about):
    assert asked_for(words) == about


@pytest.mark.parametrize("words", ["dashboards", "my dashboard", "how many dashboards do I have",
                                   "open sales dashboard", "is the dashboard updated", "Revenue dashboard is broken",
                                   "the numbers on the sales dashboard are wrong", "show me the dashboard",
                                   "add this chart to my dashboard", "remove the trend from the dashboard"])
def test_words_about_dashboards_that_ask_for_none_stay_questions(words):
    assert asked_for(words) is None


def test_a_request_without_a_verb_is_built_in_its_own_language():
    assert language("tableau de bord des ventes") == "fr" and language("Sales team dashboard") == "en"


@pytest.mark.parametrize("words, new, board, tab", [
    ("add this chart to my dashboard", False, "", ""),
    ("Pin this, please", False, "", ""),
    ("pin it", False, "", ""),
    ("add it to the dashboard", False, "", ""),
    ("save that to Board pack", False, "Board pack", ""),
    ("add this to the Board pack dashboard", False, "Board pack", ""),
    ("put it on the Orders tab", False, "", "Orders"),
    ("add this to the Overview tab of Board pack", False, "Board pack", "Overview"),
    ("add this to a new dashboard", True, "", ""),
    ("add this to a new dashboard called Weekly", True, "Weekly", ""),
    ("ajoute ça au tableau de bord", False, "", ""),
    ("épingle ce graphique", False, "", "")])
def test_the_answer_on_screen_added_is_read_with_where_it_goes(words, new, board, tab):
    reading = talk.this_answer(words)
    assert reading is not None and (reading.new, reading.board, reading.tab) == (new, board, tab)


@pytest.mark.parametrize("words", ["add that region", "include those", "keep these", "put this",
                                   "save this as a pdf", "add net amount by region to the Orders tab"])
def test_words_about_the_rows_are_never_the_answer_added(words):
    assert talk.this_answer(words) is None


def test_a_dashboard_named_in_a_change_is_found_and_the_words_read_against_it():
    boards = ["Board pack", "Sales overview", "Sales executive dashboard"]
    assert talk.naming("add net amount by region to the Board pack dashboard", boards) == (
        "Board pack", "add net amount by region to the dashboard")
    assert talk.naming('rename the "Board pack" dashboard to Q3 pack', boards) == (
        "Board pack", "rename the dashboard to Q3 pack")
    assert talk.naming("remove the trend from the Sales executive dashboard", boards) == (
        "Sales executive dashboard", "remove the trend from the dashboard")
    assert talk.naming("remove the trend from the dashboard", boards) is None
    assert talk.board_named("board", boards) == "Board pack" and talk.board_named("sales", boards) is None


# ── a dashboard around the answer on screen ─────────────────────────────────


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")


def _services(retail, plans: list[dict]) -> Services:
    built, model = retail

    def complete(stable: str, tail: str) -> str:
        if not plans:
            raise AssertionError(f"the AI was asked: {tail[:80]}")
        return json.dumps({"kind": "query", **plans.pop(0)})

    return Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=complete, index=MemberIndex(),
                    today=TODAY)


def test_a_dashboard_using_this_is_built_around_the_answer_with_no_ai(retail):
    north = {"field": "region.name", "op": "eq", "values": ["North"]}
    services = _services(retail, [{"intent": "breakdown", "measures": ["net_amount"], "group_by": ["store.name"],
                                   "filters": [north], "time": YEAR}])
    session = Session()
    answer = answer_question('net amount by store in "North" this year', services, session)
    assert answer["chart"]["title"] == "Net amount by store, for region North"

    made = answer_question("create a dashboard using this", services, session)   # the AI is not asked again
    assert made["type"] == "dashboard_built" and made["name"] == "Net amount overview — North"
    assert "other_recipes" not in made and made["not_built"] == []
    tiles = {t["title"]: t for t in made["dashboard_tiles"]}
    assert "Net amount by store, for region North" in tiles                      # the answer itself
    assert {"Net amount", "Top 10 customers by net amount"} <= set(tiles)
    assert any(t["kind"] == "trend" for t in tiles.values())
    assert all(north in t["plan"]["filters"] for t in tiles.values())           # under the answer's filter
    assert all(t["plan"]["time"]["window"] == YEAR["window"] for t in tiles.values() if t["kind"] != "trend")
    assert not [t for t in tiles.values() if t["plan"].get("group_by") == ["region.name"]]   # kept to one region


def test_a_dashboard_around_a_short_period_shows_its_trend_over_the_year(retail):
    month = {"window": {"kind": "last", "unit": "month", "n": 1}}
    services = _services(retail, [{"intent": "value", "measures": ["number_of_orders"], "time": month}])
    session = Session()
    answer_question("orders last month", services, session)
    made = answer_question("turn this into a dashboard", services, session)
    trend = next(t for t in made["dashboard_tiles"] if t["kind"] == "trend")
    assert trend["plan"]["time"]["window"] == {"kind": "last", "unit": "month", "n": 12}
    assert next(t for t in made["dashboard_tiles"] if t["kind"] == "number")["plan"]["time"]["window"] == month["window"]


def test_an_answer_the_catalogue_cannot_build_around_is_designed_from_names_never_its_members(retail):
    """When the answer's figure gives no dashboard in code, the AI designs one: told what the answer measures and
    splits by, never its title, which names the member it was kept to."""
    from core2 import dashboards

    _, model = retail
    asked: list[str] = []

    def ai(stable: str, tail: str) -> str:
        asked.append(tail)
        return json.dumps({"name": "Stores", "tiles": [
            {"title": "Net amount", "kind": "number", "question": "Net amount this year"},
            {"title": "Net amount by month", "kind": "trend", "question": "Net amount by month this year"},
            {"title": "Net amount by category", "kind": "breakdown", "question": "Net amount by category"}]})

    north = {"field": "region.name", "op": "eq", "values": ["North"]}
    shown = dashboards.Shown("net amount by store in North", {"measures": ["net_amount"], "group_by": ["store.name"],
                                                              "filters": [north], "time": YEAR},
                             "Net amount by store, for region North")
    sales = model.measures[next(k for k, m in model.measures.items() if m.slug == "net_amount")].table
    name, tiles = dashboards.around(model, shown, ai, today=TODAY,
                                    allowed={t for t in model.tables if t != sales})     # no dashboard in code
    assert asked and "North" not in asked[0] and "a dashboard about net amount by" in asked[0]
    assert tiles[0].title == "Net amount by store, for region North" and tiles[0].planned == shown.plan


def test_a_dashboard_using_this_with_nothing_on_screen_says_so(retail):
    made = answer_question("create a dashboard using this", _services(retail, []), Session())
    assert made["kind"] == "dashboard_refused" and "no answer on screen" in made["answer"]["headline"]


# ── the answer on screen added, through the chat ────────────────────────────


@pytest.fixture()
def owner():
    store.init_db()
    account = f"asked-{uuid.uuid4().hex[:10]}"
    store.upsert_client(account, "portal")
    user = store.create_user(account, "Ada Owner", f"{uuid.uuid4().hex[:8]}@test.com",
                             password="a-password-they-chose")[0]
    yield account, user
    with store.get_db() as conn:
        for table in ("pinned_chart", "dashboard_artifact_version", "dashboard_share", "dashboard_artifact",
                      "answer_trace", "portal_user", "client"):
            try:
                conn.execute(f"DELETE FROM {table} WHERE account_id=?", (account,))
            except Exception:     # noqa: BLE001 - a table this store has no account column on
                pass


def _ai(stable: str, tail: str) -> str:
    asked = re.search(r"^QUESTION: (.*)$", tail, re.MULTILINE)
    if asked and "region" in asked.group(1):
        return json.dumps({"kind": "query", "intent": "breakdown", "measures": ["net_amount"],
                           "group_by": ["region.name"], "time": YEAR})
    raise AssertionError(f"the AI was asked: {tail[:80]}")


class _Chat:
    """The new core's chat on one thread, its conversation kept as the portal keeps it."""

    def __init__(self, account: str, user: int, retail, monkeypatch, *, thread: str):
        import gateway.core2_bridge as bridge

        self.bridge, self.account, self.user = bridge, account, {"id": user, "account_id": account, "lang": "en"}
        self.asked: list[str] = []
        built, model = retail

        def services() -> Services:
            return Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=_ai, index=MemberIndex(),
                            today=TODAY)

        def portal_answer(account_id, question, portal_user, *, session_key, question_id="", progress=None):
            self.asked.append(question)
            kept = services()
            kept.progress = progress
            return answer_question(question, kept, _session(session_key))

        def portal_replay(account_id, plan_data, portal_user, *, question=""):
            # As the portal's replay: the answer drawn from the plan, which it does not give back.
            payload = _answer_question(question, services(), Session(), planned=Plan.model_validate(plan_data))
            payload.pop("plan", None)
            return payload

        monkeypatch.setattr("core2.service.portal_answer", portal_answer)
        monkeypatch.setattr("core2.service.portal_replay", portal_replay)
        monkeypatch.setattr(bridge, "_over_limit", lambda *a, **k: "")
        self.adapter = SimpleNamespace(thread_id=thread, send_lock=asyncio.Lock(),
                                       session_id=f"{account}:{user}:{thread}", selected_dashboard_id=0)

    def say(self, words: str) -> dict:
        sent: list[dict] = []

        class Socket:
            async def send_json(self, payload):
                sent.append(payload)

        asyncio.run(self.bridge.answer_instead("core2", self.adapter, Socket(), self.account, words, self.user))
        return sent[-1]


def _thread() -> str:
    return f"th-{uuid.uuid4().hex[:8]}"


def _board(account: str, user: int, name: str) -> dict:
    tile = {"title": "Net amount", "question": "Net amount", "chart_type": "kpi", "tab": "Overview",
            "layout": {"grid_x": 0, "grid_y": 0, "grid_w": 4, "grid_h": 3},
            "display_config": {"core2_plan": {"measures": ["net_amount"], "time": YEAR}, "title": "Net amount"}}
    return store.save_built_dashboard(account, user, name=name, tiles=[tile], db_config_id=None,
                                      tab_names=["Overview", "Orders"])


def _titles(board_id: int, user: int) -> dict[str, str]:
    return {c["title"]: c["dashboard_tab"] for c in store.list_dashboard_charts(board_id, user)}


def test_this_answer_goes_on_the_dashboard_the_conversation_made(owner, retail, monkeypatch):
    account, user = owner
    chat = _Chat(account, user, retail, monkeypatch, thread=_thread())
    board_id = chat.say("Create an executive dashboard")["dashboard"]["id"]
    chat.say("net amount by region")
    added = chat.say("add this chart to my dashboard")
    assert added["type"] == "dashboard_changed" and added["dashboard"]["id"] == board_id
    assert added["change"] == "Added “Net amount by region” to the Overview tab"
    tile = next(c for c in store.list_dashboard_charts(board_id, user) if c["title"] == "Net amount by region")
    assert json.loads(tile["display_config"])["core2_plan"]["group_by"] == ["region.name"]
    assert chat.asked == ["Create an executive dashboard", "net amount by region"]   # the words were no question


def test_with_several_dashboards_and_none_named_the_chat_asks_which(owner, retail, monkeypatch):
    account, user = owner
    _board(account, user, "Board pack")
    weekly = _board(account, user, "Weekly review")
    chat = _Chat(account, user, retail, monkeypatch, thread=_thread())
    chat.say("net amount by region")
    which = chat.say("pin this")
    assert which["answer"]["headline"] == "Which dashboard should “Net amount by region” go on?"
    options = [c["question"] for c in which["follow_up_suggestions"]]
    assert options == ["Add this to Weekly review", "Add this to Board pack", "Add this to a new dashboard"]
    added = chat.say("Add this to Weekly review")
    assert added["type"] == "dashboard_changed" and added["dashboard"]["id"] == weekly["id"]
    assert _titles(weekly["id"], user)["Net amount by region"] == "Overview"


def test_this_answer_goes_on_the_tab_or_new_dashboard_the_words_name(owner, retail, monkeypatch):
    account, user = owner
    board = _board(account, user, "Board pack")
    chat = _Chat(account, user, retail, monkeypatch, thread=_thread())
    chat.say("net amount by region")
    chat.say("save it to the Orders tab of Board pack")
    assert _titles(board["id"], user)["Net amount by region"] == "Orders"

    made = chat.say("add this to a new dashboard called Regions")
    assert made["type"] == "dashboard_created" and made["dashboard"]["name"] == "Regions" and made["tiles"] == 1
    assert list(_titles(made["dashboard"]["id"], user)) == ["Net amount by region"]


def test_a_dashboard_they_do_not_have_or_nothing_on_screen_is_said(owner, retail, monkeypatch):
    account, user = owner
    _board(account, user, "Board pack")
    chat = _Chat(account, user, retail, monkeypatch, thread=_thread())
    nothing = chat.say("add this to my dashboard")
    assert nothing["answer"]["headline"].startswith("There is no answer on screen to add")
    chat.say("net amount by region")
    missing = chat.say("add this to the Margins dashboard")
    assert missing["answer"]["headline"] == "You have no dashboard called “Margins”: yours are “Board pack”."
    assert [c["question"] for c in missing["follow_up_suggestions"]] == ["Add this to Board pack",
                                                                         "Add this to a new dashboard"]


def test_a_change_that_names_a_dashboard_changes_that_one(owner, retail, monkeypatch):
    account, user = owner
    board = _board(account, user, "Board pack")
    chat = _Chat(account, user, retail, monkeypatch, thread=_thread())
    made = chat.say("Create an executive dashboard")["dashboard"]["id"]
    before = store.get_dashboard(made, user, account)["version"]
    added = chat.say("add net amount by region to the Board pack dashboard")
    assert added["type"] == "dashboard_changed" and added["dashboard"]["id"] == board["id"]
    assert "Net amount by region" in _titles(board["id"], user)
    assert store.get_dashboard(made, user, account)["version"] == before        # the one the chat made is as it was


# ── the catalogue is never shown ────────────────────────────────────────────


def _portal(user_id: int):
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    import portal.routes as pr

    app = FastAPI()
    app.include_router(pr.router)
    client = TestClient(app)
    client.cookies.set(pr._COOKIE, pr._sign_session_value(user_id))
    return client


def test_new_dashboard_opens_the_chat_and_no_picker_is_left(owner):
    account, user = owner
    client = _portal(user)
    assert client.get("/portal/api/dashboard-recipes").status_code == 404
    page = client.get("/portal/dashboard")
    assert page.status_code == 200
    assert 'href="/portal/chat?new=1&amp;compose=dashboard"' in page.text
    assert "recipeDialog" not in page.text and "Start from a recipe" not in page.text
    chat = (ROOT / "portal/templates/portal_chat.html").read_text()
    assert "data-dash-recipe" not in chat and "other_recipes" not in chat
    assert "ui.chat.compose_dashboard" in chat
