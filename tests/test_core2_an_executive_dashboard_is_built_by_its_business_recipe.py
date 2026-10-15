"""An executive dashboard is built by the recipe of the business's kind: the same shape, a proper name, no AI.

"Create an executive dashboard" was designed afresh by the AI each time: two requests on the same data gave two
dashboards of different shapes, under names the AI chose. Now the whole business is built by the recipe of its
kind, read off the workspace's own names (orders and returns are sales, payroll and headcount a workforce,
prescriptions and fills a pharmacy): an Overview of its headline figures over the last 12 months against the 12
before, the analysis its kind is asked for (new and returning customers, the share of receipts on time,
employees at each month end), its first figure by month and its first two by their main grouping; then a tab per
area with a figure by month, by a grouping, and a top 10. Every tile's plan is written in code and checked by the
resolver; nothing is asked of the AI. A request that names a subject is still designed by the AI.

Once built, each headline number is checked against the trend of the same figure over the same period (a sum
adds up to its months, a level is its last month's), and the chat card says so. The card of a dashboard made is
kept in its thread: a reopened thread shows it again.

Invented data only.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import uuid
from types import SimpleNamespace

import pytest

import store
from core2 import dashboards, recipes
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
KINDS = {"retail": "Sales executive dashboard", "purchasing": "Purchasing executive dashboard",
         "bank": "Banking executive dashboard",
         "hr": "Workforce executive dashboard", "subscriptions": "Subscriptions executive dashboard",
         "compounding_pharmacy": "Pharmacy executive dashboard", "inventory": "Inventory executive dashboard",
         "finance": "Finance executive dashboard", "networking": "Network operations executive dashboard",
         "home_services": "Field service executive dashboard"}
ORDER = ["number", "trend", "breakdown", "table"]


@pytest.fixture(scope="module")
def learned():
    cache: dict[str, tuple] = {}

    def get(name: str):
        if name not in cache:
            cache[name] = learn(domains.build(name), "descriptive")
        return cache[name]

    return get


def _no_ai(stable: str, tail: str) -> str:
    raise AssertionError(f"the AI was asked: {tail[:80]}")


def _services(built_model, complete=_no_ai, allowed: set[str] | None = None) -> Services:
    built, model = built_model
    return Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=complete, index=MemberIndex(),
                    today=TODAY, allowed_tables=allowed)


@pytest.mark.parametrize("words, whole, year", [
    ("Create an executive dashboard", True, None), ("Create a dashboard", True, None),
    ("Prepare a CFO dashboard", True, None), ("Build me a KPI dashboard for 2025", True, 2025),
    ("I need an overview dashboard of the whole business", True, None),
    ("Crée-moi un tableau de bord de direction", True, None),
    ("Create a revenue dashboard", False, None), ("Build me a sales overview dashboard", False, None),
    ("I'd like a dashboard of fills by pharmacy", False, None)])
def test_a_request_for_the_whole_business_is_told_from_a_subject(words, whole, year):
    assert recipes.whole_business(dashboards.asked_for(words)) == (whole, year)


@pytest.mark.parametrize("name", sorted(KINDS))
def test_each_business_gets_its_recipe_the_same_shape_and_a_proper_name_with_no_ai(name, learned):
    payload = answer_question("Create an executive dashboard", _services(learned(name)), Session())
    assert payload["type"] == "dashboard_built" and payload["name"] == KINDS[name]
    assert payload["not_built"] == []                      # each tile's plan was checked before it was built
    tiles = payload["dashboard_tiles"]
    # As many areas as the business has, up to the recipe's limits: tabs of planned tiles, no AI call each.
    assert payload["tabs"][0] == "Overview" and 2 <= len(payload["tabs"]) <= recipes.MOST_TABS
    assert len(tiles) <= recipes.MOST_TILES
    assert {t["tab"] for t in tiles} == set(payload["tabs"])
    overview = [t["kind"] for t in tiles if t["tab"] == "Overview"]
    assert overview.count("number") >= 2 and "trend" in overview and "breakdown" in overview
    assert overview == sorted(overview, key=ORDER.index)   # the numbers lead
    for tab in payload["tabs"][1:]:                        # each area: its figures by month, by a grouping, a top 10
        assert {t["kind"] for t in tiles if t["tab"] == tab} <= {"trend", "breakdown", "table"}
    # Every headline total that has a trend beside it is that trend's sum (or its last month, for a level).
    assert payload["checks"] and all(c["agree"] for c in payload["checks"]), payload["checks"]


def test_each_kind_leads_with_its_own_figures_and_analysis(learned):
    def overview(name: str) -> list[tuple[str, str]]:
        payload = answer_question("Create an executive dashboard", _services(learned(name)), Session())
        return [(t["kind"], t["title"]) for t in payload["dashboard_tiles"] if t["tab"] == "Overview"]

    sales = overview("retail")
    assert [title for kind, title in sales if kind == "number"] == [
        "Net amount", "Number of orders", "Refund amount", "Gross amount"]
    assert ("trend", "New, returning and lapsed customers by month") in sales
    assert ("trend", "Net amount by month") in sales and ("breakdown", "Net amount by category") in sales
    workforce = overview("hr")
    assert [title for kind, title in workforce if kind == "number"] == ["Headcount", "FTE", "Gross pay",
                                                                        "Overtime hours"]
    assert ("trend", "Employees at each month end") in workforce
    assert ("breakdown", "Headcount by department") in workforce
    assert ("number", "Share of receipts on time") in overview("purchasing")


def test_the_same_data_gives_the_same_dashboard(learned):
    first, second = (answer_question("Create an executive dashboard", _services(learned("retail")), Session())
                     for _ in range(2))
    assert [(t["title"], t["tab"], t["plan"]) for t in first["dashboard_tiles"]] == [
        (t["title"], t["tab"], t["plan"]) for t in second["dashboard_tiles"]]


def test_a_year_named_is_counted_against_the_year_before_and_french_is_answered_in_french(learned):
    payload = answer_question("Crée-moi un tableau de bord de direction pour 2025", _services(learned("retail")),
                              Session())
    assert payload["name"] == "Tableau de bord de direction — Ventes 2025"
    assert payload["tabs"][0] == "Vue d'ensemble"
    number = next(t for t in payload["dashboard_tiles"] if t["kind"] == "number")
    assert number["plan"]["time"] == {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}
    assert any(t["title"] == "Net amount par mois" for t in payload["dashboard_tiles"])
    assert all(c["agree"] for c in payload["checks"])


def test_each_headline_is_a_number_card_against_the_12_months_before(learned):
    """A number asked with its comparison is drawn as a comparison (a table of two columns on a dashboard): the
    headline asks for the figure alone, and its card says how it moved against the 12 months before."""
    payload = answer_question("Create an executive dashboard", _services(learned("retail")), Session())
    numbers = [t for t in payload["dashboard_tiles"] if t["kind"] == "number"]
    assert numbers and all(t["chart_type"] == "kpi" for t in numbers)
    number = next(t for t in numbers if t["title"] == "Net amount")
    assert number["plan"]["time"] == {"window": {"kind": "last", "unit": "month", "n": 12}}
    assert number["question"] == "Net amount over the last 12 months"
    built, model = learned("retail")
    chat = answer_question(number["question"], _services(learned("retail"), lambda s, t: json.dumps(
        {"kind": "query", **number["plan"]})), Session())
    assert chat["kpi"]["change"] and chat["kpi"]["trend"]


def test_a_reader_is_built_only_what_their_tables_hold(learned):
    built, model = learned("retail")
    allowed = {k for k in model.tables if not k.endswith(".returns")}
    payload = answer_question("Create an executive dashboard", _services(learned("retail"), allowed=allowed),
                              Session())
    assert "Returns" not in payload["tabs"]
    returns = {m.slug for m in model.measures.values() if m.table.endswith(".returns")}
    assert not any(set(t["plan"].get("measures") or []) & returns for t in payload["dashboard_tiles"])


def test_a_subject_is_still_designed_by_the_ai(learned):
    asked: list[str] = []

    def ai(stable: str, tail: str) -> str:
        asked.append(tail)
        if tail.startswith("REQUEST:"):
            return json.dumps({"name": "Revenue", "tiles": [
                {"title": "Net amount", "kind": "number", "question": "Net amount this year"}]})
        return json.dumps({"kind": "query", "measures": ["net_amount"],
                           "time": {"window": {"kind": "this", "unit": "year"}}})

    payload = answer_question("I need a dashboard with net amount, orders", _services(learned("retail"), ai),
                              Session())
    assert asked[0].startswith("REQUEST:") and payload["name"] == "Revenue"


def test_names_that_say_no_kind_of_business_get_the_plain_name(learned):
    _, model = learned("retail")
    blank = model.model_copy(deep=True)
    for i, t in enumerate(blank.tables.values()):
        t.business_name = f"Table {i}"
    for i, m in enumerate(blank.measures.values()):
        m.business_name = f"Figure {i}"
    assert recipes.kind_of(blank) is recipes.GENERIC
    name, tiles = recipes.executive(blank, today=TODAY)
    assert name == "Executive dashboard" and tiles


# ── the check of each headline total against its trend ─────────────────────


def _tile(kind: str, plan: dict, rows: list[dict], title: str = "") -> dashboards.Tile:
    tile = dashboards.Tile(title or kind, kind, "q")
    tile.plan, tile.payload = plan, {"data": {"rows": rows}}
    return tile


YEAR = {"window": {"kind": "last", "unit": "month", "n": 12}}


def test_a_total_that_is_not_the_sum_of_its_trend_is_said(learned):
    _, model = learned("retail")
    number = _tile("number", {"measures": ["net_amount"], "time": {**YEAR, "compare": {"kind": "previous_period"}}},
                   [{"net_amount": 100.0}], "Net amount")
    trend = _tile("trend", {"measures": ["net_amount"], "time": {**YEAR, "grain": "month"}},
                  [{"period": "2026-01-01", "net_amount": 60.0}, {"period": "2026-02-01", "net_amount": 30.0}],
                  "Net amount by month")
    (check,) = dashboards.checked([number, trend], model)
    assert check == {"number": "Net amount", "trend": "Net amount by month", "tab": "", "how": "sum",
                     "agree": False, "total": 100.0, "of_trend": 90.0}
    trend.payload["data"]["rows"].append({"period": "2026-03-01", "net_amount": 10.0})
    assert dashboards.checked([number, trend], model)[0]["agree"]
    # Another period, another filter, or a count of distinct orders: nothing to check against.
    other = _tile("trend", {"measures": ["net_amount"], "time": {"window": {"kind": "this", "unit": "year"},
                                                                  "grain": "month"}}, [{"net_amount": 100.0}])
    assert dashboards.checked([number, other], model) == []
    orders = _tile("number", {"measures": ["number_of_orders"], "time": YEAR}, [{"number_of_orders": 9}])
    by_month = _tile("trend", {"measures": ["number_of_orders"], "time": {**YEAR, "grain": "month"}},
                     [{"number_of_orders": 5}, {"number_of_orders": 5}])
    assert dashboards.checked([orders, by_month], model) == []


def test_a_level_is_its_last_months(learned):
    _, model = learned("hr")
    number = _tile("number", {"measures": ["headcount"], "time": YEAR}, [{"headcount": 443}])
    trend = _tile("trend", {"measures": ["headcount"], "time": {**YEAR, "grain": "month"}},
                  [{"headcount": 400}, {"headcount": 443}])
    (check,) = dashboards.checked([number, trend], model)
    assert check["how"] == "last" and check["agree"]


_CARD = dict(functions=["function dashboardMadeHtml(msg)", "function escHtml(str)"],
             preamble="function qbIcon(n){return '<svg></svg>';}")
MADE = {"type": "dashboard_created", "question_id": "q-1", "tiles": 4,
        "dashboard": {"id": 4, "name": "Sales executive dashboard", "url": "/portal/dashboard?dashboard_id=4"},
        "tabs": [{"name": "Overview", "tiles": 4, "url": "/portal/dashboard?dashboard_id=4&tab=Overview"}],
        "not_built": []}


def _card(msg: dict) -> str:
    pytest.importorskip("dukpy")
    from tests.chat_js import run

    return run(f"JSON.stringify(dashboardMadeHtml({json.dumps(msg)}));", lang="en", **_CARD)


def test_the_card_says_the_totals_were_checked_or_which_did_not_match():
    agree = {"number": "Net amount", "trend": "Net amount by month", "agree": True, "total": 90, "of_trend": 90}
    assert "Checked: 2 headline totals match their trends." in _card({**MADE, "checks": [agree, agree]})
    assert "Checked: the headline total matches its trend." in _card({**MADE, "checks": [agree]})
    amiss = _card({**MADE, "checks": [agree, {**agree, "agree": False, "total": 100, "of_trend": 90}]})
    assert "“Net amount” is 100, but “Net amount by month” comes to 90." in amiss and "Checked" not in amiss
    assert "Checked" not in _card(MADE)


# ── the card of a dashboard made, kept in its thread ────────────────────────


def _portal(user_id: int):
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    import portal.routes as pr

    app = FastAPI()
    app.include_router(pr.router)
    client = TestClient(app)
    client.cookies.set(pr._COOKIE, pr._sign_session_value(user_id))
    return client


@pytest.fixture()
def reader():
    store.init_db()
    account = f"executive-{uuid.uuid4().hex[:10]}"
    store.upsert_client(account, "portal")
    owner = store.create_user(account, "Ada Owner", f"{uuid.uuid4().hex[:8]}@test.com",
                              password="a-password-they-chose")[0]
    yield account, owner
    with store.get_db() as conn:
        for table in ("pinned_chart", "dashboard_artifact_version", "dashboard_artifact", "answer_trace",
                      "portal_user", "client"):
            conn.execute(f"DELETE FROM {table} WHERE account_id=?", (account,))


def test_the_card_of_a_dashboard_made_is_kept_in_its_thread(reader, learned, monkeypatch):
    import gateway.core2_bridge as bridge

    account, owner = reader

    def portal_answer(account_id, question, portal_user, *, session_key, question_id="", progress=None):
        services = _services(learned("retail"))
        services.progress = progress
        return answer_question(question, services, Session())

    monkeypatch.setattr("core2.service.portal_answer", portal_answer)
    monkeypatch.setattr(bridge, "_over_limit", lambda *a, **k: "")
    sent: list[dict] = []

    class Socket:
        async def send_json(self, payload):
            sent.append(payload)

    adapter = SimpleNamespace(thread_id="th-1", send_lock=asyncio.Lock(), session_id=f"{account}:{owner}:thread:th-1")
    user = {"id": owner, "account_id": account, "lang": "en"}
    asyncio.run(bridge.answer_instead("core2", adapter, Socket(), account, "Create an executive dashboard", user))
    made = sent[-1]
    assert made["type"] == "dashboard_created" and made["dashboard"]["name"] == "Sales executive dashboard"
    assert made["checks"] and all(c["agree"] for c in made["checks"])
    client = _portal(owner)
    response = client.get("/portal/api/history/th-1")
    assert response.status_code == 200, response.text
    (turn,) = response.json()["turns"]
    assert turn["question"] == "Create an executive dashboard"
    again = turn["payload"]
    assert again["type"] == "dashboard_created" and again["dashboard"] == made["dashboard"]
    assert again["tabs"] == made["tabs"] and again["checks"] == made["checks"] and "pin_token" not in again
    listed = {item["thread_id"]: item for item in client.get("/portal/api/history").json()["items"]}
    assert listed["th-1"]["question"] == "Create an executive dashboard"


def test_the_page_draws_a_kept_card_as_the_card_it_was():
    from tests.chat_js import source

    page = source()
    restore = page[page.index("async function restoreServerThread"):]
    restore = restore[:restore.index("\n}\n")]
    assert "turn.payload?.type === 'dashboard_created'" in restore and "appendDashboardMade(turn.payload" in restore
    replay = page[page.index("function replayCachedEntry"):]
    replay = replay[:replay.index("\n}\n")]
    assert "kind === 'dashboard_created'" in replay
    made = page[page.index("function appendDashboardMade"):]
    made = made[:made.index("\n}\n")]
    assert "pushChatCache({ kind: msg.type" in made
