"""A dashboard says what the data cannot give, and never keeps a tile that shows nothing.

A recipe asked for in the chat ("a profit and loss dashboard") left out the tiles its data could not give, and
said nothing: the "missing" it worked out ("operating costs") stopped at the recipe. A tile whose period had no
rows ("refunds over the last 12 months", none since the spring) was saved as it was, and drawn empty on the
dashboard for good. Now:

* what the recipe wanted and the data has not is said on the "dashboard created" card ("Not in this data, so not
  on it: operating costs");
* a figure a recipe measures against a target the data has not is measured against the year before in its place
  (the stores furthest behind target become the biggest fallers), unless that is on the dashboard already;
* a tile that finds nothing in its period is drawn over all the data instead, and the card names it;
* a tile that finds nothing in any period is left out, and named with why, as a tile that could not be built is.

Invented data only.
"""

from __future__ import annotations

import datetime as dt
import json
import re

import pytest

from core import i18n
from core2 import dashboards, recipe_book
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")


@pytest.fixture(scope="module")
def networking():
    return learn(domains.build("networking"), "descriptive")


def _no_ai(stable: str, tail: str) -> str:
    raise AssertionError("the AI was asked")


def _services(retail, ai=_no_ai) -> Services:
    built, model = retail
    return Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=ai, index=MemberIndex(), today=TODAY)


# ── a tile, as answered ─────────────────────────────────────────────────────

PLAN = {"kind": "query", "intent": "value", "measures": ["net_amount"],
        "time": {"window": {"kind": "last", "unit": "month", "n": 12}, "compare": {"kind": "previous_period"}}}


def _payload(rows: list[dict], plan: dict | None = None) -> dict:
    return {"data": {"rows": rows}, "plan": plan or PLAN, "answer": {"headline": "x"}}


class Answers:
    """The service, scripted: each call's plan kept, its answer the next of ``replies``."""

    def __init__(self, *replies):
        self.replies, self.plans = list(replies), []

    def __call__(self, question: str, plan: dict | None = None) -> dict:
        self.plans.append(plan)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_a_tile_with_nothing_in_its_period_is_drawn_over_all_the_data():
    answers = Answers(_payload([{"net_amount": None}]),
                      _payload([{"net_amount": 1250.0}], {**PLAN, "time": {"window": {"kind": "all"}}}))
    tile = dashboards.answered(dashboards.Tile("Net amount", "number", "Net amount over the last 12 months",
                                               planned=PLAN), answers)
    assert not tile.error and tile.widened and tile.payload["data"]["rows"] == [{"net_amount": 1250.0}]
    # Asked again over all the data, the period before it was compared with let go: all the data has none.
    assert answers.plans[1]["time"] == {"window": {"kind": "all"}}
    assert tile.plan["time"]["window"] == {"kind": "all"}, "saved as drawn: the dashboard draws it so again"
    draft = dashboards.Draft("Sales", "a sales dashboard", [tile])
    assert dashboards.widened(draft) == ["Net amount"] and dashboards.not_built(draft) == []


def test_a_tile_with_nothing_in_any_period_is_left_out_and_said():
    answers = Answers(_payload([]), _payload([]))
    tile = dashboards.answered(dashboards.Tile("Refunds", "number", "Refunds this year", planned=PLAN), answers,
                               lang="fr")
    assert tile.error == dashboards.NOTHING["fr"] and tile.payload is None and not tile.widened
    draft = dashboards.Draft("Ventes", "un tableau de bord des ventes", [tile])
    assert dashboards.not_built(draft) == [{"title": "Refunds", "question": "Refunds this year",
                                            "error": dashboards.NOTHING["fr"], "definable": False,
                                            "left_out": False}]
    assert dashboards.widened(draft) == []


def test_a_tile_over_all_the_data_already_is_not_asked_twice():
    plain = {**PLAN, "time": {"window": {"kind": "all"}}}
    answers = Answers(_payload([], plain))
    tile = dashboards.answered(dashboards.Tile("Refunds", "number", "Refunds", planned=plain), answers)
    assert tile.error == dashboards.NOTHING["en"] and len(answers.plans) == 1


def test_a_count_of_nothing_is_a_value_and_kept_as_it_is():
    """0 cancelled orders is an answer: only a tile with no value at all is empty."""
    answers = Answers(_payload([{"cancelled_orders": 0}]))
    tile = dashboards.answered(dashboards.Tile("Cancelled", "number", "Cancelled orders", planned=PLAN), answers)
    assert not tile.error and not tile.widened and len(answers.plans) == 1


def test_a_tile_that_fails_over_all_the_data_is_left_out_not_raised():
    answers = Answers(_payload([{"net_amount": None}]), RuntimeError("the warehouse went away"))
    tile = dashboards.answered(dashboards.Tile("Net amount", "number", "Net amount", planned=PLAN), answers)
    assert tile.error == dashboards.NOTHING["en"]


# ── through the chat ────────────────────────────────────────────────────────

def test_a_recipe_given_in_part_says_what_the_data_has_not(retail):
    payload = answer_question("Create a profit and loss dashboard", _services(retail), Session())
    assert payload["type"] == "dashboard_built" and payload["dashboard_tiles"]
    assert payload["missing"] == ["operating costs"] and payload["widened"] == []


def test_a_recipe_the_data_gives_in_full_says_nothing_is_missing(retail):
    payload = answer_question("Create a dashboard for the sales team", _services(retail), Session())
    assert payload["type"] == "dashboard_built" and payload["missing"] == []


def test_a_designed_tile_with_nothing_in_its_period_is_shown_over_all_the_data(retail):
    design = {"name": "Old sales", "tiles": [
        {"title": "Net amount", "kind": "number", "question": "Net amount this year"},
        {"title": "Net amount in 2001", "kind": "number", "question": "Net amount in 2001"},
        {"title": "Net amount by month", "kind": "trend", "question": "Net amount by month this year"}]}
    plans = {"Net amount this year": {"measures": ["net_amount"], "time": {"window": {"kind": "this", "unit": "year"}}},
             "Net amount in 2001": {"measures": ["net_amount"], "time": {"window": {
                 "kind": "between", "start": "2001-01-01", "end": "2001-12-31"}}},
             "Net amount by month this year": {"intent": "trend", "measures": ["net_amount"], "time": {
                 "grain": "month", "window": {"kind": "this", "unit": "year"}}}}

    def ai(stable: str, tail: str) -> str:
        if tail.startswith("REQUEST:"):
            return json.dumps(design)
        question = re.search(r"^QUESTION: (.*)$", tail, re.MULTILINE).group(1).strip()
        return json.dumps({"kind": "query", **plans[question]})

    payload = answer_question("Build me a dashboard of old sales", _services(retail, ai), Session())
    assert payload["type"] == "dashboard_built" and payload["widened"] == ["Net amount in 2001"]
    kept = {t["title"]: t for t in payload["dashboard_tiles"]}
    # Saved as drawn, over all the data: the plan's default window, so written as no window at all.
    window = (kept["Net amount in 2001"]["plan"].get("time") or {}).get("window") or {"kind": "all"}
    assert window == {"kind": "all"} and kept["Net amount"]["plan"]["time"]["window"]["kind"] == "this"
    assert payload["not_built"] == []


# ── a target the data has not ───────────────────────────────────────────────

def test_a_target_the_data_has_not_is_measured_against_the_year_before(networking):
    _, model = networking
    review = recipe_book.build(model, "monthly_review", today=TODAY)
    assert "targets" in review.missing and review.instead == ["Biggest fallers: sites"]
    fallers = next(t for t in review.tiles if t.title == "Biggest fallers: sites")
    assert fallers.plan["time"]["compare"] == {"kind": "previous_period"} and fallers.plan["limit"] == 5


def test_a_stand_in_already_on_the_dashboard_is_not_drawn_twice(networking):
    """Sales performance measures sales against target by month: the data has no target, and "sales by month
    against the year before" is on it already, so nothing is drawn in its place (no plain "Sales by month")."""
    _, model = networking
    performance = recipe_book.build(model, "sales_performance", today=TODAY)
    titles = [t.title for t in performance.tiles]
    assert "targets" in performance.missing and performance.instead == []
    assert "Sales by month" not in titles and len(titles) == len(set(titles))


def test_the_chat_says_what_is_drawn_in_the_place_of_a_target(networking):
    payload = answer_question("Create a monthly review dashboard", _services(networking), Session())
    assert payload["type"] == "dashboard_built" and set(payload["missing"]) >= {"targets"}
    assert payload["instead"] == ["Biggest fallers: sites"]
    assert "Biggest fallers: sites" in {t["title"] for t in payload["dashboard_tiles"]}


# ── said on the card ────────────────────────────────────────────────────────

def test_the_card_says_what_is_missing_and_what_is_shown_over_all_the_data():
    pytest.importorskip("dukpy")
    from tests.chat_js import run

    msg = {"type": "dashboard_created", "tiles": 5, "dashboard": {"id": 3, "name": "Profit and loss",
                                                                  "url": "/portal/dashboard?dashboard_id=3"},
           "tabs": [], "not_built": [], "missing": ["operating costs", "targets"],
           "instead": ["Biggest fallers: stores"], "widened": ["Refunds"]}
    card = dict(functions=["function dashboardMadeHtml(msg)", "function escHtml(str)"],
                preamble="function qbIcon(n){return '<svg></svg>';}")
    markup = run(f"JSON.stringify(dashboardMadeHtml({json.dumps(msg)}));", lang="en", **card)
    assert "Not in this data, so not on it: operating costs, targets. In their place: “Biggest fallers: stores”." \
        in markup
    assert "Nothing in their period, so shown over all the data: “Refunds”." in markup
    plain = run(f"JSON.stringify(dashboardMadeHtml({json.dumps({**msg, 'missing': [], 'instead': [], 'widened': []})}));",
                lang="en", **card)
    assert "Not in this data" not in plain and "over all the data" not in plain
    french = run(f"JSON.stringify(dashboardMadeHtml({json.dumps(msg)}));", lang="fr", **card)
    assert "Absent de ces données, donc pas affiché : operating costs, targets." in french


def test_the_relay_saves_the_dashboard_and_hands_both_lists_to_the_page(retail):
    """gateway/core2_bridge.py _created: what the page is sent is what the card reads."""
    import uuid

    import store
    from gateway.core2_bridge import _created

    store.init_db()
    account = f"gaps-{uuid.uuid4().hex[:10]}"
    store.upsert_client(account, "portal")
    owner = store.create_user(account, "Ada Owner", f"{uuid.uuid4().hex[:8]}@test.com",
                              password="a-password-they-chose")[0]
    try:
        payload = answer_question("Create a profit and loss dashboard", _services(retail), Session())
        sent = _created(account, {"id": owner, "account_id": account}, {**payload, "widened": ["Refunds"]}, "q-1")
        assert sent["type"] == "dashboard_created" and sent["missing"] == ["operating costs"]
        assert sent["widened"] == ["Refunds"] and sent["instead"] == []
        assert sent["tiles"] == len(payload["dashboard_tiles"])
    finally:
        with store.get_db() as conn:
            for table in ("pinned_chart", "dashboard_artifact_version", "dashboard_artifact", "portal_user", "client"):
                conn.execute(f"DELETE FROM {table} WHERE account_id=?", (account,))


def test_the_card_words_are_in_both_languages():
    for key in ("ui.chat.dashmade.missing", "ui.chat.dashmade.instead", "ui.chat.dashmade.widened"):
        en, fr = i18n.t(key, lang="en"), i18n.t(key, lang="fr")
        assert en != key and fr != key and en != fr
