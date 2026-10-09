"""A reader makes a metric in the chat; it is theirs until their admin keeps it for everyone.

"Margin after returns = net amount minus refunds, divided by net amount. Show it by month."
The workspace's AI writes the metric from the reader's words (the admin's Add a metric path),
it is read and checked like any metric an admin types, and it answers the rest of the
question, in that chat only: the chat shows how it was counted, and the reader can ask their
admin to save it for everyone, change it or drop it. The admin accepts it under Data ->
Requests, and from then on everyone's answers can use it.

A question that only filters ("region = East") or asks what a metric is defines nothing.
Invented retail data and a recorded AI only.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from contextlib import ExitStack
from unittest.mock import patch

import pytest

from core2.plan.own_metric import OWN_PREFIX, definition_in
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
H1 = {"kind": "between", "start": "2026-01-01", "end": "2026-05-31"}
FORMULA = "(SUM([Order line · Net amount]) - SUM([Return · Refund amount])) / SUM([Order line · Net amount]) * 100"
WRITTEN = json.dumps({"formula": FORMULA, "conditions": [], "name": "Margin after returns", "synonyms": [],
                      "description": "What is kept of net amount after refunds, as a share of it.",
                      "format": "percent", "assumptions": [], "question": ""})
DEFINED = ("Margin after returns = net amount minus refunds, divided by net amount, as a percentage. "
           "Show it by month in the first half of 2026.")


@pytest.fixture(scope="module")
def retail():
    built, model = learn(domains.build("retail"), "descriptive")
    return built.con, model


class AI:
    """The recorded AI: what the metric writer and the planner answer, and what each was asked."""

    def __init__(self, written: str = WRITTEN, plans: list[dict] | None = None):
        self.written, self.plans, self.asked = written, list(plans or []), []

    def writer(self, words: str):
        def complete(stable: str, tail: str) -> str:
            self.asked.append(("writer", tail))
            return self.written
        return complete

    def planner(self, stable: str, tail: str) -> str:
        self.asked.append(("planner", stable))
        return json.dumps({"kind": "query", **self.plans.pop(0)})


def _services(retail, ai: AI, **extra) -> Services:
    con, model = retail
    return Services(**{"model": model, "warehouse": DuckDBWarehouse(con), "complete": ai.planner,
                       "index": MemberIndex(), "today": TODAY, "write_metric": ai.writer, **extra})


def _as_read(retail) -> str:
    """The formula as the editor writes it back once read."""
    from core2.model import authoring

    _, model = retail
    return authoring.to_text(model, authoring.read(model, FORMULA).expr)


def _by_month(slug: str) -> dict:
    return {"intent": "trend", "measures": [slug], "time": {"grain": "month", "window": H1}}


# ── what reads as a definition ───────────────────────────────────────────────

@pytest.mark.parametrize("question,name,how,ask", [
    (DEFINED, "Margin after returns", "net amount minus refunds, divided by net amount, as a percentage",
     "Show Margin after returns by month in the first half of 2026."),
    ("Define average order value as net amount divided by the number of orders", "average order value",
     "net amount divided by the number of orders", "average order value"),
    ("Marge après retours = montant net moins remboursements, divisé par le montant net. Montre-la par mois",
     "Marge après retours", "montant net moins remboursements, divisé par le montant net",
     "Montre Marge après retours par mois"),
])
def test_a_definition_is_read_as_a_name_how_it_is_counted_and_what_is_then_asked(question, name, how, ask):
    found = definition_in(question)
    assert (found.name, found.how, found.ask) == (name, how, ask)


@pytest.mark.parametrize("question", ["region = East", "sales where region = East", "What is gross margin?",
                                      "Net amount by store in 2026", "store = 12 for May", "status = shipped only",
                                      # a name with a filter in it is a statement about a filter, whatever follows
                                      "sales in May = the total of net amount minus refunds"])
def test_a_filter_or_a_question_defines_nothing(question):
    assert definition_in(question) is None


# ── made in the chat, answering the rest of the question ─────────────────────

def test_the_metric_is_written_from_the_readers_words_and_answers_the_rest(retail):
    ai = AI(plans=[_by_month("margin_after_returns")])
    session = Session()
    payload = answer_question(DEFINED, _services(retail, ai), session)
    assert ai.asked[0] == ("writer", "The admin's description: Margin after returns: net amount minus refunds, "
                                     "divided by net amount, as a percentage")
    assert "- margin_after_returns | Margin after returns | percent |" in ai.asked[1][1]   # the planner sees it
    own = payload["own_metric"]
    assert own["name"] == "Margin after returns" and own["formula"] == _as_read(retail)
    assert own["measure"].key == OWN_PREFIX + "margin_after_returns"
    assert payload["question"] == DEFINED
    rows = payload["data"]["rows"]
    assert len(rows) == 5                                   # January to May 2026
    value = next(v for k, v in rows[0].items() if isinstance(v, float))
    assert 0 < value < 100
    assert payload["own_metrics_used"] == ["Margin after returns"]


def test_it_stays_in_that_chat_for_the_next_questions_and_nowhere_else(retail):
    ai = AI(plans=[_by_month("margin_after_returns"), {"intent": "value", "measures": ["margin_after_returns"],
                                                         "time": {"window": H1}}])
    session = Session()
    answer_question(DEFINED, _services(retail, ai), session)
    later = answer_question("And for the half year as a whole?", _services(retail, ai), session)
    assert later.get("data") and later["own_metrics_used"] == ["Margin after returns"]
    assert "own_metric" not in later                        # made once, used again
    # Another chat, or another reader: the model everyone answers with does not have it.
    _, model = retail
    assert not [m for m in model.measures.values() if m.key.startswith(OWN_PREFIX)]
    other = AI(plans=[{"intent": "value", "measures": ["margin_after_returns"], "time": {"window": H1}}] * 2)
    elsewhere = answer_question("Margin after returns in the first half", _services(retail, other), Session())
    assert not elsewhere.get("data")


def test_defined_again_under_the_same_name_it_is_replaced(retail):
    ai = AI(plans=[_by_month("margin_after_returns"), _by_month("margin_after_returns")])
    session = Session()
    answer_question(DEFINED, _services(retail, ai), session)
    ai.written = json.dumps({**json.loads(WRITTEN), "formula": "SUM([Order line · Net amount]) - "
                                                              "SUM([Return · Refund amount])", "format": "currency"})
    answer_question("Margin after returns = net amount minus refunds. Show it by month in the first half of 2026.",
                    _services(retail, ai), session)
    (only,) = session.own.values()
    assert only.format == "currency" and only.key == OWN_PREFIX + "margin_after_returns"   # follow-ups still find it


def test_a_name_another_metric_has_is_refused(retail):
    payload = answer_question("Net amount = gross amount minus discount amount, every order",
                              _services(retail, AI()), Session())
    assert payload["kind"] == "own_metric_refused" and "already a metric called Net amount" in \
        payload["answer"]["headline"]


def test_a_metric_over_data_the_reader_cannot_see_is_refused(retail):
    _, model = retail
    orders_only = {t for t in model.tables if model.tables[t].name != "returns"}
    payload = answer_question(DEFINED, _services(retail, AI(), allowed_tables=orders_only), Session())
    assert payload["kind"] == "own_metric_refused" and "data you cannot ask about" in payload["answer"]["headline"]


def test_what_the_ai_cannot_write_is_said_plainly(retail):
    payload = answer_question(DEFINED, _services(retail, AI(written="I am not sure.")), Session())
    assert payload["kind"] == "own_metric_refused" and "did not write a formula" in payload["answer"]["headline"]


def test_without_a_metric_writer_a_definition_is_an_ordinary_question(retail):
    ai = AI(plans=[{"intent": "value", "measures": ["net_amount"], "time": {"window": H1}}])
    services = _services(retail, ai, write_metric=None)
    payload = answer_question(DEFINED, services, Session())
    assert "own_metric" not in payload and ai.asked[0][0] == "planner"


# ── from the chat to everyone, through the admin ─────────────────────────────

@pytest.fixture
def site(retail, tmp_path, monkeypatch):
    con, learned = retail
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    import sys

    import store
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    import core2.service as service
    from admin import core2_requests, core2_routes, routes as admin_routes
    from portal import core2_requests as portal_requests
    from portal import routes as portal_routes

    store.init_db()
    account = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account, "portal")
    store.save_core2_model(account, None, learned.model_dump_json())
    store.set_query_engine(account, "core2")
    reader = {"id": 7, "account_id": account, "name": "Riley Reader", "email": "riley@example.test", "role": "analyst"}
    ai = AI(plans=[_by_month("margin_after_returns") for _ in range(4)])

    def services(account_id, question, portal_user, question_id=""):
        return _services(retail, ai), None

    app = FastAPI()
    app.include_router(admin_routes.router)
    app.include_router(portal_routes.router)
    with ExitStack() as stack:
        stack.enter_context(patch.object(service, "_portal_services", services))
        stack.enter_context(patch.object(portal_routes, "_get_portal_user", lambda *a, **k: reader))
        for module in (admin_routes, core2_routes, core2_requests):
            stack.enter_context(patch.object(module, "_is_auth", return_value=True))
        for module in {id(m): m for m in (sys.modules["store"], store, portal_routes.store,
                                           portal_requests.store)}.values():
            stack.enter_context(patch.object(module, "get_allowed_tables", lambda user: None))
        yield TestClient(app), account, store, reader


def test_the_reader_asks_their_admin_and_once_accepted_everyone_can_use_it(site):
    from core2.bootstrap.service import load_model
    from core2.service import portal_answer

    client, account, store, reader = site
    payload = portal_answer(account, DEFINED, reader, session_key="k1", question_id="c2-q1")
    own = payload["own_metric"]
    assert "measure" not in own and own["token"]            # the definition stays on the server
    json.dumps(payload)                                     # the frame is sent as it is
    sent = client.post("/portal/api/requests", json={"kind": "new_metric", "token": own["token"]}).json()
    assert sent["ok"] and sent["request"]["status"] == "waiting"
    (request,) = store.list_core2_requests(account, status="waiting")
    assert (request["source"], request["target_name"], request["question_id"]) == ("chat", "Margin after returns",
                                                                                   "c2-q1")
    assert request["definition"]["key"] == OWN_PREFIX + "margin_after_returns"
    page = client.get(f"/admin/clients/{account}/requests").text
    assert "SUM([Return · Refund amount])" in page and "Accept for everyone" in page
    assert "Write it in the editor" not in page
    assert client.post(f"/admin/clients/{account}/requests/api/{request['id']}/accept", json={}).json()["ok"]
    shared = next(m for m in load_model(account, None).measures.values() if m.business_name == "Margin after returns")
    assert shared.key == "admin:margin_after_returns" and shared.provenance == "admin" and not shared.hidden
    # Undone, it is gone again.
    client.post(f"/admin/clients/{account}/requests/api/{request['id']}/undo")
    assert not [m for m in load_model(account, None).measures.values() if m.business_name == "Margin after returns"]


def test_the_admin_can_accept_it_under_another_name(site):
    from core2.bootstrap.service import load_model
    from core2.service import portal_answer

    client, account, store, reader = site
    token = portal_answer(account, DEFINED, reader, session_key="k2")["own_metric"]["token"]
    client.post("/portal/api/requests", json={"kind": "new_metric", "token": token})
    (request,) = store.list_core2_requests(account, status="waiting")
    client.post(f"/admin/clients/{account}/requests/api/{request['id']}/accept",
                json={"edits": {"name": "Net margin after returns"}})
    assert any(m.business_name == "Net margin after returns" for m in load_model(account, None).measures.values())


def test_a_metric_already_kept_under_that_name_is_not_kept_twice(site):
    from core2.service import portal_answer

    client, account, store, reader = site
    for key in ("k5", "k6"):
        token = portal_answer(account, DEFINED, reader, session_key=key)["own_metric"]["token"]
        client.post("/portal/api/requests", json={"kind": "new_metric", "token": token})
    first, second = sorted(store.list_core2_requests(account, status="waiting"), key=lambda r: r["id"])
    assert client.post(f"/admin/clients/{account}/requests/api/{first['id']}/accept", json={}).json()["ok"]
    again = client.post(f"/admin/clients/{account}/requests/api/{second['id']}/accept", json={}).json()
    assert not again["ok"] and "already a metric called Margin after returns" in again["error"]
    assert store.get_core2_request(account, second["id"])["status"] == "waiting"


def test_a_token_is_the_readers_own_and_dont_keep_it_drops_the_metric(site):
    from core2.service import _session, portal_answer

    client, account, store, reader = site
    token = portal_answer(account, DEFINED, reader, session_key="k3")["own_metric"]["token"]
    with patch("portal.routes._get_portal_user", lambda *a, **k: {**reader, "id": 8}):
        assert not client.post("/portal/api/requests", json={"kind": "new_metric", "token": token}).json()["ok"]
        assert not client.post("/portal/api/own-metrics/forget", json={"token": token}).json()["ok"]
    assert _session("k3").own
    assert client.post("/portal/api/own-metrics/forget", json={"token": token}).json()["ok"]
    assert not _session("k3").own
    gone = client.post("/portal/api/requests", json={"kind": "new_metric", "token": token}).json()
    assert not gone["ok"] and "not in your chat any more" in gone["error"]


def test_a_browser_cannot_send_a_definition_of_its_own(site):
    client, account, store, reader = site
    client.post("/portal/api/requests", json={"kind": "new_metric", "name": "Everything",
                                              "description": "A metric with a definition sent by the page itself.",
                                              "definition": {"key": "x", "table": "t", "expr": {"agg": "count"}}})
    (request,) = store.list_core2_requests(account, status="waiting")
    assert request["definition"] is None


def test_an_answer_counted_with_a_readers_own_metric_is_not_pinned(site):
    from gateway.core2_bridge import _pin
    from core2.service import portal_answer

    client, account, store, reader = site
    payload = portal_answer(account, DEFINED, reader, session_key="k4")
    assert payload["own_metrics_used"] and _pin(account, reader, DEFINED, payload) == ""


# ── the chat card ────────────────────────────────────────────────────────────

def test_the_chat_shows_the_metric_with_its_choices():
    from tests.chat_js import run
    from tests.test_the_answer_card import FUNCTIONS, PREAMBLE

    msg = {"own_metric": {"name": "Margin after returns", "formula": FORMULA, "only": ["status is Completed"],
                          "token": "tok"}}
    html = run(f"JSON.stringify(_ownMetricHtml({json.dumps(msg)}))", functions=FUNCTIONS,
               consts=["_NUMERIC_FORMATS", "_QB_CURRENCY_SYMBOL"], preamble=PREAMBLE)
    assert "A new metric, for this chat" in html and "Margin after returns" in html
    assert 'data-own="ask"' in html and 'data-own="forget"' in html and 'data-own="change"' in html
    assert "Only status is Completed" in html and 'data-own-token="tok"' in html
    used = run(f"JSON.stringify(_ownMetricHtml({json.dumps({'own_metrics_used': ['Margin after returns']})}))",
               functions=FUNCTIONS, consts=["_NUMERIC_FORMATS", "_QB_CURRENCY_SYMBOL"],
               preamble=PREAMBLE)
    assert "Counted with your own metric: Margin after returns" in used
