"""A new-core answer can be added to a dashboard, and its tile keeps the answer's shape.

The new core's answers had no pin token, so the answer card offered no "Add to
dashboard" at all. Each answer that ran a query now gets one, and the token
carries the answer's plan. The tile is drawn by running that plan again
(core2.service.portal_replay) on today's data, as its viewer is allowed to see
it: a comparison stays a dumbbell, a total stays a number, a list a table --
where re-running the stored SQL through today's chart guesser drew whatever
that guesser made of the rows.

The write site and the read site meet here with nothing handed between them by
the test: the token is created by the bridge, read by the pin API into a
pinned chart, and the chart's tile is drawn from what the store kept.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
from unittest.mock import MagicMock, patch

import pytest

from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
APRIL = {"kind": "between", "start": "2026-04-01", "end": "2026-04-30"}
COMPARE = {"intent": "compare", "measures": ["net_amount"], "group_by": ["store"],
           "time": {"window": APRIL, "compare": {"kind": "previous_period"}}}
TOTAL = {"intent": "value", "measures": ["net_amount"], "time": {"window": APRIL}}
LIST = {"intent": "list", "group_by": ["store"]}


@pytest.fixture(scope="module")
def retail():
    built, model = learn(domains.build("retail"), "descriptive")
    return built.con, model


def _services(retail, answers: list[str]) -> Services:
    con, model = retail
    return Services(model=model, warehouse=DuckDBWarehouse(con), complete=lambda s, t: answers.pop(0),
                    index=MemberIndex(), today=TODAY)


def _answer(retail, plan: dict) -> dict:
    return answer_question("q", _services(retail, [json.dumps({"kind": "query", **plan})]), Session())


def _replay(retail):
    """portal_replay on the invented warehouse: the stored plan, run again, with no AI."""
    from core2 import service

    def replay(account_id, plan_data, portal_user, *, question=""):
        services = _services(retail, [])
        ctx = service.Context(today=services.today, allowed_tables=None, max_rows=service.MAX_ROWS)
        return service._compute(question, service.parse_plan(plan_data), services, ctx, question_id="",
                                started=0.0)
    return replay


@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    from portal import routes

    routes.store.init_db()
    return routes.store


def _reader(store) -> dict:
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    db_id = store.save_db_config("azure_sql", "invented", {"server": "invented", "user": "u", "password": "p"})
    store.update_client_meta(account_id, db_config_id=db_id, chat_ui_enabled=1)
    user_id, _password = store.create_user(account_id, "Reader", f"r{os.urandom(3).hex()}@example.com",
                                           role="analyst", password="reader-pass-123")
    return {"id": user_id, "account_id": account_id, "role": "analyst"}


def _pin_and_draw(fresh_store, retail, plan: dict, *, chosen_type: str = "") -> dict:
    """The bridge's token, the pin API, the stored chart, its tile: end to end."""
    from gateway import core2_bridge
    from portal import routes

    user = _reader(fresh_store)
    payload = _answer(retail, plan)
    token = core2_bridge._pin(user["account_id"], user, "net amount question", payload)
    assert token, "an answer that ran a query is given a pin token"

    request = MagicMock()
    request.json = MagicMock(return_value=asyncio.sleep(0, result={
        "token": token, "title": "Pinned", "new_dashboard_name": "Mine", "chart_type": chosen_type}))
    with patch.object(routes, "_get_portal_user", return_value=user):
        response = asyncio.run(routes.pin_chart_api(request))
    assert json.loads(response.body)["ok"] is True

    chart = next(c for c in fresh_store.list_pinned_charts(user["id"]) if c["title"] == "Pinned")
    with patch("core2.service.portal_replay", _replay(retail)):
        return routes._refresh_chart(chart, {"db_type": "duckdb"}, user)


def test_a_comparison_is_pinned_as_the_dumbbell_it_was(fresh_store, retail):
    tile = _pin_and_draw(fresh_store, retail, COMPARE)
    drawn = json.loads(tile["chart_json"])
    assert drawn["chart_type"] == "dumbbell"
    assert drawn["compare"]["prior_label"] == "March 2026" and drawn["compare"]["current_label"] == "April 2026"
    assert tile["error"] is None and tile["row_count"] > 0
    assert tile["subtitle"] == "April 2026 vs March 2026", "the tile names its periods, as the answer did"


def test_the_type_the_reader_chose_is_kept_when_the_shape_offers_it(fresh_store, retail):
    tile = _pin_and_draw(fresh_store, retail, COMPARE, chosen_type="bar")
    assert json.loads(tile["chart_json"])["chart_type"] == "bar"


def test_a_type_the_shape_does_not_offer_is_not_forced_on_it(fresh_store, retail):
    tile = _pin_and_draw(fresh_store, retail, COMPARE, chosen_type="line")
    assert json.loads(tile["chart_json"])["chart_type"] == "dumbbell", "a category is never a line"


def test_a_total_is_pinned_as_a_number(fresh_store, retail):
    tile = _pin_and_draw(fresh_store, retail, TOTAL)
    assert tile["kpi"]["label"] == "Net amount"
    assert tile["kpi_display"].startswith("$")
    assert tile["chart_json"] is None


def test_a_list_is_pinned_as_a_table_with_its_names(fresh_store, retail):
    tile = _pin_and_draw(fresh_store, retail, LIST)
    assert tile["table_column_labels"] == {"store_name": "Store name"}
    assert tile["table_shown"] == 12 and tile["table_rows"][0]["store_name"]["d"]


def test_a_plan_that_no_longer_runs_says_why_on_its_tile(fresh_store, retail):
    from portal import routes

    refused = {"answer": {"headline": "I cannot answer that from this data: the column is gone."}, "data": None}
    with patch("core2.service.portal_replay", return_value=refused):
        tile = routes._refresh_core2_tile({"id": 1, "question": "q", "chart_type": "bar"}, {"error": None},
                                          {"account_id": "a", "id": 1}, TOTAL)
    assert tile["error"] == "I cannot answer that from this data: the column is gone."


def test_an_answer_that_ran_no_query_gets_no_token(retail):
    from gateway import core2_bridge

    question_back = {"answer": {"headline": "Which date?"}, "data": None, "trust": {}, "plan": None}
    assert core2_bridge._pin("acct", {"id": 1}, "q", question_back) == ""


def test_an_unsigned_viewer_gets_no_token(retail):
    from gateway import core2_bridge

    assert core2_bridge._pin("acct", None, "q", _answer(retail, TOTAL)) == ""
