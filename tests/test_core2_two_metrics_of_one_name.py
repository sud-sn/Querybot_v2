"""Two metrics of one name: the question's words pick one, the answer says which, the admin is told.

On a copy of a real inventory warehouse, "Average item cost" was learned on a daily balance and again on a
monthly one. Asked "Which item groups have the highest average item cost?", a real model (gpt-4o) took the
monthly one, with nothing in the question to say so. Now:

* the words that tell the two apart (each one's table, the words its slug adds to the name) pick one when the
  question says them; when it says none, the one holding the plain name is taken;
* the answer says which one it is, and where the other is;
* the AI's catalog marks each of the two;
* the Metrics page flags the pair, and says which one a question that does not say takes.

Two metrics of one name counted the same way (an admin's metric over the column Learn found) are one figure:
nothing is flagged. Synthetic data only: the retail warehouse with its refund amount renamed "Net amount".
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.bootstrap.build import assign_slugs
from core2.model.authoring import listing
from core2.plan.catalog import catalog_text, same_named
from core2.plan.ir import Plan
from core2.plan.planner import named_twin
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)


@pytest.fixture(scope="module")
def twins():
    built, model = learn(domains.build("retail"), "descriptive")
    refund = next(m for m in model.measures.values() if m.slug == "refund_amount")
    refund.business_name = "Net amount"
    assign_slugs(model)
    return built, model


def _rank(slug: str) -> Plan:
    return Plan.model_validate({"kind": "query", "intent": "rank", "measures": [slug], "group_by": ["store.name"],
                                "sort": [{"by": slug, "desc": True}], "limit": 3})


@pytest.mark.parametrize("question, ai, taken", [
    ("Which stores had the highest net amount?", "return_net_amount", "net_amount"),     # says neither: plain name
    ("Which stores had the highest net amount?", "net_amount", "net_amount"),
    ("Which stores had the highest net amount on returns?", "net_amount", "return_net_amount"),
    ("Which stores had the highest net amount on returns?", "return_net_amount", "return_net_amount"),
])
def test_the_questions_words_pick_one_and_the_plain_name_when_they_say_none(twins, question, ai, taken):
    plan = named_twin(_rank(ai), twins[1], question)
    assert plan.measures == [taken] and plan.sort[0].by == taken


def test_a_figure_named_twice_the_same_way_is_one_figure():
    built, model = learn(domains.build("retail"), "descriptive")
    net = next(m for m in model.measures.values() if m.slug == "net_amount")
    model.measures["import.metric_1"] = net.model_copy(update={"key": "import.metric_1", "slug": "net_amount_2"})
    assert same_named(model) == {}
    assert named_twin(_rank("net_amount_2"), model, "Which stores had the highest net amount?").measures == [
        "net_amount_2"]


def test_the_catalog_and_the_metrics_page_mark_the_pair(twins):
    built, model = twins
    lines = [line for line in catalog_text(model).splitlines() if "same name as" in line]
    assert len(lines) == 2 and any(line.startswith("- net_amount |") and "return_net_amount on Return" in line
                                   for line in lines)
    rows = {r["slug"]: r["same_name"] for r in listing(model, {})}
    assert "take this one" in rows["net_amount"] and "Return" in rows["net_amount"]
    assert "take the other one" in rows["return_net_amount"]
    assert not rows["quantity"]


def test_the_answer_says_which_one_it_is(twins):
    built, model = twins
    replies = [_rank("return_net_amount").model_dump(exclude_defaults=True)]
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con),
                        complete=lambda s, t: json.dumps(replies.pop(0)), index=MemberIndex(), today=TODAY)
    payload = answer_question("Which stores had the highest net amount?", services, Session())
    top = built.con.execute("SELECT SUM(net_amount) FROM order_lines GROUP BY store_id ORDER BY 1 DESC LIMIT 1"
                            ).fetchone()[0]
    assert payload["export_rows"][0]["net_amount"] == pytest.approx(float(top))
    notes = " ".join([*(payload.get("trust", {}).get("date_context") or []), *(payload.get("notes") or [])])
    assert "Net amount here is the one on Order line; Return has a figure of the same name." in notes, notes


def test_the_metrics_page_flags_the_pair(twins, tmp_path, monkeypatch):
    import os
    from unittest.mock import patch

    built, model = twins
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    import store
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from admin import core2_metrics, routes

    store.init_db()
    account = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account, "Retail (invented)")
    store.save_core2_model(account, None, model.model_dump_json())
    app = FastAPI()
    app.include_router(routes.router)
    with patch.object(routes, "_is_auth", return_value=True), patch.object(core2_metrics, "_is_auth", return_value=True):
        page = TestClient(app).get(f"/admin/clients/{account}/measures").text
    assert "2 metrics share a name with" in page
    assert page.count('class="ms-twin"') == 2
    assert "Same name as a metric on Return. Questions that do not say which take this one." in page
