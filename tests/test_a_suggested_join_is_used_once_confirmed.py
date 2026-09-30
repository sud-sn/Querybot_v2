"""
A suggested join is used in SQL only once it is confirmed -- as the admin
pages now say.

The setup wizard told the admin that "unreviewed suggestions can steer SQL
joins", and the join tester that it might "fall back to unreviewed
suggestions". Neither happens: every question is planned on confirmed joins,
whatever the workspace's suggested-joins setting (on, unless an admin turned
it off), and the tester shows the plan a question gets. The setting shapes the
follow-up questions offered and the default-date fact inference, and the
pages and the code's own notes now say that.

A scratch store; the resolver and the admin route are the product's own.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest
from starlette.requests import Request

ACCOUNT = "acct-suggested-joins"
QUESTION = "prescription orders without fills"


@pytest.fixture
def suggested_graph(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    monkeypatch.setenv("QUERYBOT_KEY_FILE", str(tmp_path / ".key"))
    import store

    store.init_db()
    store.upsert_client(ACCOUNT, "portal")
    for name, table in (("Order", "F_RX_ORDER"), ("Fill", "F_RX_FILL")):
        store.save_entity(ACCOUNT, name, table, schema_name="dbo", entity_type="fact",
                          display_name=f"Prescription {name}s", status="suggested", generated_by="heuristic")
    store.save_relationship(ACCOUNT, "Order", "Fill", "ORDER_ID", "ORDER_ID", relationship_type="one_to_many",
                            status="suggested", join_type="LEFT")
    assert int(store.get_client(ACCOUNT)["graph_use_suggested"]) == 1
    return store


def test_a_question_is_planned_on_confirmed_joins_only(suggested_graph):
    from core.graph_resolver import resolve_for_question

    result = resolve_for_question(QUESTION, ACCOUNT, "azure_sql", intent={"wants_missing_records": True})
    assert result["enabled"] is False and result["graph_scope"] == "review_only"
    assert not result.get("join_skeleton")


def test_once_confirmed_it_is_used(suggested_graph):
    from core.graph_resolver import resolve_for_question

    for name, table in (("Order", "F_RX_ORDER"), ("Fill", "F_RX_FILL")):
        suggested_graph.save_entity(ACCOUNT, name, table, schema_name="dbo", entity_type="fact",
                                    display_name=f"Prescription {name}s", status="confirmed")
    (relationship,) = suggested_graph.get_full_graph(ACCOUNT)["relationships"]
    suggested_graph.save_relationship(ACCOUNT, "Order", "Fill", "ORDER_ID", "ORDER_ID",
                                      relationship_type="one_to_many", status="confirmed", join_type="LEFT",
                                      rel_id=relationship["id"])
    result = resolve_for_question(QUESTION, ACCOUNT, "azure_sql", intent={"wants_missing_records": True})
    assert result["graph_scope"] == "confirmed"
    assert "JOIN [dbo].[F_RX_FILL]" in result["join_skeleton"]


def test_the_join_tester_shows_what_a_question_gets(suggested_graph):
    from admin import routes

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request({
        "type": "http", "method": "GET", "path": f"/admin/clients/{ACCOUNT}/graph/api/resolve", "root_path": "",
        "scheme": "http", "query_string": f"q={QUESTION.replace(' ', '+')}".encode(),
        "server": ("testserver", 80), "client": ("127.0.0.1", 1), "headers": [],
    }, receive)
    with patch.object(routes, "_is_auth", return_value=True):
        shown = json.loads(asyncio.run(routes.graph_resolve_api(request, ACCOUNT)).body)
    assert shown["graph_scope"] != "suggested_fallback" and not shown.get("join_skeleton")
