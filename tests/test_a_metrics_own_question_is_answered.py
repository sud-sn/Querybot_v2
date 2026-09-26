# -*- coding: utf-8 -*-
"""A registered metric answers the question the product suggests for it.

Every registered metric gets a suggested question: "What is our total
{name}?". Found in a browser replay: for a metric named "Order lines" --
COUNT(*), approved by an admin -- clicking that suggestion was refused with

    I understand that you want to count orders, but the semantic layer does
    not yet identify which business field represents one order.

"total" is a count cue and "order" a business-event noun, so the sentence was
read as a bare count of orders, which is governed by an approved identifier
for one order. But it names a metric, and the metric's approved formula is
the governed answer. The same happens to a metric named for any of the event
nouns: transactions, receipts, deliveries, invoices, payments...

A plain "how many orders" still waits on the identifier. Synthetic workspace;
no customer data.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from unittest.mock import patch

# The store is conftest's scratch database for the run (tests/conftest.py).
import store  # noqa: E402
from core.analytical_intent import plan_analytical_intent

ORDER_LINES = {"name": "Order lines", "formula_type": "expression", "sql_template": "COUNT(*)",
               "base_table": "MART.SLS_FCT", "synonyms": "order line count", "description": "Sales order lines"}
# Filed under "Orders", about something else.
NET_SALES = {"name": "Net sales", "formula_type": "expression", "sql_template": "SUM(NET_AMT)",
             "base_table": "MART.SLS_FCT", "synonyms": "sales", "category": "Orders"}


class TestWhatTheQuestionCounts:

    def test_a_question_naming_a_count_metric_is_that_metric(self):
        for question in ("What is our total Order lines?", "total order line count"):
            assert plan_analytical_intent(question, metrics=[ORDER_LINES]).counted_entity == "", question

    def test_a_plural_metric_name_is_its_event(self):
        deliveries = {"name": "Deliveries", "sql_template": "COUNT(*)", "base_table": "MART.DLV_FCT"}
        assert plan_analytical_intent("What is our total Deliveries?", metrics=[deliveries]).counted_entity == ""

    def test_a_metric_named_beside_the_count_is_not_the_count(self):
        """The question names "Active customers" and counts orders."""
        active = {"name": "Active customers", "sql_template": "COUNT(DISTINCT CUST_KEY)",
                  "base_table": "MART.SLS_FCT"}
        plan = plan_analytical_intent("how many orders did active customers place", metrics=[active])
        assert plan.counted_entity == "order"

    def test_without_the_metric_it_is_still_a_count_of_orders(self):
        assert plan_analytical_intent("What is our total Order lines?", metrics=[]).counted_entity == "order"

    def test_a_plain_count_still_waits_on_the_identifier(self):
        plan = plan_analytical_intent("how many orders did we have last month", metrics=[ORDER_LINES, NET_SALES])
        assert plan.counted_entity == "order"

    def test_a_metric_filed_under_the_event_is_not_named_by_it(self):
        plan = plan_analytical_intent("how many orders", metrics=[NET_SALES])
        assert plan.counted_entity == "order"


SCHEMA = {
    "WH.MART.SLS_FCT": {"columns": [{"name": n, "type": t, "nullable": False, "comment": ""} for n, t in (
        ("SLS_FCT_KEY", "bigint"), ("RGN_DMS_KEY", "int"), ("NET_AMT", "decimal(18,2)"))],
        "pk_columns": ["SLS_FCT_KEY"], "row_count": 1000, "comment": "", "schema": "MART", "database": "WH"},
    "WH.MART.RGN_DMS": {"columns": [{"name": n, "type": t, "nullable": False, "comment": ""} for n, t in (
        ("RGN_DMS_KEY", "int"), ("RGN_NM", "varchar(40)"))],
        "pk_columns": ["RGN_DMS_KEY"], "row_count": 20, "comment": "", "schema": "MART", "database": "WH"},
}


def _ask(question: str) -> tuple[list[dict], list[str]]:
    """Ask over the real chat socket; the model and the warehouse are stubbed."""
    import anyio
    import core.llm as llm
    import core.query_pipeline as qp
    import gateway.webhooks as wh
    import portal.routes as pr
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from core.graph_autopopulate import auto_populate_from_schema
    from core.semantic_contract import write_contract
    from core.semantic_model import write_semantic_model

    store.init_db()
    account = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account, "Test Ltd")
    store.update_client_meta(account, chat_ui_enabled=1)
    user_id, _ = store.create_user(account, "Ada", f"{os.urandom(4).hex()}@x.com", password="a-password-they-chose")
    group_id = store.create_group(account, "Analysts")
    store.set_group_tables(group_id, account, ["MART.SLS_FCT", "MART.RGN_DMS"])
    store.update_user(user_id, group_id=group_id)
    kb = tempfile.mkdtemp()
    with open(os.path.join(kb, "_schema.json"), "w", encoding="utf-8") as handle:
        json.dump(SCHEMA, handle)
    store.update_client_state(account, "READY", {"schema_dir": kb, "kb_dir": kb})
    auto_populate_from_schema(account, kb)
    write_semantic_model(schema_dir=kb, kb_dir=kb, account_id=account)
    store.save_metric(account, dict(ORDER_LINES))
    write_contract(account, kb)

    calls: list[str] = []

    async def model(system, user, *args, **kwargs):
        calls.append("sql" if "Return ONLY the raw SQL" in system else "other")
        return "SELECT COUNT(*) AS ORDER_LINES FROM MART.SLS_FCT", 10, 10

    class Result:
        def __init__(self, sql):
            self.sql, self.rows, self.truncated = sql, [{"ORDER_LINES": 5120}], False
            self.row_obligations, self.decision, self.analysis = [], None, None

    class Retriever:
        last_retrieval_weak = False
        last_retrieval_unscored = False

        def retrieve(self, *args, **kwargs):
            return []

        def retrieve_fact_patterns(self, *args, **kwargs):
            return []

        def _is_global(self, doc):
            return False

    provider = ("azure_openai", "gpt-4o", "key", {})
    known = {fqn.split(".", 1)[1] for fqn in SCHEMA}
    columns = {fqn.split(".", 1)[1]: {c["name"]: c["type"] for c in m["columns"]} for fqn, m in SCHEMA.items()}
    app = FastAPI()
    app.include_router(wh.router)
    client = TestClient(app)
    client.cookies.set(pr._COOKIE, pr._sign_session_value(user_id))

    def drain(ws, limit, timeout):
        frames = []
        for _ in range(limit):
            async def _recv():
                with anyio.fail_after(timeout):
                    return await ws._send_rx.receive()
            try:
                message = ws.portal.call(_recv)
            except Exception:
                break
            if message.get("type") != "websocket.send" or "text" not in message:
                break
            frames.append(json.loads(message["text"]))
        return frames

    with contextlib.ExitStack() as stack:
        for mock in (
            patch.object(qp, "get_state", return_value={"state": "READY", "schema_dir": kb, "kb_dir": kb}),
            patch.object(qp, "get_client_db", return_value={
                "db_type": "azure_sql", "credentials": {}, "name": "db", "id": 1}),
            patch.object(qp, "load_known_tables", return_value=known),
            patch.object(qp, "load_schema_columns", return_value=columns),
            patch.object(qp, "load_retriever", return_value=Retriever()),
            patch.object(qp, "llm_complete", model),
            patch.object(llm, "llm_complete", model),
            patch.object(qp, "resolve_provider", return_value=provider),
            patch.object(llm, "resolve_provider", return_value=provider),
            patch.object(wh, "resolve_provider", return_value=provider),
            patch.object(qp, "retrieve_similar_examples", return_value=[]),
            patch.object(qp, "execute_governed_query", lambda credentials, db_type, sql, *a, **k: Result(sql)),
        ):
            stack.enter_context(mock)
        with client.websocket_connect(f"/ws/chat/{account}?thread_id=t1") as ws:
            drain(ws, 5, 2.0)
            ws.send_json({"type": "message", "text": question})
            frames = drain(ws, 40, 6.0)
    return frames, calls


def test_the_suggested_question_for_a_count_metric_is_answered():
    frames, calls = _ask("What is our total Order lines?")
    answers = [frame for frame in frames if frame.get("type") == "assistant_response"]
    refusals = [frame.get("content") for frame in frames
                if frame.get("type") == "message" and "business identifier" in str(frame.get("content"))]
    assert refusals == [] and len(answers) == 1, [frame.get("type") for frame in frames]
    assert "5,120" in json.dumps(answers[0])
    assert calls.count("sql") == 1
