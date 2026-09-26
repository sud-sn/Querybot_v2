# -*- coding: utf-8 -*-
"""A date named in a question is read on the measure's own table.

"Net sales by shipment date for 2025" named a date that only the shipments
fact carries. The resolver looked for it on the facts in scope, found none
there, and fell back to every table: the shipments fact's Shipment Date became
the date of Net Sales. The graph step pulls in the fact a named date lives on,
so the facts "in scope" vouched for it anyway, and "net sales by order date and
shipment date" bound both -- while the planner's own match on the words told
the model, under "Required join path", to join the shipments fact. With nothing
settled, the card for "net sales for 2025" offered the shipments date as one of
the sales measure's dates.

Now a named date binds only on the measure's own table. When it is not there,
the reader is offered the measure's own dates and told why the one they named
is not among them; when the measure has no dates at all, they are told that.
Once the dates are resolved, the planner's other matches leave the plan. A
named date on the measure's table still binds, and with no measure the named
date still says which fact the question is about.

The real resolver on a model built by the real writer, and the real chat
socket with the model and the warehouse stubbed. Synthetic tables; no
customer data.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from unittest.mock import patch

import pytest

# The store is conftest's scratch database for the run (tests/conftest.py).
import store  # noqa: E402
from core.contextual_dates import resolve_contextual_date_binding
from core.query_pipeline import date_clarification_question
from core.semantic_model import load_semantic_model, patch_date_role, write_semantic_model

SALES, SHIPMENTS, BUDGET = "MART.SLS_FCT", "MART.SHP_FCT", "MART.BGT_FCT"


def _table(*columns):
    return {"columns": [{"name": n, "type": t, "nullable": False, "comment": ""} for n, t in columns],
            "pk_columns": [columns[0][0]], "row_count": 1000, "comment": "", "schema": "MART", "database": "WH"}


SCHEMA = {
    "WH.MART.SLS_FCT": _table(("SLS_FCT_KEY", "bigint"), ("ORD_DT_DMS_KEY", "int"), ("IVC_DT_DMS_KEY", "int"),
                              ("NET_AMT", "decimal(18,2)")),
    "WH.MART.SHP_FCT": _table(("SHP_FCT_KEY", "bigint"), ("SHP_DT_DMS_KEY", "int"), ("SHP_QTY", "decimal(18,2)")),
    "WH.MART.BGT_FCT": _table(("BGT_FCT_KEY", "bigint"), ("BGT_AMT", "decimal(18,2)")),
    "WH.MART.DT_DMS": _table(("DT_DMS_KEY", "int"), ("CAL_DT", "date"), ("YR", "int"), ("MTH", "int")),
}
NET_SALES = {"name": "Net sales", "formula_type": "expression", "sql_template": "SUM(NET_AMT)",
             "base_table": SALES, "synonyms": "sales", "description": "Invoiced sales"}
BUDGET_METRIC = {"name": "Budget", "formula_type": "expression", "sql_template": "SUM(BGT_AMT)",
                 "base_table": BUDGET, "synonyms": "", "description": "Planned amount"}


def _approved_model(kb):
    with open(os.path.join(kb, "_schema.json"), "w", encoding="utf-8") as handle:
        json.dump(SCHEMA, handle)
    write_semantic_model(schema_dir=kb, kb_dir=kb)
    for role in load_semantic_model(kb)["date_roles"]:
        patch_date_role(kb_dir=kb, fact_table=role["fact_table"], fact_column=role["fact_column"],
                        status="approved")
    return load_semantic_model(kb)["date_roles"]


@pytest.fixture(scope="module")
def roles():
    return _approved_model(tempfile.mkdtemp())


def _resolve(roles, question, metrics=(NET_SALES,), facts=(SALES, SHIPMENTS)):
    return resolve_contextual_date_binding(question, matched_metrics=[dict(m) for m in metrics], bindings=[],
                                           date_roles=roles, required_fact_tables=set(facts))


def _columns(resolution):
    items = resolution.get("options") or resolution.get("bindings") or [resolution.get("binding") or {}]
    return [(item.get("fact_table"), item.get("fact_column")) for item in items]


class TestTheResolver:

    @pytest.mark.parametrize("facts", [(SALES, SHIPMENTS), (SALES,)])
    def test_a_date_of_another_table_is_not_the_measures_date(self, roles, facts):
        resolution = _resolve(roles, "net sales by shipment date for 2025", facts=facts)
        assert resolution["status"] == "ambiguous"
        assert _columns(resolution) == [(SALES, "ORD_DT_DMS_KEY"), (SALES, "IVC_DT_DMS_KEY")]
        assert [item["fact_table"] for item in resolution["named_elsewhere"]] == [SHIPMENTS]

    def test_of_two_named_dates_only_the_measures_own_binds(self, roles):
        resolution = _resolve(roles, "net sales by order date and shipment date for 2025")
        assert resolution["status"] == "selected"
        assert _columns(resolution) == [(SALES, "ORD_DT_DMS_KEY")]

    def test_the_measures_own_named_date_binds(self, roles):
        resolution = _resolve(roles, "net sales by invoice date for 2025")
        assert _columns(resolution) == [(SALES, "IVC_DT_DMS_KEY")]

    def test_a_measure_with_no_dates_says_so(self, roles):
        resolution = _resolve(roles, "budget by shipment date for 2025", metrics=(BUDGET_METRIC,),
                              facts=(BUDGET, SHIPMENTS))
        assert resolution["status"] == "named_date_not_on_measure"
        assert resolution["options"] == []

    def test_with_no_measure_the_named_date_says_which_fact(self, roles):
        resolution = _resolve(roles, "quantity shipped by shipment date for 2025", metrics=(), facts=(SHIPMENTS,))
        assert _columns(resolution) == [(SHIPMENTS, "SHP_DT_DMS_KEY")]

    def test_the_card_offers_only_the_measures_own_dates(self, roles):
        resolution = _resolve(roles, "net sales for 2025")
        assert resolution["status"] == "ambiguous"
        assert {table for table, _column in _columns(resolution)} == {SALES}


class TestTheQuestionOnTheCard:

    def test_it_names_the_date_and_the_measure(self):
        assert date_clarification_question(ambiguous=False, allow_free_text=False, lang="en",
                                           named_date="Shipment Date", measures="Net sales") == (
            "**Shipment Date** is not a date of **Net sales**. Which of its dates should I use?")

    def test_in_french(self):
        assert date_clarification_question(ambiguous=False, allow_free_text=False, lang="fr",
                                           named_date="Date d’expédition", measures="Ventes nettes") == (
            "**Date d’expédition** n’est pas une date de **Ventes nettes**. "
            "Laquelle de ses dates dois-je utiliser ?")


# ── Over the chat socket ─────────────────────────────────────────────────────

def _ask(question: str) -> tuple[list[dict], list[str]]:
    """Ask over the real chat socket; the model and the warehouse are stubbed.
    Returns the frames and the prompts the model was sent."""
    import anyio
    import core.llm as llm
    import core.query_pipeline as qp
    import gateway.webhooks as wh
    import portal.routes as pr
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from core.graph_autopopulate import auto_populate_from_schema
    from core.semantic_contract import write_contract

    store.init_db()
    account = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account, "Test Ltd")
    store.update_client_meta(account, chat_ui_enabled=1)
    user_id, _ = store.create_user(account, "Ada", f"{os.urandom(4).hex()}@x.com", password="a-password-they-chose")
    group_id = store.create_group(account, "Analysts")
    store.set_group_tables(group_id, account, [fqn.split(".", 1)[1] for fqn in SCHEMA])
    store.update_user(user_id, group_id=group_id)
    kb = tempfile.mkdtemp()
    _approved_model(kb)
    store.update_client_state(account, "READY", {"schema_dir": kb, "kb_dir": kb})
    auto_populate_from_schema(account, kb)
    store.save_metric(account, dict(NET_SALES))
    store.save_metric(account, dict(BUDGET_METRIC))
    write_contract(account, kb)

    prompts: list[str] = []

    async def model(system, user, *args, **kwargs):
        prompts.append(f"{system}\n{user}")
        return "SELECT SUM(NET_AMT) AS NET_SALES FROM MART.SLS_FCT", 10, 10

    class Result:
        def __init__(self, sql):
            self.sql, self.rows, self.truncated = sql, [{"NET_SALES": 5120}], False
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
    return frames, prompts


def _said(frames):
    return json.dumps(frames, ensure_ascii=False)


def _join_paths(prompts):
    """The "Required join path" sections of the prompts the model was sent."""
    return ["\n".join(section.split("\n\n", 1)[0] for section in prompt.split("Required join path:")[1:])
            for prompt in prompts]


class TestOverTheChatSocket:

    def test_the_reader_is_offered_the_measures_own_dates_and_told_why(self):
        frames, prompts = _ask("net sales by shipment date for 2025")
        said = _said(frames)
        assert "**Shipment Date** is not a date of **Net sales**" in said
        assert "Order Date" in said and "Invoice Date" in said
        assert prompts == []  # asked, not answered

    def test_a_measure_with_no_dates_is_told_so(self):
        frames, _prompts = _ask("budget amount by shipment date for 2025")
        assert "and it has no other date to use" in _said(frames)

    def test_the_other_facts_date_does_not_bring_its_table_into_the_query(self):
        _frames, prompts = _ask("net sales by order date and shipment date for 2025")
        paths = _join_paths(prompts)
        assert paths and all("SLS_FCT.ORD_DT_DMS_KEY" in path for path in paths)
        assert not [path for path in paths if "SHP_FCT" in path]
