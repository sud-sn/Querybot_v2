# -*- coding: utf-8 -*-
"""A workspace the model may not read results for still explains its answers.

Found in a live test: after every answer the reader saw the card and nothing
else -- no summary, and no follow-up questions under it. The workspace was new
and nobody had chosen its compliance profile yet, and a workspace without one
counts as regulated (fail-closed). A regulated workspace keeps result rows away
from the model, which is right. But both features answered that with silence:

  * the summary after each answer returned early, although the product already
    computes one without the model, from the rows on screen -- the one this
    workspace's analysis buttons get;
  * the follow-up questions returned [], although their first tier is
    templates filled from statistical signals, with no model call.

Now both are delivered without the model, and only the model-written parts
are refused (and audited). A workspace whose profile allows the model gets the
model-written summary as before.

Driven through the real chat socket, as the reader's browser drives it; the
model and the warehouse are the boundaries. A synthetic mart; no customer data.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from unittest.mock import patch

import pytest

# The store is conftest's scratch database for the run (tests/conftest.py);
# nothing here repoints or re-imports it, so modules imported before this one
# keep the same store object as every test after it.
import store  # noqa: E402

ROWS = [
    {"RGN_NM": "North", "NET_SALES": 1200.0},
    {"RGN_NM": "South", "NET_SALES": 800.0},
    {"RGN_NM": "West", "NET_SALES": 300.0},
]
SQL = ("SELECT r.RGN_NM, SUM(f.NET_AMT) AS NET_SALES FROM MART.SLS_FCT f "
       "JOIN MART.RGN_DMS r ON f.RGN_DMS_KEY = r.RGN_DMS_KEY GROUP BY r.RGN_NM ORDER BY NET_SALES DESC")


def _table(own_key: str, *columns: tuple[str, str]) -> dict:
    return {
        "columns": [{"name": n, "type": t, "nullable": False, "comment": ""} for n, t in columns],
        "pk_columns": [own_key], "row_count": 1000, "comment": "", "schema": "MART", "database": "WH",
    }


SCHEMA = {
    "WH.MART.SLS_FCT": _table("SLS_FCT_KEY", ("SLS_FCT_KEY", "bigint"), ("RGN_DMS_KEY", "int"),
                              ("NET_AMT", "decimal(18,2)")),
    "WH.MART.RGN_DMS": _table("RGN_DMS_KEY", ("RGN_DMS_KEY", "int"), ("RGN_NM", "varchar(40)")),
}


def _workspace(profile: str | None) -> tuple[str, int, str]:
    """A workspace with one metric, one reader, and the given profile (None: never chosen)."""
    from core.graph_autopopulate import auto_populate_from_schema
    from core.semantic_contract import write_contract
    from core.semantic_model import write_semantic_model

    store.init_db()
    account = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account, "Test Ltd")
    store.update_client_meta(account, chat_ui_enabled=1)
    if profile:
        store.save_compliance_profile(account, mode=profile, industry="standard")
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
    store.save_metric(account, {
        "name": "Net sales", "formula_type": "expression", "sql_template": "SUM(NET_AMT)",
        "base_table": "MART.SLS_FCT", "synonyms": "net sales, sales", "description": "Net sales",
    })
    write_contract(account, kb)
    return account, user_id, kb


def _drain(ws, limit: int = 40, timeout: float = 6.0) -> list[dict]:
    """Every frame until the socket goes quiet (timeout-aware: silence must fail, not hang)."""
    import anyio

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


def _ask(profile: str | None) -> tuple[list[dict], list[str]]:
    """Ask "net sales by region" over the socket; return the frames and the model calls."""
    import core.llm as llm
    import core.query_pipeline as qp
    import gateway.webhooks as wh
    import portal.routes as pr
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    account, user_id, kb = _workspace(profile)
    known = {fqn.split(".", 1)[1] for fqn in SCHEMA}
    columns = {fqn.split(".", 1)[1]: {c["name"]: c["type"] for c in m["columns"]} for fqn, m in SCHEMA.items()}
    calls: list[str] = []

    async def model(system, user, *args, **kwargs):
        if "Return ONLY the raw SQL" in system:
            calls.append("sql")
            return SQL, 10, 10
        calls.append("summary")
        return ("HEADLINE: North leads net sales\nBODY: North holds over half of net sales.\n"
                "NEXT_STEP: Look at West."), 10, 10

    class Result:
        def __init__(self, sql):
            self.sql, self.rows, self.truncated = sql, [dict(r) for r in ROWS], False
            self.row_obligations, self.decision, self.analysis = [], None, None

    class Retriever:
        last_retrieval_weak = False
        last_retrieval_unscored = False

        def retrieve(self, question, n=8, allowed_tables=None):
            return []

        def retrieve_fact_patterns(self, question, n=2, allowed_tables=None):
            return []

        def _is_global(self, doc):
            return False

    provider = ("azure_openai", "gpt-4o", "key", {})
    app = FastAPI()
    app.include_router(wh.router)
    client = TestClient(app)
    client.cookies.set(pr._COOKIE, pr._sign_session_value(user_id))
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
            _drain(ws, limit=5, timeout=2.0)
            ws.send_json({"type": "message", "text": "net sales by region"})
            frames = _drain(ws)
    return frames, calls


def _one(frames: list[dict], kind: str) -> dict:
    found = [frame for frame in frames if frame.get("type") == kind]
    assert len(found) == 1, [frame.get("type") for frame in frames]
    return found[0]


@pytest.mark.parametrize("profile", [None, "regulated"], ids=["no profile yet", "regulated"])
class TestTheModelMayNotReadResults:

    def test_the_answer_is_followed_by_a_summary_computed_without_the_model(self, profile):
        frames, calls = _ask(profile)
        summary = _one(frames, "assistant_analysis")
        assert summary["computed"] is True and summary["rows_sent_to_llm"] == 0
        assert "North" in summary["body"] and "52.2%" in summary["body"]
        assert "summary" not in calls

    def test_the_answer_carries_follow_up_questions_without_the_model(self, profile):
        frames, calls = _ask(profile)
        answer = _one(frames, "assistant_response")
        questions = [chip["question"] for chip in answer.get("follow_up_suggestions") or []]
        assert questions and all(questions)
        assert "summary" not in calls


class TestTheModelMayReadResults:

    def test_the_summary_is_still_model_written(self):
        frames, calls = _ask("standard")
        summary = _one(frames, "assistant_analysis")
        assert calls.count("summary") >= 1
        assert not summary.get("computed")
