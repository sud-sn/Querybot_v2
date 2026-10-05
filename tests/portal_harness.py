"""
A conversation asked through the portal, of a harness tenant.

tests/answer_harness.py asks one question of the query pipeline directly. The
portal does more before and after it: core.dispatcher.dispatch decides whether
a message is a question for the data at all -- a model, the conversational
analyst, decides for most messages -- and a follow-up is read against the
result the conversation already has. A harness that starts at the pipeline
never sees either, and the portal offered a Proceed button for questions its
reader had just asked while every harness answered them.

``Conversation`` asks its questions one after another in one portal thread: a
real gateway.web_adapter.WebAdapter over a socket that keeps every frame, the
dispatcher and the background work it queues, the result kept between turns as
the portal keeps it. The boundaries are the harnesses' own: the warehouse is
DuckDB, the SQL writer is answered with a marker query (so an answer that
needed the model to write its SQL shows as such), and the conversational
analyst says what ``analyst`` says -- by default the reply that made the portal
offer to run a question instead of running it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from unittest.mock import patch

from tests import answer_harness as harness

# What the conversational analyst answered in the portal to questions its
# reader had just asked: an offer to run them.
OFFERING_ANALYST = ("QueryBot can analyze this for you using the metrics in this workspace. "
                    "If you'd like to proceed with retrieving this data, let me know!")
ANALYST = "conversational analyst"


class _Socket:
    """The browser end of the portal's websocket: every frame it was sent."""

    def __init__(self):
        self.frames: list[dict] = []

    async def send_json(self, payload):
        self.frames.append(json.loads(json.dumps(payload, default=str)))

    async def send_text(self, text):
        self.frames.append({"type": "text", "content": str(text)})


class Turn(dict):
    """One question's frames, and what they amount to."""

    @property
    def answer(self) -> dict | None:
        """The answer card, if the turn ended in one."""
        return next((f for f in reversed(self["frames"]) if f.get("type") == "assistant_response"), None)

    @property
    def rows(self) -> list[dict]:
        return ((self.answer or {}).get("data") or {}).get("rows") or []

    @property
    def texts(self) -> list[str]:
        """What the reader was told outside an answer card."""
        return [str(f.get("content") or f.get("text") or f.get("question") or "") for f in self["frames"]
                if f.get("type") in ("message", "assistant_error", "clarification_prompt", "text")]

    @property
    def offered(self) -> bool:
        """The reader was offered a button to run what they had just asked."""
        return any(str(option.get("label")) == "Proceed" for f in self["frames"]
                   if f.get("type") == "clarification_prompt" for option in f.get("options") or [])


class Conversation:
    """One portal thread of ``account``'s tenant over ``warehouse``."""

    def __init__(self, warehouse, account: str, connection: dict, *, lang: str = "en",
                 analyst: str = OFFERING_ANALYST):
        import store
        from gateway.web_adapter import WebAdapter

        self.warehouse = warehouse
        self.account = account
        self.connection = connection
        self.analyst = analyst
        email = "portal-reader@harness.example"
        reader = store.get_user_by_email(account, email)
        if reader is None:
            store.create_user(account, "Reader", email, password="a-password-they-chose", role="admin")
            reader = store.get_user_by_email(account, email)
        self.user = {"id": reader["id"], "role": "admin", "email": reader["email"], "name": "Reader",
                     "group_name": None, "lang": lang, "account_id": account, "last_active_at": "2026-01-01"}
        self.socket = _Socket()
        self.adapter = WebAdapter(self.socket, account, str(reader["id"]), f"t{os.urandom(4).hex()}",
                                  portal_user_id=reader["id"])

    def ask(self, question: str) -> Turn:
        import core.dispatcher as dispatcher
        import core.llm as llm
        import core.query_pipeline as qp
        import core.schema as schema_module
        from fastapi import BackgroundTasks

        executed: list[dict] = []
        model_calls: list[str] = []

        def run_azure_sql(cfg, sql, max_rows=200):
            found = self.warehouse.query(sql, max_rows)
            executed.append({"sql": sql, "rows": found})
            return found

        async def model(system, user, *args, **kwargs):
            model_calls.append(str(system))
            if harness.SQL_WRITER in str(system)[:400]:
                return f"SELECT '{harness.MARKER}' AS marker", 1, 1
            if ANALYST in str(system)[:200]:
                return self.analyst, 1, 1
            return "", 1, 1

        async def turn():
            bg = BackgroundTasks()
            event = self.adapter.make_event(question)
            await dispatcher.dispatch(self.account, event, self.adapter, bg, portal_user=self.user)
            for task in bg.tasks:
                await task.func(*task.args, **task.kwargs)

        start = len(self.socket.frames)
        original = llm.llm_complete
        provider = ("azure_openai", "gpt-4o", "k", {})
        with contextlib.ExitStack() as stack:
            for module in list(__import__("sys").modules.values()):
                if module is not None and getattr(module, "llm_complete", None) is original:
                    stack.enter_context(patch.object(module, "llm_complete", model))
            stack.enter_context(patch.object(qp, "resolve_provider", return_value=provider))
            stack.enter_context(patch.object(dispatcher, "resolve_provider", return_value=provider))
            stack.enter_context(patch.object(qp, "load_retriever", return_value=harness._Retriever()))
            stack.enter_context(patch.object(qp, "retrieve_similar_examples", return_value=[]))
            stack.enter_context(patch.object(schema_module, "_run_azure_sql", run_azure_sql))
            stack.enter_context(patch.object(qp, "get_client_db", lambda *args, **kwargs: self.connection))
            asyncio.run(turn())
        answers = [e for e in executed if harness.MARKER not in e["sql"]]
        return Turn(question=question, frames=self.socket.frames[start:], executed=answers,
                    sql=answers[-1]["sql"] if answers else "",
                    model_wrote_sql=any(harness.SQL_WRITER in call[:400] for call in model_calls),
                    asked_the_analyst=any(ANALYST in call[:200] for call in model_calls))
