"""
core/investigation.py
──────────────────────
The one tool a multi-step investigation calls: an ordinary governed question.

An investigation's value is not a new way to reach the warehouse -- it is
asking several ordinary questions in sequence instead of one, so a request
that needs several facts to answer ("why did scrap rise") gets several
answers to reason from rather than one. Every question this module runs goes
through the exact path an ordinary chat message takes: core.dispatcher.dispatch,
which resolves semantics, validates SQL, enforces ACL and compliance, and
executes through the same governed pipeline. Nothing here writes SQL, joins a
fact to another fact, or reads a row the pipeline itself would have refused.

How a sub-question is run
──────────────────────────
A fresh, disposable WebAdapter is built for each investigation, scoped to its
own result-cache session (account:user:thread:investigation-<run_id>) so its
evidence-gathering queries never touch the reader's own last-result pointer
or their result-chat memory. dispatch() is called exactly as it is for a real
chat message; the adapter's own bookkeeping (result_cache.store, on success)
is read back afterwards rather than duplicated here.

Success is "a NEW result landed for this call" -- read as the result_id
changing, not merely being present, because two calls share one scoped
session and a second call's failure must never be reported as the first
call's leftover success.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("querybot.investigation")


@dataclass(frozen=True)
class ToolResult:
    """What one tool call produced, for the planner and the evidence ledger.

    ``brief`` is a statistical summary -- labels and computed figures, never
    the row set -- built the same way the card's own conversational reply is
    built (core.insight.describe_result_for_prompt). ``error`` is the reader-
    facing text the pipeline actually sent when the call did not produce a
    result: a validator refusal, an execution failure, a clarification ask,
    or a zero-row explanation, each already reworded and checked for this
    reader by core.situation_phraser where that path was reached.
    """
    ok: bool
    kind: str
    question: str
    brief: str = ""
    result_id: str = ""
    row_count: int = 0
    error: str = ""


def result_llm_features_allowed_for_investigation(account_id: str) -> bool:
    """Whether this tenant's results may reach a planning or synthesis model.

    An investigation's whole mechanism is a model reading a brief of real
    figures and deciding what to ask next -- the same act core.result_conversation
    gates, and the same rule: a workspace with no compliance profile is
    regulated by default, so this is fail-closed without a decision needing
    to be made twice.
    """
    from core.compliance.policy_engine import result_llm_features_allowed

    return result_llm_features_allowed(account_id)


async def run_query_tool(
    *,
    account_id: str,
    portal_user: dict,
    run_id: str,
    question: str,
) -> ToolResult:
    """Ask ``question`` through the real governed pipeline; return what it found.

    A scoped WebAdapter stands in for the browser: dispatch() sends everything
    it would normally send to a websocket into that adapter's own AsyncMock,
    harmlessly. What matters is read back from the adapter's real state
    afterwards -- last_result_id (set only by adapter.cache_result, only on a
    genuine new answer) and the plain-text replies the pipeline sent for
    every other outcome (send_message, overridden here to record rather than
    transmit).

    Background tasks the pipeline queues (BackgroundTasks) are deliberately
    NOT run for a tool call -- they carry post-answer enhancements such as
    the analyst's explanation, which an internal evidence-gathering question
    does not need and which would cost an extra model call per step for
    nothing the investigation reads.
    """
    from fastapi import BackgroundTasks
    from unittest.mock import AsyncMock

    from core.agent_runtime import activate_agent_run
    from core.dispatcher import dispatch
    from core.result_cache import result_cache
    from gateway.web_adapter import WebAdapter

    question = str(question or "").strip()
    if not question:
        return ToolResult(ok=False, kind="query", question=question,
                          error="No question was given to ask.")

    user_id = str(portal_user.get("id") or "").strip()
    adapter = WebAdapter(
        AsyncMock(), account_id, user_id or "0",
        thread_id=f"investigation-{run_id}",
        portal_user_id=portal_user.get("id"),
    )
    sent_texts: list[str] = []

    async def _capture_send_message(_event, text):
        sent_texts.append(str(text or ""))

    adapter.send_message = _capture_send_message

    before = result_cache.get_snapshot(adapter.session_id)
    before_id = str(before.get("result_id") or "")

    event = adapter.make_event(question)
    try:
        # A sub-question's OWN answer trace finishes exactly like a real
        # question's does, and every trace finish reports its outcome to
        # whichever agent run is active (core.pipeline_trace._trace_finish ->
        # AgentRunSession.record_trace_outcome) -- which marks that run
        # "completed" or "failed". Left active, the FIRST tool call would
        # close out the whole investigation's run before the loop had asked
        # its second question. None suppresses that for exactly this call;
        # the investigation's own run is restored on exit and stays open
        # for next_step to keep extending.
        with activate_agent_run(None):
            await dispatch(account_id, event, adapter, BackgroundTasks(), portal_user=portal_user)
    except Exception as exc:  # noqa: BLE001 - one failed step must not end the run
        log.warning("Investigation query tool failed for %r: %s", question[:120], exc)
        return ToolResult(ok=False, kind="query", question=question,
                          error="That question could not be run.")

    after = result_cache.get_snapshot(adapter.session_id)
    after_id = str(after.get("result_id") or "")
    if after_id and after_id != before_id:
        rows = list(after.get("rows") or [])
        sql = str(after.get("sql") or "")
        try:
            from core.insight import describe_result_for_prompt

            brief = describe_result_for_prompt(rows, question, sql=sql)
        except Exception as exc:  # noqa: BLE001 - a missing brief is not a failed step
            log.warning("Investigation step brief could not be built: %s", exc)
            brief = ""
        return ToolResult(
            ok=True, kind="query", question=question, brief=brief,
            result_id=after_id, row_count=len(rows),
        )

    reason = sent_texts[-1].strip() if sent_texts else ""
    return ToolResult(
        ok=False, kind="query", question=question,
        error=reason or "That question returned nothing to work with.",
    )
