"""The web portal's door to the new core (docs/core-v2/DESIGN.md §12.2-12.3).

A workspace answers portal questions with one of three engines:

* ``legacy``  - today's pipeline only (the default; nothing here runs);
* ``compare`` - today's pipeline answers, then the new core's answer to the same
  question follows it, badged "New core (preview)", for side-by-side checking;
* ``core2``   - the new core answers; what it cannot express (``unsupported``)
  or fails on goes to today's pipeline, so no question is left unanswered.

Every new-core answer is recorded (store.log_core2_answer) for comparison, with
a question id so the reader's thumbs reach it. In ``core2`` mode an answer is
also kept the way today's answers are: the workspace's monthly limits are
checked first (at a limit, today's pipeline answers and says so), and an answer
writes an answer trace (the thread's history, the full CSV export, the audit
link) and a query-log row (usage and limits). The new core never blocks the
conversation: it runs in its own threads under a time limit, and any failure is
logged and leaves today's answer standing; side by side, it is also said on a
preview card, so a preview that will not come is not mistaken for one still coming.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import logging
import time
import uuid
from typing import Any

log = logging.getLogger("querybot.core2")

TIMEOUT_SECONDS = 120.0
PREVIEW_BADGE = "New core (preview)"

# The new core's own threads. Today's pipeline runs its warehouse queries on the
# default pool; a slow AI or warehouse call here ties up these instead. A question
# that finds them all busy waits within the same time limit, and today's answer stands.
_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="core2")


async def engine(account_id: str) -> str:
    import store

    try:
        return await asyncio.to_thread(store.get_query_engine, account_id)
    except Exception as exc:     # noqa: BLE001 - the switch failing means today's pipeline, never no answer
        log.warning("core2 engine setting unreadable for %s: %s", account_id, exc)
        return "legacy"


def new_question_id() -> str:
    """A new-core answer's id: feedback and traces find it, and the prefix says which engine answered."""
    return f"c2-{uuid.uuid4().hex[:24]}"


def _within_limits(account_id: str) -> bool:
    """The workspace's monthly question and token limits, as today's pipeline checks them."""
    from core.pipeline_context import check_query_limit, check_token_limit

    return check_query_limit(account_id)[0] and check_token_limit(account_id)[0]


async def _answer(account_id: str, question: str, portal_user: dict | None, session_key: str,
                  mode: str, question_id: str) -> tuple[str, dict[str, Any] | None]:
    """How the new core did (answered, replied, unsupported, timeout, failed) and its frame, if any."""
    from core2.service import portal_answer

    start = time.perf_counter()
    status, payload = "failed", None
    try:
        work = functools.partial(portal_answer, account_id, question, portal_user, session_key=session_key,
                                 question_id=question_id)
        payload = await asyncio.wait_for(asyncio.get_running_loop().run_in_executor(_POOL, work), TIMEOUT_SECONDS)
        status = ("unsupported" if payload.get("unsupported") else
                  "answered" if payload.get("data") else "replied")
    except asyncio.TimeoutError:
        status = "timeout"
        log.warning("core2 took over %.0fs on a question for %s", TIMEOUT_SECONDS, account_id)
    except Exception as exc:     # noqa: BLE001 - logged loudly; today's answer stands
        log.warning("core2 failed on a question for %s: %s", account_id, exc, exc_info=True)
    try:
        await asyncio.to_thread(_record, account_id, portal_user, mode, question, status, payload,
                                int((time.perf_counter() - start) * 1000), question_id)
    except Exception as exc:     # noqa: BLE001
        log.warning("core2 answer could not be recorded for %s: %s", account_id, exc)
    return status, payload


def _keep(account_id: str, portal_user: dict | None, session_id: str, question: str, question_id: str,
          payload: dict[str, Any], rows: list[dict] | None, duration_ms: int) -> int:
    """An answer kept as today's are: its trace (history, full export, audit) and its query-log row (usage)."""
    import store
    from core2.service import question_scrubber

    scrub = question_scrubber(account_id)
    text = scrub(question) if scrub else question
    trust = payload.get("trust") or {}
    sql = str(trust.get("sql") or "")
    row_count = int(trust.get("row_count") or 0)
    user_id = int(portal_user["id"]) if portal_user and portal_user.get("id") else None
    trace_id = store.create_answer_trace(account_id=account_id, question_id=question_id, question_text=text,
                                         portal_user_id=user_id, session_id=session_id, request_source="portal",
                                         route="core2")
    store.update_answer_trace(trace_id, generated_sql=sql, db_type=str(trust.get("data_source") or ""),
                              query_row_count=row_count, query_duration_ms=duration_ms, answer_type="core2",
                              final_answer_summary=str((payload.get("answer") or {}).get("headline") or "")[:500],
                              sql_validation_status="governed", status="success")
    if rows:
        store.store_protected_result_rows(account_id, question_id, rows)
    if sql and payload.get("data") is not None:      # a question back or a greeting ran no query: not counted
        store.log_query(account_id, text, sql, row_count=row_count, success=True, duration_ms=duration_ms,
                        portal_user_id=user_id, question_id=question_id, llm_provider="core2")
    return trace_id


def _record(account_id: str, portal_user: dict | None, mode: str, question: str, status: str,
            payload: dict[str, Any] | None, duration_ms: int, question_id: str = "") -> None:
    """One row per new-core answer, for comparison. The question is kept as QueryBot keeps
    questions: personal data scrubbed for a tenant under compliance (unrecorded if it cannot be)."""
    import store
    from core2.service import question_scrubber

    scrub = question_scrubber(account_id)
    answer = (payload or {}).get("answer") or {}
    trust = (payload or {}).get("trust") or {}
    store.log_core2_answer(
        account_id, user_id=str((portal_user or {}).get("id") or ""), mode=mode,
        question=scrub(question) if scrub else question, status=status, headline=str(answer.get("headline") or ""),
        sql=str(trust.get("sql") or ""), row_count=int(trust.get("row_count") or 0), plan=(payload or {}).get("plan"),
        duration_ms=duration_ms, model_version=int(trust.get("model_version") or 0), question_id=question_id)


def _session_key(account_id: str, portal_user: dict | None, thread_id: str) -> str:
    return f"{account_id}:{(portal_user or {}).get('id') or ''}:thread:{thread_id}"


async def _send(adapter: Any, websocket: Any, payload: dict[str, Any]) -> bool:
    """Send the new core's frame; a failure is logged, never raised into the reader's turn."""
    try:
        async with adapter.send_lock:
            await websocket.send_json(payload)
        return True
    except asyncio.CancelledError:
        raise
    except Exception as exc:     # noqa: BLE001 - today's answer, already sent, must not turn into an error
        log.warning("core2 answer could not be sent: %s", exc)
        return False


async def answer_instead(engine: str, adapter: Any, websocket: Any, account_id: str, question: str,
                         portal_user: dict | None) -> bool:
    """``core2`` mode: answer with the new core; False hands the question to today's pipeline.

    Any other engine is False at once: today's pipeline answers, and in ``compare``
    mode :func:`answer_beside` follows it. A workspace at its monthly limit is
    handed over too: today's pipeline answers that with its own message.
    """
    if engine != "core2":
        return False
    try:
        if not await asyncio.to_thread(_within_limits, account_id):
            return False
    except Exception as exc:     # noqa: BLE001 - an unreadable limit is today's pipeline's to report
        log.warning("core2 could not read the limits of %s: %s", account_id, exc)
        return False
    question_id = new_question_id()
    start = time.perf_counter()
    _status, payload = await _answer(account_id, question, portal_user,
                                     _session_key(account_id, portal_user, adapter.thread_id), "core2", question_id)
    if payload is None or payload.get("unsupported"):
        return False
    rows = payload.pop("export_rows", None)
    try:
        payload["trace_id"] = await asyncio.to_thread(
            _keep, account_id, portal_user, str(getattr(adapter, "session_id", "") or ""), question, question_id,
            payload, rows, int((time.perf_counter() - start) * 1000))
    except Exception as exc:     # noqa: BLE001 - the answer stands; its history and usage row are logged as lost
        log.warning("core2 answer could not be kept in history and usage for %s: %s", account_id, exc)
    sent = await _send(adapter, websocket, payload)
    if sent and payload.get("data") is not None:
        _not_todays_result(adapter)
    return sent


def _not_todays_result(adapter: Any) -> None:
    """The answer on screen is now the new core's: today's last result is no longer the current one.

    Today's follow-up routes -- "why is that?", "just the top 3", "sort by
    value" -- act on today's last result. After a new-core answer that result
    is the one before it, so they would explain or re-cut an older answer,
    under the older question's name. Forgotten, those follow-ups come to the
    new core, which has the conversation; today's earlier cards keep their own
    results and their chips still work.
    """
    forget = getattr(adapter, "forget_current_result", None)
    if forget is None:
        return
    try:
        forget()
    except Exception as exc:     # noqa: BLE001 - the answer is sent; a stale pointer is logged, not raised
        log.warning("core2 could not set aside today's last result: %s", exc)


_COULD_NOT = {
    "timeout": "The new core did not answer this within {seconds:.0f} seconds.",
    "failed": "The new core stopped with an error on this question; the service log has the details.",
}


def _could_not(question: str, status: str, question_id: str) -> dict[str, Any]:
    """A preview card saying the new core gave no answer, and why."""
    text = _COULD_NOT.get(status, _COULD_NOT["failed"]).format(seconds=TIMEOUT_SECONDS)
    return {"type": "assistant_response", "engine": "core2", "question": question, "kind": "could_not",
            "answer": {"headline": f"{text} Today's answer above stands.", "short_value": "", "comparison": "",
                       "scope_badge": "", "scope_note": ""},
            "chart": None, "kpi": None, "data": None, "confidence": {}, "insight_summary": "",
            "anomaly_callouts": [], "coverage_caveats": [], "follow_up_suggestions": [],
            "trust": {"engine": "core2", "question_id": question_id, "stopped": text}}


async def answer_beside(adapter: Any, websocket: Any, account_id: str, question: str,
                        portal_user: dict | None) -> None:
    """``compare`` mode: the new core's answer after today's, badged as a preview.

    A preview has its own question id (the reader's thumbs reach it) but is not
    kept in the thread's history and does not count toward the monthly limit:
    today's answer to the same question already did. When the new core times
    out or fails, the preview says so instead of never arriving.
    """
    question_id = new_question_id()
    status, payload = await _answer(account_id, question, portal_user,
                                    _session_key(account_id, portal_user, adapter.thread_id), "compare", question_id)
    if payload is None:
        payload = _could_not(question, status, question_id)
    elif payload.get("kind") == "smalltalk":
        return     # nothing to compare: today's reply to "thanks" needs no second one
    payload.pop("export_rows", None)
    if isinstance(payload.get("answer"), dict):
        payload["answer"]["scope_badge"] = PREVIEW_BADGE
    payload.setdefault("result_scope", {})["badge"] = PREVIEW_BADGE
    await _send(adapter, websocket, payload)
