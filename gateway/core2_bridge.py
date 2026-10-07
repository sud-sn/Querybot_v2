"""The web portal's door to the new core (docs/core-v2/DESIGN.md §12.2-12.3).

A workspace answers portal questions with one of three engines:

* ``legacy``  - today's pipeline only (the default; nothing here runs);
* ``compare`` - today's pipeline answers, then the new core's answer to the same
  question follows it, badged "New core (preview)", for side-by-side checking;
* ``core2``   - the new core answers; what it cannot express (``unsupported``)
  or fails on goes to today's pipeline, so no question is left unanswered.

Every new-core answer is recorded (store.log_core2_answer) for comparison. The
new core never blocks the conversation: it runs in a worker thread under a time
limit, and any failure is logged and leaves today's answer standing.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

log = logging.getLogger("querybot.core2")

TIMEOUT_SECONDS = 120.0
PREVIEW_BADGE = "New core (preview)"


async def engine(account_id: str) -> str:
    import store

    try:
        return await asyncio.to_thread(store.get_query_engine, account_id)
    except Exception as exc:     # noqa: BLE001 - the switch failing means today's pipeline, never no answer
        log.warning("core2 engine setting unreadable for %s: %s", account_id, exc)
        return "legacy"


async def _answer(account_id: str, question: str, portal_user: dict | None, session_key: str,
                  mode: str) -> dict[str, Any] | None:
    import store
    from core2.service import portal_answer

    start = time.perf_counter()
    status, payload = "failed", None
    try:
        payload = await asyncio.wait_for(asyncio.to_thread(
            portal_answer, account_id, question, portal_user, session_key=session_key), TIMEOUT_SECONDS)
        status = ("unsupported" if payload.get("unsupported") else
                  "answered" if payload.get("data") else "replied")
    except asyncio.TimeoutError:
        status = "timeout"
        log.warning("core2 took over %.0fs on a question for %s", TIMEOUT_SECONDS, account_id)
    except Exception as exc:     # noqa: BLE001 - logged loudly; today's answer stands
        log.warning("core2 failed on a question for %s: %s", account_id, exc, exc_info=True)
    try:
        answer = (payload or {}).get("answer") or {}
        trust = (payload or {}).get("trust") or {}
        await asyncio.to_thread(
            store.log_core2_answer, account_id, user_id=str((portal_user or {}).get("id") or ""), mode=mode,
            question=question, status=status, headline=str(answer.get("headline") or ""),
            sql=str(trust.get("sql") or ""), row_count=int(trust.get("row_count") or 0),
            plan=(payload or {}).get("plan"), duration_ms=int((time.perf_counter() - start) * 1000),
            model_version=int(trust.get("model_version") or 0))
    except Exception as exc:     # noqa: BLE001
        log.warning("core2 answer could not be recorded for %s: %s", account_id, exc)
    return payload


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
    mode :func:`answer_beside` follows it.
    """
    if engine != "core2":
        return False
    payload = await _answer(account_id, question, portal_user,
                            _session_key(account_id, portal_user, adapter.thread_id), "core2")
    if payload is None or payload.get("unsupported"):
        return False
    return await _send(adapter, websocket, payload)


async def answer_beside(adapter: Any, websocket: Any, account_id: str, question: str,
                        portal_user: dict | None) -> None:
    """``compare`` mode: the new core's answer after today's, badged as a preview."""
    payload = await _answer(account_id, question, portal_user,
                            _session_key(account_id, portal_user, adapter.thread_id), "compare")
    if payload is None or payload.get("kind") == "smalltalk":
        return     # nothing to compare: today's reply to "thanks" needs no second one
    if isinstance(payload.get("answer"), dict):
        payload["answer"]["scope_badge"] = PREVIEW_BADGE
    payload.setdefault("result_scope", {})["badge"] = PREVIEW_BADGE
    await _send(adapter, websocket, payload)
