"""The web portal's door to the new core (docs/core-v2/DESIGN.md §12.2-12.3).

A workspace answers portal questions with one of three engines:

* ``legacy``  - today's pipeline only (the default; nothing here runs);
* ``compare`` - today's pipeline answers, then the new core's answer to the same
  question follows it, badged "New core (preview)", for side-by-side checking;
* ``core2``   - the new core answers, and only the new core: what it cannot
  express gets its own reply saying why, and a failure or a timeout says so.
  Nothing is handed to today's pipeline.

Every new-core answer is recorded (store.log_core2_answer) for comparison, with
a question id so the reader's thumbs reach it. In ``core2`` mode an answer is
also kept the way today's answers are: the workspace's monthly limits are
checked first (at a limit, the reader is told the limit), and an answer
writes an answer trace (the thread's history, the full CSV export, the audit
link) and a query-log row (usage and limits). The new core never blocks the
conversation: it runs in its own threads under a time limit, and any failure is
logged and said; side by side it leaves today's answer standing, and is said on a
preview card, so a preview that will not come is not mistaken for one still coming.
"""

from __future__ import annotations

import asyncio
import collections
import concurrent.futures
import functools
import json
import logging
import os
import time
import uuid
import weakref
from typing import Any

log = logging.getLogger("querybot.core2")

TIMEOUT_SECONDS = 120.0
PREVIEW_BADGE = "New core (preview)"
MAX_KEPT_FRAME = 2_000_000      # characters of a kept answer; a larger one is reopened from its rows


def _threads() -> int:
    """The new core's answering threads in this process: QUERYBOT_CORE2_THREADS, 16 unless set, 4 to 32."""
    try:
        wanted = int(os.environ.get("QUERYBOT_CORE2_THREADS") or 16)
    except ValueError:
        log.warning("QUERYBOT_CORE2_THREADS is not a number; the new core answers with 16 threads")
        wanted = 16
    return min(max(wanted, 4), 32)


THREADS = _threads()
# One workspace's questions take at most a quarter of the threads at once: a workspace asking many questions
# together waits in its own line, and never keeps another workspace's question from starting.
PER_WORKSPACE = max(2, THREADS // 4)
# How long a question waits in line before the reader is told to ask again. Waiting is not answering: the
# answering limit (TIMEOUT_SECONDS) starts when a thread takes the question.
WAIT_LIMIT_SECONDS = 300.0

# The new core's own threads. Today's pipeline runs its warehouse queries on the
# default pool; a slow AI or warehouse call here ties up these instead.
_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=THREADS, thread_name_prefix="core2")


class _Line:
    """The fair line to the new core's threads, in one event loop: a place for each workspace (at most
    PER_WORKSPACE of its questions at once), then one of the THREADS. Both wake their waiters first come,
    first served. A place is given back when the thread finishes, not when the reader stops waiting: a
    question that ran past its time still holds its thread until it returns."""

    def __init__(self) -> None:
        self.threads = asyncio.Semaphore(THREADS)
        self.workspaces: collections.defaultdict[str, asyncio.Semaphore] = collections.defaultdict(
            lambda: asyncio.Semaphore(PER_WORKSPACE))

    def free(self, account_id: str) -> bool:
        return not self.workspaces[account_id].locked() and not self.threads.locked()

    async def enter(self, account_id: str) -> None:
        mine = self.workspaces[account_id]
        await mine.acquire()
        try:
            await self.threads.acquire()
        except BaseException:
            mine.release()
            raise

    def leave(self, account_id: str) -> None:
        self.threads.release()
        self.workspaces[account_id].release()


# A line per event loop: asyncio's semaphores belong to the loop that first waits on them.
_LINES: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _Line] = weakref.WeakKeyDictionary()


def _line() -> _Line:
    loop = asyncio.get_running_loop()
    line = _LINES.get(loop)
    if line is None:
        line = _LINES[loop] = _Line()
    return line


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


def _over_limit(account_id: str, lang: str = "en") -> str:
    """The workspace's monthly question or token limit, as today's pipeline words it, when it is reached ("" if not)."""
    from core.i18n import t
    from core.pipeline_context import check_query_limit, check_token_limit

    within, used, limit = check_query_limit(account_id)
    if not within:
        return t("terminal.query_limit_reached", lang=lang, used=used, limit=limit)
    within, used, limit = check_token_limit(account_id)
    if not within:
        return t("terminal.token_limit_reached", lang=lang, used=used, limit=limit)
    return ""


async def _answer(account_id: str, question: str, portal_user: dict | None, session_key: str,
                  mode: str, question_id: str, *, waiting: Any = None) -> tuple[str, dict[str, Any] | None]:
    """How the new core did (answered, replied, unsupported, timeout, busy, failed) and its frame, if any.

    The question takes its place in the fair line first; ``waiting`` (a coroutine function) is awaited once
    when it has to wait, to tell the reader. The answering limit starts when a thread takes it."""
    from core2.service import portal_answer

    start = time.perf_counter()
    status, payload = "failed", None
    line = _line()
    entered = False
    try:
        if line.free(account_id):
            # Room in both: taken here and now, before any other question looks (no wait, so no task switch).
            await line.enter(account_id)
        else:
            if waiting is not None:
                try:
                    await waiting()
                except Exception as exc:     # noqa: BLE001 - the notice is a courtesy; the question still waits
                    log.warning("core2 could not tell a reader of %s that their question is in line: %s",
                                account_id, exc)
            await asyncio.wait_for(line.enter(account_id), WAIT_LIMIT_SECONDS)
        entered = True
        loop = asyncio.get_running_loop()
        try:
            work = _POOL.submit(functools.partial(portal_answer, account_id, question, portal_user,
                                                  session_key=session_key, question_id=question_id))
        except BaseException:
            line.leave(account_id)       # no thread took it: its place goes back now
            raise

        def done(_: concurrent.futures.Future) -> None:
            try:
                loop.call_soon_threadsafe(line.leave, account_id)
            except RuntimeError:     # the loop has closed: its line went with it
                pass

        work.add_done_callback(done)
        payload = await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(work)), TIMEOUT_SECONDS)
        status = ("unsupported" if payload.get("unsupported") else
                  "answered" if payload.get("data") else "replied")
    except asyncio.TimeoutError:
        if entered:
            status = "timeout"
            log.warning("core2 took over %.0fs on a question for %s", TIMEOUT_SECONDS, account_id)
        else:
            status = "busy"
            log.warning("core2 was too busy to start a question for %s within %.0fs", account_id,
                        WAIT_LIMIT_SECONDS)
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
    kept = _kept_frame(payload, text)
    store.update_answer_trace(trace_id, generated_sql=sql, db_type=str(trust.get("data_source") or ""),
                              query_row_count=row_count, query_duration_ms=duration_ms, answer_type="core2",
                              final_answer_summary=str((payload.get("answer") or {}).get("headline") or "")[:500],
                              sql_validation_status="governed", status="success",
                              **({"answer_frame": kept} if kept is not None else {}))
    if rows:
        store.store_protected_result_rows(account_id, question_id, rows)
    if sql and payload.get("data") is not None:      # a question back or a greeting ran no query: not counted
        store.log_query(account_id, text, sql, row_count=row_count, success=True, duration_ms=duration_ms,
                        portal_user_id=user_id, question_id=question_id, llm_provider="core2")
    return trace_id


def _kept_frame(payload: dict[str, Any], question: str) -> dict[str, Any] | None:
    """The answer as the reader saw it, kept for a reopened thread (None when it is too large to keep).

    The question is the one kept in the trace (personal data scrubbed for a tenant under
    compliance). The tokens behind its buttons end with the chat, so they are not kept: a
    reopened answer gets a new Add to dashboard token, and a metric the reader made shows
    without the buttons that would save or forget it.
    """
    frame = {k: v for k, v in payload.items() if k not in ("pin_token", "trace_id", "export_rows")}
    frame["question"] = question
    if isinstance(frame.get("own_metric"), dict):
        frame["own_metric"] = {k: v for k, v in frame["own_metric"].items() if k != "token"}
    if isinstance(frame.get("chart"), dict):
        frame["chart"] = {k: v for k, v in frame["chart"].items() if k != "pin_token"}
    if len(json.dumps(frame, default=str)) > MAX_KEPT_FRAME:
        log.warning("A new-core answer of over %d characters is kept by its rows only", MAX_KEPT_FRAME)
        return None
    return frame


def reopened(account_id: str, portal_user: dict | None, trace: dict[str, Any],
             rows: list[dict] | None, *, chart: dict[str, Any] | None = None) -> dict[str, Any]:
    """A new-core answer of a reopened thread, drawn as the new core drew it, with Add to dashboard again.

    An answer kept with its frame comes back as it was shown. One given before answers were
    kept comes back as its sentence and its rows, with ``chart`` (the chart drawn from its rows
    that such an answer was reopened with before): its wording is never rebuilt.
    """
    import store

    question = str(trace.get("question_text_sanitized") or "")
    frame: dict[str, Any] | None = None
    raw = str(trace.get("answer_frame") or "")
    if raw:
        try:
            frame = json.loads(raw)
        except ValueError:
            log.warning("The kept answer of trace %s cannot be read; it is shown from its rows", trace.get("id"))
        frame = frame if isinstance(frame, dict) else None
    if frame is None:
        frame = _from_rows(trace, question, rows or [])
        if chart and frame.get("data"):
            frame["chart"] = chart
        plan = store.get_core2_answer_plan(account_id, str(trace.get("question_id") or ""))
        if plan:
            frame["plan"] = plan
    frame["trace_id"] = int(trace.get("id") or 0)
    try:
        token = _pin(account_id, portal_user, question, frame)
    except Exception as exc:     # noqa: BLE001 - the answer shows; it just cannot be pinned
        log.warning("A reopened new-core answer could not be given a pin token for %s: %s", account_id, exc)
        token = ""
    if token:
        frame["pin_token"] = token
        if isinstance(frame.get("chart"), dict):
            frame["chart"]["pin_token"] = token
    return frame


def _from_rows(trace: dict[str, Any], question: str, rows: list[dict]) -> dict[str, Any]:
    from core2.answer.builder import PREVIEW_ROWS
    from core2.service import _frame

    headers = list(rows[0].keys()) if rows else []
    data = {"headers": headers,
            "header_labels": {h: (h.replace("_", " ").strip().capitalize() or h) for h in headers},
            "rows": rows[:PREVIEW_ROWS], "total_rows": len(rows), "truncated": len(rows) > PREVIEW_ROWS,
            "column_formats": {}, "display_formats": {}, "currency_columns": []} if rows else None
    return _frame(question, str(trace.get("final_answer_summary") or ""), data=data,
                  trust={"engine": "core2", "sql": str(trace.get("generated_sql") or ""),
                         "row_count": int(trace.get("query_row_count") or 0),
                         "data_source": str(trace.get("db_type") or ""),
                         "question_id": str(trace.get("question_id") or "")})


def _pin(account_id: str, portal_user: dict | None, question: str, payload: dict[str, Any]) -> str:
    """A pin token for an answer that ran a query, so Add to dashboard can keep it ("" when it cannot).

    The token carries the answer's plan: its tile is drawn by running that plan again
    (core2.service.portal_replay), so it keeps the answer's shape and moves on with "last month".
    """
    import store
    from core.pipeline_trace import _create_pin_token
    from core2.service import question_scrubber

    trust = payload.get("trust") or {}
    sql, plan = str(trust.get("sql") or ""), payload.get("plan")
    if not (sql and plan and payload.get("data") is not None and portal_user and portal_user.get("id")):
        return ""
    if payload.get("own_metrics_used"):
        return ""           # a tile is drawn again outside the chat, where the reader's own metric is not
    db_config_id = int((store.get_client(account_id) or {}).get("db_config_id") or 0)
    chart = payload.get("chart") or {}
    chart_type = str(chart.get("chart_type") or ("kpi" if payload.get("kpi") else "table"))
    scrub = question_scrubber(account_id)
    return _create_pin_token(int(portal_user["id"]), account_id, scrub(question) if scrub else question, sql,
                             chart_type, db_config_id,
                             display_config={"core2_plan": plan,
                                             "column_formats": chart.get("column_formats") or {},
                                             # The tile's name until the reader gives it another, never the question.
                                             "title": str(chart.get("title")
                                                          or (payload.get("kpi") or {}).get("title")
                                                          or payload.get("title") or ""),
                                             # Several numbers: a tile each, under these names.
                                             "titles": [str(g.get("title") or g.get("label") or "")
                                                        for g in (payload.get("kpi") or {}).get("group") or []]})


async def _with_pin(account_id: str, portal_user: dict | None, question: str, payload: dict[str, Any]) -> None:
    """Give the answer its pin token (Add to dashboard), when it ran a query; a failure leaves it unpinnable."""
    try:
        token = await asyncio.to_thread(_pin, account_id, portal_user, question, payload)
    except Exception as exc:     # noqa: BLE001 - the answer stands; it just cannot be pinned
        log.warning("core2 answer could not be given a pin token for %s: %s", account_id, exc)
        token = ""
    if token:
        payload["pin_token"] = token
        if payload.get("chart"):
            payload["chart"]["pin_token"] = token


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
    """``core2`` mode: the new core answers, and only the new core; False only for another engine.

    Any other engine is False at once: today's pipeline answers, and in ``compare``
    mode :func:`answer_beside` follows it. In ``core2`` mode nothing is handed to
    today's pipeline: a question the new core cannot answer gets its own reason (what
    it considered, under "How it was counted"), a failure or a timeout says so, and a
    workspace at its monthly limit is told the limit. Today's answers in its place were
    another engine's guesses ("Due Date is not a date of Month-end stock on hand").
    """
    if engine != "core2":
        return False
    question_id = new_question_id()
    try:
        over = await asyncio.to_thread(_over_limit, account_id, str((portal_user or {}).get("lang") or "en"))
    except Exception as exc:     # noqa: BLE001 - an unreadable limit does not stop the question
        log.warning("core2 could not read the limits of %s: %s", account_id, exc)
        over = ""
    if over:
        await _send(adapter, websocket, {"type": "message", "role": "assistant", "content": over})
        return True
    start = time.perf_counter()

    async def in_line() -> None:
        from core.i18n import t

        lang = str((portal_user or {}).get("lang") or "en")
        await _send(adapter, websocket, {"type": "status", "stage": "queued",
                                         "label": t("ui.chat.in_line", lang=lang),
                                         "detail": t("ui.chat.in_line_detail", lang=lang)})

    status, payload = await _answer(account_id, question, portal_user,
                                    _session_key(account_id, portal_user, adapter.thread_id), "core2", question_id,
                                    waiting=in_line)
    if payload is None:
        await _send(adapter, websocket, _could_not(question, status, question_id, beside=False))
        return True
    rows = payload.pop("export_rows", None)
    try:
        payload["trace_id"] = await asyncio.to_thread(
            _keep, account_id, portal_user, str(getattr(adapter, "session_id", "") or ""), question, question_id,
            payload, rows, int((time.perf_counter() - start) * 1000))
    except Exception as exc:     # noqa: BLE001 - the answer stands; its history and usage row are logged as lost
        log.warning("core2 answer could not be kept in history and usage for %s: %s", account_id, exc)
    await _with_pin(account_id, portal_user, question, payload)
    sent = await _send(adapter, websocket, payload)
    if sent and payload.get("data") is not None:
        _not_todays_result(adapter)
        from core.background_tasks import spawn

        spawn(_summarize(adapter, websocket, account_id, question, payload), name="core2-summary")
    # Handled, sent or not: a frame that could not reach the reader (the socket closed) is logged by _send, and
    # today's pipeline is not run in its place -- it would query the warehouse again for nobody, and put another
    # engine's answer in the thread's history.
    return True


async def _summarize(adapter: Any, websocket: Any, account_id: str, question: str, payload: dict[str, Any]) -> None:
    """The answer's written summary, sent after it (the answer never waits for the AI) and kept with it."""
    from core2.service import portal_summary

    question_id = str((payload.get("trust") or {}).get("question_id") or "")
    try:
        # Not on the new core's own threads: a summary never keeps the next question waiting.
        text = await asyncio.wait_for(asyncio.to_thread(portal_summary, account_id, question, payload,
                                                        question_id=question_id), TIMEOUT_SECONDS)
    except Exception as exc:     # noqa: BLE001 - the answer stands without its summary
        log.warning("core2 could not summarise an answer for %s: %s", account_id, exc)
        return
    if not text:
        return
    await _send(adapter, websocket, {"type": "answer_summary", "question_id": question_id, "summary": text})
    if payload.get("trace_id"):
        try:
            await asyncio.to_thread(_keep_summary, int(payload["trace_id"]), text)
        except Exception as exc:     # noqa: BLE001 - shown; only a reopened thread will lack it
            log.warning("core2 could not keep the summary of trace %s: %s", payload.get("trace_id"), exc)


def _keep_summary(trace_id: int, text: str) -> None:
    """The summary goes into the kept answer, so a reopened thread shows it too."""
    import store

    trace = store.get_answer_trace(trace_id) or {}
    try:
        kept = json.loads(trace.get("answer_frame") or "null")
    except ValueError:
        return
    if isinstance(kept, dict):
        store.update_answer_trace(trace_id, answer_frame={**kept, "summary": text})


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
    "busy": "Too many questions are being answered right now for this one to start.",
    "failed": "The new core stopped with an error on this question; the service log has the details.",
}


def _could_not(question: str, status: str, question_id: str, *, beside: bool = True) -> dict[str, Any]:
    """A card saying the new core gave no answer, and why: beside today's answer, or alone in ``core2`` mode."""
    text = _COULD_NOT.get(status, _COULD_NOT["failed"]).format(seconds=TIMEOUT_SECONDS)
    after = "Today's answer above stands." if beside else "Ask again in a moment, or ask it another way."
    return {"type": "assistant_response", "engine": "core2", "question": question, "kind": "could_not",
            "answer": {"headline": f"{text} {after}", "short_value": "", "comparison": "",
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
    # A preview's chart can be added to a dashboard as the new core's: its tile runs the preview's own plan.
    await _with_pin(account_id, portal_user, question, payload)
    await _send(adapter, websocket, payload)
