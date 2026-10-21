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
import threading
import time
import uuid
import weakref
from typing import Any

log = logging.getLogger("querybot.core2")

TIMEOUT_SECONDS = 120.0
# A dashboard the reader described: about a chat turn per tile, built several at once on their own threads, its
# card filling in as each lands. It may take longer than one question.
DASHBOARD_SECONDS = 600.0
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
    """The new core: it answers every workspace."""
    return "core2"


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
                  mode: str, question_id: str, *, waiting: Any = None, progress: Any = None,
                  limit: float | None = None) -> tuple[str, dict[str, Any] | None]:
    """How the new core did (answered, replied, unsupported, timeout, busy, failed) and its frame, if any.

    The question takes its place in the fair line first; ``waiting`` (a coroutine function) is awaited once
    when it has to wait, to tell the reader. The answering limit (``limit`` seconds) starts when a thread takes
    it; a question's (TIMEOUT_SECONDS) unless said. ``progress`` is shown a described dashboard's card as its tiles
    land, from the answering thread."""
    limit = TIMEOUT_SECONDS if limit is None else limit
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
                                                  session_key=session_key, question_id=question_id,
                                                  progress=progress))
        except BaseException:
            line.leave(account_id)       # no thread took it: its place goes back now
            raise

        def done(_: concurrent.futures.Future) -> None:
            try:
                loop.call_soon_threadsafe(line.leave, account_id)
            except RuntimeError:     # the loop has closed: its line went with it
                pass

        work.add_done_callback(done)
        payload = await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(work)), limit)
        status = ("unsupported" if payload.get("unsupported") else
                  "answered" if payload.get("data") else "replied")
    except asyncio.TimeoutError:
        if entered:
            status = "timeout"
            log.warning("core2 took over %.0fs on a question for %s", limit, account_id)
        else:
            status = "busy"
            log.warning("core2 was too busy to start a question for %s within %.0fs", account_id,
                        WAIT_LIMIT_SECONDS)
    except Exception as exc:     # noqa: BLE001 - logged loudly; today's answer stands
        from core.llm import is_rate_limited

        if is_rate_limited(exc):
            # The workspace's AI deployment is over its requests or tokens a minute: it passes on its own.
            status = "ai_busy"
            log.warning("core2: the AI of %s is at its rate limit: %s", account_id, exc)
        else:
            log.warning("core2 failed on a question for %s: %s", account_id, exc, exc_info=True)
    try:
        await asyncio.to_thread(_record, account_id, portal_user, mode, question, status, payload,
                                int((time.perf_counter() - start) * 1000), question_id)
    except Exception as exc:     # noqa: BLE001
        log.warning("core2 answer could not be recorded for %s: %s", account_id, exc)
    return status, payload


def _keep(account_id: str, portal_user: dict | None, session_id: str, question: str, question_id: str,
          payload: dict[str, Any], rows: list[dict] | None, duration_ms: int, source: str = "portal") -> int:
    """An answer kept as today's are: its trace (history, full export, audit) and its query-log row (usage).
    ``source``: where it was asked (the portal, Teams, Slack, Zoom or the question API)."""
    import store
    from core2.service import question_scrubber

    scrub = question_scrubber(account_id)
    text = scrub(question) if scrub else question
    trust = payload.get("trust") or {}
    sql = str(trust.get("sql") or "")
    row_count = int(trust.get("row_count") or 0)
    user_id = int(portal_user["id"]) if portal_user and portal_user.get("id") else None
    trace_id = store.create_answer_trace(account_id=account_id, question_id=question_id, question_text=text,
                                         portal_user_id=user_id, session_id=session_id, request_source=source,
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
    reopened answer gets a new Add to dashboard token, a metric the reader made shows
    without the buttons that would save or forget it, and the chips that change a setting
    of the answer ("Median instead") are left out.
    """
    frame = {k: v for k, v in payload.items() if k not in ("pin_token", "trace_id", "export_rows")}
    frame["question"] = question
    if isinstance(frame.get("own_metric"), dict):
        frame["own_metric"] = {k: v for k, v in frame["own_metric"].items() if k != "token"}
    if isinstance(frame.get("figures"), list):
        frame["figures"] = [{k: v for k, v in f.items() if k not in ("token", "measure")}
                            for f in frame["figures"] if isinstance(f, dict)]
    if isinstance(frame.get("chart"), dict):
        frame["chart"] = {k: v for k, v in frame["chart"].items() if k != "pin_token"}
    if isinstance(frame.get("follow_up_suggestions"), list):
        # A chip that changes a setting runs a plan kept in the chat it was offered in: it ends with the chat.
        frame["follow_up_suggestions"] = [s for s in frame["follow_up_suggestions"]
                                          if not (isinstance(s, dict) and s.get("kind") == "setting")]
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


def _created(account_id: str, portal_user: dict | None, payload: dict[str, Any], question_id: str,
             thread_id: str = "default") -> dict[str, Any]:
    """A dashboard the reader asked for, saved at once as a new dashboard of theirs (one version, "Built from your
    request", on the design's tabs): the chat says its name, its tabs, what could not be built, and links to it.
    The tiles' plans never travel to the page."""
    import store
    from urllib.parse import quote

    from core.i18n import t

    lang = str((portal_user or {}).get("lang") or "en")
    if not portal_user or not portal_user.get("id"):
        return {"type": "message", "role": "assistant", "question_id": question_id, "state": "failed",
                "content": t("ui.chat.dashmade.sign_in", lang=lang)}
    client = store.get_client(account_id) or {}
    tiles = payload.get("dashboard_tiles") or []
    saved = store.save_built_dashboard(account_id, int(portal_user["id"]), name=str(payload.get("name") or "Dashboard"),
                                       tiles=tiles, db_config_id=int(client.get("db_config_id") or 0) or None,
                                       thread_id=thread_id, tab_names=payload.get("tabs"))
    url = f"/portal/dashboard?dashboard_id={saved['id']}"
    tabs = json.loads(saved.get("tabs_json") or "[]") or ["Overview"]
    on = collections.Counter(str(tile.get("tab") or tabs[0]) for tile in tiles)
    log.info("A dashboard was built from a request for %s: %s tiles on %s tabs, dashboard %s", account_id,
             len(tiles), len(tabs), saved["id"])
    return {"type": "dashboard_created", "engine": "core2", "question": payload.get("question") or "",
            "question_id": question_id,
            "dashboard": {"id": saved["id"], "name": saved["name"], "url": url},
            "tiles": len(tiles),
            "tabs": [{"name": tab, "tiles": on.get(tab, 0), "url": f"{url}&tab={quote(tab)}"} for tab in tabs],
            "not_built": list(payload.get("not_built") or []), "checks": list(payload.get("checks") or []),
            "missing": list(payload.get("missing") or []), "instead": list(payload.get("instead") or []),
            "widened": list(payload.get("widened") or []), "waiting": list(payload.get("waiting") or [])}


# ── Changes to a dashboard asked for in the chat (core2/dashboard_talk.py) ──

_ON: collections.OrderedDict[str, int] = collections.OrderedDict()   # conversation -> the dashboard it is on
_ON_LOCK = threading.Lock()
_ON_MOST = 5000


def _on_dashboard(key: str, dashboard_id: int | None) -> None:
    """The dashboard a conversation is on (just made or changed there), or none (it has moved on)."""
    with _ON_LOCK:
        _ON.pop(key, None)
        if dashboard_id:
            _ON[key] = dashboard_id
            while len(_ON) > _ON_MOST:
                _ON.popitem(last=False)


def _dashboard_on(key: str) -> int | None:
    with _ON_LOCK:
        return _ON.get(key)


def _talk_board(account_id: str, user_id: int, adapter: Any, key: str) -> tuple[dict | None, list[dict]]:
    """The dashboard a change in the chat is about: the one the chat was opened from, the one this
    conversation just made or changed, or the one its thread made last; and its tiles."""
    import store

    board = None
    selected = int(getattr(adapter, "selected_dashboard_id", 0) or 0)
    if selected:
        board = store.get_dashboard_for_view(selected, user_id, account_id)
    on = _dashboard_on(key)
    if board is None and on:
        board = store.get_dashboard(on, user_id, account_id)
    if board is None:
        board = store.latest_dashboard_for_thread(account_id, user_id, str(getattr(adapter, "thread_id", "") or
                                                                            "default"))
    if board is None:
        return None, []
    if int(board["user_id"]) != user_id:
        return board, store.list_dashboard_charts_for_view(int(board["id"]), user_id, account_id)
    return board, store.list_dashboard_charts(int(board["id"]), user_id)


def _talk_read(account_id: str, portal_user: dict, adapter: Any, question: str, key: str) -> tuple[Any, dict | None]:
    """What the words ask of the dashboard the conversation is on (None: they ask nothing of one), and it."""
    from core2 import dashboard_talk as talk

    import store

    explicit = talk.about_a_dashboard(question)
    on = _dashboard_on(key)
    if not explicit and not on and not getattr(adapter, "selected_dashboard_id", 0):
        return None, None
    user_id = int(portal_user["id"])
    named = None
    if explicit:
        # "Add net amount by region to the Board pack dashboard": that dashboard of theirs, whichever is open.
        mine = store.list_editable_dashboards(account_id, user_id)
        named = talk.naming(question, [str(d["name"]) for d in mine])
    if named is not None:
        row = next(d for d in mine if str(d["name"]) == named[0])
        question, charts = named[1], store.list_dashboard_charts(int(row["id"]), user_id)
    else:
        row, charts = _talk_board(account_id, user_id, adapter, key)
    tiles = []
    for chart in charts:
        try:
            config = json.loads(chart.get("display_config") or "{}")
        except ValueError:
            config = {}
        tiles.append(talk.Tile(int(chart["id"]), str(chart.get("title") or ""), str(chart.get("dashboard_tab") or ""),
                               str(chart.get("chart_type") or ""), config.get("core2_plan") if isinstance(
                                   config.get("core2_plan"), dict) else None))
    board = talk.Board(int(row["id"]), str(row["name"]), json.loads(row.get("tabs_json") or '["Overview"]')
                       or ["Overview"], tiles) if row else talk.Board(0, "", [], [])
    reading = talk.read(question, board, on_it=bool(row) and (named is not None or on == board.id or int(
        getattr(adapter, "selected_dashboard_id", 0) or 0) == board.id))
    return reading, row


def _say(question: str, question_id: str, text: str, **extra: Any) -> dict[str, Any]:
    from core2.service import _frame

    return {**_frame(question, text, kind="dashboard_talk", **extra), "question_id": question_id}


def _talk_apply(account_id: str, portal_user: dict, question: str, question_id: str, reading: Any,
                row: dict) -> dict[str, Any]:
    """The change made, as one version of the reader's dashboard, and the card that says so."""
    import store

    user_id = int(portal_user["id"])
    board_id = int(row["id"])
    if reading.kind == "undo":
        undone = store.undo_dashboard(board_id, user_id, account_id)
        if undone is None:
            return _say(question, question_id, f"There is nothing to undo on {row['name']}: it is as it was made.")
        saved, what = undone
        said = f"Undid: {what}"
    else:
        saved = store.change_dashboard(board_id, user_id, account_id, summary=reading.said, name=reading.name,
                                       tabs=reading.tabs, remove=reading.remove, tiles=reading.tiles)
        if saved is None:
            return _say(question, question_id, "That change could not be made: the dashboard changed meanwhile. "
                                               "Open it and try again.")
        said = reading.said
    return _changed(question, question_id, saved, said)


def _changed(question: str, question_id: str, saved: dict, said: str) -> dict[str, Any]:
    url = f"/portal/dashboard?dashboard_id={saved['id']}"
    return {"type": "dashboard_changed", "engine": "core2", "question": question, "question_id": question_id,
            "dashboard": {"id": int(saved["id"]), "name": saved["name"], "url": url},
            "change": said, "version": int(saved.get("version") or 1), "undo": True}


def _talk_add(account_id: str, portal_user: dict, question: str, question_id: str, reading: Any, row: dict,
              payload: dict[str, Any] | None) -> dict[str, Any]:
    """A tile added from the answer to the question it names, on the tab it names, as one version."""
    import store
    from core2 import dashboards

    plan = (payload or {}).get("plan") or {}
    kind = ("trend" if (plan.get("time") or {}).get("grain") and not plan.get("group_by") else
            "table" if plan.get("group_by") and plan.get("limit") else "breakdown" if plan.get("group_by") else "number")
    title = reading.question[:1].upper() + reading.question[1:]
    tile = dashboards.answered(dashboards.Tile(title=title, kind=kind, question=reading.question, tab=reading.tab),
                               lambda asked: payload or {})
    if tile.error:
        return _say(question, question_id, f"I could not add {reading.question}: {tile.error}")
    dashboards.lay_out([tile])
    client = store.get_client(account_id) or {}
    saved = store.save_built_dashboard(account_id, int(portal_user["id"]), name=str(row["name"]),
                                       tiles=[dashboards.kept(tile)],
                                       db_config_id=int(client.get("db_config_id") or 0) or None,
                                       dashboard_id=int(row["id"]), tab_names=[reading.tab],
                                       summary=f"Added {title}")
    if saved is None:
        return _say(question, question_id, "That tile could not be added: the dashboard is not yours to change.")
    return _changed(question, question_id, saved, f"Added \u201c{title}\u201d to the {tile.tab} tab")


async def _talked(adapter: Any, websocket: Any, account_id: str, question: str, portal_user: dict | None,
                  question_id: str, key: str, start: float) -> bool:
    """A change to the dashboard the conversation is on, made and said; False when the words ask none."""
    from core2 import dashboard_talk as talk

    if not portal_user or not portal_user.get("id"):
        return False
    try:
        reading, row = await asyncio.to_thread(_talk_read, account_id, portal_user, adapter, question, key)
    except Exception as exc:     # noqa: BLE001 - read as a question instead
        log.warning("core2 could not read a dashboard change for %s: %s", account_id, exc)
        return False
    if reading is None:
        return False
    from core.portal_roles import can, refusal

    if not can(portal_user, "talk_edit"):
        # Changing a dashboard by talking is a creator's (core/portal_roles.py).
        frame = _say(question, question_id, refusal("talk_edit", str(portal_user.get("lang") or "en")))
    elif row is None:
        frame = _say(question, question_id, "This conversation has no dashboard to change. Open the dashboard and "
                                            "ask from its chat, or ask for a new one (\u201ccreate an executive "
                                            "dashboard\u201d).")
    elif int(row["user_id"]) != int(portal_user["id"]):
        frame = _say(question, question_id, f"{row['name']} is shared with you to read: its owner can change it.")
    elif isinstance(reading, talk.Refused):
        frame = _say(question, question_id, reading.text)
    elif isinstance(reading, talk.Which):
        frame = _say(question, question_id, reading.question, follow_up_suggestions=[
            {"label": option, "question": option} for option in reading.options])
    elif reading.kind == "add":
        status, payload = await _answer(account_id, reading.question, portal_user, f"{key}:add", "core2",
                                        question_id, limit=TIMEOUT_SECONDS)
        frame = await asyncio.to_thread(_talk_add, account_id, portal_user, question, question_id, reading, row,
                                        payload)
    else:
        try:
            frame = await asyncio.to_thread(_talk_apply, account_id, portal_user, question, question_id, reading, row)
        except Exception as exc:     # noqa: BLE001 - said, never raised to the socket
            log.warning("core2 could not change dashboard %s for %s: %s", row.get("id"), account_id, exc)
            frame = _say(question, question_id, "That change could not be made. Try again, or make it on the "
                                                "dashboard itself.")
    if frame.get("type") == "dashboard_changed":
        _on_dashboard(key, int(frame["dashboard"]["id"]))
        await _keep_card(adapter, account_id, portal_user, question, question_id, frame,
                         f"Dashboard changed: {frame['change']}", start)
    await _send(adapter, websocket, frame)
    return True


def _pin_this(account_id: str, portal_user: dict, adapter: Any, question: str, question_id: str, key: str,
              reading: Any, shown: Any) -> dict[str, Any]:
    """The answer on screen added as a tile ("add this chart to my dashboard"): to a new dashboard, to the one
    named, to the one the conversation is on, or to the reader's only one; with several and none named, which (a
    button each). Its plan is run again, as a pinned answer is drawn, and kept with its title."""
    import store
    from core2 import dashboard_talk as talk
    from core2 import dashboards
    from core2.recipe_book import kind_of
    from core2.service import portal_replay

    user_id = int(portal_user["id"])
    mine = store.list_editable_dashboards(account_id, user_id)
    names = [str(d["name"]) for d in mine]
    board, tab = None, reading.tab
    if not reading.new and reading.board:
        hit = talk.board_named(reading.board, names) or talk.board_named(reading.words, names)
        board = next((d for d in mine if str(d["name"]) == hit), None) if hit else None
        if board is None:
            listed = talk._listed([f"\u201c{n}\u201d" for n in names[:6]]) if names else ""
            return _say(question, question_id, f"You have no dashboard called \u201c{reading.board}\u201d"
                        + (f": yours are {listed}." if listed else "."), follow_up_suggestions=[
                            *({"label": f"Add this to {n}", "question": f"Add this to {n}"} for n in names[:4]),
                            {"label": "Add this to a new dashboard", "question": "Add this to a new dashboard"}])
    elif not reading.new:
        here, _ = _talk_board(account_id, user_id, adapter, key)
        if here is not None and int(here["user_id"]) == user_id:
            board = here
        elif len(mine) == 1:
            board = mine[0]
        elif mine:
            return _say(question, question_id, f"Which dashboard should \u201c{shown.title}\u201d go on?",
                        follow_up_suggestions=[
                            *({"label": f"Add this to {n}", "question": f"Add this to {n}"} for n in names[:5]),
                            {"label": "Add this to a new dashboard", "question": "Add this to a new dashboard"}])
    tabs = json.loads(board.get("tabs_json") or '["Overview"]') or ["Overview"] if board is not None else []
    if tab:
        tab = next((t for t in tabs if t.casefold() == tab.casefold()), tab)
    else:
        tab = tabs[0] if tabs else ("Vue d'ensemble" if str(portal_user.get("lang") or "") == "fr" else "Overview")
    # Drawn as its tile will be, from the plan it was answered with (a replay says no plan of its own).
    payload = {**portal_replay(account_id, shown.plan, portal_user, question=shown.question), "plan": shown.plan}
    tile = dashboards.answered(dashboards.Tile(title=shown.title, kind=kind_of(shown.plan), question=shown.question,
                                               tab=tab), lambda *_: payload)
    if tile.error:
        return _say(question, question_id, f"I could not add \u201c{shown.title}\u201d: {tile.error}")
    dashboards.lay_out([tile])
    client = store.get_client(account_id) or {}
    db_config_id = int(client.get("db_config_id") or 0) or None
    if board is None:
        saved = store.save_built_dashboard(account_id, user_id, name=reading.board or shown.title,
                                           tiles=[dashboards.kept(tile)], db_config_id=db_config_id,
                                           thread_id=str(getattr(adapter, "thread_id", "") or "default"),
                                           tab_names=[tab], summary=f"Added {shown.title}")
        url = f"/portal/dashboard?dashboard_id={saved['id']}"
        return {"type": "dashboard_created", "engine": "core2", "question": question, "question_id": question_id,
                "dashboard": {"id": saved["id"], "name": saved["name"], "url": url}, "tiles": 1,
                "tabs": [{"name": tab, "tiles": 1, "url": url}], "not_built": [], "checks": []}
    saved = store.save_built_dashboard(account_id, user_id, name=str(board["name"]), tiles=[dashboards.kept(tile)],
                                       db_config_id=db_config_id, dashboard_id=int(board["id"]), tab_names=[tab],
                                       summary=f"Added {shown.title}")
    if saved is None:
        return _say(question, question_id, "That tile could not be added: the dashboard is not yours to change.")
    return _changed(question, question_id, saved, f"Added \u201c{shown.title}\u201d to the {tile.tab} tab")


async def _pinned(adapter: Any, websocket: Any, account_id: str, question: str, portal_user: dict | None,
                  question_id: str, key: str, start: float) -> bool:
    """The answer on screen added to a dashboard, said; False when the words ask nothing of the kind."""
    from core.portal_roles import can, refusal
    from core2 import dashboard_talk as talk
    from core2.service import portal_shown

    reading = talk.this_answer(question)
    if reading is None or not portal_user or not portal_user.get("id"):
        return False
    shown = portal_shown(key)
    if not can(portal_user, "save_answer"):
        frame = _say(question, question_id, refusal("save_answer", str(portal_user.get("lang") or "en")))
    elif shown is None:
        frame = _say(question, question_id, "There is no answer on screen to add. Ask a question first, then say "
                                            "\u201cadd this to my dashboard\u201d.")
    else:
        try:
            frame = await asyncio.to_thread(_pin_this, account_id, portal_user, adapter, question, question_id, key,
                                            reading, shown)
        except Exception as exc:     # noqa: BLE001 - said, never raised to the socket
            log.warning("core2 could not add the answer on screen to a dashboard for %s: %s", account_id, exc)
            frame = _say(question, question_id, "That answer could not be added. Try again, or use Add to "
                                                "dashboard under it.")
    if frame.get("type") in ("dashboard_changed", "dashboard_created"):
        _on_dashboard(key, int(frame["dashboard"]["id"]))
        await _keep_card(adapter, account_id, portal_user, question, question_id, frame,
                         f"Added to {frame['dashboard']['name']}", start)
    await _send(adapter, websocket, frame)
    return True


async def _keep_card(adapter: Any, account_id: str, portal_user: dict | None, question: str, question_id: str,
                     frame: dict[str, Any], headline: str, start: float) -> None:
    """A dashboard's card kept in the thread like an answer: a reopened thread shows it again."""
    try:
        await asyncio.to_thread(_keep, account_id, portal_user, str(getattr(adapter, "session_id", "") or ""),
                                question, question_id, {**frame, "answer": {"headline": headline}}, None,
                                int((time.perf_counter() - start) * 1000))
    except Exception as exc:     # noqa: BLE001 - the dashboard is saved; only its card in history is lost
        log.warning("core2 could not keep a dashboard's card in history for %s: %s", account_id, exc)


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

    from core2.dashboards import asked_for
    from core.portal_roles import can, refusal

    key = _session_key(account_id, portal_user, getattr(adapter, "thread_id", "default"))
    if asked_for(question) is not None and not can(portal_user, "create_dashboard"):
        # A dashboard made from a request is a creator's (core/portal_roles.py): said before any AI is asked.
        await _send(adapter, websocket, _say(question, question_id, refusal(
            "create_dashboard", str((portal_user or {}).get("lang") or "en"))))
        return True
    if asked_for(question) is None and (
            await _pinned(adapter, websocket, account_id, question, portal_user, question_id, key, start)
            or await _talked(adapter, websocket, account_id, question, portal_user, question_id, key, start)):
        return True
    progress, limit = None, TIMEOUT_SECONDS
    if asked_for(question) is not None:
        # "Create a revenue dashboard": how far it is, said once it is designed and again as each tile lands.
        loop = asyncio.get_running_loop()
        limit = DASHBOARD_SECONDS

        def progress(card: dict[str, Any]) -> None:
            asyncio.run_coroutine_threadsafe(_send(adapter, websocket, {**card, "question_id": question_id}), loop)

    status, payload = await _answer(account_id, question, portal_user,
                                    _session_key(account_id, portal_user, adapter.thread_id), "core2", question_id,
                                    waiting=in_line, progress=progress, limit=limit)
    if payload is None:
        await _send(adapter, websocket, _could_not(question, status, question_id, beside=False, seconds=limit,
                                                   lang=str((portal_user or {}).get("lang") or "en")))
        return True
    if payload.get("type") == "dashboard_built":
        made = await asyncio.to_thread(_created, account_id, portal_user, payload, question_id,
                                       str(getattr(adapter, "thread_id", "") or "default"))
        if made.get("type") == "dashboard_created":
            _on_dashboard(key, int(made["dashboard"]["id"]))
            await _keep_card(adapter, account_id, portal_user, question, question_id, made,
                             f"Dashboard created: {made['dashboard']['name']}", start)
        await _send(adapter, websocket, made)
        return True
    _on_dashboard(key, None)         # an answer: the conversation has moved on from the dashboard
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
    # An answer, or one about the data itself: its summary, note or opening paragraph follows it.
    if sent and (payload.get("data") is not None or payload.get("kind") == "describe_data"):
        from core.background_tasks import spawn

        lang = str((portal_user or {}).get("lang") or "en")
        spawn(_summarize(adapter, websocket, account_id, question, payload, lang=lang), name="core2-summary")
    # Handled, sent or not: a frame that could not reach the reader (the socket closed) is logged by _send, and
    # today's pipeline is not run in its place -- it would query the warehouse again for nobody, and put another
    # engine's answer in the thread's history.
    return True


async def _summarize(adapter: Any, websocket: Any, account_id: str, question: str, payload: dict[str, Any], *,
                     lang: str = "en") -> None:
    """The answer's analyst note (or, for an answer with no facts, its written summary), sent after it (the answer
    never waits for the AI) and kept with it."""
    from core2.service import portal_note, portal_overview, portal_reply, portal_summary

    question_id = str((payload.get("trust") or {}).get("question_id") or "")
    if payload.get("kind") == "describe_data":
        # The cards about the data, opened by a paragraph in the workspace's AI's words.
        try:
            text = await asyncio.wait_for(asyncio.to_thread(portal_overview, account_id, question, payload,
                                                            question_id=question_id, lang=lang), TIMEOUT_SECONDS)
        except Exception as exc:     # noqa: BLE001 - the cards stand without their opening
            log.warning("core2 could not open the description of the data for %s: %s", account_id, exc)
            return
        if text:
            await _send(adapter, websocket, {"type": "answer_summary", "question_id": question_id, "summary": text})
            if payload.get("trace_id"):
                try:
                    await asyncio.to_thread(_keep_summary, int(payload["trace_id"]), text)
                except Exception as exc:     # noqa: BLE001 - shown; only a reopened thread will lack it
                    log.warning("core2 could not keep the summary of trace %s: %s", payload.get("trace_id"), exc)
        return
    if payload.get("reply"):
        # What the answer saw when it looked at itself, said as a person would: it replaces the facts' own
        # sentences in place, and stands for the summary (core2/agent/talk.py).
        try:
            said = await asyncio.wait_for(asyncio.to_thread(portal_reply, account_id, question, payload,
                                                            question_id=question_id, lang=lang), TIMEOUT_SECONDS)
        except Exception as exc:     # noqa: BLE001 - the facts' own sentences stand
            log.warning("core2 could not write the reply of an answer for %s: %s", account_id, exc)
            said = None
        if said:
            reply = {**payload["reply"], "said": said, "said_lang": lang, "by": "ai"}
            await _send(adapter, websocket, {"type": "answer_reply", "question_id": question_id, "reply": reply})
            if payload.get("trace_id"):
                try:
                    await asyncio.to_thread(_keep_frame_part, int(payload["trace_id"]), "reply", reply)
                except Exception as exc:     # noqa: BLE001 - shown; only a reopened thread will lack it
                    log.warning("core2 could not keep the reply of trace %s: %s", payload.get("trace_id"), exc)
        if not payload.get("reading"):
            return
    if payload.get("reading"):
        try:
            note = await asyncio.wait_for(asyncio.to_thread(portal_note, account_id, question, payload,
                                                            question_id=question_id, lang=lang), TIMEOUT_SECONDS)
        except Exception as exc:     # noqa: BLE001 - the answer stands with its facts' own sentences
            log.warning("core2 could not write the note of an answer for %s: %s", account_id, exc)
            return
        if not note:
            return
        await _send(adapter, websocket, {"type": "answer_note", "question_id": question_id, "note": note})
        if payload.get("trace_id"):
            try:
                await asyncio.to_thread(_keep_frame_part, int(payload["trace_id"]), "note", note)
            except Exception as exc:     # noqa: BLE001 - shown; only a reopened thread will lack it
                log.warning("core2 could not keep the note of trace %s: %s", payload.get("trace_id"), exc)
        return
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


def _keep_frame_part(trace_id: int, key: str, value: Any) -> None:
    """A part that follows the answer (its note) goes into the kept answer, so a reopened thread shows it too."""
    import store

    trace = store.get_answer_trace(trace_id) or {}
    try:
        kept = json.loads(trace.get("answer_frame") or "null")
    except ValueError:
        return
    if isinstance(kept, dict):
        store.update_answer_trace(trace_id, answer_frame={**kept, key: value})


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
    "ai_busy": "The AI this workspace uses is at its limit of requests a minute right now.",
    "failed": "The new core stopped with an error on this question; the service log has the details.",
}
_COULD_NOT_FR = {
    "timeout": "Le nouveau moteur n'a pas répondu à cette question en {seconds:.0f} secondes.",
    "busy": "Trop de questions sont en cours de réponse pour que celle-ci commence.",
    "ai_busy": "L'IA de cet espace de travail a atteint sa limite de requêtes par minute.",
    "failed": "Le nouveau moteur s'est arrêté sur une erreur pour cette question ; le journal du service en donne "
              "le détail.",
}


def _could_not(question: str, status: str, question_id: str, *, beside: bool = True,
               seconds: float | None = None, lang: str = "en") -> dict[str, Any]:
    """A card saying the new core gave no answer, and why: beside today's answer, or alone in ``core2`` mode; in
    the reader's language."""
    said = _COULD_NOT_FR if lang == "fr" else _COULD_NOT
    text = said.get(status, said["failed"]).format(seconds=TIMEOUT_SECONDS if seconds is None else seconds)
    if lang == "fr":
        after = ("La réponse d'aujourd'hui ci-dessus reste valable." if beside else
                 "Reposez la question dans un instant, ou formulez-la autrement.")
    else:
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
        payload = _could_not(question, status, question_id, lang=str((portal_user or {}).get("lang") or "en"))
    elif payload.get("kind") == "smalltalk":
        return     # nothing to compare: today's reply to "thanks" needs no second one
    payload.pop("export_rows", None)
    if isinstance(payload.get("answer"), dict):
        payload["answer"]["scope_badge"] = PREVIEW_BADGE
    payload.setdefault("result_scope", {})["badge"] = PREVIEW_BADGE
    # A preview's chart can be added to a dashboard as the new core's: its tile runs the preview's own plan.
    await _with_pin(account_id, portal_user, question, payload)
    await _send(adapter, websocket, payload)


# ── Teams, Slack, Zoom and the question API ─────────────────────────────────

CHANNEL_ROWS = 10       # a chat message shows the first rows; the portal has the rest, the chart and the export
_CHANNEL_NUMBERS = {"currency", "percent", "percentage", "count", "integer", "number", "decimal", "days"}


def _cell(value: Any, format_: str) -> str:
    from core2.answer.builder import fmt

    if value is None:
        return ""
    if format_ in _CHANNEL_NUMBERS and not isinstance(value, bool):
        return fmt(value, format_)
    return str(value).replace("|", "/").replace("\n", " ").strip()


def channel_text(payload: dict[str, Any], *, link: str = "") -> str:
    """A new-core answer as a chat message: its sentence, its figures as a small table (CHANNEL_ROWS at most),
    what the rows show, what it left out, and where the rest is. Teams draws the table as a card; Slack, Zoom and
    the question API show it as written."""
    answer = payload.get("answer") or {}
    headline = str(answer.get("headline") or "").strip()
    lines = [f"**{headline}**"] if headline else []
    general = str((payload.get("general") or {}).get("label") or "").strip()
    if general and lines:
        # Business in general, from the AI's knowledge: said first, as the portal's badge says it, so it never reads
        # as an answer from the data.
        lines.insert(0, f"_{general}_")
    data = payload.get("data") or {}
    headers = [h for h in (data.get("headers") or []) if h]
    rows = data.get("rows") or []
    single = len(rows) == 1 and len(headers) == 1
    if headers and rows and not single:
        labels = data.get("header_labels") or {}
        formats = data.get("column_formats") or {}
        lines += ["", "| " + " | ".join(str(labels.get(h) or h).replace("|", "/") for h in headers) + " |",
                  "|" + "|".join("---" for _ in headers) + "|"]
        lines += ["| " + " | ".join(_cell(r.get(h), str(formats.get(h) or "")) for h in headers) + " |"
                  for r in rows[:CHANNEL_ROWS]]
    insights = [str(i).strip() for i in (payload.get("key_insights") or []) if str(i).strip()][:3]
    if insights:
        lines += [""] + [f"- {i}" for i in insights]
    caveats = [str(c).strip() for c in (payload.get("coverage_caveats") or []) if str(c).strip()][:2]
    if caveats:
        lines += [""] + [f"_{c}_" for c in caveats]
    total = int(data.get("total_rows") or len(rows))
    more = total > CHANNEL_ROWS and not single
    if more or payload.get("chart"):
        rest = (f"The first {CHANNEL_ROWS} of {total:,} rows." if more else "")
        what = ("The chart and every row are" if payload.get("chart") else "Every row is") if more else "The chart is"
        where = f"{what} in Solviz: {link}" if link else ""
        tail = " ".join(part for part in (rest, where) if part)
        if tail:
            lines += ["", f"_{tail}_"]
    return "\n".join(lines).strip()


def _conversation(event: Any) -> str:
    """The chat a message came from: a Teams conversation, a Slack channel, a Zoom chat."""
    channel = str(getattr(event, "channel_id", "") or "")
    try:
        channel = str(json.loads(channel).get("conversation_id") or channel)
    except (ValueError, AttributeError, TypeError):
        pass
    return f"{getattr(event, 'platform', '') or 'channel'}:{channel}"


async def answer_in_channel(adapter: Any, event: Any, account_id: str, question: str, portal_user: dict | None, *,
                            conversation: str | None = None) -> str:
    """Teams, Slack, Zoom and the question API: the new core answers, as a chat message; the text sent.

    The person is the Solviz user their chat account is linked to (their roles, tables, masking and row rules),
    or nobody for the question API, which answers as the workspace. Each chat is a conversation of its own:
    "and by month?" follows the answer above it. The workspace's monthly limits apply; a dashboard is made in the
    portal. ``conversation``: the conversation to follow (the question API's caller names one, or none: each
    question on its own)."""
    from core.mail.outbox import portal_base
    from core2.dashboards import asked_for

    source = str(getattr(event, "platform", "") or "api")
    lang = str((portal_user or {}).get("lang") or "en")
    question_id = new_question_id()
    link = f"{portal_base()}/portal/chat"

    async def say(text: str) -> str:
        await adapter.send_message(event, text)
        return text

    try:
        over = await asyncio.to_thread(_over_limit, account_id, lang)
    except Exception as exc:     # noqa: BLE001 - an unreadable limit does not stop the question
        log.warning("core2 could not read the limits of %s: %s", account_id, exc)
        over = ""
    if over:
        return await say(over)
    if asked_for(question) is not None:
        return await say(f"Dashboards are made in the Solviz portal: {link}")
    thread = conversation if conversation is not None else _conversation(event)
    key = _session_key(account_id, portal_user, f"{source}:{thread or question_id}")
    start = time.perf_counter()
    status, payload = await _answer(account_id, question, portal_user, key, "core2", question_id)
    if payload is None:
        return await say(str(_could_not(question, status, question_id, beside=False, lang=lang)["answer"]["headline"]))
    if payload.get("type") == "dashboard_built":     # never asked for here; said, never saved
        return await say(f"Dashboards are made in the Solviz portal: {link}")
    rows = payload.pop("export_rows", None)
    try:
        await asyncio.to_thread(_keep, account_id, portal_user, thread, question, question_id, payload, rows,
                                int((time.perf_counter() - start) * 1000), source)
    except Exception as exc:     # noqa: BLE001 - the answer stands; its history and usage row are logged as lost
        log.warning("core2 answer could not be kept in history and usage for %s: %s", account_id, exc)
    clarify = payload.get("clarify") or {}
    options = [str(o) for o in (clarify.get("options") or []) if o]
    ask_back = getattr(adapter, "send_clarification_prompt", None)
    if options and callable(ask_back):
        # A question back: a button per option where the chat has buttons (a tap sends the option's words).
        await ask_back(event, str(clarify.get("question") or (payload.get("answer") or {}).get("headline") or ""),
                       [{"id": o, "label": o} for o in options])
        return str((payload.get("answer") or {}).get("headline") or "")
    return await say(channel_text(payload, link=link))
