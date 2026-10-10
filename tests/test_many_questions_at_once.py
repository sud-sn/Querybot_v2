"""Many readers asking at once: more threads, a fair line per workspace, and the model kept between questions.

The new core answered with four threads shared by every workspace, so about four questions ran at once; one
workspace asking many questions took them all; and a question waiting for a thread spent its 120 seconds
waiting, then was told it had taken too long. The learned model was read from the store and parsed again for
every question and every dashboard tile.

Now the threads are 16 unless set (QUERYBOT_CORE2_THREADS, 4 to 32). Each workspace has its own line and uses at
most a quarter of them at once, so another workspace's question starts at once. A question that has to wait
is said to be in line, and its 120 seconds start when a thread takes it; one that cannot start within five
minutes is told so, and the new core is not asked. A thread is given back when it finishes, not when its
reader stops waiting. The model is kept per workspace with what it is made of -- the latest version, every
admin decision, discovery's file -- and built again when any of them changes; answering never changes it.

The new core's work is a stand-in here, except where the learned retail workspace answers for real.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import threading
import time
from types import SimpleNamespace

import pytest

import gateway.core2_bridge as bridge
from tests.test_core2_answers_in_the_portal import ACCOUNT, CANNED, _Connection, learned  # noqa: F401
from tests.test_core2_answers_in_the_portal import workspace  # noqa: F401 - the fixture under learned

# ── the threads ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("setting,threads", [(None, 16), ("32", 32), ("24", 24), ("100", 32), ("1", 4),
                                             ("lots", 16), ("", 16)])
def test_the_new_core_answers_with_16_threads_unless_set(monkeypatch, setting, threads):
    if setting is None:
        monkeypatch.delenv("QUERYBOT_CORE2_THREADS", raising=False)
    else:
        monkeypatch.setenv("QUERYBOT_CORE2_THREADS", setting)
    assert bridge._threads() == threads


def test_the_running_service_has_them():
    assert bridge.THREADS >= 16 and bridge._POOL._max_workers == bridge.THREADS
    assert bridge.PER_WORKSPACE == max(2, bridge.THREADS // 4)


# ── the line ──────────────────────────────────────────────────────────────


class _Work:
    """The new core's answers, held until the test lets each go: who started, in order, and how many at once."""

    def __init__(self, hold: float = 5.0):
        self.started: list[str] = []
        self.running = 0
        self.most = 0
        self.lock = threading.Lock()
        self.go: dict[str, threading.Event] = {}
        self.hold = hold

    def release(self, *questions: str) -> None:
        for q in questions:
            self.go.setdefault(q, threading.Event()).set()

    def __call__(self, account_id, question, portal_user=None, **kwargs):
        with self.lock:
            self.started.append(question)
            self.running += 1
            self.most = max(self.most, self.running)
            gate = self.go.setdefault(question, threading.Event())
        try:
            gate.wait(self.hold)
            return json.loads(json.dumps({**CANNED, "question": question}))
        finally:
            with self.lock:
                self.running -= 1


@pytest.fixture
def small(monkeypatch):
    """Four threads, two per workspace, so the line is seen with a handful of questions."""
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="core2-test")
    monkeypatch.setattr(bridge, "THREADS", 4)
    monkeypatch.setattr(bridge, "PER_WORKSPACE", 2)
    monkeypatch.setattr(bridge, "_POOL", pool)
    monkeypatch.setattr(bridge, "_record", lambda *a, **k: None)
    yield
    pool.shutdown(wait=False, cancel_futures=True)


def _ask(account: str, question: str, waited: list | None = None):
    async def waiting():
        if waited is not None:
            waited.append(question)

    return bridge._answer(account, question, {"id": 1}, f"s-{question}", "core2", f"q-{question}", waiting=waiting)


async def _until(condition, seconds: float = 3.0) -> None:
    end = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < end, "timed out waiting"
        await asyncio.sleep(0.01)


def test_one_workspace_asking_many_questions_never_keeps_another_waiting(small, monkeypatch):
    work = _Work()
    monkeypatch.setattr("core2.service.portal_answer", work)

    async def run():
        mine = [asyncio.create_task(_ask("busy-workspace", f"A{i}")) for i in range(6)]
        await _until(lambda: len(work.started) == 2)
        await asyncio.sleep(0.1)
        assert work.started == ["A0", "A1"], "one workspace took more than its share of the threads"
        theirs = asyncio.create_task(_ask("other-workspace", "B0"))
        await _until(lambda: "B0" in work.started)
        assert work.started == ["A0", "A1", "B0"], "the other workspace's question waited behind the busy one's"
        work.release("B0")
        assert (await theirs)[0] == "answered"
        work.release(*[f"A{i}" for i in range(6)])
        statuses = [s for s, _ in await asyncio.gather(*mine)]
        assert statuses == ["answered"] * 6
        assert work.started[3:] == ["A2", "A3", "A4", "A5"], "the busy workspace's line was not first come"

    asyncio.run(run())
    assert work.most <= 3


def test_every_thread_is_used_when_workspaces_ask_together(small, monkeypatch):
    work = _Work()
    monkeypatch.setattr("core2.service.portal_answer", work)

    async def run():
        tasks = [asyncio.create_task(_ask(f"workspace-{w}", f"{w}{i}")) for w in "ABC" for i in range(2)]
        await _until(lambda: len(work.started) == 4)
        await asyncio.sleep(0.1)
        assert len(work.started) == 4 and work.running == 4
        work.release(*[f"{w}{i}" for w in "ABC" for i in range(2)])
        assert [s for s, _ in await asyncio.gather(*tasks)] == ["answered"] * 6

    asyncio.run(run())


def test_waiting_in_line_does_not_count_toward_the_answering_limit(small, monkeypatch):
    def answer(account_id, question, portal_user=None, **kwargs):
        time.sleep(0.4)
        return json.loads(json.dumps(CANNED))

    monkeypatch.setattr("core2.service.portal_answer", answer)
    monkeypatch.setattr(bridge, "TIMEOUT_SECONDS", 0.6)

    async def run():
        # Two at a time for this workspace: the third waits 0.4 seconds, then answers in 0.4 -- 0.8 in all.
        return await asyncio.gather(*[_ask("one-workspace", f"Q{i}") for i in range(3)])

    started = time.monotonic()
    statuses = [s for s, _ in asyncio.run(run())]
    assert time.monotonic() - started > 0.6
    assert statuses == ["answered"] * 3


def test_a_question_that_waits_is_said_to_be_in_line_and_one_that_starts_at_once_is_not(small, monkeypatch):
    work = _Work()
    monkeypatch.setattr("core2.service.portal_answer", work)
    waited: list[str] = []

    async def run():
        tasks = [asyncio.create_task(_ask("one-workspace", f"Q{i}", waited)) for i in range(3)]
        await _until(lambda: len(work.started) == 2)
        await asyncio.sleep(0.05)
        work.release("Q0", "Q1", "Q2")
        await asyncio.gather(*tasks)

    asyncio.run(run())
    assert waited == ["Q2"]


def test_a_question_that_cannot_start_in_time_is_told_and_the_new_core_is_not_asked(small, monkeypatch):
    work = _Work()
    monkeypatch.setattr("core2.service.portal_answer", work)
    monkeypatch.setattr(bridge, "WAIT_LIMIT_SECONDS", 0.3)

    async def run():
        held = [asyncio.create_task(_ask("one-workspace", f"Q{i}")) for i in range(2)]
        await _until(lambda: len(work.started) == 2)
        status, payload = await _ask("one-workspace", "late")
        work.release("Q0", "Q1")
        await asyncio.gather(*held)
        return status, payload

    status, payload = asyncio.run(run())
    assert (status, payload) == ("busy", None)
    assert "late" not in work.started
    card = bridge._could_not("late", status, "q-late", beside=False)
    assert card["answer"]["headline"] == ("Too many questions are being answered right now for this one to start. "
                                          "Ask again in a moment, or ask it another way.")


def test_a_thread_is_given_back_when_it_finishes_not_when_its_reader_stops_waiting(small, monkeypatch):
    monkeypatch.setattr(bridge, "PER_WORKSPACE", 1)
    started: dict[str, float] = {}

    def answer(account_id, question, portal_user=None, **kwargs):
        started[question] = time.monotonic()
        time.sleep(0.6 if question == "slow" else 0.01)
        return json.loads(json.dumps(CANNED))

    monkeypatch.setattr("core2.service.portal_answer", answer)
    monkeypatch.setattr(bridge, "TIMEOUT_SECONDS", 0.2)

    async def run():
        slow = asyncio.create_task(_ask("one-workspace", "slow"))
        await _until(lambda: "slow" in started)
        nxt = asyncio.create_task(_ask("one-workspace", "next"))
        return await slow, await nxt

    (slow_status, _), (next_status, _) = asyncio.run(run())
    assert slow_status == "timeout" and next_status == "answered"
    assert started["next"] - started["slow"] >= 0.55, "the next question took a thread still answering"


def test_a_question_given_up_while_in_line_gives_its_place_back(small, monkeypatch):
    work = _Work()
    monkeypatch.setattr("core2.service.portal_answer", work)

    async def run():
        held = [asyncio.create_task(_ask("one-workspace", f"Q{i}")) for i in range(2)]
        await _until(lambda: len(work.started) == 2)
        waiting = asyncio.create_task(_ask("one-workspace", "gone"))
        await asyncio.sleep(0.05)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        work.release("Q0", "Q1")
        await asyncio.gather(*held)
        await _until(lambda: bridge._line().free("one-workspace"))
        status, _ = await _ask("one-workspace", "after")
        return status

    work.release("after")
    assert asyncio.run(run()) == "answered"
    assert "gone" not in work.started


def test_work_the_threads_refuse_gives_its_place_back(small, monkeypatch):
    class Refusing:
        def submit(self, *a, **k):
            raise RuntimeError("cannot schedule new futures after shutdown")

    async def run():
        with monkeypatch.context() as m:
            m.setattr(bridge, "_POOL", Refusing())
            status, _ = await _ask("one-workspace", "refused")
        assert bridge._line().free("one-workspace")
        return status

    assert asyncio.run(run()) == "failed"


def test_the_reader_sees_the_in_line_notice_in_their_language(small, monkeypatch):
    work = _Work()
    monkeypatch.setattr("core2.service.portal_answer", work)
    monkeypatch.setattr(bridge, "_over_limit", lambda *a, **k: "")
    monkeypatch.setattr(bridge, "_keep", lambda *a, **k: 1)

    async def no_pin(*a, **k):
        return None

    monkeypatch.setattr(bridge, "_with_pin", no_pin)
    monkeypatch.setattr(bridge, "_summarize", no_pin)
    sent: dict[str, list] = {"en": [], "fr": []}

    def reader(lang):
        class Socket:
            async def send_json(self, payload):
                sent[lang].append(payload)

        return SimpleNamespace(thread_id=f"t-{lang}", send_lock=asyncio.Lock(), session_id="s"), Socket()

    async def run():
        held = [asyncio.create_task(_ask("one-workspace", f"Q{i}")) for i in range(2)]
        await _until(lambda: len(work.started) == 2)
        asks = []
        for lang in ("en", "fr"):
            adapter, socket = reader(lang)
            asks.append(asyncio.create_task(bridge.answer_instead(
                "core2", adapter, socket, "one-workspace", f"question {lang}", {"id": 7, "lang": lang})))
        await asyncio.sleep(0.2)
        work.release("Q0", "Q1", "question en", "question fr")
        await asyncio.gather(*held, *asks)

    asyncio.run(run())
    for lang, label, detail in (
            ("en", "Your question is in line",
             "Other questions are being answered; yours starts as soon as one finishes."),
            ("fr", "Votre question est dans la file",
             "D'autres questions sont en cours ; la vôtre commence dès que l'une se termine.")):
        status, answer = sent[lang][0], sent[lang][-1]
        assert status == {"type": "status", "stage": "queued", "label": label, "detail": detail}
        assert answer["type"] == "assistant_response" and answer["answer"]["headline"] == "New core says 42."


# ── the model, kept ───────────────────────────────────────────────────────


@pytest.fixture
def kept(learned, monkeypatch):  # noqa: F811
    """The learned retail workspace, its kept models forgotten, and every build of its model counted."""
    import core2.bootstrap.service as boot

    monkeypatch.setattr(boot, "_ANSWERING", type(boot._ANSWERING)())
    builds: list[int] = []
    real = boot.load_model

    def load_model(*args, **kwargs):
        builds.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(boot, "load_model", load_model)
    store, built, schema_dir = learned
    return SimpleNamespace(store=store, built=built, schema_dir=schema_dir, builds=builds, boot=boot)


def _model(k):
    return k.boot.answering_model(ACCOUNT, k.store.get_client(ACCOUNT))


def test_the_model_is_built_once_and_shared_by_every_question(kept):
    first = _model(kept)
    assert first is not None and all(_model(kept) is first for _ in range(5))
    assert len(kept.builds) == 1


def _a_decision(k):
    db_id = k.store.get_client(ACCOUNT)["db_config_id"]
    measure = next(iter(_model(k).measures))
    k.store.set_core2_override(ACCOUNT, db_id, f"measure:{measure}", "business_name", "Takings")
    return db_id, measure


@pytest.mark.parametrize("change", ["an admin decision", "a decision changed", "a decision undone",
                                    "a new version learned", "discovery written again"])
def test_the_model_is_built_again_when_what_it_is_made_of_changes(kept, change):
    first = _model(kept)
    db_id = kept.store.get_client(ACCOUNT)["db_config_id"]
    if change == "an admin decision":
        _a_decision(kept)
    elif change == "a decision changed":
        _, measure = _a_decision(kept)
        middle = _model(kept)
        kept.store.set_core2_override(ACCOUNT, db_id, f"measure:{measure}", "business_name", "Receipts")
        assert middle is not first
        first = middle
    elif change == "a decision undone":
        _, measure = _a_decision(kept)
        first = _model(kept)
        kept.store.delete_core2_override(ACCOUNT, db_id, f"measure:{measure}", "business_name")
    elif change == "a new version learned":
        row = kept.store.load_core2_model(ACCOUNT, db_id)
        kept.store.save_core2_model(ACCOUNT, db_id, row["model_json"], source_hash="next", stats={})
    else:
        path = kept.schema_dir / "_schema.json"
        path.write_text(path.read_text() + " ")
    again = _model(kept)
    assert again is not first, "the kept model answered after what it is made of changed"
    assert _model(kept) is again
    if change == "an admin decision":
        assert any(m.business_name == "Takings" for m in again.measures.values())
    if change == "a decision changed":
        assert any(m.business_name == "Receipts" for m in again.measures.values())
    if change == "a decision undone":
        assert not any(m.business_name == "Takings" for m in again.measures.values())


def test_a_model_is_kept_for_each_workspace_and_database_and_store(kept, monkeypatch, tmp_path):
    first = _model(kept)
    client = kept.store.get_client(ACCOUNT)
    assert kept.boot.answering_model(ACCOUNT, {**client, "db_config_id": 999}) is None
    assert kept.boot.answering_model("another-workspace", client) is None
    with monkeypatch.context() as elsewhere:
        elsewhere.setenv("QUERYBOT_DB_PATH", str(tmp_path / "another.db"))
        kept.store.init_db()
        assert kept.boot.answering_model(ACCOUNT, client) is None, "another store's workspace got this one's model"
    assert _model(kept) is first


def test_no_more_than_the_kept_number_of_workspaces_are_held(kept, monkeypatch):
    monkeypatch.setattr(kept.boot, "ANSWERING_KEPT", 1)
    kept.boot._ANSWERING[("another", 1)] = (("a stamp",), object())
    model = _model(kept)
    assert list(kept.boot._ANSWERING) == [(ACCOUNT, kept.store.get_client(ACCOUNT)["db_config_id"])]
    assert _model(kept) is model


def test_answering_never_changes_the_kept_model(kept, monkeypatch):
    import core2.bootstrap.ai as ai
    from core2 import service
    from core2.warehouse import querybot

    monkeypatch.setattr(ai, "workspace_planner", lambda *a, **k: (lambda stable, tail: ""))
    monkeypatch.setattr(querybot, "QueryBotWarehouse", lambda db_type, credentials, **kw: _Connection(kept.built.con))
    monkeypatch.setattr(service, "_INDEXES", type(service._INDEXES)())
    model = _model(kept)
    before = model.model_dump_json()
    window = {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}
    plans = [
        {"kind": "query", "intent": "value", "measures": ["net_amount"], "time": {"window": window}},
        {"kind": "query", "intent": "breakdown", "measures": ["net_amount"], "group_by": ["customer"],
         "time": {"window": window}},
        {"kind": "query", "intent": "share", "measures": ["net_amount"], "group_by": ["region.name"],
         "time": {"window": window}},
        {"kind": "query", "intent": "trend", "measures": ["net_amount"], "time": {"grain": "month", "window": window}},
        {"kind": "query", "intent": "rank", "measures": ["net_amount"], "group_by": ["store"], "limit": 3,
         "time": {"window": window}},
        {"kind": "query", "intent": "compare", "measures": ["net_amount"], "group_by": ["store"],
         "time": {"window": window, "compare": {"kind": "previous_period"}}},
        {"kind": "query", "intent": "list", "group_by": ["store"]},
    ]
    from core2.resolve.resolver import ResolveError

    # An admin, answered; and a reader with no tables, refused: neither changes the model.
    admin, reader = {"id": 1, "role": "admin"}, {"id": 2, "role": "analyst"}
    answered = refused = 0
    for plan in plans:
        for person in (admin, reader):
            try:
                payload = service.portal_replay(ACCOUNT, plan, person, question="q")
            except ResolveError:
                refused += 1
                continue
            answered += bool(payload.get("data") or payload.get("kpi"))
    assert answered >= 5 and refused >= 5, (answered, refused)
    assert _model(kept) is model and model.model_dump_json() == before
    assert len(kept.builds) == 1
