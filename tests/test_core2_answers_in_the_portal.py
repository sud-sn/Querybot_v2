"""The new core in the web portal: the switch, the socket, governance, the wiring.

* The switch is per workspace and set on the "What QueryBot learned" page; the
  new core cannot be switched on before it has learned the database.
* Through the real portal socket: in "compare" mode today's answer comes first
  and the new core's follows it, badged as a preview; in "core2" mode the new
  core answers alone, and a question it cannot express goes to today's
  pipeline; a new-core failure never costs the reader today's answer, and side
  by side it is said on the preview card instead of the preview never coming.
* A "why" about the answer on screen is a question like any other: in "core2"
  mode the new core answers it first (today's analysis only for what it cannot
  express), and side by side the new core's answer follows today's analysis.
* The new core's SQL runs through QueryBot's governed executor, whose
  production rules it must pass, and returns the reference numbers.
* The production wiring finds the stored model, the reader's allowed tables and
  the workspace's AI by itself.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from unittest.mock import patch

import pytest
import sqlglot

from tests import answer_harness as harness
from tests import portal_harness as portal
from tests.test_core2_learned_page_shows_what_was_learned import (
    ACCOUNT,
    _Connection,
    _post,
    _schema_file,
    workspace,  # noqa: F401 - the fixture
)

CANNED = {"type": "assistant_response", "engine": "core2", "question": "q",
          "answer": {"headline": "New core says 42.", "short_value": "42", "comparison": "", "scope_note": ""},
          "result_scope": {"badge": "", "note": ""}, "chart": None, "kpi": None,
          "data": {"headers": ["v"], "header_labels": {"v": "V"}, "rows": [{"v": 42}], "total_rows": 1},
          "trust": {"engine": "core2", "sql": "SELECT 42 AS v", "row_count": 1, "model_version": 1}}


# ── the switch ─────────────────────────────────────────────────────────────


def test_the_switch_is_set_on_the_learned_page_and_waits_for_a_model(workspace):  # noqa: F811
    store, built, schema_dir = workspace
    from admin import core2_routes

    response = _post(core2_routes.learned_engine, f"/admin/clients/{ACCOUNT}/learned/engine", {"engine": "compare"})
    assert "error=" in response.headers["location"] and store.get_query_engine(ACCOUNT) == "legacy"
    _schema_file(built, schema_dir)
    core2_routes._run_build(ACCOUNT)
    response = _post(core2_routes.learned_engine, f"/admin/clients/{ACCOUNT}/learned/engine", {"engine": "compare"})
    assert "saved=engine" in response.headers["location"] and store.get_query_engine(ACCOUNT) == "compare"
    response = _post(core2_routes.learned_engine, f"/admin/clients/{ACCOUNT}/learned/engine", {"engine": "both"})
    assert "error=" in response.headers["location"] and store.get_query_engine(ACCOUNT) == "compare"


# ── through the portal socket ──────────────────────────────────────────────


@pytest.fixture(scope="module")
def tenant(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("core2-portal")) as built:
        yield built


def _answers(turn) -> list[dict]:
    return [f for f in turn["frames"] if f.get("type") == "assistant_response"]


def _ask(tenant, engine: str, answer, question: str = "stock on hand by warehouse"):
    import gateway.core2_bridge as bridge

    async def setting(account_id):
        return engine

    with patch.object(bridge, "engine", setting), patch("core2.service.portal_answer", answer):
        with portal.SocketConversation(tenant, harness.ACCOUNT, harness.saved_connection()) as conversation:
            return conversation.ask(question)


def test_compare_mode_shows_the_new_cores_answer_after_todays(tenant):
    turn = _ask(tenant, "compare", lambda *a, **k: json.loads(json.dumps(CANNED)))
    answers = _answers(turn)
    assert len(answers) == 2 and turn["executed"], [a.get("engine") for a in answers]
    today, preview = answers
    assert today.get("engine") != "core2"
    assert preview["engine"] == "core2" and preview["result_scope"]["badge"] == "New core (preview)"


def test_core2_mode_answers_alone_and_hands_on_what_it_cannot_express(tenant):
    turn = _ask(tenant, "core2", lambda *a, **k: json.loads(json.dumps(CANNED)))
    assert [a["engine"] for a in _answers(turn)] == ["core2"] and not turn["executed"]
    unsupported = {**CANNED, "unsupported": True}
    turn = _ask(tenant, "core2", lambda *a, **k: unsupported)
    answers = _answers(turn)
    assert answers and all(a.get("engine") != "core2" for a in answers) and turn["executed"]


def test_a_new_core_failure_never_costs_the_reader_todays_answer_and_is_said(tenant):
    def broken(*a, **k):
        raise RuntimeError("the new core fell over")

    turn = _ask(tenant, "compare", broken)
    today, card = _answers(turn)
    assert today.get("engine") != "core2" and turn["executed"]
    # Side by side, the preview that will not come says so, without the error's own words.
    assert card["engine"] == "core2" and card["kind"] == "could_not" and card["data"] is None
    assert card["answer"]["headline"] == ("The new core stopped with an error on this question; the service log "
                                          "has the details. Today's answer above stands.")
    assert card["result_scope"]["badge"] == "New core (preview)" and "fell over" not in json.dumps(card)
    import store

    logged = store.list_core2_answers(harness.ACCOUNT, 5)
    assert logged and logged[0]["status"] == "failed"
    assert logged[0]["question_id"] == card["trust"]["question_id"]     # the reader's thumbs reach the row

    turn = _ask(tenant, "core2", broken)      # answering alone, today's pipeline answers instead
    assert [a.get("engine") != "core2" for a in _answers(turn)] == [True] and turn["executed"]


def test_side_by_side_a_new_core_that_takes_too_long_says_so(learned, monkeypatch):
    import time

    import gateway.core2_bridge as bridge

    def slow(*a, **k):
        time.sleep(1.5)
        return json.loads(json.dumps(CANNED))

    monkeypatch.setattr("core2.service.portal_answer", slow)
    monkeypatch.setattr(bridge, "TIMEOUT_SECONDS", 0.3)
    sent: list[dict] = []

    class Adapter:
        thread_id = "t1"
        send_lock = asyncio.Lock()

    class Socket:
        async def send_json(self, payload):
            sent.append(payload)

    asyncio.run(bridge.answer_beside(Adapter(), Socket(), ACCOUNT, "net sales in 2025", {"id": 1}))
    (card,) = sent
    assert card["kind"] == "could_not" and card["result_scope"]["badge"] == "New core (preview)"
    assert card["answer"]["headline"] == "The new core did not answer this within 0 seconds. Today's answer above stands."
    import store

    assert store.list_core2_answers(ACCOUNT, 1)[0]["status"] == "timeout"


# ── a "why" about the answer on screen ────────────────────────────────────

WHY = "why is NORTH DEPOT the highest?"
EXPLAINED = {**CANNED, "answer": {**CANNED["answer"], "headline": "The new core explains it."}}


class _NewCore:
    """portal_answer as the bridge calls it: what it was asked, and its answer to each question."""

    def __init__(self, answers: dict[str, dict]):
        self.answers, self.asked = answers, []

    def __call__(self, account_id, question, *args, **kwargs):
        self.asked.append(question)
        return json.loads(json.dumps(self.answers.get(question, {**CANNED, "unsupported": True})))


def _conversation(tenant, engine: str, new_core: _NewCore):
    import contextlib

    import gateway.core2_bridge as bridge

    async def setting(account_id):
        return engine

    stack = contextlib.ExitStack()
    stack.enter_context(patch.object(bridge, "engine", setting))
    stack.enter_context(patch("core2.service.portal_answer", new_core))
    conversation = stack.enter_context(
        portal.SocketConversation(tenant, harness.ACCOUNT, harness.saved_connection(), idle_seconds=2.0))
    return stack, conversation


def _analyses(turn) -> list[str]:
    return [str(f.get("title") or "") for f in turn["frames"] if f.get("type") == "assistant_analysis"]


def test_in_core2_mode_a_why_about_the_answer_on_screen_is_the_new_cores(tenant):
    new_core = _NewCore({WHY: EXPLAINED})
    stack, conversation = _conversation(tenant, "core2", new_core)
    with stack:
        first = conversation.ask("stock on hand by warehouse")    # the new core cannot: today's answer, on screen
        why = conversation.ask(WHY)
    assert _answers(first) and all(a.get("engine") != "core2" for a in _answers(first))
    assert new_core.asked == ["stock on hand by warehouse", WHY]
    assert [a["answer"]["headline"] for a in _answers(why)] == ["The new core explains it."]
    assert not _analyses(why)                                     # today's analysis did not run first


def test_in_core2_mode_what_the_new_core_cannot_explain_is_todays_analysis(tenant):
    new_core = _NewCore({})
    stack, conversation = _conversation(tenant, "core2", new_core)
    with stack:
        conversation.ask("stock on hand by warehouse")
        why = conversation.ask(WHY)
    assert new_core.asked == ["stock on hand by warehouse", WHY]
    # Today's analysis of the answer on screen: no new answer of its own first (today's pipeline would ask
    # the warehouse again and analyse that).
    assert _analyses(why) == ["Why this pattern?"] and not _answers(why)


def test_side_by_side_a_why_is_todays_analysis_then_the_new_cores_preview(tenant):
    new_core = _NewCore({WHY: EXPLAINED})
    stack, conversation = _conversation(tenant, "compare", new_core)
    with stack:
        conversation.ask("stock on hand by warehouse")
        why = conversation.ask(WHY)
    kinds = [("analysis" if f.get("type") == "assistant_analysis" else f.get("engine"))
             for f in why["frames"] if f.get("type") in ("assistant_analysis", "assistant_response")]
    assert kinds == ["analysis", "core2"], kinds
    (preview,) = _answers(why)
    assert preview["answer"]["headline"] == "The new core explains it."
    assert preview["result_scope"]["badge"] == "New core (preview)"
    assert why["frames"][-1] == {"type": "typing", "active": False}       # the composer is free after both


def test_in_core2_mode_a_follow_up_never_acts_on_todays_older_result(tenant):
    """After the new core's answer, today's last result is the answer before it: a "why" is not about that."""
    new_core = _NewCore({"stock value by warehouse": json.loads(json.dumps(CANNED))})
    stack, conversation = _conversation(tenant, "core2", new_core)
    with stack:
        conversation.ask("stock on hand by warehouse")     # today's answer, its rows the current result
        conversation.ask("stock value by warehouse")       # the new core's answer, now on screen
        why = conversation.ask("why is that?")             # the new core cannot: an ordinary question now
    assert new_core.asked[-1] == "why is that?"
    assert not _analyses(why)                              # not an analysis of the stock on hand answer


def _answer_with_rows(n: int):
    """A new-core answer of ``n`` rows, using the question id the bridge hands it (as portal_answer does)."""
    def answer(*args, **kwargs):
        rows = [{"v": i} for i in range(n)]
        payload = json.loads(json.dumps(CANNED))
        payload["data"]["rows"], payload["data"]["total_rows"] = rows[:200], n
        payload["trust"]["question_id"], payload["trust"]["row_count"] = kwargs["question_id"], n
        payload["export_rows"] = rows
        return payload
    return answer


def test_a_core2_answer_is_kept_as_todays_are_history_full_export_usage_and_thumbs(tenant):
    import store

    before = store.get_monthly_query_count(harness.ACCOUNT)
    turn = _ask(tenant, "core2", _answer_with_rows(250))
    (frame,) = _answers(turn)
    assert "export_rows" not in frame                      # every row stays on the server
    question_id = frame["trust"]["question_id"]
    assert question_id.startswith("c2-")                  # the reader's thumbs reach this answer
    trace = store.get_answer_trace(int(frame["trace_id"]))
    assert trace["question_id"] == question_id and trace["route"] == "core2" and trace["status"] == "success"
    assert len(json.loads(trace["result_rows"])) == 250   # the CSV export reads all of them, not the 200 shown
    assert trace["session_id"]                            # the thread's history finds it again
    assert store.get_monthly_query_count(harness.ACCOUNT) == before + 1
    assert store.list_core2_answers(harness.ACCOUNT, 1)[0]["question_id"] == question_id
    from store.learning_store import save_feedback

    save_feedback(question_id, int(trace["portal_user_id"]), harness.ACCOUNT, -1, reason_code="wrong_data")
    assert store.list_core2_answers(harness.ACCOUNT, 1)[0]["rating"] == -1


def test_a_workspace_at_its_monthly_limit_is_answered_by_todays_pipeline(tenant):
    import store

    used = store.get_monthly_query_count(harness.ACCOUNT)
    limit = (store.get_client(harness.ACCOUNT) or {}).get("query_limit_monthly")
    store.update_client_meta(harness.ACCOUNT, query_limit_monthly=max(used, 1))
    if used == 0:
        store.log_query(harness.ACCOUNT, "an earlier question", "SELECT 1", success=True)
    try:
        turn = _ask(tenant, "core2", _answer_with_rows(3))
    finally:
        store.update_client_meta(harness.ACCOUNT, query_limit_monthly=limit or 500)
    assert all(a.get("engine") != "core2" for a in _answers(turn))
    assert any("limit" in json.dumps(f).lower() for f in turn["frames"])   # today's pipeline says so


def test_a_side_by_side_preview_takes_thumbs_but_no_history_and_no_usage(tenant):
    import store

    before = store.get_monthly_query_count(harness.ACCOUNT)
    traces = len(store.list_answer_traces(harness.ACCOUNT, 500))
    turn = _ask(tenant, "compare", _answer_with_rows(3))
    today, preview = _answers(turn)
    assert preview["trust"]["question_id"].startswith("c2-") and "trace_id" not in preview
    assert "export_rows" not in preview
    core2_traces = [t for t in store.list_answer_traces(harness.ACCOUNT, 500) if t.get("route") == "core2"]
    assert len(store.list_answer_traces(harness.ACCOUNT, 500)) - traces <= 1     # today's answer's own trace
    assert all(t["question_id"] != preview["trust"]["question_id"] for t in core2_traces)
    assert store.get_monthly_query_count(harness.ACCOUNT) - before <= 1          # today's answer counted, not the preview


# ── governance and the production wiring ──────────────────────────────────


@pytest.fixture
def learned(workspace, monkeypatch):  # noqa: F811
    store, built, schema_dir = workspace
    _schema_file(built, schema_dir)
    from admin import core2_routes

    core2_routes._run_build(ACCOUNT)
    import core.compliance.governed_query as gq

    def run_query(credentials, db_type, sql, max_rows=200):
        cursor = built.con.cursor()
        cursor.execute(sqlglot.transpile(sql, read="tsql", write="duckdb")[0])
        names = [d[0] for d in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchmany(max_rows)]

    monkeypatch.setattr(gq, "run_query", run_query)
    return store, built, schema_dir


class _CannotStandIn(Exception):
    """T-SQL the DuckDB stand-in cannot run (string + string, @@DATEFIRST); governance had already passed it."""


@pytest.mark.parametrize("name", ["retail", "inventory"])
def test_the_new_cores_sql_passes_querybots_governance_and_returns_the_reference(name, tmp_path, monkeypatch):
    import duckdb

    import store
    import store.crypto
    from core.compliance import governed_query as gq
    from core.schema import load_known_tables
    from core2.bootstrap.service import build_workspace, load_model
    from core2.compile.compiler import compile_query
    from core2.resolve.resolver import Context, resolve
    from core2.warehouse import querybot
    from core2.warehouse.governed import GovernedWarehouse
    from core2.warehouse.runner import DuckDBWarehouse
    from evals.core2 import domains
    from evals.core2.compile_eval import _Words, golden, same_rows
    from evals.core2.framework import materialize

    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    monkeypatch.setattr(store.crypto, "KEY_FILE", tmp_path / ".key")
    store.init_db()
    account = f"acct-core2-governed-{name}"
    store.upsert_client(account, "web")
    store.save_compliance_profile(account, mode="standard")
    db_id = store.save_db_config("azure_sql", name, {"server": "s", "database": "d", "user": "u", "password": "p"})
    store.update_client_meta(account, db_config_id=db_id)
    built = materialize(domains.build(name), "descriptive")
    schema_dir = tmp_path / "clients" / account / "schema"
    store.update_client_state(account, "READY", {"schema_dir": str(schema_dir)})
    _schema_file(built, schema_dir)
    monkeypatch.setattr(querybot, "QueryBotWarehouse", lambda db_type, credentials, **kw: _Connection(built.con))
    build_workspace(account)

    def run_query(credentials, db_type, sql, max_rows=200):
        try:
            cursor = built.con.cursor()
            cursor.execute(sqlglot.transpile(sql, read="tsql", write="duckdb")[0])
        except duckdb.Error as exc:
            raise _CannotStandIn(str(exc)) from None
        names = [d[0] for d in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchmany(max_rows)]

    monkeypatch.setattr(gq, "run_query", run_query)
    model = load_model(account, db_id)
    governed = GovernedWarehouse(account, {"id": 1, "role": "admin"}, store.get_db_config(db_id),
                                 known_tables=load_known_tables(str(schema_dir)))
    words, reference = _Words(model, built), DuckDBWarehouse(built.con)
    compared = 0
    for case in golden(name)["questions"]:
        if not case.get("reference_sql"):
            continue
        logical = resolve(words.plan(case["plan"]), model, Context(today=dt.date(2026, 6, 15)))
        compiled = compile_query(logical, model, "tsql")
        try:
            got = governed.query(compiled.sql, max_rows=compiled.row_cap)   # governance refusals raise here
        except _CannotStandIn:
            continue
        expected = reference.query(case["reference_sql"])
        diff = same_rows(expected.columns, expected.rows, got.columns, got.rows,
                         order_matters=bool((case.get("expect") or {}).get("order_matters")))
        assert not diff, (case["id"], diff, compiled.sql)
        compared += 1
    assert compared >= 15, compared


def test_the_portal_wiring_finds_the_model_the_readers_tables_and_the_ai(learned, monkeypatch):
    store, built, schema_dir = learned
    import core2.bootstrap.ai as ai
    from core2 import service

    plans = iter([
        json.dumps({"kind": "query", "intent": "value", "measures": ["net_amount"],
                    "time": {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}}),
        json.dumps({"kind": "query", "intent": "breakdown", "measures": ["net_amount"], "group_by": ["customer"]}),
    ])
    monkeypatch.setattr(ai, "workspace_planner", lambda *a, **k: (lambda stable, tail: next(plans)))
    monkeypatch.setattr(service, "_INDEXES", type(service._INDEXES)())
    from core2.warehouse import querybot

    monkeypatch.setattr(querybot, "QueryBotWarehouse", lambda db_type, credentials, **kw: _Connection(built.con))
    admin = {"id": 1, "role": "admin"}
    payload = service.portal_answer(ACCOUNT, "net sales in 2025", admin, session_key="s1")
    assert round(payload["kpi"]["value"], 2) == 3093601.41 and payload["engine"] == "core2"
    reader = {"id": 2, "role": "viewer"}
    allowed = {"MEMORY.MAIN.order_lines", "MEMORY.MAIN.calendar"}
    with patch.object(store, "get_allowed_tables", return_value=allowed):
        payload = service.portal_answer(ACCOUNT, "net sales by customer", reader, session_key="s2")
    assert "do not have access" in payload["answer"]["headline"]


def test_the_bridge_records_every_answer_for_comparison(learned, monkeypatch):
    store, *_ = learned
    import gateway.core2_bridge as bridge

    monkeypatch.setattr("core2.service.portal_answer", lambda *a, **k: json.loads(json.dumps(CANNED)))
    sent: list[dict] = []

    class Adapter:
        thread_id = "t1"
        send_lock = asyncio.Lock()

    class Socket:
        async def send_json(self, payload):
            sent.append(payload)

    asyncio.run(bridge.answer_beside(Adapter(), Socket(), ACCOUNT, "what is 42?", {"id": 1}))
    row = store.list_core2_answers(ACCOUNT, 1)[0]
    assert row["status"] == "answered" and row["headline"] == "New core says 42." and row["mode"] == "compare"
    assert sent and sent[0]["answer"]["scope_badge"] == "New core (preview)"


def test_the_new_core_runs_on_its_own_threads_not_on_todays_pipelines(learned, monkeypatch):
    """Today's pipeline runs its warehouse queries on the default thread pool; a slow
    new-core question must tie up the new core's own threads, never those."""
    import threading

    import gateway.core2_bridge as bridge

    ran_on: list[str] = []

    def answer(*args, **kwargs):
        ran_on.append(threading.current_thread().name)
        return json.loads(json.dumps(CANNED))

    monkeypatch.setattr("core2.service.portal_answer", answer)

    class Adapter:
        thread_id = "t1"
        send_lock = asyncio.Lock()

    class Socket:
        async def send_json(self, payload):
            return None

    asyncio.run(bridge.answer_beside(Adapter(), Socket(), ACCOUNT, "what is 42?", {"id": 1}))
    assert asyncio.run(bridge.answer_instead("core2", Adapter(), Socket(), ACCOUNT, "what is 42?", {"id": 1}))
    assert len(ran_on) == 2 and all(name.startswith("core2") for name in ran_on), ran_on


@pytest.mark.parametrize("posture", ["standard", "regulated", "unprovisioned"])
def test_a_question_is_recorded_as_querybot_keeps_questions(posture, learned, monkeypatch):
    """Under compliance (regulated, or never set up, as the agent runtime counts it) the
    recorded question has its personal data scrubbed. The same scrubber's presence is what
    keeps member values from the AI (portal_answer: values_allowed=scrub is None)."""
    store, *_ = learned
    import gateway.core2_bridge as bridge
    from core2.service import question_scrubber

    if posture == "regulated":
        store.save_compliance_profile(ACCOUNT, mode="regulated")
    elif posture == "unprovisioned":
        # The store object is_regulated holds: other test modules re-import `store`, so in
        # a full run a bare `import store` can be another object (see test_user_attestation).
        from core.compliance import policy_engine

        monkeypatch.setattr(policy_engine.store, "compliance_profile_exists", lambda account_id: False)
    monkeypatch.setattr("core2.service.portal_answer", lambda *a, **k: json.loads(json.dumps(CANNED)))

    class Adapter:
        thread_id = "t1"
        send_lock = asyncio.Lock()

    class Socket:
        async def send_json(self, payload):
            return None

    asyncio.run(bridge.answer_beside(Adapter(), Socket(), ACCOUNT, "orders for jane.doe@example.com", {"id": 1}))
    recorded = store.list_core2_answers(ACCOUNT, 1)[0]["question"]
    if posture == "standard":
        assert recorded == "orders for jane.doe@example.com" and question_scrubber(ACCOUNT) is None
    else:
        assert "jane.doe@example.com" not in recorded and "[EMAIL]" in recorded, recorded
        assert question_scrubber(ACCOUNT) is not None


def test_side_by_side_does_not_answer_thanks_twice(learned, monkeypatch):
    import gateway.core2_bridge as bridge

    monkeypatch.setattr("core2.service.portal_answer", lambda *a, **k: {**CANNED, "kind": "smalltalk"})
    sent: list[dict] = []

    class Adapter:
        thread_id = "t1"
        send_lock = asyncio.Lock()

    class Socket:
        async def send_json(self, payload):
            sent.append(payload)

    asyncio.run(bridge.answer_beside(Adapter(), Socket(), ACCOUNT, "thanks!", {"id": 1}))
    assert not sent


def test_a_frame_that_cannot_be_sent_never_turns_todays_answer_into_an_error(learned, monkeypatch):
    import gateway.core2_bridge as bridge

    monkeypatch.setattr("core2.service.portal_answer", lambda *a, **k: json.loads(json.dumps(CANNED)))

    class Adapter:
        thread_id = "t1"
        send_lock = asyncio.Lock()

    class ClosedSocket:
        async def send_json(self, payload):
            raise RuntimeError("the socket is closed")

    asyncio.run(bridge.answer_beside(Adapter(), ClosedSocket(), ACCOUNT, "what is 42?", {"id": 1}))
    assert asyncio.run(bridge.answer_instead("core2", Adapter(), ClosedSocket(), ACCOUNT, "what is 42?", {"id": 1})) is False


def test_a_row_policy_narrows_the_new_cores_answer_to_the_readers_rows(tmp_path, monkeypatch):
    """Row policies are the governed executor's: it rewrites the SQL, so core2's must stay rewritable.

    The rewriter names each table by its alias, unquoted. Core2 once aliased the
    orders table "order", a reserved word it then had to quote, so a policy on
    orders made ``WHERE order.region = 'North'``: a syntax error instead of the
    reader's numbers, in every dialect.
    """
    import types

    import duckdb
    import numpy as np
    import pandas as pd

    import store
    import store.crypto
    from core.compliance import governed_query as gq
    from core.compliance import sql_guard
    from core.compliance.models import PolicyContext
    from core.schema import load_known_tables
    from core2.bootstrap.service import build_workspace, load_model
    from core2.compile.compiler import compile_query
    from core2.plan.ir import Plan
    from core2.resolve.resolver import Context, resolve
    from core2.warehouse import querybot
    from core2.warehouse.governed import GovernedWarehouse

    rng = np.random.default_rng(1)
    con = duckdb.connect()
    n = 600
    frames = {
        "customers": pd.DataFrame({"customer_id": np.arange(1, 21), "customer_name": [f"C{i:02d}" for i in range(1, 21)]}),
        "orders": pd.DataFrame({"order_id": np.arange(1, n + 1), "customer_id": rng.integers(1, 21, n),
                                "region": np.where(rng.random(n) < 0.5, "North", "South"),
                                "order_amount": np.round(rng.uniform(10, 900, n), 2)}),
    }
    for name, frame in frames.items():
        con.register("_f", frame)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _f")
        con.unregister("_f")
    built = types.SimpleNamespace(con=con, tables={t: t for t in frames}, declared_pks={}, declared_fks=[])

    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    monkeypatch.setattr(store.crypto, "KEY_FILE", tmp_path / ".key")
    store.init_db()
    account = "acct-core2-row-policy"
    store.upsert_client(account, "web")
    store.save_compliance_profile(account, mode="standard")
    db_id = store.save_db_config("azure_sql", "orders", {"server": "s", "database": "d", "user": "u", "password": "p"})
    store.update_client_meta(account, db_config_id=db_id)
    schema_dir = tmp_path / "clients" / account / "schema"
    store.update_client_state(account, "READY", {"schema_dir": str(schema_dir)})
    _schema_file(built, schema_dir)
    monkeypatch.setattr(querybot, "QueryBotWarehouse", lambda db_type, credentials, **kw: _Connection(con))
    build_workspace(account)
    reader = {"id": 7, "role": "analyst"}
    store.replace_row_policies(account, 0, [{
        "name": "north only", "subject_type": "user", "subject_id": "7", "table_fqn": "ORDERS",
        "condition": {"field": "region", "operator": "=", "value": "North"}}])

    def run_query(credentials, db_type, sql, max_rows=200):
        cursor = con.cursor()
        cursor.execute(sqlglot.transpile(sql, read="tsql", write="duckdb")[0])
        names = [d[0] for d in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchmany(max_rows)]

    monkeypatch.setattr(gq, "run_query", run_query)
    model = load_model(account, db_id)
    measure = next(m.slug for m in model.measures.values()
                   if (getattr(m.expr, "column", None) or "").endswith("order_amount"))
    customer = next(e.slug for e in model.entities.values() if e.table.endswith("customers"))
    logical = resolve(Plan.model_validate({"kind": "query", "intent": "breakdown", "measures": [measure],
                                           "group_by": [customer]}), model, Context(today=dt.date(2026, 6, 15)))
    compiled = compile_query(logical, model, "tsql")
    governed = GovernedWarehouse(account, reader, store.get_db_config(db_id),
                                 known_tables=load_known_tables(str(schema_dir)))
    got = governed.query(compiled.sql, max_rows=compiled.row_cap)
    expected = con.execute("SELECT c.customer_name, SUM(o.order_amount) FROM orders o LEFT JOIN customers c "
                           "ON c.customer_id = o.customer_id WHERE o.region = 'North' GROUP BY 1").fetchall()
    everyone = con.execute("SELECT SUM(order_amount) FROM orders").fetchone()[0]
    assert sorted((r[0], round(r[1], 2)) for r in got.rows) == sorted((r[0], round(r[1], 2)) for r in expected)
    assert round(sum(r[1] for r in got.rows), 2) < round(everyone, 2)      # the policy did narrow it

    # The same rewrite in every warehouse the new core writes for.
    context = PolicyContext(account_id=account, user_id="7")
    for dialect, db_type in (("tsql", "azure_sql"), ("snowflake", "snowflake"), ("oracle", "oracle"),
                             ("duckdb", "duckdb")):
        rewritten, applied = sql_guard.inject_row_policies(compile_query(logical, model, dialect).sql, db_type,
                                                           context)
        assert applied, dialect
        sqlglot.parse_one(rewritten, read=sql_guard._DIALECT.get(db_type, "snowflake"))   # still valid SQL
        if dialect == "duckdb":
            assert sorted((r[0], round(r[1], 2)) for r in con.execute(rewritten).fetchall()) == \
                sorted((r[0], round(r[1], 2)) for r in expected)


def test_a_masked_column_stays_masked_in_the_new_cores_answer_on_every_warehouse():
    """Masking follows each output back to its source column (analyze_sql's lineage), then
    finds that output in the rows. Core2 names its outputs in lower case, unquoted, and
    Snowflake and Oracle return those names upper-cased: the masked customer name came
    back in clear there, and only there."""
    import duckdb
    import numpy as np
    import pandas as pd

    from core.compliance.models import PolicyDecision
    from core.compliance.result_guard import protect_rows
    from core.compliance.sql_guard import analyze_sql
    from core2.bootstrap.build import BuildOptions, build_model
    from core2.bootstrap.inventory import from_duckdb
    from core2.compile.compiler import compile_query
    from core2.plan.ir import Plan
    from core2.resolve.resolver import Context, resolve
    from core2.warehouse.runner import DuckDBWarehouse

    rng = np.random.default_rng(1)
    con = duckdb.connect()
    frames = {
        "customers": pd.DataFrame({"customer_id": np.arange(1, 6), "customer_name": [f"Person {i}" for i in range(1, 6)]}),
        "orders": pd.DataFrame({"order_id": np.arange(1, 51), "customer_id": rng.integers(1, 6, 50),
                                "order_amount": rng.uniform(1, 9, 50)}),
    }
    for name, frame in frames.items():
        con.register("_f", frame)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _f")
        con.unregister("_f")
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="t", options=BuildOptions(workers=1))
    measure = next(m.slug for m in model.measures.values()
                   if (getattr(m.expr, "column", None) or "").endswith("order_amount"))
    customer = next(e.slug for e in model.entities.values() if e.table.endswith("customers"))
    logical = resolve(Plan.model_validate({"kind": "query", "intent": "breakdown", "measures": [measure],
                                           "group_by": [customer]}), model, Context(today=dt.date(2026, 6, 15)))
    for dialect, db_type, returned_as in (("snowflake", "snowflake", str.upper), ("oracle", "oracle", str.upper),
                                          ("tsql", "azure_sql", str)):
        compiled = compile_query(logical, model, dialect)
        analysis = analyze_sql(compiled.sql, db_type)
        source = next(s for v in analysis.lineage.values() for s in v if s.endswith(".CUSTOMER_NAME"))
        decision = PolicyDecision(allowed=True, reason_code="allow", masking={source: "redact"})
        names = [returned_as(c.name) for c in compiled.columns]
        values = ["Person 1" if c.role != "measure" else 12.5 for c in compiled.columns]
        protected = protect_rows([dict(zip(names, values))], decision, analysis.lineage, account_id="t",
                                 mask_exempt_outputs=analysis.mask_exempt_outputs)
        assert "Person 1" not in str(protected), (dialect, protected)
        assert 12.5 in protected[0].values(), (dialect, protected)     # the total is not masked


def test_a_metric_the_reader_defines_reaches_the_new_core_not_the_result_on_screen(tenant):
    # "... as a percentage. Show it by month" reads like a change to the answer on screen (a format, a regrouping):
    # a definition is a question of its own.
    asked: list[str] = []

    def answer(account_id, question, portal_user, **kwargs):
        asked.append(question)
        return {**json.loads(json.dumps(CANNED)), "own_metric": {"name": "Margin after returns", "token": "t"}}

    question = ("Margin after returns = net amount minus refunds, divided by net amount, as a percentage. "
                "Show it by month in the first half of 2026")
    turn = _ask(tenant, "core2", answer, question)
    assert asked == [question]
    assert [a["engine"] for a in _answers(turn)] == ["core2"]
    assert not [f for f in turn["frames"] if f.get("type") == "assistant_error"]
