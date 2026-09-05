import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pytest

import store
import store.agent_store as agent_store
from core.agent_runtime import AgentRunSession, activate_agent_run, get_active_agent_run
from store.db import _ensure_agent_runtime_tables, _ensure_answer_trace_tables


@pytest.fixture()
def agent_db(monkeypatch):
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(
        """
        CREATE TABLE client (account_id TEXT PRIMARY KEY);
        CREATE TABLE portal_user (
            id INTEGER PRIMARY KEY, account_id TEXT NOT NULL,
            name TEXT DEFAULT '', email TEXT DEFAULT '', is_active INTEGER DEFAULT 1
        );
        INSERT INTO client(account_id) VALUES ('tenant-a'), ('tenant-b');
        INSERT INTO portal_user(id, account_id) VALUES (1, 'tenant-a'), (2, 'tenant-b');
        """
    )
    _ensure_answer_trace_tables(conn)
    _ensure_agent_runtime_tables(conn)
    conn.execute(
        "INSERT INTO answer_trace(id, account_id, portal_user_id) VALUES (42, 'tenant-a', 1)"
    )

    @contextmanager
    def fake_get_db():
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    monkeypatch.setattr(agent_store, "get_db", fake_get_db)
    monkeypatch.setattr(store, "get_allowed_tables", lambda user: {"dbo.Sales", "dbo.Customer"})
    monkeypatch.setattr(
        store,
        "get_compliance_profile",
        lambda account_id: {
            "mode": "regulated", "industry": "healthcare", "active_policy_version": 7,
        },
    )
    monkeypatch.setattr(
        store,
        "get_semantic_compiler_state",
        lambda account_id: {"active_contract_version": "contract-12"},
    )
    yield conn
    conn.close()


def _start():
    return AgentRunSession.start(
        account_id="tenant-a",
        portal_user={"id": 1, "account_id": "tenant-a"},
        external_thread_id="thread/unsafe",
        objective="Show patient revenue by month",
        selected_schema="dbo",
    )


def test_run_is_durable_bounded_and_governed(agent_db):
    run = _start()
    stored = store.get_agent_run(
        account_id="tenant-a", portal_user_id=1, run_id=run.run_id,
    )
    steps = store.list_agent_steps(
        account_id="tenant-a", portal_user_id=1, run_id=run.run_id,
    )

    assert stored["status"] == "running"
    assert stored["tool_calls_used"] == 1
    assert stored["max_tool_calls"] == 5
    assert json.loads(stored["allowed_tables_snapshot"]) == ["dbo.Customer", "dbo.Sales"]
    assert stored["policy_version"] == 7
    assert stored["contract_version"] == "contract-12"
    assert steps[0]["tool_name"] == "query_data"
    metadata = json.loads(steps[0]["metadata_json"])
    assert metadata["read_only"] is True
    assert metadata["governed"] is True
    assert metadata["uses_existing_acl"] is True
    assert metadata["uses_existing_sql_validation"] is True


def test_run_cannot_be_read_or_mutated_across_tenant_boundary(agent_db):
    run = _start()
    assert store.get_agent_run(
        account_id="tenant-b", portal_user_id=2, run_id=run.run_id,
    ) is None
    with pytest.raises(PermissionError):
        store.update_agent_run(
            run_id=run.run_id,
            account_id="tenant-b",
            portal_user_id=2,
            status="completed",
        )


def test_tool_budget_is_enforced_before_an_extra_tool_call(agent_db):
    thread = store.ensure_agent_thread(
        account_id="tenant-a", portal_user_id=1, external_thread_id="budget",
    )
    run = store.create_agent_run(
        thread_id=thread["id"], account_id="tenant-a", portal_user_id=1,
        objective_sanitized="question", max_tool_calls=1,
    )
    store.create_agent_step(
        run_id=run["id"], account_id="tenant-a", portal_user_id=1,
        step_index=1, tool_name="query_data",
    )
    with pytest.raises(RuntimeError, match="budget"):
        store.create_agent_step(
            run_id=run["id"], account_id="tenant-a", portal_user_id=1,
            step_index=2, tool_name="query_data",
        )


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"status": "success", "answer_type": "table"}, "completed"),
        ({"status": "success", "answer_type": "clarification"}, "waiting_for_user"),
        ({"status": "error", "answer_type": "policy_denied"}, "blocked"),
        ({"status": "error", "answer_type": "error", "error_message": "No table access assigned"}, "blocked"),
        ({"status": "error", "answer_type": "timeout"}, "failed"),
    ],
)
def test_governed_pipeline_outcomes_are_preserved(agent_db, fields, expected):
    run = _start()
    run.record_trace_outcome(42, **fields)
    stored = store.get_agent_run(
        account_id="tenant-a", portal_user_id=1, run_id=run.run_id,
    )
    assert stored["status"] == expected
    assert stored["answer_trace_id"] == 42


def test_clarification_resumes_the_same_owned_run(agent_db):
    run = _start()
    run.record_trace_outcome(42, status="success", answer_type="clarification")
    resumed = AgentRunSession.resume(
        account_id="tenant-a", portal_user_id=1,
        external_thread_id="threadunsafe", reply="Use invoice date",
    )
    assert resumed is not None
    assert resumed.run_id == run.run_id
    stored = store.get_agent_run(
        account_id="tenant-a", portal_user_id=1, run_id=run.run_id,
    )
    assert stored["status"] == "running"
    assert stored["completed_at"] is None


def test_context_binding_is_request_local(agent_db):
    run = _start()
    assert get_active_agent_run() is None


def test_analysis_run_and_child_tasks_are_durable_and_tenant_scoped(agent_db):
    run = AgentRunSession.start(
        account_id="tenant-a",
        portal_user={"id": 1, "account_id": "tenant-a"},
        external_thread_id="analysis-thread",
        objective="Analyze this result deeply",
        max_tool_calls=1,
        initial_tool="analyze_result",
        initial_label="Analyzing the governed result",
        initial_metadata={"database_queried": False, "rows_sent_to_llm": 0},
    )
    subtask = store.create_agent_subtask(
        parent_run_id=run.run_id,
        account_id="tenant-a",
        portal_user_id=1,
        objective_sanitized="Correlation analysis",
        tool_name="analysis_correlation",
        metadata={"input_row_count": 20},
    )
    store.finish_agent_subtask(
        subtask_id=subtask["id"], parent_run_id=run.run_id,
        account_id="tenant-a", portal_user_id=1,
        status="completed", result_summary="Strong relationship found.",
    )

    steps = store.list_agent_steps(
        account_id="tenant-a", portal_user_id=1, run_id=run.run_id,
    )
    tasks = store.list_agent_subtasks(
        account_id="tenant-a", portal_user_id=1, run_id=run.run_id,
    )
    assert steps[0]["tool_name"] == "analyze_result"
    assert json.loads(steps[0]["metadata_json"])["database_queried"] is False
    assert tasks[0]["status"] == "completed"
    assert tasks[0]["result_summary"] == "Strong relationship found."
    admin_tasks = store.list_agent_subtasks_for_account("tenant-a")
    assert admin_tasks[0]["id"] == subtask["id"]
    assert admin_tasks[0]["metadata"]["input_row_count"] == 20
    assert "rows" not in admin_tasks[0]["metadata"]
    assert store.list_agent_subtasks(
        account_id="tenant-b", portal_user_id=2, run_id=run.run_id,
    ) == []
    with activate_agent_run(run):
        assert get_active_agent_run() is run
    assert get_active_agent_run() is None


def test_portal_and_pipeline_agent_wiring_is_present():
    root = Path(__file__).resolve().parents[1]
    webhooks = (root / "gateway" / "webhooks.py").read_text("utf-8")
    adapter = (root / "gateway" / "web_adapter.py").read_text("utf-8")
    trace = (root / "core" / "pipeline_trace.py").read_text("utf-8")
    portal = (root / "portal" / "templates" / "portal_chat.html").read_text("utf-8")
    assert "AgentRunSession.start" in webhooks
    assert "activate_agent_run" in webhooks
    assert "record_stage" in adapter
    assert "record_trace_outcome" in trace
    assert "agent_run_started" in portal
    # The run-state labels moved into the message catalogue, so this asserts
    # the string still REACHES the page rather than that it is spelled out in
    # the template -- which would now pass for a page that resolves the id to
    # nothing.
    from tests.chat_render import render as _render_chat
    assert "Blocked by policy" in _render_chat(lang="en")
