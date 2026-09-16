# -*- coding: utf-8 -*-
"""The one tool an investigation calls: an ordinary governed question.

core.investigation.run_query_tool is a thin capture around
core.dispatcher.dispatch -- the exact path an ordinary chat message takes --
built for a multi-step loop that needs to ask several questions in sequence
and read back what each one found, without a browser on the other end.

Three properties matter enough to test directly:

  * success is read back correctly (a new, non-empty result landed, its
    rows and SQL become a statistical brief);
  * a FAILURE never reads as the previous call's leftover success, because
    every question in one investigation shares one scoped result-cache
    session and a second call's own outcome must be judged on its own;
  * a sub-question's own answer-trace completion must never finish the
    investigation's own agent run out from under it.

dispatch is replaced at the boundary -- the real governance, semantics,
validation and execution this tool relies on are tested exhaustively
everywhere dispatch() itself is tested; this module tests the capture, not
the pipeline.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from unittest.mock import AsyncMock, patch

import pytest

_tmp_db = os.path.join(tempfile.mkdtemp(), "test_investigation_tool.db")
os.environ["QUERYBOT_DB_PATH"] = _tmp_db
for _mod in list(sys.modules):
    if _mod.startswith("store"):
        del sys.modules[_mod]

import store  # noqa: E402

store.init_db()

import core.dispatcher as dispatcher  # noqa: E402
from core.investigation import (  # noqa: E402
    result_llm_features_allowed_for_investigation,
    run_query_tool,
)

ROWS = [
    {"WORK_CENTRE": "WC10", "SCRAP_QTY": 40.0},
    {"WORK_CENTRE": "WC20", "SCRAP_QTY": 15.0},
]
SQL = "SELECT WORK_CENTRE, SUM(SCRAP_QTY) AS SCRAP_QTY FROM DBO.F_PRODUCTION GROUP BY WORK_CENTRE"


def _account():
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    return account_id


def _portal_user():
    return {"id": 7, "name": "Ada", "role": "viewer", "group_name": ""}


def _run(coro):
    return asyncio.run(coro)


class TestASuccessfulAnswerIsReadBack:

    def test_rows_and_sql_become_a_brief(self):
        account_id = _account()

        async def _fake_dispatch(_account_id, _event, adapter, _bg, portal_user=None):
            adapter.cache_result(list(ROWS), "scrap by work centre", SQL, {})

        with patch.object(dispatcher, "dispatch", _fake_dispatch):
            result = _run(run_query_tool(
                account_id=account_id, portal_user=_portal_user(), run_id="run-1",
                question="scrap by work centre",
            ))
        assert result.ok is True
        assert result.row_count == 2
        assert result.result_id
        assert "WC10" in result.brief and "40" in result.brief

    def test_the_question_actually_asked_is_carried(self):
        account_id = _account()

        async def _fake_dispatch(_account_id, _event, adapter, _bg, portal_user=None):
            adapter.cache_result(list(ROWS), "scrap by work centre", SQL, {})

        with patch.object(dispatcher, "dispatch", _fake_dispatch):
            result = _run(run_query_tool(
                account_id=account_id, portal_user=_portal_user(), run_id="run-1",
                question="scrap by work centre for Q2 2024",
            ))
        assert result.question == "scrap by work centre for Q2 2024"


class TestAFailureIsNeverAStaleSuccess:

    def test_a_reader_facing_message_becomes_the_error(self):
        account_id = _account()

        async def _fake_dispatch(_account_id, event, adapter, _bg, portal_user=None):
            await adapter.send_message(event, "No records matched the filters.")

        with patch.object(dispatcher, "dispatch", _fake_dispatch):
            result = _run(run_query_tool(
                account_id=account_id, portal_user=_portal_user(), run_id="run-2",
                question="scrap by nonexistent plant",
            ))
        assert result.ok is False
        assert result.error == "No records matched the filters."
        assert result.row_count == 0

    def test_nothing_sent_is_a_generic_failure_not_a_crash(self):
        account_id = _account()

        async def _fake_dispatch(*a, **k):
            return None

        with patch.object(dispatcher, "dispatch", _fake_dispatch):
            result = _run(run_query_tool(
                account_id=account_id, portal_user=_portal_user(), run_id="run-3",
                question="an odd question",
            ))
        assert result.ok is False
        assert result.error

    def test_a_second_call_does_not_inherit_the_firsts_success(self):
        """One investigation's steps share one scoped session. A step that
        fails must be judged on what IT produced, not on the result its own
        earlier sibling left sitting in that same session."""
        account_id = _account()
        calls = {"n": 0}

        async def _fake_dispatch(_account_id, event, adapter, _bg, portal_user=None):
            calls["n"] += 1
            if calls["n"] == 1:
                adapter.cache_result(list(ROWS), "scrap by work centre", SQL, {})
            else:
                await adapter.send_message(event, "That could not be answered either.")

        with patch.object(dispatcher, "dispatch", _fake_dispatch):
            first = _run(run_query_tool(
                account_id=account_id, portal_user=_portal_user(), run_id="run-4",
                question="scrap by work centre",
            ))
            second = _run(run_query_tool(
                account_id=account_id, portal_user=_portal_user(), run_id="run-4",
                question="scrap by shift",
            ))
        assert first.ok is True and first.row_count == 2
        assert second.ok is False, second
        assert second.error == "That could not be answered either."

    def test_two_different_runs_never_share_a_session(self):
        account_id = _account()

        async def _fake_dispatch(_account_id, event, adapter, _bg, portal_user=None):
            await adapter.send_message(event, "nothing found")

        with patch.object(dispatcher, "dispatch", _fake_dispatch):
            result = _run(run_query_tool(
                account_id=account_id, portal_user=_portal_user(), run_id="run-5-b",
                question="scrap by shift",
            ))
        assert result.ok is False

    def test_an_exception_inside_dispatch_is_a_failed_step(self):
        account_id = _account()

        async def _boom(*a, **k):
            raise RuntimeError("provider down")

        with patch.object(dispatcher, "dispatch", _boom):
            result = _run(run_query_tool(
                account_id=account_id, portal_user=_portal_user(), run_id="run-6",
                question="scrap by work centre",
            ))
        assert result.ok is False
        assert result.error

    def test_an_empty_question_never_reaches_dispatch(self):
        account_id = _account()
        fake = AsyncMock()
        with patch.object(dispatcher, "dispatch", fake):
            result = _run(run_query_tool(
                account_id=account_id, portal_user=_portal_user(), run_id="run-7",
                question="   ",
            ))
        assert result.ok is False
        fake.assert_not_called()


class TestTheInvestigationsOwnRunSurvivesEachSubQuestion:

    def test_the_active_run_is_suppressed_during_the_call_and_restored_after(self):
        from core.agent_runtime import activate_agent_run, get_active_agent_run

        account_id = _account()
        seen_during_call = ["not set"]

        async def _fake_dispatch(_account_id, _event, _adapter, _bg, portal_user=None):
            seen_during_call[0] = get_active_agent_run()

        sentinel = object()
        with patch.object(dispatcher, "dispatch", _fake_dispatch):
            with activate_agent_run(sentinel):
                _run(run_query_tool(
                    account_id=account_id, portal_user=_portal_user(), run_id="run-8",
                    question="scrap by work centre",
                ))
                assert get_active_agent_run() is sentinel, (
                    "the investigation's own run must be restored once the "
                    "sub-question's dispatch call returns")
        assert seen_during_call[0] is None, (
            "a sub-question's own trace completion must not see an active "
            "agent run, or it would finish the whole investigation early")


class TestTheRegulatedGate:

    def test_it_delegates_to_the_real_policy_engine(self, monkeypatch):
        import core.compliance.policy_engine as pe

        monkeypatch.setattr(pe, "result_llm_features_allowed", lambda account_id: account_id == "allowed")
        assert result_llm_features_allowed_for_investigation("allowed") is True
        assert result_llm_features_allowed_for_investigation("blocked") is False

    def test_a_workspace_without_a_compliance_profile_fails_closed(self):
        assert result_llm_features_allowed_for_investigation("acct-no-profile-investigation") is False


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
