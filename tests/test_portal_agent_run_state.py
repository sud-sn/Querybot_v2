"""Regression checks for the portal's governed agent run-state reducer."""

from pathlib import Path


PORTAL = (
    Path(__file__).resolve().parents[1]
    / "portal"
    / "templates"
    / "portal_chat.html"
)


def _source() -> str:
    return PORTAL.read_text("utf-8")


def test_clarification_state_is_authoritative_over_late_typing_frame():
    source = _source()
    assert "agentRunState === 'waiting_for_user'" in source
    assert "setAgentRunState('waiting_for_user', msg)" in source
    assert "stage === 'waiting_for_user'" in source


def test_clarification_submission_never_silently_returns_while_waiting():
    source = _source()
    assert "processingActive && agentRunState !== 'waiting_for_user'" in source
    assert "QueryBot is still finishing the current step" in source
    assert "setAgentRunState('running');" in source


def test_late_events_from_an_old_run_are_ignored():
    source = _source()
    assert "activeAgentRunId" in source
    assert "eventRunId !== activeAgentRunId" in source
    assert "return false;" in source
