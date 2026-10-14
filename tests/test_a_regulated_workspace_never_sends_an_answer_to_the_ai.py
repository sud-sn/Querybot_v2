"""A regulated workspace's answers never reach the AI: no written summary, and the analyst note in its own words.

The product's rule for a regulated workspace (core/compliance/policy_engine.py::result_llm_features_allowed) is that
the AI writes the plan and never sees an answer. The old engine kept it; the new core's summary and analyst note did
not ask it: an answer of numbers alone (a monthly series, a total) was sent to the AI with its headline, which spells
out the answer's filters ("for customer type Wholesale"), and the question as typed. The trust panel's
"result-carrying calls (must be 0)" did not count those calls either. Now:

* a regulated workspace gets no AI summary and an analyst note in the facts' own sentences; the refusal is recorded;
* a standard workspace is summarised as before;
* the trust panel counts the new core's summary calls as result-carrying.

A throwaway database; the AI is replaced and fails the test if it is asked.
"""

from __future__ import annotations

import os

import pytest

PAYLOAD = {
    "engine": "core2", "question": "net sales by month in 2025",
    "answer": {"headline": "Net sales were $1.2M in 2025, for customer type Wholesale"},
    "data": {"headers": ["month", "net_sales"], "header_labels": {"month": "Month", "net_sales": "Net sales"},
             "column_formats": {"net_sales": "currency"}, "display_formats": {"month": {"type": "date"}},
             "rows": [{"month": "2025-01-01", "net_sales": 100000.0}, {"month": "2025-02-01", "net_sales": 120000.0}],
             "total_rows": 2},
    "reading": {"facts": [{"section": "what", "en": "February was 20% above January.", "size": 1}]},
}


@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    import store
    from admin import routes

    for module in {id(m): m for m in (store, routes.store)}.values():
        for namespace in (module.crypto.__dict__, module.config_store.decrypt.__globals__):
            monkeypatch.setitem(namespace, "KEY_FILE", tmp_path / ".key")
    store.init_db()
    yield store


def _workspace(store, mode: str) -> str:
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    store.save_compliance_profile(account_id, mode=mode, industry="standard")
    return account_id


@pytest.fixture
def ai(monkeypatch):
    from core2.bootstrap import ai as bootstrap_ai

    asked: list[str] = []

    def writer(account_id, client, *, question="", question_id=""):
        def complete(system: str, user: str) -> str:
            asked.append(user)
            return "Net sales rose from $100,000 in January to $120,000 in February."
        return complete

    monkeypatch.setattr(bootstrap_ai, "summary_writer", writer)
    return asked


def test_a_regulated_workspace_gets_no_ai_summary_and_the_refusal_is_recorded(fresh_store, ai):
    from core2.service import portal_summary

    account_id = _workspace(fresh_store, "regulated")
    assert portal_summary(account_id, "net sales by month in 2025", PAYLOAD) is None
    assert ai == [], "the answer was sent to the AI"
    with fresh_store.get_db() as conn:
        rows = [dict(r) for r in conn.execute("SELECT component, status FROM llm_call_log WHERE account_id=?",
                                              (account_id,))]
    assert {"component": "core2_summary", "status": "blocked"} in rows


def test_a_regulated_workspaces_note_is_the_facts_own_sentences(fresh_store, ai):
    from core2.service import portal_note

    account_id = _workspace(fresh_store, "regulated")
    note = portal_note(account_id, "net sales by month in 2025", PAYLOAD)
    assert note and note["by"] == "facts" and ai == []


def test_a_standard_workspace_is_summarised_as_before(fresh_store, ai):
    from core2.service import portal_summary

    account_id = _workspace(fresh_store, "standard")
    assert portal_summary(account_id, "net sales by month in 2025", PAYLOAD)
    assert ai and "THE ANSWER AS SHOWN" in ai[0]


def test_the_trust_panel_counts_the_new_cores_summaries_as_result_carrying():
    import store

    assert "core2_summary" in store.RESULT_LLM_COMPONENTS


def test_a_refusal_is_recorded_only_for_an_answer_the_ai_would_have_been_given(fresh_store, ai):
    from core2.service import portal_note, portal_summary

    account_id = _workspace(fresh_store, "regulated")
    nothing = {**PAYLOAD, "data": {**PAYLOAD["data"], "rows": []}}
    assert portal_summary(account_id, "net sales by month in 2025", nothing) is None
    named = {**PAYLOAD, "reading": {"facts": [{"section": "what", "en": "Wholesale led.", "size": 1,
                                               "members": ["Wholesale"]}]}}
    portal_note(account_id, "net sales by month in 2025", {**named, "personal": True})
    with fresh_store.get_db() as conn:
        blocked = conn.execute("SELECT COUNT(*) AS n FROM llm_call_log WHERE account_id=? AND status='blocked'",
                               (account_id,)).fetchone()["n"]
    assert blocked == 0 and ai == []
