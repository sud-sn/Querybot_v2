"""A workspace under compliance keeps a record of every AI call, and the exact prompt, so it can be checked.

The AI egress log recorded a call only when the workspace's "LLM audit logging" switch was on, and it is off by
default; and it kept a fingerprint (SHA-256) of the prompt with a short sanitized preview, never the prompt, so
nobody could check the fingerprint against anything. For a workspace whose data is kept from the AI (regulated,
not yet provisioned, or any compliance mode but standard), that is no evidence at all. Now:

* such a workspace's calls are recorded whatever the switch says;
* the exact prompt and response are kept with the record, encrypted with the server's key;
* an admin opens a call on the AI egress log and reads what was sent, and the page says whether it matches the
  fingerprint recorded with it;
* a standard workspace is recorded as before: only with the switch on, and without the prompt.

Driven through the real AI call (core/llm.py) with the provider's client replaced; a throwaway database and key.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

SYSTEM = "You write one JSON plan."
USER = "QUESTION: net sales by region in 2025"


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
    from core import llm

    reply = SimpleNamespace(content=[SimpleNamespace(text='{"intent": "breakdown"}')], stop_reason="end_turn",
                            usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0,
                                                  cache_creation_input_tokens=0, cache_creation=None))

    async def create(**kwargs):
        return reply

    monkeypatch.setattr(llm, "_get_anthropic_client", lambda key: SimpleNamespace(messages=SimpleNamespace(create=create)))
    yield store


def _workspace(store, mode: str, *, audit: bool = False) -> str:
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    store.save_compliance_profile(account_id, mode=mode, industry="healthcare_pharmacy" if mode != "standard"
                                  else "standard")
    store.update_client_meta(account_id, enable_llm_audit=1 if audit else 0)
    return account_id


def _call(account_id: str, *, enabled: bool) -> None:
    from core.llm import llm_complete
    from core.llm_audit import llm_audit_scope

    async def run():
        with llm_audit_scope(account_id=account_id, question="net sales by region", enabled=enabled,
                             component="core2_planner"):
            return await llm_complete(SYSTEM, USER, "anthropic", "claude-sonnet-4-6", "key")

    asyncio.run(run())


def _calls(store, account_id: str) -> list[dict]:
    with store.get_db() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM llm_call_log WHERE account_id=?", (account_id,))]


def test_a_compliance_workspace_records_every_call_with_its_switch_off(fresh_store):
    account_id = _workspace(fresh_store, "regulated")
    _call(account_id, enabled=False)
    (row,) = _calls(fresh_store, account_id)
    assert row["payload_sealed"] and SYSTEM not in row["payload_sealed"], "kept, and kept encrypted"
    from core.llm_audit import open_llm_call

    opened = open_llm_call(account_id, row["id"])
    assert opened["prompt"] == f"[SYSTEM]\n{SYSTEM}\n\n[USER]\n{USER}"
    assert opened["response"] == '{"intent": "breakdown"}'
    assert opened["matches"] is True and row["payload_hash"] == hashlib.sha256(opened["prompt"].encode()).hexdigest()


def test_a_workspace_never_set_up_for_compliance_is_kept_the_same_way(fresh_store):
    account_id = f"acct{os.urandom(4).hex()}"
    fresh_store.upsert_client(account_id, "New Ltd")
    with fresh_store.get_db() as conn:
        conn.execute("DELETE FROM compliance_profile WHERE account_id=?", (account_id,))
    _call(account_id, enabled=False)
    (row,) = _calls(fresh_store, account_id)
    assert row["payload_sealed"]


def test_a_standard_workspace_is_recorded_only_with_its_switch_on_and_never_whole(fresh_store):
    off = _workspace(fresh_store, "standard")
    _call(off, enabled=False)
    assert _calls(fresh_store, off) == []
    on = _workspace(fresh_store, "standard", audit=True)
    _call(on, enabled=True)
    (row,) = _calls(fresh_store, on)
    assert row["payload_hash"] and not row["payload_sealed"] and not row["response_sealed"]


def test_an_admin_opens_a_call_and_reads_what_was_sent(fresh_store, monkeypatch):
    from admin import ai_calls, routes

    monkeypatch.setattr(ai_calls, "_is_auth", lambda request: True)
    monkeypatch.setattr(routes, "_is_auth", lambda request: True)
    account_id = _workspace(fresh_store, "regulated")
    _call(account_id, enabled=False)
    (row,) = _calls(fresh_store, account_id)
    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)
    page = client.get(f"/admin/clients/{account_id}/ai-calls/{row['id']}").text
    assert "QUESTION: net sales by region in 2025" in page and "matches the fingerprint" in page
    other = _workspace(fresh_store, "regulated")
    response = client.get(f"/admin/clients/{other}/ai-calls/{row['id']}", follow_redirects=False)
    assert response.status_code == 404, "a call is opened only from its own workspace"
    log = client.get(f"/admin/clients/{account_id}/egress").text
    assert f"/admin/clients/{account_id}/ai-calls/{row['id']}" in log and "Always on" in log


def test_a_changed_record_no_longer_matches(fresh_store):
    from core.llm_audit import open_llm_call

    account_id = _workspace(fresh_store, "regulated")
    _call(account_id, enabled=False)
    (row,) = _calls(fresh_store, account_id)
    with fresh_store.get_db() as conn:
        conn.execute("UPDATE llm_call_log SET payload_hash=? WHERE id=?", ("0" * 64, row["id"]))
    assert open_llm_call(account_id, row["id"])["matches"] is False


def test_a_call_whose_response_was_not_kept_does_not_match(fresh_store):
    from core.llm_audit import open_llm_call

    account_id = _workspace(fresh_store, "regulated")
    _call(account_id, enabled=False)
    (row,) = _calls(fresh_store, account_id)
    with fresh_store.get_db() as conn:
        conn.execute("UPDATE llm_call_log SET response_sealed='' WHERE id=?", (row["id"],))
    opened = open_llm_call(account_id, row["id"])
    assert opened["matches"] is False and "response was not kept" in opened["problem"]
