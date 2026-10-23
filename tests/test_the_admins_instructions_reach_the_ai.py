"""The workspace's own instructions for the AI, written by its admins, are given with every question.

An admin writes, in plain words, what the data's words mean and how this business counts ("active customers bought
in the last 90 days", "sales means net amount unless a question says gross"), on Data > Knowledge base. The planner
is given them with every question, as Databricks Genie's general instructions and Snowflake Cortex Analyst's custom
instructions are. They are the admin's words, so:

    said where it goes   the page says they are sent to the AI with every question
    in the fixed part    beside the catalog, the same for every question, so the AI's prompt cache still holds
    named in the log     the AI egress log lists them as the admin's instructions
    kept short           2,000 characters at most, refused past that, never cut silently
    scrubbed             a workspace under compliance sends them with personal data taken out, as its questions are
    never more access    the plan they lead to is checked against the catalog and the reader's tables, as any plan

Invented data only (the retail domain); the AI is a recording.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import patch

import pytest

import store
from core2.service import Session, answer_question
from tests.test_core2_answers_a_question_end_to_end import _domain, _services
from evals.core2.compile_eval import _Words, golden

RULE = "Sales means net amount unless the question says gross. A season is three months starting in March."


class Seen:
    """The AI, answering from a script; it keeps both parts of what it was sent."""

    def __init__(self, *answers: str):
        self.answers = list(answers)
        self.stable: list[str] = []
        self.tails: list[str] = []

    def __call__(self, stable: str, tail: str) -> str:
        self.stable.append(stable)
        self.tails.append(tail)
        return self.answers.pop(0)


def _plan_json(case_id: str = "retail-001") -> str:
    _, built, model, _ = _domain("retail")
    case = next(c for c in golden("retail")["questions"] if c["id"] == case_id)
    return _Words(model, built).plan(case["plan"]).model_dump_json(exclude_defaults=True)


# ── Given to the AI ──────────────────────────────────────────────────────────

def test_the_instructions_are_in_the_fixed_part_of_every_question():
    _, _, model, warehouse = _domain("retail")
    ai = Seen(_plan_json(), _plan_json())
    services = _services(model, warehouse, ai, instructions=RULE)
    answer_question("What were net sales in 2025?", services, Session())
    answer_question("What were net sales in 2025 by region?", services, Session())
    assert all("WORKSPACE INSTRUCTIONS" in s and RULE in s for s in ai.stable)
    assert ai.stable[0] == ai.stable[1], "the fixed part changed between questions: the prompt cache is lost"
    assert not any(RULE in t for t in ai.tails)


def test_none_written_none_sent():
    _, _, model, warehouse = _domain("retail")
    ai = Seen(_plan_json())
    answer_question("What were net sales in 2025?", _services(model, warehouse, ai), Session())
    assert "WORKSPACE INSTRUCTIONS" not in ai.stable[0]


def test_the_ai_egress_log_names_them():
    from core.llm_audit import build_egress_manifest
    from core2.plan.planner import stable_prompt

    _, _, model, _ = _domain("retail")
    assert "admin_instructions" in build_egress_manifest(stable_prompt(model, instructions=RULE), "Q")["content"]
    assert "admin_instructions" not in build_egress_manifest(stable_prompt(model), "Q")["content"]


# ── Written by an admin ──────────────────────────────────────────────────────

def _workspace() -> str:
    account = f"instr-{uuid.uuid4().hex[:10]}"
    store.upsert_client(account, "Retail Ltd")
    return account


@pytest.fixture
def admin():
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from admin import core2_knowledge, routes

    app = FastAPI()
    app.include_router(routes.router)
    with patch.object(routes, "_is_auth", return_value=True), \
            patch.object(core2_knowledge, "_is_auth", return_value=True):
        yield TestClient(app)


def test_an_admin_writes_them_and_the_question_is_given_them(admin):
    # From the admin's page to what the planner reads, as the portal reads it: nothing handed over by the test.
    from core2.service import workspace_instructions

    account = _workspace()
    saved = admin.post(f"/admin/clients/{account}/knowledge/instructions", data={"instructions": RULE + "\r\n"},
                       follow_redirects=False)
    assert saved.status_code == 303 and "saved=instructions" in saved.headers["location"]
    assert workspace_instructions(account) == RULE
    page = admin.get(f"/admin/clients/{account}/knowledge")
    assert page.status_code == 200 and "Instructions for the AI" in page.text and RULE in page.text
    assert "sent to the AI" in page.text
    _, _, model, warehouse = _domain("retail")
    ai = Seen(_plan_json())
    answer_question("What were net sales in 2025?", _services(model, warehouse, ai,
                                                             instructions=workspace_instructions(account)), Session())
    assert RULE in ai.stable[0]


def test_past_the_limit_they_are_refused_never_cut(admin):
    from core2.service import workspace_instructions

    account = _workspace()
    admin.post(f"/admin/clients/{account}/knowledge/instructions", data={"instructions": RULE},
               follow_redirects=False)
    said = admin.post(f"/admin/clients/{account}/knowledge/instructions", data={"instructions": "x" * 2001},
                      follow_redirects=False)
    assert said.status_code == 303 and "error=" in said.headers["location"]
    assert workspace_instructions(account) == RULE, "the earlier instructions were lost or cut"


def test_cleared_they_are_gone(admin):
    from core2.service import workspace_instructions

    account = _workspace()
    admin.post(f"/admin/clients/{account}/knowledge/instructions", data={"instructions": RULE})
    admin.post(f"/admin/clients/{account}/knowledge/instructions", data={"instructions": "  "})
    assert workspace_instructions(account) == ""


def test_saving_them_keeps_the_rest_of_the_workspaces_settings(admin):
    account = _workspace()
    store.update_client_state(account, "READY", {"schema_dir": "/somewhere", "general_knowledge_enabled": False})
    admin.post(f"/admin/clients/{account}/knowledge/instructions", data={"instructions": RULE})
    client = store.get_client(account)
    state = json.loads(client["state_data"])
    assert client["state"] == "READY" and state["schema_dir"] == "/somewhere"
    assert state["general_knowledge_enabled"] is False


# ── Under compliance ─────────────────────────────────────────────────────────

def test_a_workspace_under_compliance_sends_them_scrubbed():
    from core2.service import _instructions_for

    scrub = lambda text: text.replace("Jane Smith", "<PERSON>")   # noqa: E731 - the workspace's own scrubber
    assert _instructions_for("Accounts Jane Smith manages are key accounts.", scrub)["instructions"] == \
        "Accounts <PERSON> manages are key accounts."
    assert _instructions_for("Accounts Jane Smith manages are key accounts.", None)["instructions"] == \
        "Accounts Jane Smith manages are key accounts."


class _Index:
    """The workspace's member names: one store."""

    def match(self, text, **_):
        from core2.plan.values import ValueMatch

        at = text.find("Downtown Store")
        return [ValueMatch("Downtown Store", at, at + len("Downtown Store"), "store.name", "Downtown Store")] \
            if at >= 0 else []


EXCLUDE = "Leave out Downtown Store unless a question names it."


def test_where_values_are_kept_from_the_ai_a_member_they_name_is_not_sent():
    from core2.service import _instructions_for

    sent = _instructions_for(EXCLUDE, None, _Index(), values_allowed=False)
    assert "Downtown" not in sent["instructions"] and "<a member>" in sent["instructions"]
    assert not sent["instructions_name_members"]


def test_where_values_may_be_sent_a_member_they_name_is_counted_as_sent():
    from core.llm_audit import build_egress_manifest
    from core2.plan.planner import stable_prompt
    from core2.service import _instructions_for

    sent = _instructions_for(EXCLUDE, None, _Index(), values_allowed=True)
    assert sent == {"instructions": EXCLUDE, "instructions_name_members": True}
    _, _, model, _ = _domain("retail")
    manifest = build_egress_manifest(stable_prompt(model, instructions=sent["instructions"],
                                                   instructions_name_members=True), "Q")
    assert manifest["values_sent"] and "admin_instruction_values" in manifest["value_sources"]
    plain = build_egress_manifest(stable_prompt(model, instructions=RULE), "Q")
    assert "admin_instruction_values" not in plain["value_sources"]


def test_the_page_says_a_compliance_workspace_takes_personal_data_out(admin):
    account = _workspace()
    with patch("core2.service.question_scrubber", return_value=lambda text: text):
        page = admin.get(f"/admin/clients/{account}/knowledge")
    assert "personal data taken out" in page.text
