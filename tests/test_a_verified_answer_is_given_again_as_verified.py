"""A verified answer is given again, as verified.

An admin checks an answer and marks it verified (Questions > Verified answers, or from the question log). When a
reader asks that question again -- the same words, however written -- it is answered from the verified plan, with no
AI reading it afresh, and the answer says "Verified by <admin>". As Databricks' trusted assets, Snowflake's verified
queries and Looker's golden queries do: the questions a business asks most get one checked answer.

    the same question    the same words once case, punctuation, spacing and a "please" are set aside; never a
                         different question that looks alike
    a new question only  a follow-up ("and by region?") reads the answer above, so it is never a verified question
    still governed       the verified plan runs as any plan does: the reader's own tables and rows
    kept honest          a verified plan the data no longer answers (a metric taken out) is set aside: the question
                         is read afresh, and the admin is told it needs a look
    asked often          the questions answered most that nobody has verified yet are offered to verify

Invented data only (the retail domain); the AI is a recording.
"""

from __future__ import annotations

import datetime as dt
import uuid
from unittest.mock import patch

import pytest

import store
from core2.service import Session, answer_question
from tests.test_core2_answers_a_question_end_to_end import Recorded, _domain, _services
from evals.core2.compile_eval import _Words, golden

QUESTION = "What were net sales in 2025?"


def _plan(case_id: str = "retail-001"):
    _, built, model, _ = _domain("retail")
    case = next(c for c in golden("retail")["questions"] if c["id"] == case_id)
    return _Words(model, built).plan(case["plan"])


def _workspace() -> str:
    account = f"verified-{uuid.uuid4().hex[:10]}"
    store.upsert_client(account, "Retail Ltd")
    return account


# ── The same question ────────────────────────────────────────────────────────

def test_the_same_question_however_written_is_one_question():
    from core2.verified import key

    assert key("What were net sales in 2025?") == key("  what were NET sales in 2025  ")
    assert key("What were net sales in 2025?") == key("Please, what were net sales in 2025?")
    assert key("What were net sales in 2025?") != key("What were net sales in 2024?")
    assert key("Net sales by region") != key("Net sales by store")
    assert key("") == ""
    # A sign that changes what is asked is part of the question.
    assert key("customers with > 5 orders") != key("customers with < 5 orders")
    assert key("products whose growth is below -5%") != key("products whose growth is below 5%")
    assert key("orders over $100") != key("orders over 100")


# ── Kept by the admin ────────────────────────────────────────────────────────

def test_an_answer_is_verified_listed_found_and_taken_out():
    account = _workspace()
    plan = _plan().model_dump(mode="json", exclude_defaults=True)
    made = store.verify_answer(account, QUESTION, plan, by="Adi Admin")
    assert store.verify_answer(account, QUESTION.upper(), plan, by="Adi Admin") == made, "one entry per question"
    [kept] = store.list_verified(account)
    assert kept["question"] == QUESTION and kept["by"] == "Adi Admin" and kept["uses"] == 0
    found = store.find_verified(account, "what were net sales in 2025")
    assert found is not None and found["id"] == made and found["plan"] == plan
    assert store.find_verified(_workspace(), QUESTION) is None, "another workspace's verified answer"
    assert store.remove_verified(account, made)
    assert store.find_verified(account, QUESTION) is None


# ── Given again ──────────────────────────────────────────────────────────────

def _verified_lookup(plan, by="Adi Admin"):
    seen: list[tuple[int, bool]] = []

    def find(question: str):
        from core2.verified import Verified, key

        return Verified(7, QUESTION, plan.model_dump(mode="json", exclude_defaults=True), by, "2026-06-01 09:00") \
            if key(question) == key(QUESTION) else None

    return find, (lambda vid, ok: seen.append((vid, ok))), seen


def test_a_verified_question_is_answered_from_its_plan_with_no_ai_and_says_so():
    _, _, model, warehouse = _domain("retail")
    find, used, seen = _verified_lookup(_plan())
    ai = Recorded()            # no answer to give: any call to it fails the test
    payload = answer_question("what were net sales in 2025", _services(model, warehouse, ai, verified=find,
                                                                     verified_seen=used), Session())
    assert ai.sent == [], "the AI was asked to read a verified question"
    assert payload["type"] == "assistant_response", payload
    badges = payload["answer"].get("badges") or []
    assert badges and badges[0]["kind"] == "verified" and badges[0]["text"] == "Verified by Adi Admin"
    assert payload["trust"]["verified"]["by"] == "Adi Admin"
    assert seen == [(7, True)]


def test_the_figure_is_the_one_the_ai_reading_gives():
    _, _, model, warehouse = _domain("retail")
    plan = _plan()
    read = answer_question(QUESTION, _services(model, warehouse, Recorded(plan.model_dump_json(exclude_defaults=True))),
                           Session())
    find, used, _ = _verified_lookup(plan)
    verified = answer_question(QUESTION, _services(model, warehouse, Recorded(), verified=find, verified_seen=used),
                               Session())
    assert verified["answer"]["headline"] == read["answer"]["headline"]


def test_a_follow_up_is_never_a_verified_question():
    _, _, model, warehouse = _domain("retail")
    plan = _plan()
    find, used, seen = _verified_lookup(plan)
    session = Session()
    ai = Recorded(plan.model_dump_json(exclude_defaults=True), _plan("retail-011").model_dump_json(exclude_defaults=True))
    services = _services(model, warehouse, ai, verified=find, verified_seen=used)
    answer_question("Net sales in 2025, please", services, session)
    payload = answer_question("and by region?", services, session)
    assert not any(b.get("kind") == "verified" for b in (payload.get("answer") or {}).get("badges") or [])
    assert len(ai.sent) == 2 and seen == []


def test_a_verified_plan_the_data_no_longer_answers_is_set_aside_and_the_question_read_afresh():
    _, _, model, warehouse = _domain("retail")
    plan = _plan()
    broken = plan.model_copy(update={"measures": ["no_such_metric"]})
    find, used, seen = _verified_lookup(broken)
    ai = Recorded(plan.model_dump_json(exclude_defaults=True))
    payload = answer_question(QUESTION, _services(model, warehouse, ai, verified=find, verified_seen=used), Session())
    assert len(ai.sent) == 1, "the question was not read afresh"
    assert payload["type"] == "assistant_response"
    assert not any(b.get("kind") == "verified" for b in payload["answer"].get("badges") or [])
    assert seen == [(7, False)]


def test_a_reader_without_the_table_is_refused_a_verified_answer_as_any_other():
    _, _, model, warehouse = _domain("retail")
    find, used, _ = _verified_lookup(_plan())
    payload = answer_question(QUESTION, _services(model, warehouse, Recorded(), verified=find, verified_seen=used,
                                                  allowed_tables={"customers"}), Session())
    assert not (payload.get("data") or {}).get("total_rows"), payload


def test_an_answer_the_admin_verified_reaches_the_reader_as_the_portal_finds_it():
    # From the admin's store to the reader's answer, as the portal wires it: nothing handed over by the test.
    from core2.service import _verified_lookup

    account = _workspace()
    store.verify_answer(account, QUESTION, _plan().model_dump(mode="json", exclude_defaults=True), by="Adi Admin")
    _, _, model, warehouse = _domain("retail")
    ai = Recorded()
    payload = answer_question("What were NET sales in 2025", _services(
        model, warehouse, ai, verified=_verified_lookup(account),
        verified_seen=lambda vid, ok: store.verified_seen(account, vid, ok)), Session())
    assert ai.sent == [] and payload["answer"]["badges"][0]["text"] == "Verified by Adi Admin"
    assert store.list_verified(account)[0]["uses"] == 1


def test_an_admin_with_no_name_is_said_in_the_readers_language():
    from core2.verified import Verified, badge

    said = badge(Verified(1, QUESTION, {}, "", "2026-06-01 09:00"))
    assert said["text"] == "Verified by an admin" and said["text_fr"] == "Vérifié par un administrateur"


# ── Only an answer read from the question's own words can be verified ───────

def test_a_new_question_read_by_the_ai_from_its_own_words_can_be_verified():
    _, _, model, warehouse = _domain("retail")
    payload = answer_question(QUESTION, _services(model, warehouse, Recorded(_plan().model_dump_json(
        exclude_defaults=True))), Session())
    assert payload.get("verifiable") is True


def test_a_follow_up_cannot_be_verified():
    _, _, model, warehouse = _domain("retail")
    session = Session()
    services = _services(model, warehouse, Recorded(_plan().model_dump_json(exclude_defaults=True),
                                                    _plan("retail-011").model_dump_json(exclude_defaults=True)))
    answer_question(QUESTION, services, session)
    # Its plan keeps 2025 from above, a year its own words do not name; that rule set aside, it is still no
    # question an admin can verify: it reads the answer above.
    with patch("core2.service._dates_the_words_do_not_name", return_value=False):
        assert not answer_question("and by region?", services, session).get("verifiable")


def test_an_answer_under_a_readers_standing_condition_cannot_be_verified():
    from core2.plan.ir import Filter
    from core2.standing import Standing

    _, _, model, warehouse = _domain("retail")
    downtown = Standing("dt", "Store is Downtown Store", "always",
                        filters=[Filter(field="store.name", op="eq", values=["Downtown Store"])])
    payload = answer_question(QUESTION, _services(model, warehouse, Recorded(_plan().model_dump_json(
        exclude_defaults=True)), always=[downtown]), Session())
    assert payload["type"] == "assistant_response"
    assert not payload.get("verifiable"), "Downtown Store would be given to everyone who asks, as verified"


def test_a_day_the_ai_fixed_for_words_with_no_year_cannot_be_verified():
    from core2.service import _dates_the_words_do_not_name

    since = _plan().model_copy(deep=True)
    since.time.window = since.time.window.model_copy(update={"kind": "since", "start": dt.date(2026, 1, 1),
                                                             "end": None})
    assert _dates_the_words_do_not_name(since, "net sales since January")
    assert not _dates_the_words_do_not_name(since, "net sales since January 2026")
    assert not _dates_the_words_do_not_name(_plan(), QUESTION)
    # A year named elsewhere in the question, or a number that only looks like one, names no year the plan holds.
    assert _dates_the_words_do_not_name(since, "net sales since January, against 2024")
    assert _dates_the_words_do_not_name(since, "the top 2000 customers since January")
    fiscal = since.model_copy(deep=True)
    fiscal.time.window = fiscal.time.window.model_copy(update={"kind": "between", "start": dt.date(2025, 4, 1),
                                                               "end": dt.date(2026, 3, 31)})
    assert not _dates_the_words_do_not_name(fiscal, "net sales in FY26")


def test_the_bridge_keeps_whether_an_answer_can_be_verified():
    # Recorded as the chat records every answer, and read back as the admin's Verify does.
    from gateway.core2_bridge import _record

    account = _workspace()
    plan = _plan().model_dump(mode="json", exclude_defaults=True)
    plain, under = uuid.uuid4().hex, uuid.uuid4().hex
    _record(account, {"id": 1}, "core2", QUESTION, "answered", {"plan": plan, "verifiable": True}, 10, plain)
    _record(account, {"id": 1}, "core2", QUESTION, "answered", {"plan": plan}, 10, under)
    assert store.core2_answered(account, plain) is not None
    assert store.core2_answered(account, under) is None


def test_the_answers_check_reads_its_questions_afresh():
    from core2 import answers_check

    asked = {}
    with patch("core2.service.portal_answer", lambda *a, **k: asked.update(k) or {}):
        answers_check._ask("acct", {"id": 1, "question": QUESTION}, 1)
    assert asked.get("use_verified") is False


# ── Its use and its upkeep ───────────────────────────────────────────────────

def test_each_use_is_counted_and_a_plan_set_aside_needs_a_look():
    account = _workspace()
    made = store.verify_answer(account, QUESTION, _plan().model_dump(mode="json"), by="Adi Admin")
    store.verified_seen(account, made, True)
    store.verified_seen(account, made, True)
    [kept] = store.list_verified(account)
    assert kept["uses"] == 2 and kept["last_used_at"] and not kept["needs_look"]
    store.verified_seen(account, made, False)
    assert store.list_verified(account)[0]["needs_look"]
    store.verify_answer(account, QUESTION, _plan().model_dump(mode="json"), by="Adi Admin")
    assert not store.list_verified(account)[0]["needs_look"], "verified again: looked at"


def _answered(account: str, question: str, plan: dict, status: str = "answered", verifiable: bool = True) -> str:
    qid = uuid.uuid4().hex
    store.log_core2_answer(account, user_id="1", mode="core2", question=question, status=status, headline="",
                           sql="SELECT 1", row_count=1, plan=plan, duration_ms=10, model_version=1, question_id=qid,
                           verifiable=verifiable)
    return qid


def test_the_questions_answered_most_and_not_verified_are_offered():
    account = _workspace()
    plan = _plan().model_dump(mode="json")
    for _ in range(3):
        _answered(account, "Net sales by region in 2025", plan)
    for _ in range(2):
        _answered(account, "what were net sales in 2025?", plan)
    _answered(account, "Asked once only", plan)
    _answered(account, "A refusal", plan, status="refused")
    _answered(account, "A refusal", plan, status="refused")
    for _ in range(4):
        _answered(account, "and by month?", plan, verifiable=False)      # a follow-up, asked often
    store.verify_answer(account, QUESTION, plan, by="Adi Admin")
    offered = store.verified_candidates(account)
    assert [o["question"] for o in offered] == ["Net sales by region in 2025"]
    assert offered[0]["times"] == 3 and offered[0]["question_id"]


# ── The admin pages ──────────────────────────────────────────────────────────

@pytest.fixture
def admin():
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from admin import core2_verified, routes

    app = FastAPI()
    app.include_router(routes.router)
    with patch.object(routes, "_is_auth", return_value=True), \
            patch.object(core2_verified, "_is_auth", return_value=True):
        yield TestClient(app)


def test_an_admin_verifies_an_answer_from_the_log_and_takes_it_out(admin):
    account = _workspace()
    plan = _plan().model_dump(mode="json")
    qid = _answered(account, QUESTION, plan)
    made = admin.post(f"/admin/clients/{account}/verified", data={"question_id": qid}, follow_redirects=False)
    assert made.status_code == 303
    [kept] = store.list_verified(account)
    assert kept["question"] == QUESTION and kept["plan"] == plan
    page = admin.get(f"/admin/clients/{account}/verified")
    assert page.status_code == 200 and QUESTION in page.text and "Verified answers" in page.text
    gone = admin.post(f"/admin/clients/{account}/verified/{kept['id']}/remove", follow_redirects=False)
    assert gone.status_code == 303 and store.list_verified(account) == []


def test_only_an_answered_question_of_this_workspace_can_be_verified(admin):
    account, other = _workspace(), _workspace()
    plan = _plan().model_dump(mode="json")
    refused = _answered(account, "A refusal", plan, status="refused")
    elsewhere = _answered(other, QUESTION, plan)
    follow_up = _answered(account, "and in 2024?", plan, verifiable=False)
    for qid in (refused, elsewhere, follow_up, "no-such-question"):
        said = admin.post(f"/admin/clients/{account}/verified", data={"question_id": qid}, follow_redirects=False)
        assert said.status_code == 303 and "error=" in said.headers["location"], said.headers.get("location")
    assert store.list_verified(account) == [] and store.list_verified(other) == []


def test_the_verified_page_is_a_data_stewards():
    from admin.access import rule_for

    role, scoped, account = rule_for("/clients/acct-1/verified")
    assert role == "data_steward" and scoped and account == "acct-1"
