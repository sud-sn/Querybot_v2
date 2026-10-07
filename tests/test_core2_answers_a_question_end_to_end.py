"""A question through the whole of core2, with the AI's answers recorded.

The service finds member names in the question, asks the (recorded) AI for a
plan, checks and resolves it, compiles SQL for the warehouse, runs it and builds
the portal's frame. Every golden question must come out with the reference
numbers; a regulated tenant's member values must never reach the AI; a plan the
AI gets wrong is repaired once and otherwise becomes a question back; a reader
only ever reaches the tables they may use.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.values import MemberIndex, build_index
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import _Words, golden, learn, same_rows
from evals.core2.framework import materialize

TODAY = dt.date(2026, 6, 15)
_built: dict[str, tuple] = {}


def _domain(name: str):
    if name not in _built:
        domain = domains.build(name)
        built, model = learn(domain, "descriptive")
        _built[name] = (domain, built, model, DuckDBWarehouse(materialize(domain, "descriptive").con))
    return _built[name]


def _index(model, warehouse) -> MemberIndex:
    def fetch(slug: str) -> list[object]:
        column = model.columns[model.attributes[slug].column]
        table = model.tables[column.table]
        return [r[0] for r in warehouse.query(f'SELECT DISTINCT "{column.name}" FROM "{table.name}"').rows]

    return build_index(model, fetch)


class Recorded:
    """The AI, answering from a script; it keeps what it was sent."""

    def __init__(self, *answers: str):
        self.answers = list(answers)
        self.sent: list[str] = []

    def __call__(self, stable: str, tail: str) -> str:
        self.sent.append(tail)
        return self.answers.pop(0)


def _services(model, warehouse, ai, **kw) -> Services:
    return Services(model=model, warehouse=warehouse, complete=ai, index=kw.pop("index", MemberIndex()),
                    today=TODAY, **kw)


CASES = [(name, case["id"]) for name in ("retail", "inventory") for case in golden(name)["questions"]
         if case.get("plan") and case.get("reference_sql")]


@pytest.mark.parametrize("name, case_id", CASES)
def test_every_golden_question_comes_out_with_the_reference_numbers(name, case_id):
    domain, built, model, reference = _domain(name)
    case = next(c for c in golden(name)["questions"] if c["id"] == case_id)
    plan = _Words(model, built).plan(case["plan"])
    warehouse = DuckDBWarehouse(built.con)
    ai = Recorded(plan.model_dump_json(exclude_defaults=True))
    payload = answer_question(case["question"], _services(model, warehouse, ai), Session())
    assert payload["type"] == "assistant_response" and payload["engine"] == "core2"
    # The portal's answer card reads these keys (portal_chat.html appendAssistantResponse).
    assert set(payload["answer"]) >= {"headline", "short_value", "comparison", "scope_note"}
    assert payload["answer"]["headline"] and payload["data"]["total_rows"] >= 1, payload["answer"]
    json.dumps(payload)     # the frame crosses a websocket as it is: no Decimal, no date objects
    ran = warehouse.query(warehouse.log[-1])
    expected = reference.query(case["reference_sql"])
    diff = same_rows(expected.columns, expected.rows, ran.columns, ran.rows,
                     order_matters=bool((case.get("expect") or {}).get("order_matters")))
    assert not diff, diff


def test_member_names_are_found_and_handed_to_the_ai():
    _, built, model, _ = _domain("retail")
    warehouse = DuckDBWarehouse(built.con)
    slug = next(a.slug for a in model.attributes.values() if model.columns[a.column].name == "customer_name")
    plan = {"kind": "query", "intent": "value", "measures": ["net_amount"],
            "filters": [{"field": slug, "op": "eq", "values": ["Northline Distribution 58"]}],
            "time": {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}}
    ai = Recorded(json.dumps(plan))
    payload = answer_question("net sales for northline distribution 58 in 2025",
                              _services(model, warehouse, ai, index=_index(model, warehouse)), Session())
    assert f'{slug} = "Northline Distribution 58"' in ai.sent[0]
    assert payload["kpi"] and round(payload["kpi"]["value"], 2) == 95698.79
    assert payload["answer"]["short_value"] == "$95,698.79" and payload["answer"]["comparison"] == "in 2025"


def test_a_regulated_tenants_member_values_never_reach_the_ai():
    _, built, model, _ = _domain("retail")
    warehouse = DuckDBWarehouse(built.con)
    slug = next(a.slug for a in model.attributes.values() if model.columns[a.column].name == "customer_name")
    plan = {"kind": "query", "intent": "value", "measures": ["net_amount"],
            "filters": [{"field": slug, "op": "eq", "values": ["⟨v1⟩"]}],
            "time": {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}}
    ai = Recorded(json.dumps(plan))
    services = _services(model, warehouse, ai, index=_index(model, warehouse), values_allowed=False)
    payload = answer_question("net sales for Northline Distribution 58 in 2025", services, Session())
    sent = ai.sent[0]
    assert "Northline" not in sent and "⟨v1⟩" in sent
    assert payload["plan"]["filters"][0]["values"] == ["Northline Distribution 58"]
    assert round(payload["kpi"]["value"], 2) == 95698.79


def test_personal_data_typed_into_a_question_is_scrubbed_before_the_ai():
    from core.masking import scrub_question_pii

    _, built, model, _ = _domain("retail")
    warehouse = DuckDBWarehouse(built.con)
    ai = Recorded(json.dumps({"kind": "query", "intent": "value", "measures": ["net_amount"]}))
    services = _services(model, warehouse, ai, values_allowed=False,
                         scrub=lambda text: scrub_question_pii(text, "")[0])
    answer_question("net sales for the customer jane.doe@example.com", services, Session())
    assert "jane.doe@example.com" not in ai.sent[0] and "[EMAIL]" in ai.sent[0]


def test_a_wrong_plan_is_repaired_once_and_then_asked_about():
    _, built, model, _ = _domain("retail")
    warehouse = DuckDBWarehouse(built.con)
    good = json.dumps({"kind": "query", "intent": "value", "measures": ["net_amount"]})
    ai = Recorded('{"kind": "query", "measures": ["net_sales"]}', good)
    payload = answer_question("net sales", _services(model, warehouse, ai), Session())
    assert "Allowed: net_amount" in ai.sent[1] and payload["trust"]["plan_repaired"]
    ai = Recorded('{"kind": "query", "measures": ["net_sales"]}', "not json at all")
    payload = answer_question("net sales", _services(model, warehouse, ai), Session())
    assert payload["clarify"]["about"] == "other" and payload["data"] is None


def test_a_reader_reaches_only_the_tables_they_may_use():
    _, built, model, _ = _domain("retail")
    warehouse = DuckDBWarehouse(built.con)
    customers = next(k for k, t in model.tables.items() if t.name == "customers")
    plan = json.dumps({"kind": "query", "intent": "breakdown", "measures": ["net_amount"], "group_by": ["customer"]})
    allowed = set(model.tables) - {customers}
    payload = answer_question("net sales by customer", _services(model, warehouse, Recorded(plan),
                                                                  allowed_tables=allowed), Session())
    assert "do not have access" in payload["answer"]["headline"] and not warehouse.log


def test_greetings_questions_about_the_data_and_the_unanswerable_are_answered_as_such():
    _, built, model, _ = _domain("inventory")
    warehouse = DuckDBWarehouse(built.con)
    hello = answer_question("hi", _services(model, warehouse, Recorded('{"kind": "smalltalk"}')), Session())
    assert hello["answer"]["headline"].startswith("Hello")
    about = answer_question("what can I ask?", _services(model, warehouse, Recorded('{"kind": "describe_data"}')),
                            Session())
    assert "on hand quantity" in about["answer"]["headline"].lower()
    sales = answer_question("sales last month", _services(model, warehouse, Recorded(
        '{"kind": "unsupported", "notes": ["this warehouse holds stock, not sales"]}')), Session())
    assert sales["unsupported"] and "holds stock" in sales["answer"]["headline"] and not warehouse.log


def test_a_follow_up_sees_the_previous_plan():
    _, built, model, _ = _domain("retail")
    warehouse = DuckDBWarehouse(built.con)
    first = json.dumps({"kind": "query", "intent": "value", "measures": ["net_amount"]})
    second = json.dumps({"kind": "query", "intent": "trend", "measures": ["net_amount"], "follow_up": "refine",
                         "time": {"grain": "year"}})
    ai = Recorded(first, second)
    session = Session()
    answer_question("net sales", _services(model, warehouse, ai), session)
    payload = answer_question("by year", _services(model, warehouse, ai), session)
    assert "PREVIOUS QUESTION: net sales" in ai.sent[1] and '"measures":["net_amount"]' in ai.sent[1]
    assert payload["chart"]["chart_type"] == "line"


def test_a_reply_to_a_question_back_is_read_with_that_question():
    _, built, model, _ = _domain("retail")
    warehouse = DuckDBWarehouse(built.con)
    ask = {"kind": "clarify", "clarify": {"about": "date", "question": "By order date or by ship date?",
                                          "options": ["order date", "ship date"]}}
    then = {"kind": "query", "intent": "trend", "measures": ["net_amount"],
            "time": {"date": "ship_date", "grain": "year"}}
    ai = Recorded(json.dumps(ask), json.dumps(then))
    session = Session()
    first = answer_question("net sales by year", _services(model, warehouse, ai), session)
    assert first["clarify"]["options"] == ["order date", "ship date"] and not warehouse.log
    second = answer_question("ship date", _services(model, warehouse, ai), session)
    assert "PREVIOUS QUESTION: net sales by year" in ai.sent[1] and "By order date or by ship date?" in ai.sent[1]
    assert second["data"]["total_rows"] >= 3
