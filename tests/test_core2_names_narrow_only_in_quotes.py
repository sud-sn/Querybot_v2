"""A member narrows an answer only when the reader writes its name in quotes.

Words of a question were matched against every member name the warehouse holds, offered to the AI as
names to filter on, and an ordinary word that happened to be a member's name ("available stock", "open
orders") became a WHERE condition. Now only a name in quotes ("North", “North”, « Nord ») is
offered, and a filter on a member the reader did not quote is taken out of the plan, said on the answer,
with the same question one click away with the name in quotes. A member the reader points at on screen
("the first one"), one an earlier answer of the conversation filtered on, a number and a flag still
filter. In a workspace whose member values may not reach the AI, every name the question writes is still
withheld from it, quoted or not. Questions the product writes for the reader (chips, the drill menu) put
their members in quotes.

Invented data only.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.values import MemberIndex, build_index, quoted_spans
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn
from evals.core2.framework import materialize

TODAY = dt.date(2026, 6, 15)
CUSTOMER = "Northline Distribution 58"
Y2025 = {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}


@pytest.fixture(scope="module")
def retail():
    domain = domains.build("retail")
    built, model = learn(domain, "descriptive")
    warehouse = DuckDBWarehouse(materialize(domain, "descriptive").con)

    def fetch(slug):
        column = model.columns[model.attributes[slug].column]
        table = model.tables[column.table]
        return [r[0] for r in warehouse.query(f'SELECT DISTINCT "{column.name}" FROM "{table.name}"').rows]

    slug = next(a.slug for a in model.attributes.values() if model.columns[a.column].name == "customer_name")
    return model, warehouse, build_index(model, fetch), slug


class Recorded:
    def __init__(self, *plans):
        self.plans = [json.dumps({"kind": "query", **p}) for p in plans]
        self.sent: list[str] = []

    def __call__(self, stable, tail):
        self.sent.append(tail)
        return self.plans.pop(0)


def _ask(retail, ai, *questions, values_allowed=True):
    model, warehouse, index, _ = retail
    services = Services(model=model, warehouse=warehouse, complete=ai, index=index, today=TODAY,
                        values_allowed=values_allowed)
    session = Session()
    return [answer_question(q, services, session) for q in questions]


def _total(payload):
    return round(payload["kpi"]["value"], 2)


def _for_customer(slug, value=CUSTOMER):
    return {"intent": "value", "measures": ["net_amount"], "time": Y2025,
            "filters": [{"field": slug, "op": "eq", "values": [value]}]}


def test_quoted_text_is_found_in_every_kind_of_quotes():
    for question, want in [('net sales for "North" in 2025', ["North"]), ("sales for “North”", ["North"]),
                           ("ventes pour « Nord »", ["Nord"]), ("the customer's sales for 'North'", ["North"]),
                           ("stock of 3' pipe and 4' pipe", []), ("net sales in 2025", [])]:
        assert [question[a:b] for a, b in quoted_spans(question)] == want, question


def test_a_name_not_in_quotes_never_filters(retail):
    _, _, _, slug = retail
    whole = _ask(retail, Recorded({"intent": "value", "measures": ["net_amount"], "time": Y2025}),
                 "net sales in 2025")[0]
    ai = Recorded(_for_customer(slug))
    payload = _ask(retail, ai, f"net sales for {CUSTOMER.lower()} in 2025")[0]
    assert "VALUE MATCHES" not in ai.sent[0], "an unquoted name is not offered as one to filter on"
    assert _total(payload) == _total(whole), "the filter the AI wrote anyway is taken out"
    assert any(f'Not narrowed to customer' in n and CUSTOMER in n for n in payload["trust"]["date_context"])
    chip = payload["follow_up_suggestions"][0]
    assert chip["label"] == f'Only "{CUSTOMER}"'
    assert chip["question"] == f'net sales for "{CUSTOMER.lower()}" in 2025'


@pytest.mark.parametrize("written", [f'"{CUSTOMER}"', f"“{CUSTOMER}”", f"« {CUSTOMER} »",
                                     f'"{CUSTOMER.lower()}"'])
def test_a_quoted_name_filters(retail, written):
    _, _, _, slug = retail
    ai = Recorded(_for_customer(slug))
    payload = _ask(retail, ai, f"net sales for {written} in 2025")[0]
    assert f'{slug} = "{CUSTOMER}"' in ai.sent[0]
    assert _total(payload) == 95698.79
    assert not any("Not narrowed" in n for n in payload["trust"]["date_context"])


def test_a_follow_up_keeps_the_filter_its_answer_had(retail):
    _, _, _, slug = retail
    first = _for_customer(slug)
    again = {**first, "time": {"window": {"kind": "between", "start": "2024-01-01", "end": "2024-12-31"}},
             "follow_up": "refine"}
    answers = _ask(retail, Recorded(first, again), f'net sales for "{CUSTOMER}" in 2025', "and in 2024?")
    assert answers[1]["plan"]["filters"][0]["values"] == [CUSTOMER]
    assert not any("Not narrowed" in n for n in answers[1]["trust"]["date_context"])


def test_the_member_pointed_at_on_screen_still_filters(retail):
    _, _, _, slug = retail
    ranking = {"intent": "rank", "measures": ["net_amount"], "group_by": [slug], "time": Y2025,
               "sort": [{"by": "net_amount", "desc": True}], "limit": 5}
    answers = _ask(retail, Recorded(ranking), "top 5 customers by net sales in 2025")
    top = answers[0]["data"]["rows"][0]
    leader = next(v for k, v in top.items() if isinstance(v, str))
    drill = {"intent": "trend", "measures": ["net_amount"], "time": {**Y2025, "grain": "quarter"},
             "filters": [{"field": slug, "op": "eq", "values": [leader]}], "follow_up": "refine"}
    ai = Recorded(ranking, drill)
    answers = _ask(retail, ai, "top 5 customers by net sales in 2025", "break the first one down by quarter")
    assert answers[1]["plan"]["filters"][0]["values"] == [leader]
    assert f'= "{leader}"' in ai.sent[1], "the member on screen is offered to the AI"


def test_a_number_condition_is_no_name_and_still_filters(retail):
    _, _, _, slug = retail
    plan = {"intent": "rank", "measures": ["net_amount"], "group_by": [slug], "time": Y2025,
            "filters": [{"field": "net_amount", "op": "gt", "values": [50000]}]}
    payload = _ask(retail, Recorded(plan), "customers with more than 50000 of net sales in 2025")[0]
    assert payload["plan"]["filters"] == plan["filters"]
    assert payload["data"]["rows"] and all(r.get("net_amount", 0) > 50000 for r in payload["data"]["rows"])


def test_where_values_may_not_reach_the_ai_an_unquoted_name_is_still_withheld(retail):
    _, _, _, slug = retail
    ai = Recorded(_for_customer(slug, "⟨v1⟩"))
    payload = _ask(retail, ai, f"net sales for {CUSTOMER} in 2025", values_allowed=False)[0]
    assert "Northline" not in ai.sent[0] and "⟨v1⟩" in ai.sent[0]
    assert "VALUE MATCHES" not in ai.sent[0]
    assert not payload["plan"].get("filters"), "and it does not filter: it was not quoted"


def test_the_questions_the_product_writes_put_their_members_in_quotes(retail):
    _, _, _, slug = retail
    breakdown = {"intent": "breakdown", "measures": ["net_amount"], "group_by": [slug], "time": Y2025,
                 "sort": [{"by": "net_amount", "desc": True}]}
    payload = _ask(retail, Recorded(breakdown), "net sales by customer in 2025")[0]
    drill = payload.get("drill") or (payload.get("chart") or {}).get("drill") or {}
    questions = [i["question"] for i in (drill.get("items") or [])]
    questions += [s["question"] for s in payload.get("follow_up_suggestions") or []]
    named = [q for q in questions if "{member}" in q or any(r.get(k) and str(r[k]) in q
                                                          for r in payload["data"]["rows"][:3] for k in r)]
    assert named, questions
    for q in named:
        assert '"{member}"' in q or q.count('"') >= 2, q

    filtered = {**_for_customer(slug), "intent": "breakdown", "group_by": ["product.category"]
                if "product.category" in retail[0].attributes else []}
    payload = _ask(retail, Recorded(filtered), f'net sales for "{CUSTOMER}" in 2025')[0]
    for s in payload.get("follow_up_suggestions") or []:
        if CUSTOMER in s["question"]:
            assert f'"{CUSTOMER}"' in s["question"], s


def test_the_questions_after_a_why_or_a_forecast_keep_their_members_in_quotes(retail):
    """A chip under a "why" or a forecast answer repeats the answer's conditions: sent as it is, it must still
    narrow to the same customer, and to the member the why-answer found."""
    _, _, _, slug = retail
    customer = [{"field": slug, "op": "eq", "values": [CUSTOMER]}]
    why = {"intent": "drivers", "measures": ["net_amount"], "filters": customer,
           "time": {"window": {"kind": "between", "start": "2025-07-01", "end": "2025-12-31"},
                    "compare": {"kind": "window", "window": {"kind": "between", "start": "2025-01-01",
                                                             "end": "2025-06-30"}}}}
    payload = _ask(retail, Recorded(why), f'why did net sales change for "{CUSTOMER}" in the second half of 2025?')[0]
    chips = [s["question"] for s in payload["follow_up_suggestions"]]
    assert chips and all(f'"{CUSTOMER}"' in q for q in chips), chips
    monthly = next(q for q in chips if q.startswith("Monthly"))
    grouping = payload["drivers"]["groupings"][0]
    leader = next(x["member"] for x in grouping["leaders"] if x["member"] != "Unknown")
    assert f'for "{leader}"' in monthly, monthly

    again = {"intent": "trend", "measures": ["net_amount"], "time": {"grain": "month"},
             "filters": [{"field": grouping["slug"], "op": "eq", "values": [leader]}, *customer]}
    sent = _ask(retail, Recorded(again), monthly)[0]
    assert {f["field"]: f["values"] for f in sent["plan"]["filters"]} == {grouping["slug"]: [leader], slug: [CUSTOMER]}
    assert not any("Not narrowed" in n for n in sent["trust"]["date_context"])

    ahead = {"intent": "forecast", "measures": ["net_amount"], "time": {"grain": "month"}, "filters": customer,
             "forecast": {"periods": 3}}
    payload = _ask(retail, Recorded(ahead), f'forecast net sales for "{CUSTOMER}"')[0]
    chips = [s["question"] for s in payload.get("follow_up_suggestions") or []]
    assert chips and all(f'"{CUSTOMER}"' in q for q in chips), chips
