"""A segment of members narrows, or splits, any other figure, equal to SQL written by hand.

"Net sales from returning customers by store", "net sales: new vs returning customers", "sales from new customers by
month". Each row is counted under its member's segment, the member's own activity deciding it (all of it, not only
what the figure counts):

* new: their first activity ever is in the period (the row's own period in a series, else the question's);
* returning: they had activity before it;
* won back: returning, with none in the period before;
* repeat / one-time: activity on 2 or more days of the period / on one day only.

Asked of the invented retail warehouse through the whole answer path (a scripted plan, as the planner writes it).
Invented data only.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest
import sqlglot

from core2.compile.compiler import compile_query
from core2.plan.ir import Plan
from core2.plan.values import MemberIndex
from core2.resolve.resolver import Context, resolve
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
LAST_2_MONTHS = {"kind": "last", "unit": "month", "n": 2}                 # April and May 2026
Q2_2023 = {"kind": "between", "start": "2023-04-01", "end": "2023-06-30"}
ORDERS = {"of": "customer", "measure": "number_of_orders"}
FIRST = """SELECT o.customer_id AS k, MIN(d.full_date) AS first FROM order_lines o
           JOIN calendar d ON o.order_date_key = d.date_key GROUP BY o.customer_id"""
LINES = """SELECT o.*, d.full_date AS day, s.store_name FROM order_lines o
           JOIN calendar d ON o.order_date_key = d.date_key JOIN stores s ON o.store_id = s.store_id
           WHERE o.customer_id IN (SELECT customer_id FROM customers)"""


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")


def _ask(learned, question: str, plan: dict) -> dict:
    built, model = learned
    answers = [json.dumps({"kind": "query", **plan})]
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=lambda s, t: answers.pop(0),
                        index=MemberIndex(), today=TODAY)
    payload = answer_question(question, services, Session())
    assert payload.get("type") == "assistant_response" and "export_rows" in payload, payload.get("answer")
    return payload


def _by(rows: list[dict], key_hint: str) -> dict:
    key = next(k for k in rows[0] if key_hint in k.casefold())
    value = next(k for k in rows[0] if k != key)
    return {r[key]: float(r[value]) for r in rows}


def test_net_sales_from_returning_customers_by_store(retail):
    want = dict(retail[0].con.execute(f"""
        WITH f AS ({FIRST}) SELECT l.store_name, SUM(l.net_amount) FROM ({LINES}) l JOIN f ON l.customer_id = f.k
        WHERE l.day >= DATE '2026-04-01' AND l.day < DATE '2026-06-01' AND f.first < DATE '2026-04-01'
        GROUP BY l.store_name""").fetchall())
    got = _ask(retail, "Net sales from returning customers by store in the last 2 months",
               {"measures": ["net_amount"], "group_by": ["store.name"], "segment": {**ORDERS, "status": "returning"},
                "time": {"window": LAST_2_MONTHS}})
    assert _by(got["export_rows"], "store") == pytest.approx({k: float(v) for k, v in want.items()})
    assert got["answer"]["headline"].startswith("Net amount from Apr 2026 to May 2026, for returning customers: ")


def test_new_against_returning(retail):
    want = dict(retail[0].con.execute(f"""
        WITH f AS ({FIRST}) SELECT CASE WHEN f.first >= DATE '2023-04-01' THEN 'New' ELSE 'Returning' END,
               SUM(l.net_amount) FROM ({LINES}) l JOIN f ON l.customer_id = f.k
        WHERE l.day >= DATE '2023-04-01' AND l.day < DATE '2023-07-01' GROUP BY 1""").fetchall())
    assert set(want) == {"New", "Returning"}
    got = _ask(retail, "Net sales: new vs returning customers in Q2 2023",
               {"measures": ["net_amount"], "segment": {**ORDERS, "status": "new", "split": True},
                "time": {"window": Q2_2023}})
    assert _by(got["export_rows"], "returning") == pytest.approx({k: float(v) for k, v in want.items()})
    assert got["answer"]["headline"] == "Net amount in Q2 2023: $29.5K from new customers (7.1%), $388.2K from " \
        "returning customers (93%)."


def test_new_customers_sales_month_by_month(retail):
    # In a series each row's own month decides: new in March when their first order was in March.
    want = dict(retail[0].con.execute(f"""
        WITH f AS ({FIRST}) SELECT date_trunc('month', l.day)::DATE, SUM(l.net_amount) FROM ({LINES}) l
        JOIN f ON l.customer_id = f.k
        WHERE l.day >= DATE '2023-01-01' AND l.day < DATE '2023-07-01'
          AND f.first >= date_trunc('month', l.day)::DATE GROUP BY 1""").fetchall())
    got = _ask(retail, "Net sales from new customers by month, first half of 2023",
               {"measures": ["net_amount"], "segment": {**ORDERS, "status": "new"},
                "time": {"grain": "month", "window": {"kind": "between", "start": "2023-01-01",
                                                      "end": "2023-06-30"}}})
    shown = {str(r["period"])[:10]: float(r["net_amount"]) for r in got["export_rows"]}
    assert shown == pytest.approx({str(k)[:10]: float(v) for k, v in want.items()})


@pytest.mark.parametrize("status, condition", [
    ("repeat", "n.days >= 2"),
    ("one_time", "n.days = 1"),
    ("won_back", "f.first < DATE '2026-04-01' AND NOT EXISTS (SELECT 1 FROM ({LINES}) b WHERE b.customer_id = l.customer_id "
                 "AND b.day >= DATE '2026-02-01' AND b.day < DATE '2026-04-01')"),
])
def test_a_segment_over_the_period(retail, status, condition):
    want = retail[0].con.execute(f"""
        WITH f AS ({FIRST}),
        n AS (SELECT customer_id AS k, COUNT(DISTINCT day) AS days FROM ({LINES})
              WHERE day >= DATE '2026-04-01' AND day < DATE '2026-06-01' GROUP BY customer_id)
        SELECT SUM(l.net_amount) FROM ({LINES}) l JOIN f ON l.customer_id = f.k JOIN n ON l.customer_id = n.k
        WHERE l.day >= DATE '2026-04-01' AND l.day < DATE '2026-06-01'
          AND {condition.replace('{LINES}', LINES)}""").fetchone()[0]
    got = _ask(retail, f"Net sales from {status.replace('_', '-')} customers in the last 2 months",
               {"measures": ["net_amount"], "segment": {**ORDERS, "status": status}, "time": {"window": LAST_2_MONTHS}})
    assert float(got["kpi"]["value"]) == pytest.approx(float(want))


@pytest.mark.parametrize("dialect", ["snowflake", "tsql", "oracle", "postgres"])
@pytest.mark.parametrize("segment, time", [
    ({**ORDERS, "status": "returning"}, {"window": LAST_2_MONTHS}),
    ({**ORDERS, "status": "new", "split": True}, {"window": LAST_2_MONTHS}),
    ({**ORDERS, "status": "new"}, {"grain": "month", "window": LAST_2_MONTHS}),
    ({**ORDERS, "status": "won_back"}, {"window": LAST_2_MONTHS}),
])
def test_the_sql_reads_on_every_warehouse(retail, dialect, segment, time):
    built, model = retail
    plan = Plan.model_validate({"kind": "query", "measures": ["net_amount"], "group_by": ["store.name"],
                                "segment": segment, "time": time})
    compiled = compile_query(resolve(plan, model, Context(today=TODAY)), model, dialect)
    sqlglot.parse_one(compiled.sql, read=dialect)


# ── the planner told, and put right in code ────────────────────────────────


def test_the_planner_is_shown_a_segment_in_the_workspaces_own_words(retail):
    from core2.plan.planner import stable_prompt

    prompt = stable_prompt(retail[1])
    assert "- segment: any figure for only some members" in prompt
    assert '"segment": {"of": "customer", "measure": "number_of_orders", "status": "returning"}' in prompt


@pytest.mark.parametrize("question, status, split", [
    ("Net sales from returning customers by store in the last 2 months", "returning", False),
    ("net sales from new customers by month", "new", False),
    ("What did repeat customers spend last quarter?", "repeat", False),
    ("net sales from one-time customers", "one_time", False),
    ("net sales of won-back customers in the last 2 months", "won_back", False),
    ("Net sales: new vs returning customers in Q2 2023", "new", True),
    ("net sales, returning and new customers, this year", "returning", True),
    ("ventes nettes des nouveaux clients par magasin", "new", False),
    ("ventes nettes des clients récurrents", "repeat", False),
])
def test_a_figure_said_of_some_members_gets_their_segment(retail, question, status, split):
    from core2.plan.planner import segment_asked

    got = segment_asked(Plan(kind="query", measures=["net_amount"]), retail[1], question)
    assert got.segment is not None and (got.segment.of, got.segment.status, got.segment.split) == \
        ("customer", status, split)
    assert got.segment.measure == "number_of_orders"


@pytest.mark.parametrize("question, plan", [
    ("net sales by store", {"measures": ["net_amount"]}),
    ("how many returning customers in the last 2 months", {"measures": ["number_of_customers"]}),
    ("new stores' net sales", {"measures": ["net_amount"]}),
    ("net sales for the Returning segment", {"measures": ["net_amount"],
                                             "filters": [{"field": "customer.segment", "op": "eq",
                                                          "values": ["Returning"]}]}),
])
def test_no_segment_where_none_is_said_or_a_field_holds_the_word(retail, question, plan):
    from core2.plan.planner import segment_asked

    before = Plan.model_validate({"kind": "query", **plan})
    assert segment_asked(before, retail[1], question) is before


# ── kept in a follow-up ────────────────────────────────────────────────────


def test_a_follow_up_keeps_the_segment_and_says_so(retail):
    from core2.plan.thread import kept_from_above

    model = retail[1]
    before = Plan.model_validate({"kind": "query", "measures": ["net_amount"], "group_by": ["store.name"],
                                  "segment": {**ORDERS, "status": "returning"}, "time": {"window": LAST_2_MONTHS}})
    after = Plan.model_validate({"kind": "query", "measures": ["net_amount"], "follow_up": "refine",
                                 "time": {"grain": "month", "window": LAST_2_MONTHS}})
    kept = kept_from_above(after, model, "and by month?", before, lambda p: True)
    assert kept.segment == before.segment
    # Let go when the reader says so, or asks something new.
    for question in ("overall, by month?", "same but without the filters"):
        assert kept_from_above(after, model, question, before, lambda p: True).segment is None
    new = after.model_copy(update={"follow_up": "new"})
    assert kept_from_above(new, model, "net sales by month", before, lambda p: True).segment is None


def test_a_two_turn_conversation_keeps_returning_customers(retail):
    built, model = retail
    answers = [json.dumps({"kind": "query", "measures": ["net_amount"], "group_by": ["store.name"],
                           "segment": {**ORDERS, "status": "returning"}, "time": {"window": LAST_2_MONTHS}}),
               json.dumps({"kind": "query", "measures": ["net_amount"], "follow_up": "refine",
                           "time": {"grain": "month", "window": LAST_2_MONTHS}})]
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=lambda s, t: answers.pop(0),
                        index=MemberIndex(), today=TODAY)
    session = Session()
    answer_question("Net sales from returning customers by store in the last 2 months", services, session)
    got = answer_question("and by month?", services, session)
    want = dict(built.con.execute(f"""
        WITH f AS ({FIRST}) SELECT date_trunc('month', l.day)::DATE, SUM(l.net_amount) FROM ({LINES}) l
        JOIN f ON l.customer_id = f.k WHERE l.day >= DATE '2026-04-01' AND l.day < DATE '2026-06-01'
          AND f.first < date_trunc('month', l.day)::DATE GROUP BY 1""").fetchall())
    shown = {str(r["period"])[:10]: float(r["net_amount"]) for r in got["export_rows"]}
    assert shown == pytest.approx({str(k)[:10]: float(v) for k, v in want.items()})
    assert "returning customers" in got["following"]["kept"]
