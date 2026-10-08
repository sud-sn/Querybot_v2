"""The new core never answers for members it was not told the names of, and follows "those" into another period.

"Who were our top 10 customers in the first half of 2026?", then "how did those
same customers do in the first half of 2025?", was answered "No rows match from
Jan 2025 to Jun 2025". The AI sees the question and the plan behind the answer on
screen, not its rows: it filtered on customers it could only guess at, nothing
matched, and the answer read as if those customers had bought nothing.

* The planner is told what "those" and "the same customers" are, and how to
  follow them into another period: the same ranking, compared with the other
  period -- which keeps the same members, where a new ranking would not.
* When nothing matches and a member the plan filters on is not one the data
  had, that is said ("there is no customer called …") instead of "No rows
  match". The query still runs: the index of members is read once, and a
  customer added since is real. One written in another case ("retail") is the
  stored one ("RETAIL"), since the warehouse compares text exactly.

Synthetic retail warehouse, the AI's answers recorded.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.planner import RULES
from core2.plan.values import build_index
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
H1_26 = {"kind": "between", "start": "2026-01-01", "end": "2026-06-30"}
H1_25 = {"kind": "between", "start": "2025-01-01", "end": "2025-06-30"}
TOP = {"kind": "query", "intent": "rank", "measures": ["net_amount"], "group_by": ["customer"],
       "time": {"window": H1_26}, "sort": [{"by": "net_amount", "desc": True}], "limit": 3}


@pytest.fixture(scope="module")
def retail():
    built, model = learn(domains.build("retail"), "descriptive")
    warehouse = DuckDBWarehouse(built.con)

    def fetch(slug):
        column = model.columns[model.attributes[slug].column]
        return [r[0] for r in warehouse.query(f'SELECT DISTINCT "{column.name}" FROM '
                                              f'{model.tables[column.table].name}').rows]

    return model, warehouse, build_index(model, fetch)


class Recorded:
    def __init__(self, *plans: dict):
        self.plans = [json.dumps(p) for p in plans]

    def __call__(self, stable, tail):
        return self.plans.pop(0)


class Watched(DuckDBWarehouse):
    def __init__(self, inner):
        self.inner, self.ran = inner, []

    dialect = property(lambda self: self.inner.dialect)
    db_type = property(lambda self: self.inner.db_type)

    def query(self, sql, *, max_rows=None):
        self.ran.append(sql)
        return self.inner.query(sql, max_rows=max_rows)


def _conversation(retail, *turns):
    model, warehouse, index = retail
    watched = Watched(warehouse)
    services = Services(model=model, warehouse=watched, complete=Recorded(*(p for _, p in turns)), index=index,
                        today=TODAY)
    session = Session()
    return [answer_question(q, services, session) for q, _ in turns], watched


def _customers(frame) -> list[str]:
    return [r["customer_name"] for r in frame["data"]["rows"]]


def test_the_planner_is_told_what_those_are_and_how_to_follow_them():
    rules = " ".join(RULES.split())
    assert "never filter on a name that is not in VALUE MATCHES or a previous plan" in rules
    assert "keep the previous plan's group_by, sort, limit and time.window, and put the other period in " \
           "time.compare" in rules


def test_followed_into_another_period_they_are_the_same_customers(retail):
    same = {**TOP, "intent": "compare", "follow_up": "refine",
            "time": {"window": H1_26, "compare": {"kind": "window", "window": H1_25}}}
    (first, then), _ = _conversation(retail, ("top 3 customers in H1 2026", TOP),
                                     ("how did those same customers do in H1 2025?", same))
    assert _customers(then) == _customers(first)
    assert {"net_amount", "net_amount_prior"} <= set(then["data"]["rows"][0])
    # A new ranking of H1 2025 would be other customers: that is why the comparison keeps them.
    (h1_25,), _ = _conversation(retail, ("top 3 customers in H1 2025", {**TOP, "time": {"window": H1_25}}))
    assert _customers(h1_25) != _customers(first)


def test_a_customer_the_data_does_not_have_is_said_not_no_rows(retail):
    guessed = {**TOP, "follow_up": "refine", "time": {"window": H1_25},
               "filters": [{"field": "customer", "op": "in", "values": ["Top Customer A", "Top Customer B"]}]}
    (_, then), watched = _conversation(retail, ("top 3 customers in H1 2026", TOP),
                                       ("how did those same customers do in H1 2025?", guessed))
    assert then["answer"]["headline"].startswith(
        'I could not find that in this data: there is no customer called "Top Customer A". If you mean the '
        "customers in the answer above, ask to compare that answer with the other period")
    assert then["data"] is None and "No rows match" not in json.dumps(then)
    assert then["trust"]["stopped"] == 'No customer is called "Top Customer A", and nothing matched.'
    assert "Top Customer A" in then["trust"]["sql"]                 # the query ran, and it is shown


def test_a_customer_added_after_the_index_was_read_is_answered(retail):
    """The index is read once per model version: a member it does not know may simply be new."""
    model, warehouse, index = retail
    from core2.plan.values import MemberIndex

    stale = MemberIndex()
    for key, entries in index.names.items():
        for attribute, value in entries:
            stale.add(attribute, [] if attribute == "customer.name" else [value])
    stale.attributes.add("customer.name")                    # read, but before every customer existed
    (first,), _ = _conversation(retail, ("top 3 customers in H1 2026", TOP))
    one = _customers(first)[0]
    plan = {"kind": "query", "intent": "value", "measures": ["net_amount"], "time": {"window": H1_26},
            "filters": [{"field": "customer", "op": "eq", "values": [one]}]}
    services = Services(model=model, warehouse=warehouse, complete=Recorded(plan), index=stale, today=TODAY)
    frame = answer_question(f"net sales for {one} in H1 2026", services, Session())
    assert frame["data"]["rows"] and "could not find" not in frame["answer"]["headline"]


def test_a_member_written_in_another_case_is_the_stored_one(retail):
    model, warehouse, index = retail
    stored = index.stored("customer.segment", "wholesale")
    assert stored == "Wholesale"
    plan = {"kind": "query", "intent": "value", "measures": ["net_amount"],
            "filters": [{"field": "customer.segment", "op": "eq", "values": ["wholesale"]}]}
    (frame,), watched = _conversation(retail, ("net sales for wholesale customers", plan))
    assert frame["data"]["rows"] and "'Wholesale'" in watched.ran[-1] and "'wholesale'" not in watched.ran[-1]
    # Excluding a member there is none of changes nothing, and is not refused.
    (kept,), _ = _conversation(retail, ("net sales except the mail order segment", {
        **plan, "filters": [{"field": "customer.segment", "op": "ne", "values": ["Mail order"]}]}))
    assert kept["data"]["rows"]


def test_an_empty_answer_is_not_blamed_on_a_member_only_excluded(retail):
    """Nothing in 2030; excluding a segment there is none of did not cause it, and is not named as the cause."""
    plan = {"kind": "query", "intent": "value", "measures": ["net_amount"],
            "time": {"window": {"kind": "between", "start": "2030-01-01", "end": "2030-12-31"}},
            "filters": [{"field": "customer.segment", "op": "ne", "values": ["Mail order"]}]}
    (frame,), _ = _conversation(retail, ("net sales in 2030 except the mail order segment", plan))
    assert "could not find" not in frame["answer"]["headline"] and "Mail order" not in frame["answer"]["headline"]


def test_a_total_for_a_customer_the_data_does_not_have_is_said_not_no_value(retail):
    plan = {"kind": "query", "intent": "value", "measures": ["net_amount"], "time": {"window": H1_26},
            "filters": [{"field": "customer", "op": "eq", "values": ["Top Customer A"]}]}
    (frame,), _ = _conversation(retail, ("net sales for Top Customer A in H1 2026", plan))
    assert frame["answer"]["headline"] == 'I could not find that in this data: there is no customer called "Top Customer A".'
