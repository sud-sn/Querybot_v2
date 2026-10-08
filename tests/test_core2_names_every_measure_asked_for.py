"""A ranking of several measures names each of them for the one that leads.

"Revenue, cost of goods sold and gross profit by item group in 2025" was
headlined with the revenue of the leading group alone: the two other measures
asked for were in the table only. The leader's other values follow its first:
"X leads with $20.33M (23% of the total), and $12.61M cost of goods sold and
$7.73M gross profit, across 5 item groups".
"""

from __future__ import annotations

import datetime as dt
import json

from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn


class Recorded:
    def __init__(self, *answers: str):
        self.answers = list(answers)

    def __call__(self, stable: str, tail: str) -> str:
        return self.answers.pop(0)


def _ask(measures: list[str]) -> tuple[dict, list]:
    built, model = learn(domains.build("retail"), "descriptive")
    plan = {"kind": "query", "intent": "breakdown", "measures": measures, "group_by": ["store"],
            "time": {"window": {"kind": "between", "start": "2026-01-01", "end": "2026-03-31"}}}
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=Recorded(json.dumps(plan)),
                        index=MemberIndex(), today=dt.date(2026, 6, 15))
    answer = answer_question("sales, cost and quantity by store in Q1", services, Session())
    rows = answer["data"]["rows"]
    return answer, rows


def test_the_leader_is_given_every_measure_asked_for():
    answer, rows = _ask(["net_amount", "cost_amount", "quantity"])
    headline = answer["answer"]["headline"]
    leader = headline.split(": ", 1)[1].split(" leads with ")[0]
    row = next(r for r in rows if r["store_name"] == leader and f"${r['net_amount']:,.2f}" in headline)
    assert (f"leads with ${row['net_amount']:,.2f}, and ${row['cost_amount']:,.2f} cost amount and "
            f"{row['quantity']:,.0f} {row['product_unit_of_measure']} quantity, across 12 stores.") in headline, headline


def test_one_measure_reads_as_before():
    answer, _ = _ask(["net_amount"])
    headline = answer["answer"]["headline"]
    assert ", and " not in headline and headline.endswith("across 12 stores."), headline
