"""A unit code is never read as a size.

Items sold by the metre are counted in unit "M". The answer said "19,000 M",
read as 19 billion -- beside money written "$1.2M" in the same answers. A unit
code a reader takes for a size (K, M, B, T, G, MM, BN) is written "19,000
(unit M)"; any other code as before ("19,000 EA", "250 KG").
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.answer.builder import fmt
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn


@pytest.mark.parametrize("unit, said", [
    ("M", "19,000 (unit M)"), ("m", "19,000 (unit m)"), ("K", "19,000 (unit K)"), ("MM", "19,000 (unit MM)"),
    ("EA", "19,000 EA"), ("KG", "19,000 KG"), ("BOX", "19,000 BOX"), ("", "19,000"), (None, "19,000"),
])
def test_a_unit_code_is_written_so_it_reads_as_a_unit(unit, said):
    assert fmt(19000, "number", unit=unit) == said


def test_money_keeps_its_sizes():
    assert fmt(1_234_567, "currency") == "$1.23M"


class Recorded:
    def __init__(self, *answers: str):
        self.answers = list(answers)

    def __call__(self, stable: str, tail: str) -> str:
        return self.answers.pop(0)


def test_an_answer_in_metres_is_not_read_as_millions():
    built, model = learn(domains.build("retail"), "descriptive")
    # Most items are sold by the metre; the rest by the box or the kilogram (a mix the answer keeps apart).
    built.con.execute("UPDATE products SET unit_of_measure = 'M' WHERE unit_of_measure = 'EA'")
    plan = {"kind": "query", "intent": "rank", "measures": ["quantity"], "group_by": ["product"],
            "sort": [{"by": "quantity", "desc": True}], "limit": 5}
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=Recorded(json.dumps(plan)),
                        index=MemberIndex(), today=dt.date(2026, 6, 15))
    answer = answer_question("top 5 items by quantity sold", services, Session())
    top = built.con.execute("SELECT SUM(o.quantity) FROM order_lines o JOIN products p ON p.product_id = "
                            "o.product_id WHERE p.unit_of_measure = 'M' GROUP BY p.product_id ORDER BY 1 DESC "
                            "LIMIT 1").fetchone()[0]
    headline = answer["answer"]["headline"]
    assert fmt(top, "number", unit="M") in headline and "(unit M)" in headline, answer["answer"]
