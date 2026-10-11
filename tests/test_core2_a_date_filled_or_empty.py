"""Rows with a date, or with none: "how many patients have died", "orders not yet shipped".

On a public health-records sample, "How many of them have died?" could not be planned: a filter on a date being
filled was refused ("a date filter with not_null is not supported"). A date filled or empty narrows the rows, and
the answer says so: "with a death date".

Synthetic data only.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")


@pytest.mark.parametrize("op, words", [("not_null", "with a"), ("is_null", "with no")])
def test_rows_with_a_date_or_none_are_counted(retail, op, words):
    built, model = retail
    role = next(r for r in model.date_roles.values() if not r.is_default and r.table == next(
        t.key for t in model.tables.values() if t.name == "order_lines"))
    column = model.columns[role.column].name
    plan = {"kind": "query", "intent": "value", "measures": ["number_of_order_lines"],
            "filters": [{"field": role.slug, "op": op}]}
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=lambda s, t: json.dumps(plan),
                        index=MemberIndex(), today=dt.date(2026, 6, 15))
    payload = answer_question("How many order lines have that date?", services, Session())
    where = "IS NOT NULL" if op == "not_null" else "IS NULL"
    want = built.con.execute(f'SELECT COUNT(*) FROM order_lines WHERE "{column}" {where}').fetchone()[0]
    assert payload["export_rows"][0]["number_of_order_lines"] == want, payload["answer"]["headline"]
    assert f"{words} {role.name.lower()}" in payload["answer"]["headline"], payload["answer"]["headline"]
