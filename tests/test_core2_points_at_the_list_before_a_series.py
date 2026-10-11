""""And for the third one?" after one member's series points at the list the series was drawn from.

On a public sample of an operational database, "Top 3 territories by order subtotal in 2025", then "Monthly
subtotal for the first one in 2025", then "And for the third one?" showed the second territory: the monthly answer
lists no territories to point at, and the AI guessed. A pointing phrase that finds no list on screen points at the
latest answer that showed one.

Synthetic data only.
"""

from __future__ import annotations

import datetime as dt
import json
import re

import pytest

from core2.plan.values import build_index
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

Y = {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")


def test_the_third_one_after_a_series_is_the_third_of_the_list_before(retail):
    built, model = retail

    def fetch(slug):
        column = model.columns[model.attributes[slug].column]
        table = model.tables[column.table]
        return [r[0] for r in built.con.execute(f'SELECT DISTINCT "{column.name}" FROM "{table.name}"').fetchall()]

    top = {"kind": "query", "intent": "rank", "measures": ["net_amount"], "group_by": ["store.name"],
           "sort": [{"by": "net_amount", "desc": True}], "limit": 3, "time": Y}

    def complete(stable, tail):
        asked = tail.split("QUESTION:")[-1]
        if "Top 3" in asked:
            return json.dumps(top)
        named = re.findall(r'-> store\.name = "([^"]+)"', tail)
        return json.dumps({"kind": "query", "intent": "trend", "measures": ["net_amount"], "follow_up": "refine",
                           "filters": [{"field": "store.name", "op": "eq", "values": named[-1:] or ["?"]}],
                           "time": {**Y, "grain": "month"}})

    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=complete,
                        index=build_index(model, fetch), today=dt.date(2026, 6, 15))
    session = Session()
    three = [r["store_name"] for r in answer_question("Top 3 stores by net amount in 2025", services, session)
             ["export_rows"]]
    answer_question("Monthly net amount for the first one in 2025", services, session)
    payload = answer_question("And for the third one?", services, session)
    assert three[2] in payload["answer"]["headline"], payload["answer"]["headline"]
