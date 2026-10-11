"""A refusal is asked again when the question names what the plan needs.

Found asking a public sample warehouse through the chat with a real model (gpt-4o):

* "Show gross profit by month for 2013", after the answer that worked gross profit out (sales less cost), was
  refused: "gross profit is not available in the catalog". The figure the answer before made is made again;
* "Compare actual and budget for 2011" was refused: "no measure for budget". Actual and budget are members of
  one field (the scenario): they are compared by grouping by it.

Synthetic data only.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.values import build_index
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
Y2025 = {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}
PROFIT = {"name": "Gross profit", "op": "difference", "measures": ["net_amount", "cost_amount"]}
REFUSED = {"kind": "unsupported", "notes": ["not in the catalog"]}


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")


def _chat(retail, replies: list[dict]):
    built, model = retail

    def fetch(slug):
        column = model.columns[model.attributes[slug].column]
        table = model.tables[column.table]
        return [r[0] for r in built.con.execute(f'SELECT DISTINCT "{column.name}" FROM "{table.name}"').fetchall()]

    prompts: list[str] = []

    def complete(stable, tail):
        prompts.append(tail)
        return json.dumps(replies.pop(0))

    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=complete,
                        index=build_index(model, fetch), today=TODAY)
    session = Session()
    return (lambda q: answer_question(q, services, session)), prompts


def test_a_figure_the_answer_before_worked_out_is_worked_out_again(retail):
    first = {"kind": "query", "intent": "value", "derived": [PROFIT], "time": Y2025}
    again = {"kind": "query", "intent": "trend", "derived": [PROFIT], "follow_up": "refine",
             "time": {**Y2025, "grain": "month"}}
    ask, prompts = _chat(retail, [first, REFUSED, again])
    ask("What was our gross profit in 2025?")
    payload = ask("Show gross profit by month for 2025")
    assert "worked out" in prompts[-1] and '"Gross profit"' in prompts[-1], prompts[-1]
    assert len(payload["export_rows"]) == 12, payload["answer"]["headline"]


def test_two_members_of_one_field_are_compared_by_grouping_by_it(retail):
    grouped = {"kind": "query", "intent": "compare", "measures": ["net_amount"], "group_by": ["region.name"],
               "time": Y2025}
    ask, prompts = _chat(retail, [REFUSED, grouped])
    payload = ask("Compare North and South for 2025")
    assert "are members of the field region.name" in prompts[-1] and '"North"' in prompts[-1], prompts[-1]
    names = {r.get("region_name") for r in payload["export_rows"]}
    assert {"North", "South"} <= names, payload["export_rows"]


def test_a_refusal_naming_nothing_is_not_asked_again(retail):
    ask, prompts = _chat(retail, [REFUSED])
    payload = ask("What is the weather in Paris?")
    assert len(prompts) == 1 and not payload.get("export_rows")
