"""A figure the question names by its whole name is the one meant, not asked back.

Found by the sector stress test on a manufacturer's database (October 2026): a dashboard tile "Scrapped quantity
by scrap reason" was asked back -- "Which measure do you mean by 'Scrapped Quantity': scrapped_quantity or
rejected_quantity?" -- and left off the dashboard. The question names one figure by its own name. Names shared
by several figures (an order quantity of sales, of purchases and of work orders) are still asked.

Invented rows only.
"""

from __future__ import annotations

import datetime as dt
import json

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.plan.planner import plan_question
from core2.warehouse.runner import DuckDBWarehouse


@pytest.fixture(scope="module")
def model():
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE scrap_reasons AS SELECT range + 1 AS scrap_reason_id,
               ['Paint failure', 'Drill size wrong', 'Thermoform too thin', 'Seat assembly'][range + 1] AS reason_name
        FROM range(4);
        CREATE TABLE work_orders AS SELECT range + 1 AS work_order_id, DATE '2024-01-01' + (range % 600)::INTEGER AS start_date,
               20 + range % 80 AS order_quantity, range % 7 AS scrapped_quantity, range % 5 AS rejected_quantity,
               CASE WHEN range % 7 > 0 THEN range % 4 + 1 END AS scrap_reason_id FROM range(5000);
    """)
    warehouse = DuckDBWarehouse(con)
    return build_model(warehouse, from_duckdb(warehouse), client_id="t-scrap", options=BuildOptions(workers=1))


def _planner(model, first: dict):
    slug = next(m.slug for m in model.measures.values() if m.business_name.casefold() == "scrapped quantity")
    answers = [json.dumps(first), json.dumps({"kind": "query", "measures": [slug]})]
    asked: list[str] = []

    def complete(system, user):
        asked.append(user)
        return answers.pop(0)
    return complete, asked


def test_a_figure_named_in_full_is_planned(model):
    clarify = {"kind": "clarify", "clarify": {"about": "measure", "question": "Which measure do you mean?",
                                              "options": ["scrapped_quantity", "rejected_quantity"]}}
    complete, asked = _planner(model, clarify)
    outcome = plan_question(model, "Scrapped quantity by scrap reason in 2024", complete, today=dt.date(2025, 3, 1))
    assert outcome.plan.kind == "query", outcome.plan
    assert len(asked) == 2


def test_a_question_naming_no_figure_in_full_is_still_asked(model):
    clarify = {"kind": "clarify", "clarify": {"about": "measure", "question": "Which quantity do you mean?",
                                              "options": ["scrapped_quantity", "rejected_quantity"]}}
    complete, _ = _planner(model, clarify)
    outcome = plan_question(model, "quantity lost by reason in 2024", complete, today=dt.date(2025, 3, 1))
    assert outcome.plan.kind == "clarify"
