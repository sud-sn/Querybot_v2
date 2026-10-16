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


def test_asked_back_twice_the_figure_named_in_full_is_planned_in_code(model):
    # "how many orders did we tkae" was asked back -- number of orders or number of order lines? -- though the
    # question names one in full, and gpt-4o asked again when told so. The second time, code plans it.
    slug = next(m.slug for m in model.measures.values() if m.business_name.casefold() == "scrapped quantity")
    clarify = json.dumps({"kind": "clarify", "clarify": {"about": "measure", "question": "Which measure do you mean?",
                                                         "options": ["scrapped_quantity", "rejected_quantity"]}})
    answers = [clarify, clarify]
    outcome = plan_question(model, "Scrapped quantity in 2024", lambda s, u: answers.pop(0),
                            today=dt.date(2025, 3, 1))
    assert outcome.plan.kind == "query" and outcome.plan.measures == [slug], outcome.plan
    assert outcome.plan.time.window.kind == "between" and str(outcome.plan.time.window.start) == "2024-01-01"


@pytest.mark.parametrize("asked_twice", [False, True])
def test_a_question_back_between_figures_with_no_about_is_about_the_figure(model, asked_twice):
    # gpt-4o wrote its question back beside "kind", with no "about": "how many orders did we take: Number of orders,
    # Number of order lines, Quantity or Net amount?" (the retail condition book, three runs of three). Every option
    # is a figure of the data, so it is a question about the figure, and the one the question names is planned.
    slug = next(m.slug for m in model.measures.values() if m.business_name.casefold() == "scrapped quantity")
    names = [m.business_name for m in model.measures.values() if not m.hidden][:4]
    assert "Scrapped quantity" in names or any(n.casefold() == "scrapped quantity" for n in names), names
    back = json.dumps({"kind": "clarify", "question": "Scrapped quantity in 2024", "options": names})
    answers = [back, back] if asked_twice else [back, json.dumps({"kind": "query", "measures": [slug]})]
    outcome = plan_question(model, "Scrapped quantity in 2024", lambda s, u: answers.pop(0), today=dt.date(2025, 3, 1))
    assert outcome.plan.kind == "query" and outcome.plan.measures == [slug], outcome.plan


def test_a_question_back_about_something_else_is_still_asked(model):
    back = json.dumps({"kind": "clarify", "question": "Which plant do you mean?", "options": ["North", "South"]})
    answers = [back, back, back]
    outcome = plan_question(model, "Scrapped quantity at the plant in 2024", lambda s, u: answers.pop(0),
                            today=dt.date(2025, 3, 1))
    assert outcome.plan.kind == "clarify"
