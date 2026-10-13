""""So far" is everything up to now, and interest accrued on a month-end snapshot is added up over the months.

Found by the sector stress test on a retail bank's database (October 2026):

* "How many loans have we charged off so far, and for how much?" was planned for this year alone (1 January to the
  data's last day), and missed the charge-offs of the year before. With no period named, "so far", "to date" and
  "ever" are everything up to now. "So far this year" is still this year.
* The interest accrued on each account's month-end row was taken at the end of each period, as a balance is,
  so the interest of a year was the interest of its December. Interest accrued in a month is an amount for that
  month, added up; the balance beside it is still taken at period end.

Invented rows only.
"""

from __future__ import annotations

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, learn
from core2.bootstrap.inventory import from_duckdb
from core2.plan.ir import Plan, TimeSpec, Window
from core2.plan.planner import all_time_asked
from core2.warehouse.runner import DuckDBWarehouse

THIS_YEAR = Plan(kind="query", measures=["loan_count"], time=TimeSpec(window=Window(kind="this", unit="year")))


@pytest.mark.parametrize("question", [
    "How many loans have we charged off so far, and for how much?",
    "total charge-offs to date",
    "have we ever charged off a mortgage?",
    "Combien de prêts avons-nous passés en perte jusqu'à présent ?",
])
def test_so_far_with_no_period_is_everything(question):
    assert all_time_asked(THIS_YEAR, None, question).time.window.kind == "all"


@pytest.mark.parametrize("question", [
    "How many loans have we charged off so far this year?",
    "charge-offs to date in 2025",
    "how many loans this year",
    "charge-offs since March",
])
def test_a_period_named_beside_it_stands(question):
    assert all_time_asked(THIS_YEAR, None, question) is THIS_YEAR


def test_interest_accrued_on_a_month_end_snapshot_is_added_up():
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE accounts AS SELECT range + 1 AS account_id, 'Account ' || range AS account_name,
               ['Savings', 'Checking'][range % 2 + 1] AS product FROM range(40);
        CREATE TABLE account_balances AS
        SELECT a.range + 1 AS account_id, (DATE '2024-01-31' + INTERVAL (m.range) MONTH)::DATE AS balance_date,
               round(1000 + a.range * 50 + m.range * 10 + 0.5, 2) AS ending_balance,
               round(3 + a.range * 0.1 + m.range * 0.01, 2) AS interest_accrued
        FROM range(40) a, range(24) m;
    """)
    warehouse = DuckDBWarehouse(con)
    inventory = from_duckdb(warehouse)
    found = learn(warehouse, inventory, BuildOptions(workers=1))
    by_column = {m.column: m for m in found.measures if m.column}
    assert by_column["interest_accrued"].additivity == "additive", by_column["interest_accrued"].evidence
    assert by_column["ending_balance"].additivity == "semi_additive"


# ── a question that names no figure ─────────────────────────────────────────


@pytest.fixture(scope="module")
def retail():
    from evals.core2 import domains
    from evals.core2.compile_eval import learn as learn_domain

    return learn_domain(domains.build("retail"), "descriptive")[1]


@pytest.mark.parametrize("why", ["The question is too vague to determine a specific measure or intent.",
                                 "The request is ambiguous: it does not specify what to measure."])
def test_a_question_too_vague_to_plan_is_asked_back_with_the_main_figures(retail, why):
    import datetime as dt
    import json

    from core2.plan.planner import plan_question

    outcome = plan_question(retail, "how are we doing?", lambda system, user: json.dumps(
        {"kind": "unsupported", "notes": [why]}), today=dt.date(2026, 6, 15))
    assert outcome.plan.kind == "clarify" and outcome.plan.clarify.about == "other", outcome.plan
    assert len(outcome.plan.clarify.options) >= 2


def test_a_question_nothing_in_the_data_answers_is_still_refused(retail):
    import datetime as dt
    import json

    from core2.plan.planner import plan_question

    outcome = plan_question(retail, "what is our net promoter score?", lambda system, user: json.dumps(
        {"kind": "unsupported", "notes": ["The catalog has no measure for net promoter score."]}),
        today=dt.date(2026, 6, 15))
    assert outcome.plan.kind == "unsupported"
