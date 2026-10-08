"""A ranking by a measure the question works out is ranked by it, and a sort that names nothing is never dropped.

"Which month in 2026 had the highest discount rate?" -- discounts over gross
sales, a ratio the plan works out for the question and names "Discount rate" --
answered with January: the sort named the ratio as the plan wrote it, the
output column is "discount_rate", the two never matched, and the sort was
dropped without a word. With nothing to order by, a series falls back to its
periods, so the first month was offered as the highest. The sort now finds the
ratio however its name is written, and one that names nothing is a problem the
planner's repair round sees, not a silent default.

And the answer says what was ranked: sorted lowest first ("the items with the
least gross profit"), it named the largest with "leads with"; and a breakdown
by weekday or by weekend counted them ("across 2 is weekends"). Synthetic
retail warehouse.
"""

from __future__ import annotations

import datetime as dt

import pytest

from core2.answer.builder import build_answer
from core2.compile.compiler import compile_query
from core2.plan.ir import Plan
from core2.plan.planner import check
from core2.resolve.resolver import Context, ResolveError, resolve
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
RATE = {"name": "Discount rate", "op": "ratio", "measures": ["discount_amount", "gross_amount"], "scale": 100}


@pytest.fixture(scope="module")
def retail():
    built, model = learn(domains.build("retail"), "descriptive")
    return model, DuckDBWarehouse(built.con)


def _rows(model, warehouse, plan: dict):
    logical = resolve(Plan.model_validate({"kind": "query", **plan}), model, Context(today=TODAY))
    result = warehouse.query(compile_query(logical, model, "duckdb").sql)
    return [dict(zip(result.columns, row)) for row in result.rows]


@pytest.mark.parametrize("named", ["Discount rate", "discount rate", "discount_rate", "Discount Rate"])
def test_the_month_with_the_highest_worked_out_rate_is_the_highest(retail, named):
    model, warehouse = retail
    year = {"grain": "month", "window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}
    every = _rows(model, warehouse, {"intent": "trend", "derived": [RATE], "time": year})
    highest = max(every, key=lambda r: r["discount_rate"])
    assert highest != every[0]                     # the first month is not the answer by accident
    (top,) = _rows(model, warehouse, {"intent": "rank", "derived": [RATE], "time": year,
                                      "sort": [{"by": named, "desc": True}], "limit": 1})
    assert top == highest


def test_a_sort_that_names_nothing_goes_back_to_the_planner(retail):
    model, _ = retail
    plan = Plan.model_validate({"kind": "query", "intent": "rank", "measures": ["net_amount"],
                                "time": {"grain": "month"}, "sort": [{"by": "margin of error", "desc": True}],
                                "limit": 1})
    with pytest.raises(ResolveError, match="nothing to sort by called margin of error") as caught:
        resolve(plan, model, Context(today=TODAY))
    assert "net_amount" in caught.value.options and "period" in caught.value.options
    problems, _ = check(plan, model, TODAY)
    assert problems and problems[0].startswith("nothing to sort by called margin of error.")


def _answer(model, warehouse, plan: dict) -> str:
    logical = resolve(Plan.model_validate({"kind": "query", **plan}), model, Context(today=TODAY))
    compiled = compile_query(logical, model, "duckdb")
    result = warehouse.query(compiled.sql)
    return build_answer("q", logical, compiled, result.columns, result.rows, duration_ms=1)["answer"]["headline"]


def test_ranked_lowest_first_the_answer_names_the_lowest(retail):
    model, warehouse = retail
    plan = {"intent": "rank", "measures": ["net_amount"], "group_by": ["store"],
            "sort": [{"by": "net_amount", "desc": False}], "limit": 3}
    rows = _rows(model, warehouse, plan)
    lowest = min(rows, key=lambda r: r["net_amount"])
    headline = _answer(model, warehouse, plan)
    assert f"{lowest['store_name']} is lowest of the 3 stores shown" in headline and "leads" not in headline
    top = _answer(model, warehouse, {**plan, "sort": [{"by": "net_amount", "desc": True}]})
    assert "leads with" in top and "lowest" not in top


def test_weekdays_and_weekends_are_not_counted(retail):
    model, warehouse = retail
    for group in ("time:day_of_week", "time:is_weekend"):
        headline = _answer(model, warehouse, {"intent": "breakdown", "measures": ["net_amount"], "group_by": [group]})
        assert "leads with" in headline and " across " not in headline, headline


def test_the_share_of_the_top_few_is_of_everything_not_of_the_few_shown(retail):
    """ "Top 10 customers and each one's share": the headline said 11% (of the ten) beside a column saying 3%."""
    model, warehouse = retail
    plan = {"intent": "share", "measures": ["net_amount"], "group_by": ["customer"],
            "sort": [{"by": "net_amount", "desc": True}], "limit": 3}
    rows = _rows(model, warehouse, plan)
    everything = sum(float(r[1]) for r in warehouse.query(
        'SELECT customer_name, SUM(net_amount) FROM order_lines o JOIN customers c USING (customer_id) GROUP BY 1').rows)
    assert abs(float(rows[0]["net_amount_share"]) - float(rows[0]["net_amount"]) / everything) < 1e-9
    headline = _answer(model, warehouse, plan)
    assert f"({float(rows[0]['net_amount_share']):.0%} of the total)" in headline, headline
    of_three = float(rows[0]["net_amount"]) / sum(float(r["net_amount"]) for r in rows)
    assert f"({of_three:.0%} of the total)" not in headline


def test_a_tie_is_not_a_lead(retail):
    """Five customer types of 12 customers each were "<the first type> leads with 12 (20% of the total)"."""
    from core2.answer.builder import build_answer as build

    model, warehouse = retail
    logical = resolve(Plan.model_validate({"kind": "query", "intent": "breakdown", "measures": ["net_amount"],
                                           "group_by": ["customer.segment"]}), model, Context(today=TODAY))
    compiled = compile_query(logical, model, "duckdb")
    names = [c.name for c in compiled.columns]
    group, measure = names[0], names[1]

    def headline(values):
        rows = [tuple(v if c == measure else n for c in names) for n, v in values]
        return build("q", logical, compiled, names, rows, duration_ms=1)["answer"]["headline"]

    assert headline([("Online", 12), ("Retail", 12), ("Wholesale", 12)]) == \
        "Net amount: each of the 3 segments has $12.00."
    two = headline([("Online", 12), ("Retail", 12), ("Wholesale", 6)])
    assert two.startswith("Net amount: Online and Retail lead with $12.00 each") and "leads" not in two
    assert "Wholesale leads with $12.00" in headline([("Online", 6), ("Retail", 6), ("Wholesale", 12)])
