"""A figure is never worked out from itself.

"% increase in average unit price" was planned as (price - price) / price and ranked every item at 0%, in
alphabetical order; "the total as ARR" as MRR x MRR x 12 and shown as 155 billion. A figure taken from or
divided by itself is refused, with what to ask instead (a comparison of two periods), and a figure times
itself with a scale is read as the figure times that number.

The synthetic retail warehouse (invented data); figures are checked against hand-written SQL.
"""

from __future__ import annotations

import datetime as dt

import pytest

from core2.plan.ir import Plan
from core2.plan.planner import check
from core2.resolve.resolver import Context, ResolveError, resolve
from evals.core2 import domains
from evals.core2.compile_eval import learn

YEAR = {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}


@pytest.fixture(scope="module")
def model():
    return learn(domains.build("retail"), "descriptive")[1]


def _plan(**kw) -> Plan:
    return Plan.model_validate({"kind": "query", "intent": "value", "time": YEAR, **kw})


@pytest.mark.parametrize("op", ["difference", "ratio", "sum"])
def test_a_figure_with_itself_is_refused_with_what_to_ask(model, op):
    plan = _plan(measures=[], derived=[{"name": "Change", "op": op, "measures": ["net_amount", "net_amount"],
                                        "scale": 100 if op == "ratio" else 1}])
    with pytest.raises(ResolveError) as raised:
        resolve(plan, model, Context(today=dt.date(2026, 6, 1)))
    assert "comparison of the two periods" in raised.value.message
    problems, _ = check(plan, model, dt.date(2026, 6, 1))
    assert problems and "pct_change" in problems[0]


def test_a_figure_times_itself_with_a_scale_is_the_figure_times_that_number(model):
    plan = _plan(measures=[], derived=[{"name": "Annual run rate", "op": "product",
                                        "measures": ["net_amount", "net_amount"], "scale": 12}])
    logical = resolve(plan, model, Context(today=dt.date(2026, 6, 1)))
    assert any("Annual run rate = Net amount × 12" in n for n in logical.notes)


def test_a_figure_compared_with_another_figure_is_never_a_value(model):
    """"Stores whose sales were below target" was filtered as net_amount < 'target_amount': the database compared
    the sales with that text and refused. The planner is told how such a comparison is written instead."""
    plan = _plan(measures=["net_amount"], group_by=["store"],
                 filters=[{"field": "net_amount", "op": "lt", "values": ["gross_amount"]}])
    problems, _ = check(plan, model, dt.date(2026, 6, 1))
    assert problems and "another figure" in problems[0] and "derived difference" in problems[0]
