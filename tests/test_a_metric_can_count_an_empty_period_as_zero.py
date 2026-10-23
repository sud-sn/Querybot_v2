"""A metric says whether a period with no rows is a gap or a zero.

A month with no rows is said to be a gap, not a zero: it may be data that never came in. For many metrics it is
a zero all the same: no orders that month is no sales. An admin says so per metric ("A period with no rows counts
as zero"), as dbt's metrics do with fill_nulls_with and a time spine:

    gap (as before)   the months with no rows are shown as a gap and named, "not as zero"
    zero              every month the series should hold has a row; one with no rows is 0, and the answer says
                      which months had none and that they count as zero; the total is the same
    a plain series    only a series of one figure by period: never one split by a grouping, and never where rows
                      were left out on purpose (a top N)
    adds up           only a figure that adds up (sales, a count); an average or a rate has no value for a month
                      with nothing in it, so it stays a gap

Invented data only (the retail domain: a customer with orders in 3 months of 2025); the AI is a recording.
"""

from __future__ import annotations

import json

import pytest

from core2.service import Session, answer_question
from tests.test_core2_answers_a_question_end_to_end import Recorded, _domain, _services

QUESTION = 'Net sales by month in 2025 for "Ironwood Supply"'
PLAN = {"intent": "trend", "measures": ["net_amount"],
        "time": {"grain": "month", "window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}},
        "filters": [{"field": "customer.name", "op": "eq", "values": ["Ironwood Supply"]}]}


def _model(empty: str = "gap", slug: str = "net_amount"):
    _, _, model, warehouse = _domain("retail")
    if empty != "gap":
        key = next(k for k, m in model.measures.items() if m.slug == slug)
        model = model.model_copy(update={"measures": {**model.measures,
                                                      key: model.measures[key].model_copy(update={"empty_periods": empty})}})
    return model, warehouse


def _ask(model, warehouse, plan=PLAN, question=QUESTION):
    return answer_question(question, _services(model, warehouse, Recorded(json.dumps(plan))), Session())


def test_by_default_a_month_with_no_rows_is_a_gap_and_said_to_be():
    model, warehouse = _model()
    payload = _ask(model, warehouse)
    assert payload["type"] == "assistant_response", payload
    assert payload["data"]["total_rows"] == 3
    assert "shown as a gap, not as zero" in json.dumps(payload["answer"])


def test_a_metric_that_counts_an_empty_month_as_zero_has_every_month():
    model, warehouse = _model("zero")
    gap = _ask(*_model())
    payload = _ask(model, warehouse)
    assert payload["type"] == "assistant_response", payload
    assert payload["data"]["total_rows"] == 12, "a month with no rows is missing"
    text = json.dumps(payload["answer"])
    assert "shown as a gap" not in text and "counted as zero" in text
    assert payload["answer"]["headline"] and payload["answer"].get("totals") == gap["answer"].get("totals")


def test_the_zeros_are_drawn_as_zeros_not_as_breaks_in_the_line():
    model, warehouse = _model("zero")
    payload = _ask(model, warehouse)
    chart = payload.get("chart") or {}
    rows = chart.get("rows") or []
    values = [r[y] for r in rows for y in chart.get("y_keys") or []]
    assert len(rows) == 12 and None not in values and values.count(0) == 9, values


def test_split_by_a_grouping_it_stays_a_gap():
    model, warehouse = _model("zero")
    plan = {**PLAN, "intent": "trend", "group_by": ["store"]}
    payload = _ask(model, warehouse, plan, 'Net sales by store by month in 2025 for "Ironwood Supply"')
    assert payload["type"] == "assistant_response", payload
    assert "counted as zero" not in json.dumps(payload["answer"])


def test_a_figure_that_does_not_add_up_stays_a_gap():
    _, _, base, _ = _domain("retail")
    average = next((m.slug for m in base.measures.values() if m.additivity != "additive" and m.table
                    and m.default_date), None)
    if average is None:
        pytest.skip("the retail domain has no figure that does not add up")
    model, warehouse = _model("zero", average)
    plan = {**PLAN, "measures": [average]}
    payload = _ask(model, warehouse, plan, f'{average} by month in 2025 for "Ironwood Supply"')
    assert "counted as zero" not in json.dumps(payload.get("answer") or {})


# ── The admin's setting ──────────────────────────────────────────────────────

def test_the_metric_editor_keeps_the_setting():
    from core2.model import authoring
    from core2.model.overrides import ALLOWED

    assert "empty_periods" in ALLOWED["measure"]
    model, _ = _model("zero")
    m = next(m for m in model.measures.values() if m.slug == "net_amount")
    assert authoring.editor_state(model, m)["empty_periods"] == "zero"
    assert authoring.editor_state(model, None)["empty_periods"] == "gap"


def test_the_editor_reads_it_from_the_form_and_refuses_it_for_a_figure_that_does_not_add_up():
    from admin.core2_metrics import _draft

    model, _ = _model()
    made = _draft(model, {"formula": "SUM([Order line · Net amount])", "name": "Net sales", "empty_periods": "zero"})
    assert made.empty_periods == "zero"
    assert _draft(model, {"formula": "SUM([Order line · Net amount])", "name": "Net sales"}).empty_periods == "gap"
    assert _draft(model, {"formula": "SUM([Order line · Net amount])", "name": "Net sales", "empty_periods": "nonsense"}) \
        .empty_periods == "gap"
    averaged = _draft(model, {"formula": "AVG([Order line · Net amount])", "name": "Average line", "empty_periods": "zero"})
    assert averaged.empty_periods == "gap", "an average has no value for a month with nothing in it"
