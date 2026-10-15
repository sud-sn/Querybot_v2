"""A figure no one saved is worked out for the question that asks it, read and checked like an admin's metric.

"The lowest batch yield" was answered with the average yield, sorted; "the smallest and largest quantity dispensed"
with the smallest alone; "average margin per fill, price charged minus ingredient cost" and "the average
ingredient cost of a fill this year against last" were refused. Each needs a figure no metric holds. The plan now
carries it as a formula in the admin's own formula language (core2/model/authoring.py): the planner writes it from
the names it is given, and code reads it, refuses a field the reader may not use, and counts it like any metric,
so it groups, compares periods, sorts and filters. The answer says it was worked out for this question, and how.

Each figure is checked against SQL written here on the invented pharmacy warehouse. Invented data only.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.ir import Plan
from core2.plan.values import MemberIndex
from core2.resolve.resolver import Context, ResolveError, resolve
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
Q1 = {"window": {"kind": "between", "start": "2026-01-01", "end": "2026-03-31"}}


@pytest.fixture(scope="module")
def pharmacy():
    return learn(domains.build("compounding_pharmacy"), "descriptive")


def _ask(learned, plan: dict, *, model=None) -> dict:
    built, learned_model = learned
    answers = [json.dumps({"kind": "query", **plan})]
    services = Services(model=model or learned_model, warehouse=DuckDBWarehouse(built.con),
                        complete=lambda s, t: answers.pop(0), index=MemberIndex(), today=TODAY)
    return answer_question("q", services, Session())


def _sql(learned, sql: str) -> list[tuple]:
    return learned[0].con.execute(sql).fetchall()


def _values(payload: dict) -> list[dict]:
    assert payload.get("type") == "assistant_response" and "export_rows" in payload, payload.get("answer")
    return payload["export_rows"]


def test_the_lowest_and_the_highest_of_a_field_are_its_smallest_and_largest_value(pharmacy):
    (low, high), = _sql(pharmacy, "SELECT MIN(yield_pct), MAX(yield_pct) FROM batches")
    payload = _ask(pharmacy, {"intent": "value", "figures": [
        {"name": "Lowest yield", "formula": "MIN([Batch · Yield percent])"},
        {"name": "Highest yield", "formula": "MAX([Batch · Yield percent])"}]})
    row, = _values(payload)
    assert sorted(float(v) for v in row.values()) == pytest.approx(sorted([float(low), float(high)]))


def test_arithmetic_on_each_row_is_averaged_by_month(pharmacy):
    expected = dict(_sql(pharmacy, "SELECT strftime(fill_date, '%Y-%m'), AVG(price_charged - ingredient_cost) "
                                   "FROM fills WHERE fill_date BETWEEN DATE '2026-01-01' AND DATE '2026-03-31' "
                                   "GROUP BY 1"))
    payload = _ask(pharmacy, {"intent": "trend", "figures": [
        {"name": "Margin per fill", "formula": "AVG([Fill · Price charged] - [Fill · Ingredient cost])",
         "format": "currency"}], "time": {**Q1, "grain": "month"}})
    rows = _values(payload)
    assert len(rows) == 3
    got = sorted(float(next(v for k, v in r.items() if "margin" in k)) for r in rows)
    assert got == pytest.approx(sorted(float(v) for v in expected.values()))


def test_a_figure_compares_two_periods(pharmacy):
    (now,), = _sql(pharmacy, "SELECT AVG(ingredient_cost) FROM fills "
                             "WHERE fill_date BETWEEN DATE '2026-01-01' AND DATE '2026-03-31'")
    (before,), = _sql(pharmacy, "SELECT AVG(ingredient_cost) FROM fills "
                                "WHERE fill_date BETWEEN DATE '2025-01-01' AND DATE '2025-03-31'")
    payload = _ask(pharmacy, {"intent": "compare", "figures": [
        {"name": "Ingredient cost per fill", "formula": "AVG([Fill · Ingredient cost])", "format": "currency"}],
        "time": {**Q1, "compare": {"kind": "same_period_last_year"}}})
    numbers = [float(v) for r in _values(payload) for v in r.values() if isinstance(v, (int, float))]
    assert any(n == pytest.approx(float(now)) for n in numbers)
    assert any(n == pytest.approx(float(before)) for n in numbers)


def test_a_figure_is_grouped_like_any_figure(pharmacy):
    expected = dict(_sql(pharmacy, "SELECT p.payer_name, AVG(c.paid_amount) FROM claims c "
                                   "JOIN payers p ON p.payer_id = c.payer_id GROUP BY 1"))
    payload = _ask(pharmacy, {"intent": "breakdown", "group_by": ["payer"], "figures": [
        {"name": "Paid per claim", "formula": "AVG([Claim · Paid amount])", "format": "currency"}]})
    rows = _values(payload)
    got = {next(v for k, v in r.items() if isinstance(v, str)): float(next(v for k, v in r.items() if "paid" in k))
           for r in rows}
    assert got == pytest.approx({k: float(v) for k, v in expected.items()})


def test_a_count_of_rows_meeting_a_condition(pharmacy):
    (n,), = _sql(pharmacy, "SELECT COUNT(*) FROM fills WHERE price_charged > 100")
    payload = _ask(pharmacy, {"intent": "value", "figures": [
        {"name": "Fills over 100", "formula": "COUNT(CASE WHEN [Fill · Price charged] > 100 THEN 1 END)",
         "format": "count"}]})
    row, = _values(payload)
    assert float(next(iter(row.values()))) == n


def test_the_answer_says_it_was_worked_out_for_this_question_and_how(pharmacy):
    payload = _ask(pharmacy, {"intent": "value", "figures": [
        {"name": "Lowest yield", "formula": "MIN([Batch · Yield percent])"}]})
    figure, = payload["figures"]
    assert figure["name"] == "Lowest yield" and "MIN" in figure["formula"] and "Yield percent" in figure["formula"]
    notes = " ".join(payload.get("trust", {}).get("date_context") or [])
    assert "worked out for this question" in notes


def test_a_figure_on_a_field_the_reader_may_not_use_is_refused(pharmacy):
    _, model = pharmacy
    plan = Plan.model_validate({"intent": "value", "figures": [
        {"name": "Paid per claim", "formula": "AVG([Claim · Paid amount])"}]})
    claims = next(t for t in model.tables if t.endswith(".claims"))
    with pytest.raises(ResolveError) as refused:
        resolve(plan, model, Context(today=TODAY, allowed_tables={t for t in model.tables if t != claims}))
    assert refused.value.kind == "denied"
    personal = model.model_copy(deep=True)
    personal.columns[f"{claims}.paid_amount"].sensitivity = "pii"
    with pytest.raises(ResolveError) as refused:
        resolve(plan, personal, Context(today=TODAY))
    assert refused.value.kind == "denied"


def test_a_figure_that_cannot_be_read_says_why(pharmacy):
    _, model = pharmacy
    plan = Plan.model_validate({"intent": "value", "figures": [
        {"name": "Lowest margin", "formula": "MIN([Batch · Margin])"}]})
    with pytest.raises(ResolveError) as refused:
        resolve(plan, model, Context(today=TODAY))
    assert refused.value.kind == "unknown" and "Margin" in str(refused.value)


def test_the_lowest_of_a_figure_with_nothing_to_rank_is_its_smallest_value(pharmacy):
    # Planned as the yield sorted with one row kept, and no grouping: one row was all there ever was, the average.
    (low,), = _sql(pharmacy, "SELECT MIN(yield_pct) FROM batches")
    (high,), = _sql(pharmacy, "SELECT MAX(yield_pct) FROM batches")
    lowest = _ask(pharmacy, {"intent": "value", "measures": ["yield_percent"],
                             "sort": [{"by": "yield_percent", "desc": False}], "limit": 1})
    row, = _values(lowest)
    assert float(next(iter(row.values()))) == pytest.approx(float(low))
    highest = _ask(pharmacy, {"intent": "value", "measures": ["yield_percent"],
                              "sort": [{"by": "yield_percent", "desc": True}], "limit": 1})
    row, = _values(highest)
    assert float(next(iter(row.values()))) == pytest.approx(float(high))


def test_the_planner_is_told_it_may_work_a_figure_out(pharmacy):
    from core2.plan.planner import stable_prompt

    _, model = pharmacy
    prompt = stable_prompt(model, values_allowed=True, personal_shown=False)
    assert "FIGURES WORKED OUT FOR THE QUESTION" in prompt and '"figures"' in prompt


def test_a_follow_up_that_names_the_figure_again_keeps_its_formula(pharmacy):
    # "The lowest yield", then "and by formula": the second plan names the figure, and its formula comes with it.
    built, model = pharmacy
    expected = dict(_sql(pharmacy, "SELECT f.formula_name, MIN(b.yield_pct) FROM batches b "
                                   "JOIN formulas f ON f.formula_id = b.formula_id GROUP BY 1"))
    answers = [json.dumps({"kind": "query", "intent": "value", "figures": [
                   {"name": "Lowest yield", "formula": "MIN([Batch · Yield percent])"}]}),
               json.dumps({"kind": "query", "intent": "breakdown", "measures": ["lowest_yield"],
                           "group_by": ["formula"], "follow_up": "refine"})]
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=lambda s, t: answers.pop(0),
                        index=MemberIndex(), today=TODAY)
    session = Session()
    answer_question("What is the lowest batch yield?", services, session)
    payload = answer_question("and by formula", services, session)
    rows = _values(payload)
    got = {next(v for v in r.values() if isinstance(v, str)): float(next(v for v in r.values()
                                                                         if isinstance(v, (int, float))))
           for r in rows}
    assert got == pytest.approx({k: float(v) for k, v in expected.items()})
    assert payload["figures"][0]["name"] == "Lowest yield"


def test_an_average_over_each_row_is_compared_over_two_periods(pharmacy):
    # Planned as each fill's cost averaged over the fills, against last year: refused as a figure over members,
    # it is the field's own average over each row, and is compared like any figure.
    (now,), = _sql(pharmacy, "SELECT AVG(ingredient_cost) FROM fills "
                             "WHERE fill_date BETWEEN DATE '2026-01-01' AND DATE '2026-03-31'")
    (before,), = _sql(pharmacy, "SELECT AVG(ingredient_cost) FROM fills "
                                "WHERE fill_date BETWEEN DATE '2025-01-01' AND DATE '2025-03-31'")
    payload = _ask(pharmacy, {"intent": "compare", "measures": ["ingredient_cost"],
                              "per": {"of": "number_of_fills", "summary": "avg"},
                              "time": {**Q1, "compare": {"kind": "same_period_last_year"}}})
    numbers = [float(v) for r in _values(payload) for v in r.values() if isinstance(v, (int, float))]
    assert any(n == pytest.approx(float(now)) for n in numbers)
    assert any(n == pytest.approx(float(before)) for n in numbers)


def test_a_figure_named_like_a_slug_is_shown_in_words(pharmacy):
    payload = _ask(pharmacy, {"intent": "value", "figures": [
        {"name": "average_margin_per_fill",
         "formula": "AVG([Fill · Price charged] - [Fill · Ingredient cost])"}]})
    assert payload["figures"][0]["name"] == "Average margin per fill"
    assert "average_margin_per_fill" not in json.dumps(payload.get("answer") or {})


def test_a_figure_leads_the_fields_it_is_made_of(pharmacy):
    payload = _ask(pharmacy, {"intent": "value", "measures": ["price_charged", "ingredient_cost", "margin_per_fill"],
                              "figures": [{"name": "Margin per fill", "formula":
                                           "AVG([Fill · Price charged] - [Fill · Ingredient cost])"}]})
    row, = _values(payload)
    assert list(row)[0].startswith("margin_per_fill")
    assert payload["answer"]["headline"].startswith("Margin per fill")


def _asked(learned, question: str, plan: dict, *, model=None) -> dict:
    built, learned_model = learned
    answers = [json.dumps({"kind": "query", **plan})]
    services = Services(model=model or learned_model, warehouse=DuckDBWarehouse(built.con),
                        complete=lambda s, t: answers.pop(0), index=MemberIndex(), today=TODAY)
    return answer_question(question, services, Session())


def test_an_average_of_a_figure_named_in_words_apart_averages_each_row(pharmacy):
    # "Average term in months of loans" was the terms added up: the figure's words (Loan term) were there, apart.
    (avg,), = _sql(pharmacy, "SELECT AVG(ingredient_cost) FROM fills "
                             "WHERE fill_date BETWEEN DATE '2026-01-01' AND DATE '2026-03-31'")
    payload = _asked(pharmacy, "What was the average cost of the ingredients on fills in Q1 2026?",
                     {"intent": "value", "measures": ["ingredient_cost"], "time": Q1})
    row, = _values(payload)
    assert float(next(v for v in row.values() if isinstance(v, (int, float)))) == pytest.approx(float(avg))


def test_a_series_word_inside_a_figures_name_asks_for_no_series(pharmacy):
    # "Average monthly payment on loans funded in 2025" came back month by month.
    _, model = pharmacy
    named = model.model_copy(deep=True)
    copay = next(m for m in named.measures.values() if m.slug == "copay_amount")
    copay.business_name = "Monthly copay"
    (total,), = _sql(pharmacy, "SELECT SUM(copay_amount) FROM fills "
                               "WHERE fill_date BETWEEN DATE '2026-01-01' AND DATE '2026-03-31'")
    payload = _asked(pharmacy, "What was the total monthly copay in Q1 2026?",
                     {"intent": "trend", "measures": ["copay_amount"], "time": {**Q1, "grain": "month"}}, model=named)
    row, = _values(payload)
    assert float(next(v for v in row.values() if isinstance(v, (int, float)))) == pytest.approx(float(total))
    # Asked by month, it is still a series.
    payload = _asked(pharmacy, "What was the monthly copay by month in Q1 2026?",
                     {"intent": "trend", "measures": ["copay_amount"], "time": {**Q1, "grain": "month"}}, model=named)
    assert len(_values(payload)) == 3


def test_the_fields_a_question_defines_its_figure_with_are_not_shown_beside_it(pharmacy):
    asked = "Average margin per fill, price charged minus ingredient cost"
    plan = {"intent": "value", "measures": ["price_charged", "ingredient_cost", "margin_per_fill"],
            "figures": [{"name": "Margin per fill",
                         "formula": "AVG([Fill · Price charged] - [Fill · Ingredient cost])"}]}
    row, = _values(_asked(pharmacy, asked, plan))
    assert list(row) == ["margin_per_fill"]
    # Asked for on their own as well, they stay.
    row, = _values(_asked(pharmacy, "Price charged, ingredient cost and the average margin per fill", plan))
    assert len(row) == 3
