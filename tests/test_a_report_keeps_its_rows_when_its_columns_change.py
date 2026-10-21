"""A report's follow-up that takes a column off or puts one on keeps the report's rows, in their order.

Found replaying a real model's (gpt-4o) plans for an analyst's thread: "Top 5 products by gross margin $ in 2025,
with category, net sales, margin, margin % and what share of the year's total margin each one makes up", then
"drop the share column and add units sold and how much discount we gave on each":

* "drop" was read as "fell", so the follow-up became a comparison with the year before;
* the plan the AI wrote for the follow-up had no order, so the top 5 became the 5 with the most net sales, and the
  gross margin they were ranked by left the answer.

A follow-up on the same rows that says no order of its own keeps the order above while the figure it ranks by is
still on the answer; "drop" stays a fall where it says one ("sales dropped", "a drop in sales").

Synthetic data only.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan import asks
from core2.plan.ir import Plan
from core2.plan.planner import Turn, plan_question
from evals.core2 import domains
from evals.core2.compile_eval import learn

YEAR = {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}
MARGIN = [{"name": "Gross margin $", "op": "difference", "measures": ["net_amount", "cost_amount"]},
          {"name": "Margin %", "op": "ratio", "measures": ["Gross margin $", "net_amount"], "scale": 100.0}]


@pytest.fixture(scope="module")
def model():
    return learn(domains.build("retail"), "descriptive")[1]


def _plan(**kw) -> Plan:
    return Plan.model_validate({"kind": "query", **kw})


def _after_top5(model, question: str, ai: Plan) -> Plan:
    """The follow-up's plan, the AI stood in by the plan it wrote, the top 5 by gross margin on screen."""
    written = json.dumps(ai.model_dump(mode="json", exclude_none=True))
    out = plan_question(model, question, lambda stable, tail: written, today=dt.date(2026, 6, 15),
                        history=[Turn("Top 5 products by gross margin $ in 2025", _top5_by_margin())],
                        reading="refine")
    return out.plan


def _top5_by_margin() -> Plan:
    return _plan(intent="share", measures=["net_amount"], derived=MARGIN, group_by=["product.name", "category.name"],
                 time=YEAR, sort=[{"by": "Gross margin $", "desc": True}], limit=5)


def test_dropping_a_column_is_no_fall():
    for command in ("drop the share column and add units sold and how much discount we gave on each",
                    "drop it and show the city", "show the city, and drop the margin %"):
        assert asks.ask_of(command) is None, command
    for command in ("please drop the cost column", "then drop the count", "now drop the share column"):
        assert asks.ask_of(command) is None, command
    for fall in ("did the number of orders also drop?", "did net sales just drop?", "did revenue also drop last month?",
                 "did margins also drop compared with last year?", "which stores saw orders rise and then drop this year?",
                 "sales dropped against last year", "which stores saw a drop in sales", "did sales drop?", "revenue drops",
                 "where did net sales drop the most this year, by store?", "did revenue drop this month?",
                 "which regions drop the furthest", "Drop in sales by region last quarter"):
        ask = asks.ask_of(fall)
        assert ask is not None and ask.moved, fall


def test_a_command_to_drop_something_is_no_fall_whatever_it_drops(model):
    # "Drop this column", "drop last year", "drop by store": each made the top 5 a comparison with the year before,
    # sorted by the change.
    ranking = _plan(intent="rank", measures=["net_amount"], group_by=["store.name"], time=YEAR,
                    sort=[{"by": "net_amount", "desc": True}], limit=5)
    for command in ("drop this column", "just drop this", "drop last year", "drop by store", "drop the most recent month",
                    "then drop to 5 rows", "drop the most expensive product",
                    # After a comma the command is its own clause, even after a question.
                    "which stores sold the most, and drop the share column", "how did sales do by store, and drop the margin"):
        assert asks.ask_of(command) is None, command
        kept = asks.settled(ranking, model, command)
        assert kept.intent == "rank" and kept.time.compare is None, command
    # The comparison asked beside the command stays, and is no fall.
    ask = asks.ask_of("Drop the margin and add units sold this year vs last year")
    assert ask is not None and ask.kind == "compare" and not ask.moved and not ask.falls
    # A question that goes on with "and drop" or "then drop" asks of a fall.
    for fall in ("which products did well last year and drop this year", "sales rose and then drop by 10%?",
                 "show me stores whose sales rose, then drop sharply", "drop by store vs last year",
                 "drop from last year by store", "then drop over the holidays", "then drop by ten percent",
                 "sales rose then drop to $2M", "products that sold well last year and drop this year"):
        ask = asks.ask_of(fall)
        assert ask is not None and ask.moved, fall


def test_a_column_taken_off_and_two_put_on_keep_the_same_rows_in_their_order(model):
    ai = _plan(intent="rank", measures=["net_amount", "quantity", "discount_amount"], derived=MARGIN,
               group_by=["product.name", "category.name"], time=YEAR, limit=5, follow_up="refine")
    settled = _after_top5(model, "drop the share column and add units sold and how much discount we gave on each",
                          ai)
    assert settled.time.compare is None and settled.intent != "compare"
    assert [(s.by, s.desc) for s in settled.sort] == [("Gross margin $", True)]
    assert settled.limit == 5 and settled.share_of is None and settled.intent != "share"


def test_a_follow_up_that_orders_anew_or_drops_the_ranking_figure_keeps_its_own(model):
    ordered = _plan(intent="rank", measures=["net_amount", "quantity"], derived=MARGIN,
                    group_by=["product.name", "category.name"], time=YEAR, limit=5, follow_up="refine")
    # The reader orders the rows anew: the AI's order (none here) is theirs to set, never the one above.
    assert all(s.by != "Gross margin $" for s in _after_top5(model, "add units sold, highest first", ordered).sort)
    # The figure the rows were ranked by is gone: its order goes with it.
    bare = _plan(intent="rank", measures=["quantity"], group_by=["product.name", "category.name"], time=YEAR,
                 limit=5, follow_up="refine")
    assert all(s.by != "Gross margin $" for s in _after_top5(model, "units sold instead", bare).sort)
    # Other rows (split anew): no order carried.
    other = _plan(intent="rank", measures=["net_amount"], derived=MARGIN, group_by=["store.name"], time=YEAR,
                  limit=5, follow_up="refine")
    assert all(s.by != "Gross margin $" for s in _after_top5(model, "and by store?", other).sort)


@pytest.mark.parametrize("question,measures,grain", [
    ("top 5 by number of orders, with net sales beside", ["number_of_orders", "net_amount"], None),
    ("now the top 5 by units sold instead, keep net sales", ["quantity", "net_amount"], None),
    ("and the bottom 5?", ["net_amount"], None),
    ("show them by month", ["net_amount"], "month"),
    ("the 5 best by units sold", ["quantity", "net_amount"], None),
    ("top five by units sold, keep net sales", ["quantity", "net_amount"], None),
    ("the worst performers by units sold", ["quantity", "net_amount"], None),
    ("by units sold rather than margin", ["quantity", "net_amount"], None),
    ("by units sold, instead of margin", ["quantity", "net_amount"], None),
    ("par unités vendues plutôt", ["quantity", "net_amount"], None),
    ("same 5 but by units sold instead of margin", ["quantity", "net_amount"], None)])
def test_a_follow_up_that_ranks_anew_is_never_given_the_order_above(model, question, measures, grain):
    time = {**YEAR, "grain": grain} if grain else YEAR
    ai = _plan(intent="trend" if grain else "rank", measures=measures, derived=MARGIN,
               group_by=["product.name", "category.name"], time=time, limit=None if grain else 5,
               follow_up="refine")
    assert all(s.by != "Gross margin $" for s in _after_top5(model, question, ai).sort), question


@pytest.mark.parametrize("question", [
    "drop the share column and show units sold instead", "show units sold instead of the share",
    "rather than the share, show units sold", "for the same top 5, drop the share column and add units sold",
    "add units sold for the top 5", "garde les mêmes 5 premiers et ajoute les unités vendues",
    # A figure swapped, not the rows ranked anew.
    "show the units sold by product rather than the share", "add units sold by product, instead of share",
    "montre les unités vendues par produit plutôt que la part"])
def test_a_column_swapped_on_the_same_rows_keeps_the_order_above(model, question):
    ai = _plan(intent="rank", measures=["net_amount", "quantity"], derived=MARGIN,
               group_by=["product.name", "category.name"], time=YEAR, limit=5, follow_up="refine")
    assert [(s.by, s.desc) for s in _after_top5(model, question, ai).sort] == [("Gross margin $", True)], question

