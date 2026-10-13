"""Which way a figure is good news is read from its name, so a cost going up is never shown as good.

Every learned figure was "higher is better" until an admin said otherwise on its metric page: a dashboard's "Cost
amount ▲ 52%" was a green pill, and so was a rise in refunds on Home. Learn now reads it from the figure's name:
less is better for a cost, a loss, a refund or a return, churn, a cancellation, a defect, a delay or a complaint
(English and French), unless the name also says what is gained ("return on investment", "cost savings", "revenue
after returns"). A model learned before is read the same way when it is loaded; an admin's choice still wins.

Invented data only.
"""

from __future__ import annotations

import pytest

from core2.bootstrap import names
from core2.bootstrap.service import good_news
from core2.model.overrides import apply_overrides
from evals.core2 import domains
from evals.core2.compile_eval import learn


@pytest.mark.parametrize("name, way", [
    ("Refund Amount", "lower"), ("Cost Amount", "lower"), ("RETURN_QTY", "lower"), ("Number of complaints", "lower"),
    ("Late deliveries", "lower"), ("Sales return amount", "lower"), ("write_off_amt", "lower"), ("Churned MRR", "lower"),
    ("Coût total", "lower"), ("Montant des remboursements", "lower"),
    ("Net Amount", "higher"), ("Gross margin", "higher"), ("Number of customers", "higher"),
    ("Return on investment", "higher"), ("Cost savings", "higher"), ("Revenue after returns", "higher"),
    ("Discount amount", "higher")])
def test_a_figures_name_says_which_way_is_good_news(name, way):
    assert names.better(name) == way


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")


def test_learn_reads_it_from_each_figures_name(retail):
    _, model = retail
    by_slug = {m.slug: m for m in model.measures.values()}
    assert by_slug["cost_amount"].better == "lower" and by_slug["net_amount"].better == "higher"
    assert [m.better for m in model.measures.values() if "refund" in m.slug] == ["lower"] * len(
        [m for m in model.measures.values() if "refund" in m.slug])
    assert by_slug["number_of_orders"].better == "higher"


def test_a_model_learned_before_is_read_the_same_way_and_the_admin_still_decides(retail):
    _, learned = retail
    model = learned.model_copy(deep=True)
    cost = next(m for m in model.measures.values() if m.slug == "cost_amount")
    for m in model.measures.values():          # as stored before Learn read the names: every figure "higher"
        m.better = "higher"
    mine = cost.model_copy(update={"key": "metric:my_costs", "slug": "my_costs", "kind": "metric",
                                   "provenance": "admin", "business_name": "My costs"})
    model.measures[mine.key] = mine
    good_news(model)
    assert model.measures[cost.key].better == "lower"
    assert model.measures[mine.key].better == "higher", "an admin's own metric keeps what the admin set"
    assert apply_overrides(model, [{"object_key": f"measure:{cost.key}", "field": "better", "value": "higher"}]) == []
    assert model.measures[cost.key].better == "higher"


def test_a_cost_that_rose_is_bad_news_on_its_tile():
    from core2.answer.builder import against

    assert against(120.0, 100.0, "currency", "May", better=names.better("Cost Amount"))["good"] is False
    assert against(80.0, 100.0, "currency", "May", better=names.better("Refund Amount"))["good"] is True
