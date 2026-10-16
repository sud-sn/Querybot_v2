"""A dashboard recipe never builds a figure on a ratio, and uses the business's own approved one.

"Make a dashboard for returns and quality" showed a return rate of 453% (1,116% one month): the recipe took the
approved metric "Units per order" -- itself a ratio -- as the units sold, and divided the units returned by it. The
business had approved its own "Return rate" (refunds as a share of net sales), unused. A recipe's figures are built
on totals and counts, never on a ratio; and where the business has approved the figure a recipe would work out (its
return rate, gross margin %, average order value), that metric is the one shown.

Invented retail data only.
"""
from __future__ import annotations

import datetime as dt

import pytest

from core2 import recipe_book
from core2.model.schema import Measure
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")[1]


def _slug(model, slug):
    return next(m for m in model.measures.values() if m.slug == slug)


def _approved(key: str, slug: str, name: str, a: Measure, b: Measure, *, scale: float = 1.0, fmt: str = "number",
              kpi: str | None = None) -> Measure:
    return Measure.model_validate({
        "key": key, "slug": slug, "business_name": name, "table": a.table, "kind": "metric", "provenance": "admin",
        "status": "approved", "additivity": "non_additive", "format": fmt, "kpi": kpi,
        "expr": {"op": "ratio", "scale": scale, "args": [a.expr.model_dump(), b.expr.model_dump()]}})


def _with(model, *measures: Measure):
    out = model.model_copy(deep=True)
    for m in measures:
        out.measures[m.key] = m
    return out


def _rate_tiles(model):
    built = recipe_book.build(model, "returns_and_quality", today=TODAY)
    assert built.state == "ready", built.missing
    return [t for t in built.tiles if "eturn rate" in t.title]


def test_a_ratio_metric_is_never_the_units_a_return_rate_divides_by(retail):
    per_order = _approved("admin:units_per_order", "units_per_order", "Units per order", _slug(retail, "quantity"),
                          _slug(retail, "number_of_orders"), kpi="sales.units_per_order")
    tiles = _rate_tiles(_with(retail, per_order))
    assert tiles
    for t in tiles:
        used = {s for d in t.plan.get("derived", []) for s in d["measures"]} | set(t.plan.get("measures", []))
        assert "units_per_order" not in used, (t.title, used)


def test_the_approved_return_rate_is_the_one_shown(retail):
    rate = _approved("admin:return_rate", "return_rate", "Return rate", _slug(retail, "refund_amount"),
                     _slug(retail, "net_amount"), scale=100.0, fmt="percent", kpi="sales.return_rate")
    tiles = _rate_tiles(_with(retail, rate))
    assert tiles
    for t in tiles:
        assert t.plan.get("measures") == ["return_rate"] and not t.plan.get("derived"), (t.title, t.plan)
