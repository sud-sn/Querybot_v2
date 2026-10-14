"""Measures over several tables keep to the relationships between them, as a governed data model does.

A data model's relationships say how tables connect; a measure over two tables (sales less refunds) adds up each
table on its own rows and combines the totals where they meet, so nothing is counted twice. Where the new core
strayed from that:

* a figure worked out for one question (a ratio of "sales after returns", a metric over two tables, to net sales)
  was decided by the first table each metric named: both "lived" on the order lines, so it was written out on those
  rows and refused ("each order line has several returns") where it should have been added up table by table;
* (the relationships page: tests/test_a_link_edit_is_measured_again.py) a link's conditions changed by an admin
  were never checked again against the data, and the page wrote "many to one" on any link it saved.

Every number is compared with SQL written by hand against the invented retail warehouse.
"""

from __future__ import annotations

import pytest

from core2.plan.ir import Plan
from core2.resolve.resolver import Context, ResolveError, resolve
from tests.test_core2_metrics_span_tables_and_links_keep_rows import (  # noqa: F401 - fixtures
    H1, IN_H1, TODAY, _add, _rows, _truth, learned, retail)


def test_a_figure_worked_out_from_a_metric_over_two_tables_adds_up_each_table(retail):
    model, warehouse = retail
    _add(model, "SUM([Order line · Net amount]) - SUM([Return · Refund amount])", slug="after_returns")
    kept = {"name": "Kept share", "op": "ratio", "measures": ["after_returns", "net_amount"], "scale": 100}
    rows = _rows(model, warehouse, {"intent": "breakdown", "derived": [kept], "group_by": ["store"], "time": H1})
    truth = _truth(warehouse, f"""
        with n as (select s.store_name, sum(o.net_amount) v from order_lines o join stores s using (store_id)
                   where o.order_date_key {IN_H1} group by 1),
             r as (select s.store_name, sum(r.refund_amount) v from returns r
                   join order_lines o on o.order_line_id = r.order_line_id join stores s on s.store_id = o.store_id
                   where r.return_date_key {IN_H1} group by 1)
        select n.store_name, (n.v - coalesce(r.v, 0)) * 100.0 / n.v from n left join r using (store_name)""")
    assert {r["store_name"]: float(r["kept_share"]) for r in rows} == pytest.approx(truth)


def test_a_figure_from_one_tables_metrics_stays_on_that_table(retail):
    model, _ = retail
    share = {"name": "Discount share", "op": "ratio", "measures": ["discount_amount", "gross_amount"], "scale": 100}
    if not {"discount_amount", "gross_amount"} <= {m.slug for m in model.measures.values()}:
        pytest.skip("the retail warehouse names its amounts otherwise")
    logical = resolve(Plan.model_validate({"kind": "query", "derived": [share], "time": H1}), model,
                      Context(today=TODAY))
    assert len(logical.parts) == 1, "one table: one part, no lining up"

