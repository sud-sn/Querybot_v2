"""A metric made of other metrics counts each of them on its own rows, and is still there after a reload.

An admin wrote net charge-offs as charge-offs less recoveries: two metrics of one table, each with its own
condition (the kind of event). The formula was written out with those conditions left behind, so both sides added
up every row and the editor's check showed 0 for every month; a generic dispensing rate (generic fills over all
fills) showed 100%. A metric's conditions now go with it wherever a formula names it.

The admin's metrics then came back from the store in the order of their names, and one that names another saved
after it ("Net charge-offs" before "Recoveries") was dropped as reading what is not in the data. They are added in
the order they need each other.

Every number is compared with SQL written by hand against the invented retail warehouse.
"""

from __future__ import annotations

import datetime as dt

import pytest

from core2.compile.compiler import compile_query
from core2.model import authoring
from core2.model.overrides import apply_overrides, target
from core2.model.schema import ColumnFilter, Measure
from core2.plan.ir import Plan
from core2.resolve.resolver import Context, resolve
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn
from tests.test_readers_suggest_and_admins_decide import retail, site  # noqa: F401 - the fixtures

TODAY = dt.date(2026, 6, 15)
H1 = {"window": {"kind": "between", "start": "2026-01-01", "end": "2026-06-30"}}
IN_H1 = "order_date_key between 20260101 and 20260630"


@pytest.fixture(scope="module")
def learned():
    built, model = learn(domains.build("retail"), "descriptive")
    return model, DuckDBWarehouse(built.con)


@pytest.fixture
def shop(learned):
    model, warehouse = learned
    model = model.model_copy(deep=True)
    status = _column(model, "status_code")
    for slug, code in [("delivered_sales", "D"), ("open_sales", "O")]:
        _add(model, "SUM([Order line · Net amount])", slug, [ColumnFilter(column=status, op="eq", values=[code])])
    _add(model, "COUNT([Order line])", "delivered_lines", [ColumnFilter(column=status, op="eq", values=["D"])])
    return model, warehouse


def _column(model, name: str) -> str:
    return next(c.key for c in model.columns.values() if c.table.endswith(".order_lines") and c.name == name)


def _add(model, text: str, slug: str, filters=()) -> Measure:
    read = authoring.read(model, text)
    m = Measure(key=slug, slug=slug, business_name=slug.replace("_", " ").capitalize(), table=read.table,
                expr=read.expr, filters=list(filters), format="number", status="approved")
    model.measures[slug] = m
    return m


def _rows(model, warehouse, plan: dict) -> list[dict]:
    logical = resolve(Plan.model_validate({"kind": "query", **plan}), model, Context(today=TODAY))
    for dialect in ("snowflake", "tsql", "oracle", "postgres"):
        compile_query(logical, model, dialect)
    result = warehouse.query(compile_query(logical, model, "duckdb").sql)
    return [dict(zip(result.columns, row)) for row in result.rows]


def test_one_metric_less_another_of_the_same_rows_counts_each_on_its_own_rows(shop):
    model, warehouse = shop
    _add(model, "[Delivered sales] - [Open sales]", "delivered_less_open")
    (row,) = _rows(model, warehouse, {"intent": "value", "measures": ["delivered_less_open"], "time": H1})
    (truth,) = warehouse.query(f"""select sum(net_amount) filter (where status_code = 'D')
        - sum(net_amount) filter (where status_code = 'O') from order_lines where {IN_H1}""").rows[0]
    assert float(truth) != 0
    assert float(row["delivered_less_open"]) == pytest.approx(float(truth)), "each side added up every row: 0"


def test_a_share_of_the_rows_one_metric_counts_is_not_all_of_them(shop):
    model, warehouse = shop
    _add(model, "[Delivered lines] / COUNT([Order line]) * 100", "delivered_share")
    rows = _rows(model, warehouse, {"intent": "breakdown", "measures": ["delivered_share"], "group_by": ["store"],
                                    "time": H1})
    truth = {r[0]: float(r[1]) for r in warehouse.query(f"""select s.store_name,
        count(*) filter (where o.status_code = 'D') * 100.0 / count(*) from order_lines o join stores s using (store_id)
        where o.{IN_H1} group by 1""").rows}
    assert {r["store_name"]: float(r["delivered_share"]) for r in rows} == pytest.approx(truth)
    assert all(v < 100 for v in truth.values()), "the share was once 100% for every store"


def test_the_editors_check_shows_what_a_question_will_get(shop):
    model, warehouse = shop
    read = authoring.read(model, "[Delivered sales] - [Open sales]")
    draft = Measure(key="d", slug="d", business_name="Delivered less open", table=read.table, expr=read.expr)
    checked = authoring.check(model, draft, warehouse, TODAY)
    assert checked["problem"] == ""
    (june,) = warehouse.query("""select sum(net_amount) filter (where status_code = 'D')
        - sum(net_amount) filter (where status_code = 'O') from order_lines
        where order_date_key between 20260601 and 20260630""").rows[0]
    assert checked["months"][-1]["value"] == pytest.approx(float(june))


def test_a_metric_named_before_the_one_it_is_made_of_is_kept(learned):
    model, _ = learned
    model = model.model_copy(deep=True)
    status = _column(model, "status_code")
    read = authoring.read(model, "SUM([Order line · Net amount])")
    sales = Measure(key="admin:shipped_sales", slug="shipped_sales", business_name="Shipped sales", table=read.table,
                    expr=read.expr, filters=[ColumnFilter(column=status, op="eq", values=["S"])])
    model.measures[sales.key] = sales
    net = authoring.read(model, "[Shipped sales] / [Net amount] * 100")
    share = Measure(key="admin:in_transit_share", slug="in_transit_share", business_name="In transit share",
                    table=net.table, expr=net.expr, format="percent")
    del model.measures[sales.key]
    # As the store lists them: by key, so the share comes before the metric it is made of.
    decisions = sorted([{"object_key": target("measure", m.key), "field": "define", "value": m.model_dump(mode="json")}
                        for m in (sales, share)], key=lambda d: d["object_key"])
    assert decisions[0]["object_key"] == "measure:admin:in_transit_share"
    notes = apply_overrides(model, decisions)
    assert notes == []
    assert {"admin:in_transit_share", "admin:shipped_sales"} <= set(model.measures)


def test_a_metric_made_of_one_no_longer_there_is_still_said(learned):
    model, _ = learned
    model = model.model_copy(deep=True)
    read = authoring.read(model, "[Gross amount] - [Net amount]")
    gone = read.expr.model_dump(mode="json")
    gone["args"][1] = {"measure": "admin:removed"}
    lonely = Measure(key="admin:gross_less", slug="twice", business_name="Gross less the removed one", table=read.table,
                     expr=read.expr).model_dump(mode="json")
    lonely["expr"] = gone
    notes = apply_overrides(model, [{"object_key": "measure:admin:gross_less", "field": "define", "value": lonely}])
    assert "admin:gross_less" not in model.measures
    assert any("Gross less the removed one" in n and "not in the data any more" in n for n in notes)


@pytest.mark.parametrize("named", [False, True], ids=["on_its_own", "made_of_another"])
def test_a_metric_counted_by_a_date_the_admin_added_keeps_it(learned, named):
    # The admin's dates were added after the metrics, so a metric counted by one was put back on its table's
    # default; one that waited for another metric kept it. Links, then dates, then metrics, then segments.
    model, _ = learned
    model = model.model_copy(deep=True)
    ship = next(r for r in model.date_roles.values() if r.slug == "ship_date")
    added = ship.model_copy(update={"key": "admin:shipped_on", "slug": "shipped_on", "name": "Shipped on",
                                    "is_default": False})
    read = authoring.read(model, "SUM([Order line · Net amount])")
    sales = Measure(key="admin:shipped_sales", slug="shipped_sales", business_name="Shipped sales", table=read.table,
                    expr=read.expr, default_date=added.key)
    decisions = [{"object_key": target("measure", sales.key), "field": "define", "value": sales.model_dump(mode="json")},
                 {"object_key": target("date_role", added.key), "field": "define", "value": added.model_dump(mode="json")}]
    if named:
        model.measures[sales.key] = sales
        share = authoring.read(model, "[Shipped sales] / [Net amount] * 100")
        del model.measures[sales.key]
        decisions.append({"object_key": "measure:admin:a_share", "field": "define", "value": Measure(
            key="admin:a_share", slug="a_share", business_name="A share", table=share.table, expr=share.expr,
            default_date=added.key).model_dump(mode="json")})
    notes = apply_overrides(model, sorted(decisions, key=lambda d: d["object_key"]))
    assert notes == []
    assert model.measures["admin:shipped_sales"].default_date == "admin:shipped_on"
    if named:
        assert model.measures["admin:a_share"].default_date == "admin:shipped_on"


def test_an_admin_saves_both_in_the_editor_and_the_workspace_has_both(site):  # noqa: F811
    base = f"/admin/clients/{site.account}/measures/api/save"
    status = _column(site.model(), "status_code")
    shipped = {"key": "", "name": "Shipped sales", "formula": "SUM([Order line · Net amount])", "format": "currency",
               "over_time": "added", "conditions": [{"column": status, "op": "eq", "values": ["S"]}]}
    share = {"key": "", "name": "In transit share", "formula": "[Shipped sales] / [Net amount] * 100",
             "format": "percent", "over_time": "per_period", "conditions": []}
    assert site.client.post(base, json=shipped).json()["ok"]
    assert site.client.post(base, json=share).json()["ok"]
    model = site.model()
    assert not [n for n in model.notes if n.startswith("A decision")]
    assert {"In transit share", "Shipped sales"} <= {m.business_name for m in model.measures.values()}
