"""A metric may read the fields of every table in the model, and a link may keep some rows only.

Before, a measure added up one table's rows: "net amount less refunds" (refunds are on
the returns table) could not be defined, a ratio of two measures from two tables was
refused, a measure's condition could only be on its own table, and a row formula's
conditions were dropped without a word. Links had no conditions of their own.

Now:

- a metric over several tables adds up each table on its own rows and combines the
  totals once they are lined up, so no row is counted twice (a refund is not repeated
  for every line of its order); a field that only describes the rows (a customer's
  segment on an order line) is read on those rows;
- a link can keep only some rows of the table it reaches ("the customer's current
  row"): the condition is part of the join, so a row with no kept match shows as
  Unknown, never dropped;
- an admin writes a metric as a formula over "[Table · Field]" names, metric names and
  the functions a metric may use; it is read into the model's own form (never pasted
  into a query), refused in words when it cannot be one, and checked against the data
  before it is saved.

Every number is compared with SQL written by hand against the invented retail warehouse.
"""

from __future__ import annotations

import datetime as dt

import pytest

from core2.compile.compiler import compile_query
from core2.model import authoring
from core2.model.overrides import apply_overrides
from core2.model.schema import ColumnFilter, Measure, OpExpr, RefExpr
from core2.plan.ir import Plan
from core2.resolve.resolver import Combined, Context, resolve
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
H1 = {"window": {"kind": "between", "start": "2026-01-01", "end": "2026-06-30"}}
IN_H1 = "between 20260101 and 20260630"


@pytest.fixture(scope="module")
def learned():
    built, model = learn(domains.build("retail"), "descriptive")
    return model, DuckDBWarehouse(built.con)


@pytest.fixture
def retail(learned):
    model, warehouse = learned
    return model.model_copy(deep=True), warehouse


def _measure(model, slug: str) -> Measure:
    return next(m for m in model.measures.values() if m.slug == slug)


def _column(model, table: str, name: str) -> str:
    return next(c.key for c in model.columns.values() if c.table.endswith(f".{table}") and c.name == name)


def _add(model, text: str, *, slug: str = "draft_metric", filters=()) -> Measure:
    read = authoring.read(model, text)
    m = Measure(key=slug, slug=slug, business_name=slug.replace("_", " ").capitalize(), table=read.table,
                expr=read.expr, filters=list(filters), format="number", status="approved")
    model.measures[slug] = m
    return m


def _rows(model, warehouse, plan: dict) -> list[dict]:
    logical = resolve(Plan.model_validate({"kind": "query", **plan}), model, Context(today=TODAY))
    for dialect in ("snowflake", "tsql", "oracle"):
        compile_query(logical, model, dialect)          # every warehouse can write it (and read it back)
    result = warehouse.query(compile_query(logical, model, "duckdb").sql)
    return [dict(zip(result.columns, row)) for row in result.rows]


def _truth(warehouse, sql: str) -> dict:
    return {row[0]: float(row[1]) for row in warehouse.query(sql).rows}


# ── metrics over several tables ──────────────────────────────────────────────

def test_sales_less_refunds_by_store_adds_up_each_table_on_its_own_rows(retail):
    model, warehouse = retail
    _add(model, "SUM([Order line · Net amount]) - SUM([Return · Refund amount])", slug="after_returns")
    rows = _rows(model, warehouse, {"intent": "breakdown", "measures": ["after_returns"], "group_by": ["store"],
                                    "time": H1})
    truth = _truth(warehouse, f"""
        with n as (select s.store_name, sum(o.net_amount) v from order_lines o join stores s using (store_id)
                   where o.order_date_key {IN_H1} group by 1),
             r as (select s.store_name, sum(r.refund_amount) v from returns r
                   join order_lines o on o.order_line_id = r.order_line_id join stores s on s.store_id = o.store_id
                   where r.return_date_key {IN_H1} group by 1)
        select n.store_name, n.v - coalesce(r.v, 0) from n left join r using (store_name)""")
    assert {r["store_name"]: float(r["after_returns"]) for r in rows} == pytest.approx(truth)


def test_a_ratio_of_metrics_from_two_tables_was_refused_and_is_now_worked_out(retail):
    model, warehouse = retail
    rate = {"name": "Refund rate", "op": "ratio", "measures": ["refund_amount", "net_amount"], "scale": 100}
    (row,) = _rows(model, warehouse, {"intent": "value", "derived": [rate], "time": H1})
    (truth,) = warehouse.query(f"""select (select sum(refund_amount) from returns where return_date_key {IN_H1})
        * 100.0 / (select sum(net_amount) from order_lines where order_date_key {IN_H1})""").rows[0]
    assert float(row["refund_rate"]) == pytest.approx(float(truth))


def test_a_field_of_another_table_of_events_is_added_up_on_its_own_rows(retail):
    # Returns reach their order line, so net amount could be read on the returns' rows:
    # the net amount of returned lines only. It is the order lines' own total.
    model, warehouse = retail
    _add(model, "SUM([Return · Refund amount]) / SUM([Order line · Net amount]) * 100", slug="refund_share")
    (row,) = _rows(model, warehouse, {"intent": "value", "measures": ["refund_share"], "time": H1})
    (truth,) = warehouse.query(f"""select (select sum(refund_amount) from returns where return_date_key {IN_H1})
        * 100.0 / (select sum(net_amount) from order_lines where order_date_key {IN_H1})""").rows[0]
    assert float(row["refund_share"]) == pytest.approx(float(truth))


def test_a_count_of_a_tables_rows_is_counted_on_that_table(retail):
    model, warehouse = retail
    _add(model, "COUNT([Return]) / COUNT([Order line]) * 100", slug="return_rate")
    (row,) = _rows(model, warehouse, {"intent": "value", "measures": ["return_rate"], "time": H1})
    (truth,) = warehouse.query(f"""select (select count(*) from returns where return_date_key {IN_H1}) * 100.0
        / (select count(*) from order_lines where order_date_key {IN_H1})""").rows[0]
    assert float(row["return_rate"]) == pytest.approx(float(truth))
    assert 0 < float(truth) < 50, "returns were once counted on the returns' own rows: 100%"


def test_one_part_used_twice_is_added_up_once(retail):
    model, _ = retail
    _add(model, "([Net amount] - [Refund amount]) / [Net amount] * 100", slug="kept_share")
    logical = resolve(Plan.model_validate({"kind": "query", "measures": ["kept_share"], "time": H1}), model,
                      Context(today=TODAY))
    hidden = [m for p in logical.parts for m in p.measures if m.hidden]
    assert len(hidden) == 2
    (shown,) = logical.measures
    assert isinstance(shown.expr, Combined) and not shown.hidden


def test_a_field_that_describes_the_rows_is_read_on_them(retail):
    model, warehouse = retail
    segment = _column(model, "customers", "segment")
    _add(model, "SUM([Order line · Net amount])", slug="store_sales",
         filters=[ColumnFilter(column=segment, op="ne", values=["Online"])])
    (row,) = _rows(model, warehouse, {"intent": "value", "measures": ["store_sales"], "time": H1})
    (truth,) = warehouse.query(f"""select sum(o.net_amount) from order_lines o left join customers c using (customer_id)
        where o.order_date_key {IN_H1} and (c.segment <> 'Online' or c.segment is null)""").rows[0]
    assert float(row["store_sales"]) == pytest.approx(float(truth))


def test_a_row_formula_keeps_its_conditions(retail):
    model, warehouse = retail
    status = _column(model, "order_lines", "status_code")
    _add(model, "SUM([Order line · Quantity] * [Order line · Unit price])", slug="list_value",
         filters=[ColumnFilter(column=status, op="ne", values=["C"])])
    (row,) = _rows(model, warehouse, {"intent": "value", "measures": ["list_value"], "time": H1})
    (truth,) = warehouse.query(f"""select sum(quantity * unit_price) from order_lines
        where order_date_key {IN_H1} and (status_code <> 'C' or status_code is null)""").rows[0]
    assert float(row["list_value"]) == pytest.approx(float(truth)), "the condition was dropped"


# ── links that keep some rows ────────────────────────────────────────────────

def test_a_link_keeps_only_the_rows_its_condition_names_and_the_rest_show_as_unknown(retail):
    model, warehouse = retail
    segment = _column(model, "customers", "segment")
    link = next(j for j in model.joins.values() if j.from_table.endswith(".order_lines")
                and j.to_table.endswith(".customers"))
    notes = apply_overrides(model, [{"object_key": f"join:{link.key}", "field": "conditions",
                                     "value": [{"column": segment, "op": "eq", "values": ["Wholesale"]}]}])
    assert notes == []
    rows = _rows(model, warehouse, {"intent": "breakdown", "measures": ["net_amount"],
                                    "group_by": ["customer.segment"], "time": H1})
    by_segment = {r["customer_segment"]: float(r["net_amount"]) for r in rows}
    assert set(by_segment) == {"Wholesale", None}
    total = _truth(warehouse, f"select 1, sum(net_amount) from order_lines where order_date_key {IN_H1}")[1]
    assert sum(by_segment.values()) == pytest.approx(total), "no row is dropped by the condition"
    logical = resolve(Plan.model_validate({"kind": "query", "measures": ["net_amount"],
                                           "group_by": ["customer.segment"], "time": H1}), model, Context(today=TODAY))
    assert any("a condition on the link" in n for n in logical.notes)


def test_a_link_condition_must_be_on_the_table_it_reaches(retail):
    model, _ = retail
    status = _column(model, "order_lines", "status_code")
    link = next(j for j in model.joins.values() if j.from_table.endswith(".order_lines")
                and j.to_table.endswith(".customers"))
    notes = apply_overrides(model, [{"object_key": f"join:{link.key}", "field": "conditions",
                                     "value": [{"column": status, "op": "eq", "values": ["C"]}]}])
    assert notes and model.joins[link.key].conditions == []


def test_a_changed_definition_that_reads_nothing_known_is_refused(retail):
    model, _ = retail
    net = _measure(model, "net_amount")
    before = net.expr
    notes = apply_overrides(model, [{"object_key": f"measure:{net.key}", "field": "expr",
                                     "value": {"agg": "sum", "column": "no.such.column"}}])
    assert notes and model.measures[net.key].expr == before
    notes = apply_overrides(model, [{"object_key": f"measure:{net.key}", "field": "expr",
                                     "value": OpExpr(op="subtract", args=[RefExpr(measure=net.key), RefExpr(
                                         measure=_measure(model, "refund_amount").key)]).model_dump()}])
    assert notes == [] and isinstance(model.measures[net.key].expr, OpExpr)


# ── a formula, as the admin writes it ────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "SUM([Order line · Net amount]) - SUM([Return · Refund amount])",
    "(([Net amount] - [Refund amount]) / [Net amount]) * 100",
    "COUNT(DISTINCT [Order line · Order number])",
    "SUM([Order line · Quantity] * [Order line · Unit price])",
    "(COUNT([Return]) / COUNT([Order line])) * 100",
])
def test_a_formula_reads_back_as_it_was_written(retail, text):
    model, _ = retail
    read = authoring.read(model, text)
    assert authoring.to_text(model, read.expr) == text
    assert authoring.read(model, authoring.to_text(model, read.expr)).expr == read.expr


def test_the_warehouses_own_names_are_read_too(retail):
    model, _ = retail
    assert authoring.read(model, "SUM(net_amount)").expr == authoring.read(model, "SUM([Order line · Net amount])").expr
    assert authoring.read(model, "sum(order_lines.net_amount)").table.endswith(".order_lines")


@pytest.mark.parametrize("text,said", [
    ("[Order line · Net amount]", "is a field: say how to add it up"),
    ("SUM(nothing_here)", "there is no field called nothing_here"),
    ("COUNT(*)", "COUNT([Order line])"),
    ("SUM(SUM(net_amount))", "cannot hold another one"),
    ("SUM([Order line · Net amount]) + 5", "a number can only multiply or divide"),
    ("SUM([Nowhere · Nothing])", "there is no field, metric or table called Nowhere · Nothing"),
    ("SUM([Return · Refund amount] * [Order line · Unit price]) / SUM([Store · Store name])", ""),
])
def test_a_formula_that_is_not_a_metric_is_refused_in_words(retail, text, said):
    model, _ = retail
    if not said:        # a row formula across a return and its order line is fine: they are one row apart
        authoring.read(model, text)
        return
    with pytest.raises(authoring.AuthoringError, match=said.replace("(", r"\(").replace(")", r"\)")
                       .replace("[", r"\[").replace("]", r"\]").replace("+", r"\+")):
        authoring.read(model, text)


def test_the_editor_offers_every_tables_fields_and_the_metrics(retail):
    model, _ = retail
    refs = {f["ref"]: f for f in authoring.fields(model)}
    assert refs["Return · Refund amount"]["kind"] == "number"
    assert refs["Customer · Segment"]["kind"] == "text"
    assert not any(f["table"] == "Calendar" for f in refs.values())
    assert "Net amount" in {m["ref"] for m in authoring.metrics(model)}


# ── checked against the data ─────────────────────────────────────────────────

def test_a_draft_is_checked_against_the_data_and_leaves_no_trace(retail):
    model, warehouse = retail
    status = _column(model, "order_lines", "status_code")
    read = authoring.read(model, "SUM([Order line · Net amount])")
    draft = Measure(key="d", slug="d", business_name="Completed sales", table=read.table, expr=read.expr,
                    filters=[ColumnFilter(column=status, op="ne", values=["C"])])
    checked = authoring.check(model, draft, warehouse, TODAY)
    assert checked["problem"] == ""
    assert [m["period"] for m in checked["months"]] == ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05",
                                                        "2026-06"]
    june = warehouse.query("""select count(*), count(*) filter (where status_code = 'C'),
        sum(net_amount) filter (where status_code <> 'C' or status_code is null)
        from order_lines where order_date_key between 20260601 and 20260630""").rows[0]
    assert (checked["rows"], checked["left_out"]) == (june[0] - june[1], june[1])
    assert checked["months"][-1]["value"] == pytest.approx(float(june[2]))
    assert not any(k.startswith("__draft__") for k in model.measures)


def test_a_draft_that_cannot_run_says_why(retail):
    model, warehouse = retail
    target = _column(model, "sales_targets", "target_amount")
    read = authoring.read(model, "SUM([Order line · Net amount])")
    draft = Measure(key="d", slug="d", business_name="Sales", table=read.table, expr=read.expr,
                    filters=[ColumnFilter(column=target, op="gt", values=[0])])
    checked = authoring.check(model, draft, warehouse, TODAY)
    assert "not linked" in checked["problem"]
    assert not any(k.startswith("__draft__") for k in model.measures)
