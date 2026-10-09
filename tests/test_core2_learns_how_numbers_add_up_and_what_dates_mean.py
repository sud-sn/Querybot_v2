"""Learn reads how each number adds up, and what each date means, from the rows as well as the names.

The accuracy benchmark (evals/core2/benchmark.py) found metrics summed that must never be (a price, a
percentage, a running counter), periodic amounts read as balances, contract and expiry dates read as
events, load stamps read as business dates, an employee dated by a birthday, and a balance answered
from a snapshot still being loaded. Each is checked here on the synthetic warehouses (or a small one
built for it), so the rules hold whatever the warehouse calls its columns.

Invented data only.
"""

from __future__ import annotations

import datetime as dt

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.framework import materialize
from evals.core2.learn_eval import _maps

_models: dict[tuple[str, str], tuple] = {}


def _learned(name: str, style: str = "descriptive"):
    if (name, style) not in _models:
        built = materialize(domains.build(name), style)
        warehouse = DuckDBWarehouse(built.con)
        model = build_model(warehouse, from_duckdb(warehouse, declared_fks=built.declared_fks),
                            client_id=f"t-{name}", options=BuildOptions(workers=1))
        _models[(name, style)] = (built, model)
    return _models[(name, style)]


def _measure(name: str, ref: str, style: str = "descriptive"):
    built, model = _learned(name, style)
    _, c_of = _maps(model, built)
    found = [m for m in model.measures.values() if c_of.get(getattr(m.expr, "column", None) or "") == ref]
    return found[0] if found else None


def _date(name: str, ref: str, style: str = "descriptive"):
    built, model = _learned(name, style)
    _, c_of = _maps(model, built)
    return next(r for r in model.date_roles.values() if c_of.get(r.column) == ref)


# ── how numbers add up ──────────────────────────────────────────────────────


@pytest.mark.parametrize("domain,price,amount", [
    ("retail", "order_lines.unit_price", "order_lines.gross_amount"),
    ("inventory", "stock_movements.unit_cost", "stock_movements.extended_cost"),
    ("purchasing", "purchase_order_lines.unit_price", "purchase_order_lines.line_amount"),
])
def test_a_price_is_its_amount_over_its_quantity_whatever_the_names(domain, price, amount):
    """With meaningless names (C05, C07) the rows still show amount = quantity x price."""
    p, a = _measure(domain, price, "generic"), _measure(domain, amount, "generic")
    assert (p.expr.agg, p.additivity) == ("avg", "non_additive"), p.evidence
    assert (a.expr.agg, a.additivity) == ("sum", "additive")
    assert any(e.kind == "per_unit" for e in p.evidence)


def test_the_quantity_in_a_product_is_never_read_as_a_ratio():
    q = _measure("inventory", "stock_movements.quantity", "generic")
    assert (q.expr.agg, q.additivity) == ("sum", "additive"), q.evidence


def test_a_price_charged_that_rises_with_the_quantity_is_summed():
    m = _measure("compounding_pharmacy", "fills.price_charged")
    assert (m.expr.agg, m.additivity) == ("sum", "additive")
    assert any(e.kind == "amount" for e in m.evidence)


def test_running_counters_are_read_at_their_highest():
    for ref in ("interface_counters.in_octets", "interface_counters.out_octets", "interface_counters.in_errors"):
        m = _measure("networking", ref)
        assert m is not None and (m.expr.agg, m.additivity) == ("max", "non_additive"), ref
    utilization = _measure("networking", "interface_counters.utilization_pct")
    assert utilization.expr.agg == "avg"


def test_what_a_thing_can_do_is_not_an_amount_of_it():
    assert _measure("networking", "services.bandwidth_mbps") is None
    assert _measure("networking", "interfaces.speed_mbps") is None


def test_a_recipes_quantity_per_unit_is_not_a_metric():
    assert _measure("compounding_pharmacy", "formula_ingredients.quantity_per_unit") is None


def test_an_event_table_other_tables_point_at_still_has_its_metrics():
    """Fills and quality tests point at batches; a batch number of one width is a code, not a name."""
    for ref in ("batches.batch_size", "batches.waste_qty"):
        m = _measure("compounding_pharmacy", ref)
        assert m is not None and (m.expr.agg, m.additivity) == ("sum", "additive"), ref
    assert _measure("compounding_pharmacy", "batches.yield_pct").expr.agg == "avg"


def test_a_level_between_0_and_1_stays_a_level():
    fte = _measure("hr", "headcount_monthly.fte")
    assert (fte.expr.agg, fte.additivity, fte.time_aggregation) == ("sum", "semi_additive", "last")


@pytest.mark.parametrize("domain,ref", [
    ("networking", "service_billing_monthly.billed_amount"),
    ("networking", "service_billing_monthly.data_usage_gb"),
])
def test_monthly_amounts_add_up_over_time(domain, ref):
    m = _measure(domain, ref)
    assert m.additivity == "additive"


def test_a_weekly_stock_snapshot_of_lots_is_a_snapshot():
    m = _measure("compounding_pharmacy", "ingredient_stock_weekly.on_hand_qty")
    assert (m.additivity, m.time_aggregation) == ("semi_additive", "last")
    assert _date("compounding_pharmacy", "ingredient_stock_weekly.snapshot_date").kind == "snapshot"


# ── what dates mean ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("ref,kind,default", [
    ("services.contract_start", "validity", True),
    ("services.contract_end", "validity", False),
    ("service_billing_monthly.billing_month", "event", True),
    ("alarms.raised_ts", "event", True),
    ("tickets.opened_ts", "event", True),
])
def test_networking_dates(ref, kind, default):
    role = _date("networking", ref)
    assert (role.kind, role.is_default) == (kind, default), role.evidence


@pytest.mark.parametrize("ref,kind,default", [
    ("batches.beyond_use_date", "validity", False),
    ("prescriptions.written_date", "event", True),
    ("prescriptions.created_at", "audit", False),
    ("claims.submitted_date", "event", True),
])
def test_pharmacy_dates(ref, kind, default):
    role = _date("compounding_pharmacy", ref)
    assert (role.kind, role.is_default) == (kind, default), role.evidence


@pytest.mark.parametrize("domain,ref,kind,default", [
    ("finance", "journal_lines.posting_date_key", "event", True),
    ("finance", "budgets.period_key", "event", True),
    ("hr", "payroll.pay_date", "event", True),
    ("retail", "sales_targets.period_key", "event", True),
    ("inventory", "stock_movements.expiry_date", "validity", False),
])
def test_dates_of_the_gated_warehouses(domain, ref, kind, default):
    role = _date(domain, ref)
    assert (role.kind, role.is_default) == (kind, default), role.evidence


def test_a_progress_date_is_decided_without_asking():
    """Raised, then acknowledged, then cleared: the alarm is dated by when it was raised, and no admin is asked."""
    _, model = _learned("networking")
    alarms = next(k for k, t in model.tables.items() if t.name == "alarms")
    assert f"default_date:{alarms}" not in {r.key for r in model.review}


# ── the same figure twice; a snapshot still loading ─────────────────────────


@pytest.fixture(scope="module")
def stock():
    """Daily stock per item: allocated is on hand again on most rows (a definition gone wrong); gross and net
    agree wherever nothing was discounted (two figures, rarely apart). The last day is still loading."""
    con = duckdb.connect()
    con.execute("CREATE TABLE items (item_id INTEGER PRIMARY KEY, item_name VARCHAR)")
    con.execute("INSERT INTO items SELECT i, 'Item number ' || i FROM range(1, 61) t(i)")
    con.execute("CREATE TABLE stock_daily (balance_date DATE, item_id INTEGER, on_hand_qty INTEGER, "
                "allocated_qty INTEGER, PRIMARY KEY (balance_date, item_id))")
    con.execute("INSERT INTO stock_daily SELECT DATE '2026-05-01' + d::INTEGER, i, 100 + (i * 7 + d) % 50, "
                "CASE WHEN (i + d) % 40 = 0 THEN 3 ELSE 100 + (i * 7 + d) % 50 END "
                "FROM range(0, 30) a(d), range(1, 61) b(i)")
    # 31 May: five items loaded so far
    con.execute("INSERT INTO stock_daily SELECT DATE '2026-05-31', i, 999, 999 FROM range(1, 6) t(i)")
    con.execute("CREATE TABLE sales (sale_id INTEGER PRIMARY KEY, item_id INTEGER, sale_date DATE, "
                "gross_amount DECIMAL(12,2), discount_amount DECIMAL(12,2), net_amount DECIMAL(12,2))")
    con.execute("INSERT INTO sales SELECT s, s % 60 + 1, DATE '2026-05-01' + (s % 30)::INTEGER, 10 + s % 90, "
                "CASE WHEN s % 50 = 0 THEN 2 ELSE 0 END, 10 + s % 90 - CASE WHEN s % 50 = 0 THEN 2 ELSE 0 END "
                "FROM range(1, 801) t(s)")
    warehouse = DuckDBWarehouse(con)
    fks = [{"constraint_name": f"fk_{t}_item", "parent_schema": "main", "parent_table": t, "parent_col": "item_id",
            "ref_schema": "main", "ref_table": "items", "ref_col": "item_id", "ordinal": 1, "enforced": False}
           for t in ("stock_daily", "sales")]
    model = build_model(warehouse, from_duckdb(warehouse, declared_fks=fks), client_id="stock",
                        options=BuildOptions(workers=1))
    return con, warehouse, model


def test_a_metric_equal_to_another_is_put_to_the_admin(stock):
    _, _, model = stock
    flags = [q for q in model.quality if q.kind == "same_values"]
    assert len(flags) == 1, [q.message for q in flags]
    names = {model.measures[flags[0].object].business_name.lower(),
             model.measures[flags[0].data["other"]].business_name.lower()}
    assert names == {"on hand quantity", "allocated quantity"}, names
    assert any(r.key == flags[0].key and "defined right" in r.question for r in model.review)


def test_a_partial_last_snapshot_is_not_the_latest_balance(stock):
    from core2.answer.snapshots import complete_snapshots
    from core2.compile.compiler import compile_query
    from core2.plan.ir import Plan
    from core2.resolve.resolver import Context, resolve

    con, warehouse, model = stock
    on_hand = next(m for m in model.measures.values()
                   if getattr(m.expr, "column", None) and model.columns[m.expr.column].name == "on_hand_qty")
    assert on_hand.additivity == "semi_additive"
    logical = resolve(Plan(intent="value", measures=[on_hand.slug]), model, Context(today=dt.date(2026, 6, 1)))
    logical = complete_snapshots(logical, model, warehouse)
    got = con.execute(compile_query(logical, model, "duckdb").sql).fetchall()
    expected = con.execute("SELECT SUM(on_hand_qty) FROM stock_daily WHERE balance_date = DATE '2026-05-30'").fetchone()
    assert float(got[0][-1]) == float(expected[0])
    assert any("31 May 2026" in n and "30 May 2026" in n and "incomplete" in n for n in logical.notes), logical.notes


def test_a_complete_last_snapshot_is_read_as_it_is(stock):
    from core2.answer.snapshots import complete_snapshots
    from core2.plan.ir import Plan
    from core2.resolve.resolver import Context, resolve
    from core2.plan.ir import TimeSpec, Window

    _, warehouse, model = stock
    on_hand = next(m for m in model.measures.values()
                   if getattr(m.expr, "column", None) and model.columns[m.expr.column].name == "on_hand_qty")
    plan = Plan(intent="value", measures=[on_hand.slug],
                time=TimeSpec(window=Window(kind="between", start=dt.date(2026, 5, 1), end=dt.date(2026, 5, 20))))
    logical = resolve(plan, model, Context(today=dt.date(2026, 6, 1)))
    before = (logical.window.start, logical.window.end)
    logical = complete_snapshots(logical, model, warehouse)
    assert (logical.window.start, logical.window.end) == before
    assert not any("incomplete" in n for n in logical.notes)
