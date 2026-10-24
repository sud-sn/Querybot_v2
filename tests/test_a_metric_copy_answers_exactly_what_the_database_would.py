"""A copy of a metric kept on this server answers only what it holds exactly, and exactly as the database would.

As Databricks' materialized metric views, Looker's aggregate tables and Cube's pre-aggregations do, an admin keeps a
metric by some of its fields and by day or month (core2/precompute.py). A question is read from the copy only when
the copy gives the database's own rows:

    the same numbers      every golden question of six businesses a copy holds comes out of it with the rows the
                          database gives, in the same order
    the same answer       the answer read from the copy is the answer read from the database, and says it was read
                          from a copy and when the copy was made
    nothing it lacks      a field it does not keep, a month copy asked for the last 90 days, a count of different
                          values over more than one of its rows, the rows a table leaves out by default asked for:
                          the database answers
    never a balance       a figure read on a day (stock on hand) is never added up over days: no copy of it
    never people          a field holding people's names or details is never kept in a copy
    governance as ever    the copy only replaces the fetch: a reader with a row rule is answered by the database,
                          and masking applies to the copy's rows as to the database's

Invented data only (the six synthetic businesses of evals/core2); the AI is a recording.
"""

from __future__ import annotations

import collections
import datetime as dt
import decimal
import json
import math
from unittest.mock import patch

import pytest

from core2 import precompute as PC
from core2.compile.compiler import compile_query, compiled_from
from core2.model.schema import ColumnFilter
from core2.plan.ir import Plan
from core2.resolve.resolver import Context, resolve
from core2.service import Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2.compile_eval import _Words, golden
from tests.test_core2_answers_a_question_end_to_end import TODAY, Recorded, _domain, _services

DOMAINS = ["retail", "inventory", "finance", "hr", "purchasing", "subscriptions"]


def _key(model, slug: str) -> str:
    return next(k for k, m in model.measures.items() if m.slug == slug)


def _kept(model, warehouse, tmp_path, slug: str, fields: list[str], grain: str = "day", *,
          made_at: dt.datetime | None = None, every: int = 6, copy_id: int = 1) -> PC.Ready:
    """A copy made on ``warehouse`` as Solviz makes one, kept in ``tmp_path``."""
    defn = PC.define(model, metric=_key(model, slug), fields=fields, grain=grain, today=TODAY, dialect="duckdb")
    assert not any(warehouse.query(q, max_rows=1).rows for q, _ in defn.checks)
    path = tmp_path / f"copy-{copy_id}-{slug}-{grain}.duckdb"
    PC.write(path, defn, warehouse.query(defn.sql).rows)
    return PC.Ready(id=copy_id, name=f"{slug} copy", metric=_key(model, slug), fields=fields, date="", grain=grain,
                    every_hours=every, made_at=made_at or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
                    path=path, fingerprint=defn.fingerprint)


def _number(v):
    if isinstance(v, (decimal.Decimal, float, int)) and not isinstance(v, bool):
        return float(v)
    return v.date() if isinstance(v, dt.datetime) else v


def _same(a: list[tuple], b: list[tuple]) -> bool:
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        for x, y in zip(ra, rb):
            x, y = _number(x), _number(y)
            if isinstance(x, float) and isinstance(y, float):
                if not math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-6):
                    return False
            elif x != y:
                return False
    return True


def _on_copy(kept: PC.Ready, plan: Plan, model, today=TODAY):
    """(the database query, the copy's query or None) for ``plan``."""
    logical = resolve(plan, model, Context(today=today, split_units=True))
    compiled = compile_query(logical, model, "duckdb")
    defn = kept.definition(model, "duckdb")
    assert defn is not None
    return compiled, PC.copy_query(defn, logical, model, compiled)


class Spy:
    """The warehouse, telling the queries it ran on the database from those it read from a copy."""

    dialect, db_type = "duckdb", "duckdb"

    def __init__(self, inner):
        self.inner, self.database, self.copy = inner, [], []

    def query(self, sql, *, max_rows=None, local=None):
        (self.copy if local is not None else self.database).append(sql)
        return self.inner.query(sql, max_rows=max_rows, local=local)


# ── The same numbers ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", DOMAINS)
def test_every_golden_question_a_copy_holds_comes_out_with_the_databases_rows(name, tmp_path):
    _, built, model, _ = _domain(name)
    warehouse = DuckDBWarehouse(built.con)
    words = _Words(model, built)
    plans = []
    for case in golden(name)["questions"]:
        if case.get("plan"):
            try:
                plans.append((case, words.plan(case["plan"])))
            except Exception:  # noqa: BLE001 - a golden plan the domain's words do not read is not this test's
                pass
    # A copy of each metric the questions ask, by every field they break it down or filter it by, by day and by month.
    wanted: dict[str, set[str]] = collections.defaultdict(set)
    for _, plan in plans:
        for slug in plan.measures:
            if any(m.slug == slug for m in model.measures.values()):
                wanted[slug] |= {f for f in [*plan.group_by, *(x.field for x in plan.filters)] if f in model.attributes}
    copies = []
    for slug, fields in wanted.items():
        for grain in ("day", "month"):
            try:
                copies.append(_kept(model, warehouse, tmp_path, slug, sorted(fields), grain, copy_id=len(copies)))
            except PC.CopyError:
                continue           # a balance, or a date kept by month asked by day: refused, as it must be
    served = 0
    for case, plan in plans:
        try:
            logical = resolve(plan, model, Context(today=TODAY, split_units=True))
            compiled = compile_query(logical, model, "duckdb")
        except Exception:  # noqa: BLE001 - a question this domain's model cannot resolve is not this test's
            continue
        for kept in copies:
            defn = kept.definition(model, "duckdb")
            on_copy = PC.copy_query(defn, logical, model, compiled) if defn is not None else None
            if on_copy is None:
                continue
            served += 1
            expected = warehouse.query(compiled.sql).rows
            got = [tuple(r.values()) for r in PC.read(kept.path, on_copy, compiled.row_cap, "duckdb")]
            assert _same(expected, got), (case["id"], case["question"], expected[:3], got[:3])
            break
    assert served >= 3, f"only {served} golden questions of {name} were read from a copy"


# ── The same answer, said to be read from a copy ─────────────────────────────

QUESTION = "Net sales by region in 2025"
PLAN = {"intent": "breakdown", "measures": ["net_amount"], "group_by": ["region.name"],
        "time": {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}}


def test_an_answer_read_from_a_copy_is_the_answer_the_database_gives(tmp_path):
    _, built, model, _ = _domain("retail")
    kept = _kept(model, DuckDBWarehouse(built.con), tmp_path, "net_amount", ["region.name"], "month")
    plain = answer_question(QUESTION, _services(model, DuckDBWarehouse(built.con), Recorded(json.dumps(PLAN))),
                            Session())
    spy = Spy(DuckDBWarehouse(built.con))
    served = answer_question(QUESTION, _services(model, PC.Precomputed(spy, [kept]), Recorded(json.dumps(PLAN))),
                             Session())
    assert plain["type"] == served["type"] == "assistant_response", served
    assert served["trust"]["sql"] in spy.copy and served["trust"]["sql"] not in spy.database
    assert served["answer"]["headline"] == plain["answer"]["headline"]
    assert served["data"]["rows"] == plain["data"]["rows"]
    assert (served.get("chart") or {}).get("rows") == (plain.get("chart") or {}).get("rows")
    # Said: read from a copy, and when it was made; in French too. Never said of the database's answer.
    said = [c for c in served["coverage_caveats"] if c.startswith("Read from Solviz's copy of")]
    assert said and "made" in said[0] and served["coverage_caveats_fr"][said[0]].startswith("Lu dans la copie")
    assert served["trust"]["copy"]["name"] == kept.name and served["trust"]["copy"]["every_hours"] == 6
    assert "copy" not in plain["trust"] and not any("copy" in c for c in plain["coverage_caveats"])


def test_one_answers_figures_come_from_one_place(tmp_path):
    # Net sales by product: no copy holds it, so the database answers it -- and its line against last year's total,
    # which the copy by region does hold, is read from the database too: a breakdown read now never sits beside a
    # total from a copy made hours ago. The answer before it, read from the copy, said so; this one does not.
    _, built, model, _ = _domain("retail")
    kept = _kept(model, DuckDBWarehouse(built.con), tmp_path, "net_amount", ["region.name"], "month")
    spy = Spy(DuckDBWarehouse(built.con))
    services = _services(model, PC.Precomputed(spy, [kept]),
                         Recorded(json.dumps(PLAN), json.dumps({**PLAN, "group_by": ["product.name"]})))
    first = answer_question(QUESTION, services, Session())
    assert "copy" in first["trust"] and spy.copy
    spy.copy.clear()
    second = answer_question("Net sales by product in 2025", services, Session())
    assert second["type"] == "assistant_response" and spy.copy == [], "part of a database answer read from a copy"
    assert "copy" not in second["trust"], "an answer the database gave said it was read from a copy"


# ── Nothing it does not hold ─────────────────────────────────────────────────

def _plan(**kw) -> Plan:
    return Plan.model_validate({"intent": "breakdown", "measures": ["net_amount"], **kw})


def test_a_field_the_copy_does_not_keep_goes_to_the_database(tmp_path):
    _, built, model, _ = _domain("retail")
    kept = _kept(model, DuckDBWarehouse(built.con), tmp_path, "net_amount", ["region.name"])
    _, held = _on_copy(kept, _plan(group_by=["region.name"]), model)
    _, not_held = _on_copy(kept, _plan(group_by=["product.name"]), model)
    _, filtered = _on_copy(kept, _plan(group_by=["region.name"],
                                       filters=[{"field": "product.name", "op": "eq", "values": ["Dish Soap"]}]), model)
    assert held is not None and not_held is None and filtered is None


def test_a_copy_by_month_never_answers_part_of_a_month(tmp_path):
    _, built, model, _ = _domain("retail")
    kept = _kept(model, DuckDBWarehouse(built.con), tmp_path, "net_amount", ["region.name"], "month")
    whole = {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-03-31"}}
    part = {"window": {"kind": "last", "n": 90, "unit": "day"}}
    weeks = {"grain": "week", "window": {"kind": "between", "start": "2025-01-01", "end": "2025-03-31"}}
    assert _on_copy(kept, _plan(group_by=["region.name"], time=whole), model)[1] is not None
    assert _on_copy(kept, _plan(group_by=["region.name"], time=part), model)[1] is None
    assert _on_copy(kept, _plan(group_by=["region.name"], time=weeks), model)[1] is None
    by_day = _kept(model, DuckDBWarehouse(built.con), tmp_path, "net_amount", ["region.name"], "day", copy_id=2)
    assert _on_copy(by_day, _plan(group_by=["region.name"], time=part), model)[1] is not None


def test_a_count_of_different_values_is_read_only_from_exactly_the_copys_rows(tmp_path):
    _, built, model, _ = _domain("retail")
    warehouse = DuckDBWarehouse(built.con)
    kept = _kept(model, warehouse, tmp_path, "number_of_orders", ["region.name"], "month")
    year = {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}
    exact = Plan.model_validate({"intent": "trend", "measures": ["number_of_orders"], "group_by": ["region.name"],
                                 "time": {**year, "grain": "month"}})
    compiled, on_copy = _on_copy(kept, exact, model)
    assert on_copy is not None
    assert _same(warehouse.query(compiled.sql).rows,
                 [tuple(r.values()) for r in PC.read(kept.path, on_copy, compiled.row_cap, "duckdb")])
    # An order counted in two months is one order in the quarter; an order in two regions' rows, one in the total.
    by_quarter = exact.model_copy(update={"time": exact.time.model_copy(update={"grain": "quarter"})})
    no_region = exact.model_copy(update={"group_by": []})
    assert _on_copy(kept, by_quarter, model)[1] is None
    assert _on_copy(kept, no_region, model)[1] is None


def test_the_rows_a_table_leaves_out_by_default_asked_for_go_to_the_database(tmp_path):
    _, built, base, _ = _domain("retail")
    lines = next(t for t in base.tables.values() if t.name == "order_lines")
    status = next(c for c in lines.columns if c.endswith(".status_code"))
    model = base.model_copy(update={"tables": {**base.tables, lines.key: lines.model_copy(update={
        "default_filters": [ColumnFilter(column=status, op="ne", values=["CANCELLED"])]})}})
    warehouse = DuckDBWarehouse(built.con)
    kept = _kept(model, warehouse, tmp_path, "net_amount", ["region.name"])
    compiled, on_copy = _on_copy(kept, _plan(group_by=["region.name"]), model)
    assert on_copy is not None
    assert _same(warehouse.query(compiled.sql).rows,
                 [tuple(r.values()) for r in PC.read(kept.path, on_copy, compiled.row_cap, "duckdb")])
    assert _on_copy(kept, _plan(group_by=["region.name"], include_left_out=True), model)[1] is None


# ── What is never copied ─────────────────────────────────────────────────────

def test_a_balance_read_on_a_day_is_never_copied():
    _, _, model, _ = _domain("inventory")
    balance = next(m for m in model.measures.values() if m.additivity == "semi_additive")
    with pytest.raises(PC.CopyError, match="never added up over days"):
        PC.define(model, metric=balance.key, fields=[], today=TODAY, dialect="duckdb")


def test_a_field_holding_people_or_hidden_is_never_kept():
    _, _, base, _ = _domain("retail")
    name = base.attributes["customer.name"].column
    personal = base.model_copy(update={"columns": {**base.columns, name: base.columns[name].model_copy(
        update={"personal": "name"})}})
    hidden = base.model_copy(update={"columns": {**base.columns, name: base.columns[name].model_copy(
        update={"hidden": True})}})
    for model in (personal, hidden):
        with pytest.raises(PC.CopyError):
            PC.define(model, metric=_key(model, "net_amount"), fields=["customer.name"], today=TODAY,
                      dialect="duckdb")


def test_a_date_kept_by_month_is_copied_by_month_only():
    _, _, model, _ = _domain("finance")
    budget = model.measures[_key(model, "budget_amount")]
    with pytest.raises(PC.CopyError, match="by month"):
        PC.define(model, metric=budget.key, fields=[], grain="day", today=TODAY, dialect="duckdb")
    assert PC.define(model, metric=budget.key, fields=[], grain="month", today=TODAY, dialect="duckdb").sql


# ── Fresh, and of the metric as it is ────────────────────────────────────────

def test_a_copy_too_old_or_of_a_metric_since_changed_is_not_used(tmp_path):
    _, built, model, _ = _domain("retail")
    kept = _kept(model, DuckDBWarehouse(built.con), tmp_path, "net_amount", ["region.name"])
    now = dt.datetime(2026, 6, 15, 12, 0)
    assert PC.fresh(now - dt.timedelta(hours=11), 6, now) and not PC.fresh(now - dt.timedelta(hours=13), 6, now)
    key = _key(model, "net_amount")
    changed = model.model_copy(update={"measures": {**model.measures, key: model.measures[key].model_copy(
        update={"expr": model.measures[_key(model, "gross_amount")].expr})}})
    assert kept.definition(model, "duckdb") is not None
    assert kept.definition(changed, "duckdb") is None, "a copy of the metric as it was answered for it as it is"


# ── Governance as ever ───────────────────────────────────────────────────────

def _governed(rules: list[dict], fetched: list[dict]):
    import store
    from core.compliance import governed_query as G
    from core.schema import QueryRows

    store.init_db()          # the reader's groups are read, from whichever test database this run uses
    database = QueryRows([{"x": 2}])
    copy = QueryRows(fetched)
    with patch.object(G, "validate_sql", return_value=(True, "", "")), \
            patch.object(G, "inject_row_policies", side_effect=lambda sql, db, ctx: (sql, rules)), \
            patch.object(G, "run_query", return_value=database) as ran:
        from core.compliance.policy_engine import resolve_context

        context = resolve_context("copy-acct", {"id": 7, "role": "viewer"}, action="query_execution")
        result = G.execute_governed_query({}, "postgres", "SELECT 1 AS x", context=context, known_tables=set(),
                                          fetch=lambda cap: copy)
    return result, ran


def test_only_the_fetch_changes_and_a_row_rule_sends_it_to_the_database():
    result, ran = _governed([], [{"x": 1}])
    assert result.source == "copy" and result.rows == [{"x": 1}] and not ran.called
    result, ran = _governed([{"policy_id": 3, "table": "SALES"}], [{"x": 1}])
    assert result.source == "database" and result.rows == [{"x": 2}] and ran.called


def test_masking_applies_to_a_copys_rows_as_to_the_databases():
    from core.compliance import governed_query as G

    seen = []

    def protect(rows, decision, lineage, **_):
        seen.append(list(rows))
        return [{"x": "•••"} for _ in rows]

    with patch.object(G, "protect_rows", side_effect=protect):
        result, _ = _governed([], [{"x": 1}])
    assert seen == [[{"x": 1}]] and result.rows == [{"x": "•••"}]


def test_the_compiler_remembers_what_each_query_was_written_from():
    _, _, model, _ = _domain("retail")
    logical = resolve(_plan(group_by=["region.name"]), model, Context(today=TODAY))
    compiled = compile_query(logical, model, "duckdb")
    found = compiled_from(compiled.sql)
    assert found is not None and found[0] is logical and found[1] is model
    assert compiled_from("SELECT 1") is None


# ── What a review found ──────────────────────────────────────────────────────

YEAR = {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}


def _held_or_refused(kept, warehouse, plan, model, dialect="duckdb"):
    """The copy's rows where it answers ``plan`` (None where it does not), checked against the database's."""
    logical = resolve(plan, model, Context(today=TODAY, split_units=True))
    compiled = compile_query(logical, model, "duckdb")
    on_copy = PC.copy_query(kept.definition(model, "duckdb"), logical, model, compiled)
    if on_copy is None:
        return None
    got = [tuple(r.values()) for r in PC.read(kept.path, on_copy, compiled.row_cap, dialect)]
    assert _same(warehouse.query(compiled.sql).rows, got), plan
    return got


def test_a_filter_on_the_copys_own_date_counts_its_own_dates_not_the_window(tmp_path):
    _, built, model, _ = _domain("purchasing")
    warehouse = DuckDBWarehouse(built.con)
    kept = _kept(model, warehouse, tmp_path, "line_amount", ["supplier.country"])
    by = {"intent": "breakdown", "measures": ["line_amount"], "group_by": ["supplier.country"]}
    before_march = Plan.model_validate({**by, "time": YEAR,
                                        "filters": [{"field": "order_date", "op": "lt", "values": ["2025-03-01"]}]})
    since_june = Plan.model_validate({**by, "filters": [{"field": "order_date", "op": "gte", "values": ["2025-06-01"]}]})
    assert _held_or_refused(kept, warehouse, before_march, model) is not None
    _held_or_refused(kept, warehouse, since_june, model)          # never a crash: the copy or the database


def test_a_copy_that_cannot_be_read_hands_the_question_to_the_database(tmp_path):
    _, built, model, _ = _domain("retail")
    kept = _kept(model, DuckDBWarehouse(built.con), tmp_path, "net_amount", ["region.name"], "month")
    plan = json.dumps({"intent": "breakdown", "measures": ["net_amount"], "group_by": ["region.name"], "time": YEAR})
    plain = answer_question("Net sales by region in 2025",
                            _services(model, DuckDBWarehouse(built.con), Recorded(plan)), Session())
    kept.path.unlink()             # a making finished after the answer listed the copy, and took its file away
    served = answer_question("Net sales by region in 2025",
                             _services(model, PC.Precomputed(DuckDBWarehouse(built.con), [kept]), Recorded(plan)),
                             Session())
    assert served["type"] == "assistant_response" and served["data"]["rows"] == plain["data"]["rows"]
    assert "copy" not in (served.get("trust") or {})


def test_a_sum_or_average_of_different_values_is_never_copied():
    from core2.model.schema import SqlExpr

    _, _, model, _ = _domain("purchasing")
    key = _key(model, "line_amount")
    measure = model.measures[key]
    column = measure.expr.column
    for top in ("SUM", "AVG"):
        distinct = model.model_copy(update={"measures": {**model.measures, key: measure.model_copy(update={
            "expr": SqlExpr(sql=f'{top}(DISTINCT "{column}")', columns=[column])})}})
        with pytest.raises(PC.CopyError, match="different"):
            PC.define(distinct, metric=key, fields=["supplier.country"], today=TODAY, dialect="duckdb")


def test_a_copy_of_sql_server_data_keeps_its_names_as_sql_server_compares_them(tmp_path):
    """SQL Server compares text ignoring capitals and trailing spaces: so does its copy, and a copy whose names
    differ only so is not kept (SQL Server counts them as one)."""
    _, built, model, _ = _domain("retail")
    warehouse = DuckDBWarehouse(built.con)
    key = _key(model, "net_amount")
    tsql = PC.define(model, metric=key, fields=["region.name"], grain="month", today=TODAY, dialect="tsql")
    plain = PC.define(model, metric=key, fields=["region.name"], grain="month", today=TODAY, dialect="duckdb")
    rows = warehouse.query(plain.sql).rows
    PC.kept_as_compared(tsql, rows)                                    # one spelling a name: kept
    region = next(i for i, (name, _) in enumerate(tsql.columns) if name == tsql.dims[0].name)
    twice = [*rows, tuple(f"{v.lower()} " if i == region else v for i, v in enumerate(rows[0]))]
    with pytest.raises(PC.CopyError, match="capitals"):
        PC.kept_as_compared(tsql, twice)
    PC.kept_as_compared(plain, twice)                                  # elsewhere 'East' and 'east ' are two names
    path = tmp_path / "tsql.duckdb"
    PC.write(path, tsql, rows)
    east = next(r[region] for r in rows)
    asked = _plan(group_by=["region.name"], time=YEAR,
                  filters=[{"field": "region.name", "op": "eq", "values": [f"{east.upper()} "]}])
    logical = resolve(asked, model, Context(today=TODAY, split_units=True))
    on_copy = PC.copy_query(tsql, logical, model, compile_query(logical, model, "tsql"))
    assert on_copy is not None and PC.read(path, on_copy, 100, "tsql"), "SQL Server finds it: so does its copy"


def test_names_put_in_order_and_cut_are_read_where_the_database_orders_them(tmp_path):
    _, built, model, _ = _domain("retail")
    key = _key(model, "net_amount")
    by_name = Plan.model_validate({"intent": "rank", "measures": ["net_amount"], "group_by": ["region.name"],
                                   "sort": [{"by": "region.name"}], "limit": 2, "time": YEAR})
    a_to_z = by_name.model_copy(update={"limit": None, "intent": "breakdown"})
    by_total = Plan.model_validate({"intent": "rank", "measures": ["net_amount"], "group_by": ["region.name"],
                                    "sort": [{"by": "net_amount", "desc": True}], "limit": 2, "time": YEAR})
    for dialect, cut_by_name in (("tsql", False), ("postgres", False), ("snowflake", True), ("duckdb", True)):
        defn = PC.define(model, metric=key, fields=["region.name"], grain="month", today=TODAY, dialect=dialect)
        for plan, held in ((by_name, cut_by_name), (a_to_z, cut_by_name), (by_total, True)):
            logical = resolve(plan, model, Context(today=TODAY, split_units=True))
            on_copy = PC.copy_query(defn, logical, model, compile_query(logical, model, dialect))
            assert (on_copy is not None) == held, (dialect, plan.sort)


def test_the_earliest_or_latest_date_or_name_is_kept_as_what_it_is(tmp_path):
    """The latest promised date stays a date (never a number the making cannot write), the first order number a
    name ('00123' never 123.0)."""
    from core2.model.schema import AggExpr

    _, built, model, _ = _domain("purchasing")
    warehouse = DuckDBWarehouse(built.con)
    key = _key(model, "line_amount")
    measure = model.measures[key]
    columns = {c.name: c.key for c in model.columns.values() if c.table == measure.table}
    for agg, column in (("max", "promised_date"), ("min", "po_number")):
        kept_as = model.model_copy(update={"measures": {**model.measures, key: measure.model_copy(update={
            "expr": AggExpr(agg=agg, column=columns[column])})}})
        kept = _kept(kept_as, warehouse, tmp_path, "line_amount", ["supplier.country"], copy_id=len(column))
        plan = Plan.model_validate({"intent": "breakdown", "measures": ["line_amount"],
                                    "group_by": ["supplier.country"], "time": YEAR})
        assert _held_or_refused(kept, warehouse, plan, kept_as) is not None, column


def test_rows_a_copy_would_cut_in_its_own_order_of_names_are_the_databases(tmp_path):
    _, built, model, _ = _domain("retail")
    kept = _kept(model, DuckDBWarehouse(built.con), tmp_path, "net_amount", ["region.name", "store.name"])
    sql = f"SELECT d0, d1 FROM {PC.TABLE} ORDER BY d0, d1"
    assert PC.read_as_cut(kept, sql, 1, "duckdb").truncated          # cut where DuckDB itself would cut
    with pytest.raises(PC._Unreadable):
        PC.read_as_cut(kept, sql, 1, "tsql")                         # SQL Server orders names otherwise
    assert len(PC.read_as_cut(kept, sql, 0, "tsql")) > 1             # all of them: nothing cut


def test_the_first_name_is_copied_only_where_names_are_put_in_order_as_the_copy_does():
    from core2.model.schema import AggExpr

    _, _, model, _ = _domain("purchasing")
    key = _key(model, "line_amount")
    measure = model.measures[key]
    po = next(c.key for c in model.columns.values() if c.table == measure.table and c.name == "po_number")
    first = model.model_copy(update={"measures": {**model.measures, key: measure.model_copy(update={
        "expr": AggExpr(agg="min", column=po)})}})
    for dialect in ("tsql", "postgres", "oracle"):
        with pytest.raises(PC.CopyError, match="order"):
            PC.define(first, metric=key, fields=["supplier.country"], today=TODAY, dialect=dialect)
    PC.define(first, metric=key, fields=["supplier.country"], today=TODAY, dialect="snowflake")


def test_a_sql_server_field_that_counts_capitals_is_never_copied():
    _, _, model, _ = _domain("retail")
    key = _key(model, "net_amount")
    tsql = PC.define(model, metric=key, fields=["region.name"], grain="month", today=TODAY, dialect="tsql")
    asked = []

    def collation(name):
        def ask(sql):
            asked.append(sql)
            return [(name,)] if name is not None else []
        return ask

    PC.compared_ignoring_case(model, tsql, collation("SQL_Latin1_General_CP1_CI_AS"))
    assert asked and "INFORMATION_SCHEMA.COLUMNS" in asked[0]
    for counted in ("Latin1_General_CS_AS", "Latin1_General_100_BIN2"):
        with pytest.raises(PC.CopyError, match="capitals"):
            PC.compared_ignoring_case(model, tsql, collation(counted))
    with pytest.raises(PC.CopyError, match="could not be read"):
        PC.compared_ignoring_case(model, tsql, collation(None))
    elsewhere = PC.define(model, metric=key, fields=["region.name"], grain="month", today=TODAY, dialect="postgres")
    asked.clear()
    PC.compared_ignoring_case(model, elsewhere, collation("anything"))
    assert asked == []


def test_a_copy_is_written_exactly_and_quickly(tmp_path):
    """Measured on SQL Server: a copy's 32,000 rows took 41 s to write one by one. Written in bulk, every value as
    the database handed it back: the largest decimals to the last digit, dates, empty names and none at all."""
    import time

    defn = PC.Definition(metric="m", table="t", dims=[], date_role="r", stored="date", grain="day",
                         conditions=frozenset(), joins=frozenset(), calendar=None, leaves=[], sql="", checks=[],
                         columns=[("d0", "text"), ("p", "date"), ("c2", "decimal"), ("c3", "integer"),
                                  ("c4", "float"), ("c5", "timestamp")])
    odd = [("O'Brien, \"Ltd\"", dt.datetime(2025, 3, 1, 0, 0), decimal.Decimal("12345678901234567890.1234567891"),
            2**40, 0.1, dt.datetime(2025, 3, 1, 13, 45, 7, 120000)),
           ("", dt.date(2025, 3, 2), decimal.Decimal("1E+3"), 0, -2.5e-7, None),
           (None, None, None, None, None, None)]
    rows = odd + [(f"store {i % 97}", dt.date(2024, 1, 1) + dt.timedelta(days=i % 700),
                   decimal.Decimal(i) / 100, i, i / 3, None) for i in range(60_000)]
    started = time.perf_counter()
    PC.write(tmp_path / "copy.duckdb", defn, rows)
    took = time.perf_counter() - started
    import duckdb

    con = duckdb.connect(str(tmp_path / "copy.duckdb"), read_only=True)
    try:
        back = con.execute(f"SELECT * FROM {PC.TABLE}").fetchall()
    finally:
        con.close()
    assert back[0] == ("O'Brien, \"Ltd\"", dt.date(2025, 3, 1), decimal.Decimal("12345678901234567890.1234567891"),
                       2**40, 0.1, dt.datetime(2025, 3, 1, 13, 45, 7, 120000))
    assert back[1] == ("", dt.date(2025, 3, 2), decimal.Decimal("1000"), 0, -2.5e-7, None)
    assert back[2] == (None,) * 6 and len(back) == len(rows)
    assert back[-1][2] == decimal.Decimal("599.99") and back[-1][4] == 59_999 / 3
    assert took < 20, f"{len(rows):,} rows took {took:.1f} s to write"


def test_a_value_any_driver_hands_back_is_written_as_the_database_meant_it(tmp_path):
    """Found reviewing the bulk write: a time with its offset (Snowflake TIMESTAMP_TZ, PostgreSQL timestamptz) kept
    its clock time and lost the offset, so 11:00+05:00 sorted after 10:00Z; a numpy float was written as
    "np.float64(0.1)"; a yes/no for a number column and a 2 for a yes/no column stopped the whole making; bytes were
    written as "b'abc'"."""
    import numpy as np

    defn = PC.Definition(metric="m", table="t", dims=[], date_role="r", stored="date", grain="day",
                         conditions=frozenset(), joins=frozenset(), calendar=None, leaves=[], sql="", checks=[],
                         columns=[("c0", "timestamp"), ("c1", "float"), ("c2", "integer"), ("c3", "boolean"),
                                  ("c4", "text")])
    utc, plus5 = dt.timezone.utc, dt.timezone(dt.timedelta(hours=5))
    rows = [(dt.datetime(2025, 3, 1, 10, 0, tzinfo=utc), np.float64(0.1), True, 2, b"abc"),
            (dt.datetime(2025, 3, 1, 11, 0, tzinfo=plus5), np.float32(0.5), False, 0, "plain")]
    PC.write(tmp_path / "copy.duckdb", defn, rows)
    import duckdb

    con = duckdb.connect(str(tmp_path / "copy.duckdb"), read_only=True)
    try:
        back = con.execute(f"SELECT * FROM {PC.TABLE}").fetchall()
        latest = con.execute(f"SELECT MAX(c0) FROM {PC.TABLE}").fetchone()[0]
    finally:
        con.close()
    assert back == [(dt.datetime(2025, 3, 1, 10, 0), 0.1, 1, True, "abc"),
                    (dt.datetime(2025, 3, 1, 6, 0), 0.5, 0, False, "plain")]
    assert latest == dt.datetime(2025, 3, 1, 10, 0)       # 10:00 UTC is later than 11:00 at UTC+5


def test_a_warehouse_that_cannot_take_a_copys_rows_is_asked_as_ever(tmp_path):
    """Copies are read through the governed warehouse; wrapped round one that cannot take their rows, the database
    answers every question (never 'the query could not run')."""
    _, built, model, _ = _domain("retail")
    kept = _kept(model, DuckDBWarehouse(built.con), tmp_path, "net_amount", ["region.name"], "month")

    class Plain:
        dialect, db_type = "duckdb", "duckdb"

        def __init__(self):
            self.inner = DuckDBWarehouse(built.con)

        def query(self, sql, *, max_rows=None):
            return self.inner.query(sql, max_rows=max_rows)

    plan = json.dumps({"intent": "breakdown", "measures": ["net_amount"], "group_by": ["region.name"], "time": YEAR})
    plain = answer_question("Net sales by region in 2025", _services(model, Plain(), Recorded(plan)), Session())
    served = answer_question("Net sales by region in 2025",
                             _services(model, PC.Precomputed(Plain(), [kept]), Recorded(plan)), Session())
    assert served["type"] == "assistant_response" and served["data"]["rows"] == plain["data"]["rows"]
    assert "copy" not in (served.get("trust") or {})
