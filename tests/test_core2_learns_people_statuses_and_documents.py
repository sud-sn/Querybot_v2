"""Learn reads people, statuses and documents from the data, whatever the warehouse calls its columns.

Found on a compounding pharmacy's warehouse, fixed for any warehouse:

* a status kept by number in a short lookup ("8" is "Closed - Cancelled") was never asked about, and
  revenue counted cancelled orders. Now cancelled, voided, duplicate and reversed rows are left out by
  default (every answer says so, an admin can count them again), and returned or refunded ones are asked
  about; a product called "Void Fill" is no status;
* patients were named by their street address and doctors by their practice: a member is now named by
  its own name, by first and last name together, never by a contact detail;
* people's names, emails, phones, addresses and birth dates were open to the AI: names are now shown in
  answers but never sent to the AI; contact details, birth dates and ID numbers are never shown;
* a fill number (a running number within each prescription) and the refills a prescription allows were
  added up as metrics; a document's figure written on each of its lines (a booking's travel fee) was added
  up once per line; days and minutes took another column's units; a birth date was a table's default date.

The home-services warehouse (evals/core2/domains/home_services.py) holds each of these shapes in four
naming styles, the generic one with meaningless names. Invented data only.
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

STYLES = ("descriptive", "warehouse", "pascal", "generic")
NAMED = ("descriptive", "warehouse", "pascal")
_learned: dict[tuple[str, str], tuple] = {}


def _model(style: str, name: str = "home_services"):
    if (name, style) not in _learned:
        built = materialize(domains.build(name), style)
        warehouse = DuckDBWarehouse(built.con)
        model = build_model(warehouse, from_duckdb(warehouse, declared_fks=built.declared_fks),
                            client_id=f"t-{name}", options=BuildOptions(workers=1))
        _learned[(name, style)] = (built, model, *_maps(model, built))
    return _learned[(name, style)]


def _table(model, t_of, logical):
    return next(t for t in model.tables.values() if t_of[t.key] == logical)


def _measure(model, c_of, ref):
    return next((m for m in model.measures.values() if c_of.get(getattr(m.expr, "column", None) or "") == ref), None)


# ── statuses ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("style", STYLES)
def test_cancelled_voided_and_duplicate_bookings_are_left_out_by_default(style):
    _, model, t_of, c_of = _model(style)
    bookings = _table(model, t_of, "bookings")
    [rule] = bookings.default_filters
    assert c_of[rule.column] == "bookings.status_id" and rule.op == "not_in"
    assert sorted(rule.values) == [4, 6, 7]
    assert rule.shown == ["Closed - Cancelled", "Closed - Duplicate Entry", "Closed - Voided"]
    review = next(r for r in model.review if r.key == f"default_filter:{rule.column}")
    assert review.choice_made.startswith("left out: Closed - Cancelled")
    assert "count all rows" in review.alternatives and "leave out Closed - Refunded" in review.alternatives


def test_every_answer_says_which_rows_it_left_out_and_a_reader_may_ask_for_them():
    from core2.plan.ir import Plan
    from core2.resolve.resolver import Context, resolve

    built, model, t_of, c_of = _model("descriptive")
    count = next(m for m in model.measures.values() if m.table == _table(model, t_of, "bookings").key
                 and getattr(m.expr, "agg", "") == "count")
    logical = resolve(Plan(intent="value", measures=[count.slug]), model, Context(today=dt.date(2026, 6, 15)))
    assert any("Closed - Cancelled, Closed - Duplicate Entry, Closed - Voided are left out" in n
               for n in logical.notes), logical.notes
    assert not any("status id" in n.lower() for n in logical.notes)
    every = resolve(Plan(intent="value", measures=[count.slug], include_left_out=True), model,
                    Context(today=dt.date(2026, 6, 15)))
    assert any("as asked" in n for n in every.notes)


def test_a_product_that_says_void_is_no_status():
    """Purchasing sells "Void Fill" (a packing material): its lines are never left out for it."""
    _, model, t_of, _ = _model("descriptive", "purchasing")
    assert not _table(model, t_of, "purchase_order_lines").default_filters


def test_the_learned_page_shows_learns_choice_and_the_way_to_count_them_again():
    from core2.model.view import _leave_out

    _, model, t_of, _ = _model("descriptive")
    rule = _table(model, t_of, "bookings").default_filters[0]
    got = _leave_out(model, f"default_filter:{rule.column}", rule.column)
    assert "decided" not in got and got["action"]["label"] == "Count all rows"
    assert got["action"]["value"] == "[]"


# ── people ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("style", NAMED)
def test_peoples_details_are_never_shown_and_their_names_never_reach_the_ai(style):
    _, model, _, c_of = _model(style)
    pii = sorted(c_of[k] for k, c in model.columns.items() if c.sensitivity == "pii")
    assert pii == ["customers.birth_date", "customers.email", "customers.phone", "customers.street_address",
                   "partners.office_phone", "technicians.mobile_number"]
    names = sorted(c_of[k] for k, c in model.columns.items()
                   if c.data_type == "text" and c.sensitivity == "none" and not c.values_allowed)
    assert names == ["customers.first_name", "customers.first_name+last_name", "customers.last_name",
                     "partners.contact_name", "technicians.technician_name"]
    for key, c in model.columns.items():
        if c.sensitivity == "pii" or not c.values_allowed:
            assert c.profile is None or not c.profile.top, f"{c_of[key]} keeps its values in the model"
    # A company's name is not a person's: it stays in the AI's list of names.
    assert next(c for k, c in model.columns.items() if c_of[k] == "partners.partner_name").values_allowed


def test_with_meaningless_names_contact_details_are_found_by_their_values():
    _, model, _, c_of = _model("generic")
    pii = {c_of[k] for k, c in model.columns.items() if c.sensitivity == "pii"}
    assert {"customers.email", "customers.phone", "customers.street_address", "technicians.mobile_number",
            "partners.office_phone"} <= pii


def test_no_persons_name_or_detail_reaches_the_ai():
    from core2.plan.catalog import catalog_text
    from core2.plan.planner import stable_prompt
    from core2.plan.values import build_index

    built, model, _, _ = _model("descriptive")
    customers = built.domain.table("customers").data
    names = list(customers["first_name"].unique()) + list(customers["last_name"].unique())
    prompt = stable_prompt(model) + catalog_text(model)
    assert not [n for n in names if n in prompt]
    assert not any(e in prompt for e in customers["email"].head(50))
    index = build_index(model, lambda slug: [])
    assert not [s for s in index.attributes if s.startswith(("customer.", "technician.")) and "name" in s]


def test_a_detail_never_shown_is_refused_when_asked_for():
    from core2.plan.ir import Plan
    from core2.resolve.resolver import Context, ResolveError, resolve

    _, model, _, c_of = _model("descriptive")
    email = next(a for a in model.attributes.values() if c_of[a.column] == "customers.email")
    with pytest.raises(ResolveError) as refused:
        resolve(Plan(intent="list", group_by=[email.slug]), model, Context(today=dt.date(2026, 6, 15)))
    assert refused.value.kind == "sensitive"


@pytest.mark.parametrize("style", NAMED)
def test_members_are_named_by_their_own_name_never_a_contact_detail(style):
    _, model, t_of, c_of = _model(style)
    labels = {t_of[e.table]: c_of[e.label_column] for e in model.entities.values() if e.label_column}
    assert labels["customers"] == "customers.first_name+last_name"
    assert labels["technicians"] == "technicians.technician_name"
    assert labels["partners"] == "partners.partner_name"            # the company, not its contact person
    codes = {t_of[e.table]: c_of[e.code_column] for e in model.entities.values() if e.code_column}
    assert codes.get("customers") == "customers.customer_code"      # never a phone number


def test_a_customer_named_in_two_parts_is_answered_by_the_full_name_and_kept_apart_by_key():
    from core2.compile.compiler import compile_query
    from core2.plan.ir import Plan
    from core2.resolve.resolver import Context, resolve

    built, model, t_of, c_of = _model("descriptive")
    fee = _measure(model, c_of, "bookings.travel_fee")
    plan = Plan(intent="rank", measures=[fee.slug], group_by=["customer"], limit=3,
                sort=[{"by": fee.slug, "desc": True}])
    logical = resolve(plan, model, Context(today=dt.date(2026, 6, 15)))
    sql = compile_query(logical, model, "duckdb").sql
    rows = built.con.execute(sql).fetchall()
    customers = built.domain.table("customers").data
    full = set(customers["first_name"] + " " + customers["last_name"])
    assert len(rows) == 3 and all(any(v in full for v in r if isinstance(v, str)) for r in rows), rows
    assert "TRIM(COALESCE(" in sql


@pytest.mark.parametrize("dialect", ["tsql", "snowflake", "oracle"])
def test_a_name_in_two_parts_is_joined_in_every_dialect(dialect):
    from core2.compile.compiler import column_sql

    _, model, _, c_of = _model("descriptive")
    key = next(k for k in model.columns if c_of[k] == "customers.first_name+last_name")
    text = column_sql(model, key, "c", dialect).sql(dialect=dialect)
    assert "first_name" in text and "last_name" in text and "COALESCE" in text


@pytest.mark.parametrize("style", STYLES)
def test_a_birth_date_is_never_a_date_questions_count_by(style):
    _, model, t_of, c_of = _model(style)
    customers = _table(model, t_of, "customers")
    assert c_of[model.date_roles[customers.default_date].column] == "customers.signup_date"
    if style != "generic":
        assert not any(c_of[r.column] == "customers.birth_date" for r in model.date_roles.values())


# ── documents, allowances and units ──────────────────────────────────────────


@pytest.mark.parametrize("style", NAMED)
def test_numbers_that_identify_or_allow_are_not_metrics(style):
    _, model, _, c_of = _model(style)
    learned = {c_of.get(getattr(m.expr, "column", None) or "") for m in model.measures.values()}
    assert not learned & {"booking_lines.line_number", "plan_renewals.renewal_number", "plan_renewals.renewals_allowed"}


@pytest.mark.parametrize("style", STYLES)
def test_a_bookings_fee_written_on_each_line_is_averaged_not_added_up(style):
    _, model, _, c_of = _model(style)
    on_lines = _measure(model, c_of, "booking_lines.travel_fee")
    assert (on_lines.expr.agg, on_lines.additivity) == ("avg", "non_additive"), on_lines.evidence
    assert any(r.object == on_lines.key and r.choice_made == "averaged" for r in model.review)
    # The booking's own fee and each line's own amount still add up.
    assert _measure(model, c_of, "bookings.travel_fee").additivity == "additive"
    assert _measure(model, c_of, "booking_lines.line_amount").additivity == "additive"


@pytest.mark.parametrize("style", NAMED)
def test_a_metric_named_by_its_own_unit_takes_no_other_unit(style):
    _, model, _, c_of = _model(style)
    for ref in ("visits.duration_minutes", "visits.travel_km"):
        m = _measure(model, c_of, ref)
        assert m is None or not m.unit_column, ref
    parts = _measure(model, c_of, "visits.parts_used_qty")
    assert parts.format != "currency" and c_of[parts.unit_column] == "visits.unit_of_measure"
    assert any(q.kind == "unit_mix" and q.object == parts.expr.column for q in model.quality)


def test_a_running_number_within_a_parent_is_a_key_wherever_it_starts():
    """A prescription's fills seen from its fifth on: the number runs on without a gap, never from 1."""
    from core2.bootstrap.keys import infer_keys
    from core2.bootstrap.profiler import profile_table

    con = duckdb.connect()
    con.execute("CREATE TABLE fills (fill_id INTEGER, rx TEXT, fill_no INTEGER, qty INTEGER)")
    con.execute("INSERT INTO fills SELECT row_number() OVER (), 'RX' || (r % 400), 3 + r // 400, 30 "
                "FROM range(0, 2400) t(r)")
    warehouse = DuckDBWarehouse(con)
    table = next(iter(from_duckdb(warehouse).tables.values()))
    keys = infer_keys(warehouse, table, profile_table(warehouse, table))
    assert ["rx", "fill_no"] in keys.alternate_keys


def test_a_shape_is_read_from_the_values_not_from_numbers_or_strengths():
    from core2.bootstrap.profiler import profile_table

    con = duckdb.connect()
    con.execute("CREATE TABLE t (c1 TEXT, c2 TEXT, c3 TEXT, c4 TEXT, c5 TEXT, c6 TEXT, c7 TEXT)")
    con.execute("""INSERT INTO t SELECT 'p' || r || '@example.org', '(212) 555-' || lpad(CAST(r AS TEXT), 4, '0'),
                   CAST(100 + r AS TEXT) || ' Elm Street', lpad(CAST(r % 900 + 100 AS TEXT), 3, '0') || '-45-' ||
                   lpad(CAST(r AS TEXT), 4, '0'), CAST(r % 50 AS TEXT) || ' mg per g', '1ZD3VT' || CAST(r AS TEXT),
                   CAST(1000000000 + r AS TEXT) FROM range(1, 500) t(r)""")
    warehouse = DuckDBWarehouse(con)
    table = next(iter(from_duckdb(warehouse).tables.values()))
    p = profile_table(warehouse, table)
    assert [p.columns[c].pattern for c in ("c1", "c2", "c3", "c4")] == ["email", "phone", "street", "national_id"]
    assert all(p.columns[c].pattern not in ("email", "phone", "street", "national_id") for c in ("c5", "c6", "c7"))


def test_counts_are_named_as_a_reader_says_them():
    from core2.bootstrap import names

    assert [names.plural(w) for w in ("diagnosis", "analysis", "status", "rx number")] == \
        ["diagnoses", "analyses", "statuses", "rx numbers"]
    con = duckdb.connect()
    con.execute("CREATE TABLE fact_revenue (revenue_id INTEGER PRIMARY KEY, booked_date DATE, gross_amount DECIMAL(10,2))")
    con.execute("INSERT INTO fact_revenue SELECT r, DATE '2026-01-01' + (r % 90)::INTEGER, r % 70 FROM range(1, 300) t(r)")
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="t-rev", options=BuildOptions(workers=1))
    assert "Number of revenue rows" in {m.business_name for m in model.measures.values()}


def test_words_that_only_look_like_personal_data_are_not():
    from core2.bootstrap import personal
    from core2.bootstrap.inventory import InvColumn, InvTable
    from core2.bootstrap.profiler import TableProfile
    from core2.model.schema import ColumnProfile

    def table(name, columns):
        t = InvTable(database="d", schema="s", name=name,
                     columns=[InvColumn(name=c, raw_type="VARCHAR", data_type="text") for c in columns])
        return t, TableProfile(rows=10, sampled=False, columns={c: ColumnProfile(rows=10) for c in columns})

    t, p = table("strategies", ["strategy_name", "add_on_name", "price_list_name", "fx_code"])
    assert personal.read_table(t, p) == {}
    t, p = table("customers", ["first_name", "last_name", "segment_name", "manager_name", "full_name"])
    assert personal.read_table(t, p) == {"first_name": "name", "last_name": "name", "manager_name": "name",
                                         "full_name": "name"}


def test_a_workspace_that_keeps_values_out_keeps_its_status_names_out_too():
    from core2.bootstrap.profiler import ProfileOptions

    built = materialize(domains.build("home_services"), "descriptive")
    warehouse = DuckDBWarehouse(built.con)
    model = build_model(warehouse, from_duckdb(warehouse, declared_fks=built.declared_fks), client_id="t-closed",
                        options=BuildOptions(workers=1, profile=ProfileOptions(values_allowed=lambda t, c: False)))
    assert not any(t.default_filters for t in model.tables.values())
    assert not any("Cancelled" in q.message for q in model.quality)
