"""Every field a question can name reaches the AI, read the way the question means it.

Found on a pharmacy's warehouse ("What are the shipping methods with a transit time of exactly 3 days?" was
answered "transit time is not available"), then looked for on every warehouse at hand:

* a number each member has (a ship method's typical transit days, a product's list price, a weight) was
  dropped when most members have their own value: it is now a number to compare, sort and show;
* a whole number that names (an NPI, a ZIP code, a GL account code) is an identifier, shown as written;
* a fact's own number or text that identifies its rows (a tracking or invoice number) can be looked up,
  listed and ranked by; a row's place in its document (line 2 of a journal) is still nothing;
* free text (a clinic's notes) is searched, never grouped by;
* "the cheapest" listed by name instead of by price, naming the wrong one first;
* with no field holding it, transit time is worked out from the ship and delivery dates, and a group can be
  kept by its own figure ("methods whose transit time is about 3 days").

Invented data only.
"""

from __future__ import annotations

import datetime as dt
import json

import duckdb
import numpy as np
import pandas as pd
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.plan.catalog import catalog_text
from core2.plan.ir import Plan
from core2.plan.values import build_index, listable
from core2.resolve.resolver import Context, ResolveError, resolve
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse

TODAY = dt.date(2026, 6, 15)
Y2025 = {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}
TRANSIT = {"Ground": 3, "Two Day": 2, "Overnight": 1, "Economy": 5, "Freight": 3, "Courier": 0, "Saver": 3,
           "Priority": 4}
NOTES = ["Opened after two practices merged; sports injuries and rehabilitation are most of its work.",
         "A small rural practice that also runs a weekly outreach clinic in the next valley over.",
         "Specialises in dermatology and compounded creams; most orders ship on Mondays and Thursdays."]


@pytest.fixture(scope="module")
def carrier():
    rng = np.random.default_rng(7)
    con = duckdb.connect()
    methods = pd.DataFrame({"ship_method_id": range(1, 9), "service_name": list(TRANSIT),
                            "carrier": ["Parcelco", "Fastship", "Fastship", "Postal", "Parcelco", "Local", "Postal",
                                        "Fastship"],
                            "typical_transit_days": list(TRANSIT.values()),
                            "base_rate": [6.4, 12.9, 31.5, 5.2, 48.0, 22.75, 7.1, 18.3]})
    con.register("_m", methods)
    con.execute("CREATE TABLE ship_methods AS SELECT * FROM _m")
    clinics = pd.DataFrame({"clinic_id": range(1, 31), "clinic_name": [f"Clinic {i}" for i in range(1, 31)],
                            "npi": [1003000000 + 7919 * i for i in range(1, 31)],
                            "zip_code": [10001 + 37 * i for i in range(1, 31)],
                            "notes": [NOTES[i % 3] + f" Clinic number {i} on the list." for i in range(30)]})
    con.register("_c", clinics)
    con.execute("CREATE TABLE clinics AS SELECT * FROM _c")
    n = 1600
    method = rng.integers(1, 9, n)
    ship = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 360, n), unit="D")
    days = np.array([TRANSIT[list(TRANSIT)[m - 1]] for m in method]) + rng.choice([-1, 0, 0, 0, 1], n)
    shipments = pd.DataFrame({
        "shipment_id": np.arange(1, n + 1), "ship_method_id": method, "clinic_id": rng.integers(1, 31, n),
        "ship_date": ship.date, "delivered_date": (ship + pd.to_timedelta(np.maximum(days, 0), unit="D")).date,
        "tracking_number": [f"TRK{900000 + 13 * i}" for i in range(n)],
        "shipping_cost": np.round(rng.uniform(4, 60, n), 2)})
    con.register("_s", shipments)
    con.execute("CREATE TABLE shipments AS SELECT shipment_id, ship_method_id, clinic_id, CAST(ship_date AS DATE) "
                "AS ship_date, CAST(delivered_date AS DATE) AS delivered_date, tracking_number, shipping_cost FROM _s")
    lines = pd.DataFrame({"journal_id": np.repeat(np.arange(1, 801), 2), "line_number": np.tile([1, 2], 800),
                          "gl_account_code": rng.choice([4000, 4100, 5000, 5100, 6000, 6100], 1600),
                          "posting_date": (pd.Timestamp("2025-01-01")
                                           + pd.to_timedelta(rng.integers(0, 360, 1600), unit="D")).date,
                          "amount": np.round(rng.uniform(-900, 900, 1600), 2)})
    con.register("_j", lines)
    con.execute("CREATE TABLE journal_lines AS SELECT journal_id, line_number, gl_account_code, "
                "CAST(posting_date AS DATE) AS posting_date, amount FROM _j")
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="t", options=BuildOptions(workers=1))

    def fetch(slug):
        column = model.columns[model.attributes[slug].column]
        return [r[0] for r in con.execute(f'SELECT DISTINCT "{column.name}" FROM '
                                          f'"{model.tables[column.table].name}"').fetchall()]

    return con, warehouse, model, build_index(model, fetch)


class Recorded:
    """The AI: the plans written for the questions, in order."""

    def __init__(self, *plans):
        self.plans = [json.dumps({"kind": "query", **p}) for p in plans]
        self.problems: list[str] = []

    def __call__(self, stable, tail):
        if "PROBLEMS:" in tail:
            self.problems.append(tail.split("PROBLEMS:", 1)[1][:300])
        return self.plans.pop(0) if self.plans else json.dumps({"kind": "unsupported", "notes": ["no plan"]})


def _ask(carrier, plan, question):
    con, warehouse, model, index = carrier
    ai = Recorded(plan)
    payload = answer_question(question, Services(model=model, warehouse=warehouse, complete=ai, index=index,
                                                 today=TODAY), Session())
    assert not ai.problems, ai.problems
    return payload


def _attribute(model, column_name):
    return next((a for a in model.attributes.values() if model.columns[a.column].name == column_name), None)


def _headline(payload):
    return (payload.get("answer") or {}).get("headline") or payload.get("text") or ""


def _rows(payload):
    return (payload.get("data") or {}).get("rows") or []


# ── what Learn keeps, and how ────────────────────────────────────────────────

@pytest.mark.parametrize("column, kind", [
    ("typical_transit_days", "number"), ("base_rate", "number"),
    ("npi", "identifier"), ("zip_code", "identifier"), ("gl_account_code", "identifier"),
    ("tracking_number", "identifier"), ("notes", "text"), ("carrier", "group"),
])
def test_every_field_a_question_can_name_is_kept_with_how_it_is_read(carrier, column, kind):
    attribute = _attribute(carrier[2], column)
    assert attribute is not None, f"{column} was dropped"
    assert attribute.kind == kind


def test_a_rows_place_in_its_document_is_still_not_offered(carrier):
    assert _attribute(carrier[2], "line_number") is None


def test_the_catalog_tells_the_ai_how_to_read_each_field(carrier):
    lines = {line.split("|")[0].strip(" -"): line for line in catalog_text(carrier[2]).splitlines() if "|" in line}
    transit = next(v for k, v in lines.items() if k.endswith("typical_transit_days"))
    assert "6 values: 0, 1, 2, 3, 4, 5 (a number: compare, sort or show it; group by it)" in transit
    rate = next(v for k, v in lines.items() if k.endswith("base_rate"))
    assert "a number: compare, sort or show it, from 5.2 to 48; too many values to group by" in rate
    gl = next(v for k, v in lines.items() if k.endswith("gl_account_code"))
    assert "6 values: 4000, 4100, 5000, 5100, 6000, 6100 (an identifier" in gl
    tracking = next(v for k, v in lines.items() if k.endswith("tracking_number"))
    assert "1,600 values (an identifier, shown as written" in tracking
    notes = next(v for k, v in lines.items() if k.endswith("notes"))
    assert "free text: search it with contains; never grouped by" in notes


def test_document_numbers_and_free_text_are_never_read_into_the_member_index(carrier):
    model = carrier[2]
    named = {model.columns[model.attributes[s].column].name for s in listable(model)}
    assert "tracking_number" not in named and "notes" not in named and "carrier" in named


# ── the questions ────────────────────────────────────────────────────────────

def test_the_methods_with_a_transit_time_of_exactly_3_days(carrier):
    """The question that was refused. "3" is a value, not a name: it narrows the answer without quotes."""
    slug = _attribute(carrier[2], "typical_transit_days").slug
    payload = _ask(carrier, {"intent": "list", "group_by": [slug.split(".")[0]],
                             "filters": [{"field": slug, "op": "eq", "values": ["3"]}]},
                   "What are the shipping methods with a transit time of exactly 3 days?")
    names = sorted(next(iter(r.values())) for r in _rows(payload))
    assert names == sorted(k for k, v in TRANSIT.items() if v == 3) == ["Freight", "Ground", "Saver"]
    assert _headline(payload) == "3 services where typical transit days is 3: Freight, Ground, Saver."


def test_the_lowest_by_a_number_is_listed_in_that_order(carrier):
    """ "Which ship method has the lowest base rate?" listed by name and named Courier ($22.75)."""
    rate = _attribute(carrier[2], "base_rate").slug
    entity = rate.split(".")[0]
    payload = _ask(carrier, {"intent": "list", "group_by": [entity, rate], "sort": [{"by": rate, "desc": False}],
                             "limit": 1}, "Which ship method has the lowest base rate?")
    assert _headline(payload) == "1 service: Economy (base rate 5.2)."
    payload = _ask(carrier, {"intent": "list", "group_by": [entity, rate],
                             "filters": [{"field": rate, "op": "gt", "values": [20]}],
                             "sort": [{"by": rate, "desc": True}]}, "Ship methods with a base rate above 20")
    assert _headline(payload) == ("3 services where base rate is above 20: Freight (base rate 48), "
                                  "Overnight (base rate 31.5), Courier (base rate 22.75).")


def test_a_code_reads_with_its_name_and_as_written(carrier):
    con = carrier[0]
    gl = _attribute(carrier[2], "gl_account_code").slug
    amount = next(m.slug for m in carrier[2].measures.values() if getattr(m.expr, "agg", "") == "sum"
                  and carrier[2].columns[m.expr.column].name == "amount")
    payload = _ask(carrier, {"intent": "breakdown", "measures": [amount], "group_by": [gl], "time": Y2025,
                             "sort": [{"by": amount, "desc": True}]}, "Amount by GL account code in 2025")
    top, total = con.execute("SELECT gl_account_code, SUM(amount) FROM journal_lines GROUP BY 1 ORDER BY 2 DESC "
                             "LIMIT 1").fetchone()
    assert _headline(payload).split(": ", 1)[1].startswith(f"GL account {top} leads with")
    assert (payload.get("data") or {}).get("column_formats", {}).get(gl.replace(".", "_")) == "text"


def test_a_document_is_looked_up_by_the_number_the_reader_writes(carrier):
    con, _, model, _ = carrier
    tracking = _attribute(model, "tracking_number").slug
    method = _attribute(model, "carrier").slug
    number, expected = con.execute("SELECT s.tracking_number, m.carrier FROM shipments s JOIN ship_methods m "
                                   "USING (ship_method_id) WHERE shipment_id = 42").fetchone()
    payload = _ask(carrier, {"intent": "list", "group_by": [tracking, method],
                             "filters": [{"field": tracking, "op": "eq", "values": [number]}]},
                   f'Show the shipment with tracking number "{number}"')
    assert _headline(payload) == f"Tracking number {number}: carrier {expected}."


def test_free_text_is_searched_never_grouped_by(carrier):
    con, _, model, _ = carrier
    notes = _attribute(model, "notes").slug
    entity = notes.split(".")[0]
    payload = _ask(carrier, {"intent": "list", "group_by": [entity],
                             "filters": [{"field": notes, "op": "contains", "values": ["dermatology"]}]},
                   "Which clinics' notes mention dermatology?")
    assert len(_rows(payload)) == con.execute("SELECT COUNT(*) FROM clinics WHERE notes LIKE '%dermatology%'"
                                              ).fetchone()[0] == 10
    with pytest.raises(ResolveError, match="free text"):
        resolve(Plan.model_validate({"intent": "list", "group_by": [notes]}), model, Context(today=TODAY))


# ── worked out from two dates when no field holds it ─────────────────────────

DAYS = {"name": "Days from ship to delivery", "start": "ship_date", "end": "delivered_date", "agg": "avg"}


def _by_method_days(con):
    return dict(con.execute("SELECT m.service_name, AVG(s.delivered_date - s.ship_date) FROM shipments s "
                            "JOIN ship_methods m USING (ship_method_id) GROUP BY 1").fetchall())


def test_groups_kept_by_their_own_figure_worked_out_from_two_dates(carrier):
    """ "Which shipping methods had a transit time of exactly 3 days?" with only the dates: each method's average,
    kept when it is about 3 (from 2.5 to under 3.5), not each shipment that took 3 days."""
    con = carrier[0]
    entity = _attribute(carrier[2], "typical_transit_days").slug.split(".")[0]
    payload = _ask(carrier, {"intent": "breakdown", "durations": [DAYS], "group_by": [entity], "time": Y2025,
                             "filters": [{"field": DAYS["name"], "op": "eq", "values": [3], "total": True}]},
                   "Which shipping methods had a transit time of exactly 3 days in 2025?")
    got = {next(iter(r.values())): round(r["days_from_ship_to_delivery"], 6) for r in _rows(payload)}
    want = {k: round(v, 6) for k, v in _by_method_days(con).items() if 2.5 <= v < 3.5}
    assert got == want and len(want) >= 2
    assert "where days from ship to delivery on average is about 3 days" in _headline(payload)
    payload = _ask(carrier, {"intent": "breakdown", "durations": [DAYS], "group_by": [entity], "time": Y2025,
                             "filters": [{"field": DAYS["name"], "op": "gt", "values": [4], "total": True}]},
                   "Which methods take more than 4 days on average?")
    assert sorted(next(iter(r.values())) for r in _rows(payload)) == sorted(
        k for k, v in _by_method_days(con).items() if v > 4)


def test_the_rows_reading_still_keeps_rows(carrier):
    """ "How many shipments took exactly 3 days?" counts shipments, as before."""
    con, _, model, _ = carrier
    count = next(m.slug for m in model.measures.values() if getattr(m.expr, "agg", "") == "count"
                 and model.tables[m.table].name == "shipments")
    payload = _ask(carrier, {"intent": "count", "measures": [count], "time": Y2025,
                             "durations": [{**DAYS, "measure": False}],
                             "filters": [{"field": DAYS["name"], "op": "eq", "values": [3]}]},
                   "How many shipments took exactly 3 days from ship to delivery in 2025?")
    assert _rows(payload)[0][count] == con.execute(
        "SELECT COUNT(*) FROM shipments WHERE delivered_date - ship_date = 3").fetchone()[0]
