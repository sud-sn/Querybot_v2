"""Answers read in plain words, found on a pharmacy's test threads.

* A yes/no column was split as 0 and 1 ("0 leads with $1.20M across 2 is steriles") and its chip asked for "0".
  A flag Learn found now reads as words made of its name (Sterile / Not sterile), "Sterile" in quotes finds it,
  and its chips say "sterile or not".
* "Which category has the highest cost %?" was answered "<category> leads with <its gross amount>": the measure
  the answer was ranked by now leads it.
* Rows with no member ("Unknown") led the answer and were counted as one ("across 7 territories" beside an insight
  on the 6 there are): they are said apart and never counted.
* Days between two dates "led" ("USPS leads with 4.3 days"): more days is longer, not a lead.
* A field named by a bare word ("Description") is headed with its owner's name.
* The note on how a table was reached named the step both routes share ("it could also come through the rx fill"),
  not where the other route branches off.
* "No rows match" when every matching row is one a table leaves out by default: the answer says so, and offers them.
* "a rx fill": a or an by how the words are said.

Invented data only.
"""

from __future__ import annotations

import datetime as dt
import json
import re

import duckdb
import numpy as np
import pandas as pd
import pytest

from core2.answer.builder import article
from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.model.schema import Column, value_names
from core2.plan.values import build_index
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse

TODAY = dt.date(2026, 6, 15)
Y2025 = {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}
KEPT = "status_id <> 9"            # cancelled sales are left out by default (Learn found the status)


@pytest.fixture(scope="module")
def shop():
    rng = np.random.default_rng(11)
    con = duckdb.connect()
    con.execute("CREATE TABLE stores AS SELECT * FROM (VALUES (1, 'Harbour Store'), (2, 'Hilltop Store'), "
                "(3, 'Lakeside Store')) t(store_id, store_name)")
    con.execute("CREATE TABLE employees AS SELECT i AS employee_id, 'E' || lpad(CAST(i AS VARCHAR), 3, '0') AS "
                "employee_code, 1 + (i % 3) AS store_id FROM range(1, 13) r(i)")
    con.execute("CREATE TABLE products AS SELECT i AS product_id, 'Product ' || i AS product_name, "
                "CASE WHEN i % 4 = 0 THEN 1 ELSE 0 END AS is_sterile, "
                "CASE WHEN i <= 3 THEN NULL WHEN i % 2 = 0 THEN 'Creams' ELSE 'Capsules' END AS category "
                "FROM range(1, 21) r(i)")
    con.execute("CREATE TABLE diagnoses AS SELECT * FROM (VALUES (1, 'Chronic pain'), (2, 'Fibromyalgia'), "
                "(3, 'Neuropathy')) t(diagnosis_id, description)")
    con.execute("CREATE TABLE order_statuses AS SELECT * FROM (VALUES (1, 'Open'), (2, 'Shipped'), "
                "(9, 'Cancelled')) t(status_id, status_name)")
    n = 1500
    order = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 360, n), unit="D")
    # Most sales are of the three products with no category: "Unknown" is the largest group.
    product = np.where(rng.random(n) < 0.6, rng.integers(1, 4, n), rng.integers(4, 21, n))
    frame = pd.DataFrame({
        "sale_id": np.arange(1, n + 1), "store_id": rng.integers(1, 4, n), "employee_id": rng.integers(1, 13, n),
        "product_id": product, "diagnosis_id": rng.integers(1, 4, n),
        "status_id": rng.choice([1, 2, 9], n, p=[0.3, 0.6, 0.1]), "order_date": order.date,
        "ship_date": (order + pd.to_timedelta(rng.integers(1, 9, n), unit="D")).date,
        "amount": np.round(rng.uniform(20, 400, n), 2)})
    frame["cost"] = np.round(frame["amount"] * rng.uniform(0.2, 0.6, n), 2)
    con.register("_s", frame)
    con.execute("CREATE TABLE sales AS SELECT sale_id, store_id, employee_id, product_id, diagnosis_id, status_id, "
                "CAST(order_date AS DATE) AS order_date, CAST(ship_date AS DATE) AS ship_date, amount, cost FROM _s")
    con.execute("CREATE TABLE payments AS SELECT sale_id AS payment_id, sale_id, order_date + 3 AS paid_date, "
                "amount AS paid_amount FROM sales WHERE sale_id % 3 <> 0")
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="t", options=BuildOptions(workers=1))

    def fetch(slug):
        column = model.columns[model.attributes[slug].column]
        return [r[0] for r in con.execute(f'SELECT DISTINCT "{column.name}" FROM '
                                          f'"{model.tables[column.table].name}"').fetchall()]

    return con, warehouse, model, build_index(model, fetch).named(model)


class Recorded:
    """The AI: the plans written for the questions, in order; what it was sent, kept."""

    def __init__(self, *plans):
        self.plans = [p if callable(p) else json.dumps({"kind": "query", **p}) for p in plans]
        self.sent: list[str] = []

    def __call__(self, stable, tail):
        self.sent.append(tail)
        plan = self.plans.pop(0)
        return json.dumps({"kind": "query", **plan(tail)}) if callable(plan) else plan


def _ask(shop, ai, *questions):
    con, warehouse, model, index = shop
    services = Services(model=model, warehouse=warehouse, complete=ai, index=index, today=TODAY)
    session = Session()
    return [answer_question(q, services, session) for q in questions]


def _sql(shop, sql):
    return shop[0].execute(sql).fetchall()


# ── yes/no flags ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name, yes, no", [
    ("Is sterile", "Sterile", "Not sterile"),
    ("Cancelled flag", "Cancelled", "Not cancelled"),
    ("Has sterile cleanroom", "With sterile cleanroom", "Without sterile cleanroom"),
    ("Cold chain capable", "Cold chain capable", "Not cold chain capable"),
    ("SKU indicator", "SKU", "Not SKU"),
])
def test_a_flag_reads_as_words_made_of_its_name(name, yes, no):
    column = Column(key="t.c", table="t", name="c", data_type="integer", role="flag", business_name=name)
    assert value_names(column) == {"1": yes, "0": no}


def test_names_an_admin_gave_win_and_other_columns_have_none():
    named = Column(key="t.c", table="t", name="c", data_type="integer", role="flag", business_name="Is sterile",
                   value_names={"1": "Sterile room", "0": "Open room"})
    assert value_names(named) == {"1": "Sterile room", "0": "Open room"}
    assert value_names(Column(key="t.q", table="t", name="q", data_type="integer", role="measure",
                              business_name="Is counted")) == {}


def test_a_yes_no_split_reads_as_words_and_narrows_in_quotes(shop):
    by_flag = {"intent": "breakdown", "measures": ["amount"], "group_by": ["product.is_sterile"], "time": Y2025}
    split = _ask(shop, Recorded(by_flag), "net amount for sterile versus not in 2025")[0]
    labels = {r[k] for r in split["data"]["rows"] for k in r if isinstance(r[k], str)}
    assert labels == {"Sterile", "Not sterile"}, split["data"]["rows"]
    assert "across 2" not in split["answer"]["headline"] and " 0 " not in split["answer"]["headline"], \
        split["answer"]["headline"]
    chips = [c["question"] for c in split["follow_up_suggestions"]]
    assert any("by sterile or not" in q for q in chips) and not any("is sterile" in q.lower() for q in chips), chips

    def handed(tail):
        field, value = re.findall(r'-> (\S+) = "([^"]+)"', tail)[0]
        return {"intent": "value", "measures": ["amount"], "time": Y2025,
                "filters": [{"field": field, "op": "eq", "values": [value]}]}

    ai = Recorded(handed)
    sterile = _ask(shop, ai, 'net amount for "Sterile" in 2025')[0]
    want = _sql(shop, "SELECT ROUND(SUM(s.amount), 2) FROM sales s JOIN products p USING (product_id) WHERE "
                      f"p.is_sterile = 1 AND s.{KEPT} AND s.order_date BETWEEN '2025-01-01' AND '2025-12-31'")[0][0]
    assert round(sterile["kpi"]["value"], 2) == want
    # Compared with the number it is stored as, which every warehouse reads the same way, never the text '1'.
    assert re.search(r"is_sterile\s*=\s*1\b", sterile["trust"]["sql"]) and "'1'" not in sterile["trust"]["sql"], \
        sterile["trust"]["sql"]
    headline = sterile["answer"]["headline"]
    assert "for sterile" in headline and "is sterile" not in headline.lower() and " 1" not in headline, headline


# ── what leads a ranking ─────────────────────────────────────────────────────

def test_the_measure_a_ranking_is_ranked_by_leads_it(shop):
    cost_share = {"name": "Cost %", "op": "ratio", "measures": ["cost", "amount"], "scale": 100}
    plan = {"intent": "rank", "measures": ["amount", "cost"], "derived": [cost_share], "group_by": ["store.name"],
            "time": Y2025, "sort": [{"by": "Cost %", "desc": True}], "limit": 1}
    answer = _ask(shop, Recorded(plan), "Which store has the highest cost percentage in 2025?")[0]
    want = _sql(shop, "SELECT st.store_name FROM sales s JOIN stores st USING (store_id) WHERE s." + KEPT +
                      " AND s.order_date BETWEEN '2025-01-01' AND '2025-12-31' GROUP BY 1 "
                      "ORDER BY SUM(s.cost) / SUM(s.amount) DESC LIMIT 1")[0][0]
    assert f"{want} has the highest cost %, at " in answer["answer"]["headline"], answer["answer"]["headline"]
    assert "leads with" not in answer["answer"]["headline"]


def test_rows_with_no_member_still_lead_when_no_member_has_anything(shop):
    """Where every member has nothing ("X leads with 0; 95,895 has no item group"), the rows with none are the
    answer, as they were."""
    plan = {"intent": "breakdown", "measures": ["amount"], "group_by": ["product.category"], "time": Y2025,
            "filters": [{"field": "product.name", "op": "in", "values": ["Product 1", "Product 2"]}]}
    answer = _ask(shop, Recorded(plan), 'net amount by category for "Product 1" and "Product 2" in 2025')[0]
    assert answer["answer"]["headline"].split(": ", 1)[1].startswith("Unknown"), answer["answer"]["headline"]


def test_rows_with_no_member_never_lead_and_are_never_counted(shop):
    plan = {"intent": "breakdown", "measures": ["amount"], "group_by": ["product.category"], "time": Y2025}
    answer = _ask(shop, Recorded(plan), "net amount by product category in 2025")[0]
    rows = answer["data"]["rows"]
    unknown = next(r for r in rows if "Unknown" in r.values())
    named = max((r for r in rows if "Unknown" not in r.values()), key=lambda r: r["amount"])
    assert unknown["amount"] > named["amount"], "the rows with no category are the largest group"
    headline = answer["answer"]["headline"]
    leader = next(v for v in named.values() if isinstance(v, str))
    assert f": {leader} leads with" in headline and "across 2 categories" in headline, headline
    assert "has no category" in headline and "Unknown" not in headline, headline


def test_days_between_two_dates_take_longest_rather_than_lead(shop):
    plan = {"intent": "breakdown", "group_by": ["store.name"], "time": Y2025,
            "durations": [{"name": "Days from order to ship", "start": "order_date", "end": "ship_date"}]}
    answer = _ask(shop, Recorded(plan), "average days from order to ship by store in 2025")[0]
    assert "takes longest, at" in answer["answer"]["headline"] and "leads" not in answer["answer"]["headline"], \
        answer["answer"]["headline"]


# ── names and notes ──────────────────────────────────────────────────────────

def test_a_field_named_by_a_bare_word_is_headed_with_its_owners_name(shop):
    _, _, model, _ = shop
    slug = next(s for s, a in model.attributes.items() if a.business_name == "Description")
    plan = {"intent": "breakdown", "measures": ["amount"], "group_by": [slug], "time": Y2025}
    answer = _ask(shop, Recorded(plan), "net amount by diagnosis in 2025")[0]
    labels = answer["data"]["header_labels"]
    owner = model.entities[slug.split(".")[0]].business_name
    assert f"{owner} description" in labels.values(), labels
    assert "Description" not in labels.values()


def test_the_route_note_names_where_the_other_route_branches_off(shop):
    plan = {"intent": "breakdown", "measures": ["paid_amount"], "group_by": ["store.name"], "time": Y2025}
    answer = _ask(shop, Recorded(plan), "paid amount by store in 2025")[0]
    note = next(n for n in answer["trust"]["date_context"] if "could also come through" in n)
    assert "could also come through the employee" in note and "through the sale)" not in note, note


def test_an_answer_whose_rows_are_all_left_out_by_default_says_so_and_offers_them(shop):
    cancelled = {"intent": "value", "measures": ["amount"], "time": Y2025,
                 "filters": [{"field": "order_status.name", "op": "eq", "values": ["Cancelled"]}]}
    first, then = _ask(shop, Recorded(cancelled, {**cancelled, "include_left_out": True, "follow_up": "refine"}),
                       'net amount for order status "Cancelled" in 2025', "Include the rows left out by default")
    assert "every row that matches is left out by default" in first["answer"]["headline"], first["answer"]
    chip = first["follow_up_suggestions"][0]
    assert chip == {"label": "Include the rows left out", "question": "Include the rows left out by default"}
    want = _sql(shop, "SELECT ROUND(SUM(amount), 2) FROM sales WHERE status_id = 9 AND "
                      "order_date BETWEEN '2025-01-01' AND '2025-12-31'")[0][0]
    assert round(then["kpi"]["value"], 2) == want


def test_an_empty_answer_with_nothing_left_out_is_just_empty(shop):
    nothing = {"intent": "value", "measures": ["amount"], "time": {"window": {"kind": "between",
                                                                               "start": "2030-01-01",
                                                                               "end": "2030-12-31"}}}
    answer = _ask(shop, Recorded(nothing), "net amount in 2030")[0]
    assert "left out by default" not in answer["answer"]["headline"], answer["answer"]["headline"]


@pytest.mark.parametrize("words, said", [("rx fill", "an"), ("order", "an"), ("unit", "a"), ("SKU", "an"),
                                          ("x-ray", "an"), ("one-off", "a"), ("hour", "an"), ("sale", "a")])
def test_a_or_an_as_the_words_are_said(words, said):
    assert article(words) == said


def test_a_flag_holding_other_values_keeps_them():
    from core2.model.schema import ColumnProfile, TopValue

    odd = Column(key="t.c", table="t", name="c", data_type="integer", role="flag", business_name="Is sterile",
                 profile=ColumnProfile(top=[TopValue(value="1", count=5), TopValue(value="2", count=3)]))
    assert value_names(odd) == {}
