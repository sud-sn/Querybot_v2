"""A person's name on the answer on screen never reaches the AI, in any workspace.

People's names are shown in answers (readers work with these people) but never sent to the AI. In a workspace where
member values may reach the AI, pointing at a row of an answer grouped by patient ("monthly sales for the first one")
handed the planner that patient's name as a value match, and the next question's previous plan showed it again. Now
a person's name the planner would see is a placeholder, as every member is where values are withheld, and the answer
still filters on the person. Other members (a state, a store) are still handed over as before.

An invented pharmacy in DuckDB; the AI is replaced by a script that keeps what it was sent.
"""

from __future__ import annotations

import datetime as dt
import json

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.plan.values import build_index
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse

TODAY = dt.date(2026, 6, 15)


@pytest.fixture(scope="module")
def shop():
    con = duckdb.connect()
    con.execute("CREATE TABLE patients (patient_id INTEGER PRIMARY KEY, first_name VARCHAR, last_name VARCHAR, "
                "state VARCHAR)")
    con.execute("INSERT INTO patients VALUES (1, 'Avery', 'Quill', 'OH'), (2, 'Blair', 'Moss', 'TX'), "
                "(3, 'Casey', 'Fern', 'OH')")
    con.execute("CREATE TABLE fills (fill_id INTEGER PRIMARY KEY, patient_id INTEGER, fill_date DATE, "
                "quantity INTEGER, price_charged DECIMAL(10, 2))")
    con.execute("INSERT INTO fills SELECT i, 1 + i % 3, DATE '2025-01-03' + CAST(i % 150 AS INTEGER), 1 + i % 4, "
                "round(20 + (i * 7) % 90 + (CASE WHEN i % 3 = 0 THEN 400 ELSE 0 END), 2) FROM range(1, 901) t(i)")
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="people", options=BuildOptions(workers=1))
    return con, model


class Script:
    def __init__(self, *plans):
        self.plans = [json.dumps({"kind": "query", **p}) for p in plans]
        self.sent: list[str] = []

    def __call__(self, stable: str, tail: str) -> str:
        self.sent.append(stable + "\n" + tail)
        return self.plans.pop(0)


def _person(model) -> str:
    """The patient's whole name (first and last, read joined)."""
    return next(a.slug for a in model.attributes.values()
                if model.columns[a.column].personal == "name" and model.columns[a.column].parts)


def _measure(model) -> str:
    return next(m.slug for m in model.measures.values() if "price" in m.slug and not m.hidden)


def _services(con, model, ai) -> Services:
    warehouse = DuckDBWarehouse(con)

    def fetch(slug: str) -> list[object]:
        column = model.columns[model.attributes[slug].column]
        table = model.tables[column.table]
        return [r[0] for r in warehouse.query(f'SELECT DISTINCT "{column.name}" FROM "{table.name}"').rows]

    return Services(model=model, warehouse=warehouse, complete=ai, index=build_index(model, fetch), today=TODAY,
                    values_allowed=True)


def test_pointing_at_a_person_hands_the_ai_a_placeholder_and_still_filters_on_them(shop):
    con, model = shop
    person, sales = _person(model), _measure(model)
    year = {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}
    ai = Script({"intent": "breakdown", "measures": [sales], "group_by": [person], "time": year},
                {"intent": "trend", "measures": [sales], "filters": [{"field": person, "op": "eq",
                                                                      "values": ["⟨v1⟩"]}],
                 "time": {**year, "grain": "month"}, "follow_up": "refine"},
                {"intent": "trend", "measures": [sales], "filters": [{"field": person, "op": "eq",
                                                                      "values": ["⟨v1⟩"]}],
                 "time": {"window": {"kind": "between", "start": "2026-01-01", "end": "2026-06-30"}, "grain": "month"},
                 "follow_up": "refine"})
    services, session = _services(con, model, ai), Session()
    first = answer_question("sales by patient in 2025", services, session)
    assert (first.get("data") or {}).get("rows"), (first["answer"], first.get("trust", {}).get("sql"))
    top = first["data"]["rows"][0][first["data"]["headers"][0]]
    assert top and " " in str(top), first["data"]["rows"][:2]
    pointed = answer_question("monthly sales for the first one", services, session)
    answer_question("and in 2026?", services, session)
    for sent in ai.sent[1:]:
        assert str(top) not in sent, sent[-1500:]
    assert "⟨v1⟩" in ai.sent[1]
    (applied,) = [f for f in pointed["plan"]["filters"] if f["field"] == person]
    assert applied["values"] == [top], "the answer still filters on that person"


def test_a_member_that_is_no_ones_name_is_handed_over_as_before(shop):
    con, model = shop
    sales = _measure(model)
    state = next(a.slug for a in model.attributes.values() if model.columns[a.column].name == "state")
    ai = Script({"intent": "value", "measures": [sales], "filters": [{"field": state, "op": "eq", "values": ["OH"]}]})
    answer_question('sales in "OH"', _services(con, model, ai), Session())
    assert f'{state} = "OH"' in ai.sent[0]


def test_a_column_marked_as_peoples_names_after_learn_is_never_indexed(shop):
    from core2.model.schema import SemanticModel
    from core2.plan.values import listable

    _, model = shop
    marked = SemanticModel.model_validate(model.model_dump())
    state = next(c for c in marked.columns.values() if c.name == "state")
    state.values_allowed = True
    assert any(marked.attributes[s].column == state.key for s in listable(marked))
    state.personal = "name"                  # the admin's choice on the Knowledge base page
    assert not any(marked.attributes[s].column == state.key for s in listable(marked))
