"""A grouping by a code whose rows keep its description is by the description.

Asked of a public health-records sample with a real model (gpt-4o), "Top 5 medications by total cost that year" was
grouped by the medication code and answered "medication 2563431 leads": the same rows keep each medication's
description. Unless the question asks for the code, the description is what it is grouped by. Synthetic data only.
"""

from __future__ import annotations

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.plan.ir import Plan
from core2.plan.planner import named_not_coded
from core2.warehouse.runner import DuckDBWarehouse


@pytest.fixture(scope="module")
def model():
    con = duckdb.connect()
    con.execute("CREATE TABLE Patient (PatientID INTEGER, City VARCHAR)")
    con.execute("INSERT INTO Patient SELECT i, ['North', 'South', 'East'][i % 3 + 1] FROM range(1, 301) t(i)")
    con.execute("CREATE TABLE Medication (PatientID INTEGER, StartDate DATE, MedicationCode VARCHAR, "
                "MedicationDescription VARCHAR, TotalCost DECIMAL(12,2))")
    con.execute("INSERT INTO Medication SELECT i % 300 + 1, DATE '2024-01-01' + CAST(i % 600 AS INTEGER), "
                "'M' || (100000 + i % 40), 'Drug ' || (i % 40) || ' 10 MG Oral Tablet', 5 + i % 200 "
                "FROM range(1, 5001) t(i)")
    warehouse = DuckDBWarehouse(con)
    return build_model(warehouse, from_duckdb(warehouse), client_id="meds", options=BuildOptions(workers=1))


def _slugs(model):
    code = next(a.slug for a in model.attributes.values() if a.column.endswith(".medicationcode"))
    described = next(a.slug for a in model.attributes.values() if a.column.endswith(".medicationdescription"))
    cost = next(m.slug for m in model.measures.values() if m.expr.column and m.expr.column.endswith(".totalcost"))
    return code, described, cost


def test_a_code_with_its_description_beside_it_is_grouped_by_the_description(model):
    code, described, cost = _slugs(model)
    plan = Plan.model_validate({"kind": "query", "intent": "rank", "measures": [cost], "group_by": [code],
                                "sort": [{"by": cost, "desc": True}], "limit": 5})
    assert named_not_coded(plan, model, "Top 5 medications by total cost").group_by == [described]
    assert named_not_coded(plan, model, "Top 5 medication codes by total cost") is plan
