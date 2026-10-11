"""A figure kept with no date, asked for a year: the repair round is shown the dated figures that answer it.

Asked of a public health-records sample with a real model (gpt-4o), "Which payer covered the most cost in 2025?"
took the amount each payer covered over its lifetime, kept on the payer list with no date, and gave up: "Payer
has no date". The encounters' payer coverage is dated and reaches the payers: the repair round is told so.
Synthetic data only.
"""

from __future__ import annotations

import datetime as dt

import duckdb

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.plan.ir import Plan
from core2.plan.planner import check
from core2.warehouse.runner import DuckDBWarehouse


def test_the_dated_figures_reaching_the_grouping_are_offered():
    con = duckdb.connect()
    con.execute("CREATE TABLE Payer (PayerID INTEGER, Name VARCHAR, AmountCovered DECIMAL(14,2))")
    con.execute("INSERT INTO Payer SELECT i, 'Insurer ' || i || repeat('p', i % 4), 1000 * i FROM range(1, 13) t(i)")
    con.execute("CREATE TABLE Encounter (EncounterID INTEGER, PayerID INTEGER, StartDate DATE, "
                "PayerCoverage DECIMAL(12,2), BaseCost DECIMAL(12,2))")
    con.execute("INSERT INTO Encounter SELECT i, i % 12 + 1, DATE '2024-01-01' + CAST(i % 700 AS INTEGER), "
                "20 + i % 300, 50 + i % 400 FROM range(1, 4001) t(i)")
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="payers", options=BuildOptions(workers=1))
    payer = next(e for e in model.entities.values() if model.tables[e.table].name == "Payer")
    # The amount each payer covered: a number each payer has, asked for as a figure.
    lifetime = next(a for a in model.attributes.values() if a.column.endswith(".amountcovered"))
    dated = next(m for m in model.measures.values() if m.expr.column and m.expr.column.endswith(".payercoverage"))
    plan = Plan.model_validate({"kind": "query", "intent": "rank", "measures": [lifetime.slug], "group_by": [payer.slug],
                                "sort": [{"by": lifetime.slug, "desc": True}], "limit": 1,
                                "time": {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}})
    problems, hint = check(plan, model, dt.date(2026, 6, 15), "Which payer covered the most cost in 2025?")
    assert problems and problems[0].endswith("has no date."), problems
    assert f"{dated.slug} (" in hint and f"{lifetime.slug} (" not in hint, hint
