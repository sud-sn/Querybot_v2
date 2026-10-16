"""Two members compared are shown side by side, never added together.

"Compare Small Appliances and Cookware revenue by quarter in 2025" was answered with one line: the two categories'
revenue added up, quarter by quarter ("for category Small Appliances or Cookware"). A comparison of members the
reader names, of one field, is a row (or a line) for each of them, whether the reader joins them with "and" or "vs";
"combined" adds them up.

Invented retail data; every figure is checked against SQL written by hand.
"""
from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.values import build_index
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn
from evals.core2.framework import materialize

TODAY = dt.date(2026, 6, 15)
Y2025 = {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}
HAND = """SELECT c.category_name, ((o.order_date_key // 100) % 100 - 1) // 3 + 1 AS q, SUM(o.net_amount)
          FROM order_lines o JOIN products p ON p.product_id = o.product_id
          JOIN categories c ON c.category_id = p.category_id
          WHERE c.category_name IN ('Snacks', 'Beverages') AND o.order_date_key BETWEEN 20250101 AND 20251231
          GROUP BY 1, 2"""


@pytest.fixture(scope="module")
def retail():
    domain = domains.build("retail")
    built, model = learn(domain, "descriptive")
    warehouse = DuckDBWarehouse(materialize(domain, "descriptive").con)

    def fetch(slug):
        column = model.columns[model.attributes[slug].column]
        return [r[0] for r in warehouse.query(f'SELECT DISTINCT "{column.name}" FROM '
                                              f'"{model.tables[column.table].name}"').rows]

    return model, warehouse, build_index(model, fetch)


def _ask(retail, question, plan):
    model, warehouse, index = retail
    services = Services(model=model, warehouse=warehouse, complete=lambda s, t: json.dumps({"kind": "query", **plan}),
                        index=index, today=TODAY)
    payload = answer_question(question, services, Session())
    assert payload.get("type") == "assistant_response" and "export_rows" in payload, payload.get("answer")
    return payload


def test_two_categories_compared_by_quarter_are_two_lines(retail):
    plan = {"intent": "compare", "measures": ["net_amount"],
            "filters": [{"field": "category.name", "op": "in", "values": ["Snacks", "Beverages"]}],
            "time": {"grain": "quarter", "window": Y2025}}
    payload = _ask(retail, 'Compare "Snacks" and "Beverages" net sales by quarter in 2025', plan)
    want = {(name, int(q)): round(float(v), 2) for name, q, v in retail[1].query(HAND).rows}
    rows = payload["export_rows"]
    name_col = next(k for k in rows[0] if "category" in k)
    period_col = next(k for k in rows[0] if k in ("period", "quarter") or "period" in k)
    got = {(r[name_col], (int(str(r[period_col])[5:7]) - 1) // 3 + 1): round(float(r["net_amount"]), 2) for r in rows}
    assert got == want


def test_two_categories_named_combined_stay_one_total(retail):
    plan = {"intent": "value", "measures": ["net_amount"],
            "filters": [{"field": "category.name", "op": "in", "values": ["Snacks", "Beverages"]}],
            "time": {"window": Y2025}}
    payload = _ask(retail, 'Net sales of "Snacks" and "Beverages" combined in 2025', plan)
    total = sum(float(v) for _, _, v in retail[1].query(HAND).rows)
    assert len(payload["export_rows"]) == 1
    assert float(payload["export_rows"][0]["net_amount"]) == pytest.approx(total)


def test_two_contradicting_conditions_on_one_field_become_one_compared_side_by_side(retail):
    # The AI wrote the two members as two conditions that no row meets together: "no value" was the answer.
    plan = {"intent": "compare", "measures": ["net_amount"],
            "filters": [{"field": "category.name", "op": "eq", "values": ["Snacks"]},
                        {"field": "category.name", "op": "eq", "values": ["Beverages"]}],
            "time": {"window": Y2025}}
    payload = _ask(retail, 'Net sales of "Snacks" vs "Beverages" in 2025', plan)
    want = {}
    for name, _, v in retail[1].query(HAND).rows:
        want[name] = round(want.get(name, 0.0) + float(v), 2)
    rows = payload["export_rows"]
    name_col = next(k for k in rows[0] if "category" in k)
    assert {r[name_col]: round(float(r["net_amount"]), 2) for r in rows} == want


def test_two_categories_named_without_quotes_by_quarter_are_two_lines(retail):
    # The names as typed, no quotes: the split by category the rule made was taken off again as a field "the reader
    # did not ask to split by" (meant for "vet revenue", where the members are the AI's reading of a word).
    plan = {"intent": "compare", "measures": ["net_amount"],
            "filters": [{"field": "category.name", "op": "in", "values": ["Snacks", "Beverages"]}],
            "time": {"grain": "quarter", "window": Y2025}}
    payload = _ask(retail, "Compare net sales of Snacks and Beverages by quarter in 2025", plan)
    want = {(name, int(q)): round(float(v), 2) for name, q, v in retail[1].query(HAND).rows}
    rows = payload["export_rows"]
    name_col = next((k for k in rows[0] if "category" in k), None)
    assert name_col is not None, rows[0]
    period_col = next(k for k in rows[0] if k in ("period", "quarter") or "period" in k)
    got = {(r[name_col], (int(str(r[period_col])[5:7]) - 1) // 3 + 1): round(float(r["net_amount"]), 2) for r in rows}
    assert got == want
