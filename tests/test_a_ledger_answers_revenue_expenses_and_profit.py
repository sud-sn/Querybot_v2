"""A ledger answers revenue, expenses and profit from its debits and credits by account type.

A general ledger keeps each amount twice, as a debit on one account and a credit on another, so its lines add
up to nothing. Asked "what was our revenue in fiscal 2025?", the new core summed the lines' net amount ($0 in
each currency); asked how much a division spent, it summed every debit, the cash and receivable sides
included. Learn now proposes the figures a finance reader means: revenue, the credits less the debits on
revenue accounts; expenses, the debits less the credits on expense accounts; profit, the one less the other,
each kept apart by currency like the lines' own amounts. A division with no revenue accounts shows its
expenses as a loss, not an empty profit. A table of debits and credits whose accounts carry no account type
gets none of them.

The synthetic finance warehouse (invented data), and a small one built here; figures are checked against
hand-written SQL.
"""

from __future__ import annotations

import datetime as dt
import json

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.framework import materialize

TODAY = dt.date(2026, 6, 14)
FY25 = {"kind": "between", "start": "2024-07-01", "end": "2025-06-30"}
LINES = ("journal_lines j JOIN accounts a ON a.account_id = j.account_id "
         "JOIN calendar c ON c.date_key = j.posting_date_key")


@pytest.fixture(scope="module")
def ledger():
    built = materialize(domains.build("finance"), "descriptive")
    warehouse = DuckDBWarehouse(built.con)
    model = build_model(warehouse, from_duckdb(warehouse, declared_fks=built.declared_fks), client_id="t-ledger",
                        options=BuildOptions(workers=1))
    return built.con, model


def _ask(con, model, plan):
    services = Services(model=model, warehouse=DuckDBWarehouse(con),
                        complete=lambda s, t: json.dumps({"kind": "query", **plan}), index=MemberIndex(), today=TODAY)
    payload = answer_question("q", services, Session())
    data = payload.get("data") or {}
    assert data.get("headers"), payload["answer"]["headline"]
    return data["rows"]


def _ledger(model, role):
    return next(m for m in model.measures.values() if m.key.endswith(f".ledger.{role}"))


def test_the_ledger_gets_revenue_expenses_and_profit_with_finance_words(ledger):
    _, model = ledger
    revenue, expenses, profit = (_ledger(model, r) for r in ("revenue", "expenses", "profit"))
    assert "revenue" in revenue.synonyms["en"] and "chiffre d'affaires" in revenue.synonyms["fr"]
    assert "spend" in expenses.synonyms["en"] and expenses.better == "lower"
    assert "net income" in profit.synonyms["en"]
    assert revenue.unit_column and revenue.unit_column == expenses.unit_column


def test_revenue_is_the_credits_less_debits_on_revenue_accounts_by_currency(ledger):
    con, model = ledger
    rows = _ask(con, model, {"intent": "value", "measures": [_ledger(model, "revenue").slug], "time": {"window": FY25}})
    got = {r[k]: float(r[_ledger(model, "revenue").slug]) for r in rows for k in r if k.endswith("currency_code")}
    want = dict(con.execute(f"SELECT j.currency_code, SUM(j.credit_amount - j.debit_amount) FROM {LINES} "
                            "WHERE a.account_type = 'Revenue' AND c.full_date BETWEEN DATE '2024-07-01' "
                            "AND DATE '2025-06-30' GROUP BY 1").fetchall())
    assert got == {k: pytest.approx(float(v)) for k, v in want.items()}


def test_profit_by_division_counts_a_division_with_no_revenue_as_a_loss(ledger):
    con, model = ledger
    slug = _ledger(model, "profit").slug
    rows = _ask(con, model, {"intent": "breakdown", "measures": [slug], "group_by": ["division.name"],
                             "filters": [{"field": next(a.slug for a in model.attributes.values()
                                                        if a.column.endswith(".currency_code")),
                                          "op": "eq", "values": ["CAD"]}]})
    got = {r["division_name"]: float(r[slug]) for r in rows}
    want = dict(con.execute(
        f"SELECT d.division_name, SUM(CASE WHEN a.account_type = 'Revenue' THEN j.credit_amount - j.debit_amount "
        "WHEN a.account_type = 'Expense' THEN -(j.debit_amount - j.credit_amount) ELSE 0 END) "
        f"FROM {LINES} JOIN cost_centres cc ON cc.cost_centre_id = j.cost_centre_id "
        "JOIN divisions d ON d.division_id = cc.division_id WHERE j.currency_code = 'CAD' GROUP BY 1").fetchall())
    assert got == {k: pytest.approx(float(v)) for k, v in want.items()}
    assert any(v < 0 for v in got.values())       # a division with expenses and no revenue: a loss, not empty


def test_debits_and_credits_without_account_types_get_none():
    con = duckdb.connect()
    con.execute("CREATE TABLE parties (party_id INTEGER PRIMARY KEY, party_name VARCHAR, party_kind VARCHAR)")
    con.execute("CREATE TABLE postings (posting_id INTEGER PRIMARY KEY, posted_on DATE, party_id INTEGER, "
                "debit_amount DECIMAL(12, 2), credit_amount DECIMAL(12, 2))")
    con.executemany("INSERT INTO parties VALUES (?, ?, ?)",
                    [(i, f"Party {i}", ("Customer", "Supplier", "Partner")[i % 3]) for i in range(1, 31)])
    con.executemany("INSERT INTO postings VALUES (?, ?, ?, ?, ?)",
                    [(n, dt.date(2026, 1, 1) + dt.timedelta(days=n % 150), n % 30 + 1,
                      float(n % 7) * 10, float(n % 5) * 10) for n in range(1, 400)])
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="t-noledger", options=BuildOptions(workers=1))
    assert not any(".ledger." in m.key for m in model.measures.values())
