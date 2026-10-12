"""Learn names an amount by what is taken from it, read from the rows.

The profitability stress test (gross amount, discount, invoice line amount = gross less discount, cost, margin =
line amount less cost) answered "revenue" with the gross amount, before discounts, on every question: no measure
was called revenue, and the AI took the gross for it. The rows say which is which: an amount that is another
less its discount is what was sold (revenue, net sales); an amount less its cost is the margin (gross profit), and
what the cost is taken from is what was sold. Each word goes to one measure, never to one when another is
already called so; a ledger's debit less credit is named nothing.

Invented data only.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pytest

from core2.bootstrap.arithmetic import name_by_difference
from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.framework import materialize
from evals.core2.learn_eval import _maps


def _learn(con, declared_fks=()):
    warehouse = DuckDBWarehouse(con)
    return build_model(warehouse, from_duckdb(warehouse, declared_fks=list(declared_fks)), client_id="t-names",
                       options=BuildOptions(workers=1))


def _domain(name: str, style: str):
    built = materialize(domains.build(name), style)
    model = _learn(built.con, built.declared_fks)
    _, c_of = _maps(model, built)
    by_ref = {c_of.get(getattr(m.expr, "column", None) or ""): m for m in model.measures.values()
              if getattr(m.expr, "agg", "") == "sum"}
    return model, by_ref


def _words(measure) -> list[str]:
    return [w for ws in measure.synonyms.values() for w in ws]


@pytest.mark.parametrize("style", ["descriptive", "warehouse", "pascal"])
def test_the_amount_after_discounts_is_the_revenue(style):
    model, m = _domain("retail", style)
    net, gross = m["order_lines.net_amount"], m["order_lines.gross_amount"]
    assert {"Revenue", "Net sales", "Sales"} <= set(_words(net)), net.synonyms
    assert "Chiffre d'affaires" in _words(net)
    assert not {"Revenue", "Net sales"} & set(_words(gross)), "the gross, before discounts, is not the revenue"
    assert any(e.kind == "difference" and "less" in e.detail for e in net.evidence)
    assert "after discounts" in net.description


def test_a_ledgers_debit_less_credit_is_named_nothing():
    model, m = _domain("finance", "warehouse")
    assert not any({"Revenue", "Gross profit"} & set(_words(x)) for x in model.measures.values())


def test_with_meaningless_names_the_relation_is_kept_and_nothing_named():
    model, _ = _domain("retail", "generic")
    assert any(e.kind == "difference" for x in model.measures.values() for e in x.evidence)
    assert not any("Revenue" in _words(x) for x in model.measures.values())


def _invoices(*, with_discount: bool, line_first: bool = False):
    rng = np.random.default_rng(7)
    n = 400
    gross = np.round(rng.uniform(50, 900, n), 2)
    discount = np.round(gross * rng.choice([0.0, 0.05, 0.1], n, p=[0.5, 0.3, 0.2]), 2) if with_discount else 0 * gross
    amount = np.round(gross - discount, 2)
    cost = np.round(amount * rng.uniform(0.5, 0.7, n), 2)
    con = duckdb.connect()
    order = ["lin_amt", "grs_amt", "dsc_amt"] if line_first else ["grs_amt", "dsc_amt", "lin_amt"]
    con.execute(f"CREATE TABLE ivc_lin (ivc_lin_id INTEGER, ivc_dt DATE, {', '.join(c + ' DECIMAL(12,2)' for c in order)}, "
                "cst_amt DECIMAL(12,2), mgn_amt DECIMAL(12,2))")
    value = {"grs_amt": gross, "dsc_amt": discount, "lin_amt": amount}
    con.executemany("INSERT INTO ivc_lin VALUES (?, DATE '2026-01-01' + CAST(? AS INTEGER), ?, ?, ?, ?, ?)",
                    [(i, i % 180, *(float(value[c][i]) for c in order), float(cost[i]),
                      round(float(amount[i]) - float(cost[i]), 2)) for i in range(n)])
    model = _learn(con)
    return {getattr(x.expr, "column", "").rsplit(".", 1)[-1]: x for x in model.measures.values()
            if getattr(x.expr, "agg", "") == "sum"}, model


def test_the_amount_less_its_cost_is_the_gross_profit_and_the_amount_the_revenue():
    m, _ = _invoices(with_discount=True)
    assert {"Gross profit", "Profit"} <= set(_words(m["mgn_amt"])), m["mgn_amt"].synonyms
    assert "Revenue" in _words(m["lin_amt"]) and "Revenue" not in _words(m["grs_amt"])


def test_each_word_goes_to_one_measure_and_never_to_one_when_another_is_called_so():
    m, model = _invoices(with_discount=True)
    holders = [x for x in model.measures.values() if "Revenue" in _words(x)]
    assert len(holders) == 1
    # An admin's own "Revenue" (or one the data already had) keeps the word; it is not given again.
    holders[0].synonyms = {}
    m["grs_amt"].business_name = "Revenue"
    name_by_difference(model)
    assert "Revenue" not in _words(m["lin_amt"])
    assert "Net sales" in _words(m["lin_amt"]), "the words nobody holds still go to it"


def test_the_catalog_the_ai_reads_carries_the_words():
    from core2.plan.catalog import catalog_text

    _, model = _invoices(with_discount=True)
    text = catalog_text(model)
    line = next(ln for ln in text.splitlines() if "Lin" in ln and "Revenue" in ln)
    assert "Net sales" in line


def test_the_catalog_says_which_amount_is_another_less_a_third():
    """"Show operating expense by profit centre" was the debit amount alone: on an account booked on the credit
    side it was nothing. The AI is told the net amount is debit less credit, and to total an account by it."""
    from core2.plan.catalog import catalog_text

    model, _ = _domain("finance", "warehouse")
    net = next(x for x in model.measures.values() if any(e.kind == "difference" for e in x.evidence))
    found = next(e for e in net.evidence if e.kind == "difference").data
    of, less = (next(y for y in model.measures.values() if y.table == net.table and
                     (getattr(y.expr, "column", None) or "").rsplit(".", 1)[-1].casefold() == found[k].casefold())
                     for k in ("of", "less"))
    line = next(ln for ln in catalog_text(model).splitlines() if ln.startswith(f"- {net.slug} |"))
    assert f"| is {of.slug} less {less.slug}" in line, line


@pytest.mark.parametrize("domain, style", [("finance", "warehouse"), ("retail", "descriptive"), ("retail", "pascal")])
def test_one_amount_less_another_is_read_one_way_only(domain, style):
    """Net = debit - credit holds, so credit = debit - net does too: only the net amount is the difference."""
    model, _ = _domain(domain, style)
    held = [x.business_name for x in model.measures.values() if any(e.kind == "difference" for e in x.evidence)]
    assert held and not any(w in " ".join(held).lower() for w in ("credit", "discount", "cost")), held


def test_the_amount_listed_before_what_it_is_made_of_is_still_the_revenue():
    """The profitability copy lists the line amount first, then the gross and the discount: read by column order
    alone, the discount would be "the gross less the line amount". What is taken away is named so."""
    m, _ = _invoices(with_discount=True, line_first=True)
    assert "Revenue" in _words(m["lin_amt"]), m["lin_amt"].synonyms
    assert {"Gross profit", "Profit"} <= set(_words(m["mgn_amt"]))
    assert not any(e.kind == "difference" for x in ("dsc_amt", "cst_amt") for e in m[x].evidence)


def test_a_figure_that_is_mostly_nought_is_no_difference_of_two_equal_amounts():
    """The purchase lines' rejected quantity (nought on 97% of rows) was read as "line amount less line amount CAD":
    the lines are all in dollars, the two amounts equal and their difference nought."""
    rng = np.random.default_rng(5)
    n = 400
    amount = np.round(rng.uniform(50, 900, n), 2)
    converted = amount
    rejected = np.where(rng.random(n) < 0.973, 0, rng.integers(1, 9, n)).astype(float)
    con = duckdb.connect()
    con.execute("CREATE TABLE po_lin (po_lin_id INTEGER, po_dt DATE, lin_amt DECIMAL(12,2), lin_cad_amt "
                "DECIMAL(12,2), rjt_qty DOUBLE)")
    con.executemany("INSERT INTO po_lin VALUES (?, DATE '2026-01-01' + CAST(? AS INTEGER), ?, ?, ?)",
                    [(i, i % 180, float(amount[i]), float(converted[i]), float(rejected[i])) for i in range(n)])
    model = _learn(con)
    assert not any(e.kind == "difference" for x in model.measures.values() for e in x.evidence), \
        [(x.business_name, e.detail) for x in model.measures.values() for e in x.evidence if e.kind == "difference"]
