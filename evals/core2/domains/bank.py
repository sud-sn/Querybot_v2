"""Bank: a retail bank's lending, cards, transactions and complaints, with branch targets.

What this warehouse is built to test:

* a business whose names say a bank (loans, principal, collateral, deposits, branches), with customers,
  products and transactions beside them that alone would say a shop;
* loan applications with a decision (Approved, Declined, Withdrawn, Pending): an approval rate is a share of
  applications by their decision;
* loans with a principal, a collateral value, an outstanding principal and a status (Current, Paid off,
  30-89 days past due, Charged off), and a charge-off amount on the loans charged off;
* loan payments with the amount due, the amount paid (less than due on the late ones) and the days past due;
* cards with a credit limit and a status; transactions by type and channel; complaints with compensation;
* a monthly target per branch (loan volume, new accounts), keyed by a yyyymm period number.

Activity runs from 2 January 2024 to 12 June 2026; questions are asked on 15 June 2026. Invented data only.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from evals.core2.framework import Domain, TableDef, Truth

TODAY = dt.date(2026, 6, 15)
FIRST_DAY = dt.date(2024, 1, 2)
LAST_DAY = dt.date(2026, 6, 12)

REGIONS = ["North", "South", "East", "West"]
BRANCHES = [("Harbour Street", "North"), ("Mill Lane", "North"), ("Station Road", "South"),
            ("Market Square", "South"), ("Riverside", "East"), ("Old Town", "East"), ("Parkway", "West"),
            ("Bridge End", "West")]
SEGMENTS = ["Mass market", "Affluent", "Small business", "Private banking"]
PRODUCTS = [("Everyday Checking", "Checking", "Deposits"), ("Easy Saver", "Savings", "Deposits"),
            ("Fixed Term Deposit", "Term deposit", "Deposits"), ("Home Mortgage", "Mortgage", "Lending"),
            ("Personal Loan", "Personal loan", "Lending"), ("Auto Loan", "Auto loan", "Lending"),
            ("Classic Card", "Credit card", "Cards"), ("Gold Card", "Credit card", "Cards")]


def _dates(rng: np.random.Generator, n: int, first: dt.date = FIRST_DAY, last: dt.date = LAST_DAY) -> pd.Series:
    span = (last - first).days
    return pd.Series(pd.to_datetime(first) + pd.to_timedelta(rng.integers(0, span + 1, n), unit="D"))


def build(seed: int = 7) -> Domain:
    rng = np.random.default_rng(seed)
    branches = pd.DataFrame({"branch_id": np.arange(1, len(BRANCHES) + 1),
                             "branch_name": [b for b, _ in BRANCHES], "region": [r for _, r in BRANCHES]})
    customers = pd.DataFrame({"customer_id": np.arange(1, 1201),
                              "customer_code": [f"C{100000 + i}" for i in range(1, 1201)],
                              "customer_segment": rng.choice(SEGMENTS, 1200, p=[0.55, 0.2, 0.2, 0.05]),
                              "home_branch_id": rng.integers(1, len(BRANCHES) + 1, 1200)})
    products = pd.DataFrame({"product_id": np.arange(1, len(PRODUCTS) + 1),
                             "product_name": [p for p, _, _ in PRODUCTS], "product_type": [t for _, t, _ in PRODUCTS],
                             "product_family": [f for _, _, f in PRODUCTS]})

    # ── loan applications and the loans they became ──────────────────────────
    n_apps = 2400
    applied = _dates(rng, n_apps)
    decision = rng.choice(["Approved", "Declined", "Withdrawn", "Pending"], n_apps, p=[0.55, 0.3, 0.1, 0.05])
    pending = applied > pd.Timestamp(LAST_DAY) - pd.Timedelta(days=20)
    decision = np.where(pending, "Pending", np.where(decision == "Pending", "Approved", decision))
    lending = [i + 1 for i, (_, _, f) in enumerate(PRODUCTS) if f == "Lending"]
    applications = pd.DataFrame({
        "application_id": np.arange(1, n_apps + 1),
        "customer_id": rng.integers(1, 1201, n_apps),
        "branch_id": rng.integers(1, len(BRANCHES) + 1, n_apps),
        "product_id": rng.choice(lending, n_apps),
        "applied_date": applied.dt.date,
        "requested_amount": np.round(np.exp(rng.normal(10.5, 0.8, n_apps)), -2),
        "decision": decision,
    })
    approved = applications[applications["decision"] == "Approved"].reset_index(drop=True)
    n_loans = len(approved)
    funded = pd.to_datetime(approved["applied_date"]) + pd.to_timedelta(rng.integers(3, 15, n_loans), unit="D")
    funded = funded.where(funded <= pd.Timestamp(LAST_DAY), pd.Timestamp(LAST_DAY))
    principal = np.round(approved["requested_amount"].to_numpy() * rng.uniform(0.8, 1.0, n_loans), -2)
    status = rng.choice(["Current", "Paid off", "30-89 days past due", "Charged off"], n_loans,
                        p=[0.78, 0.12, 0.07, 0.03])
    charged = status == "Charged off"
    loans = pd.DataFrame({
        "loan_id": np.arange(1, n_loans + 1),
        "loan_number": [f"L-{500000 + i}" for i in range(1, n_loans + 1)],
        "application_id": approved["application_id"],
        "customer_id": approved["customer_id"],
        "branch_id": approved["branch_id"],
        "product_id": approved["product_id"],
        "funded_date": funded.dt.date,
        "principal_amount": principal,
        "collateral_value": np.round(principal * rng.uniform(1.05, 1.6, n_loans), -2),
        "interest_rate": np.round(rng.uniform(3.5, 11.5, n_loans), 2),
        "loan_status": status,
        "outstanding_principal": np.round(np.where(status == "Paid off", 0.0, principal * rng.uniform(0.2, 0.95, n_loans)), 2),
        "charge_off_amount": np.where(charged, np.round(principal * rng.uniform(0.3, 0.8, n_loans), 2), np.nan),
    })

    # ── monthly payments on each loan ────────────────────────────────────────
    rows = []
    for loan in loans.itertuples(index=False):
        due = pd.Timestamp(loan.funded_date) + pd.DateOffset(months=1)
        installment = round(loan.principal_amount / 36 * 1.04, 2)
        late_loan = loan.loan_status in ("30-89 days past due", "Charged off")
        k = 1
        while due <= pd.Timestamp(LAST_DAY) and k <= 36:
            late = 0 if not late_loan or rng.random() > 0.35 else int(rng.integers(5, 75))
            paid = installment if late < 31 else round(installment * float(rng.uniform(0.0, 0.6)), 2)
            rows.append((loan.loan_id, k, due.date(), (due + pd.Timedelta(days=late)).date(), installment, paid,
                         round(installment * 0.25, 2), 25.0 if late > 15 else 0.0, late))
            due += pd.DateOffset(months=1)
            k += 1
    payments = pd.DataFrame(rows, columns=["loan_id", "installment_number", "due_date", "paid_date", "amount_due",
                                           "amount_paid", "interest_paid", "late_fee", "days_past_due"])
    payments.insert(0, "payment_id", np.arange(1, len(payments) + 1))

    # ── cards, transactions, complaints ──────────────────────────────────────
    n_cards = 900
    cards = pd.DataFrame({
        "card_id": np.arange(1, n_cards + 1),
        "customer_id": rng.integers(1, 1201, n_cards),
        "product_id": rng.choice([7, 8], n_cards, p=[0.75, 0.25]),
        "issued_date": _dates(rng, n_cards).dt.date,
        "card_type": rng.choice(["Credit", "Debit"], n_cards, p=[0.6, 0.4]),
        "credit_limit": np.round(rng.choice([2000, 5000, 7500, 10000, 15000], n_cards), 2),
        "card_status": rng.choice(["Active", "Closed", "Blocked"], n_cards, p=[0.85, 0.12, 0.03]),
    })
    n_tx = 20000
    tx_type = rng.choice(["Card purchase", "Deposit", "Withdrawal", "Transfer in", "Transfer out"], n_tx,
                         p=[0.45, 0.15, 0.15, 0.12, 0.13])
    transactions = pd.DataFrame({
        "transaction_id": np.arange(1, n_tx + 1),
        "customer_id": rng.integers(1, 1201, n_tx),
        "branch_id": rng.integers(1, len(BRANCHES) + 1, n_tx),
        "posted_date": _dates(rng, n_tx).dt.date,
        "transaction_type": tx_type,
        "channel": rng.choice(["Card", "Mobile", "Online", "ATM", "Branch"], n_tx),
        "amount": np.round(np.exp(rng.normal(4.0, 1.1, n_tx)), 2),
        "direction": np.where(np.isin(tx_type, ["Deposit", "Transfer in"]), "Credit", "Debit"),
    })
    n_complaints = 600
    opened = _dates(rng, n_complaints)
    complaints = pd.DataFrame({
        "complaint_id": np.arange(1, n_complaints + 1),
        "customer_id": rng.integers(1, 1201, n_complaints),
        "branch_id": rng.integers(1, len(BRANCHES) + 1, n_complaints),
        "opened_date": opened.dt.date,
        "complaint_category": rng.choice(["Fees", "Card", "Online banking", "Branch service", "Fraud"], n_complaints),
        "resolved_date": (opened + pd.to_timedelta(rng.integers(1, 40, n_complaints), unit="D")).dt.date,
        "compensation_paid": np.round(np.where(rng.random(n_complaints) < 0.3, rng.uniform(10, 400, n_complaints), 0.0), 2),
    })

    # ── a monthly target per branch ──────────────────────────────────────────
    periods = pd.period_range("2024-01", "2026-12", freq="M")
    targets = pd.DataFrame([(b, int(p.year * 100 + p.month), float(round(rng.uniform(150_000, 400_000), -3)),
                             int(rng.integers(20, 60)))
                            for b in range(1, len(BRANCHES) + 1) for p in periods],
                           columns=["branch_id", "period_key", "loan_volume_target", "new_accounts_target"])

    tables = [
        TableDef("branches", "dimension", branches, {"branch_id": "INTEGER", "branch_name": "VARCHAR",
                                                     "region": "VARCHAR"}, ["branch_id"], "Branch"),
        TableDef("customers", "dimension", customers, {"customer_id": "INTEGER", "customer_code": "VARCHAR",
                                                       "customer_segment": "VARCHAR", "home_branch_id": "INTEGER"},
                 ["customer_id"], "Customer"),
        TableDef("products", "dimension", products, {"product_id": "INTEGER", "product_name": "VARCHAR",
                                                     "product_type": "VARCHAR", "product_family": "VARCHAR"},
                 ["product_id"], "Product"),
        TableDef("loan_applications", "fact", applications,
                 {"application_id": "INTEGER", "customer_id": "INTEGER", "branch_id": "INTEGER", "product_id": "INTEGER",
                  "applied_date": "DATE", "requested_amount": "DECIMAL(14,2)", "decision": "VARCHAR"},
                 ["application_id"], "Loan application"),
        TableDef("loans", "fact", loans,
                 {"loan_id": "INTEGER", "loan_number": "VARCHAR", "application_id": "INTEGER", "customer_id": "INTEGER",
                  "branch_id": "INTEGER", "product_id": "INTEGER", "funded_date": "DATE",
                  "principal_amount": "DECIMAL(14,2)", "collateral_value": "DECIMAL(14,2)",
                  "interest_rate": "DECIMAL(6,2)", "loan_status": "VARCHAR", "outstanding_principal": "DECIMAL(14,2)",
                  "charge_off_amount": "DECIMAL(14,2)"}, ["loan_id"], "Loan"),
        TableDef("loan_payments", "fact", payments,
                 {"payment_id": "INTEGER", "loan_id": "INTEGER", "installment_number": "INTEGER", "due_date": "DATE",
                  "paid_date": "DATE", "amount_due": "DECIMAL(14,2)", "amount_paid": "DECIMAL(14,2)",
                  "interest_paid": "DECIMAL(14,2)", "late_fee": "DECIMAL(10,2)", "days_past_due": "INTEGER"},
                 ["payment_id"], "Loan payment"),
        TableDef("cards", "fact", cards,
                 {"card_id": "INTEGER", "customer_id": "INTEGER", "product_id": "INTEGER", "issued_date": "DATE",
                  "card_type": "VARCHAR", "credit_limit": "DECIMAL(12,2)", "card_status": "VARCHAR"},
                 ["card_id"], "Card"),
        TableDef("transactions", "fact", transactions,
                 {"transaction_id": "INTEGER", "customer_id": "INTEGER", "branch_id": "INTEGER", "posted_date": "DATE",
                  "transaction_type": "VARCHAR", "channel": "VARCHAR", "amount": "DECIMAL(14,2)",
                  "direction": "VARCHAR"}, ["transaction_id"], "Transaction"),
        TableDef("complaints", "fact", complaints,
                 {"complaint_id": "INTEGER", "customer_id": "INTEGER", "branch_id": "INTEGER", "opened_date": "DATE",
                  "complaint_category": "VARCHAR", "resolved_date": "DATE", "compensation_paid": "DECIMAL(10,2)"},
                 ["complaint_id"], "Complaint"),
        TableDef("branch_targets", "fact", targets,
                 {"branch_id": "INTEGER", "period_key": "INTEGER", "loan_volume_target": "DECIMAL(14,2)",
                  "new_accounts_target": "INTEGER"}, ["branch_id", "period_key"], "Branch target"),
    ]
    truth = Truth(kinds={t.name: t.kind for t in tables}, primary_keys={t.name: t.primary_key for t in tables},
                  joins=[], calendar=None, dates=[], measures=[],
                  labels={"branches": "branch_name", "products": "product_name"})
    return Domain("bank", tables, truth, TODAY, "A retail bank's lending, cards, transactions and complaints.")
