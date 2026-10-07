"""Finance: a general ledger with a monthly budget, an account hierarchy and a July fiscal year.

What this warehouse is built to test:

* a fiscal year starting in July, named by the year it ends (fiscal 2026 = July 2025
  to June 2026), and fiscal quarters;
* journal lines dated two ways: the posting date (the default) and the document date
  (an invoice dated up to three weeks before it was posted), plus the time the line
  was entered (an audit stamp written by three posting runs a day; fiscal 2024 was
  entered in one batch in July 2024);
* a signed net amount: debits positive, credits negative, so revenue sums negative
  and a journal sums to zero; debit and credit columns beside it, never negative;
* lines in two currencies (CAD and USD): amounts must not be added across them;
* a status where "REV" marks a reversed journal (3%), kept as rows;
* an account hierarchy: each account points at its parent account in the same table
  (accounts are keyed by their number, 1000 to 6190);
* cost centres in divisions (a snowflaked dimension), with names that hold symbols
  and analysis words ("R&D Lab #2", "Sales & Marketing", "O'Neil Studio",
  "Monthly Reporting Unit");
* a monthly budget per cost centre and account, keyed by a yyyymm period number and
  set through June 2027 (the end of fiscal 2027);
* 120 accounts listed, about 40 ever posted to.

Journal lines run from 3 July 2023 to 12 June 2026; questions are asked on 15 June 2026.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from evals.core2.framework import (
    CALENDAR_ATTRIBUTES,
    CALENDAR_TYPES,
    DateTruth,
    Domain,
    JoinTruth,
    MeasureTruth,
    TableDef,
    Truth,
    calendar_frame,
)

TODAY = dt.date(2026, 6, 15)
FIRST_DAY = dt.date(2023, 7, 3)
LAST_DAY = dt.date(2026, 6, 12)

DIVISIONS = ["Corporate", "Operations", "Commercial", "Research"]
COST_CENTRES = [("CC-101", "Finance Office", 1), ("CC-102", "Legal & Compliance", 1), ("CC-103", "People Services", 1),
                ("CC-201", "Plant North", 2), ("CC-202", "Plant South", 2), ("CC-203", "Logistics Hub", 2),
                ("CC-301", "Sales & Marketing", 3), ("CC-302", "Key Accounts", 3), ("CC-303", "O'Neil Studio", 3),
                ("CC-401", "R&D Lab #2", 4), ("CC-402", "Product Design", 4), ("CC-403", "Monthly Reporting Unit", 4)]
# Top-level groups (no parent) and the accounts under them.
GROUPS = [(1000, "Assets", "Asset"), (2000, "Liabilities", "Liability"), (3000, "Equity", "Equity"),
          (4000, "Revenue", "Revenue"), (5000, "Cost of Sales", "Expense"), (6000, "Operating Expenses", "Expense")]
_CHILDREN = {
    "Assets": ["Cash at Bank", "Petty Cash", "Accounts Receivable", "Inventory Asset", "Prepaid Expenses",
               "Equipment", "Vehicles", "Buildings", "Accumulated Depreciation", "Short-term Deposits"],
    "Liabilities": ["Accounts Payable", "Accrued Liabilities", "Sales Tax Payable", "Payroll Liabilities",
                    "Deferred Revenue", "Credit Card Payable", "Long-term Loan", "Lease Liability"],
    "Equity": ["Share Capital", "Retained Earnings", "Owner Contributions", "Dividends Declared"],
    "Revenue": ["Product Sales", "Service Revenue", "Subscription Revenue", "Freight Recovered", "Interest Income",
                "Rental Income", "Licensing Revenue", "Consulting Fees", "Installation Revenue", "Warranty Revenue"],
    "Cost of Sales": ["Materials", "Direct Labour", "Freight In", "Subcontractors", "Packaging", "Warranty Costs",
                      "Production Supplies", "Quality Testing"],
    "Operating Expenses": ["Salaries", "Benefits", "Rent", "Utilities", "Travel", "Meals & Entertainment",
                           "Software Subscriptions", "Advertising", "Professional Fees", "Insurance", "Training",
                           "Office Supplies", "Telephone", "Bank Charges", "Depreciation Expense", "Repairs",
                           "Recruiting", "Conferences", "Postage & Courier", "Cleaning"],
}
_FILLER = ["Reserve", "Clearing", "Suspense", "Adjustment", "Transfer", "Allocation", "Holding", "Intercompany"]


def _key(days: pd.DatetimeIndex) -> np.ndarray:
    return (days.year * 10000 + days.month * 100 + days.day).to_numpy().astype("int64")


def build(seed: int = 7) -> Domain:
    rng = np.random.default_rng(seed)
    calendar = calendar_frame(dt.date(2023, 1, 1), dt.date(2027, 12, 31), fiscal_start_month=7)

    divisions = pd.DataFrame({"division_id": np.arange(1, len(DIVISIONS) + 1), "division_name": DIVISIONS})
    cost_centres = pd.DataFrame({
        "cost_centre_id": np.arange(1, len(COST_CENTRES) + 1),
        "cost_centre_code": [c for c, _, _ in COST_CENTRES],
        "cost_centre_name": [n for _, n, _ in COST_CENTRES],
        "division_id": [d for _, _, d in COST_CENTRES],
    })

    rows = []
    group_ids: dict[str, int] = {}
    for code, name, kind in GROUPS:
        rows.append((code, str(code), name, kind, None))
        group_ids[name] = code
    for code, name, kind in GROUPS:
        children = list(_CHILDREN[name])
        n = 0
        while len(children) < 19:
            children.append(f"{name} {_FILLER[n % len(_FILLER)]} {n // len(_FILLER) + 1}")
            n += 1
        for i, child in enumerate(children[:19]):
            rows.append((code + 10 * (i + 1), str(code + 10 * (i + 1)), child, kind, group_ids[name]))
    accounts = pd.DataFrame(rows, columns=["account_id", "account_code", "account_name", "account_type",
                                           "parent_account_id"])
    accounts["parent_account_id"] = accounts["parent_account_id"].astype("Int64")

    def used(group: str, n: int) -> np.ndarray:
        ids = accounts.loc[accounts["parent_account_id"] == group_ids[group], "account_id"].to_numpy()
        return ids[:n]

    revenue_accounts, cogs_accounts = used("Revenue", 9), used("Cost of Sales", 8)
    opex_accounts = used("Operating Expenses", 20)
    cash, receivable, payable = (int(accounts.loc[accounts["account_name"] == n, "account_id"].iloc[0])
                                 for n in ("Cash at Bank", "Accounts Receivable", "Accounts Payable"))
    commercial = cost_centres.loc[cost_centres["division_id"] == 3, "cost_centre_id"].to_numpy()

    # ── journals: each two lines that net to zero ─────────────────────────────
    days = pd.bdate_range(FIRST_DAY, LAST_DAY)
    month_index = (days.year - FIRST_DAY.year) * 12 + days.month - FIRST_DAY.month
    per_day = rng.poisson(5.5 * (1 + 0.01 * month_index) * np.where(days.month == 12, 1.3, 1.0))
    posting = days.repeat(per_day)
    n = len(posting)
    revenue = rng.random(n) < 0.45
    amount = np.round(np.exp(rng.normal(7.4, 1.0, n)), 2)
    amount = np.where(revenue, amount * 1.6, amount)
    cost_centre = np.where(revenue, rng.choice(commercial, n), rng.integers(1, len(COST_CENTRES) + 1, n))
    expense_account = np.where(rng.random(n) < 0.35, rng.choice(cogs_accounts, n), rng.choice(opex_accounts, n))
    debit_account = np.where(revenue, np.where(rng.random(n) < 0.7, receivable, cash), expense_account)
    credit_account = np.where(revenue, rng.choice(revenue_accounts, n), np.where(rng.random(n) < 0.6, payable, cash))
    currency = np.where(rng.random(n) < 0.15, "USD", "CAD")
    status = np.where(rng.random(n) < 0.03, "REV", "POST")
    document = posting - pd.to_timedelta(rng.integers(0, 21, n), unit="D")
    runs = np.array([6 * 60, 12 * 60 + 30, 18 * 60])        # three posting runs a day
    entered = posting + pd.to_timedelta(rng.integers(0, 3, n), unit="D") + pd.to_timedelta(
        rng.choice(runs, n), unit="m")
    batch = posting < pd.Timestamp(2024, 7, 1)
    entered = entered.where(~batch, pd.Timestamp(2024, 7, 15, 6))      # the back-fill ran as the 06:00 run

    journal = np.arange(n)
    line_journal = np.repeat(journal, 2)
    debit_line = np.tile([True, False], n)
    amount2 = np.repeat(amount, 2)
    journal_lines = pd.DataFrame({
        "journal_line_id": np.arange(1, 2 * n + 1),
        "journal_number": np.array([f"JE-{100000 + j}" for j in journal])[line_journal],
        "line_number": np.tile([1, 2], n),
        "account_id": np.where(debit_line, np.repeat(debit_account, 2), np.repeat(credit_account, 2)),
        "cost_centre_id": np.repeat(cost_centre, 2),
        "posting_date_key": np.repeat(_key(posting), 2),
        "document_date_key": np.repeat(_key(document), 2),
        "entered_at": np.repeat(entered.to_numpy(), 2),
        "debit_amount": np.where(debit_line, amount2, 0.0),
        "credit_amount": np.where(debit_line, 0.0, amount2),
        "net_amount": np.where(debit_line, amount2, -amount2),
        "currency_code": np.repeat(currency, 2),
        "status": np.repeat(status, 2),
    })

    # ── monthly budget per cost centre and the P&L accounts it uses, to June 2027 ──
    pl = journal_lines[journal_lines["account_id"].isin(np.concatenate([revenue_accounts, cogs_accounts,
                                                                       opex_accounts]))]
    pairs = pl.groupby(["cost_centre_id", "account_id"])["debit_amount"].count().reset_index()[
        ["cost_centre_id", "account_id"]]
    monthly = pl.assign(amount=pl["debit_amount"] + pl["credit_amount"]).groupby(
        ["cost_centre_id", "account_id"])["amount"].sum() / (month_index.max() + 1)
    periods = pd.period_range("2023-07", "2027-06", freq="M")
    budget_rows = []
    for cc, acc in pairs.itertuples(index=False):
        base = float(monthly.get((cc, acc), 0.0))
        for p in periods:
            budget_rows.append((int(cc), int(acc), int(p.year * 100 + p.month), float(round(base * rng.uniform(0.9, 1.15), -1))))
    budgets = pd.DataFrame(budget_rows, columns=["cost_centre_id", "account_id", "period_key", "budget_amount"])

    tables = [
        TableDef("calendar", "calendar", calendar, CALENDAR_TYPES, ["date_key"], "Calendar"),
        TableDef("divisions", "dimension", divisions, {"division_id": "INTEGER", "division_name": "VARCHAR"},
                 ["division_id"], "Division"),
        TableDef("cost_centres", "dimension", cost_centres,
                 {"cost_centre_id": "INTEGER", "cost_centre_code": "VARCHAR", "cost_centre_name": "VARCHAR",
                  "division_id": "INTEGER"}, ["cost_centre_id"], "Cost centre"),
        TableDef("accounts", "dimension", accounts,
                 {"account_id": "INTEGER", "account_code": "VARCHAR", "account_name": "VARCHAR",
                  "account_type": "VARCHAR", "parent_account_id": "INTEGER"}, ["account_id"], "Account"),
        TableDef("journal_lines", "fact", journal_lines,
                 {"journal_line_id": "INTEGER", "journal_number": "VARCHAR", "line_number": "INTEGER",
                  "account_id": "INTEGER", "cost_centre_id": "INTEGER", "posting_date_key": "INTEGER",
                  "document_date_key": "INTEGER", "entered_at": "TIMESTAMP", "debit_amount": "DECIMAL(14,2)",
                  "credit_amount": "DECIMAL(14,2)", "net_amount": "DECIMAL(14,2)", "currency_code": "VARCHAR",
                  "status": "VARCHAR"}, ["journal_line_id"], "Journal line"),
        TableDef("budgets", "fact", budgets,
                 {"cost_centre_id": "INTEGER", "account_id": "INTEGER", "period_key": "INTEGER",
                  "budget_amount": "DECIMAL(14,2)"}, ["cost_centre_id", "account_id", "period_key"], "Budget"),
    ]

    truth = Truth(
        kinds={t.name: t.kind for t in tables},
        primary_keys={t.name: t.primary_key for t in tables},
        joins=[
            JoinTruth("journal_lines", "account_id", "accounts", "account_id"),
            JoinTruth("journal_lines", "cost_centre_id", "cost_centres", "cost_centre_id"),
            JoinTruth("journal_lines", "posting_date_key", "calendar", "date_key", "Posting date"),
            JoinTruth("journal_lines", "document_date_key", "calendar", "date_key", "Document date"),
            JoinTruth("accounts", "parent_account_id", "accounts", "account_id", "Parent account"),
            JoinTruth("cost_centres", "division_id", "divisions", "division_id"),
            JoinTruth("budgets", "cost_centre_id", "cost_centres", "cost_centre_id"),
            JoinTruth("budgets", "account_id", "accounts", "account_id"),
        ],
        calendar={"table": "calendar", "key": "date_key", "date": "full_date", "attributes": CALENDAR_ATTRIBUTES,
                  "fiscal_year_start_month": 7, "fiscal_year_named_by": "end", "placeholders": [-1]},
        dates=[
            DateTruth("journal_lines", "posting_date_key", "Posting date", "event", True, default=True),
            DateTruth("journal_lines", "document_date_key", "Document date", "event", True),
            DateTruth("journal_lines", "entered_at", "Entered at", "audit", False),
            DateTruth("budgets", "period_key", "Period", "event", False, default=True),
        ],
        measures=[
            MeasureTruth("journal_lines", "debit_amount", "sum", "additive", "currency", "Debit amount"),
            MeasureTruth("journal_lines", "credit_amount", "sum", "additive", "currency", "Credit amount"),
            MeasureTruth("journal_lines", "net_amount", "sum", "additive", "currency", "Net amount"),
            MeasureTruth("journal_lines", None, "count", "additive", "count", "Journal lines"),
            MeasureTruth("journal_lines", "journal_number", "count_distinct", "additive", "count", "Journals"),
            MeasureTruth("budgets", "budget_amount", "sum", "additive", "currency", "Budget amount"),
        ],
        labels={"accounts": "account_name", "cost_centres": "cost_centre_name", "divisions": "division_name"},
        codes={"accounts": "account_code", "cost_centres": "cost_centre_code"},
        statuses={"journal_lines.status": {"values": {"POST": "Posted", "REV": "Reversed"}, "cancelled": ["REV"]}},
        quality=[
            {"object": "journal_lines.status", "kind": "status_column"},
            {"object": "journal_lines.entered_at", "kind": "load_timestamp"},
            {"object": "accounts", "kind": "listed_vs_active"},
            {"object": "journal_lines.net_amount", "kind": "unit_mix"},
        ],
        not_measures=["journal_lines.journal_line_id", "journal_lines.line_number", "journal_lines.account_id",
                      "journal_lines.cost_centre_id", "journal_lines.posting_date_key",
                      "journal_lines.document_date_key", "budgets.cost_centre_id", "budgets.account_id",
                      "budgets.period_key", "accounts.parent_account_id", "cost_centres.division_id",
                      "calendar.year", "calendar.fiscal_year"],
    )
    return Domain("finance", tables, truth, TODAY,
                  "A manufacturer's general ledger and monthly budget, with a July fiscal year.",
                  abbreviations={"posting": "PST", "document": "DOC"})
