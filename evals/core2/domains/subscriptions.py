"""Subscriptions: plans, subscriptions, a month-end MRR snapshot and invoices.

What this warehouse is built to test:

* monthly recurring revenue and seats as levels on a month-end snapshot (taken at
  the last month end of each period, never added up over months), and February 2025
  missing from the snapshot (a load that never ran: a gap, not a zero);
* subscriptions dated by their start and their end, where a subscription still
  running has no end date (empty, not a placeholder): "how many churned" counts
  only those that ended;
* trial subscriptions at zero MRR and a status where "CNL" marks a cancelled one;
* invoices in three currencies (USD, EUR, CAD): amounts must not be added across
  them; monthly plans invoiced each month, annual plans once a year;
* a customer reached from the snapshot directly and through the subscription;
* customer names with symbols and analysis words ("Bright & Co", "Acme #9 Labs",
  "Annual Reports Inc", "Top Shelf Analytics", "O'Leary Media");
* 300 customers listed, about 220 ever subscribed.

Snapshots run from January 2024 to May 2026, invoices from January 2024 to
12 June 2026; questions are asked on 15 June 2026.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from evals.core2.framework import DateTruth, Domain, JoinTruth, MeasureTruth, TableDef, Truth

TODAY = dt.date(2026, 6, 15)
FIRST_MONTH_END = dt.date(2024, 1, 31)
LAST_MONTH_END = dt.date(2026, 5, 31)
LAST_DAY = dt.date(2026, 6, 12)
MISSING_MONTH_END = dt.date(2025, 2, 28)

PLANS = [("Starter", "monthly", 29.0), ("Growth", "monthly", 99.0), ("Scale", "monthly", 299.0),
         ("Growth Annual", "annual", 89.0), ("Scale Annual", "annual", 269.0), ("Enterprise", "annual", 1200.0)]
NAMED = ["Bright & Co", "Acme #9 Labs", "Annual Reports Inc", "Top Shelf Analytics", "O'Leary Media",
         "Daily Grind Cafe", "North Star Freight", "Blue Fjord Software"]
_WORDS = ["Apex", "Birch", "Cedar", "Delta", "Ember", "Fable", "Garnet", "Harbor", "Iris", "Juniper", "Kestrel",
          "Lumen", "Maple", "Nimbus", "Orchid", "Pioneer", "Quarry", "Raven", "Summit", "Tundra"]
_KINDS = ["Systems", "Studio", "Partners", "Retail", "Logistics", "Health", "Foods", "Legal", "Design", "Energy",
          "Media", "Works", "Group", "Labs", "Holdings"]


def _month_ends(first: dt.date, last: dt.date) -> list[dt.date]:
    return [d.date() for d in pd.date_range(first, last, freq="ME")]


def build(seed: int = 7) -> Domain:
    rng = np.random.default_rng(seed)
    plans = pd.DataFrame({"plan_id": np.arange(1, len(PLANS) + 1), "plan_name": [p for p, _, _ in PLANS],
                          "billing_period": [b for _, b, _ in PLANS], "list_price": [x for _, _, x in PLANS]})

    names = list(NAMED)
    for i in range(300 - len(NAMED)):
        names.append(f"{_WORDS[i % len(_WORDS)]} {_KINDS[(i // len(_WORDS) + i) % len(_KINDS)]} {i // 20 + 1}")
    countries = rng.choice(["USA", "Canada", "Germany", "France", "UK"], 300, p=[0.45, 0.2, 0.15, 0.1, 0.1])
    currency_of = {"USA": "USD", "Canada": "CAD", "Germany": "EUR", "France": "EUR", "UK": "USD"}
    signup = pd.Timestamp("2023-06-01") + pd.to_timedelta(rng.integers(0, 1080 * 24 * 60, 300), unit="m")
    customers = pd.DataFrame({
        "customer_id": np.arange(1, 301), "customer_name": names, "country": countries,
        "segment": rng.choice(["SMB", "Mid-market", "Enterprise"], 300, p=[0.6, 0.3, 0.1]),
        "signup_ts": signup})

    # ── subscriptions: about 220 customers, some twice ─────────────────────────
    subscribers = rng.choice(np.arange(1, 301), 220, replace=False)
    subs = []
    for customer in subscribers:
        for k in range(1 if rng.random() < 0.8 else 2):
            start = (signup[customer - 1] + pd.Timedelta(days=int(rng.integers(0, 60)) + 400 * k)).date()
            if start > LAST_DAY:
                continue
            plan = int(rng.choice(np.arange(1, 7), p=[0.3, 0.25, 0.12, 0.15, 0.1, 0.08]))
            trial = rng.random() < 0.08
            end = None
            if rng.random() < 0.3:
                end = start + dt.timedelta(days=int(rng.integers(45, 700)))
                end = None if end > LAST_DAY else end
            subs.append({"subscription_id": len(subs) + 1, "subscription_number": f"SUB-{10000 + len(subs)}",
                         "customer_id": int(customer), "plan_id": plan, "start_date": start, "end_date": end,
                         "status": "TRL" if trial else ("CNL" if end else "ACT"),
                         "seats": int(rng.integers(1, 6) * (1 + (plan in (3, 5, 6)) * 4)), "trial": trial})
    subscriptions = pd.DataFrame(subs)

    snapshot = []
    for end_of_month in _month_ends(FIRST_MONTH_END, LAST_MONTH_END):
        if end_of_month == MISSING_MONTH_END:
            continue                                    # the load that never ran
        for s in subs:
            if s["start_date"] <= end_of_month and (s["end_date"] is None or s["end_date"] > end_of_month):
                price = PLANS[s["plan_id"] - 1][2]
                grown = 1 + 0.002 * ((end_of_month.year - 2024) * 12 + end_of_month.month)
                mrr = 0.0 if s["trial"] else round(price * s["seats"] * grown, 2)
                snapshot.append((end_of_month, s["subscription_id"], s["customer_id"], s["plan_id"], mrr, s["seats"]))
    mrr_monthly = pd.DataFrame(snapshot, columns=["month_end_date", "subscription_id", "customer_id", "plan_id",
                                                  "mrr", "seats"])

    invoices = []
    for s in subs:
        if s["trial"]:
            continue
        annual = PLANS[s["plan_id"] - 1][1] == "annual"
        price = PLANS[s["plan_id"] - 1][2] * s["seats"] * (12 * 0.85 if annual else 1)
        day = pd.Timestamp(s["start_date"])
        stop = pd.Timestamp(s["end_date"] or LAST_DAY)
        currency = currency_of[str(countries[s["customer_id"] - 1])]
        rate = {"USD": 1.0, "CAD": 1.35, "EUR": 0.92}[currency]
        while day <= stop and day.date() <= LAST_DAY:
            if day.date() >= dt.date(2024, 1, 1):
                amount = round(price * rate, 2)
                invoices.append((s["subscription_id"], s["customer_id"], day.date(), currency, amount,
                                 round(amount * (0.13 if currency == "CAD" else 0.2 if currency == "EUR" else 0.0), 2)))
            day = day + pd.DateOffset(years=1) if annual else day + pd.DateOffset(months=1)
    invoices_frame = pd.DataFrame(invoices, columns=["subscription_id", "customer_id", "invoice_date", "currency_code",
                                                     "amount", "tax_amount"])
    invoices_frame = invoices_frame.sort_values(["invoice_date", "subscription_id"]).reset_index(drop=True)
    invoices_frame.insert(0, "invoice_number", [f"INV-{500000 + i}" for i in range(len(invoices_frame))])
    invoices_frame.insert(0, "invoice_id", np.arange(1, len(invoices_frame) + 1))

    tables = [
        TableDef("plans", "dimension", plans,
                 {"plan_id": "INTEGER", "plan_name": "VARCHAR", "billing_period": "VARCHAR",
                  "list_price": "DECIMAL(10,2)"}, ["plan_id"], "Plan"),
        TableDef("customers", "dimension", customers,
                 {"customer_id": "INTEGER", "customer_name": "VARCHAR", "country": "VARCHAR", "segment": "VARCHAR",
                  "signup_ts": "TIMESTAMP"}, ["customer_id"], "Customer"),
        TableDef("subscriptions", "dimension", subscriptions.drop(columns=["seats", "trial"]),
                 {"subscription_id": "INTEGER", "subscription_number": "VARCHAR", "customer_id": "INTEGER",
                  "plan_id": "INTEGER", "start_date": "DATE", "end_date": "DATE", "status": "VARCHAR"},
                 ["subscription_id"], "Subscription"),
        TableDef("mrr_monthly", "snapshot", mrr_monthly,
                 {"month_end_date": "DATE", "subscription_id": "INTEGER", "customer_id": "INTEGER", "plan_id": "INTEGER",
                  "mrr": "DECIMAL(12,2)", "seats": "INTEGER"}, ["month_end_date", "subscription_id"], "MRR"),
        TableDef("invoices", "fact", invoices_frame,
                 {"invoice_id": "INTEGER", "invoice_number": "VARCHAR", "subscription_id": "INTEGER",
                  "customer_id": "INTEGER", "invoice_date": "DATE", "currency_code": "VARCHAR",
                  "amount": "DECIMAL(12,2)", "tax_amount": "DECIMAL(12,2)"}, ["invoice_id"], "Invoice"),
    ]

    truth = Truth(
        kinds={t.name: t.kind for t in tables},
        primary_keys={t.name: t.primary_key for t in tables},
        joins=[
            JoinTruth("subscriptions", "customer_id", "customers", "customer_id"),
            JoinTruth("subscriptions", "plan_id", "plans", "plan_id"),
            JoinTruth("mrr_monthly", "subscription_id", "subscriptions", "subscription_id"),
            JoinTruth("mrr_monthly", "customer_id", "customers", "customer_id"),
            JoinTruth("mrr_monthly", "plan_id", "plans", "plan_id"),
            JoinTruth("invoices", "subscription_id", "subscriptions", "subscription_id"),
            JoinTruth("invoices", "customer_id", "customers", "customer_id"),
        ],
        calendar=None,
        dates=[
            DateTruth("customers", "signup_ts", "Signup time", "event", False, default=True),
            DateTruth("subscriptions", "start_date", "Start date", "event", False, default=True),
            DateTruth("subscriptions", "end_date", "End date", "event", False),
            DateTruth("mrr_monthly", "month_end_date", "Month end date", "snapshot", False, default=True),
            DateTruth("invoices", "invoice_date", "Invoice date", "event", False, default=True),
        ],
        measures=[
            MeasureTruth("mrr_monthly", "mrr", "sum", "semi_additive", "currency", "MRR", time_aggregation="last"),
            MeasureTruth("mrr_monthly", "seats", "sum", "semi_additive", "integer", "Seats", time_aggregation="last"),
            MeasureTruth("invoices", "amount", "sum", "additive", "currency", "Amount"),
            MeasureTruth("invoices", "tax_amount", "sum", "additive", "currency", "Tax amount"),
            MeasureTruth("invoices", None, "count", "additive", "count", "Invoices"),
        ],
        labels={"plans": "plan_name", "customers": "customer_name"},
        codes={"subscriptions": "subscription_number"},
        statuses={"subscriptions.status": {"values": {"ACT": "Active", "CNL": "Cancelled", "TRL": "Trial"},
                                           "cancelled": ["CNL"]}},
        quality=[
            {"object": "invoices.amount", "kind": "unit_mix"},
            {"object": "customers", "kind": "listed_vs_active"},
        ],
        not_measures=["mrr_monthly.subscription_id", "mrr_monthly.customer_id", "mrr_monthly.plan_id",
                      "invoices.invoice_id", "invoices.subscription_id", "invoices.customer_id"],
    )
    return Domain("subscriptions", tables, truth, TODAY,
                  "A software company's plans, subscriptions, recurring revenue and invoices.",
                  abbreviations={"subscription": "SBSC", "invoice": "INV", "seats": "SEATS"})
