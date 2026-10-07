"""Purchasing: purchase order lines and goods receipts, dated by plain DATE columns.

What this warehouse is built to test:

* no calendar table: every date is a DATE column (weeks, months and quarters are
  worked out from the date itself);
* order lines dated two ways: the order date (the default) and the promised date
  (a planned date, running up to three months into the future for open orders: never
  the default, never a snapshot period);
* receipts dated by the received date, plus a nightly load stamp written at 02:15
  the day after (a row-write time, never a business date);
* a purchase order number on each line (orders are counted as distinct numbers);
* a status where "X" marks a cancelled line (5%), kept with its amounts;
* receipts that point at the order line they fill (a fact-to-fact link) and at the
  item, so the item is reached two ways;
* a unit price that must be averaged, quantities in four units of measure;
* a data-entry outlier: one line in November 2025 typed with a quantity a thousand
  times too large;
* 0.4% of lines for a supplier missing from the supplier list (kept in every total,
  as an unknown supplier);
* supplier names with symbols and analysis words ("Smith & Sons Ltd", "Acme #4
  Components", "Daily Freight Co", "Top Quality Plastics", "O'Hara Metals");
* 80 items listed, about 55 ever ordered.

Order lines run from 2 January 2024 to 12 June 2026; questions are asked on 15 June 2026.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from evals.core2.framework import DateTruth, Domain, JoinTruth, MeasureTruth, TableDef, Truth

TODAY = dt.date(2026, 6, 15)
FIRST_DAY = dt.date(2024, 1, 2)
LAST_DAY = dt.date(2026, 6, 12)

SUPPLIERS = [("Smith & Sons Ltd", "Canada"), ("Acme #4 Components", "USA"), ("Daily Freight Co", "Canada"),
             ("Top Quality Plastics", "Mexico"), ("O'Hara Metals", "USA"), ("Northern Fasteners", "Canada"),
             ("Lakeside Chemicals", "Canada"), ("Rhein Precision GmbH", "Germany"), ("Pacific Packaging", "USA"),
             ("Monterrey Steel", "Mexico"), ("Maple Electrical Supply", "Canada"), ("Great Plains Lubricants", "USA"),
             ("Shenzhen Circuit Works", "China"), ("Valley Safety Gear", "Canada"), ("Granite Tooling", "USA"),
             ("Atlas Industrial Gases", "Canada"), ("Bluewater Pumps", "USA"), ("Prairie Pallets", "Canada"),
             ("Saxony Bearings", "Germany"), ("Coastal Abrasives", "Mexico")]
BUYERS = ["Plant Buying Team", "MRO Desk", "Packaging Desk", "Capital Projects", "Indirect Spend"]
CATEGORIES = {"Raw Materials": "KG", "Packaging": "BOX", "MRO": "EA", "Electrical": "EA", "Chemicals": "L"}
_ITEM_WORDS = {
    "Raw Materials": ["Steel Plate 4mm", "Aluminium Bar", "Copper Rod", "Resin Pellets", "Steel Coil", "Brass Sheet",
                      "Zinc Ingot", "Nylon Granules", "Stainless Tube", "Carbon Fibre Roll", "Polymer Blend", "Iron Billet",
                      "Glass Beads", "Rubber Compound", "Titanium Wire", "Lead Shot"],
    "Packaging": ["Carton 40x30", "Pallet Wrap", "Bubble Roll", "Corner Guards", "Foam Inserts", "Tape 48mm",
                  "Kraft Paper", "Strapping Band", "Label Stock", "Crate Small", "Crate Large", "Desiccant Packs",
                  "Shrink Film", "Void Fill", "Mailer Bags", "Drum Liners"],
    "MRO": ["#10 Hex Bolt", "Safety Gloves", "Grinding Disc", "Drill Bit Set", "Shop Towels", "Hydraulic Hose",
            "Bearing 6204", "Air Filter", "V-Belt A42", "Cutting Oil Pump", "Work Light", "Ear Plugs",
            "Torque Wrench", "Welding Rods", "Hand Cleaner", "Spill Kit"],
    "Electrical": ["Cable 2.5mm", "Contactor 25A", "Fuse 10A", "Junction Box", "LED Panel", "Motor Starter",
                   "Relay 24V", "Sensor M12", "Terminal Block", "Breaker 16A", "Conduit 20mm", "Cable Ties",
                   "Power Supply 24V", "Limit Switch", "Push Button", "Signal Tower"],
    "Chemicals": ["Degreaser", "Coolant Concentrate", "Hydraulic Oil", "Paint Thinner", "Epoxy Resin", "Acetone",
                  "Rust Inhibitor", "Primer Grey", "Isopropyl Alcohol", "Release Agent", "Floor Sealer",
                  "Descaler", "Adhesive Spray", "Lubricant Grease", "Antifreeze", "Cleaning Solvent"],
}
PRICE = {"Raw Materials": 6.0, "Packaging": 18.0, "MRO": 24.0, "Electrical": 42.0, "Chemicals": 9.0}


def build(seed: int = 7) -> Domain:
    rng = np.random.default_rng(seed)

    suppliers = pd.DataFrame({
        "supplier_id": np.arange(1, len(SUPPLIERS) + 1),
        "supplier_name": [n for n, _ in SUPPLIERS],
        "country": [c for _, c in SUPPLIERS],
        "payment_terms_days": rng.choice([30, 45, 60, 90], len(SUPPLIERS)),
    })
    buyers = pd.DataFrame({"buyer_id": np.arange(1, len(BUYERS) + 1), "buyer_name": BUYERS})
    rows = []
    for category, unit in CATEGORIES.items():
        for name in _ITEM_WORDS[category]:
            rows.append((len(rows) + 1, f"IT-{1001 + len(rows)}", name, category, unit))
    items = pd.DataFrame(rows, columns=["item_id", "item_code", "item_name", "category", "unit_of_measure"])
    bought = np.sort(rng.choice(items["item_id"].to_numpy(), 55, replace=False))
    item_category = items.set_index("item_id")["category"]

    # ── order lines: purchase orders of one to four lines ─────────────────────
    days = pd.bdate_range(FIRST_DAY, LAST_DAY)
    per_day = rng.poisson(3.2 * (1 + 0.012 * np.arange(len(days)) / 21), len(days))
    order_day = days.repeat(per_day)
    orders = len(order_day)
    lines_per_order = rng.integers(1, 5, orders)
    line_order = np.repeat(np.arange(orders), lines_per_order)
    n = len(line_order)
    order_date = order_day[line_order]
    supplier = rng.integers(1, len(SUPPLIERS) + 1, orders)[line_order]
    supplier = np.where(rng.random(n) < 0.004, 999, supplier)              # a supplier no longer listed
    buyer = rng.integers(1, len(BUYERS) + 1, orders)[line_order]
    item = rng.choice(bought, n)
    category = item_category.loc[item].to_numpy()
    base_price = np.array([PRICE[c] for c in category]) * (1 + (item % 7) / 10)
    unit_price = np.round(base_price * rng.uniform(0.92, 1.08, n), 2)
    qty = np.maximum(1, np.round(np.exp(rng.normal(3.6, 0.8, n)))).astype("int64")
    outlier = int(np.flatnonzero((order_date.year == 2025) & (order_date.month == 11))[17])
    qty[outlier] = qty[outlier] * 1000                                     # typed a thousand times too large
    lead = rng.integers(5, 61, n)
    promised = order_date + pd.to_timedelta(lead, unit="D")
    cancelled = rng.random(n) < 0.05
    due = promised.to_numpy() > np.datetime64(LAST_DAY - dt.timedelta(days=7))
    status = np.where(cancelled, "X", np.where(due, "O", "C"))
    order_lines = pd.DataFrame({
        "po_line_id": np.arange(1, n + 1),
        "po_number": np.array([f"PO-{d.year % 100:02d}-{i:05d}" for i, d in enumerate(order_day)])[line_order],
        "line_number": np.concatenate([np.arange(1, k + 1) for k in lines_per_order]),
        "supplier_id": supplier,
        "buyer_id": buyer,
        "item_id": item,
        "order_date": order_date.date,
        "promised_date": promised.date,
        "ordered_qty": qty,
        "unit_price": unit_price,
        "line_amount": np.round(qty * unit_price, 2),
        "status": status,
    })

    # ── receipts: closed lines received in one or two deliveries; open lines partly ──
    received = []
    for line in order_lines[order_lines["status"] != "X"].itertuples(index=False):
        if line.status == "O" and rng.random() < 0.6:
            continue
        parts = 1 if rng.random() < 0.75 else 2
        left = int(line.ordered_qty) if line.status == "C" else int(line.ordered_qty) // 2
        when = pd.Timestamp(line.promised_date) + pd.Timedelta(days=int(rng.integers(-6, 9)))
        for k in range(parts):
            qty_k = left if k == parts - 1 else left // 2
            left -= qty_k
            day = min(when + pd.Timedelta(days=7 * k), pd.Timestamp(LAST_DAY))
            day = max(day, pd.Timestamp(line.order_date) + pd.Timedelta(days=1))
            if qty_k > 0:
                received.append((line.po_line_id, line.item_id, day.date(), qty_k))
    receipts = pd.DataFrame(received, columns=["po_line_id", "item_id", "received_date", "received_qty"])
    receipts = receipts[receipts["received_date"] <= LAST_DAY].reset_index(drop=True)
    receipts.insert(0, "receipt_id", np.arange(1, len(receipts) + 1))
    receipts["rejected_qty"] = np.where(rng.random(len(receipts)) < 0.06,
                                        np.ceil(receipts["received_qty"] * rng.uniform(0.01, 0.1, len(receipts))), 0
                                        ).astype("int64")
    receipts["loaded_at"] = pd.to_datetime(receipts["received_date"]) + pd.Timedelta(days=1, hours=2, minutes=15)

    tables = [
        TableDef("suppliers", "dimension", suppliers,
                 {"supplier_id": "INTEGER", "supplier_name": "VARCHAR", "country": "VARCHAR",
                  "payment_terms_days": "INTEGER"}, ["supplier_id"], "Supplier"),
        TableDef("buyers", "dimension", buyers, {"buyer_id": "INTEGER", "buyer_name": "VARCHAR"}, ["buyer_id"],
                 "Buyer"),
        TableDef("items", "dimension", items,
                 {"item_id": "INTEGER", "item_code": "VARCHAR", "item_name": "VARCHAR", "category": "VARCHAR",
                  "unit_of_measure": "VARCHAR"}, ["item_id"], "Item"),
        TableDef("purchase_order_lines", "fact", order_lines,
                 {"po_line_id": "INTEGER", "po_number": "VARCHAR", "line_number": "INTEGER", "supplier_id": "INTEGER",
                  "buyer_id": "INTEGER", "item_id": "INTEGER", "order_date": "DATE", "promised_date": "DATE",
                  "ordered_qty": "INTEGER", "unit_price": "DECIMAL(10,2)", "line_amount": "DECIMAL(14,2)",
                  "status": "VARCHAR"}, ["po_line_id"], "Purchase order line"),
        TableDef("receipts", "fact", receipts,
                 {"receipt_id": "INTEGER", "po_line_id": "INTEGER", "item_id": "INTEGER", "received_date": "DATE",
                  "received_qty": "INTEGER", "rejected_qty": "INTEGER", "loaded_at": "TIMESTAMP"},
                 ["receipt_id"], "Receipt"),
    ]

    truth = Truth(
        kinds={t.name: t.kind for t in tables},
        primary_keys={t.name: t.primary_key for t in tables},
        joins=[
            JoinTruth("purchase_order_lines", "supplier_id", "suppliers", "supplier_id"),
            JoinTruth("purchase_order_lines", "buyer_id", "buyers", "buyer_id"),
            JoinTruth("purchase_order_lines", "item_id", "items", "item_id"),
            JoinTruth("receipts", "po_line_id", "purchase_order_lines", "po_line_id"),
            JoinTruth("receipts", "item_id", "items", "item_id"),
        ],
        calendar=None,
        dates=[
            DateTruth("purchase_order_lines", "order_date", "Order date", "event", False, default=True),
            DateTruth("purchase_order_lines", "promised_date", "Promised date", "event", False),
            DateTruth("receipts", "received_date", "Received date", "event", False, default=True),
            DateTruth("receipts", "loaded_at", "Loaded at", "audit", False),
        ],
        measures=[
            MeasureTruth("purchase_order_lines", "line_amount", "sum", "additive", "currency", "Line amount"),
            MeasureTruth("purchase_order_lines", "ordered_qty", "sum", "additive", "integer", "Ordered quantity"),
            MeasureTruth("purchase_order_lines", "unit_price", "avg", "non_additive", "currency", "Unit price"),
            MeasureTruth("purchase_order_lines", None, "count", "additive", "count", "Purchase order lines"),
            MeasureTruth("purchase_order_lines", "po_number", "count_distinct", "additive", "count", "Purchase orders"),
            MeasureTruth("receipts", "received_qty", "sum", "additive", "integer", "Received quantity"),
            MeasureTruth("receipts", "rejected_qty", "sum", "additive", "integer", "Rejected quantity"),
            MeasureTruth("receipts", None, "count", "additive", "count", "Receipts"),
        ],
        labels={"suppliers": "supplier_name", "buyers": "buyer_name", "items": "item_name"},
        codes={"items": "item_code"},
        statuses={"purchase_order_lines.status": {"values": {"O": "Open", "C": "Closed", "X": "Cancelled"},
                                                  "cancelled": ["X"]}},
        quality=[
            {"object": "purchase_order_lines.status", "kind": "status_column"},
            {"object": "purchase_order_lines.ordered_qty", "kind": "unit_mix"},
            {"object": "purchase_order_lines.line_amount", "kind": "outlier_period"},
            {"object": "receipts.loaded_at", "kind": "load_timestamp"},
            {"object": "items", "kind": "listed_vs_active"},
        ],
        not_measures=["purchase_order_lines.po_line_id", "purchase_order_lines.line_number",
                      "purchase_order_lines.supplier_id", "purchase_order_lines.buyer_id",
                      "purchase_order_lines.item_id", "receipts.receipt_id", "receipts.po_line_id",
                      "receipts.item_id"],
    )
    return Domain("purchasing", tables, truth, TODAY,
                  "A manufacturer's purchase orders and goods receipts.",
                  abbreviations={"promised": "PRM", "received": "RCV", "rejected": "RJC"})
