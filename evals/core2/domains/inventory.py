"""Inventory: stock movements, daily and month-end balances, across warehouses.

What this warehouse is built to test:

* two balance snapshots that must never be added up over time: a daily one keyed
  by a DATE (zero balances are not stored) and a month-end one keyed by a yyyymm
  period number; "on hand" is the last snapshot day in each period;
* a period table keyed yyyymm whose year rows are month 00 ("Year 2025"), the
  month-end snapshot's calendar;
* a day calendar with a fiscal year starting in April, named by the year it ends;
* movements dated three ways: when they happened (a timestamp, the default), when
  they were posted (a calendar key, -1 while unposted) and when the received stock
  expires (a yyyymmdd number, mostly beyond the calendar's last year);
* two row-write timestamps that are not business dates: the movement's
  ``created_at`` (loaded nightly, a 2024 back-fill in one batch) and the month-end
  ``balance_ts`` (written about two months after the month it closes);
* a movement type (receipt, issue, transfer, adjustment) that decides what a
  quantity means, adjustments that are negative, and a status where "V" is voided;
* quantities in three units of measure (each, box, metre);
* a supplier only on receipts (empty otherwise), and 0.2% of movement lines for an
  item missing from the item list;
* names with analysis words and symbols ("Monthly Overflow Store", "#10 Hex Bolt",
  "O'Brien Fittings", "Top Grade Metals");
* 60 items listed, 36 ever stocked.

Movements run from 1 April 2024 to 12 June 2026, daily balances from 1 October 2025
to 12 June 2026, month-end balances from April 2024 to May 2026; questions are
asked on 15 June 2026.
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
FIRST_DAY = dt.date(2024, 4, 1)
LAST_DAY = dt.date(2026, 6, 12)
DAILY_FROM = dt.date(2025, 10, 1)
LAST_MONTH_END = (2026, 5)

WAREHOUSES = [("W01", "Northfield DC", "North", "Distribution centre"),
              ("W02", "Bayview Depot", "East", "Depot"),
              ("W03", "Cedar Hill Yard", "West", "Yard"),
              ("W04", "Riverside Hub", "South", "Distribution centre"),
              ("W05", "Monthly Overflow Store", "North", "Overflow"),
              ("W06", "Lakeport Annex", "East", "Depot")]
GROUPS = ["Fasteners", "Plumbing", "Electrical", "Paint", "Tools", "Safety"]
_ITEMS = {
    "Fasteners": [("#10 Hex Bolt", "BOX"), ("Wood Screw 50mm", "BOX"), ("Wall Anchor Kit", "EA"),
                  ("Lock Washer M8", "BOX"), ("Carriage Bolt 4in", "BOX"), ("Rivet Pack", "BOX"),
                  ("Threaded Rod 1m", "EA"), ("Wing Nut M6", "BOX"), ("Top Load Washer Belt", "EA"),
                  ("Concrete Screw 75mm", "BOX")],
    "Plumbing": [("PVC Pipe 2in", "M"), ("Copper Elbow 1/2in", "EA"), ("Ball Valve 3/4in", "EA"),
                 ("PEX Tubing", "M"), ("Drain Trap Kit", "EA"), ("Pipe Sealant Tape", "EA"),
                 ("Compression Fitting", "EA"), ("Hose Clamp Set", "BOX"), ("Shower Cartridge", "EA"),
                 ("Sink Strainer", "EA")],
    "Electrical": [("LED Panel 40W", "EA"), ("Copper Cable 2.5mm", "M"), ("Junction Box", "EA"),
                   ("Circuit Breaker 20A", "EA"), ("Wall Socket Double", "EA"), ("Conduit 20mm", "M"),
                   ("Cable Ties", "BOX"), ("Dimmer Switch", "EA"), ("Average Duty Extension Lead", "EA"),
                   ("Fuse Assortment", "BOX")],
    "Paint": [("Exterior Paint 4L", "EA"), ("Primer 1L", "EA"), ("Wood Stain 2L", "EA"), ("Roller Kit", "EA"),
              ("Masking Tape", "EA"), ("Paint Thinner 1L", "EA"), ("Brush Set", "EA"), ("Drop Cloth", "EA"),
              ("Ceiling White 10L", "EA"), ("Spray Enamel", "EA")],
    "Tools": [("Claw Hammer", "EA"), ("Cordless Drill", "EA"), ("Tape Measure 8m", "EA"), ("Utility Knife", "EA"),
              ("Spirit Level", "EA"), ("Socket Set", "EA"), ("Hacksaw", "EA"), ("Pliers Set", "EA"),
              ("Stud Finder", "EA"), ("Monthly Maintenance Kit", "EA")],
    "Safety": [("Safety Gloves L", "BOX"), ("Hard Hat", "EA"), ("Ear Defenders", "EA"), ("Safety Glasses", "BOX"),
               ("Hi-Vis Vest", "EA"), ("Dust Mask Pack", "BOX"), ("First Aid Kit", "EA"), ("Knee Pads", "EA"),
               ("Fire Blanket", "EA"), ("Warning Tape", "M")],
}
SUPPLIERS = ["O'Brien Fittings", "Top Grade Metals", "North & South Hardware", "Kestrel Industrial",
             "Bluegate Supply", "Ironbark Tools", "Meridian Electrical", "Pinewood Paints", "Harlow Plumbing Co",
             "Granite Safety Gear", "Summit Fasteners", "Lindqvist Trading"]
TYPES = ["RCV", "ISS", "TRF", "ADJ"]


def _key(days: pd.DatetimeIndex) -> np.ndarray:
    return (days.year * 10000 + days.month * 100 + days.day).to_numpy().astype("int64")


def _period_table() -> pd.DataFrame:
    rows = []
    for year in range(2024, 2028):
        rows.append((year * 100, f"Year {year}", year, 0, None, None, dt.date(year, 1, 1), dt.date(year, 12, 31)))
        for month in range(1, 13):
            start = dt.date(year, month, 1)
            end = (dt.date(year + (month == 12), month % 12 + 1, 1) - dt.timedelta(days=1))
            fiscal_year = year + (1 if month >= 4 else 0)
            fiscal_quarter = ((month - 4) % 12) // 3 + 1
            rows.append((year * 100 + month, start.strftime("%b %Y"), year, month, fiscal_year, fiscal_quarter,
                         start, end))
    frame = pd.DataFrame(rows, columns=["period_key", "period_name", "calendar_year", "month_number", "fiscal_year",
                                        "fiscal_quarter", "period_start", "period_end"])
    for column in ("fiscal_year", "fiscal_quarter"):
        frame[column] = frame[column].astype("Int64")
    return frame


def build(seed: int = 7) -> Domain:
    rng = np.random.default_rng(seed)

    calendar = calendar_frame(dt.date(2024, 1, 1), dt.date(2027, 12, 31), fiscal_start_month=4)
    periods = _period_table()

    warehouses = pd.DataFrame({
        "warehouse_id": np.arange(1, len(WAREHOUSES) + 1),
        "warehouse_code": [w[0] for w in WAREHOUSES],
        "warehouse_name": [w[1] for w in WAREHOUSES],
        "region": [w[2] for w in WAREHOUSES],
        "warehouse_type": [w[3] for w in WAREHOUSES],
    })
    item_groups = pd.DataFrame({"item_group_id": np.arange(1, len(GROUPS) + 1), "item_group_name": GROUPS})

    item_rows = []
    for gi, group in enumerate(GROUPS, start=1):
        for name, uom in _ITEMS[group]:
            iid = len(item_rows) + 1
            item_rows.append((iid, f"IT-{5000 + iid}", name, gi, uom, round(float(rng.uniform(2.0, 180.0)), 2)))
    items = pd.DataFrame(item_rows, columns=["item_id", "item_code", "item_name", "item_group_id",
                                             "unit_of_measure", "standard_cost"])
    # The names questions single out are always stocked; the rest of the 36 are drawn.
    named = items.index[items["item_name"].isin(["#10 Hex Bolt", "Monthly Maintenance Kit",
                                                 "Top Load Washer Belt"])].to_numpy()
    others = rng.permutation(np.setdiff1d(items.index.to_numpy(), named))[: 36 - len(named)]
    active = items.loc[np.sort(np.concatenate([named, others])), "item_id"].to_numpy()

    suppliers = pd.DataFrame({"supplier_id": np.arange(1, len(SUPPLIERS) + 1), "supplier_name": SUPPLIERS,
                              "country": rng.choice(["Canada", "United States", "Mexico"], len(SUPPLIERS))})

    # ── balances: one random walk per item and warehouse, every day ──────────
    days = pd.date_range(FIRST_DAY, LAST_DAY, freq="D")
    combos = [(i, w) for i in active for w in warehouses["warehouse_id"]]
    n_days, n_combos = len(days), len(combos)
    start = rng.integers(0, 400, n_combos).astype(float)
    steps = rng.normal(0, 12, (n_days, n_combos)).round()
    walk = np.maximum(0, start + np.cumsum(steps, axis=0))
    empty = rng.random((n_days, n_combos)) < 0.12          # stocked out on some days
    walk[empty] = 0
    cost = items.set_index("item_id").loc[[c[0] for c in combos], "standard_cost"].to_numpy()

    daily_mask = days >= pd.Timestamp(DAILY_FROM)
    d_days = days[daily_mask]
    d_walk = walk[daily_mask]
    rows_i, rows_c = np.nonzero(d_walk > 0)               # zero balances are not stored
    on_hand = d_walk[rows_i, rows_c]
    daily = pd.DataFrame({
        "balance_date": d_days[rows_i].date,
        "item_id": [combos[c][0] for c in rows_c],
        "warehouse_id": [combos[c][1] for c in rows_c],
        "on_hand_qty": on_hand,
        "allocated_qty": np.floor(on_hand * rng.uniform(0, 0.3, len(on_hand))),
        "on_hand_value": np.round(on_hand * cost[rows_c], 2),
    }).sort_values(["balance_date", "warehouse_id", "item_id"]).reset_index(drop=True)

    month_ends = []
    for year, month in pd.period_range("2024-04", f"{LAST_MONTH_END[0]}-{LAST_MONTH_END[1]:02d}", freq="M").map(
            lambda p: (p.year, p.month)):
        end = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
        month_ends.append((year * 100 + month, int(np.flatnonzero(days == end)[0]), end))
    m_rows = []
    for period_key, index, end in month_ends:
        # Written about two months after the month closes, at night.
        written = end + pd.Timedelta(days=int(rng.integers(45, 75))) + pd.Timedelta(hours=1)
        for c, (item_id, warehouse_id) in enumerate(combos):
            qty = walk[index, c]
            m_rows.append((period_key, item_id, warehouse_id, qty, round(float(qty * cost[c]), 2), written))
    monthly = pd.DataFrame(m_rows, columns=["period_key", "item_id", "warehouse_id", "closing_qty", "closing_value",
                                            "balance_ts"])

    # ── movements: documents of one to six lines ─────────────────────────────
    months = pd.period_range(FIRST_DAY, LAST_DAY, freq="M")
    doc_at: list[pd.Timestamp] = []
    for i, month in enumerate(months):
        first = month.start_time
        last = min(month.end_time.normalize(), pd.Timestamp(LAST_DAY))
        n_days_month = (last - first).days + 1
        n = int(150 * (1 + 0.01 * i) * n_days_month / month.days_in_month * rng.uniform(0.9, 1.1))
        offsets = rng.integers(0, n_days_month * 24 * 60, n)
        doc_at.extend(first + pd.to_timedelta(np.sort(offsets), unit="min"))
    n_docs = len(doc_at)
    doc_kind = rng.choice(TYPES, n_docs, p=[0.35, 0.45, 0.12, 0.08])
    doc_warehouse = rng.integers(1, len(WAREHOUSES) + 1, n_docs)
    doc_supplier = np.where(doc_kind == "RCV", rng.integers(1, len(SUPPLIERS) + 1, n_docs), 0)
    doc_lines = rng.choice([1, 2, 3, 4, 5, 6], n_docs, p=[0.3, 0.25, 0.18, 0.12, 0.09, 0.06])
    doc = np.repeat(np.arange(n_docs), doc_lines)
    n_lines = len(doc)
    line_number = np.concatenate([np.arange(1, k + 1) for k in doc_lines])
    moved = pd.DatetimeIndex(np.array(doc_at, dtype="datetime64[ns]")[doc]) + pd.to_timedelta(line_number - 1,
                                                                                               unit="s")
    kind = doc_kind[doc]
    warehouse_id = doc_warehouse[doc]
    supplier = doc_supplier[doc]
    prefix = {"RCV": "RC", "ISS": "IS", "TRF": "TR", "ADJ": "AD"}
    movement_number = np.array([f"{prefix[k]}-{300000 + d}" for k, d in zip(kind, doc)])
    item_id = rng.choice(active, n_lines)
    orphan = rng.random(n_lines) < 0.002
    uom = items.set_index("item_id").loc[item_id, "unit_of_measure"].to_numpy()
    quantity = np.where(uom == "M", np.round(rng.uniform(1, 120, n_lines), 1),
                        rng.integers(1, 60, n_lines)).astype(float)
    quantity = np.where(kind == "ADJ", -np.ceil(quantity / 4) * np.where(rng.random(n_lines) < 0.7, 1, -1), quantity)
    unit_cost = np.round(items.set_index("item_id").loc[item_id, "standard_cost"].to_numpy()
                         * rng.uniform(0.92, 1.08, n_lines), 2)

    posted = moved.normalize() + pd.to_timedelta(rng.integers(0, 4, n_docs)[doc], unit="D")
    unposted = posted > pd.Timestamp(LAST_DAY)
    posted_key = np.where(unposted, -1, _key(pd.DatetimeIndex(posted)))
    group_of = items.set_index("item_id").loc[item_id, "item_group_id"].to_numpy()
    perishable = (kind == "RCV") & np.isin(group_of, [4, 6])
    # Mostly past the calendar's last year: a date number with no calendar to join.
    expiry = moved.normalize() + pd.to_timedelta(rng.integers(1000, 2200, n_lines), unit="D")
    expiry_key = np.where(perishable, _key(pd.DatetimeIndex(expiry)), -1)
    status = np.where(rng.random(n_docs) < 0.02, "V", "P")[doc]

    # Loaded nightly; the first quarter was back-filled in one batch in July 2024.
    created = moved.normalize() + pd.Timedelta(days=1, hours=3)
    created = created.where(~(moved < pd.Timestamp("2024-07-01")), pd.Timestamp("2024-07-15 03:00"))

    movements = pd.DataFrame({
        "movement_id": np.arange(1, n_lines + 1),
        "movement_number": movement_number,
        "line_number": line_number,
        "movement_type": kind,
        "item_id": np.where(orphan, 999, item_id),
        "warehouse_id": warehouse_id,
        "supplier_id": pd.array(supplier, dtype="Int64"),
        "movement_ts": moved.to_numpy(),
        "posted_date_key": posted_key,
        "expiry_date": pd.array(expiry_key, dtype="Int64"),
        "quantity": quantity,
        "unit_cost": unit_cost,
        "extended_cost": np.round(quantity * unit_cost, 2),
        "status": status,
        "created_at": created.to_numpy(),
    })
    movements.loc[movements["supplier_id"] == 0, "supplier_id"] = pd.NA
    movements.loc[movements["expiry_date"] == -1, "expiry_date"] = pd.NA

    tables = [
        TableDef("calendar", "calendar", calendar, CALENDAR_TYPES, ["date_key"], "Calendar"),
        TableDef("fiscal_periods", "calendar", periods,
                 {"period_key": "INTEGER", "period_name": "VARCHAR", "calendar_year": "INTEGER",
                  "month_number": "INTEGER", "fiscal_year": "INTEGER", "fiscal_quarter": "INTEGER",
                  "period_start": "DATE", "period_end": "DATE"}, ["period_key"], "Period"),
        TableDef("warehouses", "dimension", warehouses,
                 {"warehouse_id": "INTEGER", "warehouse_code": "VARCHAR", "warehouse_name": "VARCHAR",
                  "region": "VARCHAR", "warehouse_type": "VARCHAR"}, ["warehouse_id"], "Warehouse"),
        TableDef("item_groups", "dimension", item_groups, {"item_group_id": "INTEGER", "item_group_name": "VARCHAR"},
                 ["item_group_id"], "Item group"),
        TableDef("items", "dimension", items,
                 {"item_id": "INTEGER", "item_code": "VARCHAR", "item_name": "VARCHAR", "item_group_id": "INTEGER",
                  "unit_of_measure": "VARCHAR", "standard_cost": "DECIMAL(10,2)"}, ["item_id"], "Item"),
        TableDef("suppliers", "dimension", suppliers,
                 {"supplier_id": "INTEGER", "supplier_name": "VARCHAR", "country": "VARCHAR"}, ["supplier_id"],
                 "Supplier"),
        TableDef("stock_movements", "fact", movements,
                 {"movement_id": "INTEGER", "movement_number": "VARCHAR", "line_number": "INTEGER",
                  "movement_type": "VARCHAR", "item_id": "INTEGER", "warehouse_id": "INTEGER",
                  "supplier_id": "INTEGER", "movement_ts": "TIMESTAMP", "posted_date_key": "INTEGER",
                  "expiry_date": "INTEGER", "quantity": "DECIMAL(12,1)", "unit_cost": "DECIMAL(10,2)",
                  "extended_cost": "DECIMAL(14,2)", "status": "VARCHAR", "created_at": "TIMESTAMP"},
                 ["movement_id"], "Stock movement"),
        TableDef("daily_balances", "snapshot", daily,
                 {"balance_date": "DATE", "item_id": "INTEGER", "warehouse_id": "INTEGER",
                  "on_hand_qty": "DECIMAL(12,1)", "allocated_qty": "DECIMAL(12,1)", "on_hand_value": "DECIMAL(14,2)"},
                 ["balance_date", "item_id", "warehouse_id"], "Daily balance"),
        TableDef("monthly_balances", "snapshot", monthly,
                 {"period_key": "INTEGER", "item_id": "INTEGER", "warehouse_id": "INTEGER",
                  "closing_qty": "DECIMAL(12,1)", "closing_value": "DECIMAL(14,2)", "balance_ts": "TIMESTAMP"},
                 ["period_key", "item_id", "warehouse_id"], "Month-end balance"),
    ]

    truth = Truth(
        kinds={t.name: t.kind for t in tables},
        primary_keys={t.name: t.primary_key for t in tables},
        joins=[
            JoinTruth("stock_movements", "item_id", "items", "item_id"),
            JoinTruth("stock_movements", "warehouse_id", "warehouses", "warehouse_id"),
            JoinTruth("stock_movements", "supplier_id", "suppliers", "supplier_id"),
            JoinTruth("stock_movements", "posted_date_key", "calendar", "date_key"),
            JoinTruth("daily_balances", "item_id", "items", "item_id"),
            JoinTruth("daily_balances", "warehouse_id", "warehouses", "warehouse_id"),
            JoinTruth("monthly_balances", "item_id", "items", "item_id"),
            JoinTruth("monthly_balances", "warehouse_id", "warehouses", "warehouse_id"),
            JoinTruth("monthly_balances", "period_key", "fiscal_periods", "period_key"),
            JoinTruth("items", "item_group_id", "item_groups", "item_group_id"),
        ],
        calendar={"table": "calendar", "key": "date_key", "date": "full_date", "attributes": CALENDAR_ATTRIBUTES,
                  "fiscal_year_start_month": 4, "fiscal_year_named_by": "end", "placeholders": [-1]},
        dates=[
            DateTruth("stock_movements", "movement_ts", "Movement time", "event", False, default=True),
            DateTruth("stock_movements", "posted_date_key", "Posting date", "event", True),
            DateTruth("stock_movements", "expiry_date", "Expiry date", "validity", False),
            DateTruth("stock_movements", "created_at", "Created at", "audit", False),
            DateTruth("daily_balances", "balance_date", "Balance date", "snapshot", False, default=True),
            DateTruth("monthly_balances", "period_key", "Period", "snapshot", True, default=True),
            DateTruth("monthly_balances", "balance_ts", "Balance written at", "audit", False),
        ],
        measures=[
            MeasureTruth("stock_movements", "quantity", "sum", "additive", "number", "Quantity",
                         unit_column="items.unit_of_measure"),
            MeasureTruth("stock_movements", "extended_cost", "sum", "additive", "currency", "Extended cost"),
            MeasureTruth("stock_movements", "unit_cost", "avg", "non_additive", "currency", "Unit cost"),
            MeasureTruth("stock_movements", None, "count", "additive", "count", "Movement lines"),
            MeasureTruth("stock_movements", "movement_number", "count_distinct", "additive", "count", "Movements"),
            MeasureTruth("daily_balances", "on_hand_qty", "sum", "semi_additive", "number", "On hand",
                         unit_column="items.unit_of_measure", time_aggregation="last"),
            MeasureTruth("daily_balances", "allocated_qty", "sum", "semi_additive", "number", "Allocated",
                         unit_column="items.unit_of_measure", time_aggregation="last"),
            MeasureTruth("daily_balances", "on_hand_value", "sum", "semi_additive", "currency", "On-hand value",
                         time_aggregation="last"),
            MeasureTruth("monthly_balances", "closing_qty", "sum", "semi_additive", "number", "Closing quantity",
                         unit_column="items.unit_of_measure", time_aggregation="last"),
            MeasureTruth("monthly_balances", "closing_value", "sum", "semi_additive", "currency", "Closing value",
                         time_aggregation="last"),
        ],
        labels={"warehouses": "warehouse_name", "items": "item_name", "item_groups": "item_group_name",
                "suppliers": "supplier_name"},
        codes={"warehouses": "warehouse_code", "items": "item_code"},
        statuses={"stock_movements.status": {"values": {"P": "Posted", "V": "Voided"}, "cancelled": ["V"]},
                  "stock_movements.movement_type": {"values": {"RCV": "Receipt", "ISS": "Issue",
                                                               "TRF": "Transfer", "ADJ": "Adjustment"}}},
        quality=[
            {"object": "stock_movements.quantity", "kind": "negative_values"},
            {"object": "stock_movements.status", "kind": "status_column"},
            {"object": "stock_movements.posted_date_key", "kind": "placeholder_dates"},
            {"object": "stock_movements.created_at", "kind": "load_timestamp"},
            {"object": "items", "kind": "listed_vs_active"},
            {"object": "stock_movements.quantity", "kind": "unit_mix"},
        ],
        not_measures=["stock_movements.line_number", "stock_movements.movement_id", "stock_movements.item_id",
                      "stock_movements.warehouse_id", "stock_movements.supplier_id",
                      "stock_movements.posted_date_key", "stock_movements.expiry_date", "daily_balances.item_id",
                      "daily_balances.warehouse_id", "monthly_balances.period_key", "monthly_balances.item_id",
                      "monthly_balances.warehouse_id", "items.item_group_id", "calendar.year",
                      "calendar.fiscal_year", "fiscal_periods.calendar_year", "fiscal_periods.month_number",
                      "fiscal_periods.fiscal_year", "fiscal_periods.fiscal_quarter"],
    )
    return Domain("inventory", tables, truth, TODAY,
                  "A building-supplies distributor's stock movements and balances across six warehouses.")
