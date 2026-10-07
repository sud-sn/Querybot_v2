"""Retail: order lines with three dates, returns, and monthly store targets.

What this warehouse is built to test:

* one calendar playing four roles (order, ship, delivered, return dates), with a
  "-1 = Unknown" row for orders not shipped or delivered yet;
* a row-load timestamp that trails the business date by months for some years
  (it must never become the default date);
* two paths to a store: the line's own store and the customer's home store, and a
  region reached through either (the line's own path must win and be named);
* a status code where "C" means cancelled (6% of orders), stored as positive amounts;
* a margin percentage that is exactly 38 on every line, a unit price that must
  not be summed, quantities in three units of measure;
* a month (August 2025) holding a data-entry outlier;
* returns whose customer key misses 2.5% of the time (a join to propose, not
  verify) and that also point at the order line they reverse (a fact-to-fact join);
* monthly targets keyed by a yyyymm period number, defined into the future;
* customer and product names containing analysis words and symbols
  ("Northline Distribution 58", "Acme #27 Supply", "Monthly Goods Ltd");
* 150 products listed, about 110 ever sold.

Data runs from 1 January 2023 to 14 June 2026; questions are asked on 15 June 2026.
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
FIRST_DAY = dt.date(2023, 1, 1)
LAST_DAY = dt.date(2026, 6, 14)
OUTLIER_MONTH = (2025, 8)

REGIONS = ["North", "South", "East", "West", "Central"]
STORES = [("S01", "Downtown"), ("S02", "Harbourfront"), ("S03", "Airport"), ("S04", "Riverside"),
          ("S05", "Old Town"), ("S06", "Westfield"), ("S07", "Lakeshore"), ("S08", "Hillcrest"),
          ("S09", "Northgate"), ("S10", "Southpark"), ("S11", "Eastview"), ("S12", "Midtown")]
CATEGORIES = ["Beverages", "Snacks", "Household", "Personal Care", "Frozen", "Bakery", "Produce", "Pantry"]
_NOUNS = {
    "Beverages": ["Sparkling Water", "Cold Brew Coffee", "Green Tea", "Orange Juice", "Cola", "Lemonade"],
    "Snacks": ["Sea Salt Chips", "Trail Mix", "Pretzels", "Granola Bar", "Popcorn", "Rice Crackers"],
    "Household": ["Dish Soap", "Paper Towels", "Laundry Pods", "Trash Bags", "Sponges", "Glass Cleaner"],
    "Personal Care": ["Shampoo", "Toothpaste", "Hand Soap", "Body Lotion", "Deodorant", "Razor Blades"],
    "Frozen": ["Frozen Peas", "Ice Cream", "Fish Fillets", "Pizza", "Dumplings", "Berries Mix"],
    "Bakery": ["Sourdough Loaf", "Bagels", "Croissants", "Muffins", "Rye Bread", "Tortillas"],
    "Produce": ["Bananas", "Apples", "Carrots", "Potatoes", "Onions", "Tomatoes"],
    "Pantry": ["Olive Oil", "Basmati Rice", "Pasta", "Flour", "Canned Beans", "Peanut Butter"],
}
_SIZES = ["Small", "Large", "Family Pack", "Value Pack", "Classic"]
# Names that contain analysis words or symbols: they must still be found as names.
SPECIAL_CUSTOMERS = ["Northline Distribution 58", "Blue & Green Trading", "Acme #27 Supply", "Monthly Goods Ltd",
                     "Top Value Stores", "The Last Mile Co", "Trend Setters Inc", "Average Joe's Market"]
_FIRST = ["Northline", "Blue River", "Summit", "Cedar", "Harbor", "Maple", "Silverline", "Redwood", "Granite",
          "Lakeside", "Pioneer", "Sterling", "Brightway", "Oakmont", "Riverbend", "Crescent", "Evergreen",
          "Highland", "Ironwood", "Juniper", "Keystone", "Lighthouse", "Meadow", "Nightingale", "Orchard"]
_SECOND = ["Distribution", "Trading", "Supply", "Foods", "Retail", "Market", "Wholesale", "Goods", "Grocers",
           "Provisions"]
SEGMENTS = ["Retail", "Wholesale", "Online"]
CITIES = ["Springfield", "Riverton", "Lakewood", "Fairview", "Georgetown", "Franklin", "Clinton", "Salem"]
REASONS = ["DMG", "WRG", "LATE", "OTHER"]


def _key(days: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    d = pd.DatetimeIndex(days)
    return (d.year * 10000 + d.month * 100 + d.day).to_numpy().astype("int64")


def build(seed: int = 7) -> Domain:
    rng = np.random.default_rng(seed)

    calendar = calendar_frame(dt.date(2022, 1, 1), dt.date(2027, 12, 31), fiscal_start_month=4)

    regions = pd.DataFrame({"region_id": np.arange(1, len(REGIONS) + 1), "region_name": REGIONS})

    stores = pd.DataFrame({
        "store_id": np.arange(1, len(STORES) + 1),
        "store_code": [c for c, _ in STORES],
        "store_name": [f"{n} Store" for _, n in STORES],
        "region_id": [(i % len(REGIONS)) + 1 for i in range(len(STORES))],
        "opened_date": [dt.date(2015 + (i % 8), 1 + (i * 5) % 12, 1 + (i * 7) % 27) for i in range(len(STORES))],
    })

    categories = pd.DataFrame({"category_id": np.arange(1, len(CATEGORIES) + 1), "category_name": CATEGORIES})

    product_rows = []
    pid = 1
    for ci, category in enumerate(CATEGORIES, start=1):
        for noun in _NOUNS[category]:
            for size in _SIZES[: 3 if ci % 2 else 4]:
                uom = "KG" if category == "Produce" else ("BOX" if size in ("Family Pack", "Value Pack") else "EA")
                product_rows.append((pid, f"SKU-{10000 + pid}", f"{noun} {size}", ci, uom,
                                     round(float(rng.uniform(1.5, 40.0)), 2)))
                pid += 1
    products = pd.DataFrame(product_rows, columns=["product_id", "sku", "product_name", "category_id",
                                                   "unit_of_measure", "list_price"]).head(150)
    # Analysis words inside product names too.
    products.loc[3, "product_name"] = "Monthly Planner Notebook"
    products.loc[10, "product_name"] = "Top Shelf Olive Oil #3"

    names = list(SPECIAL_CUSTOMERS)
    for first in _FIRST:
        for second in _SECOND:
            if len(names) >= 240:
                break
            candidate = f"{first} {second}"
            if candidate not in names:
                names.append(candidate)
    names = names[:240]
    n_customers = len(names)
    customers = pd.DataFrame({
        "customer_id": np.arange(1, n_customers + 1),
        "customer_code": [f"C{100000 + i}" for i in range(1, n_customers + 1)],
        "customer_name": names,
        "segment": rng.choice(SEGMENTS, n_customers, p=[0.5, 0.3, 0.2]),
        "city": rng.choice(CITIES, n_customers),
        "home_store_id": rng.integers(1, len(STORES) + 1, n_customers),
        "created_at": [dt.datetime(2020, 1, 1) + dt.timedelta(days=int(d), hours=int(h))
                       for d, h in zip(rng.integers(0, 1600, n_customers), rng.integers(8, 18, n_customers))],
    })

    # ── orders ───────────────────────────────────────────────────────────────
    months = pd.period_range(FIRST_DAY, LAST_DAY, freq="M")
    order_days: list[pd.Timestamp] = []
    for i, month in enumerate(months):
        start = month.start_time
        end = min(month.end_time.normalize(), pd.Timestamp(LAST_DAY))
        n_days = (end - start).days + 1
        season = 1.25 if month.month in (11, 12) else (0.9 if month.month in (1, 2) else 1.0)
        n = int(260 * (1 + 0.012 * i) * season * n_days / month.days_in_month * rng.uniform(0.92, 1.08))
        offsets = rng.integers(0, n_days, n)
        order_days.extend(start + pd.to_timedelta(np.sort(offsets), unit="D"))
    n_orders = len(order_days)
    order_date = pd.DatetimeIndex(order_days)
    weights = 1.0 / np.arange(1, n_customers + 1) ** 0.6
    order_customer = rng.choice(np.arange(1, n_customers + 1), n_orders, p=weights / weights.sum())
    home = customers.set_index("customer_id")["home_store_id"].to_numpy()[order_customer - 1]
    order_store = np.where(rng.random(n_orders) < 0.7, home, rng.integers(1, len(STORES) + 1, n_orders))
    cancelled = rng.random(n_orders) < 0.06
    ship_date = order_date + pd.to_timedelta(rng.integers(1, 6, n_orders), unit="D")
    delivered_date = ship_date + pd.to_timedelta(rng.integers(1, 8, n_orders), unit="D")
    last = pd.Timestamp(LAST_DAY)
    shipped = (~cancelled) & (ship_date <= last)
    delivered = shipped & (delivered_date <= last)
    status = np.where(cancelled, "C", np.where(delivered, "D", np.where(shipped, "S", "O")))
    ship_key = np.where(shipped, _key(ship_date), -1)
    delivered_key = np.where(delivered, _key(delivered_date), -1)

    # ── order lines ──────────────────────────────────────────────────────────
    lines_per_order = rng.choice([1, 2, 3, 4, 5], n_orders, p=[0.3, 0.3, 0.2, 0.12, 0.08])
    order_index = np.repeat(np.arange(n_orders), lines_per_order)
    n_lines = len(order_index)
    line_number = np.concatenate([np.arange(1, k + 1) for k in lines_per_order])
    sold = rng.permutation(products["product_id"].to_numpy())[:110]
    product_id = rng.choice(sold, n_lines)
    orphan = rng.random(n_lines) < 0.003
    product_lookup = products.set_index("product_id")
    uom = product_lookup.loc[product_id, "unit_of_measure"].to_numpy()
    list_price = product_lookup.loc[product_id, "list_price"].to_numpy()
    quantity = np.where(uom == "KG", np.round(rng.uniform(0.5, 50.0, n_lines), 3),
                        np.where(uom == "BOX", rng.integers(1, 6, n_lines), rng.integers(1, 21, n_lines))).astype(float)
    line_date = order_date[order_index]
    outlier_rows = np.flatnonzero((line_date.year == OUTLIER_MONTH[0]) & (line_date.month == OUTLIER_MONTH[1]))[:3]
    quantity[outlier_rows] = quantity[outlier_rows] * 1000
    unit_price = np.round(list_price * rng.uniform(0.9, 1.1, n_lines), 2)
    gross = np.round(quantity * unit_price, 2)
    discount = np.round(gross * rng.choice([0.0, 0.05, 0.10], n_lines, p=[0.6, 0.3, 0.1]), 2)
    net = np.round(gross - discount, 2)
    cost = np.round(net * 0.62, 2)

    # Rows are loaded weekly, on Monday nights after the order; 2023 was back-filled
    # in one batch in August 2023, and March 2024 arrived three months late.
    lag = pd.to_timedelta(rng.integers(0, 3, n_lines), unit="D")
    next_monday = line_date + lag + pd.to_timedelta((7 - (line_date + lag).dayofweek) % 7 + 0, unit="D")
    loaded = next_monday + pd.Timedelta(hours=2)
    loaded = loaded.where(~(line_date.year == 2023), pd.Timestamp("2023-08-07 02:00"))
    loaded = loaded.where(~((line_date.year == 2024) & (line_date.month == 3)), pd.Timestamp("2024-06-03 02:00"))
    loaded = loaded.where(loaded <= pd.Timestamp(TODAY) + pd.Timedelta(hours=2), pd.Timestamp(TODAY) + pd.Timedelta(hours=2))

    order_numbers = np.array([f"SO-{200000 + i}" for i in range(n_orders)])
    order_lines = pd.DataFrame({
        "order_line_id": np.arange(1, n_lines + 1),
        "order_number": order_numbers[order_index],
        "line_number": line_number,
        "customer_id": order_customer[order_index],
        "product_id": np.where(orphan, 9999, product_id),
        "store_id": order_store[order_index],
        "order_date_key": _key(line_date),
        "ship_date_key": ship_key[order_index],
        "delivered_date_key": delivered_key[order_index],
        "quantity": quantity,
        "unit_price": unit_price,
        "gross_amount": gross,
        "discount_amount": discount,
        "net_amount": net,
        "cost_amount": cost,
        "margin_pct": np.full(n_lines, 38.0),
        "status_code": status[order_index],
        "currency_code": np.full(n_lines, "USD"),
        "loaded_at": loaded.to_numpy(),
    })

    # ── returns ──────────────────────────────────────────────────────────────
    delivered_lines = order_lines[order_lines["status_code"] == "D"]
    picked = delivered_lines.sample(frac=0.04, random_state=seed)
    delivered_on = pd.to_datetime(picked["delivered_date_key"].astype(str), format="%Y%m%d")
    return_on = delivered_on + pd.to_timedelta(rng.integers(3, 31, len(picked)), unit="D")
    keep = (return_on <= last).to_numpy()
    picked, return_on = picked[keep], return_on[keep]
    share = rng.uniform(0.2, 1.0, len(picked))
    return_qty = np.maximum(1, np.ceil(picked["quantity"].to_numpy() * share))
    return_qty = np.minimum(return_qty, picked["quantity"].to_numpy())
    refund = np.round(picked["net_amount"].to_numpy() * return_qty / picked["quantity"].to_numpy(), 2)
    customer_of_return = picked["customer_id"].to_numpy().copy()
    missing = rng.random(len(picked)) < 0.025
    customer_of_return[missing] = rng.integers(900, 950, int(missing.sum()))
    returns = pd.DataFrame({
        "return_id": np.arange(1, len(picked) + 1),
        "order_line_id": picked["order_line_id"].to_numpy(),
        "return_date_key": _key(return_on),
        "customer_id": customer_of_return,
        "product_id": picked["product_id"].to_numpy(),
        "return_quantity": return_qty,
        "refund_amount": refund,
        "reason_code": rng.choice(REASONS, len(picked), p=[0.4, 0.3, 0.2, 0.1]),
    })

    # ── monthly targets per store, into the future ──────────────────────────
    by_store = order_lines.groupby("store_id")["net_amount"].sum() / len(months)
    target_months = pd.period_range("2023-01", "2026-12", freq="M")
    targets = pd.DataFrame([
        (store, int(p.year * 100 + p.month), float(round(by_store.get(store, 0.0) * rng.uniform(0.95, 1.15), -2)))
        for store in stores["store_id"] for p in target_months
    ], columns=["store_id", "period_key", "target_amount"])

    tables = [
        TableDef("calendar", "calendar", calendar, CALENDAR_TYPES, ["date_key"], "Calendar"),
        TableDef("regions", "dimension", regions, {"region_id": "INTEGER", "region_name": "VARCHAR"},
                 ["region_id"], "Region"),
        TableDef("stores", "dimension", stores,
                 {"store_id": "INTEGER", "store_code": "VARCHAR", "store_name": "VARCHAR", "region_id": "INTEGER",
                  "opened_date": "DATE"}, ["store_id"], "Store"),
        TableDef("categories", "dimension", categories, {"category_id": "INTEGER", "category_name": "VARCHAR"},
                 ["category_id"], "Product category"),
        TableDef("products", "dimension", products,
                 {"product_id": "INTEGER", "sku": "VARCHAR", "product_name": "VARCHAR", "category_id": "INTEGER",
                  "unit_of_measure": "VARCHAR", "list_price": "DECIMAL(10,2)"}, ["product_id"], "Product"),
        TableDef("customers", "dimension", customers,
                 {"customer_id": "INTEGER", "customer_code": "VARCHAR", "customer_name": "VARCHAR",
                  "segment": "VARCHAR", "city": "VARCHAR", "home_store_id": "INTEGER", "created_at": "TIMESTAMP"},
                 ["customer_id"], "Customer"),
        TableDef("order_lines", "fact", order_lines,
                 {"order_line_id": "INTEGER", "order_number": "VARCHAR", "line_number": "INTEGER",
                  "customer_id": "INTEGER", "product_id": "INTEGER", "store_id": "INTEGER",
                  "order_date_key": "INTEGER", "ship_date_key": "INTEGER", "delivered_date_key": "INTEGER",
                  "quantity": "DECIMAL(14,3)", "unit_price": "DECIMAL(12,2)", "gross_amount": "DECIMAL(14,2)",
                  "discount_amount": "DECIMAL(14,2)", "net_amount": "DECIMAL(14,2)", "cost_amount": "DECIMAL(14,2)",
                  "margin_pct": "DECIMAL(5,2)", "status_code": "VARCHAR", "currency_code": "VARCHAR",
                  "loaded_at": "TIMESTAMP"}, ["order_line_id"], "Order line"),
        TableDef("returns", "fact", returns,
                 {"return_id": "INTEGER", "order_line_id": "INTEGER", "return_date_key": "INTEGER",
                  "customer_id": "INTEGER", "product_id": "INTEGER", "return_quantity": "DECIMAL(14,3)",
                  "refund_amount": "DECIMAL(14,2)", "reason_code": "VARCHAR"}, ["return_id"], "Return"),
        TableDef("sales_targets", "fact", targets,
                 {"store_id": "INTEGER", "period_key": "INTEGER", "target_amount": "DECIMAL(14,2)"},
                 ["store_id", "period_key"], "Sales target"),
    ]

    truth = Truth(
        kinds={t.name: t.kind for t in tables},
        primary_keys={t.name: t.primary_key for t in tables},
        joins=[
            JoinTruth("order_lines", "customer_id", "customers", "customer_id"),
            JoinTruth("order_lines", "product_id", "products", "product_id"),
            JoinTruth("order_lines", "store_id", "stores", "store_id"),
            JoinTruth("order_lines", "order_date_key", "calendar", "date_key", "Order date"),
            JoinTruth("order_lines", "ship_date_key", "calendar", "date_key", "Ship date"),
            JoinTruth("order_lines", "delivered_date_key", "calendar", "date_key", "Delivered date"),
            JoinTruth("customers", "home_store_id", "stores", "store_id", "Home store"),
            JoinTruth("stores", "region_id", "regions", "region_id"),
            JoinTruth("products", "category_id", "categories", "category_id"),
            JoinTruth("returns", "order_line_id", "order_lines", "order_line_id"),
            JoinTruth("returns", "product_id", "products", "product_id"),
            JoinTruth("returns", "customer_id", "customers", "customer_id", trust="proposed"),
            JoinTruth("returns", "return_date_key", "calendar", "date_key", "Return date"),
            JoinTruth("sales_targets", "store_id", "stores", "store_id"),
        ],
        calendar={"table": "calendar", "key": "date_key", "date": "full_date", "attributes": CALENDAR_ATTRIBUTES,
                  "fiscal_year_start_month": 4, "fiscal_year_named_by": "end", "placeholders": [-1]},
        dates=[
            DateTruth("order_lines", "order_date_key", "Order date", "event", True, default=True),
            DateTruth("order_lines", "ship_date_key", "Ship date", "event", True),
            DateTruth("order_lines", "delivered_date_key", "Delivered date", "event", True),
            DateTruth("order_lines", "loaded_at", "Loaded at", "audit", False),
            DateTruth("returns", "return_date_key", "Return date", "event", True, default=True),
            DateTruth("sales_targets", "period_key", "Period", "event", False, default=True),
            DateTruth("customers", "created_at", "Customer since", "event", False, default=True),
            DateTruth("stores", "opened_date", "Opened", "event", False, default=True),
        ],
        measures=[
            MeasureTruth("order_lines", "net_amount", "sum", "additive", "currency", "Net sales"),
            MeasureTruth("order_lines", "gross_amount", "sum", "additive", "currency", "Gross sales"),
            MeasureTruth("order_lines", "discount_amount", "sum", "additive", "currency", "Discount"),
            MeasureTruth("order_lines", "cost_amount", "sum", "additive", "currency", "Cost"),
            MeasureTruth("order_lines", "quantity", "sum", "additive", "number", "Quantity",
                         unit_column="products.unit_of_measure"),
            MeasureTruth("order_lines", "unit_price", "avg", "non_additive", "currency", "Unit price"),
            MeasureTruth("order_lines", "margin_pct", "avg", "non_additive", "percent", "Margin %"),
            MeasureTruth("order_lines", None, "count", "additive", "count", "Order lines"),
            MeasureTruth("order_lines", "order_number", "count_distinct", "additive", "count", "Orders"),
            MeasureTruth("returns", "refund_amount", "sum", "additive", "currency", "Refunds"),
            MeasureTruth("returns", "return_quantity", "sum", "additive", "number", "Returned quantity"),
            MeasureTruth("sales_targets", "target_amount", "sum", "additive", "currency", "Sales target"),
        ],
        labels={"customers": "customer_name", "products": "product_name", "stores": "store_name",
                "regions": "region_name", "categories": "category_name"},
        codes={"customers": "customer_code", "products": "sku", "stores": "store_code"},
        statuses={"order_lines.status_code": {"values": {"O": "Open", "S": "Shipped", "D": "Delivered",
                                                         "C": "Cancelled"}, "cancelled": ["C"]},
                  "returns.reason_code": {"values": {"DMG": "Damaged", "WRG": "Wrong item", "LATE": "Late",
                                                     "OTHER": "Other"}}},
        quality=[
            {"object": "order_lines.margin_pct", "kind": "constant"},
            {"object": "order_lines.quantity", "kind": "outlier_period", "period": "2025-08"},
            {"object": "returns.customer_id", "kind": "low_match_rate"},
            {"object": "order_lines.status_code", "kind": "status_column"},
            {"object": "order_lines.delivered_date_key", "kind": "placeholder_dates"},
            {"object": "products", "kind": "listed_vs_active"},
            {"object": "order_lines.loaded_at", "kind": "load_timestamp"},
            {"object": "order_lines.quantity", "kind": "unit_mix"},
        ],
        not_measures=["order_lines.line_number", "order_lines.order_line_id", "order_lines.customer_id",
                      "order_lines.product_id", "order_lines.store_id", "order_lines.order_date_key",
                      "order_lines.ship_date_key", "order_lines.delivered_date_key", "returns.order_line_id",
                      "sales_targets.period_key", "calendar.year", "calendar.quarter", "calendar.month_number",
                      "calendar.day_of_week", "calendar.fiscal_year", "calendar.is_weekend", "stores.region_id",
                      "customers.home_store_id", "products.category_id"],
    )
    return Domain("retail", tables, truth, TODAY,
                  "A grocery retailer's orders, returns and store targets.")
