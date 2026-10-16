"""The sample workbook: a small retail chain's data as an Excel file, for trying Solviz with no database.

Six sheets, as a team would keep them in a shared drive: Order lines (a title row above the header, as an export
writes it), Products, Stores, Customers, Returns and Monthly targets (a "Grand total" row under it). The figures are
invented, and the same ones every time it is built: two stores grow, one falls in 2026, one opened in March 2025,
the year peaks in November and December, and a few orders are cancelled.

    python tools/sample_workbook.py [--through 2026-09-30] [--out core/samples/solviz_sample_retail.xlsx]

The admin's Databases page offers the built copy (core/samples/) as "Use the sample workbook", and as a download to
put in SharePoint or OneDrive and try the link.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "core" / "samples" / "solviz_sample_retail.xlsx"
START = dt.date(2024, 1, 1)
THROUGH = dt.date(2026, 9, 30)

STORES = [  # id, name, region, city, opened, how it moves each year after 2024
    ("S01", "Harbourfront", "East", "Halifax", dt.date(2019, 4, 1), 0.06),
    ("S02", "Old Town", "East", "Quebec City", dt.date(2017, 9, 1), -0.09),
    ("S03", "Riverside", "Central", "Ottawa", dt.date(2020, 2, 1), 0.04),
    ("S04", "Union Square", "Central", "Toronto", dt.date(2016, 6, 1), 0.08),
    ("S05", "Prairie Gate", "West", "Winnipeg", dt.date(2021, 5, 1), 0.02),
    ("S06", "Mountain View", "West", "Calgary", dt.date(2018, 11, 1), 0.05),
    ("S07", "Seawall", "West", "Vancouver", dt.date(2019, 8, 1), 0.07),
    ("S08", "Lakeside", "Central", "Kingston", dt.date(2025, 3, 1), 0.0),
]
CATEGORIES = {  # category: (products, low price, high price, cost share, yearly growth)
    "Bakery": (["Sourdough Loaf", "Croissant 4-pack", "Rye Bread", "Bagels 6-pack", "Muffins 4-pack"], 3, 9, .45, .03),
    "Beverages": (["Cold Brew Coffee", "Sparkling Water 12-pack", "Orange Juice", "Green Tea 20ct", "Oat Milk"],
                  3, 14, .55, .04),
    "Produce": (["Gala Apples 2lb", "Baby Spinach", "Avocados 4-pack", "Blueberries", "Sweet Potatoes"], 3, 8, .5, .12),
    "Pantry": (["Olive Oil 1L", "Basmati Rice 2kg", "Pasta 1kg", "Tomato Passata", "Maple Syrup"], 2, 18, .6, .01),
    "Household": (["Dish Soap", "Paper Towels 6-pack", "Laundry Pods", "Trash Bags 40ct", "All-purpose Cleaner"],
                  4, 22, .58, .0),
    "Personal Care": (["Shampoo", "Toothpaste 2-pack", "Hand Soap", "Body Lotion", "Sunscreen SPF 50"], 4, 19, .5, .05),
    "Snacks": (["Sea Salt Chips", "Dark Chocolate Bar", "Trail Mix", "Granola Bars 6-pack", "Popcorn"], 2, 8, .48, .02),
    "Frozen": (["Frozen Berries", "Veggie Pizza", "Ice Cream 1L", "Frozen Peas", "Dumplings"], 4, 12, .52, -.03),
}
SEGMENTS = ["Walk-in", "Loyalty", "Business", "Online"]
FIRST = ["Bluebird", "Cedar", "Maple", "Harbor", "North Star", "Granite", "Willow", "Copper", "Juniper", "Summit",
         "Lighthouse", "Silver Birch", "Red Oak", "Fern", "Pinecone", "Riverbend", "Kestrel", "Aspen", "Heron", "Saffron"]
SECOND = ["Cafe", "Catering", "Office Supply", "Bakehouse", "Daycare", "Fitness", "Dental Clinic", "Studio", "Market",
          "Florist", "Bistro", "Hostel", "Print Shop", "Barbers", "Pet Care"]
REASONS = ["Damaged", "Wrong item", "No longer needed", "Arrived late", "Expired"]
SEASON = {1: .82, 2: .86, 3: .95, 4: .98, 5: 1.0, 6: 1.02, 7: 1.0, 8: 1.03, 9: .98, 10: 1.04, 11: 1.18, 12: 1.32}


def build(through: dt.date = THROUGH, seed: int = 7) -> bytes:
    """The workbook as an .xlsx file's bytes: the same rows for the same `through` and `seed`."""
    from openpyxl import Workbook

    rng = random.Random(seed)
    products = []
    for category, (names, low, high, cost_share, growth) in CATEGORIES.items():
        for name in names:
            price = round(rng.uniform(low, high), 2)
            products.append((f"P{len(products) + 1:03d}", name, category, round(price * cost_share, 2), price, growth))
    customers = []
    for n in range(240):
        name = f"{FIRST[n % len(FIRST)]} {SECOND[(n * 7) % len(SECOND)]}"
        if any(c[1] == name for c in customers):
            name += f" {['East', 'West', 'North', 'South'][n % 4]}"
        since = START + dt.timedelta(days=rng.randrange(0, (through - START).days - 60))
        customers.append((f"C{n + 1:04d}", name, SEGMENTS[n % len(SEGMENTS)], STORES[n % len(STORES)][3], since))

    lines, returns = [], []
    order_id = 100000
    day = START
    while day <= through:
        years = (day - START).days / 365.25
        for store_id, _, _, _, opened, trend in STORES:
            if day < opened:
                continue
            pace = 1.25 * SEASON[day.month] * (1 + trend) ** max(0.0, years - 1) * (0.7 if day.weekday() == 6 else 1)
            for _ in range(int(pace) + (rng.random() < pace - int(pace))):
                order_id += 1
                eligible = [c for c in customers if c[4] <= day]
                customer = rng.choice(eligible)[0] if eligible and rng.random() < .8 else ""
                cancelled = rng.random() < .03
                ship = None if cancelled else day + dt.timedelta(days=rng.choice([0, 1, 1, 2, 2, 3, 5]))
                for line in range(1, rng.choice([1, 1, 2, 2, 3, 4]) + 1):
                    pid, _, _, cost, price, growth = rng.choice(products)
                    qty = rng.choice([1, 1, 1, 2, 2, 3, 4, 6]) * (1 + (rng.random() < growth * years * 2))
                    discount = round(qty * price * rng.choice([0, 0, 0, 0, .05, .1, .15]), 2)
                    net = round(qty * price - discount, 2)
                    lines.append((order_id, line, day, ship, store_id, customer or None, pid, qty, price, discount,
                                  net, round(qty * cost, 2), "Cancelled" if cancelled else "Completed"))
                    if not cancelled and rng.random() < .025:
                        back = day + dt.timedelta(days=rng.randrange(3, 21))
                        if back <= through:
                            kept = rng.randrange(1, qty + 1)
                            returns.append((f"R{len(returns) + 1:05d}", order_id, line, pid, back, kept,
                                            round(net * kept / qty, 2), rng.choice(REASONS)))
        day += dt.timedelta(days=1)

    book = Workbook(write_only=True)
    sheet = book.create_sheet("Order lines")
    sheet.append([f"Order lines · exported {through:%d %b %Y}"])
    sheet.append([])
    sheet.append(["Order ID", "Line", "Order Date", "Ship Date", "Store ID", "Customer ID", "Product ID", "Quantity",
                  "Unit Price", "Discount", "Net Amount", "Cost Amount", "Status"])
    for row in lines:
        sheet.append(list(row))
    sheet = book.create_sheet("Products")
    sheet.append(["Product ID", "Product Name", "Category", "Unit Cost", "List Price"])
    for pid, name, category, cost, price, _ in products:
        sheet.append([pid, name, category, cost, price])
    sheet = book.create_sheet("Stores")
    sheet.append(["Store ID", "Store Name", "Region", "City", "Opened"])
    for store_id, name, region, city, opened, _ in STORES:
        sheet.append([store_id, name, region, city, opened])
    sheet = book.create_sheet("Customers")
    sheet.append(["Customer ID", "Customer Name", "Segment", "City", "Customer Since"])
    for row in customers:
        sheet.append(list(row))
    sheet = book.create_sheet("Returns")
    sheet.append(["Return ID", "Order ID", "Line", "Product ID", "Return Date", "Quantity", "Refund Amount", "Reason"])
    for row in returns:
        sheet.append(list(row))
    sheet = book.create_sheet("Monthly targets")
    sheet.append(["Month", "Store ID", "Target Amount"])
    actual: dict[tuple[str, int, int], float] = {}
    for row in lines:
        if row[12] == "Completed":
            key = (row[4], row[2].year, row[2].month)
            actual[key] = actual.get(key, 0.0) + row[10]
    total = 0.0
    for year in (2025, 2026):
        for month in range(1, 13):
            for store_id, _, _, _, opened, _ in STORES:
                if dt.date(year, month, 1) < opened.replace(day=1):
                    continue
                base = actual.get((store_id, year - 1, month)) or actual.get((store_id, year, 3)) or 9000.0
                target = round(base * 1.05 / 100) * 100
                total += target
                sheet.append([dt.date(year, month, 1), store_id, target])
    sheet.append([])
    sheet.append(["Grand total", None, round(total, 2)])
    out = io.BytesIO()
    book.save(out)
    return out.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--through", type=dt.date.fromisoformat, default=THROUGH)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(build(args.through))
    print(f"{args.out} ({args.out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
