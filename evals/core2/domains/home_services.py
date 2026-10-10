"""Home services: households book cleaning and repair work, done by technicians, sometimes through partners.

A benchmark domain (not in the gated set). It holds the shapes a warehouse about people and orders
has, whatever its business, and that Learn must read from the data:

* dates kept as numbers (yyyymmdd) that point at a date table holding the date itself, with a
  fiscal year that starts in July;
* a booking's status kept by number in a short lookup ("4" is "Closed - Cancelled"): cancelled,
  voided and duplicate bookings are left out of totals by default, refunded ones only asked about;
* customers named in two parts (first and last name), with an email, a phone number, a street
  address and a birth date: people's names never reach the AI, the rest is never shown;
* a technician's mobile number whose column says nothing of a phone, and a partner company's
  contact person (a person's name in a table about companies);
* booking lines numbered within each booking, and the booking's travel fee written again on each
  of its lines (a figure of the booking: averaged, never added up over its lines);
* maintenance plans renewed again and again, their renewal number running on from wherever the data
  starts, the visits each renewal includes (each renewal's own, added up) and the renewals a plan
  allows (an allowance, not an amount);
* visits measured in minutes and kilometres (their own units), and parts used counted in several
  units (never added across units).

Bookings start in January 2024; questions are asked on 15 June 2026.
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
    date_key,
)

TODAY = dt.date(2026, 6, 15)
FIRST_DAY = dt.date(2024, 1, 2)
LAST_DAY = dt.date(2026, 6, 12)

FIRST = ["Alma", "Bruno", "Celia", "Dario", "Edith", "Felix", "Greta", "Hugo", "Ines", "Jonas", "Kira", "Lionel",
         "Mira", "Nils", "Olga", "Pavel", "Rosa", "Silas", "Tilda", "Viggo"]
LAST = ["Abernathy", "Blackwood", "Crane", "Delacroix", "Everhart", "Fenwick", "Grimshaw", "Holloway", "Ivers",
        "Jarrow", "Kingsley", "Lindqvist", "Marlowe", "Nightingale", "Osgood", "Prescott", "Quimby", "Rutherford"]
STREETS = ["Maple", "Cedar", "Willow", "Juniper", "Hawthorn", "Linden", "Sycamore", "Birch", "Alder", "Laurel"]
SUFFIXES = ["Street", "Avenue", "Road", "Lane", "Drive", "Court", "Way", "Place", "Terrace", "Loop", "Row", "Crescent"]
CITIES = [("Riverton", "OR"), ("Lakemont", "WA"), ("Pinecrest", "OR"), ("Harborview", "WA"), ("Stonebridge", "ID")]
SERVICES = [("Deep cleaning", 55.0), ("Window cleaning", 40.0), ("Plumbing repair", 95.0), ("Electrical repair", 110.0),
            ("Gutter cleaning", 45.0), ("Appliance repair", 90.0), ("Painting", 60.0), ("Carpet cleaning", 50.0)]
STATUSES = [(0, "Open - Requested", "Open"), (1, "Open - Scheduled", "Open"), (2, "Done - Completed", "Done"),
            (3, "Done - Invoiced", "Done"), (4, "Closed - Cancelled", "Closed"), (5, "Closed - Refunded", "Closed"),
            (6, "Closed - Duplicate Entry", "Closed"), (7, "Closed - Voided", "Closed")]
STATUS_SHARE = [0.02, 0.04, 0.30, 0.52, 0.06, 0.03, 0.01, 0.02]
PARTNERS = ["Northgate Property Management", "Bluefin Realty", "Copperleaf Housing Co-op", "Summit Home Insurance",
            "Evergreen Estates HOA", "Tidewater Rentals"]
PARTS = [("Gasket", "EA"), ("Copper pipe", "M"), ("Sealant", "L"), ("Wire", "M"), ("Paint", "L"), ("Fuse", "EA")]


def _phone(rng: np.random.Generator) -> str:
    return f"({rng.integers(200, 990)}) 555-{rng.integers(0, 10000):04d}"


def build(seed: int = 7) -> Domain:
    rng = np.random.default_rng(seed)
    days = (LAST_DAY - FIRST_DAY).days

    calendar = calendar_frame(dt.date(2023, 7, 1), dt.date(2027, 6, 30), fiscal_start_month=7, placeholders=[])
    statuses = pd.DataFrame({"status_id": [s for s, _, _ in STATUSES], "status_name": [n for _, n, _ in STATUSES],
                             "status_group": [g for _, _, g in STATUSES]})

    # ── people ───────────────────────────────────────────────────────────────
    n_customers = 900
    first = rng.choice(FIRST, n_customers)
    last = rng.choice(LAST, n_customers)
    city = rng.integers(0, len(CITIES), n_customers)
    customers = pd.DataFrame({
        "customer_id": np.arange(1, n_customers + 1),
        "customer_code": [f"HC-{40000 + i}" for i in range(n_customers)],
        "first_name": first, "last_name": last,
        "email": [f"{f.lower()}.{la.lower()}{i}@example.org" for i, (f, la) in enumerate(zip(first, last))],
        "phone": [_phone(rng) for _ in range(n_customers)],
        "street_address": [f"{rng.integers(10, 9999)} {rng.choice(STREETS)} {rng.choice(SUFFIXES)}"
                           + (f" Apt {rng.integers(1, 40)}" if rng.random() < 0.3 else "") for _ in range(n_customers)],
        "birth_date": [dt.date(1945, 1, 1) + dt.timedelta(days=int(d)) for d in rng.integers(0, 21000, n_customers)],
        "city": [CITIES[c][0] for c in city], "state": [CITIES[c][1] for c in city],
        "signup_date": [FIRST_DAY - dt.timedelta(days=400) + dt.timedelta(days=int(d))
                        for d in rng.integers(0, days + 300, n_customers)]})
    n_tech = 24
    technicians = pd.DataFrame({
        "technician_id": np.arange(1, n_tech + 1),
        "employee_code": [f"T-{100 + i}" for i in range(n_tech)],
        "technician_name": [f"{FIRST[(i * 7) % 20]} {LAST[(i * 5 + i // 18) % 18]}" for i in range(n_tech)],
        "team": rng.choice(["Cleaning", "Repair", "Exterior"], n_tech),
        "mobile_number": [_phone(rng) for _ in range(n_tech)],
        "hire_date": [dt.date(2015, 1, 5) + dt.timedelta(days=int(d)) for d in rng.integers(0, 3500, n_tech)]})
    partners = pd.DataFrame({
        "partner_id": np.arange(1, len(PARTNERS) + 1), "partner_name": PARTNERS,
        "contact_name": [f"{FIRST[(i * 3) % 20]} {LAST[(i * 11) % 18]}" for i in range(len(PARTNERS))],
        "office_phone": [_phone(rng) for _ in PARTNERS],
        "partner_type": rng.choice(["Property manager", "Insurer", "Association"], len(PARTNERS))})

    # ── bookings and their lines ────────────────────────────────────────────
    n_bookings = 5000
    booked = sorted(FIRST_DAY + dt.timedelta(days=int(d)) for d in rng.integers(0, days - 5, n_bookings))
    service_day = [b + dt.timedelta(days=int(d)) for b, d in zip(booked, rng.integers(1, 21, n_bookings))]
    status = rng.choice([s for s, _, _ in STATUSES], n_bookings, p=STATUS_SHARE)
    travel = rng.choice([0.0, 15.0, 25.0, 35.0], n_bookings, p=[0.3, 0.3, 0.25, 0.15])
    bookings = pd.DataFrame({
        "booking_id": np.arange(1, n_bookings + 1),
        "booking_number": [f"BK-{200000 + i}" for i in range(n_bookings)],
        "customer_id": rng.integers(1, n_customers + 1, n_bookings),
        "partner_id": pd.array([int(p) if rng.random() < 0.25 else None
                                for p in rng.integers(1, len(PARTNERS) + 1, n_bookings)], dtype="Int64"),
        "booked_date_key": [date_key(d) for d in booked],
        "service_date_key": [date_key(d) for d in service_day],
        "status_id": status, "travel_fee": travel})
    lines = []
    for b in bookings.itertuples():
        for line_number in range(1, int(rng.integers(1, 5)) + 1):
            service, rate = SERVICES[int(rng.integers(0, len(SERVICES)))]
            hours = round(float(rng.choice([1.0, 1.5, 2.0, 3.0, 4.0])), 1)
            lines.append({"booking_number": b.booking_number, "line_number": line_number, "service_type": service,
                          "technician_id": int(rng.integers(1, n_tech + 1)), "booked_date_key": b.booked_date_key,
                          "hours": hours, "line_amount": round(hours * rate, 2), "travel_fee": float(b.travel_fee)})
    booking_lines = pd.DataFrame(lines)

    # ── maintenance plans, renewed; the data starts part way through many plans ─
    plans = []
    for p in range(900):
        renewals_allowed = int(rng.choice([4, 6, 12]))
        start = int(rng.integers(1, 6))               # the renewals before the data are not in it
        renewal_day = FIRST_DAY + dt.timedelta(days=int(rng.integers(0, 120)))
        visits = int(rng.choice([1, 2, 4]))
        fee = float(rng.choice([49.0, 89.0, 149.0]))
        customer = int(rng.integers(1, n_customers + 1))
        for renewal in range(start, renewals_allowed + 1):
            if renewal_day > LAST_DAY:
                break
            plans.append({"plan_number": f"MP-{3000 + p}", "renewal_number": renewal, "customer_id": customer,
                          "renewal_date_key": date_key(renewal_day), "visits_included": visits,
                          "renewals_allowed": renewals_allowed, "plan_fee": fee})
            renewal_day += dt.timedelta(days=int(rng.choice([90, 91, 92])))
    plan_renewals = pd.DataFrame(plans)

    # ── visits: minutes, kilometres, parts in several units ─────────────────
    visits = []
    for b in bookings.sample(frac=0.8, random_state=seed).itertuples():
        part, unit = PARTS[int(rng.integers(0, len(PARTS)))]
        visits.append({"visit_id": len(visits) + 1, "booking_number": b.booking_number,
                       "technician_id": int(rng.integers(1, n_tech + 1)), "visit_date_key": b.service_date_key,
                       "duration_minutes": int(rng.integers(30, 300)), "travel_km": round(float(rng.uniform(1, 60)), 1),
                       "part_name": part, "parts_used_qty": round(float(rng.uniform(0.5, 8.0)), 2),
                       "unit_of_measure": unit})
    visits_frame = pd.DataFrame(visits).sort_values("visit_id")

    tables = [
        TableDef("calendar", "calendar", calendar, CALENDAR_TYPES, ["date_key"], "Calendar"),
        TableDef("booking_statuses", "dimension", statuses,
                 {"status_id": "INTEGER", "status_name": "VARCHAR", "status_group": "VARCHAR"}, ["status_id"],
                 "Booking status"),
        TableDef("customers", "dimension", customers,
                 {"customer_id": "INTEGER", "customer_code": "VARCHAR", "first_name": "VARCHAR",
                  "last_name": "VARCHAR", "email": "VARCHAR", "phone": "VARCHAR", "street_address": "VARCHAR",
                  "birth_date": "DATE", "city": "VARCHAR", "state": "VARCHAR", "signup_date": "DATE"},
                 ["customer_id"], "Customer"),
        TableDef("technicians", "dimension", technicians,
                 {"technician_id": "INTEGER", "employee_code": "VARCHAR", "technician_name": "VARCHAR",
                  "team": "VARCHAR", "mobile_number": "VARCHAR", "hire_date": "DATE"}, ["technician_id"], "Technician"),
        TableDef("partners", "dimension", partners,
                 {"partner_id": "INTEGER", "partner_name": "VARCHAR", "contact_name": "VARCHAR",
                  "office_phone": "VARCHAR", "partner_type": "VARCHAR"}, ["partner_id"], "Partner"),
        TableDef("bookings", "fact", bookings,
                 {"booking_id": "INTEGER", "booking_number": "VARCHAR", "customer_id": "INTEGER",
                  "partner_id": "INTEGER", "booked_date_key": "INTEGER", "service_date_key": "INTEGER",
                  "status_id": "INTEGER", "travel_fee": "DECIMAL(10,2)"}, ["booking_id"], "Booking"),
        TableDef("booking_lines", "fact", booking_lines,
                 {"booking_number": "VARCHAR", "line_number": "INTEGER", "service_type": "VARCHAR",
                  "technician_id": "INTEGER", "booked_date_key": "INTEGER", "hours": "DECIMAL(6,1)",
                  "line_amount": "DECIMAL(12,2)", "travel_fee": "DECIMAL(10,2)"},
                 ["booking_number", "line_number"], "Booking line"),
        TableDef("plan_renewals", "fact", plan_renewals,
                 {"plan_number": "VARCHAR", "renewal_number": "INTEGER", "customer_id": "INTEGER",
                  "renewal_date_key": "INTEGER", "visits_included": "INTEGER", "renewals_allowed": "INTEGER",
                  "plan_fee": "DECIMAL(10,2)"}, ["plan_number", "renewal_number"], "Plan renewal"),
        TableDef("visits", "fact", visits_frame,
                 {"visit_id": "INTEGER", "booking_number": "VARCHAR", "technician_id": "INTEGER",
                  "visit_date_key": "INTEGER", "duration_minutes": "INTEGER", "travel_km": "DECIMAL(6,1)",
                  "part_name": "VARCHAR", "parts_used_qty": "DECIMAL(8,2)", "unit_of_measure": "VARCHAR"},
                 ["visit_id"], "Visit"),
    ]

    truth = Truth(
        kinds={t.name: t.kind for t in tables},
        primary_keys={t.name: t.primary_key for t in tables},
        joins=[
            JoinTruth("bookings", "customer_id", "customers", "customer_id"),
            JoinTruth("bookings", "partner_id", "partners", "partner_id"),
            JoinTruth("bookings", "booked_date_key", "calendar", "date_key", role="Booked date"),
            JoinTruth("bookings", "service_date_key", "calendar", "date_key", role="Service date"),
            JoinTruth("bookings", "status_id", "booking_statuses", "status_id"),
            JoinTruth("booking_lines", "booking_number", "bookings", "booking_number", declared=False),
            JoinTruth("booking_lines", "technician_id", "technicians", "technician_id"),
            JoinTruth("booking_lines", "booked_date_key", "calendar", "date_key"),
            JoinTruth("plan_renewals", "customer_id", "customers", "customer_id"),
            JoinTruth("plan_renewals", "renewal_date_key", "calendar", "date_key"),
            JoinTruth("visits", "booking_number", "bookings", "booking_number", declared=False),
            JoinTruth("visits", "technician_id", "technicians", "technician_id"),
            JoinTruth("visits", "visit_date_key", "calendar", "date_key"),
        ],
        calendar={"table": "calendar", "key": "date_key", "date": "full_date", "attributes": CALENDAR_ATTRIBUTES,
                  "fiscal_year_start_month": 7, "fiscal_year_named_by": "end", "placeholders": []},
        dates=[
            DateTruth("bookings", "booked_date_key", "Booked date", "event", True, default=True),
            DateTruth("bookings", "service_date_key", "Service date", "event", True),
            DateTruth("booking_lines", "booked_date_key", "Booked date", "event", True, default=True),
            DateTruth("plan_renewals", "renewal_date_key", "Renewal date", "event", True, default=True),
            DateTruth("visits", "visit_date_key", "Visit date", "event", True, default=True),
            DateTruth("customers", "signup_date", "Signup date", "event", False, default=True),
            DateTruth("technicians", "hire_date", "Hire date", "event", False, default=True),
        ],
        measures=[
            MeasureTruth("bookings", "travel_fee", "sum", "additive", "currency", "Travel fee"),
            MeasureTruth("bookings", None, "count", "additive", "count", "Bookings"),
            MeasureTruth("booking_lines", "hours", "sum", "additive", "number", "Hours"),
            MeasureTruth("booking_lines", "line_amount", "sum", "additive", "currency", "Line amount"),
            MeasureTruth("booking_lines", "travel_fee", "avg", "non_additive", "currency", "Travel fee"),
            MeasureTruth("booking_lines", None, "count", "additive", "count", "Booking lines"),
            MeasureTruth("plan_renewals", "visits_included", "sum", "additive", "integer", "Visits included"),
            MeasureTruth("plan_renewals", "plan_fee", "sum", "additive", "currency", "Plan fee"),
            MeasureTruth("plan_renewals", None, "count", "additive", "count", "Plan renewals"),
            MeasureTruth("visits", "duration_minutes", "sum", "additive", "integer", "Duration minutes"),
            MeasureTruth("visits", "travel_km", "sum", "additive", "number", "Travel km"),
            MeasureTruth("visits", "parts_used_qty", "sum", "additive", "number", "Parts used",
                         unit_column="visits.unit_of_measure"),
            MeasureTruth("visits", None, "count", "additive", "count", "Visits"),
        ],
        labels={"customers": "first_name+last_name", "technicians": "technician_name", "partners": "partner_name",
                "booking_statuses": "status_name"},
        codes={"customers": "customer_code", "technicians": "employee_code"},
        statuses={"bookings.status_id": {"values": {str(s): n for s, n, _ in STATUSES},
                                         "cancelled": ["4", "6", "7"]}},
        quality=[{"object": "visits.parts_used_qty", "kind": "unit_mix"}],
        not_measures=["booking_lines.line_number", "plan_renewals.renewal_number", "plan_renewals.renewals_allowed",
                      "bookings.customer_id", "bookings.partner_id", "bookings.status_id", "visits.technician_id",
                      "booking_lines.technician_id", "plan_renewals.customer_id"],
        sensitive={"customers.email": "pii", "customers.phone": "pii", "customers.street_address": "pii",
                   "customers.birth_date": "pii", "technicians.mobile_number": "pii", "partners.office_phone": "pii"},
    )
    return Domain("home_services", tables, truth, TODAY,
                  "A home-services business: households book cleaning and repair work, done by technicians, "
                  "sometimes through partner companies; maintenance plans are renewed every quarter.")
