"""Compounding pharmacy: prescriptions, fills, claims, compounded batches, quality tests and ingredient stock.

A benchmark domain (not in the gated set): it plants what a compounding pharmacy's warehouse
holds and what today's learning is known to miss, so the miss is measured, not hidden.

What this warehouse is built to test:

* claims reach the fill they bill by two columns, the prescription number and the fill
  number (a fill number alone repeats on every prescription);
* pharmacists are keyed by a whole number on their own table and by a zero-padded text
  code on fills ("0042" against 42): the same people, stored as different types;
* formulas and ingredients meet in a recipe table (many-to-many), whose quantity per unit
  is a recipe parameter, not a metric;
* quantities dispensed and ingredients in stock are counted in g, mL and EA: never added
  across units;
* ingredient unit cost is a price (averaged, never summed), batch yield and test potency
  are percentages;
* ingredient stock is a weekly snapshot (taken at the last week of a period, never added
  up over weeks) whose newest week holds a fraction of the lots: a partial load, not a
  collapse in stock;
* beyond-use dates are when a batch stops being usable (validity), never the batch's event;
* rows carry load and creation stamps that are audit dates, never a default;
* about 10% of fills come from stock without a batch (no batch, not an orphan), and a
  claim's paid date is empty until it is paid;
* patients' names and birth dates are personal data.

Prescriptions are written from January 2025; questions are asked on 15 June 2026.
"""

from __future__ import annotations

import bisect
import datetime as dt

import numpy as np
import pandas as pd

from evals.core2.framework import DateTruth, Domain, JoinTruth, MeasureTruth, TableDef, Truth

TODAY = dt.date(2026, 6, 15)
FIRST_DAY = dt.date(2025, 1, 2)
LAST_DAY = dt.date(2026, 6, 12)

FORMS = [("Cream", "Topical", "g"), ("Gel", "Topical", "g"), ("Capsule", "Oral", "EA"),
         ("Suspension", "Oral", "mL"), ("Troche", "Buccal", "EA"), ("Solution", "Ophthalmic", "mL")]
ACTIVES = ["Ketoprofen", "Gabapentin", "Estradiol", "Progesterone", "Baclofen", "Lidocaine", "Tacrolimus",
           "Naltrexone", "Omeprazole", "Spironolactone", "Testosterone", "Hydroquinone", "Metronidazole",
           "Diclofenac", "Clobetasol"]
BASES = [("Versabase cream", "g"), ("Lipoderm base", "g"), ("Ora-Plus vehicle", "mL"), ("Syrspend vehicle", "mL"),
         ("Microcrystalline cellulose", "g"), ("Purified water", "mL"), ("Gelatin capsule shell", "EA"),
         ("Troche base", "g")]
FIRST_NAMES = ["Avery", "Jordan", "Riley", "Casey", "Morgan", "Quinn", "Rowan", "Sage", "Emerson", "Hayden", "Kai",
               "Logan", "Parker", "Reese", "Skyler", "Tatum"]
LAST_NAMES = ["Ashford", "Bellamy", "Calloway", "Dunmore", "Ellery", "Fairbanks", "Galloway", "Hollis", "Ingram",
              "Jessup", "Kendrick", "Lockhart", "Merriweather", "Northcott", "Oakley", "Pembrook"]
SPECIALTIES = ["Dermatology", "Endocrinology", "Pain management", "Gastroenterology", "Veterinary", "Family medicine"]
PAYERS = [("Cash", "Cash"), ("Northwind Health Plan", "Commercial"), ("Harborview Mutual", "Commercial"),
          ("State Medicaid", "Medicaid"), ("Medicare Part D", "Medicare"), ("Summit Benefit Services", "Commercial")]


def build(seed: int = 7) -> Domain:
    rng = np.random.default_rng(seed)
    days = (LAST_DAY - FIRST_DAY).days

    # ── people, payers ───────────────────────────────────────────────────────
    payers = pd.DataFrame({"payer_id": np.arange(1, len(PAYERS) + 1), "payer_name": [p for p, _ in PAYERS],
                           "payer_type": [t for _, t in PAYERS]})
    n_rx_by = 120
    prescribers = pd.DataFrame({
        "prescriber_id": np.arange(1, n_rx_by + 1),
        "prescriber_npi": [f"{1000000000 + 7919 * i:010d}" for i in range(1, n_rx_by + 1)],
        "prescriber_name": [f"Dr. {FIRST_NAMES[i % 16]} {LAST_NAMES[(i * 7) % 16]} {i}" for i in range(n_rx_by)],
        "specialty": rng.choice(SPECIALTIES, n_rx_by),
        "clinic_name": [f"{LAST_NAMES[(i * 3) % 16]} Clinic {i % 30 + 1}" for i in range(n_rx_by)]})
    n_patients = 1500
    patients = pd.DataFrame({
        "patient_id": np.arange(1, n_patients + 1),
        "patient_number": [f"P-{100000 + i}" for i in range(n_patients)],
        "first_name": rng.choice(FIRST_NAMES, n_patients), "last_name": rng.choice(LAST_NAMES, n_patients),
        "birth_date": [dt.date(1940, 1, 1) + dt.timedelta(days=int(d)) for d in rng.integers(0, 29000, n_patients)],
        "state": rng.choice(["NY", "NJ", "PA", "CT", "MA"], n_patients, p=[0.4, 0.25, 0.15, 0.1, 0.1]),
        "payer_id": rng.choice(payers["payer_id"], n_patients, p=[0.25, 0.25, 0.15, 0.15, 0.12, 0.08])})
    pharmacists = pd.DataFrame({
        "pharmacist_id": np.arange(1, 19),
        "pharmacist_name": [f"{FIRST_NAMES[i % 16]} {LAST_NAMES[(i * 5 + 3 + i // 16) % 16]}" for i in range(18)],
        "role": ["Pharmacist"] * 10 + ["Technician"] * 8,
        "license_state": rng.choice(["NY", "NJ"], 18)})

    # ── formulas, ingredients and their recipes (many-to-many) ──────────────
    ingredients_rows = []
    for name in ACTIVES:
        ingredients_rows.append((name + " powder", "g", round(float(rng.uniform(0.8, 45.0)), 4)))
    for name, unit in BASES:
        ingredients_rows.append((name, unit, round(float(rng.uniform(0.01, 0.6)), 4)))
    ingredients = pd.DataFrame({
        "ingredient_id": np.arange(1, len(ingredients_rows) + 1),
        "ingredient_code": [f"ING-{300 + i}" for i in range(len(ingredients_rows))],
        "ingredient_name": [n for n, _, _ in ingredients_rows],
        "unit_of_measure": [u for _, u, _ in ingredients_rows],
        "unit_cost": [c for _, _, c in ingredients_rows]})
    n_formulas = 60
    formula_forms = rng.integers(0, len(FORMS), n_formulas)
    formulas = pd.DataFrame({
        "formula_id": np.arange(1, n_formulas + 1),
        "formula_code": [f"F-{2000 + i}" for i in range(n_formulas)],
        "formula_name": [f"{ACTIVES[i % len(ACTIVES)]} {round(0.5 + (i // len(ACTIVES)) * 0.75, 2)}% {FORMS[f][0]}"
                         for i, f in enumerate(formula_forms)],
        "dosage_form": [FORMS[f][0] for f in formula_forms], "route": [FORMS[f][1] for f in formula_forms],
        "beyond_use_days": rng.choice([14, 30, 90, 180], n_formulas)})
    recipe = []
    for fid, form in zip(formulas["formula_id"], formula_forms):
        unit = FORMS[form][2]
        active = int((fid - 1) % len(ACTIVES)) + 1
        recipe.append((int(fid), active, round(float(rng.uniform(0.005, 0.1)), 4), "g"))
        base = next(i for i, (_, u) in enumerate(BASES, start=len(ACTIVES) + 1) if u == unit or unit == "EA")
        recipe.append((int(fid), base, round(float(rng.uniform(0.85, 0.99)), 4), unit if unit != "EA" else "g"))
        if rng.random() < 0.4:
            extra = int(rng.integers(len(ACTIVES) + 1, len(ingredients_rows) + 1))
            if extra not in (active, base):
                recipe.append((int(fid), extra, round(float(rng.uniform(0.001, 0.05)), 4), "g"))
    formula_ingredients = pd.DataFrame(recipe, columns=["formula_id", "ingredient_id", "quantity_per_unit",
                                                        "unit_of_measure"])

    # ── compounded batches and their quality tests ──────────────────────────
    n_batches = 1500
    batch_formula = rng.integers(1, n_formulas + 1, n_batches)
    compounded = sorted(FIRST_DAY - dt.timedelta(days=20) + dt.timedelta(days=int(d))
                        for d in rng.integers(0, days + 20, n_batches))
    bud_days = formulas.set_index("formula_id")["beyond_use_days"]
    batch_unit = [FORMS[formula_forms[f - 1]][2] for f in batch_formula]
    size = rng.choice([250.0, 500.0, 1000.0, 2000.0], n_batches)
    batches = pd.DataFrame({
        "batch_id": np.arange(1, n_batches + 1),
        "batch_number": [f"B{d:%y%m%d}-{i:04d}" for i, d in enumerate(compounded)],
        "formula_id": batch_formula, "compounded_date": compounded,
        "beyond_use_date": [d + dt.timedelta(days=int(bud_days[f])) for d, f in zip(compounded, batch_formula)],
        "batch_size": size, "unit_of_measure": batch_unit,
        "yield_pct": np.round(rng.uniform(92.0, 99.8, n_batches), 2),
        "waste_qty": np.round(size * rng.uniform(0.002, 0.08, n_batches), 2),
        "technician_id": rng.integers(11, 19, n_batches)})
    tests = []
    for b in batches.itertuples():
        for test_type in (["Potency", "Sterility"] if rng.random() < 0.6 else ["Potency"]):
            tested = b.compounded_date + dt.timedelta(days=int(rng.integers(1, 6)))
            potency = round(float(rng.normal(100.0, 3.5)), 2) if test_type == "Potency" else None
            passed = (potency is None or 90.0 <= potency <= 110.0) and rng.random() > 0.01
            tests.append((len(tests) + 1, int(b.batch_id), test_type, tested, potency, "PASS" if passed else "FAIL"))
    quality_tests = pd.DataFrame(tests, columns=["test_id", "batch_id", "test_type", "tested_date", "potency_pct",
                                                 "result"])

    # ── prescriptions, fills and claims ─────────────────────────────────────
    n_rx = 6000
    written = sorted(FIRST_DAY + dt.timedelta(days=int(d)) for d in rng.integers(0, days - 10, n_rx))
    rx_formula = rng.integers(1, n_formulas + 1, n_rx)
    received = [pd.Timestamp(d) + pd.Timedelta(minutes=int(m)) for d, m in zip(written, rng.integers(60, 4 * 1440, n_rx))]
    prescriptions = pd.DataFrame({
        "rx_number": [f"RX{700000 + i}" for i in range(n_rx)],
        "patient_id": rng.integers(1, n_patients + 1, n_rx),
        "prescriber_npi": rng.choice(prescribers["prescriber_npi"], n_rx),
        "formula_id": rx_formula, "written_date": written, "received_ts": received,
        "quantity_prescribed": rng.choice([30.0, 60.0, 90.0, 120.0], n_rx),
        "refills_authorized": rng.choice([0, 1, 2, 3, 5, 11], n_rx, p=[0.25, 0.2, 0.2, 0.15, 0.1, 0.1]),
        "days_supply": rng.choice([30, 60, 90], n_rx, p=[0.7, 0.2, 0.1]),
        # The row's creation in the dispensing system: a load stamp, many rows at one time.
        "created_at": [pd.Timestamp(r.date()) + pd.Timedelta(hours=23, minutes=55) for r in received]})
    # Per formula, its batches by compounded date: the newest one compounded by a day and not past its use-by.
    batches_by_formula = {int(f): (list(g["compounded_date"]), list(g["beyond_use_date"]), list(g["batch_id"]))
                          for f, g in batches.sort_values("compounded_date").groupby("formula_id")}

    def usable_batch(formula_id: int, day: dt.date) -> int | None:
        made, until, ids = batches_by_formula.get(formula_id, ([], [], []))
        for i in range(bisect.bisect_right(made, day) - 1, -1, -1):
            if until[i] >= day:
                return int(ids[i])
            if day - made[i] > dt.timedelta(days=200):
                break
        return None
    fills, claims = [], []
    patient_payer = patients.set_index("patient_id")["payer_id"]
    for rx in prescriptions.itertuples():
        refills = 1 + min(int(rx.refills_authorized), int(rng.integers(0, 4)))
        day = pd.Timestamp(rx.received_ts).date() + dt.timedelta(days=int(rng.integers(1, 4)))
        unit = FORMS[formula_forms[rx.formula_id - 1]][2]
        for fill_number in range(1, refills + 1):
            if day > LAST_DAY:
                break
            batch = usable_batch(int(rx.formula_id), day) if rng.random() > 0.1 else None
            quantity = float(rx.quantity_prescribed)
            cost = round(quantity * float(rng.uniform(0.15, 1.8)), 2)
            price = round(cost * float(rng.uniform(1.6, 3.2)), 2)
            ship = day + dt.timedelta(days=int(rng.integers(0, 4)))
            payer = int(patient_payer[rx.patient_id])
            copay = round(price if payer == 1 else float(rng.choice([0.0, 10.0, 25.0, 40.0])), 2)
            fills.append({"rx_number": rx.rx_number, "fill_number": fill_number, "batch_id": batch,
                          "fill_date": day, "ship_date": ship, "quantity_dispensed": quantity,
                          "unit_of_measure": unit, "ingredient_cost": cost, "price_charged": price,
                          "copay_amount": copay,
                          "pharmacist_code": f"{int(rng.integers(1, 11)):04d}",
                          "load_ts": pd.Timestamp(day) + pd.Timedelta(days=1, hours=2)})
            if payer != 1:
                submitted = day + dt.timedelta(days=int(rng.integers(0, 3)))
                status = rng.choice(["PAID", "REJ", "PEND"], p=[0.82, 0.1, 0.08])
                paid = submitted + dt.timedelta(days=int(rng.integers(7, 45))) if status == "PAID" else None
                if paid is not None and paid > LAST_DAY:
                    paid, status = None, "PEND"
                billed = round(price - copay, 2)
                claims.append({"claim_id": len(claims) + 1, "rx_number": rx.rx_number, "fill_number": fill_number,
                               "payer_id": payer, "submitted_date": submitted, "paid_date": paid,
                               "claim_status": status, "billed_amount": billed,
                               "paid_amount": round(billed * float(rng.uniform(0.6, 0.95)), 2) if status == "PAID" else 0.0,
                               "reject_code": str(rng.choice(["75", "76", "79", "88"])) if status == "REJ" else None})
            day = day + dt.timedelta(days=int(rx.days_supply))
    fills_frame = pd.DataFrame(fills)
    claims_frame = pd.DataFrame(claims)

    # ── weekly ingredient stock, newest week a partial load ─────────────────
    lots = [(int(i), f"L{2400 + 3 * int(i) + k}") for i in ingredients["ingredient_id"] for k in range(2)]
    weeks = [d.date() for d in pd.date_range(FIRST_DAY, LAST_DAY, freq="W-SUN")]
    stock = []
    unit_of = ingredients.set_index("ingredient_id")["unit_of_measure"]
    level = {lot: float(rng.uniform(200, 4000)) for lot in lots}
    for w, week in enumerate(weeks):
        kept = lots if w < len(weeks) - 1 else lots[:5]            # the last load stopped early
        for ingredient_id, lot in kept:
            level[(ingredient_id, lot)] = max(0.0, level[(ingredient_id, lot)] * float(rng.uniform(0.9, 1.08)))
            stock.append((week, ingredient_id, lot, round(level[(ingredient_id, lot)], 2), unit_of[ingredient_id]))
    stock_weekly = pd.DataFrame(stock, columns=["snapshot_date", "ingredient_id", "lot_number", "on_hand_qty",
                                                "unit_of_measure"])

    tables = [
        TableDef("payers", "dimension", payers, {"payer_id": "INTEGER", "payer_name": "VARCHAR",
                                                 "payer_type": "VARCHAR"}, ["payer_id"], "Payer"),
        TableDef("prescribers", "dimension", prescribers,
                 {"prescriber_id": "INTEGER", "prescriber_npi": "VARCHAR", "prescriber_name": "VARCHAR",
                  "specialty": "VARCHAR", "clinic_name": "VARCHAR"}, ["prescriber_id"], "Prescriber"),
        TableDef("patients", "dimension", patients,
                 {"patient_id": "INTEGER", "patient_number": "VARCHAR", "first_name": "VARCHAR",
                  "last_name": "VARCHAR", "birth_date": "DATE", "state": "VARCHAR", "payer_id": "INTEGER"},
                 ["patient_id"], "Patient"),
        TableDef("pharmacists", "dimension", pharmacists,
                 {"pharmacist_id": "INTEGER", "pharmacist_name": "VARCHAR", "role": "VARCHAR",
                  "license_state": "VARCHAR"}, ["pharmacist_id"], "Pharmacist"),
        TableDef("ingredients", "dimension", ingredients,
                 {"ingredient_id": "INTEGER", "ingredient_code": "VARCHAR", "ingredient_name": "VARCHAR",
                  "unit_of_measure": "VARCHAR", "unit_cost": "DECIMAL(12,4)"}, ["ingredient_id"], "Ingredient"),
        TableDef("formulas", "dimension", formulas,
                 {"formula_id": "INTEGER", "formula_code": "VARCHAR", "formula_name": "VARCHAR",
                  "dosage_form": "VARCHAR", "route": "VARCHAR", "beyond_use_days": "INTEGER"},
                 ["formula_id"], "Formula"),
        TableDef("formula_ingredients", "bridge", formula_ingredients,
                 {"formula_id": "INTEGER", "ingredient_id": "INTEGER", "quantity_per_unit": "DECIMAL(10,4)",
                  "unit_of_measure": "VARCHAR"}, ["formula_id", "ingredient_id"], "Formula ingredient"),
        TableDef("batches", "fact", batches,
                 {"batch_id": "INTEGER", "batch_number": "VARCHAR", "formula_id": "INTEGER",
                  "compounded_date": "DATE", "beyond_use_date": "DATE", "batch_size": "DECIMAL(10,2)",
                  "unit_of_measure": "VARCHAR", "yield_pct": "DECIMAL(5,2)", "waste_qty": "DECIMAL(10,2)",
                  "technician_id": "INTEGER"}, ["batch_id"], "Batch"),
        TableDef("quality_tests", "fact", quality_tests,
                 {"test_id": "INTEGER", "batch_id": "INTEGER", "test_type": "VARCHAR", "tested_date": "DATE",
                  "potency_pct": "DECIMAL(6,2)", "result": "VARCHAR"}, ["test_id"], "Quality test"),
        TableDef("prescriptions", "fact", prescriptions,
                 {"rx_number": "VARCHAR", "patient_id": "INTEGER", "prescriber_npi": "VARCHAR",
                  "formula_id": "INTEGER", "written_date": "DATE", "received_ts": "TIMESTAMP",
                  "quantity_prescribed": "DECIMAL(10,2)", "refills_authorized": "INTEGER",
                  "days_supply": "INTEGER", "created_at": "TIMESTAMP"}, ["rx_number"], "Prescription"),
        TableDef("fills", "fact", fills_frame,
                 {"rx_number": "VARCHAR", "fill_number": "INTEGER", "batch_id": "INTEGER", "fill_date": "DATE",
                  "ship_date": "DATE", "quantity_dispensed": "DECIMAL(10,2)", "unit_of_measure": "VARCHAR",
                  "ingredient_cost": "DECIMAL(12,2)", "price_charged": "DECIMAL(12,2)",
                  "copay_amount": "DECIMAL(12,2)", "pharmacist_code": "VARCHAR", "load_ts": "TIMESTAMP"},
                 ["rx_number", "fill_number"], "Fill"),
        TableDef("claims", "fact", claims_frame,
                 {"claim_id": "INTEGER", "rx_number": "VARCHAR", "fill_number": "INTEGER", "payer_id": "INTEGER",
                  "submitted_date": "DATE", "paid_date": "DATE", "claim_status": "VARCHAR",
                  "billed_amount": "DECIMAL(12,2)", "paid_amount": "DECIMAL(12,2)", "reject_code": "VARCHAR"},
                 ["claim_id"], "Claim"),
        TableDef("ingredient_stock_weekly", "snapshot", stock_weekly,
                 {"snapshot_date": "DATE", "ingredient_id": "INTEGER", "lot_number": "VARCHAR",
                  "on_hand_qty": "DECIMAL(12,2)", "unit_of_measure": "VARCHAR"},
                 ["snapshot_date", "ingredient_id", "lot_number"], "Ingredient stock"),
    ]

    truth = Truth(
        kinds={t.name: t.kind for t in tables},
        primary_keys={t.name: t.primary_key for t in tables},
        joins=[
            JoinTruth("patients", "payer_id", "payers", "payer_id"),
            JoinTruth("formula_ingredients", "formula_id", "formulas", "formula_id"),
            JoinTruth("formula_ingredients", "ingredient_id", "ingredients", "ingredient_id"),
            JoinTruth("batches", "formula_id", "formulas", "formula_id"),
            JoinTruth("batches", "technician_id", "pharmacists", "pharmacist_id"),
            JoinTruth("quality_tests", "batch_id", "batches", "batch_id"),
            JoinTruth("prescriptions", "patient_id", "patients", "patient_id"),
            JoinTruth("prescriptions", "prescriber_npi", "prescribers", "prescriber_npi"),
            JoinTruth("prescriptions", "formula_id", "formulas", "formula_id"),
            JoinTruth("fills", "rx_number", "prescriptions", "rx_number"),
            JoinTruth("fills", "batch_id", "batches", "batch_id"),
            JoinTruth("fills", "pharmacist_code", "pharmacists", "pharmacist_id", declared=False, cast=True),
            JoinTruth("claims", "rx_number", "fills", "rx_number", also=(("fill_number", "fill_number"),)),
            JoinTruth("claims", "rx_number", "prescriptions", "rx_number", declared=False),
            JoinTruth("claims", "payer_id", "payers", "payer_id"),
            JoinTruth("ingredient_stock_weekly", "ingredient_id", "ingredients", "ingredient_id"),
        ],
        calendar=None,
        dates=[
            DateTruth("batches", "compounded_date", "Compounded date", "event", False, default=True),
            DateTruth("batches", "beyond_use_date", "Beyond-use date", "validity", False),
            DateTruth("quality_tests", "tested_date", "Tested date", "event", False, default=True),
            DateTruth("prescriptions", "written_date", "Written date", "event", False, default=True),
            DateTruth("prescriptions", "received_ts", "Received time", "event", False),
            DateTruth("prescriptions", "created_at", "Created at", "audit", False),
            DateTruth("fills", "fill_date", "Fill date", "event", False, default=True),
            DateTruth("fills", "ship_date", "Ship date", "event", False),
            DateTruth("fills", "load_ts", "Load time", "audit", False),
            DateTruth("claims", "submitted_date", "Submitted date", "event", False, default=True),
            DateTruth("claims", "paid_date", "Paid date", "event", False),
            DateTruth("ingredient_stock_weekly", "snapshot_date", "Snapshot date", "snapshot", False, default=True),
        ],
        measures=[
            MeasureTruth("batches", "batch_size", "sum", "additive", "number", "Batch size",
                         unit_column="batches.unit_of_measure"),
            MeasureTruth("batches", "waste_qty", "sum", "additive", "number", "Waste quantity",
                         unit_column="batches.unit_of_measure"),
            MeasureTruth("batches", "yield_pct", "avg", "non_additive", "percent", "Yield"),
            MeasureTruth("batches", None, "count", "additive", "count", "Batches"),
            MeasureTruth("quality_tests", "potency_pct", "avg", "non_additive", "percent", "Potency"),
            MeasureTruth("quality_tests", None, "count", "additive", "count", "Quality tests"),
            MeasureTruth("prescriptions", None, "count", "additive", "count", "Prescriptions"),
            MeasureTruth("fills", "quantity_dispensed", "sum", "additive", "number", "Quantity dispensed",
                         unit_column="fills.unit_of_measure"),
            MeasureTruth("fills", "ingredient_cost", "sum", "additive", "currency", "Ingredient cost"),
            MeasureTruth("fills", "price_charged", "sum", "additive", "currency", "Price charged"),
            MeasureTruth("fills", "copay_amount", "sum", "additive", "currency", "Copay amount"),
            MeasureTruth("fills", None, "count", "additive", "count", "Fills"),
            MeasureTruth("claims", "billed_amount", "sum", "additive", "currency", "Billed amount"),
            MeasureTruth("claims", "paid_amount", "sum", "additive", "currency", "Paid amount"),
            MeasureTruth("claims", None, "count", "additive", "count", "Claims"),
            MeasureTruth("ingredient_stock_weekly", "on_hand_qty", "sum", "semi_additive", "number", "On hand",
                         unit_column="ingredient_stock_weekly.unit_of_measure", time_aggregation="last"),
        ],
        labels={"payers": "payer_name", "prescribers": "prescriber_name", "pharmacists": "pharmacist_name",
                "ingredients": "ingredient_name", "formulas": "formula_name"},
        codes={"ingredients": "ingredient_code", "formulas": "formula_code", "batches": "batch_number"},
        statuses={"claims.claim_status": {"values": {"PAID": "Paid", "REJ": "Rejected", "PEND": "Pending"},
                                          "cancelled": []},
                  "quality_tests.result": {"values": {"PASS": "Passed", "FAIL": "Failed"}, "cancelled": []}},
        quality=[
            {"object": "fills.quantity_dispensed", "kind": "unit_mix"},
            {"object": "ingredient_stock_weekly", "kind": "partial_snapshot"},
            {"object": "prescriptions.created_at", "kind": "load_timestamp"},
        ],
        not_measures=["formula_ingredients.quantity_per_unit", "formulas.beyond_use_days",
                      "prescriptions.patient_id", "prescriptions.formula_id",
                      "fills.fill_number", "fills.batch_id", "claims.claim_id", "claims.fill_number",
                      "claims.payer_id", "batches.technician_id", "batches.formula_id", "quality_tests.batch_id",
                      "ingredient_stock_weekly.ingredient_id"],
        sensitive={"patients.first_name": "pii", "patients.last_name": "pii", "patients.birth_date": "pii"},
    )
    return Domain("compounding_pharmacy", tables, truth, TODAY,
                  "A compounding pharmacy's prescriptions, fills, insurance claims, compounded batches, quality "
                  "tests and ingredient stock.",
                  abbreviations={"prescription": "RX", "prescriptions": "RX", "prescriber": "PRSCR",
                                 "pharmacist": "PHRM", "pharmacists": "PHRM", "ingredient": "INGR",
                                 "ingredients": "INGR", "formula": "FRML", "formulas": "FRML", "compounded": "CMPD",
                                 "potency": "PTNCY", "beyond": "BYND", "copay": "COPAY", "claim": "CLM",
                                 "claims": "CLM", "dispensed": "DSPN", "payer": "PYR", "payers": "PYR"})
