"""HR: employees, a month-end headcount snapshot and a biweekly payroll.

What this warehouse is built to test:

* a monthly snapshot keyed by the month's last day (a DATE): headcount and FTE are
  levels taken at month end, never added up over months, while the month's salary
  on the same rows is an amount for the month, added up over time;
* an employee list dated by hire date (the default) and termination date, where an
  employee still working has the termination date 9999-12-31 and an unknown birth
  date is 1900-01-01: placeholders, never dates (a series of leavers ends in 2026);
* employees who report to a manager in the same table (a self-join);
* payroll paid every other Friday, with the department the pay was booked to
  (which differs from the employee's current department after a transfer), so a
  department is reached two ways and the row's own must win;
* a location reached only through the employee (a snowflaked path);
* personal data (names, e-mail addresses, birth dates) beside the measures;
* department names with symbols and analysis words ("Sales & Marketing",
  "Customer Care #2", "Weekly Ops Planning", "Top Talent Office").

Snapshots run from January 2023 to May 2026 (June is not closed), payroll from
6 January 2023 to 12 June 2026; questions are asked on 15 June 2026.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from evals.core2.framework import DateTruth, Domain, JoinTruth, MeasureTruth, TableDef, Truth

TODAY = dt.date(2026, 6, 15)
FIRST_MONTH_END = dt.date(2023, 1, 31)
LAST_MONTH_END = dt.date(2026, 5, 31)
FIRST_PAY = dt.date(2023, 1, 6)
LAST_PAY = dt.date(2026, 6, 12)
STILL_WORKING = dt.date(9999, 12, 31)
UNKNOWN_BIRTH = dt.date(1900, 1, 1)

DEPARTMENTS = ["Research & Development", "Sales & Marketing", "People & Culture", "Finance", "Plant Operations",
               "Customer Care #1", "Customer Care #2", "IT Services", "Weekly Ops Planning", "Top Talent Office"]
LOCATIONS = [("Toronto", "Canada"), ("Montreal", "Canada"), ("Vancouver", "Canada"), ("Austin", "USA"),
             ("Monterrey", "Mexico")]
_FIRST = ["Alex", "Sam", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie", "Avery", "Quinn", "Drew", "Robin",
          "Kai", "Noor", "Mira", "Omar", "Lena", "Ivan", "Priya", "Chen"]
_LAST = ["Martin", "Roy", "Tremblay", "Gagnon", "Wong", "Singh", "Lopez", "Novak", "Silva", "Kim", "Okafor", "Haddad",
         "Berg", "Costa", "Ito", "Nash", "Reyes", "Volk", "Patel", "Dubois"]
LEVELS = {"L1": 52_000, "L2": 64_000, "L3": 78_000, "L4": 96_000, "L5": 120_000, "L6": 155_000}


def _month_ends(first: dt.date, last: dt.date) -> list[dt.date]:
    return [d.date() for d in pd.date_range(first, last, freq="ME")]


def build(seed: int = 7) -> Domain:
    rng = np.random.default_rng(seed)
    departments = pd.DataFrame({"department_id": np.arange(1, len(DEPARTMENTS) + 1), "department_name": DEPARTMENTS})
    locations = pd.DataFrame({"location_id": np.arange(1, len(LOCATIONS) + 1),
                              "location_name": [n for n, _ in LOCATIONS], "country": [c for _, c in LOCATIONS]})

    # ── employees: 380 working on 1 January 2023, then hires and leavers each month ──
    rows: list[dict] = []
    taken: set[str] = set()

    def hire(day: dt.date) -> None:
        level = str(rng.choice(list(LEVELS), p=[0.22, 0.28, 0.22, 0.14, 0.09, 0.05]))
        kind = str(rng.choice(["FT", "PT", "CT"], p=[0.8, 0.12, 0.08]))
        n = len(rows) + 1
        first, last = _FIRST[n % len(_FIRST)], _LAST[(n * 7 + n // len(_FIRST)) % len(_LAST)]
        name = f"{first} {last}"
        initial = 0
        while name in taken:                        # a second Alex Martin is told apart by a middle initial
            name = f"{first} {chr(65 + initial)}. {last}"
            initial += 1
        taken.add(name)
        born = day.replace(year=day.year - int(rng.integers(21, 60)), day=min(day.day, 28))
        rows.append({
            "employee_id": n, "employee_number": f"E{10000 + n}", "employee_name": name,
            "email": f"{first.lower()}.{last.lower()}{n}@example.com",
            "birth_date": UNKNOWN_BIRTH if rng.random() < 0.03 else born, "hire_date": day,
            "termination_date": STILL_WORKING, "department_id": int(rng.integers(1, len(DEPARTMENTS) + 1)),
            "location_id": int(rng.choice(np.arange(1, 6), p=[0.4, 0.2, 0.15, 0.15, 0.1])),
            "employment_type": kind, "job_level": level, "status": "A",
            "salary": LEVELS[level] * float(rng.uniform(0.9, 1.15)), "fte": 0.6 if kind == "PT" else 1.0,
            "moved": None})

    for _ in range(380):
        hire(dt.date(2012, 1, 1) + dt.timedelta(days=int(rng.integers(0, 4000))))
    months = _month_ends(dt.date(2023, 1, 1), dt.date(2026, 6, 30))
    for end in months:
        start = end.replace(day=1)
        last_day = min(end, LAST_PAY)
        for r in rows:                      # leavers this month
            if r["termination_date"] == STILL_WORKING and r["hire_date"] < start and rng.random() < 0.011:
                r["termination_date"] = start + dt.timedelta(days=int(rng.integers(0, (last_day - start).days + 1)))
                r["status"] = "T"
        for _ in range(int(rng.poisson(5))):
            hire(start + dt.timedelta(days=int(rng.integers(0, (last_day - start).days + 1))))
    for r in rows:                          # 6% moved department once; the list holds the current one
        if rng.random() < 0.06:
            r["moved"] = (months[int(rng.integers(3, len(months) - 3))], r["department_id"])
            r["department_id"] = int(rng.integers(1, len(DEPARTMENTS) + 1))
    by_department: dict[int, list[dict]] = {}
    for r in sorted(rows, key=lambda r: r["hire_date"]):
        team = by_department.setdefault(r["department_id"], [])
        r["manager_id"] = int(team[int(rng.integers(0, min(len(team), 4)))]["employee_id"]) if team else None
        team.append(r)

    def department_on(r: dict, day: dt.date) -> int:
        if r["moved"] and day < r["moved"][0]:
            return int(r["moved"][1])
        return int(r["department_id"])

    def working(r: dict, day: dt.date) -> bool:
        return r["hire_date"] <= day and (r["termination_date"] == STILL_WORKING or r["termination_date"] > day)

    def salary_on(r: dict, day: dt.date) -> float:
        raises = max(0, day.year - 2023 + (1 if day.month >= 4 else 0) - 1)
        return r["salary"] * 1.03 ** raises

    snapshot = []
    for end in _month_ends(FIRST_MONTH_END, LAST_MONTH_END):
        for r in rows:
            if working(r, end):
                snapshot.append((end, r["employee_id"], department_on(r, end), r["location_id"], 1, r["fte"],
                                 round(salary_on(r, end) * r["fte"] / 12, 2)))
    headcount = pd.DataFrame(snapshot, columns=["month_end_date", "employee_id", "department_id", "location_id",
                                                "headcount", "fte", "monthly_salary"])

    pay = []
    for day in pd.date_range(FIRST_PAY, LAST_PAY, freq="14D").date:
        for r in rows:
            if working(r, day):
                gross = salary_on(r, day) * r["fte"] / 26
                overtime = int(rng.poisson(3)) if r["job_level"] in ("L1", "L2") and r["employment_type"] != "CT" else 0
                gross = round(gross + overtime * salary_on(r, day) / 2080 * 1.5, 2)
                tax = round(gross * 0.24, 2)
                pay.append((day, r["employee_id"], department_on(r, day), gross, tax, round(gross - tax, 2), overtime))
    payroll = pd.DataFrame(pay, columns=["pay_date", "employee_id", "department_id", "gross_pay", "tax_withheld",
                                         "net_pay", "overtime_hours"])
    payroll.insert(0, "payroll_line_id", np.arange(1, len(payroll) + 1))

    employees = pd.DataFrame(rows)[["employee_id", "employee_number", "employee_name", "email", "birth_date",
                                    "hire_date", "termination_date", "department_id", "location_id", "manager_id",
                                    "employment_type", "job_level", "status"]]
    employees["manager_id"] = employees["manager_id"].astype("Int64")

    tables = [
        TableDef("departments", "dimension", departments, {"department_id": "INTEGER", "department_name": "VARCHAR"},
                 ["department_id"], "Department"),
        TableDef("locations", "dimension", locations,
                 {"location_id": "INTEGER", "location_name": "VARCHAR", "country": "VARCHAR"}, ["location_id"],
                 "Location"),
        TableDef("employees", "dimension", employees,
                 {"employee_id": "INTEGER", "employee_number": "VARCHAR", "employee_name": "VARCHAR", "email": "VARCHAR",
                  "birth_date": "DATE", "hire_date": "DATE", "termination_date": "DATE", "department_id": "INTEGER",
                  "location_id": "INTEGER", "manager_id": "INTEGER", "employment_type": "VARCHAR",
                  "job_level": "VARCHAR", "status": "VARCHAR"}, ["employee_id"], "Employee"),
        TableDef("headcount_monthly", "snapshot", headcount,
                 {"month_end_date": "DATE", "employee_id": "INTEGER", "department_id": "INTEGER",
                  "location_id": "INTEGER", "headcount": "INTEGER", "fte": "DECIMAL(4,2)",
                  "monthly_salary": "DECIMAL(12,2)"}, ["month_end_date", "employee_id"], "Headcount"),
        TableDef("payroll", "fact", payroll,
                 {"payroll_line_id": "INTEGER", "pay_date": "DATE", "employee_id": "INTEGER", "department_id": "INTEGER",
                  "gross_pay": "DECIMAL(12,2)", "tax_withheld": "DECIMAL(12,2)", "net_pay": "DECIMAL(12,2)",
                  "overtime_hours": "INTEGER"}, ["payroll_line_id"], "Payroll line"),
    ]

    truth = Truth(
        kinds={t.name: t.kind for t in tables},
        primary_keys={t.name: t.primary_key for t in tables},
        joins=[
            JoinTruth("employees", "department_id", "departments", "department_id"),
            JoinTruth("employees", "location_id", "locations", "location_id"),
            JoinTruth("employees", "manager_id", "employees", "employee_id", "Manager"),
            JoinTruth("headcount_monthly", "employee_id", "employees", "employee_id"),
            JoinTruth("headcount_monthly", "department_id", "departments", "department_id"),
            JoinTruth("headcount_monthly", "location_id", "locations", "location_id"),
            JoinTruth("payroll", "employee_id", "employees", "employee_id"),
            JoinTruth("payroll", "department_id", "departments", "department_id"),
        ],
        calendar=None,
        dates=[
            DateTruth("employees", "hire_date", "Hire date", "event", False, default=True),
            DateTruth("employees", "termination_date", "Termination date", "event", False),
            DateTruth("headcount_monthly", "month_end_date", "Month end date", "snapshot", False, default=True),
            DateTruth("payroll", "pay_date", "Pay date", "event", False, default=True),
        ],
        measures=[
            MeasureTruth("headcount_monthly", "headcount", "sum", "semi_additive", "integer", "Headcount",
                        time_aggregation="last"),
            MeasureTruth("headcount_monthly", "fte", "sum", "semi_additive", "number", "FTE", time_aggregation="last"),
            MeasureTruth("headcount_monthly", "monthly_salary", "sum", "additive", "currency", "Monthly salary"),
            MeasureTruth("payroll", "gross_pay", "sum", "additive", "currency", "Gross pay"),
            MeasureTruth("payroll", "tax_withheld", "sum", "additive", "currency", "Tax withheld"),
            MeasureTruth("payroll", "net_pay", "sum", "additive", "currency", "Net pay"),
            MeasureTruth("payroll", "overtime_hours", "sum", "additive", "integer", "Overtime hours"),
            MeasureTruth("payroll", None, "count", "additive", "count", "Payroll lines"),
        ],
        labels={"departments": "department_name", "locations": "location_name", "employees": "employee_name"},
        codes={"employees": "employee_number"},
        statuses={"employees.status": {"values": {"A": "Active", "T": "Terminated"}, "cancelled": []}},
        quality=[
            {"object": "employees.termination_date", "kind": "placeholder_dates"},
        ],
        not_measures=["headcount_monthly.employee_id", "headcount_monthly.department_id",
                      "headcount_monthly.location_id", "payroll.payroll_line_id", "payroll.employee_id",
                      "payroll.department_id"],
        sensitive={"employees.employee_name": "pii", "employees.email": "pii", "employees.birth_date": "pii",
                   "headcount_monthly.monthly_salary": "confidential", "payroll.gross_pay": "confidential",
                   "payroll.net_pay": "confidential"},
    )
    return Domain("hr", tables, truth, TODAY,
                  "A manufacturer's employees, month-end headcount and payroll.",
                  abbreviations={"termination": "TRM", "overtime": "OT", "withheld": "WHLD"})
