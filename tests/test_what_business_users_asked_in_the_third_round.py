"""What the third round of business questions found, each checked against hand-written SQL.

- "Avg # of direct reports per manager" and "which 5 managers have the most direct reports?" were planned with a job
  level called "manager" and refused ("there is no job level called manager"): "manager" is the link from an
  employee to another employee, and the answer follows it.
- "And contractors?" after the part-time employees counted part-time and contractors together (101): another
  member of the field the answer was narrowed by is asked for in its place ("both", "too" ask for the two).
- "Which techs had fewer than 3 visits in December 2025?" was answered "1 rows": the members meeting a condition
  were counted, where the question asks which they are.
- "How many prescriptions did we fill in 2025?" counted the prescriptions written in 2025: the verb names the
  rows that point at them (fills), and those counted are the ones filled then, by the fill date.
- "Who are our top 10 customers by total spend?" kept the inactive customers of the answer before: what follows
  "by" is the figure ranked by, so the question names its own figure and is new.
- "What share of the 2025 bookings ended up cancelled?" was refused: the status name is kept in a lookup table.
- "Whats the gross margin % on that" was the price charged over the ingredient cost (239.8%, a markup): a margin is
  the sales less the cost, over the sales.

The invented HR firm, home-services firm, compounding pharmacy and retail shop of the test domains.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.followup import read_turn
from core2.plan.ir import Plan
from core2.plan.values import MemberIndex, build_index
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
_built: dict[str, tuple] = {}


def _domain(name: str):
    if name not in _built:
        _built[name] = learn(domains.build(name), "descriptive")
    return _built[name]


def _chat(name: str, *replies: dict, index: bool = True):
    """Ask in one conversation, the AI answering each turn with the next plan."""
    built, model = _domain(name)
    plans = list(replies)

    def fetch(slug):
        column = model.columns[model.attributes[slug].column]
        table = model.tables[column.table]
        return [r[0] for r in built.con.execute(f'SELECT DISTINCT "{column.name}" FROM "{table.name}"').fetchall()]

    def complete(stable, tail):
        return json.dumps({"kind": "query", **plans.pop(0)})

    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=complete,
                        index=build_index(model, fetch) if index else MemberIndex(), today=TODAY)
    session = Session()
    return lambda q: answer_question(q, services, session)


def _one(con, sql: str):
    return con.execute(sql).fetchone()[0]


ACTIVE = {"field": "employee.status", "op": "eq", "values": ["A"]}
MANAGER = {"field": "employee.job_level", "op": "eq", "values": ["manager"]}
REPORTS = ("SELECT m.employee_name, COUNT(DISTINCT e.employee_id) FROM employees e JOIN employees m "
           "ON e.manager_id = m.employee_id WHERE e.status = 'A' GROUP BY 1")


def test_direct_reports_per_manager_follow_the_manager_link():
    built, _ = _domain("hr")
    ask = _chat("hr", {"intent": "value", "measures": ["number_of_employees"], "filters": [ACTIVE, MANAGER],
                       "per": {"of": "employee"}})
    payload = ask("avg # of direct reports per manager")
    assert payload["type"] != "core2_refused", payload.get("text")
    want = _one(built.con, f"SELECT AVG(n) FROM ({REPORTS.replace('m.employee_name', 'm.employee_id')}) t(k, n)")
    got = next(v for k, v in payload["data"]["rows"][0].items() if k.startswith("avg"))
    assert got == pytest.approx(float(want), rel=1e-6)
    assert "manager" not in json.dumps(payload["plan"].get("filters", [])).lower()


def test_the_managers_with_most_reports_are_ranked_by_them():
    built, _ = _domain("hr")
    ask = _chat("hr", {"intent": "rank", "measures": ["number_of_employees"], "group_by": ["employee"],
                       "via": {"employee": "Manager"}, "filters": [ACTIVE, MANAGER],
                       "sort": [{"by": "number_of_employees"}], "limit": 5})
    payload = ask("Which 5 managers have the most active direct reports?")
    rows = payload["data"]["rows"]
    want = sorted(built.con.execute(REPORTS).fetchall(), key=lambda r: -r[1])[:5]
    assert [r["number_of_employees"] for r in rows] == [n for _, n in want]
    assert {r["employee_name"] for r in rows} <= {name for name, _ in built.con.execute(REPORTS).fetchall()}


def test_a_level_no_link_is_called_is_still_refused():
    ask = _chat("hr", {"intent": "breakdown", "measures": ["number_of_employees"], "group_by": ["department"],
                       "filters": [{"field": "employee.job_level", "op": "eq", "values": ["director"]}]})
    payload = ask("how many directors in each department")
    said = json.dumps(payload)
    assert not (payload.get("data") or {}).get("rows") and 'there is no job level called \\"director\\"' in said


def test_and_another_member_is_asked_in_its_place():
    built, _ = _domain("hr")
    part_time = {"field": "employee.employment_type", "op": "eq", "values": ["PT"]}
    ask = _chat("hr",
                {"intent": "value", "measures": ["number_of_employees"], "filters": [part_time, ACTIVE]},
                {"intent": "value", "measures": ["number_of_employees"], "follow_up": "refine", "filters": [
                    {"field": "employee.employment_type", "op": "in", "values": ["PT", "CT"]}, ACTIVE]},
                {"intent": "value", "measures": ["number_of_employees"], "follow_up": "refine", "filters": [
                    {"field": "employee.employment_type", "op": "in", "values": ["CT", "FT"]}, ACTIVE]})
    ask("How many part-time employees do we have right now?")
    contractors = ask("and contractors?")
    want = _one(built.con, "SELECT COUNT(*) FROM employees WHERE employment_type = 'CT' AND status = 'A'")
    assert contractors["data"]["rows"] == [{"number_of_employees": want}]
    both = ask("and full-time too?")                      # "too": the two together, as the AI planned
    want = _one(built.con, "SELECT COUNT(*) FROM employees WHERE employment_type IN ('CT', 'FT') AND status = 'A'")
    assert both["data"]["rows"] == [{"number_of_employees": want}]


DECEMBER = {"window": {"kind": "between", "start": "2025-12-01", "end": "2025-12-31"}}


def test_which_members_meet_a_condition_are_listed_by_name():
    built, _ = _domain("home_services")
    counted = {"intent": "count", "measures": ["number_of_visits"], "time": DECEMBER,
               "per": {"of": "technician", "summary": "count", "op": "lt", "values": [3]}}
    payload = _chat("home_services", counted)("which techs had fewer than 3 visits in December 2025?")
    want = built.con.execute(
        "SELECT t.technician_name, COUNT(*) FROM visits v JOIN calendar c ON v.visit_date_key = c.date_key "
        "JOIN technicians t ON v.technician_id = t.technician_id WHERE c.full_date BETWEEN '2025-12-01' AND "
        "'2025-12-31' GROUP BY 1 HAVING COUNT(*) < 3").fetchall()
    assert want, "the test data has such technicians"
    assert sorted((r["technician_name"], r["number_of_visits"]) for r in payload["data"]["rows"]) == sorted(want)
    how_many = _chat("home_services", counted)("how many techs had fewer than 3 visits in December 2025?")
    assert list(how_many["data"]["rows"][0].values())[0] == len(want)        # "how many" still counts them


Y2025 = {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}


def test_prescriptions_filled_in_a_year_are_counted_by_their_fills():
    built, _ = _domain("compounding_pharmacy")
    payload = _chat("compounding_pharmacy", {"intent": "value", "measures": ["number_of_prescriptions"],
                                             "time": Y2025})("How many prescriptions did we fill in 2025?")
    want = _one(built.con, "SELECT COUNT(DISTINCT rx_number) FROM fills WHERE fill_date BETWEEN '2025-01-01' AND "
                           "'2025-12-31'")
    written = _one(built.con, "SELECT COUNT(*) FROM prescriptions WHERE written_date BETWEEN '2025-01-01' AND "
                              "'2025-12-31'")
    assert want != written
    assert list(payload["data"]["rows"][0].values())[0] == want
    plain = _chat("compounding_pharmacy", {"intent": "value", "measures": ["number_of_prescriptions"],
                                           "time": Y2025})("How many prescriptions were written in 2025?")
    assert plain["data"]["rows"] == [{"number_of_prescriptions": written}]


def test_top_customers_by_a_figure_of_their_own_is_a_new_question():
    _, model = _domain("retail")
    before = Plan.model_validate({"kind": "query", "intent": "value", "measures": ["net_amount"],
                                  "group_by": ["store"], "time": Y2025, "filters": [
                                      {"field": "region.name", "op": "eq", "values": ["North"]}]})
    assert read_turn("Who are our top 10 stores by total takings?", previous=before, model=model).kind == "new"
    assert read_turn("Show the net amount by region", previous=before, model=model).kind == "refine"
    assert read_turn("and by region?", previous=before, model=model).kind == "refine"


def test_a_margin_percent_is_what_is_left_of_the_sales_over_the_sales():
    built, _ = _domain("compounding_pharmacy")
    markup = {"name": "Gross margin percentage", "op": "ratio", "measures": ["price_charged", "ingredient_cost"],
              "scale": 100}
    payload = _chat("compounding_pharmacy", {"intent": "value", "derived": [markup], "time": Y2025})(
        "whats the gross margin % in 2025")
    price, cost = built.con.execute("SELECT SUM(price_charged), SUM(ingredient_cost) FROM fills WHERE fill_date "
                                    "BETWEEN '2025-01-01' AND '2025-12-31'").fetchone()
    row = payload["data"]["rows"][0]
    got = next(v for k, v in row.items() if "percentage" in k)
    assert got == pytest.approx((float(price) - float(cost)) / float(price) * 100, rel=1e-6)
    assert got < 100


def test_how_many_managers_counts_the_managers():
    built, _ = _domain("hr")
    ask = _chat("hr", {"intent": "value", "measures": ["number_of_employees"], "group_by": ["employee"],
                       "via": {"employee": "Manager"}, "filters": [ACTIVE]})
    payload = ask("how many managers do we have")
    want = _one(built.con, "SELECT COUNT(DISTINCT manager_id) FROM employees WHERE status = 'A' "
                           "AND manager_id IS NOT NULL")
    assert list(payload["data"]["rows"][0].values())[0] == want


def test_a_share_of_rows_by_a_status_kept_in_a_lookup_table():
    """ "What share of the 2025 bookings ended up cancelled?" was refused: the status name is on the statuses
    table, not the bookings'. Each booking has one status, so its condition is a condition on the booking."""
    built, _ = _domain("home_services")
    rate = {"name": "Cancelled share", "measure": "number_of_bookings",
            "where": [{"field": "booking_status.name", "op": "eq", "values": ["Closed - Cancelled"]}]}
    payload = _chat("home_services", {"intent": "value", "measures": ["number_of_bookings"], "rates": [rate],
                                      "include_left_out": True, "time": Y2025})(
        "What share of the 2025 bookings ended up cancelled?")
    want = _one(built.con, "SELECT AVG(CASE WHEN s.status_name = 'Closed - Cancelled' THEN 100.0 ELSE 0 END) "
                           "FROM bookings b JOIN booking_statuses s ON s.status_id = b.status_id "
                           "JOIN calendar c ON c.date_key = b.booked_date_key "
                           "WHERE c.full_date BETWEEN '2025-01-01' AND '2025-12-31'")
    assert payload["data"]["rows"][0]["cancelled_share"] == pytest.approx(float(want), rel=1e-6)
