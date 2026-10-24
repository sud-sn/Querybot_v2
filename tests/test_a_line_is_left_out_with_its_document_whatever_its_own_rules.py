"""A line is left out with its document, whatever rules the lines have of their own.

On an ERP's reporting database, the admin saved the rows each table leaves out: the order header leaves out deleted
and cancelled orders and the test account, the order lines leave out cancelled lines. Net sales, counted on the
lines, then left out the cancelled lines alone: the lines of cancelled, deleted and test orders came back, because a
line table took its order's rules only while it had none of its own. Before the admin saved anything, the lines did
take the order's rules, so saving the client's own definitions made the figures worse.

Now both apply: a line is out when it is cancelled, and when its order is. The figures that follow members by their
lines (customers with none in a year, a list of those who booked) leave the same rows out, and a question asking for
the left-out rows too gets back those of each table whose admin lets readers ask, and only those. Synthetic data only: the home services
warehouse, its bookings leaving cancelled, duplicate and voided ones out, its lines given a rule of their own.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.model.schema import ColumnFilter
from core2.plan.values import build_index
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

LINES = "memory.main.booking_lines"
KEPT = "k.status_id NOT IN (4, 6, 7) AND l.service_type <> 'Painting'"
YEAR = {"time": {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}}


@pytest.fixture(scope="module")
def home():
    built, model = learn(domains.build("home_services"), "descriptive")
    assert model.tables["memory.main.bookings"].default_filters, "the bookings leave some of theirs out"
    model.tables[LINES].default_filters = [ColumnFilter(column=f"{LINES}.service_type", op="not_in",
                                                        values=["Painting"])]
    warehouse = DuckDBWarehouse(built.con)

    def fetch(slug: str) -> list[object]:
        column = model.columns[model.attributes[slug].column]
        return [r[0] for r in warehouse.query(
            f'SELECT DISTINCT "{column.name}" FROM "{model.tables[column.table].name}"').rows]

    return built, model, warehouse, build_index(model, fetch)


def _ask(home, question: str, plan: dict) -> dict:
    built, model, warehouse, index = home

    def complete(system: str, tail: str) -> str:
        return json.dumps({"kind": "query", **plan})

    services = Services(model=model, warehouse=warehouse, complete=complete, index=index, today=dt.date(2026, 6, 15))
    return answer_question(question, services, Session())


def test_a_lines_total_leaves_out_its_own_rows_and_its_documents(home):
    got = _ask(home, "What was our service revenue in 2025?", {"measures": ["line_amount"], **YEAR})
    want = home[0].con.execute(f"""SELECT SUM(l.line_amount) FROM booking_lines l
        LEFT JOIN bookings k USING (booking_number) JOIN calendar d ON l.booked_date_key = d.date_key
        WHERE ({KEPT} OR k.status_id IS NULL AND l.service_type <> 'Painting') AND d.year = 2025""").fetchone()[0]
    assert float(got["kpi"]["value"]) == pytest.approx(float(want)), got["answer"]["headline"]
    said = json.dumps(got)
    assert "Closed - Cancelled" in said and "Painting" in said, "the answer says both rules it applied"


J = "FROM booking_lines l JOIN bookings k USING (booking_number) JOIN calendar d ON l.booked_date_key = d.date_key"


def test_members_with_no_lines_count_the_lines_their_documents_keep(home):
    # A shape on the lines (customers who had none in 2025) took the lines' own rules alone: 96 where 120 had no
    # booking line that their bookings keep.
    got = _ask(home, "Which customers had no booking lines in 2025?",
               {"intent": "list", "absent": {"of": "customer", "measure": "number_of_booking_lines"}, **YEAR})
    want = home[0].con.execute(f"""SELECT COUNT(*) FROM customers WHERE customer_id NOT IN (
        SELECT k.customer_id {J} WHERE {KEPT} AND d.year = 2025 AND k.customer_id IS NOT NULL)""").fetchone()[0]
    assert len(got["export_rows"]) == want, got["answer"]["headline"]


def test_a_list_of_members_by_their_lines_leaves_out_the_same_rows(home):
    got = _ask(home, "List the customers who booked in 2025",
               {"intent": "list", "group_by": ["customer.name"],
                "time": {**YEAR["time"], "date": "booked_date"}})
    want = home[0].con.execute(f"SELECT COUNT(DISTINCT k.customer_id) {J} WHERE {KEPT} AND d.year = 2025").fetchone()[0]
    assert len(got["export_rows"]) == want, got["answer"]["headline"]


def test_rows_asked_for_too_come_back_from_each_table_that_lets_readers_ask(home):
    # "Including" the lines' own left-out rows skipped every rule after them: the cancelled bookings' lines came back
    # too, where the bookings' admin lets no reader ask for them.
    bookings = home[1].tables["memory.main.bookings"]
    bookings.readers_may_include = False
    try:
        got = _ask(home, "service revenue in 2025 including painting",
                   {"measures": ["line_amount"], "include_left_out": True, **YEAR})
    finally:
        bookings.readers_may_include = True
    want = home[0].con.execute("""SELECT SUM(l.line_amount) FROM booking_lines l
        LEFT JOIN bookings k USING (booking_number) JOIN calendar d ON l.booked_date_key = d.date_key
        WHERE (k.status_id NOT IN (4, 6, 7) OR k.status_id IS NULL) AND d.year = 2025""").fetchone()[0]
    assert float(got["kpi"]["value"]) == pytest.approx(float(want)), got["answer"]["headline"]
