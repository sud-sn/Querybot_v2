"""What the Phase 5 condition books found, asked of six businesses with gpt-4o and checked against SQL written by hand.

Each test is one finding, put right in code and asked again here with the AI's plan scripted, on invented data:
"YTD against the same time last year" right after an MTD answer kept the month; a period wholly after the data's
last day was answered as if nobody had done anything; "how many customers had no orders in the last 90 days" listed
them instead of counting them; and the others below, each with the book step that found it.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.values import MemberIndex, build_index
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 10, 15)          # four months after the invented retail data's last day (14 Jun 2026)
LINES = """SELECT o.*, d.full_date AS day, s.store_name FROM order_lines o
           JOIN calendar d ON o.order_date_key = d.date_key JOIN stores s ON o.store_id = s.store_id"""
NET = {"measures": ["net_amount"]}


@pytest.fixture(scope="module")
def retail():
    built, model = learn(domains.build("retail"), "descriptive")
    warehouse = DuckDBWarehouse(built.con)

    def fetch(slug: str) -> list[object]:
        column = model.columns[model.attributes[slug].column]
        table = model.tables[column.table]
        return [r[0] for r in warehouse.query(f'SELECT DISTINCT "{column.name}" FROM "{table.name}"').rows]

    return built, model, warehouse, build_index(model, fetch)


class Chat:
    """One conversation, the AI's plans scripted."""

    def __init__(self, retail, today: dt.date = TODAY):
        built, model, warehouse, index = retail
        self.plans: list[dict] = []

        def complete(system: str, tail: str) -> str:
            return json.dumps({"kind": "query", **self.plans.pop(0)})

        self.services = Services(model=model, warehouse=warehouse, complete=complete, index=index, today=today)
        self.session = Session()

    def ask(self, question: str, plan: dict | None = None) -> dict:
        if plan is not None:
            self.plans.append(plan)
        return answer_question(question, self.services, self.session)


def _sum(retail, where: str) -> float:
    return float(retail[0].con.execute(f"SELECT SUM(net_amount) FROM ({LINES}) l WHERE {where}").fetchone()[0])


# ── periods ─────────────────────────────────────────────────────────────────


def test_ytd_against_the_same_time_last_year_after_an_mtd_answer(retail):
    # Purchasing PU5-01.5, home services HP-01.5, finance FN5-01.3: "the same time last year" read as pointing at
    # the period above ("that same month"), so the year to date became the month to date.
    chat = Chat(retail)
    chat.ask("net sales mtd", {**NET, "time": {"window": {"kind": "to_date", "unit": "month"}}})
    got = chat.ask("How does our YTD net sales compare with the same time last year?",
                   {**NET, "follow_up": "refine", "time": {"window": {"kind": "to_date", "unit": "year"},
                                                           "compare": {"kind": "same_period_last_year"}}})
    assert "from 1 Jan 2026 to 14 Jun 2026" in got["answer"]["headline"]
    assert float(got["export_rows"][0]["net_amount"]) == pytest.approx(
        _sum(retail, "l.day >= DATE '2026-01-01' AND l.day < DATE '2026-06-15'"))


# ── members by what they did: the activity they did it on ─────────────────


FIRST = """SELECT customer_id, MIN(day) AS first FROM ({lines}) l
           WHERE customer_id IN (SELECT customer_id FROM customers) GROUP BY 1"""


def _new_in(retail, start: str, end: str) -> int:
    return int(retail[0].con.execute(f"""SELECT COUNT(*) FROM ({FIRST.format(lines=LINES)})
        WHERE first >= DATE '{start}' AND first < DATE '{end}'""").fetchone()[0])


@pytest.mark.parametrize("measure", ["number_of_customers", "number_of_returns"])
def test_new_customers_follow_their_orders_unless_the_question_names_another_activity(retail, measure):
    # Bank BP-06, home services HP-06: "how many new customers did we get" planned on the customer list's own
    # sign-up date (or account openings): 285 where 86 had their first booking, and a churn rate of 100%. With no
    # activity named, members are followed by the main one (orders, transactions, bookings).
    chat = Chat(retail, today=dt.date(2026, 6, 15))
    got = chat.ask("How many new customers did we get in Q2 2023?", {"lifecycle": {
        "of": "customer", "measure": measure, "status": "new"}, "time": {"window": {
            "kind": "between", "start": "2023-04-01", "end": "2023-06-30"}}})
    assert float(got["kpi"]["value"]) == _new_in(retail, "2023-04-01", "2023-07-01")


def test_an_activity_the_question_names_is_followed(retail):
    chat = Chat(retail, today=dt.date(2026, 6, 15))
    got = chat.ask("How many customers returned something for the first time in 2024?", {"lifecycle": {
        "of": "customer", "measure": "number_of_orders", "status": "new"}, "time": {"window": {
            "kind": "between", "start": "2024-01-01", "end": "2024-12-31"}}})
    want = retail[0].con.execute("""SELECT COUNT(*) FROM (SELECT r.customer_id, MIN(d.full_date) AS first
        FROM returns r JOIN calendar d ON r.return_date_key = d.date_key
        WHERE r.customer_id IN (SELECT customer_id FROM customers) GROUP BY 1)
        WHERE first >= DATE '2024-01-01' AND first < DATE '2025-01-01'""").fetchone()[0]
    assert float(got["kpi"]["value"]) == want


# ── who did nothing ─────────────────────────────────────────────────────────


def test_how_many_customers_had_no_orders_is_a_number(retail):
    # Retail RP-05.5, bank BP-07.1, home services HP-07.3: "how many ... had no orders in the last 90 days" listed
    # them; the number leads, the list stays below it.
    chat = Chat(retail)
    got = chat.ask("How many customers have had no orders in the last 90 days?", {"absent": {
        "of": "customer", "measure": "number_of_orders", "active_before": True},
        "time": {"window": {"kind": "rolling", "unit": "day", "n": 90}}})
    want = retail[0].con.execute(f"""SELECT COUNT(*) FROM customers c
        WHERE c.customer_id IN (SELECT customer_id FROM ({LINES}) WHERE day < DATE '2026-03-17')
          AND c.customer_id NOT IN (SELECT customer_id FROM ({LINES}) WHERE day >= DATE '2026-03-17'
                                    AND day < DATE '2026-06-15')""").fetchone()[0]
    assert got["kpi"]["value"] == want and len(got["export_rows"]) == want
    assert got["answer"]["headline"].startswith(f"{want} customers had no orders")


def test_a_period_after_the_data_ends_is_said_never_answered(retail):
    # Retail RP-05.7, purchasing PU5-06.6: "which of them are going to order again next month?" was answered
    # "240 customers had no orders in November 2026": every customer, for a month the data does not reach yet.
    chat = Chat(retail)
    got = chat.ask("Which customers will not order next month?", {"absent": {
        "of": "customer", "measure": "number_of_orders"}, "time": {"window": {
            "kind": "between", "start": "2026-11-01", "end": "2026-11-30"}}})
    assert not got.get("export_rows")
    assert "14 Jun 2026" in got["answer"]["headline"] and "November 2026" in got["answer"]["headline"]


# ── what members did, in the reader's words ─────────────────────────────────

MEMBERS = f"SELECT customer_id, day FROM ({LINES}) WHERE customer_id IN (SELECT customer_id FROM customers)"
COUNT_PLAN = {"measures": ["number_of_customers"]}


def _count(retail, sql: str) -> int:
    return int(retail[0].con.execute(f"SELECT COUNT(*) FROM ({sql})").fetchone()[0])


def test_on_two_or_more_days_is_repeat(retail):
    # Purchasing PU5-07.5: "how many suppliers did we order from on 2 or more days this month so far" was planned
    # as each supplier's number of suppliers (none).
    chat = Chat(retail, today=dt.date(2026, 6, 15))
    got = chat.ask("how many customers did we sell to on 2 or more days in Q2 2023",
                   {**COUNT_PLAN, "time": {"window": {"kind": "between", "start": "2023-04-01", "end": "2023-06-30"}}})
    assert float(got["kpi"]["value"]) == _count(retail, f"""SELECT customer_id FROM ({MEMBERS})
        WHERE day >= DATE '2023-04-01' AND day < DATE '2023-07-01' GROUP BY 1 HAVING COUNT(DISTINCT day) >= 2""")


def test_every_single_month_is_loyal(retail):
    # Bank BP-07.2: "how many customers transacted every single month over the last 6 months" was a dead end.
    chat = Chat(retail, today=dt.date(2026, 6, 15))
    got = chat.ask("how many customers ordered every single month in 2025",
                   {**COUNT_PLAN, "time": {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}})
    assert float(got["kpi"]["value"]) == _count(retail, f"""SELECT customer_id FROM ({MEMBERS})
        WHERE day >= DATE '2025-01-01' AND day < DATE '2026-01-01' GROUP BY 1
        HAVING COUNT(DISTINCT date_trunc('month', day)) = 12""")


def test_customers_we_lost_in_french_are_lapsed(retail):
    # Retail RP-06.2: "combien de clients avons-nous perdus sur les 3 derniers mois" listed who had no orders.
    chat = Chat(retail, today=dt.date(2026, 6, 15))
    got = chat.ask("Combien de clients avons-nous perdus sur les 3 derniers mois ?",
                   {"absent": {"of": "customer", "measure": "number_of_orders"},
                    "time": {"window": {"kind": "last", "unit": "month", "n": 3}}})
    assert float(got["kpi"]["value"]) == _count(retail, f"""SELECT customer_id FROM ({MEMBERS})
        WHERE day >= DATE '2025-12-01' AND day < DATE '2026-03-01'
        EXCEPT SELECT customer_id FROM ({MEMBERS}) WHERE day >= DATE '2026-03-01' AND day < DATE '2026-06-01'""")


# ── standing conditions ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def unread(retail):
    """The retail data with the region's names kept from the index, as a field whose values are not allowed (a
    buyer read as a person's name, a supplier's country) is on a real workspace."""
    built, model, warehouse, _ = retail
    model = model.model_copy(deep=True)
    model.columns[model.attributes["region.name"].column].values_allowed = False

    def fetch(slug: str) -> list[object]:
        column = model.columns[model.attributes[slug].column]
        return [r[0] for r in warehouse.query(
            f'SELECT DISTINCT "{column.name}" FROM "{model.tables[column.table].name}"').rows]

    index = build_index(model, fetch)
    assert "region.name" not in index.attributes
    return built, model, warehouse, index


def test_a_name_on_a_field_the_index_does_not_read(unread):
    # Purchasing PU5-03.1 and PU5-05.1: "only the "MRO Desk" buying team" kept the category MRO, and "seulement les
    # fournisseurs du pays "Germany"" found nothing called Germany: neither field's names are indexed.
    chat = Chat(unread)
    said = chat.ask('From now on only the "West" region')
    assert said["answer"]["headline"].startswith("From now on in this conversation: Region")
    got = chat.ask("net sales in 2025", {**NET, "time": {"window": {"kind": "between", "start": "2025-01-01",
                                                                     "end": "2025-12-31"}}})
    want = unread[0].con.execute(f"""SELECT SUM(l.net_amount) FROM ({LINES}) l JOIN stores s USING (store_id)
        JOIN regions r ON s.region_id = r.region_id
        WHERE r.region_name = 'West' AND l.day >= DATE '2025-01-01' AND l.day < DATE '2026-01-01'""").fetchone()[0]
    assert float(got["kpi"]["value"]) == pytest.approx(float(want))
    chat2 = Chat(unread)
    french = chat2.ask('Désormais, seulement la région "West"')
    assert french["answer"]["headline"].startswith("Désormais dans cette conversation")
    gone = chat2.ask('Supprime la condition sur la région "West"')
    assert gone["answer"]["headline"].startswith("Retiré :")


def test_removing_a_condition_that_is_not_there_is_said(retail):
    # Purchasing PU5-05.5, home services HP-04.5: "ok remove the Germany condition" was answered as a question (or
    # refused as a change to the data) when no such condition was kept.
    chat = Chat(retail)
    got = chat.ask('ok remove the "Germany" condition')
    assert got["answer"]["headline"].startswith("No condition by that name is in force")


# ── a ledger: no members followed, budgets into next year ──────────────────


@pytest.fixture(scope="module")
def finance():
    built, model = learn(domains.build("finance"), "descriptive")
    warehouse = DuckDBWarehouse(built.con)

    def fetch(slug: str) -> list[object]:
        column = model.columns[model.attributes[slug].column]
        return [r[0] for r in warehouse.query(
            f'SELECT DISTINCT "{column.name}" FROM "{model.tables[column.table].name}"').rows]

    return built, model, warehouse, build_index(model, fetch)


def test_a_ledgers_standing_period_counts_back_from_its_postings(finance):
    # Finance FN5-04.1: "from now on use the last 3 months" said "1 Jul 2026 - 30 Sep 2026", counted from today
    # because the budgets run into next year; every answer counted back from the last posting (12 Jun 2026).
    chat = Chat(finance)
    said = chat.ask("From now on use the last 3 months")
    assert "the last 3 months (1 Mar 2026 – 31 May 2026)" in said["answer"]["headline"]


def test_leave_out_a_name_four_words_after_the_verb(finance):
    # Finance FN5-06.1: "exclus le centre de coûts "O'Neil Studio"" kept only O'Neil Studio.
    chat = Chat(finance)
    said = chat.ask("Pour la suite de cette conversation, exclus le centre de coûts \"O'Neil Studio\"")
    assert said["standing"]["kept"][0]["words"].endswith("is not O'Neil Studio")


def test_last_quarter_counts_back_from_where_the_data_ends(retail):
    # Finance FN5-03.1, bank BP-06.5: "last quarter" planned as the quarter before today's (Q2 2026 in October)
    # where the data ended in June: the quarter before the one the data ends in.
    chat = Chat(retail)
    got = chat.ask("net sales last quarter", {**NET, "time": {"window": {
        "kind": "between", "start": "2026-07-01", "end": "2026-09-30"}}})
    assert "Q1 2026" in got["answer"]["headline"]


# ── follow-ups ──────────────────────────────────────────────────────────────


def test_won_back_customers_counted_as_a_figure_asks_no_date(retail):
    # Retail RP-05.4: "how many custmers did we win back last month" planned as the number of customers for
    # won-back customers asked which of a customer's dates to count by.
    chat = Chat(retail, today=dt.date(2026, 6, 15))
    got = chat.ask("how many custmers did we win back in May 2026", {
        "measures": ["number_of_customers"], "segment": {"of": "customer", "measure": "number_of_orders",
                                                         "status": "won_back"},
        "time": {"window": {"kind": "between", "start": "2026-05-01", "end": "2026-05-31"}}})
    assert got.get("type") == "assistant_response" and "kpi" in got and got["kpi"] is not None
    want = _count(retail, f"""SELECT customer_id FROM ({MEMBERS}) WHERE day >= DATE '2026-05-01'
        AND day < DATE '2026-06-01' AND customer_id IN (SELECT customer_id FROM ({MEMBERS}) WHERE day < DATE '2026-05-01')
        EXCEPT SELECT customer_id FROM ({MEMBERS}) WHERE day >= DATE '2026-04-01' AND day < DATE '2026-05-01'""")
    assert float(got["kpi"]["value"]) == want


def test_what_we_spent_with_them_after_counting_them(retail):
    # Purchasing PU5-08.4: "and what did we spend with them" after the suppliers won back gave their number again.
    chat = Chat(retail, today=dt.date(2026, 6, 15))
    window = {"kind": "between", "start": "2023-04-01", "end": "2023-06-30"}
    chat.ask("how many returning customers in Q2 2023", {"lifecycle": {
        "of": "customer", "measure": "number_of_orders", "status": "returning"}, "time": {"window": window}})
    got = chat.ask("and what did they spend", {**NET, "follow_up": "refine", "time": {"window": window}})
    want = retail[0].con.execute(f"""SELECT SUM(net_amount) FROM ({LINES}) l
        WHERE day >= DATE '2023-04-01' AND day < DATE '2023-07-01' AND customer_id IN (
            SELECT customer_id FROM ({MEMBERS}) GROUP BY 1 HAVING MIN(day) < DATE '2023-04-01')""").fetchone()[0]
    assert float(got["kpi"]["value"]) == pytest.approx(float(want))


def test_and_total_by_month_drops_the_split(retail):
    # Purchasing PU5-04.3: "and total spend by month?" after spend by category kept the categories.
    chat = Chat(retail, today=dt.date(2026, 6, 15))
    window = {"kind": "between", "start": "2026-01-01", "end": "2026-03-31"}
    chat.ask("net sales by category in Q1 2026", {**NET, "group_by": ["category.name"], "time": {"window": window}})
    got = chat.ask("and total net sales by month?", {**NET, "follow_up": "refine", "group_by": ["category.name"],
                                                     "time": {"grain": "month", "window": window}})
    assert len(got["export_rows"]) == 3 and "category" not in json.dumps(got["export_rows"][0]).lower()


def test_how_many_of_those_did_we_win_back(retail):
    # Pharmacy RXP-06.3: "how many of those did we win back" names no members; the plan's are meant.
    chat = Chat(retail, today=dt.date(2026, 6, 15))
    got = chat.ask("how many of those did we win back in May 2026?", {"absent": {
        "of": "customer", "measure": "number_of_orders"}, "time": {"window": {
            "kind": "between", "start": "2026-05-01", "end": "2026-05-31"}}})
    assert got["answer"]["headline"].startswith("Won back customers in May 2026")


def test_lost_each_month_in_french_is_a_series(retail):
    # Pharmacy RXP-07.5: "combien de patients avons-nous perdus chaque mois depuis le début de l'année" gave one
    # figure for the year.
    chat = Chat(retail, today=dt.date(2026, 6, 15))
    got = chat.ask("Combien de clients avons-nous perdus chaque mois depuis le début de l'année ?", {"lifecycle": {
        "of": "customer", "measure": "number_of_orders", "status": "lapsed"}, "time": {
            "grain": "month", "window": {"kind": "to_date", "unit": "year"}}})
    assert len(got["export_rows"]) >= 5


def test_by_week_keeps_the_week_holding_the_first_day(retail):
    # Retail RP-07.4: "new customers by week in January 2023" left out the week of 26 December 2022, which holds
    # 1 January (the data's first day, when most customers first ordered).
    chat = Chat(retail, today=dt.date(2026, 6, 15))
    got = chat.ask("new customers by week in january 2023", {"lifecycle": {
        "of": "customer", "measure": "number_of_orders", "status": "new"}, "time": {"grain": "week", "window": {
            "kind": "between", "start": "2023-01-01", "end": "2023-01-31"}}})
    want = dict(retail[0].con.execute(f"""SELECT date_trunc('week', first)::DATE, COUNT(*)
        FROM ({FIRST.format(lines=LINES)}) WHERE first < DATE '2023-02-01' GROUP BY 1""").fetchall())
    value = next(k for k in got["export_rows"][0] if k != "period")
    shown = {str(r["period"])[:10]: float(r[value]) for r in got["export_rows"] if float(r[value] or 0)}
    assert shown == {str(k)[:10]: float(v) for k, v in want.items()}


def test_the_member_above_pointed_at_by_a_word_of_its_name(retail):
    # Finance FN5-03.4: "What did the lab spend in March 2026?" after "R&D Lab #2" dropped the cost centre.
    chat = Chat(retail, today=dt.date(2026, 6, 15))
    chat.ask('net sales for "Riverside Store" in 2025', {**NET, "filters": [
        {"field": "store.name", "op": "eq", "values": ["Riverside Store"]}], "time": {"window": {
            "kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}})
   

@pytest.fixture(scope="module")
def stores_unread(retail):
    """The retail data with the stores' names kept from the index, as a buying team's are on a purchasing
    workspace."""
    built, model, warehouse, _ = retail
    model = model.model_copy(deep=True)
    model.columns[model.attributes["store.name"].column].values_allowed = False

    def fetch(slug: str) -> list[object]:
        column = model.columns[model.attributes[slug].column]
        return [r[0] for r in warehouse.query(
            f'SELECT DISTINCT "{column.name}" FROM "{model.tables[column.table].name}"').rows]

    index = build_index(model, fetch)
    assert "store.name" not in index.attributes
    return built, model, warehouse, index


def test_same_for_a_name_no_read_field_holds(stores_unread):
    # Purchasing PU5-03.5: 'same for "Packaging Desk"' under the buying team "MRO Desk" was read as the category
    # Packaging, a part of the name, with MRO Desk still applied: the buyers' names are not read.
    chat = Chat(stores_unread)
    chat.ask('From now on only the store "Downtown Store"')
    year = {"time": {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}}
    chat.ask("net sales in 2025", {**NET, **year})
    got = chat.ask('same for "Westfield Store"',
                   {**NET, **year, "filters": [{"field": "region.name", "op": "eq", "values": ["West"]}]})
    want = _sum(stores_unread, "l.store_name = 'Westfield Store' AND l.day >= DATE '2025-01-01' "
                               "AND l.day < DATE '2026-01-01'")
    assert float(got["kpi"]["value"]) == pytest.approx(want), got["answer"]["headline"]
    assert "West" not in got["answer"]["headline"].replace("Westfield", ""), got["answer"]["headline"]


def test_a_purchase_order_line_is_what_we_ordered():
    # Purchasing PU5-07.5: "how many suppliers did we order from on 2 or more days" followed their receipts, the
    # larger table: "order" is a word of "Purchase order line", not its last.
    built, model = learn(domains.build("purchasing"), "descriptive")
    from core2.plan.ir import Lifecycle, Plan
    from core2.plan.planner import activity_meant

    plan = Plan(kind="query", lifecycle=Lifecycle(of="supplier", measure="number_of_receipts", status="repeat"))
    for question in ("how many suppliers did we order from on 2 or more days this month",
                     "Combien de fournisseurs avec au moins une commande chaque mois ?"):
        assert activity_meant(plan, model, question).lifecycle.measure == "number_of_po_numbers", question
    said = activity_meant(plan, model, "how many suppliers did we receive goods from on 2 or more days")
    assert said.lifecycle.measure == "number_of_receipts"


def test_a_name_like_one_in_force_is_of_its_field(unread):
    # "MRO Desk", then "Packaging Desk": a name ending as the one in force is of the same field, though no word of
    # it names the field.
    from core2.plan.ir import Filter, Plan
    from core2.standing import quoted_unread

    built, model, warehouse, index = unread
    plan = Plan(kind="query", measures=["net_amount"])
    got = quoted_unread(plan, 'same for "Old West"', model, index,
                        [Filter(field="region.name", op="eq", values=["Far West"])])
    assert [(f.field, f.values) for f in got.filters] == [("region.name", ["Old West"])]
    # A name the AI kept whole, or no name like it in force: as it was read.
    assert quoted_unread(plan, 'same for "Old West"', model, index, []) == plan


def test_members_with_none_of_the_figure_are_not_listed(finance):
    # Finance FN5-06.2: revenue in CAD by cost centre listed the 9 cost centres that book only expenses as "no
    # value", and said "across 11 cost centres": revenue is the ledger's lines on revenue accounts, none of theirs.
    chat = Chat(finance)
    got = chat.ask('revenue in "CAD" by cost centre in 2025',
                   {"measures": ["revenue"], "group_by": ["cost_centre.centre_name"],
                    "filters": [{"field": "journal_line.currency_code", "op": "eq", "values": ["CAD"]}],
                    "time": {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}})
    want = dict(finance[0].con.execute("""SELECT c.cost_centre_name, SUM(l.credit_amount - l.debit_amount)
        FROM journal_lines l JOIN accounts a USING (account_id) JOIN cost_centres c USING (cost_centre_id)
        JOIN calendar d ON l.posting_date_key = d.date_key
        WHERE a.account_type = 'Revenue' AND l.currency_code = 'CAD' AND d.full_date BETWEEN DATE '2025-01-01' AND DATE '2025-12-31'
        GROUP BY 1""").fetchall())
    rows = got["export_rows"]
    shown = {r["cost_centre_centre_name"]: r["revenue"] for r in rows}
    assert shown == pytest.approx({k: float(v) for k, v in want.items()}), got["answer"]["headline"]
    assert f"across {len(want)} cost centres" in got["answer"]["headline"]


def test_the_lab_after_a_question_about_one_lab(finance):
    # Finance FN5-03.4: "What did the lab spend in March 2026?" after "R&D Lab #2" in CAD was read as a new
    # question, so every cost centre's spend in every currency was summed: "the lab" points at the one above.
    chat = Chat(finance)
    q1 = {"time": {"window": {"kind": "between", "start": "2026-01-01", "end": "2026-03-31"}}}
    chat.ask('How much did "R&D Lab #2" spend in "CAD" in Q1 2026?',
             {"measures": ["expenses"], **q1, "filters": [
                 {"field": "cost_centre.centre_name", "op": "eq", "values": ["R&D Lab #2"]},
                 {"field": "journal_line.currency_code", "op": "eq", "values": ["CAD"]}]})
    got = chat.ask("What did the lab spend in March 2026?",
                   {"measures": ["expenses"],
                    "time": {"window": {"kind": "between", "start": "2026-03-01", "end": "2026-03-31"}}})
    want = finance[0].con.execute("""SELECT SUM(l.debit_amount - l.credit_amount)
        FROM journal_lines l JOIN accounts a USING (account_id) JOIN cost_centres c USING (cost_centre_id)
        JOIN calendar d ON l.posting_date_key = d.date_key
        WHERE a.account_type = 'Expense' AND l.currency_code = 'CAD' AND c.cost_centre_name = 'R&D Lab #2'
          AND d.full_date BETWEEN DATE '2026-03-01' AND DATE '2026-03-31'""").fetchone()[0]
    assert float(got["kpi"]["value"]) == pytest.approx(float(want)), got["answer"]["headline"]


@pytest.mark.parametrize("field", ["account", "account.name"])
def test_accounts_under_a_parent_account(finance, field):
    # Finance FN5-04.4: 'top 5 accts under "Operating Expenses"' filtered the account called Operating Expenses,
    # a heading no line is posted to: "No rows match". "Under" a member of a list that links to itself is its
    # children, by that link.
    chat = Chat(finance)
    got = chat.ask('top 5 accts under "Operating Expenses" in "CAD"',
                   {"measures": ["expenses"], "group_by": ["account.name"],
                    "filters": [{"field": field, "op": "eq", "values": ["Operating Expenses"]},
                                {"field": "journal_line.currency_code", "op": "eq", "values": ["CAD"]}],
                    "sort": [{"by": "expenses", "desc": True}], "limit": 5})
    want = finance[0].con.execute("""SELECT a.account_name, SUM(l.debit_amount - l.credit_amount) v
        FROM journal_lines l JOIN accounts a USING (account_id) JOIN accounts p ON a.parent_account_id = p.account_id
        WHERE a.account_type = 'Expense' AND l.currency_code = 'CAD' AND p.account_name = 'Operating Expenses'
        GROUP BY 1 ORDER BY v DESC LIMIT 5""").fetchall()
    rows = got["export_rows"]
    assert [(r["account_name"], pytest.approx(float(r["expenses"]))) for r in rows] == \
        [(k, pytest.approx(float(v))) for k, v in want], got["answer"]["headline"]


def test_our_total_revenue_lets_the_account_above_go(finance):
    # Finance FN5-04.6: "So what is our total revenue in "CAD"?" after the accounts under "Operating Expenses" kept
    # that heading: "no value". Our total is the whole.
    chat = Chat(finance)
    year = {"time": {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}}
    cad = {"field": "journal_line.currency_code", "op": "eq", "values": ["CAD"]}
    chat.ask('expenses under "Operating Expenses" in "CAD" in 2025',
             {"measures": ["expenses"], **year,
              "filters": [{"field": "account", "op": "eq", "values": ["Operating Expenses"]}, cad]})
    got = chat.ask('So what is our total revenue in "CAD"?', {"measures": ["revenue"], **year, "filters": [cad],
                                                             "follow_up": "refine"})
    want = finance[0].con.execute("""SELECT SUM(l.credit_amount - l.debit_amount)
        FROM journal_lines l JOIN accounts a USING (account_id) JOIN calendar d ON l.posting_date_key = d.date_key
        WHERE a.account_type = 'Revenue' AND l.currency_code = 'CAD' AND d.year = 2025""").fetchone()[0]
    assert float(got["kpi"]["value"]) == pytest.approx(float(want)), got["answer"]["headline"]


@pytest.fixture(scope="module")
def home():
    built, model = learn(domains.build("home_services"), "descriptive")
    warehouse = DuckDBWarehouse(built.con)

    def fetch(slug: str) -> list[object]:
        column = model.columns[model.attributes[slug].column]
        return [r[0] for r in warehouse.query(
            f'SELECT DISTINCT "{column.name}" FROM "{model.tables[column.table].name}"').rows]

    return built, model, warehouse, build_index(model, fetch)


def test_a_cancellation_rate_counts_the_cancelled_bookings(home):
    # Home services HP-01.6: "le taux d'annulation des réservations" was 0.0%: cancelled bookings are left out by
    # default, and the rate's condition names the status by the lookup's name, not the bookings' own status number.
    chat = Chat(home)
    got = chat.ask("Quel a été le taux d'annulation des réservations en 2025 ?",
                   {"rates": [{"name": "Cancellation Rate", "measure": "number_of_bookings",
                               "where": [{"field": "booking_status.name", "op": "eq",
                                          "values": ["Closed - Cancelled"]}]}],
                    "time": {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}})
    want = home[0].con.execute("""SELECT 100.0 * COUNT(*) FILTER (WHERE k.status_id = 4) / COUNT(*)
        FROM bookings k JOIN calendar d ON k.booked_date_key = d.date_key WHERE d.year = 2025""").fetchone()[0]
    assert float(got["kpi"]["value"]) == pytest.approx(float(want)), got["answer"]["headline"]


def test_loyal_last_quarter_is_every_month_of_it(home):
    # Home services HP-07.2: "loyal customers last quarter, booking in every month of it" counted the quarters: 337
    # customers booked in "every quarter" of one quarter, where a handful booked in each of its three months.
    chat = Chat(home)
    got = chat.ask("How many loyal customers did we have last quarter, booking in every month of it?",
                   {"lifecycle": {"of": "customer", "measure": "number_of_bookings", "status": "loyal"},
                    "time": {"window": {"kind": "previous", "unit": "quarter"}}})
    want = home[0].con.execute("""SELECT COUNT(*) FROM (
        SELECT k.customer_id FROM bookings k JOIN calendar d ON k.booked_date_key = d.date_key
        WHERE k.status_id NOT IN (4, 6, 7) AND d.full_date BETWEEN DATE '2026-01-01' AND DATE '2026-03-31'
        GROUP BY 1 HAVING COUNT(DISTINCT date_trunc('month', d.full_date)) = 3)""").fetchone()[0]
    assert float(got["kpi"]["value"]) == want, got["answer"]["headline"]


def test_how_many_we_lost_each_month_is_a_number_not_a_rate(retail):
    # Pharmacy RXP-07.5: "Combien de patients avons-nous perdus chaque mois" was planned as the churn rate: how many
    # is a number of members.
    chat = Chat(retail)
    got = chat.ask("Combien de clients avons-nous perdus chaque mois en 2025 ?",
                   {"lifecycle": {"of": "customer", "measure": "number_of_orders", "status": "churn_rate"},
                    "time": {"grain": "month", "window": {"kind": "between", "start": "2025-01-01",
                                                          "end": "2025-12-31"}}})
    assert "lapsed" in got["export_rows"][0] and "churn_rate" not in got["export_rows"][0], got["export_rows"][0]
    rate = Chat(retail).ask("Quel est notre taux d'attrition chaque mois en 2025 ?",
                            {"lifecycle": {"of": "customer", "measure": "number_of_orders", "status": "churn_rate"},
                             "time": {"grain": "month", "window": {"kind": "between", "start": "2025-01-01",
                                                                   "end": "2025-12-31"}}})
    assert "churn_rate" in rate["export_rows"][0]


def test_ytd_is_the_calendars_unless_fiscal_is_said(retail):
    # Home services HP-08.2: "plan renewal fees ytd" in a workspace whose fiscal year starts in July was counted
    # from 1 Jul 2025: the AI read it as fiscal. Like a named year, it is the calendar's unless fiscal is said.
    from core2.plan.ir import Plan, TimeSpec, Window
    from core2.plan.planner import calendar_meant

    model = retail[1]
    fiscal = Plan(kind="query", measures=["net_amount"],
                  time=TimeSpec(window=Window(kind="to_date", unit="year", fiscal=True)))
    assert calendar_meant(fiscal, model, "plan renewal fees ytd").time.window.fiscal is False
    assert calendar_meant(fiscal, model, "plan renewal fees fiscal ytd").time.window.fiscal is True
    assert calendar_meant(fiscal, model, "net sales FYTD").time.window.fiscal is True


def test_a_cancelled_bookings_lines_are_left_out_with_it(home):
    # Home services HP-02.1, HP-07.5: "service revenue" summed the lines of cancelled, duplicate and voided bookings,
    # which the bookings leave out by default, and new against returning customers' spend mixed the two: the
    # customers were told apart by their kept bookings, their spend by every line.
    chat = Chat(home)
    year = {"time": {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}}
    got = chat.ask("What was our service revenue in 2025?", {"measures": ["line_amount"], **year})
    want = home[0].con.execute("""SELECT SUM(l.line_amount) FROM booking_lines l
        LEFT JOIN bookings k USING (booking_number) JOIN calendar d ON l.booked_date_key = d.date_key
        WHERE (k.status_id NOT IN (4, 6, 7) OR k.status_id IS NULL) AND d.year = 2025""").fetchone()[0]
    assert float(got["kpi"]["value"]) == pytest.approx(float(want)), got["answer"]["headline"]
    assert "Closed - Cancelled" in json.dumps(got), "the answer says the cancelled bookings' lines are left out"


def test_new_and_returning_spend_add_up_to_the_kept_revenue(home):
    chat = Chat(home)
    year = {"time": {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}}
    split = chat.ask("Spend from new vs returning customers in 2025",
                     {"measures": ["line_amount"], **year,
                      "segment": {"of": "customer", "measure": "number_of_bookings", "status": "new", "split": True}})
    total = Chat(home).ask("What was our service revenue in 2025?", {"measures": ["line_amount"], **year})
    shown = sum(float(r["line_amount"]) for r in split["export_rows"])
    assert shown == pytest.approx(float(total["kpi"]["value"])), split["answer"]["headline"]


@pytest.fixture(scope="module")
def purchasing():
    built, model = learn(domains.build("purchasing"), "descriptive")
    return built, model, DuckDBWarehouse(built.con), MemberIndex()


def test_a_follow_up_keeps_the_activity_above(purchasing):
    # Purchasing PU5-07.6: "and how many were one-time?" after the suppliers we ordered from on 2 or more days was
    # counted on receipts, the larger table: a follow-up naming no activity keeps the one above.
    from core2.plan.ir import Lifecycle, Plan
    from core2.plan.planner import activity_meant

    model = purchasing[1]
    asked = Plan(kind="query", follow_up="refine",
                 lifecycle=Lifecycle(of="supplier", measure="number_of_po_numbers", status="one_time"))
    assert activity_meant(asked, model, "and how many were one-time?").lifecycle.measure == "number_of_po_numbers"
    new = asked.model_copy(update={"follow_up": None})
    assert activity_meant(new, model, "how many one-time suppliers this month").lifecycle.measure == "number_of_receipts"


def test_what_we_spent_with_those_counted_is_their_spend(purchasing):
    # Purchasing PU5-08.4: "and what did we spend with them" was planned as the won-back suppliers with the line
    # amount beside them, and answered with their number again: a figure asked of those members is theirs.
    from core2.plan.ir import Lifecycle, Plan
    from core2.plan.planner import figure_of_those

    model = purchasing[1]
    plan = Plan(kind="query", measures=["line_amount"], follow_up="refine",
                lifecycle=Lifecycle(of="supplier", measure="number_of_po_numbers", status="won_back"))
    got = figure_of_those(plan, model, "and what did we spend with them")
    assert got.lifecycle is None and got.segment is not None and got.segment.status == "won_back"
    assert got.measures == ["line_amount"]
    counted = plan.model_copy(update={"measures": ["number_of_suppliers"]})
    assert figure_of_those(counted, model, "how many suppliers did we win back").lifecycle is not None


def test_how_many_new_customers_stays_their_number(retail):
    from core2.plan.ir import Lifecycle, Plan
    from core2.plan.planner import figure_of_those

    plan = Plan(kind="query", measures=["number_of_orders"],
                lifecycle=Lifecycle(of="customer", measure="number_of_orders", status="new"))
    assert figure_of_those(plan, retail[1], "how many new customers did we get in Q2").lifecycle is not None
    assert figure_of_those(plan, retail[1], "how many orders did new customers place in Q2").segment is not None


def test_customers_created_under_a_standing_period(retail):
    # Banking BP-04.2: "How many accounts did we open?" under "from now on the last 2 months" asked which date to
    # count by: the AI planned no period (the standing one is put on after it), so the verb was never read.
    chat = Chat(retail, today=dt.date(2026, 6, 15))
    chat.ask("From now on use the last 2 months")
    got = chat.ask("How many customers did we create?", {"measures": ["number_of_customers"]})
    # Counted by the customers' own date; it stops in April 2024, so the 2 months are counted back from there.
    want = retail[0].con.execute("""SELECT COUNT(*) FROM customers
        WHERE created_at >= TIMESTAMP '2024-02-01' AND created_at < TIMESTAMP '2024-04-01'""").fetchone()[0]
    assert got.get("kpi") and float(got["kpi"]["value"]) == want, got.get("answer", {}).get("headline")
    assert "Feb 2024 to Mar 2024" in got["answer"]["headline"]


def test_a_cancellation_rate_on_the_status_list_itself(home):
    # Home services HP-01.6 again: the rate's condition named the status list ("booking_status"), not its name field.
    chat = Chat(home)
    got = chat.ask("Quel a été le taux d'annulation des réservations en 2025 ?",
                   {"rates": [{"name": "Cancellation Rate", "measure": "number_of_bookings",
                               "where": [{"field": "booking_status", "op": "eq", "values": ["Closed - Cancelled"]}]}],
                    "time": {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}})
    want = home[0].con.execute("""SELECT 100.0 * COUNT(*) FILTER (WHERE k.status_id = 4) / COUNT(*)
        FROM bookings k JOIN calendar d ON k.booked_date_key = d.date_key WHERE d.year = 2025""").fetchone()[0]
    assert float(got["kpi"]["value"]) == pytest.approx(float(want)), got["answer"]["headline"]


def test_none_after_the_data_named_as_a_date_condition_is_said(home):
    # Home services HP-08.4: "which of them are going to cancel their plan next month?" was planned as the members
    # with no renewal between 1 and 30 November 2026, as a condition on the renewal date, not the question's
    # period: "543 customers ... had none in November 2026". The data ends in June: said, never answered.
    chat = Chat(home)
    got = chat.ask("which of them are going to cancel their plan next month?",
                   {"intent": "list", "absent": {
                       "of": "customer", "measure": "number_of_plan_renewals",
                       "without": [{"field": "renewal_date", "op": "between", "values": ["2026-11-01", "2026-11-30"]}],
                       "had": [{"field": "renewal_date", "op": "between", "values": ["2024-01-01", "2024-12-31"]}]}})
    said = got.get("answer", {}).get("headline") or got.get("text") or ""
    assert "Your data ends" in said and "had none" not in said, said


def test_new_vs_returning_customers_keeps_both(retail):
    # The Phase 5 screenshots: "new vs returning customers by month in 2025", planned as the number of customers
    # split new against returning, was answered with the returning customers alone.
    chat = Chat(retail)
    got = chat.ask("new vs returning customers by month in 2025",
                   {"intent": "breakdown", "measures": ["number_of_customers"],
                    "segment": {"of": "customer", "measure": "number_of_orders", "split": True},
                    "time": {"grain": "month", "window": {"kind": "between", "start": "2025-01-01",
                                                          "end": "2025-12-31"}}})
    first = got["export_rows"][0]
    assert "new" in first and "returning" in first, first


# ── after Phase 7: the last two misses ──────────────────────────────────────


@pytest.fixture(scope="module")
def ledger():
    built, model = learn(domains.build("finance"), "descriptive")
    warehouse = DuckDBWarehouse(built.con)

    def fetch(slug: str) -> list[object]:
        column = model.columns[model.attributes[slug].column]
        table = model.tables[column.table]
        return [r[0] for r in warehouse.query(f'SELECT DISTINCT "{column.name}" FROM "{table.name}"').rows]

    return built, model, warehouse, build_index(model, fetch)


Y25 = {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}, "date": "posting_date"}


def test_the_accounts_under_a_quoted_expense_parent_are_ranked_by_their_expenses(ledger):
    # Finance FN5-04.4 with gpt-4o: 'top 5 accts under "Operating Expenses" in "CAD"' right after "how many journals
    # did we post in 2025" was ranked by the number of journals kept from above. The question names no figure of its
    # own; the parent it names, "Operating Expenses", holds one: the accounts' expenses.
    chat = Chat(ledger)
    chat.ask("how many journals did we post in 2025", {"intent": "value", "measures": ["number_of_journals"], "time": Y25})
    got = chat.ask('top 5 accts under "Operating Expenses" in "CAD"', {
        "intent": "rank", "measures": ["number_of_journals"], "group_by": ["account"],
        "filters": [{"field": "account.name", "op": "eq", "values": ["Operating Expenses"]},
                    {"field": "journal_line.currency_code", "op": "eq", "values": ["CAD"]}],
        "sort": [{"by": "number_of_journals", "desc": True}], "limit": 5, "time": Y25, "follow_up": "refine"})
    readings = []
    for kept in ("TRUE", "j.status <> 'REV'"):
        readings.append([(n, round(float(v), 2)) for n, v in ledger[0].con.execute(f"""
            SELECT a.account_name, SUM(j.debit_amount - j.credit_amount) v FROM journal_lines j
            JOIN accounts a ON a.account_id = j.account_id JOIN accounts p ON p.account_id = a.parent_account_id
            WHERE p.account_name = 'Operating Expenses' AND a.account_type = 'Expense' AND j.currency_code = 'CAD'
              AND j.posting_date_key BETWEEN 20250101 AND 20251231 AND {kept}
            GROUP BY 1 ORDER BY v DESC LIMIT 5""").fetchall()])
    rows = got["export_rows"]
    shown = [(r["account_name"], round(float(r["expenses"]), 2)) for r in rows]
    assert shown in readings, (shown, readings)


def test_a_count_the_question_names_is_kept_under_a_quoted_parent(ledger):
    chat = Chat(ledger)
    got = chat.ask('top 5 accts by number of journals under "Operating Expenses" in "CAD" in 2025', {
        "intent": "rank", "measures": ["number_of_journals"], "group_by": ["account"],
        "filters": [{"field": "account.name", "op": "eq", "values": ["Operating Expenses"]},
                    {"field": "journal_line.currency_code", "op": "eq", "values": ["CAD"]}],
        "sort": [{"by": "number_of_journals", "desc": True}], "limit": 5, "time": Y25})
    assert "number_of_journals" in got["export_rows"][0]


def test_the_definition_of_won_back_said_again_is_no_filter_of_its_own(retail):
    # Pharmacy RXP-06.3 with gpt-4o: "how many of those did we win back, ones with no fills at all the quarter
    # before?" was planned as won back AND "number of fills = 0", which no plan can answer: the reader was saying what
    # won back means. Asked again here of the retail customers in Q4 2025.
    chat = Chat(retail)
    got = chat.ask("how many customers did we win back in Q4 2025, ones with no orders at all the quarter before?", {
        "intent": "value", "measures": ["number_of_customers"],
        "time": {"window": {"kind": "between", "start": "2025-10-01", "end": "2025-12-31"}},
        "lifecycle": {"of": "customer", "measure": "number_of_orders", "status": "won_back"},
        "filters": [{"field": "number_of_orders", "op": "eq", "values": [0], "total": False}]})
    want = retail[0].con.execute(f"""
        WITH days AS (SELECT DISTINCT customer_id AS k, day FROM ({LINES}) l
                      WHERE customer_id IN (SELECT customer_id FROM customers) AND day < DATE '2026-01-01')
        SELECT COUNT(DISTINCT k) FROM days
        WHERE day >= DATE '2025-10-01' AND day < DATE '2026-01-01'
          AND k IN (SELECT k FROM days WHERE day < DATE '2025-07-01')
          AND k NOT IN (SELECT k FROM days WHERE day >= DATE '2025-07-01' AND day < DATE '2025-10-01')""").fetchone()[0]
    assert want > 0
    assert got.get("kpi") is not None or got.get("export_rows"), got.get("answer") or got.get("text")
    value = got["export_rows"][0].get("won_back", next(iter(got["export_rows"][0].values())))
    assert int(value) == want
