"""A ranking with each one's share of the whole: the share of the figure ranked by, of every row.

Found replaying a real model's (gpt-4o) plans for an analyst's requests:

* "Top 5 products by gross margin $ in 2025, with net sales and what share of the year's total margin each one makes
  up" was planned as a ranking with no share: the share was never shown. Asked as a share, it was the share of the
  first figure shown (net sales), not of the margin the rows are ranked by.
* "Top 5 loan officers by number of loans booked, with their home branch and their share of all loans booked" gave
  each officer's share of the loans that had an officer: the rows with none, left out of the ranking, were left out
  of the whole too (5.44% where 4.0% is right).

Each answer goes through the whole answer path, the AI stood in by the plan it wrote, and is compared with SQL written
by hand on the invented retail warehouse, with sales that have no customer on file.

Synthetic data only.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
IN_2025 = {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}
MARGIN = {"name": "Gross margin $", "op": "difference", "measures": ["net_amount", "cost_amount"]}
WHERE_2025 = "d.full_date >= DATE '2025-01-01' AND d.full_date < DATE '2026-01-01'"


@pytest.fixture(scope="module")
def retail():
    built, model = learn(domains.build("retail"), "descriptive")
    con = built.con
    # Walk-in sales with no customer on file, more than any one customer bought: no customer, but part of all sales.
    (key,), = con.execute("SELECT MIN(date_key) FROM calendar WHERE full_date >= DATE '2025-06-01'").fetchall()
    (most,), = con.execute(f"""SELECT MAX(s) FROM (SELECT SUM(o.net_amount) s FROM order_lines o
                               JOIN calendar d ON d.date_key = o.order_date_key WHERE {WHERE_2025}
                               GROUP BY o.customer_id)""").fetchall()
    net = round(float(most) * 2, 2)
    con.execute("INSERT INTO order_lines (order_line_id, order_number, line_number, customer_id, product_id, store_id, "
                "order_date_key, quantity, unit_price, gross_amount, discount_amount, net_amount, cost_amount, "
                f"status_code) SELECT MAX(order_line_id) + 1, 'WALKIN-1', 1, NULL, MIN(product_id), 1, {key}, 1, "
                f"{net}, {net}, 0, {net}, {net / 2}, 'D' FROM order_lines")
    return built, model


def _ask(retail, question: str, plan: dict) -> dict:
    built, model = retail
    answers = [json.dumps({"kind": "query", **plan})]
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con),
                        complete=lambda s, t: answers.pop(0) if len(answers) > 1 else answers[0],
                        index=MemberIndex(), today=TODAY)
    payload = answer_question(question, services, Session())
    assert "export_rows" in payload, payload.get("answer")
    return payload


def _share(row: dict) -> float:
    """The row's share, as a fraction (the export shows it in per cent)."""
    (value,) = [v for k, v in row.items() if k.endswith("_share")]
    return float(value) / 100


def test_each_ones_share_is_of_every_row_those_with_no_member_among_them(retail):
    payload = _ask(retail, "Top 5 customers by net sales in 2025, with their segment and their share of all sales "
                           "in 2025",
                   {"intent": "share", "measures": ["net_amount"], "group_by": ["customer.name", "customer.segment"],
                    "time": IN_2025, "sort": [{"by": "net_amount", "desc": True}], "limit": 5})
    con = retail[0].con
    (whole,), = con.execute(f"""SELECT SUM(o.net_amount) FROM order_lines o
                                JOIN calendar d ON d.date_key = o.order_date_key WHERE {WHERE_2025}""").fetchall()
    top = con.execute(f"""SELECT c.customer_name, SUM(o.net_amount) FROM order_lines o
                          JOIN calendar d ON d.date_key = o.order_date_key
                          JOIN customers c ON c.customer_id = o.customer_id WHERE {WHERE_2025}
                          GROUP BY c.customer_id, c.customer_name ORDER BY 2 DESC LIMIT 5""").fetchall()
    rows = payload["export_rows"]
    assert [r["customer_name"] for r in rows] == [name for name, _ in top], "the 5 customers, the walk-ins none"
    for row, (name, net) in zip(rows, top):
        assert _share(row) == pytest.approx(float(net) / float(whole), abs=1e-6), name
    together = sum(float(net) for _, net in top) / float(whole)
    assert f"{round(100 * together)}% of the total" in payload["answer"]["headline"], payload["answer"]["headline"]


def test_a_ranking_by_margin_shows_each_ones_share_of_the_total_margin(retail):
    question = ("Top 5 products by gross margin $ in 2025, with net sales and what share of the year's total margin "
                "each one makes up")
    # As the AI wrote it: a ranking, the share asked for and never planned.
    payload = _ask(retail, question, {"intent": "rank", "measures": ["net_amount"], "derived": [MARGIN],
                                      "group_by": ["product.name"], "time": IN_2025,
                                      "sort": [{"by": "Gross margin $", "desc": True}], "limit": 5})
    con = retail[0].con
    (whole,), = con.execute(f"""SELECT SUM(o.net_amount - o.cost_amount) FROM order_lines o
                                JOIN calendar d ON d.date_key = o.order_date_key WHERE {WHERE_2025}""").fetchall()
    top = con.execute(f"""SELECT p.product_name, SUM(o.net_amount - o.cost_amount) FROM order_lines o
                          JOIN calendar d ON d.date_key = o.order_date_key
                          JOIN products p ON p.product_id = o.product_id WHERE {WHERE_2025}
                          GROUP BY p.product_id, p.product_name ORDER BY 2 DESC LIMIT 5""").fetchall()
    rows = payload["export_rows"]
    assert [r["product_name"] for r in rows] == [name for name, _ in top]
    for row, (name, margin) in zip(rows, top):
        assert _share(row) == pytest.approx(float(margin) / float(whole), abs=1e-6), name


@pytest.fixture(scope="module")
def pharmacy():
    return learn(domains.build("compounding_pharmacy"), "descriptive")


def _claims_by_code(pharmacy, *, rejected_only: bool) -> tuple[list[tuple[str, int]], int]:
    con = pharmacy[0].con
    in_2025 = "submitted_date >= DATE '2025-01-01' AND submitted_date < DATE '2026-01-01'"
    top = con.execute(f"""SELECT reject_code, COUNT(*) FROM claims WHERE {in_2025} AND reject_code IS NOT NULL
                          GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT 3""").fetchall()
    rejected = "AND reject_code IS NOT NULL" if rejected_only else ""
    (whole,), = con.execute(f"SELECT COUNT(*) FROM claims WHERE {in_2025} {rejected}").fetchall()
    return [(str(code), int(n)) for code, n in top], int(whole)


@pytest.mark.parametrize("asked,rejected_only", [
    ("what % of all 2025 rejections each one is", True), ("what % of all claims in 2025 each one is", False),
    # Said of the code itself: still each code's share of the rejections, never a rate.
    ("what % of all 2025 rejections came from that code", True),
    ("what % of all 2025 rejections were for that code", True)])
def test_a_whole_that_names_the_grouping_is_the_rows_that_have_one(pharmacy, asked, rejected_only):
    # A claim has a reject code only when it was rejected: "% of all rejections" is of those, "% of all claims" of
    # every claim.
    built, model = pharmacy
    plan = {"intent": "share", "measures": ["number_of_claims"], "group_by": ["claim.reject_code"],
            "time": {"date": "submitted_date", **IN_2025}, "sort": [{"by": "number_of_claims", "desc": True}],
            "limit": 3}
    answers = [json.dumps({"kind": "query", **plan})]
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con),
                        complete=lambda s, t: answers.pop(0) if len(answers) > 1 else answers[0],
                        index=MemberIndex(), today=TODAY)
    payload = answer_question(f"top 3 claim reject codes in 2025 - the code, how many, and {asked}", services, Session())
    assert "export_rows" in payload, payload.get("answer")
    top, count = _claims_by_code(pharmacy, rejected_only=rejected_only)
    rows = payload["export_rows"]
    assert [str(r["claim_reject_code"]) for r in rows] == [code for code, _ in top]
    for row, (code, n) in zip(rows, top):
        assert _share(row) == pytest.approx(n / count, abs=1e-6), code


def _store_shares(retail) -> dict[str, float]:
    return {name: float(share) for name, share in retail[0].con.execute(f"""
        SELECT s.store_name, SUM(o.net_amount) / SUM(SUM(o.net_amount)) OVER ()
        FROM order_lines o JOIN stores s ON s.store_id = o.store_id JOIN calendar d ON d.date_key = o.order_date_key
        WHERE {WHERE_2025} GROUP BY 1""").fetchall()}


def test_the_share_is_of_the_figure_the_whole_names_not_of_the_one_ranked_by(retail):
    payload = _ask(retail, "Top 5 stores by discount given in 2025, with net sales and their % of total net sales",
                   {"intent": "rank", "measures": ["discount_amount", "net_amount"], "group_by": ["store.name"],
                    "time": IN_2025, "sort": [{"by": "discount_amount", "desc": True}], "limit": 5})
    shares = _store_shares(retail)
    for row in payload["export_rows"]:
        assert set(k for k in row if k.endswith("_share")) == {"net_amount_share"}, row
        assert _share(row) == pytest.approx(shares[row["store_name"]], abs=1e-6), row["store_name"]
    # The discount it leads with is no share of anything: never "(24% of the total)" of the 5 stores shown.
    assert "of the total" not in payload["answer"]["headline"], payload["answer"]["headline"]


def test_a_condition_on_the_share_is_on_the_share_of_the_figure_it_is_of(retail):
    payload = _ask(retail, "stores with more than 10% of total net sales in 2025, by net sales, with their discount",
                   {"intent": "share", "measures": ["discount_amount", "net_amount"], "group_by": ["store.name"],
                    "time": IN_2025, "sort": [{"by": "net_amount", "desc": True}],
                    "filters": [{"field": "share", "op": "gt", "values": [10]}]})
    assert {r["store_name"] for r in payload["export_rows"]} == \
        {name for name, share in _store_shares(retail).items() if share > 0.10}


def test_a_share_of_each_groupings_own_total_is_kept(retail):
    payload = _ask(retail, "net sales by region and category in 2025, and the % of total regional sales for each "
                           "category",
                   {"intent": "breakdown", "measures": ["net_amount"], "group_by": ["region.name", "category.name"],
                    "time": IN_2025, "share_of": "region.name"})
    by_region: dict[str, float] = {}
    for row in payload["export_rows"]:
        by_region[row["region_name"]] = by_region.get(row["region_name"], 0.0) + _share(row)
    assert all(total == pytest.approx(1.0, abs=1e-3) for total in by_region.values()), by_region


def test_a_whole_that_names_nothing_on_the_answer_is_no_share_of_it(retail):
    # "What % of all orders were returned" is a rate of orders, never each product's share of the returns.
    payload = _ask(retail, "top 5 products by returns in 2025, and what % of all orders were returned",
                   {"intent": "rank", "measures": ["number_of_returns"], "group_by": ["product.name"],
                    "time": {"date": "return_date", **IN_2025}, "limit": 5})
    assert not any(k.endswith("_share") for r in payload["export_rows"] for k in r)


def _customer_shares(retail) -> dict[str, float]:
    return {name: float(share) for name, share in retail[0].con.execute(f"""
        SELECT c.customer_name, SUM(o.net_amount) / (SELECT SUM(o2.net_amount) FROM order_lines o2
                                                     JOIN calendar d2 ON d2.date_key = o2.order_date_key
                                                     WHERE {WHERE_2025.replace('d.', 'd2.')})
        FROM order_lines o JOIN customers c ON c.customer_id = o.customer_id
             JOIN calendar d ON d.date_key = o.order_date_key
        WHERE {WHERE_2025} GROUP BY c.customer_id, c.customer_name""").fetchall()}


def test_a_share_the_ai_wrote_of_a_figure_is_kept_whatever_the_whole_is_called(retail):
    # "Share of total spend": no figure here is called spend, the AI wrote the share of net sales.
    payload = _ask(retail, "Which 5 customers spent the most in 2025, with their segment, and what share of total "
                           "spend each one made up",
                   {"intent": "rank", "measures": ["net_amount"], "group_by": ["customer.name", "customer.segment"],
                    "time": IN_2025, "sort": [{"by": "net_amount", "desc": True}], "limit": 5,
                    "share_of": "net_amount"})
    shares = _customer_shares(retail)
    assert payload["export_rows"] and all(_share(r) == pytest.approx(shares[r["customer_name"]], abs=1e-6)
                                          for r in payload["export_rows"])


def test_a_share_within_the_row_itself_is_a_share_of_the_whole(retail):
    # The AI wrote "each product's own total" for "their share of 2025's total sales": every product at 100%.
    payload = _ask(retail, "Top 5 products by net sales in 2025, with their category and their share of 2025's total "
                           "sales",
                   {"intent": "rank", "measures": ["net_amount"], "group_by": ["product.name", "category.name"],
                    "time": IN_2025, "sort": [{"by": "net_amount", "desc": True}], "limit": 5,
                    "share_of": "product.name"})
    (whole,), = retail[0].con.execute(f"""SELECT SUM(o.net_amount) FROM order_lines o
                                          JOIN calendar d ON d.date_key = o.order_date_key
                                          WHERE {WHERE_2025}""").fetchall()
    for row in payload["export_rows"]:
        assert _share(row) == pytest.approx(float(row["net_amount"]) / float(whole), abs=1e-6), row


@pytest.mark.parametrize("asked", [
    "with each one's % of the grand total", "and their share of the overall total", "and their share of the year's total",
    "and what percent of the company total each one is", "and what share of total spend each one made up",
    "and what percentage of all turnover each made", "and their share of total business", "and their % of the total"])
def test_a_share_asked_for_in_any_words_is_worked_out(retail, asked):
    payload = _ask(retail, f"Top 5 stores by net sales in 2025, {asked}",
                   {"intent": "rank", "measures": ["net_amount"], "group_by": ["store.name"], "time": IN_2025,
                    "sort": [{"by": "net_amount", "desc": True}], "limit": 5})
    shares = _store_shares(retail)
    assert payload["export_rows"], payload["answer"]
    for row in payload["export_rows"]:
        assert _share(row) == pytest.approx(shares[row["store_name"]], abs=1e-6), (asked, row)


def test_a_rate_of_other_rows_beside_a_ranking_is_no_share_of_its_figure(retail):
    payload = _ask(retail, "top 5 products by net sales in 2025, and what % of all orders were returned",
                   {"intent": "rank", "measures": ["net_amount"], "group_by": ["product.name"], "time": IN_2025,
                    "sort": [{"by": "net_amount", "desc": True}], "limit": 5})
    assert not any(k.endswith("_share") for r in payload["export_rows"] for k in r)


STORES_2025 = {"intent": "rank", "measures": ["net_amount"], "group_by": ["store.name"], "time": IN_2025,
               "sort": [{"by": "net_amount", "desc": True}], "limit": 5}


@pytest.mark.parametrize("rate", [
    "and what % of all customers are new", "and what % of all customers came back",
    "and what share of all orders ended up returned", "and what % of all orders shipped late",
    "and what % of all orders had a return", "and what % of all orders included a discount"])
def test_a_rate_of_rows_that_did_something_is_no_share_of_the_answers_figure(retail, rate):
    # "What % of all customers are new" beside a ranking of stores was each store's share of net sales.
    payload = _ask(retail, f"Top 5 stores by net sales in 2025, {rate}", STORES_2025)
    assert not any(k.endswith("_share") for r in payload["export_rows"] for k in r), rate


@pytest.mark.parametrize("asked", [
    "and what percentage of total sales is accounted for by each store",
    "and what % of total sales was generated by these stores", "and what % of total sales is represented by them",
    "their share of total sales, and what % of all orders were returned", "with each one's % of the entire business",
    # A verb after the figure's total says which of it; the rows named after the verb are the answer's.
    "and the % of total revenue generated", "and the % of total margin earned",
    "and what % of total revenue came from that store", "and what % of total revenue came from online orders by store",
    "and what % of all sales were made by the store", "and what share of total sales each store got"])
def test_a_share_of_the_rows_said_in_the_passive_or_beside_a_rate_is_worked_out(retail, asked):
    # A share said of the rows ("by each store", "by them") was taken for a rate, and so was one beside a rate.
    payload = _ask(retail, f"Top 5 stores by net sales in 2025, {asked}", STORES_2025)
    shares = _store_shares(retail)
    assert payload["export_rows"], payload["answer"]
    for row in payload["export_rows"]:
        assert _share(row) == pytest.approx(shares[row["store_name"]], abs=1e-6), (asked, row)


def test_a_share_of_a_margin_shown_only_as_a_margin_percent_shows_the_margin(retail):
    margin = [{"name": "Gross margin $", "op": "difference", "measures": ["net_amount", "cost_amount"]},
              {"name": "Margin %", "op": "ratio", "measures": ["Gross margin $", "net_amount"], "scale": 100.0}]
    payload = _ask(retail, "Top 5 products by net sales in 2025 with their margin % and their share of total margin",
                   {"intent": "rank", "measures": ["net_amount"], "derived": margin, "group_by": ["product.name"],
                    "time": IN_2025, "sort": [{"by": "net_amount", "desc": True}], "limit": 5})
    (whole,), = retail[0].con.execute(f"""SELECT SUM(o.net_amount - o.cost_amount) FROM order_lines o
                                          JOIN calendar d ON d.date_key = o.order_date_key
                                          WHERE {WHERE_2025}""").fetchall()
    for row in payload["export_rows"]:
        assert row["gross_margin_share"] == pytest.approx(100 * float(row["gross_margin"]) / float(whole), abs=1e-3)


def test_a_headline_never_takes_a_share_of_the_rows_a_condition_kept(retail):
    # The stores over 10% of net sales lead by discount: the discount is no share of anything worked out here
    # ("66% of the total" of the two stores kept, where 12.7% of all discount is true).
    payload = _ask(retail, "stores with more than 10% of total net sales in 2025, with their discount",
                   {"intent": "breakdown", "measures": ["discount_amount", "net_amount"], "group_by": ["store.name"],
                    "time": IN_2025, "filters": [{"field": "share", "op": "gt", "values": [10]}]})
    assert payload["export_rows"] and "of the total" not in payload["answer"]["headline"], payload["answer"]["headline"]
