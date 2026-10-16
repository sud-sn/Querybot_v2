"""What a business asks of its figures in plain words is answered at once, with the periods it means.

Found by gpt-4o on six business-ask books (retail, purchasing, pharmacy, bank, home services, finance):

* "What is my revenue comparison?" and "What is our sales run rate?" were asked back (net, gross, discount or cost
  amount?). Told once that a business ask is answered at once, the AI plans the figure; asked back again, the figure
  is picked in code -- the money taken in, never a target, a cost or a discount it does not name -- the answer says
  which ("Read as Net amount"), and the others are a click away with no AI.
* A comparison means this year so far against the same days last year: "how are sales doing vs last year?" was 2025
  against 2024, "which stores grew the most vs last year?" too, "are we on track to beat last year?" the fiscal year
  so far, and "top 5 stores this year and what they sold last year" 2025 against 2024 ("last year" read as the
  period). A comparison and "how are we doing" are month by month.
* "This month so far vs the same days last month" was set against the 15 days before, "the last 12 months against
  the 12 before" against the same 12 months: the period before is the one of the same shape before it.
* "What is our run rate?": a year at the pace of the last three whole months, their total times four.
* "How is loan volume doing vs last year?" was planned on Loan volume target: a target is what was planned, and the
  question asks what happened.

Invented retail data whose last day is 14 Jun 2026, asked on 15 Oct 2026 (so periods count back from 15 Jun);
every figure checked against SQL written by hand. The plans are written as gpt-4o wrote them.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import replace

import pytest

from core2.plan import asks
from core2.plan.ir import Plan
from core2.service import Services, Session, answer_question
from evals.core2 import domains
from evals.core2.compile_eval import learn
from tests.test_the_product_looks_before_it_answers import LINES, LIVE, TODAY
from tests.test_the_product_looks_before_it_answers import retail as retail  # noqa: F401 - fixture

ASKED_BACK = {"kind": "clarify", "clarify": {"about": "other", "question": "Which one?",
                                             "options": ["net_amount", "gross_amount", "discount_amount",
                                                         "cost_amount"]}}
SO_FAR, SO_FAR_LY = ("2026-01-01", "2026-06-15"), ("2025-01-01", "2025-06-15")


class Chat:
    """One conversation, the AI's answers scripted (plans or questions back); every prompt it was sent is kept."""

    def __init__(self, retail, *answers: dict):
        built, model, warehouse, index = retail
        self.answers = list(answers)
        self.prompts: list[str] = []

        def complete(system: str, tail: str) -> str:
            self.prompts.append(tail)
            assert self.answers, "the AI was asked, with no answer scripted"
            return json.dumps(self.answers.pop(0))

        self.services = Services(model=model, warehouse=warehouse, complete=complete, index=index, today=TODAY)
        self.session = Session()

    def ask(self, question: str) -> dict:
        return answer_question(question, self.services, self.session)


def _query(**plan) -> dict:
    return {"kind": "query", **plan}


def _sql(retail, sql: str) -> list[tuple]:
    return retail[0].con.execute(sql).fetchall()


def _in(span: tuple[str, str]) -> str:
    return f"day >= DATE '{span[0]}' AND day < DATE '{span[1]}'"


def _by_month(retail, column: str = "net_amount") -> list[tuple]:
    return _sql(retail, f"""SELECT month(day), SUM(CASE WHEN {_in(SO_FAR)} THEN {column} END),
        SUM(CASE WHEN {_in(SO_FAR_LY)} THEN {column} END) FROM ({LINES}) l
        WHERE ({_in(SO_FAR)} OR {_in(SO_FAR_LY)}) AND {LIVE} GROUP BY 1 ORDER BY 1""")


def _totals(retail, now: tuple[str, str], before: tuple[str, str]) -> tuple:
    return _sql(retail, f"""SELECT SUM(CASE WHEN {_in(now)} THEN net_amount END),
        SUM(CASE WHEN {_in(before)} THEN net_amount END) FROM ({LINES}) l WHERE {LIVE}""")[0]


def _rows(got: dict) -> list[dict]:
    return (got.get("data") or {}).get("rows") or []


def _months_shown(got: dict, column: str = "net_amount") -> list[tuple]:
    return [(dt.date.fromisoformat(r["period"]).month, r[column], r[f"{column}_prior"]) for r in _rows(got)]


def _same(shown: list[tuple], want: list[tuple]) -> None:
    assert len(shown) == len(want)
    for (m, now, before), (wm, wnow, wbefore) in zip(shown, want):
        assert m == wm
        assert now == pytest.approx(float(wnow)) and before == pytest.approx(float(wbefore))


def _chips(got: dict) -> list[str]:
    return [s["label"] for s in got.get("follow_up_suggestions") or []]


# ── a comparison, asked back twice ────────────────────────────────────────────


def test_a_comparison_asked_back_twice_is_month_by_month_against_the_same_days_last_year(retail):
    chat = Chat(retail, ASKED_BACK, ASKED_BACK)
    got = chat.ask("What is my revenue comparison?")
    assert len(chat.prompts) == 2
    assert "Do not ask which figure is meant" in chat.prompts[1]
    # January to 14 June 2026, each month against the same month of 2025 (June: its first 14 days each year).
    _same(_months_shown(got), _by_month(retail))
    texts = [b["text"] for b in got["answer"]["badges"]]
    assert "Read as Net amount" in texts
    assert "1 Jan 2026 – 14 Jun 2026 vs 1 Jan 2025 – 14 Jun 2025" in texts
    assert {"Gross amount instead", "Cost amount instead"} <= set(_chips(got))


def test_another_figure_it_may_mean_is_one_click_away_with_no_ai(retail):
    chat = Chat(retail, ASKED_BACK, ASKED_BACK)
    chat.ask("What is my revenue comparison?")
    got = chat.ask("Gross amount instead")
    assert len(chat.prompts) == 2, "the click asked the AI"
    _same(_months_shown(got, "gross_amount"), _by_month(retail, "gross_amount"))


def test_the_other_figure_reads_in_french_too(retail):
    chat = Chat(retail, ASKED_BACK, ASKED_BACK)
    got = chat.ask("What is my revenue comparison?")
    chip = next(s for s in got["follow_up_suggestions"] if s["label"] == "Gross amount instead")
    assert chip["label_fr"] == "Gross amount à la place"
    _same(_months_shown(chat.ask(chip["question_fr"]), "gross_amount"), _by_month(retail, "gross_amount"))
    assert len(chat.prompts) == 2


def test_a_figure_the_ai_plans_at_once_is_answered_with_no_reading_said(retail):
    chat = Chat(retail, _query(intent="compare", measures=["net_amount"]))
    got = chat.ask("What is my revenue comparison?")
    _same(_months_shown(got), _by_month(retail))
    assert not [b for b in got["answer"]["badges"] if b["kind"] == "reading"]


# ── against last year: this year so far ───────────────────────────────────────


LAST_YEAR = {"window": {"kind": "previous", "unit": "year"}, "compare": {"kind": "same_period_last_year"}}


@pytest.mark.parametrize("asked", ["How are sales doing vs last year?",
                                   "Quelle est la comparaison des ventes avec l'an dernier ?"])
def test_how_sales_are_doing_against_last_year_is_this_year_so_far_month_by_month(retail, asked):
    chat = Chat(retail, _query(intent="compare", measures=["net_amount"], time=LAST_YEAR))
    _same(_months_shown(chat.ask(asked)), _by_month(retail))


def test_which_stores_grew_most_against_last_year_is_this_year_so_far_by_the_change(retail):
    chat = Chat(retail, _query(intent="compare", measures=["net_amount"], group_by=["store.name"], time=LAST_YEAR,
                               sort=[{"by": "change", "desc": True}]))
    got = chat.ask("Which stores grew the most vs last year?")
    want = _sql(retail, f"""SELECT store_name, SUM(CASE WHEN {_in(SO_FAR)} THEN net_amount END) AS now_,
        SUM(CASE WHEN {_in(SO_FAR_LY)} THEN net_amount END) AS before_ FROM ({LINES}) l
        WHERE ({_in(SO_FAR)} OR {_in(SO_FAR_LY)}) AND {LIVE} GROUP BY 1 ORDER BY now_ - before_ DESC LIMIT 3""")
    shown = _rows(got)[:3]
    assert [r["store_name"] for r in shown] == [w[0] for w in want]
    for r, w in zip(shown, want):
        assert r["net_amount"] == pytest.approx(float(w[1])) and r["net_amount_prior"] == pytest.approx(float(w[2]))


def test_which_stores_fell_most_puts_the_biggest_fall_first():
    _, model = learn(domains.build("retail"), "descriptive")
    plan = Plan.model_validate(_query(intent="compare", measures=["net_amount"], group_by=["store.name"],
                                      time=LAST_YEAR))
    got = asks.settled(plan, model, "Which stores declined the most vs last year?")
    assert [(s.by, s.desc) for s in got.sort] == [("change", False)]
    assert (got.time.window.kind, got.time.window.unit) == ("to_date", "year")


def test_this_year_and_last_year_named_together_is_this_year_so_far_with_last_years_beside_it(retail):
    # "Last year" in the question was read as the period, and "this year" as 2025.
    chat = Chat(retail, _query(intent="compare", measures=["net_amount"], group_by=["store.name"], limit=5,
                               sort=[{"by": "net_amount", "desc": True}],
                               time={"window": {"kind": "this", "unit": "year"},
                                     "compare": {"kind": "same_period_last_year"}}))
    got = chat.ask("top 5 stores this year and what they sold last year")
    want = _sql(retail, f"""SELECT store_name, SUM(CASE WHEN {_in(SO_FAR)} THEN net_amount END) AS now_,
        SUM(CASE WHEN {_in(SO_FAR_LY)} THEN net_amount END) FROM ({LINES}) l
        WHERE ({_in(SO_FAR)} OR {_in(SO_FAR_LY)}) AND {LIVE} GROUP BY 1 ORDER BY now_ DESC LIMIT 5""")
    assert [(r["store_name"], round(r["net_amount"], 2), round(r["net_amount_prior"], 2)) for r in _rows(got)] == \
        [(w[0], round(float(w[1]), 2), round(float(w[2]), 2)) for w in want]


def test_on_track_to_beat_last_year_is_the_calendar_year_so_far(retail):
    # Planned as the fiscal year so far (April here), though the question never called it fiscal.
    chat = Chat(retail, _query(intent="compare", measures=["net_amount"],
                               time={"window": {"kind": "this", "unit": "year", "fiscal": True},
                                     "compare": {"kind": "same_period_last_year"}}))
    got = chat.ask("Are we on track to beat last year's sales?")
    now, before = _totals(retail, SO_FAR, SO_FAR_LY)
    row = _rows(got)[0]
    assert row["net_amount"] == pytest.approx(float(now)) and row["net_amount_prior"] == pytest.approx(float(before))


# ── the period before, of the same shape ─────────────────────────────────────


def test_this_month_so_far_is_set_against_the_same_days_of_last_month(retail):
    chat = Chat(retail, _query(intent="compare", measures=["net_amount"],
                               time={"window": {"kind": "to_date", "unit": "month"},
                                     "compare": {"kind": "window",
                                                 "window": {"kind": "rolling", "unit": "day", "n": 15}}}))
    row = _rows(chat.ask("sales this month so far vs the same days last month"))[0]
    now, before = _totals(retail, ("2026-06-01", "2026-06-15"), ("2026-05-01", "2026-05-15"))
    assert row["net_amount"] == pytest.approx(float(now)) and row["net_amount_prior"] == pytest.approx(float(before))


def test_the_last_twelve_months_are_set_against_the_twelve_before(retail):
    twelve = {"kind": "last", "unit": "month", "n": 12}
    chat = Chat(retail, _query(intent="compare", measures=["net_amount"],
                               time={"window": twelve, "compare": {"kind": "window", "window": twelve}}))
    row = _rows(chat.ask("Are we growing? Sales over the last 12 months against the 12 before"))[0]
    now, before = _totals(retail, ("2025-06-01", "2026-06-01"), ("2024-06-01", "2025-06-01"))
    assert row["net_amount"] == pytest.approx(float(now)) and row["net_amount_prior"] == pytest.approx(float(before))


# ── a run rate ─────────────────────────────────────────────────────────────────


def test_a_run_rate_is_a_year_at_the_pace_of_the_last_three_whole_months(retail):
    chat = Chat(retail, ASKED_BACK, _query(intent="value", measures=["net_amount"]))
    got = chat.ask("What is our sales run rate?")
    want = _sql(retail, f"SELECT SUM(net_amount) * 4 FROM ({LINES}) l WHERE {_in(('2026-03-01', '2026-06-01'))} "
                        f"AND {LIVE}")[0][0]
    assert float(got["kpi"]["value"]) == pytest.approx(float(want))
    assert any("last 3 months" in b["text"] for b in got["answer"]["badges"])


def test_a_run_rate_planned_by_the_ai_as_its_own_figure_is_worked_out_the_same_way(retail):
    chat = Chat(retail, _query(intent="value", figures=[{"name": "Sales run rate",
                                                          "formula": "SUM([Order line · Net amount]) / 3"}],
                               derived=[{"name": "Run rate", "op": "product", "measures": ["net_amount", "net_amount"],
                                         "scale": 12}],
                               time={"window": {"kind": "rolling", "unit": "month", "n": 1}}))
    got = chat.ask("What is our sales run rate?")
    want = _sql(retail, f"SELECT SUM(net_amount) * 4 FROM ({LINES}) l WHERE {_in(('2026-03-01', '2026-06-01'))} "
                        f"AND {LIVE}")[0][0]
    assert float(got["kpi"]["value"]) == pytest.approx(float(want))


# ── never a target the question does not name ────────────────────────────────


def test_a_target_is_never_the_figure_of_a_question_that_names_none(retail):
    chat = Chat(retail, _query(intent="compare", measures=["target_amount"], time=LAST_YEAR),
                _query(intent="compare", measures=["net_amount"]))
    got = chat.ask("How is net amount doing vs last year?")
    assert len(chat.prompts) == 2 and "target_amount" in chat.prompts[1] and "what was planned" in chat.prompts[1]
    _same(_months_shown(got), _by_month(retail))


def test_a_target_the_question_names_is_kept():
    _, model = learn(domains.build("retail"), "descriptive")
    plan = Plan.model_validate(_query(intent="value", measures=["target_amount"]))
    assert asks.target_unasked(plan, model, "What is our sales target for 2026?") == []
    assert asks.target_unasked(plan, model, "What are our sales for 2026?")


# ── what is no business ask is left as the AI read it ───────────────────────


@pytest.mark.parametrize("asked", ["compare North and South revenue", "How is revenue trending by month?",
                                   "Q2 vs Q1", "sales last year", "Sales last month vs the month before",
                                   "how are North and South doing", "net sales in 2025 vs 2024",
                                   "what is the trend of sales over the last 6 months"])
def test_what_is_no_business_ask_is_left_alone(asked):
    assert asks.ask_of(asked) is None


def test_growth_by_month_keeps_the_ais_period_over_period():
    _, model = learn(domains.build("retail"), "descriptive")
    plan = Plan.model_validate(_query(intent="trend", measures=["net_amount"],
                                      time={"grain": "month", "compare": {"kind": "previous_period"},
                                            "window": {"kind": "this", "unit": "year"}}))
    assert asks.settled(plan, model, "revenue growth by month") == plan


# ── found on the books with gpt-4o ─────────────────────────────────────────────


def test_this_year_and_last_year_after_who_grew_still_compares(retail):
    # Asked right after "which stores grew the most vs last year?", gpt-4o kept the comparison and the follow-up rule
    # dropped it (the question names its own period and no "vs"): "what they sold last year" is the comparison.
    grew = _query(intent="compare", measures=["net_amount"], group_by=["store.name"], time=LAST_YEAR,
                  sort=[{"by": "change", "desc": True}])
    top = _query(intent="compare", measures=["net_amount"], group_by=["store.name"], limit=5, follow_up="refine",
                 sort=[{"by": "net_amount", "desc": True}],
                 time={"window": {"kind": "to_date", "unit": "year"}, "compare": {"kind": "same_period_last_year"}})
    chat = Chat(retail, grew, top)
    chat.ask("Which stores grew the most vs last year?")
    got = chat.ask("top 5 stores this year and what they sold last year")
    want = _sql(retail, f"""SELECT store_name, SUM(CASE WHEN {_in(SO_FAR)} THEN net_amount END) AS now_,
        SUM(CASE WHEN {_in(SO_FAR_LY)} THEN net_amount END) FROM ({LINES}) l
        WHERE ({_in(SO_FAR)} OR {_in(SO_FAR_LY)}) AND {LIVE} GROUP BY 1 ORDER BY now_ DESC LIMIT 5""")
    assert [(r["store_name"], round(r["net_amount_prior"], 2)) for r in _rows(got)] == \
        [(w[0], round(float(w[2]), 2)) for w in want]


def test_who_grew_naming_no_figure_is_about_the_money_with_the_ais_figure_a_click_away(retail):
    # "Which locations grew the most vs last year?" was counted in batches.
    chat = Chat(retail, _query(intent="compare", measures=["number_of_orders"], group_by=["store.name"],
                               time=LAST_YEAR, sort=[{"by": "change", "desc": True}]))
    got = chat.ask("Which stores grew the most vs last year?")
    assert "net_amount" in _rows(got)[0]
    assert "Read as Net amount" in [b["text"] for b in got["answer"]["badges"]]
    assert "Number of orders instead" in _chips(got)


def test_who_grew_naming_its_figure_keeps_it():
    _, model = learn(domains.build("retail"), "descriptive")
    plan = Plan.model_validate(_query(intent="compare", measures=["number_of_orders"], group_by=["store.name"],
                                      time=LAST_YEAR))
    main = [m for m in model.measures.values() if m.slug == "net_amount"]
    assert asks.main_figure(plan, model, "Which stores grew their orders the most vs last year?", main) is None
    assert asks.main_figure(plan, model, "Which stores grew the most vs last year?", main) is not None


def test_the_top_n_this_year_with_last_years_beside_them_are_ranked_by_this_years_figure(retail):
    # "top 3 partners by bookings this year and their bookings last year" came back ranked by the change.
    chat = Chat(retail, _query(intent="compare", measures=["net_amount"], group_by=["store.name"], limit=3,
                               sort=[{"by": "change", "desc": True}],
                               time={"window": {"kind": "this", "unit": "year"},
                                     "compare": {"kind": "same_period_last_year"}}))
    got = chat.ask("top 3 stores by net amount this year and their net amount last year")
    want = _sql(retail, f"""SELECT store_name, SUM(net_amount) AS now_ FROM ({LINES}) l WHERE {_in(SO_FAR)} AND {LIVE}
        GROUP BY 1 ORDER BY now_ DESC LIMIT 3""")
    assert [r["store_name"] for r in _rows(got)] == [w[0] for w in want]


def test_by_the_figure_it_counts_is_no_split(retail):
    # "loans funded this month so far vs the same days last month, by principal" was split loan by loan.
    chat = Chat(retail, _query(intent="compare", measures=["net_amount"], group_by=["store.name"],
                               time={"window": {"kind": "to_date", "unit": "month"},
                                     "compare": {"kind": "previous_period"}}))
    rows = _rows(chat.ask("sales this month so far vs the same days last month, by net amount"))
    now, before = _totals(retail, ("2026-06-01", "2026-06-15"), ("2026-05-01", "2026-05-15"))
    assert len(rows) == 1
    assert rows[0]["net_amount"] == pytest.approx(float(now)) and rows[0]["net_amount_prior"] == pytest.approx(float(before))


def test_fields_the_reader_joins_with_minus_are_one_figure_worked_out_on_each_row(retail):
    # "line_amount plus travel_fee from booking lines, by service_type, for 2025" was the two totals side by side.
    chat = Chat(retail, _query(intent="breakdown", measures=["gross_amount", "discount_amount"],
                               group_by=["store.name"],
                               time={"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}))
    got = chat.ask("gross_amount minus discount_amount from order lines, by store, for 2025")
    want = dict(_sql(retail, f"""SELECT store_name, SUM(gross_amount - discount_amount) FROM ({LINES}) l
        WHERE {_in(('2025-01-01', '2026-01-01'))} AND {LIVE} GROUP BY 1"""))
    rows = _rows(got)
    assert len(rows) == len(want)
    for r in rows:
        value = next(v for k, v in r.items() if k != "store_name" and isinstance(v, (int, float)))
        assert value == pytest.approx(float(want[r["store_name"]]))


def test_a_things_volume_is_counted_on_its_own_rows(retail):
    # "Which branches grew their loan volume the most vs last year?" was counted on the transactions' amount.
    chat = Chat(retail, _query(intent="compare", measures=["net_amount"], time=LAST_YEAR),
                _query(intent="compare", measures=["refund_amount"]))
    got = chat.ask("How is return volume doing vs last year?")
    assert len(chat.prompts) == 2 and "refund_amount" in chat.prompts[1] and "Return" in chat.prompts[1]
    assert "refund_amount" in _rows(got)[0]


def test_a_volume_on_its_own_rows_is_left_alone():
    _, model = learn(domains.build("retail"), "descriptive")
    plan = Plan.model_validate(_query(intent="compare", measures=["refund_amount"], time=LAST_YEAR))
    assert asks.volume_elsewhere(plan, model, "How is return volume doing vs last year?") == []
    # No business ask: the AI's figure stands.
    other = Plan.model_validate(_query(intent="value", measures=["net_amount"]))
    assert asks.volume_elsewhere(other, model, "return volume in 2025") == []


def test_the_figure_the_ai_chose_once_told_is_said_with_the_others_it_asked_between(retail):
    # Told that an ask is answered at once, gpt-4o picks a figure itself: the answer still says which ("Read as Net
    # amount") and offers the figures it had asked between, with no AI.
    chat = Chat(retail, ASKED_BACK, _query(intent="compare", measures=["net_amount"]))
    got = chat.ask("What is my revenue comparison?")
    assert "Read as Net amount" in [b["text"] for b in got["answer"]["badges"]]
    assert {"Gross amount instead", "Discount amount instead", "Cost amount instead"} <= set(_chips(got))
    _same(_months_shown(chat.ask("Gross amount instead"), "gross_amount"), _by_month(retail, "gross_amount"))
    assert len(chat.prompts) == 2


def test_a_figure_multiplying_the_totals_of_fields_dictated_row_by_row_is_put_right(retail):
    # "sum ordered_qty times unit_price where status is not X, by buyer, for 2025" came back as 144 billion: the AI
    # wrote the year's quantity times the sum of its prices.
    chat = Chat(retail, _query(intent="breakdown", group_by=["store.name"],
                               figures=[{"name": "Total value", "formula": "SUM([Order line · Quantity]) * "
                                                                          "SUM([Order line · Unit price])"}],
                               time={"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}))
    got = chat.ask("sum quantity times unit_price from order lines, by store, for 2025")
    want = dict(_sql(retail, f"""SELECT store_name, SUM(quantity * unit_price) FROM ({LINES}) l
        WHERE {_in(('2025-01-01', '2026-01-01'))} AND {LIVE} GROUP BY 1"""))
    rows = _rows(got)
    assert len(rows) == len(want)
    for r in rows:
        assert r["total_value"] == pytest.approx(float(want[r["store_name"]]))


def test_a_difference_of_the_fields_dictated_planned_as_two_totals_is_one_figure_worked_out_on_each_row(retail):
    # "received_qty minus rejected_qty from receipt lines, by supplier" was planned as the two totals and their
    # difference, with the unit of measure as a second split.
    chat = Chat(retail, _query(intent="breakdown", measures=["gross_amount", "discount_amount"],
                               derived=[{"name": "Gross less discount", "op": "difference",
                                         "measures": ["gross_amount", "discount_amount"]}],
                               group_by=["store.name"], sort=[{"by": "Gross less discount", "desc": True}],
                               time={"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}))
    got = chat.ask("gross_amount minus discount_amount from order lines, by store, for 2025")
    want = dict(_sql(retail, f"""SELECT store_name, SUM(gross_amount - discount_amount) FROM ({LINES}) l
        WHERE {_in(('2025-01-01', '2026-01-01'))} AND {LIVE} GROUP BY 1"""))
    rows = _rows(got)
    assert len(rows) == len(want)
    for r in rows:
        numbers = [v for k, v in r.items() if k != "store_name" and isinstance(v, (int, float))]
        assert numbers == [pytest.approx(float(want[r["store_name"]]))]
    values = [next(v for k, v in r.items() if k != "store_name") for r in rows]
    assert values == sorted(values, reverse=True)


def test_a_derived_figure_of_other_measures_than_the_fields_named_is_left_alone():
    _, model = learn(domains.build("retail"), "descriptive")
    plan = Plan.model_validate(_query(intent="breakdown", measures=["net_amount", "cost_amount"],
                                      derived=[{"name": "Margin", "op": "difference",
                                                "measures": ["net_amount", "cost_amount"]}]))
    assert asks.fields_combined(plan, model, "gross_amount minus discount_amount by store") == plan


def test_a_field_grouped_by_its_own_name_is_the_one_grouping_it_can_be(retail):
    # gpt-4o grouped by "service_type" where the catalog has booking_line.service_type: refused twice before.
    chat = Chat(retail, _query(intent="breakdown", measures=["net_amount"], group_by=["segment"],
                               time={"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}))
    got = chat.ask("net amount by segment for 2025")
    want = dict(_sql(retail, f"""SELECT c.segment, SUM(l.net_amount) FROM ({LINES}) l
        JOIN customers c ON c.customer_id = l.customer_id
        WHERE {_in(('2025-01-01', '2026-01-01'))} AND {LIVE} GROUP BY 1"""))
    rows = _rows(got)
    assert len(chat.prompts) == 1 and len(rows) == len(want)
    for r in rows:
        assert r["net_amount"] == pytest.approx(float(want[r["customer_segment"]]))


def test_a_bare_name_two_groupings_end_with_is_left_for_the_check():
    from core2.plan.planner import fill_in
    _, model = learn(domains.build("retail"), "descriptive")
    plan = Plan.model_validate(_query(intent="breakdown", measures=["net_amount"], group_by=["name"]))
    assert fill_in(plan, model, None, False).group_by == ["name"]


def test_a_things_volume_is_counted_on_the_rows_it_names_never_on_rows_with_it_in_their_name():
    # "Which branches grew their loan volume the most vs last year?" was the loan applications' requested amount.
    _, model = learn(domains.build("bank"), "descriptive")
    asked = "Which branches grew their loan volume the most vs last year?"
    applied = Plan.model_validate(_query(intent="compare", measures=["requested_amount"], group_by=["branch.name"],
                                         time=LAST_YEAR))
    problems = asks.volume_elsewhere(applied, model, asked)
    assert problems and "principal_amount" in problems[0] and "requested_amount (" not in problems[0].split(":")[-1]
    lent = applied.model_copy(update={"measures": ["principal_amount"]})
    assert asks.volume_elsewhere(lent, model, asked) == []


def test_a_target_unasked_points_at_the_figures_of_the_rows_it_names():
    _, model = learn(domains.build("bank"), "descriptive")
    plan = Plan.model_validate(_query(intent="compare", measures=["loan_volume_target"], time=LAST_YEAR))
    problems = asks.target_unasked(plan, model, "How is loan volume doing vs last year?")
    assert problems and "principal_amount" in problems[0] and "requested_amount" not in problems[0]


# ── the other readings of a comparison, one click away ────────────────────────


TWELVE_NOW, TWELVE_BEFORE = ("2025-06-01", "2026-06-01"), ("2024-06-01", "2025-06-01")


def test_a_comparison_offers_its_other_readings_and_each_runs_with_no_ai(retail):
    # Decided with the reader: a comparison is this year so far against the same days last year, month by month; the
    # last 12 months against the 12 before, and last month against the month before, are a click away.
    chat = Chat(retail, _query(intent="compare", measures=["net_amount"]))
    got = chat.ask("What is my revenue comparison?")
    chips = {s["label"]: s for s in got.get("follow_up_suggestions") or []}
    assert {"Last 12 months against the 12 before", "Last month against the month before"} <= set(chips)
    assert chips["Last month against the month before"]["label_fr"] == "Le mois dernier face au mois d'avant"

    twelve = chat.ask("Last 12 months against the 12 before")
    want = _sql(retail, f"""SELECT month(day), SUM(CASE WHEN {_in(TWELVE_NOW)} THEN net_amount END),
        SUM(CASE WHEN {_in(TWELVE_BEFORE)} THEN net_amount END) FROM ({LINES}) l
        WHERE ({_in(TWELVE_NOW)} OR {_in(TWELVE_BEFORE)}) AND {LIVE} GROUP BY 1""")
    _same(sorted(_months_shown(twelve)), sorted(want))

    # From there, the other two readings; French words run the same plan.
    assert {"This year so far against the same days last year",
            "Last month against the month before"} <= set(_chips(twelve))
    month = chat.ask(chips["Last month against the month before"]["question_fr"])
    now, before = _totals(retail, ("2026-05-01", "2026-06-01"), ("2026-04-01", "2026-05-01"))
    row = _rows(month)[0]
    assert len(_rows(month)) == 1
    assert row["net_amount"] == pytest.approx(float(now)) and row["net_amount_prior"] == pytest.approx(float(before))
    assert len(chat.prompts) == 1, "a click asked the AI"
    _same(_months_shown(chat.ask("This year so far against the same days last year")), _by_month(retail))
    assert len(chat.prompts) == 1, "a click asked the AI"


def test_a_comparison_with_its_own_period_or_members_offers_no_other_reading(retail):
    chat = Chat(retail, _query(intent="compare", measures=["net_amount"], group_by=["store.name"]))
    labels = set(_chips(chat.ask("Which stores grew the most vs last year?")))
    assert not labels & {"Last 12 months against the 12 before", "Last month against the month before"}
    _, model = learn(domains.build("retail"), "descriptive")
    plan = Plan.model_validate(_query(intent="compare", measures=["net_amount"],
                                      time={"window": {"kind": "this", "unit": "quarter"},
                                            "compare": {"kind": "previous_period"}}))
    assert asks.other_readings(plan, model, "compare this quarter's revenue with last quarter") == []


@pytest.mark.parametrize("asked", ["How does our YTD net sales compare with the same time last year?",
                                   "Compare sales so far this year with last year",
                                   "Quelle est la comparaison des ventes depuis le début de l'année avec l'an dernier ?"])
def test_the_year_to_date_named_as_such_is_one_total_against_the_same_days(asked):
    # The catalogue's "year to date vs last": one total; "what is my revenue comparison?" is month by month.
    ask = asks.ask_of(asked)
    assert ask is not None and ask.unit == "year" and not ask.by_month
    assert asks.ask_of("What is my revenue comparison?").by_month


# ── the same question asked again after no answer came back ───────────────────


Y25 = {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}
DICTATED = _query(intent="breakdown", group_by=["region.name"], time=Y25,
                  figures=[{"name": "Net amount minus discount amount",
                            "formula": "SUM([Order line · Net amount] - [Order line · Discount amount])"}],
                  filters=[{"field": "order_line.status_code", "op": "ne", "values": ["C"]}])
MARGIN = _query(intent="breakdown", group_by=["category.name"], time=Y25,
                figures=[{"name": "Margin", "format": "percent",
                          "formula": "(SUM([Order line · Net amount]) - SUM([Order line · Cost amount])) / "
                                     "SUM([Order line · Net amount])"}])


class _Limited(Exception):
    def __str__(self) -> str:
        return "Error code: 429 - rate_limit_exceeded"


def test_a_question_asked_again_after_the_rate_limit_is_read_as_the_first_time(retail):
    # Retail BR-03.2 with gpt-4o: "calculate margin as (net_amount - cost_amount) / net_amount by category in 2025",
    # refused once by the per-minute limit and asked again, kept the figure and the condition of the answer above
    # ("net_amount minus discount_amount ... only where status_code is not C"): 0.4% margin beside $841.0K.
    chat = Chat(retail, DICTATED, MARGIN)
    chat.ask("use net_amount minus discount_amount from order lines, only where status_code is not C, by region "
             "for 2025")
    answers, planned = chat.answers, chat.services.complete

    def limited(system: str, tail: str) -> str:
        raise _Limited()

    chat.services = replace(chat.services, complete=limited)
    with pytest.raises(_Limited):
        chat.ask("calculate margin as (net_amount - cost_amount) / net_amount by category in 2025")
    chat.services = replace(chat.services, complete=planned)
    got = chat.ask("calculate margin as (net_amount - cost_amount) / net_amount by category in 2025")
    assert [f["name"] for f in got["plan"]["figures"]] == ["Margin"]
    assert not got["plan"].get("filters")
    want = dict(_sql(retail, f"""SELECT k.category_name, (SUM(l.net_amount) - SUM(l.cost_amount)) / SUM(l.net_amount)
        FROM ({LINES}) l JOIN products p2 ON p2.product_id = l.product_id JOIN categories k ON k.category_id =
        p2.category_id WHERE {_in(('2025-01-01', '2026-01-01'))} AND {LIVE} GROUP BY 1"""))
    assert len(want) >= 5
    for r in _rows(got):
        if r["category_name"] in want:
            assert r["margin"] == pytest.approx(float(want[r["category_name"]]), abs=1e-5)


def test_another_question_after_one_not_answered_still_goes_on_with_it(retail):
    # The thread rule for an unanswered turn is kept: "and 2025?" right after it means its question.
    chat = Chat(retail, DICTATED)
    chat.ask("use net_amount minus discount_amount from order lines, only where status_code is not C, by region "
             "for 2025")
    planned = chat.services.complete

    def limited(system: str, tail: str) -> str:
        raise _Limited()

    chat.services = replace(chat.services, complete=limited)
    with pytest.raises(_Limited):
        chat.ask("net amount by category in 2024")
    assert chat.session.turns[-1].unanswered
    chat.services = replace(chat.services, complete=planned)
    chat.answers.append(_query(intent="breakdown", measures=["net_amount"], group_by=["category.name"], time=Y25))
    chat.ask("and 2025?")
    assert "net amount by category in 2024" in chat.prompts[-1]
