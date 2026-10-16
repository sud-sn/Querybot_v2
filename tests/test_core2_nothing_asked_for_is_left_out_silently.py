"""Every part of a request is given, or the answer says what it left out: never a silent drop.

An analyst's request names several things at once ("net sales, cost, gross margin and margin % by region"; "the
customers who bought Cola Large"). The answer gave some and dropped the rest without a word: a gross margin the
plan worked out only inside the margin % was never shown, and a product named without quotes narrowed nothing
while the summary spoke of it. Now the AI lists the request's parts before it plans (Plan.asked), code checks
each against the plan and the answer's columns (core2/agent/parts.py), a part left out is asked of the AI once
more, and what the answer still lacks is said on it with a button that asks for it. A member named in full
without quotes narrows the answer, with one click to take it off, and a summary never names what the answer
does not give.

Invented retail data; every figure is checked against SQL written by hand.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.answer import summary
from core2.plan.values import build_index
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn
from evals.core2.framework import materialize

TODAY = dt.date(2026, 6, 15)
Y2025 = {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}
BY_REGION = """
    SELECT r.region_name, SUM(o.net_amount) AS net, SUM(o.cost_amount) AS cost
    FROM order_lines o JOIN stores s ON s.store_id = o.store_id JOIN regions r ON r.region_id = s.region_id
    WHERE o.order_date_key BETWEEN 20250101 AND 20251231
    GROUP BY r.region_name"""


@pytest.fixture(scope="module")
def retail():
    domain = domains.build("retail")
    built, model = learn(domain, "descriptive")
    warehouse = DuckDBWarehouse(materialize(domain, "descriptive").con)

    def fetch(slug):
        column = model.columns[model.attributes[slug].column]
        return [r[0] for r in warehouse.query(f'SELECT DISTINCT "{column.name}" FROM '
                                              f'"{model.tables[column.table].name}"').rows]

    return model, warehouse, build_index(model, fetch)


class Recorded:
    """The AI, answering with ``plans`` in turn; it keeps what it was sent."""

    def __init__(self, *plans):
        self.plans = [json.dumps({"kind": "query", **p}) for p in plans]
        self.sent: list[str] = []

    def __call__(self, stable, tail):
        self.sent.append(tail)
        return self.plans.pop(0)


def _ask(retail, ai, question):
    model, warehouse, index = retail
    services = Services(model=model, warehouse=warehouse, complete=ai, index=index, today=TODAY)
    return answer_question(question, services, Session())


def _hand(retail, sql):
    return retail[1].query(sql).rows


def _by_label(payload, label_column):
    return {str(r[label_column]): r for r in payload["data"]["rows"]}


def _column(payload, *words):
    """The answer's column whose heading carries every one of ``words``."""
    data = payload["data"]
    labels = data.get("header_labels") or {}
    for h in data["headers"]:
        said = f"{h} {labels.get(h, '')}".lower().replace("_", " ")
        if all(w in said for w in words):
            return h
    raise AssertionError(f"no column for {words}: {data['headers']} {labels}")


def test_a_figure_worked_out_inside_another_is_shown_when_it_is_asked_for(retail):
    plan = {"intent": "breakdown", "measures": ["net_amount", "cost_amount"], "group_by": ["region.name"],
            "time": Y2025,
            "derived": [{"name": "gross_margin", "op": "difference", "measures": ["net_amount", "cost_amount"]},
                        {"name": "margin_pct", "op": "ratio", "measures": ["gross_margin", "net_amount"],
                         "scale": 100}],
            "asked": [{"words": "net sales", "kind": "figure", "plan": "net_amount"},
                      {"words": "cost", "kind": "figure", "plan": "cost_amount"},
                      {"words": "gross margin", "kind": "figure", "plan": "gross_margin"},
                      {"words": "margin %", "kind": "figure", "plan": "margin_pct"},
                      {"words": "region", "kind": "field", "plan": "region.name"}]}
    payload = _ask(retail, Recorded(plan, plan), "net sales, cost, gross margin and margin % by region in 2025")
    assert not payload.get("parts_left_out"), payload.get("parts_left_out")
    region, margin, pct = _column(payload, "region"), _column(payload, "gross", "margin"), _column(payload, "pct")
    shown = _by_label(payload, region)
    hand = _hand(retail, BY_REGION)
    assert len(shown) == len(hand) == 5
    for name, net, cost in hand:
        assert round(float(shown[name][margin]), 2) == round(float(net - cost), 2), name
        assert round(float(shown[name][pct]), 2) == round(float((net - cost) / net * 100), 2), name


def test_a_part_the_plan_cannot_give_is_asked_once_more_then_said_with_a_button(retail):
    plan = {"intent": "breakdown", "measures": ["net_amount"], "group_by": ["region.name"], "time": Y2025,
            "asked": [{"words": "net sales", "kind": "figure", "plan": "net_amount"},
                      {"words": "store visits", "kind": "figure", "plan": None},
                      {"words": "region", "kind": "field", "plan": "region.name"}]}
    ai = Recorded(plan, plan)
    payload = _ask(retail, ai, "net sales and store visits by region in 2025")
    assert len(ai.sent) == 2, "asked once more, and only once"
    assert "leaves out parts of the request" in ai.sent[1] and '"store visits"' in ai.sent[1]
    left = payload["parts_left_out"]
    assert [(p["words"], p["kind"], p["why"]) for p in left] == [("store visits", "figure", "no_plan")]
    assert left[0]["ask"] == "Add store visits" and left[0]["ask_fr"] == "Ajoute store visits"
    region, net = _column(payload, "region"), _column(payload, "net")
    shown = _by_label(payload, region)
    for name, hand_net, _ in _hand(retail, BY_REGION):
        assert round(float(shown[name][net]), 2) == round(float(hand_net), 2), name
    assert payload["trust"]["looked_again"]["kept"] == "first"


def test_a_second_reading_that_gives_the_part_is_kept_and_says_what_the_first_left_out(retail):
    first = {"intent": "breakdown", "measures": ["net_amount"], "group_by": ["region.name"], "time": Y2025,
             "asked": [{"words": "net sales", "kind": "figure", "plan": "net_amount"},
                       {"words": "gross margin", "kind": "figure", "plan": None},
                       {"words": "region", "kind": "field", "plan": "region.name"}]}
    second = {"intent": "breakdown", "measures": ["net_amount", "cost_amount"], "group_by": ["region.name"],
              "time": Y2025,
              "derived": [{"name": "gross_margin", "op": "difference", "measures": ["net_amount", "cost_amount"]}],
              "asked": [{"words": "net sales", "kind": "figure", "plan": "net_amount"},
                        {"words": "gross margin", "kind": "figure", "plan": "gross_margin"},
                        {"words": "region", "kind": "field", "plan": "region.name"}]}
    payload = _ask(retail, Recorded(first, second), "net sales and gross margin by region in 2025")
    assert not payload.get("parts_left_out")
    assert payload["reply"]["en"].startswith("My first reading left out gross margin, so I read your question again.")
    assert payload["reply"]["fr"].startswith("Ma première lecture laissait de côté « gross margin »")
    region, margin = _column(payload, "region"), _column(payload, "gross", "margin")
    shown = _by_label(payload, region)
    for name, net, cost in _hand(retail, BY_REGION):
        assert round(float(shown[name][margin]), 2) == round(float(net - cost), 2), name


def test_a_product_named_in_full_without_quotes_narrows_the_answer(retail):
    plan = {"intent": "breakdown", "measures": ["net_amount"], "group_by": ["store.name"], "time": Y2025,
            "filters": [{"field": "product.name", "op": "eq", "values": ["Cola Large"]}]}
    ai = Recorded(plan)
    payload = _ask(retail, ai, "net sales of Cola Large by store in 2025")
    assert 'product.name = "Cola Large"' in ai.sent[0], "offered to the AI as a name to filter on"
    assert payload["plan"]["filters"] == [{"field": "product.name", "op": "eq", "values": ["Cola Large"]}]
    store, net = _column(payload, "store"), _column(payload, "net")
    shown = _by_label(payload, store)
    hand = _hand(retail, """
        SELECT s.store_name, SUM(o.net_amount) FROM order_lines o JOIN stores s ON s.store_id = o.store_id
        JOIN products p ON p.product_id = o.product_id
        WHERE p.product_name = 'Cola Large' AND o.order_date_key BETWEEN 20250101 AND 20251231
        GROUP BY s.store_name""")
    assert len(shown) == len(hand) > 1
    for name, hand_net in hand:
        assert round(float(shown[name][net]), 2) == round(float(hand_net), 2), name
    assert any(c["label"] == 'Not only "Cola Large"' for c in payload["follow_up_suggestions"])


def test_a_name_two_fields_share_narrows_by_the_thing_it_names(retail):
    # "Why did Ontario revenue drop?": Ontario is a region, and also the name of a sales team. The region is the
    # thing the name names; the team is only called after it. Here a customer segment carries a store's name.
    model, warehouse, _ = retail
    store = warehouse.query("SELECT store_name FROM stores ORDER BY store_id LIMIT 1").rows[0][0]

    def fetch(slug):
        column = model.columns[model.attributes[slug].column]
        return [r[0] for r in warehouse.query(f'SELECT DISTINCT "{column.name}" FROM '
                                              f'"{model.tables[column.table].name}"').rows]

    shared = build_index(model, fetch)
    shared.add("customer.segment", [store])
    assert len(shared.match(f"at {store}")) == 2
    plan = {"intent": "value", "measures": ["net_amount"], "time": Y2025,
            "filters": [{"field": "store.name", "op": "eq", "values": [store]}]}
    ai = Recorded(plan)
    services = Services(model=model, warehouse=warehouse, complete=ai, index=shared, today=TODAY)
    payload = answer_question(f"net sales at {store} in 2025", services, Session())
    assert f'store.name = "{store}"' in ai.sent[0] and f'customer.segment = "{store}"' not in ai.sent[0]
    hand = _hand(retail, f"""SELECT SUM(o.net_amount) FROM order_lines o JOIN stores s ON s.store_id = o.store_id
                             WHERE s.store_name = '{store}' AND o.order_date_key BETWEEN 20250101 AND 20251231""")
    assert round(float(payload["kpi"]["value"]), 2) == round(float(hand[0][0]), 2)
    assert not any("Not narrowed" in str(c) for c in payload["trust"].get("date_context") or [])


def test_a_name_the_plan_leaves_unheld_is_asked_for_then_said_with_the_question_quoting_it(retail):
    whole = {"intent": "value", "measures": ["net_amount"], "time": Y2025}
    ai = Recorded(whole, whole)
    payload = _ask(retail, ai, "net sales of Cola Large in 2025")
    assert len(ai.sent) == 2
    left = payload["parts_left_out"]
    assert [(p["words"], p["kind"], p["why"]) for p in left] == [("Cola Large", "condition", "not_applied")]
    assert left[0]["ask"] == 'net sales of "Cola Large" in 2025'
    hand_total = _hand(retail, "SELECT SUM(net_amount) FROM order_lines "
                               "WHERE order_date_key BETWEEN 20250101 AND 20251231")[0][0]
    assert round(payload["kpi"]["value"], 2) == round(float(hand_total), 2), "the figure is for every product"
    given = summary.material(payload, "net sales of Cola Large in 2025")
    assert summary.checked("Cola Large brought in most of the net sales in 2025.", payload, given) is None
    assert summary.checked("Net sales in 2025 came from every product.", payload, given) is not None


def test_lower_case_words_and_everyday_words_still_narrow_nothing(retail):
    model, _, index = retail
    from core2.service import _own_names

    own = _own_names(model)
    assert index.written_names("net sales in the north in 2025", not_names=own) == []
    assert index.written_names("Net Amount by store in 2025", not_names=own) == []
    months = type(index)(names={"september": [("calendar.month_name", "September")],
                                "may": [("calendar.month_name", "May")]})
    assert months.written_names("Net sales in September and May 2025", not_names=own) == []
    assert [m.value for m in index.written_names("net sales at the Airport Store", not_names=own)] == \
        ["Airport Store"]


def test_a_summary_never_names_a_figure_the_answer_left_out(retail):
    payload = {"engine": "core2", "answer": {"headline": "Net amount in 2025: North leads with $10."},
               "data": {"headers": ["region", "net_amount"], "rows": [{"region": "North", "net_amount": 10.0}]},
               "parts_left_out": [{"words": "gross margin", "kind": "figure", "why": "not_shown"}]}
    given = summary.material(payload, "net sales and gross margin by region")
    assert "Not in this answer (never mention it): gross margin" in given
    assert summary.checked("North leads on net sales and gross margin with $10.", payload, given) is None
    assert summary.checked("North leads net sales with $10.", payload, given) == "North leads net sales with $10."


def test_a_summary_never_says_rows_are_missing_when_the_reader_sees_every_row():
    # 31 customers listed: the AI is given the first 25, the reader sees all 31. It once wrote "since six rows
    # are not shown, some details may be missing", which the answer does not bear out.
    rows = [{"customer": f"Customer {i}", "net_amount": float(i)} for i in range(31)]
    payload = {"engine": "core2", "answer": {"headline": "31 customers bought Dish Soap."},
               "data": {"headers": ["customer", "net_amount"], "rows": rows, "total_rows": 31}}
    given = summary.material(payload, "customers who bought Dish Soap")
    assert "Customer 24" in given and "Customer 25" not in given
    assert "the first 25 of 31" in given and "not shown" not in given
    assert "never say rows are missing" in " ".join(summary.SYSTEM.casefold().split())
    for said in ("31 customers bought Dish Soap. Since six rows are not shown, some details may be missing.",
                 "31 customers bought Dish Soap, but only a portion of the rows are visible.",
                 "31 customers bought Dish Soap; the remaining rows are hidden from view."):
        assert summary.checked(said, payload, given) is None, said
    assert summary.checked("31 customers bought Dish Soap, most of them in Calgary.", payload, given)


# ── what the check reads as given (core2/agent/parts.py), on plans and answers as the AI and the engine write them ──

def _missing(retail, plan, headers, question="q", before=None):
    from core2.agent import parts as P
    from core2.plan.ir import Plan

    payload = {"data": {"headers": headers, "rows": [{h: 1 for h in headers}]}}
    return [(g.words, g.why) for g in P.missing(question, Plan.model_validate({"kind": "query", **plan}), payload,
                                                retail[0], before=before)]


def test_a_date_asked_as_a_column_is_given_only_when_the_answer_is_grouped_by_it(retail):
    asked = [{"words": "order number", "kind": "field", "plan": "order_line.order_number"},
             {"words": "order date", "kind": "field", "plan": "time.date"}]
    only_march = {"asked": asked, "intent": "list", "measures": ["net_amount"], "group_by": ["order_line.order_number"],
                  "time": {"window": {"kind": "between", "start": "2026-03-01", "end": "2026-03-31"}}}
    assert _missing(retail, only_march, ["order_line_order_number", "net_amount"]) == [("order date", "not_shown")]
    by_day = {**only_march, "time": {**only_march["time"], "grain": "day"}}
    assert _missing(retail, by_day, ["period", "order_number", "net_amount"]) == [], "its column's own name"


def test_a_part_given_as_a_section_name_is_checked_against_that_section(retail):
    plan = {"asked": [{"words": "good units after scrap", "kind": "figure", "plan": "derived"},
                      {"words": "only the big orders", "kind": "condition", "plan": "filters"}],
            "intent": "breakdown", "measures": ["net_amount", "cost_amount"], "group_by": ["region.name"],
            "derived": [{"name": "margin", "op": "difference", "measures": ["net_amount", "cost_amount"]},
                        {"name": "margin_pct", "op": "ratio", "measures": ["margin", "net_amount"], "scale": 100}]}
    assert _missing(retail, plan, ["region_name", "margin_pct"]) == [
        ("good units after scrap", "not_shown"), ("only the big orders", "not_in_plan")]


def test_a_measure_worked_out_only_inside_another_figure_is_not_given(retail):
    plan = {"asked": [{"words": "the number of customers", "kind": "figure", "plan": "number_of_customers"}],
            "intent": "breakdown", "group_by": ["region.name"],
            "derived": [{"name": "spend_per_customer", "op": "ratio", "measures": ["net_amount", "number_of_customers"]}]}
    assert _missing(retail, plan, ["region_name", "spend_per_customer"]) == [("the number of customers", "not_shown")]


def test_a_follow_up_that_adds_keeps_what_the_answer_showed(retail):
    from core2.plan.ir import Plan

    before = Plan.model_validate({"kind": "query", "intent": "trend", "measures": ["net_amount", "number_of_orders"],
                                  "time": {"grain": "month"}})
    added = {"asked": [{"words": "each month's target", "kind": "figure", "plan": "target_amount"}],
             "intent": "trend", "measures": ["net_amount", "target_amount"], "time": {"grain": "month"},
             "follow_up": "refine"}
    gaps = _missing(retail, added, ["period", "net_amount", "target_amount"],
                    "put each month's target next to it", before=before)
    assert [why for _, why in gaps] == ["not_kept"] and "orders" in gaps[0][0]
    assert _missing(retail, added, ["period", "net_amount", "target_amount"],
                    "show the target instead of the orders", before=before) == []
    # Kept in the plan but not shown (the answer against target shows its own figures): not kept either.
    kept = {**added, "measures": ["net_amount", "number_of_orders"], "target": {"actual": "net_amount",
                                                                                "target": "target_amount"}}
    assert [why for _, why in _missing(retail, kept, ["period", "net_amount", "target_amount", "of_target"],
                                       "put each month's target next to it", before=before)] == ["not_kept"]
    units = {"intent": "trend", "measures": ["quantity"], "time": {"grain": "month"}, "follow_up": "refine"}
    assert _missing(retail, units, ["period", "quantity"], "show the quantity by month", before=before) == [], \
        "a follow-up naming its own figure may put it in place of the one shown"
    why = {"intent": "drivers", "measures": ["net_amount"], "follow_up": "refine"}
    assert _missing(retail, why, ["grouping", "member", "before", "after", "change"], "why did it drop?",
                    before=before) == [], "a why shows its figures in its own columns"
    top5 = {"intent": "rank", "measures": ["net_amount"], "group_by": ["region.name"], "limit": 5, "chart": "bar",
            "follow_up": "refine"}
    assert [why for _, why in _missing(retail, top5, ["region_name", "net_amount"],
                                       "only the top 5, as a bar chart", before=before)] == ["not_kept"]


def test_words_the_ai_could_not_place_are_no_part_left_out(retail):
    plan = {"asked": [{"words": "returns"}, {"words": "net sales", "kind": "figure", "plan": "net_amount"}],
            "intent": "value", "measures": ["net_amount"]}
    assert _missing(retail, plan, ["net_amount"]) == []


def test_where_values_may_not_reach_the_ai_a_name_left_out_is_asked_for_by_its_placeholder(retail):
    model, warehouse, index = retail
    whole = {"intent": "value", "measures": ["net_amount"], "time": Y2025}
    ai = Recorded(whole, whole)
    services = Services(model=model, warehouse=warehouse, complete=ai, index=index, today=TODAY,
                        values_allowed=False)
    payload = answer_question("net sales of Cola Large in 2025", services, Session())
    assert len(ai.sent) == 2
    assert all("cola large" not in sent.casefold() for sent in ai.sent), "never the name, in either reading"
    assert "⟨v1⟩" in ai.sent[1] and "leaves out parts of the request" in ai.sent[1]
    left = payload["parts_left_out"]
    assert left[0]["words"] == "Cola Large" and left[0]["private"]


def test_a_member_shown_as_a_row_of_its_grouping_is_given(retail):
    from core2.agent import parts as P
    from core2.plan.ir import Plan
    from core2.plan.values import ValueMatch

    plan = Plan.model_validate({"kind": "query", "intent": "compare", "measures": ["net_amount"],
                                "group_by": ["region.name"]})
    named = [ValueMatch("North", 8, 13, "region.name", "North"), ValueMatch("South", 18, 23, "region.name", "South")]
    payload = {"data": {"headers": ["region_name", "net_amount"], "rows": [{"region_name": "North", "net_amount": 1}]}}
    assert P.missing("Compare North and South for 2025", plan, payload, retail[0], named=named) == []


def test_a_share_of_the_change_is_given_only_by_its_own_column(retail):
    plan = {"asked": [{"words": "the change in $", "kind": "figure", "plan": "compare"},
                      {"words": "each category's % of the total increase", "kind": "figure", "plan": "rates"},
                      {"words": "each region's % of the total change", "plan": "share"}],
            "intent": "compare", "measures": ["net_amount"], "group_by": ["region.name"],
            "time": {"window": {"kind": "this", "unit": "year"}, "compare": {"kind": "previous_period"}}}
    compared = ["region_name", "net_amount", "net_amount_prior", "net_amount_change", "net_amount_pct_change"]
    assert [w for w, _ in _missing(retail, plan, compared)] == [
        "each category's % of the total increase", "each region's % of the total change"]
    assert _missing(retail, plan, [*compared, "share_of_change"]) == []


def test_a_rate_of_rows_with_a_field_filled_in_is_a_share_of_the_total():
    from core2.plan.ir import Plan
    from core2.plan.planner import _share_not_rate

    plan = Plan.model_validate({"kind": "query", "intent": "share", "group_by": ["country_region"],
                                "asked": [{"words": "percentage of sales", "kind": "figure", "plan": "rates"}],
                                "rates": [{"name": "share_of_sales", "measure": "net_sales",
                                           "where": [{"field": "country_region", "op": "not_null"}]}]})
    shared = _share_not_rate(plan)
    assert shared.rates == [] and shared.measures == ["net_sales"] and shared.intent == "share"
    assert shared.asked[0].plan == "share"
    kept = plan.model_copy(update={"rates": [plan.rates[0].model_copy(update={"where": [
        {"field": "status", "op": "eq", "values": ["Cancelled"]}]})]})
    assert _share_not_rate(Plan.model_validate(kept.model_dump())).rates, "a real condition stays a rate"


def test_a_half_year_written_as_a_grain_is_the_period_alone():
    from core2.plan.planner import parse

    plan = parse(json.dumps({"kind": "query", "intent": "compare", "measures": ["net_amount"],
                             "time": {"grain": "half", "window": {"kind": "between", "start": "2026-01-01",
                                                                  "end": "2026-06-30"}}}))
    assert plan.time.grain is None and plan.time.window.start.isoformat() == "2026-01-01"


def test_an_amount_asked_of_an_analysis_that_only_counts_is_left_out_with_a_word(retail):
    # "Receivables ageing with the amount owed in each bucket": the AI gave the amount to the ageing itself, which
    # counts the open orders and totals nothing unless a figure is asked of it. The amount was dropped silently.
    ageing = {"on_time": {"done": "delivered_date", "due": "order_date", "ageing": True}, "intent": "breakdown"}
    for part in ({"words": "the amount owed in each bucket", "plan": "ageing"},
                 {"words": "the amount owed in each bucket", "kind": "figure", "plan": "on_time"}):
        plan = {**ageing, "asked": [{"words": "receivables ageing", "kind": "figure", "plan": "ageing"}, part]}
        assert _missing(retail, plan, ["days_past_due", "open"]) == [("the amount owed in each bucket",
                                                                      "not_in_plan")]
        assert _missing(retail, {**plan, "measures": ["net_amount"]}, ["days_past_due", "open", "net_amount"]) == []
    # A count asked of it is what it gives.
    counted = {**ageing, "asked": [{"words": "how many orders are overdue", "kind": "figure", "plan": "ageing"}]}
    assert _missing(retail, counted, ["days_past_due", "open"]) == []
