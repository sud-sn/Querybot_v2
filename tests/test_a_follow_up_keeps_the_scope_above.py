"""A follow-up has the filters and the period of the answer above; a question about something else keeps none.

Found on the sector stress test (round 3): half the follow-ups failed, and the ones after a right answer failed in
the same few ways --

* the scope was lost: "How many different products did they buy in 2011?" after a country's customers counted
  every invoice of the year ("they" pointed at nothing the reader could see); "how many orders?" after net sales
  for North in March was asked back with two buttons;
* a scope was kept that the question replaced: "Break 1997 down by region" after 1997 against 1996 stayed a
  comparison; "Which branches were those loans booked at?" after loans by month stayed month by month;
* a question in French was not read as a follow-up ("et par motif de retour ?", "pour ce magasin"), and the AI
  asked "What about the answer above: change the time period, add a breakdown ...?" in English;
* a question asked right after one that got no answer ("and 2025? split both years by fee type" after a refused
  "fee income in 2024") went on with the answer before it, as if the refused one had not been asked.

The reader is never asked which: the words decide most turns (core2/plan/followup.py), the AI the rest with the
conversation in view, and what a follow-up keeps is decided in code (core2/plan/thread.py) and said on the answer.

Synthetic data only.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.followup import read_turn
from core2.plan.ir import Compare, Filter, Plan, TimeSpec, Window
from core2.plan.thread import kept_from_above, kept_parts
from core2.plan.values import build_index
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
NORTH = Filter(field="region.name", op="eq", values=["North"])
MARCH = Window(kind="between", start=dt.date(2026, 3, 1), end=dt.date(2026, 3, 31))
Y1997 = Window(kind="between", start=dt.date(1997, 1, 1), end=dt.date(1997, 12, 31))
Y1996 = Window(kind="between", start=dt.date(1996, 1, 1), end=dt.date(1996, 12, 31))


@pytest.fixture(scope="module")
def retail():
    built, model = learn(domains.build("retail"), "descriptive")
    warehouse = DuckDBWarehouse(built.con)

    def fetch(slug: str) -> list[object]:
        column = model.columns[model.attributes[slug].column]
        table = model.tables[column.table]
        return [r[0] for r in warehouse.query(f'SELECT DISTINCT "{column.name}" FROM "{table.name}"').rows]

    return model, warehouse, build_index(model, fetch)


def _always(_plan: Plan) -> bool:
    return True


# ── what a follow-up keeps, in code ─────────────────────────────────────────


def test_a_follow_up_keeps_a_filter_the_ai_left_out(retail):
    model, _, _ = retail
    before = Plan(intent="value", measures=["net_amount"], filters=[NORTH], time=TimeSpec(window=MARCH))
    ai = Plan(intent="value", measures=["number_of_orders"], follow_up="refine")
    kept = kept_from_above(ai, model, "How many orders?", before, _always)
    assert kept.filters == [NORTH]
    assert kept_parts(kept.model_copy(update={"time": TimeSpec(window=MARCH)}), before, "How many orders?") \
        == ([NORTH], MARCH)


@pytest.mark.parametrize("question", [
    "and South?",                       # another member of the same field: the AI's filter stands
    "net sales by region",              # split by the field the answer was narrowed by
    "for all regions",                  # every member of it
    "overall, how many orders?",        # everything
    "same but without the filter",
])
def test_a_follow_up_lets_go_of_a_filter_it_replaces(retail, question):
    model, _, _ = retail
    before = Plan(intent="value", measures=["net_amount"], filters=[NORTH])
    south = [Filter(field="region.name", op="eq", values=["South"])] if "South" in question else []
    group = ["region.name"] if "by region" in question else []
    ai = Plan(intent="value", measures=["net_amount"], filters=south, group_by=group, follow_up="refine")
    assert NORTH not in kept_from_above(ai, model, question, before, _always).filters


def test_a_filter_the_new_figure_cannot_be_narrowed_by_is_not_kept(retail):
    model, _, _ = retail
    before = Plan(intent="value", measures=["net_amount"], filters=[NORTH])
    ai = Plan(intent="value", measures=["number_of_orders"], follow_up="refine")
    assert kept_from_above(ai, model, "How many orders?", before, lambda p: not p.filters).filters == []


def test_a_new_question_keeps_nothing_it_does_not_write(retail):
    model, _, _ = retail
    before = Plan(intent="value", measures=["net_amount"], filters=[NORTH], time=TimeSpec(window=MARCH))
    copied = Plan(intent="value", measures=["number_of_orders"], filters=[NORTH], time=TimeSpec(window=MARCH),
                  follow_up="new")
    plan = kept_from_above(copied, model, "How many customers do we have?", before, _always)
    assert plan.filters == [] and plan.time.window.kind == "all"
    # what the question writes itself stays
    written = kept_from_above(copied, model, "How many orders did North take in March 2026?", before, _always)
    assert written.filters == [NORTH] and written.time.window == MARCH


def test_a_period_named_alone_is_no_longer_a_comparison(retail):
    """"Break 1997 down by region" after the growth of 1997 against 1996: 1997 by region."""
    model, _, _ = retail
    compare = Compare(kind="window", window=Y1996)
    before = Plan(intent="compare", measures=["net_amount"], time=TimeSpec(window=Y1997, compare=compare))
    ai = Plan(intent="compare", measures=["net_amount"], group_by=["region.name"], follow_up="refine",
              time=TimeSpec(window=Y1997, compare=compare))
    plan = kept_from_above(ai, model, "Break 1997 down by region", before, _always)
    assert plan.time.compare is None and plan.intent == "breakdown" and plan.time.window == Y1997
    # still comparing: kept
    for still in ("and the growth by region in 1997?", "1997 vs 1996 by region"):
        assert kept_from_above(ai, model, still, before, _always).time.compare == compare


def test_which_members_of_a_monthly_answer_are_no_longer_month_by_month(retail):
    """"Which stores were those sales made at?" after sales by month: the stores, not their months."""
    model, _, _ = retail
    before = Plan(intent="trend", measures=["net_amount"], time=TimeSpec(window=MARCH, grain="month"))
    ai = Plan(intent="trend", measures=["net_amount"], group_by=["store.name"], follow_up="refine",
              time=TimeSpec(window=MARCH, grain="month"))
    plan = kept_from_above(ai, model, "Which stores were those sales made at?", before, _always)
    assert plan.time.grain is None and plan.intent == "breakdown"
    asked = kept_from_above(ai, model, "Which stores sold the most each month?", before, _always)
    assert asked.time.grain == "month"


# ── the words, in English and in French ─────────────────────────────────────


@pytest.mark.parametrize("question,kind", [
    ("How many different products did they buy in 2011?", "open"),    # "they": read with the conversation in view
    ("et par motif de retour, les 5 premiers ?", "refine"),
    ("pour ce magasin, combien de tickets ?", "refine"),
    ("et leurs ventes par mois ?", "refine"),
    ("seulement les clients en ligne", "refine"),
])
def test_a_follow_up_is_read_as_one_in_either_language(retail, question, kind):
    model, _, _ = retail
    before = Plan(intent="value", measures=["net_amount"], filters=[NORTH], time=TimeSpec(window=MARCH))
    assert read_turn(question, previous=before, model=model).kind == kind


@pytest.mark.parametrize("question", [
    "Quel est le chiffre d'affaires de ce mois ?",
    "Ventes nettes par région en 2025",
    "Est-ce que les ventes ont augmenté en 2025 ?",           # "ce que" points at nothing
    "Quelles sont les ventes sur ces 12 derniers mois ?",     # nor "ces 12 derniers mois"
    "Par magasin, quelles sont les ventes nettes de 2025 ?",   # a scope of its own, opened with "par"
    "Pour 2025, quelles sont les ventes par magasin ?",
    "Quelles régions ont leurs ventes en hausse en 2025 ?",    # their own sales, not the ones on screen
    "Which stores beat their target in 2025?",
    "By store, what were net sales in 2025?",
])
def test_a_question_of_its_own_is_no_follow_up_in_either_language(retail, question):
    model, _, _ = retail
    before = Plan(intent="value", measures=["net_amount"], filters=[NORTH], time=TimeSpec(window=MARCH))
    assert read_turn(question, previous=before, model=model).kind != "refine"


@pytest.mark.parametrize("question", ["Which stores had an increase in net amount?",
                                      "What is an average net amount by store?",
                                      "Give me an overview of net amount by store"])
def test_the_word_an_names_no_period(retail, question):
    """"an" is French for a year only in French: in English it names no period, and the question is not read as one
    that sets its own."""
    model, _, _ = retail
    before = Plan(intent="value", measures=["net_amount"], filters=[NORTH], time=TimeSpec(window=MARCH))
    assert read_turn(question, previous=before, model=model).kind != "new"


@pytest.mark.parametrize("question", ["by month", "monthly", "and by quarter?", "par mois"])
def test_a_comparison_split_by_period_stays_a_comparison(retail, question):
    model, _, _ = retail
    compare = Compare(kind="window", window=Y1996)
    before = Plan(intent="compare", measures=["net_amount"], time=TimeSpec(window=Y1997, compare=compare))
    ai = before.model_copy(update={"follow_up": "refine", "time": TimeSpec(window=Y1997, compare=compare,
                                                                            grain="month")})
    assert kept_from_above(ai, model, question, before, _always).time.compare == compare


@pytest.mark.parametrize("question", ["What about by region?", "What does it look like by store?",
                                      "Which months were best by region?"])
def test_a_monthly_answer_split_another_way_stays_monthly(retail, question):
    model, _, _ = retail
    before = Plan(intent="trend", measures=["net_amount"], time=TimeSpec(window=MARCH, grain="month"))
    ai = Plan(intent="trend", measures=["net_amount"], group_by=["region.name"], follow_up="refine",
              time=TimeSpec(window=MARCH, grain="month"))
    assert kept_from_above(ai, model, question, before, _always).time.grain == "month"


def test_a_number_the_new_question_writes_is_its_own(retail):
    model, _, _ = retail
    over = Filter(field="net_amount", op="gt", values=[100])
    before = Plan(intent="list", measures=["net_amount"], filters=[over])
    plan = Plan(intent="list", measures=["number_of_orders"], filters=[over], follow_up="new")
    assert kept_from_above(plan, model, "How many orders had a net amount over 100?", before, _always).filters \
        == [over]


# ── through the question path ───────────────────────────────────────────────


class AI:
    def __init__(self, *answers):
        self.answers, self.tails = list(answers), []

    def __call__(self, stable: str, tail: str) -> str:
        self.tails.append(tail)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return json.dumps(answer)


def _ask(retail, ai):
    model, warehouse, index = retail
    services = Services(model=model, warehouse=warehouse, complete=ai, index=index, today=TODAY)
    session = Session()
    return (lambda q: answer_question(q, services, session)), session


NET_NORTH = {"kind": "query", "intent": "value", "measures": ["net_amount"],
             "filters": [{"field": "region.name", "op": "eq", "values": ["North"]}],
             "time": {"window": {"kind": "between", "start": "2026-03-01", "end": "2026-03-31"}}}


def test_the_ai_asking_what_about_the_answer_above_is_sent_back_to_decide(retail):
    menu = {"kind": "clarify", "clarify": {"about": "follow_up", "question": "What about the answer above",
                                           "options": ["Change the time period", "Add a breakdown"]}}
    orders = {"kind": "query", "intent": "value", "measures": ["number_of_orders"], "follow_up": "refine"}
    ai = AI(NET_NORTH, menu, orders)
    ask, _ = _ask(retail, ai)
    ask('Net sales in "North" in March 2026')
    payload = ask("et combien de commandes ?")
    assert "clarify" not in payload, payload.get("answer")
    assert "Do not ask whether or how the question is about the answer above" in ai.tails[2]
    assert payload["plan"]["filters"] == NET_NORTH["filters"]


def test_a_refused_question_stays_in_the_conversation(retail):
    """"And 2025?" right after a question that got no answer goes on with that question."""
    refusal = {"kind": "unsupported", "notes": ["nothing in this data records fees"]}
    orders = {"kind": "query", "intent": "value", "measures": ["number_of_orders"], "follow_up": "refine"}
    ai = AI(NET_NORTH, refusal, orders)
    ask, session = _ask(retail, ai)
    ask('Net sales in "North" in March 2026')
    refused = ask("What were the fee totals?")
    assert refused.get("unsupported") and refused["answer"]["headline"].startswith("I cannot answer that")
    assert session.turns[-1].question == "What were the fee totals?" and session.turns[-1].unanswered
    ask("and in 2025?")
    assert "PREVIOUS QUESTION (not answered: it was refused): What were the fee totals?" in ai.tails[-1]


def test_nothing_a_refused_question_tried_is_carried(retail):
    """'and for "Nowhere"?' was refused (no region is called that): the next "and by store?" follows the answer on
    screen, North, and never the refused filter; a filter the reader never quoted is never carried either."""
    nowhere = {**NET_NORTH, "filters": [{"field": "region.name", "op": "eq", "values": ["Nowhere"]}],
               "follow_up": "refine"}
    by_store = {"kind": "query", "intent": "breakdown", "measures": ["net_amount"], "group_by": ["store.name"],
                "follow_up": "refine"}
    ai = AI(NET_NORTH, nowhere, by_store)
    ask, session = _ask(retail, ai)
    ask('Net sales in "North" in March 2026')
    refused = ask('and for "Nowhere"?')
    assert "data" not in refused or not refused.get("data")
    assert session.turns[-1].unanswered and session.turns[-1].plan is None
    payload = ask("and by store?")
    assert payload["plan"]["filters"] == NET_NORTH["filters"]
    assert "Nowhere" not in json.dumps(payload["plan"])
    assert payload["following"]["question"] == 'Net sales in "North" in March 2026'


def test_a_plan_that_keeps_the_scope_above_follows_it_though_the_ai_said_nothing(retail):
    """"How many orders?" planned with North and March and no word of follow-up: rule 11 reads it as a follow-up,
    and the answer keeps them."""
    orders = {"kind": "query", "intent": "value", "measures": ["number_of_orders"],
              "filters": NET_NORTH["filters"], "time": NET_NORTH["time"]}
    ai = AI(NET_NORTH, orders)
    ask, _ = _ask(retail, ai)
    ask('Net sales in "North" in March 2026')
    payload = ask("How many orders?")
    assert payload["plan"]["follow_up"] == "refine" and payload["plan"]["filters"] == NET_NORTH["filters"]


def test_which_member_of_the_answer_above_is_a_fair_question(retail):
    """"Which store from the answer above do you mean?" is asked back as it is, never sent back to the AI."""
    asked = {"kind": "clarify", "clarify": {"about": "member", "question": "Which store from the answer above do you mean",
                                            "options": ["Airport Store", "Old Town Store"]}}
    by_store = {"kind": "query", "intent": "breakdown", "measures": ["net_amount"], "group_by": ["store.name"]}
    ai = AI(by_store, asked)
    ask, _ = _ask(retail, ai)
    ask("Net sales by store")
    payload = ask("how did that store do last month?")
    assert len(ai.tails) == 2 and payload.get("clarify", {}).get("about") == "member"


def test_member_placeholders_stay_the_same_across_a_turn_with_no_answer(retail):
    """With member values withheld from the AI, "<v1>" still means North after a question that got no answer, and a
    new name gets a placeholder of its own."""
    from core2.plan.planner import Turn, plan_question
    from core2.plan.values import Masked, ValueMatch
    from core2.service import NOT_REACHED

    model, _, _ = retail
    first = Plan.model_validate({k: v for k, v in NET_NORTH.items() if k != "kind"})
    history = [Turn('Net sales in "North" in March 2026', first, Masked('Net sales in "⟨v1⟩" in March 2026',
                                                                       {"⟨v1⟩": "North"})),
               Turn("which of the late ones?", None, None, unanswered=NOT_REACHED)]
    question = 'and "South"?'
    south = ValueMatch(text="South", start=5, end=10, attribute="region.name", value="South")
    ai = AI({"kind": "query", "intent": "value", "measures": ["net_amount"], "follow_up": "refine",
             "filters": [{"field": "region.name", "op": "eq", "values": ["⟨v2⟩"]}]})
    plan_question(model, question, ai, today=TODAY, history=history, matches=[south], values_allowed=False)
    tail = ai.tails[0]
    assert 'EARLIER QUESTION: Net sales in "⟨v1⟩" in March 2026' in tail and '"⟨v1⟩"' in tail.split("EARLIER PLAN:")[1]
    assert 'QUESTION: and "⟨v2⟩"?' in tail and "North" not in tail and "South" not in tail


def test_a_question_the_ai_never_reached_stays_in_the_conversation(retail):
    ai = AI(NET_NORTH, RuntimeError("429 rate limited"), {"kind": "query", "intent": "value",
                                                          "measures": ["number_of_orders"], "follow_up": "refine"})
    ask, session = _ask(retail, ai)
    ask('Net sales in "North" in March 2026')
    with pytest.raises(RuntimeError):
        ask("which of the late ones were in the North?")
    assert session.turns[-1].unanswered and session.turns[-1].plan is None
    payload = ask("how many orders were they?")
    tail = ai.tails[-1]
    # the answer before it is still what the question follows, and the unanswered question is said
    assert "EARLIER QUESTION: Net sales in" in tail and "which of the late ones" in tail
    assert payload["plan"]["filters"] == NET_NORTH["filters"]
    assert payload["following"]["question"] == 'Net sales in "North" in March 2026'
    assert any("North" in k for k in payload["following"]["kept"])
    assert [t.unanswered for t in session.turns] == ["", ""], "an answered turn passes the unanswered one by"
