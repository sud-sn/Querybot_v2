"""A follow-up changes the answer on screen; a new question is asked on its own; when it could be either, the
reader is asked.

"Only North", "by week instead", "why did it drop?" change the answer on screen; "What were refunds by return
reason last quarter?" asks afresh. The words decide most turns the same way every time (core2/plan/followup.py),
so the AI is told what was decided, and a new question is planned without the answer before it: nothing of
that answer can leak into it. A short complete question after a narrowed answer ("How many orders?" after net
sales for North in March) could mean either, and is asked with two buttons. Every follow-up answer says which
answer it follows, and offers the same question asked on its own.

On the labelled conversations (evals/core2/conversations) and on the held-out ones never used to tune the
rules, the turns the words decide must be read right.

Synthetic data only.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.values import build_index
from core2.service import ABOVE, AFRESH, Services, Session, answer_question, which_way
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains, followup_eval
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
MARCH = {"kind": "between", "start": "2026-03-01", "end": "2026-03-31"}
NET_BY_REGION = {"intent": "breakdown", "measures": ["net_amount"], "group_by": ["region.name"]}
NORTH = {"field": "region.name", "op": "eq", "values": ["North"]}
NET_NORTH_MARCH = {"intent": "value", "measures": ["net_amount"], "filters": [NORTH], "time": {"window": MARCH}}
ORDERS = {"intent": "value", "measures": ["number_of_orders"]}


@pytest.fixture(scope="module")
def retail():
    built, model = learn(domains.build("retail"), "descriptive")
    warehouse = DuckDBWarehouse(built.con)

    def fetch(slug: str) -> list[object]:
        column = model.columns[model.attributes[slug].column]
        table = model.tables[column.table]
        return [r[0] for r in warehouse.query(f'SELECT DISTINCT "{column.name}" FROM "{table.name}"').rows]

    return model, warehouse, build_index(model, fetch)


class AI:
    """The planner's answers, in order, and the question tail it was sent each time."""

    def __init__(self, *plans: dict):
        self.plans, self.tails = list(plans), []

    def __call__(self, stable: str, tail: str) -> str:
        self.tails.append(tail)
        plan = self.plans.pop(0)
        return json.dumps({"kind": "query", **plan} if "kind" not in plan else plan)


def _ask(retail, ai: AI):
    model, warehouse, index = retail
    services = Services(model=model, warehouse=warehouse, complete=ai, index=index, today=TODAY)
    session = Session()
    return lambda question: answer_question(question, services, session), session


def test_a_turn_that_changes_the_answer_is_planned_with_it_and_says_what_it_follows(retail):
    # The AI marks it new; the words say it changes the answer on screen, and they decide.
    ai = AI(NET_BY_REGION, {**NET_BY_REGION, "filters": [NORTH], "follow_up": "new"})
    ask, _ = _ask(retail, ai)
    ask("Net sales by region")
    payload = ask("only North")
    assert "PREVIOUS PLAN" in ai.tails[1] and "READING: this question changes the answer above" in ai.tails[1]
    assert payload["plan"]["follow_up"] == "refine"
    assert payload["following"] == {"question": "Net sales by region", "ask_new": "New question: only North"}


def test_a_question_of_its_own_is_planned_without_the_answer_before_it(retail):
    refunds = {"intent": "breakdown", "measures": ["refund_amount"], "group_by": ["return.reason_code"],
               "follow_up": "refine"}
    ai = AI(NET_NORTH_MARCH, refunds)
    ask, _ = _ask(retail, ai)
    ask("Net sales in the North region in March 2026")
    payload = ask("What were refunds by return reason last quarter?")
    assert "PREVIOUS" not in ai.tails[1] and "READING" not in ai.tails[1]
    assert payload["plan"].get("follow_up", "new") == "new"
    assert "following" not in payload


def test_a_question_that_could_go_either_way_is_asked_with_two_buttons(retail):
    ai = AI(NET_NORTH_MARCH)
    ask, session = _ask(retail, ai)
    ask("Net sales in the North region in March 2026")
    payload = ask("How many orders?")
    assert len(ai.tails) == 1, "nothing is planned until the reader says which"
    assert payload["clarify"]["about"] == "follow_up" and payload["clarify"]["options"] == [ABOVE, AFRESH]
    assert '"How many orders?"' in payload["answer"]["headline"]
    assert [c["question"] for c in payload["follow_up_suggestions"]] == [ABOVE, AFRESH]
    assert [t.question for t in session.turns] == ["Net sales in the North region in March 2026"]


def test_a_new_question_picked_is_planned_on_its_own(retail):
    ai = AI(NET_NORTH_MARCH, ORDERS)
    ask, _ = _ask(retail, ai)
    ask("Net sales in the North region in March 2026")
    ask("How many orders?")
    payload = ask(AFRESH)
    assert "PREVIOUS" not in ai.tails[1]
    assert payload["question"] == "How many orders?"
    assert payload["plan"].get("follow_up", "new") == "new" and "following" not in payload


def test_the_answer_above_picked_keeps_its_scope_and_says_so(retail):
    ai = AI(NET_NORTH_MARCH, {**ORDERS, "filters": [NORTH], "time": {"window": MARCH}})
    ask, _ = _ask(retail, ai)
    ask("Net sales in the North region in March 2026")
    ask("How many orders?")
    payload = ask(ABOVE)
    assert "PREVIOUS PLAN" in ai.tails[1] and "READING" in ai.tails[1]
    assert payload["question"] == "How many orders?" and payload["plan"]["follow_up"] == "refine"
    assert payload["following"]["question"] == "Net sales in the North region in March 2026"


def test_a_question_instead_of_a_button_is_read_as_it_is(retail):
    gross = {"intent": "breakdown", "measures": ["gross_amount"], "group_by": ["category.name"],
             "time": {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}}
    ai = AI(NET_NORTH_MARCH, gross)
    ask, session = _ask(retail, ai)
    ask("Net sales in the North region in March 2026")
    ask("How many orders?")
    payload = ask("Gross sales by category for 2025")
    assert payload["question"] == "Gross sales by category for 2025" and "PREVIOUS" not in ai.tails[1]
    assert session.pending is None


def test_ask_as_a_new_question_plans_the_same_words_on_their_own(retail):
    ai = AI(NET_BY_REGION, {**NET_BY_REGION, "filters": [NORTH]}, {**ORDERS, "filters": [NORTH]})
    ask, _ = _ask(retail, ai)
    ask("Net sales by region")
    following = ask("only North")["following"]
    payload = ask(following["ask_new"])
    assert "PREVIOUS" not in ai.tails[2] and "QUESTION: only North" in ai.tails[2]
    assert payload["question"] == "only North"
    assert payload["plan"].get("follow_up", "new") == "new" and "following" not in payload


def test_the_ai_may_say_it_cannot_tell(retail):
    """A turn the words leave open is the AI's to read; "unsure" asks the reader."""
    ai = AI(NET_BY_REGION, {**NET_BY_REGION, "follow_up": "unsure"}, NET_BY_REGION)
    ask, _ = _ask(retail, ai)
    ask("Net sales by region")
    payload = ask("what does the rest of the year look like")
    assert "PREVIOUS PLAN" in ai.tails[1] and "READING" not in ai.tails[1]
    assert payload["clarify"]["options"] == [ABOVE, AFRESH]
    ask("new")
    assert "PREVIOUS" not in ai.tails[2]


def test_a_reply_to_a_question_asked_back_keeps_the_conversation(retail):
    asked = {"kind": "clarify", "clarify": {"about": "measure", "question": "Which sales do you mean",
                                            "options": ["Net amount", "Gross amount"]}}
    ai = AI(asked, {**NET_BY_REGION, "follow_up": "refine"})
    ask, _ = _ask(retail, ai)
    ask("Sales by region")
    payload = ask("Net amount")
    assert "PREVIOUS QUESTION: Sales by region" in ai.tails[1]
    assert "following" not in payload, "an answer to a question asked back follows no answer"


@pytest.mark.parametrize("question,kind", [
    ("only North", "refine"),
    ("how did those same customers do in H1 2025?", "refine"),      # a period, but those customers
    ("What were refunds by return reason last quarter?", "new"),
    ("What are the same-store sales by region this year?", "new"),  # "same-store" points at nothing
    ("New question: only North", "new"),
    ("Which items explain the change?", "refine"),                   # the change the answer on screen shows
    ("What products are behind this drop in the second half of 2025?", "refine"),
    ("Which items drove net sales in 2025 by region?", "new"),       # no change on screen is named
])
def test_what_the_words_say(retail, question, kind):
    from core2.plan.followup import read_turn
    from core2.plan.ir import Plan

    model, _, _ = retail
    previous = Plan.model_validate({**NET_BY_REGION, "limit": 3})
    assert read_turn(question, previous=previous, model=model).kind == kind


def test_a_name_the_data_does_not_hold_is_left_to_the_ai_not_asked(retail):
    """"net sales for nowhere" sets its own scope with a name the members do not hold: the AI reads it, and the
    answer says no store is called that, not "is this about the answer above?"."""
    from core2.plan.followup import read_turn
    from core2.plan.ir import Plan

    model, _, _ = retail
    previous = Plan.model_validate(NET_NORTH_MARCH)
    assert read_turn("net sales for nowhere", previous=previous, model=model).kind == "open"
    assert read_turn("How many orders?", previous=previous, model=model).kind == "unsure"
    assert read_turn("How many orders in total?", previous=previous, model=model).kind == "unsure"


@pytest.mark.parametrize("reply,way", [
    (ABOVE, "refine"), (AFRESH, "new"), ("1", "refine"), ("2", "new"), ("the second one", "new"),
    ("new", "new"), ("a new one", "new"), ("separate question", "new"), ("same as above", "refine"),
    ("keep the answer above", "refine"), ("start over", "new"), ("hmm", None),
    ("Gross sales by category for 2025", None), ("same for South", None), ("last year", None),
])
def test_a_reply_to_the_two_buttons(reply, way):
    assert which_way(reply, [ABOVE, AFRESH]) == way


@pytest.mark.parametrize("folder", ["", "heldout"])
def test_the_words_decide_most_turns_and_decide_them_right(folder):
    """On the labelled conversations, and on held-out ones never used to tune the rules."""
    where = followup_eval.CONVERSATIONS / folder if folder else followup_eval.CONVERSATIONS
    rows = [r for name in followup_eval.available(where) for r in followup_eval.signals(name, folder=where)]
    decided = [r for r in rows if r["status"] != "open"]
    right = sum(r["status"] == "right" for r in decided)
    asked = sum(r["reading"] == "unsure" for r in rows)
    wrong = [f"{r['domain']} {r['thread']}.{r['turn']}: {r['question']} read as {r['reading']}"
             for r in decided if r["status"] == "wrong"]
    assert len(rows) >= 60
    assert len(decided) >= 0.9 * len(rows)
    assert right >= 0.95 * len(decided), wrong
    assert asked <= 0.1 * len(rows)


def test_the_answer_card_says_what_it_follows():
    from tests.test_the_answer_card import card

    msg = {"engine": "core2", "question": "only North", "answer": {"headline": "Net sales in North: $1,200"},
           "following": {"question": "Net sales by region", "ask_new": "New question: only North"}}
    html = card(msg)
    assert "Following: Net sales by region" in html
    assert 'data-question="New question: only North"' in html and "Ask as a new question" in html
    assert "Suite de : Net sales by region" in card(msg, lang="fr")
    assert "answer-following" not in card({**msg, "following": None})
