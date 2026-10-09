"""A follow-up names a row of the answer on screen by its place: "break the first one down".

The planner is given the previous question and plan, never the answer's rows. After "Which 10
warehouses have the least available stock?", "Break the first one down by item group" had no
name to filter on: the AI wrote a placeholder of its own ("<first warehouse from previous
result>") and the answer said there was no warehouse called that. The answer's members are now
kept with its turn, in the order shown, and "the first one", "the second store" or "the last one"
is handed to the planner as that member, as a name the question wrote would be: by its stored
value, or by a placeholder where member values are kept from the AI.

Words that only look like a place are not rows: "the first quarter", "top 5". Invented retail data.
"""

from __future__ import annotations

import datetime as dt
import json
import re

import pytest

from core2.plan.values import MemberIndex, ValueMatch, placed
from core2.service import Services, Session, answer_question

TODAY = dt.date(2026, 6, 15)
H1 = {"kind": "between", "start": "2026-01-01", "end": "2026-06-30"}
TOP = {"kind": "query", "intent": "rank", "measures": ["net_amount"], "group_by": ["store"],
       "sort": [{"by": "net_amount", "desc": True}], "limit": 5, "time": {"window": H1}}


@pytest.fixture(scope="module")
def retail():
    from evals.core2 import domains
    from evals.core2.compile_eval import learn

    return learn(domains.build("retail"), "descriptive")


def _converse(retail, second: str, *, values_allowed: bool = True) -> tuple[dict, dict, list[str]]:
    """Ask for the top stores, then ``second``; the AI's second plan filters on whatever member it is handed."""
    from core2.warehouse.runner import DuckDBWarehouse

    built, model = retail
    tails: list[str] = []

    def planner(stable: str, tail: str) -> str:
        tails.append(tail)
        if len(tails) == 1:
            return json.dumps(TOP)
        handed = re.findall(r'-> (\S+) = "([^"]+)"', tail)
        plan = {"kind": "query", "intent": "breakdown", "measures": ["net_amount"], "group_by": ["category"],
                "time": {"window": H1}, "follow_up": "refine"}
        if handed:
            plan["filters"] = [{"field": handed[0][0], "op": "eq", "values": [handed[0][1]]}]
        return json.dumps(plan)

    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=planner, index=MemberIndex(),
                        today=TODAY, values_allowed=values_allowed)
    session = Session()
    first = answer_question("Which 5 stores sold the most?", services, session)
    then = answer_question(second, services, session)
    return first, then, tails


def _stores(payload: dict) -> list[str]:
    data = payload["data"]
    column = next(h for h in data["headers"] if data["column_formats"].get(h) == "text")
    return [r[column] for r in data["rows"]]


def test_the_first_one_is_the_first_member_of_the_answer_on_screen(retail):
    first, then, tails = _converse(retail, "Break the first one down by category")
    leader = _stores(first)[0]
    assert f'"the first one" -> ' in tails[1] and f'= "{leader}"' in tails[1], tails[1]
    assert leader in then["answer"]["headline"] or leader in json.dumps(then.get("answer", {}).get("badges")), \
        then["answer"]["headline"]
    assert then.get("data", {}).get("rows"), then["answer"]["headline"]


def test_the_second_store_and_the_last_one_are_found_by_their_place(retail):
    first, _, tails = _converse(retail, "and the second store, by category?")
    assert f'= "{_stores(first)[1]}"' in tails[1], tails[1]
    first, _, tails = _converse(retail, "what about the last one")
    assert f'= "{_stores(first)[-1]}"' in tails[1], tails[1]


def test_where_member_values_are_kept_from_the_ai_it_is_handed_a_placeholder(retail):
    first, then, tails = _converse(retail, "Break the first one down by category", values_allowed=False)
    leader = _stores(first)[0]
    assert leader not in tails[1] and "⟨v" in tails[1], tails[1]
    assert then.get("data", {}).get("rows") and leader in json.dumps(then.get("plan")), then.get("plan")


@pytest.mark.parametrize("question", [
    "Net amount in the first quarter by category",
    "Show the top 5 categories",
    "Break it down by category for the first week",
    "Which is the top store in March?",
    "and the bottom store?",
])
def test_words_that_only_look_like_a_place_are_not_a_row(question):
    shown = [("store.name", "Old Town Store"), ("store.name", "Riverside")]
    assert placed(question, shown, []) == []


def test_a_member_the_question_names_itself_is_never_replaced():
    shown = [("store.name", "Old Town Store"), ("store.name", "Riverside")]
    named = ValueMatch("the first one", 6, 19, "store.name", "Riverside")
    assert placed("Break the first one down", shown, [named]) == []
    assert [(m.text, m.value) for m in placed("Break the first one down", shown, [])] == \
        [("the first one", "Old Town Store")]
    assert placed("the fifth one", shown, []) == []                      # not a row of this answer
    assert [m.value for m in placed("and the top one?", shown, [])] == ["Old Town Store"]
    assert [m.value for m in placed("the last store", shown, [])] == ["Riverside"]
    assert placed("the first one", [("store.name", None), *shown], []) == []    # an unnamed row has no value
