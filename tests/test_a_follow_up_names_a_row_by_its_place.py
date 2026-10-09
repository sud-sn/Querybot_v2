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


def _converse(retail, second: str, *, values_allowed: bool = True, first_plan: dict = TOP,
              filtered_on: str | None = None, index: MemberIndex | None = None) -> tuple[dict, dict, list[str]]:
    """Ask ``first_plan``, then ``second``; the AI's second plan filters on whatever member it is handed
    (or on the words ``filtered_on``, as an AI that wrote the reference itself into the filter)."""
    from core2.warehouse.runner import DuckDBWarehouse

    built, model = retail
    tails: list[str] = []

    def planner(stable: str, tail: str) -> str:
        tails.append(tail)
        if len(tails) == 1:
            return json.dumps(first_plan)
        handed = re.findall(r'-> (\S+) = "([^"]+)"', tail)
        plan = {"kind": "query", "intent": "breakdown", "measures": ["net_amount"], "group_by": ["category"],
                "time": {"window": H1}, "follow_up": "refine"}
        if filtered_on is not None:
            plan["filters"] = [{"field": "store.name", "op": "eq", "values": [filtered_on]}]
        elif handed:
            plan["filters"] = [{"field": handed[0][0], "op": "eq", "values": [handed[0][1]]}]
        return json.dumps(plan)

    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=planner,
                        index=index or MemberIndex(), today=TODAY, values_allowed=values_allowed)
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


# ── by value, and "that region" ─────────────────────────────────────────────

LEAST = {**TOP, "sort": [{"by": "net_amount", "desc": False}]}


def test_the_lowest_one_and_the_highest_one_are_found_by_their_number_not_their_place(retail):
    """"Which division does the lowest one belong to?" was filtered on a profit centre called "lowest"."""
    first, then, tails = _converse(retail, "Which category does the lowest one sell most?")
    assert f'"the lowest one" -> ' in tails[1] and f'= "{_stores(first)[-1]}"' in tails[1], tails[1]
    assert then.get("data", {}).get("rows"), then["answer"]["headline"]
    first, _, tails = _converse(retail, "and the highest one?", first_plan=LEAST)    # least first: the last row
    assert f'= "{_stores(first)[-1]}"' in tails[1], tails[1]


def test_that_region_is_the_one_region_the_answer_on_screen_showed():
    shown = [("region.name", "North", 1200.0)]
    assert [(m.text, m.value) for m in placed("net sales for that region by month", shown, [])] == \
        [("that region", "North")]
    two = [("region.name", "North", 1200.0), ("region.name", "South", 900.0)]
    assert placed("net sales for that region by month", two, []) == []          # which one: not guessed
    assert placed("net sales for that month", shown, []) == []                   # not the field's noun


def test_words_that_point_at_a_row_are_never_refused_as_a_missing_name(retail):
    """The AI wrote the reference into the filter itself: the reply asks which, never "no store called 'lowest'"."""
    first, _, _ = _converse(retail, "q")
    index = MemberIndex()
    index.add("store.name", _stores(first))
    _, then, _ = _converse(retail, "Which category does the lowest one sell most?", filtered_on="lowest",
                           index=index)
    said = then["answer"]["headline"]
    assert re.search(r'I could not tell which store( name)? "lowest" means', said) and "called" not in said, said
    _, then, _ = _converse(retail, "net sales for nowhere", filtered_on="Nowhere Store", index=index)
    said = then["answer"]["headline"]
    assert re.search(r'there is no store( name)? called "Nowhere Store"', said) and "the first one" in said, said
    assert "compare that answer with the other period" not in said
