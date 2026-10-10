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

from core2.plan.values import MemberIndex, ValueMatch, placed, which_end
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


def test_a_long_question_that_points_at_a_row_still_follows_the_answer(retail):
    """"Show monthly net amount for the first one in the first half of 2026" names a measure, a split and a period,
    which alone reads as a new question: planned afresh, the AI was never handed the store it points at."""
    first, then, tails = _converse(retail, "Show monthly net amount for the first one in the first half of 2026")
    leader = _stores(first)[0]
    assert f'"the first one" -> ' in tails[1] and f'= "{leader}"' in tails[1], tails[1]
    assert "PREVIOUS PLAN" in tails[1], "it follows the answer on screen"
    assert then["plan"]["filters"][0]["values"] == [leader]


def test_that_division_is_found_in_whichever_grouping_shows_it():
    """"Which division does the lowest one belong to?" shows the profit centre with its division: "that
    division" in the next question is the division of that row, though the division is the second grouping."""
    one = [("profit_centre.name", "PC 7", 3.1, {"division.name": "Industrial"})]
    found = placed("quarterly gross profit for that division", one, [])
    assert [(m.text, m.attribute, m.value) for m in found] == [("that division", "division.name", "Industrial")]
    same = [*one, ("profit_centre.name", "PC 9", 4.0, {"division.name": "Industrial"})]
    assert [m.value for m in placed("and that division?", same, [])] == ["Industrial"]
    two = [*one, ("profit_centre.name", "PC 9", 4.0, {"division.name": "Retail"})]
    assert placed("and that division?", two, []) == []                         # which one: not guessed
    assert [(m.text, m.value) for m in placed("monthly gross profit for that one", one, [])] == \
        [("that one", "PC 7")]
    assert placed("monthly gross profit for that one", two, []) == []          # which one: not guessed
    by_month = [("time:month_of_year", "March", 10.0)]
    assert placed("net sales this year by region", by_month, []) == []          # a period, not the row


def test_that_region_of_a_store_on_screen_narrows_the_next_answer(retail):
    """The region is the answer's second grouping: pointed at, it filters, and is not taken out as a word of
    the question the reader did not quote."""
    top = {**TOP, "group_by": ["store", "region.name"], "limit": 1}
    first, then, tails = _converse(retail, "net amount for that region by category", first_plan=top)
    region = first["data"]["rows"][0]
    region = next(v for k, v in region.items() if "region" in k)
    assert f'"that region" -> region.name = "{region}"' in tails[1], tails[1]
    assert then["plan"]["filters"] == [{"field": "region.name", "op": "eq", "values": [region]}]
    assert not any("Not narrowed" in n for n in then["trust"]["date_context"])


def test_words_that_point_at_a_row_are_never_refused_as_a_missing_name(retail):
    """The AI wrote the reference into the filter itself: the reply asks which, never "no store called 'lowest'"."""
    first, _, _ = _converse(retail, "q")
    index = MemberIndex()
    index.add("store.name", _stores(first))
    _, then, _ = _converse(retail, "Which category does the lowest one sell most?", filtered_on="lowest",
                           index=index)
    said = then["answer"]["headline"]
    assert re.search(r'I could not tell which store( name)? "lowest" means', said) and "called" not in said, said
    _, then, _ = _converse(retail, 'net sales for "nowhere"', filtered_on="Nowhere Store", index=index)
    said = then["answer"]["headline"]
    assert re.search(r'there is no store( name)? called "Nowhere Store"', said) and "the first one" in said, said
    # A name that only starts like a pointing word is a name: "Top Value Stores" is no "top one".
    _, then, _ = _converse(retail, 'net sales for "top value"', filtered_on="Top Value Stores", index=index)
    assert re.search(r'there is no store( name)? called "Top Value Stores"', then["answer"]["headline"]), \
        then["answer"]["headline"]
    assert "compare that answer with the other period" not in said


# ── by its field, by its change, "its", and "the worst one" ─────────────────

WAREHOUSES = [("warehouse.name", "Docks", 100.0, {}, 50.0), ("warehouse.name", "Airport", 300.0, {}, -80.0),
              ("warehouse.name", "Harbour", 200.0, {}, 20.0)]


@pytest.mark.parametrize("question, phrase, member", [
    ("How many items in the top warehouse cost more than $1,000 each?", "the top warehouse", "Docks"),
    ("Monthly stock for the bottom warehouse", "the bottom warehouse", "Harbour"),
    ("Within the largest warehouse, which items hold the most?", "the largest warehouse", "Airport"),
    ("Compare the number one warehouse's stock with last year", "the number one warehouse", "Docks"),
    ("What drove the increase in the warehouse that grew the most?", "the warehouse that grew the most", "Docks"),
    ("and the one that fell the most?", "the one that fell the most", "Airport"),
    ("Show the monthly stock of the biggest mover in 2026", "the biggest mover", "Airport"),
])
def test_a_row_is_named_by_its_field_its_end_or_its_change(question, phrase, member):
    assert [(m.text, m.value) for m in placed(question, WAREHOUSES, [])] == [(phrase, member)]


@pytest.mark.parametrize("question", [
    "Show the top warehouse in March",                   # a ranking of March, not the row on screen
    "Show the warehouse that grew the most in 2024",
    "Who was the biggest mover in Q3?",
    "Who is the number one warehouse in 2024?",
    "Within the largest customers, which items sold?",   # not the answer's field
])
def test_a_ranking_asked_afresh_is_not_a_row_on_screen(question):
    assert placed(question, WAREHOUSES, []) == []


def test_the_change_of_a_row_is_needed_to_find_the_one_that_grew_the_most():
    plain = [s[:3] for s in WAREHOUSES]                  # an answer that compares nothing
    assert placed("the one that grew the most by month", plain, []) == []


def test_its_is_the_one_member_or_the_one_the_answer_was_asked_for():
    question = "Was the change in its revenue driven by price or by volume?"
    assert placed(question, WAREHOUSES, []) == []                               # three rows: which one?
    assert [m.value for m in placed(question, WAREHOUSES, [], leader=True)] == ["Docks"]
    assert [m.value for m in placed(question, WAREHOUSES[1:2], [])] == ["Airport"]


def test_which_question_names_a_leader():
    from core2.service import _names_a_leader

    assert _names_a_leader("Which group's price changed the most compared with 2025?")
    assert _names_a_leader("Which customer type got the most discount?")
    assert not _names_a_leader("Which 10 warehouses have the least available stock?")
    assert not _names_a_leader("Show the top 5 stores by net sales")
    assert not _names_a_leader("What is our average selling price by item group?")


def test_the_worst_one_is_asked_never_guessed():
    assert which_end("Show the monthly rejected quantity for the worst one", WAREHOUSES, []) == \
        ("the worst one", "Docks", "Harbour")
    assert which_end("monthly stock for the best warehouse", WAREHOUSES, []) == ("the best warehouse", "Docks", "Harbour")
    assert which_end("Which is the best warehouse in 2024?", WAREHOUSES, []) is None     # a ranking asked afresh
    assert which_end("the worst one", WAREHOUSES[:1], []) is None                         # one row: nothing to ask


def test_the_worst_one_asks_which_end_and_the_reply_narrows_the_question(retail):
    """"Show net amount by category for the worst one": most or least net amount? The reader picks, and the
    question is answered for that store, in quotes, as a follow-up of the answer on screen."""
    from core2.warehouse.runner import DuckDBWarehouse

    built, model = retail
    tails: list[str] = []

    def planner(stable: str, tail: str) -> str:
        tails.append(tail)
        if len(tails) == 1:
            return json.dumps(TOP)
        handed = re.findall(r'-> (\S+) = "([^"]+)"', tail)
        return json.dumps({"kind": "query", "intent": "breakdown", "measures": ["net_amount"], "group_by": ["category"],
                           "time": {"window": H1}, "follow_up": "refine",
                           "filters": [{"field": handed[0][0], "op": "eq", "values": [handed[0][1]]}]})

    index = MemberIndex()
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=planner, index=index, today=TODAY)
    session = Session()
    first = answer_question("Which 5 stores sold the most?", services, session)
    stores = _stores(first)
    index.add("store.name", stores)
    asked = answer_question("Show net amount by category for the worst one", services, session)
    assert asked["clarify"]["options"] == [stores[0], stores[-1]], asked["clarify"]
    assert len(tails) == 1, "nothing is planned until the reader says which"
    then = answer_question(stores[-1], services, session)
    assert f'= "{stores[-1]}"' in tails[1] and "PREVIOUS PLAN" in tails[1], tails[1]
    assert then["plan"]["filters"][0]["values"] == [stores[-1]]


def test_the_one_that_grew_the_most_is_read_from_the_comparison_on_screen(retail):
    compared = {**TOP, "intent": "compare", "limit": None, "sort": [{"by": "change", "desc": False}],
                "time": {"window": H1, "compare": {"kind": "window", "window": {"kind": "between", "start": "2025-01-01",
                                                                                 "end": "2025-06-30"}}}}
    first, then, tails = _converse(retail, "net amount by category for the store that grew the most",
                                   first_plan=compared)
    rows = first["data"]["rows"]
    grew = max(rows, key=lambda r: r["net_amount_change"])
    store = next(v for k, v in grew.items() if isinstance(v, str))
    assert rows[0] is not grew, "the answer lists the biggest drop first: the place says nothing"
    assert f'"the store that grew the most" -> store.name = "{store}"' in tails[1], tails[1]
    assert then["plan"]["filters"][0]["values"] == [store]
