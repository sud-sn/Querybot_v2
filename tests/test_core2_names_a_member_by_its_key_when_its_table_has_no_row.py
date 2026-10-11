"""A ranking whose members have no name is answered by the key their rows hold, and "the first one" follows it.

On a copy of a real inventory warehouse, the table naming warehouses held none of the warehouses the stock rows
of most snapshots held. "Which 10 warehouses have the least available stock?" ranks real members and leaves out
rows with none, so it said "No rows match", and every follow-up ("break the first one down") had nothing to point
at. A ranking that finds nothing is asked again with each member whose table has no row for its key ranked by
that key ("12 (no name)"), and says so; "the first one" then filters on that key. A ranking whose members have
names is asked once, as before. A stand-in the AI writes for the member pointed at ("first_store_id") is
replaced by that member in code, with no second call.

Synthetic data only: the retail warehouse with its stores table emptied.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest
import sqlglot

from core2.compile.compiler import compile_query
from core2.plan.ir import Plan
from core2.plan.planner import pointed_in
from core2.plan.values import MemberIndex, ValueMatch
from core2.resolve.resolver import Context, resolve
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
Q1 = {"window": {"kind": "between", "start": "2026-01-01", "end": "2026-03-31"}}
RANK = {"kind": "query", "intent": "rank", "measures": ["net_amount"], "group_by": ["store.name"],
        "sort": [{"by": "net_amount", "desc": True}], "limit": 5, "time": Q1}


@pytest.fixture(scope="module")
def unnamed():
    """The retail warehouse learned with its stores, then the stores table emptied: every sale holds a store key
    no row of it has, as a copy whose dimension tables were taken in part."""
    built, model = learn(domains.build("retail"), "descriptive")
    built.con.execute("DELETE FROM stores")
    return built, model


def _services(learned, replies: list) -> Services:
    built, model = learned
    asked: list[str] = []

    def ai(stable: str, tail: str) -> str:
        asked.append(tail)
        return json.dumps(replies.pop(0))

    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=ai, index=MemberIndex(),
                        today=TODAY)
    services.asked = asked  # type: ignore[attr-defined]
    return services


def _by_store(built, where: str = "") -> list[tuple]:
    return built.con.execute(
        "SELECT o.store_id, SUM(o.net_amount) FROM order_lines o JOIN calendar d ON o.order_date_key = d.date_key "
        f"WHERE d.full_date >= DATE '2026-01-01' AND d.full_date < DATE '2026-04-01' {where} "
        "GROUP BY 1 ORDER BY 2 DESC").fetchall()


def test_a_ranking_of_members_with_no_name_ranks_them_by_their_key_and_says_so(unnamed):
    built, model = unnamed
    payload = answer_question("Which 5 stores sold the most in Q1 2026?", _services(unnamed, [RANK]), Session())
    rows = payload["export_rows"]
    expected = _by_store(built)[:5]
    assert [r["store_name"] for r in rows] == [f"{key} (no name)" for key, _ in expected]
    assert [r["net_amount"] for r in rows] == pytest.approx([float(v) for _, v in expected])
    assert "No rows match" not in payload["answer"]["headline"]
    notes = " ".join([*(payload.get("trust", {}).get("date_context") or []), *(payload.get("notes") or [])])
    assert "shown by the key the rows hold" in notes


def test_each_warehouse_reads_the_key_as_text(unnamed):
    built, model = unnamed
    logical = resolve(Plan.model_validate(RANK), model, Context(today=TODAY, unnamed_by_key=True))
    for dialect, reads in (("tsql", "tsql"), ("snowflake", "snowflake"), ("oracle", "oracle")):
        sql = compile_query(logical, model, dialect).sql
        assert "(no name)" in sql
        sqlglot.parse_one(sql, read=reads)


def test_the_first_one_of_that_ranking_is_the_rows_holding_its_key(unnamed):
    built, model = unnamed
    first = _by_store(built)[0][0]
    # The AI writes a stand-in for the member pointed at; code puts the member the words found in its place.
    follow = {"kind": "query", "intent": "value", "measures": ["net_amount"], "time": Q1, "follow_up": "refine",
              "filters": [{"field": "store.name", "op": "eq", "values": ["first_store_id"]}]}
    services = _services(unnamed, [RANK, follow])
    session = Session()
    answer_question("Which 5 stores sold the most in Q1 2026?", services, session)
    payload = answer_question("How much did the first one sell in Q1 2026?", services, session)
    assert len(services.asked) == 2                                   # no second call to repair the stand-in
    assert f'"the first one" -> store.name = "{first} (no name)"' in services.asked[1]
    total = built.con.execute(
        "SELECT SUM(o.net_amount) FROM order_lines o JOIN calendar d ON o.order_date_key = d.date_key "
        f"WHERE o.store_id = {first} AND d.full_date >= DATE '2026-01-01' AND d.full_date < DATE '2026-04-01'"
    ).fetchone()[0]
    assert payload["export_rows"][0]["net_amount"] == pytest.approx(float(total))
    assert f"{first} (no name)" in payload["answer"]["headline"]


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")


def test_a_ranking_of_named_members_is_asked_once(retail):
    services = _services(retail, [RANK])
    payload = answer_question("Which 5 stores sold the most in Q1 2026?", services, Session())
    assert len(payload["export_rows"]) == 5 and not any("(no name)" in str(r["store_name"])
                                                          for r in payload["export_rows"])
    assert "(no name)" not in payload["trust"]["sql"]


@pytest.mark.parametrize("values, matches, kept", [
    (["first_store_id"], [("store.name", "Airport")], ["Airport"]),
    (["<top store from previous answer>"], [("store.name", "Airport")], ["Airport"]),
    (["first_store_id"], [], ["first_store_id"]),                                     # nothing found: asked about
    (["first_store_id"], [("store.name", "Airport"), ("store.name", "Harbour")], ["first_store_id"]),   # which?
    (["first_store_id"], [("region.name", "North")], ["first_store_id"]),             # another field's member
    (["Airport"], [("store.name", "Harbour")], ["Airport"]),                           # a real name stays
])
def test_a_stand_in_is_replaced_only_by_the_one_member_of_its_field(retail, values, matches, kept):
    plan = Plan.model_validate({"kind": "query", "measures": ["net_amount"],
                                "filters": [{"field": "store.name", "op": "eq", "values": values}]})
    found = [ValueMatch("the first one", 0, 13, attribute, value) for attribute, value in matches]
    assert pointed_in(plan, retail[1], found).filters[0].values == kept


@pytest.mark.parametrize("value, stand_in", [
    ("top 1 from previous answer", True),
    ("the first one in the list above", True),
    ("Top Shelf Hardware", False),               # a name that starts with a place word
    ("Previous Answer Ltd", False),
])
def test_a_row_described_in_words_is_a_stand_in(value, stand_in):
    from core2.plan.planner import stands_for

    assert stands_for(value) is stand_in


def test_a_breakdown_of_members_with_no_name_shows_them_by_their_key(unnamed):
    """ "Which warehouses hold the most of it?" with no limit came back as one row, "Unknown"."""
    built, model = unnamed
    plan = {**RANK, "intent": "breakdown"}
    plan.pop("limit")
    payload = answer_question("Net amount by store in Q1 2026", _services(unnamed, [plan]), Session())
    expected = _by_store(built)
    assert {r["store_name"]: r["net_amount"] for r in payload["export_rows"]} == pytest.approx(
        {f"{key} (no name)": float(v) for key, v in expected})


def test_another_field_of_a_member_is_never_shown_by_the_members_key(unnamed):
    """A store's region left empty is no region "12 (no name)": the key the rows hold is the store's."""
    built, model = unnamed
    logical = resolve(Plan.model_validate({**RANK, "group_by": ["region.name"]}), model,
                      Context(today=TODAY, unnamed_by_key=True))
    assert "(no name)" not in compile_query(logical, model, "duckdb").sql
