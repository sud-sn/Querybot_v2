"""An everyday word in a question is not a member that happens to be coded the same.

On an inventory workspace, groups are coded DO, AS, IF and GO, a province is ON and a unit
ME. The new core matched the question's words against every member name, in any case, so
"How many items do we have?" was handed to the planner as pdc group = DO and "Inventory
value on 3 September 2026" as province = ON: each answer was filtered to nothing ("0 items
for pdc group DO", "no value ... for province ON"), and the filters carried into the next
question. A member that is also an everyday word now matches only where the question writes
it as stored ("inventory for DO"), never in its ordinary sense, and never in a question
written all in capitals, where case says nothing.

The same answer said the change between two empty dates was "$0.00": with nothing on
either side, the change is no value. And a tile with no number was drawn empty; it reads
as a dash.

Invented codes and synthetic data only.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.values import MemberIndex


@pytest.fixture
def index() -> MemberIndex:
    out = MemberIndex()
    out.add("pdc_group.code", ["DO", "AS", "IF", "GO", "01A"])
    out.add("profit_centre.province", ["ON", "QC", "NS"])
    out.add("item.unit", ["ME", "EA"])
    out.add("store.name", ["Top Value Stores", "Do It Centre"])
    return out


def _found(index: MemberIndex, question: str) -> list[tuple[str, str]]:
    return [(m.attribute, m.value) for m in index.match(question)]


@pytest.mark.parametrize("question", [
    "How many items do we have?",
    "Inventory value on 3 September 2026 compared with 17 August 2026",
    "What's the total value of our stock as of today?",
    "If stock is low, which items go first?",
    "Show me the items on hand",
])
def test_an_everyday_word_is_never_read_as_a_member_code(index, question):
    assert _found(index, question) == []


@pytest.mark.parametrize("question,expected", [
    ("Inventory value for DO", [("pdc_group.code", "DO")]),
    ("Sales in ON last month", [("profit_centre.province", "ON")]),
    ("Sales in QC and in ns", [("profit_centre.province", "QC"), ("profit_centre.province", "NS")]),
    ("on hand quantity for group 01A", [("pdc_group.code", "01A")]),
    ("net sales for top value stores", [("store.name", "Top Value Stores")]),
    ("stock at do it centre", [("store.name", "Do It Centre")]),
])
def test_a_member_written_as_stored_or_by_its_whole_name_is_still_found(index, question, expected):
    assert _found(index, question) == expected


def test_in_a_question_all_in_capitals_case_says_nothing(index):
    assert _found(index, "HOW MANY ITEMS DO WE HAVE") == []
    assert _found(index, "SALES FOR QC") == [("profit_centre.province", "QC")]      # not an everyday word


def test_the_planner_is_handed_no_member_for_an_everyday_word(index):
    from core2.plan.planner import question_tail

    tail = question_tail("How many items do we have?", today=dt.date(2026, 10, 9), history=[],
                         matches=index.match("How many items do we have?"))
    assert "VALUE MATCHES" not in tail


# ── a quarter is a period ────────────────────────────────────────────────────

@pytest.fixture
def quarters() -> MemberIndex:
    """Groups coded like periods, as on an inventory workspace (Q1 to Q4, H1, H2)."""
    out = MemberIndex()
    out.add("pdc_group.group_code", ["Q1", "Q2", "H1", "FY26", "AB"])
    return out


@pytest.mark.parametrize("question", [
    "How many deliveries, receipts and returns did we have in Q1 2026?",
    "Number of receipts in q2",
    "Sales for H1 against last year",
    "Receipts in FY26 by month",
])
def test_a_quarter_half_or_fiscal_year_is_the_period_not_a_group_coded_the_same(quarters, question):
    assert _found(quarters, question) == []


@pytest.mark.parametrize("question,expected", [
    ("Deliveries for pdc group Q1", [("pdc_group.group_code", "Q1")]),
    ("deliveries for group Q2 in Q1 2026", [("pdc_group.group_code", "Q2")]),
    ("receipts for group code H1", [("pdc_group.group_code", "H1")]),
    ("receipts for AB in Q1", [("pdc_group.group_code", "AB")]),
])
def test_a_group_coded_like_a_period_is_found_where_its_field_is_named(quarters, question, expected):
    assert _found(quarters, question) == expected


# ── a change between two empty dates ─────────────────────────────────────────

@pytest.fixture(scope="module")
def retail():
    from evals.core2 import domains
    from evals.core2.compile_eval import learn

    return learn(domains.build("retail"), "descriptive")


def _rows(retail, plan: dict) -> list[dict]:
    from core2.compile.compiler import compile_query
    from core2.plan.ir import Plan
    from core2.resolve.resolver import Context, resolve
    from core2.warehouse.runner import DuckDBWarehouse

    built, model = retail
    warehouse = DuckDBWarehouse(built.con)
    logical = resolve(Plan.model_validate({"kind": "query", **plan}), model, Context(today=dt.date(2026, 6, 15)))
    result = warehouse.query(compile_query(logical, model, warehouse.dialect).sql)
    return [dict(zip(result.columns, row)) for row in result.rows]


def _day(day: str) -> dict:
    return {"kind": "between", "start": day, "end": day}


def test_with_nothing_on_either_date_the_change_is_no_value(retail):
    # A store filter no row has: both sides empty.
    (row,) = _rows(retail, {"intent": "compare", "measures": ["net_amount"],
                            "filters": [{"field": "store", "op": "eq", "values": ["No Such Store"]}],
                            "time": {"window": _day("2026-03-03"), "compare": {"kind": "window",
                                                                                 "window": _day("2026-02-17")}}})
    change = next(v for k, v in row.items() if k.endswith("_change") and not k.endswith("pct_change"))
    assert [v for k, v in row.items() if not k.endswith("change")] == [None, None] and change is None


def test_with_one_side_empty_the_change_is_all_of_the_other(retail):
    rows = _rows(retail, {"intent": "compare", "measures": ["net_amount"], "group_by": ["store"],
                          "time": {"window": {"kind": "between", "start": "2026-03-01", "end": "2026-03-31"},
                                   "compare": {"kind": "window", "window": {"kind": "between", "start": "2010-01-01",
                                                                            "end": "2010-01-31"}}}})
    assert rows
    for row in rows:
        values = {k: v for k, v in row.items() if not isinstance(v, str)}
        current = next(v for k, v in values.items() if not k.endswith(("_prior", "change")))
        change = next(v for k, v in values.items() if k.endswith("_change") and not k.endswith("pct_change"))
        assert change == pytest.approx(current)


# ── a tile with no number ────────────────────────────────────────────────────

def test_a_tile_with_no_number_reads_as_a_dash():
    from tests.chat_js import run
    from tests.test_the_answer_card import FUNCTIONS, PREAMBLE

    msg = {"answer": {"headline": "Inventory value on 3 Sep 2026 against 17 Aug 2026."},
           "data": {"headers": ["value", "value_prior", "value_change"],
                    "header_labels": {"value": "Inventory value", "value_prior": "Inventory value (before)",
                                      "value_change": "Inventory value: change"},
                    "column_formats": {"value": "currency", "value_prior": "currency", "value_change": "currency"},
                    "rows": [{"value": None, "value_prior": 1200.5, "value_change": None}]}}
    tiles = run(f"JSON.stringify(_answerTiles({json.dumps(msg)}))", functions=FUNCTIONS,
                consts=["_NUMERIC_FORMATS", "_QB_CURRENCY_SYMBOL"], preamble=PREAMBLE)
    assert [t["value"] for t in tiles][0] == "—" and tiles[2]["value"] == "—"
    assert tiles[1]["value"].startswith("$1,200")
