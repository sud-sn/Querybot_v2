"""A headline reads as a sentence: its amounts rounded as the note and the chart round them, its names in
lower case inside it, and who led a series told as it was.

Seen in the chart audit, on the retail stress data:

* "Laundry Pods Large leads with $733,018.59 across 10 products", above an analyst note that said "$733.0K" and a
  chart that said "$733K"; "up $927,076.40 (+513.3%)" beside a month of seven figures.
* "Number of orders and net Amount by store": a name in title case had only its first letter lowered.
* "Central led in Jan 2025 ($43,298.74); Central in Dec 2025 ($52,789.39)": the same region named twice, and
  nothing said of the ten months between.

Invented data only (the retail domain of evals/core2/domains.py).
"""

from __future__ import annotations

import datetime as dt
import json
import re

import pytest

from core2.answer.builder import _lower, fmt, spoken
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)


@pytest.mark.parametrize("value, written", [
    (733018.59, "$733.0K"), (927076.40, "$927.1K"), (10000, "$10.0K"), (-14772.43, "-$14.8K"),
    (9876.5, "$9,876.50"), (180.0, "$180.00"), (3093601.41, "$3.09M"), (2.4e9, "$2.40B"),
])
def test_an_amount_in_a_sentence_is_rounded_as_the_note_rounds_it(value, written):
    assert spoken(value, "currency") == written


def test_an_amount_kept_apart_by_its_currency_is_rounded_too():
    assert spoken(803429.83, "currency", unit="usd") == "USD 803.4K"


def test_the_figure_on_its_own_keeps_every_cent():
    """A one-number answer, a tile, an email's figure: the amount as it is, not the sentence's rounding."""
    assert fmt(733018.59, "currency") == "$733,018.59" and fmt(3093601.41, "currency") == "$3.09M"


def test_counts_and_rates_keep_their_figures():
    assert spoken(3806, "count") == "3,806" and spoken(54.83, "percent") == "54.8%"


@pytest.mark.parametrize("label, inside", [
    ("Net Amount", "net amount"), ("Gross Amount", "gross amount"), ("Net amount", "net amount"),
    ("Number of orders", "number of orders"), ("GMV", "GMV"), ("Avg GL Balance", "avg GL balance"),
    ("GL accounts", "GL accounts"), ("Days To Ship", "days to ship"),
])
def test_a_name_inside_a_sentence_is_lowered_word_by_word(label, inside):
    assert _lower(label) == inside


@pytest.fixture(scope="module")
def retail():
    built, model = learn(domains.build("retail"), "descriptive")
    return built.con, model


def _answer(retail, plan: dict) -> dict:
    con, model = retail
    answers = [json.dumps({"kind": "query", **plan})]
    services = Services(model=model, warehouse=DuckDBWarehouse(con), complete=lambda s, t: answers.pop(0),
                        index=MemberIndex(), today=TODAY)
    return answer_question("q", services, Session())["answer"]


def test_a_breakdown_headline_says_its_amounts_as_the_note_does(retail):
    said = _answer(retail, {"measures": ["net_amount"], "group_by": ["store.name"], "limit": 5,
                            "sort": [{"by": "net_amount", "desc": True}],
                            "time": {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}}
                   )["headline"]
    assert not re.search(r"\$\d{1,3},\d{3}\.\d\d", said), said
    assert re.search(r"\$\d+(\.\d)?K|\$\d\.\d\dM", said), said


def test_a_series_led_mostly_by_one_member_says_how_often(retail):
    said = _answer(retail, {"intent": "trend", "measures": ["net_amount"], "group_by": ["region.name"],
                            "time": {"grain": "month",
                                     "window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}}
                   )["headline"]
    found = re.search(r": (\w+) led in (\d+) of the 12 months, among the 5 regions", said)
    assert found and int(found.group(2)) > 6, said
    assert said.count(found.group(1)) == 1, "the leader is named once"


def test_a_title_lists_its_names_as_the_first_is_written():
    from core2.answer.builder import _after

    assert _after("Net Amount", "Gross Amount") == "Gross Amount"
    assert _after("Net amount", "Gross Amount") == "gross amount"
    assert _after("Number of orders", "Net Amount") == "net amount"


def test_one_member_and_nothing_to_lead_is_said_as_all_there_is(retail):
    """A reader whose rows are one region's (row access), asking by region: "North leads with" its total led nothing."""
    said = _answer(retail, {"measures": ["net_amount"], "group_by": ["region.name"],
                            "filters": [{"field": "region.name", "op": "eq", "values": ["North"]}],
                            "time": {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}}
                   )["headline"]
    assert "leads with" not in said and re.search(r": North, \$\d", said), said


def test_the_top_one_asked_for_still_leads(retail):
    said = _answer(retail, {"measures": ["net_amount"], "group_by": ["region.name"], "limit": 1,
                            "sort": [{"by": "net_amount", "desc": True}],
                            "time": {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}}
                   )["headline"]
    assert "leads with" in said or "highest" in said, said
