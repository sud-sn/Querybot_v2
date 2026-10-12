"""An answer's note reads like an analyst's: what is typical, the one-off, the leader -- and every figure checked.

The findings under an answer used to compare the two ends of the window ("Dec 2025 is 31% above Jan 2025") and
an average one bulk month had pushed up. The facts now set the one-off apart and say the typical month; a
comparison of the last three months against the three before leaves the one-off out (it would be all of the
change); a yearly total says the one-off is in it. The workspace's AI writes the note from the facts alone, and
a sentence of its own with a figure the facts do not hold, or a cause no fact names, is dropped.

An invented shop: four stores, a steady month near 10,000 and one month of 60,000.
"""

from __future__ import annotations

import datetime as dt
import json

import duckdb
import pytest

from core2.answer import notes
from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse

STORES = {"North": 0.4, "South": 0.3, "East": 0.2, "West": 0.1}
MONTHS = [dt.date(2025, m, 1) for m in range(1, 13)]
TOTAL = {d: (60_000 if d.month == 8 else 10_000 + 150 * (d.month % 4)) for d in MONTHS}


@pytest.fixture(scope="module")
def shop():
    con = duckdb.connect()
    con.execute("CREATE TABLE sales (sale_id INTEGER, sale_date DATE, store VARCHAR, net_amount DECIMAL(12,2))")
    rows, n = [], 0
    for month in MONTHS:
        for store, share in STORES.items():
            for day in (3, 11, 19, 27):     # four sales a store a month, the month's share split evenly
                n += 1
                rows.append((n, month.replace(day=day), store, round(TOTAL[month] * share / 4, 2)))
    con.executemany("INSERT INTO sales VALUES (?, ?, ?, ?)", rows)
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="t-analyst", options=BuildOptions(workers=1))
    return con, model, "net_amount"


def _payload(shop, plan):
    con, model, _ = shop
    services = Services(model=model, warehouse=DuckDBWarehouse(con),
                        complete=lambda s, t: json.dumps({"kind": "query", **plan}), index=MemberIndex(),
                        today=dt.date(2026, 6, 14))
    return answer_question("q", services, Session())


def _trend(shop):
    return _payload(shop, {"intent": "trend", "measures": [shop[2]], "time": {
        "grain": "month", "window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}})


def _kinds(payload):
    return {f["kind"]: f for f in payload["reading"]["facts"]}


def test_a_trend_sets_the_one_off_apart_and_says_the_typical_month(shop):
    payload = _trend(shop)
    facts = _kinds(payload)
    assert "Aug" in facts["one_off"]["en"] and "6" in facts["one_off"]["en"]          # 60K is ~6 times the usual
    assert facts["level"]["en"].startswith("Net amount held near $10.")
    assert "one-off" in payload["reading"]["lead"]["en"]
    assert [o["label"] for o in payload["reading"]["one_offs"]] == ["Aug 2025"]


def test_the_last_three_months_leave_the_one_off_out(shop):
    pace = _kinds(_trend(shop))["pace"]
    # Oct-Dec against Jul-Sep without August: averages, nearly level -- not "59% down" from the one-off.
    assert pace["en"].startswith("Leaving out the one-off")
    assert "down" not in pace["en"].split("(")[0] or "%" in pace["en"]
    assert all(abs(v) < 60_000 for v in pace["figures"])


def test_a_breakdown_names_the_leader_and_its_share(shop):
    payload = _payload(shop, {"intent": "breakdown", "measures": [shop[2]], "group_by": ["sale.store"], "time": {
        "grain": None, "window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}})
    text = " ".join(f["en"] for f in payload["reading"]["facts"])
    assert "North" in text and "40%" in text
    assert "North" in payload["reading"]["lead"]["en"]


def test_each_fact_says_how_it_was_worked_out_in_both_languages(shop):
    reading = _trend(shop)["reading"]
    for fact in reading["facts"]:
        assert fact["en"] and fact["fr"] and fact["how"] and fact["how_fr"]
        assert fact["section"] in notes.SECTIONS
    note = notes.template(reading, "fr")
    assert note["how"] and all(h["how"] in {f["how_fr"] for f in reading["facts"]} for h in note["how"])
    assert not any("median" in h["how"] or "whole" in h["how"] for h in note["how"])


def test_the_note_keeps_only_sentences_the_facts_hold(shop):
    reading = _trend(shop)["reading"]
    said = {"stands_out": ["August 2025 was a one-off at $60K, about 6 times a typical month.",
                           "Sales reached $75,000 in August.",                      # a figure no fact holds
                           "Sales were steady because the stores kept their prices."],  # a cause no fact names
            "why": [], "watch": ["Read the median month, not the average."]}
    kept = notes.checked(said, reading)
    assert kept["stands_out"] == ["August 2025 was a one-off at $60K, about 6 times a typical month."]
    assert kept["watch"] == ["Read the median month, not the average."]


def test_the_ai_writes_the_note_and_the_facts_stand_in_when_it_cannot(shop):
    reading = _trend(shop)["reading"]
    good = json.dumps({"stands_out": ["August 2025 was a one-off at $60K."], "why": [], "watch": []})
    note = notes.write(reading, lambda s, t: good)
    assert note["by"] == "ai" and note["stands_out"] == ["August 2025 was a one-off at $60K."]
    assert note["watch"] == notes.template(reading)["watch"]       # a section the AI left empty keeps the facts
    for complete in (None, lambda s, t: "not json", lambda s, t: json.dumps({"stands_out": ["It came to $43,210."]})):
        assert notes.write(reading, complete)["by"] == "facts"
    assert notes.write(reading, lambda s, t: good, personal=True)["by"] == "facts"


def test_the_ai_is_given_the_facts_alone_in_the_readers_language(shop):
    reading = _trend(shop)["reading"]
    seen = []
    notes.write(reading, lambda s, t: seen.append((s, t)) or "{}", lang="fr")
    system, given = seen[0]
    assert "French" in system
    assert given.startswith("Facts:") and "exceptionnel" in given and "sale_id" not in given


_CARD = {"engine": "core2", "question": "Net sales by month in 2025",
         "answer": {"headline": "Net sales from Jan 2025 to Dec 2025, by month"},
         "summary": "An older summary.", "key_insights": ["An older finding."],
         "reading": {"lead": {"en": "Sales held near $10K a month.", "fr": "Les ventes sont restées proches de 10 k$."},
                     "facts": [{"kind": "one_off", "section": "stands_out", "size": 3.0,
                                "en": "August 2025 ($60K) is a one-off.", "fr": "Août 2025 (60 k$) est exceptionnel.",
                                "how": "Outside the median by more than 3 times the usual spread."},
                               {"kind": "average", "section": "watch", "size": 1.0, "how": "",
                                "en": "Read the median month, not the average.",
                                "fr": "Lisez le mois médian, pas la moyenne."}]}}


def test_the_card_leads_with_the_analysts_sentence_and_shows_the_note():
    from tests.test_the_answer_card import card

    html = card(_CARD)
    assert "Sales held near $10K a month." in html
    assert "Analyst note" in html and "August 2025 ($60K) is a one-off." in html and "What to watch" in html
    assert "How this was worked out" in html and "Outside the median" in html
    assert "An older summary." not in html and "An older finding." not in html


def test_the_note_is_in_the_readers_language():
    from tests.test_the_answer_card import card

    html = card(_CARD, lang="fr")
    assert "Note d'analyse" in html and "Août 2025 (60 k$) est exceptionnel." in html
    assert "Les ventes sont restées proches de 10 k$." in html


def test_the_ais_note_kept_with_the_answer_is_the_one_shown():
    from tests.test_the_answer_card import card

    note = {"stands_out": ["August was a one-off; a typical month is near $10K."], "why": [], "watch": [], "how": []}
    html = card({**_CARD, "note": note})
    assert "August was a one-off; a typical month is near $10K." in html
    assert "August 2025 ($60K) is a one-off." not in html
