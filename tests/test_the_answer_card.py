"""The answer card, as the page builds it: the finding first, the chart in the answer.

The card was a stack of bands: a value line, the same number again in the
sentence, notes, an "Open visual" button that sent the chart to a side panel,
a row of text buttons, three large FOLLOW-UP cards, a confidence pill and a
grid of rows and milliseconds. The redesign (October 2026) reads top to bottom
as an answer does:

  badges        what it counted: the period and the date it was counted by
  the finding   one number as a tile, several as tiles, or one sentence
  the chart     inside the answer, with Chart / Table and Open larger
  what stands out
  one row of actions: Add to dashboard, copy, the verdict, ask about it,
                and "How it was counted", which opens the record in place
  up to three next questions

These tests run answerCardHtml(), the page's own pure function, on answers of
each kind and read the markup it returns; the browser check
(tools/ui_check/check.py) looks at the same cards drawn.
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser

import pytest

pytest.importorskip("dukpy")

FUNCTIONS = [
    "function answerCardHtml(msg, ids, env)",
    "function _clarifyCardHtml(msg)",
    "function _answerBadgesHtml(msg)",
    "function _ownMetricHtml(msg)",
    "function _answerTiles(msg)",
    "function _answerTilesHtml(tiles)",
    "function _shortAmount(value, fmt, spec = {})",
    "function _tileValue(value, fmt, spec = {}, column = '')",
    "function _answerChartTitle(chart)",
    "function _answerChartHeight(chart)",
    "function _answerVisualHtml(msg, ids)",
    "function _answerSummaryHtml(text)",
    "function _answerChartTypes(chart)",
    "function _answerNotes(msg)",
    "function _answerHowHtml(msg, env)",
    "function escHtml(str)",
    "function _formatDisplayValue(value, fmt, spec = {}, column = '')",
    "function _parseDisplayDate(value)",
    "function _parseDisplayNumber(v)",
    "function _isPeriodLabel(raw, column)",
    "function _normaliseColumnKey(v)",
    "function _columnFormatMap(data)",
    "function _parseServerNumber(v)",
    "function _isANumericColumn(rows, header)",
    "function _displayFormatSpec(data, column)",
    "function _followUpChip(entry)",
    "function renderChartWarnings(payload)",
    "function _qbMoney(body, symbol)",
]
PREAMBLE = """
function qbIcon(name, size) { return '<svg data-icon="' + name + '"></svg>'; }
function renderDataTable(data) { return '<div class="data-table-wrap" data-rows="' + (data.rows || []).length + '"></div>'; }
function formatBotText(text) { return escHtml(text); }
function enumLabel(group, value) { return String(value); }
"""


def card(msg: dict, *, pin: str = "", feedback: bool = False, lang: str = "en") -> str:
    from tests.chat_js import run

    script = (f"JSON.stringify(answerCardHtml({json.dumps(msg)}, "
              f"{{chartId: 'c1', rcId: 'rc-1', pinToken: {json.dumps(pin)}}}, "
              f"{{feedbackEnabled: {json.dumps(feedback)}}}));")
    return run(script, lang=lang, functions=FUNCTIONS, consts=["_NUMERIC_FORMATS", "_QB_CURRENCY_SYMBOL", "ANSWER_CHART_TYPES"], preamble=PREAMBLE)


class Elements(HTMLParser):
    """The card's elements in order, each with its classes, attributes and own text."""

    def __init__(self, markup: str):
        super().__init__()
        self.found: list[dict] = []
        self._open: list[dict] = []
        self.feed(markup)

    def handle_starttag(self, tag, attrs):
        el = {"tag": tag, "attrs": dict(attrs), "classes": (dict(attrs).get("class") or "").split(), "text": ""}
        self.found.append(el)
        if tag not in ("br", "img", "input"):
            self._open.append(el)

    def handle_endtag(self, tag):
        while self._open and self._open[-1]["tag"] != tag:
            self._open.pop()
        if self._open:
            self._open.pop()

    def handle_data(self, data):
        for el in self._open:
            el["text"] += data

    def first(self, cls: str) -> dict | None:
        return next((e for e in self.found if cls in e["classes"]), None)

    def all(self, cls: str) -> list[dict]:
        return [e for e in self.found if cls in e["classes"]]

    def order(self, *classes: str) -> list[int]:
        return [next(i for i, e in enumerate(self.found) if c in e["classes"]) for c in classes]


ONE_NUMBER = {
    "engine": "core2", "question": "What was net sales in April 2026?",
    "answer": {"headline": "Net amount in April 2026: $185,933.03.", "short_value": "$185,933.03",
               "comparison": "in April 2026",
               "badges": [{"kind": "period", "text": "April 2026"}, {"kind": "date", "text": "by order date"}]},
    "kpi": {"label": "Net amount", "value": 185933.03, "format": "currency", "display_format": {}, "state": "ready",
            "note": "in April 2026"},
    "chart": None,
    "data": {"headers": ["net_amount"], "rows": [{"net_amount": 185933.03}], "column_formats": {"net_amount": "currency"}},
    "key_insights": ["Down $8,943.45 (-4.6%) on March 2026, the period before."],
    "follow_up_suggestions": [{"label": "Why did it drop?", "question": "Why did net amount change in April 2026?"}],
    "trust": {"engine": "core2", "sql": "SELECT 1", "row_count": 1, "question_id": "q1",
              "date_context": ["Net amount by order date."]},
}

BY_STORE = {
    "engine": "core2", "question": "Net sales by store in the first half of 2026",
    "answer": {"headline": "Net amount from Jan 2026 to Jun 2026: Old Town Store leads with $124,495.93 (12% of the "
                           "total) across 12 stores.",
               "badges": [{"kind": "period", "text": "Jan–Jun 2026"}, {"kind": "date", "text": "by order date"}],
               "scope_note": "Jun 2026 is only partly covered by the data: not compared as whole periods."},
    "coverage_caveats": ["Jun 2026 is only partly covered by the data: not compared as whole periods."],
    "kpi": None,
    "chart": {"title": "Net amount", "chart_type": "bar", "x_key": "store_name", "y_keys": ["net_amount"],
              "rows": [{"store_name": f"Store {i}", "net_amount": 100 - i} for i in range(12)],
              "column_roles": {"store_name": {"label": "Store", "role": "dimension"}}},
    "data": {"headers": ["store_name", "net_amount"],
             "rows": [{"store_name": f"Store {i}", "net_amount": 100 - i} for i in range(12)]},
    "key_insights": ["The top 3 of the 12 stores make 33% of the total."],
    "follow_up_suggestions": [{"label": f"Next {i}", "question": f"Next question {i}"} for i in range(5)],
    "trust": {"engine": "core2", "sql": "SELECT store", "row_count": 12, "duration_label": "41ms",
              "question_id": "q2", "date_context": ["Net amount by order date."]},
}

THREE_NUMBERS = {
    "engine": "core2", "question": "Net sales, cost and gross amount in 2025",
    "answer": {"headline": "Net amount in 2025: $3.09M; Cost amount $1.92M, Gross amount $3.15M.",
               "badges": [{"kind": "period", "text": "2025"}]},
    "kpi": None, "chart": None,
    "data": {"headers": ["net_amount", "cost_amount", "gross_amount"],
             "header_labels": {"net_amount": "Net amount", "cost_amount": "Cost amount", "gross_amount": "Gross amount"},
             "rows": [{"net_amount": 3093601.41, "cost_amount": 1918032.53, "gross_amount": 3150534.87}],
             "column_formats": {"net_amount": "currency", "cost_amount": "currency", "gross_amount": "currency"}},
    "trust": {"engine": "core2", "sql": "SELECT 1"},
}


def test_the_card_reads_badges_finding_chart_findings_actions_next():
    shown = Elements(card(BY_STORE, pin="tok"))
    assert shown.order("answer-badges", "answer-headline", "answer-visual", "answer-insights", "answer-actions",
                       "answer-next") == sorted(shown.order("answer-badges", "answer-headline", "answer-visual",
                                                             "answer-insights", "answer-actions", "answer-next"))
    assert [b["text"] for b in shown.all("badge")] == ["Jan–Jun 2026", "by order date"]


def test_the_chart_is_inside_the_answer_with_chart_table_and_open_larger():
    markup = card(BY_STORE)
    shown = Elements(markup)
    visual = shown.first("answer-visual")
    assert visual is not None, "the chart left the answer"
    assert shown.first("answer-chart")["attrs"]["id"] == "c1"
    assert "Net amount by store" in shown.first("answer-visual-title")["text"]
    assert [b["text"] for b in shown.found if b["attrs"].get("data-view")] == ["Chart", "Table"]
    assert any("data-expand" in e["attrs"] for e in shown.found), "no way to the larger view"
    assert "artifact-open-btn" not in markup, "the chart is not sent away behind a button any more"


def test_twelve_bars_get_the_height_of_twelve():
    height = int(re.search(r'id="c1" style="height:(\d+)px"', card(BY_STORE)).group(1))
    assert height >= 12 * 28


def test_one_number_is_a_tile_and_is_said_once():
    markup = card(ONE_NUMBER)
    shown = Elements(markup)
    tile = shown.first("kpi-tile")
    assert tile is not None
    assert shown.first("kpi-value")["text"] == "$185,933.03"
    assert shown.first("kpi-label")["text"] == "Net amount"
    assert markup.count("185,933.03") == 1, "the number is said twice"
    assert shown.first("answer-visual") is None, "one number needs no chart"


def test_several_numbers_with_nothing_to_group_by_are_one_tile_each():
    shown = Elements(card(THREE_NUMBERS))
    assert [e["text"] for e in shown.all("kpi-label")] == ["Net amount", "Cost amount", "Gross amount"]
    # An amount of a million or more is drawn short in its tile, so it never runs out of it; the exact
    # amount is right under it.
    assert [e["text"] for e in shown.all("kpi-value")] == ["$3.09M", "$1.92M", "$3.15M"]
    assert [e["text"] for e in shown.all("kpi-exact")] == ["$3,093,601.41", "$1,918,032.53", "$3,150,534.87"]
    assert shown.first("answer-tiles")["attrs"]["data-count"] == "3"


def test_a_caveat_is_said_once_next_to_the_finding():
    markup = card(BY_STORE)
    assert markup.count("only partly covered") == 1
    shown = Elements(markup)
    assert shown.order("answer-notes")[0] < shown.order("answer-visual")[0]


def test_what_stands_out_is_the_findings_under_their_own_label():
    shown = Elements(card(BY_STORE))
    insights = shown.first("answer-insights")
    assert shown.first("answer-insights-label")["text"] == "What stands out"
    assert "The top 3 of the 12 stores make 33% of the total." in insights["text"]
    assert "only partly covered" not in insights["text"], "a caveat is not a finding"


def test_one_row_of_actions():
    shown = Elements(card(BY_STORE, pin="tok", feedback=True))
    row = shown.first("answer-actions")
    names = [e["attrs"].get("aria-label") or e["text"].strip() for e in shown.found
             if e["tag"] == "button" and e in shown.found[shown.found.index(row):]
             and (e["attrs"].get("data-pin") or e["attrs"].get("data-copy") or e["attrs"].get("data-feedback")
                  or e["attrs"].get("data-rc-toggle") or "data-open-how" in e["attrs"])]
    assert names == ["Add to dashboard", "Copy result", "Yes, correct answer", "Not helpful",
                     "Ask about this result", "How it was counted"]


def test_no_pin_without_a_token_and_no_thumbs_without_feedback():
    shown = Elements(card(BY_STORE))
    assert not any(e["attrs"].get("data-pin") for e in shown.found)
    assert not any(e["attrs"].get("data-feedback") for e in shown.found)


def test_at_most_three_next_questions_as_compact_chips():
    shown = Elements(card(BY_STORE))
    chips = shown.all("follow-up-chip")
    assert [c["text"] for c in chips] == ["Next 0", "Next 1", "Next 2"]
    assert all(c["attrs"]["data-question"].startswith("Next question") for c in chips)
    assert "Follow-up" not in card(BY_STORE), "no FOLLOW-UP label on every chip"


def test_how_it_was_counted_holds_the_notes_and_the_record():
    shown = Elements(card(BY_STORE))
    how = shown.first("answer-how")
    assert "Net amount by order date." in how["text"]
    assert "SELECT store" in how["text"]
    assert "41ms" in how["text"]
    assert "open" not in how["attrs"], "the record is a click away, not open on every answer"


def test_a_finding_is_text_never_markup():
    msg = dict(BY_STORE, key_insights=["A <b> & B are close."])
    markup = card(msg)
    assert "A &lt;b&gt; &amp; B are close." in markup and "<b>" not in markup


def test_an_answer_with_nothing_to_show_is_its_sentence():
    msg = {"engine": "core2", "question": "hello", "answer": {"headline": "Hello! Ask me about your data."},
           "trust": {"engine": "core2"}}
    shown = Elements(card(msg))
    assert shown.first("answer-headline")["text"] == "Hello! Ask me about your data."
    assert shown.first("answer-visual") is None and shown.first("answer-tiles") is None
    assert shown.first("answer-badges") is None


def test_the_card_speaks_french():
    shown = Elements(card(BY_STORE, pin="tok", lang="fr"))
    assert shown.first("answer-insights-label")["text"] == "Ce qui ressort"
    assert [b["text"] for b in shown.found if b["attrs"].get("data-view")] == ["Graphique", "Tableau"]


def test_a_chart_is_named_by_what_it_counts_not_by_a_column():
    msg = dict(BY_STORE, chart=dict(BY_STORE["chart"], column_roles={"store_name": {"label": "Store name",
                                                                                     "role": "dimension"}}))
    assert Elements(card(msg)).first("answer-visual-title")["text"] == "Net amount by store"


def test_a_tile_does_not_repeat_the_period_its_badge_names():
    shown = Elements(card(ONE_NUMBER))
    assert shown.first("kpi-note") is None
    compared = dict(ONE_NUMBER, answer=dict(ONE_NUMBER["answer"], comparison="down $8,943.45 (-4.6%) on March 2026"))
    assert Elements(card(compared)).first("kpi-note")["text"] == "down $8,943.45 (-4.6%) on March 2026"


def test_a_question_back_is_a_question_card_with_its_choices():
    msg = {"engine": "core2", "question": "how many customers in April",
           "answer": {"headline": "Which customers: who ordered or who returned something?"},
           "clarify": {"about": "date", "question": "Which customers should I count?",
                       "options": ["Customers who ordered", "Customers who returned something"]},
           "follow_up_suggestions": [{"label": "Customers who ordered", "question": "Customers who ordered"}],
           "trust": {"engine": "core2"}}
    shown = Elements(card(msg))
    assert shown.first("clarify-question")["text"] == "Which customers should I count?"
    choices = shown.all("clarify-choice")
    assert [c["attrs"]["data-question"] for c in choices] == ["Customers who ordered", "Customers who returned something"]
    assert shown.first("answer-actions") is None, "a question has nothing to pin or count yet"
    assert len(shown.all("follow-up-chip")) == 2, "each choice once, not again as a chip"
