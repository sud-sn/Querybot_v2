"""The new core's notes are read in the portal, not drawn as empty date rows.

The new core sends what it assumed and how it counted ("Your data ends 30 Jun
2026, so recent periods are counted back from there", "Counted by the calendar:
fiscal years here run July to June") as sentences in ``trust.date_context``.
The portal drew every entry there as today's date object: each sentence became
"Business date: Business date · calendar grain · · governed date context", and
its words were never shown. A sentence is now shown as itself, under "How it
was counted"; today's date objects are drawn as before.
"""

from __future__ import annotations

import pytest

pytest.importorskip("dukpy")


def _render(contexts: list) -> str:
    """The "How it was counted" record of an answer carrying these date contexts, as the page builds it."""
    from tests.test_the_answer_card import Elements, card

    msg = {"engine": "core2", "answer": {"headline": "Net amount in 2026: $1."},
           "trust": {"engine": "core2", "sql": "SELECT 1", "date_context": contexts}}
    markup = card(msg)
    start = markup.index('<details class="trust-box')
    return markup[start:markup.index("</details>", start)], Elements(markup)


def test_a_new_core_note_is_shown_as_its_words():
    note = "Your data ends 30 Jun 2026, so recent periods are counted back from there."
    shown, elements = _render([note, "Counted by the calendar: <fiscal> years run July to June."])
    notes = elements.first("answer-how-notes")
    assert notes is not None and note in notes["text"]
    assert "&lt;fiscal&gt;" in shown and "<fiscal>" not in shown, "a note is text, never markup"
    assert "Business date" not in shown and "governed date context" not in shown


def test_todays_date_context_is_drawn_as_before():
    shown, _ = _render([{"label": "Invoice date", "table": "DB.SALES.INVOICES", "column": "INV_DT"}])
    assert "Business date" in shown and "Invoice date" in shown and "INVOICES.INV_DT" in shown
