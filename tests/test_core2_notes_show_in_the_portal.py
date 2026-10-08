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

import json
import shutil
import subprocess
from pathlib import Path

import pytest

TEMPLATE = Path(__file__).resolve().parents[1] / "portal" / "templates" / "portal_chat.html"


def _render(contexts: list) -> str:
    source = TEMPLATE.read_text(encoding="utf-8")
    start = source.index("  const dateContextHtml = dateContexts.map(")
    end = source.index(".join('');", start) + len(".join('');")
    script = f"""
const LABELS = {{'ui.chat.trust.how_counted': 'How it was counted', 'ui.chat.trust.business_date': 'Business date',
  'ui.chat.trust.calendar_grain': 'calendar grain', 'ui.chat.trust.governed': 'governed date context',
  'ui.chat.trust.inferred': 'inferred encoded field'}};
const t = (key, args) => LABELS[key] || key;
const escHtml = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const dateContexts = {json.dumps(contexts)};
{source[start:end]}
process.stdout.write(dateContextHtml);
"""
    return subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True, timeout=30).stdout


pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node runs the portal's own script")


def test_a_new_core_note_is_shown_as_its_words():
    note = "Your data ends 30 Jun 2026, so recent periods are counted back from there."
    shown = _render([note, "Counted by the calendar: <fiscal> years run July to June."])
    assert shown.count("How it was counted") == 2 and note in shown
    assert "&lt;fiscal&gt;" in shown and "<fiscal>" not in shown, "a note is text, never markup"
    assert "Business date" not in shown and "governed date context" not in shown


def test_todays_date_context_is_drawn_as_before():
    shown = _render([{"label": "Invoice date", "table": "DB.SALES.INVOICES", "column": "INV_DT"}])
    assert "Business date" in shown and "Invoice date" in shown and "INVOICES.INV_DT" in shown
    assert "How it was counted" not in shown
