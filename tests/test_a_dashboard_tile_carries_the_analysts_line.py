"""A dashboard tile carries the analyst's line of its answer, in the viewer's language.

Each pinned new-core answer is drawn again when the dashboard opens; its reading (core2/answer/facts.py) comes
with it. The tile shows the reading's lead under the chart, and the dashboard's summary on top is made of the
lines that say the most.
"""

from __future__ import annotations

import core2.service as service
from portal import routes

READING = {"lead": {"en": "Eastview leads with $847.7K, 27% of the total.", "fr": "Eastview mène avec 847,7 k$."},
           "facts": [{"kind": "leader", "section": "stands_out", "size": 2.7, "en": "x", "fr": "x"}]}


def _tile(monkeypatch, lang):
    payload = {"data": {"rows": [{"store": "Eastview", "net_amount": 847.7}]}, "chart": None,
               "answer": {"badges": []}, "reading": READING, "trust": {"row_count": 1}}
    monkeypatch.setattr(service, "portal_replay", lambda *a, **k: payload)
    chart = {"id": 1, "question": "Net sales by store", "title": "Net sales by store", "chart_type": "table"}
    user = {"account_id": "acct", "id": 1, "lang": lang}
    return routes._refresh_core2_tile(chart, {}, user, {"measures": ["net_amount"], "group_by": ["store"]})


def test_the_tile_carries_its_line_in_the_viewers_language(monkeypatch):
    assert _tile(monkeypatch, "en")["note_line"] == READING["lead"]["en"]
    assert _tile(monkeypatch, "fr")["note_line"] == READING["lead"]["fr"]
    assert _tile(monkeypatch, "en")["note_size"] == 2.7


def test_the_dashboard_shows_its_summary_and_lines():
    page = open("portal/templates/portal_dashboard.html", encoding="utf-8").read()
    assert "dash-reading" in page and "chart-card-reading" in page and "ui.dash.reading_title" in page
