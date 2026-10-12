"""A scheduled dashboard email carries the analyst's lines, as the dashboard itself does.

The dashboard opens with "What this dashboard shows" and has the analyst's line under each tile (what is typical,
the one-off, the leader). The PDF is the page, so it had them; the email's own body drew each tile's title and
picture and dropped the lines, so a reader of the email saw the charts with nothing said about them. Now the
email opens with the same summary, each chart and table is followed by its line, and the plain-text email says
the same, in the reader's language; a tile's period is said in it too ("T1 2026", not "Q1 2026", for a French
reader).

The invented retail shop of the test domains.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from tests.test_a_dashboard_reads_as_the_plan import BY_REGION, TREND, _pin
from tests.test_a_new_core_answer_can_be_pinned import _services, retail  # noqa: F401
from tests.test_dashboards_are_emailed_on_a_schedule import _owner


@pytest.fixture
def app(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    import main

    main.store.init_db()
    return main


def _replay(retail):  # noqa: F811
    from core2 import service

    def replay(account_id, plan_data, portal_user, *, question=""):
        services = _services(retail, [])
        ctx = service.Context(today=services.today, allowed_tables=None, max_rows=service.MAX_ROWS)
        return service._compute(question, service.parse_plan(plan_data), services, ctx, question_id="",
                                started=time.perf_counter())
    return replay


def _snapshot(app, retail, owner):  # noqa: F811
    from core.dashboard_snapshot import take

    store = app.store
    tiles = _pin(store, retail, owner, TREND, "net amount by month", new_dashboard_name="Sales pulse")
    _pin(store, retail, owner, BY_REGION, "net amount by region in april", dashboard_id=tiles[0]["dashboard_id"])
    board = store.get_dashboard(tiles[0]["dashboard_id"], owner["id"], owner["account_id"])
    with patch("core2.service.portal_replay", _replay(retail)):
        return asyncio.run(take(board, owner, drawer=None, pdf=False))


def test_the_email_opens_with_what_the_dashboard_shows_and_says_each_tiles_line(app, retail):  # noqa: F811
    from core.mail import render

    snapshot = _snapshot(app, retail, _owner(app.store))
    lines = {t.title: t.reading for t in snapshot.tabs[0].tiles}
    assert all(lines.values()), lines                          # each tile has its analyst's line
    assert snapshot.reading and all(r["line"] in lines.values() for r in snapshot.reading)
    html = render.dashboard(snapshot, link="https://bi.example.com/x", title="Sales pulse")
    assert "What this dashboard shows" in html
    for line in lines.values():
        assert render.esc(line) in html
    text = render.dashboard_text(snapshot, link="x")
    assert "What this dashboard shows" in text and all(line in text for line in lines.values())


def test_a_french_reader_gets_the_lines_and_the_period_in_french(app, retail):  # noqa: F811
    from core.mail import render

    owner = {**_owner(app.store), "lang": "fr"}
    snapshot = _snapshot(app, retail, owner)
    html = render.dashboard(snapshot, link="https://bi.example.com/x", title="Sales pulse", lang="fr")
    assert "Ce que montre ce tableau de bord" in html
    trend = next(t for t in snapshot.tabs[0].tiles if t.reading and "mois" in t.reading)
    assert render.esc(trend.reading) in html
    assert any("avr." in t.note or "avril" in t.note for t in snapshot.tabs[0].tiles), \
        [t.note for t in snapshot.tabs[0].tiles]
