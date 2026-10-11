"""
The "What changed" tile: each of a dashboard's numbers against the period before, and the members that moved it most,
for whoever opens the dashboard; and a few sentences by the workspace's AI from those figures alone.

    the figures   from the dashboard's own number tiles, each asked again as "why did it change" on the tile's
                  governed path: the reader's own numbers, never the whole when their rows are fewer
    the AI        writes from the figures only; a sentence with any figure not among them is not shown; a workspace
                  that keeps member names from its AI sends it nothing; asked once per change, not once per visit
    the tile      added from the dashboard's menu at the top of the tab, once; the other tiles move down; in the
                  scheduled email too
    who           whoever arranges their own dashboards (an Asker or a Creator); a Viewer is refused

Invented data only (the retail domain); the AI is a stand-in.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from tests.test_a_dashboard_reads_as_the_plan import BY_REGION, TWO, _pin
from tests.test_a_new_core_answer_can_be_pinned import _services, retail  # noqa: F401
from tests.test_dashboards_are_emailed_on_a_schedule import _owner, _person


@pytest.fixture
def app(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    import main

    main.store.init_db()
    return main


def _replay(retail, west_only=frozenset()):  # noqa: F811
    """The tiles as each reader may see them (those in ``west_only``: the West region alone, as a row rule would
    have it), timed from when they start, as the portal's own replay is."""
    from core2 import service

    def replay(account_id, plan_data, portal_user, *, question=""):
        if portal_user and int(portal_user.get("id") or 0) in west_only:
            plan_data = {**plan_data, "filters": list(plan_data.get("filters") or []) +
                         [{"field": "region.name", "op": "eq", "values": ["West"]}]}
        services = _services(retail, [])
        ctx = service.Context(today=services.today, allowed_tables=None, max_rows=service.MAX_ROWS)
        return service._compute(question, service.parse_plan(plan_data), services, ctx, question_id="",
                                started=time.perf_counter())
    return replay


class _AI:
    """The workspace's AI, standing in: says what it is given to say, and counts its calls."""

    def __init__(self, reply: str):
        self.reply, self.calls, self.given = reply, 0, ""

    def writer(self, account_id, client, *, question="", question_id=""):
        def complete(system, given):
            self.calls += 1
            self.given = given
            return self.reply
        return complete


def _board(app, retail, owner):  # noqa: F811
    store = app.store
    tiles = _pin(store, retail, owner, TWO, "net and gross in april", new_dashboard_name="Sales pulse")
    _pin(store, retail, owner, BY_REGION, "net amount by region in april", dashboard_id=tiles[0]["dashboard_id"])
    return store.get_dashboard(tiles[0]["dashboard_id"], owner["id"], owner["account_id"])


def _signed_in(app, user):
    client = TestClient(app.app, base_url="https://testserver")
    password = "owner-pass-123" if user["name"] == "Sam Owner" else "reader-pass-123"
    assert client.post("/portal/login", data={"account_id": user["account_id"], "email": user["email"],
                                              "password": password}, follow_redirects=False).status_code == 303
    return client


def _add(app, user, board):
    return _signed_in(app, user).post(f"/portal/dashboard/{board['id']}/what-changed", data={"tab": "Overview"},
                                      follow_redirects=False)


def _tile(store, board, owner):
    import json

    return next(c for c in store.list_dashboard_charts(board["id"], owner["id"])
                if json.loads(c.get("display_config") or "{}").get("what_changed"))


def _draw(retail, store, board, owner, reader, *, ai=None, west_only=frozenset()):  # noqa: F811
    from core import what_changed

    with patch("core2.service.portal_replay", _replay(retail, set(west_only))), \
            patch("core2.bootstrap.ai.summary_writer", (ai or _AI("")).writer):
        return what_changed.draw(_tile(store, board, owner), reader)


class TestTheFigures:

    def test_each_number_against_the_period_before_and_what_moved_it(self, app, retail):  # noqa: F811
        store = app.store
        owner = _owner(store)
        board = _board(app, retail, owner)
        assert _add(app, owner, board).status_code == 303
        drawn = _draw(retail, store, board, owner, owner)
        net, gross = drawn["figures"]
        assert (net["title"], net["value"], net["direction"]) == ("Net amount", "$185,933.03", "down")
        assert net["comparison"] == "down $8,943.45 (-4.6%) on March 2026"
        assert net["grouping"] == "Store" and net["leaders"][0] == {"member": "Riverside Store", "change": "-$9,982.82"}
        assert gross["title"] == "Gross amount"

    def test_each_reader_sees_their_own_figures(self, app, retail):  # noqa: F811
        store = app.store
        owner = _owner(store)
        board = _board(app, retail, owner)
        _add(app, owner, board)
        wes = _person(store, owner, "Wes West")
        store.add_dashboard_share(board["id"], owner["id"], owner["account_id"], "user", wes["id"])
        mine = _draw(retail, store, board, owner, owner)["figures"][0]
        his = _draw(retail, store, board, owner, wes, west_only={wes["id"]})["figures"][0]
        assert his["value"] != mine["value"] and "$185,933.03" not in str(his), "West's own number, never the whole"

    def test_a_tab_with_no_number_says_so(self, app, retail):  # noqa: F811
        store = app.store
        owner = _owner(store)
        tiles = _pin(store, retail, owner, BY_REGION, "net amount by region in april", new_dashboard_name="Regions")
        board = store.get_dashboard(tiles[0]["dashboard_id"], owner["id"], owner["account_id"])
        _add(app, owner, board)
        assert _draw(retail, store, board, owner, owner) == {"figures": [], "summary": "", "empty": True}


class TestTheAI:

    def test_it_writes_from_the_figures_alone_and_is_asked_once_per_change(self, app, retail):  # noqa: F811
        store = app.store
        owner = _owner(store)
        board = _board(app, retail, owner)
        _add(app, owner, board)
        ai = _AI("Net amount fell $8,943.45 on March 2026, most of it at Riverside Store.")
        assert _draw(retail, store, board, owner, owner, ai=ai)["summary"].startswith("Net amount fell $8,943.45")
        assert "Riverside Store -$9,982.82" in ai.given and "Gross amount" in ai.given
        _draw(retail, store, board, owner, owner, ai=ai)
        assert ai.calls == 1, "kept until the figures change"

    def test_a_sentence_with_a_figure_of_its_own_is_not_shown(self, app, retail):  # noqa: F811
        store = app.store
        owner = _owner(store)
        board = _board(app, retail, owner)
        _add(app, owner, board)
        ai = _AI("Net amount fell $12,400, its worst month since 2019.")
        assert _draw(retail, store, board, owner, owner, ai=ai)["summary"] == ""

    def test_a_workspace_that_keeps_member_names_from_its_ai_sends_it_nothing(self, app, retail):  # noqa: F811
        store = app.store
        owner = _owner(store)
        store.save_compliance_profile(owner["account_id"], mode="regulated", industry="healthcare_pharmacy")
        board = _board(app, retail, owner)
        _add(app, owner, board)
        ai = _AI("Anything.")
        drawn = _draw(retail, store, board, owner, owner, ai=ai)
        assert drawn["figures"] and drawn["summary"] == "" and ai.calls == 0


class TestTheTile:

    def test_it_goes_at_the_top_once_and_the_others_move_down(self, app, retail):  # noqa: F811
        store = app.store
        owner = _owner(store)
        board = _board(app, retail, owner)
        before = {c["id"]: int(c.get("grid_y") or 0) for c in store.list_dashboard_charts(board["id"], owner["id"])}
        _add(app, owner, board)
        _add(app, owner, board)
        charts = store.list_dashboard_charts(board["id"], owner["id"])
        tile = _tile(store, board, owner)
        assert len(charts) == len(before) + 1, "once a tab"
        assert (tile["grid_x"], tile["grid_y"], tile["grid_w"]) == (0, 0, 12)
        assert all(int(c["grid_y"]) >= before[c["id"]] + int(tile["grid_h"]) for c in charts if c["id"] in before)

    def test_the_page_shows_it_and_a_viewer_cannot_add_one(self, app, retail):  # noqa: F811
        store = app.store
        owner = _owner(store)
        board = _board(app, retail, owner)
        _add(app, owner, board)
        with patch("core2.service.portal_replay", _replay(retail)):
            page = _signed_in(app, owner).get(f"/portal/dashboard?dashboard_id={board['id']}").text
        assert 'class="dash-changed"' in page and "Riverside Store" in page and "Most of it by store:" in page
        viv = _person(store, owner, "Viv Viewer")
        store.add_dashboard_share(board["id"], owner["id"], owner["account_id"], "user", viv["id"])
        tiles = len(store.list_dashboard_charts(board["id"], owner["id"]))
        assert _add(app, viv, board).status_code == 403, "the role gate (portal/access.py): own_dashboards"
        assert len(store.list_dashboard_charts(board["id"], owner["id"])) == tiles

    def test_it_is_in_the_scheduled_email(self, app, retail):  # noqa: F811
        import asyncio

        from core.dashboard_snapshot import take
        from core.mail import render

        store = app.store
        owner = _owner(store)
        board = _board(app, retail, owner)
        _add(app, owner, board)
        with patch("core2.service.portal_replay", _replay(retail)), \
                patch("core2.bootstrap.ai.summary_writer", _AI("").writer):
            snapshot = asyncio.run(take(board, owner, drawer=None, pdf=False))
        tile = next(t for t in snapshot.tabs[0].tiles if t.kind == "changed")
        html = render.dashboard(snapshot, link="https://bi.example.com/x", title="Sales pulse")
        assert tile.title == "What changed" and "Riverside Store" in html and "Most of it by store:" in html
        assert "- Net amount: $185,933.03, down $8,943.45" in render.dashboard_text(snapshot, link="x")
