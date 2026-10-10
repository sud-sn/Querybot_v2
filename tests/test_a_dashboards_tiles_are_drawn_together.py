"""A dashboard's tiles are drawn several at once, off the server's loop.

Each tile may wait seconds on the warehouse. They were drawn one after another inside the page's request, on
the server's own loop: a 12-tile dashboard waited for all 12 in turn, and every other reader's page and chat
waited with it. Now they are drawn on a pool of their own, at most four of one dashboard at a time, in the
dashboard's order, in the reader's language; a tile that fails says so on its card and the others are drawn.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid

import pytest

import store

PAUSE = 0.25


@pytest.fixture()
def tiles(monkeypatch):
    """Tiles that each wait on the warehouse for PAUSE seconds: how many were drawn at once at most, on which
    threads, in which language. A tile titled "Broken" fails."""
    import portal.routes as routes
    from core.i18n import get_active_language

    state = {"running": 0, "most": 0, "threads": set(), "languages": set()}
    lock = threading.Lock()

    def draw(chart, db_cfg, user, **_):
        with lock:
            state["running"] += 1
            state["most"] = max(state["most"], state["running"])
            state["threads"].add(threading.current_thread().name)
            state["languages"].add(get_active_language())
        try:
            time.sleep(PAUSE)
            if chart["title"] == "Broken":
                raise RuntimeError("the warehouse went away")
            return {**routes._blank_tile(chart), "kpi": {"value": 1}, "kpi_display": chart["title"]}
        finally:
            with lock:
                state["running"] -= 1

    monkeypatch.setattr(routes, "_refresh_chart", draw)
    return state


def _charts(n, broken=()):
    return [{"id": i, "title": "Broken" if i in broken else f"Tile {i}", "db_config_id": None} for i in range(n)]


def test_tiles_are_drawn_several_at_once_in_the_dashboards_order(tiles):
    from portal.routes import TILES_AT_ONCE, _draw_tiles

    started = time.perf_counter()
    drawn = asyncio.run(_draw_tiles(_charts(8), None, {"id": 1}))
    took = time.perf_counter() - started
    assert [t["title"] for t in drawn] == [f"Tile {i}" for i in range(8)]
    assert tiles["most"] == TILES_AT_ONCE == 4                  # several at once, never more than four
    assert took < 8 * PAUSE * 0.6                               # two rounds of four, not eight in turn
    assert all(name.startswith("qb-tile") for name in tiles["threads"])   # their own pool


def test_the_server_answers_other_readers_while_tiles_are_drawn(tiles):
    from portal.routes import _draw_tiles

    async def scene():
        ticks = 0
        drawing = asyncio.ensure_future(_draw_tiles(_charts(4), None, {"id": 1}))
        while not drawing.done():
            ticks += 1                                          # another reader's request, served meanwhile
            await asyncio.sleep(0.01)
        await drawing
        return ticks

    assert asyncio.run(scene()) >= 10


def test_each_tile_is_drawn_in_the_readers_language(tiles):
    from core.i18n import activate_language, deactivate_language
    from portal.routes import _draw_tiles

    async def scene():
        token = activate_language("fr")
        try:
            return await _draw_tiles(_charts(3), None, {"id": 1})
        finally:
            deactivate_language(token)

    asyncio.run(scene())
    assert tiles["languages"] == {"fr"}


def test_a_tile_that_fails_says_so_and_the_others_are_drawn(tiles):
    from portal.routes import _draw_tiles

    drawn = asyncio.run(_draw_tiles(_charts(3, broken={1}), None, {"id": 1}))
    assert [t.get("kpi_display") for t in drawn] == ["Tile 0", None, "Tile 2"]
    assert drawn[1]["error"] == "This chart could not be refreshed."
    assert drawn[1]["table_rows"] == [] and drawn[1]["filter_warnings"] == [] and drawn[1]["chart_json"] is None


def test_the_dashboard_page_shows_every_tile_drawn_together(tiles):
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    import portal.routes as routes

    store.init_db()
    account = f"tiles-{uuid.uuid4().hex[:10]}"
    store.upsert_client(account, "portal")
    owner = store.create_user(account, "Ada Owner", f"{uuid.uuid4().hex[:8]}@test.com",
                              password="a-password-they-chose")[0]
    board = store.create_dashboard(account, owner, "thread", "Sales overview")
    try:
        for n in range(6):
            store.pin_chart(owner, account, f"Tile {n}", f"What is tile {n}?", "SELECT 1", "kpi", None,
                            dashboard_id=board["id"])
        app = FastAPI()
        app.include_router(routes.router)
        client = TestClient(app)
        client.cookies.set(routes._COOKIE, routes._sign_session_value(owner))
        started = time.perf_counter()
        page = client.get(f"/portal/dashboard?dashboard_id={board['id']}")
        took = time.perf_counter() - started
        assert page.status_code == 200
        places = [page.text.index(f">Tile {n}<") for n in range(6)]
        assert places == sorted(places)
        assert tiles["most"] > 1 and took < 6 * PAUSE
    finally:
        with store.get_db() as conn:
            for table in ("pinned_chart", "dashboard_artifact_version", "dashboard_artifact", "portal_user", "client"):
                conn.execute(f"DELETE FROM {table} WHERE account_id=?", (account,))


@pytest.mark.parametrize("setting, threads", [(None, 8), ("16", 16), ("1", 2), ("500", 32), ("eight", 8)])
def test_the_tile_pool_is_set_by_its_setting_within_bounds(monkeypatch, setting, threads):
    from portal.routes import _tile_threads

    if setting is None:
        monkeypatch.delenv("QUERYBOT_TILE_THREADS", raising=False)
    else:
        monkeypatch.setenv("QUERYBOT_TILE_THREADS", setting)
    assert _tile_threads() == threads


def test_the_local_warehouse_answers_queries_at_once_each_with_its_own_rows(tmp_path):
    """The invented warehouse the UI check's server and the answer tests run on: queries at once, as a
    dashboard's tiles now are, each read their own rows (they read whichever query ran last)."""
    from concurrent.futures import ThreadPoolExecutor

    from tests import answer_harness as H

    warehouse = H.Warehouse(tmp_path / "w.duckdb", {"dbo.T": {"columns": [{"name": "N", "type": "int"}]}},
                            {"T": [(n,) for n in range(2000)]})

    def ask(k):
        return warehouse.query(f"SELECT SUM(N) + {k} AS v FROM T")[0]["v"] == sum(range(2000)) + k

    with ThreadPoolExecutor(8) as pool:
        assert all(pool.map(ask, range(400)))
