"""A dashboard's row of numbers fills its width, and its charts come up under it: no empty space beside a number.

From a reader's test: a dashboard built from the chat had its numbers on top, and when there was only one, or a
space beside them, the charts stayed below instead of filling the row. A dashboard built from the chat was saved
as if every tile had been placed by hand, so whatever happened to it next (a number taken away, a tile of two
numbers split in two, a number added) went through the layout that keeps hand-placed tiles where they are: a new
number a quarter of the row wide, the rest of the row empty.

Now a dashboard Solviz built is laid out again with no gap whenever a tile comes or goes; on one a person
arranged, the numbers still fill their row and the person's charts keep their places under it. Each tab is laid
out on its own (a chart on one tab was paired with a chart on another, each alone in half its row).

The real store; invented tiles.
"""

from __future__ import annotations

import json
import uuid

import pytest

import store

NUMBER = {"grid_h": 3}
CHART = {"grid_h": 10}


@pytest.fixture()
def owner():
    store.init_db()
    account = f"rows-{uuid.uuid4().hex[:10]}"
    store.upsert_client(account, "portal")
    user_id, _ = store.create_user(account, "Owner", f"owner-{uuid.uuid4().hex[:8]}@test.com")
    yield account, user_id
    with store.get_db() as conn:
        for table in ("pinned_chart", "dashboard_artifact_version", "dashboard_artifact", "portal_user", "client"):
            conn.execute(f"DELETE FROM {table} WHERE account_id=?", (account,))


def _tile(title: str, chart_type: str, x: int, y: int, w: int, tab: str = "Overview", measures=("revenue",)) -> dict:
    plan = {"kind": "query", "intent": "value", "measures": list(measures)}
    config = {"core2_plan": plan, "title": title}
    if len(measures) > 1:
        config["titles"] = [m.title() for m in measures]
    return {"title": title, "question": title, "chart_type": chart_type, "tab": tab, "sql": "SELECT 1",
            "layout": {"grid_x": x, "grid_y": y, "grid_w": w, "grid_h": 3 if chart_type == "kpi" else 10},
            "display_config": config}


def _built(owner, tiles: list[dict], tabs=("Overview",)) -> int:
    account, user_id = owner
    board = store.save_built_dashboard(account, user_id, name="Executive overview", tiles=tiles, db_config_id=None,
                                       tab_names=list(tabs))
    return int(board["id"])


def _rects(owner, board: int) -> dict[str, tuple[int, int, int, int, str]]:
    _, user_id = owner
    return {c["title"]: (c["grid_x"], c["grid_y"], c["grid_w"], c["grid_h"], c["dashboard_tab"])
            for c in store.list_dashboard_charts(board, user_id)}


def _no_gap_beside_the_numbers(rects: dict) -> None:
    """On each tab: every row of numbers covers the twelve columns, nothing overlaps, and the charts start right
    under the numbers."""
    tabs: dict[str, list] = {}
    for title, (x, y, w, h, tab) in rects.items():
        tabs.setdefault(tab, []).append((title, x, y, w, h))
    for tab, tiles in tabs.items():
        numbers = [t for t in tiles if t[4] == 3]
        rows: dict[int, int] = {}
        for _, x, y, w, _h in numbers:
            rows[y] = rows.get(y, 0) + w
        assert all(width == 12 for width in rows.values()), (tab, tiles)
        cells = set()
        for title, x, y, w, h in tiles:
            mine = {(cx, cy) for cx in range(x, x + w) for cy in range(y, y + h)}
            assert not cells & mine, f"{title} overlaps on {tab}: {tiles}"
            cells |= mine
        band = max((y + h for _, _, y, _, h in numbers), default=0)
        assert min((y for _, _, y, _, h in tiles if h != 3), default=band) == band, (tab, tiles)


def test_a_dashboard_built_from_the_chat_is_saved_as_placed_by_solviz(owner):
    board = _built(owner, [_tile("Revenue", "kpi", 0, 0, 6), _tile("Orders", "kpi", 6, 0, 6),
                           _tile("Revenue by month", "line", 0, 3, 6), _tile("Revenue by region", "bar", 6, 3, 6)])
    charts = store.list_dashboard_charts(board, owner[1])
    assert {c["layout_locked"] for c in charts} == {0}
    assert [c["title"] for c in sorted(charts, key=lambda c: c["position"])] == [
        "Revenue", "Orders", "Revenue by month", "Revenue by region"]


def test_a_number_taken_away_leaves_the_other_across_the_row(owner):
    board = _built(owner, [_tile("Revenue", "kpi", 0, 0, 6), _tile("Orders", "kpi", 6, 0, 6),
                           _tile("Revenue by month", "line", 0, 3, 6), _tile("Revenue by region", "bar", 6, 3, 6),
                           _tile("Top products", "table", 0, 13, 12)])
    orders = next(c["id"] for c in store.list_dashboard_charts(board, owner[1]) if c["title"] == "Orders")
    assert store.remove_chart_from_dashboard(board, orders, owner[1], owner[0])
    rects = _rects(owner, board)
    assert rects["Revenue"][:4] == (0, 0, 12, 3)
    assert rects["Revenue by month"][:3] == (0, 3, 6) and rects["Revenue by region"][:3] == (6, 3, 6)
    _no_gap_beside_the_numbers(rects)


def test_a_tile_of_two_numbers_split_in_two_shares_its_row(owner):
    board = _built(owner, [_tile("Revenue and orders", "kpi", 0, 0, 12, measures=("revenue", "orders")),
                           _tile("Revenue by month", "line", 0, 3, 6), _tile("Revenue by region", "bar", 6, 3, 6)])
    assert store.split_number_groups(board, owner[1], owner[0]) == 1
    rects = _rects(owner, board)
    numbers = sorted(r[:4] for t, r in rects.items() if r[3] == 3)
    assert numbers == [(0, 0, 6, 3), (6, 0, 6, 3)], rects
    _no_gap_beside_the_numbers(rects)


def test_a_number_added_to_a_dashboard_a_person_arranged_joins_the_row(owner):
    account, user_id = owner
    board = _built(owner, [_tile("Revenue", "kpi", 0, 0, 12),
                           _tile("Revenue by month", "line", 0, 3, 6), _tile("Revenue by region", "bar", 6, 3, 6)])
    ids = {c["title"]: c["id"] for c in store.list_dashboard_charts(board, user_id)}
    # The person makes the number narrow and swaps the two charts.
    assert store.update_dashboard_layouts(board, user_id, account, [
        {"chart_id": ids["Revenue"], "x": 0, "y": 0, "w": 4, "h": 3},
        {"chart_id": ids["Revenue by region"], "x": 0, "y": 3, "w": 6, "h": 10},
        {"chart_id": ids["Revenue by month"], "x": 6, "y": 3, "w": 6, "h": 10}])
    kpi = store.pin_chart(user_id=user_id, account_id=account, title="Orders", question="Orders",
                          sql_query="SELECT 1", chart_type="kpi", db_config_id=None)
    assert store.add_chart_to_dashboard(board, kpi, user_id, account)
    rects = _rects(owner, board)
    assert sorted(r[:4] for t, r in rects.items() if r[3] == 3) == [(0, 0, 6, 3), (6, 0, 6, 3)]
    assert rects["Revenue by region"][:3] == (0, 3, 6) and rects["Revenue by month"][:3] == (6, 3, 6), \
        "the person's charts stay where they put them"
    _no_gap_beside_the_numbers(rects)


def test_each_tab_is_laid_out_on_its_own(owner):
    board = _built(owner, [_tile("Revenue", "kpi", 0, 0, 12),
                           _tile("Revenue by month", "line", 0, 3, 12),
                           _tile("Margin", "kpi", 0, 0, 12, tab="Profit"),
                           _tile("Margin by month", "line", 0, 3, 12, tab="Profit")], tabs=("Overview", "Profit"))
    assert store.tidy_dashboard_layout(board, owner[1], owner[0])
    rects = _rects(owner, board)
    # One chart on each tab: each across its own row, never half of a pair with the other tab's.
    assert rects["Revenue by month"][:4] == (0, 3, 12, 10) and rects["Margin by month"][:4] == (0, 3, 12, 10)
    _no_gap_beside_the_numbers(rects)


def test_a_dashboard_nobody_arranged_stays_gap_free_through_every_change(owner):
    account, user_id = owner
    board = _built(owner, [_tile("Revenue", "kpi", 0, 0, 4), _tile("Orders", "kpi", 4, 0, 4),
                           _tile("Customers", "kpi", 8, 0, 4),
                           _tile("Revenue by month", "line", 0, 3, 6), _tile("Revenue by region", "bar", 6, 3, 6),
                           _tile("Orders by month", "line", 0, 13, 12)])
    for title in ("Customers", "Revenue by region"):
        chart = next(c["id"] for c in store.list_dashboard_charts(board, user_id) if c["title"] == title)
        assert store.remove_chart_from_dashboard(board, chart, user_id, account)
        _no_gap_beside_the_numbers(_rects(owner, board))
    rects = _rects(owner, board)
    assert rects["Revenue"][:4] == (0, 0, 6, 3) and rects["Orders"][:4] == (6, 0, 6, 3)
    assert json.dumps(sorted(r[:4] for t, r in rects.items() if r[3] == 10)) == json.dumps(
        [[0, 3, 6, 10], [6, 3, 6, 10]])
