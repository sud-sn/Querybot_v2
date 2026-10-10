"""Durable, owner-scoped dashboard artifacts created from portal chat.

Dashboard items reference existing ``pinned_chart`` records.  This preserves
the established live refresh, validation, and compliance path and never
persists result rows in the artifact itself.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

from .db import get_db
from .crypto import decrypt_json, encrypt


def _clean_name(name: str) -> str:
    value = re.sub(r"\s+", " ", str(name or "")).strip()
    return value[:120] or "Untitled dashboard"


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _snapshot_locked(conn, dashboard_id: int, user_id: int, account_id: str, summary: str) -> None:
    dashboard = conn.execute(
        "SELECT * FROM dashboard_artifact WHERE id=? AND user_id=? AND account_id=?",
        (int(dashboard_id), int(user_id), account_id),
    ).fetchone()
    if not dashboard:
        return
    charts = conn.execute(
        """SELECT id, data_source_id, title, chart_type, color_palette,
                  position, grid_x, grid_y, grid_w, grid_h, display_config,
                  dashboard_tab, sort_enabled, layout_locked
             FROM pinned_chart WHERE dashboard_id=? AND user_id=?
             ORDER BY position, id""",
        (int(dashboard_id), int(user_id)),
    ).fetchall()
    row = dict(dashboard)
    snapshot = {
        "dashboard": {
            key: row.get(key)
            for key in (
                "name", "description", "status", "visibility",
                "refresh_schedule", "filters_json", "tabs_json",
            )
        },
        "charts": [dict(chart) for chart in charts],
    }
    conn.execute(
        """INSERT INTO dashboard_artifact_version
               (dashboard_id, account_id, user_id, version, change_summary, snapshot_json)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(dashboard_id, version) DO UPDATE SET
               change_summary=excluded.change_summary,
               snapshot_json=excluded.snapshot_json,
               created_at=datetime('now')""",
        (
            int(dashboard_id), account_id, int(user_id), int(row.get("version") or 1),
            str(summary or "Dashboard updated")[:240], _json(snapshot),
        ),
    )


def create_dashboard(
    account_id: str,
    user_id: int,
    thread_id: str,
    name: str,
    description: str = "",
    visibility: str = "personal",
) -> dict:
    visibility = str(visibility or "personal").strip().lower()
    if visibility not in {"personal", "team"}:
        visibility = "personal"
    with get_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO dashboard_artifact
                (account_id, user_id, thread_id, name, description, visibility)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                account_id,
                int(user_id),
                str(thread_id or "default")[:80],
                _clean_name(name),
                str(description or "").strip()[:500],
                visibility,
            ),
        )
        row = conn.execute(
            "SELECT * FROM dashboard_artifact WHERE id=?", (cur.lastrowid,)
        ).fetchone()
        _snapshot_locked(conn, cur.lastrowid, int(user_id), account_id, "Dashboard created")
    return dict(row)


def get_dashboard(dashboard_id: int, user_id: int, account_id: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT * FROM dashboard_artifact
             WHERE id=? AND user_id=? AND account_id=?
            """,
            (int(dashboard_id), int(user_id), account_id),
        ).fetchone()
    return dict(row) if row else None


def get_dashboard_for_view(
    dashboard_id: int, user_id: int, account_id: str
) -> dict | None:
    """Return an owned dashboard or a published team dashboard, read-only."""
    with get_db() as conn:
        row = conn.execute(
            """SELECT *, CASE WHEN user_id=? THEN 1 ELSE 0 END AS can_edit
                 FROM dashboard_artifact
                WHERE id=? AND account_id=?
                  AND (user_id=? OR (visibility='team' AND status='published'))""",
            (int(user_id), int(dashboard_id), account_id, int(user_id)),
        ).fetchone()
    return dict(row) if row else None


def latest_dashboard_for_thread(
    account_id: str, user_id: int, thread_id: str
) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT * FROM dashboard_artifact
             WHERE account_id=? AND user_id=? AND thread_id=?
             ORDER BY updated_at DESC, id DESC LIMIT 1
            """,
            (account_id, int(user_id), str(thread_id or "default")[:80]),
        ).fetchone()
    return dict(row) if row else None


def list_dashboards(account_id: str, user_id: int) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            """SELECT d.*, COUNT(c.id) AS chart_count,
                      CASE WHEN d.user_id=? THEN 1 ELSE 0 END AS can_edit
                 FROM dashboard_artifact d
                 LEFT JOIN pinned_chart c ON c.dashboard_id=d.id AND c.user_id=d.user_id
                WHERE d.account_id=?
                  AND (d.user_id=? OR (d.visibility='team' AND d.status='published'))
                GROUP BY d.id
                ORDER BY can_edit DESC, d.updated_at DESC, d.id DESC""",
            (int(user_id), account_id, int(user_id)),
        ).fetchall()
    return [dict(row) for row in rows]


def list_editable_dashboards(account_id: str, user_id: int) -> list[dict]:
    """Return named dashboards owned by this user, newest first."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT d.*, COUNT(c.id) AS chart_count, 1 AS can_edit
                 FROM dashboard_artifact d
                 LEFT JOIN pinned_chart c ON c.dashboard_id=d.id AND c.user_id=d.user_id
                WHERE d.account_id=? AND d.user_id=?
                GROUP BY d.id
                ORDER BY d.updated_at DESC, d.id DESC""",
            (account_id, int(user_id)),
        ).fetchall()
    return [dict(row) for row in rows]


def migrate_legacy_charts(account_id: str, user_id: int) -> dict | None:
    """Move pre-artifact chart pins into a named dashboard exactly once."""
    with get_db() as conn:
        legacy = conn.execute(
            """SELECT id FROM pinned_chart
                WHERE account_id=? AND user_id=? AND dashboard_id IS NULL
                ORDER BY position, id""",
            (account_id, int(user_id)),
        ).fetchall()
        if not legacy:
            return None
        dashboard = conn.execute(
            """SELECT * FROM dashboard_artifact
                WHERE account_id=? AND user_id=? AND thread_id='legacy-import'
                ORDER BY id LIMIT 1""",
            (account_id, int(user_id)),
        ).fetchone()
        if dashboard:
            dashboard_id = int(dashboard["id"])
        else:
            cur = conn.execute(
                """INSERT INTO dashboard_artifact
                       (account_id, user_id, thread_id, name, description)
                   VALUES (?, ?, 'legacy-import', 'My Dashboard',
                           'Imported from charts saved before named dashboards were enabled.')""",
                (account_id, int(user_id)),
            )
            dashboard_id = int(cur.lastrowid)
        conn.executemany(
            """UPDATE pinned_chart
                  SET dashboard_id=?, dashboard_tab='Overview', layout_locked=0
                WHERE id=? AND user_id=? AND account_id=?""",
            [
                (dashboard_id, int(row["id"]), int(user_id), account_id)
                for row in legacy
            ],
        )
        _auto_layout_locked(conn, dashboard_id, int(user_id))
        conn.execute(
            """UPDATE dashboard_artifact SET updated_at=datetime('now')
                WHERE id=? AND user_id=? AND account_id=?""",
            (dashboard_id, int(user_id), account_id),
        )
        _snapshot_locked(
            conn, dashboard_id, int(user_id), account_id, "Legacy charts imported"
        )
        row = conn.execute(
            "SELECT * FROM dashboard_artifact WHERE id=?", (dashboard_id,)
        ).fetchone()
    return dict(row) if row else None


def create_data_source(
    dashboard_id: int,
    user_id: int,
    account_id: str,
    *,
    name: str,
    question: str,
    sql_query: str,
    db_config_id: int,
    semantic_contract_version: str = "",
) -> dict:
    """Persist a named live source; never accepts rows or credentials."""
    if not str(sql_query or "").strip():
        raise ValueError("A governed SQL query is required for a dashboard source.")
    with get_db() as conn:
        owned = conn.execute(
            "SELECT id FROM dashboard_artifact WHERE id=? AND user_id=? AND account_id=?",
            (int(dashboard_id), int(user_id), account_id),
        ).fetchone()
        if not owned:
            raise PermissionError("Dashboard does not belong to this user and workspace.")
        cur = conn.execute(
            """INSERT INTO dashboard_data_source
                   (dashboard_id, account_id, user_id, name, question, sql_query,
                    db_config_id, semantic_contract_version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                int(dashboard_id), account_id, int(user_id), _clean_name(name),
                str(question or "").strip()[:500], str(sql_query).strip(),
                int(db_config_id), str(semantic_contract_version or "")[:128],
            ),
        )
        row = conn.execute(
            "SELECT * FROM dashboard_data_source WHERE id=?", (cur.lastrowid,)
        ).fetchone()
    return dict(row)


def list_data_sources(dashboard_id: int, user_id: int, account_id: str) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            """SELECT id, dashboard_id, name, source_type, question, db_config_id,
                      semantic_contract_version, created_at, updated_at
                 FROM dashboard_data_source
                WHERE dashboard_id=? AND user_id=? AND account_id=?
                ORDER BY id""",
            (int(dashboard_id), int(user_id), account_id),
        ).fetchall()
    return [dict(row) for row in rows]


def list_data_sources_for_view(
    dashboard_id: int, viewer_user_id: int, account_id: str
) -> list[dict]:
    dashboard = get_dashboard_for_view(dashboard_id, viewer_user_id, account_id)
    if not dashboard:
        return []
    return list_data_sources(dashboard_id, int(dashboard["user_id"]), account_id)


def get_data_source(source_id: int, user_id: int, account_id: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            """SELECT * FROM dashboard_data_source
                WHERE id=? AND user_id=? AND account_id=?""",
            (int(source_id), int(user_id), account_id),
        ).fetchone()
    return dict(row) if row else None


def _layout_size(chart_type: str) -> tuple[int, int]:
    kind = str(chart_type or "bar").strip().lower()
    if kind == "kpi":
        return 3, 3
    if kind == "table":
        return 12, 6
    return 6, 5


def _overlaps(candidate: tuple[int, int, int, int], occupied: list[tuple[int, int, int, int]]) -> bool:
    x, y, w, h = candidate
    return any(
        x < ox + ow and x + w > ox and y < oy + oh and y + h > oy
        for ox, oy, ow, oh in occupied
    )


def _first_open_slot(
    width: int,
    height: int,
    occupied: list[tuple[int, int, int, int]],
    *,
    start_y: int = 0,
) -> tuple[int, int]:
    step = 3 if width <= 3 else (6 if width <= 6 else 12)
    for y in range(max(0, int(start_y)), 500):
        for x in range(0, 13 - width, step):
            candidate = (x, y, width, height)
            if not _overlaps(candidate, occupied):
                return x, y
    return 0, max((y + h for _, y, _, h in occupied), default=0)


def _numbers_in(chart) -> int:
    """How many numbers a KPI tile shows: one, or each measure of a new-core answer that asked for several."""
    try:
        raw = chart["display_config"]
    except (IndexError, KeyError):
        return 1
    try:
        plan = (json.loads(raw or "{}") or {}).get("core2_plan") or {}
    except (TypeError, ValueError, AttributeError):
        return 1
    return max(1, min(4, len(plan.get("measures") or []) if isinstance(plan, dict) else 1))


def _packed(charts: list) -> list[tuple[int, int, int, int]]:
    """Tiles laid out with no gap for the grid to pull a tile into: the headline numbers share the first row
    evenly (four at most to a row), then the charts two to a row, a chart left alone in its row and a table
    across the whole width. A KPI three columns wide left nine empty beside it, the grid pulled the next
    chart up into them, and the two rows no longer lined up."""
    kpis = [c for c in charts if str(c["chart_type"] or "").lower() == "kpi"]
    rest = [c for c in charts if str(c["chart_type"] or "").lower() != "kpi"]
    rects: dict[int, tuple[int, int, int, int]] = {}
    y = 0
    # A tile of several numbers side by side takes a share of its row for each of them.
    rows: list[list] = []
    for chart in kpis:
        if not rows or sum(_numbers_in(c) for c in rows[-1]) + _numbers_in(chart) > 4:
            rows.append([])
        rows[-1].append(chart)
    for row in rows:
        slots, x = sum(_numbers_in(c) for c in row), 0
        for i, chart in enumerate(row):
            width = 12 - x if i == len(row) - 1 else 12 * _numbers_in(chart) // slots
            rects[int(chart["id"])] = (x, y, width, 3)
            x += width
        y += 3
    waiting = None                     # a chart alone, so far, in the current row
    for chart in rest:
        if str(chart["chart_type"] or "").lower() == "table":
            if waiting is not None:
                x0, y0, _, h0 = rects[waiting]
                rects[waiting], y, waiting = (x0, y0, 12, h0), y0 + h0, None
            rects[int(chart["id"])] = (0, y, 12, 6)
            y += 6
        elif waiting is None:
            rects[int(chart["id"])] = (0, y, 6, 5)
            waiting = int(chart["id"])
        else:
            rects[int(chart["id"])] = (6, rects[waiting][1], 6, 5)
            y, waiting = rects[waiting][1] + 5, None
    if waiting is not None:
        x0, y0, _, h0 = rects[waiting]
        rects[waiting] = (x0, y0, 12, h0)
    return [rects[int(c["id"])] for c in charts]


def _auto_layout_locked(conn, dashboard_id: int, user_id: int) -> None:
    """Place unlocked tiles KPI-first while respecting user-edited tiles."""
    charts = conn.execute(
        """SELECT id, chart_type, position, grid_x, grid_y, grid_w, grid_h, display_config,
                  COALESCE(layout_locked, 0) AS layout_locked
             FROM pinned_chart
            WHERE dashboard_id=? AND user_id=?
            ORDER BY CASE WHEN LOWER(chart_type)='kpi' THEN 0 ELSE 1 END,
                     position, id""",
        (int(dashboard_id), int(user_id)),
    ).fetchall()
    if not any(int(chart["layout_locked"] or 0) for chart in charts):
        # Nobody has placed a tile by hand: the whole dashboard is laid out again, gap-free.
        for position, (chart, (x, y, width, height)) in enumerate(zip(charts, _packed(charts)), start=1):
            conn.execute(
                """UPDATE pinned_chart
                      SET grid_x=?, grid_y=?, grid_w=?, grid_h=?, position=?
                    WHERE id=? AND user_id=?""",
                (x, y, width, height, position, int(chart["id"]), int(user_id)),
            )
        return
    occupied: list[tuple[int, int, int, int]] = []
    kpi_floor = 0
    for chart in charts:
        if int(chart["layout_locked"] or 0):
            rect = (
                int(chart["grid_x"] or 0), int(chart["grid_y"] or 0),
                int(chart["grid_w"] or 6), int(chart["grid_h"] or 5),
            )
            occupied.append(rect)
            if str(chart["chart_type"] or "").lower() == "kpi":
                kpi_floor = max(kpi_floor, rect[1] + rect[3])
    for position, chart in enumerate(charts, start=1):
        if int(chart["layout_locked"] or 0):
            conn.execute(
                "UPDATE pinned_chart SET position=? WHERE id=? AND user_id=?",
                (position, int(chart["id"]), int(user_id)),
            )
            continue
        width, height = _layout_size(chart["chart_type"])
        is_kpi = str(chart["chart_type"] or "").lower() == "kpi"
        x, y = _first_open_slot(
            width, height, occupied, start_y=0 if is_kpi else kpi_floor
        )
        occupied.append((x, y, width, height))
        if is_kpi:
            kpi_floor = max(kpi_floor, y + height)
        conn.execute(
            """UPDATE pinned_chart
                  SET grid_x=?, grid_y=?, grid_w=?, grid_h=?, position=?
                WHERE id=? AND user_id=?""",
            (x, y, width, height, position, int(chart["id"]), int(user_id)),
        )


def add_chart(
    dashboard_id: int,
    chart_id: int,
    user_id: int,
    account_id: str,
    *,
    data_source_id: int | None = None,
    tab: str = "Overview",
) -> bool:
    """Attach an owned chart to an owned dashboard and bump its draft version."""
    with get_db() as conn:
        dashboard = conn.execute(
            "SELECT id FROM dashboard_artifact WHERE id=? AND user_id=? AND account_id=?",
            (int(dashboard_id), int(user_id), account_id),
        ).fetchone()
        chart = conn.execute(
            "SELECT id FROM pinned_chart WHERE id=? AND user_id=? AND account_id=?",
            (int(chart_id), int(user_id), account_id),
        ).fetchone()
        if not dashboard or not chart:
            return False
        existing = conn.execute(
            """SELECT COUNT(*) AS count FROM pinned_chart
                WHERE dashboard_id=? AND user_id=? AND id<>?""",
            (int(dashboard_id), int(user_id), int(chart_id)),
        ).fetchone()
        version_increment = 1 if existing and int(existing["count"] or 0) > 0 else 0
        # Added last: its place among the dashboard's own tiles, never the number it was pinned with (which
        # counts the reader's other pins and put a new chart in among the old ones).
        last = conn.execute(
            """SELECT COALESCE(MAX(position), 0) AS last FROM pinned_chart
                WHERE dashboard_id=? AND user_id=? AND id<>?""",
            (int(dashboard_id), int(user_id), int(chart_id)),
        ).fetchone()
        conn.execute(
            """UPDATE pinned_chart
                  SET dashboard_id=?, data_source_id=?, dashboard_tab=?, layout_locked=0,
                      position=CASE WHEN dashboard_id IS ? THEN position ELSE ? END
                WHERE id=? AND user_id=?""",
            (
                int(dashboard_id), int(data_source_id) if data_source_id else None,
                _clean_name(tab)[:60], int(dashboard_id), int(last["last"] or 0) + 1,
                int(chart_id), int(user_id),
            ),
        )
        conn.execute(
            """
            UPDATE dashboard_artifact
               SET version=version+?, status='draft', updated_at=datetime('now')
             WHERE id=? AND user_id=? AND account_id=?
            """,
            (version_increment, int(dashboard_id), int(user_id), account_id),
        )
        _auto_layout_locked(conn, int(dashboard_id), int(user_id))
        _snapshot_locked(conn, dashboard_id, user_id, account_id, "Visual added")
    return True


def remove_chart(
    dashboard_id: int, chart_id: int, user_id: int, account_id: str
) -> bool:
    """Remove one visual from an owned dashboard and version the change."""
    with get_db() as conn:
        chart = conn.execute(
            """SELECT c.id, c.data_source_id
                 FROM pinned_chart c
                 JOIN dashboard_artifact d ON d.id=c.dashboard_id
                WHERE c.id=? AND c.dashboard_id=? AND c.user_id=?
                  AND c.account_id=? AND d.user_id=? AND d.account_id=?""",
            (
                int(chart_id), int(dashboard_id), int(user_id), account_id,
                int(user_id), account_id,
            ),
        ).fetchone()
        if not chart:
            return False
        conn.execute(
            "DELETE FROM pinned_chart WHERE id=? AND user_id=?",
            (int(chart_id), int(user_id)),
        )
        if chart["data_source_id"]:
            conn.execute(
                """DELETE FROM dashboard_data_source
                    WHERE id=? AND dashboard_id=? AND user_id=? AND account_id=?""",
                (
                    int(chart["data_source_id"]), int(dashboard_id),
                    int(user_id), account_id,
                ),
            )
        _auto_layout_locked(conn, int(dashboard_id), int(user_id))
        conn.execute(
            """UPDATE dashboard_artifact
                  SET version=version+1, status='draft', updated_at=datetime('now')
                WHERE id=? AND user_id=? AND account_id=?""",
            (int(dashboard_id), int(user_id), account_id),
        )
        _snapshot_locked(conn, dashboard_id, user_id, account_id, "Visual removed")
    return True


def update_dashboard_chart(
    dashboard_id: int,
    chart_id: int,
    user_id: int,
    account_id: str,
    *,
    title: str | None = None,
    chart_type: str | None = None,
    color_palette: str | None = None,
) -> bool:
    """Update one owned visual, reflowing it when its visual kind changes."""
    assignments: list[str] = []
    values: list = []
    if title is not None:
        assignments.append("title=?")
        values.append(str(title).strip()[:120])
    if chart_type is not None:
        assignments.extend(("chart_type=?", "layout_locked=0"))
        values.append(str(chart_type).strip().lower()[:20])
    if color_palette is not None:
        assignments.append("color_palette=?")
        values.append(str(color_palette).strip()[:30])
    if not assignments:
        return False
    with get_db() as conn:
        values.extend((int(chart_id), int(dashboard_id), int(user_id), account_id))
        cur = conn.execute(
            f"""UPDATE pinned_chart SET {', '.join(assignments)}
                 WHERE id=? AND dashboard_id=? AND user_id=? AND account_id=?""",
            values,
        )
        if not cur.rowcount:
            return False
        _auto_layout_locked(conn, int(dashboard_id), int(user_id))
        conn.execute(
            """UPDATE dashboard_artifact
                  SET version=version+1, status='draft', updated_at=datetime('now')
                WHERE id=? AND user_id=? AND account_id=?""",
            (int(dashboard_id), int(user_id), account_id),
        )
        _snapshot_locked(conn, dashboard_id, user_id, account_id, "Visual updated")
    return True


def list_dashboard_charts(dashboard_id: int, user_id: int) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT * FROM pinned_chart
             WHERE dashboard_id=? AND user_id=?
             ORDER BY position, id
            """,
            (int(dashboard_id), int(user_id)),
        ).fetchall()
    return [dict(row) for row in rows]


def list_dashboard_charts_for_view(
    dashboard_id: int, viewer_user_id: int, account_id: str
) -> list[dict]:
    dashboard = get_dashboard_for_view(dashboard_id, viewer_user_id, account_id)
    if not dashboard:
        return []
    return list_dashboard_charts(dashboard_id, int(dashboard["user_id"]))


def update_dashboard_layouts(
    dashboard_id: int,
    user_id: int,
    account_id: str,
    layouts: list[dict],
) -> bool:
    """Persist an owner-edited layout and lock those tiles from auto-reflow."""
    if not layouts:
        return False
    normalized: list[tuple[int, int, int, int, int, int]] = []
    for index, item in enumerate(layouts):
        try:
            chart_id = int(item.get("chart_id") or item.get("id") or 0)
            if not chart_id:
                continue
            width = max(2, min(12, int(item.get("w") or 6)))
            x = max(0, min(12 - width, int(item.get("x") or 0)))
            normalized.append((
                x,
                max(0, int(item.get("y") or 0)),
                width,
                max(2, min(12, int(item.get("h") or 5))),
                max(1, int(item.get("position") or index + 1)),
                chart_id,
            ))
        except (TypeError, ValueError):
            continue
    if not normalized:
        return False
    with get_db() as conn:
        owned = conn.execute(
            "SELECT id FROM dashboard_artifact WHERE id=? AND user_id=? AND account_id=?",
            (int(dashboard_id), int(user_id), account_id),
        ).fetchone()
        if not owned:
            return False
        changed = 0
        for x, y, width, height, position, chart_id in normalized:
            cur = conn.execute(
                """UPDATE pinned_chart
                      SET grid_x=?, grid_y=?, grid_w=?, grid_h=?, position=?, layout_locked=1
                    WHERE id=? AND dashboard_id=? AND user_id=? AND account_id=?""",
                (
                    x, y, width, height, position, chart_id,
                    int(dashboard_id), int(user_id), account_id,
                ),
            )
            changed += int(cur.rowcount or 0)
        if not changed:
            return False
        conn.execute(
            """UPDATE dashboard_artifact
                  SET version=version+1, status='draft', updated_at=datetime('now')
                WHERE id=? AND user_id=? AND account_id=?""",
            (int(dashboard_id), int(user_id), account_id),
        )
        _snapshot_locked(conn, dashboard_id, user_id, account_id, "Layout updated")
    return True


def tidy_dashboard_layout(dashboard_id: int, user_id: int, account_id: str) -> bool:
    """Lay an owned dashboard out again from scratch, gap-free (KPIs first, charts two to a row), forgetting
    where tiles were placed by hand: the reader's way back from a layout that no longer lines up."""
    with get_db() as conn:
        owned = conn.execute(
            "SELECT id FROM dashboard_artifact WHERE id=? AND user_id=? AND account_id=?",
            (int(dashboard_id), int(user_id), account_id),
        ).fetchone()
        if not owned:
            return False
        conn.execute(
            "UPDATE pinned_chart SET layout_locked=0 WHERE dashboard_id=? AND user_id=? AND account_id=?",
            (int(dashboard_id), int(user_id), account_id),
        )
        _auto_layout_locked(conn, int(dashboard_id), int(user_id))
        conn.execute(
            """UPDATE dashboard_artifact
                  SET version=version+1, status='draft', updated_at=datetime('now')
                WHERE id=? AND user_id=? AND account_id=?""",
            (int(dashboard_id), int(user_id), account_id),
        )
        _snapshot_locked(conn, dashboard_id, user_id, account_id, "Layout tidied")
    return True


def subscribe_dashboard(
    dashboard_id: int,
    user_id: int,
    account_id: str,
    *,
    cadence: str = "daily",
    channel: str = "in_app",
) -> dict:
    """Follow a dashboard the user can currently view."""
    cadence = str(cadence or "daily").lower()
    channel = str(channel or "in_app").lower()
    if cadence not in {"daily", "weekly", "monthly"}:
        cadence = "daily"
    if channel not in {"in_app", "email"}:
        channel = "in_app"
    if not get_dashboard_for_view(dashboard_id, user_id, account_id):
        raise PermissionError("Dashboard is not available to this user and workspace.")
    with get_db() as conn:
        conn.execute(
            """INSERT INTO dashboard_subscription
                   (dashboard_id, account_id, user_id, cadence, channel, status)
               VALUES (?, ?, ?, ?, ?, 'active')
               ON CONFLICT(dashboard_id, user_id) DO UPDATE SET
                   cadence=excluded.cadence, channel=excluded.channel,
                   status='active', updated_at=datetime('now')""",
            (int(dashboard_id), account_id, int(user_id), cadence, channel),
        )
        row = conn.execute(
            """SELECT * FROM dashboard_subscription
                WHERE dashboard_id=? AND user_id=? AND account_id=?""",
            (int(dashboard_id), int(user_id), account_id),
        ).fetchone()
    return dict(row)


def get_dashboard_subscription(
    dashboard_id: int, user_id: int, account_id: str
) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            """SELECT * FROM dashboard_subscription
                WHERE dashboard_id=? AND user_id=? AND account_id=?""",
            (int(dashboard_id), int(user_id), account_id),
        ).fetchone()
    return dict(row) if row else None


def unsubscribe_dashboard(
    dashboard_id: int, user_id: int, account_id: str
) -> bool:
    with get_db() as conn:
        cur = conn.execute(
            """DELETE FROM dashboard_subscription
                WHERE dashboard_id=? AND user_id=? AND account_id=?""",
            (int(dashboard_id), int(user_id), account_id),
        )
    return bool(cur.rowcount)


# A failed scheduled refresh is tried again five minutes later, then after
# twice as long each time it fails again, and never less often than the
# dashboard's own cadence.
_FIRST_RETRY = timedelta(minutes=5)
# The owner is told once a refresh has failed this many times in a row: one
# failure the next attempt recovers from is no news.
_TELL_OWNER_AFTER = 2


def _cadence(schedule: object) -> timedelta:
    return timedelta(hours=1 if schedule == "hourly" else 24 if schedule == "daily" else 168)


_STAMP = "%Y-%m-%d %H:%M:%S"


def _utc_now() -> datetime:
    """Now, in UTC, as the database's datetime('now') gives it.

    Every time the cache keeps is in UTC. A refresh was stamped in UTC and its
    expiry, its failures and its next attempt in the server's local time, so
    on a server not on UTC a notice said the data was hours newer than the
    failures, and an hourly source refreshed a minute ago was due again.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def save_source_cache(
    source: dict,
    rows: list[dict],
    *,
    policy_version: int,
    contract_version: str,
    ttl_seconds: int,
) -> None:
    """Encrypt governed release rows for the source owner."""
    expires = _utc_now() + timedelta(seconds=max(60, int(ttl_seconds or 600)))
    token = encrypt({"rows": list(rows or [])})
    with get_db() as conn:
        conn.execute(
            """INSERT INTO dashboard_source_cache
                   (source_id, dashboard_id, account_id, user_id, rows_encrypted,
                    row_count, policy_version, contract_version, status,
                    error_message, refreshed_at, expires_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'ready', '', datetime('now'), ?)
               ON CONFLICT(source_id) DO UPDATE SET
                   rows_encrypted=excluded.rows_encrypted,
                   row_count=excluded.row_count,
                   policy_version=excluded.policy_version,
                   contract_version=excluded.contract_version,
                   status='ready', error_message='',
                   refreshed_at=datetime('now'), expires_at=excluded.expires_at,
                   failed_at=NULL, failure_count=0, next_attempt_at=NULL""",
            (
                int(source["id"]), int(source["dashboard_id"]), source["account_id"],
                int(source["user_id"]), token, len(rows or []),
                max(0, int(policy_version or 0)), str(contract_version or "")[:128],
                expires.strftime(_STAMP),
            ),
        )
        # The owner was told of this dashboard's failures once; once none of
        # its sources is failing, the next run of failures is news again.
        conn.execute(
            """UPDATE dashboard_source_cache SET owner_notified_at=NULL
                WHERE dashboard_id=? AND owner_notified_at IS NOT NULL
                  AND NOT EXISTS (SELECT 1 FROM dashboard_source_cache
                                   WHERE dashboard_id=? AND failure_count > 0)""",
            (int(source["dashboard_id"]), int(source["dashboard_id"])),
        )
        conn.execute(
            """UPDATE dashboard_artifact SET last_refreshed_at=datetime('now')
                WHERE id=? AND user_id=? AND account_id=?""",
            (int(source["dashboard_id"]), int(source["user_id"]), source["account_id"]),
        )


def mark_source_cache_error(source: dict, error: str, *, now: datetime | None = None) -> dict:
    """Record a refresh of ``source`` that failed, and when to try it again.

    The rows of the last refresh that worked are kept, and so is its time: a
    failure is not a refresh. It was stamped as one, so rows a day old read as
    refreshed a moment ago; and a source never refreshed recorded nothing, and
    was due again at the next tick of the scheduler, every minute, with no one
    told. Now each failure in a row waits twice as long as the one before it,
    from five minutes up to the dashboard's own cadence, and a first failure is
    kept in a row no reader can serve: expired, and with no rows.

    The failure is counted by the first write, so two refreshes failing at
    once count two. Returns the failures in a row -- ``failure_count``,
    ``failed_at`` (when the first of them failed) and ``next_attempt_at`` --
    and ``owner_due``: the run is ``_TELL_OWNER_AFTER`` long or longer and
    nobody has told the owner of it (claim_owner_notice).
    """
    now = now or _utc_now()
    stamp = now.strftime(_STAMP)
    source_id, account_id, user_id = int(source["id"]), source["account_id"], int(source["user_id"])
    with get_db() as conn:
        conn.execute(
            """INSERT INTO dashboard_source_cache
                   (source_id, dashboard_id, account_id, user_id, rows_encrypted, row_count, status,
                    error_message, refreshed_at, expires_at, failed_at, failure_count)
               VALUES (?, ?, ?, ?, '', 0, 'error', ?, NULL, ?, ?, 1)
               ON CONFLICT(source_id) DO UPDATE SET
                   status='error', error_message=excluded.error_message,
                   failure_count=dashboard_source_cache.failure_count + 1,
                   failed_at=COALESCE(dashboard_source_cache.failed_at, excluded.failed_at)
               WHERE dashboard_source_cache.account_id=excluded.account_id
                 AND dashboard_source_cache.user_id=excluded.user_id""",
            (source_id, int(source["dashboard_id"]), account_id, user_id, str(error or "")[:500], stamp, stamp),
        )
        row = conn.execute(
            """SELECT failure_count, failed_at, owner_notified_at FROM dashboard_source_cache
                WHERE source_id=? AND account_id=? AND user_id=?""",
            (source_id, account_id, user_id),
        ).fetchone()
        count = int((row["failure_count"] if row else 0) or 0)
        wait = min(_FIRST_RETRY * 2 ** min(max(count, 1) - 1, 20), _cadence(source.get("refresh_schedule")))
        next_attempt = (now + wait).strftime(_STAMP)
        conn.execute(
            "UPDATE dashboard_source_cache SET next_attempt_at=? WHERE source_id=? AND account_id=? AND user_id=?",
            (next_attempt, source_id, account_id, user_id),
        )
    return {
        "failure_count": count,
        "failed_at": str((row["failed_at"] if row else "") or stamp),
        "next_attempt_at": next_attempt,
        "owner_due": bool(row) and count >= _TELL_OWNER_AFTER and not row["owner_notified_at"],
    }


def claim_owner_notice(source: dict, *, now: datetime | None = None) -> str | None:
    """Claim the one notice a dashboard's owner gets for a run of failed
    refreshes. The stamp it was claimed under, to release it by if no channel
    delivers it; None when it is already claimed, by this source or another
    source of the same dashboard -- a dashboard of three charts sent its owner
    three notices of the same outage.

    One conditional write, so of two refreshes failing at once, one claims it.
    """
    stamp = (now or _utc_now()).strftime(_STAMP)
    with get_db() as conn:
        claimed = conn.execute(
            """UPDATE dashboard_source_cache SET owner_notified_at=?
                WHERE source_id=? AND account_id=? AND user_id=? AND owner_notified_at IS NULL
                  AND NOT EXISTS (SELECT 1 FROM dashboard_source_cache
                                   WHERE dashboard_id=? AND owner_notified_at IS NOT NULL)""",
            (stamp, int(source["id"]), source["account_id"], int(source["user_id"]), int(source["dashboard_id"])),
        ).rowcount
    return stamp if claimed == 1 else None


def release_owner_notice(source: dict, stamp: str) -> None:
    """Give back a claim whose notice no channel delivered -- an owner with no
    portal page open and no Teams chat -- so that the next failure tries
    again. It was marked as told in the same write that counted the failure,
    before anything was sent, and an owner who was away was never told."""
    with get_db() as conn:
        conn.execute(
            "UPDATE dashboard_source_cache SET owner_notified_at=NULL WHERE source_id=? AND owner_notified_at=?",
            (int(source["id"]), stamp),
        )


def get_source_cache(
    source_id: int,
    user_id: int,
    account_id: str,
    *,
    allow_stale: bool = False,
) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            """SELECT * FROM dashboard_source_cache
                WHERE source_id=? AND user_id=? AND account_id=?""",
            (int(source_id), int(user_id), account_id),
        ).fetchone()
    if not row:
        return None
    result = dict(row)
    # Rows a failing refresh kept are served until a refresh works: the owner
    # otherwise saw an error once they expired, while the notice said the
    # dashboard still showed them. The chart says how old they are.
    if (
        not allow_stale
        and not result.get("failed_at")
        and str(result.get("expires_at") or "") <= _utc_now().strftime(_STAMP)
    ):
        return None
    try:
        payload = decrypt_json(result.pop("rows_encrypted"))
        result["rows"] = list(payload.get("rows") or [])
    except Exception:
        return None
    return result


def list_due_dashboard_sources(now: datetime | None = None) -> list[dict]:
    """Return scheduled sources whose owner-scoped cache is due. ``now`` is
    in UTC, as every time the cache keeps is."""
    now = now or _utc_now()
    with get_db() as conn:
        rows = conn.execute(
            """SELECT s.*, d.refresh_schedule, d.status AS dashboard_status,
                      d.name AS dashboard_name,
                      c.refreshed_at AS cache_refreshed_at,
                      c.expires_at AS cache_expires_at,
                      c.next_attempt_at AS cache_next_attempt_at
                 FROM dashboard_data_source s
                 JOIN dashboard_artifact d ON d.id=s.dashboard_id
                 LEFT JOIN dashboard_source_cache c ON c.source_id=s.id
                WHERE d.refresh_schedule<>'manual'
                ORDER BY s.id"""
        ).fetchall()
    due: list[dict] = []
    stamp = now.strftime(_STAMP)
    for raw in rows:
        source = dict(raw)
        # A source whose refresh failed waits for its next attempt.
        if str(source.get("cache_next_attempt_at") or "") > stamp:
            continue
        if str(source.get("cache_expires_at") or "") <= stamp:
            due.append(source)
            continue
        refreshed = source.get("cache_refreshed_at")
        if not refreshed:
            due.append(source)
            continue
        try:
            last = datetime.strptime(str(refreshed)[:19], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            due.append(source)
            continue
        if now - last >= _cadence(source.get("refresh_schedule")):
            due.append(source)
    return due


def rename_dashboard(
    dashboard_id: int, user_id: int, account_id: str, name: str
) -> dict | None:
    with get_db() as conn:
        conn.execute(
            """
            UPDATE dashboard_artifact
               SET name=?, version=version+1, updated_at=datetime('now')
             WHERE id=? AND user_id=? AND account_id=?
            """,
            (_clean_name(name), int(dashboard_id), int(user_id), account_id),
        )
        _snapshot_locked(conn, dashboard_id, user_id, account_id, "Dashboard renamed")
    return get_dashboard(dashboard_id, user_id, account_id)


def publish_dashboard(dashboard_id: int, user_id: int, account_id: str) -> dict | None:
    with get_db() as conn:
        conn.execute(
            """
            UPDATE dashboard_artifact
               SET status='published', published_at=datetime('now'),
                   updated_at=datetime('now')
             WHERE id=? AND user_id=? AND account_id=?
            """,
            (int(dashboard_id), int(user_id), account_id),
        )
        _snapshot_locked(conn, dashboard_id, user_id, account_id, "Dashboard published")
    return get_dashboard(dashboard_id, user_id, account_id)


def mark_dashboard_draft(
    dashboard_id: int, user_id: int, account_id: str
) -> dict | None:
    with get_db() as conn:
        conn.execute(
            """
            UPDATE dashboard_artifact
               SET status='draft', version=version+1, updated_at=datetime('now')
             WHERE id=? AND user_id=? AND account_id=?
            """,
            (int(dashboard_id), int(user_id), account_id),
        )
        _snapshot_locked(conn, dashboard_id, user_id, account_id, "Dashboard edited")
    return get_dashboard(dashboard_id, user_id, account_id)


def update_dashboard_controls(
    dashboard_id: int,
    user_id: int,
    account_id: str,
    *,
    visibility: str | None = None,
    refresh_schedule: str | None = None,
    filters: list[dict] | None = None,
    tabs: list[str] | None = None,
    change_summary: str = "Dashboard controls updated",
) -> dict | None:
    assignments = ["version=version+1", "status='draft'", "updated_at=datetime('now')"]
    params: list = []
    if visibility is not None:
        value = str(visibility).lower()
        if value not in {"personal", "team"}:
            raise ValueError("Visibility must be personal or team.")
        assignments.append("visibility=?")
        params.append(value)
    if refresh_schedule is not None:
        value = str(refresh_schedule).lower()
        if value not in {"manual", "hourly", "daily", "weekly"}:
            raise ValueError("Refresh schedule must be manual, hourly, daily, or weekly.")
        assignments.append("refresh_schedule=?")
        params.append(value)
    if filters is not None:
        safe_filters = []
        for item in filters[:12]:
            if not isinstance(item, dict):
                continue
            field = re.sub(r"[^A-Za-z0-9_. -]", "", str(item.get("field") or "")).strip()[:120]
            if field:
                operator = str(item.get("operator") or "equals").lower()
                if operator not in {"equals", "contains", "gte", "lte"}:
                    operator = "equals"
                safe_filters.append({
                    "field": field,
                    "label": str(item.get("label") or field)[:120],
                    "type": str(item.get("type") or "text")[:24],
                    "operator": operator,
                })
        assignments.append("filters_json=?")
        params.append(_json(safe_filters))
    if tabs is not None:
        safe_tabs: list[str] = []
        for raw in tabs[:8]:
            tab = _clean_name(raw)[:60]
            if tab.lower() not in {value.lower() for value in safe_tabs}:
                safe_tabs.append(tab)
        assignments.append("tabs_json=?")
        params.append(_json(safe_tabs or ["Overview"]))
    params.extend((int(dashboard_id), int(user_id), account_id))
    with get_db() as conn:
        cur = conn.execute(
            f"UPDATE dashboard_artifact SET {', '.join(assignments)} WHERE id=? AND user_id=? AND account_id=?",
            params,
        )
        if cur.rowcount == 0:
            return None
        _snapshot_locked(conn, dashboard_id, user_id, account_id, change_summary)
    return get_dashboard(dashboard_id, user_id, account_id)


def list_dashboard_versions(dashboard_id: int, user_id: int, account_id: str) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            """SELECT id, dashboard_id, version, change_summary, created_at
                 FROM dashboard_artifact_version
                WHERE dashboard_id=? AND user_id=? AND account_id=?
                ORDER BY version DESC""",
            (int(dashboard_id), int(user_id), account_id),
        ).fetchall()
    return [dict(row) for row in rows]


def rollback_dashboard(
    dashboard_id: int, user_id: int, account_id: str, target_version: int
) -> dict | None:
    """Restore a presentation checkpoint while preserving a new audit version."""
    with get_db() as conn:
        version_row = conn.execute(
            """SELECT snapshot_json FROM dashboard_artifact_version
                WHERE dashboard_id=? AND user_id=? AND account_id=? AND version=?""",
            (int(dashboard_id), int(user_id), account_id, int(target_version)),
        ).fetchone()
        if not version_row:
            return None
        snapshot = json.loads(version_row["snapshot_json"] or "{}")
        meta = snapshot.get("dashboard") or {}
        conn.execute(
            """UPDATE dashboard_artifact
                  SET name=?, description=?, status='draft', visibility=?,
                      refresh_schedule=?, filters_json=?, tabs_json=?,
                      version=version+1, updated_at=datetime('now')
                WHERE id=? AND user_id=? AND account_id=?""",
            (
                _clean_name(meta.get("name")), str(meta.get("description") or "")[:500],
                str(meta.get("visibility") or "personal"), str(meta.get("refresh_schedule") or "manual"),
                str(meta.get("filters_json") or "[]"), str(meta.get("tabs_json") or '["Overview"]'),
                int(dashboard_id), int(user_id), account_id,
            ),
        )
        conn.execute(
            "UPDATE pinned_chart SET dashboard_id=NULL WHERE dashboard_id=? AND user_id=?",
            (int(dashboard_id), int(user_id)),
        )
        for chart in list(snapshot.get("charts") or []):
            conn.execute(
                """UPDATE pinned_chart SET dashboard_id=?, data_source_id=?, title=?,
                          chart_type=?, color_palette=?, position=?, grid_x=?, grid_y=?,
                          grid_w=?, grid_h=?, display_config=?, dashboard_tab=?, sort_enabled=?
                    WHERE id=? AND user_id=? AND account_id=?""",
                (
                    int(dashboard_id), chart.get("data_source_id"), str(chart.get("title") or "")[:120],
                    str(chart.get("chart_type") or "bar"), str(chart.get("color_palette") or "default"),
                    int(chart.get("position") or 0), int(chart.get("grid_x") or 0), int(chart.get("grid_y") or 0),
                    int(chart.get("grid_w") or 6), int(chart.get("grid_h") or 5),
                    str(chart.get("display_config") or "{}"), str(chart.get("dashboard_tab") or "Overview")[:60],
                    1 if chart.get("sort_enabled", 1) else 0, int(chart.get("id") or 0), int(user_id), account_id,
                ),
            )
        _snapshot_locked(conn, dashboard_id, user_id, account_id, f"Rolled back to version {int(target_version)}")
    return get_dashboard(dashboard_id, user_id, account_id)
