"""
core/what_changed.py

The "What changed" tile: each of a dashboard's numbers against the period before, and the members that moved it
most, worked out for whoever opens the dashboard.

    the figures   the dashboard's number tiles that carry a plan of the new core, in its order (MAX_FIGURES at most):
                  each plan asked again as "why did it change" (core2/answer/drivers.py) on the same governed path as
                  the tile, so the reader's tables, row rules and masking apply. For each: its value now, its change on
                  the period before, and the grouping that explains most of the change with its leading members.
    in short      two or three sentences by the workspace's AI, written from those figures alone, when the workspace
                  lets them reach it (core2/service.py portal_summary: a workspace that keeps member names from the AI
                  gets none, and a sentence with any figure not among them is not shown). Kept until the figures
                  change, so the AI is asked once per change, not once per visit.

The tile is drawn like any other (portal/routes.py _refresh_chart), so it is in the dashboard's PDF and in a
scheduled email too.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Optional

import store

log = logging.getLogger("querybot.what_changed")

MAX_FIGURES = 4
LEADERS = 3
TITLE = "What changed"


def _number_tiles(chart: dict) -> list[dict]:
    """The number tiles of the tile's own dashboard, in the order they are laid out, with their plans."""
    siblings = store.list_dashboard_charts(int(chart["dashboard_id"]), int(chart["user_id"]))
    found = []
    for sibling in sorted(siblings, key=lambda c: (int(c.get("grid_y") or 0), int(c.get("grid_x") or 0), int(c["id"]))):
        if int(sibling["id"]) == int(chart["id"]) or str(sibling.get("chart_type") or "") != "kpi":
            continue
        try:
            display = json.loads(sibling.get("display_config") or "{}")
        except (TypeError, ValueError):
            continue
        plan = display.get("core2_plan") if isinstance(display, dict) else None
        if isinstance(plan, dict) and plan.get("measures"):
            found.append({"chart": sibling, "plan": plan})
        if len(found) >= MAX_FIGURES:
            break
    return found


def _figure(account_id: str, user: dict, tile: dict) -> Optional[dict]:
    """One number against the period before, and what moved it, as ``user`` may see it; None when it cannot be
    worked out (no comparison, refused, failed)."""
    from core2.answer.drivers import _signed
    from core2.service import portal_replay

    plan = {**tile["plan"], "intent": "drivers"}
    title = str(tile["chart"].get("title") or "")
    try:
        payload = portal_replay(account_id, plan, user, question=f"What changed in {title.lower()}?")
    except Exception as exc:  # noqa: BLE001 - one figure failing leaves the others
        log.warning("What changed: %s could not be worked out: %s", title, exc)
        return None
    answer = payload.get("answer") or {}
    drivers = payload.get("drivers") or {}
    if not answer.get("comparison") or not isinstance(drivers, dict):
        return None
    change = float(drivers.get("change") or 0.0)
    formats = (payload.get("data") or {}).get("column_formats") or {}
    format_ = str(formats.get("change") or "number")
    grouping = next((g for g in drivers.get("groupings") or [] if g.get("leaders")), None)
    leaders = []
    if grouping:
        leaders = [{"member": str(x["member"]), "change": _signed(float(x["change"]), format_)}
                   for x in grouping["leaders"][:LEADERS]]
    # The grouping by its word ("store"), as the drivers' own sentence says it, not its column ("Store name").
    word = re.sub(r"\s+(name|code|id|label)$", "", str((grouping or {}).get("label") or ""), flags=re.IGNORECASE)
    return {"title": title, "value": str(answer.get("short_value") or ""), "comparison": str(answer["comparison"]),
            "direction": "up" if change > 0 else ("down" if change < 0 else "flat"),
            "grouping": word, "leaders": leaders,
            "sentence": str(answer.get("headline") or "")}


def _facts(figures: list[dict]) -> list[str]:
    lines = []
    for f in figures:
        line = f"{f['title']}: {f['value']}, {f['comparison']}."
        if f["leaders"]:
            line += " Most of it by " + f["grouping"].lower() + ": " + ", ".join(
                f"{x['member']} {x['change']}" for x in f["leaders"]) + "."
        lines.append(line)
    return lines


def _summary(account_id: str, user: dict, dashboard_id: int, figures: list[dict]) -> str:
    """The AI's few sentences from the figures alone, kept until they change; "" when the workspace keeps them from the
    AI, no AI is set up, or what it wrote held a figure of its own."""
    from core2.service import portal_summary

    facts = _facts(figures)
    lang = str(user.get("lang") or "en")
    digest = hashlib.sha256(json.dumps([facts, lang]).encode()).hexdigest()
    kept = store.get_what_changed_note(int(dashboard_id), int(user["id"]), digest)
    if kept is not None:
        return kept
    # The figures are labels the summary may quote; the members that moved them are members (summary.names_members),
    # which a workspace that keeps member names from the AI never sends.
    named = any(f["leaders"] for f in figures)
    headers = ["figure", "now", "change"] + (["most_of_it"] if named else [])
    rows = [{"figure": f["title"], "now": f["value"], "change": f["comparison"],
             **({"most_of_it": ", ".join(f"{x['member']} {x['change']}" for x in f["leaders"])} if named else {})}
            for f in figures]
    payload: dict[str, Any] = {
        "engine": "core2",
        "answer": {"headline": " ".join(f["sentence"] for f in figures if f["sentence"])},
        "key_insights": facts, "coverage_caveats": [],
        "data": {"headers": headers, "header_labels": {h: h.replace("_", " ") for h in headers}, "rows": rows,
                 "total_rows": len(rows), "column_formats": {"figure": "label", "now": "label", "change": "label",
                                                             "most_of_it": "text"}},
    }
    try:
        text = portal_summary(account_id, "What changed on this dashboard, and why?", payload) or ""
    except Exception as exc:  # noqa: BLE001 - the figures stand without it
        log.warning("What changed: no summary written: %s", exc)
        text = ""
    store.keep_what_changed_note(int(dashboard_id), int(user["id"]), digest, text)
    return text


def draw(chart: dict, user: dict) -> dict:
    """The tile's content for ``user``: {"figures": [...], "summary": str, "empty": bool}."""
    account_id = str(user["account_id"])
    figures = [f for f in (_figure(account_id, user, tile) for tile in _number_tiles(chart)) if f]
    if not figures:
        return {"figures": [], "summary": "", "empty": True}
    return {"figures": figures, "summary": _summary(account_id, user, int(chart["dashboard_id"]), figures),
            "empty": False}
