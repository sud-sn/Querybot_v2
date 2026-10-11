"""Dashboards a reader describes: "I need revenue, cost % and fills; revenue by month as a line".

Recognised in code, before any plan is asked for. One AI call designs the tiles from the workspace's measures and
fields (their names only: no member value, and the request as the AI may see it): one tile for each thing the
request names, titled in the reader's own words, and a few more it did not ask for, offered as chips and never
added. Each tile is then a question of its own, planned and answered as a new conversation would be, several at
once, and the reader's card fills in as each one lands. A tile that cannot be built says why, and when the data has
nothing that measures it, offers to define it. The draft (each tile's plan and its answer) is what the reader
previews in the chat: they can rename it, take tiles out, add the ones offered, and save it as a new dashboard or
onto one of theirs, as one version. Its tiles are drawn again from their plans, with no AI.

Layout: the numbers in one row (at most six), a trend across the width (two side by side), breakdowns side by
side, tables across the width, and at most 12 tiles on a tab (the rest on a second).
"""

from __future__ import annotations

import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger("core2.dashboards")

MOST_TILES = 12           # on one tab
MOST_NUMBERS = 6          # in the row of numbers
AT_ONCE = 4               # tiles built at the same time
MOST_EXTRAS = 4           # tiles offered beside what was asked for
GRID = 12                 # the dashboard's columns
HEIGHT = {"number": 3, "trend": 10, "breakdown": 10, "table": 10}

_ASKS = re.compile(r"^\s*(?:please\s+)?(?:(?:can|could|would) you\s+)?(?:build|make|create|design|set up|give me|"
                   r"i (?:want|need|would like))\s+(?:me\s+|us\s+)?(?:(?:a|an)\s+)?(?:new\s+)?"
                   r"(?P<before>(?:[\w&'-]+\s+){0,4}?)dashboards?\b(?P<about>.*)$", re.IGNORECASE)
_DEMANDE = re.compile(r"^\s*(?:(?:peux|pourrais)-tu\s+|(?:pouvez|pourriez)-vous\s+)?(?:cr[ée]e[rz]?|construi(?:s|re|sez)|"
                      r"fai(?:s|re|tes)|pr[ée]pare[rz]?|je veux|j'aimerais|il me faut)(?:-moi|\s+moi)?\s+un\s+"
                      r"(?:nouveau\s+)?tableau de bord\b(?P<about>.*)$", re.IGNORECASE)
# What follows "dashboard" in a request for a new one: nothing, or what it is about. "the dashboard chart",
# "my dashboard to refresh" or "a new tab on the dashboard" are about a dashboard the reader has, never a new one.
_ABOUT = re.compile(r"^\s*(?:$|[.?!,:;]|(?:for|of|on|about|with|showing|covering|that shows|which shows|to show|by|"
                    r"across|from|des|de|du|sur|pour|avec|montrant|par)\b)", re.IGNORECASE)
_CONNECTOR = re.compile(r"^\s*(?:for|of|on|about|with|showing|covering|that shows|which shows|to show|des|de|du|"
                        r"sur|pour|avec|montrant)?\s*", re.IGNORECASE)
_NOT_ABOUT = {"on", "in", "of", "to", "the", "my", "your", "our", "his", "her", "their", "this", "that", "for", "at",
              "with", "from", "tab", "page", "chart", "tile", "link", "filter", "filters", "existing", "current",
              "same", "shared", "old"}
_POLITE = re.compile(r"[\s.?!]*(?:\b(?:please|thanks|thank you|merci|s'il (?:te|vous) pla[iî]t))?[\s.?!]*$",
                     re.IGNORECASE)

RULES = """You design a dashboard for QueryBot from a reader's request. You never write SQL.
Answer with exactly one JSON object: {"name": "<a short name>", "tiles": [{"title": "...", "kind": "number|trend|breakdown|table", "question": "..."}], "extras": [<tiles in the same shape>]}.
- tiles: what the request asks for, one tile for each thing it names, in its order, each titled in the reader's own
  words ("I need revenue, cost % and fills": three number tiles titled "Revenue", "Cost %" and "Fills"). When the
  request names only a subject ("a sales dashboard"), choose 4 to 8 tiles that show it: the headline numbers
  first, then trends, then breakdowns, then at most two tables. At most 12 tiles.
- extras: at most 4 more tiles that would help and that the request did not ask for. They are offered to the
  reader as suggestions, never added.
- kind: number for one figure (one measure each: "revenue and cost" is two numbers), trend for a figure over
  time, breakdown for a figure by a grouping, table for a list or a ranking of rows.
- Each question is a complete question a reader could ask on its own, in plain words, using the measure and field
  names of the catalog below ("Net amount by month for the last 12 months", "Top 10 customers by net amount this
  year"). Never a member's name or a value: only names of measures, fields and periods.
- A title is what the tile shows, in a few words ("Net amount", "Net amount by month"), never a question.
- When the request names something the catalog has not, still make its tile, titled in the reader's words and
  asking for exactly what they named: the reader is told it is missing and can define it.
- Write the name, the titles and the questions in the language of the request.
"""


def asked_for(question: str) -> str | None:
    """What a request for a new dashboard is about ("" when it names nothing), or None when it asks for none."""
    asked = _POLITE.sub("", question or "")
    found = _ASKS.match(asked) or _DEMANDE.match(asked)
    if found is None or not _ABOUT.match(found.group("about")):
        return None
    before = found.groupdict().get("before") or ""          # "a sales dashboard": about sales
    if any(w.casefold() in _NOT_ABOUT for w in before.split()):
        return None
    about = _CONNECTOR.sub("", found.group("about"), count=1).strip(" .?!,:;")
    return " ".join(w for w in (before.strip(), about) if w)


@dataclass
class Tile:
    title: str
    kind: str                       # number | trend | breakdown | table
    question: str
    plan: dict | None = None        # the plan its answer was given from: the tile is drawn again from it
    payload: dict | None = None     # its answer, for the preview (never sent with its rows to the AI)
    error: str = ""                 # why it could not be built
    definable: bool = False         # it failed for want of a measure: the reader is offered to define it
    layout: dict = field(default_factory=dict)   # grid_x, grid_y, grid_w, grid_h, tab


@dataclass
class Draft:
    name: str
    request: str
    tiles: list[Tile]
    extras: list[Tile] = field(default_factory=list)     # offered as chips, never built unless the reader adds one


def design(model, request: str, complete: Callable[[str, str], str], *, scrub: Callable[[str], str] | None = None,
           personal_shown: bool = False) -> tuple[str, list[Tile], list[Tile]]:
    """The dashboard's name, its tiles' questions and the tiles offered beside them, from one AI call that sees
    names, never values."""
    from core2.plan.catalog import catalog_text

    stable = RULES + "\n\n" + catalog_text(model, values_allowed=False, personal_shown=personal_shown)
    asked = scrub(request) if scrub else request
    text = complete(stable, f"REQUEST: {asked or 'the main figures of this data'}")
    try:
        data = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", str(text).strip(), flags=re.MULTILINE))
    except ValueError:
        raise DesignError("the design could not be read") from None
    if not isinstance(data, dict):
        raise DesignError("the design could not be read")
    tiles = _tiles(data.get("tiles"), MOST_TILES)
    if not tiles:
        raise DesignError("nothing in this data fits that dashboard")
    asked = {t.question.casefold() for t in tiles}
    extras = [t for t in _tiles(data.get("extras"), MOST_EXTRAS * 2) if t.question.casefold() not in asked]
    return str(data.get("name") or "Dashboard")[:80], tiles, extras[:MOST_EXTRAS]


def _tiles(items: Any, most: int) -> list[Tile]:
    out = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or not str(item.get("question") or "").strip():
            continue
        kind = str(item.get("kind") or "breakdown")
        out.append(Tile(title=str(item.get("title") or item["question"]).strip()[:80],
                        kind=kind if kind in HEIGHT else "breakdown", question=str(item["question"]).strip()))
    return out[:most]


class DesignError(ValueError):
    pass


def build(model, request: str, complete: Callable[[str, str], str], answer: Callable[[str], dict], *,
          scrub: Callable[[str], str] | None = None, personal_shown: bool = False,
          progress: Callable[[Draft], None] | None = None) -> Draft:
    """The draft: designed by one AI call, each tile answered on its own (several at once), laid out.

    ``answer(question)`` answers a question as a new conversation would (its own planner call, its own query).
    A tile that cannot be built says why; the others are built. ``progress(draft)`` is told the draft once it is
    designed and again as each tile lands, from the thread that built it, one call at a time."""
    name, tiles, extras = design(model, request, complete, scrub=scrub, personal_shown=personal_shown)
    draft = Draft(name=name, request=request, tiles=tiles, extras=extras)
    lay_out(list(tiles))                           # where each tile goes, while they are built
    told = threading.Lock()

    def tell() -> None:
        if progress is None:
            return
        with told:
            try:
                progress(draft)
            except Exception as exc:  # noqa: BLE001 - the card filling in is a courtesy; the draft is still built
                log.warning("core2 could not show a dashboard's progress: %s", exc)

    def one(tile: Tile) -> None:
        answered(tile, answer)
        tell()

    tell()
    with ThreadPoolExecutor(AT_ONCE, thread_name_prefix="qb-dashboard") as pool:
        list(pool.map(one, tiles))
    for tile in tiles:
        if tile.error:
            tile.layout = {}
    lay_out([t for t in tiles if not t.error])
    return draft


def answered(tile: Tile, answer: Callable[[str], dict]) -> Tile:
    """The tile with its answer, or why it could not be built: when the data has nothing that measures it (the
    planner found none, or asked what was meant), the reader is offered to define it."""
    try:
        payload = answer(tile.question)
    except Exception as exc:  # noqa: BLE001 - one tile failing never takes the dashboard with it
        log.warning("core2 could not build the tile %r: %s", tile.title, exc)
        tile.error = "This tile could not be built."
        return tile
    if payload.get("data") is None or not payload.get("plan"):
        tile.error = str((payload.get("answer") or {}).get("headline") or payload.get("text")
                         or "This tile could not be built.")
        tile.definable = bool(payload.get("unsupported") or payload.get("clarify"))
        return tile
    tile.plan, tile.payload = payload["plan"], payload
    return tile


def relayout(kept_tiles: list[dict]) -> None:
    """Places for a draft's kept tiles again, once one is taken out or added, in their order."""
    tiles = [Tile(title=str(k.get("title") or ""), kind=k.get("kind") if k.get("kind") in HEIGHT else "breakdown",
                  question="") for k in kept_tiles]
    lay_out(list(tiles))
    for k, t in zip(kept_tiles, tiles):
        k["layout"] = dict(t.layout)


def lay_out(tiles: list[Tile]) -> None:
    """Grid places: the numbers in a row, a trend across (or two side by side), breakdowns side by side, tables
    across; at most 12 tiles on a tab, the rest on the next."""
    order = {"number": 0, "trend": 1, "breakdown": 2, "table": 3}
    tiles.sort(key=lambda t: order.get(t.kind, 2))
    for tab in range(0, len(tiles), MOST_TILES):
        _tab(tiles[tab:tab + MOST_TILES], tab // MOST_TILES)


def _tab(tiles: list[Tile], tab: int) -> None:
    y = 0

    def place(t: Tile, x: int, width: int) -> None:
        t.layout = {"grid_x": x, "grid_y": y, "grid_w": width, "grid_h": HEIGHT[t.kind], "tab": tab}

    numbers = [t for t in tiles if t.kind == "number"]
    for row in range(0, len(numbers), MOST_NUMBERS):
        line = numbers[row:row + MOST_NUMBERS]
        for i, t in enumerate(line):
            place(t, i * (GRID // len(line)), GRID // len(line))
        y += HEIGHT["number"]
    for kind, alone_full in (("trend", True), ("breakdown", False)):
        group = [t for t in tiles if t.kind == kind]
        if len(group) == 1 and alone_full:
            place(group[0], 0, GRID)                # one trend across the width
            y += HEIGHT[kind]
            continue
        for i in range(0, len(group), 2):           # side by side, two to a row
            pair = group[i:i + 2]
            for j, t in enumerate(pair):
                place(t, j * (GRID // 2), GRID if len(pair) == 1 and alone_full else GRID // 2)
            y += HEIGHT[kind]
    for t in (t for t in tiles if t.kind == "table"):
        place(t, 0, GRID)
        y += HEIGHT["table"]


def preview(draft: Draft) -> dict[str, Any]:
    """The draft as the chat shows it: each tile's name, kind, place and a small view of its answer, or why it
    failed, or that it is still being built; and the tiles offered beside it."""
    out = [preview_tile(t, i) for i, t in enumerate(draft.tiles)]
    return {"name": draft.name, "request": draft.request, "tiles": out,
            "extras": [{"title": e.title, "kind": e.kind, "question": e.question} for e in draft.extras],
            "built": sum(1 for t in draft.tiles if t.payload is not None and not t.error),
            "failed": sum(1 for t in draft.tiles if t.error),
            "pending": sum(1 for t in draft.tiles if t.payload is None and not t.error)}


def preview_tile(t: Tile, i: int) -> dict[str, Any]:
    answer = (t.payload or {}).get("answer") or {}
    kpi = (t.payload or {}).get("kpi") or {}
    return {"id": i, "title": t.title, "kind": t.kind, "question": t.question,
            "value": str(answer.get("short_value") or "") if kpi else "",
            "headline": str(answer.get("headline") or ""), "error": t.error, "definable": t.definable,
            "pending": t.payload is None and not t.error, "layout": t.layout,
            "chart": _small(t.payload, t.kind) if t.payload else None}


def _small(payload: dict, kind: str) -> dict | None:
    """A tile's answer drawn small in the preview: a trend's latest points, a breakdown's first bars, a table's
    first rows. Never all its rows."""
    chart = payload.get("chart") or {}
    data = payload.get("data") or {}
    if payload.get("kpi") and not chart:
        return None                                  # a number: its value says it
    x = chart.get("x_key")
    ys = [y for y in (chart.get("y_keys") or []) if isinstance(y, str)]
    if kind == "table" or not (x and ys):
        rows = data.get("rows") or []
        columns = [str(c) for c in (data.get("headers") or [])][:3]
        labels = data.get("header_labels") if isinstance(data.get("header_labels"), dict) else {}
        formats = data.get("column_formats") if isinstance(data.get("column_formats"), dict) else {}
        return {"chart_type": "table", "columns": [str(labels.get(c) or c) for c in columns],
                "rows": [[_cell(r, c, i, formats.get(c)) for i, c in enumerate(columns)] for r in rows[:4]]}
    rows = chart.get("rows") or []
    taken = rows[-24:] if kind == "trend" else rows[:6]
    return {"chart_type": "line" if kind == "trend" else "bar", "x_key": x, "y_key": ys[0],
            "rows": [{"x": str(r.get(x) if isinstance(r, dict) else ""),
                      "y": r.get(ys[0]) if isinstance(r, dict) and isinstance(r.get(ys[0]), (int, float)) else None}
                     for r in taken]}


def _cell(row: Any, column: str, at: int, format_: str | None) -> str:
    from core2.answer.builder import fmt

    value = row.get(column) if isinstance(row, dict) else (row[at] if isinstance(row, (list, tuple)) and
                                                             at < len(row) else None)
    if value is None:
        return ""
    if format_ and isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return fmt(value, format_)[:40]
        except Exception:  # noqa: BLE001 - a cell the preview cannot format is shown as it came
            pass
    return str(value)[:40]


def kept(tile: Tile) -> dict[str, Any]:
    """A built tile as the draft keeps it on the server until it is saved: its plan (the tile is drawn again from
    it, with no AI), its place, and how its answer was drawn. Never its rows."""
    payload = tile.payload or {}
    chart = payload.get("chart") or {}
    kpi = payload.get("kpi") or {}
    chart_type = "kpi" if kpi and tile.kind == "number" else str(chart.get("chart_type") or ("kpi" if kpi else "table"))
    return {"title": tile.title, "kind": tile.kind, "question": tile.question, "plan": tile.plan,
            "layout": dict(tile.layout), "chart_type": chart_type,
            "sql": str((payload.get("trust") or {}).get("sql") or ""),
            "display_config": {"core2_plan": tile.plan, "column_formats": chart.get("column_formats") or {},
                               "title": tile.title}}
