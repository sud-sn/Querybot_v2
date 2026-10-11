"""Dashboards a reader asks for in the chat: "create an executive dashboard", "a revenue dashboard", "I need
revenue by month, top 10 formulas and fills by pharmacy".

Recognised in code, before any plan is asked for. One AI call designs the tiles from the workspace's measures and
fields (their names only: no member value, and the request as the AI may see it). Each tile is then a question of
its own, planned and answered as a new conversation would be, several at once, while the chat says how many are
ready. The tiles built are saved at once as a new dashboard of the reader's (gateway/core2_bridge.py), drawn again
from their plans with no AI, and the chat says its name, its tabs and where it is. A tile that cannot be built
says why, and when the data has nothing that measures it, offers to define it.

Three kinds of request: a list of what the reader needs ("revenue by month, top 10 formulas and fills by
pharmacy": those tiles, in that order), a subject ("a revenue dashboard": its headline numbers, trend,
breakdowns and a top 10), or the whole business ("an executive dashboard": the headline figures of each main
area). A design too thin for a subject or the whole business (one tile for "a revenue dashboard") is asked for
again, once. A dashboard of several areas, or of more tiles than one page holds, is on named tabs: "Overview"
first, then a tab per area ("Revenue", "Fills").

Layout, on each tab: the numbers in one row (at most six), trends and breakdowns two side by side (one alone
across the width), tables across the width; at most 12 tiles on a tab, at most 16 on a dashboard.
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
MOST_ALL = 16             # on a dashboard: each tile is a planner call, and a workspace's AI has a rate a minute
MOST_TABS = 6
FEWEST = 3                # tiles a subject's or the whole business's dashboard has at least
MOST_NUMBERS = 6          # in the row of numbers
AT_ONCE = 4               # tiles built at the same time
GRID = 12                 # the dashboard's columns
HEIGHT = {"number": 3, "trend": 10, "breakdown": 10, "table": 10}

_ASKS = re.compile(r"^\s*(?:please\s+)?(?:(?:can|could|would) you\s+)?(?:build|make|create|design|set up|give me|"
                   r"prepare|generate|produce|draft|put together|show me|get me|"
                   r"i(?: want| need| would like|'d like|’d like))\s+(?:me\s+|us\s+)?(?:(?:a|an)\s+)?(?:new\s+)?"
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
Answer with exactly one JSON object: {"name": "<a short name>", "tiles": [{"title": "...", "kind": "number|trend|breakdown|table", "question": "...", "tab": "..."}]}.
- The request is one of three kinds:
  1. A list of what the reader needs ("revenue by month, top 10 formulas and fills by pharmacy", "I need revenue,
     cost % and fills"): one tile for each thing it names, in its order, titled in the reader's own words, and
     nothing else in tiles.
  2. A subject ("a revenue dashboard", "shipping performance", "our fills in 2026", "a sales and margin
     dashboard"): 6 to 10 tiles about that subject: its 2 to 4 headline numbers first, then its trend over time,
     then its breakdowns by the main groupings of the catalog, then at most two tables (a top 10). Never fewer
     than 4 tiles for a subject.
  3. The whole business ("an executive dashboard", "an overview", "a KPI dashboard", "a CFO dashboard", or no
     subject at all): the headline figures of each main area of the catalog (sales or revenue, volumes, costs,
     operations), with a trend and one or two breakdowns for each area: 10 to 16 tiles.
- tab: the tab the tile is on, in a word or two. A list of up to 12 tiles, or a subject's dashboard, is on one
  tab, "Overview". A dashboard of several areas (the whole business), or of more than 12 tiles, puts the headline
  numbers and the main trend on "Overview" and gives each area its own tab named for it ("Revenue", "Fills",
  "Shipping"); at most 6 tabs and at most 16 tiles in all.
- kind: number for one figure (one measure each: "revenue and cost" is two numbers), trend for a figure over
  time, breakdown for a figure by a grouping, table for a list or a ranking of rows.
- Each question is a complete question a reader could ask on its own, in plain words, using the measure and field
  names of the catalog below ("Net amount by month for the last 12 months", "Top 10 customers by net amount this
  year"), with the request's period when it names one, and the request's words that say which figure it means
  ("internet sales", "reseller orders": never "sales" alone when the request says whose). Never a member's name or
  a value: only names of measures, fields and periods.
- A title is what the tile shows, in a few words ("Net amount", "Net amount by month"), never a question.
- When the request names something the catalog has not, still make its tile, titled in the reader's words and
  asking for exactly what they named: the reader is told it is missing and can define it.
- Write the name, the titles, the tabs and the questions in the language of the request.
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
    payload: dict | None = None     # its answer (never sent with its rows to the AI)
    error: str = ""                 # why it could not be built
    definable: bool = False         # it failed for want of a measure: the reader is offered to define it
    layout: dict = field(default_factory=dict)   # grid_x, grid_y, grid_w, grid_h, tab (its place in the tabs)
    tab: str = ""                   # the tab the design put it on ("Overview", "Revenue")


@dataclass
class Draft:
    name: str
    request: str
    tiles: list[Tile]
    tabs: list[str] = field(default_factory=list)        # the tabs' names, in order; a tile's layout tab is a place here


def design(model, request: str, complete: Callable[[str, str], str], *, scrub: Callable[[str], str] | None = None,
           personal_shown: bool = False) -> tuple[str, list[Tile]]:
    """The dashboard's name and its tiles' questions, from one AI call that sees names, never values (two, when a
    subject's or the whole business's design came back too thin)."""
    from core2.plan.catalog import catalog_text

    stable = RULES + "\n\n" + catalog_text(model, values_allowed=False, personal_shown=personal_shown)
    asked = scrub(request) if scrub else request
    tail = f"REQUEST: {asked or 'the main figures of this data (the whole business)'}"
    text = complete(stable, tail)
    data = _read(text)
    tiles = _tiles(data.get("tiles"), MOST_ALL)
    if len(tiles) < FEWEST and not _lists(asked):
        # "A dashboard of our revenue" came back as one number: a subject's dashboard is never one tile.
        again = _read(complete(stable, f"{tail}\n\nYOUR PREVIOUS ANSWER:\n{text}\n\nPROBLEM: {len(tiles)} tile(s) "
                                       "is not a dashboard. The request names a subject or the whole business, not a "
                                       "list of tiles: return 6 to 10 tiles for a subject (its headline numbers, its "
                                       "trend, its breakdowns, a top 10), 10 to 16 for the whole business. "
                                       "Return the corrected design as one JSON object."))
        more = _tiles(again.get("tiles"), MOST_ALL)
        if len(more) > len(tiles):
            data, tiles = again, more
    if not tiles:
        raise DesignError("nothing in this data fits that dashboard")
    return str(data.get("name") or "Dashboard")[:80], tiles


def _read(text: str) -> dict[str, Any]:
    try:
        data = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", str(text).strip(), flags=re.MULTILINE))
    except ValueError:
        raise DesignError("the design could not be read") from None
    if not isinstance(data, dict):
        raise DesignError("the design could not be read")
    return data


# "revenue, cost % and fills", "revenue by month and top 10 formulas": a request that lists what it needs.
_LIST = re.compile(r",|;|\band\b.*\b(?:by|top|per)\b|\b(?:by|top|per)\b.*\band\b|\bet\b.*\bpar\b", re.IGNORECASE)


def _lists(request: str) -> bool:
    """Does the request list the tiles it needs, rather than name a subject?"""
    return bool(_LIST.search(request or ""))


def _tiles(items: Any, most: int) -> list[Tile]:
    out = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or not str(item.get("question") or "").strip():
            continue
        kind = str(item.get("kind") or "breakdown")
        tab = " ".join(str(item.get("tab") or "").split())[:30]
        out.append(Tile(title=str(item.get("title") or item["question"]).strip()[:80],
                        kind=kind if kind in HEIGHT else "breakdown", question=str(item["question"]).strip(),
                        tab=tab))
    return out[:most]


class DesignError(ValueError):
    pass


def build(model, request: str, complete: Callable[[str, str], str], answer: Callable[[str], dict], *,
          scrub: Callable[[str], str] | None = None, personal_shown: bool = False,
          progress: Callable[[Draft], None] | None = None) -> Draft:
    """The dashboard: designed by one AI call, each tile answered on its own (several at once), laid out.

    ``answer(question)`` answers a question as a new conversation would (its own planner call, its own query).
    A tile that cannot be built says why; the others are built. ``progress(draft)`` is told the draft once it is
    designed and again as each tile lands, from the thread that built it, one call at a time."""
    name, tiles = design(model, request, complete, scrub=scrub, personal_shown=personal_shown)
    draft = Draft(name=name, request=request, tiles=tiles)
    draft.tabs = lay_out(list(tiles))              # where each tile goes, while they are built
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
    draft.tabs = lay_out([t for t in tiles if not t.error])
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


def lay_out(tiles: list[Tile]) -> list[str]:
    """Grid places, tab by tab: the numbers in a row, trends and breakdowns two side by side (one alone across),
    tables across. The tabs are the design's, "Overview" first, at most six; a tile with none (or on a tab past
    the sixth) goes on the first, and a tab of more than 12 tiles goes on over a second ("Revenue 2"). Each tile's
    tab is set to the one it is on. Returns the tabs' names in order."""
    order = {"number": 0, "trend": 1, "breakdown": 2, "table": 3}
    named: dict[str, str] = {}                      # the tab's name as compared, as written first
    for t in tiles:
        if t.tab:
            named.setdefault(t.tab.casefold(), t.tab)
    tabs = sorted(named.values(), key=lambda n: 0 if n.casefold() in _FIRST_TABS else 1)[:MOST_TABS] or ["Overview"]
    on: dict[str, list[Tile]] = {}
    for t in tiles:
        on.setdefault(next((n for n in tabs if n.casefold() == t.tab.casefold()), tabs[0]), []).append(t)
    names: list[str] = []
    for tab in (n for n in tabs if n in on):
        group = sorted(on[tab], key=lambda t: order.get(t.kind, 2))
        for part in range(0, len(group), MOST_TILES):
            names.append(tab if part == 0 else f"{tab} {part // MOST_TILES + 1}")
            _tab(group[part:part + MOST_TILES], len(names) - 1)
            for t in group[part:part + MOST_TILES]:
                t.tab = names[-1]
    return names or ["Overview"]


_FIRST_TABS = {"overview", "summary", "headlines", "vue d'ensemble", "synthèse"}


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
    for kind in ("trend", "breakdown"):
        group = [t for t in tiles if t.kind == kind]
        for i in range(0, len(group), 2):           # side by side, two to a row; one alone spans the row
            pair = group[i:i + 2]
            for j, t in enumerate(pair):
                place(t, j * (GRID // 2), GRID if len(pair) == 1 else GRID // 2)
            y += HEIGHT[kind]
    for t in (t for t in tiles if t.kind == "table"):
        place(t, 0, GRID)
        y += HEIGHT["table"]


def status(draft: Draft) -> dict[str, Any]:
    """How far the dashboard is, as the chat says it while the tiles are built: never a tile's answer."""
    return {"name": draft.name, "count": len(draft.tiles), "tabs": list(draft.tabs or ["Overview"]),
            "built": sum(1 for t in draft.tiles if t.payload is not None and not t.error),
            "failed": sum(1 for t in draft.tiles if t.error),
            "pending": sum(1 for t in draft.tiles if t.payload is None and not t.error)}


def not_built(draft: Draft) -> list[dict[str, Any]]:
    """The tiles that could not be built, and why: the chat names them, and offers to define what nothing measures."""
    return [{"title": t.title, "question": t.question, "error": t.error, "definable": t.definable}
            for t in draft.tiles if t.error]


def kept(tile: Tile) -> dict[str, Any]:
    """A built tile as it is saved: its plan (the tile is drawn again from it, with no AI), its place and tab, and
    how its answer was drawn. Never its rows."""
    payload = tile.payload or {}
    chart = payload.get("chart") or {}
    kpi = payload.get("kpi") or {}
    chart_type = "kpi" if kpi and tile.kind == "number" else str(chart.get("chart_type") or ("kpi" if kpi else "table"))
    return {"title": tile.title, "kind": tile.kind, "question": tile.question, "plan": tile.plan,
            "layout": dict(tile.layout), "tab": tile.tab, "chart_type": chart_type,
            "sql": str((payload.get("trust") or {}).get("sql") or ""),
            "display_config": {"core2_plan": tile.plan, "column_formats": chart.get("column_formats") or {},
                               "title": tile.title}}
