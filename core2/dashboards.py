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
area). The whole business is built by the recipe of its kind (core2/recipes.py), with no AI: the same shape and
a proper name every time ("Sales executive dashboard"), each tile's plan written in code. A design too thin for
a subject (one tile for "a revenue dashboard") is asked for again, once. A dashboard of several areas, or of more
tiles than one page holds, is on named tabs: "Overview" first, then a tab per area ("Revenue", "Fills").

Once built, each headline number is checked against the trend of the same figure over the same period, where
there is one: a total that adds up is the sum of its months, a level (a headcount, a balance) its last month's.
The chat says the totals were checked, or which did not match.

Layout, on each tab: the numbers in one row (at most six), trends and breakdowns two side by side (one alone
across the width), tables across the width; at most 12 tiles on a tab, at most 16 on a dashboard.
"""

from __future__ import annotations

import datetime as dt
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

RULES = """You design a dashboard for Solviz from a reader's request. You never write SQL.
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
- name: what the dashboard shows, in two to five words ("Internet sales 2013", "Customers and products"), never
  "Custom dashboard", "My dashboard" or "Dashboard" alone.
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
- A grouping is a field the catalog lists ("by country name", "by state name"), never a table "named by its key"
  on its own ("by geography"): its key is no name a reader can read.
- A tile for an analysis keeps the analysis's own words in its question, never reworded as a plain breakdown:
  "ABC classes of products by net amount" (never "net amount by product class"), "Customers making 80% of net
  amount", "Customers by decile of net amount", "Funnel of orders from order date to delivered date", "Products
  bought together on orders", "Net amount against target amount by month", "New and returning customers by
  month", "Cohort retention of customers by first month".
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
    planned: dict | None = None     # a recipe's tile: the plan written for it, run as it is (no AI)


@dataclass
class Draft:
    name: str
    request: str
    tiles: list[Tile]
    tabs: list[str] = field(default_factory=list)        # the tabs' names, in order; a tile's layout tab is a place here
    checks: list[dict[str, Any]] = field(default_factory=list)   # each headline number against its trend
    others: list[str] = field(default_factory=list)      # other recipes the request named that the data gives


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
    for tile in tiles:
        tile.question = _in_words(model, tile.question)
    return str(data.get("name") or "Dashboard")[:80], tiles


# A name as the catalog spells it: "sales_quota.quarter", "reseller_sale_order_date".
_SLUG = re.compile(r"\b[a-z][a-z0-9]*(?:[._][a-z0-9]+)+\b")


def _in_words(model, question: str) -> str:
    """A field or date the design named as the catalog spells it, in a reader's words ("by sales_quota.quarter" is
    "by quarter"): asked by its catalog name, it is that one table's column ("the quota table's quarter"), where the
    design meant the period of every figure on the tile. A measure keeps its name: it says whose figure it is."""
    from core2.resolve.resolver import find_slug

    def word(found: re.Match) -> str:
        hit = find_slug(model, found.group(0))
        if hit is None or hit[0] == "measure":
            return found.group(0)
        thing = hit[1]
        name = getattr(thing, "business_name", None) or getattr(thing, "name", None)
        return str(name).lower() if name else found.group(0)

    return _SLUG.sub(word, question)


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


# The words that ask for an analysis rather than a plain figure: a tile whose title has one its question lost
# ("ABC classes of products" asked as "sales by product class") is asked in its title's words.
_ANALYSIS = re.compile(r"\b(?:abc|pareto|deciles?|quartiles?|quintiles?|\d+\s*%\s+of|funnel|bought together|"
                       r"against|attainment|cohorts?|retention|returning|lapsed|won back|churn|on time|overdue|"
                       r"ageing|aging|headcount)\b", re.IGNORECASE)


def _asked(title: str, question: str) -> str:
    lost = {w.casefold() for w in _ANALYSIS.findall(title)} - {w.casefold() for w in _ANALYSIS.findall(question)}
    return title if lost else question


def _tiles(items: Any, most: int) -> list[Tile]:
    out = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or not str(item.get("question") or "").strip():
            continue
        kind = str(item.get("kind") or "breakdown")
        tab = " ".join(str(item.get("tab") or "").split())[:30]
        title = str(item.get("title") or item["question"]).strip()[:80]
        out.append(Tile(title=title, kind=kind if kind in HEIGHT else "breakdown",
                        question=_asked(title, str(item["question"]).strip()), tab=tab))
    return out[:most]


class DesignError(ValueError):
    pass


def language(request: str) -> str:
    """The language a request for a dashboard is written in: "fr" for "Crée-moi un tableau de bord ...", else "en"."""
    return "fr" if _DEMANDE.match(_POLITE.sub("", request or "")) else "en"


def executive(model, request: str, *, today: dt.date, allowed: set[str] | None = None) -> tuple[str, list[Tile]] | None:
    """The whole business's dashboard by the recipe of its kind, when the request asks for the whole business
    ("an executive dashboard", "a CFO dashboard for 2025", a dashboard of nothing named)."""
    from core2 import recipes

    about = asked_for(request)
    if about is None:
        return None
    whole, year = recipes.whole_business(about)
    if not whole:
        return None
    made = recipes.executive(model, today=today, year=year, language=language(request), allowed=allowed)
    if made is None:
        return None
    name, tiles = made
    return name, [Tile(title=t.title, kind=t.kind, question=t.question, tab=t.tab, planned=t.plan) for t in tiles]


def from_book(model, request: str, *, today: dt.date, allowed: set[str] | None = None
              ) -> tuple[str, list[Tile], list[str]] | None:
    """The dashboard of the catalogue's recipe the request names (core2/recipe_book.py: "a dashboard for the sales
    team", "customer retention dashboard", "a CFO dashboard"), when the data gives it, at least in part: its
    name, its tiles, and the other recipes the request also names that the data gives (the reader may want one of
    them instead). None when it names none the data gives: the whole business's recipe, or the AI's design."""
    from core2 import recipe_book

    about = asked_for(request)
    if about is None:
        return None
    lang = language(request)
    built = [b for b in (recipe_book.build(model, key, today=today, lang=lang, allowed=allowed)
                         for key in recipe_book.find(about, lists=_lists(about))) if b.state != "no"]
    if not built:
        return None
    first = built[0]
    tiles = [Tile(title=t.title, kind=t.kind, question=t.question, tab=t.tab, planned=t.plan) for t in first.tiles]
    return first.name, tiles, [b.book.key for b in built[1:3]]


def build(model, request: str, complete: Callable[[str, str], str], answer: Callable[..., dict], *,
          scrub: Callable[[str], str] | None = None, personal_shown: bool = False,
          progress: Callable[[Draft], None] | None = None, today: dt.date | None = None,
          allowed: set[str] | None = None) -> Draft:
    """The dashboard: the recipe of the business's kind, or designed by one AI call; each tile answered on its
    own (several at once), laid out, and its headline numbers checked against their trends.

    ``answer(question)`` answers a question as a new conversation would (its own planner call, its own query);
    ``answer(question, plan)`` runs a recipe tile's plan as it is. A tile that cannot be built says why; the
    others are built. ``progress(draft)`` is told the draft once it is designed and again as each tile lands,
    from the thread that built it, one call at a time."""
    day = today or dt.date.today()
    book = from_book(model, request, today=day, allowed=allowed)
    recipe = executive(model, request, today=day, allowed=allowed) if book is None else None
    if book is not None:
        name, tiles, others = book
    else:
        others = []
        name, tiles = recipe if recipe is not None else design(model, request, complete, scrub=scrub,
                                                               personal_shown=personal_shown)
    draft = Draft(name=name, request=request, tiles=tiles, others=others)
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
    draft.checks = checked(draft.tiles, model)
    return draft


def answered(tile: Tile, answer: Callable[..., dict]) -> Tile:
    """The tile with its answer, or why it could not be built: when the data has nothing that measures it (the
    planner found none, or asked what was meant), the reader is offered to define it. A recipe's tile is its
    plan, run as it is: one that fails is left out, never offered to be defined."""
    try:
        payload = answer(tile.question) if tile.planned is None else answer(tile.question, tile.planned)
    except Exception as exc:  # noqa: BLE001 - one tile failing never takes the dashboard with it
        log.warning("core2 could not build the tile %r: %s", tile.title, exc)
        tile.error = "This tile could not be built."
        return tile
    if payload.get("data") is None or not payload.get("plan"):
        tile.error = str((payload.get("answer") or {}).get("headline") or payload.get("text")
                         or "This tile could not be built.")
        tile.definable = tile.planned is None and bool(payload.get("unsupported") or payload.get("clarify"))
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


_SHAPES = ("per", "rolling", "within", "bands", "absent", "gap", "lifecycle", "cohort", "active", "on_time", "forecast",
           "drivers", "derived", "durations")


def _figure(plan: dict[str, Any]) -> tuple | None:
    """What a plan totals, as compared: its one measure, its filters, its date and period. None for a shaped
    answer (a share, a running total, a band), whose total is no sum of a trend."""
    measures = plan.get("measures") or []
    if len(measures) != 1 or plan.get("group_by") or any(plan.get(k) for k in _SHAPES):
        return None
    time = plan.get("time") or {}
    return (measures[0], json.dumps(plan.get("filters") or [], sort_keys=True), time.get("date"),
            json.dumps(time.get("window") or {"kind": "all"}, sort_keys=True), bool(plan.get("include_left_out")))


def checked(tiles: list[Tile], model) -> list[dict[str, Any]]:
    """Each headline number against the trend of the same figure over the same period: a total that adds up is
    the sum of its periods, a level (a headcount, a balance kept each month) its last period's, or their average
    where it is kept that way. A count of distinct things (orders over a year, fewer than the sum of each month's
    when one spans two) is not checked."""
    by_slug = {m.slug: m for m in model.measures.values()}
    out = []
    for n in tiles:
        if n.kind != "number" or n.error or not n.plan or n.payload is None:
            continue
        figure = _figure(n.plan)
        m = by_slug.get(figure[0]) if figure else None
        if m is None or m.additivity == "non_additive":
            continue
        trend = next((t for t in tiles if t.kind == "trend" and not t.error and t.plan and t.payload is not None
                      and (t.plan.get("time") or {}).get("grain") and _figure(t.plan) == figure), None)
        if trend is None:
            continue
        number = ((n.payload.get("data") or {}).get("rows") or [{}])[0].get(m.slug)
        data = trend.payload.get("data") or {}
        points = [r.get(m.slug) for r in data.get("rows") or []]
        if not isinstance(number, (int, float)) or not points or data.get("truncated"):
            continue
        values = [float(v) for v in points if isinstance(v, (int, float))]
        if m.additivity == "additive":
            how, total = "sum", sum(values)
        elif m.time_aggregation == "average":
            how, total = "average", sum(values) / len(values) if values else 0.0
        elif m.time_aggregation == "first":
            how, total = "first", values[0] if values else 0.0
        else:
            how, total = "last", values[-1] if values else 0.0
        agree = abs(float(number) - total) <= max(0.01 * len(values), abs(float(number)) * 1e-6)
        if not agree:
            log.warning("core2 dashboard: %r (%s) is not the %s of its trend %r (%s)", n.title, number, how,
                        trend.title, total)
        out.append({"number": n.title, "trend": trend.title, "tab": n.tab, "how": how, "agree": agree,
                    "total": float(number), "of_trend": round(total, 6)})
    return out


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
