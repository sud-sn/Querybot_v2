"""Changes to a dashboard asked for in the chat: "remove the Shipping tab", "make the trend quarterly", "rename it
Board pack", "undo".

Read in code, against the dashboard the conversation is on (the one the chat was opened from, or the one this
thread made or changed last), never by the AI. A change names the dashboard, a tab, a tile, or what the
dashboard has ("the trend", a tile by its title); words that name nothing of it are a question, answered as one.
Right after a dashboard was made or changed, "make the trend quarterly" or "undo" is about it; any other time
the words say so ("the Shipping tab", "on the dashboard"). Words that only might be about it ("show net amount by
month", "show that as a pie") are a change only when they name what it has (a tile's title, "the trend"):
otherwise they are a question, as they would be anywhere.

What can be changed:

* undo: the version before the last change comes back (a tile taken off returns).
* the dashboard's name ("rename the dashboard to Board pack", "call it Board pack").
* a tab: taken off with its tiles ("remove the Shipping tab"), or renamed ("rename the Orders tab to Sales").
* a tile: taken off ("remove Top 10 customers"), moved ("move Net amount by region to the Orders tab"), drawn
  another way ("make Net amount by category a pie"), or counted by another period ("make the trend quarterly",
  "show the trends by week").
* every tile's period ("show the dashboard for 2025", "set the period to this year").
* a tile added ("add net amount by region to the Orders tab"): asked as a new question, its answer added.
* the answer on screen added ("add this chart to my dashboard", "pin this", "save that to Board pack", "put it on
  the Sales tab", "add this to a new dashboard"): read by ``this_answer``, added by the server.

A dashboard of the reader's may be named in the words ("add net amount by region to the Board pack dashboard",
"rename the Board pack dashboard to Q3 pack"): ``naming`` finds it, and the change is read against it.

Each change is one version of the dashboard, saying what it was ("Removed the Shipping tab"); its history lists
them, and undo puts the one before back. Words that point at several tiles ("the trend", with three) ask which,
one button each; a tab or tile the dashboard does not have is said, with what it has.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from core2.bootstrap import names


@dataclass
class Tile:
    id: int
    title: str
    tab: str
    chart_type: str
    plan: dict[str, Any] | None = None       # the new core's plan it is drawn from (None: today's pinned SQL)

    @property
    def kind(self) -> str:
        plan = self.plan or {}
        if (plan.get("time") or {}).get("grain") and not plan.get("group_by"):
            return "trend"
        if plan.get("group_by"):
            return "table" if plan.get("limit") or self.chart_type == "table" else "breakdown"
        return "number" if plan else ("number" if self.chart_type == "kpi" else "chart")


@dataclass
class Board:
    id: int
    name: str
    tabs: list[str]
    tiles: list[Tile]


@dataclass
class Change:
    """One change, as the store makes it (store.change_dashboard), and what it is in words."""
    said: str                                   # "Removed the Shipping tab"
    kind: str = "change"                        # change | undo | add
    name: str | None = None
    tabs: list[str] | None = None
    remove: list[int] = field(default_factory=list)
    tiles: dict[int, dict[str, Any]] = field(default_factory=dict)
    question: str = ""                          # add: the question whose answer is added
    tab: str = ""                               # add: the tab it goes on


@dataclass
class Which:
    """The words point at several tiles: which one, a button each (each a request of its own)."""
    question: str
    options: list[str]


@dataclass
class ThisAnswer:
    """The answer on screen, to be added to a dashboard: a new one (``new``, named ``board`` or after the answer),
    the one named (``board``), or, with neither, the one the conversation is on, else the reader's only one, else
    which (a button each). ``tab``: the tab named, kept or made."""
    new: bool = False
    board: str = ""
    tab: str = ""
    words: str = ""                              # what followed "to": a dashboard's name, or a tab's


@dataclass
class Refused:
    """A change of something the dashboard does not have, or cannot be made: said, with what it has."""
    text: str


Reading = Change | Which | Refused

# Words that say the request is about a dashboard, at any time. Not "card": a credit card is data.
_ABOUT = re.compile(r"\b(?:dashboard|tab|tabs|tile|tiles|tableau de bord|onglet|onglets|tuile|tuiles)\b",
                    re.IGNORECASE)
_POLITE = re.compile(r"^\s*(?:(?:please|ok(?:ay)?|now|and|then|also|can you|could you|would you|pls)[\s,]+)+",
                     re.IGNORECASE)
_END = re.compile(r"[\s.!?,]*(?:\bplease\b|\bthanks?\b|\bthank you\b)?[\s.!?]*$", re.IGNORECASE)

_UNDO = re.compile(r"^(?:undo|revert|roll ?back|go back|put (?:it|that) back|annule[rz]?|défais)"
                   r"(?:\s+(?:that|it|this|the last change|the last one|the change|my last change|cela|ça|"
                   r"la derni[eè]re modification))?"
                   r"(?:\s+(?:on|to|of)\s+(?:the|this|my)\s+dashboard|\s+(?:du|sur le)\s+tableau de bord)?$",
                   re.IGNORECASE)
_RENAME_BOARD = re.compile(r"^(?:rename|call|name|retitle)\s+(?:the\s+|this\s+|my\s+)?(?:dashboard|it)\s+"
                           r"(?:to\s+|as\s+)?[\"“']?(?P<new>.+?)[\"”']?$", re.IGNORECASE)
_RENAME_TAB = re.compile(r"^(?:rename|call|name)\s+(?:the\s+)?[\"“']?(?P<old>.+?)[\"”']?\s+tab\s+"
                         r"(?:to\s+|as\s+)?[\"“']?(?P<new>.+?)[\"”']?$", re.IGNORECASE)
_REMOVE = r"(?:remove|delete|drop|hide|get rid of|take out|take off|close|supprime[rz]?|enl[eè]ve[rz]?|retire[rz]?)"
_REMOVE_TAB = re.compile(rf"^{_REMOVE}\s+(?:the\s+|l'|l’|le\s+)?(?:tab\s+|onglet\s+)?[\"“']?(?P<name>.+?)[\"”']?"
                         r"(?:\s+(?:tab|onglet))(?:\s+from\s+(?:the|this|my)\s+dashboard)?$", re.IGNORECASE)
_REMOVE_TAB_NAMED = re.compile(rf"^{_REMOVE}\s+(?:the\s+)?(?:tab|onglet)\s+(?:called\s+|named\s+)?"
                               r"[\"“']?(?P<name>.+?)[\"”']?$", re.IGNORECASE)
_REMOVE_TILE = re.compile(rf"^{_REMOVE}\s+(?:the\s+|l'|l’|la\s+|le\s+)?[\"“']?(?P<name>.+?)[\"”']?"
                          r"(?:\s+(?:tile|tiles|chart|charts|card|cards|graph|visual|tuile))?"
                          r"(?:\s+(?:from|on)\s+(?:the\s+)?(?P<tab>.+?)\s+tab)?"
                          r"(?:\s+from\s+(?:the|this|my)\s+dashboard)?$", re.IGNORECASE)
_MOVE = re.compile(r"^(?:move|put|place)\s+(?:the\s+)?[\"“']?(?P<name>.+?)[\"”']?(?:\s+(?:tile|chart|card))?\s+"
                   r"(?:to|on|onto|into)\s+(?:the\s+)?[\"“']?(?P<tab>.+?)[\"”']?\s+tab$", re.IGNORECASE)
_CHART = re.compile(r"^(?P<verb>make|show|turn|change|switch|draw|display)\s+(?:the\s+)?[\"“']?(?P<name>.+?)[\"”']?"
                    r"(?:\s+(?:tile|chart|card))?\s+(?:into|to|as|in)?\s*(?:a\s+|an\s+)?"
                    r"(?P<type>bar|bars|column|line|area|pie|donut|doughnut|table)(?:\s+chart|\s+graph)?$",
                    re.IGNORECASE)
_GRAIN_WORDS = {"daily": "day", "day": "day", "days": "day", "weekly": "week", "week": "week", "weeks": "week",
                "monthly": "month", "month": "month", "months": "month", "quarterly": "quarter",
                "quarter": "quarter", "quarters": "quarter", "yearly": "year", "annual": "year",
                "annually": "year", "year": "year", "years": "year"}
_GRAIN = re.compile(r"^(?P<verb>make|show|switch|change|turn|put|set|display|have)\s+(?:the\s+)?[\"“']?(?P<name>.+?)"
                    r"[\"”']?\s+(?:to\s+|into\s+|as\s+|in\s+)?(?:by\s+|per\s+|each\s+)?"
                    r"(?P<grain>daily|weekly|monthly|quarterly|yearly|annual|annually|day|week|month|quarter|year)"
                    r"(?:\s+(?:periods?|points?|figures?|totals?))?$", re.IGNORECASE)
_PERIOD = re.compile(r"^(?:show|make|change|set|switch|put|count|have)\s+(?:the\s+)?"
                     r"(?:whole\s+|entire\s+)?(?:dashboard|period|dates?|everything|every tile|all(?: the)? tiles|it)"
                     r"(?:'s period)?\s+(?:to\s+|for\s+|as\s+|in\s+|over\s+|on\s+)+(?P<period>.+)$", re.IGNORECASE)
_ADD = re.compile(r"^(?:add|put|include)\s+(?:a\s+tile\s+(?:of|for|with|showing)\s+|a\s+chart\s+(?:of|for|showing)\s+)?"
                  r"(?P<what>.+?)(?:\s+(?:to|on|onto|in)\s+(?:the\s+)?[\"“']?(?P<tab>[^\"”']+?)[\"”']?\s+tab)?"
                  r"(?:\s+(?:to|on|onto)\s+(?:the|this|my)\s+dashboard)?$", re.IGNORECASE)
# "Add a chart of returns by courier": a tile, said so, right after a dashboard was made or changed.
_A_TILE = re.compile(r"^add\s+(?:a|an)\s+(?:tile|chart|graph|kpi|number|table|card)\b", re.IGNORECASE)
_NEW_TAB = re.compile(r"^(?:add|create|make|open)\s+(?:a\s+|an\s+)?(?:new\s+)?(?:tab|onglet)\s+(?:called\s+|named\s+|for\s+)?"
                      r"[\"“']?(?P<name>.+?)[\"”']?(?:\s+(?:to|on)\s+(?:the|this|my)\s+dashboard)?$", re.IGNORECASE)
_TYPES = {"bars": "bar", "column": "bar", "doughnut": "donut"}
_KIND_WORDS = {"trend": "trend", "trends": "trend", "line": "trend", "lines": "trend", "number": "number",
               "numbers": "number", "kpi": "number", "kpis": "number", "figure": "number", "figures": "number",
               "breakdown": "breakdown", "breakdowns": "breakdown", "bar": "breakdown", "bars": "breakdown",
               "table": "table", "tables": "table", "ranking": "table", "top 10": "table"}
_ALL = re.compile(r"^(?:all(?:\s+of)?\s+(?:the\s+)?|every\s+|each\s+)", re.IGNORECASE)
_YEAR = re.compile(r"^(?:(?:the\s+)?year\s+)?((?:19|20)\d\d)$")
_UNITS = {"day": "day", "days": "day", "week": "week", "weeks": "week", "month": "month", "months": "month",
          "quarter": "quarter", "quarters": "quarter", "year": "year", "years": "year"}


# "Add this to my dashboard", "pin this", "save that chart to Board pack", "put it on the Sales tab", "ajoute ça au
# tableau de bord". Never "include those" or "keep these": those are about the rows of the answer.
_THIS_WORDS = (r"(?:this|that|it|the (?:chart|answer|graph|table|number|figure|result) above|ceci|cela|ça|ca|"
               r"ce graphique|cette réponse|ce résultat)")
_PIN_THIS = re.compile(
    r"^(?P<verb>add|pin|save|put|place|stick|ajoute[rz]?|épingle[rz]?|epingle[rz]?|mets|mettez|enregistre[rz]?)\s+"
    rf"{_THIS_WORDS}(?:\s+(?:chart|graph|answer|table|number|figure|kpi|tile|result|one|graphique|réponse))?"
    r"(?:\s+(?:to|on|onto|in|into|à|au|sur|dans)\s+(?P<where>.+?))?$", re.IGNORECASE)
_ALONE = {"add", "pin", "save", "ajoute", "ajouter", "ajoutez", "épingle", "épingler", "épinglez", "epingle", "epingler",
          "epinglez", "enregistre", "enregistrer", "enregistrez"}
_NEW_BOARD = re.compile(r"^(?:(?:a|an)\s+)?new\s+(?:dashboard|one)(?:\s+(?:called|named)\s+(?P<name>.+))?$|"
                        r"^un\s+nouveau\s+tableau de bord(?:\s+(?:appelé|nommé)\s+(?P<nom>.+))?$", re.IGNORECASE)
_THE_BOARD = re.compile(r"^(?:(?:the|my|this|our|a)\s+)?dashboard$|^(?:le|mon|ce)\s+tableau de bord$|^tableau de bord$",
                        re.IGNORECASE)
_ON_TAB = re.compile(r"^(?:the\s+|my\s+)?[\"“']?(?P<tab>.+?)[\"”']?\s+tab(?:\s+(?:of|on|in)\s+(?P<board>.+))?$|"
                     r"^(?:l'|l’)?onglet\s+[\"“«]?(?P<onglet>.+?)[\"”»]?(?:\s+(?:du|de)\s+(?P<tableau>.+))?$",
                     re.IGNORECASE)
_NAMED_BOARD = re.compile(r"^(?:(?:the|my)\s+)?[\"“']?(?P<name>.+?)[\"”']?\s+dashboard$|"
                          r"^(?:le\s+)?tableau de bord\s+[\"“«]?(?P<nom>.+?)[\"”»]?$", re.IGNORECASE)


def this_answer(text: str) -> ThisAnswer | None:
    """The words add the answer on screen to a dashboard ("add this chart to my dashboard"): where; None when they
    do not."""
    words = _END.sub("", _POLITE.sub("", (text or "").strip()))
    found = _PIN_THIS.match(words)
    if found is None:
        return None
    where = _clean(found["where"] or "")
    if not where:
        return ThisAnswer() if found["verb"].casefold() in _ALONE else None
    new = _NEW_BOARD.match(where)
    if new:
        return ThisAnswer(new=True, board=_clean(new["name"] or new["nom"] or ""))
    if _THE_BOARD.match(where):
        return ThisAnswer()
    tab = _ON_TAB.match(where)
    if tab:
        board = _clean(tab["board"] or tab["tableau"] or "")
        board = _clean(re.sub(r"^(?:the|my|le|mon)\s+|\s+(?:dashboard|tableau de bord)$", "", board, flags=re.IGNORECASE))
        return ThisAnswer(board=board, tab=_clean(tab["tab"] or tab["onglet"] or ""), words=where)
    named = _NAMED_BOARD.match(where)
    if named:
        return ThisAnswer(board=_clean(named["name"] or named["nom"] or ""), words=where)
    return ThisAnswer(board=_clean(re.sub(r"^(?:the|my)\s+", "", where, flags=re.IGNORECASE)), words=where)


def board_named(name: str, boards: list[str]) -> str | None:
    """The reader's dashboard a name means: its own name (any case), else the one name that holds every word."""
    wanted = _plain(name)
    if not wanted:
        return None
    exact = [b for b in boards if _plain(b) == wanted]
    if exact:
        return exact[0]
    inside = [b for b in boards if set(wanted.split()) <= set(_plain(b).split())]
    return inside[0] if len(inside) == 1 else None


def naming(text: str, boards: list[str]) -> tuple[str, str] | None:
    """A dashboard of the reader's the words name ("to the Board pack dashboard", "rename the "Board pack"
    dashboard to ..."), and the words with it said as "the dashboard"; None when they name none."""
    for name in sorted(boards, key=len, reverse=True):
        if not name.strip():
            continue
        quoted = re.escape(name)
        for pattern in (rf"(?:\b(?:the|my)\s+)?[\"“']?{quoted}[\"”']?\s+dashboard\b",
                        rf"(?:\b(?:the|my)\s+)?[\"“']{quoted}[\"”']",
                        *([rf"(?:\b(?:the|my)\s+)?\b{quoted}\b"] if name.casefold().endswith("dashboard") else [])):
            found = re.search(pattern, text, re.IGNORECASE)
            if found:
                return name, text[:found.start()] + "the dashboard" + text[found.end():]
    return None


def about_a_dashboard(text: str) -> bool:
    """Do the words say they are about a dashboard (a tab, a tile, the dashboard)?"""
    return bool(_ABOUT.search(text or ""))


def read(text: str, board: Board, *, on_it: bool) -> Reading | None:
    """The change the words ask of ``board``, or None when they ask none (a question). ``on_it``: the
    conversation is on this dashboard (it was just made or changed), so words need not name it."""
    words = _END.sub("", _POLITE.sub("", (text or "").strip()))
    explicit = about_a_dashboard(words)
    if not words or not (on_it or explicit):
        return None

    def missing(refusal: Refused) -> Refused | None:
        """A tab or tile the dashboard lacks: said when the words were about a dashboard, else a question."""
        return refusal if explicit else None
    if _UNDO.match(words):
        return Change("Undid the last change", kind="undo")
    found = _RENAME_TAB.match(words)
    if found:
        tab = _tab(found["old"], board)
        if tab is None:
            return missing(_no_tab(found["old"], board))
        new = _clean(found["new"])
        return Change(f"Renamed the {tab} tab to {new}", tabs=[new if t == tab else t for t in board.tabs],
                      tiles={t.id: {"dashboard_tab": new, "in_place": True} for t in board.tiles if t.tab == tab})
    found = _RENAME_BOARD.match(words)
    if found:
        new = _clean(found["new"])
        return Change(f"Renamed the dashboard to {new}", name=new)
    for pattern in (_REMOVE_TAB, _REMOVE_TAB_NAMED):
        found = pattern.match(words)
        if found:
            tab = _tab(found["name"], board)
            if tab is None:
                return missing(_no_tab(found["name"], board))
            return _remove_tab(tab, board)
    found = _MOVE.match(words)
    if found:
        tab = _tab(found["tab"], board)
        if tab is None:
            return missing(_no_tab(found["tab"], board))
        picked = _tiles(found["name"], board)
        if not isinstance(picked, list):
            return picked if picked is not None else missing(_no_tile(found["name"], board))
        moved = [t for t in picked if t.tab != tab]
        return Change(f"Moved {_titles(picked)} to the {tab} tab", tiles={t.id: {"dashboard_tab": tab} for t in moved})
    found = _PERIOD.match(words)
    if found:
        window = period(found["period"])
        if window is None:
            return Refused(f"I could not read “{found['period']}” as a period: say a year (2025), “this "
                           "year”, “last year” or “the last 12 months”.")
        planned = [t for t in board.tiles if t.plan]
        if not planned:
            return Refused("None of this dashboard's tiles is drawn by the new core, so its period cannot be "
                           "changed here: use the period filter on the dashboard.")
        return Change(f"Set every tile to {_period_words(window)}", tiles={
            t.id: {"display_config": {"core2_plan": _with(t.plan, ("time", "window"), window)}} for t in planned})
    found = _CHART.match(words)
    if found and not _GRAIN.match(words) and not _asks(found, explicit):
        picked = _tiles(found["name"], board, one=True, ask=lambda t: f"Make {t} a {found['type'].lower()}")
        if not isinstance(picked, list):
            return picked if picked is not None else missing(_no_tile(found["name"], board))
        kind = _TYPES.get(found["type"].lower(), found["type"].lower())
        return Change(f"Drew {_titles(picked)} as a {kind if kind != 'table' else 'table'}",
                      tiles={t.id: {"chart_type": kind} for t in picked})
    found = _GRAIN.match(words)
    if found and not _asks(found, explicit):
        grain = _GRAIN_WORDS[found["grain"].lower()]
        picked = _tiles(found["name"], board, ask=lambda t: f"Make {t} {_ADVERB[grain]}",
                        only=lambda t: t.kind == "trend")
        if not isinstance(picked, list):
            return picked if picked is not None else missing(_no_tile(found["name"], board, trend=True))
        changed = {}
        for t in picked:
            plan = _with(t.plan, ("time", "grain"), grain)
            changed[t.id] = {"display_config": {"core2_plan": plan}, "title": _regrained(t.title, grain)}
        return Change(f"Made {_titles(picked)} {_ADVERB[grain]}", tiles=changed)
    for pattern in (_REMOVE_TILE,):
        found = pattern.match(words)
        if found:
            if found["tab"] is None and _tab(found["name"], board) is not None and not _tiles(found["name"], board):
                return _remove_tab(_tab(found["name"], board) or "", board)
            scope = _tab(found["tab"], board) if found["tab"] else None
            picked = _tiles(found["name"], board, within=scope, ask=lambda t: f"Remove {t}")
            if not isinstance(picked, list):
                return picked if picked is not None else missing(_no_tile(found["name"], board))
            return Change(f"Removed {_titles(picked)}", remove=[t.id for t in picked])
    found = _NEW_TAB.match(words)
    if found:
        new = _clean(found["name"])
        if _tab(new, board) is not None:
            return Refused(f"{board.name} already has a {_tab(new, board)} tab.")
        return Change(f"Added a {new} tab", tabs=[*board.tabs, new])
    found = _ADD.match(words)
    if found and found["what"] and (explicit or _A_TILE.match(words)):
        tab = _tab(found["tab"], board) if found["tab"] else None
        new_tab = _clean(found["tab"]) if found["tab"] and tab is None else ""
        what = _clean(found["what"])
        return Change(f"Added {what}", kind="add", question=what, tab=tab or new_tab or (board.tabs or ["Overview"])[0])
    return None


def _asks(found: re.Match, explicit: bool) -> bool:
    """"Show net amount by month" or "show that as a pie" asks a question; "show the trends by week" changes the
    dashboard, as "make Net amount by month quarterly" does."""
    if explicit or found["verb"].lower() not in ("show", "display"):
        return False
    name = _ALL.sub("", re.sub(r"^(?:the|my|this)\s+", "", found["name"].strip(), flags=re.IGNORECASE))
    return name.casefold() not in _KIND_WORDS


_ADVERB = {"day": "daily", "week": "weekly", "month": "monthly", "quarter": "quarterly", "year": "yearly"}


def _clean(text: str) -> str:
    return " ".join(str(text or "").strip(" \"“”'.").split())[:80]


def _plain(text: str) -> str:
    return " ".join(names.singular(w) for w in re.findall(r"[\w%]+", (text or "").casefold()))


def _tab(name: str | None, board: Board) -> str | None:
    wanted = _plain(re.sub(r"^(?:the|my|this)\s+", "", name or "", flags=re.IGNORECASE))
    return next((t for t in board.tabs if _plain(t) == wanted), None) if wanted else None


def _titles(tiles: list[Tile]) -> str:
    shown = [f"“{t.title}”" for t in tiles]
    return shown[0] if len(shown) == 1 else ", ".join(shown[:-1]) + f" and {shown[-1]}"


def _tiles(name: str, board: Board, *, within: str | None = None, one: bool = False, ask=None,
           only=None) -> list[Tile] | Which | None:
    """The tiles the words name: a title (as written, or every word of it), or a kind of tile ("the trend",
    "all the trends", "the numbers"), on a tab when one is named. Several, where the words say one: which."""
    pool = [t for t in board.tiles if (within is None or t.tab == within) and (only is None or only(t))]
    words = (name or "").strip()
    every = bool(_ALL.match(words))
    words = _ALL.sub("", words)
    scope = re.match(r"^(?P<what>.+?)\s+(?:on|in)\s+(?:the\s+)?(?P<tab>.+?)\s+tab$", words, re.IGNORECASE)
    if scope and _tab(scope["tab"], board):
        pool = [t for t in pool if t.tab == _tab(scope["tab"], board)]
        words = scope["what"]
    words = re.sub(r"^(?:the|my|this|that)\s+", "", words, flags=re.IGNORECASE).strip()
    words = re.sub(r"^(?:tile|chart|card|graph|visual)\s+(?:called\s+|named\s+)?", "", words, flags=re.IGNORECASE)
    plain = _plain(words.strip(" \"“”'"))
    exact = [t for t in pool if _plain(t.title) == plain]
    if exact:
        return exact
    kind = _KIND_WORDS.get(words.casefold())
    if kind is not None:
        found = [t for t in pool if t.kind == kind]
        many = every or words.casefold().endswith("s") and not words.casefold().endswith("ss")
    else:
        wanted = set(plain.split())
        found = [t for t in pool if wanted and wanted <= set(_plain(t.title).split())]
        many = every
    if not found:
        return None
    if len(found) > 1 and not many:
        if ask is None:
            return found
        return Which(f"Which {'tile' if kind is None else kind} do you mean?", [ask(t.title) for t in found[:6]])
    return found


def _remove_tab(tab: str, board: Board) -> Reading:
    if len(board.tabs) <= 1:
        return Refused(f"The {tab} tab is this dashboard's only tab: remove its tiles instead, or the dashboard.")
    on = [t.id for t in board.tiles if t.tab == tab]
    return Change(f"Removed the {tab} tab" + (f" and its {len(on)} tile{'s' if len(on) != 1 else ''}" if on else ""),
                  tabs=[t for t in board.tabs if t != tab], remove=on)


def _listed(items: list[str]) -> str:
    if not items:
        return "none"
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" and {items[-1]}"


def _no_tab(name: str, board: Board) -> Refused:
    if not board.tabs:
        return Refused(f"{board.name or 'This dashboard'} has no tabs.")
    return Refused(f"{board.name} has no tab called “{_clean(name)}”. Its tabs are {_listed(board.tabs)}.")


def _no_tile(name: str, board: Board, *, trend: bool = False) -> Refused:
    if trend and not any(t.kind == "trend" for t in board.tiles):
        return Refused(f"{board.name} has no trend drawn by the new core to count another way.")
    titles = [t.title for t in board.tiles][:8]
    return Refused(f"{board.name} has no tile “{_clean(name)}”. Its tiles include {_listed(titles)}."
                   if titles else f"{board.name} has no tiles yet.")


def _with(plan: dict[str, Any] | None, path: tuple[str, str], value: Any) -> dict[str, Any]:
    out = dict(plan or {})
    inner = dict(out.get(path[0]) or {})
    inner[path[1]] = value
    out[path[0]] = inner
    return out


_BY = re.compile(r"\b(by|per|par)\s+(day|week|month|quarter|year|jour|semaine|mois|trimestre|an|année)\b",
                 re.IGNORECASE)
_FR = {"day": "jour", "week": "semaine", "month": "mois", "quarter": "trimestre", "year": "an"}


def _regrained(title: str, grain: str) -> str:
    """"Net amount by month" made quarterly is "Net amount by quarter"; a title of the reader's own stays."""
    def to(found: re.Match) -> str:
        return f"{found[1]} {_FR[grain] if found[1].lower() == 'par' else grain}"

    return _BY.sub(to, title, count=1)


def period(text: str) -> dict[str, Any] | None:
    """A period in the reader's words as a window: a year, this/last year, quarter or month, the last N units,
    year to date, all time."""
    words = " ".join((text or "").casefold().replace("’", "'").split()).strip(" .")
    words = re.sub(r"^the\s+", "", words)
    year = _YEAR.match(words)
    if year:
        return {"kind": "between", "start": f"{year[1]}-01-01", "end": f"{year[1]}-12-31"}
    found = re.match(r"^(this|current|last|previous|past)\s+(day|week|month|quarter|year)$", words)
    if found:
        return {"kind": "this" if found[1] in ("this", "current") else "previous", "unit": found[2]}
    found = re.match(r"^(?:last|past|previous)\s+(\d{1,3})\s+(days|weeks|months|quarters|years)$", words)
    if found:
        return {"kind": "last", "unit": _UNITS[found[2]], "n": int(found[1])}
    if words in ("year to date", "ytd", "this year to date"):
        return {"kind": "to_date", "unit": "year"}
    if words in ("all time", "all dates", "everything", "all the data", "all data"):
        return {"kind": "all"}
    return None


def _period_words(window: dict[str, Any]) -> str:
    kind = window.get("kind")
    if kind == "between":
        return str(window.get("start", ""))[:4]
    if kind == "this":
        return f"this {window.get('unit')}"
    if kind == "previous":
        return f"last {window.get('unit')}"
    if kind == "last":
        return f"the last {window.get('n')} {window.get('unit')}s"
    if kind == "to_date":
        return f"{window.get('unit')} to date"
    if kind == "rolling":
        return f"the last {window.get('n')} {window.get('unit')}s up to today"
    return "all time"
