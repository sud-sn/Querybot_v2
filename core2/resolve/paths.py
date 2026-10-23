"""One rule for join paths, used by every query shape (DESIGN §6.4).

From a measure's table to the table holding an attribute:

1. only steps that never multiply the measure's rows (many-to-one or one-to-one),
   never through a calendar, never back through a table already on the path;
2. the measure table's own key first: a direct link wins over any longer path;
3. then the most trusted weakest link, then the best weakest match rate, then
   fewer steps, then the smallest join keys (deterministic, and reported).

When two different direct links tie (a requester and an approver both pointing
at the people table), the rule does not guess: the question must say which,
or an admin's default decides. Two longer paths that tie on everything but
their keys (a fill's formula through its batch, or through its prescription)
may be different answers: as a governed model's active relationship, the first
is taken, and the answer names the others (``Path.others``, ``distinct``) and
how to ask for one. A route whose links carry a role name is never the
default, so an admin decides by naming the role of the route not wanted.
"""

from __future__ import annotations

import contextvars
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from core2.model.schema import Join, SemanticModel

TRUST_RANK = {"admin": 0, "verified": 1, "declared": 2, "proposed": 3}
MAX_STEPS = 4


@dataclass
class Path:
    joins: list[Join]
    # Routes as good as this one that reach the member another way (best_path): the answer says so.
    others: list["Path"] = field(default_factory=list, compare=False, repr=False)

    @property
    def tables(self) -> list[str]:
        return [self.joins[0].from_table] + [j.to_table for j in self.joins] if self.joins else []

    @property
    def score(self) -> tuple:
        if not self.joins:
            return (0, 0, 0.0, 0, ())
        weakest_trust = max(TRUST_RANK.get(j.trust, 9) for j in self.joins)
        weakest_match = min(j.match_rate for j in self.joins)
        return (0 if len(self.joins) == 1 else 1, weakest_trust, -round(weakest_match, 3), len(self.joins),
                tuple(j.key for j in self.joins))

    def describe(self, model: SemanticModel) -> str:
        if not self.joins:
            return ""
        steps = []
        for j in self.joins:
            target = model.tables[j.to_table].business_name
            steps.append(f"{j.role} ({target})" if j.role else target)
        return " > ".join(steps)


class Ambiguous(Exception):
    """Two different paths are equally good: the question has to say which."""

    def __init__(self, options: list[Path]):
        super().__init__("ambiguous path")
        self.options = options


def usable(j: Join) -> bool:
    """A link a question may follow: not rejected, and not one of several a column could mean.

    A column whose values fit two tables equally (two small dimensions keyed 1..6)
    is a question for an admin, never a guess: until one link is approved, neither is used.
    """
    if j.trust == "rejected":
        return False
    return j.trust == "admin" or j.status == "approved" or not any(e.kind == "ambiguous" for e in j.evidence)


def repeats(j: Join) -> bool:
    """Its target holds more than one row for some of its keys, as measured or as its cardinality says:
    followed from its source, it would count each source row once for every match."""
    return (j.cardinality not in ("many_to_one", "one_to_one") or j.max_fanout > 1
            or (j.target_checked and not j.to_unique))


def unchecked(j: Join) -> bool:
    """No one measured whether its target repeats: one the database declares, or brought over from today's
    setup, in a model learned before Learn measured it. A link Learn found by its values points at a column
    whose every value is unique, and a calendar keeps one row per day."""
    return not j.target_checked and j.provenance != "profile" and not j.to_calendar


# While one model is read many times and not changed (a question map judging thousands of questions), each way
# of reading its links is built once. Held for one thread of work, and only for the model it was read from.
_FIXED: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("core2_links_fixed", default=None)


@contextmanager
def fixed(model: SemanticModel) -> Iterator[None]:
    """Inside, ``model``'s links are taken not to change: its link graph is built once for each way it is read."""
    token = _FIXED.set({"model": model, "edges": {}})
    try:
        yield
    finally:
        _FIXED.reset(token)


def _edges(model: SemanticModel, unconfirmed: bool = False, skip: frozenset[str] = frozenset(),
           repeating: bool = False) -> dict[str, list[Join]]:
    """``repeating``: also the links that would count a row more than once -- to say why a question stops."""
    held = _FIXED.get()
    if held is None or held["model"] is not model:
        return _read_edges(model, unconfirmed, skip, repeating)
    key = (unconfirmed, frozenset(skip), repeating)
    if key not in held["edges"]:
        held["edges"][key] = _read_edges(model, unconfirmed, skip, repeating)
    return held["edges"][key]


def _read_edges(model: SemanticModel, unconfirmed: bool, skip: frozenset[str],
                repeating: bool) -> dict[str, list[Join]]:
    out: dict[str, list[Join]] = {}
    for j in sorted(model.joins.values(), key=lambda j: j.key):
        allowed = usable(j) or (unconfirmed and j.trust != "rejected")
        if j.to_calendar or not allowed or j.key in skip or (repeats(j) and not repeating):
            continue
        if model.tables.get(j.to_table) is not None and model.tables[j.to_table].kind == "calendar":
            continue
        out.setdefault(j.from_table, []).append(j)
    return out


def all_paths(model: SemanticModel, start: str, goal: str, *, through: str | None = None,
              unconfirmed: bool = False, skip: frozenset[str] = frozenset(), repeating: bool = False) -> list[Path]:
    """Every row-safe path from ``start`` to ``goal`` (optionally passing ``through`` a table), best first.

    ``unconfirmed`` also follows links waiting for an admin, to say what a question
    would need confirmed, never to answer it; ``repeating`` the links that would count a
    row more than once, to say why a question stops; ``skip`` leaves links out by key.
    """
    if start == goal:
        return [Path([])]
    return _walk(_edges(model, unconfirmed, skip, repeating), start, goal, through)


def _walk(edges: dict[str, list[Join]], start: str, goal: str, through: str | None = None) -> list[Path]:
    found: list[Path] = []
    stack: list[tuple[str, list[Join]]] = [(start, [])]
    while stack:
        table, path = stack.pop()
        if len(path) >= MAX_STEPS:
            continue
        for j in edges.get(table, []):
            if j.to_table == start or any(step.to_table == j.to_table for step in path):
                continue
            new = path + [j]
            if j.to_table == goal:
                candidate = Path(new)
                if through is None or through in candidate.tables[1:]:
                    found.append(candidate)
                continue
            stack.append((j.to_table, new))
    return sorted(found, key=lambda p: p.score)


def best_path(model: SemanticModel, start: str, goal: str, *, through: str | None = None,
              skip: frozenset[str] = frozenset()) -> Path | None:
    paths = all_paths(model, start, goal, through=through, skip=skip)
    if not paths:
        return None
    direct = [p for p in paths if len(p.joins) == 1]
    if len(direct) > 1:
        # Two direct links to the same table: a role-playing dimension (requested
        # by, approved by). A link without a role name is the table's own;
        # otherwise the question has to say which.
        plain = [p for p in direct if not p.joins[0].role]
        if len(plain) == 1:
            return plain[0]
        raise Ambiguous(direct)
    if not direct:
        # No link of its own: two longer routes as good as each other (the same trust, match and steps) may reach
        # different members. A route of links without a role is the table's own when it is the only one; routes
        # that all carry roles are asked about, as two direct links are; of several without, the first is taken
        # (its key decides, so always the same one) and carries the others, for the answer to name. A route through a
        # link with a role beside a plain one between the same tables (an invoice line's payer customer, through its
        # invoice, beside its customer) is taken only when asked for, as a direct one is, however well its values
        # match; an order's only customer link, its bill-to, is ranked like any other.
        plain = {(j.from_table, j.to_table) for p in paths for j in p.joins if not j.role}
        paths = [p for p in paths if not any(j.role and (j.from_table, j.to_table) in plain for j in p.joins)] or paths
        # Two links with a role between the same tables and none without (an invoice's bill-to and payer customer,
        # from its lines): which one is the question's to say, as for two direct links, whichever matches better. (A
        # route through the same tables with other links is one with other roles: the plain ones went above.)
        others = [p for p in paths[1:] if p.tables == paths[0].tables and roles(p) != roles(paths[0])]
        if others:
            raise Ambiguous([paths[0], *others])
        tied = [p for p in paths if p.score[:4] == paths[0].score[:4]]
        if len(tied) > 1:
            plain = [p for p in tied if not roles(p)]
            if not plain:
                raise Ambiguous(tied)
            plain[0].others = plain[1:]
            return plain[0]
    return paths[0]


def distinct(path: Path, others: list[Path]) -> str | None:
    """The table that sets ``path`` apart from the other routes (its customer, theirs the supplier): the first it
    passes that some other does not."""
    for table in path.tables[1:-1]:
        if any(table not in other.tables for other in others if other is not path):
            return table
    return None


def reverse(j: Join) -> Join:
    """``j`` followed from its target back to its source: a bridge table entered from one of the tables it links.
    Its key starts with "~", so a path walked through it is told apart from one walked forward."""
    return j.model_copy(update={
        "key": f"~{j.key}", "from_table": j.to_table, "to_table": j.from_table, "from_columns": list(j.to_columns),
        "to_columns": list(j.from_columns), "conditions": [], "role": None,
        "cardinality": "one_to_one" if j.cardinality == "one_to_one" else "one_to_many"})


def is_reverse(j: Join) -> bool:
    return j.key.startswith("~")


def crossings(model: SemanticModel, start: str, goal: str, *, through: str | None = None,
              skip: frozenset[str] = frozenset()) -> list[Path]:
    """Paths from ``start`` to ``goal`` that cross one bridge table (an actor in many films: FILM_ACTOR), best
    first: row-safe steps to a table the bridge links, into the bridge, out to the other table, row-safe steps on.

    Only a bridge holding each pair once (its key is the two links' columns) is crossed: then a row of ``start``
    is under each member of ``goal`` at most once, and a figure for one member is right. Across members it is
    counted once for each (``overlaps``), unless the bridge holds one row for each of the entered table's.
    """
    found: list[Path] = []
    edges: dict[str, list[Join]] | None = None
    walked: dict[tuple[str, str], list[Path]] = {}

    def paths(a: str, b: str) -> list[Path]:
        # The model does not change while the bridges are tried: its links are read once, each route walked once
        # (a schema of 70 tables otherwise read them hundreds of thousands of times for one question map).
        nonlocal edges
        if a == b:
            return [Path([])]
        if (a, b) not in walked:
            if edges is None:
                edges = _edges(model, False, skip, False)
            walked[(a, b)] = _walk(edges, a, b)
        return walked[(a, b)]

    for bridge in sorted(model.tables.values(), key=lambda t: t.key):
        if bridge.kind != "bridge" or not bridge.primary_key or bridge.key in (start, goal):
            continue
        out = [j for j in model.joins.values() if j.from_table == bridge.key and usable(j) and not repeats(j)
               and not j.to_calendar and j.key not in skip]
        for entered in out:
            for left in out:
                if left.key == entered.key or left.to_table == entered.to_table:
                    continue
                if not set(bridge.primary_key) <= set(entered.from_columns) | set(left.from_columns):
                    continue          # a pair could be held twice: a row would count twice under one member
                before = paths(start, entered.to_table)
                after = paths(left.to_table, goal)
                if not before or not after:
                    continue
                joins = before[0].joins + [reverse(entered), left] + after[0].joins
                path = Path(joins)
                tables = path.tables
                if len(set(tables)) != len(tables) or (through is not None and through not in tables[1:]):
                    continue
                found.append(path)
    return sorted(found, key=lambda p: (len(p.joins), p.score))


def overlaps(path: Path) -> bool:
    """Does ``path`` count a row once under each of several members (entering a bridge that holds several
    rows for each row of the table it is entered from)?"""
    return any(is_reverse(j) and j.cardinality != "one_to_one" for j in path.joins)


def bridge_of(path: Path) -> str | None:
    """The bridge table ``path`` crosses, if it crosses one."""
    return next((j.to_table for j in path.joins if is_reverse(j)), None)


def roles(path: Path) -> list[str]:
    return [j.role for j in path.joins if j.role]


def _words(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", text.casefold()))


def _and_self_links(model: SemanticModel, start: str, goal: str, through: str | None = None) -> list[Path]:
    """Every row-safe path to ``goal``, and each of them followed one step further along a link of ``goal`` to
    itself ("Manager": an employee's manager is another employee; "Parent account"). Such a link is never a
    way to reach a table: only a role a question names follows it."""
    paths = all_paths(model, start, goal, through=through)
    selves = [j for j in _edges(model).get(goal, []) if j.to_table == goal and j.role]
    return paths + [Path([*p.joins, j]) for j in selves for p in paths
                    if not any(step.key == j.key for step in p.joins)]


def with_role(model: SemanticModel, start: str, goal: str, role: str, *, through: str | None = None) -> list[Path]:
    """The paths that go through a link named ``role`` ("Bill to customer", "Home store"), best first.

    Spelling is forgiven ("ship-to customer"), and so is a role written without its
    target's noun ("Ship to" for "Ship to customer") when only one role starts so. A table's link to
    itself is followed by its role ("Manager" of an employee).

    A role kept by a question asked before Learn named the links in whole words is still found: "Pyr
    customer" is the "Payer customer", and "Customer", the name Learn once gave the plain link beside a
    payer customer, goes the way a question naming no role does, from the invoice or from its lines.
    """
    from core2.bootstrap import names

    def party(text: str) -> tuple[str, ...]:
        return tuple(names.PARTY_WORDS.get(w, w) for w in _words(text))

    wanted = _words(role)
    paths = _and_self_links(model, start, goal, through=through)
    named = {r for p in paths for r in roles(p)}
    chosen = {r for r in named if _words(r) == wanted}
    parts = [part.strip() for part in role.split(",") if part.strip()]
    if not chosen and len(parts) > 1:
        # A choice offered by two roles ("Original order, Payer customer"): the route carrying each of them.
        found = [{tuple(j.key for j in p.joins) for p in with_role(model, start, goal, part, through=through)}
                 for part in parts]
        return [p for p in paths if all(tuple(j.key for j in p.joins) in keys for keys in found)]
    if not chosen and wanted:
        chosen = {r for r in named if party(r) == party(role)}
        if len(chosen) > 1:
            return []
    if not chosen and wanted:
        chosen = {r for r in named if _words(r)[:len(wanted)] == wanted}
        if len(chosen) > 1:
            return []          # "Ship" could be either: the reader is asked, never guessed for
    if not chosen and wanted and start != goal and wanted in (
            _words(model.tables[goal].business_name),
            _words(names.read_tokens(names.core_table(model.tables[goal].name)))):
        try:
            path = best_path(model, start, goal, through=through)
        except Ambiguous:
            return []
        return [path] if path is not None and not roles(path) else []
    return [p for p in paths if chosen & set(roles(p))]


def through_entities(model: SemanticModel, start: str, goal: str) -> list[str]:
    """The tables of the things ``start``'s rows reach ``goal`` through by different routes, each the table that
    links to ``goal`` (an order line's nation is its customer's or its supplier's): [] for one route or none."""
    entities = {e.table for e in model.entities.values()}
    found: list[str] = []
    for p in all_paths(model, start, goal):
        last = p.tables[-2] if len(p.joins) > 1 else None
        if last in entities and last not in found:
            found.append(last)
    return found if len(found) > 1 else []


def role_names(model: SemanticModel, start: str, goal: str) -> list[str]:
    """The named links a question can choose between to reach ``goal`` from ``start``."""
    return sorted({r for p in _and_self_links(model, start, goal) for r in roles(p)})
