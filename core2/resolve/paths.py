"""One rule for join paths, used by every query shape (DESIGN §6.4).

From a measure's table to the table holding an attribute:

1. only steps that never multiply the measure's rows (many-to-one or one-to-one),
   never through a calendar, never back through a table already on the path;
2. the measure table's own key first: a direct link wins over any longer path;
3. then the most trusted weakest link, then the best weakest match rate, then
   fewer steps, then the smallest join keys (deterministic, and reported).

When two different direct links tie (a requester and an approver both pointing
at the people table), the rule does not guess: the question must say which,
or an admin's default decides.
"""

from __future__ import annotations

from dataclasses import dataclass

from core2.model.schema import Join, SemanticModel

TRUST_RANK = {"admin": 0, "verified": 1, "declared": 2, "proposed": 3}
MAX_STEPS = 4


@dataclass
class Path:
    joins: list[Join]

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


def _edges(model: SemanticModel) -> dict[str, list[Join]]:
    out: dict[str, list[Join]] = {}
    for j in sorted(model.joins.values(), key=lambda j: j.key):
        if j.to_calendar or j.trust == "rejected" or j.cardinality not in ("many_to_one", "one_to_one"):
            continue
        if model.tables.get(j.to_table) is not None and model.tables[j.to_table].kind == "calendar":
            continue
        out.setdefault(j.from_table, []).append(j)
    return out


def all_paths(model: SemanticModel, start: str, goal: str, *, through: str | None = None) -> list[Path]:
    """Every row-safe path from ``start`` to ``goal`` (optionally passing ``through`` a table), best first."""
    if start == goal:
        return [Path([])]
    edges = _edges(model)
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


def best_path(model: SemanticModel, start: str, goal: str, *, through: str | None = None) -> Path | None:
    paths = all_paths(model, start, goal, through=through)
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
    return paths[0]
