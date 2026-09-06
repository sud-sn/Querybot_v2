"""One verdict on whether a join can ever be used, for every writer of one.

The entity graph has three writers -- the admin canvas, the bulk editor, and
schema discovery -- and until now none of them asked whether the edge they
were storing could be used. ``core.join_planner.relationship_is_admissible``
has always known that a fact-to-fact join is prohibited and that a
many-to-many needs a bridge; it just ran at QUERY time, where a refusal turns
into "no path" and the admin who saved the edge never hears about it.

An edge saved through the canvas today comes back HTTP 200 with
``status='confirmed'`` even when it is a fact-to-fact many-to-many on a column
that does not exist. The planner then refuses it on every question, forever.
That is worse than an error message: the admin has a green tick and a dead
edge, and ``status='confirmed'`` means discovery will never overwrite it
either.

Two classes of problem, deliberately handled differently:

**Structurally impossible → refuse the write.** A fact-to-fact edge, a
many-to-many with no bridge, a dimension-to-fact direction: the planner will
never traverse these, so storing one creates work that can only ever be
undone. Telling somebody at the moment they press save is the cheapest
correction the product can offer.

**Unverifiable → store it and say so.** A column the discovered schema does
not contain may be a real column behind a stale discovery, and refusing on
that basis would make the product unusable the day a warehouse adds a field.
The edge is stored with ``validation_status='broken'`` so the graph health
page and the review queue can surface it.

Nothing here reads the store or the database. The caller passes what it has,
which is what lets the same function serve a save (rows not yet written), a
health sweep (rows already written), and a CSV import (rows in a file).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

log = logging.getLogger("querybot.join_governance")

# Verdict codes. Stable strings, because a CSV import reports them per row and
# an admin pasting one into a support ticket should get a hit.
REFUSE_UNKNOWN_ENTITY = "unknown_entity"
REFUSE_SELF_JOIN = "self_join"
REFUSE_NO_COLUMNS = "no_columns"
REFUSE_INADMISSIBLE = "inadmissible"
WARN_COLUMN_MISSING = "column_missing"
WARN_NO_SCHEMA = "no_schema"


@dataclass(frozen=True)
class JoinVerdict:
    """Can this edge ever be used, and if not, why not."""

    ok: bool = True
    code: str = ""
    reason: str = ""
    # "error" refuses the write; "warning" stores it and flags it.
    severity: str = ""
    missing_columns: tuple[str, ...] = ()

    @property
    def refuses(self) -> bool:
        return self.severity == "error"

    @property
    def flags(self) -> bool:
        return self.severity == "warning"

    def as_dict(self) -> dict:
        return {
            "ok": self.ok, "code": self.code, "reason": self.reason,
            "severity": self.severity,
            "missing_columns": list(self.missing_columns),
        }


def join_pairs(relationship: dict) -> list[tuple[str, str]]:
    """Every column pair this edge joins on, top-level and composite.

    Mirrors core.relationship_validator._join_pairs. Duplicated deliberately
    rather than imported: that module reaches the store and the live database
    on the way in, and this one must stay callable from a save handler that
    has neither.
    """
    pairs: list[tuple[str, str]] = []
    top = (str(relationship.get("from_column") or "").strip(),
           str(relationship.get("to_column") or "").strip())
    if top[0] and top[1]:
        pairs.append(top)

    raw = relationship.get("join_conditions") or []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw or "[]")
        except (TypeError, ValueError):
            raw = []
    for condition in raw or []:
        if not isinstance(condition, dict):
            continue
        left = str(condition.get("from_col") or "").strip()
        right = str(condition.get("to_col") or "").strip()
        if left and right and (left, right) not in pairs:
            pairs.append((left, right))
    return pairs


def _table_key(entity: dict) -> str:
    schema = str(entity.get("schema_name") or "").strip()
    table = str(entity.get("table_name") or "").strip()
    return f"{schema}.{table}".strip(".").upper()


def _columns_for(entity: dict, schema_columns: dict | None) -> set[str] | None:
    """The columns the discovered schema holds for this entity's table.

    None means "the schema cannot answer" -- no schema loaded, or the table
    is absent from it -- which is a different state from "the table has no
    columns" and must not be reported as a missing column.
    """
    if not schema_columns:
        return None
    table = str(entity.get("table_name") or "").strip().upper()
    if not table:
        return None
    qualified = _table_key(entity)
    for key, columns in schema_columns.items():
        candidate = str(key).upper()
        if candidate == qualified or candidate.split(".")[-1] == table:
            return {str(c).upper() for c in (columns or {})}
    return None


def check_join(
    relationship: dict,
    entities: dict[str, dict],
    schema_columns: dict | None = None,
) -> JoinVerdict:
    """Whether this edge can ever be used.

    ``entities`` maps entity_name to its graph row; ``schema_columns`` is the
    discovered schema as ``{table: {COLUMN: type}}`` (what
    core.schema.load_schema_columns returns), or None when none is available.
    """
    from core.join_planner import relationship_is_admissible

    from_name = str(relationship.get("from_entity") or "").strip()
    to_name = str(relationship.get("to_entity") or "").strip()
    from_entity = entities.get(from_name)
    to_entity = entities.get(to_name)

    if not from_entity or not to_entity:
        missing = from_name if not from_entity else to_name
        return JoinVerdict(
            ok=False, code=REFUSE_UNKNOWN_ENTITY, severity="error",
            reason=(f"'{missing or '(blank)'}' is not an entity in this "
                    f"workspace. Add the table to the graph first."),
        )

    pairs = join_pairs(relationship)
    if not pairs:
        return JoinVerdict(
            ok=False, code=REFUSE_NO_COLUMNS, severity="error",
            reason="A join needs a column on each side.",
        )

    if from_name == to_name and any(left.upper() == right.upper()
                                    for left, right in pairs):
        # A self-join on the same column matches every row to itself: it adds
        # no rows, no columns and no meaning, and it makes the traversal
        # search visit the table twice.
        return JoinVerdict(
            ok=False, code=REFUSE_SELF_JOIN, severity="error",
            reason=(f"'{from_name}' joined to itself on the same column has "
                    f"no effect."),
        )

    admissible, why = relationship_is_admissible(relationship, entities)
    if not admissible:
        return JoinVerdict(
            ok=False, code=REFUSE_INADMISSIBLE, severity="error",
            # The planner's own words. One vocabulary for one rule, so the
            # message an admin gets at save time is the message a trace shows
            # when a query is refused.
            reason=(f"This join cannot be used: {why}. "
                    f"{_remedy_for(why)}").strip(),
        )

    from_columns = _columns_for(from_entity, schema_columns)
    to_columns = _columns_for(to_entity, schema_columns)
    if from_columns is None or to_columns is None:
        return JoinVerdict(
            ok=True, code=WARN_NO_SCHEMA, severity="warning",
            reason=("The join is structurally sound but its columns could not "
                    "be checked: no discovered schema covers these tables."),
        )

    missing: list[str] = []
    for left, right in pairs:
        if left.upper() not in from_columns:
            missing.append(f"{from_name}.{left}")
        if right.upper() not in to_columns:
            missing.append(f"{to_name}.{right}")
    if missing:
        # Stored, not refused: a column absent from the DISCOVERED schema may
        # be a real column behind a stale discovery, and refusing on that
        # basis breaks the product the day a warehouse adds a field.
        return JoinVerdict(
            ok=True, code=WARN_COLUMN_MISSING, severity="warning",
            reason=("Saved, but not found in the discovered schema: "
                    + ", ".join(missing[:4])
                    + ". Re-run schema discovery, or check the spelling."),
            missing_columns=tuple(missing),
        )

    return JoinVerdict(ok=True)


_REMEDIES = {
    "raw fact-to-fact joins are prohibited":
        "Join both facts to a shared dimension instead, or model the link "
        "explicitly as a bridge.",
    "many-to-many relationship requires an explicit bridge":
        "Set one side's role to 'bridge', or pick the column pair that makes "
        "this many-to-one.",
    "dimension-to-fact fk direction is not a star/snowflake edge":
        "Swap the two sides: the table holding the foreign key goes first.",
    "relationship validation is broken":
        "Re-run validation on this join, or fix the columns it names.",
    "relationship was rejected":
        "Re-open the join from the review queue before editing it.",
}


assert all(key == key.lower() for key in _REMEDIES), \
    "a _REMEDIES key with upper-case can never be matched"


def _remedy_for(reason: str) -> str:
    """What to do about it, beside why it was refused.

    A refusal that only names a rule leaves the admin exactly where they were.
    Falls back to nothing rather than to generic advice: "try something else"
    is worse than silence next to a specific reason.

    Keys are lower-cased on both sides. The planner's reasons carry mixed case
    ("dimension-to-fact FK direction..."), and a table keyed on the original
    spelling silently loses its entry the day somebody edits the case of one
    word in the sentence.
    """
    return _REMEDIES.get(str(reason or "").strip().lower(), "")
