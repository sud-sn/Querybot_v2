"""Identity of warehouse objects, fixed once for every core2 writer and reader.

A table is identified by ``database.schema.table`` and a column by the table's
key plus ``.column``, each part as the warehouse spells it but compared through
:func:`norm`. Keys are what dictionaries in the semantic model are keyed by,
what overrides point at and what plans are bound to; objects keep the stored
spelling separately for writing SQL. Keys are never parsed back into parts.

Two tables whose names differ only by case (possible in Snowflake with quoted
identifiers) share a key. That is accepted: such a pair is vanishingly rare and
a warehouse that has one confuses its own users the same way.
"""

from __future__ import annotations

import re
import unicodedata

_QUOTES = '"`[]'


def norm(name: str | None) -> str:
    """The comparison form of one identifier part: unquoted and casefolded."""
    text = (name or "").strip()
    if len(text) >= 2 and text[0] in '"`[' and text[-1] in '"`]':
        text = text[1:-1]
    return text.casefold()


def table_key(database: str | None, schema: str | None, table: str) -> str:
    """``database.schema.table`` in comparison form; empty parts are left out."""
    return ".".join(norm(part) for part in (database, schema, table) if norm(part))


def column_key(table: str, column: str) -> str:
    """A column's key from its table's key (already in comparison form)."""
    return f"{table}.{norm(column)}"


def join_key(from_table: str, from_columns: list[str], to_table: str, to_columns: list[str],
             role: str | None = None) -> str:
    """One key per way two tables are joined; a role tells apart the same columns used twice."""
    left = ",".join(norm(c) for c in from_columns)
    right = ",".join(norm(c) for c in to_columns)
    key = f"{from_table}({left})->{to_table}({right})"
    return f"{key}#{slug(role)}" if role else key


def same(a: str | None, b: str | None) -> bool:
    return norm(a) == norm(b)


def slug(text: str | None, *, fallback: str = "item") -> str:
    """A planner-facing name: lowercase ASCII words joined by underscores.

    ``"Chiffre d'affaires (CAD)"`` becomes ``chiffre_d_affaires_cad``. Accents are
    folded, anything else that is not a letter or digit separates words, and a
    leading digit gets a prefix so the result is always a valid identifier.
    """
    folded = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode("ascii")
    words = re.findall(r"[a-z0-9]+", folded.lower())
    value = "_".join(words) or fallback
    return f"n{value}" if value[0].isdigit() else value


def unique_slug(base: str, taken: set[str]) -> str:
    """``base``, or ``base_2``, ``base_3``... whichever is free; the result is added to ``taken``."""
    candidate, n = base, 2
    while candidate in taken:
        candidate = f"{base}_{n}"
        n += 1
    taken.add(candidate)
    return candidate
