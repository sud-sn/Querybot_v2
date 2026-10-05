"""
A plain listing: "list the item names", "what are the item groups?".

Such a question names one thing a dimension keeps and asks for its members.
It names no measure, so the analytical request plan -- which needs one -- was
left incomplete: the reader was asked which of two inventory snapshots to read
(neither is read for a list of items), and then told the semantic layer had no
measure to calculate. A listing reads the dimension alone: its distinct
members, in order, through the governed executor.
"""

from __future__ import annotations

import re

# "list item names", "show me the suppliers", "what are the item groups?"
_LISTING_RE = re.compile(
    r"^\s*(?:please\s+)?(?:can\s+you\s+)?(?:list|show(?:\s+me)?|give\s+me|display|what\s+are|which\s+are|name)"
    r"\s+(?:me\s+)?(?:all\s+)?(?:of\s+)?(?:the\s+|our\s+|my\s+)?(?P<thing>[a-z][a-z0-9 '\-]{1,60}?)\s*[?.!]*\s*$",
    re.I,
)
# What makes a "list" something more than its members: a breakdown, a
# measure, a period, a condition, a number.
_MORE_THAN_MEMBERS = re.compile(
    r"\b(?:by|per|each|top|bottom|total|sum|average|count|how\s+many|trend|last|this|since|between|with|without"
    r"|where|more|less|over|under|above|below|that|which|who|sold|stock|value|quantity|quantities|sales|revenue"
    # A member or a condition it is narrowed by: "items in the FITTINGS
    # group", "customers from Ontario" -- never dropped to list them all.
    r"|in|from|at|for|on|whose|named|called|like|except|excluding|not|no|only|without)\b|\d",
    re.I,
)
_LISTED_ROLES = frozenset({"attribute", "display_dimension", "dimension"})
_DIALECTS = {"azure_sql": "tsql", "sql_server": "tsql", "mssql": "tsql", "postgresql": "postgres"}
LIMIT = 200


# "what items do we have?", "which items are available?", "which warehouses
# are there?"
_HELD_RE = re.compile(
    r"^\s*(?:what|which)\s+(?P<thing>[a-z][a-z0-9 '\-]{1,40}?)\s+"
    r"(?:do|does)\s+(?:we|i|you|the\s+company)\s+(?:have|hold|keep|carry|stock)\s*[?.!]*\s*$"
    r"|^\s*(?:what|which)\s+(?P<other>[a-z][a-z0-9 '\-]{1,40}?)\s+(?:are|is)\s+(?:there|available|present|listed)\s*[?.!]*\s*$",
    re.I,
)
# Words that say which members are meant only in passing: "the available
# items present" are the items.
_IN_PASSING = re.compile(r"\b(?:available|availables|present|existing|current|all|our|the|of)\b", re.I)


def listing_target(question: str) -> str:
    """The thing a plain listing asks for the members of, or ""."""
    found = _LISTING_RE.match(str(question or ""))
    thing = found.group("thing") if found else ""
    if not thing:
        held = _HELD_RE.match(str(question or ""))
        thing = (held.group("thing") or held.group("other")) if held else ""
    if not thing or _MORE_THAN_MEMBERS.search(thing):
        return ""
    thing = " ".join(_IN_PASSING.sub(" ", thing).split()) or thing.strip()
    # A thing, not a sentence: "item names", "item groups", "suppliers".
    return thing if len(thing.split()) <= 3 else ""


def _bare(table: str) -> str:
    return re.sub(r"[\[\]\"`]", "", str(table or "")).split(".")[-1].strip().upper()


def listed_field(semantic_plan: dict | None, fact_tables: set[str] | None) -> dict | None:
    """The one dimension column a listing's plan reads, or None when it reads
    none, several, or a fact's."""
    facts = {_bare(table) for table in fact_tables or ()}
    found: dict[tuple[str, str], dict] = {}
    for field in (semantic_plan or {}).get("fields") or []:
        if not isinstance(field, dict) or str(field.get("role") or "") not in _LISTED_ROLES:
            continue
        table, column = str(field.get("table") or ""), str(field.get("column") or "")
        if not table or not column or _bare(table) in facts:
            continue
        found.setdefault((_bare(table), column.upper()), field)
    return next(iter(found.values())) if len(found) == 1 else None


def listing_sql(field: dict, db_type: str, limit: int = LIMIT) -> str:
    """Its distinct members, in order: SELECT DISTINCT TOP n column FROM its
    table, written for ``db_type``."""
    parts = [part for part in re.split(r"\.", re.sub(r"[\[\]\"`]", "", str(field["table"]))) if part][-2:]
    table = ".".join(f"[{part}]" for part in parts)
    column = re.sub(r"[\[\]\"`]", "", str(field["column"]))
    alias = re.sub(r"[^A-Za-z0-9]+", "_", str(field.get("term") or column)).strip("_").upper() or column.upper()
    sql = (f"SELECT DISTINCT TOP {int(limit)} listed.[{column}] AS [{alias}] FROM {table} AS listed "
           f"WHERE listed.[{column}] IS NOT NULL ORDER BY listed.[{column}]")
    dialect = _DIALECTS.get(str(db_type or "").lower(), str(db_type or "").lower() or "tsql")
    if dialect == "tsql":
        return sql
    import sqlglot

    return sqlglot.transpile(sql, read="tsql", write=dialect)[0]
