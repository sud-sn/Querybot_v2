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


_FILLER = frozenset({"name", "names", "list", "code", "codes", "description", "descriptions", "label", "labels"})


def _singular(word: str) -> str:
    """"items" is "item", "categories" "category", "addresses" "address"."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("sses", "shes", "ches", "xes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _words(text: str) -> set[str]:
    from core.word_forms import base_form

    return {_singular(base_form(word)) for word in re.findall(r"[a-z0-9]+", str(text or "").lower())}


def listed_field(semantic_plan: dict | None, fact_tables: set[str] | None, target: str = "",
                 role_labels: list[str] | None = None) -> dict | None:
    """The one dimension column a listing reads, or None.

    None unless the plan reads exactly that one column, of a dimension, and
    every word of what the reader asked for is the column's own -- "item
    names" of the item name, not "FITTINGS items", "item prices" or "French
    item names" -- and none is a role a fact reaches the dimension in: "the
    buyers" are the parties some fact names as its buyer, which the
    dimension alone does not say."""
    facts = {_bare(table) for table in fact_tables or ()}
    found: dict[tuple[str, str], dict] = {}
    for field in (semantic_plan or {}).get("fields") or []:
        if not isinstance(field, dict):
            continue
        found.setdefault((_bare(field.get("table")), str(field.get("column") or "").upper()), field)
    if len(found) != 1:
        return None
    field = next(iter(found.values()))
    if str(field.get("role") or "") not in _LISTED_ROLES or _bare(field.get("table")) in facts:
        return None
    if not field.get("table") or not field.get("column"):
        return None
    # A key is no member's name: "the customers" are not 1, 2, 3, 4.
    if re.search(r"(?:_?KEY|_?ID|_?SK)$", str(field.get("column")), re.I):
        return None
    asked = _words(target) - _words(" ".join(_FILLER))
    from core.semantic_planner import _words_form

    own = _words(field.get("term")) | _words(_words_form(str(field.get("column"))).replace("_", " "))
    if target and not asked <= own:
        return None
    if asked & _words(" ".join(role_labels or [])):
        return None
    return field


def listing_sql(field: dict, db_type: str, limit: int = LIMIT, *, leave_out: dict | None = None) -> str:
    """Its distinct members, in order: SELECT DISTINCT TOP n column FROM its
    table, written for ``db_type``. ``leave_out`` is the table's unknown-
    member policy (core.unknown_members): its placeholder rows are no
    members. One more than ``limit`` is read, so a list cut short can say so."""
    parts = [part for part in re.split(r"\.", re.sub(r"[\[\]\"`]", "", str(field["table"]))) if part][-2:]
    table = ".".join(f"[{part}]" for part in parts)
    column = re.sub(r"[\[\]\"`]", "", str(field["column"]))
    alias = re.sub(r"[^A-Za-z0-9]+", "_", str(field.get("term") or column)).strip("_").upper() or column.upper()
    where = f"listed.[{column}] IS NOT NULL"
    if leave_out and leave_out.get("keys") and leave_out.get("key_column"):
        from core.unknown_members import exclusion_predicate

        key = re.sub(r"[\[\]\"`]", "", str(leave_out["key_column"]))
        where += " AND " + exclusion_predicate(f"listed.[{key}]", leave_out)
    sql = (f"SELECT DISTINCT TOP {int(limit) + 1} listed.[{column}] AS [{alias}] FROM {table} AS listed "
           f"WHERE {where} ORDER BY listed.[{column}]")
    dialect = _DIALECTS.get(str(db_type or "").lower(), str(db_type or "").lower() or "tsql")
    if dialect == "tsql":
        return sql
    import sqlglot

    return sqlglot.transpile(sql, read="tsql", write=dialect)[0]
