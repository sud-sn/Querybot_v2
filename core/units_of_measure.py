"""
A quantity is counted in its item's unit, and units do not add up.

A stock fact holds each item's quantity in that item's unit of measure: most
items counted in eaches (EA), some in feet (FT), metres (ME), rolls (RL).
Summed across items, "total stock on hand" adds eaches to feet to metres -- a
number that measures nothing, from SQL that looks entirely ordinary.

So a total of a quantity is read per unit: the query groups by the unit of
measure, keeps to one unit, or stays within one item (an item has one unit).
The rule rides on the semantic plan like the period-row rule: the prompt states
it, the validator refuses a total that mixes units, and the answer card says
the quantities are per unit. A value -- a quantity times a cost -- is currency
and still adds up across units.

Tenant-neutral: the policy comes from the discovered schema and the graph --
which facts hold quantities, which columns name a unit of measure (the fact's
own, or that of the item dimension the fact joins), and which identify an item.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger("querybot.units_of_measure")

_FLAG_SUFFIXES = ("_FLG", "_FLAG", "_IND")
_ITEM_NAME_SUFFIXES = ("CD", "CODE", "NM", "NAME", "DSC", "DESC", "NUM", "NO", "ID")
_DIMENSION_AFFIXES = re.compile(r"^(?:DIM_|D_)|(?:_DMS|_DIM|_D)$")

_ASKED_FOR = re.compile(
    r"\b(?:all|any|regardless\s+of|across|whatever)\s+(?:the\s+)?units?\b"
    r"|\btoutes?\s+(?:les\s+)?unit[ée]s\b|quelle\s+que\s+soit\s+l['’]unit[ée]",
    re.IGNORECASE,
)


def _tokens(column: str) -> list[str]:
    return [t for t in str(column or "").upper().split("_") if t]


def is_unit_column(column: str) -> bool:
    """UNT_OF_MSR, UOM, BASE_UOM, UNIT_OF_MEASURE -- not UNIT_PRICE."""
    tokens = _tokens(column)
    return "UOM" in tokens or (
        bool({"UNT", "UNIT"} & set(tokens)) and bool({"MSR", "MEASURE"} & set(tokens))
    )


def is_quantity_column(column: str) -> bool:
    """ON_HND_QTY, QTY_ON_HAND, ORDER_QUANTITY -- not a flag about one."""
    name = str(column or "").upper()
    tokens = _tokens(name)
    return (
        bool({"QTY", "QUANTITY"} & set(tokens))
        and not name.endswith(_FLAG_SUFFIXES)
        and not is_unit_column(name)
    )


def question_asks_across_units(*texts: str) -> bool:
    return any(_ASKED_FOR.search(str(text or "")) for text in texts)


def _qualified(entity: dict) -> str:
    schema = str(entity.get("schema_name") or "").strip()
    table = str(entity.get("table_name") or "").strip()
    return f"{schema}.{table}" if schema else table


def _item_columns(entity: dict, columns: list[tuple[str, str]], key: str) -> list[str]:
    """The columns that name one item: its key, and its own code, name or
    description (ITM_CD, ITM_NM) -- not ITM_TYP_CD, which names a type."""
    stem = _DIMENSION_AFFIXES.sub("", str(entity.get("table_name") or "").upper())
    own = {f"{stem}_{suffix}" for suffix in _ITEM_NAME_SUFFIXES}
    return [key] + [name for name, _ in columns if name.upper() in own and name.upper() != key.upper()]


def unit_policies(account_id: str) -> list[dict]:
    """One policy per fact that holds quantities and has a unit of measure --
    its own column, or the one on a dimension it joins."""
    import store
    from core.unknown_members import _columns_of, _discovered_columns

    state = store.get_client_state(account_id) or {}
    tables = _discovered_columns(state.get("schema_dir", ""))
    if not tables:
        return []
    entities = {e["entity_name"]: e for e in store.list_entities(account_id, active_only=True)}
    relationships = store.list_relationships(account_id, active_only=True)
    policies: list[dict] = []
    for entity in entities.values():
        if (entity.get("entity_type") or "") != "fact":
            continue
        columns = _columns_of(entity, tables)
        quantities = [name for name, _ in columns if is_quantity_column(name)]
        if not quantities:
            continue
        fact = _qualified(entity)
        units = [{"table": fact, "column": name} for name, _ in columns if is_unit_column(name)]
        items: list[dict] = []
        joins: list[dict] = []
        for rel in relationships:
            if rel.get("from_entity") != entity["entity_name"]:
                continue
            target = entities.get(rel.get("to_entity"))
            if target is None:
                continue
            target_columns = _columns_of(target, tables)
            target_units = [name for name, _ in target_columns if is_unit_column(name)]
            if not target_units:
                continue
            table = _qualified(target)
            units.extend({"table": table, "column": name} for name in target_units)
            joins.append({
                "table": table,
                "fact_column": str(rel.get("from_column") or ""),
                "key": str(rel.get("to_column") or ""),
            })
            items.append({"table": fact, "column": str(rel.get("from_column") or "")})
            items.extend(
                {"table": table, "column": name}
                for name in _item_columns(target, target_columns, str(rel.get("to_column") or ""))
            )
        if not units:
            continue
        policies.append({
            "kind": "units_of_measure",
            "fact_table": fact,
            "quantities": quantities,
            "unit_columns": _unique(units),
            "item_columns": _unique([item for item in items if item["column"]]),
            "unit_joins": _unique([join for join in joins if join["fact_column"] and join["key"]]),
        })
    return sorted(policies, key=lambda p: p["fact_table"])


def _unique(refs: list[dict]) -> list[dict]:
    seen: list[dict] = []
    for ref in refs:
        if ref not in seen:
            seen.append(ref)
    return seen


def attach_unit_policies(semantic_plan: dict | None, account_id: str, *questions: str) -> list[dict]:
    """Put the policies on the plan -- unless the question asks for a total
    across units, when it gets one."""
    if not isinstance(semantic_plan, dict) or question_asks_across_units(*questions):
        return []
    policies = unit_policies(account_id)
    if policies:
        semantic_plan["unit_policies"] = policies
    return policies


def _bare(name: str) -> str:
    return str(name or "").split(".")[-1].strip('[]"`').upper()


def policies_in_scope(policies: list[dict] | None, *texts: str) -> list[dict]:
    haystack = " ".join(str(text or "") for text in texts).upper()
    return [
        policy for policy in policies or []
        if isinstance(policy, dict) and re.search(
            rf"(?<![A-Z0-9_]){re.escape(_bare(policy.get('fact_table', '')))}(?![A-Z0-9_])", haystack,
        )
    ]


def _preferred_unit(policy: dict) -> dict:
    """The item's unit before the fact's own: the item always has one, and a
    fact's copy can be blank on most rows."""
    fact = _bare(policy["fact_table"])
    return next(
        (ref for ref in policy["unit_columns"] if _bare(ref["table"]) != fact),
        policy["unit_columns"][0],
    )


def format_unit_rules(policies: list[dict] | None) -> str:
    lines: list[str] = []
    for policy in policies or []:
        unit = _preferred_unit(policy)
        others = [
            f"{_bare(ref['table'])}.{ref['column']}" for ref in policy["unit_columns"] if ref != unit
        ]
        lines.append(
            f"- {policy['fact_table']}: {', '.join(policy['quantities'][:8])}"
            + (", ..." if len(policy["quantities"]) > 8 else "")
            + f" are in each item's unit ({_bare(unit['table'])}.{unit['column']}"
            + (f"; also {', '.join(others)}" if others else "") + ")."
        )
    if not lines:
        return ""
    return "\n".join([
        "## Units of measure — REQUIRED",
        "A quantity is counted in its item's unit of measure (each, feet, metres, ...), and "
        "different units do not add up. A SUM or AVG of a quantity across items MUST keep one "
        "total per unit: put the unit column in the SELECT and the GROUP BY (join the item "
        "table for it if needed) -- or filter to one unit, or group by item. A value (quantity "
        "times a cost) is currency and adds up across units.",
        *lines,
    ])


# ── Reading a query for a total that mixes units ────────────────────────────


def _quantity_aggregates(select, quantities: set[str]) -> list:
    """The SUM/AVG in this SELECT whose argument is made of quantities alone
    (ON_HND_QTY, ON_HND_QTY - ALC_QTY) -- not a value (ON_HND_QTY * ITM_CST)."""
    from sqlglot import exp

    found = []
    for node in select.find_all(exp.Sum, exp.Avg):
        if node.find_ancestor(exp.Select) is not select:
            continue
        columns = [str(c.name).upper() for c in node.this.find_all(exp.Column)]
        if columns and all(name in quantities for name in columns):
            found.append(node)
    return found


def _parse(sql, db_type: str):
    """The query's tree -- read the way the validator reads it -- or None."""
    import sqlglot

    from core.validator import _DIALECT, normalize_generated_sql

    if not isinstance(sql, str):
        return sql
    try:
        return sqlglot.parse_one(
            normalize_generated_sql(sql, db_type), read=_DIALECT.get(db_type, "snowflake"),
        )
    except Exception:
        return None


def _grouped_names(select) -> set[str]:
    from core.unknown_members import _grouped_columns

    if select.args.get("group") is None:
        return set()
    return {str(column.name).upper() for column in _grouped_columns(select)}


def unit_mixes(sql, policies: list[dict] | None, db_type: str = "azure_sql") -> list[dict]:
    """Each SELECT that totals a quantity across items without keeping units
    apart, with the fix: ``[{"policy", "required"}]``. Empty when every total
    is per unit, per item, or of one unit."""
    from sqlglot import exp

    from core.unknown_members import _conjuncts

    governed = [p for p in policies or [] if isinstance(p, dict) and p.get("quantities")]
    tree = _parse(sql, db_type) if governed else None
    if tree is None:
        return []

    mixes: list[dict] = []
    for select in tree.find_all(exp.Select):
        for policy in governed:
            if not _quantity_aggregates(select, {q.upper() for q in policy["quantities"]}):
                continue
            unit_names = {str(ref["column"]).upper() for ref in policy["unit_columns"]}
            item_names = {str(ref["column"]).upper() for ref in policy["item_columns"]}
            if _grouped_names(select) & (unit_names | item_names):
                continue
            if any(_keeps_one(conjunct, unit_names | item_names) for conjunct in _conjuncts(
                (select.args.get("where") or exp.Where()).this,
            )):
                continue
            mixes.append({"policy": policy, "required": _required(select, policy)})
    return mixes


def _keeps_one(conjunct, names: set[str]) -> bool:
    """column = 'EA', or column IN ('EA'): one unit, or one item."""
    from sqlglot import exp

    if isinstance(conjunct, exp.EQ):
        sides = (conjunct.this, conjunct.expression)
        return any(
            isinstance(a, exp.Column) and str(a.name).upper() in names and isinstance(b, exp.Literal)
            for a, b in (sides, sides[::-1])
        )
    if isinstance(conjunct, exp.In) and isinstance(conjunct.this, exp.Column):
        return str(conjunct.this.name).upper() in names and len(conjunct.expressions) == 1
    return False


def _required(select, policy: dict) -> str:
    """The unit column to group by, as this query can reach it. The item's
    unit is named before the fact's own copy, which can be blank on most rows
    -- with the join to reach it when the query does not read the item yet."""
    from core.unknown_members import _same_table, _sources

    sources = _sources(select)
    unit = _preferred_unit(policy)
    for table, _on, _inner in sources:
        if _same_table(table, unit["table"]):
            return f"{table.alias_or_name}.{unit['column']}"
    join = next((j for j in policy.get("unit_joins") or [] if j["table"] == unit["table"]), None)
    fact = next((t for t, _on, _inner in sources if _same_table(t, policy["fact_table"])), None)
    if join and fact is not None:
        item = _bare(unit["table"])
        return (
            f"{item}.{unit['column']} (JOIN {unit['table']} AS {item} "
            f"ON {fact.alias_or_name}.{join['fact_column']} = {item}.{join['key']})"
        )
    for ref in policy["unit_columns"]:
        for table, _on, _inner in sources:
            if _same_table(table, ref["table"]):
                return f"{table.alias_or_name}.{ref['column']}"
    return f"{unit['table']}.{unit['column']}"


def totals_per_unit(sql, policies: list[dict] | None, db_type: str = "azure_sql") -> bool:
    """Whether the query totals a quantity grouped by its unit of measure --
    an answer the card should say is per unit."""
    from sqlglot import exp

    governed = [p for p in policies or [] if isinstance(p, dict) and p.get("quantities")]
    tree = _parse(sql, db_type) if governed else None
    if tree is None:
        return False
    return any(
        _quantity_aggregates(select, {q.upper() for q in policy["quantities"]})
        and _grouped_names(select) & {str(ref["column"]).upper() for ref in policy["unit_columns"]}
        for select in tree.find_all(exp.Select)
        for policy in governed
    )
