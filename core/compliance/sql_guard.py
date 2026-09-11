from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.lineage import lineage as build_lineage

import store
from core.compliance.models import PolicyContext, ResourceRef


_DIALECT = {
    "azure_sql": "tsql",
    "snowflake": "snowflake",
    "oracle": "oracle",
}


@dataclass
class SqlPolicyAnalysis:
    sql: str
    resources: list[ResourceRef]
    tables: list[str] = field(default_factory=list)
    lineage: dict[str, list[str]] = field(default_factory=dict)
    aggregate_outputs: set[str] = field(default_factory=set)
    mask_exempt_outputs: set[str] = field(default_factory=set)
    has_star: bool = False


def _table_name(node: exp.Table) -> str:
    parts = [node.catalog, node.db, node.name]
    return ".".join(str(part) for part in parts if part).upper()


def analyze_sql(sql: str, db_type: str) -> SqlPolicyAnalysis:
    dialect = _DIALECT.get(db_type, "snowflake")
    tree = sqlglot.parse_one(sql, dialect=dialect)
    cte_aliases = {
        str(cte.alias).upper()
        for cte in tree.find_all(exp.CTE)
        if cte.alias
    }
    aliases: dict[str, str] = {}
    tables: list[str] = []
    for table in tree.find_all(exp.Table):
        if table.name.upper() in cte_aliases:
            continue
        name = _table_name(table)
        if not name:
            continue
        tables.append(name)
        aliases[(table.alias_or_name or table.name).upper()] = name
        aliases[table.name.upper()] = name

    resources: dict[str, ResourceRef] = {}
    lineage: dict[str, list[str]] = {}
    aggregate_outputs: set[str] = set()
    mask_exempt_outputs: set[str] = set()
    has_star = any(isinstance(node, exp.Star) for node in tree.walk())
    safe_aggregates = {
        "count", "sum", "avg", "stddev", "variance", "var",
        "stddev_pop", "variance_pop",
    }
    for expression in tree.selects:
        alias = str(expression.alias_or_name or expression.sql(dialect=dialect))
        sources: list[str] = []
        aggregate_nodes: list[exp.Expression] = []
        try:
            lineage_node = build_lineage(alias, tree, dialect=dialect)
            for node in lineage_node.walk():
                aggregate_nodes.extend(
                    part for part in node.expression.walk()
                    if isinstance(part, exp.AggFunc)
                )
                if not isinstance(node.expression, exp.Table):
                    continue
                source_name = str(node.name or "").rsplit(".", 1)[-1].strip('[]"`')
                if not source_name or source_name == "*":
                    continue
                table_name = _table_name(node.expression)
                if not table_name or node.expression.name.upper() in cte_aliases:
                    continue
                resource = ResourceRef(
                    table=table_name,
                    column=source_name,
                    output_alias=alias,
                )
                resources[resource.key] = resource
                sources.append(resource.key)
        except Exception:
            # Conservative fallback for expressions sqlglot lineage cannot
            # resolve. Qualified columns remain attributable; ambiguous bare
            # columns are not guessed.
            for column in expression.find_all(exp.Column):
                if column.name == "*":
                    continue
                table_name = aliases.get((column.table or "").upper(), "")
                if not table_name and len(tables) == 1:
                    table_name = tables[0]
                if not table_name:
                    continue
                resource = ResourceRef(table=table_name, column=column.name, output_alias=alias)
                resources[resource.key] = resource
                sources.append(resource.key)
            aggregate_nodes = [
                node for node in expression.walk() if isinstance(node, exp.AggFunc)
            ]

        lineage[alias] = sorted(set(sources))
        # UNION outputs are exempt only after branch-by-branch aggregation can
        # be proven. Until then, keep them maskable/aggregate-restricted.
        if aggregate_nodes and not isinstance(tree, exp.Union):
            aggregate_outputs.add(alias)
            if all(
                str(getattr(node, "key", "")).lower() in safe_aggregates
                for node in aggregate_nodes
            ):
                mask_exempt_outputs.add(alias)

    return SqlPolicyAnalysis(
        sql=sql,
        resources=list(resources.values()),
        tables=sorted(set(tables)),
        lineage=lineage,
        aggregate_outputs=aggregate_outputs,
        mask_exempt_outputs=mask_exempt_outputs,
        has_star=has_star,
    )


def _resolve_value(condition: dict, context: PolicyContext) -> Any:
    source = str(condition.get("value_source") or "static")
    if source == "user.id":
        return context.user_id
    if source == "user.groups":
        return context.groups
    if source.startswith("user.attributes."):
        return context.user_attributes.get(source.split(".", 2)[-1])
    return condition.get("value")


def _literal(value: Any) -> exp.Expression:
    if value is None:
        return exp.Null()
    if isinstance(value, bool):
        return exp.Boolean(this=value)
    if isinstance(value, (int, float)):
        return exp.Literal.number(value)
    return exp.Literal.string(str(value))


def _condition_expression(condition: dict, alias: str, context: PolicyContext) -> exp.Expression:
    field = str(condition.get("field") or "").strip()
    if not field:
        raise ValueError("Row policy is missing a field.")
    operator = str(condition.get("operator") or "=").upper()
    value = _resolve_value(condition, context)
    column = exp.column(field, table=alias or None)
    if operator in {"IN", "NOT IN"}:
        values = value if isinstance(value, list) else [value]
        node = exp.In(this=column, expressions=[_literal(item) for item in values])
        return exp.Not(this=node) if operator == "NOT IN" else node
    if operator == "IS NULL":
        return exp.Is(this=column, expression=exp.Null())
    if operator == "IS NOT NULL":
        return exp.Not(this=exp.Is(this=column, expression=exp.Null()))
    operations = {
        "=": exp.EQ,
        "!=": exp.NEQ,
        "<>": exp.NEQ,
        ">": exp.GT,
        ">=": exp.GTE,
        "<": exp.LT,
        "<=": exp.LTE,
    }
    cls = operations.get(operator)
    if not cls:
        raise ValueError(f"Unsupported row-policy operator: {operator}")
    return cls(this=column, expression=_literal(value))


def _policy_subject_matches(policy: dict, context: PolicyContext) -> bool:
    """Does this row policy bind the subject making the request?"""
    subject_type = policy.get("subject_type")
    if subject_type == "role":
        return policy.get("subject_id") == context.role
    if subject_type == "user":
        return str(policy.get("subject_id")) == context.user_id
    if subject_type == "group":
        return str(policy.get("subject_id")) in context.groups
    return True


def _policy_covers_table(policy: dict, table_name: str) -> bool:
    """Does this row policy govern the table the query names?"""
    configured = str(policy.get("table_fqn") or "").upper()
    if not configured:
        return False
    return (table_name == configured
            or table_name.endswith("." + configured)
            or configured.endswith("." + table_name))


def row_policy_scope(context: PolicyContext, tables: Any) -> str:
    """A fingerprint of the row restrictions this subject reads `tables` under.

    Empty when nothing restricts them -- so an unrestricted workspace keeps
    exactly the caching it had.

    This exists because a value READ OUT OF row-filtered data must not be
    cached under a subject-independent key. The business-date anchor was: the
    probe runs through execute_governed_query, so `MAX(invoice_date)` returns
    the newest date THIS READER can see, and the cache key was
    (account, fact, column). One reader's slice was then served to the whole
    workspace, including readers whose own data ends days earlier.

    The rendered condition is part of the fingerprint, not just the policy id,
    because a condition may interpolate the reader's own attributes -- two
    readers under one policy can still be looking at different rows.
    """
    try:
        policies = store.list_row_policies(
            context.account_id, context.policy_version or None)
    except Exception:                                          # pragma: no cover
        # Fail CLOSED: a scope we cannot compute must not collide with the
        # unrestricted one, or an unreadable policy table silently restores the
        # bug this function exists to prevent.
        return "unknown"
    if not policies:
        return ""

    names = sorted({
        str(name or "").strip().strip('[]"`').upper()
        for name in (tables or [])
        if str(name or "").strip()
    })
    parts: list[str] = []
    for table_name in names:
        for policy in policies:
            if not _policy_subject_matches(policy, context):
                continue
            if not _policy_covers_table(policy, table_name):
                continue
            try:
                rendered = _condition_expression(
                    policy.get("condition") or {}, "t", context).sql()
            except Exception:
                # An unsupported operator would have raised inside the
                # injection too. Keep it in the fingerprint rather than
                # treating the reader as unrestricted.
                rendered = repr(policy.get("condition") or {})
            parts.append(f"{table_name}|{policy.get('id')}|{rendered}")
    if not parts:
        return ""
    digest = hashlib.sha256("\n".join(sorted(parts)).encode("utf-8")).hexdigest()
    return f"v{int(context.policy_version or 0)}:{digest[:16]}"


def inject_row_policies(
    sql: str,
    db_type: str,
    context: PolicyContext,
) -> tuple[str, list[dict]]:
    policies = store.list_row_policies(context.account_id, context.policy_version or None)
    if not policies:
        return sql, []
    dialect = _DIALECT.get(db_type, "snowflake")
    tree = sqlglot.parse_one(sql, dialect=dialect)
    applied: list[dict] = []
    predicates_by_select: dict[int, tuple[exp.Select, list[exp.Expression]]] = {}
    cte_aliases = {
        str(cte.alias).upper()
        for cte in tree.find_all(exp.CTE)
        if cte.alias
    }
    for table in tree.find_all(exp.Table):
        if table.name.upper() in cte_aliases:
            continue
        table_name = _table_name(table)
        alias = table.alias_or_name or table.name
        for policy in policies:
            # Shared with row_policy_scope: if these two ever disagree about
            # which policies bind this reader, the anchor cache goes back to
            # serving one reader's slice to the workspace.
            if not _policy_subject_matches(policy, context):
                continue
            if not _policy_covers_table(policy, table_name):
                continue
            predicate = _condition_expression(policy.get("condition") or {}, alias, context)
            select = table.find_ancestor(exp.Select)
            if select is None:
                continue
            key = id(select)
            if key not in predicates_by_select:
                predicates_by_select[key] = (select, [])
            predicates_by_select[key][1].append(predicate)
            applied.append(
                {
                    "policy_id": policy["id"],
                    "table": table_name,
                    "condition": policy.get("condition") or {},
                }
            )
    for select, predicates in predicates_by_select.values():
        if not predicates:
            continue
        combined = predicates[0]
        for predicate in predicates[1:]:
            combined = exp.and_(combined, predicate)
        existing = select.args.get("where")
        if existing:
            select.set("where", exp.Where(this=exp.and_(existing.this, combined)))
        else:
            select.set("where", exp.Where(this=combined))
    return tree.sql(dialect=dialect), applied
