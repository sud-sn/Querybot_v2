from __future__ import annotations

import hashlib
import re
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
    "postgres": "postgres",
    "excel": "duckdb",
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
                    and part.find_ancestor(exp.Window) is None
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
                node for node in expression.walk()
                if isinstance(node, exp.AggFunc)
                and node.find_ancestor(exp.Window) is None
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


def aggregate_only_violations(
    analysis: SqlPolicyAnalysis, required: set[str]
) -> set[str]:
    """Which of `required` (resource keys under an aggregate_only obligation)
    reach the caller in a form other than a genuine, row-reducing aggregate.

    Three independent call sites -- execute_governed_query, chart_policy's
    aggregate_only_gate_passes, and the dashboard chart route -- each computed
    this themselves, all three the same way, and all three the same way wrong:
    `required_aggregate - aggregate_sources` only asks "does this resource
    appear in SOME aggregated output?" A column can satisfy that and still
    reach the reader raw:

        SELECT SUM(SALARY) AS TOTAL, SALARY AS RAW FROM EMPLOYEES

    TOTAL puts EMPLOYEES.SALARY in aggregate_sources, so the old check saw
    nothing missing -- RAW was never asked about. The obligation is about
    what reaches the reader, not about whether the query also computed
    something safe elsewhere in the same result set. So this checks the
    complement too: is the resource ALSO a source of some NON-aggregated
    output? If it is, that output IS the leak, regardless of what the
    aggregated one shows beside it.

    A window function (FIRST_VALUE/LAST_VALUE/LAG/NTH_VALUE, or a "safe"
    aggregate like SUM used with OVER (...)) never counts as the row-reducing
    kind: it preserves cardinality -- one output row per input row -- so it
    exposes the same raw value a bare column would. analyze_sql already
    excludes it from aggregate_outputs; this function only combines what
    analyze_sql computed, so the fix reaches all three call sites without
    each having to know why.
    """
    if not required:
        return set()
    aggregate_sources = {
        source
        for output, sources in analysis.lineage.items()
        if output in analysis.aggregate_outputs
        for source in sources
    }
    raw_sources = {
        source
        for output, sources in analysis.lineage.items()
        if output not in analysis.aggregate_outputs
        for source in sources
    }
    never_aggregated = required - aggregate_sources
    exposed_raw = required & raw_sources
    return never_aggregated | exposed_raw


def _resolve_value(condition: dict, context: PolicyContext) -> Any:
    source = str(condition.get("value_source") or "static")
    if source == "user.id":
        return context.user_id
    if source == "user.groups":
        # Their groups' names, never their numbers: a group's number is no value of anyone's data.
        return list(getattr(context, "group_names", None) or [g for g in context.groups if not str(g).isdigit()])
    if source.startswith("user.attributes."):
        return context.user_attributes.get(source.split(".", 2)[-1])
    return condition.get("value")


def as_number(value: Any) -> int | float | None:
    """A value as the number it is, or None: 1042, "1042" -> 1042; "12.5" -> 12.5; "D-12", "", True -> None."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    text = str(value).strip()
    if re.fullmatch(r"[+-]?\d+", text):
        return int(text)
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|[+-]?\d+[eE][+-]?\d+", text):
        return float(text)
    return None


def _literal(value: Any) -> exp.Expression:
    if value is None:
        return exp.Null()
    if isinstance(value, bool):
        return exp.Boolean(this=value)
    if isinstance(value, (int, float)):
        return exp.Literal.number(value)
    return exp.Literal.string(str(value))


def _never() -> exp.Expression:
    """A condition no row meets, in every dialect (T-SQL has no FALSE)."""
    return exp.EQ(this=exp.Literal.number(1), expression=exp.Literal.number(0))


def _condition_expression(condition: dict, alias: str, context: PolicyContext) -> exp.Expression:
    field = str(condition.get("field") or "").strip()
    if not field:
        raise ValueError("Row policy is missing a field.")
    operator = str(condition.get("operator") or "=").upper()
    value = _resolve_value(condition, context)
    column = exp.column(field, table=alias or None)
    # A reader's own values may be several ("their regions": North, East): equal to one of them.
    if operator == "=" and isinstance(value, list):
        operator = "IN"
    if operator in {"IN", "NOT IN"}:
        values = [item for item in (value if isinstance(value, list) else [value]) if item is not None and item != ""]
        if condition.get("numeric"):
            # An ID kept as a number, compared as one. A value that is no number ("D-12" typed as a person's own
            # value) matches no row: compared as written, it failed every question on the database's conversion.
            values = [n for n in (as_number(item) for item in values) if n is not None]
        if not values:
            # One of nothing: no row (a reader with no value of their own sees nothing, never everything).
            return _never() if operator == "IN" else exp.Not(this=_never())
        node = exp.In(this=column, expressions=[_literal(item) for item in values])
        return exp.Not(this=node) if operator == "NOT IN" else node
    if operator == "IS NULL":
        return exp.Is(this=column, expression=exp.Null())
    if operator == "IS NOT NULL":
        return exp.Not(this=exp.Is(this=column, expression=exp.Null()))
    if value is None or value == "":
        return _never()
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
    if subject_type == "everyone":
        # Every person: a reader in the portal, a person a schedule or an alert runs for. Not the workspace's own
        # service work (an admin's check of a link, Learn), which runs as no one.
        return context.role != "system"
    if subject_type == "role":
        return policy.get("subject_id") == context.role
    if subject_type == "user":
        return str(policy.get("subject_id")) == context.user_id
    if subject_type == "group":
        from core.compliance.policy_engine import in_group

        return in_group(str(policy.get("subject_id")), context)
    return True


def _policy_covers_table(policy: dict, table_name: str) -> bool:
    """Does this row policy govern the table the query names?"""
    configured = str(policy.get("table_fqn") or "").upper()
    if not configured:
        return False
    return (table_name == configured
            or table_name.endswith("." + configured)
            or configured.endswith("." + table_name))


def _sees_every_row(policy: dict) -> bool:
    return str((policy.get("condition") or {}).get("operator") or "").upper() == "ALL"


class _Links:
    """The workspace's links, read once per query and only when a rule needs them."""

    def __init__(self, account_id: str):
        self.account_id, self._found = account_id, None

    def __call__(self) -> list:
        if self._found is None:
            from core.compliance import row_links
            self._found = row_links.links(self.account_id)
        return self._found


def _binding(policies: list[dict], context: PolicyContext, table_name: str, links: _Links) -> list[tuple[dict, list]]:
    """Each rule that binds this reader and reaches `table_name`, with the way it reaches it: [] when the rule is
    written on that table, the links from it to the rule's table otherwise.

    Shared by inject_row_policies and row_policy_scope: if these two ever disagree about which rules bind a
    reader, a value kept for one reader is served to another."""
    from core.compliance import row_links

    out = []
    for policy in policies:
        if not _policy_subject_matches(policy, context):
            continue
        configured = str(policy.get("table_fqn") or "").upper()
        if not configured:
            continue
        if _policy_covers_table(policy, table_name):
            out.append((policy, []))
            continue
        way = row_links.path(table_name, configured, links())
        if way:
            out.append((policy, way))
    return out


def _reached(policy: dict, way: list, alias: str, context: PolicyContext, dialect: str) -> exp.Expression:
    """One rule as a condition on the table a query reads: on its own field when the rule is written on that
    table, else on the column that links it, towards the rule's table:
    ``s.store_id IN (SELECT rla1.store_id FROM stores AS rla1 WHERE rla1.region = 'West')``."""
    condition = policy.get("condition") or {}
    if not way:
        return _condition_expression(condition, alias, context)
    from core2.warehouse.dialect import ident, table_expr

    def column(name: str, at: str) -> exp.Column:
        return exp.Column(this=ident(name, dialect), table=exp.to_identifier(at))

    last = len(way)
    inner = f"rla{last}"
    query = (exp.select(column(way[-1].to_column, inner))
             .from_(table_expr(*way[-1].to_parts, dialect, alias=inner))
             .where(_condition_expression(condition, inner, context)))
    for i in range(last - 1, 0, -1):
        here = f"rla{i}"
        query = (exp.select(column(way[i - 1].to_column, here))
                 .from_(table_expr(*way[i - 1].to_parts, dialect, alias=here))
                 .where(exp.In(this=column(way[i].column, here), query=query.subquery())))
    return exp.In(this=column(way[0].column, alias), query=query.subquery())


def _restriction(bound: list[tuple[dict, list]], alias: str, context: PolicyContext,
                 dialect: str) -> exp.Expression | None:
    """What a reader may see of one table, or None when nothing narrows it.

    The rules on one field of one table are ways in, any of which lets a row through ("the West team's region",
    and Ada's own region besides): they are OR'd. The fields are each a limit of their own (a region, and a
    category): AND'd. A rule that sees every row of a field ("managers: every region") lifts that field."""
    fields: dict[tuple[str, str], list[tuple[dict, list]]] = {}
    for policy, way in bound:
        condition = policy.get("condition") or {}
        key = (str(policy.get("table_fqn") or "").upper(), str(condition.get("field") or "").upper())
        fields.setdefault(key, []).append((policy, way))
    limits = []
    for members in fields.values():
        if any(_sees_every_row(policy) for policy, _ in members):
            continue
        ways_in = [_reached(policy, way, alias, context, dialect) for policy, way in members]
        node = ways_in[0]
        for other in ways_in[1:]:
            node = exp.or_(node, other)
        limits.append(exp.Paren(this=node) if len(ways_in) > 1 else node)
    if not limits:
        return None
    node = limits[0]
    for other in limits[1:]:
        node = exp.and_(node, other)
    return node


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
    readers under one policy can still be looking at different rows. A rule on
    another table that reaches these through their links is part of it too.
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
    links = _Links(context.account_id)
    parts: list[str] = []
    for table_name in names:
        bound = _binding(policies, context, table_name, links)
        if not bound:
            continue
        try:
            restriction = _restriction(bound, "t", context, "snowflake")
            rendered = restriction.sql() if restriction is not None else "every row"
        except Exception:
            # An unsupported operator would have raised inside the
            # injection too. Keep it in the fingerprint rather than
            # treating the reader as unrestricted.
            rendered = repr([policy.get("condition") or {} for policy, _ in bound])
        ids = ",".join(str(policy.get("id")) for policy, _ in bound)
        parts.append(f"{table_name}|{ids}|{rendered}")
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
    links = _Links(context.account_id)
    # The tables the query names, listed before any condition is added: a condition's own subquery names the
    # rule's table, and must not be narrowed again.
    for table in list(tree.find_all(exp.Table)):
        if table.name.upper() in cte_aliases:
            continue
        table_name = _table_name(table)
        alias = table.alias_or_name or table.name
        bound = _binding(policies, context, table_name, links)
        if not bound:
            continue
        select = table.find_ancestor(exp.Select)
        if select is None:
            continue
        predicate = _restriction(bound, alias, context, dialect)
        for policy, way in bound:
            applied.append({
                "policy_id": policy["id"],
                "table": table_name,
                "condition": policy.get("condition") or {},
                "via": [link.to_table for link in way],
            })
        if predicate is None:
            continue
        key = id(select)
        if key not in predicates_by_select:
            predicates_by_select[key] = (select, [])
        predicates_by_select[key][1].append(predicate)
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
