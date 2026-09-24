"""
Unknown members: the rows a dimension keeps for "no value" and "no match".

A warehouse that loads a fact row whose source value is empty, or matches no
member of a dimension, points the key at a reserved member rather than leaving
it NULL: key 0 "NO_VALUE -- NULL value provided", key 777 "NO_MATCH -- Value
provided does not match", -1 "Unknown". Grouped, ranked and counted as if they
were members, they make "NULL value provided" the biggest buyer and add two
suppliers that do not exist.

After discovery, each dimension with a one-column numeric key is asked for its
rows at the keys such members use. A row is an unknown member only when its
text reads as one too -- every value in its text columns, not just one -- so a
real member that happens to sit at key 0 is left alone. What is found is stored
with the dimension as a suggestion: an admin can reject it, and a later
discovery does not bring a rejected one back.
"""

from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Any

log = logging.getLogger("querybot.unknown_members")

NOT_SPECIFIED = "not_specified"
UNMATCHED = "unmatched"
UNKNOWN = "unknown"

# The keys warehouses reserve for these members. Being at one is not enough:
# the member's own text decides.
CANDIDATE_KEYS = (0, -1, -2, -3, -9, -99, -999, 777, 888, 999, 9999, 99999, 999999)

_UNMATCHED_TEXT = frozenset({
    "NO MATCH", "NO MATCH FOUND", "UNMATCHED", "NOT MATCHED", "NOT FOUND",
    "INVALID", "INVALID VALUE", "SANS CORRESPONDANCE",
})
_NOT_SPECIFIED_TEXT = frozenset({
    "NO VALUE", "NULL", "NULL VALUE", "N/A", "NOT APPLICABLE", "NONE",
    "NOT SPECIFIED", "UNSPECIFIED", "NOT PROVIDED", "MISSING", "BLANK", "EMPTY",
    "UNASSIGNED", "NOT ASSIGNED", "NOT AVAILABLE",
    "NON RENSEIGNE", "NON APPLICABLE", "AUCUN", "AUCUNE",
})
_UNKNOWN_TEXT = frozenset({"UNKNOWN", "UNK", "UNDEFINED", "?", "INCONNU", "INCONNUE"})

_NUMERIC_TYPES = ("INT", "NUMBER", "NUMERIC", "DECIMAL")
_TEXT_TYPES = ("CHAR", "TEXT", "STRING", "CLOB")

# A probe that cannot run ends the pass after this many: the warehouse is
# unreachable, and each attempt waits out its own timeout.
_GIVE_UP_AFTER = 3


def _normal(text: object) -> str:
    plain = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode()
    return re.sub(r"[\s_\-]+", " ", plain).strip().upper()


def _text_kind(text: object) -> str | None:
    """The kind of unknown member one value names; "" for a value that says
    nothing either way (empty, digits, punctuation); None for a real value."""
    t = _normal(text)
    if not t or not re.search(r"[A-Z?]", t):
        return ""
    if t in _UNMATCHED_TEXT or "NOT MATCH" in t or t.startswith("NO MATCH"):
        return UNMATCHED
    if t in _NOT_SPECIFIED_TEXT or t.startswith(("NULL VALUE", "NO VALUE")):
        return NOT_SPECIFIED
    if t in _UNKNOWN_TEXT:
        return UNKNOWN
    return None


def member_kind(texts: Iterable[object]) -> str | None:
    """What kind of unknown member a row's text values describe, or None
    when the row is a real member.

    Every value that says anything must read as an unknown member, and at
    least one must: key 0 named "Main warehouse" with a type of "N/A" is a
    warehouse.
    """
    kinds: set[str] = set()
    for text in texts:
        kind = _text_kind(text)
        if kind is None:
            return None
        if kind:
            kinds.add(kind)
    for kind in (UNMATCHED, NOT_SPECIFIED, UNKNOWN):
        if kind in kinds:
            return kind
    return None


def _discovered_columns(schema_dir: str) -> dict[str, list[tuple[str, str]]]:
    """{table (upper; full, SCHEMA.TABLE and bare): [(column, type)]}, the
    columns spelled as discovered -- a quoted identifier is case-sensitive."""
    from core.schema import _normalize_schema

    path = Path(schema_dir) / "_schema.json" if schema_dir else None
    if path is None or not path.exists():
        return {}
    try:
        master = _normalize_schema(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        log.warning("Unknown members: could not read %s: %s", path, exc)
        return {}
    tables: dict[str, list[tuple[str, str]]] = {}
    for fqn, info in master.items():
        if str(fqn).startswith("__") or not isinstance(info, dict):
            continue
        columns = [
            (str(col["name"]), str(col.get("type") or ""))
            for col in info.get("columns") or []
            if isinstance(col, dict) and col.get("name")
        ]
        parts = str(fqn).upper().split(".")
        for variant in (str(fqn).upper(), ".".join(parts[-2:]), parts[-1]):
            tables.setdefault(variant, columns)
    return tables


def _columns_of(entity: dict, tables: dict[str, list[tuple[str, str]]]) -> list[tuple[str, str]]:
    table = str(entity.get("table_name") or "").upper()
    schema = str(entity.get("schema_name") or "").upper()
    for key in (f"{schema}.{table}" if schema else "", table):
        if key and key in tables:
            return tables[key]
    return []


def _probe_columns(entity: dict, columns: list[tuple[str, str]]) -> tuple[str, list[str]] | None:
    """(key column, text columns to read) for a dimension that can have
    unknown members found this way, else None."""
    from core.schema_enrichment import _INFRA_PREFIXES

    key = str(entity.get("pk_column") or "").strip().upper()
    key_type = next((ctype for name, ctype in columns if name.upper() == key), "").upper()
    if not any(t in key_type for t in _NUMERIC_TYPES):
        return None
    # A load's audit columns (AZ_LST_UPD_USR = "dbo") say nothing about the
    # member, and would make every reserved row read as a real one.
    texts = [
        name for name, ctype in columns
        if any(t in ctype.upper() for t in _TEXT_TYPES)
        and not name.upper().startswith(_INFRA_PREFIXES)
    ]
    key_column = next(name for name, _ in columns if name.upper() == key)
    return (key_column, texts) if texts else None


def build_probe_sql(db_type: str, entity: dict, key_column: str, text_columns: list[str]) -> str:
    from core.relationship_validator import _quote_col, _quote_table

    table = _quote_table(entity.get("schema_name", ""), entity.get("table_name", ""), db_type)
    select = ", ".join(_quote_col(column, db_type) for column in [key_column, *text_columns])
    keys = ", ".join(str(k) for k in CANDIDATE_KEYS)
    return f"SELECT {select} FROM {table} WHERE {_quote_col(key_column, db_type)} IN ({keys})"


def _key_text(value: object) -> str:
    """A key as it is written in SQL: 0, not 0.0 or Decimal('0')."""
    try:
        number = float(str(value))
    except ValueError:
        return str(value)
    return str(int(number)) if number.is_integer() else str(value)


def members_in(rows: Iterable[tuple], text_columns: list[str]) -> list[dict]:
    """The unknown members among a probe's rows (key first, then the text
    columns in order)."""
    found = []
    for row in rows:
        values = list(row)
        kind = member_kind(values[1:])
        if kind is None:
            continue
        found.append({
            "key_value": _key_text(values[0]),
            "kind": kind,
            "member_text": {
                column: str(value) for column, value in zip(text_columns, values[1:])
                if value not in (None, "")
            },
        })
    return found


def detect_unknown_members(account_id: str, *, timeout_seconds: int = 20) -> dict[str, int]:
    """Find and store each dimension's unknown members. Returns counts:
    dimensions probed, members found, dimensions not probed (probe failed)."""
    import store
    from core.relationship_validator import run_probe

    summary = {"dimensions": 0, "members": 0, "not_probed": 0}
    client = store.get_client(account_id) or {}
    db_cfg_id = client.get("db_config_id")
    raw_cfg = store.get_db_config(db_cfg_id) if db_cfg_id else None
    if not raw_cfg:
        log.warning("Unknown members for %s: no warehouse connection is assigned", account_id)
        return summary
    db_type = raw_cfg.get("db_type", "azure_sql")
    state = store.get_client_state(account_id) or {}
    tables_columns = _discovered_columns(state.get("schema_dir", ""))

    # One probe per table: the graph can name one table twice (a date
    # dimension is also the canonical "Date" entity), and each name gets the
    # table's members.
    tables: dict[tuple[str, str], list[dict]] = {}
    for entity in store.list_entities(account_id, active_only=True):
        if (entity.get("entity_type") or "dimension") != "dimension":
            continue
        table = (str(entity.get("schema_name") or "").upper(), str(entity.get("table_name") or "").upper())
        tables.setdefault(table, []).append(entity)

    failed = 0
    for entities in tables.values():
        entity = entities[0]
        probe = _probe_columns(entity, _columns_of(entity, tables_columns))
        if probe is None:
            continue
        if failed >= _GIVE_UP_AFTER:
            log.warning(
                "Unknown members for %s: stopped after %d probes could not run",
                account_id, failed,
            )
            break
        key_column, text_columns = probe
        sql = build_probe_sql(db_type, entity, key_column, text_columns)
        try:
            rows = run_probe(db_type, raw_cfg, sql, timeout_seconds=timeout_seconds)
        except Exception as exc:
            log.warning("Unknown members for %s/%s: probe failed: %s: %s",
                        account_id, entity.get("table_name"), type(exc).__name__, exc)
            summary["not_probed"] += 1
            failed += 1
            continue
        members = members_in(rows, text_columns)
        for named in entities:
            store.save_unknown_members(account_id, str(named["entity_name"]), key_column, members)
        summary["dimensions"] += 1
        summary["members"] += len(members)
    return summary


# ── Rankings and counts leave them out ──────────────────────────────────────
#
# A listing or a total keeps an unknown member: its rows are real stock, only
# their buyer is not known. A ranking or a count of members does not: "NULL
# value provided" is not a top buyer, and NO_MATCH is not a supplier. Unless
# the question asks about them, the rule rides on the semantic plan, like the
# period-row rule: the prompt states the predicate, the validator refuses a
# ranking or a count that lacks it, and the answer card says they were left
# out.

_ASKED_FOR = re.compile(
    r"\b(?:unknown|unspecified|not\s+specified|unmatched|no\s+match|not\s+matched|"
    r"missing|unassigned|not\s+assigned|without|with\s+no|has\s+no|have\s+no|blank|"
    r"no\s+value|null|n/a|placeholders?|"
    r"inconnue?s?|non\s+renseign\w*|non\s+sp[ée]cifi\w*|sans\s+correspondance|"
    r"manquante?s?|non\s+attribu\w*|non\s+affect\w*|sans|vides?)\b",
    re.IGNORECASE,
)


def question_asks_for_unknown_members(*texts: str) -> bool:
    """Whether the question is about the members nobody could name -- "stock
    with no buyer", "articles sans fournisseur", "unmatched suppliers"."""
    return any(_ASKED_FOR.search(str(text or "")) for text in texts)


def _qualified(entity: dict) -> str:
    schema = str(entity.get("schema_name") or "").strip()
    table = str(entity.get("table_name") or "").strip()
    return f"{schema}.{table}" if schema else table


def _bare(name: str) -> str:
    return str(name or "").split(".")[-1].strip('[]"`').upper()


def unknown_member_policies(account_id: str) -> list[dict]:
    """One policy per dimension table with unknown members: its key, the
    members' keys, kinds and text, and the columns of other tables that point
    at it (a fact's key, a role's key)."""
    import store

    members = store.list_unknown_members(account_id)
    if not members:
        return []
    entities = {e["entity_name"]: e for e in store.list_entities(account_id, active_only=True)}
    policies: dict[str, dict] = {}
    for member in members:
        entity = entities.get(member["entity_name"])
        if not entity:
            continue
        table = _qualified(entity)
        policy = policies.setdefault(table.upper(), {
            "kind": "unknown_members",
            "table": table,
            "entities": [],
            "key_column": member["key_column"],
            "keys": [],
            "member_kinds": {},
            "member_text": {},
            "references": [],
        })
        if entity["entity_name"] not in policy["entities"]:
            policy["entities"].append(entity["entity_name"])
        key = str(member["key_value"])
        if key not in policy["keys"]:
            policy["keys"].append(key)
            policy["member_kinds"][key] = member["kind"]
            policy["member_text"][key] = dict(member.get("member_text") or {})
    for rel in store.list_relationships(account_id, active_only=True):
        source = entities.get(rel.get("from_entity"))
        if source is None:
            continue
        for policy in policies.values():
            if rel.get("to_entity") not in policy["entities"]:
                continue
            if str(rel.get("to_column") or "").upper() != str(policy["key_column"]).upper():
                continue
            reference = {"table": _qualified(source), "column": str(rel.get("from_column") or "")}
            if reference["column"] and reference not in policy["references"]:
                policy["references"].append(reference)
    return [policies[name] for name in sorted(policies)]


def attach_unknown_member_policies(
    semantic_plan: dict | None, account_id: str, *questions: str,
) -> list[dict]:
    """Put the policies on the plan the prompt, the validator, the compiler
    and the answer card all read -- unless the question asks about them."""
    if not isinstance(semantic_plan, dict) or question_asks_for_unknown_members(*questions):
        return []
    policies = unknown_member_policies(account_id)
    if policies:
        semantic_plan["unknown_member_policies"] = policies
    return policies


def _literal(key: str) -> str:
    return key if re.fullmatch(r"-?\d+", str(key)) else "'" + str(key).replace("'", "''") + "'"


def exclusion_predicate(column_ref: str, policy: dict) -> str:
    """The predicate that leaves a policy's members out, on `column_ref`."""
    return f"{column_ref} NOT IN ({', '.join(_literal(k) for k in policy.get('keys') or [])})"


def policies_in_scope(policies: list[dict] | None, *texts: str) -> list[dict]:
    """The policies whose dimension table is named anywhere in ``texts``."""
    haystack = " ".join(str(text or "") for text in texts).upper()
    return [
        policy for policy in policies or []
        if isinstance(policy, dict) and _bare(str(policy.get("table") or ""))
        and re.search(rf"(?<![A-Z0-9_]){re.escape(_bare(str(policy['table'])))}(?![A-Z0-9_])", haystack)
    ]


def format_unknown_member_rules(policies: list[dict] | None) -> str:
    """The prompt block that states the rule for each dimension in scope."""
    lines: list[str] = []
    for policy in policies or []:
        key = str(policy.get("key_column") or "")
        keys = [str(k) for k in policy.get("keys") or []]
        if not key or not keys:
            continue
        described = ", ".join(
            f"{k} ({str(policy['member_kinds'].get(k) or UNKNOWN).replace('_', ' ')})" for k in keys
        )
        refs = ", ".join(f"{_bare(r['table'])}.{r['column']}" for r in policy.get("references") or [])
        lines.append(
            f"- {policy['table']}: {key} {described} -- "
            f"WHERE {exclusion_predicate('<alias>.' + key, policy)}"
            + (f" (or the same on the key that points at it: {refs})" if refs else "")
        )
    if not lines:
        return ""
    return "\n".join([
        "## Unknown members — REQUIRED",
        "These dimension rows are placeholders a fact points at when its value was empty or "
        "matched nothing; they are not members. A query that ranks members (TOP/LIMIT/FETCH "
        "with ORDER BY, or RANK/ROW_NUMBER) or counts them (COUNT(DISTINCT ...), or COUNT of "
        "the dimension table) MUST leave them out with the predicate shown, using your alias "
        "for the table. A listing or a total keeps them.",
        *lines,
    ])


# ── Reading a query for rankings and counts of members ──────────────────────
#
# Whether a predicate leaves a member out is decided by evaluating it with the
# member's key (or its text) in place of the column, not by its spelling:
# NOT IN (0, 777), <> 0 AND <> 777, BETWEEN 1 AND 776, and the period-row
# rule's key % 100 BETWEEN 1 AND 12 all leave 0 and 777 out.

_CANNOT = object()  # a value this reading cannot know


def _value(node, bind):
    from sqlglot import exp

    if isinstance(node, exp.Paren):
        return _value(node.this, bind)
    if isinstance(node, exp.Column):
        return bind(node)
    if isinstance(node, exp.Literal):
        if node.is_string:
            return str(node.this)
        try:
            number = float(node.this)
        except ValueError:
            return _CANNOT
        return int(number) if number.is_integer() else number
    if isinstance(node, exp.Null):
        return None
    if isinstance(node, (exp.Cast, exp.TryCast)):
        return _value(node.this, bind)
    if isinstance(node, exp.Neg):
        value = _value(node.this, bind)
        return -value if isinstance(value, (int, float)) else (value if value is None else _CANNOT)
    if isinstance(node, exp.Coalesce):
        for part in [node.this, *node.expressions]:
            value = _value(part, bind)
            if value is _CANNOT or value is not None:
                return value
        return None
    if isinstance(node, (exp.Mod, exp.Add, exp.Sub, exp.Mul)):
        left, right = _value(node.this, bind), _value(node.expression, bind)
        if left is _CANNOT or right is _CANNOT:
            return _CANNOT
        if left is None or right is None:
            return None
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            return _CANNOT
        if isinstance(node, exp.Mod):
            # SQL's remainder takes the sign of the dividend: -1 % 100 is -1.
            return _CANNOT if right == 0 else int(math.fmod(left, right))
        if isinstance(node, exp.Add):
            return left + right
        return left - right if isinstance(node, exp.Sub) else left * right
    return _CANNOT


def _same(left, right):
    if isinstance(left, str) and isinstance(right, str):
        return left.casefold() == right.casefold()
    if isinstance(left, str) or isinstance(right, str):
        try:
            return float(str(left)) == float(str(right))
        except ValueError:
            return False
    return left == right


def _truth(node, bind):
    """True, False, None (SQL NULL) or _CANNOT."""
    from sqlglot import exp

    if isinstance(node, exp.Paren):
        return _truth(node.this, bind)
    if isinstance(node, (exp.And, exp.Or)):
        left, right = _truth(node.this, bind), _truth(node.expression, bind)
        decisive, other = (False, True) if isinstance(node, exp.And) else (True, False)
        if decisive in (left, right) and not (left is _CANNOT and right is _CANNOT):
            if left is decisive or right is decisive:
                return decisive
        if left is _CANNOT or right is _CANNOT:
            return _CANNOT
        if left is None or right is None:
            return None
        return other
    if isinstance(node, exp.Not):
        inner = _truth(node.this, bind)
        return inner if inner is None or inner is _CANNOT else not inner
    if isinstance(node, exp.Is):
        value = _value(node.this, bind)
        if value is _CANNOT or not isinstance(node.expression, exp.Null):
            return _CANNOT
        return value is None
    if isinstance(node, (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)):
        left, right = _value(node.this, bind), _value(node.expression, bind)
        if left is _CANNOT or right is _CANNOT:
            return _CANNOT
        if left is None or right is None:
            return None
        if isinstance(node, (exp.EQ, exp.NEQ)):
            equal = _same(left, right)
            return equal if isinstance(node, exp.EQ) else not equal
        if isinstance(left, str) or isinstance(right, str):
            return _CANNOT
        return {
            exp.GT: left > right, exp.GTE: left >= right, exp.LT: left < right, exp.LTE: left <= right,
        }[type(node)]
    if isinstance(node, exp.Between):
        value = _value(node.this, bind)
        low, high = _value(node.args.get("low"), bind), _value(node.args.get("high"), bind)
        if _CANNOT in (value, low, high) or any(isinstance(v, str) for v in (value, low, high)):
            return _CANNOT
        if value is None or low is None or high is None:
            return None
        return low <= value <= high
    if isinstance(node, exp.In):
        if node.args.get("query") is not None:
            return _CANNOT
        value = _value(node.this, bind)
        items = [_value(item, bind) for item in node.expressions]
        if value is _CANNOT or _CANNOT in items:
            return _CANNOT
        if value is None:
            return None
        if any(item is not None and _same(value, item) for item in items):
            return True
        return None if None in items else False
    return _CANNOT


def _conjuncts(node) -> list:
    from sqlglot import exp

    if node is None:
        return []
    if isinstance(node, exp.Paren):
        return _conjuncts(node.this)
    if isinstance(node, exp.And):
        return _conjuncts(node.this) + _conjuncts(node.expression)
    return [node]


def _leaves_out(conjuncts: list, bind) -> bool:
    """Whether some conjunct that reads only what `bind` knows is FALSE or
    NULL for it -- so WHERE drops the row."""
    from sqlglot import exp

    for conjunct in conjuncts:
        columns = list(conjunct.find_all(exp.Column))
        if not columns or any(bind(column) is _CANNOT for column in columns):
            continue
        if _truth(conjunct, bind) in (False, None):
            return True
    return False


def _same_table(node, name: str) -> bool:
    wanted = [part.strip('[]"`').upper() for part in str(name or "").split(".") if part]
    if not wanted or str(node.name or "").upper() != wanted[-1]:
        return False
    return not (len(wanted) > 1 and node.db and str(node.db).upper() != wanted[-2])


def _own(select, kind) -> list:
    return [node for node in select.find_all(kind) if node.find_ancestor(type(select)) is select]


def _sources(select) -> list[tuple]:
    """(table, ON condition, is an inner join) for each table the SELECT reads."""
    from sqlglot import exp

    found = []
    source = select.args.get("from_") or select.args.get("from")
    if source is not None and isinstance(source.this, exp.Table):
        found.append((source.this, None, True))
    for join in select.args.get("joins") or []:
        if isinstance(join.this, exp.Table):
            inner = not join.args.get("side") and str(join.args.get("kind") or "").upper() in {"", "INNER"}
            found.append((join.this, join.args.get("on"), inner))
    return found


def _feeders(select, ctes: dict) -> list:
    """The SELECTs whose rows this one reads: its derived tables and CTEs."""
    from sqlglot import exp

    fed: list[Any] = []
    source = select.args.get("from_") or select.args.get("from")
    nodes = ([source.this] if source is not None else []) + [
        join.this for join in select.args.get("joins") or []
    ]
    for node in nodes:
        if isinstance(node, exp.Subquery):
            fed.extend(node.this.find_all(exp.Select) if isinstance(node.this, exp.Union) else [node.this])
        elif isinstance(node, exp.Table) and not node.db and str(node.name).upper() in ctes:
            fed.append(ctes[str(node.name).upper()])
    return [s for s in fed if isinstance(s, exp.Select)]


def _ranks(select) -> bool:
    from sqlglot import exp

    if select.args.get("order") is not None and select.args.get("limit") is not None:
        return True
    ranking = (exp.RowNumber, exp.Rank, exp.DenseRank, exp.PercentRank, exp.CumeDist, exp.Ntile)
    return any(isinstance(window.this, ranking) for window in _own(select, exp.Window))


def _grouped_columns(select) -> list:
    """The columns a SELECT groups by -- or, ungrouped, the ones it returns."""
    from sqlglot import exp

    projection = list(select.expressions or [])
    by_alias = {str(e.alias).upper(): e.this for e in projection if isinstance(e, exp.Alias)}
    group = select.args.get("group")
    if group is None:
        targets = [e for e in projection if not list(e.find_all(exp.AggFunc))]
    else:
        targets = []
        for item in group.expressions:
            if isinstance(item, exp.Literal) and not item.is_string and str(item.this).isdigit():
                position = int(item.this) - 1
                if 0 <= position < len(projection):
                    targets.append(projection[position])
            elif isinstance(item, exp.Column) and not item.table and str(item.name).upper() in by_alias:
                targets.append(by_alias[str(item.name).upper()])
            else:
                targets.append(item)
    return [column for target in targets for column in target.find_all(exp.Column)]


def member_scopes(sql, policies: list[dict] | None, db_type: str = "azure_sql") -> list[dict]:
    """Each place a query ranks or counts a policy's members, and whether the
    members are left out there.

    Returns ``[{"policy", "how": "ranking"|"count", "required", "excluded"}]``.
    """
    import sqlglot
    from sqlglot import exp

    from core.validator import _DIALECT, normalize_generated_sql

    governed = [p for p in policies or [] if isinstance(p, dict) and p.get("keys") and p.get("table")]
    if not governed:
        return []
    tree = sql
    if isinstance(sql, str):
        try:
            tree = sqlglot.parse_one(
                normalize_generated_sql(sql, db_type), read=_DIALECT.get(db_type, "snowflake"),
            )
        except Exception:
            return []
    ctes = {str(cte.alias).upper(): cte.this for cte in tree.find_all(exp.CTE)}

    # Which SELECTs rank (and every SELECT that feeds one), and which count.
    ranked: dict[int, tuple] = {}
    frontier: list[tuple[Any, tuple]] = [
        (select, ()) for select in tree.find_all(exp.Select) if _ranks(select)
    ]
    while frontier:
        select, above = frontier.pop()
        if id(select) in ranked:
            continue
        ranked[id(select)] = (select, above)
        frontier.extend((fed, (*above, select)) for fed in _feeders(select, ctes))
    counted: dict[int, tuple] = {}
    for select in tree.find_all(exp.Select):
        counts = _own(select, exp.Count)
        if any(isinstance(count.this, exp.Distinct) for count in counts):
            counted[id(select)] = (select, (), "distinct")
        elif counts and select.args.get("group") is None:
            counted[id(select)] = (select, (), "members")
        if counts:
            for fed in _feeders(select, ctes):
                if fed.args.get("distinct") is not None:
                    counted[id(fed)] = (fed, (select,), "rows")

    scopes: list[dict] = []
    for how, entries in (("ranking", [(s, a, "") for s, a in ranked.values()]), ("count", list(counted.values()))):
        for select, above, mode in entries:
            sources = _sources(select)
            for policy in governed:
                scope = _scope(select, above, sources, policy, how, mode)
                if scope:
                    scopes.append(scope)
    return scopes


def _scope(select, above, sources, policy: dict, how: str, mode: str) -> dict | None:
    from sqlglot import exp

    key = str(policy["key_column"]).upper()
    # Upper-cased to match, spelled as the query wrote them to name.
    dimension = {
        str(table.alias_or_name).upper(): str(table.alias_or_name)
        for table, _on, _inner in sources if _same_table(table, policy["table"])
    }
    pointers = {
        (str(table.alias_or_name).upper(), str(ref["column"]).upper()):
            f"{table.alias_or_name}.{ref['column']}"
        for table, _on, _inner in sources
        for ref in policy.get("references") or []
        if _same_table(table, ref["table"])
    }
    alone = len(sources) == 1 and bool(dimension)

    def role(column) -> str:
        qualifier, name = str(column.table or "").upper(), str(column.name or "").upper()
        if qualifier:
            if qualifier in dimension:
                return "key" if name == key else "attribute"
            return "key" if (qualifier, name) in pointers else ""
        if alone:
            return "key" if name == key else "attribute"
        if (name == key and dimension) or any(name == column_name for _, column_name in pointers):
            return "key"
        return ""

    if how == "ranking":
        involved = any(role(column) for column in _grouped_columns(select))
    elif mode == "distinct":
        involved = any(
            role(column)
            for count in _own(select, exp.Count) if isinstance(count.this, exp.Distinct)
            for column in count.this.find_all(exp.Column)
        )
    elif mode == "members":
        involved = alone
    else:
        involved = any(role(column) for column in _grouped_columns(select))
    if not involved:
        return None

    conjuncts = _conjuncts((select.args.get("where") or exp.Where()).this)
    for table, on, inner in sources:
        if inner and on is not None:
            conjuncts += _conjuncts(on)
    outer = [c for parent in above for c in _conjuncts((parent.args.get("where") or exp.Where()).this)]
    names = {key} | {column_name for _, column_name in pointers} | {
        str(ref["column"]).upper() for ref in policy.get("references") or []
    }

    def excluded(member: str) -> bool:
        number = int(member) if re.fullmatch(r"-?\d+", member) else member
        text = {str(k).upper(): v for k, v in (policy["member_text"].get(member) or {}).items()}

        def bind(column):
            kind = role(column)
            if kind == "key":
                return number
            if kind == "attribute":
                return text.get(str(column.name).upper(), _CANNOT)
            return _CANNOT

        def bind_outer(column):
            name = str(column.name).upper()
            return number if name in names else text.get(name, _CANNOT)

        return _leaves_out(conjuncts, bind) or _leaves_out(outer, bind_outer)

    if dimension:
        column_ref = f"{dimension[sorted(dimension)[0]]}.{policy['key_column']}"
    elif pointers:
        column_ref = pointers[sorted(pointers)[0]]
    else:
        column_ref = str(policy["key_column"])
    return {
        "policy": policy,
        "how": how,
        "required": exclusion_predicate(column_ref, policy),
        "excluded": all(excluded(str(member)) for member in policy["keys"]),
    }
