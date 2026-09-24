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
import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path

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
