"""
core/graph_health.py

Entity-graph health checker and schema-drift detector for the Semantic Layer.

Checks performed
────────────────
  1. ENTITY_NO_TABLE          — entity has no table_name set
  2. TABLE_MISSING_IN_SCHEMA  — entity table not found in discovered _schema.json (drift)
  3. PK_MISSING_IN_SCHEMA     — pk_column not found in schema columns for that table
  4. ORPHANED_RELATIONSHIP    — relationship from/to entity not in the graph
  5. DISCONNECTED_ENTITY      — entity has no relationships (isolated node)
  6. DUPLICATE_TABLE_MAPPING  — two active entities map to the same table
  7. ENTITY_NO_PROPERTIES     — entity has no properties defined
  8. UNMAPPED_TABLE           — table in schema has no entity pointing to it (gap/info)

Score formula
─────────────
  Start: 100
  Each error:   −10  (min 0)
  Each warning: −3

Integration points
──────────────────
  admin/routes.py — GET /clients/{account_id}/graph/api/health
  client_graph.html — health badge + issue panel + unmapped-table quick-add
"""

from __future__ import annotations

import json as _json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from core.join_governance import check_join

log = logging.getLogger("querybot.graph_health")

# ══════════════════════════════════════════════════════════════════════════════
# Data types
# ══════════════════════════════════════════════════════════════════════════════

SEVERITY_ERROR   = "error"
SEVERITY_WARNING = "warning"
SEVERITY_INFO    = "info"


@dataclass
class HealthIssue:
    severity: str   # "error" | "warning" | "info"
    code: str       # e.g. "TABLE_MISSING_IN_SCHEMA"
    entity: str     # entity name this applies to, "" for graph-level issues
    message: str

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "code":     self.code,
            "entity":   self.entity,
            "message":  self.message,
        }


@dataclass
class HealthReport:
    score: int                        = 100
    entity_count: int                 = 0
    relationship_count: int           = 0
    property_count: int               = 0
    has_schema: bool                  = False
    issues: list[HealthIssue]         = field(default_factory=list)
    unmapped_tables: list[dict]       = field(default_factory=list)
    # per-entity issue index for fast UI lookup: {entity_name: ["error"|"warning"]}
    entity_severity: dict[str, str]   = field(default_factory=dict)

    def add_error(self, code: str, entity: str, msg: str) -> None:
        self.issues.append(HealthIssue(SEVERITY_ERROR, code, entity, msg))

    def add_warning(self, code: str, entity: str, msg: str) -> None:
        self.issues.append(HealthIssue(SEVERITY_WARNING, code, entity, msg))

    def add_info(self, code: str, entity: str, msg: str) -> None:
        self.issues.append(HealthIssue(SEVERITY_INFO, code, entity, msg))

    def _compute_score(self) -> None:
        errors   = sum(1 for i in self.issues if i.severity == SEVERITY_ERROR)
        warnings = sum(1 for i in self.issues if i.severity == SEVERITY_WARNING)
        self.score = max(0, 100 - errors * 10 - warnings * 3)

    def _build_entity_severity(self) -> None:
        """Build a {entity_name: worst_severity} index for the UI."""
        order = {SEVERITY_ERROR: 0, SEVERITY_WARNING: 1, SEVERITY_INFO: 2}
        result: dict[str, str] = {}
        for issue in self.issues:
            if not issue.entity:
                continue
            current = result.get(issue.entity)
            if current is None or order[issue.severity] < order[current]:
                result[issue.entity] = issue.severity
        self.entity_severity = result

    def finalise(self) -> None:
        self._compute_score()
        self._build_entity_severity()

    def to_dict(self) -> dict:
        return {
            "score":              self.score,
            "entity_count":       self.entity_count,
            "relationship_count": self.relationship_count,
            "property_count":     self.property_count,
            "has_schema":         self.has_schema,
            "issues":             [i.to_dict() for i in self.issues],
            "unmapped_tables":    self.unmapped_tables,
            "entity_severity":    self.entity_severity,
            "error_count":        sum(1 for i in self.issues if i.severity == SEVERITY_ERROR),
            "warning_count":      sum(1 for i in self.issues if i.severity == SEVERITY_WARNING),
            "info_count":         sum(1 for i in self.issues if i.severity == SEVERITY_INFO),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def check_graph_health(account_id: str) -> HealthReport:
    """
    Run all health checks for an account's entity graph.

    Loads entities, relationships, and properties from the store, then
    cross-references them against the discovered _schema.json.

    Returns a HealthReport with score, issues, and unmapped_tables.
    """
    import store

    report = HealthReport()

    # ── Load graph data ───────────────────────────────────────────────────────
    entities      = store.list_entities(account_id, active_only=False)
    relationships = store.list_relationships(account_id, active_only=False)

    report.entity_count       = len(entities)
    report.relationship_count = len(relationships)

    # Load properties for all entities
    all_properties: list[dict] = []
    for ent in entities:
        props = store.list_entity_properties(account_id, ent["entity_name"])
        report.property_count += len(props)
        all_properties.extend(props)

    # ── Load schema ───────────────────────────────────────────────────────────
    schema_columns = _load_schema(account_id)
    report.has_schema = schema_columns is not None

    # Normalised schema: lower-case FQN → {columns: set[str], original_fqn: str}
    schema_map: dict[str, dict] = {}
    if schema_columns:
        for fqn, cols in schema_columns.items():
            schema_map[fqn.lower()] = {
                "columns":      {c.lower() for c in cols},
                "original_fqn": fqn,
            }

    # ── Build lookup sets ─────────────────────────────────────────────────────
    entity_names = {e["entity_name"] for e in entities}
    active_entities = {e["entity_name"] for e in entities if e.get("is_active", 1)}

    # ── Check 1 & 2: entity table presence + schema drift ────────────────────
    table_to_entities: dict[str, list[str]] = {}  # fqn_lower → [entity_name]

    for ent in entities:
        ename = ent["entity_name"]
        tname = (ent.get("table_name") or "").strip()
        sname = (ent.get("schema_name") or "").strip()

        if not tname:
            report.add_error(
                "ENTITY_NO_TABLE", ename,
                f"Entity '{ename}' has no table mapped. "
                "Set a base table so the resolver can generate SQL."
            )
            continue

        # Build FQN for schema lookup
        fqn_lower = _resolve_fqn(sname, tname, schema_map)

        if fqn_lower is not None:
            table_to_entities.setdefault(fqn_lower, []).append(ename)
        elif report.has_schema:
            # Table in entity but not in schema → drift
            display_fqn = f"{sname}.{tname}" if sname else tname
            report.add_error(
                "TABLE_MISSING_IN_SCHEMA", ename,
                f"Entity '{ename}' maps to table '{display_fqn}' which was not found "
                "in the discovered schema. The table may have been renamed or dropped."
            )
        # else: no schema loaded, skip drift check

        # ── Check 3: pk_column in schema ──────────────────────────────────────
        pk = (ent.get("pk_column") or "").strip()
        if pk and fqn_lower and report.has_schema:
            tbl_info = schema_map.get(fqn_lower)
            if tbl_info and pk.lower() not in tbl_info["columns"]:
                report.add_warning(
                    "PK_MISSING_IN_SCHEMA", ename,
                    f"Entity '{ename}' pk_column '{pk}' was not found in "
                    f"the schema for its table. Check spelling or run schema discovery."
                )

    # ── Check 6: duplicate table mappings ────────────────────────────────────
    for fqn_lower, enames in table_to_entities.items():
        active_mappers = [n for n in enames if n in active_entities]
        if len(active_mappers) > 1:
            report.add_warning(
                "DUPLICATE_TABLE_MAPPING", active_mappers[0],
                f"Table '{fqn_lower}' is mapped by multiple active entities: "
                + ", ".join(f"'{n}'" for n in active_mappers)
                + ". Only one entity should own a table."
            )

    # ── Check 4: orphaned relationships ──────────────────────────────────────
    for rel in relationships:
        from_e = rel.get("from_entity", "")
        to_e   = rel.get("to_entity", "")
        if from_e not in entity_names:
            report.add_error(
                "ORPHANED_RELATIONSHIP", "",
                f"Relationship references unknown entity '{from_e}'. "
                "The entity may have been deleted."
            )
        if to_e not in entity_names:
            report.add_error(
                "ORPHANED_RELATIONSHIP", "",
                f"Relationship references unknown entity '{to_e}'. "
                "The entity may have been deleted."
            )

    # ── Check 4b: joins that can never be used ───────────────────────────────
    #
    # Every check above this asks whether the graph refers to things that
    # exist. None of them asked whether a JOIN could be traversed -- so a
    # workspace whose only fact tables were wired directly to each other, on a
    # column neither of them has, scored 94/100 and was told its entities had
    # no field properties.
    #
    # The verdict is core.join_governance's, which is core.join_planner's:
    # one vocabulary, so a health warning and a query-time refusal say the
    # same thing about the same edge. J1 refuses these at the moment of
    # saving; this is how the ones saved before it get found.
    entity_by_name = {e["entity_name"]: dict(e) for e in entities}
    schema_for_check = schema_columns if schema_columns else None
    for rel in relationships:
        if not rel.get("is_active", 1):
            continue
        from_e = str(rel.get("from_entity") or "")
        to_e = str(rel.get("to_entity") or "")
        if from_e not in entity_names or to_e not in entity_names:
            continue    # Check 4 already reported this one
        try:
            verdict = check_join(dict(rel), entity_by_name, schema_for_check)
        except Exception as exc:  # noqa: BLE001
            # A health report that dies on one bad row tells an admin nothing
            # about the other two hundred.
            log.warning("Join health check failed for %s→%s: %s", from_e, to_e, exc)
            continue
        edge = f"{from_e} → {to_e}"
        if verdict.refuses:
            report.add_error(
                "RELATIONSHIP_UNUSABLE", from_e,
                # The verdict already opens with "This join cannot be used";
                # naming the edge is all this has to add.
                f"{edge}: {verdict.reason}")
        elif verdict.code == "column_missing":
            report.add_error(
                "RELATIONSHIP_COLUMN_MISSING", from_e,
                f"Join {edge} names columns that are not in the discovered "
                f"schema: {', '.join(verdict.missing_columns[:4])}.")

        # Recorded by the validator's live probe, and only by it. A join
        # nobody has probed is not broken -- it is unmeasured, which is a
        # different thing to tell somebody and a different thing to fix.
        validation = str(rel.get("validation_status") or "untested")
        if validation == "broken":
            report.add_error(
                "RELATIONSHIP_VALIDATION_BROKEN", from_e,
                f"Join {edge} failed validation the last time it was checked.")
        elif validation == "warning":
            report.add_warning(
                "RELATIONSHIP_VALIDATION_WARNING", from_e,
                f"Join {edge} matched fewer rows than expected when it was "
                f"last validated.")

    # ── Check 5: disconnected entities ───────────────────────────────────────
    connected: set[str] = set()
    for rel in relationships:
        connected.add(rel.get("from_entity", ""))
        connected.add(rel.get("to_entity", ""))

    for ent in entities:
        if not ent.get("is_active", 1):
            continue
        ename = ent["entity_name"]
        if ename not in connected and len(active_entities) > 1:
            report.add_warning(
                "DISCONNECTED_ENTITY", ename,
                f"Entity '{ename}' has no relationships. "
                "Add joins so the resolver can navigate to related tables."
            )

    # ── Check 7: no properties ────────────────────────────────────────────────
    entities_with_props = {p["entity_name"] for p in all_properties}
    for ent in entities:
        if not ent.get("is_active", 1):
            continue
        ename = ent["entity_name"]
        if ename not in entities_with_props:
            report.add_warning(
                "ENTITY_NO_PROPERTIES", ename,
                f"Entity '{ename}' has no field properties defined. "
                "Add properties so the resolver knows which columns are metrics, "
                "dimensions, and dates."
            )

    # ── Check 8: unmapped tables ──────────────────────────────────────────────
    if report.has_schema:
        mapped_fqns = set(table_to_entities.keys())
        for fqn_lower, info in schema_map.items():
            if fqn_lower not in mapped_fqns:
                parts = info["original_fqn"].split(".")
                report.unmapped_tables.append({
                    "fqn":        info["original_fqn"],
                    "table_name": parts[-1],
                    "schema_name": parts[-2] if len(parts) >= 2 else "",
                })
        report.unmapped_tables.sort(key=lambda t: (t["schema_name"], t["table_name"]))

    report.finalise()
    return report


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_schema(account_id: str) -> Optional[dict[str, list[str]]]:
    """Load discovered schema column map for an account."""
    try:
        import store
        state      = store.get_client_state(account_id)
        schema_dir = (state or {}).get("schema_dir") or ""
        if not schema_dir:
            return None
        schema_path = Path(schema_dir) / "_schema.json"
        if not schema_path.exists():
            return None
        master = _json.loads(schema_path.read_text(encoding="utf-8"))
        result: dict[str, list[str]] = {}
        for fqn, tbl_data in master.items():
            if isinstance(tbl_data, dict):
                cols = [
                    c.get("name", "") if isinstance(c, dict) else str(c)
                    for c in (tbl_data.get("columns") or [])
                ]
                result[fqn] = [c for c in cols if c]
        return result or None
    except Exception as exc:
        log.debug("_load_schema failed for %s: %s", account_id, exc)
        return None


def _resolve_fqn(
    schema_name: str,
    table_name: str,
    schema_map: dict[str, dict],
) -> Optional[str]:
    """
    Find the lower-case FQN key in schema_map that best matches
    the given schema_name + table_name.

    Tries in order:
      1. Exact full lower-case match (schema.table)
      2. Case-insensitive suffix match on table name alone
    """
    if not table_name:
        return None

    # Exact match
    candidates = []
    if schema_name:
        candidate = f"{schema_name}.{table_name}".lower()
        if candidate in schema_map:
            return candidate
        # Try with the schema prepended to existing FQN suffixes
    table_lower = table_name.lower()

    for fqn_lower in schema_map:
        parts = fqn_lower.split(".")
        if parts[-1] == table_lower:
            if schema_name and len(parts) >= 2 and parts[-2] == schema_name.lower():
                return fqn_lower          # full match
            candidates.append(fqn_lower)  # partial (table name only)

    return candidates[0] if len(candidates) == 1 else (candidates[0] if candidates else None)
