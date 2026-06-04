"""
Structured semantic model artifacts for QueryBot.

The markdown KB is useful for humans and retrieval.  This module writes a
machine-readable model beside it so future SQL planning can rely on explicit
fields, display preferences, date roles, filters, and relationship roles.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from core.date_roles import detect_date_role, find_date_dimension_key, is_date_dimension_table
from core.naming_convention import match_column_suffix, match_entity_prefix, match_table_suffix
from core.schema_enrichment import EnrichedColumn, enrich_columns

log = logging.getLogger(__name__)

# Keys copied from an approved field/dimension/measure/date-role onto the freshly
# generated entry during a KB rebuild so admin approvals are never lost.
_APPROVED_FIELD_KEYS = frozenset(
    {"approved_meaning", "approved_use_case", "status", "confidence"}
)
_APPROVED_DIMENSION_KEYS = frozenset({"status", "confidence", "approved_meaning"})
_APPROVED_MEASURE_KEYS = frozenset({"status", "confidence", "expression"})
_APPROVED_DATE_ROLE_KEYS = frozenset({"status", "confidence"})


MODEL_JSON = "_semantic_model.json"
MODEL_YAML = "_semantic_model.yaml"


def _read_schema(schema_dir: str) -> dict[str, Any]:
    schema_path = Path(schema_dir) / "_schema.json"
    if not schema_path.exists():
        return {}
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _schema_table_name(fqn: str, meta: dict[str, Any]) -> str:
    return str(meta.get("table") or fqn.split(".")[-1])


def _schema_name(fqn: str, meta: dict[str, Any]) -> str:
    schema = meta.get("schema")
    if schema:
        return str(schema)
    parts = fqn.split(".")
    return parts[-2] if len(parts) >= 2 else ""


def _database_name(fqn: str, meta: dict[str, Any]) -> str:
    db = meta.get("database")
    if db:
        return str(db)
    parts = fqn.split(".")
    return parts[-3] if len(parts) >= 3 else ""


def _qualified_name(fqn: str, meta: dict[str, Any]) -> str:
    schema = _schema_name(fqn, meta)
    table = _schema_table_name(fqn, meta)
    return f"{schema}.{table}" if schema else table


def _column_names(meta: dict[str, Any]) -> list[str]:
    return [str(c.get("name") or "") for c in meta.get("columns", []) if c.get("name")]


def _field_type(meta: dict[str, Any], column: str) -> str:
    for c in meta.get("columns", []) or []:
        if str(c.get("name") or "").upper() == column.upper():
            return str(c.get("type") or "")
    return ""


def _table_type(table: str) -> str:
    rule = match_table_suffix(table)
    if rule:
        if rule.table_type == "fact_table":
            return "fact"
        if rule.table_type == "dimension_table":
            return "dimension"
        if "bridge" in rule.table_type:
            return "bridge"
    upper = table.upper()
    if upper.endswith("_FCT") or "_FCT" in upper or upper.startswith(("FACT_", "FCT_")):
        return "fact"
    if upper.endswith("_DMS") or upper.startswith(("DIM_", "DMS_")):
        return "dimension"
    return "dimension"


def _entity_name(table: str) -> str:
    bare = table.split(".")[-1]
    for suffix in ("_FCT", "_DMS", "_DIM"):
        if bare.upper().endswith(suffix):
            bare = bare[: -len(suffix)]
            break
    for prefix in ("FACT_", "FCT_", "DIM_"):
        if bare.upper().startswith(prefix):
            bare = bare[len(prefix):]
            break
    return " ".join(part.capitalize() for part in bare.split("_") if part)


def _display_field_for_columns(columns: list[str], prefix: str = "") -> str:
    upper_to_raw = {c.upper(): c for c in columns}
    prefixes = [prefix.upper()] if prefix else []
    if prefix and prefix.upper().endswith("_DMS_KEY"):
        prefixes.append(prefix.upper()[: -len("_DMS_KEY")])
    if prefix and "_" in prefix:
        prefixes.append(prefix.upper().split("_")[0])
    prefixes = [p for p in dict.fromkeys(prefixes) if p]

    candidates: list[str] = []
    for p in prefixes:
        candidates.extend([f"{p}_DSC", f"{p}_DESC", f"{p}_DESCRIPTION", f"{p}_NM", f"{p}_NAME"])
    candidates.extend([c for c in upper_to_raw if c.endswith(("_DSC", "_DESC", "_DESCRIPTION", "_NM", "_NAME"))])
    for candidate in candidates:
        if candidate in upper_to_raw:
            return upper_to_raw[candidate]
    for c in columns:
        rule = match_column_suffix(c)
        if rule and rule.role == "display":
            return c
    return ""


def _code_field_for_columns(columns: list[str], prefix: str = "") -> str:
    upper_to_raw = {c.upper(): c for c in columns}
    prefixes = [prefix.upper()] if prefix else []
    if prefix and prefix.upper().endswith("_DMS_KEY"):
        prefixes.append(prefix.upper()[: -len("_DMS_KEY")])
    prefixes = [p for p in dict.fromkeys(prefixes) if p]
    for p in prefixes:
        for suffix in ("_CD", "_CODE"):
            candidate = f"{p}{suffix}"
            if candidate in upper_to_raw:
                return upper_to_raw[candidate]
    for c in columns:
        rule = match_column_suffix(c)
        if rule and rule.role == "code":
            return c
    return ""


def _business_role_from_column(column: str) -> str:
    role = detect_date_role(column)
    if role:
        return role.key
    entity = match_entity_prefix(column)
    if entity:
        return re.sub(r"[^a-z0-9]+", "_", entity.lower()).strip("_")
    col = column.upper()
    if col.endswith("_DMS_KEY"):
        col = col[: -len("_DMS_KEY")]
    for suffix in ("_KEY", "_ID", "_NUM", "_NO", "_CD", "_CODE"):
        if col.endswith(suffix):
            col = col[: -len(suffix)]
    return "_".join(t.lower() for t in re.split(r"[_\W]+", col) if t)


def _relationship_id(from_table: str, to_table: str, role: str, from_col: str, to_col: str) -> str:
    raw = f"{from_table}_{to_table}_{role}_{from_col}_{to_col}".lower()
    return re.sub(r"[^a-z0-9]+", "_", raw).strip("_")


def _table_lookup(schema: dict[str, Any]) -> dict[str, tuple[str, dict[str, Any]]]:
    lookup: dict[str, tuple[str, dict[str, Any]]] = {}
    for fqn, meta in schema.items():
        table = _schema_table_name(fqn, meta)
        qname = _qualified_name(fqn, meta)
        for key in {fqn.upper(), table.upper(), qname.upper()}:
            lookup[key] = (fqn, meta)
    return lookup


def _find_dimension_for_key(schema: dict[str, Any], source_key: str) -> tuple[str, dict[str, Any]] | None:
    source_upper = source_key.upper()
    source_prefix = source_upper[: -len("_DMS_KEY")] if source_upper.endswith("_DMS_KEY") else source_upper
    best: tuple[str, dict[str, Any]] | None = None
    for fqn, meta in schema.items():
        table = _schema_table_name(fqn, meta)
        if _table_type(table) != "dimension":
            continue
        cols = {c.upper() for c in _column_names(meta)}
        if source_upper in cols:
            return fqn, meta
        table_upper = table.upper()
        if table_upper == f"{source_prefix}_DMS" or table_upper.endswith(f".{source_prefix}_DMS"):
            best = (fqn, meta)
    return best


def _field_entry(item: EnrichedColumn, meta: dict[str, Any]) -> dict[str, Any]:
    rule = match_column_suffix(item.column)
    entity_prefix = match_entity_prefix(item.column) or ""
    return {
        "column": item.column,
        "data_type": item.data_type or _field_type(meta, item.column),
        "nullable": item.nullable,
        "role": item.role,
        "expanded_name": item.expanded_name,
        "business_candidates": item.business_candidates,
        "confidence": item.confidence,
        "evidence": item.evidence,
        "warnings": item.warnings,
        "default_filter": item.default_filter,
        "join_equivalents": item.join_equivalents,
        "date_role": item.date_role,
        "naming_role": rule.role if rule else "",
        "aggregation": rule.aggregation if rule else "",
        "format_hint": rule.format_hint if rule else "",
        "entity_prefix": entity_prefix,
        "status": "generated" if item.confidence >= 70 else "needs_review",
    }


def _default_filters(fields: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = []
    for field in fields:
        if not field.get("default_filter"):
            continue
        filters.append({
            "column": field["column"],
            "expression": field["default_filter"],
            "meaning": field.get("expanded_name", ""),
            "status": field.get("status", "generated"),
        })
    return filters


def _measure_candidates(fields: list[dict[str, Any]]) -> list[dict[str, Any]]:
    measures: list[dict[str, Any]] = []
    for field in fields:
        if field.get("role") not in {"measure", "measure_candidate"}:
            continue
        measures.append({
            "name": (field.get("business_candidates") or [field.get("expanded_name") or field["column"]])[0],
            "column": field["column"],
            "expression": f"SUM({field['column']})",
            "aggregation": field.get("aggregation") or "additive",
            "format": field.get("format_hint") or "number",
            "synonyms": field.get("business_candidates", []),
            "status": "suggested",
            "confidence": field.get("confidence", 50),
        })
    return measures


def _dimension_candidates(
    *,
    schema: dict[str, Any],
    table_fqn: str,
    meta: dict[str, Any],
    fields: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    dimensions: list[dict[str, Any]] = []
    table = _schema_table_name(table_fqn, meta)
    columns = _column_names(meta)
    for field in fields:
        role = field.get("role")
        naming_role = field.get("naming_role")
        if role == "dimension_key" or naming_role == "surrogate_fk":
            dim = _find_dimension_for_key(schema, field["column"])
            dim_fqn = dim[0] if dim else ""
            dim_meta = dim[1] if dim else {}
            dim_cols = _column_names(dim_meta) if dim_meta else []
            display = _display_field_for_columns(dim_cols, field["column"]) if dim else ""
            code = _code_field_for_columns(dim_cols, field["column"]) if dim else ""
            dimensions.append({
                "name": _entity_name(_schema_table_name(dim_fqn, dim_meta)) if dim else field.get("expanded_name", field["column"]),
                "source_table": _qualified_name(table_fqn, meta),
                "source_key": field["column"],
                "display_table": _qualified_name(dim_fqn, dim_meta) if dim else "",
                "display_key": field["column"] if dim and field["column"].upper() in {c.upper() for c in dim_cols} else "",
                "display_column": display,
                "code_column": code,
                "status": "generated" if display else "needs_review",
                "confidence": field.get("confidence", 50),
            })
        elif role in {"dimension", "identifier", "attribute"} and naming_role in {"display", "code"}:
            dimensions.append({
                "name": field.get("expanded_name") or field["column"],
                "source_table": _qualified_name(table_fqn, meta),
                "source_key": "",
                "display_table": _qualified_name(table_fqn, meta),
                "display_key": "",
                "display_column": field["column"] if naming_role == "display" else "",
                "code_column": field["column"] if naming_role == "code" else "",
                "status": "generated",
                "confidence": field.get("confidence", 50),
            })

    if _table_type(table) == "dimension":
        display = _display_field_for_columns(columns)
        code = _code_field_for_columns(columns)
        if display or code:
            dimensions.insert(0, {
                "name": _entity_name(table),
                "source_table": _qualified_name(table_fqn, meta),
                "source_key": "",
                "display_table": _qualified_name(table_fqn, meta),
                "display_key": "",
                "display_column": display,
                "code_column": code,
                "status": "generated",
                "confidence": 90,
            })
    return dimensions


def _date_roles(schema: dict[str, Any], table_fqn: str, meta: dict[str, Any]) -> list[dict[str, Any]]:
    roles: list[dict[str, Any]] = []
    date_dims = [
        (fqn, m, find_date_dimension_key(m.get("columns", [])))
        for fqn, m in schema.items()
        if is_date_dimension_table(fqn, m.get("columns", []))
    ]
    date_dims = [(fqn, m, pk) for fqn, m, pk in date_dims if pk]
    if not date_dims:
        return roles

    date_fqn, date_meta, date_pk = date_dims[0]
    for col in _column_names(meta):
        role = detect_date_role(col)
        if not role:
            continue
        roles.append({
            "name": role.label,
            "business_role": role.key,
            "fact_table": _qualified_name(table_fqn, meta),
            "fact_column": col,
            "dimension_table": _qualified_name(date_fqn, date_meta),
            "dimension_key": date_pk,
            "synonyms": role.synonyms,
            "status": "generated",
            "confidence": 90,
        })
    return roles


def _relationships(schema: dict[str, Any]) -> list[dict[str, Any]]:
    relationships: list[dict[str, Any]] = []
    seen: set[str] = set()

    for fqn, meta in schema.items():
        table = _schema_table_name(fqn, meta)
        if _table_type(table) != "fact":
            continue
        for col in _column_names(meta):
            if not col.upper().endswith("_DMS_KEY"):
                continue
            dim = _find_dimension_for_key(schema, col)
            if not dim:
                continue
            dim_fqn, dim_meta = dim
            dim_cols = _column_names(dim_meta)
            display_col = _display_field_for_columns(dim_cols, col)
            code_col = _code_field_for_columns(dim_cols, col)
            role = _business_role_from_column(col)
            dim_key = col if col.upper() in {c.upper() for c in dim_cols} else ""
            if not dim_key:
                # Fall back to the first _DMS_KEY on the dimension table.
                dim_key = next((c for c in dim_cols if c.upper().endswith("_DMS_KEY")), "")
            rel_id = _relationship_id(_qualified_name(fqn, meta), _qualified_name(dim_fqn, dim_meta), role, col, dim_key)
            if rel_id in seen:
                continue
            seen.add(rel_id)
            relationships.append({
                "id": rel_id,
                "from_table": _qualified_name(fqn, meta),
                "to_table": _qualified_name(dim_fqn, dim_meta),
                "business_role": role,
                "relationship_type": "many_to_one",
                "join_type": "LEFT",
                "conditions": [{"from_column": col, "to_column": dim_key}],
                "display_column": display_col,
                "code_column": code_col,
                "use_when": [role.replace("_", " "), display_col, code_col],
                "status": "generated" if dim_key else "needs_review",
                "confidence": 88 if display_col else 75,
            })

    for fqn, meta in schema.items():
        for role in _date_roles(schema, fqn, meta):
            rel_id = _relationship_id(
                role["fact_table"], role["dimension_table"], role["business_role"],
                role["fact_column"], role["dimension_key"],
            )
            if rel_id in seen:
                continue
            seen.add(rel_id)
            relationships.append({
                "id": rel_id,
                "from_table": role["fact_table"],
                "to_table": role["dimension_table"],
                "business_role": role["business_role"],
                "relationship_type": "many_to_one",
                "join_type": "LEFT",
                "conditions": [{"from_column": role["fact_column"], "to_column": role["dimension_key"]}],
                "display_column": "",
                "code_column": "",
                "use_when": [role["name"], *role.get("synonyms", [])],
                "status": "generated",
                "confidence": role["confidence"],
            })
    return relationships


def build_semantic_model(schema_dir: str, *, business_desc: str = "", account_id: str = "") -> dict[str, Any]:
    schema = _read_schema(schema_dir)
    tables: list[dict[str, Any]] = []
    all_date_roles: list[dict[str, Any]] = []

    for fqn, meta in schema.items():
        table = _schema_table_name(fqn, meta)
        columns = _column_names(meta)
        enriched = enrich_columns(columns)
        fields = [_field_entry(item, meta) for item in enriched]
        date_roles = _date_roles(schema, fqn, meta)
        all_date_roles.extend(date_roles)
        tables.append({
            "fqn": fqn,
            "database": _database_name(fqn, meta),
            "schema": _schema_name(fqn, meta),
            "table": table,
            "qualified_name": _qualified_name(fqn, meta),
            "entity": _entity_name(table),
            "type": _table_type(table),
            "grain": "needs_admin_context",
            "fields": fields,
            "default_filters": _default_filters(fields),
            "measures": _measure_candidates(fields),
            "dimensions": _dimension_candidates(schema=schema, table_fqn=fqn, meta=meta, fields=fields),
            "date_roles": date_roles,
            "status": "generated",
        })

    model = {
        "version": 1,
        "account_id": account_id,
        "business_description": business_desc or "",
        "source": "kb_generation",
        "tables": tables,
        "relationships": _relationships(schema),
        "date_roles": all_date_roles,
        "status": "generated",
    }
    return model


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if not text:
        return '""'
    if re.search(r"[:\n\r]|^\s|\s$|[{}\[\],]", text):
        return json.dumps(text)
    return text


def _to_yaml(value: Any, indent: int = 0) -> str:
    space = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, val in value.items():
            if isinstance(val, (dict, list)):
                lines.append(f"{space}{key}:")
                lines.append(_to_yaml(val, indent + 2))
            else:
                lines.append(f"{space}{key}: {_yaml_scalar(val)}")
        return "\n".join(lines)
    if isinstance(value, list):
        if not value:
            return f"{space}[]"
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{space}-")
                lines.append(_to_yaml(item, indent + 2))
            else:
                lines.append(f"{space}- {_yaml_scalar(item)}")
        return "\n".join(lines)
    return f"{space}{_yaml_scalar(value)}"


def write_semantic_model(
    *,
    schema_dir: str,
    kb_dir: str,
    business_desc: str = "",
    account_id: str = "",
) -> dict[str, Any]:
    model = build_semantic_model(schema_dir, business_desc=business_desc, account_id=account_id)
    kb_path = Path(kb_dir)
    kb_path.mkdir(parents=True, exist_ok=True)

    # Preserve admin-approved entries that exist in the current model before
    # overwriting with the freshly generated one.  This ensures a KB rebuild
    # never silently discards field approvals, dimension approvals, or approved
    # metric expressions an admin has saved via the Semantic Layer UI.
    old_model = load_semantic_model(kb_dir)
    if old_model:
        model, drift = preserve_approvals(old_model, model)
        _log_drift(drift)

    (kb_path / MODEL_JSON).write_text(json.dumps(model, indent=2, sort_keys=True), encoding="utf-8")
    (kb_path / MODEL_YAML).write_text(_to_yaml(model) + "\n", encoding="utf-8")
    return model


def _log_drift(drift: dict[str, Any]) -> None:
    removed_tables = drift.get("removed_tables") or []
    removed_fields = drift.get("removed_approved_fields") or []
    changed_types  = drift.get("type_changed_fields") or []
    if removed_tables:
        log.warning(
            "Schema drift: %d table(s) removed from semantic model: %s",
            len(removed_tables), removed_tables,
        )
    if removed_fields:
        log.warning(
            "Schema drift: %d approved field(s) no longer in schema (column dropped): %s",
            len(removed_fields),
            [f"{r['table']}.{r['column']}" for r in removed_fields],
        )
    if changed_types:
        log.warning(
            "Schema drift: %d field(s) changed data type: %s",
            len(changed_types),
            [f"{c['table']}.{c['column']} {c['old_type']}->{c['new_type']}" for c in changed_types],
        )


def load_semantic_model(kb_dir: str) -> dict[str, Any]:
    path = Path(kb_dir) / MODEL_JSON
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def preserve_approvals(
    old_model: dict[str, Any],
    new_model: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Overlay admin-approved entries from *old_model* into *new_model*.

    Called by ``write_semantic_model`` before persisting, so a KB rebuild never
    silently discards approvals an admin has made.

    Returns ``(merged_model, drift_report)`` where *drift_report* contains:

    * ``removed_tables``         – tables present in old but absent in new schema
    * ``removed_approved_fields``– approved fields whose column was dropped from the schema
    * ``type_changed_fields``    – fields where the SQL data-type changed
    """
    drift: dict[str, Any] = {
        "removed_tables": [],
        "removed_approved_fields": [],
        "type_changed_fields": [],
    }

    # Build a fast lookup for new tables keyed by qualified_name / fqn / bare table
    new_lookup: dict[str, dict[str, Any]] = {}
    for t in new_model.get("tables") or []:
        for key in (
            str(t.get("qualified_name") or "").upper(),
            str(t.get("fqn") or "").upper(),
            str(t.get("table") or "").upper(),
        ):
            if key:
                new_lookup[key] = t

    # ── Detect removed tables ────────────────────────────────────────────────
    for old_t in old_model.get("tables") or []:
        qname = str(old_t.get("qualified_name") or old_t.get("table") or "").upper()
        if qname and qname not in new_lookup:
            drift["removed_tables"].append(
                old_t.get("qualified_name") or old_t.get("table") or old_t.get("fqn", "")
            )

    # ── Overlay approvals table by table ────────────────────────────────────
    for old_t in old_model.get("tables") or []:
        qname = str(old_t.get("qualified_name") or old_t.get("table") or "").upper()
        new_t = new_lookup.get(qname)
        if not new_t:
            continue  # table removed — already recorded in drift

        # Fields
        old_by_col: dict[str, dict[str, Any]] = {
            str(f.get("column") or "").upper(): f
            for f in (old_t.get("fields") or [])
        }
        new_cols: set[str] = set()
        for new_f in new_t.get("fields") or []:
            col_u = str(new_f.get("column") or "").upper()
            new_cols.add(col_u)
            old_f = old_by_col.get(col_u)
            if not old_f or old_f.get("status") != "approved":
                continue
            # Track data-type changes on approved fields
            if (
                new_f.get("data_type")
                and old_f.get("data_type")
                and new_f["data_type"] != old_f["data_type"]
            ):
                drift["type_changed_fields"].append({
                    "table": new_t.get("qualified_name"),
                    "column": new_f["column"],
                    "old_type": old_f["data_type"],
                    "new_type": new_f["data_type"],
                })
            for k in _APPROVED_FIELD_KEYS:
                if k in old_f:
                    new_f[k] = old_f[k]

        # Detect approved columns that no longer exist in the new schema
        for col_u, old_f in old_by_col.items():
            if old_f.get("status") == "approved" and col_u not in new_cols:
                drift["removed_approved_fields"].append({
                    "table": old_t.get("qualified_name"),
                    "column": old_f.get("column"),
                    "approved_meaning": old_f.get("approved_meaning", ""),
                })

        # Dimensions  (keyed by source_key + display_column pair)
        old_dims: dict[str, dict[str, Any]] = {
            str(d.get("source_key") or "").upper()
            + "|"
            + str(d.get("display_column") or "").upper(): d
            for d in (old_t.get("dimensions") or [])
        }
        for new_d in new_t.get("dimensions") or []:
            key = (
                str(new_d.get("source_key") or "").upper()
                + "|"
                + str(new_d.get("display_column") or "").upper()
            )
            old_d = old_dims.get(key)
            if not old_d or old_d.get("status") != "approved":
                continue
            for k in _APPROVED_DIMENSION_KEYS:
                if k in old_d:
                    new_d[k] = old_d[k]

        # Measures  (keyed by column name)
        old_measures: dict[str, dict[str, Any]] = {
            str(m.get("column") or "").upper(): m
            for m in (old_t.get("measures") or [])
        }
        for new_m in new_t.get("measures") or []:
            col_u = str(new_m.get("column") or "").upper()
            old_m = old_measures.get(col_u)
            if not old_m or old_m.get("status") != "approved":
                continue
            for k in _APPROVED_MEASURE_KEYS:
                if k in old_m:
                    new_m[k] = old_m[k]

        # Date roles  (keyed by fact_column)
        old_date_roles: dict[str, dict[str, Any]] = {
            str(r.get("fact_column") or "").upper(): r
            for r in (old_t.get("date_roles") or [])
        }
        for new_r in new_t.get("date_roles") or []:
            col_u = str(new_r.get("fact_column") or "").upper()
            old_r = old_date_roles.get(col_u)
            if not old_r or old_r.get("status") != "approved":
                continue
            for k in _APPROVED_DATE_ROLE_KEYS:
                if k in old_r:
                    new_r[k] = old_r[k]

    # ── Top-level relationships  (keyed by relationship id) ─────────────────
    old_rels: dict[str, dict[str, Any]] = {
        str(r.get("id") or "").upper(): r
        for r in (old_model.get("relationships") or [])
    }
    for new_r in new_model.get("relationships") or []:
        rel_id = str(new_r.get("id") or "").upper()
        old_r = old_rels.get(rel_id)
        if not old_r or old_r.get("status") != "approved":
            continue
        for k in ("status", "confidence", "join_type", "display_column", "code_column"):
            if k in old_r:
                new_r[k] = old_r[k]

    return new_model, drift


def build_field_plan_repair_note(semantic_plan: dict[str, Any]) -> str:
    """Return a specific LLM repair instruction for a ``field_plan_mismatch`` error.

    Unlike the generic "use the semantic plan" message, this injects the exact
    display field name, dimension table, and join key so the LLM knows precisely
    what to change without having to search the prompt context.
    """
    required = [
        f for f in (semantic_plan.get("fields") or [])
        if f.get("display_required") and f.get("table") and f.get("column")
    ]
    if not required:
        return (
            "\nSEMANTIC FIELD PLAN REPAIR RULE:\n"
            "- The SQL ignored one or more deterministic field-source mappings.\n"
            "- Use the exact table.column pairs and required joins from the Semantic field-source plan.\n"
            "- Do not move mapped fields to another table and do not remove underscores from column names.\n"
        )

    field_lines = "\n".join(
        "  - '{term}': SELECT {dim_table}.{display_col}"
        "  —  requires LEFT JOIN {dim_table} ON {src_table}.{join_key} = {dim_table}.{join_key}".format(
            term=f.get("term") or f.get("column", ""),
            dim_table=f.get("table", ""),
            display_col=f.get("column", ""),
            src_table=f.get("source_table", ""),
            join_key=f.get("source_key_column", ""),
        )
        for f in required
    )

    return (
        "\nSEMANTIC DISPLAY FIELD REPAIR RULE:\n"
        "- The SQL returned a raw _DMS_KEY column as a business label instead of the required display field.\n"
        "- Fix ALL of the following — the _DMS_KEY must ONLY appear in the JOIN ON clause:\n"
        f"{field_lines}\n"
        "- Never use a _DMS_KEY in SELECT or GROUP BY as a label. Always JOIN to the dimension table and SELECT its display column.\n"
        "- Keep all other query structure (date filters, WHERE clauses, metrics) unchanged.\n"
    )


def _terms_for_text(text: str) -> set[str]:
    terms = {t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if len(t) >= 3}
    expanded = set(terms)
    for term in terms:
        if term.endswith("ies") and len(term) > 4:
            expanded.add(term[:-3] + "y")
        elif term.endswith("es") and len(term) > 4:
            expanded.add(term[:-2])
        elif term.endswith("s") and len(term) > 3:
            expanded.add(term[:-1])
    return expanded


def _runtime_match_score(question_terms: set[str], values: list[str]) -> int:
    if not question_terms:
        return 1
    score = 0
    for value in values:
        value_terms = _terms_for_text(value)
        if not value_terms:
            continue
        score += len(question_terms & value_terms)
        joined = " ".join(value_terms)
        for term in question_terms:
            if term in joined:
                score += 1
    return score


def _question_asks_for_key(question: str, dimension_name: str = "") -> bool:
    q = re.sub(r"[^a-z0-9]+", " ", (question or "").lower()).strip()
    if not q:
        return False
    names = _terms_for_text(dimension_name)
    key_words = ("key", "keys", "id", "ids", "code", "codes")
    if not any(word in q.split() for word in key_words):
        return False
    if not names:
        return True
    for name in names:
        if re.search(rf"\b{name}\s+(?:key|keys|id|ids|code|codes)\b", q):
            return True
        if re.search(rf"\b(?:key|keys|id|ids|code|codes)\s+(?:for\s+)?{name}\b", q):
            return True
    return False


def _in_schema_scope(table: dict[str, Any], selected_schema: str = "") -> bool:
    if not selected_schema:
        return True
    schema = str(table.get("schema") or "").upper()
    return not schema or schema == selected_schema.upper()


def build_runtime_semantic_context(
    kb_dir: str,
    *,
    question: str = "",
    selected_schema: str = "",
    max_lines: int = 18,
) -> str:
    """Return compact model hints for SQL generation.

    This is guidance, not an approved metric registry. It helps the LLM choose
    display fields, date roles, and generated relationship roles without
    flooding the prompt with the full JSON model.
    """
    model = load_semantic_model(kb_dir)
    if not model:
        return ""

    selected_schema = (selected_schema or "").upper().strip()
    tables = [
        t for t in model.get("tables", []) or []
        if _in_schema_scope(t, selected_schema)
    ]
    table_names = {str(t.get("qualified_name") or t.get("table") or "").upper() for t in tables}
    table_names.update({str(t.get("table") or "").upper() for t in tables})
    if not tables:
        return ""

    q_terms = _terms_for_text(question)
    scored_lines: list[tuple[int, str]] = []

    for table in tables:
        qname = str(table.get("qualified_name") or table.get("table") or "")
        for dimension in table.get("dimensions", []) or []:
            display_col = str(dimension.get("display_column") or "")
            display_table = str(dimension.get("display_table") or "")
            source_key = str(dimension.get("source_key") or "")
            if not display_col:
                continue
            if display_table and display_table.upper() not in table_names:
                continue
            role_label = _business_role_from_column(source_key).replace("_", " ") if source_key else ""
            name = role_label.title() if role_label else str(dimension.get("name") or display_col)
            values = [
                name,
                role_label,
                display_col,
                display_table,
                source_key,
                str(dimension.get("approved_meaning") or ""),
            ]
            score = _runtime_match_score(q_terms, values)
            if score <= 0:
                continue
            if source_key and display_table and display_table.upper() != qname.upper():
                line = (
                    f"- Dimension '{name}': from {qname}.{source_key}, join {display_table} "
                    f"and display {display_table}.{display_col}"
                )
            else:
                line = f"- Dimension '{name}': display {qname}.{display_col}"
            code_col = str(dimension.get("code_column") or "")
            if code_col:
                line += f"; code field {display_table or qname}.{code_col}"
            scored_lines.append((score + int(dimension.get("confidence") or 0) // 20, line))

        for field in table.get("fields", []) or []:
            if str(field.get("status") or "") != "approved":
                continue
            column = str(field.get("column") or "")
            meaning = str(field.get("approved_meaning") or "")
            use_case = str(field.get("approved_use_case") or "")
            score = _runtime_match_score(q_terms, [column, meaning, use_case])
            if score <= 0:
                continue
            scored_lines.append((score + 6, f"- Approved field: use {qname}.{column} for {meaning or use_case}"))

        for date_role in table.get("date_roles", []) or []:
            name = str(date_role.get("name") or "")
            fact_column = str(date_role.get("fact_column") or "")
            dim_table = str(date_role.get("dimension_table") or "")
            dim_key = str(date_role.get("dimension_key") or "")
            synonyms = [str(s) for s in date_role.get("synonyms", []) or []]
            score = _runtime_match_score(q_terms, [name, fact_column, *synonyms])
            if score <= 0:
                continue
            scored_lines.append((
                score + 4,
                f"- Date role '{name}': {qname}.{fact_column} maps to {dim_table}.{dim_key}"
            ))

    for rel in model.get("relationships", []) or []:
        from_table = str(rel.get("from_table") or "")
        to_table = str(rel.get("to_table") or "")
        if selected_schema:
            if from_table.upper() not in table_names or to_table.upper() not in table_names:
                continue
        values = [
            str(rel.get("business_role") or ""),
            str(rel.get("display_column") or ""),
            str(rel.get("code_column") or ""),
            *[str(v) for v in rel.get("use_when", []) or []],
        ]
        score = _runtime_match_score(q_terms, values)
        if score <= 0:
            continue
        conditions = rel.get("conditions") or []
        condition_text = " AND ".join(
            f"{from_table}.{c.get('from_column')} = {to_table}.{c.get('to_column')}"
            for c in conditions
            if c.get("from_column") and c.get("to_column")
        )
        if not condition_text:
            continue
        line = f"- Relationship '{rel.get('business_role')}': {rel.get('join_type') or 'LEFT'} JOIN {to_table} ON {condition_text}"
        if rel.get("display_column"):
            line += f"; prefer display {to_table}.{rel.get('display_column')}"
        scored_lines.append((score + 5, line))

    if not scored_lines:
        return ""

    deduped: list[str] = []
    seen: set[str] = set()
    for _, line in sorted(scored_lines, key=lambda item: item[0], reverse=True):
        if line in seen:
            continue
        seen.add(line)
        deduped.append(line)
        if len(deduped) >= max_lines:
            break

    header = [
        "STRUCTURED SEMANTIC MODEL CONTEXT:",
        "Use this generated/approved semantic model guidance for display fields, date roles, and relationship roles.",
        "Approved metric formulas still take precedence over generated measure candidates.",
    ]
    if selected_schema:
        header.append(f"Schema scope: {selected_schema}. Ignore semantic model entries from other schemas.")
    return "\n".join(header + deduped)


def build_runtime_semantic_plan(
    kb_dir: str,
    *,
    question: str = "",
    selected_schema: str = "",
    max_fields: int = 8,
) -> dict[str, Any]:
    """Build validator-ready requirements from the structured semantic model."""
    model = load_semantic_model(kb_dir)
    if not model:
        return {"enabled": False, "fields": [], "joins": [], "required_tables": [], "reason": "no semantic model"}

    selected_schema = (selected_schema or "").upper().strip()
    tables = [
        t for t in model.get("tables", []) or []
        if _in_schema_scope(t, selected_schema)
    ]
    if not tables:
        return {"enabled": False, "fields": [], "joins": [], "required_tables": [], "reason": "no tables in selected schema"}

    q_terms = _terms_for_text(question)
    fields: list[dict[str, Any]] = []
    joins: list[dict[str, Any]] = []
    seen_fields: set[tuple[str, str]] = set()
    seen_joins: set[tuple[str, str, tuple[tuple[str, str], ...]]] = set()

    for table in tables:
        source_table = str(table.get("qualified_name") or table.get("table") or "")
        for dimension in table.get("dimensions", []) or []:
            display_col = str(dimension.get("display_column") or "")
            display_table = str(dimension.get("display_table") or "")
            source_key = str(dimension.get("source_key") or "")
            display_key = str(dimension.get("display_key") or source_key)
            if not display_col or not display_table or not source_key:
                continue
            role_label = _business_role_from_column(source_key).replace("_", " ") if source_key else ""
            name = role_label.title() if role_label else str(dimension.get("name") or display_col)
            if _question_asks_for_key(question, name) or (role_label and _question_asks_for_key(question, role_label)):
                continue
            score = _runtime_match_score(
                q_terms,
                [
                    name,
                    role_label,
                    display_col,
                    display_table,
                    source_key,
                    str(dimension.get("approved_meaning") or ""),
                ],
            )
            if score <= 0:
                continue
            field_key = (display_table.upper(), display_col.upper())
            if field_key not in seen_fields:
                seen_fields.add(field_key)
                fields.append({
                    "term": name,
                    "table": display_table,
                    "column": display_col,
                    "role": "display_dimension",
                    "display_required": True,
                    "source_table": source_table,
                    "source_key_column": source_key,
                    "confidence": dimension.get("confidence", 80),
                    "source": "semantic_model",
                })
            if source_key and display_key and source_table.upper() != display_table.upper():
                conditions = ((source_key.upper(), display_key.upper()),)
                join_key = (source_table.upper(), display_table.upper(), conditions)
                if join_key not in seen_joins:
                    seen_joins.add(join_key)
                    joins.append({
                        "from": source_table,
                        "to": display_table,
                        "conditions": [(source_key, display_key)],
                        "source": "semantic_model",
                    })
            if len(fields) >= max_fields:
                break
        if len(fields) >= max_fields:
            break

    if not fields:
        return {"enabled": False, "fields": [], "joins": [], "required_tables": [], "reason": "no matching semantic model fields"}

    required_tables = sorted({f["table"] for f in fields} | {j["from"] for j in joins} | {j["to"] for j in joins})
    return {
        "enabled": True,
        "fields": fields,
        "joins": joins,
        "required_tables": required_tables,
        "reason": "structured semantic model",
    }


def patch_field_approval(
    *,
    kb_dir: str,
    table_fqn: str = "",
    table_name: str = "",
    schema_name: str = "",
    column_name: str,
    approved_meaning: str,
    approved_use_case: str = "",
) -> bool:
    model = load_semantic_model(kb_dir)
    if not model:
        return False

    table_fqn_u = (table_fqn or "").upper()
    table_u = (table_name or "").upper()
    schema_u = (schema_name or "").upper()
    column_u = column_name.upper()
    changed = False

    for table in model.get("tables", []) or []:
        matches_table = False
        if table_fqn_u and str(table.get("fqn", "")).upper() == table_fqn_u:
            matches_table = True
        elif table_u and str(table.get("table", "")).upper() == table_u:
            matches_table = not schema_u or str(table.get("schema", "")).upper() == schema_u
        if not matches_table:
            continue
        for field in table.get("fields", []) or []:
            if str(field.get("column", "")).upper() != column_u:
                continue
            field["approved_meaning"] = approved_meaning.strip()
            field["approved_use_case"] = approved_use_case.strip()
            field["status"] = "approved"
            field["confidence"] = 100
            candidates = field.setdefault("business_candidates", [])
            if approved_meaning and approved_meaning.lower() not in {str(c).lower() for c in candidates}:
                candidates.insert(0, approved_meaning)
            changed = True

        for dimension in table.get("dimensions", []) or []:
            if str(dimension.get("display_column", "")).upper() == column_u or str(dimension.get("code_column", "")).upper() == column_u:
                dimension["status"] = "approved"
                dimension["confidence"] = 100
                if approved_meaning:
                    dimension["approved_meaning"] = approved_meaning.strip()
                changed = True

    if not changed:
        return False

    kb_path = Path(kb_dir)
    (kb_path / MODEL_JSON).write_text(json.dumps(model, indent=2, sort_keys=True), encoding="utf-8")
    (kb_path / MODEL_YAML).write_text(_to_yaml(model) + "\n", encoding="utf-8")
    return True
