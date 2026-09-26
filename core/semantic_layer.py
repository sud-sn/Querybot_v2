"""Read-only Semantic Layer metadata extracted from generated KB markdown."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable

log = logging.getLogger("querybot.semantic_layer")


def table_name_variants(name: str) -> set[str]:
    parts = [p.strip().strip("[]`\"").upper() for p in re.split(r"\s*\.\s*", name or "") if p.strip()]
    if not parts:
        return set()
    variants = {".".join(parts), parts[-1]}
    if len(parts) >= 2:
        variants.add(".".join(parts[-2:]))
    return variants


def table_allowed(table_ref: str, allowed_tables: Iterable[str] | None) -> bool:
    if allowed_tables is None:
        return True
    ref_variants = table_name_variants(table_ref)
    return any(ref_variants & table_name_variants(allowed) for allowed in allowed_tables)


def build_semantic_layer_tables(
    *,
    kb_dir: str,
    schema_dir: str = "",
    allowed_tables: Iterable[str] | None = None,
    approved_feedback: dict[tuple[str, str], dict] | None = None,
    pending_feedback: set[tuple[str, str]] | None = None,
    field_overrides: dict | None = None,
    account_id: str = "",
) -> list[dict]:
    """
    Build table/field metadata for the user portal.

    It intentionally does not expose the full KB markdown. Each field's
    meaning, use and terms are resolved by core.meaning from every store that
    holds one -- an admin's override, an approved suggestion, the semantic
    model's approvals, the join graph's confirmed properties, confirmed
    business meanings, the table's column terms, the database's comment and
    the knowledge base's prose -- and carry where they came from. Pending
    feedback is only marked. ``account_id`` reads the stores kept in the
    database; without it the page resolves from the files alone.
    """
    root = Path(kb_dir) if kb_dir else None
    if not root or not root.exists():
        return []

    schema_root = Path(schema_dir) if schema_dir else root
    schema_map = _load_schema_json(schema_root)
    approved_feedback = approved_feedback or {}
    pending_feedback = pending_feedback or set()
    field_overrides = field_overrides or {}
    stores = _account_stores(account_id, root)

    tables: list[dict] = []
    for kb_file in sorted(root.glob("*_kb.md")):
        try:
            content = kb_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        # Try to get FQN from the KB file header first.
        # LLM-generated KB files often write a plain heading like "# Attendance"
        # without schema prefix. Fall back to the corresponding schema .md file
        # (written by _az_md which always includes the full FQN) or _schema.json.
        fqn = _extract_fqn(content)

        if not fqn or "." not in fqn:
            # Try the matching schema file — same stem without "_kb"
            schema_stem = kb_file.stem.replace("_kb", "")
            schema_md   = Path(schema_root) / f"{schema_stem}.md"
            if schema_md.exists():
                try:
                    schema_content = schema_md.read_text(encoding="utf-8", errors="replace")
                    fqn = _extract_fqn(schema_content) or fqn
                except Exception:
                    pass

        if not fqn or "." not in fqn:
            # Last resort: match against schema_map keys using just the table name
            stem_upper = kb_file.stem.replace("_kb", "").upper()
            for key in schema_map:
                if table_name_variants(key) & {stem_upper}:
                    fqn = key
                    break
            else:
                fqn = fqn or stem_upper

        if not table_allowed(fqn, allowed_tables):
            continue

        db_name, schema_name, table_name = _split_fqn(fqn)
        fields = _parse_kb_columns(content)
        schema_fields = _schema_fields_for_table(schema_map, fqn)

        if not fields:
            fields = schema_fields
        else:
            fields = _merge_schema_details(fields, schema_fields)

        from core.field_overrides import table_overrides
        table_field_overrides = table_overrides(
            field_overrides,
            fqn,
            table_name,
            kb_file.stem.replace("_kb", ""),
        )
        fields = [
            _resolved_field(field, fqn, table_name, stores,
                            approved_feedback.get((fqn.upper(), field["column"].upper())),
                            table_field_overrides.get(field["column"].upper()),
                            (fqn.upper(), field["column"].upper()) in pending_feedback)
            for field in fields
        ]

        avg_conf = round(sum(f["confidence"] for f in fields) / len(fields)) if fields else 0
        tables.append({
            "file": kb_file.name,
            "file_stem": kb_file.stem.replace("_kb", ""),
            "fqn": fqn.upper(),
            "database": db_name,
            "schema": schema_name,
            "table": table_name,
            "overview": _extract_overview(content),
            "fields": fields,
            "field_count": len(fields),
            "confidence": avg_conf,
        })

    return tables


# How a resolved description's evidence is named on the page (field["source"]).
_SOURCE_CODE = {
    "admin override": "admin_override",
    "approved edit in the knowledge base": "admin_override",
    "approved suggestion": "approved_feedback",
    "approved in the semantic model": "semantic_model",
    "database comment": "database_comment",
    "knowledge base (generated)": "generated",
}


def _account_stores(account_id: str, kb_root: Path) -> dict:
    """What the database-held stores say, read once per page: the join
    graph's column properties by table, the tables' column terms, column
    meanings by compact code, and the semantic model's fields by table.
    Which of them count (confirmed, approved) is core.meaning's decision."""
    stores: dict = {"properties": {}, "column_terms": {}, "meanings": {}, "model_fields": {}}
    try:
        from core.semantic_model import load_semantic_model

        for table in (load_semantic_model(str(kb_root)) or {}).get("tables") or []:
            key = str(table.get("table") or "").upper()
            stores["model_fields"][key] = {str(f.get("column") or "").upper(): f for f in table.get("fields") or []}
    except Exception as exc:
        log.warning("Semantic Layer: the model's approvals could not be read: %s", exc)
    if not account_id:
        return stores
    import store

    try:
        tables = {str(e.get("entity_name") or ""): str(e.get("table_name") or "").upper()
                  for e in store.list_entities(account_id, active_only=False)}
        for prop in store.list_all_entity_properties(account_id):
            table = tables.get(str(prop.get("entity_name") or ""), "")
            if table:
                stores["properties"].setdefault(table, {})[str(prop.get("column_name") or "").upper()] = prop
    except Exception as exc:
        log.warning("Semantic Layer: the join graph's properties could not be read for %s: %s", account_id, exc)
    try:
        for name, row in store.list_table_descriptions(account_id).items():
            stores["column_terms"][str(name).split(".")[-1].upper()] = dict(row.get("column_synonym_map") or {})
    except Exception as exc:
        log.warning("Semantic Layer: column terms could not be read for %s: %s", account_id, exc)
    try:
        # Every status: core.meaning is where only a confirmed meaning counts.
        for meaning in store.list_business_meanings(account_id):
            if str(meaning.get("scope") or "") == "column":
                stores["meanings"][re.sub(r"[^A-Za-z0-9]", "", str(meaning.get("subject") or "")).upper()] = meaning
    except Exception as exc:
        log.warning("Semantic Layer: business meanings could not be read for %s: %s", account_id, exc)
    return stores


def _resolved_field(field: dict, fqn: str, table_name: str, stores: dict,
                    approved: dict | None, override: dict | None, pending: bool) -> dict:
    """One field of the page, its meaning resolved by core.meaning."""
    from core.meaning import CONFIDENCE, column_meaning

    column = field["column"].upper()
    table_key = str(table_name or fqn.split(".")[-1]).upper()
    # What the knowledge base says, without the placeholders the parsers fill
    # in for a column it does not describe.
    kb_meaning = str(field.get("meaning") or "")
    if kb_meaning in (_fallback_meaning(field["column"]), _NO_MEANING):
        kb_meaning = ""
    kb_use_case = str(field.get("use_case") or "")
    if kb_use_case == _default_use_case(field["column"]):
        kb_use_case = ""
    resolved = column_meaning(
        column=column,
        override=override,
        approved_feedback=approved,
        model_field=stores["model_fields"].get(table_key, {}).get(column),
        graph_property=stores["properties"].get(table_key, {}).get(column),
        business_meaning=stores["meanings"].get(re.sub(r"[^A-Za-z0-9]", "", column)),
        column_terms=stores["column_terms"].get(table_key, {}).get(column),
        db_comment=field.get("db_comment") or "",
        kb_field={"meaning": kb_meaning, "use_case": kb_use_case, "synonyms": field.get("synonyms") or [],
                  "approved": bool(field.get("approved"))},
    )
    out = dict(field)
    description = resolved.get("description")
    out["meaning"] = description.value if description else _fallback_meaning(field["column"])
    out["use_case"] = resolved["use_case"].value if "use_case" in resolved else _default_use_case(field["column"])
    out["synonyms"] = list(resolved["synonyms"].value) if "synonyms" in resolved else []
    out["label"] = resolved["label"].value if "label" in resolved else ""
    out["approved"] = bool(description and description.confirmed)
    out["pending"] = pending
    # Nothing describes it, or only the knowledge base does and it said the
    # column needs context.
    out["needs_context"] = description is None or (description.source == "ai" and bool(field.get("needs_context")))
    out["source"] = _SOURCE_CODE.get(description.evidence, "generated") if description else "generated"
    out["meaning_evidence"] = description.evidence if description else ""
    if description is None:
        out["confidence"] = 45
    elif description.source != "ai":
        out["confidence"] = CONFIDENCE[description.source]
    out["admin_note"] = (override or {}).get("admin_note") or ""
    out["updated_at"] = (override or {}).get("updated_at") or ""
    return out


def find_semantic_field(tables: list[dict], table_fqn: str, column_name: str) -> tuple[dict, dict] | None:
    wanted_table = table_fqn.upper()
    wanted_col = column_name.upper()
    for table in tables:
        if table["fqn"].upper() != wanted_table:
            continue
        for field in table["fields"]:
            if field["column"].upper() == wanted_col:
                return table, field
    return None


def _extract_fqn(content: str) -> str:
    for line in content.splitlines()[:8]:
        stripped = line.strip().lstrip("#").strip()
        match = re.match(r"^([A-Z0-9_]+\.[A-Z0-9_]+(?:\.[A-Z0-9_]+)?)(?:\s|$)", stripped.upper())
        if match:
            return match.group(1)
    return ""


def _split_fqn(fqn: str) -> tuple[str, str, str]:
    parts = [p.strip().strip("[]`\"") for p in (fqn or "").split(".") if p.strip()]
    if len(parts) >= 3:
        return parts[-3].upper(), parts[-2].upper(), parts[-1].upper()
    if len(parts) == 2:
        return "", parts[0].upper(), parts[1].upper()
    return "", "", (parts[0] if parts else "").upper()


def _extract_overview(content: str) -> str:
    lines = content.splitlines()
    in_overview = False
    chunks: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            if in_overview:
                break
            in_overview = stripped.lower().startswith("## overview")
            continue
        if in_overview and stripped:
            chunks.append(stripped.lstrip("- ").strip())
    return " ".join(chunks)[:300]


def _parse_kb_columns(content: str) -> list[dict]:
    section = _section_lines(content, "columns")
    fields: list[dict] = []
    synonyms = _parse_business_synonyms(content)
    metrics = _parse_key_metrics(content)

    for idx, line in enumerate(section):
        stripped = line.strip()
        if not stripped:
            continue
        next_line = section[idx + 1].strip() if idx + 1 < len(section) else ""
        parsed = _parse_column_bullet(stripped) or _parse_column_table_row(stripped, next_line)
        if not parsed:
            continue
        col = parsed["column"].upper()
        # Business terms get their own structured "synonyms" box in the portal
        # UI now (see build_semantic_layer_tables) instead of being stitched
        # as a "Business terms: ..." fragment into the free-text use_case —
        # that stitching only ever ran for columns the KB-build LLM happened
        # to add a Business Synonyms row for, which is why the fragment used
        # to appear on some fields but not others.
        parsed["synonyms"] = list(synonyms.get(col, {}).get("terms") or [])
        use_bits: list[str] = []
        if col in metrics:
            use_bits.append("Metric: " + ", ".join(metrics[col]))
        if col in synonyms and synonyms[col].get("notes"):
            use_bits.append(synonyms[col]["notes"])
        existing_use_case = (parsed.get("use_case") or "").strip()
        if existing_use_case and use_bits:
            parsed["use_case"] = existing_use_case + " | " + " | ".join(use_bits)
        else:
            parsed["use_case"] = existing_use_case or " | ".join(use_bits) or _default_use_case(parsed["column"])
        if not parsed.get("approved"):
            parsed["confidence"] = _confidence(parsed, bool(use_bits))
        fields.append(parsed)

    return fields


def _section_lines(content: str, name: str) -> list[str]:
    lines = content.splitlines()
    in_section = False
    collected: list[str] = []
    wanted = f"## {name}".lower()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            if in_section:
                break
            in_section = stripped.lower().startswith(wanted)
            continue
        if in_section:
            collected.append(line)
    return collected


def _parse_column_bullet(line: str) -> dict | None:
    match = re.match(r"^-\s*`([^`]+)`\s*(?:\(([^)]+)\))?\s*:\s*(.+)$", line)
    if not match:
        return None
    meaning = _clean_meaning(match.group(3))
    return {
        "column": match.group(1).strip(),
        "type": (match.group(2) or "").strip(),
        "nullable": "",
        "meaning": meaning,
        "distinct_values": _extract_values(match.group(3)),
        "needs_context": "[NEEDS CONTEXT]" in line.upper(),
    }


def _parse_column_table_row(line: str, next_line: str = "") -> dict | None:
    if not line.startswith("|") or "---" in line or "column" in line.lower():
        return None
    cells = [c.strip() for c in line.strip("|").split("|")]
    if len(cells) < 2:
        return None
    col = cells[0].strip("` ")
    if not col:
        return None
    base = {
        "column": col,
        "type": cells[1],
        "nullable": cells[2] if len(cells) > 2 else "",
        "meaning": _fallback_meaning(col),
        "distinct_values": cells[3] if len(cells) > 3 else "",
        "needs_context": False,
    }
    if len(cells) > 4 and cells[4].strip():
        base["meaning"] = cells[4].strip()
        base["use_case"] = cells[5].strip() if len(cells) > 5 else ""
        confidence_cell = cells[6] if len(cells) > 6 else ""
        base["confidence"] = _parse_confidence_value(confidence_cell or "100")
        # Approved when the row says so -- the approval patcher's source
        # marker, or a confidence written as 100% -- and not because a
        # generated row has no confidence cell: that read the knowledge base's
        # own prose as an admin's decision.
        base["approved"] = (
            (len(cells) > 7 and "admin-approved semantic layer edit" in cells[7].lower())
            or (bool(re.search(r"\d", confidence_cell)) and base["confidence"] >= 100)
        )
        base["needs_context"] = False
    # If the next line contains an admin approval comment, use that meaning
    if next_line and "<!-- Approved:" in next_line:
        m = re.search(r"<!--\s*Approved:\s*([^|>]+?)(?:\s*\|\s*Use case:\s*([^|>]+?))?(?:\s*\|[^>]*)?\s*-->",
                      next_line)
        if m:
            base["meaning"]      = m.group(1).strip()
            base["use_case"]     = (m.group(2) or "").strip()
            base["confidence"]   = 100
            base["approved"]     = True
            base["needs_context"] = False
    return base


def _parse_confidence_value(value: str) -> int:
    match = re.search(r"\d+", str(value or ""))
    if not match:
        return 100
    return max(0, min(100, int(match.group(0))))


def _parse_business_synonyms(content: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for line in _section_lines(content, "business synonyms"):
        stripped = line.strip()
        if not stripped.startswith("|") or "---" in stripped or "plain english" in stripped.lower():
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 2:
            continue
        terms = [t.strip() for t in cells[0].split(",") if t.strip()]
        col = cells[1].strip("` ").upper()
        if not col:
            continue
        current = result.setdefault(col, {"terms": [], "notes": ""})
        current["terms"].extend(t for t in terms if t not in current["terms"])
        if len(cells) > 2 and cells[2] and not current["notes"]:
            current["notes"] = cells[2]
    return result


def _parse_key_metrics(content: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for line in _section_lines(content, "key metrics"):
        match = re.match(r"^-\s*\*\*([^*]+)\*\*\s*:?\s*`?([^`\n]*)`?", line.strip())
        if not match:
            continue
        metric = match.group(1).strip()
        expr = match.group(2).strip()
        for col in re.findall(r"[A-Z][A-Z0-9_]*", expr.upper()):
            result.setdefault(col, []).append(metric)
    return result


_NO_MEANING = "Needs business review."


def _clean_meaning(text: str) -> str:
    text = re.sub(r"\bvalues are\b.*$", "", text, flags=re.IGNORECASE).strip()
    return text.rstrip(". ") or _NO_MEANING


def _extract_values(text: str) -> str:
    match = re.search(r"values are\s+(.+?)(?:\.|$)", text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _fallback_meaning(column: str) -> str:
    return f"{column.replace('_', ' ').title()} field from the selected table."


def _default_use_case(column: str) -> str:
    return f"Used when a question explicitly refers to {column.replace('_', ' ').lower()}."


def _confidence(field: dict, has_business_mapping: bool) -> int:
    if field.get("needs_context"):
        return 45
    meaning = field.get("meaning") or ""
    if meaning.startswith(field["column"].replace("_", " ").title()):
        return 62
    score = 82 if len(meaning) >= 30 else 72
    if field.get("distinct_values"):
        score += 5
    if has_business_mapping:
        score += 8
    return min(score, 98)


def _load_schema_json(schema_root: Path) -> dict:
    try:
        path = schema_root / "_schema.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _schema_fields_for_table(schema_map: dict, fqn: str) -> list[dict]:
    meta = None
    for key, value in schema_map.items():
        if table_name_variants(key) & table_name_variants(fqn):
            meta = value
            break
    if not isinstance(meta, dict):
        return []
    columns = meta.get("columns") or meta.get("Columns") or []
    fields: list[dict] = []
    for col in columns:
        if not isinstance(col, dict):
            continue
        name = col.get("COLUMN_NAME") or col.get("column_name") or col.get("name") or ""
        if not name:
            continue
        ctype = col.get("DATA_TYPE") or col.get("data_type") or col.get("type") or ""
        nullable = col.get("IS_NULLABLE") or col.get("nullable") or ""
        fields.append({
            "column": str(name),
            "type": str(ctype),
            "nullable": str(nullable),
            "meaning": _fallback_meaning(str(name)),
            "use_case": _default_use_case(str(name)),
            "distinct_values": "",
            "db_comment": " ".join(str(col.get("comment") or col.get("COMMENT") or "").split()),
            "needs_context": False,
            "confidence": 62,
        })
    return fields


def _merge_schema_details(fields: list[dict], schema_fields: list[dict]) -> list[dict]:
    """
    Merge KB-parsed fields with schema fields.

    The schema fields (from _schema.json) are the authoritative source for
    which columns exist.  KB fields provide meaning/use-case/confidence.

    Logic:
      - Start with ALL schema_fields as the base (so every column is shown)
      - For each schema column, if the KB also has data for it, override with
        the KB meaning/confidence/use_case/distinct_values
      - Schema columns NOT in the KB get "needs context" status (confidence 45)
      - KB columns NOT in schema_fields are appended as-is (extra context)
    """
    by_kb_col   = {f["column"].upper(): f for f in fields}
    by_sch_col  = {f["column"].upper(): f for f in schema_fields}
    merged      = []
    seen        = set()

    # Pass 1: all schema columns, enriched by KB where available
    for sch_field in schema_fields:
        col_upper = sch_field["column"].upper()
        seen.add(col_upper)
        kb_field = by_kb_col.get(col_upper)
        if kb_field:
            # KB has data — use it, fill in type/nullable from schema if missing
            result = dict(kb_field)
            if not result.get("type"):
                result["type"] = sch_field.get("type", "")
            if not result.get("nullable"):
                result["nullable"] = sch_field.get("nullable", "")
            result["db_comment"] = sch_field.get("db_comment", "")
        else:
            # Schema column with no KB meaning — show as "needs context"
            result = dict(sch_field)
            result["needs_context"] = True
            result["confidence"]    = 45
        merged.append(result)

    # Pass 2: any KB columns not in schema (e.g. approved edits for renamed cols)
    for kb_field in fields:
        if kb_field["column"].upper() not in seen:
            merged.append(kb_field)

    return merged
