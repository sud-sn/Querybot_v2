"""Read-only Semantic Layer metadata extracted from generated KB markdown."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable


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
) -> list[dict]:
    """
    Build table/field metadata for the user portal.

    It intentionally does not expose the full KB markdown. Approved feedback can
    override displayed metadata, while pending feedback is only marked.
    """
    root = Path(kb_dir) if kb_dir else None
    if not root or not root.exists():
        return []

    schema_root = Path(schema_dir) if schema_dir else root
    schema_map = _load_schema_json(schema_root)
    approved_feedback = approved_feedback or {}
    pending_feedback = pending_feedback or set()
    field_overrides = field_overrides or {}

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
        for field in fields:
            key = (fqn.upper(), field["column"].upper())
            approved = approved_feedback.get(key)
            if approved:
                field["meaning"] = approved.get("suggested_meaning") or field["meaning"]
                field["use_case"] = approved.get("suggested_use_case") or field["use_case"]
                field["confidence"] = 100
                field["approved"] = True
            else:
                field["approved"] = False
            field["pending"] = key in pending_feedback
            override = table_field_overrides.get(field["column"].upper())
            if override:
                field["meaning"] = override.get("meaning") or field["meaning"]
                field["use_case"] = override.get("use_case") or field["use_case"]
                field["synonyms"] = override.get("synonyms") or []
                field["admin_note"] = override.get("admin_note") or ""
                field["updated_at"] = override.get("updated_at") or ""
                field["confidence"] = 100
                field["approved"] = True
                field["needs_context"] = False
                field["source"] = "admin_override"
            else:
                field.setdefault("synonyms", [])
                field.setdefault("admin_note", "")
                field.setdefault("updated_at", "")
                field.setdefault("source", "generated")

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
        use_bits: list[str] = []
        if col in metrics:
            use_bits.append("Metric: " + ", ".join(metrics[col]))
        if col in synonyms:
            use_bits.append("Business terms: " + ", ".join(synonyms[col]["terms"]))
            if synonyms[col].get("notes"):
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
        base["confidence"] = _parse_confidence_value(cells[6] if len(cells) > 6 else "100")
        base["approved"] = (
            len(cells) > 7
            and "admin-approved semantic layer edit" in cells[7].lower()
        ) or base["confidence"] >= 100
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


def _clean_meaning(text: str) -> str:
    text = re.sub(r"\bvalues are\b.*$", "", text, flags=re.IGNORECASE).strip()
    return text.rstrip(". ") or "Needs business review."


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
