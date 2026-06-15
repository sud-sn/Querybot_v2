"""Persistent admin-owned field metadata overrides for the Knowledge Base."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


_VERSION = 1


def override_path(account_id: str) -> Path:
    return Path("clients") / account_id / "field_overrides.json"


def load_field_overrides(account_id: str) -> dict:
    path = override_path(account_id)
    if not path.exists():
        return {"version": _VERSION, "tables": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": _VERSION, "tables": {}}
    if not isinstance(data, dict):
        return {"version": _VERSION, "tables": {}}
    if not isinstance(data.get("tables"), dict):
        data["tables"] = {}
    data["version"] = _VERSION
    return data


def write_field_overrides(account_id: str, data: dict) -> None:
    path = override_path(account_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(data or {})
    payload["version"] = _VERSION
    payload.setdefault("tables", {})
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def save_field_override(
    *,
    account_id: str,
    table_fqn: str,
    schema_name: str,
    table_name: str,
    file_stem: str,
    column_name: str,
    meaning: str,
    use_case: str = "",
    synonyms: list[str] | None = None,
    admin_note: str = "",
) -> dict:
    data = load_field_overrides(account_id)
    table_key = _canonical_table_key(table_fqn or f"{schema_name}.{table_name}")
    table_entry = data["tables"].setdefault(table_key, {
        "table_fqn": table_fqn,
        "schema_name": schema_name,
        "table_name": table_name,
        "file_stem": file_stem,
        "fields": {},
    })
    table_entry.update({
        "table_fqn": table_fqn or table_entry.get("table_fqn", ""),
        "schema_name": schema_name or table_entry.get("schema_name", ""),
        "table_name": table_name or table_entry.get("table_name", ""),
        "file_stem": file_stem or table_entry.get("file_stem", ""),
    })
    clean_synonyms = _clean_synonyms(synonyms or [])
    field = {
        "column_name": column_name,
        "meaning": meaning.strip(),
        "use_case": use_case.strip(),
        "synonyms": clean_synonyms,
        "admin_note": admin_note.strip(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    table_entry.setdefault("fields", {})[column_name.upper()] = field
    write_field_overrides(account_id, data)
    return field


def table_overrides(
    data: dict,
    *table_names: str,
) -> dict[str, dict]:
    wanted = set()
    for name in table_names:
        wanted.update(_table_variants(name))
    if not wanted:
        return {}
    for key, entry in (data.get("tables") or {}).items():
        candidates = set()
        candidates.update(_table_variants(key))
        candidates.update(_table_variants(entry.get("table_fqn", "")))
        candidates.update(_table_variants(entry.get("table_name", "")))
        candidates.update(_table_variants(entry.get("file_stem", "")))
        if wanted & candidates:
            fields = entry.get("fields") or {}
            return fields if isinstance(fields, dict) else {}
    return {}


def field_override_map(data: dict) -> dict[tuple[str, str], dict]:
    result: dict[tuple[str, str], dict] = {}
    for key, entry in (data.get("tables") or {}).items():
        table_names = {
            key,
            entry.get("table_fqn", ""),
            entry.get("table_name", ""),
            entry.get("file_stem", ""),
        }
        variants = set()
        for name in table_names:
            variants.update(_table_variants(name))
        for column_key, field in (entry.get("fields") or {}).items():
            if not isinstance(field, dict):
                continue
            column = (field.get("column_name") or column_key).upper()
            for variant in variants:
                result[(variant, column)] = field
    return result


def parse_synonyms(value: str) -> list[str]:
    return _clean_synonyms(
        part
        for line in (value or "").splitlines()
        for part in line.split(",")
    )


def _clean_synonyms(values) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        term = " ".join(str(value or "").strip().split())
        key = term.lower()
        if not term or key in seen:
            continue
        seen.add(key)
        cleaned.append(term)
    return cleaned


def _canonical_table_key(name: str) -> str:
    variants = _split_table_name(name)
    return ".".join(variants).upper()


def _split_table_name(name: str) -> list[str]:
    cleaned = str(name or "").replace("[", "").replace("]", "")
    cleaned = cleaned.replace("`", "").replace('"', "")
    if "__" in cleaned and "." not in cleaned:
        cleaned = cleaned.replace("__", ".")
    return [part.strip() for part in cleaned.split(".") if part.strip()]


def _table_variants(name: str) -> set[str]:
    parts = [part.upper() for part in _split_table_name(name)]
    if not parts:
        return set()
    variants = {".".join(parts), parts[-1]}
    if len(parts) >= 2:
        variants.add(".".join(parts[-2:]))
    return variants
