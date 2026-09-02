"""
store/table_description_store.py

Per-table business descriptions for the tables an admin selected.

Two fields, and the split between them is the whole point:

  description — prose. Reaches KB generation, and only KB generation. It
                improves the document the model reads about this table.
  synonyms    — business terms. Reach source resolution, which never reads
                prose (core/source_resolution._table_aliases matches on names,
                labels and synonyms). These are what decide WHICH table a
                question lands on.

Writing a beautiful paragraph and expecting the right table to be picked is the
mistake this module is shaped to prevent.
"""

from __future__ import annotations

import re
from typing import Any

from store.db import get_db


def _norm_table(table_name: str) -> str:
    """Compare tables the way the selection list writes them: SCHEMA.TABLE."""
    cleaned = str(table_name or "").strip().strip("[]").replace("[", "").replace("]", "")
    return cleaned.upper()


def _bare_name(table_name: str) -> str:
    """The table alone, without database or schema qualification."""
    return _norm_table(table_name).split(".")[-1]


def _match_key(stored_keys, table_name: str) -> str | None:
    """Find the stored key for a table named at ANY level of qualification.

    Callers do not agree on how a table is written and never will. The setup
    page posts what the picker listed (SCHEMA.TABLE, sometimes with the
    database in front); core/knowledge.py:688 uses the schema FILE STEM, which
    is the bare table; the SQL builder quotes it as [SCHEMA].[TABLE]. An exact
    string match satisfies only the first of those, which is how the first
    version of this shipped with the KB half silently reading nothing at all.

    Falls back to the bare name only when it is UNAMBIGUOUS. Two schemas each
    holding an ORDERS table is ordinary, and quietly attaching one schema's
    description to the other's KB document is worse than attaching none.
    """
    wanted = _norm_table(table_name)
    if wanted in stored_keys:
        return wanted
    bare = _bare_name(table_name)
    if not bare:
        return None
    hits = [key for key in stored_keys if _bare_name(key) == bare]
    return hits[0] if len(hits) == 1 else None


def split_synonyms(raw: Any) -> list[str]:
    """Comma/newline separated terms, de-duplicated, order preserved."""
    if isinstance(raw, (list, tuple, set)):
        parts = [str(item) for item in raw]
    else:
        parts = re.split(r"[,;\n]+", str(raw or ""))
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        term = " ".join(part.split()).strip()
        if not term:
            continue
        key = term.casefold()
        if key not in seen:
            seen.add(key)
            out.append(term)
    return out


def _row_to_dict(row) -> dict[str, Any]:
    entry = dict(row)
    entry["synonym_list"] = split_synonyms(entry.get("synonyms"))
    return entry


def get_table_description(account_id: str, table_name: str) -> dict[str, Any] | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM table_description WHERE account_id=? AND UPPER(table_name)=?",
            (account_id, _norm_table(table_name)),
        ).fetchone()
        if row:
            return _row_to_dict(row)
    # No exact match: the caller may be naming the table at a different level
    # of qualification than the setup page stored it at. This is the ordinary
    # case for the KB builder, not an edge case.
    stored = list_table_descriptions(account_id)
    key = _match_key(stored.keys(), table_name)
    return stored.get(key) if key else None


def list_table_descriptions(account_id: str) -> dict[str, dict[str, Any]]:
    """Every stored description for this tenant, keyed by NORMALISED table name."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM table_description WHERE account_id=? ORDER BY table_name",
            (account_id,),
        ).fetchall()
    return {_norm_table(row["table_name"]): _row_to_dict(row) for row in rows}


def save_table_description(
    account_id: str,
    table_name: str,
    *,
    description: str = "",
    synonyms: Any = "",
    updated_by: str = "",
) -> None:
    """Upsert one table's description. Blanking both fields removes the row, so
    an emptied form does not leave a stale record claiming to be described."""
    table = str(table_name or "").strip()
    if not table:
        raise ValueError("A table name is required.")
    text = str(description or "").strip()
    terms = ", ".join(split_synonyms(synonyms))
    with get_db() as conn:
        if not text and not terms:
            conn.execute(
                "DELETE FROM table_description WHERE account_id=? AND UPPER(table_name)=?",
                (account_id, _norm_table(table)),
            )
            return
        conn.execute(
            """
            INSERT INTO table_description
                (account_id, table_name, description, synonyms, updated_by, updated_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(account_id, table_name) DO UPDATE SET
                description = excluded.description,
                synonyms    = excluded.synonyms,
                updated_by  = excluded.updated_by,
                updated_at  = datetime('now')
            """,
            (account_id, table, text, terms, str(updated_by or "")),
        )


def describe_selected_tables(
    account_id: str,
    selected_tables: list[str] | None,
    built_tables: set[str] | None = None,
) -> list[dict[str, Any]]:
    """One row per SELECTED table, with its description and its status.

    Scoped to the selection on purpose: a schema can hold hundreds of tables
    and the admin is only accountable for the ones they turned on.

    `built_tables` is the set that already has a generated KB. A selected table
    absent from it was added after the last build, which is the case the admin
    most needs surfaced -- it is the one that will otherwise reach users with no
    business context at all.
    """
    stored = list_table_descriptions(account_id)
    # Built-table names arrive from the KB directory, where each document is
    # named by the BARE table, while the selection is fully qualified. Comparing
    # the two directly matched nothing, so every table on a fully-built client
    # reported as "new" -- the same qualification mismatch _match_key exists to
    # absorb. Both forms are held so either spelling resolves.
    built: set[str] = set()
    for name in (built_tables or set()):
        built.add(_norm_table(name))
        built.add(_bare_name(name))
    out: list[dict[str, Any]] = []
    for table in selected_tables or []:
        name = str(table or "").strip()
        if not name:
            continue
        key = _norm_table(name)
        entry = stored.get(key) or {}
        has_text = bool(str(entry.get("description") or "").strip())
        is_new = bool(built) and key not in built and _bare_name(name) not in built
        out.append({
            "table_name": name,
            "description": str(entry.get("description") or ""),
            "synonyms": str(entry.get("synonyms") or ""),
            "synonym_list": entry.get("synonym_list") or [],
            "updated_at": entry.get("updated_at") or "",
            "updated_by": entry.get("updated_by") or "",
            "described": has_text,
            # Newly selected outranks undescribed in the UI: both need the same
            # action, but only one of them is a change since the admin last looked.
            "status": "new" if is_new else ("described" if has_text else "undescribed"),
        })
    return out


def description_coverage(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total": len(rows),
        "described": sum(1 for row in rows if row.get("described")),
        "undescribed": sum(1 for row in rows if row.get("status") == "undescribed"),
        "new": sum(1 for row in rows if row.get("status") == "new"),
    }
