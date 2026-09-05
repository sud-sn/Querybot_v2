"""Named subject areas inside one workspace.

A domain is a table set with a name, a description and the words people use
for it. Everything else about scoping already exists: ``allowed_tables`` is
threaded through the planner, the value index, the workspace guide and the
clarifier, so a domain attaches a name to a mechanism rather than adding a
second one beside it.
"""

from __future__ import annotations

import json
from typing import Any

from store.db import get_db


def _row(row) -> dict[str, Any]:
    item = dict(row)
    try:
        item["tables"] = json.loads(item.pop("tables_json", "[]") or "[]")
    except Exception:
        item["tables"] = []
    return item


def save_domain(
    account_id: str,
    name: str,
    *,
    tables: list[str] | None = None,
    description: str = "",
    synonyms: str = "",
) -> int:
    """Create or replace one domain. Table names are stored uppercase.

    Retrieval, the planner and the value index all key on uppercase
    fully-qualified names, and a domain whose members are stored in the case
    an admin happened to type would match none of them -- the quietest kind
    of bug, because every lookup misses and nothing raises.
    """
    clean = str(name or "").strip()
    if not clean:
        raise ValueError("A domain name is required.")
    members = sorted({
        str(t).strip().upper() for t in (tables or []) if str(t).strip()
    })
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO domain
                (account_id, name, description, tables_json, synonyms, updated_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(account_id, name) DO UPDATE SET
                description = excluded.description,
                tables_json = excluded.tables_json,
                synonyms    = excluded.synonyms,
                is_active   = 1,
                updated_at  = datetime('now')
            """,
            (account_id, clean, str(description or ""),
             json.dumps(members), str(synonyms or "")),
        )
        row = conn.execute(
            "SELECT id FROM domain WHERE account_id=? AND name=?",
            (account_id, clean),
        ).fetchone()
    return int(row["id"]) if row else 0


def list_domains(account_id: str, *, active_only: bool = True) -> list[dict]:
    where = "WHERE account_id=?" + (" AND is_active=1" if active_only else "")
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT * FROM domain {where} ORDER BY name", (account_id,),
        ).fetchall()
    return [_row(row) for row in rows]


def get_domain(account_id: str, name: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM domain WHERE account_id=? AND name=?",
            (account_id, str(name or "").strip()),
        ).fetchone()
    return _row(row) if row else None


def delete_domain(account_id: str, name: str) -> bool:
    """Deactivate rather than delete, so an answer that cited it still resolves."""
    with get_db() as conn:
        cur = conn.execute(
            "UPDATE domain SET is_active=0, updated_at=datetime('now') "
            "WHERE account_id=? AND name=?",
            (account_id, str(name or "").strip()),
        )
    return bool(cur.rowcount)
