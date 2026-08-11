"""Small, dependency-free helpers for semantic-plan enforcement.

Semantic plans deliberately retain losing field and join candidates as
``enforcement="optional"`` hints.  Those hints must never be promoted back to
required tables by retrieval, graph reconciliation, or recovery prompts.
Keeping this rule in one helper prevents each pipeline stage from interpreting
the same plan differently.
"""

from __future__ import annotations

from typing import Any


def required_semantic_tables(plan: dict[str, Any] | None) -> set[str]:
    """Return only the tables the plan actually requires.

    Modern plans carry fields and joins with explicit enforcement.  When those
    structures exist, they are the source of truth and optional entries are
    excluded.  The ``required_tables`` list is used only for older/minimal
    plans that contain no structured fields or joins.
    """
    plan = plan or {}
    fields = [item for item in (plan.get("fields") or []) if isinstance(item, dict)]
    joins = [item for item in (plan.get("joins") or []) if isinstance(item, dict)]

    if fields or joins:
        tables = {
            str(field.get("table") or "")
            for field in fields
            if field.get("enforcement") != "optional" and field.get("table")
        }
        for join in joins:
            if join.get("enforcement") == "optional":
                continue
            tables.update(
                str(join.get(key) or "")
                for key in ("from", "to", "from_table", "to_table")
                if join.get(key)
            )
        return {table for table in tables if table}

    return {
        str(table)
        for table in (plan.get("required_tables") or [])
        if table
    }
