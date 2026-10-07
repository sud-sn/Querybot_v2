"""Questions' SQL through QueryBot's governed executor, as every answer's SQL goes.

core2 writes the SQL; QueryBot's governance decides whether it may run and what
comes back: the reader's allowed tables, row policies, masking of classified
columns, aggregate-only fields, the decision log. Nothing core2 runs for a
reader bypasses it. (Bootstrap profiling, which returns only aggregates and runs
under the workspace's service connection, uses :mod:`core2.warehouse.querybot`.)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from core2.warehouse import dialect as D
from core2.warehouse.runner import QueryResult, assert_read_only


@dataclass
class GovernedWarehouse:
    account_id: str
    user: dict[str, Any] | None
    db_config: dict[str, Any]
    known_tables: set[str]
    allowed_tables: set[str] | None = None
    channel: str = "portal"
    log: list[str] = field(default_factory=list)

    @property
    def db_type(self) -> str:
        return str(self.db_config.get("db_type") or "")

    @property
    def dialect(self) -> str:
        return D.for_db_type(self.db_type)

    def query(self, sql: str, *, max_rows: int | None = None) -> QueryResult:
        from core.compliance.governed_query import execute_governed_query
        from core.compliance.policy_engine import resolve_context

        assert_read_only(sql, self.dialect)
        context = resolve_context(self.account_id, self.user, action="query_execution", channel=self.channel)
        start = time.perf_counter()
        self.log.append(sql)
        governed = execute_governed_query(
            self.db_config["credentials"], self.db_type, sql, context=context, known_tables=self.known_tables,
            allowed_tables=self.allowed_tables, semantic_context={"production_sql": True},
            max_rows=max_rows or 5000)
        rows = list(governed.rows or [])
        columns = list(rows[0].keys()) if rows else _columns_of(sql, self.dialect)
        return QueryResult(columns=columns, rows=[tuple(r.get(c) for c in columns) for r in rows],
                           truncated=bool(getattr(governed, "truncated", False)),
                           elapsed_ms=(time.perf_counter() - start) * 1000)


def _columns_of(sql: str, dialect: str) -> list[str]:
    """The output names of a query that returned no rows (from its outermost SELECT)."""
    import sqlglot

    tree = sqlglot.parse_one(sql, read=dialect)
    return [e.alias_or_name for e in getattr(tree, "expressions", [])]
