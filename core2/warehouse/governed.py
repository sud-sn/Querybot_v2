"""Questions' SQL through QueryBot's governed executor, as every answer's SQL goes.

core2 writes the SQL; QueryBot's governance decides whether it may run and what
comes back: the reader's allowed tables, row policies, masking of classified
columns, aggregate-only fields, the decision log. Nothing core2 runs for a
reader bypasses it. (Bootstrap profiling, which returns only aggregates and runs
under the workspace's service connection, uses :mod:`core2.warehouse.querybot`.)

People's data Learn found (``personal``: names, and details such as emails) is
masked here too, in a workspace under compliance, whether or not the workspace's
classifications name it (Learn reads an email column called C07 by its values;
a classification written by its name would miss it). A reader who has signed the
confidentiality attestation sees the values as stored, and each such release is
written to the decision log; when it cannot be written, the values stay masked.
A name becomes a stable alias ("Patient K-3F2") and a detail a stable token, so a
grouping by them still keeps each person apart; counts and sums are never masked.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from core2.warehouse import dialect as D
from core2.warehouse.runner import QueryResult, assert_read_only

log = logging.getLogger(__name__)
_STRATEGY = {"name": "safe_alias_name", "detail": "tokenize"}


@dataclass
class GovernedWarehouse:
    account_id: str
    user: dict[str, Any] | None
    db_config: dict[str, Any]
    known_tables: set[str]
    allowed_tables: set[str] | None = None
    channel: str = "portal"
    log: list[str] = field(default_factory=list)
    # (table, column) casefolded -> "name" | "detail": people's data, masked for a reader not cleared to see it.
    personal: dict[tuple[str, str], str] = field(default_factory=dict)
    released: bool = False     # people's data went out as stored, to a cleared reader (kept with the answer)

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
        if self.personal and rows:
            rows = self._protect(rows, governed, context)
        columns = list(rows[0].keys()) if rows else _columns_of(sql, self.dialect)
        return QueryResult(columns=columns, rows=[tuple(r.get(c) for c in columns) for r in rows],
                           truncated=bool(getattr(governed, "truncated", False)),
                           elapsed_ms=(time.perf_counter() - start) * 1000)


    def _protect(self, rows: list[dict], governed: Any, context: Any) -> list[dict]:
        """People's data in the rows masked, unless the reader is cleared to see it (and the release is logged)."""
        import store
        from core.compliance.result_guard import _mask

        analysis = governed.analysis
        exempt = {str(o).casefold() for o in (getattr(analysis, "mask_exempt_outputs", None) or set())}
        masked = [str(k).upper() for k in (getattr(governed.decision, "masking", None) or {})]

        def masked_already(source: str) -> bool:
            # The same column, however much of its database and schema each side names.
            source = source.upper()
            return any(source == k or source.endswith("." + k) or k.endswith("." + source) for k in masked)

        touched: dict[str, tuple[str, str]] = {}          # output -> (name | detail, its source)
        for output, sources in (getattr(analysis, "lineage", None) or {}).items():
            if str(output).casefold() in exempt:
                continue                                   # a count or a sum of them, never a person
            for source in sources:
                parts = str(source).split(".")
                kind = self.personal.get((parts[-2].casefold(), parts[-1].casefold())) if len(parts) >= 2 else None
                if kind and not masked_already(str(source)):      # the workspace's own masking already did it
                    touched[str(output).casefold()] = (kind, str(source))
                    break
        if not touched:
            return rows
        if store.user_attestation_valid(self.account_id, context.user_id) and _covers_people(
                self.account_id, context.user_id):
            try:
                store.log_policy_decision(
                    account_id=self.account_id, user_id=context.user_id, action="result_release",
                    purpose_id=context.purpose_id, channel=context.channel, allowed=True,
                    reason_code="attested_unmasked_release",
                    resources=sorted({source for _, source in touched.values()}),
                    obligations={"masking_waived": {source: _STRATEGY[kind] for kind, source in touched.values()}},
                    policy_version=context.policy_version or 0)
                self.released = True
                return rows
            except Exception as exc:  # noqa: BLE001 - a release with no audit row stays masked
                log.warning("core2: the unmasked release for %s could not be logged; masking: %s",
                            self.account_id, exc)
        out = []
        for row in rows:
            item = dict(row)
            for key in item:
                hit = touched.get(str(key).casefold())
                if hit:
                    item[key] = _mask(item[key], _STRATEGY[hit[0]], self.account_id, source=hit[1])
            out.append(item)
        return out


def _covers_people(account_id: str, user_id: str) -> bool:
    """Does the reader's attestation cover people's data (every class, or PII or PHI)? Unreadable, or none
    valid any more (it ended since it was checked): no."""
    import store

    try:
        scope = store.user_attestation_scope(account_id, user_id)
    except Exception:  # noqa: BLE001 - a scope that cannot be read releases nothing
        return False
    return scope is not None and ("*" in scope or bool(scope & {"PII", "PHI"}))


def _columns_of(sql: str, dialect: str) -> list[str]:
    """The output names of a query that returned no rows (from its outermost SELECT)."""
    import sqlglot

    tree = sqlglot.parse_one(sql, read=dialect)
    return [e.alias_or_name for e in getattr(tree, "expressions", [])]
