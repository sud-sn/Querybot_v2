"""Run a metric formula against the live database and report whether it binds.

This is the only check that catches a formula which validates cleanly but does
not actually execute -- a column that exists in ``_schema.json`` but not in the
database, a join alias that never resolves, a type mismatch the parser cannot
see.  ``core.metric_validator.validate_metric`` is a pure parse and knows none
of that.

Lifted verbatim out of ``admin/routes.py::metrics_test_formula``, where it was
reachable only by an admin clicking "Test formula".  A metric composed by the
bot has no one to click that button, so the check has to be callable.

THE PROBE ORDER IS LOAD-BEARING.  ``join_probe`` is tried before the
multi-table token heuristic, because a row-calculated or date-gap formula
references join aliases (``due_dt.DMS_DT``) that only exist once the metric's
own ``required_joins`` are applied.  The token heuristic sees two table names
and splits into per-table column probes, which can never bind an alias --
"multi-part identifier could not be bound".  Reversing these two branches
reintroduces that bug.

``DryRunOutcome.value`` is the probe's actual scalar result.  It is real
customer data and it never passes the compliance boundary, because a probe is
not a governed query -- ``core.compliance.result_guard.protect_rows`` does not
run on it.  Admin surfaces may show it; a portal surface must read ``status``
and ``detail`` only.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import store

log = logging.getLogger("querybot.metric_dryrun")

# Identifiers that are SQL, not column names.  Kept as the route had it -- this
# list only has to be good enough to guess which tables a formula touches; a
# miss costs a redundant probe, never a wrong verdict.
_SQL_KW = {
    "SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "IN", "IS", "NULL", "AS", "CASE",
    "WHEN", "THEN", "ELSE", "END", "BETWEEN", "LIKE", "TOP", "LIMIT", "DISTINCT",
    "WITH", "NOLOCK", "SUM", "AVG", "COUNT", "MIN", "MAX", "COALESCE", "NULLIF",
    "ISNULL", "NVL", "CAST", "CONVERT", "ROUND", "ABS", "FLOOR", "CEILING", "IF",
    "IIF", "DATEDIFF", "DATEADD", "DATE", "LEFT", "RIGHT", "MID", "LEN", "TRIM",
    "UPPER", "LOWER", "REPLACE", "SUBSTRING", "CONCAT", "ROWNUM",
}
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")

_PROBE_TIMEOUT_SECONDS = 20
_COLUMN_PROBE_TIMEOUT_SECONDS = 15


@dataclass(frozen=True)
class DryRunOutcome:
    """What happened when the formula was run against the database."""

    status: str                       # "ok" | "error" | "skipped"
    detail: str = ""                  # safe to show anyone: names tables, never values
    probe_kind: str = "none"          # join_probe | single_table | multi_table | none
    tables_probed: tuple[str, ...] = ()
    value: Any = None                 # ADMIN ONLY -- real data, never sent to a portal user
    # Filter columns whose value matched no rows. A formula that BINDS and a
    # formula that MATCHES are different claims, and only the first was ever
    # checked -- see check_filter_matches.
    empty_filters: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self, *, include_value: bool = False) -> dict[str, Any]:
        """Serialisable form. ``include_value`` is opt-in for exactly this reason."""
        payload = {
            "status": self.status,
            "detail": self.detail,
            "probe_kind": self.probe_kind,
            "tables_probed": list(self.tables_probed),
        }
        if include_value:
            payload["value"] = self.value
        return payload


def _skipped(detail: str) -> DryRunOutcome:
    return DryRunOutcome(status="skipped", detail=detail)


def _error(detail: str, *, probe_kind: str = "none", tables: tuple[str, ...] = ()) -> DryRunOutcome:
    return DryRunOutcome(status="error", detail=detail, probe_kind=probe_kind, tables_probed=tables)


def _fqn_to_sql(fqn: str, db_type: str) -> str:
    """Dialect-quoted table reference, or "" when the FQN is too short to qualify."""
    parts = str(fqn or "").split(".")
    if db_type == "azure_sql" and len(parts) >= 2:
        return f"[{parts[-2]}].[{parts[-1]}]"
    if db_type == "snowflake" and len(parts) >= 3:
        return f'"{parts[0]}"."{parts[1]}"."{parts[2]}"'
    if db_type == "oracle" and len(parts) >= 2:
        return f'"{parts[-2]}"."{parts[-1]}"'
    return ""


def _table_sql(fqn: str, db_type: str) -> str:
    """As ``_fqn_to_sql`` but falls back to the raw FQN rather than "".

    The route kept two near-identical helpers that differ only in this
    fallback; both are preserved because the caller of each depends on the
    difference -- the anchor lookup needs "" to mean "keep looking".
    """
    return _fqn_to_sql(fqn, db_type) or str(fqn or "")


def _load_schema_master(account_id: str) -> dict[str, Any]:
    state = store.get_client_state(account_id)
    schema_dir = (state or {}).get("schema_dir") or ""
    if not schema_dir:
        return {}
    path = Path(schema_dir) / "_schema.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        log.warning("Dry run could not read the schema manifest for %s: %s", account_id, exc)
        return {}


def _tables_for_formula(formula: str, master: dict[str, Any]) -> dict[str, list[str]]:
    """Map each table the formula appears to touch to the columns it uses.

    Deliberately name-based: a formula is an expression fragment, not parseable
    SQL on its own, so there is nothing to hand a real parser here.
    """
    tokens = [
        token.upper() for token in _IDENTIFIER_RE.findall(formula or "")
        if token.upper() not in _SQL_KW
    ]
    column_to_tables: dict[str, list[str]] = {}
    for fqn, table in (master or {}).items():
        if not isinstance(table, dict):
            # "__db_fk_constraints__" and friends are lists, not tables.
            continue
        for column in (table.get("columns") or []):
            name = str(
                (column.get("name") if isinstance(column, dict) else column) or ""
            ).upper()
            if name:
                column_to_tables.setdefault(name, []).append(fqn)

    table_to_columns: dict[str, list[str]] = {}
    for token in tokens:
        if token in column_to_tables:
            table_to_columns.setdefault(column_to_tables[token][0], []).append(token)
    return table_to_columns


def _wants_join_probe(formula: str, metric_builder_config: str) -> bool:
    """Whether this formula references join aliases that need the joins applied."""
    config_raw = (metric_builder_config or "").strip()
    if not config_raw or formula.upper().startswith("SELECT"):
        return False
    try:
        config = json.loads(config_raw)
    except Exception:
        return False
    return bool(
        isinstance(config, dict)
        and config.get("mode") in ("row_calculated", "date_gap")
        and config.get("required_joins")
    )


def _filters_from_config(metric_builder_config: str) -> list[dict[str, Any]]:
    """Every structured filter in a builder config, whatever its mode."""
    try:
        config = json.loads((metric_builder_config or "").strip() or "{}")
    except Exception:
        return []
    if not isinstance(config, dict):
        return []
    filters: list[dict[str, Any]] = []
    for holder in (config, config.get("numerator"), config.get("denominator")):
        if isinstance(holder, dict):
            for item in holder.get("filters") or []:
                if isinstance(item, dict) and item.get("field"):
                    filters.append(item)
    return filters


async def check_filter_matches(
    account_id: str,
    *,
    metric_builder_config: str,
    timeout: int = _COLUMN_PROBE_TIMEOUT_SECONDS,
) -> tuple[str, ...]:
    """Return the filter columns whose value matches no rows at all.

    The dry run proves a formula BINDS: the columns exist, the types work, the
    joins resolve. It cannot prove the formula MATCHES anything, and that gap
    matters precisely because a composed metric's filter VALUES are the one part
    the model is guessing. It is shown column names, never their contents, so
    "active" becomes ACT_FLG = 'Y' or 'true' or 1 depending on the day -- all
    three bind, one at most is right, and the wrong ones return a confident
    number computed over nothing.

    A filter matching zero rows is not proof of a wrong guess (a category can be
    legitimately empty today), but when the value was guessed rather than known
    it is much the likeliest explanation, so the caller treats it as one and
    asks.

    Returns column names only. The values live in the database and stay there:
    a probe does not pass through the compliance boundary, so nothing it sees
    may be shown to a portal user.
    """
    filters = _filters_from_config(metric_builder_config)
    if not filters:
        return ()

    client = store.get_client(account_id)
    if not client or not client.get("db_config_id"):
        return ()
    raw_cfg = store.get_db_config(client["db_config_id"])
    if not raw_cfg:
        return ()
    db_type = raw_cfg.get("db_type", "azure_sql")
    creds = raw_cfg.get("credentials", {})
    master = _load_schema_master(account_id)
    if not master:
        return ()

    column_to_table: dict[str, str] = {}
    for fqn, table in master.items():
        if not isinstance(table, dict):
            continue
        for column in (table.get("columns") or []):
            name = str((column.get("name") if isinstance(column, dict) else column) or "").upper()
            if name and name not in column_to_table:
                column_to_table[name] = fqn

    try:
        from core.metric_builder import _compile_condition
        from core.schema import _az_connect, _sf_connect, _ora_connect
    except Exception:
        return ()

    def _get_conn():
        if db_type == "azure_sql":
            return _az_connect(creds)
        if db_type == "snowflake":
            return _sf_connect(creds)
        return _ora_connect(creds)

    def _count(sql: str) -> int:
        conn = _get_conn()
        try:
            cur = conn.cursor()
            cur.execute(sql)
            row = cur.fetchone()
            return int(row[0]) if row and row[0] is not None else 0
        finally:
            conn.close()

    loop = asyncio.get_running_loop()
    empty: list[str] = []
    for filt in filters:
        field = str(filt.get("field") or "").strip()
        table_fqn = column_to_table.get(field.upper())
        if not field or not table_fqn:
            continue
        try:
            predicate = _compile_condition(
                field, str(filt.get("operator") or "equals"), str(filt.get("value") or ""),
            )
        except Exception:
            continue
        probe = f"SELECT COUNT(*) FROM {_table_sql(table_fqn, db_type)} WHERE {predicate}"
        if db_type == "azure_sql":
            probe = (
                f"SELECT COUNT(*) FROM {_table_sql(table_fqn, db_type)} "
                f"WITH (NOLOCK) WHERE {predicate}"
            )
        try:
            matched = await asyncio.wait_for(
                loop.run_in_executor(None, _count, probe), timeout=timeout,
            )
        except Exception as exc:
            # A probe that cannot run is not evidence of an empty filter.
            log.debug("Filter probe failed for %s: %s", field, exc)
            continue
        if matched == 0 and field.upper() not in {c.upper() for c in empty}:
            empty.append(field)
    return tuple(empty)


async def dry_run_metric_formula(
    account_id: str,
    *,
    formula: str,
    base_table: str = "",
    metric_builder_config: str = "",
    timeout: int = _PROBE_TIMEOUT_SECONDS,
) -> DryRunOutcome:
    """Execute ``formula`` against the account's database and report the outcome.

    Returns ``skipped`` -- never ``error`` -- when the account simply is not set
    up to be probed (no database, no discovered schema). A caller gating on
    ``outcome.ok`` should treat that as "unverified", not "broken".
    """
    formula = (formula or "").strip()
    if not formula:
        return _error("Formula is empty")

    client = store.get_client(account_id)
    if not client:
        return _skipped("Account not found")
    db_cfg_id = client.get("db_config_id")
    if not db_cfg_id:
        return _skipped("No database configured for this account")
    raw_cfg = store.get_db_config(db_cfg_id)
    if not raw_cfg:
        return _skipped("Database config not found")

    db_type = raw_cfg.get("db_type", "azure_sql")
    creds = raw_cfg.get("credentials", {})

    base_table_raw = (base_table or "").strip()
    join_probe = _wants_join_probe(formula, metric_builder_config)

    master = _load_schema_master(account_id)
    first_table_sql = _fqn_to_sql(base_table_raw, db_type) if base_table_raw else ""
    if not first_table_sql:
        for fqn in master:
            first_table_sql = _fqn_to_sql(fqn, db_type)
            if first_table_sql:
                break
    if not master and not first_table_sql:
        return _skipped("No tables found in schema — run discovery first")

    table_to_columns = _tables_for_formula(formula, master)
    multi_table = len(table_to_columns) > 1
    single_table_fqn = next(iter(table_to_columns)) if len(table_to_columns) == 1 else None

    try:
        from core.schema import _az_connect, _sf_connect, _ora_connect

        def _get_conn():
            if db_type == "azure_sql":
                return _az_connect(creds)
            if db_type == "snowflake":
                return _sf_connect(creds)
            return _ora_connect(creds)

        def _run_probe(sql: str):
            conn = _get_conn()
            try:
                cur = conn.cursor()
                cur.execute(sql)
                row = cur.fetchone()
                return row[0] if row else None
            finally:
                conn.close()

        loop = asyncio.get_running_loop()

        # ── Join probe FIRST. See the module docstring: the multi-table branch
        # below cannot bind a join alias, and would misdiagnose this formula.
        if join_probe:
            if not base_table_raw or not first_table_sql:
                return _error(
                    "This metric's formula uses join aliases, so the test needs the "
                    "Base table — set the Base table field (Advanced options) first.",
                    probe_kind="join_probe",
                )
            from core.pipeline_helpers import _build_row_metric_join_sql

            # No "AS" in the skeleton — the helper reads the last token of the
            # FROM line as the anchor alias.
            join_sql = _build_row_metric_join_sql(
                [{"metric_builder_config": metric_builder_config}],
                db_type,
                f"FROM {first_table_sql} base",
            )
            if db_type == "azure_sql":
                probe = (
                    f"SELECT TOP 1 ({formula}) AS _result "
                    f"FROM {first_table_sql} AS base WITH (NOLOCK)\n{join_sql}"
                )
            elif db_type == "snowflake":
                probe = f"SELECT ({formula}) AS _result FROM {first_table_sql} base\n{join_sql}\nLIMIT 1"
            else:
                probe = (
                    f"SELECT ({formula}) AS _result FROM {first_table_sql} base\n"
                    f"{join_sql}\nWHERE ROWNUM <= 1"
                )

            result = await asyncio.wait_for(
                loop.run_in_executor(None, _run_probe, probe), timeout=timeout,
            )
            if result is not None and not isinstance(result, (int, float, str, bool)):
                result = str(result)
            return DryRunOutcome(
                status="ok",
                detail=f"Formula binds against {base_table_raw or 'the base table'}.",
                probe_kind="join_probe",
                tables_probed=(base_table_raw,) if base_table_raw else (),
                value=result,
            )

        if multi_table:
            # The formula spans tables, so it cannot be evaluated as one
            # expression. Prove each table has the columns it needs instead.
            probe_results: list[str] = []
            all_ok = True
            for fqn, columns in table_to_columns.items():
                column_list = ", ".join(columns)
                tbl_sql = _table_sql(fqn, db_type)
                if db_type == "azure_sql":
                    probe = f"SELECT TOP 1 {column_list} FROM {tbl_sql} WITH (NOLOCK)"
                elif db_type == "snowflake":
                    probe = f"SELECT {column_list} FROM {tbl_sql} LIMIT 1"
                else:
                    probe = f"SELECT {column_list} FROM {tbl_sql} WHERE ROWNUM <= 1"
                try:
                    await asyncio.wait_for(
                        loop.run_in_executor(None, _run_probe, probe),
                        timeout=_COLUMN_PROBE_TIMEOUT_SECONDS,
                    )
                    probe_results.append(f"✓ {fqn.split('.')[-1]} ({column_list})")
                except Exception as exc:
                    probe_results.append(f"✗ {fqn.split('.')[-1]} ({column_list}): {exc}")
                    all_ok = False

            summary = (
                f"Multi-table formula — {len(table_to_columns)} tables probed:\n"
                + "\n".join(probe_results)
            )
            return DryRunOutcome(
                status="ok" if all_ok else "error",
                detail=summary,
                probe_kind="multi_table",
                tables_probed=tuple(table_to_columns),
            )

        tbl_sql = _table_sql(single_table_fqn, db_type) if single_table_fqn else first_table_sql
        if not tbl_sql:
            return _skipped("No tables found in schema — run discovery first")
        if db_type == "azure_sql":
            probe = f"SELECT TOP 1 ({formula}) AS _result FROM {tbl_sql} WITH (NOLOCK)"
        elif db_type == "snowflake":
            probe = f"SELECT ({formula}) AS _result FROM {tbl_sql} LIMIT 1"
        else:
            probe = f"SELECT ({formula}) AS _result FROM {tbl_sql} WHERE ROWNUM <= 1"

        result = await asyncio.wait_for(
            loop.run_in_executor(None, _run_probe, probe), timeout=timeout,
        )
        if result is not None and not isinstance(result, (int, float, str, bool)):
            result = str(result)
        _probed = (single_table_fqn,) if single_table_fqn else ()
        return DryRunOutcome(
            status="ok",
            detail=f"Formula evaluates against {single_table_fqn or 'the base table'}.",
            probe_kind="single_table",
            tables_probed=_probed,
            value=result,
        )

    except asyncio.TimeoutError:
        return _error(f"Query timed out ({timeout} s)")
    except Exception as exc:
        return _error(str(exc))
