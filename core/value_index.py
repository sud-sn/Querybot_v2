"""
core/value_index.py

Per-client index of filterable column values for literal grounding.

Schema discovery only captures distinct values for categorical-looking columns
and silently drops anything with more than ~30 distinct values — so customer
names, item descriptions, and other high-cardinality filter columns have zero
value representation in the KB. When a user asks "sales for Emco corp" the LLM
must guess the WHERE literal; the data may say 'EMCO Corporation' and the
query returns zero rows with no explanation.

This module builds a SQLite value index at discovery time
(clients/{account_id}/value_index.sqlite) and provides millisecond lookups so
the query pipeline can resolve user-typed literals to exact database values
BEFORE the LLM writes SQL (core/value_resolver.py) and explain unmatched
literals after a zero-row result (core/answer_rca.py).

Privacy layers (values from masked/PII columns must never be indexed):
  1. Columns listed in the table's `masked_fields` (and tables with
     mask_mode == "all") from _schema.json are skipped.
  2. core.masking.detect_sensitive_columns name-pattern hits are skipped.
  3. Harvested values are scanned with core.masking's value-level PII check;
     a column whose values look like PII is dropped entirely.
  4. Values longer than 200 chars or containing newlines are rejected
     (free-text content, and a prompt-injection guard — indexed values are
     later quoted into LLM prompts).

The index lives beside column_context.json, so the existing client-reset
flow (which deletes clients/{account_id}/ wholesale) cleans it up for free.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("querybot.value_index")

_DEFAULT_BASE_DIR = "clients"
_INDEX_FILENAME = "value_index.sqlite"

# Value hygiene: anything longer or multi-line is free text, not a filter value.
_MAX_VALUE_LEN = 200

# Fuzzy thresholds — aligned with core/clarification.py's typo resolver.
FUZZY_VERIFIED = 0.87
FUZZY_CANDIDATE = 0.75

# Suffix roles from core/naming_convention.py that mark filterable display text.
# Gated to dimension tables — indexing every _NM/_DSC column on wide fact
# tables would balloon the index with row-level text.
_FILTERABLE_ROLES = {"display", "code"}
# State/classification roles are low-cardinality by nature and live on FACT
# tables as often as dimensions (order status on the order fact), so they are
# indexed regardless of table classification. The _STS naming rule itself
# says "Use in WHERE to filter by state. Check distinct values in the KB for
# valid codes" — without indexing them, "how many orders are cancelled" gets
# no value grounding and a wrong status literal gets no zero-row explanation.
_FILTERABLE_ANY_TABLE_ROLES = {"status", "type", "group"}


# ── Paths / normalization ─────────────────────────────────────────────────────

def _index_path(account_id: str, base_dir: str = _DEFAULT_BASE_DIR) -> Path:
    return Path(base_dir) / (account_id or "") / _INDEX_FILENAME


def normalize_value(value: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. Shared by the
    builder, the question-time resolver, and the zero-row RCA matcher so all
    three agree on what 'the same value' means."""
    text = (value or "").lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def value_index_enabled(state_data: dict | None) -> bool:
    """Per-client opt-out via client state_data; default ON."""
    if not isinstance(state_data, dict):
        return True
    flag = state_data.get("value_index_enabled")
    if flag is None:
        return True
    return bool(flag) and str(flag).lower() not in ("0", "false", "no", "off")


# ── Column selection ──────────────────────────────────────────────────────────

def _is_dimension_table(bare_table: str) -> bool:
    upper = (bare_table or "").upper()
    return upper.endswith(("_DMS", "_DIM")) or upper.startswith(("DIM_", "DMS_", "D_"))


def _is_string_type(col_type: str) -> bool:
    from core.schema import _CATEGORICAL_TYPES
    base = (col_type or "").lower().split("(")[0].strip()
    return any(t in base for t in _CATEGORICAL_TYPES)


def _clearance_gate(account_id: str) -> tuple[Callable[[str, str], bool] | None, str]:
    """Return (predicate, industry) governing what may be written to the index.

    The predicate mirrors ``_cleared`` in core/value_resolver.py, which decides
    what a regulated tenant may SEE. Until now nothing decided what a regulated
    tenant may STORE: this module imported nothing from core.compliance, so an
    admin-reviewed PHI classification did not stop the harvest, and up to
    per_column_cap real values per column were written to a plain file on disk.

    Deliberately STRICTER than the query-time rule in one place: a column with
    no classification row at all is refused. At query time an unclassified
    column merely fails to be cleared; here it would be persisted, and a
    durable store must not inherit a fail-open default.

    Returns (None, industry) for an unregulated tenant — no gate, unchanged
    behaviour. Raises nothing: a tenant whose compliance state cannot be read
    is treated as regulated-with-no-pack, which refuses everything.
    """
    try:
        from core.compliance.policy_engine import is_regulated
        if not is_regulated(account_id):
            return None, ""
    except Exception:
        log.warning("value index: compliance state unreadable for %s; refusing to index",
                    account_id, exc_info=True)
        return (lambda _t, _c: False), ""

    try:
        import store
        from core.compliance.packs import get_pack

        profile = store.get_compliance_profile(account_id) or {}
        industry = str(profile.get("industry") or "")
        # get_pack is keyed by policy_pack_key, not by industry.
        pack = get_pack(str(profile.get("policy_pack_key") or "")) or {}
        sensitive = {str(t).upper() for t in (pack.get("sensitive_tags") or [])}
        classifications = store.get_classification_map(account_id) or {}
    except Exception:
        log.warning("value index: policy pack unreadable for %s; refusing to index",
                    account_id, exc_info=True)
        return (lambda _t, _c: False), ""

    if not sensitive:
        # A regulated tenant with no resolvable pack has no definition of
        # "sensitive", so nothing can be cleared as safe. Same reading as the
        # query path takes, for the same reason.
        log.warning("value index: regulated tenant %s has no resolvable policy pack "
                    "(policy_pack_key=%r); indexing nothing",
                    account_id, profile.get("policy_pack_key"))
        return (lambda _t, _c: False), industry

    def _cleared(table_fqn: str, column: str) -> bool:
        row = classifications.get(f"{str(table_fqn).upper()}.{str(column).upper()}")
        if not row or not row.get("reviewed"):
            return False
        tags = {str(t).upper() for t in (row.get("tags") or [])}
        return not (tags & sensitive)

    return _cleared, industry


def purge_uncleared_columns(account_id: str, base_dir: str = _DEFAULT_BASE_DIR) -> dict:
    """Re-apply the clearance gate to an index that already exists.

    Building the index is the only moment the gate ran, so a column reclassified
    as PHI after the build kept its values on disk indefinitely: the compliance
    profile save rewrote classifications and never touched this file, making it
    the one artifact where correcting a classification had no effect.

    Called after any change to a tenant's posture or classifications. Deletes
    rows only -- it can never add a value the gate would refuse -- so it is safe
    to run on every such change, including ones that turn out to be no-ops.
    """
    path = _index_path(account_id, base_dir)
    if not path.exists():
        return {"purged_columns": 0, "purged_values": 0, "reason": "no_index"}

    cleared, _industry = _clearance_gate(account_id)
    if cleared is None:
        return {"purged_columns": 0, "purged_values": 0, "reason": "tenant_not_regulated"}

    conn = sqlite3.connect(path)
    try:
        pairs = conn.execute(
            "SELECT DISTINCT table_fqn, column_name FROM column_value"
        ).fetchall()
        purged_columns = purged_values = 0
        for table_fqn, column_name in pairs:
            if cleared(table_fqn, column_name):
                continue
            cur = conn.execute(
                "DELETE FROM column_value WHERE table_fqn=? AND column_name=?",
                (table_fqn, column_name),
            )
            purged_columns += 1
            purged_values += cur.rowcount or 0
        if purged_columns:
            conn.commit()
            # Reclaim the pages so the removed values are not still readable in
            # the file's free list. A delete that leaves the plaintext on disk
            # is not a retraction.
            conn.execute("VACUUM")
            log.warning(
                "value index purge for %s: removed %d columns (%d values) that are "
                "no longer cleared", account_id, purged_columns, purged_values,
            )
        return {
            "purged_columns": purged_columns,
            "purged_values": purged_values,
            "reason": "applied",
        }
    finally:
        conn.close()


def select_filterable_columns(
    schema: dict, vocab=None, *, account_id: str = "", industry: str = "",
    cleared: Callable[[str, str], bool] | None = None,
) -> list[dict]:
    """
    Choose columns worth value-indexing from a normalized _schema.json dict.

    Included: string-typed columns whose naming role is display/code
    (_NM/_NAME/_DSC/_DESC/_CD/_CODE …) on dimension-classified tables, plus
    categorical columns discovery already scans (so the RCA matcher covers
    them uniformly). Excluded: every masking signal (see module docstring),
    and — for a regulated tenant — everything `cleared` does not clear.

    `industry` reaches detect_sensitive_columns, whose third pass runs the
    banking/healthcare classifier and only fires when it is set. Omitting it
    silently disabled that pass for every regulated tenant.
    """
    from core.masking import detect_sensitive_columns
    from core.naming_convention import match_column_suffix
    from core.schema import _is_categorical

    selected: list[dict] = []
    for fqn, meta in (schema or {}).items():
        if str(fqn).startswith("__") or not isinstance(meta, dict):
            continue
        if (meta.get("mask_mode") or "") == "all":
            continue
        columns = meta.get("columns") or []
        col_defs = [c for c in columns if isinstance(c, dict) and c.get("name")]
        masked = {str(f).upper() for f in (meta.get("masked_fields") or [])}
        sensitive = {str(c).upper() for c in detect_sensitive_columns(col_defs, industry)}

        # NOT expanded before this check, and that is a decision rather than an
        # oversight. Expanding first looks like the obvious fix for CUS_NM and
        # PAT_NM slipping past patterns written in spelled-out English -- but
        # the bare-name rule is `(?<![a-z])name(?![a-z])`, so "warehouse name",
        # "region name" and "profit centre name" all match it too. Measured on
        # the EMCO mart, expansion blocked WHS_NM, RGN_NM, PFT_CTR_NM and
        # ITM_DSC: every dimension display column, which is the whole index.
        #
        # The abbreviated-PHI risk it was meant to address is already closed
        # one layer up: a regulated tenant indexes only columns an admin has
        # reviewed and cleared (see _clearance_gate), so PAT_NM is refused on
        # classification, not on spelling. For an unregulated tenant, customer
        # and warehouse names are ordinary business data and indexing them is
        # the feature working.

        parts = str(fqn).split(".")
        bare_table = parts[-1]
        for col in col_defs:
            name = str(col.get("name") or "")
            ctype = str(col.get("type") or "")
            upper = name.upper()
            # A reviewed classification is a human decision about this exact
            # column; `sensitive` is a regex over its name. Where both have an
            # opinion the human wins, or the naming heuristic silently vetoes
            # the admin -- which is what kept a pharmacy's drug catalog out of
            # the index (GENERIC_NAME reads as "drug name" to the pattern) and
            # so kept value grounding from working on the one column the
            # question was about.
            #
            # This only ever RE-ADMITS a column the gate has already cleared,
            # so it cannot widen anything for an unregulated tenant, where
            # `cleared` is None and the heuristic remains the only check.
            admin_cleared = cleared is not None and cleared(str(fqn), name)
            if upper in masked and not admin_cleared:
                continue
            if upper in sensitive and not admin_cleared:
                continue
            if not _is_string_type(ctype):
                continue

            rule = match_column_suffix(name)
            is_display = bool(rule and rule.role in _FILTERABLE_ROLES) and _is_dimension_table(bare_table)
            is_state = bool(rule and rule.role in _FILTERABLE_ANY_TABLE_ROLES)
            is_cat = _is_categorical(name, ctype)
            if not (is_display or is_state or is_cat):
                continue
            # The write-time gate. Last check before a column becomes eligible
            # to have its real values copied onto disk.
            if cleared is not None and not cleared(str(fqn), name):
                continue

            business_name = ""
            try:
                from core.schema_enrichment import enrich_columns
                enriched = enrich_columns([name], vocab=vocab)
                if enriched:
                    business_name = enriched[0].expanded_name
            except Exception:
                pass
            selected.append({
                "table_fqn": str(fqn),
                "column": name,
                "type": ctype,
                "business_name": business_name,
                "database": str(meta.get("database") or ""),
                "schema": str(meta.get("schema") or ""),
                "table": bare_table,
            })
    return selected


# ── DISTINCT harvesting ───────────────────────────────────────────────────────

def _distinct_sql(db_type: str, database: str, schema: str, table: str, column: str, cap: int) -> str:
    if db_type == "azure_sql":
        tbl = f"[{schema}].[{table}]" if schema else f"[{table}]"
        return (
            f"SELECT DISTINCT TOP {cap + 1} [{column}] FROM {tbl} "
            f"WHERE [{column}] IS NOT NULL"
        )
    if db_type == "oracle":
        tbl = f'"{schema.upper()}"."{table.upper()}"' if schema else f'"{table.upper()}"'
        return (
            f'SELECT * FROM (SELECT DISTINCT "{column.upper()}" FROM {tbl} '
            f'WHERE "{column.upper()}" IS NOT NULL) WHERE ROWNUM <= {cap + 1}'
        )
    # snowflake / default
    parts = [p for p in (database, schema, table) if p]
    tbl = ".".join(f'"{p}"' for p in parts)
    return (
        f'SELECT DISTINCT "{column}" FROM {tbl} '
        f'WHERE "{column}" IS NOT NULL LIMIT {cap + 1}'
    )


def _values_look_like_pii(column: str, col_type: str, values: list[str]) -> bool:
    """Value-level PII gate reusing core.masking's scanner."""
    from core.masking import scan_values_for_pii
    rows = [{column: v} for v in values[:200]]
    col_defs = [{"name": column, "type": col_type or "varchar"}]
    try:
        return bool(scan_values_for_pii(rows, col_defs))
    except Exception:
        return False


def build_value_index(
    account_id: str,
    credentials: dict,
    db_type: str,
    schema_dir: str,
    *,
    per_column_cap: int = 5000,
    vocab=None,
    run_query_fn: Callable[..., list[dict]] | None = None,
    base_dir: str = _DEFAULT_BASE_DIR,
) -> dict:
    """
    Harvest distinct values for filterable columns into
    clients/{account_id}/value_index.sqlite. Atomic: written to a .tmp file
    and os.replace'd, so readers never see a half-built index.

    run_query_fn is injectable for tests; defaults to core.schema.run_query.
    Returns build stats (also persisted in the index's meta table).
    """
    from core.schema import load_schema_json

    if run_query_fn is None:
        from core.schema import run_query as run_query_fn  # type: ignore[no-redef]

    schema = load_schema_json(schema_dir)
    cleared, industry = _clearance_gate(account_id)
    considered_ungated = len(select_filterable_columns(schema, vocab=vocab))
    columns = select_filterable_columns(
        schema, vocab=vocab, account_id=account_id, industry=industry, cleared=cleared,
    )
    if cleared is not None:
        log.info(
            "value index for regulated tenant %s: %d of %d otherwise-eligible "
            "columns cleared for indexing",
            account_id, len(columns), considered_ungated,
        )

    final_path = _index_path(account_id, base_dir)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_suffix(".sqlite.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    stats: dict[str, Any] = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "per_column_cap": per_column_cap,
        "columns_considered": len(columns),
        # Regulated tenants only: how many eligible columns the clearance gate
        # refused. A build that indexes nothing is a governed outcome, not a
        # failure, and the two must be distinguishable in the stats.
        "regulated": cleared is not None,
        "industry": industry,
        "columns_refused_unclassified": (
            considered_ungated - len(columns) if cleared is not None else 0
        ),
        "columns_indexed": 0,
        "values_indexed": 0,
        "columns_skipped_pii": 0,
        "columns_failed": 0,
        "truncated_columns": [],
    }

    conn = sqlite3.connect(tmp_path)
    try:
        conn.executescript(
            """
            CREATE TABLE column_value (
              table_fqn TEXT NOT NULL, column_name TEXT NOT NULL,
              business_name TEXT NOT NULL DEFAULT '',
              value TEXT NOT NULL, value_norm TEXT NOT NULL);
            CREATE INDEX ix_cv_norm ON column_value(value_norm);
            CREATE INDEX ix_cv_col ON column_value(table_fqn, column_name);
            CREATE TABLE value_index_meta (key TEXT PRIMARY KEY, value TEXT);
            """
        )
        for col in columns:
            sql = _distinct_sql(
                db_type, col["database"], col["schema"], col["table"],
                col["column"], per_column_cap,
            )
            try:
                rows = run_query_fn(credentials, db_type, sql, max_rows=per_column_cap + 1)
            except Exception as exc:
                stats["columns_failed"] += 1
                log.debug("Value index: DISTINCT failed for %s.%s: %s",
                          col["table_fqn"], col["column"], exc)
                continue

            values: list[str] = []
            for row in rows:
                raw = next(iter(row.values())) if isinstance(row, dict) and row else None
                if raw is None:
                    continue
                text = str(raw).strip()
                if not text or len(text) > _MAX_VALUE_LEN or "\n" in text or "\r" in text:
                    continue
                values.append(text)

            if not values:
                continue
            if _values_look_like_pii(col["column"], col.get("type", ""), values):
                stats["columns_skipped_pii"] += 1
                log.info("Value index: skipping %s.%s — values look like PII",
                         col["table_fqn"], col["column"])
                continue
            if len(values) > per_column_cap:
                values = values[:per_column_cap]
                stats["truncated_columns"].append(f"{col['table_fqn']}.{col['column']}")

            conn.executemany(
                "INSERT INTO column_value (table_fqn, column_name, business_name, value, value_norm) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (col["table_fqn"], col["column"], col["business_name"], v, normalize_value(v))
                    for v in values
                ],
            )
            stats["columns_indexed"] += 1
            stats["values_indexed"] += len(values)

        conn.executemany(
            "INSERT INTO value_index_meta (key, value) VALUES (?, ?)",
            [(k, json.dumps(v)) for k, v in stats.items()],
        )
        conn.commit()
    finally:
        conn.close()

    os.replace(tmp_path, final_path)
    log.info(
        "Value index built for %s: %d columns, %d values (%d PII-skipped, %d failed)",
        account_id, stats["columns_indexed"], stats["values_indexed"],
        stats["columns_skipped_pii"], stats["columns_failed"],
    )
    return stats


# ── Lookup API ────────────────────────────────────────────────────────────────

def index_exists(account_id: str, base_dir: str = _DEFAULT_BASE_DIR) -> bool:
    return _index_path(account_id, base_dir).is_file()


def load_index_stats(account_id: str, base_dir: str = _DEFAULT_BASE_DIR) -> dict:
    path = _index_path(account_id, base_dir)
    if not path.is_file():
        return {}
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            rows = conn.execute("SELECT key, value FROM value_index_meta").fetchall()
        finally:
            conn.close()
        return {k: json.loads(v) for k, v in rows}
    except Exception:
        return {}


def _open_ro(account_id: str, base_dir: str) -> sqlite3.Connection | None:
    path = _index_path(account_id, base_dir)
    if not path.is_file():
        return None
    try:
        return sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except Exception:
        return None


def _table_allowed(table_fqn: str, allowed_tables: set[str] | None) -> bool:
    if allowed_tables is None:
        return True
    upper = table_fqn.upper()
    parts = upper.split(".")
    variants = {upper, parts[-1]}
    if len(parts) >= 2:
        variants.add(".".join(parts[-2:]))
    allowed_upper = {str(t).upper() for t in allowed_tables}
    return bool(variants & allowed_upper)


def lookup_exact(
    account_id: str,
    phrase: str,
    allowed_tables: set[str] | None = None,
    base_dir: str = _DEFAULT_BASE_DIR,
) -> list[dict]:
    """Case-insensitive exact match, then normalized match. Returns
    [{table_fqn, column, business_name, value, method}]."""
    conn = _open_ro(account_id, base_dir)
    if conn is None:
        return []
    try:
        out: list[dict] = []
        seen: set[tuple[str, str, str]] = set()
        for method, sql, arg in (
            ("exact", "SELECT table_fqn, column_name, business_name, value FROM column_value "
                      "WHERE value = ? COLLATE NOCASE LIMIT 50", (phrase or "").strip()),
            ("normalized", "SELECT table_fqn, column_name, business_name, value FROM column_value "
                           "WHERE value_norm = ? LIMIT 50", normalize_value(phrase)),
        ):
            if not arg:
                continue
            for tf, cn, bn, val in conn.execute(sql, (arg,)).fetchall():
                key = (tf, cn, val)
                if key in seen or not _table_allowed(tf, allowed_tables):
                    continue
                seen.add(key)
                out.append({"table_fqn": tf, "column": cn, "business_name": bn,
                            "value": val, "method": method, "score": 1.0})
            if out:
                break   # exact hits are authoritative; skip weaker tier
        return out
    finally:
        conn.close()


def lookup_fuzzy(
    account_id: str,
    phrase: str,
    allowed_tables: set[str] | None = None,
    limit: int = 5,
    base_dir: str = _DEFAULT_BASE_DIR,
    min_score: float = FUZZY_CANDIDATE,
) -> list[dict]:
    """
    Fuzzy match: SQL LIKE prefilter (first token prefix OR longest-token
    containment, capped at 300 candidates), then difflib.SequenceMatcher
    scoring against the normalized phrase. Returns matches with
    score >= min_score (default FUZZY_CANDIDATE for prompt injection; the
    zero-row RCA passes a looser floor since 'closest values' suggestions
    only inform the user, they never rewrite a query).
    """
    norm = normalize_value(phrase)
    if len(norm) < 3:
        return []
    conn = _open_ro(account_id, base_dir)
    if conn is None:
        return []
    try:
        tokens = norm.split()
        longest = max(tokens, key=len)
        first = tokens[0]
        # Three probes: first-token prefix (fast path), longest-token
        # containment, and a 2-char prefix so first-syllable typos still
        # reach the scorer ("emko corp" must find "emco corporation").
        # Short rows first keeps the most comparable candidates inside the
        # LIMIT when a 2-char prefix is common.
        candidates = conn.execute(
            "SELECT table_fqn, column_name, business_name, value, value_norm FROM column_value "
            "WHERE value_norm LIKE ? OR value_norm LIKE ? OR value_norm LIKE ? "
            "ORDER BY length(value_norm) LIMIT 300",
            (f"{first}%", f"%{longest}%", f"{norm[:2]}%"),
        ).fetchall()
    finally:
        conn.close()

    scored: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for tf, cn, bn, val, vn in candidates:
        key = (tf, cn, val)
        if key in seen or not _table_allowed(tf, allowed_tables):
            continue
        seen.add(key)
        score = SequenceMatcher(None, norm, vn).ratio()
        # Containment bonus: "emco" inside "emco corporation" is a strong
        # signal SequenceMatcher under-scores for length-mismatched strings.
        if norm and norm in vn:
            score = max(score, 0.60 + 0.40 * (len(norm) / max(len(vn), 1)))
        if score >= min_score:
            scored.append({"table_fqn": tf, "column": cn, "business_name": bn,
                           "value": val, "method": "fuzzy", "score": round(score, 4)})
    scored.sort(key=lambda m: m["score"], reverse=True)
    return scored[:limit]
