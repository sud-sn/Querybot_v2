"""
core/semantic_contract.py
─────────────────────────
The compiled semantic contract — ONE deterministic artifact holding every
approved source of semantic truth for an account.

Why this exists
───────────────
Semantic truth used to live in eight places (semantic model JSON, metric
registry, entity graph, business terms, field overrides, column context,
vocab packs, KB markdown) and was merged per-question at runtime. Approvals
mutated those sources asymmetrically — a metric or date-role approval updated
`_semantic_model.json` only, so the runtime context and the KB could diverge
until the next full KB build. Every supersession/staleness bug traced back to
that drift.

The contract fixes it structurally:
  • `compile_contract(account_id)` gathers all APPROVED sources into one dict
    (pure DB/file reads — no LLM, cheap enough to run on every approval).
  • `write_contract(account_id)` persists it as `kb_dir/_semantic_contract.json`
    with a deterministic `contract_version` (md5[:12] of the canonical body).
  • `load_contract(kb_dir)` is the single runtime read point (mtime-cached).
  • Every admin approval route calls `recompile_contract(account_id)` so the
    artifact — and its version stamp — always reflects the approved state.

The version threads through answer traces, learning candidates, and eval runs
so answer quality can be correlated with exact semantic states.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("querybot.semantic_contract")

CONTRACT_FILENAME = "_semantic_contract.json"
_compile_locks_guard = threading.Lock()
_compile_locks: dict[str, threading.Lock] = {}


# ══════════════════════════════════════════════════════════════════════════════
# Source gathering
# ══════════════════════════════════════════════════════════════════════════════

def _canonical(obj: Any) -> str:
    """Stable serialization — the same logical content always hashes equal."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _hash(obj: Any) -> str:
    return hashlib.md5(_canonical(obj).encode("utf-8")).hexdigest()[:12]


def _load_column_context(account_id: str) -> dict:
    path = Path("clients") / account_id / "column_context.json"
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception as exc:
        log.debug("column_context.json unreadable for %s: %s", account_id, exc)
    return {}


def _compile_lock(account_id: str) -> threading.Lock:
    with _compile_locks_guard:
        return _compile_locks.setdefault(account_id, threading.Lock())


def _source_conflict(source: str, exc: Exception) -> dict[str, Any]:
    return {
        "conflict_key": f"source_unavailable:{source}",
        "code": "semantic_source_unavailable",
        "severity": "ERROR",
        "object_type": "semantic_source",
        "object_id": source,
        "origin": source,
        "message": f"The {source} semantic source could not be read.",
        "evidence": {"error_type": type(exc).__name__, "detail": str(exc)[:500]},
        "suggestions": ["Restore the source and compile again."],
    }


def _compile_contract_internal(
    account_id: str, kb_dir: str = "",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Gather every approved semantic source into one deterministic dict.

    kb_dir is resolved from client state when not supplied. Missing sources
    compile to empty sections — a contract is valid (and versioned) even for
    a half-configured account, so consumers never need existence special-cases
    beyond "contract file absent".
    """
    import store
    from core.field_overrides import load_field_overrides
    from core.semantic_model import load_semantic_model

    if not kb_dir:
        state = store.get_client_state(account_id) or {}
        kb_dir = state.get("kb_dir") or ""

    diagnostics: list[dict[str, Any]] = []
    try:
        model = load_semantic_model(kb_dir) if kb_dir else {}
    except Exception as exc:
        log.warning("Contract compile: model unavailable for %s: %s", account_id, exc)
        diagnostics.append(_source_conflict("semantic_model", exc))
        model = {}

    try:
        metrics = store.list_metrics(account_id, active_only=True)
    except Exception as exc:
        log.warning("Contract compile: metrics unavailable for %s: %s", account_id, exc)
        diagnostics.append(_source_conflict("metrics", exc))
        metrics = []

    try:
        date_contexts = store.list_metric_date_contexts(account_id, active_only=True)
    except Exception as exc:
        log.warning("Contract compile: date contexts unavailable for %s: %s", account_id, exc)
        diagnostics.append(_source_conflict("date_contexts", exc))
        date_contexts = []

    try:
        graph = store.get_full_graph(account_id)
    except Exception as exc:
        log.warning("Contract compile: graph unavailable for %s: %s", account_id, exc)
        diagnostics.append(_source_conflict("entity_graph", exc))
        graph = {}

    try:
        terms = store.list_terms(account_id, active_only=True)
    except Exception as exc:
        log.warning("Contract compile: terms unavailable for %s: %s", account_id, exc)
        diagnostics.append(_source_conflict("business_terms", exc))
        terms = []

    try:
        field_overrides = load_field_overrides(account_id)
    except Exception as exc:
        log.warning("Contract compile: overrides unavailable for %s: %s", account_id, exc)
        diagnostics.append(_source_conflict("field_overrides", exc))
        field_overrides = {}

    body = {
        # The approval-preserving structured model: tables (fields, measures,
        # dimensions, date_roles, grain, default_filters) and relationships.
        "model": model,
        # Approved metric formulas — the highest-authority semantic layer.
        "metrics": metrics,
        # Governed metric/context -> role-playing date mappings. Keeping these
        # in the versioned contract makes date policy changes traceable even
        # though runtime resolution also reads the tenant-scoped DB records.
        "date_contexts": date_contexts,
        # Entity graph: join contracts between business entities.
        "graph": graph,
        # Business glossary terms (canonical SQL expressions per phrase).
        "terms": terms,
        # Persistent admin field overrides + free-form column context notes.
        "overrides": {
            "field_overrides": field_overrides,
            "column_context": _load_column_context(account_id),
        },
    }

    contract_version = _hash(body)
    contract = {
        "meta": {
            "contract_version": contract_version,
            "account_id": account_id,
            "compiled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            # Per-source fingerprints make "what changed?" answerable at a
            # glance when two contract versions differ.
            "sources": {key: _hash(value) for key, value in body.items()},
        },
        **body,
    }
    return contract, diagnostics


def compile_contract(account_id: str, kb_dir: str = "") -> dict[str, Any]:
    """Backward-compatible pure compiler. Governance uses the diagnostics too."""
    return _compile_contract_internal(account_id, kb_dir)[0]


# ══════════════════════════════════════════════════════════════════════════════
# Persistence + cached load
# ══════════════════════════════════════════════════════════════════════════════

def contract_path(kb_dir: str) -> Path:
    return Path(kb_dir) / CONTRACT_FILENAME


def _write_contract_file(contract: dict[str, Any], kb_dir: str) -> None:
    target = contract_path(kb_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(target)


def write_contract(account_id: str, kb_dir: str = "") -> dict[str, Any]:
    """Compile and atomically persist the contract. Returns the contract dict
    ({} when the account has no kb_dir yet — nothing to anchor the file to)."""
    import store

    if not kb_dir:
        state = store.get_client_state(account_id) or {}
        kb_dir = state.get("kb_dir") or ""
    if not kb_dir:
        log.debug("write_contract skipped for %s — no kb_dir yet", account_id)
        return {}

    contract = compile_contract(account_id, kb_dir)
    _write_contract_file(contract, kb_dir)
    # Existing tenants receive a grandfathered baseline without changing the
    # runtime contract format or requiring an admin migration step.
    try:
        state = store.get_semantic_compiler_state(account_id)
        version = contract["meta"]["contract_version"]
        store.save_semantic_contract_version(
            account_id, contract, status="active", created_by="legacy_compile",
        )
        store.publish_semantic_contract_version(
            account_id, version, baseline=not bool(state.get("baseline_version")),
        )
    except Exception as exc:
        log.debug("Contract baseline persistence skipped for %s: %s", account_id, exc)
    log.info(
        "Semantic contract compiled for %s: version=%s (%d metrics, %d terms, %d tables)",
        account_id,
        contract["meta"]["contract_version"],
        len(contract.get("metrics") or []),
        len(contract.get("terms") or []),
        len((contract.get("model") or {}).get("tables") or []),
    )
    return contract


def governed_recompile_contract(
    account_id: str, *, trigger: str = "approval", initiated_by: str = "system",
) -> dict[str, Any]:
    """Compile, record diagnostics, and safely switch the active contract.

    The current JSON artifact remains the runtime read path. A failed compile
    never replaces it. Shadow is the default migration mode; enforce is ready
    for a later tenant-by-tenant rollout.
    """
    import store

    with _compile_lock(account_id):
        client_state = store.get_client_state(account_id) or {}
        kb_dir = client_state.get("kb_dir") or ""
        compiler_state = store.get_semantic_compiler_state(account_id)
        current = load_contract(kb_dir) if kb_dir else {}
        current_version = str(
            compiler_state.get("active_version")
            or (current.get("meta") or {}).get("contract_version")
            or ""
        )

        # Grandfather the exact contract already serving traffic.
        if current and not compiler_state.get("baseline_version"):
            store.save_semantic_contract_version(
                account_id, current, status="active", created_by="migration_baseline",
            )
            store.publish_semantic_contract_version(
                account_id, current_version, baseline=True,
            )
            compiler_state = store.get_semantic_compiler_state(account_id)

        run_id = store.create_semantic_compile_run(
            account_id,
            trigger=trigger,
            initiated_by=initiated_by,
            mode=compiler_state.get("mode") or "shadow",
            base_version=current_version,
        )
        try:
            if not kb_dir:
                raise ValueError("Client has no Knowledge Base directory yet")
            contract, conflicts = _compile_contract_internal(account_id, kb_dir)
            version = str((contract.get("meta") or {}).get("contract_version") or "")
            store.save_semantic_contract_version(
                account_id, contract, status="draft",
                compile_run_id=run_id, created_by=initiated_by,
            )
            store.set_semantic_draft_version(account_id, version)
            store.save_semantic_conflicts(run_id, account_id, conflicts)

            error_count = sum(c.get("severity") == "ERROR" for c in conflicts)
            warning_count = sum(c.get("severity") == "WARNING" for c in conflicts)
            info_count = len(conflicts) - error_count - warning_count
            mode = compiler_state.get("mode") or "shadow"
            hard_source_failure = any(
                c.get("code") == "semantic_source_unavailable" for c in conflicts
            )
            if not hard_source_failure:
                store.reconcile_semantic_conflicts(
                    account_id,
                    {str(c.get("conflict_key") or "") for c in conflicts},
                    run_id=run_id,
                )
            blocks = hard_source_failure or (mode == "enforce" and error_count > 0)
            auto_publish = (
                compiler_state.get("publish_mode") or "auto_publish_clean"
            ) == "auto_publish_clean"

            published = ""
            status = "invalid" if blocks else "valid"
            message = "Compile completed."
            if blocks:
                message = "Compile rejected; the last active contract remains in service."
            elif auto_publish:
                _write_contract_file(contract, kb_dir)
                try:
                    store.publish_semantic_contract_version(account_id, version)
                except Exception:
                    # Keep the materialized runtime artifact aligned with the
                    # active-version pointer if the metadata switch fails.
                    if current:
                        _write_contract_file(current, kb_dir)
                    raise
                published = version
                status = "published"
                message = "Compile completed and the active contract was switched atomically."

            store.finish_semantic_compile_run(
                run_id,
                status=status,
                draft_version=version,
                published_version=published,
                error_count=error_count,
                warning_count=warning_count,
                info_count=info_count,
                source_fingerprints=(contract.get("meta") or {}).get("sources") or {},
                message=message,
            )
            return {
                "run_id": run_id,
                "status": status,
                "draft_version": version,
                "published_version": published,
                "active_version": published or current_version,
                "conflicts": conflicts,
                "message": message,
            }
        except Exception as exc:
            store.finish_semantic_compile_run(
                run_id, status="failed", message=str(exc)[:1000], error_count=1,
            )
            log.error("Governed contract compile failed for %s: %s", account_id, exc)
            return {
                "run_id": run_id,
                "status": "failed",
                "active_version": current_version,
                "published_version": "",
                "conflicts": [],
                "message": str(exc),
            }


def recompile_contract(account_id: str, *, trigger: str = "approval") -> str:
    """Best-effort recompile used by admin approval routes — one line to call,
    never raises. Returns the new contract_version ("" on skip/failure)."""
    try:
        result = governed_recompile_contract(account_id, trigger=trigger)
        return str(result.get("active_version") or "")
    except Exception as exc:
        log.error("Contract recompile failed for %s: %s", account_id, exc)
        return ""


_cache_lock = threading.Lock()
_cache: dict[str, tuple[float, dict]] = {}   # path -> (mtime, contract)


def load_contract(kb_dir: str) -> dict[str, Any]:
    """Runtime read point — mtime-cached so per-question cost is one stat()."""
    if not kb_dir:
        return {}
    path = contract_path(kb_dir)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    with _cache_lock:
        cached = _cache.get(str(path))
        if cached and cached[0] == mtime:
            return cached[1]
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Semantic contract unreadable at %s: %s", path, exc)
        return {}
    if not isinstance(contract, dict):
        return {}
    with _cache_lock:
        _cache[str(path)] = (mtime, contract)
    return contract


def contract_fingerprint(kb_dir: str) -> str:
    """The current contract_version, or "" when no contract is compiled yet."""
    return ((load_contract(kb_dir).get("meta") or {}).get("contract_version", ""))
