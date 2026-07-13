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


def compile_contract(account_id: str, kb_dir: str = "") -> dict[str, Any]:
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

    model = load_semantic_model(kb_dir) if kb_dir else {}

    try:
        metrics = store.list_metrics(account_id, active_only=True)
    except Exception as exc:
        log.warning("Contract compile: metrics unavailable for %s: %s", account_id, exc)
        metrics = []

    try:
        graph = store.get_full_graph(account_id)
    except Exception as exc:
        log.warning("Contract compile: graph unavailable for %s: %s", account_id, exc)
        graph = {}

    try:
        terms = store.list_terms(account_id, active_only=True)
    except Exception as exc:
        log.warning("Contract compile: terms unavailable for %s: %s", account_id, exc)
        terms = []

    try:
        field_overrides = load_field_overrides(account_id)
    except Exception as exc:
        log.warning("Contract compile: overrides unavailable for %s: %s", account_id, exc)
        field_overrides = {}

    body = {
        # The approval-preserving structured model: tables (fields, measures,
        # dimensions, date_roles, grain, default_filters) and relationships.
        "model": model,
        # Approved metric formulas — the highest-authority semantic layer.
        "metrics": metrics,
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
    return {
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


# ══════════════════════════════════════════════════════════════════════════════
# Persistence + cached load
# ══════════════════════════════════════════════════════════════════════════════

def contract_path(kb_dir: str) -> Path:
    return Path(kb_dir) / CONTRACT_FILENAME


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
    target = contract_path(kb_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(target)
    log.info(
        "Semantic contract compiled for %s: version=%s (%d metrics, %d terms, %d tables)",
        account_id,
        contract["meta"]["contract_version"],
        len(contract.get("metrics") or []),
        len(contract.get("terms") or []),
        len((contract.get("model") or {}).get("tables") or []),
    )
    return contract


def recompile_contract(account_id: str) -> str:
    """Best-effort recompile used by admin approval routes — one line to call,
    never raises. Returns the new contract_version ("" on skip/failure)."""
    try:
        contract = write_contract(account_id)
        return (contract.get("meta") or {}).get("contract_version", "")
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
