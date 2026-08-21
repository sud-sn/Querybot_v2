"""
core/vocab_packs.py

Pluggable ERP/warehouse terminology packs.

The KB build and the deterministic semantic layer historically read hardcoded
Infor-M3-flavoured vocabularies (ERP_COLUMN_DICT, ABBREVIATIONS, suffix rules,
date-role regexes…). This module makes that vocabulary data-driven:

  packs/<pack_id>.json          — shipped terminology packs (Infor M3, generic
                                  star schema; SAP/EBS/JDE/… stubs for Phase 2)
  clients/<account_id>/vocab.json — optional per-client overlay, merged last
  client.erp_packs (JSON array) — which packs a client has enabled

Backward compatibility contract: with no packs selected, every consumer
behaves EXACTLY as before — `builtin_vocab()` mirrors the module constants,
and every consumer API takes an optional `vocab=None` that falls back to the
active vocab, whose default is the builtin. The Infor M3 pack preserves every
builtin mapping and adds M3-only physical record prefixes and source metadata.

ContextVar note: `activate_vocab()` scopes the vocab for a request/build.
ContextVars do NOT propagate into `loop.run_in_executor` threads — pass
`vocab=` explicitly to anything called through an executor.
"""

from __future__ import annotations

import json
import logging
import re
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("querybot.vocab_packs")

_PACKS_DIR = Path(__file__).resolve().parents[1] / "packs"
_CLIENTS_DIR = Path(__file__).resolve().parents[1] / "clients"
_NAMING_PROFILE_FILE = "naming_profile.json"


@dataclass
class MergedVocab:
    # {CODE: (label, [synonyms])} — ERP short-code dictionary
    column_dict: dict[str, tuple[str, list[str]]] = field(default_factory=dict)
    # {TABLE: {"label":…, "synonyms":[…], "type": "fact"|"dimension"}}
    table_dict: dict[str, dict] = field(default_factory=dict)
    # {TOKEN: expansion} — schema-enrichment token expansion (superset)
    abbreviations: dict[str, str] = field(default_factory=dict)
    # {TOKEN: expansion} — semantic-planner alias matching. Deliberately a
    # SUBSET of `abbreviations` in the builtins (the planner is conservative
    # to avoid false required-field matches); packs contribute to both.
    planner_abbreviations: dict[str, str] = field(default_factory=dict)
    # {COLUMN: {alias phrases}}
    direct_aliases: dict[str, set[str]] = field(default_factory=dict)
    # {COLUMN: {equivalent join columns}}
    join_synonyms: dict[str, set[str]] = field(default_factory=dict)
    # {PREFIX: business entity}
    entity_prefixes: dict[str, str] = field(default_factory=dict)
    # {two-character record prefix: canonical ERP file}. Infor M3 physical
    # columns commonly prepend a record/file prefix to a governed field code
    # (for example OB + ORNO = OBORNO). This is deliberately pack-scoped so
    # generic schemas never have their first two characters stripped.
    record_prefixes: dict[str, str] = field(default_factory=dict)
    # [(compiled pattern with one capture group, label template "… {}", role)]
    numbered_series: list[tuple[re.Pattern, str, str]] = field(default_factory=list)
    raw_identifier_codes: set[str] = field(default_factory=set)
    raw_measure_codes: set[str] = field(default_factory=set)
    raw_date_codes: set[str] = field(default_factory=set)
    raw_status_codes: set[str] = field(default_factory=set)
    # Identifier codes that are RELATIONAL keys — safe to join two tables on
    # when both carry them. Deliberately narrower than raw_identifier_codes:
    # a division or facility code identifies a grouping, appears on many
    # unrelated tables, and joining on it manufactures a false edge. Which
    # identifiers are relational is ERP knowledge, so it lives in the pack.
    join_key_codes: set[str] = field(default_factory=set)
    # Codes that belong in an ON clause but must never CREATE one. A client,
    # company or partition code sits on nearly every table in an ERP, so
    # letting it establish an edge makes every pair of tables adjacent and the
    # join planner routes through nonsense. These are added only to a pair a
    # real key already linked.
    join_qualifier_codes: set[str] = field(default_factory=set)
    # Column-name suffixes that may establish a join edge, for warehouses that
    # use a naming convention instead of fixed short codes. Pack-scoped
    # because the convention is the modeller's, not ours: "_DMS_KEY" is Infor
    # M3's, "_SK" is Kimball's, and a generic "_ID" is safe on one warehouse
    # and a sort key on another.
    join_key_suffixes: set[str] = field(default_factory=set)
    # [(compiled pattern, date-role key)] — PACK-added patterns only; builtin
    # date-role regexes stay in core/date_roles.py and are checked after these.
    date_role_patterns: list[tuple[re.Pattern, str]] = field(default_factory=list)
    fact_patterns: list[re.Pattern] = field(default_factory=list)
    dimension_patterns: list[re.Pattern] = field(default_factory=list)
    bridge_patterns: list[re.Pattern] = field(default_factory=list)
    fact_tables: set[str] = field(default_factory=set)
    dimension_tables: set[str] = field(default_factory=set)
    source_packs: list[str] = field(default_factory=list)


# ── Pack loading ──────────────────────────────────────────────────────────────

_pack_cache: dict[str, tuple[float, dict]] = {}


def list_available_packs() -> list[dict]:
    """Return pack manifests (id, erp_name, status, description) for the UI."""
    manifests: list[dict] = []
    if not _PACKS_DIR.is_dir():
        return manifests
    for path in sorted(_PACKS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Skipping unreadable pack %s: %s", path.name, exc)
            continue
        manifests.append({
            "pack_id": data.get("pack_id") or path.stem,
            "erp_name": data.get("erp_name") or path.stem,
            "status": data.get("status") or "complete",
            "description": data.get("description") or "",
            "version": data.get("version") or 1,
        })
    return manifests


def load_pack(pack_id: str) -> dict:
    """Load a pack JSON by id; {} when missing/invalid. mtime-cached."""
    safe_id = re.sub(r"[^A-Za-z0-9_\-]", "", pack_id or "")
    if not safe_id:
        return {}
    path = _PACKS_DIR / f"{safe_id}.json"
    if not path.is_file():
        log.warning("Terminology pack not found: %s", safe_id)
        return {}
    mtime = path.stat().st_mtime
    cached = _pack_cache.get(safe_id)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("pack root must be an object")
    except Exception as exc:
        log.warning("Terminology pack %s is invalid: %s", safe_id, exc)
        return {}
    _pack_cache[safe_id] = (mtime, data)
    return data


def _compile(pattern: str, origin: str) -> re.Pattern | None:
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        log.warning("Skipping bad regex %r in %s: %s", pattern, origin, exc)
        return None
    if not (pattern.startswith(("^", "(?:^")) or pattern.endswith("$")):
        log.warning("Pack regex %r in %s is not anchored (^ or $) — it may over-match", pattern, origin)
    return compiled


def _merge_pack(vocab: MergedVocab, pack: dict, origin: str) -> None:
    """Merge one pack dict into vocab. Dicts override per-key; code sets union;
    pattern lists PREPEND (pack patterns are checked before builtin ones)."""
    for code, entry in (pack.get("column_dict") or {}).items():
        if isinstance(entry, dict):
            label = str(entry.get("label") or code)
            syns = [str(s) for s in (entry.get("synonyms") or [])]
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            label, syns = str(entry[0]), [str(s) for s in entry[1]]
        else:
            continue
        vocab.column_dict[str(code).upper()] = (label, syns)

    for tbl, entry in (pack.get("table_dict") or {}).items():
        if isinstance(entry, dict):
            vocab.table_dict[str(tbl).upper()] = entry
            ttype = str(entry.get("type") or "").lower()
            if ttype == "fact":
                vocab.fact_tables.add(str(tbl).upper())
            elif ttype == "dimension":
                vocab.dimension_tables.add(str(tbl).upper())

    for tok, expansion in (pack.get("abbreviations") or {}).items():
        vocab.abbreviations[str(tok).upper()] = str(expansion)
        vocab.planner_abbreviations[str(tok).upper()] = str(expansion)

    for col, aliases in (pack.get("direct_aliases") or {}).items():
        vocab.direct_aliases[str(col).upper()] = {str(a) for a in (aliases or [])}

    for col, syns in (pack.get("join_synonyms") or {}).items():
        vocab.join_synonyms[str(col).upper()] = {str(s).upper() for s in (syns or [])}

    for prefix, entity in (pack.get("entity_prefixes") or {}).items():
        vocab.entity_prefixes[str(prefix).upper()] = str(entity)

    for prefix, table in (pack.get("record_prefixes") or {}).items():
        prefix_key = str(prefix).upper().strip()
        if len(prefix_key) != 2 or not prefix_key.isalnum():
            log.warning("Skipping invalid record prefix %r in %s", prefix, origin)
            continue
        vocab.record_prefixes[prefix_key] = str(table).upper().strip()

    new_series: list[tuple[re.Pattern, str, str]] = []
    for item in pack.get("numbered_series") or []:
        if not isinstance(item, dict):
            continue
        compiled = _compile(str(item.get("pattern") or ""), origin)
        if compiled is None:
            continue
        new_series.append((compiled, str(item.get("label") or "{}"), str(item.get("role") or "attribute")))
    if new_series:
        new_pats = {p.pattern for p, _, _ in new_series}
        vocab.numbered_series = new_series + [
            entry for entry in vocab.numbered_series if entry[0].pattern not in new_pats
        ]

    vocab.raw_identifier_codes |= {str(c).upper() for c in (pack.get("raw_identifier_codes") or [])}
    vocab.raw_measure_codes |= {str(c).upper() for c in (pack.get("raw_measure_codes") or [])}
    vocab.raw_date_codes |= {str(c).upper() for c in (pack.get("raw_date_codes") or [])}
    vocab.raw_status_codes |= {str(c).upper() for c in (pack.get("raw_status_codes") or [])}
    vocab.join_key_codes |= {str(c).upper() for c in (pack.get("join_key_codes") or [])}
    vocab.join_qualifier_codes |= {
        str(c).upper() for c in (pack.get("join_qualifier_codes") or [])
    }
    vocab.join_key_suffixes |= {
        str(c).upper() for c in (pack.get("join_key_suffixes") or []) if str(c).strip()
    }

    new_dr: list[tuple[re.Pattern, str]] = []
    try:
        from core.date_roles import DATE_ROLES
        valid_roles = {r.key for r in DATE_ROLES}
    except Exception:
        valid_roles = set()
    for item in pack.get("date_role_patterns") or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "")
        if valid_roles and role not in valid_roles:
            log.warning("Skipping date-role pattern with unknown role %r in %s", role, origin)
            continue
        compiled = _compile(str(item.get("pattern") or ""), origin)
        if compiled is None:
            continue
        new_dr.append((compiled, role))
    if new_dr:
        new_pats = {p.pattern for p, _ in new_dr}
        vocab.date_role_patterns = new_dr + [
            entry for entry in vocab.date_role_patterns if entry[0].pattern not in new_pats
        ]

    tc = pack.get("table_classification") or {}
    for key, target in (
        ("fact_patterns", vocab.fact_patterns),
        ("dimension_patterns", vocab.dimension_patterns),
        ("bridge_patterns", vocab.bridge_patterns),
    ):
        new_pats = [p for p in (
            _compile(str(raw), origin) for raw in (tc.get(key) or [])
        ) if p is not None]
        target[:0] = new_pats
    vocab.fact_tables |= {str(t).upper() for t in (tc.get("fact_tables") or [])}
    vocab.dimension_tables |= {str(t).upper() for t in (tc.get("dimension_tables") or [])}

    origin_id = pack.get("pack_id") or origin
    if origin_id not in vocab.source_packs:
        vocab.source_packs.append(origin_id)


# ── Builtin vocabulary (mirrors module constants exactly) ─────────────────────

_builtin_cache: MergedVocab | None = None


def builtin_vocab() -> MergedVocab:
    """MergedVocab equal to the hardcoded module constants (lazy, cached).
    Function-level imports avoid circular-import issues at module load."""
    global _builtin_cache
    if _builtin_cache is not None:
        return _builtin_cache

    from core.erp_column_dict import ERP_COLUMN_DICT
    from core.schema_enrichment import (
        ABBREVIATIONS, _NUMBERED_SERIES,
        RAW_IDENTIFIER_CODES, RAW_MEASURE_CODES, RAW_DATE_CODES,
    )
    from core.semantic_planner import (
        _DIRECT_ALIASES, _JOIN_KEY_SUFFIXES, _JOIN_SYNONYMS,
    )
    from core.semantic_planner import _ABBREVIATIONS as _PLANNER_ABBREVIATIONS
    from core.naming_convention import ENTITY_PREFIX_VOCABULARY

    vocab = MergedVocab()
    vocab.column_dict = {k: (v[0], list(v[1])) for k, v in ERP_COLUMN_DICT.items()}
    vocab.abbreviations = dict(ABBREVIATIONS)
    vocab.planner_abbreviations = dict(_PLANNER_ABBREVIATIONS)
    vocab.direct_aliases = {k: set(v) for k, v in _DIRECT_ALIASES.items()}
    vocab.join_synonyms = {k: set(v) for k, v in _JOIN_SYNONYMS.items()}
    vocab.join_key_suffixes = set(_JOIN_KEY_SUFFIXES)
    vocab.entity_prefixes = dict(ENTITY_PREFIX_VOCABULARY)
    vocab.numbered_series = list(_NUMBERED_SERIES)
    vocab.raw_identifier_codes = set(RAW_IDENTIFIER_CODES)
    vocab.raw_measure_codes = set(RAW_MEASURE_CODES)
    vocab.raw_date_codes = set(RAW_DATE_CODES)
    # No pack-added date-role or table-classification patterns by default —
    # builtin regexes in date_roles.py / naming_convention.py already apply.
    vocab.source_packs = ["builtin"]
    _builtin_cache = vocab
    return vocab


# ── Per-account resolution ────────────────────────────────────────────────────

_account_cache: dict[str, tuple[tuple, MergedVocab]] = {}


def _clone_builtin() -> MergedVocab:
    b = builtin_vocab()
    return MergedVocab(
        column_dict=dict(b.column_dict),
        table_dict=dict(b.table_dict),
        abbreviations=dict(b.abbreviations),
        planner_abbreviations=dict(b.planner_abbreviations),
        direct_aliases={k: set(v) for k, v in b.direct_aliases.items()},
        join_synonyms={k: set(v) for k, v in b.join_synonyms.items()},
        entity_prefixes=dict(b.entity_prefixes),
        record_prefixes=dict(b.record_prefixes),
        numbered_series=list(b.numbered_series),
        raw_identifier_codes=set(b.raw_identifier_codes),
        raw_measure_codes=set(b.raw_measure_codes),
        raw_date_codes=set(b.raw_date_codes),
        raw_status_codes=set(b.raw_status_codes),
        join_key_codes=set(b.join_key_codes),
        join_qualifier_codes=set(b.join_qualifier_codes),
        join_key_suffixes=set(b.join_key_suffixes),
        date_role_patterns=list(b.date_role_patterns),
        fact_patterns=list(b.fact_patterns),
        dimension_patterns=list(b.dimension_patterns),
        bridge_patterns=list(b.bridge_patterns),
        fact_tables=set(b.fact_tables),
        dimension_tables=set(b.dimension_tables),
        source_packs=list(b.source_packs),
    )


def _client_pack_ids(account_id: str) -> list[str]:
    try:
        import store
        client = store.get_client(account_id) or {}
        raw = client.get("erp_packs") or "[]"
        ids = json.loads(raw) if isinstance(raw, str) else raw
        return [str(p) for p in ids if str(p).strip()] if isinstance(ids, list) else []
    except Exception as exc:
        log.debug("erp_packs lookup failed for %s: %s", account_id, exc)
        return []


def _profile_path(account_id: str) -> Path | None:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "", str(account_id or ""))
    if not safe_id or safe_id != str(account_id or ""):
        return None
    return _CLIENTS_DIR / safe_id / _NAMING_PROFILE_FILE


def load_detected_naming_profile(account_id: str) -> dict[str, Any]:
    path = _profile_path(account_id)
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        log.warning("Naming profile for %s is invalid: %s", account_id, exc)
        return {}


def save_detected_naming_profile(account_id: str, profile: dict[str, Any]) -> Path | None:
    """Atomically persist the build-time detector result for runtime reuse."""
    path = _profile_path(account_id)
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(profile or {}, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    _account_cache.pop(account_id, None)
    return path


def _detected_pack_ids(account_id: str) -> list[str]:
    profile = load_detected_naming_profile(account_id)
    values = profile.get("auto_applied_packs") or []
    return [str(value) for value in values if str(value).strip()] if isinstance(values, list) else []


def vocab_for_account(account_id: str) -> MergedVocab:
    """Builtin + detected/selected packs + clients/<id>/vocab.json.

    Explicit admin selection always wins. Automatic detection is used only
    when no pack was selected, and only contains the detector's high-confidence
    winner. The client overlay remains the final authority.
    """
    explicit_pack_ids = _client_pack_ids(account_id)
    detected_pack_ids = [] if explicit_pack_ids else _detected_pack_ids(account_id)
    pack_ids = [*detected_pack_ids, *explicit_pack_ids]
    overlay_path = _CLIENTS_DIR / (account_id or "") / "vocab.json"
    profile_path = _profile_path(account_id)
    overlay_mtime = overlay_path.stat().st_mtime if overlay_path.is_file() else 0.0
    profile_mtime = profile_path.stat().st_mtime if profile_path and profile_path.is_file() else 0.0
    pack_mtimes = tuple(
        (_PACKS_DIR / f"{p}.json").stat().st_mtime if (_PACKS_DIR / f"{p}.json").is_file() else 0.0
        for p in pack_ids
    )
    cache_key = (tuple(pack_ids), pack_mtimes, overlay_mtime, profile_mtime)
    cached = _account_cache.get(account_id)
    if cached and cached[0] == cache_key:
        return cached[1]

    vocab = _clone_builtin()
    for pack_id in pack_ids:
        pack = load_pack(pack_id)
        if pack:
            _merge_pack(vocab, pack, pack_id)
    if overlay_mtime:
        try:
            overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
            if isinstance(overlay, dict):
                _merge_pack(vocab, overlay, f"clients/{account_id}/vocab.json")
        except Exception as exc:
            log.warning("Client vocab overlay for %s is invalid: %s", account_id, exc)

    _account_cache[account_id] = (cache_key, vocab)
    return vocab


# ── Request/build scoping ─────────────────────────────────────────────────────

_ACTIVE: ContextVar[MergedVocab | None] = ContextVar("querybot_active_vocab", default=None)


def get_active_vocab() -> MergedVocab:
    return _ACTIVE.get() or builtin_vocab()


def dimension_key_suffixes(vocab: "MergedVocab | None" = None) -> tuple[str, ...]:
    """Suffixes that mark a column as a key pointing at a dimension.

    One accessor rather than a literal per call site. Nine places had spelled
    Infor M3's "_DMS_KEY" into Python -- deciding whether a column is a
    dimension key, whether a relationship exists, which prefix names a
    dimension's label column -- so a warehouse using _SK or _DIM_KEY had those
    branches silently fail while EMCO's took them.
    """
    v = vocab if vocab is not None else get_active_vocab()
    return tuple(sorted(
        {str(s).strip().upper() for s in (v.join_key_suffixes or set()) if str(s).strip()},
        key=len, reverse=True,
    ))


# Suffixes that mark a column as an identifier even though they are too
# ambiguous to justify joining two tables on. Kept separate on purpose: whether
# a shared column may ESTABLISH a join is a question about a relationship, and
# a bare _KEY or _ID cannot answer it (it is a surrogate key on one warehouse
# and a sort key on another). Whether a column is a key rather than something
# to SUM is a question about the column alone, and there the same suffix is
# decisive -- typing DEPARTMENT_KEY as a measure is far worse than declining to
# join on it.
_AMBIGUOUS_KEY_SUFFIXES = ("_KEY", "_ID")


def key_column_suffixes(vocab: "MergedVocab | None" = None) -> tuple[str, ...]:
    """Suffixes that mark a column as an identifier rather than a measure."""
    return tuple(sorted(
        set(dimension_key_suffixes(vocab)) | set(_AMBIGUOUS_KEY_SUFFIXES),
        key=len, reverse=True,
    ))


def is_key_column(column: Any, vocab: "MergedVocab | None" = None) -> bool:
    col = str(column or "").strip().strip('[]"`').upper()
    return bool(col.endswith(key_column_suffixes(vocab)))


def is_dimension_key_column(column: Any, vocab: "MergedVocab | None" = None) -> bool:
    suffixes = dimension_key_suffixes(vocab)
    col = str(column or "").strip().strip('[]"`').upper()
    return bool(suffixes and col.endswith(suffixes))


def strip_dimension_key_suffix(column: Any, vocab: "MergedVocab | None" = None) -> str:
    """Return the column name with its dimension-key suffix removed.

    Returns "" when the column carries no such suffix, so a caller can tell
    "this is a key to a dimension" from "this merely ends in something".
    Longest suffix first, so _DMS_KEY is stripped whole rather than leaving a
    stray "_DMS" behind once "_KEY" is also in the set.
    """
    col = str(column or "").strip().strip('[]"`').upper()
    for suffix in dimension_key_suffixes(vocab):
        if col.endswith(suffix) and len(col) > len(suffix):
            return col[: -len(suffix)]
    return ""


def activate_vocab(vocab: MergedVocab):
    """Set the active vocab for this context; returns a token for deactivate()."""
    return _ACTIVE.set(vocab)


def deactivate_vocab(token) -> None:
    try:
        _ACTIVE.reset(token)
    except Exception:
        pass
