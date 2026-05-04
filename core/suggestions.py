"""
core/suggestions.py

Dynamic question suggestion engine for the portal chat UI.

Sources (priority order):
  1. Stage 2 *_queries.md files — natural language questions generated at KB
     build time from the actual schema. Available from day 1 after KB build.
     Cached as suggested_questions.json in the kb_dir so portal loads are fast.
  2. Validated examples (SQLite) — questions that have been proven against the
     real DB. Highest trust, but only available after Stage 2 validation runs.
  3. Metric registry — admin-defined metrics formatted as natural questions.
     Reliable fallback when neither of the above has enough content.

All sources are filtered by the user's allowed_tables so users never see
suggestions for data they cannot access.

Suggestions are shuffled per call so the panel shows variety across sessions.
"""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Optional

log = logging.getLogger("querybot.suggestions")

_CACHE_FILENAME = "suggested_questions.json"
_SKIP_PREFIXES = ("sql:", "select ", "with ", "--")  # non-question lines
_SQL_TABLE_REF_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+("
    r"(?:\[[^\]]+\]|[A-Z0-9_]+)"
    r"(?:\s*\.\s*(?:\[[^\]]+\]|[A-Z0-9_]+)){0,2}"
    r")",
    re.IGNORECASE,
)


# ── Cache build (called after KB generation) ──────────────────────────────────

def build_suggestion_cache(kb_dir: str) -> int:
    """
    Parse all *_queries.md Stage 2 files and write a JSON cache of
    {table, fqn, question} dicts to kb_dir/suggested_questions.json.

    fqn is extracted from the KB file header (# DB.SCHEMA.TABLE) so the
    suggestion carries its fully-qualified table name.  This fixes the ACL
    filter mismatch where bare table names never matched FQN-style allowed sets.
    """
    kb_path = Path(kb_dir)
    if not kb_path.exists():
        return 0

    entries: list[dict] = []
    for qfile in sorted(kb_path.glob("*_queries.md")):
        bare_name = qfile.stem.replace("_queries", "").upper()

        # Try to find the matching KB file to extract the FQN from its header
        kb_file = kb_path / qfile.name.replace("_queries.md", "_kb.md")
        fqn = bare_name
        if kb_file.exists():
            try:
                header_content = kb_file.read_text(encoding="utf-8", errors="replace")
                extracted = _fqn_from_kb_header(header_content)
                if extracted:
                    fqn = extracted
            except Exception:
                pass
        # Also try to extract FQN from the queries file itself
        if fqn == bare_name:
            try:
                q_content = qfile.read_text(encoding="utf-8", errors="replace")
                extracted = _fqn_from_kb_header(q_content)
                if extracted:
                    fqn = extracted
            except Exception:
                pass

        try:
            content = qfile.read_text(encoding="utf-8")
            questions = _extract_questions(content)
            for q in questions:
                entries.append({"table": bare_name, "fqn": fqn, "question": q})
        except Exception as e:
            log.debug("Suggestion cache: failed to parse %s: %s", qfile.name, e)

    cache_path = kb_path / _CACHE_FILENAME
    cache_path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    log.info("Suggestion cache built: %d questions from %d tables in %s",
             len(entries), len({e["table"] for e in entries}), kb_dir)
    return len(entries)


def _fqn_from_kb_header(content: str) -> str | None:
    """
    Extract FQN from the first heading of a KB markdown file.
    Recognises patterns: # DB.SCHEMA.TABLE or # SCHEMA.TABLE
    """
    import re as _re
    for line in content.splitlines()[:6]:
        stripped = line.strip().lstrip("#").strip()
        # Must look like an identifier: DB.SCHEMA.TABLE (no spaces in the FQN part)
        m = _re.match(r"^([A-Z0-9_]+\.[A-Z0-9_]+(?:\.[A-Z0-9_]+)?)(?:\s|$)",
                      stripped.upper())
        if m:
            return m.group(1)
    return None


def _extract_questions(content: str) -> list[str]:
    """Extract Q: lines from a Stage 2 *_queries.md file."""
    questions: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped.upper().startswith("Q:"):
            continue
        q = stripped[2:].strip()
        if not q:
            continue
        # Skip lines that look like SQL leaking into Q: position
        if any(q.lower().startswith(p) for p in _SKIP_PREFIXES):
            continue
        # Must look like a natural language question (contains a space, not all caps)
        if " " not in q:
            continue
        if re.fullmatch(r"[A-Z0-9_\s]+", q):
            continue
        questions.append(q)
    return questions


def _load_cache(kb_dir: str) -> list[dict]:
    """Load the suggestion cache JSON. Returns [] if missing or corrupt."""
    try:
        path = Path(kb_dir) / _CACHE_FILENAME
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _clean_identifier_part(value: str) -> str:
    return value.strip().strip("[]`\"").upper()


def _name_variants(name: str) -> set[str]:
    """
    Return comparable table-name variants for FQN, schema.table, and bare names.
    """
    raw = (name or "").strip()
    if not raw:
        return set()
    parts = [_clean_identifier_part(p) for p in re.split(r"\s*\.\s*", raw) if p.strip()]
    parts = [p for p in parts if p]
    if not parts:
        return set()
    variants = {".".join(parts), parts[-1]}
    if len(parts) >= 2:
        variants.add(".".join(parts[-2:]))
    return variants


def _matches_any_known(ref: str, known_tables: Optional[set[str]]) -> bool:
    if known_tables is None:
        return True
    ref_variants = _name_variants(ref)
    if not ref_variants:
        return False
    for known in known_tables:
        if ref_variants & _name_variants(known):
            return True
    return False


def _resolve_ref_to_known_fqn(ref: str, known_tables: Optional[set[str]]) -> str:
    """
    Prefer the schema-discovered FQN for a SQL/table ref.
    """
    ref_variants = _name_variants(ref)
    if not ref_variants:
        return ""
    if known_tables:
        for known in known_tables:
            if ref_variants & _name_variants(known):
                return known.upper()
    return next((v for v in ref_variants if "." in v), next(iter(ref_variants)))


def _extract_sql_table_refs(sql: str) -> list[str]:
    refs: list[str] = []
    for match in _SQL_TABLE_REF_RE.finditer(sql or ""):
        raw = match.group(1)
        parts = [_clean_identifier_part(p) for p in re.split(r"\s*\.\s*", raw) if p.strip()]
        if parts:
            refs.append(".".join(parts))
    return refs


def _cache_indexes(cached: list[dict]) -> tuple[dict[str, dict], dict[str, dict]]:
    by_question: dict[str, dict] = {}
    by_table: dict[str, dict] = {}
    for entry in cached:
        q = (entry.get("question") or "").strip().lower()
        if q and q not in by_question:
            by_question[q] = entry
        for ref in (entry.get("fqn") or "", entry.get("table") or ""):
            for variant in _name_variants(ref):
                by_table.setdefault(variant, entry)
    return by_question, by_table


# ── Main public function ──────────────────────────────────────────────────────

def get_suggestions(
    account_id: str,
    kb_dir: str,
    allowed_tables: Optional[set[str]],
    n: int = 6,
    schema_dir: str = "",
) -> list[dict]:
    """
    Return up to n dynamic question suggestions for the portal chat UI.

    Returns list of {"question": str, "fqn": str} dicts.
    fqn is the fully-qualified table name (DB.SCHEMA.TABLE) so the chat
    UI can pass it as a schema hint when the suggestion is clicked.

    Priority:
      1. Validated examples from SQLite
      2. Metric registry fallback
      3. Stage 2 cache as metadata only, not as raw user-facing prompts

    All sources respect allowed_tables scoping.
    Results are shuffled so each session feels fresh.
    """
    suggestions: list[dict] = []
    seen: set[str] = set()

    def _add(q: str, fqn: str = "") -> bool:
        q = q.strip()
        key = q.lower()
        if not q or key in seen or len(suggestions) >= n:
            return False
        seen.add(key)
        suggestions.append({"question": q, "fqn": fqn or ""})
        return True

    allowed_upper = (
        {t.upper() for t in allowed_tables}
        if allowed_tables is not None else None
    )

    schema_tables: Optional[set[str]] = None
    if schema_dir:
        try:
            _p = Path(schema_dir) / "_schema.json"
            if _p.exists():
                schema_tables = {t.upper() for t in json.loads(_p.read_text())}
        except Exception:
            pass

    def _table_allowed(entry: dict) -> bool:
        """
        Check if a suggestion entry passes the ACL filter.

        Compares every common table-name shape: DB.SCHEMA.TABLE,
        SCHEMA.TABLE, and the bare table name.
        """
        if allowed_upper is None:
            return True
        entry_variants: set[str] = set()
        for ref in (entry.get("fqn") or "", entry.get("table") or ""):
            entry_variants |= _name_variants(ref)
        if not entry_variants:
            return False
        for allowed in allowed_upper:
            if entry_variants & _name_variants(allowed):
                return True
        return False

    def _entry_matches_schema(entry: dict) -> bool:
        if schema_tables is None:
            return True
        for ref in (entry.get("fqn") or "", entry.get("table") or ""):
            if _matches_any_known(ref, schema_tables):
                return True
        return False

    def _entry_from_example(ex: dict, cache_by_question: dict[str, dict],
                            cache_by_table: dict[str, dict]) -> dict:
        q = (ex.get("question") or "").strip().lower()
        if q and q in cache_by_question:
            cached_entry = cache_by_question[q]
            return {
                "table": (cached_entry.get("table") or ex.get("table_name") or "").upper(),
                "fqn": (cached_entry.get("fqn") or ex.get("table_name") or "").upper(),
            }

        table_name = str(ex.get("table_name") or "").upper()
        for variant in _name_variants(table_name):
            if variant in cache_by_table:
                cached_entry = cache_by_table[variant]
                return {
                    "table": (cached_entry.get("table") or table_name).upper(),
                    "fqn": (cached_entry.get("fqn") or table_name).upper(),
                }

        for ref in _extract_sql_table_refs(ex.get("sql_query") or ""):
            fqn = _resolve_ref_to_known_fqn(ref, schema_tables)
            if fqn:
                return {"table": fqn.split(".")[-1], "fqn": fqn}

        return {"table": table_name, "fqn": table_name}

    def _metric_allowed(sql: str) -> bool:
        refs = _extract_sql_table_refs(sql)
        if not refs:
            return allowed_upper is None
        if not all(_matches_any_known(ref, schema_tables) for ref in refs):
            return False
        if allowed_upper is None:
            return True
        for ref in refs:
            entry = {
                "table": ref.split(".")[-1],
                "fqn": _resolve_ref_to_known_fqn(ref, schema_tables),
            }
            if not _table_allowed(entry):
                return False
        return True

    cached = _load_cache(kb_dir)
    cache_by_question, cache_by_table = _cache_indexes(cached)

    # Tier 1: validated examples. These have already executed successfully.
    try:
        import store
        examples = store.get_validated_examples(account_id, limit=80)
        random.shuffle(examples)
        for ex in examples:
            q = (ex.get("question") or "").strip()
            if not q:
                continue
            entry = _entry_from_example(ex, cache_by_question, cache_by_table)
            if not _table_allowed(entry) or not _entry_matches_schema(entry):
                continue
            _add(q, entry.get("fqn", ""))
            if len(suggestions) >= n:
                break
    except Exception as e:
        log.debug("Suggestion tier 1 (validated examples) failed: %s", e)

    # Tier 2: metric registry. These route through deterministic SQL templates.
    if len(suggestions) < n:
        try:
            import store
            metrics = store.list_metrics(account_id)
            random.shuffle(metrics)
            for metric in metrics:
                name = (metric.get("name") or "").strip()
                sql = (metric.get("sql_template") or "").strip()
                if not name or not sql or not _metric_allowed(sql):
                    continue
                q = f"What is our total {name.replace('_', ' ')}?"
                _add(q, "")
                if len(suggestions) >= n:
                    break
        except Exception as e:
            log.debug("Suggestion tier 2 (metric registry) failed: %s", e)

    return suggestions

    # ── Tier 1: Stage 2 query pattern cache ──────────────────────────────────
    cached = _load_cache(kb_dir)
    if cached:
        scoped = [
            e for e in cached
            if _table_allowed(e)
            and (schema_tables is None or
                 e.get("table", "").upper() in schema_tables or
                 # Also check bare part of FQN against schema_tables
                 (e.get("fqn", "").upper().split(".")[-1] in schema_tables))
        ]
        random.shuffle(scoped)
        for e in scoped:
            _add(e["question"], e.get("fqn", ""))
            if len(suggestions) >= n:
                break

    # ── Tier 2: Validated examples ────────────────────────────────────────────
    if len(suggestions) < n:
        try:
            import store
            examples = store.get_validated_examples(account_id, limit=60)
            random.shuffle(examples)
            for ex in examples:
                table_name = str(ex.get("table_name", "")).upper()
                q = (ex.get("question") or "").strip()
                if not q:
                    continue
                entry = {"table": table_name, "fqn": table_name}
                if not _table_allowed(entry):
                    continue
                _add(q, table_name)
                if len(suggestions) >= n:
                    break
        except Exception as e:
            log.debug("Suggestion tier 2 (validated examples) failed: %s", e)

    # ── Tier 3: Metric registry fallback ─────────────────────────────────────
    if len(suggestions) < n:
        try:
            import store
            metrics = store.list_metrics(account_id)
            random.shuffle(metrics)
            for metric in metrics:
                name = (metric.get("name") or "").strip()
                desc = (metric.get("description") or "").strip()
                sql  = (metric.get("sql_template") or "").upper()
                if not name:
                    continue
                if schema_tables and sql:
                    tables_in_sql = {
                        w.strip("[];()") for w in sql.split()
                        if "." in w or w.strip("[];()").replace("_", "").isalpha()
                    }
                    if not any(
                        any(t in ref.upper() for ref in tables_in_sql)
                        for t in schema_tables
                    ):
                        continue
                human = name.replace("_", " ")
                label = desc if (desc and len(desc) <= 60) else human
                q = f"What is our total {label}?"
                _add(q, "")
                if len(suggestions) >= n:
                    break
        except Exception as e:
            log.debug("Suggestion tier 3 (metric registry) failed: %s", e)

    return suggestions
