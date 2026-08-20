"""
core/suggestions.py

Dynamic question suggestion engine for the portal chat UI.

Sources, in descending order of trust — which is also the order they are used:
  1. Validated examples (SQLite) — questions PROVEN against the real database.
     Only available once Stage 2 validation has run.
  2. Metric registry — admin-defined metrics formatted as natural questions.
     Deterministic SQL, so they cannot fail for planning reasons.
  3. Stage 2 *_queries.md files — questions the LLM WROTE during KB generation.
     Never executed, so least trustworthy; available from day 1, which is why
     they are kept at all. Cached as suggested_questions.json for fast loads,
     and filtered against the entity graph before being offered.

A suggestion is a promise: the user clicks a button this module supplied, so
offering one that cannot be answered is worse than offering nothing. Tier 3 in
particular used to run FIRST, which is how the portal came to show clickable
questions that failed — the least trustworthy source filled the panel before
the proven ones were consulted.

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


def prune_suggestion_cache(kb_dir: str, valid_questions: set[str]) -> int:
    """Drop cached suggestions whose SQL did not survive Stage-2 validation.

    The cache is written from every Q:/SQL: pair in the Stage-2 files at KB
    build time — BEFORE validation runs — and nothing ever revisited it with
    the results. A question whose SQL failed to compile against the real
    database stayed in the panel indefinitely, and clicking it put the user
    through the pipeline for a question the product had already established it
    could not answer.

    Returns the number of entries removed.
    """
    kb_path = Path(kb_dir)
    cache_path = kb_path / _CACHE_FILENAME
    if not valid_questions or not cache_path.exists():
        return 0
    try:
        entries = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Suggestion cache could not be pruned (%s)", exc)
        return 0

    keep_keys = {str(q or "").strip().lower() for q in valid_questions}
    kept = [
        entry for entry in entries
        if str(entry.get("question") or "").strip().lower() in keep_keys
    ]
    removed = len(entries) - len(kept)
    if not removed:
        return 0
    try:
        cache_path.write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        log.warning("Suggestion cache could not be rewritten after pruning (%s)", exc)
        return 0
    log.info(
        "Suggestion cache pruned for %s: %d question(s) removed whose SQL did "
        "not validate, %d kept", kb_dir, removed, len(kept),
    )
    return removed


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


def _date_scope_check(kb_dir: str):
    """Withhold a date-scoped suggestion when no approved date role exists.

    "Revenue last month" only has a governed answer if some fact carries an
    admin-approved date role — that is what resolves the window against the
    business-date anchor. With none, the question either gets refused or falls
    through to a wall-clock date, and either way clicking the chip is a dead
    end. Nothing checked this at suggestion time: a workspace mid-onboarding
    offered date questions its own metadata could not answer.

    Cheap by construction — it reads the compiled model already on disk and
    makes no database call, so it can run per candidate during a page render.
    Fails OPEN like the reachability gate: it can only ever remove questions
    we can prove are dead ends.
    """
    try:
        from core.contextual_dates import detect_temporal_window
        from core.semantic_model import find_default_date_roles

        has_role = bool(find_default_date_roles(kb_dir))
    except Exception as exc:
        log.debug("Suggestion date-scope check unavailable: %s", exc)
        return lambda _question: True

    if has_role:
        return lambda _question: True

    def _check(question: str) -> bool:
        try:
            if not detect_temporal_window(question or ""):
                return True
        except Exception:
            return True
        log.debug(
            "Withholding date-scoped suggestion %r — no approved date role "
            "exists to resolve the period against", question,
        )
        return False

    return _check


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
        # Clarification retries retain an internal wrapper in the audit log so
        # SQL generation can reproduce the resolved choice. That metadata is
        # not a user-facing question and must not leak into starter cards.
        try:
            from core.clarification import extract_original_question
            q = extract_original_question(q)
        except Exception:
            pass
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

    # Loaded once and shared by tiers 1 and 3. Pure and in-memory — no LLM, no
    # warehouse round-trip — so every candidate is free to test.
    _reachable = _graph_reachability_check(account_id)
    _date_scope_is_answerable = _date_scope_check(kb_dir)

    # Tier 1: validated examples. "Validated" is two different guarantees
    # wearing one name, and the weaker one was ranked as if it were the
    # stronger. A kb_stage2 example is SQL the LLM authored during the KB build
    # and core.examples COMPILE-checked (sp_describe_first_result_set and its
    # equivalents) — that proves the tables, columns and syntax resolve, and
    # nothing whatever about whether the question returns anything. A query_log
    # example is a question a user actually asked that actually came back with
    # rows. Proven-answerable ones go first.
    #
    # Neither says the NL pipeline can re-plan the sentence from its text, so
    # both still pass through the same reachability gate as tier 3.
    try:
        import store
        examples = store.get_validated_examples(account_id, limit=80)
        random.shuffle(examples)
        examples.sort(
            key=lambda ex: 0 if str(ex.get("source") or "") == "query_log" else 1
        )
        for ex in examples:
            q = (ex.get("question") or "").strip()
            if not q:
                continue
            entry = _entry_from_example(ex, cache_by_question, cache_by_table)
            if not _table_allowed(entry) or not _entry_matches_schema(entry):
                continue
            if not _reachable(q):
                continue
            if not _date_scope_is_answerable(q):
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
                # Gated like every other tier. A metric's SQL template having
                # once been valid says nothing about whether the NL pipeline can
                # re-plan this sentence: the question goes back through entity
                # detection and the pathfinder like any other, and a metric
                # spanning entities the graph cannot join dead-ends on the same
                # "Missing governed path" the user sees from a typed question.
                # This tier was the one source offered unchecked.
                if not _reachable(q) or not _date_scope_is_answerable(q):
                    continue
                _add(q, "")
                if len(suggestions) >= n:
                    break
        except Exception as e:
            log.debug("Suggestion tier 2 (metric registry) failed: %s", e)

    # ── Tier 3: Stage 2 query patterns — LAST, and only what the graph can reach
    #
    # These are questions the LLM WROTE while generating the knowledge base.
    # They have never been executed, so they are the least trustworthy source
    # here. They still earn a place when the safer tiers cannot fill the list
    # (a new tenant has no validated examples yet), but only if the entity
    # graph can actually reach what they name.
    # Unlike tier 1, this tier is only reached when the check actually ran. An
    # unreadable or empty graph vouches for nothing, and the least trustworthy
    # source in the function is exactly the one that must not be offered on the
    # strength of a check that did not happen.
    if len(suggestions) < n and cached and getattr(_reachable, "verified", False):
        scoped = [e for e in cached if _table_allowed(e) and _entry_matches_schema(e)]
        random.shuffle(scoped)
        for e in scoped:
            question_text = str(e.get("question") or "").strip()
            if not question_text or not _reachable(question_text):
                continue
            if not _date_scope_is_answerable(question_text):
                continue
            _add(question_text, e.get("fqn", ""))
            if len(suggestions) >= n:
                break

    return suggestions


def _graph_reachability_check(account_id: str):
    """Return a predicate: can the entity graph reach what this question names?

    Offering a question the graph cannot resolve is worse than offering nothing
    -- the user clicks a button we supplied and is told to go ask an
    administrator. The check reuses the resolver the real pipeline runs before
    SQL generation, which is pure and in-memory (no LLM, no warehouse query), so
    the graph is loaded once here and every candidate is free to test.

    Fails OPEN: any error, or a graph that is empty or review-only, returns True
    so this can only ever remove questions we can prove are dead ends.

    The returned predicate carries ``.verified`` — False when it fell open and
    is therefore vouching for nothing. Failing open is right when FILTERING a
    trusted source (never withhold a proven question just because the graph is
    unreadable) and wrong when PROMOTING an untrusted one, so the caller can
    tell the two apart.
    """
    def _open(_reason: str):
        predicate = lambda question: True  # noqa: E731
        predicate.verified = False
        return predicate

    try:
        import store
        from core.graph_resolver import detect_entities

        graph = store.get_full_graph(account_id) or {}
        if not (graph.get("entities") or []):
            return _open("empty graph")
    except Exception as exc:
        log.debug("Suggestion reachability check unavailable: %s", exc)
        return _open(str(exc))

    # Entities that carry no relationship at all, but whose physical table is
    # also owned by a sibling entity that does. That pairing is always a defect
    # -- a duplicate entity minted for a table that already had one -- and it is
    # fatal rather than merely untidy: the detector matches the question to the
    # jointless twin by name, the pathfinder cannot reach it, and the whole
    # question is refused with "Missing governed path for: <name>". Live example
    # on this shape: "Customer" with zero edges beside "Cus Dms" with fifteen.
    connected: set[str] = set()
    for rel in graph.get("relationships") or []:
        for side in ("from_entity", "to_entity"):
            if rel.get(side):
                connected.add(str(rel[side]))

    def _table_of(entity: dict) -> str:
        table = str(entity.get("table_name") or "").strip().upper()
        schema = str(entity.get("schema_name") or "").strip().upper()
        return f"{schema}.{table}" if schema else table

    tables_with_a_connected_owner = {
        _table_of(entity)
        for entity in graph.get("entities") or []
        if str(entity.get("entity_name") or "") in connected and _table_of(entity)
    }
    jointless_twins = {
        str(entity.get("entity_name") or "")
        for entity in graph.get("entities") or []
        if str(entity.get("entity_name") or "") not in connected
        and _table_of(entity) in tables_with_a_connected_owner
    }
    # Which entities can actually be joined to which. A question naming two
    # entities in different components has no governed path between them, and
    # the pipeline refuses it with "Missing governed path" AFTER the user has
    # clicked a button this module supplied.
    #
    # This is the general case; the jointless-twin rule below is one instance of
    # it. Until now only that instance was checked, so on any graph without a
    # twin the predicate returned True for every question and reported itself
    # verified — the gate looked like it was working precisely when it was
    # doing nothing.
    # Built from the SAME subgraph the pipeline will plan on, not from every
    # row in the graph. The pipeline narrows to admin-confirmed entities and
    # relationships unless the client has explicitly opted into unreviewed
    # ones, so a question joinable only through a suggested edge passed this
    # gate and was then refused with "Missing governed path" after the user
    # clicked the chip — the gate vouching for a path the planner would not
    # use. Broken edges are excluded here for the same reason the pathfinder
    # excludes them.
    try:
        from core.graph_resolver import (
            _client_allows_suggested, _confirmed_subgraph,
        )
        planning_graph = (
            graph if _client_allows_suggested(account_id) else _confirmed_subgraph(graph)
        )
    except Exception as exc:
        log.debug("Suggestion gate could not narrow to the planning subgraph: %s", exc)
        planning_graph = graph

    _adjacency: dict[str, set[str]] = {}
    for rel in planning_graph.get("relationships") or []:
        if str(rel.get("validation_status") or "").lower() == "broken":
            continue
        a, b = str(rel.get("from_entity") or ""), str(rel.get("to_entity") or "")
        if a and b:
            _adjacency.setdefault(a, set()).add(b)
            _adjacency.setdefault(b, set()).add(a)

    _component: dict[str, int] = {}
    for start in _adjacency:
        if start in _component:
            continue
        marker = len(_component)
        stack = [start]
        while stack:
            node = stack.pop()
            if node in _component:
                continue
            _component[node] = marker
            stack.extend(_adjacency.get(node, ()) - _component.keys())

    def _spans_components(detected: set[str]) -> set[str]:
        """Entities that cannot be joined to the rest of what the question names.

        A single entity is always fine — a one-table question needs no path.
        """
        if len(detected) < 2:
            return set()
        seen = {name: _component.get(name) for name in detected}
        groups = {marker for marker in seen.values() if marker is not None}
        unjoinable = {name for name, marker in seen.items() if marker is None}
        if len(groups) > 1:
            # More than one island: report the minority side as the blocker.
            majority = max(groups, key=lambda g: sum(1 for m in seen.values() if m == g))
            unjoinable |= {n for n, m in seen.items() if m is not None and m != majority}
        return unjoinable
    log.info(
        "Suggestion filter active — %d entity/entities have no relationships "
        "while a sibling on the same table does: %s",
        len(jointless_twins), sorted(jointless_twins),
    )

    def _reachable(question: str) -> bool:
        try:
            detected = set(detect_entities(question, graph))
        except Exception as exc:
            log.debug("Suggestion reachability check failed for %r: %s", question, exc)
            return True

        blocked_by = detected & jointless_twins
        if blocked_by:
            log.info(
                "Suggestion withheld — %r resolves to %s, which has no governed "
                "relationships; a sibling entity on the same table does",
                question, sorted(blocked_by),
            )
            return False

        unjoinable = _spans_components(detected)
        if unjoinable:
            log.info(
                "Suggestion withheld — %r names %s, and %s cannot be joined to "
                "the rest through any governed relationship",
                question, sorted(detected), sorted(unjoinable),
            )
            return False
        return True

    _reachable.verified = True
    return _reachable
