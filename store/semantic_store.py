"""
store/semantic_store.py

Business semantic layer — CRUD + term matching.

The business_term table is the canonical source of truth for business
concepts. Every layer that needs to understand business meaning queries
this table rather than parsing KB markdown:

  - Clarification layer uses match_terms_in_question() to detect which
    known terms appear in the user's question, and uses
    clarification_options JSON for precise disambiguation.

  - SQL generation prompt injects resolved terms as grounding hints
    via build_term_injection().

  - KB generation auto-populates terms from the "Business Synonyms"
    section of Stage 1 docs (extract_terms_from_kb).

  - Admin UI does full CRUD through this module.
"""

import json
import logging
import re
from typing import Optional

from store.db import get_db

log = logging.getLogger("querybot.semantic")


# ══════════════════════════════════════════════════════════════════════════════
# CRUD
# ══════════════════════════════════════════════════════════════════════════════

def save_term(
    account_id: str,
    term: str,
    kind: str = "metric",
    canonical_expression: str = "",
    tables_involved: str = "",
    grain: str = "",
    aliases: str = "",
    definition: str = "",
    requires_clarification: bool = False,
    clarification_options: Optional[list[dict]] = None,
    source: str = "manual",
    term_id: Optional[int] = None,
) -> int:
    """Insert or update a business term. Returns the term id."""
    opts_json = json.dumps(clarification_options) if clarification_options else ""
    # Normalize term and aliases for reliable matching
    term_norm = term.strip().lower()
    aliases_norm = ",".join(
        s.strip().lower() for s in aliases.split(",") if s.strip()
    )
    with get_db() as conn:
        if term_id:
            conn.execute(
                """
                UPDATE business_term SET
                    term = ?, kind = ?, canonical_expression = ?,
                    tables_involved = ?, grain = ?, aliases = ?,
                    definition = ?, requires_clarification = ?,
                    clarification_options = ?, source = ?,
                    updated_at = datetime('now')
                WHERE id = ? AND account_id = ?
                """,
                (term_norm, kind, canonical_expression, tables_involved,
                 grain, aliases_norm, definition,
                 1 if requires_clarification else 0, opts_json, source,
                 term_id, account_id),
            )
            return term_id
        try:
            cur = conn.execute(
                """
                INSERT INTO business_term
                    (account_id, term, kind, canonical_expression,
                     tables_involved, grain, aliases, definition,
                     requires_clarification, clarification_options, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (account_id, term_norm, kind, canonical_expression,
                 tables_involved, grain, aliases_norm, definition,
                 1 if requires_clarification else 0, opts_json, source),
            )
            return cur.lastrowid
        except Exception as e:
            # UNIQUE violation — update existing instead
            if "UNIQUE" in str(e):
                existing = conn.execute(
                    "SELECT id FROM business_term WHERE account_id = ? AND term = ?",
                    (account_id, term_norm),
                ).fetchone()
                if existing:
                    return save_term(
                        account_id, term, kind, canonical_expression,
                        tables_involved, grain, aliases, definition,
                        requires_clarification, clarification_options,
                        source, term_id=existing["id"],
                    )
            raise


def get_term(term_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM business_term WHERE id = ?", (term_id,)
        ).fetchone()
    if not row:
        return None
    return _row_to_dict(row)


def list_terms(
    account_id: str,
    kind: Optional[str] = None,
    active_only: bool = True,
) -> list[dict]:
    where = ["account_id = ?"]
    params: list = [account_id]
    if active_only:
        where.append("is_active = 1")
    if kind:
        where.append("kind = ?")
        params.append(kind)
    sql = (
        f"SELECT * FROM business_term WHERE {' AND '.join(where)} "
        "ORDER BY kind, term"
    )
    with get_db() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_dict(r) for r in rows]


def delete_term(term_id: int, account_id: str) -> None:
    """Hard delete — caller should confirm. Scoped to account_id for safety."""
    with get_db() as conn:
        conn.execute(
            "DELETE FROM business_term WHERE id = ? AND account_id = ?",
            (term_id, account_id),
        )


def set_term_active(term_id: int, account_id: str, active: bool) -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE business_term SET is_active = ?, updated_at = datetime('now') "
            "WHERE id = ? AND account_id = ?",
            (1 if active else 0, term_id, account_id),
        )


def _row_to_dict(row) -> dict:
    d = dict(row)
    # Parse JSON fields
    raw_opts = d.get("clarification_options") or ""
    try:
        d["clarification_options"] = json.loads(raw_opts) if raw_opts else []
    except Exception:
        d["clarification_options"] = []
    return d


# ══════════════════════════════════════════════════════════════════════════════
# Term matching — called at clarification time
# ══════════════════════════════════════════════════════════════════════════════

def match_terms_in_question(
    account_id: str,
    question: str,
    allowed_tables: Optional[set[str]] = None,
    terms: Optional[list[dict]] = None,
) -> list[dict]:
    """
    Find all business terms that appear in the user's question.
    
    Matches whole-word against both the canonical term and its aliases.
    If allowed_tables is set, filters out terms whose tables_involved
    reference tables the user can't access (so we don't leak schema
    information through clarification).
    
    Returns matched terms ordered by match specificity — longer
    matches first, so "active customer" beats "customer".
    """
    q_lower = question.lower()
    if terms is None:
        terms = list_terms(account_id, active_only=True)

    matches: list[tuple[int, dict]] = []  # (match_length, term_dict)
    for t in terms:
        all_forms = [t["term"]]
        if t.get("aliases"):
            all_forms.extend(
                s.strip() for s in t["aliases"].split(",") if s.strip()
            )

        for form in all_forms:
            form_lower = form.lower().strip()
            if not form_lower:
                continue
            pattern = r"\b" + re.escape(form_lower) + r"\b"
            if re.search(pattern, q_lower):
                # Scope filter — skip terms whose tables aren't accessible
                if allowed_tables is not None and t.get("tables_involved"):
                    term_tables = {
                        s.strip().upper()
                        for s in t["tables_involved"].split(",")
                        if s.strip()
                    }
                    # Keep term only if at least one of its tables is allowed
                    if term_tables and not (term_tables & allowed_tables):
                        continue
                matches.append((len(form_lower), t))
                break  # don't double-count one term via multiple aliases

    # Sort longest-match-first so specific terms beat generic ones
    matches.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in matches]


def find_ambiguous_term(
    account_id: str,
    question: str,
    allowed_tables: Optional[set[str]] = None,
) -> Optional[dict]:
    """
    Return the first matched term that requires clarification, or None.
    
    This is the fast-path check: if a term in the question has
    requires_clarification=true, we ask the user to pick from the
    predefined options rather than calling the ambiguity LLM.
    """
    matches = match_terms_in_question(account_id, question, allowed_tables)
    for t in matches:
        if t.get("requires_clarification") and t.get("clarification_options"):
            return t
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Prompt injection — for SQL retry after clarification
# ══════════════════════════════════════════════════════════════════════════════

def _metric_synonym_set(account_id: str) -> set[str]:
    """
    Return a lower-cased set of all name + synonym phrases from active metrics
    for account_id.  Used by build_term_injection to suppress business-term
    entries that collide with a registered metric, so the metric always wins.
    """
    try:
        from store.config_store import list_metrics as _list_metrics
    except ImportError:
        return set()
    phrases: set[str] = set()
    for m in _list_metrics(account_id, active_only=True):
        name = (m.get("name") or "").strip().lower()
        if name:
            phrases.add(name)
        for s in (m.get("synonyms") or "").split(","):
            s = s.strip().lower()
            if s:
                phrases.add(s)
    return phrases


def build_term_injection(
    terms_or_account_id,
    question: Optional[str] = None,
    allowed_tables: Optional[set[str]] = None,
    max_terms: int = 5,
    terms: Optional[list[dict]] = None,
) -> str:
    """
    Build a compact SQL-prompt block from resolved business terms.

    Two call shapes are accepted:

    1. build_term_injection(terms_list)
       Pass a pre-fetched list of term dicts (e.g. from match_terms_in_question).

    2. build_term_injection(account_id, question, allowed_tables)
       Convenience form — does the matching internally.

    This is appended to the SQL generation system prompt. It tells the
    LLM exactly what SQL fragment to use for each matched term, so the
    model doesn't have to guess.

    Metric-formula priority guarantee
    ──────────────────────────────────
    Any business term whose canonical term name OR any alias exactly matches
    a metric name or synonym is silently suppressed here.  The metric formula
    context block (APPROVED METRIC FORMULAS) is always injected separately and
    must take precedence — see list_metric_formula_context / _format_metric_formula_context.
    This prevents the LLM from using a raw column synonym when the admin has
    registered a computed formula for the same concept.
    """
    account_id: Optional[str] = None

    # Disambiguate call shape
    if isinstance(terms_or_account_id, str):
        # Shape 2: (account_id, question, allowed_tables). The `terms` kwarg
        # (from the compiled semantic contract) overrides the DB read for the
        # candidate pool; matching still happens against the question here.
        if not question:
            return ""
        account_id = terms_or_account_id
        terms = match_terms_in_question(
            account_id, question, allowed_tables, terms=terms,
        )
    else:
        # Shape 1: terms_list
        terms = terms_or_account_id or []

    if not terms:
        return ""

    # ── Metric-collision guard ──────────────────────────────────────────────
    # Build the metric synonym set once per call (only when we have account_id)
    metric_phrases: set[str] = _metric_synonym_set(account_id) if account_id else set()

    def _term_collides_with_metric(t: dict) -> bool:
        """Return True if this business term overlaps with any metric synonym."""
        if not metric_phrases:
            return False
        all_forms = [t.get("term", "")]
        for alias in (t.get("aliases") or t.get("synonyms") or "").split(","):
            alias = alias.strip()
            if alias:
                all_forms.append(alias)
        return any(f.lower() in metric_phrases for f in all_forms if f)

    filtered_terms = [t for t in terms if not _term_collides_with_metric(t)]
    # ──────────────────────────────────────────────────────────────────────────

    if not filtered_terms:
        return ""
    lines = [
        "BUSINESS TERM DEFINITIONS (use these EXACT expressions when the "
        "user's question mentions these terms — do not substitute your own):"
    ]
    for t in filtered_terms[:max_terms]:
        expr = (t.get("canonical_expression") or "").strip()
        if not expr:
            continue
        kind = t.get("kind", "metric")
        line = f"  • {t['term']} ({kind}): `{expr}`"
        if t.get("definition"):
            line += f" — {t['definition'][:100]}"
        lines.append(line)
    if len(lines) == 1:
        return ""  # no terms had usable expressions
    return "\n".join(lines) + "\n"


def build_term_injection_from_choice(
    term_or_meta: dict, chosen_label: str
) -> str:
    """
    When a clarification has been resolved by user choice, build an
    injection string that locks the chosen interpretation.
    
    Accepts either:
      - A term dict (with `clarification_options` inside), OR
      - A clarification meta dict (with `options` at the top level, as
        saved by check_ambiguity_glossary_first).
    """
    # Accept either shape
    opts = (
        term_or_meta.get("options")
        or term_or_meta.get("clarification_options")
        or []
    )
    term_name = term_or_meta.get("term", "this concept")

    if not opts:
        return ""

    chosen_norm = chosen_label.lower().strip()
    chosen = None

    # 1. Exact match
    for o in opts:
        if o.get("label", "").lower().strip() == chosen_norm:
            chosen = o
            break

    # 2. Substring match (either direction)
    if not chosen:
        for o in opts:
            label_norm = o.get("label", "").lower()
            if chosen_norm in label_norm or label_norm in chosen_norm:
                chosen = o
                break

    # 3. Word-overlap match — handles paraphrased replies like
    #    user says "number of late days" vs label "Number of late/absent days"
    if not chosen:
        chosen_words = {w for w in re.split(r"\W+", chosen_norm) if len(w) >= 3}
        if chosen_words:
            best_score = 0
            for o in opts:
                label_words = {
                    w for w in re.split(r"\W+", o.get("label", "").lower())
                    if len(w) >= 3
                }
                score = len(chosen_words & label_words)
                if score > best_score:
                    best_score = score
                    chosen = o
            # Require at least 2 overlapping words to be confident
            if best_score < 2:
                chosen = None

    # 4. Single-option fallback: if there's only one option and the user
    #    replied with anything affirmative, use it
    if not chosen and len(opts) == 1:
        chosen = opts[0]

    if not chosen:
        return ""

    expr = chosen.get("expression", "")
    definition = chosen.get("definition", "")
    return (
        f"RESOLVED BUSINESS TERM — the user has clarified that by "
        f"'{term_name}' they mean: {chosen.get('label')}.\n"
        f"Use this EXACT SQL expression for this concept: `{expr}`\n"
        + (f"Definition: {definition}\n" if definition else "")
    )


# ══════════════════════════════════════════════════════════════════════════════
# Auto-extraction from KB
# ══════════════════════════════════════════════════════════════════════════════

def extract_terms_from_kb(
    account_id: str,
    kb_dir: str,
) -> int:
    """
    Parse Stage 1 KB markdown files and auto-populate business_term
    with entries from the "Business Synonyms" and "Key Metrics" sections.
    
    Safe to re-run — existing terms with source='kb_extracted' are
    updated; 'manual' entries are never overwritten.
    
    Returns number of terms added/updated.
    """
    from pathlib import Path

    kb_path = Path(kb_dir)
    if not kb_path.exists():
        return 0

    added = 0
    # Load existing terms once — don't clobber manual entries
    existing = {t["term"]: t for t in list_terms(account_id, active_only=False)}

    for md_file in sorted(kb_path.glob("*_kb.md")):
        table_name = md_file.stem.replace("_kb", "").upper()
        try:
            content = md_file.read_text(encoding="utf-8")
        except Exception:
            continue

        # Extract from "Key Metrics" section — metric definitions
        metric_rows = _extract_key_metrics(content, table_name)
        for metric in metric_rows:
            term_key = metric["term"]
            if term_key in existing and existing[term_key].get("source") == "manual":
                continue  # never overwrite manual entries
            try:
                save_term(
                    account_id=account_id,
                    term=metric["term"],
                    kind="metric",
                    canonical_expression=metric.get("expression", ""),
                    tables_involved=table_name,
                    aliases=metric.get("aliases", ""),
                    definition=metric.get("definition", ""),
                    source="kb_extracted",
                    term_id=existing.get(term_key, {}).get("id"),
                )
                added += 1
            except Exception as e:
                log.debug("Failed to save term %s: %s", term_key, e)

        # Extract from "Business Synonyms" table — dimension/filter mappings
        synonym_rows = _extract_synonyms(content, table_name)
        for syn in synonym_rows:
            term_key = syn["term"]
            if term_key in existing and existing[term_key].get("source") == "manual":
                continue
            try:
                save_term(
                    account_id=account_id,
                    term=syn["term"],
                    kind=syn.get("kind", "dimension"),
                    canonical_expression=syn.get("column", ""),
                    tables_involved=table_name,
                    aliases=syn.get("aliases", ""),
                    definition=syn.get("notes", ""),
                    source="kb_extracted",
                    term_id=existing.get(term_key, {}).get("id"),
                )
                added += 1
            except Exception as e:
                log.debug("Failed to save synonym %s: %s", term_key, e)

    log.info("Auto-extracted %d business terms for %s", added, account_id)
    return added


def _extract_key_metrics(content: str, table_name: str) -> list[dict]:
    """Parse the 'Key Metrics' section of a Stage 1 KB doc."""
    results = []
    lines = content.splitlines()
    in_section = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            in_section = stripped.lower().startswith("## key metrics")
            continue
        if not in_section or not stripped.startswith("-"):
            continue
        # Parse format: - **Metric name**: `COLUMN_NAME` — Filter: `WHERE ...`
        m = re.match(
            r"-\s*\*\*([^*]+)\*\*\s*:?\s*`?([^`\n]*)`?\s*(?:—|-)?\s*(.*)$",
            stripped,
        )
        if not m:
            continue
        term = m.group(1).strip().lower()
        expr = m.group(2).strip()
        notes = m.group(3).strip()
        if not term:
            continue
        # If expression is just a column name, wrap it as SUM() for metrics
        # that read like aggregatable quantities
        if expr and not any(
            expr.upper().startswith(agg)
            for agg in ("SUM(", "COUNT(", "AVG(", "MIN(", "MAX(", "CASE")
        ):
            # Leave it as-is — the SQL prompt will wrap it appropriately.
            pass
        results.append({
            "term": term,
            "expression": expr,
            "definition": notes[:200],
            "aliases": "",
        })
    return results


def _extract_synonyms(content: str, table_name: str) -> list[dict]:
    """Parse the 'Business Synonyms' markdown table."""
    results = []
    lines = content.splitlines()
    in_section = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            in_section = stripped.lower().startswith("## business synonyms")
            continue
        if not in_section:
            continue
        # Table row: | plain english | COLUMN | notes |
        if not stripped.startswith("|"):
            continue
        # Skip header and separator rows
        if "plain english" in stripped.lower() or stripped.startswith("|---"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 2:
            continue
        # Aliases often come as comma-separated in first cell
        raw_terms = cells[0]
        column = cells[1].strip("`").strip()
        notes = cells[2] if len(cells) > 2 else ""
        # Skip WARNING-only rows
        if raw_terms.upper().startswith("WARNING"):
            continue
        term_list = [
            s.strip().lower() for s in raw_terms.split(",") if s.strip()
        ]
        if not term_list or not column:
            continue
        # First term becomes canonical, rest become aliases
        primary = term_list[0]
        aliases = ",".join(term_list[1:]) if len(term_list) > 1 else ""
        results.append({
            "term": primary,
            "column": column,
            "aliases": aliases,
            "kind": "dimension",
            "notes": notes[:200],
        })
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Stats — for admin UI
# ══════════════════════════════════════════════════════════════════════════════

def glossary_stats(account_id: str) -> dict:
    """Return counts by kind and source for the admin UI dashboard."""
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT kind, source, COUNT(*) AS n
            FROM business_term
            WHERE account_id = ? AND is_active = 1
            GROUP BY kind, source
            """,
            (account_id,),
        ).fetchall()
    result = {"total": 0, "by_kind": {}, "by_source": {}, "needing_review": 0}
    for r in rows:
        result["total"] += r["n"]
        result["by_kind"][r["kind"]] = result["by_kind"].get(r["kind"], 0) + r["n"]
        result["by_source"][r["source"]] = result["by_source"].get(r["source"], 0) + r["n"]

    # Terms that need clarification but have no options defined = needs review
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS n FROM business_term
            WHERE account_id = ? AND is_active = 1
              AND requires_clarification = 1
              AND (clarification_options = '' OR clarification_options IS NULL)
            """,
            (account_id,),
        ).fetchone()
    result["needing_review"] = row["n"] if row else 0

    # Total terms flagged as requires_clarification (regardless of whether
    # their options are defined) — useful for the admin dashboard header.
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS n FROM business_term
            WHERE account_id = ? AND is_active = 1
              AND requires_clarification = 1
            """,
            (account_id,),
        ).fetchone()
    result["requires_clarification"] = row["n"] if row else 0
    return result
