"""
core/clarification.py  —  HARDENED v2

Business-aware clarification layer.

Changes vs v1 (see CHANGES.md at repo root for rationale):
  • Fix #1  — LLM ambiguity check uses a CONSTRAINED MENU built from the
              real glossary. Returns real option IDs (backed by term_ids)
              instead of an empty options list. Retry SQL prompt now gets
              a locked-in business term expression on LLM-sourced clars.
  • Fix #2  — combine_with_clarification accepts an explicit
              selected_option_id so the resolved option chosen by the
              dispatcher is used verbatim, not re-matched by substring.
  • Fix #5  — _parse_ambiguity_json is a tolerant JSON parser that
              handles ```json fences, preamble text, and brace-balanced
              extraction. No more silent fail-open on parse errors.
  • Fix #7  — mark_recently_expired / was_recently_expired let the
              dispatcher tell a user their clarification expired rather
              than silently treating their reply as a fresh query.

Flow (unchanged):
  1. SQL generation returns CANNOT_GENERATE or zero rows
  2. check_ambiguity_glossary_first() runs:
     a. Match against business_term glossary (requires_clarification=1)
     b. Multi-metric overlap
     c. CONSTRAINED-MENU LLM ambiguity check (options from real glossary)
  3. If AMBIGUOUS → save pending_clarification in SQLite, send question
  4. User reply → combine_with_clarification() enriches the rewrite
     with the resolved term expression for the retry SQL prompt
  5. Combined question + term_injection goes back to handle_query()

Pending state expires in 5 minutes; a 10-minute grace trail in memory
lets us tell users their clarification lapsed.
"""

import json
import logging
import re
import time
from difflib import SequenceMatcher
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional

from core.query_semantics import analyze_query_intent, summarize_query_intent
from core.semantic_registry import find_registry_clarification, validated_options

log = logging.getLogger("querybot.clarification")

_EXPIRE_MINUTES = 5

# ── KB annotation noise patterns that must never appear in chip labels ────────
_KB_NOISE_RES = [
    re.compile(r'\[NEEDS\s+CONTEXT[^\]]*\]', re.I),
    re.compile(r'\[SPLIT\s+NAME[^\]]*\]', re.I),
    re.compile(r'PRIMARY\s+MEASURE\s+FOR\s+[^.]+\.?', re.I),
    re.compile(r'business\s+rule\s+unknown[^.]*\.?', re.I),
    re.compile(r'do\s+not\s+use\s+in\s+filters\.?', re.I),
    re.compile(r'always\s+(concatenate|exclude)[^.]*\.?', re.I),
]


def _clean_chip_label(definition: str, term_name: str = "") -> str:
    """
    Return a short, user-friendly label for a clarification chip.

    Strips KB annotation markers ([NEEDS CONTEXT], PRIMARY MEASURE FOR…,
    etc.) that are meaningful to the SQL prompt but confusing to end users.
    Falls back to the term name if the cleaned text is too short or still
    contains KB syntax noise.
    """
    fallback = (term_name or "").replace("_", " ").strip().title() or "Option"

    text = (definition or "").strip()
    if not text:
        return fallback

    # Strip all noise patterns
    for pattern in _KB_NOISE_RES:
        text = pattern.sub("", text)

    # Strip leading/trailing punctuation and whitespace left behind
    text = re.sub(r"^[\s.,;:\-–—]+|[\s.,;:\-–—]+$", "", text)
    # Collapse multiple spaces
    text = re.sub(r"\s{2,}", " ", text)

    # If still looks noisy or too short, fall back
    if len(text) < 4 or re.search(r"\[|\]|PRIMARY\s+MEASURE|NEEDS\s+CONTEXT", text, re.I):
        return fallback

    # Truncate for display
    if len(text) > 55:
        text = text[:52].rstrip() + "…"

    return text
_RECENT_EXPIRY_GRACE_SECONDS = 600  # 10 min — for "your clarification expired" hint

# ──────────────────────────────────────────────────────────────────────────────
# In-memory trail of recently-cleared pending clarifications (Fix #7)
# ──────────────────────────────────────────────────────────────────────────────
# Map (account_id, zoom_user_id) → monotonic timestamp of expiry.
# In-process; fine for a single-worker deployment. For multi-worker, move to
# Redis or add an 'expired_at' column to pending_clarification.
_RECENT_EXPIRED: dict[tuple[str, str], float] = {}


def mark_recently_expired(account_id: str, zoom_user_id: str) -> None:
    """Record that a pending clarification just expired."""
    _RECENT_EXPIRED[(account_id, zoom_user_id)] = time.monotonic()
    if len(_RECENT_EXPIRED) > 2048:  # opportunistic GC
        now = time.monotonic()
        stale = [k for k, ts in _RECENT_EXPIRED.items()
                 if now - ts > _RECENT_EXPIRY_GRACE_SECONDS]
        for k in stale:
            _RECENT_EXPIRED.pop(k, None)


def was_recently_expired(account_id: str, zoom_user_id: str) -> bool:
    """Return True if a clarification for this user expired in the grace window."""
    ts = _RECENT_EXPIRED.get((account_id, zoom_user_id))
    if not ts:
        return False
    if time.monotonic() - ts > _RECENT_EXPIRY_GRACE_SECONDS:
        _RECENT_EXPIRED.pop((account_id, zoom_user_id), None)
        return False
    return True


def acknowledge_recently_expired(account_id: str, zoom_user_id: str) -> None:
    """Drop the recent-expired marker after we've surfaced it to the user."""
    _RECENT_EXPIRED.pop((account_id, zoom_user_id), None)


# ══════════════════════════════════════════════════════════════════════════════
# Option normalisation and matching
# ══════════════════════════════════════════════════════════════════════════════

def _with_option_ids(options: list[dict]) -> list[dict]:
    normalized = []
    for idx, o in enumerate(options or [], start=1):
        if not isinstance(o, dict):
            continue
        item = dict(o)
        item["id"] = str(item.get("id") or f"opt{idx}")
        if not item.get("value"):
            item["value"] = item.get("label", "")
        normalized.append(item)
    return normalized


def resolve_option_text(options: list[dict], text: str) -> dict | None:
    """
    Resolve a free-text clarification reply against known option labels/values.

    Web chat sends explicit option ids; text channels still need a tolerant
    matcher so users can type the visible option text.
    """
    reply = (text or "").strip().lower()
    if not reply:
        return None

    normalized = _with_option_ids(options)

    # 1. Exact label / value match
    for option in normalized:
        label = str(option.get("label", "")).strip().lower()
        value = str(option.get("value", option.get("label", ""))).strip().lower()
        if reply in {label, value}:
            return option

    # 2. Substring match (either direction)
    for option in normalized:
        label = str(option.get("label", "")).strip().lower()
        value = str(option.get("value", option.get("label", ""))).strip().lower()
        if reply and (reply in label or reply in value or label in reply or value in reply):
            return option

    # 3. Word-overlap scoring (≥2 overlapping content words)
    reply_words = {w for w in re.split(r"\W+", reply) if len(w) >= 3}
    if not reply_words:
        return None

    best_match = None
    best_score = 0
    for option in normalized:
        candidate_words = {
            w for w in re.split(
                r"\W+",
                f"{option.get('label', '')} {option.get('value', option.get('label', ''))}".lower(),
            )
            if len(w) >= 3
        }
        score = len(reply_words & candidate_words)
        if score > best_score:
            best_score = score
            best_match = option

    return best_match if best_score >= 2 else None


_COMMON_STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "their", "there", "that",
    "this", "those", "these", "show", "find", "give", "list", "what", "which",
    "where", "when", "who", "are", "were", "was", "have", "has", "had", "will",
    "would", "should", "could", "please", "count", "total", "number", "based",
    "each", "per", "group", "grouped", "department", "employee", "employees",
}
_SCHEMA_KEYWORDS = {
    "select", "from", "where", "group", "order", "limit", "distinct",
    "table", "column", "values", "notes", "nullable", "type",
}


def _tokenize_words(text: str) -> list[str]:
    return [
        token for token in re.split(r"[^a-z0-9_]+", (text or "").lower())
        if token
    ]


def _scoped_active_terms(account_id: str, allowed_tables: Optional[set[str]] = None) -> list[dict]:
    import store

    terms = store.list_terms(account_id, active_only=True)
    if allowed_tables is None:
        return terms

    scoped: list[dict] = []
    for term in terms:
        term_tables = {
            part.strip().upper()
            for part in (term.get("tables_involved") or "").split(",")
            if part.strip()
        }
        if not term_tables or (term_tables & allowed_tables):
            scoped.append(term)
    return scoped


def _extract_context_value_candidates(context: str, max_items: int = 18) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        candidate = re.sub(r"\s+", " ", (raw or "")).strip(" `\"'[]()")
        candidate_lower = candidate.lower()
        if not candidate or len(candidate) < 2 or len(candidate) > 40:
            return
        if candidate_lower in seen or candidate_lower in _COMMON_STOPWORDS:
            return
        if candidate_lower in _SCHEMA_KEYWORDS:
            return
        if re.fullmatch(r"[-0-9./_%]+", candidate_lower):
            return
        seen.add(candidate_lower)
        candidates.append(candidate)

    for match in re.finditer(r"[`'\"]([^`'\"\n]{2,40})[`'\"]", context):
        _add(match.group(1))

    for match in re.finditer(r"values?\s+(?:are|include|includes|:)\s*([^\n.]{3,180})", context, re.I):
        fragment = match.group(1)
        for part in re.split(r",|\bor\b|/", fragment):
            _add(part)

    for line in context.splitlines():
        if "|" not in line or "Distinct Values" in line or re.match(r"\|\s*-", line):
            continue
        cells = [cell.strip() for cell in line.split("|") if cell.strip()]
        for cell in cells[-2:]:
            if "," in cell or "'" in cell or '"' in cell or "`" in cell:
                for part in re.split(r",|\bor\b|/", cell):
                    _add(part)

    return candidates[:max_items]


def _build_candidate_phrases(
    account_id: str,
    context: str,
    allowed_tables: Optional[set[str]] = None,
) -> list[str]:
    phrases: list[str] = []
    seen: set[str] = set()

    def _append(value: str) -> None:
        normalized = re.sub(r"\s+", " ", (value or "")).strip()
        key = normalized.lower()
        if not normalized or key in seen:
            return
        seen.add(key)
        phrases.append(normalized)

    for term in _scoped_active_terms(account_id, allowed_tables):
        _append(term.get("term", ""))
        for alias in (term.get("aliases") or "").split(","):
            _append(alias.strip())

    for candidate in _extract_context_value_candidates(context):
        _append(candidate)

    for candidate in _schema_distinct_value_candidates(account_id, allowed_tables):
        _append(candidate)

    return phrases


def _schema_dir_for_account(account_id: str) -> str:
    import store

    try:
        client = store.get_client(account_id) or {}
    except Exception:
        return ""
    raw_state = client.get("state_data") or "{}"
    if isinstance(raw_state, str):
        try:
            state_data = json.loads(raw_state)
        except Exception:
            state_data = {}
    elif isinstance(raw_state, dict):
        state_data = raw_state
    else:
        state_data = {}
    return str(state_data.get("schema_dir") or "").strip()


@lru_cache(maxsize=64)
def _load_schema_distinct_value_candidates(
    schema_dir: str,
    allowed_tables_key: tuple[str, ...],
) -> tuple[str, ...]:
    if not schema_dir:
        return ()

    schema_path = Path(schema_dir)
    if not schema_path.exists():
        return ()

    allowed_tables = {table.upper() for table in allowed_tables_key}
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        candidate = re.sub(r"\s+", " ", (raw or "")).strip(" `\"'[]()")
        candidate_lower = candidate.lower()
        if not candidate or len(candidate) < 2 or len(candidate) > 40:
            return
        if candidate_lower in seen or candidate_lower in _COMMON_STOPWORDS:
            return
        if candidate_lower in _SCHEMA_KEYWORDS:
            return
        if candidate_lower in {"-", "n/a", "none"}:
            return
        seen.add(candidate_lower)
        candidates.append(candidate)

    for schema_file in sorted(schema_path.glob("*.md")):
        if schema_file.name.startswith("_"):
            continue
        if allowed_tables and schema_file.stem.upper() not in allowed_tables:
            continue
        try:
            schema_text = schema_file.read_text(encoding="utf-8")
        except Exception:
            continue
        for line in schema_text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("| `"):
                continue
            cells = [cell.strip() for cell in line.split("|")[1:-1]]
            if not cells:
                continue
            distinct_values = cells[-1]
            if not distinct_values:
                continue
            for part in re.split(r",|\bor\b|/", distinct_values):
                _add(part)

    return tuple(candidates[:160])


def _schema_distinct_value_candidates(
    account_id: str,
    allowed_tables: Optional[set[str]] = None,
) -> list[str]:
    schema_dir = _schema_dir_for_account(account_id)
    allowed_key = tuple(sorted((allowed_tables or set())))
    return list(_load_schema_distinct_value_candidates(schema_dir, allowed_key))


def _find_exact_value_hits(question: str, candidate_phrases: list[str], max_items: int = 6) -> list[str]:
    question_text = (question or "").lower()
    question_tokens = set(_tokenize_words(question))
    hits: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        candidate = re.sub(r"\s+", " ", (raw or "")).strip()
        key = candidate.lower()
        if not candidate or key in seen:
            return
        seen.add(key)
        hits.append(candidate)

    for phrase in candidate_phrases:
        candidate = re.sub(r"\s+", " ", (phrase or "")).strip().lower()
        if len(candidate) >= 2 and re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", question_text):
            _add(phrase)
        if len(hits) >= max_items:
            return hits

    for phrase in candidate_phrases:
        for token in _tokenize_words(phrase):
            if len(token) >= 3 and token in question_tokens:
                _add(token)
            if len(hits) >= max_items:
                return hits

    return hits


def _find_typo_resolutions(question: str, candidate_phrases: list[str], max_items: int = 3) -> list[dict]:
    candidate_tokens: set[str] = set()
    for phrase in candidate_phrases:
        for token in _tokenize_words(phrase):
            if len(token) >= 3 and token not in _SCHEMA_KEYWORDS:
                candidate_tokens.add(token)

    resolutions: list[dict] = []
    for token in _tokenize_words(question):
        if len(token) < 4 or token in _COMMON_STOPWORDS or token in candidate_tokens:
            continue
        viable = [
            candidate for candidate in candidate_tokens
            if candidate[:1] == token[:1] and abs(len(candidate) - len(token)) <= 2
        ]
        best_candidate = None
        best_score = 0.0
        runner_up = 0.0
        for candidate in viable:
            score = SequenceMatcher(None, token, candidate).ratio()
            if score > best_score:
                runner_up = best_score
                best_score = score
                best_candidate = candidate
            elif score > runner_up:
                runner_up = score
        threshold = 0.74 if len(token) <= 4 else 0.82
        if best_candidate and best_score >= threshold and (best_score - runner_up) >= 0.03:
            resolutions.append({
                "token": token,
                "candidate": best_candidate,
                "score": round(best_score, 3),
            })
        if len(resolutions) >= max_items:
            break

    return resolutions


def _build_schema_grounded_evidence(
    account_id: str,
    question: str,
    context: str,
    allowed_tables: Optional[set[str]] = None,
) -> dict:
    import store

    matched_terms = store.match_terms_in_question(account_id, question, allowed_tables)
    candidate_phrases = _build_candidate_phrases(account_id, context, allowed_tables)
    typo_resolutions = _find_typo_resolutions(question, candidate_phrases)
    return {
        "intent": analyze_query_intent(question),
        "intent_summary": summarize_query_intent(question),
        "matched_terms": [term.get("term", "") for term in matched_terms[:6] if term.get("term")],
        "context_values": _extract_context_value_candidates(context),
        "exact_value_hits": _find_exact_value_hits(question, candidate_phrases),
        "typo_resolutions": typo_resolutions,
    }


def build_schema_grounded_clarification_hint(
    account_id: str,
    question: str,
    context: str,
    allowed_tables: Optional[set[str]] = None,
) -> str:
    """
    Build a compact schema/business evidence hint for SQL and clarification prompts.
    """
    evidence = _build_schema_grounded_evidence(
        account_id,
        question,
        context,
        allowed_tables=allowed_tables,
    )
    lines = [
        "SCHEMA-GROUNDED INTERPRETATION EVIDENCE:",
        "- Use only the business terms and schema-backed categorical values below when deciding whether clarification is needed.",
        "- Exact schema-backed categorical values are authoritative. Preserve them exactly even if they look misspelled.",
        "- If there is one high-confidence typo or value resolution from the evidence, prefer that resolution over asking a clarification question.",
        "- Missing JOIN paths or unsupported metrics are not ambiguity. They should not trigger a clarification question.",
    ]
    if evidence["intent_summary"]:
        lines.append(f"- Query intent: {evidence['intent_summary']}.")
    if evidence["matched_terms"]:
        lines.append(
            "- Matched business terms: " + ", ".join(evidence["matched_terms"][:6]) + "."
        )
    if evidence["context_values"]:
        lines.append(
            "- Schema-backed categorical values seen in relevant KB context: "
            + ", ".join(evidence["context_values"][:8]) + "."
        )
    if evidence["exact_value_hits"]:
        lines.append(
            "- Exact schema-backed values already present in the user question: "
            + ", ".join(evidence["exact_value_hits"][:6]) + "."
        )
    if evidence["typo_resolutions"]:
        typo_parts = [
            f"'{item['token']}' -> '{item['candidate']}'"
            for item in evidence["typo_resolutions"]
        ]
        lines.append(
            "- High-confidence typo/value resolutions from schema or business context: "
            + ", ".join(typo_parts) + "."
        )
    return "\n".join(lines) + "\n"


def _scored_terms_for_ambiguity_menu(
    account_id: str,
    question: str,
    allowed_tables: Optional[set[str]] = None,
    evidence: Optional[dict] = None,
) -> list[dict]:
    terms = _scoped_active_terms(account_id, allowed_tables)
    q_words = {
        word for word in _tokenize_words(question)
        if len(word) >= 3 and word not in _COMMON_STOPWORDS
    }
    evidence = evidence or {}
    matched_terms = {term.lower() for term in evidence.get("matched_terms") or []}
    for item in evidence.get("typo_resolutions") or []:
        candidate = str(item.get("candidate") or "").lower().strip()
        if candidate:
            q_words.add(candidate)

    scored: list[tuple[int, dict]] = []
    for term in terms:
        label = f"{term.get('term', '')} {term.get('aliases', '')} {term.get('definition', '')}".lower()
        label_words = {
            word for word in _tokenize_words(label)
            if len(word) >= 3 and word not in _SCHEMA_KEYWORDS
        }
        score = len(q_words & label_words)
        if term.get("term", "").lower() in matched_terms:
            score += 3
        scored.append((score, term))

    positives = [item for item in scored if item[0] > 0]
    chosen = positives if len(positives) >= 2 else scored
    chosen.sort(key=lambda item: item[0], reverse=True)
    return [term for _, term in chosen[:20]]


# ══════════════════════════════════════════════════════════════════════════════
# Glossary-first ambiguity check (deterministic when possible)
# ══════════════════════════════════════════════════════════════════════════════

async def check_ambiguity_glossary_first(
    account_id: str,
    question: str,
    context: str,
    provider: str,
    model: str,
    api_key: str,
    extra_kwargs: dict,
    allowed_tables: Optional[set[str]] = None,
) -> tuple[bool, str, dict]:
    """
    Business-aware ambiguity check.

    Tries the glossary first (zero LLM cost, fully deterministic).
    Falls back to a CONSTRAINED-MENU LLM check only when the glossary
    can't resolve the ambiguity.

    Returns:
        (is_ambiguous, clarifying_question, meta)

        meta contains:
          - source: 'glossary' | 'glossary_multi' | 'llm_menu' | 'llm' | 'none'
          - term_id: int if glossary-resolved
          - options: list[{id, label, value, expression, _term_id, valid}]
                     populated whenever is_ambiguous=True AND the glossary
                     has enough coverage (after Fix #1)
    """
    import store

    # Step 1: Term with requires_clarification=True and predefined options.
    registry_match = find_registry_clarification(
        account_id,
        question,
        allowed_tables=allowed_tables,
    )
    if registry_match:
        ambiguous_term = registry_match["term"]
        opts = registry_match["options"]
        option_labels = [o.get("label", "") for o in opts if o.get("label")]
        if len(option_labels) >= 2:
            clarifying_q = _format_glossary_clarification(
                term=ambiguous_term["term"],
                options=option_labels[:3],
            )
            log.info(
                "Glossary clarification for '%s' in q='%s'",
                ambiguous_term["term"], question[:50],
            )
            return True, clarifying_q, {
                "source": "glossary",
                "term_id": ambiguous_term["id"],
                "term": ambiguous_term["term"],
                "question": clarifying_q,
                "options": opts,
                "generated_options_ignored": False,
            }

    # Step 2: Question overlaps multiple distinct metrics.
    matches = store.match_terms_in_question(account_id, question, allowed_tables)
    metric_matches = [m for m in matches if m.get("kind") == "metric"]
    if len(metric_matches) >= 2:
        # If the user is explicitly asking to compare/breakdown multiple metrics,
        # they want ALL of them — don't ask which one they meant.
        _intent = analyze_query_intent(question)
        if not _intent.get("wants_comparison") and not _intent.get("wants_conditional_split"):
            implicit_opts = [
                {
                    "label": _clean_chip_label(m.get("definition", ""), m["term"]),
                    "value": m["term"],
                    "expression": m.get("canonical_expression", ""),
                    "definition": m.get("definition", ""),
                    "_term_id": m["id"],
                    "valid": True,
                }
                for m in metric_matches[:3]
            ]
            implicit_opts = validated_options(implicit_opts)
            if len(implicit_opts) >= 2:
                clarifying_q = (
                    "I see a few ways to interpret this. Which did you mean:\n"
                    + "\n".join(f"  • {o['label']}" for o in implicit_opts)
                )
                return True, clarifying_q, {
                    "source": "glossary_multi",
                    "question": clarifying_q,
                    "options": implicit_opts,
                    "generated_options_ignored": False,
                }

    # Step 3: Schema-grounded evidence for ambiguity and typo resolution.
    glossary_hint = _build_glossary_hint(account_id, allowed_tables)
    schema_hint = build_schema_grounded_clarification_hint(
        account_id,
        question,
        context,
        allowed_tables=allowed_tables,
    )
    evidence = _build_schema_grounded_evidence(
        account_id,
        question,
        context,
        allowed_tables=allowed_tables,
    )
    enriched_parts = [part for part in (schema_hint, glossary_hint, context) if part]
    enriched_context = "\n\n".join(enriched_parts)

    # If the question already contains an exact schema-backed value and the
    # phrasing looks like a categorical filter, preserve that literal value
    # instead of escalating to another clarification turn.
    if (
        evidence.get("exact_value_hits")
        and evidence.get("intent", {}).get("wants_status_filter")
        and len(metric_matches) <= 1
    ):
        log.info(
            "Schema-backed exact value resolved without clarification for q='%s' via %s",
            question[:80],
            evidence["exact_value_hits"][:3],
        )
        return False, "", {
            "source": "none",
            "question": "",
            "options": [],
            "generated_options_ignored": False,
        }

    # If we have enough schema/business-backed candidates, force the LLM to
    # choose only from that constrained menu. Otherwise fall back to the plain
    # CLEAR/AMBIGUOUS classifier, but still pass the schema-grounded evidence.
    menu_terms = _scored_terms_for_ambiguity_menu(
        account_id,
        question,
        allowed_tables=allowed_tables,
        evidence=evidence,
    )
    if len(menu_terms) >= 2:
        # If the question uses comparison framing and ≥2 of the menu terms are
        # explicitly named in the question verbatim, the user wants all of them —
        # multiple matching entries is not ambiguity, it is the feature working.
        _cmp_intent = analyze_query_intent(question)
        if _cmp_intent.get("wants_comparison") or _cmp_intent.get("wants_conditional_split"):
            _q_lower = question.lower()
            _explicit = sum(
                1 for _t in menu_terms
                if re.search(
                    r"\b" + re.escape((_t.get("term") or "").lower()) + r"\b",
                    _q_lower,
                )
            )
            if _explicit >= 2:
                log.info(
                    "Skipping LLM ambiguity check: comparison intent + %d explicitly-named terms for q='%s'",
                    _explicit,
                    question[:80],
                )
                return False, "", {
                    "source": "none",
                    "question": "",
                    "options": [],
                    "generated_options_ignored": False,
                }
        is_amb, q, options = await _llm_ambiguity_check_constrained(
            account_id,
            question,
            enriched_context,
            provider,
            model,
            api_key,
            extra_kwargs,
            allowed_tables=allowed_tables,
            candidate_terms=menu_terms,
            evidence_hint=schema_hint,
        )
        return is_amb, q, {
            "source": "llm_menu" if (is_amb and options) else ("llm" if is_amb else "none"),
            "question": q,
            "options": options,
            "generated_options_ignored": False,
        }

    is_amb, q, llm_opts = await _llm_ambiguity_check(
        question,
        enriched_context,
        provider,
        model,
        api_key,
        extra_kwargs,
        evidence_hint=schema_hint,
    )
    return is_amb, q, {
        "source": "llm" if is_amb else "none",
        "question": q,
        "options": [],
        "generated_options_ignored": bool(llm_opts),
    }


def _format_glossary_clarification(term: str, options: list[str]) -> str:
    """Render a clarification question from predefined options."""
    if len(options) == 2:
        return (
            f"When you say *{term}*, do you mean:\n"
            f"  • {options[0]}\n"
            f"  • {options[1]}?"
        )
    lines = [f"When you say *{term}*, which do you mean?"]
    for opt in options:
        lines.append(f"  • {opt}")
    return "\n".join(lines)


def _build_glossary_hint(
    account_id: str,
    allowed_tables: Optional[set[str]] = None,
) -> str:
    """Build a compact glossary hint to enrich LLM prompts."""
    import store

    terms = store.list_terms(account_id, active_only=True)
    if not terms:
        return ""

    if allowed_tables is not None:
        scoped = []
        for t in terms:
            tbls = {
                s.strip().upper()
                for s in (t.get("tables_involved") or "").split(",")
                if s.strip()
            }
            if not tbls or (tbls & allowed_tables):
                scoped.append(t)
        terms = scoped

    if not terms:
        return ""

    lines = [
        "KNOWN BUSINESS TERMS (use these exact names when asking clarifying "
        "questions — they are the only terms this business defines):",
    ]
    by_kind: dict[str, list[str]] = {}
    for t in terms[:30]:
        kind = t.get("kind", "metric")
        by_kind.setdefault(kind, []).append(t["term"])
    for kind in ("metric", "dimension", "filter", "entity"):
        if kind in by_kind:
            lines.append(f"  {kind}s: {', '.join(by_kind[kind])}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# LLM ambiguity check — constrained menu (Fix #1) + tolerant JSON (Fix #5)
# ══════════════════════════════════════════════════════════════════════════════

async def _llm_ambiguity_check_constrained(
    account_id: str,
    question: str,
    context: str,
    provider: str,
    model: str,
    api_key: str,
    extra_kwargs: dict,
    allowed_tables: Optional[set[str]] = None,
    candidate_terms: Optional[list[dict]] = None,
    evidence_hint: str = "",
) -> tuple[bool, str, list[dict]]:
    """
    Constrained-menu LLM ambiguity check.

    The model detects ambiguity and picks 2–3 option IDs from a menu
    we build from the real glossary. It cannot invent options; every
    option returned carries a real term_id and a real SQL expression
    so the retry prompt can lock in the correct interpretation.

    Falls back to the plain CLEAR/AMBIGUOUS classifier when the
    glossary is too sparse to build a menu.
    """
    import store
    from core.llm import llm_complete
    from core.llm_audit import llm_audit_component

    terms = candidate_terms or _scored_terms_for_ambiguity_menu(
        account_id,
        question,
        allowed_tables=allowed_tables,
    )

    # Menu too sparse → fall back to the plain classifier with no options.
    if len(terms) < 2:
        is_amb, q, _opts = await _llm_ambiguity_check(
            question,
            context,
            provider,
            model,
            api_key,
            extra_kwargs,
            evidence_hint=evidence_hint,
        )
        return is_amb, q, []

    id_to_term: dict[str, dict] = {}
    menu_lines: list[str] = []
    for t in terms:
        tid = f"t{t['id']}"
        id_to_term[tid] = t
        kind = t.get("kind", "metric")
        defn = (t.get("definition") or "")[:80]
        menu_lines.append(f"  {tid} ({kind}): {t['term']} — {defn}")

    menu = "\n".join(menu_lines)

    system = (
        "You detect ambiguous business questions and pick from a FIXED MENU.\n\n"
        "Output ONLY valid JSON, one of:\n"
        '  {"status":"CLEAR"}\n'
        '  {"status":"AMBIGUOUS","question":"<=25 words>","option_ids":["id1","id2"]}\n\n'
        "Rules:\n"
        "1. Return AMBIGUOUS only when two or more menu entries are plausible\n"
        "   interpretations of the user's question.\n"
        "2. option_ids MUST come from the MENU below. Do NOT invent IDs.\n"
        "3. Return at most 3 option_ids, at least 2 to be ambiguous.\n"
        "4. If the menu does not cover the ambiguity, return CLEAR.\n"
        "5. Never mention SQL, tables, or columns in the question text.\n"
        "6. Exact schema-backed categorical values in the question are authoritative, even if they look oddly spelled -> CLEAR.\n"
        "6. Typos that map to known column values are NOT ambiguous → CLEAR.\n"
        "7. Missing JOIN paths are NOT ambiguous → CLEAR. The SQL engine handles joins.\n"
        "8. If the question uses comparison framing (compare, vs, versus, difference, contrast)\n"
        "   AND explicitly names two or more distinct menu entries by their label, return CLEAR —\n"
        "   the user wants all of them together, they are NOT asking which one to use.\n"
    )

    user_msg = (
        (f"Schema-grounded evidence:\n{evidence_hint[:1200]}\n\n" if evidence_hint else "")
        + f"MENU:\n{menu}\n\n"
        f"Business context:\n{context[:1800]}\n\n"
        f"User question: {question}"
    )

    try:
        with llm_audit_component("clarification_menu"):
            raw, _, _ = await llm_complete(
                system, user_msg, provider, model, api_key,
                max_tokens=220, temperature=0.0, **extra_kwargs,
            )
    except Exception as e:
        log.warning("Constrained ambiguity check failed: %s", e)
        return False, "", []

    parsed = _parse_ambiguity_json(raw)
    if not parsed:
        log.warning("Could not parse ambiguity JSON: %s", (raw or "")[:120])
        return False, "", []

    status = str(parsed.get("status", "")).upper()
    if status != "AMBIGUOUS":
        return False, "", []

    q = str(parsed.get("question", "")).strip() or "Which did you mean?"
    chosen_ids = [
        i for i in (parsed.get("option_ids") or [])
        if isinstance(i, str) and i in id_to_term
    ]
    if len(chosen_ids) < 2:
        return False, "", []

    options: list[dict] = []
    for idx, tid in enumerate(chosen_ids[:3], start=1):
        t = id_to_term[tid]
        options.append({
            "id": f"opt{idx}",
            "label": _clean_chip_label(t.get("definition", ""), t["term"]),
            "value": t["term"],
            "expression": t.get("canonical_expression", ""),
            "_term_id": t["id"],
            "valid": True,
        })

    log.info(
        "LLM-menu clarification chose %d options for q='%s'",
        len(options), question[:60],
    )
    return True, q, options


async def _llm_ambiguity_check(
    question: str,
    context: str,
    provider: str,
    model: str,
    api_key: str,
    extra_kwargs: dict,
    evidence_hint: str = "",
) -> tuple[bool, str, list[dict]]:
    """
    Plain AMBIGUOUS/CLEAR classifier. Used as a fallback when the
    glossary is too sparse for the constrained-menu path.
    """
    from core.llm import llm_complete
    from core.llm_audit import llm_audit_component

    system = (
        "You detect genuinely ambiguous business analytics questions.\n\n"
        "Return ONLY valid JSON using exactly one of these shapes:\n"
        '{"status": "CLEAR"}\n'
        "or\n"
        '{"status": "AMBIGUOUS", "question": "..."}\n\n'
        "Rules:\n"
        "1. Ask for clarification ONLY when two distinct interpretations are genuinely "
        "supported by the provided context.\n"
        "2. Use only business vocabulary already present in the context.\n"
        "3. Do NOT generate candidate options, labels, or metric lists.\n"
        "4. Keep the question under 25 words.\n"
        "5. Never mention SQL, tables, columns, or technical identifiers.\n"
        "6. When in doubt, return CLEAR.\n"
        "7. Exact schema-backed categorical values in the question are authoritative, even if they look oddly spelled -> CLEAR.\n"
        "7. Typos that map to known column values are NOT ambiguous → CLEAR.\n"
        "8. Missing JOIN paths are NOT ambiguous → CLEAR.\n"
    )

    user_msg = (
        (f"Schema-grounded evidence:\n{evidence_hint[:1200]}\n\n" if evidence_hint else "")
        + f"Business context (tables, columns, KB, known business terms, values):\n"
        f"{context[:2500]}\n\n"
        f"User question: {question}\n"
    )

    try:
        with llm_audit_component("clarification_fallback"):
            raw, _, _ = await llm_complete(
                system, user_msg, provider, model, api_key,
                max_tokens=220, temperature=0.0, **extra_kwargs,
            )
    except Exception as e:
        log.warning("Plain ambiguity check failed: %s", e)
        return False, "", []

    parsed = _parse_ambiguity_json(raw)
    if parsed and str(parsed.get("status", "")).upper() == "AMBIGUOUS":
        q = str(parsed.get("question") or "I need a bit more context to answer that.").strip()
        if q:
            return True, q, []

    # Legacy text-shape fallback: "AMBIGUOUS: <question>"
    if (raw or "").strip().upper().startswith("AMBIGUOUS:"):
        q = raw.split(":", 1)[1].strip() or "I need a bit more context to answer that."
        return True, q, []

    return False, "", []


def _parse_ambiguity_json(raw: str) -> Optional[dict]:
    """
    Tolerant JSON extractor (Fix #5).

    Handles:
      • Pure JSON objects
      • ```json … ``` fenced blocks
      • Preamble text before the JSON ("Here is the result: {...}")
      • Trailing content after the JSON

    Returns the parsed dict, or None on any failure.
    """
    if not raw:
        return None
    s = raw.strip()

    # Strip triple-backtick fences if present.
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        s = s.rsplit("```", 1)[0].strip()

    # Locate the first '{' and balance braces from there.
    start = s.find("{")
    if start < 0:
        return None

    depth = 0
    end = -1
    in_string = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end < 0:
        return None

    try:
        parsed = json.loads(s[start:end])
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Pending clarification state (SQLite) — with recent-expiry tracking (Fix #7)
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_table(conn) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pending_clarification (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id    TEXT    NOT NULL,
            zoom_user_id  TEXT    NOT NULL,
            original_q    TEXT    NOT NULL,
            context       TEXT    NOT NULL DEFAULT '',
            expires_at    TEXT    NOT NULL,
            created_at    TEXT    DEFAULT (datetime('now')),
            UNIQUE(account_id, zoom_user_id)
        )
    """)
    try:
        existing = [
            row[1] for row in
            conn.execute("PRAGMA table_info(pending_clarification)").fetchall()
        ]
        if "clarification_meta" not in existing:
            conn.execute(
                "ALTER TABLE pending_clarification ADD COLUMN "
                "clarification_meta TEXT DEFAULT ''"
            )
    except Exception:
        pass


def save_pending(
    account_id: str,
    zoom_user_id: str,
    original_question: str,
    context: str = "",
    clarification_meta: Optional[dict] = None,
) -> None:
    """Save a pending clarification."""
    from store.db import get_db
    expires = (
        datetime.now(timezone.utc) + timedelta(minutes=_EXPIRE_MINUTES)
    ).strftime("%Y-%m-%d %H:%M:%S")

    meta_json = json.dumps(clarification_meta) if clarification_meta else ""

    with get_db() as conn:
        _ensure_table(conn)
        conn.execute(
            """
            INSERT INTO pending_clarification
                (account_id, zoom_user_id, original_q, context, expires_at,
                 clarification_meta)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(account_id, zoom_user_id) DO UPDATE SET
                original_q         = excluded.original_q,
                context            = excluded.context,
                expires_at         = excluded.expires_at,
                clarification_meta = excluded.clarification_meta,
                created_at         = datetime('now')
            """,
            (account_id, zoom_user_id, original_question,
             context[:3000], expires, meta_json),
        )
    log.info(
        "Saved pending clarification for user %s in %s (source=%s)",
        zoom_user_id, account_id,
        (clarification_meta or {}).get("source", "llm"),
    )


def get_pending(account_id: str, zoom_user_id: str) -> Optional[dict]:
    """
    Return pending clarification for this user if not expired.

    On expiry, marks the slot as "recently expired" (Fix #7) so the
    dispatcher can surface a friendly message instead of silently
    reprocessing the user's reply as a fresh query.
    """
    from store.db import get_db
    with get_db() as conn:
        _ensure_table(conn)
        row = conn.execute(
            """
            SELECT * FROM pending_clarification
            WHERE account_id = ? AND zoom_user_id = ?
            """,
            (account_id, zoom_user_id),
        ).fetchone()

        if not row:
            return None

        row = dict(row)
        expiry = datetime.strptime(
            row["expires_at"], "%Y-%m-%d %H:%M:%S"
        ).replace(tzinfo=timezone.utc)

        if datetime.now(timezone.utc) > expiry:
            conn.execute(
                """
                DELETE FROM pending_clarification
                WHERE account_id = ? AND zoom_user_id = ?
                """,
                (account_id, zoom_user_id),
            )
            mark_recently_expired(account_id, zoom_user_id)
            return None

        try:
            row["clarification_meta"] = (
                json.loads(row.get("clarification_meta") or "")
                if row.get("clarification_meta") else {}
            )
        except Exception:
            row["clarification_meta"] = {}

        return row


def clear_pending(account_id: str, zoom_user_id: str) -> None:
    """Remove pending clarification after it has been used."""
    from store.db import get_db
    with get_db() as conn:
        _ensure_table(conn)
        conn.execute(
            """
            DELETE FROM pending_clarification
            WHERE account_id = ? AND zoom_user_id = ?
            """,
            (account_id, zoom_user_id),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Combine original question with the user's clarification reply
# ══════════════════════════════════════════════════════════════════════════════

def combine_with_clarification(
    original_q: str,
    clarification: str,
    meta: Optional[dict] = None,
    selected_option_id: Optional[str] = None,   # Fix #2
) -> tuple[str, str]:
    """
    Merge the original question with the user's clarification reply.

    Parameters
    ----------
    original_q : str
        The question the user originally asked.
    clarification : str
        The user's free-text reply OR the chosen option's value/label.
    meta : dict, optional
        The clarification metadata saved at save_pending() time.
    selected_option_id : str, optional  (Fix #2)
        When dispatch has already resolved the user's reply to a specific
        option via resolve_option_text(), the option's `id` is forwarded
        here. combine_with_clarification uses it verbatim — skipping the
        substring re-matching that previously drifted away from dispatch's
        choice on ambiguous labels.

    Returns
    -------
    (combined_question, term_injection)
        combined_question — feed to handle_query on retry
        term_injection    — append to the SQL system prompt; empty if no
                            business term was resolved
    """
    import store

    clarification_text = (clarification or "").strip()
    combined = (
        f"{original_q}\n\n"
        f"Clarification for the same request: {clarification_text}.\n"
        f"Use this clarification to interpret the original request; "
        f"do not treat it as a separate question."
    )
    injection = ""

    if not meta:
        return combined.strip(), injection

    source = meta.get("source", "")

    # Glossary-first path — term_id is already known.
    if source == "glossary" and meta.get("term_id"):
        term = store.get_term(meta["term_id"])
        if term:
            injection = store.build_term_injection_from_choice(term, clarification)
        return combined.strip(), injection

    # For glossary_multi / llm_menu / llm — we have a list of options.
    opts = meta.get("options") or []
    if not opts:
        return combined.strip(), injection

    chosen = _pick_option(opts, clarification_text, selected_option_id)
    if not chosen:
        return combined.strip(), injection

    chosen_expr  = chosen.get("expression") or ""
    chosen_label = chosen.get("label") or chosen.get("value") or ""

    if chosen_expr:
        injection = (
            f"RESOLVED BUSINESS TERM — the user has clarified they want: "
            f"{chosen_label}.\n"
            f"Use this EXACT SQL expression for this concept: `{chosen_expr}`\n"
        )
    elif chosen.get("_term_id"):
        # Fall back to pulling the term from the glossary if expression
        # wasn't embedded in the option dict.
        try:
            term = store.get_term(int(chosen["_term_id"]))
            if term and term.get("canonical_expression"):
                injection = (
                    f"RESOLVED BUSINESS TERM — the user has clarified they want: "
                    f"{chosen_label}.\n"
                    f"Use this EXACT SQL expression for this concept: "
                    f"`{term['canonical_expression']}`\n"
                )
        except Exception:
            pass

    return combined.strip(), injection


def _pick_option(
    options: list[dict],
    clarification_text: str,
    selected_option_id: Optional[str],
) -> Optional[dict]:
    """
    Resolve which option the user chose.

    Precedence (Fix #2):
      1. Explicit selected_option_id (dispatch / WebSocket has already matched)
      2. Exact label / value match on clarification_text
      3. resolve_option_text substring + word-overlap match
    """
    normalized = _with_option_ids(options)

    if selected_option_id:
        for o in normalized:
            if str(o.get("id")) == str(selected_option_id):
                return o
        # Explicit ID didn't match any option. Do NOT silently fall back
        # to text matching — that's how dispatch/combine drift apart.
        log.warning(
            "selected_option_id=%s not in options %s",
            selected_option_id, [o.get("id") for o in normalized],
        )
        return None

    txt = clarification_text.lower().strip()
    if not txt:
        return None

    for o in normalized:
        label = str(o.get("label", "")).strip().lower()
        value = str(o.get("value", label)).strip().lower()
        if txt in {label, value}:
            return o

    return resolve_option_text(normalized, clarification_text)
