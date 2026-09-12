"""
core/failure_messages.py

Translate hard failures (validator rejections, database errors) into the same
business-readable {headline / most_likely_reason / suggested_next_step /
technical_notes} shape core/answer_rca.build_business_rca uses for zero rows —
so every failure the user sees leads with plain language and a concrete next
step, with the technical detail kept but demoted.

Raw error text is NEVER lost: callers keep logging the original string to
query_log / answer_trace; this module only shapes what reaches the chat.
"""

from __future__ import annotations

import re
from typing import Any


def _t(msg_id: str, **kw) -> str:
    """Resolve a catalogue id in the reader's language.

    Deferred import: core.i18n is large and this module is imported from the
    validator's error paths. The language comes from the request's ContextVar,
    which _handle_query_impl activates around the whole turn.
    """
    from core.i18n import t

    return t(msg_id, **kw)

_MAX_TECH_CHARS = 300


def _clip(text: Any, limit: int = _MAX_TECH_CHARS) -> str:
    """Trim to `limit` without cutting a word in half, and say it was trimmed.

    A bare slice cut Azure's paused-database message at "open the Compute and
    Storage tab from the database menu on the Azur" -- the sentence that tells
    an administrator exactly how to fix the outage, ending mid-word with no
    sign anything was missing. A truncation that hides the fact that it
    truncated is worse than a longer message.
    """
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    # Back up to the last word boundary, unless that would throw away most of
    # the message (a single very long token).
    cut = value[:limit]
    space = cut.rfind(" ")
    if space > limit * 0.6:
        cut = cut[:space]
    return cut.rstrip(" ,;:.") + "…"


# ── DB error sanitizer ────────────────────────────────────────────────────────

# Ordered: first match wins. Matchers run against the CLEANED error text
# (driver prefixes stripped), case-insensitive.
#
# The second element is a catalogue stem, not prose: fail.db.<stem>.reason and
# .next_step. These sentences are the body of the card a reader actually reads,
# and holding them here as English literals is what made a French failure card
# French only down to its headings.
_DB_ERROR_MAP: list[tuple[str, str]] = [
    ('login timeout|HYT00', "login_timeout"),
    ('login failed|\\b18456\\b|\\b28000\\b', "login_failed"),
    ('invalid object name|\\b208\\b.*object|table or view does not exist|\\bORA-00942\\b', "missing_table"),
    ('invalid column name|\\bORA-00904\\b', "missing_column"),
    ('multi-part identifier .* could not be bound|\\b4104\\b', "unbound_identifier"),
    ('is invalid in the select list because it is not contained in either an aggregate|\\b8120\\b|not a GROUP BY expression|\\bORA-00979\\b', "group_by_shape"),
    ('permission was denied|\\b229\\b.*denied|\\b297\\b|insufficient privileges|\\bORA-01031\\b', "permission_denied"),
    ('incorrect syntax|syntax error|\\bORA-00933\\b|\\bORA-00936\\b', "syntax"),
    ('conversion failed|error converting data type|\\b241\\b|\\b245\\b|\\b8114\\b|\\bORA-01722\\b|\\bORA-01861\\b', "conversion"),
    ('divide by zero|\\b8134\\b|\\bORA-01476\\b', "divide_by_zero"),
    ('timed out|timeout expired|query timeout', "timeout"),
    ('communication link|\\b08S01\\b|connection (?:was )?(?:closed|reset|broken)|TCP Provider', "link_lost"),
    ('deadlock|\\b1205\\b', "deadlock"),
    ('free (?:amount )?(?:allowance|limit)|monthly free amount|database is paused|is paused for the remainder|auto-paused|resource limit (?:has been )?reached|quota (?:has been )?exceeded|service objective .* exhausted|\\b40613\\b', "service_limit"),
]


_QUERY_TIMEOUT_RE = re.compile(
    r"timeout expired|query timeout|\bHYT00\b|timed out",
    re.IGNORECASE,
)


def is_query_timeout(raw: str) -> bool:
    """Whether a database error is a statement timeout rather than a defect.

    This distinction decides whether the pipeline should repair the SQL. A
    timeout means the query was VALID and the database was too slow — usually a
    missing index on the filtered key. Rewriting it cannot make the database
    faster: the retry burns another full timeout window, costs a generation
    call, and can replace a clear "the query timed out" diagnostic with a
    misleading validator error from the regenerated SQL. Observed live: a
    governed revenue query timed out at 120 s, the retry ran for another 120 s
    and then failed validation on the entity-graph join plan, so the user waited
    four minutes to be told the wrong thing.

    Login/connection timeouts are excluded — those are covered by the
    connectivity patterns and are genuinely worth one more attempt.
    """
    text = str(raw or "")
    if not text:
        return False
    if re.search(r"login timeout", text, re.IGNORECASE):
        return False
    return bool(_QUERY_TIMEOUT_RE.search(text))


def build_query_timeout_guidance(
    semantic_plan: dict | None = None,
    *,
    timeout_seconds: int | None = None,
) -> dict[str, str]:
    """Turn a statement timeout into something an administrator can act on.

    "The database did not respond in time" is true but not useful. When the plan
    carries a governed date role we know exactly which column the query filters
    and joins on, so we can name the index that would make it fast.
    """
    policies = [
        policy for policy in ((semantic_plan or {}).get("temporal_policies") or [])
        if isinstance(policy, dict)
    ]
    detail = (
        _t("fail.timeout.detail_seconds", seconds=timeout_seconds)
        if timeout_seconds
        else _t("fail.timeout.detail")
    )
    for policy in policies:
        fact = str(policy.get("fact_table") or policy.get("anchor_table") or "").strip()
        key = str(policy.get("fact_column") or "").strip()
        if fact and key:
            return {
                "reason": _t("fail.timeout.indexed.reason",
                             detail=detail, fact=fact, key=key),
                "next_step": _t("fail.timeout.indexed.next_step",
                                fact=fact, key=key),
            }
    return {
        "reason": detail,
        "next_step": _t("fail.timeout.generic.next_step"),
    }


def _clean_db_error(raw: str) -> str:
    """Strip driver noise: pyodbc tuple wrapping and [Vendor][Driver] prefixes."""
    text = (raw or "").strip()
    # pyodbc renders errors as ('42S02', "[Microsoft]...message. (208) (SQLExecDirectW)")
    m = re.match(r"^\(\s*'[^']*'\s*,\s*[\"'](.*)[\"']\s*\)$", text, re.DOTALL)
    if m:
        text = m.group(1)
    # Drop every leading [ ... ] bracket group (driver/vendor/server chain).
    text = re.sub(r"^(?:\[[^\]]*\]\s*)+", "", text).strip()
    # Drop trailing pyodbc call-site markers like (SQLExecDirectW)
    text = re.sub(r"\s*\(SQL[A-Za-z]+W?\)\s*$", "", text).strip()
    return text


def sanitize_db_error(raw: str) -> dict[str, str]:
    """
    Return {plain_reason, next_step, cleaned} for a raw driver/database error.

    cleaned = the original message minus driver prefixes — still technical,
    kept for the technical-details section. plain_reason/next_step come from
    the ordered matcher table; unknown errors fall back to the first sentence
    of the cleaned text so the user is never shown bracket soup.
    """
    cleaned = _clean_db_error(raw)
    probe = f"{raw or ''} || {cleaned}"
    for pattern, stem in _DB_ERROR_MAP:
        if re.search(pattern, probe, re.IGNORECASE):
            return {
                "plain_reason": _t(f"fail.db.{stem}.reason"),
                "next_step": _t(f"fail.db.{stem}.next_step"),
                "cleaned": cleaned,
            }
    # `cleaned` is the database's own words and stays exactly as it came --
    # untranslated on purpose, because it is what support searches on.
    first_sentence = re.split(r"(?<=[.!?])\s", cleaned, maxsplit=1)[0][:200].strip()
    return {
        "plain_reason": first_sentence or _t("fail.db.unexpected.reason"),
        "next_step": _t("fail.db.unexpected.next_step"),
        "cleaned": cleaned,
    }


_ERR_VALUE_PARENS_RE = re.compile(
    r"((?:duplicate key value|key value|value)\s+is\s*)\([^)]*\)", re.IGNORECASE
)
_ERR_SINGLE_QUOTED_RE = re.compile(r"'([^']*)'")
_ERR_DOUBLE_QUOTED_RE = re.compile(r'"([^"]*)"')
_ERR_LONG_NUMBER_RE = re.compile(r"\b\d{6,}\b")


def _err_is_schema_identifier(s: str) -> bool:
    """
    True when a quoted token inside a DB error is a schema identifier rather
    than a data value.

    Deliberately biased toward masking: a token is only kept when it carries a
    positive structural signal of being an identifier — an underscore, an
    all-uppercase shape, or a dotted qualified name. A bare mixed-case word
    ("Lipitor") is treated as data even though it *could* be a PascalCase
    column, because leaking a value is worse than losing a name the caller's
    column-fix note already supplies.
    """
    s = s.strip()
    if not s or " " in s:
        return False
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", s):
        return False
    return "_" in s or "." in s or s.isupper()


def _mask_err_quoted(match: re.Match, quote: str) -> str:
    inner = match.group(1)
    if _err_is_schema_identifier(inner):
        return match.group(0)
    return f"{quote}[value]{quote}"


def scrub_error_for_llm(raw: str) -> str:
    """
    Prepare a raw database error for inclusion in an LLM repair prompt.

    Database engines routinely echo the offending row value back in the error
    text — "Conversion failed when converting the varchar value 'Lipitor 40mg'
    to data type int", or "The duplicate key value is (Priya Raghunathan)".
    Interpolating that verbatim puts real data into a prompt on a path that is
    otherwise metadata-only, so it is masked here.

    The repair signal survives: driver/vendor prefixes are stripped (they carry
    no repair information) while the error class, the data types involved and
    schema identifiers are preserved.

    This is stricter than core.llm_audit.sanitize_llm_text, which is tuned for
    human-readable audit previews: that helper masks any quoted token of ten or
    more characters (so it would drop 'NET_REVENUE') and only catches unquoted
    runs of twenty or more (so it would keep 'Priya Raghunathan'). Both are the
    wrong trade for a prompt.

    Applied for every tenant, not only regulated ones — the value is never
    needed to fix the SQL, and one unconditional path avoids the mode-predicate
    drift that gates elsewhere in this codebase suffer from.
    """
    if not raw:
        return ""
    text = _clean_db_error(raw)
    # Parenthesised payloads of known value-bearing templates, before quoting
    # rules — these values are usually unquoted.
    text = _ERR_VALUE_PARENS_RE.sub(r"\1([value])", text)
    text = _ERR_SINGLE_QUOTED_RE.sub(lambda m: _mask_err_quoted(m, "'"), text)
    text = _ERR_DOUBLE_QUOTED_RE.sub(lambda m: _mask_err_quoted(m, '"'), text)
    text = _ERR_LONG_NUMBER_RE.sub("[number]", text)
    text = text.strip()
    if len(text) > 400:
        text = text[:397].rstrip() + "..."
    return text


# ── "Did you mean" suggestions ───────────────────────────────────────────────

_SUGGEST_STOPWORDS = {
    "what", "the", "for", "and", "per", "each", "show", "give", "list",
    "total", "sum", "avg", "average", "count", "number", "how", "many",
    "much", "previous", "last", "current", "this", "year", "month", "week",
    "quarter", "date", "top", "bottom", "highest", "lowest", "with", "from",
}


def _suggest_stem(token: str) -> str:
    """Naive stem so 'ordered'/'orders' meet 'order' — good enough for
    overlap scoring, never shown to the user."""
    t = token
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("ing") and len(t) > 5:
        return t[:-3]
    if t.endswith("ed") and len(t) > 4:
        return t[:-2]
    if t.endswith("s") and len(t) > 3:
        return t[:-1]
    return t


def _suggest_tokens(text: str) -> set[str]:
    return {
        _suggest_stem(t)
        for t in re.split(r"[^a-z0-9]+", (text or "").lower())
        if len(t) > 2 and t not in _SUGGEST_STOPWORDS
    }


def suggest_closest_terms(
    question: str,
    account_id: str = "",
    kb_dir: str = "",
    limit: int = 3,
) -> list[str]:
    """Closest known vocabulary to the question, for 'did you mean' lines on
    terminal failures (CANNOT_GENERATE / unknown_column).

    Sources — all existing, no new index: registry metric names+synonyms,
    business-term glossary entries+aliases, and the semantic model's approved
    meanings / business candidates. A tester who asks for 'customer ordered
    quantity' and dead-ends gets pointed at 'purchase order quantity' instead
    of a generic 'try rephrasing'. Never raises; returns [] on any problem.
    """
    q_tokens = _suggest_tokens(question)
    if not q_tokens:
        return []

    scored: dict[str, tuple[float, str]] = {}

    def _consider(phrase: Any) -> None:
        display = re.sub(r"\s+", " ", str(phrase or "").strip())
        if len(display) < 4 or len(display) > 60:
            return
        p_tokens = _suggest_tokens(display)
        if not p_tokens:
            return
        overlap = q_tokens & p_tokens
        # Require either two shared meaningful tokens or a fully-contained
        # phrase — single-word grazes ("amount") would suggest everything.
        if len(overlap) < 2 and overlap != p_tokens:
            return
        score = len(overlap) + len(overlap) / len(p_tokens)
        key = display.lower()
        if key not in scored or score > scored[key][0]:
            scored[key] = (score, display)

    try:
        import store
        for metric in store.list_metrics(account_id) or []:
            _consider(metric.get("name"))
            for syn in str(metric.get("synonyms") or "").split(","):
                _consider(syn)
        for term_row in store.list_terms(account_id) or []:
            _consider(term_row.get("term"))
            for alias in str(term_row.get("aliases") or "").split(","):
                _consider(alias)
    except Exception:
        pass

    try:
        if kb_dir:
            from core.semantic_model import load_semantic_model
            for table in (load_semantic_model(kb_dir) or {}).get("tables") or []:
                for field in table.get("fields") or []:
                    if str(field.get("status") or "") == "approved":
                        _consider(field.get("approved_meaning"))
                    for cand in (field.get("business_candidates") or [])[:3]:
                        _consider(cand)
    except Exception:
        pass

    ranked = sorted(scored.values(), key=lambda item: -item[0])
    return [display for _score, display in ranked[:max(1, int(limit))]]


# ── Validator-code translations ───────────────────────────────────────────────

_VALIDATION_REASONS: dict[str, str] = {
    "field_plan_mismatch": "fail.v.field_plan_mismatch.reason",
    "entity_field_unavailable": "fail.v.entity_field_unavailable.reason",
    "unknown_column": "fail.v.unknown_column.reason",
    "unknown_table": "fail.v.unknown_table.reason",
    "access_denied": "fail.v.access_denied.reason",
    "anti_join_shape": "fail.v.anti_join_shape.reason",
    "composition_shape": "fail.v.composition_shape.reason",
    "date_key_format": "fail.v.date_key_format.reason",
    "metric_formula_mismatch": "fail.v.metric_formula_mismatch.reason",
    "null_aggregate_diagnostic": "fail.v.null_aggregate_diagnostic.reason",
    "order_alias_mismatch": "fail.v.order_alias_mismatch.reason",
    "period_comparison_shape": "fail.v.period_comparison_shape.reason",
    "parse": "fail.v.parse.reason",
    "ddl": "fail.v.ddl.reason",
    "cannot_generate": "fail.v.cannot_generate.reason",
    "dialect_mismatch": "fail.v.dialect_mismatch.reason",
    "production_shape": "fail.v.production_shape.reason",
    "top_n_shape": "fail.v.top_n_shape.reason",
    "graph_plan_mismatch": "fail.v.graph_plan_mismatch.reason",
    "multi_statement": "fail.v.multi_statement.reason",
    "not_select": "fail.v.not_select.reason",
    "reused_plan_empty": "fail.v.reused_plan_empty.reason",
    "surrogate_date_conversion": "fail.v.surrogate_date_conversion.reason",
    "temporal_anchor_missing": "fail.v.temporal_anchor_missing.reason",
    "temporal_anchor_mismatch": "fail.v.temporal_anchor_mismatch.reason",
    "temporal_role_mismatch": "fail.v.temporal_role_mismatch.reason",
    "temporal_anchor_unscoped": "fail.v.temporal_anchor_unscoped.reason",
    "source_fact_mismatch": "fail.v.source_fact_mismatch.reason",
    "raw_fact_to_fact_join": "fail.v.raw_fact_to_fact_join.reason",
    "fanout_aggregate": "fail.v.fanout_aggregate.reason",
    "derived_measure_mismatch": "fail.v.derived_measure_mismatch.reason",
    "locking_select": "fail.v.locking_select.reason",
    # The sixteen the validator could emit and this map had never heard of. The
    # cost was not a vaguer card: core/query_pipeline.py's terminal handler gated
    # the card itself on membership here, so each of these ended the turn with the
    # raw validator sentence and no card, in English whatever the reader's
    # language. tests/test_every_validator_refusal_has_a_reader.py now fails if a
    # seventeenth is added without one.
    "cartesian_join": "fail.v.cartesian_join.reason",
    "missing_join_condition": "fail.v.missing_join_condition.reason",
    "graph_join_missing": "fail.v.graph_join_missing.reason",
    "graph_join_type_mismatch": "fail.v.graph_join_type_mismatch.reason",
    "join_plan_unresolved": "fail.v.join_plan_unresolved.reason",
    "field_plan_join_missing": "fail.v.field_plan_join_missing.reason",
    "bridge_allocation_missing": "fail.v.bridge_allocation_missing.reason",
    "bridge_allocation_unresolved": "fail.v.bridge_allocation_unresolved.reason",
    "multi_fact_not_isolated": "fail.v.multi_fact_not_isolated.reason",
    "multi_fact_not_aggregated": "fail.v.multi_fact_not_aggregated.reason",
    "multi_fact_shared_cte": "fail.v.multi_fact_shared_cte.reason",
    "multi_fact_cte_contract": "fail.v.multi_fact_cte_contract.reason",
    "multi_fact_missing_subplan": "fail.v.multi_fact_missing_subplan.reason",
    "temporal_anchor_ungoverned": "fail.v.temporal_anchor_ungoverned.reason",
    "observed_period_shape": "fail.v.observed_period_shape.reason",
    "select_star": "fail.v.select_star.reason",
}

_VALIDATION_NEXT_STEPS: dict[str, str] = {
    "cartesian_join": "fail.v.cartesian_join.next_step",
    "join_plan_unresolved": "fail.v.join_plan_unresolved.next_step",
    "bridge_allocation_missing": "fail.v.bridge_allocation_missing.next_step",
    "temporal_anchor_ungoverned": "fail.v.temporal_anchor_ungoverned.next_step",
    "select_star": "fail.v.select_star.next_step",
    "access_denied": "fail.v.access_denied.next_step",
    "cannot_generate": "fail.v.cannot_generate.next_step",
    "dialect_mismatch": "fail.v.dialect_mismatch.next_step",
    "graph_plan_mismatch": "fail.v.graph_plan_mismatch.next_step",
    "composition_shape": "fail.v.composition_shape.next_step",
    "entity_field_unavailable": "fail.v.entity_field_unavailable.next_step",
    "surrogate_date_conversion": "fail.v.surrogate_date_conversion.next_step",
    "reused_plan_empty": "fail.v.reused_plan_empty.next_step",
    "temporal_anchor_missing": "fail.v.temporal_anchor_missing.next_step",
    "temporal_anchor_mismatch": "fail.v.temporal_anchor_mismatch.next_step",
    "temporal_role_mismatch": "fail.v.temporal_role_mismatch.next_step",
    "temporal_anchor_unscoped": "fail.v.temporal_anchor_unscoped.next_step",
    "source_fact_mismatch": "fail.v.source_fact_mismatch.next_step",
    "raw_fact_to_fact_join": "fail.v.raw_fact_to_fact_join.next_step",
    "fanout_aggregate": "fail.v.fanout_aggregate.next_step",
    "derived_measure_mismatch": "fail.v.derived_measure_mismatch.next_step",
    "locking_select": "fail.v.locking_select.next_step",
    "order_alias_mismatch": "fail.v.order_alias_mismatch.next_step",
}
_DEFAULT_VALIDATION_NEXT_STEP = "fail.v.default.next_step"


def translate_failure(
    *,
    kind: str,
    code: str = "",
    reason: str = "",
    exception_text: str = "",
    sql: str = "",
    question: str = "",
    context: dict[str, Any] | None = None,
    suggestions: list[str] | None = None,
) -> dict[str, Any]:
    """
    Build a business-readable failure RCA. Same dict shape as
    core/answer_rca.build_business_rca: {headline, most_likely_reason,
    suggested_next_step, technical_notes}. Never raises.
    """
    try:
        if kind == "execution":
            info = sanitize_db_error(exception_text or reason)
            technical = []
            if info["cleaned"]:
                technical.append(
                    _t("fail.db.error_prefix", detail=_clip(info["cleaned"])))
            return {
                "kind": "execution",
                "headline": _t("fail.exec.headline"),
                "most_likely_reason": info["plain_reason"],
                "suggested_next_step": info["next_step"],
                "technical_notes": technical,
            }

        if kind == "validation":
            code_key = (code or "").strip().lower()
            if code_key == "entity_field_unavailable" and reason:
                # This guarded failure contains safe, schema-derived entity
                # alternatives and is more useful than a generic translation.
                plain = _clip(reason)
            else:
                plain = (
                    _t(_VALIDATION_REASONS[code_key])
                    if code_key in _VALIDATION_REASONS
                    else _t("fail.v.default.reason")
                )
            technical = []
            if code_key:
                technical.append(f"Validation: {code_key}")
            if reason and code_key != "entity_field_unavailable":
                technical.append(_clip(reason))
            next_step = _t(_VALIDATION_NEXT_STEPS.get(
                code_key, _DEFAULT_VALIDATION_NEXT_STEP))
            if suggestions:
                next_step += " " + _t("fail.v.suggestions",
                                      terms=", ".join(suggestions))
            return {
                "kind": "validation",
                "headline": _t("fail.validation.headline"),
                "most_likely_reason": plain,
                "suggested_next_step": next_step,
                "technical_notes": technical,
            }

        # Unknown kind — generic but safe.
        technical = [note for note in [_clip(reason or exception_text)] if note]
        return {
            "kind": "validation",
            "headline": _t("fail.generic.headline"),
            "most_likely_reason": _t("fail.generic.reason"),
            "suggested_next_step": _t("fail.generic.next_step_technical"),
            "technical_notes": technical,
        }
    except Exception:
        # Fail-open: a reader gets a card rather than a stack trace. Kept
        # literal-free so even this path speaks their language.
        return {
            "kind": "validation",
            "headline": _t("fail.generic.headline"),
            "most_likely_reason": _t("fail.generic.reason"),
            "suggested_next_step": _t("fail.generic.next_step"),
            "technical_notes": [],
        }
