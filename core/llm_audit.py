from __future__ import annotations

import contextvars
import hashlib
import json as _json
import logging
import re
from contextlib import contextmanager
from typing import Iterator
from uuid import uuid4

log = logging.getLogger("querybot.llm_audit")

_AUDIT_SCOPE: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "querybot_llm_audit_scope",
    default=None,
)

_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_GUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.I,
)
_LONG_NUMBER_RE = re.compile(r"\b\d{6,}\b")

# Long opaque tokens (API keys, base64 blobs). Broad match — we refine in the
# callback so we don't mask SCREAMING_SNAKE_CASE column names / table names
# that happen to be long.
_LONG_TOKEN_RE = re.compile(r"\b[A-Za-z0-9_\-]{20,}\b")

_PHONE_RE = re.compile(r"\+?\d[\d\-\s()]{7,}\d")

# Pre-compiled patterns used inside _mask_long_token to classify identifiers.
# Defined at module level so re.fullmatch doesn't compile them on every call.
_RE_UPPER_IDENT = re.compile(r"[A-Z0-9_]+")   # SCREAMING_SNAKE / schema names
_RE_LOWER_IDENT = re.compile(r"[a-z0-9_]+")   # lowercase_snake identifiers

# Quoted literals. We match 1–120 chars (not 2+) so that a short literal like
# 'Y' doesn't get skipped — if skipped, the regex engine would then consume
# text between a pair of alternating quotes and treat it as one long literal.
# The length/shape decision of whether to mask happens in _looks_like_data_value.
_SINGLE_QUOTED_RE = re.compile(r"'([^'\n]{1,120})'")
_DOUBLE_QUOTED_RE = re.compile(r'"([^"\n]{1,120})"')


_SHORT_CATEGORICALS = {
    # Short tokens we want to keep in previews because they make audit rows
    # readable and they almost never carry PII on their own.
    "active", "inactive", "late", "early", "absent", "present", "pending",
    "approved", "rejected", "open", "closed", "draft", "new", "old",
    "yes", "no", "true", "false", "male", "female", "unknown", "none",
    "y", "n", "m", "f",
}


def _looks_like_data_value(s: str) -> bool:
    """Return True if a quoted literal should be redacted as a data value."""
    # Empty or single-char literals carry no PII on their own.
    if len(s) <= 1:
        return False
    # Known short categoricals (case-insensitive) — always preserve.
    if s.lower().strip(",.") in _SHORT_CATEGORICALS:
        return False
    # Anything ≥10 chars gets masked regardless of shape.
    if len(s) >= 10:
        return True
    # Multi-word values are almost always data, not schema vocab.
    if " " in s:
        return True
    # A capitalised word of 3+ letters looks like a proper noun.
    if re.search(r"[A-Z][a-z]{2,}", s):
        return True
    return False


def _mask_quoted(match: re.Match, quote: str) -> str:
    inner = match.group(1)
    if _looks_like_data_value(inner):
        return f"{quote}[literal]{quote}"
    return match.group(0)


def _mask_long_token(match: re.Match) -> str:
    """
    Mask a run of 20+ alphanumeric/underscore/dash characters iff it looks
    like an opaque secret (API key, base64 blob). Preserves:
      • SCREAMING_SNAKE_CASE  — table / column names (pure uppercase + _)
      • lowercase snake_case identifiers that contain an underscore
    Masks:
      • Pure alphanumeric runs with no underscore (random tokens / hashes)
      • Mixed-case tokens (API keys like sk-abc123XYZ...)
    """
    s = match.group(0)
    # Pure uppercase + underscores + digits → almost always a schema identifier.
    if _RE_UPPER_IDENT.fullmatch(s):
        return s
    # Lowercase snake_case identifier (must contain an underscore so random
    # hex strings like 'a1b2c3d4...' don't get a free pass).
    if _RE_LOWER_IDENT.fullmatch(s) and "_" in s:
        return s
    # Otherwise treat as opaque token.
    return "[token]"


# Module-level callables for _SINGLE_QUOTED_RE / _DOUBLE_QUOTED_RE substitutions.
# Avoids recreating a closure object on every sanitize_llm_text() call.
def _mask_single_quoted(match: re.Match) -> str:
    return _mask_quoted(match, "'")


def _mask_double_quoted(match: re.Match) -> str:
    return _mask_quoted(match, '"')


def make_llm_audit_request_id() -> str:
    return uuid4().hex[:12]


@contextmanager
def llm_audit_scope(
    *,
    account_id: str,
    question: str,
    enabled: bool,
    request_id: str | None = None,
    question_id: str | None = None,
    component: str = "general",
    egress: dict | None = None,
) -> Iterator[dict]:
    current = _AUDIT_SCOPE.get() or {}
    merged = {
        **current,
        "account_id":  account_id,
        "question":    (question or "").strip(),
        "enabled":     bool(enabled),
        # request_id is unique PER CALL. If not supplied, generate one.
        "request_id":  request_id or current.get("request_id") or make_llm_audit_request_id(),
        # question_id is stable across the WHOLE USER QUESTION including
        # all drilldowns and follow-ups. Falls back to request_id for scopes
        # that don't supply one (e.g. KB build jobs where there's no parent question).
        "question_id": question_id or current.get("question_id") or "",
        "component":   component or current.get("component") or "general",
        # Descriptive egress metadata the call site knows and the prompt text
        # cannot be reliably parsed for — chiefly which tables and columns were
        # described to the model. Optional: build_egress_manifest still derives
        # the compliance-critical values_sent flag without it, so a call site
        # that supplies nothing is recorded honestly rather than not at all.
        "egress":      dict(egress) if egress else (current.get("egress") or {}),
    }
    token = _AUDIT_SCOPE.set(merged)
    try:
        yield merged
    finally:
        _AUDIT_SCOPE.reset(token)


@contextmanager
def llm_audit_component(
    component: str,
    *,
    question: str | None = None,
    new_request_id: bool = False,
) -> Iterator[dict | None]:
    """
    Narrow the ambient scope to one component.

    ``new_request_id`` mints a fresh request_id for this component instead of
    inheriting the parent's. Long-running parents — chiefly the KB build, which
    opens a single scope and then makes one LLM call per table — otherwise give
    every child call the same request_id, so individual calls cannot be told
    apart in the audit log. Grouping is unaffected: the admin view buckets by
    question_id, which still comes from the parent.
    """
    current = _AUDIT_SCOPE.get()
    if not current:
        yield None
        return
    merged = dict(current)
    merged["component"] = component or merged.get("component") or "general"
    if new_request_id:
        merged["request_id"] = make_llm_audit_request_id()
    if question is not None and question.strip():
        merged["question"] = question.strip()
    token = _AUDIT_SCOPE.set(merged)
    try:
        yield merged
    finally:
        _AUDIT_SCOPE.reset(token)


def get_current_llm_audit_scope() -> dict | None:
    return _AUDIT_SCOPE.get()


# ── Egress manifest ──────────────────────────────────────────────────────────
#
# Markers of prompt sections that carry REAL DATA VALUES. Each is a literal
# header emitted by the code that injects the section, so detection is exact
# rather than heuristic — which is what lets values_sent be computed evidence
# instead of a claim. Adding a new value-bearing prompt section means adding
# its marker here; that is the one maintenance obligation this design carries.
_VALUE_BEARING_MARKERS: tuple[tuple[str, str], ...] = (
    # core/value_resolver.py::build_verified_values_injection
    ("VERIFIED FILTER VALUES", "value_index"),
    # core/query_pipeline.py / gateway/webhooks.py repair prompts. The DB error
    # itself is masked by core.failure_messages.scrub_error_for_llm, so it is
    # NOT the value source here — the echoed prior SQL is, because its WHERE
    # literals are reproduced verbatim. Those literals were authored by the
    # model on the previous turn rather than newly disclosed, which is why the
    # repair path was left intact in Phase 0; the manifest still reports them
    # so the record stays honest about what the prompt contained.
    ("The following SQL failed with this error", "echoed_sql"),
)

_CONTENT_MARKERS: tuple[tuple[str, str], ...] = (
    ("COLUMN SYNONYM MAP", "synonyms"),
    ("BUSINESS TERM DEFINITIONS", "business_terms"),
    ("Session context", "conversation_history"),
    ("Example queries", "few_shot"),
    ("Entity graph", "entity_graph"),
)


def build_egress_manifest(system: str, user: str, scope: dict | None = None) -> dict:
    """
    Describe what this call actually sent to the model.

    Two sources, deliberately separated by trustworthiness:

    * ``values_sent`` / ``value_sources`` are DERIVED from the assembled prompt
      text by matching literal section headers. Nothing has to be declared, so
      a call site cannot forget to report a leak, and an unaudited-by-omission
      path is impossible.
    * ``tables`` / ``columns`` come from the audit scope's ``egress`` dict,
      because they cannot be recovered from prose. They are descriptive only;
      no compliance decision rests on them.

    Returns a plain dict for JSON storage. Never raises — a manifest failure
    must not lose the audit row it belongs to.
    """
    manifest: dict = {
        "tables": [],
        "columns": [],
        "content": [],
        "values_sent": False,
        "value_sources": [],
    }
    try:
        blob = f"{system or ''}\n{user or ''}"
        sources = [label for marker, label in _VALUE_BEARING_MARKERS if marker in blob]
        manifest["value_sources"] = sources
        manifest["values_sent"] = bool(sources)
        manifest["content"] = [label for marker, label in _CONTENT_MARKERS if marker in blob]
        if user:
            manifest["content"].insert(0, "question")

        egress = (scope or {}).get("egress") or {}
        tables = egress.get("tables") or []
        columns = egress.get("columns") or []
        # Bounded and de-duplicated: this is an audit summary, not a payload.
        manifest["tables"] = sorted({str(t) for t in tables if t})[:50]
        manifest["columns"] = sorted({str(c) for c in columns if c})[:200]
    except Exception as exc:  # noqa: BLE001 — never break the audit write
        log.debug("Egress manifest build failed: %s", exc)
    return manifest


def sanitize_llm_text(text: str, *, limit: int = 1200) -> str:
    if not text:
        return ""
    cleaned = text
    cleaned = _EMAIL_RE.sub("[email]", cleaned)
    cleaned = _GUID_RE.sub("[guid]", cleaned)
    cleaned = _PHONE_RE.sub("[phone]", cleaned)
    cleaned = _LONG_NUMBER_RE.sub("[number]", cleaned)
    cleaned = _LONG_TOKEN_RE.sub(_mask_long_token, cleaned)
    cleaned = _SINGLE_QUOTED_RE.sub(_mask_single_quoted, cleaned)
    cleaned = _DOUBLE_QUOTED_RE.sub(_mask_double_quoted, cleaned)
    cleaned = cleaned.strip()
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 3].rstrip() + "..."
    return cleaned


def sanitize_payload_preview(system: str, user: str) -> str:
    parts: list[str] = []
    system_preview = sanitize_llm_text(system, limit=900)
    user_preview = sanitize_llm_text(user, limit=1500)
    if system_preview:
        parts.append(f"[SYSTEM]\n{system_preview}")
    if user_preview:
        parts.append(f"[USER]\n{user_preview}")
    preview = "\n\n".join(parts)
    if len(preview) > 2600:
        preview = preview[:2597].rstrip() + "..."
    return preview


def _payload_hash(system: str, user: str) -> str:
    raw = f"[SYSTEM]\n{system}\n\n[USER]\n{user}".encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()


def _response_hash(response: str) -> str:
    return hashlib.sha256((response or "").encode("utf-8", errors="ignore")).hexdigest()


def record_llm_call(
    *,
    llm_provider: str,
    llm_model: str,
    system: str,
    user: str,
    status: str,
    error_msg: str = "",
    response: str = "",
) -> None:
    scope = _AUDIT_SCOPE.get()
    if not scope or not scope.get("enabled") or not scope.get("account_id"):
        return

    try:
        import store

        store.log_llm_call(
            account_id=scope["account_id"],
            question_id=str(scope.get("question_id") or scope.get("request_id") or ""),
            request_id=str(scope.get("request_id") or ""),
            question=sanitize_llm_text(str(scope.get("question") or ""), limit=400),
            component=str(scope.get("component") or "general"),
            llm_provider=llm_provider or "",
            llm_model=llm_model or "",
            status=status,
            payload_hash=_payload_hash(system, user),
            payload_preview_sanitized=sanitize_payload_preview(system, user),
            prompt_chars=len(system or "") + len(user or ""),
            error_msg=(error_msg or "")[:500],
            # Same design as the prompt side: the hash is tamper-evident
            # proof of the EXACT text the model returned; only a sanitized
            # preview is retained so the audit log never becomes a store of
            # raw generated content. Empty on error rows (no response).
            response_hash=_response_hash(response) if response else "",
            response_preview_sanitized=sanitize_llm_text(response, limit=1200),
            response_chars=len(response or ""),
            egress_manifest=_json.dumps(
                build_egress_manifest(system, user, scope), separators=(",", ":")
            ),
        )
    except Exception as exc:
        log.warning("LLM audit write failed: %s", exc)


def record_llm_blocked(component: str, reason: str) -> None:
    """
    Record a proof-of-refusal row: a call site decided NOT to invoke the LLM
    at all (see core.compliance.policy_engine.result_llm_features_allowed)
    and is logging that decision instead of the (never-built) prompt.

    Unlike record_llm_call, there is no system/user prompt to hash or
    preview — status="blocked" plus the reason IS the audit record. Uses the
    same ambient scope (account_id/enabled) as record_llm_call, so this
    respects the client's existing "enable LLM audit" toggle rather than
    forcing extra rows for clients who opted out of audit logging entirely.
    """
    scope = _AUDIT_SCOPE.get()
    if not scope or not scope.get("enabled") or not scope.get("account_id"):
        return
    try:
        import store

        store.log_llm_call(
            account_id=scope["account_id"],
            question_id=str(scope.get("question_id") or scope.get("request_id") or ""),
            request_id=str(scope.get("request_id") or ""),
            question=sanitize_llm_text(str(scope.get("question") or ""), limit=400),
            component=component,
            llm_provider="",
            llm_model="",
            status="blocked",
            payload_hash="",
            payload_preview_sanitized=reason,
            prompt_chars=0,
            error_msg="",
        )
    except Exception as exc:
        log.warning("LLM audit blocked-record write failed: %s", exc)
