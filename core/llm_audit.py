from __future__ import annotations

import contextvars
import hashlib
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
    if re.fullmatch(r"[A-Z0-9_]+", s):
        return s
    # Lowercase snake_case identifier (must contain an underscore so random
    # hex strings like 'a1b2c3d4...' don't get a free pass).
    if re.fullmatch(r"[a-z0-9_]+", s) and "_" in s:
        return s
    # Otherwise treat as opaque token.
    return "[token]"


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
    }
    token = _AUDIT_SCOPE.set(merged)
    try:
        yield merged
    finally:
        _AUDIT_SCOPE.reset(token)


@contextmanager
def llm_audit_component(component: str, *, question: str | None = None) -> Iterator[dict | None]:
    current = _AUDIT_SCOPE.get()
    if not current:
        yield None
        return
    merged = dict(current)
    merged["component"] = component or merged.get("component") or "general"
    if question is not None and question.strip():
        merged["question"] = question.strip()
    token = _AUDIT_SCOPE.set(merged)
    try:
        yield merged
    finally:
        _AUDIT_SCOPE.reset(token)


def get_current_llm_audit_scope() -> dict | None:
    return _AUDIT_SCOPE.get()


def sanitize_llm_text(text: str, *, limit: int = 1200) -> str:
    if not text:
        return ""
    cleaned = text
    cleaned = _EMAIL_RE.sub("[email]", cleaned)
    cleaned = _GUID_RE.sub("[guid]", cleaned)
    cleaned = _PHONE_RE.sub("[phone]", cleaned)
    cleaned = _LONG_NUMBER_RE.sub("[number]", cleaned)
    cleaned = _LONG_TOKEN_RE.sub(_mask_long_token, cleaned)
    cleaned = _SINGLE_QUOTED_RE.sub(lambda m: _mask_quoted(m, "'"), cleaned)
    cleaned = _DOUBLE_QUOTED_RE.sub(lambda m: _mask_quoted(m, '"'), cleaned)
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


def record_llm_call(
    *,
    llm_provider: str,
    llm_model: str,
    system: str,
    user: str,
    status: str,
    error_msg: str = "",
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
        )
    except Exception as exc:
        log.warning("LLM audit write failed: %s", exc)
