"""
core/situation_phraser.py
─────────────────────────
A failed turn, told in the reader's own terms -- by the model, and checked.

When the product cannot answer -- the validator refused the SQL, the database
raised, the query returned no rows -- the reader gets a diagnostic card whose
sentences come from the catalogue: correct, translated, and the same every
time. core/failure_messages.translate_failure and core/answer_rca.build_business_rca
establish WHAT happened; nothing on that path reads the question the reader
asked, so the card cannot say "there is no 2023 scrap figure for the Lyon
plant in this data", only "no records matched the filters".

This module asks the model to tell the reader that same situation -- and only
that situation -- in two short parts, tied to their question and in their
language, and checks the prose before it is shown:

  • every figure the prose asserts must be one the situation states;
  • every quoted term and every column-like identifier must come from the
    situation or the question;
  • both parts must be present, and short.

Anything that fails the check, a provider error, or a regulated tenant
(result_llm_features_allowed refuses, and the refusal is recorded) leaves the
reader with the catalogue's sentences, exactly as before. The system's
diagnosis is never changed: the model rewords the WHY and the NEXT STEP the
system produced; it does not produce its own.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger("querybot.situation_phraser")

_MAX_TOKENS = 260
_MAX_REASON_CHARS = 420
_MAX_NEXT_CHARS = 260

# The two labelled parts the prompt asks for. Labels are wire format between
# this module and the model, kept in English whatever language the sentences
# are in -- the same convention core/insight.py uses for HEADLINE:/BODY:.
_PARTS_RE = re.compile(
    r"REASON:\s*(?P<reason>.+?)\s*NEXT:\s*(?P<next>.+)", re.IGNORECASE | re.DOTALL,
)
_QUOTED_RE = re.compile(r"[\"«]\s*([^\"»\n]{2,80}?)\s*[\"»]|'([^'\n]{2,80})'|`([^`\n]{2,80})`")
# NET_SLS_AMT, F_SALES, DBO.F_SALES: a column or table spelled the way a
# warehouse spells it. A plain capitalised word is not one.
_IDENTIFIER_RE = re.compile(r"\b[A-Z][A-Z0-9]*_[A-Z0-9_]+\b")
_ROUNDING_TOLERANCE = 0.51


def situation_text(rca: dict, question: str) -> str:
    """Everything the model was told, as one text the checks read against."""
    parts = [question, rca.get("headline"), rca.get("most_likely_reason"),
             rca.get("suggested_next_step")]
    return "\n".join(str(p) for p in parts if p)


def verify_phrasing(reason: str, next_step: str, situation: str) -> tuple[bool, str]:
    """Is the model's wording only a rewording of the situation?

    ``(ok, why)``. Small numbers pass ("the top 3" is not a claim about the
    data), a figure within a rounding step of a stated one passes, and a
    quoted term or identifier passes only if the situation or the question
    contains it.
    """
    from core.analysis_narrative import SMALL_NUMBER_CEILING, numbers_in

    reason = (reason or "").strip()
    next_step = (next_step or "").strip()
    if not reason or not next_step:
        return False, "missing_part"
    if len(reason) > _MAX_REASON_CHARS or len(next_step) > _MAX_NEXT_CHARS:
        return False, "too_long"
    prose = f"{reason}\n{next_step}"
    known = numbers_in(situation or "")
    for value in numbers_in(prose):
        if abs(value) <= SMALL_NUMBER_CEILING:
            continue
        if any(abs(value - k) <= _ROUNDING_TOLERANCE for k in known):
            continue
        return False, f"uncomputed_figure:{value:g}"
    folded = (situation or "").lower()
    for match in _QUOTED_RE.finditer(prose):
        term = next((g for g in match.groups() if g), "").strip()
        if term and term.lower() not in folded:
            return False, f"unknown_term:{term[:40]}"
    for ident in _IDENTIFIER_RE.findall(prose):
        if ident.lower() not in folded:
            return False, f"unknown_identifier:{ident}"
    return True, ""


def parse_parts(raw: str) -> tuple[str, str]:
    """(reason, next step) from the model's reply, or ("", "") if it did not
    keep to the two labelled parts."""
    match = _PARTS_RE.search(raw or "")
    if not match:
        return "", ""
    return match.group("reason").strip(), match.group("next").strip()


def build_phrasing_prompt(rca: dict, *, question: str, lang: str | None = None) -> tuple[str, str]:
    """The system and user halves of the request to reword a situation."""
    from core.i18n import prompt_language_rule

    system = (
        "You are QueryBot's analyst. The product could not answer the reader's "
        "question, and the system has already established why. Tell the reader "
        "that situation in their own terms, in two short parts:\n"
        "REASON: one or two sentences on what happened, in plain business "
        "language, tied to what they asked.\n"
        "NEXT: one sentence on what they can do now.\n\n"
        + prompt_language_rule(lang, shape="prose") +
        "RULES:\n"
        "1. Say only what the situation below says. Do not guess at other causes.\n"
        "2. Keep every quoted term, name and number exactly as given, and add none.\n"
        "3. Do not mention SQL, tables, columns or systems unless the situation names them.\n"
        "4. No apology longer than three words. No filler.\n"
        "5. The labels REASON: and NEXT: stay in English, in capitals, each at the "
        "start of its line, whatever language the sentences are in.\n"
    )
    user = (
        f"Reader's question: {question}\n"
        f"What happened: {rca.get('headline') or ''}\n"
        f"Why: {rca.get('most_likely_reason') or ''}\n"
        f"What they can do: {rca.get('suggested_next_step') or ''}"
    )
    return system, user


async def phrase_failure(
    rca: dict,
    *,
    question: str,
    account_id: str,
    client: dict | None,
    provider: str,
    model: str,
    api_key: str,
    lang: str | None = None,
    **extra_kwargs,
) -> dict:
    """The RCA with its reason and next step reworded for this reader -- or
    the RCA exactly as it came, whenever the rewording cannot be trusted.

    The headline, the kind and the technical notes are the system's and stay;
    a note is added saying the wording is the model's. ``extra_kwargs`` is
    what resolve_provider returned as its fourth value (the Azure endpoint
    and deployment), forwarded to the model call.
    """
    if not isinstance(rca, dict) or not str(rca.get("most_likely_reason") or "").strip():
        return rca
    from core.compliance.policy_engine import result_llm_features_allowed
    from core.i18n import t as _t
    from core.llm import llm_complete
    from core.llm_audit import llm_audit_scope, record_llm_blocked

    try:
        with llm_audit_scope(
            account_id=account_id,
            question=str(question or "")[:500],
            enabled=bool((client or {}).get("enable_llm_audit")),
            component="failure_phraser",
        ):
            if not result_llm_features_allowed(account_id):
                record_llm_blocked(
                    "failure_phraser",
                    "Failure explanation kept in the catalogue's words -- "
                    "regulated tenant, the model was not asked to reword it.",
                )
                return rca
            system, user = build_phrasing_prompt(rca, question=question, lang=lang)
            raw, _tok_in, _tok_out = await llm_complete(
                system, user, provider, model, api_key,
                max_tokens=_MAX_TOKENS, temperature=0.3,
                allow_truncated=True,
                **extra_kwargs,
            )
    except Exception as exc:  # noqa: BLE001 - the catalogue's sentences stand
        log.warning("Failure phrasing skipped (%s); the catalogue's sentences stand", exc)
        return rca

    reason, next_step = parse_parts(str(raw or ""))
    ok, why = verify_phrasing(reason, next_step, situation_text(rca, question))
    if not ok:
        log.warning("Failure phrasing rejected (%s); the catalogue's sentences stand", why)
        return rca
    notes = list(rca.get("technical_notes") or [])
    notes.append(_t("fail.phrased_note"))
    return {
        **rca,
        "most_likely_reason": reason,
        "suggested_next_step": next_step,
        "technical_notes": notes,
        "phrasing": "llm",
    }
