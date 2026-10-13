"""Answers kept for history and export, seen again under the reader's clearance now, not then.

A reader cleared to see people's data (a valid confidentiality attestation) is shown it as stored, and the
answer is kept so. When the attestation is revoked or runs out, the kept answer must not show it again:
rows the existing core answered with are masked again under the workspace's policies today; an answer of
the new core (its wording may name the people too) is withheld, with a sentence that says why, when it
showed people's data as stored: the new core said so when it answered, or it was given while the reader was
cleared and today's policies mask something its query read. An answer the reader was never cleared for was
masked when given, and is shown as kept.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any

import store

log = logging.getLogger(__name__)

WITHHELD = ("This answer showed people's data you are no longer cleared to see. Ask the question again to see "
            "it as your access allows today.")


def cleared(account_id: str, user_id: Any) -> bool:
    try:
        return bool(store.user_attestation_valid(account_id, str(user_id or "")))
    except Exception:  # noqa: BLE001 - a clearance that cannot be read is no clearance
        return False


def _said_released(trace: dict[str, Any]) -> bool:
    """The new core kept with the answer that people's data went out as stored."""
    try:
        frame = json.loads(trace.get("answer_frame") or "null")
    except (TypeError, ValueError):
        frame = None
    return isinstance(frame, dict) and bool(frame.get("released"))


def was_released(account_id: str, user_id: Any, trace: dict[str, Any]) -> bool:
    """Was this answer given to its reader unmasked: the new core said so, or it was given while the reader
    held a valid attestation?"""
    if _said_released(trace):
        return True
    at = str(trace.get("created_at") or "").replace("T", " ")[:19]
    if not at:
        return False
    try:
        rows = store.list_user_attestations(account_id)
    except Exception:  # noqa: BLE001 - unreadable: treated as released, so it is masked again
        return True
    for row in rows:
        if str(row.get("portal_user_id") or "") != str(user_id or ""):
            continue
        start = str(row.get("granted_at") or "")[:19]
        ends = [str(e)[:19] for e in (row.get("revoked_at"), row.get("expires_at")) if e]
        if start <= at and (not ends or at < min(ends)):
            return True
    return False


def withheld(account_id: str, user: Any, trace: dict[str, Any]) -> bool:
    """A kept answer of the new core that showed people's data to a reader no longer cleared to see it.

    ``user`` is the reader (or their id)."""
    user = user if isinstance(user, dict) else {"id": user}
    if trace.get("route") != "core2" or cleared(account_id, user.get("id")):
        return False
    if _said_released(trace):
        return True
    if not was_released(account_id, user.get("id"), trace):
        return False
    # Given while cleared: withheld when today's policies would mask something its query read.
    from core.compliance.policy_engine import evaluate, resolve_context
    from core.compliance.sql_guard import analyze_sql

    sql = str(trace.get("generated_sql") or "")
    if not sql:
        return False
    try:
        analysis = analyze_sql(sql, str(trace.get("db_type") or "azure_sql"))
        decision = evaluate(resolve_context(account_id, user, action="result_release", channel="portal"),
                            analysis.resources)
    except Exception as exc:  # noqa: BLE001 - an answer that cannot be checked again is not shown
        log.warning("Kept answer of trace %s could not be checked again for %s: %s", trace.get("id"), account_id, exc)
        return True
    return bool(decision.masking)


NARROWED = ("The rows you may see have changed since this answer was given. Ask the question again to see it as "
            "your access allows today.")


def narrowed_since(account_id: str, user: Any, trace: dict[str, Any]) -> bool:
    """A kept answer given before what this reader may see was narrowed: a row rule that binds them added or
    changed since, or their own values or group changed since. Its rows were released under the rules then, and
    may hold rows they may no longer see; it is asked again, not shown or exported as kept. A rule taken away
    only widens what they see, and leaves the answer as it was."""
    user = user if isinstance(user, dict) else {"id": user}
    at = str(trace.get("created_at") or "").replace("T", " ")[:19]
    if not at:
        return False
    try:
        from core.compliance.sql_guard import _policy_subject_matches

        # Who the rules name, read as the governed executor reads it (role, person, group), but not through the
        # whole policy engine: a failure there is the export guard's to judge, not a reason to withhold every
        # kept answer.
        person = user
        if user.get("id") not in (None, "") and "role" not in user:
            person = {**(store.get_user(int(user["id"])) or {}), **user}
        subject = SimpleNamespace(
            role=str(person.get("role") or "system"), user_id=str(person.get("id") or ""),
            groups=[str(person[k]) for k in ("group_id", "group_name") if person.get(k) not in (None, "")])
        version = int(store.get_compliance_profile(account_id).get("active_policy_version") or 0)
        changed = [str(p.get("updated_at") or p.get("created_at") or "")[:19]
                   for p in store.list_row_policies(account_id, version or None)
                   if _policy_subject_matches(p, subject)]
        if user.get("id") not in (None, ""):
            with store.get_db() as conn:
                row = conn.execute("SELECT access_changed_at FROM portal_user WHERE id=?",
                                   (int(user["id"]),)).fetchone()
            if row is not None:
                changed.append(str(row["access_changed_at"] or "")[:19])
    except Exception as exc:  # noqa: BLE001 - an answer that cannot be checked again is not shown
        log.warning("Kept answer of trace %s could not be checked against row rules for %s: %s",
                    trace.get("id"), account_id, exc)
        return True
    return any(when and when > at for when in changed)


def remasked(account_id: str, user: dict[str, Any], trace: dict[str, Any], rows: list[dict]) -> list[dict]:
    """Kept rows as the reader may see them today: masked again under the workspace's policies when they were
    given unmasked and the reader is no longer cleared; as kept otherwise."""
    if not rows or not was_released(account_id, user.get("id"), trace) or cleared(account_id, user.get("id")):
        return rows
    from core.compliance.policy_engine import evaluate, resolve_context
    from core.compliance.result_guard import protect_rows
    from core.compliance.sql_guard import analyze_sql

    sql = str(trace.get("generated_sql") or "")
    try:
        analysis = analyze_sql(sql, str(trace.get("db_type") or "azure_sql"))
        decision = evaluate(resolve_context(account_id, user, action="result_release", channel="portal"),
                            analysis.resources)
    except Exception as exc:  # noqa: BLE001 - rows that cannot be checked again are not shown
        log.warning("Kept rows of trace %s could not be checked again for %s: %s", trace.get("id"), account_id, exc)
        return []
    if not decision.masking:
        return rows
    return protect_rows(rows, decision, analysis.lineage, account_id=account_id,
                        mask_exempt_outputs=analysis.mask_exempt_outputs)
