"""The artefact a customer hands to their auditor.

Everything in here is generated from logs the product already writes. That is
the whole point: a governance claim backed by a policy document is a promise,
and a governance claim backed by a per-call log with a computed
``values_sent`` flag is a measurement. This module turns the second into
something a person can read.

Five sections, each answering a question an auditor actually asks:

``egress``      Where may this workspace's questions go, and did anything
                leave? For an air-gapped workspace this is the assertion that
                no external endpoint was configured at all.
``calls``       Every model call in the period, and how many of them sent a
                data value. ``values_sent`` is *derived* from the assembled
                prompt by ``core.llm_audit.build_egress_manifest``, not
                declared by the call site, so a path cannot forget to report a
                leak.
``refusals``    Every call the product declined to make, with its reason.
                Proof that the boundary fires, not just that it exists.
``grounding``   What value grounding suppressed, and how much of the schema
                is classified — because a boundary that depends on
                classification is only as complete as the classification.
``integrity``   Whether the hash-chained decision log verifies, and a
                fingerprint over this pack so the document itself is
                tamper-evident.

The pack never contains a data value. It contains counts, column names, and
the reasons the product recorded — a review of what the audit trail says,
which is exactly what an auditor is entitled to and no more.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

log = logging.getLogger("querybot.compliance.proof_pack")

# The pack's own schema version. An auditor comparing two packs from different
# releases needs to know whether a missing section means "nothing to report"
# or "this release did not report it".
PACK_VERSION = 1


def _cutoff(days: int) -> str:
    return (
        datetime.now(timezone.utc) - timedelta(days=int(days))
    ).strftime("%Y-%m-%d %H:%M:%S")


def _manifest_of(row: dict) -> dict:
    """One call's manifest, or {} when there is nothing readable.

    A row whose manifest is absent or will not parse comes back empty, and
    the caller counts an empty one as UNKNOWN rather than as clean — an audit
    that reads missing evidence as good news is worse than no audit. The
    absent case and the corrupt case are deliberately not distinguished:
    neither is evidence, and a separate marker for the second would be a
    branch that can never change the answer.
    """
    try:
        return json.loads(row.get("egress_manifest") or "{}") or {}
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# Sections
# ══════════════════════════════════════════════════════════════════════════════

def egress_section(account_id: str) -> dict:
    """Where this workspace's questions were permitted to go."""
    from core.compliance.egress import describe

    described = describe(account_id)
    return {
        **described,
        # The headline line for an air-gapped workspace. Stated separately
        # from `external_egress` because an auditor reads sentences, not
        # booleans.
        "statement": (
            "No external model endpoint is permitted for this workspace."
            if not described.get("external_egress")
            else "Model calls are permitted to the endpoints listed."
        ),
    }


def calls_section(account_id: str, days: int) -> dict:
    """Every model call in the period, and what each one sent.

    ``values_sent`` counts are the number that matters. ``unknown`` is
    reported separately from ``with_values``: a row whose manifest predates
    the manifest column, or will not parse, is not evidence of cleanliness.
    """
    import store

    cutoff = _cutoff(days)
    with store.get_db() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT status, component, llm_provider, llm_model, egress_manifest "
            "FROM llm_call_log WHERE account_id=? AND created_at >= ?",
            (account_id, cutoff),
        ).fetchall()]

    by_status: dict[str, int] = {}
    by_component: dict[str, int] = {}
    endpoints: set[str] = set()
    tables: set[str] = set()
    columns: set[str] = set()
    value_sources: dict[str, int] = {}
    with_values = 0
    without_values = 0
    unknown = 0

    for row in rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
        component = row.get("component") or "general"
        by_component[component] = by_component.get(component, 0) + 1
        if row.get("llm_provider"):
            endpoints.add(f"{row['llm_provider']}:{row.get('llm_model') or '?'}")
        if row["status"] == "blocked":
            # A refused call sent nothing; counting it as "clean" would
            # inflate the headline with calls that never happened.
            continue
        manifest = _manifest_of(row)
        if "values_sent" not in manifest:
            unknown += 1
            continue
        tables.update(str(t) for t in manifest.get("tables") or [])
        columns.update(str(c) for c in manifest.get("columns") or [])
        if manifest.get("values_sent"):
            with_values += 1
            for source in manifest.get("value_sources") or []:
                value_sources[str(source)] = value_sources.get(str(source), 0) + 1
        else:
            without_values += 1

    sent = with_values + without_values

    # An empty log is not evidence that nothing happened. `enable_llm_audit`
    # is `INTEGER NOT NULL DEFAULT 0` (store/db.py:93), so on a workspace that
    # never turned it on there are no rows to find -- and this section used to
    # read that silence as "No model calls were made in this period." and put
    # it in front of an auditor. It is the one sentence in the pack that must
    # never be wrong, and on a default workspace it always was.
    #
    # Absence of a record and absence of a call are different claims, and the
    # pack may only make the one it can support.
    audit_enabled = _audit_is_enabled(account_id)
    if sent:
        statement = f"{without_values} of {sent} model calls carried no data value."
    elif audit_enabled:
        statement = "No model calls were made in this period."
    else:
        statement = (
            "No model calls are recorded for this period. Call auditing is not "
            "enabled for this workspace, so this is an absence of records and "
            "not evidence that no calls were made. Enable call auditing to make "
            "this section attestable."
        )

    return {
        "window_days": int(days),
        "total": len(rows),
        "call_audit_enabled": audit_enabled,
        "by_status": by_status,
        "by_component": by_component,
        "endpoints_used": sorted(endpoints),
        "tables_described": sorted(tables),
        "columns_described": sorted(columns),
        "values_sent": {
            "calls_carrying_a_data_value": with_values,
            "calls_carrying_no_data_value": without_values,
            "calls_with_no_manifest": unknown,
            "sources": value_sources,
        },
        "attestable": bool(audit_enabled),
        "statement": statement,
    }


def _audit_is_enabled(account_id: str) -> bool:
    """Is call auditing on for this workspace?

    Fails closed: if the flag cannot be read, the pack reports the section as
    not attestable rather than claiming a clean period it cannot evidence.
    """
    import store

    try:
        client = store.get_client(account_id) or {}
        return bool(client.get("enable_llm_audit"))
    except Exception:
        log.warning("Could not read enable_llm_audit for %s; reporting the "
                    "calls section as not attestable", account_id, exc_info=True)
        return False


def refusals_section(account_id: str, days: int, limit: int = 200) -> dict:
    """Calls the product declined to make, and why.

    The most persuasive section in the pack, and the one nothing else in this
    market produces: evidence that the boundary fires, taken from the
    ``record_llm_blocked`` rows the refusing call sites already write.
    """
    import store

    cutoff = _cutoff(days)
    with store.get_db() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT component, payload_preview_sanitized AS reason, created_at "
            "FROM llm_call_log WHERE account_id=? AND created_at >= ? "
            "AND status='blocked' ORDER BY created_at DESC LIMIT ?",
            (account_id, cutoff, int(limit)),
        ).fetchall()]
        total = conn.execute(
            "SELECT COUNT(*) AS n FROM llm_call_log "
            "WHERE account_id=? AND created_at >= ? AND status='blocked'",
            (account_id, cutoff),
        ).fetchone()["n"]

    by_component: dict[str, int] = {}
    for row in rows:
        component = row.get("component") or "general"
        by_component[component] = by_component.get(component, 0) + 1

    return {
        "total": int(total),
        "by_component": by_component,
        "listed": rows,
        # Truncation is stated, never silent: a pack that quietly shows 200 of
        # 4,000 refusals reads as a complete list.
        "truncated": int(total) > len(rows),
        "statement": f"{int(total)} model calls were refused and recorded.",
    }


def grounding_section(account_id: str) -> dict:
    """How complete the classification is, and what it suppressed.

    Value grounding for a regulated tenant is restricted to columns an admin
    has reviewed and the pack's pack tags do not mark sensitive
    (``core.value_resolver.filter_resolved_for_compliance``), degrading to
    full suppression when classification is incomplete. That degradation is
    the safe behaviour and it is also a gap, so the pack reports the coverage
    rather than only the outcome.
    """
    import store

    try:
        classifications = store.list_classifications(account_id)
    except Exception as exc:
        log.warning("Classification coverage unavailable for %s: %s", account_id, exc)
        classifications = []

    reviewed = sum(1 for c in classifications if c.get("reviewed"))
    tagged = sum(1 for c in classifications if c.get("tags"))
    by_strategy: dict[str, int] = {}
    for item in classifications:
        strategy = str(item.get("mask_strategy") or "none")
        by_strategy[strategy] = by_strategy.get(strategy, 0) + 1

    from core.compliance.policy_engine import is_regulated

    regulated = is_regulated(account_id)
    return {
        "regulated": regulated,
        "columns_classified": len(classifications),
        "columns_reviewed": reviewed,
        "columns_tagged_sensitive": tagged,
        "mask_strategies": by_strategy,
        # The claim this section supports, in the auditor's language.
        "statement": (
            "Verified filter values are injected only for columns an "
            "administrator has reviewed and that carry no sensitive tag; "
            "unreviewed columns are suppressed entirely."
            if regulated
            else "This workspace is not on the regulated boundary; verified "
                 "filter values are injected for any indexed column."
        ),
    }


def _json_field(raw, fallback):
    """A stored JSON column, back as the object the hash was taken over."""
    if raw in (None, ""):
        return fallback
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return fallback


def integrity_section(account_id: str) -> dict:
    """Does the hash-chained decision log still verify?

    Recomputed here rather than trusted: the chain is only evidence if
    somebody checks it, and the pack is where that check belongs. A break is
    reported with the record it starts at, so the gap is locatable rather than
    merely known about.
    """
    import store

    try:
        with store.get_db() as conn:
            # SELECT *, not the hash columns alone. The check below RECOMPUTES
            # each record's hash from its content, and it cannot do that
            # without the content: selecting only id/previous_hash/record_hash
            # is what made this section verify the links of the chain while
            # never verifying that any link still describes its record.
            rows = [dict(r) for r in conn.execute(
                "SELECT * FROM policy_decision_log WHERE account_id=? "
                "ORDER BY seq ASC, created_at ASC, id ASC",
                (account_id,),
            ).fetchall()]
    except Exception as exc:
        log.warning("Decision log unavailable for %s: %s", account_id, exc)
        return {
            "records": 0, "verified": False, "reason": "log_unavailable",
            "statement": "The decision log could not be read for verification.",
        }

    if not rows:
        return {
            "records": 0, "verified": True, "reason": "no_decisions_recorded",
            "statement": "No policy decisions were recorded for this workspace.",
        }

    # A fork is reported separately from a break. Two records claiming the
    # same predecessor is what a chain looks like when its ordering was
    # ambiguous rather than when somebody edited it, and telling an auditor
    # "tampered" for what is a historical ordering defect would be wrong.
    seen_predecessors: dict[str, str] = {}
    expected = ""
    for row in rows:
        previous = row.get("previous_hash") or ""
        if previous and previous in seen_predecessors:
            return {
                "records": len(rows),
                "verified": False,
                "reason": "chain_forked",
                "first_broken_record": row["id"],
                "first_broken_at": row.get("created_at", ""),
                "also_claimed_by": seen_predecessors[previous],
                "statement": (
                    f"Two of the decision log's {len(rows)} records claim the "
                    "same predecessor; ordering cannot be verified."
                ),
            }
        if previous != expected:
            return {
                "records": len(rows),
                "verified": False,
                "reason": "chain_broken",
                "first_broken_record": row["id"],
                "first_broken_at": row.get("created_at", ""),
                "statement": (
                    f"The decision log's hash chain does not verify: record "
                    f"{row['id']} does not follow the record before it."
                ),
            }

        # The link holds. Does the hash still DESCRIBE the record?
        #
        # This is the check the docstring above promised and the code did not
        # make. Editing a record's content -- flipping a denied PII export to
        # allowed, emptying its resource list -- leaves previous_hash and
        # record_hash untouched, so the chain still links perfectly and the
        # pack reported "an unbroken hash chain" over a rewritten log.
        # Recomputed through store.policy_decision_hash, the same function the
        # writer uses, so the two cannot drift into disagreeing.
        recomputed = store.policy_decision_hash(
            audit_id=row.get("id") or "",
            account_id=row.get("account_id") or "",
            user_id=row.get("user_id"),
            action=row.get("action") or "",
            purpose_id=row.get("purpose_id") or "",
            channel=row.get("channel") or "",
            allowed=row.get("allowed"),
            reason_code=row.get("reason_code") or "",
            resources=_json_field(row.get("resource_json"), []),
            obligations=_json_field(row.get("obligation_json"), {}),
            policy_version=row.get("policy_version"),
            previous_hash=previous,
        )
        if recomputed != (row.get("record_hash") or ""):
            return {
                "records": len(rows),
                "verified": False,
                "reason": "record_altered",
                "first_broken_record": row["id"],
                "first_broken_at": row.get("created_at", ""),
                "statement": (
                    f"The decision log's record {row['id']} no longer matches "
                    "its own hash: its content has been altered since it was "
                    "written."
                ),
            }

        seen_predecessors[previous] = row["id"]
        expected = row.get("record_hash") or ""

    return {
        "records": len(rows),
        "verified": True,
        "head": expected,
        "statement": (
            f"The decision log's {len(rows)} records form an unbroken hash chain."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# The pack
# ══════════════════════════════════════════════════════════════════════════════

def fingerprint(pack: dict) -> str:
    """A hash over the pack's content, excluding the fingerprint itself.

    Makes the document tamper-evident on its own terms: two people holding
    what should be the same pack can compare one string.
    """
    body = {k: v for k, v in pack.items() if k != "fingerprint"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_proof_pack(account_id: str, *, days: int = 30) -> dict[str, Any]:
    """The whole artefact for one workspace over one period.

    Every section is best-effort and reports its own failure rather than
    taking down the pack: an auditor is better served by four sections and a
    stated gap than by an exception. ``generated_at`` is the only field that
    is not derived from stored evidence.
    """
    sections: dict[str, Any] = {}
    for name, build in (
        ("egress", lambda: egress_section(account_id)),
        ("calls", lambda: calls_section(account_id, days)),
        ("refusals", lambda: refusals_section(account_id, days)),
        ("grounding", lambda: grounding_section(account_id)),
        ("integrity", lambda: integrity_section(account_id)),
    ):
        try:
            sections[name] = build()
        except Exception as exc:
            log.error("Proof-pack section %r failed for %s: %s", name, account_id, exc)
            sections[name] = {"error": str(exc), "available": False}

    pack = {
        "pack_version": PACK_VERSION,
        "account_id": account_id,
        "window_days": int(days),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **sections,
    }
    pack["fingerprint"] = fingerprint(pack)
    return pack


def summary_lines(pack: dict) -> list[str]:
    """The pack in five sentences, for the top of the page.

    Every line is a `statement` a section computed, so the summary cannot
    drift away from the evidence beneath it.
    """
    lines: list[str] = []
    for name in ("egress", "calls", "refusals", "grounding", "integrity"):
        section = pack.get(name) or {}
        statement = section.get("statement")
        if statement:
            lines.append(str(statement))
    return lines
