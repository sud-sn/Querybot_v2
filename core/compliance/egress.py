"""Where a tenant's questions are allowed to go — declared, and enforced.

The product's strongest governance claim is the one it cannot currently make:
*nothing leaves your network*. This module is that claim's enforcement point.

Three postures, from most permissive to least:

``cloud``
    A hosted model API. Metadata only — table and column names, the question
    text, and whatever ``core.value_resolver.filter_resolved_for_compliance``
    allows through. This is the default and what every existing tenant has.

``private``
    A hosted model under zero-retention terms and pinned to a region.
    Identical to ``cloud`` in what is sent; different in what the provider may
    do with it, and in which endpoints are permitted.

``airgapped``
    A model running on the tenant's own hardware. Nothing leaves. Embeddings
    are already local (``core.vector_store`` runs SentenceTransformer
    in-process), so with a local completion provider this posture is the whole
    picture rather than a partial one.

**One predicate.** The reason ``core.compliance.policy_engine.is_regulated``
exists is that the codebase once asked "is this tenant regulated?" four
different ways and one of them failed open. There is exactly one way to ask
where a tenant's data may go.

**What "fails closed" means here, and where it stops.** An unrecognised
posture value and a store that will not answer both resolve to ``airgapped``,
the posture that permits least. An *undeclared* posture does not, and the
distinction is deliberate.

``is_regulated`` treats an unprovisioned tenant as regulated, and that is
right there because the strict answer *filters*: the workspace still answers
questions, with less narration. The strict answer here *blocks* — an
air-gapped workspace with no local model configured cannot generate SQL at
all. A newly created client has no compliance_profile row until an admin
completes setup (``_backfill_compliance_profiles`` covers clients that
existed at upgrade, not ones created afterwards), so applying the strict
answer to an undeclared posture would mean every new workspace is dead on
arrival. An outage is not a safe default.

So an undeclared posture reads as ``cloud`` — the behaviour every tenant has
today — and ``describe()`` reports ``declared: False`` so the gap is visible
in readiness rather than silent. What is absolute is the other direction: a
tenant that has *declared* ``airgapped`` can never reach a hosted endpoint,
whatever a caller passes.
"""

from __future__ import annotations

import logging

log = logging.getLogger("querybot.compliance.egress")

CLOUD = "cloud"
PRIVATE = "private"
AIRGAPPED = "airgapped"

POSTURES = (CLOUD, PRIVATE, AIRGAPPED)

# What an undeclared posture reads as: the behaviour every tenant has today.
DEFAULT_POSTURE = CLOUD
# What a broken lookup or an unrecognised stored value reads as.
FAIL_CLOSED_POSTURE = AIRGAPPED

# Which providers each posture permits. `local` is the only provider that
# keeps every byte inside the tenant's network, so it is the only one an
# air-gapped workspace may use.
_PERMITTED_PROVIDERS: dict[str, frozenset[str]] = {
    CLOUD: frozenset({"anthropic", "openai", "azure_openai", "local"}),
    PRIVATE: frozenset({"azure_openai", "anthropic", "local"}),
    AIRGAPPED: frozenset({"local"}),
}


def normalise_posture(value) -> str:
    """A stored value mapped onto a posture, failing closed on anything else."""
    tag = str(value or "").strip().lower()
    if tag in POSTURES:
        return tag
    if tag:
        log.warning("Unrecognised egress posture %r — treating as %s",
                    value, FAIL_CLOSED_POSTURE)
        return FAIL_CLOSED_POSTURE
    return ""


def posture_is_declared(account_id: str) -> bool:
    """Has anyone actually chosen this workspace's posture?

    Separate from ``egress_posture`` because "nobody has decided" and "someone
    decided cloud" are different facts, and only the first belongs in a
    readiness report as something to go and do.
    """
    import store

    try:
        if not store.compliance_profile_exists(account_id):
            return False
        profile = store.get_compliance_profile(account_id) or {}
    except Exception as exc:
        log.error("Egress posture lookup failed for %s: %s", account_id, exc)
        return False
    return bool(normalise_posture(profile.get("egress_posture")))


def egress_posture(account_id: str) -> str:
    """The single answer to "where may this tenant's questions go?".

    Undeclared reads as ``cloud``; a broken lookup or an unrecognised stored
    value reads as ``airgapped``. See this module's docstring for why those
    two cases differ — the short version is that the strict answer here
    blocks rather than filters, and a new workspace must not be born dead.
    """
    import store

    try:
        if not store.compliance_profile_exists(account_id):
            log.info(
                "No compliance profile for account %s — egress posture "
                "undeclared, reading as %s. Complete compliance setup to "
                "declare one.", account_id, DEFAULT_POSTURE,
            )
            return DEFAULT_POSTURE
        profile = store.get_compliance_profile(account_id) or {}
    except Exception as exc:
        log.error("Egress posture lookup failed for %s (%s) — using %s",
                  account_id, exc, FAIL_CLOSED_POSTURE)
        return FAIL_CLOSED_POSTURE

    stored = normalise_posture(profile.get("egress_posture"))
    return stored or DEFAULT_POSTURE


def provider_allowed(account_id: str, provider: str) -> tuple[bool, str]:
    """May this tenant's questions be completed by this provider?

    Returns ``(allowed, reason)``; ``reason`` is empty when allowed and is
    what goes into the ``record_llm_blocked`` proof row when it is not.
    """
    posture = egress_posture(account_id)
    permitted = _PERMITTED_PROVIDERS.get(posture, _PERMITTED_PROVIDERS[AIRGAPPED])
    name = str(provider or "").strip().lower()
    if name in permitted:
        return True, ""
    return False, (
        f"egress posture {posture!r} does not permit provider {name or 'unset'!r}; "
        f"permitted: {', '.join(sorted(permitted))}"
    )


def describe(account_id: str) -> dict:
    """The posture, for the UI and for the proof pack.

    ``external_egress`` is the line an auditor reads: for ``airgapped`` it is
    the assertion that no request left the tenant's network at all.
    """
    posture = egress_posture(account_id)
    return {
        "posture": posture,
        "declared": posture_is_declared(account_id),
        "permitted_providers": sorted(_PERMITTED_PROVIDERS[posture]),
        "external_egress": posture != AIRGAPPED,
        "embeddings_local": True,   # core.vector_store, always in-process
    }
