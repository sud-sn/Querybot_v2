"""
core/process_secrets.py

A secret that must exist to sign sessions or compute PII pseudonyms, but that
this product has always let an operator skip setting -- so it needs a
fallback that does not become the vulnerability the secret exists to close.

Three call sites need one: admin session signing (admin/routes.py), portal
session signing (portal/routes.py), and the PII pseudonym HMAC key
(core/compliance/result_guard.py). All three used to fall back to a fixed,
literal string checked into this public repository --
"change-me-in-production", "querybot-development-pseudonym-secret". Because
the fallback was PUBLIC, "the operator forgot to set an env var" and "an
attacker who has read this file" ended in the exact same key: anyone could
forge an admin or portal session cookie, or recompute the HMAC that produces
a PII pseudonym and match it against candidate values, for any deployment
that missed one environment variable -- with nothing enforcing the variable
was ever set, only a startup log line.

The fix is not "crash if unset". A hard startup failure here would break
every existing dev/test deployment that has never needed to set these, for a
property (sessions and pseudonyms surviving a process restart) most of them
do not need either -- and this module has no way to know which deployments
that describes. Instead: generate a real random secret once, the first time
it's needed, and hold it in memory for the life of this process. An operator
who never sets the env var gets sessions and pseudonyms that are still
cryptographically real -- unguessable, and different every restart -- rather
than a known public string. Sessions signed before a restart become invalid
(a plain logout, not a compromise); pseudonyms recomputed after a restart no
longer match ones shown before it. Both are a strictly better failure than a
known secret with no expiry.
"""

from __future__ import annotations

import logging
import os
import secrets
import threading

log = logging.getLogger("querybot.process_secrets")

_lock = threading.Lock()
_generated: dict[str, str] = {}


def env_secret_or_random(*env_names: str, purpose: str) -> str:
    """The value of the first `env_names` entry that is set and non-empty, or
    a random secret generated once for `purpose` and cached for the life of
    this process.

    `purpose` is a short, fixed identifier ("admin_session",
    "portal_session", "pii_pseudonym") -- never a tenant- or user-supplied
    value. It keys the per-process cache (so two callers asking for the same
    purpose get the same generated secret within one process) and names
    which secret is missing in the warning log.

    Checked in the order given, matching each call site's existing
    precedence exactly -- this only replaces the LAST resort (a literal
    string), not the fallback chain an operator may already be relying on
    (e.g. a shared SESSION_SECRET covering both admin and portal).
    """
    for name in env_names:
        value = os.getenv(name)
        if value:
            return value
    with _lock:
        cached = _generated.get(purpose)
        if cached is None:
            cached = secrets.token_hex(32)
            _generated[purpose] = cached
            log.warning(
                "None of %s is set -- generated a random secret for %s. "
                "It will not survive a restart of this process: sessions "
                "signed with it are invalidated, and pseudonyms computed "
                "with it will not match ones shown before the restart. Set "
                "%s to a persistent secret before deploying to production.",
                ", ".join(env_names), purpose, env_names[0],
            )
        return cached
