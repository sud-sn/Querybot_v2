"""
store/passwords.py

One way to hash and check a password, for portal users and the admin console.

PBKDF2-HMAC-SHA256 with a per-password salt, at the iteration count OWASP
recommends. Before this there were three: the admin password was unsalted
SHA-256 compared with !=, portal passwords were PBKDF2 at 200,000 iterations
with an unsalted-SHA-256 fallback that was never retired, and accounts approved
from Teams, Slack or Zoom all carried SHA-256("__platform_user__") -- a string
in the source -- which that fallback accepted as a password.

An older hash still signs in once, and the caller replaces it
(``needs_rehash``). An account that must never sign in with a password stores
``UNUSABLE``, which no password matches.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

# PBKDF2-HMAC-SHA256, OWASP Password Storage Cheat Sheet (2023).
ITERATIONS = 600_000

# Starts with a character a hash of any scheme here never does.
UNUSABLE = "!"

# What platform-approved accounts carried before UNUSABLE. Never a password,
# whatever the stored row says.
LEGACY_PLATFORM_PLACEHOLDER = hashlib.sha256(b"__platform_user__").hexdigest()

_PREFIX = "pbkdf2_sha256$"


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), ITERATIONS)
    return f"{_PREFIX}{ITERATIONS}${salt}${derived.hex()}"


def verify_password(stored: str | None, password: str) -> bool:
    if not stored or stored.startswith(UNUSABLE) or stored == LEGACY_PLATFORM_PLACEHOLDER:
        return False
    if stored.startswith(_PREFIX):
        try:
            _, rounds, salt, expected = stored.split("$", 3)
            derived = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), int(rounds))
            return hmac.compare_digest(expected, derived.hex())
        except Exception:
            return False
    # Unsalted SHA-256, from before salted hashes: accepted, then replaced.
    legacy = hashlib.sha256(password.encode()).hexdigest()
    return hmac.compare_digest(stored, legacy)


def needs_rehash(stored: str | None) -> bool:
    """A hash that verified but should be replaced: unsalted, or too few rounds."""
    if not stored or stored.startswith(UNUSABLE):
        return False
    if not stored.startswith(_PREFIX):
        return True
    try:
        return int(stored.split("$")[1]) < ITERATIONS
    except (IndexError, ValueError):
        return True
