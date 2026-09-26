"""
admin/credentials.py

The admin console's one password: whether it has been set, setting the first
one exactly once, checking it, and setting a new one from the server.

First-run setup used to be decided by ``get_system("admin_password_hash")``
coming back empty. That reads a row that cannot be decrypted as empty too, and
the setup form's POST did not ask at all, so anyone who could reach the server
could submit it and replace the admin password -- and a server whose
encryption key had changed reopened setup to everyone. Now the question is
whether a row exists, and the first password is stored by one INSERT that the
database lets only one request win.

When the password is lost, or cannot be read because the key changed, the way
back is ``python -m admin.reset_password`` on the server: only someone with a
shell there can use it.
"""

from __future__ import annotations

import hashlib
import hmac

import store

ADMIN_PASSWORD_KEY = "admin_password_hash"
MIN_LENGTH = 8


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def is_set() -> bool:
    """A password has been stored, whether or not it can be read now."""
    return store.system_key_is_set(ADMIN_PASSWORD_KEY)


def is_readable() -> bool:
    """The stored password can be decrypted (False when none is stored)."""
    return bool(store.get_system(ADMIN_PASSWORD_KEY, ""))


def claim_first(password: str) -> bool:
    """Store the first admin password. False when one already exists."""
    return store.claim_system_key(ADMIN_PASSWORD_KEY, hash_password(password))


def verify(password: str) -> bool:
    stored = store.get_system(ADMIN_PASSWORD_KEY, "")
    return bool(stored) and hmac.compare_digest(hash_password(password), stored)


def set_password(password: str) -> None:
    store.set_system(ADMIN_PASSWORD_KEY, hash_password(password))
