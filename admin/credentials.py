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

Every admin session cookie carries the session version it was issued under.
Setting a password -- from the System page or from the server -- bumps it, so
a password change ends every admin session but the one that made it, which
gets a new cookie.
"""

from __future__ import annotations

import store
from store import passwords

ADMIN_PASSWORD_KEY = "admin_password_hash"
SESSION_VERSION_KEY = "admin_session_version"
MIN_LENGTH = 8


def hash_password(password: str) -> str:
    # Salted PBKDF2 (store/passwords.py). It was unsalted SHA-256, compared
    # with !=; that form still signs in once and is replaced (upgrade_hash).
    return passwords.hash_password(password)


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
    return passwords.verify_password(store.get_system(ADMIN_PASSWORD_KEY, ""), password)


def upgrade_hash(password: str) -> bool:
    """After a successful sign-in, replace an outdated hash of the same password.

    Not a password change: the version stays, so no admin session ends.
    """
    stored = store.get_system(ADMIN_PASSWORD_KEY, "")
    if not passwords.needs_rehash(stored) or not passwords.verify_password(stored, password):
        return False
    store.set_system(ADMIN_PASSWORD_KEY, hash_password(password))
    return True


def set_password(password: str) -> None:
    """Set the admin password, and end every admin session issued before."""
    store.set_system(ADMIN_PASSWORD_KEY, hash_password(password))
    store.set_system(SESSION_VERSION_KEY, str(session_version() + 1))


def session_version() -> int:
    try:
        return int(store.get_system(SESSION_VERSION_KEY, "") or 1)
    except ValueError:
        return 1
