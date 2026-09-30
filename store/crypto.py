"""
store/crypto.py

All credential encryption lives here.
Uses Fernet symmetric encryption (AES-128-CBC + HMAC-SHA256).

Key file: ~/.querybot_key  (outside project directory, chmod 600)
Override with env var: QUERYBOT_KEY_FILE=/custom/path
"""

import json
import logging
import os
from pathlib import Path

log = logging.getLogger("querybot.crypto")

KEY_FILE = Path(os.getenv("QUERYBOT_KEY_FILE", Path.home() / ".querybot_key"))


def key_file_problem() -> str | None:
    """Why this server's key file cannot be used, or None when it can:
    "missing" when there is none, "unreadable" when the service may not read
    it -- a key directory another user owns, say."""
    try:
        KEY_FILE.read_bytes()
    except FileNotFoundError:
        return "missing"
    except OSError:
        return "unreadable"
    return None


def _get_fernet(*, create: bool = False):
    try:
        from cryptography.fernet import Fernet
    except ImportError:
        raise RuntimeError("Run: pip install cryptography")

    # Only a save writes a key. When a read wrote one, a key file lost in an
    # upgrade was replaced by the first page that listed a saved credential,
    # with a key none of them can be read under, and every check after that
    # said the key had changed rather than that it had gone.
    if create and not KEY_FILE.exists():
        key = Fernet.generate_key()
        KEY_FILE.write_bytes(key)
        KEY_FILE.chmod(0o600)
        log.info("Generated encryption key at %s — back this file up.", KEY_FILE)
    return Fernet(KEY_FILE.read_bytes().strip())


def encrypt(value: "dict | str") -> str:
    """Encrypt a dict or string. Returns a base64 Fernet token safe for DB storage."""
    raw = json.dumps(value).encode() if isinstance(value, dict) else str(value).encode()
    return _get_fernet(create=True).encrypt(raw).decode()


def decrypt(token: str) -> str:
    """Decrypt a Fernet token back to a plain string."""
    return _get_fernet().decrypt(token.encode()).decode()


def decrypt_json(token: str) -> dict:
    """Decrypt and JSON-parse a previously encrypt()ed dict."""
    return json.loads(decrypt(token))


def mask(value: str, show: int = 4) -> str:
    """Return a masked string for display. e.g. 'sk-ant-abc123' → '•••••••••123'"""
    if not value:
        return ""
    visible = value[-show:] if len(value) > show else value
    return "•" * (len(value) - len(visible)) + visible
