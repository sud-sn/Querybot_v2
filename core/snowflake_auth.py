"""How QueryBot signs in to Snowflake.

Snowflake is blocking single-factor password sign-in for users without MFA,
and a server cannot answer an MFA prompt. Snowflake's answer for applications
is a service user (``TYPE = SERVICE``): it has no password and signs in with a
key pair or a programmatic access token. A connection's ``auth_method`` says
which of these it uses; ``password`` stays for connections made before the
change, until Snowflake blocks them.

Every Snowflake connection goes through ``core.schema._sf_connect``, which asks
``connect_kwargs`` how to sign in, so one decision covers questions, discovery,
the connection test and the log export alike.

A private key is stored unencrypted inside the connection's credentials, which
are encrypted at rest: a passphrase stored next to its key would add nothing, so
an encrypted key is opened once, when it is saved.
"""
from __future__ import annotations

import base64
import hashlib
import re

KEYPAIR = "keypair"
TOKEN = "pat"
PASSWORD = "password"
AUTH_METHODS = (KEYPAIR, TOKEN, PASSWORD)

# What a connection needs besides account, user and warehouse, by sign-in method.
_SECRET_FIELD = {KEYPAIR: "private_key", TOKEN: "token", PASSWORD: "password"}
_COMMON_REQUIRED = ["account", "user", "warehouse"]

# Settings passed to the connector as they are; secrets are added per method.
_PLAIN_SETTINGS = (
    "account", "user", "warehouse", "database", "schema", "role",
    "login_timeout", "network_timeout", "client_session_keep_alive",
)

# Snowflake's query history shows this tag on every statement QueryBot runs.
QUERY_TAG = "QueryBot"

# Snowflake refuses RSA keys shorter than this for key-pair sign-in.
MIN_RSA_BITS = 2048

# The stored secrets of each sign-in method; a connection keeps only its own.
METHOD_SECRETS = {KEYPAIR: ("private_key",), TOKEN: ("token",), PASSWORD: ("password",)}
SECRET_FIELDS = frozenset({"password", "token", "private_key", "private_key_passphrase"})


class SnowflakeAuthError(ValueError):
    """The stored sign-in cannot be used; the message says how to fix it."""


def auth_method(creds: dict) -> str:
    """The connection's sign-in method. A connection saved before methods existed
    signs in with its password, as it always has."""
    method = str(creds.get("auth_method") or "").strip().lower()
    if method in AUTH_METHODS:
        return method
    if creds.get("private_key"):
        return KEYPAIR
    if creds.get("token"):
        return TOKEN
    return PASSWORD


def required_fields(creds: dict) -> list[str]:
    """The fields a Snowflake connection must have for its sign-in method."""
    return [*_COMMON_REQUIRED, _SECRET_FIELD[auth_method(creds)]]


def _load_private_key(pem: str, passphrase: str = ""):
    from cryptography.exceptions import UnsupportedAlgorithm
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    text = str(pem or "").strip()
    if "-----BEGIN" not in text:
        raise SnowflakeKeyErrorMessages.not_pem()
    password = passphrase.encode("utf-8") if passphrase else None
    try:
        key = serialization.load_pem_private_key(text.encode("utf-8"), password=password)
    except TypeError:
        # An encrypted key read without a passphrase, or a plain key given one.
        raise (SnowflakeKeyErrorMessages.needs_passphrase() if password is None
               else SnowflakeKeyErrorMessages.no_passphrase_expected())
    except UnsupportedAlgorithm:
        raise SnowflakeKeyErrorMessages.not_rsa()   # a key type Snowflake doesn't take either
    except ValueError:
        raise (SnowflakeKeyErrorMessages.wrong_passphrase() if password is not None
               else SnowflakeKeyErrorMessages.unreadable())
    if not isinstance(key, rsa.RSAPrivateKey):
        raise SnowflakeKeyErrorMessages.not_rsa()
    if key.key_size < MIN_RSA_BITS:
        raise SnowflakeKeyErrorMessages.too_short(key.key_size)
    return key


class SnowflakeKeyErrorMessages:
    """The sentences an admin sees when a private key cannot be used."""

    @staticmethod
    def not_pem() -> SnowflakeAuthError:
        return SnowflakeAuthError(
            "The private key must be in PEM format: the text that starts with "
            "-----BEGIN PRIVATE KEY----- or -----BEGIN ENCRYPTED PRIVATE KEY-----."
        )

    @staticmethod
    def needs_passphrase() -> SnowflakeAuthError:
        return SnowflakeAuthError("This private key is encrypted. Enter its passphrase.")

    @staticmethod
    def no_passphrase_expected() -> SnowflakeAuthError:
        return SnowflakeAuthError(
            "This private key is not encrypted, so it takes no passphrase. Clear the passphrase field."
        )

    @staticmethod
    def wrong_passphrase() -> SnowflakeAuthError:
        return SnowflakeAuthError("The passphrase does not open this private key.")

    @staticmethod
    def unreadable() -> SnowflakeAuthError:
        return SnowflakeAuthError(
            "The private key could not be read. Paste the whole key, including its BEGIN and END lines."
        )

    @staticmethod
    def not_rsa() -> SnowflakeAuthError:
        return SnowflakeAuthError(
            f"Snowflake key-pair sign-in needs an RSA key ({MIN_RSA_BITS} bits or more)."
        )

    @staticmethod
    def too_short(bits: int) -> SnowflakeAuthError:
        return SnowflakeAuthError(
            f"This RSA key has {bits} bits; Snowflake needs {MIN_RSA_BITS} bits or more. Generate a new key."
        )


def private_key_der(pem: str, passphrase: str = "") -> bytes:
    """The private key as unencrypted PKCS#8 DER bytes, the form the connector takes."""
    from cryptography.hazmat.primitives import serialization

    key = _load_private_key(pem, passphrase)
    return key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def unlocked_private_key_pem(pem: str, passphrase: str = "") -> str:
    """The key as unencrypted PKCS#8 PEM, the form QueryBot stores it in.

    The credentials it is stored with are encrypted at rest, so keeping the key
    encrypted under a passphrase stored beside it would protect nothing, and the
    passphrase would have to be kept in step with the key on every edit.
    """
    from cryptography.hazmat.primitives import serialization

    key = _load_private_key(pem, passphrase)
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")


def public_key_details(pem: str, passphrase: str = "") -> dict:
    """The public half of a stored private key, as the Snowflake admin needs it.

    ``public_key`` is the value for ``ALTER USER ... SET RSA_PUBLIC_KEY``
    (base64, no header lines); ``fingerprint`` matches the RSA_PUBLIC_KEY_FP
    that ``DESC USER`` shows, so the two can be compared.
    """
    from cryptography.hazmat.primitives import serialization

    key = _load_private_key(pem, passphrase)
    der = key.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return {
        "public_key": base64.b64encode(der).decode("ascii"),
        "fingerprint": "SHA256:" + base64.b64encode(hashlib.sha256(der).digest()).decode("ascii"),
    }


def generate_private_key_pem(bits: int = 2048) -> str:
    """A new RSA private key, unencrypted PKCS#8 PEM. It is stored with the
    connection's other credentials, which are encrypted at rest."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=bits)
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")


def alter_user_statement(user: str, public_key: str) -> str:
    """The statement a Snowflake admin runs to let the service user sign in with this key.

    Snowflake keeps two keys per user; a user that already signs in with a key
    takes a second one in RSA_PUBLIC_KEY_2, so it can be changed without downtime.
    """
    name = str(user or "").strip() or "<SERVICE_USER>"
    if not name.replace("_", "").replace("$", "").isalnum():
        name = '"' + name.replace('"', '""') + '"'
    return f"ALTER USER {name} SET RSA_PUBLIC_KEY='{public_key}';"


def keep_method_secrets(creds: dict) -> dict:
    """The credentials with only the secret of their own sign-in method. A
    connection that moved to a key pair no longer keeps the old password, and a
    stored key is already unlocked, so no passphrase is kept."""
    keep = set(METHOD_SECRETS[auth_method(creds)])
    return {k: v for k, v in creds.items() if k not in SECRET_FIELDS or k in keep}


def unusable_settings(cfg: dict, *, warehouse, database) -> str:
    """Snowflake accepts a sign-in whose warehouse or database this user and role
    cannot use: the session simply has none, and every question then fails.
    Say which, from what the session reports, or return ""."""
    missing = [f"{label} {cfg[key]}" for key, label, now in (
        ("warehouse", "warehouse", warehouse), ("database", "database", database),
    ) if cfg.get(key) and not now]
    if not missing:
        return ""
    return ("Signed in, but this user and role cannot use the " + " or the ".join(missing)
            + ". Grant USAGE on it to the role, or check the name.")


def connect_kwargs(cfg: dict) -> dict:
    """The arguments for ``snowflake.connector.connect`` for this connection.

    Only the secret of the connection's own sign-in method is passed: a key-pair
    connection never sends a stored password, and a password connection never
    sends a key.
    """
    kwargs = {key: cfg[key] for key in _PLAIN_SETTINGS if cfg.get(key)}
    method = auth_method(cfg)
    if method == KEYPAIR:
        kwargs["private_key"] = private_key_der(
            cfg.get("private_key") or "", cfg.get("private_key_passphrase") or "",
        )
        kwargs["authenticator"] = "SNOWFLAKE_JWT"
    elif method == TOKEN:
        token = str(cfg.get("token") or "").strip()
        if not token:
            raise SnowflakeAuthError("Enter the programmatic access token for the Snowflake user.")
        kwargs["authenticator"] = "PROGRAMMATIC_ACCESS_TOKEN"
        kwargs["token"] = token
    else:
        if cfg.get("password"):
            kwargs["password"] = cfg["password"]
        if cfg.get("authenticator"):
            kwargs["authenticator"] = cfg["authenticator"]
    kwargs["session_parameters"] = {"QUERY_TAG": QUERY_TAG}
    return kwargs


# The connector names the host it tried (account.snowflakecomputing.com:443);
# an account name such as "teamfast-prod" must not read as "MFA".
_HOST_RE = re.compile(r"\S+\.snowflakecomputing\.(?:com|cn)(?::\d+)?\S*", re.I)


def friendly_error(raw: str) -> str | None:
    """A plain explanation for the Snowflake sign-in errors an admin can fix, or None."""
    low = _HOST_RE.sub(" ", str(raw or "")).lower()
    if "jwt token is invalid" in low or re.search(r"\bjwt\b", low) and "invalid" in low:
        return (
            "Snowflake rejected the key. Check that the public key QueryBot shows is set on this user "
            "(ALTER USER ... SET RSA_PUBLIC_KEY), and that the account and user name are right."
        )
    if "programmatic access token" in low and ("invalid" in low or "expired" in low):
        return "Snowflake rejected the programmatic access token: it is wrong, expired or revoked."
    if re.search(r"multi-factor|\bmfa\b|\bduo\b", low):
        return (
            "Snowflake asks this user for multi-factor sign-in, which a server cannot answer. "
            "Connect with a service user (TYPE = SERVICE) and a key pair instead."
        )
    if "incorrect username or password" in low:
        return (
            "Snowflake did not accept the user name and password. If this user must use MFA, or is a "
            "service user, switch the sign-in method to Key pair."
        )
    return None
