"""
core/mail/settings.py

How this server sends email, as a Super admin set it on Admin › System › Email (admin/email_routes.py).

Two ways, one at a time:

    graph   Microsoft 365, through Microsoft Graph: an app registration in the company's Entra ID with the
            Mail.Send application permission, sending as one mailbox (a shared one, say reports@). No SMTP
            port is opened and no mailbox password is kept: the app's client secret is.
    smtp    any mail server that takes SMTP: host, port, STARTTLS or TLS, and a user and password if it asks.

Every value is kept in system_config, encrypted like every other setting. ``allowed_domains`` limits where
email may go (a comma list; empty: anywhere): a message to any other address is recorded as skipped, never
sent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import store
from core.brand import PRODUCT_NAME

GRAPH, SMTP = "graph", "smtp"
PROVIDERS = (GRAPH, SMTP)
SECURITY = ("starttls", "ssl", "none")

# setting name -> system_config key
KEYS = {
    "provider": "email_provider",
    "from_address": "email_from_address",
    "from_name": "email_from_name",
    "reply_to": "email_reply_to",
    "allowed_domains": "email_allowed_domains",
    "graph_tenant_id": "email_graph_tenant_id",
    "graph_client_id": "email_graph_client_id",
    "graph_client_secret": "email_graph_client_secret",
    "smtp_host": "email_smtp_host",
    "smtp_port": "email_smtp_port",
    "smtp_security": "email_smtp_security",
    "smtp_username": "email_smtp_username",
    "smtp_password": "email_smtp_password",
}
SECRETS = ("graph_client_secret", "smtp_password")


@dataclass(frozen=True)
class MailSettings:
    provider: str = ""
    from_address: str = ""
    from_name: str = PRODUCT_NAME
    reply_to: str = ""
    allowed_domains: tuple[str, ...] = field(default_factory=tuple)
    graph_tenant_id: str = ""
    graph_client_id: str = ""
    graph_client_secret: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_security: str = "starttls"
    smtp_username: str = ""
    smtp_password: str = ""

    @property
    def configured(self) -> bool:
        """Set up enough to send."""
        if not self.from_address or "@" not in self.from_address:
            return False
        if self.provider == GRAPH:
            return bool(self.graph_tenant_id and self.graph_client_id and self.graph_client_secret)
        if self.provider == SMTP:
            return bool(self.smtp_host)
        return False

    def allows(self, address: str) -> bool:
        """Whether email may go to ``address``: its domain is one of the allowed ones (or any is allowed)."""
        address = str(address or "").strip().lower()
        if "@" not in address:
            return False
        if not self.allowed_domains:
            return True
        domain = address.rsplit("@", 1)[1]
        return any(domain == d or domain.endswith("." + d) for d in self.allowed_domains)

    def missing(self) -> list[str]:
        """What is still needed before email can be sent, in words for the settings page."""
        needed = []
        if self.provider not in PROVIDERS:
            return ["a way to send: Microsoft 365 or SMTP"]
        if not self.from_address or "@" not in self.from_address:
            needed.append("the address email comes from")
        if self.provider == GRAPH:
            needed += [label for value, label in ((self.graph_tenant_id, "the tenant (directory) ID"),
                                                  (self.graph_client_id, "the application (client) ID"),
                                                  (self.graph_client_secret, "the client secret")) if not value]
        elif not self.smtp_host:
            needed.append("the SMTP server")
        return needed


def _domains(text: str) -> tuple[str, ...]:
    parts = [p.strip().lower().lstrip("@").strip(".") for p in str(text or "").replace(";", ",").split(",")]
    return tuple(sorted({p for p in parts if p and "." in p}))


def load() -> MailSettings:
    raw = {name: str(store.get_system(key, "") or "") for name, key in KEYS.items()}
    try:
        port = int(raw["smtp_port"] or 0) or (465 if raw["smtp_security"] == "ssl" else 587)
    except ValueError:
        port = 587
    return MailSettings(
        provider=raw["provider"] if raw["provider"] in PROVIDERS else "",
        from_address=raw["from_address"].strip(), from_name=raw["from_name"].strip() or PRODUCT_NAME,
        reply_to=raw["reply_to"].strip(), allowed_domains=_domains(raw["allowed_domains"]),
        graph_tenant_id=raw["graph_tenant_id"].strip(), graph_client_id=raw["graph_client_id"].strip(),
        graph_client_secret=raw["graph_client_secret"], smtp_host=raw["smtp_host"].strip(), smtp_port=port,
        smtp_security=raw["smtp_security"] if raw["smtp_security"] in SECURITY else "starttls",
        smtp_username=raw["smtp_username"].strip(), smtp_password=raw["smtp_password"])


def save(values: dict[str, str]) -> None:
    """Keep the settings a Super admin entered. A secret left blank keeps the one saved before."""
    for name, key in KEYS.items():
        if name not in values:
            continue
        value = str(values[name] or "")
        if name in SECRETS and not value:
            continue
        if name == "allowed_domains":
            value = ", ".join(_domains(value))
        store.set_system(key, value.strip() if name not in SECRETS else value)


def configured() -> bool:
    return load().configured
