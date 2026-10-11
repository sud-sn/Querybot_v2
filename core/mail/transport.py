"""
core/mail/transport.py

One message handed to the mail service: Microsoft Graph or an SMTP server (core/mail/settings.py).

``send`` returns the service's id for the message, or raises ``Permanent`` (it will never go: a refused
address, a sender the app may not send as, bad credentials) or ``Transient`` (worth another try later: the
service is down, busy, or throttling). The outbox (core/mail/outbox.py) retries the second kind only.
"""

from __future__ import annotations

import base64
import smtplib
import ssl
import threading
import time
import uuid
from dataclasses import dataclass, field
from email.message import EmailMessage
from email.utils import formataddr, make_msgid
from typing import Optional

from core.mail.settings import GRAPH, SMTP, MailSettings

TIMEOUT = 30
GRAPH_LOGIN = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
GRAPH_SEND = "https://graph.microsoft.com/v1.0/users/{sender}/sendMail"
# Graph takes attachments inline up to about 3 MB in all; bigger needs an upload session.
GRAPH_INLINE_BYTES = 3 * 1024 * 1024


class Permanent(Exception):
    """The message will never go as it is."""


class Transient(Exception):
    """Try again later."""


@dataclass
class Outgoing:
    to: str
    subject: str
    text: str
    html: str = ""
    attachments: list[tuple[str, str, bytes]] = field(default_factory=list)
    headers: dict[str, str] = field(default_factory=dict)


def send(settings: MailSettings, message: Outgoing, *, http=None) -> str:
    if not settings.configured:
        raise Permanent("Email is not set up on this server.")
    if settings.provider == GRAPH:
        return _send_graph(settings, message, http=http)
    if settings.provider == SMTP:
        return _send_smtp(settings, message)
    raise Permanent("Email is not set up on this server.")


# ── SMTP ─────────────────────────────────────────────────────────────────────

def _mime(settings: MailSettings, message: Outgoing) -> EmailMessage:
    mime = EmailMessage()
    mime["From"] = formataddr((settings.from_name, settings.from_address))
    mime["To"] = message.to
    mime["Subject"] = message.subject
    if settings.reply_to:
        mime["Reply-To"] = settings.reply_to
    mime["Message-ID"] = make_msgid(domain=settings.from_address.rsplit("@", 1)[-1])
    for name, value in message.headers.items():
        mime[name] = value
    mime.set_content(message.text)
    if message.html:
        mime.add_alternative(message.html, subtype="html")
    for filename, content_type, data in message.attachments:
        maintype, _, subtype = (content_type or "application/octet-stream").partition("/")
        mime.add_attachment(data, maintype=maintype, subtype=subtype or "octet-stream", filename=filename)
    return mime


def _send_smtp(settings: MailSettings, message: Outgoing) -> str:
    mime = _mime(settings, message)
    context = ssl.create_default_context()
    try:
        if settings.smtp_security == "ssl":
            server = smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, timeout=TIMEOUT, context=context)
        else:
            server = smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=TIMEOUT)
        with server:
            server.ehlo()
            if settings.smtp_security == "starttls":
                server.starttls(context=context)
                server.ehlo()
            if settings.smtp_username:
                server.login(settings.smtp_username, settings.smtp_password)
            refused = server.send_message(mime)
    except smtplib.SMTPAuthenticationError as exc:
        raise Permanent(f"The mail server refused the user and password ({exc.smtp_code}).") from exc
    except smtplib.SMTPRecipientsRefused as exc:
        codes = [code for code, _ in exc.recipients.values()]
        if any(int(code) >= 500 for code in codes):
            raise Permanent(f"The mail server refused the address {message.to} ({codes[0]}).") from exc
        raise Transient(f"The mail server put off the address {message.to} ({codes[0]}).") from exc
    except smtplib.SMTPSenderRefused as exc:
        raise Permanent(f"The mail server refused to send as {settings.from_address} ({exc.smtp_code}).") from exc
    except smtplib.SMTPResponseException as exc:
        if int(exc.smtp_code) >= 500:
            raise Permanent(f"The mail server refused the message ({exc.smtp_code}).") from exc
        raise Transient(f"The mail server put the message off ({exc.smtp_code}).") from exc
    except smtplib.SMTPNotSupportedError as exc:
        raise Permanent("The mail server does not offer STARTTLS: choose TLS or no encryption.") from exc
    except ssl.SSLError as exc:
        raise Permanent(f"The secure connection to the mail server failed: {exc.reason or exc}.") from exc
    except (OSError, smtplib.SMTPException) as exc:
        raise Transient(f"The mail server could not be reached: {exc}.") from exc
    if refused:
        raise Permanent(f"The mail server refused the address {message.to}.")
    return str(mime["Message-ID"] or "")


# ── Microsoft Graph ──────────────────────────────────────────────────────────

_tokens: dict[tuple[str, str], tuple[str, float]] = {}
_tokens_lock = threading.Lock()


def _client(http):
    if http is not None:
        return http, False
    import httpx

    return httpx.Client(timeout=TIMEOUT), True


def _token(settings: MailSettings, http) -> str:
    key = (settings.graph_tenant_id, settings.graph_client_id)
    with _tokens_lock:
        held = _tokens.get(key)
        if held and held[1] > time.time() + 60:
            return held[0]
    try:
        response = http.post(GRAPH_LOGIN.format(tenant=settings.graph_tenant_id), data={
            "client_id": settings.graph_client_id, "client_secret": settings.graph_client_secret,
            "scope": "https://graph.microsoft.com/.default", "grant_type": "client_credentials"})
    except Exception as exc:  # noqa: BLE001 - the network: worth another try
        raise Transient(f"Microsoft sign-in could not be reached: {exc}.") from exc
    if response.status_code >= 500 or response.status_code == 429:
        raise Transient(f"Microsoft sign-in is not answering ({response.status_code}).")
    body = _json(response)
    if response.status_code != 200 or not body.get("access_token"):
        detail = str(body.get("error_description") or body.get("error") or response.status_code).splitlines()[0]
        raise Permanent(f"Microsoft refused the app's sign-in: {detail[:200]}")
    token = str(body["access_token"])
    with _tokens_lock:
        _tokens[key] = (token, time.time() + int(body.get("expires_in") or 3600))
    return token


def _json(response) -> dict:
    try:
        body = response.json()
    except Exception:  # noqa: BLE001
        return {}
    return body if isinstance(body, dict) else {}


def _graph_body(settings: MailSettings, message: Outgoing) -> dict:
    content = {"contentType": "HTML", "content": message.html} if message.html else \
        {"contentType": "Text", "content": message.text}
    mail = {
        "subject": message.subject,
        "body": content,
        "toRecipients": [{"emailAddress": {"address": message.to}}],
        "from": {"emailAddress": {"address": settings.from_address, "name": settings.from_name}},
    }
    if settings.reply_to:
        mail["replyTo"] = [{"emailAddress": {"address": settings.reply_to}}]
    if message.headers:
        # Graph takes only custom x- headers; List-Unsubscribe is carried as x-list-unsubscribe.
        mail["internetMessageHeaders"] = [{"name": f"x-{name.lower()}" if not name.lower().startswith("x-")
                                           else name, "value": value} for name, value in message.headers.items()]
    if message.attachments:
        if sum(len(data) for _, _, data in message.attachments) > GRAPH_INLINE_BYTES:
            raise Permanent("The attachments are larger than Microsoft Graph takes in one message (3 MB).")
        mail["attachments"] = [{"@odata.type": "#microsoft.graph.fileAttachment", "name": filename,
                                "contentType": content_type, "contentBytes": base64.b64encode(data).decode()}
                               for filename, content_type, data in message.attachments]
    return {"message": mail, "saveToSentItems": False}


def _send_graph(settings: MailSettings, message: Outgoing, *, http=None) -> str:
    client, owned = _client(http)
    try:
        token = _token(settings, client)
        request_id = str(uuid.uuid4())
        try:
            response = client.post(GRAPH_SEND.format(sender=settings.from_address), json=_graph_body(settings, message),
                                   headers={"Authorization": f"Bearer {token}", "client-request-id": request_id})
        except Exception as exc:  # noqa: BLE001
            raise Transient(f"Microsoft Graph could not be reached: {exc}.") from exc
        if response.status_code in (200, 202):
            return response.headers.get("request-id") or request_id
        body = _json(response).get("error") or {}
        detail = f"{body.get('code') or response.status_code}: {str(body.get('message') or '')[:200]}".strip(": ")
        if response.status_code == 401:
            with _tokens_lock:
                _tokens.pop((settings.graph_tenant_id, settings.graph_client_id), None)
            raise Transient(f"Microsoft Graph asked to sign in again ({detail}).")
        if response.status_code == 429 or response.status_code >= 500:
            raise Transient(f"Microsoft Graph is busy ({detail}).")
        if response.status_code == 403:
            raise Permanent(f"The app may not send as {settings.from_address}: give it Mail.Send ({detail}).")
        raise Permanent(f"Microsoft Graph refused the message ({detail}).")
    finally:
        if owned:
            client.close()


def forget_tokens() -> None:
    with _tokens_lock:
        _tokens.clear()


def describe(settings: Optional[MailSettings]) -> str:
    if settings is None or not settings.provider:
        return "not set up"
    if settings.provider == GRAPH:
        return f"Microsoft 365 as {settings.from_address}"
    return f"SMTP {settings.smtp_host}:{settings.smtp_port} as {settings.from_address}"
