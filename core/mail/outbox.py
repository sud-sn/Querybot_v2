"""
core/mail/outbox.py

Every email the product sends goes through here: queued, then sent by the scheduler (every minute, with
core/notification_scheduler.py), tried again after a pause when the mail service could not take it, and kept
in the delivery log (store/email_store.py) sent, failed or skipped, with the reason.

    queue(...)            a message for one address; recorded as skipped, never sent, when email is not set up,
                          the address is outside the allowed domains, or its owner stopped this kind of email
    deliver_due()         send what is due (the scheduler)
    send_now(id)          send one message at once (a test email; an invitation)

A failure the service may get over (it is down, busy, throttling) is tried again after 1, 5, 30, 120 and 360
minutes; then the message is failed. A refusal that will not change (a wrong address, a sender the app may not
send as, a wrong password) fails it at once. An admin can retry a failed message from the delivery log.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Optional

import store
from core.mail import settings as mail_settings
from core.mail import transport
from store import email_store

log = logging.getLogger("querybot.mail")

BACKOFF_MINUTES = (1, 5, 30, 120, 360)
BATCH = 20


def queue(*, kind: str, to: str, subject: str, text: str, html: str = "", account_id: str = "",
          user_id: Optional[int] = None, attachments: tuple = (), dedupe_key: Optional[str] = None,
          ref: str = "", opt_out_kind: Optional[str] = None, unsubscribe_url: str = "") -> Optional[int]:
    """A message in the outbox (its id), or None when ``dedupe_key`` was queued already.

    ``opt_out_kind``: the kind of email its owner can stop (a follow, a report); None for mail they cannot stop
    (an invitation, a test)."""
    config = mail_settings.load()
    reason = ""
    if not config.configured:
        reason = "Email is not set up on this server."
    elif not config.allows(to):
        reason = f"{to} is outside the domains email may go to."
    elif opt_out_kind and email_store.opted_out(to, opt_out_kind, ref):
        reason = f"{to} stopped this email."
    return email_store.queue_email(
        kind=kind, to_address=to, subject=subject, body_text=text, body_html=html, account_id=account_id,
        user_id=user_id, attachments=attachments, dedupe_key=dedupe_key, ref=ref, unsubscribe_url=unsubscribe_url,
        status=email_store.SKIPPED if reason else email_store.QUEUED, reason=reason)


def _headers(email: dict) -> dict[str, str]:
    unsubscribe = str(email.get("unsubscribe_url") or "")
    return {"List-Unsubscribe": f"<{unsubscribe}>"} if unsubscribe else {}


def _attempt(email: dict, config: mail_settings.MailSettings, *, http=None) -> str:
    """Send one claimed message; record what happened. Its status after."""
    email = email_store.get_email(int(email["id"])) or email          # as claimed: its attempts counted
    message = transport.Outgoing(to=email["to_address"], subject=email["subject"], text=email["body_text"],
                                 html=email.get("body_html") or "", attachments=email_store.email_attachments(email),
                                 headers=_headers(email))
    try:
        provider_id = transport.send(config, message, http=http)
    except transport.Permanent as exc:
        email_store.mark_failed(int(email["id"]), error=str(exc), provider=config.provider)
        log.warning("Email %s to %s failed: %s", email["id"], email["to_address"], exc)
        return email_store.FAILED
    except transport.Transient as exc:
        attempts = int(email.get("attempts") or 1)         # this one included: the claim counted it
        if attempts > len(BACKOFF_MINUTES):
            email_store.mark_failed(int(email["id"]), error=f"Gave up after {attempts - 1} tries: {exc}",
                                    provider=config.provider)
            return email_store.FAILED
        email_store.mark_retry(int(email["id"]), error=str(exc), provider=config.provider,
                               after=timedelta(minutes=BACKOFF_MINUTES[attempts - 1]))
        log.info("Email %s to %s put off: %s", email["id"], email["to_address"], exc)
        return email_store.QUEUED
    except Exception as exc:  # noqa: BLE001 - a fault of ours: logged, failed, never stuck in "sending"
        email_store.mark_failed(int(email["id"]), error=f"Could not be sent: {exc}", provider=config.provider)
        log.exception("Email %s could not be sent", email["id"])
        return email_store.FAILED
    email_store.mark_sent(int(email["id"]), provider=config.provider, provider_id=provider_id)
    return email_store.SENT


def deliver_due(*, http=None, limit: int = BATCH) -> dict[str, int]:
    """Send every message that is due, up to ``limit``. How many ended in each status."""
    done = {email_store.SENT: 0, email_store.QUEUED: 0, email_store.FAILED: 0}
    email_store.release_stuck_emails()
    due = email_store.due_emails(limit)
    if not due:
        return done
    config = mail_settings.load()
    for email in due:
        if not config.configured:
            break                                       # set up later: they wait in the queue
        if not email_store.claim_email(int(email["id"])):
            continue
        status = _attempt(email, config, http=http)
        done[status] = done.get(status, 0) + 1
    return done


def send_now(email_id: int, *, http=None) -> dict:
    """Send one queued message at once; the message after (its status and, if it failed, why)."""
    email = email_store.get_email(email_id)
    if email and email["status"] == email_store.QUEUED and email_store.claim_email(int(email_id)):
        _attempt(email, mail_settings.load(), http=http)
    return email_store.get_email(email_id) or {}


def portal_base() -> str:
    """The address readers open the portal at, for links in an email."""
    from core.pipeline_context import get_portal_base

    return get_portal_base()
