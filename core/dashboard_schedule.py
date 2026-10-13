"""
core/dashboard_schedule.py

Dashboards emailed on a schedule.

A dashboard's owner, when they are a Creator, schedules it from the dashboard's ⋯ menu: every day, every weekday,
weekly on a day, or monthly on a date or the last day, at a time in a time zone; to people and groups of the
workspace, and to themselves; every tab or some; under the filters it is shown with when they schedule it; with
the PDF or without; with a subject and a note of their own. Saving shares the dashboard with whoever it goes to.

core/notification_scheduler.py runs ``run_due_schedules`` every minute. Each person is sent the dashboard as they
are allowed to see it (core/dashboard_snapshot.py): their tables, row rules, masking and language. The email
carries the dashboard itself -- numbers and tables as text, charts as pictures -- with the PDF attached, through
the outbox (core/mail/outbox.py).

    once            a run is claimed for the time it was due before anything is sent (store/schedule_store.py), and
                    every email of it has its own dedupe key: never twice, even after a restart mid-run
    late            a server that was down sends the run it missed once, late, then keeps to the schedule
    skipped         someone who can no longer open the dashboard, has no email address, or stopped this email; the
                    run's detail names how many and why
    paused          the owner can no longer send it (gone, no longer a Creator) or the dashboard is gone: the schedule
                    stops and says why, on the dashboard's schedule list
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import re
from typing import Optional
from urllib.parse import urlencode

import store
from store import schedule_store

log = logging.getLogger("querybot.dashboard_schedule")

STALE_MINUTES = 30                   # a run still "running" this long after it started was cut short
MAX_EMAIL_BYTES = 20 * 1024 * 1024   # the PDF is left out of an email that would be bigger than this
_TIME = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def _stamp(moment: dt.datetime) -> str:
    return moment.strftime("%Y-%m-%d %H:%M:%S")


def _parse(stamp) -> Optional[dt.datetime]:
    try:
        return dt.datetime.fromisoformat(str(stamp or "")[:19].replace("T", " "))
    except ValueError:
        return None


def _zone(name: str):
    from zoneinfo import ZoneInfo

    try:
        return ZoneInfo(str(name or "UTC"))
    except Exception:  # noqa: BLE001 - a zone this server does not know
        return ZoneInfo("UTC")


def known_zone(name: str) -> bool:
    from zoneinfo import ZoneInfo

    try:
        ZoneInfo(str(name or ""))
        return bool(name)
    except Exception:  # noqa: BLE001
        return False


# ── When ─────────────────────────────────────────────────────────────────────

def _matches(schedule: dict, day: dt.date) -> bool:
    cadence = schedule.get("cadence")
    if cadence == "daily":
        return True
    if cadence == "weekdays":
        return day.weekday() < 5
    if cadence == "weekly":
        return day.weekday() == int(schedule.get("weekday") or 0)
    if cadence == "monthly":
        wanted = int(schedule.get("month_day") or 0)
        if wanted <= 0:                                   # the last day of the month
            return (day + dt.timedelta(days=1)).month != day.month
        return day.day == wanted
    return False


def next_run(schedule: dict, after: dt.datetime) -> Optional[dt.datetime]:
    """The first time after ``after`` (UTC) the schedule is due, in UTC. Its time is wall-clock time in its zone:
    08:00 in London is 08:00 in summer and in winter. A time a clock change skips is sent an hour later that day."""
    found = _TIME.match(str(schedule.get("send_time") or ""))
    if not found:
        return None
    hour, minute = int(found.group(1)), int(found.group(2))
    zone = _zone(schedule.get("timezone"))
    local_after = after.replace(tzinfo=dt.timezone.utc).astimezone(zone)
    for offset in range(0, 400):
        day = local_after.date() + dt.timedelta(days=offset)
        if not _matches(schedule, day):
            continue
        local = dt.datetime(day.year, day.month, day.day, hour, minute, tzinfo=zone)
        moment = local.astimezone(dt.timezone.utc).replace(tzinfo=None)
        if moment > after:
            return moment
    return None


def describe(schedule: dict, lang: str = "en") -> str:
    """"Every Monday at 08:00 (Europe/London)", in the reader's language."""
    from core.i18n import t

    cadence = str(schedule.get("cadence") or "weekly")
    keys = {"weekday": t(f"reply.weekday.{int(schedule.get('weekday') or 0)}", lang=lang),
            "day": int(schedule.get("month_day") or 0), "time": schedule.get("send_time") or "",
            "zone": schedule.get("timezone") or "UTC"}
    if cadence == "monthly" and int(schedule.get("month_day") or 0) <= 0:
        return t("schedule.when.monthly_last", lang=lang, **keys)
    return t(f"schedule.when.{cadence}", lang=lang, **keys)


# ── Saving one ───────────────────────────────────────────────────────────────

def read_form(payload: dict, dashboard: dict, owner: dict) -> dict | str:
    """The schedule fields from what the dialog sent, checked: a dict for store/schedule_store.py, or what is
    wrong, as a message id (schedule.error.*)."""
    cadence = str(payload.get("cadence") or "")
    if cadence not in schedule_store.CADENCES:
        return "schedule.error.cadence"
    send_time = str(payload.get("send_time") or "")
    if not _TIME.match(send_time):
        return "schedule.error.time"
    zone = str(payload.get("timezone") or "")
    if not known_zone(zone):
        return "schedule.error.zone"
    try:
        weekday = int(payload.get("weekday") or 0)
        month_day = int(payload.get("month_day") or 0)
    except (TypeError, ValueError):
        return "schedule.error.day"
    if not 0 <= weekday <= 6 or not 0 <= month_day <= 28:
        return "schedule.error.day"
    try:
        own_tabs = [str(t) for t in json.loads(dashboard.get("tabs_json") or '["Overview"]')]
    except (TypeError, ValueError):
        own_tabs = ["Overview"]
    tabs = [str(t) for t in (payload.get("tabs") or []) if str(t) in own_tabs]
    from portal.dashboard_view import clean_view

    view = clean_view(payload.get("view"))
    recipients = []
    for item in payload.get("recipients") or []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type") or "")
        try:
            subject_id = int(item.get("id") or 0)
        except (TypeError, ValueError):
            continue
        if kind == "user":
            person = store.get_user(subject_id)
            if person and person.get("account_id") == owner["account_id"] and person.get("is_active"):
                recipients.append(("user", subject_id))
        elif kind == "group":
            group = store.get_group(subject_id)
            if group and group.get("account_id") == owner["account_id"]:
                recipients.append(("group", subject_id))
    recipients = list(dict.fromkeys(recipients))[:schedule_store.MAX_SUBJECTS]
    include_owner = bool(payload.get("include_owner", True))
    if not recipients and not include_owner:
        return "schedule.error.nobody"
    return {
        "cadence": cadence, "weekday": weekday, "month_day": month_day, "send_time": send_time, "timezone": zone,
        "tabs_json": json.dumps(tabs), "view_json": json.dumps(view),
        "attach_pdf": 1 if payload.get("attach_pdf", True) else 0, "include_owner": 1 if include_owner else 0,
        # One line: a subject is a mail header, where a line break is never allowed.
        "subject": " ".join(str(payload.get("subject") or "").split())[:200],
        "message": str(payload.get("message") or "").strip()[:1000],
        "recipients": recipients,
    }


def save(owner: dict, dashboard: dict, fields: dict, *, schedule_id: Optional[int] = None) -> int:
    """Keep a schedule (a new one, or ``schedule_id`` changed) and share the dashboard with everyone it names who
    cannot open it yet. Its next time is worked out from now."""
    recipients = fields.pop("recipients")
    upcoming = next_run({**fields}, _utc_now())
    stamp = _stamp(upcoming) if upcoming else None
    if schedule_id is None:
        schedule_id = schedule_store.create_schedule(owner["account_id"], int(dashboard["id"]), int(owner["id"]),
                                                     next_run_at=stamp, recipients=recipients, **fields)
    else:
        schedule_store.update_schedule(schedule_id, next_run_at=stamp, recipients=recipients, **fields)
        schedule_store.set_schedule_active(schedule_id, True)
    for subject_type, subject_id in recipients:
        try:
            store.add_dashboard_share(int(dashboard["id"]), int(owner["id"]), owner["account_id"], subject_type,
                                      int(subject_id))
        except Exception as exc:  # noqa: BLE001 - the schedule stands; whoever cannot open it is skipped and named
            log.warning("Schedule %s: could not share dashboard %s with %s %s: %s", schedule_id, dashboard.get("id"),
                        subject_type, subject_id, exc)
    return schedule_id


def resume(schedule: dict) -> None:
    upcoming = next_run(schedule, _utc_now())
    schedule_store.set_schedule_active(int(schedule["id"]), True, next_run_at=_stamp(upcoming) if upcoming else None)


# ── Sending ──────────────────────────────────────────────────────────────────

def run_due_schedules(now: Optional[dt.datetime] = None) -> int:
    """Send every schedule that is due, and finish any run a restart cut short. How many runs were sent."""
    now = now or _utc_now()
    sent = 0
    for run in schedule_store.stale_runs(_stamp(now - dt.timedelta(minutes=STALE_MINUTES))):
        schedule = schedule_store.get_schedule(int(run["schedule_id"]))
        if schedule and schedule_store.restart_run(int(run["id"])):
            log.info("Schedule %s: taking up the run due %s again", schedule["id"], run["due_at"])
            try:
                deliver(schedule, int(run["id"]), str(run["due_at"]), kind=str(run.get("kind") or "scheduled"))
            except Exception as exc:  # noqa: BLE001 - the schedules due now still go
                log.exception("Schedule %s: the run due %s could not be taken up", schedule["id"], run["due_at"])
                schedule_store.finish_run(int(run["id"]), status=schedule_store.FAILED, recipients=0, queued=0,
                                          skipped=0, detail=f"Could not be sent: {exc}")
    for schedule in schedule_store.due_schedules(_stamp(now)):
        due_at = str(schedule["next_run_at"])
        run_id = schedule_store.claim_run(int(schedule["id"]), due_at)
        if run_id is None:
            continue                                     # another scheduler has it
        upcoming = next_run(schedule, max(now, _parse(due_at) or now))
        schedule_store.advance_schedule(int(schedule["id"]), next_run_at=_stamp(upcoming) if upcoming else None,
                                        last_run_at=_stamp(now))
        try:
            deliver(schedule, run_id, due_at)
            sent += 1
        except Exception as exc:  # noqa: BLE001 - one schedule failing must not stop the others
            log.exception("Schedule %s could not be sent", schedule.get("id"))
            schedule_store.finish_run(run_id, status=schedule_store.FAILED, recipients=0, queued=0, skipped=0,
                                      detail=f"Could not be sent: {exc}")
    return sent


def send_now(schedule: dict, *, test: bool = False) -> dict:
    """Send a schedule at once: to everyone it names, or (``test``) to its owner alone, marked as a test. The run
    as it ended, with the outbox ids of its emails."""
    kind = "test" if test else "now"
    due_at = f"{kind}:{_stamp(_utc_now())}"
    run_id = schedule_store.claim_run(int(schedule["id"]), due_at, kind=kind)
    if run_id is None:
        return {"status": schedule_store.NOTHING, "detail": "Already sending."}
    done = deliver(schedule, run_id, due_at, kind=kind)
    return {**(schedule_store.get_run(run_id) or {}), "email_ids": done.get("email_ids") or []}


def _owner_problem(schedule: dict, owner: Optional[dict], board: Optional[dict]) -> str:
    from core.portal_roles import can

    if not owner or not owner.get("is_active") or owner.get("account_id") != schedule["account_id"]:
        return "Its owner can no longer sign in."
    if not can(owner, "reports"):
        return "Its owner is no longer a Creator, who may send scheduled dashboards."
    if not board:
        return "The dashboard was deleted."
    return ""


def deliver(schedule: dict, run_id: int, due_at: str, *, kind: str = "scheduled") -> dict:
    """Send one run of a schedule; record how it went. {"queued", "skipped", "status", "detail"}."""
    owner = store.get_user(int(schedule["owner_id"]))
    board = store.get_dashboard(int(schedule["dashboard_id"]), int(schedule["owner_id"]), schedule["account_id"])
    problem = _owner_problem(schedule, owner, board)
    if problem:
        schedule_store.set_schedule_active(int(schedule["id"]), False, reason=problem)
        schedule_store.finish_run(run_id, status=schedule_store.FAILED, recipients=0, queued=0, skipped=0,
                                  detail=f"Paused: {problem}")
        log.info("Schedule %s paused: %s", schedule["id"], problem)
        return {"status": schedule_store.FAILED, "detail": problem, "queued": 0, "skipped": 0}
    people = [owner] if kind == "test" else schedule_store.schedule_people(schedule)
    done = asyncio.run(_deliver_all(schedule, owner, people, due_at, test=kind == "test"))
    queued, skipped = done["queued"], done["skipped"]
    if not people:
        status = schedule_store.NOTHING
    elif queued == 0:
        status = schedule_store.FAILED                   # it reached nobody: the detail says why
    else:
        status = schedule_store.SENT if not skipped and not done["failed"] else schedule_store.PARTIAL
    detail = "; ".join(f"{count} {why}" for why, count in done["why"].items())
    schedule_store.finish_run(run_id, status=status, recipients=len(people), queued=queued, skipped=skipped,
                              detail=detail)
    return {"status": status, "detail": detail, "queued": queued, "skipped": skipped,
            "email_ids": done["email_ids"]}


async def _deliver_all(schedule: dict, owner: dict, people: list[dict], due_at: str, *, test: bool) -> dict:
    from core.dashboard_snapshot import Drawer
    from store import email_store

    done = {"queued": 0, "skipped": 0, "failed": 0, "why": {}, "email_ids": []}

    def skip(why: str) -> None:
        done["skipped"] += 1
        done["why"][why] = done["why"].get(why, 0) + 1

    async with Drawer() as drawer:
        for person in people:
            if not person.get("email") or "@" not in str(person.get("email")):
                skip("with no email address")
                continue
            board = store.get_dashboard_for_view(int(schedule["dashboard_id"]), int(person["id"]),
                                                 schedule["account_id"])
            if not board:
                skip("who can no longer open the dashboard")
                continue
            try:
                email_id = await _one(schedule, owner, board, person, due_at, drawer, test=test)
            except Exception as exc:  # noqa: BLE001 - one reader failing must not stop the others
                log.exception("Schedule %s: the email to user %s could not be made", schedule["id"], person.get("id"))
                done["failed"] += 1
                done["why"][f"failed ({exc})"[:120]] = done["why"].get(f"failed ({exc})"[:120], 0) + 1
                continue
            if email_id is None:
                skip("sent already")
                continue
            email = email_store.get_email(email_id)
            done["email_ids"].append(email_id)
            if email and email.get("status") == "skipped":
                skip(_skip_reason(str(email.get("last_error") or "")))
            else:
                done["queued"] += 1
    return done


def _skip_reason(reason: str) -> str:
    if "stopped" in reason:
        return "who stopped this email"
    if "outside the domains" in reason:
        return "outside the domains email may go to"
    if "not set up" in reason:
        return "because email is not set up"
    return "skipped"


def dashboard_link(base: str, schedule: dict) -> str:
    from portal.dashboard_view import address

    params = [("dashboard_id", str(int(schedule["dashboard_id"]))), *address(schedule.get("view"))]
    return f"{base}/portal/dashboard?{urlencode(params)}"


async def _one(schedule: dict, owner: dict, board: dict, person: dict, due_at: str, drawer, *, test: bool):
    """The email for one person, queued: its outbox id, or None when it was queued already."""
    from core.dashboard_snapshot import take
    from core.i18n import format_date, t
    from core.mail import links, outbox, render

    lang = str(person.get("lang") or "en")
    snapshot = await take(board, person, drawer=drawer, tabs=schedule.get("tabs") or None,
                          view=schedule.get("view") or {}, zone=str(schedule.get("timezone") or "UTC"),
                          pdf=bool(int(schedule.get("attach_pdf") or 0)))
    snapshot.problems = [t(p, lang=lang) if p.startswith("snapshot.") else p for p in snapshot.problems]
    base = outbox.portal_base()
    link = dashboard_link(base, schedule)
    local = _utc_now().replace(tzinfo=dt.timezone.utc).astimezone(_zone(schedule.get("timezone")))
    day = format_date(local, "day_month_short_year", lang=lang)
    subject = str(schedule.get("subject") or "").strip() or t("email.schedule.subject", lang=lang,
                                                              name=snapshot.name, date=day)
    if test:
        subject = t("email.schedule.test_subject", lang=lang, subject=subject)
    address = str(person["email"])
    ref = str(schedule["id"])
    stop = "" if test else links.unsubscribe_url(base, address, "schedule", ref)
    why = t("email.schedule.why", lang=lang, owner=owner.get("name") or owner.get("email") or "",
            name=snapshot.name, when=describe(schedule, lang).rstrip("."))
    files = [(f"chart-{tile.id}.png", "image/png", tile.image, render.tile_cid(tile))
             for tile in snapshot.charts() if tile.image]
    pdf_attached = False
    if snapshot.pdf:
        size = sum(len(f[2]) for f in files) + len(snapshot.pdf)
        if size <= MAX_EMAIL_BYTES:
            safe = re.sub(r"[^A-Za-z0-9 _.-]+", "", snapshot.name).strip() or "Dashboard"
            files.append((f"{safe} {local:%Y-%m-%d}.pdf", "application/pdf", snapshot.pdf))
            pdf_attached = True
        else:
            snapshot.problems.append(t("snapshot.pdf_too_large", lang=lang))
    note, note_from = str(schedule.get("message") or ""), str(owner.get("name") or owner.get("email") or "")
    html = render.dashboard(snapshot, link=link, title=subject, note=note, note_from=note_from,
                            pdf_attached=pdf_attached, test=test, why=why, unsubscribe=stop, lang=lang)
    text = render.dashboard_text(snapshot, link=link, note=note, note_from=note_from, pdf_attached=pdf_attached,
                                 test=test, why=why, unsubscribe=stop, lang=lang)
    return outbox.queue(kind="schedule", to=address, subject=subject, text=text, html=html,
                        account_id=str(schedule["account_id"]), user_id=int(person["id"]), ref=ref,
                        attachments=tuple(files), opt_out_kind=None if test else "schedule", unsubscribe_url=stop,
                        dedupe_key=None if test else f"schedule:{schedule['id']}:{due_at}:{person['id']}")
