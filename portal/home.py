"""The reader's Home page: a welcome, what to do next, the business's headline figures and their dashboards.

What it shows, and where each part comes from:

- The welcome: the reader's first name, their workspace, how far the data goes (the latest date Learn found in
  a subject they may see, never past today) and when Learn last ran.
- New questions: when Learn has run since the reader last signed in, the questions to try that the Data guide
  lists (core2/answer/describe.py), as this reader may ask them.
- Headline figures: the executive dashboard's number tiles (core2/recipes.py executive), each run as a pinned
  answer is, as this reader may see the data (core2.service.portal_replay applies their tables, row rules and
  masking): the value, its change on the period before and its trend. Drawn after the page, by its own request,
  so a slow warehouse never holds the page; kept a few minutes for the same reader under the same rules.
- Dashboards: the reader's own and those shared with them, with a miniature of each one's layout.

Only a workspace the new core answers has figures and new questions; any other gets the welcome, the actions
and its dashboards.
"""

from __future__ import annotations

import contextvars
import datetime as dt
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import store

log = logging.getLogger(__name__)

MOST_FIGURES = 4
MOST_BOARDS = 6
MOST_DRAWN = 60       # cards on the dashboards page drawn in miniature; any more are their name alone
MOST_QUESTIONS = 3
FIGURES_KEPT = 300          # seconds a reader's figures are kept, at most

_KEPT: dict[tuple, tuple[float, list[dict[str, Any]]]] = {}
_KEPT_LOCK = threading.Lock()


def first_name(user: dict) -> str:
    """The reader's first name, as the welcome says it; "" when they have no name on file."""
    name = " ".join(str(user.get("name") or user.get("display_name") or "").split())
    return name.split(" ")[0] if name else ""


def _answers(account_id: str) -> bool:
    try:
        return store.get_query_engine(account_id) == "core2"
    except Exception:  # noqa: BLE001 - no switch table: today's pipeline
        return False


def _model(user: dict, client: dict):
    """The model the reader's questions are answered with, and their tables as its keys (None: all); (None, None)
    when the new core does not answer this workspace or has not learned it."""
    account_id = str(user["account_id"])
    if not _answers(account_id):
        return None, None
    from core2.service import _allowed_model_tables, current_model

    model = current_model(account_id, client)
    if model is None:
        return None, None
    return model, _allowed_model_tables(model, store.get_allowed_tables(user))


def _data_to(model, allowed: set[str] | None, today: dt.date) -> dt.date | None:
    """The latest date the data reaches in a subject the reader may see: each table's leading date, as the Data
    guide's overview counts it. A budget set into next year is not data yet, so nothing past today counts."""
    leading: dict[str, Any] = {}
    for role in sorted(model.date_roles.values(), key=lambda r: (not r.is_default, r.name.casefold())):
        if role.kind != "audit" and (allowed is None or role.table in allowed):
            leading.setdefault(role.table, role)
    ends = [r.last for r in leading.values() if r.last and r.first and r.last <= today]
    return max(ends, default=None)


def welcome(user: dict, client: dict, *, today: dt.date) -> dict[str, Any]:
    """{"name", "workspace", "data_to": date | None, "learned": datetime | None}."""
    out: dict[str, Any] = {"name": first_name(user), "workspace": str(client.get("client_name") or ""),
                           "data_to": None, "learned": None}
    try:
        model, allowed = _model(user, client)
    except Exception as exc:  # noqa: BLE001 - the welcome stands without what Learn found
        log.warning("Home: the model could not be read for %s: %s", user.get("account_id"), exc)
        return out
    if model is not None:
        out["data_to"] = _data_to(model, allowed, today)
        out["learned"] = model.built_at
    return out


def _last_visit(user: dict) -> dt.datetime | None:
    """When the reader signed in before this visit: the sign-in before the latest (this one)."""
    try:
        from store import profile_store

        seen = profile_store.recent_sign_ins(int(user["id"]), 2)
    except Exception:  # noqa: BLE001 - no record: every finding is new to them
        return None
    if len(seen) < 2:
        return None
    try:
        return dt.datetime.fromisoformat(str(seen[1].get("at") or "").replace("Z", ""))
    except ValueError:
        return None


def news(user: dict, client: dict, *, today: dt.date) -> dict[str, Any] | None:
    """The questions Learn's latest run lets this reader ask, when it ran since they last signed in:
    {"questions": [...], "learned": datetime, "version": int}; None when there is nothing new for them, or they
    may not ask."""
    from core.portal_roles import can

    if not (client.get("chat_ui_enabled") and can(user, "ask")):
        return None
    try:
        model, allowed = _model(user, client)
        if model is None or model.built_at is None:
            return None
        before = _last_visit(user)
        learned = model.built_at.replace(tzinfo=None)
        if before is not None and learned <= before:
            return None
        import json

        from core.value_index import value_index_enabled
        from core2.answer.describe import describe
        from core2.service import question_scrubber

        state = json.loads(client.get("state_data") or "{}")
        described = describe(model, [], today=today, allowed=allowed,
                             values=question_scrubber(str(user["account_id"])) is None and value_index_enabled(state))
    except Exception as exc:  # noqa: BLE001 - the page stands without the banner
        log.warning("Home: new questions could not be read for %s: %s", user.get("account_id"), exc)
        return None
    questions = [q for q in described.examples if q][:MOST_QUESTIONS]
    if not questions:
        return None
    return {"questions": questions, "learned": model.built_at, "version": int(model.version or 0)}


# ── Dashboards ───────────────────────────────────────────────────────────────

def _layout(dashboard_id: int, user: dict) -> list[dict[str, Any]]:
    """The dashboard's first tab, as boxes in a 12-column grid: {"x", "y", "w", "h", "kind"} ("number", "line",
    "bars" or "table"), for its card's miniature. No data is read."""
    try:
        charts = store.list_dashboard_charts_for_view(int(dashboard_id), int(user["id"]), str(user["account_id"]))
    except Exception:  # noqa: BLE001 - a card without its miniature
        return []
    tabs = [str(c.get("dashboard_tab") or "") for c in charts]
    first = tabs[0] if tabs else ""
    boxes = []
    for c in charts:
        if str(c.get("dashboard_tab") or "") != first:
            continue
        kind = str(c.get("chart_type") or "")
        shape = ("number" if kind == "kpi" else "table" if kind == "table"
                 else "line" if kind in ("line", "area") else "bars")
        boxes.append({"x": int(c.get("grid_x") or 0), "y": int(c.get("grid_y") or 0),
                      "w": max(1, int(c.get("grid_w") or 4)), "h": max(1, int(c.get("grid_h") or 3)), "kind": shape})
    return boxes[:24]


def layouts(found: list[dict], user: dict, *, most: int = MOST_DRAWN) -> dict[int, list[dict[str, Any]]]:
    """The first tab of each of these dashboards in miniature (_layout), by id, for the first `most` of them."""
    return {int(board["id"]): _layout(int(board["id"]), user) for board in found[:most]}


def boards(user: dict) -> list[dict[str, Any]]:
    """The reader's dashboards, most recently changed first, theirs before those shared with them:
    {"id", "name", "tiles", "mine", "owner", "shared", "layout"}."""
    try:
        found = store.list_dashboards(str(user["account_id"]), int(user["id"]))
    except Exception as exc:  # noqa: BLE001 - the page stands without its dashboards
        log.warning("Home: dashboards could not be listed for user %s: %s", user.get("id"), exc)
        return []
    out = []
    for board in found[:MOST_BOARDS]:
        out.append({"id": int(board["id"]), "name": str(board.get("name") or ""),
                    "tiles": int(board.get("chart_count") or 0), "mine": bool(board.get("can_edit")),
                    "owner": str(board.get("owner_name") or ""),
                    "shared": str(board.get("visibility") or "") == "team",
                    "layout": _layout(int(board["id"]), user)})
    return out


# ── Headline figures ─────────────────────────────────────────────────────────

def _kept_key(user: dict, model, allowed: set[str] | None, lang: str, today: dt.date) -> tuple:
    account_id = str(user["account_id"])
    try:
        policy = int(store.get_compliance_profile(account_id).get("active_policy_version") or 0)
    except Exception:  # noqa: BLE001 - no compliance profile: one version
        policy = 0
    # Row rules change without a new policy version, and a rule may read the reader's own values: both are part
    # of what these figures were counted under, or a changed rule waited out the figures kept before it.
    try:
        rules = store.row_rules_stamp(account_id)
    except Exception:  # noqa: BLE001 - when in doubt, a key never kept before
        rules = dt.datetime.now().isoformat()
    own = json.dumps(store.user_attributes(user), sort_keys=True)
    return (account_id, int(user["id"]), str(user.get("portal_role") or ""), str(user.get("role") or ""),
            int(user.get("session_version") or 0),
            int(model.version or 0), str(model.built_at or ""), tuple(sorted(allowed)) if allowed is not None else None,
            policy, rules, own, json.dumps(sorted([g["id"], g["name"]] for g in user.get("groups") or [])), lang,
            today.isoformat())


def _kept_for(user: dict) -> int:
    """How long this reader's figures may be kept: a few minutes, never past access of theirs that ends by
    itself (an attestation's term, an emergency grant)."""
    from core.dashboard_refresh import _kept_for as kept

    try:
        return kept(str(user["account_id"]), user, FIGURES_KEPT)
    except Exception:  # noqa: BLE001 - when in doubt, nothing is kept
        return 0


def _figure(made, payload: dict[str, Any], lang: str) -> dict[str, Any] | None:
    kpi = payload.get("kpi") or {}
    if kpi.get("value") is None:
        return None
    from core.response_builder import _format_number
    from portal.routes import _kpi_compact

    badges = (payload.get("answer") or {}).get("badges") or []
    period = next((str((lang == "fr" and b.get("text_fr")) or b.get("text") or "") for b in badges
                   if b.get("kind") == "period"), "")
    return {"title": str(made.title), "question": str(made.question), "period": period,
            "value": _kpi_compact(kpi.get("value"), kpi.get("format")),
            "full": _format_number(kpi.get("value"), kpi.get("format")),
            "change": kpi.get("change") or None, "trend": kpi.get("trend") or None}


def figures(user: dict, client: dict, *, today: dt.date, lang: str = "en") -> list[dict[str, Any]]:
    """The business's headline figures as this reader may see them: up to four of
    {"title", "question", "period", "value", "full", "change", "trend"}. [] when the new core does not answer the
    workspace, has not learned it, or the reader's tables hold no figure to lead with."""
    model, allowed = _model(user, client)
    if model is None:
        return []
    key = _kept_key(user, model, allowed, lang, today)
    now = time.monotonic()
    with _KEPT_LOCK:
        hit = _KEPT.get(key)
        if hit and hit[0] > now:
            return hit[1]
    from core2 import recipes
    from core2.kpi_proposals import industry_of
    from core2.service import portal_replay

    found = recipes.executive(model, today=today, language=lang, allowed=allowed,
                              industry=industry_of(str(user["account_id"])))
    if not found:
        return []
    _, tiles = found
    wanted = [m for m in tiles if m.kind == "number"][:MOST_FIGURES]

    def run(made):
        try:
            return made, portal_replay(str(user["account_id"]), made.plan, user, question=made.question)
        except Exception as exc:  # noqa: BLE001 - one figure fails, the others stand
            log.warning("Home: a headline figure failed for %s: %s", user.get("account_id"), exc)
            return made, None
    # Side by side, as a dashboard's tiles are; each in a copy of this request's context, so the reader's
    # language goes with it to its thread.
    contexts = [contextvars.copy_context() for _ in wanted]
    with ThreadPoolExecutor(max_workers=max(1, len(wanted)), thread_name_prefix="home-figure") as pool:
        done = list(pool.map(lambda pair: pair[0].run(run, pair[1]), zip(contexts, wanted)))
    out, released = [], False
    for made, payload in done:
        if payload is None:
            continue
        released = released or bool(payload.get("released"))
        figure = _figure(made, payload, lang)
        if figure:
            out.append(figure)
    # People's data released to a cleared reader is logged at each release: never served again from here.
    ttl = 0 if released else _kept_for(user)
    if ttl:
        with _KEPT_LOCK:
            for stale in [k for k, (until, _) in _KEPT.items() if until <= now]:
                _KEPT.pop(stale, None)
            _KEPT[key] = (now + ttl, out)
    return out


def forget() -> None:
    """Every kept figure goes (tests; an admin's change takes the next request anyway, through the key)."""
    with _KEPT_LOCK:
        _KEPT.clear()
