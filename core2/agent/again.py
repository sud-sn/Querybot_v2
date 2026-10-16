"""Looking again: one more reading of a question whose first answer looks wrong when it is checked.

The AI writes a plan; code checks and answers it. Two answers are a sign the plan misread the question rather than
the data saying so:

* nothing found, and the condition that empties it is one the question never states (not in its words, not kept
  from the answer above, not a standing condition), while the same period has rows without it;
* a share of exactly 0% or 100%: none, or every one, of the rows counted meets its condition;
* a rate at exactly 100% in every row of a breakdown: each group's rows all meet its condition, so it tells the
  groups nothing apart ("what % of all returns each reason is", read as each reason's own rate);
* "how much" answered with a count of rows alone ("How much did we invoice?" answered with the number of invoices);
* two conditions on one field that no row can meet together (filled and empty, two different members).

The AI is then asked once more, with its first plan and what the check found (member names only where they may
reach it). The second plan goes through every rule the first did. It is kept only when it puts the sign right --
rows found, a share that is neither 0% nor 100%, a rate that tells the groups apart -- and the answer says it looked again, with the first reading one
click away. Otherwise the first answer stands as it was, and says that it was checked. Never more than once.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from core2.model.schema import SemanticModel
from core2.plan.ir import Plan

KEY = "_look_again"           # on a first answer: (question, forced, plan, reasons), taken off before it is shown
FIRST_EN, FIRST_FR = "Show my first reading", "Voir ma première lecture"


@dataclass
class Reason:
    kind: str           # unstated | all_or_none | every_row | left_out | amount | neither
    en: str             # for the AI (and the reader, said plainly)
    fr: str
    field: str = ""


# ── is it worth a second look? ───────────────────────────────────────────────


def suspect(question: str, plan: Plan, payload: dict[str, Any], looked: Any, *, kept: list, empty: bool,
            model: SemanticModel, values: bool) -> list[Reason]:
    """The signs that the plan misread the question, from its answer and what was seen when it was looked at."""
    if plan.kind != "query" or payload.get("data") is None:
        return []
    out: list[Reason] = []
    if empty and looked:
        said = question.casefold()
        held = {(f.field, tuple(str(v) for v in f.values)) for f in kept}
        elsewhere = {s.about for s in looked.seen if s.kind == "elsewhere" and s.about}
        for f in plan.filters:
            if f.op not in ("eq", "in") or not f.values or (f.field, tuple(str(v) for v in f.values)) in held:
                continue
            if any(str(v).casefold() in said for v in f.values) or f.field not in elsewhere:
                continue
            name = model.attributes[f.field].business_name if f.field in model.attributes else f.field
            what = f"{name} is {', '.join(str(v) for v in f.values)}" if values else f"the condition on {name}"
            out.append(Reason(
                "unstated",
                f"Nothing matched. The question does not say {what}; without that condition there are rows in the "
                "same period.",
                f"Rien ne correspondait. La question ne dit pas « {what} » ; sans cette condition, la même période a "
                "des lignes.", f.field))
    kpi = payload.get("kpi") if isinstance(payload.get("kpi"), dict) else None
    rows = (payload.get("data") or {}).get("rows") or []
    if kpi is not None and len(rows) == 1 and kpi.get("format") in ("percent", "percentage"):
        value = kpi.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value in (0, 100):
            none = value == 0
            out.append(Reason(
                "all_or_none",
                f"The figure came out at exactly {value:g}%: {'none' if none else 'every one'} of the rows counted "
                "meets its condition. Check that the condition is the one the question means (rows a table leaves "
                "out by default, such as cancelled ones, are counted only with include_left_out).",
                f"Le chiffre vaut exactement {value:g} % : aucune des lignes comptées ne remplit sa condition."
                if none else
                f"Le chiffre vaut exactement {value:g} % : toutes les lignes comptées remplissent sa condition."))
    if _AMOUNT_ASKED.search(question) and not _NOT_AN_AMOUNT.search(question) and _counts_only(plan, model):
        out.append(Reason(
            "amount",
            "The question asks how much: an amount (money, or units), and the plan counts rows alone. Give the amount "
            "the question means: the sales or invoice amount, counted by the date of the event it names when it "
            "names one (invoiced: the invoice date; shipped: the ship date; paid or collected: the payment date).",
            "La question demande combien en montant, et le plan ne fait que compter des lignes."))
    for field in _never_together(plan.filters):
        name = model.attributes[field].business_name if field in model.attributes else \
            model.entities[field].business_name if field in model.entities else field
        out.append(Reason(
            "neither",
            f"The plan has two conditions on {name} that no row can meet both of. To set rows with {name} against "
            f"rows without it (or one member against another), show each side: group by {name}, or a figure for each "
            "side, never both conditions at once.",
            f"Le plan pose deux conditions sur {name} qu'aucune ligne ne remplit à la fois.", field))
    for name in _of_every_row([r.name for r in plan.rates] if plan.group_by else [], payload):
        out.append(Reason(
            "every_row",
            f"{name} came out at exactly 100% in every one of the {len(rows)} rows: every row counted in each group "
            "meets its condition, so it tells the groups nothing apart. If the question asks what share of the "
            "whole each one makes, that is each one's share of the total (intent \"share\"), never a rate.",
            f"{name} vaut exactement 100 % dans chacune des {len(rows)} lignes : toutes les lignes comptées dans "
            "chaque groupe remplissent sa condition."))
    return out


_AMOUNT_ASKED = re.compile(r"\bhow\s+much\b", re.IGNORECASE)
_NOT_AN_AMOUNT = re.compile(r"\bhow\s+much\s+(?:time|longer|later|earlier|sooner)\b", re.IGNORECASE)


def _counts_only(plan: Plan, model: SemanticModel) -> bool:
    """Does the plan give a count of rows and nothing else (no amount, no figure worked out, no analysis)?"""
    if not plan.measures or plan.derived or plan.figures or plan.rates or plan.durations or plan.shaped():
        return False
    from core2.resolve.resolver import find_slug

    found = [find_slug(model, slug) for slug in plan.measures]
    return all(f is not None and f[0] == "measure" and getattr(f[1], "format", "") == "count" for f in found)


def _never_together(filters: list) -> list[str]:
    """The fields the conditions of a plan ask two things of that no row can be: filled and empty, or two
    different members ("is A" and "is B")."""
    out = []
    by_field: dict[str, list] = {}
    for f in filters:
        by_field.setdefault(f.field, []).append(f)
    for field, fs in by_field.items():
        ops = {f.op for f in fs}
        members = [{str(v).casefold() for v in f.values} for f in fs if f.op in ("eq", "in") and f.values]
        if {"is_null", "not_null"} <= ops or (len(members) > 1 and not set.intersection(*members)) or \
                ("is_null" in ops and members):
            out.append(field)
    return out


def _of_every_row(rates: list[str], payload: dict[str, Any]) -> list[str]:
    """The rates (by name) at exactly 100% in every row of an answer with more than one row."""
    data = payload.get("data") or {}
    rows = data.get("rows") or []
    if len(rows) < 2:
        return []

    def plain(text: object) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(text or "").casefold())

    def whole(value: object) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and abs(float(value) - 100) < 1e-6

    out = []
    for name in rates:
        column = next((h for h in data.get("headers") or [] if plain(h) == plain(name)), None)
        if column is not None and all(whole(r.get(column)) for r in rows):
            out.append(name)
    return out


def feedback(first: Plan, reasons: list[Reason], *, values: bool) -> str:
    """What the AI is told when it is asked once more: its first plan (member names only where they may reach
    it) and what the check of its answer found."""
    plan = first.model_dump(mode="json", exclude_defaults=True)
    if not values:
        for f in [*plan.get("filters", []), *(w for r in plan.get("rates", []) for w in r.get("where", []))]:
            if f.get("values"):
                f["values"] = ["<value>" for _ in f["values"]]
    found = "\n".join(f"- {r.en}" for r in reasons)
    return ("A FIRST PLAN FOR THIS QUESTION WAS ANSWERED, AND THE ANSWER WAS CHECKED BEFORE BEING SHOWN.\n"
            f"First plan: {json.dumps(plan, sort_keys=True)}\nWhat the check found:\n{found}\n"
            "If the first plan read the question wrongly, return the plan that answers it as asked. If it read it "
            "rightly, return the same plan.")


def better(second: dict[str, Any], first: Plan, payload: dict[str, Any], reasons: list[Reason]) -> bool:
    """Does the second reading put the sign right: another plan, an answer with rows, no share of 0% or 100%
    and no rate of every row where that was the sign?"""
    plan = second.get("plan")
    if second.get("data") is None or not plan or second.get("unsupported") or second.get("clarify"):
        return False
    if plan == first.model_dump(mode="json", exclude_defaults=True):
        return False
    rows = (second.get("data") or {}).get("rows") or []
    if any(r.kind == "unstated" for r in reasons):
        if not rows or all(all(v is None or v == 0 for v in row.values()) for row in rows):
            return False
    if any(r.kind == "all_or_none" for r in reasons):
        kpi = second.get("kpi") if isinstance(second.get("kpi"), dict) else {}
        if kpi.get("value") in (0, 100, None):
            return False
    if any(r.kind == "every_row" for r in reasons):
        rates = [str(r.get("name") or "") for r in plan.get("rates") or []] if plan.get("group_by") else []
        if _of_every_row(rates, second):
            return False
    if any(r.kind == "amount" for r in reasons) and plan.get("measures") == first.model_dump(
            mode="json", exclude_defaults=True).get("measures"):
        return False                        # still the count it gave the first time
    if any(r.kind == "neither" for r in reasons):
        from core2.plan.ir import Filter

        if not rows or _never_together([Filter.model_validate(f) for f in plan.get("filters") or []]):
            return False
    if any(r.kind == "left_out" for r in reasons):
        # Parts of the request left out (core2/agent/parts.py): the second reading must leave out fewer.
        if len(second.get("parts_left_out") or []) >= len(payload.get("parts_left_out") or []):
            return False
    return True


# ── saying it ─────────────────────────────────────────────────────────────────


def say(second: dict[str, Any], first_payload: dict[str, Any], first: Plan, reasons: list[Reason], session: Any,
        plain: Callable[[str], str]) -> None:
    """The second answer says it looked again and why, with the first reading one click away (no AI)."""
    from core2.answer.summary import names_stated

    said = str((first_payload.get("answer") or {}).get("headline") or "").rstrip(".")
    # The first answer's sentence is quoted: the names it states are the reply's (kept from the AI where need be).
    seen = {"kind": "looked_again",
            "en": f"My first reading did not look right ({said}), so I read your question again.",
            "fr": f"Ma première lecture ne semblait pas juste ({said}) : j'ai relu votre question.",
            "figures": [], "days": [], "members": names_stated(first_payload), "about": ""}
    if all(r.kind == "left_out" for r in reasons):
        # Only parts of the request were missing: said by the reader's own words for them, never the first answer.
        left = first_payload.get("parts_left_out") or []
        lost = [str(p.get("words")) for p in left]
        en, fr = ", ".join(lost), ", ".join(f"« {w} »" for w in lost)
        # The reader's words may hold a member's name (a condition's, or one inside a figure's words): each counts
        # as one stated, kept from the AI where need be.
        seen.update({"en": f"My first reading left out {en}, so I read your question again.",
                     "fr": f"Ma première lecture laissait de côté {fr} : j'ai relu votre question.",
                     "members": lost})
    checked = [{"en": f"Checked the first reading: {r.en}", "fr": f"Vérifié la première lecture : {r.fr}"}
               for r in reasons]
    offer = {"label": FIRST_EN, "question": FIRST_EN, "label_fr": FIRST_FR, "question_fr": FIRST_FR, "kind": "look"}
    reply = second.get("reply")
    if isinstance(reply, dict) and reply.get("en"):
        reply.update({"en": f"{seen['en']} {reply['en']}", "fr": f"{seen['fr']} {reply.get('fr') or ''}".strip(),
                      "seen": [seen, *(reply.get("seen") or [])], "checked": [*checked, *(reply.get("checked") or [])],
                      "offers": [*(reply.get("offers") or []), offer]})
    else:
        second["reply"] = {"en": seen["en"], "fr": seen["fr"], "by": "facts", "seen": [seen], "checked": checked,
                           "offers": [offer], "nothing": False}
    second["follow_up_suggestions"] = [offer, *(second.get("follow_up_suggestions") or [])][:6]
    for words in (FIRST_EN, FIRST_FR):
        session.offers[plain(words)] = first
    trust = second.setdefault("trust", {})
    trust["looked_again"] = {"first_plan": first.model_dump(mode="json", exclude_defaults=True),
                             "why": [r.en for r in reasons]}


def unchanged(payload: dict[str, Any], reasons: list[Reason]) -> None:
    """The first answer stands: what was checked says the second reading changed nothing."""
    reply = payload.get("reply")
    line = {"en": "Read the question again with the AI: the same reading.",
            "fr": "Relu la question avec l'IA : même lecture."}
    if isinstance(reply, dict):
        reply["checked"] = [*(reply.get("checked") or []), line]
    payload.setdefault("trust", {})["looked_again"] = {"why": [r.en for r in reasons], "kept": "first"}


# ── the conversation's state, kept for a reading that is not kept ───────────


def state(session: Any) -> tuple:
    return list(session.turns), session.pending, dict(session.offers), session.yes


def restore(session: Any, kept: tuple) -> None:
    session.turns, session.pending, session.offers, session.yes = list(kept[0]), kept[1], dict(kept[2]), kept[3]
