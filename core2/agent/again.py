"""Looking again: one more reading of a question whose first answer looks wrong when it is checked.

The AI writes a plan; code checks and answers it. Two answers are a sign the plan misread the question rather than
the data saying so:

* nothing found, and the condition that empties it is one the question never states (not in its words, not kept
  from the answer above, not a standing condition), while the same period has rows without it;
* a share of exactly 0% or 100%: none, or every one, of the rows counted meets its condition.

The AI is then asked once more, with its first plan and what the check found (member names only where they may
reach it). The second plan goes through every rule the first did. It is kept only when it puts the sign right --
rows found, a share that is neither 0% nor 100% -- and the answer says it looked again, with the first reading one
click away. Otherwise the first answer stands as it was, and says that it was checked. Never more than once.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from core2.model.schema import SemanticModel
from core2.plan.ir import Plan

KEY = "_look_again"           # on a first answer: (question, forced, plan, reasons), taken off before it is shown
FIRST_EN, FIRST_FR = "Show my first reading", "Voir ma première lecture"


@dataclass
class Reason:
    kind: str           # unstated | all_or_none
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
    """Does the second reading put the sign right: another plan, an answer with rows, and no share of 0% or
    100% where that was the sign?"""
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
    return True


# ── saying it ─────────────────────────────────────────────────────────────────


def say(second: dict[str, Any], first_payload: dict[str, Any], first: Plan, reasons: list[Reason], session: Any,
        plain: Callable[[str], str]) -> None:
    """The second answer says it looked again and why, with the first reading one click away (no AI)."""
    said = str((first_payload.get("answer") or {}).get("headline") or "").rstrip(".")
    seen = {"kind": "looked_again",
            "en": f"My first reading did not look right ({said}), so I read your question again.",
            "fr": f"Ma première lecture ne semblait pas juste ({said}) : j'ai relu votre question.",
            "figures": [], "days": [], "members": [], "about": ""}
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
