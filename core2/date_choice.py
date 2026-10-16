"""The date an answer counted by, the other dates of the same rows, and putting the metric's own date right.

Every figure over time is counted by one date of its rows: an order's date, its ship date, the day a row was
written. Learn picks one for each table and each metric. When that pick is not the reader's, the answer's date badge
lists the rows' other dates, each a click that counts the same figure by it again, with no AI ("By ship date
instead"). The pick holds for the rest of the conversation (a standing condition, core2/standing.py). An answer
counted by a date other than its metric's own offers to make it the metric's date for everyone: an admin's click
changes it at once, a reader's is a suggestion on the admin's Requests page (core2/request_service.py).
"""

from __future__ import annotations

from typing import Any

from core2.model.schema import DateRole, SemanticModel
from core2.plan.ir import Plan
from core2.standing import counted_on

MOST = 3          # other dates offered on one answer


def instead(role: DateRole) -> tuple[str, str]:
    """A date's choice on the badge, in English and in French: what a click sends."""
    name = role.name.lower()
    return f"By {name} instead", f"Selon {name} à la place"


def choices(plan: Plan, logical: Any, model: SemanticModel) -> dict[str, Any] | None:
    """The date the answer was counted by and the other dates its rows have, each with the words a click sends;
    and, when the date is not its metric's own, the change that would make it so. None when the answer was
    counted by no date, or by more than one."""
    if plan.kind != "query" or (plan.time.window.kind == "all" and plan.time.grain is None
                                and plan.time.compare is None):
        return None
    used = {p.date.role.key: p.date.role for p in getattr(logical, "parts", []) if p.date is not None}
    if len(used) != 1:
        return None
    role = next(iter(used.values()))
    if counted_on(plan, model) != {role.table}:
        return None
    others = sorted((r for r in model.date_roles_of(role.table)
                     if r.key != role.key and r.kind != "audit" and r.name and r.name.lower() != role.name.lower()),
                    key=lambda r: (not r.is_default, -r.score, r.name))[:MOST]
    out: dict[str, Any] = {"used": {"key": role.key, "name": role.name}, "others": []}
    for r in others:
        en, fr = instead(r)
        out["others"].append({"key": r.key, "slug": r.slug, "name": r.name, "words": en, "words_fr": fr})
    measures = [m for m in model.measures.values() if m.slug in plan.measures]
    if len(measures) == 1 and measures[0].table == role.table:
        measure = measures[0]
        own = measure.default_date or model.tables[measure.table].default_date or ""
        if own and own != role.key:
            was = model.date_roles.get(own)
            name, date = measure.business_name, role.name.lower()
            out["fix"] = {
                "target_kind": "measure", "target_key": measure.key, "date": role.key,
                "measure": name, "date_name": role.name, "was": was.name if was is not None else "",
                "words_admin": f"Make {date} the date of {name} for everyone",
                "words_admin_fr": f"Compter {name} selon {date} pour tous",
                "words_reader": f"Suggest {date} for {name} to my admin",
                "words_reader_fr": f"Proposer {date} pour {name} à mon admin",
                "done_admin": f"{name} is now counted by {date} for everyone.",
                "done_admin_fr": f"{name} est maintenant compté selon {date} pour tous.",
                "done_reader": "Sent to your admin.", "done_reader_fr": "Envoyé à votre admin.",
            }
    return out if out["others"] or "fix" in out else None
