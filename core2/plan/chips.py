"""Chips that change one setting of an analytical answer and run its plan again, with no AI.

"Median instead", "95th percentile instead", "6-month moving average", "Top 5 in each store", "No orders in the
last 180 days", "Counting every customer listed": each is the answer's own plan with one setting changed, kept on
the server in the chat it was offered in. The reader's click sends the chip's words, which name that plan: it runs
as it is, so the answer changes exactly as the chip says and nothing else is read into it.
"""

from __future__ import annotations

from core2.bootstrap.names import plural
from core2.model.schema import SemanticModel
from core2.plan.ir import Plan
from core2.resolve.resolver import find_slug

MOST = 3          # chips of this kind on one answer
_UNITS = {"fiscal_month": "month", "fiscal_quarter": "quarter", "fiscal_year": "year"}


def _ordinal(p: float) -> str:
    n = int(p)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _near(current: float, choices: tuple) -> list:
    """The choice just below ``current`` and the one just above it; the two nearest when it is at an end."""
    below = [c for c in choices if c < current]
    above = [c for c in choices if c > current]
    if below and above:
        return [below[-1], above[0]]
    return sorted(sorted(below + above, key=lambda c: (abs(c - current), c))[:2])


def _name(model: SemanticModel, slug: str | None) -> str:
    found = find_slug(model, slug) if slug else None
    if found is None:
        return ""
    name = str(getattr(found[1], "business_name", "") or getattr(found[1], "name", "")).lower()
    return name.removesuffix(" name") if found[0] == "attribute" else name


def _noun(name: str) -> str:
    for lead in ("number of ", "count of ", "no. of "):
        if name.startswith(lead):
            return name[len(lead):]
    return name


def offered(plan: Plan, model: SemanticModel) -> list[tuple[str, Plan]]:
    """(the chip's words, the plan it runs) for an analytical answer; none for any other."""
    chips: list[tuple[str, Plan]] = []
    if plan.per is not None:
        chips += _per(plan, model)
    if plan.rolling is not None:
        chips += _rolling(plan)
    if plan.within is not None:
        chips += _within(plan, model)
    if plan.absent is not None:
        chips += _absent(plan, model)
    if plan.bands is not None and plan.bands.edges:
        chips.append(("Ranges picked from the data", plan.model_copy(update={
            "bands": plan.bands.model_copy(update={"edges": []})})))
    seen: set[str] = set()
    out = []
    for words, changed in chips:
        if words.casefold() not in seen:
            seen.add(words.casefold())
            out.append((words, changed.model_copy(update={"follow_up": "refine"})))
    return out[:MOST]


def _per(plan: Plan, model: SemanticModel) -> list[tuple[str, Plan]]:
    per = plan.per
    assert per is not None

    def with_(**changes) -> Plan:
        return plan.model_copy(update={"per": per.model_copy(update=changes)})

    chips: list[tuple[str, Plan]] = []
    if per.summary == "avg":
        chips.append(("Median instead", with_(summary="median", everyone=False)))
    elif per.summary == "median":
        chips.append(("Average instead", with_(summary="avg")))
        chips += [(f"{_ordinal(q)} percentile instead", with_(summary="percentile", p=q)) for q in (75, 90)]
    elif per.summary == "percentile" and per.p is not None:
        for q in _near(per.p, (50, 75, 90, 95)):
            chips.append(("Median instead", with_(summary="median", p=None)) if q == 50 else
                         (f"{_ordinal(q)} percentile instead", with_(p=q)))
    elif per.summary in ("min", "max"):
        chips.append(("Highest instead" if per.summary == "min" else "Lowest instead",
                      with_(summary="max" if per.summary == "min" else "min")))
    elif per.op == "gte" and len(per.values) == 1 and float(per.values[0]).is_integer():
        n = int(per.values[0])
        what = _noun(_name(model, plan.measures[0])) if plan.measures else ""
        for m in (n - 1, n + 1):
            if m >= 1:
                chips.append((f"{m} or more {what} instead".replace("  ", " "), with_(values=[float(m)])))
    member = _name(model, per.of)
    found = find_slug(model, per.of) if per.of else None
    if found is not None and found[0] == "entity" and per.summary in ("avg", "count", "share", "min", "max") \
            and not plan.group_by:
        if per.everyone:
            chips.append((f"Only {plural(member)} with rows", with_(everyone=False)))
        else:
            chips.append((f"Counting every {member} listed", with_(everyone=True)))
    return chips


def _rolling(plan: Plan) -> list[tuple[str, Plan]]:
    r = plan.rolling
    assert r is not None
    unit = _UNITS.get(plan.time.grain or "month", plan.time.grain or "month")

    def with_(**changes) -> Plan:
        return plan.model_copy(update={"rolling": r.model_copy(update=changes)})

    if r.kind == "running":
        if r.reset is None:
            return [("Year to date instead", with_(reset="year"))]
        return [("Running total since the data begins", with_(reset=None))]
    words = "moving average" if r.kind == "moving_average" else "total"
    choices = {"month": (3, 6, 12), "week": (4, 8, 13), "quarter": (2, 4, 8), "day": (7, 30, 90)}.get(unit, (3, 6, 12))
    chips = [(f"{q}-{unit} {words}" if words == "moving average" else f"Total of the last {q} {unit}s",
              with_(periods=q)) for q in _near(r.periods, choices)]
    if r.kind == "moving_average":
        chips.append(("Running total instead", with_(kind="running", reset=None)))
    return chips


def _within(plan: Plan, model: SemanticModel) -> list[tuple[str, Plan]]:
    w = plan.within
    assert w is not None
    group = _name(model, w.group)
    top = w.top or plan.limit or 0
    return [(f"Top {q} in each {group}", plan.model_copy(update={"within": w.model_copy(update={"top": q})}))
            for q in _near(top, (3, 5, 10))]


def _absent(plan: Plan, model: SemanticModel) -> list[tuple[str, Plan]]:
    a = plan.absent
    assert a is not None
    activity = _noun(_name(model, a.measure))
    window = plan.time.window
    chips: list[tuple[str, Plan]] = []
    if window.kind == "last" and window.unit == "day" and window.n:
        for q in _near(window.n, (30, 60, 90, 180, 365)):
            chips.append((f"No {activity} in the last {q} days", plan.model_copy(update={
                "time": plan.time.model_copy(update={"window": window.model_copy(update={"n": q})})})))
    if window.kind != "all":
        if a.active_before:
            chips.append((f"Every {_name(model, a.of)} listed", plan.model_copy(update={
                "absent": a.model_copy(update={"active_before": False})})))
        else:
            chips.append((f"Only those with {activity} before", plan.model_copy(update={
                "absent": a.model_copy(update={"active_before": True})})))
    return chips
