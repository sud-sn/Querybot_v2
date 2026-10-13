"""
portal/dashboard_view.py

What a dashboard is looked at for: the dates its picker chose and the members its filters chose. Read from the
address, so a dashboard can be shared as it is seen, and put on each new-core tile's plan (``view_plan``).

The dates: one picker over every tile, from presets (this month, the default; last quarter; the fiscal ones where
the workspace's year does not start in January) or two days picked on its calendar. Each tile counts by its own
date -- an order's, a shipment's, a hire's: the span picked goes on that date, never on one date for the whole page.
"own" leaves each tile its own window, as it was pinned.

The members: the fields the tiles are grouped by, offered by the tiles themselves (four at most), and the fields
the dashboard's owner added to it (eight at most); each takes several members ("Region: North, South").
"""

from __future__ import annotations

import copy
import datetime as dt
import re
from typing import Any, Iterable

DEFAULT_PERIOD = "this_month"
OWN = "own"
CUSTOM = "custom"

DASH_MEMBER_FILTERS = 4          # the fields its tiles are grouped by, offered by the tiles
CREATOR_FILTERS = 8              # the fields its owner added
FIELD_FILTERS = DASH_MEMBER_FILTERS + CREATOR_FILTERS
MEMBER_VALUES = 50               # members chosen of one field
# A chart over time shows the span picked when it makes a line of at least this many points ("This year" on a
# monthly trend); a shorter one is shaded on the chart's own range ("This month" on the same trend: one point).
TREND_POINTS = 4
DRILL_DEPTH = 3                  # members a tile is drilled through, one a level
# A plan that is a measure counted for each member of one field: it can be drilled into, a member at a time. Not one
# worked out per member, in bands, over a funnel ... (core2/plan/ir.Plan.shaped), nor a list.
DRILLABLE_INTENTS = ("breakdown", "rank", "share", "compare")
_SHAPES = ("per", "rolling", "within", "bands", "absent", "gap", "lifecycle", "cohort", "active", "on_time", "pareto",
           "funnel", "together", "target", "drivers")
_FIELD = re.compile(r"^[A-Za-z0-9_.]{1,120}$")

# Each a window as a plan writes one, resolved by the new core from where the data ends (core2/resolve/time.py):
# "last quarter" on data that ends in June is January to March.
DASH_PERIODS: dict[str, dict] = {
    "today": {"kind": "this", "unit": "day"},
    "yesterday": {"kind": "previous", "unit": "day"},
    "last_7_days": {"kind": "last", "unit": "day", "n": 7, "include_current": True},
    "last_30_days": {"kind": "last", "unit": "day", "n": 30, "include_current": True},
    "last_90_days": {"kind": "last", "unit": "day", "n": 90, "include_current": True},
    "this_week": {"kind": "this", "unit": "week"},
    "last_week": {"kind": "previous", "unit": "week"},
    "this_month": {"kind": "this", "unit": "month"},
    "last_month": {"kind": "previous", "unit": "month"},
    "last_12_months": {"kind": "last", "unit": "month", "n": 12, "include_current": True},
    "this_quarter": {"kind": "this", "unit": "quarter"},
    "last_quarter": {"kind": "previous", "unit": "quarter"},
    "this_year": {"kind": "to_date", "unit": "year"},
    "last_year": {"kind": "previous", "unit": "year"},
}
# Offered where the fiscal year does not start in January (the model's settings).
FISCAL_PERIODS: dict[str, dict] = {
    "this_fiscal_quarter": {"kind": "this", "unit": "quarter", "fiscal": True},
    "last_fiscal_quarter": {"kind": "previous", "unit": "quarter", "fiscal": True},
    "fiscal_year_to_date": {"kind": "to_date", "unit": "year", "fiscal": True},
    "last_fiscal_year": {"kind": "previous", "unit": "year", "fiscal": True},
}
# How the picker lists them.
PERIOD_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("days", ("today", "yesterday", "last_7_days", "last_30_days", "last_90_days")),
    ("weeks", ("this_week", "last_week")),
    ("months", ("this_month", "last_month", "last_12_months")),
    ("quarters", ("this_quarter", "last_quarter")),
    ("years", ("this_year", "last_year")),
    ("fiscal", tuple(FISCAL_PERIODS)),
)


def fiscal_offered(fiscal_start: int | None) -> bool:
    return bool(fiscal_start) and int(fiscal_start or 1) != 1


def known_period(key: str, fiscal_start: int | None = None) -> bool:
    return key in DASH_PERIODS or (key in FISCAL_PERIODS and fiscal_offered(fiscal_start))


def _day(value: Any) -> dt.date | None:
    try:
        return dt.date.fromisoformat(str(value or "")[:10])
    except ValueError:
        return None


def _period(period: str, first: Any, last: Any, *, fiscal: bool, missing: str) -> tuple[str, str, str]:
    """(period, start, end) as one the dashboard knows: a custom span needs both its days (in order); a fiscal
    preset, a fiscal year that does not start in January."""
    if period == CUSTOM:
        start, end = _day(first), _day(last)
        if start and end:
            start, end = min(start, end), max(start, end)
            return CUSTOM, start.isoformat(), end.isoformat()
        return missing, "", ""
    if period == OWN or period in DASH_PERIODS or (fiscal and period in FISCAL_PERIODS):
        return period, "", ""
    return missing, "", ""


def read_members(pairs: Iterable[tuple[str, Any]]) -> dict[str, list[str]]:
    """``f.<field>=<member>`` pairs as {field: [members]}: a field may come several times, one member each."""
    members: dict[str, list[str]] = {}
    for key, value in pairs:
        key, value = str(key), str(value).strip()[:200]
        field = key[2:200]
        if not key.startswith("f.") or not field or not value:
            continue
        if field not in members and len(members) >= FIELD_FILTERS:
            continue
        values = members.setdefault(field, [])
        if value not in values and len(values) < MEMBER_VALUES:
            values.append(value)
    return members


def read_view(params: Any, *, fiscal_start: int | None = None) -> dict:
    """The view an address asks for: {"period", "start", "end", "members"}. With no period, or one this
    dashboard does not offer, it is this month; ``period=own`` is each tile's own window."""
    period, start, end = _period(str(params.get("period") or ""), params.get("from"), params.get("to"),
                                 fiscal=fiscal_offered(fiscal_start), missing=DEFAULT_PERIOD)
    pairs = params.multi_items() if hasattr(params, "multi_items") else list(params.items())
    return {"period": period, "start": start, "end": end, "members": read_members(pairs)}


def clean_view(view: Any) -> dict:
    """A view kept with a schedule, as one: a schedule saved before the picker had a default ("") drew each tile
    in its own window, and still does. A fiscal period kept with it stays one (the fiscal year is the admin's)."""
    view = view if isinstance(view, dict) else {}
    period, start, end = _period(str(view.get("period") or OWN), view.get("start"), view.get("end"),
                                 fiscal=True, missing=OWN)
    members = view.get("members") if isinstance(view.get("members"), dict) else {}
    pairs = [(f"f.{field}", value) for field, values in members.items()
             for value in (values if isinstance(values, list) else [values])]
    return {"period": period, "start": start, "end": end, "members": read_members(pairs)}


def address(view: Any) -> list[tuple[str, str]]:
    """A view kept with a schedule as an address's parameters: what the email's link and its pictures open."""
    view = clean_view(view)
    params = [("period", view["period"])]
    if view["period"] == CUSTOM:
        params += [("from", view["start"]), ("to", view["end"])]
    return params + [(f"f.{field}", value) for field, values in view["members"].items() for value in values]


def is_default(view: dict) -> bool:
    return view.get("period") == DEFAULT_PERIOD and not view.get("members")


def period_window(view: dict | None) -> dict | None:
    """The window the view puts on every tile, as a plan writes one; None: each tile's own."""
    view = view or {}
    period = str(view.get("period") or "")
    if period == CUSTOM:
        if view.get("start") and view.get("end"):
            return {"kind": "between", "start": str(view["start"]), "end": str(view["end"])}
        return None
    found = DASH_PERIODS.get(period) or FISCAL_PERIODS.get(period)
    return dict(found) if found else None


def span(window: dict, *, today: dt.date | None = None, last_data: dt.date | None = None,
         fiscal_start: int | None = None) -> tuple[dt.date, dt.date] | None:
    """The first and the last day of ``window`` (both counted)."""
    from core2.plan.ir import Window
    from core2.resolve.time import resolve_window

    try:
        found = resolve_window(Window(**window), today=today or dt.date.today(), last_data=last_data,
                               fiscal_start=fiscal_start)
    except Exception:  # noqa: BLE001 - a span is shown, never needed
        return None
    if found.start is None or found.end is None or found.end <= found.start:
        return None
    return found.start, found.end - dt.timedelta(days=1)


def points(window: dict, grain: str, *, fiscal_start: int | None = None, today: dt.date | None = None) -> int:
    """How many periods of ``grain`` the window spans: the points a chart over time would draw for it."""
    from core2.resolve.time import add_units, unit_start

    found = span(window, today=today, fiscal_start=fiscal_start)
    unit = str(grain or "").replace("fiscal_", "")
    if found is None or unit not in ("day", "week", "month", "quarter", "year"):
        return 0
    first, last = found
    at, count = unit_start(first, unit, fiscal_start=fiscal_start if str(grain).startswith("fiscal_") else None), 0
    while at <= last and count < 1000:
        count += 1
        at = add_units(at, unit, 1)
    return count


def view_plan(plan: dict, view: dict | None) -> tuple[dict, list[str]]:
    """A tile's plan under the dashboard's dates and filters, and the member filters applied to it.

    The span picked replaces the window of a tile that counts one span (a number, a breakdown, a ranking), on the
    tile's own date. A tile over time shows the span when it makes a line of TREND_POINTS points or more; a shorter
    one is shaded on its own range (``shaded_period``). A running or moving figure is shown over the span, and
    still reads the periods before it that its first points need (core2/resolve/shapes.py). A list of who had none
    keeps its own window: the window is what it is about ("no order in the last 90 days"). A member filter
    ("Region: North, South") narrows every tile but one grouped by that field, which keeps all its members (it is
    where the filter's choices come from)."""
    view = view or {}
    out = copy.deepcopy(plan)
    window = period_window(view)
    if window and not out.get("absent"):
        if not isinstance(out.get("time"), dict):
            out["time"] = {}
        time_spec = out["time"]
        grain = time_spec.get("grain")
        if not grain or out.get("rolling") or points(
                window, grain, fiscal_start=view.get("fiscal_start"), today=view.get("today")) >= TREND_POINTS:
            time_spec["window"] = window
    applied: list[str] = []
    groups = {str(g) for g in out.get("group_by") or []}
    for field, chosen in (view.get("members") or {}).items():
        values = [chosen] if isinstance(chosen, str) else [str(v) for v in chosen or []]
        values = [v for v in values if v.strip()]
        if field in groups or not values:
            continue
        out.setdefault("filters", []).append({"field": field, "op": "in", "values": values})
        applied.append(field)
    return out, applied


# ── Drilling into a tile ─────────────────────────────────────────────────────

def drillable(plan: dict | None) -> bool:
    """Can a tile drawn from ``plan`` be drilled into: a measure counted for each member of one field."""
    plan = plan or {}
    groups = [str(g) for g in plan.get("group_by") or [] if not str(g).startswith("time:")]
    return (plan.get("intent") in DRILLABLE_INTENTS and len(groups) == 1 and len(plan.get("group_by") or []) == 1
            and not (plan.get("time") or {}).get("grain") and not any(plan.get(k) for k in _SHAPES))


def read_drills(params: Any) -> dict[int, dict]:
    """The tiles drilled into, from the address: ``d.<tile>=<field>:<member>`` once a level, in order, and
    ``by.<tile>=<field>``, what the deepest level is broken down by. {tile: {"path": [{field, member}], "by"}}."""
    pairs = params.multi_items() if hasattr(params, "multi_items") else list(params.items())
    paths: dict[int, list[dict]] = {}
    for key, value in pairs:
        key = str(key)
        if not key.startswith("d.") or not key[2:].isdigit():
            continue
        field, sep, member = str(value).partition(":")
        member = member.strip()[:200]
        path = paths.setdefault(int(key[2:]), [])
        if sep and _FIELD.match(field) and member and len(path) < DRILL_DEPTH:
            path.append({"field": field, "member": member})
    out = {}
    for tile, path in paths.items():
        by = str(params.get(f"by.{tile}") or "")
        if path and _FIELD.match(by):
            out[tile] = {"path": path, "by": by}
    return out


def drill_address(drills: dict | None) -> list[tuple[str, str]]:
    """The address parameters of the tiles drilled into."""
    out = []
    for tile, drill in sorted((drills or {}).items()):
        out += [(f"d.{tile}", f"{step['field']}:{step['member']}") for step in drill.get("path") or []]
        out.append((f"by.{tile}", str(drill.get("by") or "")))
    return out


def drill_plan(plan: dict, drill: dict) -> dict:
    """A tile's plan drilled into: each member of the path a filter, broken down by the field chosen last (its
    measure, period, comparison, order and how many it shows kept)."""
    out = copy.deepcopy(plan)
    out["group_by"] = [drill["by"]]
    out["filters"] = [*(out.get("filters") or []),
                      *({"field": step["field"], "op": "in", "values": [step["member"]]} for step in drill["path"])]
    out.pop("chart", None)       # the shape chosen for the top level: the level below draws as its rows read
    return out


_ITS_ROWS = object()


def shaded_period(view: dict | None, shape: dict | None, *, last_data: Any = _ITS_ROWS) -> dict | None:
    """The span picked, on a chart over time that keeps its own range: shaded on it, where the chart has it.

    Its relative dates ("last quarter") are counted back from ``last_data`` -- the day the dashboard's other tiles
    counted theirs from, None for today -- and, when that is not known, from the chart's own last point."""
    window = period_window(view)
    if not window or not shape or ((shape.get("chart_spec") or {}).get("x") or {}).get("role") != "temporal":
        return None
    stamps = sorted(d for d in (_day(str(r.get(shape.get("x_key")) or "")[:10]) for r in shape.get("rows") or []) if d)
    if not stamps:
        return None
    anchor = stamps[-1] if last_data is _ITS_ROWS else last_data
    found = span(window, today=(view or {}).get("today"), last_data=anchor, fiscal_start=(view or {}).get("fiscal_start"))
    # The chart's last point stands for its whole period (a month, a week): the gap between its last two.
    step = (stamps[-1] - stamps[-2]).days if len(stamps) > 1 else 31
    if found is None or found[1] < stamps[0] or found[0] >= stamps[-1] + dt.timedelta(days=step):
        return None
    return {"start": found[0].isoformat(), "end": (found[1] + dt.timedelta(days=1)).isoformat()}


# ── Words ──────────────────────────────────────────────────────────────────

_NAMING_WORDS = ("name", "description", "desc")   # never "code" or "title": "Postal code", "Job title"


def field_word(field: str) -> str:
    """A field as the reader thinks of it: "Region", not "Region name" or "region.name"."""
    words = str(field or "").replace(".", " ").replace("_", " ").split()
    while len(words) > 1 and words[-1].lower() in _NAMING_WORDS:
        words.pop()
    text = " ".join(words)
    return text[:1].upper() + text[1:]


def drill_word(field: str) -> str:
    """A field on a drilled tile's path as its menu offered it: the attribute without its table ("Status code" for
    order_line.status_code, "Segment" for customer.segment), a member by its own name ("Region" for region.name)."""
    owner, _, own = str(field or "").rpartition(".")
    return field_word(own if owner and own.lower() not in (*_NAMING_WORDS, "code", "id", "key") else owner or own)


def span_label(first: dt.date, last: dt.date, lang: str) -> str:
    """"1–13 Oct 2026", "1 Jan – 31 Mar 2026", "1 Dec 2025 – 31 Jan 2026"; one day alone."""
    from core.i18n import month_name

    def day(d: dt.date, year: bool = True) -> str:
        return f"{d.day} {month_name(d.month, short=True, lang=lang)}" + (f" {d.year}" if year else "")

    if first == last:
        return day(first)
    if (first.year, first.month) == (last.year, last.month):
        return f"{first.day}–{day(last)}"
    if first.year == last.year:
        return f"{day(first, year=False)} – {day(last)}"
    return f"{day(first)} – {day(last)}"


def period_label(view: dict, lang: str) -> str:
    from core.i18n import t

    period = str(view.get("period") or "")
    if period == CUSTOM:
        first, last = _day(view.get("start")), _day(view.get("end"))
        return span_label(first, last, lang) if first and last else ""
    return t(f"ui.dash.period.{period}", lang=lang) if period else ""


def filter_parts(view: dict, labels: dict[str, str], lang: str) -> list[dict]:
    """What the dashboard was drawn for, as [{"name", "value"}]: the period (unless each tile's own), then each
    field's members. An email and a PDF say it at their top; the schedule's form says what it will keep."""
    from core.i18n import t

    parts = []
    if view.get("period") and view.get("period") != OWN:
        parts.append({"name": t("ui.dash.period", lang=lang), "value": period_label(view, lang)})
    for field, values in (view.get("members") or {}).items():
        values = [values] if isinstance(values, str) else list(values or [])
        if values:
            parts.append({"name": labels.get(field) or field_word(field), "value": ", ".join(values)})
    return parts


def filters_line(parts: list[dict]) -> str:
    return " · ".join(f"{p['name']}: {p['value']}" for p in parts)


# ── The picker and the filters, for the page's script ───────────────────────

def picker(view: dict, *, fiscal_start: int | None, lang: str, today: dt.date | None = None,
           data_end: str | None = None) -> dict:
    """The date picker's state: what is picked, every preset by group with the days it spans (the calendar shows
    them), and today. ``data_end``: the data's last day, where the tiles counted their dates back from it rather
    than from today (it ended long ago) -- "this month" is then the month it ends in, and the picker says so."""
    from core.i18n import t

    today = today or dt.date.today()
    ended = _day(data_end)
    groups = []
    for group, keys in PERIOD_GROUPS:
        if group == "fiscal" and not fiscal_offered(fiscal_start):
            continue
        presets = []
        for key in keys:
            found = span(DASH_PERIODS.get(key) or FISCAL_PERIODS[key], today=today, last_data=ended,
                         fiscal_start=fiscal_start)
            presets.append({"key": key, "label": t(f"ui.dash.period.{key}", lang=lang),
                            "start": found[0].isoformat() if found else "", "end": found[1].isoformat() if found else ""})
        groups.append({"key": group, "label": t(f"ui.dash.dp.group.{group}", lang=lang), "presets": presets})
    window = period_window(view)
    found = span(window, today=today, last_data=ended, fiscal_start=fiscal_start) if window else None
    behind = bool(ended and window and view.get("period") != CUSTOM)
    return {
        "period": view.get("period") or OWN, "start": view.get("start") or "", "end": view.get("end") or "",
        "label": period_label(view, lang) if view.get("period") != OWN else t("ui.dash.period.own", lang=lang),
        "span": span_label(*found, lang) if found and view.get("period") != CUSTOM else "",
        "from": found[0].isoformat() if found else "", "to": found[1].isoformat() if found else "",
        "default": DEFAULT_PERIOD, "groups": groups, "today": today.isoformat(), "lang": lang,
        "note": t("ui.dash.dp.behind", lang=lang) if behind else "",
    }
