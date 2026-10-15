"""Every question a workspace's data can answer, worked out from the model and checked.

"The patient count by each doctor" was refused because nothing knew, before a reader asked, which questions the
data holds and how each is reached. The question map writes them down once per model: every figure by every
grouping a reader can split it by (and how: on its own rows, through a link, or counted on the rows that name both,
as patients by prescriber on prescriptions), every figure on its own, and every analysis the tables support
(returning customers, retention, the top 80%...). Each is resolved and compiled with no AI call; the leading ones
are run once on the data. What cannot be answered says why: two routes to pick from, no link, or a grouping under
the counted list.

Learn builds it at its end; it is rebuilt when an admin's decisions change the model. Suggested questions, the
planner's list of routes, the nearest questions offered after a refusal and the admin's What readers can ask page
all read it. It holds names and plans only: never a member's value, never a personal column.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from core2.model.schema import Measure, SemanticModel
from core2.resolve import paths as P

log = logging.getLogger("querybot.core2")

May = Callable[[str], bool]
MOST_GROUPS = 30          # groupings worked out for each figure
MOST_CHECKED = 400        # entries run on the data in one check
CHECK_BUDGET = 120.0      # seconds a check may take (Learn's last step)
MOST_ROUTES = 40          # rows-routes named to the planner
MOST_ACTIVITIES = 12      # the parties' activities whose analyses the map lists

ANSWERABLE = "answerable"
NEEDS_PICK = "needs_pick"         # two routes to the grouping: the reader is asked which
NOT_LINKED = "not_linked"         # no link and no rows naming both
CANNOT_SPLIT = "cannot_split"     # the grouping sits under the counted list (regions by customer)
UNCONFIRMED = "unconfirmed"       # reached through a link an admin has not confirmed yet
REFUSED = "refused"               # anything else the resolver would not answer

ANALYSES = ("lifecycle", "cohort", "gap", "pareto", "funnel", "together", "target", "active", "on_time")


def _all(table: str) -> bool:
    return True


@dataclass
class Entry:
    key: str                     # "<measure>|<grouping>", "<measure>", or "analysis:<kind>:<n>"
    kind: str                    # figure | split | analysis
    question: str                # in English, by the workspace's names
    question_fr: str
    plan: dict[str, Any]
    subject: str                 # the business name of the figure's table: its area
    status: str = ANSWERABLE
    route: str = ""              # own | link | rows (counted on rows naming both)
    rows_on: str = ""            # for route "rows": the rows' table, by name
    reason: str = ""
    options: list[str] = field(default_factory=list)
    measure: str = ""
    group: str = ""
    analysis: str = ""
    tables: list[str] = field(default_factory=list)   # every table the answer reads (a reader must see them all)
    ran: bool | None = None      # run on the data: None not yet, True returned rows, False returned none
    rows: int | None = None


@dataclass
class QuestionMap:
    version: int = 0             # the model version it was built on
    stamp: str = ""              # the model and its admin decisions, as core2_model_stamp gives them
    built_at: str = ""
    checked_at: str = ""
    entries: list[Entry] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.entries:
            out[e.status] = out.get(e.status, 0) + 1
        return out

    def to_json(self) -> str:
        return json.dumps({"version": self.version, "stamp": self.stamp, "built_at": self.built_at,
                           "checked_at": self.checked_at, "entries": [asdict(e) for e in self.entries]})

    @classmethod
    def from_json(cls, text: str) -> "QuestionMap":
        data = json.loads(text)
        return cls(version=int(data.get("version") or 0), stamp=str(data.get("stamp") or ""),
                   built_at=str(data.get("built_at") or ""), checked_at=str(data.get("checked_at") or ""),
                   entries=[Entry(**e) for e in data.get("entries") or []])


# ── what can be asked ──────────────────────────────────────────────────────────


def _figures(model: SemanticModel, may: May) -> list:
    return [m for m in model.measures.values() if not m.hidden and m.table in model.tables and may(m.table)
            and m.status != "rejected"]


def _groupings(model: SemanticModel, may: May) -> list[tuple[str, str]]:
    """What a reader splits figures by: each list (customers, products), then each category of few values (a
    status, a segment, a region). Never a person's own detail."""
    out: list[tuple[str, str]] = []
    labels = set()
    for e in model.entities.values():
        if e.table in model.tables and may(e.table):
            out.append((e.slug, e.business_name or e.slug))
            if e.label_column:
                labels.add(e.label_column)
    few = []
    for a in model.attributes.values():
        column = model.columns.get(a.column)
        if column is None or a.kind != "group" or a.column in labels or column.hidden or column.personal != "none" \
                or column.sensitivity != "none" or column.record_number or not may(column.table):
            continue
        members = a.members or (column.profile.distinct if column.profile else 0) or 0
        if 2 <= members <= 60:
            few.append((members, a.slug, a.business_name or a.slug))
    out += [(slug, name) for _, slug, name in sorted(few)]
    return out


def _tables_of(logical) -> list[str]:
    out: list[str] = []
    for part in logical.parts:
        for t in [part.table, *(j.table for j in part.joins), *(j.table for s in part.semis for j in s.joins)]:
            if t not in out:
                out.append(t)
    return out


def _resolve(model: SemanticModel, plan: dict, today: dt.date, allowed: set[str] | None = None):
    from core2.plan.ir import Plan
    from core2.resolve.resolver import Context, resolve

    return resolve(Plan.model_validate({"kind": "query", **plan}), model, Context(today=today,
                                                                                    allowed_tables=allowed))


def _judge(entry: Entry, model: SemanticModel, today: dt.date, measure: Measure | None = None) -> Entry:
    from core2.compile.compiler import compile_query
    from core2.resolve.resolver import ResolveError

    try:
        logical = _resolve(model, entry.plan, today)
        compile_query(logical, model, "duckdb")
    except ResolveError as exc:
        entry.reason, entry.options = exc.message, list(exc.options or [])
        entry.status = NEEDS_PICK if exc.kind == "ambiguous" and exc.options else \
            UNCONFIRMED if exc.kind == "unconfirmed" else \
            NOT_LINKED if "is not linked to" in exc.message else \
            CANNOT_SPLIT if "cannot split" in exc.message else REFUSED
        return entry
    except Exception as exc:  # noqa: BLE001 - a plan the compiler cannot write is no answer: said, never shown
        entry.status, entry.reason = REFUSED, f"{type(exc).__name__}: {exc}"
        return entry
    entry.tables = _tables_of(logical)
    on = logical.parts[0].table if logical.parts else ""
    if measure is not None and on and on != measure.table and on in model.tables:
        entry.route, entry.rows_on = "rows", model.tables[on].business_name or model.tables[on].name
    elif entry.kind == "split":
        entry.route = "own" if len(entry.tables) == 1 else "link"
    return entry


def build(model: SemanticModel, *, today: dt.date, may: May = _all, version: int = 0, stamp: str = "") \
        -> QuestionMap:
    """The map of a model: every figure, every figure by every grouping, every analysis, each judged."""
    with P.fixed(model):
        entries = _entries(model, today, may)
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return QuestionMap(version=version, stamp=stamp, built_at=now, entries=entries)


def _entries(model: SemanticModel, today: dt.date, may: May) -> list[Entry]:
    from core2.plan.analyses import suggestions

    entries: list[Entry] = []
    groupings = _groupings(model, may)
    for m in _figures(model, may):
        subject = model.tables[m.table].business_name or model.tables[m.table].name
        entries.append(_judge(Entry(key=m.slug, kind="figure", question=m.business_name, question_fr=m.business_name,
                                    plan={"intent": "value", "measures": [m.slug]}, subject=subject,
                                    measure=m.slug), model, today, m))
        for slug, name in groupings[:MOST_GROUPS]:
            entries.append(_judge(Entry(
                key=f"{m.slug}|{slug}", kind="split", question=f"{m.business_name} by {name.lower()}",
                question_fr=f"{m.business_name} par {name.lower()}",
                plan={"intent": "breakdown", "measures": [m.slug], "group_by": [slug]}, subject=subject,
                measure=m.slug, group=slug), model, today, m))
    try:
        found = suggestions(model, this_year=today.year, may=may)
    except Exception as exc:  # noqa: BLE001 - the figures stand without the analyses
        log.warning("core2: the analyses of a model could not be read for its question map: %s", exc)
        found = []
    # Every analysis of every other party's activity (patients' prescriptions, payers' claims), not only the
    # main one's: what a reader can ask of any table of dated rows that names them.
    from core2.plan.analyses import activities, of_activity

    shown = {json.dumps(plan, sort_keys=True) for _, plan in found}
    for act in activities(model, may)[1:MOST_ACTIVITIES]:
        for question, plan in of_activity(act):
            if json.dumps(plan, sort_keys=True) not in shown:
                shown.add(json.dumps(plan, sort_keys=True))
                found.append((question, plan))
    for i, (question, plan) in enumerate(found):
        plan = {k: v for k, v in plan.items() if k != "kind"}
        kind = next((k for k in ANALYSES if plan.get(k)), "analysis")
        of = plan.get(kind) or {}
        measure = str(of.get("measure") or of.get("actual") or of.get("basket") or "") if isinstance(of, dict) else ""
        m = next((x for x in model.measures.values() if x.slug == measure), None)
        subject = (model.tables[m.table].business_name if m is not None and m.table in model.tables else "") or ""
        entries.append(_judge(Entry(key=f"analysis:{kind}:{i}", kind="analysis", question=question,
                                    question_fr=question, plan=plan, subject=subject, measure=measure,
                                    analysis=kind), model, today, m))
    return entries


def check(qmap: QuestionMap, model: SemanticModel, warehouse: Any, *, today: dt.date,
          budget: float | None = CHECK_BUDGET, most: int = MOST_CHECKED) -> QuestionMap:
    """Run the answerable entries on the data (figures, then analyses, then splits), within ``budget`` seconds:
    each learns whether it returns rows. One that fails to run is marked refused with the reason."""
    from core2.compile.compiler import compile_query

    order = {"figure": 0, "analysis": 1, "split": 2}
    todo = sorted((e for e in qmap.entries if e.status == ANSWERABLE and e.ran is None),
                  key=lambda e: order.get(e.kind, 3))[:most]
    began = time.monotonic()
    with P.fixed(model):
        for e in todo:
            if budget is not None and time.monotonic() - began > budget:
                break
            try:
                sql = compile_query(_resolve(model, e.plan, today), model, warehouse.dialect).sql
                rows = warehouse.query(sql, max_rows=50).rows
            except Exception as exc:  # noqa: BLE001 - the data says no: the entry is not offered
                e.status, e.reason, e.ran = REFUSED, f"it did not run on the data: {exc}", False
                continue
            e.ran, e.rows = bool(rows) and any(any(v is not None for v in r) for r in rows), len(rows)
    qmap.checked_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return qmap


# ── reading it ─────────────────────────────────────────────────────────────────


def answerable(qmap: QuestionMap, allowed: set[str] | None = None) -> list[Entry]:
    """The entries a reader may ask and get rows for: answerable, never found empty, on tables they may use."""
    return [e for e in qmap.entries if e.status == ANSWERABLE and e.ran is not False
            and (allowed is None or set(e.tables) <= allowed)]


def nearest(qmap: QuestionMap, *, measure: str = "", group: str = "", allowed: set[str] | None = None,
            most: int = 4) -> list[Entry]:
    """The questions closest to one that cannot be answered: the same figure by other groupings first, then the
    same grouping of other figures."""
    ok = [e for e in answerable(qmap, allowed) if e.kind == "split"]
    same = [e for e in ok if measure and e.measure == measure and e.group != group]
    other = [e for e in ok if group and e.group == group and e.measure != measure]
    return (same + other)[:most]


def routes_text(qmap: QuestionMap | None, most: int = MOST_ROUTES) -> str:
    """For the planner: the figures counted on the rows that name both lists, by name only."""
    if qmap is None:
        return ""
    by: dict[str, list[Entry]] = {}
    for e in qmap.entries:
        if e.status == ANSWERABLE and e.route == "rows" and e.kind == "split":
            by.setdefault(e.measure, []).append(e)
    lines = []
    for routes in list(by.values())[:most]:
        figure = routes[0].question.split(" by ")[0]
        groups = list(dict.fromkeys(e.question[len(figure) + 4:] for e in routes))
        rows = list(dict.fromkeys(e.rows_on.lower() for e in routes))
        lines.append(f"- {figure} by {', '.join(groups[:8])}{' and more' if len(groups) > 8 else ''}: counted on "
                     f"{' or '.join(rows)} rows")
    if not lines:
        return ""
    return ("COUNTED THROUGH ROWS THAT NAME BOTH (plan them as any figure by a grouping; the counting is done for "
            "you):\n" + "\n".join(lines))


def coverage(qmap: QuestionMap) -> list[dict[str, Any]]:
    """By subject: how many questions are answerable, and the ones that are not, with why."""
    by: dict[str, dict[str, Any]] = {}
    for e in qmap.entries:
        s = by.setdefault(e.subject or "Other", {"subject": e.subject or "Other", "answerable": 0, "empty": 0,
                                                  "gaps": []})
        if e.status == ANSWERABLE:
            if e.ran is False:
                s["empty"] += 1
            else:
                s["answerable"] += 1
        elif e.kind == "split" or e.kind == "analysis":
            s["gaps"].append({"question": e.question, "status": e.status, "reason": e.reason,
                              "options": e.options})
    return sorted(by.values(), key=lambda s: (-s["answerable"], s["subject"]))


# ── kept with the model ────────────────────────────────────────────────────────


def stamp_of(account_id: str, db_id: int | None) -> str:
    import hashlib

    import store

    stamp = store.core2_model_stamp(account_id, db_id)
    return hashlib.sha256(json.dumps(stamp, default=str).encode()).hexdigest()[:16] if stamp else ""


def save(account_id: str, db_id: int | None, qmap: QuestionMap) -> None:
    import store

    store.save_core2_question_map(account_id, db_id, qmap.stamp, qmap.to_json())


def load(account_id: str, db_id: int | None) -> QuestionMap | None:
    import store

    row = store.load_core2_question_map(account_id, db_id)
    if not row:
        return None
    try:
        return QuestionMap.from_json(row["map_json"])
    except (TypeError, ValueError, KeyError) as exc:
        log.warning("core2: the question map of %s could not be read: %s", account_id, exc)
        return None


def build_for(account_id: str, client: dict[str, Any] | None = None, *, warehouse: Any = None,
              today: dt.date | None = None, budget: float | None = CHECK_BUDGET,
              say: Callable[[str], None] | None = None) -> QuestionMap | None:
    """Build, check and keep a workspace's map, on the model its questions are answered with."""
    import store
    from core2.bootstrap.service import answering_model

    client = client or store.get_client(account_id) or {}
    db_id = client.get("db_config_id")
    model = answering_model(account_id, client)
    if model is None:
        return None
    today = today or dt.date.today()
    qmap = build(model, today=today, version=int(model.version or 0), stamp=stamp_of(account_id, db_id))
    if warehouse is None:
        from core2.kpi_proposals import governed_warehouse

        warehouse = governed_warehouse(account_id, client)
    if warehouse is not None:
        check(qmap, model, warehouse, today=today, budget=budget)
    save(account_id, db_id, qmap)
    if say is not None:
        counts = qmap.counts()
        gaps = sum(v for k, v in counts.items() if k != ANSWERABLE)
        say(f"Mapped {counts.get(ANSWERABLE, 0)} questions this data answers"
            + (f"; {gaps} it cannot, each with the reason on the What readers can ask page" if gaps else ""))
    return qmap


REBUILD_BUDGET = 30.0     # seconds a rebuild after an admin's change spends checking on the data
_REBUILDING: set[str] = set()
_REBUILDING_LOCK = threading.Lock()


def _rebuild(account_id: str, client: dict[str, Any]) -> None:
    try:
        build_for(account_id, client, budget=REBUILD_BUDGET)
    except Exception as exc:  # noqa: BLE001 - asked again on the next question
        log.warning("core2: the question map of %s could not be rebuilt: %s", account_id, exc, exc_info=True)
    finally:
        with _REBUILDING_LOCK:
            _REBUILDING.discard(account_id)


def current(account_id: str, client: dict[str, Any] | None = None, *, rebuild: bool = True) -> QuestionMap | None:
    """The kept map when it was built on the model as it is now. Else None, and (``rebuild``) it is rebuilt in the
    background, once at a time: an admin's change to a metric, a link or a date is in it a few seconds later."""
    import store

    client = client or store.get_client(account_id) or {}
    db_id = client.get("db_config_id")
    stamp = stamp_of(account_id, db_id)
    qmap = load(account_id, db_id)
    if qmap is not None and qmap.stamp == stamp:
        return qmap
    if rebuild and stamp:
        with _REBUILDING_LOCK:
            if account_id in _REBUILDING:
                return None
            _REBUILDING.add(account_id)
        threading.Thread(target=_rebuild, args=(account_id, client), daemon=True,
                         name=f"question-map-{account_id}").start()
    return None


__all__ = ["Entry", "QuestionMap", "build", "check", "answerable", "nearest", "routes_text", "coverage", "save",
           "load", "build_for", "current", "ANSWERABLE", "NEEDS_PICK", "NOT_LINKED", "CANNOT_SPLIT", "UNCONFIRMED",
           "REFUSED"]
