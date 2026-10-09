"""core2's answer to a question, for any channel (DESIGN §12.1).

:func:`answer_question` takes everything it needs injected (the model, a
warehouse, the AI, the member index, today), so tests run it on DuckDB with a
recorded AI and production runs it on the workspace's governed connection and
configured provider. It returns the frame the web portal renders.

    question -> member names found -> plan (AI, checked, one repair) -> logical query
             -> SQL for the warehouse -> rows (governed) -> answer

Nothing here writes SQL from text: a question the plan cannot express is said so
(``unsupported``), and in compare mode the current pipeline's answer stands.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from core2 import ids
from core2.bootstrap import names
from core2.answer.builder import build_answer
from core2.answer.describe import describe
from core2.answer.drivers import answer_drivers
from core2.answer.forecast import answer_forecast
from core2.answer.suggestions import follow_ups
from core2.compile.compiler import CompileError, compile_query
from core2.model.schema import Attribute, DateRole, Entity, Measure, SemanticModel
from core2.plan.ir import Clarify, Compare, Plan, Window
from core2.plan.planner import Complete, Outcome, Turn, plan_question
from core2.plan.values import Masked, MemberIndex, ValueMatch, build_index, listable
from core2.resolve.resolver import Context, ResolveError, find_slug, resolve
from core2.warehouse.runner import Guarded, QueryFailed, Warehouse

log = logging.getLogger("querybot.core2")

HISTORY = 3          # turns of the conversation the planner sees
MAX_ROWS = 5000


@dataclass
class Services:
    model: SemanticModel
    warehouse: Warehouse
    complete: Complete
    index: MemberIndex
    today: dt.date
    values_allowed: bool = True                 # may member values reach the AI (False for regulated tenants)
    allowed_tables: set[str] | None = None      # model table keys the reader may use; None = all
    data_source: str = ""
    scrub: Callable[[str], str] | None = None   # personal data out of the question before the AI (regulated)
    split_units: bool = True                    # a quantity in several units is answered per unit (issue E4)


@dataclass
class Pending:
    """A question back that code completes: the plan waiting for the link the reader names."""

    plan: Plan
    field: str                       # the slug whose link the reply names (plan.via), or "time.date"
    options: list[str]
    masked: Masked | None = None


@dataclass
class Session:
    turns: list[Turn] = field(default_factory=list)
    pending: Pending | None = None


_ORDINALS = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2, "fourth": 3, "4th": 3,
             "fifth": 4, "5th": 4, "sixth": 5, "6th": 5, "last": -1}
_FILLER = {"the", "one", "option", "number", "no", "please", "use", "i", "mean", "meant", "it", "is", "via",
           "by", "a", "an"}


def _plain(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def choose(reply: str, options: list[str]) -> str | None:
    """The option a reply picks: its name, its number ("2", "the second one"), or a start only it has."""
    text = _plain(reply)
    if not text or not options:
        return None
    plain = [_plain(o) for o in options]
    if text in plain:
        return options[plain.index(text)]
    words = [w for w in text.split() if w not in _FILLER]
    if len(words) == 1:
        word = words[0]
        if word.isdigit() and 1 <= int(word) <= len(options):
            return options[int(word) - 1]
        if word in _ORDINALS and _ORDINALS[word] < len(options):
            return options[_ORDINALS[word]]
    starts = [o for o, p in zip(options, plain) if p.startswith(text)]
    if len(starts) == 1:
        return starts[0]
    inside = [o for o, p in zip(options, plain) if p and re.search(rf"\b{re.escape(p)}\b", text)]
    return inside[0] if len(inside) == 1 else None


def _frame(question: str, text: str, **extra: Any) -> dict[str, Any]:
    answer = {"headline": text, "short_value": "", "comparison": "", "scope_badge": "", "scope_note": ""}
    return {"type": "assistant_response", "engine": "core2", "question": question, "answer": answer, "chart": None,
            "kpi": None, "data": None, "trust": {"engine": "core2"}, "confidence": {}, "insight_summary": "",
            "anomaly_callouts": [], "coverage_caveats": [], "follow_up_suggestions": [], **extra}


def _clarification(question: str, clarify: Clarify) -> dict[str, Any]:
    """A question back: the options in the sentence, and each one a chip that answers it."""
    options = [o for o in clarify.options if o][:6]
    asked = clarify.question.rstrip(" ?:.")
    if options:
        listed = options[0] if len(options) == 1 else ", ".join(options[:-1]) + f" or {options[-1]}"
        text = f"{asked}: {listed}?"
    else:
        text = clarify.question
    return _frame(question, text, clarify={"about": clarify.about, "question": clarify.question, "options": options},
                  follow_up_suggestions=[{"label": o, "question": o} for o in options])


def _describe(question: str, plan: Plan, services: Services) -> dict[str, Any]:
    """A question about the data itself, answered from the model: a short lead, a section per subject or thing
    asked about, and example questions to ask next (core2/answer/describe.py)."""
    said = describe(services.model, plan.about, today=services.today, allowed=services.allowed_tables,
                    values=services.values_allowed)
    frame = _frame(question, said.headline, sections=said.sections,
                   follow_up_suggestions=[{"label": q, "question": q} for q in said.examples],
                   trust={"engine": "core2", "data_source": services.data_source,
                          "model_version": services.model.version})
    frame["answer"]["scope_note"] = said.note
    return frame


_OPS = {"eq": "is", "in": "is one of", "ne": "is not", "not_in": "is none of", "gt": ">", "gte": ">=", "lt": "<",
        "lte": "<=", "between": "between", "contains": "contains", "starts_with": "starts with",
        "is_null": "is empty", "not_null": "is not empty"}


def _span(window: Window) -> str:
    unit = f"{'fiscal ' if window.fiscal else ''}{window.unit or 'period'}"
    n = window.n or 1
    return {"between": f"{window.start} to {window.end}", "since": f"since {window.start}",
            "until": f"until {window.end}", "last": f"the last {n} {unit}{'s' if n > 1 else ''}",
            "this": f"this {unit}", "to_date": f"{unit} to date", "previous": f"the previous {unit}"}.get(window.kind, "")


def _window(plan: Plan, model: SemanticModel) -> str:
    """The plan's dates in words: the date it is on, the window, the grain and the comparison."""
    time = plan.time
    said = [_span(time.window)]
    if time.grain:
        said.append(f"{time.grain} by {time.grain}")
    if time.compare is not None:
        other = _span(time.compare.window) if time.compare.window is not None else ""
        said.append(f"compared with {other or 'the ' + time.compare.kind.replace('_', ' ')}")
    text = ", ".join(part for part in said if part)
    if time.date:
        found = find_slug(model, time.date)
        role = found[1] if found is not None and isinstance(found[1], DateRole) else None
        text = f"{role.name if role is not None and role.name else time.date}: {text}".rstrip(": ")
    return text


def _considered(plan: Plan | None, model: SemanticModel) -> list[dict[str, str]]:
    """What a plan that was not answered asked for, in the model's names and with each one's table.

    A refusal's "How this answer was produced": no query ran, so this is what
    was looked at -- the measure and its table, the breakdown, the filters,
    the dates -- beside why it stopped.
    """
    if plan is None or plan.kind != "query":
        return []

    def table(key: str) -> str:
        return model.tables[key].business_name if key in model.tables else key

    def named(slug: str) -> str:
        found = find_slug(model, slug)
        thing = found[1] if found is not None else None
        if isinstance(thing, Attribute):
            column = model.columns.get(thing.column)
            return f"{thing.business_name or slug} ({table(column.table)})" if column else thing.business_name or slug
        if isinstance(thing, DateRole):
            return f"{thing.name or slug} ({table(thing.table)})"
        if isinstance(thing, (Measure, Entity)):
            return f"{thing.business_name or slug} ({table(thing.table)})"
        return slug.removeprefix("time:").replace("_", " ")

    rows: list[dict[str, str]] = []
    measures = [named(s) for s in plan.measures]
    measures += [f"{d.name} ({d.op} of {' and '.join(named(m) for m in d.measures)})" for d in plan.derived]
    measures += [f"{d.name} (days from {named(d.start)} to {named(d.end)})" for d in plan.durations]
    if measures:
        rows.append({"key": "measure", "value": "; ".join(measures)})
    if plan.group_by:
        rows.append({"key": "by", "value": ", ".join(named(s) for s in plan.group_by)})
    if plan.filters:
        rows.append({"key": "filter", "value": "; ".join(
            f"{named(f.field)} {_OPS.get(f.op, f.op)} {', '.join(str(v) for v in f.values)}".strip()
            for f in plan.filters)})
    dates = _window(plan, model)
    if dates:
        rows.append({"key": "dates", "value": dates})
    return rows


def _refused(question: str, text: str, plan: Plan | None, services: Services, stopped: str,
             **extra: Any) -> dict[str, Any]:
    """A refusal, with what was considered and why it stopped under "How this answer was produced"."""
    trust = {"engine": "core2", "considered": _considered(plan, services.model), "stopped": stopped,
             "model_version": services.model.version, "data_source": services.data_source,
             **extra.pop("trust", {})}
    return _frame(question, text, trust=trust, **extra)


_REFUSALS = {
    "denied": "That needs data you do not have access to: {message}.",
    "unconfirmed": "{message}. An admin can confirm it on the What QueryBot learned page.",
    "unsupported": "I cannot answer that from this data: {message}.",
    "unknown": "I could not find that in this data: {message}.",
    "empty": "{message}.",
    "sensitive": "{message}.",
}


def answer_question(question: str, services: Services, session: Session, *, question_id: str = "") -> dict[str, Any]:
    """The portal frame answering ``question``; the session keeps the plan for follow-ups."""
    start = time.perf_counter()
    model = services.model
    pending, session.pending = session.pending, None
    picked = choose(question, pending.options) if pending is not None else None
    if pending is not None and picked is not None:
        # The reply names one of the links or dates offered: the waiting plan, completed by code.
        if pending.field == "time.date":
            # The offered dates are tables' defaults; another table may have a date of the same name.
            named = [r for r in model.date_roles.values() if r.name == picked]
            role = next((r for r in named if r.is_default), named[0] if named else None)
            update: dict[str, Any] = {"time": pending.plan.time.model_copy(
                update={"date": role.slug if role is not None else picked})}
        else:
            update = {"via": {**pending.plan.via, pending.field: picked}}
        outcome = Outcome(pending.plan.model_copy(update={**update, "follow_up": "refine"}), masked=pending.masked)
    else:
        matches = _visible(services.index.match(question), model, services.allowed_tables)
        outcome = plan_question(model, question, services.complete, today=services.today,
                                history=session.turns[-HISTORY:], matches=matches,
                                values_allowed=services.values_allowed, scrub=services.scrub)
    plan = outcome.plan
    if plan.kind == "smalltalk":
        return _frame(question, "Hello! Ask me about your data, for example a total, a trend or a ranking.",
                      kind="smalltalk")
    if plan.kind == "describe_data":
        return _describe(question, plan, services)
    if plan.kind == "unsupported":
        why = " ".join(plan.notes) or "nothing in this data measures it"
        # After a repair round, the plan that was tried first says what was looked at.
        return _refused(question, f"I cannot answer that from this data: {why}", outcome.tried or plan, services,
                        why, unsupported=True)
    if plan.kind == "clarify" and plan.clarify is not None:
        # Kept in the conversation: the reply ("the first one") answers this question.
        session.turns.append(Turn(question, plan, outcome.masked))
        del session.turns[:-HISTORY]
        asked = _clarification(question, plan.clarify)
        if outcome.problems:      # the planner's plans did not check out: what it tried, and why not
            asked["trust"] = {**asked["trust"], "considered": _considered(outcome.tried, model),
                              "stopped": " ".join(outcome.problems)}
        return asked

    plan, missing = _members_named(plan, model, services.index)

    ctx = Context(today=services.today, allowed_tables=services.allowed_tables, max_rows=MAX_ROWS)
    try:
        payload = _compute(question, plan, services, ctx, question_id=question_id, started=start)
    except ResolveError as exc:
        if exc.kind == "ambiguous":
            session.turns.append(Turn(question, plan, outcome.masked))     # the reply names the role this asks for
            del session.turns[:-HISTORY]
            if exc.field:
                session.pending = Pending(plan, exc.field, list(exc.options), outcome.masked)
            return _clarification(question, Clarify(about="path", question=f"{exc.message}. Which one do you mean",
                                                    options=exc.options))
        template = _REFUSALS.get(exc.kind, "{message}.")
        stopped = exc.message + (f" ({', '.join(exc.options)})" if exc.kind == "unconfirmed" and exc.options else "")
        return _refused(question, template.format(message=exc.message), plan, services, stopped,
                        unsupported=exc.kind == "unsupported")
    except QueryFailed as exc:     # a refused or failed query is told, never raised to the socket
        log.warning("core2 query failed for %r: %s", question, exc.cause)
        cause = exc.cause
        reason = str(cause).split(":", 1)[-1].strip() if "Policy" in type(cause).__name__ else "the database refused it"
        return _refused(question, f"The question was understood but the query could not run ({reason}).", plan,
                        services, f"The query could not run: {reason}", trust={"sql": exc.sql})
    except (CompileError, ValueError) as exc:
        log.warning("core2 could not compile a plan for %r: %s", question, exc)
        return _refused(question, f"I cannot answer that from this data yet: {exc}.", plan, services, str(exc),
                        unsupported=True)
    if missing is not None and _found_nothing(payload):
        # Nothing matched, and a member the plan named is not one the data had when its members were read:
        # say so, not "No rows match", which reads as if that member bought nothing. (A member added since
        # is found by the query, so it is never refused for being new.)
        thing, value = missing
        text = _REFUSALS["unknown"].format(message=f'there is no {thing} called "{value}"')
        if plan.follow_up == "refine":
            text += (f" If you mean the {names.plural(thing)} in the answer above, ask to compare that answer with "
                     "the other period, for example \"the same ranking against last year\".")
        return _refused(question, text, plan, services, f'No {thing} is called "{value}", and nothing matched.',
                        trust={"sql": (payload.get("trust") or {}).get("sql", "")})
    payload["plan"] = plan.model_dump(mode="json", exclude_defaults=True)
    if outcome.repaired:
        payload["trust"]["plan_repaired"] = True
    session.turns.append(Turn(question, plan, outcome.masked))
    del session.turns[:-HISTORY]
    return payload


def _compute(question: str, plan: Plan, services: Services, ctx: Context, *, question_id: str,
             started: float) -> dict[str, Any]:
    """The answer to a checked plan: one query, or the several a "why" or a forecast needs."""
    model = services.model
    warehouse = Guarded(services.warehouse)
    common: dict[str, Any] = {"data_source": services.data_source, "question_id": question_id,
                              "model_version": model.version}
    if plan.intent == "drivers":
        return answer_drivers(question, plan, model=model, warehouse=warehouse, ctx=ctx, started=started, **common)
    if plan.intent == "forecast":
        return answer_forecast(question, plan, model=model, warehouse=warehouse, ctx=ctx, started=started, **common)
    logical = resolve(plan, model, replace(ctx, split_units=services.split_units))
    compiled = compile_query(logical, model, warehouse.dialect)
    result = warehouse.query(compiled.sql, max_rows=compiled.row_cap)
    payload = build_answer(question, logical, compiled, result.columns, result.rows,
                           duration_ms=(time.perf_counter() - started) * 1000, truncated=result.truncated,
                           chart_type=plan.chart, **common)
    payload["follow_up_suggestions"] = follow_ups(plan, logical, payload, model, services.allowed_tables)
    context = _against_before(question, plan, services, ctx, warehouse, logical, payload)
    if context:
        payload["key_insights"] = [context, *payload.get("key_insights", [])]
    return payload


def _against_before(question: str, plan: Plan, services: Services, ctx: Context, warehouse: Guarded,
                    logical: Any, payload: dict[str, Any]) -> str:
    """One number for a period: how it compares with the period before, from one more small query.

    "Gross profit in Q2 2026: $8.51M" says nothing of whether that is good; the
    period before is the first thing a reader checks. Only for a single value of a
    bounded period that is not already a comparison; a failure costs the finding,
    never the answer.
    """
    if plan.group_by or plan.time.grain or plan.time.compare or logical.compare is not None \
            or plan.intent not in (None, "value", "count") or len(logical.measures) != 1:
        return ""
    if logical.window.start is None or logical.window.end is None or any(
            c.kind == "date" for c in logical.conditions):
        return ""                     # "invoiced in March for orders placed in February" has no period before
    rows = (payload.get("data") or {}).get("rows") or []
    if len(rows) != 1 or _found_nothing(payload):
        return ""
    before = plan.model_copy(update={"intent": "compare", "time": plan.time.model_copy(
        update={"compare": Compare(kind="previous_period")})})
    try:
        compared = resolve(before, services.model, replace(ctx, split_units=services.split_units))
        role = compared.parts[0].date.role if compared.parts and compared.parts[0].date else None
        if compared.compare is None or compared.compare.start is None or role is None or role.first is None \
                or compared.compare.start < role.first:
            return ""                 # data that starts later is not a period with nothing in it
        compiled = compile_query(compared, services.model, warehouse.dialect)
        result = warehouse.query(compiled.sql, max_rows=compiled.row_cap)
        moved = build_answer(question, compared, compiled, result.columns, result.rows)
    except Exception as exc:  # noqa: BLE001 - the answer stands without the comparison
        log.warning("core2 could not compare %r with the period before: %s", question, exc)
        return ""
    said = str((moved.get("answer") or {}).get("comparison") or "")
    if not said or said.startswith("nothing"):
        return ""
    return f"{said[:1].upper()}{said[1:]}, the period before."


def _found_nothing(payload: dict[str, Any]) -> bool:
    """No rows, or only empty values: a total over nothing comes back as one row with no value."""
    rows = (payload.get("data") or {}).get("rows") or []
    return all(all(v is None for v in row.values()) for row in rows)


def _members_named(plan: Plan, model: SemanticModel, index: MemberIndex) -> tuple[Plan, tuple[str, str] | None]:
    """The plan with each member it filters on spelled as stored, and the first one the index does not know.

    "How did those same customers do in 2025?" was planned with customers the AI
    was never told the names of (it sees the plan behind the answer on screen,
    not its rows): the filter matched nothing, and the answer was "No rows
    match", as if they had bought nothing. A member the index does not know is
    returned so that an empty answer can say so; the query still runs, since a
    member added after the index was read is real. One written in another case
    ("retail") is the stored one ("RETAIL"): the warehouse compares text exactly.
    """
    filters = []
    missing: tuple[str, str] | None = None
    for f in plan.filters:
        found = find_slug(model, f.field) if f.op in ("eq", "in", "ne", "not_in") else None
        attribute: Attribute | None = None
        thing = ""
        if found is not None and isinstance(found[1], Attribute):
            attribute = found[1]
            thing = attribute.business_name.lower()
        elif found is not None and isinstance(found[1], Entity):
            entity = found[1]
            column = entity.label_column or entity.code_column
            attribute = next((a for a in model.attributes.values() if a.column == column), None)
            thing = entity.business_name.lower()
        if attribute is None or attribute.slug not in index.attributes:
            filters.append(f)
            continue
        values = []
        for value in f.values:
            stored = index.stored(attribute.slug, value) if isinstance(value, str) else value
            if stored is None and f.op in ("eq", "in") and missing is None:
                missing = (thing, str(value))      # excluding a member there is none of is harmless
            values.append(value if stored is None else stored)
        filters.append(f.model_copy(update={"values": values}))
    return plan.model_copy(update={"filters": filters}), missing


def _visible(matches: list[ValueMatch], model: SemanticModel, allowed: set[str] | None) -> list[ValueMatch]:
    """Member names only from tables the reader may use."""
    if allowed is None:
        return matches
    return [m for m in matches
            if m.attribute in model.attributes and model.columns[model.attributes[m.attribute].column].table in allowed]


# ── the portal's wiring ────────────────────────────────────────────────────

MEMBERS_HOURS = 12.0          # member names are read again, in the background, after this long


@dataclass
class _Members:
    index: MemberIndex
    listable: frozenset[str]      # the attributes that could be listed when it was read
    read_at: float
    refreshing: bool = False


_INDEXES: OrderedDict[tuple[str, int], _Members] = OrderedDict()
_SESSIONS: OrderedDict[str, Session] = OrderedDict()
_LOCK = threading.Lock()


def _session(key: str) -> Session:
    with _LOCK:
        session = _SESSIONS.pop(key, None) or Session()
        _SESSIONS[key] = session
        while len(_SESSIONS) > 2000:
            _SESSIONS.popitem(last=False)
        return session


def _member_index(account_id: str, model: SemanticModel, db_config: dict[str, Any], *,
                  read: bool = True) -> MemberIndex:
    """Members of the attributes QueryBot may list, read under the service connection.

    Read once per model version and kept, then read again in the background every
    MEMBERS_HOURS, so customers and items added since are found by name. An attribute
    the admin hides or marks sensitive is left out at once; one newly allowed is read at
    once. ``read`` False (the admin turned value indexing off) reads nothing.
    """
    if not read:
        return MemberIndex()
    key = (account_id, model.version)
    may = frozenset(listable(model))
    stale = False
    with _LOCK:
        kept = _INDEXES.get(key)
        if kept is not None:
            _INDEXES.move_to_end(key)
            stale = not kept.refreshing and time.monotonic() - kept.read_at > MEMBERS_HOURS * 3600
            kept.refreshing = kept.refreshing or stale
    if kept is None or not may <= kept.listable:
        kept = _read_members(key, model, db_config, may)
    elif stale:
        threading.Thread(target=_read_again, args=(key, model, db_config, may), daemon=True,
                         name=f"core2-members-{account_id}").start()
    return (kept.index if may == kept.listable else kept.index.only(may)).named(model)


def _read_members(key: tuple[str, int], model: SemanticModel, db_config: dict[str, Any],
                  may: frozenset[str]) -> _Members:
    from sqlglot import exp

    from core2.warehouse import dialect as D
    from core2.warehouse.querybot import QueryBotWarehouse

    def fetch(slug: str) -> list[object]:
        column = model.columns[model.attributes[slug].column]
        table = model.tables[column.table]
        c = exp.column(D.ident(column.name, warehouse.dialect))
        sql = exp.select(c).distinct().from_(D.table_expr(table.database, table.schema_name, table.name,
                                                          warehouse.dialect)).where(
            exp.not_(exp.Is(this=c.copy(), expression=exp.Null()))).sql(dialect=warehouse.dialect)
        return [r[0] for r in warehouse.query(sql, max_rows=50_001).rows]

    with QueryBotWarehouse(str(db_config.get("db_type") or ""), db_config["credentials"]) as warehouse:
        kept = _Members(build_index(model, fetch, budget_seconds=30.0), may, time.monotonic())
    with _LOCK:
        _INDEXES[key] = kept
        _INDEXES.move_to_end(key)
        while len(_INDEXES) > 20:
            _INDEXES.popitem(last=False)
    return kept


def _read_again(key: tuple[str, int], model: SemanticModel, db_config: dict[str, Any],
                may: frozenset[str]) -> None:
    try:
        _read_members(key, model, db_config, may)
    except Exception as exc:  # noqa: BLE001 - the names already read stand, and are tried again after the same wait
        log.warning("core2: member names of %s could not be read again: %s", key[0], exc)
        with _LOCK:
            kept = _INDEXES.get(key)
            if kept is not None:
                kept.refreshing, kept.read_at = False, time.monotonic()


def read_members(account_id: str) -> int:
    """Read the workspace's member names now (at the end of Learn, and on the admin's Refresh values).

    Returns how many names were read; 0 when the workspace has no model, or its admin
    turned value indexing off.
    """
    import store
    from core.value_index import value_index_enabled
    from core2.bootstrap.service import answering_model

    client = store.get_client(account_id) or {}
    db_id = int(client.get("db_config_id") or 0)
    db_config = store.get_db_config(db_id) if db_id else None
    model = answering_model(account_id, client) if db_config else None
    if model is None or db_config is None or not value_index_enabled(store.get_client_state(account_id)):
        return 0
    return len(_read_members((account_id, model.version), model, db_config, frozenset(listable(model))).index.names)


def _allowed_model_tables(model: SemanticModel, allowed: set[str] | None) -> set[str] | None:
    """The reader's allowed tables (as QueryBot names them) as model table keys."""
    if allowed is None:
        return None
    wanted = {ids.norm(name) for name in allowed}
    out = set()
    for key, table in model.tables.items():
        names = {ids.norm(".".join(p for p in parts if p)) for parts in (
            (table.database, table.schema_name, table.name), (table.schema_name, table.name), (table.name,))}
        if names & wanted:
            out.add(key)
    return out


def question_scrubber(account_id: str) -> Callable[[str], str] | None:
    """Personal data out of a question, for a tenant under compliance; None for one that is not.

    The agent runtime's rule (core/agent_runtime.py): regulated, unprovisioned, or any
    mode but standard. It may widen, never narrow. The same tenants keep member values
    from the AI.
    """
    import store
    from core.compliance.policy_engine import is_regulated

    profile = store.get_compliance_profile(account_id) or {}
    if not (is_regulated(account_id) or str(profile.get("mode") or "standard").lower() != "standard"):
        return None
    from core.masking import scrub_question_pii

    industry = str(profile.get("industry") or "")
    return lambda text: scrub_question_pii(text, industry)[0]


def _portal_services(account_id: str, question: str, portal_user: dict[str, Any] | None,
                     question_id: str = "") -> tuple[Services | None, dict[str, Any] | None]:
    """The services a portal answer runs on, or the frame that says why there are none."""
    import store
    from core.schema import load_known_tables
    from core.value_index import value_index_enabled
    from core2.bootstrap.ai import workspace_planner
    from core2.bootstrap.service import answering_model, keep_decisions_current
    from core2.warehouse.governed import GovernedWarehouse

    client = store.get_client(account_id) or {}
    db_id = int(client.get("db_config_id") or 0)
    db_config = store.get_db_config(db_id) if db_id else None
    if not db_config:
        return None, _frame(question, "The new core has no database for this workspace yet.")
    keep_decisions_current(account_id, db_id, client)
    model = answering_model(account_id, client)
    if model is None:
        return None, _frame(question, "The new core has not learned this workspace yet: an admin can build it on "
                                      "the What QueryBot learned page.")
    state = store.get_client_state(account_id) or {}
    allowed = store.get_allowed_tables(portal_user) if portal_user else None
    warehouse = GovernedWarehouse(account_id, portal_user, db_config,
                                  known_tables=load_known_tables(state.get("schema_dir", "")),
                                  allowed_tables=allowed)
    scrub = question_scrubber(account_id)
    # The admin's "value indexing" switch, as today's pipeline reads it: off, no member
    # name is read from the warehouse or put before the AI.
    indexing = value_index_enabled(state)
    return Services(
        model=model, warehouse=warehouse,
        complete=workspace_planner(account_id, client, question=question, question_id=question_id),
        index=_member_index(account_id, model, db_config, read=indexing), today=dt.date.today(),
        values_allowed=scrub is None and indexing, allowed_tables=_allowed_model_tables(model, allowed),
        data_source=str(db_config.get("db_type") or ""), scrub=scrub), None


def portal_answer(account_id: str, question: str, portal_user: dict[str, Any] | None, *, session_key: str,
                  question_id: str = "") -> dict[str, Any]:
    """The new core's answer in the web portal (compare mode, or core2 mode)."""
    services, refused = _portal_services(account_id, question, portal_user, question_id)
    if services is None:
        return refused or _frame(question, "The new core cannot answer here yet.")
    return answer_question(question, services, _session(session_key), question_id=question_id)


def portal_replay(account_id: str, plan_data: dict[str, Any], portal_user: dict[str, Any] | None, *,
                  question: str = "") -> dict[str, Any]:
    """A pinned answer drawn again: its stored plan run on today's data, as the viewer is allowed to see it.

    No AI is asked: the plan is the one the answer was given from, so a dashboard tile keeps the
    answer's own shape (a comparison stays a dumbbell, a total a number). A window written
    relative to today ("last month") moves on with it.
    """
    services, refused = _portal_services(account_id, question, portal_user)
    if services is None:
        return refused or _frame(question, "The new core cannot answer here yet.")
    plan = parse_plan(plan_data)
    ctx = Context(today=services.today, allowed_tables=services.allowed_tables, max_rows=MAX_ROWS)
    return _compute(question, plan, services, ctx, question_id="", started=time.perf_counter())


def parse_plan(data: dict[str, Any]) -> Plan:
    """A stored plan back as a Plan (for replaying recorded answers)."""
    return Plan.model_validate(data)
