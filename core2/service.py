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
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core2 import ids
from core2.answer.builder import build_answer
from core2.compile.compiler import CompileError, compile_query
from core2.model.schema import SemanticModel
from core2.plan.ir import Clarify, Plan
from core2.plan.planner import Complete, Turn, plan_question
from core2.plan.values import MemberIndex, ValueMatch, build_index
from core2.resolve.resolver import Context, ResolveError, resolve
from core2.warehouse.runner import Warehouse

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


@dataclass
class Session:
    turns: list[Turn] = field(default_factory=list)


def _frame(question: str, text: str, **extra: Any) -> dict[str, Any]:
    answer = {"headline": text, "short_value": "", "comparison": "", "scope_badge": "", "scope_note": ""}
    return {"type": "assistant_response", "engine": "core2", "question": question, "answer": answer, "chart": None,
            "kpi": None, "data": None, "trust": {"engine": "core2"}, "confidence": {}, "insight_summary": "",
            "anomaly_callouts": [], "coverage_caveats": [], "follow_up_suggestions": [], **extra}


def _clarification(question: str, clarify: Clarify) -> dict[str, Any]:
    options = [o for o in clarify.options if o][:6]
    listed = "".join(f"\n• {o}" for o in options)
    return _frame(question, f"{clarify.question}{listed}", clarify={"about": clarify.about,
                                                                     "question": clarify.question, "options": options})


def _describe(question: str, model: SemanticModel) -> dict[str, Any]:
    subjects: dict[str, list[str]] = {}
    for m in sorted(model.measures.values(), key=lambda m: m.business_name):
        if not m.hidden and m.kind != "count":
            subjects.setdefault(model.tables[m.table].business_name, []).append(m.business_name.lower())
    lines = [f"• {subject}: {', '.join(names[:6])}{'…' if len(names) > 6 else ''}"
             for subject, names in sorted(subjects.items())]
    groups = sorted({e.business_name.lower() for e in model.entities.values()})
    text = "Here is what I can answer from this data:\n" + "\n".join(lines)
    if groups:
        text += f"\nBroken down by: {', '.join(groups[:12])}."
    return _frame(question, text)


_REFUSALS = {
    "denied": "That needs data you do not have access to: {message}.",
    "unconfirmed": "{message}. An admin can confirm it on the What QueryBot learned page.",
    "unsupported": "I cannot answer that from this data: {message}.",
    "unknown": "I could not find that in this data: {message}.",
    "empty": "{message}.",
}


def answer_question(question: str, services: Services, session: Session, *, question_id: str = "") -> dict[str, Any]:
    """The portal frame answering ``question``; the session keeps the plan for follow-ups."""
    start = time.perf_counter()
    model = services.model
    matches = _visible(services.index.match(question), model, services.allowed_tables)
    outcome = plan_question(model, question, services.complete, today=services.today,
                            history=session.turns[-HISTORY:], matches=matches,
                            values_allowed=services.values_allowed, scrub=services.scrub)
    plan = outcome.plan
    if plan.kind == "smalltalk":
        return _frame(question, "Hello! Ask me about your data, for example a total, a trend or a ranking.",
                      kind="smalltalk")
    if plan.kind == "describe_data":
        return _describe(question, model)
    if plan.kind == "unsupported":
        why = " ".join(plan.notes) or "nothing in this data measures it"
        return _frame(question, f"I cannot answer that from this data: {why}", unsupported=True)
    if plan.kind == "clarify" and plan.clarify is not None:
        # Kept in the conversation: the reply ("the first one") answers this question.
        session.turns.append(Turn(question, plan))
        del session.turns[:-HISTORY]
        return _clarification(question, plan.clarify)

    try:
        logical = resolve(plan, model, Context(today=services.today, allowed_tables=services.allowed_tables,
                                               max_rows=MAX_ROWS))
    except ResolveError as exc:
        if exc.kind == "ambiguous":
            session.turns.append(Turn(question, plan))     # the reply names the role this asks for
            del session.turns[:-HISTORY]
            return _clarification(question, Clarify(about="path", question=f"{exc.message}: which one?",
                                                    options=exc.options))
        template = _REFUSALS.get(exc.kind, "{message}.")
        return _frame(question, template.format(message=exc.message), unsupported=exc.kind == "unsupported")
    try:
        compiled = compile_query(logical, model, services.warehouse.dialect)
    except (CompileError, ValueError) as exc:
        log.warning("core2 could not compile a plan for %r: %s", question, exc)
        return _frame(question, f"I cannot answer that from this data yet: {exc}.", unsupported=True)
    try:
        result = services.warehouse.query(compiled.sql, max_rows=compiled.row_cap)
    except Exception as exc:     # noqa: BLE001 - a refused or failed query is told, never raised to the socket
        log.warning("core2 query failed for %r: %s", question, exc)
        reason = str(exc).split(":", 1)[-1].strip() if "Policy" in type(exc).__name__ else "the database refused it"
        return _frame(question, f"The question was understood but the query could not run ({reason}).",
                      trust={"engine": "core2", "sql": compiled.sql})
    payload = build_answer(question, logical, compiled, result.columns, result.rows,
                           duration_ms=(time.perf_counter() - start) * 1000, data_source=services.data_source,
                           question_id=question_id, model_version=model.version, truncated=result.truncated)
    payload["plan"] = plan.model_dump(mode="json", exclude_defaults=True)
    if outcome.repaired:
        payload["trust"]["plan_repaired"] = True
    session.turns.append(Turn(question, plan))
    del session.turns[:-HISTORY]
    return payload


def _visible(matches: list[ValueMatch], model: SemanticModel, allowed: set[str] | None) -> list[ValueMatch]:
    """Member names only from tables the reader may use."""
    if allowed is None:
        return matches
    return [m for m in matches
            if m.attribute in model.attributes and model.columns[model.attributes[m.attribute].column].table in allowed]


# ── the portal's wiring ────────────────────────────────────────────────────

_INDEXES: OrderedDict[tuple[str, int], MemberIndex] = OrderedDict()
_SESSIONS: OrderedDict[str, Session] = OrderedDict()
_LOCK = threading.Lock()


def _session(key: str) -> Session:
    with _LOCK:
        session = _SESSIONS.pop(key, None) or Session()
        _SESSIONS[key] = session
        while len(_SESSIONS) > 2000:
            _SESSIONS.popitem(last=False)
        return session


def _member_index(account_id: str, model: SemanticModel, db_config: dict[str, Any]) -> MemberIndex:
    """Members of the attributes QueryBot may list, read once per model version under the service connection."""
    key = (account_id, model.version)
    with _LOCK:
        if key in _INDEXES:
            _INDEXES.move_to_end(key)
            return _INDEXES[key]
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
        index = build_index(model, fetch, budget_seconds=30.0)
    with _LOCK:
        _INDEXES[key] = index
        while len(_INDEXES) > 20:
            _INDEXES.popitem(last=False)
    return index


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


def portal_answer(account_id: str, question: str, portal_user: dict[str, Any] | None, *, session_key: str,
                  question_id: str = "") -> dict[str, Any]:
    """The new core's answer in the web portal (compare mode, or core2 mode)."""
    import store
    from core.compliance.policy_engine import is_regulated
    from core.schema import load_known_tables
    from core2.bootstrap.ai import workspace_planner
    from core2.bootstrap.service import load_model
    from core2.warehouse.governed import GovernedWarehouse

    client = store.get_client(account_id) or {}
    db_id = int(client.get("db_config_id") or 0)
    db_config = store.get_db_config(db_id) if db_id else None
    if not db_config:
        return _frame(question, "The new core has no database for this workspace yet.")
    model = load_model(account_id, db_id)
    if model is None:
        return _frame(question, "The new core has not learned this workspace yet: an admin can build it on the "
                                "What QueryBot learned page.")
    state = store.get_client_state(account_id) or {}
    allowed = store.get_allowed_tables(portal_user) if portal_user else None
    warehouse = GovernedWarehouse(account_id, portal_user, db_config,
                                  known_tables=load_known_tables(state.get("schema_dir", "")),
                                  allowed_tables=allowed)
    profile = store.get_compliance_profile(account_id) or {}
    scrub = None
    if profile.get("mode") == "regulated":
        from core.masking import scrub_question_pii

        industry = str(profile.get("industry") or "")
        scrub = lambda text: scrub_question_pii(text, industry)[0]   # noqa: E731
    services = Services(
        model=model, warehouse=warehouse, complete=workspace_planner(account_id, client, question=question),
        index=_member_index(account_id, model, db_config), today=dt.date.today(),
        values_allowed=not is_regulated(account_id), allowed_tables=_allowed_model_tables(model, allowed),
        data_source=str(db_config.get("db_type") or ""), scrub=scrub)
    return answer_question(question, services, _session(session_key), question_id=question_id)


def parse_plan(data: dict[str, Any]) -> Plan:
    """A stored plan back as a Plan (for replaying recorded answers)."""
    return Plan.model_validate(data)
