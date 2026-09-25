"""
core/readiness_gate.py

Is this workspace ready for its readers' questions?

core/model_readiness.py orders what to model next; nothing said when the
modelling is done. The gate does, from what the workspace already records,
and says READY only when, for the facts in the model:

1. each has a checked time axis: an approved business date, and where that
   date is reached through the date table, a join to it that was checked
   against the data or confirmed by the admin;
2. their joins are checked or confirmed: every other join between the model's
   tables was probed against the data -- not broken, not matching nothing,
   not waiting for review since its tables changed -- or confirmed;
3. no date role is ambiguous: asked a period question, the date resolver the
   pipeline uses settles each fact's date without asking the reader which;
4. the key metrics are certified: each fact has a metric, and every live
   metric was certified by the admin (store.certify_metric) -- a metric whose
   formula no longer validates cannot be.

Otherwise it lists what blocks it, each with what to do and where. Like the
backlog it is advice: it never raises, and it never stops a question.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger("querybot.readiness_gate")

# The checks, in the order an admin works through them.
CHECKS = ("model", "time_axis", "joins", "date_roles", "metrics")

# A period question, as the resolver reads one: whether it can settle each
# fact's date without asking is what "not ambiguous" means.
_PERIOD_QUESTION = "total by month"


@dataclass(frozen=True)
class Blocker:
    """One thing that keeps the workspace from being ready."""

    check: str      # one of CHECKS
    subject: str    # the fact, join or metric it is about
    problem: str    # what is wrong
    remedy: str     # what to do about it
    where: str      # the admin page it is done on: date_roles, graph, metrics, setup


@dataclass(frozen=True)
class GateVerdict:
    facts: tuple[str, ...] = ()
    blockers: tuple[Blocker, ...] = ()

    @property
    def ready(self) -> bool:
        return not self.blockers

    def passed(self, check: str) -> bool:
        return not any(blocker.check == check for blocker in self.blockers)

    def by_check(self) -> dict[str, list[Blocker]]:
        grouped: dict[str, list[Blocker]] = {check: [] for check in CHECKS}
        for blocker in self.blockers:
            grouped.setdefault(blocker.check, []).append(blocker)
        return grouped


def _fact_name(entity: dict) -> str:
    table = str(entity.get("table_name") or entity.get("entity_name") or "")
    schema = str(entity.get("schema_name") or "")
    return f"{schema}.{table}" if schema else table


def _join_name(rel: dict) -> str:
    return (f"{rel.get('from_entity')}.{rel.get('from_column')} → "
            f"{rel.get('to_entity')}.{rel.get('to_column')}")


def _join_problem(rel: dict) -> tuple[str, str]:
    """(problem, remedy) for a join that is not checked, or ("", "")."""
    validation = str(rel.get("validation_status") or "")
    if validation == "broken":
        return ("is broken: a column it joins on is missing",
                "Fix the join or remove it.")
    if str(rel.get("join_multiplicity") or "") == "zero_match":
        return ("matches nothing: no key on one side is found on the other",
                "Check the keys, then fix the join or remove it.")
    if validation == "needs_review":
        return ("changed tables since it was checked",
                "Check it against the data again, or confirm it.")
    if str(rel.get("status") or "") == "confirmed" or str(rel.get("last_profiled_at") or ""):
        return "", ""
    return ("has not been checked against the data",
            "Check it against the data (Validate on the graph), or confirm it.")


def _date_joins(fact_entity: str, role: dict, relationships: list[dict], tables: dict[str, str]) -> list[dict]:
    """The joins from a date role's key to its date table. Discovery points a
    role-playing date at an entity of its own ("Invoice Date") on the date
    table, so the target is matched by the table an entity reads, not its name."""
    from core.contextual_dates import same_date_fact

    column = str(role.get("fact_column") or "").upper()
    target = str(role.get("dimension_table") or "")
    return [
        rel for rel in relationships
        if str(rel.get("from_entity") or "").upper() == fact_entity.upper()
        and str(rel.get("from_column") or "").upper() == column
        and same_date_fact(tables.get(str(rel.get("to_entity") or ""), rel.get("to_entity")), target)
    ]


def check_readiness(account_id: str) -> GateVerdict:
    """The gate's verdict for one workspace. Never raises."""
    try:
        return _check(account_id)
    except Exception as exc:  # noqa: BLE001 - advice that breaks the page is worse than none
        log.warning("Readiness gate failed for %s: %s", account_id, exc, exc_info=True)
        return GateVerdict(blockers=(Blocker(
            "model", "", "The readiness check could not run", "See the service log.", "setup"),))


def _check(account_id: str) -> GateVerdict:
    import store
    from core.contextual_dates import _role_is_complete, resolve_contextual_date_binding, same_date_fact
    from core.date_roles import normalize_date_key_type
    from core.metric_coverage import fact_date_roles_for

    # Rejecting an entity or a join deactivates it: the active ones are the model.
    entities = store.list_entities(account_id, active_only=True)
    facts = [entity for entity in entities if str(entity.get("entity_type") or "") == "fact"]
    if not facts:
        return GateVerdict(blockers=(Blocker(
            "model", "", "No fact table is in the model",
            "Select the tables questions are about and discover the schema.", "setup"),))

    tables = {str(e.get("entity_name") or ""): _fact_name(e) for e in entities}
    # A join stays active when an entity it reaches is rejected; it is out of the model all the same.
    relationships = [
        rel for rel in store.list_relationships(account_id, active_only=True)
        if rel.get("from_entity") in tables and rel.get("to_entity") in tables
    ]
    roles = fact_date_roles_for(account_id)
    metrics = store.list_metrics(account_id)
    contexts: dict[int, list[dict]] = {}
    for row in store.list_metric_date_contexts(account_id):
        contexts.setdefault(int(row.get("metric_id") or 0), []).append(dict(row))

    blockers: list[Blocker] = []
    time_axis_joins: set[int] = set()
    names = []
    for fact in facts:
        name = _fact_name(fact)
        names.append(name)
        entity = str(fact.get("entity_name") or fact.get("table_name") or "")

        # 1. A checked time axis.
        approved = [
            role for role in roles
            if same_date_fact(role.get("fact_table"), name)
            and str(role.get("status") or "") == "approved" and _role_is_complete(role)
        ]
        if not approved:
            blockers.append(Blocker(
                "time_axis", name, f"{name} has no approved business date",
                "Approve the date its rows are counted on.", "date_roles"))
        for role in approved:
            if normalize_date_key_type(role.get("date_key_type") or "surrogate_fk") != "surrogate_fk":
                continue    # a date the fact carries itself is joined to nothing
            label = f"{name}.{role.get('fact_column')}"
            joins = _date_joins(entity, role, relationships, tables)
            if not joins:
                blockers.append(Blocker(
                    "time_axis", label,
                    f"{label} has no join to the date table {role.get('dimension_table')}",
                    "Add the join to the date table, or rediscover the schema.", "graph"))
                continue
            for join in joins:
                time_axis_joins.add(int(join.get("id") or 0))
                problem, remedy = _join_problem(join)
                if problem:
                    blockers.append(Blocker(
                        "time_axis", label, f"The join from {label} to the date table {problem}",
                        remedy, "graph"))

        # 3. No ambiguous date role -- asked of the resolver the pipeline uses,
        # for a question naming no date, alone and with each of the fact's metrics.
        on_fact = [m for m in metrics if same_date_fact(m.get("base_table"), name)]
        undecided = []
        for metric in [None, *on_fact]:
            verdict = resolve_contextual_date_binding(
                _PERIOD_QUESTION,
                matched_metrics=[dict(metric)] if metric else [],
                bindings=contexts.get(int(metric.get("id") or 0), []) if metric else [],
                date_roles=[dict(role) for role in roles],
                required_fact_tables={name},
            )
            if str(verdict.get("status") or "") == "ambiguous":
                undecided.append(str(metric.get("name")) if metric else "questions naming no metric")
        if undecided:
            blockers.append(Blocker(
                "date_roles", name,
                f"{name} has more than one business date and none is its default "
                f"(asked of {', '.join(undecided)})",
                "Choose the date questions about it are counted on by default.", "date_roles"))

        # 4a. A metric to answer with.
        if not on_fact:
            blockers.append(Blocker(
                "metrics", name, f"{name} has no metric",
                "Define one, or accept a proposed one.", "metrics"))

    # 2. Every other join.
    for rel in relationships:
        if int(rel.get("id") or 0) in time_axis_joins:
            continue
        problem, remedy = _join_problem(rel)
        if problem:
            blockers.append(Blocker("joins", _join_name(rel), f"{_join_name(rel)} {problem}", remedy, "graph"))

    # 4b. Every live metric certified.
    for metric in metrics:
        metric_name = str(metric.get("name") or "")
        if str(metric.get("metric_status") or "") == "draft":
            blockers.append(Blocker(
                "metrics", metric_name, f"{metric_name}'s formula does not validate",
                "Fix its formula, then check its numbers and certify it.", "metrics"))
        elif not str(metric.get("certified_at") or ""):
            blockers.append(Blocker(
                "metrics", metric_name, f"{metric_name} is not certified",
                "Check its numbers against a figure the business trusts, then certify it.", "metrics"))

    return GateVerdict(facts=tuple(names), blockers=tuple(blockers))
