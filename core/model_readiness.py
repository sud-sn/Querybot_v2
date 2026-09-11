"""What to model next, in the order that fixes the most questions.

The goal was never to remove data modelling — it is to stop the product
needing a modeller to guess. Four separate scores already exist
(``core.kb_quality``, ``core.curation_weight``, ``core.graph_health``,
``core.compliance.readiness``), each true and none of them answering the only
question an admin actually has: *what do I do next, and what will it fix?*

This module answers that. It reads what the workspace declares, turns each
gap into an item with a named remedy, and orders them by how many failing
question shapes the remedy would resolve — sourced from the metric coverage
reports rather than from a weighting somebody invented. Modelling becomes
review of a shrinking list rather than open-ended authoring.

It also computes ``metadata_version``: a hash over the schema, graph, metric
and curation state that answers are stamped with. Two answers to the same
question that disagree because the model changed between them are otherwise
indistinguishable from two answers that disagree because one is wrong.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field

log = logging.getLogger("querybot.model_readiness")

# What each kind of gap is worth when nothing else distinguishes two items.
# Only a tie-break: the primary ordering is how many question shapes the
# remedy unblocks, which is measured rather than assumed.
# How many undescribed tables to list at once. The rest are counted, not
# enumerated: a backlog longer than an afternoon is a backlog nobody starts.
MAX_CURATION_ITEMS = 20

_KIND_WEIGHT: dict[str, int] = {
    "date_role": 5,
    "dimensions": 4,
    "described": 3,
    "synonym": 3,
    "grain": 2,
    "column_synonyms": 1,
}


@dataclass(frozen=True)
class BacklogItem:
    """One thing to fix, and what fixing it buys."""

    kind: str
    subject: str
    remedy: str
    unblocks: int = 0
    detail: str = ""

    @property
    def weight(self) -> int:
        return _KIND_WEIGHT.get(self.kind, 1)


@dataclass(frozen=True)
class ReadinessReport:
    items: tuple[BacklogItem, ...] = ()
    tables_described: int = 0
    tables_total: int = 0
    metrics_complete: int = 0
    metrics_total: int = 0
    metadata_version: str = ""

    @property
    def total_unblocked(self) -> int:
        return sum(item.unblocks for item in self.items)

    def by_kind(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.items:
            counts[item.kind] = counts.get(item.kind, 0) + 1
        return counts


# ══════════════════════════════════════════════════════════════════════════════
# The backlog
# ══════════════════════════════════════════════════════════════════════════════

def metric_backlog(account_id: str, *, lang: str = "en") -> list[BacklogItem]:
    """Gaps drawn from what each metric cannot answer.

    ``unblocks`` is a count of real question shapes, taken from
    ``core.metric_coverage`` — not a score. An admin deciding between two
    afternoon's work should be comparing outcomes, not weights.
    """
    import store

    from core.metric_coverage import coverage_report, fact_date_roles_for

    try:
        metrics = store.list_metrics(account_id)
    except Exception as exc:
        log.warning("Metrics unavailable for readiness on %s: %s", account_id, exc)
        return []

    try:
        contexts = store.list_metric_date_contexts(account_id)
    except Exception as exc:
        log.warning("Date contexts unavailable for readiness on %s: %s",
                    account_id, exc)
        contexts = []
    roles_by_metric: dict[int, list[dict]] = {}
    for row in contexts:
        roles_by_metric.setdefault(int(row.get("metric_id") or 0), []).append(row)
    # The fact's own approved Date Roles. Most workspaces govern dates here and
    # never create a metric date context at all; without these, every metric in
    # such a workspace is reported as having no business date.
    fact_roles = fact_date_roles_for(account_id)

    items: list[BacklogItem] = []
    for metric in metrics:
        name = str(metric.get("name") or "")
        if not name:
            continue
        report = coverage_report(
            metric,
            metric_date_contexts=roles_by_metric.get(
                int(metric.get("id") or 0), []),
            fact_date_roles=fact_roles,
            lang=lang,
        )
        # One item per REMEDY, not per failing question: an admin binds one
        # date role and eleven shapes start working, and listing those eleven
        # separately makes a two-item afternoon look like a twelve-item week.
        by_remedy: dict[tuple[str, str], list] = {}
        for gap in report.gaps:
            by_remedy.setdefault((gap.missing, gap.reason), []).append(gap)
        for (missing, reason), gaps in by_remedy.items():
            items.append(BacklogItem(
                kind=_kind_for(missing),
                subject=name,
                remedy=reason,
                unblocks=len(gaps),
                detail=gaps[0].variation.question,
            ))
    return items


def _kind_for(missing: str) -> str:
    if missing in _KIND_WEIGHT:
        return missing
    # A named dimension ("Salesperson") rather than a category.
    return "dimensions"


def curation_backlog(account_id: str) -> list[BacklogItem]:
    """Tables nobody has described, ordered by how raw they are.

    ``unblocks`` is 0 here on purpose: an undescribed table does not block a
    specific question shape the way a missing date role does — it degrades
    every answer that touches it, which is real but not countable. Ordering
    therefore falls to the kind weight, below every measured item.
    """
    from core.curation_weight import signals_for

    try:
        signals = signals_for(account_id)
    except Exception as exc:
        log.warning("Curation signals unavailable for %s: %s", account_id, exc)
        return []

    items: list[BacklogItem] = []
    for table, flags in sorted(signals.items()):
        if not flags.get("described"):
            items.append(BacklogItem(
                kind="described", subject=table,
                remedy="Describe what one row of this table means.",
            ))
        elif not flags.get("synonyms"):
            # Only once the table is described: telling someone to add
            # synonyms to a table nobody has explained is the wrong order.
            items.append(BacklogItem(
                kind="synonym", subject=table,
                remedy="Add the words people use for this table.",
            ))
    # A workspace with a hundred raw tables does not need a hundred-item
    # list; it needs the next few, and the summary still reports the total.
    return items[:MAX_CURATION_ITEMS]


def rank_backlog(items: list[BacklogItem]) -> list[BacklogItem]:
    """Most-unblocking first, then by kind, then stably by subject."""
    return sorted(
        items,
        key=lambda i: (-i.unblocks, -i.weight, i.kind, i.subject),
    )


def build_report(account_id: str, *, lang: str = "en") -> ReadinessReport:
    """The whole backlog for one workspace, ordered by what to do next."""
    items = rank_backlog(metric_backlog(account_id, lang=lang)
                         + curation_backlog(account_id))

    described = total_tables = 0
    try:
        from core.curation_weight import signals_for
        signals = signals_for(account_id)
        total_tables = len(signals)
        described = sum(1 for flags in signals.values() if flags.get("described"))
    except Exception as exc:
        log.warning("Curation coverage unavailable for %s: %s", account_id, exc)

    complete = total_metrics = 0
    try:
        import store

        from core.metric_coverage import coverage_report, fact_date_roles_for
        metrics = store.list_metrics(account_id)
        contexts = store.list_metric_date_contexts(account_id)
        roles: dict[int, list[dict]] = {}
        for row in contexts:
            roles.setdefault(int(row.get("metric_id") or 0), []).append(row)
        fact_roles = fact_date_roles_for(account_id)
        total_metrics = len(metrics)
        complete = sum(
            1 for m in metrics
            if coverage_report(
                m,
                metric_date_contexts=roles.get(int(m.get("id") or 0), []),
                fact_date_roles=fact_roles,
                lang=lang).complete
        )
    except Exception as exc:
        log.warning("Metric coverage unavailable for %s: %s", account_id, exc)

    return ReadinessReport(
        items=tuple(items),
        tables_described=described,
        tables_total=total_tables,
        metrics_complete=complete,
        metrics_total=total_metrics,
        metadata_version=metadata_version(account_id),
    )


# ══════════════════════════════════════════════════════════════════════════════
# The freshness contract
# ══════════════════════════════════════════════════════════════════════════════

def metadata_version(account_id: str) -> str:
    """A stable fingerprint of everything an answer depends on.

    Stamped on an answer so two answers to the same question that disagree
    because the model changed between them can be told apart from two that
    disagree because one is wrong. Covers the graph, the metric registry and
    the curation state — the assets whose edits change what SQL gets written.

    Returns "" when it cannot be computed, and the caller then makes no
    freshness claim at all: an invented version is worse than none, because
    it asserts sameness that was never checked.
    """
    import store

    try:
        entities = store.list_entities(account_id, active_only=False)
        relationships = store.list_relationships(account_id, active_only=False)
        metrics = store.list_metrics(account_id, active_only=False)
        descriptions = store.list_table_descriptions(account_id)
    except Exception as exc:
        log.warning("metadata_version unavailable for %s: %s", account_id, exc)
        return ""

    payload = {
        "entities": sorted(
            f"{e.get('entity_name')}|{e.get('table_name')}|{e.get('entity_type')}"
            f"|{e.get('is_active')}" for e in entities
        ),
        "relationships": sorted(
            f"{r.get('from_entity')}.{r.get('from_column')}"
            f"->{r.get('to_entity')}.{r.get('to_column')}|{r.get('join_type')}"
            f"|{r.get('is_active')}" for r in relationships
        ),
        "metrics": sorted(
            f"{m.get('name')}|{m.get('sql_template')}|{m.get('allowed_dimensions')}"
            f"|{m.get('grain')}|{m.get('is_active')}" for m in metrics
        ),
        "descriptions": sorted(
            f"{table}|{(row or {}).get('description', '')[:200]}"
            f"|{(row or {}).get('synonyms', '')}"
            for table, row in descriptions.items()
        ),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def is_stale(stamped: str, account_id: str) -> bool:
    """Has the model changed since an answer was stamped?

    False when either side is unknown. A freshness warning nobody can verify
    is noise, and noise in a governance surface is worse than silence.
    """
    if not stamped:
        return False
    current = metadata_version(account_id)
    if not current:
        return False
    return stamped != current
