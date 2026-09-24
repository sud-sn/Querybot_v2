"""Starter stock metrics, proposed -- never made live -- from the semantic model.

A warehouse nobody has modelled has no metrics, so its first readers get raw
columns: ON_HND_QTY, CUR_ON_HND_QTY, ALC_ON_HND_QTY, ITM_CST. The semantic
model already knows which of those are stock levels, which are movements and
which are unit costs (core.analysis_contract), and a periodic snapshot of stock
has a few questions everyone asks of it: how much is on hand, what it is worth,
how much of it is allocated and how much is free, how much came in and how
much went out.

Each is filed as a PENDING proposal in the administrator's metric queue with
what it measures, the evidence for every column it uses, and English and French
synonyms. Nothing changes an answer until an administrator accepts it. A name
that is already a metric, is waiting in the queue, or was suggested before and
turned down is not suggested again, so a rebuild is quiet.

Tenant-neutral: columns are chosen by what their names mean, never by a
tenant's table or column names.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from core.analysis_contract import is_on_hand_quantity, measure_additivity

log = logging.getLogger("querybot.starter_metrics")

GENERATED_BY = "starter_model"

_TOKEN_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+|(?<=[a-z0-9])(?=[A-Z])")

# A part of the stock rather than the stock: allocated, reserved, on
# inspection, rejected, on hold, consigned, in transit, back-ordered.
_PART_OF_STOCK_TOKENS = frozenset({
    "ALC", "ALLOC", "ALLOCATED", "RSV", "RES", "RESERVED", "ISP", "INSP",
    "INSPECTION", "RJC", "REJ", "REJECTED", "HLD", "HOLD", "CSG", "CONSIGNED",
    "TRN", "TRANSIT", "BCK", "BACKORDER",
})
# A figure about a quantity rather than the quantity: an average, a minimum or
# maximum, a percentage, a flag, a count of SKUs.
_FIGURE_TOKENS = frozenset({
    "AVG", "AVERAGE", "MIN", "MAX", "PCT", "PERCENT", "RATE", "RATIO",
    "IND", "FLG", "FLAG", "SKU", "CNT", "COUNT", "DAYS", "DYS",
})
# A position other than the one at the snapshot: opening, prior period, last
# year, a target, a forecast, a budget, a plan.
_OTHER_POSITION_TOKENS = frozenset({
    "BEG", "BEGIN", "BEGINNING", "BGN", "OPN", "OPENING", "PRV", "PREV", "PRIOR",
    "LST", "LAST", "LY", "PY", "TGT", "TARGET", "FCST", "FORECAST", "BUD",
    "BUDGET", "PLN", "PLAN",
})
_ALLOCATED_TOKENS = frozenset({"ALC", "ALLOC", "ALLOCATED"})
_COST_TOKENS = frozenset({"CST", "COST"})
_AVERAGE_TOKENS = frozenset({"AVG", "AVERAGE"})
_PURCHASE_TOKENS = frozenset({"PCH", "PURCH", "PURCHASE", "PURCHASED", "PURCHASES"})
_SOLD_TOKENS = frozenset({"SLD", "SOLD"})
_QUANTITY_TOKENS = frozenset({"QTY", "QUANTITY", "UNITS"})
# The position at the end of the snapshot's period, preferred when a fact
# also carries an unqualified one.
_CURRENT_TOKENS = frozenset({"CUR", "CURRENT", "END", "ENDING", "CLOSING", "EOP", "EOM"})


def _tokens(name: str) -> set[str]:
    return {part.upper() for part in _TOKEN_SPLIT_RE.split(str(name or "")) if part}


@dataclass(frozen=True)
class StarterMetric:
    key: str
    name: str
    sql_template: str
    base_table: str
    required_columns: tuple[str, ...]
    synonyms: tuple[str, ...]
    description: str
    evidence: tuple[str, ...]
    result_format: str = "number"
    confidence: int = 70

    def as_metric(self) -> dict[str, Any]:
        """metric_registry-shaped, as the accept route saves it."""
        return {
            "name": self.name,
            "synonyms": ", ".join(self.synonyms),
            "description": self.description,
            "sql_template": self.sql_template,
            "formula_type": "expression",
            "result_format": self.result_format,
            "required_columns": ", ".join(self.required_columns),
            "base_table": self.base_table,
            "category": "inventory",
        }


def _pick(candidates: list[str], prefer: frozenset[str] = frozenset()) -> str:
    """The candidate carrying a preferred token, then the shortest name."""
    if not candidates:
        return ""
    return sorted(
        candidates,
        key=lambda name: (0 if _tokens(name) & prefer else 1, len(name), name),
    )[0]


def _others(chosen: str, candidates: list[str]) -> str:
    rest = [name for name in candidates if name != chosen]
    return f" Also on this table: {', '.join(sorted(rest))}." if rest else ""


def _on_hand(columns: list[str]) -> tuple[str, list[str]]:
    candidates = [
        name for name in columns
        if is_on_hand_quantity(name)
        and measure_additivity(name)[1] == "balance"
        and not _tokens(name) & (_PART_OF_STOCK_TOKENS | _FIGURE_TOKENS | _OTHER_POSITION_TOKENS)
    ]
    return _pick(candidates, _CURRENT_TOKENS), candidates


def _unit_cost(columns: list[str]) -> tuple[str, list[str]]:
    candidates = [
        name for name in columns
        if measure_additivity(name)[1] == "unit_value"
        and _tokens(name) & _COST_TOKENS
        and not _tokens(name) & _OTHER_POSITION_TOKENS
    ]
    # The standing unit cost before an average one; which of them values the
    # stock is the administrator's call, and the evidence names both.
    standing = [name for name in candidates if not _tokens(name) & _AVERAGE_TOKENS]
    return _pick(standing) or _pick(candidates), candidates


def _allocated(columns: list[str]) -> tuple[str, list[str]]:
    candidates = [
        name for name in columns
        if _tokens(name) & _ALLOCATED_TOKENS
        and _tokens(name) & _QUANTITY_TOKENS
        and not _tokens(name) & (_FIGURE_TOKENS | _OTHER_POSITION_TOKENS)
    ]
    # The allocated part OF the stock on hand, so that on hand less allocated
    # is what is free; then the position at the snapshot.
    on_hand = [name for name in candidates if is_on_hand_quantity(name)]
    return _pick(on_hand or candidates, _CURRENT_TOKENS), candidates


def _movement(columns: list[str], kind: frozenset[str]) -> tuple[str, list[str]]:
    candidates = [
        name for name in columns
        if measure_additivity(name)[1] == "flow"
        and _tokens(name) & kind
        and _tokens(name) & _QUANTITY_TOKENS
        and not _tokens(name) & (_FIGURE_TOKENS | _OTHER_POSITION_TOKENS)
    ]
    return _pick(candidates), candidates


_LEVEL_NOTE = (
    "A level: it adds up across items and warehouses, never across {span} -- a "
    "question over a range reads the latest {unit} in it."
)
_MOVEMENT_NOTE = "A movement: it adds up across periods."


def _metrics_for(base: str, columns: list[str], monthly: bool) -> list[StarterMetric]:
    """Every starter metric one periodic snapshot fact supports."""
    span, unit = ("months", "month") if monthly else ("days or periods", "snapshot")
    level = _LEVEL_NOTE.format(span=span, unit=unit)
    at = "at the end of each month" if monthly else "at the snapshot"
    on_hand, on_hand_all = _on_hand(columns)
    cost, cost_all = _unit_cost(columns)
    allocated, allocated_all = _allocated(columns)
    purchased, purchased_all = _movement(columns, _PURCHASE_TOKENS)
    sold, sold_all = _movement(columns, _SOLD_TOKENS)

    on_hand_why = (
        f"{on_hand}: stock on hand by name, not qualified as allocated, reserved, "
        f"on inspection or rejected, nor an average or an earlier position."
        + _others(on_hand, on_hand_all)
    )
    found: list[StarterMetric] = []
    if on_hand:
        found.append(StarterMetric(
            "on_hand", "Stock on hand", f"SUM({on_hand})", base, (on_hand,),
            ("stock on hand", "on hand", "quantity on hand", "inventory on hand",
             "units in stock", "quantité en stock", "stock en main",
             "quantité en main", "inventaire en main", "stock physique"),
            f"Units in stock {at} ({on_hand}). {level}",
            (on_hand_why,), confidence=75,
        ))
    if on_hand and cost:
        found.append(StarterMetric(
            "inventory_value", "Inventory value", f"SUM({on_hand} * {cost})",
            base, (on_hand, cost),
            ("inventory value", "stock value", "value of stock", "stock valuation",
             "inventory valuation", "valeur du stock", "valeur des stocks",
             "valeur de l'inventaire", "valorisation du stock"),
            f"Stock on hand valued at its unit cost ({on_hand} x {cost}), {at}. {level}",
            (on_hand_why,
             f"{cost}: a cost per unit, never summed on its own; confirm it is the "
             f"cost this business values stock at." + _others(cost, cost_all)),
            result_format="currency", confidence=60,
        ))
    if allocated:
        found.append(StarterMetric(
            "allocated", "Allocated quantity", f"SUM({allocated})", base, (allocated,),
            ("allocated quantity", "allocated stock", "quantity allocated",
             "quantité allouée", "stock alloué"),
            f"Units of stock allocated to orders {at} ({allocated}). {level}",
            (f"{allocated}: allocated stock by name." + _others(allocated, allocated_all),),
        ))
    if on_hand and allocated:
        found.append(StarterMetric(
            "available", "Available quantity",
            f"SUM({on_hand}) - COALESCE(SUM({allocated}), 0)", base, (on_hand, allocated),
            ("available quantity", "available stock", "quantity available",
             "stock available", "quantité disponible", "stock disponible"),
            f"Stock on hand not allocated to orders ({on_hand} less {allocated}), "
            f"{at}. Stock on inspection or rejected is not subtracted; change the "
            f"formula if this business excludes it. {level}",
            (on_hand_why,
             f"{allocated}: allocated stock by name." + _others(allocated, allocated_all),
             "Available = on hand less allocated is the common definition, not "
             "one read from this warehouse."),
            confidence=50,
        ))
    if purchased:
        found.append(StarterMetric(
            "purchased", "Purchased quantity", f"SUM({purchased})", base, (purchased,),
            ("purchased quantity", "quantity purchased", "units purchased",
             "quantité achetée", "unités achetées"),
            f"Units purchased in the period ({purchased}). {_MOVEMENT_NOTE}",
            (f"{purchased}: a purchase movement by name." + _others(purchased, purchased_all),),
            confidence=75,
        ))
    if sold:
        found.append(StarterMetric(
            "sold", "Units sold", f"SUM({sold})", base, (sold,),
            ("units sold", "quantity sold", "sold quantity",
             "quantité vendue", "unités vendues"),
            f"Units sold in the period ({sold}). {_MOVEMENT_NOTE}",
            (f"{sold}: a sales movement by name." + _others(sold, sold_all),),
            confidence=75,
        ))
    return found


# A fact keyed by a yyyymm period holds month-end positions: its levels are
# named for the month's end, so they never share a name with the daily ones.
_MONTH_END: dict[str, tuple[str, tuple[str, ...]]] = {
    "on_hand": ("Month-end stock on hand", (
        "month-end stock on hand", "month-end stock", "stock at month end",
        "month end inventory", "stock en fin de mois", "stock fin de mois",
        "inventaire de fin de mois")),
    "inventory_value": ("Month-end inventory value", (
        "month-end inventory value", "month-end stock value",
        "stock value at month end", "valeur du stock en fin de mois",
        "valeur du stock fin de mois")),
    "allocated": ("Month-end allocated quantity", (
        "month-end allocated quantity", "allocated at month end",
        "quantité allouée en fin de mois")),
    "available": ("Month-end available quantity", (
        "month-end available quantity", "available at month end",
        "quantité disponible en fin de mois")),
}


def _period_keyed_tables(model: dict) -> set[str]:
    from core.period_rows import period_row_policies

    return {
        str(policy.get("fact_table") or "").upper()
        for policy in period_row_policies(model)
    }


def starter_metrics(model: dict | None) -> list[StarterMetric]:
    """The starter metrics a model's periodic snapshot facts support.

    Only a fact the model calls a periodic snapshot, and only a metric whose
    every column was found by meaning. Two facts supporting the same metric
    never share a name or a synonym: the second is named for its table and
    carries no synonyms, since one phrase resolving to two metrics makes every
    question that uses it ambiguous.
    """
    model = model or {}
    period_keyed = _period_keyed_tables(model)
    metrics: list[StarterMetric] = []
    names: set[str] = set()
    for table in model.get("tables") or []:
        if not isinstance(table, dict):
            continue
        if table.get("type") != "fact" or table.get("fact_type") != "periodic_snapshot":
            continue
        base = str(table.get("qualified_name") or table.get("table") or "")
        columns = [
            str(field.get("column") or "") for field in table.get("fields") or []
            if isinstance(field, dict)
            and str(field.get("role") or "") in {"measure", "measure_candidate"}
            and field.get("column")
        ]
        monthly = base.upper() in period_keyed
        for metric in _metrics_for(base, columns, monthly):
            name, synonyms = metric.name, metric.synonyms
            if monthly and metric.key in _MONTH_END:
                name, synonyms = _MONTH_END[metric.key]
            if name.casefold() in names:
                # Named for the table; for the schema too if two schemas hold
                # a table of that name.
                name = next(
                    label for label in (f"{name} ({base.split('.')[-1]})", f"{name} ({base})")
                    if label.casefold() not in names
                )
                synonyms = ()
            names.add(name.casefold())
            metrics.append(StarterMetric(
                metric.key, name, metric.sql_template, metric.base_table,
                metric.required_columns, tuple(synonyms), metric.description,
                metric.evidence, metric.result_format, metric.confidence,
            ))
    return metrics


def _model_columns(model: dict | None) -> dict[str, list[str]]:
    return {
        str(table.get("qualified_name") or table.get("table") or ""): [
            str(field.get("column") or "")
            for field in table.get("fields") or [] if isinstance(field, dict)
        ]
        for table in (model or {}).get("tables") or [] if isinstance(table, dict)
    }


def propose_starter_metrics(
    account_id: str,
    model: dict | None,
    *,
    db_type: str = "azure_sql",
    schema_columns: dict[str, list[str]] | None = None,
) -> list[int]:
    """File each starter metric as a pending proposal. Returns the new ids.

    Skips a name that is already a metric (active or not), is waiting in the
    queue, or was suggested here before -- an administrator who rejected a
    suggestion is not asked again on every rebuild. Skips any definition the
    metric validator refuses, and says so. Nothing here changes an answer: the
    proposal-accept route is the only way into the registry.
    """
    import store
    from core.metric_validator import validate_metric

    if schema_columns is None:
        schema_columns = _model_columns(model)
    taken = {
        str(metric.get("name") or "").strip().casefold()
        for metric in store.list_metrics(account_id, active_only=False) or []
    }
    for proposal in store.list_metric_proposals(account_id) or []:
        if proposal.get("status") == "pending" or proposal.get("generated_by") == GENERATED_BY:
            taken.add(str((proposal.get("payload") or {}).get("name") or "").strip().casefold())
    created: list[int] = []
    for metric in starter_metrics(model):
        if metric.name.casefold() in taken:
            continue
        payload = metric.as_metric()
        verdict = validate_metric(payload, db_type=db_type, schema_columns=schema_columns)
        if not verdict.valid:
            log.warning(
                "Starter metric %r not proposed for %s: %s",
                metric.name, account_id, "; ".join(verdict.errors[:3]),
            )
            continue
        created.append(store.create_metric_proposal(
            account_id,
            payload=payload,
            action="create_metric",
            confidence_score=metric.confidence,
            generated_by=GENERATED_BY,
            reason=" ".join(metric.evidence),
            validation={"valid": True},
            dryrun={"status": "skipped",
                    "reason": "suggested from the semantic model when the knowledge base was built"},
        ))
    return created
