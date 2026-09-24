"""Governed analytical-shape contracts for generated SQL.

The metric registry remains authoritative.  This module covers the common case
where a user asks for a new dimensional breakdown that has no pre-written SQL:
it records the resolved measure/dimension evidence and constrains the LLM and
validator to a grouped composition query instead of a row-level distribution.
It contains no tenant-specific vocabulary or physical schema names.
"""

from __future__ import annotations

import re
from typing import Any

from core.contribution_analysis import detect_composition_intent


_IDENTIFIER_RE = re.compile(r"(?:^|_)(?:ID|KEY|CODE|NUM|NO|NBR|SEQ)$", re.I)
_NON_ADDITIVE_RE = re.compile(r"(?:PCT|PERCENT|RATE|RATIO|AVG|AVERAGE)", re.I)
_SEMI_ADDITIVE_RE = re.compile(
    r"(?:BALANCE|ON_HAND|INVENTORY|SNAPSHOT|CLOSING)", re.I
)
# The spelled-out words above only ever match a mart that writes them out. Real
# ERP marts abbreviate, so the very columns this classifier exists to protect
# read as additive: BAL_VAL_AMT and OH_QTY both matched _ADDITIVE_RE on "AMT" /
# "QTY" and were summed across snapshots. That is the failure this set closes.
#
# Matched per TOKEN, never as a substring, which is the whole reason it is a set
# and not another alternation bolted onto the regex above: "BAL" as a substring
# also fires on GLOBAL_AMT and VERBAL_SCORE, and a false semi-additive verdict
# suppresses a legitimate SUM. Splitting on underscores and camel humps first
# means GLOBAL_AMT tokenises to {GLOBAL, AMT} and never matches.
#
# Deliberately excluded: "INV" (invoice far more often than inventory), "CLS"
# (class), "OPN" (open orders), and the qualifier-only names on a snapshot fact
# such as AVL/ALC -- those are semi-additive because of the grain they sit at,
# not because of what they are called, and guessing from the name would be the
# kind of rule this module's contract says is worse than no rule.
#
# STOCK and HAND are here as well as in the spelled-out set above because that
# one requires the underscore form: a camel-case mart writes StockOnHandQty,
# which tokenises to {STOCK, ON, HAND, QTY} and matches "ON_HAND" nowhere.
#
# ENDING is a token for the same reason BAL is: as a substring it fired inside
# PENDING_QTY and SPENDING_AMT, and suppressed their SUM. The cumulative-to-date
# prefixes are balances too -- a year-to-date figure summed across months counts
# January twelve times.
_SEMI_ADDITIVE_TOKENS = frozenset({
    "BAL", "BALS", "OH", "QOH", "SOH", "STK", "STOCK", "HAND", "SNAP",
    "EOD", "EOM", "EOP", "EOQ", "EOY", "CLSG", "OPNG", "ENDING",
    "YTD", "QTD", "MTD", "WTD", "PTD", "LTD", "ITD", "CUM", "CUMULATIVE",
})
_ADDITIVE_RE = re.compile(
    r"(?:AMT|AMOUNT|REVENUE|SALES|COST|SPEND|PROFIT|INCOME|EXPENSE|"
    r"QTY|QUANTITY|VOLUME|UNITS|COUNT|CNT)",
    re.I,
)
# The abbreviations the naming convention already treats as amounts (_CST is a
# cost amount, _PFT a profit) and _ADDITIVE_RE does not spell.
_ADDITIVE_TOKENS = frozenset({"CST", "PFT"})
_TOKEN_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+|(?<=[a-z0-9])(?=[A-Z])")

# "On hand", abbreviated the way inventory marts write it -- ON_HND_QTY,
# CUR_ON_HND_QTY. HND alone is not claimed: HND_CHG_AMT is a handling charge,
# a flow. Only ON followed by HND/HAND, with or without the underscore.
_ON_HAND_RE = re.compile(r"(?:^|_)ON_?HA?ND(?:_|$)", re.I)
# An aging bucket of stock: AGE_0_9_QTY, AGE_24_PLU, AGED_30_60_AMT -- how much
# stock sits in that age band at the snapshot, a balance like the stock itself.
# A bare AGE (an employee's age) has no bucket bounds and does not match.
_AGING_BUCKET_RE = re.compile(
    r"(?:^|_)AG(?:E|ED|ING)(?:_\d+)+(?:_(?:PLU|PLUS|OVR|OVER))?(?:_|$)", re.I,
)
# A price or a cost PER unit is never summed: two items' unit costs add up to
# nothing. ITM_CST, UNT_CST, STD_CST, AVG_ITM_CST and any price qualify. An
# extended or total figure (EXT_CST, LIN_AMT, CST_VAL) is a line amount and
# still adds; a quantity (PCE_QTY) is not a price; a rate is a rate.
_UNIT_MARKER_TOKENS = frozenset({
    "ITM", "ITEM", "UNT", "UNIT", "STD", "PER", "AVG", "AVERAGE",
})
_COST_TOKENS = frozenset({"CST", "COST"})
_PRICE_TOKENS = frozenset({"PRC", "PCE", "PRICE"})
_NOT_A_UNIT_VALUE_TOKENS = frozenset({
    "EXT", "EXTD", "EXTENDED", "TOT", "TOTAL", "LIN", "LINE", "VAL", "VALUE",
    "AMT", "AMOUNT", "QTY", "QUANTITY", "UNITS", "CNT", "COUNT",
    "PCT", "PERCENT", "PERC", "RATE", "RATIO",
})
# "Number of <events>": receipts, returns, deliveries, counts taken. What
# happened during the period, so it adds across periods even on a snapshot.
_EVENT_COUNT_RE = re.compile(r"^(?:NUM|NBR|NO)_OF_", re.I)
# Movement during the period. On a periodic snapshot these sit beside the
# balances but are not balances: purchases in March plus purchases in April
# are the purchases of the two months.
_FLOW_TOKENS = frozenset({
    "PCH", "PURCH", "PURCHASE", "PURCHASED", "PURCHASES",
    "SLD", "SOLD", "SLS", "SALE", "SALES", "REVENUE",
    "SHP", "SHIP", "SHIPPED", "SHIPMENT", "SHIPMENTS",
    "RCT", "RCPT", "RECEIPT", "RECEIPTS", "RCV", "RECEIVED",
    "RET", "RTN", "RETURN", "RETURNS", "RETURNED",
    "DLV", "DELIVERY", "DELIVERIES", "DELIVERED",
    "TFR", "TRF", "TRANSFER", "TRANSFERS", "TRANSFERRED",
    "RECLASS", "RECLS", "ISS", "ISSUE", "ISSUED", "ISSUES",
    "ADJ", "ADJUSTMENT", "ADJUSTMENTS", "SCRP", "SCRAP", "SCRAPPED",
    "USG", "USAGE", "CONSUMED", "CONSUMPTION", "IVC", "INVOICED", "BOOKED",
})
_RECLASS_RE = re.compile(r"(?:^|_)RE_CLS(?:_|$)", re.I)

# Why a measure is, or is not, summable, as far as its name can tell. A flow
# and a count of events stay additive on a snapshot fact; a name that carries
# no such evidence ("generic") takes its behaviour over time from the grain of
# the table it sits on.
GRAIN_EXEMPT_BASES = frozenset({"flow", "event_count"})


def _name_tokens(name: str) -> set[str]:
    """Upper-cased tokens of a column name, split on separators and camel humps."""
    return {token.upper() for token in _TOKEN_SPLIT_RE.split(str(name or "")) if token}


def measure_additivity(column: str, field: dict[str, Any] | None = None) -> tuple[str, str]:
    """``(aggregation, basis)`` for one measure column, strongest evidence first.

    ``aggregation`` is additive / semi_additive / non_additive / unknown, the
    vocabulary every caller already speaks. ``basis`` says why, because the
    table's grain needs to know: on a periodic snapshot the balances are
    semi-additive, but the flows beside them (purchased, sold, transferred)
    and the counts of events are not -- summing March and April purchases is
    right, summing March and April stock on hand is not.

    An admin's declaration outranks every name rule ("declared"). Then, from
    the name: a price or cost per unit ("unit_value"), an average or rate
    ("average_or_rate"), a balance or level ("balance"), a count of events
    ("event_count"), a movement ("flow"), a plain amount or quantity
    ("generic"). A unit cost is non-additive even on a snapshot fact, and a
    balance is semi-additive even on a transaction fact.
    """
    field = field or {}
    declared = str(
        field.get("aggregation_semantics")
        or field.get("aggregation")
        or ""
    ).strip().lower().replace("-", "_")
    if declared in {"additive", "semi_additive", "non_additive"}:
        return declared, "declared"
    name = str(column or "")
    tokens = _name_tokens(name)
    if not tokens & _NOT_A_UNIT_VALUE_TOKENS and (
        tokens & _PRICE_TOKENS or (tokens & _COST_TOKENS and tokens & _UNIT_MARKER_TOKENS)
    ):
        return "non_additive", "unit_value"
    if _NON_ADDITIVE_RE.search(name):
        return "non_additive", "average_or_rate"
    # Semi-additive is checked before additive because a balance column almost
    # always also carries an additive suffix (BAL_VAL_AMT, OH_QTY). Reversing
    # these two lines is what made the abbreviation blind spot invisible.
    if (
        _SEMI_ADDITIVE_RE.search(name)
        or tokens & _SEMI_ADDITIVE_TOKENS
        or _ON_HAND_RE.search(name)
        or _AGING_BUCKET_RE.search(name)
    ):
        return "semi_additive", "balance"
    if _EVENT_COUNT_RE.search(name):
        return "additive", "event_count"
    if tokens & _FLOW_TOKENS or _RECLASS_RE.search(name):
        return "additive", "flow"
    if _ADDITIVE_RE.search(name) or tokens & _ADDITIVE_TOKENS:
        return "additive", "generic"
    return "unknown", "unknown"


def _measure_class(column: str, field: dict[str, Any]) -> str:
    return measure_additivity(column, field)[0]


def grained_aggregation(
    column: str, fallback: str = "", *, snapshot_fact: bool = False,
) -> tuple[str, str]:
    """``(aggregation, basis)`` a measure column is modelled and documented with.

    The name's own evidence first. A name with none ("generic", "unknown")
    keeps ``fallback`` -- the naming convention's suffix verdict, possibly
    blank. On a snapshot fact, a measure that is neither a flow nor a count of
    events is a level held at each snapshot, so it is semi-additive whatever
    its suffix says.

    One function for the semantic model and the knowledge-base hints, so the
    governed planner and the text the SQL model reads cannot disagree.
    """
    aggregation, basis = measure_additivity(column)
    if basis in {"generic", "unknown"}:
        aggregation = fallback
    if (
        snapshot_fact
        and aggregation in {"", "additive"}
        and basis not in GRAIN_EXEMPT_BASES
    ):
        aggregation = "semi_additive"
    return aggregation, basis


def is_count_or_aging_measure(column: str) -> bool:
    """True for a count of events (NUM_OF_RCT) or an aging bucket (AGE_24_PLU).

    Both are measures that carry none of the amount or quantity suffixes, so
    the column-role rules need to be told.
    """
    name = str(column or "")
    return bool(_EVENT_COUNT_RE.search(name) or _AGING_BUCKET_RE.search(name))


def collapse_rows_by_label(
    rows: list[dict], label_col: str, value_col: str,
    *, measure_name: str = "",
) -> list[tuple[str, float]] | None:
    """One (label, value) pair per label, or None when they may not be merged.

    A result grouped by two things carries a row per label PER PERIOD. Reading
    those rows as though each label appeared once is what made a leader and a
    runner-up the same warehouse in different months, divided a category's
    share by a total counted once per period, and drew one slice of a pie
    several times.

    Merging means summing, and summing is only sound when the measure adds up.
    A margin percentage summed across three months is arithmetic on nothing,
    and so is a stock balance -- semi-additive means exactly "not across time",
    which is the axis being collapsed here. Both return None: no answer beats a
    plausible wrong one, and the caller decides what to say instead.

    Returns the pairs unchanged, in first-seen order, when every label already
    appears once -- so a one-dimensional result costs nothing and is never
    subject to the additivity rule, because there is nothing to add.

    ``measure_name`` names the column for the additivity decision when the rows
    have been projected onto working keys and no longer carry it. A caller that
    hands over {"_l": ..., "_v": ...} still has to say the measure was
    GRS_MARG_PCT, or the rule reads a key it has no opinion about and defaults
    to summing.
    """
    pairs: list[tuple[str, float]] = []
    for row in rows or []:
        raw = row.get(value_col)
        try:
            value = float(str(raw).replace(",", ""))
        except (TypeError, ValueError):
            continue
        if value != value:            # NaN
            continue
        pairs.append((str(row.get(label_col, "")), value))

    if len({label for label, _ in pairs}) == len(pairs):
        return pairs
    if measure_class_for_column(measure_name or value_col) != "additive":
        return None

    totals: dict[str, float] = {}
    for label, value in pairs:
        totals[label] = totals.get(label, 0.0) + value
    return list(totals.items())


def measure_class_for_column(column: str) -> str:
    """Additivity inferred from a column NAME alone.

    For callers holding a result set and no field metadata -- the narrative
    builder, which has to decide whether rows sharing a label may be summed
    into one. Returns the same vocabulary as the metric classifier:
    additive / semi_additive / non_additive / unknown.
    """
    return _measure_class(column, {})


def measure_class_for_metric(metric: dict[str, Any]) -> str:
    """Classify a registry metric as additive / semi_additive / non_additive.

    An admin-declared `aggregation_semantics` (or `aggregation`) is
    authoritative; otherwise the measure columns named in the approved formula
    are inspected. Semi-additive wins any tie because it is the restrictive
    reading — a balance summed across time is wrong, whereas an additive
    measure aggregated at one grain is merely narrower than it needed to be.

    Tenant-neutral: reads only registry metadata and column naming, never
    physical schema names or a client's vocabulary.
    """
    declared = str(
        metric.get("aggregation_semantics")
        or metric.get("aggregation")
        or ""
    ).strip().lower().replace("-", "_")
    if declared in {"additive", "semi_additive", "non_additive"}:
        return declared

    formula = str(
        metric.get("sql_template")
        or metric.get("formula")
        or metric.get("expression")
        or ""
    )
    source = f"{formula} {metric.get('column') or ''}"
    verdicts = [
        measure_additivity(token)
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", source)
    ]
    # A price or unit cost multiplied by a quantity takes the quantity's
    # additivity: SUM(SLD_QTY * ITM_CST) is the cost of what was sold, a flow,
    # and SUM(ON_HND_QTY * ITM_CST) is a stock valuation, a balance. Only a
    # unit value aggregated on its own is non-additive.
    has_quantity = any(
        aggregation in {"additive", "semi_additive"} for aggregation, _ in verdicts
    )
    classes = {
        aggregation for aggregation, basis in verdicts
        if not (basis == "unit_value" and has_quantity)
    }
    for candidate in ("semi_additive", "non_additive", "additive"):
        if candidate in classes:
            return candidate
    return "unknown"


def build_analysis_contract(
    question: str,
    *,
    analytical_intents: dict[str, Any] | None = None,
    semantic_plan: dict[str, Any] | None = None,
    metric_formulas: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a tenant-neutral contract for SQL generation and validation."""
    intents = analytical_intents or {}
    composition = bool(intents.get("contribution")) or detect_composition_intent(question)
    if not composition:
        return {"enabled": False, "mode": "none"}

    measures: list[dict[str, Any]] = []
    dimensions: list[dict[str, Any]] = []

    for metric in metric_formulas or []:
        formula = str(metric.get("sql_template") or "").strip()
        if not formula:
            continue
        measures.append({
            "name": str(metric.get("name") or metric.get("label") or "approved metric"),
            "formula": formula,
            "source": "approved_metric",
            "classification": str(metric.get("aggregation_semantics") or "approved"),
            "authoritative": True,
        })

    for field in (semantic_plan or {}).get("fields") or []:
        role = str(field.get("role") or "").lower()
        column = str(field.get("column") or "")
        table = str(field.get("table") or field.get("source_table") or "")
        if not column:
            continue
        if role in {"measure", "measure_candidate"} and not _IDENTIFIER_RE.search(column):
            if not measures:
                measures.append({
                    "name": str(field.get("term") or column),
                    "table": table,
                    "column": column,
                    "source": "semantic_field",
                    "classification": _measure_class(column, field),
                    "authoritative": str(field.get("source") or "").startswith("approved"),
                    "confidence": field.get("confidence"),
                })
        elif role in {
            "dimension", "display_dimension", "identifier", "attribute",
            "date_dimension", "dimension_key",
        }:
            dimensions.append({
                "name": str(field.get("term") or column),
                "table": table,
                "column": column,
                "source": str(field.get("source") or "semantic_field"),
                "authoritative": field.get("enforcement") == "required",
            })

    return {
        "enabled": True,
        "mode": "composition",
        "requires_grouping": True,
        "requires_aggregate": True,
        "requires_row_values": False,
        "measure_source": (
            "approved_metric" if any(m.get("source") == "approved_metric" for m in measures)
            else "semantic_field" if measures
            else "schema_grounded"
        ),
        "measures": measures,
        "dimensions": dimensions,
        "chart_policy": {
            "preferred": "pie",
            "max_default_categories": 6,
            "fallback": "bar",
            "requires_non_negative_values": True,
            "requires_positive_total": True,
        },
    }


def format_analysis_contract(contract: dict[str, Any]) -> str:
    """Format the contract as concise, copy-safe SQL-generation guidance."""
    if not contract or not contract.get("enabled"):
        return ""
    lines = [
        "GOVERNED ANALYSIS CONTRACT:",
        "- Intent: categorical composition/share, not a statistical histogram.",
        "- Return one row per requested business category.",
        "- SELECT the category plus an aggregated measure and GROUP BY the category.",
        "- Do NOT return individual measure rows for automatic binning.",
        "- Preserve governed joins, metric filters, row grain, and tenant scope.",
    ]
    for measure in contract.get("measures") or []:
        if measure.get("source") == "approved_metric":
            lines.append(
                f"- Approved measure '{measure.get('name')}': use this exact formula: "
                f"{measure.get('formula')}"
            )
            continue
        ref = ".".join(
            part for part in (measure.get("table"), measure.get("column")) if part
        )
        classification = str(measure.get("classification") or "unknown")
        lines.append(
            f"- Schema-grounded measure '{measure.get('name')}': {ref or 'resolve from the schema'} "
            f"({classification})."
        )
        if classification == "non_additive":
            lines.append(
                "- This measure is non-additive: never SUM the stored rate/ratio; derive it from governed components or return CANNOT_GENERATE."
            )
        elif classification == "semi_additive":
            lines.append(
                "- This measure is semi-additive: aggregate only at one governed snapshot/date grain; do not sum across snapshots."
            )
        elif classification == "unknown":
            lines.append(
                "- Aggregation semantics are not proven: do not invent a formula; use explicit KB guidance or return CANNOT_GENERATE."
            )
    if not contract.get("measures"):
        lines.append(
            "- No registered metric matched. Resolve exactly one additive measure from schema/KB evidence; if multiple business definitions remain, return CANNOT_GENERATE rather than guessing."
        )
    return "\n".join(lines)
