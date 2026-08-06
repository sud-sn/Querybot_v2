"""Deterministic, tenant-neutral warehouse table classification.

The classifier deliberately treats names as one signal rather than truth.  It
combines declared keys, foreign-key direction, column roles and optional
terminology packs so cryptic ERP tables can still be classified safely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


_NUMERIC_TYPES = (
    "INT", "NUMBER", "NUMERIC", "DECIMAL", "FLOAT", "DOUBLE", "REAL", "MONEY",
)
_DATE_TYPES = ("DATE", "TIME")
_MEASURE_TOKENS = (
    "AMT", "AMOUNT", "QTY", "QUANTITY", "PRICE", "COST", "VALUE", "REVENUE",
    "SALES", "BALANCE", "WEIGHT", "RATE", "TOTAL", "COUNT",
)
_DESCRIPTIVE_TOKENS = (
    "NAME", "NM", "DESC", "DSC", "DESCRIPTION", "LABEL", "CODE", "CD",
    "TYPE", "TYP", "STATUS", "CATEGORY", "REGION", "CITY", "COUNTRY",
)
_CALENDAR_TOKENS = (
    "YEAR", "MONTH", "WEEK", "QUARTER", "DAY", "FISCAL", "CALENDAR",
)


def _bare(name: str) -> str:
    return str(name or "").split(".")[-1]


def _column_name(column: Any) -> str:
    if isinstance(column, dict):
        return str(column.get("name") or column.get("column_name") or column.get("COLUMN_NAME") or "")
    return str(column or "")


def _column_type(column: Any) -> str:
    if isinstance(column, dict):
        return str(column.get("type") or column.get("data_type") or column.get("DATA_TYPE") or "").upper()
    return ""


def _contains_token(name: str, tokens: Iterable[str]) -> bool:
    upper = str(name or "").upper()
    parts = {part for part in upper.replace("-", "_").split("_") if part}
    return any(token in parts or upper.endswith("_" + token) for token in tokens)


@dataclass(frozen=True)
class TableRole:
    role: str
    confidence: int
    evidence: tuple[str, ...] = field(default_factory=tuple)
    grain: str = "needs_admin_context"
    grain_columns: tuple[str, ...] = field(default_factory=tuple)
    fact_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "grain": self.grain,
            "grain_columns": list(self.grain_columns),
            "fact_type": self.fact_type,
            "classifier_version": 2,
        }


def _name_signal(table_name: str, vocab=None) -> str:
    try:
        from core.naming_convention import match_table_suffix
        rule = match_table_suffix(table_name, vocab=vocab)
        table_type = str(getattr(rule, "table_type", "") or "").lower()
        if "bridge" in table_type:
            return "bridge"
        if table_type == "fact_table":
            return "fact"
        if table_type == "dimension_table":
            return "dimension"
    except Exception:
        pass
    return ""


def classify_table(
    table_name: str,
    metadata: dict[str, Any] | None = None,
    *,
    declared_fks: Iterable[dict[str, Any]] = (),
    vocab=None,
    confirmed_role: str = "",
    metric_source: bool = False,
) -> TableRole:
    """Classify one table using independently auditable evidence.

    ``confirmed_role`` is authoritative.  Otherwise no single heuristic can
    produce maximum confidence; a production-safe classification needs
    corroborating structural/content evidence.
    """
    meta = metadata or {}
    table = _bare(table_name)
    table_u = table.upper()
    confirmed = str(confirmed_role or "").strip().lower()
    if confirmed in {"fact", "dimension", "bridge", "date_dimension"}:
        return TableRole(confirmed, 100, ("administrator-confirmed role",))

    columns = list(meta.get("columns") or [])
    column_names = [_column_name(column) for column in columns if _column_name(column)]
    column_upper = [name.upper() for name in column_names]
    pk_columns = tuple(str(value) for value in (meta.get("pk_columns") or []) if value)

    outgoing = []
    incoming = []
    for fk in declared_fks or ():
        if not isinstance(fk, dict):
            continue
        parent = str(fk.get("parent_table") or fk.get("from_table") or "").upper()
        target = str(fk.get("ref_table") or fk.get("to_table") or "").upper()
        if parent == table_u:
            outgoing.append(fk)
        if target == table_u:
            incoming.append(fk)

    scores = {"fact": 0, "dimension": 0, "bridge": 0, "date_dimension": 0}
    evidence: dict[str, list[str]] = {key: [] for key in scores}

    name_role = _name_signal(table, vocab=vocab)
    if name_role:
        scores[name_role] += 26
        evidence[name_role].append(f"terminology/name pattern indicates {name_role}")
    if metric_source:
        scores["fact"] += 48
        evidence["fact"].append("approved metric uses this table as its source")

    measure_columns = [name for name in column_names if _contains_token(name, _MEASURE_TOKENS)]
    descriptive_columns = [name for name in column_names if _contains_token(name, _DESCRIPTIVE_TOKENS)]
    numeric_columns = [
        _column_name(column) for column in columns
        if any(token in _column_type(column) for token in _NUMERIC_TYPES)
    ]
    physical_dates = [
        _column_name(column) for column in columns
        if any(token in _column_type(column) for token in _DATE_TYPES)
    ]
    calendar_columns = [name for name in column_names if _contains_token(name, _CALENDAR_TOKENS)]

    if measure_columns and len(measure_columns) >= max(2, len(descriptive_columns)):
        scores["fact"] += 25
        evidence["fact"].append(f"measure-heavy columns ({len(measure_columns)} detected)")
    elif descriptive_columns:
        scores["dimension"] += 18
        evidence["dimension"].append(f"descriptive lookup columns ({len(descriptive_columns)} detected)")

    if len(outgoing) >= 2:
        scores["fact"] += 15
        evidence["fact"].append(f"owns {len(outgoing)} declared foreign keys")
    if incoming and len(outgoing) <= 1:
        scores["dimension"] += 18
        evidence["dimension"].append(f"referenced by {len(incoming)} declared foreign keys")

    outgoing_cols = {
        str(fk.get("parent_col") or fk.get("from_column") or "").upper()
        for fk in outgoing
    }
    pk_upper = {column.upper() for column in pk_columns}
    bridge_shape = (
        len(outgoing) >= 2
        and bool(pk_upper)
        and pk_upper.issubset(outgoing_cols)
        and not measure_columns
    )
    if bridge_shape:
        scores["bridge"] += 60
        evidence["bridge"].append("composite primary key is made from multiple foreign keys")

    date_key_named = any("DATE" in name or name.endswith(("_DT", "_DTE")) for name in column_upper)
    if (
        (physical_dates and len(calendar_columns) >= 2)
        or (date_key_named and len(calendar_columns) >= 3)
    ):
        scores["date_dimension"] += 70
        evidence["date_dimension"].append("calendar attributes and a governed date value/key are present")

    if pk_columns:
        scores["dimension"] += 8
        evidence["dimension"].append("declared primary/unique key is present")
    if numeric_columns and not measure_columns and descriptive_columns:
        scores["dimension"] += 6
        evidence["dimension"].append("numeric identifiers are accompanied by descriptive attributes")

    role = max(scores, key=lambda key: scores[key])
    top_score = scores[role]
    sorted_scores = sorted(scores.values(), reverse=True)
    margin = top_score - (sorted_scores[1] if len(sorted_scores) > 1 else 0)

    # Preserve conservative compatibility for genuinely unclassified lookup
    # tables, but expose the low confidence so admins can review them.
    if top_score <= 0:
        role = "dimension"
        top_score = 20
        margin = 0
        evidence[role].append("no strong evidence; conservative lookup-table fallback")

    confidence = min(98, max(35, 45 + min(35, top_score // 2) + min(18, margin // 2)))
    if role == "bridge":
        grain = "one row per related key combination"
        grain_columns = pk_columns
        fact_type = ""
    elif role == "date_dimension":
        grain = "one row per calendar date"
        grain_columns = pk_columns or tuple(physical_dates[:1])
        fact_type = ""
    elif role == "dimension":
        grain = "one row per business member"
        grain_columns = pk_columns
        fact_type = ""
    else:
        grain_columns = pk_columns
        upper = table_u
        if any(token in upper for token in ("SNAP", "BAL", "INVENTORY", "STOCK")):
            fact_type = "periodic_snapshot"
            grain = "one row per snapshot grain"
        elif any(token in upper for token in ("ACCUM", "LIFECYCLE")):
            fact_type = "accumulating_snapshot"
            grain = "one row per process instance"
        else:
            fact_type = "transaction"
            grain = "one row per business event"

    return TableRole(
        role=role,
        confidence=confidence,
        evidence=tuple(evidence[role]),
        grain=grain,
        grain_columns=tuple(grain_columns),
        fact_type=fact_type,
    )


def classify_schema_tables(
    schema: dict[str, Any],
    *,
    vocab=None,
    metric_source_tables: Iterable[str] = (),
) -> dict[str, TableRole]:
    declared_fks = schema.get("__db_fk_constraints__", []) or []
    metric_sources = {_bare(value).upper() for value in metric_source_tables or ()}
    return {
        fqn: classify_table(
            fqn,
            meta,
            declared_fks=declared_fks,
            vocab=vocab,
            metric_source=_bare(fqn).upper() in metric_sources,
        )
        for fqn, meta in schema.items()
        if isinstance(meta, dict) and not str(fqn).startswith("__")
    }
