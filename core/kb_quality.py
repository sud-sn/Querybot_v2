"""Deterministic quality checks for the generated Knowledge Base.

This module deliberately does not decide whether a client can query data.  It
identifies the semantic gaps an administrator should review before relying on
the KB for broad self-service analytics.  The report is stored beside the
semantic model and can be regenerated without an LLM or a database connection.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


QUALITY_REPORT_FILENAME = "_kb_quality.json"


def _issue(
    severity: str,
    code: str,
    message: str,
    *,
    table: str = "",
    action: str = "",
) -> dict[str, str]:
    return {
        "severity": severity,
        "code": code,
        "message": message,
        "table": table,
        "action": action,
    }


def _is_display_dimension(dimension: dict[str, Any]) -> bool:
    return bool(
        str(dimension.get("display_column") or "").strip()
        or str(dimension.get("code_column") or "").strip()
    )


def evaluate_kb_quality(
    semantic_model: dict[str, Any] | None,
    *,
    kb_dir: str = "",
) -> dict[str, Any]:
    """Return a compact, deterministic readiness report for a semantic model."""
    model = semantic_model or {}
    tables = model.get("tables") or []
    relationships = model.get("relationships") or []
    issues: list[dict[str, str]] = []

    if not tables:
        issues.append(_issue(
            "critical",
            "no_tables",
            "No tables were generated in the semantic model.",
            action="Run schema discovery and rebuild the Knowledge Base.",
        ))

    relationship_counts: dict[str, int] = {}
    for relationship in relationships:
        for key in ("from_table", "to_table"):
            value = str(relationship.get(key) or "").upper()
            if value:
                relationship_counts[value] = relationship_counts.get(value, 0) + 1
                relationship_counts[value.rsplit(".", 1)[-1]] = relationship_counts.get(value.rsplit(".", 1)[-1], 0) + 1

    fact_tables = 0
    low_confidence_fields = 0
    generated_relationships = 0
    for relationship in relationships:
        if str(relationship.get("status") or "generated") != "approved":
            generated_relationships += 1

    for table in tables:
        table_name = str(table.get("qualified_name") or table.get("table") or "")
        table_type = str(table.get("type") or "unknown").lower()
        fields = table.get("fields") or []
        dimensions = table.get("dimensions") or []
        measures = table.get("measures") or []
        date_roles = table.get("date_roles") or []

        if table_type == "fact":
            fact_tables += 1
            grain = str(table.get("grain") or "").strip()
            if not grain or grain == "needs_admin_context":
                issues.append(_issue(
                    "warning",
                    "grain_needs_review",
                    "Fact-table grain is not confirmed.",
                    table=table_name,
                    action="Confirm what one row represents before approving cross-fact joins or metrics.",
                ))
            if not measures:
                issues.append(_issue(
                    "warning",
                    "fact_without_measure",
                    "Fact table has no detected measure candidates.",
                    table=table_name,
                    action="Review numeric fields and create or approve the intended business metrics.",
                ))
            if not date_roles:
                issues.append(_issue(
                    "info",
                    "fact_without_date_role",
                    "No date role was detected for this fact table.",
                    table=table_name,
                    action="Map an order, invoice, delivery, receipt, or other business date if time analysis is needed.",
                ))
            if len(tables) > 1 and not relationship_counts.get(table_name.upper()) and not relationship_counts.get(table_name.rsplit(".", 1)[-1].upper()):
                issues.append(_issue(
                    "warning",
                    "unconnected_fact",
                    "Fact table has no generated relationship to the current model.",
                    table=table_name,
                    action="Validate or add the correct entity-graph join before using cross-table questions.",
                ))

        if table_type == "dimension" and fields and dimensions and not any(_is_display_dimension(d) for d in dimensions):
            issues.append(_issue(
                "warning",
                "dimension_without_display_field",
                "Dimension has no detected display or code field.",
                table=table_name,
                action="Approve the user-facing name/description field so answers do not expose internal IDs.",
            ))

        for field in fields:
            try:
                confidence = int(float(field.get("confidence") or 0))
            except (TypeError, ValueError):
                confidence = 0
            if confidence >= 70 or field.get("status") == "approved":
                continue
            low_confidence_fields += 1
            issues.append(_issue(
                "warning",
                "low_confidence_field",
                f"Field `{field.get('column') or ''}` has low-confidence generated meaning.",
                table=table_name,
                action="Provide a business definition in the Semantic Layer and approve it.",
            ))

    if relationships and generated_relationships:
        issues.append(_issue(
            "info",
            "relationships_need_review",
            f"{generated_relationships} relationship(s) are generated rather than admin-approved.",
            action="Review high-value join paths in the Entity Graph before relying on them for financial metrics.",
        ))

    if kb_dir:
        path = Path(kb_dir)
        for table in tables:
            name = str(table.get("table") or "")
            if not name:
                continue
            if not (path / f"{name}_kb.md").exists():
                issues.append(_issue(
                    "critical",
                    "missing_table_contract",
                    "Generated table contract document is missing.",
                    table=str(table.get("qualified_name") or name),
                    action="Rebuild the Knowledge Base for this schema scope.",
                ))
            if not (path / f"{name}_queries.md").exists():
                issues.append(_issue(
                    "warning",
                    "missing_query_examples",
                    "No generated question-to-SQL examples were found for this table.",
                    table=str(table.get("qualified_name") or name),
                    action="Rebuild the Knowledge Base, then validate the generated examples.",
                ))

    severity_weight = {"critical": 25, "warning": 5, "info": 0}
    score = max(0, 100 - sum(severity_weight.get(issue["severity"], 0) for issue in issues))
    critical_count = sum(issue["severity"] == "critical" for issue in issues)
    warning_count = sum(issue["severity"] == "warning" for issue in issues)
    info_count = sum(issue["severity"] == "info" for issue in issues)
    return {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "score": score,
        "status": "blocked" if critical_count else ("needs_review" if warning_count else "ready"),
        "summary": {
            "tables": len(tables),
            "fact_tables": fact_tables,
            "relationships": len(relationships),
            "low_confidence_fields": low_confidence_fields,
            "critical": critical_count,
            "warnings": warning_count,
            "info": info_count,
        },
        "issues": issues,
    }


def write_kb_quality_report(
    semantic_model: dict[str, Any] | None,
    kb_dir: str,
) -> dict[str, Any]:
    """Evaluate and atomically persist ``_kb_quality.json`` beside KB files."""
    report = evaluate_kb_quality(semantic_model, kb_dir=kb_dir)
    path = Path(kb_dir)
    path.mkdir(parents=True, exist_ok=True)
    target = path / QUALITY_REPORT_FILENAME
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(target)
    return report


def load_kb_quality_report(kb_dir: str) -> dict[str, Any]:
    """Load the last report, returning an empty shape when KB has not run yet."""
    target = Path(kb_dir) / QUALITY_REPORT_FILENAME
    if not target.exists():
        return {}
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
