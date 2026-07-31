"""Metadata-only planning for conversational report/playbook building.

The LLM may see ONLY the calling user's own ACL-filtered metric names and
descriptions -- never row data, and never a metric the user could not
already see via the existing checkbox report-creation form (the caller
must filter by store.get_allowed_tables BEFORE building this input, the
same gate report_engine.run_metric_for_report already enforces at
execution time). Every metric reference is resolved locally via a
METRIC_REF_n token the model can only reference, never invent an id for
-- the same discipline core/result_planner.py uses for VALUE_REF_n.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

_ALLOWED_CADENCES = {"daily", "weekly"}
_PLAN_KEYS = {
    "operation", "name", "metric_refs", "cadence", "day_of_week", "hour", "confidence",
}
_MIN_CONFIDENCE = 0.0
_MAX_CONFIDENCE = 1.0
# Fail closed: a response missing/malformed confidence is not evidence of
# high confidence -- mirrors core/result_planner.py's same discipline.
_DEFAULT_CONFIDENCE = _MIN_CONFIDENCE


@dataclass(frozen=True)
class ReportPlan:
    name: str
    metric_ids: tuple[int, ...]
    cadence: str = ""  # "" = no schedule requested, just define the report
    day_of_week: int = 0
    hour: int = 8
    confidence: float = _DEFAULT_CONFIDENCE


@dataclass(frozen=True)
class ReportPlanInput:
    system_prompt: str
    user_prompt: str
    bindings: dict[str, int] = field(default_factory=dict)  # METRIC_REF_n -> metric_id


def build_report_plan_input(text: str, metrics: list[dict]) -> ReportPlanInput:
    """``metrics``: ACL-already-filtered store.list_metrics() rows -- the
    caller must filter by get_allowed_tables before calling this, so a
    metric the user cannot see is never even offered as an option."""
    bindings: dict[str, int] = {}
    manifest_lines: list[str] = []
    for metric in metrics:
        try:
            metric_id = int(metric.get("id"))
        except (TypeError, ValueError):
            continue
        ref = f"METRIC_REF_{len(bindings) + 1}"
        bindings[ref] = metric_id
        name = str(metric.get("name") or "").strip()
        desc = str(metric.get("description") or "").strip()
        manifest_lines.append(f'{ref}: "{name}"' + (f" -- {desc}" if desc else ""))
    manifest_text = "\n".join(manifest_lines)[:6000]

    system_prompt = (
        "You plan a report from a plain-language request. Return exactly one JSON "
        "object, no markdown, no explanation. You receive ONLY report metric names "
        "and descriptions -- never row data. Reference a metric ONLY by its "
        "METRIC_REF_n token from the manifest below; never invent a metric or an id.\n\n"
        "Allowed keys: operation, name, metric_refs, cadence, day_of_week, hour, "
        "confidence.\n"
        'operation must be "define_report". metric_refs must be a list of one or '
        "more METRIC_REF_n tokens from the manifest -- never a raw metric name. name "
        "is a short report title you invent from the request. cadence must be "
        '"daily" or "weekly" -- omit it entirely if no schedule was requested. '
        "day_of_week is 0=Monday..6=Sunday (only meaningful with weekly cadence). "
        "hour is 0-23 (default 8 if not stated). If the request cannot be expressed "
        'using only the metrics below, return {"operation":"unsupported"}.\n\n'
        'Always include "confidence": a number from 0.0 to 1.0 rating how certain '
        "you are every metric and schedule detail in your plan matches what the "
        "user meant, given only the manifest and their text. Use a high value "
        "(0.9-1.0) only when every metric reference and schedule detail is "
        "unambiguous.\n\n"
        f"Metric manifest:\n{manifest_text}"
    )
    user_prompt = json.dumps(
        {"request": str(text or "").strip()},
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return ReportPlanInput(system_prompt=system_prompt, user_prompt=user_prompt, bindings=bindings)


async def parse_report_plan(
    text: str,
    metrics: list[dict],
    complete: Callable[..., Awaitable[tuple[str, int, int]]],
) -> tuple[ReportPlan | None, str]:
    """Ask a model for a constrained AST, then compile+validate it locally."""
    plan_input = build_report_plan_input(text, metrics)
    try:
        raw, _, _ = await complete(
            system=plan_input.system_prompt,
            user=plan_input.user_prompt,
            temperature=0.0,
            max_tokens=300,
        )
    except Exception:
        return None, "The report planner was unavailable."
    return compile_report_plan_response(raw, plan_input.bindings)


def compile_report_plan_response(
    raw_response: str,
    bindings: dict[str, int],
) -> tuple[ReportPlan | None, str]:
    """Validate a model JSON plan against the local METRIC_REF bindings and
    compile it into a local ReportPlan. Never trusts a metric id the model
    names unless it matches an already-bound METRIC_REF_n token."""
    plan = _parse_json_object(raw_response)
    if plan is None:
        return None, "The report planner did not return valid JSON."
    if set(plan) - _PLAN_KEYS:
        return None, "The report planner returned unsupported fields."

    operation = str(plan.get("operation") or "").strip().lower()
    if operation == "unsupported":
        return None, "That report could not be built from the metrics available to you."
    if operation != "define_report":
        return None, "The report planner returned an unsupported operation."

    refs = plan.get("metric_refs")
    if not isinstance(refs, list) or not refs:
        return None, "No metrics were resolved for that report."
    metric_ids: list[int] = []
    for ref in refs:
        ref_str = str(ref).strip()
        if ref_str not in bindings:
            return None, f'"{ref_str}" is not a metric available to you.'
        metric_id = bindings[ref_str]
        if metric_id not in metric_ids:
            metric_ids.append(metric_id)
    if not metric_ids:
        return None, "No metrics were resolved for that report."

    name = str(plan.get("name") or "").strip() or "Untitled report"

    cadence_raw = str(plan.get("cadence") or "").strip().lower()
    cadence = cadence_raw if cadence_raw in _ALLOWED_CADENCES else ""

    day_of_week = _clamp_int(plan.get("day_of_week"), default=0, low=0, high=6)
    hour = _clamp_int(plan.get("hour"), default=8, low=0, high=23)

    return ReportPlan(
        name=name,
        metric_ids=tuple(metric_ids),
        cadence=cadence,
        day_of_week=day_of_week,
        hour=hour,
        confidence=_extract_confidence(plan),
    ), ""


def _clamp_int(raw_value: Any, *, default: int, low: int, high: int) -> int:
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, value))


def _extract_confidence(plan: dict) -> float:
    raw_value = plan.get("confidence")
    if raw_value is None:
        return _DEFAULT_CONFIDENCE
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return _DEFAULT_CONFIDENCE
    if value != value:  # NaN guard
        return _DEFAULT_CONFIDENCE
    return max(_MIN_CONFIDENCE, min(_MAX_CONFIDENCE, value))


def _parse_json_object(raw_response: str) -> dict | None:
    raw = str(raw_response or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None
