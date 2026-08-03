"""Constrained, metadata-only planning for conversational dashboards.

The planner can decompose one dashboard request into a small set of business
questions.  It never receives result rows, credentials, SQL, or unrestricted
schema data.  Every resulting question still goes through QueryBot's normal
semantic, validation, compliance, and governed execution pipeline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

_CHART_TYPES = {"auto", "bar", "line", "area", "pie", "donut", "scatter", "table", "kpi"}
_SCHEDULES = {"manual", "hourly", "daily", "weekly"}
_VISIBILITIES = {"personal", "team"}
_PLAN_KEYS = {"operation", "name", "widgets", "tabs", "refresh_schedule", "visibility", "confidence"}
_WIDGET_KEYS = {"question", "title", "chart_type", "tab"}


@dataclass(frozen=True)
class DashboardWidgetPlan:
    question: str
    title: str = ""
    chart_type: str = "auto"
    tab: str = "Overview"


@dataclass(frozen=True)
class DashboardPlan:
    name: str
    widgets: tuple[DashboardWidgetPlan, ...]
    tabs: tuple[str, ...] = ("Overview",)
    refresh_schedule: str = "manual"
    visibility: str = "personal"
    confidence: float = 0.0


@dataclass(frozen=True)
class DashboardPlanInput:
    system_prompt: str
    user_prompt: str
    metric_manifest: tuple[str, ...] = field(default_factory=tuple)


def build_dashboard_plan_input(text: str, metrics: list[dict] | None = None) -> DashboardPlanInput:
    """Build a row-free prompt from an ACL-filtered metric manifest."""
    manifest: list[str] = []
    for metric in list(metrics or [])[:80]:
        name = re.sub(r"\s+", " ", str(metric.get("name") or "")).strip()[:100]
        description = re.sub(r"\s+", " ", str(metric.get("description") or "")).strip()[:180]
        if name:
            manifest.append(f"- {name}" + (f": {description}" if description else ""))
    manifest_text = "\n".join(manifest)[:7000] or "- No governed metric names were supplied; preserve the user's wording."
    system_prompt = (
        "You are a dashboard work planner. Return exactly one JSON object and no markdown. "
        "Decompose the request into 1-6 independent business questions. Never write SQL and "
        "never invent a metric, column, filter, or business definition. Preserve the user's "
        "time ranges, grain, dimensions, and filters in each question. Prefer KPI for a single "
        "total, line for time trends, bar for ranked/category comparisons, table for detail, "
        "and auto when the shape is unclear.\n\n"
        "Allowed top-level keys: operation, name, widgets, tabs, refresh_schedule, visibility, confidence. "
        'operation must be "create_dashboard". widgets is a list whose only allowed keys are '
        "question, title, chart_type, tab. chart_type is auto, bar, line, area, pie, donut, "
        "scatter, table, or kpi. tabs is a short list of tab names. refresh_schedule is manual, "
        "hourly, daily, or weekly. visibility is personal or team. confidence is 0.0-1.0. "
        "If the request is ambiguous, keep the unresolved wording and use low confidence; do not guess.\n\n"
        f"Governed metric names available to this user (context only):\n{manifest_text}"
    )
    return DashboardPlanInput(
        system_prompt=system_prompt,
        user_prompt=json.dumps({"request": str(text or "").strip()}, ensure_ascii=True, separators=(",", ":")),
        metric_manifest=tuple(manifest),
    )


async def parse_dashboard_plan(
    text: str,
    metrics: list[dict] | None,
    complete: Callable[..., Awaitable[tuple[str, int, int]]],
) -> tuple[DashboardPlan | None, str]:
    built = build_dashboard_plan_input(text, metrics)
    try:
        raw, _, _ = await complete(
            system=built.system_prompt,
            user=built.user_prompt,
            temperature=0.0,
            max_tokens=900,
        )
    except Exception:
        return None, "The dashboard planner was unavailable."
    return compile_dashboard_plan_response(raw)


def compile_dashboard_plan_response(raw_response: str) -> tuple[DashboardPlan | None, str]:
    plan = _parse_json_object(raw_response)
    if plan is None:
        return None, "The dashboard planner did not return valid JSON."
    if set(plan) - _PLAN_KEYS:
        return None, "The dashboard planner returned unsupported fields."
    if str(plan.get("operation") or "").lower() != "create_dashboard":
        return None, "The dashboard planner returned an unsupported operation."

    raw_widgets = plan.get("widgets")
    if not isinstance(raw_widgets, list) or not raw_widgets:
        return None, "Tell me what the dashboard should track."
    if len(raw_widgets) > 6:
        return None, "A dashboard can add up to six visuals at a time."

    widgets: list[DashboardWidgetPlan] = []
    for raw_widget in raw_widgets:
        if not isinstance(raw_widget, dict) or set(raw_widget) - _WIDGET_KEYS:
            return None, "The dashboard planner returned an invalid widget."
        question = re.sub(r"\s+", " ", str(raw_widget.get("question") or "")).strip(" .")[:400]
        if not question:
            return None, "A planned dashboard visual did not include a business question."
        if re.match(r"(?is)^\s*(?:select|insert|update|delete|drop|alter|create\s+table)\b", question):
            return None, "Dashboard widgets must be business questions, not SQL."
        chart_type = str(raw_widget.get("chart_type") or "auto").strip().lower()
        if chart_type not in _CHART_TYPES:
            chart_type = "auto"
        title = re.sub(r"\s+", " ", str(raw_widget.get("title") or "")).strip()[:120]
        tab = re.sub(r"\s+", " ", str(raw_widget.get("tab") or "Overview")).strip()[:60] or "Overview"
        widgets.append(DashboardWidgetPlan(question, title, chart_type, tab))

    tabs: list[str] = []
    for raw_tab in list(plan.get("tabs") or []):
        tab = re.sub(r"\s+", " ", str(raw_tab or "")).strip()[:60]
        if tab and tab.lower() not in {value.lower() for value in tabs}:
            tabs.append(tab)
    for widget in widgets:
        if widget.tab.lower() not in {value.lower() for value in tabs}:
            tabs.append(widget.tab)
    tabs = tabs[:8] or ["Overview"]

    schedule = str(plan.get("refresh_schedule") or "manual").strip().lower()
    visibility = str(plan.get("visibility") or "personal").strip().lower()
    confidence = _confidence(plan.get("confidence"))
    name = re.sub(r"\s+", " ", str(plan.get("name") or "Analysis dashboard")).strip()[:120]
    return DashboardPlan(
        name=name or "Analysis dashboard",
        widgets=tuple(widgets),
        tabs=tuple(tabs),
        refresh_schedule=schedule if schedule in _SCHEDULES else "manual",
        visibility=visibility if visibility in _VISIBILITIES else "personal",
        confidence=confidence,
    ), ""


def looks_like_multi_widget_request(text: str) -> bool:
    value = str(text or "")
    return bool(
        re.search(r"[;\n]", value)
        or len(re.findall(r",", value)) >= 2
        or re.search(r"\b(?:plus|along with)\b", value, re.I)
        or re.search(r"\band\s+(?:a|an|another)?\s*(?:chart|visual|table|kpi|trend|breakdown)\b", value, re.I)
    )


def _confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number:
        return 0.0
    return max(0.0, min(1.0, number))


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
