"""Metadata-only planning for governed custom Python result analysis."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from core.analysis_sandbox import UnsafeAnalysisCode, validate_python_analysis

_PLAN_KEYS = {"operation", "title", "code", "confidence", "explanation"}


@dataclass(frozen=True)
class PythonAnalysisPlan:
    title: str
    code: str
    explanation: str
    confidence: float
    code_hash: str


@dataclass(frozen=True)
class PythonPlanInput:
    system_prompt: str
    user_prompt: str
    schema: tuple[dict, ...]


def extract_user_python(text: str) -> str:
    value = str(text or "")
    fenced = re.search(r"```(?:python|py)?\s*\n(?P<code>.*?)```", value, re.I | re.S)
    if fenced:
        return fenced.group("code").strip()
    marker = re.search(r"\b(?:run|execute|use)\s+(?:this\s+)?python\s*:\s*(?P<code>.+)$", value, re.I | re.S)
    return marker.group("code").strip() if marker else ""


def build_python_plan_input(text: str, rows: list[dict]) -> PythonPlanInput:
    schema = tuple(_metadata_schema(rows))
    helper_contract = (
        "Available read-only input: rows (list of dictionaries). Required output: assign result "
        "to a dictionary, scalar, or list of dictionaries. Allowed calls only: abs, all, any, "
        "bool, dict, enumerate, float, get(mapping,key,default), group_sum(rows,group_column,value_column), "
        "int, items, keys, len, list, max, mean, median, min, number, percentile, range, round, "
        "sorted, stdev, str, sum, top(rows,count,column,descending), tuple, values(rows,column), zip. "
        "Use get(row, 'column') instead of row.get. Do not use attributes, imports, functions, "
        "classes, exceptions, while loops, files, network, print, eval, exec, or external packages."
    )
    system = (
        "You plan a governed Python calculation over an already-released database result. Return "
        "exactly one JSON object and no markdown. Never write SQL. Never request another database "
        "query. You receive column metadata only—no row values, samples, source SQL, credentials, "
        "or policy data. Preserve the user's requested calculation and use only supplied columns. "
        f"{helper_contract} Allowed JSON keys: operation, title, code, confidence, explanation. "
        "operation must be python_analysis. code must be valid source for the governed language. "
        "confidence is 0.0-1.0; use less than 0.65 when the requested calculation cannot be derived "
        "from the available columns."
    )
    user = json.dumps(
        {
            "request": re.sub(r"\s+", " ", str(text or "")).strip()[:2000],
            "row_count": min(len(rows), 5000),
            "columns": list(schema),
        },
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return PythonPlanInput(system, user, schema)


async def parse_python_analysis_plan(
    text: str,
    rows: list[dict],
    complete: Callable[..., Awaitable[tuple[str, int, int]]],
) -> tuple[PythonAnalysisPlan | None, str]:
    built = build_python_plan_input(text, rows)
    try:
        raw, _, _ = await complete(
            system=built.system_prompt,
            user=built.user_prompt,
            temperature=0.0,
            max_tokens=1200,
        )
    except Exception:
        return None, "The governed Python planner was unavailable."
    return compile_python_plan_response(raw)


def compile_python_plan_response(raw_response: str) -> tuple[PythonAnalysisPlan | None, str]:
    payload = _parse_json_object(raw_response)
    if payload is None:
        return None, "The governed Python planner did not return valid JSON."
    if set(payload) - _PLAN_KEYS:
        return None, "The governed Python planner returned unsupported fields."
    if str(payload.get("operation") or "").lower() != "python_analysis":
        return None, "The governed Python planner returned an unsupported operation."
    code = str(payload.get("code") or "").strip()
    try:
        validation = validate_python_analysis(code)
    except UnsafeAnalysisCode as exc:
        return None, f"The generated analysis was rejected: {exc}"
    confidence = _confidence(payload.get("confidence"))
    title = re.sub(r"\s+", " ", str(payload.get("title") or "Python analysis")).strip()[:120]
    explanation = re.sub(r"\s+", " ", str(payload.get("explanation") or "")).strip()[:500]
    return PythonAnalysisPlan(
        title=title or "Python analysis",
        code=code,
        explanation=explanation,
        confidence=confidence,
        code_hash=validation.code_hash,
    ), ""


def _metadata_schema(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    schema = []
    for column in list(rows[0])[:64]:
        values = [row.get(column) for row in rows[:100] if row.get(column) is not None]
        if values and all(_is_number(value) for value in values):
            kind = "number"
        elif values and all(_is_boolean(value) for value in values):
            kind = "boolean"
        elif re.search(r"date|time|month|year|week|quarter|period", str(column), re.I):
            kind = "date_or_period"
        else:
            kind = "text"
        schema.append({"name": str(column)[:120], "type": kind, "nullable": len(values) < min(len(rows), 100)})
    return schema


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        float(str(value).replace(",", "").replace("$", "").replace("%", ""))
        return True
    except (TypeError, ValueError):
        return False


def _is_boolean(value: Any) -> bool:
    return isinstance(value, bool) or str(value).lower() in {"true", "false"}


def _confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number)) if number == number else 0.0


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
