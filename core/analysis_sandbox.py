"""Isolated, allowlisted analysis for governed result rows.

A spawned worker receives only already-released rows. It runs either a fixed
operation or source accepted by the deliberately small governed-Python
language, has hard row/column/time/output bounds, and returns JSON-shaped
statistics suitable for an auditable artifact.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import multiprocessing
import platform
import re
import statistics
from dataclasses import dataclass
from itertools import combinations
from typing import Any

_OPERATIONS = {"profile", "correlation", "outliers", "trend"}
_MAX_ROWS = 5000
_MAX_COLUMNS = 64
_MAX_CODE_CHARS = 8000
_MAX_AST_NODES = 1200
_MAX_OUTPUT_ROWS = 2000
_MAX_OUTPUT_BYTES = 2_000_000

_SAFE_CALLS = {
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "get",
    "group_sum", "int", "items", "keys", "len", "list", "max", "mean",
    "median", "min", "number", "percentile", "range", "round", "sorted",
    "stdev", "str", "sum", "top", "tuple", "values", "zip",
}
_ALLOWED_AST_NODES = {
    ast.Module, ast.Assign, ast.AugAssign, ast.Expr, ast.If, ast.For,
    ast.Break, ast.Continue, ast.Pass,
    ast.Name, ast.Load, ast.Store, ast.Constant,
    ast.List, ast.Tuple, ast.Dict, ast.Set,
    ast.Subscript, ast.Slice,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.ListComp, ast.DictComp, ast.SetComp, ast.comprehension,
    ast.Call, ast.keyword,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
    ast.UAdd, ast.USub, ast.Not,
    ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.In, ast.NotIn, ast.Is, ast.IsNot,
    ast.JoinedStr, ast.FormattedValue,
}


@dataclass(frozen=True)
class AnalysisResult:
    operation: str
    summary: str
    rows: list[dict]
    metadata: dict


@dataclass(frozen=True)
class PythonValidation:
    code_hash: str
    assigned_names: tuple[str, ...]
    helper_calls: tuple[str, ...]
    ast_nodes: int


class UnsafeAnalysisCode(ValueError):
    """Raised when custom analysis code exceeds the governed language."""


def validate_python_analysis(code: str) -> PythonValidation:
    """Validate custom analysis source without executing it.

    The governed language deliberately has no attributes, imports, functions,
    classes, exceptions, context managers, while loops, or dynamic execution.
    Input is the read-only ``rows`` list and output must be assigned to
    ``result``.
    """
    source = str(code or "").strip()
    if not source:
        raise UnsafeAnalysisCode("Python analysis code is empty.")
    if len(source) > _MAX_CODE_CHARS:
        raise UnsafeAnalysisCode(f"Python analysis code exceeds {_MAX_CODE_CHARS} characters.")
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise UnsafeAnalysisCode(f"Python syntax error on line {exc.lineno or 1}.") from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > _MAX_AST_NODES:
        raise UnsafeAnalysisCode("Python analysis is too complex for the governed worker.")

    assigned: set[str] = set()
    calls: set[str] = set()
    result_assigned = False
    iterative_nodes = (ast.For, ast.ListComp, ast.DictComp, ast.SetComp)
    for outer in nodes:
        if isinstance(outer, (ast.ListComp, ast.DictComp, ast.SetComp)) and len(outer.generators) > 1:
            raise UnsafeAnalysisCode("Nested loops or comprehensions are not allowed.")
        if isinstance(outer, iterative_nodes) and any(
            child is not outer and isinstance(child, iterative_nodes)
            for child in ast.walk(outer)
        ):
            raise UnsafeAnalysisCode("Nested loops or comprehensions are not allowed.")
    for node in nodes:
        if type(node) not in _ALLOWED_AST_NODES:
            label = type(node).__name__
            raise UnsafeAnalysisCode(f"{label} is not allowed in governed Python analysis.")
        if isinstance(node, ast.Attribute):
            raise UnsafeAnalysisCode("Object attributes are not allowed in governed Python analysis.")
        if isinstance(node, ast.Name):
            if node.id.startswith("_") or "__" in node.id:
                raise UnsafeAnalysisCode("Private or dunder names are not allowed.")
            if isinstance(node.ctx, ast.Store):
                if node.id in (_SAFE_CALLS | {"rows"}):
                    raise UnsafeAnalysisCode(f"The protected name '{node.id}' cannot be reassigned.")
                assigned.add(node.id)
                result_assigned = result_assigned or node.id == "result"
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_CALLS:
                raise UnsafeAnalysisCode("Only documented analysis helper calls are allowed.")
            calls.add(node.func.id)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str) and len(node.value) > 2000:
                raise UnsafeAnalysisCode("String literals are limited to 2,000 characters.")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            operands = (node.left, node.right)
            multiplier = next((value.value for value in operands if isinstance(value, ast.Constant) and isinstance(value.value, int)), 0)
            if abs(multiplier) > 1_000:
                raise UnsafeAnalysisCode("Constant multiplication is limited to 1,000.")
    if not result_assigned:
        raise UnsafeAnalysisCode("Assign the final JSON-shaped output to 'result'.")
    return PythonValidation(
        code_hash=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        assigned_names=tuple(sorted(assigned)),
        helper_calls=tuple(sorted(calls)),
        ast_nodes=len(nodes),
    )


def run_governed_python_analysis(
    rows: list[dict], code: str, *, timeout_seconds: float = 8.0
) -> AnalysisResult:
    """Execute validated analysis code in a spawned, hard-time-bounded worker."""
    validation = validate_python_analysis(code)
    safe_rows = _bounded_rows(rows)
    if not safe_rows:
        raise ValueError("There are no result rows to analyze.")
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_python_worker_entry,
        args=(child, safe_rows, str(code)),
        daemon=True,
    )
    process.start()
    child.close()
    try:
        if not parent.poll(max(0.5, float(timeout_seconds))):
            process.terminate()
            process.join(timeout=1)
            raise TimeoutError("The governed Python analysis exceeded its time limit.")
        payload = parent.recv()
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=1)
    if not payload.get("ok"):
        error_type = str(payload.get("error_type") or "analysis_error")
        message = str(payload.get("error") or "Python analysis failed.")
        if error_type == "unsafe_code":
            raise UnsafeAnalysisCode(message)
        raise ValueError(message)
    return AnalysisResult(
        operation="python",
        summary=str(payload.get("summary") or "Custom governed Python analysis completed."),
        rows=list(payload.get("rows") or []),
        metadata={
            **dict(payload.get("metadata") or {}),
            "code_hash": validation.code_hash,
            "ast_nodes": validation.ast_nodes,
            "helper_calls": list(validation.helper_calls),
        },
    )


def plan_analysis_operations(text: str, rows: list[dict]) -> list[str]:
    value = str(text or "").lower()
    requested: list[str] = []
    if re.search(r"\b(correlat(?:e|ed|es|ing|ion|ions)?|relationships?|associations?)\b", value):
        requested.append("correlation")
    if re.search(r"\b(outliers?|anomal(?:y|ies|ous)?|unusual|extremes?)\b", value):
        requested.append("outliers")
    if re.search(r"\b(trends?|changes?|growth|over time|movements?)\b", value):
        requested.append("trend")
    if re.search(r"\b(deep|thorough|full|profile|analyse|analyze)\b", value) or not requested:
        requested.insert(0, "profile")
    return list(dict.fromkeys(requested))[:3]


def run_isolated_analysis(
    rows: list[dict], operation: str, *, timeout_seconds: float = 8.0
) -> AnalysisResult:
    operation = str(operation or "").lower()
    if operation not in _OPERATIONS:
        raise ValueError("Unsupported analysis operation.")
    safe_rows = _bounded_rows(rows)
    if not safe_rows:
        raise ValueError("There are no result rows to analyze.")
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker_entry,
        args=(child, safe_rows, operation),
        daemon=True,
    )
    process.start()
    child.close()
    try:
        if not parent.poll(max(0.5, float(timeout_seconds))):
            process.terminate()
            process.join(timeout=1)
            raise TimeoutError("The isolated analysis exceeded its time limit.")
        payload = parent.recv()
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=1)
    if not payload.get("ok"):
        raise ValueError(str(payload.get("error") or "Analysis failed."))
    return AnalysisResult(
        operation=operation,
        summary=str(payload.get("summary") or ""),
        rows=list(payload.get("rows") or []),
        metadata=dict(payload.get("metadata") or {}),
    )


def _worker_entry(connection, rows: list[dict], operation: str) -> None:
    try:
        _apply_worker_limits()
        result = _analyze(rows, operation)
        connection.send({"ok": True, **result})
    except Exception as exc:
        connection.send({"ok": False, "error": str(exc)[:500]})
    finally:
        connection.close()


def _python_worker_entry(connection, rows: list[dict], code: str) -> None:
    try:
        _apply_worker_limits()
        validation = validate_python_analysis(code)
        environment = _python_environment(rows)
        compiled = compile(ast.parse(code, mode="exec"), "<governed-analysis>", "exec")
        exec(compiled, {"__builtins__": {}}, environment)
        output = _normalise_python_output(environment.get("result"))
        encoded = json.dumps(output, ensure_ascii=True, allow_nan=False, default=str)
        if len(encoded.encode("utf-8")) > _MAX_OUTPUT_BYTES:
            raise ValueError("Python analysis output exceeds the 2 MB release limit.")
        connection.send({
            "ok": True,
            "summary": (
                f"Governed Python produced {len(output)} derived row"
                f"{'s' if len(output) != 1 else ''} from {len(rows)} released input rows."
            ),
            "rows": output,
            "metadata": {
                "input_rows": len(rows),
                "output_rows": len(output),
                "output_columns": list(output[0]) if output else [],
                "code_hash": validation.code_hash,
                "ast_nodes": validation.ast_nodes,
                "helper_calls": list(validation.helper_calls),
                "database_queried": False,
                "rows_sent_to_llm": 0,
            },
        })
    except UnsafeAnalysisCode as exc:
        connection.send({"ok": False, "error_type": "unsafe_code", "error": str(exc)[:500]})
    except BaseException as exc:
        connection.send({
            "ok": False,
            "error_type": type(exc).__name__,
            "error": f"{type(exc).__name__}: {str(exc)}"[:500],
        })
    finally:
        connection.close()


_WINDOWS_JOB_HANDLE = None


def _apply_worker_limits() -> None:
    """Apply hard process limits on POSIX and Windows; fail closed."""
    if platform.system() == "Windows":
        _apply_windows_job_limits()
        return
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
        memory = 512 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
    except (ImportError, AttributeError, OSError, ValueError) as exc:
        raise RuntimeError("Analysis worker resource limits could not be established.") from exc


def _apply_windows_job_limits() -> None:
    """Put the current worker in a memory/CPU/process-bounded Windows Job."""
    import ctypes
    from ctypes import wintypes

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
    limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    limits.BasicLimitInformation.PerProcessUserTimeLimit = 4 * 10_000_000
    limits.BasicLimitInformation.ActiveProcessLimit = 1
    limits.ProcessMemoryLimit = 512 * 1024 * 1024
    limits.BasicLimitInformation.LimitFlags = (
        0x00000002  # JOB_OBJECT_LIMIT_PROCESS_TIME
        | 0x00000008  # JOB_OBJECT_LIMIT_ACTIVE_PROCESS
        | 0x00000100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
        | 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    )
    if not kernel32.SetInformationJobObject(
        job, 9, ctypes.byref(limits), ctypes.sizeof(limits),
    ):
        error = ctypes.get_last_error()
        kernel32.CloseHandle(job)
        raise OSError(error, "SetInformationJobObject failed")
    if not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
        error = ctypes.get_last_error()
        kernel32.CloseHandle(job)
        raise OSError(error, "AssignProcessToJobObject failed")
    global _WINDOWS_JOB_HANDLE
    _WINDOWS_JOB_HANDLE = job


def _python_environment(rows: list[dict]) -> dict[str, Any]:
    return {
        "rows": rows,
        "result": None,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "get": _safe_get,
        "group_sum": _safe_group_sum,
        "int": int,
        "items": _safe_items,
        "keys": _safe_keys,
        "len": len,
        "list": list,
        "max": max,
        "mean": _safe_mean,
        "median": _safe_median,
        "min": min,
        "number": _number,
        "percentile": _safe_percentile,
        "range": _safe_range,
        "round": round,
        "sorted": sorted,
        "stdev": _safe_stdev,
        "str": str,
        "sum": sum,
        "top": _safe_top,
        "tuple": tuple,
        "values": _safe_values,
        "zip": zip,
    }


def _normalise_python_output(value: Any) -> list[dict]:
    if isinstance(value, dict):
        rows = [value]
    elif isinstance(value, (list, tuple)):
        rows = list(value)
    elif value is None:
        raise ValueError("Python analysis did not assign a value to 'result'.")
    else:
        rows = [{"value": value}]
    if len(rows) > _MAX_OUTPUT_ROWS:
        raise ValueError(f"Python analysis output exceeds {_MAX_OUTPUT_ROWS} rows.")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Python analysis output must be a dictionary or a list of dictionaries.")
    normalised: list[dict] = []
    for row in rows:
        if len(row) > _MAX_COLUMNS:
            raise ValueError(f"Python analysis output exceeds {_MAX_COLUMNS} columns.")
        cleaned = {
            str(key)[:120]: _json_value(cell)
            for key, cell in row.items()
            if str(key) and not str(key).startswith("_")
        }
        normalised.append(cleaned)
    return json.loads(json.dumps(normalised, ensure_ascii=True, allow_nan=False))


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value[:1000]]
    if isinstance(value, dict):
        return {
            str(key)[:120]: _json_value(item)
            for key, item in list(value.items())[:1000]
        }
    return str(value)[:2000]


def _safe_get(mapping: Any, key: Any, default: Any = None) -> Any:
    return mapping.get(key, default) if isinstance(mapping, dict) else default


def _safe_items(mapping: Any) -> list[tuple[Any, Any]]:
    return list(mapping.items())[:_MAX_OUTPUT_ROWS] if isinstance(mapping, dict) else []


def _safe_keys(mapping: Any) -> list[Any]:
    return list(mapping.keys())[:_MAX_OUTPUT_ROWS] if isinstance(mapping, dict) else []


def _safe_values(value: Any, column: Any = None) -> list[Any]:
    if column is not None and isinstance(value, list):
        return [row.get(column) for row in value if isinstance(row, dict) and row.get(column) is not None]
    if isinstance(value, dict):
        return list(value.values())[:_MAX_OUTPUT_ROWS]
    return list(value)[:_MAX_ROWS] if isinstance(value, (list, tuple)) else []


def _numeric_values(values: Any) -> list[float]:
    return [number for number in (_number(value) for value in list(values or [])[:_MAX_ROWS]) if number is not None]


def _safe_mean(values: Any) -> float | None:
    numeric = _numeric_values(values)
    return statistics.fmean(numeric) if numeric else None


def _safe_median(values: Any) -> float | None:
    numeric = _numeric_values(values)
    return statistics.median(numeric) if numeric else None


def _safe_stdev(values: Any) -> float:
    numeric = _numeric_values(values)
    return statistics.stdev(numeric) if len(numeric) > 1 else 0.0


def _safe_percentile(values: Any, percentile_value: Any) -> float | None:
    numeric = sorted(_numeric_values(values))
    if not numeric:
        return None
    percentile_number = _number(percentile_value)
    if percentile_number is None or percentile_number < 0 or percentile_number > 100:
        raise ValueError("Percentile must be between 0 and 100.")
    position = (len(numeric) - 1) * percentile_number / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return numeric[lower]
    return numeric[lower] + (numeric[upper] - numeric[lower]) * (position - lower)


def _safe_range(*args: Any) -> range:
    numeric = [int(value) for value in args]
    value = range(*numeric)
    if len(value) > 10_000:
        raise ValueError("range() is limited to 10,000 iterations.")
    return value


def _safe_group_sum(rows: Any, group_column: Any, value_column: Any) -> list[dict]:
    totals: dict[str, float] = {}
    for row in list(rows or [])[:_MAX_ROWS]:
        if not isinstance(row, dict):
            continue
        group = str(row.get(group_column, "Unknown"))[:240]
        number = _number(row.get(value_column))
        if number is not None:
            totals[group] = totals.get(group, 0.0) + number
    return [
        {str(group_column): group, str(value_column): value}
        for group, value in totals.items()
    ]


def _safe_top(rows: Any, count: Any = 10, column: Any = None, descending: Any = True) -> list[Any]:
    bounded = list(rows or [])[:_MAX_ROWS]
    limit = max(0, min(int(count), _MAX_OUTPUT_ROWS))
    if column is None:
        return bounded[:limit]
    return sorted(
        bounded,
        key=lambda row: (_number(row.get(column)) if isinstance(row, dict) else None) or 0,
        reverse=bool(descending),
    )[:limit]


def _bounded_rows(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    headers = list(rows[0].keys())[:_MAX_COLUMNS]
    bounded = [{header: row.get(header) for header in headers} for row in rows[:_MAX_ROWS]]
    return json.loads(json.dumps(bounded, default=str))


def _numeric_columns(rows: list[dict]) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    required = 1 if len(rows) == 1 else max(2, math.ceil(len(rows) / 2))
    for column in rows[0]:
        values = [_number(row.get(column)) for row in rows]
        numeric = [value for value in values if value is not None]
        if len(numeric) >= required:
            result[column] = numeric
    return result


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        text = str(value).strip()
        accounting = text.startswith("(") and text.endswith(")")
        text = re.sub(r"(?i)\b(?:USD|INR|EUR|GBP|CAD|AUD|JPY)\b", "", text)
        text = text.translate(str.maketrans("", "", ",$₹€£¥%()"))
        number = float(text.strip())
        if accounting:
            number = -abs(number)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _analyze(rows: list[dict], operation: str) -> dict:
    numeric = _numeric_columns(rows)
    if operation == "profile":
        output = []
        for column, values in numeric.items():
            output.append({
                "column": column,
                "count": len(values),
                "missing": len(rows) - len(values),
                "mean": round(statistics.fmean(values), 4),
                "median": round(statistics.median(values), 4),
                "stdev": round(statistics.stdev(values), 4) if len(values) > 1 else 0,
                "min": min(values),
                "max": max(values),
            })
        return {
            "summary": f"Profiled {len(numeric)} numeric columns across {len(rows)} rows.",
            "rows": output,
            "metadata": {"input_rows": len(rows), "numeric_columns": list(numeric)},
        }
    if operation == "correlation":
        output = []
        for left, right in combinations(numeric, 2):
            pairs = [
                (_number(row.get(left)), _number(row.get(right))) for row in rows
            ]
            pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
            if len(pairs) < 3:
                continue
            corr = _pearson([a for a, _ in pairs], [b for _, b in pairs])
            output.append({"left": left, "right": right, "correlation": round(corr, 4), "strength": _strength(corr), "pairs": len(pairs)})
        output.sort(key=lambda item: abs(item["correlation"]), reverse=True)
        return {
            "summary": f"Compared {len(output)} numeric column pairs; strongest relationships are shown first.",
            "rows": output[:20],
            "metadata": {"input_rows": len(rows), "pair_count": len(output)},
        }
    if operation == "outliers":
        output = []
        for column, values in numeric.items():
            if len(values) < 3:
                continue
            mean = statistics.fmean(values)
            stdev = statistics.stdev(values)
            if not stdev:
                continue
            for index, row in enumerate(rows):
                value = _number(row.get(column))
                if value is None:
                    continue
                zscore = (value - mean) / stdev
                if abs(zscore) >= 2:
                    output.append({"row": index + 1, "column": column, "value": value, "z_score": round(zscore, 3), "direction": "high" if zscore > 0 else "low"})
        output.sort(key=lambda item: abs(item["z_score"]), reverse=True)
        return {
            "summary": f"Found {len(output)} values at least two standard deviations from their column mean.",
            "rows": output[:50],
            "metadata": {"input_rows": len(rows), "outlier_count": len(output), "threshold": 2.0},
        }
    if operation == "trend":
        if not numeric:
            raise ValueError("Trend analysis needs at least one numeric column.")
        value_column = next(iter(numeric))
        dimension = next((column for column in rows[0] if column != value_column), "row")
        output = []
        previous = None
        for index, row in enumerate(rows):
            value = _number(row.get(value_column))
            if value is None:
                continue
            change = None if previous is None else value - previous
            change_pct = None if previous in (None, 0) else change / abs(previous) * 100
            output.append({"period": row.get(dimension, index + 1), "value": value, "change": None if change is None else round(change, 4), "change_pct": None if change_pct is None else round(change_pct, 2)})
            previous = value
        return {
            "summary": f"Calculated sequential change for {value_column} across {len(output)} ordered rows.",
            "rows": output,
            "metadata": {"input_rows": len(rows), "dimension": dimension, "value_column": value_column},
        }
    raise ValueError("Unsupported analysis operation.")


def _pearson(left: list[float], right: list[float]) -> float:
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left)
        * sum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator if denominator else 0.0


def _strength(value: float) -> str:
    absolute = abs(value)
    return "strong" if absolute >= 0.7 else "moderate" if absolute >= 0.4 else "weak"
