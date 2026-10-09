"""A short written summary of an answer, by the workspace's AI, from what the answer already holds.

The answer's sentence and findings are worked out from its rows (builder.py, insights.py,
drivers.py); the summary says in two or three plain sentences what they mean together, for a
reader deciding what to do next. It is written from the answer as shown -- its sentence, its
findings, its notes and up to 25 of its rows -- and from nothing else, and every figure in it
must be one the answer already holds (rounded as a reader rounds: "$1.2M" for 1,234,567): a
summary with any other number is not shown. It follows the answer, so the answer never waits
for it.

Where the workspace keeps member values from the AI (a regulated tenant, or value indexing
switched off), an answer that names members (stores, customers, products) gets no summary:
its rows would have to reach the AI to write one. An answer of numbers alone (a total, a
monthly series) still gets one.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

log = logging.getLogger("querybot.core2")

MAX_ROWS = 25
MAX_WORDS = 90
SMALL = 12.0                 # "the top 3", "2 of the 5 stores": counts, not claims about the data
SYSTEM = """You write the summary shown under an answer from a company's business data.

You are given the reader's question, the answer's own sentence, the findings already worked out
from its rows, its notes, and some of its rows. Write two or three short sentences, in plain
business English, that say what the answer means together and what a reader would look at or do
next. Say it to the reader directly.

Rules:
- Use only figures that appear in what you are given, written the same way or rounded as a reader
  rounds them. Never work out a new figure: no sums, differences, averages or percentages of your own.
- Do not repeat the answer's sentence; build on it.
- No headings, lists, markdown or quotation marks. Do not mention SQL, queries, tables or columns.
- If the notes say the data is partial or a period is missing, say what that means for the reading.
Reply with the sentences only."""


def names_members(payload: dict[str, Any]) -> bool:
    """Does the answer show members (a store, a customer, a product), not only numbers and periods?"""
    data = payload.get("data") or {}
    formats = data.get("column_formats") or {}
    display = data.get("display_formats") or {}
    for header in data.get("headers") or []:
        if (display.get(header) or {}).get("type") == "date":
            continue
        if formats.get(header) in (None, "text"):
            return True
    return False


def eligible(payload: dict[str, Any], *, values_allowed: bool) -> bool:
    """Does this answer get a summary, given what may reach the workspace's AI?"""
    data = payload.get("data") or {}
    if payload.get("unsupported") or payload.get("engine") != "core2" or not data.get("rows"):
        return False
    return values_allowed or not names_members(payload)


def material(payload: dict[str, Any], question: str) -> str:
    """What the AI is given: the answer as shown, and nothing else."""
    answer = payload.get("answer") or {}
    data = payload.get("data") or {}
    labels = data.get("header_labels") or {}
    headers = list(data.get("headers") or [])
    parts = [f"Question: {question}", f"Answer: {answer.get('headline') or ''}"]
    if answer.get("comparison"):
        parts.append(f"Compared: {answer.get('short_value') or ''} {answer['comparison']}".strip())
    findings = [str(f) for f in payload.get("key_insights") or [] if f]
    if findings:
        parts.append("Findings:\n" + "\n".join(f"- {f}" for f in findings))
    notes = [str(n) for n in payload.get("coverage_caveats") or [] if n]
    if notes:
        parts.append("Notes:\n" + "\n".join(f"- {n}" for n in notes))
    rows = list(data.get("rows") or [])[:MAX_ROWS]
    if rows and headers:
        shown = [{labels.get(h, h): r.get(h) for h in headers} for r in rows]
        more = int(data.get("total_rows") or len(rows)) - len(rows)
        parts.append(f"Rows ({len(rows)} shown" + (f", {more} more not shown" if more > 0 else "") + "):\n"
                     + "\n".join(json.dumps(r, default=str, ensure_ascii=False) for r in shown))
    return "\n\n".join(parts)


_FIGURE = re.compile(r"(?<![\w.])[-+−]?\$?\d[\d,]*(?:\.\d+)?\s?(?:[kKmMbB](?:n|illion)?\b|thousand\b|million\b|billion\b)?%?")
_SCALE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9}


def figures(text: str) -> list[tuple[float, float]]:
    """Every figure a text states, with "K", "M" and "B" read as the amounts they stand for, and half the
    step it is written to ("$1.2M": 50,000; "33%": 0.5), which is how far rounding may have moved it."""
    out = []
    for token in _FIGURE.findall(text or ""):
        cleaned = token.replace("−", "-").replace("$", "").replace("%", "").strip()
        suffix = re.search(r"[a-zA-Z]+$", cleaned)
        scale = 1.0
        if suffix:
            scale = _SCALE.get(suffix.group(0).lower(), 1.0)
            cleaned = cleaned[: suffix.start()].strip()
        digits = cleaned.replace(",", "")
        decimals = len(digits.split(".", 1)[1]) if "." in digits else 0
        try:
            out.append((float(digits) * scale, 0.5 * 10 ** -decimals * scale))
        except ValueError:
            continue
    return out


def _known(payload: dict[str, Any], given: str) -> list[float]:
    known = [value for value, _ in figures(given)]
    for row in (payload.get("data") or {}).get("rows") or []:
        for value in row.values():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                known.append(float(value))
    return known


def _held(value: float, step: float, known: list[float]) -> bool:
    if abs(value) <= SMALL:
        return True
    # The same figure, or one rounded to the step it is written to ($1.2M for 1,234,567; 33% for 33.4%).
    return any(abs(value - k) <= max(step, 0.51) * 1.0001 for k in known)


def checked(text: str, payload: dict[str, Any], given: str) -> str | None:
    """The summary when every figure in it is one the answer holds; None otherwise."""
    text = " ".join(str(text or "").split()).strip().strip('"')
    if not text or len(text.split()) > MAX_WORDS or text.startswith(("#", "-", "*")):
        return None
    known = _known(payload, given)
    stray = [v for v, step in figures(text) if not _held(v, step, known)]
    if stray:
        log.info("A summary was not shown: it states %s, which the answer does not hold", stray[:3])
        return None
    return text


def write(payload: dict[str, Any], question: str, complete: Callable[[str, str], str]) -> str | None:
    """The summary of ``payload`` by the workspace's AI, checked; None when there is none to show."""
    given = material(payload, question)
    try:
        said = complete(SYSTEM, given)
    except Exception as exc:  # noqa: BLE001 - the answer stands without its summary
        log.warning("The workspace's AI could not write a summary: %s", exc)
        return None
    return checked(said, payload, given)
