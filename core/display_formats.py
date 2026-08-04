"""Governed display-format contracts for cached analytical results.

Formatting is deliberately presentation-only: specs are validated here and
stored beside an immutable result snapshot. Raw values and source SQL are not
changed, and no database call is required.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any


DATE_STYLES = {
    "month_year_short", "month_year_long", "iso", "day_month_year",
    "month_day_year", "day_month_name_year", "year", "month_name",
}
NUMBER_STYLES = {"standard", "compact"}
CURRENCY_CODES = {"USD", "INR", "EUR", "GBP", "CAD", "AUD", "JPY"}
DISPLAY_TYPES = {"date", "number", "currency", "percentage", "text"}

_CURRENCY_ALIASES = {
    "dollar": "USD", "dollars": "USD", "usd": "USD",
    "rupee": "INR", "rupees": "INR", "inr": "INR",
    "euro": "EUR", "euros": "EUR", "eur": "EUR",
    "pound": "GBP", "pounds": "GBP", "gbp": "GBP",
    "cad": "CAD", "aud": "AUD", "yen": "JPY", "jpy": "JPY",
}


def normalize_display_format(spec: dict | None) -> dict[str, Any]:
    """Return a small allow-listed display spec, or an empty dict."""
    if not isinstance(spec, dict):
        return {}
    kind = str(spec.get("type") or "").strip().lower()
    if kind not in DISPLAY_TYPES:
        return {}
    out: dict[str, Any] = {"type": kind}
    if kind == "date":
        style = str(spec.get("style") or "iso").strip().lower()
        out["style"] = style if style in DATE_STYLES else "iso"
    elif kind in {"number", "currency", "percentage"}:
        try:
            digits = int(spec.get("fraction_digits", 2))
        except (TypeError, ValueError):
            digits = 2
        out.update({
            "style": "compact" if str(spec.get("style") or "standard").lower() == "compact" else "standard",
            "fraction_digits": max(0, min(digits, 6)),
            "grouping": bool(spec.get("grouping", True)),
        })
        if kind == "currency":
            code = str(spec.get("currency_code") or "").strip().upper()
            if code in CURRENCY_CODES:
                out["currency_code"] = code
            out["accounting"] = bool(spec.get("accounting", False))
        elif kind == "percentage":
            scale = str(spec.get("scale") or "as_is").strip().lower()
            out["scale"] = scale if scale in {"as_is", "fraction"} else "as_is"
    return out


def coarse_format(spec: dict | None) -> str:
    normalized = normalize_display_format(spec)
    return str(normalized.get("type") or "number")


def parse_format_request(text: str) -> dict[str, Any] | None:
    """Parse common natural-language formatting requests without an LLM.

    The return value may be intentionally incomplete (for example currency
    without a currency_code); the executor then asks a targeted clarification.
    """
    value = " ".join(str(text or "").strip().split())
    lower = value.casefold()
    if not value or not re.search(
        r"\b(?:format|display|show|give|provide|present|return|output|change|convert|round|use|make)\b", lower
    ):
        return None
    strong_format_verb = bool(re.search(r"\b(?:format|change|convert|round|use|make)\b", lower))
    if not strong_format_verb and re.search(
        r"\b(?:contribution|ratio|average|aggregate|total|sum|count|group|filter|sort|top|bottom)\b",
        lower,
    ):
        return None
    if not strong_format_verb and not re.search(r"\b(?:as|in|into|with)\b", lower):
        # Do not steal analytical requests such as "show percentage
        # contribution" from the result-command engine.
        return None

    spec: dict[str, Any] = {}
    if re.search(r"\b(?:yyyy[- /]?mm|year[- /]?month|iso month)\b", lower):
        spec = {"type": "date", "style": "iso"}
    elif re.search(r"\b(?:full month(?: name)?(?: and| with)? year|month name and year)\b", lower):
        spec = {"type": "date", "style": "month_year_long"}
    elif re.search(r"\b(?:month and year|month year|mmm[- /]?yy)\b", lower):
        spec = {"type": "date", "style": "month_year_short"}
    elif re.search(r"\b(?:dd[- /]mmm[- /]yyyy|day month name year)\b", lower):
        spec = {"type": "date", "style": "day_month_name_year"}
    elif re.search(r"\b(?:dd[- /]mm[- /]yyyy|day month year)\b", lower):
        spec = {"type": "date", "style": "day_month_year"}
    elif re.search(r"\b(?:mm[- /]dd[- /]yyyy|month day year)\b", lower):
        spec = {"type": "date", "style": "month_day_year"}
    elif re.search(r"\b(?:date format|as a date|format the date)\b", lower):
        spec = {"type": "date"}
    elif re.search(r"\b(?:percent|percentage)\b", lower):
        spec = {"type": "percentage"}
    elif any(re.search(rf"\b{re.escape(alias)}\b", lower) for alias in _CURRENCY_ALIASES) or re.search(r"\bcurrency\b", lower):
        spec = {"type": "currency"}
        for alias, code in _CURRENCY_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", lower):
                spec["currency_code"] = code
                break
    elif re.search(r"\b(?:decimal places?|decimals?|whole numbers?|no decimals?|number format|compact numbers?|thousands separators?)\b", lower):
        spec = {"type": "number"}
    else:
        return None

    digits = re.search(r"\b(\d)\s*(?:decimal places?|decimals?|dp)\b", lower)
    if digits:
        spec["fraction_digits"] = int(digits.group(1))
    elif re.search(r"\b(?:whole numbers?|no decimals?|zero decimals?)\b", lower):
        spec["fraction_digits"] = 0
    if re.search(r"\b(?:compact|abbreviat(?:e|ed)|in k(?: and)? m|thousands and millions)\b", lower):
        spec["style"] = "compact"
    if re.search(r"\b(?:without|no)\s+(?:thousand )?separators?\b", lower):
        spec["grouping"] = False
    if re.search(r"\baccounting\b", lower):
        spec["accounting"] = True
    if spec.get("type") == "percentage":
        if re.search(r"\b(?:fraction|ratio|0\s*(?:to|-)\s*1)\b", lower):
            spec["scale"] = "fraction"
        elif re.search(r"\b(?:already|as is|0\s*(?:to|-)\s*100)\b", lower):
            spec["scale"] = "as_is"

    target = _extract_target(value)
    return {"spec": spec, "target_text": target}


def parse_format_requests(text: str) -> list[dict[str, Any]]:
    """Parse one or more independently targeted format instructions.

    Natural follow-ups commonly combine presentation changes, for example
    ``format month as MMM-YY and revenue as USD currency with no decimals``.
    Each clause retains its own target and allow-listed format contract.
    """
    value = " ".join(str(text or "").strip().split())
    if not value:
        return []
    # Protect conjunctions that are part of a single format name before
    # splitting independently targeted clauses.
    protected = re.sub(r"\bmonth\s+and\s+year\b", "month __QB_AND__ year", value, flags=re.I)
    clauses = re.split(
        r"\s*(?:;|,?\s+and)\s+(?=(?:(?:format|display|show|give|provide|present|return|output|change|convert|round|use|make)\b|(?:the\s+)?[A-Za-z_][A-Za-z0-9_ ]{0,60}\s+(?:as|to|in|with)\s+(?:USD|INR|EUR|GBP|CAD|AUD|JPY|currency|percent|percentage|number|MMM|YYYY|DD)))",
        protected,
        flags=re.I,
    )
    parsed: list[dict[str, Any]] = []
    for clause in clauses:
        clause = clause.replace("__QB_AND__", "and")
        candidate = clause
        if not re.match(r"^(?:format|display|show|give|provide|present|return|output|change|convert|round|use|make)\b", candidate, re.I):
            candidate = f"format {candidate}"
        request = parse_format_request(candidate)
        if request:
            parsed.append(request)
    if len(parsed) > 1:
        return parsed
    single = parse_format_request(value)
    return [single] if single else []


def _extract_target(text: str) -> str:
    patterns = (
        r"\b(?:format|change|display|show|give|provide|present|return|output|convert|round|make)\s+(?:me\s+)?(?:the\s+)?(.+?)\s+(?:as|to|in|with|into)\s+",
        r"\b(?:use|apply)\s+.+?\s+(?:for|on|to)\s+(?:the\s+)?(.+?)(?:\s*[.!?]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            target = re.sub(
                r"\b(?:this|that|the|current|previous|above|cached)\s+"
                r"(?:result|answer|table|data|chart)\b",
                "",
                match.group(1),
                flags=re.I,
            )
            return " ".join(target.split()).strip(" ,")
    return ""


def looks_temporal(value: Any) -> bool:
    if isinstance(value, (date, datetime)):
        return True
    text = str(value or "").strip()
    return bool(re.fullmatch(r"\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?", text))


def candidate_columns(rows: list[dict], kind: str) -> list[str]:
    if not rows:
        return []
    headers = list(rows[0].keys())
    if kind == "date":
        return [
            h for h in headers
            if re.search(r"\b(?:date|month|period|year|week|quarter)\b", h.replace("_", " "), re.I)
            or any(looks_temporal(row.get(h)) for row in rows[:20] if row.get(h) not in (None, ""))
        ]
    if kind in {"number", "currency", "percentage"}:
        output = []
        for header in headers:
            values = [row.get(header) for row in rows[:20] if row.get(header) not in (None, "")]
            if values:
                try:
                    if any(float(str(value).replace(",", "")) == float(str(value).replace(",", "")) for value in values):
                        output.append(header)
                except (TypeError, ValueError):
                    pass
        return output
    return headers
