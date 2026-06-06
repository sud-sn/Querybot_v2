"""
core/export.py

Pure CSV export utilities for the "Download CSV" chip (Sprint E).

Design principles
─────────────────
• rows_to_csv() is a pure function — no DB, no LLM, no file I/O.
• Column format hints (currency, percentage) are applied so the CSV looks
  the same as the on-screen table, not as raw floats.
• The original rows are never mutated.

Public API
──────────
  rows_to_csv(rows, column_formats={}) → str
      Convert a list of row dicts to a CSV string.

  build_csv_filename(question) → str
      Derive a filesystem-safe filename from a natural-language question.
"""

from __future__ import annotations

import csv
import io
import re
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# Core transform
# ══════════════════════════════════════════════════════════════════════════════

def rows_to_csv(
    rows: list[dict],
    *,
    column_formats: dict | None = None,
) -> str:
    """
    Convert a list of row dicts to a RFC-4180-compliant CSV string.

    All values are emitted as strings.  Optional ``column_formats`` applies
    human-friendly formatting (same as the on-screen table) before writing:
      • "currency"   → ``$1,234.56``
      • "percentage" → ``12.34%``
      • all others   → ``str(value)``

    None values become empty strings.

    Parameters
    ──────────
    rows           : result rows (not mutated)
    column_formats : optional header → format_type map

    Returns
    ───────
    UTF-8 CSV string, header row first, newline-terminated.
    Returns an empty string when ``rows`` is empty or ``None``.
    """
    if not rows:
        return ""

    headers = list(rows[0].keys())
    fmts = column_formats or {}

    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=headers,
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()

    for row in rows:
        out_row = {h: _format_cell(row.get(h), fmts.get(h, "")) for h in headers}
        writer.writerow(out_row)

    return buf.getvalue()


def _format_cell(val: Any, fmt: str) -> str:
    """Convert a single cell value to a display-ready CSV string."""
    if val is None:
        return ""
    raw = str(val).replace(",", "")
    if fmt == "currency":
        try:
            return f"${float(raw):,.2f}"
        except (TypeError, ValueError):
            pass
    if fmt == "percentage":
        try:
            return f"{float(raw):,.2f}%"
        except (TypeError, ValueError):
            pass
    return str(val)


# ══════════════════════════════════════════════════════════════════════════════
# Filename helper
# ══════════════════════════════════════════════════════════════════════════════

def build_csv_filename(question: str) -> str:
    """
    Derive a filesystem-safe filename from a natural-language question.

    Rules
    ─────
    1. Strip non-word characters.
    2. Lowercase and collapse whitespace/hyphens to underscores.
    3. Cap at 60 characters (keeps the filename shell-friendly).
    4. Append ".csv".

    Examples
    ────────
    "What is total revenue by region?" → "what_is_total_revenue_by_region.csv"
    "Top 10 products"                  → "top_10_products.csv"
    ""                                 → "querybot_result.csv"
    """
    safe = re.sub(r"[^\w\s-]", "", (question or "").lower())
    safe = re.sub(r"[\s_-]+", "_", safe).strip("_")
    safe = safe[:60].rstrip("_")
    return (safe or "querybot_result") + ".csv"
