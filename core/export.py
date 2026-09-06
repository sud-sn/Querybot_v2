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

  content_disposition(filename) → str
      The header value for a download, safe for a non-ASCII filename.
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
    """Convert a single cell value to a display-ready CSV string.

    Deliberately NOT localised, and pinned by a test. A CSV is an interchange
    format: a French decimal comma inside a comma-delimited file is ambiguous
    to every downstream parser, and the reader who exported it may not be the
    one who opens it. The screen is localised; the file stays machine-readable.
    """
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

def ascii_filename(name: str, *, fallback: str = "querybot_result") -> str:
    """The same filename with every non-ASCII character folded away.

    An HTTP header is latin-1, so a filename carrying a letter outside it —
    "Škoda", "Łukasz", any CJK — raises UnicodeEncodeError when the response is
    encoded and the reader gets a 500 instead of their download. Those are
    ordinary values in a European dataset, and the filename is built from the
    question, so the reader supplies them without knowing.

    Accents fold to their base letter rather than being dropped, so "région"
    stays readable as "region". The extension is preserved separately, because
    a stem that folds away to nothing must still come back as a .csv rather
    than as an extensionless fallback the browser will not open.
    """
    import unicodedata

    raw = str(name or "")
    stem, dot, ext = raw.rpartition(".")
    if not dot or len(ext) > 8 or not ext.isalnum():
        stem, ext = raw, ""

    decomposed = unicodedata.normalize("NFKD", stem)
    folded = "".join(c for c in decomposed if not unicodedata.combining(c))
    ascii_stem = folded.encode("ascii", "ignore").decode("ascii")
    ascii_stem = re.sub(r"[^\w.-]+", "_", ascii_stem)
    ascii_stem = re.sub(r"_+", "_", ascii_stem).strip("_ .")
    if not ascii_stem:
        # The fallback is a caller's string and goes through the same filter:
        # it reaches the header on exactly the path where the reader's own
        # name folded away to nothing, so trusting it is trusting the one
        # input nobody looks at.
        ascii_stem = re.sub(r"[^\w.-]+", "_", str(fallback or "")).strip("_ .")
    if not ascii_stem:
        ascii_stem = "download"

    ascii_ext = ext.encode("ascii", "ignore").decode("ascii")
    return f"{ascii_stem}.{ascii_ext}" if ascii_ext else ascii_stem


def content_disposition(filename: str, *, fallback: str = "querybot_result") -> str:
    """A Content-Disposition value that survives a non-ASCII filename.

    RFC 6266: an ASCII ``filename=`` for anything old, and a percent-encoded
    ``filename*=UTF-8''`` that every current browser prefers. Emitting only the
    first loses the accents; emitting only the raw name 500s the request.
    """
    from urllib.parse import quote

    # ascii_filename is the single place a name is made header-safe, fallback
    # included, so there is no second escape here to drift from it.
    quoted = ascii_filename(filename, fallback=fallback)
    encoded = quote(str(filename or quoted), safe="")
    return f"attachment; filename=\"{quoted}\"; filename*=UTF-8''{encoded}"


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
