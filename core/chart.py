"""
Chart module.
Detects whether query results are chartable, generates a matplotlib PNG,
and uploads it to Zoom as a file attachment.
"""

import io
import logging
import tempfile
import os
from typing import Optional

import re as _re

import httpx

log = logging.getLogger("querybot.chart")

# ── Dimension / ID column detection ──────────────────────────────────────────
# Columns whose names end with these suffixes are identifiers, not metrics.
_DIM_SUFFIX_RE = _re.compile(
    r"(?i)(^|_)(id|key|code|num|no|nbr|nr|ref|pk|fk|seq|idx|index|rank|number)$"
)
_DIM_EXACT = frozenset({"id", "key", "code", "number", "num", "no", "ref", "rank", "index"})


def _is_dimension_col(col_name: str, values: list) -> bool:
    """
    Return True when a numeric column is really a dimension/identifier that
    should NOT be plotted as a metric series.

    Two checks (either is enough):
    1. Column name ends with a known ID/key suffix  (e.g. Prescriber_ID, order_no)
    2. High-cardinality integers: >80 % of values are unique AND all are whole
       numbers  (IDs like 200425, 213947 pass this; aggregated dollar amounts don't)
    """
    if _DIM_SUFFIX_RE.search(col_name):
        return True
    if col_name.lower() in _DIM_EXACT:
        return True
    # Cardinality / integer check
    if values:
        try:
            int_vals = [int(float(v)) for v in values if v is not None]
            if len(int_vals) >= 2 and len(int_vals) == len([v for v in values if v is not None]):
                # All values are whole numbers
                float_vals = [float(v) for v in values if v is not None]
                all_int = all(fv == int(fv) for fv in float_vals)
                if all_int and len(set(int_vals)) / len(int_vals) > 0.8:
                    return True
        except (ValueError, TypeError):
            pass
    return False


def _looks_temporal_labels(values: list[str]) -> bool:
    sample = " ".join(v.lower() for v in values[:8] if v)
    if not sample:
        return False
    substr_tokens = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "week", "month", "year", "date", "day",
    ]
    wb_tokens = [
        "jan", "feb", "mar", "apr", "jun", "jul", "aug",
        "sep", "oct", "nov", "dec", "q1", "q2", "q3", "q4",
    ]
    return (
        bool(_re.search(r"\b\d{4}[-/]\d{1,2}([-/]\d{1,2})?\b", sample))
        or any(tok in sample for tok in substr_tokens)
        or any(_re.search(r"\b" + tok + r"\b", sample) for tok in wb_tokens)
    )


# ── Chart detection ───────────────────────────────────────────────────────────

_TREND_RE = _re.compile(
    r"\b(trend|over time|monthly|weekly|daily|yearly|by month|by week|"
    r"by year|by quarter|evolution|progression|growth|history|timeline)\b",
    _re.IGNORECASE,
)
_SHARE_RE = _re.compile(
    r"\b(share|proportion|breakdown|distribution|percent|percentage|"
    r"split|composition|mix|by .{1,20} type|by .{1,20} category)\b",
    _re.IGNORECASE,
)
_SCATTER_RE = _re.compile(
    r"\b(correlat|vs\.?|versus|scatter|relationship between|compare .{1,30} with|"
    r"related to|association between|show .{1,30} vs|x vs y)\b",
    _re.IGNORECASE,
)


def detect_chart_type(rows: list[dict], question: str = "") -> Optional[str]:
    """
    Inspect result columns, row count, and optional question text to choose
    the best chart type.

    Returns: 'bar' | 'line' | 'area' | 'scatter' | 'pie' | 'donut' | None
      • area  — temporal line with prominent fill (good for trend questions)
      • donut — ring-style pie (good for share/breakdown questions)
    """
    if not rows or len(rows) < 2:
        return None

    headers = list(rows[0].keys())
    if len(headers) < 2:
        return None

    numeric_cols, text_cols = [], []
    for h in headers:
        try:
            vals = [float(str(r.get(h) or 0).replace(",", "")) for r in rows if r.get(h) is not None]
            if vals:
                numeric_cols.append(h)
        except (ValueError, TypeError):
            text_cols.append(h)

    # Strip out ID/dimension columns — they parse as numeric but aren't metrics
    numeric_cols = [h for h in numeric_cols
                    if not _is_dimension_col(h, [r.get(h) for r in rows])]

    if not numeric_cols:
        return None

    q = question or ""
    trend_q   = bool(_TREND_RE.search(q))
    share_q   = bool(_SHARE_RE.search(q))
    scatter_q = bool(_SCATTER_RE.search(q))

    # Scatter — explicit correlation/vs question with 2+ numeric cols, OR
    # question signals scatter intent and data has at least 2 numeric cols
    if scatter_q and len(numeric_cols) >= 2:
        return "scatter"

    n = len(rows)
    if text_cols and numeric_cols:
        labels = [str(r.get(text_cols[0], "")) for r in rows]
        temporal = _looks_temporal_labels(labels)

        # Donut / Pie — small categorical sets with a single value column
        if n <= 10 and len(numeric_cols) == 1 and not scatter_q:
            return "donut" if share_q else "pie"

        # Area / Line — temporal data or explicit trend question
        if trend_q or temporal:
            return "area" if n <= 36 else "line"

        return "bar"

    if len(numeric_cols) >= 2:
        return "scatter"
    return None


# ── Chart rendering ───────────────────────────────────────────────────────────

def generate_chart(rows: list[dict], chart_type: str, title: str = "Results") -> Optional[bytes]:
    """
    Render a chart as PNG bytes.
    Returns None if matplotlib is not installed or rendering fails.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed — chart skipped")
        return None

    try:
        return _render(rows, chart_type, title, plt)
    except Exception as e:
        log.error("Chart render error: %s", e)
        return None


def _render(rows, chart_type, title, plt) -> bytes:
    headers = list(rows[0].keys())

    # Separate numeric and text columns — exclude ID/dimension cols from metrics
    numeric_cols, text_cols = [], []
    for h in headers:
        try:
            [float(str(r.get(h) or 0).replace(",", "")) for r in rows if r.get(h) is not None]
            numeric_cols.append(h)
        except (ValueError, TypeError):
            text_cols.append(h)
    numeric_cols = [h for h in numeric_cols
                    if not _is_dimension_col(h, [r.get(h) for r in rows])]

    y_col    = numeric_cols[0]
    y_values = [float(str(r.get(y_col) or 0).replace(",", "")) for r in rows]

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#FAFAF8")
    ax.set_facecolor("#FFFFFF")

    BLUE   = "#378ADD"
    GRAY   = "#888780"
    TEXT   = "#2C2C2A"
    BORDER = "#D3D1C7"

    if chart_type == "bar" and text_cols:
        x_labels = [str(r.get(text_cols[0], ""))[:22] for r in rows]
        bars = ax.bar(range(len(x_labels)), y_values, color=BLUE, width=0.6, zorder=2)
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=30 if len(x_labels) > 6 else 0,
                           ha="right", fontsize=9, color=GRAY)
        for bar, val in zip(bars, y_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(y_values) * 0.01,
                f"{val:,.0f}", ha="center", va="bottom", fontsize=8, color=TEXT,
            )

    elif chart_type == "line":
        x_labels = [str(r.get(text_cols[0], ""))[:22] for r in rows] if text_cols else list(range(len(rows)))
        xs = range(len(rows))
        ax.plot(xs, y_values, color=BLUE, linewidth=2, marker="o", markersize=4, zorder=2)
        ax.fill_between(xs, y_values, alpha=0.08, color=BLUE)
        ax.set_xticks(xs)
        step = max(1, len(rows) // 10)
        ax.set_xticklabels(
            [x_labels[i] if i % step == 0 else "" for i in xs],
            rotation=30, ha="right", fontsize=9, color=GRAY,
        )

    elif chart_type == "scatter" and len(numeric_cols) >= 2:
        x_vals = [float(str(r.get(numeric_cols[0]) or 0).replace(",", "")) for r in rows]
        y_vals = [float(str(r.get(numeric_cols[1]) or 0).replace(",", "")) for r in rows]
        ax.scatter(x_vals, y_vals, color=BLUE, alpha=0.7, s=55, edgecolors="none", zorder=2)
        ax.set_xlabel(numeric_cols[0], fontsize=10, color=GRAY)
        ax.set_ylabel(numeric_cols[1], fontsize=10, color=GRAY)

    # Styling
    ax.set_title(title, fontsize=12, fontweight="bold", pad=14, color=TEXT)
    if chart_type != "scatter":
        ax.set_ylabel(y_col, fontsize=10, color=GRAY)
    ax.tick_params(colors=GRAY, labelsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(BORDER)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, color=BORDER, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def build_chart_payload(rows: list[dict], chart_type: str, title: str = "Results") -> Optional[dict]:
    """Return a frontend-friendly interactive chart payload."""
    if not rows:
        return None

    headers = list(rows[0].keys())
    if len(headers) < 2:
        return None

    numeric_cols, text_cols = [], []
    for h in headers:
        try:
            vals = [float(str(r.get(h) or 0).replace(",", "")) for r in rows if r.get(h) is not None]
            if vals:
                numeric_cols.append(h)
        except (ValueError, TypeError):
            text_cols.append(h)

    # Strip out ID/dimension columns — they parse as numeric but aren't metrics
    numeric_cols = [h for h in numeric_cols
                    if not _is_dimension_col(h, [r.get(h) for r in rows])]

    if not numeric_cols:
        return None

    # If all numeric columns were stripped, fall back to using the first one as x-axis label
    x_key = text_cols[0] if text_cols else headers[0]
    if chart_type == "scatter":
        y_keys = numeric_cols[:2]
    elif chart_type in ("pie", "donut"):
        y_keys = numeric_cols[:1]
    else:
        # bar / line / area — expose all numeric columns for multi-series rendering
        y_keys = numeric_cols

    clean_rows = []
    for r in rows:
        item = {}
        all_keys = [x_key, *y_keys]
        for key in all_keys:
            val = r.get(key)
            if key in y_keys:
                try:
                    item[key] = float(str(val or 0).replace(",", ""))
                except (ValueError, TypeError):
                    item[key] = 0.0
            else:
                item[key] = "" if val is None else str(val)
        clean_rows.append(item)

    return {
        "title": title,
        "chart_type": chart_type,
        "x_key": x_key,
        "y_keys": y_keys,
        "rows": clean_rows,
    }


# ── Zoom file upload ──────────────────────────────────────────────────────────

async def upload_chart_to_zoom(
    chart_bytes: bytes,
    to_jid: str,
    account_id: str,
    token: str,
    filename: str = "chart.png",
) -> None:
    """
    Upload chart PNG to Zoom's file endpoint and send it as a chat file message.
    Zoom bot file upload: POST https://file.zoom.us/v2/im/chat/messages/files
    """
    from config import ZOOM_BOT_JID

    # Write bytes to a temp file (httpx needs a file-like object)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(chart_bytes)
        tmp_path = tmp.name

    try:
        async with httpx.AsyncClient() as client:
            with open(tmp_path, "rb") as f:
                resp = await client.post(
                    "https://file.zoom.us/v2/im/chat/messages/files",
                    headers={"Authorization": f"Bearer {token}"},
                    data={
                        "robot_jid":  ZOOM_BOT_JID,
                        "to_jid":     to_jid,
                        "account_id": account_id,
                    },
                    files={"file": (filename, f, "image/png")},
                    timeout=20,
                )
            resp.raise_for_status()
            log.info("Chart uploaded to Zoom (%d bytes)", len(chart_bytes))
    except Exception as e:
        log.warning("Chart upload failed: %s — skipping chart", e)
    finally:
        os.unlink(tmp_path)
