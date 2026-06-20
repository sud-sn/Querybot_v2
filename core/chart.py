"""
Chart helpers.

The portal renders interactive charts with ECharts. Teams/Zoom style adapters
still use matplotlib PNG rendering. Both paths use the same lightweight chart
payload produced here.
"""

from __future__ import annotations

import io
import logging
import os
import re
import tempfile
from typing import Optional

import httpx

from core.chart_spec import infer_chart_spec

log = logging.getLogger("querybot.chart")

_DIM_SUFFIX_RE = re.compile(
    r"(?i)(^|_)(id|key|code|num|no|nbr|nr|ref|pk|fk|seq|idx|index|rank|number)$"
)
_DIM_EXACT = frozenset({"id", "key", "code", "number", "num", "no", "ref", "rank", "index"})
_METRIC_NAME_RE = re.compile(
    r"(?i)(revenue|sales|amount|amt|charge|cost|cogs|price|margin|profit|usd|"
    r"value|balance|total|percent|percentage|pct|rate|ratio|share|count|qty|quantity)"
)


def _to_float(value) -> float | None:
    try:
        raw = str(value).strip().replace("$", "").replace(",", "").replace("%", "")
        if not raw:
            return None
        n = float(raw)
        return None if n != n else n
    except (TypeError, ValueError):
        return None


def _is_dimension_col(col_name: str, values: list) -> bool:
    """Return True when a numeric-looking column is really an ID/dimension."""
    if _METRIC_NAME_RE.search(col_name or ""):
        return False
    if _DIM_SUFFIX_RE.search(col_name or ""):
        return True
    if (col_name or "").lower() in _DIM_EXACT:
        return True
    numeric = [_to_float(v) for v in values if v is not None]
    numeric = [v for v in numeric if v is not None]
    non_null_count = len([v for v in values if v is not None])
    if len(numeric) < 2 or len(numeric) != non_null_count:
        return False
    all_int = all(float(v).is_integer() for v in numeric)
    if not all_int:
        return False
    return len(set(int(v) for v in numeric)) / max(len(numeric), 1) > 0.8


def _classify_columns(rows: list[dict]) -> tuple[list[str], list[str]]:
    numeric_cols, text_cols = [], []
    if not rows:
        return numeric_cols, text_cols
    for h in rows[0].keys():
        vals = [_to_float(r.get(h)) for r in rows if r.get(h) is not None]
        vals = [v for v in vals if v is not None]
        if vals:
            numeric_cols.append(h)
        else:
            text_cols.append(h)
    numeric_cols = [
        h for h in numeric_cols
        if not _is_dimension_col(h, [r.get(h) for r in rows])
    ]
    return numeric_cols, text_cols


def detect_chart_type(
    rows: list[dict],
    question: str = "",
    column_formats: dict | None = None,
) -> Optional[str]:
    """
    Inspect result rows and choose the safest chart type.

    Returns one of: bar, line, area, scatter, pie, donut, waterfall, heatmap, or None.
    """
    spec = infer_chart_spec(rows, question=question, column_formats=column_formats)
    if not spec:
        return None
    recommended = spec.get("recommended_type")
    if recommended in {"bar", "line", "area", "scatter", "pie", "donut", "waterfall", "heatmap"}:
        return recommended
    return None


def generate_chart(rows: list[dict], chart_type: str, title: str = "Results") -> Optional[bytes]:
    """
    Render a chart as PNG bytes for non-portal adapters.

    Returns None if matplotlib is not installed or rendering fails.
    """
    if not rows:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed - chart skipped")
        return None

    try:
        return _render(rows, chart_type, title, plt)
    except Exception as e:
        log.error("Chart render error: %s", e)
        return None


def _render(rows: list[dict], chart_type: str, title: str, plt) -> bytes:
    numeric_cols, text_cols = _classify_columns(rows)
    if not numeric_cols:
        raise ValueError("No numeric columns to chart")

    y_col = numeric_cols[0]
    y_values = [_to_float(r.get(y_col)) or 0.0 for r in rows]

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#FFFFFF")

    blue = "#2563EB"
    gray = "#64748B"
    text = "#0F172A"
    border = "#D8E0EA"

    if chart_type == "scatter" and len(numeric_cols) >= 2:
        x_col = numeric_cols[0]
        y_col = numeric_cols[1]
        x_vals = [_to_float(r.get(x_col)) or 0.0 for r in rows]
        y_vals = [_to_float(r.get(y_col)) or 0.0 for r in rows]
        ax.scatter(x_vals, y_vals, color=blue, alpha=0.76, s=55, edgecolors="none", zorder=2)
        ax.set_xlabel(x_col, fontsize=10, color=gray)
        ax.set_ylabel(y_col, fontsize=10, color=gray)
    elif chart_type in {"line", "area"}:
        labels = [str(r.get(text_cols[0], i))[:22] for i, r in enumerate(rows)] if text_cols else list(range(len(rows)))
        xs = range(len(rows))
        ax.plot(xs, y_values, color=blue, linewidth=2.4, marker="o", markersize=4, zorder=2)
        if chart_type == "area":
            ax.fill_between(xs, y_values, alpha=0.12, color=blue)
        ax.set_xticks(list(xs))
        step = max(1, len(rows) // 10)
        ax.set_xticklabels(
            [labels[i] if i % step == 0 else "" for i in xs],
            rotation=30,
            ha="right",
            fontsize=9,
            color=gray,
        )
        ax.set_ylabel(y_col, fontsize=10, color=gray)
    else:
        labels = [str(r.get(text_cols[0], i))[:22] for i, r in enumerate(rows)] if text_cols else [str(i) for i in range(len(rows))]
        bars = ax.bar(range(len(labels)), y_values, color=blue, width=0.6, zorder=2)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30 if len(labels) > 6 else 0, ha="right", fontsize=9, color=gray)
        if y_values:
            top = max(y_values)
            for bar, val in zip(bars, y_values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + top * 0.01,
                    f"{val:,.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=text,
                )
        ax.set_ylabel(y_col, fontsize=10, color=gray)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=14, color=text)
    ax.tick_params(colors=gray, labelsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(border)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, color=border, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def build_chart_payload(
    rows: list[dict],
    chart_type: str | None,
    title: str = "Results",
    question: str = "",
    column_formats: dict | None = None,
) -> Optional[dict]:
    """Return a frontend-friendly interactive chart payload."""
    if not rows:
        return None
    headers = list(rows[0].keys())
    if len(headers) < 2:
        return None

    spec = infer_chart_spec(
        rows,
        question=question or title,
        column_formats=column_formats,
        title=title,
    )
    if not spec:
        return None

    roles = spec.get("column_roles") or {}
    numeric_cols = [
        col for col in headers
        if roles.get(col, {}).get("role") == "measure"
    ]
    text_cols = [
        col for col in headers
        if roles.get(col, {}).get("role") in {"dimension", "identifier", "temporal"}
    ]
    if not numeric_cols:
        return None

    allowed = set(spec.get("renderable_types") or [])
    requested = (chart_type or "").lower().strip()
    effective_type = requested if requested in allowed else spec.get("recommended_type")
    if effective_type not in {"bar", "line", "area", "scatter", "pie", "donut", "waterfall", "heatmap"}:
        return None

    x_spec = spec.get("x") or {}
    x_key = x_spec.get("column") or (text_cols[0] if text_cols else headers[0])
    spec_y = [c.get("column") for c in (spec.get("y") or []) if c.get("column")]
    if effective_type == "scatter":
        y_keys = (spec_y or numeric_cols)[:2]
    elif effective_type in {"pie", "donut"}:
        y_keys = (spec_y or numeric_cols)[:1]
    else:
        y_keys = spec_y or numeric_cols

    clean_rows = []
    for r in rows:
        item = {}
        for key in [x_key, *y_keys]:
            val = r.get(key)
            if key in y_keys:
                item[key] = _to_float(val)
            else:
                item[key] = "" if val is None else str(val)
        clean_rows.append(item)

    return {
        "title": title,
        "chart_type": effective_type,
        "requested_chart_type": requested or None,
        "x_key": x_key,
        "y_keys": y_keys,
        "rows": clean_rows,
        "chart_spec": spec,
        "intent": spec.get("intent"),
        "recommended_type": spec.get("recommended_type"),
        "allowed_types": spec.get("allowed_types") or [],
        "renderable_types": spec.get("renderable_types") or [],
        "column_roles": spec.get("column_roles") or {},
        "chart_warnings": spec.get("warnings") or [],
        "chart_confidence": spec.get("confidence"),
        "column_formats": column_formats or {},
    }


async def upload_chart_to_zoom(
    chart_bytes: bytes,
    to_jid: str,
    account_id: str,
    token: str,
    filename: str = "chart.png",
) -> None:
    """
    Upload chart PNG to Zoom's file endpoint and send it as a chat file message.
    """
    from config import ZOOM_BOT_JID

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
                        "robot_jid": ZOOM_BOT_JID,
                        "to_jid": to_jid,
                        "account_id": account_id,
                    },
                    files={"file": (filename, f, "image/png")},
                    timeout=20,
                )
            resp.raise_for_status()
            log.info("Chart uploaded to Zoom (%d bytes)", len(chart_bytes))
    except Exception as e:
        log.warning("Chart upload failed: %s - skipping chart", e)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
