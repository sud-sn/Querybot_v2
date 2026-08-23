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


class UnsupportedChartTypeError(Exception):
    """
    Raised by _render() for a chart_type with no PNG rendering implemented.

    Distinct from a generic render failure: a caller that catches this
    specifically knows to tell the user the chart type isn't available here
    (e.g. Teams's PNG fallback) rather than silently sending nothing, or --
    the bug this replaces -- silently substituting a plain bar chart that
    renders the right numbers in the wrong, misleading shape.
    """
    def __init__(self, chart_type: str):
        self.chart_type = chart_type
        super().__init__(f"No PNG rendering available for chart type '{chart_type}'")


# Types with no dedicated PNG branch below. heatmap/treemap need genuinely
# 2D/hierarchical layouts (a value grid, nested rectangle packing) that a
# single-axis matplotlib render can't approximate honestly -- unlike
# waterfall/funnel/histogram/boxplot below, which fit a 1D bar-family
# rendering (or, for boxplot, matplotlib's native bxp() summary-stats input).
_NO_PNG_RENDERING = {"heatmap", "treemap"}

_DIM_SUFFIX_RE = re.compile(
    r"(?i)(^|_)(id|key|code|num|no|nbr|nr|ref|pk|fk|seq|idx|index|rank|number)$"
)
_DIM_EXACT = frozenset({"id", "key", "code", "number", "num", "no", "ref", "rank", "index"})
_METRIC_NAME_RE = re.compile(
    r"(?i)(revenue|sales|amount|amt|charge|cost|cogs|price|margin|profit|usd|"
    r"value|balance|total|percent|percentage|pct|rate|ratio|share|count|qty|quantity)"
)


def _json_safe(value):
    """Coerce a database value into something json.dumps accepts.

    Decimal is the one that bites: pyodbc returns it for every NUMERIC column,
    and it survives all the way to the WebSocket send before failing. date and
    datetime are here for the same reason -- a passthrough payload keeps every
    column, so anything the driver returns can reach the wire.
    """
    from datetime import date as _date, datetime as _datetime
    from decimal import Decimal as _Decimal

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, _Decimal):
        return float(value)
    if isinstance(value, (_datetime, _date)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


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

    Returns one of: bar, line, area, scatter, pie, donut, waterfall, heatmap,
    funnel, forecast, histogram, boxplot, treemap, or None.
    """
    # Tier 3 structural signals take priority over spec inference. The marker
    # list lives in chart_spec so this and infer_chart_spec cannot disagree —
    # they used to, and the disagreement is exactly what made these types
    # unreachable: this returned "forecast" while the spec offered only bar.
    from core.chart_spec import structural_chart_type
    _structural = structural_chart_type(rows)
    if _structural:
        return _structural

    spec = infer_chart_spec(rows, question=question, column_formats=column_formats)
    if not spec:
        return None
    recommended = spec.get("recommended_type")
    _all_types = {
        "bar", "line", "area", "scatter", "pie", "donut",
        "waterfall", "heatmap", "funnel", "forecast", "histogram", "boxplot", "treemap",
    }
    if recommended in _all_types:
        return recommended
    return None


def generate_chart(rows: list[dict], chart_type: str, title: str = "Results") -> Optional[bytes]:
    """
    Render a chart as PNG bytes for non-portal adapters.

    Returns None if matplotlib is not installed or rendering fails for an
    unexpected reason. Raises UnsupportedChartTypeError (does NOT return
    None) when chart_type has no PNG rendering at all -- callers must
    handle that case explicitly (tell the user, don't silently substitute
    a different chart shape) rather than have it look identical to "render
    failed, say nothing".
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

    if chart_type in _NO_PNG_RENDERING:
        raise UnsupportedChartTypeError(chart_type)

    try:
        return _render(rows, chart_type, title, plt)
    except UnsupportedChartTypeError:
        raise
    except Exception as e:
        log.error("Chart render error: %s", e)
        return None


def _render_boxplot(rows: list[dict], title: str, plt, colors: dict) -> bytes:
    """Render precomputed 5-number-summary rows (core/distribution_analysis.py's
    bp_min/bp_q1/bp_median/bp_q3/bp_max shape) via matplotlib's native bxp(),
    which takes exactly this stats-dict input -- no approximation needed."""
    labels = [str(r.get("group", i))[:22] for i, r in enumerate(rows)]
    stats = [{
        "label":  labels[i],
        "whislo": _to_float(r.get("bp_min")) or 0.0,
        "q1":     _to_float(r.get("bp_q1")) or 0.0,
        "med":    _to_float(r.get("bp_median")) or 0.0,
        "q3":     _to_float(r.get("bp_q3")) or 0.0,
        "whishi": _to_float(r.get("bp_max")) or 0.0,
        "fliers": [_to_float(v) for v in (r.get("bp_outliers") or [])],
    } for i, r in enumerate(rows)]

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#FFFFFF")
    bp = ax.bxp(stats, showfliers=True, patch_artist=True)
    for box in bp["boxes"]:
        box.set_facecolor(colors["blue"])
        box.set_alpha(0.35)
        box.set_edgecolor(colors["blue"])
    for median in bp["medians"]:
        median.set_color(colors["blue"])
        median.set_linewidth(2)
    ax.set_xticklabels(labels, rotation=30 if len(labels) > 6 else 0, ha="right", fontsize=9, color=colors["gray"])
    ax.set_title(title, fontsize=12, fontweight="bold", pad=14, color=colors["text"])
    ax.tick_params(colors=colors["gray"], labelsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(colors["border"])
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, color=colors["border"], zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _render(rows: list[dict], chart_type: str, title: str, plt) -> bytes:
    blue = "#2563EB"
    gray = "#64748B"
    text = "#0F172A"
    border = "#D8E0EA"

    # Boxplot rows carry a completely different shape (precomputed
    # summary-stats columns, not a plain numeric/text split) -- handle it
    # before the generic column classification below even runs.
    if chart_type == "boxplot" and rows and "bp_data" in rows[0]:
        return _render_boxplot(rows, title, plt, {"blue": blue, "gray": gray, "text": text, "border": border})

    numeric_cols, text_cols = _classify_columns(rows)
    if not numeric_cols:
        raise ValueError("No numeric columns to chart")

    y_col = numeric_cols[0]
    y_values = [_to_float(r.get(y_col)) or 0.0 for r in rows]

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#FFFFFF")

    if chart_type == "scatter" and len(numeric_cols) >= 2:
        x_col = numeric_cols[0]
        y_col = numeric_cols[1]
        x_vals = [_to_float(r.get(x_col)) or 0.0 for r in rows]
        y_vals = [_to_float(r.get(y_col)) or 0.0 for r in rows]
        ax.scatter(x_vals, y_vals, color=blue, alpha=0.76, s=55, edgecolors="none", zorder=2)
        ax.set_xlabel(x_col, fontsize=10, color=gray)
        ax.set_ylabel(y_col, fontsize=10, color=gray)
    elif chart_type == "waterfall":
        # Floating bars from a running cumulative baseline: green for an
        # increase, red for a decrease -- the actual waterfall shape,
        # not a plain bar chart of the same raw values (which throws away
        # the cumulative-total story a waterfall exists to show).
        labels = [str(r.get(text_cols[0], i))[:22] for i, r in enumerate(rows)] if text_cols else [str(i) for i in range(len(rows))]
        cumulative = 0.0
        bottoms, heights, bar_colors = [], [], []
        for v in y_values:
            if v >= 0:
                bottoms.append(cumulative)
                cumulative += v
            else:
                cumulative += v
                bottoms.append(cumulative)
            heights.append(abs(v))
            bar_colors.append("#16A34A" if v >= 0 else "#DC2626")
        ax.bar(range(len(labels)), heights, bottom=bottoms, color=bar_colors, width=0.6, zorder=2)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30 if len(labels) > 6 else 0, ha="right", fontsize=9, color=gray)
        ax.set_ylabel(y_col, fontsize=10, color=gray)
    elif chart_type == "funnel":
        # Horizontal bars, top-of-funnel first, length proportional to
        # stage count -- reads as a narrowing funnel top-to-bottom, unlike
        # the vertical generic-bar substitution this replaces.
        labels = [str(r.get(text_cols[0], i))[:22] for i, r in enumerate(rows)] if text_cols else [str(i) for i in range(len(rows))]
        n = len(rows)
        ax.barh(range(n), y_values, color=blue, height=0.6, zorder=2)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=9, color=gray)
        ax.invert_yaxis()
        ax.set_xlabel(y_col, fontsize=10, color=gray)
    elif chart_type == "histogram":
        # Touching bars (width=1.0, no gaps) is the one thing that visually
        # distinguishes a histogram from a bar chart of the same counts --
        # bars represent continuous bins, not discrete unrelated categories.
        bin_col = "bin_label" if rows and "bin_label" in rows[0] else (text_cols[0] if text_cols else None)
        labels = [str(r.get(bin_col, i)) for i, r in enumerate(rows)] if bin_col else [str(i) for i in range(len(rows))]
        ax.bar(range(len(labels)), y_values, color=blue, width=1.0, edgecolor="#FFFFFF", linewidth=0.5, zorder=2)
        ax.set_xticks(range(len(labels)))
        step = max(1, len(labels) // 10)
        ax.set_xticklabels(
            [labels[i] if i % step == 0 else "" for i in range(len(labels))],
            rotation=30, ha="right", fontsize=9, color=gray,
        )
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


_ANNOTATABLE_TYPES = {"bar", "line", "area"}


def build_chart_annotations(rows: list[dict], question: str = "") -> dict | None:
    """Return governed trend callouts for chart-capable period results.

    The statistical brief is the single source of truth for gain/drop periods.
    Keeping this helper beside ``build_chart_payload`` lets chat, history, and
    refreshed dashboard charts render the same annotations instead of only the
    live WebSocket response carrying them.
    """
    try:
        from core.insight import compute_data_brief

        time_series = (
            compute_data_brief(rows, question).get("time_series") or {}
        )
        annotations = {
            key: time_series.get(key)
            for key in ("biggest_period_drop", "biggest_period_gain")
            if time_series.get(key)
        }
        return annotations or None
    except Exception:
        return None


def build_chart_payload(
    rows: list[dict],
    chart_type: str | None,
    title: str = "Results",
    question: str = "",
    column_formats: dict | None = None,
    annotations: dict | None = None,
) -> Optional[dict]:
    """
    Return a frontend-friendly interactive chart payload.

    annotations, when given, is the time-series brief's already-computed
    biggest_period_drop/biggest_period_gain (core/insight.py's
    _compute_time_series_brief shape: {"to_period", "absolute_change",
    "pct_change", ...}) -- reused as-is, never recomputed here. Only
    attached for bar/line/area (the chart types with a genuine per-point
    x-axis position to anchor an annotation to) and only when the
    referenced period actually appears in the rendered rows, so a stale
    or mismatched annotation never gets sent to the frontend to draw.
    """
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
    _renderable = {
        "bar", "line", "area", "scatter", "pie", "donut",
        "waterfall", "heatmap", "funnel", "forecast", "histogram", "boxplot", "treemap",
    }
    if effective_type not in _renderable:
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

    # For structural chart types, keep every column — the marker columns
    # (is_forecast, bp_data, funnel_pct) are what the renderer draws from, and
    # projecting down to [x, *y] would strip them.
    #
    # "As-is" still has to mean JSON-serialisable. The projection branch below
    # coerces its values on the way past, so nothing noticed that a database
    # Decimal reaches the payload untouched here — until forecast became the
    # first passthrough type that actually renders, and every answer died on
    # "Object of type Decimal is not JSON serializable" AFTER the SQL had run
    # and the forecast had been computed.
    _passthrough_types = {"funnel", "histogram", "boxplot", "forecast"}
    if effective_type in _passthrough_types:
        clean_rows = [
            {key: _json_safe(value) for key, value in row.items()} for row in rows
        ]
    else:
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

    # Hoist the forecast's fit metadata off row 0 and onto the payload, where
    # it describes the whole series rather than pretending to be a cell of the
    # first one. The row-0 keys stay for one release so an un-refreshed browser
    # tab keeps captioning the chart.
    forecast_meta = None
    for row in clean_rows:
        if isinstance(row.get("__forecast_meta"), dict):
            forecast_meta = row.pop("__forecast_meta")
            break
    for row in clean_rows:
        row.pop("__forecast_meta", None)

    payload = {
        "title": title,
        "chart_type": effective_type,
        "forecast_meta": forecast_meta,
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

    if annotations and effective_type in _ANNOTATABLE_TYPES:
        known_periods = {str(r.get(x_key, "")) for r in clean_rows}
        chart_annotations = {}
        for kind in ("biggest_period_drop", "biggest_period_gain"):
            entry = annotations.get(kind)
            if not entry:
                continue
            period = str(entry.get("to_period", ""))
            if period not in known_periods:
                continue
            chart_annotations[kind] = {
                "period": period,
                "absolute_change": entry.get("absolute_change"),
                "pct_change": entry.get("pct_change"),
            }
        if chart_annotations:
            payload["annotations"] = chart_annotations

    return payload


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
