"""
core/dashboard_snapshot.py

A dashboard as one reader sees it, drawn for an email: its numbers and tables as text, a picture of each chart, and
a PDF of its tabs in the dashboard's own design.

Each tab is drawn by the dashboard page itself (portal/routes.py ``_render_dashboard``) for that reader -- their
tables, row rules, masking and language -- in snapshot mode: the page without its sidebar, menus or controls, with
when and for whom it was drawn under its name. The numbers and tables are read from what the page drew. The chart
pictures and the PDF come from the same page opened in a headless Chromium that reaches nothing: the page is
served from memory, /static from disk, and every other request is refused.

    chart pictures   the page at email width, where the grid folds to one column: each chart drawn as wide as it is
                     shown in the email
    the PDF          the page at desktop width, one page per tab as tall as the tab, merged

Chromium is Playwright's: ``pip install playwright`` then ``python -m playwright install --with-deps chromium``, or
QUERYBOT_CHROMIUM_PATH naming a Chromium already on the server. Without it the email still goes, with every
number and table, and says why the charts are not in it.
"""

from __future__ import annotations

import datetime as dt
import logging
import mimetypes
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlencode, urlparse

log = logging.getLogger("querybot.dashboard_snapshot")

HOST = "querybot.snapshot"
STATIC = Path(__file__).resolve().parent.parent / "static"
EMAIL_WIDTH = 640            # the page width chart pictures are drawn at (the grid is one column under 768)
PDF_WIDTH = 1280             # the page width the PDF is drawn at
TABLE_ROWS = 15              # rows of a table tile written into an email
WAIT_MS = 20000              # how long a page may take to draw its charts


@dataclass
class Tile:
    id: int
    title: str
    kind: str                                 # number | chart | table | changed | error | empty
    note: str = ""
    value: str = ""                           # number
    label: str = ""
    change: Optional[dict] = None
    columns: list[str] = field(default_factory=list)      # table
    numeric: list[bool] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    total: int = 0
    image: Optional[bytes] = None             # chart, a PNG drawn at twice its size
    width: int = 0                            # its size on the page, in CSS pixels
    height: int = 0
    error: str = ""
    changed: Optional[dict] = None            # "What changed": {"summary", "figures"} (core/what_changed.py)
    reading: str = ""                         # the analyst's line under the tile (core2/answer/facts.py)


@dataclass
class Tab:
    name: str
    tiles: list[Tile] = field(default_factory=list)


@dataclass
class Snapshot:
    dashboard_id: int
    name: str
    drawn: str                                # "Drawn 12 Oct 2026, 08:00 UTC for Riley Reader"
    filters: str = ""                         # "Period: Last quarter · Region: West"
    reading: list[dict] = field(default_factory=list)      # "What this dashboard shows": {"title", "line"}
    tabs: list[Tab] = field(default_factory=list)
    pdf: Optional[bytes] = None
    problems: list[str] = field(default_factory=list)       # message ids (snapshot.*)

    def charts(self) -> list[Tile]:
        return [tile for tab in self.tabs for tile in tab.tiles if tile.kind == "chart"]


# ── The page, drawn for one reader ───────────────────────────────────────────

def _request(dashboard_id: int, tab: str, view: dict, lang: str):
    from starlette.requests import Request

    params = [("dashboard_id", str(int(dashboard_id))), ("tab", tab)]
    if view.get("period"):
        params.append(("period", str(view["period"])))
    for name, value in (view.get("members") or {}).items():
        params.append((f"f.{name}", str(value)))
    return Request({
        "type": "http", "http_version": "1.1", "method": "GET", "scheme": "http",
        "server": (HOST, 80), "client": ("127.0.0.1", 0), "root_path": "",
        "path": "/portal/dashboard", "raw_path": b"/portal/dashboard", "query_string": urlencode(params).encode(),
        "headers": [(b"host", HOST.encode()), (b"cookie", f"qb_lang={lang}".encode())],
    })


def local_time(when: dt.datetime, zone: str, lang: str) -> str:
    """A UTC moment as "12 Oct 2026, 08:00 BST" in ``zone``, the month named in the reader's language."""
    from zoneinfo import ZoneInfo

    from core.i18n import format_date

    try:
        local = when.replace(tzinfo=dt.timezone.utc).astimezone(ZoneInfo(zone or "UTC"))
    except Exception:  # noqa: BLE001 - a zone this server does not know: UTC
        local = when.replace(tzinfo=dt.timezone.utc)
    return f"{format_date(local, 'day_month_short_year', lang=lang)}, {local:%H:%M} {local.tzname() or ''}".strip()


def drawn_line(user: dict, when: dt.datetime, zone: str, lang: str) -> str:
    """"Drawn 12 Oct 2026, 08:00 BST for Riley Reader", in the reader's language and the schedule's time zone."""
    from core.i18n import t

    return t("snapshot.drawn_for", lang=lang, at=local_time(when, zone, lang),
             name=user.get("name") or user.get("email") or "")


async def _tab_page(dashboard_id: int, user: dict, tab: str, view: dict, lang: str, drawn: str) -> tuple[str, dict]:
    """The page of one tab as ``user`` sees it, and what it drew (its template context)."""
    from core.i18n import activate_language, deactivate_language
    from portal.routes import _render_dashboard

    token = activate_language(lang)
    try:
        response = await _render_dashboard(_request(dashboard_id, tab, view, lang), user, snapshot={"drawn": drawn})
    finally:
        deactivate_language(token)
    return response.body.decode("utf-8"), dict(getattr(response, "context", {}) or {})


def _tiles(charts: list[dict]) -> list[Tile]:
    """The tiles a page drew, in the order they are laid out (row, then column)."""
    ordered = sorted(charts, key=lambda c: (int(c.get("grid_y") or 0), int(c.get("grid_x") or 0), int(c.get("id") or 0)))
    tiles = []
    for chart in ordered:
        tile = Tile(id=int(chart.get("id") or 0), title=str(chart.get("title") or ""), kind="empty")
        tile.note = " · ".join(n for n in (str(chart.get("top_note") or ""), str(chart.get("subtitle") or "")) if n)
        tile.reading = str(chart.get("note_line") or "")
        if chart.get("error"):
            tile.kind, tile.error = "error", str(chart["error"])
        elif chart.get("what_changed"):
            changed = chart["what_changed"] or {}
            tile.kind = "empty" if changed.get("empty") else "changed"
            tile.changed = changed
        elif chart.get("kpi"):
            kpi = chart["kpi"] or {}
            tile.kind = "number"
            tile.value = str(chart.get("kpi_display") or kpi.get("value") or "")
            label = str(kpi.get("label") or "")
            tile.label = label if label and label.lower() != tile.title.lower() else ""
            change = kpi.get("change")
            tile.change = change if isinstance(change, dict) and change.get("direction") else None
        elif chart.get("table_rows"):
            columns = list(chart.get("table_columns") or [])
            labels = chart.get("table_column_labels") or {}
            numeric = set(chart.get("table_numeric") or [])
            tile.kind = "table"
            tile.columns = [str(labels.get(c, c)) for c in columns]
            tile.numeric = [c in numeric for c in columns]
            tile.rows = [[str((row.get(c) or {}).get("d", "")) if isinstance(row.get(c), dict) else str(row.get(c) or "")
                          for c in columns] for row in chart["table_rows"]]
            tile.total = int(chart.get("row_count") or len(tile.rows))
        elif chart.get("chart_json"):
            tile.kind = "chart"
        tiles.append(tile)
    return tiles


def _filters_line(context: dict, lang: str) -> str:
    from core.i18n import t

    view = context.get("dash_view") or {}
    parts = []
    if view.get("period"):
        parts.append(f"{t('ui.dash.period', lang=lang)}: {t('ui.dash.period.' + view['period'], lang=lang)}")
    for grouping in context.get("dash_groupings") or []:
        if grouping.get("selected"):
            parts.append(f"{grouping.get('label')}: {grouping['selected']}")
    return " · ".join(parts)


# ── The headless browser ─────────────────────────────────────────────────────

class Drawer:
    """One headless Chromium for a run: ``async with Drawer() as drawer``. ``drawer.problem`` is why it could not
    start (no Playwright, no Chromium), as a message id; the snapshots are then taken without pictures or a PDF."""

    def __init__(self):
        self.problem = ""
        self._playwright = None
        self._browser = None

    async def __aenter__(self) -> "Drawer":
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            self.problem = "snapshot.no_playwright"
            return self
        try:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                executable_path=os.getenv("QUERYBOT_CHROMIUM_PATH") or None)
        except Exception as exc:  # noqa: BLE001 - no Chromium: the email goes without its pictures
            self.problem = "snapshot.no_chromium"
            log.warning("Headless Chromium could not start: %s", exc)
            await self._stop()
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stop()

    async def _stop(self) -> None:
        for closing in (self._browser, self._playwright):
            if closing is None:
                continue
            try:
                await (closing.close() if closing is self._browser else closing.stop())
            except Exception:  # noqa: BLE001
                pass
        self._browser = self._playwright = None

    @property
    def ready(self) -> bool:
        return self._browser is not None

    async def _open(self, pages: dict[str, str], width: int, zone: str, lang: str):
        """A page on its own context at ``width``, served ``pages`` (by their ?page= key) and /static only. The
        page's clock is the schedule's time zone and its locale the reader's, so what the page writes for itself
        (a time, a number) reads as the email around it."""
        context = await self._browser.new_context(
            viewport={"width": width, "height": 900}, device_scale_factor=2, reduced_motion="reduce",
            timezone_id=zone if _known_zone(zone) else "UTC", locale={"fr": "fr-FR"}.get(lang, "en-GB"))

        async def serve(route):
            url = urlparse(route.request.url)
            if url.hostname != HOST:
                return await route.abort()
            if url.path == "/portal/dashboard":
                key = (parse_qs(url.query).get("page") or [""])[0]
                if key in pages:
                    return await route.fulfill(status=200, content_type="text/html; charset=utf-8", body=pages[key])
            if url.path.startswith("/static/"):
                path = (STATIC / url.path[len("/static/"):]).resolve()
                if path.is_file() and STATIC.resolve() in path.parents:
                    kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                    return await route.fulfill(status=200, content_type=kind, body=path.read_bytes())
            return await route.abort()

        await context.route("**/*", serve)
        return context, await context.new_page()

    async def _drawn(self, page, key: str) -> None:
        await page.goto(f"http://{HOST}/portal/dashboard?page={key}", wait_until="load", timeout=WAIT_MS)
        await page.wait_for_function(
            "() => [...document.querySelectorAll('[data-chart]')].every(n => !n.querySelector('[data-chart-skeleton]'))",
            timeout=WAIT_MS)
        await page.evaluate("() => document.fonts ? document.fonts.ready.then(() => true) : true")
        await page.wait_for_timeout(250)

    async def pictures(self, html: str, chart_ids: list[int], *, zone: str = "UTC",
                       lang: str = "en") -> dict[int, tuple[bytes, int, int]]:
        """Each chart of a tab drawn at email width: {tile id: (PNG, width, height)}. A chart that could not be
        drawn is left out."""
        if not chart_ids:
            return {}
        context, page = await self._open({"email": html}, EMAIL_WIDTH, zone, lang)
        found = {}
        try:
            await self._drawn(page, "email")
            for chart_id in chart_ids:
                node = page.locator(f"#dashboard-chart-{int(chart_id)}")
                try:
                    box = await node.bounding_box()
                    if not box or box["width"] < 10 or box["height"] < 10:
                        continue
                    found[chart_id] = (await node.screenshot(type="png", animations="disabled"),
                                       int(round(box["width"])), int(round(box["height"])))
                except Exception as exc:  # noqa: BLE001 - one chart missing never stops the others
                    log.warning("Chart %s could not be pictured: %s", chart_id, exc)
        finally:
            await context.close()
        return found

    async def pdf(self, htmls: list[str], *, zone: str = "UTC", lang: str = "en") -> Optional[bytes]:
        """The tabs at desktop width, one PDF page each, as tall as the tab, merged."""
        pages = {str(i): html for i, html in enumerate(htmls)}
        context, page = await self._open(pages, PDF_WIDTH, zone, lang)
        parts = []
        try:
            await page.emulate_media(media="screen")
            for key in pages:
                await self._drawn(page, key)
                height = int(await page.evaluate("() => Math.ceil(document.documentElement.scrollHeight)"))
                parts.append(await page.pdf(width=f"{PDF_WIDTH}px", height=f"{max(height, 600) + 2}px",
                                            print_background=True, margin={"top": "0", "bottom": "0",
                                                                           "left": "0", "right": "0"}))
        finally:
            await context.close()
        return merge_pdfs(parts)


def _known_zone(zone: str) -> bool:
    from zoneinfo import ZoneInfo

    try:
        ZoneInfo(str(zone or ""))
        return bool(zone)
    except Exception:  # noqa: BLE001
        return False


def merge_pdfs(parts: list[bytes]) -> Optional[bytes]:
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    import io

    from pypdf import PdfWriter

    writer = PdfWriter()
    for part in parts:
        writer.append(io.BytesIO(part))
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


# ── One snapshot ─────────────────────────────────────────────────────────────

async def take(dashboard: dict, user: dict, *, drawer: Optional[Drawer], tabs: Optional[list[str]] = None,
               view: Optional[dict] = None, zone: str = "UTC", pdf: bool = True,
               now: Optional[dt.datetime] = None) -> Snapshot:
    """``dashboard`` drawn for ``user``: the tabs named (every tab when none of them is on it), under ``view``
    (period and member filters). ``drawer`` None, or not ready: numbers and tables only."""
    import json

    lang = str(user.get("lang") or "en")
    now = now or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    drawn = drawn_line(user, now, zone, lang)
    try:
        own = [str(t) for t in json.loads(dashboard.get("tabs_json") or '["Overview"]') if str(t).strip()]
    except (TypeError, ValueError):
        own = []
    own = own or ["Overview"]
    chosen = [t for t in own if t in (tabs or [])] or own
    snapshot = Snapshot(dashboard_id=int(dashboard["id"]), name=str(dashboard.get("name") or ""), drawn=drawn)
    pages = []
    for tab in chosen:
        html, context = await _tab_page(int(dashboard["id"]), user, tab, view or {}, lang, drawn)
        snapshot.tabs.append(Tab(name=tab, tiles=_tiles(list(context.get("charts") or []))))
        snapshot.filters = snapshot.filters or _filters_line(context, lang)
        snapshot.reading = snapshot.reading or [r for r in context.get("dashboard_reading") or [] if r.get("line")]
        pages.append(html)
    if drawer is None or not drawer.ready:
        if drawer is not None and drawer.problem:
            snapshot.problems.append(drawer.problem)
        return snapshot
    for tab, html in zip(snapshot.tabs, pages):
        charts = [tile for tile in tab.tiles if tile.kind == "chart"]
        try:
            pictured = await drawer.pictures(html, [tile.id for tile in charts], zone=zone, lang=lang)
        except Exception as exc:  # noqa: BLE001 - the email goes with its numbers
            log.warning("Dashboard %s tab %s: charts not pictured: %s", dashboard.get("id"), tab.name, exc)
            pictured = {}
        for tile in charts:
            if tile.id in pictured:
                tile.image, tile.width, tile.height = pictured[tile.id]
        if len(pictured) < len(charts):
            snapshot.problems.append("snapshot.some_charts")
    if pdf:
        try:
            snapshot.pdf = await drawer.pdf(pages, zone=zone, lang=lang)
        except Exception as exc:  # noqa: BLE001
            log.warning("Dashboard %s: PDF not made: %s", dashboard.get("id"), exc)
            snapshot.problems.append("snapshot.no_pdf")
    return snapshot
