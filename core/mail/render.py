"""
core/mail/render.py

The look of every email the product sends: the QueryBot name, a title, the body, a button, and a footer that
says why it came and how to stop it. Inline styles and a table layout, as mail clients need; a plain-text part
goes with it for clients that show no HTML.
"""

from __future__ import annotations

import html
from typing import Optional

INK, MUTED, PRIMARY, LINE, PAPER = "#1b2433", "#5b6676", "#1f4fd1", "#e1e6ee", "#f3f5f9"


def esc(text) -> str:
    return html.escape(str(text if text is not None else ""), quote=True)


def page(title: str, paragraphs: list[str], *, button: Optional[tuple[str, str]] = None,
         rows: Optional[list[tuple[str, str]]] = None, why: str = "", unsubscribe: str = "",
         lang: str = "en") -> str:
    """An email's HTML. ``paragraphs`` are plain text (escaped here); ``rows`` a small (label, value) table;
    ``button`` (label, url); ``why`` why the reader got it; ``unsubscribe`` the link to stop it."""
    body = "".join(f'<p style="margin:0 0 14px;font-size:15px;line-height:1.55;color:{INK}">{esc(p)}</p>'
                   for p in paragraphs if p)
    if rows:
        cells = "".join(
            f'<tr><td style="padding:8px 12px;border-top:1px solid {LINE};color:{MUTED};font-size:14px">{esc(k)}</td>'
            f'<td style="padding:8px 12px;border-top:1px solid {LINE};color:{INK};font-size:15px;font-weight:600;'
            f'text-align:right">{esc(v)}</td></tr>' for k, v in rows)
        body += (f'<table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="border-collapse:'
                 f'collapse;border-bottom:1px solid {LINE};margin:4px 0 18px">{cells}</table>')
    if button:
        label, url = button
        body += (f'<p style="margin:8px 0 6px"><a href="{esc(url)}" style="display:inline-block;background:{PRIMARY};'
                 f'color:#ffffff;text-decoration:none;font-weight:600;font-size:15px;padding:11px 20px;'
                 f'border-radius:8px">{esc(label)}</a></p>')
    stop = {"en": "Stop these emails", "fr": "Ne plus recevoir ces courriels"}.get(lang, "Stop these emails")
    foot = esc(why) + (f' · <a href="{esc(unsubscribe)}" style="color:{MUTED}">{stop}</a>' if unsubscribe else "")
    return (
        f'<!doctype html><html lang="{esc(lang)}"><head><meta charset="utf-8"><meta name="viewport" '
        f'content="width=device-width"><title>{esc(title)}</title></head>'
        f'<body style="margin:0;padding:0;background:{PAPER};font-family:-apple-system,Segoe UI,Roboto,Helvetica,'
        f'Arial,sans-serif">'
        f'<table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="background:{PAPER}">'
        f'<tr><td align="center" style="padding:28px 12px">'
        f'<table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="max-width:560px">'
        f'<tr><td style="padding:0 4px 14px;font-size:17px;font-weight:700;color:{INK}">Query'
        f'<span style="color:{PRIMARY}">Bot</span></td></tr>'
        f'<tr><td style="background:#ffffff;border:1px solid {LINE};border-radius:12px;padding:26px 28px">'
        f'<h1 style="margin:0 0 14px;font-size:20px;line-height:1.3;color:{INK}">{esc(title)}</h1>{body}</td></tr>'
        f'<tr><td style="padding:14px 6px;font-size:12px;line-height:1.5;color:{MUTED}">{foot}</td></tr>'
        f'</table></td></tr></table></body></html>')


def text(title: str, paragraphs: list[str], *, button: Optional[tuple[str, str]] = None,
         rows: Optional[list[tuple[str, str]]] = None, why: str = "", unsubscribe: str = "",
         lang: str = "en") -> str:
    lines = [title, "=" * min(len(title), 60), ""]
    lines += [p for p in paragraphs if p]
    if rows:
        lines += [""] + [f"{k}: {v}" for k, v in rows]
    if button:
        lines += ["", f"{button[0]}: {button[1]}"]
    if why or unsubscribe:
        stop = {"en": "Stop these emails", "fr": "Ne plus recevoir ces courriels"}.get(lang, "Stop these emails")
        lines += ["", "--", why] + ([f"{stop}: {unsubscribe}"] if unsubscribe else [])
    return "\n".join(lines).strip() + "\n"


# ── A dashboard in the email itself ──────────────────────────────────────────
# A scheduled dashboard arrives as the dashboard: its numbers and tables as text the mail program shows (and a
# screen reader reads), its charts as pictures carried in the email (mail programs run no chart scripts), the
# PDF attached. Drawn for each reader by core/dashboard_snapshot.py.

GOOD, BAD = "#157f3b", "#b42318"
WIDTH = 560                      # the content width of the email, in pixels


def tile_cid(tile) -> str:
    return f"tile-{int(tile.id)}@querybot"


def _change(change: dict, lang: str) -> tuple[str, str]:
    """A number's change line as the dashboard writes it: (text, colour)."""
    from core.i18n import t

    direction = str(change.get("direction") or "")
    moved = str((change.get("change") if direction == "flat" else (change.get("pct") or change.get("change"))) or "")
    moved = moved.replace("+", "").replace("-", "")
    arrow = {"up": "▲", "down": "▼"}.get(direction, "—")
    good = change.get("good") if "good" in change else direction == "up"
    colour = MUTED if direction == "flat" or good is None else (GOOD if good else BAD)
    vs = t("ui.kpi.vs", lang=lang, period=change.get("vs") or "") if change.get("vs") else ""
    return f"{arrow} {moved} {vs}".strip(), colour


def _numbers_html(tiles: list, lang: str) -> str:
    """Number tiles side by side, two to a row."""
    rows = []
    for start in range(0, len(tiles), 2):
        cells = []
        for tile in tiles[start:start + 2]:
            change = ""
            if tile.change:
                words, colour = _change(tile.change, lang)
                change = f'<div style="margin-top:4px;font-size:13px;color:{colour};font-weight:600">{esc(words)}</div>'
            label = f'<div style="font-size:12px;color:{MUTED}">{esc(tile.label)}</div>' if tile.label else ""
            cells.append(
                f'<td width="50%" valign="top" style="padding:6px"><div style="border:1px solid {LINE};'
                f'border-radius:10px;padding:14px 16px;background:#ffffff">'
                f'<div style="font-size:13px;color:{MUTED};font-weight:600">{esc(tile.title)}</div>{label}'
                f'<div style="margin-top:6px;font-size:26px;line-height:1.15;font-weight:700;color:{INK}">'
                f'{esc(tile.value)}</div>{change}</div></td>')
        if len(cells) == 1:
            cells.append('<td width="50%" style="padding:6px"></td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (f'<table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="margin:0 -6px 8px;'
            f'border-collapse:separate">{"".join(rows)}</table>')


def _heading(tile) -> str:
    note = f'<span style="font-weight:400;color:{MUTED};font-size:13px"> · {esc(tile.note)}</span>' if tile.note else ""
    return f'<div style="margin:18px 0 8px;font-size:15px;font-weight:700;color:{INK}">{esc(tile.title)}{note}</div>'


def _chart_html(tile, link: str, lang: str) -> str:
    from core.i18n import t

    if not tile.image:
        return (_heading(tile) + f'<p style="margin:0 0 8px;font-size:14px;color:{MUTED}">'
                f'{esc(t("email.schedule.chart_on_dashboard", lang=lang))} '
                f'<a href="{esc(link)}" style="color:{PRIMARY}">{esc(t("email.open_dashboard", lang=lang))}</a></p>')
    width = min(int(tile.width or WIDTH), WIDTH)
    height = int(round((tile.height or 0) * width / max(int(tile.width or width), 1))) if tile.height else 0
    size = f' width="{width}"' + (f' height="{height}"' if height else "")
    return (_heading(tile) +
            f'<img src="cid:{esc(tile_cid(tile))}" alt="{esc(tile.title)}"{size} style="display:block;width:100%;'
            f'max-width:{width}px;height:auto;border:1px solid {LINE};border-radius:10px">')


def _table_html(tile, lang: str) -> str:
    from core.i18n import t

    from core.dashboard_snapshot import TABLE_ROWS

    shown = tile.rows[:TABLE_ROWS]
    head = "".join(
        f'<th align="{"right" if num else "left"}" style="padding:7px 10px;border-bottom:1px solid {LINE};'
        f'font-size:12px;color:{MUTED};font-weight:600;text-transform:uppercase;letter-spacing:.03em">{esc(c)}</th>'
        for c, num in zip(tile.columns, tile.numeric))
    body = "".join(
        "<tr>" + "".join(
            f'<td align="{"right" if num else "left"}" style="padding:7px 10px;border-bottom:1px solid {LINE};'
            f'font-size:14px;color:{INK}{";white-space:nowrap" if num else ""}">{esc(v)}</td>'
            for v, num in zip(row, tile.numeric)) + "</tr>"
        for row in shown)
    total = max(int(tile.total or 0), len(tile.rows))
    more = ""
    if total > len(shown):
        more = (f'<p style="margin:6px 0 0;font-size:13px;color:{MUTED}">'
                f'{esc(t("email.schedule.more_rows", lang=lang, shown=len(shown), total=total))}</p>')
    return (_heading(tile) + f'<table role="presentation" cellpadding="0" cellspacing="0" width="100%" '
            f'style="border-collapse:collapse;border:1px solid {LINE};border-radius:10px">'
            f'<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>{more}')


def dashboard(snapshot, *, link: str, title: str, note: str = "", note_from: str = "", pdf_attached: bool = False,
              test: bool = False, why: str = "", unsubscribe: str = "", lang: str = "en") -> str:
    """The HTML of a scheduled dashboard email for one reader (``snapshot``: core/dashboard_snapshot.Snapshot)."""
    from core.i18n import t

    parts = []
    if test:
        parts.append(f'<p style="margin:0 0 14px;padding:10px 12px;border-radius:8px;background:#fff7e6;'
                     f'color:#7a4b00;font-size:13px">{esc(t("email.schedule.test", lang=lang))}</p>')
    sub = " · ".join(p for p in (snapshot.drawn, snapshot.filters) if p)
    parts.append(f'<p style="margin:-6px 0 16px;font-size:13px;color:{MUTED}">{esc(sub)}</p>')
    if note:
        parts.append(f'<div style="margin:0 0 16px;padding:12px 14px;border-left:3px solid {PRIMARY};background:{PAPER};'
                     f'border-radius:4px"><div style="font-size:12px;color:{MUTED};margin-bottom:4px">'
                     f'{esc(t("email.schedule.note_from", lang=lang, name=note_from))}</div>'
                     f'<div style="font-size:14px;line-height:1.5;color:{INK};white-space:pre-line">{esc(note)}</div></div>')
    many = len(snapshot.tabs) > 1
    for tab in snapshot.tabs:
        if many:
            parts.append(f'<div style="margin:22px 0 4px;padding-top:14px;border-top:2px solid {LINE};font-size:17px;'
                         f'font-weight:700;color:{INK}">{esc(tab.name)}</div>')
        run: list = []
        for tile in tab.tiles + [None]:
            if tile is not None and tile.kind == "number":
                run.append(tile)
                continue
            if run:
                parts.append(_numbers_html(run, lang))
                run = []
            if tile is None:
                break
            if tile.kind == "chart":
                parts.append(_chart_html(tile, link, lang))
            elif tile.kind == "table":
                parts.append(_table_html(tile, lang))
            elif tile.kind == "error":
                parts.append(_heading(tile) + f'<p style="margin:0;font-size:14px;color:{BAD}">'
                             f'{esc(t("email.schedule.tile_failed", lang=lang, error=tile.error))}</p>')
            else:
                parts.append(_heading(tile) + f'<p style="margin:0;font-size:14px;color:{MUTED}">'
                             f'{esc(t("ui.dash.no_data", lang=lang))}</p>')
        if not tab.tiles:
            parts.append(f'<p style="font-size:14px;color:{MUTED}">{esc(t("ui.dash.no_data", lang=lang))}</p>')
    for problem in dict.fromkeys(snapshot.problems):
        parts.append(f'<p style="margin:16px 0 0;font-size:13px;color:{MUTED}">{esc(problem)}</p>')
    if pdf_attached:
        parts.append(f'<p style="margin:16px 0 0;font-size:13px;color:{MUTED}">'
                     f'{esc(t("email.schedule.pdf_attached", lang=lang))}</p>')
    parts.append(f'<p style="margin:20px 0 4px"><a href="{esc(link)}" style="display:inline-block;background:{PRIMARY};'
                 f'color:#ffffff;text-decoration:none;font-weight:600;font-size:15px;padding:11px 20px;'
                 f'border-radius:8px">{esc(t("email.open_dashboard", lang=lang))}</a></p>')
    stop = {"en": "Stop these emails", "fr": "Ne plus recevoir ces courriels"}.get(lang, "Stop these emails")
    foot = esc(why) + (f' · <a href="{esc(unsubscribe)}" style="color:{MUTED}">{stop}</a>' if unsubscribe else "")
    return (
        f'<!doctype html><html lang="{esc(lang)}"><head><meta charset="utf-8"><meta name="viewport" '
        f'content="width=device-width"><title>{esc(title)}</title></head>'
        f'<body style="margin:0;padding:0;background:{PAPER};font-family:-apple-system,Segoe UI,Roboto,Helvetica,'
        f'Arial,sans-serif">'
        f'<table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="background:{PAPER}">'
        f'<tr><td align="center" style="padding:28px 12px">'
        f'<table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="max-width:{WIDTH + 56}px">'
        f'<tr><td style="padding:0 4px 14px;font-size:17px;font-weight:700;color:{INK}">Query'
        f'<span style="color:{PRIMARY}">Bot</span></td></tr>'
        f'<tr><td style="background:#ffffff;border:1px solid {LINE};border-radius:12px;padding:26px 28px">'
        f'<h1 style="margin:0 0 10px;font-size:22px;line-height:1.3;color:{INK}">{esc(snapshot.name)}</h1>'
        f'{"".join(parts)}</td></tr>'
        f'<tr><td style="padding:14px 6px;font-size:12px;line-height:1.5;color:{MUTED}">{foot}</td></tr>'
        f'</table></td></tr></table></body></html>')


def dashboard_text(snapshot, *, link: str, note: str = "", note_from: str = "", pdf_attached: bool = False,
                   test: bool = False, why: str = "", unsubscribe: str = "", lang: str = "en") -> str:
    """The same email as plain text."""
    from core.i18n import t

    from core.dashboard_snapshot import TABLE_ROWS

    lines = [snapshot.name, "=" * min(len(snapshot.name), 60)]
    if test:
        lines += [t("email.schedule.test", lang=lang)]
    lines += [" · ".join(p for p in (snapshot.drawn, snapshot.filters) if p), ""]
    if note:
        lines += [t("email.schedule.note_from", lang=lang, name=note_from), note, ""]
    for tab in snapshot.tabs:
        if len(snapshot.tabs) > 1:
            lines += [f"## {tab.name}", ""]
        for tile in tab.tiles:
            if tile.kind == "number":
                change = f"  ({_change(tile.change, lang)[0]})" if tile.change else ""
                lines.append(f"{tile.title}: {tile.value}{change}")
            elif tile.kind == "table":
                lines += ["", tile.title, " | ".join(tile.columns)]
                lines += [" | ".join(row) for row in tile.rows[:TABLE_ROWS]]
                if max(tile.total, len(tile.rows)) > TABLE_ROWS:
                    lines.append(t("email.schedule.more_rows", lang=lang, shown=min(TABLE_ROWS, len(tile.rows)),
                                   total=max(tile.total, len(tile.rows))))
                lines.append("")
            elif tile.kind == "chart":
                lines.append(f"{tile.title}: {t('email.schedule.chart_in_html', lang=lang)}")
            elif tile.kind == "error":
                lines.append(f"{tile.title}: {t('email.schedule.tile_failed', lang=lang, error=tile.error)}")
        lines.append("")
    lines += list(dict.fromkeys(snapshot.problems))
    if pdf_attached:
        lines.append(t("email.schedule.pdf_attached", lang=lang))
    lines += ["", f"{t('email.open_dashboard', lang=lang)}: {link}"]
    if why or unsubscribe:
        stop = {"en": "Stop these emails", "fr": "Ne plus recevoir ces courriels"}.get(lang, "Stop these emails")
        lines += ["", "--", why] + ([f"{stop}: {unsubscribe}"] if unsubscribe else [])
    return "\n".join(lines).strip() + "\n"
