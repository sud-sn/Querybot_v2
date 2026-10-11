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
