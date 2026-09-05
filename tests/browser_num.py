"""
tests/browser_num.py

The shell's number and date helpers, for any harness that executes page JS.

Not a test module. portal_base.html defines window.QB_NUM, window.qbNum and
window.qbParseNum once for the whole document, and both page templates now
format through them -- so a harness that lifts a page function without them
raises "window is not defined", which is the harness being wrong rather than
the page. This assembles the same `window` the browser has, from the real
shell and the real core/i18n.py separators.
"""

from __future__ import annotations

import json
from pathlib import Path

from core import i18n
from tests.js_lift import function as lift

SHELL = (Path(__file__).resolve().parents[1]
         / "portal" / "templates" / "portal_base.html").read_text(encoding="utf-8")


def preamble(lang: str = "en") -> str:
    """JS that puts the shell's number and date contract in scope, for `lang`."""
    return f"""
var window = window || {{}};
window.QB_LANG = {json.dumps(lang)};
window.QB_I18N = {json.dumps(i18n.catalogue_for(lang, "date."))};
window.QB_NUM = {json.dumps(i18n.number_format(lang))};
window.QB_DATE = {json.dumps(i18n.date_format(lang))};
{lift(SHELL, "window.qbT = function (id, vars)")}
{lift(SHELL, "window.qbNum = function (value, options)")}
{lift(SHELL, "window.qbParseNum = function (value)")}
{lift(SHELL, "window.qbMonth = function (month, short)")}
{lift(SHELL, "window.qbDate = function (year, month, day, style)")}
{lift(SHELL, "window.qbTime = function (date)")}
"""
