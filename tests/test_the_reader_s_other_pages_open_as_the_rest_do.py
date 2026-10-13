"""The reader's other pages open as Home, the chat and the dashboards do.

After the shell, Home, the chat and the dashboards took the approved design, the Data guide, the notifications,
settings, a password and the sign-in page kept the earlier one: a small heading, cards with a shaded head and a
shadow, square buttons, tabs as an underline, and the notifications drawn with inline styles (a heading set
beside its line, not over it). They now share one vocabulary in portal.css:

- a page opens with its name in the display face, its line under it;
- a card sits on the card radius with no shaded head or shadow, its name in the display face;
- buttons are pills, a set of tabs one segmented control (as a dashboard's tabs);
- a page of cards is one column as wide as reading wants (.reader-page), a card's entries rows (.reader-row);
- the Data guide's questions to try are pills with an arrow that wrap when long (one was cut at the card's edge),
  its subjects one segmented control;
- the sign-in page names itself in its card.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.portal_render import render, visible

ROOT = Path(__file__).resolve().parents[1]
PORTAL_CSS = (ROOT / "static/css/portal.css").read_text(encoding="utf-8")
READER = {"id": 1, "name": "Riley Reader", "role": "analyst", "group_name": "Sales", "account_id": "acct"}


def _rule(css: str, selector: str) -> str:
    """The declarations of the first rule whose selector list holds `selector`."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        selectors = [s.strip() for s in match.group(1).split(",")]
        if selector in selectors:
            return match.group(2)
    raise AssertionError(f"no rule for {selector}")


def _main(html: str) -> str:
    start = html.index('<main class="main"')
    return html[start:html.index("</main>", start)]


def test_a_page_opens_with_its_name_in_the_display_face():
    heading = _rule(PORTAL_CSS, ".portal-app-shell .page-header h1")
    assert "var(--font-display)" in heading and "clamp(" in heading
    assert "var(--text-secondary)" in _rule(PORTAL_CSS, ".portal-app-shell .page-header p")
    # Its line under it, not beside it: the shared header is a row only when it carries actions.
    assert "display: block" in _rule(
        PORTAL_CSS, ".portal-app-shell .page-header:not(.page-header-actions):not(:has(.page-header-actions))")


def test_a_card_a_button_and_tabs_are_the_design_s():
    card = _rule(PORTAL_CSS, ".portal-app-shell .main .card")
    assert "var(--radius-card)" in card and "box-shadow: none" in card
    head = _rule(PORTAL_CSS, ".portal-app-shell .main .card-header")
    assert "background: none" in head and "border-bottom: 0" in head
    assert "var(--font-display)" in _rule(PORTAL_CSS, ".portal-app-shell .main .card-header h2")
    assert "var(--radius-pill)" in _rule(PORTAL_CSS, ".portal-app-shell .main .btn")
    tabs = _rule(PORTAL_CSS, ".portal-app-shell .main .qb-tabs")
    assert "var(--radius-pill)" in tabs and "var(--surface-alt)" in tabs
    chosen = _rule(PORTAL_CSS, '.portal-app-shell .main .qb-tab[aria-selected="true"]')
    assert "var(--surface-raised)" in chosen, "the open tab is raised on the control"


def test_the_notifications_are_cards_of_rows_with_no_inline_styles():
    alerts = [{"id": "a1", "question": "Revenue drop?", "metric_col": "Revenue", "condition": "change_pct",
               "threshold": 10.0, "check_interval_minutes": 60, "status": "active", "last_checked": None}]
    reports = [{"id": 1, "name": "Daily Ops", "description": "Ops summary"},
               {"id": 2, "name": "Weekly margin", "description": ""}]
    subscriptions = {1: {"id": 5, "cadence": "weekly", "day_of_week": 2, "hour": 9, "status": "active",
                         "last_sent": None}}
    html = render("portal_notifications.html", path="/portal/notifications", user=READER, alerts=alerts,
                  reports=reports, subscriptions=subscriptions, my_reports=[{"id": 7, "name": "Mine"}],
                  notices=[], saved=None, error=None)
    main = _main(html)
    assert ' style="' not in main, "every look comes from the stylesheet"
    assert '<div class="reader-page">' in main
    assert main.count('class="reader-row"') == 2 and main.count('class="reader-row reader-row-stack"') == 2
    assert main.count('<h2 id="notice') == 4, "each card names itself in its head"
    # What the page does is unchanged: delete with a confirmation, unsubscribe, subscribe at a chosen time.
    assert 'id="del-alert-a1"' in main and 'id="del-my-report-7"' in main and "data-confirm=" in main
    assert 'action="/portal/notifications/unsubscribe"' in main
    subscribe = main[main.index('action="/portal/notifications/subscribe"'):]
    for name in ("report_id", "cadence", "day_of_week", "hour"):
        assert f'name="{name}"' in subscribe
    assert "Revenue drop?" in visible(main) and "Daily Ops" in visible(main)


def test_settings_and_a_password_are_reader_pages():
    settings = _main(render("portal_settings.html", path="/portal/settings", user=READER, tab="profile"))
    assert 'class="settings-page reader-page"' in settings
    assert 'class="qb-tabs settings-tabs"' in settings, "the tabs take the segmented control"
    page = _main(render("portal_change_password.html", path="/portal/change-password", user=READER,
                        forced=False, error=None))
    assert 'class="reader-page reader-page-narrow"' in page
    assert ' style="' not in page
    assert 'name="current_pw"' in page and 'name="new_pw"' in page and 'name="confirm_pw"' in page


def test_the_data_guide_s_questions_are_links_that_wrap():
    from unittest.mock import MagicMock

    described = MagicMock(headline="This data covers 1 subject.", note="",
                          sections=[{"title": "Order line", "body": "", "bullets": ["Measures: net amount"]}],
                          examples=["New, returning and lapsed customers by month in 2026"])
    html = render("portal_kb.html", path="/portal/kb", user=READER, learned=described, semantic_tables=[],
                  visible_tables=[], schemas=[], pending_count=0)
    pill = re.search(r'<a class="try-question" href="([^"]+)">(.*?)</a>', html, re.S)
    assert pill, "a question to try is a link with an arrow"
    assert pill.group(1) == "/portal/chat?new=1&amp;ask=New%2C%20returning%20and%20lapsed%20customers%20by%20month%20in%202026"
    assert "#arrow-right" in pill.group(2) and "<span>New, returning and lapsed" in pill.group(2)
    rule = _rule(PORTAL_CSS, ".try-question")
    assert "nowrap" not in rule and "max-width: 100%" in rule, "a long question wraps inside its card"
    source = (ROOT / "portal/templates/portal_kb.html").read_text(encoding="utf-8")
    assert source.count('class="try-question"') == 2, "the guide's side card and the learned summary alike"
    assert "border-radius:var(--radius-pill);background:var(--surface-alt)" in source, "subjects: one control"
    assert ".ask-tab.active{background:var(--surface-raised)" in source
    assert 'class="ask-tab{% if loop.first %} active{% endif %}" role="tab"' in source, "a tab of its own, not a chip"


def test_the_sign_in_page_names_itself_in_its_card():
    html = render("portal_login.html", path="/portal/login", user=None, error=None, workspace=None)
    card = html[html.index('<div class="card login-card">'):]
    assert re.search(r'<h1 class="login-title">\s*Sign in\s*</h1>', card)
    assert html.count("<h1") == 1, "one heading, seen, not a hidden one beside it"
    for field in ('id="login-account"', 'id="login-email"', 'id="login-password"', 'class="btn btn-primary login-submit"'):
        assert field in card
    css = html[html.index("<style>"):html.index("</style>")]
    assert ".login-title {" in css and "var(--font-display)" in css
    assert re.search(r"\.login-submit \{[^}]*var\(--radius-pill\)", css)
    assert re.search(r"\.login-card \{[^}]*var\(--radius-card\)", css)
