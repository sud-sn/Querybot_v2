"""A dashboard opens to read, and is edited only when its owner asks to.

Every tile carried a drag handle, a Remove button, a row of type switches, six
palette swatches and the question it came from, all the time, for everyone
who could edit -- the controls outweighed the numbers. The header led with
"Dashboard artifact", a Draft pill and "Version 3" on a personal dashboard.

Now (the approved dashboard step, U4):

- The page opens in view: no handles, no remove buttons, no type switches, no
  question line; the grid is static. Edit shows them and lets the tiles move;
  Done puts them away. Renaming a tile is an edit too.
- No palette switcher: one palette, in a fixed order, everywhere.
- The header says the dashboard's name, who sees it and when its tiles were
  drawn. Draft or published shows only on a dashboard the team shares, and a
  refresh schedule only when there is one.
- The dashboards page lists the dashboards first; the workspace's own numbers
  sit in a drawer below, and an empty list says how to make one.
"""

from __future__ import annotations

import re

from tests.dashboard_render import CHART, render


def _markup(html: str) -> str:
    """The page without its scripts: an absence must mean the markup lacks it."""
    return re.sub(r"<script\b.*?</script>", " ", html, flags=re.S | re.I)


def _page(**kw) -> str:
    return _markup(render(charts=[dict(CHART)], **kw))


def test_the_page_opens_in_view():
    html = _page()
    assert re.search(r'class="dashboard-page" data-mode="view"', html)


def test_its_owner_is_offered_edit():
    html = _page()
    toggle = re.search(r'<button[^>]*id="dashEditToggle"[^>]*>', html)
    assert toggle and 'aria-pressed="false"' in toggle.group(0)


def test_a_reader_who_cannot_edit_is_not():
    html = _page(artifact={"id": 5, "name": "Shared", "status": "published", "visibility": "team",
                           "version": 3, "can_edit": 0, "refresh_schedule": "manual", "thread_id": "t"})
    assert 'id="dashEditToggle"' not in html
    assert "chart-drag-handle" not in html


def test_remove_waits_for_edit():
    html = _page()
    remove = re.search(r'<button[^>]*onclick="unpinChart\([^>]*>', html)
    assert remove and "dash-edit-only" in remove.group(0)


def test_the_view_hides_the_editing_controls():
    css = open("static/css/dashboard.css", encoding="utf-8").read()
    block = css[css.index('.dashboard-page[data-mode="view"]'):]
    block = block[:block.index("}")]
    for part in (".chart-drag-handle", ".dash-edit-only", ".dash-ctrl-row"):
        assert part in block, part
    assert "display: none" in block


def test_a_personal_dashboard_says_who_sees_it_and_when_it_was_drawn():
    html = _page(artifact={
        "id": 5, "name": "Mine", "status": "draft", "visibility": "personal", "version": 4,
        "can_edit": 1, "refresh_schedule": "manual", "thread_id": "t"})
    start = html.index('<header class="dash-header">')
    header = html[start:html.index("</header>", start)]
    assert "Only you" in header                       # who sees it: no one else yet
    assert "dash-status" not in header, "a shared dashboard is always live: no draft or published status"
    assert "Version" not in header and "manual" not in header.lower()
    assert "Dashboard artifact" not in header


def test_a_team_dashboard_says_it_is_shared_with_the_workspace():
    """Always live: a dashboard shared with the workspace shows its current version, so its header says who
    sees it, never "Draft" or "Published"."""
    page = _page()
    header = page[page.index('<header class="dash-header">'):].split("</header>", 1)[0]
    assert "Shared with the workspace" in header and "dash-status" not in header


def test_the_dashboards_page_lists_the_dashboards_first():
    html = render(artifact=False, library=[{"id": 1, "name": "Sales overview", "status": "draft",
                                            "visibility": "personal", "version": 2, "can_edit": 1,
                                            "chart_count": 4}])
    body = _markup(html)
    assert body.index("Sales overview") < body.index("Workspace usage")
    item = re.search(r'<a class="dashboard-library-item".*?</a>', body, re.S).group(0)
    assert "v2" not in item and "Draft" not in item, "a personal dashboard needs no version or status"


def test_no_dashboards_yet_says_how_to_make_one():
    body = _markup(render(artifact=False, library=[]))
    assert "No dashboards yet" in body and 'href="/portal/chat"' in body


def test_the_trail_names_the_page_as_the_menu_does():
    """It said "Dashboard", in English for every reader, under a menu saying "Dashboards"."""
    trail = _page()
    trail = trail[trail.index('class="breadcrumb'):]
    assert "Dashboards" in trail[:trail.index("</nav>")]
