"""The Relationships page shows its Tables list, and says why it is empty.

The graph chat panel slides in from the left of the canvas. Closed, it was
moved one width to the left -- onto the Tables list, where nothing clips it --
and covered the list for good. A closed panel is now hidden once it has slid
away (the review panel too, which the page edge happened to clip), so neither
covers anything or takes the keyboard's focus. The browser check
(tools/ui_check/check.py) fails if anything sits over the Tables list.

With no tables at all, the list said "No tables match. Try a different
filter." when no filter was set.
"""

from __future__ import annotations

import re
from pathlib import Path

PAGE = (Path(__file__).resolve().parents[1] / "admin/templates/client_graph.html").read_text(encoding="utf-8")


def _rule(selector: str) -> str:
    return re.search(re.escape(selector) + r"\{([^}]*)\}", PAGE).group(1)


def test_a_closed_panel_is_hidden_once_it_has_slid_away():
    for panel in ("#chat-panel", "#review-panel"):
        closed, opened = _rule(panel), _rule(panel + ".open")
        assert "visibility:hidden" in closed and "visibility 0s linear .2s" in closed, panel
        assert "visibility:visible" in opened, panel


def test_an_empty_list_says_where_tables_come_from():
    assert "No tables yet.<br>Discovery on Setup brings them in." in PAGE
    empty = PAGE[PAGE.index("if (visible === 0) list.innerHTML"):]
    assert empty.index("graphData.entities.length") < empty.index("No tables match")


def test_the_browser_check_looks_for_a_covered_list():
    check = (Path(__file__).resolve().parents[1] / "tools/ui_check/check.py").read_text(encoding="utf-8")
    assert "tables list covered" in check and "closest('#eg-sb')" in check
