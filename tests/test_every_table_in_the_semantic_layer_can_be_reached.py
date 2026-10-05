"""
Every table in the Semantic Layer's list can be reached, and read whole.

The list of a schema's tables stays in view while the reader scrolls through
the tables themselves (position: sticky). With seventeen tables it was taller
than the window, and a sticky box taller than the window never shows its end:
the last tables could not be scrolled to. And a long qualified name
(DATABASE.SCHEMA.TABLE) could not wrap, so it pushed each table's score off
the card's edge, where the card clipped it.

Checked in Chromium when it was fixed (17 tables, a 1919x934 window): before,
the list ran to 1,279px and all 17 scores were clipped; after, it ends inside
the window, scrolls on its own to the last table, and no score is clipped.
This file holds the two rules that made that so, since the suite has no
browser.
"""

from __future__ import annotations

import re

from tests.portal_render import render

_TABLES = [{
    "table": f"TABLE_NUMBER_{number}", "schema": "SALESMART", "fqn": f"WAREHOUSE_DB.SALESMART.TABLE_NUMBER_{number}",
    "field_count": 7, "confidence": 88, "overview": "",
    "fields": [{"column": "COL", "type": "int", "nullable": "", "distinct_values": "", "meaning": "m",
                "use_case": "u", "synonyms": [], "confidence": 90, "pending": False, "approved": False,
                "needs_context": False}],
} for number in range(17)]


def _page() -> str:
    return str(render("portal_kb.html", lang="en", path="/portal/kb", user={"id": 1, "name": "U", "account_id": "a"},
                      pending_count=0, saved=False, semantic_tables=_TABLES, schemas=["SALESMART"],
                      selected_schema="SALESMART", visible_tables=_TABLES))


def _rule(page: str, selector: str) -> str:
    found = re.search(r"(?:^|[}\s])" + re.escape(selector) + r"\{([^}]*)\}", page)
    assert found, selector
    return found.group(1)


def test_a_list_that_stays_in_view_fits_the_window_and_scrolls():
    page = _page()

    if "position:sticky" in _rule(page, ".table-list"):
        assert "max-height:calc(100vh" in _rule(page, ".table-list")
        assert "overflow-y:auto" in _rule(page, ".table-list-body")
    links = re.search(r'<div class="table-list-body">(.*?)</aside>', page, re.S)
    assert links and links.group(1).count('class="table-link"') == 17


def test_a_long_name_wraps_at_its_dots_and_leaves_the_score_in_place():
    page = _page()

    assert "WAREHOUSE_DB.<wbr>SALESMART.<wbr>TABLE_NUMBER_16" in page
    assert "min-width:0" in _rule(page, ".table-link-text")
    assert "flex-shrink:0" in _rule(page, ".table-link .status-pill")
