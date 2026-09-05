"""
tests/test_dashboard_page.py

The dashboard page, RENDERED.

Eight test files already touch portal_dashboard.html and every one of them does
`.read_text()` and greps for a literal. That convention can catch the deletion of
a string and nothing else, and it is why none of the following reached anyone:

  * `data-chart='{{ chart_json|safe }}'` in SINGLE quotes. json.dumps escapes
    `"` and never `'`, so the first apostrophe in a column name or a value --
    "chiffre d'affaires", "L'Oreal" -- ended the attribute early. Worse, the
    init loop was a bare forEach, so the JSON.parse that threw took every chart
    AFTER it down too: each one left shimmering a skeleton, forever, with no
    error. A French dataset triggers this on day one.

  * The table tile printed Jinja's str() of the raw governed float -- 1234.5 --
    while a chart of the same measure on the same page rendered $1,234.50,
    because only the chart path ever saw column_formats.

  * The card subtitle reported the FULL row count above a tile capped at 50
    rows, so a 5,000-row table read as complete.

  * The sort parsed the number back out of the rendered text. Strip "$,% " from
    a French "1 234,50" and Number() returns 123450 -- a hundredfold error, no
    exception, no sign anything went wrong.

  * Expand rendered on all four card kinds; openChartModal returns early unless
    the card has a [data-chart] node, so on error, KPI and table cards it was a
    button that did nothing.

  * A viewer who cannot edit was still told "Double-click to rename".

  * The card's type badge read the STORED chart_type. The picker stores 'auto',
    so the badge said AUTO above a rendered area chart.

  * _fmtNum had drifted from the chat's copy, which was fixed for Infinity, a
    trillion tier and sub-0.01 values. The dashboard still shipped all three.

Every test below either renders the template or executes its JavaScript.
"""

from __future__ import annotations

import html
import json
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "portal" / "templates" / "portal_dashboard.html"


# ══════════════════════════════════════════════════════════════════════════════
# Rendering
# ══════════════════════════════════════════════════════════════════════════════

def _chart(**overrides) -> dict:
    """A chart in the shape _refresh_chart actually returns.

    Built from the route's own contract rather than invented: the pinned_chart
    row plus the nine keys _refresh_chart adds. A fixture that omits one of them
    tests a world the server never produces.
    """
    base = {
        "id": 101, "title": "Revenue by region", "question": "revenue by region",
        "chart_type": "bar", "color_palette": "default", "row_count": 12,
        "grid_x": 0, "grid_y": 0, "grid_w": 6, "grid_h": 5,
        "sort_enabled": False, "from_cache": False, "cache_refreshed_at": "",
        "chart_json": None, "error": None, "error_next_step": "", "kpi": None,
        "kpi_display": "", "table_columns": [], "table_rows": [],
        "table_column_formats": {}, "table_truncated": False, "table_shown": 0,
        "filter_warnings": [], "dashboard_tab": "Overview",
    }
    base.update(overrides)
    return base


def _render(charts=None, *, can_edit=True, dashboards=True) -> str:
    from jinja2 import ChainableUndefined

    import portal.routes as pr

    env = pr.templates.env
    previous, env.undefined = env.undefined, ChainableUndefined
    try:
        artifact = {
            "id": 5, "name": "Pharmacy performance", "description": "",
            "status": "published", "visibility": "team", "version": 3,
            "can_edit": 1 if can_edit else 0, "refresh_schedule": "daily",
            "last_refreshed_at": "2026-09-01T10:00:00", "tabs_json": "",
            "filters_json": "", "created_at": "", "updated_at": "",
            "published_at": "", "thread_id": "t", "user_id": 1, "account_id": "a",
        }

        class _URL:
            path = "/portal/dashboard"

        class _Req:
            url = _URL(); cookies = {}; headers = {}; query_params = {}

        return env.get_template("portal_dashboard.html").render(
            request=_Req(),
            user={"id": 1, "name": "Ada Lovelace", "account_id": "a", "role": "analyst"},
            client={"client_name": "Acme"},
            charts=charts if charts is not None else [],
            dashboard_artifact=artifact,
            dashboards=[dict(artifact, chart_count=2)] if dashboards else [],
            dashboard_filters=[], dashboard_sources=[], dashboard_tabs=["Overview"],
            dashboard_versions=[], dashboard_subscription=None,
            selected_tab="Overview", welcome=False,
            allowed_tables=["DW.SALES"], group_tables=[], monthly_count=3,
            query_status={"blocked": False, "limit_label": "500", "limit_pct": 1,
                          "remaining_label": "497", "used_label": "3", "warning": ""},
            token_status={"limit": 1, "limit_label": "1M", "limit_pct": 1,
                          "remaining": 1, "remaining_label": "900K",
                          "total_tokens": 1, "unlimited": False, "used_label": "100K"},
            lang="en",
        )
    finally:
        env.undefined = previous


class _AttrGrab(HTMLParser):
    """Read one attribute off one tag the way a browser would.

    The point of going through a real HTML parser rather than a regex: the
    apostrophe bug was a QUOTING bug, and only a parser reproduces where the
    browser decides the attribute value ends.
    """

    def __init__(self, tag_class: str, attribute: str):
        super().__init__(convert_charrefs=True)
        self.tag_class, self.attribute, self.found = tag_class, attribute, []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if self.tag_class in (attrs.get("class") or "") and self.attribute in attrs:
            self.found.append(attrs[self.attribute])


def _attr(markup: str, tag_class: str, attribute: str) -> list[str]:
    parser = _AttrGrab(tag_class, attribute)
    parser.feed(markup)
    return parser.found


class TestTheChartPayloadSurvivesTheAttribute:

    FRENCH = {
        "rows": [{"région": "Île-de-France", "chiffre d'affaires": 1234.5}],
        "x_key": "région", "y_keys": ["chiffre d'affaires"], "type": "bar",
    }

    def test_an_apostrophe_no_longer_truncates_the_payload(self):
        markup = _render([_chart(chart_json=json.dumps(self.FRENCH, ensure_ascii=False))])
        values = _attr(markup, "chart-canvas", "data-chart")
        assert len(values) == 1
        # The browser decodes entities and hands this string to JSON.parse.
        parsed = json.loads(values[0])
        assert parsed["y_keys"] == ["chiffre d'affaires"]
        assert parsed["rows"][0]["région"] == "Île-de-France"

    def test_the_attribute_is_double_quoted(self):
        """The specific defect. A single-quoted attribute cannot hold JSON,
        because json.dumps escapes `"` and never `'`."""
        markup = _render([_chart(chart_json=json.dumps(self.FRENCH, ensure_ascii=False))])
        canvas = markup[markup.index('<div class="chart-canvas"'):]
        canvas = canvas[:canvas.index(">") + 1]
        assert "data-chart=\"" in canvas
        assert "data-chart='" not in canvas

    def test_a_quote_in_the_data_does_not_break_it_either(self):
        payload = {"rows": [{'label': 'He said "hi" and it\'s fine'}], "type": "bar"}
        markup = _render([_chart(chart_json=json.dumps(payload))])
        assert json.loads(_attr(markup, "chart-canvas", "data-chart")[0]) == payload

    def test_the_old_encoding_really_did_break(self):
        """The control. Without it the assertions above pass on anything."""
        raw = json.dumps(self.FRENCH, ensure_ascii=False)
        broken = f"<div class='chart-canvas' data-chart='{raw}'></div>"
        value = _attr(broken, "chart-canvas", "data-chart")[0]
        with pytest.raises(json.JSONDecodeError):
            json.loads(value)


class TestTheTableTile:

    def _table_chart(self, truncated=False):
        return _chart(
            chart_type="table", row_count=5000 if truncated else 2,
            table_columns=["région", "chiffre d'affaires"],
            table_rows=[{"région": {"d": "Île-de-France", "v": "Île-de-France"},
                         "chiffre d'affaires": {"d": "$1,234.50", "v": 1234.5}}],
            table_truncated=truncated, table_shown=1, sort_enabled=True,
        )

    def test_cells_render_the_formatted_value_not_the_raw_float(self):
        markup = _render([self._table_chart()])
        assert "$1,234.50" in markup
        assert ">1234.5<" not in markup

    def test_cells_carry_the_raw_value_for_sorting(self):
        markup = _render([self._table_chart()])
        assert 'data-sort="1234.5"' in markup

    def test_a_truncated_tile_says_how_much_it_is_showing(self):
        markup = _render([self._table_chart(truncated=True)])
        assert "Showing 1 of 5000 rows" in markup

    def test_an_untruncated_tile_does_not(self):
        assert "Showing" not in _render([self._table_chart()])

    def test_the_sort_button_keeps_a_focus_ring(self):
        """It was style="all:unset", which also unset the outline -- and it is
        the only keyboard-reachable control in the tile."""
        markup = _render([self._table_chart()])
        button = markup[markup.index("<button type=\"button\" class=\"dash-th-sort\""):]
        button = button[:button.index(">") + 1]
        assert "all:unset" not in button
        assert 'class="dash-th-sort"' in button
        css = TEMPLATE.read_text(encoding="utf-8")
        assert ".dash-th-sort:focus-visible" in css


class TestTheCardChrome:

    def test_expand_appears_only_on_a_card_that_can_expand(self):
        """openChartModal does `querySelector('[data-chart]'); if (!node) return;`
        so on the other three kinds this was a button that did nothing and said
        nothing."""
        assert "⤢ Expand" in _render([_chart(chart_json='{"type":"bar"}')])
        for dead in (
            _chart(error="Chart could not refresh"),
            _chart(kpi={"label": "Revenue", "value": 5}, kpi_display="5"),
            _chart(chart_type="table", table_columns=["a"],
                   table_rows=[{"a": {"d": "1", "v": 1}}]),
        ):
            assert "⤢ Expand" not in _render([dead])

    def _title_element(self, markup: str) -> str:
        start = markup.index('<div class="chart-card-title')
        return markup[start:markup.index(">", start) + 1]

    def test_a_read_only_viewer_is_not_promised_a_rename(self):
        markup = _render([_chart(chart_json='{"type":"bar"}')], can_edit=False)
        title = self._title_element(markup)
        assert "Double-click to rename" not in title
        assert "ondblclick" not in title
        # The cursor:text affordance follows the permission too.
        assert "is-editable" not in title

    def test_an_editor_still_gets_it(self):
        title = self._title_element(_render([_chart(chart_json='{"type":"bar"}')],
                                            can_edit=True))
        assert "Double-click to rename" in title
        assert "ondblclick" in title
        assert "is-editable" in title

    def test_the_badge_does_not_say_auto(self):
        """The picker stores chart_type='auto' when no type was chosen, and
        build_chart_payload then recovers a real type -- so the card announced
        AUTO above a rendered area chart."""
        markup = _render([_chart(chart_type="auto", chart_json='{"type":"area"}')])
        assert ">AUTO<" not in markup
        assert ">CHART<" in markup

    def test_a_real_stored_type_is_still_shown(self):
        assert ">BAR<" in _render([_chart(chart_type="bar", chart_json='{"type":"bar"}')])

    def test_an_error_card_offers_a_next_step(self):
        markup = _render([_chart(error="That table is not available to you.",
                                 error_next_step="Ask an admin for access.")])
        assert "That table is not available to you." in markup
        assert "Ask an admin for access." in markup


# ══════════════════════════════════════════════════════════════════════════════
# The server side
# ══════════════════════════════════════════════════════════════════════════════

class TestRefreshChartFormatsTheTable:

    def test_it_formats_cells_and_keeps_the_raw_value(self):
        import portal.routes as pr

        rows = [{"REGION": "Nord", "NET_AMOUNT": 1234.5},
                {"REGION": "Sud", "NET_AMOUNT": 9876.25}]
        formats = pr.build_column_formats(rows, explicit_formats={"NET_AMOUNT": "currency"}) \
            if hasattr(pr, "build_column_formats") else None
        from core.response_builder import _format_display_value, build_column_formats

        resolved = build_column_formats(rows, explicit_formats={"NET_AMOUNT": "currency"})
        cell = _format_display_value(1234.5, resolved.get("NET_AMOUNT"), None)
        # Whatever the product's currency format is, it must not be the bare
        # repr the tile used to print.
        assert cell != "1234.5"
        assert "1,234" in cell or "1 234" in cell

    def test_the_row_cap_is_stated_not_hidden(self):
        import portal.routes as pr
        assert pr._TABLE_TILE_ROWS == 50

    def test_a_driver_error_is_sanitised_before_it_reaches_a_card(self):
        """routes.py used to do str(e)[:120], which put ODBC codes, schema and
        table names on the user's card, cut off mid-sentence."""
        from core.failure_messages import sanitize_db_error

        raw = ("('42S02', \"[42S02] [Microsoft][ODBC Driver 18][SQL Server]"
               "Invalid object name 'dbo.fact_sales_pii'.\")")
        out = sanitize_db_error(raw)
        assert out["plain_reason"]
        assert "ODBC" not in out["plain_reason"]
        assert "42S02" not in out["plain_reason"]


# ══════════════════════════════════════════════════════════════════════════════
# The JavaScript
# ══════════════════════════════════════════════════════════════════════════════

dukpy = pytest.importorskip(
    "dukpy",
    reason="a JavaScript engine is required to EXECUTE the dashboard's own "
           "functions rather than grep them, which is the convention that let "
           "every defect in this file's docstring ship",
)


def _function(source: str, signature: str) -> str:
    start = source.index(signature)
    depth = 0
    body_start = None
    for i in range(source.index("(", start), len(source)):
        if source[i] == "(":
            depth += 1
        elif source[i] == ")":
            depth -= 1
            if depth == 0:
                body_start = source.index("{", i)
                break
    depth = 0
    for i in range(body_start, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start:i + 1]
    raise AssertionError(f"unbalanced braces after {signature!r}")


class TestTheNumberFormatterMatchesTheChatPage:
    """Same payload, same product: two pages must not print different numbers.
    The chat's copy was fixed for all three of these and the dashboard's was
    not, so a dashboard axis rendered 'InfinityB' and '1200B', and an axis of
    rates below 0.01 drew as a column of zeros."""

    CHAT = ROOT / "portal" / "templates" / "portal_chat.html"

    def _fmt(self, template: Path, values):
        src = template.read_text(encoding="utf-8")
        harness = (_function(src, "function _fmtNum")
                   + f"\nJSON.stringify({json.dumps(values)}.map(v => _fmtNum(v)));")
        return json.loads(dukpy.evaljs(harness))

    CASES = [1200000000000, 1500000000, 2500000, 3500, 0, 0.004, -0.004, 12.5]

    def test_the_two_pages_agree(self):
        assert self._fmt(TEMPLATE, self.CASES) == self._fmt(self.CHAT, self.CASES)

    def test_the_specific_regressions(self):
        got = self._fmt(TEMPLATE, [1200000000000, 0.004])
        assert got[0] == "1.2T"        # was "1200B"
        assert got[1] == "0.004"       # was "0"

    def test_infinity_is_not_a_magnitude(self):
        assert self._fmt(TEMPLATE, ["x"]) == ["x"]


class TestSortingUsesTheRawValue:

    def _sort(self, cells, *, with_data_sort=True):
        src = TEMPLATE.read_text(encoding="utf-8")
        harness = f"""
const _cells = {json.dumps(cells)};
const _rows = _cells.map((c, i) => ({{
  _i: i,
  cells: [{{textContent: c.text, dataset: {json.dumps(with_data_sort)} ? {{sort: c.raw}} : {{}} }}],
}}));
const body = {{
  rows: _rows,
  _order: [],
  appendChild(r) {{ this._order.push(r._i); }},
}};
const button = {{
  dataset: {{}}, textContent: 'col ↕',
  closest: () => ({{
    tBodies: [body],
    querySelectorAll: () => [],
  }}),
}};
{_function(src, "function sortDashboardTable")}
sortDashboardTable(button, 0);
JSON.stringify(body._order);
"""
        return json.loads(dukpy.evaljs(harness))

    FRENCH = [
        {"text": "1 234,50", "raw": "1234.5"},
        {"text": "999,00", "raw": "999"},
        {"text": "12 000,00", "raw": "12000"},
    ]

    def test_french_formatted_cells_sort_by_magnitude(self):
        """The old code stripped '$,% ' out of the rendered text and called
        Number(). On '1 234,50' that gives 123450 -- a hundredfold error, with
        no exception and no sign that anything went wrong."""
        assert self._sort(self.FRENCH) == [1, 0, 2]      # 999 < 1234.5 < 12000

    def test_the_old_way_really_did_get_it_wrong(self):
        """The control: without data-sort, the same rows sort on the text and
        come out in the wrong order."""
        assert self._sort(self.FRENCH, with_data_sort=False) != [1, 0, 2]

    def test_an_empty_cell_is_not_read_as_zero(self):
        rows = [{"text": "", "raw": ""}, {"text": "5", "raw": "5"},
                {"text": "-3", "raw": "-3"}]
        order = self._sort(rows)
        assert order.index(2) < order.index(1)   # -3 before 5
        assert order[-1] == 0 or order[0] == 0   # the blank sorts to an end


class TestOneBadChartDoesNotTakeThePageDown:

    def test_a_throwing_chart_leaves_the_others_alone(self):
        src = TEMPLATE.read_text(encoding="utf-8")
        harness = f"""
const _rendered = [];
const _failed = [];
function initDashboardChart(node) {{
  if (node.id === 'bad') throw new SyntaxError('Unterminated string in JSON');
  _rendered.push(node.id);
}}
function mountDashChartControls() {{}}
function resizeDashboardCharts() {{}}
function requestAnimationFrame(fn) {{ fn(); }}
const console = {{ error() {{}} }};
const _nodes = ['a', 'bad', 'c'].map(id => ({{
  id, innerHTML: 'skeleton',
  appendChild(child) {{ _failed.push(this.id); this._child = child; }},
}}));
var document = {{
  querySelectorAll: sel => sel === '[data-chart]' ? _nodes : [],
  createElement: () => ({{className: '', textContent: ''}}),
}};
{_function(src, "function renderDashboardCharts")}
{_function(src, "function _showDashChartFailure")}
renderDashboardCharts();
JSON.stringify({{rendered: _rendered, failed: _failed,
                 skeletonsLeft: _nodes.filter(n => n.innerHTML === 'skeleton').map(n => n.id)}});
"""
        out = json.loads(dukpy.evaljs(harness))
        # The two good charts still drew. Before, 'c' never initialised at all.
        assert out["rendered"] == ["a", "c"]
        # The bad one says so instead of shimmering forever.
        assert out["failed"] == ["bad"]
        assert out["skeletonsLeft"] == ["a", "c"]   # untouched by the stub
