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
STYLESHEET = ROOT / "static" / "css" / "dashboard.css"


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


class _URL:
    path = "/portal/dashboard"


class _Req:
    """Enough of a Request for the language context processor and the base
    template. cookies/headers are what _request_language actually reads."""

    url = _URL()
    query_params: dict = {}

    def __init__(self, lang=None):
        self.cookies = {"qb_lang": lang} if lang else {}
        self.headers = {}


def _language_context(lang=None) -> dict:
    """The language keys a real render gets, built by the PRODUCTION function.

    Not a hand-written fixture: portal.routes registers _language_context as a
    Jinja context processor, and a direct env.render() does not run it. Calling
    the real one means the fixture cannot drift from what the page is served.
    """
    import portal.routes as pr

    return pr._language_context(_Req(lang))


def _render(charts=None, *, can_edit=True, dashboards=True, lang=None,
            artifact=None, library=None) -> str:
    from jinja2 import ChainableUndefined

    import portal.routes as pr

    env = pr.templates.env
    previous, env.undefined = env.undefined, ChainableUndefined
    try:
        artifact = artifact if artifact is not None else {
            "id": 5, "name": "Pharmacy performance", "description": "",
            "status": "published", "visibility": "team", "version": 3,
            "can_edit": 1 if can_edit else 0, "refresh_schedule": "daily",
            "last_refreshed_at": "2026-09-01T10:00:00", "tabs_json": "",
            "filters_json": "", "created_at": "", "updated_at": "",
            "published_at": "", "thread_id": "t", "user_id": 1, "account_id": "a",
        }


        return env.get_template("portal_dashboard.html").render(
            request=_Req(lang),
            user={"id": 1, "name": "Ada Lovelace", "account_id": "a", "role": "analyst"},
            client={"client_name": "Acme"},
            charts=charts if charts is not None else [],
            dashboard_artifact=artifact or None,
            dashboards=library if library is not None else (
                [dict(artifact, chart_count=2)] if (artifact and dashboards) else []),
            dashboard_filters=[], dashboard_sources=[], dashboard_tabs=["Overview"],
            dashboard_versions=[], dashboard_subscription=None,
            selected_tab="Overview", welcome=False,
            allowed_tables=["DW.SALES"], group_tables=[], monthly_count=3,
            query_status={"blocked": False, "limit_label": "500", "limit_pct": 1,
                          "remaining_label": "497", "used_label": "3", "warning": ""},
            token_status={"limit": 1, "limit_label": "1M", "limit_pct": 1,
                          "remaining": 1, "remaining_label": "900K",
                          "total_tokens": 1, "unlimited": False, "used_label": "100K"},
            **_language_context(lang),
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
        assert ">1234.5<" not in _visible(markup)

    def test_cells_carry_the_raw_value_for_sorting(self):
        markup = _render([self._table_chart()])
        assert 'data-sort="1234.5"' in markup

    def test_a_truncated_tile_says_how_much_it_is_showing(self):
        markup = _render([self._table_chart(truncated=True)])
        assert "Showing 1 of 5000 rows" in markup

    def test_an_untruncated_tile_does_not(self):
        assert "Showing" not in _visible(_render([self._table_chart()]))

    def test_the_sort_button_keeps_a_focus_ring(self):
        """It was style="all:unset", which also unset the outline -- and it is
        the only keyboard-reachable control in the tile."""
        markup = _render([self._table_chart()])
        button = markup[markup.index("<button type=\"button\" class=\"dash-th-sort\""):]
        button = button[:button.index(">") + 1]
        assert "all:unset" not in button
        assert 'class="dash-th-sort"' in button
        css = STYLESHEET.read_text(encoding="utf-8")
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
            assert "Expand</button>" not in _visible(_render([dead]))

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
        assert ">AUTO<" not in _visible(markup)
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


# ══════════════════════════════════════════════════════════════════════════════
# The redesign
# ══════════════════════════════════════════════════════════════════════════════

def _visible(markup: str) -> str:
    """The page's markup with its <script> blocks removed.

    The message catalogue is injected as `const I18N = {...}`, so EVERY English
    string is present in the page source regardless of what renders. Any
    assertion of the form "this text is not on the page" has to look at the
    markup only, or it passes forever -- which is precisely the false assurance
    this file exists to avoid.
    """
    return re.sub(r"<script\b.*?</script>", " ", markup, flags=re.S | re.I)


def _headings(markup: str) -> list[tuple[str, str]]:
    return [(m.group(1), re.sub(r"<[^>]+>", "", m.group(2)).strip())
            for m in re.finditer(r"<(h[1-3])[^>]*>(.*?)</\1>", markup, re.S)]


class TestThePageLeadsWithTheDashboard:
    """The hierarchy was inverted. The h1 was a permanently generic
    "Dashboards"; the dashboard you had actually opened was an h2 sitting BELOW
    the account KPI strip and the table-access card."""

    def test_the_open_dashboard_is_the_h1(self):
        headings = _headings(_render([_chart(chart_json='{"type":"bar"}')]))
        assert headings[0] == ("h1", "Pharmacy performance")

    def test_the_generic_heading_is_gone_when_a_dashboard_is_open(self):
        markup = _render([_chart(chart_json='{"type":"bar"}')])
        assert ">Dashboards</h1>" not in _visible(markup)

    def test_the_library_view_still_leads_with_dashboards(self):
        """On the library view the account IS the subject, so it keeps the
        generic heading and the KPI strip up front."""
        markup = _render(
            [], artifact={},
            library=[{"id": 5, "name": "Ops", "status": "published",
                      "version": 2, "chart_count": 3, "can_edit": 1}],
        )
        assert _headings(markup)[0] == ("h1", "Dashboards")
        assert "Ops" in markup

    def test_the_account_context_is_demoted_but_not_lost(self):
        """Still reachable on the dashboard view -- just no longer outranking
        the artifact."""
        markup = _render([_chart(chart_json='{"type":"bar"}')])
        assert "Workspace usage and table access" in markup
        assert "Remaining tokens" in markup      # the wiring test's literal
        assert "Query limit" in markup


class TestTheUnpublishedTeamDashboardIsVisible:
    """Adding a chart, DRAGGING a tile, changing a palette or renaming a card
    all mark a team dashboard draft, and get_dashboard_for_view gates on
    `visibility='team' AND status='published'` -- so one drag made it vanish
    from every teammate's portal. The page said nothing, and publishing was a
    chat command only."""

    def _render_status(self, status, visibility, can_edit=True, lang=None):
        return _render(
            [_chart(chart_json='{"type":"bar"}')], lang=lang, library=[],
            artifact={"id": 5, "name": "Ops", "status": status,
                      "visibility": visibility, "version": 3,
                      "can_edit": 1 if can_edit else 0,
                      "refresh_schedule": "daily", "thread_id": "t"},
        )

    def test_a_draft_team_dashboard_says_teammates_cannot_see_it(self):
        markup = self._render_status("draft", "team")
        assert "teammates cannot see this dashboard" in markup.lower()
        assert 'action="/portal/dashboard/5/publish"' in markup

    def test_a_published_team_dashboard_says_nothing(self):
        markup = self._render_status("published", "team")
        assert "teammates cannot see" not in _visible(markup).lower()
        assert "/publish" not in _visible(markup)

    def test_a_personal_draft_says_nothing(self):
        """A personal dashboard has no audience to lose."""
        markup = self._render_status("draft", "personal")
        assert "teammates cannot see" not in _visible(markup).lower()

    def test_a_viewer_who_cannot_edit_is_not_asked_to_publish(self):
        markup = self._render_status("draft", "team", can_edit=False)
        assert "/publish" not in _visible(markup)

    def test_the_status_is_a_pill_not_a_run_on_sentence(self):
        markup = self._render_status("draft", "team")
        assert 'class="dash-status is-draft"' in markup
        assert 'class="dash-status is-published"' in self._render_status("published", "team")


class TestThePublishEndpoint:

    def _client(self):
        import os
        import tempfile
        os.environ.setdefault("QUERYBOT_DB_PATH",
                              os.path.join(tempfile.mkdtemp(), "dash.db"))
        from fastapi import FastAPI
        from starlette.testclient import TestClient
        import portal.routes as pr
        import store
        store.init_db()
        app = FastAPI()
        app.include_router(pr.router)
        return TestClient(app), pr, store

    def _signed_in(self):
        import os
        client, pr, store = self._client()
        account_id = f"acct{os.urandom(4).hex()}"
        store.upsert_client(account_id, "T")
        user_id, _ = store.create_user(account_id, "Ada", f"{os.urandom(4).hex()}@x.com")
        client.cookies.set(pr._COOKIE, pr._sign_session_value(user_id))
        return client, store, account_id, user_id

    def test_it_publishes_and_the_dashboard_reappears_for_the_team(self):
        client, store, account_id, user_id = self._signed_in()
        dashboard = store.create_dashboard(account_id, user_id, "t", "Ops",
                                           visibility="team")
        dashboard_id = int(dashboard["id"])
        assert store.get_dashboard(dashboard_id, user_id, account_id)["status"] == "draft"

        response = client.post(f"/portal/dashboard/{dashboard_id}/publish",
                               follow_redirects=False)
        assert response.status_code == 303
        assert store.get_dashboard(dashboard_id, user_id, account_id)["status"] == "published"

    def test_anonymous_cannot_publish(self):
        client, _, _ = self._client()
        response = client.post("/portal/dashboard/1/publish", follow_redirects=False)
        assert response.status_code in (303, 401)
        assert "/portal/login" in response.headers.get("location", "") or \
            response.status_code == 401

    def test_it_cannot_publish_someone_elses_dashboard(self):
        """publish_dashboard's UPDATE is scoped by user_id AND account_id, so a
        wrong owner is a no-op rather than an error -- assert the no-op."""
        import os
        client, store, account_id, user_id = self._signed_in()
        other_id, _ = store.create_user(account_id, "Bob", f"{os.urandom(4).hex()}@x.com")
        dashboard = store.create_dashboard(account_id, other_id, "t", "Theirs",
                                           visibility="team")
        client.post(f"/portal/dashboard/{int(dashboard['id'])}/publish",
                    follow_redirects=False)
        assert store.get_dashboard(int(dashboard["id"]), other_id,
                                   account_id)["status"] == "draft"


class TestLandmarksAndTableSemantics:

    def test_the_provenance_block_is_a_landmark(self):
        markup = _render([_chart(chart_json='{"type":"bar"}')])
        assert '<section class="artifact-details"' in markup
        assert 'aria-label="Dashboard provenance"' in markup

    def test_the_page_css_lives_in_a_stylesheet_not_the_template(self):
        """66 lines of CSS in a 989-line template is why nobody noticed that
        eight of its declarations had been dead since production.css landed."""
        assert "<style>" not in TEMPLATE.read_text(encoding="utf-8")
        assert "/static/css/dashboard.css" in TEMPLATE.read_text(encoding="utf-8")

    def test_the_stylesheet_still_loads_before_production_css(self):
        """production.css calls itself the layer "loaded after page-specific
        styles" -- a quieting pass that flattens gradients and removes the hover
        lift. Loading this page's CSS after it would silently revert that."""
        base = (ROOT / "portal" / "templates" / "portal_base.html").read_text(encoding="utf-8")
        assert base.index("{% block head %}") < base.index("production.css")

    def test_the_dead_declarations_were_deleted_not_promoted(self):
        css = STYLESHEET.read_text(encoding="utf-8")
        body = css[css.index("*/") + 2:]
        # Each of these was overridden by production.css and read as though it
        # applied. Re-adding one would be a silent visual regression.
        assert "translateY(-1px)" not in body
        assert "shadow-md" not in body
        assert "linear-gradient(135deg" not in body


# ══════════════════════════════════════════════════════════════════════════════
# French
# ══════════════════════════════════════════════════════════════════════════════

class TestThePageRendersInFrench:

    def test_the_dashboard_header_is_french(self):
        markup = _visible(_render([_chart(chart_json='{"type":"bar"}')], lang="fr"))
        assert "Tableau de bord" in markup            # the kicker
        assert "Discuter du tableau de bord" in markup
        assert "Chat with dashboard" not in markup

    def test_the_empty_state_is_french(self):
        markup = _visible(_render([], lang="fr"))
        assert "Aucun visuel dans ce tableau de bord" in markup

    def test_the_workspace_drawer_is_french(self):
        markup = _visible(_render([_chart(chart_json='{"type":"bar"}')], lang="fr"))
        assert "Jetons restants" in markup            # was "Remaining tokens"
        assert "Limite de requêtes" in markup         # was "Query limit"
        assert "Remaining tokens" not in markup

    def test_no_english_label_survives_on_a_french_page(self):
        markup = _visible(_render([_chart(chart_json='{"type":"bar"}')], lang="fr"))
        for literal in ("Dashboard artifact", "Read only", "Subscribe",
                        "Apply filters", "Live governed data", "Remove",
                        "Queries this month", "Tables access",
                        "Workspace usage and table access"):
            assert literal not in markup, f"still English: {literal!r}"


class TestServerEnumsAreTranslatedNotCapitalised:
    """|capitalize cannot translate, and it also mangles whatever case the
    database happens to hold."""

    def test_status_and_visibility(self):
        markup = _visible(_render([_chart(chart_json='{"type":"bar"}')], lang="fr"))
        assert "Publié" in markup
        assert "Équipe" in markup
        assert ">Published<" not in markup

    def test_the_refresh_schedule_agrees_in_gender(self):
        """French adjectives agree with the noun. "Actualisation quotidienne",
        not "Actualisation quotidien" -- which is why the schedule has its own
        enum group rather than sharing the cadence one."""
        markup = _visible(_render([_chart(chart_json='{"type":"bar"}')], lang="fr"))
        assert "Actualisation quotidienne" in markup

    def test_an_unknown_enum_value_never_renders_as_a_message_id(self):
        """These come from the database. A new status must not put
        "ui.enum.status.archived" on a customer's screen."""
        from core import i18n
        assert i18n.enum_label("status", "archived", lang="fr") == "Archived"
        assert i18n.enum_label("status", "", lang="fr") == ""

    def test_the_chart_badge_translates_but_keeps_the_acronym(self):
        table = _chart(chart_type="table", table_columns=["a"],
                       table_rows=[{"a": {"d": "1", "v": 1}}])
        assert ">TABLEAU<" in _visible(_render([table], lang="fr"))
        kpi = _chart(chart_type="kpi", kpi={"label": "x", "value": 1}, kpi_display="1")
        assert ">KPI<" in _visible(_render([kpi], lang="fr"))


class TestCountsUseTheRightPluralRule:
    """The page had the rule inline as `{% if n != 1 %}s{% endif %}`, which is
    right for English and wrong for French: French takes the SINGULAR at zero,
    so that markup rendered "0 visuels"."""

    def test_french_treats_zero_as_singular(self):
        from core import i18n
        assert i18n.plural("ui.dash.visuals", 0, lang="fr") == "0 visuel"
        assert i18n.plural("ui.dash.visuals", 1, lang="fr") == "1 visuel"
        assert i18n.plural("ui.dash.visuals", 2, lang="fr") == "2 visuels"

    def test_english_does_not(self):
        from core import i18n
        assert i18n.plural("ui.dash.visuals", 0, lang="en") == "0 visuals"
        assert i18n.plural("ui.dash.visuals", 1, lang="en") == "1 visual"

    def test_it_reaches_the_rendered_row_count(self):
        one = _visible(_render([_chart(row_count=1, chart_json='{"type":"bar"}')], lang="fr"))
        many = _visible(_render([_chart(row_count=12, chart_json='{"type":"bar"}')], lang="fr"))
        assert "1 ligne" in one and "1 lignes" not in one
        assert "12 lignes" in many

    def test_a_non_numeric_count_does_not_raise(self):
        """row_count comes off a database row and this runs inside a render."""
        from core import i18n
        assert i18n.plural("ui.dash.visuals", None, lang="fr")
        assert i18n.plural("ui.dash.visuals", "many", lang="en")


class TestTheAccountBlockExistsOnce:

    def test_it_is_a_macro_not_two_copies(self):
        """The restructure left the KPI strip and table-access card rendered
        twice -- once in the dashboard view's drawer and once on the library
        view -- which would have meant translating and maintaining both."""
        source = TEMPLATE.read_text(encoding="utf-8")
        assert source.count('<div class="metrics">') == 1
        assert "{% macro workspace_usage(" in source

    def test_both_views_still_show_it(self):
        dashboard_view = _visible(_render([_chart(chart_json='{"type":"bar"}')]))
        library_view = _visible(_render([], artifact={}, library=[]))
        for markup in (dashboard_view, library_view):
            assert "Remaining tokens" in markup
            assert "Query limit" in markup


class TestTheTranslatorIsNotShadowedInTheScript:

    def test_the_chart_type_buttons_call_the_translator(self):
        """`DASH_TYPES.map(t => ...)` shadowed the page's own t() inside the
        template literal, so calling t('ui.enum...') there would have invoked
        the loop variable -- a string -- as a function."""
        source = TEMPLATE.read_text(encoding="utf-8")
        script = source[source.index("<script>"):]
        assert "DASH_TYPES.filter(t =>" not in script
        assert "DASH_TYPES.filter(kind =>" in script
        assert script.count("t('ui.enum.charttype.' + kind)") == 2

    def test_the_page_has_an_html_escaper_now(self):
        """Every innerHTML on this page was built without one."""
        source = TEMPLATE.read_text(encoding="utf-8")
        assert "function escHtmlDash(" in source
