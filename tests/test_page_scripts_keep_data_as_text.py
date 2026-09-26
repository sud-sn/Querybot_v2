"""
What a page's own JavaScript writes into markup stays text.

tests/test_names_cannot_run_as_script.py covers what templates write. The
admin pages also build markup in the browser, from what the server returns,
and there several escapes did too little and several handlers took raw values:

- The setup page's masking preview writes up to three real warehouse rows,
  each cell also as a title="..." tooltip, through an escape that left quotes
  alone: a cell holding  x" onmouseover="...  ran script when the admin moved
  the mouse over it -- and anyone who can write a row to a previewed table can
  write that.
- Table, schema and column names went into handlers raw ('${fqn}'), or as
  JSON inside a single-quoted attribute, where a quote in the name ends the
  attribute: the setup, databases and relationships pages.
- The metric pickers escaped a name for JavaScript by hand and missed the
  backslash; metric names can be text a model wrote in chat.
- The databases page wrote a driver's error message as markup; the system page
  a deployment's status and the vector store's version and manager; the
  column suggestions their names and types.
- Chart tooltips wrote the value as markup.

Now each page's escape covers & < > " ', a value handed to a handler is a JSON
literal escaped for its attribute (jsArg), and server text is escaped
wherever it is written.

Each test lifts the page's own function out of the template (tests/js_lift.py)
and runs it in a JavaScript engine against a stand-in DOM, with hostile names
and values; then reads the markup as a browser would: no element or handler
the function did not write, attributes holding each value exactly, and every
handler run with its targets recorded, receiving each value exactly.
"""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path

import pytest

from tests.js_lift import function as lift

dukpy = pytest.importorskip("dukpy", reason="dukpy runs the pages' own JavaScript")

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "admin" / "templates" / "client_setup.html"
DATABASES = ROOT / "admin" / "templates" / "databases.html"
GRAPH = ROOT / "admin" / "templates" / "client_graph.html"
METRICS = ROOT / "admin" / "templates" / "metrics" / "_scripts.html"
SYSTEM = ROOT / "admin" / "templates" / "system.html"
CHARTS = ROOT / "static" / "js" / "qb-charts.js"

# Each closes a different context: a double-quoted attribute, a single-quoted
# one, the element, and a JavaScript string (the backslash escapes the quote
# a hand-written escape adds).
HOSTILE = [
    'x" onmouseover="alert(1)" y="',
    "x' onmouseover='alert(1)' y='",
    "</td><img src=x onerror=alert(1)>",
    "a\\');alert(1)//",
    "O'Brien & \"Sons\" <Ltd>",
]

# Names a warehouse can hold and a file system can too (no slash).
TABLE_NAMES = ['T1" onmouseover="alert(1)', "T2' onmouseover='alert(1)", "T3\\');alert(1)//", "T4 & <b>"]

DOM = r"""
function El(tag) {
  this.tagName = tag; this.children = []; this.style = {}; this.dataset = {};
  this.innerHTML = ''; this.className = ''; this.textContent = ''; this.value = '';
  this.classList = {add: function(){}, remove: function(){}, toggle: function(){},
                    contains: function(){ return false; }};
}
El.prototype.appendChild = function(c) { this.children.push(c); return c; };
El.prototype.querySelector = function() { return null; };
El.prototype.querySelectorAll = function() { return []; };
El.prototype.addEventListener = function() {};
El.prototype.remove = function() {};
El.prototype.setAttribute = function() {};
var els = {};
var document = {
  getElementById: function(id) { return els[id] || (els[id] = new El('div')); },
  createElement: function(tag) { return new El(tag); },
  querySelectorAll: function() { return []; },
  querySelector: function() { return null; },
  body: new El('body'),
};
var window = this;
function markup(el) { return el.innerHTML + el.children.map(markup).join(''); }
"""


class _Markup(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.elements: list[tuple[str, dict]] = []
        self.text: list[str] = []

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, {k: (v or "") for k, v in attrs}))

    def handle_data(self, data):
        self.text.append(data)


def parse(html: str) -> _Markup:
    parser = _Markup()
    parser.feed(html)
    return parser


def assert_nothing_injected(page: _Markup, handlers=()):
    """No element and no handler the function did not write itself."""
    for tag, attrs in page.elements:
        assert tag not in {"img", "script", "svg", "iframe"}, (tag, attrs)
        for key in attrs:
            if key.startswith("on"):
                assert key in handlers, (tag, key, attrs[key])


def handler_calls(page: _Markup, targets) -> list[list]:
    """Run every inline handler, with the named targets recording their calls."""
    handlers = [value for _tag, attrs in page.elements for key, value in attrs.items()
                if key.startswith("on")]
    stubs = "".join(
        f"var {name} = function() {{ calls.push([{json.dumps(name)}].concat([].slice.call(arguments, 0, 3))); }};"
        for name in targets)
    return dukpy.evaljs(
        "var calls = [];" + stubs + """
        var ev = {stopPropagation: function(){}, preventDefault: function(){}};
        dukpy['handlers'].forEach(function(h) {
          new Function('event', h).call({checked: true, value: 'q'}, ev);
        });
        calls;""",
        handlers=handlers)


def _code(path: Path, *signatures: str) -> str:
    src = path.read_text(encoding="utf-8")
    # ";" after each: a lifted `window.f = function(){...}` is an expression.
    return "".join(lift(src, s) + ";\n" for s in signatures if s in src)


def _args(calls, name, index=0):
    return [c[1 + index] for c in calls if c[0] == name]


# ── The setup page ───────────────────────────────────────────────────────────

SETUP_HELPERS = ("function esc(", "function jsArg(", "function ref(", "function checkboxId(")


class TestTheMaskingPreview:

    def _page(self):
        data = {"status": "ok", "columns": ["NAME", "NOTE"], "masked_fields": ["NAME"],
                "before": [{"NAME": v, "NOTE": v} for v in HOSTILE[:3]],
                "after": [{"NAME": "***", "NOTE": v} for v in HOSTILE[:3]]}
        html = dukpy.evaljs(
            DOM + _code(SETUP, *SETUP_HELPERS, "function _renderMaskPreview(")
            + "_renderMaskPreview(dukpy['data'], 'DB.S.T'); els['mask-preview-body'].innerHTML;",
            data=data)
        return parse(html)

    def test_no_cell_becomes_markup(self):
        assert_nothing_injected(self._page())

    def test_each_cell_is_its_own_tooltip(self):
        titles = [a["title"] for tag, a in self._page().elements if tag == "td" and "title" in a]
        for value in HOSTILE[:3]:
            assert titles.count(value) == 3  # before NAME, before NOTE, after NOTE


class TestTheSchemaTree:

    def _page(self):
        tree = {"DB": {"SALES": {"tables": TABLE_NAMES, "views": []}}}
        html = dukpy.evaljs(
            DOM + "var selected = new Set();" + _code(SETUP, *SETUP_HELPERS, "function renderTree(")
            + "renderTree(dukpy['tree']); markup(els['setup-scope-tree']);",
            tree=tree)
        return parse(html)

    REFS = [f"DB.SALES.{t}".upper() for t in TABLE_NAMES]

    def test_no_name_becomes_markup(self):
        assert_nothing_injected(self._page(), handlers={"onchange", "onclick", "oninput"})

    def test_each_checkbox_and_fields_button_is_handed_its_table(self):
        calls = handler_calls(self._page(),
                              ["setupScopeCheck", "_toggleTableFields", "setupScopeToggleSchema",
                               "filterScopeTree"])
        assert sorted(_args(calls, "setupScopeCheck")) == sorted(self.REFS)
        assert sorted(_args(calls, "_toggleTableFields")) == sorted(self.REFS)
        assert [sorted(refs) for refs in _args(calls, "setupScopeToggleSchema")] == [sorted(self.REFS)]

    def test_the_filter_reads_each_name(self):
        names = [a["data-table-name"] for _t, a in self._page().elements if "data-table-name" in a]
        assert sorted(names) == sorted(t.lower() for t in TABLE_NAMES)


class TestTheTableFieldsPanel:

    FQN = "DB.SALES.O'BRIEN\"X"

    def _page(self):
        data = {"columns": [{"name": v, "type": "varchar"} for v in HOSTILE],
                "auto_masked": [HOSTILE[0]], "strategy_map": {}}
        html = dukpy.evaljs(
            DOM + "var selected = new Set([dukpy['fqn'].toUpperCase()]); var maskingConfig = new Map();"
            + _code(SETUP, *SETUP_HELPERS, "function _renderTblFields(")
            + "var panel = new El('div'); panel.id = 'panel-1';"
            + "_renderTblFields(dukpy['fqn'], panel, dukpy['data']); panel.innerHTML;",
            data=data, fqn=self.FQN)
        return parse(html)

    def test_no_name_becomes_markup(self):
        assert_nothing_injected(self._page(), handlers={"onclick"})

    def test_each_handler_is_handed_the_table_and_the_column(self):
        calls = handler_calls(self._page(), ["_tblMaskPiiOnly", "_tblMaskAll", "_tblMaskNone",
                                             "_tblFieldMaskToggle"])
        assert set(_args(calls, "_tblFieldMaskToggle")) == {self.FQN}
        assert sorted(_args(calls, "_tblFieldMaskToggle", 1)) == sorted(HOSTILE)
        assert _args(calls, "_tblMaskAll", 2) == [HOSTILE]
        assert _args(calls, "_tblMaskNone") == [self.FQN] and _args(calls, "_tblMaskNone", 1) == ["panel-1"]


class TestTheMaskingPanel:

    FQNS = ["DB.S.T1' ONMOUSEOVER='ALERT(1)", "DB.S.T2\\');ALERT(1)//"]

    def test_each_table_row_is_handed_its_table(self):
        html = dukpy.evaljs(
            DOM + "var selected = new Set(dukpy['fqns']); var IS_REGULATED = false;"
            + "function _updateMaskBadge() {}"
            + _code(SETUP, *SETUP_HELPERS, "function _refreshMaskingPanel(")
            + "_refreshMaskingPanel(); markup(els['mask-table-list']);",
            fqns=self.FQNS)
        page = parse(html)
        assert_nothing_injected(page, handlers={"onclick"})
        calls = handler_calls(page, ["_toggleMaskItem", "previewMasking"])
        assert sorted(_args(calls, "_toggleMaskItem")) == sorted(self.FQNS)
        assert sorted(_args(calls, "previewMasking")) == sorted(self.FQNS)

    def test_each_field_is_handed_its_table_and_name(self):
        fqn = self.FQNS[0]
        data = {"columns": [{"name": v, "type": "int"} for v in HOSTILE], "auto_masked": [],
                "strategy_map": {}}
        html = dukpy.evaljs(
            DOM + "var maskingConfig = new Map(); var STRATEGY_LABELS = {}; function _updateMaskBadge() {}"
            + _code(SETUP, *SETUP_HELPERS, "function _renderMaskFields(")
            + "var body = new El('div'); _renderMaskFields(dukpy['fqn'], body, dukpy['data']); body.innerHTML;",
            fqn=fqn, data=data)
        page = parse(html)
        assert_nothing_injected(page, handlers={"onchange"})
        calls = handler_calls(page, ["_maskSelectAll", "_toggleMaskField2"])
        assert _args(calls, "_maskSelectAll") == [fqn]
        assert sorted(_args(calls, "_toggleMaskField2", 1)) == sorted(HOSTILE)


def test_the_discovery_hint_shows_schema_and_table_names_as_text():
    tables = [f"DB.{name}.{name}" for name in TABLE_NAMES]
    html = dukpy.evaljs(
        DOM + _code(SETUP, *SETUP_HELPERS, "function _updateDiscoveryScopeHint(")
        + "_updateDiscoveryScopeHint(dukpy['tables']);"
        + "els['discovery-scope-hint'].innerHTML + els['discovery-file-list'].innerHTML;",
        tables=tables)
    page = parse(html)
    assert_nothing_injected(page)
    shown = "".join(page.text)
    for name in TABLE_NAMES:
        assert name.upper() in shown and name in shown


# ── The databases page ───────────────────────────────────────────────────────

DB_HELPERS = ("function dbScopeEsc(", "function dbScopeJsArg(", "function dbScopeRef(",
              "function dbScopeId(", "function row(")


class TestTheDatabaseScopeTree:

    def test_each_checkbox_is_handed_its_table(self):
        tree = {"DB": {"SALES": {"tables": TABLE_NAMES, "views": ["V' onmouseover='x"]}}}
        html = dukpy.evaljs(
            DOM + "var dbScopeSelected = new Set(); function dbScopeSync() {}"
            + _code(DATABASES, *DB_HELPERS, "function dbScopeRender(")
            + "dbScopeRender(dukpy['tree']); markup(els['db-scope-tree']);",
            tree=tree)
        page = parse(html)
        assert_nothing_injected(page, handlers={"onchange", "oninput"})
        calls = handler_calls(page, ["dbScopeCheck", "dbScopeToggleSchema", "dbScopeFilter"])
        refs = sorted(f"DB.SALES.{t}".upper() for t in TABLE_NAMES)
        assert sorted(_args(calls, "dbScopeCheck")) == refs
        assert [sorted(r) for r in _args(calls, "dbScopeToggleSchema")] == [refs]


class TestTheLogDiagnosis:

    def _page(self, d):
        return parse(dukpy.evaljs(DOM + _code(DATABASES, *DB_HELPERS, "function diagnosisHtml(")
                                  + "diagnosisHtml(dukpy['d']);", d=d))

    @pytest.mark.parametrize("d", [
        {"error": HOSTILE[2]},
        {"error": HOSTILE[2], "local_egress_count": 0, "external_table_exists": True,
         "missing_columns": [HOSTILE[0], HOSTILE[2]], "local_egress_max_id": HOSTILE[2]},
    ])
    def test_the_servers_text_stays_text(self, d):
        page = self._page(d)
        assert_nothing_injected(page)
        assert HOSTILE[2] in "".join(page.text)


# ── The relationships page ───────────────────────────────────────────────────

GRAPH_HELPERS = ("function _esc(", "function _jsArg(", "function _parseJC(")


class TestTheRelationshipsPage:

    ENTITY = "SALES' onmouseover='alert(1)"
    PARTNER = 'STORE" onmouseover="alert(1)" O\'Brien \\'

    def _joins(self, relationships):
        html = dukpy.evaljs(
            DOM + "var graphData = {relationships: dukpy['rels'], entities: []};"
            + "function _buildJoinEditForm() { return ''; }"
            + _code(GRAPH, *GRAPH_HELPERS, "function renderJoinCard(", "function renderJoinsTab(")
            + "renderJoinsTab(dukpy['entity']); els['bp-content'].innerHTML;",
            rels=relationships, entity=self.ENTITY)
        return parse(html)

    def test_the_join_list_hands_each_name_to_its_handler(self):
        rels = [{"id": 7, "from_entity": self.ENTITY, "to_entity": self.PARTNER, "join_type": "left",
                 "from_column": HOSTILE[2], "to_column": HOSTILE[0]}]
        page = self._joins(rels)
        assert_nothing_injected(page, handlers={"onclick"})
        calls = handler_calls(page, ["toggleJoinCard", "focusNode", "openAddJoinModal"])
        assert _args(calls, "focusNode") == [self.PARTNER]
        assert _args(calls, "openAddJoinModal") == [self.ENTITY]
        assert _args(calls, "toggleJoinCard") == [7]

    def test_an_entity_with_no_joins_offers_the_first_one(self):
        page = self._joins([])
        assert_nothing_injected(page, handlers={"onclick"})
        assert _args(handler_calls(page, ["openAddJoinModal"]), "openAddJoinModal") == [self.ENTITY]

    def test_the_sidebar_hands_each_entity_to_its_type_switch(self):
        entities = [{"entity_name": name, "entity_type": "fact", "schema_name": "S", "table_name": name}
                    for name in (self.ENTITY, self.PARTNER)]
        html = dukpy.evaljs(
            DOM + "var graphData = {entities: dukpy['entities'], relationships: []};"
            + "var activeTypeFilter = ''; var _selectedEntity = '';"
            + "var TYPE_COLORS = {dimension: '#000', fact: '#111'}; var TYPE_LABELS = {fact: 'Fact'};"
            + "var cy = {getElementById: function() { return {addClass: function(){}, removeClass: function(){}}; }};"
            + _code(GRAPH, *GRAPH_HELPERS, "function buildSidebar(")
            + "buildSidebar(); markup(els['sb-list']);",
            entities=entities)
        page = parse(html)
        assert_nothing_injected(page, handlers={"onclick"})
        assert sorted(_args(handler_calls(page, ["cycleEntityType"]), "cycleEntityType")) == sorted(
            [self.ENTITY, self.PARTNER])


# ── The metrics page ─────────────────────────────────────────────────────────

METRIC_HELPERS = ("function escHtml(", "function escJsArg(", "function _mcTypeCls(", "function _mcTypeIcon(",
                  "function _displayLabel(")


class TestTheMetricPickers:

    def test_each_saved_metric_inserts_its_own_name(self):
        metrics = [{"name": v, "description": v, "result_format": "number"} for v in HOSTILE]
        html = dukpy.evaljs(
            DOM + "window._metricsData = dukpy['metrics'];"
            + _code(METRICS, *METRIC_HELPERS, "window._buildMcMetricsList = function(")
            + "window._buildMcMetricsList(); els['mc-metrics-body'].innerHTML;",
            metrics=metrics)
        page = parse(html)
        assert_nothing_injected(page, handlers={"onclick"})
        assert sorted(_args(handler_calls(page, ["insertMetricRef"]), "insertMetricRef")) == sorted(HOSTILE)

    def test_each_field_inserts_its_own_name(self):
        columns = [{"column": v, "table": "T' x", "type": "varchar"} for v in HOSTILE]
        html = dukpy.evaljs(
            DOM + _code(METRICS, *METRIC_HELPERS, "window._buildMcFieldsList = function(")
            + "window._buildMcFieldsList(dukpy['columns']); els['mc-fields-list'].innerHTML;",
            columns=columns)
        page = parse(html)
        assert_nothing_injected(page, handlers={"onclick"})
        calls = handler_calls(page, ["insertColumnFromMcDialog", "toggleMcFieldGroup"])
        assert sorted(_args(calls, "insertColumnFromMcDialog")) == sorted(HOSTILE)

    def test_column_suggestions_show_names_and_types_as_text(self):
        columns = [{"column": "ab" + v, "table": v, "type": v} for v in HOSTILE]
        html = dukpy.evaljs(
            DOM + "var _allColumns = dukpy['columns']; var _dupCols = new Set(); var _suggestTarget, _suggestIdx;"
            + "function _mountSuggestionTray() {}"
            + _code(METRICS, *METRIC_HELPERS, "function _showTagSuggestions(")
            + "_showTagSuggestions({}, 'ab'); els['col-suggest'].innerHTML;",
            columns=columns)
        page = parse(html)
        assert_nothing_injected(page)
        cols = [a["data-col"] for _t, a in page.elements if "data-col" in a]
        assert sorted(cols) == sorted("ab" + v for v in HOSTILE)


# ── The system page ──────────────────────────────────────────────────────────

class TestTheDeploymentList:

    def test_each_button_selects_its_own_deployment(self):
        deploys = [{"name": v, "model": v, "status": v} for v in HOSTILE]
        html = dukpy.evaljs(
            DOM + "var _selectedQuery = ''; var _selectedKb = '';"
            + _code(SYSTEM, "function escHtml(", "function renderDeployments(")
            + "renderDeployments(dukpy['deploys']); markup(els['az-deploy-list']) + markup(els['az-deploy-cards']);",
            deploys=deploys)
        page = parse(html)
        assert_nothing_injected(page, handlers={"onclick"})
        calls = handler_calls(page, ["selectDeployment"])
        assert sorted(_args(calls, "selectDeployment", 1)) == sorted(HOSTILE * 2)


# ── Chart tooltips ───────────────────────────────────────────────────────────

def test_a_tooltip_shows_its_value_as_text():
    html = dukpy.evaljs(
        _code(CHARTS, "function escHtml(", "function tipRow(")
        + "tipRow('#000', dukpy['name'], dukpy['value'], {ink: '#000', ink2: '#333'}, 'swatch');",
        name=HOSTILE[0], value=HOSTILE[2])
    page = parse(html)
    assert_nothing_injected(page)
    assert HOSTILE[2] in "".join(page.text) and HOSTILE[0] in "".join(page.text)
