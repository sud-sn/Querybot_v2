"""What a reader asked for after the first answers on the new core.

* A share worked out by the query ("3% of the total") was in the table only: the chart drew
  the amounts alone. Each bar now says its share beside its value, quieter, and its
  tooltip says it too; the axis keeps one unit.
* In compare mode the new core's answer had no Add to dashboard: only the new-core-only
  mode gave it a pin token.
* Answer tables: a number column's heading sits over its numbers (right), every other
  row is lightly shaded (still even after a filter), and the row under the pointer is
  lifted with a quick wash; the same in the follow-up result card and a dashboard tile.
* A reader's own word or abbreviation ("sales" for Revenue, "prd qty" for Product
  quantity) is read without the AI and handed to it beside the question.

Synthetic data only (the invented retail warehouse).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import re
from pathlib import Path

import pytest

from evals.core2 import domains
from evals.core2.compile_eval import learn

ROOT = Path(__file__).resolve().parents[1]
TODAY = dt.date(2026, 6, 15)


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")


def _answer(retail, plan: dict) -> dict:
    from core2.answer.builder import build_answer
    from core2.compile.compiler import compile_query
    from core2.plan.ir import Plan
    from core2.resolve.resolver import Context, resolve
    from core2.warehouse.runner import DuckDBWarehouse

    built, model = retail
    warehouse = DuckDBWarehouse(built.con)
    logical = resolve(Plan.model_validate({"kind": "query", **plan}), model, Context(today=TODAY))
    compiled = compile_query(logical, model, warehouse.dialect)
    result = warehouse.query(compiled.sql)
    return build_answer("q", logical, compiled, result.columns, result.rows)


H1 = {"window": {"kind": "between", "start": "2026-01-01", "end": "2026-06-30"}}


# ── the share on the chart ───────────────────────────────────────────────────

def test_the_share_the_query_worked_out_travels_with_the_chart(retail):
    payload = _answer(retail, {"intent": "share", "measures": ["net_amount"], "group_by": ["region"], "time": H1})
    chart = payload["chart"]
    share = chart["share_key"]
    assert share and chart["column_roles"][share] == {"column": share, "label": chart["column_roles"][share]["label"],
                                                      "role": "share", "format": "percentage"}
    assert share not in chart["y_keys"]                    # never drawn as a second amount
    shares = [r[share] for r in chart["rows"]]
    assert sum(shares) == pytest.approx(100, abs=0.05)     # percents, as the table shows them
    table = {r[chart["x_key"]]: r for r in payload["data"]["rows"]}
    first = chart["rows"][0]
    assert first[share] == pytest.approx(table[first[chart["x_key"]]][share])


def test_a_chart_without_a_share_carries_none(retail):
    chart = _answer(retail, {"measures": ["net_amount", "quantity"], "group_by": ["region"], "time": H1})["chart"]
    assert chart.get("share_key") is None


def _drawn(payload: dict, expression: str):
    from tests.test_chart_annotation_language import _build

    return json.loads(_build("portal_chat.html", "en", payload, f"JSON.stringify({expression})"))


SHARED = {"rows": [{"CUSTOMER": "Rideau Valley Trading 20", "REVENUE": 1327838.92, "SHARE": 2.99},
                   {"CUSTOMER": "Prairie Mechanical 40", "REVENUE": 1320180.5, "SHARE": 2.97},
                   {"CUSTOMER": "Sunset Coast Hardware 60", "REVENUE": 1319193.44, "SHARE": 2.97}],
          "x_key": "CUSTOMER", "y_keys": ["REVENUE"], "chart_type": "bar", "share_key": "SHARE",
          "column_formats": {"REVENUE": "currency", "SHARE": "percentage"},
          "column_roles": {"CUSTOMER": {"column": "CUSTOMER", "role": "dimension", "label": "Customer"},
                           "REVENUE": {"column": "REVENUE", "role": "measure", "label": "Revenue", "format": "currency"},
                           "SHARE": {"column": "SHARE", "role": "share", "label": "Share of revenue",
                                     "format": "percentage"}},
          "chart_spec": {"x": {"column": "CUSTOMER", "role": "dimension"}}}


def test_each_bar_says_its_share_beside_its_value_and_in_its_tooltip():
    drawn = _drawn(SHARED, "{label: opt.series[0].label.formatter({value: 1327838.92, dataIndex: 0}),"
                           " series: opt.series.length, rich: Object.keys(opt.series[0].label.rich || {}),"
                           " tip: opt.tooltip.formatter({name: 'Rideau Valley Trading 20', value: 1327838.92,"
                           " dataIndex: 0, color: '#2a78d6'})}")
    assert drawn["series"] == 1                            # one measure, one axis
    assert re.fullmatch(r"\{v\|\$1\.3M\}\{s\|  ·  3(\.0)?\s?%\}", drawn["label"]), drawn["label"]
    assert drawn["rich"] == ["v", "s"]
    assert "Share of revenue" in drawn["tip"] and re.search(r"2\.99\s?%", drawn["tip"])


def test_on_a_narrow_card_the_share_is_in_the_tooltip_alone():
    # A phone: the bars would be squeezed to make room for the longer label.
    drawn = _drawn(SHARED, "(function () { var o = QBCharts.buildOption(" + json.dumps(SHARED) + ", {width: 340, height: 400});"
                           " return {label: o.series[0].label.formatter({value: 1327838.92, dataIndex: 0}),"
                           " tip: o.tooltip.formatter({name: 'x', value: 1327838.92, dataIndex: 0, color: '#2a78d6'})}; })()")
    assert drawn["label"] == "$1.3M" and "Share of revenue" in drawn["tip"]


def test_a_column_says_its_share_under_its_value():
    # Periods stand as columns: the share goes under the value, not beside it.
    payload = {**SHARED, "rows": [{"CUSTOMER": f"Q{i + 1} 2026", "REVENUE": 100.0 - i, "SHARE": 25.0} for i in range(4)],
               "chart_spec": {"x": {"column": "CUSTOMER", "role": "temporal"}}}
    drawn = _drawn(payload, "{axis: opt.yAxis.type, label: opt.series[0].label.formatter({value: 100, dataIndex: 0})}")
    assert drawn["axis"] == "value" and re.fullmatch(r"\{v\|\$100\}\n\{s\|25\s?%\}", drawn["label"]), drawn


def test_without_a_share_the_label_is_the_value_alone():
    payload = {k: v for k, v in SHARED.items() if k != "share_key"}
    label = _drawn(payload, "opt.series[0].label.formatter({value: 1327838.92, dataIndex: 0})")
    assert label == "$1.3M"


# ── Add to dashboard in compare mode ─────────────────────────────────────────

def test_the_compare_mode_answer_can_be_added_to_a_dashboard(monkeypatch):
    import gateway.core2_bridge as bridge

    answer = {"type": "assistant_response", "engine": "core2", "question": "q",
              "answer": {"headline": "Revenue by region."}, "result_scope": {"badge": "", "note": ""},
              "chart": {"chart_type": "bar", "title": "Revenue by region"}, "kpi": None,
              "data": {"headers": ["r"], "rows": [{"r": 1}], "total_rows": 1},
              "plan": {"kind": "query", "measures": ["net_amount"]},
              "trust": {"engine": "core2", "sql": "SELECT 1 AS r", "row_count": 1}}

    async def answered(*args, **kwargs):
        return "ok", json.loads(json.dumps(answer))

    made: list[tuple] = []
    monkeypatch.setattr(bridge, "_answer", answered)
    monkeypatch.setattr(bridge, "_pin", lambda account, user, question, payload: made.append((account, question)) or "tok")
    sent: list[dict] = []

    class Adapter:
        thread_id = "t1"
        send_lock = asyncio.Lock()

    class Socket:
        async def send_json(self, payload):
            sent.append(payload)

    asyncio.run(bridge.answer_beside(Adapter(), Socket(), "acct-x", "revenue by region", {"id": 1}))
    (card,) = sent
    assert card["pin_token"] == "tok" and card["chart"]["pin_token"] == "tok"
    assert card["result_scope"]["badge"] == bridge.PREVIEW_BADGE and made == [("acct-x", "revenue by region")]


# ── answer tables ────────────────────────────────────────────────────────────

_TABLE_STUBS = """
var window = {qbNum: function (n) { return String(n); }};
function escHtml(s) { return String(s == null ? '' : s); }
function _normaliseColumnKey(k) { return String(k || '').toLowerCase(); }
function _columnFormatMap(data) {
  var m = new Map(); var f = (data && data.column_formats) || {};
  Object.keys(f).forEach(function (k) { m.set(k.toLowerCase(), f[k]); }); return m;
}
function _displayFormatSpec() { return {}; }
function _formatDisplayValue(v) { return String(v == null ? '' : v); }
function _parseServerNumber(v) { return Number(v); }
function t(id) { return id; }
function plural(id, n, v) { return String(n); }
function qbIcon() { return ''; }
var _dtIdCounter = 0;
"""


def _table(data: dict) -> str:
    import dukpy

    from tests.js_lift import function as lift

    page = (ROOT / "portal" / "templates" / "portal_chat.html").read_text(encoding="utf-8")
    return dukpy.evaljs("\n".join([_TABLE_STUBS, lift(page, "function _isANumericColumn(rows, header)"),
                                   lift(page, "function _isNumericColumn(rows, header, format)"),
                                   lift(page, "function renderDataTable(data,"), f"renderDataTable({json.dumps(data)})"]))


def test_a_number_columns_heading_sits_over_its_numbers():
    html = _table({"headers": ["CUSTOMER", "REVENUE", "SHARE", "POSTCODE"],
                   "column_formats": {"REVENUE": "currency", "SHARE": "percentage", "POSTCODE": "text"},
                   "rows": [{"CUSTOMER": "A", "REVENUE": 10, "SHARE": 2.5, "POSTCODE": "02134"},
                            {"CUSTOMER": "B", "REVENUE": 8, "SHARE": 2.0, "POSTCODE": "10001"}]})
    heads = re.findall(r"<th data-col=\"\d\"( class=\"num\")?[^>]*>([^<]+)</th>", html)
    assert [(bool(num), name) for num, name in heads] == [(False, "CUSTOMER"), (True, "REVENUE"), (True, "SHARE"),
                                                         (False, "POSTCODE")]
    first = html.split("<tbody>")[1].split("</tr>")[0]
    assert re.findall(r"<td( class=\"num\")?", first) == ["", ' class="num"', ' class="num"', ""]


def test_rows_are_banded_and_lift_under_the_pointer():
    css = (ROOT / "portal" / "templates" / "portal_chat.html").read_text(encoding="utf-8")
    assert ".data-table tbody tr:nth-child(even of :not(.dt-hidden)) td{background:" in css   # even after a filter
    assert ".rc-table tbody tr:nth-child(even) td{background:" in css
    assert re.search(r"\.data-table tbody td,\.rc-table tbody td\{transition:background-color \.16s", css)
    assert ".data-table tbody tr:hover td:first-child,.rc-table tbody tr:hover td:first-child{box-shadow:inset 3px 0 0 var(--primary)}" in css
    assert "@media (prefers-reduced-motion: reduce){.data-table tbody td,.rc-table tbody td{transition:none}}" in css
    dashboard = (ROOT / "static" / "css" / "dashboard.css").read_text(encoding="utf-8")
    assert ".dash-table tbody tr:nth-child(even) td" in dashboard and ".dash-table th.num { text-align: right; }" in dashboard


def test_a_dashboard_table_tile_aligns_its_amounts():
    from decimal import Decimal

    from portal.routes import _numeric_columns

    rows = [{"CUSTOMER": "A", "REVENUE": Decimal("10.5"), "ORDERS": 3, "WEEK": "202530", "FLAG": True},
            {"CUSTOMER": "B", "REVENUE": None, "ORDERS": 4, "WEEK": "202531", "FLAG": False}]
    assert _numeric_columns(rows, list(rows[0]), {}) == ["REVENUE", "ORDERS"]
    assert _numeric_columns(rows, list(rows[0]), {"ORDERS": "text", "WEEK": "number"}) == ["REVENUE", "WEEK"]
    from jinja2 import Environment, FileSystemLoader

    template = (ROOT / "portal" / "templates" / "portal_dashboard.html").read_text(encoding="utf-8")
    block = template[template.index("{% elif chart.table_rows %}"):template.index("{% elif chart.chart_json %}")]
    html = Environment(loader=FileSystemLoader(str(ROOT / "portal" / "templates"))).from_string(
        block.replace("{% elif chart.table_rows %}", "", 1)).render(chart={
            "table_columns": ["CUSTOMER", "REVENUE"], "table_numeric": ["REVENUE"],
            "table_rows": [{"CUSTOMER": {"d": "A", "v": "A"}, "REVENUE": {"d": "$10.50", "v": 10.5}}]})
    assert '<th class="num">REVENUE</th>' in html and '<td data-sort="10.5" class="num">$10.50</td>' in html
    assert 'class="dash-table"' in html and "style=" not in html


# ── the reader's own words ───────────────────────────────────────────────────

@pytest.fixture(scope="module")
def named(retail):
    """Revenue and Product quantity, as a workspace names them, with no other words for them."""
    _, model = retail
    model = model.model_copy(deep=True)
    by = {m.slug: m for m in model.measures.values()}
    by["net_amount"].business_name, by["net_amount"].synonyms = "Revenue", {}
    by["quantity"].business_name, by["quantity"].synonyms = "Product quantity", {}
    return model


@pytest.mark.parametrize("question,said,slug", [
    ("total sales by customer", "sales", "net_amount"),
    ("prd qty by store last month", "prd qty", "quantity"),
    ("disc amt by store", "disc amt", "discount_amount"),
    ("cust seg breakdown of revenue", "cust seg", "customer.segment"),
    ("how many clients bought in Q2", "clients", "customer"),
    ("rev by region", "rev", "net_amount"),                 # an abbreviation alone, as the list reads it
    ("revenue by seg", "seg", "customer.segment"),
    ("units by store", "units", "quantity_alone"),
])
def test_a_readers_word_or_abbreviation_names_what_it_means(named, question, said, slug):
    from core2.plan.words import readings

    model = named
    if slug == "quantity_alone":          # a one-word name: "units" is an everyday word for quantity
        model = named.model_copy(deep=True)
        next(m for m in model.measures.values() if m.slug == "quantity").business_name = "Quantity"
        slug = "quantity"
    assert (said, slug) in [(r.said, r.slug) for r in readings(model, question)]


@pytest.mark.parametrize("question", ["Revenue by customer", "top 10 customers by revenue", "amt by region",
                                      "orders shipped last week"])
def test_a_name_as_written_or_a_word_too_vague_reads_as_nothing(named, question):
    from core2.plan.words import readings

    assert readings(named, question) == []


def test_one_word_is_never_read_as_the_start_of_a_longer_one(named):
    from core2.plan.words import readings

    model = named.model_copy(deep=True)
    segment = model.attributes["customer.segment"]
    segment.business_name = "Salesperson"
    found = readings(model, "sales by salesperson")
    assert [(r.said, r.slug) for r in found] == [("sales", "net_amount")]
    segment.business_name = "Shipment"
    assert readings(model, "revenue by ship") == []
    # Two words or more may be shortenings the abbreviation list does not hold.
    segment.business_name = "Shipment method"
    assert [(r.said, r.slug) for r in readings(model, "revenue by ship meth")] == [("ship meth", "customer.segment")]


def test_the_planner_is_handed_the_readings_beside_the_question(named):
    from core2.plan.planner import plan_question

    prompts: list[str] = []

    def complete(system: str, user: str) -> str:
        prompts.append(user)
        return json.dumps({"kind": "query", "measures": ["quantity"], "group_by": ["store"]})

    outcome = plan_question(named, "prd qty by store", complete, today=TODAY)
    assert outcome.plan.measures == ["quantity"]
    assert 'READER\'S WORDS' in prompts[0] and '- "prd qty" -> measure quantity (Product quantity)' in prompts[0]
    assert prompts[0].index("READER'S WORDS") < prompts[0].index("QUESTION: prd qty by store")


def test_with_values_withheld_a_member_name_never_shows_through_a_reading(named):
    from core2.plan.planner import plan_question
    from core2.plan.values import ValueMatch

    model = named.model_copy(deep=True)
    prompts: list[str] = []

    def complete(system: str, user: str) -> str:
        prompts.append(user)
        return json.dumps({"kind": "query", "measures": ["net_amount"]})

    question = "net amount for Sales Depot"       # a member whose name holds an everyday word
    match = ValueMatch("Sales Depot", question.index("Sales"), len(question), "store", "Sales Depot")
    plan_question(model, question, complete, today=TODAY, matches=[match], values_allowed=False)
    assert "Sales Depot" not in prompts[0] and '"Sales' not in prompts[0]
