# -*- coding: utf-8 -*-
"""The suggested questions on a new thread are this workspace's, and they work.

Found in a live test ("when we have clicked the new thread, more questions are
not working"), then replayed in a browser:

  * A reader who comes back after half an hour -- or clicks New Thread then --
    is greeted on connecting. The greeting hid the suggested questions under
    it, the clickable ones, and listed the product's own examples instead:
    "What is our total revenue this month?", "Show top 10 customers by sales",
    "How many orders were created last week?". A stock or a finance workspace
    cannot answer any of them, so the reader tried them and they failed.
  * Refresh ("Show different suggestions") reshuffled the four questions on
    screen. It never showed a different one: the page was given six and drew
    four.
  * A reader with no table access was offered the whole workspace's
    questions: the empty set of tables they may read was taken for "no
    restriction".

The socket and the page are the real ones; the page's own functions run in a
JavaScript engine against a DOM thin enough for them. Synthetic workspace; no
customer data.
"""

from __future__ import annotations

import json
import os
import re
import tempfile

# The store is conftest's scratch database for the run (tests/conftest.py).
import store  # noqa: E402
from tests.chat_js import source as chat_page_source
from tests.js_lift import function as lift

PRODUCT_EXAMPLES = ("revenue this month", "top 10 customers", "orders were created")
METRICS = (("Net sales", "SUM(NET_AMT)"), ("Units sold", "SUM(QTY)"), ("Gross margin", "SUM(GRS_MRG_AMT)"),
           ("Discount amount", "SUM(DSC_AMT)"), ("Freight cost", "SUM(FRT_AMT)"), ("Tax amount", "SUM(TAX_AMT)"),
           ("Returns amount", "SUM(RTN_AMT)"))


def _table(own_key: str, *columns: tuple[str, str]) -> dict:
    return {
        "columns": [{"name": n, "type": t, "nullable": False, "comment": ""} for n, t in columns],
        "pk_columns": [own_key], "row_count": 1000, "comment": "", "schema": "MART", "database": "WH",
    }


SCHEMA = {
    "WH.MART.SLS_FCT": _table(
        "SLS_FCT_KEY", ("SLS_FCT_KEY", "bigint"), ("RGN_DMS_KEY", "int"), ("NET_AMT", "decimal(18,2)"),
        ("QTY", "int"), ("GRS_MRG_AMT", "decimal(18,2)"), ("DSC_AMT", "decimal(18,2)"),
        ("FRT_AMT", "decimal(18,2)"), ("TAX_AMT", "decimal(18,2)"), ("RTN_AMT", "decimal(18,2)")),
    "WH.MART.RGN_DMS": _table("RGN_DMS_KEY", ("RGN_DMS_KEY", "int"), ("RGN_NM", "varchar(40)")),
}


def _workspace(metrics=(), *, tables=("MART.SLS_FCT", "MART.RGN_DMS")) -> tuple[str, int]:
    from core.graph_autopopulate import auto_populate_from_schema
    from core.semantic_contract import write_contract
    from core.semantic_model import write_semantic_model

    store.init_db()
    account = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account, "Test Ltd")
    store.update_client_meta(account, chat_ui_enabled=1)
    user_id, _ = store.create_user(account, "Ada", f"{os.urandom(4).hex()}@x.com", password="a-password-they-chose")
    group_id = store.create_group(account, "Analysts")
    store.set_group_tables(group_id, account, list(tables))
    store.update_user(user_id, group_id=group_id)
    kb = tempfile.mkdtemp()
    with open(os.path.join(kb, "_schema.json"), "w", encoding="utf-8") as handle:
        json.dump(SCHEMA, handle)
    store.update_client_state(account, "READY", {"schema_dir": kb, "kb_dir": kb})
    auto_populate_from_schema(account, kb)
    write_semantic_model(schema_dir=kb, kb_dir=kb, account_id=account)
    for name, template in metrics:
        store.save_metric(account, {
            "name": name, "formula_type": "expression", "sql_template": template,
            "base_table": "MART.SLS_FCT", "synonyms": name.lower(), "description": name,
        })
    write_contract(account, kb)
    return account, user_id


def _client(user_id: int):
    import main
    import portal.routes as pr
    from starlette.testclient import TestClient

    client = TestClient(main.app)
    client.cookies.set(pr._COOKIE, pr._sign_session_value(user_id))
    return client


def _greeting(account: str, user_id: int) -> dict:
    """The first frame a reader gets on connecting after time away."""
    with _client(user_id).websocket_connect(f"/ws/chat/{account}?thread_id=new1") as ws:
        return ws.receive_json()


class TestTheGreeting:

    def test_it_is_marked_so_the_page_keeps_the_questions_under_it(self):
        account, user_id = _workspace(METRICS[:1])
        frame = _greeting(account, user_id)
        assert frame["type"] == "message" and frame["greeting"] is True

    def test_it_offers_this_workspaces_questions(self):
        account, user_id = _workspace(METRICS[:1])
        content = _greeting(account, user_id)["content"]
        assert "What is our total Net sales?" in content
        assert not any(example in content for example in PRODUCT_EXAMPLES), content

    def test_a_workspace_with_no_questions_yet_is_offered_none_of_the_products(self):
        account, user_id = _workspace(())
        content = _greeting(account, user_id)["content"]
        assert not any(example in content for example in PRODUCT_EXAMPLES), content
        assert "For example" not in content and "`help`" in content


def _chips(html: str) -> list[dict]:
    panel = html[html.index('id="suggestions"'):html.index("</div>", html.index('id="suggestions"'))]
    return [{"question": question, "hidden": "display:none" in attrs}
            for attrs, question in re.findall(
                r'<button class="suggestion-chip"(.*?)data-question="([^"]*)"', panel, re.S)]


def _more_buttons(html: str) -> list[bool]:
    return [" hidden" in tag for tag in re.findall(r"<button[^>]*data-suggestions-more[^>]*>", html)]


class TestTheSuggestedQuestionsOnThePage:

    def test_every_question_is_on_the_page_four_at_a_time(self):
        account, user_id = _workspace(METRICS)
        html = _client(user_id).get("/portal/chat?new=1").text
        chips = _chips(html)
        assert len(chips) == len(METRICS), chips
        assert [chip["hidden"] for chip in chips] == [False] * 4 + [True] * (len(METRICS) - 4)
        assert _more_buttons(html) == [False, False]

    def test_refresh_is_not_offered_when_there_is_nothing_else_to_show(self):
        account, user_id = _workspace(METRICS[:2])
        html = _client(user_id).get("/portal/chat?new=1").text
        assert len(_chips(html)) == 2
        assert _more_buttons(html) == [True, True]

    def test_a_reader_with_no_table_access_is_offered_nothing(self):
        from portal.routes import _build_chat_suggestions

        account, user_id = _workspace(METRICS, tables=())
        reader = store.get_user(user_id)
        assert store.get_allowed_tables(reader) == set()
        assert _build_chat_suggestions(reader) == []


class TestAMetricQuestionReachesTheReadersWhoCanAskIt:
    """core.suggestions offered a metric's question to admins only: a
    registry metric is an expression whose table is in base_table, and the
    check looked for tables in its SQL. The greeting and the workspace guide
    take their examples from this list, so for every other reader they had
    none -- which is when the greeting fell back to the product's."""

    def _questions(self, account, tables):
        from core.suggestions import get_suggestions

        state = store.get_client(account) or {}
        dirs = json.loads(state.get("state_data") or "{}")
        return [item["question"] for item in get_suggestions(
            account, dirs["kb_dir"], tables, n=12, schema_dir=dirs["schema_dir"])]

    def test_a_reader_who_can_read_its_table_is_offered_it(self):
        account, _ = _workspace(METRICS[:1])
        for tables in ({"MART.SLS_FCT", "MART.RGN_DMS"}, {"SLS_FCT"}, None):
            assert self._questions(account, tables) == ["What is our total Net sales?"], tables

    def test_a_reader_who_cannot_is_not(self):
        account, _ = _workspace(METRICS[:1])
        for tables in ({"MART.RGN_DMS"}, set()):
            assert self._questions(account, tables) == [], tables


PAGE_FUNCTIONS = (
    "function hideWelcome(opts = {})",
    "function _suggestionInSchema(chip, schemaName)",
    "function _showSuggestionPage(schemaName = _activeSchema)",
    "function _refreshSuggestionsForSchema(schemaName)",
    "function shuffleSuggestions()",
    "function _visibleSuggestionTexts()",
)


def _page(script: str, *, chips: list[str | tuple[str, str]], schema: str = "") -> dict:
    """Run the chat page's own functions over a panel of `chips`.

    A chip is a question, or (question, fqn). The first four start visible,
    as the server renders them.
    """
    import dukpy

    src = chat_page_source()
    per_page = re.search(r"const SUGGESTIONS_PER_PAGE = (\d+);", src).group(1)
    panel = [chip if isinstance(chip, tuple) else (chip, "") for chip in chips]
    harness = f"""
function _el(extra) {{ return Object.assign({{style: {{display: ''}}, hidden: false, dataset: {{}}}}, extra || {{}}); }}
const _chips = {json.dumps(panel)}.map((c, i) => _el({{
  dataset: {{question: c[0], fqn: c[1]}}, textContent: c[0], style: {{display: i < 4 ? '' : 'none'}}}}));
const _more = [_el({{hidden: _chips.length <= 4}}), _el({{hidden: _chips.length <= 4}})];
const _nodes = {{welcomeState: _el(), suggestionsWrap: _el(), suggestions: _el()}};
var document = {{
  getElementById: id => _nodes[id] || null,
  querySelectorAll: sel => sel.indexOf('suggestion-chip') >= 0 ? _chips
                         : sel === '[data-suggestions-more]' ? _more : [],
}};
function setTimeout(fn) {{ fn(); return 0; }}
const SUGGESTIONS_PER_PAGE = {per_page};
let _suggestionPage = 0;
let _activeSchema = {json.dumps(schema)};
{chr(10).join(lift(src, signature) for signature in PAGE_FUNCTIONS)}
const _shown = () => _chips.filter(c => c.style.display !== 'none').map(c => c.dataset.question);
{script}
"""
    return json.loads(dukpy.evaljs(harness))


TEN = [f"Question {n}?" for n in range(1, 11)]


class TestThePageShowsDifferentQuestions:

    def test_refresh_steps_through_every_question_and_back(self):
        pages = _page("""
            const seen = [_shown()];
            for (let i = 0; i < 3; i++) { shuffleSuggestions(); seen.push(_shown()); }
            JSON.stringify(seen);""", chips=TEN)
        assert pages == [TEN[0:4], TEN[4:8], TEN[8:10], TEN[0:4]]

    def test_refresh_keeps_to_the_chosen_schema(self):
        chips = [("Sales A?", "WH.MART.SLS"), ("Stock A?", "WH.INV.STK"), ("Any?", ""),
                 ("Sales B?", "WH.MART.ORD"), ("Stock B?", "WH.INV.BAL"), ("Sales C?", "WH.MART.RTN"),
                 ("Sales D?", "WH.MART.PAY")]
        pages = _page("""
            _refreshSuggestionsForSchema('MART');
            const seen = [_shown(), _more.map(b => b.hidden)];
            shuffleSuggestions(); seen.push(_shown());
            JSON.stringify(seen);""", chips=chips, schema="MART")
        assert pages == [["Sales A?", "Any?", "Sales B?", "Sales C?"], [False, False], ["Sales D?"]]

    def test_refresh_is_withdrawn_when_the_schema_leaves_nothing_else(self):
        chips = [("Stock A?", "WH.INV.STK"), ("Sales A?", "WH.MART.SLS"), ("Any?", ""),
                 ("Stock B?", "WH.INV.BAL"), ("Sales B?", "WH.MART.ORD"), ("Sales C?", "WH.MART.RTN")]
        state = _page("""
            _refreshSuggestionsForSchema('INV');
            const inv = [_shown(), _more.map(b => b.hidden)];
            _refreshSuggestionsForSchema('');
            JSON.stringify([inv, [_shown(), _more.map(b => b.hidden)]]);""", chips=chips, schema="INV")
        assert state == [[["Stock A?", "Any?", "Stock B?"], [True, True]],
                         [["Stock A?", "Sales A?", "Any?", "Stock B?"], [False, False]]]

    def test_the_greeting_keeps_the_questions_and_an_answer_clears_them(self):
        state = _page("""
            hideWelcome({keepSuggestions: true});
            const afterGreeting = _nodes.suggestionsWrap.style.display;
            hideWelcome();
            JSON.stringify([afterGreeting, _nodes.suggestionsWrap.style.display,
                            _nodes.welcomeState.style.display]);""", chips=TEN)
        assert state == ["", "none", "none"]

    def test_only_the_questions_on_screen_count_as_seen(self):
        """The behavioural ranker marks the chips a reader passed over as
        dismissed. A chip on a page they never opened was not passed over."""
        shown = _page("JSON.stringify(_visibleSuggestionTexts());", chips=TEN)
        assert shown == TEN[:4]


def _frames_through_the_socket_handler(frames: list[dict]) -> list[dict]:
    """Feed frames to the handler the page's own connect() installs, and
    report what each one asked appendBot for. Everything the handler calls
    besides appendBot is a recorder."""
    import dukpy

    src = chat_page_source()
    harness = f"""
var location = {{protocol: 'https:', host: 'portal.example'}};
let _socket = null;
function WebSocket(url) {{ this.url = url; _socket = this; }}
const ACCOUNT_ID = 'acct', THREAD_ID = 'thread-1', DASHBOARD_ID = 0;
let reconnectAttempts = 0, wsReady = false, agentRunState = 'idle', _pendingSuggestion = null;
function t(id) {{ return id; }}
function setConnectionState() {{}}
function safeJsonParse(text) {{ try {{ return JSON.parse(text); }} catch (e) {{ return null; }} }}
const _appended = [];
function appendBot(text, opts) {{ _appended.push({{text: text, keepSuggestions: !!(opts || {{}}).keepSuggestions}}); }}
function appendSystem() {{}}
function setAgentRunState() {{ return true; }}
function _markActiveOutbound() {{}}
function _setStageState() {{}}
function _genieEvent() {{}}
function refreshQueryLimitStatus() {{}}
function showMascotError() {{}}
{lift(src, "function connect()")}
connect();
{json.dumps(frames)}.forEach(frame => _socket.onmessage({{data: JSON.stringify(frame)}}));
JSON.stringify(_appended);
"""
    return json.loads(dukpy.evaljs(harness))


def test_the_page_keeps_the_questions_under_the_greeting_frame_only():
    """The seam between the two halves above: the socket marks the greeting,
    and the page's handler is what has to read the mark."""
    appended = _frames_through_the_socket_handler([
        {"type": "message", "role": "assistant", "content": "Hello, Ada!", "greeting": True},
        {"type": "message", "role": "assistant", "content": "Revenue is up."},
    ])
    assert appended == [{"text": "Hello, Ada!", "keepSuggestions": True},
                        {"text": "Revenue is up.", "keepSuggestions": False}]
