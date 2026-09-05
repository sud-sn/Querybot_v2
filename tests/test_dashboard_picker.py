"""
tests/test_dashboard_picker.py

The "Add to dashboard" picker, executed rather than read.

The reported complaint was that it "doesn't look good". Mapping it found that
most of that is literal breakage rather than taste, and each defect below has a
test here that fails against the old code:

  * `.dashboard-picker-new` sets `display:grid`, which beats the user agent's
    `[hidden]{display:none}`. The "Create new" panel was therefore never
    actually hidden and both panels rendered at once. Only an author rule at
    equal specificity, later in the sheet, wins -- so that rule is asserted.

  * Errors went to `toast()`: 1.4 seconds, bottom-right, behind the backdrop,
    describing a field the user is still looking at. Every failure now lands in
    an inline `role="alert"` region inside the modal.

  * Failures were classified by reading English prose out of `error.message`.
    The server now returns a stable `code`, which is both what makes the
    recoverable/terminal split possible and what stops the classification
    breaking the moment the server speaks French.

  * The submit button was relabelled mid-flight and then restored from the
    mode, so a mode switch during a request left the wrong word on the button.

  * Above 520px the artifact pane rendered TWO "Add to dashboard" buttons, and
    only one of them was ever hidden.

The async shell around `fetch` cannot run here -- dukpy parses `await` but never
drains the microtask queue, so continuations never execute. That is why the
decision logic was split into `_collectPinRequest`, `_classifyPinOutcome` and
`_applyPinOutcome`: they are synchronous, they hold everything worth asserting,
and they run for real below.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

dukpy = pytest.importorskip(
    "dukpy",
    reason="a JavaScript engine is required to EXECUTE the picker's own "
           "functions; asserting on their source text instead is the failure "
           "mode this file exists to avoid",
)

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "portal" / "templates" / "portal_chat.html"


def _function(source: str, signature: str) -> str:
    """A whole function by brace balance, as tests/test_stage_trail.py does.

    Not a character window: a window stops covering the code it names the moment
    anything is inserted above it.

    One difference from the copy in test_stage_trail.py, and it matters here.
    That version takes the first "{" after the signature, which is the BODY only
    when no parameter contains a brace. `_pickerError(message, {focus = null} =
    {})` has a destructured default, so the first brace is the parameter's and
    the balance closes on it -- lifting one line of a function and handing the
    engine a syntax error. Walk the parameter list by paren balance first, then
    take the brace after it.
    """
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
    if body_start is None:
        raise AssertionError(f"unbalanced parentheses in {signature!r}")
    depth = 0
    for i in range(body_start, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start:i + 1]
    raise AssertionError(f"unbalanced braces after {signature!r}")


def _const_block(source: str, name: str) -> str:
    """A `const NAME = {...}` object literal, by brace balance."""
    start = source.index(f"const {name} = {{")
    depth = 0
    for i in range(source.index("{", start), len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start:i + 1] + ";"
    raise AssertionError(f"unbalanced braces after {name!r}")


PICKER_FUNCTIONS = (
    "function _pickerError(message, {focus = null} = {})",
    "function _clearPickerError()",
    "function _pickerSubmitLabel()",
    "function _setPickerBusy(busy)",
    "function closeDashboardPicker()",
    "function setDashboardPickerMode(mode)",
    "function selectDashboardPickerOption(id)",
    "function renderDashboardPickerOptions(query = '')",
    "function _collectPinRequest()",
    "function _classifyPinOutcome(httpOk, data)",
    "function _applyPinOutcome(result)",
)


def _run(script: str, *, items=None, mode="existing", selected=0, context=None,
         fields=None) -> dict:
    """Execute the real picker functions against a DOM thin enough for them."""
    tmpl = TEMPLATE.read_text(encoding="utf-8")
    lifted = "\n".join(_function(tmpl, sig) for sig in PICKER_FUNCTIONS)
    harness = f"""
const _log = {{toast: [], toastLong: []}};
function _el(extra) {{
  return Object.assign({{
    innerHTML: '', textContent: '', value: '', hidden: false, disabled: false,
    classList: {{
      _c: {{}},
      add(n) {{ this._c[n] = true; }},
      remove(n) {{ delete this._c[n]; }},
      toggle(n, on) {{ if (on === undefined) on = !this._c[n];
                       if (on) this._c[n] = true; else delete this._c[n]; return on; }},
      contains(n) {{ return !!this._c[n]; }},
    }},
    _attrs: {{}},
    setAttribute(k, v) {{ this._attrs[k] = String(v); }},
    getAttribute(k) {{ return this._attrs[k] === undefined ? null : this._attrs[k]; }},
    removeAttribute(k) {{ delete this._attrs[k]; }},
    focus() {{ _log.focused = this._id; }},
  }}, extra || {{}});
}}
const _nodes = {{}};
for (const id of ['dashboardPickerBackdrop','dashboardPickerError','dashboardPickerSubmit',
                  'dashboardExistingMode','dashboardNewMode','dashboardExistingPanel',
                  'dashboardNewPanel','dashboardPickerList','dashboardPickerSearch',
                  'dashboardNewName','dashboardNewDescription','dashboardNewVisibility']) {{
  _nodes[id] = _el(); _nodes[id]._id = id;
}}
const _fields = {json.dumps(fields or {})};
for (const k in _fields) if (_nodes[k]) _nodes[k].value = _fields[k];
if (!_nodes.dashboardNewVisibility.value) _nodes.dashboardNewVisibility.value = 'personal';

var document = {{
  getElementById: id => _nodes[id] || null,
  // Only the two selectors the lifted code actually uses.
  querySelectorAll: sel => {{
    if (sel.indexOf('aria-invalid') >= 0)
      return Object.keys(_nodes).map(k => _nodes[k]).filter(n => n.getAttribute('aria-invalid'));
    return [];
  }},
}};
function escHtml(s) {{
  return String(s == null ? '' : s).replace(/[&<>"']/g,
    c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
}}
function toast(m) {{ _log.toast.push(m); }}
function toastLong(m) {{ _log.toastLong.push(m); }}
// setDashboardPickerMode defers the focus so it lands after the panel is
// visible. Run it immediately here: the harness has no event loop, and a
// no-op stub would hide whether the focus happens at all.
function setTimeout(fn) {{ fn(); return 0; }}
const THREAD_ID = 'thread-1';
const DASHBOARD_ID = 0;

{_const_block(tmpl, "_PIN_ERRORS")}

let _dashboardAddContext = {json.dumps(context) if context is not None else "null"};
let _dashboardPickerItems = {json.dumps(items or [])};
let _dashboardPickerMode = {json.dumps(mode)};
let _dashboardPickerSelected = {json.dumps(selected)};
let _dashboardPickerBusy = false;
const _dashboardPinnedTokens = new Map();

{lifted}

{script}

JSON.stringify({{
  result: typeof _result === 'undefined' ? null : _result,
  log: _log,
  mode: _dashboardPickerMode,
  selected: _dashboardPickerSelected,
  busy: _dashboardPickerBusy,
  pinned: Array.from(_dashboardPinnedTokens.entries()),
  nodes: Object.keys(_nodes).reduce((acc, id) => {{
    const n = _nodes[id];
    acc[id] = {{html: n.innerHTML, text: n.textContent, value: n.value,
                hidden: n.hidden, disabled: n.disabled,
                classes: Object.keys(n.classList._c), attrs: n._attrs}};
    return acc;
  }}, {{}}),
  focused: _log.focused || null,
}});
"""
    return json.loads(dukpy.evaljs(harness))


DASHBOARDS = [
    {"id": 3, "name": "Pharmacy performance", "description": "monthly ops review",
     "chart_count": 4, "status": "published", "visibility": "team"},
    {"id": 7, "name": "Finance", "description": "", "chart_count": 1,
     "status": "draft", "visibility": "personal"},
]


# ══════════════════════════════════════════════════════════════════════════════
# The panel that would not hide
# ══════════════════════════════════════════════════════════════════════════════

class TestTheModeSwitchActuallySwitches:

    def test_switching_to_new_hides_the_existing_panel_and_shows_the_other(self):
        out = _run("setDashboardPickerMode('new');")
        assert out["nodes"]["dashboardExistingPanel"]["hidden"] is True
        assert out["nodes"]["dashboardNewPanel"]["hidden"] is False
        assert out["mode"] == "new"

    def test_switching_back_reverses_it(self):
        out = _run("setDashboardPickerMode('new'); setDashboardPickerMode('existing');")
        assert out["nodes"]["dashboardExistingPanel"]["hidden"] is False
        assert out["nodes"]["dashboardNewPanel"]["hidden"] is True

    def test_the_stylesheet_actually_honours_hidden(self):
        """`hidden` on the element is not enough. `.dashboard-picker-new` sets
        `display:grid`, which beats the user agent's `[hidden]{display:none}`,
        so both panels rendered at once no matter what the JS set. Only an
        author rule can win, and this is the one place that can assert it --
        the JS above cannot see the cascade."""
        css = TEMPLATE.read_text(encoding="utf-8")
        assert ".dashboard-picker-new[hidden]" in css
        assert "#dashboardExistingPanel[hidden]" in css

    def test_an_unknown_mode_falls_back_to_existing(self):
        assert _run("setDashboardPickerMode('nonsense');")["mode"] == "existing"

    def test_the_submit_label_follows_the_mode(self):
        assert _run("setDashboardPickerMode('new');")["nodes"]["dashboardPickerSubmit"]["text"] \
            == "Create and add"
        assert _run("setDashboardPickerMode('existing');")["nodes"]["dashboardPickerSubmit"]["text"] \
            == "Add chart"

    def test_a_mode_switch_mid_flight_does_not_relabel_a_busy_button(self):
        """The old code set the label from the mode in a finally block, so a
        switch during the request left the wrong word on the button."""
        out = _run("_setPickerBusy(true); setDashboardPickerMode('new');")
        assert out["nodes"]["dashboardPickerSubmit"]["text"] == "Adding…"
        assert out["nodes"]["dashboardPickerSubmit"]["disabled"] is True
        assert out["nodes"]["dashboardPickerSubmit"]["attrs"]["aria-busy"] == "true"


# ══════════════════════════════════════════════════════════════════════════════
# Errors the user can actually read
# ══════════════════════════════════════════════════════════════════════════════

class TestErrorsAppearInTheModal:

    def test_a_validation_error_is_shown_inline_not_toasted(self):
        out = _run("_result = _collectPinRequest(); _pickerError(_result.error, {focus: _result.focus});",
                   mode="new", context={"token": "t1"})
        error = out["nodes"]["dashboardPickerError"]
        assert "name" in error["text"].lower()
        assert "show" in error["classes"]
        assert out["log"]["toast"] == [] and out["log"]["toastLong"] == []

    def test_it_focuses_and_marks_the_field_at_fault(self):
        out = _run("const r = _collectPinRequest(); _pickerError(r.error, {focus: r.focus});",
                   mode="new", context={"token": "t1"})
        assert out["focused"] == "dashboardNewName"
        assert out["nodes"]["dashboardNewName"]["attrs"].get("aria-invalid") == "true"

    def test_clearing_removes_the_message_and_the_invalid_marks(self):
        out = _run("const r = _collectPinRequest(); _pickerError(r.error, {focus: r.focus});"
                   " _clearPickerError();", mode="new", context={"token": "t1"})
        assert out["nodes"]["dashboardPickerError"]["text"] == ""
        assert "show" not in out["nodes"]["dashboardPickerError"]["classes"]
        assert "aria-invalid" not in out["nodes"]["dashboardNewName"]["attrs"]

    def test_closing_clears_a_stale_error(self):
        """Otherwise the next open shows the previous attempt's failure."""
        out = _run("_pickerError('boom'); closeDashboardPicker();")
        assert "show" not in out["nodes"]["dashboardPickerError"]["classes"]


# ══════════════════════════════════════════════════════════════════════════════
# What the user asked for
# ══════════════════════════════════════════════════════════════════════════════

class TestCollectingTheRequest:

    def test_existing_mode_sends_the_selected_dashboard(self):
        out = _run("_result = _collectPinRequest();", mode="existing", selected=7,
                   context={"token": "tok", "title": "Revenue by region",
                            "chart_type": "bar", "color_palette": "ocean"})
        payload = out["result"]["payload"]
        assert payload["dashboard_id"] == 7
        assert payload["token"] == "tok"
        assert payload["title"] == "Revenue by region"
        assert payload["chart_type"] == "bar"
        assert payload["color_palette"] == "ocean"
        assert "new_dashboard_name" not in payload

    def test_existing_mode_with_nothing_selected_is_a_recoverable_error(self):
        out = _run("_result = _collectPinRequest();", mode="existing", selected=0,
                   context={"token": "tok"})
        assert "payload" not in out["result"]
        assert "Choose a dashboard" in out["result"]["error"]

    def test_new_mode_sends_the_typed_fields(self):
        out = _run("_result = _collectPinRequest();", mode="new",
                   context={"token": "tok"},
                   fields={"dashboardNewName": "  Ops  ",
                           "dashboardNewDescription": " weekly ",
                           "dashboardNewVisibility": "team"})
        payload = out["result"]["payload"]
        assert payload["new_dashboard_name"] == "Ops"        # trimmed
        assert payload["new_dashboard_description"] == "weekly"
        assert payload["visibility"] == "team"
        assert "dashboard_id" not in payload

    def test_a_whitespace_only_name_is_refused(self):
        out = _run("_result = _collectPinRequest();", mode="new",
                   context={"token": "tok"}, fields={"dashboardNewName": "   "})
        assert out["result"]["focus"] == "dashboardNewName"

    def test_no_context_means_no_request(self):
        """The modal can be submitted after the context was cleared by a
        previous terminal failure."""
        out = _run("_result = _collectPinRequest();", context=None)
        assert "payload" not in out["result"]


# ══════════════════════════════════════════════════════════════════════════════
# What the server said
# ══════════════════════════════════════════════════════════════════════════════

class TestClassifyingTheServerResponse:

    def _classify(self, http_ok, data):
        return _run(f"_result = _classifyPinOutcome({json.dumps(http_ok)}, {json.dumps(data)});")["result"]

    def test_success_carries_the_dashboard_through(self):
        out = self._classify(True, {"ok": True, "dashboard": {"id": 3, "name": "Ops", "url": "/x"}})
        assert out["outcome"] == "success"
        assert out["dashboard"] == {"id": 3, "name": "Ops", "url": "/x"}

    def test_success_without_a_dashboard_body_still_reads_sensibly(self):
        out = self._classify(True, {"ok": True})
        assert out["dashboard"]["name"] == "dashboard"
        assert out["dashboard"]["url"] == "/portal/dashboard"

    @pytest.mark.parametrize("code", ["no_target", "no_access"])
    def test_a_fixable_problem_keeps_the_modal_open(self, code):
        assert self._classify(False, {"ok": False, "code": code})["outcome"] == "recoverable"

    @pytest.mark.parametrize("code", [
        "expired_token", "missing_token", "wrong_workspace",
        "not_authenticated", "add_failed",
    ])
    def test_an_unfixable_problem_is_terminal(self, code):
        out = self._classify(False, {"ok": False, "code": code})
        assert out["outcome"] == "terminal"
        assert out["message"]

    def test_add_failed_never_invites_a_retry(self):
        """The pin token is already spent by the time the server can return
        409, so a retry is a guaranteed 400. The copy has to say run the
        question again, not try again."""
        message = self._classify(False, {"ok": False, "code": "add_failed"})["message"]
        assert "run the question again" in message.lower()
        assert "nothing was changed" in message.lower()

    def test_an_unknown_code_is_terminal_with_readable_copy(self):
        out = self._classify(False, {"ok": False, "code": "something_new"})
        assert out["outcome"] == "terminal"
        assert "could not be added" in out["message"]

    def test_a_body_that_is_not_json_produces_no_parser_message(self):
        """The old code surfaced error.message, which on a non-JSON response is
        'Unexpected token < in JSON at position 0'."""
        out = self._classify(False, {})
        assert out["outcome"] == "terminal"
        assert "token" not in out["message"].lower()
        assert "JSON" not in out["message"]

    def test_classification_never_reads_the_server_prose(self):
        """The English `error` field is about to become French. Two bodies with
        the same code and different prose must classify identically."""
        a = self._classify(False, {"ok": False, "code": "no_target", "error": "Choose one."})
        b = self._classify(False, {"ok": False, "code": "no_target", "error": "Choisissez-en un."})
        assert a == b


class TestApplyingTheOutcome:

    def test_success_closes_records_and_confirms(self):
        out = _run("_applyPinOutcome({outcome:'success', dashboard:{id:3,name:'Ops',url:'/x'}});",
                   context={"token": "tok"})
        assert "open" not in out["nodes"]["dashboardPickerBackdrop"]["classes"]
        assert out["log"]["toast"] == ["Added to Ops"]
        assert out["pinned"] == [["tok", {"id": 3, "name": "Ops", "url": "/x"}]]

    def test_a_recoverable_outcome_leaves_the_modal_open(self):
        out = _run("document.getElementById('dashboardPickerBackdrop').classList.add('open');"
                   "_applyPinOutcome({outcome:'recoverable', message:'Pick one.', focus:null});",
                   context={"token": "tok"})
        assert "open" in out["nodes"]["dashboardPickerBackdrop"]["classes"]
        assert out["nodes"]["dashboardPickerError"]["text"] == "Pick one."
        assert out["log"]["toast"] == []

    def test_a_terminal_outcome_closes_and_drops_the_dead_token(self):
        """Leaving it open leaves a spent token behind a button the user is
        about to press again."""
        out = _run("document.getElementById('dashboardPickerBackdrop').classList.add('open');"
                   "_applyPinOutcome({outcome:'terminal', message:'Run it again.'});",
                   context={"token": "tok"})
        assert "open" not in out["nodes"]["dashboardPickerBackdrop"]["classes"]
        assert out["log"]["toastLong"] == ["Run it again."]
        assert out["pinned"] == []


# ══════════════════════════════════════════════════════════════════════════════
# The list
# ══════════════════════════════════════════════════════════════════════════════

class TestTheDashboardList:

    def test_it_renders_one_option_per_dashboard(self):
        html = _run("renderDashboardPickerOptions('');", items=DASHBOARDS)["nodes"]["dashboardPickerList"]["html"]
        assert html.count('role="option"') == 2
        assert "Pharmacy performance" in html
        assert "4 visuals" in html and "1 visual" in html      # singular/plural

    def test_it_shows_the_status_and_visibility_the_api_already_returns(self):
        html = _run("renderDashboardPickerOptions('');", items=DASHBOARDS)["nodes"]["dashboardPickerList"]["html"]
        assert "published" in html and "team" in html

    def test_search_matches_the_description_too(self):
        """Two dashboards can share a name; the name alone gave the user no way
        to tell them apart."""
        html = _run("renderDashboardPickerOptions('monthly ops');", items=DASHBOARDS)["nodes"]["dashboardPickerList"]["html"]
        assert "Pharmacy performance" in html
        assert "Finance" not in html

    def test_no_match_and_no_dashboards_say_different_things(self):
        no_match = _run("renderDashboardPickerOptions('zzz');", items=DASHBOARDS)["nodes"]["dashboardPickerList"]["html"]
        none_yet = _run("renderDashboardPickerOptions('');", items=[])["nodes"]["dashboardPickerList"]["html"]
        assert "No dashboard matches" in no_match
        assert "no dashboards yet" in none_yet.lower()
        assert no_match != none_yet

    def test_a_dashboard_name_cannot_inject_markup(self):
        evil = [{"id": 1, "name": "<img src=x onerror=alert(1)>", "chart_count": 0,
                 "status": "draft", "visibility": "personal", "description": ""}]
        html = _run("renderDashboardPickerOptions('');", items=evil)["nodes"]["dashboardPickerList"]["html"]
        assert "<img" not in html
        assert "&lt;img" in html

    def test_selection_is_exposed_to_assistive_tech_not_just_css(self):
        out = _run("renderDashboardPickerOptions(''); selectDashboardPickerOption(7);",
                   items=DASHBOARDS)
        assert out["selected"] == 7
        html = out["nodes"]["dashboardPickerList"]["html"]
        assert 'aria-selected="true"' in html or 'aria-selected="false"' in html

    def test_rendering_clears_the_busy_flag(self):
        out = _run("document.getElementById('dashboardPickerList').setAttribute('aria-busy','true');"
                   "renderDashboardPickerOptions('');", items=DASHBOARDS)
        assert out["nodes"]["dashboardPickerList"]["attrs"]["aria-busy"] == "false"


# ══════════════════════════════════════════════════════════════════════════════
# The two buttons
# ══════════════════════════════════════════════════════════════════════════════

class TestTheArtifactPaneOffersOneButton:

    def test_the_static_duplicate_is_gone(self):
        """Above 520px the pane rendered two "Add to dashboard" buttons at once
        -- the static header one and the one renderArtifact injects. Only the
        header one was hidden, and only below 520px. The injected one carries
        the real logic (it becomes an "Open <dashboard>" link once pinned, and
        is suppressed for an analysis artifact); the static one had none of
        that and, on an analysis artifact, opened a picker with no token."""
        src = TEMPLATE.read_text(encoding="utf-8")
        assert "data-artifact-dashboard" not in src
        assert src.count('onclick="addArtifactToDashboard()"') == 1

    def test_the_injected_button_still_exists(self):
        """The control the user needs must not have gone with the duplicate."""
        src = TEMPLATE.read_text(encoding="utf-8")
        assert "addArtifactToDashboard()" in src
        assert "function addArtifactToDashboard()" in src


# ══════════════════════════════════════════════════════════════════════════════
# The server half of the contract
# ══════════════════════════════════════════════════════════════════════════════

class TestThePinEndpointReturnsCodes:
    """The client classifies by `code`, never by prose. If the server stops
    sending one, every failure silently becomes terminal and the recoverable
    ones start closing the modal on the user."""

    def _client(self):
        import os
        import tempfile

        os.environ.setdefault("QUERYBOT_DB_PATH",
                              os.path.join(tempfile.mkdtemp(), "picker.db"))
        from fastapi import FastAPI
        from starlette.testclient import TestClient

        import portal.routes as pr
        import store

        store.init_db()
        app = FastAPI()
        app.include_router(pr.router)
        return TestClient(app), pr, store

    def test_an_anonymous_pin_is_coded(self):
        client, _, _ = self._client()
        response = client.post("/portal/api/pin-chart", json={"token": "x"})
        assert response.status_code == 401
        assert response.json()["code"] == "not_authenticated"

    def test_a_missing_token_is_coded(self):
        client, pr, store = self._client()
        account_id = f"acct{os.urandom(4).hex()}"
        store.upsert_client(account_id, "T")
        user_id, _ = store.create_user(account_id, "Ada", f"{os.urandom(4).hex()}@x.com")
        client.cookies.set(pr._COOKIE, pr._sign_session_value(user_id))
        body = client.post("/portal/api/pin-chart", json={}).json()
        assert body["code"] == "missing_token"

    def test_an_unknown_token_is_coded_expired(self):
        client, pr, store = self._client()
        account_id = f"acct{os.urandom(4).hex()}"
        store.upsert_client(account_id, "T")
        user_id, _ = store.create_user(account_id, "Ada", f"{os.urandom(4).hex()}@x.com")
        client.cookies.set(pr._COOKIE, pr._sign_session_value(user_id))
        response = client.post("/portal/api/pin-chart", json={"token": "nope"})
        assert response.status_code == 400
        assert response.json()["code"] == "expired_token"

    def test_every_code_the_client_knows_is_one_the_server_can_send(self):
        """The two lists are the contract. A code in one and not the other is a
        failure path with no copy, or copy for a path that cannot happen."""
        import re

        routes = (ROOT / "portal" / "routes.py").read_text(encoding="utf-8")
        pin = routes[routes.index('@router.post("/api/pin-chart")'):
                     routes.index('@router.post("/api/update-chart")')]
        server_codes = set(re.findall(r'"code":\s*"([a-z_]+)"', pin))
        client_codes = set(re.findall(
            r"^\s{2}([a-z_]+):\s*\{recoverable",
            _const_block(TEMPLATE.read_text(encoding="utf-8"), "_PIN_ERRORS"),
            re.M))
        assert server_codes == client_codes, (
            f"server-only: {sorted(server_codes - client_codes)}; "
            f"client-only: {sorted(client_codes - server_codes)}")
