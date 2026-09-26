"""
A name on a page is text, never script.

The admin users page built its delete confirmation inside an inline handler:
onclick="qbConfirm({body:'Delete {{ u.name }}? ...'})". Jinja escapes a quote
to &#39;, but the browser decodes the attribute before the script runs, so a
display name like  x'});alert(document.cookie);//  -- and names arrive from
Teams, Slack and Zoom when access is approved -- closed the string and ran in
the admin console. The same shape was on a dozen other buttons (groups,
platforms, databases, domains, attestations, metrics, billing rows), and
account ids went into script blocks inside plain quotes. Three buttons were broken
outright -- the setup page's "Fields" toggle and the portal's two Delete
buttons on My Notifications: a |tojson value inside a double-quoted attribute,
where tojson's own double quotes ended the attribute. And the glossary editor
wrote a term's stored clarification options into a <textarea> with |safe, so
an option containing </textarea> ended the editor and became markup.

Now a confirmation's text travels in data-confirm attributes, which the page
escapes and a delegated listener in each layout hands to qbConfirm; ids in
handlers are |int; a value a handler needs as a string is a data attribute, or
|tojson inside a single-quoted attribute; and script blocks take values through
|tojson. |safe appears only after |tojson.

Two kinds of check. The rule, over every template: a value inside an inline
handler or a script block is |int or |tojson (and |tojson only where its
quotes cannot end the attribute). And the pages themselves, rendered by their
routes with hostile names, parsed the way a browser parses them; a handler
is compiled and run by a JavaScript engine to show it does what the button says.
"""

from __future__ import annotations

import asyncio
import html
import os
import re
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch

import pytest
from starlette.requests import Request

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIRS = [ROOT / "admin" / "templates", ROOT / "portal" / "templates"]

_EXPR = re.compile(r"\{\{(.*?)\}\}", re.S)
_HANDLER = re.compile(r"""(?<=\s)(on[a-z]+)\s*=\s*("([^"]*)"|'([^']*)')""", re.S)
_SCRIPT = re.compile(r"<script\b([^>]*)>(.*?)</script>", re.S | re.I)
_JS_HREF = re.compile(r"""href\s*=\s*("\s*javascript:[^"]*\{\{|'\s*javascript:[^']*\{\{)""", re.I)


def _ends_with(expr: str, filter_name: str) -> bool:
    # A |safe after |tojson changes nothing: tojson's output is already marked safe.
    return re.search(r"\|\s*" + filter_name + r"\s*(\|\s*safe\s*)?$", expr.strip()) is not None


_COMMENT = re.compile(r"\{#.*?#\}", re.S)
_SAFE = re.compile(r"\{\{(.*?)\}\}", re.S)


def unsafe_values(text: str) -> list[str]:
    """Template values that reach JavaScript, or markup, without being made safe for it."""
    text = _COMMENT.sub("", text)
    problems = []
    for expr in _SAFE.findall(text):
        if re.search(r"\|\s*safe\b", expr) and not _ends_with(expr, "tojson"):
            problems.append(f"markup: {{{{{expr.strip()}}}}}")
    for match in _HANDLER.finditer(text):
        single_quoted = match.group(4) is not None
        value = match.group(4) if single_quoted else match.group(3)
        for expr in _EXPR.findall(value):
            if _ends_with(expr, "int"):
                continue
            if _ends_with(expr, "tojson") and single_quoted:
                continue
            problems.append(f"{match.group(1)}: {{{{{expr.strip()}}}}}")
    for match in _SCRIPT.finditer(text):
        attrs, body = match.group(1), match.group(2)
        if "src=" in attrs:
            continue
        for expr in _EXPR.findall(body):
            stripped = expr.strip()
            if _ends_with(stripped, "tojson") or stripped.startswith("ic("):
                continue
            problems.append(f"script: {{{{{stripped}}}}}")
    for match in _JS_HREF.finditer(text):
        problems.append(f"javascript: URL {match.group(0)[:60]}")
    return problems


class TestTheRuleHoldsInEveryTemplate:

    def test_no_template_puts_a_value_into_script_unescaped(self):
        found = {}
        for folder in TEMPLATE_DIRS:
            for path in sorted(folder.rglob("*.html")):
                problems = unsafe_values(path.read_text(encoding="utf-8"))
                if problems:
                    found[str(path.relative_to(ROOT))] = problems
        assert found == {}

    @pytest.mark.parametrize("snippet", [
        """<button onclick="qbConfirm({body:'Delete {{ u.name }}?'})">""",
        """<button onclick="f('{{ m.name | e }}')">""",
        """<button onclick="f({{ stem | tojson }})">""",
        """<script>const ACCOUNT = "{{ client.account_id }}";</script>""",
        """<a href="javascript:go('{{ name }}')">""",
        """<textarea>{{ t.clarification_options|safe }}</textarea>""",
    ])
    def test_the_rule_catches_each_unsafe_shape(self, snippet):
        assert unsafe_values(snippet)

    @pytest.mark.parametrize("snippet", [
        """<button onclick="f({{ m.id|int }})">""",
        """<button onclick='f({{ stem | tojson }}, this)'>""",
        """<button data-confirm="Delete {{ u.name }}?">""",
        """<script>const ACCOUNT = {{ client.account_id|tojson }};</script>""",
        """<script src="/static/x.js?v={{ version }}"></script>""",
        """<script>const EDGES = {{ relationships | tojson | safe }};</script>""",
        """<th aria-controls="{{ panel }}" data-onclick="{{ name }}">""",
        """{# a comment that mentions onclick="f('{{ name }}')" and |safe #}""",
    ])
    def test_the_rule_allows_each_safe_shape(self, snippet):
        assert unsafe_values(snippet) == []


# ── The users page, rendered with hostile names ─────────────────────────────

HOSTILE = [
    "x'});alert(document.cookie);//",
    "O'Brien",
    '</script><script>alert(1)</script>',
    'Ada "the admin"',
]


class _Elements(HTMLParser):
    """Elements with their attributes as a browser reads them, script and
    textarea contents, and the form action each element sits inside."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.elements: list[tuple[str, dict]] = []
        self.scripts: list[str] = []
        self.textareas: list[str] = []
        self.text: list[str] = []
        self.form_of: dict[int, str] = {}
        self.forms: dict[str, dict[str, str]] = {}
        self._raw: str | None = None
        self._form: str | None = None
        self._field: str | None = None

    def handle_starttag(self, tag, attrs):
        attributes = {k: (v or "") for k, v in attrs}
        self.elements.append((tag, attributes))
        if tag == "form":
            self._form = attributes.get("action", "")
            self.forms.setdefault(self._form, {})
        elif self._form is not None:
            self.form_of[id(attributes)] = self._form
            self._collect(tag, attributes)
        if tag in ("script", "textarea"):
            self._raw = tag
            (self.scripts if tag == "script" else self.textareas).append("")

    def _collect(self, tag, attributes):
        # What the browser would send for this form, as it stands.
        fields, name = self.forms[self._form], attributes.get("name")
        if tag == "input" and name:
            kind = attributes.get("type", "text")
            if kind not in ("checkbox", "radio"):
                fields[name] = attributes.get("value", "")
            elif "checked" in attributes:
                fields[name] = attributes.get("value", "on")
        elif tag in ("textarea", "select"):
            self._field = name
        elif tag == "option" and self._field:
            if self._field not in fields or "selected" in attributes:
                fields[self._field] = attributes.get("value", "")

    def handle_endtag(self, tag):
        if tag == "textarea" and self._form is not None and self._field:
            self.forms[self._form][self._field] = self.textareas[-1]
        if tag in ("textarea", "select"):
            self._field = None
        if tag == self._raw:
            self._raw = None
        if tag == "form":
            self._form = None

    def handle_data(self, data):
        if self._raw == "script":
            self.scripts[-1] += data
        elif self._raw == "textarea":
            self.textareas[-1] += data
        else:
            self.text.append(data)

    def report_id(self) -> str:
        return next(action.split("/")[3] for action in self.form_of.values()
                    if action.startswith("/portal/reports/"))


@pytest.fixture
def admin_store(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    from admin import routes

    routes.store.init_db()
    return routes.store


def _workspace(store):
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    return account_id


def _admin_page(route, path, *args):
    from admin import routes

    request = Request({"type": "http", "method": "GET", "path": path,
                       "root_path": "", "scheme": "http", "query_string": b"", "headers": [],
                       "server": ("testserver", 80), "client": ("127.0.0.1", 1)})
    with patch.object(routes, "_is_auth", return_value=True):
        page = asyncio.run(getattr(routes, route)(request, *args))
    parser = _Elements()
    parser.feed(page.body.decode())
    return parser


@pytest.fixture
def users_page(admin_store):
    account_id = _workspace(admin_store)
    for i, name in enumerate(HOSTILE):
        admin_store.create_user(account_id, name, f"user{i}@example.com", password="a-password-1")
    return _admin_page("users_page", f"/admin/clients/{account_id}/users", account_id)


class TestTheUsersPage:

    def test_no_handler_carries_a_name(self, users_page):
        for _tag, attrs in users_page.elements:
            for key, value in attrs.items():
                if key.startswith("on"):
                    for name in HOSTILE:
                        assert name not in value and "alert(" not in value, (key, value)

    def test_the_delete_confirmation_shows_each_name_as_written(self, users_page):
        messages = {attrs["data-confirm"] for _tag, attrs in users_page.elements
                    if "data-confirm" in attrs}
        for name in HOSTILE:
            assert f"Delete {name}? This removes their portal access and any pinned charts." in messages

    def test_no_name_opens_a_script_of_its_own(self, users_page):
        # One script per <script> the template wrote; a name that closed one
        # and opened another would add to the count and carry alert(1).
        assert not any("alert(" in script for script in users_page.scripts)

    def test_the_reset_button_carries_the_name_as_data(self, users_page):
        names = {attrs.get("data-user-name") for _tag, attrs in users_page.elements
                 if "data-reset-user" in attrs}
        assert names == set(HOSTILE)


# ── The setup page's "Fields" button ────────────────────────────────────────

def _run_fields_handler(handler: str) -> str:
    """Run the button's onclick in a JavaScript engine; return the table it asked to open."""
    dukpy = pytest.importorskip("dukpy", reason="dukpy runs the button's own JavaScript")
    return dukpy.evaljs(
        "var opened = null;"
        "var _toggleSchemaCardFields = function(name, btn) { opened = name; };"
        "new Function('event', dukpy['handler'])({stopPropagation: function() {}});"
        "opened;",
        handler=handler,
    )


class TestTheSetupFieldsButton:

    TABLES = ["SALES_FACT", "O'Brien \"orders\"", "x');alert(1);('"]

    def test_each_button_opens_the_table_it_is_on(self, admin_store, tmp_path):
        account_id = _workspace(admin_store)
        schema_dir = tmp_path / "schema"
        schema_dir.mkdir()
        for name in self.TABLES:
            (schema_dir / f"{name}.md").write_text("# table\n", encoding="utf-8")
        admin_store.update_client_state(account_id, "SCHEMA_DISCOVERED", {"schema_dir": str(schema_dir)})
        page = _admin_page("client_setup_page", f"/admin/clients/{account_id}/setup", account_id)
        handlers = [attrs.get("onclick", "") for _tag, attrs in page.elements
                    if "tbl-fields-btn" in attrs.get("class", "").split()]
        assert len(handlers) == len(self.TABLES)
        assert sorted(_run_fields_handler(h) for h in handlers) == sorted(self.TABLES)


# ── The glossary editor ─────────────────────────────────────────────────────

class TestTheGlossaryEditor:

    OPTIONS = [
        {"label": "</textarea><script>alert(1)</script>", "expression": "SUM(a)",
         "definition": "Ada's \"total\""},
        {"label": "Net of returns", "expression": "SUM(a) - SUM(r)"},
    ]

    def _page(self, store):
        account_id = _workspace(store)
        term_id = store.save_term(account_id, "revenue", requires_clarification=True,
                                  clarification_options=self.OPTIONS, definition="Sales before returns")
        page = _admin_page("glossary_page", f"/admin/clients/{account_id}/glossary", account_id)
        return account_id, term_id, page

    def test_a_stored_option_stays_inside_the_editor(self, admin_store):
        import json

        _account_id, _term_id, page = self._page(admin_store)
        assert not any("alert(" in script for script in page.scripts)
        assert json.dumps(self.OPTIONS, ensure_ascii=False, indent=2) in page.textareas

    def test_the_page_lists_each_option(self, admin_store):
        _account_id, _term_id, page = self._page(admin_store)
        shown = "".join(page.text)
        assert "Net of returns" in shown and "</textarea><script>alert(1)</script>" in shown

    def test_saving_the_edit_form_unchanged_keeps_the_options(self, admin_store):
        from fastapi import FastAPI
        from starlette.testclient import TestClient

        from admin import routes

        account_id, term_id, page = self._page(admin_store)
        action = f"/admin/clients/{account_id}/glossary/{term_id}/update"
        app = FastAPI()
        app.include_router(routes.router)
        with patch.object(routes, "_is_auth", return_value=True), \
                patch.object(routes, "_after_semantic_approval"):
            response = TestClient(app).post(action, data=page.forms[action], follow_redirects=False)
        assert response.status_code == 303 and "error" not in response.headers["location"]
        term = admin_store.get_term(term_id)
        assert term["clarification_options"] == self.OPTIONS
        assert term["definition"] == "Sales before returns"


# ── The portal's Delete buttons on My Notifications ─────────────────────────

@pytest.fixture
def notifications(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from portal import routes
    from store import report_store

    store = routes.store
    store.init_db()
    account_id = _workspace(store)
    user_id, _ = store.create_user(account_id, "Ada", "ada@example.com", password="a-password-1")
    report_store.create_report(account_id, "Weekly stock", created_by_user_id=user_id)
    alert = {"id": "al1", "account_id": account_id, "user_id": str(user_id), "question": "stock",
             "metric_col": "qty", "condition": "above", "threshold": 5, "status": "active"}
    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)
    client.cookies.set(routes._COOKIE, routes._sign_session_value(user_id))

    def page(lang):
        with patch("core.alert_engine.list_alerts", return_value=[alert]):
            response = client.get("/portal/notifications", headers={"accept-language": lang})
        assert response.status_code == 200
        parser = _Elements()
        parser.feed(response.text)
        return parser

    return page


class TestTheNotificationsDeleteButtons:

    @pytest.mark.parametrize("lang, expected", [
        ("en", {("Delete alert?", "This cannot be undone.", "Delete"),
                ("Delete report?", "This cannot be undone. Subscriptions to it will stop.", "Delete")}),
        ("fr", {("Supprimer l'alerte ?", "Cette action est irréversible.", "Supprimer"),
                ("Supprimer le rapport ?",
                 "Cette action est irréversible. Les abonnements à ce rapport prendront fin.", "Supprimer")}),
    ])
    def test_each_asks_in_the_readers_language_then_submits_its_form(self, notifications, lang, expected):
        page = notifications(lang)
        buttons = [attrs for tag, attrs in page.elements if tag == "button" and "data-confirm" in attrs]
        asked = {(b["data-confirm-title"], b["data-confirm"], b["data-confirm-label"]) for b in buttons}
        assert asked == expected
        # A submit button with no handler of its own: the layout's listener
        # asks, then clicks it again, and the form it sits in is submitted.
        assert all(b.get("type") == "submit" and not any(k.startswith("on") for k in b) for b in buttons)
        assert {page.form_of[id(b)] for b in buttons} == {
            "/portal/notifications/alerts/al1/delete", f"/portal/reports/{page.report_id()}/delete"}


# ── The listener that asks before a data-confirm button acts ────────────────

# A button on the page. Clicking it dispatches a click through the document's
# listeners; unless one prevents it, the browser submits the button's form.
_BUTTONS = """
function Button(data) { El.call(this, 'button'); this.dataset = data; this.submitted = 0; }
Button.prototype = Object.create(El.prototype);
Button.prototype.closest = function() { return 'confirm' in this.dataset ? this : null; };
Button.prototype.click = function() { press(this); };
function press(button) {
  var ev = {target: button, prevented: false,
            preventDefault: function() { this.prevented = true; }, stopPropagation: function() {}};
  (docListeners.click || []).forEach(function(fn) { fn(ev); });
  if (!ev.prevented) button.submitted += 1;
}
window.qbT = function(key) { return key; };
"""

_SCENARIO = """
function submit(d) { return d.find(function(e) { return e.getAttribute('type') === 'submit'; }); }
function cancel(d) { return d.byClass('qb-dialog-footer').children[0]; }
var asked = new Button({confirm: dukpy['body'], confirmTitle: 'Delete user?', confirmLabel: 'Delete user'});
press(asked);
var d = dialogNow();
var shown = [d.byClass('qb-dialog-title').textContent, d.byClass('qb-dialog-body').textContent,
             submit(d).textContent, d.open];
var before = asked.submitted;
d.byClass('qb-dialog-form').fire('submit');
var after = asked.submitted;
press(asked);
var again = [asked.submitted, !!dialogNow() && dialogNow().open];
cancel(dialogNow()).fire('click');
var cancelled = new Button({confirm: 'Delete Bob?'});
press(cancelled);
cancel(dialogNow()).fire('click');
var plain = new Button({});
press(plain);
({shown: shown, before: before, after: after, again: again, cancelled: cancelled.submitted,
  plain: plain.submitted})
"""


def _confirm_behaviour(page):
    from tests.js_fakedom import FAKE_DOM

    dukpy = pytest.importorskip("dukpy", reason="dukpy runs the layout's own JavaScript")
    # The layout loads the shared component; run that file, as the page does.
    src = next(attrs["src"] for tag, attrs in page.elements
               if tag == "script" and attrs.get("src", "").startswith("/static/js/qb-ui.js"))
    component = (Path(__file__).resolve().parents[1] / src.split("?")[0].lstrip("/")).read_text(encoding="utf-8")
    return dukpy.evaljs(FAKE_DOM + _BUTTONS + component + _SCENARIO, body=HOSTILE[0])


class TestTheConfirmListener:
    """The layout's own script, taken from the rendered page and run against a
    stand-in DOM: without the listener these buttons would act on first click."""

    EXPECTED = {
        "shown": ["Delete user?", HOSTILE[0], "Delete user", True],
        "before": 0,      # the first click only asks
        "after": 1,       # confirming submits, once
        "again": [1, True],  # and the next click asks again
        "cancelled": 0,   # cancelling never submits
        "plain": 1,       # a button without data-confirm is left alone
    }

    def test_the_admin_layout(self, users_page):
        assert _confirm_behaviour(users_page) == self.EXPECTED

    def test_the_portal_layout(self, notifications):
        assert _confirm_behaviour(notifications("en")) == self.EXPECTED


def test_the_parsed_attribute_is_the_escaped_text():
    # What the browser does with the attribute before any script sees it.
    assert html.unescape("Delete x&#39;});alert(1);//?") == "Delete x'});alert(1);//?"
