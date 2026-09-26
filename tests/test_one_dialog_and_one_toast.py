"""
One dialog and one toast, for both consoles.

Before: the admin console and the portal each built their own confirm dialog
(a styled <div>, no focus trap, focus lost on close), and between them there
were three toast systems plus a fixed notification box per console. The
portal chat called window.qbToast, which only the admin page defined, so its
warnings never appeared. The admin toast wrote its message as HTML while its
callers pass server error text. The dashboard's undo toast was English on a
French page.

Now static/js/qb-ui.js holds qbConfirm (a native modal <dialog>) and qbToast,
both shells load it before any page script, and every page uses it. The tests
run the real qb-ui.js in dukpy against a small fake DOM -- the component code
is executed, only the browser is stubbed -- and read the pages as data for
what must no longer be there.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import dukpy
import pytest

from tests.js_fakedom import FAKE_DOM as _FAKE_DOM

ROOT = Path(__file__).resolve().parents[1]
UI_JS = ROOT / "static" / "js" / "qb-ui.js"



def _run(script: str, *, french: bool = False):
    catalogue = ""
    if french:
        catalogue = ("window.qbT = function (k) { return ({'ui.shell.cancel': 'Annuler', 'ui.shell.confirm': 'Confirmer', "
                     "'ui.shell.dismiss': 'Fermer', 'ui.shell.confirm_title': \"Confirmer l'action ?\"})[k] || k; };\n")
    return dukpy.evaljs(_FAKE_DOM + catalogue + UI_JS.read_text(encoding="utf-8") + "\n" + script)


class TestTheToast:

    def test_its_text_is_text_never_markup(self):
        out = _run("""
          var t = window.qbToast.notify({title: 'Saved', body: '<img src=x onerror=alert(1)> 12 entities'});
          var body = t.element.byClass('qb-toast-body');
          JSON.stringify({text: body.textContent, html: body.innerHTML, kids: body.children.length,
                          title: t.element.byClass('qb-toast-title').textContent});
        """)
        assert json.loads(out) == {"text": "<img src=x onerror=alert(1)> 12 entities", "html": "",
                                   "kids": 0, "title": "Saved"}

    @pytest.mark.parametrize("tone,cls,role", [("error", "qb-toast--danger", "alert"),
                                               ("blocked", "qb-toast--danger", "alert"),
                                               ("warning", "qb-toast--warning", None),
                                               ("nonsense", "qb-toast--info", None)])
    def test_a_tone_names_its_class_and_danger_is_an_alert(self, tone, cls, role):
        out = json.loads(_run(f"""
          var el = window.qbToast.notify({{body: 'x', tone: {tone!r}}}).element;
          JSON.stringify({{cls: el.className, role: el.getAttribute('role')}});
        """))
        assert cls in out["cls"].split() and out["role"] == role

    def test_it_leaves_after_its_tones_time_and_says_so(self):
        out = json.loads(_run("""
          var gone = false;
          window.qbToast.notify({body: 'x', tone: 'danger', onDismiss: function () { gone = true; }});
          var delays = timers.map(function (t) { return t.ms; });
          var before = toastsNow().length; runTimers();
          JSON.stringify({delays: delays, before: before, after: toastsNow().length, gone: gone});
        """))
        assert 8000 in out["delays"] and out["before"] == 1 and out["after"] == 0 and out["gone"] is True

    def test_duration_zero_stays_until_dismissed(self):
        out = json.loads(_run("""
          var t = window.qbToast.notify({body: 'x', duration: 0});
          runTimers(); var stayed = toastsNow().length;
          t.element.byClass('qb-toast-close').fire('click'); runTimers();
          JSON.stringify({stayed: stayed, after: toastsNow().length});
        """))
        assert out == {"stayed": 1, "after": 0}

    def test_an_action_link_goes_where_it_says(self):
        out = _run("""
          var el = window.qbToast.notify({body: 'x', action: {label: 'Review 3', href: '/admin/clients/a/graph'}}).element;
          var a = el.byClass('qb-toast-action');
          a.tagName + ' ' + a.getAttribute('href') + ' ' + a.textContent;
        """)
        assert out == "A /admin/clients/a/graph Review 3"

    def test_an_action_button_runs_once_and_closes_the_toast(self):
        out = json.loads(_run("""
          var ran = 0, gone = 0;
          var el = window.qbToast.notify({body: 'x', duration: 0, action: {label: 'Undo', onClick: function () { ran++; }},
                                          onDismiss: function () { gone++; }}).element;
          el.byClass('qb-toast-action').fire('click'); runTimers();
          JSON.stringify({ran: ran, gone: gone, left: toastsNow().length});
        """))
        assert out == {"ran": 1, "gone": 1, "left": 0}

    def test_the_older_one_line_form_still_works(self):
        out = _run("window.qbToast.error('Save failed').className")
        assert "qb-toast--danger" in out.split()

    def test_its_close_button_is_labelled_in_the_readers_language(self):
        out = _run("window.qbToast.notify({body: 'x'}).element.byClass('qb-toast-close').getAttribute('aria-label')",
                   french=True)
        assert out == "Fermer"


class TestTheDialog:

    def test_it_asks_with_text_and_opens_modally(self):
        out = json.loads(_run("""
          window.qbConfirm({title: 'Delete <b>Ada</b>?', body: 'This cannot be undone.'});
          var d = dialogNow();
          JSON.stringify({open: d.open, title: d.byClass('qb-dialog-title').textContent,
                          body: d.byClass('qb-dialog-body').textContent,
                          danger: d.find(function (e) { return e.getAttribute('type') === 'submit'; }).className,
                          focused: document.activeElement.getAttribute('type')});
        """))
        assert out == {"open": True, "title": "Delete <b>Ada</b>?", "body": "This cannot be undone.",
                       "danger": "btn btn-danger", "focused": "submit"}

    def test_confirming_runs_the_callback_and_resolves_true(self):
        out = json.loads(_run("""
          var called = 0, result = null;
          window.qbConfirm({title: 'Sure?', onConfirm: function () { called++; }}).then(function (v) { result = v; });
          dialogNow().byClass('qb-dialog-form').fire('submit');
          JSON.stringify({called: called, result: result, left: !!dialogNow()});
        """))
        assert out == {"called": 1, "result": True, "left": False}

    def test_escape_answers_no_and_gives_focus_back(self):
        out = json.loads(_run("""
          var opener = document.createElement('button'); document.body.appendChild(opener); opener.focus();
          var called = 0, result = null;
          window.qbConfirm({title: 'Sure?', onConfirm: function () { called++; }}).then(function (v) { result = v; });
          var ev = dialogNow().fire('cancel');
          JSON.stringify({called: called, result: result, prevented: ev.defaultPrevented,
                          back: document.activeElement === opener, left: !!dialogNow()});
        """))
        assert out == {"called": 0, "result": False, "prevented": True, "back": True, "left": False}

    def test_it_collects_one_line_in_place_of_a_prompt(self):
        out = json.loads(_run("""
          var got = null, result = null;
          window.qbConfirm({title: 'Name it', input: {value: 'Q3'}, variant: 'primary',
                            onConfirm: function (v) { got = v; }}).then(function (v) { result = v; });
          var d = dialogNow(); var input = d.byClass('qb-dialog-input');
          var focusedInput = document.activeElement === input; input.value = 'Q4 review';
          d.byClass('qb-dialog-form').fire('submit');
          JSON.stringify({got: got, result: result, focusedInput: focusedInput});
        """))
        assert out == {"got": "Q4 review", "result": "Q4 review", "focusedInput": True}

    def test_a_new_question_answers_the_open_one_with_no(self):
        out = json.loads(_run("""
          var first = null;
          window.qbConfirm({title: 'First'}).then(function (v) { first = v; });
          window.qbConfirm({title: 'Second'});
          var dialogs = document.body.children.filter(function (e) { return e.tagName === 'DIALOG'; });
          JSON.stringify({first: first, open: dialogs.length, title: dialogs[0].byClass('qb-dialog-title').textContent});
        """))
        assert out == {"first": False, "open": 1, "title": "Second"}

    def test_its_labels_are_in_the_readers_language(self):
        out = json.loads(_run("""
          window.qbConfirm({});
          var d = dialogNow();
          JSON.stringify({title: d.byClass('qb-dialog-title').textContent,
                          buttons: d.byClass('qb-dialog-footer').children.map(function (b) { return b.textContent; })});
        """, french=True))
        assert out == {"title": "Confirmer l'action ?", "buttons": ["Annuler", "Confirmer"]}

    def test_a_data_confirm_button_asks_then_acts(self):
        out = json.loads(_run("""
          var btn = document.createElement('button');
          btn.dataset.confirm = 'Delete Ada? <script>'; btn.dataset.confirmLabel = 'Delete';
          btn.closest = function () { return btn; };
          var ev = {target: btn, preventDefault: function () { ev.stopped = true; }, stopPropagation: function () {}};
          docListeners.click.forEach(function (f) { f(ev); });
          var d = dialogNow(); var asked = d.byClass('qb-dialog-body').textContent;
          d.byClass('qb-dialog-form').fire('submit');
          JSON.stringify({stopped: !!ev.stopped, asked: asked, clicked: btn.clicks, confirmed: btn.dataset.confirmed});
        """))
        assert out == {"stopped": True, "asked": "Delete Ada? <script>", "clicked": 1, "confirmed": "1"}


def _page(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


class TestEveryPageUsesThem:

    @pytest.mark.parametrize("shell", ["admin/templates/base.html", "portal/templates/portal_base.html"])
    def test_both_shells_load_them_before_any_page_script(self, shell):
        page = _page(shell)
        tag = re.search(r'<script src="\{\{ asset\(\'(js/qb-ui\.js)\'\) \}\}"></script>', page)
        assert tag and (ROOT / "static" / tag.group(1)).is_file()
        head = page.split("</head>", 1)[0]
        assert head.index(tag.group(0)) < head.index("{% block head %}")
        assert head.index("qb-icons.js") < head.index(tag.group(0))

    def test_no_page_builds_its_own(self):
        own = []
        for page in list((ROOT / "admin" / "templates").rglob("*.html")) + list((ROOT / "portal" / "templates").rglob("*.html")):
            text = page.read_text(encoding="utf-8")
            for pattern in (r"window\.qbConfirm\s*=", r"window\.qbToast\s*=", r'id="qbDialogBackdrop"',
                            r'id="portalLiveToast"', r'id="semanticPendingToast"', r'id="toast"', r'id="unpin-toast"'):
                if re.search(pattern, text):
                    own.append(f"{page.name}: {pattern}")
        assert not own, own

    def test_the_dashboards_undo_is_in_the_catalogue(self):
        from core.i18n import MESSAGES

        for key in ("ui.dash.chart_removed", "ui.shell.undo", "ui.shell.dismiss"):
            assert MESSAGES[key]["en"] and MESSAGES[key]["fr"], key
        page = _page("portal/templates/portal_dashboard.html")
        assert "Chart removed from dashboard" not in page and ">Undo<" not in page


def test_every_icon_a_banner_names_is_in_the_sprite():
    # The alert() banner macro draws its icon by name: the portal dashboard's
    # limit warning asked for 'exclamation-triangle', which no icon set held,
    # and drew a blank box.
    symbols = set(re.findall(r'<symbol id="([a-z0-9-]+)"', _page("static/icons/qb-icons.svg")))
    named = set()
    for page in list((ROOT / "admin" / "templates").rglob("*.html")) + list((ROOT / "portal" / "templates").rglob("*.html")):
        for call in re.findall(r"\balert\((.*?)\)\s*\}\}", page.read_text(encoding="utf-8"), re.S):
            named |= set(re.findall(r"icon_name\s*=\s*[\"']([\w-]+)[\"']", call))
            positional = re.findall(r"[\"']([^\"']*)[\"']", re.sub(r"\w+\s*=\s*[^,]+", "", call))
            if len(positional) >= 3:
                named.add(positional[2])
    assert named and named <= symbols, sorted(named - symbols)
