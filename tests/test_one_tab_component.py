"""
One tab component, with the semantics and keys a tab bar is expected to have.

There were seven tab styles, each with its own script: a class toggled on the
button, another on the panel, no aria-selected on most, every tab in the Tab
order, no arrow keys. The compliance page kept its tab in location.hash with a
page script; the chat's artifact panel had none of it.

qbTabs (static/js/qb-ui.js) is the one component: the selected tab is
aria-selected and the only one in the Tab order, arrow keys, Home and End move
between tabs, unselected panels are hidden, and with data-qb-tabs-hash the tab
follows location.hash. The tests run the real component in dukpy against the
stand-in DOM (tests/js_fakedom.py), and the adopters' markup as the pages
build it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import dukpy
import pytest

from tests.js_fakedom import FAKE_DOM

ROOT = Path(__file__).resolve().parents[1]
UI_JS = ROOT / "static" / "js" / "qb-ui.js"

_BAR = """
var list = make('div', {'role': 'tablist', 'data-qb-tabs': ''});
var names = ['profile', 'policies', 'egress'];
names.forEach(function (n, i) {
  make('button', {'role': 'tab', 'data-tab': n, 'id': 'tab-' + n, 'aria-controls': 'panel-' + n,
                  'aria-selected': SELECTED === n ? 'true' : 'false'}, list);
  make('section', {'role': 'tabpanel', 'id': 'panel-' + n}, document.body);
});
function tab(n) { return document.getElementById('tab-' + n); }
function state() {
  return names.map(function (n) {
    return [n, tab(n).getAttribute('aria-selected'), tab(n).getAttribute('tabindex'),
            !document.getElementById('panel-' + n).hidden];
  });
}
function key(k) { var ev = list.fire('keydown', {key: k}); return ev.defaultPrevented; }
"""


def _run(script: str, *, selected: str = "", hash_: str = "", follow_hash: bool = False) -> object:
    setup = _BAR.replace("SELECTED", json.dumps(selected))
    if follow_hash:
        setup += "list.setAttribute('data-qb-tabs-hash', '');\n"
    return json.loads(dukpy.evaljs(
        FAKE_DOM + f"window.location.hash = {json.dumps(hash_)};\n" + UI_JS.read_text(encoding="utf-8")
        + "\n" + setup + script))


def _selected(state):
    return [row for row in state if row[1] == "true"]


class TestSelection:

    def test_the_marked_tab_starts_selected_and_alone_in_the_tab_order(self):
        out = _run("var calls = 0; window.qbTabs(list, {onChange: function () { calls++; }});"
                   "JSON.stringify({state: state(), calls: calls});", selected="policies")
        assert out["state"] == [["profile", "false", "-1", False], ["policies", "true", "0", True],
                                ["egress", "false", "-1", False]]
        assert out["calls"] == 0          # setting up is not a change the page must react to

    def test_with_none_marked_the_first_is(self):
        out = _run("window.qbTabs(list); JSON.stringify(state());")
        assert _selected(out) == [["profile", "true", "0", True]]

    def test_a_click_selects_and_says_which(self):
        out = _run("""
          var seen = [];
          window.qbTabs(list, {onChange: function (name, t, panel) { seen.push([name, panel.id]); }});
          list.fire('click', {target: tab('egress')});
          JSON.stringify({state: state(), seen: seen});
        """)
        assert _selected(out["state"]) == [["egress", "true", "0", True]]
        assert [row[3] for row in out["state"]] == [False, False, True]
        assert out["seen"] == [["egress", "panel-egress"]]

    def test_select_by_name_and_an_unknown_name_changes_nothing(self):
        out = _run("""
          var api = window.qbTabs(list);
          var ok = api.select('egress'), bad = api.select('nope');
          JSON.stringify({ok: ok, bad: bad, state: state()});
        """)
        assert out["ok"] is True and out["bad"] is False
        assert _selected(out["state"]) == [["egress", "true", "0", True]]

    def test_a_tab_bar_inside_a_panel_keeps_its_own_tabs(self):
        out = _run("""
          var inner = make('div', {'role': 'tablist'}, document.getElementById('panel-profile'));
          make('button', {'role': 'tab', 'data-tab': 'inner', 'id': 'tab-inner'}, inner);
          window.qbTabs(list);
          list.fire('click', {target: document.getElementById('tab-inner')});
          JSON.stringify({outer: state(), inner: document.getElementById('tab-inner').getAttribute('aria-selected')});
        """)
        assert _selected(out["outer"]) == [["profile", "true", "0", True]]
        assert out["inner"] is None


class TestKeys:

    @pytest.mark.parametrize("start,key,lands", [("profile", "ArrowRight", "policies"),
                                                 ("profile", "ArrowLeft", "egress"),
                                                 ("egress", "ArrowRight", "profile"),
                                                 ("policies", "Home", "profile"),
                                                 ("profile", "End", "egress")])
    def test_arrows_home_and_end_move_focus_and_selection(self, start, key, lands):
        out = _run(f"""
          window.qbTabs(list); tab({start!r}).focus();
          var handled = key({key!r});
          JSON.stringify({{handled: handled, focus: document.activeElement.id, state: state()}});
        """, selected=start)
        assert out["handled"] is True and out["focus"] == f"tab-{lands}"
        assert _selected(out["state"]) == [[lands, "true", "0", True]]

    def test_other_keys_are_left_alone(self):
        out = _run("window.qbTabs(list); tab('profile').focus(); JSON.stringify({h: key('Enter'), s: state()});")
        assert out["h"] is False and _selected(out["s"])[0][0] == "profile"


class TestTheHash:

    def test_a_link_to_a_tab_opens_it(self):
        out = _run("window.qbTabs(list); JSON.stringify(state());", hash_="#egress", follow_hash=True)
        assert _selected(out) == [["egress", "true", "0", True]]

    def test_choosing_a_tab_writes_the_hash_and_a_hash_change_follows(self):
        out = _run("""
          window.qbTabs(list);
          list.fire('click', {target: tab('policies')}); var written = window.location.hash;
          goToHash('#egress');
          JSON.stringify({written: written, state: state()});
        """, follow_hash=True)
        assert out["written"] == "#policies"
        assert _selected(out["state"]) == [["egress", "true", "0", True]]

    def test_without_the_attribute_the_hash_is_not_touched(self):
        out = _run("window.qbTabs(list); list.fire('click', {target: tab('egress')}); JSON.stringify(window.location.hash);")
        assert out == ""


def _page(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


class TestTheAdopters:

    def test_the_compliance_panels_are_tabpanels_of_its_tabs(self):
        page = _page("admin/templates/client_compliance.html")
        bar = re.search(r'<div class="compliance-nav qb-tabs"([^>]*)>', page)
        assert bar and 'role="tablist"' in bar.group(1) and "data-qb-tabs" in bar.group(1)
        assert "data-qb-tabs-hash" in bar.group(1)
        ids = re.findall(r"\('([a-z]+)','[^']+'\)", page[bar.end():page.index("{% endfor %}", bar.end())])
        panels = dict(re.findall(r'<section class="compliance-panel" id="panel-([a-z]+)" role="tabpanel" '
                                 r'aria-labelledby="tab-\1"( hidden)?>', page))
        assert ids and set(panels) == set(ids)
        assert [pid for pid, hidden in panels.items() if not hidden] == [ids[0]]
        assert "function showPanel" not in page

    def test_the_artifact_tabs_and_panels_name_each_other(self):
        from tests.js_lift import function as lift

        page = _page("portal/templates/portal_chat.html")
        out = json.loads(dukpy.evaljs("\n".join([
            "function escHtml(s) { return String(s).replace(/</g, '&lt;'); }",
            lift(page, "function _artifactTab(name, label, selected)"),
            lift(page, "function _artifactPanel(name, html, selected)"),
            "JSON.stringify([_artifactTab('data', 'Data <b>', true), _artifactPanel('data', '<p>x</p>', true),"
            " _artifactTab('query', 'Query', false), _artifactPanel('query', '', false)])",
        ])))
        tab_on, panel_on, tab_off, panel_off = out
        assert 'role="tab"' in tab_on and 'aria-controls="artifact-panel-data"' in tab_on
        assert 'id="artifact-tab-data"' in tab_on and 'aria-selected="true"' in tab_on and "Data &lt;b>" in tab_on
        assert 'role="tabpanel"' in panel_on and 'aria-labelledby="artifact-tab-data"' in panel_on
        assert "hidden" not in panel_on and "hidden" in panel_off and 'aria-selected="false"' in tab_off
        assert "window.qbTabs(root.querySelector('.artifact-tabs'), {onChange: _artifactTabShown});" in page
