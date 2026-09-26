"""
One searchable select, over the native one.

The admin console's long lists -- every metric, every approved date role, every
table and column of a warehouse -- were plain <select> elements: no typing to
find, a scroll through hundreds of rows, and each page that wanted better built
its own. The date-role forms are where an admin picks among a warehouse's
columns, and where a wrong pick saves a wrong date.

qbSelect (static/js/qb-ui.js) puts a combobox over the native <select>, which
stays in the form with its name, value and change event, so the page posts and
reacts exactly as before. The tests run the real component in dukpy against the
stand-in DOM (tests/js_fakedom.py), and the date-roles page's own script
functions against a wrapped select.
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

# A field as the admin pages write one: a label, then the select.
_FIELD = """
var form = make('form');
var field = make('div', {'class': 'dr-field'}, form);
var lbl = make('label', {}, field); lbl.textContent = 'Date column';
var select = make('select', {'name': 'fact_column'}, field); select.form = form;
var changes = 0; select.addEventListener('change', function () { changes++; });
function opt(value, text, parent, extra) {
  var o = make('option', extra || {}, parent || select);
  if (value !== null) o.setAttribute('value', value);
  o.textContent = text; return o;
}
opt('', 'Select a column');
var dims = make('optgroup', {'label': 'Livraisons'}, select);
opt('SHIP_DT', '\\n   Date de livraison   \\n', dims);
opt('DELAY', 'Délai moyen', dims);
var sales = make('optgroup', {'label': 'Sales'}, select);
opt('ORDER_DT', 'Order date', sales);
opt('OLD_DT', 'Retired date', sales, {'disabled': ''});
function rows() {
  return api.input.parentNode.byClass('qb-select-list').children.map(function (li) {
    return [li.className, li.textContent];
  });
}
function options() {
  return rows().filter(function (r) { return r[0].indexOf('qb-select-option') >= 0; }).map(function (r) { return r[1]; });
}
function type(text) { api.input.value = text; api.input.fire('input'); }
function key(k) { return api.input.fire('keydown', {key: k}).defaultPrevented; }
function listShown() { return !api.input.parentNode.byClass('qb-select-list').hidden; }
"""


def _run(script: str, *, french: bool = False) -> object:
    catalogue = ""
    if french:
        catalogue = "window.qbT = function (k) { return ({'ui.shell.no_matches': 'Aucun résultat'})[k] || k; };\n"
    return json.loads(dukpy.evaljs(
        FAKE_DOM + catalogue + UI_JS.read_text(encoding="utf-8") + "\n" + _FIELD
        + "var api = window.qbSelect(select);\n" + script))


class TestItStandsOverTheNativeSelect:

    def test_the_native_select_stays_in_the_form_out_of_the_way(self):
        out = _run("""
          var wrap = api.input.parentNode;
          JSON.stringify({wrap: wrap.className, parent: wrap.parentNode === field, holds: wrap.children.indexOf(select) >= 0,
                          name: select.getAttribute('name'), hidden: select.getAttribute('aria-hidden'),
                          tab: select.getAttribute('tabindex'), native: select.classList.contains('qb-select-native')});
        """)
        assert out == {"wrap": "qb-select", "parent": True, "holds": True, "name": "fact_column",
                       "hidden": "true", "tab": "-1", "native": True}

    def test_its_chevron_is_the_icon_sets_and_silent(self):
        out = _run("var c = api.input.parentNode.byClass('qb-select-chevron');"
                   "JSON.stringify([c.getAttribute('aria-hidden'), c.innerHTML]);")
        assert out == ["true", '<svg data-icon="chevron-down"></svg>']

    def test_the_box_is_a_combobox_named_by_the_fields_label(self):
        out = _run("""
          var list = api.input.parentNode.byClass('qb-select-list');
          JSON.stringify({role: api.input.getAttribute('role'), controls: api.input.getAttribute('aria-controls'),
                          list: list.id, listRole: list.getAttribute('role'), closed: list.hidden,
                          named: api.input.getAttribute('aria-labelledby'), label: lbl.id,
                          expanded: api.input.getAttribute('aria-expanded')});
        """)
        assert out["role"] == "combobox" and out["controls"] == out["list"] and out["listRole"] == "listbox"
        assert out["closed"] is True and out["expanded"] == "false"
        assert out["named"] and out["named"] == out["label"]

    def test_a_label_for_the_select_now_points_at_the_box(self):
        out = _run("""
          var other = make('select', {'id': 'drFactTable'}); opt('a', 'Alpha', other);
          var forLabel = make('label', {'for': 'drFactTable'});
          var api2 = window.qbSelect(other);
          JSON.stringify({for: forLabel.getAttribute('for'), box: api2.input.id,
                          named: api2.input.getAttribute('aria-labelledby') === forLabel.id});
        """)
        assert out["for"] == out["box"] and out["named"] is True

    def test_it_shows_the_choice_and_the_blank_options_text_as_its_hint(self):
        out = _run("""
          var before = [api.input.value, api.input.placeholder];
          select.value = 'SHIP_DT';
          JSON.stringify({before: before, after: api.input.value});
        """)
        # The template's line breaks inside an option are not the option's text.
        assert out == {"before": ["", "Select a column"], "after": "Date de livraison"}

    def test_wrapping_twice_gives_the_same_one(self):
        out = _run("JSON.stringify(window.qbSelect(select) === api && field.querySelectorAll('.qb-select').length);")
        assert out == 1


class TestTypingFinds:

    def test_every_live_option_is_offered_under_its_group(self):
        out = _run("api.input.fire('click'); JSON.stringify({shown: listShown(), rows: rows()});")
        assert out["shown"] is True
        assert out["rows"] == [["qb-select-group", "Livraisons"],
                               ["qb-select-option is-active", "Date de livraison"],
                               ["qb-select-option", "Délai moyen"],
                               ["qb-select-group", "Sales"],
                               ["qb-select-option", "Order date"]]

    @pytest.mark.parametrize("typed,found", [("DATE", ["Date de livraison", "Order date"]),
                                             ("delai", ["Délai moyen"]),
                                             ("DÉLAI", ["Délai moyen"]),
                                             ("sales", ["Order date"]),
                                             ("retired", [])])
    def test_case_accents_and_group_names_are_ignored_or_matched(self, typed, found):
        assert _run(f"type({typed!r}); JSON.stringify(options());") == found

    def test_nothing_found_says_so_in_the_readers_language(self):
        out = _run("type('zzz'); JSON.stringify(rows());", french=True)
        assert out == [["qb-select-empty", "Aucun résultat"]]


class TestPicking:

    def test_arrows_and_enter_pick_and_the_page_hears_a_change(self):
        out = _run("""
          api.input.fire('click'); key('ArrowDown'); key('ArrowDown');
          var active = api.input.getAttribute('aria-activedescendant');
          var handled = key('Enter');
          JSON.stringify({handled: handled, value: select.value, box: api.input.value, changes: changes,
                          open: listShown(), active: !!active});
        """)
        assert out == {"handled": True, "value": "ORDER_DT", "box": "Order date", "changes": 1,
                       "open": False, "active": True}

    def test_arrow_up_from_the_top_wraps_to_the_last(self):
        assert _run("api.input.fire('click'); key('ArrowUp'); key('Enter'); JSON.stringify(select.value);") == "ORDER_DT"

    def test_a_mouse_pick_keeps_the_focus_and_picks(self):
        out = _run("""
          type('lai');
          var li = api.input.parentNode.byClass('qb-select-option');
          var ev = li.fire('mousedown');
          JSON.stringify({kept: ev.defaultPrevented, value: select.value, changes: changes});
        """)
        assert out == {"kept": True, "value": "DELAY", "changes": 1}

    def test_picking_the_current_choice_is_not_a_change(self):
        assert _run("select.value = 'DELAY'; api.input.fire('click'); key('Enter'); JSON.stringify(changes);") == 0

    def test_escape_puts_the_choice_back(self):
        out = _run("""
          select.value = 'DELAY'; type('ord'); var handled = key('Escape');
          JSON.stringify({handled: handled, value: select.value, box: api.input.value, open: listShown(), changes: changes});
        """)
        assert out == {"handled": True, "value": "DELAY", "box": "Délai moyen", "open": False, "changes": 0}

    def test_leaving_half_typed_puts_the_choice_back(self):
        out = _run("select.value = 'DELAY'; type('ord'); api.input.fire('blur');"
                   "JSON.stringify([select.value, api.input.value]);")
        assert out == ["DELAY", "Délai moyen"]

    def test_emptying_the_box_clears_an_optional_choice(self):
        out = _run("select.value = 'DELAY'; type(''); api.input.fire('blur'); JSON.stringify([select.value, changes]);")
        assert out == ["", 1]

    def test_emptying_the_box_leaves_a_required_choice(self):
        out = _run("select.required = true; select.value = 'DELAY'; type(''); api.input.fire('blur');"
                   "JSON.stringify([select.value, api.input.value, changes]);")
        assert out == ["DELAY", "Délai moyen", 0]


class TestItFollowsThePagesScript:

    def test_options_replaced_and_a_value_set_by_script_show(self):
        out = _run("""
          select.innerHTML = '<option value="">Select dimension key</option>';
          var hint = api.input.placeholder;
          var o = document.createElement('option'); o.value = 'DATE_KEY'; o.textContent = 'DATE_KEY (int)';
          select.appendChild(o);
          select.value = 'DATE_KEY';
          api.input.fire('click');
          JSON.stringify({hint: hint, box: api.input.value, offered: options()});
        """)
        assert out == {"hint": "Select dimension key", "box": "DATE_KEY (int)", "offered": ["DATE_KEY (int)"]}

    def test_a_disabled_select_disables_the_box_and_will_not_open(self):
        out = _run("select.disabled = true; var off = api.input.disabled; api.input.fire('click');"
                   "select.disabled = false; JSON.stringify([off, listShown(), api.input.disabled]);")
        assert out == [True, False, False]

    def test_a_form_reset_shows_the_reset_value(self):
        # A browser resets the form after its reset event, without a change
        # event and without going through the value setter.
        out = _run("""
          select.value = 'ORDER_DT'; var shown = api.input.value;
          form.fire('reset');
          select.options.forEach(function (o) { o.selected = false; });
          var before = api.input.value; runTimers();
          JSON.stringify([shown, before, api.input.value]);
        """)
        assert out == ["Order date", "Order date", ""]


def test_a_marked_select_is_set_up_when_the_page_loads():
    out = json.loads(dukpy.evaljs(FAKE_DOM + """
      var s = make('select', {'data-qb-select': '', 'name': 'metric_id'});
      var o = make('option', {'value': 'm1'}, s); o.textContent = 'Net sales';
      var plain = make('select', {'name': 'mode'});
    """ + UI_JS.read_text(encoding="utf-8") + """
      JSON.stringify({wrapped: s.parentNode.className, box: s.parentNode.byClass('qb-select-input').value,
                      plain: plain.parentNode === document.body});
    """))
    assert out == {"wrapped": "qb-select", "box": "Net sales", "plain": True}


class TestTheDateRolePage:

    PAGE = "admin/templates/client_date_roles.html"

    def test_its_long_lists_are_searchable(self):
        page = (ROOT / self.PAGE).read_text(encoding="utf-8")
        for name in ("metric_id", "role_identity", "fact_table", "fact_column", "dimension_table",
                     "dimension_key", "date_value_column"):
            tag = re.search(rf'<select name="{name}"[^>]*>', page)
            assert tag and "data-qb-select" in tag.group(0), name

    def test_its_own_fill_and_pick_show_in_the_boxes(self):
        # The page's script refills the key pickers when a dimension is known
        # and picks the key for the admin. Run those functions, as the page
        # has them, against wrapped selects.
        from tests.js_lift import function as lift

        page = (ROOT / self.PAGE).read_text(encoding="utf-8")
        script = "\n".join([
            "var BINDINGS = {};",
            "var byName = {'DW.DATE_DIM': {fields: [{column: 'DATE_KEY', data_type: 'int'},"
            " {column: 'CAL_DATE', data_type: 'date'}]}};",
            lift(page, "function columnName(field)") + ";",
            lift(page, "function bindingFor(tableName, column)") + ";",
            lift(page, "function fill(select, tableName, predicate, placeholder)") + ";",
            lift(page, "function selectValue(select, value)") + ";",
            """
            var dimTable = make('select', {'id': 'drDimensionTable'});
            ['', 'dw.date_dim'].forEach(function (v) { var o = make('option', {'value': v}, dimTable); o.textContent = v || 'Select table'; });
            var dimKey = make('select', {'id': 'drDimensionKey'});
            var o0 = make('option', {'value': ''}, dimKey); o0.textContent = 'Select dimension';
            var tableBox = window.qbSelect(dimTable).input, keyApi = window.qbSelect(dimKey);
            var found = selectValue(dimTable, 'DW.DATE_DIM');
            fill(dimKey, dimTable.value, function () { return true; }, 'Select dimension key');
            selectValue(dimKey, 'date_key');
            keyApi.input.fire('click');
            JSON.stringify({found: found, table: tableBox.value, key: keyApi.input.value, hint: keyApi.input.placeholder,
                            offered: keyApi.input.parentNode.byClass('qb-select-list').children.map(function (li) { return li.textContent; })});
            """,
        ])
        out = json.loads(dukpy.evaljs(FAKE_DOM + UI_JS.read_text(encoding="utf-8") + "\n" + script))
        assert out == {"found": True, "table": "dw.date_dim", "key": "DATE_KEY (int)", "hint": "Select dimension key",
                       "offered": ["DATE_KEY (int)", "CAL_DATE (date)"]}
