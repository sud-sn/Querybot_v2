"""
tests/test_browser_i18n_helpers.py

window.qbPlural and window.qbEnumLabel against core.i18n.plural and
core.i18n.enum_label.

The browser builds sentences the server never sees, so these two rules exist
twice. Two implementations of one rule is the shape that drifts, and the drift
is invisible: a French page showing "0 lignes" is grammatically wrong and
looks like ordinary output. So the JavaScript is EXECUTED here and its answer
is compared to Python's for every case, rather than the two being read
side by side and assumed to match.
"""

from __future__ import annotations

import json

import pytest

from core import i18n
from tests.js_lift import function as lift

dukpy = pytest.importorskip(
    "dukpy",
    reason="a JavaScript engine is required to EXECUTE the browser's copy of "
           "the plural rule; comparing the two by reading is the drift this "
           "file exists to catch",
)

SHELL = (__import__("pathlib").Path(__file__).resolve().parents[1]
         / "portal" / "templates" / "portal_base.html").read_text(encoding="utf-8")


def _js(lang, expression):
    harness = f"""
var window = {{
  QB_I18N: {json.dumps(i18n.catalogue_for(lang))},
  QB_LANG: {json.dumps(lang)},
}};
{lift(SHELL, "window.qbT = function (id, vars)")}
{lift(SHELL, "window.qbPlural = function (stem, count, vars)")}
{lift(SHELL, "window.qbEnumLabel = function (group, value)")}
{expression}
"""
    return dukpy.evaljs(harness)


COUNTS = (0, 1, 2, 3, 11, 100, -1, -2, 0.5, 1.5)


class TestThePluralRuleIsTheSameOnBothSides:

    @pytest.mark.parametrize("lang", ("en", "fr"))
    @pytest.mark.parametrize("count", COUNTS)
    def test_the_two_implementations_agree(self, lang, count):
        stem = "ui.chat.table_count"
        assert _js(lang, f"window.qbPlural({json.dumps(stem)}, {count});") == \
            i18n.plural(stem, count, lang=lang)

    def test_french_takes_the_singular_at_zero(self):
        """The rule the inline `n !== 1 ? 's' : ''` could not express, and the
        one an English speaker writing French copy gets wrong."""
        assert _js("fr", "window.qbPlural('ui.chat.trust.child_tasks', 0);") == \
            "0 tâche enfant"
        assert _js("en", "window.qbPlural('ui.chat.trust.child_tasks', 0);") == \
            "0 child tasks"

    def test_a_stem_with_its_own_placeholder_interpolates_both(self):
        got = _js("fr", "window.qbPlural('ui.shell.limit_reached_body', 3, {limit: 5});")
        assert got == i18n.plural("ui.shell.limit_reached_body", 3, lang="fr", limit=5)
        assert "3" in got and "5" in got

    def test_a_non_number_picks_the_same_form_on_both_sides(self):
        """A count that arrives as undefined is a bug, but a sentence that
        never renders is a worse one. Both sides fall back to the plural; they
        print the missing value differently, which is not the rule's job."""
        js = _js("fr", "window.qbPlural('ui.chat.table_count', undefined);")
        py = i18n.plural("ui.chat.table_count", None, lang="fr")
        assert js.endswith("tables") and py.endswith("tables")


class TestTheEnumRuleIsTheSameOnBothSides:

    @pytest.mark.parametrize("lang", ("en", "fr"))
    @pytest.mark.parametrize("group,value", [
        ("status", "draft"), ("status", "published"),
        ("visibility", "personal"), ("visibility", "team"),
        ("role", "admin"), ("role", "analyst"),
        ("charttype", "bar"), ("charttype", "kpi"),
    ])
    def test_the_two_implementations_agree(self, lang, group, value):
        assert _js(lang, f"window.qbEnumLabel({json.dumps(group)}, {json.dumps(value)});") \
            == i18n.enum_label(group, value, lang=lang)

    @pytest.mark.parametrize("lang", ("en", "fr"))
    @pytest.mark.parametrize("value", ["ARCHIVED", "  Draft  ", "in review", "", None])
    def test_they_agree_on_the_awkward_values_too(self, lang, value):
        """Case, padding and a space in the middle. An unknown value must
        prettify identically on both sides or the same row reads differently
        depending on which side drew it."""
        assert _js(lang, f"window.qbEnumLabel('status', {json.dumps(value)});") \
            == i18n.enum_label("status", value, lang=lang)

    def test_an_unknown_value_never_renders_as_an_id(self):
        """These come from the database. A new status must not reach a
        customer's screen as "ui.enum.status.archived"."""
        got = _js("fr", "window.qbEnumLabel('status', 'archived');")
        assert got == "Archived"
        assert "ui.enum" not in got
