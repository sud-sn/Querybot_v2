"""
tests/test_history_panel_and_dead_api.py

Two low-severity findings from the second sweep, both of the "it looks like it
works" kind.

  portal_chat.html   toggleHistoryPanel read its forceOpen argument only inside
                     the compact-viewport branch. On anything wider than
                     1100px it fell straight through to panel.classList.add
                     ('open') whatever it was passed — and the panel's own
                     close button calls toggleHistoryPanel(false). A × that
                     opened the panel it is drawn on.

  window_analytics   four functions the module docstring listed as its
                     post-execution entry points, with no caller outside
                     tests. Four passing tests made them look live, so a reader
                     had every reason to believe rolling averages were computed
                     there. They are not: the design routes window analytics
                     through SQL, which is what the module's own philosophy
                     note says.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))


class TestTheHistoryPanelCloseButtonCloses(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pytest.importorskip(
            "dukpy",
            reason="a JavaScript engine is required to EXECUTE the toggle; "
                   "reading it and assuming what it does is the defect")

    STUB = """
var _classes = {open: %s};
var panel = {classList: {
  add: function(c){_classes[c]=true;}, remove: function(c){_classes[c]=false;},
  toggle: function(c,on){_classes[c]= (on===undefined? !_classes[c] : !!on);},
  contains: function(c){return !!_classes[c];}}};
var btn = {classList:{add:function(){},remove:function(){},toggle:function(){}},
           setAttribute:function(){}};
var _historyLoaded = true;
function loadHistory(){}
var document = {getElementById: function(id){
  if (id==='historyPanel') return panel;
  if (id==='historyToggleBtn') return btn;
  return {focus:function(){}};
}, querySelector:function(){return null;}};
var window = {matchMedia: function(){return {matches: false};}};
"""

    def _toggle(self, argument, start_open):
        import dukpy

        from js_lift import function as lift

        source = (ROOT / "portal" / "templates" / "portal_chat.html").read_text(
            encoding="utf-8")
        body = ((self.STUB % ("true" if start_open else "false"))
                + lift(source, "function toggleHistoryPanel(forceOpen)")
                + f"\ntoggleHistoryPanel({argument}); JSON.stringify(_classes.open)")
        return json.loads(dukpy.evaljs(body))

    def test_false_closes_an_open_panel(self):
        # The finding: this returned True.
        self.assertFalse(self._toggle("false", start_open=True))

    def test_false_leaves_a_closed_panel_closed(self):
        self.assertFalse(self._toggle("false", start_open=False))

    def test_true_still_opens(self):
        """The control. A function that always closes would satisfy both tests
        above and break the button that opens the panel."""
        self.assertTrue(self._toggle("true", start_open=False))
        self.assertTrue(self._toggle("true", start_open=True))

    def test_no_argument_toggles(self):
        """The header button passes nothing, and it has to keep working."""
        self.assertTrue(self._toggle("undefined", start_open=False))
        self.assertFalse(self._toggle("undefined", start_open=True))

    def test_the_close_button_really_passes_false(self):
        """Ties the executed behaviour to the markup that depends on it."""
        source = (ROOT / "portal" / "templates" / "portal_chat.html").read_text(
            encoding="utf-8")
        close_button = next(line for line in source.splitlines()
                            if 'class="hp-close"' in line)
        self.assertIn("toggleHistoryPanel(false)", close_button)


class TestTheWindowAnalyticsDocstringIsHonest(unittest.TestCase):

    UNWIRED = ("compute_rolling_average", "compute_running_total",
               "compute_row_delta", "compute_rank_in_group")
    WIRED = ("detect_window_intent", "build_window_sql_hint")

    def _production_callers(self, name):
        skip = {"__pycache__", "venv", ".venv", ".git", "node_modules", "tests"}
        found = []
        for path in ROOT.rglob("*.py"):
            if skip & set(path.relative_to(ROOT).parts):
                continue
            if path.name == "window_analytics.py":
                continue
            if f"{name}(" in path.read_text(encoding="utf-8"):
                found.append(str(path.relative_to(ROOT)))
        return found

    def test_the_wired_two_really_are_wired(self):
        """The control: if these had no callers either, the module would be
        entirely dead and the docstring's whole premise wrong."""
        for name in self.WIRED:
            with self.subTest(name=name):
                self.assertTrue(self._production_callers(name),
                                f"{name} is advertised as wired and is not")

    def test_the_other_four_are_not_advertised_as_entry_points(self):
        """The docstring's "Entry points" list said they were, and four passing
        smoke tests agreed. Nothing calls them."""
        import core.window_analytics as window_analytics

        doc = window_analytics.__doc__ or ""
        entry_points = doc[doc.index("Entry points"):doc.index("NOT entry points")]
        for name in self.UNWIRED:
            with self.subTest(name=name):
                self.assertNotIn(name, entry_points)

    def test_and_the_docstring_says_plainly_that_nothing_calls_them(self):
        import core.window_analytics as window_analytics

        doc = window_analytics.__doc__ or ""
        self.assertIn("NOT entry points", doc)
        for name in self.UNWIRED:
            with self.subTest(name=name):
                self.assertIn(name, doc)

    def test_the_claim_is_still_true(self):
        """If someone wires one of them, this fails — and the docstring is then
        the thing that needs correcting, which is the point."""
        for name in self.UNWIRED:
            with self.subTest(name=name):
                self.assertEqual(
                    self._production_callers(name), [],
                    f"{name} now has a production caller; the docstring says "
                    f"it has none")

    def test_they_still_work_because_they_are_kept_on_purpose(self):
        from core.window_analytics import compute_running_total

        rows = [{"M": "Jan", "V": 10.0}, {"M": "Feb", "V": 5.0}]
        out = compute_running_total(rows, "V", "M")
        self.assertEqual([r.get("running_total") for r in out], [10.0, 15.0])


if __name__ == "__main__":
    unittest.main()
