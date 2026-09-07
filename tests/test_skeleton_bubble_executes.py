"""
tests/test_skeleton_bubble_executes.py

The chat page threw a TypeError on the first status frame of every question.

`_showSkeletonBubble` opened with

    const t = thread();

which shadows the translator `t(id, vars)` declared at the top of the same
script. Two lines later the function calls `t('ui.chat.delivery.understanding')`
-- by then `t` is the thread <div>, so the call is a TypeError. The skeleton
bubble never appeared, and `startDoodle()` and the elapsed timer, both below
it in `setProcessing`, never ran either.

Nothing caught it. The page PARSES fine, so the whole-block parse guard in
tests/test_template_scripts_parse.py is blind to it. And the test whose stated
job is proving this function's behaviour -- tests/test_stop_query.py:250 --
lifts `setProcessing`'s source text and asserts strings against it without
executing anything, which is the practice the owner's standing rule forbids
and the exact reason a runtime error shipped green.

So this file EXECUTES it. duktape runs the real function lifted from the real
template against the real message catalogue, with a DOM shim small enough to
read. A function-scope shadow is invisible to a parser and obvious to an
interpreter, so this is the only kind of test that can see it.
"""

from __future__ import annotations

import json
import unittest

import pytest

from tests.chat_js import run as run_js

dukpy = pytest.importorskip("dukpy", reason="template JS execution needs duktape")


# A DOM small enough to audit. Only what _showSkeletonBubble touches: an
# element registry, createElement, appendChild, and querySelector returning
# nothing so the brand-cloning branch is skipped.
DOM = """
var _byId = {};
function _mkEl(tag){
  return {
    tagName: tag, id: '', className: '', innerHTML: '',
    dataset: {}, children: [],
    appendChild: function(child){
      this.children.push(child);
      if (child.id) { _byId[child.id] = child; }
      return child;
    },
    querySelector: function(){ return null; },
    cloneNode: function(){ return _mkEl(this.tagName); },
    setAttribute: function(){},
    classList: { add: function(){}, remove: function(){}, toggle: function(){} }
  };
}
var _thread = _mkEl('div');
_thread.id = 'thread';
_byId['thread'] = _thread;

var document = {
  getElementById: function(id){ return _byId[id] || null; },
  createElement: function(tag){ return _mkEl(tag); }
};
function escHtml(s){ return String(s); }
function scrollBottom(){ }
"""


def _show_bubble(lang="en"):
    """Run the real _showSkeletonBubble and report what it produced."""
    return run_js(
        """
        var threw = null;
        try { _showSkeletonBubble(); }
        catch (e) { threw = (e && e.name ? e.name : 'Error') + ': ' + (e && e.message); }
        var bubble = document.getElementById('skeletonBubble');
        JSON.stringify({
          threw: threw,
          appended: !!bubble,
          html: bubble ? bubble.innerHTML : ''
        });
        """,
        lang=lang,
        functions=("function thread(", "function _showSkeletonBubble("),
        preamble=DOM,
    )


class TheSkeletonBubbleActuallyRuns(unittest.TestCase):

    def test_it_does_not_throw(self):
        """The whole defect, in one assertion. Before the rename this was
        `TypeError: t is not a function`."""
        result = _show_bubble()
        self.assertIsNone(
            result["threw"],
            f"_showSkeletonBubble threw: {result['threw']}",
        )

    def test_the_bubble_reaches_the_thread(self):
        """Not merely 'did not throw' -- the element has to be appended.
        The throw happened while building innerHTML, before appendChild."""
        self.assertTrue(_show_bubble()["appended"])

    def test_the_labels_are_translated_not_raw_ids(self):
        """Proves `t` resolved to the translator rather than to anything else
        that happens not to throw. A shadow that returned a string would pass
        the two tests above and render the raw id on screen."""
        html = _show_bubble()["html"]
        self.assertNotIn("ui.chat.delivery.understanding", html)
        self.assertIn("answerStageLabel", html)
        self.assertIn("answerStageDetail", html)

    def test_it_translates_for_a_french_reader(self):
        english = _show_bubble("en")["html"]
        french = _show_bubble("fr")["html"]
        self.assertNotIn("ui.chat.delivery.understanding", french)
        self.assertNotEqual(english, french,
                            "the two languages rendered identical markup")

    def test_calling_it_twice_appends_one_bubble(self):
        """The early return guard, executed rather than read."""
        result = run_js(
            """
            var threw = null;
            try { _showSkeletonBubble(); _showSkeletonBubble(); }
            catch (e) { threw = String(e && e.message); }
            JSON.stringify({threw: threw, count: _thread.children.length});
            """,
            functions=("function thread(", "function _showSkeletonBubble("),
            preamble=DOM,
        )
        self.assertIsNone(result["threw"])
        self.assertEqual(result["count"], 1)


class NoFunctionShadowsTheTranslator(unittest.TestCase):
    """The rename fixes one site. This stops the next one.

    A source scan by necessity -- the collision is legal JavaScript and only
    fails when the function is reached, so there is nothing to execute for a
    function no test drives. The tests above execute the one that shipped
    broken; this one covers the rest of the file.
    """

    def test_no_local_binding_called_t_in_a_function_that_calls_t(self):
        import re
        from tests.chat_js import source

        # Comments are stripped first, blanked rather than removed so line
        # numbers still point at the real code. Without this the scan flags
        # the comment above the fix, which quotes the broken line verbatim.
        lines = [
            "" if line.lstrip().startswith("//") else line
            for line in source().splitlines()
        ]
        starts = [i for i, line in enumerate(lines)
                  if re.match(r"\s*(?:async\s+)?function\s+\w+\s*\(", line)]
        starts.append(len(lines))

        offenders = []
        for start, end in zip(starts, starts[1:]):
            body = lines[start:end]
            name = re.match(r"\s*(?:async\s+)?function\s+(\w+)", body[0]).group(1)
            shadows = any(re.search(r"\b(?:const|let|var)\s+t\s*=", line)
                          for line in body)
            calls = any(re.search(r"[^\w.]t\(\s*['\"]", line) for line in body)
            if shadows and calls:
                offenders.append(f"{name}() at line {start + 1}")

        self.assertEqual(
            offenders, [],
            "a local `t` shadows the translator in a function that calls "
            f"t('...'): {offenders}",
        )


if __name__ == "__main__":
    unittest.main()
