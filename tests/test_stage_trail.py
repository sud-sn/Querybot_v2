"""
tests/test_stage_trail.py

The pipeline reports about sixteen distinct stages while it works. The card
showed exactly one of them and then forgot it: every status frame overwrote one
label and one detail, so a user watching a slow query learnt nothing about
where the time went.

The scaffolding for the list version was already in the template and entirely
dead -- STAGE_ORDER was declared and never referenced, _completedStages was
only ever reset and never pushed to, _currentStage was assigned and never read.
This is that list, wired.

These tests EXECUTE the real functions, lifted out of the real template by
brace balance and run in a JavaScript interpreter, rather than asserting that
strings appear in the file. A source scan would have been satisfied by the dead
declarations that started all this.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

dukpy = pytest.importorskip(
    "dukpy",
    reason="a JavaScript engine is required to EXECUTE the template's own "
           "functions; asserting on their source text instead is the failure "
           "mode this file exists to avoid",
)

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "portal" / "templates" / "portal_chat.html"


def _function(source: str, signature: str) -> str:
    """A whole function by brace balance.

    Not a character window: a window stops covering the code it names the
    moment anything is inserted above it, which has already broken one test in
    this suite.
    """
    start = source.index(signature)
    depth = 0
    for i in range(source.index("{", start), len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start:i + 1]
    raise AssertionError(f"unbalanced braces after {signature!r}")


def _run(frames: list[tuple[str, str, str]]) -> dict:
    """Feed status frames through the template's own stage functions."""
    from core import i18n
    tmpl = TEMPLATE.read_text(encoding="utf-8")
    catalogue = json.dumps(i18n.catalogue_for("en"))
    harness = f"""
// A DOM thin enough for these three functions and no thinner.
const _nodes = {{}};
function _el() {{
  // hidden: true mirrors the real markup, which creates the container as
  // `<div class="answer-progress-steps" id="answerProgressSteps" hidden>`.
  // Starting it visible made the harness disagree with the page and produced
  // two failures that were the fixture's fault, not the code's.
  return {{ innerHTML: '', hidden: true, textContent: '' }};
}}
for (const id of ['answerProgressSteps', 'stageLabel', 'answerStageLabel', 'answerStageDetail'])
  _nodes[id] = _el();
var document = {{ getElementById: id => _nodes[id] || null }};
function escHtml(s) {{
  return String(s == null ? '' : s).replace(/[&<>"']/g,
    c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
}}
const STATUS_FALLBACK = {{}};
// The generic fallback label goes through the catalogue now. The REAL one, so
// this suite notices an id the page uses and the catalogue does not have.
const I18N = {catalogue};
function t(id, vars) {{
  let out = Object.prototype.hasOwnProperty.call(I18N, id) ? I18N[id] : id;
  if (vars) for (const k in vars) out = out.split('{{' + k + '}}').join(String(vars[k]));
  return out;
}}
const BRAND_STAGE_STATES = {{}};
function _setStageState() {{}}
let _completedStages = [];
let _currentStage = null;

{_function(tmpl, "function _renderStageTrail()")}
{_function(tmpl, "function _updateStageSteps(stage, label, detail)")}
{_function(tmpl, "function _resetStageSteps()")}

for (const f of {json.dumps(frames)}) _updateStageSteps(f[0], f[1], f[2]);
JSON.stringify({{
  html: _nodes.answerProgressSteps.innerHTML,
  hidden: _nodes.answerProgressSteps.hidden,
  trailLength: _completedStages.length,
  current: _currentStage,
  head: {{ label: _nodes.answerStageLabel.textContent,
           detail: _nodes.answerStageDetail.textContent }},
  composer: _nodes.stageLabel.textContent,
}});
"""
    return json.loads(dukpy.evaljs(harness))


PIPELINE = [
    ("authorization", "Checking access", "Confirming which tables you may query."),
    ("metric_registry", "Using known metric", "Revenue is approved on CUS_ORD_IVC_FCT."),
    ("generating_sql", "Generating query", ""),
    ("validating_sql", "Checking query safety", "Schema, access and join path."),
    ("repairing_query", "Repairing query", "One column did not resolve."),
    ("validating_sql", "Checking query safety", "Re-checking after the repair."),
    ("executing_query", "Running query", "Against your warehouse."),
]


class TestTheTrailAccumulates:
    def test_every_finished_stage_is_kept(self):
        """The whole point. Sixteen stages used to leave one line behind."""
        result = _run(PIPELINE)
        assert result["trailLength"] == len(PIPELINE) - 1, (
            "every stage but the running one belongs in the trail"
        )

    def test_the_running_stage_is_in_the_head_and_not_in_the_trail(self):
        """No entry appears twice: the head is 'now', the trail is 'done'."""
        result = _run(PIPELINE)
        assert result["head"]["label"] == "Running query"
        assert result["html"].count("Running query") == 0

    def test_a_repeated_stage_appears_once_per_occurrence(self):
        """validate -> repair -> validate is the most interesting thing this
        can show. Keying entries by stage name would collapse it into one line
        and hide that a repair happened at all."""
        result = _run(PIPELINE)
        assert result["html"].count("Checking query safety") == 2

    def test_an_empty_trail_stays_hidden(self):
        result = _run([])
        assert result["hidden"] is True
        assert result["html"] == ""

    def test_the_first_stage_alone_leaves_the_trail_empty(self):
        """One stage in flight is not yet a history of anything."""
        result = _run(PIPELINE[:1])
        assert result["trailLength"] == 0
        assert result["hidden"] is True


class TestDisclosureIsEarned:
    def test_a_stage_with_detail_gets_a_chevron(self):
        result = _run(PIPELINE)
        assert "<details" in result["html"]
        assert "Confirming which tables you may query." in result["html"]

    def test_a_stage_with_no_detail_is_a_plain_row(self):
        """A chevron that opens onto nothing is worse than no chevron. The head
        still shows filler text so it never looks half-rendered, but the trail
        keeps the RAW detail -- so a stage that said nothing extra does not earn
        a disclosure."""
        result = _run(PIPELINE)
        segment = result["html"]
        idx = segment.index("Generating query")
        # Walk back to the element that contains it.
        opening = segment.rfind("<", 0, idx)
        assert segment[opening:idx].startswith("<span"), segment[opening - 80:idx]
        assert "stage-step--plain" in segment[:idx]

    def test_the_head_still_gets_a_subtitle_when_the_stage_gave_none(self):
        result = _run(PIPELINE[:3])
        assert result["head"]["label"] == "Generating query"
        assert result["head"]["detail"], "the head must not look half-rendered"


class TestItIsSafeToRender:
    def test_a_label_carrying_markup_is_escaped(self):
        """Labels and details originate server-side and are interpolated into
        innerHTML."""
        result = _run([
            ("a", "<img src=x onerror=alert(1)>", "d1"),
            ("b", "second", "<script>alert(2)</script>"),
            ("c", "third", "d3"),
        ])
        assert "<img" not in result["html"] and "&lt;img" in result["html"]
        assert "<script>" not in result["html"] and "&lt;script&gt;" in result["html"]

    def test_a_missing_container_does_not_throw(self):
        """The trail renders into the skeleton bubble, which does not exist
        between questions -- so every status frame outside a turn would throw."""
        from core import i18n
        tmpl = TEMPLATE.read_text(encoding="utf-8")
        catalogue = json.dumps(i18n.catalogue_for("en"))
        harness = f"""
var document = {{ getElementById: () => null }};
function escHtml(s) {{ return String(s == null ? '' : s); }}
const STATUS_FALLBACK = {{}}; const BRAND_STAGE_STATES = {{}};
const I18N = {catalogue};
function t(id) {{ return Object.prototype.hasOwnProperty.call(I18N, id) ? I18N[id] : id; }}
function _setStageState() {{}}
let _completedStages = []; let _currentStage = null;
{_function(tmpl, "function _renderStageTrail()")}
{_function(tmpl, "function _updateStageSteps(stage, label, detail)")}
{_function(tmpl, "function _resetStageSteps()")}
_updateStageSteps('a', 'one', 'x');
_updateStageSteps('b', 'two', 'y');
_resetStageSteps();
'ok';
"""
        assert dukpy.evaljs(harness) == "ok"


class TestTheDeadScaffoldingIsAliveNow:
    """These three were declared when the card was built and never read. The
    point of this file is that they carry the feature now, so a future reader
    does not delete them as unused a second time."""

    def test_the_trail_state_is_actually_consumed(self):
        source = TEMPLATE.read_text(encoding="utf-8")
        render = _function(source, "function _renderStageTrail()")
        update = _function(source, "function _updateStageSteps(stage, label, detail)")
        assert "_completedStages" in render, "the trail is rendered from the array"
        assert "_completedStages.push" in update, "and the array is appended to"
        assert "_currentStage" in update

    def test_the_skeleton_bars_are_gone(self):
        """Three grey bars stood in for 'something is happening'. The trail says
        what is happening, so they have nothing left to stand in for."""
        source = TEMPLATE.read_text(encoding="utf-8")
        skeleton = re.search(r"answer-progress-skeleton\"><span", source)
        assert skeleton is None, "the placeholder bars are still being rendered"
        assert 'id="answerProgressSteps"' in source
