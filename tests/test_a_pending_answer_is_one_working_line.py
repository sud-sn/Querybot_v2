"""A pending answer is one working line: the wave, what it is doing, the seconds, and Stop.

The approved redesign (the logo sheet's "Working" state): "Shown only on the
pending answer, next to 'Working out the answer · 4 s' and a Stop button." The
page still drew a card: "Understanding your question" over the pipeline's
subtitle, a "Governed agent · query_data · read only" badge with a run id, and
the seconds only after eight of them. The badge and the subtitle are the
pipeline's words, not the reader's; they stay in the markup for the code that
fills them and are never shown.

Run through the page's own functions (tests/chat_js.py).
"""

from __future__ import annotations

import re

import pytest

from tests.chat_js import run as run_js
from tests.test_skeleton_bubble_executes import _show_bubble

pytest.importorskip("dukpy", reason="template JS execution needs duktape")

PAGE = open("portal/templates/portal_chat.html", encoding="utf-8").read()


def _line(lang="en") -> str:
    out = _show_bubble(lang)
    assert out["threw"] is None, out["threw"]
    return out["html"]


def test_it_says_it_is_working_out_the_answer_and_what_on():
    html = _line()
    assert "Working out the answer</b> · " in html
    assert re.search(r'id="answerStageLabel">Understanding your question<', html)


def test_it_offers_stop():
    stop = re.search(r'<button[^>]*answer-progress-stop[^>]*>.*?</button>', _line(), re.S).group(0)
    assert 'onclick="stopQuery()"' in stop and "Stop" in stop


def test_in_french_too():
    html = _line("fr")
    assert "Je prépare la réponse" in html and "Arrêter" in html


def test_the_pipeline_s_words_are_not_shown():
    html = _line()
    assert re.search(r'id="answerStageDetail" hidden', html)
    assert re.search(r'id="agentRunMeta" hidden', html)


def test_a_run_event_leaves_the_badge_hidden():
    preamble = """
var _nodes = {};
function _el(){ return {textContent: '', hidden: true, dataset: {}}; }
['agentRunMeta', 'agentRunState', 'agentRunTool', 'agentRunId'].forEach(function(id){ _nodes[id] = _el(); });
var document = {getElementById: function(id){ return _nodes[id] || null; }};
function _updateStageSteps() {}
"""
    out = run_js(
        "_applyAgentRunEvent({run_status: 'running', tool: 'query_data', read_only: true, run_id: 'abcdef123'});"
        "JSON.stringify({hidden: _nodes.agentRunMeta.hidden, status: _nodes.agentRunMeta.dataset.status});",
        lang="en", functions=("function _applyAgentRunEvent(msg)",), preamble=preamble)
    assert out == {"hidden": True, "status": "running"}


def test_the_seconds_show_from_the_third():
    timer = PAGE[PAGE.index("processingElapsedTimer = setInterval"):]
    timer = timer[:timer.index("}, 1000);")]
    assert "elapsed >= 3" in timer and "t('ui.chat.seconds'" in timer
