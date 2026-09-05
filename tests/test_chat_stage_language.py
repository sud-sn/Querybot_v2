"""
tests/test_chat_stage_language.py

The live stage trail and the connection status, in the reader's language.

These are the sentences a reader watches while an answer is being built, so
they are the most-read copy in the product after the answer itself.

The pipeline half is executed against the real _send_live_stage call sites; the
browser half is executed in duktape against the page's own functions and the
REAL catalogue -- so an id the page uses and the catalogue does not have fails
here rather than rendering as a raw id on a customer's screen.
"""

from __future__ import annotations

import pytest

from core import i18n
from tests.chat_js import run as run_js


@pytest.fixture
def french():
    token = i18n.activate_language("fr")
    try:
        yield
    finally:
        i18n.deactivate_language(token)


# ══════════════════════════════════════════════════════════════════════════════
# What the pipeline pushes
# ══════════════════════════════════════════════════════════════════════════════

def _stage_pushes():
    """Every _send_live_stage call in the pipeline, as (stage, label, detail)
    argument NODES.

    An AST walk rather than seventeen fixtures: the call sites are scattered
    through handle_query and reaching them means running a whole governed
    answer. What has to hold for all of them is structural -- the label and the
    detail come from the catalogue and not from a literal -- and this is the
    only way to check every one.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    pushes = []
    for module in ("core/query_pipeline.py", "core/result_renderer.py"):
        tree = ast.parse((root / module).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", "") or getattr(node.func, "attr", "")
            if name != "_send_live_stage":
                continue
            pushes.append((module, node.lineno, node.args[2:5]))
    return pushes


class TestEveryStagePushGoesThroughTheCatalogue:

    def test_there_are_call_sites_to_check(self):
        """Guards the walk itself: a rename would otherwise make every
        assertion below pass over an empty list."""
        assert len(_stage_pushes()) >= 15

    def test_no_call_site_passes_an_english_literal(self):
        """One missed call site is one stage that stays English mid-answer, in
        the middle of a French trail, and nothing else would notice."""
        import ast
        offenders = []
        for module, line, args in _stage_pushes():
            for arg in args[1:]:      # label and detail; args[0] is the stage key
                if isinstance(arg, ast.Constant) and str(arg.value).strip():
                    offenders.append(f"{module}:{line} {arg.value!r}")
        assert not offenders, offenders

    def test_every_call_site_resolves_to_real_copy_in_both_languages(self):
        """An id with no catalogue entry renders as "stage.x.label" on screen,
        which is exactly the failure the literals above could not have."""
        import ast
        for module, line, args in _stage_pushes():
            for arg in args[1:]:
                assert isinstance(arg, ast.Call), f"{module}:{line}"
                msg_id = arg.args[0].value
                for lang in ("en", "fr"):
                    value = i18n.t(msg_id, lang=lang)
                    assert value and value != msg_id, (module, line, msg_id, lang)

    def test_the_stage_key_stays_a_literal(self):
        """The page picks the animated mark's state from it and the trail keys
        repetition on it. A translated key is a dead animation."""
        import ast
        for module, line, args in _stage_pushes():
            assert isinstance(args[0], ast.Constant), f"{module}:{line}"
            assert args[0].value.islower(), f"{module}:{line}"


class TestTheStageCopyItself:

    def test_a_stage_is_french(self):
        assert i18n.t("stage.executing_query.label", lang="fr") == "Exécution de la requête"
        assert i18n.t("stage.executing_query.detail", lang="fr") == \
            "Exécution du SQL sur votre source de données connectée."

    def test_english_is_unchanged(self):
        assert i18n.t("stage.executing_query.label", lang="en") == "Running query"
        assert i18n.t("stage.executing_query.detail", lang="en") == \
            "Executing the SQL against your connected data source."

    def test_the_active_language_is_what_the_pipeline_reads(self, french):
        """The pipeline calls _t with no lang, so the ContextVar handle_query
        activates is what decides. A test that always passed lang= would never
        notice that activation going missing."""
        from core.i18n import t
        assert t("stage.authorization.label") == "Vérification des accès"

    def test_every_stage_the_page_knows_has_copy_in_both_languages(self):
        """The page's STATUS_FALLBACK enumerates the stages it can show. Any
        one of them missing from the catalogue renders as "stage.x.label"."""
        keys = run_js("JSON.stringify({keys: Object.keys(STATUS_FALLBACK)});",
                      consts=["STATUS_FALLBACK"])["keys"]
        assert keys, "the page knows no stages"
        for key in keys:
            for suffix in ("label", "detail"):
                for lang in ("en", "fr"):
                    value = i18n.t(f"stage.{key}.{suffix}", lang=lang)
                    assert value and not value.startswith("stage."), (key, suffix, lang)

    def test_the_pipeline_and_the_page_agree_on_one_sentence(self):
        """They each had their own copy for executing_query -- "Executing the
        SQL against your connected data source" against "Executing the query
        against your connected database". Same stage, two sentences, and no way
        to tell which one a reader saw."""
        fallback = run_js("JSON.stringify({f: STATUS_FALLBACK});",
                          consts=["STATUS_FALLBACK"])["f"]
        assert fallback["executing_query"] == [
            i18n.t("stage.executing_query.label", lang="en"),
            i18n.t("stage.executing_query.detail", lang="en"),
        ]


# ══════════════════════════════════════════════════════════════════════════════
# What the page shows
# ══════════════════════════════════════════════════════════════════════════════

STAGE_FUNCTIONS = (
    "function _renderStageTrail()",
    "function _updateStageSteps(stage, label, detail)",
)

STAGE_PREAMBLE = """
const _nodes = {};
function _el() {
  return {innerHTML: '', textContent: '', hidden: true};
}
for (const id of ['answerProgressSteps', 'stageLabel', 'answerStageLabel',
                  'answerStageDetail']) _nodes[id] = _el();
var document = {getElementById: id => _nodes[id] || null};
function escHtml(s){ return String(s == null ? '' : s); }
const BRAND_STAGE_STATES = {};
function _setStageState() {}
let _completedStages = []; let _currentStage = null;
"""


def _stage_head(script, lang):
    return run_js(
        script + """
JSON.stringify({label: _nodes.answerStageLabel.textContent,
                detail: _nodes.answerStageDetail.textContent,
                composer: _nodes.stageLabel.textContent});""",
        lang=lang, consts=["STATUS_FALLBACK"],
        functions=STAGE_FUNCTIONS, preamble=STAGE_PREAMBLE)


class TestTheTrailRendersInTheReadersLanguage:

    def test_a_labelless_frame_falls_back_in_french(self):
        """The server sends the label; when it does not, the page has to fill
        in, and its filler is the reader's language too."""
        out = _stage_head("_updateStageSteps('generating_sql', '', '');", "fr")
        assert out["label"] == "Génération de la requête"
        assert out["detail"] == "Traduction de la question métier en SQL."

    def test_english_is_unchanged(self):
        out = _stage_head("_updateStageSteps('generating_sql', '', '');", "en")
        assert out["label"] == "Generating query"
        assert out["detail"] == "Translating the business question into SQL."

    def test_an_unknown_stage_still_says_something_useful(self):
        """A stage the page has never heard of is the case where a raw id on
        screen would be most likely and least explicable."""
        out = _stage_head("_updateStageSteps('teleporting', '', '');", "fr")
        assert out["label"] == "Traitement de votre réponse"
        assert out["detail"] == "Préparation d'une réponse fiable."

    def test_a_server_label_still_wins(self):
        """The fallback is a fallback. The pipeline sends the specific
        sentence -- "Retrying query" rather than "Running query" -- and that
        must not be replaced by the generic one for the stage."""
        out = _stage_head(
            "_updateStageSteps('executing_query', 'Nouvelle tentative', 'x');", "fr")
        assert out["label"] == "Nouvelle tentative"

    def test_the_composer_label_gets_the_ellipsis_in_both(self):
        out = _stage_head("_updateStageSteps('generating_sql', '', '');", "fr")
        assert out["composer"] == "Génération de la requête…"


# ══════════════════════════════════════════════════════════════════════════════
# The run badge and the composer mascot
# ══════════════════════════════════════════════════════════════════════════════

RUN_PREAMBLE = """
const _nodes = {};
function _el(){ return {textContent: '', hidden: true, dataset: {}}; }
for (const id of ['agentRunMeta', 'agentRunState', 'agentRunTool', 'agentRunId'])
  _nodes[id] = _el();
var document = {getElementById: id => _nodes[id] || null};
function _updateStageSteps() {}
"""


def _run_badge(msg, lang):
    return run_js(
        f"_applyAgentRunEvent({msg});"
        "JSON.stringify({state: _nodes.agentRunState.textContent,"
        " tool: _nodes.agentRunTool.textContent,"
        " status: _nodes.agentRunMeta.dataset.status});",
        lang=lang, functions=("function _applyAgentRunEvent(msg)",),
        preamble=RUN_PREAMBLE)


class TestTheRunBadge:

    def test_the_states_are_french(self):
        assert _run_badge("{run_status:'blocked'}", "fr")["state"] == \
            "Bloqué par la politique"
        assert _run_badge("{run_status:'completed'}", "fr")["state"] == \
            "Réponse gouvernée"

    def test_english_is_unchanged(self):
        assert _run_badge("{run_status:'blocked'}", "en")["state"] == "Blocked by policy"
        assert _run_badge("{run_status:'cancelled'}", "en")["state"] == "Run cancelled"

    def test_an_unknown_status_falls_back_rather_than_showing_an_id(self):
        out = _run_badge("{run_status:'teleporting'}", "fr")
        assert out["state"] == "Agent gouverné"
        assert "ui.chat.run" not in out["state"]

    def test_the_wire_status_is_kept_as_sent(self):
        """The badge's colour is chosen from data-status in CSS."""
        assert _run_badge("{run_status:'blocked'}", "fr")["status"] == "blocked"

    def test_the_tool_name_is_never_translated(self):
        """It is a governed tool id, not copy."""
        out = _run_badge("{run_status:'running',tool:'query_data',read_only:true}", "fr")
        assert out["tool"] == "query_data · lecture seule"

    def test_english_read_only_is_unchanged(self):
        out = _run_badge("{run_status:'running',tool:'query_data',read_only:true}", "en")
        assert out["tool"] == "query_data · read only"


# ══════════════════════════════════════════════════════════════════════════════
# The connection status
# ══════════════════════════════════════════════════════════════════════════════

class TestTheConnectionCopyReachesTheBrowser:

    def _catalogue(self, lang):
        from tests.chat_render import render, catalogue
        return catalogue(render(lang=lang))

    def test_the_states_are_french(self):
        fr = self._catalogue("fr")
        assert fr["ui.chat.connected"] == "Connecté"
        assert fr["ui.chat.session_active"] == "Session en direct active"
        assert fr["ui.chat.reconnecting"] == "Reconnexion…"

    def test_the_retry_countdown_keeps_its_placeholder(self):
        """A translation that drops {seconds} leaves a sentence that never
        tells the reader how long they are waiting."""
        assert "{seconds}" in self._catalogue("fr")["ui.chat.connection_lost_retry"]
        assert i18n.placeholders("ui.chat.connection_lost_retry") == {"seconds"}

    def test_english_is_unchanged(self):
        en = self._catalogue("en")
        assert en["ui.chat.connected"] == "Connected"
        assert en["ui.chat.session_active"] == "Live session active"
