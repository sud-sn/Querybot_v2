"""
tests/test_chat_toast_language.py

The chat page's interactive copy: toasts, composer drafts, the send/stop
button, the message actions and the feedback panel.

Executed rather than read. The catalogue is injected into the page as JSON, so
a source-text assertion now passes against a page that resolves the id to
nothing -- these run the page's own functions in duktape against the REAL
catalogue and assert on the DOM they produce.
"""

from __future__ import annotations

import pytest

from core import i18n
from tests.chat_js import run as run_js
from tests.chat_render import catalogue, render


DOM = """
const _nodes = {};
function _el(){
  return {textContent: '', value: '', title: '', className: '', innerHTML: '',
          disabled: false, style: {}, dataset: {}, _attrs: {},
          classList: {_c: {},
            add(n){ this._c[n] = true; },
            remove(n){ delete this._c[n]; },
            toggle(n, on){ if (on) this._c[n] = true; else delete this._c[n]; },
            contains(n){ return !!this._c[n]; }},
          setAttribute(k, v){ this._attrs[k] = String(v); },
          getAttribute(k){ return this._attrs[k] === undefined ? null : this._attrs[k]; },
          focus(){}};
}
for (const id of ['composerState', 'input', 'sendBtn', 'sendBtnIcon'])
  _nodes[id] = _el();
var document = {getElementById: id => _nodes[id] || null};
const _store = {};
var sessionStorage = {
  getItem: k => (k in _store ? _store[k] : null),
  setItem: (k, v) => { _store[k] = String(v); },
  removeItem: k => { delete _store[k]; },
};
const DRAFT_STORAGE_KEY = 'draft';
let wsReady = true;
function updateCharCount(){}
function stopQuery(){}
function sendMessage(){}
"""

COMPOSER_FUNCTIONS = (
    "function _setComposerState(message, tone = '')",
    "function _saveComposerDraft()",
    "function _restoreComposerDraft()",
    "function _setSendButtonMode(isStop)",
)


def _composer(script, lang, online=True):
    return run_js(
        (f"wsReady = {'true' if online else 'false'};\n" + script + """
JSON.stringify({state: _nodes.composerState.textContent,
                tone: _nodes.composerState.dataset.tone || '',
                title: _nodes.sendBtn.title,
                label: _nodes.sendBtn.getAttribute('aria-label')});"""),
        lang=lang, functions=COMPOSER_FUNCTIONS, preamble=DOM)


# ══════════════════════════════════════════════════════════════════════════════
# The composer's draft state
# ══════════════════════════════════════════════════════════════════════════════

class TestTheDraftState:

    def test_a_saved_draft_says_so_in_french(self):
        out = _composer("_nodes.input.value = 'marge par region';"
                        "_saveComposerDraft();", "fr")
        assert out["state"] == "Brouillon enregistré"

    def test_an_offline_draft_is_a_different_sentence(self):
        """Not "saved" with a warning tone bolted on: offline is a different
        promise, and the tone is what colours it."""
        out = _composer("_nodes.input.value = 'x';_saveComposerDraft();",
                        "fr", online=False)
        assert out["state"] == "Brouillon conservé hors ligne"
        assert out["tone"] == "warning"

    def test_english_is_unchanged(self):
        online = _composer("_nodes.input.value = 'x';_saveComposerDraft();", "en")
        offline = _composer("_nodes.input.value = 'x';_saveComposerDraft();",
                            "en", online=False)
        assert online["state"] == "Draft saved"
        assert offline["state"] == "Draft kept while offline"

    def test_an_empty_draft_clears_the_state_rather_than_labelling_it(self):
        out = _composer("_nodes.input.value = '   ';_saveComposerDraft();", "fr")
        assert out["state"] == ""

    def test_a_restored_draft_is_french(self):
        out = _composer("sessionStorage.setItem('draft', 'marge');"
                        "_restoreComposerDraft();", "fr")
        assert out["state"] == "Brouillon restauré"
        assert out["tone"] == "success"

    def test_the_draft_text_itself_is_never_touched(self):
        """It is the reader's own words."""
        out = run_js(
            "sessionStorage.setItem('draft', 'Draft saved');"
            "_restoreComposerDraft();"
            "JSON.stringify({value: _nodes.input.value});",
            lang="fr", functions=COMPOSER_FUNCTIONS, preamble=DOM)
        assert out["value"] == "Draft saved"


# ══════════════════════════════════════════════════════════════════════════════
# The send / stop button
# ══════════════════════════════════════════════════════════════════════════════

class TestTheSendButton:

    def test_send_is_french(self):
        out = _composer("_setSendButtonMode(false);", "fr")
        assert out["title"] == "Envoyer le message"
        assert out["label"] == "Envoyer le message"

    def test_stop_is_french(self):
        out = _composer("_setSendButtonMode(true);", "fr")
        assert out["title"] == "Arrêter la génération"

    def test_english_is_unchanged(self):
        assert _composer("_setSendButtonMode(true);", "en")["title"] == "Stop generating"
        assert _composer("_setSendButtonMode(false);", "en")["title"] == "Send message"

    def test_the_title_and_the_accessible_name_cannot_drift(self):
        """They were two separate ternaries over the same pair of literals."""
        for lang in ("en", "fr"):
            for stop in ("true", "false"):
                out = _composer(f"_setSendButtonMode({stop});", lang)
                assert out["title"] == out["label"]


# ══════════════════════════════════════════════════════════════════════════════
# Everything the page hands to toast()
# ══════════════════════════════════════════════════════════════════════════════

class TestTheToastCopyReachesTheBrowser:

    def test_the_toasts_are_french(self):
        fr = catalogue(render(lang="fr"))
        assert fr["ui.chat.toast.copied"] == "Copié dans le presse-papiers"
        assert fr["ui.chat.toast.copy_failed"] == "Échec de la copie"
        assert fr["ui.chat.toast.thread_unavailable"] == "Conversation indisponible"

    def test_the_long_ones_too(self):
        fr = catalogue(render(lang="fr"))
        assert fr["ui.chat.toast.offline_saved"].startswith("Vous êtes hors ligne.")
        assert fr["ui.chat.system.session_unavailable"].startswith(
            "Session de chat indisponible.")

    def test_english_is_unchanged(self):
        en = catalogue(render(lang="en"))
        assert en["ui.chat.toast.copied"] == "Copied to clipboard"
        assert en["ui.chat.system.action_timed_out"] == \
            "This action did not finish in time. Please retry it."

    def test_the_trimmed_paste_keeps_its_number(self):
        """A translation that drops {limit} leaves a warning that never says
        what the limit was."""
        assert i18n.placeholders("ui.chat.draft.trimmed") == {"limit"}
        assert "500" in i18n.t("ui.chat.draft.trimmed", lang="fr", limit=500)

    def test_no_toast_id_the_page_uses_is_missing_from_the_catalogue(self):
        """A missing id renders the raw "ui.chat.toast.x" in a toast, which is
        the least explicable place for one to appear."""
        import re
        from tests.chat_js import source

        used = set(re.findall(r"t\('(ui\.chat\.[a-z_.]+)'\)", source()))
        assert used, "the page resolves no chat ids"
        for msg_id in sorted(used):
            for lang in ("en", "fr"):
                value = i18n.t(msg_id, lang=lang)
                assert value != msg_id, (msg_id, lang)


# ══════════════════════════════════════════════════════════════════════════════
# The feedback panel
# ══════════════════════════════════════════════════════════════════════════════

class TestTheFeedbackPanel:

    def test_the_buttons_are_french(self):
        fr = catalogue(render(lang="fr"))
        assert fr["ui.chat.feedback.up"] == "Oui, réponse correcte"
        assert fr["ui.chat.feedback.down"] == "Non, quelque chose n'allait pas"
        assert fr["ui.chat.feedback.comments"] == "Commentaires supplémentaires (facultatif)"

    def test_the_thumbs_down_keeps_a_distinct_accessible_name(self):
        """The title says what the button MEANS and the accessible name says
        what it DOES. Collapsing them onto one id would have been the easy
        thing to do while translating."""
        fr = catalogue(render(lang="fr"))
        assert fr["ui.chat.feedback.not_helpful"] != fr["ui.chat.feedback.down"]

    def test_english_is_unchanged(self):
        en = catalogue(render(lang="en"))
        assert en["ui.chat.feedback.up"] == "Yes, correct answer"
        assert en["ui.chat.feedback.not_helpful"] == "Not helpful"


# ══════════════════════════════════════════════════════════════════════════════
# The answer card the browser draws
# ══════════════════════════════════════════════════════════════════════════════

class TestTheTrustDisclosure:

    def test_the_labels_are_french(self):
        fr = catalogue(render(lang="fr"))
        assert fr["ui.chat.trust.summary"] == "Comment cette réponse a été produite"
        assert fr["ui.chat.trust.rows"] == "Lignes"
        assert fr["ui.chat.trust.runtime"] == "Durée"
        assert fr["ui.chat.trust.result_scope"] == "Portée du résultat"

    def test_the_governance_claims_are_french(self):
        """These are the sentences that say what was and was not sent to a
        model. A reader who cannot read them cannot check them."""
        fr = catalogue(render(lang="fr"))
        assert fr["ui.chat.trust.planner_none"] == "Aucun appel au modèle"
        assert fr["ui.chat.trust.planner_none_detail"] == \
            "0 ligne de résultat transmise à un modèle"
        assert fr["ui.chat.trust.no_db_query"] == "Aucune requête à la base"

    def test_english_is_unchanged(self):
        en = catalogue(render(lang="en"))
        assert en["ui.chat.trust.summary"] == "How this answer was produced"
        assert en["ui.chat.trust.planner_none_detail"] == "0 result rows sent to any model"

    def test_the_placeholders_survive_translation(self):
        """A French sentence that drops {id} or {hash} silently stops being
        evidence of anything."""
        for msg_id, expected in (
            ("ui.chat.trust.from_result", {"id"}),
            ("ui.chat.trust.validated_code", {"hash"}),
            ("ui.chat.trust.ast_nodes", {"count", "input"}),
            ("ui.chat.trust.schema_named", {"schema"}),
            ("ui.chat.trust.grain", {"grain"}),
        ):
            assert i18n.placeholders(msg_id) == expected, msg_id
            for lang in ("en", "fr"):
                rendered = i18n.t(msg_id, lang=lang, **{k: "X" for k in expected})
                assert "{" not in rendered, (msg_id, lang)


class TestTheChartAndTableControls:

    def test_they_are_french(self):
        fr = catalogue(render(lang="fr"))
        assert fr["ui.chat.table.filter"] == "Filtrer les lignes…"
        assert fr["ui.chat.chart.expand"] == "Agrandir le graphique"
        assert fr["ui.chat.chart.pin"] == "Ajouter au tableau de bord"

    def test_the_empty_chart_notices_are_french(self):
        fr = catalogue(render(lang="fr"))
        assert fr["ui.chat.chart.no_rows"] == "Aucune ligne à tracer"
        assert fr["ui.chat.chart.no_value_column"] == "Aucune colonne de valeurs à tracer"

    def test_the_pin_label_matches_the_dialog_it_opens(self):
        """The chart's button and the modal's title are the same promise, and
        a reader who sees two different phrasings has to work out whether they
        are the same action."""
        fr = catalogue(render(lang="fr"))
        assert fr["ui.chat.chart.pin"] == fr["ui.pin.title"]

    def test_english_is_unchanged(self):
        en = catalogue(render(lang="en"))
        assert en["ui.chat.table.filter"] == "Filter rows…"
        assert en["ui.chat.chart.pin"] == "Add to dashboard"


class TestTheFeedbackReasons:

    REASONS = ("other", "wrong_metric", "wrong_dimension", "wrong_filter",
               "wrong_join", "wrong_data", "incomplete", "confusing",
               "expected_data_missing")

    def test_every_reason_is_translated(self):
        fr = catalogue(render(lang="fr"))
        for reason in self.REASONS:
            value = fr[f"ui.chat.reason.{reason}"]
            assert value and value != i18n.t(f"ui.chat.reason.{reason}", lang="en"), reason

    def test_the_option_values_are_untranslated_wire_enums(self):
        """store/learning_store.py groups feedback by this value. A translated
        one silently splits a French tenant's reasons into their own buckets."""
        markup = render(lang="fr")
        for reason in self.REASONS:
            assert f"'{reason}'" in markup, reason
        assert "wrong_metric" in markup
