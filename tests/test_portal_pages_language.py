"""
tests/test_portal_pages_language.py

The six portal pages outside chat and the dashboard: sign-in, registration,
change-password, pin-confirm, new-report, notifications and the Semantic Layer.

Every test renders the real template through the portal's own environment. The
last class sweeps all of them at once, because the failure mode for a set of
small pages is not a bad translation -- it is one page nobody remembered.
"""

from __future__ import annotations

import re

import pytest

from core import i18n
from tests.portal_render import render, unescaped, visible

USER = {"id": 1, "name": "Ada Lovelace", "account_id": "a",
        "role": "analyst", "group_name": "Analysts"}

KB_TABLE = {
    "table": "SALES", "schema": "DW", "fqn": "DW.SALES", "field_count": 2,
    "confidence": 91, "overview": "",
    "fields": [{"column": "REVENUE", "type": "decimal", "nullable": "",
                "distinct_values": "", "meaning": "m", "use_case": "u",
                "synonyms": [], "confidence": 88, "pending": True,
                "approved": False, "needs_context": False}],
}

# Every page, with just enough context to render it. The sweep at the bottom
# walks this table, so a page added to the portal and not added here is a page
# this file cannot check -- which is the point of test_the_sweep_covers_them_all.
PAGES = {
    "portal_login.html": dict(error=""),
    "portal_register.html": dict(token="tok", client_name="Acme", error=""),
    "portal_change_password.html": dict(forced=False, error=""),
    "portal_pin_confirm.html": dict(
        token="t", question="revenue by region", sql="SELECT 1", error="",
        dashboards=[{"id": 1, "name": "Ops", "chart_count": 1},
                    {"id": 2, "name": "Finance", "chart_count": 4}]),
    "portal_report_new.html": dict(
        error="", all_metrics=[{"id": 1, "name": "Revenue", "base_table": "DW.SALES"}]),
    "portal_notifications.html": dict(
        user=USER, saved=False, error="", alerts=[], my_reports=[], reports=[],
        subscriptions={}),
    "portal_kb.html": dict(
        user=USER, pending_count=3, saved=False, semantic_tables=[KB_TABLE],
        schemas=["DW"], selected_schema="DW", visible_tables=[KB_TABLE]),
}


def _page(name, lang, **overrides):
    context = dict(PAGES[name])
    context.setdefault("user", None)
    context.update(overrides)
    return render(name, lang=lang, path="/portal/x", **context)


# ══════════════════════════════════════════════════════════════════════════════
# Sign-in — the first screen, and the only one a reader sees before any
# preference exists to read
# ══════════════════════════════════════════════════════════════════════════════

class TestTheSignInPage:

    def test_it_is_french(self):
        markup = visible(_page("portal_login.html", "fr"))
        assert "Espace sécurisé d'intelligence des données" in markup
        assert "Identifiant du compte" in markup
        assert "Se connecter" in markup

    def test_english_is_unchanged(self):
        markup = visible(_page("portal_login.html", "en"))
        assert "Secure data intelligence workspace" in markup
        assert "Account ID" in markup
        assert "Sign in" in markup

    def test_the_browser_preference_decides_before_sign_in(self):
        """There is no account yet, so Accept-Language is the only signal. A
        French customer must not have to sign in through an English page to
        set a preference that would then have made it French."""
        from tests.portal_render import Req, render as _render
        request = Req("/portal/login")
        request.headers = {"accept-language": "fr-FR,fr;q=0.9"}
        markup = visible(_render("portal_login.html", request=request,
                                 user=None, error=""))
        assert "Se connecter" in markup

    def test_the_reveal_button_relabels_itself_in_french(self):
        """The label is swapped in JavaScript, so translating the markup alone
        leaves it English the moment anyone clicks it."""
        markup = _page("portal_login.html", "fr")
        assert "window.qbT(show ? 'ui.auth.hide_password' : 'ui.auth.show_password')" in markup
        assert i18n.t("ui.auth.hide_password", lang="fr") == "Masquer le mot de passe"


# ══════════════════════════════════════════════════════════════════════════════
# The forms
# ══════════════════════════════════════════════════════════════════════════════

class TestTheRegistrationPage:

    def test_it_is_french(self):
        markup = visible(_page("portal_register.html", "fr"))
        assert "Créez votre compte" in markup
        assert "Lien d'inscription à usage unique" in markup

    def test_the_client_name_is_the_customer_s(self):
        markup = visible(_page("portal_register.html", "fr", client_name="Acme"))
        assert "Configuration de votre compte pour Acme" in markup

    def test_the_expired_link_help_is_french(self):
        markup = visible(_page("portal_register.html", "fr", token="", error="Expired"))
        assert "Écrivez à votre QueryBot dans Zoom" in markup


class TestTheChangePasswordPage:

    def test_the_two_states_are_different_sentences(self):
        """Forced and voluntary are different situations, and the page said so
        in English with two branches. It still does."""
        forced = visible(_page("portal_change_password.html", "fr", forced=True))
        chosen = visible(_page("portal_change_password.html", "fr", forced=False))
        assert "Définissez votre mot de passe" in forced
        assert "Vous devez définir un nouveau mot de passe" in forced
        assert "Changer le mot de passe" in chosen
        assert "Mettez à jour le mot de passe" in chosen

    def test_english_is_unchanged(self):
        forced = visible(_page("portal_change_password.html", "en", forced=True))
        assert "Set your password" in forced
        assert "You must set a new password before continuing." in forced

    def test_the_cancel_button_only_shows_when_it_is_optional(self):
        """Pins the condition the translation sits inside. Matched on the link
        itself: the shell's confirm dialog carries a Cancel button on every
        page, so a bare text search would always find one."""
        link = '<a href="/portal/dashboard" class="btn btn-secondary">Annuler</a>'
        assert link in _page("portal_change_password.html", "fr", forced=False)
        assert link not in _page("portal_change_password.html", "fr", forced=True)


class TestThePinConfirmPage:

    def test_it_is_french(self):
        markup = visible(_page("portal_pin_confirm.html", "fr"))
        assert "Ajouter le graphique à un tableau de bord" in markup
        assert "Question d'origine" in markup

    def test_the_visual_count_agrees_with_its_number(self):
        markup = visible(_page("portal_pin_confirm.html", "fr"))
        assert "Ops · 1 visuel" in markup
        assert "Finance · 4 visuels" in markup

    def test_english_is_unchanged(self):
        markup = visible(_page("portal_pin_confirm.html", "en"))
        assert "Ops · 1 visual" in markup
        assert "Finance · 4 visuals" in markup

    def test_the_default_dashboard_name_is_translated(self):
        """It is prefilled into the field and then STORED, so a French user's
        first dashboard should be named in French."""
        markup = _page("portal_pin_confirm.html", "fr", dashboards=[])
        assert 'value="Mon tableau de bord"' in markup

    def test_the_question_and_the_sql_are_shown_as_they_are(self):
        markup = visible(_page("portal_pin_confirm.html", "fr"))
        assert "revenue by region" in markup
        assert "SELECT 1" in markup


class TestTheNewReportPage:

    def test_it_is_french(self):
        markup = visible(_page("portal_report_new.html", "fr"))
        assert "Nouveau rapport" in markup
        assert "Indicateurs à inclure" in markup

    def test_the_example_placeholder_survives_translation(self):
        """The braces in "what's my {name} report?" are literal -- the sentence
        shows the reader the SHAPE of the question. t() leaves an unsupplied
        placeholder in place, so this is safe either way."""
        assert "{name}" in i18n.t("ui.report.intro", lang="fr")
        assert "{name}" in visible(_page("portal_report_new.html", "fr"))

    def test_the_metric_names_are_the_customer_s(self):
        markup = visible(_page("portal_report_new.html", "fr"))
        assert "Revenue" in markup and "DW.SALES" in markup


# ══════════════════════════════════════════════════════════════════════════════
# Notifications
# ══════════════════════════════════════════════════════════════════════════════

class TestTheNotificationsPage:

    ALERT = {"id": "a1", "question": "Revenue drop?", "metric_col": "Revenue",
             "condition": "change_pct", "threshold": 10.0,
             "check_interval_minutes": 60, "status": "active",
             "last_checked": "2026-07-27 08:00:00"}

    def test_the_empty_states_are_french(self):
        markup = visible(_page("portal_notifications.html", "fr"))
        assert "Aucune alerte pour l'instant" in markup
        assert "Aucun rapport n'a encore été créé pour ce compte" in markup

    def test_an_alert_condition_is_a_label_not_a_wire_value(self):
        """It rendered `{{ a.condition }}` -- "change_pct" on screen."""
        markup = visible(_page("portal_notifications.html", "fr", alerts=[self.ALERT]))
        assert "varie de" in markup
        assert "change_pct" not in markup

    def test_english_reads_as_a_sentence_too(self):
        markup = visible(_page("portal_notifications.html", "en", alerts=[self.ALERT]))
        assert "Revenue changes by 10.0" in markup
        assert "change_pct" not in markup

    def test_the_alert_status_is_translated(self):
        paused = dict(self.ALERT, status="paused")
        markup = visible(_page("portal_notifications.html", "fr", alerts=[paused]))
        assert "en pause" in markup

    def test_the_weekday_options_are_french(self):
        markup = visible(_page("portal_notifications.html", "fr",
                               reports=[{"id": 1, "name": "Ops", "description": ""}]))
        for day in ("Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"):
            assert f">{day}<" in markup, day

    def test_the_weekday_values_stay_indices(self):
        """The form posts these to the scheduler."""
        markup = _page("portal_notifications.html", "fr",
                       reports=[{"id": 1, "name": "Ops", "description": ""}])
        for index in range(7):
            assert f'<option value="{index}">' in markup

    def test_the_confirm_dialog_survives_an_apostrophe(self):
        """The handler is JavaScript inside an HTML attribute. Jinja escapes an
        apostrophe to &#39;, the parser decodes it back BEFORE the JS is
        parsed, and a bare string would end early -- taking the whole dialog
        with it. tojson emits \\u0027 and survives both passes."""
        markup = _page("portal_notifications.html", "fr", alerts=[self.ALERT])
        handler = markup[markup.index("qbConfirm({"):]
        handler = handler[:handler.index('">')]
        assert "&#39;" not in handler
        assert "\\u0027" in handler
        assert "Supprimer l\\u0027alerte" in handler


# ══════════════════════════════════════════════════════════════════════════════
# The Semantic Layer
# ══════════════════════════════════════════════════════════════════════════════

class TestTheSemanticLayerPage:

    def test_the_page_is_french(self):
        markup = visible(_page("portal_kb.html", "fr"))
        assert "Couche sémantique" in markup
        assert "Proposer une correction" in markup
        assert "Envoyer pour validation" in markup

    def test_the_table_headings_are_french(self):
        markup = visible(_page("portal_kb.html", "fr"))
        for heading in ("Champ", "Ce qu'est ce champ", "À quoi il sert",
                        "Termes métier", "Confiance"):
            assert heading in markup, heading

    def test_the_pending_count_agrees_with_its_number(self):
        one = visible(_page("portal_kb.html", "fr", pending_count=1))
        many = visible(_page("portal_kb.html", "fr", pending_count=3))
        assert "1 correction en attente" in one
        assert "3 corrections en attente" in many

    def test_english_is_unchanged(self):
        markup = visible(_page("portal_kb.html", "en", pending_count=1))
        assert "1 pending review" in markup
        assert "Suggest edit" in markup

    def test_a_missing_count_does_not_cost_the_reader_the_page(self):
        """Jinja's Undefined raises UndefinedError, not TypeError, so plural()
        had to widen its guard. A route that forgets a key should degrade the
        way an inline `s` did, not 500."""
        assert i18n.plural("ui.kb.pending", None, lang="fr") == "None corrections en attente"
        markup = visible(_page("portal_kb.html", "fr", pending_count=None))
        assert "corrections en attente" in markup

    def test_the_schema_and_column_names_are_the_customer_s(self):
        markup = visible(_page("portal_kb.html", "fr"))
        assert "DW.SALES" in markup and "REVENUE" in markup

    def test_the_no_results_sentence_keeps_its_query_slot(self):
        """The query is a <strong> the script fills in, so the sentence is
        split ON the placeholder. A translation that moves it still puts the
        query where the sentence wants it."""
        markup = _page("portal_kb.html", "fr")
        assert '<strong id="slNoResultsQuery"></strong>' in markup
        assert "Aucune table ni aucun champ ne correspond à" in markup

    def test_the_match_count_uses_the_shared_plural_rule(self):
        """It was `(n !== 1 ? 's' : '')` twice in one sentence, which is wrong
        in French at zero -- on every search that finds nothing, which is
        exactly when the sentence is read."""
        markup = _page("portal_kb.html", "fr")
        assert "window.qbPlural('ui.kb.matched.tables'" in markup
        assert i18n.plural("ui.kb.matched.fields", 0, lang="fr") == "0 champ trouvé"
        assert i18n.plural("ui.kb.matched.fields", 0, lang="en") == "0 fields matched"


# ══════════════════════════════════════════════════════════════════════════════
# All of them at once
# ══════════════════════════════════════════════════════════════════════════════

class TestNoPageWasMissed:

    def test_the_sweep_covers_them_all(self):
        """The failure mode for a set of small pages is not a bad translation,
        it is one page nobody remembered. This is the list, and it has to match
        what is on disk."""
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "portal" / "templates"
        pages = {p.name for p in root.glob("portal_*.html")}
        covered = set(PAGES) | {"portal_base.html", "portal_chat.html",
                                "portal_dashboard.html"}
        assert pages == covered, pages ^ covered

    @pytest.mark.parametrize("name", sorted(PAGES))
    def test_the_page_renders_in_both_languages(self, name):
        for lang in ("en", "fr"):
            assert len(_page(name, lang)) > 500, (name, lang)

    @pytest.mark.parametrize("name", sorted(PAGES))
    def test_every_id_the_page_uses_resolves(self, name):
        """A mistyped id renders as "ui.kb.col.feild" on screen."""
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1] / "portal" / "templates"
                  / name).read_text(encoding="utf-8")
        used = set(re.findall(r"(?<![A-Za-z0-9_$.])t\(\s*'([a-z][a-zA-Z0-9_.]+)'", source))
        used |= set(re.findall(r"qbT\(\s*'([a-z][a-zA-Z0-9_.]+)'", source))
        stems = set(re.findall(r"(?:plural|qbPlural)\(\s*'([a-z][a-zA-Z0-9_.]+)'", source))
        # An id ending in "." is a concatenation -- t('ui.notif.day.' ~ day).
        # The forms it builds are covered by the weekday test above; there is
        # nothing to look up here.
        used = {msg_id for msg_id in used if not msg_id.endswith(".")}
        assert used or stems, f"{name} resolves no ids"
        for msg_id in sorted(used):
            for lang in ("en", "fr"):
                assert i18n.t(msg_id, lang=lang) != msg_id, (name, msg_id, lang)
        for stem in sorted(stems):
            for form in ("one", "other"):
                for lang in ("en", "fr"):
                    key = f"{stem}.{form}"
                    assert i18n.t(key, lang=lang) != key, (name, key, lang)

    @pytest.mark.parametrize("name", sorted(PAGES))
    def test_the_page_reads_as_french(self, name):
        """A page that was never touched renders identically in both
        languages. Compared on the page's OWN body, not the whole document:
        the shell around it is translated already, so a full-page comparison
        would differ for a page nobody had touched -- which is precisely the
        case this test exists to catch."""
        def body(lang):
            markup = visible(_page(name, lang))
            start = markup.index('<div class="main">')
            end = markup.index('<div id="qbDialogBackdrop"')
            return markup[start:end]

        assert body("fr") != body("en"), name
