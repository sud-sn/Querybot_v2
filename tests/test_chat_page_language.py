"""
tests/test_chat_page_language.py

The chat page in the reader's language.

The page a reader spends their session in. Every test renders the real
template through the portal's own Jinja environment and asserts on what comes
out; the catalogue is injected into the page as JSON, so every absence
assertion runs against visible(), which drops the <script> blocks.
"""

from __future__ import annotations

import pytest

from core import i18n
from tests.chat_render import render, visible
from tests.portal_render import unescaped


# ══════════════════════════════════════════════════════════════════════════════
# The phrase that must not be translated literally
# ══════════════════════════════════════════════════════════════════════════════

class TestPlainEnglishMeansPlainLanguage:

    def test_the_hero_does_not_tell_a_french_reader_to_use_english(self):
        """"Ask your data in plain English" means plain LANGUAGE. Translated
        literally it becomes an instruction to write questions in English --
        the opposite of what the French build is for."""
        markup = visible(render(lang="fr"))
        assert "Interrogez vos données en langage courant" in markup
        assert "anglais" not in markup.lower()

    def test_the_welcome_copy_says_the_same_thing(self):
        markup = visible(render(lang="fr"))
        assert "Posez une question métier en langage courant" in markup

    def test_english_is_unchanged(self):
        markup = visible(render(lang="en"))
        assert "Ask your data in plain English" in markup
        assert "Ask a business question in plain English" in markup


# ══════════════════════════════════════════════════════════════════════════════
# The page chrome
# ══════════════════════════════════════════════════════════════════════════════

class TestTheChromeIsTranslated:

    def test_the_thread_panel(self):
        markup = visible(render(lang="fr"))
        for expected in ("Conversations récentes", "Vos conversations",
                         "Espace de travail"):
            assert expected in markup, expected

    def test_the_conversation_shell(self):
        markup = visible(render(lang="fr"))
        for expected in ("Analyste QueryBot en direct", "Connexion…",
                         "En attente de la session en direct", "Historique"):
            assert expected in markup, expected

    def test_the_composer(self):
        markup = unescaped(render(lang="fr"))
        assert 'placeholder="Posez n\'importe quelle question sur vos données…"' in markup
        assert 'aria-label="Envoyer le message"' in markup

    def test_the_artifact_pane(self):
        markup = visible(render(lang="fr"))
        assert "Aperçu du résultat" in markup
        assert "Les graphiques, les indicateurs et les tableaux de résultats" in markup

    def test_the_english_chrome_is_gone(self):
        markup = visible(render(lang="fr"))
        for absent in ("Recent Threads", "Your conversations", "QueryBot live analyst",
                       "Waiting for live session", ">History<", "All schemas",
                       "Result preview", "Start with a workspace question"):
            assert absent not in markup, absent

    def test_the_accessible_names_are_translated(self):
        markup = unescaped(render(lang="fr"))
        for expected in ('aria-label="Conversations récentes"',
                         'aria-label="Historique des requêtes récentes"',
                         'aria-label="Panneau d\'analyse"',
                         'aria-label="Fermer le panneau"'):
            assert expected in markup, expected

    def test_the_history_close_button_says_what_it_closes(self):
        """Its accessible name was "Close navigation" -- the sidebar's label,
        copied onto the history panel's close button. Carrying that into a
        second language would have made the same button wrong twice."""
        markup = render(lang="en")
        assert 'class="hp-close"' in markup
        close = markup[markup.index('class="hp-close"'):]
        assert 'aria-label="Close history"' in close[:200]

    def test_the_browser_tab_title_follows_the_reader(self):
        assert "<title>Chat — QueryBot</title>" in render(lang="en")
        assert "<title>Chat — QueryBot</title>" in render(lang="fr")


# ══════════════════════════════════════════════════════════════════════════════
# Values that are the customer's, not ours
# ══════════════════════════════════════════════════════════════════════════════

class TestCustomerDataIsNeverTranslated:

    def test_the_workspace_name_is_shown_as_stored(self):
        assert "Acme" in visible(render(lang="fr", client_name="Acme"))

    def test_a_missing_workspace_name_falls_back_in_french(self):
        markup = visible(render(lang="fr", client_name=""))
        assert "Votre espace de travail" in markup
        assert "Your workspace" not in markup

    def test_the_suggestions_are_the_workspace_s_own_questions(self):
        """Generated from the customer's schema. Translating one would ask a
        question the warehouse cannot answer."""
        markup = visible(render(lang="fr", suggestions=[
            {"question": "revenue by region", "fqn": "DW.SALES"}]))
        assert "revenue by region" in markup

    def test_the_schema_names_are_shown_as_stored(self):
        markup = visible(render(lang="fr", schemas=[
            {"name": "HR", "table_count": 3}, {"name": "FIN", "table_count": 1}]))
        assert ">\n          HR\n" in markup or "HR" in markup
        assert "FIN" in markup

    def test_the_role_is_a_label_not_the_raw_column(self):
        """It rendered `{{ user.role|capitalize }}` -- a filter that cannot
        translate and mangles whatever case the database holds."""
        assert "Rôle · Analyste" in visible(render(lang="fr"))
        assert "Role · Analyst" in visible(render(lang="en"))


# ══════════════════════════════════════════════════════════════════════════════
# Counts and meters
# ══════════════════════════════════════════════════════════════════════════════

class TestTheMetersAndCounts:

    def test_the_usage_pills_carry_their_numbers_in_french(self):
        markup = visible(render(lang="fr"))
        assert "Jetons ce mois-ci · 1M" in markup
        assert "Requêtes restantes · 497" in markup

    def test_the_hover_titles_too(self):
        markup = unescaped(render(lang="fr"))
        assert 'title="Entrée 600K · Sortie 400K"' in markup
        assert 'title="3 / 500 requêtes utilisées ce mois-ci"' in markup

    def test_a_single_table_takes_the_singular(self):
        markup = render(lang="en")
        assert 'title="1 table"' in markup      # FIN, one table
        assert 'title="3 tables"' in markup     # HR, three

    def test_french_takes_the_singular_at_zero(self):
        """The template had `{{ 's' if count != 1 else '' }}` inline, which is
        right for English and wrong for French at zero."""
        assert i18n.plural("ui.chat.table_count", 0, lang="fr") == "0 table"
        assert i18n.plural("ui.chat.table_count", 0, lang="en") == "0 tables"

    def test_the_selector_is_hidden_for_a_single_schema(self):
        """Pins the condition the plural sits inside, so a future edit to the
        count cannot quietly start rendering a one-option selector. Asserted on
        a rendered tab, not the class name -- that also appears in the page's
        stylesheet."""
        one = render(lang="fr", schemas=[{"name": "FIN", "table_count": 1}])
        assert '<button class="schema-tab"' not in one
        assert '<button class="schema-tab"' in render(lang="fr")


# ══════════════════════════════════════════════════════════════════════════════
# The workspace has chat turned off
# ══════════════════════════════════════════════════════════════════════════════

class TestTheDisabledState:

    def test_it_is_french(self):
        markup = visible(render(lang="fr", enabled=False))
        assert "Chat interne non activé" in markup
        assert "Votre administrateur n'a pas activé le chat interne" in markup
        assert "Retour au tableau de bord" in markup

    def test_english_is_unchanged(self):
        markup = visible(render(lang="en", enabled=False))
        assert "Internal Chat not enabled" in markup
        assert "Back to dashboard" in markup


# ══════════════════════════════════════════════════════════════════════════════
# Every other portal page's tab title
# ══════════════════════════════════════════════════════════════════════════════

class TestTheOtherTabTitles:

    def _title(self, template, lang, **context):
        from tests.portal_render import render as _render
        markup = _render(template, lang=lang, **context)
        return markup[markup.index("<title>") + 7:markup.index("</title>")]

    def test_the_login_page(self):
        assert self._title("portal_login.html", "fr") == "Connexion — Portail QueryBot"
        assert self._title("portal_login.html", "en") == "Sign In — QueryBot Portal"

    def test_the_notifications_page(self):
        title = self._title("portal_notifications.html", "fr",
                            user={"id": 1, "name": "A", "role": "analyst"},
                            alerts=[], reports=[], subscriptions={})
        assert title == "Mes notifications — Portail QueryBot"

    def test_the_semantic_layer_page_reuses_the_nav_label(self):
        """One id for the nav item and the tab, so they cannot drift apart."""
        assert self._title("portal_kb.html", "fr").startswith("Couche sémantique — ")

    def test_a_page_with_no_title_block_falls_back_to_the_suffix(self):
        assert self._title("portal_base.html", "fr") == "Portail QueryBot"
