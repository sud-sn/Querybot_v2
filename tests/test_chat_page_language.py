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

    def test_the_page_does_not_tell_a_french_reader_to_use_english(self):
        """"Ask in plain English" means plain LANGUAGE. Translated literally it
        becomes an instruction to write questions in English -- the opposite of
        what the French build is for. (The hero that said it is gone; the
        welcome says it now.)"""
        markup = visible(render(lang="fr"))
        assert "anglais" not in markup.lower()

    def test_the_welcome_copy_says_the_same_thing(self):
        # The start page greets the reader (core2/suggest.py); its heading until the page knows their name and hour.
        markup = visible(render(lang="fr"))
        assert "Comment puis-je vous aider aujourd'hui ?" in markup and "Pour vous" in markup

    def test_english_is_unchanged(self):
        markup = visible(render(lang="en"))
        assert "How can I help you today?" in markup and "For you" in markup


# ══════════════════════════════════════════════════════════════════════════════
# The page chrome
# ══════════════════════════════════════════════════════════════════════════════

class TestTheChromeIsTranslated:

    def test_the_thread_panel(self):
        # One heading over the conversations; the search behind the sidebar's
        # search button keeps its French label.
        markup = visible(render(lang="fr"))
        assert "Vos conversations" in markup
        assert "Your conversations" not in markup
        assert 'placeholder="Rechercher' in unescaped(render(lang="fr"))

    def test_the_conversation_shell(self):
        # The header is the workspace's name and, until the session is live,
        # its connection state; the history is the sidebar's thread list.
        markup = visible(render(lang="fr"))
        for expected in ("Connexion…", "En attente de la session en direct"):
            assert expected in markup, expected

    def test_the_composer(self):
        markup = unescaped(render(lang="fr"))
        assert ('placeholder="Posez n\'importe quelle question sur vos données… Mettez un nom entre « guillemets » '
                'pour vous y limiter"') in markup
        assert 'aria-label="Envoyer le message"' in markup

    def test_the_artifact_pane(self):
        markup = visible(render(lang="fr"))
        assert "Aperçu du résultat" in markup
        assert "Les graphiques, les indicateurs et les tableaux de résultats" in markup

    def test_the_english_chrome_is_gone(self):
        markup = visible(render(lang="fr"))
        for absent in ("Recent Threads", "Your conversations", "Solviz live analyst",
                       "Waiting for live session", ">History<", "All schemas",
                       "Result preview", "Start with a workspace question"):
            assert absent not in markup, absent

    def test_the_accessible_names_are_translated(self):
        markup = unescaped(render(lang="fr"))
        for expected in ('aria-label="Vos conversations"',
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
        assert "<title>Chat — Solviz</title>" in render(lang="en")
        assert "<title>Chat — Solviz</title>" in render(lang="fr")


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
        """Made from the workspace's own names (core2/suggest.py). In French the words around them are French; the
        names stay as the workspace has them -- translated, they would name nothing the warehouse holds."""
        from core2.suggest import view

        card = {"kind": "breakdown", "question": "Net revenue by region", "question_fr": "Net revenue par region",
                "plan": {"intent": "breakdown", "measures": ["net_revenue"], "group_by": ["region"]}}
        page = view({"state": "ready", "for_you": [[card]], "explore": []}, "fr")
        assert page["for_you"][0][0]["question"] == "Net revenue par region"
        assert page["for_you"][0][0]["label"] == "Répartition"

    def test_the_schema_names_are_shown_as_stored(self):
        markup = visible(render(lang="fr", schemas=[
            {"name": "HR", "table_count": 3}, {"name": "FIN", "table_count": 1}]))
        assert ">\n          HR\n" in markup or "HR" in markup
        assert "FIN" in markup

    def test_the_role_is_a_label_not_the_raw_column(self):
        """It rendered `{{ user.role|capitalize }}` -- a filter that cannot
        translate and mangles whatever case the database holds. The role is
        shown once, under the reader's name in the sidebar: their portal role
        (core/portal_roles.py), a creator for a reader made before roles."""
        assert "Analysts · Créateur" in visible(render(lang="fr"))
        assert "Analysts · Creator" in visible(render(lang="en"))


# ══════════════════════════════════════════════════════════════════════════════
# Counts and meters
# ══════════════════════════════════════════════════════════════════════════════

class TestTheMetersAndCounts:

    def test_the_usage_meters_are_not_on_a_readers_page(self):
        """Tokens and queries used are the admin's numbers (the workspace's
        "This month"); a reader is told about the limit only as they near it,
        in the notice above the question box."""
        markup = visible(render(lang="fr"))
        assert "Jetons ce mois-ci" not in markup
        assert "Requêtes restantes" not in markup
        assert 'id="queryLimitNotice"' in render(lang="fr")

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
        assert self._title("portal_login.html", "fr") == "Connexion — Portail Solviz"
        assert self._title("portal_login.html", "en") == "Sign In — Solviz Portal"

    def test_the_notifications_page(self):
        title = self._title("portal_notifications.html", "fr",
                            user={"id": 1, "name": "A", "role": "analyst"},
                            alerts=[], reports=[], subscriptions={})
        assert title == "Notifications — Portail Solviz"

    def test_the_semantic_layer_page_reuses_the_nav_label(self):
        """One id for the nav item and the tab, so they cannot drift apart."""
        assert self._title("portal_kb.html", "fr").startswith("Guide des données — ")

    def test_a_page_with_no_title_block_falls_back_to_the_suffix(self):
        assert self._title("portal_base.html", "fr") == "Portail Solviz"
