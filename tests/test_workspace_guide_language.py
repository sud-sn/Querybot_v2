"""
tests/test_workspace_guide_language.py

The governed workspace guide, in both languages.

core/workspace_guide.py answers "what can you do", "what data do you have" and
"how does the Semantic Layer work" from metadata only -- no SQL, no LLM, no
result rows. It is the onboarding surface: it is read by someone who does not
yet know what the product does, which is the worst possible moment to be
reading a second language.

Two things had to happen together, and the first is what makes the second
reachable:

  * `detect_conversational` routes French to the six guide kinds. Until it
    did, "quelles données avez-vous ?" fell through to SQL generation -- and
    the French `vague` reply shipped last commit points the reader at exactly
    that phrase, so the product was handing out a dead route in its own words.
  * The guide's own prose is translated, counts included. Every count in the
    file was `'s' if n != 1 else ''`, an English rule that reports "0 tables"
    where French wants "0 table" -- and an empty workspace is exactly what a
    new tenant has.

The English output is asserted to be byte-identical, kind by kind, because
this was a translation and not a rewrite.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import i18n
from core.conversational import detect_conversational
from core.workspace_guide import (
    GUIDE_KINDS,
    _table_summary,
    render_workspace_guide,
)
from test_workspace_guide import MODEL

USER = {"id": 7, "role": "user", "name": "Delivery Manager"}
KINDS = ("capability_overview", "business_overview", "data_inventory",
         "table_meanings", "semantic_explainer", "question_examples")


class _InLanguage:
    def __init__(self, lang):
        self.lang = lang

    def __enter__(self):
        self._token = i18n.activate_language(self.lang)
        return self

    def __exit__(self, *exc):
        i18n.deactivate_language(self._token)
        return False


class _Workspace:
    """The repo's own guide fixture, with the volumes a test needs to vary.

    Everything patched here is a boundary -- the store, the semantic model,
    the suggestion engine -- so render_workspace_guide itself runs unmodified.
    """

    def __init__(self, *, metrics=1, terms=1, dashboards=1, tables=True,
                 examples=2, business=True):
        allowed = ({"DB.SALES.F_ORDERS", "DB.SALES.D_CUSTOMER", "DB.SALES.D_DATE"}
                   if tables else set())
        self._patchers = [
            patch("core.workspace_guide.store.get_client", return_value={
                "business_desc": ("We manage retail orders and customer "
                                  "performance.") if business else ""}),
            patch("core.workspace_guide.get_state", return_value={
                "state": "READY", "kb_dir": "Z:/missing/kb",
                "schema_dir": "Z:/missing/schema"}),
            patch("core.workspace_guide.load_semantic_model", return_value=(
                MODEL if business else {**MODEL, "business_description": ""})),
            patch("core.workspace_guide.store.get_allowed_tables", return_value=allowed),
            patch("core.workspace_guide.store.list_metrics", return_value=[
                {"name": f"Metric {i}", "description": "d",
                 "base_table": "DB.SALES.F_ORDERS", "is_active": 1}
                for i in range(metrics)]),
            patch("core.workspace_guide.store.list_terms", return_value=[
                {"term": f"term{i}", "definition": "d",
                 "tables_involved": "DB.SALES.D_CUSTOMER"}
                for i in range(terms)]),
            patch("core.workspace_guide.store.list_dashboards", return_value=[
                {"id": i, "name": f"Board {i}", "chart_count": 1}
                for i in range(dashboards)]),
            patch("core.workspace_guide._safe_examples", return_value=[
                f"Question {i}?" for i in range(examples)]),
        ]

    def __enter__(self):
        for patcher in self._patchers:
            patcher.start()
        return self

    def __exit__(self, *exc):
        for patcher in reversed(self._patchers):
            patcher.stop()
        return False

    def render(self, kind, lang, **kw):
        with _InLanguage(lang):
            return render_workspace_guide(kind, "acct", USER, **kw)[0]


class EveryGuideKindIsTranslated(unittest.TestCase):

    def test_every_kind_speaks_both_languages(self):
        """Line by line, not kind by kind: these are multi-paragraph answers,
        and one English line among ten translated ones still leaves the two
        renders different."""
        with _Workspace() as workspace:
            for kind in KINDS:
                with self.subTest(kind=kind):
                    english = workspace.render(kind, "en")
                    french = workspace.render(kind, "fr")
                    self.assertTrue(french.strip(), kind)
                    self.assertNotIn("guide.", french)
                    self.assertEqual(self._prose(english) & self._prose(french),
                                     set(), f"{kind} keeps English lines")

    # Lines that are the TENANT's, not the product's: the workspace
    # description, an entity name, a validated question from the suggestion
    # engine. They are identical in both languages on purpose, and
    # test_the_tenants_own_words_ride_through_untranslated asserts that.
    TENANT_LINES = {
        "We manage retail orders and customer performance.",
        "• Orders",
    }

    def _prose(self, text):
        return {
            line.strip() for line in text.splitlines()
            if line.strip()
            and line.strip() not in self.TENANT_LINES
            and not line.strip().startswith("• _")
        }

    def test_the_tenants_own_words_ride_through_untranslated(self):
        """The control for the filter above, and a claim in its own right: a
        translated business description or entity name would be the product
        rewriting the customer's data."""
        with _Workspace() as workspace:
            for lang in ("en", "fr"):
                with self.subTest(lang=lang):
                    text = workspace.render("business_overview", lang)
                    self.assertIn("We manage retail orders and customer "
                                  "performance.", text)
                    self.assertIn("  • Orders", text)
                    self.assertIn("Metric 0", text)

    def test_the_kinds_covered_are_the_kinds_that_exist(self):
        """A seventh guide kind added later and not translated fails here
        rather than shipping."""
        self.assertEqual(set(KINDS), set(GUIDE_KINDS))

    def test_an_unknown_kind_is_still_empty(self):
        with _Workspace() as workspace:
            self.assertEqual(workspace.render("teleport", "fr"), "")


class TheEnglishIsUnchanged(unittest.TestCase):
    """A translation, not a rewrite. These are the sentences the existing
    tests in tests/test_workspace_guide.py pin, asserted here as whole lines
    so a reflow shows up too."""

    def test_the_capability_answer_is_word_for_word(self):
        with _Workspace() as workspace:
            text = workspace.render("capability_overview", "en")
        self.assertIn("*What I can do in this workspace*", text)
        self.assertIn("You currently have access to 1 named dashboard.", text)
        self.assertIn("This workspace currently exposes 1 governed metric, "
                      "1 business term, 1 relationship, and 1 date role "
                      "within your access.", text)
        self.assertIn("  • Create and maintain named dashboards, add results, "
                      "arrange visuals, apply filters, and subscribe to "
                      "updates.", text)

    def test_the_semantic_explainer_is_word_for_word(self):
        with _Workspace() as workspace:
            text = workspace.render("semantic_explainer", "en")
        self.assertIn("Within your current access, I can use 1 metric, "
                      "1 business term, 1 governed relationship, and "
                      "1 date role.", text)

    def test_the_business_overview_is_word_for_word(self):
        with _Workspace() as workspace:
            text = workspace.render("business_overview", "en")
        self.assertIn("Your access covers 1 table across SALES (1 table).", text)
        self.assertIn("Ask _which tables are available and what do they mean?_",
                      text)

    def test_the_empty_workspace_is_word_for_word(self):
        with _Workspace(tables=False, metrics=0, terms=0, dashboards=0,
                        examples=0) as workspace:
            text = workspace.render("data_inventory", "en")
        self.assertIn("No queryable tables are assigned to your current access.",
                      text)
        self.assertIn("Open Semantic Layer from the Portal sidebar", text)


class TheCountsFollowTheLanguagesOwnRule(unittest.TestCase):
    """`'s' if n != 1 else ''` is English. French takes the singular at zero as
    well as one, and an empty workspace is what a new tenant has."""

    def test_english_pluralises_zero_and_french_does_not(self):
        with _Workspace(metrics=0, terms=0, dashboards=0) as workspace:
            english = workspace.render("capability_overview", "en")
            french = workspace.render("capability_overview", "fr")
        self.assertIn("0 governed metrics", english)
        self.assertIn("0 business terms", english)
        self.assertIn("0 indicateur gouverné", french)
        self.assertIn("0 terme métier", french)
        self.assertNotIn("0 indicateurs", french)
        self.assertNotIn("0 termes", french)

    def test_one_is_singular_in_both(self):
        with _Workspace(metrics=1, terms=1, dashboards=1) as workspace:
            self.assertIn("1 governed metric,", workspace.render("capability_overview", "en"))
            self.assertIn("1 indicateur gouverné,", workspace.render("capability_overview", "fr"))

    def test_several_is_plural_in_both(self):
        with _Workspace(metrics=3, terms=2, dashboards=4) as workspace:
            english = workspace.render("capability_overview", "en")
            french = workspace.render("capability_overview", "fr")
        self.assertIn("3 governed metrics", english)
        self.assertIn("4 named dashboards", english)
        self.assertIn("3 indicateurs gouvernés", french)
        self.assertIn("4 tableaux de bord nommés", french)

    def test_the_schema_entry_counts_its_tables(self):
        """One schema, two tables. The singular/plural split here is the one
        place the two languages agree at every reachable count -- a schema
        with zero tables cannot exist, because the count is derived from the
        tables themselves -- so what this pins is that the entry interpolates
        its name and its count at all, in both."""
        two_in_one_schema = {
            **MODEL,
            "tables": [
                MODEL["tables"][0],
                {"fqn": "DB.SALES.F_RETURNS", "schema": "SALES",
                 "table": "F_RETURNS", "entity": "Returns", "type": "fact",
                 "grain": "one row per return"},
            ],
        }
        with _Workspace() as workspace, \
             patch("core.workspace_guide.load_semantic_model",
                   return_value=two_in_one_schema), \
             patch("core.workspace_guide.store.get_allowed_tables",
                   return_value={"DB.SALES.F_ORDERS", "DB.SALES.F_RETURNS"}):
            self.assertIn("SALES (2 tables)", workspace.render("data_inventory", "en"))
            self.assertIn("SALES (2 tables)", workspace.render("data_inventory", "fr"))
        with _Workspace() as workspace:
            self.assertIn("SALES (1 table)", workspace.render("data_inventory", "en"))
            self.assertIn("SALES (1 table)", workspace.render("data_inventory", "fr"))

    def test_the_truncated_table_list_counts_the_remainder(self):
        """_table_lines shows 15 and then says how many it left out. The
        fixture elsewhere has one table, so this is the only place that line
        is reachable at all."""
        many = {
            **MODEL,
            "tables": [
                {"fqn": f"DB.SALES.F_T{i:02d}", "schema": "SALES",
                 "table": f"F_T{i:02d}", "entity": f"Thing {i:02d}",
                 "type": "fact", "grain": "one row"}
                for i in range(20)
            ],
        }
        allowed = {f"DB.SALES.F_T{i:02d}" for i in range(20)}
        with _Workspace() as workspace, \
             patch("core.workspace_guide.load_semantic_model", return_value=many), \
             patch("core.workspace_guide.store.get_allowed_tables", return_value=allowed):
            english = workspace.render("data_inventory", "en")
            french = workspace.render("data_inventory", "fr")
        self.assertIn("…and 5 more accessible tables in the Semantic Layer.", english)
        self.assertIn("…et 5 autres tables accessibles dans la couche sémantique.",
                      french)
        self.assertIn("SALES (20 tables)", english)
        self.assertIn("SALES (20 tables)", french)

    def test_no_dashboards_gets_the_other_sentence_entirely(self):
        """Not "0 named dashboards" -- a different sentence, in both."""
        with _Workspace(dashboards=0) as workspace:
            english = workspace.render("capability_overview", "en")
            french = workspace.render("capability_overview", "fr")
        self.assertIn("You can create a named dashboard from a result", english)
        self.assertIn("Vous pouvez créer un tableau de bord nommé", french)
        self.assertNotIn("0 named dashboard", english)
        self.assertNotIn("0 tableau de bord", french)


class TheFallbacksAndEdgesAreTranslated(unittest.TestCase):

    def test_a_table_with_only_a_grain_is_described_in_french(self):
        table = {"fqn": "DB.SALES.F_ORDERS", "entity": "Orders",
                 "grain": "one row per order"}
        self.assertEqual(_table_summary(table, {})["meaning"],
                         "Represents orders at one row per order.")
        with _InLanguage("fr"):
            french = _table_summary(table, {})["meaning"]
        self.assertEqual(french, "Représente orders, au grain : one row per order.")

    def test_a_table_with_only_a_type_is_described_in_french(self):
        table = {"fqn": "DB.SALES.F_ORDERS", "entity": "Orders", "type": "fact"}
        self.assertEqual(_table_summary(table, {})["meaning"],
                         "Represents orders fact data.")
        with _InLanguage("fr"):
            self.assertIn("type fact", _table_summary(table, {})["meaning"])

    def test_a_table_with_neither_says_so_in_french(self):
        table = {"fqn": "DB.SALES.F_ORDERS", "entity": "Orders"}
        with _InLanguage("fr"):
            self.assertEqual(
                _table_summary(table, {})["meaning"],
                "Aucune description métier n'a encore été rédigée pour cette table.")

    def test_no_validated_examples_says_so_in_french(self):
        with _Workspace(examples=0) as workspace:
            french = workspace.render("question_examples", "fr")
        self.assertIn("Aucune question de démarrage validée", french)

    def test_an_empty_workspace_says_so_in_french(self):
        with _Workspace(tables=False, metrics=0, terms=0, dashboards=0) as workspace:
            french = workspace.render("data_inventory", "fr")
        self.assertIn("Aucune table interrogeable", french)

    def test_a_workspace_with_no_business_description_says_so_in_french(self):
        with _Workspace(business=False) as workspace:
            french = workspace.render("business_overview", "fr")
        self.assertIn("Aucune description de l'activité", french)

    def test_the_examples_block_is_omitted_on_request_in_both(self):
        """build_reply_split renders the questions as buttons, so the message
        must not repeat them."""
        with _Workspace() as workspace:
            for lang in ("en", "fr"):
                with self.subTest(lang=lang):
                    text = workspace.render("capability_overview", lang,
                                            include_examples=False)
                    self.assertNotIn("Question 0?", text)


class TheFrenchGuideIsReachable(unittest.TestCase):
    """Translating the guide without routing French to it would be a catalogue
    of sentences no French reader could produce."""

    FRENCH_ROUTES = {
        "data_inventory": ["quelles données avez-vous ?",
                           "quelles tables sont disponibles ?",
                           "montre-moi les tables disponibles",
                           "liste les tables",
                           "qu'y a-t-il dans la base ?",
                           "que puis-je demander ?"],
        "table_meanings": ["quelles tables sont disponibles et que signifient-elles ?",
                           "explique-moi les tables",
                           "signification métier des tables",
                           "que représente chaque table ?"],
        "semantic_explainer": ["comment fonctionne la couche sémantique ?",
                               "explique-moi la couche sémantique",
                               "qu'est-ce que la couche sémantique ?"],
        "business_overview": ["explique-moi l'entreprise",
                              "parle-moi de l'activité",
                              "que fait cette entreprise ?",
                              "vue d'ensemble de l'activité"],
        "question_examples": ["quelles questions puis-je poser ?",
                              "donne-moi des exemples de questions",
                              "comment poser une question ?"],
        "capability_overview": ["que peux-tu faire ?",
                                "quelles sont tes capacités ?",
                                "comment peux-tu m'aider ?",
                                "qu'est-ce que querybot ?"],
    }

    def test_french_reaches_every_guide_kind(self):
        for kind, phrases in self.FRENCH_ROUTES.items():
            for text in phrases:
                with self.subTest(kind=kind, text=text):
                    self.assertEqual(detect_conversational(text), kind)

    # The control: a broad enough pattern would swallow real questions.
    NOT_A_GUIDE_ASK = [
        "quelles tables contiennent les ventes par région ?",
        "montre-moi les ventes par table de produits",
        "explique-moi la variation du chiffre d'affaires",
        "que fait le client ACME ce mois-ci ?",
        "combien de questions ont été posées la semaine dernière ?",
        "liste les commandes de janvier",
        "explique-moi le calcul de la marge",
    ]

    def test_a_real_french_question_is_not_routed_to_the_guide(self):
        for text in self.NOT_A_GUIDE_ASK:
            with self.subTest(text=text):
                self.assertIsNone(detect_conversational(text))

    def test_the_french_copy_does_not_point_at_a_dead_route(self):
        """Two French sentences tell the reader what to type next. Both have
        to reach the kind they promise."""
        import re

        for msg_id, expected in [("reply.vague.help_hint", "data_inventory"),
                                 ("guide.business.ask_tables", "table_meanings")]:
            with self.subTest(msg_id=msg_id):
                quoted = re.findall(r"_([^_]+)_", i18n.t(msg_id, lang="fr"))
                self.assertTrue(quoted, msg_id)
                self.assertIn(expected,
                              [detect_conversational(q) for q in quoted])

    def test_the_english_copy_points_at_a_live_route_too(self):
        import re

        for msg_id, expected in [("reply.vague.help_hint", "data_inventory"),
                                 ("guide.business.ask_tables", "table_meanings")]:
            with self.subTest(msg_id=msg_id):
                quoted = re.findall(r"_([^_]+)_", i18n.t(msg_id, lang="en"))
                self.assertIn(expected,
                              [detect_conversational(q) for q in quoted])


if __name__ == "__main__":
    unittest.main()
