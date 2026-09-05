"""
tests/test_conversational_language.py

The behavioural front door, in both languages.

core/dispatcher.py catches greetings, thanks, goodbyes and frustration before
any guard, because they used to fall through into the SQL pipeline -- "thanks"
was answered with "I couldn't find the right tables or columns". The catch is a
hand-written English regex, so for a French reader the front door was never
there: "bonjour", "merci" and "au revoir" went straight to SQL generation, and
"au revoir" reached it as "to revoir" after the question normaliser had had a
go at it.

So this file has two halves, and the first is what makes the second reachable:

  * `detect_conversational` classifies French small talk. Translating the
    replies without this would be a catalogue of sentences no French reader
    could ever produce.
  * `build_reply` and `clarification_rejection_message` come back in the
    reader's language.

The refusal detector gets the same treatment, and the same care: its docstring
promises that a data question merely containing a negation is NOT a refusal,
and the French additions have to keep that promise -- "commandes annulées par
mois" and "clients sans commande" are questions.

Every test executes the real function.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import i18n
from core.clarification import (
    clarification_rejection_message,
    is_clarification_rejection,
)
from core.conversational import (
    build_reply,
    build_reply_split,
    detect_conversational,
)


class _InLanguage:
    def __init__(self, lang):
        self.lang = lang

    def __enter__(self):
        self._token = i18n.activate_language(self.lang)
        return self

    def __exit__(self, *exc):
        i18n.deactivate_language(self._token)
        return False


KINDS = ("greeting", "thanks", "goodbye", "frustration", "opinion", "vague")


# ══════════════════════════════════════════════════════════════════════════════
# 1. The front door opens for French
# ══════════════════════════════════════════════════════════════════════════════

class TheDetectorClassifiesFrenchSmallTalk(unittest.TestCase):

    FRENCH = {
        "greeting": ["bonjour", "Bonjour !", "bonsoir", "salut", "coucou",
                     "bonjour à tous", "Bonjour 👋", "allo", "bonjour tout le monde"],
        "thanks": ["merci", "Merci beaucoup !", "merci bien", "super merci",
                   "parfait", "génial", "c'est parfait", "nickel", "mille mercis",
                   "ça marche", "merci pour votre aide"],
        "goodbye": ["au revoir", "à bientôt", "à plus tard", "bonne journée",
                    "bonne soirée", "adieu", "à demain", "à la prochaine"],
        "frustration": ["c'est faux", "ça ne marche pas", "ce n'est pas correct",
                        "mauvaise réponse", "c'est nul", "ça ne sert à rien",
                        "tu ne m'aides pas", "c'est inutile"],
        "opinion": ["quel est ton avis ?", "qu'en penses-tu ?",
                    "devrions-nous investir davantage", "comment allons-nous ?",
                    "recommandes-tu cette approche"],
        "vague": ["montre-moi les données", "donne-moi un rapport", "résumé",
                  "analyse", "quoi de neuf ?", "fais une analyse", "bilan",
                  "envoie-moi un aperçu"],
    }

    def test_every_kind_is_detected_in_french(self):
        for kind, samples in self.FRENCH.items():
            for text in samples:
                with self.subTest(kind=kind, text=text):
                    self.assertEqual(detect_conversational(text), kind)

    def test_accents_are_optional(self):
        """Accents are the first thing a hurried typist drops, and the
        tokenizer downstream treats one as a word boundary."""
        for accented, plain in [("à bientôt", "a bientot"),
                                ("bonne journée", "bonne journee"),
                                ("génial", "genial"),
                                ("résumé", "resume"),
                                ("ça ne marche pas", "ca ne marche pas")]:
            with self.subTest(text=plain):
                self.assertEqual(detect_conversational(plain),
                                 detect_conversational(accented))
                self.assertIsNotNone(detect_conversational(plain))

    def test_the_typographic_apostrophe_works_as_well_as_the_ascii_one(self):
        for text in ("c'est faux", "c’est faux", "qu'en penses-tu",
                     "qu’en penses-tu"):
            with self.subTest(text=text):
                self.assertIsNotNone(detect_conversational(text))

    # The control. Without it, a pattern broad enough to catch "analyse" would
    # also swallow every French question that happens to contain it.
    NOT_SMALL_TALK = [
        "quel est le chiffre d'affaires par région ?",
        "montre-moi les données de ventes par mois",
        "analyse des ventes par client",
        "combien de commandes ce mois-ci",
        "rapport de ventes par région",
        "salut, quel est le CA ?",
        "merci de me donner le CA par mois",
        "donne-moi le résumé des ventes du trimestre",
        "bilan comptable par filiale",
    ]

    def test_a_real_french_question_is_never_small_talk(self):
        for text in self.NOT_SMALL_TALK:
            with self.subTest(text=text):
                self.assertIsNone(detect_conversational(text))

    def test_english_classification_is_unchanged(self):
        """Accent folding runs before every pattern now. It is a no-op on
        ASCII, and this is what says so."""
        for text, kind in [("hi", "greeting"), ("hello there", "greeting"),
                           ("thanks", "thanks"), ("thank you so much", "thanks"),
                           ("bye", "goodbye"), ("see you later", "goodbye"),
                           ("this is wrong", "frustration"),
                           ("what's your opinion", "opinion"),
                           ("show me the data", "vague"),
                           ("hi, what is revenue this month", None),
                           ("show customers with no orders", None)]:
            with self.subTest(text=text):
                self.assertEqual(detect_conversational(text), kind)


# ══════════════════════════════════════════════════════════════════════════════
# 2. And the reply comes back in the reader's language
# ══════════════════════════════════════════════════════════════════════════════

class TheConversationalRepliesAreTranslated(unittest.TestCase):

    USER = {"name": "Ada Lovelace"}

    def _reply(self, lang, kind, user=None):
        with _InLanguage(lang):
            return build_reply(kind, "acct", self.USER if user is None else user)

    def test_every_kind_speaks_both_languages(self):
        """Line by line, not reply by reply. These replies are several
        sentences each, and one English line among five translated ones still
        leaves the two replies different -- which is how a half-translated
        greeting passes a whole-reply comparison."""
        for kind in KINDS:
            with self.subTest(kind=kind):
                english = self._reply("en", kind)
                french = self._reply("fr", kind)
                self.assertTrue(english.strip(), kind)
                self.assertTrue(french.strip(), kind)
                self.assertNotIn("reply.", french)
                shared = ({line.strip() for line in english.splitlines() if line.strip()}
                          & {line.strip() for line in french.splitlines() if line.strip()})
                self.assertEqual(shared, set(), f"{kind} keeps English lines")

    def test_the_greeting_introduces_itself_in_french(self):
        french = self._reply("fr", "greeting")
        self.assertIn("Je suis QueryBot", french)
        self.assertNotIn("ask me anything", french)

    def test_the_greeting_uses_the_readers_first_name(self):
        self.assertTrue(self._reply("en", "greeting").startswith("Hello, Ada! 👋"))
        # French greets a name without the comma, and puts a space before the
        # exclamation mark -- which is why the name is not spliced into a
        # shared sentence.
        self.assertTrue(self._reply("fr", "greeting").startswith("Bonjour Ada ! 👋"))

    def test_an_anonymous_greeting_drops_the_name_cleanly(self):
        self.assertTrue(self._reply("en", "greeting", user={}).startswith("Hello! 👋"))
        self.assertTrue(self._reply("fr", "greeting", user={}).startswith("Bonjour ! 👋"))

    def test_help_stays_help(self):
        """`help` is compared by equality in core/dispatcher.py. A translated
        command is a command nobody can run."""
        for kind in ("greeting", "vague"):
            with self.subTest(kind=kind):
                self.assertIn("`help`", self._reply("fr", kind))

    def test_the_markdown_survives_translation(self):
        """The chat page renders these with formatBotText; a bullet that lost
        its underscores renders as a run-on line."""
        french = self._reply("fr", "vague")
        self.assertIn("  • _", french)
        self.assertEqual(french.count("  • _"), 3)

    def test_the_fallback_examples_are_french_for_a_french_reader(self):
        """They go back through the pipeline when someone types one, and
        core/question_normalizer.py canonicalises a French question to English
        before any detector reads it."""
        french = self._reply("fr", "greeting")
        self.assertIn("chiffre d'affaires total ce mois-ci", french)
        self.assertNotIn("total revenue this month", french)

    def test_the_split_greeting_is_translated_too(self):
        """build_reply_split is the branch the portal actually takes, because
        WebAdapter can render the questions as buttons."""
        with _InLanguage("fr"):
            intro, questions = build_reply_split("greeting", "acct", self.USER)
        self.assertTrue(intro.startswith("Bonjour Ada ! 👋"))
        self.assertIn("Voici quelques questions pour commencer", intro)
        self.assertIsInstance(questions, list)

    def test_the_split_vague_intro_is_translated(self):
        with _InLanguage("fr"):
            intro, _ = build_reply_split("vague", "acct", self.USER)
        self.assertIn("j'ai seulement besoin de savoir quoi mesurer", intro)

    def test_an_unknown_kind_is_still_empty(self):
        self.assertEqual(self._reply("fr", "teleport"), "")

    def test_the_english_wording_is_unchanged(self):
        """This was a translation, not a rewrite. Anything that moved in
        English moved by accident."""
        self.assertEqual(
            self._reply("en", "thanks"),
            "You're welcome! Ask me another question whenever you're ready.")
        self.assertEqual(
            self._reply("en", "goodbye"),
            "Goodbye! I'll be here whenever you need your data. 👋")
        self.assertIn(
            "Type `help` for commands, or just ask in plain English.",
            self._reply("en", "greeting"))


# ══════════════════════════════════════════════════════════════════════════════
# 3. Turning down a clarification
# ══════════════════════════════════════════════════════════════════════════════

class TheRejectionMessageIsTranslated(unittest.TestCase):

    SOURCES = ("graph_join_path", "source_scope", "metric_date_context", "", "other")

    def test_every_source_speaks_both_languages(self):
        for source in self.SOURCES:
            with self.subTest(source=source):
                with _InLanguage("en"):
                    english = clarification_rejection_message({"source": source})
                with _InLanguage("fr"):
                    french = clarification_rejection_message({"source": source})
                self.assertTrue(french.strip())
                self.assertNotEqual(english, french)
                self.assertNotIn("reply.", french)

    def test_the_three_sentences_stay_distinct_in_french(self):
        """Each names what it will not use. Collapsing them would tell a user
        their date choice was dropped when it was their join path."""
        with _InLanguage("fr"):
            said = {clarification_rejection_message({"source": s})
                    for s in ("graph_join_path", "metric_date_context", "")}
        self.assertEqual(len(said), 3)

    def test_a_missing_meta_still_answers(self):
        with _InLanguage("fr"):
            self.assertTrue(clarification_rejection_message(None).strip())

    def test_the_source_is_a_wire_token_not_copy(self):
        """The French join sentence has to be selected by the same stored
        token the English one is."""
        with _InLanguage("fr"):
            joins = clarification_rejection_message({"source": "graph_join_path"})
            scope = clarification_rejection_message({"source": "source_scope"})
        self.assertEqual(joins, scope)
        self.assertIn("chemins de relation", joins)


class TheRefusalDetectorUnderstandsFrench(unittest.TestCase):

    REFUSALS = [
        "non", "Non merci", "aucun", "aucune option", "aucun des deux",
        "aucune de ces options", "aucun de ces choix", "aucun ne convient",
        "ni l'un ni l'autre", "annule", "annuler", "annule ça",
        "annulez la question", "laisse tomber", "peu importe", "tant pis",
        "ignore ça", "oublie", "je ne veux aucune de ces options",
        "n'utilise pas ces options", "recommence", "on recommence à zéro",
        "ce n'est pas pertinent", "pas pertinent", "sans objet",
        "rien de tout ça", "non, annule", "svp annulez",
    ]

    # The promise in is_clarification_rejection's own docstring: a data
    # question that merely CONTAINS a negation is not a refusal.
    QUESTIONS = [
        "commandes annulées par mois",
        "clients sans commande",
        "chiffre d'affaires hors retours",
        "n'inclus pas les commandes annulées",
        "montre les clients avec aucune commande",
        "aucun client n'a commandé ce mois-ci",
        "aucune option de livraison par région",
        "annule le rapport de ventes par region et par mois",
        "combien de commandes ont été annulées",
    ]

    def test_french_refusals_are_recognised(self):
        for text in self.REFUSALS:
            with self.subTest(text=text):
                self.assertTrue(is_clarification_rejection(text))

    def test_a_french_question_containing_a_negation_is_not_a_refusal(self):
        for text in self.QUESTIONS:
            with self.subTest(text=text):
                self.assertFalse(is_clarification_rejection(text))

    def test_the_english_behaviour_is_unchanged(self):
        for text in ("no", "nope", "neither", "none of these", "don't use this",
                     "do not use either", "cancel", "skip this", "never mind",
                     "start over", "not relevant"):
            with self.subTest(text=text):
                self.assertTrue(is_clarification_rejection(text))
        for text in ("show customers with no orders",
                     "do not include cancelled orders",
                     "revenue excluding returned invoices",
                     "warehouses with no available stock"):
            with self.subTest(text=text):
                self.assertFalse(is_clarification_rejection(text))

    def test_an_accent_is_no_longer_a_word_boundary(self):
        """_rejection_tokens splits on anything outside [a-z0-9], so before
        folding "annulé" tokenized to ["annul"] and "ça" to ["a"]."""
        self.assertTrue(is_clarification_rejection("annulé"))
        self.assertTrue(is_clarification_rejection("ignore ça"))

    def test_a_long_reply_is_never_a_refusal(self):
        """The 8-token bound is what stops a French request reaching the
        patterns at all; the additions must not have widened it."""
        self.assertFalse(is_clarification_rejection(
            "annule et montre-moi plutôt le chiffre d'affaires par région "
            "pour les six derniers mois"))


if __name__ == "__main__":
    unittest.main()
