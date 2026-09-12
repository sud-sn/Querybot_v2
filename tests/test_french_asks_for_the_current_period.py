"""
tests/test_french_asks_for_the_current_period.py

"Quel est le total des ventes nettes du mois en cours ?" carried no window.

detect_temporal_window is the gate on the whole data-relative regime: a question
it does not recognise carries no window, so the fail-closed check for "a period
was asked for and this fact has no governed business date" never fires, no
temporal policy reaches the compiler, and generation falls through to the
dialect's own date recipes -- which use the SERVER CLOCK. Its own docstring says
so. What it reads is the CANONICAL text, so in French the gate is only as good
as core/question_normalizer.py.

Measured before this change, every one of these reached the detector with no
window at all:

    du mois en cours          du mois courant        du mois actuel
    de l'annee en cours       de l'annee courante    de l'annee actuelle
    du trimestre en cours     du trimestre courant   du trimestre actuel
    de la semaine en cours    de la semaine courante de la semaine actuelle
    annee a date              mois a date            trimestre a date
    cumul mensuel             cumul trimestriel
    depuis le debut du mois   depuis le debut du trimestre
    les 4 dernieres semaines  les 2 dernieres annees
    les six derniers mois     les trente derniers jours

The failure was selective, which is what made it dangerous: "les 4 derniers
mois" and "les 30 derniers jours" worked, so a rehearsal that happened to use a
masculine noun and a digit would look fine. semaine and annee are feminine, and
`derniers?` matched neither "derniere" nor "dernieres".

Three separate causes, all in the canonicaliser:

  1. No entry for the current period in any of its three spellings ("en cours",
     "courant", "actuel"). The keys added are all anchored on a CALENDAR UNIT,
     which is what makes "en cours" safe to read at all -- alone it means "in
     progress", and a distributor's questions are full of it.
  2. `derniers?` missed the feminine inflections.
  3. A count spelled as a word ("les six derniers mois") was never a count.

The last one also closed a Top-N asymmetry: detect_top_n_intent reads "top five"
and "top 5" alike, so English was never exposed, while "les cinq meilleurs
clients" requested no row limit and handed the reader every customer.
"""

from __future__ import annotations

import pytest

from core.contextual_dates import detect_temporal_window
from core.pipeline_helpers import _COMPILABLE_WINDOW_KINDS
from core.query_pipeline import BUSINESS_DATE_WINDOW_KINDS
from core.query_semantics import detect_top_n_intent
from core.question_normalizer import canonical_question


def window(french: str) -> dict:
    """The window the pipeline would see for a French question."""
    return detect_temporal_window(canonical_question(f"ventes nettes {french}", "fr"))


CURRENT_PERIOD = [
    ("du mois en cours", "this_month"),
    ("de l'année en cours", "this_year"),
    ("du trimestre en cours", "this_quarter"),
    ("de la semaine en cours", "this_week"),
    ("en cours d'année", "this_year"),
    ("du mois courant", "this_month"),
    ("de l'année courante", "this_year"),
    ("du trimestre courant", "this_quarter"),
    ("de la semaine courante", "this_week"),
    ("du mois actuel", "this_month"),
    ("de l'année actuelle", "this_year"),
    ("du trimestre actuel", "this_quarter"),
    ("de la semaine actuelle", "this_week"),
]

TO_DATE = [
    ("année à date", "this_year"),
    ("mois à date", "this_month"),
    ("trimestre à date", "this_quarter"),
    ("semaine à date", "this_week"),
    ("cumul annuel", "this_year"),
    ("cumul mensuel", "this_month"),
    ("cumul trimestriel", "this_quarter"),
    ("cumul hebdomadaire", "this_week"),
    ("depuis le début de l'année", "this_year"),
    ("depuis le début du mois", "this_month"),
    ("depuis le début du trimestre", "this_quarter"),
    ("depuis le début de la semaine", "this_week"),
]


class TestTheCurrentPeriodIsAWindow:

    @pytest.mark.parametrize("french,kind", CURRENT_PERIOD)
    def test_it_is_detected(self, french, kind):
        assert window(french).get("kind") == kind, window(french)

    @pytest.mark.parametrize("french,kind", CURRENT_PERIOD)
    def test_it_anchors_on_the_data_not_the_clock(self, french, kind):
        """The kind alone is not the point -- the anchor policy is. This is the
        field that stops the dialect's GETDATE() recipes being used."""
        assert window(french).get("anchor_policy") == "latest_available"

    @pytest.mark.parametrize("french,kind", CURRENT_PERIOD)
    def test_the_governance_gate_recognises_the_kind(self, french, kind):
        """A detected window only matters if it reaches the fail-closed check
        for "a period was asked for and this fact has no governed business
        date". That check is keyed on this set."""
        assert window(french)["kind"] in BUSINESS_DATE_WINDOW_KINDS


class TestToDateIsTheCurrentPeriod:
    """A to-date window IS the current period anchored on the data, which is
    why detect_temporal_window already maps every "X to date" onto this_X and
    nothing downstream needed a new case."""

    @pytest.mark.parametrize("french,kind", TO_DATE)
    def test_it_is_detected(self, french, kind):
        assert window(french).get("kind") == kind, window(french)


class TestAFeminineNounStillCounts:
    """semaine and annee are feminine. `derniers?` matched the masculine
    inflections only, so exactly half the units in the language were lost."""

    @pytest.mark.parametrize("french,amount,unit", [
        ("les 4 dernières semaines", 4, "week"),
        ("les 2 dernières années", 2, "year"),
        ("ces 6 dernières années", 6, "year"),
        ("les 13 dernières semaines", 13, "week"),
        ("les 6 derniers mois", 6, "month"),
        ("les 30 derniers jours", 30, "day"),
    ])
    def test_the_rolling_window_carries_the_number(self, french, amount, unit):
        found = window(french)
        assert found.get("kind") == "last_n", found
        assert found["amount"] == amount
        assert found["unit"] == unit

    @pytest.mark.parametrize("french", [
        "les 4 dernières semaines", "les 2 dernières années",
    ])
    def test_and_the_compiler_can_build_it(self, french):
        """last_n is deterministically compilable, so this is not a window that
        is merely detected and then declined."""
        assert window(french)["kind"] in _COMPILABLE_WINDOW_KINDS


class TestACountSpelledAsAWordIsACount:

    @pytest.mark.parametrize("french,amount,unit", [
        ("les six derniers mois", 6, "month"),
        ("les trente derniers jours", 30, "day"),
        ("les douze dernières semaines", 12, "week"),
        ("les deux dernières années", 2, "year"),
        ("les dix-huit derniers mois", 18, "month"),
        ("les quatre derniers trimestres", 4, "quarter"),
    ])
    def test_the_window_carries_the_number(self, french, amount, unit):
        found = window(french)
        assert found.get("kind") == "last_n", found
        assert (found["amount"], found["unit"]) == (amount, unit)

    def test_the_spelling_does_not_change_the_window(self):
        assert window("les six derniers mois") == window("les 6 derniers mois")


class TestTheTopNAsymmetryIsClosed:
    """detect_top_n_intent reads "top five" as readily as "top 5", so English
    never lost a row limit to a spelled number and French always did."""

    @pytest.mark.parametrize("spelled,digits", [
        ("les cinq meilleurs clients", "les 5 meilleurs clients"),
        ("les trois pires produits", "les 3 pires produits"),
        ("les dix premières succursales", "les 10 premières succursales"),
    ])
    def test_both_spellings_request_the_same_limit(self, spelled, digits):
        from_spelled = detect_top_n_intent(canonical_question(spelled, "fr"))
        from_digits = detect_top_n_intent(canonical_question(digits, "fr"))
        assert from_digits is not None, digits
        assert from_spelled == from_digits

    def test_the_limit_is_the_number_that_was_asked_for(self):
        intent = detect_top_n_intent(
            canonical_question("les cinq meilleurs clients", "fr"))
        assert intent is not None and intent.limit == 5


class TestTheWordsThatAreNotDates:
    """"en cours" alone means "in progress", "courant" belongs to "compte
    courant", and "neuf" means "new". Every new key is anchored on a calendar
    unit and every count is read only in the count position, so none of these
    acquires a window or loses its meaning."""

    @pytest.mark.parametrize("french", [
        "commandes en cours par succursale",
        "travaux en cours",
        "solde du compte courant",
        "les produits neufs par catégorie",
        "ventes par succursale",
        "quel est le cours du mois précédent du dollar",
    ])
    def test_no_rolling_or_current_window_is_invented(self, french):
        found = detect_temporal_window(canonical_question(french, "fr"))
        assert found.get("kind") not in {
            "this_month", "this_year", "this_quarter", "this_week", "last_n",
        }, (french, found)

    @pytest.mark.parametrize("french,forbidden", [
        ("un produit neuf", "9"),
        ("les produits neufs", "9"),
        ("commandes en cours", "this "),
        ("solde du compte courant", "this "),
    ])
    def test_and_the_canonical_text_is_not_rewritten(self, french, forbidden):
        assert forbidden not in canonical_question(french, "fr")


class TestEnglishIsUntouched:

    @pytest.mark.parametrize("text", [
        "net sales for the current month",
        "net sales for the last 4 weeks",
        "the five best customers",
        "orders in progress by branch",
    ])
    def test_the_text_is_returned_verbatim(self, text):
        assert canonical_question(text, "en") == text

    @pytest.mark.parametrize("text,kind", [
        ("net sales for the current month", "this_month"),
        ("net sales year to date", "this_year"),
        ("net sales for the last 4 weeks", "last_n"),
    ])
    def test_and_detects_exactly_what_it_detected_before(self, text, kind):
        assert detect_temporal_window(text).get("kind") == kind
