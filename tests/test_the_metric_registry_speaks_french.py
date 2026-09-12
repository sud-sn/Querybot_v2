"""
tests/test_the_metric_registry_speaks_french.py

"ventes nettes par succursale" was answered with an ungrouped total, and
"ventes nettes le mois dernier" with the all-time total.

match_metric has two guards, and both exist to stop one thing: a stored SQL
template answering a question it was not written for. The grouping guard skips
the registry when the question asks for a breakdown the template has no GROUP BY
for; the temporal guard skips it when the question scopes a window the matched
wording does not carry. Its own docstring says "a fixed template only answers the
question it was written for".

Both guards are written in ENGLISH, and both ran on the reader's raw text.
Measured across eleven questions and their French twins, NINE diverged, and every
divergence was the same direction -- the guard fired in English and did not fire
in French:

    English                          French                          French got
    net sales by branch    skip      ventes nettes par succursale    the ungrouped total
    net sales last month   skip      ventes nettes le mois dernier   the all-time total
    net sales today        skip      ventes nettes aujourd'hui       the all-time total
    breakdown of net sales skip      répartition des ventes nettes   the ungrouped total

"par" is not "per", and "le mois dernier" is not a window any English regex
reads. Neither guard fired at all.

resolve_metric_scope failed in the opposite direction and just as completely. Its
_norm keeps only [a-z0-9], so an accented French word is SHREDDED rather than
merely unmatched -- "année" becomes "ann e" -- and measured on the same
questions it returned NO metric where the English twin returned one. So the
formula that should have been enforced was not enforced.

The fix is the same in both places and it is not "canonicalise everything":

  * the GUARDS read the canonical English, because they are English;
  * the SYNONYMS are matched against the reader's own words as well, because a
    synonym is admin-authored and may be in either language -- an English-built
    KB saying "net sales" was unreachable in French, and canonicalising alone
    would have made a French-authored synonym unreachable in turn;
  * both sides are FOLDED, because "chiffre d'affaires du mois en cours" typed by
    an admin and typed by a reader shared no byte at the apostrophe.

For an English reader the canonical form IS the input and _fold is exactly
.lower(), so every existing tenant runs the same bytes it ran before.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
import uuid

import pytest

from core.metric_scope import resolve_metric_scope
from core.question_normalizer import canonical_question

TEMPLATE = "SELECT SUM(NET_SLS_AMT) AS NET_SALES FROM EMDW_DMART.CUS_ORD_IVC_FCT"

# (English question, its French twin). Every pair must reach the same verdict.
PAIRS = [
    ("net sales", "ventes nettes"),
    ("net sales by branch", "ventes nettes par succursale"),
    ("net sales per branch", "ventes nettes par succursale"),
    ("breakdown of net sales", "répartition des ventes nettes"),
    ("net sales for each branch", "ventes nettes pour chaque succursale"),
    ("net sales last month", "ventes nettes le mois dernier"),
    ("net sales this month", "ventes nettes du mois en cours"),
    ("net sales for the last 6 months", "ventes nettes des 6 derniers mois"),
    ("net sales today", "ventes nettes aujourd'hui"),
    ("net sales year to date", "ventes nettes depuis le début de l'année"),
    ("net sales year to date", "ventes nettes année à ce jour"),
    ("net sales year to date", "ventes nettes cumul annuel"),
]

# The questions a fixed all-time template must NOT answer, in French.
MUST_DECLINE = [
    "ventes nettes par succursale",
    "ventes nettes par entrepôt",
    "répartition des ventes nettes",
    "ventes nettes pour chaque succursale",
    "ventes nettes le mois dernier",
    "ventes nettes du mois en cours",
    "ventes nettes des 6 derniers mois",
    "ventes nettes aujourd'hui",
    "ventes nettes des 4 dernières semaines",
    "ventes nettes en mars 2026",
]


class _Registry(unittest.TestCase):
    """A real store, because match_metric reads one."""

    def setUp(self):
        import store

        self.store = store
        self._dir = tempfile.mkdtemp(prefix="qb-metric-fr-")
        self._saved = {k: os.environ.get(k) for k in ("DB_PATH", "QUERYBOT_DB_PATH")}
        path = os.path.join(self._dir, "d.db")
        os.environ["DB_PATH"] = path
        os.environ["QUERYBOT_DB_PATH"] = path
        store.init_db()

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self._dir, ignore_errors=True)

    def registry(self, synonyms, name="Net Sales", sql=TEMPLATE):
        account_id = f"acct-{uuid.uuid4().hex[:8]}"
        self.store.upsert_client(account_id, "portal")
        self.store.save_metric(account_id, {
            "name": name, "label": name, "synonyms": synonyms,
            "formula_type": "query", "sql_template": sql,
            "base_table": "EMDW_DMART.CUS_ORD_IVC_FCT",
        })
        return account_id

    def hit(self, account_id, question, lang="en"):
        return bool(self.store.match_metric(account_id, question, lang=lang))


class TestBothGuardsFireInFrench(_Registry):

    SYNONYMS = "net sales, revenue, ventes nettes, chiffre d'affaires"

    def test_the_english_and_french_twins_reach_the_same_verdict(self):
        account_id = self.registry(self.SYNONYMS)
        for english, french in PAIRS:
            with self.subTest(french=french):
                self.assertEqual(
                    self.hit(account_id, english),
                    self.hit(account_id, french, lang="fr"),
                    f"{english!r} and {french!r} disagree",
                )

    def test_a_bare_measure_question_still_uses_the_template(self):
        """Without this the test above would pass by declining everything."""
        account_id = self.registry(self.SYNONYMS)
        self.assertTrue(self.hit(account_id, "ventes nettes", lang="fr"))
        self.assertTrue(self.hit(account_id, "chiffre d’affaires", lang="fr"))
        self.assertTrue(self.hit(account_id, "net sales"))

    def test_a_scoped_or_grouped_question_does_not(self):
        account_id = self.registry(self.SYNONYMS)
        for french in MUST_DECLINE:
            with self.subTest(french=french):
                self.assertFalse(
                    self.hit(account_id, french, lang="fr"),
                    f"{french!r} was answered from the all-time template",
                )


class TestASynonymIsReachableInEitherLanguage(_Registry):

    def test_an_english_authored_metric_is_reachable_in_french(self):
        """The common case: a KB built in English, read in French. The canonical
        form is what makes "chiffre d'affaires" reach a metric called revenue."""
        account_id = self.registry("net sales, revenue, turnover")
        self.assertTrue(self.hit(account_id, "chiffre d’affaires", lang="fr"))
        self.assertTrue(self.hit(account_id, "ventes nettes", lang="fr"))

    def test_a_french_authored_metric_stays_reachable(self):
        """Canonicalising alone would have broken this -- the canonical text no
        longer contains the French words the admin typed."""
        account_id = self.registry("ventes nettes, chiffre d'affaires")
        self.assertTrue(self.hit(account_id, "ventes nettes", lang="fr"))
        self.assertTrue(self.hit(account_id, "chiffre d’affaires", lang="fr"))

    def test_the_apostrophe_shape_does_not_decide(self):
        """The admin typed one and the reader typed the other."""
        account_id = self.registry("chiffre d'affaires")
        for typed in ("chiffre d'affaires", "chiffre d’affaires",
                      "Chiffre d’Affaires", "quel est le chiffre d’affaires"):
            with self.subTest(typed=typed):
                self.assertTrue(self.hit(account_id, typed, lang="fr"), typed)

    def test_an_accent_does_not_decide_either(self):
        account_id = self.registry("marge brute prévue")
        self.assertTrue(self.hit(account_id, "marge brute prévue", lang="fr"))
        self.assertTrue(self.hit(account_id, "marge brute prevue", lang="fr"))

    def test_an_unrelated_question_matches_nothing(self):
        account_id = self.registry("net sales, ventes nettes")
        self.assertFalse(self.hit(account_id, "combien de clients", lang="fr"))
        self.assertFalse(self.hit(account_id, "how many customers"))


class TestTheWindowCarryingSynonymStillWorks(_Registry):
    """The one case a template MAY answer a time-scoped question: the admin
    authored it for that window. Tightening the guard must not close it."""

    def test_an_admin_authored_current_month_metric_matches_its_question(self):
        account_id = self.registry(
            "chiffre d'affaires du mois en cours, ca mensuel", name="CA du mois")
        self.assertTrue(self.hit(
            account_id, "chiffre d’affaires du mois en cours", lang="fr"))

    def test_and_the_english_equivalent_does_too(self):
        account_id = self.registry("revenue this month, mtd revenue",
                                   name="MTD Revenue")
        self.assertTrue(self.hit(account_id, "revenue this month"))

    def test_but_it_declines_a_different_window(self):
        account_id = self.registry(
            "chiffre d'affaires du mois en cours", name="CA du mois")
        self.assertFalse(self.hit(
            account_id, "chiffre d’affaires le mois dernier", lang="fr"))
        self.assertFalse(self.hit(
            account_id, "chiffre d’affaires des 6 derniers mois", lang="fr"))

    def test_and_an_absolute_date_never_qualifies(self):
        account_id = self.registry(
            "chiffre d'affaires du mois en cours", name="CA du mois")
        self.assertFalse(self.hit(
            account_id, "chiffre d’affaires en mars 2026", lang="fr"))

    def test_and_a_grouping_still_wins_over_the_window_match(self):
        account_id = self.registry(
            "chiffre d'affaires du mois en cours", name="CA du mois")
        self.assertFalse(self.hit(
            account_id, "chiffre d’affaires du mois en cours par succursale",
            lang="fr"))


class TestEnglishRunsTheBytesItRanBefore(_Registry):

    def test_the_default_language_needs_no_argument(self):
        account_id = self.registry("net sales, revenue")
        self.assertTrue(bool(self.store.match_metric(account_id, "net sales")))
        self.assertFalse(bool(self.store.match_metric(
            account_id, "net sales by branch")))

    def test_passing_english_explicitly_changes_nothing(self):
        account_id = self.registry("net sales, revenue")
        for question in [q for q, _fr in PAIRS]:
            with self.subTest(question=question):
                self.assertEqual(
                    bool(self.store.match_metric(account_id, question)),
                    bool(self.store.match_metric(account_id, question, lang="en")),
                )

    def test_an_unknown_language_is_treated_as_english(self):
        account_id = self.registry("net sales, revenue")
        self.assertTrue(bool(self.store.match_metric(
            account_id, "net sales", lang="de")))
        self.assertFalse(bool(self.store.match_metric(
            account_id, "net sales by branch", lang="de")))


# ── resolve_metric_scope ─────────────────────────────────────────────────────

EN_METRICS = [
    {"name": "Net Sales", "synonyms": "net sales, revenue",
     "base_table": "EMDW_DMART.CUS_ORD_IVC_FCT", "formula_type": "expression",
     "sql_template": "SUM(NET_SLS_AMT)"},
    {"name": "Returns", "synonyms": "returns, credits",
     "base_table": "EMDW_DMART.CUS_RTN_FCT", "formula_type": "expression",
     "sql_template": "SUM(RTN_AMT)"},
]
FR_METRIC = {
    "name": "Ventes nettes", "synonyms": "ventes nettes, chiffre d'affaires",
    "base_table": "EMDW_DMART.CUS_ORD_IVC_FCT", "formula_type": "expression",
    "sql_template": "SUM(NET_SLS_AMT)",
}
COLUMNS = {
    "EMDW_DMART.CUS_ORD_IVC_FCT": {"NET_SLS_AMT": "decimal", "WHS_NBR": "int"},
    "EMDW_DMART.CUS_RTN_FCT": {"RTN_AMT": "decimal", "WHS_NBR": "int"},
}


def scoped(metrics, french):
    """As the pipeline calls it: canonical text, reader's words alongside."""
    result = resolve_metric_scope(
        metrics, canonical_question(french, "fr"), COLUMNS,
        reader_question=french)
    return [metric["name"] for metric in result.metrics]


class TestTheFormulaIsStillEnforcedInFrench:

    @pytest.mark.parametrize("english,french,expected", [
        ("net sales by branch", "ventes nettes par succursale", "Net Sales"),
        ("revenue last month", "chiffre d'affaires le mois dernier", "Net Sales"),
        ("returns by branch", "retours par succursale", "Returns"),
    ])
    def test_the_same_metric_is_scoped_as_in_english(
            self, english, french, expected):
        english_names = [
            metric["name"] for metric in
            resolve_metric_scope(EN_METRICS, english, COLUMNS).metrics
        ]
        assert expected in english_names, english
        assert expected in scoped(EN_METRICS, french), french

    def test_a_word_with_no_french_entry_still_reaches_its_english_synonym(self):
        """"avoirs" is French for credit notes and equally for assets, so it is
        deliberately NOT in the canonicaliser -- reading one as the other points
        a question at the wrong fact. The cost is bounded, and this is the bound:
        the metric stays reachable by the English synonym the admin authored,
        because the reader's own words are scored too."""
        assert scoped(EN_METRICS, "credits par succursale") == ["Returns"]
        assert scoped(EN_METRICS, "avoirs par succursale") == []

    def test_a_french_authored_metric_is_found_by_its_own_words(self):
        assert "Ventes nettes" in scoped(EN_METRICS + [FR_METRIC],
                                        "ventes nettes par succursale")

    def test_and_an_english_one_is_not_lost_when_both_exist(self):
        names = scoped(EN_METRICS + [FR_METRIC], "ventes nettes par succursale")
        assert "Net Sales" in names

    def test_an_unrelated_question_still_scopes_nothing(self):
        assert scoped(EN_METRICS, "combien de succursales avons-nous") == []

    def test_english_is_unchanged_when_no_reader_question_is_given(self):
        for english in ("net sales by branch", "returns by branch", "revenue"):
            with_arg = [
                m["name"] for m in resolve_metric_scope(
                    EN_METRICS, english, COLUMNS, reader_question=english).metrics
            ]
            without = [
                m["name"] for m in
                resolve_metric_scope(EN_METRICS, english, COLUMNS).metrics
            ]
            assert with_arg == without, english


class TestTheMeasureNounsAFrenchReaderActuallyTypes:
    """The canonicaliser knew "ventes" and not "ventes nettes", so EMCO's
    headline measure came out "sales nettes" -- matching no synonym, no KB field
    and no BM25 token. The returns fact had no English name at all."""

    @pytest.mark.parametrize("french,expected", [
        ("ventes nettes", "net sales"),
        ("ventes brutes", "gross sales"),
        ("retours", "returns"),
        ("retour", "returns"),
    ])
    def test_the_measure_is_named_in_english(self, french, expected):
        assert canonical_question(french, "fr") == expected

    def test_return_on_investment_is_not_a_returns_question(self):
        """Longest-first ordering is what keeps these apart, and getting it
        wrong would point an ROI question at the credit-notes fact."""
        assert canonical_question("retour sur investissement", "fr") == \
            "return on investment"
        assert canonical_question(
            "le retour sur investissement par produit", "fr") == \
            "the return on investment by product"

    def test_the_plain_noun_is_untouched(self):
        assert canonical_question("les ventes", "fr") == "the sales"


class TestThePipelinePassesTheLanguageThrough:
    """match_metric and resolve_metric_scope can only do this if the pipeline
    tells them the language, and those call sites sit inside a 6,600-line
    coroutine that needs a warehouse and a websocket to execute. So they are read
    as a SYNTAX TREE: the question is which keyword arguments the calls carry,
    which is a structural fact rather than a substring.
    """

    @staticmethod
    def _calls(name):
        import ast
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1]
                  / "core" / "query_pipeline.py").read_text(encoding="utf-8")
        found = []
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if called == name:
                found.append(node)
        return found

    def test_match_metric_is_told_the_readers_language(self):
        calls = self._calls("match_metric")
        assert calls, "store.match_metric is no longer called from the pipeline"
        for call in calls:
            assert "lang" in {kw.arg for kw in call.keywords}

    def test_every_metric_scope_call_carries_the_readers_own_words(self):
        calls = self._calls("resolve_metric_scope")
        assert len(calls) >= 2, len(calls)
        for call in calls:
            assert "reader_question" in {kw.arg for kw in call.keywords}

    def test_the_scope_calls_are_given_the_canonical_text_positionally(self):
        """The second positional argument is the text the English scorer reads.
        Passing the raw question there is the defect this closes, and it is not
        visible in the keyword list."""
        import ast

        for call in self._calls("resolve_metric_scope"):
            assert len(call.args) >= 2
            question_arg = call.args[1]
            assert isinstance(question_arg, ast.Name), ast.dump(question_arg)
            assert question_arg.id == "_semantic_plan_question", question_arg.id
