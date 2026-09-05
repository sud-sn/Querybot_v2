"""
tests/test_drill_dimension_language.py

Every exit from the "Break down by X" chip, in both languages.

core/drill_dimension.py's docstring says "every failure exits with a clear
fallback message -- never a Python exception propagated to the WebSocket".
That was true and entirely English: seven fallbacks and a title, concatenated
around the dimension name. The chip itself is rendered in French by the chat
page, so a French reader clicked a French button and got an English refusal.

Each test here runs the real generate_drill_by_dimension down one of its exits
and asserts on the dict it returns. The dimension name is the tenant's own word
from the semantic model and is interpolated into both languages unchanged --
translating it would name a dimension that does not exist.
"""

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import i18n
from core.drill_dimension import generate_drill_by_dimension

DIM = {
    "name": "Region",
    "display_table": "DBO.DIM_REGION", "display_column": "REGION_NAME",
    "source_table": "DBO.F_SALES", "source_key_column": "REGION_ID",
    "display_key_column": "REGION_ID",
}
PLAN = {"available_dimensions": [DIM]}
SQL = "SELECT SUM(AMOUNT) AS REVENUE FROM DBO.F_SALES"


def _drill(lang, *, plan=PLAN, sql=SQL, executor=None):
    token = i18n.activate_language(lang)
    try:
        return asyncio.run(generate_drill_by_dimension(
            dim_name="Region", rows=[{"REVENUE": 100}], question="show revenue",
            original_sql=sql, semantic_plan=plan,
            db_cfg={"db_type": "azure_sql", "credentials": {}},
            known_tables={"DBO.F_SALES", "DBO.DIM_REGION"},
            provider="test", model="test", api_key="",
            query_executor=executor,
        ))
    finally:
        i18n.deactivate_language(token)


class EveryFallbackIsTranslated(unittest.TestCase):

    def _both(self, **kw):
        return _drill("en", **kw), _drill("fr", **kw)

    def test_the_title_names_the_dimension_in_both_languages(self):
        english, french = self._both(plan={})
        self.assertEqual(english["title"], "Break down by Region")
        self.assertEqual(french["title"], "Ventiler par Region")

    def test_the_action_key_is_a_wire_token(self):
        """The browser routes the error card on this. A translated action key
        would break the card before anyone read it."""
        french = _drill("fr", plan={})
        self.assertEqual(french["action"], "drill_dim")
        self.assertEqual(french["type"], "assistant_error")

    def test_an_unknown_dimension(self):
        english, french = self._both(plan={})
        self.assertIn("not found in the semantic model", english["content"])
        self.assertIn("introuvable dans le modèle sémantique", french["content"])
        self.assertIn("Region", french["content"])
        self.assertIn("Ventile par Region", french["suggestion"])

    def test_a_dimension_that_will_not_compile_into_a_join(self):
        """A dimension the plan knows about but whose metadata cannot produce a
        governed join -- the SQL here has no FROM the compiler can attach to."""
        english, french = self._both(sql="")
        self.assertIn("not safely joinable", english["content"])
        self.assertIn("ne peut pas être jointe en toute sécurité", french["content"])
        self.assertIn("ventilation par Region", french["suggestion"])

    def test_a_rewritten_query_that_fails_validation(self):
        with patch("core.validator.validate_sql", return_value=(False, "nope", "")):
            english, french = self._both()
        self.assertEqual(english["content"], "The rewritten query failed validation.")
        self.assertEqual(french["content"],
                         "La requête réécrite n'a pas passé la validation.")
        self.assertIn("Affiche [indicateur] par Region", french["suggestion"])

    def test_a_validator_that_raises(self):
        with patch("core.validator.validate_sql", side_effect=RuntimeError("boom")):
            english, french = self._both()
        self.assertIn("Validation error", english["content"])
        self.assertIn("Erreur de validation", french["content"])

    def test_a_query_that_fails_to_execute(self):
        """The database's own error text is data and rides through untouched --
        only the sentence around it is copy."""
        def _boom(db_cfg, sql):
            raise RuntimeError("ODBC-12514 listener refused")

        english, french = self._both(executor=_boom)
        self.assertIn("failed to execute", english["content"])
        self.assertIn("a échoué", french["content"])
        for card in (english, french):
            self.assertIn("ODBC-12514 listener refused", card["content"])

    def test_a_breakdown_that_comes_back_empty(self):
        class _Empty:
            rows: list = []
            sql = SQL

        english, french = self._both(executor=lambda db_cfg, sql: _Empty())
        self.assertIn("returned no data", english["content"])
        self.assertIn("n'a renvoyé aucune donnée", french["content"])
        self.assertIn("période filtrée", french["suggestion"])

    def test_no_fallback_leaves_an_unresolved_catalogue_id_on_screen(self):
        """lookup() returns the id when an entry is missing, so a typo ships as
        `reply.drill.no_data` in the error card rather than raising."""
        cases = [
            dict(plan={}),
            dict(sql=""),
            dict(executor=lambda db_cfg, sql: (_ for _ in ()).throw(RuntimeError("x"))),
        ]
        for lang in i18n.SUPPORTED_LANGUAGES:
            for kw in cases:
                with self.subTest(lang=lang, case=sorted(kw)):
                    card = _drill(lang, **kw)
                    for field in ("title", "content", "suggestion"):
                        self.assertNotIn("reply.", card[field], field)


class TheQuestionStaysCanonicalEnglish(unittest.TestCase):
    """The drill question is read by every detector build_assistant_response
    runs, and gateway/webhooks.py caches it as the question the NEXT result
    action reads. It is a wire value, and this pins it as one."""

    def test_the_suffix_is_not_translated(self):
        class _Rows:
            rows = [{"REGION_NAME": "North", "REVENUE": 100}]
            sql = "SELECT REGION_NAME, SUM(AMOUNT) AS REVENUE FROM DBO.F_SALES"

        with patch("core.validator.validate_sql", return_value=(True, "", "")):
            french = _drill("fr", executor=lambda db_cfg, sql: _Rows())
        self.assertEqual(french["type"], "assistant_response")
        self.assertEqual(french["drill_dimension"], "Region")
        self.assertIn("broken down by Region", french["question"])


if __name__ == "__main__":
    unittest.main()
