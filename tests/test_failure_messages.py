"""
tests/test_failure_messages.py

Business-readable hard-failure messages:
  1. sanitize_db_error — driver-prefix stripping + plain-language mapping
  2. translate_failure — validation/execution/unknown kinds, never raises
  3. format_failure_business_response — diagnostic-card label conventions
  4. Pipeline wiring — raw validator prose / raw ODBC text no longer sent
"""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.failure_messages import sanitize_db_error, translate_failure
from core.answer_formatter import format_failure_business_response


class SanitizeDbErrorTests(unittest.TestCase):
    def test_strips_pyodbc_tuple_and_driver_prefixes(self):
        raw = (
            "('42S02', \"[Microsoft][ODBC Driver 18 for SQL Server][SQL Server]"
            "Invalid object name 'dbo.FOO'. (208) (SQLExecDirectW)\")"
        )
        info = sanitize_db_error(raw)
        self.assertNotIn("[Microsoft]", info["cleaned"])
        self.assertNotIn("(SQLExecDirectW)", info["cleaned"])
        self.assertIn("Invalid object name", info["cleaned"])
        self.assertIn("does not exist", info["plain_reason"])
        self.assertIn("schema discovery", info["next_step"])

    def test_login_timeout_mapping(self):
        info = sanitize_db_error(
            "('HYT00', '[Microsoft][ODBC Driver 18 for SQL Server]Login timeout expired (0) (SQLDriverConnect)')"
        )
        self.assertIn("did not respond in time", info["plain_reason"])

    def test_permission_denied_mapping(self):
        info = sanitize_db_error(
            "[Microsoft][ODBC Driver 18 for SQL Server][SQL Server]The SELECT permission was denied on the object 'FNN_FCT'. (229)"
        )
        self.assertIn("permission", info["plain_reason"].lower())
        self.assertIn("grant read access", info["next_step"])

    def test_multipart_identifier_mapping(self):
        info = sanitize_db_error(
            "[Microsoft][ODBC Driver 18 for SQL Server][SQL Server]The multi-part identifier \"due_dt.DMS_DT\" could not be bound. (4104)"
        )
        self.assertIn("was not joined", info["plain_reason"])

    def test_group_by_error_mapping(self):
        info = sanitize_db_error(
            "[SQL Server]Column 'X.PAY_DT_DMS_KEY' is invalid in the select list because "
            "it is not contained in either an aggregate function or the GROUP BY clause. (8120)"
        )
        self.assertIn("grouped and ungrouped", info["plain_reason"])

    def test_unknown_error_falls_back_to_first_sentence(self):
        info = sanitize_db_error(
            "[Vendor][Driver]Frobnication register overflow at cluster 7. Contact DBA."
        )
        self.assertEqual(info["plain_reason"], "Frobnication register overflow at cluster 7.")
        self.assertNotIn("[Vendor]", info["plain_reason"])

    def test_empty_input_is_safe(self):
        info = sanitize_db_error("")
        self.assertTrue(info["plain_reason"])
        self.assertTrue(info["next_step"])


class TranslateFailureTests(unittest.TestCase):
    _KEYS = {"headline", "most_likely_reason", "suggested_next_step", "technical_notes"}

    def test_validation_kind_maps_known_code(self):
        rca = translate_failure(
            kind="validation", code="field_plan_mismatch",
            reason="Generated SQL does not follow the semantic field-source plan. SQL did not use required semantic field customer: CUS_DMS.CUS_NM.",
        )
        self.assertEqual(set(rca), self._KEYS)
        self.assertIn("trusted query", rca["headline"])
        self.assertIn("approved business field mapping", rca["most_likely_reason"])
        # Raw validator prose demoted to technical notes, not lost.
        self.assertTrue(any("CUS_NM" in n for n in rca["technical_notes"]))

    def test_validation_unknown_code_gets_generic_reason(self):
        rca = translate_failure(kind="validation", code="something_new", reason="raw text")
        self.assertIn("safety and accuracy checks", rca["most_likely_reason"])

    def test_execution_kind_uses_sanitizer(self):
        rca = translate_failure(
            kind="execution",
            exception_text="[Microsoft][ODBC Driver 18 for SQL Server]Login timeout expired",
        )
        self.assertIn("could not run this query", rca["headline"])
        self.assertIn("did not respond in time", rca["most_likely_reason"])
        self.assertTrue(all("[Microsoft]" not in n for n in rca["technical_notes"]))

    def test_unknown_kind_and_empty_inputs_never_raise(self):
        rca = translate_failure(kind="galactic", code="", reason="", exception_text="")
        self.assertEqual(set(rca), self._KEYS)
        self.assertTrue(rca["headline"])


class FormatFailureResponseTests(unittest.TestCase):
    def test_labels_match_diagnostic_card_conventions(self):
        rca = translate_failure(kind="execution", exception_text="Login timeout expired")
        text = format_failure_business_response(rca=rca, sql="SELECT 1", sql_preview_fn=lambda s: s)
        self.assertIn("Most likely reason:", text)
        self.assertIn("Suggested next step:", text)
        self.assertIn("Technical details:", text)
        self.assertIn("SQL tried:\n```sql\nSELECT 1\n```", text)

    def test_sql_fence_omitted_without_sql(self):
        rca = translate_failure(kind="validation", code="cannot_generate", reason="x")
        text = format_failure_business_response(rca=rca, sql="")
        self.assertNotIn("SQL tried:", text)


class PipelineWiringTests(unittest.TestCase):
    def test_pipeline_no_longer_sends_raw_failures(self):
        src = (ROOT / "core" / "query_pipeline.py").read_text(encoding="utf-8")
        # The raw exec_error may still feed the internal LLM repair prompt,
        # but the user-facing "Database error" send must be gone.
        self.assertNotIn("Database error — could not execute after retry", src)
        self.assertIn("format_failure_business_response", src)
        self.assertIn('kind="execution"', src)
        self.assertIn('kind="validation"', src)
        # Non-validator codes (policy denials) still pass through verbatim —
        # the raw send remains but only inside the else branch after the
        # _VALIDATION_REASONS gate.
        self.assertIn("_VALIDATION_REASONS", src)

    def test_portal_kicker_is_derived(self):
        tmpl = (ROOT / "portal" / "templates" / "portal_chat.html").read_text(encoding="utf-8")
        self.assertIn("'Query failed'", tmpl)
        self.assertIn("'Validation issue'", tmpl)
        self.assertIn("${kicker}", tmpl)


if __name__ == "__main__":
    unittest.main()
