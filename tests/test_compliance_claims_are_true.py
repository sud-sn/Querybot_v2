"""
tests/test_compliance_claims_are_true.py

Two places where the product asserted something about its own compliance that
was not true. Different mechanisms, same class of failure: a statement the
code could not support.

THE REPAIR PROMPT LEAKED THE VALUES THE LINE ABOVE MASKED. The retry prompt
built after a failed execution reads, in order:

    Error: {scrub_error_for_llm(exec_error)}
    What that means: {diagnosis}

`diagnosis` comes from `sanitize_db_error`, which returns a catalogue sentence
when a pattern in `_DB_ERROR_MAP` matches and the DRIVER'S OWN FIRST SENTENCE
when none does -- verbatim and deliberately, so support can search on it. That
is right for the user's error card and wrong two lines under a scrubbed error.
Executed against the shipped code:

    raw        Arithmetic overflow error for type varchar, value = 1234567.890000.
    line 1     ... value = [number].[number].          <- masked
    line 3     ... value = 1234567.890000.             <- not masked, same prompt

THE PROOF PACK TOLD AN AUDITOR NOTHING HAPPENED. `calls_section` reported
"No model calls were made in this period." whenever the log was empty. But
`enable_llm_audit` is `INTEGER NOT NULL DEFAULT 0`, so a workspace that never
switched auditing on has an empty log by construction. Absence of a record and
absence of a call are different claims, and the pack was making the one it
could not support -- in the one document whose entire purpose is being true.
"""

import unittest

from core.execution_correction import diagnose_execution_error
from core.failure_messages import scrub_error_for_llm

# A driver sentence that matches no pattern in _DB_ERROR_MAP, so sanitize_db_error
# falls through to returning it verbatim. That fallback is the whole defect.
UNMATCHED_ERROR = (
    "Arithmetic overflow error for type varchar, value = 1234567.890000."
)


class TheRepairPromptNeverCarriesAnUnmaskedValue(unittest.TestCase):

    def test_the_diagnosis_channel_really_does_return_raw_text(self):
        """Guards the premise. If sanitize_db_error ever starts scrubbing its
        own fallback, the fix below becomes redundant and this test says so
        rather than passing silently."""
        diagnosis = diagnose_execution_error(UNMATCHED_ERROR).diagnosis
        self.assertIn("1234567.890000", diagnosis,
                      "the unmatched fallback no longer returns the raw sentence")

    def test_scrubbing_it_removes_the_value(self):
        """The fix, executed: what the prompt now interpolates."""
        diagnosis = diagnose_execution_error(UNMATCHED_ERROR).diagnosis
        self.assertNotIn("1234567.890000", scrub_error_for_llm(diagnosis))
        self.assertIn("[number]", scrub_error_for_llm(diagnosis))

    def test_a_quoted_data_value_is_masked_too(self):
        error = ("Conversion failed when converting the varchar value "
                 "'CUST-99881' to data type int.")
        diagnosis = diagnose_execution_error(error).diagnosis
        self.assertNotIn("CUST-99881", scrub_error_for_llm(diagnosis))

    def test_a_matched_diagnosis_survives_scrubbing_intact(self):
        """Scrubbing must not damage the catalogue sentences, which are the
        reason the diagnosis channel exists at all."""
        error = "Invalid object name 'dbo.NO_SUCH_TABLE'."
        diagnosis = diagnose_execution_error(error).diagnosis
        self.assertTrue(diagnosis)
        self.assertEqual(scrub_error_for_llm(diagnosis).count("[value]"),
                         diagnosis.count("[value]"))

    def test_both_diagnosis_lines_in_the_prompt_are_scrubbed(self):
        """The prompt interpolates diagnosis AND next_step. Executed against
        the pipeline source because building the prompt needs a live governed
        run; the assertion is that neither reaches the f-string unwrapped."""
        import inspect
        import core.query_pipeline as qp

        source = inspect.getsource(qp._handle_query_impl)
        marker = "What that means: "
        self.assertIn(marker, source)
        window = source[source.index(marker) - 200:source.index(marker) + 400]
        self.assertIn("scrub_error_for_llm(_diagnosis.diagnosis)", window)
        self.assertIn("scrub_error_for_llm(_diagnosis.next_step)", window)
        self.assertNotIn("{_diagnosis.diagnosis}", window)
        self.assertNotIn("{_diagnosis.next_step}", window)


class TheProofPackOnlyClaimsWhatItCanEvidence(unittest.TestCase):

    def setUp(self):
        """A fresh account per test, and the flag set explicitly.

        Setting QUERYBOT_DB here does not isolate anything: `store` is already
        imported by the time setUp runs, so its database path is long since
        resolved. An earlier draft relied on that and the suite was
        order-dependent -- the "auditing on" test set the flag, the shared row
        kept it, and the three tests that sort after it failed. Per-test
        account ids plus an explicit write are what actually isolate them.
        """
        import store

        store.init_db()
        self.account = f"acct_proof_pack_{self.id().rsplit('.', 1)[-1]}"
        store.upsert_client(self.account, "web")
        store.update_client_meta(self.account, enable_llm_audit=0)

    def tearDown(self):
        import store

        try:
            store.update_client_meta(self.account, enable_llm_audit=0)
        except Exception:
            pass

    def _section(self):
        from core.compliance.proof_pack import calls_section
        return calls_section(self.account, 30)

    def test_auditing_is_off_by_default(self):
        """The premise. If this default ever changes, the distinction below
        stops mattering and this test is where that shows up."""
        import store
        self.assertFalse(bool((store.get_client(self.account) or {})
                              .get("enable_llm_audit")))

    def test_an_empty_log_with_auditing_off_is_not_a_clean_period(self):
        section = self._section()
        self.assertFalse(section["attestable"])
        self.assertNotIn("No model calls were made", section["statement"])
        self.assertIn("not enabled", section["statement"])

    def test_it_says_what_to_do_about_it(self):
        self.assertIn("Enable call auditing", self._section()["statement"])

    def test_an_empty_log_with_auditing_on_is_a_clean_period(self):
        import store
        store.update_client_meta(self.account, enable_llm_audit=1)
        section = self._section()
        self.assertTrue(section["attestable"])
        self.assertEqual(section["statement"],
                         "No model calls were made in this period.")

    def test_the_flag_is_reported_in_the_section(self):
        """An auditor reading the JSON should not have to infer it from the
        prose."""
        self.assertIn("call_audit_enabled", self._section())
        self.assertFalse(self._section()["call_audit_enabled"])


if __name__ == "__main__":
    unittest.main()
