import unittest

from core.masking import (
    detect_sensitive_columns, mask_rows, strategy_for_field,
    scrub_embedded_pii, scrub_unmasked_free_text,
)
from core.schema import _resolve_masking_fields


class MaskingPrivacyTests(unittest.TestCase):
    def test_free_text_name_strategy_redacts(self):
        cols = [{"name": "notes", "type": "varchar(100)"}]
        rows = [{"notes": "Patient has HIV and phone 555-1234"}]
        detected = detect_sensitive_columns(cols)
        self.assertEqual(detected["notes"], "free_text")
        masked = mask_rows(rows, set(detected), cols)
        self.assertEqual(masked[0]["notes"], "[REDACTED TEXT]")

    def test_free_text_type_strategy_redacts(self):
        cols = [{"name": "misc_blob", "type": "TEXT"}]
        rows = [{"misc_blob": "SSN 123-45-6789"}]
        detected = detect_sensitive_columns(cols)
        masked = mask_rows(rows, set(detected), cols)
        self.assertEqual(masked[0]["misc_blob"], "[REDACTED TEXT]")

    def test_distinct_scan_can_skip_masked_columns(self):
        cols = [
            {"name": "status", "type": "varchar(20)"},
            {"name": "notes", "type": "varchar(100)"},
        ]
        masked = _resolve_masking_fields(cols, "auto", set())
        self.assertIn("notes", masked)
        self.assertNotIn("status", masked)


class HealthcareIdentifierPatternTests(unittest.TestCase):
    """B1 — MRN / NPI / DEA / license / policy / payment identifiers must
    resolve to a redacting identifier strategy, never a passthrough/text_mask."""

    REDACTING = {"ssn", "credit_card", "redact", "free_text"}

    def _strategy(self, name: str) -> str:
        return strategy_for_field(name, "varchar(40)")

    def test_mrn(self):
        self.assertEqual(self._strategy("MRN"), "ssn")
        self.assertEqual(self._strategy("medical_record_number"), "ssn")

    def test_npi(self):
        self.assertEqual(self._strategy("NPI"), "ssn")
        self.assertEqual(self._strategy("provider_number"), "ssn")

    def test_dea(self):
        self.assertEqual(self._strategy("dea_number"), "ssn")

    def test_license(self):
        self.assertEqual(self._strategy("prescriber_license"), "ssn")
        self.assertEqual(self._strategy("license_no"), "ssn")

    def test_policy_member(self):
        self.assertEqual(self._strategy("policy_number"), "ssn")
        self.assertEqual(self._strategy("member_id"), "ssn")
        self.assertEqual(self._strategy("subscriber_id"), "ssn")

    def test_payment_aliases(self):
        self.assertEqual(self._strategy("payment_token"), "credit_card")
        self.assertEqual(self._strategy("account_number"), "credit_card")
        self.assertEqual(self._strategy("iban"), "credit_card")

    def test_clinical_free_text(self):
        for nm in ("clinical_notes", "chief_complaint", "diagnosis_text",
                   "observation", "assessment"):
            self.assertEqual(self._strategy(nm), "free_text", nm)

    def test_all_identifiers_redact(self):
        for nm in ("MRN", "NPI", "dea_number", "policy_number",
                   "payment_token", "clinical_notes"):
            self.assertIn(self._strategy(nm), self.REDACTING, nm)


class EmbeddedPiiScrubTests(unittest.TestCase):
    """B2 — embedded PII in an UNFLAGGED narrative column is scrubbed."""

    def test_scrub_single_value(self):
        txt = "Patient called from john.doe@mail.com or 555-123-4567, SSN 123-45-6789, dob 1985-03-15"
        out = scrub_embedded_pii(txt)
        self.assertNotIn("john.doe@mail.com", out)
        self.assertNotIn("123-45-6789", out)
        self.assertNotIn("555-123-4567", out)
        self.assertNotIn("1985-03-15", out)
        self.assertIn("[EMAIL]", out)
        self.assertIn("[SSN]", out)

    def test_scrub_unmasked_column(self):
        # 'remark_field' is NOT matched by name/type heuristics, so it ships
        # unmasked — the scrubber is the safety net.
        cols = [{"name": "remark_field", "type": "varchar(60)"}]
        rows = [{"remark_field": "called patient at 555-123-4567 today about refill"}]
        out = scrub_unmasked_free_text(rows, cols, skip_fields=set())
        self.assertNotIn("555-123-4567", out[0]["remark_field"])
        self.assertIn("[PHONE]", out[0]["remark_field"])

    def test_short_codes_untouched(self):
        cols = [{"name": "code", "type": "varchar(20)"}]
        rows = [{"code": "AB-1234"}]
        out = scrub_unmasked_free_text(rows, cols, skip_fields=set())
        self.assertEqual(out[0]["code"], "AB-1234")


class PerAccountSeedingTests(unittest.TestCase):
    """B3 — masking is account-isolated; synthetic differs across accounts."""

    def test_mask_rows_differs_by_seed_key(self):
        cols = [{"name": "patient_name", "type": "varchar(60)"}]
        rows = [{"patient_name": "Jane Doe"}]
        a = mask_rows([dict(rows[0])], {"patient_name"}, cols, seed_key="acct_A")
        b = mask_rows([dict(rows[0])], {"patient_name"}, cols, seed_key="acct_B")
        a2 = mask_rows([dict(rows[0])], {"patient_name"}, cols, seed_key="acct_A")
        # Deterministic within an account, different across accounts.
        self.assertEqual(a[0]["patient_name"], a2[0]["patient_name"])
        self.assertNotEqual(a[0]["patient_name"], b[0]["patient_name"])

    def test_synthetic_differs_by_account(self):
        from core.synthetic import generate_synthetic_sample
        cols = [{"name": "first_name", "type": "VARCHAR"}]
        a = generate_synthetic_sample(cols, n_rows=3, seed="acct_A")
        b = generate_synthetic_sample(cols, n_rows=3, seed="acct_B")
        a2 = generate_synthetic_sample(cols, n_rows=3, seed="acct_A")
        self.assertEqual(a, a2)            # deterministic within account
        self.assertNotEqual(a, b)          # differs across accounts


class ModeNoneGuardTests(unittest.TestCase):
    """B4 — mode='none' without allow_unmasked downgrades to auto (masks PII)."""

    def test_mode_none_downgrades_to_auto(self):
        from core.schema import _apply_masking
        cols = [
            {"name": "email", "type": "varchar(80)"},
            {"name": "status", "type": "varchar(10)"},
        ]
        rows = [{"email": "real.person@corp.com", "status": "A"}]
        out, masked, repl, synth = _apply_masking(
            fetch_fn=lambda: [dict(rows[0])],
            col_defs=cols, mode="none", explicit_fields=set(),
            table_name="PATIENTS", seed_key="acct_A", allow_unmasked=False,
        )
        # email must be masked despite mode='none'
        self.assertIn("email", masked)
        self.assertNotEqual(out[0]["email"], "real.person@corp.com")

    def test_mode_none_allowed_when_opted_in(self):
        from core.schema import _apply_masking
        cols = [{"name": "email", "type": "varchar(80)"}]
        rows = [{"email": "real.person@corp.com"}]
        out, masked, repl, synth = _apply_masking(
            fetch_fn=lambda: [dict(rows[0])],
            col_defs=cols, mode="none", explicit_fields=set(),
            table_name="PATIENTS", seed_key="acct_A", allow_unmasked=True,
        )
        self.assertEqual(masked, set())
        self.assertEqual(out[0]["email"], "real.person@corp.com")


if __name__ == "__main__":
    unittest.main()
