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


class DrugNameStrategyTests(unittest.TestCase):
    """Drug/medication name columns must get the drug_name strategy — the
    bare 'name' pattern used to match their _NAME suffix and substitute fake
    PERSON names, misleading KB generation into treating the field as a
    person identifier."""

    def _strategy(self, name: str) -> str:
        return strategy_for_field(name, "varchar(80)")

    def test_drug_columns_get_drug_name_strategy(self):
        for nm in ("DRUG_NAME", "RX_DRUG_NAME", "MEDICATION_NAME",
                   "MED_NAME", "COMPOUND_NAME", "INGREDIENT_NAME",
                   "GENERIC_NAME"):
            self.assertEqual(self._strategy(nm), "drug_name", nm)

    def test_person_name_columns_unchanged(self):
        self.assertEqual(self._strategy("FIRST_NAME"), "first_name")
        self.assertEqual(self._strategy("LAST_NAME"), "last_name")
        self.assertEqual(self._strategy("CUSTOMER_NAME"), "name")
        self.assertEqual(self._strategy("PATIENT_NAME"), "name")

    def test_masked_value_is_fictional_drug_not_person(self):
        from core.masking import _DRUG_NAMES, _FIRST, _LAST
        cols = [{"name": "DRUG_NAME", "type": "varchar(80)"}]
        rows = [{"DRUG_NAME": "Metformin"}]
        masked = mask_rows(rows, {"DRUG_NAME"}, cols, seed_key="acct-1")
        fake = masked[0]["DRUG_NAME"]
        self.assertIn(fake, _DRUG_NAMES)
        self.assertNotIn(fake, _FIRST)
        self.assertNotIn(fake, _LAST)

    def test_deterministic_across_calls(self):
        cols = [{"name": "DRUG_NAME", "type": "varchar(80)"}]
        rows = [{"DRUG_NAME": "Metformin"}]
        a = mask_rows(rows, {"DRUG_NAME"}, cols, seed_key="acct-1")[0]["DRUG_NAME"]
        b = mask_rows(rows, {"DRUG_NAME"}, cols, seed_key="acct-1")[0]["DRUG_NAME"]
        self.assertEqual(a, b)

    def test_nondeterministic_path_also_uses_drug_pool(self):
        from core.masking import _DRUG_NAMES
        cols = [{"name": "DRUG_NAME", "type": "varchar(80)"}]
        rows = [{"DRUG_NAME": "Metformin"}]
        masked = mask_rows(rows, {"DRUG_NAME"}, cols)  # no seed_key
        self.assertIn(masked[0]["DRUG_NAME"], _DRUG_NAMES)


class SafeFakeIdentifierTests(unittest.TestCase):
    """Generated fake identifiers must be provably fake — SSA never issues
    area numbers 900-999, and NNN-555-0100..0199 is the NANP-reserved
    fictional phone block. Random values in the real ranges could collide
    with an actual person's identifier."""

    def _mask_one(self, col: str, value: str, seed_key: str = ""):
        cols = [{"name": col, "type": "varchar(40)"}]
        return mask_rows([{col: value}], {col}, cols, seed_key=seed_key)[0][col]

    def test_ssn_area_never_issued_deterministic(self):
        for i in range(20):
            fake = self._mask_one("SSN", f"123-45-{6000 + i}", seed_key="acct-1")
            area = int(fake.split("-")[0])
            self.assertGreaterEqual(area, 900, fake)

    def test_ssn_area_never_issued_nondeterministic(self):
        for i in range(20):
            fake = self._mask_one("SSN", f"123-45-{6000 + i}")
            area = int(fake.split("-")[0])
            self.assertGreaterEqual(area, 900, fake)

    def test_phone_in_reserved_fictional_block_deterministic(self):
        for i in range(20):
            fake = self._mask_one("PHONE", f"+1-415-867-{5300 + i}", seed_key="acct-1")
            self.assertIn("-555-01", fake, fake)

    def test_phone_in_reserved_fictional_block_nondeterministic(self):
        for i in range(20):
            fake = self._mask_one("PHONE", f"+1-415-867-{5300 + i}")
            self.assertIn("-555-01", fake, fake)

    def test_identifiers_stay_deterministic_per_seed(self):
        a = self._mask_one("SSN", "123-45-6789", seed_key="acct-1")
        b = self._mask_one("SSN", "123-45-6789", seed_key="acct-1")
        c = self._mask_one("SSN", "123-45-6789", seed_key="acct-2")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c, "different accounts must map to different fakes")


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


class _FakeNerAnalyzer:
    """Stand-in for presidio's AnalyzerEngine — finds 'John Smith' spans.

    Lets the NER-path tests run without presidio/spaCy installed; the real
    engine is exercised in deployment, the wiring is exercised here."""

    def __init__(self, target: str = "John Smith"):
        self.target = target
        self.calls: list[str] = []

    def analyze(self, text, entities, language, score_threshold):
        assert entities == ["PERSON"]
        self.calls.append(text)
        class _Span:
            def __init__(self, start, end):
                self.start, self.end = start, end
        spans = []
        idx = text.find(self.target)
        while idx != -1:
            spans.append(_Span(idx, idx + len(self.target)))
            idx = text.find(self.target, idx + 1)
        return spans


class NerPersonNameScrubTests(unittest.TestCase):
    """Presidio-backed person-name scrubbing — the one embedded-PII category
    regex structurally cannot catch. Regulated-industry KB discovery only."""

    NARRATIVE = "Patient John Smith reported dizziness after the evening dose"

    def test_person_span_replaced(self):
        from core.masking import scrub_person_names_ner
        out = scrub_person_names_ner(self.NARRATIVE, analyzer=_FakeNerAnalyzer())
        self.assertNotIn("John Smith", out)
        self.assertIn("[PERSON]", out)
        self.assertIn("reported dizziness", out)  # rest of narrative survives

    def test_multiple_spans_replaced_without_offset_corruption(self):
        from core.masking import scrub_person_names_ner
        text = "John Smith spoke to Dr. John Smith about the refill"
        out = scrub_person_names_ner(text, analyzer=_FakeNerAnalyzer())
        self.assertNotIn("John Smith", out)
        self.assertEqual(out.count("[PERSON]"), 2)
        self.assertIn("about the refill", out)

    def test_all_caps_values_skipped(self):
        # ERP dimension values ("MARTIN SUPPLY CO") are not prose — NER is
        # unreliable there and a false positive corrupts a legitimate sample.
        from core.masking import scrub_person_names_ner
        fake = _FakeNerAnalyzer(target="MARTIN")
        out = scrub_person_names_ner("MARTIN SUPPLY CO WAREHOUSE 822", analyzer=fake)
        self.assertEqual(out, "MARTIN SUPPLY CO WAREHOUSE 822")
        self.assertEqual(fake.calls, [], "analyzer must not even be invoked for ALL-CAPS values")

    def test_unavailable_presidio_returns_text_unchanged(self):
        from unittest.mock import patch
        from core.masking import scrub_person_names_ner
        with patch("core.masking._get_presidio", return_value=None):
            self.assertEqual(scrub_person_names_ner(self.NARRATIVE), self.NARRATIVE)

    def test_analyzer_failure_returns_text_unchanged(self):
        from core.masking import scrub_person_names_ner
        class _Broken:
            def analyze(self, **kw):
                raise RuntimeError("model not loaded")
        self.assertEqual(
            scrub_person_names_ner(self.NARRATIVE, analyzer=_Broken()),
            self.NARRATIVE,
        )

    def test_regulated_industry_gets_ner_pass_in_free_text_scrub(self):
        from unittest.mock import patch
        fake = _FakeNerAnalyzer()
        cols = [{"name": "remark_field", "type": "varchar(80)"}]
        rows = [{"remark_field": self.NARRATIVE}]
        with patch("core.masking._get_presidio", return_value=fake):
            out = scrub_unmasked_free_text(
                rows, cols, skip_fields=set(), industry="healthcare_pharmacy"
            )
        self.assertNotIn("John Smith", out[0]["remark_field"])
        self.assertIn("[PERSON]", out[0]["remark_field"])

    def test_standard_client_never_invokes_ner(self):
        from unittest.mock import patch
        fake = _FakeNerAnalyzer()
        cols = [{"name": "remark_field", "type": "varchar(80)"}]
        for industry in ("", "standard"):
            rows = [{"remark_field": self.NARRATIVE}]
            with patch("core.masking._get_presidio", return_value=fake):
                out = scrub_unmasked_free_text(
                    rows, cols, skip_fields=set(), industry=industry
                )
            self.assertIn("John Smith", out[0]["remark_field"], industry)
        self.assertEqual(fake.calls, [])

    def test_ner_runs_after_regex_scrub_and_masked_columns_skipped(self):
        from unittest.mock import patch
        fake = _FakeNerAnalyzer()
        cols = [
            {"name": "remark_field", "type": "varchar(120)"},
            {"name": "notes", "type": "varchar(600)"},
        ]
        rows = [{
            "remark_field": "John Smith called from j.smith@mail.com about the refill",
            "notes": "already masked upstream — must be skipped",
        }]
        with patch("core.masking._get_presidio", return_value=fake):
            out = scrub_unmasked_free_text(
                rows, cols, skip_fields={"notes"}, industry="healthcare_pharmacy"
            )
        val = out[0]["remark_field"]
        self.assertIn("[EMAIL]", val)          # regex scrub still applied
        self.assertIn("[PERSON]", val)         # NER applied on top
        self.assertNotIn("John Smith", val)
        self.assertEqual(out[0]["notes"], "already masked upstream — must be skipped")


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
