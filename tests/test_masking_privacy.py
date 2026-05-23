import unittest

from core.masking import detect_sensitive_columns, mask_rows
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


if __name__ == "__main__":
    unittest.main()
