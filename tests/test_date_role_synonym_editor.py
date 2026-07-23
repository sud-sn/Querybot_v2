from pathlib import Path
import unittest

from core.contextual_dates import resolve_contextual_date_binding


class DateRoleSynonymEditorTests(unittest.TestCase):
    def test_date_role_editors_expose_synonyms(self):
        template = (
            Path(__file__).resolve().parents[1]
            / "admin"
            / "templates"
            / "client_date_roles.html"
        ).read_text(encoding="utf-8")

        self.assertNotIn('type="hidden" name="synonyms"', template)
        self.assertEqual(3, template.count('name="synonyms"'))
        self.assertIn("Comma-separated phrases users may use for this date role.", template)

    def test_approved_synonym_selects_role_over_default(self):
        metric = {
            "name": "Revenue",
            "formula": "SUM(SALES.FACT_REVENUE.REVENUE_AMOUNT)",
            "formula_tables": ["SALES.FACT_REVENUE"],
        }
        bindings = [
            {
                "metric_name": "Revenue",
                "fact_table": "SALES.FACT_REVENUE",
                "fact_column": "BOOKED_DATE_KEY",
                "business_role": "booked_date",
                "is_default": 1,
            }
        ]
        roles = [
            {
                "name": "Invoice Date",
                "business_role": "invoice_date",
                "synonyms": ["invoice date", "invoice month", "inv dt"],
                "fact_table": "SALES.FACT_REVENUE",
                "fact_column": "INVOICE_DATE_KEY",
                "dimension_table": "SALES.DIM_DATE",
                "dimension_key": "DATE_KEY",
                "date_value_column": "FULL_DATE",
                "date_key_type": "surrogate_fk",
                "status": "approved",
            }
        ]

        result = resolve_contextual_date_binding(
            "show revenue by inv dt",
            matched_metrics=[metric],
            bindings=bindings,
            date_roles=roles,
            required_fact_tables={"SALES.FACT_REVENUE"},
        )

        self.assertEqual("selected", result["status"])
        self.assertEqual("INVOICE_DATE_KEY", result["binding"]["fact_column"])
        self.assertEqual("explicit_date_role", result["binding"]["resolution_source"])


if __name__ == "__main__":
    unittest.main()
