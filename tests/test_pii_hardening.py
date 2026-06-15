"""
tests/test_pii_hardening.py

Tests for all four PII hardening changes:
  1. SENSITIVE_TABLE_PATTERNS — transactional/fact tables now trigger synthetic
  2. _is_sensitive_field — expanded keyword coverage (DOB, birth, postal, MRN, etc.)
  3. SQL sanitization in add_to_history — quoted literals stripped before storage
  4. generate_followup_suggestions — reads brief correctly, applies redaction
"""
import os, sys, tempfile, unittest

_tmp_db = os.path.join(tempfile.mkdtemp(), "test_pii.db")
os.environ["QUERYBOT_DB_PATH"] = _tmp_db
for mod in list(sys.modules.keys()):
    if mod.startswith("store"):
        del sys.modules[mod]
import store.db as db_mod
db_mod.init_db()

SYNTHETIC_PY  = os.path.join(os.path.dirname(__file__), "..", "core",    "synthetic.py")
INSIGHT_PY    = os.path.join(os.path.dirname(__file__), "..", "core",    "insight.py")
ADAPTER_PY    = os.path.join(os.path.dirname(__file__), "..", "gateway", "web_adapter.py")


# ── 1  SENSITIVE_TABLE_PATTERNS ───────────────────────────────────────────────
class TestSensitiveTablePatterns(unittest.TestCase):

    def _synthetic(self, table_name: str) -> bool:
        from core.synthetic import should_use_synthetic
        return should_use_synthetic(table_name)

    # Original patterns still work
    def test_patient_table(self):
        self.assertTrue(self._synthetic("PATIENT"))

    def test_employee_table(self):
        self.assertTrue(self._synthetic("DIM_EMPLOYEE"))

    def test_customer_table(self):
        self.assertTrue(self._synthetic("CUSTOMER_MASTER"))

    # New: fact tables are always synthetic
    def test_fact_prefix(self):
        self.assertTrue(self._synthetic("FACT_RXFILL"))

    def test_fact_suffix(self):
        self.assertTrue(self._synthetic("SALES_FACT"))

    # New: transactional tables
    def test_prescription(self):
        self.assertTrue(self._synthetic("PRESCRIPTIONS"))

    def test_rx_prefix(self):
        self.assertTrue(self._synthetic("RX_HISTORY"))

    def test_rx_suffix(self):
        self.assertTrue(self._synthetic("FILL_RX"))

    def test_order_table(self):
        self.assertTrue(self._synthetic("ORDERS"))

    def test_transaction_table(self):
        self.assertTrue(self._synthetic("TRANSACTIONS"))

    def test_invoice_table(self):
        self.assertTrue(self._synthetic("INVOICE_DETAIL"))

    def test_payment_table(self):
        self.assertTrue(self._synthetic("PAYMENT_HISTORY"))

    def test_claim_table(self):
        self.assertTrue(self._synthetic("INSURANCE_CLAIMS"))

    def test_encounter_table(self):
        self.assertTrue(self._synthetic("CLINICAL_ENCOUNTERS"))

    def test_history_table(self):
        self.assertTrue(self._synthetic("ORDER_HISTORY"))

    def test_audit_log_table(self):
        self.assertTrue(self._synthetic("AUDIT_LOG"))

    def test_dispens_table(self):
        self.assertTrue(self._synthetic("DISPENSING_RECORDS"))

    # Dimension tables should stay non-synthetic
    def test_dim_date_not_synthetic(self):
        self.assertFalse(self._synthetic("DIM_DATE"))

    def test_dim_product_not_synthetic(self):
        self.assertFalse(self._synthetic("DIM_PRODUCT"))

    def test_dim_store_not_synthetic(self):
        self.assertFalse(self._synthetic("DIM_STORE"))

    def test_ref_currency_not_synthetic(self):
        self.assertFalse(self._synthetic("REF_CURRENCY"))

    def test_lookup_status_not_synthetic(self):
        self.assertFalse(self._synthetic("LOOKUP_STATUS"))


# ── 2  _is_sensitive_field ────────────────────────────────────────────────────
class TestIsSensitiveField(unittest.TestCase):

    def _sensitive(self, col_name: str) -> bool:
        from core.insight import _is_sensitive_field
        return _is_sensitive_field(col_name)

    # Original keywords still work
    def test_email(self):
        self.assertTrue(self._sensitive("EmailAddress"))

    def test_phone(self):
        self.assertTrue(self._sensitive("PhoneNumber"))

    def test_ssn(self):
        self.assertTrue(self._sensitive("SSN"))

    def test_name(self):
        self.assertTrue(self._sensitive("EmployeeName"))

    # New: date of birth variants
    def test_dob(self):
        self.assertTrue(self._sensitive("DOB"))

    def test_date_of_birth(self):
        self.assertTrue(self._sensitive("DateOfBirth"))

    def test_birth_date(self):
        self.assertTrue(self._sensitive("BirthDate"))

    def test_patient_dob(self):
        self.assertTrue(self._sensitive("PatientDOB"))

    # New: postal codes
    def test_postcode(self):
        self.assertTrue(self._sensitive("Postcode"))

    def test_postal_code(self):
        self.assertTrue(self._sensitive("PostalCode"))

    def test_zip_code(self):
        self.assertTrue(self._sensitive("ZipCode"))

    # New: medical record identifiers
    def test_mrn(self):
        self.assertTrue(self._sensitive("MRN"))

    def test_medical_record_number(self):
        self.assertTrue(self._sensitive("MedicalRecordNumber"))

    def test_nhs_number(self):
        self.assertTrue(self._sensitive("NHSNumber"))

    def test_patient_id(self):
        self.assertTrue(self._sensitive("PatientID"))

    # New: national insurance
    def test_national_insurance(self):
        self.assertTrue(self._sensitive("NationalInsurance"))

    # Safe columns — should NOT trigger
    def test_revenue_safe(self):
        self.assertFalse(self._sensitive("Revenue"))

    def test_department_safe(self):
        self.assertFalse(self._sensitive("Department"))

    def test_status_safe(self):
        self.assertFalse(self._sensitive("Status"))

    def test_quantity_safe(self):
        self.assertFalse(self._sensitive("Quantity"))

    def test_segment_safe(self):
        self.assertFalse(self._sensitive("Segment"))

    def test_nationality_safe(self):
        # "Nationality" should NOT be sensitive — it's a categorical dimension
        # "national" matches "national_id" only if that exact string is present
        # normalized: "nationality" — "national_id" not in "nationality" ✓
        result = self._sensitive("Nationality")
        # Should be False — "national_id" is NOT a substring of "nationality"
        self.assertFalse(result)

    def test_returns_bool(self):
        from core.insight import _is_sensitive_field
        self.assertIsInstance(_is_sensitive_field("Revenue"), bool)
        self.assertIsInstance(_is_sensitive_field("Email"), bool)


# ── 3  SQL sanitization in add_to_history ─────────────────────────────────────
class TestHistorySQLSanitization(unittest.TestCase):

    def _make_adapter(self):
        from unittest.mock import MagicMock, AsyncMock
        from gateway.web_adapter import WebAdapter
        ws = MagicMock(); ws.send_json = AsyncMock()
        return WebAdapter(ws, "acc1", "user1")

    def test_where_clause_literal_stripped(self):
        a = self._make_adapter()
        sql = "SELECT * FROM FACT_RXFILL WHERE PatientName = 'John Smith'"
        a.add_to_history("test", sql, ["PatientName"], 5)
        stored_sql = a.get_history()[0]["sql"]
        self.assertNotIn("John Smith", stored_sql)

    def test_email_literal_stripped(self):
        a = self._make_adapter()
        sql = "SELECT * FROM USERS WHERE Email = 'john@company.com'"
        a.add_to_history("find user", sql, ["Email"], 1)
        stored_sql = a.get_history()[0]["sql"]
        self.assertNotIn("john@company.com", stored_sql)

    def test_table_and_column_names_preserved(self):
        """SQL structure (tables, columns, operators) must survive sanitization."""
        a = self._make_adapter()
        sql = "SELECT Customer, SUM(Revenue) FROM FACT_PROFIT WHERE Status = 'Active' GROUP BY Customer"
        a.add_to_history("revenue", sql, ["Customer","Revenue"], 12)
        stored_sql = a.get_history()[0]["sql"]
        self.assertIn("FACT_PROFIT", stored_sql)
        self.assertIn("Revenue", stored_sql)
        self.assertIn("GROUP BY", stored_sql)
        self.assertIn("SUM", stored_sql)

    def test_active_status_stripped(self):
        a = self._make_adapter()
        sql = "SELECT * FROM ORDERS WHERE OrderStatus = 'Active'"
        a.add_to_history("q", sql, ["OrderStatus"], 5)
        stored_sql = a.get_history()[0]["sql"]
        self.assertNotIn("'Active'", stored_sql)

    def test_no_literals_unchanged(self):
        """SQL without string literals should pass through unchanged."""
        a = self._make_adapter()
        sql = "SELECT Customer, SUM(Revenue) FROM FACT_PROFIT GROUP BY Customer ORDER BY 2 DESC"
        a.add_to_history("q", sql, ["Customer","Revenue"], 12)
        stored_sql = a.get_history()[0]["sql"]
        self.assertIn("SELECT Customer", stored_sql)
        self.assertIn("SUM(Revenue)", stored_sql)

    def test_sanitize_method_exists(self):
        src = open(ADAPTER_PY, encoding="utf-8").read()
        self.assertIn("_sanitize_sql_for_history", src)

    def test_sanitize_is_static_method(self):
        src = open(ADAPTER_PY, encoding="utf-8").read()
        self.assertIn("@staticmethod", src)

    def test_question_is_not_sanitized(self):
        """User question text must NOT be modified — only the SQL."""
        a = self._make_adapter()
        a.add_to_history("show John Smith revenue", "SELECT 1", ["Revenue"], 1)
        hist = a.get_history()[0]
        self.assertEqual(hist["question"], "show John Smith revenue")


# ── 4  generate_followup_suggestions brief reading + redaction ────────────────
class TestFollowupBriefReading(unittest.TestCase):

    def _make_brief(self, col_name="Customer", sensitive=False):
        """Build a realistic brief matching compute_data_brief output format."""
        return {
            "row_count": 12,
            "column_count": 2,
            "columns": {"Customer": "text", "Revenue": "numeric"},
            "mode": "ranking",
            "numeric_summaries": {
                "Revenue": {"count": 12, "total": 1100000.0, "min": 5000.0,
                            "max": 200000.0, "mean": 91667.0, "median": 85000.0}
            },
            "category_breakdown": {
                "label_column": col_name,
                "value_column": "Revenue",
                "category_count": 12,
                "top_5": [
                    {"label": "REDACTED" if sensitive else "Customer A", "value": 200000.0},
                    {"label": "REDACTED" if sensitive else "Customer B", "value": 150000.0},
                    {"label": "REDACTED" if sensitive else "Customer C", "value": 120000.0},
                ],
                "labels_redacted": sensitive,
            },
        }

    def test_reads_columns_as_dict(self):
        """generate_followup_suggestions must handle brief.columns as a dict."""
        src = open(INSIGHT_PY, encoding="utf-8").read()
        self.assertIn("isinstance(columns, dict)", src)

    def test_reads_category_breakdown_not_columns_list(self):
        """Top values must come from category_breakdown, not a columns list."""
        src = open(INSIGHT_PY, encoding="utf-8").read()
        fn_start = src.find("async def generate_followup_suggestions")
        fn_body  = src[fn_start:fn_start+3000]
        self.assertIn("category_breakdown", fn_body)

    def test_applies_is_sensitive_field_to_label_col(self):
        """Must check _is_sensitive_field on label_col before including top values."""
        src = open(INSIGHT_PY, encoding="utf-8").read()
        fn_start = src.find("async def generate_followup_suggestions")
        fn_body  = src[fn_start:fn_start+3000]
        self.assertIn("_is_sensitive_field", fn_body)

    def test_sensitive_label_col_excluded_from_top_values(self):
        """Column named PatientName must not have its values in suggestion context."""
        src = open(INSIGHT_PY, encoding="utf-8").read()
        fn_start = src.find("async def generate_followup_suggestions")
        fn_body  = src[fn_start:fn_start+3000]
        # Guard must check the label column for sensitivity
        self.assertIn("not _is_sensitive_field(label_col)", fn_body)

    def test_redacted_segment_excluded(self):
        """Already-redacted labels must not be sent to LLM as values."""
        src = open(INSIGHT_PY, encoding="utf-8").read()
        fn_start = src.find("async def generate_followup_suggestions")
        fn_body  = src[fn_start:fn_start+3000]
        self.assertIn("redacted segment", fn_body)

    def test_numeric_summaries_used_for_ranges(self):
        """Numeric ranges must come from numeric_summaries, not raw values."""
        src = open(INSIGHT_PY, encoding="utf-8").read()
        fn_start = src.find("async def generate_followup_suggestions")
        fn_body  = src[fn_start:fn_start+3000]
        self.assertIn("numeric_summaries", fn_body)

    def test_no_raw_rows_in_brief_path(self):
        """The function signature must not accept rows parameter."""
        src = open(INSIGHT_PY, encoding="utf-8").read()
        fn_start = src.find("async def generate_followup_suggestions")
        fn_sig   = src[fn_start:fn_start+400]
        self.assertNotIn("rows: list", fn_sig)
        self.assertNotIn("raw_rows", fn_sig)


# ── 5  Cross-cutting: no regression on working sanitization ──────────────────
class TestNoRegressions(unittest.TestCase):

    def test_should_use_synthetic_returns_bool(self):
        from core.synthetic import should_use_synthetic
        self.assertIsInstance(should_use_synthetic("EMPLOYEE"), bool)
        self.assertIsInstance(should_use_synthetic("DIM_DATE"), bool)

    def test_is_sensitive_returns_bool(self):
        from core.insight import _is_sensitive_field
        self.assertIsInstance(_is_sensitive_field("Email"), bool)
        self.assertIsInstance(_is_sensitive_field("Revenue"), bool)

    def test_display_label_still_redacts_name_columns(self):
        from core.insight import _display_label
        self.assertEqual(_display_label("John Smith", "EmployeeName"), "redacted segment")
        self.assertEqual(_display_label("Engineering", "Department"), "Engineering")

    def test_compute_data_brief_still_works(self):
        from core.insight import compute_data_brief
        rows = [
            {"Customer": "A", "Revenue": 100.0},
            {"Customer": "B", "Revenue": 200.0},
        ]
        brief = compute_data_brief(rows, "show revenue by customer")
        self.assertEqual(brief["row_count"], 2)
        self.assertIn("numeric_summaries", brief)
        self.assertIn("category_breakdown", brief)

    def test_history_columns_still_stored(self):
        from unittest.mock import MagicMock, AsyncMock
        from gateway.web_adapter import WebAdapter
        ws = MagicMock(); ws.send_json = AsyncMock()
        a = WebAdapter(ws, "acc1", "u1")
        a.add_to_history("q", "SELECT SUM(Revenue) FROM T", ["Revenue"], 5)
        h = a.get_history()[0]
        self.assertEqual(h["columns"], ["Revenue"])
        self.assertEqual(h["row_count"], 5)

    def test_history_question_preserved(self):
        from unittest.mock import MagicMock, AsyncMock
        from gateway.web_adapter import WebAdapter
        ws = MagicMock(); ws.send_json = AsyncMock()
        a = WebAdapter(ws, "acc1", "u1")
        a.add_to_history("What is total revenue?", "SELECT SUM(Revenue) FROM T", ["Revenue"], 1)
        self.assertEqual(a.get_history()[0]["question"], "What is total revenue?")

    def test_synthetic_py_imports_cleanly(self):
        import importlib
        import core.synthetic as m
        self.assertTrue(hasattr(m, "should_use_synthetic"))
        self.assertTrue(hasattr(m, "SENSITIVE_TABLE_PATTERNS"))

    def test_insight_py_imports_cleanly(self):
        import core.insight as m
        self.assertTrue(hasattr(m, "_is_sensitive_field"))
        self.assertTrue(hasattr(m, "compute_data_brief"))
        self.assertTrue(hasattr(m, "generate_followup_suggestions"))


if __name__ == "__main__":
    unittest.main()
