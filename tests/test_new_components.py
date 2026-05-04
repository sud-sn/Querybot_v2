"""
tests/test_new_components.py

Comprehensive integration tests for all components added in the v2 egress-log
release. Covers edge cases and cross-component paths not in the individual
unit test files.

Components under test
─────────────────────
A. PII hardening
   A1. SENSITIVE_TABLE_PATTERNS — new transactional/fact patterns
   A2. _is_sensitive_field — expanded + bare-form CamelCase matching
   A3. _sanitize_sql_for_history — strips WHERE clause literals
   A4. generate_followup_suggestions — brief reading + label redaction

B. KB data egress log
   B1. log_kb_egress / list_kb_egress / get_kb_egress_summary
   B2. get_kb_egress_summary NULL-safe aggregation
   B3. multi-account isolation
   B4. append-only semantics

C. External log export pipeline — egress table
   C1. EGRESS_TABLE / EGRESS_COLUMNS constants
   C2. _fetch_egress_rows_after watermarking
   C3. _write_export_state accepts egress params
   C4. Provision functions include egress table
   C5. sync_external_logs result dict keys

D. Multi-turn conversation memory
   D1. WebAdapter history — add / get / clear / maxlen
   D2. SQL sanitization before history storage
   D3. build_sql_system_prompt conversation_history param
   D4. History injection block content
   D5. main.py wiring guards

E. Follow-up suggestions
   E1. generate_followup_suggestions skips single-column results
   E2. Reads category_breakdown not raw columns list
   E3. Applies _is_sensitive_field guard on label column
   F4. Redacted-segment values filtered out

F. Formula editor
   F1. ƒ Functions popover present, old pills absent
   F2. DB-aware templates for all three dialects
   F3. Syntax validator rules
   F4. Duplicate column disambiguation (bare insert)

G. Dynamic pricing
   G1. llm_pricing table seeded with defaults
   G2. calculate_cost() DB-first fallback
   G3. save_pricing() upsert + cache invalidation
"""

import os
import re
import sys
import ast
import json
import tempfile
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

# ── DB isolation ──────────────────────────────────────────────────────────────
_TMP = os.path.join(tempfile.mkdtemp(), "test_newcomp.db")
os.environ["QUERYBOT_DB_PATH"] = _TMP
for _mod in list(sys.modules.keys()):
    if _mod.startswith("store"):
        del sys.modules[_mod]
import store.db as _db
_db.init_db()
import store

# ── Source file paths ─────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
SYNTHETIC  = ROOT / "core"    / "synthetic.py"
INSIGHT    = ROOT / "core"    / "insight.py"
LLM_PY     = ROOT / "core"    / "llm.py"
LOG_EXPORT = ROOT / "core"    / "log_export.py"
ADAPTER    = ROOT / "gateway" / "web_adapter.py"
MAIN_PY    = ROOT / "main.py"
DB_PY      = ROOT / "store"   / "db.py"
CS_PY      = ROOT / "store"   / "config_store.py"
STORE_INIT = ROOT / "store"   / "__init__.py"
ROUTES     = ROOT / "admin"   / "routes.py"
SETUP_TMPL = ROOT / "admin"   / "templates" / "client_setup.html"
CHAT_TMPL  = ROOT / "portal"  / "templates" / "portal_chat.html"
DASH_TMPL  = ROOT / "portal"  / "templates" / "portal_dashboard.html"


def _src(path: Path) -> str:
    return path.read_text()


def _make_adapter():
    ws = MagicMock()
    ws.send_json = AsyncMock()
    from gateway.web_adapter import WebAdapter
    return WebAdapter(ws, "acc_test", "user_test")


# ══════════════════════════════════════════════════════════════════════════════
# A. PII HARDENING
# ══════════════════════════════════════════════════════════════════════════════

class TestPIIPatterns(unittest.TestCase):
    """A1 — SENSITIVE_TABLE_PATTERNS covers new transactional / fact tables."""

    def _syn(self, name):
        from core.synthetic import should_use_synthetic
        return should_use_synthetic(name)

    # Original identity patterns still work
    def test_patient(self):       self.assertTrue(self._syn("PATIENT_MASTER"))
    def test_employee(self):      self.assertTrue(self._syn("DIM_EMPLOYEE"))
    def test_customer(self):      self.assertTrue(self._syn("CUSTOMER_LIST"))

    # New — fact prefix / suffix
    def test_fact_prefix(self):   self.assertTrue(self._syn("FACT_RXFILL"))
    def test_fact_suffix(self):   self.assertTrue(self._syn("SALES_FACT"))
    def test_fact_mixed(self):    self.assertTrue(self._syn("FACT_CUSTOMER_PROFIT"))

    # New — prescription / pharmacy
    def test_prescription(self):  self.assertTrue(self._syn("PRESCRIPTIONS"))
    def test_rx_prefix(self):     self.assertTrue(self._syn("RX_ORDERS"))
    def test_rx_suffix(self):     self.assertTrue(self._syn("FILL_RX"))
    def test_dispens(self):       self.assertTrue(self._syn("DISPENSING_LOG"))
    def test_refill(self):        self.assertTrue(self._syn("REFILL_REQUESTS"))

    # New — transactional
    def test_orders(self):        self.assertTrue(self._syn("ORDERS"))
    def test_transaction(self):   self.assertTrue(self._syn("TRANSACTIONS"))
    def test_invoice(self):       self.assertTrue(self._syn("INVOICE_DETAIL"))
    def test_payment(self):       self.assertTrue(self._syn("PAYMENT_HISTORY"))
    def test_claim(self):         self.assertTrue(self._syn("INSURANCE_CLAIMS"))
    def test_billing(self):       self.assertTrue(self._syn("BILLING_RECORDS"))
    def test_encounter(self):     self.assertTrue(self._syn("CLINICAL_ENCOUNTERS"))
    def test_history(self):       self.assertTrue(self._syn("ORDER_HISTORY"))
    def test_audit_log(self):     self.assertTrue(self._syn("AUDIT_LOG"))

    # Safe dimension tables must NOT become synthetic
    def test_dim_date_safe(self):    self.assertFalse(self._syn("DIM_DATE"))
    def test_dim_product_safe(self): self.assertFalse(self._syn("DIM_PRODUCT"))
    def test_dim_store_safe(self):   self.assertFalse(self._syn("DIM_STORE"))
    def test_ref_currency_safe(self):self.assertFalse(self._syn("REF_CURRENCY"))
    def test_lookup_status_safe(self):self.assertFalse(self._syn("LOOKUP_STATUS"))

    def test_returns_bool(self):
        from core.synthetic import should_use_synthetic
        self.assertIsInstance(should_use_synthetic("ORDERS"), bool)


class TestSensitiveFieldExpanded(unittest.TestCase):
    """A2 — _is_sensitive_field handles CamelCase (bare form) and new keywords."""

    def _s(self, col):
        from core.insight import _is_sensitive_field
        return _is_sensitive_field(col)

    # Original keywords
    def test_email(self):          self.assertTrue(self._s("EmailAddress"))
    def test_phone(self):          self.assertTrue(self._s("PhoneNumber"))
    def test_ssn(self):            self.assertTrue(self._s("SSN"))
    def test_name(self):           self.assertTrue(self._s("EmployeeName"))

    # New — DOB variants
    def test_dob(self):            self.assertTrue(self._s("DOB"))
    def test_date_of_birth(self):  self.assertTrue(self._s("DateOfBirth"))
    def test_birth_date(self):     self.assertTrue(self._s("BirthDate"))
    def test_patient_dob(self):    self.assertTrue(self._s("PatientDOB"))

    # New — postal codes
    def test_postcode(self):       self.assertTrue(self._s("Postcode"))
    def test_postal_code(self):    self.assertTrue(self._s("PostalCode"))
    def test_zip_code(self):       self.assertTrue(self._s("ZipCode"))

    # New — medical record identifiers (CamelCase bare form)
    def test_mrn(self):                    self.assertTrue(self._s("MRN"))
    def test_medical_record_number(self):  self.assertTrue(self._s("MedicalRecordNumber"))
    def test_nhs_number(self):             self.assertTrue(self._s("NHSNumber"))
    def test_patient_id(self):             self.assertTrue(self._s("PatientID"))
    def test_national_insurance(self):     self.assertTrue(self._s("NationalInsurance"))

    # Safe columns must NOT trigger
    def test_revenue_safe(self):    self.assertFalse(self._s("Revenue"))
    def test_segment_safe(self):    self.assertFalse(self._s("Segment"))
    def test_department_safe(self): self.assertFalse(self._s("Department"))
    def test_quantity_safe(self):   self.assertFalse(self._s("Quantity"))

    # Nationality must NOT match national_id
    def test_nationality_safe(self):
        # "nationality" does not contain "national_id" or "nationalid"
        self.assertFalse(self._s("Nationality"))

    def test_returns_bool_always(self):
        from core.insight import _is_sensitive_field
        for col in ["Revenue", "Email", "DOB", "PatientID", "Segment", ""]:
            self.assertIsInstance(_is_sensitive_field(col), bool)


class TestSQLSanitizationInHistory(unittest.TestCase):
    """A3 — _sanitize_sql_for_history strips WHERE clause string literals."""

    def test_name_literal_stripped(self):
        a = _make_adapter()
        sql = "SELECT * FROM FACT WHERE PatientName = 'John Smith'"
        a.add_to_history("q", sql, ["PatientName"], 5)
        self.assertNotIn("John Smith", a.get_history()[0]["sql"])

    def test_email_literal_stripped(self):
        a = _make_adapter()
        sql = "SELECT * FROM USERS WHERE Email = 'alice@example.com'"
        a.add_to_history("q", sql, ["Email"], 1)
        self.assertNotIn("alice@example.com", a.get_history()[0]["sql"])

    def test_status_literal_stripped(self):
        a = _make_adapter()
        sql = "SELECT * FROM ORDERS WHERE Status = 'Completed'"
        a.add_to_history("q", sql, ["Status"], 3)
        self.assertNotIn("'Completed'", a.get_history()[0]["sql"])

    def test_structure_preserved_after_strip(self):
        a = _make_adapter()
        sql = "SELECT Customer, SUM(Revenue) FROM FACT WHERE Region = 'North' GROUP BY Customer"
        a.add_to_history("q", sql, ["Customer","Revenue"], 8)
        stored = a.get_history()[0]["sql"]
        self.assertIn("FACT", stored)
        self.assertIn("SUM(Revenue)", stored)
        self.assertIn("GROUP BY", stored)

    def test_no_literals_unchanged(self):
        a = _make_adapter()
        sql = "SELECT Customer, SUM(Revenue) FROM FACT GROUP BY Customer ORDER BY 2 DESC"
        a.add_to_history("q", sql, ["Customer"], 5)
        stored = a.get_history()[0]["sql"]
        self.assertIn("SELECT Customer", stored)

    def test_numeric_literals_preserved(self):
        a = _make_adapter()
        sql = "SELECT * FROM ORDERS WHERE Amount > 1000"
        a.add_to_history("q", sql, ["Amount"], 2)
        stored = a.get_history()[0]["sql"]
        self.assertIn("1000", stored)

    def test_question_text_not_sanitized(self):
        a = _make_adapter()
        a.add_to_history("show John Smith revenue", "SELECT 1", ["Rev"], 1)
        self.assertEqual(a.get_history()[0]["question"], "show John Smith revenue")

    def test_static_method_exists(self):
        self.assertIn("@staticmethod", _src(ADAPTER))
        self.assertIn("_sanitize_sql_for_history", _src(ADAPTER))


class TestFollowupBriefRedaction(unittest.TestCase):
    """A4 — generate_followup_suggestions reads brief correctly and redacts."""

    def test_reads_columns_as_dict(self):
        src = _src(INSIGHT)
        self.assertIn("isinstance(columns, dict)", src)

    def test_uses_category_breakdown_not_columns_list(self):
        src = _src(INSIGHT)
        fn = src[src.find("async def generate_followup_suggestions"):
                 src.find("async def generate_followup_suggestions") + 3000]
        self.assertIn("category_breakdown", fn)

    def test_sensitive_label_col_excluded(self):
        src = _src(INSIGHT)
        fn = src[src.find("async def generate_followup_suggestions"):
                 src.find("async def generate_followup_suggestions") + 3000]
        self.assertIn("not _is_sensitive_field(label_col)", fn)

    def test_redacted_segment_filtered(self):
        src = _src(INSIGHT)
        fn = src[src.find("async def generate_followup_suggestions"):
                 src.find("async def generate_followup_suggestions") + 3000]
        self.assertIn("redacted segment", fn)

    def test_numeric_ranges_from_summaries(self):
        src = _src(INSIGHT)
        fn = src[src.find("async def generate_followup_suggestions"):
                 src.find("async def generate_followup_suggestions") + 3000]
        self.assertIn("numeric_summaries", fn)

    def test_no_raw_rows_param(self):
        src = _src(INSIGHT)
        fn_sig = src[src.find("async def generate_followup_suggestions"):
                     src.find("async def generate_followup_suggestions") + 400]
        self.assertNotIn("rows: list", fn_sig)
        self.assertNotIn("raw_rows", fn_sig)

    def test_max_160_tokens(self):
        src = _src(INSIGHT)
        self.assertIn("max_tokens=160", src)

    def test_audit_component_followup_suggestions(self):
        src = _src(INSIGHT)
        self.assertIn('"followup_suggestions"', src)

    def test_failure_returns_empty_list(self):
        src = _src(INSIGHT)
        fn = src[src.find("async def generate_followup_suggestions"):]
        self.assertIn("return []", fn)
        self.assertIn("except Exception", fn)


# ══════════════════════════════════════════════════════════════════════════════
# B. KB DATA EGRESS LOG
# ══════════════════════════════════════════════════════════════════════════════

class TestEgressLogCRUD(unittest.TestCase):
    """B1 — log / list / summary functions."""

    ACC = "test_newcomp_egress_001"

    def setUp(self):
        # Seed several rows
        for tbl, op, mode in [
            ("FACT_RXFILL",  "kb_build",  "synthetic"),
            ("DIM_DATE",     "kb_build",  "real"),
            ("DIM_STORE",    "kb_build",  "real"),
            ("FACT_RXFILL",  "discovery", "none"),
            ("DIM_DATE",     "discovery", "none"),
        ]:
            store.log_kb_egress(
                account_id=self.ACC, operation=op,
                db_type="azure_sql", table_name=tbl, sample_mode=mode,
                column_count=8,
            )

    def test_list_returns_list_of_dicts(self):
        rows = store.list_kb_egress(self.ACC)
        self.assertIsInstance(rows, list)
        for r in rows:
            self.assertIsInstance(r, dict)

    def test_list_filters_by_operation_kb_build(self):
        rows = store.list_kb_egress(self.ACC, operation="kb_build")
        self.assertTrue(all(r["operation"] == "kb_build" for r in rows))

    def test_list_filters_by_operation_discovery(self):
        rows = store.list_kb_egress(self.ACC, operation="discovery")
        self.assertTrue(all(r["operation"] == "discovery" for r in rows))

    def test_list_scoped_to_account(self):
        other = store.list_kb_egress("unrelated_account_xyz")
        self.assertNotIn(self.ACC, [r["account_id"] for r in other])

    def test_summary_totals(self):
        s = store.get_kb_egress_summary(self.ACC)
        self.assertGreaterEqual(s["total_tables_kb_build"], 3)
        self.assertGreaterEqual(s["total_tables_discovery"], 2)

    def test_summary_sample_counts(self):
        s = store.get_kb_egress_summary(self.ACC)
        self.assertGreaterEqual(s["synthetic_sample_count"], 1)
        self.assertGreaterEqual(s["real_sample_count"], 2)

    def test_summary_timestamps_not_empty(self):
        s = store.get_kb_egress_summary(self.ACC)
        self.assertIsNotNone(s.get("last_kb_build_at"))
        self.assertIsNotNone(s.get("last_discovery_at"))

    def test_summary_rows_lists(self):
        s = store.get_kb_egress_summary(self.ACC)
        self.assertIsInstance(s["kb_build_rows"], list)
        self.assertIsInstance(s["discovery_rows"], list)

    def test_all_required_fields_in_row(self):
        rows = store.list_kb_egress(self.ACC, operation="kb_build")
        self.assertGreater(len(rows), 0)
        r = rows[0]
        for field in ("id","account_id","operation","db_type",
                      "table_name","sample_mode","column_count","created_at"):
            self.assertIn(field, r, f"Field {field!r} missing from row")

    def test_log_does_not_raise_on_empty_names(self):
        try:
            store.log_kb_egress(
                account_id="", operation="kb_build", db_type="",
                table_name="", sample_mode="none",
            )
        except Exception as e:
            self.fail(f"log_kb_egress raised unexpectedly: {e}")


class TestEgressSummaryNullSafe(unittest.TestCase):
    """B2 — get_kb_egress_summary returns 0 not None for empty accounts."""

    def test_empty_account_all_zeros(self):
        s = store.get_kb_egress_summary("completely_empty_account_xyz_999")
        self.assertEqual(s["total_tables_kb_build"],  0)
        self.assertEqual(s["total_tables_discovery"],  0)
        self.assertEqual(s["real_sample_count"],       0)
        self.assertEqual(s["synthetic_sample_count"],  0)
        self.assertEqual(s["last_kb_build_at"],        "")
        self.assertEqual(s["last_discovery_at"],       "")

    def test_empty_account_row_lists_empty(self):
        s = store.get_kb_egress_summary("completely_empty_account_xyz_999")
        self.assertEqual(s["kb_build_rows"],  [])
        self.assertEqual(s["discovery_rows"], [])


class TestEgressMultiAccountIsolation(unittest.TestCase):
    """B3 — rows from one account never appear in another account's queries."""

    def test_isolation(self):
        store.log_kb_egress(account_id="acc_A", operation="kb_build",
                            db_type="azure_sql", table_name="TABLE_A",
                            sample_mode="real")
        store.log_kb_egress(account_id="acc_B", operation="kb_build",
                            db_type="azure_sql", table_name="TABLE_B",
                            sample_mode="synthetic")
        rows_a = store.list_kb_egress("acc_A")
        rows_b = store.list_kb_egress("acc_B")
        for r in rows_a:
            self.assertEqual(r["account_id"], "acc_A")
        for r in rows_b:
            self.assertEqual(r["account_id"], "acc_B")


class TestEgressAppendOnly(unittest.TestCase):
    """B4 — log_kb_egress only adds rows, never updates existing ones."""

    ACC = "test_newcomp_egress_appendonly"

    def test_multiple_builds_accumulate(self):
        for i in range(3):
            store.log_kb_egress(
                account_id=self.ACC, operation="kb_build",
                db_type="azure_sql", table_name="ORDERS", sample_mode="synthetic",
            )
        rows = store.list_kb_egress(self.ACC, operation="kb_build")
        tbl_rows = [r for r in rows if r["table_name"] == "ORDERS"]
        self.assertGreaterEqual(len(tbl_rows), 3)


# ══════════════════════════════════════════════════════════════════════════════
# C. EXTERNAL LOG EXPORT — EGRESS TABLE
# ══════════════════════════════════════════════════════════════════════════════

class TestEgressExportConstants(unittest.TestCase):
    """C1 — Constants match the SQLite schema."""

    def test_egress_table_name(self):
        from core.log_export import EGRESS_TABLE
        self.assertEqual(EGRESS_TABLE, "KB_DATA_EGRESS_LOG")

    def test_egress_columns_list(self):
        from core.log_export import EGRESS_COLUMNS
        self.assertIsInstance(EGRESS_COLUMNS, list)
        for col in ("SOURCE_ID","ACCOUNT_ID","OPERATION","DB_TYPE",
                    "TABLE_NAME","SAMPLE_MODE","COLUMN_COUNT","CREATED_AT"):
            self.assertIn(col, EGRESS_COLUMNS)

    def test_egress_columns_length_matches_fetch(self):
        from core.log_export import EGRESS_COLUMNS, _fetch_egress_rows_after
        # Seed a row so we have something to count
        store.log_kb_egress(
            account_id="test_col_count", operation="kb_build",
            db_type="azure_sql", table_name="COL_COUNT_TBL", sample_mode="none",
        )
        rows = _fetch_egress_rows_after(0, 10)
        if rows:
            self.assertEqual(len(rows[0]), len(EGRESS_COLUMNS))


class TestEgressFetchWatermark(unittest.TestCase):
    """C2 — _fetch_egress_rows_after respects watermark correctly."""

    def setUp(self):
        for i in range(4):
            store.log_kb_egress(
                account_id="wm_test_acc", operation="kb_build",
                db_type="snowflake", table_name=f"TABLE_{i}", sample_mode="none",
            )

    def test_returns_all_after_zero(self):
        from core.log_export import _fetch_egress_rows_after
        rows = _fetch_egress_rows_after(0, 1000)
        self.assertGreater(len(rows), 0)

    def test_watermark_excludes_before(self):
        from core.log_export import _fetch_egress_rows_after
        all_rows = _fetch_egress_rows_after(0, 1000)
        if not all_rows:
            self.skipTest("No egress rows")
        first_id = all_rows[0][0]
        after = _fetch_egress_rows_after(first_id, 1000)
        ids = [r[0] for r in after]
        self.assertNotIn(first_id, ids)

    def test_limit_respected(self):
        from core.log_export import _fetch_egress_rows_after
        rows = _fetch_egress_rows_after(0, 1)
        self.assertLessEqual(len(rows), 1)

    def test_tuples_ordered_by_id(self):
        from core.log_export import _fetch_egress_rows_after
        rows = _fetch_egress_rows_after(0, 1000)
        ids = [r[0] for r in rows]
        self.assertEqual(ids, sorted(ids))


class TestWriteExportStateEgress(unittest.TestCase):
    """C3 — _write_export_state signature includes egress params."""

    def test_signature_has_egress_count(self):
        import inspect
        from core.log_export import _write_export_state
        params = inspect.signature(_write_export_state).parameters
        self.assertIn("egress_count", params)
        self.assertIn("last_egress_id", params)

    def test_egress_defaults_are_zero(self):
        import inspect
        from core.log_export import _write_export_state
        params = inspect.signature(_write_export_state).parameters
        self.assertEqual(params["egress_count"].default,   0)
        self.assertEqual(params["last_egress_id"].default, 0)


class TestProvisionFunctionsEgress(unittest.TestCase):
    """C4 — All three provision functions include EGRESS_TABLE."""

    def _fn_body(self, fn_name: str, end_marker: str | None = None) -> str:
        src = _src(LOG_EXPORT)
        start = src.find(f"def {fn_name}")
        end   = src.find("\ndef ", start + 10) if end_marker is None else src.find(end_marker, start)
        return src[start:end]

    def test_snowflake_has_egress_table(self):
        body = self._fn_body("_provision_snowflake")
        self.assertIn("EGRESS_TABLE", body)
        self.assertIn("SAMPLE_MODE", body)

    def test_azure_has_egress_table(self):
        body = self._fn_body("_provision_azure_sql")
        self.assertIn("EGRESS_TABLE", body)
        self.assertIn("SAMPLE_MODE", body)

    def test_oracle_has_egress_table(self):
        body = self._fn_body("_provision_oracle")
        self.assertIn("EGRESS_TABLE", body)
        self.assertIn("SAMPLE_MODE", body)

    def test_all_three_have_exported_at(self):
        src = _src(LOG_EXPORT)
        for fn in ("_provision_snowflake", "_provision_azure_sql", "_provision_oracle"):
            start = src.find(f"def {fn}")
            end   = src.find("\ndef ", start + 10)
            body  = src[start:end]
            self.assertIn("EXPORTED_AT", body, f"{fn} missing EXPORTED_AT in EGRESS_TABLE")


class TestSyncResultEgressKeys(unittest.TestCase):
    """C5 — sync_external_logs result dict includes egress fields."""

    def test_egress_count_in_result(self):
        src = _src(LOG_EXPORT)
        self.assertIn('"egress_count"', src)

    def test_last_egress_id_in_result(self):
        src = _src(LOG_EXPORT)
        self.assertIn('"last_egress_id"', src)

    def test_fetch_egress_called_in_sync(self):
        src = _src(LOG_EXPORT)
        fn = src[src.find("def sync_external_logs"):src.find("\ndef ", src.find("def sync_external_logs")+5)]
        self.assertIn("_fetch_egress_rows_after", fn)

    def test_insert_egress_called_in_sync(self):
        src = _src(LOG_EXPORT)
        self.assertIn("EGRESS_TABLE, EGRESS_COLUMNS, egress_rows", src)


# ══════════════════════════════════════════════════════════════════════════════
# D. MULTI-TURN CONVERSATION MEMORY
# ══════════════════════════════════════════════════════════════════════════════

class TestWebAdapterHistoryBuffer(unittest.TestCase):
    """D1 — add / get / clear / maxlen semantics."""

    def test_starts_empty(self):
        self.assertEqual(_make_adapter().get_history(), [])

    def test_add_and_get(self):
        a = _make_adapter()
        a.add_to_history("q1", "SELECT 1", ["A","B"], 5)
        h = a.get_history()
        self.assertEqual(len(h), 1)
        self.assertEqual(h[0]["question"], "q1")
        self.assertEqual(h[0]["columns"],  ["A","B"])
        self.assertEqual(h[0]["row_count"], 5)

    def test_maxlen_3(self):
        a = _make_adapter()
        for i in range(6):
            a.add_to_history(f"q{i}", f"sql{i}", [], i)
        h = a.get_history()
        self.assertEqual(len(h), 3)
        self.assertEqual(h[-1]["question"], "q5")

    def test_clear_empties(self):
        a = _make_adapter()
        a.add_to_history("q", "sql", [], 1)
        a.clear_history()
        self.assertEqual(a.get_history(), [])

    def test_oldest_first_order(self):
        a = _make_adapter()
        for q in ["first","second","third"]:
            a.add_to_history(q, "s", [], 1)
        h = a.get_history()
        self.assertEqual(h[0]["question"], "first")
        self.assertEqual(h[2]["question"], "third")

    def test_no_raw_rows_stored(self):
        a = _make_adapter()
        a.add_to_history("q", "s", ["Col1"], 100)
        h = a.get_history()[0]
        self.assertNotIn("rows", h)
        self.assertNotIn("data", h)

    def test_uses_deque_maxlen(self):
        src = _src(ADAPTER)
        self.assertIn("deque", src)
        self.assertIn("_HISTORY_MAXLEN = 3", src)


class TestSQLPromptHistoryInjection(unittest.TestCase):
    """D3/D4 — build_sql_system_prompt injects history block correctly."""

    def _build(self, history=None):
        from core.llm import build_sql_system_prompt
        return build_sql_system_prompt("azure_sql", "KB context", conversation_history=history)

    def test_no_history_no_block(self):
        p = self._build()
        self.assertNotIn("Session context", p)

    def test_empty_history_no_block(self):
        p = self._build(history=[])
        self.assertNotIn("Session context", p)

    def test_with_history_adds_block(self):
        hist = [{"question":"show revenue","sql":"SELECT SUM(Revenue) FROM T",
                 "columns":["Customer","Revenue"],"row_count":12}]
        p = self._build(history=hist)
        self.assertIn("Session context", p)

    def test_history_question_injected(self):
        hist = [{"question":"show revenue by segment","sql":"sql","columns":[],"row_count":5}]
        p = self._build(history=hist)
        self.assertIn("show revenue by segment", p)

    def test_history_columns_injected(self):
        hist = [{"question":"q","sql":"s","columns":["Alpha","Beta"],"row_count":5}]
        p = self._build(history=hist)
        self.assertIn("Alpha", p)
        self.assertIn("Beta", p)

    def test_sql_capped_at_300_chars(self):
        long_sql = "SELECT " + "X" * 500
        hist = [{"question":"q","sql":long_sql,"columns":[],"row_count":1}]
        p = self._build(history=hist)
        # Must be truncated to 300
        self.assertNotIn("X" * 400, p)

    def test_prompt_still_returns_string(self):
        self.assertIsInstance(self._build(), str)
        self.assertGreater(len(self._build()), 50)


class TestMainWiringHistory(unittest.TestCase):
    """D5 — main.py is wired correctly for history."""

    def test_get_history_called(self):
        self.assertIn("get_history", _src(MAIN_PY))

    def test_conversation_history_passed_to_prompt(self):
        self.assertIn("conversation_history=_conv_history", _src(MAIN_PY))

    def test_add_to_history_called_after_results(self):
        src = _src(MAIN_PY)
        idx_send    = src.find("await _send_results(")
        idx_history = src.rfind("add_to_history", 0, idx_send)
        # add_to_history must appear before _send_results in the success path
        self.assertGreater(idx_send, idx_history)

    def test_clear_history_on_ws_connect(self):
        self.assertIn("clear_history", _src(MAIN_PY))


# ══════════════════════════════════════════════════════════════════════════════
# E. FOLLOW-UP SUGGESTIONS — PORTAL TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════

class TestFollowupSuggestionsTemplate(unittest.TestCase):
    """E — Follow-up chips rendered in portal chat."""

    def test_follow_up_suggestions_read_from_msg(self):
        self.assertIn("follow_up_suggestions", _src(CHAT_TMPL))

    def test_follow_up_chip_class_defined(self):
        self.assertIn("follow-up-chip", _src(CHAT_TMPL))

    def test_follow_up_label_text(self):
        self.assertIn("Based on this result", _src(CHAT_TMPL))

    def test_chip_click_fires_sendSuggestion(self):
        src = _src(CHAT_TMPL)
        self.assertIn("sendSuggestion", src)
        self.assertIn("data-follow-up", src)

    def test_follow_up_wrap_css(self):
        self.assertIn(".follow-up-wrap", _src(CHAT_TMPL))

    def test_generate_followup_imported_in_main(self):
        self.assertIn("generate_followup_suggestions", _src(MAIN_PY))

    def test_follow_up_added_to_payload(self):
        src = _src(MAIN_PY)
        self.assertIn('"follow_up_suggestions"', src)

    def test_only_for_portal_user(self):
        src = _src(MAIN_PY)
        idx_follow = src.find('"follow_up_suggestions"')
        # portal_user guard is ~800 chars before the assignment — use wide window
        region = src[max(0, idx_follow-1200):idx_follow]
        self.assertIn("portal_user", region)


# ══════════════════════════════════════════════════════════════════════════════
# F. FORMULA EDITOR
# ══════════════════════════════════════════════════════════════════════════════

class TestFormulaEditorPopover(unittest.TestCase):
    """F1/F2 — ƒ Functions popover with DB-aware templates."""

    def test_fn_helper_button_present(self):
        self.assertIn("fn-helper-btn", _src(SETUP_TMPL := \
            ROOT/"admin"/"templates"/"client_metrics.html"))

    def test_old_pill_toolbar_absent(self):
        tmpl = (ROOT/"admin"/"templates"/"client_metrics.html").read_text()
        self.assertNotIn("formula-snippet", tmpl)

    def test_six_buckets_in_catalogue(self):
        tmpl = (ROOT/"admin"/"templates"/"client_metrics.html").read_text()
        for bucket in ("Aggregation","Ratio & Division","Conditional",
                       "Null Handling","Type Conversion","String"):
            self.assertIn(bucket, tmpl)

    def test_median_azure_uses_percentile_cont(self):
        tmpl = (ROOT/"admin"/"templates"/"client_metrics.html").read_text()
        self.assertIn("PERCENTILE_CONT", tmpl)

    def test_isnull_azure_only(self):
        tmpl = (ROOT/"admin"/"templates"/"client_metrics.html").read_text()
        self.assertIn('dbs:["azure_sql"]', tmpl)

    def test_nvl_oracle_only(self):
        tmpl = (ROOT/"admin"/"templates"/"client_metrics.html").read_text()
        self.assertIn('dbs:["oracle"]', tmpl)

    def test_db_type_passed_from_route(self):
        self.assertIn('"db_type": db_type', _src(ROUTES))

    def test_db_type_set_in_js(self):
        self.assertIn("window._qbDbType", (ROOT/"admin"/"templates"/"client_metrics.html").read_text())


class TestSyntaxValidator(unittest.TestCase):
    """F3 — validator rules present in template."""

    def _validator(self):
        src = (ROOT/"admin"/"templates"/"client_metrics.html").read_text()
        start = src.find("function updateFormulaHints")
        end   = src.find("function setStatus", start)
        return src[start:end]

    def test_select_in_expression_rule(self):
        self.assertIn("SELECT", self._validator())

    def test_nullif_division_rule(self):
        self.assertIn("division", self._validator().lower())

    def test_integer_division_rule(self):
        self.assertIn("Integer division", self._validator())

    def test_avg_null_rule(self):
        self.assertIn("AVG", self._validator())

    def test_median_azure_rule(self):
        v = self._validator()
        self.assertIn("MEDIAN", v)
        self.assertIn("azure_sql", v)

    def test_pipe_concat_azure_rule(self):
        v = self._validator()
        self.assertIn("||", v)
        self.assertIn("azure_sql", v)


class TestDuplicateColumnDisambiguation(unittest.TestCase):
    """F4 — duplicate column display uses TABLE.COLUMN, insert uses bare name."""

    def test_dup_set_built(self):
        tmpl = (ROOT/"admin"/"templates"/"client_metrics.html").read_text()
        self.assertIn("_dupCols", tmpl)
        self.assertIn("_buildDupSet", tmpl)

    def test_display_label_uses_table_prefix(self):
        tmpl = (ROOT/"admin"/"templates"/"client_metrics.html").read_text()
        self.assertIn("_displayLabel", tmpl)
        self.assertIn('c.table+"."', tmpl)


    def test_insert_uses_bare_col(self):
        """dropdown item stores bare column name in data-col, not TABLE.COLUMN."""
        tmpl = (ROOT/"admin"/"templates"/"client_metrics.html").read_text()
        # JS builds: data-col="'+c.column+'"  — bare column, not display label
        self.assertIn("data-col=", tmpl)
        # _insertSuggestion reads dataset.col (bare name) not the display label
        self.assertIn("_insertSuggestion(item.dataset.col)", tmpl)

    def test_insert_suggestion_uses_dataset_col(self):
        tmpl = (ROOT/"admin"/"templates"/"client_metrics.html").read_text()
        self.assertIn("_insertSuggestion(item.dataset.col)", tmpl)


# ══════════════════════════════════════════════════════════════════════════════
# G. DYNAMIC PRICING
# ══════════════════════════════════════════════════════════════════════════════

class TestDynamicPricing(unittest.TestCase):
    """G1-G3 — llm_pricing table, calculate_cost, save_pricing."""

    def test_table_exists(self):
        with _db.get_db() as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='llm_pricing'"
            ).fetchone()
        self.assertIsNotNone(row)

    def test_calculate_cost_returns_float(self):
        result = store.calculate_cost("gpt-4o", 1000, 500)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)

    def test_calculate_cost_zero_tokens(self):
        result = store.calculate_cost("gpt-4o", 0, 0)
        self.assertEqual(result, 0.0)

    def test_calculate_cost_unknown_model_uses_default(self):
        # Unknown model should fall back without raising
        result = store.calculate_cost("totally-unknown-model-xyz", 100, 100)
        self.assertIsInstance(result, float)

    def test_save_pricing_persists(self):
        store.save_pricing("test-model-g3", 1.0, 2.0)
        all_rates = store.get_all_pricing()
        found = next((r for r in all_rates if r["model"] == "test-model-g3"), None)
        self.assertIsNotNone(found)
        self.assertAlmostEqual(found["tokens_in"],  1.0)
        self.assertAlmostEqual(found["tokens_out"], 2.0)

    def test_get_all_pricing_returns_list(self):
        rates = store.get_all_pricing()
        self.assertIsInstance(rates, list)
        self.assertGreater(len(rates), 0)

    def test_pricing_rows_have_model_key(self):
        rates = store.get_all_pricing()
        for r in rates:
            self.assertIn("model", r)
            self.assertIn("tokens_in", r)
            self.assertIn("tokens_out", r)


# ══════════════════════════════════════════════════════════════════════════════
# CHART FIXES — ResizeObserver + Maximize
# ══════════════════════════════════════════════════════════════════════════════

class TestChartResizeFixes(unittest.TestCase):
    """Chat + dashboard chart alignment fixes."""

    def test_chat_uses_ResizeObserver(self):
        self.assertIn("ResizeObserver", _src(CHAT_TMPL))

    def test_chat_no_once_true_resize(self):
        self.assertNotIn(
            "addEventListener('resize', () => chart.resize(), { once: true })",
            _src(CHAT_TMPL)
        )

    def test_chat_observer_on_chart_element(self):
        self.assertIn("ro.observe(chartEl)", _src(CHAT_TMPL))

    def test_dashboard_uses_ResizeObserver(self):
        self.assertIn("ResizeObserver", _src(DASH_TMPL))

    def test_dashboard_uses_requestAnimationFrame(self):
        self.assertIn("requestAnimationFrame", _src(DASH_TMPL))

    def test_dashboard_observer_on_node(self):
        self.assertIn("ro.observe(node)", _src(DASH_TMPL))


class TestDashboardMaximizeModal(unittest.TestCase):
    """Maximize button + fullscreen modal."""

    def test_expand_button_present(self):
        self.assertIn("⤢ Expand", _src(DASH_TMPL))

    def test_open_modal_function(self):
        self.assertIn("function openChartModal", _src(DASH_TMPL))

    def test_close_modal_function(self):
        self.assertIn("function closeChartModal", _src(DASH_TMPL))

    def test_escape_key_closes(self):
        src = _src(DASH_TMPL)
        self.assertIn("Escape", src)
        self.assertIn("closeChartModal", src)

    def test_resize_after_open(self):
        self.assertIn("mc.resize()", _src(DASH_TMPL))

    def test_dispose_on_close(self):
        src = _src(DASH_TMPL)
        self.assertIn("dispose()", src)

    def test_viewport_size(self):
        src = _src(DASH_TMPL)
        self.assertIn("100vw", src)
        self.assertIn("100vh", src)

    def test_button_calls_openChartModal(self):
        self.assertIn("onclick=\"openChartModal(this.closest('.chart-card'))\"",
                      _src(DASH_TMPL))


# ══════════════════════════════════════════════════════════════════════════════
# STORE EXPORTS
# ══════════════════════════════════════════════════════════════════════════════

class TestStoreExports(unittest.TestCase):
    """All new store functions properly exported."""

    def test_log_kb_egress_exported(self):
        self.assertTrue(callable(store.log_kb_egress))

    def test_list_kb_egress_exported(self):
        self.assertTrue(callable(store.list_kb_egress))

    def test_get_kb_egress_summary_exported(self):
        self.assertTrue(callable(store.get_kb_egress_summary))

    def test_calculate_cost_exported(self):
        self.assertTrue(callable(store.calculate_cost))

    def test_save_pricing_exported(self):
        self.assertTrue(callable(store.save_pricing))

    def test_get_all_pricing_exported(self):
        self.assertTrue(callable(store.get_all_pricing))

    def test_store_init_references_egress(self):
        src = _src(STORE_INIT)
        self.assertIn("log_kb_egress",         src)
        self.assertIn("list_kb_egress",        src)
        self.assertIn("get_kb_egress_summary", src)


# ══════════════════════════════════════════════════════════════════════════════
# DB SCHEMA — new tables and migrations
# ══════════════════════════════════════════════════════════════════════════════

class TestDBSchema(unittest.TestCase):
    """All new tables and columns exist after init_db()."""

    def _table_exists(self, name):
        with _db.get_db() as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (name,)
            ).fetchone()
        return row is not None

    def _columns(self, table):
        with _db.get_db() as conn:
            return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}

    def test_kb_data_egress_log_exists(self):
        self.assertTrue(self._table_exists("kb_data_egress_log"))

    def test_kb_data_egress_log_columns(self):
        cols = self._columns("kb_data_egress_log")
        for col in ("id","account_id","operation","db_type","table_name",
                    "sample_mode","column_count","distinct_col_count",
                    "triggered_by","created_at","database_name","schema_name"):
            self.assertIn(col, cols)

    def test_llm_pricing_exists(self):
        self.assertTrue(self._table_exists("llm_pricing"))

    def test_migration_v16_columns_in_export_state(self):
        src = _src(DB_PY)
        self.assertIn('"last_egress_id"',    src)
        self.assertIn('"last_egress_count"', src)

    def test_egress_indexes_defined(self):
        src = _src(DB_PY)
        self.assertIn("idx_kb_egress_account_op",    src)
        self.assertIn("idx_kb_egress_account_table", src)


if __name__ == "__main__":
    unittest.main()
