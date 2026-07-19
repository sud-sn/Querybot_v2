"""
tests/test_fqn_validator.py

Regression tests for the FQN (fully-qualified name) mismatch fix.

Before the fix, the SQL validator and ACL check used bare table names
from sqlglot's node.name, but known_tables and allowed_tables contained
3-part FQN keys (DB.SCHEMA.TABLE) after the multi-schema discovery
refactor.  This caused every query to fail with either
"Table not found in connected database" or "Access denied" even when
the SQL was perfectly valid.

These tests confirm:
  1. load_known_tables expands FQN keys into all name variants
  2. validate_sql passes when LLM generates bare table names
  3. validate_sql passes when LLM generates 2-part schema.table names
  4. validate_sql passes when LLM generates 3-part DB.SCHEMA.TABLE names
  5. ACL check (allowed_tables) works against FQN-style permission sets
  6. Invalid tables (genuinely not in the schema) still fail correctly
  7. Access denial still fires for tables not in the user's allowed set
  8. The SQL from the screenshot: [CHATBOT_DB].[HR].[Employee] passes

Run: python -m unittest tests.test_fqn_validator
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.schema as schema
from core.validator import validate_sql


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_schema_json(fqn_keys: list[str]) -> str:
    """Build a _schema.json string from a list of FQN keys."""
    data = {k: {"columns": [], "schema": k.split(".")[-2] if "." in k else "",
                "database": k.split(".")[0] if k.count(".") >= 2 else ""}
            for k in fqn_keys}
    return json.dumps(data)


# ── load_known_tables tests ────────────────────────────────────────────────────

class LoadKnownTablesExpansionTests(unittest.TestCase):
    """
    Verify that load_known_tables expands 3-part FQN keys into all variants
    so the validator's set-membership test works for any name format the LLM
    might produce.
    """

    def _load(self, fqn_keys: list[str]) -> set[str]:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "_schema.json"
            p.write_text(_make_schema_json(fqn_keys))
            return schema.load_known_tables(d)

    def test_three_part_fqn_adds_bare_name(self):
        tables = self._load(["CHATBOT_DB.PHARMACY.FACT_PRESCRIPTION_FILL"])
        self.assertIn("FACT_PRESCRIPTION_FILL", tables)

    def test_three_part_fqn_adds_schema_table(self):
        tables = self._load(["CHATBOT_DB.PHARMACY.FACT_PRESCRIPTION_FILL"])
        self.assertIn("PHARMACY.FACT_PRESCRIPTION_FILL", tables)

    def test_three_part_fqn_keeps_full_fqn(self):
        tables = self._load(["CHATBOT_DB.PHARMACY.FACT_PRESCRIPTION_FILL"])
        self.assertIn("CHATBOT_DB.PHARMACY.FACT_PRESCRIPTION_FILL", tables)

    def test_multiple_tables_all_expanded(self):
        tables = self._load([
            "CHATBOT_DB.HR.EMPLOYEE",
            "CHATBOT_DB.DBO.ORDERS",
            "CHATBOT_DB.LOGS.QUERY_LOG",
        ])
        # Bare names
        self.assertIn("EMPLOYEE",   tables)
        self.assertIn("ORDERS",     tables)
        self.assertIn("QUERY_LOG",  tables)
        # 2-part names
        self.assertIn("HR.EMPLOYEE",   tables)
        self.assertIn("DBO.ORDERS",    tables)
        self.assertIn("LOGS.QUERY_LOG", tables)
        # Full FQNs
        self.assertIn("CHATBOT_DB.HR.EMPLOYEE",    tables)
        self.assertIn("CHATBOT_DB.DBO.ORDERS",     tables)
        self.assertIn("CHATBOT_DB.LOGS.QUERY_LOG", tables)

    def test_two_part_key_adds_bare_name(self):
        """Legacy 2-part keys still expand correctly."""
        tables = self._load(["PHARMACY.DIM_PATIENT"])
        self.assertIn("DIM_PATIENT",         tables)
        self.assertIn("PHARMACY.DIM_PATIENT", tables)

    def test_bare_key_kept_as_is(self):
        """A bare (1-part) key is still included."""
        tables = self._load(["DIM_FORMULA"])
        self.assertIn("DIM_FORMULA", tables)

    def test_case_insensitive(self):
        """Keys are stored uppercase regardless of input case."""
        tables = self._load(["chatbot_db.pharmacy.dim_date"])
        self.assertIn("DIM_DATE",                      tables)
        self.assertIn("PHARMACY.DIM_DATE",             tables)
        self.assertIn("CHATBOT_DB.PHARMACY.DIM_DATE",  tables)

    def test_empty_schema_returns_empty_set(self):
        with tempfile.TemporaryDirectory() as d:
            result = schema.load_known_tables(d)
        self.assertEqual(result, set())


# ── validate_sql FQN tests ────────────────────────────────────────────────────

class ValidatorFQNKnownTablesTests(unittest.TestCase):
    """
    Verify that validate_sql accepts SQL with bare, 2-part, or 3-part table
    names when known_tables contains FQN-expanded entries.

    This is the core regression: before the fix all three formats returned
    "unknown_table" because only the bare name was checked against a set
    of FQN strings.
    """

    # Simulate the expanded set load_known_tables now returns
    KNOWN = {
        # Full FQN
        "CHATBOT_DB.HR.EMPLOYEE",
        "CHATBOT_DB.PHARMACY.DIM_PRESCRIBER",
        "CHATBOT_DB.PHARMACY.FACT_PRESCRIPTION_FILL",
        # 2-part
        "HR.EMPLOYEE",
        "PHARMACY.DIM_PRESCRIBER",
        "PHARMACY.FACT_PRESCRIPTION_FILL",
        # Bare
        "EMPLOYEE",
        "DIM_PRESCRIBER",
        "FACT_PRESCRIPTION_FILL",
    }

    def _validate(self, sql, allowed=None):
        return validate_sql(sql, self.KNOWN, db_type="azure_sql",
                            allowed_tables=allowed)

    # ── Bare table name (what sqlglot node.name returns for simple SQL) ────────

    def test_bare_table_name_passes(self):
        ok, _, code = self._validate(
            "SELECT TOP 20 * FROM Employee"
        )
        self.assertTrue(ok, f"Expected ok but got code={code}")
        self.assertEqual(code, "ok")

    def test_bare_table_name_select_with_where(self):
        ok, _, code = self._validate(
            "SELECT TOP 10 Prescriber_ID, City FROM DIM_Prescriber "
            "WHERE City IS NOT NULL"
        )
        self.assertTrue(ok)
        self.assertEqual(code, "ok")

    # ── 2-part schema.table name ───────────────────────────────────────────────

    def test_two_part_schema_table_passes(self):
        ok, _, code = self._validate(
            "SELECT TOP 20 * FROM [PHARMACY].[DIM_Prescriber]"
        )
        self.assertTrue(ok, f"Expected ok, got code={code}")
        self.assertEqual(code, "ok")

    def test_two_part_join_passes(self):
        ok, _, code = self._validate(
            "SELECT TOP 20 p.Prescriber_ID, f.Rx_ID "
            "FROM [PHARMACY].[DIM_Prescriber] p "
            "JOIN [PHARMACY].[FACT_PRESCRIPTION_FILL] f "
            "ON p.Prescriber_ID = f.Prescriber_ID"
        )
        self.assertTrue(ok)

    # ── 3-part DB.SCHEMA.TABLE name (what the new prompt produces) ────────────

    def test_three_part_name_passes(self):
        """The SQL from the screenshot: [CHATBOT_DB].[HR].[Employee]"""
        ok, _, code = self._validate(
            "SELECT StartDate, SUM(ActiveEmployeeCount) AS ActiveEmployeeCount "
            "FROM [CHATBOT_DB].[HR].[Employee] "
            "GROUP BY StartDate ORDER BY StartDate"
        )
        self.assertTrue(ok, f"Expected ok, got code={code}")
        self.assertEqual(code, "ok")

    def test_three_part_select_top(self):
        ok, _, code = self._validate(
            "SELECT TOP 10 Prescriber_ID, City "
            "FROM [CHATBOT_DB].[PHARMACY].[DIM_Prescriber] "
            "WHERE City IS NOT NULL ORDER BY City"
        )
        self.assertTrue(ok)
        self.assertEqual(code, "ok")

    def test_three_part_cross_schema_join(self):
        ok, _, code = self._validate(
            "SELECT TOP 20 p.Prescriber_ID, f.Fill_Status "
            "FROM [CHATBOT_DB].[PHARMACY].[DIM_Prescriber] p "
            "JOIN [CHATBOT_DB].[PHARMACY].[FACT_PRESCRIPTION_FILL] f "
            "ON p.Prescriber_ID = f.Prescriber_ID"
        )
        self.assertTrue(ok)
        self.assertEqual(code, "ok")

    # ── Unknown table still fails ──────────────────────────────────────────────

    def test_unknown_table_still_rejected(self):
        ok, msg, code = self._validate(
            "SELECT * FROM NONEXISTENT_TABLE"
        )
        self.assertFalse(ok)
        self.assertEqual(code, "unknown_table")
        self.assertIn("NONEXISTENT_TABLE", msg)

    def test_ddl_still_rejected(self):
        ok, _, code = self._validate("DROP TABLE Employee")
        self.assertFalse(ok)
        self.assertEqual(code, "ddl")

    def test_cannot_generate_sentinel(self):
        ok, _, code = self._validate("CANNOT_GENERATE")
        self.assertFalse(ok)
        self.assertEqual(code, "cannot_generate")


# ── ACL (allowed_tables) FQN tests ────────────────────────────────────────────

class ValidatorFQNAllowedTablesTests(unittest.TestCase):
    """
    Verify that the allowed_tables ACL check works when the permission set
    contains FQN-style names but the SQL uses bare or qualified names.

    Before the fix, a user with allowed_tables={"CHATBOT_DB.HR.EMPLOYEE"}
    would get "access denied" for every query even when they queried Employee
    — because "EMPLOYEE" was not in {"CHATBOT_DB.HR.EMPLOYEE"}.
    """

    KNOWN = {
        "CHATBOT_DB.HR.EMPLOYEE", "HR.EMPLOYEE", "EMPLOYEE",
        "CHATBOT_DB.PHARMACY.DIM_PRESCRIBER", "PHARMACY.DIM_PRESCRIBER",
        "DIM_PRESCRIBER",
        "CHATBOT_DB.PHARMACY.FACT_PRESCRIPTION_FILL",
        "PHARMACY.FACT_PRESCRIPTION_FILL", "FACT_PRESCRIPTION_FILL",
    }

    # Permission set stored as FQN (how it comes from the DB/group config)
    ALLOWED_FQN = {
        "CHATBOT_DB.HR.EMPLOYEE",
        "CHATBOT_DB.PHARMACY.DIM_PRESCRIBER",
    }

    def _validate(self, sql):
        return validate_sql(sql, self.KNOWN, db_type="azure_sql",
                            allowed_tables=self.ALLOWED_FQN)

    def test_bare_name_allowed_table_passes_acl(self):
        """User allowed CHATBOT_DB.HR.EMPLOYEE, SQL uses bare Employee — must pass."""
        ok, _, code = self._validate("SELECT TOP 20 * FROM Employee")
        self.assertTrue(ok, f"Expected ok but got code={code}")
        self.assertEqual(code, "ok")

    def test_two_part_allowed_table_passes_acl(self):
        ok, _, code = self._validate(
            "SELECT TOP 10 * FROM [PHARMACY].[DIM_Prescriber]"
        )
        self.assertTrue(ok)
        self.assertEqual(code, "ok")

    def test_three_part_allowed_table_passes_acl(self):
        """The screenshot SQL — must pass full ACL check."""
        ok, _, code = self._validate(
            "SELECT StartDate, SUM(ActiveEmployeeCount) "
            "FROM [CHATBOT_DB].[HR].[Employee] "
            "GROUP BY StartDate ORDER BY StartDate"
        )
        self.assertTrue(ok)
        self.assertEqual(code, "ok")

    def test_disallowed_table_still_denied(self):
        """User NOT allowed FACT_PRESCRIPTION_FILL — must be denied."""
        ok, msg, code = self._validate(
            "SELECT TOP 20 * FROM FACT_PRESCRIPTION_FILL"
        )
        self.assertFalse(ok)
        self.assertEqual(code, "access_denied")
        self.assertIn("FACT_PRESCRIPTION_FILL", msg)

    def test_admin_unrestricted_passes_any_table(self):
        """allowed_tables=None means admin — all tables pass."""
        ok, _, code = validate_sql(
            "SELECT * FROM FACT_PRESCRIPTION_FILL",
            self.KNOWN, db_type="azure_sql", allowed_tables=None,
        )
        self.assertTrue(ok)
        self.assertEqual(code, "ok")

    def test_mixed_allowed_and_denied_in_join(self):
        """JOIN where one table is allowed and one is not — access_denied."""
        ok, _, code = self._validate(
            "SELECT TOP 10 e.Employee_ID, f.Fill_Status "
            "FROM [CHATBOT_DB].[HR].[Employee] e "
            "JOIN [CHATBOT_DB].[PHARMACY].[FACT_PRESCRIPTION_FILL] f "
            "ON e.Employee_ID = f.Patient_ID"
        )
        self.assertFalse(ok)
        self.assertEqual(code, "access_denied")


# ── Integration: load_known_tables → validate_sql pipeline ───────────────────

class EndToEndFQNPipelineTests(unittest.TestCase):
    """
    Full integration: write a _schema.json with FQN keys (as schema.py
    now produces), load it with load_known_tables, then run validate_sql.
    This mirrors the exact runtime flow.
    """

    FQN_KEYS = [
        "CHATBOT_DB.HR.EMPLOYEE",
        "CHATBOT_DB.PHARMACY.DIM_PRESCRIBER",
        "CHATBOT_DB.PHARMACY.FACT_PRESCRIPTION_FILL",
    ]

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        p = Path(self._tmpdir.name) / "_schema.json"
        p.write_text(_make_schema_json(self.FQN_KEYS))
        self.known = schema.load_known_tables(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_screenshot_sql_passes_end_to_end(self):
        """
        The exact SQL from the user's screenshot must pass the full
        load_known_tables → validate_sql pipeline.
        """
        sql = (
            "SELECT StartDate, SUM(ActiveEmployeeCount) AS ActiveEmployeeCount "
            "FROM [CHATBOT_DB].[HR].[Employee] "
            "GROUP BY StartDate "
            "ORDER BY StartDate"
        )
        ok, msg, code = validate_sql(
            sql, self.known, db_type="azure_sql",
            allowed_tables={"CHATBOT_DB.HR.EMPLOYEE"},
        )
        self.assertTrue(ok, f"Screenshot SQL failed: code={code} msg={msg}")
        self.assertEqual(code, "ok")

    def test_bare_name_query_passes_end_to_end(self):
        sql = "SELECT TOP 20 * FROM DIM_Prescriber"
        ok, _, code = validate_sql(
            sql, self.known, db_type="azure_sql",
            allowed_tables={"CHATBOT_DB.PHARMACY.DIM_PRESCRIBER"},
        )
        self.assertTrue(ok)
        self.assertEqual(code, "ok")

    def test_cross_schema_join_passes_end_to_end(self):
        sql = (
            "SELECT TOP 10 p.Prescriber_ID, f.Fill_Status "
            "FROM [CHATBOT_DB].[PHARMACY].[DIM_Prescriber] p "
            "JOIN [CHATBOT_DB].[PHARMACY].[FACT_PRESCRIPTION_FILL] f "
            "ON p.Prescriber_ID = f.Prescriber_ID"
        )
        ok, _, code = validate_sql(
            sql, self.known, db_type="azure_sql",
            allowed_tables={
                "CHATBOT_DB.PHARMACY.DIM_PRESCRIBER",
                "CHATBOT_DB.PHARMACY.FACT_PRESCRIPTION_FILL",
            },
        )
        self.assertTrue(ok)
        self.assertEqual(code, "ok")

    def test_unknown_table_still_fails_end_to_end(self):
        sql = "SELECT * FROM GHOST_TABLE"
        ok, msg, code = validate_sql(
            sql, self.known, db_type="azure_sql", allowed_tables=None
        )
        self.assertFalse(ok)
        self.assertEqual(code, "unknown_table")


class SchemaMetadataKeyToleranceTests(unittest.TestCase):
    """Regression: discovery writes top-level "__*" metadata arrays (e.g.
    __fk_constraints) into _schema.json. _normalize_schema passes them
    through UNWRAPPED (as lists), and load_schema_columns crashed with
    "'list' object has no attribute 'get'" on every chat WebSocket connect
    for clients re-discovered after that change. load_known_tables silently
    added "__FK_CONSTRAINTS" to the validator's known-table set too."""

    def setUp(self):
        self._dir = tempfile.mkdtemp()
        payload = {
            "__fk_constraints": [
                {"from": "ERP.FACT_SALES.CUST_KEY", "to": "ERP.DIM_CUSTOMER.CUST_KEY"},
            ],
            "ERP.FACT_SALES": {
                "columns": [
                    {"name": "CUST_KEY", "type": "int"},
                    {"name": "NET_AMT", "type": "decimal(18,2)"},
                ],
            },
            # Legacy shape: columns stored as a bare list.
            "ERP.DIM_CUSTOMER": [
                {"name": "CUST_KEY", "type": "int"},
                {"name": "CUST_NAME", "type": "varchar(80)"},
            ],
        }
        (Path(self._dir) / "_schema.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def test_load_schema_columns_skips_metadata_and_reads_both_shapes(self):
        cols = schema.load_schema_columns(self._dir)  # crashed before the fix
        self.assertNotIn("__FK_CONSTRAINTS", cols)
        self.assertIn("NET_AMT", cols["ERP.FACT_SALES"])
        self.assertIn("CUST_NAME", cols["ERP.DIM_CUSTOMER"])  # legacy list shape

    def test_load_known_tables_excludes_metadata_keys(self):
        known = schema.load_known_tables(self._dir)
        self.assertNotIn("__FK_CONSTRAINTS", known)
        self.assertIn("ERP.FACT_SALES", known)
        self.assertIn("DIM_CUSTOMER", known)  # bare-name expansion intact


if __name__ == "__main__":
    unittest.main()
