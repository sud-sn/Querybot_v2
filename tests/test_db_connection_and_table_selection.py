import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import core.schema as schema


class _FakeConn:
    def __init__(self, row):
        self.row = row
        self.closed = False

    def cursor(self, *args, **kwargs):
        return self

    def execute(self, *args, **kwargs):
        return None

    def fetchone(self):
        return self.row

    def close(self):
        self.closed = True


class TestConnectionTests(unittest.TestCase):

    def test_snowflake_smoke_query_returns_connection_details(self):
        conn = _FakeConn(("APP_DB", "PUBLIC", "BOT_USER"))
        with patch("core.schema._sf_connect", return_value=conn) as mock_connect:
            details = schema.test_connection({"account": "acct"}, "snowflake")

        # test_connection builds a smoke_cfg with login_timeout and network_timeout
        # capped at 15 s — verify the caps are applied
        call_cfg = mock_connect.call_args.args[0]
        self.assertEqual(call_cfg["account"], "acct")
        self.assertEqual(call_cfg["login_timeout"], 15)
        self.assertEqual(call_cfg["network_timeout"], 15)
        self.assertEqual(mock_connect.call_args.kwargs["max_retries"], 1)
        self.assertTrue(conn.closed)
        self.assertEqual(details["database"], "APP_DB")
        self.assertEqual(details["schema"], "PUBLIC")
        self.assertEqual(details["user"], "BOT_USER")

    def test_azure_smoke_query_caps_login_timeout_for_ui(self):
        conn = _FakeConn(("APP_DB", "dbo", "bot_user"))
        with patch("core.schema._az_connect", return_value=conn) as mock_connect:
            details = schema.test_connection({"login_timeout": 60}, "azure_sql")

        cfg, kwargs = mock_connect.call_args.args[0], mock_connect.call_args.kwargs
        self.assertEqual(cfg["login_timeout"], 15)
        self.assertEqual(kwargs["max_retries"], 1)
        self.assertEqual(details["database"], "APP_DB")


class TableSelectionMatchTests(unittest.TestCase):

    def test_qualified_selection_matches_selected_schema_only(self):
        allowed = {"CRM.SALES.CUSTOMERS"}
        self.assertTrue(schema._allowed_table_match(allowed, "CUSTOMERS", "SALES", "CRM"))
        self.assertFalse(schema._allowed_table_match(allowed, "CUSTOMERS", "DBO", "CRM"))
        self.assertFalse(schema._allowed_table_match(allowed, "CUSTOMERS", "SALES", "OTHER_DB"))

    def test_legacy_bare_selection_still_matches_by_table_name(self):
        self.assertTrue(schema._allowed_table_match({"CUSTOMERS"}, "CUSTOMERS", "SALES", "CRM"))


class AzureSqlDiscoverySelectionTests(unittest.TestCase):

    def test_discovery_writes_only_selected_schema_qualified_table(self):
        class FakeCursor:
            description = []

            def __init__(self):
                self.rows = []
                self._fetchone_val = None

            def execute(self, sql, *params):
                sql_u = " ".join(sql.upper().split())  # normalise whitespace
                if "DB_NAME" in sql_u:
                    self._fetchone_val = ("CRM",)
                    self.rows = []
                elif "SYS.DATABASES" in sql_u:
                    self.rows = []
                elif "INFORMATION_SCHEMA.TABLES" in sql_u:
                    self.rows = [
                        ("dbo", "CUSTOMERS", "BASE TABLE"),
                        ("sales", "CUSTOMERS", "BASE TABLE"),
                        ("sales", "ORDERS", "BASE TABLE"),
                    ]
                elif "INFORMATION_SCHEMA.COLUMNS" in sql_u:
                    self.rows = [("ID", "int", "NO", None, 10)]
                else:
                    self.rows = []

            def fetchall(self):
                return self.rows

            def fetchone(self):
                v = self._fetchone_val
                self._fetchone_val = None
                return v

        class FakeConn:
            def __init__(self):
                self.cur = FakeCursor()

            def cursor(self):
                return self.cur

            def close(self):
                pass

        written = {}

        def fake_write_text(path, text, encoding=None):
            written[path.name] = text
            return len(text)

        with patch.object(Path, "write_text", fake_write_text):
            with patch("core.schema._az_connect", return_value=FakeConn()):
                count = schema._discover_azure_sql(
                    {"database": "CRM", "schema": "dbo"},
                    ROOT / "_virtual_schema_out",
                    allowed={"CRM.SALES.CUSTOMERS"},
                )

        self.assertEqual(count, 1)
        self.assertIn("CUSTOMERS.md", written)
        master = json.loads(written["_schema.json"])
        # Master keys are now 3-part FQNs: DB.SCHEMA.TABLE
        self.assertEqual(list(master.keys()), ["CRM.SALES.CUSTOMERS"])
        self.assertEqual(master["CRM.SALES.CUSTOMERS"]["schema"], "sales")

    def test_discovery_keeps_duplicate_table_names_from_different_schemas(self):
        class FakeCursor:
            description = []

            def __init__(self):
                self.rows = []
                self._fetchone_val = None

            def execute(self, sql, *params):
                sql_u = " ".join(sql.upper().split())  # normalise whitespace
                if "DB_NAME" in sql_u:
                    self._fetchone_val = ("CRM",)
                    self.rows = []
                elif "SYS.DATABASES" in sql_u:
                    self.rows = []
                elif "INFORMATION_SCHEMA.TABLES" in sql_u:
                    self.rows = [
                        ("dbo", "CUSTOMERS", "BASE TABLE"),
                        ("sales", "CUSTOMERS", "BASE TABLE"),
                    ]
                elif "INFORMATION_SCHEMA.COLUMNS" in sql_u:
                    self.rows = [("ID", "int", "NO", None, 10)]
                else:
                    self.rows = []

            def fetchall(self):
                return self.rows

            def fetchone(self):
                v = self._fetchone_val
                self._fetchone_val = None
                return v

        class FakeConn:
            def __init__(self):
                self.cur = FakeCursor()

            def cursor(self):
                return self.cur

            def close(self):
                pass

        written = {}

        def fake_write_text(path, text, encoding=None):
            written[path.name] = text
            return len(text)

        with patch.object(Path, "write_text", fake_write_text):
            with patch("core.schema._az_connect", return_value=FakeConn()):
                count = schema._discover_azure_sql(
                    {"database": "CRM", "schema": "dbo"},
                    ROOT / "_virtual_schema_out",
                    allowed={"CRM.DBO.CUSTOMERS", "CRM.SALES.CUSTOMERS"},
                )

        self.assertEqual(count, 2)
        self.assertIn("dbo__CUSTOMERS.md", written)
        self.assertIn("sales__CUSTOMERS.md", written)
        master = json.loads(written["_schema.json"])
        # Master keys are now 3-part FQNs: DB.SCHEMA.TABLE
        self.assertEqual(
            set(master.keys()),
            {"CRM.DBO.CUSTOMERS", "CRM.SALES.CUSTOMERS"}
        )


if __name__ == "__main__":
    unittest.main()
