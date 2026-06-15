import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import core.log_export as log_export
import core.schema as schema


class FakeCursor:
    def __init__(self, max_values=None):
        self.sql = []
        self.params = []
        self.max_values = max_values or {}
        self._next = (0,)

    def execute(self, sql, params=None):
        self.sql.append(sql)
        self.params.append(params)
        upper = sql.upper()
        if "MAX(SOURCE_ID)" in upper and "QUERY_LOG" in upper:
            self._next = (self.max_values.get("QUERY_LOG", 0),)
        elif "MAX(SOURCE_ID)" in upper and "LLM_CALL_LOG" in upper:
            self._next = (self.max_values.get("LLM_CALL_LOG", 0),)
        return self

    def fetchone(self):
        return self._next


class FakeConn:
    def __init__(self, cursor):
        self._cursor = cursor
        self.committed = False
        self.closed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True

    def close(self):
        self.closed = True


class ExternalLogExportTests(unittest.TestCase):

    def test_azure_provision_creates_schema_and_both_tables(self):
        cur = FakeCursor()
        conn = FakeConn(cur)
        cfg = {
            "db_type": "azure_sql",
            "credentials": {
                "server": "example.database.windows.net",
                "database": "app",
                "user": "bot",
                "password": "secret",
                "log_schema": "logs",
            },
        }

        with patch("core.log_export._az_connect", return_value=conn):
            result = log_export.provision_external_log_store(cfg, record_state=False)

        ddl = "\n".join(cur.sql)
        self.assertEqual(result["schema"], "LOGS")
        self.assertIn("CREATE SCHEMA [LOGS]", ddl)
        self.assertIn("[LOGS].[QUERY_LOG]", ddl)
        self.assertIn("[LOGS].[LLM_CALL_LOG]", ddl)
        self.assertTrue(conn.committed)
        self.assertTrue(conn.closed)

    def test_sync_uses_target_max_source_ids_and_inserts_new_rows(self):
        cur = FakeCursor({"QUERY_LOG": 10, "LLM_CALL_LOG": 20})
        conn = FakeConn(cur)
        query_rows = [(11, "acct", None, "", "q", "select 1", 1, 1, "", "openai", "gpt-4o", 1, 2, 0.01, 50, "2026-04-25")]
        llm_rows = [(21, "acct", "qid", "rid", "q", "sql", "openai", "gpt-4o", "success", "hash", "preview", 100, "", "2026-04-25")]
        cfg = {
            "id": 0,
            "db_type": "azure_sql",
            "credentials": {
                "server": "example.database.windows.net",
                "database": "app",
                "user": "bot",
                "password": "secret",
                "log_schema": "logs",
            },
        }

        with patch("core.log_export.provision_external_log_store") as provision, \
             patch("core.log_export._az_connect", return_value=conn), \
             patch("core.log_export._fetch_query_rows_after", return_value=query_rows) as fetch_q, \
             patch("core.log_export._fetch_llm_rows_after", return_value=llm_rows) as fetch_l, \
             patch("core.log_export._fetch_egress_rows_after", return_value=[]):
            result = log_export.sync_external_logs(cfg)

        provision.assert_called_once()
        fetch_q.assert_called_once_with(10, 10000)
        fetch_l.assert_called_once_with(20, 10000)
        insert_sql = [sql for sql in cur.sql if sql.upper().startswith("INSERT INTO")]
        self.assertEqual(len(insert_sql), 2)
        self.assertEqual(result["query_count"], 1)
        self.assertEqual(result["llm_count"], 1)
        self.assertEqual(result["last_query_id"], 11)
        self.assertEqual(result["last_llm_id"], 21)

    def test_snowflake_connection_ignores_log_export_metadata(self):
        connector = types.ModuleType("snowflake.connector")
        connector.errors = types.SimpleNamespace(OperationalError=RuntimeError)
        captured = {}

        def fake_connect(**kwargs):
            captured.update(kwargs)
            return object()

        connector.connect = fake_connect
        snowflake_pkg = types.ModuleType("snowflake")
        snowflake_pkg.connector = connector

        with patch.dict(sys.modules, {
            "snowflake": snowflake_pkg,
            "snowflake.connector": connector,
        }):
            schema._sf_connect({
                "account": "acct",
                "user": "bot",
                "password": "secret",
                "warehouse": "wh",
                "database": "app",
                "schema": "PUBLIC",
                "log_export_enabled": "1",
                "log_schema": "LOGS",
                "log_export_time": "02:00",
            }, max_retries=1)

        self.assertIn("account", captured)
        self.assertNotIn("log_export_enabled", captured)
        self.assertNotIn("log_schema", captured)
        self.assertNotIn("log_export_time", captured)


if __name__ == "__main__":
    unittest.main()
