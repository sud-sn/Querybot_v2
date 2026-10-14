"""
A PostgreSQL connection's Schemas field finds its schemas whatever case they were typed in.

PostgreSQL keeps an unquoted schema name in lower case. A connection limited to "SALES" listed nothing in Browse
schemas for the schema sales, and its discovery read no table, with nothing on screen to say why. The field is
now matched without regard to case, and a Browse that finds nothing says which schemas the user can read instead.

The real browse and discovery; the database is a stand-in answering their catalogue queries.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from core.schema_discovery import NothingVisible, _discover_postgres

# (schema, table, relkind) as pg_class lists what the user may read.
CATALOGUE = [("public", "decoy", "r"), ("sales", "orders", "r"), ("sales", "customers", "r"), ("sales", "v_month", "v")]


class _Database:
    def __init__(self, rows=CATALOGUE):
        self.rows = rows

    def cursor(self):
        return self

    def execute(self, sql, *_):
        self.sql = " ".join(sql.split())

    def fetchone(self):
        return ("retail_dw",)

    def fetchall(self):
        return list(self.rows)

    def close(self):
        pass


def _browse(schemas: str, rows=CATALOGUE) -> dict:
    with patch("core.schema._pg_connect", return_value=_Database(rows)):
        return _discover_postgres({"host": "db", "database": "retail_dw", "user": "qbreader", "schemas": schemas})


class TestBrowseSchemas:

    @pytest.mark.parametrize("typed", ["sales", "SALES", " Sales "])
    def test_the_schema_is_found_whatever_its_case(self, typed):
        tree = _browse(typed)
        assert tree == {"RETAIL_DW": {"sales": {"tables": ["orders", "customers"], "views": ["v_month"]}}}

    def test_a_schema_with_nothing_readable_names_the_schemas_that_have_tables(self):
        with pytest.raises(NothingVisible) as said:
            _browse("finance")
        assert "finance" in str(said.value) and "public, sales" in str(said.value)

    def test_a_user_who_may_read_nothing_is_told_the_grants(self):
        with pytest.raises(NothingVisible, match="GRANT SELECT ON ALL TABLES IN SCHEMA"):
            _browse("", rows=[])


class TestDiscovery:

    def test_the_schema_typed_in_capitals_is_read(self, tmp_path):
        from core import schema

        class _Discovered(_Database):
            def fetchall(self):
                if "FROM pg_class c" in self.sql:
                    return [(s, t, "BASE TABLE", 3, None, i) for i, (s, t, _k) in enumerate(CATALOGUE)]
                return []

        with patch("core.schema._pg_connect", return_value=_Discovered()):
            count = schema.discover_and_write(
                {"host": "db", "database": "retail_dw", "user": "qbreader", "schemas": "SALES"}, "postgres",
                str(tmp_path), allowed_tables=None, masking_config={})
        assert count == 3
