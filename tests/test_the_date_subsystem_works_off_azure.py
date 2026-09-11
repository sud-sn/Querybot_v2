"""
tests/test_the_date_subsystem_works_off_azure.py

Two dialect defects, each of which switched off a whole warehouse.

1. core.date_roles.physical_date_key_type decides whether a column is a
   business date at all. A spelling it does not know is not a date, so no date
   role is discovered for it, and a fact whose only date column is spelled that
   way has no governed business date -- the reader gets a refusal, or an
   ungoverned guess. It knew the SQL Server datetime family and PostgreSQL's
   two long forms. Executed against what the catalogues actually return:

       Snowflake  TIMESTAMP_NTZ / _LTZ / _TZ     -> not a date
       Oracle     TIMESTAMP(6), TIMESTAMP(6) WITH TIME ZONE -> not a date

   Those are THE standard timestamp types on those two warehouses.

2. Oracle rejects AS before a TABLE alias -- `FROM "M"."T" AS t` is ORA-00933,
   "SQL command not properly ended" -- and every business-date anchor probe
   carries one. So on Oracle every probe failed, the exception was caught and
   logged as a probe failure, the negative cache suppressed the retry, and the
   answer fell back to the in-query anchor. The fast path was permanently off
   for a whole dialect and nothing said so, because failing open is exactly
   what this code is designed to do.

Both fixes stay conservative where conservatism was deliberate: SQL Server's
bare `timestamp` is a rowversion binary value and is still not a date. It does
not need a dialect argument to tell them apart, because the spellings do:
TIMESTAMP_NTZ exists only on Snowflake, a precision only on Oracle and
PostgreSQL, and SQL Server's rowversion is always bare.
"""

from __future__ import annotations

import pytest
import sqlglot

from core.date_anchor import build_anchor_probe_sql, build_key_order_check_sql
from core.date_roles import physical_date_key_type

SEMI_JOIN = {
    "anchor_policy": "latest_available",
    "fact_table": "MART.INVOICE_FCT", "fact_column": "IVC_DT_KEY",
    "date_column": "DT_VAL",
    "dimension_table": "MART.DT_DMS", "dimension_key": "DT_KEY",
}
FACT_NATIVE = {
    "anchor_policy": "latest_available",
    "fact_table": "MART.INVOICE_FCT", "fact_column": "IVC_DT",
    "date_column": "IVC_DT",
}
DIALECTS = {"azure_sql": "tsql", "snowflake": "snowflake", "oracle": "oracle"}


class TestEveryWarehousesOwnTimestampSpelling:

    @pytest.mark.parametrize("spelling", [
        "TIMESTAMP_NTZ", "TIMESTAMP_LTZ", "TIMESTAMP_TZ",
        "TIMESTAMP_NTZ(9)", "timestamp_ntz", "DATETIME",
    ])
    def test_snowflake(self, spelling):
        assert physical_date_key_type(spelling) == "timestamp", spelling

    @pytest.mark.parametrize("spelling", [
        "TIMESTAMP(6)", "TIMESTAMP(9)", "TIMESTAMP (6)",
        "TIMESTAMP(6) WITH TIME ZONE", "TIMESTAMP(6) WITH LOCAL TIME ZONE",
    ])
    def test_oracle(self, spelling):
        assert physical_date_key_type(spelling) == "timestamp", spelling

    @pytest.mark.parametrize("spelling", [
        "timestamp with time zone", "timestamp without time zone",
        "timestamptz", "timestamp(3) without time zone",
    ])
    def test_postgres(self, spelling):
        assert physical_date_key_type(spelling) == "timestamp", spelling

    @pytest.mark.parametrize("spelling", [
        "datetime", "datetime2", "datetime2(7)", "smalldatetime",
        "datetimeoffset", "datetimeoffset(7)",
    ])
    def test_sql_server_is_unchanged(self, spelling):
        assert physical_date_key_type(spelling) == "timestamp", spelling

    @pytest.mark.parametrize("spelling", ["date", "DATE", " Date "])
    def test_a_plain_date_is_a_plain_date_everywhere(self, spelling):
        assert physical_date_key_type(spelling) == "native_date", spelling


class TestTheDeliberateExclusionsSurvived:
    """The conservative choices here were right and must not be traded away
    for the fix above."""

    @pytest.mark.parametrize("spelling", ["timestamp", "TIMESTAMP", "rowversion"])
    def test_sql_servers_rowversion_is_still_not_a_date(self, spelling):
        """A bare `timestamp` on SQL Server is a binary row version. Treating
        it as a business date would put a governed date filter on a column of
        opaque bytes -- and a precision or a time-zone clause is what separates
        the real thing, because SQL Server's never carries either."""
        assert physical_date_key_type(spelling) == ""

    @pytest.mark.parametrize("spelling", [
        "time", "time(7)", "int", "bigint", "varchar(10)", "binary(8)",
        "bit", "numeric(18,2)", "", None, "   ",
    ])
    def test_nothing_else_became_a_date(self, spelling):
        assert physical_date_key_type(spelling) == ""


class TestEveryProbeParsesInItsOwnDialect:
    """The statements are built as strings, so nothing else checks they are
    legal in the dialect they were built for. This is what "every Oracle probe
    failed" looked like from outside: a caught exception and a log line."""

    @pytest.mark.parametrize("db_type,dialect", list(DIALECTS.items()))
    @pytest.mark.parametrize("policy,label", [
        (SEMI_JOIN, "semi-join"), (FACT_NATIVE, "fact-native"),
    ])
    def test_the_anchor_probe_parses(self, db_type, dialect, policy, label):
        sql = build_anchor_probe_sql(policy, db_type)
        assert sql, (db_type, label)
        sqlglot.parse_one(sql, dialect=dialect)

    @pytest.mark.parametrize("db_type,dialect", list(DIALECTS.items()))
    def test_the_key_order_check_parses(self, db_type, dialect):
        sql = build_key_order_check_sql(SEMI_JOIN, db_type)
        assert sql, db_type
        sqlglot.parse_one(sql, dialect=dialect)


class TestOracleGetsNoASBeforeATableAlias:

    @pytest.mark.parametrize("policy", [SEMI_JOIN, FACT_NATIVE])
    def test_the_probe_has_no_as_before_a_table_alias(self, policy):
        sql = build_anchor_probe_sql(policy, "oracle")
        assert " AS anchor_fact" not in sql, sql
        assert " AS anchor_date" not in sql, sql
        # The alias is still there -- the probe references it.
        assert "anchor_fact" in sql or "anchor_date" in sql

    def test_the_derived_table_alias_too(self):
        sql = build_key_order_check_sql(SEMI_JOIN, "oracle")
        assert ") AS ordered" not in sql, sql
        assert ") ordered" in sql, sql

    def test_a_column_alias_keeps_its_AS(self):
        """Oracle accepts AS for a column alias and rejects it for a table
        alias. Stripping both would be a different bug, and the probe's result
        is read back by the name max_business_date."""
        sql = build_anchor_probe_sql(FACT_NATIVE, "oracle")
        assert "AS max_business_date" in sql, sql

    @pytest.mark.parametrize("db_type", ["azure_sql", "snowflake"])
    def test_the_other_dialects_keep_theirs(self, db_type):
        """Snowflake and SQL Server both accept AS, and removing it there would
        be a gratuitous change to working SQL."""
        sql = build_anchor_probe_sql(SEMI_JOIN, db_type)
        assert " AS anchor_date" in sql, sql
        assert " AS anchor_fact" in sql, sql


class TestTheProbeStillReadsTheRightThing:
    """A dialect fix that changed WHAT is being measured would be worse than
    the bug."""

    @pytest.mark.parametrize("db_type", list(DIALECTS))
    def test_the_semi_join_reads_the_dimensions_date_and_proves_fact_rows(
            self, db_type):
        sql = build_anchor_probe_sql(SEMI_JOIN, db_type)
        assert "MAX(anchor_date." in sql
        assert "DT_VAL" in sql
        assert "EXISTS" in sql
        assert "IVC_DT_KEY" in sql and "DT_KEY" in sql

    @pytest.mark.parametrize("db_type", list(DIALECTS))
    def test_the_fact_native_probe_needs_no_join(self, db_type):
        sql = build_anchor_probe_sql(FACT_NATIVE, db_type)
        assert "MAX(anchor_fact." in sql
        assert "EXISTS" not in sql
        assert "JOIN" not in sql.upper()

    @pytest.mark.parametrize("db_type", list(DIALECTS))
    def test_an_incomplete_policy_still_probes_nothing(self, db_type):
        assert build_anchor_probe_sql({"fact_table": "MART.F"}, db_type) == ""
        assert build_anchor_probe_sql(None, db_type) == ""
