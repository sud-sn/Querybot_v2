"""Learn this database survives what a real Azure SQL warehouse holds.

The second Learn on a real database stopped with "The column 'CUS_DMS_KEY' was
specified multiple times for 'x' (8156)". Discovery reads primary keys by table
name across schemas, so a table of the same name in a staging schema added its
key to the mart's: the key read CUS_DMS_KEY, CUS_DMS_KEY, and the key check
grouped by it twice. DuckDB, where Learn was tested, names a repeated column
for itself.

So here is a warehouse with what real ones have: discovery's own mistakes (a
key listed twice, a key naming another schema's column, a column listed twice,
a foreign key to a column that is not there), a same-named table in another
schema, a view, Azure SQL's awkward types (bit, ntext, uniqueidentifier,
datetimeoffset, time, money, float), date keys that are not dates (0, -1,
20250231, 99991231), dates in the years 1 and 9999, an empty table, a one-row
table, an all-null column, names with spaces, accents and reserved words, and a
table the service account may not read. Learn runs in Azure SQL's own SQL, every
query checked against its rules, sampling forced on: it must finish, learn the
mart, and say what it left out.
"""

from __future__ import annotations

import logging

import duckdb
import pytest

from core2.bootstrap import joins as join_step
from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_schema_json
from core2.bootstrap.profiler import ProfileOptions
from evals.core2.warehouse_sql import SpeakingWarehouse, Types

logging.getLogger("sqlglot").setLevel(logging.ERROR)

DAYS = 3 * 365


def _warehouse() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("CREATE SCHEMA stg")
    con.execute(f"""
        CREATE TABLE DT_DMS AS
        SELECT CAST(strftime(d, '%Y%m%d') AS INTEGER) AS DT_DMS_KEY, CAST(d AS DATE) AS DMS_DT,
               year(d) AS YR, month(d) AS MTH_NO, quarter(d) AS QTR
        FROM (SELECT DATE '2023-01-01' + INTERVAL (i) DAY AS d FROM range({DAYS}) r(i))
        UNION ALL SELECT -1, DATE '1900-01-01', 1900, 1, 1""")
    con.execute("""
        CREATE TABLE CUS_DMS AS
        SELECT i AS CUS_DMS_KEY, 'C' || lpad(CAST(i AS VARCHAR), 5, '0') AS CUS_CD, 'Customer ' || i AS CUS_NM,
               ['Retail', 'Trade', 'Government', 'Industrial'][1 + i % 4] AS "Group",
               i % 7 <> 0 AS ACTV_FLG, uuid() AS CUS_GUID, repeat('note ', 3 + i % 20) AS NOTE_TXT,
               ['Québec', 'Ontario', 'Manitoba'][1 + i % 3] AS "Libellé",
               CAST(NULL AS INTEGER) AS LGCY_ID
        FROM range(1, 301) r(i)""")
    con.execute(f"""
        CREATE TABLE PFT_CTR_CUS_DAT AS
        SELECT i AS LN_KEY, 'INV' || CAST(i // 3 AS VARCHAR) AS DOC_NO, 1 + i % 3 AS LN_NO,
               1 + i % 300 AS CUS_DMS_KEY,
               CAST(strftime(DATE '2023-01-01' + INTERVAL (i % {DAYS}) DAY, '%Y%m%d') AS INTEGER) AS IVC_DT_DMS_KEY,
               CASE WHEN i % 97 = 0 THEN -1 ELSE
                    CAST(strftime(DATE '2023-01-01' + INTERVAL (i % 200) DAY, '%Y%m%d') AS INTEGER) END
                   AS FST_CUS_IVC_DT_DMS_KEY,
               [0, -1, 20250231, 99991231, 20240115][1 + i % 5] AS ODD_DT_KEY,
               CASE WHEN i % 500 = 0 THEN DATE '0001-01-01' WHEN i % 501 = 0 THEN DATE '9999-12-31'
                    ELSE DATE '2023-01-01' + INTERVAL (i % {DAYS}) DAY END AS "Order Date",
               CAST(100 + (i * 37) % 9000 AS DECIMAL(18, 2)) AS SLS_AMT,
               CAST(60 + (i * 23) % 5000 AS DECIMAL(19, 4)) AS CST_AMT,
               1000000 + (i * 7919) % 90000000 AS QTY,
               CAST((i * 13) % 1000 AS DOUBLE) / 7 AS WGT,
               TIMESTAMPTZ '2023-01-02 03:04:05+00' + INTERVAL (i % {DAYS}) DAY AS LOAD_TS,
               TIME '02:00:00' + INTERVAL (i % 3600) SECOND AS LOAD_TM,
               CAST(NULL AS INTEGER) AS EMPTY_COL
        FROM range(1, 20001) r(i)""")
    con.execute("CREATE TABLE stg.PFT_CTR_CUS_DAT AS SELECT i AS STG_KEY, i AS CUS_DMS_KEY FROM range(10) r(i)")
    con.execute("CREATE TABLE EMPTY_T (A INTEGER, B VARCHAR)")
    con.execute("CREATE TABLE ONE_ROW AS SELECT 1 AS K, 'only' AS V")
    con.execute("CREATE TABLE SECRET_T AS SELECT i AS K, i * 2 AS V FROM range(50) r(i)")
    con.execute("CREATE VIEW SLS_VW AS SELECT CUS_DMS_KEY, IVC_DT_DMS_KEY, SLS_AMT FROM PFT_CTR_CUS_DAT")
    return con


# What discovery wrote, with the mistakes discovery makes, and Azure SQL's own type names.
AZURE_TYPES = {
    "DT_DMS": {"DT_DMS_KEY": "int", "DMS_DT": "date", "YR": "smallint", "MTH_NO": "tinyint", "QTR": "tinyint"},
    "CUS_DMS": {"CUS_DMS_KEY": "int", "CUS_CD": "nvarchar(10)", "CUS_NM": "nvarchar(80)", "Group": "varchar(20)",
                "ACTV_FLG": "bit", "CUS_GUID": "uniqueidentifier", "NOTE_TXT": "ntext", "Libellé": "nvarchar(40)",
                "LGCY_ID": "int"},
    "PFT_CTR_CUS_DAT": {"LN_KEY": "bigint", "DOC_NO": "varchar(20)", "LN_NO": "int", "CUS_DMS_KEY": "int",
                        "IVC_DT_DMS_KEY": "int", "FST_CUS_IVC_DT_DMS_KEY": "int", "ODD_DT_KEY": "int",
                        "Order Date": "date", "SLS_AMT": "decimal(18,2)", "CST_AMT": "money", "QTY": "int",
                        "WGT": "float", "LOAD_TS": "datetimeoffset", "LOAD_TM": "time", "EMPTY_COL": "int"},
    "EMPTY_T": {"A": "int", "B": "varchar(10)"},
    "ONE_ROW": {"K": "int", "V": "varchar(10)"},
    "SECRET_T": {"K": "int", "V": "int"},
    "SLS_VW": {"CUS_DMS_KEY": "int", "IVC_DT_DMS_KEY": "int", "SLS_AMT": "decimal(18,2)"},
}
PK = {"DT_DMS": ["DT_DMS_KEY", "STG_KEY"],                 # a staging table's key column mixed in
      "CUS_DMS": ["CUS_DMS_KEY"],
      "PFT_CTR_CUS_DAT": ["CUS_DMS_KEY", "CUS_DMS_KEY"],   # listed twice: the 8156 the server met
      "ONE_ROW": ["K"]}


def _schema(con: duckdb.DuckDBPyConnection) -> dict:
    schema: dict = {}
    for table, columns in AZURE_TYPES.items():
        rows = con.execute(f'SELECT COUNT(*) FROM main."{table}"').fetchone()[0]
        listed = [{"name": name, "type": raw, "nullable": "YES"} for name, raw in columns.items()]
        if table == "CUS_DMS":
            listed.append({"name": "CUS_CD", "type": "nvarchar(10)", "nullable": "YES"})   # listed twice
        schema[f"main.{table}"] = {"database": "", "schema": "main", "columns": listed, "row_count": rows,
                                   "pk_columns": PK.get(table, [])}
    schema["stg.PFT_CTR_CUS_DAT"] = {"database": "", "schema": "stg", "row_count": 10, "pk_columns": ["STG_KEY"],
                                    "columns": [{"name": "STG_KEY", "type": "int"}, {"name": "CUS_DMS_KEY", "type": "int"}]}
    schema["__db_fk_constraints__"] = [
        {"parent_schema": "main", "parent_table": "PFT_CTR_CUS_DAT", "parent_col": "CUS_DMS_KEY",
         "ref_schema": "main", "ref_table": "CUS_DMS", "ref_col": "CUS_DMS_KEY", "constraint_name": "FK_CUS"},
        {"parent_schema": "main", "parent_table": "PFT_CTR_CUS_DAT", "parent_col": "GONE_KEY",   # not there
         "ref_schema": "main", "ref_table": "CUS_DMS", "ref_col": "CUS_DMS_KEY", "constraint_name": "FK_GONE"}]
    return schema


class AzureSql(SpeakingWarehouse):
    """Azure SQL as the service account sees it: SECRET_T is not granted."""

    def query(self, sql, *, max_rows=None):
        if "SECRET_T" in sql.upper():
            raise RuntimeError("[42000] The SELECT permission was denied on the object 'SECRET_T' (229)")
        return super().query(sql, max_rows=max_rows)


@pytest.fixture(scope="module")
def learned():
    con = _warehouse()
    inventory = from_schema_json(_schema(con), "azure_sql")
    warehouse = AzureSql(con, "tsql", Types.of(inventory, views={"SLS_VW"}))
    lines: list[str] = []
    saved = join_step.SAMPLE_ABOVE
    join_step.SAMPLE_ABOVE = 5_000   # every big table and view is sampled, as on a real warehouse
    try:
        model = build_model(warehouse, inventory, options=BuildOptions(
            workers=2, progress=lines.append,
            profile=ProfileOptions(sample_threshold=5_000, sample_rows=4_000, exact_distinct_up_to=1_000)))
    finally:
        join_step.SAMPLE_ABOVE = saved
    return model, warehouse, lines, inventory


def test_every_query_learn_sends_is_one_azure_sql_runs(learned):
    _model, warehouse, _lines, _inventory = learned
    assert not warehouse.refused, "\n".join(warehouse.refused)
    assert len(warehouse.log) > 40


def test_discoverys_mistakes_are_corrected_and_said(learned):
    model, _warehouse, _lines, inventory = learned
    tables = {t.name: t for t in inventory.tables.values()}
    assert [c.name for c in tables["CUS_DMS"].columns].count("CUS_CD") == 1
    assert tables["PFT_CTR_CUS_DAT"].primary_key == ["CUS_DMS_KEY"] or tables["PFT_CTR_CUS_DAT"].schema == "stg"
    assert next(t for t in inventory.tables.values() if t.name == "DT_DMS").primary_key == []
    assert [fk.name for fk in inventory.foreign_keys] == ["FK_CUS"]
    assert any("DT_DMS names STG_KEY" in n for n in model.notes)
    assert any("PFT_CTR_CUS_DAT lists CUS_DMS_KEY more than once" in n for n in model.notes)


def test_the_mart_is_learned(learned):
    model, _warehouse, _lines, _inventory = learned
    key = {(t.schema_name, t.name): k for k, t in model.tables.items()}
    fact, customers, calendar = key[("main", "PFT_CTR_CUS_DAT")], key[("main", "CUS_DMS")], key[("main", "DT_DMS")]
    assert model.tables[fact].kind in ("fact", "snapshot")
    trusted = {(j.from_table, model.columns[j.from_columns[0]].name, j.to_table)
               for j in model.joins.values() if j.trust in ("declared", "verified")}
    assert (fact, "CUS_DMS_KEY", customers) in trusted
    assert calendar in model.calendars
    # The invoice date, not the customer's first invoice date: a milestone of the customer, repeated on its rows.
    default = next(r for r in model.date_roles.values() if r.table == fact and r.is_default)
    assert model.columns[default.column].name == "IVC_DT_DMS_KEY"
    assert {model.columns[m.expr.column].name for m in model.measures.values()
            if m.table == fact and getattr(m.expr, "column", None)} >= {"SLS_AMT", "CST_AMT"}


def test_what_it_could_not_read_is_left_out_and_said(learned):
    model, _warehouse, lines, _inventory = learned
    assert "SECRET_T" not in {t.name for t in model.tables.values()}
    # Nothing else: every other check ran in Azure SQL's own SQL.
    assert [n for n in model.notes if "was left out" in n] == [next(n for n in model.notes if "SECRET_T" in n)]
    assert any(n.startswith("The table SECRET_T was left out: the database refused it") for n in model.notes)
    assert any(line.startswith("Left out the table SECRET_T") for line in lines)


def test_the_view_that_cannot_be_sampled_is_read_from_its_first_rows(learned):
    model, _warehouse, lines, _inventory = learned
    assert any(line.startswith("SLS_VW cannot be sampled") for line in lines)
    assert "SLS_VW" in {t.name for t in model.tables.values()}


def test_the_progress_says_each_step(learned):
    _model, _warehouse, lines, _inventory = learned
    assert lines[0].startswith("The key recorded for") or lines[0].startswith("Reading")
    for step in ("Reading ", "Finding each table's key", "Looking for calendar", "Testing ", "Finding the dates",
                 "Sorting tables", "Checking the data"):
        assert any(line.startswith(step) for line in lines), step
    assert sum(line.startswith("Read ") for line in lines) == len(AZURE_TYPES) + 1 - 1   # all but SECRET_T
