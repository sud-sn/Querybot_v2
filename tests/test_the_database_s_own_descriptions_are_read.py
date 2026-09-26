"""
What the database says a column or table is gets read, and read as what it is.

Discovery reads the comments a DBA writes on tables and columns -- Snowflake
and Oracle COMMENT, SQL Server's MS_Description -- and writes them into the
schema markdown's Notes column and _schema.json. Then:

  * core.schema_enrichment.parse_schema_markdown, written for a four-cell row,
    went on reading cells by position after Notes was added: every documented
    column had its comment taken for its distinct values, and its real values
    ignored, so the role rules that read values read prose.
  * Nothing read the comments in _schema.json. The semantic model, the SQL
    prompt and the Semantic Layer described every column from its name alone,
    beside a sentence the warehouse's own owner had written.
  * Azure SQL discovery wrote an empty comment for every table: a table's
    MS_Description has minor_id 0, and the column query joins sys.columns.

These run the real discovery writers, parser, model builder, prompt builder
and Semantic Layer on synthetic tables, and Azure discovery against a stand-in
driver.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from unittest.mock import patch

# ── The markdown discovery writes, read back ─────────────────────────────────

COLUMNS = [
    {"COLUMN_NAME": "ORD_STS", "DATA_TYPE": "char", "IS_NULLABLE": "NO", "CHARACTER_MAXIMUM_LENGTH": 1,
     "COMMENT": "Order status | O=open, C=closed"},
    {"COLUMN_NAME": "NET_AMT", "DATA_TYPE": "decimal", "IS_NULLABLE": "YES", "NUMERIC_PRECISION": 18,
     "COMMENT": "Invoiced amount after rebates"},
]
DISTINCT = {"ORD_STS": ["O", "C"]}


def _snowflake_md():
    from core.schema import _sf_md

    return _sf_md("ORDERS", {"COMMENT": "One row per order", "TABLE_TYPE": "BASE TABLE"},
                  COLUMNS, [], DISTINCT, pk_columns=[])


def _azure_md():
    from core.schema import _az_md

    return _az_md("ORDERS", {"TABLE_TYPE": "BASE TABLE", "COMMENT": "One row per order"}, COLUMNS, [], "sales",
                  DISTINCT, database="CRM", stats_map={"NET_AMT": {"min": 0, "max": 99}}, row_count=10)


class TestTheMarkdownIsReadByItsHeader:

    def test_a_comment_is_a_comment_and_the_values_are_the_values(self):
        from core.schema_enrichment import parse_schema_markdown

        for md in (_snowflake_md(), _azure_md()):
            parsed = parse_schema_markdown(md)
            assert parsed["ORD_STS"]["comment"] == "Order status | O=open, C=closed"
            assert parsed["ORD_STS"]["distinct_values"] == "'O', 'C'"
            assert parsed["NET_AMT"]["comment"] == "Invoiced amount after rebates"
            assert parsed["NET_AMT"]["distinct_values"] == ""

    def test_the_column_statistics_table_is_not_a_second_description(self):
        from core.schema_enrichment import parse_schema_markdown

        md = _azure_md() + "\n\n## Column Statistics\n\n| Column | Min | Max |\n|---|---|---|\n| `NET_AMT` | 0 | 99 |\n"
        assert parse_schema_markdown(md)["NET_AMT"]["type"].startswith("decimal")

    def test_a_file_from_before_the_notes_column_still_reads(self):
        from core.schema_enrichment import parse_schema_markdown

        legacy = "| Column | Type | Nullable | Distinct Values |\n|---|---|---|---|\n| `FLAG` | char(1) | No | 'Y', 'N' |"
        assert parse_schema_markdown(legacy)["FLAG"] == {
            "type": "char(1)", "nullable": "No", "comment": "", "distinct_values": "'Y', 'N'"}

    def test_enrichment_carries_the_comment_and_does_not_read_prose_as_values(self):
        from core.schema_enrichment import enrich_columns

        net = next(c for c in enrich_columns(["NET_AMT"], schema_md=_snowflake_md()) if c.column == "NET_AMT")
        assert net.comment == "Invoiced amount after rebates"
        assert net.distinct_values == ""
        assert not [w for w in net.warnings if "low-cardinality" in w]


# ── The semantic model, the prompt, the Semantic Layer ───────────────────────

def _table(key, *columns, comment=""):
    return {"columns": [{"name": n, "type": t, "nullable": False, "comment": c} for n, t, c in columns],
            "pk_columns": [key], "row_count": 1000, "comment": comment, "schema": "MART", "database": "WH"}


SCHEMA = {
    "WH.MART.SALES_FACT": _table(
        "SALES_KEY", ("SALES_KEY", "bigint", ""), ("CUSTOMER_SK", "int", ""),
        ("GRS_AMT", "decimal(18,2)", "Gross invoiced amount before rebates"),
        ("NET_AMT", "decimal(18,2)", "Invoiced amount after rebates"),
        comment="One row per invoice line"),
    "WH.MART.CUSTOMER_DIM": _table("CUSTOMER_SK", ("CUSTOMER_SK", "int", ""), ("CUSTOMER_NAME", "varchar(80)", "")),
}


def _model_dir():
    from core.semantic_model import write_semantic_model

    kb = tempfile.mkdtemp()
    with open(os.path.join(kb, "_schema.json"), "w", encoding="utf-8") as handle:
        json.dump(SCHEMA, handle)
    write_semantic_model(schema_dir=kb, kb_dir=kb, account_id="acct")
    return kb


def _field(kb, column):
    from core.semantic_model import load_semantic_model

    table = next(t for t in load_semantic_model(kb)["tables"] if t["table"] == "SALES_FACT")
    return table, next(f for f in table["fields"] if f["column"] == column)


class TestTheModelKeepsTheDatabasesWords:

    def test_a_documented_column_carries_its_description_and_where_it_came_from(self):
        _, field = _field(_model_dir(), "NET_AMT")
        assert (field["description"], field["description_source"]) == \
            ("Invoiced amount after rebates", "database comment")

    def test_an_undocumented_column_claims_no_description(self):
        _, field = _field(_model_dir(), "SALES_KEY")
        assert (field["description"], field["description_source"]) == ("", "")

    def test_a_documented_table_carries_its_description(self):
        table, _ = _field(_model_dir(), "NET_AMT")
        assert table["description"] == "One row per invoice line"


class TestTheSqlPromptQuotesTheDatabaseForColumnsTheQuestionTouches:

    def test_a_question_in_the_comment_s_words_surfaces_the_documented_column(self):
        from core.semantic_model import build_runtime_semantic_context

        prompt = build_runtime_semantic_context(_model_dir(), question="invoiced amount after rebates", max_lines=40)
        assert 'MART.SALES_FACT.NET_AMT is described in the database as "Invoiced amount after rebates"' in prompt

    def test_a_question_about_something_else_does_not(self):
        from core.semantic_model import build_runtime_semantic_context

        prompt = build_runtime_semantic_context(_model_dir(), question="customer name", max_lines=40)
        assert "described in the database" not in prompt


def _semantic_layer(kb, **kwargs):
    from core.semantic_layer import build_semantic_layer_tables

    Path(kb, "SALES_FACT_kb.md").write_text(
        "# WH.MART.SALES_FACT\n\n## Columns\n\n| Column | Meaning | Use case |\n|---|---|---|\n"
        "| NET_AMT | Net sales amount (generated) | Summing sales |\n", encoding="utf-8")
    tables = build_semantic_layer_tables(kb_dir=kb, schema_dir=kb, **kwargs)
    return next(f for t in tables for f in t["fields"] if f["column"].upper() == "NET_AMT")


class TestTheSemanticLayerShowsTheDatabasesWords:

    def test_the_database_s_description_is_the_meaning_shown(self):
        field = _semantic_layer(_model_dir())
        assert (field["meaning"], field["source"]) == ("Invoiced amount after rebates", "database_comment")
        assert field["needs_context"] is False

    def test_an_admin_override_still_wins(self):
        kb = _model_dir()
        overrides = {"tables": {"WH.MART.SALES_FACT": {"fields": {"NET_AMT": {"meaning": "Net sales, CAD"}}}}}
        field = _semantic_layer(kb, field_overrides=overrides)
        assert (field["meaning"], field["source"]) == ("Net sales, CAD", "admin_override")

    def test_approved_feedback_still_wins(self):
        kb = _model_dir()
        field = _semantic_layer(kb, approved_feedback={
            ("WH.MART.SALES_FACT", "NET_AMT"): {"suggested_meaning": "Net sales as finance reports them"}})
        assert field["meaning"] == "Net sales as finance reports them"

    def test_the_reader_is_told_where_the_meaning_came_from_in_their_language(self):
        from tests.portal_render import render

        field = _semantic_layer(_model_dir())
        table = {"fqn": "WH.MART.SALES_FACT", "database": "WH", "schema": "MART", "table": "SALES_FACT",
                 "overview": "", "fields": [field], "field_count": 1, "confidence": 90, "file_stem": "SALES_FACT"}
        user = {"id": 1, "name": "Ada", "account_id": "a", "role": "analyst", "group_name": "Analysts"}
        for lang, badge in (("en", "from the database"), ("fr", "issu de la base de données")):
            html = render("portal_kb.html", path="/portal/kb", lang=lang, user=user, semantic_tables=[table],
                          visible_tables=[table], schemas=["MART"], selected_schema="MART", pending_count=0)
            badges = re.findall(r'<div class="status-badges">([\s\S]*?)</div>', html)
            assert badges and badge in badges[0], (lang, badges)


# ── Azure SQL table descriptions ─────────────────────────────────────────────

class _Cursor:
    description = []

    def __init__(self):
        self.rows, self._one = [], None

    def execute(self, sql, *params):
        sql_u = " ".join(sql.upper().split())
        self.rows, self._one = [], None
        if "DB_NAME" in sql_u:
            self._one = ("CRM",)
        elif "INFORMATION_SCHEMA.TABLES" in sql_u:
            self.rows = [("sales", "CUSTOMERS", "BASE TABLE")]
        elif "INFORMATION_SCHEMA.COLUMNS" in sql_u:
            self.rows = [("ID", "int", "NO", None, 10)]
        elif "SYS.EXTENDED_PROPERTIES" in sql_u and "MINOR_ID = 0" in sql_u:
            self._one = ("Customers who have bought at least once",)
        elif "SYS.EXTENDED_PROPERTIES" in sql_u:
            self.rows = [("ID", "Customer number")]

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self._one


class _Conn:
    def __init__(self):
        self.cur = _Cursor()

    def cursor(self):
        return self.cur

    def close(self):
        pass


class TestAzureDiscoveryReadsTheTablesDescription:

    def test_the_table_s_ms_description_reaches_the_schema_file_and_the_markdown(self):
        from core import schema

        written: dict[str, str] = {}

        def _write(path, text, encoding=None):
            written[path.name] = text
            return len(text)

        with patch.object(Path, "write_text", _write), patch("core.schema._az_connect", return_value=_Conn()):
            schema._discover_azure_sql({"database": "CRM", "schema": "sales"}, Path(tempfile.mkdtemp()),
                                       allowed={"CRM.SALES.CUSTOMERS"})
        table = json.loads(written["_schema.json"])["CRM.SALES.CUSTOMERS"]
        assert table["comment"] == "Customers who have bought at least once"
        assert table["columns"][0]["comment"] == "Customer number"
        assert "Customers who have bought at least once" in written["CUSTOMERS.md"]

    def test_a_principal_that_cannot_read_descriptions_still_discovers_the_table(self):
        from core.schema import _az_table_description

        class _Refusing:
            def execute(self, *a, **k):
                raise PermissionError("VIEW DEFINITION denied")

        assert _az_table_description(_Refusing(), "sales", "CUSTOMERS") == ""
