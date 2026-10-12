"""PostgreSQL in the old question pipeline: the SQL its deterministic builders and checks write.

Two kinds of test.

Every builder's PostgreSQL SQL is read by sqlglot's PostgreSQL dialect and carries nothing that only
Azure SQL, Snowflake or Oracle accepts (TOP, DATEADD, COUNT_BIG, N'', brackets, ROWNUM, ADD_MONTHS...).
These need no server.

And the SQL that answers questions is RUN on a real PostgreSQL server (tests/postgres_warehouse.py;
skipped without QUERYBOT_TEST_POSTGRES_URL), against the synthetic retail warehouse copied there twice:
once with its names in small letters, once with capitals (FactOrderLines.NetAmount), which PostgreSQL
finds only quoted. The names are handed to the builders the way the old pipeline carries them --
upper-cased, three-part -- with the workspace's spellings read from its own discovered _schema.json.
Each answer is compared with the same question asked of the warehouse in DuckDB.
"""

from __future__ import annotations

import asyncio
import copy
import datetime as dt
import decimal
import os
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import sqlglot

_TMP = tempfile.mkdtemp(prefix="querybot_pg_old_")
os.environ.setdefault("QUERYBOT_DB_PATH", str(Path(_TMP) / "querybot.db"))
os.environ.setdefault("DB_PATH", str(Path(_TMP) / "querybot.db"))
os.environ.setdefault("QUERYBOT_KEY_FILE", str(Path(_TMP) / "key"))

from core import postgres_names  # noqa: E402
from tests import postgres_warehouse as pg  # noqa: E402

# Constructs PostgreSQL refuses: T-SQL's, Snowflake's and Oracle's own.
FORBIDDEN = re.compile(
    r"\bTOP\s*\(|\bTOP\s+\d|\bDATEADD\b|\bDATEDIFF\b|\bGETDATE\b|\bCOUNT_BIG\b|\bNOLOCK\b|\bROWNUM\b"
    r"|\bN'|\[|\bADD_MONTHS\b|(?<![A-Z_])TRUNC\s*\(|\bDATEFROMPARTS\b|\bDATE_FROM_PARTS\b|\bTRY_CONVERT\b"
    r"|\bTRY_TO_DATE\b|\bTO_VARCHAR\b|\bNVL\b|\bIFF\b|\bISNULL\s*\(|\bSYSDATE\b|\bFETCH\s+FIRST\b"
    r"|\bARRAY_CONSTRUCT|\bMONTHS_BETWEEN\b|\bDATEPART\b|\bQUALIFY\b|\bLEN\s*\(|INTERVAL\s+'\d+\s+quarter",
    re.I,
)


def postgres_sql(sql: str) -> str:
    assert sql, "the builder wrote nothing for PostgreSQL"
    sqlglot.parse_one(sql, read="postgres")
    assert not FORBIDDEN.findall(sql), (FORBIDDEN.findall(sql), sql)
    return sql


# ══════════════════════════════════════════════════════════════════════════════
# A retail star, as the old pipeline carries it
# ══════════════════════════════════════════════════════════════════════════════

def _names(table, column, *, table_columns, known, net_amount: str):
    """``table(logical)`` and ``column("table.column")`` give the names as the pipeline holds them;
    ``net_amount`` is the measure as an admin writes it in a metric formula on PostgreSQL."""
    n = SimpleNamespace(table=table, column=column, table_columns=table_columns, known=known)
    n.fact, n.calendar, n.stores = table("order_lines"), table("calendar"), table("stores")
    n.revenue = {"name": "Revenue", "formula_type": "expression", "sql_template": f"SUM({net_amount})",
                 "base_table": n.fact}
    n.joins = [
        {"from": n.fact, "to": n.calendar, "enforcement": "required",
         "conditions": [[column("order_lines.order_date_key"), column("calendar.date_key")]]},
        {"from": n.fact, "to": n.stores, "enforcement": "required",
         "conditions": [[column("order_lines.store_id"), column("stores.store_id")]]},
    ]
    return n


def _policy(n, **overrides):
    policy = {
        "kind": "last_n", "amount": 2, "unit": "month", "requested_grain": "month",
        "fact_table": n.fact, "fact_column": n.column("order_lines.order_date_key"),
        "dimension_table": n.calendar, "dimension_key": n.column("calendar.date_key"),
        "date_column": n.column("calendar.full_date"), "date_key_type": "surrogate_fk",
        "role_alias": "order_date", "anchor_table": n.fact,
    }
    policy.update(overrides)
    return policy


def ranking_context(n, top=5, **policy):
    return {
        "question": f"top {top} stores by revenue in the last 2 months",
        "top_n": {"limit": top, "direction": "descending", "tie_policy": "exactly_n"},
        "metric_formulas": [dict(n.revenue)],
        "semantic_plan": {
            "fields": [
                {"term": "Revenue", "table": n.fact, "column": n.column("order_lines.net_amount"), "role": "measure"},
                {"term": "Store", "table": n.stores, "column": n.column("stores.store_name"),
                 "role": "display_dimension", "display_required": True},
            ],
            "joins": copy.deepcopy(n.joins), "temporal_policies": [_policy(n, **policy)],
        },
        "analytical_request_plan": {"status": "compiled", "intent": "ranking", "source_fact": n.fact,
                                    "source_facts": [n.fact], "top_n": top, "output_shape": "table"},
    }


def window_context(n, question, *, trend=False, **policy):
    plan = {"status": "compiled", "source_fact": n.fact, "source_facts": [n.fact]}
    if trend:
        plan.update(intent="trend", output_shape="time_series")
    temporal = _policy(n, **policy)
    if temporal["kind"] == "latest_n_observed":
        temporal["anchor_policy"] = "observed_periods"
        plan["temporal_operations"] = [temporal]
    return {
        "question": question,
        "metric_formulas": [dict(n.revenue)],
        "semantic_plan": {"fields": [], "joins": copy.deepcopy(n.joins), "temporal_policies": [temporal]},
        "analytical_request_plan": plan,
    }


def fewer_orders_context(n):
    context = ranking_context(n)
    context["question"] = "which stores have fewer orders in the last 2 months"
    context.pop("top_n")
    context["metric_formulas"] = []
    context["analytical_request_plan"].update(
        intent="ranking", top_n=None,
        derived_measure={"semantics": "count_distinct_business_identifier", "business_entity": "order",
                         "target_table": n.fact, "target_column": n.column("order_lines.order_number")},
        analytical_recipe={"kind": "period_over_period_entity_change", "direction": "decrease"})
    return context


def compile_window(n, context):
    from core.pipeline_helpers import compile_governed_temporal_metric_sql

    return compile_governed_temporal_metric_sql("postgres", n.known, n.known, n.table_columns, context)


def compile_comparison(n, grain="month"):
    from core.pipeline_helpers import attempt_governed_temporal_metric_repair

    context = window_context(n, f"compare revenue this {grain} versus last {grain}", kind=f"this_{grain}",
                             requested_grain=grain)
    return attempt_governed_temporal_metric_repair("", "postgres", n.known, n.known, n.table_columns, context)


# The star without a server: names as the pipeline keys them, types as PostgreSQL's catalog gives them.
_STATIC_TABLES = {"order_lines": "RETAIL.ORDER_LINES", "calendar": "RETAIL.CALENDAR",
                  "stores": "RETAIL.STORES", "regions": "RETAIL.REGIONS"}
_STATIC_COLUMNS = {
    "RETAIL.ORDER_LINES": {"ORDER_LINE_ID": "integer", "ORDER_NUMBER": "character varying",
                           "ORDER_DATE_KEY": "integer", "SHIP_DATE_KEY": "integer", "STORE_ID": "integer",
                           "NET_AMOUNT": "numeric(14,2)", "STATUS_CODE": "character varying",
                           "INVOICE_DT": "integer", "ORDERED_AT": "timestamp without time zone"},
    "RETAIL.CALENDAR": {"DATE_KEY": "integer", "FULL_DATE": "date", "YEAR": "integer", "QUARTER": "integer",
                        "MONTH_NUMBER": "integer", "MONTH_NAME": "character varying"},
    "RETAIL.STORES": {"STORE_ID": "integer", "STORE_NAME": "character varying", "REGION_ID": "integer"},
    "RETAIL.REGIONS": {"REGION_ID": "integer", "REGION_NAME": "character varying"},
}


@pytest.fixture(scope="module")
def static():
    return _names(lambda logical: _STATIC_TABLES[logical],
                  lambda logical: logical.split(".", 1)[1].upper(),
                  table_columns=_STATIC_COLUMNS, known=set(_STATIC_COLUMNS), net_amount="net_amount")


# ══════════════════════════════════════════════════════════════════════════════
# 1  How a name is written
# ══════════════════════════════════════════════════════════════════════════════

class TestANameIsWrittenAsPostgresKeepsIt:

    @pytest.mark.parametrize("carried, written", [
        ("PROD.SALES.ORDER_LINES", "sales.order_lines"),       # no database part; folded case
        ("SALES.FactOrderLines", 'sales."FactOrderLines"'),     # its own spelling, quoted
        ('"Sales"."Order Lines"', '"Sales"."Order Lines"'),
        ("[dbo].[Orders]", 'dbo."Orders"'),
        ("ORDER_LINES", "order_lines"),
    ])
    def test_a_table(self, carried, written):
        assert postgres_names.table(carried) == written

    @pytest.mark.parametrize("carried, written", [
        ("NET_AMOUNT", "net_amount"), ("NetAmount", '"NetAmount"'), ("YEAR", '"year"'), ("year", '"year"'),
    ])
    def test_a_column(self, carried, written):
        assert postgres_names.ident(carried) == written

    def test_an_upper_cased_name_takes_the_spelling_discovery_recorded(self):
        spellings = postgres_names.spellings_from_master({
            "PROD.SALES.FactOrderLines": {"schema": "Sales", "columns": [{"name": "OrderDateKey"}]},
            "__db_fk_constraints__": [],
        })
        with postgres_names.using(spellings):
            assert postgres_names.table("PROD.SALES.FACTORDERLINES") == '"Sales"."FactOrderLines"'
            assert postgres_names.ident("ORDERDATEKEY") == '"OrderDateKey"'
            assert postgres_names.table("FACTORDERLINES") == '"Sales"."FactOrderLines"'
        assert postgres_names.ident("ORDERDATEKEY") == "orderdatekey"   # nothing active: folded


# ══════════════════════════════════════════════════════════════════════════════
# 2  Every builder writes PostgreSQL (no server)
# ══════════════════════════════════════════════════════════════════════════════

class TestEveryBuilderWritesPostgresSQL:

    def test_a_ranking_ends_with_limit(self, static):
        from core.pipeline_helpers import compile_governed_temporal_metric_sql

        sql = postgres_sql(compile_governed_temporal_metric_sql(
            "postgres", static.known, static.known, static.table_columns, ranking_context(static)))
        assert sql.rstrip().endswith("LIMIT 5")
        assert "INTERVAL '2 month'" in sql

    @pytest.mark.parametrize("question, policy", [
        ("revenue for the last 2 months", {"kind": "last_n", "amount": 2, "unit": "month"}),
        ("revenue for the last 3 weeks", {"kind": "last_n", "amount": 3, "unit": "week"}),
        ("revenue for the last 2 quarters", {"kind": "last_n", "amount": 2, "unit": "quarter"}),
        ("revenue trend for the last 7 days", {"kind": "last_n", "amount": 7, "unit": "day", "requested_grain": "day"}),
        ("revenue today", {"kind": "today", "amount": 1, "unit": "day"}),
        ("revenue yesterday", {"kind": "yesterday", "amount": 1, "unit": "day"}),
        ("revenue this month", {"kind": "this_month", "amount": 1, "unit": "month"}),
        ("revenue for the previous month", {"kind": "previous_month", "amount": 1, "unit": "month"}),
        ("revenue this quarter", {"kind": "this_quarter", "amount": 1, "unit": "quarter"}),
        ("revenue for the previous quarter", {"kind": "previous_quarter", "amount": 1, "unit": "quarter"}),
        ("revenue this year", {"kind": "this_year", "amount": 1, "unit": "year"}),
        ("revenue for the previous year", {"kind": "previous_year", "amount": 1, "unit": "year"}),
    ])
    @pytest.mark.parametrize("anchored", [False, True], ids=["anchor-read", "anchor-resolved"])
    def test_a_window(self, static, question, policy, anchored):
        context = window_context(static, question, trend="trend" in question, **policy)
        if anchored:
            context["resolved_date_anchor"] = {"value": "2026-06-14", "fact_table": static.fact,
                                               "fact_column": static.column("order_lines.order_date_key")}
        sql = postgres_sql(compile_window(static, context))
        assert ("2026-06-14" in sql) is anchored

    def test_the_latest_observed_days(self, static):
        sql = postgres_sql(compile_window(static, window_context(
            static, "show revenue for the latest 2 data days", trend=True,
            kind="latest_n_observed", amount=2, unit="day", requested_grain="day")))
        assert "LIMIT 2" in sql

    @pytest.mark.parametrize("key_type, column", [("yyyymmdd_integer", "INVOICE_DT"), ("native_date", "ORDERED_AT")])
    def test_a_date_on_the_fact_itself(self, static, key_type, column):
        sql = postgres_sql(compile_window(static, window_context(
            static, "revenue trend for the last 2 months", trend=True,
            date_key_type=key_type, fact_column=column, date_column=column, dimension_table="", dimension_key="")))
        assert ("MAKE_DATE" in sql) is (key_type == "yyyymmdd_integer")

    @pytest.mark.parametrize("grain", ["week", "month", "quarter", "year"])
    def test_this_period_against_the_last(self, static, grain):
        sql = postgres_sql(compile_comparison(static, grain))
        assert f"DATE_TRUNC('{grain}'" in sql

    def test_the_stores_with_fewer_orders_than_the_period_before(self, static):
        from core.pipeline_helpers import compile_governed_temporal_metric_sql

        sql = postgres_sql(compile_governed_temporal_metric_sql(
            "postgres", static.known, static.known, static.table_columns, fewer_orders_context(static)))
        assert "INTERVAL '4 month'" in sql

    @pytest.mark.parametrize("grain", ["day", "week", "month", "quarter", "year"])
    @pytest.mark.parametrize("calendar", [False, True], ids=["from-the-date", "from-the-calendar"])
    def test_a_period_bucket(self, grain, calendar):
        from core.contextual_dates import format_period_bucket_expression

        attributes = {"year": "YEAR", "quarter": "QUARTER", "month_number": "MONTH_NUMBER"} if calendar else {}
        bucket = format_period_bucket_expression("d.full_date", grain, "postgres", role_alias="d",
                                                 calendar_attributes=attributes)
        postgres_sql(f"SELECT {bucket} AS period FROM sales.calendar AS d GROUP BY {bucket}")
        if not calendar and grain != "day":
            assert bucket == f"CAST(DATE_TRUNC('{grain}', d.full_date) AS date)"

    @pytest.mark.parametrize("key_type", ["yyyymmdd_integer", "yyyymm_integer"])
    def test_a_date_kept_as_a_number(self, key_type):
        from core.contextual_dates import format_date_value_expression

        expression = format_date_value_expression("f", '"OrderDateKey"', key_type, "postgres")
        postgres_sql(f"SELECT {expression} FROM sales.f")
        assert "TO_DATE" not in expression

    def test_the_anchor_handed_to_the_model(self, static):
        from core.contextual_dates import format_required_anchor

        sql = postgres_sql("SELECT " + format_required_anchor(_policy(static), "postgres"))
        assert "retail.calendar" in sql and "RETAIL" not in sql

    def test_the_anchor_probes(self, static):
        from core.date_anchor import build_anchor_probe_sql, build_cheap_anchor_probe_sql, build_key_order_check_sql

        for build in (build_anchor_probe_sql, build_cheap_anchor_probe_sql, build_key_order_check_sql):
            postgres_sql(build(_policy(static), "postgres"))

    def test_the_last_date_inside_a_period(self, static):
        from core.period_siblings import build_period_rows_probe_sql

        policy = _policy(static)
        policy["date_value_column"] = policy["date_column"]
        sql = postgres_sql(build_period_rows_probe_sql(policy, "2026-05-01", "2026-06-01", "postgres",
                                                       measures=("NET_AMOUNT",)))
        assert "DATE '2026-05-01'" in sql

    def test_two_named_periods(self, static):
        from core.multi_period import _period_predicate

        policy = {"role_alias": "order_date", "date_value_column": "FULL_DATE", "date_key_type": "surrogate_fk",
                  "calendar_attributes": {"year": "YEAR", "month_number": "MONTH_NUMBER", "quarter": "QUARTER"}}
        for label in ("March 2026", "Q1 2026", "2025"):
            predicate = _period_predicate(label, policy, "month", "postgres")
            postgres_sql(f"SELECT 1 FROM retail.calendar AS order_date WHERE {predicate}")
        bare = {"table": "RETAIL.CALENDAR", "column": "FULL_DATE", "date_key_type": "native_date"}
        postgres_sql("SELECT 1 FROM retail.calendar WHERE " + _period_predicate("March 2026", bare, "month", "postgres"))

    def test_the_metric_probes(self):
        from core.metric_dryrun import _fqn_to_sql, _table_sql

        assert _fqn_to_sql("PROD.RETAIL.ORDER_LINES", "postgres") == "retail.order_lines"
        assert _table_sql("PROD.Retail.FactOrderLines", "postgres") == '"Retail"."FactOrderLines"'

    @pytest.mark.parametrize("unit", ["day", "month"])
    @pytest.mark.parametrize("missing_to", ["null", "zero", "today", "today_if_overdue_else_zero"])
    def test_a_gap_between_two_dates(self, unit, missing_to):
        from core.metric_builder import compile_metric_builder_config

        compiled = compile_metric_builder_config({
            "enabled": True, "mode": "date_gap", "aggregation": "AVG", "unit": unit, "missing_to": missing_to,
            "date_column": "FullDate",
            "from_join": {"alias": "ordered", "table": "retail.DimCalendar", "from_column": "OrderDateKey",
                          "to_column": "DateKey"},
            "to_join": {"alias": "shipped", "table": "retail.DimCalendar", "from_column": "ShipDateKey",
                        "to_column": "DateKey"},
        }, "postgres")
        postgres_sql(f"SELECT {compiled.formula} FROM retail.f")
        assert 'ordered."FullDate"' in compiled.formula
        assert ("CURRENT_DATE" in compiled.formula) is missing_to.startswith("today")

    def test_the_join_checks(self):
        from core.relationship_validator import build_match_exists_sql, build_probe_sql, build_profile_sql

        rel = {"from_column": "StoreKey", "to_column": "StoreKey", "join_type": "LEFT"}
        left = {"schema_name": "RETAIL", "table_name": "FactOrderLines"}
        right = {"schema_name": "RETAIL", "table_name": "DimStores"}
        for build in (build_probe_sql, build_profile_sql, build_match_exists_sql):
            sql = postgres_sql(build("postgres", rel, left, right))
            assert 'retail."FactOrderLines"' in sql

    def test_the_unknown_member_probe(self):
        from core.unknown_members import build_probe_sql

        sql = postgres_sql(build_probe_sql("postgres", {"schema_name": "RETAIL", "table_name": "DimStores"},
                                           "StoreKey", ["StoreName"]))
        assert sql.startswith('SELECT "StoreKey", "StoreName" FROM retail."DimStores"')

    def test_the_value_index_read(self):
        from core.value_index import _distinct_sql

        sql = postgres_sql(_distinct_sql("postgres", "warehouse", "retail", "DimStores", "StoreName", 5000))
        assert '"retail"."DimStores"' in sql and sql.endswith("LIMIT 5001")

    def test_a_listing_a_drill_and_a_regrain(self, static):
        from core.drill_dimension import build_deterministic_drill_sql
        from core.listing import listing_sql
        from core.result_regrain import build_regrain_sql

        postgres_sql(listing_sql({"table": "PROD.RETAIL.STORES", "column": "STORE_NAME", "term": "Store"},
                                 "postgres", leave_out={"keys": ["-1"], "key_column": "STORE_ID"}))
        drilled = build_deterministic_drill_sql(
            "SELECT SUM(o.net_amount) AS revenue FROM retail.order_lines AS o",
            {"name": "Store", "display_table": "PROD.RETAIL.STORES", "display_column": "STORE_NAME",
             "source_table": "PROD.RETAIL.ORDER_LINES", "source_key_column": "STORE_ID"}, "postgres")
        assert "retail.stores AS qb_dim" in postgres_sql(drilled)
        regrained, why = build_regrain_sql(
            "SELECT SUM(o.net_amount) AS revenue FROM retail.order_lines AS o "
            "JOIN retail.calendar AS c ON o.order_date_key = c.date_key",
            {**_policy(static), "temporal_grain": "day"}, "month", "postgres")
        assert "DATE_TRUNC('MONTH', c.full_date)" in postgres_sql(regrained), why

    def test_the_hints_given_to_the_model(self):
        from core.anomaly_detection import build_anomaly_sql_hint
        from core.fiscal_calendar import build_fiscal_sql_hint
        from core.window_analytics import WindowIntent, build_window_sql_hint

        fiscal = build_fiscal_sql_hint("revenue for FY2025", 4, "postgres")
        expressions = re.findall(r"^\s+(.+?)\s+AS fiscal_(?:year|quarter)", fiscal, re.M)
        assert len(expressions) == 2
        for expression in expressions:
            postgres_sql(f"SELECT {expression} FROM t")
        for kind in ("rolling_avg", "running_total", "rank_in_group"):
            hint = build_window_sql_hint(WindowIntent(type=kind), "postgres")
            pattern = re.search(r"^\s+(.+ OVER \(.+\).*)$", hint, re.M).group(1)
            postgres_sql(f"SELECT {pattern} FROM t")
        anomaly = build_anomaly_sql_hint("postgres")
        postgres_sql(anomaly[anomaly.index("WITH stats"):anomaly.index("Replace 'metric_col'")])

    def test_what_a_row_count_and_a_report_read(self):
        from core.pipeline_helpers import _date_literal, _member_literal, _quote_table_for_count
        from core.report_engine import _apply_latest_date_filter

        assert _quote_table_for_count("PROD.RETAIL.ORDER_LINES", "postgres") == "retail.order_lines"
        assert _date_literal("2026-01-31", "postgres") == "DATE '2026-01-31'"
        assert _member_literal("Men's Jacket", "postgres") == "'Men''s Jacket'"
        postgres_sql(_apply_latest_date_filter("SELECT SUM(net_amount) AS revenue FROM retail.order_lines",
                                               "PROD.RETAIL.ORDER_LINES", "ORDER_DATE_KEY", "postgres"))

    def test_an_example_is_checked_with_a_row_cap(self):
        from core.examples import _add_row_cap

        assert _add_row_cap("SELECT a FROM t", "postgres") == "SELECT a FROM t LIMIT 1"
        assert _add_row_cap("SELECT a FROM t LIMIT 5", "postgres") == "SELECT a FROM t LIMIT 5"

    def test_the_metric_function_list(self):
        from core.metric_validator import validate_metric

        ok = validate_metric({"name": "Lead time", "formula_type": "expression",
                              "sql_template": "AVG(CAST(shipped - ordered AS numeric)) + COUNT(*) * 0"},
                             db_type="postgres")
        assert ok.valid, ok.errors
        refused = validate_metric({"name": "Lead time", "formula_type": "expression",
                                   "sql_template": "AVG(DATEDIFF(day, ordered, shipped))"}, db_type="postgres")
        assert any("DATEDIFF" in error for error in refused.errors)


class TestAYearRowIsNotReadAsJanuary:
    """PostgreSQL's TO_DATE reads month 00 as January: TO_DATE('202400', 'YYYYMM') is 2024-01-01. A
    window on a period key decoded that way counts each year's total as its January, so on
    PostgreSQL the validator does not take it for a decode that leaves year rows out."""

    POLICY = [{"fact_table": "RETAIL.SALES_TARGETS", "fact_column": "PERIOD_KEY"}]

    def _errors(self, where, db_type):
        from core.validator import _DIALECT, _period_row_errors

        tree = sqlglot.parse_one(f"SELECT SUM(t.target_amount) FROM retail.sales_targets AS t WHERE {where}",
                                 read=_DIALECT[db_type])
        return _period_row_errors(tree, self.POLICY, db_type)

    def test_to_date_is_not_enough_on_postgres(self):
        where = "TO_DATE(CAST(t.period_key AS TEXT), 'YYYYMM') >= DATE '2025-01-01'"
        assert self._errors(where, "postgres")
        assert not self._errors(where, "snowflake")      # TO_DATE raises there: unchanged

    def test_the_decode_the_pipeline_writes_is(self):
        from core.contextual_dates import format_date_value_expression

        decoded = format_date_value_expression("t", "period_key", "yyyymm_integer", "postgres")
        assert not self._errors(f"{decoded} >= DATE '2025-01-01'", "postgres")


class TestLogsAreNotStoredInPostgres:
    """QueryBot opens every PostgreSQL session read only: its logs are stored elsewhere, and the
    connection form neither offers nor keeps the option."""

    def test_the_export_refuses_with_the_reason(self):
        from core.log_export import POSTGRES_NOT_SUPPORTED, _connect_for_export, log_export_supported

        assert not log_export_supported("postgres")
        assert log_export_supported("azure_sql") and log_export_supported("snowflake") and log_export_supported("oracle")
        with pytest.raises(ValueError) as refused:
            _connect_for_export({"db_type": "postgres", "credentials": {"host": "h", "log_export_enabled": "1"}})
        assert str(refused.value) == POSTGRES_NOT_SUPPORTED

    def test_the_form_does_not_keep_it(self):
        from admin.routes import _db_credentials_from_form

        form = {"db_type": "postgres", "name": "Warehouse", "pg_host": "h", "pg_database": "d", "pg_user": "u",
                "pg_password": "p", "log_export_enabled": "1", "log_schema": "LOGS"}
        _type, _name, _id, creds = _db_credentials_from_form(form)
        assert creds["log_export_enabled"] == "0"
        _type, _name, _id, creds = _db_credentials_from_form({**form, "db_type": "azure_sql", "az_server": "s"})
        assert creds["log_export_enabled"] == "1"


# ══════════════════════════════════════════════════════════════════════════════
# 3  Run on a real PostgreSQL server, compared with DuckDB
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def creds():
    return pg.credentials()


@pytest.fixture(scope="module", params=["descriptive", "pascal"])
def warehouse(request, creds, tmp_path_factory):
    """The retail warehouse in PostgreSQL, discovered as QueryBot discovers it, and its names as the
    old pipeline carries them: upper-cased, DATABASE.SCHEMA.TABLE."""
    from evals.core2 import domains
    from evals.core2.framework import materialize
    from core.schema import discover_and_write, load_known_tables, load_schema_columns

    built = materialize(domains.build("retail"), request.param)
    with pg.copied(built, creds) as schema:
        cfg = {**creds, "schemas": schema}
        out = tmp_path_factory.mktemp(f"schema_{request.param}")
        discover_and_write(cfg, "postgres", str(out))
        database = creds["database"].upper()
        n = _names(lambda logical: f"{database}.{schema}.{built.t(logical)}".upper(),
                   lambda logical: built.c(logical)[1].upper(),
                   table_columns=load_schema_columns(str(out)), known=load_known_tables(str(out)),
                   net_amount=postgres_names.ident(built.c("order_lines.net_amount")[1]))
        yield SimpleNamespace(built=built, creds=cfg, schema=schema, schema_dir=str(out), style=request.param,
                              names=n, spellings=postgres_names.spellings_from_schema(str(out)))


def _value(value):
    if isinstance(value, (decimal.Decimal, float)):
        return round(float(value), 2)
    if isinstance(value, dt.datetime):
        return value.date()
    return value


def on_postgres(w, sql):
    from core.schema import run_query

    return [tuple(_value(v) for v in row.values()) for row in run_query(w.creds, "postgres", sql, max_rows=5000)]


def on_duckdb(w, sql):
    return [tuple(_value(v) for v in row) for row in w.built.con.execute(sql).fetchall()]


def q(w, logical: str) -> str:
    """A physical name for the DuckDB reference: "order_lines" or "order_lines.net_amount"."""
    if "." in logical:
        return '"' + w.built.c(logical)[1] + '"'
    return '"' + w.built.t(logical) + '"'


def _star(w, *, stores: bool = False) -> str:
    """The fact on its order date, and on its store when the compiled SQL joins it too."""
    star = (f"FROM {q(w, 'order_lines')} o JOIN {q(w, 'calendar')} c "
            f"ON o.{q(w, 'order_lines.order_date_key')} = c.{q(w, 'calendar.date_key')}")
    if stores:
        star += f" JOIN {q(w, 'stores')} s ON o.{q(w, 'order_lines.store_id')} = s.{q(w, 'stores.store_id')}"
    return star


def _anchor(w, *, stores: bool = False) -> dt.date:
    return on_duckdb(w, f"SELECT MAX(c.{q(w, 'calendar.full_date')}) {_star(w, stores=stores)}")[0][0]


def _revenue_between(w, start: dt.date, end: dt.date, *, group: str = "") -> list:
    """Revenue for full_date in [start, end], optionally by a grouping expression."""
    select = f"{group}, " if group else ""
    tail = f" GROUP BY {group} ORDER BY {group}" if group else ""
    return on_duckdb(w, f"SELECT {select}SUM(o.{q(w, 'order_lines.net_amount')}) {_star(w)} "
                        f"WHERE c.{q(w, 'calendar.full_date')} BETWEEN DATE '{start}' AND DATE '{end}'{tail}")


def _months_back(day: dt.date, months: int) -> dt.date:
    month = day.month - months
    year = day.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    import calendar

    return dt.date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


class TestOnAPostgresServer:

    def test_the_spellings_are_the_workspaces_own(self, warehouse):
        """What handle_query activates, read from the _schema.json discovery wrote: nothing handed over."""
        token = postgres_names.activate_for_workspace({"db_type": "postgres"}, {"schema_dir": warehouse.schema_dir})
        try:
            written = postgres_names.table(warehouse.names.fact)
        finally:
            postgres_names.deactivate(token)
        assert written == f"{warehouse.schema}.{postgres_names.ident(warehouse.built.t('order_lines'))}"
        token = postgres_names.activate_for_workspace({"db_type": "azure_sql"}, {"schema_dir": warehouse.schema_dir})
        try:
            assert postgres_names.ident(warehouse.names.column("order_lines.net_amount")) == \
                warehouse.names.column("order_lines.net_amount").lower()
        finally:
            postgres_names.deactivate(token)

    def test_the_top_stores_by_revenue(self, warehouse):
        w, n = warehouse, warehouse.names
        from core.pipeline_helpers import compile_governed_temporal_metric_sql

        with postgres_names.using(w.spellings):
            sql = postgres_sql(compile_governed_temporal_metric_sql(
                "postgres", n.known, n.known, n.table_columns, ranking_context(n)))
        anchor = _anchor(w, stores=True)
        expected = on_duckdb(w, f"SELECT s.{q(w, 'stores.store_name')}, SUM(o.{q(w, 'order_lines.net_amount')}) AS r "
                                f"{_star(w, stores=True)} WHERE c.{q(w, 'calendar.full_date')} > DATE '{_months_back(anchor, 2)}' "
                                f"AND c.{q(w, 'calendar.full_date')} <= DATE '{anchor}' GROUP BY 1 ORDER BY r DESC LIMIT 5")
        assert on_postgres(w, sql) == expected

    def test_the_stores_with_fewer_orders_than_the_period_before(self, warehouse):
        w, n = warehouse, warehouse.names
        from core.pipeline_helpers import compile_governed_temporal_metric_sql

        with postgres_names.using(w.spellings):
            sql = postgres_sql(compile_governed_temporal_metric_sql(
                "postgres", n.known, n.known, n.table_columns, fewer_orders_context(n)))
        anchor = _anchor(w, stores=True)
        full_date, order = f"c.{q(w, 'calendar.full_date')}", f"o.{q(w, 'order_lines.order_number')}"
        current, prior = _months_back(anchor, 2), _months_back(anchor, 4)
        expected = on_duckdb(w, f"SELECT * FROM (SELECT s.{q(w, 'stores.store_name')}, "
                                f"COUNT(DISTINCT CASE WHEN {full_date} > DATE '{current}' THEN {order} END) AS now_, "
                                f"COUNT(DISTINCT CASE WHEN {full_date} <= DATE '{current}' THEN {order} END) AS before_ "
                                f"{_star(w, stores=True)} WHERE {full_date} > DATE '{prior}' AND {full_date} <= DATE '{anchor}' "
                                f"GROUP BY 1) x WHERE now_ < before_")
        rows = on_postgres(w, sql)
        assert sorted(row[:3] for row in rows) == sorted(expected) and expected
        assert all(row[3] == row[1] - row[2] for row in rows)

    @pytest.mark.parametrize("anchored", [False, True], ids=["anchor-read", "anchor-resolved"])
    def test_a_monthly_trend_over_the_last_three_months(self, warehouse, anchored):
        w, n = warehouse, warehouse.names
        anchor = _anchor(w)
        context = window_context(n, "revenue trend for the last 3 months", trend=True, amount=3)
        if anchored:
            context["resolved_date_anchor"] = {"value": anchor.isoformat(), "fact_table": n.fact,
                                               "fact_column": n.column("order_lines.order_date_key")}
        with postgres_names.using(w.spellings):
            sql = postgres_sql(compile_window(n, context))
        full_date = f"c.{q(w, 'calendar.full_date')}"
        expected = on_duckdb(w, f"SELECT CAST(DATE_TRUNC('month', {full_date}) AS DATE), "
                                f"SUM(o.{q(w, 'order_lines.net_amount')}) {_star(w)} "
                                f"WHERE {full_date} > DATE '{_months_back(anchor, 3)}' AND {full_date} <= DATE '{anchor}' "
                                f"GROUP BY 1 ORDER BY 1")
        assert on_postgres(w, sql) == expected

    def test_the_latest_observed_days(self, warehouse):
        w, n = warehouse, warehouse.names
        with postgres_names.using(w.spellings):
            sql = postgres_sql(compile_window(n, window_context(
                n, "show revenue for the latest 3 data days", trend=True,
                kind="latest_n_observed", amount=3, unit="day", requested_grain="day")))
        full_date = f"c.{q(w, 'calendar.full_date')}"
        expected = on_duckdb(w, f"SELECT {full_date}, SUM(o.{q(w, 'order_lines.net_amount')}) {_star(w)} "
                                f"WHERE {full_date} IN (SELECT DISTINCT {full_date} {_star(w)} "
                                f"ORDER BY 1 DESC LIMIT 3) GROUP BY 1 ORDER BY 1")
        assert on_postgres(w, sql) == expected and len(expected) == 3

    @pytest.mark.parametrize("kind", ["this_month", "previous_month", "this_quarter", "previous_quarter",
                                      "this_year", "previous_year", "today", "yesterday"])
    def test_a_calendar_period(self, warehouse, kind):
        w, n = warehouse, warehouse.names
        unit = kind.split("_")[-1] if kind not in {"today", "yesterday"} else "day"
        with postgres_names.using(w.spellings):
            sql = postgres_sql(compile_window(n, window_context(n, f"revenue {kind.replace('_', ' ')}",
                                                                kind=kind, amount=1, unit=unit)))
        anchor = _anchor(w)
        quarter_start = dt.date(anchor.year, 3 * ((anchor.month - 1) // 3) + 1, 1)
        start, end = {
            "this_month": (anchor.replace(day=1), anchor),
            "previous_month": (_months_back(anchor.replace(day=1), 1), anchor.replace(day=1) - dt.timedelta(days=1)),
            "this_quarter": (quarter_start, anchor),
            "previous_quarter": (_months_back(quarter_start, 3), quarter_start - dt.timedelta(days=1)),
            "this_year": (dt.date(anchor.year, 1, 1), anchor),
            "previous_year": (dt.date(anchor.year - 1, 1, 1), dt.date(anchor.year - 1, 12, 31)),
            "today": (anchor, anchor),
            "yesterday": (anchor - dt.timedelta(days=1), anchor - dt.timedelta(days=1)),
        }[kind]
        assert on_postgres(w, sql) == _revenue_between(w, start, end)

    def test_this_month_against_last_month(self, warehouse):
        w, n = warehouse, warehouse.names
        with postgres_names.using(w.spellings):
            sql = postgres_sql(compile_comparison(n, "month"))
        anchor = _anchor(w)
        rows = on_postgres(w, sql)
        months = _revenue_between(w, _months_back(anchor.replace(day=1), 1), _months_back(anchor.replace(day=1), -1)
                                  - dt.timedelta(days=1), group=f"CAST(DATE_TRUNC('month', c.{q(w, 'calendar.full_date')}) AS DATE)")
        assert [row[:2] for row in rows] == months
        assert rows[1][2] == months[0][1]
        assert rows[1][3] == pytest.approx(months[1][1] - months[0][1], abs=0.01)

    def test_a_date_kept_as_a_yyyymmdd_number_on_the_fact(self, warehouse):
        """The ship date key holds -1 for an order not shipped: TO_DATE stops the query on it."""
        from core.contextual_dates import format_date_value_expression, format_period_bucket_expression

        w, n = warehouse, warehouse.names
        with postgres_names.using(w.spellings):
            shipped_pg = format_date_value_expression(
                "o", postgres_names.ident(n.column("order_lines.ship_date_key")), "yyyymmdd_integer", "postgres")
            month = format_period_bucket_expression(shipped_pg, "month", "postgres")
            net = postgres_names.ident(n.column("order_lines.net_amount"))
            sql = postgres_sql(f"SELECT {month}, SUM(o.{net}) FROM {postgres_names.table(n.fact)} o "
                               f"WHERE {shipped_pg} IS NOT NULL GROUP BY 1 ORDER BY 1")
        shipped = f"TRY_STRPTIME(CAST(o.{q(w, 'order_lines.ship_date_key')} AS VARCHAR), '%Y%m%d')::DATE"
        expected = on_duckdb(w, f"SELECT CAST(DATE_TRUNC('month', {shipped}) AS DATE), "
                                f"SUM(o.{q(w, 'order_lines.net_amount')}) FROM {q(w, 'order_lines')} o "
                                f"WHERE {shipped} IS NOT NULL GROUP BY 1 ORDER BY 1")
        assert on_postgres(w, sql) == expected

    def test_a_window_on_that_number(self, warehouse):
        """Compiled where its name does not mark it a calendar's surrogate key, refused where it does --
        as on Azure SQL, whose decode the validator reads the same way."""
        from core.pipeline_helpers import compile_governed_temporal_metric_sql

        w, n = warehouse, warehouse.names
        column = n.column("order_lines.ship_date_key")
        context = window_context(n, "revenue trend for the last 2 months", trend=True, date_key_type="yyyymmdd_integer",
                                 fact_column=column, date_column=column, dimension_table="", dimension_key="",
                                 role_alias="ship_date")
        with postgres_names.using(w.spellings):
            sql = compile_window(n, copy.deepcopy(context))
        on_azure = compile_governed_temporal_metric_sql("azure_sql", n.known, n.known, n.table_columns, context)
        assert bool(sql) is bool(on_azure) is (w.style == "pascal")
        if not sql:
            return
        shipped = f"TRY_STRPTIME(CAST(o.{q(w, 'order_lines.ship_date_key')} AS VARCHAR), '%Y%m%d')::DATE"
        anchor = on_duckdb(w, f"SELECT MAX({shipped}) FROM {q(w, 'order_lines')} o")[0][0]
        expected = on_duckdb(w, f"SELECT CAST(DATE_TRUNC('month', {shipped}) AS DATE), "
                                f"SUM(o.{q(w, 'order_lines.net_amount')}) FROM {q(w, 'order_lines')} o "
                                f"WHERE {shipped} > DATE '{_months_back(anchor, 2)}' AND {shipped} <= DATE '{anchor}' "
                                f"GROUP BY 1 ORDER BY 1")
        assert on_postgres(w, postgres_sql(sql)) == expected

    def test_a_key_that_is_no_date_decodes_to_nothing(self, warehouse):
        from core.contextual_dates import format_date_value_expression

        days = format_date_value_expression("v", "k", "yyyymmdd_integer", "postgres")
        months = format_date_value_expression("v", "k", "yyyymm_integer", "postgres")
        rows = on_postgres(warehouse, f"SELECT v.k, {days} AS days, {months} AS months FROM (VALUES (0), (-1), (777), (20240100), "
                                      f"(20230231), (20240229), (19000101), (202400), (202402), (202413)) AS v(k) "
                                      f"ORDER BY v.k")
        assert rows == [(-1, None, None), (0, None, None), (777, None, None), (202400, None, None),
                        (202402, None, dt.date(2024, 2, 1)), (202413, None, None),
                        (19000101, dt.date(1900, 1, 1), None), (20230231, None, None),
                        (20240100, None, None), (20240229, dt.date(2024, 2, 29), None)]

    @pytest.mark.parametrize("grain", ["week", "month", "quarter", "year"])
    def test_a_period_bucket(self, warehouse, grain):
        from core.contextual_dates import format_period_bucket_expression

        w = warehouse
        attributes = {"year": w.built.c("calendar.year")[1], "quarter": w.built.c("calendar.quarter")[1],
                      "month_number": w.built.c("calendar.month_number")[1]}
        with postgres_names.using(w.spellings):
            from_date = format_period_bucket_expression(
                f"c.{postgres_names.ident(w.built.c('calendar.full_date')[1])}", grain, "postgres")
            from_calendar = format_period_bucket_expression(
                "c.unused", grain, "postgres", role_alias="c",
                calendar_attributes={k: v.upper() for k, v in attributes.items()})
            table = postgres_names.table(w.names.calendar)
        full_date = f"c.{q(w, 'calendar.full_date')}"
        window = f"BETWEEN DATE '2025-01-01' AND DATE '2025-12-31'"
        expected = on_duckdb(w, f"SELECT CAST(DATE_TRUNC('{grain}', {full_date}) AS DATE), COUNT(*) "
                                f"FROM {q(w, 'calendar')} c WHERE {full_date} {window} GROUP BY 1 ORDER BY 1")
        for bucket in (from_date, from_calendar):
            if grain == "week" and bucket is from_calendar:
                continue   # a calendar keeps no week to build one from: it is the date's
            got = on_postgres(w, f"SELECT {bucket}, COUNT(*) FROM {table} c WHERE "
                                 f"c.{postgres_names.ident(w.built.c('calendar.full_date')[1])} {window} GROUP BY 1 ORDER BY 1")
            if grain == "year" and bucket is from_calendar:
                expected_years = [(row[0].year, row[1]) for row in expected]
                assert got == expected_years
            else:
                assert got == expected, (grain, bucket)

    def test_the_anchor_probes_and_the_anchor_handed_to_the_model(self, warehouse):
        from core.contextual_dates import format_required_anchor
        from core.date_anchor import build_anchor_probe_sql, build_cheap_anchor_probe_sql, build_key_order_check_sql

        w, n = warehouse, warehouse.names
        policy = _policy(n)
        with postgres_names.using(w.spellings):
            probes = [build_anchor_probe_sql(policy, "postgres"), build_cheap_anchor_probe_sql(policy, "postgres"),
                      "SELECT " + format_required_anchor(policy, "postgres")]
            ordering = build_key_order_check_sql(policy, "postgres")
        anchor = on_duckdb(w, f"SELECT MAX(c.{q(w, 'calendar.full_date')}) FROM {q(w, 'order_lines')} o "
                              f"JOIN {q(w, 'calendar')} c ON o.{q(w, 'order_lines.order_date_key')} = "
                              f"c.{q(w, 'calendar.date_key')}")
        for probe in probes:
            assert on_postgres(w, postgres_sql(probe)) == anchor, probe
        out_of_order = on_duckdb(w, f"SELECT COUNT(*) FROM (SELECT {q(w, 'calendar.full_date')} AS d, "
                                    f"LAG({q(w, 'calendar.full_date')}) OVER (ORDER BY {q(w, 'calendar.date_key')}) "
                                    f"AS p FROM {q(w, 'calendar')}) x WHERE d IS NULL OR d < p")
        assert on_postgres(w, postgres_sql(ordering)) == out_of_order

    def test_the_last_date_with_revenue_inside_a_period(self, warehouse):
        from core.period_siblings import build_period_rows_probe_sql

        w, n = warehouse, warehouse.names
        policy = _policy(n)
        policy["date_value_column"] = policy["date_column"]
        with postgres_names.using(w.spellings):
            sql = postgres_sql(build_period_rows_probe_sql(policy, "2025-05-01", "2025-06-01", "postgres",
                                                           measures=(n.column("order_lines.net_amount"),)))
        assert on_postgres(w, sql) == on_duckdb(w, f"SELECT MAX(c.{q(w, 'calendar.full_date')}) {_star(w)} WHERE "
                                                  f"c.{q(w, 'calendar.full_date')} >= DATE '2025-05-01' AND "
                                                  f"c.{q(w, 'calendar.full_date')} < DATE '2025-06-01'")

    def test_two_named_periods_side_by_side(self, warehouse):
        from core.multi_period import _period_predicate

        w, n = warehouse, warehouse.names
        policy = {"role_alias": "c", "date_value_column": n.column("calendar.full_date"), "date_key_type": "surrogate_fk",
                  "calendar_attributes": {"year": n.column("calendar.year"), "month_number": n.column("calendar.month_number"),
                                          "quarter": n.column("calendar.quarter")}}
        with postgres_names.using(w.spellings):
            star = (f"FROM {postgres_names.table(n.fact)} o JOIN {postgres_names.table(n.calendar)} c ON "
                    f"o.{postgres_names.ident(n.column('order_lines.order_date_key'))} = "
                    f"c.{postgres_names.ident(n.column('calendar.date_key'))}")
            net = f"o.{postgres_names.ident(n.column('order_lines.net_amount'))}"
            for label, start, end in (("March 2026", "2026-03-01", "2026-03-31"), ("Q1 2026", "2026-01-01", "2026-03-31"),
                                      ("2025", "2025-01-01", "2025-12-31")):
                predicate = _period_predicate(label, policy, "month", "postgres")
                got = on_postgres(w, postgres_sql(f"SELECT SUM({net}) {star} WHERE {predicate}"))
                assert got == _revenue_between(w, dt.date.fromisoformat(start), dt.date.fromisoformat(end)), label

    def test_the_distinct_values_the_value_index_keeps(self, warehouse):
        import json

        from core.value_index import _distinct_sql

        w = warehouse
        master = json.loads((Path(w.schema_dir) / "_schema.json").read_text())
        for logical in ("stores.store_name", "order_lines.status_code"):
            table, column = w.built.c(logical)
            meta = next(v for k, v in master.items() if not k.startswith("__") and k.split(".")[-1] == table)
            sql = postgres_sql(_distinct_sql("postgres", meta["database"], meta["schema"], table, column, 5000))
            assert sorted(on_postgres(w, sql)) == sorted(
                on_duckdb(w, f'SELECT DISTINCT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL'))

    def test_a_listing_and_a_drill(self, warehouse):
        from core.drill_dimension import build_deterministic_drill_sql
        from core.listing import listing_sql

        w, n = warehouse, warehouse.names
        with postgres_names.using(w.spellings):
            listed = postgres_sql(listing_sql({"table": n.stores, "column": n.column("stores.store_name"), "term": "Store"},
                                              "postgres"))
            base = (f"SELECT SUM(o.{postgres_names.ident(n.column('order_lines.net_amount'))}) AS revenue "
                    f"FROM {postgres_names.table(n.fact)} AS o")
            drilled = postgres_sql(build_deterministic_drill_sql(base, {
                "name": "Store", "display_table": n.stores, "display_column": n.column("stores.store_name"),
                "source_table": n.fact, "source_key_column": n.column("order_lines.store_id"),
                "display_key": n.column("stores.store_id")}, "postgres"))
        # Compared as sets: the two servers order text by different rules.
        assert sorted(on_postgres(w, listed)) == sorted(on_duckdb(
            w, f"SELECT DISTINCT {q(w, 'stores.store_name')} FROM {q(w, 'stores')}"))
        by_store = lambda row: str(row[1])  # noqa: E731
        assert sorted(on_postgres(w, drilled), key=by_store) == sorted(on_duckdb(
            w, f"SELECT SUM(o.{q(w, 'order_lines.net_amount')}), s.{q(w, 'stores.store_name')} FROM {q(w, 'order_lines')} o "
               f"LEFT JOIN {q(w, 'stores')} s ON o.{q(w, 'order_lines.store_id')} = s.{q(w, 'stores.store_id')} "
               f"GROUP BY 2"), key=by_store)

    def test_the_tables_counted_after_an_empty_answer(self, warehouse):
        from core.pipeline_helpers import _count_tables_for_zero_row

        w, n = warehouse, warehouse.names
        with postgres_names.using(w.spellings):
            counts = _count_tables_for_zero_row({"db_type": "postgres", "credentials": w.creds}, [n.fact, n.stores])
        assert counts == {n.fact: len(on_duckdb(w, f"SELECT 1 FROM {q(w, 'order_lines')}")),
                          n.stores: len(on_duckdb(w, f"SELECT 1 FROM {q(w, 'stores')}"))}

    def test_a_report_read_at_the_latest_date(self, warehouse):
        from core.report_engine import _apply_latest_date_filter

        w, n = warehouse, warehouse.names
        net = postgres_names.ident(w.built.c("order_lines.net_amount")[1])
        with postgres_names.using(w.spellings):
            sql = postgres_sql(_apply_latest_date_filter(
                f"SELECT SUM({net}) AS revenue FROM {postgres_names.table(n.fact)}",
                n.fact, n.column("order_lines.order_date_key"), "postgres"))
        key = q(w, "order_lines.order_date_key")
        assert on_postgres(w, sql) == on_duckdb(
            w, f"SELECT SUM({q(w, 'order_lines.net_amount')}) FROM {q(w, 'order_lines')} "
               f"WHERE {key} = (SELECT MAX({key}) FROM {q(w, 'order_lines')})")


class TestTheAdminChecksOnAPostgresServer:
    """What the admin's checks open and run: never an Oracle connection, and names as the database
    spells them, read from the workspace's own discovered schema."""

    @pytest.fixture()
    def account(self, warehouse, monkeypatch):
        import store
        from core import metric_dryrun

        # Some test modules import a fresh `store`: the module the dry run holds and the one an
        # import finds now can differ, and both are read.
        for module in {id(store): store, id(metric_dryrun.store): metric_dryrun.store}.values():
            monkeypatch.setattr(module, "get_client", lambda account_id: {"account_id": account_id, "db_config_id": 7})
            monkeypatch.setattr(module, "get_db_config", lambda config_id: {"id": 7, "db_type": "postgres",
                                                                            "credentials": warehouse.creds})
            monkeypatch.setattr(module, "get_client_state", lambda account_id: {"schema_dir": warehouse.schema_dir})
        return "acct-pg"

    def test_a_metric_formula_is_run_against_the_database(self, warehouse, account):
        from core.metric_dryrun import dry_run_metric_formula

        w, n = warehouse, warehouse.names
        net = postgres_names.ident(w.built.c("order_lines.net_amount")[1])
        outcome = asyncio.run(dry_run_metric_formula(account, formula=f"SUM({net})", base_table=n.fact))
        assert outcome.status == "ok" and outcome.probe_kind == "single_table", outcome.detail
        # A value that is no plain number (a numeric) is kept as its text.
        assert _value(decimal.Decimal(outcome.value)) == on_duckdb(
            w, f"SELECT SUM({q(w, 'order_lines.net_amount')}) FROM {q(w, 'order_lines')}")[0][0]
        broken = asyncio.run(dry_run_metric_formula(account, formula="SUM(no_such_column)", base_table=n.fact))
        assert broken.status == "error" and "no_such_column" in broken.detail
        assert "ORA-" not in broken.detail and "DPY-" not in broken.detail

    def test_a_formula_over_two_tables_is_checked_table_by_table(self, warehouse, account):
        from core.metric_dryrun import dry_run_metric_formula

        w = warehouse
        net, name = w.built.c("order_lines.net_amount")[1], w.built.c("stores.store_name")[1]
        formula = f"SUM({postgres_names.ident(net)}) / COUNT(DISTINCT {postgres_names.ident(name)})"
        outcome = asyncio.run(dry_run_metric_formula(account, formula=formula))
        assert outcome.status == "ok" and outcome.probe_kind == "multi_table", outcome.detail

    def test_a_gap_between_two_dates_is_run_through_its_joins(self, warehouse, account):
        import json

        from core.metric_builder import compile_metric_builder_config
        from core.metric_dryrun import dry_run_metric_formula

        w, n = warehouse, warehouse.names
        calendar_table = f"{w.schema}.{w.built.t('calendar')}"
        config = {"enabled": True, "mode": "date_gap", "aggregation": "AVG", "unit": "day", "missing_to": "null",
                  "date_column": w.built.c("calendar.full_date")[1], "invalid_keys": [-1],
                  "from_join": {"alias": "ordered", "table": calendar_table, "to_column": w.built.c("calendar.date_key")[1],
                                "from_column": w.built.c("order_lines.order_date_key")[1]},
                  "to_join": {"alias": "shipped", "table": calendar_table, "to_column": w.built.c("calendar.date_key")[1],
                              "from_column": w.built.c("order_lines.ship_date_key")[1]}}
        compiled = compile_metric_builder_config(config, "postgres")
        outcome = asyncio.run(dry_run_metric_formula(account, formula=compiled.formula, base_table=n.fact,
                                                     metric_builder_config=compiled.config_json))
        assert outcome.status == "ok" and outcome.probe_kind == "join_probe", outcome.detail
        # The formula is an aggregate: the probe's one row is the average over every order.
        expected = on_duckdb(w, f"SELECT AVG(CAST(s.{q(w, 'calendar.full_date')} - c.{q(w, 'calendar.full_date')} AS DOUBLE)) "
                                f"FROM {q(w, 'order_lines')} o "
                                f"JOIN {q(w, 'calendar')} c ON o.{q(w, 'order_lines.order_date_key')} = c.{q(w, 'calendar.date_key')} "
                                f"JOIN {q(w, 'calendar')} s ON o.{q(w, 'order_lines.ship_date_key')} = s.{q(w, 'calendar.date_key')} "
                                f"WHERE o.{q(w, 'order_lines.ship_date_key')} <> -1")
        assert [(_value(outcome.value),)] == expected
        assert json.loads(compiled.config_json)["required_joins"]

    def test_a_filter_that_matches_nothing_is_named(self, warehouse, account):
        import json

        from core.metric_dryrun import check_filter_matches

        w = warehouse
        status = w.built.c("order_lines.status_code")[1]
        config = {"mode": "filtered_aggregate", "measure": w.built.c("order_lines.net_amount")[1],
                  "filters": [{"field": status, "operator": "equals", "value": "Z"}]}
        assert asyncio.run(check_filter_matches(account, metric_builder_config=json.dumps(config))) == (status,)
        config["filters"][0]["value"] = "S"
        assert asyncio.run(check_filter_matches(account, metric_builder_config=json.dumps(config))) == ()

    def test_a_join_is_profiled_as_the_graph_keeps_it(self, warehouse):
        from core.relationship_validator import _execute_probe
        from core.schema import build_entity_graph_from_schema

        w = warehouse
        graph = build_entity_graph_from_schema(w.schema_dir)
        entities = {entity["entity_name"]: entity for entity in graph["entities"]}
        rel = next(r for r in graph["relationships"]
                   if entities[r["from_entity"]]["table_name"] == w.built.t("order_lines")
                   and entities[r["to_entity"]]["table_name"] == w.built.t("stores"))
        # Without the workspace's spellings: the graph keeps the table's own name, its schema upper-cased.
        result = _execute_probe("acct-pg", rel, entities[rel["from_entity"]], entities[rel["to_entity"]],
                                db_type="postgres", raw_cfg={"credentials": w.creds})
        assert result.checked_by == "db" and result.match_rate == 100.0, result.message
        assert result.target_rows == on_duckdb(w, f"SELECT COUNT(*) FROM {q(w, 'stores')}")[0][0]
        assert result.target_duplicate_keys == 0 and result.orphan_rate == 0.0

    def test_a_join_is_validated_as_the_admin_validates_it(self, warehouse, account, monkeypatch):
        import store
        from core.relationship_validator import validate_relationship
        from core.schema import build_entity_graph_from_schema

        w = warehouse
        graph = build_entity_graph_from_schema(w.schema_dir)
        entities = {entity["entity_name"]: entity for entity in graph["entities"]}
        rel = next(r for r in graph["relationships"]
                   if entities[r["from_entity"]]["table_name"] == w.built.t("returns")
                   and entities[r["to_entity"]]["table_name"] == w.built.t("order_lines"))
        monkeypatch.setattr(store, "get_relationship", lambda account_id, rel_id: {**rel, "id": rel_id})
        monkeypatch.setattr(store, "list_entities", lambda account_id, active_only=True: graph["entities"])
        result = validate_relationship(account, 11, execute=True)
        assert result.checked_by == "db" and result.status == "valid", result.message
        assert result.target_rows == on_duckdb(w, f"SELECT COUNT(*) FROM {q(w, 'order_lines')}")[0][0]

    def test_the_join_skeleton_the_model_is_given(self, warehouse):
        from core.graph_resolver import build_join_skeleton
        from core.schema import build_entity_graph_from_schema

        w = warehouse
        graph = build_entity_graph_from_schema(w.schema_dir)
        entities = {entity["entity_name"]: entity for entity in graph["entities"]}
        rel = next(r for r in graph["relationships"]
                   if entities[r["from_entity"]]["table_name"] == w.built.t("order_lines")
                   and entities[r["to_entity"]]["table_name"] == w.built.t("stores"))
        skeleton = build_join_skeleton([{**rel, "_direction": "forward", "join_type": "INNER"}], entities,
                                       rel["from_entity"], "postgres")
        assert on_postgres(w, postgres_sql(f"SELECT COUNT(*) {skeleton}")) == on_duckdb(
            w, f"SELECT COUNT(*) FROM {q(w, 'order_lines')} o JOIN {q(w, 'stores')} s "
               f"ON o.{q(w, 'order_lines.store_id')} = s.{q(w, 'stores.store_id')}")

    def test_the_unknown_members_of_a_dimension(self, warehouse):
        from core.unknown_members import build_probe_sql

        w = warehouse
        entity = {"schema_name": w.schema.upper(), "table_name": w.built.t("calendar")}
        key, name = w.built.c("calendar.date_key")[1], w.built.c("calendar.month_name")[1]
        rows = on_postgres(w, postgres_sql(build_probe_sql("postgres", entity, key, [name])))
        assert rows == on_duckdb(w, f'SELECT "{key}", "{name}" FROM {q(w, "calendar")} WHERE "{key}" IN '
                                    f"(0, -1, -2, -3, -9, -99, -999, 777, 888, 999, 9999, 99999, 999999)")
        assert rows

    def test_an_example_is_compiled_without_being_run(self, warehouse):
        from core.examples import _execute_on_connection, _open_connection

        w, n = warehouse, warehouse.names
        conn = _open_connection(w.creds, "postgres")
        assert conn is not None
        try:
            with postgres_names.using(w.spellings):
                good = f"SELECT COUNT(*) FROM {postgres_names.table(n.fact)}"
            _execute_on_connection(conn, "postgres", good)
            with pytest.raises(Exception, match="no_such_column"):
                _execute_on_connection(conn, "postgres", good.replace("COUNT(*)", "SUM(no_such_column)"))
        finally:
            conn.close()



class TestAQuestionIsAnsweredInTheWorkspacesSpelling:
    """handle_query, end to end with the model and the warehouse stubbed: a PostgreSQL workspace's
    names, read from the _schema.json its discovery wrote, are the ones its SQL is written in while
    the question is answered -- and are put back when it is done."""

    SCHEMA = {
        "WH.MART.SalesFact": {
            "columns": [{"name": "SalesKey", "type": "bigint", "nullable": False, "comment": ""},
                        {"name": "SalesAmount", "type": "numeric(18,2)", "nullable": True, "comment": ""}],
            "pk_columns": ["SalesKey"], "row_count": 10, "comment": "", "schema": "Mart", "database": "wh",
        },
    }

    def _ask(self, tmp_path, db_type):
        import contextlib
        import json
        from unittest.mock import patch

        import core.query_pipeline as qp
        import store
        from gateway.base import PlatformEvent

        store.init_db()
        account = f"acct{os.urandom(4).hex()}"
        store.upsert_client(account, "Test Ltd")
        (tmp_path / "_schema.json").write_text(json.dumps(self.SCHEMA), encoding="utf-8")
        store.update_client_state(account, "READY", {"schema_dir": str(tmp_path)})
        store.save_metric(account, {
            "name": "Sales Amount", "formula_type": "expression", "sql_template": 'SUM("SalesAmount")',
            "base_table": "Mart.SalesFact", "synonyms": "sales, sales amount", "description": "Invoiced sales",
        })
        seen: dict = {}

        class Retriever:
            last_retrieval_weak = False
            last_retrieval_unscored = False

            def retrieve(self, question, n=8, allowed_tables=None):
                return ['# wh.Mart.SalesFact\n**SQL table name:** `"Mart"."SalesFact"`\n| `"SalesAmount"` | numeric |']

            def retrieve_fact_patterns(self, question, n=2, allowed_tables=None):
                return []

            def _is_global(self, doc):
                return False

        async def model_reply(system, user, *args, **kwargs):
            seen.setdefault("written", postgres_names.table("WH.MART.SALESFACT"))
            return 'SELECT SUM("SalesAmount") AS total FROM "Mart"."SalesFact"', 1, 1

        class Adapter:
            platform, session_id, thread_id, last_result_id = "portal", f"{account}:portal:u1", "t1", None

            async def send_message(self, event, text, **kwargs):
                seen.setdefault("messages", []).append(str(text)[:300])

            async def send_typing(self, *args, **kwargs):
                return None

            def add_to_history(self, **kwargs):
                return None

            def cache_result(self, *args, **kwargs):
                return None

        async def card(*args, **kwargs):
            # Shown on the question's own task: the warehouse is read on a worker thread,
            # which a ContextVar does not reach, so the SQL is written before it is sent there.
            seen.setdefault("written", postgres_names.table("WH.MART.SALESFACT"))

        def execute(credentials, db_type_, sql, **kwargs):
            seen.setdefault("executed", sql)
            return SimpleNamespace(sql=sql, rows=[{"total": 1.0}])

        async def ask():
            await qp.handle_query(account, PlatformEvent(account, "u1", "c1", "total sales amount", "portal"),
                                  Adapter(), "total sales amount",
                                  {"id": 1, "role": "admin", "email": "u@x.com", "name": "U", "group_name": None,
                                   "lang": "en"})
            return postgres_names._ACTIVE.get()

        with contextlib.ExitStack() as stack:
            for mock in (
                patch.object(qp, "get_state", return_value={
                    "state": "READY", "schema_dir": str(tmp_path), "kb_dir": str(tmp_path)}),
                patch.object(qp, "get_client_db", return_value={
                    "db_type": db_type, "credentials": {}, "name": "db", "id": 1}),
                patch.object(qp, "load_retriever", return_value=Retriever()),
                patch.object(qp, "llm_complete", model_reply),
                patch.object(qp, "_send_results", card),
                patch.object(qp, "execute_governed_query", execute),
                patch.object(qp, "resolve_provider", return_value=("anthropic", "claude", "key", {})),
                patch.object(qp, "_log_q", lambda *args, **kwargs: None),
                patch.object(qp, "retrieve_similar_examples", return_value=[]),
            ):
                stack.enter_context(mock)
            after = asyncio.run(ask())
        return seen, after

    def test_postgres(self, tmp_path):
        seen, after = self._ask(tmp_path, "postgres")
        assert "written" in seen, seen
        assert seen["written"] == '"Mart"."SalesFact"'
        assert after is None

    def test_another_database_activates_none(self, tmp_path):
        seen, after = self._ask(tmp_path, "azure_sql")
        assert seen["written"] == "mart.salesfact" and after is None
