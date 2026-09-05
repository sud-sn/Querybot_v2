"""
tests/test_trend_regrain_followup.py

Regression suite for the "provide the trend" follow-up.

Scenario: "What was my revenue for the past 5 days?" answers with one total.
The user then asks "provide the trend" and expects the SAME query, with the
governed business date added to the GROUP BY, over the SAME 5-day window.

Before this route existed the follow-up was neither answerable from the result
cache (a single total cannot be un-aggregated) nor answerable as a new question
(three words carry no metric, so the metric, the business date and the window
were all re-derived from nothing).

The DuckDB tests at the bottom are the real proof: they execute the parent SQL
and the re-grained SQL against actual rows and assert the daily series sums
back to the parent total over exactly the same window.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_tmpdir = tempfile.mkdtemp(prefix="querybot_trend_regrain_")
os.environ.setdefault("DB_PATH", str(Path(_tmpdir) / "test_querybot.db"))
os.environ.setdefault("QUERYBOT_KEY_FILE", str(Path(_tmpdir) / "test_key"))

from core.result_regrain import (  # noqa: E402
    build_regrain_sql,
    parse_trend_regrain_request,
    regrain_question_text,
    resolve_regrain_grain,
    temporal_policy_from_plan,
)

PIPELINE_SRC = (ROOT / "core" / "query_pipeline.py").read_text(encoding="utf-8")


# ── fixtures: a generic sales star with a role-playing date dimension ─────────

def _policy(**overrides):
    policy = {
        "kind": "last_n_days",
        "anchor_policy": "latest_available",
        "fact_table": "ERP.F_SALES_INVOICE",
        "fact_column": "INVOICE_DT_DMS_KEY",
        "dimension_table": "ERP.DT_DMS",
        "dimension_key": "DT_DMS_KEY",
        "date_column": "CAL_DATE",
        "date_key_type": "surrogate_fk",
        "temporal_grain": "day",
        "business_role": "Invoice Date",
        "anchor_table": "ERP.F_SALES_INVOICE",
    }
    policy.update(overrides)
    return policy


PARENT_SQL = (
    "SELECT SUM(f.NET_REVENUE) AS TOTAL_REVENUE "
    "FROM ERP.F_SALES_INVOICE f "
    "INNER JOIN ERP.DT_DMS d ON f.INVOICE_DT_DMS_KEY = d.DT_DMS_KEY "
    "WHERE d.CAL_DATE >= DATEADD(day, -4, (SELECT MAX(d2.CAL_DATE) "
    "FROM ERP.F_SALES_INVOICE f2 INNER JOIN ERP.DT_DMS d2 "
    "ON f2.INVOICE_DT_DMS_KEY = d2.DT_DMS_KEY))"
)


# ══════════════════════════════════════════════════════════════════════════════
# 1  Intent detection
# ══════════════════════════════════════════════════════════════════════════════
class TestTrendRegrainIntent(unittest.TestCase):

    def test_trend_follow_ups_are_recognized(self):
        for text in (
            "provide the trend", "Provide the trend.", "show the trend",
            "give me the trend", "trend", "the trend", "trendline",
            "plot the trend", "chart the trend", "trend over time",
            "over time", "show it as a time series", "time series",
            "by day", "per day", "day wise", "break it down by day",
            "break down by week", "group by month", "split by week",
            "daily trend", "weekly breakdown", "monthly trend",
            "show me the daily numbers", "can you provide the trend",
        ):
            with self.subTest(text=text):
                self.assertIsNotNone(parse_trend_regrain_request(text), text)

    def test_new_questions_are_not_regrain_requests(self):
        """Anything naming new business content must reach the full pipeline."""
        for text in (
            "show the trend of orders",
            "revenue trend by warehouse",
            "trend of new signups by region",
            "what is my revenue for the past 5 days",
            "show top 10 customers",
            "break it down by warehouse",
            "by customer",
            "group by customer",
            "why did revenue drop",
            "compare to last month",
            "provide the total",
            "show all rows",
            "what was the trend in headcount across departments last year",
            "exclude cancelled orders",
            "no dont use this",
        ):
            with self.subTest(text=text):
                self.assertIsNone(parse_trend_regrain_request(text), text)

    def test_blank_input_is_not_a_request(self):
        for text in ("", "   ", None):
            self.assertIsNone(parse_trend_regrain_request(text))

    def test_named_grain_is_extracted(self):
        for text, grain in (
            ("daily trend", "day"),
            ("by day", "day"),
            ("weekly breakdown", "week"),
            ("monthly trend", "month"),
            ("group by month", "month"),
            ("by quarter", "quarter"),
            ("yearly trend", "year"),
        ):
            with self.subTest(text=text):
                self.assertEqual(parse_trend_regrain_request(text).grain, grain)

    def test_unnamed_grain_is_inherited(self):
        self.assertEqual(parse_trend_regrain_request("provide the trend").grain, "")


# ══════════════════════════════════════════════════════════════════════════════
# 2  Grain resolution
# ══════════════════════════════════════════════════════════════════════════════
class TestGrainResolution(unittest.TestCase):

    def test_unnamed_grain_follows_the_governed_role(self):
        request = parse_trend_regrain_request("provide the trend")
        self.assertEqual(resolve_regrain_grain(request, _policy()), "day")
        self.assertEqual(
            resolve_regrain_grain(request, _policy(temporal_grain="month")), "month",
        )

    def test_named_grain_is_honored(self):
        self.assertEqual(
            resolve_regrain_grain(parse_trend_regrain_request("monthly trend"), _policy()),
            "month",
        )

    def test_request_cannot_go_finer_than_the_stored_grain(self):
        """A month-grain role has no days to slice into."""
        self.assertEqual(
            resolve_regrain_grain(
                parse_trend_regrain_request("daily trend"),
                _policy(temporal_grain="month"),
            ),
            "month",
        )

    def test_missing_grain_metadata_defaults_to_day(self):
        self.assertEqual(
            resolve_regrain_grain(
                parse_trend_regrain_request("provide the trend"),
                _policy(temporal_grain=""),
            ),
            "day",
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3  The governed date comes from the parent answer, never from the follow-up
# ══════════════════════════════════════════════════════════════════════════════
class TestTemporalPolicyFromPlan(unittest.TestCase):

    def test_compiled_temporal_policy_is_preferred(self):
        policy = temporal_policy_from_plan({"temporal_policies": [_policy()]})
        self.assertEqual(policy["fact_column"], "INVOICE_DT_DMS_KEY")
        self.assertEqual(policy["business_role"], "Invoice Date")

    def test_contextual_date_field_is_the_fallback(self):
        policy = temporal_policy_from_plan({
            "fields": [{
                "term": "Invoice Date", "role": "contextual_date",
                "table": "ERP.DT_DMS", "column": "CAL_DATE",
                "source_table": "ERP.F_SALES_INVOICE",
                "source_key_column": "INVOICE_DT_DMS_KEY",
                "date_key_type": "surrogate_fk", "temporal_grain": "day",
            }],
        })
        self.assertEqual(policy["fact_table"], "ERP.F_SALES_INVOICE")
        self.assertEqual(policy["dimension_table"], "ERP.DT_DMS")
        self.assertEqual(policy["date_column"], "CAL_DATE")

    def test_native_date_field_carries_no_dimension(self):
        policy = temporal_policy_from_plan({
            "fields": [{
                "term": "Invoice Date", "role": "contextual_date",
                "table": "ERP.F_SALES_INVOICE", "column": "INVOICE_DATE",
                "source_table": "ERP.F_SALES_INVOICE",
                "source_key_column": "INVOICE_DATE",
                "date_key_type": "native_date",
            }],
        })
        self.assertEqual(policy["dimension_table"], "")

    def test_no_governed_date_returns_nothing(self):
        self.assertEqual(temporal_policy_from_plan({}), {})
        self.assertEqual(temporal_policy_from_plan(None), {})
        self.assertEqual(
            temporal_policy_from_plan({"fields": [{"role": "dimension"}]}), {},
        )


# ══════════════════════════════════════════════════════════════════════════════
# 4  SQL rewrite — same query, date in the GROUP BY
# ══════════════════════════════════════════════════════════════════════════════
class TestRegrainSql(unittest.TestCase):

    def _build(self, sql=PARENT_SQL, policy=None, grain="day", db="azure_sql"):
        return build_regrain_sql(sql, policy or _policy(), grain, db)

    def test_date_is_added_to_select_group_by_and_order_by(self):
        sql, reason = self._build()
        self.assertEqual(reason, "")
        self.assertIn("GROUP BY", sql.upper())
        self.assertIn("ORDER BY", sql.upper())
        self.assertIn("CAL_DATE", sql)
        self.assertIn("[Invoice Date]", sql)

    def test_the_period_leads_the_select_list(self):
        """Chart selection treats the first column as the category axis."""
        sql, _ = self._build()
        projections = sql.split(" FROM ", 1)[0]
        self.assertLess(projections.index("CAL_DATE"), projections.index("SUM("))

    def test_the_where_clause_is_preserved_exactly(self):
        """The window the user asked about must not move."""
        sql, _ = self._build()
        parent_where = PARENT_SQL.split(" WHERE ", 1)[1]
        child_where = sql.split(" WHERE ", 1)[1].split(" GROUP BY ", 1)[0]

        def _norm(text):
            return re.sub(r"\s+|\bAS\b", " ", text, flags=re.I).replace(" ", "").upper()

        self.assertEqual(_norm(parent_where), _norm(child_where))

    def test_the_measure_is_untouched(self):
        sql, _ = self._build()
        self.assertIn("SUM(f.NET_REVENUE) AS TOTAL_REVENUE", sql)

    def test_no_second_join_is_added_to_the_date_dimension(self):
        """The governed role join is already there; reuse that exact alias."""
        sql, _ = self._build()
        self.assertEqual(sql.upper().count("ERP.DT_DMS AS D "), 1)

    def test_coarser_grains_bucket_the_date(self):
        for grain, marker in (
            ("month", "DATEFROMPARTS"),
            ("quarter", "DATEPART(QUARTER"),
            ("year", "DATEFROMPARTS"),
            ("week", "DATEPART(WEEKDAY"),
        ):
            with self.subTest(grain=grain):
                sql, reason = self._build(grain=grain)
                self.assertEqual(reason, "")
                self.assertIn(marker, sql.upper().replace("DATEPART(QUARTER", "DATEPART(QUARTER"))

    def test_grain_is_named_in_the_column_label(self):
        sql, _ = self._build(grain="month")
        self.assertIn("[Invoice Date (Month)]", sql)

    def test_native_date_role_needs_no_dimension_join(self):
        sql, reason = self._build(
            sql=(
                "SELECT SUM(f.NET_REVENUE) AS TOTAL FROM ERP.F_SALES_INVOICE f "
                "WHERE f.INVOICE_DATE >= '2026-01-01'"
            ),
            policy=_policy(
                dimension_table="", dimension_key="",
                fact_column="INVOICE_DATE", date_column="INVOICE_DATE",
                date_key_type="native_date",
            ),
        )
        self.assertEqual(reason, "")
        self.assertIn("INVOICE_DATE", sql)
        self.assertIn("GROUP BY", sql.upper())

    def test_snowflake_and_postgres_dialects_render(self):
        for db, marker in (("snowflake", "DATE_TRUNC"), ("postgres", "DATE_TRUNC")):
            with self.subTest(db=db):
                sql, reason = self._build(grain="month", db=db)
                self.assertEqual(reason, "")
                self.assertIn(marker, sql.upper())

    def test_an_existing_group_by_is_extended_not_replaced(self):
        grouped = (
            "SELECT w.WHS_NAME, SUM(f.NET_REVENUE) AS T FROM ERP.F_SALES_INVOICE f "
            "INNER JOIN ERP.DT_DMS d ON f.INVOICE_DT_DMS_KEY = d.DT_DMS_KEY "
            "INNER JOIN ERP.D_WHS w ON f.WHS_KEY = w.WHS_KEY "
            "WHERE d.CAL_DATE >= '2026-01-01' GROUP BY w.WHS_NAME ORDER BY T DESC"
        )
        sql, reason = self._build(sql=grouped)
        self.assertEqual(reason, "")
        group_clause = sql.upper().split("GROUP BY", 1)[1].split("ORDER BY", 1)[0]
        self.assertIn("WHS_NAME", group_clause)
        self.assertIn("CAL_DATE", group_clause)
        # Chronological first, then whatever ranking the parent applied.
        order_clause = sql.upper().split("ORDER BY", 1)[1]
        self.assertLess(order_clause.index("CAL_DATE"), order_clause.index("T DESC"))

    def test_refusals_are_explicit_and_never_guess(self):
        cases = {
            "parent answer already reports this date": (
                "SELECT d.CAL_DATE, SUM(f.NET_REVENUE) AS T "
                "FROM ERP.F_SALES_INVOICE f "
                "INNER JOIN ERP.DT_DMS d ON f.INVOICE_DT_DMS_KEY = d.DT_DMS_KEY "
                "GROUP BY d.CAL_DATE",
                _policy(),
            ),
            "parent answer is row-limited": (
                "SELECT TOP 10 c.NAME, SUM(f.NET_REVENUE) AS T "
                "FROM ERP.F_SALES_INVOICE f "
                "INNER JOIN ERP.DT_DMS d ON f.INVOICE_DT_DMS_KEY = d.DT_DMS_KEY "
                "INNER JOIN ERP.D_CUST c ON f.CUST_KEY = c.CUST_KEY GROUP BY c.NAME",
                _policy(),
            ),
            "parent answer is not aggregated": (
                "SELECT f.NET_REVENUE FROM ERP.F_SALES_INVOICE f", _policy(),
            ),
            "governed date dimension is not joined in the parent SQL": (
                "SELECT SUM(f.NET_REVENUE) AS T FROM ERP.F_SALES_INVOICE f", _policy(),
            ),
            "no governed date on the parent answer": (PARENT_SQL, {}),
            "no parent SQL": ("", _policy()),
        }
        for expected_reason, (sql, policy) in cases.items():
            with self.subTest(reason=expected_reason):
                built, reason = build_regrain_sql(sql, policy, "day", "azure_sql")
                self.assertEqual(built, "")
                self.assertEqual(reason, expected_reason)

    def test_unparseable_parent_sql_refuses_rather_than_raising(self):
        built, reason = build_regrain_sql("this is not sql at all (", _policy(), "day")
        self.assertEqual(built, "")
        self.assertTrue(reason)


# ══════════════════════════════════════════════════════════════════════════════
# 5  Fallback question text
# ══════════════════════════════════════════════════════════════════════════════
class TestRegrainQuestionText(unittest.TestCase):

    def test_the_parent_question_carries_the_metric_and_window(self):
        self.assertEqual(
            regrain_question_text("What was my revenue for past 5 days?", "day"),
            "What was my revenue for past 5 days by day",
        )

    def test_grain_is_not_duplicated(self):
        self.assertEqual(
            regrain_question_text("revenue by month", "month"), "revenue by month",
        )

    def test_empty_parent_still_yields_something_answerable(self):
        self.assertEqual(regrain_question_text("", "week"), "Trend by week")


# ══════════════════════════════════════════════════════════════════════════════
# 6  Executed against real rows: the trend must sum back to the parent total
# ══════════════════════════════════════════════════════════════════════════════
class TestRegrainAgainstRealRows(unittest.TestCase):
    """The business guarantee, verified by execution rather than by inspection."""

    @classmethod
    def setUpClass(cls):
        try:
            import duckdb
        except Exception as exc:                        # pragma: no cover
            raise unittest.SkipTest(f"duckdb unavailable: {exc}")
        cls.conn = duckdb.connect(":memory:")
        cls.conn.execute("CREATE SCHEMA ERP")
        cls.conn.execute(
            "CREATE TABLE ERP.DT_DMS (DT_DMS_KEY INTEGER, CAL_DATE DATE)"
        )
        cls.conn.execute(
            "CREATE TABLE ERP.F_SALES_INVOICE "
            "(INVOICE_DT_DMS_KEY INTEGER, LAST_MOD_DT_DMS_KEY INTEGER, NET_REVENUE DECIMAL(18,2))"
        )
        # 8 calendar days; the last 5 are the window under test.
        for key, day in enumerate(range(1, 9), start=101):
            cls.conn.execute(
                "INSERT INTO ERP.DT_DMS VALUES (?, ?)",
                [key, f"2026-03-{day:02d}"],
            )
        # Two invoices a day, so a daily group is a real aggregate.
        for key in range(101, 109):
            for amount in (100 + key, 5):
                cls.conn.execute(
                    "INSERT INTO ERP.F_SALES_INVOICE VALUES (?, ?, ?)",
                    [key, 101, amount],
                )

    PARENT = (
        "SELECT SUM(f.NET_REVENUE) AS TOTAL_REVENUE "
        "FROM ERP.F_SALES_INVOICE f "
        "INNER JOIN ERP.DT_DMS d ON f.INVOICE_DT_DMS_KEY = d.DT_DMS_KEY "
        "WHERE d.CAL_DATE >= (SELECT MAX(d2.CAL_DATE) - INTERVAL 4 DAY "
        "FROM ERP.F_SALES_INVOICE f2 INNER JOIN ERP.DT_DMS d2 "
        "ON f2.INVOICE_DT_DMS_KEY = d2.DT_DMS_KEY)"
    )

    def test_daily_trend_sums_back_to_the_parent_total(self):
        parent_total = self.conn.execute(self.PARENT).fetchone()[0]

        sql, reason = build_regrain_sql(self.PARENT, _policy(), "day", "duckdb")
        self.assertEqual(reason, "")
        rows = self.conn.execute(sql).fetchall()

        self.assertEqual(len(rows), 5, "a 5-day window must return 5 daily rows")
        self.assertEqual(
            sum(row[1] for row in rows), parent_total,
            "the daily series must sum back to the total the user is looking at",
        )
        # Chronological, and confined to the parent's own window.
        dates = [row[0] for row in rows]
        self.assertEqual(dates, sorted(dates))
        self.assertEqual(str(dates[0]), "2026-03-04")
        self.assertEqual(str(dates[-1]), "2026-03-08")

    def test_the_trend_never_moves_to_another_date_role(self):
        """The audit-date key exists on the fact and must stay unused."""
        sql, _ = build_regrain_sql(self.PARENT, _policy(), "day", "duckdb")
        self.assertNotIn("LAST_MOD_DT_DMS_KEY", sql)

    def test_weekly_regrain_also_sums_back(self):
        parent_total = self.conn.execute(self.PARENT).fetchone()[0]
        sql, reason = build_regrain_sql(self.PARENT, _policy(), "week", "duckdb")
        self.assertEqual(reason, "")
        rows = self.conn.execute(sql).fetchall()
        self.assertGreaterEqual(len(rows), 1)
        self.assertLess(len(rows), 5, "weeks must be coarser than days")
        self.assertEqual(sum(row[1] for row in rows), parent_total)


class TestRegrainAgainstRealRowsSqlite(unittest.TestCase):
    """The same guarantee on stdlib SQLite, so it runs everywhere.

    DuckDB is a runtime dependency but not always installed for unit runs; this
    class keeps the sum-back property executed rather than merely asserted.
    """

    POLICY = {
        "fact_table": "F_SALES_INVOICE",
        "fact_column": "INVOICE_DT_DMS_KEY",
        "dimension_table": "DT_DMS",
        "dimension_key": "DT_DMS_KEY",
        "date_column": "CAL_DATE",
        "date_key_type": "surrogate_fk",
        "temporal_grain": "day",
        "business_role": "Invoice Date",
    }
    PARENT = (
        "SELECT SUM(f.NET_REVENUE) AS TOTAL_REVENUE "
        "FROM F_SALES_INVOICE f "
        "INNER JOIN DT_DMS d ON f.INVOICE_DT_DMS_KEY = d.DT_DMS_KEY "
        "WHERE d.CAL_DATE >= (SELECT DATE(MAX(d2.CAL_DATE), '-4 day') "
        "FROM F_SALES_INVOICE f2 INNER JOIN DT_DMS d2 "
        "ON f2.INVOICE_DT_DMS_KEY = d2.DT_DMS_KEY)"
    )

    def setUp(self):
        import sqlite3

        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("CREATE TABLE DT_DMS (DT_DMS_KEY INTEGER, CAL_DATE TEXT)")
        self.conn.execute(
            "CREATE TABLE F_SALES_INVOICE (INVOICE_DT_DMS_KEY INTEGER, "
            "LAST_MOD_DT_DMS_KEY INTEGER, NET_REVENUE REAL)"
        )
        for key, day in enumerate(range(1, 9), start=101):
            self.conn.execute(
                "INSERT INTO DT_DMS VALUES (?, ?)", (key, f"2026-03-{day:02d}"),
            )
        for key in range(101, 109):
            for amount in (100 + key, 5):
                self.conn.execute(
                    "INSERT INTO F_SALES_INVOICE VALUES (?, ?, ?)", (key, 101, amount),
                )
        self.conn.commit()

    def tearDown(self):
        self.conn.close()

    def test_daily_trend_sums_back_to_the_parent_total(self):
        parent_total = self.conn.execute(self.PARENT).fetchone()[0]
        self.assertGreater(parent_total, 0)

        sql, reason = build_regrain_sql(self.PARENT, self.POLICY, "day", "sqlite")
        self.assertEqual(reason, "")
        rows = self.conn.execute(sql).fetchall()

        self.assertEqual(len(rows), 5, "a 5-day window must return 5 daily rows")
        self.assertAlmostEqual(
            sum(row[1] for row in rows), parent_total, places=6,
            msg="the daily series must sum back to the total the user is looking at",
        )
        dates = [row[0] for row in rows]
        self.assertEqual(dates, sorted(dates), "a trend must read chronologically")
        self.assertEqual(dates[0], "2026-03-04")
        self.assertEqual(dates[-1], "2026-03-08")

    def test_the_trend_never_moves_to_another_date_role(self):
        sql, _ = build_regrain_sql(self.PARENT, self.POLICY, "day", "sqlite")
        self.assertNotIn("LAST_MOD_DT_DMS_KEY", sql)

    def test_the_window_is_unchanged_by_the_regrain(self):
        """Every returned day must also satisfy the parent's own filter."""
        sql, _ = build_regrain_sql(self.PARENT, self.POLICY, "day", "sqlite")
        regrained_days = {row[0] for row in self.conn.execute(sql).fetchall()}
        parent_days = {
            row[0] for row in self.conn.execute(
                self.PARENT.replace(
                    "SELECT SUM(f.NET_REVENUE) AS TOTAL_REVENUE",
                    "SELECT DISTINCT d.CAL_DATE",
                    1,
                )
            ).fetchall()
        }
        self.assertEqual(regrained_days, parent_days)


# ══════════════════════════════════════════════════════════════════════════════
# 7  Pipeline wiring
# ══════════════════════════════════════════════════════════════════════════════
class TestPipelineWiring(unittest.TestCase):

    def test_the_route_exists_and_runs_before_the_metric_registry(self):
        regrain = PIPELINE_SRC.find("parse_trend_regrain_request(_analysis_question)")
        metric = PIPELINE_SRC.find("Step 3: Metric registry")
        self.assertGreater(regrain, 0, "the trend re-grain route must be wired in")
        self.assertLess(
            regrain, metric,
            "a trend follow-up must be resolved before metric matching, which "
            "would otherwise answer it as an unrelated new question",
        )

    def test_the_route_uses_the_cached_parent_answer(self):
        block = PIPELINE_SRC.split("if _regrain_request:", 1)[1].split(
            "Step 3: Metric registry", 1,
        )[0]
        self.assertIn("result_cache.get_snapshot(", block)
        self.assertIn('_regrain_snapshot.get("sql")', block)
        self.assertIn("temporal_policy_from_plan(", block)
        self.assertIn("build_regrain_sql(", block)

    def test_the_route_executes_through_the_governed_policy_path(self):
        block = PIPELINE_SRC.split("if _regrain_request:", 1)[1].split(
            "Step 3: Metric registry", 1,
        )[0]
        self.assertIn("_execute_with_policy", block)
        self.assertIn("PolicyDeniedError", block)
        self.assertIn("_send_results(", block)

    def test_no_llm_is_involved_in_the_route(self):
        block = PIPELINE_SRC.split("if _regrain_request:", 1)[1].split(
            "Step 3: Metric registry", 1,
        )[0]
        for forbidden in ("llm_complete", "build_sql_system_prompt", "resolve_provider"):
            self.assertNotIn(forbidden, block, f"{forbidden} must not be reachable here")

    def test_the_fallback_restates_the_parent_question(self):
        block = PIPELINE_SRC.split("if _regrain_request:", 1)[1].split(
            "Step 3: Metric registry", 1,
        )[0]
        self.assertIn("question = _regrain_question", block)
        self.assertIn("plan_analytical_intent(", block)

    def test_the_module_itself_never_imports_an_llm(self):
        source = (ROOT / "core" / "result_regrain.py").read_text(encoding="utf-8")
        for forbidden in ("core.llm", "llm_complete", "openai", "anthropic"):
            self.assertNotIn(forbidden, source)


# ══════════════════════════════════════════════════════════════════════════════
# 7b  Every channel must cache the governed plan the follow-up depends on
# ══════════════════════════════════════════════════════════════════════════════
class TestEveryChannelCachesTheGovernedPlan(unittest.TestCase):
    """A missing plan looks exactly like "no governed date" to the caller.

    WebAdapter stored semantic_plan in the snapshot metadata from the start; the
    shared channel mixin did not, so on Teams/Slack/Zoom the deterministic trend
    route silently fell back to the pipeline instead of reusing the parent SQL —
    the whole feature degraded with nothing in the logs to say why.
    """

    PLAN = {"temporal_policies": [_policy()]}

    def _store_via(self, adapter, session_id):
        from core.result_cache import result_cache

        result_cache.clear(session_id)
        adapter.cache_result(
            [{"TOTAL_REVENUE": 1234.0}],
            "What was my revenue for past 5 days?",
            PARENT_SQL,
            {"id": 7, "db_type": "azure_sql"},
            "",
            question_id=None,
            column_formats={},
            data_brief={},
            semantic_plan=self.PLAN,
            contract_version="contract-9",
        )
        return result_cache.get_snapshot(session_id)

    def test_web_adapter_snapshot_carries_the_plan(self):
        from unittest.mock import AsyncMock
        from gateway.web_adapter import WebAdapter

        adapter = WebAdapter(
            AsyncMock(), "tenant-regrain", "web_9", thread_id="thread-9",
            portal_user_id=9,
        )
        snapshot = self._store_via(adapter, adapter.session_id)
        self.assertEqual(
            temporal_policy_from_plan(
                (snapshot.get("metadata") or {}).get("semantic_plan")
            ).get("fact_column"),
            "INVOICE_DT_DMS_KEY",
        )

    def test_governed_channel_mixin_snapshot_carries_the_plan(self):
        from gateway.session_state import GovernedChannelSessionMixin

        class _Channel(GovernedChannelSessionMixin):
            platform_type = "teams"

        adapter = _Channel()
        adapter.bind_session("tenant-regrain", "teams-user-1")
        snapshot = self._store_via(adapter, adapter.session_id)
        metadata = snapshot.get("metadata") or {}
        self.assertTrue(
            metadata.get("semantic_plan"),
            "every channel must cache the governed plan, or the trend route "
            "silently degrades on that channel",
        )
        self.assertEqual(
            temporal_policy_from_plan(metadata.get("semantic_plan")).get("fact_column"),
            "INVOICE_DT_DMS_KEY",
        )
        self.assertEqual(metadata.get("db_config_id"), 7)

    def test_mixin_adopt_restores_the_plan_with_the_rows(self):
        from gateway.session_state import GovernedChannelSessionMixin

        class _Channel(GovernedChannelSessionMixin):
            platform_type = "teams"

        adapter = _Channel()
        adapter.bind_session("tenant-regrain", "teams-user-2")
        snapshot = self._store_via(adapter, adapter.session_id)

        adapter.last_result = {}
        restored = adapter.adopt_cached_snapshot(snapshot)
        self.assertTrue(
            restored.get("semantic_plan"),
            "an action on a restored snapshot must resolve against the same "
            "date role the answer was built from",
        )
        self.assertEqual(restored.get("contract_version"), "contract-9")


# ══════════════════════════════════════════════════════════════════════════════
# 8  The route through the real pipeline
# ══════════════════════════════════════════════════════════════════════════════
class TestPipelineRoute(unittest.TestCase):
    """Drives _handle_query_impl so the route is proven to actually fire.

    Source assertions cannot show that the insertion sits in a reachable branch,
    which is exactly the class of bug a 4,000-line function invites.
    """

    ACCOUNT = "acct-trend-regrain"
    SESSION = "acct-trend-regrain:portal:u1"

    def _run(self, follow_up, *, execute):
        import asyncio
        import contextlib
        from unittest.mock import patch

        import store.db as _db
        import core.query_pipeline as qp
        from core.result_cache import result_cache
        from gateway.base import PlatformEvent

        _db.init_db()
        result_cache.clear(self.SESSION)
        result_cache.store(
            self.SESSION,
            [{"TOTAL_REVENUE": 1234.0}],
            "What was my revenue for past 5 days?",
            PARENT_SQL,
            {},
            metadata={"semantic_plan": {"temporal_policies": [_policy()]}},
        )

        sent: dict = {}
        executed: list[str] = []

        class _Adapter:
            platform = "portal"
            session_id = self.SESSION
            thread_id = "thread-1"
            last_result_id = None

            def __init__(self):
                self.messages: list[str] = []

            def make_event(self, text):
                return PlatformEvent(
                    TestPipelineRoute.ACCOUNT, "u1", "c1", text, "portal",
                )

            async def send_message(self, event, text, **kwargs):
                self.messages.append(text)

            async def send_typing(self, *args, **kwargs):
                return None

            def add_to_history(self, **kwargs):
                return None

            def cache_result(self, *args, **kwargs):
                return None

        async def _capture_results(event, adapter, question, rows, sql, *args, **kwargs):
            sent.update({"question": question, "sql": sql, "rows": rows})

        class _Governed:
            def __init__(self, sql, rows):
                self.sql, self.rows = sql, rows

        def _fake_execute(credentials, db_type, sql, **kwargs):
            executed.append(sql)
            return _Governed(sql, execute(sql))

        adapter = _Adapter()
        with contextlib.ExitStack() as stack:
            for mock in (
                patch.object(qp, "get_state",
                             return_value={"state": "READY", "schema_dir": _tmpdir}),
                patch.object(qp, "get_client_db", return_value={
                    "db_type": "azure_sql", "credentials": {}, "name": "db"}),
                patch.object(qp.store, "get_client", return_value={
                    "account_id": self.ACCOUNT, "state": "READY", "name": "T"}),
                patch.object(qp, "load_known_tables",
                             return_value={"ERP.F_SALES_INVOICE", "ERP.DT_DMS"}),
                patch.object(qp, "load_schema_columns", return_value={
                    "ERP.F_SALES_INVOICE": {
                        "NET_REVENUE": "decimal", "INVOICE_DT_DMS_KEY": "int"},
                    "ERP.DT_DMS": {"DT_DMS_KEY": "int", "CAL_DATE": "date"}}),
                patch.object(qp, "_send_results", _capture_results),
                patch.object(qp, "execute_governed_query", _fake_execute),
                patch.object(qp, "resolve_provider",
                             return_value=("anthropic", "claude", "key", {})),
            ):
                stack.enter_context(mock)
            asyncio.run(qp._handle_query_impl(
                self.ACCOUNT, adapter.make_event(follow_up), adapter, follow_up,
                {"id": 1, "role": "admin", "email": "u@x.com",
                 "name": "U", "group_name": None},
            ))
        return sent, executed, adapter

    @staticmethod
    def _trend_rows(_sql):
        return [
            {"Invoice Date": f"2026-03-0{day}", "TOTAL_REVENUE": 200 + day}
            for day in range(4, 9)
        ]

    def test_provide_the_trend_answers_from_the_parent_sql(self):
        sent, executed, _adapter = self._run(
            "provide the trend", execute=self._trend_rows,
        )
        self.assertEqual(
            len(executed), 1,
            "the trend must run exactly one query — never re-enter the pipeline",
        )
        self.assertIn("GROUP BY", executed[0].upper())
        self.assertIn("CAL_DATE", executed[0])
        self.assertEqual(sent.get("sql"), executed[0])
        self.assertEqual(len(sent.get("rows") or []), 5)
        # The answer is titled by the parent question, not by "provide the trend".
        self.assertIn("revenue", str(sent.get("question") or "").lower())
        self.assertIn("by day", str(sent.get("question") or "").lower())

    def test_the_windows_where_clause_survives_the_route(self):
        _sent, executed, _adapter = self._run(
            "provide the trend", execute=self._trend_rows,
        )
        self.assertIn("DATEADD", executed[0].upper())
        self.assertIn("MAX(", executed[0].upper())

    def test_a_named_grain_reaches_the_executed_sql(self):
        _sent, executed, _adapter = self._run(
            "monthly trend", execute=self._trend_rows,
        )
        self.assertIn("DATEFROMPARTS", executed[0].upper())

    def test_bookkeeping_failure_never_discards_the_answer(self):
        """The query already hit the production database; the answer must ship."""
        import core.query_pipeline as qp
        from unittest.mock import patch

        with patch.object(qp, "_log_q", side_effect=RuntimeError("log is down")):
            sent, executed, _adapter = self._run(
                "provide the trend", execute=self._trend_rows,
            )
        self.assertEqual(len(executed), 1)
        self.assertEqual(len(sent.get("rows") or []), 5)


if __name__ == "__main__":
    unittest.main()
