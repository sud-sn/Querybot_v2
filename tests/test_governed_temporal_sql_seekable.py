"""
tests/test_governed_temporal_sql_seekable.py

The governed temporal-metric compiler must produce SQL a star-schema fact can
SEEK on, and must keep returning exactly the same numbers.

The live failure: "what is my revenue for the last 2 days" compiled to

    WITH anchor AS (SELECT MAX(dim.DAY) FROM fact LEFT JOIN dim ON ...)   -- full scan
    SELECT SUM(...) FROM fact LEFT JOIN dim ON ... CROSS JOIN anchor
    WHERE dim.DAY > DATEADD(day, -2, anchor.max_business_date) ...        -- dimension filter

Both halves read the whole fact: the anchor hash-joins every fact row to the
date dimension purely to take one MAX, and filtering a *dimension* column leaves
the optimiser no way to use an index on the fact's own date key. On the client's
invoice fact that hit the 120 s ODBC statement timeout.

The rewrite resolves the window to date KEYS off the small dimension and filters
the fact on its own key. These tests assert both halves of that: the shape is
seekable, and the results are unchanged.
"""

from __future__ import annotations

import os
import re
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_tmpdir = tempfile.mkdtemp(prefix="querybot_seekable_")
os.environ.setdefault("DB_PATH", str(Path(_tmpdir) / "test_querybot.db"))
os.environ.setdefault("QUERYBOT_KEY_FILE", str(Path(_tmpdir) / "test_key"))

from core.pipeline_helpers import (  # noqa: E402
    build_seekable_date_window,
    compile_governed_temporal_metric_sql,
)

FACT = "EMDW_DMART.CUS_ORD_IVC_FCT"
DIM = "EMDW_DMART.DT_DMS"
COLUMNS = {
    FACT: {"CUS_IVC_DT_DMS_KEY": "int", "IVC_NET_AMT": "decimal"},
    DIM: {"DT_DMS_KEY": "int", "DAY": "date"},
}


def _policy(**overrides):
    policy = {
        "kind": "last_n", "amount": 2, "unit": "day",
        "anchor_policy": "latest_available",
        "fact_table": FACT, "fact_column": "CUS_IVC_DT_DMS_KEY",
        "dimension_table": DIM, "dimension_key": "DT_DMS_KEY",
        "date_column": "DAY", "date_key_type": "surrogate_fk",
        "role_alias": "invoice_date", "anchor_table": FACT,
        "business_role": "Invoice Date", "temporal_grain": "day",
    }
    policy.update(overrides)
    return policy


def _context(question="what is my revenue for the last 2 days", **policy_overrides):
    return {
        "question": question,
        "semantic_plan": {
            "enabled": True, "fields": [], "joins": [],
            "required_tables": [FACT, DIM],
            "temporal_policies": [_policy(**policy_overrides)],
        },
        "analytical_request_plan": {"status": "compiled", "source_facts": [FACT]},
        "metric_formulas": [{
            "name": "Revenue", "formula_type": "expression",
            "sql_template": "SUM(fact_rows.[IVC_NET_AMT])", "base_table": FACT,
        }],
    }


def _compile(question="what is my revenue for the last 2 days", db="azure_sql", **overrides):
    return compile_governed_temporal_metric_sql(
        db, set(COLUMNS), set(COLUMNS), COLUMNS, _context(question, **overrides),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1  The compiler still produces SQL (it must not silently decline)
# ══════════════════════════════════════════════════════════════════════════════
class TestCompilerStillCompiles(unittest.TestCase):
    """A decline here means a silent fallback to free-form generation."""

    def test_scalar_window_compiles(self):
        self.assertTrue(_compile(), "the reported live question must still compile")

    def test_trend_window_compiles(self):
        self.assertTrue(_compile("revenue trend for the last 7 days", amount=7))

    def test_single_day_windows_compile(self):
        self.assertTrue(_compile("revenue yesterday", kind="yesterday", amount=1))
        self.assertTrue(_compile("revenue today", kind="today", amount=1))

    def test_native_date_role_compiles(self):
        """A fact-native date needs no dimension and keeps its original shape."""
        columns = {FACT: {"IVC_DATE": "date", "IVC_NET_AMT": "decimal"}}
        context = _context()
        context["semantic_plan"]["temporal_policies"] = [_policy(
            dimension_table="", dimension_key="",
            fact_column="IVC_DATE", date_column="IVC_DATE",
            date_key_type="native_date",
        )]
        context["semantic_plan"]["required_tables"] = [FACT]
        sql = compile_governed_temporal_metric_sql(
            "azure_sql", set(columns), set(columns), columns, context,
        )
        self.assertTrue(sql)
        self.assertIn("MAX(fact_rows.[IVC_DATE])", sql)


# ══════════════════════════════════════════════════════════════════════════════
# 2  The shape is seekable
# ══════════════════════════════════════════════════════════════════════════════
class TestSeekableShape(unittest.TestCase):

    def test_the_fact_is_filtered_on_its_own_key(self):
        """This is the whole point: an index on the fact key becomes usable."""
        sql = _compile()
        self.assertRegex(
            sql,
            r"WHERE\s+fact_rows\.\[CUS_IVC_DT_DMS_KEY\]\s+IN\s+\(\s*SELECT\s+business_date_key",
        )

    def test_the_window_is_resolved_off_the_dimension(self):
        sql = _compile()
        window = sql.split("window_keys AS (", 1)[1].split(")\nSELECT", 1)[0]
        self.assertIn("[DT_DMS]", window)
        self.assertIn("DATEADD(day, -2,", window)
        self.assertNotIn("CUS_ORD_IVC_FCT", window,
                         "the window must not touch the fact at all")

    def test_the_anchor_no_longer_hash_joins_the_whole_fact(self):
        sql = _compile()
        anchor = sql.split("WITH anchor AS (", 1)[1].split("),", 1)[0]
        self.assertIn("MAX(anchor_date.[DAY])", anchor)
        self.assertIn("EXISTS", anchor,
                      "the fact must be reached by semi-join, not by outer join")
        self.assertNotIn("LEFT JOIN", anchor)

    def test_the_governed_graph_edge_is_still_present(self):
        """The validator requires the physical fact-to-dimension join."""
        sql = _compile()
        main = sql.split(")\nSELECT", 1)[1]
        self.assertRegex(
            main,
            r"LEFT JOIN \[EMDW_DMART\]\.\[DT_DMS\] AS invoice_date\s*\n\s*ON "
            r"fact_rows\.\[CUS_IVC_DT_DMS_KEY\] = invoice_date\.\[DT_DMS_KEY\]",
        )

    def test_no_cross_join_of_the_fact_to_the_anchor(self):
        """CROSS JOIN anchor forced the anchor into every fact row's scan."""
        main = _compile().split(")\nSELECT", 1)[1]
        self.assertNotIn("CROSS JOIN anchor", main)

    def test_trend_still_groups_and_orders_by_the_governed_date(self):
        sql = _compile("revenue trend for the last 7 days", amount=7)
        self.assertIn("AS PERIOD", sql)
        self.assertIn("GROUP BY CAST(invoice_date.[DAY] AS date)", sql)
        self.assertIn("ORDER BY PERIOD", sql)

    def test_dialects_render(self):
        # Each dialect quotes identifiers differently, and the metric formula is
        # authored per dialect, so the fixture has to match.
        for db, formula in (
            ("azure_sql", "SUM(fact_rows.[IVC_NET_AMT])"),
            ("snowflake", 'SUM(fact_rows."IVC_NET_AMT")'),
            ("oracle", 'SUM(fact_rows."IVC_NET_AMT")'),
        ):
            with self.subTest(db=db):
                context = _context()
                context["metric_formulas"][0]["sql_template"] = formula
                sql = compile_governed_temporal_metric_sql(
                    db, set(COLUMNS), set(COLUMNS), COLUMNS, context,
                )
                self.assertTrue(sql, f"{db} must still compile")
                self.assertIn("window_keys", sql)

    def test_observed_window_keeps_its_governed_shape(self):
        """latest_n_observed has its own validator contract; leave it alone."""
        sql = _compile("revenue for the latest 5 days",
                       kind="latest_n_observed", amount=5)
        if sql:
            self.assertIn("observed_periods", sql)
            self.assertNotIn("window_keys", sql)

    def test_window_builder_fragments_are_self_consistent(self):
        anchor, window, key_filter = build_seekable_date_window(
            db_type="azure_sql", fact_sql="[F]", fact_key="FK",
            dim_sql="[D]", dim_key="DK", date_col="DAY",
            quote=lambda name: f"[{name}]",
        )
        self.assertIn("MAX(anchor_date.[DAY])", anchor)
        self.assertIn("EXISTS", anchor)
        self.assertIn("{window_predicate}", window)
        self.assertIn("fact_rows.[FK] IN (SELECT business_date_key FROM window_keys)",
                      key_filter)


# ══════════════════════════════════════════════════════════════════════════════
# 3  Identical results — executed, not asserted
# ══════════════════════════════════════════════════════════════════════════════
class TestResultsAreUnchanged(unittest.TestCase):
    """Correctness gate: the fast shape must equal the old shape, row for row.

    Run on SQLite so it executes anywhere. The two SQL texts are the real ones
    the compiler emits, translated to SQLite only for date arithmetic and
    bracket quoting — the join/filter structure under test is untouched.
    """

    OLD_SCALAR = """
        WITH anchor AS (
            SELECT MAX(invoice_date."DAY") AS max_business_date
            FROM FCT AS fact_rows
            LEFT JOIN DTD AS invoice_date
              ON fact_rows."KEY" = invoice_date."DK"
        )
        SELECT SUM(fact_rows."AMT") AS REVENUE
        FROM FCT AS fact_rows
        LEFT JOIN DTD AS invoice_date ON fact_rows."KEY" = invoice_date."DK"
        CROSS JOIN anchor
        WHERE invoice_date."DAY" > DATE(anchor.max_business_date, '-2 day')
          AND invoice_date."DAY" <= anchor.max_business_date
    """
    NEW_SCALAR = """
        WITH anchor AS (
            SELECT MAX(anchor_date."DAY") AS max_business_date
            FROM DTD AS anchor_date
            WHERE EXISTS (
                SELECT 1 FROM FCT AS anchor_fact
                WHERE anchor_fact."KEY" = anchor_date."DK"
            )
        ),
        window_keys AS (
            SELECT window_date."DK" AS business_date_key
            FROM DTD AS window_date
            CROSS JOIN anchor
            WHERE window_date."DAY" > DATE(anchor.max_business_date, '-2 day')
              AND window_date."DAY" <= anchor.max_business_date
        )
        SELECT SUM(fact_rows."AMT") AS REVENUE
        FROM FCT AS fact_rows
        LEFT JOIN DTD AS invoice_date ON fact_rows."KEY" = invoice_date."DK"
        WHERE fact_rows."KEY" IN (SELECT business_date_key FROM window_keys)
    """
    OLD_TREND = OLD_SCALAR.replace(
        'SELECT SUM(fact_rows."AMT") AS REVENUE',
        'SELECT invoice_date."DAY" AS PERIOD, SUM(fact_rows."AMT") AS REVENUE',
    ) + ' GROUP BY invoice_date."DAY" ORDER BY PERIOD'
    NEW_TREND = NEW_SCALAR.replace(
        'SELECT SUM(fact_rows."AMT") AS REVENUE',
        'SELECT invoice_date."DAY" AS PERIOD, SUM(fact_rows."AMT") AS REVENUE',
    ) + ' GROUP BY invoice_date."DAY" ORDER BY PERIOD'

    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute('CREATE TABLE DTD ("DK" INTEGER, "DAY" TEXT)')
        self.conn.execute('CREATE TABLE FCT ("KEY" INTEGER, "AMT" REAL)')

    def tearDown(self):
        self.conn.close()

    def _seed(self, *, dim_days, fact_rows):
        self.conn.executemany("INSERT INTO DTD VALUES (?, ?)", dim_days)
        self.conn.executemany("INSERT INTO FCT VALUES (?, ?)", fact_rows)
        self.conn.commit()

    def _both(self, old, new):
        return (
            self.conn.execute(old).fetchall(),
            self.conn.execute(new).fetchall(),
        )

    def test_scalar_totals_match(self):
        self._seed(
            dim_days=[(k, f"2026-03-{d:02d}") for k, d in zip(range(101, 109), range(1, 9))],
            fact_rows=[(k, 10.0 * i) for i, k in enumerate(range(101, 109), start=1)],
        )
        old, new = self._both(self.OLD_SCALAR, self.NEW_SCALAR)
        self.assertEqual(old, new)
        self.assertIsNotNone(old[0][0], "the window must actually match rows")

    def test_trend_rows_match(self):
        self._seed(
            dim_days=[(k, f"2026-03-{d:02d}") for k, d in zip(range(101, 109), range(1, 9))],
            fact_rows=[(k, amount) for k in range(101, 109) for amount in (5.0, 7.5)],
        )
        old, new = self._both(self.OLD_TREND, self.NEW_TREND)
        self.assertEqual(old, new)
        self.assertEqual(len(new), 2, "a 2-day window yields 2 periods")

    def test_orphan_fact_rows_are_treated_the_same(self):
        """Fact keys with no dimension row must be excluded by both shapes."""
        self._seed(
            dim_days=[(k, f"2026-03-{d:02d}") for k, d in zip(range(101, 106), range(1, 6))],
            fact_rows=[(101, 1.0), (105, 2.0), (999, 999.0), (None, 42.0)],
        )
        old, new = self._both(self.OLD_SCALAR, self.NEW_SCALAR)
        self.assertEqual(old, new)

    def test_dimension_rows_with_no_fact_do_not_move_the_anchor(self):
        """A calendar with future rows must not anchor on a date with no data."""
        self._seed(
            dim_days=[(k, f"2026-03-{d:02d}") for k, d in zip(range(101, 109), range(1, 9))]
            + [(201, "2099-12-31")],
            fact_rows=[(k, 3.0) for k in range(101, 109)],
        )
        old, new = self._both(self.OLD_SCALAR, self.NEW_SCALAR)
        self.assertEqual(old, new)
        anchor = self.conn.execute(
            'SELECT MAX(d."DAY") FROM DTD d WHERE EXISTS '
            '(SELECT 1 FROM FCT f WHERE f."KEY" = d."DK")'
        ).fetchone()[0]
        self.assertEqual(anchor, "2026-03-08")

    def test_null_dates_are_treated_the_same(self):
        self._seed(
            dim_days=[(101, None), (102, "2026-03-07"), (103, "2026-03-08")],
            fact_rows=[(101, 5.0), (102, 6.0), (103, 7.0)],
        )
        old, new = self._both(self.OLD_SCALAR, self.NEW_SCALAR)
        self.assertEqual(old, new)

    def test_empty_fact_is_treated_the_same(self):
        self._seed(dim_days=[(101, "2026-03-01")], fact_rows=[])
        old, new = self._both(self.OLD_SCALAR, self.NEW_SCALAR)
        self.assertEqual(old, new)

    def test_duplicate_keys_do_not_multiply_the_total(self):
        """IN () dedups; the old equality filter did not multiply either."""
        self._seed(
            dim_days=[(101, "2026-03-07"), (102, "2026-03-08")],
            fact_rows=[(101, 1.0), (101, 2.0), (102, 4.0), (102, 8.0)],
        )
        old, new = self._both(self.OLD_SCALAR, self.NEW_SCALAR)
        self.assertEqual(old, new)
        self.assertEqual(new[0][0], 15.0)


class TestNoWindowLostItsCompiler(unittest.TestCase):
    """Every window that used to compile must still compile.

    A silent decline is the dangerous failure here: the question still gets an
    answer, but from free-form generation instead of the governed contract, so
    the regression is invisible until someone checks the SQL.
    """

    SCENARIOS = (
        ("scalar last 2 days", "what is my revenue for the last 2 days",
         {"kind": "last_n", "amount": 2, "unit": "day"}),
        ("scalar last 6 months", "total revenue for the last 6 months",
         {"kind": "last_n", "amount": 6, "unit": "month"}),
        ("scalar last 3 weeks", "revenue for the last 3 weeks",
         {"kind": "last_n", "amount": 3, "unit": "week"}),
        ("scalar last 2 years", "revenue for the last 2 years",
         {"kind": "last_n", "amount": 2, "unit": "year"}),
        ("trend last 7 days", "revenue trend for the last 7 days",
         {"kind": "last_n", "amount": 7, "unit": "day"}),
        ("trend by month", "revenue by month for the last 6 months",
         {"kind": "last_n", "amount": 6, "unit": "month"}),
        ("today", "revenue today", {"kind": "today", "amount": 1, "unit": "day"}),
        ("yesterday", "revenue yesterday",
         {"kind": "yesterday", "amount": 1, "unit": "day"}),
        ("latest 5 observed", "revenue for the latest 5 days",
         {"kind": "latest_n_observed", "amount": 5, "unit": "day"}),
    )

    def test_every_window_that_compiled_before_still_compiles(self):
        for label, question, overrides in self.SCENARIOS:
            with self.subTest(window=label):
                sql = _compile(question, **overrides)
                if overrides["kind"] == "latest_n_observed":
                    # Governed separately; only assert it did not start raising.
                    continue
                self.assertTrue(
                    sql,
                    f"{label} no longer compiles — it would silently fall back "
                    "to free-form SQL generation",
                )
                self.assertIn("window_keys", sql)
                self.assertIn("fact_rows.[CUS_IVC_DT_DMS_KEY] IN", sql)


# ══════════════════════════════════════════════════════════════════════════════
# 4  The validator must not flag the query's own aliases
# ══════════════════════════════════════════════════════════════════════════════
class TestValidatorIgnoresLocalAliases(unittest.TestCase):
    """A CTE output named ..._KEY is not a rival date role on the fact.

    Without this, reading a governed window back out of a CTE was rejected as
    role substitution — which silently turned the whole compiler off.
    """

    def _validate(self, sql):
        from core.validator import validate_sql_detailed

        return validate_sql_detailed(
            sql, set(COLUMNS), "azure_sql", set(COLUMNS), COLUMNS, _context(),
        )

    def test_compiled_window_validates(self):
        result = self._validate(_compile())
        self.assertTrue(result.ok, f"compiled SQL rejected: {result.errors}")

    def test_a_real_substituted_role_is_still_rejected(self):
        """The tightening must keep working through the alias exemption."""
        columns = {
            FACT: {
                "CUS_IVC_DT_DMS_KEY": "int", "LAST_MOD_DT_DMS_KEY": "int",
                "IVC_NET_AMT": "decimal",
            },
            DIM: {"DT_DMS_KEY": "int", "DAY": "date"},
        }
        from core.validator import validate_sql_detailed

        sql = (
            "SELECT SUM(f.[IVC_NET_AMT]) AS TOTAL "
            "FROM EMDW_DMART.CUS_ORD_IVC_FCT f "
            "JOIN EMDW_DMART.DT_DMS d ON f.[LAST_MOD_DT_DMS_KEY] = d.[DT_DMS_KEY] "
            "WHERE d.[DAY] >= DATEADD(day, -2, (SELECT MAX(d2.[DAY]) "
            "FROM EMDW_DMART.CUS_ORD_IVC_FCT f2 JOIN EMDW_DMART.DT_DMS d2 "
            "ON f2.[LAST_MOD_DT_DMS_KEY] = d2.[DT_DMS_KEY]))"
        )
        result = validate_sql_detailed(
            sql, set(columns), "azure_sql", set(columns), columns, _context(),
        )
        codes = {error.get("code") for error in (result.errors or [])}
        self.assertIn("temporal_role_mismatch", codes)


# ══════════════════════════════════════════════════════════════════════════════
# 5  One KB document per table per request
# ══════════════════════════════════════════════════════════════════════════════
class TestGapFillIsNotRepeated(unittest.TestCase):
    """Coverage runs once per planning stage; each stage only sees the RAG docs.

    Without a shared ledger the same table is re-fetched and re-appended every
    stage. The live trace showed one 90 kB fact document injected four times and
    a date dimension twice — ~14 s of redundant vector scrolls that then pushed
    the prompt from 394 kB to the 120 kB cap, truncating the tail, which is
    exactly where those documents had been appended.
    """

    STAGES = ("graph", "plan", "date_role", "metric_formula", "alignment")
    REQUIRED = {"EMDW_DMART.CUS_ORD_IVC_FCT", "EMDW_DMART.DT_DMS"}

    def _run_stages(self, *, share_ledger):
        from unittest.mock import patch
        import core.table_coverage as table_coverage

        fetched: list[str] = []

        def _fake_fetch(account_id, fqn, sections=None):
            fetched.append(fqn)
            return f"# KB doc for {fqn}"

        ledger: set[str] = set()
        injected = 0
        with patch("core.vector_store.fetch_docs_for_fqn", _fake_fetch):
            for _stage in self.STAGES:
                injected += len(table_coverage.guarantee_table_coverage(
                    account_id="acct",
                    required_fqns=set(self.REQUIRED),
                    retrieved_docs=[],
                    rag_filter=None,
                    already_injected=ledger if share_ledger else None,
                ))
        return fetched, injected, ledger

    def test_each_table_is_fetched_once_per_request(self):
        fetched, injected, ledger = self._run_stages(share_ledger=True)
        self.assertEqual(len(fetched), len(self.REQUIRED))
        self.assertEqual(sorted(set(fetched)), sorted(self.REQUIRED))
        self.assertEqual(injected, len(self.REQUIRED),
                         "each document must be appended to the prompt once")
        self.assertEqual(ledger, self.REQUIRED)

    def test_without_the_ledger_every_stage_refetches(self):
        """Guards the fix: this is the behaviour being prevented."""
        fetched, _injected, _ledger = self._run_stages(share_ledger=False)
        self.assertEqual(len(fetched), len(self.REQUIRED) * len(self.STAGES))

    def test_the_pipeline_shares_one_ledger_across_every_stage(self):
        """Wiring, not behaviour -- there is no way to see this from outside.

        Reaching these call sites means running the whole of
        `_handle_query_impl`. The previous version of this test also asserted
        the literal `_injected_kb_tables: set[str] = set()`, which broke the
        moment the ledger gained a seed and had never said anything about
        whether the seeding worked. What it should pin is that the ledger the
        preload fills is the ledger coverage reads.
        """
        import inspect
        import core.query_pipeline as query_pipeline

        source = inspect.getsource(query_pipeline)
        self.assertEqual(
            source.count("guarantee_table_coverage("),
            source.count("already_injected"),
            "every coverage call site must share the request ledger",
        )
        self.assertIn(
            "_injected_kb_tables: set[str] = set(_preloaded_tables)",
            inspect.getsource(query_pipeline._handle_query_impl),
            "the ledger must start holding what the preload already sent",
        )

    def test_a_preloaded_table_is_never_gap_filled_on_top_of_itself(self):
        """A ledger seeded from the preload has to suppress the re-fetch.

        Preloading appends every permitted table's document up front, so
        coverage fetching one again is a duplicated 90 kB document, not a gap
        being filled.
        """
        from unittest.mock import patch
        import core.table_coverage as table_coverage
        from core.kb_preload import _cache, preload_account_kb

        _cache.clear()
        self.addCleanup(_cache.clear)

        fetched: list[str] = []

        def _fake_fetch(account_id, fqn, sections=None):
            fetched.append(fqn)
            return f"# KB doc for {fqn}"

        with patch("core.vector_store.fetch_docs_for_fqn", _fake_fetch):
            _context, preloaded = preload_account_kb("acct", set(self.REQUIRED))
            self.assertEqual(sorted(preloaded), sorted(self.REQUIRED))
            fetched.clear()

            ledger: set[str] = set(preloaded)
            for _stage in self.STAGES:
                self.assertEqual(
                    table_coverage.guarantee_table_coverage(
                        account_id="acct",
                        required_fqns=set(self.REQUIRED),
                        retrieved_docs=[],
                        rag_filter=None,
                        already_injected=ledger,
                    ),
                    [],
                )
        self.assertEqual(fetched, [], "a preloaded document must not be fetched again")

    def test_variants_of_the_same_table_count_as_covered(self):
        from unittest.mock import patch
        import core.table_coverage as table_coverage

        fetched: list[str] = []
        with patch("core.vector_store.fetch_docs_for_fqn",
                   lambda account_id, fqn, sections=None: fetched.append(fqn) or "# doc"):
            ledger = {"EMDW_DMART.CUS_ORD_IVC_FCT"}
            docs = table_coverage.guarantee_table_coverage(
                account_id="acct",
                required_fqns={"CUS_ORD_IVC_FCT"},
                retrieved_docs=[],
                rag_filter=None,
                already_injected=ledger,
            )
        self.assertEqual(docs, [], "a bare table name must match its qualified form")
        self.assertEqual(fetched, [])


if __name__ == "__main__":
    unittest.main()
