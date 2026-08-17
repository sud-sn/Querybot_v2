"""
tests/test_business_date_anchor_cache.py

Relative-date questions must anchor on the data, not the clock (7ace868), and
the anchor must prove the date exists in the FACT, not just the calendar
(975214b — an unrestricted date dimension can hold rows with no facts and would
anchor to a date with zero results).

Both rules are right. Their cost is that every such question first asked the
database a question no SQL shape can answer without reading the fact: "what is
the newest date present here?". On a large fact with no index on the date key
that alone exhausted the 120 s statement timeout, and no rewrite could fix it —
the client cannot add the index.

The answer changes only when the warehouse loads. So resolve it once, cache it
per (account, fact, date key), and let the compiled SQL carry a literal window.
Still data-relative, still provenance-stamped, TTL-bounded — but nothing reads
the fact merely to discover its own latest date.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_tmpdir = tempfile.mkdtemp(prefix="querybot_anchor_")
os.environ.setdefault("DB_PATH", str(Path(_tmpdir) / "test_querybot.db"))
os.environ.setdefault("QUERYBOT_KEY_FILE", str(Path(_tmpdir) / "test_key"))

from core.date_anchor import (  # noqa: E402
    _coerce_anchor,
    anchor_for_policy,
    build_anchor_probe_sql,
    cached_anchor,
    clear_cache,
    resolve_business_anchor,
)
from core.pipeline_helpers import compile_governed_temporal_metric_sql  # noqa: E402

FACT = "EMDW_DMART.CUS_ORD_IVC_FCT"
DIM = "EMDW_DMART.DT_DMS"
COLUMNS = {
    FACT: {"CUS_IVC_DT_DMS_KEY": "int", "IVC_NET_AMT": "decimal"},
    DIM: {"DT_DMS_KEY": "int", "DMS_DT": "date"},
}
POLICY = {
    "kind": "last_n", "amount": 2, "unit": "day",
    "anchor_policy": "latest_available",
    "fact_table": FACT, "fact_column": "CUS_IVC_DT_DMS_KEY",
    "dimension_table": DIM, "dimension_key": "DT_DMS_KEY",
    "date_column": "DMS_DT", "date_key_type": "surrogate_fk",
    "role_alias": "invoice_date", "anchor_table": FACT,
    "business_role": "Invoice Date", "temporal_grain": "day",
}
ANCHOR = {
    "value": "2026-08-15", "fact_table": FACT,
    "fact_column": "CUS_IVC_DT_DMS_KEY", "date_column": "DMS_DT",
    "source": "probed_from_fact_rows",
}


def _context(anchor=None, **policy_overrides):
    policy = {**POLICY, **policy_overrides}
    context = {
        "question": "what is my revenue for the last 2 days",
        "semantic_plan": {
            "enabled": True, "fields": [], "joins": [],
            "required_tables": [FACT, DIM], "temporal_policies": [policy],
        },
        "analytical_request_plan": {"status": "compiled", "source_facts": [FACT]},
        "metric_formulas": [{
            "name": "Revenue", "formula_type": "expression",
            "sql_template": "SUM(fact_rows.[IVC_NET_AMT])", "base_table": FACT,
        }],
    }
    if anchor:
        context["resolved_date_anchor"] = anchor
    return context


def _compile(anchor=None, db="azure_sql", **policy_overrides):
    return compile_governed_temporal_metric_sql(
        db, set(COLUMNS), set(COLUMNS), COLUMNS,
        _context(anchor, **policy_overrides),
    )


class TestLiteralWindowWhenTheAnchorIsKnown(unittest.TestCase):

    def test_the_fact_is_not_read_to_find_its_own_latest_date(self):
        sql = _compile(ANCHOR)
        self.assertTrue(sql)
        self.assertNotIn("WITH anchor AS", sql)
        self.assertIn("2026-08-15", sql)

    def test_the_window_is_still_two_bounds_not_an_open_range(self):
        sql = _compile(ANCHOR)
        window = sql.split("window_keys AS (", 1)[1].split(")\nSELECT", 1)[0]
        self.assertIn("DATEADD(day, -2,", window)
        self.assertIn("<=", window)
        self.assertNotIn("CROSS JOIN anchor", window)

    def test_the_fact_is_still_reached_by_its_own_key(self):
        self.assertIn(
            "fact_rows.[CUS_IVC_DT_DMS_KEY] IN (SELECT business_date_key",
            _compile(ANCHOR),
        )

    def test_single_day_windows_use_the_literal_too(self):
        for kind in ("today", "yesterday"):
            with self.subTest(kind=kind):
                sql = _compile(ANCHOR, kind=kind, amount=1)
                self.assertTrue(sql)
                self.assertIn("2026-08-15", sql)
                self.assertNotIn("WITH anchor AS", sql)

    def test_trend_windows_use_the_literal_too(self):
        sql = compile_governed_temporal_metric_sql(
            "azure_sql", set(COLUMNS), set(COLUMNS), COLUMNS,
            {**_context(ANCHOR),
             "question": "revenue trend for the last 7 days"},
        )
        self.assertTrue(sql)
        self.assertIn("AS PERIOD", sql)
        self.assertNotIn("WITH anchor AS", sql)

    def test_other_dialects_render(self):
        for db, formula in (
            ("snowflake", 'SUM(fact_rows."IVC_NET_AMT")'),
            ("oracle", 'SUM(fact_rows."IVC_NET_AMT")'),
        ):
            with self.subTest(db=db):
                context = _context(ANCHOR)
                context["metric_formulas"][0]["sql_template"] = formula
                sql = compile_governed_temporal_metric_sql(
                    db, set(COLUMNS), set(COLUMNS), COLUMNS, context,
                )
                self.assertTrue(sql, f"{db} must compile")
                self.assertIn("2026-08-15", sql)


class TestTheAnchorCannotBeMisapplied(unittest.TestCase):
    """An anchor probed for one fact/key must never date a different one."""

    def test_no_anchor_keeps_the_in_query_form(self):
        sql = _compile()
        self.assertTrue(sql)
        self.assertIn("WITH anchor AS", sql)

    def test_an_anchor_from_another_fact_is_ignored(self):
        sql = _compile({**ANCHOR, "fact_table": "EMDW_DMART.PCH_ORD_RCT_FCT"})
        self.assertIn("WITH anchor AS", sql)
        self.assertNotIn("2026-08-15", sql)

    def test_an_anchor_from_another_date_role_is_ignored(self):
        sql = _compile({**ANCHOR, "fact_column": "LAST_MOD_DT_DMS_KEY"})
        self.assertIn("WITH anchor AS", sql)
        self.assertNotIn("2026-08-15", sql)

    def test_anchor_for_policy_matches_on_fact_and_key(self):
        self.assertTrue(anchor_for_policy(ANCHOR, POLICY))
        self.assertFalse(anchor_for_policy({**ANCHOR, "value": ""}, POLICY))
        self.assertFalse(anchor_for_policy(ANCHOR, {**POLICY, "fact_column": "OTHER"}))
        self.assertFalse(anchor_for_policy({}, POLICY))


class TestProbeAndCache(unittest.TestCase):

    def setUp(self):
        clear_cache()

    def tearDown(self):
        clear_cache()

    def test_the_probe_reaches_the_fact_by_key_not_by_join(self):
        sql = build_anchor_probe_sql(POLICY, "azure_sql")
        self.assertIn("MAX(anchor_date.[DMS_DT])", sql)
        self.assertIn("EXISTS", sql)
        self.assertNotIn("LEFT JOIN", sql)

    def test_a_native_date_role_needs_no_dimension(self):
        sql = build_anchor_probe_sql(
            {**POLICY, "dimension_table": "", "dimension_key": "",
             "date_column": "IVC_DATE"},
            "azure_sql",
        )
        self.assertIn("MAX(anchor_fact.[IVC_DATE])", sql)
        self.assertNotIn("DT_DMS", sql)

    def test_an_incomplete_policy_is_not_probed(self):
        self.assertEqual(build_anchor_probe_sql({}, "azure_sql"), "")
        self.assertEqual(
            build_anchor_probe_sql({"fact_table": FACT}, "azure_sql"), "",
        )

    def test_the_probe_runs_once_for_many_questions(self):
        calls: list[str] = []

        def _probe(sql):
            calls.append(sql)
            return [{"max_business_date": "2026-08-15"}]

        first = resolve_business_anchor("acct", POLICY, "azure_sql", _probe)
        second = resolve_business_anchor("acct", POLICY, "azure_sql", _probe)
        self.assertEqual(len(calls), 1, "the second question must hit the cache")
        self.assertEqual(first["value"], "2026-08-15")
        self.assertTrue(second["cached"])
        self.assertFalse(first["cached"])

    def test_a_failed_probe_falls_back_rather_than_guessing(self):
        def _boom(sql):
            raise RuntimeError("Query timeout expired")

        self.assertEqual(
            resolve_business_anchor("acct", POLICY, "azure_sql", _boom), {},
        )

    def test_an_unusable_probe_result_falls_back(self):
        for rows in ([], [{"max_business_date": None}], [{"x": "not a date"}]):
            with self.subTest(rows=rows):
                clear_cache()
                self.assertEqual(
                    resolve_business_anchor(
                        "acct", POLICY, "azure_sql", lambda sql: rows,
                    ),
                    {},
                )

    def test_only_data_relative_policies_are_probed(self):
        called: list[str] = []
        result = resolve_business_anchor(
            "acct", {**POLICY, "anchor_policy": "explicit"}, "azure_sql",
            lambda sql: called.append(sql) or [{"m": "2026-08-15"}],
        )
        self.assertEqual(result, {})
        self.assertEqual(called, [], "the clock/explicit case needs no probe")

    def test_the_cache_is_scoped_per_account_and_key(self):
        resolve_business_anchor(
            "acct-a", POLICY, "azure_sql", lambda sql: [{"m": "2026-08-15"}],
        )
        self.assertTrue(cached_anchor("acct-a", POLICY))
        self.assertFalse(cached_anchor("acct-b", POLICY))
        self.assertFalse(
            cached_anchor("acct-a", {**POLICY, "fact_column": "OTHER_KEY"}),
        )

    def test_clear_cache_is_account_scoped(self):
        for account in ("acct-a", "acct-b"):
            resolve_business_anchor(
                account, POLICY, "azure_sql", lambda sql: [{"m": "2026-08-15"}],
            )
        clear_cache("acct-a")
        self.assertFalse(cached_anchor("acct-a", POLICY))
        self.assertTrue(cached_anchor("acct-b", POLICY))

    def test_probed_values_are_normalised_to_iso_dates(self):
        self.assertEqual(_coerce_anchor(date(2026, 8, 15)), "2026-08-15")
        self.assertEqual(_coerce_anchor(datetime(2026, 8, 15, 3, 4)), "2026-08-15")
        self.assertEqual(_coerce_anchor("2026-08-15 00:00:00"), "2026-08-15")
        for bad in (None, "", "not a date", "15/08/2026"):
            with self.subTest(value=bad):
                self.assertEqual(_coerce_anchor(bad), "")

    def test_provenance_is_recorded(self):
        anchor = resolve_business_anchor(
            "acct", POLICY, "azure_sql", lambda sql: [{"m": "2026-08-15"}],
        )
        self.assertEqual(anchor["source"], "probed_from_fact_rows")
        self.assertEqual(anchor["fact_table"], FACT)
        self.assertIn("probed_at", anchor)
        self.assertGreater(anchor["ttl_seconds"], 0)


class TestValidatorAcceptsOnlyTheGovernedLiteral(unittest.TestCase):
    """The literal exemption must not let the model invent a date."""

    def _validate(self, sql, anchor=None):
        from core.validator import validate_sql_detailed

        return validate_sql_detailed(
            sql, set(COLUMNS), "azure_sql", set(COLUMNS), COLUMNS,
            _context(anchor),
        )

    def test_the_compiled_literal_window_validates(self):
        sql = _compile(ANCHOR)
        self.assertTrue(self._validate(sql, ANCHOR).ok)

    def test_a_literal_window_without_a_resolved_anchor_is_rejected(self):
        """No probe means no exemption — the in-query MAX is still required."""
        sql = _compile(ANCHOR)
        result = self._validate(sql, anchor=None)
        self.assertFalse(result.ok)
        self.assertIn(
            "temporal_anchor_missing",
            {error.get("code") for error in (result.errors or [])},
        )

    def test_an_invented_date_is_rejected(self):
        sql = _compile(ANCHOR).replace("2026-08-15", "2020-01-01")
        result = self._validate(sql, ANCHOR)
        self.assertFalse(result.ok)


class TestQueryTimeoutIsConfigurable(unittest.TestCase):

    def test_env_override_applies_when_the_db_config_is_silent(self):
        from core.schema import _query_timeout_seconds

        original = os.environ.get("QUERYBOT_QUERY_TIMEOUT_SECONDS")
        try:
            os.environ["QUERYBOT_QUERY_TIMEOUT_SECONDS"] = "300"
            self.assertEqual(_query_timeout_seconds({}), 300)
            # The db_config value still wins when present.
            self.assertEqual(
                _query_timeout_seconds({"query_timeout_seconds": 90}), 90,
            )
            os.environ["QUERYBOT_QUERY_TIMEOUT_SECONDS"] = "99999"
            self.assertEqual(_query_timeout_seconds({}), 600, "must stay clamped")
            os.environ["QUERYBOT_QUERY_TIMEOUT_SECONDS"] = "nonsense"
            from core.schema import _DEFAULT_QUERY_TIMEOUT_SECONDS
            self.assertEqual(
                _query_timeout_seconds({}), _DEFAULT_QUERY_TIMEOUT_SECONDS,
                "bad value -> default",
            )
        finally:
            if original is None:
                os.environ.pop("QUERYBOT_QUERY_TIMEOUT_SECONDS", None)
            else:
                os.environ["QUERYBOT_QUERY_TIMEOUT_SECONDS"] = original


class TestPipelineWiring(unittest.TestCase):

    def test_the_pipeline_resolves_the_anchor_before_compiling(self):
        source = (ROOT / "core" / "query_pipeline.py").read_text(encoding="utf-8")
        anchor_at = source.find("resolve_business_anchor(")
        compile_at = source.find("compile_governed_temporal_metric_sql(")
        self.assertGreater(anchor_at, 0)
        self.assertLess(anchor_at, compile_at)
        self.assertIn('"resolved_date_anchor"', source)
        self.assertIn("business_date_anchor", source)


if __name__ == "__main__":
    unittest.main()
