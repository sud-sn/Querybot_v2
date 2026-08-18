"""
tests/test_business_date_anchor_persistence.py

"The latest business date present in this fact" is a property of the WAREHOUSE
LOAD, not of a process. It was held only in a module-level dict, so every
restart, redeploy or crash threw it away and the next relative-date question
re-ran the probe.

Measured on the live EMCO warehouse: 800 seconds. A user asked "what is my
revenue for the last 2 days" and watched a spinner for thirteen minutes, with
no indication that anything was happening. That is what made the system
unusable when handed to a client for unattended testing — not the SQL, which
was correct throughout.

Persisting the anchor makes the probe run once per warehouse load rather than
once per process.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_tmpdir = tempfile.mkdtemp(prefix="querybot_anchor_persist_")
os.environ.setdefault("DB_PATH", str(Path(_tmpdir) / "test_querybot.db"))
os.environ.setdefault("QUERYBOT_KEY_FILE", str(Path(_tmpdir) / "test_key"))

import store  # noqa: E402
from core.date_anchor import (  # noqa: E402
    clear_cache,
    resolve_business_anchor,
)
from store.db import get_db, init_db  # noqa: E402

ACCOUNT = "anchor_persist_acct"
FACT = "EMDW_DMART.CUS_ORD_IVC_FCT"
KEY = "CUS_IVC_DT_DMS_KEY"
POLICY = {
    "kind": "last_n", "amount": 2, "unit": "day",
    "anchor_policy": "latest_available",
    "fact_table": FACT, "fact_column": KEY,
    "dimension_table": "EMDW_DMART.DT_DMS", "dimension_key": "DT_DMS_KEY",
    "date_column": "DMS_DT", "date_key_type": "surrogate_fk",
    "business_role": "Invoice Date",
}


def _store_module():
    """The store module core.date_anchor will actually import.

    Several suites delete and re-import `store`, so a module object bound at
    test-import time is not necessarily the one under test at run time.
    Resolving through sys.modules keeps the patch effective in a full run.
    """
    return sys.modules["store"]


def _probe_returning(value, calls=None):
    def _probe(sql):
        if calls is not None:
            calls.append(sql)
        if "out_of_order" in sql:
            return [{"out_of_order": 0}]
        return [{"max_business_date": value}]
    return _probe


class TestTheAnchorSurvivesARestart(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()
        with get_db() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO client (account_id, platform_type) VALUES (?,?)",
                (ACCOUNT, "zoom"),
            )

    def setUp(self):
        clear_cache(ACCOUNT, persistent=True)

    def tearDown(self):
        clear_cache(ACCOUNT, persistent=True)

    def _restart(self):
        """Everything a process restart destroys: the in-memory cache only."""
        clear_cache(ACCOUNT)

    def test_a_restart_does_not_re_probe_the_warehouse(self):
        calls: list[str] = []
        first = resolve_business_anchor(
            ACCOUNT, POLICY, "azure_sql", _probe_returning("2025-04-17", calls))
        self.assertEqual(first["value"], "2025-04-17")
        self.assertTrue(calls, "the first question must probe")

        self._restart()
        calls.clear()

        after = resolve_business_anchor(
            ACCOUNT, POLICY, "azure_sql", _probe_returning("2025-04-17", calls))
        self.assertEqual(
            calls, [],
            "the warehouse was re-probed after a restart — this is the 800s wait",
        )
        self.assertEqual(after["value"], "2025-04-17")
        self.assertEqual(after.get("restored_from"), "store")

    def test_the_restored_anchor_is_still_bound_to_its_own_fact_and_key(self):
        resolve_business_anchor(
            ACCOUNT, POLICY, "azure_sql", _probe_returning("2025-04-17"))
        self._restart()
        restored = resolve_business_anchor(
            ACCOUNT, POLICY, "azure_sql", _probe_returning("2025-04-17"))
        self.assertEqual(restored["fact_table"], FACT)
        self.assertEqual(restored["fact_column"], KEY)

    def test_another_fact_does_not_borrow_this_anchor(self):
        resolve_business_anchor(
            ACCOUNT, POLICY, "azure_sql", _probe_returning("2025-04-17"))
        self._restart()
        calls: list[str] = []
        other = {**POLICY, "fact_table": "EMDW_DMART.PCH_ORD_RCT_FCT",
                 "fact_column": "PCH_ORD_RCT_DT_DMS_KEY"}
        resolve_business_anchor(
            ACCOUNT, other, "azure_sql", _probe_returning("2025-03-01", calls))
        self.assertTrue(calls, "a different fact must resolve its own anchor")

    def test_another_account_does_not_borrow_this_anchor(self):
        resolve_business_anchor(
            ACCOUNT, POLICY, "azure_sql", _probe_returning("2025-04-17"))
        self._restart()
        self.assertEqual(
            _store_module().load_business_date_anchor("someone_else", FACT, KEY), {},
        )

    def test_clearing_persistently_forces_a_fresh_probe(self):
        # After a warehouse reload the stored value is stale and an admin needs
        # a way to make the next question re-probe.
        resolve_business_anchor(
            ACCOUNT, POLICY, "azure_sql", _probe_returning("2025-04-17"))
        clear_cache(ACCOUNT, persistent=True)
        calls: list[str] = []
        fresh = resolve_business_anchor(
            ACCOUNT, POLICY, "azure_sql", _probe_returning("2025-06-30", calls))
        self.assertTrue(calls)
        self.assertEqual(fresh["value"], "2025-06-30")

    def test_a_memory_only_clear_does_not_lose_the_stored_value(self):
        resolve_business_anchor(
            ACCOUNT, POLICY, "azure_sql", _probe_returning("2025-04-17"))
        clear_cache(ACCOUNT)
        self.assertEqual(
            _store_module().load_business_date_anchor(ACCOUNT, FACT, KEY).get("value"),
            "2025-04-17",
        )

    def test_a_store_failure_is_loud(self):
        # Falling back silently would reintroduce the 800s probe on every
        # restart with nothing in the log to explain it.
        from core.date_anchor import _persist_anchor
        with patch.object(_store_module(), "save_business_date_anchor",
                          side_effect=RuntimeError("disk full")), \
                patch("core.date_anchor.log") as logger:
            _persist_anchor(ACCOUNT, POLICY, {"value": "2025-04-17"})
        self.assertTrue(
            logger.warning.called,
            "a failed persist must say so — otherwise the 800s probe returns "
            "after every restart with nothing explaining why",
        )

    def test_a_store_failure_does_not_break_the_answer(self):
        with patch.object(_store_module(), "save_business_date_anchor",
                          side_effect=RuntimeError("disk full")):
            anchor = resolve_business_anchor(
                ACCOUNT, POLICY, "azure_sql", _probe_returning("2025-04-17"))
        self.assertEqual(anchor["value"], "2025-04-17")

    def test_an_unreadable_store_falls_back_to_probing(self):
        with patch.object(_store_module(), "load_business_date_anchor",
                          side_effect=RuntimeError("db locked")):
            calls: list[str] = []
            anchor = resolve_business_anchor(
                ACCOUNT, POLICY, "azure_sql", _probe_returning("2025-04-17", calls))
        self.assertEqual(anchor["value"], "2025-04-17")
        self.assertTrue(calls, "an unreadable store must not block the answer")


if __name__ == "__main__":
    unittest.main(verbosity=2)
