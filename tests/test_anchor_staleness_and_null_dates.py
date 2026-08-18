"""
tests/test_anchor_staleness_and_null_dates.py

Two defects found by adversarial review of the anchor persistence + cheap probe,
both of which produce a CONFIDENTLY WRONG date with no error anywhere.

1. NULL blinds the ordering check.
   build_key_order_check_sql decided whether the dimension's key order matches
   its date order, which is what licenses reading the anchor as MAX(fact.key)
   instead of a semi-join. It counted only pairs where both dates were non-NULL.
   A NULL-dated member -- the standard unknown/N-A row -- is skipped as `d`
   (NULL < prev_d is UNKNOWN) *and* becomes the next row's prev_d, so it erases
   the one comparison spanning it. Proven against SQLite: dimension
   (1,'2025-04-17'), (2,NULL), (3,'2020-01-01') with fact keys 1 and 3 returned
   out_of_order = 0, and the cheap probe then answered 2020-01-01 where the
   semi-join answers 2025-04-17. Five years stale, stamped
   source='probed_from_fact_rows', and persisted.

2. A persisted anchor that never ages out.
   The in-memory TTL bounded staleness within a process. Persisting it removed
   that bound entirely: every TTL expiry fell through to the store and
   re-armed the TTL from the same value, so the warehouse was probed once and
   never again. After the client's overnight reload every relative-date
   question would answer against the pre-reload date forever, silently
   excluding every newly loaded row -- the exact opposite of the rule the
   anchor exists to enforce.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_tmpdir = tempfile.mkdtemp(prefix="querybot_anchor_stale_")
os.environ.setdefault("DB_PATH", str(Path(_tmpdir) / "test_querybot.db"))
os.environ.setdefault("QUERYBOT_KEY_FILE", str(Path(_tmpdir) / "test_key"))

import core.date_anchor as anchor  # noqa: E402

FACT = "EMDW_DMART.CUS_ORD_IVC_FCT"
DIM = "EMDW_DMART.DT_DMS"
POLICY = {
    "fact_table": FACT, "fact_column": "CUS_IVC_DT_DMS_KEY",
    "dimension_table": DIM, "dimension_key": "DT_DMS_KEY",
    "date_column": "DMS_DT", "date_key_type": "surrogate_fk",
    "business_role": "Invoice Date",
}


def _sqlite_check(rows):
    """Run the generated order check against a real dimension."""
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE DT_DMS (DT_DMS_KEY INT, DMS_DT TEXT)")
    con.executemany("INSERT INTO DT_DMS VALUES (?,?)", rows)
    sql = anchor.build_key_order_check_sql(POLICY, "sqlite")
    sql = sql.replace("[", '"').replace("]", '"').replace(f'"{DIM}"', "DT_DMS")
    for token in (f'"{DIM}"', DIM, "EMDW_DMART.DT_DMS"):
        sql = sql.replace(token, "DT_DMS")
    return int(con.execute(sql).fetchone()[0])


class TestANullDateDisqualifiesTheCheapPath(unittest.TestCase):

    def test_a_clean_monotonic_dimension_still_qualifies(self):
        self.assertEqual(
            _sqlite_check([(1, "2025-01-01"), (2, "2025-01-02"), (3, "2025-01-03")]), 0,
        )

    def test_a_null_dated_member_is_counted_as_a_violation(self):
        # The live shape: an unknown/N-A member sitting between real dates.
        self.assertGreater(
            _sqlite_check([(1, "2025-04-17"), (2, None), (3, "2020-01-01")]), 0,
            "a NULL date blinded the ordering check and licensed the cheap probe",
        )

    def test_a_leading_unknown_member_also_disqualifies(self):
        self.assertGreater(_sqlite_check([(0, None), (1, "2025-01-01")]), 0)

    def test_a_genuine_inversion_is_still_caught(self):
        self.assertGreater(_sqlite_check([(1, "2025-01-02"), (2, "2025-01-01")]), 0)


class TestAStoredAnchorMustAgeOut(unittest.TestCase):

    def setUp(self):
        anchor.clear_cache()

    def tearDown(self):
        anchor.clear_cache()

    def _stored(self, age_hours):
        stamp = (datetime.utcnow() - timedelta(hours=age_hours)).isoformat(sep=" ")
        return {
            "value": "2025-04-17", "fact_table": FACT,
            "fact_column": "CUS_IVC_DT_DMS_KEY", "date_column": "DMS_DT",
            "source": "probed_from_fact_rows", "resolved_at": stamp,
        }

    def _read(self, stored):
        with patch("store.load_business_date_anchor", return_value=stored):
            return anchor._stored_anchor("acct", POLICY)

    def test_a_fresh_stored_anchor_is_served(self):
        self.assertEqual(self._read(self._stored(1)).get("value"), "2025-04-17")

    def test_a_stored_anchor_older_than_the_max_age_is_refused(self):
        self.assertEqual(
            self._read(self._stored(48)), {},
            "a day-old anchor was served, so a warehouse reload is never noticed",
        )

    def test_an_unreadable_timestamp_is_refused_rather_than_trusted(self):
        stale = {**self._stored(1), "resolved_at": "not a date"}
        self.assertEqual(self._read(stale), {})

    def test_the_max_age_is_configurable_and_can_be_disabled(self):
        with patch.dict(os.environ, {"QUERYBOT_DATE_ANCHOR_MAX_AGE_SECONDS": "0"}):
            self.assertEqual(
                self._read(self._stored(24 * 365)).get("value"), "2025-04-17",
                "explicitly disabling the max age must keep the stored value",
            )

    def test_age_is_measured_from_when_the_probe_ran(self):
        self.assertIsNone(anchor._anchor_age_seconds(""))
        self.assertIsNone(anchor._anchor_age_seconds("rubbish"))
        recent = anchor._anchor_age_seconds(
            (datetime.utcnow() - timedelta(seconds=90)).isoformat(sep=" "))
        self.assertIsNotNone(recent)
        self.assertLess(abs(recent - 90), 30)


class TestThereIsAWayToInvalidateIt(unittest.TestCase):
    """The stored row previously had no production caller that could clear it."""

    def test_an_admin_route_exists_to_force_a_reprobe(self):
        source = (ROOT / "admin" / "routes.py").read_text(encoding="utf-8")
        self.assertIn("date-roles/refresh-anchor", source)
        self.assertIn("clear_business_date_anchor", source)

    def test_the_store_clear_is_reachable_from_the_package(self):
        import store
        self.assertTrue(callable(getattr(store, "clear_business_date_anchor", None)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
