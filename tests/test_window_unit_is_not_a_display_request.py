"""
tests/test_window_unit_is_not_a_display_request.py

"what is my revenue for the last 2 days" asks for ONE NUMBER. The runtime plan
was carrying two required display dimensions for it, one of them
DT_DMS.DAY_NM — the weekday NAME — because the token "days" scored against a
date dimension reached through a date-role key.

The word is the UNIT of the window, not a request to display a date column.
This is the same failure the date-role gate fixes one layer up: a generic
temporal word read as a request for a specific named thing.

It is expensive. A required display field is one of the guards in
compile_governed_temporal_metric_sql, so the question left the governed
compiler for free-form generation, which loses the cached anchor and the
literal window:

    -- compiled: one pass, anchor already known
    WHERE business_date_key IN (SELECT ... BETWEEN '2026-06-29' AND '2026-06-30')

    -- free-form, observed live on this mart:
    WHERE inv.DMS_DT >= DATEADD(DAY, -2, (SELECT MAX(inv.DMS_DT) FROM fact ...))

The second scans the fact twice. On the 9.2M-row live warehouse with no index
on the date key that is what exhausted the statement timeout.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_tmpdir = tempfile.mkdtemp(prefix="querybot_windowunit_")
os.environ.setdefault("DB_PATH", str(Path(_tmpdir) / "test_querybot.db"))
os.environ.setdefault("QUERYBOT_KEY_FILE", str(Path(_tmpdir) / "test_key"))

from core.semantic_model import (  # noqa: E402
    _relative_window_terms,
    build_runtime_semantic_plan,
)

FACT = "EMDW_DMART.CUS_ORD_IVC_FCT"
DATE_DIM = "EMDW_DMART.DT_DMS"
WHS_DIM = "EMDW_DMART.WHS_DMS"

MODEL = {
    "tables": [{
        "qualified_name": FACT,
        "schema": "EMDW_DMART",
        "fields": [],
        "dimensions": [
            {   # the date dimension, reached by a date-role key
                "name": "Date",
                "source_key": "CUS_IVC_DT_DMS_KEY",
                "display_table": DATE_DIM,
                "display_column": "DAY_NM",
                "display_key": "DT_DMS_KEY",
                "confidence": 90,
            },
            {   # an ordinary business dimension, for contrast
                "name": "Warehouse",
                "source_key": "WHS_DMS_KEY",
                "display_table": WHS_DIM,
                "display_column": "WHS_NM",
                "display_key": "WHS_DMS_KEY",
                "confidence": 90,
            },
        ],
    }],
}


def _required_display(question):
    plan = build_runtime_semantic_plan(
        "", question=question, selected_schema="EMDW_DMART", model=MODEL,
    )
    return {
        f"{f.get('table')}.{f.get('column')}"
        for f in plan.get("fields") or []
        if f.get("display_required") and f.get("enforcement") != "optional"
    }


class TestTheWindowUnitDoesNotRequestADateColumn(unittest.TestCase):

    def test_a_scalar_window_question_requires_no_date_display_column(self):
        self.assertNotIn(
            f"{DATE_DIM}.DAY_NM", _required_display(
                "what is my revenue for the last 2 days"),
        )

    def test_the_same_holds_for_the_named_single_day_windows(self):
        for question in ("what is today's revenue", "revenue yesterday"):
            with self.subTest(question=question):
                self.assertNotIn(f"{DATE_DIM}.DAY_NM", _required_display(question))

    def test_an_explicit_grouping_request_still_gets_its_date_column(self):
        # "by day" is a real grain request and must survive untouched.
        self.assertIn(f"{DATE_DIM}.DAY_NM", _required_display("revenue by day"))

    def test_a_named_date_role_still_gets_its_column(self):
        self.assertIn(
            f"{DATE_DIM}.DAY_NM", _required_display("show revenue by invoice date"),
        )

    def test_a_grouping_request_inside_a_window_survives(self):
        # Both signals present: the window consumes "days", "by warehouse" is
        # independent evidence and must not be collateral damage.
        required = _required_display(
            "revenue by warehouse for the last 2 days")
        self.assertIn(f"{WHS_DIM}.WHS_NM", required)

    def test_non_date_dimensions_are_never_affected(self):
        self.assertIn(
            f"{WHS_DIM}.WHS_NM", _required_display("revenue by warehouse"),
        )


class TestWhichTokensTheWindowConsumes(unittest.TestCase):

    def test_it_claims_only_the_unit_of_a_real_window(self):
        self.assertEqual(
            _relative_window_terms("what is my revenue for the last 2 days"),
            {"day", "days"},
        )

    def test_a_question_with_no_window_claims_nothing(self):
        for question in ("revenue by day", "show revenue by invoice date",
                         "revenue by warehouse"):
            with self.subTest(question=question):
                self.assertEqual(_relative_window_terms(question), set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
