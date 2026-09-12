# -*- coding: utf-8 -*-
"""tests/test_masking_never_fails_open.py

A salary column containing "$85,000" reached a KB sample unmasked, silently.

Five strategies in core/masking.py -- coordinate (both the non-deterministic
and the deterministic path), birthdate, date_shift, salary and numeric_shift
-- each caught their OWN conversion failure internally and did `return
value`: the RAW, unmasked value, with no exception raised and no log line.

Because these functions returned normally instead of raising, the ONE safety
net that exists -- _mask_row's `except Exception: result[field] =
"[MASKED]"` -- never fired for exactly the failure it exists to catch. A
masking strategy silently declining to mask is indistinguishable, from the
caller's side, from masking having worked.

mask_rows() feeds core/schema.py's KB schema/sample discovery (the path that
puts sample values into an LLM prompt to help it write SQL) and
admin/routes.py's masking preview -- both are boundaries where "always masked
or refused" is the actual guarantee, not "masked when the value happens to
parse cleanly."

Concrete failure shapes, each reproduced against the unfixed functions before
this file existed:

    coordinate  "not-a-number"                 -- non-numeric latitude
    birthdate   "1985-03-02T00:00:00+05:00"     -- ISO timestamp with a
                                                    timezone offset; matches
                                                    none of the five formats
                                                    _shift_date tries
    salary      "$85,000"                       -- currency symbol and comma

The fix does not add a new safety net -- it makes the EXISTING one reachable:
each strategy now lets its conversion failure propagate, and _mask_row's
except clause (now with a log.warning naming the field and strategy, never
the value or str(exc) -- Python's own ValueError from float()/strptime()
embeds the unparseable string verbatim, which would turn the log itself into
the leak) substitutes "[MASKED]" exactly as it always was meant to.
"""

from __future__ import annotations

import logging
import unittest

from core.masking import _apply, _apply_det, _mask_row, _shift_date, _shift_numeric, mask_rows


class TheFiveStrategiesRaiseRatherThanReturnRaw(unittest.TestCase):
    """Unit tests directly on the strategy functions, independent of the
    outer safety net -- proves each one no longer swallows its own failure."""

    def test_shift_date_raises_on_an_unrecognized_format(self):
        with self.assertRaises(Exception):
            _shift_date("1985-03-02T00:00:00+05:00", max_days=730)

    def test_shift_date_raises_on_plain_garbage(self):
        with self.assertRaises(Exception):
            _shift_date("not a date at all")

    def test_shift_numeric_raises_on_a_currency_string(self):
        with self.assertRaises(Exception):
            _shift_numeric("$85,000", pct=0.15)

    def test_shift_numeric_raises_on_none_shaped_garbage(self):
        with self.assertRaises(Exception):
            _shift_numeric("not-a-number")

    def test_apply_coordinate_raises_on_a_non_numeric_value(self):
        with self.assertRaises(Exception):
            _apply("not-a-number", "coordinate", faker=None)

    def test_apply_det_coordinate_raises_on_a_non_numeric_value(self):
        import random

        with self.assertRaises(Exception):
            _apply_det("not-a-number", "coordinate", random.Random(1))

    def test_a_value_that_genuinely_converts_still_works(self):
        """No regression: the happy path for every one of these strategies
        must keep producing a shifted value, not start raising on input that
        was always fine."""
        self.assertIsInstance(_shift_date("1985-03-02", max_days=5), str)
        self.assertIsInstance(_shift_numeric("85000", pct=0.1), int)
        self.assertIsInstance(_apply("37.7749", "coordinate", faker=None), float)


class MaskRowSubstitutesMaskedInsteadOfLeaking(unittest.TestCase):
    """Executes _mask_row and mask_rows for real -- the actual caller path a
    KB sample or admin masking preview goes through -- and proves the
    substitution happens instead of the raw value passing through."""

    COLS = [
        {"name": "SALARY", "type": "varchar"},
        {"name": "BIRTHDATE", "type": "varchar"},
        {"name": "LAT", "type": "varchar"},
    ]
    STRATEGIES = {"SALARY": "salary", "BIRTHDATE": "birthdate", "LAT": "coordinate"}

    def test_every_unparseable_field_becomes_masked_not_raw(self):
        rows = [{
            "SALARY": "$85,000",
            "BIRTHDATE": "1985-03-02T00:00:00+05:00",
            "LAT": "not-a-number",
        }]
        out = mask_rows(rows, set(self.STRATEGIES), self.COLS,
                        strategy_overrides=self.STRATEGIES)
        self.assertEqual(out[0], {
            "SALARY": "[MASKED]", "BIRTHDATE": "[MASKED]", "LAT": "[MASKED]",
        })

    def test_the_deterministic_seeded_path_has_the_same_guarantee(self):
        """mask_rows(seed_key=...) is the path every real KB/admin caller
        actually uses -- FK-consistent masking needs a seed. Checked
        separately from the unseeded path because it calls _apply_det, a
        DIFFERENT function with its own copy of the same bug."""
        rows = [{
            "SALARY": "$85,000",
            "BIRTHDATE": "1985-03-02T00:00:00+05:00",
            "LAT": "not-a-number",
        }]
        out = mask_rows(rows, set(self.STRATEGIES), self.COLS,
                        seed_key="acct-123", strategy_overrides=self.STRATEGIES)
        self.assertEqual(out[0], {
            "SALARY": "[MASKED]", "BIRTHDATE": "[MASKED]", "LAT": "[MASKED]",
        })

    def test_a_genuinely_maskable_row_is_still_masked_normally(self):
        """No regression: values that DO convert must keep being shifted, not
        start reading as failures and collapsing to [MASKED]."""
        rows = [{"SALARY": 85000, "BIRTHDATE": "1985-03-02", "LAT": "37.7749"}]
        out = mask_rows(rows, set(self.STRATEGIES), self.COLS,
                        strategy_overrides=self.STRATEGIES)
        row = out[0]
        self.assertNotEqual(row["SALARY"], "[MASKED]")
        self.assertIsInstance(row["SALARY"], int)
        self.assertNotEqual(row["BIRTHDATE"], "[MASKED]")
        self.assertNotEqual(row["BIRTHDATE"], "1985-03-02")  # actually shifted
        self.assertNotEqual(row["LAT"], "[MASKED]")
        self.assertNotEqual(row["LAT"], 37.7749)  # actually shifted

    def test_one_bad_field_does_not_take_down_the_rest_of_the_row(self):
        rows = [{"SALARY": "$85,000", "BIRTHDATE": "1985-03-02", "LAT": "37.7749"}]
        out = mask_rows(rows, set(self.STRATEGIES), self.COLS,
                        strategy_overrides=self.STRATEGIES)
        row = out[0]
        self.assertEqual(row["SALARY"], "[MASKED]")
        self.assertNotEqual(row["BIRTHDATE"], "[MASKED]")
        self.assertNotEqual(row["LAT"], "[MASKED]")


class TheFailureIsLoggedWithoutBecomingANewLeak(unittest.TestCase):
    """A silent fail-open is one defect. Fixing it by logging str(exc) would
    be a second, more embarrassing one: Python's own ValueError from
    float("$85,000") or strptime() embeds the unparseable string verbatim in
    its message, so logging the exception text would put the very value
    masking failed to protect into the application log instead."""

    def test_a_masking_failure_is_logged_at_warning(self):
        with self.assertLogs("core.masking", level="WARNING") as captured:
            out = mask_rows(
                [{"SALARY": "$85,000"}], {"SALARY"},
                [{"name": "SALARY", "type": "varchar"}],
                strategy_overrides={"SALARY": "salary"},
            )
        self.assertEqual(out[0]["SALARY"], "[MASKED]")
        self.assertTrue(captured.output)
        joined = " ".join(captured.output)
        self.assertIn("SALARY", joined)
        self.assertIn("salary", joined)

    def test_the_log_line_never_contains_the_source_value(self):
        secret_value = "$UNIQUE_SECRET_85731_VALUE"
        with self.assertLogs("core.masking", level="WARNING") as captured:
            mask_rows(
                [{"SALARY": secret_value}], {"SALARY"},
                [{"name": "SALARY", "type": "varchar"}],
                strategy_overrides={"SALARY": "salary"},
            )
        joined = " ".join(captured.output)
        self.assertNotIn(secret_value, joined)
        self.assertNotIn("UNIQUE_SECRET_85731", joined)

    def test_a_successful_masking_logs_nothing_at_warning(self):
        """The log line marks a FAILURE, not every masking operation -- a
        chatty log here would bury the signal it exists to give."""
        import core.masking as masking_module

        original_level = logging.getLogger("core.masking").level
        logging.getLogger("core.masking").setLevel(logging.WARNING)
        handler = logging.Handler()
        records: list[logging.LogRecord] = []
        handler.emit = records.append
        logging.getLogger("core.masking").addHandler(handler)
        try:
            out = mask_rows(
                [{"SALARY": 85000}], {"SALARY"},
                [{"name": "SALARY", "type": "varchar"}],
                strategy_overrides={"SALARY": "salary"},
            )
        finally:
            logging.getLogger("core.masking").removeHandler(handler)
            logging.getLogger("core.masking").setLevel(original_level)
        self.assertNotEqual(out[0]["SALARY"], "[MASKED]")
        self.assertEqual(records, [])


if __name__ == "__main__":
    unittest.main()
