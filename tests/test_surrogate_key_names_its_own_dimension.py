"""
tests/test_surrogate_key_names_its_own_dimension.py

Catalogue check B1 — "total amount of confirmed purchase orders by profit
center" answered with CUSTOMER NAMES under a column headed PROFIT_CENTER.

Confidently. No validation error, no repair retry, 95/100 confidence: the SQL
was valid, every join was a real declared foreign key, and the grouping column
was a real business name. Only the business meaning was wrong, which is the
worst failure mode this system has.

_find_dimension_for_key returned the first dimension that merely CONTAINED the
column. On any snowflaked model that is decided by iteration order. CUS_DMS
carries PFT_CTR_DMS_KEY (customers belong to a profit centre) and sorts before
PFT_CTR_DMS, so the term "profit center" bound to CUS_DMS.CUS_NM.

A key identifies the dimension where it is the PRIMARY key. A dimension that
merely carries the column is a peer holding a foreign key to it. The name
convention was already checked, but only as a fallback the early return never
reached.
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

_tmpdir = tempfile.mkdtemp(prefix="querybot_dimkey_")
os.environ.setdefault("DB_PATH", str(Path(_tmpdir) / "test_querybot.db"))
os.environ.setdefault("QUERYBOT_KEY_FILE", str(Path(_tmpdir) / "test_key"))

from core.semantic_model import _find_dimension_for_key  # noqa: E402


def _table(columns, pk):
    return {
        "columns": [{"name": c, "type": "int" if c.endswith("_KEY") else "nvarchar"}
                    for c in columns],
        "pk_columns": list(pk),
    }


# Deliberately ordered so the WRONG answer comes first alphabetically.
SCHEMA = {
    "EMDW_DMART.CUS_DMS": _table(
        ["CUS_DMS_KEY", "CUS_CD", "CUS_NM", "PFT_CTR_DMS_KEY", "CTY_NM"],
        ["CUS_DMS_KEY"],
    ),
    "EMDW_DMART.PFT_CTR_CUS_DAT": _table(   # the bridge: composite key
        ["PFT_CTR_DMS_KEY", "CUS_DMS_KEY", "PRY_FLG"],
        ["PFT_CTR_DMS_KEY", "CUS_DMS_KEY"],
    ),
    "EMDW_DMART.PFT_CTR_DMS": _table(
        ["PFT_CTR_DMS_KEY", "PFT_CTR_CD", "PFT_CTR_NM", "RGN_NM"],
        ["PFT_CTR_DMS_KEY"],
    ),
    "EMDW_DMART.WHS_DMS": _table(
        ["WHS_DMS_KEY", "WHS_CD", "WHS_NM", "PFT_CTR_DMS_KEY"],
        ["WHS_DMS_KEY"],
    ),
}


def _resolved(key, schema=None):
    found = _find_dimension_for_key(schema if schema is not None else SCHEMA, key)
    return found[0] if found else None


class TestTheKeyResolvesToItsOwnDimension(unittest.TestCase):

    def test_the_profit_centre_key_resolves_to_the_profit_centre_dimension(self):
        self.assertEqual(_resolved("PFT_CTR_DMS_KEY"), "EMDW_DMART.PFT_CTR_DMS")

    def test_it_is_not_captured_by_a_peer_that_merely_holds_the_key(self):
        self.assertNotEqual(_resolved("PFT_CTR_DMS_KEY"), "EMDW_DMART.CUS_DMS")
        self.assertNotEqual(_resolved("PFT_CTR_DMS_KEY"), "EMDW_DMART.WHS_DMS")

    def test_a_bridge_never_identifies_a_dimension(self):
        # Its composite key identifies a pairing, not an entity.
        self.assertNotEqual(_resolved("PFT_CTR_DMS_KEY"), "EMDW_DMART.PFT_CTR_CUS_DAT")
        self.assertNotEqual(_resolved("CUS_DMS_KEY"), "EMDW_DMART.PFT_CTR_CUS_DAT")

    def test_the_customer_key_still_resolves_to_the_customer_dimension(self):
        self.assertEqual(_resolved("CUS_DMS_KEY"), "EMDW_DMART.CUS_DMS")

    def test_iteration_order_no_longer_decides_the_answer(self):
        reordered = dict(reversed(list(SCHEMA.items())))
        self.assertEqual(
            _resolved("PFT_CTR_DMS_KEY", reordered), "EMDW_DMART.PFT_CTR_DMS",
        )

    def test_the_name_convention_still_works_without_pk_metadata(self):
        # Discovery does not always return primary keys.
        no_pks = {fqn: {"columns": meta["columns"], "pk_columns": []}
                  for fqn, meta in SCHEMA.items()}
        self.assertEqual(
            _resolved("PFT_CTR_DMS_KEY", no_pks), "EMDW_DMART.PFT_CTR_DMS",
        )

    def test_containment_still_answers_when_nothing_better_exists(self):
        # A convention this code does not model must not regress to None.
        odd = {"WAREHOUSE.DIM_LOCATION": _table(
            ["LOCATION_SK", "LOCATION_NAME"], ["LOCATION_SK"])}
        self.assertEqual(_resolved("LOCATION_SK", odd), "WAREHOUSE.DIM_LOCATION")

    def test_an_unknown_key_resolves_to_nothing(self):
        self.assertIsNone(_resolved("NO_SUCH_DMS_KEY"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
