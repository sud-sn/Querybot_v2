"""
tests/test_conflict_counts_account_for_the_total.py

Model Health subtitles its open-conflict total with a severity breakdown. On a
live workspace the two disagreed:

    OPEN CONFLICTS  2177
    0 errors · 2067 warnings

110 conflicts the reader could see the total of and never the nature of. The
SQL was not wrong — errors, warnings and open_total were each counted
correctly — it simply had no bucket for a severity that is neither, and INFO
exists in the detectors. A breakdown that silently omits a bucket reads as a
complete account of the total, so the invariant is the sum, not the individual
counters.
"""

from __future__ import annotations

import tempfile
import unittest
import uuid
from pathlib import Path


class ConflictCountsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import store
        import store.database
        import store.db

        cls.tmp = tempfile.TemporaryDirectory()
        cls.db_path = Path(cls.tmp.name) / "conflict-counts.db"
        cls.old_database_path = store.database.DB_PATH
        cls.old_db_path = store.db.DB_PATH
        store.database.DB_PATH = cls.db_path
        store.db.DB_PATH = cls.db_path
        store.init_db()

    @classmethod
    def tearDownClass(cls):
        import store.database
        import store.db

        store.database.DB_PATH = cls.old_database_path
        store.db.DB_PATH = cls.old_db_path
        cls.tmp.cleanup()

    def test_the_breakdown_adds_up_to_the_total_it_subtitles(self):
        from store.config_store import upsert_client
        from store.semantic_compile_store import (
            create_semantic_compile_run,
            get_semantic_compiler_summary,
            save_semantic_conflicts,
        )

        account = f"acct_{uuid.uuid4().hex[:8]}"
        upsert_client(account, "zoom")  # semantic_compile_run FKs to client
        run_id = create_semantic_compile_run(
            account, trigger="test", initiated_by="tests", mode="shadow",
            base_version="v0",
        )
        # Three ERROR, four WARNING, two INFO — INFO being the severity the
        # breakdown had no bucket for.
        conflicts = (
            [{"severity": "ERROR", "conflict_key": f"e{i}"} for i in range(3)]
            + [{"severity": "WARNING", "conflict_key": f"w{i}"} for i in range(4)]
            + [{"severity": "INFO", "conflict_key": f"i{i}"} for i in range(2)]
        )
        save_semantic_conflicts(run_id, account, conflicts)

        counts = get_semantic_compiler_summary(account)["counts"]
        self.assertEqual(counts["errors"], 3)
        self.assertEqual(counts["warnings"], 4)
        self.assertEqual(counts["other"], 2)
        self.assertEqual(counts["open_total"], 9)
        self.assertEqual(
            counts["errors"] + counts["warnings"] + counts["other"],
            counts["open_total"],
            "the severity breakdown must account for every open conflict",
        )


if __name__ == "__main__":
    unittest.main()
