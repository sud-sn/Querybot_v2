"""
tests/test_global_doc_identity.py

Every account-wide KB document shared one Qdrant point.

`_build_whole_doc_point` derived its id from (account_id, fqn, doc_type) with
no per-file part, and `upsert_kb_directory` indexes every file whose stem
starts with "_" under the SAME fqn "_global" and doc_type "global". So
`_business_kb.md`, `_join_map.md` and `_naming_convention.md` all resolved to
one id, each upsert overwrote the last, and only the alphabetically final file
survived. On a normal account that silently discarded the join map -- the one
document that tells the model how tables connect -- while retrieval's "pinned
global docs" pinned whatever had won.

The fix gives each file its own identity. That has a consequence worth pinning
down too: there are now up to three pinned documents where there was one, so
the ranked table documents needed their own budget or fixing the index bug
would have shown up as worse answers.

Nothing here asserts on source text; the id function and the prompt assembly
are executed.
"""

import unittest
from unittest.mock import patch

from core.vector_store import _point_id

GLOBAL_FILES = ("_business_kb.md", "_join_map.md", "_naming_convention.md")


class EveryGlobalDocumentGetsItsOwnPoint(unittest.TestCase):

    def test_distinct_files_no_longer_collide(self):
        ids = {
            _point_id("acct", "_global", "global", extra=name)
            for name in GLOBAL_FILES
        }
        self.assertEqual(
            len(ids), len(GLOBAL_FILES),
            "each account-wide document needs its own point or the upserts "
            "overwrite each other",
        )

    def test_the_collision_is_what_the_old_key_produced(self):
        """Guards the fix by showing the behaviour it replaced."""
        collided = {_point_id("acct", "_global", "global") for _ in GLOBAL_FILES}
        self.assertEqual(len(collided), 1)

    def test_ids_stay_deterministic_across_calls(self):
        """Upserting the same file twice must be idempotent, not additive."""
        first = _point_id("acct", "_global", "global", extra="_join_map.md")
        second = _point_id("acct", "_global", "global", extra="_join_map.md")
        self.assertEqual(first, second)

    def test_accounts_do_not_share_a_point(self):
        self.assertNotEqual(
            _point_id("acct_a", "_global", "global", extra="_join_map.md"),
            _point_id("acct_b", "_global", "global", extra="_join_map.md"),
        )

    def test_the_legacy_id_is_never_reissued(self):
        """So deleting it cannot take a live document with it."""
        legacy = _point_id("acct", "_global", "global")
        live = {
            _point_id("acct", "_global", "global", extra=name)
            for name in GLOBAL_FILES
        }
        self.assertNotIn(legacy, live)

    def test_a_whole_doc_point_carries_its_file_identity(self):
        from core import vector_store

        captured = {}

        class _Point:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with patch.object(vector_store, "_embed", lambda texts: [[0.0]]), \
             patch.dict("sys.modules"):
            import types
            fake = types.ModuleType("qdrant_client.models")
            fake.PointStruct = _Point
            import sys
            sys.modules["qdrant_client.models"] = fake
            vector_store._build_whole_doc_point(
                "acct", "_global", "global", "# Join map\n", "_join_map.md",
            )

        self.assertEqual(captured["payload"]["source_file"], "_join_map.md")
        self.assertEqual(
            captured["id"],
            _point_id("acct", "_global", "global", extra="_join_map.md"),
        )


class PinnedDocumentsDoNotStarveTheRankedOnes(unittest.TestCase):
    """Three globals must not push two table documents out of the prompt."""

    def _select(self, pinned, table_kbs):
        # The real selection the pipeline calls -- not a copy of it. The
        # expression used to be inline in a 6,500-line function, where the only
        # way to test it was to restate it, which tests the restatement.
        from core.pipeline_helpers import select_prompt_documents
        return select_prompt_documents(pinned, table_kbs)

    def test_the_table_budget_is_unchanged_by_extra_globals(self):
        table_kbs = [f"table {i}" for i in range(8)]
        with_one = self._select(["global A"], table_kbs)
        with_three = self._select(["global A", "global B", "global C"], table_kbs)

        tables_in = lambda sel: [d for d in sel if d.startswith("table")]
        self.assertEqual(len(tables_in(with_one)), 6)
        self.assertEqual(
            len(tables_in(with_three)), 6,
            "fixing the index collision must not cost the model two tables",
        )

    def test_pinned_documents_are_bounded(self):
        """_is_global is a text heuristic, so a table doc can land in `pinned`."""
        sel = self._select([f"global {i}" for i in range(9)], ["table 0"])
        self.assertEqual(len([d for d in sel if d.startswith("global")]), 3)

    def test_the_old_shared_budget_is_what_would_have_starved_them(self):
        """Guards the fix by showing the behaviour it replaced."""
        table_kbs = [f"table {i}" for i in range(8)]
        old = (["global A", "global B", "global C"] + table_kbs)[:7]
        self.assertEqual(len([d for d in old if d.startswith("table")]), 4)


if __name__ == "__main__":
    unittest.main()
