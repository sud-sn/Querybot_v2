"""
tests/test_gap_fill_sections.py

Gap-fill must send schema truth, not illustrative SQL.

A table reaches gap-fill because retrieval missed it, so it is normally a join
partner or the owner of a single measure — not the subject of the question.
Injecting its whole KB document is self-defeating: gap-fill appends to the END
of the prompt, which is exactly where the character cap truncates.

Measured in production: five gap-filled Infor M3 tables produced

    Prompt context clamped: 253664 chars -> 120000 cap (tail truncated)

so the model planned fields from context it never actually saw — which is how
"purchase" was bound to an inventory quantity column.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_tmpdir = tempfile.mkdtemp(prefix="querybot_gap_fill_")
os.environ.setdefault("DB_PATH", str(Path(_tmpdir) / "test_querybot.db"))
os.environ.setdefault("QUERYBOT_KEY_FILE", str(Path(_tmpdir) / "test_key"))

import core.vector_store as vector_store  # noqa: E402

FQN = "EMDW_DMART.PCH_ORD_RCT_FCT"

# Section sizes shaped like a wide M3 fact: the column list is what the model
# genuinely needs, the query-pattern block is the bulk.
SECTION_SIZES = {
    "overview": 800,
    "key_metrics": 1200,
    "always_exclude": 400,
    "columns": 24000,
    "join_keys": 900,
    "synonyms": 700,
    "patterns": 62000,
}


class _Point:
    def __init__(self, slug: str, size: int, fqn: str = FQN):
        self.payload = {
            "account_id": "acct", "fqn": fqn, "doc_type": "kb",
            "section_type": slug,
            "content": f"# {fqn}\n\n## {slug}\n" + ("x" * size),
        }


def _qdrant_returning(points):
    client = MagicMock()
    client.scroll.return_value = (points, None)
    return client


def _fetch(sections=None, points=None):
    if points is None:
        points = [_Point(slug, size) for slug, size in SECTION_SIZES.items()]
    with patch.object(vector_store, "_qdrant", lambda: _qdrant_returning(points)):
        return vector_store.fetch_docs_for_fqn("acct", FQN, sections=sections)


class TestGapFillSectionSelection(unittest.TestCase):

    def test_schema_truth_is_kept(self):
        doc = _fetch(sections=vector_store.GAP_FILL_SECTIONS)
        for slug in ("columns", "join_keys", "key_metrics", "always_exclude",
                     "overview", "synonyms"):
            with self.subTest(section=slug):
                self.assertIn(f"## {slug}", doc)

    def test_illustrative_sql_is_dropped(self):
        doc = _fetch(sections=vector_store.GAP_FILL_SECTIONS)
        self.assertNotIn("## patterns", doc)

    def test_the_document_gets_substantially_smaller(self):
        whole = _fetch()
        trimmed = _fetch(sections=vector_store.GAP_FILL_SECTIONS)
        self.assertLess(len(trimmed), len(whole) // 2)

    def test_the_column_list_is_never_truncated_mid_section(self):
        """A half-visible column list invites invented column names."""
        trimmed = _fetch(sections=vector_store.GAP_FILL_SECTIONS)
        self.assertEqual(trimmed.count("x" * SECTION_SIZES["columns"]), 1)

    def test_join_keys_survive_because_joins_are_the_point(self):
        trimmed = _fetch(sections=vector_store.GAP_FILL_SECTIONS)
        self.assertIn("## join_keys", trimmed)

    def test_no_filter_returns_the_whole_document(self):
        doc = _fetch()
        self.assertIn("## patterns", doc)

    def test_an_unmatched_filter_falls_back_to_the_whole_document(self):
        """A tenant whose sections are named differently must still be sent."""
        doc = _fetch(sections=("no_such_section",))
        self.assertIn("## columns", doc)
        self.assertIn("## patterns", doc)

    def test_legacy_whole_doc_payloads_are_unaffected(self):
        legacy = MagicMock()
        legacy.payload = {
            "account_id": "acct", "fqn": FQN, "doc_type": "kb",
            "section_type": "full", "content": f"# {FQN}\n\nlegacy body",
        }
        doc = _fetch(sections=vector_store.GAP_FILL_SECTIONS, points=[legacy])
        self.assertIn("legacy body", doc)

    def test_missing_table_still_returns_nothing(self):
        self.assertIsNone(_fetch(sections=vector_store.GAP_FILL_SECTIONS, points=[]))


class TestCoverageUsesTheTrimmedSections(unittest.TestCase):

    def test_guarantee_table_coverage_requests_the_gap_fill_sections(self):
        import core.table_coverage as table_coverage

        seen: dict = {}

        def _fake_fetch(account_id, fqn, sections=None):
            seen["sections"] = sections
            return f"# {fqn}\n\ndoc"

        with patch("core.vector_store.fetch_docs_for_fqn", _fake_fetch):
            docs = table_coverage.guarantee_table_coverage(
                account_id="acct", required_fqns={FQN},
                retrieved_docs=[], rag_filter=None,
            )
        self.assertEqual(len(docs), 1)
        self.assertEqual(seen["sections"], vector_store.GAP_FILL_SECTIONS)
        self.assertNotIn("patterns", seen["sections"])

    def test_an_explicit_section_set_overrides_the_default(self):
        import core.table_coverage as table_coverage

        seen: dict = {}

        def _fake_fetch(account_id, fqn, sections=None):
            seen["sections"] = sections
            return "doc"

        with patch("core.vector_store.fetch_docs_for_fqn", _fake_fetch):
            table_coverage.guarantee_table_coverage(
                account_id="acct", required_fqns={FQN},
                retrieved_docs=[], rag_filter=None,
                sections=("columns",),
            )
        self.assertEqual(seen["sections"], ("columns",))


if __name__ == "__main__":
    unittest.main()
