"""
Tests for core/vector_store.py

All Qdrant calls are mocked — no real Qdrant instance required.
Tests verify:
  - Point IDs are deterministic (idempotent upserts)
  - FQN parsing handles 1/2/3-part names
  - MD header FQN extraction
  - Retriever correctly builds ACL filter including _global docs
  - Multi-schema queries are not blocked by schema boundaries
  - Empty results are handled gracefully

Run: python -m unittest tests.test_vector_store
"""

import hashlib
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Helpers to patch the Qdrant singleton before it is used
# ---------------------------------------------------------------------------

def _mock_qdrant():
    """Return a MagicMock that acts like a QdrantClient."""
    m = MagicMock()
    m.get_collections.return_value = MagicMock(collections=[])
    m.count.return_value = MagicMock(count=10)
    return m


def _mock_embedder(n_texts=1):
    """Return a MagicMock that returns n float vectors (one per input text)."""
    import numpy as np
    m = MagicMock()
    # side_effect lets us return the right number of rows based on input length
    def _encode(texts, **kwargs):
        return np.zeros((len(texts), 384), dtype="float32")
    m.encode.side_effect = _encode
    return m


# ---------------------------------------------------------------------------
# Pure-logic tests (no Qdrant, no model)
# ---------------------------------------------------------------------------

class FQNParseTests(unittest.TestCase):

    def test_three_part(self):
        from core.vector_store import _parse_fqn
        db, sch, tbl = _parse_fqn("CHATBOT_DB.HR.EMPLOYEES")
        self.assertEqual(db,  "CHATBOT_DB")
        self.assertEqual(sch, "HR")
        self.assertEqual(tbl, "EMPLOYEES")

    def test_two_part(self):
        from core.vector_store import _parse_fqn
        db, sch, tbl = _parse_fqn("DBO.ORDERS")
        self.assertEqual(db,  "")
        self.assertEqual(sch, "DBO")
        self.assertEqual(tbl, "ORDERS")

    def test_bare_name(self):
        from core.vector_store import _parse_fqn
        db, sch, tbl = _parse_fqn("ORDERS")
        self.assertEqual(db,  "")
        self.assertEqual(sch, "")
        self.assertEqual(tbl, "ORDERS")

    def test_lowercase_is_uppercased(self):
        from core.vector_store import _parse_fqn
        db, sch, tbl = _parse_fqn("mydb.public.orders")
        self.assertEqual(db,  "MYDB")
        self.assertEqual(sch, "PUBLIC")
        self.assertEqual(tbl, "ORDERS")


class PointIdTests(unittest.TestCase):

    def test_same_inputs_produce_same_id(self):
        from core.vector_store import _point_id
        id1 = _point_id("acct_a", "DB.HR.EMP", "kb")
        id2 = _point_id("acct_a", "DB.HR.EMP", "kb")
        self.assertEqual(id1, id2)

    def test_different_account_produces_different_id(self):
        from core.vector_store import _point_id
        id1 = _point_id("acct_a", "DB.HR.EMP", "kb")
        id2 = _point_id("acct_b", "DB.HR.EMP", "kb")
        self.assertNotEqual(id1, id2)

    def test_different_fqn_produces_different_id(self):
        from core.vector_store import _point_id
        id1 = _point_id("acct", "DB.HR.EMP",     "kb")
        id2 = _point_id("acct", "DB.DBO.ORDERS",  "kb")
        self.assertNotEqual(id1, id2)

    def test_id_is_uuid_format(self):
        from core.vector_store import _point_id
        pid = _point_id("acct", "DB.HR.EMP", "kb")
        parts = pid.split("-")
        self.assertEqual(len(parts), 5)
        self.assertEqual([len(p) for p in parts], [8, 4, 4, 4, 12])


class FQNFromHeaderTests(unittest.TestCase):

    def test_three_part_heading(self):
        from core.vector_store import _fqn_from_md_header
        md = "# CHATBOT_DB.HR.EMPLOYEES\n\nSome content."
        self.assertEqual(_fqn_from_md_header(md), "CHATBOT_DB.HR.EMPLOYEES")

    def test_two_part_heading(self):
        from core.vector_store import _fqn_from_md_header
        md = "# DBO.ORDERS\n\nThis table stores orders."
        self.assertEqual(_fqn_from_md_header(md), "DBO.ORDERS")

    def test_no_fqn_returns_none(self):
        from core.vector_store import _fqn_from_md_header
        # A heading that is plain English with no dot-separated identifier
        md = "# Welcome to the Knowledge Base\n\nContent follows here."
        self.assertIsNone(_fqn_from_md_header(md))

    def test_heading_with_description(self):
        from core.vector_store import _fqn_from_md_header
        md = "# HR.EMPLOYEE_ATTENDANCE — Tracks daily attendance"
        self.assertEqual(_fqn_from_md_header(md), "HR.EMPLOYEE_ATTENDANCE")


# ---------------------------------------------------------------------------
# Qdrant interaction tests (client mocked)
# ---------------------------------------------------------------------------

class UpsertKBFileTests(unittest.TestCase):

    def test_upsert_calls_qdrant_with_correct_payload(self):
        with patch("core.vector_store._qdrant_client", _mock_qdrant()) as mock_client, \
             patch("core.vector_store._embed_model", _mock_embedder()):

            import core.vector_store as vs
            mock_client.return_value = mock_client

            vs.upsert_kb_file(
                account_id="acct_test",
                fqn="CHATBOT_DB.HR.EMPLOYEES",
                doc_type="kb",
                content="The EMPLOYEES table stores employee records.",
                source_file="HR__EMPLOYEES_kb.md",
            )

            mock_client.upsert.assert_called_once()
            call_args = mock_client.upsert.call_args
            points = call_args.kwargs.get("points") or call_args.args[1]
            self.assertEqual(len(points), 1)
            p = points[0]
            self.assertEqual(p.payload["account_id"],  "acct_test")
            self.assertEqual(p.payload["fqn"],         "CHATBOT_DB.HR.EMPLOYEES")
            self.assertEqual(p.payload["doc_type"],    "kb")
            self.assertEqual(p.payload["database"],    "CHATBOT_DB")
            self.assertEqual(p.payload["schema_name"], "HR")
            self.assertEqual(p.payload["table_name"],  "EMPLOYEES")

    def test_empty_content_is_skipped(self):
        with patch("core.vector_store._qdrant_client", _mock_qdrant()) as mock_client, \
             patch("core.vector_store._embed_model", _mock_embedder()):
            import core.vector_store as vs
            vs.upsert_kb_file("acct", "DB.HR.EMP", "kb", "   ")
            mock_client.upsert.assert_not_called()


class UpsertExamplesTests(unittest.TestCase):

    def test_examples_have_correct_doc_type(self):
        with patch("core.vector_store._qdrant_client", _mock_qdrant()) as mock_client, \
             patch("core.vector_store._embed_model", _mock_embedder()):
            import core.vector_store as vs
            vs.upsert_examples("acct_test", [
                ("How many employees?", "SELECT COUNT(*) FROM HR.EMPLOYEES",
                 "CHATBOT_DB.HR.EMPLOYEES"),
                ("Total orders?", "SELECT COUNT(*) FROM DBO.ORDERS",
                 "CHATBOT_DB.DBO.ORDERS"),
            ])
            mock_client.upsert.assert_called_once()
            points = mock_client.upsert.call_args.kwargs.get("points") \
                     or mock_client.upsert.call_args.args[1]
            self.assertEqual(len(points), 2)
            for p in points:
                self.assertEqual(p.payload["doc_type"],   "example")
                self.assertEqual(p.payload["account_id"], "acct_test")
            fqns = {p.payload["fqn"] for p in points}
            self.assertIn("CHATBOT_DB.HR.EMPLOYEES",  fqns)
            self.assertIn("CHATBOT_DB.DBO.ORDERS",    fqns)


# ---------------------------------------------------------------------------
# Retriever tests — verify ACL filter construction
# ---------------------------------------------------------------------------

class RetrieverACLFilterTests(unittest.TestCase):
    """
    The most important tests: verify that the ACL filter passed to Qdrant
    always includes _global docs AND the user's allowed tables, and never
    leaks tables the user cannot see.
    """

    def _make_retriever(self, mock_client):
        with patch("core.vector_store._qdrant_client", mock_client):
            from core.vector_store import QdrantKBRetriever
            return QdrantKBRetriever("acct_test")

    def test_filter_includes_global_when_allowed_tables_set(self):
        mock_client = _mock_qdrant()
        hit = MagicMock()
        hit.payload = {"content": "Employees table context", "doc_type": "kb"}
        mock_result = MagicMock()
        mock_result.points = [hit]
        mock_client.query_points.return_value = mock_result

        with patch("core.vector_store._qdrant_client", mock_client), \
             patch("core.vector_store._embed_model", _mock_embedder()):
            from core.vector_store import QdrantKBRetriever
            r = QdrantKBRetriever("acct_test")
            r.retrieve("how many employees", n=4,
                       allowed_tables={"CHATBOT_DB.HR.EMPLOYEES"})

        mock_client.query_points.assert_called_once()
        call_kw = mock_client.query_points.call_args.kwargs
        filt = call_kw.get("query_filter")
        must_conditions = filt.must
        fqn_condition = next(
            c for c in must_conditions if hasattr(c, "match") and
            hasattr(c.match, "any")
        )
        allowed_in_filter = fqn_condition.match.any
        self.assertIn("_global",                    allowed_in_filter)
        self.assertIn("CHATBOT_DB.HR.EMPLOYEES",    allowed_in_filter)

    def test_no_allowed_tables_passes_no_fqn_filter(self):
        """When allowed_tables is None (admin), no FQN restriction is applied."""
        mock_client = _mock_qdrant()
        mock_result = MagicMock()
        mock_result.points = []
        mock_client.query_points.return_value = mock_result

        with patch("core.vector_store._qdrant_client", mock_client), \
             patch("core.vector_store._embed_model", _mock_embedder()):
            from core.vector_store import QdrantKBRetriever
            r = QdrantKBRetriever("acct_test")
            r.retrieve("show everything", n=4, allowed_tables=None)

        filt = mock_client.query_points.call_args.kwargs.get("query_filter")
        fqn_conditions = [
            c for c in filt.must
            if hasattr(c, "key") and c.key == "fqn"
        ]
        self.assertEqual(len(fqn_conditions), 0)

    def test_multi_schema_allowed_tables_all_present_in_filter(self):
        mock_client = _mock_qdrant()
        mock_result = MagicMock()
        mock_result.points = []
        mock_client.query_points.return_value = mock_result

        allowed = {
            "CHATBOT_DB.HR.EMPLOYEES",
            "CHATBOT_DB.DBO.ORDERS",
            "CHATBOT_DB.LOGS.QUERY_LOG",
        }

        with patch("core.vector_store._qdrant_client", mock_client), \
             patch("core.vector_store._embed_model", _mock_embedder()):
            from core.vector_store import QdrantKBRetriever
            r = QdrantKBRetriever("acct_test")
            r.retrieve("show cross schema data", n=6, allowed_tables=allowed)

        filt = mock_client.query_points.call_args.kwargs.get("query_filter")
        fqn_condition = next(
            c for c in filt.must
            if hasattr(c, "match") and hasattr(c.match, "any")
        )
        in_filter = set(fqn_condition.match.any)
        self.assertIn("CHATBOT_DB.HR.EMPLOYEES",    in_filter)
        self.assertIn("CHATBOT_DB.DBO.ORDERS",      in_filter)
        self.assertIn("CHATBOT_DB.LOGS.QUERY_LOG",  in_filter)
        self.assertIn("_global",                    in_filter)

    def test_empty_result_returns_empty_list(self):
        mock_client = _mock_qdrant()
        mock_client.search.return_value = []

        with patch("core.vector_store._qdrant_client", mock_client), \
             patch("core.vector_store._embed_model", _mock_embedder()):
            from core.vector_store import QdrantKBRetriever
            r = QdrantKBRetriever("acct_test")
            result = r.retrieve("anything", allowed_tables={"DB.HR.EMP"})
        self.assertEqual(result, [])

    def test_zero_count_returns_empty_without_searching(self):
        mock_client = _mock_qdrant()
        mock_client.count.return_value = MagicMock(count=0)

        with patch("core.vector_store._qdrant_client", mock_client), \
             patch("core.vector_store._embed_model", _mock_embedder()):
            from core.vector_store import QdrantKBRetriever
            r = QdrantKBRetriever("acct_test")
            result = r.retrieve("anything")
        mock_client.search.assert_not_called()
        self.assertEqual(result, [])


class DeleteKBTests(unittest.TestCase):

    def test_delete_uses_account_id_filter(self):
        mock_client = _mock_qdrant()
        with patch("core.vector_store._qdrant_client", mock_client):
            from core.vector_store import delete_kb_for_client
            delete_kb_for_client("acct_xyz")

        mock_client.delete.assert_called_once()
        filt = mock_client.delete.call_args.kwargs.get("points_selector") \
               or mock_client.delete.call_args.args[1]
        account_cond = filt.must[0]
        self.assertEqual(account_cond.match.value, "acct_xyz")


if __name__ == "__main__":
    unittest.main()
