"""Pins the hybrid retrieval pipeline (dense + BM25 + RRF + rerank) in
core/vector_store.py.

The BM25 leg exists for exact-token recall: dense MiniLM embeddings are weak
on rare identifiers ("NDC_CODE", product codes, verbatim column names), and
the whole leg silently degrades to dense-only if rank_bm25 is missing — so
these tests pin both the behavior AND the deployment guarantee.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.vector_store import (  # noqa: E402
    _bm25_search, _bm25_tokenize, _invalidate_bm25_cache,
    _bm25_index_cache, _rrf_fuse,
)


def _make_docs():
    return [
        {"id": 1, "fqn": "RX.FACT_SALES", "doc_type": "kb",
         "content": "# RX.FACT_SALES\n\n## Overview\nMonthly sales revenue and quantity by product and region"},
        {"id": 2, "fqn": "RX.DIM_DRUG", "doc_type": "kb",
         "content": "# RX.DIM_DRUG\n\n## Columns\nNDC_CODE varchar - National Drug Code identifier, DRUG_NAME varchar"},
        {"id": 3, "fqn": "RX.DIM_CUSTOMER", "doc_type": "kb",
         "content": "# RX.DIM_CUSTOMER\n\n## Overview\nCustomer master with region and segment"},
    ]


class Bm25DependencyTests(unittest.TestCase):
    def test_rank_bm25_is_installed_and_required(self):
        # Graceful degradation means a missing package would silently turn
        # hybrid retrieval into dense-only — pin that it's a real dependency.
        import rank_bm25  # noqa: F401
        req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertRegex(req, r"(?mi)^rank-bm25")


class Bm25SearchTests(unittest.TestCase):
    def test_tokenizer_preserves_underscored_identifiers(self):
        # "NDC_CODE" must survive as one token — splitting on "_" would make
        # exact column-name queries match every doc containing "code".
        self.assertIn("ndc_code", _bm25_tokenize("top drugs by NDC_CODE"))

    def test_exact_identifier_query_surfaces_the_right_table(self):
        from rank_bm25 import BM25Okapi
        docs = _make_docs()
        bm25 = BM25Okapi([_bm25_tokenize(d["content"]) for d in docs])
        hits = _bm25_search(bm25, docs, "top drugs by NDC_CODE", n=2)
        self.assertEqual(hits[0]["fqn"], "RX.DIM_DRUG")

    def test_doc_type_filter_applies(self):
        from rank_bm25 import BM25Okapi
        docs = _make_docs()
        docs[1] = {**docs[1], "doc_type": "queries"}
        bm25 = BM25Okapi([_bm25_tokenize(d["content"]) for d in docs])
        hits = _bm25_search(bm25, docs, "NDC_CODE drug", n=3, doc_types=["kb"])
        self.assertTrue(all(h["doc_type"] == "kb" for h in hits))


class RrfFusionTests(unittest.TestCase):
    def test_bm25_only_hit_joins_the_candidate_pool(self):
        # The point of the lexical leg: a doc dense search missed entirely
        # must still reach the cross-encoder rerank pool.
        docs = _make_docs()
        dense = [docs[0], docs[2]]          # dense missed DIM_DRUG
        lexical = [docs[1]]                 # BM25 found it
        fused = _rrf_fuse(dense, lexical)
        self.assertIn("RX.DIM_DRUG", [d["fqn"] for d in fused])

    def test_duplicates_across_legs_are_merged_not_repeated(self):
        docs = _make_docs()
        fused = _rrf_fuse([docs[0], docs[1]], [docs[1], docs[2]])
        fqns = [d["fqn"] for d in fused]
        self.assertEqual(len(fqns), len(set(fqns)))
        # Appearing in BOTH legs sums RRF contributions — docs[1] (rank 2 +
        # rank 1) must outrank docs[2] (rank 2 in one leg only).
        self.assertLess(fqns.index("RX.DIM_DRUG"), fqns.index("RX.DIM_CUSTOMER"))


class Bm25CacheInvalidationTests(unittest.TestCase):
    def test_invalidate_removes_only_that_accounts_entries(self):
        _bm25_index_cache.clear()
        _bm25_index_cache["acct-a:*:*"] = (9e12, object(), [])
        _bm25_index_cache["acct-a:FQN1:kb"] = (9e12, object(), [])
        _bm25_index_cache["acct-b:*:*"] = (9e12, object(), [])
        try:
            _invalidate_bm25_cache("acct-a")
            self.assertNotIn("acct-a:*:*", _bm25_index_cache)
            self.assertNotIn("acct-a:FQN1:kb", _bm25_index_cache)
            self.assertIn("acct-b:*:*", _bm25_index_cache)
        finally:
            _bm25_index_cache.clear()

    def test_every_upsert_path_invalidates_the_cache(self):
        # A stale BM25 corpus would serve deleted/edited KB text for up to
        # the TTL — every write path must invalidate.
        src = (ROOT / "core/vector_store.py").read_text(encoding="utf-8")
        for fn in ("def upsert_kb_file", "def upsert_kb_directory", "def re_embed_single_file"):
            body = src[src.index(fn):]
            body = body[:body.index("\ndef ") if "\ndef " in body else len(body)]
            self.assertIn("_invalidate_bm25_cache", body, fn)


if __name__ == "__main__":
    unittest.main()
