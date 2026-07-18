"""Retrieval telemetry + KB doc-quality loop.

Layer 1 — QdrantKBRetriever.last_retrieval_stats: per-table best rerank
score + kept/dropped flag, built inside _apply_relevance_floor.
Layer 2 — answer_trace.retrieved_kb_scores: the pipeline persists those
stats on every question's trace.
Layer 3 — store.get_kb_doc_quality: aggregates traces into the Model
Health "most retrieved, least answerable" ranking that tells the admin
which KB doc to edit first.
"""
from __future__ import annotations

import sys
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.vector_store import QdrantKBRetriever  # noqa: E402


def _retriever() -> QdrantKBRetriever:
    # __new__ skips __init__ so no live Qdrant connection is needed —
    # _apply_relevance_floor only touches instance attributes.
    r = QdrantKBRetriever.__new__(QdrantKBRetriever)
    r.last_retrieval_weak = False
    r.last_retrieval_stats = []
    return r


def _hit(fqn, score=None, section="columns"):
    h = {"fqn": fqn, "section_type": section, "content": f"# {fqn}"}
    if score is not None:
        h["_rerank_score"] = score
    return h


class RetrievalStatsTests(unittest.TestCase):
    def test_stats_record_best_score_per_table_and_kept_flag(self):
        r = _retriever()
        hits = [
            _hit("RX.FACT_SALES", 0.91), _hit("RX.FACT_SALES", 0.40),
            _hit("RX.DIM_DRUG", 0.02),          # below the 0.05 floor → dropped
            _hit("_global", 0.99),               # global docs never in stats
        ]
        r._apply_relevance_floor(hits)
        stats = {s["fqn"]: s for s in r.last_retrieval_stats}
        self.assertEqual(set(stats), {"RX.FACT_SALES", "RX.DIM_DRUG"})
        self.assertEqual(stats["RX.FACT_SALES"]["best_score"], 0.91)
        self.assertEqual(stats["RX.FACT_SALES"]["sections"], 2)
        self.assertTrue(stats["RX.FACT_SALES"]["kept"])
        self.assertFalse(stats["RX.DIM_DRUG"]["kept"])
        self.assertFalse(r.last_retrieval_weak)

    def test_weak_retrieval_keeps_only_best_and_marks_rest_dropped(self):
        r = _retriever()
        hits = [_hit("A.T1", 0.03), _hit("A.T2", 0.01)]
        r._apply_relevance_floor(hits)
        self.assertTrue(r.last_retrieval_weak)
        stats = {s["fqn"]: s for s in r.last_retrieval_stats}
        self.assertTrue(stats["A.T1"]["kept"])
        self.assertFalse(stats["A.T2"]["kept"])

    def test_unscored_hits_recorded_with_null_score_and_kept(self):
        # Re-ranker unavailable → no scores; floor never drops, stats still
        # record the candidate tables so the trace isn't blind.
        r = _retriever()
        r._apply_relevance_floor([_hit("A.T1"), _hit("A.T2")])
        stats = {s["fqn"]: s for s in r.last_retrieval_stats}
        self.assertEqual(set(stats), {"A.T1", "A.T2"})
        self.assertIsNone(stats["A.T1"]["best_score"])
        self.assertTrue(all(s["kept"] for s in stats.values()))


class KbDocQualityAggregationTests(unittest.TestCase):
    def setUp(self):
        # Fresh account per TEST — the ranking assertions assume a clean
        # slate, and a shared account would leak rows across tests.
        import store
        store.init_db()
        self.store = store
        self.account_id = f"acct-kbdq-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")

    def tearDown(self):
        with self.store.get_db() as conn:
            conn.execute(
                "DELETE FROM answer_trace WHERE account_id=?", (self.account_id,)
            )
            conn.execute("DELETE FROM client WHERE account_id=?", (self.account_id,))

    def _trace(self, stats, *, sql="", status="completed", answer_type="table"):
        tid = self.store.create_answer_trace(
            account_id=self.account_id,
            question_id=uuid.uuid4().hex[:12],
            question_text="q",
        )
        self.store.update_answer_trace(
            tid,
            retrieved_kb_scores=stats,
            generated_sql=sql,
            status=status,
            answer_type=answer_type,
        )
        return tid

    def test_most_retrieved_least_used_floats_to_top(self):
        good = {"fqn": "RX.FACT_SALES", "sections": 3, "best_score": 0.9, "kept": True}
        noisy = {"fqn": "RX.DIM_MISC", "sections": 1, "best_score": 0.12, "kept": True}
        # FACT_SALES: retrieved twice, used in SQL both times, no failures.
        # DIM_MISC: retrieved twice, never used, one outright failure.
        self._trace([good, noisy], sql="SELECT * FROM RX.FACT_SALES")
        self._trace(
            [good, noisy], sql="SELECT * FROM RX.FACT_SALES",
            status="error", answer_type="error",
        )
        ranked = self.store.get_kb_doc_quality(self.account_id, days=7)
        by_fqn = {r["fqn"]: r for r in ranked}
        self.assertIn("RX.DIM_MISC", by_fqn)
        misc, sales = by_fqn["RX.DIM_MISC"], by_fqn["RX.FACT_SALES"]
        self.assertEqual(misc["retrieved"], 2)
        self.assertEqual(misc["used_in_sql"], 0)
        self.assertEqual(misc["borderline"], 2)       # 0.12 < 0.3 threshold
        self.assertEqual(sales["used_in_sql"], 2)
        # The noisy, never-used table demands more attention than the
        # heavily-used healthy one.
        self.assertGreater(misc["attention_score"], sales["attention_score"])
        self.assertEqual(ranked[0]["fqn"], "RX.DIM_MISC")

    def test_bare_table_name_matching_uses_word_boundaries(self):
        # "SALES" must not count as used just because "SALES_ORDER" appears.
        stats = [{"fqn": "ERP.SALES", "sections": 1, "best_score": 0.8, "kept": True}]
        self._trace(stats, sql="SELECT * FROM ERP.SALES_ORDER")
        self._trace(stats, sql="SELECT * FROM ERP.SALES_ORDER")
        row = next(
            r for r in self.store.get_kb_doc_quality(self.account_id, days=7)
            if r["fqn"] == "ERP.SALES"
        )
        self.assertEqual(row["used_in_sql"], 0)

    def test_below_min_retrievals_excluded(self):
        stats = [{"fqn": "ERP.ONE_OFF", "sections": 1, "best_score": 0.9, "kept": True}]
        self._trace(stats, sql="SELECT 1")
        ranked = self.store.get_kb_doc_quality(self.account_id, days=7, min_retrievals=2)
        self.assertNotIn("ERP.ONE_OFF", [r["fqn"] for r in ranked])


class WiringTests(unittest.TestCase):
    def test_pipeline_persists_stats_on_trace_and_step(self):
        src = (ROOT / "core/query_pipeline.py").read_text(encoding="utf-8")
        anchor = src.index('retrieved_kb_chunk_ids=store.kb_chunk_refs(relevant_kbs)')
        window = src[anchor - 2000:anchor + 2000]
        self.assertIn("retrieved_kb_scores=_retrieval_stats", window)
        self.assertIn('"weak_retrieval": _weak_retrieval', window)
        self.assertIn('"tables": _retrieval_stats', window)

    def test_model_health_route_passes_kb_doc_quality(self):
        src = (ROOT / "admin/routes.py").read_text(encoding="utf-8")
        fn = src[src.index("async def model_health_page("):]
        fn = fn[:fn.index("\n@router.")]
        self.assertIn("get_kb_doc_quality", fn)
        self.assertIn('"kb_doc_quality": kb_doc_quality', fn)

    def test_template_renders_doc_quality_panel(self):
        src = (ROOT / "admin/templates/client_model_health.html").read_text(encoding="utf-8")
        self.assertIn("KB Doc Quality", src)
        self.assertIn("kb_doc_quality", src)
        self.assertIn("attention_score", src)
        self.assertIn("No retrieval telemetry yet", src)  # cold-start state


if __name__ == "__main__":
    unittest.main()
