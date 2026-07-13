"""
tests/test_learning_staleness.py

Learning Queue design gap: schema_scope / semantic_model_version /
metric_version / schema_version columns existed on learning_candidate but
were never populated at harvest time (core/pipeline_trace.py's
_create_learning_candidate never passed them) and never read at retrieval
time (retrieve_governed_examples had no version param at all) — so an
approved example from before a KB rebuild or metric edit could keep
guiding SQL generation toward a stale shape indefinitely, with the
"schema/version isolation" the columns implied never actually happening.

Covers:
  1. compute_learning_versions (core/pipeline_trace.py) — the fingerprint
     helper stamped onto every new learning_candidate.
  2. End-to-end wiring markers: the harvest call site passes the new
     kwargs, the retrieval call site computes and passes the current
     version, and the governed-store write/read functions accept it.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_tmp_db = os.path.join(tempfile.mkdtemp(), "test_learning_staleness.db")
os.environ["QUERYBOT_DB_PATH"] = _tmp_db
for mod in list(sys.modules.keys()):
    if mod.startswith("store"):
        del sys.modules[mod]
import store.db as db_mod
db_mod.init_db()

import store
from core.pipeline_trace import compute_learning_versions


class ComputeLearningVersionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        store.upsert_client("acct_learn", "portal")

    def test_all_empty_when_no_dirs_and_no_metrics(self):
        versions = compute_learning_versions("acct_learn")
        self.assertEqual(versions, {
            "semantic_model_version": "", "schema_version": "", "metric_version": "0",
            "contract_version": "",
        })

    def test_semantic_model_version_from_kb_dir(self):
        with tempfile.TemporaryDirectory() as kb_dir:
            (Path(kb_dir) / "_semantic_model.json").write_text('{"tables": []}', encoding="utf-8")
            versions = compute_learning_versions("acct_learn", kb_dir=kb_dir)
        self.assertNotEqual(versions["semantic_model_version"], "")

    def test_schema_version_from_schema_dir(self):
        with tempfile.TemporaryDirectory() as schema_dir:
            (Path(schema_dir) / "_schema.json").write_text('{"T": {}}', encoding="utf-8")
            versions = compute_learning_versions("acct_learn", schema_dir=schema_dir)
        self.assertNotEqual(versions["schema_version"], "")

    def test_missing_files_never_raise(self):
        versions = compute_learning_versions(
            "acct_learn", kb_dir="/no/such/dir", schema_dir="/no/such/dir",
        )
        self.assertEqual(versions["semantic_model_version"], "")
        self.assertEqual(versions["schema_version"], "")

    @staticmethod
    def _insert_metric(account_id: str) -> int:
        with store.get_db() as conn:
            cur = conn.execute(
                "INSERT INTO metric_registry (account_id, name, sql_template) VALUES (?, 'Revenue', 'SUM(x)')",
                (account_id,),
            )
            return int(cur.lastrowid)

    def test_metric_version_reflects_latest_metric_version_row(self):
        metric_id = self._insert_metric("acct_learn")
        with store.get_db() as conn:
            conn.execute(
                "INSERT INTO metric_version (metric_id, account_id, version, name, sql_template) "
                "VALUES (?, 'acct_learn', 1, 'Revenue', 'SUM(x)')",
                (metric_id,),
            )
        v1 = compute_learning_versions("acct_learn")
        self.assertNotEqual(v1["metric_version"], "0")

        with store.get_db() as conn:
            conn.execute(
                "INSERT INTO metric_version (metric_id, account_id, version, name, sql_template) "
                "VALUES (?, 'acct_learn', 2, 'Revenue', 'SUM(y)')",
                (metric_id,),
            )
        v2 = compute_learning_versions("acct_learn")
        self.assertGreater(int(v2["metric_version"]), int(v1["metric_version"]))

    def test_metric_version_isolated_per_account(self):
        store.upsert_client("acct_learn_other", "portal")
        metric_id = self._insert_metric("acct_learn_other")
        with store.get_db() as conn:
            conn.execute(
                "INSERT INTO metric_version (metric_id, account_id, version, name, sql_template) "
                "VALUES (?, 'acct_learn_other', 5, 'Cost', 'SUM(z)')",
                (metric_id,),
            )
        versions = compute_learning_versions("acct_learn")
        self.assertEqual(versions["metric_version"], "0")  # unaffected by the other account


class WiringMarkerTests(unittest.TestCase):
    """Source-string assertions — same convention as this session's other
    wiring-guard tests (e.g. tests/test_stop_query.py)."""

    def test_harvest_call_site_passes_version_kwargs(self):
        src = (ROOT / "core" / "query_pipeline.py").read_text(encoding="utf-8")
        anchor = src.index("_create_learning_candidate(")
        block = src[anchor:anchor + 700]
        self.assertIn("schema_scope      = schema_hint", block)
        self.assertIn('kb_dir            = state.get("kb_dir", "")', block)
        self.assertIn('schema_dir        = state.get("schema_dir", "")', block)

    def test_create_candidate_computes_and_forwards_versions(self):
        src = (ROOT / "core" / "pipeline_trace.py").read_text(encoding="utf-8")
        start = src.index("def _create_learning_candidate(")
        body = src[start:start + 3000]
        self.assertIn("versions = compute_learning_versions(", body)
        self.assertIn('semantic_model_version = versions["semantic_model_version"]', body)
        self.assertIn('metric_version         = versions["metric_version"]', body)
        self.assertIn('schema_version         = versions["schema_version"]', body)

    def test_fire_governed_upsert_forwards_semantic_model_version(self):
        src = (ROOT / "store" / "learning_store.py").read_text(encoding="utf-8")
        start = src.index("def _fire_governed_upsert(")
        body = src[start:start + 1200]
        self.assertIn('semantic_model_version=candidate.get("semantic_model_version", "")', body)

    def test_backfill_forwards_semantic_model_version(self):
        src = (ROOT / "core" / "governed_store.py").read_text(encoding="utf-8")
        start = src.index("def backfill_approved_candidates(")
        body = src[start:start + 1600]
        self.assertIn('semantic_model_version=c.get("semantic_model_version", "")', body)

    def test_examples_retrieval_computes_current_version_from_kb_dir(self):
        # retrieve_similar_examples is defined TWICE in this file — an
        # earlier legacy ChromaDB-based version, then the real dual-source
        # (governed + legacy Qdrant) one that's actually live (Python keeps
        # only the last definition). rindex to make sure this test checks
        # the function that actually runs, not the dead first definition.
        src = (ROOT / "core" / "examples.py").read_text(encoding="utf-8")
        start = src.rindex("def retrieve_similar_examples(")
        body = src[start:start + 3200]
        self.assertIn('kb_dir: str = ""', body)
        self.assertIn("from core.semantic_model import semantic_model_fingerprint", body)
        self.assertIn("current_semantic_model_version=current_version", body)

    def test_query_pipeline_passes_kb_dir_to_example_retrieval(self):
        src = (ROOT / "core" / "query_pipeline.py").read_text(encoding="utf-8")
        anchor = src.index("examples = retrieve_similar_examples(")
        block = src[anchor:anchor + 300]
        self.assertIn('kb_dir=state.get("kb_dir", "")', block)


if __name__ == "__main__":
    unittest.main()
