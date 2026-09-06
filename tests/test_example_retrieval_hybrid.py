"""
tests/test_example_retrieval_hybrid.py

E2b — example retrieval on the same path the knowledge base uses.

An exemplar is injected into every SQL prompt, and it was retrieved with the
weakest mechanism in the codebase: one embedding, one ANN search, no lexical
leg, no reranker — while the KB it sits beside in that same prompt had long
since moved to dense + BM25 + RRF + cross-encoder.

Examples are the one retrieval in the product where the lexical leg matters
MOST. An exemplar earns its place by sharing a measure name with the question,
and a column name is exactly the token a dense embedding smooths away.

They were also anonymous and undated. An example now carries who stands behind
it, when it last ran, and which semantic model it was written against — so one
whose tables have since changed shape is demoted rather than trusted equally,
and one somebody has marked wrong is dropped outright.

Nothing here asserts on source text. The real functions run against a Qdrant
double shaped like the client, exactly as tests/test_vector_store.py does.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import core.vector_store as vs


def _embedder():
    import numpy as np

    model = MagicMock()
    model.encode.side_effect = lambda texts, **kw: np.zeros(
        (len(texts), 384), dtype="float32")
    return model


def _payload(question, sql, *, table="SALES.ORDERS", **extra):
    base = {
        "account_id": "acct", "doc_type": "example", "fqn": table.upper(),
        "content": question, "question": question, "sql": sql,
        "table_name": table.upper(), "author": "", "verified_at": "",
        "semantic_model_version": "", "revoked": False,
    }
    base.update(extra)
    return base


def _client(payloads, *, count=None):
    """A Qdrant double: dense search returns `payloads`, scroll feeds BM25."""
    client = MagicMock()
    client.count.return_value = MagicMock(
        count=len(payloads) if count is None else count)
    client.query_points.return_value = MagicMock(
        points=[MagicMock(payload=p) for p in payloads])
    client.scroll.return_value = ([MagicMock(payload=p) for p in payloads], None)
    return client


def _retrieve(payloads, question="revenue by month", *, client=None, **kwargs):
    vs._invalidate_bm25_cache("acct")
    with patch.object(vs, "_qdrant_client", client or _client(payloads)), \
            patch.object(vs, "_embed_model", _embedder()):
        return vs.retrieve_similar_examples("acct", question, **kwargs)


EXAMPLES = [
    _payload("total revenue by month", "SELECT month, SUM(revenue) FROM sales GROUP BY month"),
    _payload("count of NDC_CODE values", "SELECT COUNT(DISTINCT NDC_CODE) FROM drugs",
             table="RX.DRUGS"),
    _payload("customers by region", "SELECT region, COUNT(*) FROM customers",
             table="SALES.CUSTOMERS"),
]


class TestTheLexicalLegIsActuallyRun(unittest.TestCase):
    """
    The half that was missing. Dense-only is not visible as a bug — it returns
    something plausible every time — so these check that BM25 is reached and
    that its result changes the outcome.
    """

    def test_bm25_is_built_for_examples_only(self):
        seen = {}

        def _fake_bm25(account_id, allowed_fqns, doc_types=None):
            seen["doc_types"] = doc_types
            return None

        with patch.object(vs, "_get_bm25", side_effect=_fake_bm25):
            _retrieve(EXAMPLES)
        # Not the whole corpus: a KB chunk is not an exemplar, and fusing the
        # two ranked lists would put table documentation where SQL should be.
        self.assertEqual(seen["doc_types"], ["example"])

    def test_a_rare_identifier_finds_its_example(self):
        # The case dense retrieval is worst at, and the reason this leg exists.
        results = _retrieve(EXAMPLES, "how many NDC_CODE do we have")
        self.assertTrue(results)
        self.assertEqual(results[0]["question"], "count of NDC_CODE values")

    def test_a_missing_bm25_degrades_to_dense_rather_than_to_nothing(self):
        with patch.object(vs, "_get_bm25", return_value=None):
            results = _retrieve(EXAMPLES)
        self.assertEqual(len(results), 3)

    def test_a_bm25_that_raises_degrades_to_dense(self):
        with patch.object(vs, "_get_bm25", side_effect=RuntimeError("no corpus")):
            results = _retrieve(EXAMPLES)
        self.assertEqual(len(results), 3)

    def test_the_reranker_actually_decides_the_order(self):
        # The cross-encoder is absent in this environment, so _rerank falls
        # back to input order and skipping it entirely looks identical. A stub
        # that reverses makes the difference observable.
        def _reverse(query, candidates, top_n):
            return list(reversed(candidates))[:top_n]

        with patch.object(vs, "_rerank", side_effect=_reverse):
            reranked = _retrieve(EXAMPLES)
        straight = _retrieve(EXAMPLES)
        self.assertEqual([r["question"] for r in reranked],
                         list(reversed([r["question"] for r in straight])))

    def test_the_reranker_sees_the_fused_candidates_not_the_dense_ones(self):
        seen = {}

        def _capture(query, candidates, top_n):
            seen["questions"] = [c.get("question") for c in candidates]
            return candidates[:top_n]

        with patch.object(vs, "_rerank", side_effect=_capture):
            _retrieve(EXAMPLES, "how many NDC_CODE do we have")
        # Every example reached it, not only whatever the ANN search ranked
        # first -- fusing and then reranking only the dense leg would throw
        # away the lexical leg's whole contribution.
        self.assertEqual(len(seen["questions"]), len(EXAMPLES))

    def test_a_dense_failure_still_leaves_the_lexical_leg(self):
        client = _client(EXAMPLES)
        client.query_points.side_effect = RuntimeError("qdrant down")
        results = _retrieve(EXAMPLES, "NDC_CODE", client=client)
        self.assertTrue(results, "hybrid collapsed when one leg failed")


class TestRevocationIsAFilterNotADemotion(unittest.TestCase):

    def test_a_revoked_example_never_appears(self):
        payloads = [_payload("bad one", "SELECT wrong FROM t", revoked=True)] + EXAMPLES
        questions = {r["question"] for r in _retrieve(payloads)}
        # Demoting a wrong answer still puts it in the prompt on a quiet day.
        self.assertNotIn("bad one", questions)

    def test_revocation_holds_even_when_it_is_the_only_example(self):
        payloads = [_payload("bad one", "SELECT wrong FROM t", revoked=True)]
        self.assertEqual(_retrieve(payloads), [])

    def test_an_example_with_no_sql_is_dropped(self):
        payloads = [_payload("empty", "")] + EXAMPLES
        self.assertNotIn("empty", {r["question"] for r in _retrieve(payloads)})


class TestStalenessDemotesAndNeverDrops(unittest.TestCase):

    CURRENT = "v2"

    def _payloads(self):
        return [
            _payload("stale one", "SELECT 1 FROM t", semantic_model_version="v1"),
            _payload("current one", "SELECT 2 FROM t", semantic_model_version="v2"),
        ]

    def test_the_current_example_outranks_the_stale_one(self):
        results = _retrieve(self._payloads(), semantic_model_version=self.CURRENT)
        self.assertEqual(results[0]["question"], "current one")

    def test_the_stale_one_is_still_returned(self):
        # An exemplar written before a KB rebuild is usually still the best
        # answer available; hard-filtering it leaves the prompt with nothing
        # where it used to have something imperfect.
        questions = [r["question"] for r in
                     _retrieve(self._payloads(), semantic_model_version=self.CURRENT)]
        self.assertIn("stale one", questions)

    def test_the_result_says_which_ones_are_stale(self):
        results = {r["question"]: r for r in
                   _retrieve(self._payloads(), semantic_model_version=self.CURRENT)}
        self.assertTrue(results["stale one"]["stale"])
        self.assertFalse(results["current one"]["stale"])

    def test_an_example_from_before_the_field_existed_is_not_stale(self):
        # Marking every legacy example stale on the day the field ships would
        # demote the entire corpus at once — a change in retrieval quality
        # dressed up as a provenance improvement.
        results = _retrieve([_payload("legacy", "SELECT 1 FROM t")],
                            semantic_model_version=self.CURRENT)
        self.assertFalse(results[0]["stale"])

    def test_no_current_version_means_nothing_is_stale(self):
        results = _retrieve(self._payloads())
        self.assertFalse(any(r["stale"] for r in results))

    def test_staleness_is_checked_directly(self):
        self.assertTrue(vs._example_is_current({}, "v2"))
        self.assertTrue(vs._example_is_current({"semantic_model_version": "v2"}, "v2"))
        self.assertFalse(vs._example_is_current({"semantic_model_version": "v1"}, "v2"))
        self.assertTrue(vs._example_is_current({"semantic_model_version": "v1"}, ""))


class TestTheAclStillHolds(unittest.TestCase):

    def test_a_table_outside_the_scope_is_not_returned(self):
        results = _retrieve(EXAMPLES, allowed_tables={"SALES.ORDERS"})
        tables = {r["table"] for r in results}
        self.assertEqual(tables, {"SALES.ORDERS"})

    def test_the_scope_reaches_the_lexical_leg_too(self):
        # A BM25 corpus built without the ACL would surface another tenant's
        # example through the leg that has no filter of its own.
        seen = {}

        def _fake_bm25(account_id, allowed_fqns, doc_types=None):
            seen["allowed_fqns"] = allowed_fqns
            return None

        with patch.object(vs, "_get_bm25", side_effect=_fake_bm25):
            _retrieve(EXAMPLES, allowed_tables={"sales.orders"})
        self.assertEqual(seen["allowed_fqns"], ["SALES.ORDERS"])

    def test_no_scope_means_no_table_filter(self):
        seen = {}

        def _fake_bm25(account_id, allowed_fqns, doc_types=None):
            seen["allowed_fqns"] = allowed_fqns
            return None

        with patch.object(vs, "_get_bm25", side_effect=_fake_bm25):
            _retrieve(EXAMPLES)
        self.assertIsNone(seen["allowed_fqns"])

    def test_an_empty_collection_returns_nothing_without_searching(self):
        client = _client(EXAMPLES, count=0)
        self.assertEqual(_retrieve(EXAMPLES, client=client), [])
        client.query_points.assert_not_called()


class TestProvenanceIsWrittenAndReadBack(unittest.TestCase):
    """
    A write test and a read test that meet in the middle: the write API is
    called with an author, and the read API returns it, with nothing handed
    between them by the test but the payload the writer built.
    """

    def _written_payloads(self, **kwargs):
        client = MagicMock()
        with patch.object(vs, "_qdrant_client", client), \
                patch.object(vs, "_embed_model", _embedder()):
            vs.upsert_examples("acct", [
                ("total revenue by month", "SELECT 1 FROM t", "SALES.ORDERS"),
            ], **kwargs)
        points = (client.upsert.call_args.kwargs.get("points")
                  or client.upsert.call_args.args[1])
        return [p.payload for p in points]

    def test_the_author_is_stored(self):
        self.assertEqual(self._written_payloads(author="ana")[0]["author"], "ana")

    def test_the_verification_date_is_stamped_automatically(self):
        stamped = self._written_payloads()[0]["verified_at"]
        self.assertTrue(stamped)
        # ISO-8601 with a timezone, so "when did this last run" is answerable
        # across deployments rather than in whatever the server's clock says.
        self.assertRegex(stamped, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00$")

    def test_the_model_version_is_stored(self):
        payload = self._written_payloads(semantic_model_version="v9")[0]
        self.assertEqual(payload["semantic_model_version"], "v9")

    def test_revocation_is_stored(self):
        self.assertIs(self._written_payloads(revoked=True)[0]["revoked"], True)
        self.assertIs(self._written_payloads()[0]["revoked"], False)

    def test_what_the_writer_stores_is_what_the_reader_returns(self):
        # The write-to-read seam. Nothing is passed between the two halves by
        # this test except the payload upsert_examples actually built.
        written = self._written_payloads(author="ana", semantic_model_version="v2")
        results = _retrieve(written, semantic_model_version="v2")
        self.assertEqual(results[0]["author"], "ana")
        self.assertEqual(results[0]["verified_at"], written[0]["verified_at"])
        self.assertFalse(results[0]["stale"])

    def test_a_revoked_example_written_now_is_unreachable_now(self):
        written = self._written_payloads(revoked=True)
        self.assertEqual(_retrieve(written), [])


class TestTheWrapperThreadsTheModelVersionToBothStores(unittest.TestCase):
    """
    Staleness that never reaches the retriever is a feature that is shipped
    and inert — the defect class this whole branch has been closing. Both
    stores use the version the same way, so it is computed once and handed to
    both.
    """

    def _call(self, kb_dir="/kb"):
        seen = {}

        def _legacy(account_id, question, n=3, allowed_tables=None,
                    semantic_model_version=""):
            seen["legacy"] = semantic_model_version
            return []

        def _governed(account_id, question, n=3, allowed_tables=None,
                      schema_scope="", current_semantic_model_version=""):
            seen["governed"] = current_semantic_model_version
            return []

        import core.examples as examples
        import core.governed_store as governed_store
        import core.semantic_model as semantic_model

        with patch.object(vs, "retrieve_similar_examples", side_effect=_legacy), \
                patch.object(governed_store, "retrieve_governed_examples",
                             side_effect=_governed), \
                patch.object(semantic_model, "semantic_model_fingerprint",
                             return_value="fingerprint-v7"):
            examples.retrieve_similar_examples("q", "acct", kb_dir=kb_dir)
        return seen

    def test_both_stores_are_told_which_model_is_in_force(self):
        seen = self._call()
        self.assertEqual(seen["legacy"], "fingerprint-v7")
        self.assertEqual(seen["governed"], "fingerprint-v7")

    def test_they_are_told_the_same_thing(self):
        # Computing it twice is how two consumers of one fact drift.
        seen = self._call()
        self.assertEqual(seen["legacy"], seen["governed"])

    def test_no_kb_directory_means_no_version_and_no_crash(self):
        seen = self._call(kb_dir="")
        self.assertEqual(seen["legacy"], "")
        self.assertEqual(seen["governed"], "")

    def test_a_retrieval_failure_is_reported_at_warning_not_debug(self):
        # Non-fatal is not the same as unremarkable. This returning nothing
        # means every SQL prompt loses its few-shot grounding, which fails
        # nothing and quietly makes every answer worse — the exact shape that
        # let a signature mismatch sit undetected behind a bare except.
        import logging

        import core.examples as examples
        import core.governed_store as governed_store

        with patch.object(vs, "retrieve_similar_examples",
                          side_effect=TypeError("unexpected keyword argument")), \
                patch.object(governed_store, "retrieve_governed_examples",
                             return_value=[]), \
                self.assertLogs("querybot", level=logging.WARNING) as captured:
            examples.retrieve_similar_examples("q", "acct")
        self.assertTrue(any("few-shot" in line for line in captured.output),
                        captured.output)

    def test_a_fingerprint_that_raises_does_not_cost_the_examples(self):
        import core.examples as examples
        import core.governed_store as governed_store
        import core.semantic_model as semantic_model

        with patch.object(vs, "retrieve_similar_examples",
                          return_value=[{"question": "q", "sql": "SELECT 1",
                                         "table": "T"}]), \
                patch.object(governed_store, "retrieve_governed_examples",
                             return_value=[]), \
                patch.object(semantic_model, "semantic_model_fingerprint",
                             side_effect=RuntimeError("no model")):
            results = examples.retrieve_similar_examples("q", "acct", kb_dir="/kb")
        self.assertEqual(len(results), 1)


class TestTheDeadChromaPathIsGone(unittest.TestCase):

    def test_examples_no_longer_import_chromadb(self):
        # A ChromaDB retriever lived in core/examples.py under the same name
        # as the Qdrant one defined below it, so it was shadowed at import and
        # never ran again while the embedder lost its last caller. Between
        # them they were the last chromadb usage in the product.
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1] / "core" / "examples.py"
                  ).read_text(encoding="utf-8")
        self.assertNotIn("import chromadb", source)
        self.assertNotIn("PersistentClient", source)

    def test_only_one_retriever_is_defined(self):
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1] / "core" / "examples.py"
                  ).read_text(encoding="utf-8")
        self.assertEqual(source.count("\ndef retrieve_similar_examples("), 1)

    def test_the_surviving_retriever_is_the_qdrant_one(self):
        import inspect

        import core.examples as examples
        self.assertIn("vector_store",
                      inspect.getsource(examples.retrieve_similar_examples))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
