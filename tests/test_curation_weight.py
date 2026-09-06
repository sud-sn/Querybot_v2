"""Curation-weighted retrieval — core/curation_weight.py and its wiring.

The load-bearing property is the one that keeps this safe: the weight is
applied to the ORDERING only, and the score the relevance floor reads is left
untouched. A well-curated irrelevant table must never be promoted past a
relevant one; it may only settle a tie the reranker was unsure about. Every
test here executes the real scoring or the real retriever method.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.curation_weight import (  # noqa: E402
    MAX_BOOST,
    MIN_DESCRIPTION_CHARS,
    SIGNAL_WEIGHTS,
    apply,
    invalidate,
    ordering_key,
    score_from_signals,
    scores_for,
    signals_for,
)

DESCRIPTION = "One row per customer order line, restated at invoice grain."


class CurationCase(unittest.TestCase):

    def setUp(self):
        import store

        self._dir = tempfile.mkdtemp(prefix="qb-curation-")
        self._saved = {k: os.environ.get(k) for k in ("DB_PATH", "QUERYBOT_DB_PATH")}
        path = os.path.join(self._dir, "c.db")
        os.environ["DB_PATH"] = path
        os.environ["QUERYBOT_DB_PATH"] = path
        store.init_db()
        self.account_id = f"acct-cur-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")
        invalidate()

    def tearDown(self):
        invalidate()
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self._dir, ignore_errors=True)


class TestScoring(unittest.TestCase):

    def test_the_weights_add_up_to_one(self):
        # A fully-curated table must score exactly 1.0, or MAX_BOOST stops
        # meaning what its name says.
        self.assertAlmostEqual(sum(SIGNAL_WEIGHTS.values()), 1.0, places=6)

    def test_a_fully_curated_table_scores_one(self):
        self.assertEqual(
            score_from_signals({key: True for key in SIGNAL_WEIGHTS}), 1.0)

    def test_an_uncurated_table_scores_zero(self):
        self.assertEqual(score_from_signals({}), 0.0)
        self.assertEqual(
            score_from_signals({key: False for key in SIGNAL_WEIGHTS}), 0.0)

    def test_a_description_is_worth_more_than_any_other_single_signal(self):
        # It is what the model actually reads.
        described = score_from_signals({"described": True})
        for other in SIGNAL_WEIGHTS:
            if other == "described":
                continue
            self.assertGreater(described, score_from_signals({other: True}), other)

    def test_more_curation_never_scores_lower(self):
        keys = list(SIGNAL_WEIGHTS)
        previous = 0.0
        for i in range(len(keys) + 1):
            score = score_from_signals({k: True for k in keys[:i]})
            self.assertGreaterEqual(score, previous)
            previous = score


class TestSignalsComeFromWhatAdminsActuallyDo(CurationCase):

    def test_a_described_table_is_recognised(self):
        import store
        store.save_table_description(
            self.account_id, "SALES.ORDERS", description=DESCRIPTION)
        signals = signals_for(self.account_id)
        self.assertTrue(signals["SALES.ORDERS"]["described"])

    def test_a_placeholder_description_does_not_count_as_documentation(self):
        # "Orders table" tells the model nothing it could not read off the
        # name, and letting it score would make the backlog say a table is
        # done when nobody has explained it.
        import store
        store.save_table_description(
            self.account_id, "SALES.ORDERS", description="Orders table")
        self.assertLess(len("Orders table"), MIN_DESCRIPTION_CHARS)
        self.assertFalse(signals_for(self.account_id)["SALES.ORDERS"]["described"])

    def test_synonyms_and_column_synonyms_are_separate_signals(self):
        import store
        store.save_table_description(
            self.account_id, "SALES.ORDERS",
            description=DESCRIPTION, synonyms="sales, bookings")
        signals = signals_for(self.account_id)["SALES.ORDERS"]
        self.assertTrue(signals["synonyms"])
        self.assertFalse(signals["column_synonyms"])

    def test_a_table_modelled_as_an_entity_is_recognised(self):
        import store
        store.save_entity(
            self.account_id, entity_name="Order", table_name="ORDERS",
            schema_name="SALES", pk_column="ORDER_ID",
            display_name="Order", description="", entity_type="fact",
        )
        signals = signals_for(self.account_id)
        self.assertTrue(signals["SALES.ORDERS"]["modelled_as_entity"])
        # Recorded under the bare name too, because retrieval payloads carry
        # whichever form the KB build wrote.
        self.assertTrue(signals["ORDERS"]["modelled_as_entity"])

    def test_a_workspace_with_nothing_curated_scores_nothing(self):
        self.assertEqual(scores_for(self.account_id, use_cache=False), {})

    def test_a_store_failure_leaves_retrieval_working(self):
        import store
        store.save_table_description(
            self.account_id, "SALES.ORDERS", description=DESCRIPTION)
        with patch.object(store, "list_table_descriptions",
                          side_effect=RuntimeError("db down")):
            signals = signals_for(self.account_id)
        self.assertIsInstance(signals, dict)


class TestTheCache(CurationCase):

    def test_an_edit_is_visible_after_invalidation(self):
        import store
        self.assertEqual(scores_for(self.account_id), {})
        store.save_table_description(
            self.account_id, "SALES.ORDERS", description=DESCRIPTION)
        # Still cached as uncurated — this is what the admin route's
        # invalidate() call exists to prevent.
        self.assertEqual(scores_for(self.account_id), {})
        invalidate(self.account_id)
        self.assertGreater(scores_for(self.account_id).get("SALES.ORDERS", 0), 0)

    def test_invalidating_one_workspace_leaves_another_cached(self):
        import store
        other = f"acct-other-{uuid.uuid4().hex[:8]}"
        store.upsert_client(other, "portal")
        scores_for(self.account_id)
        scores_for(other)
        store.save_table_description(other, "X.Y", description=DESCRIPTION)
        invalidate(self.account_id)
        self.assertEqual(scores_for(other), {})


class TestTheOrderingIsNudgedNotOverridden(unittest.TestCase):

    def test_a_near_tie_goes_to_the_curated_table(self):
        hits = [
            {"fqn": "S.RAW", "_rerank_score": 0.52},
            {"fqn": "S.CURATED", "_rerank_score": 0.50},
        ]
        ordered = apply(hits, {"S.CURATED": 1.0, "S.RAW": 0.0})
        self.assertEqual([h["fqn"] for h in ordered], ["S.CURATED", "S.RAW"])

    def test_a_clear_relevance_gap_is_never_overturned(self):
        # The property that makes this safe. The cross-encoder separates
        # relevant from irrelevant by an order of magnitude; curation must not
        # be able to cross that, or a well-documented wrong table wins.
        hits = [
            {"fqn": "S.RELEVANT", "_rerank_score": 0.90},
            {"fqn": "S.CURATED_BUT_WRONG", "_rerank_score": 0.10},
        ]
        ordered = apply(hits, {"S.CURATED_BUT_WRONG": 1.0, "S.RELEVANT": 0.0})
        self.assertEqual(ordered[0]["fqn"], "S.RELEVANT")

    def test_the_boost_is_bounded_by_max_boost(self):
        hit = {"fqn": "S.A", "_rerank_score": 1.0}
        self.assertAlmostEqual(ordering_key(hit, {"S.A": 1.0}), 1.0 + MAX_BOOST)
        self.assertAlmostEqual(ordering_key(hit, {"S.A": 0.0}), 1.0)
        self.assertLess(MAX_BOOST, 0.5)

    def test_the_relevance_score_the_floor_reads_is_never_touched(self):
        # If curation raised _rerank_score, a curated table could clear the
        # relevance floor it should have been dropped by — the floor would
        # stop meaning "the reranker found this relevant".
        hits = [{"fqn": "S.A", "_rerank_score": 0.02},
                {"fqn": "S.B", "_rerank_score": 0.80}]
        apply(hits, {"S.A": 1.0, "S.B": 0.0})
        self.assertEqual(hits[0]["_rerank_score"], 0.02)
        self.assertEqual(hits[1]["_rerank_score"], 0.80)

    def test_the_curation_score_is_recorded_for_the_trace(self):
        hits = [{"fqn": "S.A", "_rerank_score": 0.5}]
        ordered = apply(hits, {"S.A": 0.65})
        self.assertEqual(ordered[0]["_curation_score"], 0.65)

    def test_an_unscored_candidate_is_neither_boosted_nor_reordered(self):
        # The reranker was unavailable. Curation is a tie-breaker between
        # relevance judgements, not a substitute for one.
        hits = [
            {"fqn": "S.SCORED", "_rerank_score": 0.4},
            {"fqn": "S.UNSCORED"},
        ]
        ordered = apply(hits, {"S.UNSCORED": 1.0})
        self.assertEqual([h["fqn"] for h in ordered], ["S.SCORED", "S.UNSCORED"])
        self.assertNotIn("_curation_score", ordered[1])

    def test_no_curation_anywhere_leaves_the_order_exactly_as_it_was(self):
        hits = [{"fqn": f"S.T{i}", "_rerank_score": 0.9 - i * 0.1} for i in range(4)]
        before = [h["fqn"] for h in hits]
        self.assertEqual([h["fqn"] for h in apply(hits, {})], before)

    def test_case_differences_in_the_table_name_still_match(self):
        hits = [{"fqn": "sales.orders", "_rerank_score": 0.5},
                {"fqn": "S.OTHER", "_rerank_score": 0.52}]
        ordered = apply(hits, {"SALES.ORDERS": 1.0})
        self.assertEqual(ordered[0]["fqn"], "sales.orders")


class TestTheRetrieverWiring(CurationCase):

    def _retriever(self):
        import core.vector_store as vs
        retriever = vs.QdrantKBRetriever.__new__(vs.QdrantKBRetriever)
        retriever._account_id = self.account_id
        return retriever

    def test_curation_reorders_what_the_reranker_returned(self):
        import store
        store.save_table_description(
            self.account_id, "SALES.ORDERS",
            description=DESCRIPTION, synonyms="sales, bookings")
        invalidate(self.account_id)

        hits = [
            {"fqn": "SALES.RAW_STAGING", "_rerank_score": 0.52},
            {"fqn": "SALES.ORDERS", "_rerank_score": 0.50},
        ]
        ordered = self._retriever()._apply_curation("revenue", hits)
        self.assertEqual(ordered[0]["fqn"], "SALES.ORDERS")

    def test_a_workspace_with_no_curation_is_left_alone(self):
        hits = [{"fqn": "S.A", "_rerank_score": 0.5},
                {"fqn": "S.B", "_rerank_score": 0.4}]
        ordered = self._retriever()._apply_curation("q", hits)
        self.assertEqual([h["fqn"] for h in ordered], ["S.A", "S.B"])

    def test_a_curation_failure_costs_the_ordering_not_the_answer(self):
        import core.vector_store as vs
        hits = [{"fqn": "S.A", "_rerank_score": 0.5},
                {"fqn": "S.B", "_rerank_score": 0.4}]
        with patch("core.curation_weight.scores_for",
                   side_effect=RuntimeError("store down")):
            ordered = self._retriever()._apply_curation("q", hits)
        self.assertEqual([h["fqn"] for h in ordered], ["S.A", "S.B"])
        self.assertTrue(vs)

    def test_saving_a_description_in_admin_changes_retrieval_order(self):
        # Write API to read API: the admin route must invalidate the cache,
        # or an admin describes a table and retrieval keeps ranking it as
        # undocumented for five minutes -- which reads as the description
        # having no effect at all.
        import asyncio
        import json as _json

        from unittest.mock import MagicMock

        import admin.routes as routes
        import store

        store.update_client_state(
            self.account_id, "READY", {"kb_tables": ["SALES.ORDERS"]})
        hits = [{"fqn": "SALES.RAW_STAGING", "_rerank_score": 0.52},
                {"fqn": "SALES.ORDERS", "_rerank_score": 0.50}]
        self.assertEqual(
            self._retriever()._apply_curation("q", list(hits))[0]["fqn"],
            "SALES.RAW_STAGING")

        req = MagicMock()
        req.query_params = {}

        async def _json_body():
            return {"table_name": "SALES.ORDERS", "description": DESCRIPTION,
                    "synonyms": "sales, bookings"}

        req.json = _json_body
        with patch.object(routes, "_is_auth", return_value=True):
            asyncio.run(routes.admin_setup_save_table_description(
                req, self.account_id))

        # Nothing handed in below this line.
        self.assertEqual(
            self._retriever()._apply_curation("q", list(hits))[0]["fqn"],
            "SALES.ORDERS")
        self.assertTrue(_json)




class TestTheSearchPathActuallyCallsIt(CurationCase):
    """A weighting step nothing calls is a weighting step that does nothing.

    The tests above exercise ``_apply_curation`` directly, which proves the
    reordering is right and proves nothing about whether retrieval reaches it.
    These run the real ``_hybrid_search`` with the vector store and the
    cross-encoder mocked at their boundaries — everything between them,
    including the curation step, is production code.
    """

    def _retriever(self):
        import core.vector_store as vs
        retriever = vs.QdrantKBRetriever.__new__(vs.QdrantKBRetriever)
        retriever._account_id = self.account_id
        return retriever

    @staticmethod
    def _candidates():
        return [
            {"fqn": "SALES.RAW_STAGING", "text": "staging", "doc_type": "kb"},
            {"fqn": "SALES.ORDERS", "text": "orders", "doc_type": "kb"},
        ]

    def _search(self, retriever, n=2):
        """Run the real _hybrid_search with only its boundaries mocked."""
        import core.vector_store as vs

        def _fake_rerank(query, candidates, top_n):
            # The reranker's verdict: staging edges ahead by a hair. Exactly
            # the near-tie curation exists to settle.
            scores = {"SALES.RAW_STAGING": 0.52, "SALES.ORDERS": 0.50}
            for candidate in candidates:
                candidate["_rerank_score"] = scores[candidate["fqn"]]
            return sorted(candidates, key=lambda c: -c["_rerank_score"])[:top_n]

        with (
            patch.object(type(retriever), "_search",
                         lambda self, q, pool, fqns, doc_types=None: self._candidates_ref),
            patch.object(vs, "_get_bm25", return_value=None),
            patch.object(vs, "_rerank", _fake_rerank),
            patch("core.identifier_intelligence.expand_question_for_retrieval",
                  lambda q: q),
        ):
            retriever._candidates_ref = self._candidates()
            return retriever._hybrid_search("revenue by customer", n, None)

    def test_an_uncurated_workspace_keeps_the_rerankers_order(self):
        results = self._search(self._retriever())
        self.assertEqual([r["fqn"] for r in results],
                         ["SALES.RAW_STAGING", "SALES.ORDERS"])

    def test_curating_a_table_changes_what_the_search_returns(self):
        import store
        store.save_table_description(
            self.account_id, "SALES.ORDERS",
            description=DESCRIPTION, synonyms="sales, bookings")
        invalidate(self.account_id)
        results = self._search(self._retriever())
        self.assertEqual(results[0]["fqn"], "SALES.ORDERS")
        # And the trace records why, so "why that table" is answerable from
        # the record rather than re-derived.
        self.assertGreater(results[0]["_curation_score"], 0)


class TestOrderingKeyDirectly(unittest.TestCase):
    """``ordering_key`` is public and callable outside ``apply``."""

    def test_an_unscored_candidate_sorts_last_rather_than_being_invented(self):
        # apply() filters these out before it gets here, so this branch is
        # only reachable by a direct caller — and a direct caller inventing a
        # relevance score from a curation score is exactly the inversion this
        # whole module is arranged to prevent.
        self.assertEqual(ordering_key({"fqn": "S.A"}, {"S.A": 1.0}), 0.0)
        self.assertEqual(
            ordering_key({"fqn": "S.A", "_rerank_score": None}, {"S.A": 1.0}), 0.0)

    def test_a_scored_candidate_with_no_curation_keeps_its_score(self):
        self.assertAlmostEqual(
            ordering_key({"fqn": "S.A", "_rerank_score": 0.42}, {}), 0.42)


class TestTheHeaderSpellingTheIndexActuallyCarries(unittest.TestCase):
    """The lookup has to survive the spellings a KB header really uses.

    core/schema.py writes `# DB.SCHEMA.TABLE` when a database is configured
    and `# [SCHEMA].[TABLE]` when one is not, and the retriever returns that
    line verbatim as the hit's fqn. The signals are keyed on what the stores
    hold — the bare or schema-qualified name; the entity graph has no database
    column at all. Matching the whole string only meant the bracketed form
    matched nothing on any tenant, and a three-part header never matched a
    two-part signal, so a table modelled as an entity with column roles scored
    zero unless it also carried a description. Silent: the boost became 1.0
    and nothing logged.
    """

    SCORES = {"SALES.ORDERS": 0.8, "ORDERS": 0.1}

    def _order(self, fqn):
        from core.curation_weight import ordering_key

        return ordering_key({"fqn": fqn, "_rerank_score": 1.0}, self.SCORES)

    def test_a_bracketed_header_gets_the_boost_its_curation_earned(self):
        self.assertEqual(self._order("[SALES].[ORDERS]"), self._order("SALES.ORDERS"))
        self.assertGreater(self._order("[SALES].[ORDERS]"), 1.0)

    def test_a_database_qualified_header_gets_it_too(self):
        self.assertEqual(self._order("MYDB.SALES.ORDERS"), self._order("SALES.ORDERS"))

    def test_a_quoted_header_gets_it_too(self):
        self.assertEqual(self._order('"sales"."orders"'), self._order("SALES.ORDERS"))

    def test_the_most_qualified_match_wins(self):
        from core.curation_weight import score_for_table

        # DB.SALES.ORDERS must resolve to the SALES.ORDERS entry, not to the
        # bare ORDERS one -- otherwise two tables of the same name in
        # different schemas would share a score.
        self.assertEqual(score_for_table(self.SCORES, "MYDB.SALES.ORDERS"), 0.8)
        self.assertEqual(score_for_table(self.SCORES, "MYDB.OTHER.ORDERS"), 0.1)

    def test_a_table_nobody_curated_is_still_unboosted(self):
        from core.curation_weight import score_for_table

        self.assertEqual(score_for_table(self.SCORES, "FIN.LEDGER"), 0.0)
        self.assertEqual(self._order("FIN.LEDGER"), 1.0)

    def test_an_empty_or_missing_name_scores_nothing(self):
        from core.curation_weight import score_for_table

        for value in ("", None, "  ", "..", "[].[]"):
            self.assertEqual(score_for_table(self.SCORES, value), 0.0, repr(value))


class TestEveryTestInThisFileRunsWhenTheFileIsRun(unittest.TestCase):
    """A stray unittest.main() sat in the middle of this module.

    Run as `python tests/test_curation_weight.py` it executed 25 of the 29
    tests and printed OK — silently skipping the two classes below it,
    including the one that proves the search path reaches the curation step.
    pytest collected all 29 either way, so CI never noticed.
    """

    def test_the_entry_point_guard_is_the_last_thing_in_the_file(self):
        from pathlib import Path

        source = Path(__file__).read_text(encoding="utf-8")
        # Split so this assertion does not match itself.
        needle = 'if __name__ == "__' + 'main__":'
        self.assertEqual(source.count(needle), 1)
        remaining = source[source.index(needle):]
        self.assertNotIn("\nclass ", remaining)
        self.assertNotIn("\ndef test", remaining)


if __name__ == "__main__":
    unittest.main()
