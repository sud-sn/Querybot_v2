"""
Answering-capability hardening — Phases A-C regression tests.

Covers:
  A  — behavioral front door (core/conversational.py + dispatcher wiring)
  B1 — retrieval relevance floor + weak_retrieval confidence penalty
  B2 — prompt-size clamps (per-doc section trim + final context cap)
  B3 — semantic-plan gap-fill fallback + entity-without-table health warning
  C1 — channel-agnostic why/causal insight route
  C2 — compound-question detection + guided split
"""

import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def _src(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# Phase A — behavioral front door
# ══════════════════════════════════════════════════════════════════════════════

class ConversationalDetectionTests(unittest.TestCase):
    def _kind(self, text):
        from core.conversational import detect_conversational
        return detect_conversational(text)

    def test_smalltalk_kinds(self):
        self.assertEqual(self._kind("hi"), "greeting")
        self.assertEqual(self._kind("Good morning!"), "greeting")
        self.assertEqual(self._kind("thanks a lot"), "thanks")
        self.assertEqual(self._kind("bye"), "goodbye")
        self.assertEqual(self._kind("this is useless"), "frustration")

    def test_data_aware_kinds(self):
        self.assertEqual(self._kind("what data do you have"), "data_inventory")
        self.assertEqual(self._kind("show me the data"), "vague")

    def test_workspace_guide_kinds(self):
        self.assertEqual(self._kind("what can you do"), "capability_overview")
        self.assertEqual(self._kind("what all can you do"), "capability_overview")
        self.assertEqual(self._kind("explain this business"), "business_overview")
        self.assertEqual(self._kind("explain me about the business"), "business_overview")
        self.assertEqual(self._kind("explain the data available"), "data_inventory")
        self.assertEqual(self._kind("explain me about the data available"), "data_inventory")
        self.assertEqual(
            self._kind("which tables are available and what do they mean"),
            "table_meanings",
        )
        self.assertEqual(
            self._kind("what are the available tables and their business meaning"),
            "table_meanings",
        )
        self.assertEqual(
            self._kind("how does the semantic layer work"),
            "semantic_explainer",
        )
        self.assertEqual(
            self._kind("explain me about the semantic layer"),
            "semantic_explainer",
        )
        self.assertEqual(
            self._kind("give me sample questions"),
            "question_examples",
        )
        self.assertEqual(
            self._kind("what types of NL questions can I ask"),
            "question_examples",
        )

    def test_data_questions_never_match(self):
        for q in (
            "what is my revenue this month",
            "show top 10 customers by sales",
            "hi-tech products revenue by region",
            "good count of orders by day",
            "why did sales drop in Q3",
            "thanks to the discount, did margin change?",
        ):
            self.assertIsNone(self._kind(q), q)

    def test_long_messages_skipped(self):
        self.assertIsNone(self._kind("hello " * 50))

    def test_dispatcher_wires_front_door(self):
        src = _src("core/dispatcher.py")
        self.assertIn("detect_conversational", src)
        # State checks and registration MUST come before the small-talk branch,
        # otherwise unregistered users get a greeting instead of a registration link.
        self.assertLess(
            src.index('state = get_state(account_id).get("state", "NEW")'),
            src.index('("greeting", "thanks", "goodbye", "frustration")'),
        )
        # …and data-aware kinds inside READY, after the clarification checks.
        self.assertLess(
            src.index("was_recently_expired(account_id, event.user_id)"),
            src.index('_conv_kind in (*_WORKSPACE_GUIDE_KINDS, "opinion", "vague")'),
        )
        # data-aware branch fires before the DDL guard / SQL enqueue
        self.assertLess(
            src.index('_conv_kind in (*_WORKSPACE_GUIDE_KINDS, "opinion", "vague")'),
            src.index("if is_ddl_attempt(text):"),
        )

    def test_whoami_and_status_bypass_workspace_state_guard(self):
        # whoami/status are diagnostic commands — status's whole purpose is
        # to show setup progress, so both must be reachable in every
        # workspace state (NEW/SCHEMA_READY/KB_BUILDING/READY), not just
        # once the workspace is fully READY. A prior fix moved the state
        # guard above the conversational front door to close a registration
        # bypass, which had the side effect of also gating whoami/status
        # behind "workspace not ready yet" — this pins them ahead of it.
        src = _src("core/dispatcher.py")
        state_guard_pos = src.index('if state in ("NEW", "SCHEMA_READY"):')
        self.assertLess(src.index('text.lower() == "whoami"'), state_guard_pos)
        self.assertLess(src.index('text.lower() == "status"'), state_guard_pos)
        # Only one definition of each command should remain (no leftover
        # duplicate further down from before the reorder).
        self.assertEqual(src.count('text.lower() == "whoami"'), 1)
        self.assertEqual(src.count('text.lower() == "status"'), 1)


# ══════════════════════════════════════════════════════════════════════════════
# Phase B1 — relevance floor + weak-retrieval confidence
# ══════════════════════════════════════════════════════════════════════════════

def _floor(hits):
    from core.vector_store import QdrantKBRetriever
    r = QdrantKBRetriever.__new__(QdrantKBRetriever)
    r.last_retrieval_weak = False
    return r._apply_relevance_floor(hits), r.last_retrieval_weak


class RelevanceFloorTests(unittest.TestCase):
    def test_irrelevant_table_dropped_relevant_kept(self):
        hits = [
            {"fqn": "ERP.SALES", "_rerank_score": 0.9, "text": "s"},
            {"fqn": "ERP.WEATHER", "_rerank_score": 0.01, "text": "w"},
        ]
        out, weak = _floor(hits)
        fqns = {h["fqn"] for h in out}
        self.assertEqual(fqns, {"ERP.SALES"})
        self.assertFalse(weak)

    def test_all_below_floor_keeps_best_and_flags_weak(self):
        hits = [
            {"fqn": "ERP.A", "_rerank_score": 0.02, "text": "a"},
            {"fqn": "ERP.B", "_rerank_score": 0.01, "text": "b"},
        ]
        out, weak = _floor(hits)
        self.assertEqual([h["fqn"] for h in out], ["ERP.A"])  # never empties
        self.assertTrue(weak)

    def test_global_docs_always_survive(self):
        hits = [
            {"fqn": "_global", "_rerank_score": 0.001, "text": "vocab"},
            {"fqn": "ERP.A", "_rerank_score": 0.01, "text": "a"},
            {"fqn": "ERP.B", "_rerank_score": 0.9, "text": "b"},
        ]
        out, weak = _floor(hits)
        fqns = [h["fqn"] for h in out]
        self.assertIn("_global", fqns)
        self.assertIn("ERP.B", fqns)
        self.assertNotIn("ERP.A", fqns)
        self.assertFalse(weak)

    def test_table_survives_if_any_chunk_relevant(self):
        hits = [
            {"fqn": "ERP.A", "_rerank_score": 0.01, "text": "weak section"},
            {"fqn": "ERP.A", "_rerank_score": 0.8, "text": "strong section"},
        ]
        out, weak = _floor(hits)
        self.assertEqual(len(out), 2)  # both chunks kept — floor is per table
        self.assertFalse(weak)

    def test_unscored_hits_never_floored(self):
        hits = [{"fqn": "ERP.A", "text": "no score"}]
        out, weak = _floor(hits)
        self.assertEqual(out, hits)
        self.assertFalse(weak)


class WeakRetrievalConfidenceTests(unittest.TestCase):
    def test_weak_retrieval_penalises_score_and_warns(self):
        from core.answer_confidence import build_answer_confidence
        base = build_answer_confidence(row_count=10)
        weak = build_answer_confidence(row_count=10, weak_retrieval=True)
        self.assertEqual(base["score"] - weak["score"], 20)
        self.assertTrue(any("weakly" in w for w in weak["warnings"]))

    def test_pipeline_threads_weak_flag(self):
        src = _src("core/query_pipeline.py")
        self.assertIn('getattr(retriever, "last_retrieval_weak", False)', src)
        self.assertIn('"weak_retrieval": _weak_retrieval', src)
        rsrc = _src("core/result_renderer.py")
        self.assertIn('weak_retrieval=bool(confidence_context.get("weak_retrieval"))', rsrc)


# ══════════════════════════════════════════════════════════════════════════════
# Phase B2 — prompt-size clamps
# ══════════════════════════════════════════════════════════════════════════════

class PromptClampTests(unittest.TestCase):
    DOC = (
        "# ERP.SALES\n## Overview\n" + "x" * 8000
        + "\n## Columns\nCOL_A int\n## Join Keys\nK1\n## Sample Data\n"
        + "y" * 4000 + "\n## Query Patterns\n" + "q" * 3000
        + "\n## Business Synonyms\n" + "z" * 2000
    )

    def test_doc_clamp_drops_droppable_sections_keeps_columns(self):
        from core.pipeline_helpers import _clamp_kb_doc
        out = _clamp_kb_doc(self.DOC, cap=9000)
        self.assertLessEqual(len(out), 9000 + 50)
        self.assertIn("## Columns", out)
        self.assertIn("## Join Keys", out)
        self.assertNotIn("## Sample Data", out)

    def test_business_synonyms_is_the_last_droppable_section_to_go(self):
        """It is the only section mapping plain-English terms to exact columns,
        and the only one carrying the 'could be confused with' warnings — and
        the downstream COLUMN SYNONYM MAP is rebuilt from this clamped text.
        Dropping in file order made it the FIRST casualty, on exactly the
        biggest, most column-dense tables where disambiguation matters most."""
        from core.pipeline_helpers import _clamp_kb_doc

        out = _clamp_kb_doc(self.DOC, cap=9000)
        self.assertIn("## Business Synonyms", out)
        self.assertNotIn("## Sample Data", out)

    def test_the_cheapest_sections_go_first(self):
        from core.pipeline_helpers import _clamp_kb_doc

        # Only enough must go to fit; sample data alone is 4000 chars.
        out = _clamp_kb_doc(self.DOC, cap=len(self.DOC) - 3000)
        self.assertNotIn("## Sample Data", out)
        self.assertIn("## Query Patterns", out)
        self.assertIn("## Overview", out)
        self.assertIn("## Business Synonyms", out)

    def test_doc_clamp_untouched_when_under_cap(self):
        from core.pipeline_helpers import _clamp_kb_doc
        doc = "# T\n## Columns\nabc"
        self.assertEqual(_clamp_kb_doc(doc), doc)

    def test_context_clamp_preserves_head(self):
        from core.pipeline_helpers import _clamp_prompt_context
        ctx = "PRIORITY-SEMANTIC-BLOCK " + "z" * 200000
        out = _clamp_prompt_context(ctx, cap=50000)
        self.assertTrue(out.startswith("PRIORITY-SEMANTIC-BLOCK"))
        self.assertIn("truncated for prompt size", out)
        self.assertLess(len(out), 51000)

    def test_pipeline_applies_clamps(self):
        src = _src("core/query_pipeline.py")
        self.assertIn("_clamp_kb_doc(d) for d in (pinned + table_kbs)[:7]", src)
        self.assertIn("context_with_terms = _clamp_prompt_context(context_with_terms)", src)
        # final clamp must run before the system prompt is built
        self.assertLess(
            src.index("context_with_terms = _clamp_prompt_context(context_with_terms)"),
            src.index("system = build_sql_system_prompt("),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Phase B3 — gap-fill fallback + entity health warning
# ══════════════════════════════════════════════════════════════════════════════

class GapFillFallbackTests(unittest.TestCase):
    def test_fallback_block_present_and_graph_gated(self):
        src = _src("core/query_pipeline.py")
        self.assertIn('if not _graph_ctx.get("enabled")', src)
        self.assertIn("_plan_gap_docs", src)

    def test_entity_without_table_flagged(self):
        from core.kb_quality import evaluate_kb_quality
        model = {"tables": [{"qualified_name": "ERP.SALES", "type": "dimension",
                             "fields": [], "measures": [], "dimensions": []}],
                 "relationships": []}
        with patch("store.list_entities", return_value=[
            {"entity_name": "Customer", "table_name": ""},
            {"entity_name": "Product", "table_name": "DIM_PRODUCT"},
        ]):
            report = evaluate_kb_quality(model, account_id="acct-1")
        issues = [i for i in report["issues"] if i["code"] == "entity_without_table"]
        self.assertEqual(len(issues), 1)
        self.assertIn("Customer", issues[0]["message"])

    def test_entity_check_skipped_without_account(self):
        from core.kb_quality import evaluate_kb_quality
        model = {"tables": [{"qualified_name": "ERP.SALES", "type": "dimension",
                             "fields": [], "measures": [], "dimensions": []}],
                 "relationships": []}
        report = evaluate_kb_quality(model)
        codes = {i["code"] for i in report["issues"]}
        self.assertNotIn("entity_without_table", codes)


# ══════════════════════════════════════════════════════════════════════════════
# Phase C1 — channel-agnostic why route
# ══════════════════════════════════════════════════════════════════════════════

class CausalRouteTests(unittest.TestCase):
    def test_causal_detection_strict(self):
        from core.insight import is_causal_question
        for q in ("why did revenue drop last month",
                  "what drove the decline in sales",
                  "what changed in Q3",
                  "root cause of the margin dip"):
            self.assertTrue(is_causal_question(q), q)
        for q in ("revenue by region",
                  "analyze revenue by region",     # broad word ≠ causal
                  "explain the numbers",
                  "show top 10 customers"):
            self.assertFalse(is_causal_question(q), q)

    def test_insight_markdown_formatting(self):
        from core.query_pipeline import _format_insight_markdown
        text = _format_insight_markdown({
            "headline": "Revenue fell 12%",
            "body": "Driven by APAC.",
            "bullets": ["APAC -20%", "EU flat"],
            "next_step": "Drill into APAC",
        })
        self.assertIn("*Revenue fell 12%*", text)
        self.assertIn("• APAC -20%", text)
        self.assertIn("Next step", text)
        self.assertEqual(_format_insight_markdown({}), "")

    def test_pipeline_hooks_after_both_success_paths(self):
        src = _src("core/query_pipeline.py")
        self.assertIn("_why_mode = bool(not is_clarification and is_causal_question(question))", src)
        self.assertEqual(src.count("await _send_why_insight("), 2)  # metric + main paths
        # insight is sent after the factual answer, never instead of it
        self.assertLess(
            src.index("await _send_results(event, adapter, question, rows, sql_from_metric"),
            src.index("await _send_why_insight("),
        )

    def test_send_why_insight_prefers_native_analysis_channel(self):
        src = _src("core/query_pipeline.py")
        self.assertIn('getattr(adapter, "send_analysis_response", None)', src)


# ══════════════════════════════════════════════════════════════════════════════
# Phase C2 — compound question split
# ══════════════════════════════════════════════════════════════════════════════

class CompoundQuestionTests(unittest.TestCase):
    def _dc(self, text):
        from core.conversational import detect_compound_question
        return detect_compound_question(text)

    def test_detects_two_independent_asks(self):
        self.assertEqual(
            self._dc("revenue by region and also top 10 customers"),
            ("revenue by region", "top 10 customers"),
        )
        self.assertIsNotNone(self._dc("show revenue by month; list top 5 products by sales"))
        self.assertIsNotNone(self._dc("total sales this year as well as count of new customers"))
        self.assertIsNotNone(self._dc("gross margin by product plus show top 10 customers"))

    def test_single_intent_never_split(self):
        for q in (
            "revenue and cost by region",              # bare "and" is not a joiner
            "show revenue plus tax by region",          # arithmetic "plus"
            "revenue by region and also by product",    # grouping continuation
            "compare revenue and cogs for each customer",
            "top customers this month",
        ):
            self.assertIsNone(self._dc(q), q)

    def test_dispatcher_offers_split_and_runs_choice_standalone(self):
        src = _src("core/dispatcher.py")
        self.assertIn("detect_compound_question", src)
        self.assertIn('"compound_split"', src)
        # the chosen half must run standalone, not combined with the original
        self.assertLess(
            src.index('cmeta.get("source") == "compound_split"'),
            src.index("combine_with_clarification(\n"
                      if "combine_with_clarification(\n" in src
                      else "combine_with_clarification("),
        )
        # never auto-fan-out: the split branch returns after prompting
        split_block = src[src.index("detect_compound_question(text)"):]
        self.assertIn("send_clarification_prompt", split_block)


if __name__ == "__main__":
    unittest.main()
