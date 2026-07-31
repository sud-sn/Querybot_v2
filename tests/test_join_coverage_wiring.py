"""
Lossy-join caveat wiring: core.graph_resolver.resolve_for_question's already-
computed resolved_edges get threaded into confidence_context (one line, no
new plumbing) and consumed by core/result_renderer.py::_send_results
alongside the date-coverage check, both feeding the same coverage_caveats
list.
"""

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class GraphEdgesThreadingWiringTests(unittest.TestCase):
    def setUp(self):
        self.source = (ROOT / "core" / "query_pipeline.py").read_text(encoding="utf-8")

    def test_confidence_context_carries_resolved_graph_edges(self):
        start = self.source.index("_confidence_context = {")
        end = self.source.index("\n    }", start)
        block = self.source[start:end]
        self.assertIn('"graph_edges"', block)
        self.assertIn('(_graph_ctx or {}).get("resolved_edges")', block)


class JoinCoverageResultRendererWiringTests(unittest.TestCase):
    def setUp(self):
        self.source = (ROOT / "core" / "result_renderer.py").read_text(encoding="utf-8")

    def test_join_coverage_check_reads_graph_edges_from_confidence_context(self):
        self.assertIn('confidence_context.get("graph_edges")', self.source)

    def test_join_coverage_check_runs_before_both_rich_and_plaintext_paths(self):
        check_pos = self.source.index("check_join_coverage(")
        rich_pos = self.source.index('rich_sender = getattr(adapter, "send_assistant_response"')
        self.assertLess(check_pos, rich_pos)

    def test_join_coverage_check_never_raises_uncaught(self):
        start = self.source.index("_graph_edges = confidence_context.get")
        end = self.source.index('rich_sender = getattr(adapter, "send_assistant_response"')
        block = self.source[start:end]
        self.assertIn("try:", block)
        self.assertIn("except Exception", block)

    def test_join_coverage_extends_the_same_coverage_caveats_list_as_date_coverage(self):
        self.assertIn("coverage_caveats.extend(check_join_coverage(", self.source)
        # And the date-coverage check appends to the same list, not a
        # separate one -- both must feed one shared, ordered list.
        self.assertIn("coverage_caveats.append(_gap.message)", self.source)

    def test_no_new_database_query_for_join_coverage(self):
        # Unlike date coverage (which explicitly needs one), the join
        # coverage check must be a pure read of already-persisted/resolved
        # data -- no run_query or execute_governed_query call in its block.
        start = self.source.index("_graph_edges = confidence_context.get")
        end = self.source.index('rich_sender = getattr(adapter, "send_assistant_response"')
        block = self.source[start:end]
        self.assertNotIn("run_query(", block)
        self.assertNotIn("execute_governed_query(", block)


if __name__ == "__main__":
    unittest.main()
