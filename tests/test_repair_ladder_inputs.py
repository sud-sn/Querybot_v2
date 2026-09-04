"""
tests/test_repair_ladder_inputs.py

Three things the repair ladder needed and was not given.

SESSION CONTEXT. The column-repair note tells the model to "check the 'Session
context' section — if the previous turn returned a column that represents the
same concept, reuse that EXACT column name". That section is built only when
`conversation_history` is passed to `build_sql_system_prompt`, and neither
repair call passed it. The instruction pointed at a heading that was not in the
prompt, on precisely the failure — an invented column name — where the previous
turn's answer is the best available evidence.

THE PROVEN-WRONG FILTER. `find_unmatched_literals` parses the executed SQL's
WHERE clauses and tests each string literal against the per-tenant value index,
reporting only columns the index covers — so a hit is proof the value is not in
the data, and it carries the closest real values. That proof was computed on the
zero-row path and spent entirely on the apology the user reads. The repair never
saw it, so a retry had to guess which predicate was wrong.

REPAIR TOKENS. Both repair calls unpacked their usage as `_, _`. The cost signal
was blind to exactly the calls that make a question expensive.
"""

import ast
import inspect
import unittest
from pathlib import Path
from unittest.mock import patch

import core.query_pipeline as qp

ROOT = Path(__file__).resolve().parents[1]


class TheRepairPromptCarriesTheSessionContext(unittest.TestCase):

    def setUp(self):
        self.source = inspect.getsource(qp._handle_query_impl)

    def test_every_sql_prompt_build_passes_conversation_history(self):
        """The first attempt always did; the two repairs did not."""
        builds = self.source.count("build_sql_system_prompt(")
        passes = self.source.count("conversation_history=_conv_history")
        self.assertEqual(
            passes, builds,
            f"{builds} prompt builds but only {passes} pass conversation history",
        )

    def test_there_are_three_of_them(self):
        """Guards the guard: 0 == 0 would satisfy the equality above."""
        self.assertEqual(self.source.count("build_sql_system_prompt("), 3)

    def test_the_note_that_needs_it_still_points_at_that_section(self):
        self.assertIn("'Session context' section", self.source)


class TheProvenWrongFilterReachesTheRepair(unittest.TestCase):

    PROOF = [{
        "column": "SCHEDULE",
        "business_name": "DEA schedule",
        "literal": "CIII-N",
        "closest": ["CIII", "CV", "NONE"],
    }]

    def _lines(self, proof):
        with patch("core.value_resolver.find_unmatched_literals", return_value=proof):
            return qp._unmatched_literal_repair_lines("SELECT 1 WHERE X='Y'", "acct")

    def test_it_names_the_column_and_the_absent_value(self):
        lines = self._lines(self.PROOF)
        self.assertIn("DEA schedule", lines)
        self.assertIn("CIII-N", lines)

    def test_it_offers_the_real_values_the_index_knows(self):
        lines = self._lines(self.PROOF)
        self.assertIn("CIII, CV, NONE", lines)

    def test_it_forbids_inventing_a_variant(self):
        """The failure mode being repaired is an invented value."""
        self.assertIn("Do not invent a variant", self._lines(self.PROOF))

    def test_a_column_with_no_close_values_still_says_stop_filtering_on_it(self):
        lines = self._lines([{"column": "X", "literal": "zzz", "closest": []}])
        self.assertIn("Do not filter on that value", lines)

    def test_nothing_proven_means_nothing_said(self):
        """An unproven suggestion would send the repair chasing a good filter."""
        self.assertEqual(self._lines([]), "")

    def test_a_broken_value_index_is_silent_rather_than_fatal(self):
        with patch("core.value_resolver.find_unmatched_literals",
                   side_effect=RuntimeError("index missing")):
            self.assertEqual(
                qp._unmatched_literal_repair_lines("SELECT 1", "acct"), "")

    def test_it_is_wired_into_the_zero_row_repair_note(self):
        source = inspect.getsource(qp._handle_query_impl)
        self.assertIn("_unmatched_literal_repair_lines(sql, account_id)", source)
        self.assertIn("+ _literal_lines", source)


class RepairTokensAreCounted(unittest.TestCase):
    """Both calls unpacked usage as `_, _`, so their cost vanished."""

    def setUp(self):
        self.source = inspect.getsource(qp._handle_query_impl)

    def test_neither_repair_discards_its_usage(self):
        self.assertNotIn("sql_retry, _, _ = await llm_complete(", self.source)
        self.assertNotIn("_progressive_sql, _, _ = await llm_complete(", self.source)

    def test_both_are_added_to_the_question_total(self):
        for name in ("_retry_tok_in", "_retry_tok_out", "_prog_tok_in", "_prog_tok_out"):
            with self.subTest(token=name):
                self.assertIn(f"+= {name}", self.source)

    def test_every_captured_repair_token_is_actually_accumulated(self):
        """Capturing into a name and never adding it would look identical at
        the call site and change nothing in the books."""
        tree = ast.parse((ROOT / "core" / "query_pipeline.py").read_text(encoding="utf-8"))
        captured, accumulated = set(), set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Tuple):
                        for element in target.elts:
                            if isinstance(element, ast.Name) and "_tok_" in element.id:
                                captured.add(element.id)
            if isinstance(node, ast.AugAssign) and isinstance(node.value, ast.Name):
                if "_tok_" in node.value.id:
                    accumulated.add(node.value.id)
        self.assertTrue(captured, "no repair usage is captured at all")
        self.assertEqual(
            sorted(captured - accumulated), [],
            f"captured but never counted: {sorted(captured - accumulated)}",
        )


if __name__ == "__main__":
    unittest.main()
