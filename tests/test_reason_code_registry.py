"""
tests/test_reason_code_registry.py

A rejection code the repair ladder had never heard of was a dead end.

`core/validator.py` emits 46 distinct rejection codes. The pipeline held its own
hand-maintained list of which were worth retrying — in TWO places, which had
already drifted apart by one code (`order_alias_mismatch` was in the tuple and
not the set) — and nothing tied either list to the codes actually emitted.

The failure was silent and one-directional. A code in neither list defaults to
unrepairable, so every rejection added to the validator after those lists were
written became a dead end the moment it shipped: the user got "I couldn't
answer that" for a fault the model could have fixed on a second pass. 21 of the
46 were in that state, including `cartesian_join`, `select_star` and all five
`multi_fact_*` contracts — the codes that say most precisely what is wrong.

So the lists are inverted. Everything is repairable unless it is named
terminal, and this file enumerates the validator's own emitted codes against
the registry so a new one cannot be added without landing in it.

The enumeration is the point. A test that merely asserted a hand-written list
matched another hand-written list would have passed throughout.
"""

import ast
import unittest
from pathlib import Path

from core.validator import (
    ALL_REASON_CODES,
    PIPELINE_REASON_CODES,
    REPAIRABLE_REASON_CODES,
    TERMINAL_REASON_CODES,
)

ROOT = Path(__file__).resolve().parents[1]


def _codes_emitted_by_the_validator() -> set[str]:
    """Every string this module hands back as a rejection code.

    Two shapes: the third positional of SqlValidationResult(...), and the
    "code" key of the error dicts the detector functions return.
    """
    tree = ast.parse((ROOT / "core" / "validator.py").read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "id", "") == "SqlValidationResult"
                and len(node.args) >= 3
                and isinstance(node.args[2], ast.Constant)
                and isinstance(node.args[2].value, str)):
            found.add(node.args[2].value)
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (isinstance(key, ast.Constant) and key.value == "code"
                        and isinstance(value, ast.Constant)
                        and isinstance(value.value, str)):
                    found.add(value.value)
    return found - {"ok", ""}


class TheRegistryMatchesWhatIsActuallyEmitted(unittest.TestCase):
    """The check that did not exist, and whose absence made dead ends silent."""

    def setUp(self):
        self.emitted = _codes_emitted_by_the_validator()

    def test_the_enumeration_finds_a_realistic_number_of_codes(self):
        """Guards the guard: a broken AST walk would pass everything below."""
        self.assertGreater(len(self.emitted), 30)
        self.assertIn("unknown_table", self.emitted)
        self.assertIn("fanout_aggregate", self.emitted)

    def test_no_emitted_code_is_missing_from_the_registry(self):
        missing = sorted(self.emitted - ALL_REASON_CODES)
        self.assertEqual(missing, [], f"emitted but unregistered: {missing}")

    def test_the_registry_carries_nothing_that_is_never_emitted(self):
        stale = sorted(ALL_REASON_CODES - self.emitted)
        self.assertEqual(stale, [], f"registered but never emitted: {stale}")

    def test_every_code_is_either_repairable_or_terminal(self):
        """The property that makes a new dead end impossible to add quietly."""
        for code in sorted(ALL_REASON_CODES | PIPELINE_REASON_CODES):
            with self.subTest(code=code):
                self.assertNotEqual(
                    code in REPAIRABLE_REASON_CODES,
                    code in TERMINAL_REASON_CODES,
                    f"{code} must be in exactly one of the two sets",
                )


class TerminalMeansTerminalForAReason(unittest.TestCase):

    def test_the_terminal_set_stays_small(self):
        """It is an exception list. If it grows, the default has gone wrong."""
        self.assertLessEqual(len(TERMINAL_REASON_CODES), 5)

    def test_a_policy_refusal_is_not_retried(self):
        """Rewriting the query cannot grant access the user does not have."""
        self.assertIn("access_denied", TERMINAL_REASON_CODES)

    def test_an_explicit_decline_is_not_retried(self):
        """Asking again asks the same question of the same context."""
        self.assertIn("cannot_generate", TERMINAL_REASON_CODES)

    def test_ddl_is_terminal_as_a_governance_stance(self):
        """Not because it is unfixable — `not_select` is the same shape of slip
        and IS repairable. A model emitting DROP against a governed warehouse
        is an anomaly worth surfacing rather than smoothing over."""
        self.assertIn("ddl", TERMINAL_REASON_CODES)
        self.assertIn("not_select", REPAIRABLE_REASON_CODES)


class TheCodesThatUsedToDeadEnd(unittest.TestCase):
    """Each says precisely what is wrong, and none was ever retried."""

    RECOVERED = (
        "cartesian_join",
        "missing_join_condition",
        "select_star",
        "raw_fact_to_fact_join",
        "graph_join_missing",
        "multi_fact_not_aggregated",
        "bridge_allocation_missing",
        "derived_measure_mismatch",
    )

    def test_they_are_repairable_now(self):
        for code in self.RECOVERED:
            with self.subTest(code=code):
                self.assertIn(code, REPAIRABLE_REASON_CODES)

    def test_they_are_genuinely_emitted_rather_than_invented_here(self):
        emitted = _codes_emitted_by_the_validator()
        for code in self.RECOVERED:
            with self.subTest(code=code):
                self.assertIn(code, emitted)


class ThePipelineUsesTheRegistry(unittest.TestCase):
    """Wiring: the two hand-maintained copies had to actually go."""

    def setUp(self):
        self.source = (ROOT / "core" / "query_pipeline.py").read_text(encoding="utf-8")

    def test_it_imports_the_shared_set(self):
        self.assertIn("REPAIRABLE_REASON_CODES as _sql_repair_reason_codes", self.source)

    def test_the_hand_maintained_tuple_is_gone(self):
        self.assertNotIn("retryable = (not ok and (last_code or code) in (", self.source)

    def test_the_second_hand_maintained_copy_is_gone_too(self):
        """It had already drifted from the first by one code."""
        self.assertNotIn('_sql_repair_reason_codes = {', self.source)


if __name__ == "__main__":
    unittest.main()
