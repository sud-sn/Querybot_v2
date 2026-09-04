"""
tests/test_table_grain_context.py

What one row of a fact table represents never reached the model.

It decides whether "average price" means per order or per order LINE, and it
cannot be read off column names — which is exactly the error an analyst catches
by saying "wait, one fill has several revenue rows". The product derives it
(core/semantic_model.py _infer_table_grain, whose `_LIN_`/`PONR`/`POSX` rules
resolve real Infor M3 fact tables to "one row per transaction line"), stores it
on every table of the semantic model, and then used it for a count-target hint
string and a KB-quality warning. Nothing put it in the SQL prompt.

This is deliberately ADVISORY, not a validator gate, and the tests pin why:

  - `grain_columns` is `pk_columns` renamed, not a derived grain, and is
    dropped before it reaches the entity graph;
  - `grain_confidence` on the classifier branch is the ROLE score;
  - run over the real Test schema the classifier calls an order header a
    dimension.

Gating on that would reject correct SQL. So the rule here is: state a grain
when one has been established, stay silent when it has not, and never assert a
grain nobody confirmed.
"""

import json
import unittest
from pathlib import Path

from core.semantic_model import build_grain_context, grain_is_sub_event

ROOT = Path(__file__).resolve().parents[1]


def _model(*tables):
    return {"tables": list(tables)}


def _table(name, grain, status="generated"):
    return {
        "qualified_name": f"DW.{name}", "table": name,
        "grain": grain, "grain_status": status,
    }


class ItSpeaksOnlyWhenSomethingWasEstablished(unittest.TestCase):

    def test_an_unestablished_grain_says_nothing(self):
        """`needs_admin_context` is the honest answer, not a sentence to print."""
        model = _model(_table("F_RX_FILL", "needs_admin_context", "needs_review"))
        self.assertEqual(build_grain_context(model, {"DW.F_RX_FILL"}), "")

    def test_needs_review_is_silent_even_with_prose_attached(self):
        model = _model(_table("F_RX_FILL", "one row per transaction line", "needs_review"))
        self.assertEqual(build_grain_context(model, {"DW.F_RX_FILL"}), "")

    def test_the_real_test_account_produces_nothing(self):
        """Every fact on this account is needs_admin_context. Executed, not assumed."""
        model = json.loads(
            (ROOT / "clients" / "Test" / "kb" / "_semantic_model.json")
            .read_text(encoding="utf-8")
        )
        names = {
            str(t.get("qualified_name") or "") for t in model["tables"]
            if t.get("type") == "fact"
        }
        self.assertTrue(names, "fixture should still contain fact tables")
        self.assertEqual(build_grain_context(model, names), "")

    def test_generic_dimension_grain_alone_is_not_worth_the_prompt(self):
        """Every dimension gets the same sentence; it says nothing new."""
        model = _model(_table("CUST_DMS", "one row per lookup member"))
        self.assertEqual(build_grain_context(model, {"DW.CUST_DMS"}), "")

    def test_a_real_grain_carries_its_generic_neighbours_along(self):
        model = _model(
            _table("ORD_LIN_FCT", "one row per transaction line"),
            _table("CUST_DMS", "one row per lookup member"),
        )
        out = build_grain_context(model, {"DW.ORD_LIN_FCT", "DW.CUST_DMS"})
        self.assertIn("one row per transaction line", out)
        self.assertIn("CUST_DMS", out)


class ItSaysWhoEstablishedTheGrain(unittest.TestCase):

    def test_an_administrator_confirmation_is_marked_as_one(self):
        model = _model(_table("RX_FILL_FCT", "one row per dispensed fill", "approved"))
        out = build_grain_context(model, {"DW.RX_FILL_FCT"})
        self.assertIn("confirmed by an administrator", out)

    def test_an_inferred_grain_is_not_dressed_up_as_confirmed(self):
        model = _model(_table("ORD_LIN_FCT", "one row per transaction line"))
        out = build_grain_context(model, {"DW.ORD_LIN_FCT"})
        self.assertIn("one row per transaction line", out)
        self.assertNotIn("confirmed by an administrator", out)

    def test_an_approved_grain_speaks_even_when_it_reads_generic(self):
        """A human typed it, so it outranks the "says nothing new" rule."""
        model = _model(_table("CUST_DMS", "one row per lookup member", "approved"))
        self.assertIn("CUST_DMS", build_grain_context(model, {"DW.CUST_DMS"}))


class TheSubEventWarning(unittest.TestCase):
    """The whole point: a LINE grain is finer than the unit a user names."""

    def test_a_line_grain_earns_the_aggregate_first_instruction(self):
        model = _model(_table("ORD_LIN_FCT", "one row per transaction line"))
        out = build_grain_context(model, {"DW.ORD_LIN_FCT"})
        self.assertIn("COUNT(DISTINCT", out)
        self.assertIn("several rows per business event", out)

    def test_an_event_grain_does_not(self):
        """A false warning on a correct answer costs more than a missing one."""
        model = _model(_table("ORD_FCT", "one row per business event"))
        out = build_grain_context(model, {"DW.ORD_FCT"})
        self.assertNotIn("COUNT(DISTINCT", out)

    def test_the_predicate_is_narrow_on_purpose(self):
        self.assertTrue(grain_is_sub_event("one row per transaction line"))
        self.assertTrue(grain_is_sub_event("One row per ORDER LINE"))
        self.assertFalse(grain_is_sub_event("one row per business event"))
        self.assertFalse(grain_is_sub_event("one row per snapshot grain"))
        self.assertFalse(grain_is_sub_event(""))


class ScopeAndSafety(unittest.TestCase):

    def test_only_tables_in_play_are_described(self):
        model = _model(
            _table("ORD_LIN_FCT", "one row per transaction line"),
            _table("OTHER_FCT", "one row per receipt transaction"),
        )
        out = build_grain_context(model, {"DW.ORD_LIN_FCT"})
        self.assertIn("ORD_LIN_FCT", out)
        self.assertNotIn("OTHER_FCT", out)

    def test_a_bare_table_name_still_matches(self):
        model = _model(_table("ORD_LIN_FCT", "one row per transaction line"))
        self.assertIn("ORD_LIN_FCT", build_grain_context(model, {"ORD_LIN_FCT"}))

    def test_no_model_and_no_tables_are_both_survivable(self):
        self.assertEqual(build_grain_context(None, {"DW.X"}), "")
        self.assertEqual(build_grain_context(_model(), None), "")
        self.assertEqual(build_grain_context(_model(), set()), "")

    def test_the_block_is_bounded(self):
        model = _model(*[
            _table(f"F{i}", "one row per transaction line") for i in range(20)
        ])
        out = build_grain_context(model, {f"DW.F{i}" for i in range(20)}, max_tables=4)
        self.assertEqual(out.count("one row per transaction line"), 4)


if __name__ == "__main__":
    unittest.main()
