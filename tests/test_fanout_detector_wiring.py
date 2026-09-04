"""
tests/test_fanout_detector_wiring.py

The fan-out detector was fully built, correctly wired into the validator, and
completely inert.

`_fanout_aggregate_errors` decides whether a join multiplies rows by reading
`edge["cardinality"]` and `edge["many_to_many"]`. Neither is a column of
`entity_relationships` (store/db.py:783-805) and neither was ever written by
anything — a repo-wide grep finds only reads. Three of its four trigger
conditions were therefore permanently False. The fourth needs `fanout_ratio`,
which sits at its -1 sentinel until an admin runs a live relationship profile.

So on any account that had not been profiled — which is all of them — a query
that averaged across a one-to-many join sailed through. That is the exact error
class the detector exists to catch: "one order has many order lines", so
AVG over the joined rows measures lines, not orders.

The cardinality the product genuinely records is `relationship_type`, which
defaults to 'many_to_one' and is set on every row. It just was not carried onto
the resolved edge, and was not read.

Two properties matter and both are asserted here:
  - the ordinary star join (fact -> dimension, forward) must NOT be flagged, or
    turning the detector on would break the most common query in the product;
  - the same edge read BACKWARDS must be flagged, because from the "one" side a
    many-to-one relationship is one-to-many.
"""

import unittest

from core.validator import (
    _edge_multiplies_grain,
    _fanout_explanation,
    validate_sql_detailed,
)


def _edge(**kwargs):
    base = {"from_entity": "ORDER_LINE", "to_entity": "ORDER", "direction": "forward"}
    base.update(kwargs)
    return base


class WhichRelationshipsMultiplyRows(unittest.TestCase):

    def test_the_ordinary_star_join_is_not_a_fanout(self):
        """Fact -> dimension, the most common join in the product."""
        self.assertFalse(_edge_multiplies_grain(_edge(
            relationship_type="many_to_one", direction="forward",
            from_entity="ORDER", to_entity="CUSTOMER",
        )))

    def test_the_same_edge_read_backwards_is_a_fanout(self):
        """From the 'one' side, many-to-one IS one-to-many."""
        self.assertTrue(_edge_multiplies_grain(_edge(
            relationship_type="many_to_one", direction="backward",
        )))

    def test_many_to_many_multiplies_in_either_direction(self):
        for direction in ("forward", "backward"):
            with self.subTest(direction=direction):
                self.assertTrue(_edge_multiplies_grain(
                    _edge(relationship_type="many_to_many", direction=direction)
                ))

    def test_one_to_one_never_multiplies(self):
        for direction in ("forward", "backward"):
            with self.subTest(direction=direction):
                self.assertFalse(_edge_multiplies_grain(
                    _edge(relationship_type="one_to_one", direction=direction)
                ))

    def test_a_measured_ratio_outranks_a_clean_declared_type(self):
        """A live profile saw the rows; the declared type is a claim about them."""
        self.assertTrue(_edge_multiplies_grain(_edge(
            relationship_type="many_to_one", direction="forward", fanout_ratio=3.4,
        )))

    def test_the_unprofiled_sentinel_is_not_evidence_of_a_fanout(self):
        self.assertFalse(_edge_multiplies_grain(_edge(
            relationship_type="many_to_one", direction="forward", fanout_ratio=-1,
        )))

    def test_an_unknown_direction_is_read_as_forward(self):
        self.assertFalse(_edge_multiplies_grain(_edge(
            relationship_type="many_to_one", direction="sideways",
        )))

    def test_the_keys_it_used_to_read_carry_no_signal(self):
        """Guards the fix: this is the edge shape the product actually produced,
        and every trigger the detector had was False on it."""
        self.assertFalse(_edge_multiplies_grain({
            "cardinality": "", "many_to_many": False,
            "direction": "backward", "fanout_ratio": -1,
            "from_entity": "REVENUE_LINE", "to_entity": "RX_FILL",
        }))


class TheRejectionNamesTheRelationship(unittest.TestCase):
    """A rejection the reader cannot act on is barely better than a wrong answer."""

    def test_it_says_which_side_has_more(self):
        self.assertEqual(
            _fanout_explanation(_edge(
                relationship_type="many_to_one", direction="backward",
                from_entity="REVENUE_LINE", to_entity="RX_FILL",
            )),
            "each RX_FILL can match many REVENUE_LINE",
        )

    def test_direction_decides_which_noun_leads(self):
        self.assertEqual(
            _fanout_explanation(_edge(
                relationship_type="many_to_many", direction="forward",
                from_entity="PRODUCT", to_entity="CATEGORY",
            )),
            "each PRODUCT can match many CATEGORY",
        )

    def test_an_unnamed_edge_explains_nothing_rather_than_guessing(self):
        self.assertEqual(
            _fanout_explanation({"from_entity": "", "to_entity": "ORDER"}), "",
        )


class EndToEndThroughTheValidator(unittest.TestCase):
    """The detector is reached through validate_sql_detailed, not called directly."""

    known_tables = {"ANALYTICS.RX_FILL", "ANALYTICS.REVENUE_LINE"}
    table_columns = {
        "ANALYTICS.RX_FILL": {"ID": "int", "PATIENT_ID": "int"},
        "ANALYTICS.REVENUE_LINE": {
            "RX_ID": "int", "UNIT_PRICE": "decimal", "AMOUNT": "decimal",
        },
    }

    def _graph(self, edges):
        return {
            "enabled": True,
            "anchor": "RxFill",
            "entities": [
                {"entity_name": "RxFill", "schema_name": "ANALYTICS", "table_name": "RX_FILL"},
                {"entity_name": "RevenueLine", "schema_name": "ANALYTICS", "table_name": "REVENUE_LINE"},
            ],
            "resolved_edges": edges,
        }

    def _validate(self, sql, edges):
        return validate_sql_detailed(
            sql, self.known_tables, "azure_sql",
            table_columns=self.table_columns,
            semantic_context={"graph_context": self._graph(edges)},
        )

    _FANNED_SQL = (
        "SELECT AVG(r.ID) FROM ANALYTICS.RX_FILL r "
        "JOIN ANALYTICS.REVENUE_LINE l ON r.ID = l.RX_ID"
    )

    def test_an_average_across_a_one_to_many_join_is_rejected(self):
        result = self._validate(self._FANNED_SQL, [{
            "relationship_type": "many_to_one", "direction": "backward",
            "from_entity": "RevenueLine", "to_entity": "RxFill",
        }])
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "fanout_aggregate")

    def test_the_reason_tells_the_reader_which_relationship_did_it(self):
        result = self._validate(self._FANNED_SQL, [{
            "relationship_type": "many_to_one", "direction": "backward",
            "from_entity": "RevenueLine", "to_entity": "RxFill",
        }])
        self.assertIn("each RxFill can match many RevenueLine", result.reason)

    def test_the_same_sql_passed_before_the_wiring_was_fixed(self):
        """The edge shape the product actually produced, which triggered nothing."""
        result = self._validate(self._FANNED_SQL, [{
            "cardinality": "", "many_to_many": False,
            "direction": "backward", "fanout_ratio": -1,
            "from_entity": "RevenueLine", "to_entity": "RxFill",
        }])
        self.assertTrue(result.ok, result.reason)

    def test_counting_a_distinct_anchor_key_is_the_way_through(self):
        result = self._validate(
            "SELECT COUNT(DISTINCT r.ID) FROM ANALYTICS.RX_FILL r "
            "JOIN ANALYTICS.REVENUE_LINE l ON r.ID = l.RX_ID",
            [{"relationship_type": "many_to_one", "direction": "backward",
              "from_entity": "RevenueLine", "to_entity": "RxFill"}],
        )
        self.assertTrue(result.ok, result.reason)

    def test_a_forward_star_join_is_left_alone(self):
        """The regression that would matter: this is most queries in the product."""
        result = self._validate(self._FANNED_SQL, [{
            "relationship_type": "many_to_one", "direction": "forward",
            "from_entity": "RxFill", "to_entity": "RevenueLine",
        }])
        self.assertTrue(result.ok, result.reason)


if __name__ == "__main__":
    unittest.main()
