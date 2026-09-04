"""
tests/test_answer_grounding.py

The model was asked to explain figures it had never been shown.

`_build_safe_llm_payload` projects the data brief into what the LLM may see,
and it had branches for `ranking`, `time_series` and `numeric_table` only. The
brief also produces `single_value`, `table` and `text_table` — and
`single_value` is the commonest shape there is: ask "what is our total
revenue", press explain, and the contract went out with `headline_number: None`
and not one figure anywhere in it. The answer was written from the question
wording.

Separately, everything about HOW the figure was governed — which business date
the query resolved to, what scope was applied, whether the read was complete —
is computed by this product and was then withheld from the model. An answer
could state a total and never say what the total was OF, which is the whole
difference between a number and a defensible number.

Both are projection defects, not model defects. Widening the projection is the
fix; the "statistical summaries only" rule was never the ceiling.
"""

import unittest

from core.insight import (
    build_action_contract,
    build_insight_prompt_from_contract,
    compute_data_brief,
)

SINGLE = [{"TOTAL_REVENUE": 13055856.25}]
RANKED = [
    {"SCHEDULE": "NONE", "REVENUE": 11671077.60},
    {"SCHEDULE": "CIII", "REVENUE": 1186233.27},
    {"SCHEDULE": "CV", "REVENUE": 198545.38},
]
TEXTY = [{"NAME": "Acme"}, {"NAME": "Globex"}]


def _contract(rows, question="what is our total revenue", action="explain", **kw):
    return build_action_contract(action, question, compute_data_brief(rows, question), **kw)


class TheModelIsShownTheNumber(unittest.TestCase):

    def test_a_single_cell_result_carries_its_value(self):
        """The shape where the omission was total."""
        contract = _contract(SINGLE)
        self.assertEqual(contract["headline_number"], 13055856.25)

    def test_the_column_it_came_from_is_named(self):
        self.assertEqual(_contract(SINGLE)["top_item"], "TOTAL_REVENUE")

    def test_the_number_reaches_the_prompt_text(self):
        """The contract is not the prompt — the value has to survive rendering."""
        _system, user = build_insight_prompt_from_contract(_contract(SINGLE))
        self.assertIn("13055856", user.replace(",", ""))

    def test_ranking_still_carries_its_leader(self):
        """The branches that already worked must keep working."""
        contract = _contract(RANKED, "revenue by DEA schedule")
        self.assertEqual(contract["headline_number"], 11671077.60)

    def test_a_text_only_result_invents_nothing(self):
        """No numbers exist, so none should appear."""
        contract = _contract(TEXTY, "list our customers")
        self.assertIsNone(contract["headline_number"])


class TheModelIsToldHowMuchItIsLookingAt(unittest.TestCase):

    def test_row_and_column_counts_reach_every_action(self):
        shape = _contract(RANKED, "revenue by DEA schedule")["result_shape"]
        self.assertEqual(shape["row_count"], 3)
        self.assertEqual(shape["column_count"], 2)

    def test_it_is_present_on_a_single_value_too(self):
        self.assertEqual(_contract(SINGLE)["result_shape"]["row_count"], 1)


class ProvenanceReachesTheAnswer(unittest.TestCase):

    GROUNDING = {
        "business_date": "2026-08-31 (newest date in CUS_ORD_IVC_FCT)",
        "period_label": "YTD 2026",
        "scope": "Booked/Order status, business days",
        "completeness": "full read, not truncated",
        "tables": ["EMDW_DMART.CUS_ORD_IVC_FCT"],
    }

    def _prompt(self, **kw):
        return build_insight_prompt_from_contract(_contract(SINGLE, **kw))[1]

    def test_the_business_date_is_stated(self):
        user = self._prompt(grounding=self.GROUNDING)
        self.assertIn("2026-08-31", user)
        self.assertIn("newest date in CUS_ORD_IVC_FCT", user)

    def test_scope_and_completeness_are_stated(self):
        user = self._prompt(grounding=self.GROUNDING)
        self.assertIn("Booked/Order status", user)
        self.assertIn("not truncated", user)

    def test_a_list_renders_readably_rather_than_as_a_repr(self):
        self.assertIn("EMDW_DMART.CUS_ORD_IVC_FCT", self._prompt(grounding=self.GROUNDING))
        self.assertNotIn("['EMDW", self._prompt(grounding=self.GROUNDING))

    def test_the_model_is_told_not_to_invent_the_missing_parts(self):
        self.assertIn("never invent one that is absent",
                      self._prompt(grounding=self.GROUNDING))

    def test_nothing_known_means_nothing_claimed(self):
        """A half-filled provenance block invites the model to fill the gaps."""
        self.assertNotIn("HOW THIS RESULT WAS PRODUCED", self._prompt())

    def test_a_partial_grounding_emits_only_what_it_has(self):
        user = self._prompt(grounding={"period_label": "YTD 2026"})
        self.assertIn("YTD 2026", user)
        self.assertNotIn("Business date", user)
        self.assertNotIn("Completeness", user)


class TheWritingContract(unittest.TestCase):
    """A format the parser accepts but the prompt never asks for is dead."""

    def setUp(self):
        self.system = build_insight_prompt_from_contract(_contract(SINGLE))[0]

    def test_it_asks_for_the_scope_up_front(self):
        self.assertIn("naming what the figures cover", self.system)

    def test_it_asks_for_real_figures(self):
        self.assertIn("Use the actual figures", self.system)

    def test_it_still_offers_sections_for_multi_part_questions(self):
        self.assertIn("SECTION:", self.system)


class TheGroundingIsActuallyBuilt(unittest.TestCase):
    """The seam spans three files; a prompt slot nothing fills is dead."""

    def test_date_disclosures_become_a_stated_date_context(self):
        from core.insight import build_answer_grounding

        g = build_answer_grounding(semantic_plan={"date_disclosures": [
            {"label": "year to date, to the newest date in CUS_ORD_IVC_FCT",
             "table": "EMDW_DMART.CUS_ORD_IVC_FCT",
             "resolution_source": "newest date present"},
        ]})
        self.assertIn("year to date", g["date_context"][0])
        self.assertIn("newest date present", g["business_date"])

    def test_truncation_is_stated_and_completeness_otherwise_stays_quiet(self):
        """Saying "full read" every time trains the reader to skip the line."""
        from core.insight import build_answer_grounding

        self.assertIn("TRUNCATED",
                      build_answer_grounding(row_count=200, truncated=True)["completeness"])
        self.assertNotIn("completeness",
                         build_answer_grounding(row_count=12, truncated=False))

    def test_a_malformed_plan_yields_fewer_facts_not_an_exception(self):
        """This decorates an answer; it must never be why a question fails."""
        from core.insight import build_answer_grounding

        for junk in ({"date_disclosures": "not-a-list"},
                     {"date_disclosures": [None, 7, "x"]},
                     {}):
            with self.subTest(plan=junk):
                self.assertIsInstance(build_answer_grounding(semantic_plan=junk), dict)

    def test_the_pipeline_builds_it_where_the_facts_are(self):
        """Wiring only — these lines sit inside a 6,000-line function."""
        import inspect
        import core.query_pipeline as qp

        source = inspect.getsource(qp._handle_query_impl)
        self.assertIn("build_answer_grounding(", source)
        self.assertIn("grounding=_answer_grounding", source)

    def test_every_hop_passes_it_on(self):
        import inspect

        import core.query_pipeline as qp
        import core.response_builder as rb

        self.assertIn("grounding", inspect.signature(qp._send_why_insight).parameters)
        self.assertIn("grounding", inspect.signature(rb.generate_analysis_response).parameters)

    def test_it_is_explicit_rather_than_riding_extra_kwargs(self):
        """generate_drilldown_insight forwards **extra_kwargs into llm_complete,
        so an unexpected key there is a TypeError, not a harmless extra."""
        import inspect

        import core.response_builder as rb

        param = inspect.signature(rb.generate_analysis_response).parameters["grounding"]
        self.assertIs(param.default, None)
        self.assertNotEqual(param.kind, inspect.Parameter.VAR_KEYWORD)


class TheModuleDocumentsWhatItActuallySends(unittest.TestCase):
    """The old claim was false and invited people to widen the brief on it."""

    def test_the_never_raw_data_claim_is_gone(self):
        import core.insight as insight

        doc = insight.__doc__ or ""
        self.assertIn("never receives the ROW SET", doc)
        self.assertIn("single-value result passes its number", doc)


if __name__ == "__main__":
    unittest.main()
