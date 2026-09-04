"""
tests/test_multi_period_answer_surface.py

Fixing the SQL alone does not fix the answer, and that was proved twice.

compute_data_brief hardcodes value_column = numeric_cols[0], and build_answer
does the same. On the widened result those are the OLDEST period's column, so
the card opened with

    headline    Pumps leads at 3,400,000.
    short_value 3,400,000
    comparison  1,000,000 above the next result

-- a 2024 leaderboard with a cross-category gap chip that reads exactly like a
year-over-year delta. portal_chat.html renders short_value and headline as the
LEAD of the card; insight_summary is a footnote below it. Fixing only the
footnote leaves the wrong number in the largest type on the page.

The target question is not causal (is_causal_question is False), so no LLM
narration runs at all. These deterministic sentences ARE the answer the reader
gets.

Every test calls build_assistant_response and asserts on the dict it returns.
"""

import unittest

from core.response_builder import build_assistant_response

QUESTION = ("Compare 2025 against 2024 by revenue category which grew, "
            "which shrank, and what did each contribute to the overall change?")

# One row per category, one column per period -- the shape the hint asks for.
# Obviously synthetic values; no tenant data anywhere in this file.
WIDE_ROWS = [
    {"REVENUE_CATEGORY": "Pumps", "NET_AMOUNT_2024": 3400000, "NET_AMOUNT_2025": 3800000,
     "CHANGE_ABS": 400000, "CHANGE_PCT": 11.76, "SHARE_OF_CHANGE_PCT": 46.0},
    {"REVENUE_CATEGORY": "Valves", "NET_AMOUNT_2024": 2400000, "NET_AMOUNT_2025": 2220000,
     "CHANGE_ABS": -180000, "CHANGE_PCT": -7.5, "SHARE_OF_CHANGE_PCT": -20.7},
    {"REVENUE_CATEGORY": "Seals", "NET_AMOUNT_2024": 1000000, "NET_AMOUNT_2025": 1650000,
     "CHANGE_ABS": 650000, "CHANGE_PCT": 65.0, "SHARE_OF_CHANGE_PCT": 74.7},
]

PERIOD_CONTEXT = {"period_comparison": {"labels": ["2024", "2025"],
                                        "aliases": ["2024", "2025"]}}


def _respond(rows, question=QUESTION, display_context=None):
    return build_assistant_response(
        question=question,
        rows=list(rows),
        sql="SELECT 1",
        duration_ms=12,
        display_context=display_context,
    )


class ThePeriodComparisonLeadsWithTheChange(unittest.TestCase):

    def test_period_comparison_answer_and_summary(self):
        payload = _respond(WIDE_ROWS, display_context=PERIOD_CONTEXT)
        answer = payload["answer"]

        # The defect, pinned: the lead line must not be a 2024 leaderboard.
        self.assertNotIn("leads at", answer["headline"])
        self.assertNotIn("above the next result", answer["comparison"])

        for label in ("2024", "2025"):
            self.assertIn(label, answer["headline"])
        self.assertTrue(
            any(word in answer["headline"] for word in ("rose", "fell", "was flat")),
            answer["headline"],
        )
        self.assertIn("2024", answer["comparison"])
        self.assertIn("%", answer["comparison"])

        # short_value is the newest period's total, not the biggest 2024 cell.
        self.assertEqual(answer["short_value"], "7,670,000")

        summary = payload["insight_summary"]
        self.assertIn("2 grew", summary)
        self.assertIn("1 shrank", summary)
        self.assertIn("net change", summary)
        self.assertIn("Seals", summary)      # the top mover, not the top level

    def test_the_pre_change_answer_is_what_this_replaces(self):
        """The same rows without the plan labels still take the old path. Two
        purposes: it records what the reader used to see, and it proves the new
        branch is what changed the answer rather than something else."""
        old = _respond(WIDE_ROWS)["answer"]
        self.assertIn("leads at", old["headline"])
        self.assertEqual(old["short_value"], "3,400,000")

    def test_the_decision_signal_makes_no_concentration_claim(self):
        """Every line _build_decision_signal can produce for a ranking is about
        a LEVEL. "Top entries drive 100% of the total" computed over the oldest
        period's column reads as a claim about today."""
        self.assertEqual(
            _respond(WIDE_ROWS, display_context=PERIOD_CONTEXT)["decision_signal"], {})
        self.assertTrue(_respond(WIDE_ROWS)["decision_signal"])

    def test_three_periods_compare_the_outer_two(self):
        rows = [
            {"REGION": "North", "REVENUE_2023": 100, "REVENUE_2024": 130, "REVENUE_2025": 200},
            {"REGION": "South", "REVENUE_2023": 300, "REVENUE_2024": 280, "REVENUE_2025": 250},
        ]
        payload = _respond(
            rows, question="compare revenue in 2023, 2024 and 2025 by region",
            display_context={"period_comparison": {"labels": ["2023", "2024", "2025"]}},
        )
        self.assertIn("2023", payload["answer"]["headline"])
        self.assertIn("2025", payload["answer"]["headline"])
        self.assertIn("North", payload["insight_summary"])


class OrdinaryResultsAreUntouched(unittest.TestCase):
    """The branch must be invisible to every other answer in the product. Each
    case runs the same rows twice -- with the period labels present and absent
    -- and requires the two payloads to be identical, so a branch that fired on
    the wrong shape shows up as a difference."""

    def _identical(self, rows, question):
        with_labels = _respond(rows, question, display_context=PERIOD_CONTEXT)
        without = _respond(rows, question)
        for key in ("headline", "short_value", "comparison"):
            with self.subTest(field=key):
                self.assertEqual(with_labels["answer"][key], without["answer"][key])
        self.assertEqual(with_labels["insight_summary"], without["insight_summary"])
        return without

    def test_a_plain_ranking_result_is_unchanged(self):
        rows = [{"REGION": "North", "REVENUE": 900},
                {"REGION": "South", "REVENUE": 400},
                {"REGION": "East", "REVENUE": 250}]
        answer = self._identical(rows, "revenue by region")["answer"]
        self.assertIn("leads at", answer["headline"])

    def test_erp_part_columns_are_not_read_as_periods(self):
        """P_CODE / P_QTY is the shape an alias regex would have eaten."""
        rows = [{"P_CODE": "A-1", "P_QTY": 12, "QTY": 3},
                {"P_CODE": "A-2", "P_QTY": 40, "QTY": 9}]
        self._identical(rows, "list part quantities")

    def test_a_single_period_result_keeps_its_old_answer(self):
        """The miss case. The hint did not land, the plan labels are not
        published, and the card reads exactly as it did before."""
        rows = [{"REVENUE_CATEGORY": "Pumps", "NET_AMOUNT": 3800000},
                {"REVENUE_CATEGORY": "Valves", "NET_AMOUNT": 2220000}]
        self._identical(rows, QUESTION)

    def test_a_compare_prior_style_wide_row_still_narrates_itself(self):
        """_period_comparison_from_rows owns the CURRENT_x/PREVIOUS_x shape the
        compare_prior chip produces. The new branch must not shadow it."""
        rows = [{"CURRENT_MONTH": "2026-03", "PREVIOUS_MONTH": "2026-02",
                 "CURRENT_REVENUE": 500, "PREVIOUS_REVENUE": 400}]
        summary = _respond(rows, "revenue this month versus last")["insight_summary"]
        self.assertIn("2026-03", summary)
        self.assertIn("up 25.0%", summary)


class SensitiveCategoryLabelsNeverReachTheSentence(unittest.TestCase):

    ROWS = [
        {"EMPLOYEE_NAME": "Ada Lovelace", "NET_AMOUNT_2024": 100000, "NET_AMOUNT_2025": 160000},
        {"EMPLOYEE_NAME": "Alan Turing", "NET_AMOUNT_2024": 200000, "NET_AMOUNT_2025": 150000},
        {"EMPLOYEE_NAME": "Grace Hopper", "NET_AMOUNT_2024": 90000, "NET_AMOUNT_2025": 95000},
    ]

    def test_sensitive_label_column_redacted(self):
        payload = _respond(self.ROWS,
                           question="compare 2025 against 2024 by employee",
                           display_context=PERIOD_CONTEXT)
        text = f"{payload['answer']['headline']} {payload['insight_summary']}"

        for name in ("Ada", "Lovelace", "Alan", "Turing", "Grace", "Hopper"):
            with self.subTest(name=name):
                self.assertNotIn(name, text)
        # The sentence still says something: it falls back to the label-free
        # form rather than printing "redacted segment" as if it were a name.
        self.assertIn("largest increase", payload["insight_summary"])
        self.assertNotIn("redacted segment", text)

    def test_the_same_rows_under_a_neutral_column_do_name_the_mover(self):
        """The control. Without it the assertions above would pass on a summary
        that never names anything."""
        rows = [{k.replace("EMPLOYEE_NAME", "REVENUE_CATEGORY"): v
                 for k, v in row.items()} for row in self.ROWS]
        payload = _respond(rows, display_context=PERIOD_CONTEXT)
        self.assertIn("Ada Lovelace", payload["insight_summary"])


class TheModeDowngradeDoesNotClobberAChangeResult(unittest.TestCase):
    """keep_top / sort / contribution rewrite a time_series result into a
    ranking, because period labels sorted by a measure are not a chronology.
    A named-period comparison keeps its periods in COLUMNS, so the downgrade
    would relabel a change result as a leaderboard."""

    ROWS = [
        {"MONTH": "2025-01", "NET_AMOUNT_2024": 10, "NET_AMOUNT_2025": 40},
        {"MONTH": "2025-02", "NET_AMOUNT_2024": 20, "NET_AMOUNT_2025": 25},
        {"MONTH": "2025-03", "NET_AMOUNT_2024": 30, "NET_AMOUNT_2025": 90},
    ]

    def test_a_period_comparison_survives_the_downgrade(self):
        payload = _respond(
            self.ROWS, QUESTION,
            display_context={"result_operation": "sort", **PERIOD_CONTEXT})
        # The mode reaches the frontend as analysis_contract and steers the
        # brief and the follow-up chips. Relabelled "ranking", a change result
        # is described as a leaderboard.
        self.assertEqual(payload["analysis_contract"]["mode"], "time_series")
        self.assertNotIn("leads at", payload["answer"]["headline"])
        self.assertIn("2025", payload["answer"]["headline"])

    def test_the_downgrade_is_what_this_guard_prevents(self):
        """Same rows, same sort operation, no plan labels: the downgrade fires,
        which is what makes the assertion above discriminate."""
        payload = _respond(self.ROWS, QUESTION,
                           display_context={"result_operation": "sort"})
        self.assertEqual(payload["analysis_contract"]["mode"], "ranking")

    def test_an_ordinary_sorted_series_is_still_downgraded(self):
        rows = [{"MONTH": r["MONTH"], "NET_AMOUNT": r["NET_AMOUNT_2025"]}
                for r in self.ROWS]
        payload = _respond(rows, "top months by revenue",
                           display_context={"result_operation": "sort"})
        self.assertEqual(payload["analysis_contract"]["mode"], "ranking")


if __name__ == "__main__":
    unittest.main()
