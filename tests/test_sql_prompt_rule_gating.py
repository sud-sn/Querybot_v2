"""The SQL system prompt gates narrowly-scoped rules on the question.

Two properties matter and are pinned here:

  1. Safety — with no question (or an unrecognised one) every rule is still
     emitted, so the prompt is exactly what it was before gating existed.
     Dropping a needed rule silently degrades SQL; keeping a spare one costs
     tokens. The tests are written around that asymmetry.

  2. Selectivity — a question that plainly needs one pattern gets that rule
     and not the five unrelated ones.
"""

import unittest

from core.llm import build_sql_system_prompt
from core.sql_prompt_rules import RuleContext, applied_rules, rule_applies

CTX = "## Tables\nD_DATE\nF_SALES\nD_WAREHOUSE\n"

MARKERS = {
    "year_over_year":   "YEAR-OVER-YEAR",
    "month_over_month": "MONTH-OVER-MONTH",
    "moving_average":   "MOVING AVERAGE RULE",
    "anti_join":        "ANTI-JOIN RULE",
    "benchmark":        "ABOVE/BELOW BENCHMARK",
    "avg_interval":     "AVG INTERVAL BETWEEN",
    "correlation":      "CORRELATION / SCATTER RULE",
}


def prompt(question: str = "", **kw) -> str:
    return build_sql_system_prompt("azure_sql", CTX, question=question, **kw)


class GatingIsFailOpenTests(unittest.TestCase):

    def test_no_question_emits_every_gated_rule(self):
        """The pre-gating prompt. Any regression here changes every query."""
        p = prompt()
        for rule_id, marker in MARKERS.items():
            self.assertIn(marker, p, f"{rule_id} missing when no question given")

    def test_blank_and_whitespace_questions_emit_every_rule(self):
        for q in ("", "   ", "\n\t"):
            p = prompt(q)
            for rule_id, marker in MARKERS.items():
                self.assertIn(marker, p, f"{rule_id} dropped for question {q!r}")

    def test_unknown_rule_id_is_included(self):
        ctx = RuleContext(question="total revenue")
        self.assertTrue(rule_applies("no_such_rule", ctx))

    def test_missing_context_includes_everything(self):
        self.assertTrue(rule_applies("moving_average", None))

    def test_predicate_exception_includes_the_rule(self):
        import core.sql_prompt_rules as mod

        def boom(_ctx):
            raise RuntimeError("predicate blew up")

        original = mod.GATED_RULES["benchmark"]
        mod.GATED_RULES["benchmark"] = boom
        try:
            self.assertTrue(rule_applies("benchmark", RuleContext(question="anything")))
        finally:
            mod.GATED_RULES["benchmark"] = original


class GatingIsSelectiveTests(unittest.TestCase):

    def test_plain_aggregation_drops_all_six(self):
        p = prompt("total revenue by warehouse")
        for rule_id, marker in MARKERS.items():
            self.assertNotIn(marker, p, f"{rule_id} kept for a plain aggregation")

    def test_each_question_keeps_only_its_own_rule(self):
        cases = {
            "year_over_year":   "revenue this year vs last year",
            "moving_average":   "3 month rolling average of sales",
            "anti_join":        "customers who have never placed an order",
            "benchmark":        "which regions are above the overall average",
            "avg_interval":     "average time between order and shipment",
            "correlation":      "is there a correlation between price and volume",
        }
        for expected, question in cases.items():
            p = prompt(question)
            self.assertIn(MARKERS[expected], p, f"{expected} missing for {question!r}")
            for other in ("moving_average", "anti_join", "avg_interval"):
                if other == expected:
                    continue
                self.assertNotIn(
                    MARKERS[other], p,
                    f"{other} leaked into {question!r}",
                )

    def test_gating_shrinks_the_prompt_materially(self):
        full = len(prompt())
        lean = len(prompt("total revenue by warehouse"))
        self.assertLess(lean, full)
        # Worth having as a real number: this is the attention budget reclaimed.
        self.assertGreater(full - lean, 5000)

    def test_graph_anti_join_flag_keeps_the_rule_without_phrasing(self):
        """The graph resolver can detect an anti-join the wording doesn't show."""
        p = prompt("employees and their absences", graph_context={"anti_join": True})
        self.assertIn(MARKERS["anti_join"], p)

    def test_plan_comparison_keeps_period_rules_without_phrasing(self):
        plan = {"analytical_request_plan": {"comparison_periods": [{"kind": "prior_year"}]}}
        p = prompt("revenue by region", semantic_plan=plan)
        self.assertIn(MARKERS["year_over_year"], p)


class AppliedRulesReportTests(unittest.TestCase):

    def test_report_covers_every_gated_rule(self):
        report = applied_rules(RuleContext(question="total revenue"))
        self.assertEqual(set(report), set(MARKERS))
        self.assertTrue(all(v is False for v in report.values()))

    def test_report_is_all_true_without_a_question(self):
        self.assertTrue(all(applied_rules(RuleContext()).values()))


if __name__ == "__main__":
    unittest.main()
