"""
tests/test_an_absolute_period_is_a_period.py

"Revenue in March 2026" was not a question about a period.

core/query_pipeline.py gates the whole date-role resolution on
`_temporal_intent or _snapshot_intent`, and core.date_roles
.question_has_temporal_intent looks for words -- day, week, month, quarter,
year, ytd, today. "March 2026" contains none of them. Nor does "Q1 2026", nor
"2026-03-15", nor "for February 2026". Executed at HEAD~1, every absolute-period
phrasing returned temporal_intent=False, so the gate logged SKIPPED and:

  * no date role was resolved, so nothing bound WHICH date March means;
  * no governed date column or date_key_policy reached the compiler or the
    validator;
  * no date_disclosure reached the trust box and no provenance clause reached
    the prose, so the reader could not see which date was used;
  * and on a fact carrying both an invoice date and a delivery date, the model
    picked one. Same question, two different numbers, nothing to tell them
    apart.

The product already knew these were dates. question_has_explicit_date_filter
identifies every one of them correctly -- it is used to stop a "latest
available" window from overriding a stated date -- and the date gate simply
never consulted it.

It is not consulted now either, because it matches any four-digit 19xx/20xx
number, which is right for its job and wrong for this one: "orders over 2000
dollars" would become a date question, and on a fact with several dates and no
default that means interrupting the reader with "which date should I use?"
about a question with no dates in it. question_names_a_calendar_period is the
narrower predicate, and a bare year counts only where a word has made it a
period.

Note what does NOT change: an absolute period carries no relative window, so
detect_temporal_window still returns {} for it and the fail-closed refusal for
"a period was asked for and this fact has no governed business date" does not
start firing on these. What changes is that the date COLUMN becomes governed
and disclosed, and a genuinely ambiguous fact asks -- which "revenue in March
2026" on a fact with two dates genuinely is.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from core.contextual_dates import (
    detect_temporal_window, question_has_explicit_date_filter,
    question_names_a_calendar_period, resolve_contextual_date_binding,
)
from core.date_roles import question_has_temporal_intent

ABSOLUTE = [
    "revenue in March 2026",
    "revenue for February 2026",
    "revenue in Q1 2026",
    "revenue in 2026 q3",
    "revenue on 2026-03-15",
    "revenue 15/03/2026",
    "revenue in 2026",
    "revenue during 2025",
    "sales fy26",
    "revenue for 202603",
    "revenue in 2026 and 2027",
]

NOT_A_PERIOD = [
    "orders over 2000 dollars",
    "top 2000 customers",
    "invoices above 1999",
    "customers with more than 2020 orders",
    "total revenue",
    "revenue by region",
    "how many customers are there",
]


class TestTheWordsTheOldGateMissed:

    @pytest.mark.parametrize("question", ABSOLUTE)
    def test_an_absolute_period_is_recognised(self, question):
        assert question_names_a_calendar_period(question), question

    @pytest.mark.parametrize("question", ABSOLUTE)
    def test_and_the_old_gate_really_did_miss_it(self, question):
        """The premise of this file. If this ever fails the gap closed
        somewhere else and the widening below may be redundant."""
        assert not question_has_temporal_intent(question), question

    @pytest.mark.parametrize("question", NOT_A_PERIOD)
    def test_a_number_that_is_not_a_date_is_not_a_period(self, question):
        assert not question_names_a_calendar_period(question), question

    @pytest.mark.parametrize("question", [
        "orders over 2000 dollars", "top 2000 customers",
        "customers with more than 2020 orders",
    ])
    def test_this_is_narrower_than_the_explicit_date_filter(self, question):
        """The reason for a second predicate rather than reusing the first:
        these all match question_has_explicit_date_filter, and using it to gate
        date resolution would interrupt the reader over a quantity."""
        assert question_has_explicit_date_filter(question), question
        assert not question_names_a_calendar_period(question), question


class TestRelativeWordingIsUnaffected:
    """Two separate mechanisms. An absolute period names its own window; a
    relative one has to be anchored on the data. Neither may capture the
    other."""

    @pytest.mark.parametrize("question", [
        "revenue last month", "revenue ytd", "revenue last 7 days",
        "revenue this quarter", "revenue yesterday",
    ])
    def test_relative_wording_still_carries_a_window(self, question):
        assert detect_temporal_window(question), question

    @pytest.mark.parametrize("question", [
        "revenue by month", "revenue last month", "revenue ytd",
        "revenue this quarter", "revenue yesterday", "revenue last 7 days",
        "revenue by day", "how has revenue changed over the last 12 months",
    ])
    def test_the_absolute_predicate_does_not_claim_relative_wording(
            self, question):
        """The two predicates answer different questions and must stay
        distinct. Overlap would not change today's gate outcome -- these
        already pass question_has_temporal_intent -- which is exactly why it
        would go unnoticed, and then "names its own period" and "must be
        anchored on the data" would stop being separable."""
        assert not question_names_a_calendar_period(question), question

    @pytest.mark.parametrize("question", ABSOLUTE)
    def test_an_absolute_period_carries_no_relative_window(self, question):
        """It must not, or the reader's stated month would be replaced by the
        newest date in the data -- and the fail-closed refusal for a fact with
        no governed date would start firing on questions that name their own
        period."""
        assert detect_temporal_window(question) == {}, question


class TestTheGateActuallyConsultsIt:
    """_handle_query_impl needs a warehouse, a websocket and a model, so its
    gate is read from the tree -- as a SYNTAX TREE, asking whether the
    predicate is one of the conditions, not whether some text appears."""

    @staticmethod
    def _gate_conditions():
        import core.query_pipeline as qp

        tree = ast.parse(inspect.getsource(qp._handle_query_impl).lstrip())
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            test = node.test
            if not (isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or)):
                continue
            names = {
                operand.id for operand in test.values
                if isinstance(operand, ast.Name)
            }
            if "_temporal_intent" in names:
                return names
        return set()

    def test_the_absolute_period_is_one_of_the_gate_conditions(self):
        names = self._gate_conditions()
        assert names, "the date-context gate could not be found"
        assert "_absolute_period" in names, (
            "the date gate does not admit an absolute-period question, so "
            f"'revenue in March 2026' skips date resolution. Gate reads: {names}")

    def test_the_predicate_is_what_fills_it(self):
        """A condition assigned from something else would satisfy the test
        above while checking nothing."""
        import core.query_pipeline as qp

        tree = ast.parse(inspect.getsource(qp._handle_query_impl).lstrip())
        calls = [
            node.value.func.id for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name)
                    and target.id == "_absolute_period"
                    for target in node.targets)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
        ]
        assert calls == ["question_names_a_calendar_period"], calls


class TestWhatTheReaderGetsNow:
    """The resolution itself, executed. Entering date resolution has to produce
    a governed column -- or a question -- rather than merely not crashing."""

    FACT = "DBO.F_SALES"
    METRIC = {"id": 1, "name": "Net Revenue", "base_table": FACT}

    def _role(self, column, key, label, **kw):
        role = {
            "fact_table": self.FACT, "fact_column": column,
            "role_key": key, "label": label, "name": label,
            "business_role": key, "status": "approved", "is_default": 0,
            "date_key_type": "native_date", "_matched_phrase": label.lower(),
        }
        role.update(kw)
        return role

    def _resolve(self, question, roles):
        return resolve_contextual_date_binding(
            question, matched_metrics=[dict(self.METRIC)], bindings=[],
            date_roles=roles, required_fact_tables={self.FACT})

    @pytest.mark.parametrize("question", ABSOLUTE)
    def test_an_approved_default_governs_the_column(self, question):
        result = self._resolve(question, [
            self._role("INVOICE_DT", "invoice_date", "Invoice Date",
                       is_default=1)])
        assert result["status"] == "selected", (question, result)
        assert result["binding"]["fact_column"] == "INVOICE_DT"
        assert result["binding"]["resolution_source"] == "fact_default_date_role"

    def test_two_dates_and_no_default_is_a_real_question(self):
        """"Revenue in March 2026" on a fact with an invoice date and a
        delivery date IS ambiguous. Asking is the correct outcome, and it was
        not happening at all."""
        result = self._resolve("revenue in March 2026", [
            self._role("INVOICE_DT", "invoice_date", "Invoice Date"),
            self._role("DELIVERY_DT", "delivery_date", "Delivery Date"),
        ])
        assert result["status"] == "ambiguous"
        assert len(result["options"]) == 2

    def test_a_date_named_in_the_question_still_wins(self):
        """"Revenue by delivery date in March 2026" names its own date, and
        that outranks the default."""
        result = self._resolve("revenue by delivery date in March 2026", [
            self._role("INVOICE_DT", "invoice_date", "Invoice Date",
                       is_default=1),
            self._role("DELIVERY_DT", "delivery_date", "Delivery Date"),
        ])
        assert result["status"] == "selected"
        assert result["binding"]["fact_column"] == "DELIVERY_DT"
