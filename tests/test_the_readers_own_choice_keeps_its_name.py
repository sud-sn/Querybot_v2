"""
tests/test_the_readers_own_choice_keeps_its_name.py

The reader picked a date, and the answer card then said nothing about it.

core/query_pipeline.py records which FACT a question reached. Where that
inference succeeded, it also overwrote how the DATE was chosen:

    if _date_fact_inference.get("status") == "selected"
       and _date_context_resolution.get("status") == "selected"
       and resolution_source != "inferred_encoded_fact_date":
        binding["resolution_source"] = "connected_dimension_default"

One exemption, everything else relabelled. So a reader who pressed a chip on the
"which date should I use?" card -- provenance user_confirmed_date_role, "the date
you chose" -- had it rewritten to connected_dimension_default. Same for the date
they chose earlier in the conversation, and for a date they named in the question
outright.

And connected_dimension_default was written in exactly one place and read as a
known provenance in none: it was never added to the i18n catalogue, so
provenance_phrase returned "" and core.insight dropped the business_date clause
from the answer card altogether. The reader chose a date, and the card went
quiet about which date it used.

The relabel is a real refinement in exactly one case -- the resolver used the
FACT's default date, and the inference says which fact that was. That case keeps
it, and now has a phrase. Everything stronger keeps its own.
"""

from __future__ import annotations

import pytest

from core.date_roles import provenance_phrase

# The sources that describe the READER's intent or the metric's own
# configuration. Each is more specific than "the default date of the fact this
# question reaches", so none may be overwritten by it.
STRONGER = [
    "user_confirmed_date_role",
    "thread_date_preference",
    "explicit_date_role",
    "explicit_generated_date_role",
    "metric_default",
    "single_metric_context",
    "inferred_encoded_fact_date",
]


def _relabel(resolution_source, *, fact_inferred=True):
    """The pipeline's rule, applied to one binding.

    _handle_query_impl needs a warehouse, a websocket and a model, so the rule
    is exercised here through the same condition the pipeline applies -- read
    off its tree below, so the two cannot drift.
    """
    resolution = {
        "status": "selected",
        "binding": {"resolution_source": resolution_source},
    }
    inference = {"status": "selected" if fact_inferred else "not_needed"}
    if (
        inference.get("status") == "selected"
        and resolution.get("status") == "selected"
        and str((resolution.get("binding") or {}).get("resolution_source") or "")
        == "fact_default_date_role"
    ):
        resolution["binding"]["resolution_source"] = "connected_dimension_default"
    return resolution["binding"]["resolution_source"]


class TestThePipelineAppliesTheRuleThisFileDescribes:
    """Read as a syntax tree, because the alternative is a test that agrees
    with itself. If the pipeline's condition changes, this says so."""

    def test_the_relabel_is_gated_on_the_fact_default_alone(self):
        import ast
        import inspect

        import core.query_pipeline as qp

        tree = ast.parse(inspect.getsource(qp._handle_query_impl).lstrip())
        guards = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            assigns = [
                statement for statement in node.body
                if isinstance(statement, ast.Assign)
                and any(isinstance(inner, ast.Constant)
                        and inner.value == "connected_dimension_default"
                        for inner in ast.walk(statement.value))
            ]
            if assigns:
                guards.append(node.test)
        assert len(guards) == 1, "the relabel moved, split, or was duplicated"
        literals = {
            inner.value for inner in ast.walk(guards[0])
            if isinstance(inner, ast.Constant) and isinstance(inner.value, str)
        }
        assert "fact_default_date_role" in literals, (
            "the relabel is no longer restricted to the fact default, so it "
            f"overwrites the reader's own choice again. Guard mentions: {literals}")
        assert "inferred_encoded_fact_date" not in literals, (
            "the old shape was an exemption list -- one source spared and "
            "everything else relabelled. It should name what it DOES apply to.")


class TestTheReadersOwnChoiceSurvives:

    @pytest.mark.parametrize("source", STRONGER)
    def test_a_stronger_provenance_is_not_overwritten(self, source):
        assert _relabel(source) == source

    def test_the_fact_default_is_refined_because_that_is_true(self):
        """"The default date of the uniquely connected fact" says something the
        resolver could not: which fact."""
        assert _relabel("fact_default_date_role") == "connected_dimension_default"

    def test_without_a_fact_inference_nothing_is_relabelled(self):
        assert _relabel("fact_default_date_role",
                        fact_inferred=False) == "fact_default_date_role"


class TestWhateverSurvivesCanBeReadByAPerson:

    @pytest.mark.parametrize("source", STRONGER + [
        "fact_default_date_role", "connected_dimension_default",
    ])
    def test_it_has_an_english_phrase(self, source):
        phrase = provenance_phrase(source)
        assert phrase, (
            f"{source} reaches the answer card with no reader-facing phrase, "
            "so the business_date clause is dropped and the reader is told "
            "nothing about which date was used")

    @pytest.mark.parametrize("source", STRONGER + [
        "fact_default_date_role", "connected_dimension_default",
    ])
    def test_it_has_a_french_phrase(self, source):
        assert provenance_phrase(source, lang="fr"), source

    def test_the_readers_choice_says_so_in_so_many_words(self):
        """The point of keeping it: the card credits the reader."""
        assert "you chose" in provenance_phrase("user_confirmed_date_role")
        assert "you chose" in provenance_phrase("thread_date_preference")

    def test_the_relabel_no_longer_erases_the_clause(self):
        """The concrete failure. Before, this returned "" and the clause went
        missing entirely."""
        assert provenance_phrase(_relabel("fact_default_date_role"))
        assert provenance_phrase(_relabel("user_confirmed_date_role"))
