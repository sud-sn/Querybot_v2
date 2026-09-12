"""
tests/test_the_governed_compiler_refuses_in_french.py

"Comparer les revenus de ce trimestre avec l'an dernier" compiled to a single
number.

The governed compilers exist to keep the fully-resolved case away from the LLM: a
single approved expression metric plus a single approved rolling Date Role
already determines the query. The scalar branch is "intentionally narrow", and
the thing that keeps it narrow is one refusal gate -- a regex over the question
looking for compare / versus / difference / rank / top / distribution / why /
forecast. Anything it matches leaves the scalar compiler for the analytical
planner.

That regex is English and it ran on the reader's raw words. Measured across ten
questions whose English twins were ALL refused, every single French one compiled:

    comparer les revenus de ce trimestre avec l'an dernier   compiled
    revenus par rapport à l'an dernier                       compiled
    quelle est la différence de revenus                      compiled
    classer les succursales par revenus                      compiled
    les 5 meilleurs revenus                                  compiled
    pourquoi les revenus ont-ils changé                      compiled
    répartition des revenus                                  compiled
    prévision des revenus                                    compiled
    part des revenus                                         compiled
    corrélation des revenus                                  compiled

"comparer" is not "compare" -- the \\b after "compare" fails on the "r" -- so the
gate saw an ordinary scalar request. The reader asked for two numbers and a
difference and was handed one number, over one window, with no indication that
the comparison had been dropped.

Two fixes, and the second matters beyond French. The gates now read the CANONICAL
English carried in the semantic context beside the reader's own words, which
closes eight of the ten. The last two exposed a gap the gate had in English too:
"the 5 best revenue" and "breakdown of revenue" are a ranking and a distribution
in any language and matched none of its words. So the gate also reads the
STRUCTURED top-N signal that detect_top_n_intent already resolved upstream --
which reads "top 5" and "top five" alike -- instead of re-deriving intent from
vocabulary, and its word list gained the superlatives it was missing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.pipeline_helpers import (
    _gate_question,
    compile_governed_temporal_metric_sql,
)
from core.question_normalizer import canonical_question

FACT = "EMDW_DMART.CUS_ORD_IVC_FCT"
DIM = "EMDW_DMART.DT_DMS"
COLUMNS = {
    FACT: {"CUS_IVC_DT_DMS_KEY": "int", "IVC_NET_AMT": "decimal"},
    DIM: {"DT_DMS_KEY": "int", "DMS_DT": "date"},
}
POLICY = {
    "kind": "last_n", "amount": 2, "unit": "day",
    "anchor_policy": "latest_available",
    "fact_table": FACT, "fact_column": "CUS_IVC_DT_DMS_KEY",
    "dimension_table": DIM, "dimension_key": "DT_DMS_KEY",
    "date_column": "DMS_DT", "date_key_type": "surrogate_fk",
    "role_alias": "invoice_date", "anchor_table": FACT,
    "business_role": "Invoice Date", "temporal_grain": "day",
}

# (English question, its French twin). Every English one is refused today; every
# French one must reach the same verdict.
REFUSED_PAIRS = [
    ("compare revenue this quarter with last year",
     "comparer les revenus de ce trimestre avec l'an dernier"),
    ("revenue versus last year", "revenus par rapport à l'an dernier"),
    ("what is the difference in revenue", "quelle est la différence de revenus"),
    ("rank branches by revenue", "classer les succursales par revenus"),
    ("top 5 revenue", "les 5 meilleurs revenus"),
    ("why did revenue change", "pourquoi les revenus ont-ils changé"),
    ("revenue distribution", "répartition des revenus"),
    ("revenue forecast", "prévision des revenus"),
    ("revenue share by branch", "part des revenus"),
    ("revenue correlation", "corrélation des revenus"),
]


def context(question, lang="en", *, top_n=None, canonical=True):
    payload = {
        "question": question,
        "top_n": top_n,
        "semantic_plan": {
            "enabled": True, "fields": [], "joins": [],
            "required_tables": [FACT, DIM], "temporal_policies": [POLICY],
        },
        "analytical_request_plan": {"status": "compiled", "source_facts": [FACT]},
        "metric_formulas": [{
            "name": "Revenue", "formula_type": "expression",
            "sql_template": "SUM(fact_rows.[IVC_NET_AMT])", "base_table": FACT,
        }],
    }
    if canonical:
        payload["canonical_question"] = canonical_question(question, lang)
    return payload


def compiled(question, lang="en", **kw):
    return compile_governed_temporal_metric_sql(
        "azure_sql", set(COLUMNS), set(COLUMNS), COLUMNS,
        context(question, lang, **kw),
    )


class TestTheRefusalGateFiresInFrench:

    @pytest.mark.parametrize("english,french", REFUSED_PAIRS)
    def test_the_english_twin_is_refused(self, english, french):
        """Ground truth. Without this the pair test could pass by compiling
        neither for the wrong reason."""
        assert compiled(english) == "", english

    @pytest.mark.parametrize("english,french", REFUSED_PAIRS)
    def test_and_so_is_the_french_one(self, english, french):
        assert compiled(french, "fr") == "", french

    def test_a_plain_scalar_request_still_compiles_in_both(self):
        """The gate must not have become "refuse everything" -- the whole point
        of this compiler is that the resolved case never reaches the LLM."""
        assert compiled("total revenue for the last 2 days")
        assert compiled("le total des revenus des 2 derniers jours", "fr")
        assert compiled("ventes nettes des 2 derniers jours", "fr")

    def test_the_compiled_sql_is_the_same_for_both_languages(self):
        """Reading the canonical text for the GATES must not change the SQL: the
        physical choices come from the policy and the metric, not the wording."""
        assert compiled("total revenue for the last 2 days") == \
            compiled("le total des revenus des 2 derniers jours", "fr")


class TestTheStructuredIntentIsReadInsteadOfGuessed:
    """detect_top_n_intent already resolved the ranking upstream and reads "top
    5" and "top five" alike. Re-deriving it from vocabulary is what left holes.

    These assert the GATE, not just the outcome, and deliberately so: the
    validator rejects a scalar SQL carrying a top_n context anyway, one layer
    later, so "the compiler returned nothing" cannot tell the two apart. That
    coincidence is a property of this shape, not a guarantee -- a context with a
    top_n and a shape the validator accepted would answer a ranking request with
    one number. So the test watches for the decline happening BEFORE the SQL is
    ever built and handed to the validator.
    """

    @staticmethod
    def _validator_calls(question, lang="en", **kw):
        from unittest.mock import patch

        # Patched at its own module: the compiler does `from core.validator
        # import validate_sql_detailed` inside the function body, so there is no
        # attribute on core.pipeline_helpers to replace.
        with patch("core.validator.validate_sql_detailed") as validator:
            sql = compiled(question, lang, **kw)
        return sql, validator.call_count

    def test_a_top_n_intent_declines_without_building_any_sql(self):
        sql, calls = self._validator_calls(
            "total revenue for the last 2 days",
            top_n={"limit": 5, "direction": "descending"})
        assert sql == ""
        assert calls == 0, "the scalar SQL was built and only then rejected"

    def test_even_when_no_ranking_word_appears_at_all(self):
        """The case the word list can never cover: a follow-up that carries the
        ranking structurally, with nothing in the sentence to read."""
        sql, calls = self._validator_calls(
            "les revenus des 2 derniers jours", "fr",
            top_n={"limit": 5, "direction": "descending"})
        assert sql == ""
        assert calls == 0

    def test_and_no_intent_means_the_sql_is_built_and_validated(self):
        """The other half: without a top_n the compiler does its job."""
        _sql, calls = self._validator_calls("total revenue for the last 2 days")
        assert calls == 1
        assert compiled("total revenue for the last 2 days")

    @pytest.mark.parametrize("question", [
        "the 5 best revenue",
        "the worst revenue",
        "breakdown of revenue",
        "the highest revenue",
        "the lowest revenue",
        "revenue ranking",
    ])
    def test_the_words_the_gate_was_missing_in_english_too(self, question):
        """These were compiled as scalars in ENGLISH before the word list was
        widened -- the French failures surfaced a gap that was never French."""
        assert compiled(question) == "", question


class TestTheGateTextIsChosenNotAssumed:

    def test_the_canonical_text_wins_when_present(self):
        assert _gate_question({
            "question": "comparer les revenus",
            "canonical_question": "compare the revenus",
        }) == "compare the revenus"

    def test_the_readers_text_is_the_fallback(self):
        """A caller that has not been updated must behave exactly as before."""
        assert _gate_question({"question": "compare revenue"}) == "compare revenue"

    @pytest.mark.parametrize("payload", [
        {}, None, {"question": ""}, {"question": "", "canonical_question": ""},
        {"canonical_question": "", "question": "compare revenue"},
    ])
    def test_an_absent_or_empty_canonical_falls_back_without_raising(self, payload):
        result = _gate_question(payload)
        assert isinstance(result, str)
        if payload and payload.get("question"):
            assert result == payload["question"]

    def test_a_caller_with_no_canonical_key_still_refuses_in_english(self):
        """The fallback is not a bypass: an English comparison is still refused
        through the old path."""
        assert compiled("compare revenue this quarter with last year",
                        canonical=False) == ""

    def test_and_that_is_exactly_the_french_defect_it_cannot_fix(self):
        """Stated as the limit of the fallback: without the canonical text the
        French question compiles, which is why the pipeline must supply it."""
        assert compiled("comparer les revenus de ce trimestre avec l'an dernier",
                        "fr", canonical=False) != ""


class TestThePipelineActuallySuppliesIt:
    """The gate can only read the canonical text if the pipeline puts it in the
    context, and that assembly needs a warehouse and a websocket to execute. So
    it is read as a SYNTAX TREE -- the question is whether the key is in the dict
    literal, which is a structural fact, not a substring.
    """

    @staticmethod
    def _context_keys():
        source = (Path(__file__).resolve().parents[1]
                  / "core" / "query_pipeline.py").read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Assign):
                continue
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "_generation_semantic_context" not in targets:
                continue
            if not isinstance(node.value, ast.Dict):
                continue
            return {
                key.value for key in node.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }, node.value
        raise AssertionError(
            "_generation_semantic_context is no longer a dict literal in "
            "core/query_pipeline.py — this check covers nothing")

    def test_the_context_carries_the_canonical_question(self):
        keys, _node = self._context_keys()
        assert "canonical_question" in keys

    def test_and_still_carries_the_readers_own_words(self):
        """Two fields, two jobs. Replacing "question" would send the canonical
        text to the chart title and the audit log."""
        keys, _node = self._context_keys()
        assert "question" in keys

    def test_the_two_are_not_the_same_expression(self):
        _keys, node = self._context_keys()
        values = {
            key.value: value for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant)
        }
        assert ast.dump(values["question"]) != \
            ast.dump(values["canonical_question"])
