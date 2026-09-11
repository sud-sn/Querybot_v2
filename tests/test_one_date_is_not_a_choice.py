"""
tests/test_one_date_is_not_a_choice.py

The product trusted a guess more than it trusted the warehouse.

When a fact has exactly ONE business date and nobody has approved it, the
resolver has to decide whether to use it or ask. It decided that on the
physical ENCODING, and the ordering came out upside down. Executed against the
resolver at HEAD~1:

    one native DATE column, confidence 98      -> ambiguous, with ONE option
    one TIMESTAMP column, confidence 98        -> ambiguous, with ONE option
    one surrogate FK, declared join, conf 99   -> ambiguous, with ONE option
    one integer YYYYMMDD guessed from the NAME -> selected, silently

So INVOICE_DT holding 20260315 -- whose date-ness was read off the column's
NAME -- was used without asking, while a column the warehouse itself declares
as DATE interrupted the reader. And the interruption is not even a choice: the
card reads "I found these relevant business dates, but none is an unambiguous
approved default. Which date should I use?" above a single chip. The reader has
strictly less information than the product does, and their answer is remembered
for the thread only, so the next conversation asks again.

The ordering is now by how the date-ness is KNOWN, not by how it is stored:

  * native_date / timestamp -- the warehouse declares the type, which is not
    an inference, so no confidence threshold applies;
  * surrogate_fk with the whole join declared -- the date VALUE comes from a
    real date column in the dimension, the key is only how it is reached;
  * an encoded integer -- weakest, because the claim came from a name. It
    keeps exactly the bar it already had.

Weaker than that still asks, because then there is a real question: not "which
of these dates" but "is this a date at all".

None of this is an approval and every case is disclosed -- the first two
through the discovered_date_role provenance, the third through
inferred_fallback.
"""

from __future__ import annotations

import pytest

from core.contextual_dates import (
    _single_candidate_verdict, resolve_contextual_date_binding,
)
from core.date_roles import provenance_phrase

FACT = "DBO.F_SALES"
METRIC = {"id": 1, "name": "Net Revenue", "base_table": FACT}
QUESTION = "net revenue by month"


def _role(**overrides):
    role = {
        "fact_table": FACT, "fact_column": "INVOICE_DT",
        "role_key": "invoice_date", "label": "Invoice Date",
        "name": "Invoice Date", "business_role": "invoice_date",
        "status": "generated", "confidence": 98,
        "date_key_type": "native_date",
        "inference_source": "column_type",
        "_matched_phrase": "invoice date",
    }
    role.update(overrides)
    return role


SURROGATE = dict(
    date_key_type="surrogate_fk", fact_column="IVC_DT_KEY",
    dimension_table="DBO.D_DATE", dimension_key="DT_KEY",
    date_value_column="FULL_DT", confidence=99,
)


def _resolve(role):
    return resolve_contextual_date_binding(
        QUESTION, matched_metrics=[dict(METRIC)], bindings=[],
        date_roles=[role], required_fact_tables={FACT})


class TestTheWarehousesOwnTypeIsTheStrongestEvidence:

    @pytest.mark.parametrize("key_type", ["native_date", "timestamp"])
    def test_a_real_date_column_is_used_rather_than_asked_about(self, key_type):
        result = _resolve(_role(date_key_type=key_type))
        assert result["status"] == "selected", result
        assert result["binding"]["resolution_source"] == "discovered_date_role"

    @pytest.mark.parametrize("key_type", ["native_date", "timestamp"])
    def test_it_needs_no_confidence_score(self, key_type):
        """A declared column type is not an inference, so there is nothing for
        a confidence number to be about. A role carrying none must not be
        treated as weaker than a guess carrying 95."""
        result = _resolve(_role(date_key_type=key_type, confidence=0,
                                inference_source=""))
        assert result["status"] == "selected", result

    def test_a_declared_join_to_a_date_dimension_counts_too(self):
        result = _resolve(_role(**SURROGATE))
        assert result["status"] == "selected", result
        assert result["binding"]["resolution_source"] == "discovered_date_role"
        assert result["binding"]["fact_column"] == "IVC_DT_KEY"


class TestTheGuessKeepsExactlyTheBarItHad:

    def test_a_deterministic_encoding_is_still_used(self):
        result = _resolve(_role(date_key_type="yyyymmdd_integer",
                                confidence=95))
        assert result["status"] == "selected"
        assert result["binding"]["resolution_source"] == \
            "inferred_encoded_fact_date"
        assert result["binding"]["inferred_fallback"] is True

    @pytest.mark.parametrize("weaker", [
        {"confidence": 94},
        {"inference_source": ""},
    ])
    def test_a_weaker_guess_still_asks(self, weaker):
        """Not because there is a choice to make, but because there is a real
        question: is this integer a date at all."""
        role = _role(date_key_type="yyyymmdd_integer", confidence=95)
        role.update(weaker)
        assert _resolve(role)["status"] == "ambiguous"

    def test_a_rejected_role_is_not_a_candidate_at_all(self):
        """Different outcome, and the right one: a role an administrator threw
        out must not come back as something to offer the reader. It is
        filtered out before the single-candidate question is even asked."""
        role = _role(date_key_type="yyyymmdd_integer", confidence=95,
                     status="rejected")
        assert _resolve(role)["status"] == "none"

    def test_the_bar_did_not_move_for_yyyymm_either(self):
        result = _resolve(_role(date_key_type="yyyymm_integer", confidence=96))
        assert result["status"] == "selected"
        assert result["binding"]["inferred_fallback"] is True


class TestTheInversionIsGone:
    """The comparison itself, which is the actual finding: a guessed encoding
    must never be trusted MORE than a declared one."""

    def test_nothing_the_warehouse_declares_is_weaker_than_a_name_guess(self):
        guessed = _resolve(_role(date_key_type="yyyymmdd_integer",
                                 confidence=95))
        assert guessed["status"] == "selected", "the guess stopped working"
        for declared in ("native_date", "timestamp"):
            result = _resolve(_role(date_key_type=declared))
            assert result["status"] == "selected", (
                f"{declared} interrupts the reader while a date inferred from "
                "a column NAME is used silently")
        assert _resolve(_role(**SURROGATE))["status"] == "selected"


class TestEveryAutomaticChoiceTellsTheReader:
    """None of these is an approval. A reader must be able to see that the
    date was picked for them."""

    @pytest.mark.parametrize("overrides", [
        {}, {"date_key_type": "timestamp"}, SURROGATE,
        {"date_key_type": "yyyymmdd_integer", "confidence": 95},
    ])
    def test_the_provenance_says_it_is_not_an_approved_default(self, overrides):
        binding = _resolve(_role(**overrides))["binding"]
        phrase = provenance_phrase(binding["resolution_source"])
        assert phrase, binding["resolution_source"]
        assert "not an approved default" in phrase, phrase

    @pytest.mark.parametrize("overrides", [
        {}, {"date_key_type": "timestamp"}, SURROGATE,
        {"date_key_type": "yyyymmdd_integer", "confidence": 95},
    ])
    def test_the_phrase_exists_in_french_too(self, overrides):
        binding = _resolve(_role(**overrides))["binding"]
        assert provenance_phrase(binding["resolution_source"], lang="fr")

    def test_only_the_name_guess_is_flagged_as_inferred(self):
        """inferred_fallback drives its own note about a date read off a
        column name. A declared DATE column was not inferred from anything,
        so claiming it was would be a false caveat."""
        declared = _resolve(_role())["binding"]
        assert not declared.get("inferred_fallback")
        guessed = _resolve(_role(date_key_type="yyyymmdd_integer",
                                 confidence=95))["binding"]
        assert guessed["inferred_fallback"] is True


class TestSeveralDatesStillAskBecauseThatIsARealChoice:
    """The change is about ONE candidate. Where there is something to choose
    between, the reader must still choose."""

    def test_two_native_date_columns_are_a_question(self):
        result = resolve_contextual_date_binding(
            QUESTION, matched_metrics=[dict(METRIC)], bindings=[],
            date_roles=[
                _role(),
                _role(fact_column="SHIP_DT", role_key="ship_date",
                      name="Ship Date", label="Ship Date",
                      business_role="ship_date", _matched_phrase="ship date"),
            ],
            required_fact_tables={FACT})
        assert result["status"] == "ambiguous"
        assert len(result["options"]) == 2

    def test_an_approved_role_is_still_the_stronger_source(self):
        """Approval outranks discovery, and says so in the provenance."""
        result = _resolve(_role(status="approved"))
        assert result["status"] == "selected"
        assert result["binding"]["resolution_source"] == \
            "single_approved_date_role"


class TestTheVerdictFunctionOnItsOwn:

    @pytest.mark.parametrize("key_type", ["native_date", "timestamp"])
    def test_declared_types_resolve_to_the_discovered_source(self, key_type):
        source, reason = _single_candidate_verdict(_role(date_key_type=key_type))
        assert source == "discovered_date_role"
        assert reason, "a selected date with no reason records nothing"

    def test_an_incomplete_surrogate_join_is_not_evidence(self):
        """The key alone proves nothing: without the dimension and its date
        column there is no date to read, only an integer."""
        source, _ = _single_candidate_verdict(
            _role(date_key_type="surrogate_fk", fact_column="IVC_DT_KEY"))
        assert source == ""

    @pytest.mark.parametrize("missing", [
        "dimension_table", "dimension_key", "date_value_column",
    ])
    def test_every_part_of_the_join_is_required(self, missing):
        role = _role(**SURROGATE)
        role[missing] = ""
        assert _single_candidate_verdict(role)[0] == ""

    def test_junk_does_not_raise(self):
        assert _single_candidate_verdict({}) == ("", "")
        assert _single_candidate_verdict(None) == ("", "")
