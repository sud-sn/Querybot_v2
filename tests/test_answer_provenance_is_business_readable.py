"""
tests/test_answer_provenance_is_business_readable.py

The most reader-facing part of a governed answer was written in warehouse
identifiers.

build_answer_grounding collects how a figure was produced, and
_format_grounding_for_prompt renders it under a heading that tells the model:

    HOW THIS RESULT WAS PRODUCED — state the relevant parts in your answer,
    and never invent one that is absent here

so whatever is in that block can reach the reader verbatim. What was in it:

    - Business date the query resolved to: metric_default on CUS_ORD_IVC_FCT
    - Tables read: DBO.CUS_ORD_IVC_FCT, DBO.DIM_DATE

A machine token, a raw fact table, and a list of physical table names. The one
distinction a reader would actually act on was the one the token hid: an
APPROVED default is a governed choice, and a discovered or inferred role is the
product's own guess. Both are legitimate; being unable to tell them apart is
what makes a number indefensible.

The tables are gone rather than translated. There is no business name for a
table in this product -- display_label names columns -- so there was nothing to
translate them into, and the answer card's trust box already shows the SQL and
the tables it read to anyone who opens it.
"""

from __future__ import annotations

import pytest

from core import i18n
from core.date_roles import provenance_phrase
from core.insight import _format_grounding_for_prompt, build_answer_grounding

FACT = "CUS_ORD_IVC_FCT"
TABLES = {"DBO.CUS_ORD_IVC_FCT", "DBO.DIM_DATE"}


def _grounding(**disclosure):
    plan = {"date_disclosures": [{"table": FACT, "column": "IVC_DT_KEY",
                                 **disclosure}]}
    return build_answer_grounding(
        semantic_plan=plan, row_count=12, truncated=False, tables=set(TABLES))


def _prompt(**disclosure):
    return _format_grounding_for_prompt(_grounding(**disclosure))


class TestNoWarehouseIdentifierReachesTheReader:

    def test_the_fact_table_is_not_in_the_business_date_line(self):
        grounding = _grounding(label="Invoice Date",
                               resolution_source="metric_default")
        assert FACT not in grounding["business_date"], grounding
        assert "CUS_ORD" not in str(grounding), grounding

    def test_no_table_name_is_anywhere_in_the_block(self):
        rendered = _prompt(label="Invoice Date",
                           resolution_source="metric_default")
        for table in TABLES:
            assert table not in rendered, rendered
        assert "Tables read" not in rendered, rendered

    def test_the_machine_token_is_not_shown_either(self):
        for source in ("metric_default", "discovered_date_role",
                       "inferred_encoded_fact_date", "user_confirmed_date_role"):
            rendered = _prompt(label="Invoice Date", resolution_source=source)
            assert source not in rendered, (source, rendered)

    def test_the_business_date_is_still_named(self):
        """Removing the identifiers must not remove the fact."""
        grounding = _grounding(label="Invoice Date",
                               resolution_source="metric_default")
        assert "Invoice Date" in grounding["business_date"]


class TestTheReaderCanTellAGovernedChoiceFromAGuess:

    @pytest.mark.parametrize("source,expected", [
        ("metric_default", "approved default"),
        ("metric_default_time_column", "approved default"),
        ("approved_metric_date_context", "approved default"),
        ("fact_default_date_role", "approved default"),
        ("single_approved_date_role", "only approved date"),
        ("single_metric_context", "only date configured"),
    ])
    def test_a_governed_choice_says_so(self, source, expected):
        assert expected in _grounding(
            label="Invoice Date", resolution_source=source)["business_date"]

    @pytest.mark.parametrize("source", [
        "discovered_date_role", "inferred_encoded_fact_date",
    ])
    def test_a_guess_says_it_is_not_an_approved_default(self, source):
        """The distinction the token hid. A reader shown a number measured on a
        date the product guessed at should be able to see that."""
        line = _grounding(label="Invoice Date",
                          resolution_source=source)["business_date"]
        assert "not an approved default" in line, line

    @pytest.mark.parametrize("source,expected", [
        ("user_confirmed_date_role", "the date you chose"),
        ("thread_date_preference", "chose earlier in this conversation"),
        ("explicit_date_role", "named in your question"),
    ])
    def test_the_readers_own_choice_is_credited_to_them(self, source, expected):
        assert expected in _grounding(
            label="Invoice Date", resolution_source=source)["business_date"]


class TestItDegradesRatherThanGuesses:

    def test_an_unknown_source_omits_the_clause_instead_of_printing_it(self):
        """A token with no phrase must not fall back to the token."""
        line = _grounding(label="Invoice Date",
                          resolution_source="some_new_source_v2")["business_date"]
        assert line == "Invoice Date", line
        assert provenance_phrase("some_new_source_v2") == ""

    def test_no_source_at_all_still_names_the_date(self):
        grounding = _grounding(label="Order Date")
        assert grounding.get("date_context") == ["Order Date"], grounding

    def test_no_label_at_all_still_says_how_it_was_chosen(self):
        line = _grounding(resolution_source="metric_default")["business_date"]
        assert "approved default" in line, line

    def test_nothing_known_emits_nothing(self):
        assert build_answer_grounding(semantic_plan={"date_disclosures": [{}]}) == {}

    def test_a_malformed_plan_is_not_a_failed_answer(self):
        """This decorates an answer; it must never be why a question fails."""
        for junk in ({"date_disclosures": "not-a-list"},
                     {"date_disclosures": [None, 7, "x"]},
                     {}):
            assert isinstance(build_answer_grounding(semantic_plan=junk), dict)


class TestTheSameDateIsNotStatedTwice:

    def test_one_date_produces_one_line(self):
        """business_date and date_context sat side by side saying "Invoice
        Date" twice, which reads like the answer is unsure of itself."""
        grounding = _grounding(label="Invoice Date",
                               resolution_source="metric_default")
        assert "date_context" not in grounding, grounding

    def test_a_second_date_is_still_carried(self):
        plan = {"date_disclosures": [
            {"label": "Invoice Date", "table": "F1",
             "resolution_source": "user_confirmed_date_role"},
            {"label": "Delivery Date", "table": "F2"},
        ]}
        grounding = build_answer_grounding(semantic_plan=plan, row_count=3)
        assert "Invoice Date" in grounding["business_date"]
        assert grounding["date_context"] == ["Delivery Date"], grounding


class TestItIsInTheReadersLanguage:

    @pytest.mark.parametrize("source,expected_fr", [
        ("metric_default", "par défaut approuvée"),
        ("user_confirmed_date_role", "que vous avez choisie"),
        ("inferred_encoded_fact_date", "déduite"),
    ])
    def test_a_french_reader_gets_french_provenance(self, source, expected_fr):
        token = i18n.activate_language("fr")
        try:
            line = _grounding(label="Date de facture",
                              resolution_source=source)["business_date"]
        finally:
            i18n.deactivate_language(token)
        assert expected_fr in line, line

    def test_an_english_reader_gets_english(self):
        line = _grounding(label="Invoice Date",
                          resolution_source="metric_default")["business_date"]
        assert "approved default date" in line


class TestEveryResolutionSourceThePipelineCanProduceHasAPhrase:
    """A source with no phrase silently drops the clause, so a new one added to
    the resolver must not be able to arrive here unnoticed."""

    @staticmethod
    def _sources_in_the_resolver():
        import re
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        found = set()
        for name in ("core/contextual_dates.py", "core/query_pipeline.py"):
            text = (root / name).read_text(encoding="utf-8")
            found.update(re.findall(r'source="([a-z_]+)"', text))
            found.update(re.findall(r'"resolution_source"\]\s*=\s*"([a-z_]+)"', text))
            found.update(re.findall(r'"resolution_source":\s*"([a-z_]+)"', text))
        # Only the ones that name a DATE role resolution.
        return {s for s in found
                if "date" in s or s in {"metric_default", "single_metric_context",
                                        "thread_date_preference"}}

    def test_the_scan_found_the_sources(self):
        sources = self._sources_in_the_resolver()
        assert len(sources) >= 8, sources

    def test_every_one_of_them_has_a_reader_facing_phrase(self):
        missing = sorted(s for s in self._sources_in_the_resolver()
                         if not provenance_phrase(s))
        assert not missing, (
            "these resolution sources reach the provenance block with no "
            f"phrase, so the clause is silently dropped: {missing}")

    def test_every_phrase_exists_in_french_too(self):
        for source in sorted(self._sources_in_the_resolver()):
            assert provenance_phrase(source, lang="fr"), source
