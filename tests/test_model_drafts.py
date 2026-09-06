"""
tests/test_model_drafts.py

The drafters that turn the modelling backlog into a review queue.

``core.model_readiness`` says what to do next. This says what the answer
probably is, so the admin confirms rather than types. Everything about that is
only safe because of one property, and it is the property most of this file
tests: **a draft is never applied**. A column's role decides whether a filter
reads as a date or a category; a synonym decides which measure a question
resolves to. Both move answers silently when they are wrong.

Every test calls the real drafter and asserts on the ``Draft`` objects it
returns. There is no fixture that hands a drafter the answer it is being asked
to produce.
"""

from __future__ import annotations

import pytest

from core.model_drafts import (
    MIN_PHRASE_OCCURRENCES,
    Draft,
    column_vocabulary_drafts,
    date_role_drafts,
    metric_shape_drafts,
    question_shape,
)


def by_target(drafts) -> dict[str, Draft]:
    return {d.target: d for d in drafts}


# ── The diff a reviewer reads ────────────────────────────────────────────────

class TestADraftShowsOnlyWhatWouldChange:

    def test_a_field_that_already_matches_is_not_in_the_diff(self):
        draft = Draft(kind="date_role", entity="S", column="C",
                      payload={"role": "date", "display_name": "Invoice Date"},
                      before={"role": "date", "display_name": ""},
                      reason="", confidence=90)
        assert set(draft.changes) == {"display_name"}
        assert draft.changes["display_name"] == ("", "Invoice Date")

    def test_a_draft_that_changes_nothing_says_so(self):
        draft = Draft(kind="date_role", entity="S", column="C",
                      payload={"role": "date"}, before={"role": "date"},
                      reason="", confidence=90)
        assert draft.changes == {}
        assert not draft.is_change

    def test_it_survives_a_round_trip_through_as_dict(self):
        draft = Draft(kind="date_role", entity="S", column="C",
                      payload={"role": "date"}, before={},
                      reason="because", confidence=90, evidence=("a", "b"))
        data = draft.as_dict()
        assert data["target"] == "S.C"
        assert data["changes"] == {"role": ["", "date"]}
        assert data["evidence"] == ["a", "b"]
        assert data["reason"] == "because"


# ── 1 · Date roles ───────────────────────────────────────────────────────────

COLUMNS = [
    {"entity": "Sales", "column": "INVOICE_DATE"},
    {"entity": "Sales", "column": "ORDER_DATE"},
    {"entity": "Sales", "column": "DISPENSE_DATE_ID"},
    {"entity": "Sales", "column": "CUSTOMER_ID"},
    {"entity": "Sales", "column": "NET_AMOUNT"},
]


class TestDateRolesAreDrafted:

    def test_a_date_column_gets_its_role_its_label_and_its_words(self):
        drafts = by_target(date_role_drafts(COLUMNS, []))
        draft = drafts["Sales.INVOICE_DATE"]
        assert draft.payload["role"] == "date"
        assert draft.payload["display_name"] == "Invoice Date"
        # The synonyms are what let a reader's phrasing reach this column;
        # a role with only its own label is barely worth writing down.
        terms = [t.strip() for t in draft.payload["synonyms"].split(",")]
        assert "invoice date" in terms
        assert len(terms) >= 4, terms

    def test_a_column_that_is_not_a_date_is_not_drafted(self):
        drafts = by_target(date_role_drafts(COLUMNS, []))
        assert "Sales.CUSTOMER_ID" not in drafts
        assert "Sales.NET_AMOUNT" not in drafts

    def test_a_confirmed_row_is_never_proposed_over(self):
        confirmed = [{"entity_name": "SALES", "column_name": "INVOICE_DATE",
                      "role": "dimension", "display_name": "Whatever the admin said",
                      "synonyms": "", "status": "confirmed"}]
        # Deliberately wrong-looking: the admin said dimension, the vocabulary
        # says date. The admin wins. Re-proposing asks somebody to re-decide
        # something they already decided, and a review queue that does that
        # stops being read.
        assert "Sales.INVOICE_DATE" not in by_target(date_role_drafts(COLUMNS, confirmed))

    def test_a_suggested_row_is_fair_game(self):
        suggested = [{"entity_name": "SALES", "column_name": "INVOICE_DATE",
                      "role": "dimension", "display_name": "", "synonyms": "",
                      "status": "suggested"}]
        drafts = by_target(date_role_drafts(COLUMNS, suggested))
        assert drafts["Sales.INVOICE_DATE"].changes["role"] == ("dimension", "date")

    def test_a_row_with_no_status_counts_as_confirmed(self):
        # The column defaults to 'confirmed', so a missing status means a row
        # written before it existed -- and those are admin-authored.
        legacy = [{"entity_name": "SALES", "column_name": "INVOICE_DATE",
                   "role": "dimension", "display_name": "", "synonyms": ""}]
        assert "Sales.INVOICE_DATE" not in by_target(date_role_drafts(COLUMNS, legacy))

    def test_the_lookup_is_case_insensitive_on_both_halves(self):
        # The graph stores entity names as the admin typed them and columns as
        # the warehouse spells them. A case-sensitive lookup finds nothing,
        # proposes over a confirmed row, and the admin re-confirms.
        for entity, column in (("sales", "invoice_date"), ("SALES", "Invoice_Date"),
                               ("Sales", "INVOICE_DATE")):
            confirmed = [{"entity_name": entity, "column_name": column,
                          "status": "confirmed", "role": "date"}]
            assert "Sales.INVOICE_DATE" not in by_target(
                date_role_drafts(COLUMNS, confirmed)), (entity, column)

    def test_a_dictionary_role_outranks_a_derived_one(self):
        drafts = by_target(date_role_drafts(COLUMNS, []))
        # DISPENSE_DATE_ID is not in the vocabulary, so its role is derived
        # from its own words and should say so with a lower confidence.
        assert drafts["Sales.INVOICE_DATE"].confidence > \
               drafts["Sales.DISPENSE_DATE_ID"].confidence

    @pytest.mark.parametrize("column,role_key,expected_term", [
        ("ORDER_DATE_ID", "order_date", "sales order date"),
        ("INVOICE_DATE_KEY", "invoice_date", "billing date"),
    ])
    def test_a_derived_key_that_names_a_known_role_gets_the_known_terms(
            self, column, role_key, expected_term):
        # A column the detector cannot match by name reaches its role by
        # DERIVATION -- the role is rebuilt from the column's own words, so it
        # answers with exactly one synonym: its own label. When that derived
        # key names a role the dictionary already holds, the dictionary's
        # entry is the same role with five more ways to ask for it.
        from core.date_roles import date_role_terms, detect_date_role
        derived = detect_date_role(column)
        assert derived.key == role_key
        assert len(date_role_terms(derived)) == 1, (
            f"{column} is a dictionary hit now, so it no longer exercises "
            f"the upgrade this test is named for -- pick another column")

        drafts = by_target(date_role_drafts([{"entity": "S", "column": column}], []))
        terms = [t.strip() for t in drafts[f"S.{column}"].payload["synonyms"].split(",")]
        assert len(terms) >= 4, terms
        assert expected_term in terms

    def test_the_reason_names_the_column_and_the_role(self):
        draft = by_target(date_role_drafts(COLUMNS, []))["Sales.INVOICE_DATE"]
        assert "INVOICE_DATE" in draft.reason
        assert "Invoice Date" in draft.reason
        assert draft.evidence == ("date_roles:invoice_date",)

    def test_the_highest_confidence_draft_is_first(self):
        drafts = date_role_drafts(COLUMNS, [])
        assert [d.confidence for d in drafts] == sorted(
            (d.confidence for d in drafts), reverse=True)

    def test_the_list_is_capped(self):
        many = [{"entity": "S", "column": f"INVOICE_DATE_{i}"} for i in range(60)]
        assert len(date_role_drafts(many, [], limit=5)) == 5

    def test_rows_without_an_entity_or_a_column_are_skipped(self):
        assert date_role_drafts([{"column": "INVOICE_DATE"}, {"entity": "S"}, {}], []) == []


# ── 2 · Column vocabulary ────────────────────────────────────────────────────

OPAQUE = [{"entity": "Sales", "column": "STAT_CD",
           "values": ["Shipped", "Pending", "Cancelled"]}]
QUESTIONS = [
    "revenue for shipped orders last year",
    "total revenue for shipped orders",
    "how many shipped orders in march",
    "cancelled orders by region",
    "which pending orders are late",
    "revenue by region",
    "revenue by product",
    "margin by region",
]


class TestColumnVocabularyIsHarvestedFromValuesAndQuestions:

    def test_the_word_readers_use_beside_a_value_is_proposed(self):
        drafts = by_target(column_vocabulary_drafts(OPAQUE, QUESTIONS, []))
        assert "Sales.STAT_CD" in drafts
        assert "orders" in drafts["Sales.STAT_CD"].payload["synonyms"]

    def test_the_measure_standing_before_the_value_is_never_proposed(self):
        # "revenue for shipped orders" -- revenue is what the question is
        # about, not a name for a status column. Proposing it would resolve
        # every question mentioning revenue onto this column.
        drafts = by_target(column_vocabulary_drafts(
            OPAQUE, ["revenue for shipped orders"] * 6, []))
        proposed = drafts["Sales.STAT_CD"].payload["synonyms"]
        assert "revenue" not in proposed, proposed

    def test_a_column_whose_own_name_carries_the_word_proposes_nothing(self):
        named = [{"entity": "Sales", "column": "ORDER_STATUS",
                  "values": ["Shipped", "Pending", "Cancelled"]}]
        assert column_vocabulary_drafts(named, QUESTIONS, []) == []

    def test_the_plural_of_the_column_name_is_not_a_new_word(self):
        named = [{"entity": "Sales", "column": "ORDER",
                  "values": ["Shipped", "Pending", "Cancelled"]}]
        assert column_vocabulary_drafts(named, QUESTIONS, []) == []

    def test_a_phrase_is_cut_at_the_first_word_that_carries_no_meaning(self):
        # "shipped orders last year" offers "orders" and stops. Allowing a run
        # that merely is not ENTIRELY stopwords proposes "orders last", which
        # is a piece of sentence rather than a name for a column.
        drafts = by_target(column_vocabulary_drafts(
            OPAQUE, ["revenue for shipped orders last year"] * 6, []))
        proposed = [t.strip() for t
                    in drafts["Sales.STAT_CD"].payload["synonyms"].split(",")]
        assert "orders" in proposed
        assert proposed == ["orders"], proposed

    @pytest.mark.parametrize("question", [
        "shipped shipped shipped orders",       # the value beside itself
        "shipped pending orders",               # the value beside another value
        "shipped orders pending",               # and on the other side
    ])
    def test_the_value_itself_is_never_proposed_as_the_columns_word(self, question):
        # "shipped" names a row, not a column. As a synonym it would send
        # every question about a shipped anything here.
        drafts = column_vocabulary_drafts(OPAQUE, [question] * 6, [])
        for draft in drafts:
            proposed = draft.payload["synonyms"].casefold()
            for value in ("shipped", "pending", "cancelled"):
                assert value not in proposed, (question, proposed)

    def test_one_question_votes_once_however_many_values_it_names(self):
        # A single chatty question mentioning three values must not clear a
        # threshold that exists to mean "three different people".
        chatty = ["shipped orders pending orders cancelled orders"]
        assert column_vocabulary_drafts(OPAQUE, chatty, []) == []

    def test_below_the_threshold_nothing_is_proposed(self):
        few = ["revenue for shipped orders"] * (MIN_PHRASE_OCCURRENCES - 1)
        assert column_vocabulary_drafts(OPAQUE, few, []) == []

    def test_a_question_that_names_no_value_does_not_vote(self):
        assert column_vocabulary_drafts(
            OPAQUE, ["revenue by region", "orders by region"] * 5, []) == []

    def test_a_confirmed_row_is_never_proposed_over(self):
        confirmed = [{"entity_name": "SALES", "column_name": "STAT_CD",
                      "status": "confirmed", "synonyms": ""}]
        assert column_vocabulary_drafts(OPAQUE, QUESTIONS, confirmed) == []

    def test_existing_synonyms_are_kept_and_the_new_ones_appended(self):
        suggested = [{"entity_name": "SALES", "column_name": "STAT_CD",
                      "status": "suggested", "synonyms": "state"}]
        drafts = by_target(column_vocabulary_drafts(OPAQUE, QUESTIONS, suggested))
        merged = drafts["Sales.STAT_CD"].payload["synonyms"]
        assert merged.startswith("state"), merged
        assert "orders" in merged

    def test_a_word_already_in_the_synonyms_is_not_added_twice(self):
        suggested = [{"entity_name": "SALES", "column_name": "STAT_CD",
                      "status": "suggested", "synonyms": "orders"}]
        assert column_vocabulary_drafts(OPAQUE, QUESTIONS, suggested) == []

    def test_the_evidence_names_the_phrase_and_how_often(self):
        draft = by_target(column_vocabulary_drafts(OPAQUE, QUESTIONS, []))["Sales.STAT_CD"]
        assert draft.evidence, draft
        assert "orders" in draft.evidence[0]
        assert "STAT_CD" in draft.reason
        # The reason names the values that matched, which is the half of the
        # evidence that came from the warehouse rather than from a person.
        assert any(v in draft.reason for v in ("Shipped", "Cancelled", "Pending"))

    def test_no_questions_means_no_drafts(self):
        assert column_vocabulary_drafts(OPAQUE, [], []) == []
        assert column_vocabulary_drafts(OPAQUE, None, []) == []

    def test_no_values_means_no_drafts(self):
        assert column_vocabulary_drafts(
            [{"entity": "S", "column": "C", "values": []}], QUESTIONS, []) == []


# ── 3 · Metric shapes ────────────────────────────────────────────────────────

class TestAQuestionShapeSurvivesItsVariableParts:

    @pytest.mark.parametrize("question", [
        "gross margin for March 2024",
        "gross margin for April 2025",
        "gross margin last quarter",
        "gross margin for Q1",
        "what was the gross margin in 2023",
        "show me gross margin ytd",
    ])
    def test_the_same_measure_asked_differently_is_one_shape(self, question):
        assert question_shape(question) == "gross margin"

    def test_two_different_measures_are_two_shapes(self):
        assert question_shape("gross margin by region") != \
               question_shape("net revenue by region")

    def test_a_quoted_literal_is_a_variable_part(self):
        assert question_shape("revenue for 'ACME Corp'") == \
               question_shape("revenue for 'Globex'")

    def test_an_empty_question_has_no_shape(self):
        assert question_shape("") == ""
        assert question_shape("   ") == ""
        assert question_shape(None) == ""

    def test_a_question_of_nothing_but_variable_parts_has_no_shape(self):
        assert question_shape("last month vs 2024") == ""


class TestMetricShapesAreDraftedFromRepetition:

    ROWS = ([{"question": "gross margin for March 2024"}] * 4
            + [{"question": "gross margin for April 2025"}] * 2
            + [{"question": "revenue by region", "metric_id": 7}] * 5
            + [{"question": "a question asked once"}])

    def test_a_shape_asked_often_and_never_answered_is_drafted(self):
        drafts = metric_shape_drafts(self.ROWS)
        shapes = {d.payload["shape"] for d in drafts}
        assert "gross margin" in shapes

    def test_the_variants_are_counted_as_one_shape(self):
        draft = next(d for d in metric_shape_drafts(self.ROWS)
                     if d.payload["shape"] == "gross margin")
        # Four askings in March and two in April are six askings of one
        # missing measure. Counting them as two shapes of four and two puts
        # both under a threshold that neither deserves to be under.
        assert draft.payload["occurrences"] == 6

    def test_a_shape_that_a_metric_already_answers_is_not_drafted(self):
        shapes = {d.payload["shape"] for d in metric_shape_drafts(self.ROWS)}
        assert "revenue region" not in shapes

    def test_a_shape_asked_once_is_a_question_not_a_gap(self):
        shapes = {d.payload["shape"] for d in metric_shape_drafts(self.ROWS)}
        assert not any("asked once" in s for s in shapes)

    def test_a_metric_name_counts_as_answered_too(self):
        rows = [{"question": "gross margin now", "metric_name": "Gross Margin"}] * 5
        assert metric_shape_drafts(rows) == []

    def test_one_answered_asking_clears_the_whole_shape(self):
        # The shape is not a gap if the workspace can answer it at all; the
        # other askings failed for some other reason, which is a different
        # report.
        rows = ([{"question": "gross margin for March"}] * 5
                + [{"question": "gross margin for April", "metric_id": 3}])
        assert metric_shape_drafts(rows) == []

    def test_the_draft_carries_the_actual_questions_as_evidence(self):
        draft = next(d for d in metric_shape_drafts(self.ROWS)
                     if d.payload["shape"] == "gross margin")
        assert "gross margin for March 2024" in draft.evidence
        assert "gross margin for April 2025" in draft.evidence
        assert "6" in draft.reason

    def test_no_sql_is_ever_drafted(self):
        # A definition guessed from question text is a governed measure that
        # nobody wrote, which is worse than a missing one. What is drafted is
        # the case for building it.
        for draft in metric_shape_drafts(self.ROWS):
            assert "sql" not in {k.casefold() for k in draft.payload}
            assert not any("select" in str(v).casefold()
                           for v in draft.payload.values())

    def test_the_threshold_is_honoured(self):
        rows = [{"question": "gross margin now"}] * 2
        assert metric_shape_drafts(rows, min_occurrences=3) == []
        assert metric_shape_drafts(rows, min_occurrences=2)

    def test_the_most_asked_shape_is_first(self):
        rows = ([{"question": "alpha measure"}] * 3
                + [{"question": "beta measure"}] * 9)
        assert metric_shape_drafts(rows)[0].payload["shape"] == "beta measure"

    def test_empty_input_is_an_empty_report(self):
        assert metric_shape_drafts([]) == []
        assert metric_shape_drafts(None) == []


class TestNothingHereWrites:
    """
    The property the whole module rests on. A drafter that reached the store
    would be applying a change nobody approved, and every other guarantee in
    this file would be decoration.
    """

    def test_no_drafter_touches_the_store(self, monkeypatch):
        import store

        def _explode(*args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("a drafter wrote to the store")

        for name in ("save_entity_property", "confirm_entity_property",
                     "save_metric", "get_db"):
            if hasattr(store, name):
                monkeypatch.setattr(store, name, _explode)

        date_role_drafts(COLUMNS, [])
        column_vocabulary_drafts(OPAQUE, QUESTIONS, [])
        metric_shape_drafts(TestMetricShapesAreDraftedFromRepetition.ROWS)

    def test_the_module_imports_no_store_at_module_level(self):
        # A module-level store import is how a "report" acquires a write path
        # later without anybody noticing the change in character.
        source = (__import__("pathlib").Path(__file__).resolve().parents[1]
                  / "core" / "model_drafts.py").read_text(encoding="utf-8")
        head = source[:source.index("@dataclass")]
        assert "import store" not in head, "model_drafts reaches the store at import"
