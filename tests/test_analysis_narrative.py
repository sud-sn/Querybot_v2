# -*- coding: utf-8 -*-
"""The narrative layer — core/analysis_narrative.py.

Every test executes the real phraser or the real checker. The prose
assertions compare the two languages line by line rather than whole
paragraphs, because a whole-paragraph comparison passes when only one
sentence in three was translated.

The load-bearing test in this file is
``TestTheCheckedNumbersRule``: the product's claim is that a model only
phrases what was computed, and that claim is a matter of trust until
something rejects a paragraph carrying a figure the evidence does not hold.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.analysis_evidence import (  # noqa: E402
    CONCENTRATION_LEADER,
    CORRELATION,
    OUTLIERS,
    SPREAD_HIGH,
    TREND_UP,
    AnalysisEvidence,
    Finding,
    build_evidence,
)
from core.analysis_narrative import (  # noqa: E402
    DEFAULT_LIMIT,
    LABEL_BEARING_KINDS,
    MIN_MATERIALITY,
    SMALL_NUMBER_CEILING,
    Narrative,
    TemplatePhraser,
    build_narrative,
    enforce_checked_numbers,
    evidence_for_prompt,
    evidence_id,
    format_number_for,
    humanise_column,
    message_id_for,
    numbers_in,
    select_findings,
    verify_against_evidence,
)
from core.i18n import MESSAGES  # noqa: E402

# A leader-dominated result: one finding that names a customer, one that does
# not. Enough to exercise both governance paths in one fixture.
LEADER_ROWS = [
    {"CUSTOMER": name, "NET_REVENUE_AMT": value}
    for name, value in [
        ("Ospedale San Raffaele", 6200), ("Acme", 900), ("Borel", 700),
        ("Duval", 500), ("Fabre", 400), ("Gide", 300),
        ("Hugo", 200), ("Ivry", 100),
    ]
]

SERIES_ROWS = [
    {"MONTH": f"2026-{i:02d}", "REVENUE": v}
    for i, v in enumerate([100, 108, 121, 130, 142, 155, 170, 188], start=1)
]


# ══════════════════════════════════════════════════════════════════════════════
# Selection
# ══════════════════════════════════════════════════════════════════════════════

class TestSelection(unittest.TestCase):

    def test_only_findings_above_the_floor_are_said_out_loud(self):
        evidence = AnalysisEvidence(findings=(
            Finding(kind=TREND_UP, columns=("A",), materiality=0.9),
            Finding(kind=OUTLIERS, columns=("B",), materiality=0.01),
        ))
        chosen = select_findings(evidence)
        self.assertEqual([f.kind for f in chosen], [TREND_UP])

    def test_the_reader_is_not_given_more_than_the_limit(self):
        evidence = AnalysisEvidence(findings=tuple(
            Finding(kind=TREND_UP, columns=(f"C{i}",), materiality=0.9)
            for i in range(10)
        ))
        self.assertEqual(len(select_findings(evidence)), DEFAULT_LIMIT)
        self.assertEqual(len(select_findings(evidence, limit=2)), 2)

    def test_the_floor_is_a_real_bar_not_zero(self):
        self.assertGreater(MIN_MATERIALITY, 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# Phrasing
# ══════════════════════════════════════════════════════════════════════════════

class TestTemplatePhrasing(unittest.TestCase):

    def test_a_narrative_is_produced_with_the_numbers_in_it(self):
        narrative = build_narrative(build_evidence(SERIES_ROWS), lang="en")
        self.assertTrue(narrative.sentences)
        joined = " ".join(narrative.sentences)
        self.assertIn("Revenue", joined)
        self.assertIn("188", joined)

    def test_the_same_evidence_produces_byte_identical_prose_every_time(self):
        evidence = build_evidence(LEADER_ROWS)
        first = build_narrative(evidence, lang="en").sentences
        second = build_narrative(evidence, lang="en").sentences
        self.assertEqual(first, second)

    def test_no_sentence_leaves_a_placeholder_on_the_page(self):
        # t() deliberately leaves an unsupplied placeholder in place rather
        # than raising, so a catalogue/slot mismatch reaches the reader as
        # literal "{leader}". Every kind, both languages, labels on and off.
        for rows in (LEADER_ROWS, SERIES_ROWS,
                     [{"X": i, "Y": i * 3.0} for i in range(1, 12)],
                     [{"V": v} for v in [1, 500, 3, 900, 2, 700]]):
            for lang in ("en", "fr"):
                for labels in (True, False):
                    evidence = build_evidence(rows, include_labels=labels)
                    for sentence in build_narrative(evidence, lang=lang).sentences:
                        self.assertNotIn("{", sentence, (rows[0], lang, labels))
                        self.assertNotIn("}", sentence, (rows[0], lang, labels))

    def test_every_finding_selected_becomes_exactly_one_sentence(self):
        evidence = build_evidence(LEADER_ROWS)
        chosen = select_findings(evidence)
        narrative = build_narrative(evidence, lang="en")
        self.assertEqual(len(narrative.sentences), len(chosen))
        self.assertEqual(narrative.finding_kinds, tuple(f.kind for f in chosen))

    def test_a_result_with_nothing_notable_still_says_something_useful(self):
        flat = [{"REGION": f"r{i}", "V": 100} for i in range(8)]
        narrative = build_narrative(build_evidence(flat), lang="en")
        self.assertTrue(narrative.sentences)
        self.assertFalse(narrative.is_empty)

    def test_an_empty_result_does_not_raise(self):
        narrative = build_narrative(build_evidence([]), lang="en")
        self.assertIsInstance(narrative, Narrative)
        self.assertTrue(narrative.sentences)

    def test_a_sentence_missing_a_value_is_dropped_rather_than_printed(self):
        # Defence in depth behind the label-free fix. t() deliberately leaves
        # an unsupplied placeholder in place rather than raising, so any
        # future catalogue entry whose slots outrun what the finding supplies
        # would render "{leader}" to the reader. A finding of a label-bearing
        # kind with no label, phrased as though labels were available, is
        # exactly that situation.
        naked = Finding(kind=CONCENTRATION_LEADER, columns=("CUSTOMER", "AMT"),
                        numbers={"share": 66.0, "value": 10.0}, labels={},
                        materiality=0.9)
        sentences = TemplatePhraser().phrase(
            [naked], lang="en", labels_available=True)
        self.assertEqual(sentences, [])

    def test_a_sentence_with_every_value_supplied_is_kept(self):
        # The drop above is only meaningful if the normal case survives it.
        complete = Finding(kind=CONCENTRATION_LEADER, columns=("CUSTOMER", "AMT"),
                           numbers={"share": 66.0, "value": 10.0},
                           labels={"leader": "Acme"}, materiality=0.9)
        sentences = TemplatePhraser().phrase(
            [complete], lang="en", labels_available=True)
        self.assertEqual(len(sentences), 1)
        self.assertIn("Acme", sentences[0])

    def test_a_broken_phraser_costs_its_wording_not_the_analysis(self):
        class Exploding:
            name = "exploding"

            def phrase(self, findings, *, lang, labels_available):
                raise RuntimeError("no")

        narrative = build_narrative(
            build_evidence(LEADER_ROWS), lang="en", phraser=Exploding())
        self.assertTrue(narrative.sentences)
        self.assertEqual(narrative.phrasing, "template")
        self.assertTrue(narrative.substituted)
        self.assertEqual(narrative.substitution_reason, "phraser_failed")


class TestBothLanguages(unittest.TestCase):

    def test_every_line_differs_between_english_and_french(self):
        # Line by line, not paragraph by paragraph: a whole-paragraph
        # comparison passes when only one sentence of three was translated.
        for rows in (LEADER_ROWS, SERIES_ROWS,
                     [{"V": v} for v in [1, 500, 3, 900, 2, 700]]):
            english = build_narrative(build_evidence(rows), lang="en").sentences
            french = build_narrative(build_evidence(rows), lang="fr").sentences
            self.assertEqual(len(english), len(french))
            self.assertFalse(set(english) & set(french), rows[0])

    def test_french_writes_its_own_numbers(self):
        french = " ".join(build_narrative(build_evidence(LEADER_ROWS), lang="fr").sentences)
        self.assertIn("66,7 %", french)      # comma decimal, nbsp before %
        self.assertIn(" ", french)           # narrow no-break thousands group
        english = " ".join(build_narrative(build_evidence(LEADER_ROWS), lang="en").sentences)
        self.assertIn("66.7%", english)
        self.assertIn("6,200", english)

    def test_every_narrative_id_exists_in_both_languages(self):
        ids = [k for k in MESSAGES if k.startswith("narrative.")]
        self.assertGreater(len(ids), 15)
        for msg_id in ids:
            entry = MESSAGES[msg_id]
            self.assertTrue(entry.get("en"), msg_id)
            self.assertTrue(entry.get("fr"), msg_id)
            self.assertNotEqual(entry["en"], entry["fr"], msg_id)

    def test_a_label_free_sentence_exists_for_every_kind_that_needs_one(self):
        for kind in LABEL_BEARING_KINDS:
            self.assertIn(f"narrative.{kind}.unlabelled", MESSAGES, kind)

    def test_the_two_forms_of_a_label_bearing_sentence_carry_the_same_facts(self):
        # The label-free version must say the same thing minus the name, not
        # less. Both carry the share and the value.
        for lang in ("en", "fr"):
            labelled = MESSAGES[f"narrative.{CONCENTRATION_LEADER}"][lang]
            plain = MESSAGES[f"narrative.{CONCENTRATION_LEADER}.unlabelled"][lang]
            for slot in ("{share}", "{value}", "{column}"):
                self.assertIn(slot, labelled, (lang, slot))
                self.assertIn(slot, plain, (lang, slot))
            self.assertIn("{leader}", labelled)
            self.assertNotIn("{leader}", plain)


# ══════════════════════════════════════════════════════════════════════════════
# Governance — the regulated path
# ══════════════════════════════════════════════════════════════════════════════

class TestALabelFreeNarrative(unittest.TestCase):

    def test_a_customers_name_never_appears_when_labels_are_withheld(self):
        narrative = build_narrative(
            build_evidence(LEADER_ROWS, include_labels=False), lang="en")
        joined = " ".join(narrative.sentences)
        self.assertNotIn("Ospedale", joined)
        self.assertNotIn("Raffaele", joined)
        self.assertFalse(narrative.labels_included)

    def test_it_still_states_the_share_the_name_would_have_carried(self):
        # The point of the label-free sentence: the reader loses the name, not
        # the finding. A regulated tenant is told a single customer holds two
        # thirds of the total; they are simply not told which.
        plain = build_narrative(
            build_evidence(LEADER_ROWS, include_labels=False), lang="en")
        named = build_narrative(
            build_evidence(LEADER_ROWS, include_labels=True), lang="en")
        self.assertEqual(len(plain.sentences), len(named.sentences))
        self.assertIn("66.7%", " ".join(plain.sentences))
        self.assertIn("66.7%", " ".join(named.sentences))

    def test_the_customers_name_appears_when_labels_are_allowed(self):
        # The negative test above proves nothing unless the name can reach the
        # page at all.
        named = build_narrative(
            build_evidence(LEADER_ROWS, include_labels=True), lang="en")
        self.assertIn("Ospedale San Raffaele", " ".join(named.sentences))

    def test_the_label_free_id_is_chosen_from_the_kind_not_the_stripped_finding(self):
        # build_evidence(include_labels=False) empties `labels`, so a finding
        # no longer knows it would have carried one. Reading that runtime flag
        # chose the labelled sentence and printed a bare "{leader}".
        stripped = Finding(kind=CONCENTRATION_LEADER, columns=("CUSTOMER", "AMT"),
                           numbers={"share": 66.0, "value": 10.0}, labels={})
        self.assertEqual(
            message_id_for(stripped, labels_available=False),
            f"narrative.{CONCENTRATION_LEADER}.unlabelled")
        self.assertEqual(
            message_id_for(stripped, labels_available=True),
            f"narrative.{CONCENTRATION_LEADER}")

    def test_the_prompt_payload_carries_findings_and_never_a_row(self):
        payload = evidence_for_prompt(
            build_evidence(LEADER_ROWS, include_labels=False))
        self.assertEqual(payload["row_count"], len(LEADER_ROWS))
        self.assertTrue(payload["findings"])
        rendered = repr(payload)
        for row in LEADER_ROWS:
            self.assertNotIn(row["CUSTOMER"], rendered)
        for finding in payload["findings"]:
            self.assertNotIn("labels", finding)

    def test_the_prompt_payload_carries_labels_only_when_they_are_allowed(self):
        payload = evidence_for_prompt(
            build_evidence(LEADER_ROWS, include_labels=True))
        leaders = [f for f in payload["findings"] if f["kind"] == CONCENTRATION_LEADER]
        self.assertTrue(leaders)
        self.assertEqual(leaders[0]["labels"]["leader"], "Ospedale San Raffaele")


# ══════════════════════════════════════════════════════════════════════════════
# The checked-numbers rule
# ══════════════════════════════════════════════════════════════════════════════

class TestTheCheckedNumbersRule(unittest.TestCase):

    def setUp(self):
        self.evidence = build_evidence(SERIES_ROWS)
        self.assertTrue(self.evidence.findings)

    def test_prose_repeating_computed_figures_is_accepted(self):
        computed = self.evidence.all_numbers()
        text = f"Revenue climbed from {computed[1]:g} to {computed[2]:g} over the period."
        ok, reason = verify_against_evidence(text, self.evidence)
        self.assertTrue(ok, reason)

    def test_prose_carrying_a_figure_we_never_computed_is_rejected(self):
        # 4,517 is not in the evidence and is not a ratio of anything in it.
        # It must not be read as "4.517" and waved through as a small number:
        # that is precisely how an invented figure reaches the page.
        text = "Revenue climbed over the period, driven by a 4,517 unit swing."
        self.assertIn(4517.0, numbers_in(text))
        ok, reason = verify_against_evidence(text, self.evidence)
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("uncomputed_figure:"), reason)

    def test_rejection_falls_back_to_the_computed_sentences(self):
        result = enforce_checked_numbers(
            "Revenue grew by 4,517 units.", self.evidence, lang="en")  # noqa: E501
        self.assertEqual(result.phrasing, "template")
        self.assertTrue(result.substituted)
        self.assertTrue(result.sentences)
        self.assertNotIn("4,517", " ".join(result.sentences))

    def test_acceptance_keeps_the_models_wording(self):
        computed = self.evidence.all_numbers()
        text = f"Revenue rose steadily to {computed[2]:g}."
        result = enforce_checked_numbers(text, self.evidence, lang="en")
        self.assertEqual(result.phrasing, "llm")
        self.assertFalse(result.substituted)
        self.assertIn(text, result.sentences)

    def test_rounding_a_computed_figure_is_not_an_invention(self):
        # A model writing "66%" where the evidence holds 66.2 is rounding.
        evidence = build_evidence(LEADER_ROWS)
        share = [f for f in evidence.findings if f.kind == CONCENTRATION_LEADER]
        self.assertTrue(share)
        exact = share[0].numbers["share"]
        ok, reason = verify_against_evidence(f"About {round(exact):g}% of the total.", evidence)
        self.assertTrue(ok, reason)

    def test_small_numbers_in_fluent_prose_do_not_trip_the_rule(self):
        ok, _ = verify_against_evidence(
            "There are 3 things worth noting, and 2 of them concern the top rows.",
            self.evidence)
        self.assertTrue(ok)
        self.assertGreater(SMALL_NUMBER_CEILING, 1)

    def test_a_figure_just_above_the_small_ceiling_is_still_challenged(self):
        # 13 clears the ceiling by one and is neither computed nor derivable.
        self.assertEqual(SMALL_NUMBER_CEILING, 12.0)
        ok, _ = verify_against_evidence("The figure was 13 across the board.",
                                        self.evidence)
        self.assertFalse(ok)

    def test_the_derivation_allowance_does_not_span_unrelated_findings(self):
        # Across the whole evidence there are N-squared ratios, and that set
        # is dense enough to accept almost anything. Two findings that happen
        # to divide into an invented figure must not license it.
        evidence = AnalysisEvidence(findings=(
            Finding(kind=TREND_UP, columns=("A",),
                    numbers={"first": 100.0}, materiality=0.9),
            Finding(kind=SPREAD_HIGH, columns=("B",),
                    numbers={"mean": 188.0}, materiality=0.5),
        ))
        # 100/188*100 = 53.19 -- derivable only by crossing two findings.
        ok, reason = verify_against_evidence("Coverage sat at 53%.", evidence)
        self.assertFalse(ok, reason)

    def test_a_restatement_within_one_finding_is_allowed(self):
        evidence = AnalysisEvidence(findings=(
            Finding(kind=TREND_UP, columns=("A",),
                    numbers={"first": 100.0, "last": 188.0}, materiality=0.9),
        ))
        ok, reason = verify_against_evidence(
            "It ended at 188% of where it started.", evidence)
        self.assertTrue(ok, reason)

    def test_the_rule_reads_french_notation(self):
        evidence = build_evidence(LEADER_ROWS)
        # 6 200 with a narrow no-break space, and 66,7 with a comma decimal:
        # a naive parser reads these as 6 and 66, both small, and waves
        # through a paragraph it never actually checked.
        self.assertIn(6200.0, numbers_in("Le total atteint 6 200 euros."))
        self.assertIn(66.7, numbers_in("Soit 66,7 % du total."))
        ok, _ = verify_against_evidence(
            "Un client représente 66,7 % du total, soit 6 200.", evidence)
        self.assertTrue(ok)

    def test_a_french_paragraph_with_an_invented_figure_is_still_rejected(self):
        ok, reason = verify_against_evidence(
            "Le chiffre d'affaires a bondi de 4 517 sur la période.",
            build_evidence(LEADER_ROWS))
        self.assertFalse(ok, reason)

    def test_a_share_derived_from_two_computed_figures_is_allowed(self):
        evidence = AnalysisEvidence(findings=(
            Finding(kind=SPREAD_HIGH, columns=("V",),
                    numbers={"part": 250.0, "whole": 1000.0}, materiality=0.5),
        ))
        ok, reason = verify_against_evidence("That is 25% of the whole.", evidence)
        self.assertTrue(ok, reason)

    def test_an_empty_paragraph_is_not_treated_as_a_violation(self):
        ok, _ = verify_against_evidence("", self.evidence)
        self.assertTrue(ok)


# ══════════════════════════════════════════════════════════════════════════════
# Traceability
# ══════════════════════════════════════════════════════════════════════════════

class TestEvidenceId(unittest.TestCase):

    def test_the_same_findings_get_the_same_id(self):
        self.assertEqual(
            evidence_id(build_evidence(LEADER_ROWS)),
            evidence_id(build_evidence(list(LEADER_ROWS))))

    def test_different_findings_get_different_ids(self):
        self.assertNotEqual(
            evidence_id(build_evidence(LEADER_ROWS)),
            evidence_id(build_evidence(SERIES_ROWS)))

    def test_the_id_survives_the_language_it_is_spoken_in(self):
        evidence = build_evidence(LEADER_ROWS)
        self.assertEqual(
            build_narrative(evidence, lang="en").evidence_id,
            build_narrative(evidence, lang="fr").evidence_id)

    def test_redaction_does_not_change_the_id(self):
        # The id names the computation, and redacting a label does not change
        # what was computed. A reviewer following an id from a regulated
        # tenant's proof pack must land on the same findings as one following
        # it from the same query in a standard workspace.
        from core.analysis_evidence import redact_labels

        full = build_evidence(LEADER_ROWS, include_labels=True)
        self.assertEqual(evidence_id(full), evidence_id(redact_labels(full)))

    def test_the_id_changes_when_a_number_changes(self):
        one = AnalysisEvidence(findings=(
            Finding(kind=TREND_UP, columns=("A",), numbers={"pct": 10.0}),))
        two = AnalysisEvidence(findings=(
            Finding(kind=TREND_UP, columns=("A",), numbers={"pct": 11.0}),))
        self.assertNotEqual(evidence_id(one), evidence_id(two))

    def test_the_narrative_carries_the_id_a_reviewer_would_look_up(self):
        narrative = build_narrative(build_evidence(LEADER_ROWS), lang="en")
        self.assertTrue(narrative.evidence_id)
        self.assertEqual(len(narrative.evidence_id), 16)


# ══════════════════════════════════════════════════════════════════════════════
# Formatting details
# ══════════════════════════════════════════════════════════════════════════════

class TestFormatting(unittest.TestCase):

    def test_a_row_count_is_written_as_a_whole_number(self):
        self.assertEqual(format_number_for("count", 3.0, "en"), "3")
        self.assertEqual(format_number_for("n_total", 8.0, "en"), "8")

    def test_a_share_is_written_as_a_percentage(self):
        self.assertEqual(format_number_for("share", 66.2, "en"), "66.2%")
        self.assertEqual(format_number_for("share", 50.0, "en"), "50%")

    def test_a_correlation_keeps_two_places(self):
        self.assertEqual(format_number_for("r", 0.9, "en"), "0.90")

    def test_a_measured_amount_is_grouped(self):
        self.assertEqual(format_number_for("value", 6200.0, "en"), "6,200")

    def test_a_column_name_is_made_readable_without_losing_the_real_one(self):
        self.assertEqual(humanise_column("NET_REVENUE_AMT"), "Net Revenue Amt")
        self.assertEqual(humanise_column("net_revenue"), "Net Revenue")
        self.assertEqual(humanise_column("NetRevenue"), "NetRevenue")
        self.assertEqual(humanise_column(""), "")
        # The finding keeps the real name for the trace and the proof pack.
        finding = build_evidence(LEADER_ROWS).of_kind(CONCENTRATION_LEADER)[0]
        self.assertIn("NET_REVENUE_AMT", finding.columns)

    def test_a_correlations_sign_picks_the_sentence_not_a_placeholder(self):
        positive = Finding(kind=CORRELATION, columns=("A", "B"), numbers={"r": 0.9})
        negative = Finding(kind=CORRELATION, columns=("A", "B"), numbers={"r": -0.9})
        self.assertEqual(message_id_for(positive, labels_available=True),
                         "narrative.correlation.positive")
        self.assertEqual(message_id_for(negative, labels_available=True),
                         "narrative.correlation.negative")

    def test_a_negative_correlation_reads_as_opposition_in_both_languages(self):
        rows = [{"X": i, "Y": -i * 3.0} for i in range(1, 12)]
        english = " ".join(build_narrative(build_evidence(rows), lang="en").sentences)
        french = " ".join(build_narrative(build_evidence(rows), lang="fr").sentences)
        self.assertIn("opposite", english)
        self.assertIn("inverse", french)


if __name__ == "__main__":
    unittest.main()


# ══════════════════════════════════════════════════════════════════════════════
# The egress boundary
# ══════════════════════════════════════════════════════════════════════════════

class TestRedactingAtTheBoundary(unittest.TestCase):
    """``redact_labels`` exists so the decision sits at the egress boundary.

    The narrative shown to the user may name the customer they are already
    looking at in the table above it; the payload handed to a model must not.
    Doing it at construction time would force every caller to know which of
    those two it is.
    """

    def test_labels_survive_into_the_narrative_but_not_into_the_payload(self):
        from core.analysis_evidence import redact_labels

        evidence = build_evidence(LEADER_ROWS, include_labels=True)
        shown = build_narrative(evidence, lang="en")
        self.assertIn("Ospedale San Raffaele", " ".join(shown.sentences))

        sent = evidence_for_prompt(redact_labels(evidence))
        self.assertNotIn("Ospedale", repr(sent))
        self.assertNotIn("Raffaele", repr(sent))

    def test_redaction_keeps_every_finding_and_every_number(self):
        from core.analysis_evidence import redact_labels

        full = build_evidence(LEADER_ROWS, include_labels=True)
        redacted = redact_labels(full)
        self.assertEqual([f.kind for f in full.findings],
                         [f.kind for f in redacted.findings])
        self.assertEqual([f.numbers for f in full.findings],
                         [f.numbers for f in redacted.findings])
        self.assertFalse(redacted.labels_included)

    def test_redacting_twice_is_the_same_as_redacting_once(self):
        from core.analysis_evidence import redact_labels

        once = redact_labels(build_evidence(LEADER_ROWS, include_labels=True))
        self.assertIs(redact_labels(once), once)

    def test_a_redacted_narrative_still_uses_the_label_free_sentence(self):
        from core.analysis_evidence import redact_labels

        redacted = redact_labels(build_evidence(LEADER_ROWS, include_labels=True))
        sentences = build_narrative(redacted, lang="en").sentences
        joined = " ".join(sentences)
        self.assertNotIn("Ospedale", joined)
        self.assertNotIn("{", joined)
        self.assertIn("66.7%", joined)
