"""
tests/test_the_pipelines_disclosures_speak_french.py

Three messages the pipeline sends BEFORE the answer were English f-strings built
inline, so a French session got English prose in the middle of its own
conversation.

The worst of the three is the ranked-relationship disclosure. Near-tied join
paths are ranked deterministically rather than escalated to a clarification, and
that trade is only honest because the choice is DISCLOSED and the reader is told
how to redirect it:

    Using the **Invoice Date** relationship to reach EMDW_DMART.DT_DMS.
    Ask again naming *Shipment Date* to use the other one.

An instruction nobody can read is the same as no disclosure at all -- so in
French the join path was chosen silently, which is exactly what ranking instead
of escalating was supposed to stop being. On EMCO's shape this fires often: one
fact reaches the shared DT_DMS dimension through several role aliases, so there
is nearly always an alternative to name.

The other two: the fixed-SQL-metric warning that explains why a "par succursale"
was ignored, and the notice that identifiers were removed from the question. Both
were reader-facing English literals on the same path.

The two that carry data are extracted into functions so they can be EXECUTED
here rather than read. The third is a bare catalogue lookup at its call site, and
is covered by the structural check at the bottom of this file.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.i18n import MESSAGES, t
from core.query_pipeline import (
    metric_not_groupable_disclosure,
    ranked_relationship_disclosure,
)

RANKED = {
    "chosen": "Invoice Date",
    "target": "EMDW_DMART.DT_DMS",
    "alternatives": ["Shipment Date", "Order Date"],
}

NEW_KEYS = [
    "disclosure.relationship.ranked",
    "disclosure.metric.query_not_groupable",
    "disclosure.metric.this_metric",
    "disclosure.question.identifiers_removed",
]


class TestTheJoinPathChoiceIsReadable:

    def test_english_says_what_it_always_said(self):
        note = ranked_relationship_disclosure(RANKED, lang="en")
        assert "Using the **Invoice Date** relationship" in note
        assert "EMDW_DMART.DT_DMS" in note
        assert "*Shipment Date*" in note

    def test_french_is_french(self):
        note = ranked_relationship_disclosure(RANKED, lang="fr")
        assert "Utilisation de la relation" in note
        assert "Using the" not in note
        assert "Ask again" not in note

    def test_both_name_the_path_taken_and_the_one_to_ask_for(self):
        """Translating the sentence must not drop the two names that make it
        actionable -- the path used and the word to say for the other."""
        for lang in ("en", "fr"):
            note = ranked_relationship_disclosure(RANKED, lang=lang)
            assert "Invoice Date" in note, lang
            assert "Shipment Date" in note, lang
            assert "EMDW_DMART.DT_DMS" in note, lang

    def test_the_first_alternative_is_the_one_offered(self):
        note = ranked_relationship_disclosure(RANKED, lang="fr")
        assert "Shipment Date" in note
        assert "Order Date" not in note

    @pytest.mark.parametrize("ranked", [
        {},
        None,
        {"chosen": "Invoice Date", "target": "EMDW_DMART.DT_DMS"},
        {"chosen": "Invoice Date", "alternatives": []},
        {"alternatives": ["Shipment Date"]},
        {"chosen": "", "alternatives": ["Shipment Date"]},
        {"chosen": "Invoice Date", "alternatives": [""]},
    ])
    def test_nothing_is_disclosed_when_there_was_no_choice(self, ranked):
        """One thing for the caller to check instead of three, and no empty
        "Using the **** relationship" sent to anyone."""
        assert ranked_relationship_disclosure(ranked, lang="fr") == ""

    def test_a_missing_target_does_not_cost_the_message(self):
        note = ranked_relationship_disclosure(
            {"chosen": "Invoice Date", "alternatives": ["Shipment Date"]},
            lang="fr")
        assert note == "" or "Invoice Date" in note


class TestTheFixedMetricWarningIsReadable:

    def test_it_names_the_metric_in_both_languages(self):
        for lang in ("en", "fr"):
            note = metric_not_groupable_disclosure(
                {"label": "Ventes nettes", "name": "net_sales"}, lang=lang)
            assert "Ventes nettes" in note, lang

    def test_french_is_french(self):
        note = metric_not_groupable_disclosure({"label": "Ventes nettes"},
                                               lang="fr")
        assert "requête SQL figée" in note
        assert "fixed SQL query" not in note

    def test_the_label_is_preferred_over_the_machine_name(self):
        note = metric_not_groupable_disclosure(
            {"label": "Ventes nettes", "name": "net_sales_amt"}, lang="fr")
        assert "Ventes nettes" in note
        assert "net_sales_amt" not in note

    def test_the_name_is_used_when_there_is_no_label(self):
        note = metric_not_groupable_disclosure({"name": "net_sales"}, lang="fr")
        assert "net_sales" in note

    @pytest.mark.parametrize("metric", [{}, None, {"label": "", "name": ""}])
    def test_the_unnamed_fallback_is_translated_too(self, metric):
        """"This metric" was inside the f-string's own default, so it was the
        one fragment a translation of the sentence would have missed."""
        note = metric_not_groupable_disclosure(metric, lang="fr")
        assert "Cet indicateur" in note
        assert "This metric" not in note


class TestTheCatalogueCarriesBothLanguages:

    @pytest.mark.parametrize("key", NEW_KEYS)
    def test_the_key_exists_in_english_and_french(self, key):
        assert key in MESSAGES, key
        assert MESSAGES[key].get("en")
        assert MESSAGES[key].get("fr")

    @pytest.mark.parametrize("key", NEW_KEYS)
    def test_the_french_was_actually_written(self, key):
        """A copy of the English in the fr slot passes every other test here."""
        assert MESSAGES[key]["fr"] != MESSAGES[key]["en"], key

    @pytest.mark.parametrize("key", NEW_KEYS)
    def test_every_placeholder_survives_translation(self, key):
        import re
        placeholders = set(re.findall(r"\{(\w+)\}", MESSAGES[key]["en"]))
        assert set(re.findall(r"\{(\w+)\}", MESSAGES[key]["fr"])) == placeholders

    def test_the_identifier_notice_is_reachable_by_key(self):
        english = t("disclosure.question.identifiers_removed", lang="en")
        french = t("disclosure.question.identifiers_removed", lang="fr")
        assert "Personal identifiers" in english
        assert "identifiants personnels" in french
        assert "Personal identifiers" not in french


# ── The ratchet ──────────────────────────────────────────────────────────────
#
# A structural check, not a behavioural one, and it is here for one reason: the
# three messages fixed above are three of twenty-seven, and nothing stopped a
# twenty-eighth. The rest are the H8 backlog -- terminal failure and limit
# messages, which are the next pass, not this one. They are QUARANTINED by their
# opening text rather than hidden, so an existing one cannot mask a new one, and
# emptying this list is the definition of that pass being done.
#
# Matched by opening text, not by line number: a line number stops covering the
# code it names the moment anything is inserted above it.
_UNTRANSLATED_BACKLOG = {
    "",  # a pass-through f-string that opens with an already-translated variable
    "I can't answer this confidently yet — it tou",
    "I could not confirm a safe business identifi",
    "I could not safely apply that operation to t",
    "I resolved the business event to count, but ",
    "I retained your requested time period, but c",
    "I still do not have enough governed context ",
    "I understand that you want to count ",
    "I understand the analytical request, but I c",
    "This metric is blocked by the workspace data",
    "This request is blocked by the workspace dat",
    "⏱ Query timed out after 3 minutes. Try addin",
    "⏱ The trend query timed out after 3 minutes.",
    "⚠️ ",
    "⚠️ AI error: ",
    "⚠️ Config error: ",
    "⚠️ Knowledge Base not ready.",
    "⚠️ No database assigned. Contact your admini",
    "⚠️ No tables are available to query. Contact",
    "⚠️ No tables from the **",
    "❌ ",
    "❌ Monthly query limit reached (",
    "❌ Monthly token limit reached (",
    "🔒 *No table access assigned.*\n\nYour account ",
}


def _reader_literals() -> list[tuple[int, str]]:
    """Every send_message payload in the pipeline that is a hardcoded string.

    Read as a syntax tree rather than as text: the question is whether the
    argument IS a literal, which no substring search can answer.
    """
    source = (Path(__file__).resolve().parents[1]
              / "core" / "query_pipeline.py").read_text(encoding="utf-8")
    function = next(
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
        and node.name == "_handle_query_impl"
    )
    found = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name != "send_message":
            continue
        for arg in node.args[1:]:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                found.append((node.lineno, arg.value))
            elif isinstance(arg, ast.JoinedStr):
                opening = next(
                    (str(part.value) for part in arg.values
                     if isinstance(part, ast.Constant) and str(part.value).strip()),
                    "",
                )
                found.append((node.lineno, opening))
    return found


class TestNoNewEnglishLiteralReachesTheReader:

    def test_the_scan_finds_the_call_sites_it_is_named_for(self):
        """If send_message is ever renamed this check silently covers nothing."""
        assert len(_reader_literals()) >= 20

    def test_the_three_fixed_here_are_gone(self):
        openings = {text for _line, text in _reader_literals()}
        for fragment in ("ℹ️ Using the", "is a fixed SQL query",
                         "Personal identifiers in your question"):
            assert not any(fragment in opening for opening in openings), fragment

    def test_no_literal_outside_the_known_backlog(self):
        new = sorted(
            f"line {line}: {text[:60]!r}"
            for line, text in _reader_literals()
            if text[:44] not in _UNTRANSLATED_BACKLOG
        )
        assert not new, (
            "a reader-facing message must come from the catalogue, not a "
            "literal — a French session reads these:\n  " + "\n  ".join(new))
