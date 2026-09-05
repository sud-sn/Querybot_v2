"""
tests/test_answer_card_language.py

The answer card's deterministic sentences, in the reader's language.

These are not chrome. When a question is not causal enough for the narration
model to run, build_answer's sentences ARE the answer -- so an English card
under a French portal is the product failing to answer the question in the
language it was asked in.

Every test calls the real builder and asserts on its return value. The English
assertions are as load bearing as the French ones: the whole surface was
rewritten from f-strings onto the catalogue, and a wording that shifted while
being "translated" is a regression in the language nobody was testing.
"""

from __future__ import annotations

import pytest

from core import i18n
from core.response_builder import (
    build_answer,
    build_assistant_response,
    infer_result_scope,
)


@pytest.fixture
def french():
    token = i18n.activate_language("fr")
    try:
        yield
    finally:
        i18n.deactivate_language(token)


# ══════════════════════════════════════════════════════════════════════════════
# The scope badge
# ══════════════════════════════════════════════════════════════════════════════

class TestTheScopeBadge:

    def test_the_badge_and_note_are_translated(self, french):
        scope = infer_result_scope([{"a": 1}], "revenue", "SELECT a FROM t")
        assert scope["badge"] == "Résultat renvoyé"
        assert scope["note"] == "Ceci correspond aux lignes renvoyées par la requête."

    def test_the_top_n_badge_carries_its_number(self, french):
        scope = infer_result_scope(
            [{"a": 1}] * 10, "top 10 customers",
            "SELECT a FROM t ORDER BY a DESC LIMIT 10", mode="ranking")
        assert scope["badge"] == "10 premiers uniquement"
        assert "10 premières lignes" in scope["note"]

    def test_the_single_top_result_has_its_own_wording(self, french):
        scope = infer_result_scope(
            [{"a": 1}], "best customer",
            "SELECT a FROM t ORDER BY a DESC LIMIT 1", mode="ranking")
        assert scope["badge"] == "Meilleur résultat uniquement"

    def test_english_is_unchanged(self):
        scope = infer_result_scope(
            [{"a": 1}] * 10, "top 10", "SELECT a FROM t LIMIT 10", mode="ranking")
        assert scope["badge"] == "Top 10 only"
        assert scope["note"] == "This result is based only on the top 10 returned rows."

    def test_the_stable_key_is_published_for_the_prompt(self):
        """core/insight.py puts the scope into the narration prompt. Sending a
        French token there would silently change what the model is asked; the
        answer's language is set by the prompt's own language rule."""
        scope = infer_result_scope(
            [{"a": 1}] * 10, "top 10", "SELECT a FROM t LIMIT 10", mode="ranking")
        assert scope["badge_key"] == "top_n"
        token = i18n.activate_language("fr")
        try:
            french_scope = infer_result_scope(
                [{"a": 1}] * 10, "top 10", "SELECT a FROM t LIMIT 10", mode="ranking")
        finally:
            i18n.deactivate_language(token)
        assert french_scope["badge_key"] == "top_n"

    def test_the_prompt_reads_the_key_not_the_label(self, french):
        """Executed against the real prompt builder rather than read for."""
        from core.insight import build_insight_prompt_from_contract
        contract = {
            "action": "explain", "mode": "ranking", "question": "top customers",
            "result_scope": infer_result_scope(
                [{"a": 1}] * 10, "top 10", "SELECT a FROM t LIMIT 10",
                mode="ranking"),
        }
        system, user = build_insight_prompt_from_contract(contract)
        assert "Scope: top_n" in user
        assert "10 premiers uniquement" not in user


# ══════════════════════════════════════════════════════════════════════════════
# The sentences
# ══════════════════════════════════════════════════════════════════════════════

class TestNothingMatched:

    def test_it_is_french(self, french):
        answer = build_answer([], "revenue in 2099")
        assert answer["headline"] == \
            "Aucune donnée correspondante n'a été trouvée pour cette question."
        assert answer["comparison"] == "Essayez d'ajuster les filtres ou la période."

    def test_zero_takes_the_french_singular(self, french):
        """English says "0 rows"; French says "0 ligne". The inline
        `{'s' if n != 1 else ''}` this replaces got that wrong in French and
        there was no way to express the difference."""
        assert build_answer([], "x")["short_value"] == "0 ligne"

    def test_english_still_says_zero_rows(self):
        assert build_answer([], "x")["short_value"] == "0 rows"


class TestASingleScalar:

    def test_french_puts_a_space_before_the_colon(self, french):
        """Not a nicety: "Marge: 12" is a typographic error in French. It is
        also why this is a message and not an f-string join."""
        answer = build_answer([{"total_revenue": 1250}], "total revenue")
        assert answer["headline"] == "Total Revenue : 1,250"

    def test_english_does_not(self):
        answer = build_answer([{"total_revenue": 1250}], "total revenue")
        assert answer["headline"] == "Total Revenue: 1,250"

    def test_the_column_name_is_never_translated(self, french):
        """It is the customer's schema. A translated header stops matching the
        table rendered underneath it."""
        answer = build_answer([{"marge_brute": 12}], "marge")
        assert "Marge Brute" in answer["headline"]


class TestARanking:

    ROWS = [{"region": "North", "revenue": 900},
            {"region": "South", "revenue": 400},
            {"region": "East", "revenue": 100}]

    def test_the_leader_sentence_is_french(self, french):
        answer = build_answer(self.ROWS, "revenue by region")
        assert answer["headline"] == "North arrive en tête avec 900."

    def test_the_gap_sentence_is_french(self, french):
        answer = build_answer(self.ROWS, "revenue by region")
        assert answer["comparison"] == "500 de plus que le résultat suivant"

    def test_english_is_unchanged(self):
        answer = build_answer(self.ROWS, "revenue by region")
        assert answer["headline"] == "North leads at 900."
        assert answer["comparison"] == "500 above the next result"

    def test_a_category_value_is_never_translated(self, french):
        """Row data is the customer's, not ours."""
        rows = [{"region": "Nord", "revenue": 900}, {"region": "Sud", "revenue": 1}]
        assert "Nord" in build_answer(rows, "revenue")["headline"]

    def test_the_single_top_result_wording(self, french):
        scope = infer_result_scope(
            [self.ROWS[0]], "best region",
            "SELECT region, revenue FROM t ORDER BY revenue DESC LIMIT 1",
            mode="ranking")
        answer = build_answer([self.ROWS[0]], "best region", scope)
        assert answer["headline"] == "Résultat le mieux classé : North, avec 900."
        assert answer["comparison"] == "Cette carte n'affiche que la première ligne"


class TestATimeSeries:

    ROWS = [{"month": "2026-01", "revenue": 100},
            {"month": "2026-02", "revenue": 150},
            {"month": "2026-03", "revenue": 220}]

    # `comparison` is `scope.get("badge") or <the sentence>`, and every scope
    # infer_result_scope builds carries a badge -- so the trend sentence is
    # reached only by a caller that passes a scope of its own without one. That
    # precedence is the code's, not this test's, and it predates the catalogue.
    # Truthy, so build_answer keeps it instead of inferring one, but with
    # no badge for `comparison` to prefer.
    NO_BADGE: dict = {"kind": "time_series"}

    def test_the_close_is_french(self, french):
        assert build_answer(self.ROWS, "revenue by month")["headline"] == \
            "2026-03 a terminé à 220."

    def test_the_trend_sentence_is_french(self, french):
        answer = build_answer(self.ROWS, "revenue by month", dict(self.NO_BADGE))
        assert answer["comparison"] == "Tendance à la hausse par rapport à 100 au départ"

    def test_a_falling_series_conjugates_differently(self, french):
        answer = build_answer(list(reversed(self.ROWS)), "revenue by month",
                              dict(self.NO_BADGE))
        assert answer["comparison"] == "Tendance à la baisse par rapport à 220 au départ"

    def test_english_is_unchanged(self):
        answer = build_answer(self.ROWS, "revenue by month", dict(self.NO_BADGE))
        assert answer["headline"] == "2026-03 closed at 220."
        assert answer["comparison"] == "Trend is up versus 100 at the start"

    def test_the_badge_still_wins_when_the_scope_has_one(self):
        """Pins the precedence above, so a future edit to the sentence cannot
        quietly start overriding a scope badge that used to show."""
        assert build_answer(self.ROWS, "revenue by month")["comparison"] == \
            "Returned result"


class TestANamedPeriodComparison:

    ROWS = [{"category": "Pumps", "revenue_2025": 100, "revenue_2026": 150},
            {"category": "Valves", "revenue_2025": 200, "revenue_2026": 180}]
    LABELS = ["2025", "2026"]

    def _answer(self):
        return build_answer(self.ROWS, "revenue 2025 vs 2026",
                            period_labels=self.LABELS)

    def test_the_movement_is_one_conjugated_sentence(self, french):
        """English joins a verb to a percentage -- "rose 10.0%". French
        conjugates and agrees the participle with the measure, so there is no
        seam in the middle for a translated adverb to slot into."""
        headline = self._answer()["headline"]
        assert headline.startswith("Revenue a augmenté de 10.0% entre 2025 et 2026")
        assert "rose" not in headline

    def test_the_mover_clause_is_french(self, french):
        assert "c'est Pumps qui a le plus varié" in self._answer()["headline"]

    def test_the_comparison_is_french(self, french):
        assert self._answer()["comparison"] == "+10.0% par rapport à 2025"

    def test_english_is_unchanged(self):
        answer = self._answer()
        assert answer["headline"].startswith("Revenue rose 10.0% from 2025 to 2026")
        assert "; Pumps moved the most, +50" in answer["headline"]
        assert answer["comparison"] == "+10.0% versus 2025"

    def test_a_flat_pair_does_not_claim_a_direction(self, french):
        rows = [{"category": "Pumps", "revenue_2025": 100, "revenue_2026": 100}]
        headline = build_answer(rows, "revenue", period_labels=self.LABELS)["headline"]
        assert "est resté stable" in headline
        assert "augmenté" not in headline and "diminué" not in headline


class TestAListOfNames:

    def test_the_count_sentence_is_french(self, french):
        rows = [{"name": n} for n in ("Ada", "Grace", "Alan", "Edsger", "Barbara")]
        answer = build_answer(rows, "who are the engineers?", {"kind": "table"})
        assert answer["headline"] == "5 résultats trouvés pour : who are the engineers"
        assert answer["short_value"] == "5 lignes"
        assert "+2 autres" in answer["comparison"]

    def test_one_result_takes_the_singular(self, french):
        # Two columns: a one-row one-column result is a scalar, not a list.
        answer = build_answer([{"first": "Ada", "last": "Lovelace"}], "who?")
        assert answer["headline"].startswith("1 résultat trouvé pour")

    def test_english_is_unchanged(self):
        rows = [{"name": n} for n in ("Ada", "Grace", "Alan", "Edsger", "Barbara")]
        answer = build_answer(rows, "who are the engineers?", {"kind": "table"})
        assert answer["headline"] == "Found 5 results for: who are the engineers"
        assert answer["short_value"] == "5 rows"
        assert "+2 more" in answer["comparison"]


class TestAnEmptyAggregate:

    ROWS = [{"total_margin": None}]

    def test_the_calm_copy_is_french(self, french):
        answer = build_answer(self.ROWS, "total margin last month")
        assert answer["short_value"] == "Aucune donnée"
        assert answer["headline"] == \
            "Aucune donnée de total Margin n'a été trouvée pour la période demandée."
        assert answer["comparison"] == \
            "La requête a abouti, mais aucune valeur de mesure n'a été renvoyée."

    def test_the_non_temporal_target_differs(self, french):
        answer = build_answer(self.ROWS, "total margin for pumps")
        assert "les filtres actuels" in answer["headline"]

    def test_english_is_unchanged(self):
        answer = build_answer(self.ROWS, "total margin last month")
        assert answer["headline"] == \
            "No total Margin data was found for the requested period."
        assert answer["short_value"] == "No data"


# ══════════════════════════════════════════════════════════════════════════════
# End to end
# ══════════════════════════════════════════════════════════════════════════════

class TestTheWholePayload:

    ROWS = [{"region": "North", "revenue": 900}, {"region": "South", "revenue": 400}]

    def _payload(self):
        return build_assistant_response(
            question="revenue by region", rows=self.ROWS,
            sql="SELECT region, revenue FROM t", duration_ms=42)

    def test_the_card_a_french_reader_receives_is_french(self, french):
        payload = self._payload()
        assert payload["answer"]["headline"] == "North arrive en tête avec 900."
        assert payload["answer"]["comparison"] == "500 de plus que le résultat suivant"
        assert payload["answer"]["scope_badge"] == "Distribution complète"
        assert payload["result_scope"]["note"] == \
            "Ce résultat reflète la distribution complète renvoyée."

    def test_the_same_call_in_english_is_english(self):
        payload = self._payload()
        assert payload["answer"]["headline"] == "North leads at 900."
        assert payload["answer"]["scope_badge"] == "Full distribution"
        assert payload["result_scope"]["note"] == \
            "This result reflects the full returned distribution."

    def test_the_language_is_read_at_call_time_not_import_time(self):
        """A module-level t() would freeze the catalogue at import and every
        reader would get whichever language happened to be active first."""
        english = self._payload()["answer"]["headline"]
        token = i18n.activate_language("fr")
        try:
            french = self._payload()["answer"]["headline"]
        finally:
            i18n.deactivate_language(token)
        again = self._payload()["answer"]["headline"]
        assert english == again != french


class TestTheLanguageRuleReachesTheModel:

    def test_a_french_reader_gets_a_french_output_rule(self, french):
        from core.insight import build_insight_prompt_from_contract
        system, _ = build_insight_prompt_from_contract(
            {"action": "explain", "mode": "table", "question": "revenue",
             "result_scope": {}})
        assert "Rédigez toute votre réponse en français" in system

    def test_an_english_reader_gets_no_extra_rule(self):
        """Every prompt in the product is written in English and has been
        tuned. "Answer in English" is a change to it for no behavioural gain."""
        from core.insight import build_insight_prompt_from_contract
        system, _ = build_insight_prompt_from_contract(
            {"action": "explain", "mode": "table", "question": "revenue",
             "result_scope": {}})
        assert "LANGUE" not in system
        assert "français" not in system

    def test_the_structural_labels_are_pinned_to_english(self, french):
        """parse_insight_response matches "HEADLINE:" and friends literally, so
        a translated label is a response that parses as unlabelled prose."""
        from core.insight import build_insight_prompt_from_contract
        system, _ = build_insight_prompt_from_contract(
            {"action": "explain", "mode": "table", "question": "revenue",
             "result_scope": {}})
        rule = system[system.index("LANGUE"):system.index("RULES:")]
        for label in ("HEADLINE:", "SECTION:", "BODY:", "DETAIL:", "NEXT:"):
            assert label in rule, label

    def test_the_schema_is_pinned_too(self, french):
        from core.insight import build_insight_prompt_from_contract
        system, _ = build_insight_prompt_from_contract(
            {"action": "explain", "mode": "table", "question": "revenue",
             "result_scope": {}})
        rule = system[system.index("LANGUE"):system.index("RULES:")]
        assert "ne les traduisez pas" in rule

    def test_the_period_comparison_prompt_carries_it_too(self, french):
        from core.period_comparison import build_period_comparison_narrative_prompt
        system, _ = build_period_comparison_narrative_prompt(
            current_brief={}, prior_brief={},
            question="revenue this year vs last",
            current_label="2026", prior_label="2025",
        )
        assert "Rédigez toute votre réponse en français" in system
