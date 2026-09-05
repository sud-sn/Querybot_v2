"""
tests/test_coverage_caveat_language.py

The coverage caveats, in both languages, executed rather than read.

A coverage caveat is the sentence that says the answer above it is less
complete than it looks: a forecast that was refused, a join that silently drops
rows, a window whose data stops early, a result cut off at its row cap. Five
modules built them by string concatenation, so a French reader got a French
answer with English warnings under it -- the one part of the response they most
need to be able to act on.

Every test here runs the real producer with the language active and asserts on
what it returns; none of them reads source text. Three of them also fail
against the ENGLISH code that was there before, because concatenation was
already getting English wrong:

  * "1 of 24 days are missing from this series" agreed the verb with the wrong
    number;
  * an "EBITDA" metric was reported as "ebitda records", because the whole
    phrase was lowercased to fit mid-sentence;
  * an unnamed metric was reported as "but The selected metric was nonzero",
    capitalised in the middle of a sentence.
"""

import asyncio
import textwrap
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from core import i18n
from core.date_coverage import check_date_coverage
from core.forecast_gate import assess_fit, evaluate_forecast_request
from core.join_coverage import check_join_coverage
from core.multi_period import PeriodPlan, annotate_period_change

REPO = Path(__file__).resolve().parents[1]


class _InFrench:
    """`with _InFrench():` -- the answer language for the block."""

    def __init__(self, lang="fr"):
        self.lang = lang

    def __enter__(self):
        self._token = i18n.activate_language(self.lang)
        return self

    def __exit__(self, *exc):
        i18n.deactivate_language(self._token)
        return False


def _months(values, start=1):
    return [{"PERIOD": f"2025-{start + i:02d}", "REVENUE": v}
            for i, v in enumerate(values)]


# ══════════════════════════════════════════════════════════════════════════════
# 1. core/forecast_gate.py -- the refusals
# ══════════════════════════════════════════════════════════════════════════════

class TheForecastRefusalsAreTranslated(unittest.TestCase):
    """Fifteen sentences that all begin "I did not project future periods".
    Each one is the only explanation the reader gets for a chart that did not
    appear, so an English one under a French answer is a dead end."""

    # Every refusal a real result can reach, with the rows that reach it.
    SCENARIOS = {
        "policy_blocked": lambda: evaluate_forecast_request(
            _months([100 + i * 5 for i in range(8)]),
            policy_allows_derived_visual=False),
        "no_temporal_axis": lambda: evaluate_forecast_request(
            [{"REGION": "North", "REVENUE": 1}, {"REGION": "South", "REVENUE": 2}]),
        "no_measure": lambda: evaluate_forecast_request(
            [{"PERIOD": "2025-01"}, {"PERIOD": "2025-02"}]),
        "masked_series": lambda: evaluate_forecast_request(
            _months([100, 110, 120, 130, 140, 150])[:5]
            + [{"PERIOD": "2025-06", "REVENUE": "[REDACTED]"}]),
        "truncated_result": lambda: evaluate_forecast_request(
            _months([100 + i * 5 for i in range(8)]), truncated=True),
        "multi_series": lambda: evaluate_forecast_request(
            [{"PERIOD": f"2025-{m:02d}", "REGION": r, "REVENUE": m * 10}
             for m in range(1, 5) for r in ("North", "South")]),
        "unordered_series": lambda: evaluate_forecast_request(
            list(reversed(_months([100 + i * 5 for i in range(8)])))),
        "irregular_cadence": lambda: evaluate_forecast_request(
            [{"PERIOD": p, "REVENUE": v} for p, v in
             [("2025-01", 10), ("2025-02", 20), ("2025-03", 30), ("2025-04", 40),
              ("2025-05", 50), ("2025-06", 60), ("2025-07", 70), ("2025-12", 80)]]),
        "constant_series": lambda: evaluate_forecast_request(_months([100] * 8)),
        "poor_fit": lambda: assess_fit(
            evaluate_forecast_request(_months([100 + i * 5 for i in range(8)])),
            0.10, 45.0),
    }

    def test_every_reachable_refusal_comes_back_in_french(self):
        for code, build in self.SCENARIOS.items():
            with self.subTest(reason=code):
                english = build()
                with _InFrench():
                    french = build()
                self.assertEqual(french.reason_code, code)
                self.assertTrue(french.caveat, "a refusal nobody can read")
                self.assertTrue(
                    french.caveat.startswith("Je n'ai pas projeté de périodes futures"),
                    french.caveat,
                )
                self.assertNotEqual(french.caveat, english.caveat)

    def test_the_reason_code_is_a_wire_token_and_never_translates(self):
        """reason_code is logged, branched on and compared. Translating it
        would make every consumer of it stop recognising the refusal."""
        with _InFrench():
            decision = evaluate_forecast_request(_months([100] * 8))
        self.assertEqual(decision.reason_code, "constant_series")

    def test_the_grain_on_the_decision_is_a_wire_token_too(self):
        """The chart and the pipeline read decision.grain. The sentence names
        the grain in French; the field must not."""
        with _InFrench():
            decision = evaluate_forecast_request(_months([100 + i * 5 for i in range(8)]))
        self.assertEqual(decision.grain, "month")
        self.assertTrue(decision.allowed)

    def test_a_short_series_names_its_grain_in_french(self):
        with _InFrench():
            decision = evaluate_forecast_request(_months([100, 110, 120, 130]))
        self.assertEqual(decision.reason_code, "too_short")
        self.assertIn("cette série compte 4 mois", decision.caveat)
        self.assertNotIn("months", decision.caveat)

    def test_the_grain_is_pluralised_by_the_catalogue_not_by_an_s(self):
        """`f"{grain}s"` is one rule for two languages that do not share it.
        "mois" is already its own plural, and French takes the singular at
        zero, where English does not."""
        self.assertEqual(i18n.grain_label("month", 1), "month")
        self.assertEqual(i18n.grain_label("month", 4), "months")
        self.assertEqual(i18n.grain_label("day", 0), "days")
        with _InFrench():
            self.assertEqual(i18n.grain_label("month", 1), "mois")
            self.assertEqual(i18n.grain_label("month", 4), "mois")
            self.assertEqual(i18n.grain_label("day", 0), "jour")
            self.assertEqual(i18n.grain_label("day", 4), "jours")

    def test_the_capped_horizon_note_is_translated(self):
        english = evaluate_forecast_request(
            _months([100 + i * 5 for i in range(8)]), horizon=9)
        with _InFrench():
            french = evaluate_forecast_request(
                _months([100 + i * 5 for i in range(8)]), horizon=9)
        self.assertEqual(english.notes,
                         ("I projected 4 months rather than 9: beyond about half "
                          "the length of the history a projection is guesswork.",))
        self.assertEqual(len(french.notes), 1)
        self.assertIn("J'ai projeté 4 mois au lieu de 9", french.notes[0])

    def test_the_grain_mismatch_note_names_both_grains_in_french(self):
        """The English built "this data is {grain}ly" -- an adverb derived from
        a wire token. The French adjective would have had to agree with two
        different nouns in the same sentence.

        The question here is the canonical English one, because the gate reads
        it with an English regex; section 6 asserts the pipeline hands it that
        rather than the reader's own words."""
        rows = _months([100 + i * 5 for i in range(8)])
        with _InFrench():
            french = evaluate_forecast_request(
                rows, question="show revenue by week", horizon=1)
        note = " ".join(french.notes)
        self.assertIn("par semaine", note)
        self.assertIn("par mois", note)
        self.assertNotIn("monthly", note)
        self.assertNotIn("moisly", note)

    def test_the_poor_fit_percentages_use_the_french_spacing(self):
        decision = evaluate_forecast_request(_months([100 + i * 5 for i in range(8)]))
        english = assess_fit(decision, 0.10, 45.0)
        with _InFrench():
            french = assess_fit(decision, 0.10, 45.0)
        self.assertIn("10%", english.caveat)
        self.assertIn("45%", english.caveat)
        # French puts a space before the sign, and never writes "10%".
        self.assertIn("10 %", french.caveat)
        self.assertIn("45 %", french.caveat)
        self.assertNotIn("10%", french.caveat)

    def test_a_single_missing_period_reads_grammatically(self):
        """Fails against the English that was here: "1 of 24 days are missing
        from this series" agreed the verb with `expected`, not `missing`."""
        days = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(20)]
        rows = ([{"PERIOD": d.date().isoformat(), "REVENUE": 100 + i * 3}
                 for i, d in enumerate(days)]
                + [{"PERIOD": "2025-01-24", "REVENUE": 200}])
        english = evaluate_forecast_request(rows)
        self.assertEqual(english.reason_code, "gaps_in_series")
        self.assertIn("this series is missing 3 of the 24 days", english.caveat)
        self.assertNotIn("are missing", english.caveat)
        with _InFrench():
            french = evaluate_forecast_request(rows)
        self.assertIn("il manque 3 des 24 jours", french.caveat)


# ══════════════════════════════════════════════════════════════════════════════
# 2. core/date_coverage.py -- the window that stops early
# ══════════════════════════════════════════════════════════════════════════════

class TheDateCoverageGapIsTranslated(unittest.TestCase):
    """The only live read of the true current data date. Every one of its four
    sentences is assembled from a role, a metric name and two counts."""

    DB = {"db_type": "azure_sql", "credentials": {}}
    BASE = {"amount": 7, "unit": "day", "fact_table": "DBO.F_ORDERS",
            "fact_column": "ORDER_DATE", "date_key_type": "native"}

    def _gap(self, policy=None, days=3, metric_days=None, **kw):
        replies = [[{"AnchorDate": "2026-07-20"}], [{"DaysWithData": days}]]
        if metric_days is not None:
            replies.append([{"DaysWithMetricData": metric_days}])
        with patch("core.contextual_dates.format_required_anchor",
                   return_value="(SELECT MAX(ORDER_DATE) FROM DBO.F_ORDERS)"), \
             patch("core.date_coverage.run_query", side_effect=replies):
            return check_date_coverage(
                self.DB, {**self.BASE, **(policy or {})}, "azure_sql", **kw)

    def test_the_sparse_day_window_is_french(self):
        with _InFrench():
            gap = self._gap()
        self.assertEqual(
            gap.message,
            "Vous avez demandé les 7 derniers jours, mais des enregistrements "
            "n'ont été trouvés que sur 3 dates métier (3 jours avec des "
            "données), jusqu'au 2026-07-20. Le résultat reflète les données "
            "disponibles.",
        )

    def test_the_sparse_metric_window_is_french(self):
        with _InFrench():
            gap = self._gap(policy={"amount": 2}, days=2, metric_days=1,
                            metric_name="Revenue",
                            metric_formula="SUM(REVENUE_AMOUNT)")
        self.assertIn("Des enregistrements existaient sur 2 dates métier", gap.message)
        self.assertIn("mais Revenue n'affichait une valeur non nulle que sur 1 jour",
                      gap.message)
        self.assertNotIn("nonzero", gap.message)

    def test_the_non_day_window_is_french_and_counts_the_unit(self):
        """English compounds the window as "6-month period"; French counts it.
        The two languages needed different words for the same value, which is
        exactly the shape the placeholder-parity guard cannot check -- so both
        count it now."""
        english = self._gap(policy={"amount": 6, "unit": "month"}, days=2,
                            metric_name="Revenue")
        with _InFrench():
            french = self._gap(policy={"amount": 6, "unit": "month"}, days=2,
                               metric_name="Revenue")
        self.assertIn("within the requested period of 6 months", english.message)
        self.assertIn("dans la période de 6 mois demandée", french.message)
        self.assertTrue(french.message.startswith("Les enregistrements de Revenue"))

    def test_the_grain_only_note_is_french(self):
        with patch("core.contextual_dates.format_required_anchor",
                   return_value="(SELECT MAX(ORDER_DATE) FROM DBO.F_ORDERS)"), \
             patch("core.date_coverage.run_query",
                   side_effect=[[{"AnchorDate": "2026-07-20"}]]), _InFrench():
            gap = check_date_coverage(
                self.DB,
                {**self.BASE, "amount": 1, "unit": "month",
                 "temporal_grain": "month", "counts_days": False},
                "azure_sql",
            )
        self.assertTrue(gap.message.startswith("Cette source est enregistrée par mois."))

    def test_french_takes_the_singular_at_one_and_at_zero(self):
        """English says "0 business dates". French says "0 date métier"; the
        rule the concatenation had was `if actual_days != 1: label += "s"`."""
        with _InFrench():
            one = self._gap(policy={"amount": 2}, days=1)
            zero = self._gap(days=0)
        self.assertIn("sur 1 date métier (1 jour avec des données)", one.message)
        self.assertIn("sur 0 date métier (0 jour avec des données)", zero.message)
        english_zero = self._gap(days=0)
        self.assertIn("on only 0 business dates (0 days with data)",
                      english_zero.message)

    def test_a_tenant_role_is_quoted_rather_than_glued_to_a_preposition(self):
        """The role is the tenant's own English word -- "invoice", "posting".
        "date de invoice" needs an elision French cannot decide from an
        arbitrary word, so the label is quoted instead."""
        with _InFrench():
            gap = self._gap(policy={"business_role": "Invoice Date"}, days=2)
        self.assertIn("sur 2 dates « invoice »", gap.message)
        self.assertNotIn("de invoice", gap.message)

    def test_the_metric_keeps_its_own_case(self):
        """Fails against the English that was here: the whole phrase was
        lowercased to sit mid-sentence, so an EBITDA metric was reported as
        "ebitda records"."""
        gap = self._gap(days=2, metric_name="EBITDA")
        self.assertIn("but EBITDA records were found", gap.message)

    def test_an_unnamed_metric_is_not_capitalised_mid_sentence(self):
        """Fails against the English that was here: "but The selected metric
        was nonzero on only 1 day"."""
        gap = self._gap(policy={"amount": 2}, days=2, metric_days=1,
                        metric_name="", metric_formula="SUM(REVENUE_AMOUNT)")
        self.assertIn("but the selected metric was nonzero", gap.message)
        with _InFrench():
            french = self._gap(policy={"amount": 2}, days=2, metric_days=1,
                               metric_name="", metric_formula="SUM(REVENUE_AMOUNT)")
        self.assertIn("mais l'indicateur sélectionné", french.message)


# ══════════════════════════════════════════════════════════════════════════════
# 3. core/join_coverage.py -- the join that drops rows
# ══════════════════════════════════════════════════════════════════════════════

class TheLossyJoinCaveatIsTranslated(unittest.TestCase):
    EDGE = {"id": 1, "from_entity": "Orders", "to_entity": "Customer"}

    def _message(self, *, rate=15.0, age_days=None, edge=None):
        rel = {"orphan_rate": rate}
        if age_days is not None:
            rel["last_profiled_at"] = (
                datetime.now() - timedelta(days=age_days, hours=1)).isoformat()
        with patch("store.get_relationship", return_value=rel):
            return check_join_coverage("acct", [edge or self.EDGE])[0]

    def test_the_undated_measurement_is_french(self):
        with _InFrench():
            message = self._message()
        self.assertTrue(message.startswith(
            "La jointure de Orders vers Customer a été mesurée comme excluant "
            "environ 15 % des lignes sans correspondance."))
        self.assertNotIn("15%", message)

    def test_the_recent_measurement_is_french(self):
        with _InFrench():
            message = self._message(rate=23.0, age_days=4)
        self.assertIn("exclut environ 23 % des lignes sans correspondance "
                      "(mesuré il y a 4 jours)", message)

    def test_the_stale_measurement_is_french(self):
        with _InFrench():
            message = self._message(rate=23.0, age_days=400)
        self.assertIn("lors de son dernier profilage, il y a 400 jours", message)
        self.assertIn("Reprofilez la relation", message)

    def test_today_and_one_day_ago_are_translated_not_suffixed(self):
        """`f"{n} day{'s' if n != 1 else ''} ago"` is an English rule with an
        English word inside it."""
        with _InFrench():
            today = self._message(age_days=0)
            yesterday = self._message(age_days=1)
        self.assertIn("(mesuré aujourd'hui)", today)
        self.assertIn("(mesuré il y a 1 jour)", yesterday)
        self.assertNotIn("jours)", yesterday)

    def test_the_unnamed_entities_fall_back_in_french(self):
        with _InFrench():
            message = self._message(edge={"id": 1})
        self.assertIn("La jointure de la table source vers la table jointe", message)

    def test_the_english_is_unchanged(self):
        """The control: this rewrite was a translation, not a rewording."""
        self.assertIn(
            "The join from Orders to Customer excludes about 23% of rows with "
            "no match (measured 4 days ago) — some data may not be counted.",
            self._message(rate=23.0, age_days=4),
        )


# ══════════════════════════════════════════════════════════════════════════════
# 4. core/multi_period.py -- the comparison that came back incomplete
# ══════════════════════════════════════════════════════════════════════════════

class TheNamedPeriodWarningsAreTranslated(unittest.TestCase):
    """Two of these were module-level constants. A constant is built at import
    time -- before any request, and therefore before any answer language -- so
    a French reader got the English sentence whatever they had chosen. These
    tests fail against a constant, because the module is already imported by
    the time the language is set."""

    def _plan(self, **overrides):
        """A plan per test -- `warnings` is the mutable channel annotate writes
        to, so a shared instance would leak between cases."""
        fields = {"labels": ["2024", "2025"], "aliases": ["2024", "2025"],
                  "predicates": ["a", "b"], "grain": "yearly", "date_field": {}}
        fields.update(overrides)
        return PeriodPlan(**fields)

    def test_the_truncated_warning_is_french(self):
        plan = self._plan()
        rows = [{"REVENUE_CATEGORY": "Pumps",
                 "NET_AMOUNT_2024": 1000, "NET_AMOUNT_2025": 1600}]
        with _InFrench():
            annotate_period_change(rows, plan, truncated=True)
        self.assertTrue(plan.warnings)
        self.assertIn("Le résultat s'est arrêté à son plafond de lignes",
                      plan.warnings[0])

    def test_the_cancelling_warning_is_french(self):
        plan = self._plan()
        rows = [
            {"REVENUE_CATEGORY": "Pumps", "NET_AMOUNT_2024": 1000, "NET_AMOUNT_2025": 2000},
            {"REVENUE_CATEGORY": "Valves", "NET_AMOUNT_2024": 2000, "NET_AMOUNT_2025": 1010},
        ]
        with _InFrench():
            annotate_period_change(rows, plan)
        self.assertTrue(plan.warnings, "the cancelling guard did not fire")
        self.assertIn("se compensent presque", plan.warnings[0])

    def test_the_missing_column_warning_is_french(self):
        plan = self._plan(labels=["2023", "2024", "2025"],
                          aliases=["2023", "2024", "2025"],
                          predicates=["a", "b", "c"])
        rows = [{"REVENUE_CATEGORY": "Pumps",
                 "NET_AMOUNT_2024": 1000, "NET_AMOUNT_2025": 1600}]
        with _InFrench():
            annotate_period_change(rows, plan)
        self.assertTrue(plan.warnings)
        self.assertIn("colonne pour 2023", plan.warnings[0])
        self.assertIn("ne porte donc qu'entre 2023 et 2025", plan.warnings[0])

    def test_the_english_warning_is_unchanged(self):
        plan = self._plan()
        rows = [{"REVENUE_CATEGORY": "Pumps",
                 "NET_AMOUNT_2024": 1000, "NET_AMOUNT_2025": 1600}]
        annotate_period_change(rows, plan, truncated=True)
        self.assertEqual(
            plan.warnings,
            ["The result stopped at its row cap, so I did not compute the "
             "change between periods or each category's share of it -- both "
             "would be statistics over the first rows only, not the whole "
             "result."],
        )


# ══════════════════════════════════════════════════════════════════════════════
# 5. core/result_renderer.py -- the truncated result, through the real renderer
# ══════════════════════════════════════════════════════════════════════════════

class TheTruncationCaveatReachesTheRenderedPayload(unittest.TestCase):
    """_send_results is executed, not read: the assertion is on the payload the
    adapter receives."""

    def _payload(self, *, rows, confidence_context):
        captured: dict = {}

        class _Adapter:
            async def send_assistant_response(self, event, response):
                captured.update(response)

            async def send_message(self, event, text):
                captured["text"] = text

        class _Event:
            platform = "portal"
            schema_hint = ""

        import core.result_renderer as rr
        asyncio.run(rr._send_results(
            _Event(), _Adapter(), "list every order", rows, "SELECT 1", 10,
            None, "acct", {"db_type": "azure_sql", "id": 1},
            confidence_context=confidence_context, cache_result=False,
        ))
        return captured

    ROWS = [{"ORDER_ID": i, "AMOUNT": i * 3} for i in range(1234)]

    def test_the_english_groups_thousands_with_a_comma(self):
        payload = self._payload(rows=list(self.ROWS),
                                confidence_context={"rows_truncated": True})
        caveats = payload.get("coverage_caveats") or []
        self.assertTrue(caveats)
        self.assertIn("Showing the first 1,234 rows", caveats[0])

    def test_the_french_groups_thousands_with_a_space_not_a_comma(self):
        """A comma is the decimal separator in French, so "1,234 lignes" reads
        as one and a bit -- an off-by-a-thousand in the sentence that says how
        much of the result the reader is being shown."""
        with _InFrench():
            payload = self._payload(rows=list(self.ROWS),
                                    confidence_context={"rows_truncated": True})
        caveats = payload.get("coverage_caveats") or []
        self.assertTrue(caveats)
        self.assertIn("Affichage des 1 234 premières lignes", caveats[0])
        self.assertNotIn("1,234", caveats[0])
        self.assertNotIn("Showing", caveats[0])

    def test_an_untruncated_result_gains_no_caveat(self):
        """The control. Without it the two above would pass on a renderer that
        appended the sentence to every answer."""
        payload = self._payload(rows=list(self.ROWS), confidence_context={})
        self.assertNotIn("coverage_caveats", payload)

    def test_a_forecast_caveat_travels_through_the_renderer_unaltered(self):
        """The renderer copies the pipeline's forecast caveats verbatim, so the
        translation has to have happened where they were produced."""
        with _InFrench():
            decision = evaluate_forecast_request(_months([100] * 8))
            payload = self._payload(
                rows=list(self.ROWS[:3]),
                confidence_context={"forecast_caveats": [decision.caveat]})
        self.assertIn(decision.caveat, payload.get("coverage_caveats") or [])
        self.assertIn("Je n'ai pas projeté", decision.caveat)


# ══════════════════════════════════════════════════════════════════════════════
# 6. core/query_pipeline.py -- the model-fallback note, from the real block
# ══════════════════════════════════════════════════════════════════════════════

class TheForecastBlockPutsFrenchCaveatsOnTheContext(unittest.TestCase):
    """The post-processing block compiled out of the real file and executed,
    the same way tests/test_post_process_actually_runs.py does it. A caveat
    that is translated in forecast_gate but concatenated here would pass every
    test in section 1 and still reach the reader in English."""

    def _block(self) -> str:
        src = (REPO / "core" / "query_pipeline.py").read_text(encoding="utf-8")
        start = src.index('            if _post_intents.get("forecast")')
        end = src.index('            if _post_intents.get("histogram")')
        block = textwrap.dedent(src[start:end])
        for marker in ("evaluate_forecast_request", "assess_fit", "compute_forecast("):
            assert marker in block, f"stale read of query_pipeline.py: {marker!r}"
        return block

    def _run(self, rows, analysis_question="forecast my revenue for the next 3 months"):
        import core.query_pipeline as qp

        class _Log:
            def info(self, msg, *a):
                pass

        env = {
            **vars(qp),
            "rows": rows,
            # What the reader typed, and the product's canonical English of it.
            # Every detector in this block reads the second one.
            "question": "prévois mon chiffre d'affaires pour les 3 prochains mois",
            "_analysis_question": analysis_question,
            "_post_intents": {"forecast": True},
            "_rows_truncated": False,
            "_confidence_context": {},
            "account_id": "acct", "portal_user": None, "event": None,
            "sql": "SELECT PERIOD, SUM(AMT) AS REVENUE FROM F GROUP BY PERIOD",
            "db_cfg": {"db_type": "azure_sql"},
            "log": _Log(),
        }
        exec(compile(self._block(), "<forecast-block>", "exec"), env)
        return env

    def test_a_refusal_reaches_the_context_in_french(self):
        with _InFrench():
            env = self._run(_months([100, 110, 120, 130]))
        caveats = env["_confidence_context"].get("forecast_caveats") or []
        self.assertTrue(caveats)
        self.assertTrue(caveats[0].startswith("Je n'ai pas projeté"), caveats[0])
        self.assertNotIn("did not project", caveats[0])

    def test_the_grain_mismatch_caveat_fires_for_a_french_reader(self):
        """The gate spots "you asked by week, this data is monthly" with a
        hand-written English regex. The block used to hand it the reader's own
        words, so on "par semaine" the regex matched nothing and a French user
        was never told the projection had changed grain -- the caveat was
        translated and unreachable at the same time.

        `question` here is French and `_analysis_question` is the product's
        canonical English of it, exactly as _handle_query_impl binds them.
        """
        with _InFrench():
            env = self._run(_months([100 + i * 5 for i in range(8)]),
                            analysis_question="show revenue by week")
        caveats = env["_confidence_context"].get("forecast_caveats") or []
        self.assertTrue(caveats, "the grain-mismatch caveat never fired")
        self.assertIn("par semaine", caveats[0])
        self.assertIn("par mois", caveats[0])

    def test_the_model_fallback_note_is_french(self):
        """The one caveat sentence that is written in the pipeline rather than
        in the gate, so it is the one a sweep of core/forecast_gate.py misses."""
        import core.forecast as forecast

        rows = _months([100 + i * 5 for i in range(14)])

        def _fake_forecast(rows_in, period_col, value_col, horizon, **kw):
            out = [dict(r) for r in rows_in]
            out[0]["__forecast_meta"] = {
                "model": "ols", "fell_back_from": "sarimax",
                "r2": 0.99, "backtest_mape": 2.0,
            }
            return out

        with patch.object(forecast, "compute_forecast", _fake_forecast), _InFrench():
            env = self._run(rows)
        caveats = env["_confidence_context"].get("forecast_caveats") or []
        self.assertIn(
            "J'ai projeté ces périodes avec un modèle ols ; le modèle sarimax "
            "n'était pas disponible ici.",
            caveats,
        )

    def test_the_english_model_fallback_note_is_unchanged(self):
        import core.forecast as forecast

        rows = _months([100 + i * 5 for i in range(14)])

        def _fake_forecast(rows_in, period_col, value_col, horizon, **kw):
            out = [dict(r) for r in rows_in]
            out[0]["__forecast_meta"] = {
                "model": "ols", "fell_back_from": "sarimax",
                "r2": 0.99, "backtest_mape": 2.0,
            }
            return out

        with patch.object(forecast, "compute_forecast", _fake_forecast):
            env = self._run(rows)
        self.assertIn(
            "I projected these periods with a ols model; the sarimax model was "
            "not available here.",
            env["_confidence_context"].get("forecast_caveats") or [],
        )


# ══════════════════════════════════════════════════════════════════════════════
# 7. The language does not leak
# ══════════════════════════════════════════════════════════════════════════════

class TheLanguageIsScopedToTheRequest(unittest.TestCase):
    def test_the_default_is_still_english_after_a_french_answer(self):
        with _InFrench():
            evaluate_forecast_request(_months([100] * 8))
        decision = evaluate_forecast_request(_months([100] * 8))
        self.assertTrue(decision.caveat.startswith("I did not project"))

    def test_an_unrecognised_grain_passes_through_untranslated(self):
        """A tenant's own temporal_grain is data. It must not come back as a
        missing-catalogue token."""
        with _InFrench():
            self.assertEqual(i18n.grain_label("fiscal fortnight", 3),
                             "fiscal fortnight")


if __name__ == "__main__":
    unittest.main()
