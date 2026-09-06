# -*- coding: utf-8 -*-
"""Digests and alerts, in the language of whoever receives them.

These are the least forgiving place in the product to be in the wrong
language. A reader who asked a question has their own words on screen to give
an English sentence context; a reader who is simply sent a digest at 8am, or
an alert because a number moved, has nothing.

They are also the hardest place to get the language from. Both run on the
notification scheduler's thread with no request behind them, so the ContextVar
every other surface reads is empty — and would stay empty even if something
had set it, because it does not cross a thread boundary. The language comes
from the recipient's own row and is passed in.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.i18n import MESSAGES  # noqa: E402
from core.report_engine import _format_metric_line, build_report_response  # noqa: E402

NBSP = " "      # U+00A0, what format_percent puts before the sign
NNBSP = " "     # U+202F, the French thousands separator

_DEFAULT_RECIPIENT = {"id": 7, "role": "analyst"}


class TestADigestSpeaksToItsRecipient(unittest.TestCase):

    def _line(self, result, lang):
        return _format_metric_line(result, lang)

    def test_a_value_line_is_translated(self):
        result = {"ok": True, "metric_name": "Net Revenue",
                  "rows": [{"VALUE": 1234.5}]}
        self.assertNotEqual(self._line(result, "fr"), self._line(result, "en"))

    def test_every_metric_outcome_is_translated(self):
        outcomes = {
            "no data": {"ok": True, "metric_name": "M", "rows": []},
            "chart": {"ok": True, "metric_name": "M",
                      "rows": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]},
            "no access": {"ok": False, "metric_name": "M", "reason": "access_denied"},
            "failed": {"ok": False, "metric_name": "M", "reason": "boom"},
        }
        for name, result in outcomes.items():
            with self.subTest(name):
                self.assertNotEqual(self._line(result, "fr"), self._line(result, "en"))

    def test_the_metric_name_and_value_survive_translation(self):
        result = {"ok": True, "metric_name": "Chiffre d'affaires",
                  "rows": [{"VALUE": 1234.5}]}
        line = self._line(result, "fr")
        self.assertIn("Chiffre d'affaires", line)
        self.assertIn("1", line)

    def test_an_unnamed_metric_is_named_in_the_readers_language(self):
        result = {"ok": True, "metric_name": "", "rows": []}
        self.assertNotEqual(self._line(result, "fr"), self._line(result, "en"))

    def test_the_language_comes_from_the_recipient_row_by_default(self):
        """Nothing activates a ContextVar on the scheduler's thread."""
        with patch("store.list_metrics", return_value=[]):
            french = build_report_response("acct", {"lang": "fr"}, {"name": "Daily"})
            english = build_report_response("acct", {"lang": "en"}, {"name": "Daily"})
        self.assertNotEqual(french["message"], english["message"])

    def test_an_explicit_language_overrides_the_row(self):
        with patch("store.list_metrics", return_value=[]):
            forced = build_report_response("acct", {"lang": "en"}, {"name": "D"}, "fr")
            row = build_report_response("acct", {"lang": "fr"}, {"name": "D"})
        self.assertEqual(forced["message"], row["message"])

    def test_a_user_with_no_language_still_gets_a_message(self):
        with patch("store.list_metrics", return_value=[]):
            for user in ({}, {"lang": ""}, {"lang": None}):
                self.assertTrue(build_report_response("a", user, {"name": "D"})["message"])

    def test_the_report_name_survives_into_the_french_message(self):
        with patch("store.list_metrics", return_value=[{"id": 1}]), \
                patch("store.report_store.list_report_metrics", return_value=[]):
            message = build_report_response(
                "acct", {"lang": "fr"}, {"id": 1, "name": "Ventes du jour"})["message"]
        self.assertIn("Ventes du jour", message)


class TestAnAlertSpeaksToItsRecipient(unittest.TestCase):
    """
    Every assertion here comes from a real ``check_alert_now`` return value.

    An earlier draft of this class read the function's source for
    ``format_decimal(current, 2, lang=lang)`` and friends. That proves the
    characters are present, not that they run: the same assertions pass over a
    branch that never executes, and they object to any honest refactor. The
    fixture below fakes only boundaries — the store, the governed executor,
    the compliance profile — so the whole body of the function runs.
    """

    @staticmethod
    def _literal_chunks(msg_id, lang):
        """The template's own words, with the placeholders taken out.

        Read from the catalogue rather than pasted here, so rewording a
        message does not turn this into a failing test about nothing.
        """
        import re
        template = MESSAGES[msg_id][lang]
        return [c for c in re.split(r"\{[a-z_]+\}", template) if len(c.strip()) >= 5]

    def _assert_sentence_is(self, message, msg_id, lang):
        other = "en" if lang == "fr" else "fr"
        wanted = self._literal_chunks(msg_id, lang)
        self.assertTrue(wanted, msg_id)
        for chunk in wanted:
            self.assertIn(chunk, message, (msg_id, lang, chunk))
        for chunk in self._literal_chunks(msg_id, other):
            if chunk not in wanted:
                self.assertNotIn(chunk, message, (msg_id, other, chunk))

    def _check(self, lang, *, triggered=True, user=_DEFAULT_RECIPIENT):
        import store

        import core.alert_engine as engine
        from core.compliance.governed_query import GovernedQueryResult

        alert = {
            "id": "a1", "account_id": "acct", "user_id": 7,
            "sql": "SELECT 1", "baseline_value": 1000.0,
            "condition": "change_pct", "threshold": 5.0 if triggered else 500.0,
            "metric_col": "revenue", "status": "active",
        }
        governed = GovernedQueryResult(
            rows=[{"revenue": 1234.5}], sql="SELECT 1", decision=None,
            analysis=type("A", (), {"resources": []})(), row_obligations=[],
            truncated=False,
        )
        with patch.object(engine, "get_alert", return_value=alert), \
                patch.object(engine, "_load", return_value=[alert]), \
                patch.object(engine, "_save"), \
                patch.object(store, "get_user", return_value=user), \
                patch.object(store, "get_client_state", return_value={"schema_dir": ""}), \
                patch.object(store, "get_allowed_tables", return_value=[]), \
                patch.object(store, "get_compliance_profile", return_value={"mode": "standard"}), \
                patch("core.compliance.governed_query.execute_governed_query",
                      return_value=governed):
            return engine.check_alert_now(
                "a1", {"db_type": "azure_sql", "credentials": {}}, lang)

    def test_the_alert_message_is_translated(self):
        french, english = self._check("fr"), self._check("en")
        self.assertTrue(french["ok"] and english["ok"], (french, english))
        self.assertNotEqual(french["message"], english["message"])
        # Not enough on its own: the direction word and the figures are
        # interpolated, so they alone make the two unequal while the sentence
        # around them stays English. Assert on the sentence.
        self._assert_sentence_is(french["message"], "alert.triggered", "fr")
        self._assert_sentence_is(english["message"], "alert.triggered", "en")

    def test_the_untriggered_message_is_translated_too(self):
        # An alert that fires "all clear" is sent just as often as one that
        # does not, and it goes through a different message id.
        french, english = self._check("fr", triggered=False), self._check("en", triggered=False)
        self.assertFalse(french["triggered"])
        self.assertNotEqual(french["message"], english["message"])
        self._assert_sentence_is(french["message"], "alert.ok", "fr")
        self._assert_sentence_is(english["message"], "alert.ok", "en")

    def test_the_figures_are_written_in_the_readers_notation(self):
        # Not an f-string: "1,234.50" reads as one-and-a-bit to a French reader.
        message = self._check("fr")["message"]
        self.assertIn("1" + NNBSP + "234,50", message)
        self.assertIn("23,4" + NBSP + "%", message)
        self.assertNotIn("1,234.50", message)
        self.assertIn("1,234.50", self._check("en")["message"])

    def test_the_direction_is_translated_with_the_sentence(self):
        # The direction word is interpolated, so it can stay English while the
        # sentence around it turns French.
        french = self._check("fr")["message"]
        self.assertIn(MESSAGES["alert.direction.up"]["fr"], french)
        self.assertNotIn(MESSAGES["alert.direction.up"]["en"], french)

    def test_an_unknown_language_still_produces_a_message(self):
        result = self._check("de")
        self.assertTrue(result["ok"])
        self.assertEqual(result["message"], self._check("en")["message"])

    def test_the_catalogue_carries_both_languages(self):
        for msg_id in ("alert.triggered", "alert.ok",
                       "alert.direction.up", "alert.direction.down"):
            self.assertIn(msg_id, MESSAGES, msg_id)
            self.assertTrue(MESSAGES[msg_id]["fr"], msg_id)
            self.assertNotEqual(MESSAGES[msg_id]["en"], MESSAGES[msg_id]["fr"], msg_id)


class TestTheSchedulerPicksTheLanguageUp(unittest.TestCase):
    """
    ``run_due_alert_checks`` is the only caller that knows who the alert is
    for, so it is the only place the recipient's language can be read. These
    tests watch what it hands to ``check_alert_now``.
    """

    def _run_scheduler(self, *, recipient):
        import store

        import core.alert_engine as engine
        import core.pipeline_context as pipeline_context

        due = [{"id": "a1", "account_id": "acct", "user_id": 7, "status": "active"}]
        seen = {}

        def _capture(alert_id, db_cfg, lang=None):
            seen["lang"] = lang
            # Untriggered, so the scheduler stops before delivery — this test
            # is about the language handed in, not about notifying anyone.
            return {"ok": True, "triggered": False, "alert_id": alert_id}

        get_user = (patch.object(store, "get_user", side_effect=RuntimeError("no row"))
                    if recipient is RuntimeError
                    else patch.object(store, "get_user", return_value=recipient))
        with patch.object(engine, "list_alerts", return_value=due), \
                patch.object(engine, "_alert_due", return_value=True), \
                patch.object(engine, "check_alert_now", side_effect=_capture), \
                patch.object(pipeline_context, "get_client_db",
                             return_value={"db_type": "azure_sql", "credentials": {}}), \
                get_user:
            engine.run_due_alert_checks()
        return seen

    def test_the_recipients_language_reaches_the_builder(self):
        self.assertEqual(self._run_scheduler(recipient={"id": 7, "lang": "fr"})["lang"], "fr")

    def test_a_recipient_with_no_language_gets_english(self):
        for row in ({"id": 7}, {"id": 7, "lang": ""}, {"id": 7, "lang": None}):
            with self.subTest(row=row):
                self.assertEqual(self._run_scheduler(recipient=row)["lang"], "en")

    def test_a_missing_recipient_does_not_stop_the_alert(self):
        # A reader who cannot be read still needs to hear that a number moved.
        self.assertEqual(self._run_scheduler(recipient=None)["lang"], "en")

    def test_a_store_that_raises_does_not_stop_the_alert(self):
        self.assertEqual(self._run_scheduler(recipient=RuntimeError)["lang"], "en")


class TestTheFiguresAreWrittenForTheReader(unittest.TestCase):

    def test_french_notation_reaches_an_alert_message(self):
        from core.i18n import format_decimal, format_percent, t

        message = t("alert.triggered", lang="fr", metric="revenue",
                    current=format_decimal(1234.5, 2, lang="fr"),
                    direction=t("alert.direction.up", lang="fr"),
                    delta=format_percent(23.45, 1, lang="fr"),
                    baseline=format_decimal(1000.0, 2, lang="fr"))
        self.assertIn("1" + NNBSP + "234,50", message)
        self.assertIn("23,4" + NBSP + "%", message)
        self.assertNotIn("1,234.50", message)

    def test_english_notation_is_unchanged(self):
        from core.i18n import format_decimal, format_percent, t

        message = t("alert.triggered", lang="en", metric="revenue",
                    current=format_decimal(1234.5, 2, lang="en"),
                    direction=t("alert.direction.up", lang="en"),
                    delta=format_percent(23.45, 1, lang="en"),
                    baseline=format_decimal(1000.0, 2, lang="en"))
        self.assertIn("1,234.50", message)
        self.assertIn("23.4%", message)

    def test_no_placeholder_is_left_unfilled(self):
        from core.i18n import t

        for lang in ("en", "fr"):
            message = t("alert.triggered", lang=lang, metric="m", current="1",
                        direction="d", delta="2%", baseline="3")
            self.assertNotIn("{", message)


if __name__ == "__main__":
    unittest.main()
