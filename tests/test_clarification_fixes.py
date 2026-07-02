"""
Tests for the hardened clarification loop.

Run:  cd querybot && python -m unittest tests.test_clarification_fixes
"""

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Point the DB at a temp location before importing anything that touches it.
_tmpdir = tempfile.mkdtemp(prefix="querybot_tests_")
os.environ["DB_PATH"] = str(Path(_tmpdir) / "test_querybot.db")
os.environ["QUERYBOT_KEY_FILE"] = str(Path(_tmpdir) / "test_key")


from core.clarification import (
    _parse_ambiguity_json,
    _pick_option,
    combine_with_clarification,
    extract_original_question,
    mark_recently_expired,
    was_recently_expired,
    acknowledge_recently_expired,
    resolve_option_text,
)
from core.webhook_dedup import (
    is_duplicate_event,
    remember_event,
    _reset_for_tests,
)
from gateway.base import PlatformEvent


# ──────────────────────────────────────────────────────────────────────────────
# Fix #5 — tolerant JSON parsing
# ──────────────────────────────────────────────────────────────────────────────
class TolerantJSONParseTests(unittest.TestCase):

    def test_plain_object(self):
        self.assertEqual(
            _parse_ambiguity_json('{"status":"CLEAR"}'),
            {"status": "CLEAR"},
        )

    def test_json_fenced(self):
        raw = '```json\n{"status":"AMBIGUOUS","question":"which?"}\n```'
        self.assertEqual(
            _parse_ambiguity_json(raw),
            {"status": "AMBIGUOUS", "question": "which?"},
        )

    def test_unlabelled_fence(self):
        raw = '```\n{"status":"CLEAR"}\n```'
        self.assertEqual(_parse_ambiguity_json(raw), {"status": "CLEAR"})

    def test_preamble(self):
        raw = 'Here is the result:\n{"status":"AMBIGUOUS","question":"x"}\nThanks!'
        parsed = _parse_ambiguity_json(raw)
        self.assertEqual(parsed["status"], "AMBIGUOUS")

    def test_braces_in_string_dont_confuse_balance(self):
        # The question field contains a brace character — parser must not
        # bail out early on it.
        raw = '{"status":"AMBIGUOUS","question":"metric {}?","option_ids":["t1","t2"]}'
        parsed = _parse_ambiguity_json(raw)
        self.assertEqual(parsed["option_ids"], ["t1", "t2"])

    def test_malformed_returns_none(self):
        self.assertIsNone(_parse_ambiguity_json("this is not JSON"))
        self.assertIsNone(_parse_ambiguity_json(""))
        self.assertIsNone(_parse_ambiguity_json("{unclosed"))


# ──────────────────────────────────────────────────────────────────────────────
# Fix #2 — dispatch forwards an explicit option id; combine uses it verbatim
# ──────────────────────────────────────────────────────────────────────────────
class OptionPickPassThroughTests(unittest.TestCase):

    def _opts(self):
        return [
            {
                "id": "opt1",
                "label": "Late days",
                "value": "late_days",
                "expression": "SUM(late_days)",
            },
            {
                "id": "opt2",
                "label": "Late/absent days",
                "value": "late_absent_days",
                "expression": "SUM(late_days) + SUM(absent_days)",
            },
        ]

    def test_explicit_id_wins_even_when_text_is_ambiguous(self):
        # User's reply text "late" would substring-match BOTH options.
        # Without Fix #2 this is non-deterministic. With Fix #2, the
        # explicit id from dispatch pins the choice.
        chosen = _pick_option(self._opts(), "late", selected_option_id="opt2")
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen["id"], "opt2")
        self.assertEqual(chosen["expression"], "SUM(late_days) + SUM(absent_days)")

    def test_unknown_id_does_not_silently_fall_back(self):
        # If dispatch forwarded a bad id, combine must NOT silently try
        # text matching — that's exactly how dispatch/combine drifted apart.
        chosen = _pick_option(self._opts(), "late days", selected_option_id="opt99")
        self.assertIsNone(chosen)

    def test_no_explicit_id_falls_through_to_text(self):
        chosen = _pick_option(self._opts(), "late/absent days", selected_option_id=None)
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen["id"], "opt2")

    def test_combine_respects_explicit_option_id(self):
        meta = {"source": "llm_menu", "options": self._opts()}
        _, injection = combine_with_clarification(
            "count late employees",
            "late",  # ambiguous by substring
            meta,
            selected_option_id="opt1",
        )
        self.assertIn("SUM(late_days)", injection)
        self.assertNotIn("absent_days", injection)


# ──────────────────────────────────────────────────────────────────────────────
# Fix #7 — expiry grace trail
# ──────────────────────────────────────────────────────────────────────────────
class ExpiryTrailTests(unittest.TestCase):

    def test_mark_and_check(self):
        mark_recently_expired("acct1", "user1")
        self.assertTrue(was_recently_expired("acct1", "user1"))
        self.assertFalse(was_recently_expired("acct1", "user2"))
        self.assertFalse(was_recently_expired("acct2", "user1"))

    def test_acknowledge_clears(self):
        mark_recently_expired("acct1", "user1")
        acknowledge_recently_expired("acct1", "user1")
        self.assertFalse(was_recently_expired("acct1", "user1"))


# ──────────────────────────────────────────────────────────────────────────────
# Fix #8 — webhook idempotency
# ──────────────────────────────────────────────────────────────────────────────
class WebhookDedupTests(unittest.TestCase):

    def setUp(self):
        _reset_for_tests()

    def _make_zoom_event(self, message_id: str, text: str = "hello"):
        return PlatformEvent(
            account_id="acct1",
            user_id="u1",
            channel_id="c1",
            text=text,
            platform="zoom",
            raw={"payload": {"message_id": message_id}},
        )

    def _make_slack_event(self, event_id: str, text: str = "hello"):
        return PlatformEvent(
            account_id="acct1",
            user_id="u1",
            channel_id="c1",
            text=text,
            platform="slack",
            raw={"event_id": event_id, "event": {"ts": "1234.5"}},
        )

    def test_first_delivery_not_duplicate(self):
        ev = self._make_zoom_event("m1")
        self.assertFalse(is_duplicate_event(ev))
        remember_event(ev)

    def test_second_delivery_is_duplicate(self):
        ev = self._make_zoom_event("m1")
        self.assertFalse(is_duplicate_event(ev))
        remember_event(ev)
        ev2 = self._make_zoom_event("m1")  # same message id
        self.assertTrue(is_duplicate_event(ev2))

    def test_different_message_ids_not_duplicate(self):
        ev1 = self._make_zoom_event("m1")
        remember_event(ev1)
        ev2 = self._make_zoom_event("m2")
        self.assertFalse(is_duplicate_event(ev2))

    def test_slack_uses_event_id(self):
        ev1 = self._make_slack_event("E111")
        remember_event(ev1)
        ev2 = self._make_slack_event("E111")
        self.assertTrue(is_duplicate_event(ev2))
        ev3 = self._make_slack_event("E222")
        self.assertFalse(is_duplicate_event(ev3))

    def test_cross_platform_keys_dont_collide(self):
        zoom = self._make_zoom_event("X1")
        slack = self._make_slack_event("X1")
        remember_event(zoom)
        # Same raw identifier but different platform → not a duplicate
        self.assertFalse(is_duplicate_event(slack))

    def test_content_hash_fallback(self):
        # Event with no message_id at all — falls back to content hash
        ev = PlatformEvent(
            account_id="acct1", user_id="u1", channel_id="c1",
            text="same text", platform="zoom", raw={"payload": {}},
        )
        self.assertFalse(is_duplicate_event(ev))
        remember_event(ev)
        ev2 = PlatformEvent(
            account_id="acct1", user_id="u1", channel_id="c1",
            text="same text", platform="zoom", raw={"payload": {}},
        )
        self.assertTrue(is_duplicate_event(ev2))


# ──────────────────────────────────────────────────────────────────────────────
# Fix #1 — LLM constrained-menu ambiguity check
# ──────────────────────────────────────────────────────────────────────────────
class ConstrainedMenuAmbiguityTests(unittest.TestCase):
    """
    End-to-end: the LLM returns option_ids from a menu built from the
    real glossary. The returned options carry real term_ids and expressions.
    """

    def test_llm_menu_returns_real_option_ids(self):
        from core import clarification as clar

        async def _run():
            fake_terms = [
                {
                    "id": 10, "term": "absenteeism", "kind": "metric",
                    "aliases": "absent rate",
                    "definition": "Attendance-status absence count",
                    "canonical_expression": "SUM(CASE WHEN status='Absent' THEN 1 ELSE 0 END)",
                    "tables_involved": "",
                },
                {
                    "id": 11, "term": "attrition", "kind": "metric",
                    "aliases": "turnover",
                    "definition": "Headcount lost in period",
                    "canonical_expression": "COUNT(DISTINCT employee_id) FILTER(WHERE exit_date IS NOT NULL)",
                    "tables_involved": "",
                },
            ]
            # Tell the model to pick both options
            model_reply = (
                '{"status":"AMBIGUOUS",'
                '"question":"Absenteeism or attrition?",'
                '"option_ids":["t10","t11"]}'
            )
            with patch("store.list_terms", return_value=fake_terms), \
                 patch("core.llm.llm_complete",
                       return_value=(model_reply, 100, 50)):
                return await clar._llm_ambiguity_check_constrained(
                    account_id="acct",
                    question="show me absenteeism",
                    context="",
                    provider="test", model="test", api_key="",
                    extra_kwargs={},
                )

        is_amb, q, options = asyncio.run(_run())
        self.assertTrue(is_amb)
        self.assertEqual(len(options), 2)
        # The critical assertion — options carry real term_ids and expressions,
        # fixing the original "options always empty" bug.
        self.assertEqual(options[0]["_term_id"], 10)
        self.assertEqual(options[1]["_term_id"], 11)
        self.assertIn("CASE WHEN status", options[0]["expression"])
        self.assertIn("COUNT(DISTINCT employee_id)", options[1]["expression"])

    def test_llm_returns_clear_short_circuits(self):
        from core import clarification as clar

        async def _run():
            fake_terms = [
                {"id": 1, "term": "a", "kind": "metric", "aliases": "", "definition": "A", "canonical_expression": "1", "tables_involved": ""},
                {"id": 2, "term": "b", "kind": "metric", "aliases": "", "definition": "B", "canonical_expression": "2", "tables_involved": ""},
            ]
            with patch("store.list_terms", return_value=fake_terms), \
                 patch("core.llm.llm_complete",
                       return_value=('{"status":"CLEAR"}', 50, 10)):
                return await clar._llm_ambiguity_check_constrained(
                    account_id="acct", question="whatever", context="",
                    provider="test", model="test", api_key="", extra_kwargs={},
                )

        is_amb, q, options = asyncio.run(_run())
        self.assertFalse(is_amb)
        self.assertEqual(options, [])

    def test_invented_option_ids_are_dropped(self):
        from core import clarification as clar

        async def _run():
            fake_terms = [
                {"id": 1, "term": "a", "kind": "metric", "aliases": "", "definition": "", "canonical_expression": "x", "tables_involved": ""},
                {"id": 2, "term": "b", "kind": "metric", "aliases": "", "definition": "", "canonical_expression": "y", "tables_involved": ""},
            ]
            # Model invents "t999" which is NOT in the menu
            reply = '{"status":"AMBIGUOUS","question":"?","option_ids":["t1","t999"]}'
            with patch("store.list_terms", return_value=fake_terms), \
                 patch("core.llm.llm_complete", return_value=(reply, 50, 10)):
                return await clar._llm_ambiguity_check_constrained(
                    account_id="acct", question="q", context="",
                    provider="test", model="test", api_key="", extra_kwargs={},
                )

        is_amb, q, options = asyncio.run(_run())
        # Only one valid id survived → not enough for ambiguity
        self.assertFalse(is_amb)

    def test_sparse_glossary_falls_back_to_plain_classifier(self):
        from core import clarification as clar

        async def _run():
            # Only one term — menu can't be built
            fake_terms = [{"id": 1, "term": "a", "kind": "metric", "aliases": "", "definition": "", "canonical_expression": "x", "tables_involved": ""}]
            with patch("store.list_terms", return_value=fake_terms), \
                 patch("core.llm.llm_complete", return_value=('{"status":"CLEAR"}', 10, 5)):
                return await clar._llm_ambiguity_check_constrained(
                    account_id="acct", question="q", context="",
                    provider="test", model="test", api_key="", extra_kwargs={},
                )

        is_amb, q, options = asyncio.run(_run())
        self.assertFalse(is_amb)
        self.assertEqual(options, [])


# ──────────────────────────────────────────────────────────────────────────────
# Existing guards still hold (regression suite)
# ──────────────────────────────────────────────────────────────────────────────
class RegressionTests(unittest.TestCase):

    def test_resolve_option_text_word_overlap(self):
        options = [
            {"id": "o1", "label": "Absenteeism based on attendance status", "value": "attendance_status"},
            {"id": "o2", "label": "Absenteeism by attrition count", "value": "attrition_count"},
        ]
        resolved = resolve_option_text(options, "yes absenteeism by attrition count")
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["id"], "o2")

    def test_free_text_clarification_keeps_original_question(self):
        combined, injection = combine_with_clarification(
            "find unique employee counts per department who are lete",
            "attendance status called Late",
            {"source": "llm", "question": "Which status?", "options": []},
        )
        self.assertIn("find unique employee counts", combined.lower())
        self.assertIn("clarification for the same request", combined.lower())
        self.assertIn("attendance status called late", combined.lower())
        self.assertEqual(injection, "")


# ──────────────────────────────────────────────────────────────────────────────
# extract_original_question — strip the clarification wrapper before feeding
# text into deterministic semantic-field matchers
# ──────────────────────────────────────────────────────────────────────────────
class ExtractOriginalQuestionTests(unittest.TestCase):

    def test_round_trips_through_combine_with_clarification(self):
        original = "what is the number of days present between payment by each customer"
        combined, _ = combine_with_clarification(original, "customer id, customer key")
        self.assertEqual(extract_original_question(combined), original)

    def test_plain_question_passes_through_unchanged(self):
        q = "show revenue by customer"
        self.assertEqual(extract_original_question(q), q)

    def test_empty_string_passes_through(self):
        self.assertEqual(extract_original_question(""), "")

    def test_clarification_chip_text_does_not_pollute_semantic_field_plan(self):
        # Regression: a real production clarification reply used a business
        # term's raw synonym-list chip label ("Synonyms: customer id,
        # customer key") as its value. combine_with_clarification appends
        # this verbatim into the "combined" question text, and feeding that
        # straight into build_semantic_field_plan spuriously matched an
        # extra "customer id" field (CUS_ID) purely from the chip label —
        # UI metadata, not something the user actually asked for.
        from core.semantic_planner import build_semantic_field_plan

        table_columns = {
            "EMDW_DMART.CUS_ORD_IVC_FCT": {"CUS_DMS_KEY": "bigint", "PAY_DT_DMS_KEY": "bigint"},
            "EMDW_DMART.CUS_DMS": {"CUS_DMS_KEY": "bigint", "CUS_NM": "varchar", "CUS_ID": "varchar"},
            "EMDW_DMART.DT_DMS": {"DT_DMS_KEY": "bigint", "DAY": "int", "DT_DSC": "varchar"},
        }
        original = "what is the number of days present between payment by each customer"
        combined, _ = combine_with_clarification(original, "Synonyms: customer id, customer key")

        polluted_plan = build_semantic_field_plan(combined, table_columns)
        clean_plan = build_semantic_field_plan(extract_original_question(combined), table_columns)

        polluted_columns = {f["column"] for f in polluted_plan.get("fields", [])}
        clean_columns = {f["column"] for f in clean_plan.get("fields", [])}

        self.assertIn("CUS_ID", polluted_columns, "test fixture must reproduce the pollution")
        self.assertNotIn("CUS_ID", clean_columns)

    def test_query_pipeline_wires_extraction_into_both_field_plan_builders(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "core" / "query_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("extract_original_question", src)
        self.assertIn("_semantic_plan_question = extract_original_question(question)", src)
        self.assertIn("build_semantic_field_plan(\n            _semantic_plan_question,", src)
        self.assertIn("question=_semantic_plan_question,", src)


if __name__ == "__main__":
    unittest.main()
