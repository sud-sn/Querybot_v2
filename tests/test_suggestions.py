import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.examples import _table_name_for_query_file
from core.suggestions import build_suggestion_cache, get_suggestions


class SuggestionTrustTests(unittest.TestCase):
    def _tmpdir(self):
        return tempfile.TemporaryDirectory(dir=Path.cwd())

    def _kb_dir(self, tmp: str) -> Path:
        kb = Path(tmp) / "kb"
        kb.mkdir()
        (kb / "_schema.json").write_text(
            json.dumps({"CHATBOT_DB.HR.ATTENDANCE": {"columns": []}}),
            encoding="utf-8",
        )
        (kb / "Attendance_kb.md").write_text(
            "# CHATBOT_DB.HR.Attendance\n\n**SQL table name:** [HR].[Attendance]\n",
            encoding="utf-8",
        )
        (kb / "Attendance_queries.md").write_text(
            "Q: What is our total in status?\n"
            "SQL: SELECT COUNT(*) FROM [HR].[Attendance];\n",
            encoding="utf-8",
        )
        return kb

    def test_raw_stage2_cache_is_not_surfaced(self):
        with self._tmpdir() as tmp:
            kb = self._kb_dir(tmp)
            build_suggestion_cache(str(kb))

            with patch("store.get_validated_examples", return_value=[]), \
                 patch("store.list_metrics", return_value=[]):
                suggestions = get_suggestions(
                    "acct",
                    str(kb),
                    allowed_tables=None,
                    n=6,
                    schema_dir=str(kb),
                )

            self.assertEqual([], suggestions)

    def test_validated_example_gets_fqn_hint_from_cache(self):
        with self._tmpdir() as tmp:
            kb = self._kb_dir(tmp)
            build_suggestion_cache(str(kb))
            examples = [{
                "question": "What is our total in status?",
                "sql_query": "SELECT COUNT(*) FROM [HR].[Attendance];",
                "table_name": "ATTENDANCE",
            }]

            with patch("store.get_validated_examples", return_value=examples), \
                 patch("store.list_metrics", return_value=[]):
                suggestions = get_suggestions(
                    "acct",
                    str(kb),
                    allowed_tables={"CHATBOT_DB.HR.ATTENDANCE"},
                    n=6,
                    schema_dir=str(kb),
                )

            self.assertEqual(1, len(suggestions))
            self.assertEqual("CHATBOT_DB.HR.ATTENDANCE", suggestions[0]["fqn"])

    def test_validation_stores_fqn_from_sibling_kb_header(self):
        with self._tmpdir() as tmp:
            kb = self._kb_dir(tmp)
            self.assertEqual(
                "CHATBOT_DB.HR.ATTENDANCE",
                _table_name_for_query_file(kb / "Attendance_queries.md"),
            )


if __name__ == "__main__":
    unittest.main()
