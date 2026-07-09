import tempfile
import unittest
from pathlib import Path

from core.semantic_layer import build_semantic_layer_tables, find_semantic_field, table_allowed


class SemanticLayerTests(unittest.TestCase):
    def _tmpdir(self):
        return tempfile.TemporaryDirectory(dir=Path.cwd())

    def test_fqn_allowed_matches_bare_and_schema_names(self):
        allowed = {"CHATBOT_DB.HR.ATTENDANCE"}
        self.assertTrue(table_allowed("ATTENDANCE", allowed))
        self.assertTrue(table_allowed("HR.ATTENDANCE", allowed))
        self.assertTrue(table_allowed("CHATBOT_DB.HR.ATTENDANCE", allowed))
        self.assertFalse(table_allowed("FINANCE.PAYROLL", allowed))

    def test_builds_field_metadata_without_full_markdown(self):
        with self._tmpdir() as tmp:
            root = Path(tmp)
            (root / "Attendance_kb.md").write_text(
                "# CHATBOT_DB.HR.Attendance\n\n"
                "## Overview\nAttendance facts by employee.\n\n"
                "## Columns\n"
                "- `InStatus` (varchar): Employee punch-in status. values are 'Lete', 'OnTime'.\n"
                "- `EmployeeNo` (int): Employee identifier used for joins.\n\n"
                "## Key Metrics\n"
                "- **late count**: `InStatus` - Filter: `WHERE InStatus = 'Lete'`\n\n"
                "## Business Synonyms\n"
                "| Plain English | Column | Notes |\n"
                "|---|---|---|\n"
                "| late, punch-in status | `InStatus` | Used for attendance filters |\n",
                encoding="utf-8",
            )

            tables = build_semantic_layer_tables(
                kb_dir=str(root),
                allowed_tables={"CHATBOT_DB.HR.ATTENDANCE"},
            )

        self.assertEqual(1, len(tables))
        field = find_semantic_field(tables, "CHATBOT_DB.HR.ATTENDANCE", "InStatus")
        self.assertIsNotNone(field)
        _, meta = field
        self.assertIn("Employee punch-in status", meta["meaning"])
        # Business terms get their own structured "synonyms" list — they are
        # no longer stitched as a "Business terms: ..." fragment into the
        # free-text use_case (that stitching only fired for columns the
        # KB-build LLM happened to give a Business Synonyms row, which is
        # exactly why the fragment appeared on some fields but not others).
        self.assertNotIn("Business terms", meta["use_case"])
        self.assertEqual(meta["synonyms"], ["late", "punch-in status"])
        self.assertGreaterEqual(meta["confidence"], 85)

    def test_field_without_business_synonyms_row_has_empty_synonyms(self):
        with self._tmpdir() as tmp:
            root = Path(tmp)
            (root / "Attendance_kb.md").write_text(
                "# CHATBOT_DB.HR.Attendance\n\n"
                "## Columns\n"
                "- `EmployeeNo` (int): Employee identifier used for joins.\n",
                encoding="utf-8",
            )
            tables = build_semantic_layer_tables(kb_dir=str(root))
        _, meta = find_semantic_field(tables, "CHATBOT_DB.HR.ATTENDANCE", "EmployeeNo")
        self.assertEqual(meta["synonyms"], [])

    def test_field_override_synonyms_take_precedence_over_kb_parsed(self):
        with self._tmpdir() as tmp:
            root = Path(tmp)
            (root / "Attendance_kb.md").write_text(
                "# CHATBOT_DB.HR.Attendance\n\n"
                "## Columns\n"
                "- `InStatus` (varchar): Employee punch-in status.\n\n"
                "## Business Synonyms\n"
                "| Plain English | Column | Notes |\n"
                "|---|---|---|\n"
                "| late | `InStatus` |  |\n",
                encoding="utf-8",
            )
            overrides = {
                "tables": {
                    "CHATBOT_DB.HR.ATTENDANCE": {
                        "table_fqn": "CHATBOT_DB.HR.ATTENDANCE",
                        "fields": {
                            "INSTATUS": {
                                "column_name": "InStatus",
                                "meaning": "Employee punch-in status",
                                "synonyms": ["tardy", "on time status"],
                            }
                        },
                    }
                }
            }
            tables = build_semantic_layer_tables(kb_dir=str(root), field_overrides=overrides)
        _, meta = find_semantic_field(tables, "CHATBOT_DB.HR.ATTENDANCE", "InStatus")
        self.assertEqual(meta["synonyms"], ["tardy", "on time status"])

    def test_approved_feedback_synonyms_parsed_into_list(self):
        with self._tmpdir() as tmp:
            root = Path(tmp)
            (root / "Attendance_kb.md").write_text(
                "# CHATBOT_DB.HR.Attendance\n\n"
                "## Columns\n"
                "- `InStatus` (varchar): Employee punch-in status.\n",
                encoding="utf-8",
            )
            approved = {
                ("CHATBOT_DB.HR.ATTENDANCE", "INSTATUS"): {
                    "suggested_meaning": "Approved admin meaning",
                    "suggested_use_case": "Approved admin use case",
                    "suggested_synonyms": "tardy, on time status",
                }
            }
            tables = build_semantic_layer_tables(kb_dir=str(root), approved_feedback=approved)
        _, meta = find_semantic_field(tables, "CHATBOT_DB.HR.ATTENDANCE", "InStatus")
        self.assertEqual(meta["synonyms"], ["tardy", "on time status"])

    def test_approved_feedback_sets_confidence_to_100(self):
        with self._tmpdir() as tmp:
            root = Path(tmp)
            (root / "Attendance_kb.md").write_text(
                "# CHATBOT_DB.HR.Attendance\n\n"
                "## Columns\n"
                "- `InStatus` (varchar): Employee punch-in status.\n",
                encoding="utf-8",
            )
            approved = {
                ("CHATBOT_DB.HR.ATTENDANCE", "INSTATUS"): {
                    "suggested_meaning": "Approved admin meaning",
                    "suggested_use_case": "Approved admin use case",
                }
            }

            tables = build_semantic_layer_tables(
                kb_dir=str(root),
                approved_feedback=approved,
            )

        _, meta = find_semantic_field(tables, "CHATBOT_DB.HR.ATTENDANCE", "InStatus")
        self.assertEqual("Approved admin meaning", meta["meaning"])
        self.assertEqual("Approved admin use case", meta["use_case"])
        self.assertEqual(100, meta["confidence"])


if __name__ == "__main__":
    unittest.main()
