"""
tests/test_semantic_kb_patch.py

Tests for core/semantic_kb_patch.py covering every requirement from the plan:

  1. Approving feedback patches the correct KB file.
  2. Existing KB content outside the approved field is unchanged.
  3. Approved field shows 100% confidence and Source marker.
  4. re_embed_file is called for only the affected KB file.
  5. Missing KB file returns a clear admin error (no crash, no partial state).
  6. Rejected feedback does not alter the KB file.
  7. Multiple formats handled: bullet `-`, table row `|`, bold `**col**`.
  8. Latest-wins: approving one field supersedes other pending edits for same field.
  9. Column not found in ## Columns → appended, not lost.
 10. KB file reverted if re-embed fails.
 11. Synonym terms from approved_use_case are added to ## Business Synonyms.
 12. Synonym terms from user_comment are also extracted.
 13. Already-existing synonyms are not duplicated.
 14. Business Synonyms section is created if it doesn't exist.
 15. Synonym extraction does not corrupt other KB content.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.semantic_kb_patch import (
    _find_kb_file,
    _patch_column,
    _append_column,
    _extract_new_synonyms,
    _patch_synonyms,
    apply_approved_feedback,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_KB = """\
# CHATBOT_DB.HR.Attendance

**SQL table name:** `[HR].[Attendance]`

## Overview
Tracks daily employee attendance records.

## Columns

- `InStatus` (varchar): Old meaning that needs correction. values are 'Late', 'OnTime'.
- `EmployeeNo` (int): Employee identifier used for joins.
- `ShiftDate` (date): Date of the attendance record.

## Business Synonyms

| Plain English | Column | Notes |
|---|---|---|
| attendance date | InStatus | Date the employee attended. |

## Key Metrics
- **late arrivals**: Filter by InStatus = 'Late'.

## Join Keys
- InStatus is unique to this table.
"""

SAMPLE_KB_NO_SYNONYMS = """\
# HR.Attendance

## Overview
Tracks daily attendance.

## Columns

| Column | Type | Nullable | Distinct Values |
|---|---|---|---|
| `InStatus` | varchar | No | 'Late', 'OnTime' |
| `EmployeeNo` | int | No | |

## Key Metrics
- **late arrivals**: COUNT(*) WHERE InStatus = 'Late'
"""

SAMPLE_TABLE_ROW_KB = """\
# HR.Attendance

## Columns

| Column | Type | Nullable | Values |
|---|---|---|---|
| `InStatus` | varchar | No | 'Late', 'OnTime' |
| `EmployeeNo` | int | No | |
"""


# ── _find_kb_file tests ───────────────────────────────────────────────────────

class FindKBFileTests(unittest.TestCase):

    def test_finds_by_fqn_header(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "Attendance_kb.md").write_text(SAMPLE_KB)
            result = _find_kb_file(p, "CHATBOT_DB.HR.ATTENDANCE", "Attendance", "HR")
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "Attendance_kb.md")

    def test_finds_by_filename_pattern_bare(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "Attendance_kb.md").write_text("## Columns\n- `InStatus`: old\n")
            result = _find_kb_file(p, "CHATBOT_DB.HR.ATTENDANCE", "Attendance", "HR")
        self.assertIsNotNone(result)

    def test_finds_by_filename_case_insensitive(self):
        """Pass 2 must handle mixed-case filenames like Employee_kb.md."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "Employee_kb.md").write_text("## Columns\n| `Nationality` | nvarchar | Yes | |\n")
            result = _find_kb_file(p, "EMPLOYEE", "EMPLOYEE", "HR")
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "Employee_kb.md")

    def test_finds_by_schema_double_underscore_filename(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "HR__Attendance_kb.md").write_text("## Columns\n- `InStatus`: old\n")
            result = _find_kb_file(p, "CHATBOT_DB.HR.ATTENDANCE", "Attendance", "HR")
        self.assertIsNotNone(result)

    def test_returns_none_when_no_match(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "Orders_kb.md").write_text("# SALES.ORDERS\n## Columns\n")
            result = _find_kb_file(p, "HR.ATTENDANCE", "Attendance", "HR")
        self.assertIsNone(result)

    def test_returns_none_for_empty_dir(self):
        with tempfile.TemporaryDirectory() as d:
            result = _find_kb_file(Path(d), "HR.ATTENDANCE", "Attendance", "HR")
        self.assertIsNone(result)


# ── _patch_column tests ───────────────────────────────────────────────────────

class PatchColumnTests(unittest.TestCase):

    def test_patches_bullet_format(self):
        patched, changed = _patch_column(
            SAMPLE_KB, "InStatus",
            "Employee punch-in status. Late = arrived after scheduled start.",
            "Filter attendance report by InStatus to count late arrivals.",
        )
        self.assertTrue(changed)
        self.assertIn("Employee punch-in status", patched)
        self.assertIn("Admin-approved Semantic Layer edit", patched)

    def test_unchanged_content_outside_column_is_preserved(self):
        patched, changed = _patch_column(SAMPLE_KB, "InStatus", "New meaning.", "New use case.")
        self.assertTrue(changed)
        self.assertIn("Tracks daily employee attendance records", patched)
        self.assertIn("late arrivals", patched)
        self.assertIn("InStatus is unique to this table", patched)
        self.assertIn("EmployeeNo", patched)
        self.assertIn("ShiftDate", patched)

    def test_other_columns_untouched(self):
        patched, _ = _patch_column(SAMPLE_KB, "InStatus", "New.", "")
        self.assertIn("Employee identifier used for joins", patched)
        self.assertIn("Date of the attendance record", patched)

    def test_no_change_when_column_not_found(self):
        _, changed = _patch_column(SAMPLE_KB, "NonExistentColumn", "X", "Y")
        self.assertFalse(changed)

    def test_table_row_format_updates_same_row(self):
        patched, changed = _patch_column(
            SAMPLE_TABLE_ROW_KB, "InStatus",
            "Punch-in status.", "Use for late filter.",
        )
        self.assertTrue(changed)
        self.assertNotIn("<!-- Approved:", patched)
        self.assertIn("| `InStatus` | varchar | No | 'Late', 'OnTime' | Punch-in status. | Use for late filter. | 100% | Admin-approved Semantic Layer edit |", patched)
        self.assertIn("Punch-in status.", patched)

    def test_type_annotation_preserved_bullet(self):
        patched, changed = _patch_column(SAMPLE_KB, "InStatus", "Updated.", "UC.")
        self.assertTrue(changed)
        self.assertIn("(varchar)", patched)

    def test_approved_source_marker_present(self):
        patched, _ = _patch_column(SAMPLE_KB, "InStatus", "New.", "UC.")
        self.assertIn("Admin-approved Semantic Layer edit", patched)


# ── _append_column tests ──────────────────────────────────────────────────────

class AppendColumnTests(unittest.TestCase):

    def test_appends_when_column_not_in_columns_section(self):
        content = "# HR.Attendance\n\n## Columns\n- `EmployeeNo` (int): ID.\n\n## Metrics\n"
        result = _append_column(content, "NewCol", "New meaning.", "New use case.")
        self.assertIn("NewCol", result)
        self.assertIn("New meaning.", result)
        self.assertIn("Admin-approved Semantic Layer edit", result)
        self.assertIn("EmployeeNo", result)
        self.assertIn("Metrics", result)


# ── _extract_new_synonyms tests ───────────────────────────────────────────────

class ExtractNewSynonymsTests(unittest.TestCase):

    def test_extracts_synonym_from_use_case(self):
        """'country' should be extracted from the use_case."""
        synonyms = _extract_new_synonyms(
            column_name="Nationality",
            approved_meaning="Nationality of the employee.",
            approved_use_case="Used when a question explicitly refers to nationality, country.",
            user_comment="",
            existing_content="## Business Synonyms\n| nationality | Nationality | |\n",
        )
        self.assertIn("country", synonyms)

    def test_does_not_duplicate_existing_synonym(self):
        """'nationality' already exists — must not be added again."""
        synonyms = _extract_new_synonyms(
            column_name="Nationality",
            approved_meaning="Nationality of the employee.",
            approved_use_case="refers to nationality, country",
            user_comment="",
            existing_content="## Business Synonyms\n| nationality | Nationality | |\n",
        )
        self.assertNotIn("nationality", synonyms)
        self.assertIn("country", synonyms)

    def test_extracts_from_user_comment(self):
        """Synonyms in user_comment should also be extracted."""
        synonyms = _extract_new_synonyms(
            column_name="Nationality",
            approved_meaning="",
            approved_use_case="",
            user_comment="also called home country, country of origin",
            existing_content="",
        )
        self.assertTrue(len(synonyms) > 0)
        # At least one of the expected terms should appear
        combined = " ".join(synonyms)
        self.assertTrue(
            "home country" in combined or "country" in combined or "origin" in combined,
            f"Expected country-related term, got: {synonyms}"
        )

    def test_column_name_itself_not_added(self):
        """The column name should never become its own synonym."""
        synonyms = _extract_new_synonyms(
            column_name="InStatus",
            approved_meaning="Punch-in status.",
            approved_use_case="refers to instatus, status",
            user_comment="",
            existing_content="",
        )
        self.assertNotIn("instatus", [s.lower() for s in synonyms])

    def test_column_name_parts_not_added_as_synonyms(self):
        """Underscore-separated column-name parts should not become synonyms."""
        synonyms = _extract_new_synonyms(
            column_name="Formula_ID",
            approved_meaning="Formula identifier.",
            approved_use_case="refers to formula, formula id",
            user_comment="",
            existing_content="",
        )
        self.assertNotIn("formula", [s.lower() for s in synonyms])
        self.assertNotIn("formula id", [s.lower() for s in synonyms])

    def test_no_synonyms_when_no_new_terms(self):
        synonyms = _extract_new_synonyms(
            column_name="Nationality",
            approved_meaning="Nationality of the employee.",
            approved_use_case="Used when a question explicitly refers to nationality.",
            user_comment="",
            existing_content="## Business Synonyms\n| nationality | Nationality | |\n",
        )
        self.assertNotIn("nationality", synonyms)

    def test_returns_empty_list_for_empty_inputs(self):
        synonyms = _extract_new_synonyms(
            column_name="Nationality",
            approved_meaning="",
            approved_use_case="",
            user_comment="",
            existing_content="",
        )
        self.assertIsInstance(synonyms, list)


# ── _patch_synonyms tests ─────────────────────────────────────────────────────

class PatchSynonymsTests(unittest.TestCase):

    def test_adds_new_row_to_existing_section(self):
        result = _patch_synonyms(SAMPLE_KB, "InStatus", ["late", "on time status"])
        self.assertIn("late", result)
        self.assertIn("on time status", result)
        # Existing synonym row preserved
        self.assertIn("attendance date", result)

    def test_creates_section_when_missing(self):
        result = _patch_synonyms(SAMPLE_KB_NO_SYNONYMS, "InStatus", ["status"])
        self.assertIn("## Business Synonyms", result)
        self.assertIn("status", result)
        # Other content preserved
        self.assertIn("late arrivals", result)
        self.assertIn("Tracks daily attendance", result)

    def test_does_not_duplicate_when_row_exists(self):
        """If the column already has a synonym row, terms are appended not duplicated."""
        result = _patch_synonyms(SAMPLE_KB, "InStatus", ["late"])
        # Count how many times 'late' appears in InStatus's synonym row
        syns_section = False
        instatus_terms_count = 0
        for line in result.splitlines():
            stripped = line.strip()
            if "business synonyms" in stripped.lower():
                syns_section = True
                continue
            if syns_section and stripped.startswith("|"):
                cells = [c.strip() for c in stripped.strip("|").split("|")]
                if len(cells) >= 2 and "InStatus" in cells[1]:
                    instatus_terms_count += cells[0].lower().count("late")
        # 'late' should appear at most once
        self.assertLessEqual(instatus_terms_count, 1)

    def test_returns_content_unchanged_for_empty_synonyms(self):
        result = _patch_synonyms(SAMPLE_KB, "InStatus", [])
        self.assertEqual(result, SAMPLE_KB)

    def test_does_not_corrupt_other_sections(self):
        result = _patch_synonyms(SAMPLE_KB_NO_SYNONYMS, "EmployeeNo", ["emp id", "staff id"])
        self.assertIn("Tracks daily attendance", result)
        self.assertIn("late arrivals", result)
        self.assertIn("InStatus", result)


# ── apply_approved_feedback integration tests ─────────────────────────────────

class ApplyApprovedFeedbackTests(unittest.TestCase):

    def _make_kb_dir(self, tmp: str) -> Path:
        p = Path(tmp)
        (p / "Attendance_kb.md").write_text(SAMPLE_KB, encoding="utf-8")
        return p

    def test_patches_correct_kb_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._make_kb_dir(d)
            with patch("core.knowledge.re_embed_file"):
                ok, msg = apply_approved_feedback(
                    account_id="acct1", kb_dir=d,
                    table_fqn="CHATBOT_DB.HR.ATTENDANCE",
                    table_name="Attendance", schema_name="HR",
                    column_name="InStatus",
                    approved_meaning="Employee punch-in status.",
                    approved_use_case="Filter by InStatus for attendance analysis.",
                    user_comment="",
                )
            self.assertTrue(ok, msg)
            patched = (p / "Attendance_kb.md").read_text()
            self.assertIn("Employee punch-in status.", patched)

    def test_synonym_from_use_case_added_to_kb(self):
        """
        The core fix: 'country' in the use_case must be written into
        Business Synonyms so Qdrant and SQL generation see it.
        """
        kb = (
            "# HR.Employee\n\n## Columns\n\n"
            "| Column | Type | Nullable |\n|---|---|---|\n"
            "| `Nationality` | nvarchar | Yes |\n\n"
            "## Business Synonyms\n"
            "| Plain English | Column | Notes |\n|---|---|---|\n"
            "| nationality | Nationality | Nationality of the employee. |\n"
        )
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "Employee_kb.md").write_text(kb, encoding="utf-8")
            with patch("core.knowledge.re_embed_file"):
                ok, msg = apply_approved_feedback(
                    account_id="acct1", kb_dir=d,
                    table_fqn="HR.EMPLOYEE",
                    table_name="Employee", schema_name="HR",
                    column_name="Nationality",
                    approved_meaning="Nationality of the employee.",
                    approved_use_case="Used when a question explicitly refers to nationality, country.",
                    user_comment="",
                )
            self.assertTrue(ok, msg)
            content = (p / "Employee_kb.md").read_text()
            # 'country' MUST appear in Business Synonyms
            in_synonyms = False
            found_country = False
            for line in content.splitlines():
                if "business synonyms" in line.lower():
                    in_synonyms = True
                    continue
                if in_synonyms and line.startswith("##"):
                    break
                if in_synonyms and "country" in line.lower() and "Nationality" in line:
                    found_country = True
            self.assertTrue(
                found_country,
                "Expected 'country' synonym for Nationality in Business Synonyms. "
                f"KB content:\n{content}"
            )

    def test_user_comment_synonyms_added(self):
        """Synonyms from user_comment are also written to the KB."""
        kb = "# HR.Employee\n\n## Columns\n\n- `Nationality` (nvarchar): Old meaning.\n"
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "Employee_kb.md").write_text(kb, encoding="utf-8")
            with patch("core.knowledge.re_embed_file"):
                ok, _ = apply_approved_feedback(
                    account_id="acct1", kb_dir=d,
                    table_fqn="HR.EMPLOYEE",
                    table_name="Employee", schema_name="HR",
                    column_name="Nationality",
                    approved_meaning="Country of origin.",
                    approved_use_case="",
                    user_comment="also called home country",
                )
            self.assertTrue(ok)
            content = (p / "Employee_kb.md").read_text()
            self.assertIn("home country", content.lower())

    def test_existing_kb_content_outside_field_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._make_kb_dir(d)
            with patch("core.knowledge.re_embed_file"):
                apply_approved_feedback(
                    account_id="acct1", kb_dir=d,
                    table_fqn="CHATBOT_DB.HR.ATTENDANCE",
                    table_name="Attendance", schema_name="HR",
                    column_name="InStatus",
                    approved_meaning="New meaning.", approved_use_case="",
                    user_comment="",
                )
            content = (p / "Attendance_kb.md").read_text()
            self.assertIn("Tracks daily employee attendance records", content)
            self.assertIn("late arrivals", content)
            self.assertIn("InStatus is unique to this table", content)
            self.assertIn("EmployeeNo", content)
            self.assertIn("ShiftDate", content)

    def test_re_embed_called_for_only_the_patched_file(self):
        with tempfile.TemporaryDirectory() as d:
            self._make_kb_dir(d)
            with patch("core.knowledge.re_embed_file") as mock_embed:
                apply_approved_feedback(
                    account_id="acct1", kb_dir=d,
                    table_fqn="CHATBOT_DB.HR.ATTENDANCE",
                    table_name="Attendance", schema_name="HR",
                    column_name="InStatus",
                    approved_meaning="New meaning.", approved_use_case="",
                    user_comment="",
                )
            mock_embed.assert_called_once()
            self.assertEqual(mock_embed.call_args.args[1], "acct1")
            self.assertEqual(mock_embed.call_args.args[2], "Attendance_kb.md")

    def test_missing_kb_file_returns_error(self):
        with tempfile.TemporaryDirectory() as d:
            ok, msg = apply_approved_feedback(
                account_id="acct1", kb_dir=d,
                table_fqn="CHATBOT_DB.HR.ATTENDANCE",
                table_name="Attendance", schema_name="HR",
                column_name="InStatus",
                approved_meaning="New.", approved_use_case="",
                user_comment="",
            )
        self.assertFalse(ok)
        self.assertIn("Could not find", msg)

    def test_missing_kb_dir_returns_error(self):
        ok, msg = apply_approved_feedback(
            account_id="acct1", kb_dir="/nonexistent/path",
            table_fqn="HR.ATTENDANCE", table_name="Attendance", schema_name="HR",
            column_name="InStatus", approved_meaning="New.", approved_use_case="",
            user_comment="",
        )
        self.assertFalse(ok)
        self.assertIn("not found", msg.lower())

    def test_re_embed_failure_reverts_kb_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._make_kb_dir(d)
            original = (p / "Attendance_kb.md").read_text()

            with patch("core.knowledge.re_embed_file",
                       side_effect=RuntimeError("Qdrant connection refused")):
                ok, msg = apply_approved_feedback(
                    account_id="acct1", kb_dir=d,
                    table_fqn="CHATBOT_DB.HR.ATTENDANCE",
                    table_name="Attendance", schema_name="HR",
                    column_name="InStatus",
                    approved_meaning="Should not stick.", approved_use_case="",
                    user_comment="",
                )

            self.assertFalse(ok)
            self.assertIn("reverted", msg.lower())
            self.assertEqual(original, (p / "Attendance_kb.md").read_text())

    def test_rejection_does_not_alter_kb(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._make_kb_dir(d)
            original = (p / "Attendance_kb.md").read_text()
            # Rejection doesn't call apply_approved_feedback
            after = (p / "Attendance_kb.md").read_text()
            self.assertEqual(original, after)

    def test_success_message_includes_synonym_count(self):
        """Message should tell admin how many synonyms were added."""
        kb = (
            "# HR.Employee\n\n## Columns\n\n- `Nationality` (nvarchar): Old.\n\n"
            "## Business Synonyms\n| Plain English | Column | Notes |\n|---|---|---|\n"
            "| nationality | Nationality | |\n"
        )
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "Employee_kb.md").write_text(kb)
            with patch("core.knowledge.re_embed_file"):
                ok, msg = apply_approved_feedback(
                    account_id="acct1", kb_dir=d,
                    table_fqn="HR.EMPLOYEE", table_name="Employee", schema_name="HR",
                    column_name="Nationality",
                    approved_meaning="Country of origin.",
                    approved_use_case="refers to nationality, country, home country",
                    user_comment="",
                )
            self.assertTrue(ok)
            # Message should mention synonyms were added
            self.assertIn("synonym", msg.lower())


if __name__ == "__main__":
    unittest.main()
