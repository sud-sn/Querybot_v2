import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.field_overrides import (
    load_field_overrides,
    parse_synonyms,
    save_field_override,
    table_overrides,
)
from core.semantic_kb_patch import (
    apply_approved_feedback,
    apply_field_overrides_to_content,
)
from core.semantic_layer import build_semantic_layer_tables, find_semantic_field


SAMPLE_KB = """\
# DB.HR.Employee

## Overview
Employee master data.

## Columns

- `Nationality` (nvarchar): [NEEDS CONTEXT] Business rule unknown.
- `EmployeeNo` (int): Employee identifier.

## Business Synonyms

| Plain English | Column | Notes |
|---|---|---|
| nationality | Nationality | Generated term. |
"""


class FieldOverrideStoreTests(unittest.TestCase):
    def test_save_load_and_replace_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "field_overrides.json"
            with patch("core.field_overrides.override_path", return_value=path):
                save_field_override(
                    account_id="acct",
                    table_fqn="DB.HR.EMPLOYEE",
                    schema_name="HR",
                    table_name="EMPLOYEE",
                    file_stem="Employee",
                    column_name="Nationality",
                    meaning="Employee country of citizenship.",
                    use_case="Use for country-based employee analysis.",
                    synonyms=["country", "citizenship"],
                    admin_note="Confirmed by HR.",
                )
                data = load_field_overrides("acct")
                fields = table_overrides(data, "HR.EMPLOYEE")
                self.assertEqual(
                    "Employee country of citizenship.",
                    fields["NATIONALITY"]["meaning"],
                )

                save_field_override(
                    account_id="acct",
                    table_fqn="DB.HR.EMPLOYEE",
                    schema_name="HR",
                    table_name="EMPLOYEE",
                    file_stem="Employee",
                    column_name="Nationality",
                    meaning="Updated meaning.",
                    use_case="",
                    synonyms=[],
                    admin_note="",
                )
                fields = table_overrides(load_field_overrides("acct"), "EMPLOYEE")
                self.assertEqual("Updated meaning.", fields["NATIONALITY"]["meaning"])
                self.assertEqual([], fields["NATIONALITY"]["synonyms"])

    def test_parse_synonyms_deduplicates_and_accepts_newlines(self):
        self.assertEqual(
            ["country", "citizenship", "home nation"],
            parse_synonyms("country, citizenship\nCountry, home nation"),
        )


class FieldOverrideApplicationTests(unittest.TestCase):
    def test_generated_content_is_deterministically_overridden(self):
        patched = apply_field_overrides_to_content(
            SAMPLE_KB,
            {
                "NATIONALITY": {
                    "column_name": "Nationality",
                    "meaning": "Employee country of citizenship.",
                    "use_case": "Use for workforce analysis by country.",
                    "synonyms": ["country", "citizenship"],
                }
            },
        )
        self.assertIn("Employee country of citizenship.", patched)
        self.assertNotIn("[NEEDS CONTEXT] Business rule unknown", patched)
        self.assertIn("country", patched)
        self.assertIn("citizenship", patched)

    def test_live_edit_persists_and_reembeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kb_file = root / "Employee_kb.md"
            override_file = root / "field_overrides.json"
            kb_file.write_text(SAMPLE_KB, encoding="utf-8")
            with patch("core.field_overrides.override_path", return_value=override_file), \
                 patch("core.knowledge.re_embed_file") as re_embed:
                ok, message = apply_approved_feedback(
                    account_id="acct",
                    kb_dir=str(root),
                    table_fqn="DB.HR.EMPLOYEE",
                    table_name="EMPLOYEE",
                    schema_name="HR",
                    column_name="Nationality",
                    approved_meaning="Employee country of citizenship.",
                    approved_use_case="Use for country-based employee analysis.",
                    approved_synonyms=["country", "citizenship"],
                    admin_note="Confirmed by HR.",
                    persist_override=True,
                    infer_synonyms=False,
                )
                fields = table_overrides(load_field_overrides("acct"), "DB.HR.EMPLOYEE")
            self.assertTrue(ok, message)
            re_embed.assert_called_once_with(str(root), "acct", "Employee_kb.md")
            self.assertEqual(
                "Employee country of citizenship.",
                fields["NATIONALITY"]["meaning"],
            )
            self.assertIn("citizenship", kb_file.read_text(encoding="utf-8"))

    def test_semantic_layer_prefers_persistent_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Employee_kb.md").write_text(SAMPLE_KB, encoding="utf-8")
            override_data = {
                "version": 1,
                "tables": {
                    "DB.HR.EMPLOYEE": {
                        "table_fqn": "DB.HR.EMPLOYEE",
                        "table_name": "EMPLOYEE",
                        "schema_name": "HR",
                        "file_stem": "Employee",
                        "fields": {
                            "NATIONALITY": {
                                "column_name": "Nationality",
                                "meaning": "Approved country definition.",
                                "use_case": "Group employees by country.",
                                "synonyms": ["country"],
                                "admin_note": "Verified.",
                            }
                        },
                    }
                },
            }
            tables = build_semantic_layer_tables(
                kb_dir=str(root),
                field_overrides=override_data,
            )
            _, field = find_semantic_field(
                tables,
                "DB.HR.EMPLOYEE",
                "Nationality",
            )
            self.assertEqual("Approved country definition.", field["meaning"])
            self.assertEqual(100, field["confidence"])
            self.assertTrue(field["approved"])
            self.assertFalse(field["needs_context"])


class AdminFieldEditorWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.template = (root / "admin" / "templates" / "client_kb.html").read_text(
            encoding="utf-8"
        )
        cls.routes = (root / "admin" / "routes.py").read_text(encoding="utf-8")
        cls.knowledge = (root / "core" / "knowledge.py").read_text(encoding="utf-8")

    def test_structured_field_save_route_is_wired(self):
        self.assertIn('/kb/fields/save"', self.routes)
        self.assertIn("async def kb_field_save", self.routes)
        self.assertIn("persist_override=True", self.routes)

    def test_responsive_field_editor_controls_exist(self):
        for token in [
            "kb-fields-layout",
            "field-drawer",
            "openFieldEditor",
            "fieldEditorDirty",
            "kbFieldSearch",
            "kbFieldFilter",
            "@media(max-width:600px)",
        ]:
            self.assertIn(token, self.template)

    def test_raw_files_are_an_advanced_workspace(self):
        self.assertIn("Advanced files", self.template)
        self.assertIn("Advanced editor.", self.template)
        self.assertIn('id="kb-editor"', self.template)

    def test_rebuild_fingerprint_includes_admin_inputs(self):
        for token in [
            '"field_overrides": _tbl_overrides',
            '"business_description": table_biz_desc',
            '"confirmed_joins": _confirmed_snippets',
            "_build_hash",
            "apply_field_overrides_to_content",
        ]:
            self.assertIn(token, self.knowledge)


if __name__ == "__main__":
    unittest.main()
