"""
tests/test_grain_review.py

Fact-table grain review from the Model Health page:
  1. patch_grain_approval updates JSON + YAML, matches by fqn/qualified_name/
     bare name, rejects unknown tables and empty grain
  2. Approved grain survives preserve_approvals across a KB rebuild
  3. get_model_health surfaces grain/grain_status per table
  4. Route + template wiring markers
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.semantic_model import (
    MODEL_JSON, MODEL_YAML, get_model_health,
    patch_grain_approval, preserve_approvals,
)


def _write_model(kb_dir: Path, model: dict) -> None:
    (kb_dir / MODEL_JSON).write_text(json.dumps(model), encoding="utf-8")


def _model():
    return {
        "tables": [
            {
                "table": "FNN_FCT",
                "qualified_name": "EMDW_DMART.FNN_FCT",
                "schema": "EMDW_DMART",
                "type": "fact",
                "grain": "needs_admin_context",
                "grain_status": "needs_review",
                "grain_confidence": 0,
                "fields": [], "measures": [], "dimensions": [], "date_roles": [],
            },
            {
                "table": "CUS_DMS",
                "qualified_name": "EMDW_DMART.CUS_DMS",
                "schema": "EMDW_DMART",
                "type": "dimension",
                "grain": "one row per lookup member",
                "grain_status": "generated",
                "fields": [], "measures": [], "dimensions": [], "date_roles": [],
            },
        ],
        "relationships": [],
        "date_roles": [],
    }


class PatchGrainApprovalTests(unittest.TestCase):
    def setUp(self):
        self.kb_dir = Path(tempfile.mkdtemp())
        _write_model(self.kb_dir, _model())

    def test_approves_by_qualified_name_and_writes_both_files(self):
        ok = patch_grain_approval(
            kb_dir=str(self.kb_dir),
            table_fqn="EMDW_DMART.FNN_FCT",
            grain="one row per payment transaction",
        )
        self.assertTrue(ok)
        model = json.loads((self.kb_dir / MODEL_JSON).read_text(encoding="utf-8"))
        fnn = next(t for t in model["tables"] if t["table"] == "FNN_FCT")
        self.assertEqual(fnn["grain"], "one row per payment transaction")
        self.assertEqual(fnn["grain_status"], "approved")
        self.assertEqual(fnn["grain_confidence"], 100)
        yaml_text = (self.kb_dir / MODEL_YAML).read_text(encoding="utf-8")
        self.assertIn("one row per payment transaction", yaml_text)

    def test_approves_by_bare_table_name(self):
        ok = patch_grain_approval(
            kb_dir=str(self.kb_dir), table_name="FNN_FCT",
            grain="one row per payment",
        )
        self.assertTrue(ok)

    def test_unknown_table_returns_false(self):
        self.assertFalse(patch_grain_approval(
            kb_dir=str(self.kb_dir), table_fqn="NOPE.NOT_A_TABLE", grain="x",
        ))

    def test_empty_grain_returns_false(self):
        self.assertFalse(patch_grain_approval(
            kb_dir=str(self.kb_dir), table_fqn="EMDW_DMART.FNN_FCT", grain="   ",
        ))

    def test_approved_grain_survives_rebuild(self):
        patch_grain_approval(
            kb_dir=str(self.kb_dir),
            table_fqn="EMDW_DMART.FNN_FCT",
            grain="one row per payment transaction",
        )
        old_model = json.loads((self.kb_dir / MODEL_JSON).read_text(encoding="utf-8"))
        fresh = _model()   # rebuild regenerates needs_admin_context
        merged, _drift = preserve_approvals(old_model, fresh)
        fnn = next(t for t in merged["tables"] if t["table"] == "FNN_FCT")
        self.assertEqual(fnn["grain"], "one row per payment transaction")
        self.assertEqual(fnn["grain_status"], "approved")


class ModelHealthGrainTests(unittest.TestCase):
    def test_table_summaries_include_grain(self):
        kb_dir = Path(tempfile.mkdtemp())
        _write_model(kb_dir, _model())
        health = get_model_health(str(kb_dir))
        fnn = next(t for t in health["table_summaries"] if t["table"] == "FNN_FCT")
        self.assertEqual(fnn["grain"], "needs_admin_context")
        self.assertEqual(fnn["grain_status"], "needs_review")


class WiringGuardTests(unittest.TestCase):
    def test_grain_route_refreshes_quality_report(self):
        src = (ROOT / "admin" / "routes.py").read_text(encoding="utf-8")
        self.assertIn("model-health/grain", src)
        self.assertIn("patch_grain_approval", src)
        # Quality report must refresh immediately so grain_needs_review clears
        grain_route = src[src.index("model_health_approve_grain"):]
        self.assertIn("write_kb_quality_report", grain_route[:2500])

    def test_template_has_grain_review_and_vocab_coverage(self):
        tmpl = (ROOT / "admin" / "templates" / "client_model_health.html").read_text(encoding="utf-8")
        self.assertIn("model-health/grain", tmpl)
        self.assertIn('name="grain"', tmpl)
        self.assertIn("Vocabulary Coverage", tmpl)
        self.assertIn("Columns Needing Context", tmpl)
        self.assertIn("value-index/refresh", tmpl)


if __name__ == "__main__":
    unittest.main()
