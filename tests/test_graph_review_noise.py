from __future__ import annotations

import json
import tempfile
import unittest
import uuid
from pathlib import Path

from core.graph_autopopulate import classify_harvested_property, enrich_graph_from_kb


class HarvestClassificationTests(unittest.TestCase):
    def test_keys_dates_and_measures_are_not_confused(self):
        self.assertEqual(classify_harvested_property("RX_ORDER_ID", "int", metric_claim=True), "identifier")
        self.assertEqual(classify_harvested_property("THERAPY_START_DATE_ID", "int", metric_claim=True), "date")
        self.assertEqual(classify_harvested_property("NET_AMOUNT", "decimal(18,2)", metric_claim=True), "metric")


class HarvestLifecycleTests(unittest.TestCase):
    def setUp(self):
        import store
        store.init_db()
        self.store = store
        self.account_id = f"acct-noise-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")
        store.save_entity(
            self.account_id, "FACT_RX", "FACT_RX", schema_name="PHARMA",
            entity_type="fact", status="suggested",
        )

    def tearDown(self):
        with self.store.get_db() as conn:
            for table in ("entity_properties", "entity_relationships", "entity_graph"):
                conn.execute(f"DELETE FROM {table} WHERE account_id=?", (self.account_id,))
            conn.execute("DELETE FROM client WHERE account_id=?", (self.account_id,))

    def _write_fixture(self, root: Path) -> tuple[Path, Path]:
        schema_dir = root / "schema"
        kb_dir = root / "kb"
        schema_dir.mkdir()
        kb_dir.mkdir()
        schema = {
            "PHARMA.FACT_RX": {
                "columns": [
                    {"name": "RX_ORDER_ID", "type": "int"},
                    {"name": "THERAPY_START_DATE_ID", "type": "int"},
                    {"name": "NET_AMOUNT", "type": "decimal(18,2)"},
                ],
                "pk_columns": ["RX_ORDER_ID"],
            }
        }
        (schema_dir / "_schema.json").write_text(json.dumps(schema), encoding="utf-8")
        (kb_dir / "FACT_RX_kb.md").write_text(
            "## Overview\nPrescription transaction facts.\n\n"
            "## Business Synonyms\n"
            "| Plain English | Column | Notes |\n|---|---|---|\n"
            "| order number | RX_ORDER_ID | business identifier |\n"
            "| therapy start | THERAPY_START_DATE_ID | role date |\n"
            "| net sales | NET_AMOUNT | monetary measure |\n\n"
            "## Key Metrics\n"
            "- **Order count**: `RX_ORDER_ID`\n"
            "- **Therapy date**: `THERAPY_START_DATE_ID`\n"
            "- **Net revenue**: `NET_AMOUNT`\n",
            encoding="utf-8",
        )
        return schema_dir, kb_dir

    def test_harvest_uses_physical_metadata_and_prunes_stale_suggestions(self):
        self.store.save_entity_property(
            self.account_id, "FACT_RX", "OLD_NOISE", status="suggested",
            generated_by="kb_harvest", reason="old build",
        )
        with tempfile.TemporaryDirectory() as tmp:
            schema_dir, kb_dir = self._write_fixture(Path(tmp))
            enrich_graph_from_kb(
                self.account_id, str(kb_dir), schema_dir=str(schema_dir)
            )
        props = {p["column_name"]: p for p in self.store.list_entity_properties(self.account_id, "FACT_RX")}
        self.assertEqual(props["RX_ORDER_ID"]["role"], "identifier")
        self.assertEqual(props["THERAPY_START_DATE_ID"]["role"], "date")
        self.assertEqual(props["NET_AMOUNT"]["role"], "metric")
        self.assertNotIn("OLD_NOISE", props)

    def test_rejected_harvest_decision_survives_rebuild(self):
        self.store.save_entity_property(
            self.account_id, "FACT_RX", "RX_ORDER_ID", role="identifier",
            status="rejected", generated_by="kb_harvest",
        )
        with tempfile.TemporaryDirectory() as tmp:
            schema_dir, kb_dir = self._write_fixture(Path(tmp))
            enrich_graph_from_kb(self.account_id, str(kb_dir), schema_dir=str(schema_dir))
        prop = next(p for p in self.store.list_entity_properties(self.account_id, "FACT_RX") if p["column_name"] == "RX_ORDER_ID")
        self.assertEqual(prop["status"], "rejected")


if __name__ == "__main__":
    unittest.main()
