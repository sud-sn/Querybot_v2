"""
tests/test_accepting_a_date_role_draft_approves_it.py

The drafts queue said the work was done and the resolver never heard about it.

The product offers the same decision on two surfaces. The Date Roles admin
screen calls core.semantic_model.patch_date_role(..., status="approved"), which
writes _semantic_model.json -- and model.date_roles[].status == "approved" is
the only thing core.contextual_dates.resolve_contextual_date_binding reads. The
drafts queue offered the identical proposal and called
core.draft_review.apply_property_proposal, which wrote entity_properties with
role='date' and stopped.

entity_properties is read by the graph resolver, which decides which TABLE a
question is about. It is not read by the date resolver, which decides which
DATE. So an admin who accepted the date-role draft got a confirmed column
property, no approved Date Role, and a measure that carried on asking "which
date should I use?" on every period question -- while the queue that had just
been emptied said it was finished.

core/draft_review.py already had exactly this pattern once, and named it:
publish_column_terms exists because "entity_properties.synonyms reaches
core.graph_resolver ... it does not reach direct_aliases". The date role is the
same shape and was missed. publish_date_role is its sibling.

These tests start at the accept API and end at the RESOLVER, with nothing
passed between them by the test itself -- which is the only arrangement that
can tell "accepted" from "took effect".
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.contextual_dates import (  # noqa: E402
    resolve_contextual_date_binding,
)
from core.draft_review import apply_property_proposal  # noqa: E402
from core.semantic_model import MODEL_JSON, load_semantic_model  # noqa: E402

ENTITY = "Invoices"
SCHEMA = "EMDW"
TABLE = "CUS_ORD_IVC_FCT"
FACT = f"{SCHEMA}.{TABLE}"
COLUMN = "IVC_DT"
METRIC = {"id": 1, "name": "Net Revenue", "base_table": FACT}


class DateRoleDraftCase(unittest.TestCase):

    def setUp(self):
        import store

        self._dir = tempfile.mkdtemp(prefix="qb-draft-date-")
        self._saved = {k: os.environ.get(k) for k in ("DB_PATH", "QUERYBOT_DB_PATH")}
        path = os.path.join(self._dir, "d.db")
        os.environ["DB_PATH"] = path
        os.environ["QUERYBOT_DB_PATH"] = path
        store.init_db()
        self.account_id = f"acct-draft-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")

        # The graph entity the draft's target_id names.
        store.save_entity(self.account_id, ENTITY, TABLE, schema_name=SCHEMA,
                          entity_type="fact")

        # A generated date role on disk, exactly as a KB build leaves it.
        self.kb_dir = Path(self._dir) / "kb"
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        self._write_model(status="generated")
        from core.pipeline_context import save_state
        save_state(self.account_id, "READY", {"kb_dir": str(self.kb_dir)})

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self._dir, ignore_errors=True)

    def _write_model(self, *, status, column=COLUMN, fact=FACT):
        role = {
            "fact_table": fact, "fact_column": column,
            "business_role": "invoice_date", "name": "Invoice Date",
            "role_key": "invoice_date", "label": "Invoice Date",
            "date_key_type": "native_date", "status": status,
            "confidence": 80, "is_default": 0,
        }
        (self.kb_dir / MODEL_JSON).write_text(json.dumps({
            "tables": [{"table": fact, "date_roles": [dict(role)]}],
            "date_roles": [dict(role)],
        }), encoding="utf-8")

    # ── the two ends ────────────────────────────────────────────────────────

    def _accept(self, column=COLUMN, role="date"):
        """The accept API, called the way the drafts queue calls it."""
        return apply_property_proposal(self.account_id, {
            "target_id": f"{ENTITY}.{column}",
            "payload": {"role": role, "display_name": "Invoice Date",
                        "synonyms": "invoice date, billing date"},
            "before": {},
            "generated_by": "model_drafts",
            "reason": "date role proposal",
        })

    def _resolver_says(self):
        """The other end: what the runtime does with a period question. Reads
        the model back off disk -- the test hands it nothing."""
        model = load_semantic_model(str(self.kb_dir))
        return resolve_contextual_date_binding(
            "net revenue by month",
            matched_metrics=[dict(METRIC)],
            bindings=[],
            date_roles=list(model.get("date_roles") or []),
            required_fact_tables={FACT},
        )


class TestAcceptingItMakesTheRuntimeUseIt(DateRoleDraftCase):

    def test_before_accepting_the_date_is_only_discovered(self):
        """The premise, stated as the provenance the reader is shown.

        A generated role on a column the warehouse types as DATE IS usable --
        the resolver binds it rather than asking, which is deliberate -- but it
        binds it as discovered_date_role, whose reader-facing phrase is "a
        business date found on this data, not an approved default". Approving it
        is what removes that caveat, and what makes it survive a rebuild as a
        governed decision rather than a fresh guess.
        """
        resolved = self._resolver_says()
        self.assertEqual(resolved["binding"].get("resolution_source"),
                         "discovered_date_role")

    def test_accepting_it_approves_it_in_the_semantic_model(self):
        applied, message = self._accept()
        self.assertTrue(applied, message)
        roles = load_semantic_model(str(self.kb_dir)).get("date_roles") or []
        self.assertEqual([r.get("status") for r in roles], ["approved"])

    def test_and_the_per_table_copy_too(self):
        """build_runtime_semantic_plan reads the per-table list, not the
        top-level one. Approving only one of them is a half-approval that looks
        complete on the Date Roles screen."""
        self._accept()
        model = load_semantic_model(str(self.kb_dir))
        table_roles = (model.get("tables") or [{}])[0].get("date_roles") or []
        self.assertEqual([r.get("status") for r in table_roles], ["approved"])

    def test_after_accepting_the_runtime_treats_it_as_approved(self):
        """The whole point, end to end. Nothing is passed from the accept to
        the resolver by this test: the model is re-read off disk."""
        self._accept()
        resolved = self._resolver_says()
        self.assertEqual(resolved.get("status"), "selected", resolved)
        self.assertEqual(resolved["binding"].get("fact_column"), COLUMN)
        self.assertEqual(resolved["binding"].get("resolution_source"),
                         "single_approved_date_role")

    def test_the_caveat_the_reader_saw_is_gone(self):
        """Read through the phrase the answer card actually prints, because
        that is where "not an approved default" reaches a person."""
        from core.date_roles import provenance_phrase

        before = provenance_phrase(
            self._resolver_says()["binding"]["resolution_source"])
        self.assertIn("not an approved default", before)
        self._accept()
        after = provenance_phrase(
            self._resolver_says()["binding"]["resolution_source"])
        self.assertNotIn("not an approved default", after)
        self.assertTrue(after)

    def test_the_column_property_is_still_written(self):
        """The original behaviour is additional, not replaced -- the graph
        resolver reads it to decide which table a question is about."""
        import store
        self._accept()
        rows = store.list_entity_properties(self.account_id, ENTITY)
        match = [r for r in rows
                 if str(r.get("column_name") or "").upper() == COLUMN]
        self.assertTrue(match)
        self.assertEqual(match[0].get("status"), "confirmed")

    def test_re_accepting_a_stale_proposal_is_refused_not_reapplied(self):
        """Pre-existing and correct: the second accept carries before={} while
        the column now has a confirmed value, so property_conflict refuses it.
        Asserted so the publishing step below is not mistaken for it."""
        self._accept()
        applied, _ = self._accept()
        self.assertFalse(applied)

    def test_publishing_it_again_is_idempotent(self):
        """The step this commit adds, on its own. An admin approving the same
        role from both surfaces must not end up with two entries or a
        half-written model."""
        from core.draft_review import publish_date_role

        self.assertEqual(
            publish_date_role(self.account_id, ENTITY, COLUMN, "Invoice Date",
                              "invoice date"), "")
        self.assertEqual(
            publish_date_role(self.account_id, ENTITY, COLUMN, "Invoice Date",
                              "invoice date"), "")
        roles = load_semantic_model(str(self.kb_dir)).get("date_roles") or []
        self.assertEqual([r.get("status") for r in roles], ["approved"])
        self.assertEqual(self._resolver_says()["binding"]["resolution_source"],
                         "single_approved_date_role")


class TestANonDateProposalIsUntouched(DateRoleDraftCase):
    """The date role is one kind of property proposal. The others must not
    start patching the semantic model."""

    def test_a_dimension_proposal_approves_no_date_role(self):
        applied, _ = self._accept(role="dimension")
        self.assertTrue(applied)
        roles = load_semantic_model(str(self.kb_dir)).get("date_roles") or []
        self.assertEqual([r.get("status") for r in roles], ["generated"])


class TestItSaysSoWhenItCannot(DateRoleDraftCase):
    """A draft accepted into nothing is worse than one never offered, because
    the admin has no reason to look again."""

    def test_a_column_absent_from_the_model_is_reported(self):
        applied, message = self._accept(column="SOME_OTHER_DT")
        self.assertTrue(applied, "the column property should still be saved")
        self.assertIn("not in the semantic model", message)

    def test_a_workspace_with_no_kb_directory_is_reported(self):
        from core.pipeline_context import save_state
        save_state(self.account_id, "READY", {})
        applied, message = self._accept()
        self.assertTrue(applied)
        self.assertIn("KB directory", message)

    def test_an_entity_with_no_table_is_reported(self):
        import store
        store.save_entity(self.account_id, "Orphan", "", schema_name="")
        applied, message = apply_property_proposal(self.account_id, {
            "target_id": f"Orphan.{COLUMN}",
            "payload": {"role": "date", "display_name": "D", "synonyms": ""},
            "before": {},
        })
        self.assertTrue(applied)
        self.assertIn("does not map to a table", message)

    def test_the_accept_never_raises(self):
        """It is a queue action behind a button. An exception here loses the
        admin's decision entirely."""
        import core.draft_review as dr
        original = dr.patch_date_role if hasattr(dr, "patch_date_role") else None
        import core.semantic_model as sm
        broken = sm.patch_date_role
        sm.patch_date_role = lambda **kw: (_ for _ in ()).throw(
            RuntimeError("model file locked"))
        try:
            applied, message = self._accept()
            self.assertTrue(applied)
            self.assertIn("Date Roles screen", message)
        finally:
            sm.patch_date_role = broken
            if original is not None:
                dr.patch_date_role = original


if __name__ == "__main__":
    unittest.main()
