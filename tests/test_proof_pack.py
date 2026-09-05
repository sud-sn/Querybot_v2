"""The proof pack — core/compliance/proof_pack.py.

Every test seeds real rows through the real store functions and then builds
the real pack. Nothing is handed in: if the pack ever stopped reading the
columns the audit path writes, these go red.

Two properties are asserted hardest, because they are what makes the pack
worth handing to anybody:

  * a call the product refused is never counted as a clean call, and a call
    whose manifest cannot be read is never counted as clean either — an
    audit that treats missing evidence as good news is worse than none; and
  * the hash chain is RECOMPUTED, not trusted, so a tampered decision log is
    reported as broken with the record it breaks at.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.compliance.proof_pack import (  # noqa: E402
    PACK_VERSION,
    build_proof_pack,
    calls_section,
    egress_section,
    fingerprint,
    grounding_section,
    integrity_section,
    refusals_section,
    summary_lines,
)


class ProofPackCase(unittest.TestCase):
    """Each case owns its database, so counts are counts of what it wrote."""

    def setUp(self):
        import store

        self._dir = tempfile.mkdtemp(prefix="qb-pack-")
        self._saved = {k: os.environ.get(k) for k in ("DB_PATH", "QUERYBOT_DB_PATH")}
        path = os.path.join(self._dir, "pack.db")
        os.environ["DB_PATH"] = path
        os.environ["QUERYBOT_DB_PATH"] = path
        store.init_db()
        self.account_id = f"acct-pack-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")

    def tearDown(self):
        import shutil

        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self._dir, ignore_errors=True)

    def _call(self, *, status="success", component="sql_generation",
              manifest=None, provider="anthropic", model="claude-sonnet-4-6",
              reason=""):
        import store

        store.log_llm_call(
            account_id=self.account_id,
            question_id="q1", request_id=uuid.uuid4().hex,
            question="how much revenue", component=component,
            llm_provider=provider, llm_model=model, status=status,
            payload_hash="h", payload_preview_sanitized=reason,
            prompt_chars=100, error_msg="",
            egress_manifest=json.dumps(manifest) if manifest is not None else "",
        )

    @staticmethod
    def _manifest(values_sent=False, sources=(), tables=(), columns=()):
        return {
            "tables": list(tables), "columns": list(columns),
            "content": ["question", "schema"],
            "values_sent": bool(values_sent),
            "value_sources": list(sources),
        }


class TestTheCallsSection(ProofPackCase):

    def test_clean_calls_are_counted_as_carrying_no_data_value(self):
        for _ in range(3):
            self._call(manifest=self._manifest(
                tables=["SALES.ORDERS"], columns=["NET_REVENUE"]))
        section = calls_section(self.account_id, 30)
        self.assertEqual(section["total"], 3)
        self.assertEqual(section["values_sent"]["calls_carrying_no_data_value"], 3)
        self.assertEqual(section["values_sent"]["calls_carrying_a_data_value"], 0)
        self.assertIn("3 of 3", section["statement"])

    def test_a_call_that_sent_a_value_is_counted_and_its_source_named(self):
        self._call(manifest=self._manifest())
        self._call(manifest=self._manifest(
            values_sent=True, sources=["verified filter values"]))
        section = calls_section(self.account_id, 30)
        self.assertEqual(section["values_sent"]["calls_carrying_a_data_value"], 1)
        self.assertEqual(section["values_sent"]["calls_carrying_no_data_value"], 1)
        self.assertEqual(
            section["values_sent"]["sources"], {"verified filter values": 1})

    def test_a_call_with_no_manifest_is_unknown_not_clean(self):
        # A row predating the manifest column, or one whose JSON will not
        # parse, is missing evidence. Counting it as clean is how an audit
        # ends up saying something it cannot support.
        self._call(manifest=None)
        self._call(manifest=self._manifest())
        section = calls_section(self.account_id, 30)
        self.assertEqual(section["values_sent"]["calls_with_no_manifest"], 1)
        self.assertEqual(section["values_sent"]["calls_carrying_no_data_value"], 1)
        self.assertIn("1 of 1", section["statement"])

    def test_an_unparseable_manifest_is_also_unknown(self):
        import store
        store.log_llm_call(
            account_id=self.account_id, question_id="q", request_id="r",
            question="q", component="sql_generation",
            llm_provider="anthropic", llm_model="m", status="success",
            payload_hash="", payload_preview_sanitized="", prompt_chars=1,
            error_msg="", egress_manifest="{not json",
        )
        section = calls_section(self.account_id, 30)
        self.assertEqual(section["values_sent"]["calls_with_no_manifest"], 1)
        self.assertEqual(section["values_sent"]["calls_carrying_no_data_value"], 0)

    def test_a_refused_call_is_never_counted_as_a_clean_one(self):
        # A refusal sent nothing. Counting it in the clean column inflates the
        # headline with calls that never happened.
        self._call(status="blocked", reason="regulated tenant")
        self._call(manifest=self._manifest())
        section = calls_section(self.account_id, 30)
        self.assertEqual(section["total"], 2)
        self.assertEqual(section["values_sent"]["calls_carrying_no_data_value"], 1)
        self.assertEqual(section["values_sent"]["calls_with_no_manifest"], 0)
        self.assertEqual(section["by_status"]["blocked"], 1)

    def test_the_tables_and_columns_described_are_collected(self):
        self._call(manifest=self._manifest(
            tables=["SALES.ORDERS"], columns=["NET_REVENUE", "ORDER_DATE"]))
        self._call(manifest=self._manifest(
            tables=["SALES.CUSTOMER"], columns=["NET_REVENUE"]))
        section = calls_section(self.account_id, 30)
        self.assertEqual(section["tables_described"],
                         ["SALES.CUSTOMER", "SALES.ORDERS"])
        self.assertEqual(section["columns_described"],
                         ["NET_REVENUE", "ORDER_DATE"])

    def test_the_endpoints_actually_used_are_listed(self):
        self._call(provider="local", model="llama3.1:8b",
                   manifest=self._manifest())
        self._call(provider="anthropic", model="claude-sonnet-4-6",
                   manifest=self._manifest())
        section = calls_section(self.account_id, 30)
        self.assertEqual(section["endpoints_used"],
                         ["anthropic:claude-sonnet-4-6", "local:llama3.1:8b"])

    def test_a_period_with_no_calls_says_so_rather_than_claiming_zero_leaks(self):
        section = calls_section(self.account_id, 30)
        self.assertEqual(section["total"], 0)
        self.assertIn("No model calls", section["statement"])

    def test_another_workspaces_calls_are_not_counted(self):
        import store
        other = f"acct-other-{uuid.uuid4().hex[:8]}"
        store.upsert_client(other, "portal")
        store.log_llm_call(
            account_id=other, question_id="q", request_id="r", question="q",
            component="sql_generation", llm_provider="openai", llm_model="m",
            status="success", payload_hash="", payload_preview_sanitized="",
            prompt_chars=1, error_msg="",
            egress_manifest=json.dumps(self._manifest(values_sent=True)),
        )
        self._call(manifest=self._manifest())
        section = calls_section(self.account_id, 30)
        self.assertEqual(section["total"], 1)
        self.assertEqual(section["values_sent"]["calls_carrying_a_data_value"], 0)


class TestTheRefusalsSection(ProofPackCase):

    def test_every_refusal_is_reported_with_its_reason(self):
        self._call(status="blocked", component="analysis",
                   reason="regulated tenant, LLM never received result rows")
        self._call(status="blocked", component="llm_complete",
                   reason="egress posture 'airgapped' does not permit 'openai'")
        section = refusals_section(self.account_id, 30)
        self.assertEqual(section["total"], 2)
        self.assertEqual(section["by_component"],
                         {"analysis": 1, "llm_complete": 1})
        reasons = " ".join(r["reason"] for r in section["listed"])
        self.assertIn("regulated tenant", reasons)
        self.assertIn("airgapped", reasons)

    def test_successful_calls_are_not_reported_as_refusals(self):
        self._call(manifest=self._manifest())
        self.assertEqual(refusals_section(self.account_id, 30)["total"], 0)

    def test_truncation_is_stated_rather_than_silent(self):
        # A pack that quietly lists 3 of 5 refusals reads as a complete list.
        for _ in range(5):
            self._call(status="blocked", reason="nope")
        section = refusals_section(self.account_id, 30, limit=3)
        self.assertEqual(section["total"], 5)
        self.assertEqual(len(section["listed"]), 3)
        self.assertTrue(section["truncated"])

    def test_a_complete_list_is_not_marked_truncated(self):
        for _ in range(2):
            self._call(status="blocked", reason="nope")
        section = refusals_section(self.account_id, 30, limit=10)
        self.assertFalse(section["truncated"])


class TestTheIntegritySection(ProofPackCase):

    def _decision(self, action="export"):
        import store
        return store.log_policy_decision(
            account_id=self.account_id, user_id="u1", action=action,
            purpose_id="p1", channel="portal", allowed=True,
            reason_code="ok", resources=["SALES.ORDERS"], obligations={},
            policy_version=1,
        )

    def test_an_untouched_chain_verifies(self):
        for _ in range(3):
            self._decision()
        section = integrity_section(self.account_id)
        self.assertTrue(section["verified"])
        self.assertEqual(section["records"], 3)
        self.assertTrue(section["head"])

    def test_an_empty_log_verifies_and_says_why(self):
        section = integrity_section(self.account_id)
        self.assertTrue(section["verified"])
        self.assertEqual(section["records"], 0)
        self.assertEqual(section["reason"], "no_decisions_recorded")

    def test_a_tampered_record_breaks_the_chain_and_is_located(self):
        # The chain is only evidence if something recomputes it. Rewriting a
        # decision's reason without recomputing its hash is exactly the edit
        # a hash chain exists to catch.
        import store
        self._decision()
        target = self._decision()
        self._decision()
        with store.get_db() as conn:
            conn.execute(
                "UPDATE policy_decision_log SET record_hash='tampered' WHERE id=?",
                (target,),
            )
        section = integrity_section(self.account_id)
        self.assertFalse(section["verified"])
        self.assertEqual(section["reason"], "chain_broken")
        # The break is reported at the record AFTER the edited one, because
        # that is the first record whose previous_hash no longer matches.
        self.assertTrue(section["first_broken_record"])
        self.assertNotEqual(section["first_broken_record"], target)

    def test_deleting_a_record_from_the_middle_breaks_the_chain(self):
        import store
        self._decision()
        middle = self._decision()
        self._decision()
        with store.get_db() as conn:
            conn.execute("DELETE FROM policy_decision_log WHERE id=?", (middle,))
        section = integrity_section(self.account_id)
        self.assertFalse(section["verified"])
        self.assertEqual(section["records"], 2)


class TestTheEgressAndGroundingSections(ProofPackCase):

    def test_an_airgapped_workspace_states_that_nothing_may_leave(self):
        import store
        store.save_compliance_profile(self.account_id, egress_posture="airgapped")
        section = egress_section(self.account_id)
        self.assertEqual(section["posture"], "airgapped")
        self.assertFalse(section["external_egress"])
        self.assertIn("No external model endpoint", section["statement"])

    def test_a_cloud_workspace_states_the_permitted_endpoints(self):
        import store
        store.save_compliance_profile(self.account_id, egress_posture="cloud")
        section = egress_section(self.account_id)
        self.assertTrue(section["external_egress"])
        self.assertIn("local", section["permitted_providers"])
        # The statement must not claim the air-gapped guarantee for a
        # workspace that does not have it — that sentence is the whole
        # section as far as a reader is concerned.
        self.assertNotIn("No external model endpoint", section["statement"])
        self.assertIn("permitted", section["statement"])

    def test_classification_coverage_counts_reviewed_columns(self):
        import store
        store.save_classification(
            self.account_id, "SALES.CUSTOMER", "CUSTOMER_NAME",
            sensitivity="RESTRICTED", identifiability="DIRECT", tags=["PII"],
            confidence=1.0, reviewed=True, reviewed_by="admin",
            mask_strategy="redact", source="admin",
        )
        store.save_classification(
            self.account_id, "SALES.ORDERS", "NET_REVENUE",
            sensitivity="INTERNAL", identifiability="NONE", tags=[],
            confidence=0.8, reviewed=False, reviewed_by="",
            mask_strategy="none", source="auto",
        )
        section = grounding_section(self.account_id)
        self.assertEqual(section["columns_classified"], 2)
        self.assertEqual(section["columns_reviewed"], 1)
        self.assertEqual(section["columns_tagged_sensitive"], 1)
        self.assertIn("redact", section["mask_strategies"])

    def test_a_regulated_workspace_states_the_suppression_rule(self):
        import store
        store.save_compliance_profile(self.account_id, mode="regulated")
        section = grounding_section(self.account_id)
        self.assertTrue(section["regulated"])
        self.assertIn("suppressed", section["statement"])

    def test_a_standard_workspace_says_the_boundary_does_not_apply(self):
        import store
        store.save_compliance_profile(self.account_id, mode="standard")
        section = grounding_section(self.account_id)
        self.assertFalse(section["regulated"])
        self.assertIn("not on the regulated boundary", section["statement"])


class TestTheWholePack(ProofPackCase):

    def test_a_pack_carries_every_section_and_a_fingerprint(self):
        pack = build_proof_pack(self.account_id, days=30)
        for section in ("egress", "calls", "refusals", "grounding", "integrity"):
            self.assertIn(section, pack, section)
        self.assertEqual(pack["pack_version"], PACK_VERSION)
        self.assertEqual(pack["account_id"], self.account_id)
        self.assertTrue(pack["fingerprint"])
        self.assertTrue(pack["generated_at"].endswith("Z"))

    def test_the_fingerprint_changes_when_the_evidence_does(self):
        first = build_proof_pack(self.account_id, days=30)
        self._call(manifest=self._manifest())
        second = build_proof_pack(self.account_id, days=30)
        self.assertNotEqual(first["fingerprint"], second["fingerprint"])

    def test_the_fingerprint_does_not_cover_itself(self):
        pack = build_proof_pack(self.account_id, days=30)
        recomputed = fingerprint(pack)
        self.assertEqual(recomputed, pack["fingerprint"])

    def test_a_failing_section_is_reported_rather_than_losing_the_pack(self):
        # Four sections and a stated gap serve an auditor better than an
        # exception.
        import core.compliance.proof_pack as module

        with patch.object(module, "integrity_section",
                          side_effect=RuntimeError("log unreadable")):
            pack = build_proof_pack(self.account_id, days=30)
        self.assertFalse(pack["integrity"]["available"])
        self.assertIn("log unreadable", pack["integrity"]["error"])
        self.assertIn("calls", pack)
        self.assertTrue(pack["fingerprint"])

    def test_the_summary_says_one_line_per_section(self):
        import store
        store.save_compliance_profile(self.account_id, egress_posture="airgapped")
        self._call(manifest=self._manifest())
        self._call(status="blocked", reason="regulated tenant")
        lines = summary_lines(build_proof_pack(self.account_id, days=30))
        self.assertEqual(len(lines), 5)
        joined = " ".join(lines)
        self.assertIn("No external model endpoint", joined)
        self.assertIn("1 of 1", joined)
        self.assertIn("1 model calls were refused", joined)

    def test_the_pack_contains_no_data_value_anywhere(self):
        # The pack is a review of the audit trail: counts, column names and
        # recorded reasons. A customer's data must not be reachable through
        # the document that proves it was protected.
        import store
        store.save_classification(
            self.account_id, "SALES.CUSTOMER", "CUSTOMER_NAME",
            sensitivity="RESTRICTED", identifiability="DIRECT", tags=["PII"],
            confidence=1.0, reviewed=True, reviewed_by="admin",
            mask_strategy="redact", source="admin",
        )
        self._call(manifest=self._manifest(
            tables=["SALES.CUSTOMER"], columns=["CUSTOMER_NAME"]))
        rendered = json.dumps(build_proof_pack(self.account_id, days=30))
        # Column and table NAMES are expected; a value is not.
        self.assertIn("CUSTOMER_NAME", rendered)
        self.assertNotIn("Ospedale", rendered)
        self.assertNotIn("how much revenue", rendered)


if __name__ == "__main__":
    unittest.main()


class TestTheChainOrdersByInsertion(ProofPackCase):
    """The chain used to fork under any burst inside one second.

    created_at has one-second resolution and the primary key is a random
    UUID, so "the previous record" was resolved by
    ORDER BY created_at DESC, id DESC — which returns whichever UUID sorts
    highest, not the true predecessor. Three decisions logged in the same
    second produced two records claiming the same one. A forked chain proves
    no ordering and cannot detect a deleted branch, which is most of what a
    hash chain is for. A single question logs several decisions, so this was
    the normal case, not an edge one.
    """

    def _decision(self):
        import store
        return store.log_policy_decision(
            account_id=self.account_id, user_id="u1", action="export",
            purpose_id="p1", channel="portal", allowed=True, reason_code="ok",
            resources=["SALES.ORDERS"], obligations={}, policy_version=1,
        )

    def test_a_burst_inside_one_second_still_forms_one_chain(self):
        import store
        # Ten in a row, all landing in the same wall-clock second.
        ids = [self._decision() for _ in range(10)]
        with store.get_db() as conn:
            rows = [dict(r) for r in conn.execute(
                "SELECT id, previous_hash, record_hash, seq FROM policy_decision_log "
                "WHERE account_id=? ORDER BY seq ASC", (self.account_id,),
            ).fetchall()]
        self.assertEqual(len(rows), 10)
        # Every record after the first points at exactly the one before it.
        for earlier, later in zip(rows, rows[1:]):
            self.assertEqual(later["previous_hash"], earlier["record_hash"])
        # And no two share a predecessor — the fork this guards against.
        predecessors = [r["previous_hash"] for r in rows]
        self.assertEqual(len(predecessors), len(set(predecessors)))
        self.assertEqual(len(ids), 10)

    def test_the_sequence_is_dense_and_starts_at_one(self):
        for _ in range(4):
            self._decision()
        import store
        with store.get_db() as conn:
            seqs = [r["seq"] for r in conn.execute(
                "SELECT seq FROM policy_decision_log WHERE account_id=? "
                "ORDER BY seq ASC", (self.account_id,),
            ).fetchall()]
        self.assertEqual(seqs, [1, 2, 3, 4])

    def test_a_burst_verifies_in_the_proof_pack(self):
        for _ in range(10):
            self._decision()
        section = integrity_section(self.account_id)
        self.assertTrue(section["verified"], section)
        self.assertEqual(section["records"], 10)

    def test_two_workspaces_keep_separate_chains(self):
        import store
        other = f"acct-other-{uuid.uuid4().hex[:8]}"
        store.upsert_client(other, "portal")
        self._decision()
        store.log_policy_decision(
            account_id=other, user_id="u", action="export", purpose_id="p",
            channel="portal", allowed=True, reason_code="ok",
            resources=["T"], obligations={}, policy_version=1,
        )
        self._decision()
        self.assertTrue(integrity_section(self.account_id)["verified"])
        self.assertTrue(integrity_section(other)["verified"])
        self.assertEqual(integrity_section(self.account_id)["records"], 2)

    def test_a_forked_chain_is_reported_as_forked_not_as_tampering(self):
        # Historical rows written before the sequence existed can genuinely
        # fork. Calling that "tampered" would tell an auditor something false.
        import store
        self._decision()
        second = self._decision()
        third = self._decision()
        with store.get_db() as conn:
            first_hash = conn.execute(
                "SELECT previous_hash FROM policy_decision_log WHERE id=?",
                (second,),
            ).fetchone()["previous_hash"]
            conn.execute(
                "UPDATE policy_decision_log SET previous_hash=? WHERE id=?",
                (first_hash, third),
            )
        section = integrity_section(self.account_id)
        self.assertFalse(section["verified"])
        self.assertEqual(section["reason"], "chain_forked")
        self.assertEqual(section["also_claimed_by"], second)
        self.assertIn("same predecessor", section["statement"])


class TestTheAdminDownload(ProofPackCase):
    """The route an admin actually clicks."""

    def _get(self, **params):
        import asyncio
        from unittest.mock import MagicMock

        import admin.routes as routes

        req = MagicMock()
        req.query_params = params
        with patch.object(routes, "_is_auth", return_value=True):
            return asyncio.run(routes.compliance_proof_pack(req, self.account_id))

    def test_the_route_returns_a_pack_for_this_workspace(self):
        self._call(manifest=self._manifest())
        response = self._get()
        body = json.loads(response.body)
        self.assertEqual(body["account_id"], self.account_id)
        self.assertEqual(body["window_days"], 30)
        self.assertTrue(body["fingerprint"])
        self.assertEqual(body["calls"]["total"], 1)

    def test_it_downloads_rather_than_renders(self):
        # The fingerprint only means anything on a document the auditor can
        # keep; a page they have to screenshot is not that.
        response = self._get()
        disposition = response.headers.get("content-disposition", "")
        self.assertIn("attachment", disposition)
        self.assertIn(self.account_id, disposition)

    def test_the_window_comes_from_the_query_string(self):
        body = json.loads(self._get(days="90").body)
        self.assertEqual(body["window_days"], 90)

    def test_a_nonsense_window_falls_back_rather_than_erroring(self):
        for value in ("banana", "", "0", "99999"):
            body = json.loads(self._get(days=value).body)
            self.assertGreaterEqual(body["window_days"], 1)
            self.assertLessEqual(body["window_days"], 365)

    def test_an_unauthenticated_request_gets_nothing(self):
        import asyncio
        from unittest.mock import MagicMock

        import admin.routes as routes
        from fastapi import HTTPException

        req = MagicMock()
        req.query_params = {}
        with patch.object(routes, "_is_auth", return_value=False):
            with self.assertRaises(HTTPException):
                asyncio.run(routes.compliance_proof_pack(req, self.account_id))
