# -*- coding: utf-8 -*-
"""tests/test_preflight_reads_health.py

The preflight reported "0 error(s)" over a graph that had them.

deploy/preflight_live.py exists to separate "this case failed" from "this case
could never have passed here". It compared each issue's severity against
"ERROR" while core.graph_health stores SEVERITY_ERROR = "error", so the count
was structurally zero on every workspace — and a graph error is precisely the
condition it is meant to stop a tester walking into: a join that can never be
used, which every question routed through it fails on.

Told a live tenant its 14-issue graph had no errors. Executed against a real
workspace holding a fact-to-fact edge.
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

SCHEMA = {"S.F_A": {"ID": "int"}, "S.F_B": {"ID": "int"}, "S.D_C": {"ID": "int"}}


class TestThePreflightCountsRealErrors(unittest.TestCase):

    def setUp(self):
        import store

        self._dir = tempfile.mkdtemp(prefix="qb-pf-")
        self._saved = {k: os.environ.get(k)
                       for k in ("DB_PATH", "QUERYBOT_DB_PATH", "QUERYBOT_KEY_FILE")}
        path = os.path.join(self._dir, "d.db")
        os.environ["DB_PATH"] = path
        os.environ["QUERYBOT_DB_PATH"] = path
        os.environ["QUERYBOT_KEY_FILE"] = os.path.join(self._dir, ".key")
        store.init_db()
        self.account_id = f"acct-pf-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")

        schema_dir = os.path.join(self._dir, "schema")
        os.makedirs(schema_dir)
        (Path(schema_dir) / "_schema.json").write_text(json.dumps({
            fqn: {"columns": [{"name": c, "type": t} for c, t in cols.items()]}
            for fqn, cols in SCHEMA.items()
        }), encoding="utf-8")
        store.update_client_state(self.account_id, "READY",
                                  {"schema_dir": schema_dir})
        for name, table, kind in (("A", "F_A", "fact"), ("B", "F_B", "fact"),
                                  ("C", "D_C", "dimension")):
            store.save_entity(account_id=self.account_id, entity_name=name,
                              table_name=table, schema_name="S",
                              entity_type=kind)

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self._dir, ignore_errors=True)

    def _join(self, frm, to):
        import store

        store.save_relationship(account_id=self.account_id, from_entity=frm,
                                to_entity=to, from_column="ID", to_column="ID")

    def severities(self):
        from core.graph_health import check_graph_health

        return [i.severity for i in (check_graph_health(self.account_id).issues or [])]

    def test_the_module_spells_severity_in_lower_case(self):
        # The whole defect: a literal "ERROR" that matches nothing.
        from core.graph_health import SEVERITY_ERROR, SEVERITY_WARNING

        self.assertEqual(SEVERITY_ERROR, "error")
        self.assertEqual(SEVERITY_WARNING, "warning")

    def test_a_fact_to_fact_edge_raises_an_error_severity(self):
        # The fixture is real before anything is claimed about counting it.
        self._join("A", "B")
        self.assertIn("error", self.severities())

    def test_the_preflight_counts_it(self):
        import importlib.util

        self._join("A", "B")
        spec = importlib.util.spec_from_file_location(
            "_preflight", Path(__file__).resolve().parents[1]
            / "deploy" / "preflight_live.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        printed: list[str] = []
        module.line = lambda mark, label, detail="": printed.append(
            f"{mark}|{label}|{detail}")
        module.report(self.account_id)

        health = [row for row in printed if "|health|" in row]
        self.assertEqual(len(health), 1, printed)
        self.assertIn("1 error(s)", health[0])
        self.assertTrue(health[0].startswith(module.GAP),
                        f"a graph error is a blocking gap, got {health[0]!r}")

    def _report_code(self) -> int:
        """The exit code, with every OTHER blocking gap removed.

        A script or a CI step reads the code, so marking the health line GAP
        without counting it would be the same false assurance in a second
        place. Chat UI on and Qdrant answering, so the graph is the only thing
        left that can block.
        """
        import importlib.util
        from unittest.mock import patch

        import store

        store.update_client_meta(self.account_id, chat_ui_enabled=1)
        spec = importlib.util.spec_from_file_location(
            "_preflight_code", Path(__file__).resolve().parents[1]
            / "deploy" / "preflight_live.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.line = lambda *a, **k: None
        import core.vector_store as vector_store

        with patch.object(vector_store, "_qdrant", return_value=object()):
            return module.report(self.account_id)

    def test_a_graph_error_blocks_the_run(self):
        self._join("A", "B")
        self.assertEqual(self._report_code(), 1)

    def test_a_workspace_with_no_graph_error_does_not_block(self):
        # The other half: without this the check above passes for a preflight
        # that blocks on everything.
        self._join("A", "C")
        self.assertEqual(self._report_code(), 0)

    def test_a_clean_graph_still_reads_as_clean(self):
        # The guard must not fire on the workspace it sits next to.
        self._join("A", "C")
        self.assertNotIn("error", self.severities())


class TestThePreflightReadsTheCompliancePosture(unittest.TestCase):
    """F31 · Four cases in the plan only mean anything on a regulated tenant.

    L3-1, L3-2, L3-3 and L8-10 — the whole Phase-C computed-analysis section —
    require regulated mode. deploy/preflight_live.py never read the compliance
    profile at all: a grep for "compliance", "regulated" or "policy_engine"
    returned nothing across the file. So on a standard workspace it printed
    "No blocking gaps", exited 0, and a tester recorded four passes for
    behaviour that was never exercised.

    That is the same shape as the bug the class above fixes, and as the pytest
    command that ran zero tests: a tool that cannot see a prerequisite reports
    its own blind spot as readiness.

    A warn rather than a gap — the workspace is fine, a section of the plan is
    not runnable on it. And BOTH postures get one, because advising a standard
    tenant to switch to regulated without saying what that turns off would be
    the same defect mirrored.
    """

    def lines(self, mode, pack_key="", regulated=None):
        """Every line the real report() prints, for one stored posture."""
        import importlib.util
        from unittest.mock import patch

        import store

        spec = importlib.util.spec_from_file_location(
            "_preflight_posture", Path(__file__).resolve().parents[1]
            / "deploy" / "preflight_live.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        printed: list[str] = []
        module.line = lambda mark, label, detail="": printed.append(
            f"{mark}|{label}|{detail}")
        profile = ({"mode": mode, "policy_pack_key": pack_key}
                   if mode is not None else {})
        is_reg = (str(mode or "") == "regulated") if regulated is None else regulated
        with patch.object(store, "get_client",
                          return_value={"client_name": "T", "state": "READY",
                                        "chat_ui_enabled": 1}), \
             patch.object(store, "get_client_state",
                          return_value={"schema_dir": "/nonexistent"}), \
             patch.object(store, "get_compliance_profile", return_value=profile), \
             patch("core.compliance.policy_engine.is_regulated",
                   return_value=is_reg):
            module.report("acct")
        return printed, module

    def posture(self, *args, **kw):
        printed, module = self.lines(*args, **kw)
        found = [row for row in printed if "compliance mode" in row
                 or "no posture chosen" in row]
        self.assertTrue(found, printed)
        return found[0], module

    def test_a_standard_workspace_is_told_which_cases_it_cannot_run(self):
        row, module = self.posture("standard")
        self.assertTrue(row.startswith(module.WARN), row)
        for case in ("L3-1", "L3-2", "L3-3", "L8-10"):
            with self.subTest(case=case):
                self.assertIn(case, row)

    def test_and_told_not_to_record_them_as_passes(self):
        row, _ = self.posture("standard")
        self.assertIn("not-run", row)

    def test_a_regulated_workspace_with_a_pack_is_told_they_are_runnable(self):
        row, module = self.posture("regulated", "healthcare_pharmacy_v1")
        self.assertTrue(row.startswith(module.OK), row)
        self.assertIn("L3-1", row)

    def test_regulated_with_no_pack_is_not_reported_as_ready(self):
        # is_regulated is true, but the value index harvests nothing without a
        # resolvable pack, so a different section of the plan goes dark.
        row, module = self.posture("regulated", "")
        self.assertTrue(row.startswith(module.WARN), row)
        self.assertIn("L10-1", row)

    def test_regulated_mode_is_told_what_it_turns_off(self):
        # The mirrored defect: advising a switch without naming the cost.
        printed, _ = self.lines("regulated", "healthcare_pharmacy_v1")
        disabled = [r for r in printed if "disables cases too" in r]
        self.assertTrue(disabled, printed)
        self.assertIn("L11-31", disabled[0])

    def test_an_unset_posture_is_not_silently_treated_as_either(self):
        row, module = self.posture(None)
        self.assertTrue(row.startswith(module.WARN), row)
        self.assertIn("L3-1", row)

    def test_a_mode_stored_in_the_wrong_case_is_called_out(self):
        # is_regulated compares the exact literal, so "REGULATED" behaves as
        # standard — which looks like a configured regulated tenant to anyone
        # reading the settings page.
        row, module = self.posture("REGULATED", regulated=False)
        self.assertTrue(row.startswith(module.WARN), row)
        self.assertIn("L3-1", row)
        # Asserted on the DETAIL, not the row: "REGULATED" is in the label
        # either way, so checking the row alone passes whether or not the
        # explanation is there.
        detail = row.split("|", 2)[2]
        self.assertIn("behaves as standard", detail)
        self.assertIn("'regulated'", detail)

    def test_the_section_is_printed_at_all(self):
        # The finding itself: there was no compliance line anywhere.
        printed, _ = self.lines("standard")
        self.assertTrue(any("compliance" in row.lower() or "posture" in row.lower()
                            for row in printed), printed)


class TestThePreflightStaysReadOnly(unittest.TestCase):
    """The tool's own docstring: "It reads and prints; it changes nothing."

    The readiness block called core.compliance.readiness.assess(), which
    INSERTs a row into compliance_assessment_run. So every time an operator ran
    the preflight before a test session it wrote to the tenant's compliance
    history, and an auditor reading that history could not tell a real
    assessment from a preflight.
    """

    def _run(self, latest, users=()):
        import importlib.util
        from unittest.mock import patch

        import store

        spec = importlib.util.spec_from_file_location(
            "_preflight_readonly", Path(__file__).resolve().parents[1]
            / "deploy" / "preflight_live.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        printed: list[str] = []
        module.line = lambda mark, label, detail="": printed.append(
            f"{mark}|{label}|{detail}")

        import core.compliance.readiness as readiness

        def _must_not_run(*a, **kw):
            raise AssertionError(
                "the preflight ran a readiness assessment, which writes a row "
                "to compliance_assessment_run")

        with patch.object(store, "get_client",
                          return_value={"client_name": "T", "state": "READY",
                                        "chat_ui_enabled": 1}), \
             patch.object(store, "get_client_state",
                          return_value={"schema_dir": "/nonexistent"}), \
             patch.object(store, "get_compliance_profile",
                          return_value={"mode": "standard"}), \
             patch.object(store, "get_latest_assessment", return_value=latest), \
             patch.object(store, "list_users", return_value=list(users)), \
             patch.object(readiness, "assess", _must_not_run), \
             patch("core.compliance.policy_engine.is_regulated",
                   return_value=False):
            module.report("acct")
        return printed, module

    def test_it_does_not_start_an_assessment(self):
        printed, _ = self._run(None)
        self.assertTrue(printed)

    def test_it_says_so_when_there_is_none_on_record(self):
        printed, module = self._run(None)
        row = next(r for r in printed if "readiness assessment" in r)
        self.assertTrue(row.startswith(module.WARN), row)
        self.assertIn("write to the audit history", row)

    def test_a_failing_control_is_named(self):
        """The block read (…).get("controls") with c.get("id"), and assess()
        returns "results" keyed "control_key" — so `failing` was always empty
        and this line could not print on any workspace. Reaching it, every
        control would have been named "?"."""
        printed, module = self._run({
            "created_at": "2026-09-01",
            "results": [{"control_key": "provider_agreement", "status": "fail"},
                        {"control_key": "profile_selected", "status": "pass"}],
        })
        row = next(r for r in printed if "readiness control" in r)
        self.assertTrue(row.startswith(module.WARN), row)
        self.assertIn("provider_agreement", row)
        self.assertNotIn("?", row.split("|", 2)[2])

    def test_all_passing_says_when_it_was_assessed(self):
        printed, module = self._run({
            "created_at": "2026-09-01",
            "results": [{"control_key": "profile_selected", "status": "pass"}],
        })
        row = next(r for r in printed if "readiness control" in r)
        self.assertTrue(row.startswith(module.OK), row)
        self.assertIn("2026-09-01", row)


class TestThePreflightChecksTheUserPreconditionItNames(unittest.TestCase):
    """The line said "L1-3 and L9-x need one RESTRICTED and one unrestricted"
    and then counted rows. Two unrestricted analysts reported [ok]."""

    def _users_row(self, users):
        import importlib.util
        from unittest.mock import patch

        import store

        spec = importlib.util.spec_from_file_location(
            "_preflight_users", Path(__file__).resolve().parents[1]
            / "deploy" / "preflight_live.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        printed: list[str] = []
        module.line = lambda mark, label, detail="": printed.append(
            f"{mark}|{label}|{detail}")
        allowed = {u["id"]: u.pop("_allowed") for u in users}

        with patch.object(store, "get_client",
                          return_value={"client_name": "T", "state": "READY",
                                        "chat_ui_enabled": 1}), \
             patch.object(store, "get_client_state",
                          return_value={"schema_dir": "/nonexistent"}), \
             patch.object(store, "get_compliance_profile", return_value={}), \
             patch.object(store, "get_latest_assessment", return_value=None), \
             patch.object(store, "list_users", return_value=users), \
             patch.object(store, "get_allowed_tables",
                          side_effect=lambda u: allowed[u["id"]]), \
             patch("core.compliance.policy_engine.is_regulated",
                   return_value=False):
            module.report("acct")
        return next(r for r in printed if "portal users" in r), module

    UNRESTRICTED = {"id": 1, "role": "admin", "_allowed": None}
    RESTRICTED = {"id": 2, "role": "analyst", "_allowed": {"DB.S.SALES"}}

    def test_two_unrestricted_users_are_not_enough(self):
        row, module = self._users_row([dict(self.UNRESTRICTED),
                                       dict(self.UNRESTRICTED, id=3)])
        self.assertTrue(row.startswith(module.WARN), row)
        self.assertIn("RESTRICTED", row)

    def test_two_restricted_users_are_not_enough_either(self):
        row, module = self._users_row([dict(self.RESTRICTED),
                                       dict(self.RESTRICTED, id=4)])
        self.assertTrue(row.startswith(module.WARN), row)
        self.assertIn("unrestricted", row)

    def test_one_of_each_is(self):
        row, module = self._users_row([dict(self.UNRESTRICTED),
                                       dict(self.RESTRICTED)])
        self.assertTrue(row.startswith(module.OK), row)
        self.assertIn("1 restricted", row)

    def test_and_the_cases_that_need_it_are_named(self):
        row, _ = self._users_row([dict(self.UNRESTRICTED),
                                  dict(self.UNRESTRICTED, id=3)])
        for case in ("L1-3", "L9-3", "L8-14"):
            with self.subTest(case=case):
                self.assertIn(case, row)


if __name__ == "__main__":
    unittest.main()
