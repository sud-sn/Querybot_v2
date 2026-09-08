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


if __name__ == "__main__":
    unittest.main()
