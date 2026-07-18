"""
Compiled semantic contract + golden-eval gates — regression tests.

Covers:
  1a — contract compile determinism + versioning
  1b — recompile hooks wired into approval routes
  1c — consumers read the contract instead of their stores
  1d — version stamping (learning versions, metric_version snapshots)
  1e — KB divergence warning data
  2a — health-score passed_cases fix
  2c — eval regression detection
  2e — golden-suite seeder (harvest + merge, never clobber)
  2f — offline cases skip the LLM
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def _src(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


_MODEL = {
    "tables": [{
        "qualified_name": "ERP.SALES_FCT", "table": "SALES_FCT", "type": "fact",
        "grain": "one row per line", "fields": [
            {"column": "NET_AMT", "status": "approved", "role": "measure",
             "approved_meaning": "net revenue", "business_candidates": ["revenue"]},
        ],
        "measures": [], "dimensions": [], "date_roles": [],
    }],
    "relationships": [],
    "date_roles": [],
}

_METRICS = [{"id": 1, "name": "Revenue", "synonyms": "sales", "sql_template": "SUM(NET_AMT)",
             "formula_type": "expression", "is_active": 1}]
_GRAPH = {"entities": [{"entity_name": "Customer", "table_name": "DIM_CUSTOMER"}],
          "relationships": [], "properties": []}
_TERMS = [{"id": 7, "term": "active customer", "aliases": "", "definition": "",
           "canonical_expression": "STATUS='A'", "tables_involved": ""}]
_DATE_CONTEXTS = [{"id": 3, "metric_id": 1, "metric_name": "Revenue",
                   "context_name": "Sales", "date_role": "invoice_date",
                   "fact_table": "ERP.SALES_FCT", "fact_column": "INVOICE_DATE_KEY",
                   "dimension_table": "ERP.DIM_DATE", "dimension_key": "DATE_KEY",
                   "date_value_column": "FULL_DATE", "is_default": 1}]


def _compile_patched(kb_dir: str, model=_MODEL, metrics=_METRICS, graph=_GRAPH, terms=_TERMS,
                     date_contexts=_DATE_CONTEXTS):
    from core.semantic_contract import compile_contract
    with patch("store.list_metrics", return_value=metrics), \
         patch("store.list_metric_date_contexts", return_value=date_contexts), \
         patch("store.get_full_graph", return_value=graph), \
         patch("store.list_terms", return_value=terms), \
         patch("core.field_overrides.load_field_overrides", return_value={}), \
         patch("core.semantic_model.load_semantic_model", return_value=model):
        return compile_contract("acct-test", kb_dir)


class ContractCompileTests(unittest.TestCase):
    def test_deterministic_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            c1 = _compile_patched(tmp)
            c2 = _compile_patched(tmp)
        self.assertEqual(
            c1["meta"]["contract_version"], c2["meta"]["contract_version"]
        )
        self.assertTrue(len(c1["meta"]["contract_version"]) == 12)

    def test_any_source_change_bumps_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = _compile_patched(tmp)["meta"]["contract_version"]
            changed_metric = _compile_patched(
                tmp, metrics=[{**_METRICS[0], "sql_template": "SUM(GROSS_AMT)"}]
            )["meta"]["contract_version"]
            changed_term = _compile_patched(
                tmp, terms=[{**_TERMS[0], "canonical_expression": "STATUS='ACTIVE'"}]
            )["meta"]["contract_version"]
            changed_graph = _compile_patched(
                tmp, graph={**_GRAPH, "entities": []}
            )["meta"]["contract_version"]
            changed_date_context = _compile_patched(
                tmp, date_contexts=[{**_DATE_CONTEXTS[0], "fact_column": "POSTING_DATE_KEY"}]
            )["meta"]["contract_version"]
        self.assertNotEqual(base, changed_metric)
        self.assertNotEqual(base, changed_term)
        self.assertNotEqual(base, changed_graph)
        self.assertNotEqual(base, changed_date_context)
        self.assertNotEqual(changed_metric, changed_term)

    def test_write_load_roundtrip_and_cache(self):
        from core.semantic_contract import load_contract, contract_fingerprint, contract_path
        with tempfile.TemporaryDirectory() as tmp:
            contract = _compile_patched(tmp)
            contract_path(tmp).write_text(json.dumps(contract), encoding="utf-8")
            loaded = load_contract(tmp)
            self.assertEqual(loaded["meta"]["contract_version"],
                             contract["meta"]["contract_version"])
            self.assertEqual(contract_fingerprint(tmp),
                             contract["meta"]["contract_version"])

    def test_missing_contract_is_empty_dict(self):
        from core.semantic_contract import load_contract, contract_fingerprint
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_contract(tmp), {})
            self.assertEqual(contract_fingerprint(tmp), "")
        self.assertEqual(load_contract(""), {})


class ApprovalHookTests(unittest.TestCase):
    def test_routes_call_after_semantic_approval(self):
        src = _src("admin/routes.py")
        # one shared helper, called from every semantic approval surface
        self.assertIn("def _after_semantic_approval(", src)
        self.assertGreaterEqual(src.count("_after_semantic_approval(account_id"), 20)
        # helper recompiles the contract AND fires evals
        helper = src[src.index("def _after_semantic_approval("):]
        helper = helper[:helper.index("\ndef ")]
        self.assertIn("recompile_contract", helper)
        self.assertIn("_run_default_evals_async", helper)

    def test_kb_build_records_contract_baseline(self):
        src = _src("admin/routes.py")
        self.assertIn('"kb_built_contract_version": _kb_contract_version', src)
        ksrc = _src("core/knowledge.py")
        self.assertIn("write_contract(account_id, kb_dir)", ksrc)


class GraphCanvasApprovalHookTests(unittest.TestCase):
    """The canvas/JSON graph-editor API mutates entity_graph/entity_relationships
    directly and was missed by the original hook-wiring pass — found by a
    fresh audit. Asserts each route's function body calls the shared hook
    (search is whitespace-tolerant since some calls wrap across lines)."""

    _ROUTE_FUNCS = [
        "graph_api_entity_upsert",
        "graph_api_entity_delete",
        "graph_api_entity_type",
        "graph_api_entity_filter",
        "graph_api_rel_upsert",
        "graph_api_rel_delete",
        "graph_api_rel_bulk",
        "graph_api_purge_audit_joins",
        "graph_api_prop_save",
        "graph_suggest",
    ]

    def _function_body(self, src: str, fn_name: str) -> str:
        marker = f"async def {fn_name}("
        start = src.index(marker)
        # body runs until the next top-level route decorator or function def
        rest = src[start:]
        next_route = rest.find("\n@router.", 1)
        next_def = rest.find("\nasync def ", 1)
        candidates = [x for x in (next_route, next_def) if x != -1]
        end = min(candidates) if candidates else len(rest)
        return rest[:end]

    def test_every_canvas_route_calls_the_hook(self):
        src = _src("admin/routes.py")
        for fn_name in self._ROUTE_FUNCS:
            body = self._function_body(src, fn_name)
            self.assertIn(
                "_after_semantic_approval(", body,
                f"{fn_name} does not recompile the semantic contract",
            )

    def test_position_and_validation_routes_are_exempt(self):
        # Cosmetic canvas coordinates and read-only validation diagnostics
        # must NOT trigger a recompile+eval run on every drag/click.
        src = _src("admin/routes.py")
        for fn_name in ("graph_api_entity_position", "graph_api_rel_validate_all",
                        "graph_api_rel_validate"):
            body = self._function_body(src, fn_name)
            self.assertNotIn("_after_semantic_approval(", body, fn_name)

    def test_no_duplicate_rel_bulk_handler(self):
        src = _src("admin/routes.py")
        self.assertEqual(src.count("async def graph_api_rel_bulk("), 1)


class EntityPropertyTermSyncTests(unittest.TestCase):
    """graph_api_prop_save called store.save_term(column_name=, table_hint=,
    is_active=) — none of which exist on save_term's real signature — so the
    'sync to semantic layer business terms' path raised TypeError on every
    call and was silently swallowed. Verify the corrected call actually
    succeeds and produces a listable term."""

    def test_route_uses_real_save_term_kwargs(self):
        src = _src("admin/routes.py")
        body = src[src.index("async def graph_api_prop_save("):]
        body = body[:body.index("\n@router.")]
        self.assertNotIn("column_name  = data[", body)
        self.assertNotIn("table_hint", body)
        self.assertIn("canonical_expression", body)
        self.assertIn("tables_involved", body)
        # "entity_graph" is not a legal business_term.source (CHECK
        # constraint allows only manual/kb_extracted/metric_registry) —
        # the route must use one of the real allowed values.
        self.assertNotIn('source          = "entity_graph"', body)
        self.assertIn('source          = "manual"', body)

    def test_corrected_save_term_call_succeeds_and_is_listable(self):
        from store.semantic_store import save_term, list_terms
        import store
        store.init_db()
        account_id = "acct-entity-prop-sync-test"
        store.upsert_client(account_id, "portal")
        # Same call shape graph_api_prop_save now makes.
        term_id = save_term(
            account_id=account_id,
            term="Customer Region",
            canonical_expression="REGION_CD",
            tables_involved="ERP.DIM_CUSTOMER",
            source="manual",
        )
        self.assertIsInstance(term_id, int)
        terms = list_terms(account_id, active_only=True)
        self.assertTrue(any(t["term"] == "customer region" for t in terms))

    def test_entity_graph_is_not_a_legal_source_value(self):
        # Documents the CHECK constraint so a future change to the allowed
        # source set is a deliberate decision, not a silent reintroduction
        # of this bug.
        from store.semantic_store import save_term
        import store
        store.init_db()
        account_id = "acct-entity-prop-sync-constraint-test"
        store.upsert_client(account_id, "portal")
        with self.assertRaises(Exception):
            save_term(account_id=account_id, term="Bad Source Term", source="entity_graph")


class ConsumerRepointingTests(unittest.TestCase):
    def test_pipeline_loads_contract_once_and_threads_sections(self):
        src = _src("core/query_pipeline.py")
        self.assertIn("_contract = load_contract(state.get(\"kb_dir\", \"\"))", src)
        self.assertIn("terms=_contract_terms", src)
        self.assertIn("metrics=_contract_metrics", src)
        self.assertEqual(src.count("model=_contract_model"), 2)  # context + plan
        self.assertIn('_contract.get("graph")', src)
        self.assertIn("contract_version=_contract_version", src)

    def test_plan_builder_accepts_model_override(self):
        from core.semantic_model import build_runtime_semantic_plan
        plan = build_runtime_semantic_plan(
            "Z:/nonexistent-kb-dir", question="net revenue by customer",
            model=_MODEL,
        )
        self.assertTrue(plan.get("enabled"))
        fields = {(f["table"], f["column"]) for f in plan.get("fields", [])}
        self.assertIn(("ERP.SALES_FCT", "NET_AMT"), fields)

    def test_metric_context_accepts_metrics_override(self):
        from store.config_store import list_metric_formula_context
        out = list_metric_formula_context(
            "no-such-account", "total revenue this year", metrics=_METRICS,
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["name"], "Revenue")

    def test_term_matching_accepts_terms_override(self):
        from store.semantic_store import match_terms_in_question
        out = match_terms_in_question(
            "no-such-account", "show active customer count", terms=_TERMS,
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["term"], "active customer")


class VersionStampTests(unittest.TestCase):
    def test_learning_versions_include_contract(self):
        from core.pipeline_trace import compute_learning_versions
        with tempfile.TemporaryDirectory() as tmp:
            from core.semantic_contract import contract_path
            contract = _compile_patched(tmp)
            contract_path(tmp).write_text(json.dumps(contract), encoding="utf-8")
            versions = compute_learning_versions("acct-test", kb_dir=tmp)
        self.assertEqual(versions["contract_version"],
                         contract["meta"]["contract_version"])

    def test_metric_save_writes_version_snapshot(self):
        src = _src("store/config_store.py")
        self.assertIn("def _snapshot_metric_version(", src)
        # both the insert and the content-changing update snapshot
        self.assertGreaterEqual(src.count("_snapshot_metric_version(conn,"), 2)

    def test_trace_allowlist_includes_contract_version(self):
        src = _src("store/trace_store.py")
        self.assertIn('"contract_version"', src)


class EvalGateTests(unittest.TestCase):
    def test_health_score_reads_passed_cases(self):
        src = _src("admin/routes.py")
        self.assertNotIn('eval_run.get("pass_count")', src)
        self.assertIn('eval_run.get("passed_cases")', src)

    def test_run_suite_records_regression_fields(self):
        src = _src("evals/run.py")
        self.assertIn("previous_eval_run(account_id", src)
        self.assertIn("regressed = pass_rate < prev_pass_rate", src)
        self.assertIn("trigger_label=trigger", src)
        self.assertIn("contract_version=contract_version", src)

    def test_latest_regressed_run_clears_after_recovery(self):
        # A regressed run followed by a newer clean run of the same case
        # file must NOT keep the banner up. Uses the shared test DB with a
        # dedicated account — purging store modules mid-suite creates a
        # second connection pool and "database is locked" flakes.
        import uuid
        import store
        store.init_db()
        acct = f"acct-evregr-{uuid.uuid4().hex[:8]}"
        store.upsert_client(acct, "portal")
        kw = dict(account_id=acct, schema_name="s", case_file="f.yaml",
                  total_cases=10)
        store.save_eval_run(**kw, passed_cases=9)
        store.save_eval_run(**kw, passed_cases=6, prev_pass_rate=0.9,
                            regressed=True, trigger_label="metric 'X' updated")
        hit = store.latest_regressed_run(acct)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["trigger_label"], "metric 'X' updated")
        store.save_eval_run(**kw, passed_cases=10, prev_pass_rate=0.6)
        self.assertIsNone(store.latest_regressed_run(acct))


class GoldenSeedTests(unittest.TestCase):
    def test_extract_tables(self):
        from evals.seed import extract_tables_from_sql
        sql = ("SELECT c.NAME, SUM(s.NET_AMT) FROM ERP.SALES_FCT s "
               "JOIN [ERP].[DIM_CUSTOMER] c ON s.CUST_KEY=c.CUST_KEY "
               "LEFT JOIN dbo.region r ON r.id=c.region_id")
        self.assertEqual(
            extract_tables_from_sql(sql),
            ["ERP.SALES_FCT", "ERP.DIM_CUSTOMER", "DBO.REGION"],
        )

    def test_seed_merges_never_clobbers(self):
        import yaml
        from evals import seed as seed_mod
        harvested = {
            "HR": [
                {"id": "auto_aaa", "question": "headcount by dept",
                 "generated_sql": "SELECT 1 FROM HR.EMP", "expected_tables": ["HR.EMP"],
                 "min_score": 0.85},
                {"id": "auto_bbb", "question": "attrition this year",
                 "generated_sql": "SELECT 1 FROM HR.EXIT", "expected_tables": ["HR.EXIT"],
                 "min_score": 0.85},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(seed_mod, "harvest_golden_cases", return_value=harvested), \
             patch.object(seed_mod.Path, "cwd"):
            # Redirect the evals/ root into the tempdir
            real_path = seed_mod.Path
            target_dir = Path(tmp) / "evals" / "clients" / "acct-g" / "HR"
            target_dir.mkdir(parents=True)
            hand_case = {"id": "manual_1", "question": "headcount by dept",
                         "expected_tables": ["HR.EMPLOYEES"], "min_score": 0.9}
            (target_dir / "golden_questions.yaml").write_text(
                yaml.safe_dump({"cases": [hand_case]}), encoding="utf-8")
            import os
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                summary = seed_mod.seed_golden_suite("acct-g")
            finally:
                os.chdir(old_cwd)
            data = yaml.safe_load((target_dir / "golden_questions.yaml").read_text(encoding="utf-8"))
            cases = data["cases"]
        ids = [c["id"] for c in cases]
        # hand case untouched and still first
        self.assertEqual(cases[0], hand_case)
        # duplicate question ("headcount by dept") skipped, new one merged
        self.assertNotIn("auto_aaa", ids)
        self.assertIn("auto_bbb", ids)
        self.assertEqual(summary["added"], 1)
        self.assertEqual(summary["skipped_existing"], 1)


class OfflineEvalTests(unittest.TestCase):
    def test_cases_with_sql_never_call_llm(self):
        # run_eval_suite only generates when a case has no generated_sql
        src = _src("evals/run.py")
        self.assertIn("if not sql and generate:", src)


if __name__ == "__main__":
    unittest.main()
