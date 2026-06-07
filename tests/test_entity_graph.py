"""
tests/test_entity_graph.py

Tests for the structured entity graph — join resolver pipeline:
  1. DB tables created on init_db()
  2. CRUD — save_entity / list_entities / get_entity / delete_entity
  3. CRUD — save_relationship / list_relationships / delete_relationship
  4. CRUD — save_entity_property / list_entity_properties
  5. get_full_graph() structure
  6. graph_resolver — entity detection
  7. graph_resolver — BFS pathfinder
  8. graph_resolver — JOIN skeleton builder (all 3 DB types)
  9. graph_resolver — resolve_for_question() public API
 10. SQL prompt — graph_context injection
 11. main.py wiring guards
 12. Admin routes wired
"""
import os, sys, tempfile, unittest

_TMP = os.path.join(tempfile.mkdtemp(), "test_graph.db")
os.environ["QUERYBOT_DB_PATH"] = _TMP
for mod in list(sys.modules.keys()):
    if mod.startswith("store"):
        del sys.modules[mod]
import store.db as _db
_db.init_db()
import store

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
LLM_PY           = ROOT / "core" / "llm.py"
MAIN_PY          = ROOT / "main.py"
QUERY_PIPELINE   = ROOT / "core" / "query_pipeline.py"
ROUTES           = ROOT / "admin" / "routes.py"
GRAPH_TMPL       = ROOT / "admin" / "templates" / "client_graph.html"


# ── helpers ───────────────────────────────────────────────────────────────────
def _seed_graph(account_id: str) -> None:
    """Seed a small pharmacy schema graph."""
    store.save_entity(account_id, "Prescription", "FACT_RXFILL", schema_name="dbo",
                      pk_column="RxID", entity_type="fact")
    store.save_entity(account_id, "Customer",     "DIM_CUSTOMER", schema_name="dbo",
                      pk_column="CustomerID", entity_type="dimension")
    store.save_entity(account_id, "Drug",         "DIM_DRUG",      schema_name="dbo",
                      pk_column="DrugCode",   entity_type="dimension")
    store.save_relationship(account_id,
        from_entity="Prescription", from_column="CustomerID",
        to_entity="Customer",       to_column="CustomerID",
        relationship_type="many_to_one", join_type="INNER")
    store.save_relationship(account_id,
        from_entity="Prescription", from_column="DrugCode",
        to_entity="Drug",           to_column="DrugCode",
        relationship_type="many_to_one", join_type="LEFT")


# ══════════════════════════════════════════════════════════════════════════════
# 1  DB tables
# ══════════════════════════════════════════════════════════════════════════════
class TestDBTables(unittest.TestCase):

    def _exists(self, name):
        with _db.get_db() as conn:
            r = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
            ).fetchone()
        return r is not None

    def test_entity_graph_table(self):
        self.assertTrue(self._exists("entity_graph"))

    def test_entity_relationships_table(self):
        self.assertTrue(self._exists("entity_relationships"))

    def test_entity_properties_table(self):
        self.assertTrue(self._exists("entity_properties"))

    def test_entity_graph_columns(self):
        with _db.get_db() as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(entity_graph)").fetchall()}
        for c in ("account_id","entity_name","table_name","schema_name",
                  "pk_column","entity_type","is_active"):
            self.assertIn(c, cols)

    def test_relationships_columns(self):
        with _db.get_db() as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(entity_relationships)").fetchall()}
        for c in ("from_entity","to_entity","from_column","to_column",
                  "join_type","relationship_type"):
            self.assertIn(c, cols)


# ══════════════════════════════════════════════════════════════════════════════
# 2  Entity CRUD
# ══════════════════════════════════════════════════════════════════════════════
class TestEntityCRUD(unittest.TestCase):

    ACC = "test_ent_001"

    def test_save_and_get(self):
        store.save_entity(self.ACC, "Customer", "DIM_CUSTOMER",
                          schema_name="dbo", pk_column="CustomerID",
                          entity_type="dimension")
        e = store.get_entity(self.ACC, "Customer")
        self.assertIsNotNone(e)
        self.assertEqual(e["table_name"], "DIM_CUSTOMER")
        self.assertEqual(e["entity_type"], "dimension")

    def test_upsert_updates(self):
        store.save_entity(self.ACC, "Drug", "DIM_DRUG_OLD", entity_type="dimension")
        store.save_entity(self.ACC, "Drug", "DIM_DRUG_NEW", entity_type="fact")
        e = store.get_entity(self.ACC, "Drug")
        self.assertEqual(e["table_name"], "DIM_DRUG_NEW")
        self.assertEqual(e["entity_type"], "fact")

    def test_list_returns_list(self):
        store.save_entity(self.ACC, "Prescription", "FACT_RXFILL", entity_type="fact")
        entities = store.list_entities(self.ACC)
        self.assertIsInstance(entities, list)
        names = [e["entity_name"] for e in entities]
        self.assertIn("Prescription", names)

    def test_delete_removes_entity_and_rels(self):
        store.save_entity(self.ACC, "TempEnt", "TEMP_TBL", entity_type="dimension")
        store.save_relationship(self.ACC, "Prescription", "TempID",
                                "TempEnt", "TempID")
        store.delete_entity(self.ACC, "TempEnt")
        self.assertIsNone(store.get_entity(self.ACC, "TempEnt"))
        rels = store.list_relationships(self.ACC)
        self.assertFalse(any(r["to_entity"] == "TempEnt" or r["from_entity"] == "TempEnt"
                             for r in rels))

    def test_isolation_by_account(self):
        store.save_entity(self.ACC, "AccA_Ent", "TBL_A", entity_type="dimension")
        others = store.list_entities("completely_other_account")
        self.assertFalse(any(e["entity_name"] == "AccA_Ent" for e in others))


# ══════════════════════════════════════════════════════════════════════════════
# 3  Relationship CRUD
# ══════════════════════════════════════════════════════════════════════════════
class TestRelationshipCRUD(unittest.TestCase):

    ACC = "test_rel_002"

    def setUp(self):
        _seed_graph(self.ACC)

    def test_list_returns_all(self):
        rels = store.list_relationships(self.ACC)
        self.assertGreaterEqual(len(rels), 2)

    def test_rel_fields_present(self):
        rels = store.list_relationships(self.ACC)
        r = next(r for r in rels if r["to_entity"] == "Customer")
        self.assertEqual(r["from_entity"], "Prescription")
        self.assertEqual(r["from_column"], "CustomerID")
        self.assertEqual(r["to_column"],   "CustomerID")
        self.assertEqual(r["join_type"],   "INNER")

    def test_delete_relationship(self):
        rels = store.list_relationships(self.ACC)
        rid  = rels[0]["id"]
        store.delete_relationship(self.ACC, rid)
        rels2 = store.list_relationships(self.ACC)
        self.assertFalse(any(r["id"] == rid for r in rels2))

    def test_left_join_stored(self):
        rels = store.list_relationships(self.ACC)
        left = [r for r in rels if r["join_type"] == "LEFT"]
        self.assertGreaterEqual(len(left), 1)


# ══════════════════════════════════════════════════════════════════════════════
# 4  Properties CRUD
# ══════════════════════════════════════════════════════════════════════════════
class TestPropertiesCRUD(unittest.TestCase):

    ACC = "test_prop_003"

    def test_save_and_list(self):
        store.save_entity_property(self.ACC, "Customer", "Revenue",
                                   role="metric", display_name="Total Revenue")
        props = store.list_entity_properties(self.ACC, "Customer")
        r = next((p for p in props if p["column_name"] == "Revenue"), None)
        self.assertIsNotNone(r)
        self.assertEqual(r["role"], "metric")
        self.assertEqual(r["display_name"], "Total Revenue")

    def test_upsert_updates_role(self):
        store.save_entity_property(self.ACC, "Customer", "Segment",
                                   role="dimension")
        store.save_entity_property(self.ACC, "Customer", "Segment",
                                   role="filter")
        props = store.list_entity_properties(self.ACC, "Customer")
        r = next(p for p in props if p["column_name"] == "Segment")
        self.assertEqual(r["role"], "filter")


# ══════════════════════════════════════════════════════════════════════════════
# 5  get_full_graph
# ══════════════════════════════════════════════════════════════════════════════
class TestGetFullGraph(unittest.TestCase):

    ACC = "test_full_004"

    def setUp(self):
        _seed_graph(self.ACC)

    def test_returns_entities_and_relationships(self):
        g = store.get_full_graph(self.ACC)
        self.assertIn("entities", g)
        self.assertIn("relationships", g)
        self.assertIsInstance(g["entities"], list)
        self.assertIsInstance(g["relationships"], list)

    def test_entity_count(self):
        g = store.get_full_graph(self.ACC)
        self.assertGreaterEqual(len(g["entities"]), 3)

    def test_relationship_count(self):
        g = store.get_full_graph(self.ACC)
        self.assertGreaterEqual(len(g["relationships"]), 2)


# ══════════════════════════════════════════════════════════════════════════════
# 6  Entity detection
# ══════════════════════════════════════════════════════════════════════════════
class TestEntityDetection(unittest.TestCase):

    ACC = "test_detect_005"

    def setUp(self):
        _seed_graph(self.ACC)
        self._graph = store.get_full_graph(self.ACC)

    def _detect(self, q):
        from core.graph_resolver import detect_entities
        return detect_entities(q, self._graph)

    def test_detects_customer(self):
        found = self._detect("show revenue by customer")
        self.assertIn("Customer", found)

    def test_detects_drug(self):
        found = self._detect("show fill rate by drug category")
        self.assertIn("Drug", found)

    def test_detects_prescription(self):
        found = self._detect("show prescription count by month")
        self.assertIn("Prescription", found)

    def test_detects_multiple(self):
        found = self._detect("show revenue by customer and drug")
        self.assertIn("Customer", found)
        self.assertIn("Drug", found)

    def test_empty_question_returns_fact_or_empty(self):
        from core.graph_resolver import detect_entities
        found = detect_entities("", self._graph)
        # Should return [] or just the fact entity fallback
        self.assertIsInstance(found, list)

    def test_returns_list(self):
        found = self._detect("revenue by segment")
        self.assertIsInstance(found, list)


# ══════════════════════════════════════════════════════════════════════════════
# 7  BFS pathfinder
# ══════════════════════════════════════════════════════════════════════════════
class TestBFSPathfinder(unittest.TestCase):

    ACC = "test_bfs_006"

    def setUp(self):
        _seed_graph(self.ACC)
        self._graph = store.get_full_graph(self.ACC)

    def _path(self, entities):
        from core.graph_resolver import find_join_path
        return find_join_path(entities, self._graph)

    def test_two_entity_path(self):
        path = self._path(["Prescription", "Customer"])
        self.assertGreater(len(path), 0)

    def test_three_entity_path(self):
        path = self._path(["Prescription", "Customer", "Drug"])
        self.assertGreater(len(path), 0)

    def test_path_has_required_fields(self):
        path = self._path(["Prescription", "Customer"])
        self.assertTrue(len(path) > 0)
        step = path[0]
        for field in ("from_entity", "to_entity", "from_column", "to_column", "_direction"):
            self.assertIn(field, step)

    def test_single_entity_returns_empty(self):
        path = self._path(["Customer"])
        self.assertEqual(path, [])

    def test_returns_list(self):
        path = self._path(["Prescription", "Drug"])
        self.assertIsInstance(path, list)


# ══════════════════════════════════════════════════════════════════════════════
# 8  JOIN skeleton builder — all 3 DB types
# ══════════════════════════════════════════════════════════════════════════════
class TestJoinSkeletonBuilder(unittest.TestCase):

    ACC = "test_skeleton_007"

    def setUp(self):
        _seed_graph(self.ACC)
        g = store.get_full_graph(self.ACC)
        self._emap = {e["entity_name"]: e for e in g["entities"]}
        from core.graph_resolver import find_join_path
        self._path = find_join_path(["Prescription", "Customer"], g)

    def _build(self, db_type):
        from core.graph_resolver import build_join_skeleton
        return build_join_skeleton(self._path, self._emap, "Prescription", db_type)

    def test_azure_sql_uses_brackets(self):
        s = self._build("azure_sql")
        self.assertIn("[dbo].[FACT_RXFILL]", s)
        self.assertIn("[CustomerID]", s)

    def test_azure_sql_has_from(self):
        s = self._build("azure_sql")
        self.assertTrue(s.startswith("FROM "))

    def test_azure_sql_has_join(self):
        s = self._build("azure_sql")
        self.assertIn("JOIN", s)

    def test_snowflake_uses_quotes(self):
        s = self._build("snowflake")
        self.assertIn('"dbo"."FACT_RXFILL"', s)

    def test_left_join_keyword_present(self):
        # The Prescription→Drug relationship is LEFT JOIN
        g = store.get_full_graph(self.ACC)
        from core.graph_resolver import find_join_path, build_join_skeleton
        path = find_join_path(["Prescription", "Drug"], g)
        emap = {e["entity_name"]: e for e in g["entities"]}
        s    = build_join_skeleton(path, emap, "Prescription", "azure_sql")
        self.assertIn("LEFT", s)

    def test_no_duplicate_tables(self):
        s = self._build("azure_sql")
        # Each entity should appear once in the FROM/JOIN chain
        self.assertEqual(s.count("[FACT_RXFILL]"), 1)
        self.assertEqual(s.count("[DIM_CUSTOMER]"), 1)

    def test_returns_string(self):
        self.assertIsInstance(self._build("azure_sql"), str)


# ══════════════════════════════════════════════════════════════════════════════
# 9  resolve_for_question — public API
# ══════════════════════════════════════════════════════════════════════════════
class TestResolveForQuestion(unittest.TestCase):

    ACC = "test_resolve_008"

    def setUp(self):
        _seed_graph(self.ACC)

    def _resolve(self, q, db="azure_sql"):
        from core.graph_resolver import resolve_for_question
        graph = store.get_full_graph(self.ACC)
        return resolve_for_question(q, self.ACC, db, graph=graph)

    def test_returns_dict(self):
        r = self._resolve("show revenue by customer")
        self.assertIsInstance(r, dict)

    def test_enabled_for_matching_question(self):
        r = self._resolve("show revenue by customer")
        self.assertTrue(r["enabled"])

    def test_join_skeleton_not_empty(self):
        r = self._resolve("show revenue by customer")
        self.assertGreater(len(r.get("join_skeleton", "")), 0)

    def test_detected_entities_list(self):
        r = self._resolve("show fill rate by drug")
        self.assertIsInstance(r["detected"], list)
        self.assertIn("Drug", r["detected"])

    def test_entity_count_returned(self):
        r = self._resolve("anything")
        self.assertGreaterEqual(r.get("entity_count", 0), 3)

    def test_disabled_on_empty_graph(self):
        from core.graph_resolver import resolve_for_question
        r = resolve_for_question("show revenue", "empty_account_xyz", "azure_sql",
                                  graph={"entities": [], "relationships": []})
        self.assertFalse(r["enabled"])

    def test_no_extra_llm_call_needed(self):
        """Resolver must work entirely from graph data — no external calls."""
        import inspect
        from core import graph_resolver
        src = inspect.getsource(graph_resolver)
        # There must be no openai / anthropic import in the resolver
        self.assertNotIn("openai", src)
        self.assertNotIn("anthropic", src)

    def test_single_entity_returns_from_only(self):
        """When only one entity is matched, return a simple FROM clause."""
        from core.graph_resolver import resolve_for_question
        graph = store.get_full_graph(self.ACC)
        r = resolve_for_question("show revenue by month", self.ACC, "azure_sql", graph=graph)
        # Even with one entity, if enabled the skeleton must have FROM
        if r["enabled"]:
            self.assertIn("FROM", r["join_skeleton"])


# ══════════════════════════════════════════════════════════════════════════════
# 10  SQL prompt — graph_context injection
# ══════════════════════════════════════════════════════════════════════════════
class TestSQLPromptGraphInjection(unittest.TestCase):

    def _build_prompt(self, graph_ctx=None):
        from core.llm import build_sql_system_prompt
        return build_sql_system_prompt("azure_sql", "KB context", graph_context=graph_ctx)

    def test_no_graph_ctx_no_block(self):
        p = self._build_prompt()
        self.assertNotIn("Entity graph", p)

    def test_disabled_graph_no_block(self):
        p = self._build_prompt({"enabled": False, "join_skeleton": ""})
        self.assertNotIn("Entity graph", p)

    def test_enabled_graph_adds_block(self):
        ctx = {
            "enabled": True,
            "join_skeleton": "FROM [dbo].[FACT_RXFILL] f INNER JOIN [dbo].[DIM_CUSTOMER] c ON f.[CustomerID]=c.[CustomerID]",
            "detected": ["Prescription", "Customer"],
        }
        p = self._build_prompt(ctx)
        self.assertIn("Entity graph", p)
        self.assertIn("FACT_RXFILL", p)

    def test_join_skeleton_in_prompt(self):
        skeleton = "FROM [dbo].[FACT_RXFILL] f"
        p = self._build_prompt({"enabled": True, "join_skeleton": skeleton, "detected": []})
        self.assertIn(skeleton, p)

    def test_must_use_instruction_in_prompt(self):
        ctx = {"enabled": True, "join_skeleton": "FROM [dbo].[T] t", "detected": []}
        p = self._build_prompt(ctx)
        self.assertIn("MUST use this exact", p)

    def test_detected_entities_listed(self):
        ctx = {
            "enabled": True,
            "join_skeleton": "FROM [dbo].[T] t",
            "detected": ["Alpha", "Beta"],
        }
        p = self._build_prompt(ctx)
        self.assertIn("Alpha", p)
        self.assertIn("Beta", p)

    def test_prompt_still_returns_string(self):
        p = self._build_prompt()
        self.assertIsInstance(p, str)
        self.assertGreater(len(p), 50)

    def test_function_signature_has_graph_context(self):
        import inspect
        from core.llm import build_sql_system_prompt
        params = inspect.signature(build_sql_system_prompt).parameters
        self.assertIn("graph_context", params)


# ══════════════════════════════════════════════════════════════════════════════
# 11  main.py wiring
# ══════════════════════════════════════════════════════════════════════════════
class TestMainWiring(unittest.TestCase):

    def test_graph_resolver_imported(self):
        # Import lives in core/query_pipeline.py after the main.py split
        src = QUERY_PIPELINE.read_text(encoding="utf-8")
        self.assertIn("graph_resolver", src)

    def test_resolve_for_question_called(self):
        # Call site lives in core/query_pipeline.py after the main.py split
        src = QUERY_PIPELINE.read_text(encoding="utf-8")
        self.assertIn("_graph_resolve", src)

    def test_graph_ctx_passed_to_prompt(self):
        # Logic lives in core/query_pipeline.py after the main.py split
        src = QUERY_PIPELINE.read_text(encoding="utf-8")
        self.assertIn("graph_context=_graph_ctx", src)

    def test_graph_load_uses_store(self):
        # Logic lives in core/query_pipeline.py after the main.py split
        src = QUERY_PIPELINE.read_text(encoding="utf-8")
        self.assertIn("store.get_full_graph", src)

    def test_graph_failure_is_non_fatal(self):
        """Graph errors must be caught and fall back gracefully."""
        # Logic lives in core/query_pipeline.py after the main.py split
        src = QUERY_PIPELINE.read_text(encoding="utf-8")
        # The exception block for graph resolution
        self.assertIn("except Exception as _gex", src)
        self.assertIn("Graph resolution skipped", src)


# ══════════════════════════════════════════════════════════════════════════════
# 12  Admin routes + template
# ══════════════════════════════════════════════════════════════════════════════
class TestAdminRoutes(unittest.TestCase):

    def test_graph_page_route_exists(self):
        src = ROUTES.read_text(encoding="utf-8")
        self.assertIn('"/clients/{account_id}/graph"', src)

    def test_entity_create_route(self):
        src = ROUTES.read_text(encoding="utf-8")
        self.assertIn("graph/entities/create", src)

    def test_entity_delete_route(self):
        src = ROUTES.read_text(encoding="utf-8")
        self.assertIn("graph/entities/{entity_name}/delete", src)

    def test_relationship_create_route(self):
        src = ROUTES.read_text(encoding="utf-8")
        self.assertIn("graph/relationships/create", src)

    def test_relationship_delete_route(self):
        src = ROUTES.read_text(encoding="utf-8")
        self.assertIn("graph/relationships/{rel_id}/delete", src)

    def test_json_api_route(self):
        src = ROUTES.read_text(encoding="utf-8")
        self.assertIn("graph/api/graph.json", src)

    def test_resolve_api_route(self):
        src = ROUTES.read_text(encoding="utf-8")
        self.assertIn("graph/api/resolve", src)

    def test_template_exists(self):
        self.assertTrue(GRAPH_TMPL.exists())

    def test_template_has_svg_canvas(self):
        src = GRAPH_TMPL.read_text(encoding="utf-8")
        self.assertIn("graph-svg", src)
        self.assertIn("<svg", src)

    def test_template_has_entity_form(self):
        src = GRAPH_TMPL.read_text(encoding="utf-8")
        # new UI uses modal-based form
        self.assertIn("entity-modal", src)
        self.assertIn("m-entity-name", src)

    def test_template_has_relationship_form(self):
        src = GRAPH_TMPL.read_text(encoding="utf-8")
        self.assertIn("from_entity", src)
        self.assertIn("to_entity", src)
        self.assertIn("from_column", src)

    def test_template_has_resolver_test(self):
        src = GRAPH_TMPL.read_text(encoding="utf-8")
        self.assertIn("resolve-q", src)
        self.assertIn("resolve-output", src)

    def test_template_has_drag_support(self):
        src = GRAPH_TMPL.read_text(encoding="utf-8")
        # new UI uses startDrag + addEventListener pattern
        self.assertIn("startDrag", src)
        self.assertIn("mousedown", src)

    def test_template_renders_suggested_graph_rows(self):
        src = GRAPH_TMPL.read_text(encoding="utf-8")
        self.assertNotIn("if (e.status === 'suggested') return", src)
        self.assertNotIn("fe.status === 'suggested' || te.status === 'suggested'", src)
        self.assertIn("const isSuggested = (e.status === 'suggested')", src)
        self.assertNotIn("{% if e.status != 'suggested' %}", src)
        self.assertIn("gs-entity-item{% if e.status == 'suggested' %} suggested{% endif %}", src)
        self.assertIn("data-status=", src)

    def test_template_keeps_schema_selector_visible_for_one_schema(self):
        src = GRAPH_TMPL.read_text(encoding="utf-8")
        self.assertIn('id="schema-filter"', src)
        self.assertNotIn("schemas.length < 2 ? 'none'", src)
        self.assertNotIn("schemas.length <= 1) { sel.style.display = 'none'", src)
        self.assertIn("fallback.length === 1 ? fallback[0]", src)

    def test_template_keeps_entity_type_tabs_filterable_after_rebuild(self):
        src = GRAPH_TMPL.read_text(encoding="utf-8")
        self.assertIn("window.filterSidebarByType", src)
        self.assertIn('data-etype="${esc(e.entity_type)}"', src)
        self.assertIn("_applySidebarEntFilter", src)
        self.assertIn("gs-ent-type-label", src)

    def test_template_uses_persistent_three_pane_model_workbench(self):
        src = GRAPH_TMPL.read_text(encoding="utf-8")
        self.assertIn("grid-template-columns:minmax(420px,1fr) var(--graph-inspector-width)", src)
        self.assertIn("Persistent model inspector", src)
        self.assertIn('id="graph-drawer"', src)
        self.assertNotIn(".graph-shell.drawer-open .graph-drawer{height:", src)

    def test_template_has_business_physical_and_query_path_lenses(self):
        src = GRAPH_TMPL.read_text(encoding="utf-8")
        self.assertIn("setGraphView('business')", src)
        self.assertIn("setGraphView('physical')", src)
        self.assertIn("setGraphView('query')", src)
        self.assertIn("queryPathEntities", src)
        self.assertIn("_fitEntitySet(queryPathEntities)", src)

    def test_template_keeps_inspector_actions_visible_when_selected(self):
        src = GRAPH_TMPL.read_text(encoding="utf-8")
        self.assertIn("bar.classList.toggle('open'", src)
        self.assertIn('id="drawer-action-bar" class="gd-action-bar"', src)
        self.assertIn("Validate joins", src)
        self.assertIn("Live probe", src)

    def test_setup_page_has_graph_nav(self):
        tmpl = (ROOT / "admin" / "templates" / "client_setup.html").read_text(encoding="utf-8")
        self.assertIn("/graph", tmpl)
        self.assertIn("Entity Graph", tmpl)


class TestEntityGraphScopePruning(unittest.TestCase):

    ACC = "test_graph_prune_013"

    def test_prune_removes_entities_outside_current_schema_scope(self):
        store.save_entity(self.ACC, "Invoice", "CUS_ORD_IVC_FCT",
                          schema_name="PROFITABILITY", entity_type="fact")
        store.save_entity(self.ACC, "Invoice Date", "DT_DMS",
                          schema_name="PROFITABILITY", entity_type="dimension")
        store.save_entity(self.ACC, "Patient", "DIM_PATIENT",
                          schema_name="PHARMACY", entity_type="dimension")
        store.save_entity_property(self.ACC, "Invoice", "CUS_IVC_LIN_AMT",
                                   role="metric", display_name="Revenue")
        store.save_entity_property(self.ACC, "Patient", "AGE",
                                   role="dimension", display_name="Age")
        store.save_relationship(
            self.ACC,
            from_entity="Invoice",
            to_entity="Invoice Date",
            from_column="CUS_IVC_DT_DMS_KEY",
            to_column="DT_DMS_KEY",
            join_type="LEFT",
        )
        store.save_relationship(
            self.ACC,
            from_entity="Invoice",
            to_entity="Patient",
            from_column="PATIENT_ID",
            to_column="PATIENT_ID",
            join_type="LEFT",
        )

        removed = store.prune_entity_graph_to_tables(
            self.ACC,
            {
                "CHATBOTDB.PROFITABILITY.CUS_ORD_IVC_FCT",
                "CHATBOTDB.PROFITABILITY.DT_DMS",
            },
        )

        self.assertEqual(removed["entities_removed"], 1)
        self.assertIsNone(store.get_entity(self.ACC, "Patient"))
        self.assertIsNotNone(store.get_entity(self.ACC, "Invoice"))
        self.assertIsNotNone(store.get_entity(self.ACC, "Invoice Date"))
        rels = store.list_relationships(self.ACC)
        self.assertTrue(any(r["to_entity"] == "Invoice Date" for r in rels))
        self.assertFalse(any(r["to_entity"] == "Patient" or r["from_entity"] == "Patient"
                             for r in rels))
        self.assertTrue(store.list_entity_properties(self.ACC, "Invoice"))
        self.assertFalse(store.list_entity_properties(self.ACC, "Patient"))


if __name__ == "__main__":
    unittest.main()
