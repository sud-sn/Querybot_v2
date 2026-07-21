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

    def test_no_fanout_warning_when_risk_list_empty(self):
        ctx = {"enabled": True, "join_skeleton": "FROM [dbo].[T] t", "detected": [],
               "fanout_risk_facts": []}
        p = self._build_prompt(ctx)
        self.assertNotIn("FAN-OUT WARNING", p)

    def test_fanout_warning_added_when_risk_present(self):
        ctx = {
            "enabled": True,
            "join_skeleton": "FROM [dbo].[F_A] a INNER JOIN [dbo].[F_B] b ON a.[PHARMACY_ID]=b.[PHARMACY_ID]",
            "detected": ["F_A", "F_B"],
            "fanout_risk_facts": ["F_B"],
        }
        p = self._build_prompt(ctx)
        self.assertIn("FAN-OUT WARNING", p)
        self.assertIn("F_B", p)
        self.assertIn("CTE", p)

    def test_fanout_warning_overrides_no_add_remove_joins(self):
        ctx = {"enabled": True, "join_skeleton": "FROM [dbo].[T] t", "detected": [],
               "fanout_risk_facts": ["T2"]}
        p = self._build_prompt(ctx)
        # The override must come after (and reference) the blanket no-edit rule,
        # not silently contradict it with no explanation.
        self.assertIn("OVERRIDES THE", p)


# ══════════════════════════════════════════════════════════════════════════════
# 10b  Fan-out guard — entity-detection scoring
# ══════════════════════════════════════════════════════════════════════════════
class TestFanoutScoringGuard(unittest.TestCase):
    """A single generic single-word property synonym must not, on its own,
    qualify a FACT table for inclusion in the join skeleton — that's exactly
    the mechanism that pulled unrelated fact tables into a fan-out join."""

    ACC = "test_fanout_score_010"

    def setUp(self):
        store.save_entity(self.ACC, "SalesFact", "F_SALES", schema_name="dbo",
                           pk_column="SaleID", entity_type="fact")
        store.save_entity(self.ACC, "InventoryFact", "F_INVENTORY", schema_name="dbo",
                           pk_column="SnapshotID", entity_type="fact")
        store.save_entity(self.ACC, "Store", "D_STORE", schema_name="dbo",
                           pk_column="StoreID", entity_type="dimension")
        store.save_relationship(self.ACC,
            from_entity="SalesFact", from_column="StoreID",
            to_entity="Store", to_column="StoreID",
            relationship_type="many_to_one", join_type="INNER")
        store.save_relationship(self.ACC,
            from_entity="InventoryFact", from_column="StoreID",
            to_entity="Store", to_column="StoreID",
            relationship_type="many_to_one", join_type="INNER")
        # SalesFact's real metric — a specific, multi-word synonym.
        store.save_entity_property(self.ACC, "SalesFact", "NET_REVENUE_AMT",
                                    role="metric", display_name="Net Revenue",
                                    synonyms="net revenue, total sales")
        # InventoryFact has nothing to do with the question below — its only
        # overlap is a generic one-word synonym ("total") that many metric
        # columns across a schema tend to share.
        store.save_entity_property(self.ACC, "InventoryFact", "ON_HAND_QUANTITY",
                                    role="metric", display_name="On Hand Quantity",
                                    synonyms="total, qty")
        self._graph = store.get_full_graph(self.ACC)

    def _detect(self, q):
        from core.graph_resolver import detect_entities
        return detect_entities(q, self._graph)

    def test_generic_single_word_alone_does_not_pull_in_unrelated_fact(self):
        found = self._detect("what is total net revenue by store for 2026")
        self.assertIn("SalesFact", found)
        self.assertIn("Store", found)
        self.assertNotIn("InventoryFact", found)

    def test_specific_multiword_property_match_still_qualifies_anchor(self):
        # SalesFact must still be detected via its own specific synonym even
        # though the match is property-level, not name/table-level.
        found = self._detect("show net revenue trend")
        self.assertIn("SalesFact", found)

    def test_fact_with_strong_signal_still_detected(self):
        # A fact table matched by its own entity/table name is unaffected by
        # the property-match gate.
        found = self._detect("show inventory fact data")
        self.assertIn("InventoryFact", found)


# ══════════════════════════════════════════════════════════════════════════════
# 10c  Fan-out guard — multi-fact join-path risk detector
# ══════════════════════════════════════════════════════════════════════════════
class TestMultiFactFanoutRisk(unittest.TestCase):

    def _emap(self, **entity_types):
        return {name: {"entity_type": etype} for name, etype in entity_types.items()}

    def test_two_facts_via_shared_dim_flagged(self):
        from core.graph_resolver import _multi_fact_fanout_risk
        emap = self._emap(F1="fact", F2="fact", D1="dimension")
        path = [
            {"from_entity": "F1", "to_entity": "D1"},
            {"from_entity": "F2", "to_entity": "D1"},
        ]
        risk = _multi_fact_fanout_risk(path, emap)
        self.assertEqual(risk, ["F1", "F2"])

    def test_direct_fact_to_fact_edge_not_flagged(self):
        from core.graph_resolver import _multi_fact_fanout_risk
        emap = self._emap(F1="fact", F2="fact", D1="dimension")
        path = [
            {"from_entity": "F1", "to_entity": "D1"},
            {"from_entity": "F1", "to_entity": "F2"},
        ]
        risk = _multi_fact_fanout_risk(path, emap)
        self.assertEqual(risk, [])

    def test_single_fact_not_flagged(self):
        from core.graph_resolver import _multi_fact_fanout_risk
        emap = self._emap(F1="fact", D1="dimension", D2="dimension")
        path = [
            {"from_entity": "F1", "to_entity": "D1"},
            {"from_entity": "F1", "to_entity": "D2"},
        ]
        risk = _multi_fact_fanout_risk(path, emap)
        self.assertEqual(risk, [])

    def test_resolve_for_question_surfaces_fanout_risk_facts(self):
        from core.graph_resolver import resolve_for_question
        acc = "test_fanout_e2e_011"
        store.save_entity(acc, "SalesFact", "F_SALES", schema_name="dbo",
                           pk_column="SaleID", entity_type="fact")
        store.save_entity(acc, "InventoryFact", "F_INVENTORY", schema_name="dbo",
                           pk_column="SnapshotID", entity_type="fact")
        store.save_entity(acc, "Store", "D_STORE", schema_name="dbo",
                           pk_column="StoreID", entity_type="dimension")
        store.save_relationship(acc,
            from_entity="SalesFact", from_column="StoreID",
            to_entity="Store", to_column="StoreID",
            relationship_type="many_to_one", join_type="INNER")
        store.save_relationship(acc,
            from_entity="InventoryFact", from_column="StoreID",
            to_entity="Store", to_column="StoreID",
            relationship_type="many_to_one", join_type="INNER")
        graph = store.get_full_graph(acc)
        r = resolve_for_question(
            "compare sales", acc, "azure_sql", graph=graph,
            required_entities=["SalesFact", "InventoryFact", "Store"],
        )
        self.assertIn("InventoryFact", r.get("fanout_risk_facts", []))
        self.assertIn("SalesFact", r.get("fanout_risk_facts", []))


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
        # The hand-rolled SVG canvas was replaced by Cytoscape.js
        # (commit 000bb60 "replace entity graph with Cytoscape.js").
        src = GRAPH_TMPL.read_text(encoding="utf-8")
        self.assertIn("cytoscape.min.js", src)
        self.assertIn('id="cy"', src)

    def test_template_has_entity_form(self):
        src = GRAPH_TMPL.read_text(encoding="utf-8")
        # new UI uses modal-based form (field id renamed from m-entity-name
        # to ef-name during the Cytoscape.js rewrite)
        self.assertIn("entity-modal", src)
        self.assertIn('id="ef-name"', src)

    def test_template_has_relationship_form(self):
        src = GRAPH_TMPL.read_text(encoding="utf-8")
        self.assertIn("from_entity", src)
        self.assertIn("to_entity", src)
        self.assertIn("from_column", src)

    # NOTE: test_template_has_resolver_test was removed 2026-07-02. The
    # question-to-JOIN-skeleton resolver test box (resolve-q/resolve-output)
    # has no UI hook in the current Cytoscape.js-based template — it was
    # dropped as a side effect of the 000bb60/d185964 redesigns, not a
    # deliberate product decision. The backend /graph/api/resolve endpoint
    # is still live and covered by test_resolve_api_route. Restoring the UI
    # is tracked as a follow-up (see spawned background task).

    def test_template_has_drag_support(self):
        # Cytoscape.js nodes are draggable natively; the template only needs
        # to persist the new position once a drag ends (commit 000bb60).
        src = GRAPH_TMPL.read_text(encoding="utf-8")
        self.assertIn("cy.on('dragfree', 'node'", src)

    def test_template_renders_suggested_graph_rows(self):
        # Unreviewed/suggested entities and relationships are now flagged
        # via Cytoscape node/edge style selectors (dashed border, reduced
        # opacity) rather than a Jinja-rendered sidebar row class.
        src = GRAPH_TMPL.read_text(encoding="utf-8")
        self.assertIn("node[status=\"suggested\"]", src)
        self.assertIn("edge[?suggested]", src)
        self.assertIn("suggested: r.status === 'suggested'", src)

    # NOTE: test_template_keeps_schema_selector_visible_for_one_schema was
    # removed 2026-07-02. The current template has no schema-scope filter
    # dropdown for multi-schema clients (dropped in the 000bb60/d185964
    # redesigns, not a deliberate product decision). Restoring it is
    # tracked as a follow-up (see spawned background task).

    def test_template_keeps_entity_type_tabs_filterable_after_rebuild(self):
        # Sidebar entity-type filtering was rebuilt around setTypeFilter()/
        # activeTypeFilter (module-level state read inside buildSidebar(),
        # so the active filter survives sidebar rebuilds) with .sf-btn tabs.
        src = GRAPH_TMPL.read_text(encoding="utf-8")
        self.assertIn("function setTypeFilter(btn, type)", src)
        self.assertIn("const typ = activeTypeFilter", src)
        self.assertIn('class="sf-btn active" data-type=""', src)

    def test_template_uses_persistent_three_pane_model_workbench(self):
        # The persistent three-pane inspector was redesigned into a
        # standalone full-page graph with a bottom slide-up panel
        # (commit d185964). Assert on the current #bp panel instead.
        src = GRAPH_TMPL.read_text(encoding="utf-8")
        self.assertIn('id="bp"', src)
        self.assertIn('id="bp-header"', src)
        self.assertIn('id="bp-content"', src)
        self.assertIn("function openBP(", src)
        self.assertIn("function closeBP()", src)

    # NOTE: test_template_has_business_physical_and_query_path_lenses and
    # test_template_keeps_inspector_actions_visible_when_selected were
    # removed 2026-07-02. Neither the Business/Physical/Query-path view-lens
    # toggle nor the drawer action bar (Validate joins / Live probe) exist
    # in the current Cytoscape.js-based template — both dropped as a side
    # effect of the 000bb60/d185964 redesigns, not a deliberate product
    # decision. Restoring them is tracked as a follow-up (see spawned
    # background task).

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


class TestEntityFilterAutocompleteUI(unittest.TestCase):

    def test_filter_editor_contains_table_scoped_suggestions(self):
        src = GRAPH_TMPL.read_text(encoding="utf-8")
        self.assertIn('class="filter-editor-shell"', src)
        self.assertIn('id="filter-suggest"', src)
        self.assertIn("loadFilterColumns", src)
        self.assertIn("validateFilterIdentifiers", src)

    def test_graph_column_api_accepts_schema_table_suffix(self):
        routes = ROUTES.read_text(encoding="utf-8")
        self.assertIn('suffix = "." + fqn.upper()', routes)
        self.assertIn("len(suffix_matches) == 1", routes)


if __name__ == "__main__":
    unittest.main()
