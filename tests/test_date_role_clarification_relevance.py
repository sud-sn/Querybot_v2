"""
tests/test_date_role_clarification_relevance.py

Regression suite for irrelevant date-role and join-path clarification.

The defect: "What is my revenue for the last 2 days?" matched the graph entity
"Last Modified Date" on the single word "last", so an audit date was forced
into the join skeleton beside the metric's own approved date. Two edges into
the same date dimension then either returned no rows or provoked a join-path
clarification whose options were not business-distinguishable — and replying
"no, don't use this" matched no option, so the same card was re-sent forever.

Everything here is metadata-driven: the fixtures below are a generic sales
star schema, and no table, column or business-date name is special-cased in
the code under test.
"""

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_tmpdir = tempfile.mkdtemp(prefix="querybot_date_role_clarification_")
os.environ["DB_PATH"] = str(Path(_tmpdir) / "test_querybot.db")
os.environ["QUERYBOT_DB_PATH"] = str(Path(_tmpdir) / "test_querybot.db")
os.environ["QUERYBOT_KEY_FILE"] = str(Path(_tmpdir) / "test_key")

from core.clarification import (  # noqa: E402
    clarification_reply_matches_option,
    clarification_rejection_message,
    is_clarification_rejection,
)
from core.graph_resolver import (  # noqa: E402
    _resolve_on_graph,
    date_role_entity_for_binding,
    detect_entities,
    is_date_role_entity,
)
from core.query_pipeline import (  # noqa: E402
    _date_option_labels,
    _date_option_identity,
    _unique_date_bindings,
)
from core.semantic_resolution import build_planner_alignment  # noqa: E402


# ── fixtures ──────────────────────────────────────────────────────────────────
# A generic sales star: one fact, one physical date dimension reached through
# three role-playing keys, one ordinary business dimension.

def _entity(name, table, entity_type="dimension"):
    return {
        "entity_name": name, "table_name": table, "schema_name": "dbo",
        "entity_type": entity_type, "status": "confirmed",
    }


def _rel(rel_id, from_col, to_entity, label):
    return {
        "id": rel_id, "from_entity": "Revenue Fact", "from_column": from_col,
        "to_entity": to_entity, "to_column": "DT_KEY", "label": label,
        "join_type": "INNER", "relationship_type": "many_to_one",
        "status": "confirmed",
    }


def _graph():
    return {
        "entities": [
            _entity("Revenue Fact", "FACT_SALES", "fact"),
            _entity("Invoice Date", "DIM_DATE"),
            _entity("Order Date", "DIM_DATE"),
            _entity("Last Modified Date", "DIM_DATE"),
            _entity("Warehouse", "DIM_WAREHOUSE"),
        ],
        "relationships": [
            _rel(1, "INVOICE_DT_KEY", "Invoice Date", "Invoice Date"),
            _rel(2, "ORDER_DT_KEY", "Order Date", "Order Date"),
            _rel(3, "LAST_MOD_DT_KEY", "Last Modified Date", "Last Modified Date"),
            {
                "id": 4, "from_entity": "Revenue Fact", "from_column": "WHS_KEY",
                "to_entity": "Warehouse", "to_column": "WHS_KEY",
                "label": "Warehouse", "join_type": "INNER",
                "relationship_type": "many_to_one", "status": "confirmed",
            },
        ],
        "properties": [
            {"entity_name": "Revenue Fact", "column_name": "NET_REVENUE",
             "display_name": "Net Revenue", "synonyms": "revenue, net sales"},
            {"entity_name": "Invoice Date", "column_name": "CAL_DATE",
             "display_name": "Calendar Date", "synonyms": "date, day"},
            {"entity_name": "Order Date", "column_name": "CAL_DATE",
             "display_name": "Calendar Date", "synonyms": "date, day"},
            {"entity_name": "Last Modified Date", "column_name": "CAL_DATE",
             "display_name": "Calendar Date", "synonyms": "date, day"},
            {"entity_name": "Warehouse", "column_name": "WHS_NAME",
             "display_name": "Warehouse Name", "synonyms": "warehouse, site"},
        ],
    }


def _invoice_binding():
    return {
        "context_name": "Invoice Date", "date_role": "invoice_date",
        "fact_table": "dbo.FACT_SALES", "fact_column": "INVOICE_DT_KEY",
        "dimension_table": "dbo.DIM_DATE", "dimension_key": "DT_KEY",
        "date_value_column": "CAL_DATE", "date_key_type": "surrogate_fk",
        "temporal_grain": "calendar", "governance_status": "approved",
    }


# ══════════════════════════════════════════════════════════════════════════════
# 1  Entity detection — a date role must be named, not guessed
# ══════════════════════════════════════════════════════════════════════════════
class TestDateRoleEntityDetection(unittest.TestCase):

    def _detect(self, question, **kwargs):
        return detect_entities(question, _graph(), **kwargs)

    def test_relative_range_does_not_detect_audit_date(self):
        """The reported defect: "last 2 days" is a range, not Last Modified Date."""
        detected = self._detect("What is my revenue for the last 2 days?")
        self.assertNotIn("Last Modified Date", detected)
        self.assertIn("Revenue Fact", detected)

    def test_generic_temporal_wording_never_implies_a_date_role(self):
        for question in (
            "What is my revenue for the last 2 days?",
            "Latest 5 orders",
            "Recent warehouse revenue",
            "Previous month sales",
            "Show revenue for last quarter",
            "What is my revenue per warehouse for the available dates?",
            "revenue trend over time",
            "revenue by day",
            "how much revenue today",
        ):
            with self.subTest(question=question):
                detected = self._detect(question)
                for role in ("Invoice Date", "Order Date", "Last Modified Date"):
                    self.assertNotIn(role, detected, f"{role} detected for {question!r}")

    def test_explicit_audit_date_is_detected(self):
        detected = self._detect("Show revenue by last modified date")
        self.assertIn("Last Modified Date", detected)

    def test_explicit_audit_date_does_not_pull_other_roles(self):
        detected = self._detect("Show revenue by last modified date")
        self.assertNotIn("Invoice Date", detected)
        self.assertNotIn("Order Date", detected)

    def test_explicit_invoice_date_is_detected_alone(self):
        detected = self._detect("Show revenue by invoice date")
        self.assertIn("Invoice Date", detected)
        self.assertNotIn("Last Modified Date", detected)
        self.assertNotIn("Order Date", detected)

    def test_registered_synonym_names_the_role(self):
        """A builtin business-date synonym is an authoritative signal."""
        for question, expected in (
            ("Show revenue by billing date", "Invoice Date"),
            ("Show revenue by invoiced date", "Invoice Date"),
            ("Show revenue by ordered date", "Order Date"),
            ("Show revenue by updated date", "Last Modified Date"),
        ):
            with self.subTest(question=question):
                self.assertIn(expected, self._detect(question))

    def test_business_half_of_the_role_name_is_enough(self):
        """"modified date" names the role; the generic "last" prefix is optional."""
        self.assertIn("Last Modified Date", self._detect("revenue by modified date"))

    def test_entity_noun_alone_does_not_imply_its_date_role(self):
        """"invoice number" is about invoices, not about Invoice Date."""
        detected = self._detect("Show revenue by invoice number")
        self.assertNotIn("Invoice Date", detected)

    def test_shared_date_dimension_table_name_is_not_a_signal(self):
        """Every role shares DIM_DATE, so the table name identifies none of them."""
        detected = self._detect("show revenue from dim_date")
        for role in ("Invoice Date", "Order Date", "Last Modified Date"):
            self.assertNotIn(role, detected)

    def test_required_entities_still_force_a_date_role_in(self):
        """The governed sources speak through required_entities, and are honored."""
        detected = self._detect(
            "What is my revenue for the last 2 days?",
            required_entities={"Revenue Fact", "Invoice Date"},
        )
        self.assertIn("Invoice Date", detected)
        self.assertNotIn("Last Modified Date", detected)

    def test_ordinary_dimensions_keep_loose_matching(self):
        """The strict gate is only for role-playing dates."""
        detected = self._detect("revenue by warehouse")
        self.assertIn("Warehouse", detected)

    def test_is_date_role_entity_classifies_from_metadata(self):
        self.assertTrue(is_date_role_entity(_entity("Invoice Date", "DIM_DATE")))
        self.assertTrue(is_date_role_entity(_entity("Posting Period", "DIM_PERIOD")))
        self.assertTrue(is_date_role_entity(_entity("Fiscal Calendar", "D_CAL")))
        self.assertFalse(is_date_role_entity(_entity("Warehouse", "DIM_WAREHOUSE")))
        self.assertFalse(is_date_role_entity(_entity("Customer", "DIM_CUSTOMER")))
        self.assertFalse(is_date_role_entity(_entity("Revenue Fact", "FACT_SALES", "fact")))


# ══════════════════════════════════════════════════════════════════════════════
# 2  Graph planning — only the required date edge is selected
# ══════════════════════════════════════════════════════════════════════════════
class TestGraphSelectsOnlyTheRequiredDateEdge(unittest.TestCase):

    def _edges(self, question, required, facts=("dbo.FACT_SALES",)):
        resolved = _resolve_on_graph(
            question, "azure_sql", _graph(), None, set(required), set(), set(facts),
        )
        return resolved, [
            (edge["from_entity"], edge["conditions"][0][0], edge["to_entity"])
            for edge in resolved["resolved_edges"]
        ]

    def test_required_invoice_date_selects_only_the_invoice_edge(self):
        resolved, edges = self._edges(
            "What is my revenue for the last 2 days?",
            {"Revenue Fact", "Invoice Date"},
        )
        self.assertEqual(
            edges, [("Revenue Fact", "INVOICE_DT_KEY", "Invoice Date")],
        )
        self.assertIn("INVOICE_DT_KEY", resolved["join_skeleton"])
        self.assertNotIn("LAST_MOD_DT_KEY", resolved["join_skeleton"])
        self.assertNotIn("ORDER_DT_KEY", resolved["join_skeleton"])

    def test_grouped_question_joins_the_dimension_and_the_approved_date(self):
        resolved, edges = self._edges(
            "What is my revenue per warehouse for the available dates?",
            {"Revenue Fact", "Invoice Date"},
        )
        self.assertIn(("Revenue Fact", "INVOICE_DT_KEY", "Invoice Date"), edges)
        self.assertIn(("Revenue Fact", "WHS_KEY", "Warehouse"), edges)
        self.assertNotIn("LAST_MOD_DT_KEY", resolved["join_skeleton"])

    def test_explicit_audit_date_uses_the_governed_audit_edge(self):
        _resolved, edges = self._edges("Show revenue by last modified date", set())
        self.assertEqual(
            edges, [("Revenue Fact", "LAST_MOD_DT_KEY", "Last Modified Date")],
        )

    def test_date_role_entity_is_resolved_by_physical_edge_not_by_table(self):
        """Three roles share DIM_DATE; a table lookup would pick one at random."""
        graph = _graph()
        self.assertEqual(
            date_role_entity_for_binding(graph, _invoice_binding()), "Invoice Date",
        )
        self.assertEqual(
            date_role_entity_for_binding(graph, {
                **_invoice_binding(), "context_name": "Last Modified Date",
                "date_role": "modified_date", "fact_column": "LAST_MOD_DT_KEY",
            }),
            "Last Modified Date",
        )

    def test_date_role_entity_falls_back_to_the_business_name(self):
        """A stale fact column still resolves through the role's own name."""
        self.assertEqual(
            date_role_entity_for_binding(
                _graph(), {**_invoice_binding(), "fact_column": "GONE"},
            ),
            "Invoice Date",
        )

    def test_date_role_entity_is_blank_when_genuinely_ambiguous(self):
        """Forcing an arbitrary role is the defect, so resolve to nothing."""
        self.assertEqual(
            date_role_entity_for_binding(_graph(), {
                "context_name": "", "date_role": "",
                "fact_table": "dbo.FACT_SALES", "fact_column": "GONE",
                "dimension_table": "dbo.DIM_DATE", "dimension_key": "DT_KEY",
            }),
            "",
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3  Planner alignment — a lexical guess never becomes authoritative
# ══════════════════════════════════════════════════════════════════════════════
class TestPlannerAlignmentDropsUngovernedDateRoles(unittest.TestCase):

    def _align(self, detected, resolution):
        return build_planner_alignment(
            graph=_graph(),
            graph_ctx={"detected": list(detected), "anchor": "Revenue Fact"},
            semantic_plan={},
            metric_formula_tables={"dbo.FACT_SALES"},
            date_context_resolution=resolution,
        )

    def test_ungoverned_date_role_is_dropped(self):
        alignment = self._align(
            ["Revenue Fact", "Warehouse", "Last Modified Date"],
            {"status": "selected", "binding": _invoice_binding()},
        )
        self.assertEqual(alignment["governed_date_entities"], ["Invoice Date"])
        self.assertEqual(alignment["dropped_date_entities"], ["Last Modified Date"])
        self.assertNotIn("Last Modified Date", alignment["required_entities"])
        self.assertIn("Invoice Date", alignment["required_entities"])
        self.assertIn("Warehouse", alignment["required_entities"])

    def test_shared_dimension_table_does_not_reintroduce_another_role(self):
        """dimension_table is DIM_DATE, owned by all three roles."""
        alignment = self._align(
            ["Revenue Fact"], {"status": "selected", "binding": _invoice_binding()},
        )
        self.assertEqual(
            [
                name for name in alignment["required_entities"]
                if name in {"Invoice Date", "Order Date", "Last Modified Date"}
            ],
            ["Invoice Date"],
        )

    def test_governed_audit_date_keeps_the_audit_role(self):
        alignment = self._align(
            ["Revenue Fact", "Last Modified Date", "Invoice Date"],
            {"status": "selected", "binding": {
                **_invoice_binding(), "context_name": "Last Modified Date",
                "date_role": "modified_date", "fact_column": "LAST_MOD_DT_KEY",
            }},
        )
        self.assertIn("Last Modified Date", alignment["required_entities"])
        self.assertEqual(alignment["dropped_date_entities"], ["Invoice Date"])

    def test_multiple_explicit_roles_are_both_governed(self):
        alignment = self._align(
            ["Revenue Fact", "Invoice Date", "Order Date"],
            {"status": "selected_many", "bindings": [
                _invoice_binding(),
                {**_invoice_binding(), "context_name": "Order Date",
                 "date_role": "order_date", "fact_column": "ORDER_DT_KEY"},
            ]},
        )
        self.assertIn("Invoice Date", alignment["required_entities"])
        self.assertIn("Order Date", alignment["required_entities"])
        self.assertEqual(alignment["dropped_date_entities"], [])

    def test_no_governed_date_leaves_detection_untouched(self):
        """Without a resolved binding there is nothing authoritative to narrow to."""
        alignment = self._align(
            ["Revenue Fact", "Last Modified Date"], {"status": "none"},
        )
        self.assertIn("Last Modified Date", alignment["required_entities"])
        self.assertEqual(alignment["dropped_date_entities"], [])

    def test_unrelated_fact_is_still_dropped(self):
        """The pre-existing fact-scoping behavior is preserved."""
        graph = _graph()
        graph["entities"].append(_entity("Inventory Fact", "FACT_INVENTORY", "fact"))
        alignment = build_planner_alignment(
            graph=graph,
            graph_ctx={"detected": ["Revenue Fact", "Inventory Fact"], "anchor": "Revenue Fact"},
            semantic_plan={},
            metric_formula_tables={"dbo.FACT_SALES"},
            date_context_resolution={"status": "none"},
        )
        self.assertEqual(alignment["dropped_fact_entities"], ["Inventory Fact"])


# ══════════════════════════════════════════════════════════════════════════════
# 4  Clarification options — never two buttons a user cannot choose between
# ══════════════════════════════════════════════════════════════════════════════
class TestDateClarificationOptions(unittest.TestCase):

    def test_duplicate_physical_rows_collapse_to_one_option(self):
        """Same fact column offered twice is one option, not two."""
        collapsed = _unique_date_bindings([
            {"context_name": "Last Modified Date", "fact_table": "dbo.FACT_SALES",
             "fact_column": "LAST_MOD_DT_KEY", "date_key_type": "surrogate_fk",
             "temporal_grain": "calendar"},
            {"context_name": "Last Modified Date", "fact_table": "dbo.FACT_SALES",
             "fact_column": "LAST_MOD_DT_KEY", "date_key_type": "surrogate_fk",
             "temporal_grain": "calendar", "governance_status": "approved"},
        ])
        self.assertEqual(len(collapsed), 1)

    def test_equivalent_encodings_of_one_business_date_collapse(self):
        """A native date beside its integer twin is not a business choice."""
        collapsed = _unique_date_bindings([
            {"context_name": "Last Modified Date", "fact_table": "dbo.FACT_SALES",
             "fact_column": "LAST_MOD_YYYYMMDD", "date_key_type": "yyyymmdd_integer",
             "temporal_grain": "calendar", "governance_status": "approved"},
            {"context_name": "Last Modified Date", "fact_table": "dbo.FACT_SALES",
             "fact_column": "LAST_MOD_DATE", "date_key_type": "native_date",
             "temporal_grain": "calendar", "governance_status": "approved"},
        ])
        self.assertEqual(len(collapsed), 1)
        # The safest executable representation survives.
        self.assertEqual(collapsed[0]["fact_column"], "LAST_MOD_DATE")

    def test_genuinely_different_roles_stay_separate(self):
        collapsed = _unique_date_bindings([
            {"context_name": "Invoice Date", "fact_table": "dbo.FACT_SALES",
             "fact_column": "INVOICE_DT_KEY", "date_key_type": "surrogate_fk",
             "temporal_grain": "calendar"},
            {"context_name": "Order Date", "fact_table": "dbo.FACT_SALES",
             "fact_column": "ORDER_DT_KEY", "date_key_type": "surrogate_fk",
             "temporal_grain": "calendar"},
        ])
        self.assertEqual(
            sorted(item["context_name"] for item in collapsed),
            ["Invoice Date", "Order Date"],
        )

    def test_option_labels_are_unique_and_business_distinguishable(self):
        bindings = _unique_date_bindings([
            {"context_name": "Snapshot Date", "fact_table": "dbo.F_INV",
             "fact_column": "SNAP_DT", "date_key_type": "native_date",
             "temporal_grain": "calendar"},
            {"context_name": "Snapshot Date", "fact_table": "dbo.F_INV",
             "fact_column": "SNAP_MONTH", "date_key_type": "native_date",
             "temporal_grain": "month"},
        ])
        labels = _date_option_labels(bindings)
        self.assertEqual(len(set(labels.values())), len(bindings))
        # No bare positional ordinal — the source field is named instead.
        for label in labels.values():
            self.assertNotRegex(label, r"\(\d+\)$")
        self.assertTrue(any("SNAP_DT" in label for label in labels.values()))

    def test_distinct_roles_keep_their_plain_business_names(self):
        bindings = [
            {"context_name": "Invoice Date", "fact_table": "dbo.FACT_SALES",
             "fact_column": "INVOICE_DT_KEY", "date_key_type": "surrogate_fk"},
            {"context_name": "Order Date", "fact_table": "dbo.FACT_SALES",
             "fact_column": "ORDER_DT_KEY", "date_key_type": "surrogate_fk"},
        ]
        labels = _date_option_labels(bindings)
        self.assertEqual(
            sorted(labels.values()), ["Invoice Date", "Order Date"],
        )

    def test_option_identity_is_stable(self):
        binding = _invoice_binding()
        self.assertEqual(
            _date_option_identity(binding), ("DBO.FACT_SALES", "INVOICE_DT_KEY"),
        )


# ══════════════════════════════════════════════════════════════════════════════
# 5  Clarification rejection — narrow, and it stops the loop
# ══════════════════════════════════════════════════════════════════════════════
class TestClarificationRejectionDetection(unittest.TestCase):

    def test_rejections_are_recognized(self):
        for text in (
            "No", "no", "Nope", "nah", "No.",
            "No, don't use this", "no dont use this", "Don't use this",
            "Do not use either", "don't use any of these",
            "Neither", "Neither of these", "None of these",
            "none of the above", "not these", "use neither",
            "I don't want either", "Cancel", "cancel this",
            "Skip this", "skip", "never mind", "nvm", "start over",
            "not relevant", "none apply", "no thanks",
            "Please don't use these paths", "Don't use these relationships",
        ):
            with self.subTest(text=text):
                self.assertTrue(is_clarification_rejection(text), text)

    def test_data_questions_containing_negation_are_not_rejections(self):
        for text in (
            "Show customers with no orders.",
            "Do not include cancelled orders.",
            "Revenue excluding returned invoices.",
            "Warehouses with no available stock.",
            "show me warehouses with no stock and no orders",
            "which customers have no invoices",
            "no orders last month",
            "no revenue this month",
            "exclude cancelled orders",
            "cancel rate by region",
            "skipped shipments this week",
            "customers with none of these products",
            "show total revenue by customer instead",
            "do not use the invoice date, use the order date",
            "don't use tax in the revenue formula",
            "revenue by order date",
            "Invoice Date",
            "Last Modified Date",
        ):
            with self.subTest(text=text):
                self.assertFalse(is_clarification_rejection(text), text)

    def test_blank_reply_is_not_a_rejection(self):
        for text in ("", "   ", None):
            self.assertFalse(is_clarification_rejection(text))

    def test_an_offered_no_option_still_owns_the_word_no(self):
        """A card that deliberately offers "No" is not cancelled by "no"."""
        cmeta = {
            "source": "result_reference_confirmation",
            "options": [
                {"id": "use-previous-result", "label": "Yes — use the previous result",
                 "value": "use_previous_result"},
                {"id": "new-question", "label": "No — this is a new question",
                 "value": "new_question"},
            ],
        }
        self.assertTrue(clarification_reply_matches_option(cmeta, "No — this is a new question"))

    def test_date_options_do_not_absorb_a_rejection(self):
        cmeta = {
            "source": "metric_date_context",
            "options": [{"id": "date_role_1", "label": "Invoice Date",
                         "value": "Invoice Date", "context_name": "Invoice Date",
                         "fact_table": "dbo.FACT_SALES", "fact_column": "INVOICE_DT_KEY"}],
        }
        self.assertFalse(clarification_reply_matches_option(cmeta, "no dont use this"))

    def test_rejection_message_is_actionable(self):
        for source in ("graph_join_path", "metric_date_context", "llm", ""):
            message = clarification_rejection_message({"source": source})
            self.assertTrue(message.strip())
            self.assertIn("again", message.lower())


# ══════════════════════════════════════════════════════════════════════════════
# 6  Both entry points clear the pending clarification on rejection
# ══════════════════════════════════════════════════════════════════════════════
class _RecordingAdapter:
    """Minimal adapter double: records what the user would have been sent."""

    platform = "portal"
    session_id = "acct-reject:portal:u1"
    thread_id = "thread-1"

    def __init__(self):
        self.messages: list[str] = []
        self.prompts: list[tuple[str, list]] = []

    def make_event(self, text):
        from gateway.base import PlatformEvent
        return PlatformEvent(
            account_id="acct-reject", user_id="u1", channel_id="c1",
            text=text, platform="portal",
        )

    async def send_message(self, event, text, **kwargs):
        self.messages.append(text)

    async def send_clarification_prompt(self, event, question, options):
        self.prompts.append((question, list(options)))

    async def send_typing(self, *args, **kwargs):
        return None


class TestRejectionClearsPendingState(unittest.TestCase):
    """The loop only ends if the pending row is actually gone."""

    ACCOUNT = "acct-reject"
    USER = "u1"

    def setUp(self):
        from core.clarification import clear_pending, save_pending
        import store.db as _db

        _db.init_db()
        self.save_pending = save_pending
        self.clear_pending = clear_pending
        self.session_id = _RecordingAdapter.session_id
        clear_pending(self.ACCOUNT, self.USER, session_id=self.session_id)

    def tearDown(self):
        self.clear_pending(self.ACCOUNT, self.USER, session_id=self.session_id)

    def _seed_pending(self):
        self.save_pending(
            self.ACCOUNT, self.USER,
            "What is my revenue for the last 2 days?",
            clarification_meta={
                "source": "metric_date_context",
                "question": "Which date context should I use?",
                "options": [
                    {"id": "date_role_1", "label": "Last Modified Date",
                     "value": "Last Modified Date"},
                    {"id": "date_role_2", "label": "Last Modified Date (date code)",
                     "value": "Last Modified Date (date code)"},
                ],
                "pending_id": "p-reject-1",
            },
            session_id=self.session_id,
        )

    def test_dispatcher_rejection_clears_pending_and_does_not_replay(self):
        from unittest.mock import patch
        import core.dispatcher as dispatcher_module
        from core.clarification import get_pending
        from core.dispatcher import dispatch

        self._seed_pending()
        adapter = _RecordingAdapter()
        event = adapter.make_event("no, don't use this")
        enqueued: list[str] = []

        with patch("core.dispatcher.get_state", return_value={"state": "READY"}), \
             patch("core.dispatcher.get_client_db", return_value={"db_type": "azure_sql"}), \
             patch("core.dispatcher.clarification_session_id", return_value=self.session_id), \
             patch("core.dispatcher._enqueue_query",
                   side_effect=lambda *a, **k: enqueued.append(a[4])), \
             patch.object(dispatcher_module.store, "get_client", return_value={
                 "account_id": self.ACCOUNT, "state": "READY", "name": "T"}):
            asyncio.run(dispatch(
                self.ACCOUNT, event, adapter, None,
                portal_user={"id": 1, "role": "admin", "email": "u@x.com"},
            ))

        self.assertIsNone(
            get_pending(self.ACCOUNT, self.USER, session_id=self.session_id),
            f"pending clarification must be cleared on rejection; "
            f"sent={adapter.messages} prompts={adapter.prompts}",
        )
        self.assertEqual(enqueued, [], "the rejected option must not be run")
        self.assertEqual(adapter.prompts, [], "the same options must not be re-sent")
        self.assertTrue(adapter.messages, "the user must be told the step was cancelled")

    def test_dispatcher_new_question_after_clarification_is_processed(self):
        """Scenario F — a genuine new question is not a malformed reply."""
        from unittest.mock import patch
        import core.dispatcher as dispatcher_module
        from core.clarification import get_pending
        from core.dispatcher import dispatch

        self._seed_pending()
        adapter = _RecordingAdapter()
        event = adapter.make_event("Show total revenue by customer instead")
        enqueued: list[str] = []

        with patch("core.dispatcher.get_state", return_value={"state": "READY"}), \
             patch("core.dispatcher.get_client_db", return_value={"db_type": "azure_sql"}), \
             patch("core.dispatcher.clarification_session_id", return_value=self.session_id), \
             patch("core.dispatcher._enqueue_query",
                   side_effect=lambda *a, **k: enqueued.append(a[4])), \
             patch.object(dispatcher_module.store, "get_client", return_value={
                 "account_id": self.ACCOUNT, "state": "READY", "name": "T"}):
            asyncio.run(dispatch(
                self.ACCOUNT, event, adapter, None,
                portal_user={"id": 1, "role": "admin", "email": "u@x.com"},
            ))

        self.assertIsNone(
            get_pending(self.ACCOUNT, self.USER, session_id=self.session_id),
        )
        self.assertEqual(enqueued, ["Show total revenue by customer instead"])

    def test_websocket_handler_rejects_before_replaying_options(self):
        """The portal card path must behave identically to the dispatcher."""
        source = Path(ROOT / "gateway" / "webhooks.py").read_text(encoding="utf-8")
        rejection_index = source.find("is_clarification_rejection(_rejection_text)")
        self.assertGreater(
            rejection_index, 0,
            "the WebSocket clarification handler must check for a rejection",
        )
        # It has to run before the generic "couldn't match that" re-prompt.
        replay_index = source.find('_t("reply.clarify.choose_option")')
        self.assertGreater(replay_index, rejection_index)
        self.assertIn("clarification_rejection_message(cmeta)", source)


# ══════════════════════════════════════════════════════════════════════════════
# 7  Validator — role substitution is caught on every key convention
# ══════════════════════════════════════════════════════════════════════════════
class TestValidatorCatchesDateRoleSubstitution(unittest.TestCase):
    """The role-substitution guard used to skip the *_DT_DMS_KEY convention.

    is_plain_surrogate_date_role_column() excludes that family because it has
    its own arithmetic-decode rule — an exclusion about how a key converts to a
    calendar value, not about which business date a query may use. Reusing it to
    police role substitution left every warehouse on the DMS convention
    unguarded: a query could filter AND anchor entirely on LAST_MOD_DT_DMS_KEY
    while the approved role was INVOICE_DT_DMS_KEY, and validation passed clean.
    """

    COLUMNS = {
        "ERP.F_SALES_INVOICE": {
            "NET_REVENUE": "decimal", "INVOICE_DT_DMS_KEY": "int",
            "LAST_MOD_DT_DMS_KEY": "int", "WHS_KEY": "int",
        },
        "ERP.DT_DMS": {"DT_DMS_KEY": "int", "CAL_DATE": "date"},
    }
    PLAN = {
        "enabled": True,
        "fields": [{
            "term": "Invoice Date", "table": "ERP.DT_DMS", "column": "CAL_DATE",
            "role": "date_dimension", "enforcement": "optional",
        }],
        "joins": [], "required_tables": [],
        "temporal_policies": [{
            "kind": "last_n_days", "anchor_policy": "latest_available",
            "fact_table": "ERP.F_SALES_INVOICE", "fact_column": "INVOICE_DT_DMS_KEY",
            "date_table": "ERP.DT_DMS", "date_column": "CAL_DATE",
            "dimension_table": "ERP.DT_DMS", "dimension_key": "DT_DMS_KEY",
            "date_key_type": "surrogate_fk", "business_role": "invoice_date",
            "anchor_table": "ERP.F_SALES_INVOICE",
        }],
    }

    def _validate(self, sql, plan=None):
        from core.validator import validate_sql_detailed
        return validate_sql_detailed(
            sql, set(self.COLUMNS), "azure_sql",
            table_columns=self.COLUMNS,
            semantic_context={"semantic_plan": plan or self.PLAN},
        )

    def _sql(self, role_key):
        return (
            "SELECT SUM(f.NET_REVENUE) AS TOTAL "
            "FROM ERP.F_SALES_INVOICE f "
            f"JOIN ERP.DT_DMS d ON f.{role_key} = d.DT_DMS_KEY "
            "WHERE d.CAL_DATE >= DATEADD(day, -2, (SELECT MAX(d2.CAL_DATE) "
            "  FROM ERP.F_SALES_INVOICE f2 "
            f"  JOIN ERP.DT_DMS d2 ON f2.{role_key} = d2.DT_DMS_KEY))"
        )

    def test_audit_role_substituted_in_the_join_is_rejected(self):
        result = self._validate(self._sql("LAST_MOD_DT_DMS_KEY"))
        self.assertFalse(result.ok, "substituting the audit date must not validate")
        codes = {error.get("code") for error in (result.errors or [])}
        self.assertIn("temporal_role_mismatch", codes)
        flagged = {
            error.get("column") for error in result.errors
            if error.get("code") == "temporal_role_mismatch"
        }
        self.assertIn("LAST_MOD_DT_DMS_KEY", flagged)

    def test_approved_role_still_validates(self):
        """The tightening must not reject the correct query."""
        result = self._validate(self._sql("INVOICE_DT_DMS_KEY"))
        self.assertTrue(result.ok, f"approved role rejected: {result.errors}")

    def test_second_governed_role_is_not_flagged(self):
        """A question resolving two roles must not flag either of them."""
        plan = {
            **self.PLAN,
            "temporal_policies": [
                self.PLAN["temporal_policies"][0],
                {
                    **self.PLAN["temporal_policies"][0],
                    "anchor_policy": "explicit",
                    "fact_column": "LAST_MOD_DT_DMS_KEY",
                    "business_role": "modified_date",
                },
            ],
        }
        result = self._validate(self._sql("LAST_MOD_DT_DMS_KEY"), plan=plan)
        codes = {error.get("code") for error in (result.errors or [])}
        self.assertNotIn("temporal_role_mismatch", codes)

    def test_policy_without_an_approved_key_flags_nothing(self):
        """Without a governed fact key, "not the approved key" is unknowable."""
        plan = {
            **self.PLAN,
            "temporal_policies": [{
                "anchor_policy": "latest_available", "date_column": "CAL_DATE",
                "anchor_table": "ERP.F_SALES_INVOICE",
                "fact_table": "ERP.F_SALES_INVOICE",
            }],
        }
        result = self._validate(self._sql("LAST_MOD_DT_DMS_KEY"), plan=plan)
        codes = {error.get("code") for error in (result.errors or [])}
        self.assertNotIn("temporal_role_mismatch", codes)

    def test_key_convention_coverage(self):
        from core.date_roles import (
            is_plain_surrogate_date_role_column,
            is_surrogate_date_role_key_column,
        )
        for column in (
            "LAST_MOD_DT_DMS_KEY", "INVOICE_DT_DMS_KEY", "ORDER_DT_DMS_KEY",
            "UPDATED_DATE_DMS_KEY", "DISPENSE_DATE_ID",
        ):
            with self.subTest(column=column):
                self.assertTrue(is_surrogate_date_role_key_column(column))
        # The DMS family is exactly what the narrower predicate skips.
        self.assertFalse(is_plain_surrogate_date_role_column("LAST_MOD_DT_DMS_KEY"))
        # Non-date and native-date columns stay out of scope.
        for column in ("NET_REVENUE", "WHS_KEY", "ORDER_DATE", "CAL_DATE"):
            with self.subTest(column=column):
                self.assertFalse(is_surrogate_date_role_key_column(column))


if __name__ == "__main__":
    unittest.main()
