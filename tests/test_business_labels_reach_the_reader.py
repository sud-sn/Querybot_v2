# -*- coding: utf-8 -*-
"""tests/test_business_labels_reach_the_reader.py

F9 · "Break down by Sup" — the string L4 named as the bug, still live.

Commit c7d0689 (L4) taught the narrative layer to ask the tenant's vocabulary
what a column is called, and its message named "Break down by Sup" as a symptom
of the same class. It could not reach it. L4 rewrote
core/response_builder.py::_display_label and re-pointed seven inline copies
INSIDE that module; the drill chip's label is built somewhere else entirely, at
plan-build time in core/semantic_model.py, from a different helper. Executed
against the shipped code the chips read:

    Break down by Sup          (SUP_DMS_KEY)
    Break down by Vnd          (VND_DMS_KEY)
    Break down by Warehouse    (WHS_DMS_KEY — and only because "WHS" happens to
                                sit in a small entity-prefix table)

That is the shape of this whole family of defects: one helper is fixed, and its
siblings in other modules keep printing the warehouse's spelling. So the fix
here is not a second helper. `_display_label`'s body moves to
core.schema_enrichment.display_label — next to the vocabulary it consults, in a
module that imports nothing which imports it back — and the narrative layer's
name becomes a one-line delegation. There stays exactly one implementation, and
the producers that could not reach it can.

The vocabulary is asked about the STRIPPED ROLE, never the source key. That
distinction is the whole fix and the naive version is worse than the bug:
display_label("SUP_DMS_KEY") is "Supplier Dimension Key" and
display_label("PFT_CTR_DMS_KEY") is "Profit Ctr Dimension Key".

Every test runs the real plan builder, the real chip builder and the real SQL
builder. The only boundary mocked is the temporary directory the plan builder
genuinely reads a schema from.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SCHEMA = {
    "CHATBOTDB.PROFITABILITY.CUS_ORD_IVC_FCT": {
        "database": "CHATBOTDB", "schema": "PROFITABILITY",
        "table": "CUS_ORD_IVC_FCT",
        "columns": [
            {"name": "CUS_ORD_IVC_FCT_KEY", "type": "bigint"},
            {"name": "WHS_DMS_KEY", "type": "bigint"},
            {"name": "SUP_DMS_KEY", "type": "bigint"},
            {"name": "VND_DMS_KEY", "type": "bigint"},
            {"name": "PFT_CTR_DMS_KEY", "type": "bigint"},
            {"name": "CUS_IVC_DT_DMS_KEY", "type": "bigint"},
            {"name": "SOP_CUS_IVC_LIN_AMT", "type": "decimal"},
        ],
    },
    "CHATBOTDB.PROFITABILITY.WHS_DMS": {
        "database": "CHATBOTDB", "schema": "PROFITABILITY", "table": "WHS_DMS",
        "columns": [{"name": "WHS_DMS_KEY", "type": "bigint"},
                    {"name": "WHS_NM", "type": "nvarchar"},
                    {"name": "WHS_CD", "type": "nvarchar"}],
    },
    "CHATBOTDB.PROFITABILITY.SUP_DMS": {
        "database": "CHATBOTDB", "schema": "PROFITABILITY", "table": "SUP_DMS",
        "columns": [{"name": "SUP_DMS_KEY", "type": "bigint"},
                    {"name": "SUP_NM", "type": "nvarchar"},
                    {"name": "SUP_CD", "type": "nvarchar"}],
    },
    "CHATBOTDB.PROFITABILITY.VND_DMS": {
        "database": "CHATBOTDB", "schema": "PROFITABILITY", "table": "VND_DMS",
        "columns": [{"name": "VND_DMS_KEY", "type": "bigint"},
                    {"name": "VND_NM", "type": "nvarchar"}],
    },
    "CHATBOTDB.PROFITABILITY.PFT_CTR_DMS": {
        "database": "CHATBOTDB", "schema": "PROFITABILITY",
        "table": "PFT_CTR_DMS",
        "columns": [{"name": "PFT_CTR_DMS_KEY", "type": "bigint"},
                    {"name": "PFT_CTR_NM", "type": "nvarchar"}],
    },
    "CHATBOTDB.PROFITABILITY.DT_DMS": {
        "database": "CHATBOTDB", "schema": "PROFITABILITY", "table": "DT_DMS",
        "columns": [{"name": "DT_DMS_KEY", "type": "bigint"},
                    {"name": "CAL_DT", "type": "date"},
                    {"name": "MONTH", "type": "int"}],
    },
}

CHIP_CONTEXT = {"mode": "ranking", "row_count": 6,
                "numeric_cols": ["SOP_CUS_IVC_LIN_AMT"], "text_cols": ["MONTH"],
                "comparison_stats": {"leader": "x", "leader_share_pct": 30.0}}


def real_plan(schema=None):
    """A semantic plan built by the real builder from a real schema on disk."""
    from core.semantic_model import build_runtime_semantic_plan, write_semantic_model

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        schema_dir, kb_dir = root / "schema", root / "kb"
        schema_dir.mkdir()
        (schema_dir / "_schema.json").write_text(
            json.dumps(schema or SCHEMA), encoding="utf-8")
        write_semantic_model(schema_dir=str(schema_dir), kb_dir=str(kb_dir))
        return build_runtime_semantic_plan(
            str(kb_dir), question="total invoice amount by month",
            selected_schema="PROFITABILITY")


def names_of(plan) -> list[str]:
    return [d["name"] for d in plan.get("available_dimensions") or []]


class TestTheChipNamesTheBusinessEntity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.plan = real_plan()

    def test_the_supplier_is_called_the_supplier(self):
        # 'Sup' before. This is the assertion the L4 commit believed it had
        # already satisfied.
        self.assertIn("Supplier", names_of(self.plan))
        self.assertNotIn("Sup", names_of(self.plan))

    def test_and_the_vendor_the_vendor(self):
        self.assertIn("Vendor", names_of(self.plan))
        self.assertNotIn("Vnd", names_of(self.plan))

    def test_a_dimension_already_in_words_is_left_alone(self):
        self.assertIn("Warehouse", names_of(self.plan))

    def test_the_vocabulary_is_asked_about_the_role_not_the_source_key(self):
        # The naive fix -- display_label(source_key) -- gives "Profit Ctr
        # Dimension Key", which is worse than the bug it replaces.
        self.assertIn("Profit Center", names_of(self.plan))
        for wrong in ("Profit Ctr Dimension Key", "Supplier Dimension Key",
                      "Sup Dms Key"):
            self.assertNotIn(wrong, names_of(self.plan))

    def test_the_chip_the_reader_actually_clicks(self):
        # The two real functions, end to end. The existing test for this line
        # hand-feeds a dimension dict straight into the chip builder, so it
        # could never have seen this.
        from core.response_builder import compute_chip_eligibility

        chips = compute_chip_eligibility(ctx=CHIP_CONTEXT, brief={},
                                         semantic_plan=self.plan)
        labels = [c["label"] for c in chips if str(c.get("id", "")).startswith("drill_dim")]
        self.assertIn("Break down by Supplier", labels)
        self.assertNotIn("Break down by Sup", labels)

    def test_the_chip_id_still_finds_its_dimension(self):
        # The name is the label AND the lookup key. A rename that broke the
        # round trip would leave a chip that does nothing.
        from core.drill_dimension import find_drill_candidate

        found = find_drill_candidate("Supplier", self.plan)
        self.assertIsNotNone(found)
        self.assertEqual(found["source_key_column"], "SUP_DMS_KEY")

    def test_the_drill_result_headers_the_column_supplier(self):
        # The name is also the SELECT alias, so the reader's table header
        # changes with the chip. Deliberate, and pinned rather than incidental.
        from core.drill_dimension import build_deterministic_drill_sql, find_drill_candidate

        sql = build_deterministic_drill_sql(
            "SELECT SUM(f.SOP_CUS_IVC_LIN_AMT) AS AMT "
            "FROM CHATBOTDB.PROFITABILITY.CUS_ORD_IVC_FCT f",
            find_drill_candidate("Supplier", self.plan), "azure_sql")
        self.assertIn("[Supplier]", sql)
        self.assertIn("SUP_NM", sql)


class TestItDoesNotOverwriteWhatSomeoneChose(unittest.TestCase):

    def test_two_entities_that_resolve_alike_do_not_share_one_chip(self):
        # The expansion collapses spellings, and the name is the chip id. Two
        # chips with one id means the drill answers with whichever dimension
        # iterated first -- a wrong answer, not a cosmetic one.
        schema = json.loads(json.dumps(SCHEMA))
        fact = schema["CHATBOTDB.PROFITABILITY.CUS_ORD_IVC_FCT"]
        fact["columns"].append({"name": "SUPPLIER_KEY", "type": "bigint"})
        schema["CHATBOTDB.PROFITABILITY.SUPPLIER_DIM"] = {
            "database": "CHATBOTDB", "schema": "PROFITABILITY",
            "table": "SUPPLIER_DIM",
            "columns": [{"name": "SUPPLIER_KEY", "type": "bigint"},
                        {"name": "SUPPLIER_NAME", "type": "nvarchar"}],
        }
        names = names_of(real_plan(schema))
        self.assertEqual(len(names), len(set(n.lower() for n in names)), names)

    def test_an_admins_own_words_are_never_rewritten(self):
        # Two things at once. The vocabulary is asked ONLY when the name is the
        # bare role, so a curated "Customer Segment" is not flattened back to
        # "Customer" by an expansion that knows nothing about this tenant's
        # segments. And routing through _dimension_label revives a fallback
        # that had gone dead: the loop already skips a dimension with no source
        # key, so the old line's `dim.get("name")` branch was unreachable and
        # every curated name was silently overwritten by its role.
        from core.semantic_model import build_runtime_semantic_plan, write_semantic_model

        schema = json.loads(json.dumps(SCHEMA))
        schema["CHATBOTDB.PROFITABILITY.CUS_ORD_IVC_FCT"]["columns"].append(
            {"name": "CUS_SEG_DMS_KEY", "type": "bigint"})
        schema["CHATBOTDB.PROFITABILITY.CUS_SEG_DMS"] = {
            "database": "CHATBOTDB", "schema": "PROFITABILITY",
            "table": "CUS_SEG_DMS",
            "columns": [{"name": "CUS_SEG_DMS_KEY", "type": "bigint"},
                        {"name": "CUS_SEG_NM", "type": "nvarchar"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            schema_dir, kb_dir = root / "schema", root / "kb"
            schema_dir.mkdir()
            (schema_dir / "_schema.json").write_text(json.dumps(schema),
                                                     encoding="utf-8")
            write_semantic_model(schema_dir=str(schema_dir), kb_dir=str(kb_dir))
            model = json.loads(
                (kb_dir / "_semantic_model.json").read_text(encoding="utf-8"))
            for table in model.get("tables", []):
                for dim in table.get("dimensions", []) or []:
                    if str(dim.get("source_key")) == "CUS_SEG_DMS_KEY":
                        dim["name"] = "Customer Segment"
            plan = build_runtime_semantic_plan(
                str(kb_dir), question="total invoice amount by month",
                selected_schema="PROFITABILITY", model=model)

        self.assertIn("Customer Segment", names_of(plan))
        self.assertNotIn("Customer", names_of(plan))

    def test_a_broken_vocabulary_costs_a_label_not_a_plan(self):
        import core.schema_enrichment as se

        with patch.object(se, "enrich_columns", side_effect=RuntimeError("boom")):
            plan = real_plan()
        self.assertTrue(plan["enabled"])
        # Back to the old spelling, and the plan still works.
        self.assertIn("Sup", names_of(plan))


class TestThereIsOnlyOneImplementation(unittest.TestCase):
    """The point of moving it rather than copying it."""

    def test_the_narrative_layer_delegates_to_the_shared_helper(self):
        # Executed, not read: patch the shared helper and check the narrative
        # layer's name changes with it. A second copy would not move.
        import core.schema_enrichment as se
        from core.response_builder import _display_label

        with patch.object(se, "display_label", return_value="Sentinel"):
            self.assertEqual(_display_label("WHS_NM"), "Sentinel")

    def test_the_shared_helper_expands_against_the_vocabulary(self):
        from core.schema_enrichment import display_label

        self.assertEqual(display_label("WHS_NM"), "Warehouse Name")
        self.assertEqual(display_label("sup"), "Supplier")

    def test_it_falls_back_to_the_plain_spelling(self):
        from core.schema_enrichment import display_label

        # An infrastructure field resolves to "data platform field: ...", which
        # is not a business name.
        self.assertEqual(display_label("AZ_UPD_TS"), "Az Upd Ts")
        self.assertEqual(display_label(""), "")

    def test_it_never_raises(self):
        import core.schema_enrichment as se
        from core.schema_enrichment import display_label

        with patch.object(se, "enrich_columns", side_effect=RuntimeError("boom")):
            self.assertEqual(display_label("WHS_NM"), "Whs Nm")


# ══════════════════════════════════════════════════════════════════════════════
# F10 · the chart's labels, and F13 · the scalar reply's caption
# ══════════════════════════════════════════════════════════════════════════════

CHART_ROWS = [{"WHS_NM": "Dallas", "BAL_VAL_AMT": 13_557_410.0, "ORD_QTY": 42},
              {"WHS_NM": "Chennai", "BAL_VAL_AMT": 9_100_000.0, "ORD_QTY": 31}]


class TestTheChartSaysWhatTheProseSays(unittest.TestCase):
    """F10 · core/chart_spec.py::_display_name was a plain underscore-strip.

    It left an all-caps column all-caps, so the chart's tooltip and axis label
    said "WHS NM" and "BAL VAL AMT" beside prose that L4 had already taught to
    say "Warehouse Name" and "Balance Value Amount" -- same reader, same answer
    card, two spellings of one column.
    """

    def labels(self, rows=None, chart_type="bar"):
        from core.chart import build_chart_payload

        payload = build_chart_payload(rows or CHART_ROWS, chart_type,
                                      question="balance by warehouse")
        return {k: v["label"] for k, v in payload["column_roles"].items()}

    def test_the_payload_the_browser_reads_carries_business_names(self):
        self.assertEqual(self.labels(), {
            "WHS_NM": "Warehouse Name",
            "BAL_VAL_AMT": "Balance Value Amount",
            "ORD_QTY": "Order Quantity",
        })

    def test_the_chart_and_the_prose_never_disagree(self):
        # The invariant the finding is actually about. Four of these six
        # differed before.
        from core.chart_spec import _display_name
        from core.response_builder import _display_label

        for column in ("WHS_NM", "BAL_VAL_AMT", "SUP_NM", "REVENUE",
                       "total_revenue", "AZ_UPD_TS", "count", "bin_label"):
            with self.subTest(column=column):
                self.assertEqual(_display_name(column), _display_label(column))

    def test_a_histogram_keeps_its_own_column_names(self):
        # The post-processor emits bin_label + count, and the chart type is
        # detected from exactly those keys.
        self.assertEqual(
            self.labels([{"bin_label": "0-10", "count": 5},
                         {"bin_label": "10-20", "count": 8}]),
            {"bin_label": "Bin Label", "count": "Count"})

    def test_a_broken_vocabulary_costs_a_label_not_a_chart(self):
        # The plain spelling, and a chart that still builds. Note it is
        # display_label's own fallback that produces this, not a second copy in
        # chart_spec -- which is why there is no try/except there to keep in
        # step with this one.
        import core.schema_enrichment as se

        with patch.object(se, "enrich_columns", side_effect=RuntimeError("boom")):
            labels = self.labels()
        self.assertEqual(labels["WHS_NM"], "Whs Nm")

    def test_nothing_in_nothing_out(self):
        from core.chart_spec import _display_name

        self.assertEqual(_display_name(""), "")
        self.assertEqual(_display_name(None), "")


class TestTheTeamsCardNamesItsAxes(unittest.TestCase):
    """F10, second half. The one surface that really does set axis titles."""

    def card(self, rows=None):
        from gateway.teams_chart_card import build_teams_chart_card
        from core.chart import build_chart_payload

        return build_teams_chart_card(build_chart_payload(
            rows or [{"WHS_NM": "Dallas", "BAL_VAL_AMT": 13_557_410.0},
                     {"WHS_NM": "Chennai", "BAL_VAL_AMT": 9_100_000.0}],
            "bar", question="balance by warehouse"))

    def test_the_axes_are_named_in_business_terms(self):
        element = self.card()["body"][-1]
        self.assertEqual(element["xAxisTitle"], "Warehouse Name")
        self.assertEqual(element["yAxisTitle"], "Balance Value Amount")

    def test_it_agrees_with_the_portal(self):
        from core.chart import build_chart_payload

        payload = build_chart_payload(
            [{"WHS_NM": "Dallas", "BAL_VAL_AMT": 13_557_410.0},
             {"WHS_NM": "Chennai", "BAL_VAL_AMT": 9_100_000.0}],
            "bar", question="balance by warehouse")
        element = self.card()["body"][-1]
        self.assertEqual(element["xAxisTitle"],
                         payload["column_roles"]["WHS_NM"]["label"])


class TestAnOrdinaryWordIsNotAnAbbreviation(unittest.TestCase):
    """Found while wiring the chart up, and live on the shipped L4 code.

    "count" is not in the segmenter's list of words that are already words, so
    the compact-code splitter was free to read it as CO + UNT and expand it to
    "company unit". A tenant asking "how many orders?" got a KPI card headed
    Company Unit -- produced by the very change that was meant to stop columns
    being printed in the warehouse's spelling, and shipped.

    It is also the single most common alias SQL produces.
    """

    def test_how_many_orders_is_not_a_company_unit(self):
        from core.response_builder import build_assistant_response

        answer = build_assistant_response(
            rows=[{"count": 40}], question="how many orders",
            sql="SELECT COUNT(*) AS count FROM orders", duration_ms=1200)
        self.assertEqual(answer["kpi"]["label"], "Count")

    def test_the_word_survives_wherever_it_appears(self):
        from core.schema_enrichment import display_label

        for spelling in ("count", "COUNT", "Count"):
            with self.subTest(spelling=spelling):
                self.assertEqual(display_label(spelling), "Count")

    def test_the_real_abbreviation_still_expands(self):
        # The fix must not cost the abbreviation table its job.
        from core.schema_enrichment import display_label

        self.assertEqual(display_label("CNT"), "Count")
        self.assertEqual(display_label("ORD_CNT"), "Order Count")
        self.assertEqual(display_label("CO_NM"), "Company Name")

    def test_no_ordinary_business_word_is_shredded(self):
        # The general statement, rather than one word patched in isolation.
        from core.schema_enrichment import display_label

        for word in ("count", "sum", "median", "minimum", "maximum", "total",
                     "region", "country", "company", "segment", "channel",
                     "customer", "supplier", "vendor", "product", "invoice",
                     "revenue", "margin", "quantity", "quarter", "status"):
            with self.subTest(word=word):
                self.assertEqual(display_label(word).lower(), word)


class TestTheScalarReplyOnAChatSurface(unittest.IsolatedAsyncioTestCase):
    """F13 · the caption Slack, Teams and Zoom see.

    The portal's KPI card guards on the identical condition and labels its
    column through the vocabulary. This branch -- the one every adapter without
    send_assistant_response falls through to -- printed the raw name, then
    printed it AGAIN underscore-stripped in the footer:

        BAL_VAL_AMT / *$13,557,410.00* / _1.2s · BAL VAL AMT_
    """

    class Adapter:
        def __init__(self):
            self.sent = []

        async def send_message(self, event, text):
            self.sent.append(text)

    async def reply(self, rows):
        from core.result_renderer import _send_results

        adapter = self.Adapter()
        await _send_results({}, adapter, "What is the total balance?", rows,
                            "SELECT 1", 1200, None, 1,
                            {"id": 1, "db_type": "azure_sql"},
                            question_id=None, confidence_context={})
        return "\n".join(adapter.sent)

    async def test_the_caption_is_the_business_name(self):
        text = await self.reply([{"BAL_VAL_AMT": 13_557_410}])
        self.assertIn("Balance Value Amount", text)

    async def test_neither_spelling_of_the_raw_column_survives(self):
        text = await self.reply([{"BAL_VAL_AMT": 13_557_410}])
        self.assertNotIn("BAL_VAL_AMT", text)
        self.assertNotIn("BAL VAL AMT", text)

    async def test_the_footer_stops_repeating_the_caption(self):
        text = await self.reply([{"BAL_VAL_AMT": 13_557_410}])
        self.assertTrue(text.rstrip().endswith("_1.2s_"), text[-60:])

    async def test_the_number_is_untouched(self):
        # col_name still drives currency detection; only the label changed. A
        # careless rebind would have changed the VALUE.
        text = await self.reply([{"BAL_VAL_AMT": 13_557_410}])
        self.assertIn("$13,557,410.00", text)

    async def test_it_matches_the_portal_for_the_same_answer(self):
        from core.response_builder import build_assistant_response

        rows = [{"BAL_VAL_AMT": 13_557_410}]
        portal = build_assistant_response(
            rows=rows, question="What is the total balance?",
            sql="SELECT 1", duration_ms=1200)
        self.assertIn(portal["kpi"]["label"], await self.reply(rows))

    async def test_a_column_the_vocabulary_cannot_expand_keeps_its_spelling(self):
        text = await self.reply([{"AZ_UPD_TS": "2026-01-01"}])
        self.assertIn("Az Upd Ts", text)
        self.assertNotIn("data platform field", text)


if __name__ == "__main__":
    unittest.main()
