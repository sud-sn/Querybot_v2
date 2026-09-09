"""
tests/test_last_raw_column_sites.py

The four surfaces still printing the warehouse's own spelling, after two
sweeps and a commit that said the family was closed.

  core/chart.py                 the PNG chart posted into Teams. matplotlib,
                                not ECharts, so it shares no code with the
                                browser charts that were fixed — and the axis
                                title is the ONLY thing naming the measure
                                there: no <th> beside it, no tooltip to hover.
  gateway/teams_chart_card.py   the Adaptive Card pie, which had the OTHER
                                defect too: one slice per row.
  portal/routes.py              a table tile pinned to a dashboard, whose <th>
                                said WHS_NM beside a chart tile on the same
                                page whose axis says "Warehouse Name".
  core/result_commands.py       the format-command clarification, asking the
                                reader to choose between BAL_VAL_AMT and
                                NET_AMT. F12 fixed two clarifications in this
                                file and missed this one.
"""
from __future__ import annotations

import unittest

from core import i18n


class TestThePngChartAxisTitles(unittest.TestCase):

    def test_the_axis_label_is_the_business_name(self):
        from core.chart import _axis_label

        self.assertEqual(_axis_label("BAL_VAL_AMT"), "Balance Value Amount")
        self.assertEqual(_axis_label("WHS_NM"), "Warehouse Name")

    def test_every_axis_call_goes_through_it(self):
        """A source check, narrowly: matplotlib draws to a buffer and the label
        cannot be read back without rendering a PNG and OCR-ing it. What can be
        pinned is that no call site sets an axis title from a raw column — and
        there are seven of them, so fixing one and missing six is exactly the
        pattern this file exists for."""
        import ast
        import inspect

        import core.chart as chart

        tree = ast.parse(inspect.getsource(chart))
        raw = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            attr = getattr(node.func, "attr", "")
            if attr not in {"set_xlabel", "set_ylabel"}:
                continue
            first = ast.unparse(node.args[0]) if node.args else ""
            if not first.startswith("_axis_label("):
                raw.append(f"line {node.lineno}: {first}")
        self.assertFalse(raw, "these axis titles print the raw column: " + str(raw))
        self.assertGreaterEqual(
            sum(1 for n in ast.walk(tree) if isinstance(n, ast.Call)
                and getattr(n.func, "attr", "") in {"set_xlabel", "set_ylabel"}),
            7, "the scan stopped finding the axis calls")


class TestTheTeamsCardPie(unittest.TestCase):

    GRID = [{"WHS_NM": name, "REV": value}
            for _ in ("Jan", "Feb", "Mar")
            for name, value in (("Halifax", 100.0), ("Calgary", 50.0))]

    def _slices(self, rows):
        from gateway.teams_chart_card import _pie_element

        element = _pie_element("pie", rows, "WHS_NM", "REV", "T")
        return [(d["legend"], d["value"]) for d in (element or {}).get("data", [])]

    def test_each_category_gets_one_slice(self):
        self.assertEqual(self._slices(self.GRID),
                         [("Halifax", 300.0), ("Calgary", 150.0)])

    def test_a_one_dimensional_result_is_unchanged(self):
        flat = [{"WHS_NM": "Halifax", "REV": 300.0},
                {"WHS_NM": "Calgary", "REV": 150.0}]
        self.assertEqual(self._slices(flat),
                         [("Halifax", 300.0), ("Calgary", 150.0)])

    def test_first_seen_order_is_kept(self):
        """Adaptive Cards colours slices by position, so a reordering changes
        the picture for no reason."""
        self.assertEqual([name for name, _ in self._slices(self.GRID)],
                         ["Halifax", "Calgary"])

    def test_a_category_that_nets_to_zero_is_dropped_not_drawn(self):
        """A pie slice of zero is a legend entry pointing at nothing."""
        rows = [{"WHS_NM": "A", "REV": 5.0}, {"WHS_NM": "A", "REV": -5.0},
                {"WHS_NM": "B", "REV": 3.0}]
        self.assertEqual(self._slices(rows), [("B", 3.0)])

    def test_an_empty_result_still_returns_nothing(self):
        from gateway.teams_chart_card import _pie_element

        self.assertIsNone(_pie_element("pie", [], "WHS_NM", "REV", "T"))


class TestThePinnedTableTile(unittest.TestCase):

    def test_the_header_labels_are_carried_beside_the_keys(self):
        """The template uses the column as BOTH the header text and the row
        lookup key, so the key stays raw and the label travels with it."""
        from core.schema_enrichment import display_label

        columns = ["WHS_NM", "BAL_VAL_AMT"]
        labels = {c: display_label(c) for c in columns}
        self.assertEqual(labels["WHS_NM"], "Warehouse Name")
        self.assertEqual(labels["BAL_VAL_AMT"], "Balance Value Amount")

    def test_the_template_renders_the_label_and_keys_on_the_code(self):
        from pathlib import Path

        page = (Path(__file__).resolve().parents[1] / "portal" / "templates"
                / "portal_dashboard.html").read_text(encoding="utf-8")
        header_row = next(line for line in page.splitlines()
                          if "chart.table_columns" in line and "<thead>" in line)
        # BOTH renderings -- the sortable button and the plain header. Checking
        # that the label expression appears at all passes with one of the two
        # still printing the raw code, which is what a mutation proved.
        self.assertEqual(
            header_row.count("chart.table_column_labels.get(col, col)"), 2,
            "one of the two header renderings still prints the raw column")
        self.assertNotIn(">{{ col }}<", header_row)
        # The cell still looks the row up by the RAW key.
        body_row = next(line for line in page.splitlines()
                        if "for row in chart.table_rows" in line)
        self.assertIn("row[col]", body_row)

    def test_the_route_always_defines_the_map(self):
        """A tile built on a path that skips the assignment would raise in the
        template rather than fall back."""
        import ast
        import inspect

        import portal.routes as routes

        tree = ast.parse(inspect.getsource(routes))
        assigned = [
            node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.Assign) and node.targets
            and "table_column_labels" in ast.unparse(node.targets[0])
        ]
        self.assertGreaterEqual(
            len(assigned), 2,
            "the label map is set on the populate path but never defaulted, so "
            "a tile that skips it raises in the template")


class TestTheFormatClarificationLabels(unittest.TestCase):

    def _clarification(self, lang):
        """The real format command, run to the point where it asks."""
        import core.result_commands as rc
        from core.result_cache import ResultCache

        rows = [{"BAL_VAL_AMT": 1.0, "NET_AMT": 2.0},
                {"BAL_VAL_AMT": 3.0, "NET_AMT": 4.0}]
        cache = ResultCache(max_sessions=2)
        session_id = "acct:user"
        result_id = cache.store(session_id, rows, "q", "SELECT 1")
        source = {"result_id": result_id, "rows": rows,
                  "column_formats": {}, "metadata": {}, "question": "q"}

        token = i18n.activate_language(lang)
        try:
            return rc._execute_format_command(
                session_id, source, rc.parse_result_command("format as currency"),
                cache=cache)
        finally:
            i18n.deactivate_language(token)

    def test_the_column_choice_is_offered_by_business_name(self):
        """Executed, not asserted on the helper: the first version of this test
        called _column_display_label directly, which passes with the call site
        still handing over the raw column."""
        options = self._clarification("en").clarification_options
        self.assertEqual([o["label"] for o in options],
                         ["Balance Value Amount", "Net Amount"])

    def test_but_the_question_keeps_the_real_column(self):
        """core/dispatcher.py re-plans the value, and "format Balance Value
        Amount as currency" names no column the warehouse has."""
        for option in self._clarification("fr").clarification_options:
            with self.subTest(option=option):
                self.assertRegex(option["value"], r"^format [A-Z_]+ as currency$")
                self.assertEqual(option["value"], option["resolved_question"])

    def test_the_format_labels_are_translated(self):
        for msg_id in ("reply.rc.fmt.currency", "reply.rc.fmt.percentage",
                       "reply.rc.fmt.number", "reply.rc.fmt.percent_fraction",
                       "reply.rc.fmt.full_month_year"):
            with self.subTest(msg_id=msg_id):
                english = i18n.lookup(msg_id, "en")
                french = i18n.lookup(msg_id, "fr")
                self.assertTrue(english.strip() and french.strip())
                self.assertNotEqual(english, msg_id)
                self.assertNotEqual(english, french)

    def test_the_month_sample_shows_what_this_reader_would_see(self):
        """It is a sample of the OUTPUT, so an English rendering of it beside a
        French result is a lie about what pressing the button will do."""
        self.assertEqual(i18n.lookup("reply.rc.fmt.full_month_year", "fr"),
                         "janvier 2026")


if __name__ == "__main__":
    unittest.main()
