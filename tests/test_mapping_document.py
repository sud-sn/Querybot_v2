# -*- coding: utf-8 -*-
"""tests/test_mapping_document.py

The source mapping document: joins and column terms as a file a data team can
actually produce, edit in a spreadsheet, and send back.

Before this, joins were editable one at a time on a canvas or in bulk through a
JSON endpoint driven by that canvas, and the one import that existed was a
whole-graph REPLACE keyed on internal ids — a clone tool, not a mapping
document. Column terms were a free-text box an admin had to think of unprompted.

Three properties decide whether this is usable, and most of this file is about
holding them:

  **Tables, not entity names.** The file speaks the warehouse's language and
  the importer resolves to entities on the way in.

  **Additive, and a dry run first.** Nothing is written until the plan has been
  shown; an upload amends what it names and leaves the rest alone.

  **Round trip.** What comes out goes back in and reproduces itself, composite
  keys included.

Every test here runs the real parse, the real plan, the real apply, or the real
route against a real database. The one end-to-end test starts at the file and
ends at core.vocab_packs — nothing is handed in by the test in between.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.mapping_csv import (  # noqa: E402
    JOIN_COLUMNS,
    MAX_ROWS,
    TERM_COLUMNS,
    group_composites,
    parse_joins,
    parse_terms,
    render_joins,
    render_terms,
)
from core.mapping_import import (  # noqa: E402
    CREATE,
    REJECT,
    UNCHANGED,
    UPDATE,
    apply_joins,
    apply_terms,
    plan_joins,
    plan_terms,
)

ROOT = Path(__file__).resolve().parents[1]

ENTITIES = {
    "Orders":   ("F_ORDER",      "SALES", "fact"),
    "Shipments": ("F_SHIPMENT",  "SALES", "fact"),
    "Customer": ("DIM_CUSTOMER", "SALES", "dimension"),
    "Calendar": ("DIM_DATE",     "SALES", "dimension"),
    "Warehouse": ("DIM_WAREHOUSE", "SALES", "dimension"),
    # Same bare table name, another schema. A warehouse that keeps a live and
    # an archived copy of a table is ordinary, and a mapping document that
    # qualifies its tables has to reach the one it named.
    "ArchivedOrders": ("F_ORDER", "ARCHIVE", "fact"),
}

SCHEMA = {
    "SALES.F_ORDER": {"CUSTOMER_ID": "int", "ORDER_DATE_KEY": "int",
                      "DELIVERY_DATE_KEY": "int", "WAREHOUSE_ID": "int",
                      "DIVISION": "varchar", "BAL_VAL_AMT": "decimal"},
    "SALES.F_SHIPMENT": {"CUSTOMER_ID": "int", "WAREHOUSE_ID": "int"},
    "SALES.DIM_CUSTOMER": {"CUSTOMER_ID": "int", "DIVISION": "varchar"},
    "SALES.DIM_DATE": {"DATE_KEY": "int"},
    "SALES.DIM_WAREHOUSE": {"WAREHOUSE_ID": "int"},
    "ARCHIVE.F_ORDER": {"CUSTOMER_ID": "int"},
}


def csv_text(header, *rows) -> str:
    """A CSV the way a spreadsheet writes one, quoting included.

    Written with the real csv module rather than by joining on commas: a terms
    cell holds a comma-separated list, and a fixture that does not quote it is
    testing a file nobody would ever upload.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(list(header))
    for row in rows:
        writer.writerow([str(cell) for cell in row])
    return buffer.getvalue()


JOIN_HEADER = list(JOIN_COLUMNS)


def join_row(from_table="SALES.F_ORDER", from_column="CUSTOMER_ID",
             to_table="SALES.DIM_CUSTOMER", to_column="CUSTOMER_ID",
             relationship="many_to_one", join_type="LEFT", label="", group=""):
    return [from_table, from_column, to_table, to_column,
            relationship, join_type, label, group]


# ── The parser, on its own ────────────────────────────────────────────────────

class TestReadingTheFile(unittest.TestCase):

    def test_a_good_file_parses_every_row(self):
        result = parse_joins(csv_text(
            JOIN_HEADER,
            join_row(),
            join_row(from_column="WAREHOUSE_ID", to_table="SALES.DIM_WAREHOUSE",
                     to_column="WAREHOUSE_ID")))
        self.assertEqual(result.fatal, "")
        self.assertEqual(result.problems, [])
        self.assertEqual([r.from_column for r in result.rows],
                         ["CUSTOMER_ID", "WAREHOUSE_ID"])

    def test_each_bad_row_is_reported_on_its_own_line_and_the_others_survive(self):
        result = parse_joins(csv_text(
            JOIN_HEADER,
            join_row(),                                   # line 2, good
            join_row(from_column=""),                     # line 3, blank required
            join_row(relationship="sort_of_related"),     # line 4
            join_row(join_type="SIDEWAYS"),               # line 5
            join_row(from_column="WAREHOUSE_ID", to_table="SALES.DIM_WAREHOUSE",
                     to_column="WAREHOUSE_ID")))          # line 6, good
        self.assertEqual([(p.line, p.reason) for p in result.problems],
                         [(3, "missing_value"), (4, "bad_relationship"),
                          (5, "bad_join_type")])
        self.assertEqual(len(result.rows), 2)

    def test_a_problem_says_what_would_fix_it(self):
        problem = parse_joins(csv_text(
            JOIN_HEADER, join_row(relationship="1:n"))).problems[0]
        self.assertIn("1:n", problem.detail)
        self.assertIn("many_to_one", problem.detail)

    def test_a_blank_line_between_blocks_is_not_an_error(self):
        text = csv_text(JOIN_HEADER, join_row()) + "\n" + \
            ",".join(join_row(from_column="WAREHOUSE_ID",
                              to_table="SALES.DIM_WAREHOUSE",
                              to_column="WAREHOUSE_ID")) + "\n"
        result = parse_joins(text)
        self.assertEqual(result.problems, [])
        self.assertEqual(len(result.rows), 2)

    def test_a_file_that_has_been_through_a_spreadsheet_still_loads(self):
        # BOM, upper-case header, spaces around the names. Every one of these
        # is what Excel hands back, and any of them used to be a dead file.
        text = "﻿" + csv_text([h.upper().replace("_", "_") for h in JOIN_HEADER],
                                   join_row())
        text = text.replace("FROM_TABLE", " From_Table ")
        result = parse_joins(text)
        self.assertEqual(result.fatal, "")
        self.assertEqual(len(result.rows), 1)

    def test_a_missing_header_column_names_it_rather_than_failing_row_by_row(self):
        result = parse_joins(csv_text(
            ["from_table", "from_column", "to_table"], ["a", "b", "c"]))
        self.assertIn("to_column", result.fatal)
        self.assertEqual(result.rows, [])

    def test_an_empty_file_says_so(self):
        self.assertIn("empty", parse_joins("   ").fatal)

    def test_an_export_sized_file_is_refused_whole_rather_than_half_applied(self):
        rows = [join_row(label=f"j{i}") for i in range(MAX_ROWS + 5)]
        result = parse_joins(csv_text(JOIN_HEADER, *rows))
        self.assertIn(str(MAX_ROWS), result.fatal)
        self.assertEqual(result.rows, [])

    def test_the_defaults_are_the_ones_a_person_would_leave_blank(self):
        row = parse_joins(csv_text(
            JOIN_HEADER,
            ["SALES.F_ORDER", "CUSTOMER_ID", "SALES.DIM_CUSTOMER",
             "CUSTOMER_ID", "", "", "", ""])).rows[0]
        self.assertEqual(row.relationship, "many_to_one")
        self.assertEqual(row.join_type, "LEFT")


class TestCompositeKeys(unittest.TestCase):

    def test_rows_sharing_a_group_are_one_join_with_two_pairs(self):
        rows = parse_joins(csv_text(
            JOIN_HEADER,
            join_row(from_column="DIVISION", to_column="DIVISION", group="k1"),
            join_row(from_column="CUSTOMER_ID", to_column="CUSTOMER_ID",
                     group="k1"))).rows
        folded = group_composites(rows)
        self.assertEqual(len(folded), 1)
        self.assertEqual([p["from_col"] for p in folded[0][1]],
                         ["DIVISION", "CUSTOMER_ID"])

    def test_two_blank_groups_are_not_the_same_composite(self):
        # The mistake a `group or ''` key would make: every ungrouped row
        # between the same two tables silently welded into one composite.
        rows = parse_joins(csv_text(
            JOIN_HEADER,
            join_row(from_column="ORDER_DATE_KEY", to_table="SALES.DIM_DATE",
                     to_column="DATE_KEY"),
            join_row(from_column="DELIVERY_DATE_KEY", to_table="SALES.DIM_DATE",
                     to_column="DATE_KEY"))).rows
        folded = group_composites(rows)
        self.assertEqual(len(folded), 2)
        self.assertEqual([len(pairs) for _, pairs in folded], [1, 1])

    def test_a_group_is_scoped_to_its_table_pair(self):
        rows = parse_joins(csv_text(
            JOIN_HEADER,
            join_row(group="k"),
            join_row(from_column="WAREHOUSE_ID", to_table="SALES.DIM_WAREHOUSE",
                     to_column="WAREHOUSE_ID", group="k"))).rows
        self.assertEqual(len(group_composites(rows)), 2)


class TestReadingTermFiles(unittest.TestCase):

    def test_terms_split_on_commas_inside_one_cell(self):
        row = parse_terms(
            'table,column,terms\nSALES.F_ORDER,BAL_VAL_AMT,'
            '"inventory value, stock value, balance value"\n').rows[0]
        self.assertEqual(row.terms,
                         ("inventory value", "stock value", "balance value"))

    def test_the_same_term_twice_is_kept_once(self):
        row = parse_terms(
            'table,column,terms\nT,C,"stock value, Stock  Value"\n').rows[0]
        self.assertEqual(row.terms, ("stock value",))

    def test_clearing_a_column_is_allowed_and_reported(self):
        result = parse_terms(csv_text(TERM_COLUMNS, ["T", "C", ""]))
        self.assertEqual(result.rows[0].terms, ())
        self.assertEqual(result.problems[0].reason, "no_terms")

    def test_a_row_with_no_column_is_a_problem_not_a_row(self):
        result = parse_terms(csv_text(TERM_COLUMNS, ["T", "", "a, b"]))
        self.assertEqual(result.rows, [])
        self.assertEqual(result.problems[0].reason, "missing_value")


class TestTheRoundTrip(unittest.TestCase):
    """Download, edit in a spreadsheet, upload. A format you can only write is
    a format nobody maintains."""

    RELATIONSHIPS = [
        {"id": 1, "from_entity": "Orders", "to_entity": "Customer",
         "from_column": "CUSTOMER_ID", "to_column": "CUSTOMER_ID",
         "relationship_type": "many_to_one", "join_type": "LEFT",
         "label": "placed by", "join_conditions": None},
        {"id": 2, "from_entity": "Orders", "to_entity": "Calendar",
         "from_column": "ORDER_DATE_KEY", "to_column": "DATE_KEY",
         "relationship_type": "many_to_one", "join_type": "INNER",
         "label": "ordered on", "join_conditions": None},
        {"id": 3, "from_entity": "Orders", "to_entity": "Warehouse",
         "from_column": "DIVISION", "to_column": "DIVISION",
         "relationship_type": "many_to_one", "join_type": "LEFT", "label": "",
         "relationship_key": "wh",
         "join_conditions": json.dumps(
             [{"from_col": "WAREHOUSE_ID", "to_col": "WAREHOUSE_ID"}])},
    ]

    def _table_for(self, name):
        table, schema, _ = ENTITIES[name]
        return f"{schema}.{table}"

    def test_what_comes_out_goes_back_in(self):
        text = render_joins(self.RELATIONSHIPS, self._table_for)
        result = parse_joins(text)
        self.assertEqual(result.fatal, "")
        self.assertEqual(result.problems, [])
        folded = group_composites(result.rows)
        self.assertEqual(
            [(row.from_table, row.to_table, len(pairs)) for row, pairs in folded],
            [("SALES.F_ORDER", "SALES.DIM_CUSTOMER", 1),
             ("SALES.F_ORDER", "SALES.DIM_DATE", 1),
             ("SALES.F_ORDER", "SALES.DIM_WAREHOUSE", 2)])

    def test_the_composite_survives_the_trip_rather_than_flattening(self):
        folded = group_composites(
            parse_joins(render_joins(self.RELATIONSHIPS, self._table_for)).rows)
        _, pairs = folded[-1]
        self.assertEqual([(p["from_col"], p["to_col"]) for p in pairs],
                         [("DIVISION", "DIVISION"),
                          ("WAREHOUSE_ID", "WAREHOUSE_ID")])

    def test_an_entity_with_no_table_is_left_out_rather_than_written_blank(self):
        # Not merely "produces no importable row" — a row with a blank table
        # would come back as a problem on every re-import, so the export has
        # to omit it entirely.
        text = render_joins(
            [{"from_entity": "Orders", "to_entity": "Ghost",
              "from_column": "A", "to_column": "B"}],
            lambda name: self._table_for(name) if name in ENTITIES else "")
        self.assertEqual(text.strip().splitlines()[1:], [])
        result = parse_joins(text)
        self.assertEqual(result.rows, [])
        self.assertEqual(result.problems, [])

    def test_terms_come_back_in_the_shape_they_go_out(self):
        text = render_terms({
            "SALES.F_ORDER": {"column_synonym_map": {
                "BAL_VAL_AMT": ["inventory value", "stock value"]}},
        })
        row = parse_terms(text).rows[0]
        self.assertEqual(row.table, "SALES.F_ORDER")
        self.assertEqual(row.column, "BAL_VAL_AMT")
        self.assertEqual(row.terms, ("inventory value", "stock value"))


# ── Against a real workspace ──────────────────────────────────────────────────

class RealWorkspace(unittest.TestCase):

    def setUp(self):
        import store

        self._dir = tempfile.mkdtemp(prefix="qb-mapping-")
        self._saved = {k: os.environ.get(k) for k in ("DB_PATH", "QUERYBOT_DB_PATH")}
        path = os.path.join(self._dir, "d.db")
        os.environ["DB_PATH"] = path
        os.environ["QUERYBOT_DB_PATH"] = path
        store.init_db()
        self.account_id = f"acct-map-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")

        schema_dir = os.path.join(self._dir, "schema")
        os.makedirs(schema_dir)
        (Path(schema_dir) / "_schema.json").write_text(json.dumps({
            fqn: {"columns": [{"name": c, "type": t} for c, t in cols.items()]}
            for fqn, cols in SCHEMA.items()
        }), encoding="utf-8")
        store.update_client_state(self.account_id, "READY",
                                  {"schema_dir": schema_dir})
        for name, (table, schema, kind) in ENTITIES.items():
            store.save_entity(account_id=self.account_id, entity_name=name,
                              table_name=table, schema_name=schema,
                              entity_type=kind)

    def tearDown(self):
        try:
            from core.vocab_packs import forget_account_vocab
            forget_account_vocab(self.account_id)
        except Exception:
            pass
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self._dir, ignore_errors=True)

    def plan(self, *rows, header=None):
        return plan_joins(self.account_id,
                          parse_joins(csv_text(header or JOIN_HEADER, *rows)))

    def apply(self, *rows):
        plan = self.plan(*rows)
        return plan, apply_joins(self.account_id, plan)

    def stored(self):
        import store

        return store.list_relationships(self.account_id, active_only=False)

    def actions(self, plan):
        return [(row.line, row.action) for row in plan.rows]


class TestTheDryRun(RealWorkspace):

    def test_planning_writes_nothing(self):
        plan = self.plan(join_row())
        self.assertEqual(self.actions(plan), [(2, CREATE)])
        self.assertEqual(self.stored(), [])

    def test_applying_writes_what_the_plan_said(self):
        plan, result = self.apply(join_row(label="placed by"))
        self.assertEqual(result["applied"], 1)
        rows = self.stored()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["from_entity"], "Orders")
        self.assertEqual(rows[0]["to_entity"], "Customer")
        self.assertEqual(rows[0]["label"], "placed by")

    def test_only_the_accepted_rows_are_written(self):
        plan, result = self.apply(
            join_row(),                                          # good
            join_row(to_table="SALES.NOT_A_TABLE"),              # unknown
            join_row(from_column="WAREHOUSE_ID",
                     to_table="SALES.DIM_WAREHOUSE",
                     to_column="WAREHOUSE_ID"))                  # good
        self.assertEqual(result["applied"], 2)
        self.assertEqual(len(self.stored()), 2)
        self.assertEqual(result["rejected"], 1)

    def test_a_fatal_file_plans_nothing_at_all(self):
        plan = plan_joins(self.account_id, parse_joins("not,a,mapping\n1,2,3\n"))
        self.assertTrue(plan.fatal)
        self.assertEqual(plan.rows, [])
        self.assertEqual(apply_joins(self.account_id, plan)["applied"], 0)

    def test_a_file_that_cannot_be_read_does_not_go_near_the_workspace(self):
        # A hundred-thousand-row file should not load the whole graph before
        # deciding it was never readable.
        import store

        with patch.object(store, "list_relationships") as listed:
            plan_joins(self.account_id, parse_joins("not,a,mapping\n1,2,3\n"))
        listed.assert_not_called()

    def test_terms_that_cannot_be_read_do_not_go_near_the_workspace(self):
        import store

        with patch.object(store, "list_table_descriptions") as listed:
            plan_terms(self.account_id, parse_terms("nope\n1\n"))
        listed.assert_not_called()


class TestResolvingTables(RealWorkspace):

    def test_a_table_the_workspace_does_not_have_is_named_in_the_rejection(self):
        row = self.plan(join_row(to_table="SALES.DIM_SUPPLIER")).rows[0]
        self.assertEqual(row.action, REJECT)
        self.assertIn("SALES.DIM_SUPPLIER", row.detail)

    def test_a_bare_table_name_resolves_to_the_schema_qualified_entity(self):
        # The file is written by a data team who may or may not qualify.
        self.assertEqual(
            self.actions(self.plan(join_row(from_table="F_ORDER",
                                            to_table="DIM_CUSTOMER"))),
            [(2, CREATE)])

    def test_an_over_qualified_table_name_still_resolves(self):
        self.assertEqual(
            self.actions(self.plan(join_row(from_table="WAREHOUSE.SALES.F_ORDER",
                                            to_table="WAREHOUSE.SALES.DIM_CUSTOMER"))),
            [(2, CREATE)])

    def test_the_schema_qualifier_decides_between_two_tables_of_the_same_name(self):
        # SALES.F_ORDER and ARCHIVE.F_ORDER are different tables. Indexing on
        # the bare name alone gives whichever was inserted first, so a file
        # that qualifies would silently join the wrong one.
        plan = self.plan(join_row(from_table="ARCHIVE.F_ORDER"))
        self.assertEqual(self.actions(plan), [(2, CREATE)])
        apply_joins(self.account_id, plan)
        self.assertEqual(self.stored()[0]["from_entity"], "ArchivedOrders")

    def test_the_other_schemas_copy_is_reached_by_its_own_qualifier(self):
        plan = self.plan(join_row(from_table="SALES.F_ORDER"))
        apply_joins(self.account_id, plan)
        self.assertEqual(self.stored()[0]["from_entity"], "Orders")

    def test_the_file_never_has_to_know_the_entity_name(self):
        # The graph calls SALES.F_ORDER "Orders" because somebody typed it on
        # a canvas. A file naming the entity is a file only this product could
        # have written.
        row = self.plan(join_row(from_table="Orders", to_table="Customer")).rows[0]
        self.assertEqual(row.action, REJECT)


class TestGovernanceStillApplies(RealWorkspace):

    def test_a_fact_to_fact_join_is_refused_by_the_import_too(self):
        row = self.plan(join_row(to_table="SALES.F_SHIPMENT")).rows[0]
        self.assertEqual(row.action, REJECT)
        self.assertIn("fact-to-fact", row.detail)
        apply_joins(self.account_id, self.plan(join_row(to_table="SALES.F_SHIPMENT")))
        self.assertEqual(self.stored(), [])

    def test_the_wrong_direction_is_refused(self):
        row = self.plan(join_row(from_table="SALES.DIM_CUSTOMER",
                                 from_column="CUSTOMER_ID",
                                 to_table="SALES.F_ORDER")).rows[0]
        self.assertEqual(row.action, REJECT)

    def test_a_column_the_schema_does_not_have_is_stored_and_flagged(self):
        # A column absent from the DISCOVERED schema may be a real column
        # behind a stale discovery — flagged, not refused.
        plan, result = self.apply(join_row(from_column="CUST_NO"))
        self.assertEqual(result["applied"], 1)
        self.assertEqual(self.stored()[0]["validation_status"], "broken")

    def test_a_verified_join_is_not_flagged(self):
        self.apply(join_row())
        self.assertNotEqual(self.stored()[0]["validation_status"], "broken")

    def test_the_flag_reason_reaches_the_preview(self):
        row = self.plan(join_row(from_column="CUST_NO")).rows[0]
        self.assertEqual(row.action, CREATE)
        self.assertIn("CUST_NO", row.detail)

    def test_the_summary_counts_what_will_be_stored_unverified(self):
        # A caveat the admin has to see BEFORE applying, not find on the graph
        # afterwards.
        plan = self.plan(join_row(from_column="CUST_NO"), join_row())
        summary = plan.summary()
        self.assertEqual(summary["created"], 2)
        self.assertEqual(summary["flagged"], 1)

    def test_nothing_is_flagged_when_every_column_is_real(self):
        self.assertEqual(self.plan(join_row()).summary()["flagged"], 0)


class TestAdditiveNotReplace(RealWorkspace):

    def test_uploading_the_same_file_twice_changes_nothing_the_second_time(self):
        self.apply(join_row(label="placed by"))
        first = self.stored()
        plan, result = self.apply(join_row(label="placed by"))
        self.assertEqual(self.actions(plan), [(2, UNCHANGED)])
        self.assertEqual(result["applied"], 0)
        self.assertEqual([r["id"] for r in self.stored()], [r["id"] for r in first])

    def test_an_edit_amends_the_row_that_is_already_there(self):
        self.apply(join_row(label="placed by"))
        rel_id = self.stored()[0]["id"]
        plan, _ = self.apply(join_row(label="ordered by", join_type="INNER"))
        self.assertEqual(self.actions(plan), [(2, UPDATE)])
        rows = self.stored()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], rel_id)
        self.assertEqual(rows[0]["label"], "ordered by")
        self.assertEqual(rows[0]["join_type"], "INNER")

    def test_the_preview_says_what_the_edit_changes(self):
        self.apply(join_row(label="placed by"))
        row = self.plan(join_row(label="ordered by")).rows[0]
        self.assertIn("placed by", row.detail)
        self.assertIn("ordered by", row.detail)

    def test_a_file_naming_one_join_leaves_the_others_alone(self):
        # The whole-graph JSON import this sits beside would have deleted them.
        self.apply(join_row(),
                   join_row(from_column="WAREHOUSE_ID",
                            to_table="SALES.DIM_WAREHOUSE",
                            to_column="WAREHOUSE_ID"))
        self.apply(join_row(label="renamed"))
        self.assertEqual(len(self.stored()), 2)

    def test_two_role_playing_joins_do_not_overwrite_each_other(self):
        # Order date and delivery date both point at the calendar. Matching a
        # stored join on the ENTITY pair alone would make the second row of
        # this file overwrite the first.
        plan, result = self.apply(
            join_row(from_column="ORDER_DATE_KEY", to_table="SALES.DIM_DATE",
                     to_column="DATE_KEY", label="ordered on"),
            join_row(from_column="DELIVERY_DATE_KEY", to_table="SALES.DIM_DATE",
                     to_column="DATE_KEY", label="delivered on"))
        self.assertEqual(result["applied"], 2)
        rows = self.stored()
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["from_column"] for r in rows},
                         {"ORDER_DATE_KEY", "DELIVERY_DATE_KEY"})

    def test_re_uploading_role_playing_joins_still_matches_each_to_itself(self):
        rows = (join_row(from_column="ORDER_DATE_KEY", to_table="SALES.DIM_DATE",
                         to_column="DATE_KEY", label="ordered on"),
                join_row(from_column="DELIVERY_DATE_KEY", to_table="SALES.DIM_DATE",
                         to_column="DATE_KEY", label="delivered on"))
        self.apply(*rows)
        plan, result = self.apply(*rows)
        self.assertEqual([a for _, a in self.actions(plan)],
                         [UNCHANGED, UNCHANGED])
        self.assertEqual(len(self.stored()), 2)


class TestCompositeImport(RealWorkspace):

    ROWS = (join_row(from_column="DIVISION", to_table="SALES.DIM_WAREHOUSE",
                     to_column="WAREHOUSE_ID", group="wh"),
            join_row(from_column="WAREHOUSE_ID", to_table="SALES.DIM_WAREHOUSE",
                     to_column="WAREHOUSE_ID", group="wh"))

    def test_a_grouped_pair_becomes_one_join_with_a_second_condition(self):
        plan, result = self.apply(*self.ROWS)
        self.assertEqual(result["applied"], 1)
        rows = self.stored()
        self.assertEqual(len(rows), 1)
        conditions = rows[0]["join_conditions"]
        if isinstance(conditions, str):
            conditions = json.loads(conditions or "[]")
        self.assertEqual([(c["from_col"], c["to_col"]) for c in conditions],
                         [("WAREHOUSE_ID", "WAREHOUSE_ID")])
        self.assertEqual(rows[0]["from_column"], "DIVISION")

    def test_re_uploading_a_composite_recognises_it(self):
        self.apply(*self.ROWS)
        plan, result = self.apply(*self.ROWS)
        self.assertEqual([a for _, a in self.actions(plan)], [UNCHANGED])
        self.assertEqual(len(self.stored()), 1)

    def test_the_stored_composite_exports_and_re_imports_unchanged(self):
        self.apply(*self.ROWS)
        text = render_joins(self.stored(),
                            lambda name: "{}.{}".format(*reversed(ENTITIES[name][:2]))
                            if name in ENTITIES else "")
        plan = plan_joins(self.account_id, parse_joins(text))
        self.assertEqual([a for _, a in self.actions(plan)], [UNCHANGED])


# ── Column terms ──────────────────────────────────────────────────────────────

TERM_HEADER = list(TERM_COLUMNS)


class TestColumnTerms(RealWorkspace):

    def terms_plan(self, *rows):
        return plan_terms(self.account_id, parse_terms(csv_text(TERM_HEADER, *rows)))

    def terms_apply(self, *rows):
        plan = self.terms_plan(*rows)
        return plan, apply_terms(self.account_id, plan)

    def saved(self, table="SALES.F_ORDER"):
        import store

        entry = (store.list_table_descriptions(self.account_id) or {}).get(table) or {}
        return entry.get("column_synonym_map") or {}

    def test_a_term_file_writes_the_terms(self):
        plan, result = self.terms_apply(
            ["SALES.F_ORDER", "BAL_VAL_AMT", "inventory value, stock value"])
        self.assertEqual(result["applied"], 1)
        self.assertEqual(self.saved()["BAL_VAL_AMT"],
                         ["inventory value", "stock value"])

    def test_planning_terms_writes_nothing(self):
        self.terms_plan(["SALES.F_ORDER", "BAL_VAL_AMT", "inventory value"])
        self.assertEqual(self.saved(), {})

    def test_a_second_file_merges_rather_than_dropping_the_first(self):
        # A file naming three columns of a table must not silently delete the
        # terms an admin typed for its other twenty.
        self.terms_apply(["SALES.F_ORDER", "BAL_VAL_AMT", "inventory value"])
        self.terms_apply(["SALES.F_ORDER", "DIVISION", "business unit"])
        self.assertEqual(sorted(self.saved()), ["BAL_VAL_AMT", "DIVISION"])

    def test_the_same_file_twice_is_unchanged(self):
        row = ["SALES.F_ORDER", "BAL_VAL_AMT", "inventory value"]
        self.terms_apply(row)
        plan, result = self.terms_apply(row)
        self.assertEqual([r.action for r in plan.rows], [UNCHANGED])
        self.assertEqual(result["applied"], 0)

    def test_editing_a_column_replaces_that_columns_terms(self):
        self.terms_apply(["SALES.F_ORDER", "BAL_VAL_AMT", "inventory value"])
        plan, _ = self.terms_apply(["SALES.F_ORDER", "BAL_VAL_AMT", "stock value"])
        self.assertEqual([r.action for r in plan.rows], [UPDATE])
        self.assertEqual(self.saved()["BAL_VAL_AMT"], ["stock value"])

    def test_an_empty_terms_cell_clears_that_column_only(self):
        self.terms_apply(["SALES.F_ORDER", "BAL_VAL_AMT", "inventory value"],
                         ["SALES.F_ORDER", "DIVISION", "business unit"])
        self.terms_apply(["SALES.F_ORDER", "BAL_VAL_AMT", ""])
        self.assertEqual(sorted(self.saved()), ["DIVISION"])

    def test_a_column_the_schema_does_not_have_is_saved_and_said_so(self):
        plan, result = self.terms_apply(
            ["SALES.F_ORDER", "MADE_UP_COL", "whatever"])
        self.assertEqual(result["applied"], 1)
        self.assertIn("schema", plan.rows[0].detail)

    def test_the_lower_case_spelling_of_a_column_amends_rather_than_duplicates(self):
        self.terms_apply(["SALES.F_ORDER", "BAL_VAL_AMT", "inventory value"])
        plan, _ = self.terms_apply(["SALES.F_ORDER", "bal_val_amt", "stock value"])
        self.assertEqual([r.action for r in plan.rows], [UPDATE])
        self.assertEqual(self.saved(), {"BAL_VAL_AMT": ["stock value"]})


class TestTheTermsReachTheVocabulary(RealWorkspace):
    """The end-to-end one: start at the file, end at the reader, hand nothing in
    between. A term that is saved and never resolved is the failure this whole
    feature exists to prevent."""

    def test_a_word_uploaded_in_a_file_is_a_word_the_planner_can_resolve(self):
        from core.vocab_packs import forget_account_vocab, vocab_for_account

        forget_account_vocab(self.account_id)
        before = vocab_for_account(self.account_id)
        self.assertNotIn("stockholding value",
                         [t.lower() for t in
                          (before.direct_aliases or {}).get("BAL_VAL_AMT", [])])

        apply_terms(self.account_id, plan_terms(self.account_id, parse_terms(
            csv_text(TERM_HEADER,
                     ["SALES.F_ORDER", "BAL_VAL_AMT", "stockholding value"]))))

        after = vocab_for_account(self.account_id)
        self.assertIn("stockholding value",
                      [t.lower() for t in
                       (after.direct_aliases or {}).get("BAL_VAL_AMT", [])])

    def test_the_cache_is_dropped_even_though_no_file_changed(self):
        # The vocabulary cache key is built from file MTIMES, so a term saved
        # to the database changes nothing it watches. Without the explicit
        # invalidation the stale vocabulary is served until the process
        # restarts — "I saved it and nothing happened".
        from core.vocab_packs import vocab_for_account

        vocab_for_account(self.account_id)          # warm the cache
        apply_terms(self.account_id, plan_terms(self.account_id, parse_terms(
            csv_text(TERM_HEADER, ["SALES.F_ORDER", "DIVISION", "profit centre"]))))
        self.assertIn("profit centre",
                      [t.lower() for t in
                       (vocab_for_account(self.account_id).direct_aliases or {})
                       .get("DIVISION", [])])


# ── The routes ────────────────────────────────────────────────────────────────

class _Upload:
    def __init__(self, text: str, filename: str = "mapping.csv"):
        self.filename = filename
        self._raw = text.encode("utf-8")

    async def read(self):
        return self._raw


class _Form(dict):
    pass


def _request(form=None, query=None):
    request = MagicMock()
    request.query_params = query or {}
    request.session = {"admin_id": "admin_user_1"}
    async def _read_form():
        return _Form(form or {})

    request.form = _read_form
    return request


class TestTheRoutes(RealWorkspace):

    def _call(self, coro_fn, form=None, query=None, authed=True):
        import admin.routes as routes

        with patch.object(routes, "_is_auth", return_value=authed), \
                patch.object(routes, "_resp", side_effect=lambda r, t, c=None: c):
            return asyncio.run(coro_fn(_request(form, query), self.account_id))

    def test_the_preview_route_shows_the_plan_and_writes_nothing(self):
        import admin.routes as routes

        ctx = self._call(routes.mapping_preview,
                         {"kind": "joins",
                          "file": _Upload(csv_text(JOIN_HEADER, join_row()))})
        self.assertEqual(ctx["plan"]["summary"]["created"], 1)
        self.assertEqual(ctx["plan"]["rows"][0]["action"], CREATE)
        self.assertEqual(self.stored(), [])

    def test_the_preview_carries_the_file_forward_so_apply_needs_no_re_upload(self):
        import admin.routes as routes

        text = csv_text(JOIN_HEADER, join_row())
        ctx = self._call(routes.mapping_preview,
                         {"kind": "joins", "file": _Upload(text)})
        self.assertEqual(ctx["text"], text)
        self.assertEqual(ctx["kind"], "joins")

    def test_the_apply_route_writes(self):
        import admin.routes as routes

        response = self._call(routes.mapping_apply,
                              {"kind": "joins",
                               "text": csv_text(JOIN_HEADER, join_row())})
        self.assertEqual(response.status_code, 303)
        self.assertEqual(len(self.stored()), 1)

    def test_the_apply_route_recomputes_rather_than_trusting_the_preview(self):
        # The graph can move between the two clicks. A plan carried over from
        # the preview would write a row that has since stopped being valid.
        import store

        import admin.routes as routes

        text = csv_text(JOIN_HEADER, join_row(to_table="SALES.DIM_WAREHOUSE",
                                              to_column="WAREHOUSE_ID",
                                              from_column="WAREHOUSE_ID"))
        ctx = self._call(routes.mapping_preview, {"kind": "joins",
                                                  "file": _Upload(text)})
        self.assertEqual(ctx["plan"]["summary"]["created"], 1)
        store.delete_entity(self.account_id, "Warehouse")
        self._call(routes.mapping_apply, {"kind": "joins", "text": text})
        self.assertEqual(self.stored(), [])

    def test_the_terms_route_writes_terms(self):
        import store

        import admin.routes as routes

        self._call(routes.mapping_apply,
                   {"kind": "terms",
                    "text": csv_text(TERM_HEADER,
                                     ["SALES.F_ORDER", "DIVISION", "profit centre"])})
        entry = (store.list_table_descriptions(self.account_id) or {}).get("SALES.F_ORDER")
        self.assertEqual(entry["column_synonym_map"]["DIVISION"], ["profit centre"])

    def test_a_file_with_nothing_in_it_is_an_error_not_a_traceback(self):
        import admin.routes as routes

        ctx = self._call(routes.mapping_preview, {"kind": "joins"})
        self.assertTrue(ctx["error"])
        self.assertIsNone(ctx["plan"])

    def test_an_unreadable_header_reports_rather_than_applying(self):
        import admin.routes as routes

        ctx = self._call(routes.mapping_preview,
                         {"kind": "joins",
                          "file": _Upload("a,b,c\n1,2,3\n")})
        self.assertTrue(ctx["plan"]["summary"]["fatal"])
        self.assertEqual(ctx["plan"]["rows"], [])

    def _download(self, route):
        import admin.routes as routes

        async def go():
            with patch.object(routes, "_is_auth", return_value=True):
                response = await route(_request(), self.account_id)
            chunks = [chunk if isinstance(chunk, (bytes, bytearray))
                      else str(chunk).encode("utf-8")
                      async for chunk in response.body_iterator]
            return response, b"".join(chunks).decode("utf-8")

        return asyncio.run(go())

    def test_the_download_gives_back_a_file_the_importer_accepts(self):
        import admin.routes as routes

        self.apply(join_row(label="placed by"))
        response, body = self._download(routes.mapping_download_joins)
        self.assertIn("attachment",
                      response.headers.get("content-disposition", ""))
        parsed = parse_joins(body)
        self.assertEqual(parsed.fatal, "")
        self.assertEqual(len(parsed.rows), 1)
        self.assertEqual(parsed.rows[0].from_table, "SALES.F_ORDER")
        self.assertEqual(parsed.rows[0].label, "placed by")

    def test_the_terms_download_round_trips(self):
        import admin.routes as routes

        self._call(routes.mapping_apply,
                   {"kind": "terms",
                    "text": csv_text(TERM_HEADER,
                                     ["SALES.F_ORDER", "DIVISION", "profit centre"])})
        _, body = self._download(routes.mapping_download_terms)
        self.assertEqual(parse_terms(body).rows[0].terms, ("profit centre",))

    def test_an_unauthenticated_upload_is_turned_away(self):
        import admin.routes as routes

        response = self._call(
            routes.mapping_apply,
            {"kind": "joins", "text": csv_text(JOIN_HEADER, join_row())},
            authed=False)
        self.assertEqual(response.status_code, 303)
        self.assertEqual(self.stored(), [])

    def test_the_import_asks_for_a_recompile_only_when_something_was_written(self):
        import admin.routes as routes

        with patch.object(routes, "_after_semantic_approval") as recompile, \
                patch.object(routes, "_is_auth", return_value=True):
            asyncio.run(routes.mapping_apply(
                _request({"kind": "joins",
                          "text": csv_text(JOIN_HEADER,
                                           join_row(to_table="SALES.F_SHIPMENT"))}),
                self.account_id))
        recompile.assert_not_called()

        with patch.object(routes, "_after_semantic_approval") as recompile, \
                patch.object(routes, "_is_auth", return_value=True):
            asyncio.run(routes.mapping_apply(
                _request({"kind": "joins",
                          "text": csv_text(JOIN_HEADER, join_row())}),
                self.account_id))
        recompile.assert_called_once()


class TestThePage(RealWorkspace):

    def _render(self, ctx):
        from jinja2 import Environment, FileSystemLoader

        env = Environment(loader=FileSystemLoader(str(ROOT / "admin" / "templates")))
        env.globals["ic"] = lambda name, size=16: ""
        request = type("R", (), {
            "url": type("U", (), {"path": f"/admin/clients/{self.account_id}/mapping"})(),
            "query_params": {}})()
        payload = {"client": {"account_id": self.account_id,
                              "client_name": "EMCO", "state": "READY"}}
        payload.update(ctx)
        return env.get_template("client_mapping.html").render(
            request=request, **payload)

    def _context(self, form):
        import admin.routes as routes

        with patch.object(routes, "_is_auth", return_value=True), \
                patch.object(routes, "_resp", side_effect=lambda r, t, c=None: c):
            return asyncio.run(routes.mapping_preview(_request(form), self.account_id))

    def test_the_preview_page_renders_the_plan_the_route_built(self):
        ctx = self._context({"kind": "joins",
                             "file": _Upload(csv_text(
                                 JOIN_HEADER,
                                 join_row(),
                                 join_row(to_table="SALES.F_SHIPMENT")))})
        html = self._render(ctx)
        self.assertIn("SALES.F_ORDER.CUSTOMER_ID", html)
        self.assertIn("fact-to-fact", html)
        self.assertIn("Apply 1 change", html)

    def test_a_plan_with_nothing_to_do_offers_no_apply_button(self):
        self.apply(join_row())
        html = self._render(self._context(
            {"kind": "joins", "file": _Upload(csv_text(JOIN_HEADER, join_row()))}))
        self.assertNotIn("Apply", html)
        self.assertIn("already present", html)

    def test_the_file_is_carried_in_the_form_so_apply_does_not_re_upload(self):
        text = csv_text(JOIN_HEADER, join_row())
        html = self._render(self._context({"kind": "joins", "file": _Upload(text)}))
        self.assertIn("/mapping/apply", html)
        self.assertIn("CUSTOMER_ID", html)

    def test_the_page_offers_both_downloads(self):
        html = self._render({
            "join_columns": list(JOIN_COLUMNS), "term_columns": list(TERM_COLUMNS),
            "max_rows": MAX_ROWS, "kind": "joins", "text": "", "plan": None,
            "error": "", "saved": ""})
        self.assertIn("/mapping/joins.csv", html)
        self.assertIn("/mapping/terms.csv", html)


if __name__ == "__main__":
    unittest.main()
