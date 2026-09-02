"""
tests/test_per_table_descriptions.py

A table's own business description, for the tables the admin selected.

Before this, core/knowledge._build_table_business_desc had exactly two levels:
one description for the whole client, plus optional per-SCHEMA blocks. Every
table in a schema therefore received byte-identical business context in its KB
prompt, so the sentence that distinguishes one table from its neighbours --
"this is the month-end stock snapshot, one row per item per warehouse, never
sum it across months" -- had nowhere to live.

The feature has two halves and they go to different places, which is the thing
these tests are mostly here to hold:

  description  -> the KB document for that table, and nothing else.
  synonyms     -> source resolution, which never reads prose.

That split is not a detail. Catalogue case B4 ("show me the inventory value by
warehouse") failed for months because no term connected the word "inventory" to
ITM_BAL_PRD_FCT, and it was fixed by adding STRUCTURED vocabulary, not prose. An
admin who writes a perfect paragraph and expects the right table to be chosen
would be no better off than before.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_tmpdir = tempfile.mkdtemp(prefix="querybot_tabledesc_")
os.environ.setdefault("DB_PATH", str(Path(_tmpdir) / "test_querybot.db"))
os.environ.setdefault("QUERYBOT_KEY_FILE", str(Path(_tmpdir) / "test_key"))

from core.knowledge import _build_table_business_desc  # noqa: E402
from core.source_resolution import _table_aliases  # noqa: E402
from store.db import init_db  # noqa: E402
from store.table_description_store import (  # noqa: E402
    describe_selected_tables,
    description_coverage,
    get_table_description,
    list_table_descriptions,
    save_table_description,
    split_synonyms,
)

SNAPSHOT = "EMDW_DMART.ITM_BAL_PRD_FCT"
INVOICE = "EMDW_DMART.CUS_ORD_IVC_FCT"
WAREHOUSE = "EMDW_DMART.WHS_DMS"
PROSE = ("Month-end stock snapshot: one row per item per warehouse per period. "
         "Never sum across months.")


class _Base(unittest.TestCase):
    account = "acct_tabledesc"

    @classmethod
    def setUpClass(cls):
        init_db()

    def setUp(self):
        from store.db import get_db
        with get_db() as conn:
            conn.execute("DELETE FROM table_description WHERE account_id=?", (self.account,))


class TestStoringOneTablesDescription(_Base):

    def test_it_round_trips(self):
        save_table_description(self.account, SNAPSHOT,
                               description=PROSE, synonyms="inventory, stock balance")
        entry = get_table_description(self.account, SNAPSHOT)
        self.assertEqual(entry["description"], PROSE)
        self.assertEqual(entry["synonym_list"], ["inventory", "stock balance"])

    def test_saving_again_edits_rather_than_duplicates(self):
        """The admin must be able to come back and correct what they wrote."""
        save_table_description(self.account, SNAPSHOT, description="first")
        save_table_description(self.account, SNAPSHOT, description="second")
        self.assertEqual(get_table_description(self.account, SNAPSHOT)["description"], "second")
        self.assertEqual(len(list_table_descriptions(self.account)), 1)

    def test_clearing_both_fields_removes_the_record(self):
        """An emptied form must not leave a row behind claiming to be described,
        or the coverage count reports work that was undone."""
        save_table_description(self.account, SNAPSHOT, description=PROSE, synonyms="inventory")
        save_table_description(self.account, SNAPSHOT, description="  ", synonyms="")
        self.assertIsNone(get_table_description(self.account, SNAPSHOT))

    def test_the_table_name_matches_however_it_was_written(self):
        save_table_description(self.account, SNAPSHOT, description=PROSE)
        for variant in (SNAPSHOT.lower(), f"[{SNAPSHOT}]", "  " + SNAPSHOT + "  "):
            with self.subTest(variant=variant):
                self.assertIsNotNone(get_table_description(self.account, variant))

    def test_one_tenants_description_is_invisible_to_another(self):
        save_table_description(self.account, SNAPSHOT, description=PROSE)
        self.assertIsNone(get_table_description("someone_else", SNAPSHOT))

    def test_synonyms_are_split_and_de_duplicated(self):
        self.assertEqual(
            split_synonyms("inventory, Inventory\nstock balance; inventory"),
            ["inventory", "stock balance"],
        )


class TestOnlySelectedTablesAreListed(_Base):

    def test_the_list_follows_the_selection_not_the_schema(self):
        """A schema can hold hundreds of tables. The admin is accountable for
        the ones they turned on."""
        save_table_description(self.account, "EMDW_DMART.SOMETHING_ELSE", description="x")
        rows = describe_selected_tables(self.account, [SNAPSHOT, INVOICE])
        self.assertEqual([r["table_name"] for r in rows], [SNAPSHOT, INVOICE])

    def test_a_table_with_no_kb_yet_is_flagged_new(self):
        """The case the admin most needs surfaced: selected after the last
        build, so it will reach users with no business context."""
        rows = describe_selected_tables(
            self.account, [SNAPSHOT, INVOICE, WAREHOUSE],
            built_tables={SNAPSHOT, INVOICE},
        )
        self.assertEqual({r["table_name"]: r["status"] for r in rows}[WAREHOUSE], "new")

    def test_describing_a_new_table_does_not_clear_the_new_flag(self):
        """It still has no Knowledge Base. Saying otherwise would hide the
        rebuild the admin still owes."""
        save_table_description(self.account, WAREHOUSE, description="Warehouse master.")
        rows = describe_selected_tables(
            self.account, [WAREHOUSE], built_tables={SNAPSHOT},
        )
        self.assertEqual(rows[0]["status"], "new")
        self.assertTrue(rows[0]["described"])

    def test_status_reflects_whether_prose_was_written(self):
        save_table_description(self.account, SNAPSHOT, description=PROSE)
        rows = describe_selected_tables(
            self.account, [SNAPSHOT, INVOICE], built_tables={SNAPSHOT, INVOICE},
        )
        got = {r["table_name"]: r["status"] for r in rows}
        self.assertEqual(got[SNAPSHOT], "described")
        self.assertEqual(got[INVOICE], "undescribed")

    def test_synonyms_alone_do_not_count_as_described(self):
        """Terms and prose do different jobs; having one is not having the other."""
        save_table_description(self.account, SNAPSHOT, synonyms="inventory")
        rows = describe_selected_tables(self.account, [SNAPSHOT], built_tables={SNAPSHOT})
        self.assertFalse(rows[0]["described"])

    def test_coverage_counts_what_the_admin_still_owes(self):
        save_table_description(self.account, SNAPSHOT, description=PROSE)
        rows = describe_selected_tables(
            self.account, [SNAPSHOT, INVOICE, WAREHOUSE], built_tables={SNAPSHOT, INVOICE},
        )
        self.assertEqual(
            description_coverage(rows),
            {"total": 3, "described": 1, "undescribed": 1, "new": 1},
        )

    def test_no_built_set_means_nothing_is_called_new(self):
        """Before the first build every table is un-built; calling them all
        "new" would be noise, not a signal."""
        rows = describe_selected_tables(self.account, [SNAPSHOT, INVOICE])
        self.assertEqual({r["status"] for r in rows}, {"undescribed"})


class TestTheKbBuilderCanActuallyFindIt(_Base):
    """The lookup, executed — not the formatter with the text handed to it.

    The first version of this feature passed every test and was dead: the setup
    page stores what the table picker listed (SCHEMA.TABLE), while
    core/knowledge.py:688 looks up by `md_file.stem`, which is the BARE table
    name. The lookup returned None, the except never fired, nothing was logged,
    and the tests were green because they called the formatter directly with
    the description already in hand.

    Every name below is one a real caller uses.
    """

    def test_every_way_a_caller_names_the_table_finds_it(self):
        save_table_description(self.account, SNAPSHOT, description=PROSE, synonyms="inventory")
        for caller, name in (
            ("setup page (as stored)", SNAPSHOT),
            ("core/knowledge.py:688 md_file.stem", "ITM_BAL_PRD_FCT"),
            ("SQL builder quoting", "[EMDW_DMART].[ITM_BAL_PRD_FCT]"),
            ("database-qualified", "CHATBOT_DB." + SNAPSHOT),
            ("lower case", SNAPSHOT.lower()),
        ):
            with self.subTest(caller=caller):
                entry = get_table_description(self.account, name)
                self.assertIsNotNone(entry, f"{caller} could not find it")
                self.assertEqual(entry["description"], PROSE)

    def test_an_ambiguous_bare_name_resolves_to_nothing(self):
        """Two schemas each with an ORDERS table is ordinary. Attaching one
        schema's description to the other's KB document is worse than none."""
        save_table_description(self.account, "SALES.ORDERS", description="Sales orders.")
        save_table_description(self.account, "OPS.ORDERS", description="Work orders.")
        self.assertIsNone(get_table_description(self.account, "ORDERS"))
        # Qualified lookups still work.
        self.assertEqual(
            get_table_description(self.account, "SALES.ORDERS")["description"], "Sales orders.")

    def test_the_terms_reach_the_semantic_model_under_the_bare_name(self):
        """The resolution half has the same two-callers problem: the model is
        built from the discovered schema, the terms were stored from the picker."""
        from store.table_description_store import _match_key, list_table_descriptions
        save_table_description(self.account, SNAPSHOT, synonyms="inventory, stock balance")
        stored = list_table_descriptions(self.account)
        key = _match_key(stored.keys(), "ITM_BAL_PRD_FCT")
        self.assertIsNotNone(key, "the model builder could not match the bare name")
        self.assertEqual(stored[key]["synonym_list"], ["inventory", "stock balance"])


class TestItReachesTheKnowledgeBase(_Base):

    def test_the_table_description_is_in_the_kb_context(self):
        text = _build_table_business_desc("EMCO distributes plumbing supplies.",
                                          "EMDW_DMART", PROSE, "inventory, stock balance")
        self.assertIn("EMCO distributes plumbing supplies.", text)
        self.assertIn("Never sum across months.", text)
        self.assertIn("inventory, stock balance", text)

    def test_the_client_description_still_works_alone(self):
        """Every existing tenant has no per-table descriptions, and their KB
        context must be byte-identical to what it was."""
        self.assertEqual(_build_table_business_desc("Just the client.", "EMDW_DMART"),
                         "Just the client.")

    def test_two_tables_no_longer_get_identical_context(self):
        """The defect, stated directly."""
        first = _build_table_business_desc("Shared.", "EMDW_DMART", "Stock snapshot.")
        second = _build_table_business_desc("Shared.", "EMDW_DMART", "Invoice lines.")
        self.assertNotEqual(first, second)

    def test_the_per_schema_block_is_preserved(self):
        text = _build_table_business_desc(
            "Overall.\n[EMDW_DMART] The mart.", "EMDW_DMART", "Stock snapshot.")
        self.assertIn("Overall.", text)
        self.assertIn("The mart.", text)
        self.assertIn("Stock snapshot.", text)


class TestTermsReachSourceResolution(_Base):
    """The half that decides WHICH table a question lands on.

    core/source_resolution._table_aliases matches on names, labels and synonyms
    and reads no prose, so this is the only route by which anything an admin
    types can change table selection.
    """

    TABLE = {"fqn": SNAPSHOT, "table": "ITM_BAL_PRD_FCT",
             "entity": "Itm Bal Prd", "fields": []}

    def test_without_terms_the_business_word_does_not_reach_the_table(self):
        self.assertNotIn("inventory", _table_aliases(dict(self.TABLE)))

    def test_an_admin_term_makes_the_table_reachable(self):
        aliases = _table_aliases(dict(self.TABLE, business_synonyms=["inventory", "stock balance"]))
        self.assertIn("inventory", aliases)
        self.assertIn("stock balance", aliases)

    def test_prose_alone_changes_nothing_about_resolution(self):
        """Stated as a test so nobody later "simplifies" the two fields into
        one and quietly removes the only half that affects table choice."""
        self.assertNotIn("inventory", _table_aliases(dict(self.TABLE, description=PROSE)))

    def test_a_table_with_no_terms_is_unaffected(self):
        self.assertEqual(_table_aliases(dict(self.TABLE)),
                         _table_aliases(dict(self.TABLE, business_synonyms=[])))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestWhichTablesCountAsBuilt(_Base):
    """"New" must mean "selected since the last KB build", not "we have no
    audit row for it".

    Live on EMCO, 2026-09-02: all fourteen tables displayed "new — not yet
    built" under a banner reading "14 tables added since the last Knowledge
    Base build", on a client whose KB was fully built. The lookup read
    kb_data_egress_log for operation='kb_build' -- an audit trail of what was
    sent to the model, and therefore empty for any KB built before that logging
    existed. The KB documents themselves are the authority.
    """

    def _built(self, kb_dir):
        # Patched on admin.routes.store, not on "store": another test file in
        # this suite purges and re-imports the store package, so admin.routes
        # can hold a different module object than `import store` resolves to,
        # and patching the latter silently misses.
        import admin.routes as routes
        from unittest.mock import patch
        client = {"state_data": json.dumps({"kb_dir": str(kb_dir)})}
        with patch.object(routes.store, "get_client", return_value=client):
            return routes._kb_built_tables(self.account)

    def test_a_written_kb_document_counts_as_built(self):
        kb = Path(tempfile.mkdtemp(prefix="kb_"))
        (kb / "ITM_BAL_PRD_FCT.md").write_text("# doc", encoding="utf-8")
        (kb / "CUS_DMS.md").write_text("# doc", encoding="utf-8")
        self.assertEqual(self._built(kb), {"ITM_BAL_PRD_FCT", "CUS_DMS"})

    def test_internal_files_are_not_tables(self):
        """_schema.json and friends share the directory; only table docs count."""
        kb = Path(tempfile.mkdtemp(prefix="kb_"))
        (kb / "CUS_DMS.md").write_text("# doc", encoding="utf-8")
        (kb / "_index.md").write_text("# not a table", encoding="utf-8")
        self.assertEqual(self._built(kb), {"CUS_DMS"})

    def test_an_empty_kb_directory_means_nothing_is_built(self):
        self.assertEqual(self._built(Path(tempfile.mkdtemp(prefix="kb_"))), set())

    def test_a_missing_directory_does_not_raise(self):
        self.assertEqual(self._built(Path("/no/such/kb/dir")), set())

    def test_a_built_table_is_not_reported_as_new(self):
        """The user-visible consequence, asserted end to end."""
        kb = Path(tempfile.mkdtemp(prefix="kb_"))
        (kb / "ITM_BAL_PRD_FCT.md").write_text("# doc", encoding="utf-8")
        rows = describe_selected_tables(
            self.account, [SNAPSHOT, WAREHOUSE], built_tables=self._built(kb),
        )
        status = {r["table_name"]: r["status"] for r in rows}
        self.assertNotEqual(status[SNAPSHOT], "new")
        self.assertEqual(status[WAREHOUSE], "new")
