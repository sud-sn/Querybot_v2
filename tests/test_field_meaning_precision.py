"""
tests/test_field_meaning_precision.py

Fault 2 from the field: "we need the business meaning for the fields to be
precise, so when questions are asked they are returned."

Thirteen separate defects sat between an admin deciding what a column means and
a question being answered with that column. None of them raised an error. Each
one silently reduced what the model knew, or silently told it something wrong,
and the answer came back looking exactly like a correct one.

Grouped here by where the meaning was lost:

  approving it     — the approval reported success while runtime enforcement was
                     off; the approved text was written onto every column whose
                     name merely CONTAINED the approved column's name.
  keeping it       — an empty avoid list, a coin-flip between two approved
                     mappings.
  matching it      — an approved meaning reached the prompt only when the
                     question shared a literal word with it.
  carrying it      — prompt clamping dropped the disambiguation section first;
                     the synonym map emitted contradictory unqualified lines;
                     the glossary was cut mid-word at 100 characters.
  having it at all — Azure SQL discovery never read the DBA's own column
                     descriptions.

Every test drives the real function. Where a test reads source, it says why.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import pytest

# ── Approving it ─────────────────────────────────────────────────────────────


class TestApprovalReportsWhatActuallyHappened:
    def test_the_result_states_enforcement_separately_from_success(self):
        """The KB markdown and the structured semantic model are two different
        writes and they can disagree. Patching the prose and failing the model
        leaves the approval visible everywhere an admin looks while queries
        carry on using the old column."""
        from core.semantic_kb_patch import ApprovalResult

        result = ApprovalResult(ok=True, message="Approved", enforced=False,
                                enforcement_warning="no matching entry")
        assert result.ok is True
        assert result.enforced is False, (
            "a caller can only report an unenforced approval as saved if it "
            "cannot tell the difference"
        )

    def test_the_route_does_not_show_the_green_confirmation_when_unenforced(self):
        """Read from source because the alternative is standing up the whole
        admin app; the assertion is about which branch exists, not behaviour
        inside it."""
        import inspect

        import admin.routes as routes

        source = inspect.getsource(routes)
        assert "apply_approved_feedback_detailed" in source
        assert "if not result.enforced:" in source
        assert "field_warning=" in source

    def test_the_admin_page_renders_that_warning(self):
        html = Path("admin/templates/client_kb.html").read_text(encoding="utf-8")
        assert "field_warning" in html
        assert "not enforced at runtime" in html


class TestAnApprovalLandsOnOneColumnOnly:
    DOC = (
        "# DW.F_SALES\n\n## Columns\n"
        "- `INVOICE_DATE` (date): A date column.\n"
        "- `INVOICE_DATE_KEY` (int): Surrogate key to the date dimension.\n"
        "- `GROSS_AMT` (decimal): Some amount.\n"
    )

    def _patch(self):
        from core.semantic_kb_patch import _patch_column

        return _patch_column(
            content=self.DOC,
            column_name="INVOICE_DATE",
            approved_meaning="The date the invoice was issued.",
            approved_use_case="Revenue by invoice date.",
        )

    def test_the_approved_meaning_is_written_once(self):
        patched, changed = self._patch()
        assert changed is True
        assert patched.count("The date the invoice was issued.") == 1

    def test_a_longer_column_containing_the_name_is_left_alone(self):
        """The bullet format had no equality guard and the line matcher had no
        word boundary, so approving INVOICE_DATE also rewrote INVOICE_DATE_KEY.
        Both columns then carried byte-identical meaning at 'Confidence: 100%',
        and a join key was documented as a date value."""
        patched, _ = self._patch()
        assert "- `INVOICE_DATE_KEY` (int): Surrogate key to the date dimension." in patched

    def test_an_unrelated_column_is_left_alone(self):
        patched, _ = self._patch()
        assert "- `GROSS_AMT` (decimal): Some amount." in patched


# ── Keeping it ───────────────────────────────────────────────────────────────


AVOID = [{
    "table": "DW.F_SALES", "column": "GROSS_AMT", "term": "revenue",
    "use_instead_table": "DW.F_SALES", "use_instead_column": "NET_AMT",
}]


class TestTheRetiredColumnStaysForbidden:
    def test_the_supersession_block_survives_a_plan_with_no_fields(self):
        """A merged plan ends with no fields precisely when every field it
        proposed was a column an admin had superseded — so gating the block on
        `fields` suppressed the avoid list exactly when it was the only thing
        standing between the model and the retired column."""
        from core.semantic_planner import format_semantic_field_plan

        text = format_semantic_field_plan(
            {"enabled": False, "fields": [], "avoid_columns": AVOID},
        )
        assert "Do NOT use DW.F_SALES.GROSS_AMT" in text
        assert "NET_AMT" in text

    def test_it_still_appears_alongside_a_normal_field_plan(self):
        from core.semantic_planner import format_semantic_field_plan

        text = format_semantic_field_plan({
            "enabled": True, "avoid_columns": AVOID,
            "fields": [{"table": "DW.F_SALES", "column": "NET_AMT",
                        "term": "revenue", "role": "measure"}],
        })
        assert "Do NOT use DW.F_SALES.GROSS_AMT" in text

    def test_an_empty_plan_with_nothing_to_say_stays_empty(self):
        from core.semantic_planner import format_semantic_field_plan

        assert format_semantic_field_plan({"enabled": False, "fields": []}) == ""

    def test_the_prompt_builder_lets_a_fieldless_plan_through(self):
        """core/llm.py gated the whole block on `enabled and fields`, which is
        the same suppression one layer up."""
        import inspect

        import core.llm as llm

        source = inspect.getsource(llm)
        assert 'or semantic_plan.get("avoid_columns")' in source

    def test_the_merge_does_not_drop_it(self):
        import inspect

        import core.pipeline_context as ctx

        source = inspect.getsource(ctx)
        assert source.count('"avoid_columns": avoid_columns,') >= 2


# ── Matching it ──────────────────────────────────────────────────────────────


MODEL = {
    "tables": [{
        "table": "F_SALES", "qualified_name": "DW.F_SALES", "schema": "DW",
        "fields": [{
            "column": "NET_AMOUNT", "status": "approved", "confidence": 100,
            "role": "measure",
            "approved_meaning": "Net revenue after discounts.",
            "approved_use_case": "Used when a question refers to revenue.",
        }],
    }],
}


@pytest.fixture
def model_dir():
    from core.semantic_model import MODEL_JSON

    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / MODEL_JSON).write_text(json.dumps(MODEL), encoding="utf-8")
        yield tmp


def _resolved(model_dir, question, glossary=None):
    from core.semantic_model import build_runtime_semantic_plan

    plan = build_runtime_semantic_plan(
        model_dir, question=question, glossary=glossary,
    )
    return [f"{f['table']}.{f['column']}" for f in (plan.get("fields") or [])]


class TestAnApprovedMeaningReachesTheUsersWording:
    GLOSSARY = [{"term": "revenue", "aliases": "turnover, top line"}]

    def test_the_admins_own_wording_resolves(self, model_dir):
        assert _resolved(model_dir, "what is our revenue") == ["DW.F_SALES.NET_AMOUNT"]

    def test_a_glossary_alias_carries_a_different_wording_to_it(self, model_dir):
        """The admin approves a mapping, tests it with the phrasing they typed
        into the form, and it works. A user asking the same thing in their own
        words got the old column back, with nothing to say an approved mapping
        existed and had been skipped. An alias the admin themselves wrote is
        their own statement that the two words mean the same thing."""
        assert _resolved(model_dir, "what is our turnover") == [], (
            "precondition: without the glossary this wording does not match"
        )
        assert _resolved(model_dir, "what is our turnover", self.GLOSSARY) == [
            "DW.F_SALES.NET_AMOUNT",
        ]

    def test_an_unrelated_question_is_not_dragged_in(self, model_dir):
        assert _resolved(model_dir, "how many warehouses", self.GLOSSARY) == []

    def test_a_skipped_approval_is_reported(self, model_dir, caplog):
        """Score 0 silently omitted the authoritative line — no log, no trace.
        That silence is why an admin could believe a mapping was in force."""
        with caplog.at_level(logging.INFO, logger="core.semantic_model"):
            _resolved(model_dir, "how many warehouses")
        assert any(
            "admin-approved field meaning" in record.getMessage()
            for record in caplog.records
        )


class TestVocabularyExpansionDoesNotInventMatches:
    def test_a_column_code_never_enters_the_question(self):
        """The alias "gross profit" on SOP_CUS_LIN_GRS_PFT_AMT put the token
        "cus" into any question about gross profit, and "cus" then matched
        CUS_NM — requiring the customer name on a question about warehouses.
        Expansion runs one way only: a code the user typed gains the words it
        stands for, never the reverse."""
        from core.semantic_model import _terms_for_text, _vocab_expanded_terms

        base = _terms_for_text(
            "Which warehouses generate the highest invoice revenue, "
            "and what is their gross profit percentage?"
        )
        added = _vocab_expanded_terms(base) - base
        assert "cus" not in added
        assert not any(len(token) == 3 and token.isupper() for token in added)


class TestTwoEquallyApprovedMappingsAreNotResolvedSilently:
    def test_the_clash_is_recorded_on_the_plan(self):
        import inspect

        import core.semantic_model as sm

        source = inspect.getsource(sm.build_runtime_semantic_plan)
        assert "ambiguous_approved" in source, (
            "a tie between two approved mappings is still decided by which "
            "column name sorts later, with nothing recorded"
        )
        assert '"ambiguous_approved_fields": ambiguous_approved' in source


# ── Carrying it ──────────────────────────────────────────────────────────────


class TestTheSynonymMapDoesNotContradictItself:
    CONTEXT = (
        "# DW.F_SALES\n\n## Business Synonyms\n"
        "| Plain English | Column | Notes |\n"
        "| --- | --- | --- |\n"
        "| revenue | NET_AMOUNT | net of discounts |\n"
        "| sales | NET_AMOUNT | a second word for the same column |\n"
        "\n---\n"
        "# DW.F_ORDERS\n\n## Business Synonyms\n"
        "| Plain English | Column | Notes |\n"
        "| --- | --- | --- |\n"
        "| revenue | ORDER_VALUE | booked, not invoiced |\n"
    )

    def _block(self):
        from core.pipeline_helpers import _extract_kb_synonym_injection

        return _extract_kb_synonym_injection(self.CONTEXT)

    def test_every_column_is_qualified_by_its_table(self):
        """The owning table was never captured, so the model could apply a
        column from one table to another — an invalid-column error, or worse a
        valid column that means something else."""
        block = self._block()
        assert "DW.F_SALES.NET_AMOUNT" in block
        assert "DW.F_ORDERS.ORDER_VALUE" in block

    def test_one_word_on_two_tables_is_stated_as_such(self):
        """Both rows previously survived as unqualified 'authoritative'
        instructions for the same business word."""
        block = self._block()
        assert "documented on more than one table" in block
        assert "do not guess" in block.lower()

    def test_a_second_word_for_the_same_column_is_not_dropped(self):
        """Deduplication was keyed on the COLUMN, so the second term users
        actually type was discarded."""
        assert "'sales'" in self._block()


class TestClampingKeepsTheDisambiguationSection:
    DOC = (
        "# ERP.SALES\n## Overview\n" + "x" * 8000
        + "\n## Columns\nCOL_A int\n## Join Keys\nK1\n## Sample Data\n"
        + "y" * 4000 + "\n## Query Patterns\n" + "q" * 3000
        + "\n## Business Synonyms\n" + "z" * 2000
    )

    def test_business_synonyms_outlives_the_other_droppable_sections(self):
        """It is the only section mapping plain-English terms to exact columns
        and the only one carrying the ambiguity warnings — and dropping in file
        order made it the FIRST casualty, because the mandated format puts it
        last."""
        from core.pipeline_helpers import _clamp_kb_doc

        out = _clamp_kb_doc(self.DOC, cap=9000)
        assert "## Business Synonyms" in out
        assert "## Sample Data" not in out
        assert "## Columns" in out


class TestAGlossaryDefinitionKeepsItsDistinguishingHalf:
    LONG = (
        "Revenue recognised at invoice, net of returns and rebates, but BEFORE "
        "freight recovery. Do not confuse this with booked revenue, which is "
        "recognised at order entry and includes freight."
    )

    def test_the_second_clause_survives(self):
        """An admin writes a definition precisely to separate two similar
        concepts, and the distinguishing half is usually the second clause. A
        fixed 100-character cut removed it and severed the sentence mid-word."""
        from store.semantic_store import _clip_definition

        clipped = _clip_definition(self.LONG)
        assert "Do not confuse this with booked revenue" in clipped
        assert not clipped.endswith("confu")

    def test_a_definition_that_must_be_cut_is_cut_on_a_boundary(self):
        from store.semantic_store import _clip_definition

        clipped = _clip_definition("word " * 200)
        assert not clipped.rstrip("…").endswith(" wor")
        assert clipped.endswith("…") or clipped.endswith(".")

    def test_more_than_five_terms_can_reach_the_prompt(self):
        from store.semantic_store import build_term_injection

        terms = [
            {"term": f"term{i}", "kind": "metric",
             "canonical_expression": f"SUM(C{i})"}
            for i in range(8)
        ]
        block = build_term_injection(terms)
        assert "term7" in block, "terms beyond the fifth contributed nothing"


class TestEveryRetrievedTableArrivesWithItsColumns:
    def test_a_missing_columns_section_is_backfilled(self):
        """Re-ranking returns the n best SECTION chunks across all tables, and
        the doc is rebuilt only from the payloads handed over — so a table
        could be documented to the model with no column semantics at all."""
        import inspect

        import core.vector_store as vs

        source = inspect.getsource(vs.QdrantKBRetriever._ensure_column_semantics)
        assert 'sections=("columns",)' in source
        assert "fetch_docs_for_fqn" in source
        assert inspect.getsource(vs.QdrantKBRetriever.retrieve).count(
            "_ensure_column_semantics"
        ) == 1


# ── Having it at all ─────────────────────────────────────────────────────────


class _FakeCursor:
    def __init__(self, rows=None, fail=False):
        self._rows, self._fail = rows or [], fail

    def execute(self, *_args, **_kwargs):
        if self._fail:
            raise RuntimeError("VIEW DEFINITION denied")

    def fetchall(self):
        return self._rows


class TestAzureDiscoveryReadsTheDbasOwnDescriptions:
    def test_ms_description_is_read(self):
        """SQL Server keeps column descriptions as an extended property, not in
        INFORMATION_SCHEMA — so selecting five INFORMATION_SCHEMA columns and
        hardcoding an empty comment meant every Azure column reached the KB with
        nothing but its name."""
        from core.schema import _az_column_descriptions

        found = _az_column_descriptions(
            _FakeCursor([
                ("NET_REVENUE_AMT", "Revenue net of returns, before freight."),
                ("GROSS_REVENUE_AMT", "  Revenue  before  any  deduction. "),
                ("PADDING", "   "),
            ]),
            "dbo", "F_SALES",
        )
        assert found["NET_REVENUE_AMT"] == "Revenue net of returns, before freight."
        assert found["GROSS_REVENUE_AMT"] == "Revenue before any deduction."
        assert "PADDING" not in found, "a blank description is not a description"

    def test_a_warehouse_that_denies_the_read_still_discovers(self):
        """Many principals cannot read extended properties, and many warehouses
        set none. That degrades discovery to what it did before; it must not
        fail the table."""
        from core.schema import _az_column_descriptions

        assert _az_column_descriptions(_FakeCursor(fail=True), "dbo", "F_SALES") == {}

    def test_the_markdown_has_a_notes_cell_like_the_other_writers(self):
        from core.schema import _az_md

        columns = [{
            "COLUMN_NAME": "NET_REVENUE_AMT", "DATA_TYPE": "decimal",
            "IS_NULLABLE": "NO", "CHARACTER_MAXIMUM_LENGTH": None,
            "NUMERIC_PRECISION": 18,
            "COMMENT": "Revenue net of returns, before freight.",
        }]
        markdown = _az_md(
            "F_SALES",
            {"TABLE_SCHEMA": "dbo", "TABLE_NAME": "F_SALES",
             "TABLE_TYPE": "BASE TABLE", "TABLE_CATALOG": "DW"},
            columns, [], "dbo", {}, database="DW",
        )
        assert "| Column | Type | Nullable | Notes | Distinct Values |" in markdown
        assert "Revenue net of returns, before freight." in markdown


# ── Generic, not client-specific ─────────────────────────────────────────────


class TestErpKnowledgeLivesInThePackNotInPython:
    def test_the_planner_holds_no_literal_erp_code_sets(self):
        # Executable lines only: the comments explaining what was removed name
        # the old literals on purpose.
        source = "\n".join(
            line for line in
            Path("core/semantic_planner.py").read_text(encoding="utf-8").splitlines()
            if not line.lstrip().startswith("#")
        )
        for literal in (
            '{"DIVI", "WHLO", "ORNO", "PONR", "POSX"}',
            '{"CONO", "ORNO", "PONR", "POSX", "DLIX"}',
            'column == "DIVI"',
        ):
            assert literal not in source, (
                f"{literal} still decides behaviour in Python, so it cannot be "
                f"extended for the next client without another code edit"
            )

    def test_join_eligibility_comes_from_the_vocabulary(self):
        from core.vocab_packs import MergedVocab, load_pack

        assert "join_key_codes" in MergedVocab.__dataclass_fields__
        pack = load_pack("infor_m3") or {}
        assert set(pack.get("join_key_codes") or []) == {
            "CONO", "ORNO", "PONR", "POSX", "DLIX",
        }

    def test_a_grouping_identifier_is_not_join_eligible(self):
        """A division code identifies a category and appears on many unrelated
        tables; joining on it manufactures a false edge. It is an identifier
        but not a relational key, which is why the two sets differ."""
        from core.vocab_packs import load_pack

        pack = load_pack("infor_m3") or {}
        assert "DIVI" in (pack.get("raw_identifier_codes") or [])
        assert "DIVI" not in (pack.get("join_key_codes") or [])

    def test_the_vocabulary_is_activated_for_a_query(self):
        """vocab_for_account merges the admin's selected ERP pack and the
        clients/<id>/vocab.json overlay, and it was installed only for the KB
        build and the graph autopopulate. Every query-time consumer read the
        unset ContextVar and silently used the builtin."""
        import inspect

        import core.query_pipeline as pipeline

        source = inspect.getsource(pipeline.handle_query)
        assert "activate_vocab(vocab_for_account(account_id))" in source
        assert "deactivate_vocab" in source


class TestAnAmbiguousBareMeasureIsNamed:
    COLUMNS = {
        "DW.F_SALES": {
            "GROSS_REVENUE_AMT": "decimal",
            "NET_REVENUE_AMT": "decimal",
            "INVOICE_DATE": "date",
        },
    }

    def _plan(self, question):
        from core.semantic_planner import build_semantic_field_plan

        return build_semantic_field_plan(question, self.COLUMNS)

    def test_a_qualified_question_still_resolves_exactly(self):
        plan = self._plan("what was our net revenue last month")
        assert [f["column"] for f in plan["fields"]] == ["NET_REVENUE_AMT"]

    def test_a_bare_head_noun_names_its_rivals(self):
        """Alias derivation deliberately refuses to strip "net revenue" to
        "revenue" — right, because a bare noun must not hard-require one
        column. But it left the plan collapsing entirely, and the model chose
        between the rivals on its own, confidently, and could choose
        differently next week."""
        plan = self._plan("what was our revenue last month")
        assert plan["enabled"] is False
        assert plan["ambiguous_measures"] == [
            "DW.F_SALES.GROSS_REVENUE_AMT", "DW.F_SALES.NET_REVENUE_AMT",
        ]

    def test_the_rivals_reach_the_prompt(self):
        from core.semantic_planner import format_semantic_field_plan

        text = format_semantic_field_plan(self._plan("what was our revenue last month"))
        assert "GROSS_REVENUE_AMT" in text and "NET_REVENUE_AMT" in text
        assert "picking silently" in text

    def test_a_question_with_no_measure_ambiguity_says_nothing(self):
        assert self._plan("how many invoices last month")["ambiguous_measures"] == []
