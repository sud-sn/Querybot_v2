"""
A column means one thing, and every page that shows it says where that came from.

What a column is called, what it means and the words readers use for it were
kept in seven stores: the field overrides, approved Semantic Layer
suggestions, the semantic model's approvals, the join graph's confirmed
column properties, confirmed business meanings, the table setup's column
terms and the database's own comment -- beside the knowledge base's generated
prose. Each reader picked its own winner:

  * The Semantic Layer read the prose, approved suggestions and overrides,
    and never the join graph, the column terms, confirmed business meanings
    or the semantic model's approvals.
  * Confirming a column in the join graph wrote an 'approved' suggestion whose
    meaning was the column's synonyms ("client, account holder" as what
    CUSTOMER_NAME means), and another copy on every confirm.
  * It also put the column in the glossary as a metric whatever its role, so
    two dimensions named in one question were offered as two metrics to
    choose between.
  * The count-target label put the machine's reading of a name ahead of the
    meaning an admin approved.
  * A column nothing described was shown with a sentence made from its name
    and no "needs context" flag.

core.meaning resolves every store by one rule: admin > database >
vocabulary > rule > ai. These build a tenant in a scratch store, write
through the store's own APIs and read through the portal page, the Semantic
Layer builder, the glossary and the count-target resolver.
"""

from __future__ import annotations

import json
import os
import re

import pytest


def _table(key, *columns):
    return {"columns": [{"name": n, "type": t, "nullable": False, "comment": c} for n, t, c in columns],
            "pk_columns": [key], "row_count": 1000, "comment": "", "schema": "MART", "database": "WH"}


def _schema(net_amt_comment="Invoiced amount after rebates"):
    return {
        "WH.MART.SALES_FACT": _table(
            "SALES_KEY", ("SALES_KEY", "bigint", ""), ("CUSTOMER_SK", "int", ""), ("DOC_NO", "varchar(20)", ""),
            ("NET_AMT", "decimal(18,2)", net_amt_comment), ("FREIGHT_AMT", "decimal(18,2)", "")),
        "WH.MART.CUSTOMER_DIM": _table(
            "CUSTOMER_SK", ("CUSTOMER_SK", "int", ""), ("CUSTOMER_NAME", "varchar(80)", ""),
            ("REGION_CD", "char(2)", ""), ("SEGMENT_CD", "char(2)", "")),
    }


KB_FILES = {
    "SALES_FACT_kb.md": (
        "# WH.MART.SALES_FACT\n\n## Overview\nOne row per invoice line.\n\n## Columns\n"
        "- `SALES_KEY` (bigint): Surrogate key of the invoice line.\n"
        "- `NET_AMT` (decimal): Net amount as the knowledge base wrote it.\n"
        "- `DOC_NO` (varchar): Document number.\n\n"
        "## Business Synonyms\n| Plain English | Column | Notes |\n|---|---|---|\n"
        "| revenue, turnover | NET_AMT | |\n"),
    "CUSTOMER_DIM_kb.md": (
        "# WH.MART.CUSTOMER_DIM\n\n## Columns\n"
        "- `CUSTOMER_SK` (int): Customer surrogate key.\n"
        "- `CUSTOMER_NAME` (varchar): Name of the customer.\n"
        "- `REGION_CD` (char): Region code [NEEDS CONTEXT]\n"),
}


def _make_tenant(tmp_path, monkeypatch, schema):
    import store
    from core import field_overrides
    from core.semantic_model import write_semantic_model

    # Field overrides are a file under clients/ in the working directory.
    monkeypatch.setattr(field_overrides, "override_path",
                        lambda account_id: tmp_path / "clients" / account_id / "field_overrides.json")
    store.init_db()
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "_schema.json").write_text(json.dumps(schema), encoding="utf-8")
    for name, text in KB_FILES.items():
        (kb / name).write_text(text, encoding="utf-8")
    store.update_client_state(account_id, "READY", {"schema_dir": str(kb), "kb_dir": str(kb)})
    write_semantic_model(schema_dir=str(kb), kb_dir=str(kb), account_id=account_id)
    store.save_entity(account_id=account_id, entity_name="Customer", table_name="CUSTOMER_DIM",
                      schema_name="MART", pos_x=0.0, pos_y=0.0)
    store.save_entity(account_id=account_id, entity_name="Sales", table_name="SALES_FACT",
                      schema_name="MART", pos_x=300.0, pos_y=0.0)
    return account_id, str(kb)


@pytest.fixture
def tenant(tmp_path, monkeypatch):
    return _make_tenant(tmp_path, monkeypatch, _schema())


def _page_field(tenant, column):
    """The field as the Semantic Layer builds it, from what the routes pass."""
    import store
    from core.field_overrides import load_field_overrides
    from core.semantic_layer import build_semantic_layer_tables

    account_id, kb = tenant
    approved, pending = store.semantic_feedback_maps(account_id)
    tables = build_semantic_layer_tables(kb_dir=kb, schema_dir=kb, approved_feedback=approved,
                                         pending_feedback=pending, field_overrides=load_field_overrides(account_id),
                                         account_id=account_id)
    return next(f for t in tables for f in t["fields"] if f["column"].upper() == column)


def _confirm(account_id, entity, column, *, role="dimension", display_name="", synonyms=""):
    import store

    store.save_entity_property(account_id=account_id, entity_name=entity, column_name=column, role=role,
                               display_name=display_name, synonyms=synonyms, status="suggested")
    store.confirm_entity_property(account_id, entity, column)


def _portal_row(account_id, column):
    """The field's row on the reader's Semantic Layer page, over HTTP."""
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    import store
    import portal.routes as pr

    user_id, _ = store.create_user(account_id, "Ada", f"{os.urandom(4).hex()}@x.com",
                                   password="a-password-they-chose", role="admin")
    app = FastAPI()
    app.include_router(pr.router)
    client = TestClient(app)
    client.cookies.set(pr._COOKIE, pr._sign_session_value(user_id))
    page = client.get("/portal/kb?schema=MART").text
    rows = re.findall(r'<tr class="semantic-field-row"[^>]*data-column-name="' + column + r'">([\s\S]*?)</tr>', page)
    assert rows, f"no row for {column}"
    return rows[0]


# ── The join graph reaches the Semantic Layer ────────────────────────────────

class TestAColumnConfirmedInTheJoinGraph:

    def test_shows_its_business_name_and_terms_on_the_reader_s_page(self, tenant):
        account_id, _ = tenant
        _confirm(account_id, "Customer", "CUSTOMER_NAME", display_name="Customer name",
                 synonyms="client, account holder")
        row = _portal_row(account_id, "CUSTOMER_NAME")
        assert '<div class="field-business-name">Customer name</div>' in row
        chips = re.findall(r'<span class="term-chip">([^<]*)</span>', row)
        assert chips[:2] == ["client", "account holder"], chips

    def test_shows_them_on_the_admin_s_field_editor_too(self, tenant):
        from unittest.mock import patch

        from fastapi import FastAPI
        from starlette.testclient import TestClient

        import admin.routes as routes

        account_id, _ = tenant
        _confirm(account_id, "Customer", "CUSTOMER_NAME", display_name="Customer name",
                 synonyms="client, account holder")
        app = FastAPI()
        app.include_router(routes.router)
        with patch.object(routes, "_is_auth", return_value=True):
            page = TestClient(app).get(f"/admin/clients/{account_id}/kb?view=fields").text
        card = re.search(r'id="field-CUSTOMER_DIM-CUSTOMER_NAME"[\s\S]*?</article>', page).group(0)
        assert '<div class="kb-field-business-name">Customer name</div>' in card
        assert "<strong>Synonyms:</strong> client, account holder</div>" in card

    def test_keeps_what_the_column_means_and_does_not_call_it_approved(self, tenant):
        # The admin named the column and gave it terms; nobody decided what
        # it means, so the knowledge base's description stands, unapproved.
        import store

        account_id, _ = tenant
        _confirm(account_id, "Customer", "CUSTOMER_NAME", display_name="Customer name",
                 synonyms="client, account holder")
        field = _page_field(tenant, "CUSTOMER_NAME")
        assert (field["meaning"], field["approved"]) == ("Name of the customer", False)
        assert store.list_semantic_field_feedback(account_id, limit=50) == []

    def test_confirming_again_writes_nothing_twice(self, tenant):
        import store

        account_id, _ = tenant
        _confirm(account_id, "Customer", "CUSTOMER_NAME", display_name="Customer name")
        store.confirm_entity_property(account_id, "Customer", "CUSTOMER_NAME")
        assert store.list_semantic_field_feedback(account_id, limit=50) == []
        assert [t["term"] for t in store.list_terms(account_id)] == ["customer name"]

    def test_a_suggestion_nobody_confirmed_says_nothing(self, tenant):
        import store

        account_id, _ = tenant
        store.save_entity_property(account_id=account_id, entity_name="Customer", column_name="CUSTOMER_NAME",
                                   display_name="Customer name", synonyms="client", status="suggested")
        field = _page_field(tenant, "CUSTOMER_NAME")
        assert (field["label"], field["synonyms"]) == ("", [])


# ── The glossary term a confirmation writes ──────────────────────────────────

class TestTheGlossaryTermIsOfTheColumnsKind:

    @pytest.mark.parametrize("role,kind", [
        ("metric", "metric"), ("dimension", "dimension"), ("date", "dimension"),
        ("filter", "filter"), ("identifier", "dimension"),
    ])
    def test_each_role_names_its_kind(self, tenant, role, kind):
        import store

        account_id, _ = tenant
        _confirm(account_id, "Customer", "SEGMENT_CD", role=role, display_name="Customer segment")
        assert [(t["term"], t["kind"]) for t in store.list_terms(account_id)] == [("customer segment", kind)]

    def test_an_ignored_column_is_not_named_in_the_glossary(self, tenant):
        import store

        account_id, _ = tenant
        _confirm(account_id, "Customer", "SEGMENT_CD", role="ignore", display_name="Customer segment")
        assert store.list_terms(account_id) == []

    def test_the_term_answers_to_the_column_s_synonyms(self, tenant):
        import store

        account_id, _ = tenant
        _confirm(account_id, "Customer", "CUSTOMER_NAME", display_name="Customer name",
                 synonyms="client, account holder")
        found = store.match_terms_in_question(account_id, "net sales by account holder")
        assert [t["term"] for t in found] == ["customer name"]

    def test_two_confirmed_dimensions_are_not_two_metrics_to_choose_between(self, tenant):
        from core.clarification import has_ambiguity_signal

        account_id, _ = tenant
        _confirm(account_id, "Customer", "CUSTOMER_NAME", display_name="Customer name")
        _confirm(account_id, "Customer", "REGION_CD", display_name="Customer region")
        assert has_ambiguity_signal(account_id, "net sales by customer name and customer region") is False

    def test_saving_a_property_on_the_graph_writes_the_same_term(self, tenant):
        import asyncio
        from unittest.mock import MagicMock, patch

        import store
        import admin.routes as routes

        account_id, _ = tenant
        request = MagicMock()

        async def _json():
            return {"entity_name": "Customer", "column_name": "REGION_CD", "role": "dimension",
                    "display_name": "Sales region", "synonyms": "territory"}

        request.json = _json
        with patch.object(routes, "_is_auth", return_value=True):
            asyncio.run(routes.graph_api_prop_save(request, account_id))
        terms = store.list_terms(account_id)
        assert [(t["term"], t["kind"], t["canonical_expression"], t["tables_involved"], t["aliases"], t["source"])
                for t in terms] == [("sales region", "dimension", "REGION_CD", "MART.CUSTOMER_DIM", "territory",
                                     "manual")]


# ── One rule decides ─────────────────────────────────────────────────────────

def _override(account_id, table, column, meaning):
    from core.field_overrides import save_field_override

    save_field_override(account_id=account_id, table_fqn=f"WH.MART.{table}", schema_name="MART",
                        table_name=table, file_stem=table, column_name=column, meaning=meaning)


def _approved_suggestion(account_id, meaning):
    import store

    user_id, _ = store.create_user(account_id, "Ada", f"{os.urandom(4).hex()}@x.com",
                                   password="a-password-they-chose", role="analyst")
    feedback_id = store.save_semantic_field_feedback(
        account_id=account_id, portal_user_id=user_id, table_fqn="WH.MART.SALES_FACT", schema_name="MART",
        table_name="SALES_FACT", column_name="NET_AMT", suggested_meaning=meaning)
    assert store.review_semantic_field_feedback(feedback_id, account_id, status="approved")


def _model_approval(kb, meaning):
    from core.semantic_model import patch_field_approval

    assert patch_field_approval(kb_dir=kb, table_fqn="WH.MART.SALES_FACT", table_name="SALES_FACT",
                                schema_name="MART", column_name="NET_AMT", approved_meaning=meaning)


RUNGS = {
    "override": ("Net sales in CAD, as finance closes the month", "admin_override"),
    "suggestion": ("Net sales as the sales team reports them", "approved_feedback"),
    "model": ("Net sales after returns", "semantic_model"),
    "comment": ("Invoiced amount after rebates", "database_comment"),
    "kb": ("Net amount as the knowledge base wrote it", "generated"),
}


class TestOneRuleDecidesWhatAColumnMeans:

    @pytest.mark.parametrize("present,winner", [
        (("override", "suggestion", "model", "comment", "kb"), "override"),
        (("suggestion", "model", "comment", "kb"), "suggestion"),
        (("model", "comment", "kb"), "model"),
        (("comment", "kb"), "comment"),
        (("kb",), "kb"),
    ])
    def test_the_highest_authority_present_wins(self, tmp_path, monkeypatch, present, winner):
        tenant = _make_tenant(tmp_path, monkeypatch,
                              _schema(RUNGS["comment"][0] if "comment" in present else ""))
        account_id, kb = tenant
        if "override" in present:
            _override(account_id, "SALES_FACT", "NET_AMT", RUNGS["override"][0])
        if "suggestion" in present:
            _approved_suggestion(account_id, RUNGS["suggestion"][0])
        if "model" in present:
            _model_approval(kb, RUNGS["model"][0])
        field = _page_field(tenant, "NET_AMT")
        assert (field["meaning"], field["source"]) == RUNGS[winner]

    def test_approved_means_an_admin_decided_the_meaning(self, tmp_path, monkeypatch):
        tenant = _make_tenant(tmp_path, monkeypatch, _schema())
        field = _page_field(tenant, "NET_AMT")
        assert (field["approved"], field["meaning_evidence"], field["confidence"]) == \
            (False, "database comment", 90)
        _model_approval(tenant[1], RUNGS["model"][0])
        field = _page_field(tenant, "NET_AMT")
        assert (field["approved"], field["meaning_evidence"], field["confidence"]) == \
            (True, "approved in the semantic model", 100)

    def test_the_table_setup_s_column_terms_come_before_the_generated_ones(self, tenant):
        import store

        account_id, _ = tenant
        store.save_table_description(account_id, "MART.SALES_FACT",
                                     column_synonyms="NET_AMT = net sales, Revenue")
        # One term once, in the admin's spelling.
        assert _page_field(tenant, "NET_AMT")["synonyms"] == ["net sales", "Revenue", "turnover"]

    def test_a_synonym_list_the_admin_saved_is_the_whole_list(self, tenant):
        # The editor shows the generated terms; "turnover" was taken out.
        from core.field_overrides import save_field_override

        account_id, _ = tenant
        save_field_override(account_id=account_id, table_fqn="WH.MART.SALES_FACT", schema_name="MART",
                            table_name="SALES_FACT", file_stem="SALES_FACT", column_name="NET_AMT",
                            meaning="Net sales", synonyms=["revenue"])
        assert _page_field(tenant, "NET_AMT")["synonyms"] == ["revenue"]

    def test_a_name_an_admin_gave_outranks_one_read_from_an_approved_meaning(self, tenant):
        from core.semantic_model import patch_field_approval

        account_id, kb = tenant
        assert patch_field_approval(kb_dir=kb, table_fqn="WH.MART.CUSTOMER_DIM", table_name="CUSTOMER_DIM",
                                    schema_name="MART", column_name="CUSTOMER_NAME",
                                    approved_meaning="Legal name of the customer. As registered.")
        assert _page_field(tenant, "CUSTOMER_NAME")["label"] == "Legal name of the customer"
        _confirm(account_id, "Customer", "CUSTOMER_NAME", display_name="Customer name")
        assert _page_field(tenant, "CUSTOMER_NAME")["label"] == "Customer name"

    def test_a_confirmed_business_meaning_names_the_column(self, tenant):
        import store

        account_id, _ = tenant
        store.save_business_meanings(account_id, [{"scope": "column", "subject": "REGION_CD",
                                                   "reading": "sales region", "synonyms": ["territory"],
                                                   "rule": "values", "confidence": 80}])
        assert _page_field(tenant, "REGION_CD")["label"] == ""
        store.decide_business_meaning(account_id, "column", "REGION_CD", "confirmed", reading="sales territory")
        field = _page_field(tenant, "REGION_CD")
        assert (field["label"], field["synonyms"]) == ("sales territory", ["territory"])

    def test_a_confirmation_with_the_synonyms_cleared_keeps_none(self, tenant):
        import store

        account_id, _ = tenant
        store.save_business_meanings(account_id, [{"scope": "column", "subject": "REGION_CD",
                                                   "reading": "sales region", "synonyms": ["territory"],
                                                   "rule": "values", "confidence": 80}])
        store.decide_business_meaning(account_id, "column", "REGION_CD", "confirmed", synonyms=[])
        assert _page_field(tenant, "REGION_CD")["synonyms"] == []


# ── What the knowledge base says, read as what it is ─────────────────────────

class TestTheKnowledgeBaseIsReadAsWhatItIs:

    def test_a_column_nothing_describes_needs_context(self, tenant):
        # FREIGHT_AMT is in the schema and nowhere in the prose: its sentence
        # is made from its name, which is not a meaning.
        field = _page_field(tenant, "FREIGHT_AMT")
        assert (field["needs_context"], field["confidence"], field["approved"]) == (True, 45, False)

    def test_a_row_that_only_lists_values_needs_context_too(self, tenant):
        # "values are ..." with nothing before it leaves no meaning: the page
        # showed the parser's "Needs business review." at 77% and no flag.
        from pathlib import Path

        _, kb = tenant
        path = Path(kb, "SALES_FACT_kb.md")
        path.write_text(path.read_text(encoding="utf-8").replace(
            "- `DOC_NO` (varchar): Document number.\n",
            "- `DOC_NO` (varchar): Document number.\n- `FREIGHT_AMT` (decimal): values are 0, 12.5\n"),
            encoding="utf-8")
        field = _page_field(tenant, "FREIGHT_AMT")
        assert (field["needs_context"], field["confidence"]) == (True, 45)

    def test_a_row_the_knowledge_base_flagged_still_shows_what_it_said(self, tenant):
        field = _page_field(tenant, "REGION_CD")
        assert (field["meaning"], field["needs_context"]) == ("Region code [NEEDS CONTEXT]", True)

    def test_an_admin_s_meaning_settles_a_flagged_row(self, tenant):
        account_id, _ = tenant
        _override(account_id, "CUSTOMER_DIM", "REGION_CD", "Sales region the customer is served from")
        field = _page_field(tenant, "REGION_CD")
        assert (field["meaning"], field["needs_context"], field["approved"]) == \
            ("Sales region the customer is served from", False, True)

    def test_a_generated_row_without_a_confidence_is_not_an_approval(self, tenant):
        # A knowledge base written as a table: the row carries a meaning and
        # no confidence cell. It was read as an admin's approval.
        from pathlib import Path

        _, kb = tenant
        Path(kb, "SALES_FACT_kb.md").write_text(
            "# WH.MART.SALES_FACT\n\n## Columns\n\n"
            "| Column | Type | Nullable | Values | Meaning | Use case |\n|---|---|---|---|---|---|\n"
            "| NET_AMT | decimal | No | | Net amount as the knowledge base wrote it | Summing sales |\n",
            encoding="utf-8")
        field = _page_field(tenant, "NET_AMT")
        assert (field["meaning"], field["approved"]) == ("Invoiced amount after rebates", False)

    def test_an_approval_the_patcher_wrote_into_a_row_is_one(self, tenant):
        from pathlib import Path

        _, kb = tenant
        Path(kb, "SALES_FACT_kb.md").write_text(
            "# WH.MART.SALES_FACT\n\n## Columns\n\n"
            "| Column | Type | Nullable | Values | Meaning | Use case | Confidence | Source |\n"
            "|---|---|---|---|---|---|---|---|\n"
            "| NET_AMT | decimal | No | | Net sales after returns | Summing sales | 100% | "
            "Admin-approved Semantic Layer edit |\n",
            encoding="utf-8")
        field = _page_field(tenant, "NET_AMT")
        assert (field["meaning"], field["approved"], field["meaning_evidence"]) == \
            ("Net sales after returns", True, "approved edit in the knowledge base")


# ── What a count is said to count ────────────────────────────────────────────

class TestTheCountTargetLabel:

    def test_an_approved_meaning_names_what_is_counted(self, tenant):
        from core.count_target_resolver import count_target_clarification_options, resolve_count_target
        from core.semantic_model import load_semantic_model, patch_field_approval

        _, kb = tenant
        assert patch_field_approval(kb_dir=kb, table_fqn="WH.MART.SALES_FACT", table_name="SALES_FACT",
                                    schema_name="MART", column_name="DOC_NO",
                                    approved_meaning="Invoice number. Printed on every invoice.")
        resolution = resolve_count_target("invoice", load_semantic_model(kb))
        assert [(c["column"], c["business_name"]) for c in resolution["candidates"]] == [("DOC_NO", "Invoice number")]
        assert count_target_clarification_options(resolution)[0]["value"] == "Invoice number"

    def test_without_an_approval_the_reading_of_the_name_is_used(self):
        from core.count_target_resolver import resolve_count_target

        model = {"tables": [{"table": "MART.ORDER_FACT", "type": "fact", "fields": [
            {"column": "ORDER_NO", "expanded_name": "order number", "role": "identifier", "status": "generated"}]}]}
        resolution = resolve_count_target("order", model)
        assert [c["business_name"] for c in resolution["candidates"]] == ["order number"]
