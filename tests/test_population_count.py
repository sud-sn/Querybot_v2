"""
tests/test_population_count.py

B11, found by asking the live warehouse "how many customers do we have by
customer type" and reading the SQL it produced:

    SELECT CUS_TYP.CUS_TYP_NM, COUNT(DISTINCT CUS_ORD.CUS_DMS_KEY)
    FROM   ...CUS_ORD_IVC_FCT CUS_ORD
    JOIN   ...CUS_DMS ...
    JOIN   ...CUS_TYP_DMS ...

A customer with no invoice is not in that answer. The question asked how many
customers exist; the answer said how many customers have been invoiced, with no
warning that the two differ.

Underneath it the count was not governed at all. The counted entity came from a
hand-written list of event nouns -- orders, invoices, shipments, claims,
prescriptions -- which has no entry for customers, so no governed count target
was resolved and the model wrote whatever COUNT it liked over whichever fact
source arbitration had picked.

The tests below run against the real semantic model checked in at
clients/Test/kb/_semantic_model.json, so the resolver is exercised on metadata
a discovery run actually produced rather than on a hand-built fixture.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import pytest

from core.analytical_intent import (
    detect_business_event_count,
    detect_population_count,
    plan_analytical_intent,
)
from core.analytical_request_plan import (
    compile_analytical_request_plan,
    format_analytical_request_plan,
)
from core.count_target_resolver import resolve_population_count_target
from core.semantic_planner import format_semantic_field_plan
from core.validator import validate_sql_detailed


MODEL_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "clients" / "Test" / "kb" / "_semantic_model.json"
)


@pytest.fixture(scope="module")
def model() -> dict:
    return json.loads(MODEL_PATH.read_text(encoding="utf-8"))


# ── What the question is asking to count ─────────────────────────────────────


class TestTheCountedThingIsTheSubjectOfTheCountPhrase:
    @pytest.mark.parametrize("question,entity", [
        ("how many customers do we have by customer type", "customer"),
        ("how many customers do we have", "customer"),
        ("how many suppliers do we have", "supplier"),
        ("how many warehouses are there", "warehouse"),
        ("count of customers by region", "customer"),
        ("what is the total number of patients", "patient"),
        ("how many distinct suppliers do we have in total", "supplier"),
        ("how many profit centres do we have", "profit centre"),
        ("How many items do we have?", "item"),
    ])
    def test_a_population_question_names_its_population(self, question, entity):
        assert detect_population_count(question) == entity

    @pytest.mark.parametrize("question", [
        # Each of these narrows the population to activity, and the master
        # table on its own cannot answer that. They must keep the treatment
        # they have today rather than being answered from the master.
        "how many customers ordered last month",
        "how many drugs are in stock",
        "how many items do we stock",
        "how many customers bought from us in June",
        "how many suppliers delivered late",
        # Not a population at all.
        "how many days between order and delivery",
        "how many tables do you have",
        "how many rows are in the invoice table",
        "what is the total revenue",
        "show top customers by total order value",
    ])
    def test_anything_narrower_than_a_population_is_left_alone(self, question):
        assert detect_population_count(question) == ""

    def test_an_event_noun_in_a_predicate_is_not_what_is_counted(self):
        """"How many customers placed orders in June" used to report the number
        of ORDERS. The count phrase names its own subject; the event noun after
        it is a predicate about that subject."""
        assert detect_business_event_count("how many customers placed orders in June") == ""

    @pytest.mark.parametrize("question,event", [
        ("how many orders do we have", "order"),
        ("how many invoices were raised", "invoice"),
        ("Retrieve the top 10 warehouses with the most customer orders.", "order"),
        ("what is the total orders placed by each customer in the last 2 months", "order"),
        ("which customers have reduced orders recently", "order"),
    ])
    def test_real_event_counts_are_unaffected(self, question, event):
        assert detect_business_event_count(question) == event

    def test_the_two_detectors_never_both_claim_a_question(self):
        for question in [
            "how many orders do we have",
            "how many customers do we have",
            "how many invoices this month",
        ]:
            plan = plan_analytical_intent(question)
            assert not (plan.counted_entity and plan.population_entity), question


# ── Which table defines the population ───────────────────────────────────────


class TestThePopulationComesFromItsMasterTable:
    @pytest.mark.parametrize("entity,table,column", [
        ("patient", "PHARMA_LAB.D_PATIENT", "PATIENT_ID"),
        ("supplier", "PHARMA_LAB.D_SUPPLIER", "SUPPLIER_ID"),
        ("drug", "PHARMA_LAB.D_DRUG", "DRUG_ID"),
        ("prescriber", "PHARMA_LAB.D_PRESCRIBER", "PRESCRIBER_ID"),
    ])
    def test_the_master_table_identifier_is_the_count_target(
        self, model, entity, table, column,
    ):
        resolution = resolve_population_count_target(entity, model)
        assert resolution["status"] == "selected"
        assert resolution["selected"]["table"] == table
        assert resolution["selected"]["column"] == column

    def test_no_fact_table_is_ever_offered(self, model):
        """The whole point: counting from a fact answers a different question."""
        for entity in ("patient", "supplier", "drug", "payer", "pharmacy"):
            for candidate in resolve_population_count_target(entity, model)["candidates"]:
                assert ".F_" not in candidate["table"].upper()

    def test_a_business_identifier_beats_a_version_key(self, model):
        """On a slowly changing master the surrogate counts row versions; the
        business identifier counts members."""
        resolution = resolve_population_count_target("supplier", model)
        columns = [c["column"] for c in resolution["candidates"]]
        assert columns[0] == "SUPPLIER_ID"
        assert "SUPPLIER_CODE" in columns

    def test_a_neighbouring_master_cannot_answer_for_the_entity(self):
        """CUSTOMER_TYPE_CODE identifies a type of customer, not a customer.
        This is the live shape: the customer master sits beside a customer-type
        master, and both carry a column whose meaning starts with "customer"."""
        model = {"tables": [
            {"qualified_name": "DW.CUSTOMER_TYPE_DIM", "type": "dimension", "fields": [
                {"column": "CUSTOMER_TYPE_CODE", "expanded_name": "customer type code"},
                {"column": "CUSTOMER_TYPE_NAME", "expanded_name": "customer type name"},
            ]},
        ]}
        assert resolve_population_count_target("customer", model)["status"] == "missing"

    def test_a_name_column_is_not_an_identifier(self):
        model = {"tables": [
            {"qualified_name": "DW.SUPPLIER_DIM", "type": "dimension", "fields": [
                {"column": "SUPPLIER_NAME", "expanded_name": "supplier name"},
            ]},
        ]}
        assert resolve_population_count_target("supplier", model)["status"] == "missing"

    def test_two_masters_claiming_one_population_is_not_resolved(self, model):
        """A bridge table carries the same identifier as the dimension it
        bridges. Picking either would be a guess about which is authoritative,
        so the question keeps the treatment it has today."""
        assert resolve_population_count_target("diagnosis", model)["status"] == "ambiguous"

    @pytest.mark.parametrize("entity", ["customer", "order", "claim", "widget"])
    def test_an_entity_with_no_master_resolves_to_nothing(self, model, entity):
        assert resolve_population_count_target(entity, model)["status"] == "missing"

    def test_the_live_shape_that_started_this(self):
        """The warehouse this was found on: a customer master sitting beside a
        customer-type master and a customer-order-invoice fact, every one of
        them carrying a column whose meaning begins with "customer". The count
        has to land on the master, and the supplier master shows the other
        half -- its physical name expands to "sup no", so the only thing that
        connects it to the word "supplier" is the approved synonym."""
        model = {"tables": [
            {"qualified_name": "EMDW_DMART.CUS_DMS", "type": "dimension", "fields": [
                {"column": "CUS_DMS_KEY", "expanded_name": "customer dimension key",
                 "role": "dimension_key"},
                {"column": "CUS_NO", "expanded_name": "customer no"},
                {"column": "CUS_NM", "expanded_name": "customer name"},
            ]},
            {"qualified_name": "EMDW_DMART.CUS_TYP_DMS", "type": "dimension", "fields": [
                {"column": "CUS_TYP_CD", "expanded_name": "customer type code"},
                {"column": "CUS_TYP_NM", "expanded_name": "customer type name"},
            ]},
            {"qualified_name": "EMDW_DMART.SUP_DMS", "type": "dimension", "fields": [
                {"column": "SUP_NO", "expanded_name": "sup no",
                 "business_candidates": ["supplier"]},
                {"column": "SUP_NM", "expanded_name": "sup name",
                 "business_candidates": ["supplier"]},
            ]},
            {"qualified_name": "EMDW_DMART.CUS_ORD_IVC_FCT", "type": "fact", "fields": [
                {"column": "CUS_DMS_KEY", "expanded_name": "customer dimension key",
                 "role": "dimension_key"},
                {"column": "IVC_NO", "expanded_name": "invoice no"},
            ]},
        ]}

        customer = resolve_population_count_target("customers", model)
        assert customer["selected"]["table"] == "EMDW_DMART.CUS_DMS"
        assert customer["selected"]["column"] == "CUS_NO"

        # The qualifier master answers for the qualifier, not for the entity.
        assert resolve_population_count_target("customer types", model)["selected"] == {
            **resolve_population_count_target("customer type", model)["selected"],
        }
        assert (
            resolve_population_count_target("customer type", model)["selected"]["column"]
            == "CUS_TYP_CD"
        )

        supplier = resolve_population_count_target("supplier", model)
        assert supplier["selected"]["table"] == "EMDW_DMART.SUP_DMS"
        assert supplier["selected"]["column"] == "SUP_NO"

        # An invoice is an event, not a population: the business-event path
        # owns it, and this resolver must not compete for it.
        assert resolve_population_count_target("invoice", model)["status"] == "missing"

    def test_a_synonym_alone_does_not_make_a_name_an_identifier(self):
        model = {"tables": [
            {"qualified_name": "DW.SUP_DMS", "type": "dimension", "fields": [
                {"column": "SUP_NM", "expanded_name": "sup name",
                 "business_candidates": ["supplier"]},
            ]},
        ]}
        assert resolve_population_count_target("supplier", model)["status"] == "missing"

    def test_a_measure_is_never_a_count_target(self):
        model = {"tables": [
            {"qualified_name": "DW.SUPPLIER_DIM", "type": "dimension", "fields": [
                {"column": "SUPPLIER_ID", "expanded_name": "supplier id", "role": "measure"},
            ]},
        ]}
        assert resolve_population_count_target("supplier", model)["status"] == "missing"

    def test_plural_and_singular_wording_resolve_alike(self, model):
        singular = resolve_population_count_target("supplier", model)
        plural = resolve_population_count_target("suppliers", model)
        assert singular["selected"] == plural["selected"]


# ── What the rest of the pipeline is told ────────────────────────────────────


def _population_plan(model, question, entity, *, extra_fields=()):
    """Build the semantic plan exactly as the pipeline branch assembles it."""
    intent = plan_analytical_intent(question)
    resolution = resolve_population_count_target(entity, model)
    selected = resolution["selected"]
    plan = {
        "fields": list(extra_fields),
        "joins": [],
        "count_target": resolution,
        "source_scope": {
            "status": "selected",
            "selected_fact": selected["table"],
            "selected_facts": [],
            "candidates": [],
            "source_kind": "master",
            "reason": "governed population master table",
        },
    }
    intent = dataclasses.replace(
        intent,
        counted_entity=entity,
        measure_semantics="count_distinct_business_identifier",
    )
    return plan, intent, selected


class TestTheCompiledRequestCountsThePopulation:
    def test_the_request_compiles_against_the_master(self, model):
        plan, intent, selected = _population_plan(
            model, "how many suppliers do we have", "supplier",
        )
        compiled = compile_analytical_request_plan(
            "how many suppliers do we have", plan,
            analytical_intent_plan=intent.to_dict(),
        )
        assert compiled["status"] == "compiled"
        assert compiled["missing_slots"] == []
        assert compiled["source_facts"] == [selected["table"]]
        assert compiled["derived_measure"]["target_table"] == selected["table"]
        assert compiled["derived_measure"]["target_column"] == "SUPPLIER_ID"

    def test_an_incidental_numeric_on_the_master_cannot_move_the_source(self, model):
        """Master tables carry numbers -- a credit limit, a lead time. Letting
        one claim the measure fact would move the request off the very table
        the governed count target lives on, and the validator would then demand
        both."""
        plan, intent, selected = _population_plan(
            model, "how many suppliers do we have", "supplier",
            extra_fields=[{
                "term": "lead time", "table": "PHARMA_LAB.F_PURCHASE_RECEIPT",
                "column": "LEAD_TIME_DAYS", "role": "measure", "enforcement": "required",
            }],
        )
        compiled = compile_analytical_request_plan(
            "how many suppliers do we have", plan,
            analytical_intent_plan=intent.to_dict(),
        )
        assert compiled["source_fact"] == selected["table"]
        assert compiled["source_facts"] == [selected["table"]]

    def test_the_prompt_forbids_reaching_the_population_through_a_fact(self, model):
        plan, intent, _ = _population_plan(
            model, "how many suppliers do we have", "supplier",
        )
        plan["analytical_request_plan"] = compile_analytical_request_plan(
            "how many suppliers do we have", plan,
            analytical_intent_plan=intent.to_dict(),
        )
        text = format_semantic_field_plan(plan, db_type="snowflake")
        assert "defines the population this question counts" in text
        assert "drop every member with no activity" in text
        assert "single fact for measures" not in text

    def test_an_ordinary_measure_question_keeps_its_fact_wording(self):
        plan = {
            "fields": [{
                "term": "amount", "table": "PHARMA_LAB.F_CLAIM",
                "column": "PAID_AMOUNT", "role": "measure",
            }],
            "joins": [],
            "source_scope": {"status": "selected", "selected_fact": "PHARMA_LAB.F_CLAIM"},
        }
        text = format_semantic_field_plan(plan, db_type="snowflake")
        assert "single fact for measures" in text
        assert "defines the population" not in text

    def test_the_plan_text_names_the_exact_count(self, model):
        plan, intent, _ = _population_plan(
            model, "how many suppliers do we have", "supplier",
        )
        compiled = compile_analytical_request_plan(
            "how many suppliers do we have", plan,
            analytical_intent_plan=intent.to_dict(),
        )
        text = format_analytical_request_plan(compiled)
        assert "COUNT(DISTINCT PHARMA_LAB.D_SUPPLIER.SUPPLIER_ID)" in text
        assert "without joining a fact" in text


class TestTheValidatorHoldsTheAnswerToThePopulation:
    KNOWN = {"PHARMA_LAB.D_SUPPLIER", "PHARMA_LAB.F_PURCHASE_RECEIPT"}
    COLUMNS = {
        "PHARMA_LAB.D_SUPPLIER": {
            "SUPPLIER_ID": "int", "SUPPLIER_NAME": "varchar", "STATE_CODE": "varchar",
        },
        "PHARMA_LAB.F_PURCHASE_RECEIPT": {
            "SUPPLIER_ID": "int", "RECEIPT_ID": "int",
        },
    }

    def _validate(self, model, sql):
        plan, intent, _ = _population_plan(
            model, "how many suppliers do we have by state", "supplier",
        )
        compiled = compile_analytical_request_plan(
            "how many suppliers do we have by state", plan,
            analytical_intent_plan=intent.to_dict(),
        )
        return validate_sql_detailed(
            sql, self.KNOWN, db_type="snowflake",
            allowed_tables=self.KNOWN, table_columns=self.COLUMNS,
            semantic_context={"analytical_request_plan": compiled},
        )

    def test_counting_the_master_is_accepted(self, model):
        result = self._validate(model, (
            "SELECT s.STATE_CODE, COUNT(DISTINCT s.SUPPLIER_ID) AS SUPPLIER_COUNT "
            "FROM PHARMA_LAB.D_SUPPLIER s GROUP BY s.STATE_CODE"
        ))
        assert result.ok, result.message

    def test_counting_suppliers_seen_on_a_fact_is_rejected(self, model):
        """The live defect, in one assertion: this SQL answers "how many
        suppliers have a receipt", and the question did not ask that."""
        result = self._validate(model, (
            "SELECT COUNT(DISTINCT r.SUPPLIER_ID) AS SUPPLIER_COUNT "
            "FROM PHARMA_LAB.F_PURCHASE_RECEIPT r"
        ))
        assert not result.ok

    def test_counting_rows_instead_of_members_is_rejected(self, model):
        result = self._validate(model, (
            "SELECT COUNT(*) AS SUPPLIER_COUNT FROM PHARMA_LAB.D_SUPPLIER"
        ))
        assert not result.ok


# ── The pipeline glue, executed ──────────────────────────────────────────────


class TestThePipelineActuallyTakesTheBranch:
    """This repository's recurring failure is code that ships and never runs:
    a route with no caller, a handler that fails open at debug level, a test
    that asserts on source text while the fix sits in a dead branch. So rather
    than assert that the branch LOOKS right, lift it out of the shipped
    function and run it.
    """

    @staticmethod
    def _branch_source() -> str:
        import ast

        source = (
            pathlib.Path(__file__).resolve().parent.parent
            / "core" / "query_pipeline.py"
        ).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.If):
                continue
            if ast.unparse(node.test) != "_analytical_plan.counted_entity":
                continue
            for branch in node.orelse:
                if (
                    isinstance(branch, ast.If)
                    and ast.unparse(branch.test) == "_analytical_plan.population_entity"
                ):
                    return ast.unparse(branch)
        raise AssertionError(
            "the population-count branch is no longer attached to the "
            "governed-count conditional in _handle_query_impl"
        )

    def _run(self, model, question, *, planner=None, seen=None):
        import logging

        from core.count_target_resolver import resolve_population_count_target
        from core.pipeline_context import _merge_semantic_plans

        def _default_planner(*_args, **kwargs):
            if seen is not None:
                seen.append(sorted(kwargs.get("preferred_fact_tables") or []))
            return {
                "fields": [{
                    "term": "state", "table": "PHARMA_LAB.D_SUPPLIER",
                    "column": "STATE_CODE", "role": "dimension",
                }],
                "joins": [],
            }

        build = planner or _default_planner
        namespace = {
            "_analytical_plan": plan_analytical_intent(question),
            "_source_model": model,
            "_semantic_plan": {"fields": [], "joins": []},
            "_source_scope": {
                "status": "selected",
                "selected_fact": "PHARMA_LAB.F_PURCHASE_RECEIPT",
            },
            "_preferred_facts": {"PHARMA_LAB.F_PURCHASE_RECEIPT"},
            "_count_target_resolution": {},
            "resolve_population_count_target": resolve_population_count_target,
            "_merge_semantic_plans": _merge_semantic_plans,
            "build_semantic_field_plan": build,
            "build_runtime_semantic_plan": build,
            "_dataclass_replace": dataclasses.replace,
            "_trace_step": lambda *a, **k: None,
            "trace_id": None,
            "log": logging.getLogger("test.population"),
            "account_id": "acct",
            "_semantic_plan_question": question,
            "all_columns": {},
            "query_scope_tables": set(),
            "schema_hint": "",
            "_vocab": None,
            "_planner_fact_tables": {"PHARMA_LAB.F_PURCHASE_RECEIPT"},
            "state": {"kb_dir": ""},
            "_contract_model": model,
            "_planner_terms": [],
        }
        exec(self._branch_source(), namespace)
        return namespace

    def test_the_branch_anchors_the_request_on_the_master(self, model):
        after = self._run(model, "how many suppliers do we have by state")

        assert after["_count_target_resolution"]["status"] == "selected"
        assert after["_source_scope"]["selected_fact"] == "PHARMA_LAB.D_SUPPLIER"
        assert after["_source_scope"]["source_kind"] == "master"
        assert after["_preferred_facts"] == {"PHARMA_LAB.D_SUPPLIER"}
        assert after["_semantic_plan"]["count_target"]["selected"]["column"] == "SUPPLIER_ID"
        assert after["_semantic_plan"]["source_scope"]["source_kind"] == "master"

    def test_the_branch_promotes_the_population_to_the_counted_entity(self, model):
        after = self._run(model, "how many suppliers do we have")
        plan = after["_analytical_plan"]

        assert plan.counted_entity == "supplier"
        assert plan.measure_semantics == "count_distinct_business_identifier"

    def test_the_planners_are_rerun_against_the_master(self, model):
        """Both field planners ran before the master was known, so both are
        rebuilt against it -- otherwise fields, joins and required tables still
        describe the fact the request has just moved off."""
        seen: list[list[str]] = []
        self._run(model, "how many suppliers do we have", seen=seen)
        assert seen == [["PHARMA_LAB.D_SUPPLIER"], ["PHARMA_LAB.D_SUPPLIER"]]

    def test_an_unresolvable_population_changes_nothing(self, model):
        """"How many widgets do we have" has no master table. The question must
        be left exactly as it was rather than start refusing."""
        after = self._run(model, "how many widgets do we have")

        assert after["_count_target_resolution"] == {}
        assert after["_source_scope"]["selected_fact"] == "PHARMA_LAB.F_PURCHASE_RECEIPT"
        assert after["_preferred_facts"] == {"PHARMA_LAB.F_PURCHASE_RECEIPT"}
        assert "count_target" not in after["_semantic_plan"]
        assert after["_analytical_plan"].counted_entity == ""

    def test_a_failed_replan_leaves_no_half_applied_count(self, model):
        """A half-applied target compiles a request that demands a count target
        it no longer has, which refuses a question the ordinary path could
        still have answered."""
        def _boom(*_args, **_kwargs):
            raise RuntimeError("planner unavailable")

        after = self._run(
            model, "how many suppliers do we have", planner=_boom,
        )

        assert after["_count_target_resolution"] == {}
        assert "count_target" not in after["_semantic_plan"]
        assert after["_analytical_plan"].counted_entity == ""
        assert after["_source_scope"]["selected_fact"] == "PHARMA_LAB.F_PURCHASE_RECEIPT"
