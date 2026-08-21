"""
tests/test_dimension_key_vocabulary.py

Nine places had spelled Infor M3's "_DMS_KEY" into Python and used it to decide
something about every tenant: whether a column is a key or a measure, whether a
relationship exists at all, which prefix names a dimension's label column,
which date keys are YYYYMMDD-encoded, and what the repair prompt calls the
column it wants moved out of SELECT.

Each test below runs the real function twice on the SAME schema shape -- once
spelled M3's way, once spelled the way a Kimball warehouse or an ERP extract
would spell it -- and asserts the two agree. That is the property that was
missing; asserting the literal is gone would not have caught the divergence.

The worst of them was core/semantic_planner.py::_role_for_column. It was the
only structural signal that a column is a dimension key, so on every other
warehouse a foreign key fell through to the numeric branch and typed as a
MEASURE -- something to SUM. That is not cosmetic: role gates the display-name
upgrade, and a mis-typed key can win the measure-fact anchor and demote the
real measure to optional, so a staffing fact answers a question about charges.
"""

from __future__ import annotations

import pytest

import core.semantic_model as semantic_model
import core.semantic_planner as planner
from core.vocab_packs import (
    dimension_key_suffixes,
    is_dimension_key_column,
    is_key_column,
    key_column_suffixes,
    strip_dimension_key_suffix,
)


# ── The two questions the one literal used to answer ─────────────────────────


class TestJoiningAndTypingAreDifferentQuestions:
    """Whether a shared column may ESTABLISH a join is a question about a
    relationship, and a bare _KEY or _ID cannot answer it -- it is a surrogate
    key on one warehouse and a sort key on another. Whether a column is a key
    rather than something to SUM is a question about the column alone, and
    there the same suffix is decisive."""

    @pytest.mark.parametrize("column", [
        "CUS_DMS_KEY", "PLANT_SK", "PATIENT_DIM_KEY", "ORDER_FK",
    ])
    def test_an_unambiguous_key_answers_both(self, column):
        assert is_dimension_key_column(column)
        assert is_key_column(column)

    @pytest.mark.parametrize("column", ["DEPARTMENT_KEY", "PATIENT_ID"])
    def test_an_ambiguous_key_types_as_a_key_but_does_not_join(self, column):
        assert is_key_column(column)
        assert not is_dimension_key_column(column)

    @pytest.mark.parametrize("column", ["CHARGE_AMOUNT", "CUS_NM", "NET_QTY"])
    def test_a_measure_or_a_label_answers_neither(self, column):
        assert not is_key_column(column)
        assert not is_dimension_key_column(column)

    def test_the_typing_set_is_a_superset_of_the_joining_set(self):
        assert set(dimension_key_suffixes()) < set(key_column_suffixes())

    @pytest.mark.parametrize("column,prefix", [
        ("CUS_DMS_KEY", "CUS"), ("PATIENT_SK", "PATIENT"),
        ("PRODUCT_DIM_KEY", "PRODUCT"), ("WHS_DMS_KEY", "WHS"),
    ])
    def test_the_prefix_is_whatever_remains(self, column, prefix):
        assert strip_dimension_key_suffix(column) == prefix

    def test_a_two_part_suffix_comes_off_whole(self):
        """Longest first, or "_KEY" strips from CUS_DMS_KEY and leaves a stray
        "_DMS" that matches no dimension."""
        assert strip_dimension_key_suffix("CUS_DMS_KEY") == "CUS"

    def test_a_column_with_no_key_suffix_yields_nothing(self):
        assert strip_dimension_key_suffix("CUS_NM") == ""


# ── A foreign key is not a measure, whatever it is called ────────────────────


class TestAKeyIsTypedAsAKeyOnAnyWarehouse:
    @pytest.mark.parametrize("column", [
        "WHS_DMS_KEY", "PLANT_SK", "PLANT_FK", "DEPARTMENT_KEY",
        "PATIENT_ID", "PRODUCT_DIM_KEY", "ACCT_DIMENSION_KEY",
    ])
    def test_an_integer_foreign_key_is_a_dimension_not_a_measure(self, column):
        """The bug: only the M3 spelling reached the dimension branch, so every
        other convention fell through to "any numeric column is a measure"."""
        assert planner._role_for_column(column, "INT") == "dimension"

    @pytest.mark.parametrize("column", ["CHARGE_AMOUNT", "NET_QTY", "UNIT_CST"])
    def test_a_real_measure_is_still_a_measure(self, column):
        assert planner._role_for_column(column, "DECIMAL") == "measure"

    @pytest.mark.parametrize("column", [
        "INVOICE_DT_DMS_KEY", "ADMIT_DATE_SK", "SERVICE_DATE_ID", "POSTING_DATE_KEY",
    ])
    def test_a_date_key_is_a_date_key_on_any_warehouse(self, column):
        assert planner._role_for_column(column, "INT") == "date_key"


# ── The relationship graph is not gated on one ERP's suffix ─────────────────


def _schema(fact_key: str, dim_table: str, dim_key: str) -> dict:
    return {
        "DW.SALES_FACT": {
            "database": "DW", "schema": "DW", "table": "SALES_FACT",
            "columns": [
                {"name": fact_key, "type": "int"},
                {"name": "NET_AMOUNT", "type": "decimal"},
            ],
        },
        f"DW.{dim_table}": {
            "database": "DW", "schema": "DW", "table": dim_table,
            "columns": [
                {"name": dim_key, "type": "int"},
                {"name": "CUSTOMER_NAME", "type": "varchar"},
            ],
        },
    }


class TestEveryConventionProducesRelationships:
    @pytest.mark.parametrize("fact_key,dim_table,dim_key", [
        ("CUS_DMS_KEY", "CUS_DMS", "CUS_DMS_KEY"),
        ("CUSTOMER_SK", "CUSTOMER_DIM", "CUSTOMER_SK"),
        ("CUSTOMER_DIM_KEY", "CUSTOMER_DIM", "CUSTOMER_DIM_KEY"),
    ])
    def test_a_fact_key_finds_its_dimension(self, fact_key, dim_table, dim_key):
        """`_relationships` skipped any column not ending _DMS_KEY, so on a
        Kimball warehouse the semantic model came back with no relationships
        at all -- an empty graph, not a wrong one."""
        rels = semantic_model._relationships(_schema(fact_key, dim_table, dim_key))
        assert rels, f"{fact_key} produced no relationship"
        assert any(dim_table in str(r.get("to_table") or r.get("to") or r) for r in rels)


class TestTheLabelColumnIsFoundByAnyKeyName:
    COLUMNS = ["PATIENT_SK", "PATIENT_NAME", "PATIENT_CODE", "ADMIT_DATE_SK"]

    def test_the_display_field_is_found_from_a_kimball_key(self):
        assert semantic_model._display_field_for_columns(
            self.COLUMNS, prefix="PATIENT_SK",
        ) == "PATIENT_NAME"

    def test_the_code_field_is_found_from_a_kimball_key(self):
        assert semantic_model._code_field_for_columns(
            self.COLUMNS, prefix="PATIENT_SK",
        ) == "PATIENT_CODE"

    def test_the_m3_spelling_still_works(self):
        columns = ["CUS_DMS_KEY", "CUS_NM", "CUS_CD"]
        assert semantic_model._display_field_for_columns(columns, prefix="CUS_DMS_KEY") == "CUS_NM"
        assert semantic_model._code_field_for_columns(columns, prefix="CUS_DMS_KEY") == "CUS_CD"


class TestTheBusinessRoleDropsThePlumbing:
    @pytest.mark.parametrize("column,role", [
        ("CUS_DMS_KEY", "customer"),
        ("PATIENT_DIM_KEY", "patient"),
        ("PRODUCT_DIMENSION_KEY", "product"),
    ])
    def test_the_key_suffix_never_reaches_the_user(self, column, role):
        """A suffix left on the name surfaces in the UI -- INVOICE_DATE_SK once
        read back as "Invoice Sk Date"."""
        assert semantic_model._business_role_from_column(column) == role


# ── The prompts stop naming one client's schema ─────────────────────────────


class TestThePromptsAreTenantNeutral:
    def test_the_star_join_example_uses_placeholders(self):
        from core.llm import build_sql_system_prompt

        prompt = build_sql_system_prompt("snowflake", "TABLE: DW.ANY\n  COL: X\n")
        for literal in ("CUS_ORD_IVC_FCT", "PC_DVN_DMS", "PC_DVN_DMS_KEY"):
            assert literal not in prompt, f"{literal} is in every tenant's prompt"

    def test_the_display_repair_rule_names_no_erp_convention(self):
        plan = {
            "fields": [{
                "term": "patient", "table": "DW.PATIENT_DIM", "column": "PATIENT_NAME",
                "source_table": "DW.ENCOUNTER_FACT", "source_key_column": "PATIENT_SK",
                "display_required": True,
            }],
        }
        rule = semantic_model.build_field_plan_repair_note(plan, [{
            "code": "field_plan_mismatch",
            "table": "DW.PATIENT_DIM", "column": "PATIENT_NAME",
        }])
        assert rule, "the fixture must produce a repair note"
        assert "_DMS_KEY" not in rule, (
            "the repair instruction named one ERP's key spelling three times, so "
            "it described a warehouse the tenant does not have"
        )
        assert "surrogate key" in rule
        assert "PATIENT_SK" in rule, "it should name the tenant's actual join key"


# ── The date-key rule follows governed metadata, not a suffix ───────────────


class TestTheDateKeyRuleFollowsTheTenantsOwnMetadata:
    """One ERP's suffix used to decide, for every tenant, whether a date key is
    a plain surrogate or a YYYYMMDD smart key -- and the rule it gates makes an
    affirmative factual claim about the client's own column. Anything not
    spelled _DT_DMS_KEY/_DATE_DMS_KEY was declared a plain surrogate, so a
    warehouse with a declared YYYYMMDD key got a capitalised instruction saying
    the column "has NO inherent calendar meaning and is NOT YYYYMMDD-encoded"
    and that the decode expression -- the one the plan emits for that column in
    the same request -- is wrong.
    """

    MARK = "has NO inherent calendar meaning"

    def _rule_present(self, context: str, plan: dict | None = None) -> bool:
        from core.llm import build_sql_system_prompt

        return self.MARK in build_sql_system_prompt(
            "azure_sql", context, semantic_plan=plan,
        )

    def test_a_kimball_surrogate_date_key_now_gets_the_rule(self):
        """_SK was invisible: this module kept its own (_ID|_KEY)$ copy of a
        predicate core/date_roles.py had already widened to include _SK/_FK."""
        assert self._rule_present("TABLE: DW.ENCOUNTER_FACT\n  ADMIT_DATE_SK int\n")

    def test_a_declared_encoded_key_is_not_told_it_is_unencoded(self):
        assert not self._rule_present(
            "TABLE: DW.CHARGES\n  SERVICE_DATE_KEY int\n",
            {"fields": [{"column": "SERVICE_DATE_KEY", "date_key_type": "yyyymmdd_integer"}]},
        )

    def test_the_same_column_declared_a_surrogate_does_get_the_rule(self):
        """The column name is identical -- only the tenant's governed encoding
        differs, which is the whole point."""
        assert self._rule_present(
            "TABLE: DW.CHARGES\n  SERVICE_DATE_KEY int\n",
            {"fields": [{"column": "SERVICE_DATE_KEY", "date_key_type": "surrogate_fk"}]},
        )

    def test_the_encoded_m3_key_is_unchanged(self):
        assert not self._rule_present(
            "TABLE: EMDW_DMART.CUS_ORD_IVC_FCT\n  INVOICE_DT_DMS_KEY bigint\n"
        )

    def test_a_plain_date_column_is_still_left_alone(self):
        """is_date_role_column also matches ORDER_DATE, a native DATE column;
        telling the model to avoid YEAR() on one would be wrong."""
        assert not self._rule_present("TABLE: DW.ORDERS\n  ORDER_DATE date\n")
