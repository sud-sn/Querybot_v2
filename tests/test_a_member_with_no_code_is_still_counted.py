"""
A member whose code was never filled in is still counted.

A population is counted on its own table by its members' code, so that a
history table's versions of one member count once. A member with no code
dropped out of that count: of 172 profit centres in one province, one had no
code, and "number of profit centres by province" said 171. Members with a
code are still counted by it; each row without one is now counted too, and a
placeholder row is still left out.

Only on the members' own table and by their own code. A master table keeps
other entities' codes too -- a warehouse table its region's -- and a
warehouse with no region code is no region: counting such rows as members
turned two regions into four. And an empty table counts none, where a SUM of
no rows is NULL.

A synthetic tenant (tests/answer_harness.py) whose warehouse table keeps a
placeholder member and a warehouse with no code; the warehouse and the model
are the only stand-ins.
"""

from __future__ import annotations

import pytest

from tests import answer_harness as harness


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("uncoded")) as built:
        yield built


def _members() -> int:
    """Every warehouse but the placeholder, with a code or without."""
    return sum(1 for key in harness.WAREHOUSES if key != 0)


class TestTheProductAnswers:

    @pytest.mark.parametrize("question,lang", [
        ("How many warehouses do we have?", "en"),
        ("Combien d'entrepôts avons-nous ?", "fr"),
    ])
    def test_the_warehouse_with_no_code_is_counted(self, warehouse, question, lang):
        assert any(code is None for key, (code, _name) in harness.WAREHOUSES.items() if key)
        answer = harness.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is False
        ((row,),) = [answer["rows"]]
        assert row["WAREHOUSE_COUNT"] == _members()


class TestTheColumnType:

    @pytest.mark.parametrize("declared,text", [
        ("nvarchar", True), ("varchar(20)", True), ("CHAR(4)", True), ("text", True), ("string", True),
        ("int", False), ("bigint", False), ("decimal(18,2)", False), ("", False),
    ])
    def test_only_a_text_code_is_trimmed(self, declared, text):
        from core.pipeline_helpers import _is_text_type

        assert _is_text_type(declared) is text

    def test_the_type_is_read_whatever_the_names_case(self):
        from core.pipeline_helpers import _column_type

        columns = {"WH.MART.WHS_DMS": {"whs_cd": "nvarchar", "WHS_DMS_KEY": "int"}}
        assert _column_type(columns, "MART.WHS_DMS", "WHS_CD") == "nvarchar"
        assert _column_type(columns, "MART.WHS_DMS", "[whs_dms_key]") == "int"
        assert _column_type(columns, "MART.OTHER", "WHS_CD") == ""


class TestTheTablesOwnMembers:

    MODEL = {"tables": [{
        "table": "WHS_DMS", "qualified_name": "MART.WHS_DMS", "type": "dimension",
        "grain_columns": ["WHS_DMS_KEY"],
        "fields": [
            {"column": "WHS_DMS_KEY", "role": "dimension_key", "aggregation": "identifier",
             "entity_prefix": "Warehouse", "business_candidates": ["warehouse key"]},
            {"column": "WHS_CD", "role": "attribute", "entity_prefix": "Warehouse",
             "business_candidates": ["warehouse code"]},
            {"column": "RGN_CD", "role": "attribute", "entity_prefix": "Region",
             "business_candidates": ["region code"]},
            # Another entity's code under the table's own prefix.
            {"column": "WHS_TYP_CD", "role": "attribute", "entity_prefix": "Warehouse",
             "business_candidates": ["warehouse type code"]},
        ],
    }]}

    @pytest.mark.parametrize("entity,column,own", [
        ("warehouse", "WHS_CD", True), ("region", "RGN_CD", False), ("warehouse type", "WHS_TYP_CD", False)])
    def test_its_own_code_names_its_members_and_another_entitys_does_not(self, entity, column, own):
        from core.count_target_resolver import resolve_population_count_target

        selected = resolve_population_count_target(entity, self.MODEL)["selected"]
        assert (selected["column"], selected["own_member_key"]) == (column, own)

    @pytest.mark.parametrize("grain,column,own", [
        ("WHS_DMS_KEY", "WHS_DMS_KEY", True), ("WHS_DMS_KEY", "WHS_CD", True), ("ITM_DMS_KEY", "ITM_NO", True),
        ("PFT_CTR_DMS_KEY", "PFT_CTR_CD", True), ("WarehouseKey", "WarehouseCode", True),
        # A warehouse with no type, region or province is no type, region or
        # province: "how many warehouse types" said 4 where there were 2.
        ("WHS_DMS_KEY", "WHS_TYP_CD", False), ("WHS_DMS_KEY", "WHS_RGN_CD", False),
        ("PFT_CTR_DMS_KEY", "PFT_CTR_PRV", False), ("WHS_DMS_KEY", "RGN_CD", False),
        # A child's table keeps its parent's code: a warehouse location with no
        # warehouse code is no warehouse, a ship-to with no customer number no
        # customer. Read as the table's own, "how many warehouses" would count
        # each location without one as a warehouse.
        ("WHS_LOC_DMS_KEY", "WHS_CD", False), ("CUST_SHP_TO_DMS_KEY", "CUST_NO", False),
        ("ITM_WHS_DMS_KEY", "ITM_CD", False), ("PTY_ADR_DMS_KEY", "PTY_CD", False),
        ("PFT_CTR_DMS_KEY", "PFT_CD", False), ("WHS_LOC_DMS_KEY", "WHS_LOC_CD", True),
        # A warehouse's storage rows are no warehouse either.
        ("WHS_ROW_DMS_KEY", "WHS_CD", False),
        # A history, hash, sequence or version key is still the table's own.
        ("WHS_HIST_KEY", "WHS_CD", True), ("WHS_HK", "WHS_CD", True), ("WHS_SID", "WHS_CD", True),
        ("WHS_VER_KEY", "WHS_CD", True), ("WHS_SCD_KEY", "WHS_CD", True), ("DIM_WHS_KEY", "WHS_CD", True),
        ("WHS_DW_KEY", "WHS_CD", True), ("WHS_SEQ", "WHS_CD", True), ("WHS_BK", "WHS_CD", True),
    ])
    def test_the_entity_the_grain_names_and_an_identifier(self, grain, column, own):
        from core.count_target_resolver import _names_the_tables_own_members

        assert _names_the_tables_own_members({"grain_columns": [grain]}, {"column": column}) is own

    @pytest.mark.parametrize("own", [True, False])
    def test_only_its_own_members_count_the_rows_without_a_code(self, own):
        from core.analytical_request_plan import compile_analytical_request_plan

        plan = compile_analytical_request_plan(
            "how many regions do we have",
            {"source_scope": {"source_kind": "master", "selected_fact": "MART.WHS_DMS"},
             "count_target": {"status": "selected", "selected": {
                 "table": "MART.WHS_DMS", "column": "WHS_CD" if own else "RGN_CD", "own_member_key": own}}},
            analytical_intent_plan={"measure_semantics": "count_distinct_business_identifier",
                                    "counted_entity": "warehouse" if own else "region"},
        )
        assert plan["derived_measure"]["members_table"] is own

    def test_an_empty_table_counts_none(self, warehouse, tmp_path):
        import copy

        answer = harness.ask(warehouse, "How many warehouses do we have?")
        empty = harness.Warehouse(tmp_path / "empty.duckdb", schema=copy.deepcopy(harness.SCHEMA),
                                  table_rows={**harness.rows(), "WHS_DMS": []})
        assert empty.query(answer["sql"]) == [{"WAREHOUSE_COUNT": 0}]
