"""
A key with no dimension of its own names none.

"Top 5 product groups by inventory value" was never answered on the sample
tenant; one of the three things it bound "product group" to was the ITEM NAME.
The item table carries a product-group key whose own table the warehouse does
not have. A key is resolved to the dimension it is the primary key of, then to
the table its name points at, and last to the first dimension table that
carries the column -- which, for a key no other table carries, is the item
table itself. So the item became a "Product Group" dimension shown by item
names, and a purchase-group key a "Purchase Group" one, beside the real product
group the question meant.

A synthetic tenant (tests/answer_harness.py) whose item table carries a
product-group key with no table of its own.
"""

from __future__ import annotations

import pytest

from tests import answer_harness as harness


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    import store
    from core.semantic_model import load_semantic_model

    with harness.tenant_in(tmp_path_factory.mktemp("keys-with-no-dimension")):
        yield load_semantic_model(store.get_client_state(harness.ACCOUNT)["kb_dir"])


def _entries(model: dict, table: str, key: str) -> list[dict]:
    return [
        dimension
        for entry in model.get("tables") or []
        if str(entry.get("qualified_name") or "").upper().endswith(table)
        for dimension in entry.get("dimensions") or []
        if dimension.get("source_key") == key
    ]


class TestTheModel:

    def test_the_items_product_group_key_names_no_dimension(self, model):
        (entry,) = _entries(model, "ITM_DMS", "PRU_GRP_DMS_KEY")
        assert (entry["display_table"], entry["display_column"]) == ("", "")

    def test_the_item_is_still_the_items_dimension(self, model):
        (entry,) = _entries(model, "ITM_BAL_DLY_FCT", "ITM_DMS_KEY")
        assert (entry["display_table"].upper(), entry["display_column"]) == ("MART.ITM_DMS", "ITM_NM")


def _table(columns, pk=()):
    return {"columns": [{"name": column, "type": "int"} for column in columns], "pk_columns": list(pk)}


_SCHEMA = {
    "WH.MART.ITM_DMS": _table(["ITM_DMS_KEY", "ITM_NM", "PRU_GRP_DMS_KEY"], ["ITM_DMS_KEY"]),
    "WH.MART.CUS_DMS": _table(["CUS_DMS_KEY", "CUS_NM", "SEG_DMS_KEY"], ["CUS_DMS_KEY"]),
}


class TestTheRule:

    @pytest.mark.parametrize("source", ["WH.MART.ITM_DMS", "MART.ITM_DMS", "[MART].[ITM_DMS]", "itm_dms"])
    def test_containment_never_answers_with_the_keys_own_table(self, source):
        from core.semantic_model import _find_dimension_for_key

        assert _find_dimension_for_key(_SCHEMA, "PRU_GRP_DMS_KEY", source) is None

    def test_the_keys_own_primary_key_still_names_it(self):
        # A table no naming convention finds is named by its primary key alone.
        from core.semantic_model import _find_dimension_for_key

        schema = {"WH.MART.DIM_LOCATION": _table(["LOCATION_SK", "LOCATION_NAME"], ["LOCATION_SK"])}
        assert _find_dimension_for_key(schema, "LOCATION_SK", "WH.MART.DIM_LOCATION")[0] == "WH.MART.DIM_LOCATION"

    def test_another_table_that_carries_it_still_answers(self):
        # Only the key's own table is set aside: what the schema offers
        # elsewhere is still the last resort it was.
        from core.semantic_model import _find_dimension_for_key

        assert _find_dimension_for_key(_SCHEMA, "SEG_DMS_KEY", "WH.MART.ITM_DMS")[0] == "WH.MART.CUS_DMS"
