"""
A member named in several columns is still named.

"Online sales in Canada in 2025" was answered, on a made-up retailer's
warehouse, with every country's sales for 2025. "Canada" is a value in the
warehouse -- in the sales territory's region and country, and in the
customer's country, in English and in French -- and the value resolver,
finding it verbatim in more than one column, dropped it without a word: the
question named no member as far as the pipeline knew, so the governed compiler,
which writes no member filter, answered it whole.

A value found in several columns is now kept as named: the compilers step
aside for the SQL writer, which is told every column the value is in and that
the filter may not be left out. What a regulated tenant's writer is told is
screened column by column, as every other grounded value is.

tests/star_harness.py keeps Canada as both a region and a country of its
sales territories.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("several-columns")) as built:
        yield built


class TestTheProductAnswers:

    @pytest.mark.parametrize("question,lang", [
        ("Sales amount in Canada", "en"),
        ("Montant des ventes au Canada", "fr"),
    ])
    def test_the_member_is_never_left_out(self, warehouse, question, lang):
        answer = star.ask(warehouse, question, lang)
        # No compiled answer for every country: the SQL writer is asked, and
        # told where Canada is.
        assert answer["model_wrote_sql"] is True and answer["rows"] == []
        prompt = answer["prompts"][-1]
        assert "is the value 'Canada' in several columns" in prompt
        assert "DimSalesTerritory.SalesTerritoryRegion" in prompt
        assert "DimSalesTerritory.SalesTerritoryCountry" in prompt


def _index(base: str, values: dict[str, list[str]]) -> None:
    from core.value_index import build_value_index

    schema_dir = Path(tempfile.mkdtemp(dir=base)) / "schema"
    schema_dir.mkdir()
    schema = {"DB.SALES.TERRITORY_DIM": {
        "columns": [{"name": column, "type": "varchar(40)"} for column in values],
        "schema": "SALES", "database": "DB", "masked_fields": [], "mask_mode": "partial"}}
    (schema_dir / "_schema.json").write_text(json.dumps(schema), encoding="utf-8")

    def run(creds, db_type, sql, max_rows=200):
        for column, found in values.items():
            if column in sql:
                return [{column: value} for value in found]
        return []

    with patch("core.compliance.policy_engine.is_regulated", return_value=False):
        build_value_index("acct-several", {}, "azure_sql", str(schema_dir), run_query_fn=run, base_dir=base)


class TestTheResolver:

    def test_a_value_in_two_columns(self, tmp_path):
        from core.value_resolver import resolve_literals

        _index(str(tmp_path), {"SALES_REGION": ["Canada", "Northwest"], "MARKET_REGION": ["Canada", "France"]})
        found = resolve_literals("acct-several", "sales in Canada", base_dir=str(tmp_path))
        assert found["verified"] == []
        assert [(item["phrase"], item["value"], sorted(p["column"] for p in item["columns"]))
                for item in found["several"]] == [("Canada", "Canada", ["MARKET_REGION", "SALES_REGION"])]

    def test_a_value_in_one_column_is_verified_as_before(self, tmp_path):
        from core.value_resolver import resolve_literals

        _index(str(tmp_path), {"SALES_REGION": ["Canada", "Northwest"], "MARKET_REGION": ["France"]})
        found = resolve_literals("acct-several", "sales in Canada", base_dir=str(tmp_path))
        assert [(item["phrase"], item["column"]) for item in found["verified"]] == [("Canada", "SALES_REGION")]
        assert found.get("several", []) == []


SEVERAL = {"several": [{"phrase": "Canada", "value": "Canada", "columns": [
    {"table_fqn": "DB.SALES.TERRITORY_DIM", "column": "SALES_REGION", "business_name": "region"},
    {"table_fqn": "DB.SALES.CUSTOMER_DIM", "column": "MARKET_REGION", "business_name": "country"}]}]}


def _regulated(classifications: dict):
    return (patch("core.compliance.policy_engine.is_regulated", return_value=True),
            patch("store.get_compliance_profile", return_value={"industry": "healthcare_pharmacy",
                                                                 "policy_pack_key": "healthcare_pharmacy_v1"}),
            patch("store.get_classification_map", return_value=classifications))


class TestWhatTheWriterIsTold:

    def test_every_column_and_the_filter_kept(self):
        from core.value_resolver import build_verified_values_injection

        block = build_verified_values_injection(SEVERAL)
        assert "'Canada' is the value 'Canada' in several columns" in block
        assert "DB.SALES.TERRITORY_DIM.SALES_REGION, DB.SALES.CUSTOMER_DIM.MARKET_REGION" in block
        assert "never leave the filter out" in block

    def test_a_regulated_tenant_is_told_only_the_cleared_columns(self):
        from core.value_resolver import filter_resolved_for_compliance

        regulated, profile, classes = _regulated({
            "DB.SALES.TERRITORY_DIM.SALES_REGION": {"reviewed": True, "tags": []},
            "DB.SALES.CUSTOMER_DIM.MARKET_REGION": {"reviewed": True, "tags": ["PII"]}})
        with regulated, profile, classes:
            out, evidence = filter_resolved_for_compliance("acct", SEVERAL)
        assert [p["column"] for p in out["several"][0]["columns"]] == ["SALES_REGION"]
        assert evidence["dropped_columns"] == ["DB.SALES.CUSTOMER_DIM.MARKET_REGION"]

    def test_with_no_column_cleared_nothing(self):
        from core.value_resolver import filter_resolved_for_compliance

        regulated, profile, classes = _regulated({})
        with regulated, profile, classes:
            out, _evidence = filter_resolved_for_compliance("acct", SEVERAL)
        assert out["several"] == []

    def test_a_tenant_with_no_policy_pack_is_told_nothing(self):
        from core.value_resolver import filter_resolved_for_compliance

        regulated, profile, classes = _regulated({
            "DB.SALES.TERRITORY_DIM.SALES_REGION": {"reviewed": True, "tags": []}})
        with regulated, profile, classes, patch("core.compliance.packs.get_pack", return_value={}):
            out, evidence = filter_resolved_for_compliance("acct", SEVERAL)
        assert out["several"] == []
        assert evidence["reason"] == "no_policy_pack"

    def test_a_filter_that_fails_tells_nothing(self):
        from core.value_resolver import filter_resolved_for_compliance

        with patch("core.compliance.policy_engine.is_regulated", side_effect=RuntimeError("down")):
            out, _evidence = filter_resolved_for_compliance("acct", SEVERAL)
        assert out["several"] == []

    def test_an_unregulated_tenant_is_told_all(self):
        from core.value_resolver import filter_resolved_for_compliance

        with patch("core.compliance.policy_engine.is_regulated", return_value=False):
            out, _evidence = filter_resolved_for_compliance("acct", SEVERAL)
        assert out["several"] == SEVERAL["several"]
