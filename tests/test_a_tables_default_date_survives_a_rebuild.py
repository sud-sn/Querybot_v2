"""
A table's default date survives a knowledge-base rebuild.

An admin marks one of a fact's approved date roles as its default, so a
question that names no date ("revenue for 2026") uses it instead of asking. The
rebuild re-derives every date role and copies an approved role's settings back
from a fixed list of keys -- which did not include the default flag, nor the
flag that records a deliberate "no default". So after every KB build the chosen
default was gone (generic questions asked which date again), and a fact whose
only approved role the admin had taken off default silently got it back.

The real model writer and date-role functions on a synthetic schema.
"""

from __future__ import annotations

import json

import pytest

from core.semantic_model import (
    clear_default_date_role,
    find_default_date_roles,
    load_semantic_model,
    patch_date_role,
    set_default_date_role,
    write_semantic_model,
)

FACT = "SALES.SALES_FACT"
SCHEMA = {
    "SYNDB.SALES.SALES_FACT": {"database": "SYNDB", "schema": "SALES", "table": "SALES_FACT", "columns": [
        {"name": "SALES_FACT_KEY", "type": "bigint"}, {"name": "ORD_DT_KEY", "type": "bigint"},
        {"name": "INV_DT_KEY", "type": "bigint"}, {"name": "NET_SLS_AMT", "type": "decimal"}]},
    "SYNDB.SALES.DT_DMS": {"database": "SYNDB", "schema": "SALES", "table": "DT_DMS", "columns": [
        {"name": "DT_DMS_KEY", "type": "bigint"}, {"name": "CAL_DT", "type": "date"},
        {"name": "YEAR", "type": "int"}, {"name": "MONTH", "type": "int"}]},
}


@pytest.fixture
def model_dirs(tmp_path):
    schema_dir, kb_dir = tmp_path / "schema", tmp_path / "kb"
    schema_dir.mkdir()
    (schema_dir / "_schema.json").write_text(json.dumps(SCHEMA))
    write_semantic_model(schema_dir=str(schema_dir), kb_dir=str(kb_dir))
    return str(schema_dir), str(kb_dir)


def _approve(kb_dir, *columns):
    for column in columns:
        patch_date_role(kb_dir=kb_dir, fact_table=FACT, fact_column=column, dimension_table="SALES.DT_DMS",
                        dimension_key="DT_DMS_KEY", business_role=column.lower(), date_value_column="CAL_DT",
                        status="approved")


def _defaults(kb_dir):
    return [role["fact_column"] for role in find_default_date_roles(model=load_semantic_model(kb_dir))]


def test_the_chosen_default_is_still_the_default_after_a_rebuild(model_dirs):
    schema_dir, kb_dir = model_dirs
    _approve(kb_dir, "ORD_DT_KEY", "INV_DT_KEY")
    assert set_default_date_role(kb_dir, FACT, "INV_DT_KEY")
    assert _defaults(kb_dir) == ["INV_DT_KEY"]
    write_semantic_model(schema_dir=schema_dir, kb_dir=kb_dir)
    assert _defaults(kb_dir) == ["INV_DT_KEY"]


def test_a_default_taken_away_stays_away_after_a_rebuild(model_dirs):
    schema_dir, kb_dir = model_dirs
    _approve(kb_dir, "ORD_DT_KEY")
    assert _defaults(kb_dir) == ["ORD_DT_KEY"]  # one approved role is the default by itself
    set_default_date_role(kb_dir, FACT, "ORD_DT_KEY")
    clear_default_date_role(kb_dir, FACT, "ORD_DT_KEY")
    assert _defaults(kb_dir) == []
    write_semantic_model(schema_dir=schema_dir, kb_dir=kb_dir)
    assert _defaults(kb_dir) == []


def test_a_rebuild_does_not_invent_a_default(model_dirs):
    schema_dir, kb_dir = model_dirs
    _approve(kb_dir, "ORD_DT_KEY", "INV_DT_KEY")
    write_semantic_model(schema_dir=schema_dir, kb_dir=kb_dir)
    assert _defaults(kb_dir) == []
