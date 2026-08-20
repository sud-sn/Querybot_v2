"""
tests/test_contract_model_fallback.py

The query pipeline reads runtime semantics from the compiled contract's model
section rather than from the file, so approved meanings come from one versioned
artifact. The three readers took that section as authoritative whenever it was
`is not None` — which included the case where it was an empty dict.

An empty model section is not a statement that the workspace has no semantics.
_compile_contract_internal compiles every source fail-soft: an unreadable model
file, or kb_dir unset at compile time, yields `model: {}` and the contract is
published regardless. From that moment every question was planned as if no
field had ever been approved — build_runtime_semantic_plan returning
`enabled: False`, no field bindings, no join requirements, no date roles —
while the admin console still listed every approval as saved. Nothing failed;
answers just started choosing between rival columns out of raw prose again.

These tests drive the three readers with a real model file on disk, because the
defect is entirely about which of two sources is consulted.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import pytest

from core.semantic_model import (
    MODEL_JSON,
    build_runtime_semantic_context,
    build_runtime_semantic_plan,
    find_default_date_roles,
)

QUESTION = "net revenue by warehouse"

MODEL = {
    "tables": [
        {
            "table": "F_SALES",
            "qualified_name": "DW.F_SALES",
            "schema": "DW",
            "measures": [
                {"name": "net revenue", "column": "NET_AMOUNT", "status": "approved"},
            ],
            "dimensions": [
                {
                    "name": "Warehouse",
                    "source_key": "WHS_KEY",
                    "display_column": "WHS_NAME",
                    "display_table": "DW.D_WAREHOUSE",
                    "status": "approved",
                },
            ],
        },
        {"table": "D_WAREHOUSE", "qualified_name": "DW.D_WAREHOUSE", "schema": "DW"},
    ],
    "date_roles": [
        {
            "fact_table": "DW.F_SALES",
            "fact_column": "INVOICE_DATE",
            "date_column": "INVOICE_DATE",
            "status": "approved",
            "business_role": "invoice date",
        },
    ],
}


@pytest.fixture
def kb_dir():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / MODEL_JSON).write_text(json.dumps(MODEL), encoding="utf-8")
        yield tmp


def _readings(kb, model):
    return {
        "context": build_runtime_semantic_context(kb, question=QUESTION, model=model),
        "plan": build_runtime_semantic_plan(kb, question=QUESTION, model=model),
        "roles": find_default_date_roles(kb, model=model),
    }


def test_a_populated_contract_model_is_still_authoritative(kb_dir):
    """The override is the whole point of passing the section — a contract
    that HAS a model must be used in preference to the file."""
    only_dates = {"tables": [], "date_roles": MODEL["date_roles"]}
    plan = build_runtime_semantic_plan(kb_dir, question=QUESTION, model=only_dates)
    assert plan.get("enabled") is False, (
        "the file was read even though the contract carried its own model"
    )


def test_an_empty_contract_model_falls_back_to_the_file(kb_dir):
    """The defect: `{}` was accepted as authoritative, so every approved
    meaning vanished at once."""
    empty = _readings(kb_dir, {})
    assert empty["plan"].get("enabled") is True, (
        "an empty contract model disabled the deterministic semantic plan, "
        "which is every approved field meaning gone in one step"
    )
    assert empty["plan"].get("fields"), "no field bindings survived"
    assert empty["roles"], "no approved date role survived"
    assert empty["context"], "no model context reached the prompt"


def test_an_empty_section_reads_identically_to_no_section_at_all(kb_dir):
    """`model=None` (no contract compiled) always fell back. `model={}` is the
    same statement — nothing to override with — and must behave the same."""
    assert _readings(kb_dir, {}) == _readings(kb_dir, None)


def test_the_fallback_is_reported_at_warning_level(kb_dir, caplog):
    """A contract that lost its model section is a degraded state someone has
    to fix. Recovering silently is how it would be discovered from wrong
    answers weeks later instead."""
    with caplog.at_level(logging.WARNING, logger="core.semantic_model"):
        build_runtime_semantic_plan(kb_dir, question=QUESTION, model={})
    assert any(
        "empty model section" in record.getMessage()
        for record in caplog.records
    ), "the fallback happened silently"


def test_no_model_anywhere_still_degrades_quietly():
    """With neither a contract model nor a file there is genuinely nothing to
    report — this must stay a normal empty result, not a warning."""
    with tempfile.TemporaryDirectory() as tmp:
        assert build_runtime_semantic_plan(tmp, question=QUESTION, model={}).get("enabled") is False
        assert find_default_date_roles(tmp, model={}) == []
        assert build_runtime_semantic_context(tmp, question=QUESTION, model={}) == ""


def test_all_three_readers_share_one_resolution_rule():
    """Three copies of this decision is how one of them would drift back."""
    import inspect

    import core.semantic_model as sm

    for func in (build_runtime_semantic_context, build_runtime_semantic_plan,
                 find_default_date_roles):
        source = inspect.getsource(func)
        assert "_model_source(model, kb_dir)" in source, (
            f"{func.__name__} resolves the model source on its own"
        )
    assert "is not None else load_semantic_model" not in inspect.getsource(sm), (
        "a reader still treats an empty contract model as authoritative"
    )
