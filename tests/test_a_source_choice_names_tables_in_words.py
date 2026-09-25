# -*- coding: utf-8 -*-
"""A "which source?" card names each table in words, through the tenant's vocabulary.

A live test's source card offered two chips: "Item Balance by Period" and
"Itm Bal Dly". The first came from the ERP pack's table dictionary. The second
was the semantic model's entity for ITM_BAL_DLY_FCT, which is the table's own
name cut at the underscores, and nothing read it through a vocabulary on the
way to the reader. So a code an admin had confirmed on the Business Meanings
page never reached that card either.

Now the label is the table's name read through the tenant's vocabulary: known
codes are words ("Item Balance"), a code with two readings stays as written
until an admin confirms one ("Dly" -- daily or delivery), and a confirmed
reading joins the words ("Item Balance Daily"). A pack's own label still wins.

Driven through resolve_source_scope and source_clarification_options over a
semantic model built from a schema file, with the tenant's vocabulary built by
vocab_for_account from the store. A synthetic mart; no customer data.
"""

from __future__ import annotations

import json
import os

import pytest

# The store is conftest's scratch database for the run (tests/conftest.py);
# nothing here repoints or re-imports it, so modules imported before this one
# keep the same store object as every test after it.
import store  # noqa: E402
from core.semantic_model import build_semantic_model  # noqa: E402
from core.source_resolution import resolve_source_scope, source_clarification_options  # noqa: E402
from core.vocab_packs import vocab_for_account  # noqa: E402


def _table(own_key: str, *columns: tuple[str, str]) -> dict:
    return {
        "columns": [{"name": name, "type": dtype, "nullable": True, "comment": ""} for name, dtype in columns],
        "pk_columns": [own_key], "row_count": 1000, "comment": "", "schema": "MART", "database": "WH",
    }


SCHEMA = {
    "WH.MART.ITM_BAL_DLY_FCT": _table(
        "ITM_BAL_DLY_FCT_KEY", ("ITM_BAL_DLY_FCT_KEY", "bigint"), ("ITM_DMS_KEY", "int"),
        ("ITM_BAL_EFC_DT_DMS_KEY", "int"), ("ON_HND_QTY", "decimal(18,4)"),
    ),
    "WH.MART.ITM_BAL_PRD_FCT": _table(
        "ITM_BAL_PRD_FCT_KEY", ("ITM_BAL_PRD_FCT_KEY", "bigint"), ("ITM_DMS_KEY", "int"),
        ("PRD_DMS_KEY", "int"), ("CUR_ON_HND_QTY", "decimal(18,4)"),
    ),
    # A fact whose name spells its cadence out.
    "WH.MART.SLS_DAILY_FCT": _table(
        "SLS_DAILY_FCT_KEY", ("SLS_DAILY_FCT_KEY", "bigint"), ("ITM_DMS_KEY", "int"),
        ("SLS_DT_DMS_KEY", "int"), ("NET_SLS_AMT", "decimal(18,2)"),
    ),
    "WH.MART.ITM_DMS": _table("ITM_DMS_KEY", ("ITM_DMS_KEY", "int"), ("ITM_DSC", "varchar(60)")),
    "WH.MART.DT_DMS": _table("DT_DMS_KEY", ("DT_DMS_KEY", "int"), ("DMS_DT", "date")),
    "WH.MART.PRD_DMS": _table("PRD_DMS_KEY", ("PRD_DMS_KEY", "int"), ("PRD_DSC", "varchar(40)")),
}


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    directory = tmp_path_factory.mktemp("source_labels")
    (directory / "_schema.json").write_text(json.dumps(SCHEMA), encoding="utf-8")
    return build_semantic_model(str(directory))


def _account(**meta) -> str:
    store.init_db()
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    if meta:
        store.update_client_meta(account_id, **meta)
    return account_id


# Both balance facts answer it, so the reader is asked which.
QUESTION = "on hand quantity by item"


def _scope(model, account_id: str) -> dict:
    return resolve_source_scope(QUESTION, model, vocab=vocab_for_account(account_id))


def _labels(model, account_id: str) -> dict[str, str]:
    return {str(c["table"]).split(".")[-1]: c["label"] for c in _scope(model, account_id)["candidates"]}


class TestTheCardNamesTablesInWords:

    def test_known_codes_are_words(self, model):
        assert _labels(model, _account())["ITM_BAL_PRD_FCT"] == "Item Balance Period"

    def test_a_code_with_two_readings_stays_as_written(self, model):
        assert _labels(model, _account())["ITM_BAL_DLY_FCT"] == "Item Balance Dly"

    def test_a_confirmed_meaning_reaches_the_card(self, model):
        account_id = _account()
        store.save_business_meanings(account_id, [{
            "scope": "code", "subject": "DLY", "reading": "daily", "rule": "grain",
            "confidence": 85, "evidence": ["ITM_BAL_DLY_FCT: the word before FCT says what one row covers"],
            "where": ["ITM_BAL_DLY_FCT"],
        }])
        assert store.decide_business_meaning(account_id, "code", "DLY", "confirmed")
        labels = _labels(model, account_id)
        assert labels["ITM_BAL_DLY_FCT"] == "Item Balance Daily"

    def test_the_reader_is_asked_with_those_words(self, model):
        scope = _scope(model, _account())
        assert scope["status"] == "ambiguous"
        chips = {option["value"].split(".")[-1]: option["label"] for option in source_clarification_options(scope)}
        assert chips == {"ITM_BAL_DLY_FCT": "Item Balance Dly", "ITM_BAL_PRD_FCT": "Item Balance Period"}


    def test_a_cadence_the_name_already_says_is_not_said_twice(self, model):
        """SLS_DAILY_FCT: the grain says daily and so does the name. The card
        said "Daily Sls Daily"."""
        scope = resolve_source_scope("net sls amt by item", model, vocab=vocab_for_account(_account()))
        label = next(c["label"] for c in scope["candidates"] if str(c["table"]).endswith("SLS_DAILY_FCT"))
        assert label.split().count("Daily") == 1, label


    def test_a_name_that_starts_with_its_kind_drops_it(self):
        """FACT_PURCHASE_RECEIPT reads "Purchase Receipt", as its entity did."""
        from core.source_resolution import _business_source_label

        table = {"qualified_name": "MART.FACT_PURCHASE_RECEIPT", "table": "FACT_PURCHASE_RECEIPT",
                 "entity": "Purchase Receipt", "type": "fact"}
        assert _business_source_label(table, vocab=vocab_for_account(_account())) == "Purchase Receipt"


class TestWhatDidNotChange:

    def test_a_packs_label_still_wins(self, model):
        labels = _labels(model, _account(erp_packs=json.dumps(["infor_m3"])))
        assert labels["ITM_BAL_PRD_FCT"] == "Item Balance by Period"
