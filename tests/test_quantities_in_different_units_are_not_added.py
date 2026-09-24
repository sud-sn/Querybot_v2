"""
Quantities counted in different units of measure are never added together.

A stock or sales fact holds each item's quantity in that item's unit: most in
eaches (EA), some in feet (FT), metres (ME), rolls (RL). "Total stock on hand"
summed across items added eaches to feet to metres -- a number that measures
nothing, from SQL that looked entirely ordinary.

A total of a quantity is now kept per unit: the query groups by the unit of
measure, keeps to one unit, or stays within one item. The rule rides on the
semantic plan: the prompt states it, the validator refuses a total that mixes
units -- naming the item's unit and the join to it, since a fact's own copy can
be blank on most rows -- and the answer card says the totals are per unit. A
value, a quantity times a cost, is currency and still adds up. A question
that asks for a total across all units gets one.

Synthetic tables in a mart's naming convention; no customer data, no real
database. `store` is imported where it is used.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from unittest.mock import patch

import pytest

from core.failure_messages import translate_failure
from core.i18n import activate_language, deactivate_language
from core.llm import build_sql_system_prompt
from core.result_renderer import _send_results
from core.units_of_measure import (
    attach_unit_policies,
    is_quantity_column,
    is_unit_column,
    question_asks_across_units,
    unit_policies,
)
from core.validator import REPAIRABLE_REASON_CODES, validate_sql_detailed


def _table(own_key: str, *columns: tuple[str, str]) -> dict:
    return {
        "columns": [
            {"name": name, "type": dtype, "nullable": False, "comment": ""}
            for name, dtype in columns
        ],
        "pk_columns": [own_key],
        "row_count": 1000,
        "comment": "",
        "schema": "MART",
        "database": "WH",
    }


SCHEMA = {
    # Its own unit column, and the item's.
    "WH.MART.ITM_BAL_DLY_FCT": _table(
        "ITM_BAL_DLY_FCT_KEY", ("ITM_BAL_DLY_FCT_KEY", "bigint"), ("ITM_DMS_KEY", "int"),
        ("WHS_DMS_KEY", "int"), ("UNT_OF_MSR", "nvarchar"), ("ON_HND_QTY", "decimal(18,4)"),
        ("ALC_QTY", "decimal(18,4)"), ("ITM_CST", "decimal(18,4)"),
    ),
    # No unit column of its own: the item's is the one.
    "WH.MART.SLS_TRX_FCT": _table(
        "SLS_TRX_FCT_KEY", ("SLS_TRX_FCT_KEY", "bigint"), ("ITM_DMS_KEY", "int"),
        ("WHS_DMS_KEY", "int"), ("SLS_QTY", "decimal(18,4)"),
    ),
    # Quantities, but nothing says in which unit.
    "WH.MART.RCT_TRX_FCT": _table(
        "RCT_TRX_FCT_KEY", ("RCT_TRX_FCT_KEY", "bigint"), ("WHS_DMS_KEY", "int"),
        ("RCT_QTY", "decimal(18,4)"),
    ),
    "WH.MART.ITM_DMS": _table(
        "ITM_DMS_KEY", ("ITM_DMS_KEY", "int"), ("ITM_CD", "nvarchar"), ("ITM_NM", "nvarchar"),
        ("ITM_TYP_CD", "nvarchar"), ("UNT_OF_MSR", "nvarchar"),
    ),
    "WH.MART.WHS_DMS": _table("WHS_DMS_KEY", ("WHS_DMS_KEY", "int"), ("WHS_DSC", "nvarchar")),
}
KNOWN_TABLES = {fqn.split(".", 1)[1] for fqn in SCHEMA}
TABLE_COLUMNS = {
    fqn.split(".", 1)[1]: {column["name"]: column["type"] for column in meta["columns"]}
    for fqn, meta in SCHEMA.items()
}


@pytest.fixture
def account(tmp_path):
    import store
    from core.graph_autopopulate import auto_populate_from_schema

    store.init_db()
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    (tmp_path / "_schema.json").write_text(json.dumps(SCHEMA), encoding="utf-8")
    store.update_client_state(account_id, "READY", {"schema_dir": str(tmp_path)})
    auto_populate_from_schema(account_id, str(tmp_path))
    store.save_metric(account_id, {
        "name": "Sales Quantity", "formula_type": "expression", "sql_template": "SUM(SLS_QTY)",
        "base_table": "MART.SLS_TRX_FCT", "synonyms": "sales quantity, units sold",
        "description": "Units sold",
    })
    return account_id


@pytest.fixture
def policies(account):
    return unit_policies(account)


def _validate(sql: str, policies: list[dict], db_type: str = "azure_sql"):
    return validate_sql_detailed(
        sql, KNOWN_TABLES, db_type, None, TABLE_COLUMNS,
        {"semantic_plan": {"unit_policies": policies}},
    )


class TestWhatIsAQuantityAndWhatIsAUnit:

    @pytest.mark.parametrize("column, unit", [
        ("UNT_OF_MSR", True), ("UOM", True), ("BASE_UOM", True), ("UNIT_OF_MEASURE", True),
        ("UNIT_PRICE", False), ("UNT_CST", False), ("MSR_DT", False),
    ])
    def test_a_unit_of_measure(self, column, unit):
        assert is_unit_column(column) is unit

    @pytest.mark.parametrize("column, quantity", [
        ("ON_HND_QTY", True), ("QTY_ON_HAND", True), ("ORDER_QUANTITY", True),
        ("ON_HND_QTY_FLG", False), ("ITM_CST", False), ("UNT_OF_MSR", False),
    ])
    def test_a_quantity(self, column, quantity):
        assert is_quantity_column(column) is quantity


class TestThePolicyComesFromTheSchema:

    def test_a_fact_with_its_own_unit_and_the_items(self, policies):
        daily = next(p for p in policies if p["fact_table"] == "MART.ITM_BAL_DLY_FCT")
        assert daily["quantities"] == ["ON_HND_QTY", "ALC_QTY"]
        assert daily["unit_columns"] == [
            {"table": "MART.ITM_BAL_DLY_FCT", "column": "UNT_OF_MSR"},
            {"table": "MART.ITM_DMS", "column": "UNT_OF_MSR"},
        ]
        assert {(i["table"], i["column"]) for i in daily["item_columns"]} == {
            ("MART.ITM_BAL_DLY_FCT", "ITM_DMS_KEY"), ("MART.ITM_DMS", "ITM_DMS_KEY"),
            ("MART.ITM_DMS", "ITM_CD"), ("MART.ITM_DMS", "ITM_NM"),
        }

    def test_a_fact_whose_unit_is_only_the_items(self, policies):
        sales = next(p for p in policies if p["fact_table"] == "MART.SLS_TRX_FCT")
        assert sales["unit_columns"] == [{"table": "MART.ITM_DMS", "column": "UNT_OF_MSR"}]
        assert sales["unit_joins"] == [
            {"table": "MART.ITM_DMS", "fact_column": "ITM_DMS_KEY", "key": "ITM_DMS_KEY"},
        ]

    def test_no_policy_where_nothing_says_the_unit(self, policies):
        assert "MART.RCT_TRX_FCT" not in {p["fact_table"] for p in policies}

    @pytest.mark.parametrize("question", [
        "total stock across all units", "total stock regardless of unit",
        "stock total toutes unités confondues",
    ])
    def test_a_question_for_a_total_across_units_gets_one(self, account, question):
        plan: dict = {}
        assert question_asks_across_units(question)
        assert attach_unit_policies(plan, account, question) == []
        assert "unit_policies" not in plan

    def test_any_other_question_carries_the_rule(self, account):
        plan: dict = {}
        attach_unit_policies(plan, account, "stock on hand by warehouse")
        assert {p["fact_table"] for p in plan["unit_policies"]} == {
            "MART.ITM_BAL_DLY_FCT", "MART.SLS_TRX_FCT"}


DAILY = "FROM MART.ITM_BAL_DLY_FCT f JOIN MART.WHS_DMS w ON f.WHS_DMS_KEY = w.WHS_DMS_KEY"
SALES = "FROM MART.SLS_TRX_FCT s JOIN MART.WHS_DMS w ON s.WHS_DMS_KEY = w.WHS_DMS_KEY"

REFUSED = {
    "a total by warehouse": ("azure_sql",
        f"SELECT w.WHS_DSC, SUM(f.ON_HND_QTY) AS OH {DAILY} GROUP BY w.WHS_DSC"),
    "an average by warehouse": ("snowflake",
        f"SELECT w.WHS_DSC, AVG(f.ON_HND_QTY) AS OH {DAILY} GROUP BY 1"),
    "a difference of quantities": ("oracle",
        f"SELECT w.WHS_DSC, SUM(f.ON_HND_QTY - f.ALC_QTY) AS FREE_QTY {DAILY} GROUP BY w.WHS_DSC"),
    "a total inside a CTE": ("azure_sql",
        f"WITH t AS (SELECT w.WHS_DSC, SUM(f.ON_HND_QTY) AS OH {DAILY} GROUP BY w.WHS_DSC) "
        "SELECT WHS_DSC, OH FROM t"),
    "sales by warehouse": ("azure_sql",
        f"SELECT w.WHS_DSC, SUM(s.SLS_QTY) AS QTY {SALES} GROUP BY w.WHS_DSC"),
    "two units kept": ("azure_sql",
        f"SELECT w.WHS_DSC, SUM(f.ON_HND_QTY) AS OH {DAILY} WHERE f.UNT_OF_MSR IN ('EA', 'FT') "
        "GROUP BY w.WHS_DSC"),
    "grouped by the item's type, not the item": ("azure_sql",
        "SELECT i.ITM_TYP_CD, SUM(f.ON_HND_QTY) AS OH FROM MART.ITM_BAL_DLY_FCT f "
        "JOIN MART.ITM_DMS i ON f.ITM_DMS_KEY = i.ITM_DMS_KEY GROUP BY i.ITM_TYP_CD"),
}

ACCEPTED = {
    "per unit, the fact's own column": ("azure_sql",
        f"SELECT w.WHS_DSC, f.UNT_OF_MSR, SUM(f.ON_HND_QTY) AS OH {DAILY} "
        "GROUP BY w.WHS_DSC, f.UNT_OF_MSR"),
    "per unit, the item's column": ("snowflake",
        f"SELECT w.WHS_DSC, i.UNT_OF_MSR, SUM(s.SLS_QTY) AS QTY {SALES} "
        "JOIN MART.ITM_DMS i ON s.ITM_DMS_KEY = i.ITM_DMS_KEY GROUP BY 1, 2"),
    "per item": ("azure_sql",
        "SELECT i.ITM_CD, SUM(f.ON_HND_QTY) AS OH FROM MART.ITM_BAL_DLY_FCT f "
        "JOIN MART.ITM_DMS i ON f.ITM_DMS_KEY = i.ITM_DMS_KEY GROUP BY i.ITM_CD"),
    "one unit": ("azure_sql",
        f"SELECT w.WHS_DSC, SUM(f.ON_HND_QTY) AS OH {DAILY} WHERE f.UNT_OF_MSR = 'EA' "
        "GROUP BY w.WHS_DSC"),
    "one item": ("azure_sql",
        "SELECT w.WHS_DSC, SUM(f.ON_HND_QTY) AS OH FROM MART.ITM_BAL_DLY_FCT f "
        "JOIN MART.WHS_DMS w ON f.WHS_DMS_KEY = w.WHS_DMS_KEY "
        "JOIN MART.ITM_DMS i ON f.ITM_DMS_KEY = i.ITM_DMS_KEY WHERE i.ITM_CD = 'A100' "
        "GROUP BY w.WHS_DSC"),
    "a value, not a quantity": ("azure_sql",
        f"SELECT w.WHS_DSC, SUM(f.ON_HND_QTY * f.ITM_CST) AS VAL {DAILY} GROUP BY w.WHS_DSC"),
    "a count of rows with stock": ("azure_sql",
        f"SELECT w.WHS_DSC, COUNT(f.ON_HND_QTY) AS N {DAILY} GROUP BY w.WHS_DSC"),
    "a fact nothing says the unit of": ("azure_sql",
        "SELECT w.WHS_DSC, SUM(r.RCT_QTY) AS RCT FROM MART.RCT_TRX_FCT r "
        "JOIN MART.WHS_DMS w ON r.WHS_DMS_KEY = w.WHS_DMS_KEY GROUP BY w.WHS_DSC"),
}


class TestTheValidatorRefusesAMixedTotal:

    @pytest.mark.parametrize("label", sorted(REFUSED))
    def test_refused(self, policies, label):
        db_type, sql = REFUSED[label]
        result = _validate(sql, policies, db_type)
        assert (result.ok, result.code) == (False, "units_mixed"), result.reason

    @pytest.mark.parametrize("label", sorted(ACCEPTED))
    def test_accepted(self, policies, label):
        db_type, sql = ACCEPTED[label]
        result = _validate(sql, policies, db_type)
        assert result.ok, (result.code, result.reason)

    def test_the_fix_names_the_items_unit_and_the_join_to_it(self, policies):
        result = _validate(REFUSED["sales by warehouse"][1], policies)
        assert result.errors[0]["required_column"] == (
            "ITM_DMS.UNT_OF_MSR (JOIN MART.ITM_DMS AS ITM_DMS ON s.ITM_DMS_KEY = ITM_DMS.ITM_DMS_KEY)")
        assert "units_mixed" in REPAIRABLE_REASON_CODES

    def test_the_items_unit_comes_before_the_facts_own_copy(self, policies):
        result = _validate(REFUSED["a total by warehouse"][1], policies)
        assert result.errors[0]["required_column"] == (
            "ITM_DMS.UNT_OF_MSR (JOIN MART.ITM_DMS AS ITM_DMS ON f.ITM_DMS_KEY = ITM_DMS.ITM_DMS_KEY)")

    def test_the_item_already_joined_is_used_as_it_is(self, policies):
        result = _validate(REFUSED["grouped by the item's type, not the item"][1], policies)
        assert result.errors[0]["required_column"] == "i.UNT_OF_MSR"

    def test_without_the_rule_the_same_total_passes(self):
        assert _validate(REFUSED["a total by warehouse"][1], []).ok


class TestThePromptStatesTheRule:

    def _prompt(self, policies, context):
        return build_sql_system_prompt(
            "azure_sql", context, semantic_plan={"unit_policies": policies},
            question="stock on hand by warehouse",
        )

    def test_it_is_stated_for_a_fact_in_context(self, policies):
        prompt = self._prompt(policies, "## Table: MART.ITM_BAL_DLY_FCT\n| `ON_HND_QTY` | decimal |")
        block = prompt[prompt.index("## Units of measure"):]
        assert "MART.ITM_BAL_DLY_FCT: ON_HND_QTY, ALC_QTY are in each item's unit (ITM_DMS.UNT_OF_MSR" in block
        assert "SLS_TRX_FCT" not in block

    def test_it_is_left_out_when_no_such_fact_is(self, policies):
        assert "## Units of measure" not in self._prompt(policies, "## Table: MART.WHS_DMS")


class _Adapter:
    def __init__(self):
        self.sent: list[str] = []

    async def send_message(self, event, text):
        self.sent.append(text)


def _card(policies, sql: str, lang: str = "en") -> str:
    adapter = _Adapter()
    token = activate_language(lang)
    try:
        asyncio.run(_send_results(
            {}, adapter, "stock on hand by warehouse",
            [{"WHS_DSC": "North", "UNT_OF_MSR": "EA", "OH": 135.0}],
            sql, 900, None, 1, {"id": 1, "db_type": "azure_sql"},
            question_id=None, confidence_context={"semantic_plan": {"unit_policies": policies}},
        ))
    finally:
        deactivate_language(token)
    return "\n".join(adapter.sent)


class TestTheReaderIsTold:

    @pytest.mark.parametrize("lang, phrase", [
        ("en", "totalled per unit of measure"), ("fr", "totalisées par unité de mesure"),
    ])
    def test_the_card_says_the_totals_are_per_unit(self, policies, lang, phrase):
        assert phrase in _card(policies, ACCEPTED["per unit, the fact's own column"][1], lang)

    def test_a_value_total_says_nothing(self, policies):
        assert "per unit of measure" not in _card(policies, ACCEPTED["a value, not a quantity"][1])

    @pytest.mark.parametrize("lang, phrase", [("en", "different units"), ("fr", "unités de mesure différentes")])
    def test_a_refusal_that_survived_every_repair_explains_itself(self, lang, phrase):
        token = activate_language(lang)
        try:
            card = translate_failure(kind="validation", code="units_mixed")
        finally:
            deactivate_language(token)
        assert phrase in card["most_likely_reason"]
        assert card["suggested_next_step"]


class TestTheRealPipeline:
    """_handle_query_impl, end to end with the model and the warehouse stubbed:
    the model's total across units is refused and repaired, and only the
    per-unit total reaches the warehouse -- unless the question asked for a
    total across units."""

    FIRST = f"SELECT w.WHS_DSC, SUM(s.SLS_QTY) AS QTY {SALES} GROUP BY w.WHS_DSC"
    REPAIRED = (f"SELECT w.WHS_DSC, i.UNT_OF_MSR, SUM(s.SLS_QTY) AS QTY {SALES} "
                "JOIN MART.ITM_DMS i ON s.ITM_DMS_KEY = i.ITM_DMS_KEY GROUP BY w.WHS_DSC, i.UNT_OF_MSR")

    def _run(self, account, tmp_path, question):
        import core.query_pipeline as qp
        from gateway.base import PlatformEvent

        seen: dict = {"prompts": [], "executed": [], "card": None}

        class Retriever:
            last_retrieval_weak = False
            last_retrieval_unscored = False

            def retrieve(self, question, n=8, allowed_tables=None):
                return [
                    "# MART.SLS_TRX_FCT\n| `WHS_DMS_KEY` | int |\n| `SLS_QTY` | decimal |",
                    "# MART.WHS_DMS\n| `WHS_DMS_KEY` | int |\n| `WHS_DSC` | nvarchar |",
                ]

            def retrieve_fact_patterns(self, question, n=2, allowed_tables=None):
                return []

            def _is_global(self, doc):
                return False

        async def model_reply(system, user, *args, **kwargs):
            seen["prompts"].append((system, user))
            if "UNIT-OF-MEASURE REPAIR REQUIRED" in user:
                return self.REPAIRED, 1, 1
            return self.FIRST, 1, 1

        class Governed:
            def __init__(self, sql):
                self.sql, self.rows = sql, [{"WHS_DSC": "North", "QTY": 135.0}]

        def execute(credentials, db_type, sql, **kwargs):
            seen["executed"].append(sql)
            return Governed(sql)

        async def card(event, adapter, question, rows, sql, *args, **kwargs):
            seen["card"] = (sql, (kwargs.get("confidence_context") or {}).get("semantic_plan"))

        class Adapter:
            platform, session_id, thread_id, last_result_id = "portal", f"{account}:portal:u1", "t1", None

            async def send_message(self, event, text, **kwargs):
                return None

            async def send_typing(self, *args, **kwargs):
                return None

            def add_to_history(self, **kwargs):
                return None

            def cache_result(self, *args, **kwargs):
                return None

        with contextlib.ExitStack() as stack:
            for mock in (
                patch.object(qp, "get_state", return_value={
                    "state": "READY", "schema_dir": str(tmp_path), "kb_dir": str(tmp_path)}),
                patch.object(qp, "get_client_db", return_value={
                    "db_type": "azure_sql", "credentials": {}, "name": "db", "id": 1}),
                patch.object(qp.store, "get_client", return_value={
                    "account_id": account, "state": "READY", "name": "T"}),
                patch.object(qp, "load_known_tables", return_value=set(KNOWN_TABLES)),
                patch.object(qp, "load_schema_columns", return_value=TABLE_COLUMNS),
                patch.object(qp, "load_retriever", return_value=Retriever()),
                patch.object(qp, "llm_complete", model_reply),
                patch.object(qp, "_send_results", card),
                patch.object(qp, "execute_governed_query", execute),
                patch.object(qp, "resolve_provider",
                             return_value=("anthropic", "claude", "key", {})),
                patch.object(qp, "_log_q", lambda *args, **kwargs: None),
                patch.object(qp, "retrieve_similar_examples", return_value=[]),
            ):
                stack.enter_context(mock)
            asyncio.run(qp._handle_query_impl(
                account, PlatformEvent(account, "u1", "c1", question, "portal"),
                Adapter(), question,
                {"id": 1, "role": "admin", "email": "u@x.com", "name": "U", "group_name": None},
            ))
        return seen

    def test_a_total_by_warehouse_is_kept_per_unit(self, account, tmp_path):
        seen = self._run(account, tmp_path, "sales quantity by warehouse")
        assert any("## Units of measure" in system for system, _user in seen["prompts"])
        assert any("UNIT-OF-MEASURE REPAIR REQUIRED" in user for _system, user in seen["prompts"])
        assert seen["executed"] == [self.REPAIRED]
        sql, plan = seen["card"]
        assert sql == self.REPAIRED and plan["unit_policies"]

    def test_a_total_across_all_units_is_given_as_asked(self, account, tmp_path):
        seen = self._run(account, tmp_path, "sales quantity by warehouse across all units")
        assert not any("## Units of measure" in system for system, _user in seen["prompts"])
        assert seen["executed"] == [self.FIRST]
