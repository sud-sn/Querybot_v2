"""
A placeholder member is named for what it is in an answer.

A listing keeps a dimension's placeholder members -- their rows are real
stock, only their warehouse or buyer was never known -- and the reader saw
them as the warehouse stores them: "NULL value provided", NO_MATCH, a key of
0. The answer card now says "Not specified", "Unmatched" or "Unknown" (in
French too), in the table, the chart, the narrative and the card's own
follow-ups, and says once what those names stand for. Only a text that reads
as a placeholder is renamed, never a number that sits in a placeholder row; a
key only in a column named for it. The copy kept for authorized export keeps
the warehouse's values.

A question ABOUT them ("stock with an unspecified buyer") gets their keys in
the prompt, since the name is not in the data.

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

from core.i18n import activate_language, deactivate_language
from core.llm import build_sql_system_prompt
from core.result_renderer import _send_results
from core.unknown_members import attach_unknown_member_policies, label_unknown_members, member_labels


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
    "WH.MART.ITM_BAL_DLY_FCT": _table(
        "ITM_BAL_DLY_FCT_KEY", ("ITM_BAL_DLY_FCT_KEY", "bigint"), ("WHS_DMS_KEY", "int"),
        ("BYR_PTY_DMS_KEY", "int"), ("ON_HND_QTY", "decimal(18,4)"),
    ),
    "WH.MART.WHS_DMS": _table(
        "WHS_DMS_KEY", ("WHS_DMS_KEY", "int"), ("WHS_CD", "nvarchar"), ("WHS_DSC", "nvarchar"),
        ("SQ_FT", "nvarchar"),
    ),
    "WH.MART.PTY_DMS": _table(
        "PTY_DMS_KEY", ("PTY_DMS_KEY", "int"), ("PTY_CD", "nvarchar"), ("PTY_NM", "nvarchar"),
    ),
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
    # The warehouse's own placeholders -- with a size of "0" in the row, as
    # a text column can hold.
    store.save_unknown_members(account_id, "WHS_DMS", "WHS_DMS_KEY", [
        {"key_value": "0", "kind": "not_specified",
         "member_text": {"WHS_CD": "NO_VALUE", "WHS_DSC": "NULL value provided", "SQ_FT": "0"}},
        {"key_value": "777", "kind": "unmatched",
         "member_text": {"WHS_CD": "NO_MATCH", "WHS_DSC": "Value provided does not match"}},
    ])
    store.save_unknown_members(account_id, "PTY_DMS", "PTY_DMS_KEY", [
        {"key_value": "-1", "kind": "unknown", "member_text": {"PTY_CD": "UNK", "PTY_NM": "Unknown"}},
    ])
    store.save_metric(account_id, {
        "name": "Stock On Hand", "formula_type": "expression", "sql_template": "SUM(ON_HND_QTY)",
        "base_table": "MART.ITM_BAL_DLY_FCT", "synonyms": "stock on hand, stock",
        "description": "Units on hand",
    })
    return account_id


def _names(kind: str) -> str:
    return {"not_specified": "Not specified", "unmatched": "Unmatched", "unknown": "Unknown"}[kind]


class TestWhatIsRenamed:

    def test_the_warehouses_texts_become_names(self, account):
        rows, changed = label_unknown_members([
            {"WHS_DSC": "NULL value provided", "OH": 12.0},
            {"WHS_DSC": "Value provided does not match", "OH": 3.0},
            {"WHS_DSC": "North depot", "OH": 5.0},
            {"PTY_NM": "unknown ", "OH": 1.0},
        ], member_labels(account), _names)
        assert [row.get("WHS_DSC") or row.get("PTY_NM") for row in rows] == [
            "Not specified", "Unmatched", "North depot", "Unknown"]
        assert changed == 3

    def test_a_key_is_named_in_its_own_column_and_in_one_that_points_at_it(self, account):
        rows, _ = label_unknown_members(
            [{"WHS_DMS_KEY": 0, "BYR_PTY_DMS_KEY": -1, "OH": 0}], member_labels(account), _names,
        )
        assert rows == [{"WHS_DMS_KEY": "Not specified", "BYR_PTY_DMS_KEY": "Unknown", "OH": 0}]

    def test_a_number_in_a_placeholder_row_is_left_alone(self, account):
        rows, changed = label_unknown_members(
            [{"SQ_FT": "0", "WHS_CD": "0", "ITEMS": 0, "FLAG": False}], member_labels(account), _names,
        )
        assert rows == [{"SQ_FT": "0", "WHS_CD": "0", "ITEMS": 0, "FLAG": False}] and changed == 0


class _Adapter:
    def __init__(self):
        self.sent: list[str] = []
        self.cached: list = []

    async def send_message(self, event, text):
        self.sent.append(text)

    def cache_result(self, rows, *args, **kwargs):
        self.cached.append(rows)


ROWS = [{"WHS_DSC": "NULL value provided", "OH": 12.0}, {"WHS_DSC": "North depot", "OH": 5.0}]
SQL = ("SELECT w.WHS_DSC, SUM(f.ON_HND_QTY) AS OH FROM MART.ITM_BAL_DLY_FCT f "
       "JOIN MART.WHS_DMS w ON f.WHS_DMS_KEY = w.WHS_DMS_KEY GROUP BY w.WHS_DSC")


def _card(account, rows, lang: str = "en", **kwargs) -> _Adapter:
    adapter = _Adapter()
    token = activate_language(lang)
    try:
        asyncio.run(_send_results(
            {}, adapter, "stock by warehouse", rows, SQL, 900, None, account,
            {"id": 1, "db_type": "azure_sql"}, confidence_context={}, **kwargs,
        ))
    finally:
        deactivate_language(token)
    return adapter


class TestTheCardNamesThem:

    @pytest.mark.parametrize("lang, name, note", [
        ("en", "Not specified", "placeholders for a value that was empty"),
        ("fr", "Non renseigné", "membres de remplacement de l'entrepôt"),
    ])
    def test_the_table_and_the_note_say_what_it_is(self, account, lang, name, note):
        text = "\n".join(_card(account, ROWS, lang).sent)
        assert name in text and note in text
        assert "NULL value provided" not in text

    def test_the_cards_own_follow_ups_see_the_name(self, account):
        adapter = _card(account, ROWS)
        assert [row["WHS_DSC"] for row in adapter.cached[0]] == ["Not specified", "North depot"]

    def test_the_copy_kept_for_export_keeps_the_warehouses_value(self, account):
        import core.result_renderer as renderer

        with patch.object(renderer.store, "store_protected_result_rows") as protected, \
                patch.object(renderer.store, "get_compliance_profile", return_value={}):
            _card(account, ROWS, question_id="q1")
        assert protected.call_args.args[2][0]["WHS_DSC"] == "NULL value provided"

    def test_an_answer_with_none_of_them_is_unchanged(self, account):
        text = "\n".join(_card(account, [{"WHS_DSC": "North depot", "OH": 5.0}]).sent)
        assert "North depot" in text and "placeholders" not in text


class TestAQuestionAboutThemGetsTheirKeys:

    def test_the_plan_carries_their_keys_and_no_rule(self, account):
        plan: dict = {}
        assert attach_unknown_member_policies(plan, account, "stock with an unknown buyer") == []
        assert "unknown_member_policies" not in plan
        assert {p["table"] for p in plan["unknown_member_reference"]} == {"MART.WHS_DMS", "MART.PTY_DMS"}

    def test_the_prompt_names_the_keys_even_from_the_facts_side(self, account):
        plan: dict = {}
        attach_unknown_member_policies(plan, account, "stock with an unknown buyer")
        prompt = build_sql_system_prompt(
            "azure_sql", "## Table: MART.ITM_BAL_DLY_FCT\n| `BYR_PTY_DMS_KEY` | int |",
            semantic_plan=plan, question="stock with an unknown buyer",
        )
        block = prompt[prompt.index("## Unknown members — the question asks about them"):]
        assert "PTY_DMS_KEY -1 = unknown" in block
        assert "ITM_BAL_DLY_FCT.BYR_PTY_DMS_KEY" in block
        assert "never by a label" in block

    def test_a_question_not_about_them_gets_the_rule_instead(self, account):
        plan: dict = {}
        attach_unknown_member_policies(plan, account, "top 5 warehouses by stock")
        prompt = build_sql_system_prompt(
            "azure_sql", "## Table: MART.WHS_DMS\n| `WHS_DSC` | nvarchar |",
            semantic_plan=plan, question="top 5 warehouses by stock",
        )
        assert "## Unknown members — REQUIRED" in prompt
        assert "the question asks about them" not in prompt


class TestTheRealPipeline:
    """_handle_query_impl, end to end with the model and the warehouse stubbed:
    a question about unspecified warehouses reaches the model with their keys."""

    # Null-aware, as a filtered total must be.
    SQL = ("SELECT COUNT(*) AS MATCHED_ROWS, COUNT(f.ON_HND_QTY) AS OH_ROWS, "
           "COALESCE(SUM(f.ON_HND_QTY), 0) AS OH FROM MART.ITM_BAL_DLY_FCT f "
           "WHERE f.WHS_DMS_KEY = 0")

    def test_the_model_is_told_which_keys_they_are(self, account, tmp_path):
        import core.query_pipeline as qp
        from gateway.base import PlatformEvent

        seen: dict = {"prompts": [], "executed": []}

        class Retriever:
            last_retrieval_weak = False
            last_retrieval_unscored = False

            def retrieve(self, question, n=8, allowed_tables=None):
                return ["# MART.ITM_BAL_DLY_FCT\n| `WHS_DMS_KEY` | int |\n| `ON_HND_QTY` | decimal |"]

            def retrieve_fact_patterns(self, question, n=2, allowed_tables=None):
                return []

            def _is_global(self, doc):
                return False

        async def model_reply(system, user, *args, **kwargs):
            seen["prompts"].append(system)
            return self.SQL, 1, 1

        class Governed:
            def __init__(self, sql):
                self.sql, self.rows = sql, [{"OH": 42.0}]

        def execute(credentials, db_type, sql, **kwargs):
            seen["executed"].append(sql)
            return Governed(sql)

        async def card(*args, **kwargs):
            return None

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

        question = "stock on hand for unknown warehouses"
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
        assert any("WHS_DMS_KEY 0 = not specified" in system for system in seen["prompts"])
        assert seen["executed"] == [self.SQL]
