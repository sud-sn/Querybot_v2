"""
A label the warehouse keeps in two languages is shown in the reader's.

A warehouse built for two languages keeps a label twice -- ITM_GRP_DSC and
ITM_GRP_FR_DSC, MTH_NM and MTH_FR_NM -- and nothing knew the second was the
first in French: a French reader was shown the English label, and an English
reader could be shown the French one. The French copy is also blank on many
rows, so showing it as it stands gives a column of blanks.

A French reader is now shown the French label where it is filled and the
English one where it is blank; an English reader the English one; a question
that names a language, that one. The rule rides on the semantic plan: the
prompt states it, the validator refuses a label in the wrong language -- and
a grouped query that does not group by what it shows -- and the answer card
says where the other language stands in. A twin belongs to its table: a
table that keeps only the English label is left alone.

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
from core.label_language import (
    attach_label_policies,
    label_policies,
    language_twins,
    question_names_a_language,
)
from core.llm import build_sql_system_prompt
from core.result_renderer import _send_results
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
    "WH.MART.SLS_TRX_FCT": _table(
        "SLS_TRX_FCT_KEY", ("SLS_TRX_FCT_KEY", "bigint"), ("ITM_DMS_KEY", "int"),
        ("DT_DMS_KEY", "int"), ("WHS_DMS_KEY", "int"), ("SLS_AMT", "decimal(18,2)"),
    ),
    "WH.MART.ITM_DMS": _table(
        "ITM_DMS_KEY", ("ITM_DMS_KEY", "int"), ("ITM_GRP_DMS_KEY", "int"), ("ITM_CD", "nvarchar"),
        ("ITM_NM", "nvarchar"), ("ITM_FR_NM", "nvarchar"),
    ),
    "WH.MART.ITM_GRP_DMS": _table(
        "ITM_GRP_DMS_KEY", ("ITM_GRP_DMS_KEY", "int"), ("ITM_GRP_CD", "nvarchar"),
        ("ITM_GRP_DSC", "nvarchar"), ("ITM_GRP_FR_DSC", "nvarchar"),
    ),
    "WH.MART.DT_DMS": _table(
        "DT_DMS_KEY", ("DT_DMS_KEY", "int"), ("CAL_DT", "date"), ("MTH_NM", "nvarchar"),
        ("MTH_FR_NM", "nvarchar"),
    ),
    # Retired items: the English name only, no French copy.
    "WH.MART.ITM_ARC_DMS": _table(
        "ITM_ARC_DMS_KEY", ("ITM_ARC_DMS_KEY", "int"), ("ITM_NM", "nvarchar"),
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
        "name": "Sales Amount", "formula_type": "expression", "sql_template": "SUM(SLS_AMT)",
        "base_table": "MART.SLS_TRX_FCT", "synonyms": "sales, sales amount, ventes",
        "description": "Invoiced sales",
    })
    return account_id


@pytest.fixture
def policies(account):
    return label_policies(account)


def _validate(sql: str, policies: list[dict], language: str, db_type: str = "azure_sql"):
    return validate_sql_detailed(
        sql, KNOWN_TABLES, db_type, None, TABLE_COLUMNS,
        {"semantic_plan": {"label_policies": policies, "label_language": language}},
    )


class TestWhichColumnsAreTwins:

    @pytest.mark.parametrize("columns, twins", [
        (["ITM_GRP_DSC", "ITM_GRP_FR_DSC"], [("ITM_GRP_DSC", "ITM_GRP_FR_DSC", "fr")]),
        (["RGN_DSC", "RGN_DSC_FR"], [("RGN_DSC", "RGN_DSC_FR", "fr")]),
        (["Item_Name", "Item_Name_French"], [("Item_Name", "Item_Name_French", "fr")]),
        (["ITM_DSC", "ITM_EN_DSC"], [("ITM_DSC", "ITM_EN_DSC", "en")]),
        (["PC_EN_NM", "PC_FR_NM"], [("PC_EN_NM", "PC_FR_NM", "fr")]),
    ])
    def test_a_label_and_its_copy_in_another_language(self, columns, twins):
        assert [(t["base"], t["twin"], t["language"]) for t in language_twins(columns)] == twins

    @pytest.mark.parametrize("columns", [
        ["PRC_DT", "PRC_FR_DT"],        # FR is "from": a date, not a label
        ["ITM_FR_NM"],                  # a copy of nothing
        ["CUR_DSC", "CUR_FRN_DSC"],     # FRN is foreign, not French
        ["FRT_DSC", "FRT_FRE_DSC"],     # nor FRE
    ])
    def test_what_is_not_a_twin(self, columns):
        assert language_twins(columns) == []

    def test_the_policies_come_from_the_discovered_schema(self, policies):
        assert {p["table"].split(".", 1)[-1]: [t["twin"] for t in p["twins"]] for p in policies} == {
            "MART.DT_DMS": ["MTH_FR_NM"], "MART.ITM_DMS": ["ITM_FR_NM"],
            "MART.ITM_GRP_DMS": ["ITM_GRP_FR_DSC"],
        }


class TestTheLanguageShown:

    @pytest.mark.parametrize("question, language", [
        ("sales by item group in French", "fr"), ("French names of the items", "fr"),
        ("ventes par groupe d'articles en français", "fr"), ("noms français des articles", "fr"),
        ("sales by item group in English", "en"), ("libellés en anglais", "en"),
        ("sales in France", ""), ("French-speaking customers", ""), ("sales by item group", ""),
    ])
    def test_a_question_that_names_a_language(self, question, language):
        assert question_names_a_language(question) == language

    @pytest.mark.parametrize("reader, question, shown", [
        ("fr", "ventes par groupe d'articles", "fr"),
        ("en", "sales by item group", "en"),
        ("en", "sales by item group in French", "fr"),
        ("fr", "ventes par groupe d'articles, libellés en anglais", "en"),
        ("", "sales by item group", "en"),
    ])
    def test_the_reader_s_language_unless_the_question_names_one(self, account, reader, question, shown):
        plan: dict = {}
        attach_label_policies(plan, account, reader, question)
        assert plan["label_language"] == shown
        assert len(plan["label_policies"]) == 3

    def test_a_warehouse_with_no_twins_carries_no_rule(self, tmp_path):
        import store

        store.init_db()
        account_id = f"acct{os.urandom(4).hex()}"
        store.upsert_client(account_id, "Test Ltd")
        (tmp_path / "_schema.json").write_text(json.dumps({
            "WH.MART.WHS_DMS": SCHEMA["WH.MART.WHS_DMS"]}), encoding="utf-8")
        store.update_client_state(account_id, "READY", {"schema_dir": str(tmp_path)})
        plan: dict = {}
        assert attach_label_policies(plan, account_id, "fr", "ventes par entrepôt") == []
        assert plan == {}


SALES = (
    "FROM MART.SLS_TRX_FCT s JOIN MART.ITM_DMS i ON s.ITM_DMS_KEY = i.ITM_DMS_KEY "
    "JOIN MART.ITM_GRP_DMS g ON i.ITM_GRP_DMS_KEY = g.ITM_GRP_DMS_KEY"
)
FR_GROUP = "COALESCE(NULLIF(TRIM(g.ITM_GRP_FR_DSC), ''), g.ITM_GRP_DSC)"
BY_MONTH = "FROM MART.SLS_TRX_FCT s JOIN MART.DT_DMS d ON s.DT_DMS_KEY = d.DT_DMS_KEY"

REFUSED = {
    "the English label to a French reader": ("fr",
        f"SELECT g.ITM_GRP_DSC, SUM(s.SLS_AMT) AS AMT {SALES} GROUP BY g.ITM_GRP_DSC"),
    "the English month to a French reader": ("fr",
        f"SELECT d.MTH_NM, SUM(s.SLS_AMT) AS AMT {BY_MONTH} GROUP BY d.MTH_NM"),
    "an item's name, the only table": ("fr", "SELECT ITM_NM FROM MART.ITM_DMS"),
    "the French label to an English reader": ("en",
        f"SELECT g.ITM_GRP_FR_DSC, SUM(s.SLS_AMT) AS AMT {SALES} GROUP BY g.ITM_GRP_FR_DSC"),
    "the fallback to an English reader": ("en", f"SELECT {FR_GROUP} AS GRP {SALES}"),
    "the fallback grouped by the English label alone": ("fr",
        f"SELECT {FR_GROUP} AS GRP, SUM(s.SLS_AMT) AS AMT {SALES} GROUP BY g.ITM_GRP_DSC"),
    "the fallback grouped by the French label alone": ("fr",
        f"SELECT {FR_GROUP} AS GRP, SUM(s.SLS_AMT) AS AMT {SALES} GROUP BY g.ITM_GRP_FR_DSC"),
    "both passed up, the English shown": ("fr",
        f"WITH t AS (SELECT g.ITM_GRP_DSC, g.ITM_GRP_FR_DSC, SUM(s.SLS_AMT) AS AMT {SALES} "
        "GROUP BY g.ITM_GRP_DSC, g.ITM_GRP_FR_DSC) SELECT t.ITM_GRP_DSC, t.AMT FROM t"),
}

ACCEPTED = {
    "the fallback, grouped by it": ("fr",
        f"SELECT {FR_GROUP} AS GRP, SUM(s.SLS_AMT) AS AMT {SALES} GROUP BY {FR_GROUP}"),
    "the fallback, grouped by both labels": ("fr",
        f"SELECT {FR_GROUP} AS GRP, SUM(s.SLS_AMT) AS AMT {SALES} "
        "GROUP BY g.ITM_GRP_FR_DSC, g.ITM_GRP_DSC"),
    "the French month where filled": ("fr",
        "SELECT COALESCE(NULLIF(TRIM(d.MTH_FR_NM), ''), d.MTH_NM) AS MOIS, SUM(s.SLS_AMT) AS AMT "
        f"{BY_MONTH} GROUP BY COALESCE(NULLIF(TRIM(d.MTH_FR_NM), ''), d.MTH_NM)"),
    "both passed up, the fallback shown": ("fr",
        f"WITH t AS (SELECT g.ITM_GRP_DSC, g.ITM_GRP_FR_DSC, SUM(s.SLS_AMT) AS AMT {SALES} "
        "GROUP BY g.ITM_GRP_DSC, g.ITM_GRP_FR_DSC) "
        "SELECT COALESCE(NULLIF(TRIM(t.ITM_GRP_FR_DSC), ''), t.ITM_GRP_DSC) AS GRP, t.AMT FROM t"),
    "the fallback made inside, passed up under the English name": ("fr",
        f"WITH t AS (SELECT {FR_GROUP} AS ITM_GRP_DSC, SUM(s.SLS_AMT) AS AMT {SALES} "
        f"GROUP BY {FR_GROUP}) SELECT t.ITM_GRP_DSC, t.AMT FROM t"),
    "both labels side by side": ("fr",
        f"SELECT g.ITM_GRP_DSC, g.ITM_GRP_FR_DSC, SUM(s.SLS_AMT) AS AMT {SALES} "
        "GROUP BY g.ITM_GRP_DSC, g.ITM_GRP_FR_DSC"),
    "the fallback inside an aggregate, not grouped": ("fr",
        "SELECT COALESCE(NULLIF(TRIM(d.MTH_FR_NM), ''), d.MTH_NM) AS MOIS, "
        f"MIN({FR_GROUP}) AS PREMIER {BY_MONTH} "
        "JOIN MART.ITM_DMS i ON s.ITM_DMS_KEY = i.ITM_DMS_KEY "
        "JOIN MART.ITM_GRP_DMS g ON i.ITM_GRP_DMS_KEY = g.ITM_GRP_DMS_KEY "
        "GROUP BY COALESCE(NULLIF(TRIM(d.MTH_FR_NM), ''), d.MTH_NM)"),
    "the English label counted": ("fr", f"SELECT COUNT(DISTINCT g.ITM_GRP_DSC) AS N {SALES}"),
    "the English label compared": ("fr",
        f"SELECT {FR_GROUP} AS GRP, SUM(s.SLS_AMT) AS AMT {SALES} "
        f"WHERE g.ITM_GRP_DSC = 'Pipes' GROUP BY {FR_GROUP}"),
    "the English label compared in the SELECT list": ("fr",
        f"SELECT SUM(CASE WHEN g.ITM_GRP_DSC = 'Pipes' THEN s.SLS_AMT ELSE 0 END) AS PIPES {SALES}"),
    "the English label partitioning a window": ("fr",
        "SELECT s.SLS_AMT, ROW_NUMBER() OVER (PARTITION BY g.ITM_GRP_DSC "
        f"ORDER BY s.SLS_AMT DESC) AS RN {SALES}"),
    "a table that keeps no French copy": ("fr", "SELECT a.ITM_NM FROM MART.ITM_ARC_DMS a"),
    "the English label to an English reader": ("en",
        f"SELECT g.ITM_GRP_DSC, SUM(s.SLS_AMT) AS AMT {SALES} GROUP BY g.ITM_GRP_DSC"),
}


class TestTheValidator:

    @pytest.mark.parametrize("db_type", ["azure_sql", "snowflake"])
    @pytest.mark.parametrize("label", sorted(REFUSED))
    def test_refused(self, policies, label, db_type):
        language, sql = REFUSED[label]
        result = _validate(sql, policies, language, db_type)
        assert (result.ok, result.code) == (False, "label_language"), result.reason

    @pytest.mark.parametrize("db_type", ["azure_sql", "snowflake"])
    @pytest.mark.parametrize("label", sorted(ACCEPTED))
    def test_accepted(self, policies, label, db_type):
        language, sql = ACCEPTED[label]
        result = _validate(sql, policies, language, db_type)
        assert result.ok, (result.code, result.reason)

    @pytest.mark.parametrize("db_type, required", [
        ("azure_sql", "COALESCE(NULLIF(TRIM(g.[ITM_GRP_FR_DSC]), ''), g.[ITM_GRP_DSC])"),
        ("snowflake", "COALESCE(NULLIF(TRIM(g.\"ITM_GRP_FR_DSC\"), ''), g.\"ITM_GRP_DSC\")"),
    ])
    def test_the_fix_is_the_french_label_where_filled(self, policies, db_type, required):
        result = _validate(REFUSED["the English label to a French reader"][1], policies, "fr", db_type)
        assert [e["required_column"] for e in result.errors] == [required]
        assert "label_language" in REPAIRABLE_REASON_CODES

    def test_the_fix_for_an_english_reader_is_the_english_label(self, policies):
        result = _validate(REFUSED["the French label to an English reader"][1], policies, "en")
        assert [e["required_column"] for e in result.errors] == ["g.[ITM_GRP_DSC]"]

    def test_a_query_that_shows_but_does_not_group_is_told_to_group(self, policies):
        result = _validate(REFUSED["the fallback grouped by the English label alone"][1], policies, "fr")
        assert "does not group by" in result.errors[0]["message"]

    def test_passed_up_whole_the_fix_reads_the_cte(self, policies):
        result = _validate(REFUSED["both passed up, the English shown"][1], policies, "fr")
        assert [e["required_column"] for e in result.errors] == [
            "COALESCE(NULLIF(TRIM(t.[ITM_GRP_FR_DSC]), ''), t.[ITM_GRP_DSC])"]

    def test_without_the_rule_the_same_label_passes(self):
        assert _validate(REFUSED["the English label to a French reader"][1], [], "fr").ok


class TestThePromptStatesTheRule:

    def _block(self, policies, language, context):
        prompt = build_sql_system_prompt(
            "azure_sql", context, semantic_plan={"label_policies": policies, "label_language": language},
            question="sales by item group",
        )
        marker = "## Labels in the reader's language"
        return prompt[prompt.index(marker):] if marker in prompt else ""

    def test_a_french_reader_is_given_the_expression(self, policies):
        block = self._block(policies, "fr", "## Table: MART.ITM_GRP_DMS\n| `ITM_GRP_DSC` | nvarchar |")
        assert "The reader reads labels in French" in block
        assert ("- ITM_GRP_DMS: COALESCE(NULLIF(TRIM(ITM_GRP_DMS.[ITM_GRP_FR_DSC]), ''), "
                "ITM_GRP_DMS.[ITM_GRP_DSC])") in block
        assert "MTH_FR_NM" not in block

    def test_an_english_reader_is_told_which_to_show(self, policies):
        block = self._block(policies, "en", "## Table: MART.ITM_GRP_DMS\n| `ITM_GRP_DSC` | nvarchar |")
        assert "- ITM_GRP_DMS: ITM_GRP_DSC, not ITM_GRP_FR_DSC" in block

    def test_it_is_left_out_when_no_such_table_is_in_context(self, policies):
        assert self._block(policies, "fr", "## Table: MART.WHS_DMS\n| `WHS_DSC` | nvarchar |") == ""


class _Adapter:
    def __init__(self):
        self.sent: list[str] = []

    async def send_message(self, event, text):
        self.sent.append(text)


def _card(policies, language: str, sql: str, lang: str = "en") -> str:
    adapter = _Adapter()
    token = activate_language(lang)
    try:
        asyncio.run(_send_results(
            {}, adapter, "sales by item group",
            [{"GRP": "Plomberie", "AMT": 1200.0}, {"GRP": "Pipes", "AMT": 900.0}],
            sql, 900, None, 1, {"id": 1, "db_type": "azure_sql"}, question_id=None,
            confidence_context={"semantic_plan": {"label_policies": policies, "label_language": language}},
        ))
    finally:
        deactivate_language(token)
    return "\n".join(adapter.sent)


class TestTheReaderIsTold:

    @pytest.mark.parametrize("lang, phrase", [
        ("en", "Labels are shown in French where the warehouse has a French label, and in English"),
        ("fr", "Les libellés sont affichés en français lorsque l'entrepôt en fournit un, et en anglais"),
    ])
    def test_the_card_says_where_the_english_stands_in(self, policies, lang, phrase):
        assert phrase in _card(policies, "fr", ACCEPTED["the fallback, grouped by it"][1], lang)

    def test_both_labels_side_by_side_say_nothing(self, policies):
        assert "Labels are shown" not in _card(policies, "fr", ACCEPTED["both labels side by side"][1])

    def test_a_placeholder_is_not_the_english_standing_in(self, policies):
        sql = (f"SELECT COALESCE(g.ITM_GRP_FR_DSC, '-') AS GRP, SUM(s.SLS_AMT) AS AMT {SALES} "
               "GROUP BY g.ITM_GRP_FR_DSC")
        assert "Labels are shown" not in _card(policies, "fr", sql)

    def test_an_english_label_to_an_english_reader_says_nothing(self, policies):
        card = _card(policies, "en", ACCEPTED["the English label to an English reader"][1])
        assert "Labels are shown" not in card

    def test_an_english_twin_is_said_the_other_way_round(self):
        policies = [{"kind": "label_language", "table": "MART.PC_DMS", "twins": [
            {"base": "PC_DSC", "twin": "PC_EN_DSC", "language": "en"}]}]
        sql = "SELECT COALESCE(NULLIF(TRIM(p.PC_EN_DSC), ''), p.PC_DSC) AS PC FROM MART.PC_DMS p"
        assert "Labels are shown in English where the warehouse has an English label, and in French" in (
            _card(policies, "en", sql))

    @pytest.mark.parametrize("lang, phrase", [
        ("en", "both English and French"), ("fr", "en anglais et en français"),
    ])
    def test_a_refusal_that_survived_every_repair_explains_itself(self, lang, phrase):
        token = activate_language(lang)
        try:
            card = translate_failure(kind="validation", code="label_language")
        finally:
            deactivate_language(token)
        assert phrase in card["most_likely_reason"]
        assert card["suggested_next_step"]


class TestTheRealPipeline:
    """_handle_query_impl, end to end with the model and the warehouse stubbed:
    a French reader's English label is refused and repaired, and only the
    French-where-filled label reaches the warehouse; an English reader's
    English label runs as written -- unless the question asks for French."""

    FIRST = f"SELECT g.ITM_GRP_DSC, SUM(s.SLS_AMT) AS AMT {SALES} GROUP BY g.ITM_GRP_DSC"
    REPAIRED = f"SELECT {FR_GROUP} AS GRP, SUM(s.SLS_AMT) AS AMT {SALES} GROUP BY {FR_GROUP}"

    def _run(self, account, tmp_path, question, lang):
        import core.query_pipeline as qp
        from gateway.base import PlatformEvent

        seen: dict = {"prompts": [], "executed": [], "card": None}

        class Retriever:
            last_retrieval_weak = False
            last_retrieval_unscored = False

            def retrieve(self, question, n=8, allowed_tables=None):
                return [
                    "# MART.SLS_TRX_FCT\n| `ITM_DMS_KEY` | int |\n| `SLS_AMT` | decimal |",
                    "# MART.ITM_DMS\n| `ITM_DMS_KEY` | int |\n| `ITM_GRP_DMS_KEY` | int |",
                    "# MART.ITM_GRP_DMS\n| `ITM_GRP_DMS_KEY` | int |\n| `ITM_GRP_DSC` | nvarchar |"
                    "\n| `ITM_GRP_FR_DSC` | nvarchar |",
                ]

            def retrieve_fact_patterns(self, question, n=2, allowed_tables=None):
                return []

            def _is_global(self, doc):
                return False

        async def model_reply(system, user, *args, **kwargs):
            seen["prompts"].append((system, user))
            if "LABEL-LANGUAGE REPAIR REQUIRED" in user:
                return self.REPAIRED, 1, 1
            return self.FIRST, 1, 1

        class Governed:
            def __init__(self, sql):
                self.sql, self.rows = sql, [{"GRP": "Plomberie", "AMT": 1200.0}]

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
                {"id": 1, "role": "admin", "email": "u@x.com", "name": "U", "group_name": None,
                 "lang": lang},
            ))
        return seen

    def test_a_french_reader_is_shown_the_french_label_where_filled(self, account, tmp_path):
        seen = self._run(account, tmp_path, "ventes par groupe d'articles", "fr")
        assert any("## Labels in the reader's language" in system for system, _user in seen["prompts"])
        assert any("LABEL-LANGUAGE REPAIR REQUIRED" in user for _system, user in seen["prompts"])
        assert seen["executed"] == [self.REPAIRED]
        sql, plan = seen["card"]
        assert sql == self.REPAIRED and plan["label_language"] == "fr"

    def test_an_english_reader_is_shown_the_english_label(self, account, tmp_path):
        seen = self._run(account, tmp_path, "sales by item group", "en")
        assert not any("LABEL-LANGUAGE REPAIR REQUIRED" in user for _system, user in seen["prompts"])
        assert seen["executed"] == [self.FIRST]

    def test_an_english_reader_who_asks_for_french_is_shown_it(self, account, tmp_path):
        seen = self._run(account, tmp_path, "sales by item group in French", "en")
        assert seen["executed"] == [self.REPAIRED]
