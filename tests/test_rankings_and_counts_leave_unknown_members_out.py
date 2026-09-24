"""
A ranking or a count of a dimension's members leaves its placeholder members out.

A warehouse keeps, in each dimension, rows a fact points at when its value was
empty (key 0, NO_VALUE, "NULL value provided") or matched nothing (key 777,
NO_MATCH). Discovery finds them (core/unknown_members.py). Ranked, they make
"NULL value provided" the biggest buyer; counted, they add two suppliers that
do not exist. A listing or a total keeps them -- their rows are real stock.

Unless the question is about them, the rule rides on the semantic plan, as the
period-row rule does: the prompt states the predicate, the validator refuses a
ranking or a count without it, the governed compiler writes it, and the answer
card says they were left out.

Synthetic tables in a mart's naming convention; no customer data, no real
database. `store` is imported where it is used: modules collected after this
one re-import it, and the code under test resolves it at call time.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from core.failure_messages import translate_failure
from core.i18n import activate_language, deactivate_language
from core.llm import build_sql_system_prompt
from core.pipeline_helpers import compile_governed_temporal_metric_sql
from core.result_renderer import _send_results
from core.unknown_members import (
    attach_unknown_member_policies,
    question_asks_for_unknown_members,
    unknown_member_policies,
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
    "WH.MART.ITM_BAL_DLY_FCT": _table(
        "ITM_BAL_DLY_FCT_KEY", ("ITM_BAL_DLY_FCT_KEY", "bigint"), ("ITM_DMS_KEY", "int"),
        ("WHS_DMS_KEY", "int"), ("BYR_PTY_DMS_KEY", "int"), ("ON_HND_QTY", "decimal(18,4)"),
    ),
    "WH.MART.ITM_BAL_PRD_FCT": _table(
        "ITM_BAL_PRD_FCT_KEY", ("ITM_BAL_PRD_FCT_KEY", "bigint"), ("ITM_DMS_KEY", "int"),
        ("WHS_DMS_KEY", "int"), ("PRD_DMS_KEY", "int"), ("PCH_QTY", "decimal(18,4)"),
    ),
    "WH.MART.SLS_TRX_FCT": _table(
        "SLS_TRX_FCT_KEY", ("SLS_TRX_FCT_KEY", "bigint"), ("WHS_DMS_KEY", "int"),
        ("SLS_QTY", "decimal(18,4)"),
    ),
    "WH.MART.WHS_DMS": _table(
        "WHS_DMS_KEY", ("WHS_DMS_KEY", "int"), ("WHS_CD", "nvarchar"), ("WHS_DSC", "nvarchar"),
    ),
    "WH.MART.PTY_DMS": _table(
        "PTY_DMS_KEY", ("PTY_DMS_KEY", "int"), ("PTY_CD", "nvarchar"), ("PTY_NM", "nvarchar"),
    ),
    "WH.MART.PRD_DMS": _table(
        "PRD_DMS_KEY", ("PRD_DMS_KEY", "int"), ("PRD_CD", "nvarchar"), ("PRD_DSC", "nvarchar"),
    ),
    # Its warehouse keeps no placeholder members.
    "WH.MART.ITM_DMS": _table(
        "ITM_DMS_KEY", ("ITM_DMS_KEY", "int"), ("ITM_CD", "nvarchar"), ("ITM_NM", "nvarchar"),
    ),
}
KNOWN_TABLES = {fqn.split(".", 1)[1] for fqn in SCHEMA}
TABLE_COLUMNS = {
    fqn.split(".", 1)[1]: {column["name"]: column["type"] for column in meta["columns"]}
    for fqn, meta in SCHEMA.items()
}


def _members(code: str, name: str) -> list[dict]:
    return [
        {"key_value": "0", "kind": "not_specified",
         "member_text": {code: "NO_VALUE", name: "NULL value provided"}},
        {"key_value": "777", "kind": "unmatched",
         "member_text": {code: "NO_MATCH", name: "Value provided does not match"}},
    ]


@pytest.fixture
def account(tmp_path):
    """A client whose graph was discovered and whose warehouse, party and
    period dimensions keep placeholder members."""
    import store
    from core.graph_autopopulate import auto_populate_from_schema

    store.init_db()
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    (tmp_path / "_schema.json").write_text(json.dumps(SCHEMA), encoding="utf-8")
    store.update_client_state(account_id, "READY", {"schema_dir": str(tmp_path)})
    auto_populate_from_schema(account_id, str(tmp_path))
    # A ranking needs a governed measure; without one the reader is asked
    # which measure to rank by.
    store.save_metric(account_id, {
        "name": "Sales Quantity", "formula_type": "expression", "sql_template": "SUM(SLS_QTY)",
        "base_table": "MART.SLS_TRX_FCT", "synonyms": "sales quantity, units sold",
        "description": "Units sold",
    })
    store.save_unknown_members(account_id, "WHS_DMS", "WHS_DMS_KEY", _members("WHS_CD", "WHS_DSC"))
    store.save_unknown_members(account_id, "PRD_DMS", "PRD_DMS_KEY", _members("PRD_CD", "PRD_DSC"))
    store.save_unknown_members(account_id, "PTY_DMS", "PTY_DMS_KEY", [
        *_members("PTY_CD", "PTY_NM"),
        {"key_value": "-1", "kind": "unknown", "member_text": {"PTY_NM": "Unknown"}},
    ])
    store.set_unknown_member_status(account_id, "PTY_DMS", "-1", "rejected")
    return account_id


@pytest.fixture
def policies(account):
    return unknown_member_policies(account)


def _validate(sql: str, policies: list[dict], db_type: str = "azure_sql"):
    return validate_sql_detailed(
        sql, KNOWN_TABLES, db_type, None, TABLE_COLUMNS,
        {"semantic_plan": {"unknown_member_policies": policies}},
    )


class TestThePolicyComesFromTheStore:

    def test_each_dimension_with_members_has_one(self, policies):
        by_table = {p["table"]: p for p in policies}
        assert set(by_table) == {"MART.WHS_DMS", "MART.PTY_DMS", "MART.PRD_DMS"}
        assert by_table["MART.WHS_DMS"]["keys"] == ["0", "777"]
        assert by_table["MART.WHS_DMS"]["member_kinds"] == {"0": "not_specified", "777": "unmatched"}

    def test_a_rejected_member_is_not_in_it(self, policies):
        party = next(p for p in policies if p["table"] == "MART.PTY_DMS")
        assert party["keys"] == ["0", "777"]

    def test_it_knows_the_keys_that_point_at_the_dimension_roles_included(self, policies):
        by_table = {p["table"]: p for p in policies}
        assert {(r["table"], r["column"]) for r in by_table["MART.WHS_DMS"]["references"]} == {
            ("MART.ITM_BAL_DLY_FCT", "WHS_DMS_KEY"), ("MART.ITM_BAL_PRD_FCT", "WHS_DMS_KEY"),
            ("MART.SLS_TRX_FCT", "WHS_DMS_KEY"),
        }
        assert ("MART.ITM_BAL_DLY_FCT", "BYR_PTY_DMS_KEY") in {
            (r["table"], r["column"]) for r in by_table["MART.PTY_DMS"]["references"]
        }


class TestTheQuestionDecides:

    @pytest.mark.parametrize("question", [
        "top 5 warehouses by stock on hand",
        "how many suppliers have stock",
        "les 5 premiers entrepôts par quantité en stock",
        "combien de fournisseurs ont du stock",
    ])
    def test_a_ranking_or_a_count_carries_the_rule(self, account, question):
        plan: dict = {}
        attach_unknown_member_policies(plan, account, question)
        assert [p["table"] for p in plan["unknown_member_policies"]] == [
            "MART.PRD_DMS", "MART.PTY_DMS", "MART.WHS_DMS"]

    @pytest.mark.parametrize("question", [
        "stock with no buyer",
        "top 5 unmatched warehouses by stock",
        "how much stock has an unknown warehouse",
        "articles sans fournisseur",
        "stock des acheteurs non renseignés",
        "entrepôts inconnus",
    ])
    def test_a_question_about_them_does_not(self, account, question):
        plan: dict = {}
        assert question_asks_for_unknown_members(question)
        assert attach_unknown_member_policies(plan, account, question) == []
        assert "unknown_member_policies" not in plan


FACT_JOIN = ("FROM MART.ITM_BAL_DLY_FCT f "
             "JOIN MART.WHS_DMS w ON f.WHS_DMS_KEY = w.WHS_DMS_KEY")

REFUSED = {
    "a top-N ranking": ("azure_sql",
        f"SELECT TOP 5 w.WHS_DSC, SUM(f.ON_HND_QTY) AS OH {FACT_JOIN} "
        "GROUP BY w.WHS_DSC ORDER BY OH DESC"),
    "a LIMIT ranking that leaves out key 0 only": ("snowflake",
        f"SELECT w.WHS_DSC, SUM(f.ON_HND_QTY) AS OH {FACT_JOIN} WHERE f.WHS_DMS_KEY > 0 "
        "GROUP BY 1 ORDER BY 2 DESC LIMIT 5"),
    "a FETCH FIRST ranking": ("oracle",
        f"SELECT w.WHS_DSC, SUM(f.ON_HND_QTY) AS OH {FACT_JOIN} "
        "GROUP BY w.WHS_DSC ORDER BY OH DESC FETCH FIRST 5 ROWS ONLY"),
    "a RANK() window": ("azure_sql",
        f"SELECT w.WHS_DSC, RANK() OVER (ORDER BY SUM(f.ON_HND_QTY) DESC) AS RNK {FACT_JOIN} "
        "GROUP BY w.WHS_DSC"),
    "a ranking in the query over a CTE": ("azure_sql",
        f"WITH by_whs AS (SELECT w.WHS_DSC, SUM(f.ON_HND_QTY) AS OH {FACT_JOIN} "
        "GROUP BY w.WHS_DSC) SELECT TOP 5 WHS_DSC, OH FROM by_whs ORDER BY OH DESC"),
    "the buyer role ranked": ("azure_sql",
        "SELECT TOP 5 b.PTY_NM, SUM(f.ON_HND_QTY) AS OH FROM MART.ITM_BAL_DLY_FCT f "
        "JOIN MART.PTY_DMS b ON f.BYR_PTY_DMS_KEY = b.PTY_DMS_KEY "
        "GROUP BY b.PTY_NM ORDER BY OH DESC"),
    "a distinct count of the fact's key": ("azure_sql",
        "SELECT COUNT(DISTINCT f.WHS_DMS_KEY) AS N FROM MART.ITM_BAL_DLY_FCT f"),
    "a distinct count of the buyer": ("azure_sql",
        "SELECT COUNT(DISTINCT f.BYR_PTY_DMS_KEY) AS N FROM MART.ITM_BAL_DLY_FCT f"),
    "a count of the dimension's rows": ("azure_sql",
        "SELECT COUNT(*) AS N FROM MART.WHS_DMS"),
    "a LEFT JOIN that leaves them out only in its ON clause": ("azure_sql",
        "SELECT TOP 5 w.WHS_DSC, SUM(f.ON_HND_QTY) AS OH FROM MART.ITM_BAL_DLY_FCT f "
        "LEFT JOIN MART.WHS_DMS w ON f.WHS_DMS_KEY = w.WHS_DMS_KEY "
        "AND w.WHS_DMS_KEY NOT IN (0, 777) GROUP BY w.WHS_DSC ORDER BY OH DESC"),
}

ACCEPTED = {
    "NOT IN on the dimension's key": ("azure_sql",
        f"SELECT TOP 5 w.WHS_DSC, SUM(f.ON_HND_QTY) AS OH {FACT_JOIN} "
        "WHERE w.WHS_DMS_KEY NOT IN (0, 777) GROUP BY w.WHS_DSC ORDER BY OH DESC"),
    "<> on the fact's key": ("snowflake",
        f"SELECT w.WHS_DSC, SUM(f.ON_HND_QTY) AS OH {FACT_JOIN} "
        "WHERE f.WHS_DMS_KEY <> 0 AND f.WHS_DMS_KEY <> 777 GROUP BY 1 ORDER BY 2 DESC LIMIT 5"),
    "a filter to named warehouses": ("oracle",
        f"SELECT w.WHS_DSC, SUM(f.ON_HND_QTY) AS OH {FACT_JOIN} WHERE w.WHS_CD IN ('W01', 'W02') "
        "GROUP BY w.WHS_DSC ORDER BY OH DESC FETCH FIRST 5 ROWS ONLY"),
    "the exclusion inside the CTE": ("azure_sql",
        f"WITH by_whs AS (SELECT w.WHS_DSC, SUM(f.ON_HND_QTY) AS OH {FACT_JOIN} "
        "WHERE w.WHS_DMS_KEY NOT IN (0, 777) GROUP BY w.WHS_DSC) "
        "SELECT TOP 5 WHS_DSC, OH FROM by_whs ORDER BY OH DESC"),
    "an inner join that carries it": ("azure_sql",
        "SELECT TOP 5 w.WHS_DSC, SUM(f.ON_HND_QTY) AS OH FROM MART.ITM_BAL_DLY_FCT f "
        "JOIN MART.WHS_DMS w ON f.WHS_DMS_KEY = w.WHS_DMS_KEY AND w.WHS_DMS_KEY NOT IN (0, 777) "
        "GROUP BY w.WHS_DSC ORDER BY OH DESC"),
    "a distinct count that leaves them out": ("azure_sql",
        "SELECT COUNT(DISTINCT f.WHS_DMS_KEY) AS N FROM MART.ITM_BAL_DLY_FCT f "
        "WHERE f.WHS_DMS_KEY NOT IN (0, 777)"),
    "a listing, not a ranking": ("azure_sql",
        f"SELECT w.WHS_DSC, SUM(f.ON_HND_QTY) AS OH {FACT_JOIN} GROUP BY w.WHS_DSC ORDER BY OH DESC"),
    "a total": ("azure_sql", "SELECT SUM(f.ON_HND_QTY) AS OH FROM MART.ITM_BAL_DLY_FCT f"),
    "a ranking of a dimension with no placeholder members": ("azure_sql",
        "SELECT TOP 5 i.ITM_NM, SUM(f.ON_HND_QTY) AS OH FROM MART.ITM_BAL_DLY_FCT f "
        "JOIN MART.ITM_DMS i ON f.ITM_DMS_KEY = i.ITM_DMS_KEY GROUP BY i.ITM_NM ORDER BY OH DESC"),
    "the exclusion in the query over the CTE": ("azure_sql",
        "WITH by_whs AS (SELECT f.WHS_DMS_KEY, SUM(f.ON_HND_QTY) AS OH "
        "FROM MART.ITM_BAL_DLY_FCT f GROUP BY f.WHS_DMS_KEY) "
        "SELECT TOP 5 WHS_DMS_KEY, OH FROM by_whs WHERE WHS_DMS_KEY NOT IN (0, 777) ORDER BY OH DESC"),
    "a count of the dimension's named members": ("azure_sql",
        "SELECT COUNT(*) AS N FROM MART.WHS_DMS WHERE WHS_CD IN ('W01', 'W02')"),
    "a period ranking that keeps month rows": ("azure_sql",
        "SELECT TOP 3 p.PRD_DMS_KEY, SUM(p.PCH_QTY) AS PCH FROM MART.ITM_BAL_PRD_FCT p "
        "WHERE p.PRD_DMS_KEY % 100 BETWEEN 1 AND 12 GROUP BY p.PRD_DMS_KEY ORDER BY PCH DESC"),
}


class TestTheValidatorRefusesARankingWithThemIn:

    @pytest.mark.parametrize("label", sorted(REFUSED))
    def test_refused(self, policies, label):
        db_type, sql = REFUSED[label]
        result = _validate(sql, policies, db_type)
        assert (result.ok, result.code) == (False, "unknown_members_ranked"), result.reason

    @pytest.mark.parametrize("label", sorted(ACCEPTED))
    def test_accepted(self, policies, label):
        db_type, sql = ACCEPTED[label]
        result = _validate(sql, policies, db_type)
        assert result.ok, (result.code, result.reason)

    def test_the_refusal_names_the_predicate_with_the_querys_own_alias(self, policies):
        result = _validate(REFUSED["a top-N ranking"][1], policies)
        assert result.errors[0]["required_predicate"] == "w.WHS_DMS_KEY NOT IN (0, 777)"
        assert "unknown_members_ranked" in REPAIRABLE_REASON_CODES

    def test_without_the_rule_the_same_ranking_passes(self):
        assert _validate(REFUSED["a top-N ranking"][1], []).ok

    def test_a_table_of_the_same_name_in_another_schema_is_another_table(self, policies):
        from core.unknown_members import member_scopes

        sql = ("SELECT TOP 5 x.WHS_DSC, SUM(f.ON_HND_QTY) AS OH FROM MART.ITM_BAL_DLY_FCT f "
               "JOIN STAGE.WHS_DMS x ON f.ITM_DMS_KEY = x.WHS_DMS_KEY GROUP BY x.WHS_DSC ORDER BY OH DESC")
        assert member_scopes(sql, policies) == []


class TestThePromptStatesTheRule:

    def _prompt(self, policies, context):
        return build_sql_system_prompt(
            "azure_sql", context,
            semantic_plan={"unknown_member_policies": policies},
            question="top 5 warehouses by stock on hand",
        )

    def test_it_is_stated_for_the_dimension_in_context(self, policies):
        prompt = self._prompt(policies, "## Table: MART.WHS_DMS\n| `WHS_DSC` | nvarchar |")
        block = prompt[prompt.index("## Unknown members"):]
        assert "<alias>.WHS_DMS_KEY NOT IN (0, 777)" in block
        assert "ITM_BAL_DLY_FCT.WHS_DMS_KEY" in block
        assert "PTY_DMS" not in block

    def test_it_is_left_out_when_no_such_dimension_is(self, policies):
        prompt = self._prompt(policies, "## Table: MART.ITM_DMS\n| `ITM_NM` | nvarchar |")
        assert "## Unknown members" not in prompt


OPS_COLUMNS = {
    "OPS.F_SALES": {
        "SALES_LINE_SK": "bigint", "INVOICE_DATE_SK": "int", "WAREHOUSE_SK": "int",
        "CUSTOMER_SK": "int", "NET_REVENUE_AMOUNT": "decimal",
    },
    "OPS.D_DATE": {"DATE_SK": "int", "FULL_DATE": "date"},
    "OPS.D_WAREHOUSE": {"WAREHOUSE_SK": "int", "WAREHOUSE_NAME": "varchar"},
    "OPS.D_CUSTOMER": {"CUSTOMER_SK": "int", "CUSTOMER_NAME": "varchar"},
}


def _ops_policy(table: str, key: str, fact_column: str) -> dict:
    return {
        "kind": "unknown_members", "table": table, "entities": [table.split(".")[-1]],
        "key_column": key, "keys": ["0", "777"],
        "member_kinds": {"0": "not_specified", "777": "unmatched"},
        "member_text": {"0": {}, "777": {}},
        "references": [{"table": "OPS.F_SALES", "column": fact_column}],
    }


def _ops_context(policies: list[dict], *, derived: dict | None = None) -> dict:
    return {
        "question": "top 10 warehouses by revenue in the last 2 months",
        "top_n": {"limit": 10, "direction": "descending", "tie_policy": "exactly_n"},
        "metric_formulas": [] if derived else [{
            "name": "Revenue", "formula_type": "expression",
            "sql_template": "SUM(NET_REVENUE_AMOUNT)", "base_table": "OPS.F_SALES",
        }],
        "semantic_plan": {
            "fields": [{
                "term": "Warehouse", "table": "OPS.D_WAREHOUSE", "column": "WAREHOUSE_NAME",
                "role": "display_dimension", "display_required": True,
            }],
            "joins": [
                {"from": "OPS.F_SALES", "to": "OPS.D_DATE",
                 "conditions": [["INVOICE_DATE_SK", "DATE_SK"]], "enforcement": "required"},
                {"from": "OPS.F_SALES", "to": "OPS.D_WAREHOUSE",
                 "conditions": [["WAREHOUSE_SK", "WAREHOUSE_SK"]], "enforcement": "required"},
            ],
            "temporal_policies": [{
                "kind": "last_n", "amount": 2, "unit": "month", "requested_grain": "month",
                "fact_table": "OPS.F_SALES", "fact_column": "INVOICE_DATE_SK",
                "dimension_table": "OPS.D_DATE", "dimension_key": "DATE_SK",
                "date_column": "FULL_DATE", "date_key_type": "surrogate_fk",
                "role_alias": "invoice_date",
            }],
            "unknown_member_policies": policies,
        },
        "analytical_request_plan": {
            "status": "compiled", "intent": "ranking", "source_fact": "OPS.F_SALES",
            "source_facts": ["OPS.F_SALES"], "top_n": 10, "output_shape": "table",
            **({"derived_measure": derived} if derived else {}),
        },
    }


class TestTheGovernedCompilerWritesIt:

    def _compile(self, context):
        return compile_governed_temporal_metric_sql(
            "azure_sql", set(OPS_COLUMNS), set(OPS_COLUMNS), OPS_COLUMNS, context,
        )

    def test_a_compiled_ranking_leaves_them_out_and_validates(self):
        context = _ops_context([_ops_policy("OPS.D_WAREHOUSE", "WAREHOUSE_SK", "WAREHOUSE_SK")])
        sql = self._compile(context)
        assert "business_dimension.[WAREHOUSE_SK] NOT IN (0, 777)" in sql
        result = validate_sql_detailed(
            sql, set(OPS_COLUMNS), "azure_sql", set(OPS_COLUMNS), OPS_COLUMNS, context,
        )
        assert result.ok, result.reason

    def test_a_compiled_count_of_a_key_leaves_them_out(self):
        context = _ops_context(
            [_ops_policy("OPS.D_CUSTOMER", "CUSTOMER_SK", "CUSTOMER_SK")],
            derived={
                "semantics": "count_distinct_business_identifier", "business_entity": "customer",
                "target_table": "OPS.F_SALES", "target_column": "CUSTOMER_SK",
            },
        )
        sql = self._compile(context)
        assert "fact_rows.[CUSTOMER_SK] NOT IN (0, 777)" in sql
        assert "business_dimension.[WAREHOUSE_SK] NOT IN" not in sql

    def test_without_the_rule_it_compiles_as_before(self):
        assert "NOT IN" not in self._compile(_ops_context([]))


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
            {}, adapter, "top 5 warehouses by stock", [{"WHS_DSC": "North", "OH": 135.0}],
            sql, 900, None, 1, {"id": 1, "db_type": "azure_sql"},
            question_id=None,
            confidence_context={"semantic_plan": {"unknown_member_policies": policies}},
        ))
    finally:
        deactivate_language(token)
    return "\n".join(adapter.sent)


class TestTheReaderIsTold:

    @pytest.mark.parametrize("lang, phrase", [
        ("en", "left out of this ranking"), ("fr", "exclus de ce classement"),
    ])
    def test_the_card_says_they_were_left_out_of_the_ranking(self, policies, lang, phrase):
        assert phrase in _card(policies, ACCEPTED["NOT IN on the dimension's key"][1], lang)

    @pytest.mark.parametrize("lang, phrase", [("en", "are not counted"), ("fr", "ne sont pas comptés")])
    def test_and_out_of_the_count(self, policies, lang, phrase):
        assert phrase in _card(policies, ACCEPTED["a distinct count that leaves them out"][1], lang)

    def test_a_listing_says_nothing(self, policies):
        text = _card(policies, ACCEPTED["a listing, not a ranking"][1])
        assert "left out of this ranking" not in text and "are not counted" not in text

    @pytest.mark.parametrize("lang, phrase", [("en", "NULL value provided"), ("fr", "NULL value provided")])
    def test_a_refusal_that_survived_every_repair_explains_itself(self, lang, phrase):
        token = activate_language(lang)
        try:
            card = translate_failure(kind="validation", code="unknown_members_ranked")
        finally:
            deactivate_language(token)
        assert phrase in card["most_likely_reason"]
        assert card["suggested_next_step"]


class TestTheRealPipeline:
    """_handle_query_impl, end to end with the model and the warehouse stubbed:
    the policy is attached on the path every generated query takes, the
    model's first ranking is refused and repaired, and only the repaired query
    reaches the warehouse -- unless the question asked about them."""

    FIRST = ("SELECT TOP 5 w.WHS_DSC, SUM(s.SLS_QTY) AS QTY FROM MART.SLS_TRX_FCT s "
             "JOIN MART.WHS_DMS w ON s.WHS_DMS_KEY = w.WHS_DMS_KEY "
             "GROUP BY w.WHS_DSC ORDER BY QTY DESC")
    REPAIRED = ("SELECT TOP 5 w.WHS_DSC, SUM(s.SLS_QTY) AS QTY FROM MART.SLS_TRX_FCT s "
                "JOIN MART.WHS_DMS w ON s.WHS_DMS_KEY = w.WHS_DMS_KEY "
                "WHERE w.WHS_DMS_KEY NOT IN (0, 777) GROUP BY w.WHS_DSC ORDER BY QTY DESC")

    def _run(self, account, tmp_path, question):
        import contextlib
        from unittest.mock import patch

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
            if "UNKNOWN-MEMBER REPAIR REQUIRED" in user:
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

    def test_a_ranking_question_is_answered_without_them(self, account, tmp_path):
        seen = self._run(account, tmp_path, "top 5 warehouses by sales quantity")
        assert any("## Unknown members" in system for system, _user in seen["prompts"])
        assert any("UNKNOWN-MEMBER REPAIR REQUIRED" in user for _system, user in seen["prompts"])
        assert seen["executed"] == [self.REPAIRED]
        sql, plan = seen["card"]
        assert sql == self.REPAIRED
        assert "MART.WHS_DMS" in [p["table"] for p in plan["unknown_member_policies"]]

    def test_a_question_about_them_is_answered_with_them(self, account, tmp_path):
        # ("unmatched warehouses" is a missing-records question: an anti-join.)
        seen = self._run(account, tmp_path, "top 5 warehouses by sales quantity, unknown warehouses included")
        assert not any("## Unknown members" in system for system, _user in seen["prompts"])
        assert seen["executed"] == [self.FIRST]
