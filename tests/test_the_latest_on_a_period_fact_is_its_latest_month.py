# -*- coding: utf-8 -*-
"""On a period fact, "latest" is its newest month, not the time a row was written.

A reader asked "What is our total stock on hand?", was asked which source to
use, and chose the monthly balance fact. The answer was a refusal: "I could not
build a trusted query", reason period_rows_mixed, after three model calls.

The monthly fact is keyed by a yyyymm period and also carries BAL_TS, the time
each row was written. Discovery names that column "Balance Date", and a stock
question binds the finest-grained date on the fact -- so it bound BAL_TS, not
the period. The prompt then carried two REQUIRED rules that no SQL satisfies
together: "copy this exact subquery as the anchor: (SELECT MAX(BAL_TS) FROM
<fact>)", and "every SELECT that reads <fact> -- including each subquery -- must
keep month rows only". The model copied the subquery, the validator refused it,
and every repair asked for the same copy.

Four fixes, each tested here:

  * a timestamp named as one, on a fact keyed by a period, is not offered as
    that fact's date -- unless an admin approved it or the reader names it;
  * an anchor on such a fact keeps month rows, in the prompt and in the probe;
  * a date bound to the question is never counted as its measure -- BAL_TS was
    the plan's "Measures:" line, and that alone let the plan compile;
  * end to end: a model that copies the anchor it is given gets an answer.

A synthetic mart in the same naming convention; no customer data.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import sqlite3
import tempfile
from unittest.mock import patch

import pytest

# The store is conftest's scratch database for the run (tests/conftest.py);
# nothing here repoints or re-imports it, so modules imported before this one
# keep the same store object as every test after it.
import store  # noqa: E402
from core.analytical_request_plan import compile_analytical_request_plan  # noqa: E402
from core.contextual_dates import (  # noqa: E402
    build_contextual_date_plan,
    resolve_contextual_date_binding,
)
from core.date_anchor import build_anchor_probe_sql  # noqa: E402
from core.period_rows import attach_period_row_policies  # noqa: E402
from core.semantic_model import build_semantic_model  # noqa: E402
from core.semantic_planner import format_semantic_field_plan  # noqa: E402
from core.validator import validate_sql_detailed  # noqa: E402

PERIOD_FACT = "MART.ITM_BAL_PRD_FCT"
STOCK = "What is our total stock on hand?"


def _table(own_key: str, *columns: tuple[str, str]) -> dict:
    return {
        "columns": [{"name": name, "type": dtype, "nullable": True, "comment": ""} for name, dtype in columns],
        "pk_columns": [own_key], "row_count": 1000, "comment": "", "schema": "MART", "database": "WH",
    }


SCHEMA = {
    # A month-end balance fact: a yyyymm period key, and the time each row was written.
    "WH.MART.ITM_BAL_PRD_FCT": _table(
        "ITM_BAL_PRD_FCT_KEY",
        ("ITM_BAL_PRD_FCT_KEY", "bigint"), ("ITM_DMS_KEY", "int"), ("WHS_DMS_KEY", "int"),
        ("PRD_DMS_KEY", "int"), ("CUR_ON_HND_QTY", "decimal(18,4)"), ("SLD_QTY", "decimal(18,4)"),
        ("BAL_TS", "datetime2"),
    ),
    # A period fact whose other date is a calendar key, reached through DT_DMS.
    "WH.MART.ITM_RCT_PRD_FCT": _table(
        "ITM_RCT_PRD_FCT_KEY",
        ("ITM_RCT_PRD_FCT_KEY", "bigint"), ("PRD_DMS_KEY", "int"),
        ("LST_RCT_DT_DMS_KEY", "int"), ("RCT_QTY", "decimal(18,4)"),
    ),
    # A daily snapshot kept in a DATETIME column named like a day, beside a fiscal period.
    "WH.MART.STK_SNP_FCT": _table(
        "STK_SNP_FCT_KEY",
        ("STK_SNP_FCT_KEY", "bigint"), ("ITM_DMS_KEY", "int"), ("SNP_DT", "datetime"),
        ("FSC_PRD_DMS_KEY", "int"), ("SNP_ON_HND_QTY", "decimal(18,4)"),
    ),
    # A period fact whose other date is typed DATE, whatever its name says.
    "WH.MART.CUS_BAL_PRD_FCT": _table(
        "CUS_BAL_PRD_FCT_KEY",
        ("CUS_BAL_PRD_FCT_KEY", "bigint"), ("PRD_DMS_KEY", "int"),
        ("VAL_DATETIME", "date"), ("BAL_AMT", "decimal(18,2)"),
    ),
    # A timestamp on a fact with no period key.
    "WH.MART.GL_TRN_FCT": _table(
        "GL_TRN_FCT_KEY", ("GL_TRN_FCT_KEY", "bigint"), ("PST_TS", "datetime2"), ("TRN_AMT", "decimal(18,2)"),
    ),
    "WH.MART.PRD_DMS": _table("PRD_DMS_KEY", ("PRD_DMS_KEY", "int"), ("PRD_DSC", "varchar(40)")),
    "WH.MART.DT_DMS": _table("DT_DMS_KEY", ("DT_DMS_KEY", "int"), ("DMS_DT", "date"), ("YR", "int"), ("MTH", "int")),
    "WH.MART.WHS_DMS": _table("WHS_DMS_KEY", ("WHS_DMS_KEY", "int"), ("WHS_DSC", "varchar(60)")),
    "WH.MART.ITM_DMS": _table("ITM_DMS_KEY", ("ITM_DMS_KEY", "int"), ("ITM_DSC", "varchar(60)")),
}
KNOWN_TABLES = {fqn.split(".", 1)[1] for fqn in SCHEMA}
TABLE_COLUMNS = {
    fqn.split(".", 1)[1]: {column["name"]: column["type"] for column in meta["columns"]}
    for fqn, meta in SCHEMA.items()
}


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    directory = tmp_path_factory.mktemp("period_fact_dates")
    (directory / "_schema.json").write_text(json.dumps(SCHEMA), encoding="utf-8")
    return build_semantic_model(str(directory))


def _roles(model, **status_by_column) -> list[dict]:
    return [
        dict(role, status=status_by_column.get(role["fact_column"], role["status"]))
        for role in model["date_roles"]
    ]


def _resolve(question: str, fact: str, roles: list[dict], **kwargs) -> dict:
    return resolve_contextual_date_binding(
        question, matched_metrics=[], bindings=[], date_roles=roles,
        required_fact_tables={fact}, **kwargs,
    )


def _bound(result: dict) -> str:
    return str((result.get("binding") or {}).get("fact_column") or "")


class TestAPeriodFactsDateIsItsPeriod:

    def test_a_stock_question_binds_the_period(self, model):
        result = _resolve(STOCK, PERIOD_FACT, _roles(model))
        assert (result["status"], _bound(result)) == ("selected", "PRD_DMS_KEY")

    def test_a_year_question_binds_it_without_asking(self, model):
        result = _resolve("How many units were sold in 2022?", PERIOD_FACT, _roles(model))
        assert (result["status"], _bound(result)) == ("selected", "PRD_DMS_KEY")

    def test_an_admin_who_approved_the_period_gets_the_period(self, model):
        result = _resolve(STOCK, PERIOD_FACT, _roles(model, PRD_DMS_KEY="approved"))
        assert _bound(result) == "PRD_DMS_KEY"


class TestWhatStillBindsTheTimestamp:

    def test_an_admin_who_approved_it(self, model):
        result = _resolve(STOCK, PERIOD_FACT, _roles(model, BAL_TS="approved"))
        assert _bound(result) == "BAL_TS"

    def test_a_reader_who_names_it(self, model):
        result = _resolve("What is our stock on hand by balance date?", PERIOD_FACT, _roles(model))
        assert _bound(result) == "BAL_TS"

    def test_a_fact_with_no_period_key(self, model):
        result = _resolve(STOCK, "MART.GL_TRN_FCT", _roles(model))
        assert _bound(result) == "PST_TS"

    def test_a_fact_whose_period_key_an_admin_rejected(self, model):
        result = _resolve(STOCK, PERIOD_FACT, _roles(model, PRD_DMS_KEY="rejected"))
        assert _bound(result) == "BAL_TS"

    def test_a_column_typed_as_a_day_whatever_its_name(self, model):
        result = _resolve(STOCK, "MART.CUS_BAL_PRD_FCT", _roles(model))
        assert _bound(result) == "VAL_DATETIME"

    def test_a_day_kept_in_a_datetime_column_stays_the_snapshot(self, model):
        """SNP_DT is typed DATETIME but named a day: the finest grain still wins,
        so a month's daily snapshots are never summed into one."""
        result = _resolve(STOCK, "MART.STK_SNP_FCT", _roles(model))
        assert _bound(result) == "SNP_DT"


def _plan_for(model, binding: dict, question: str = STOCK) -> dict:
    """The date plan the pipeline builds, with the period rule attached."""
    date_plan = build_contextual_date_plan(binding, question)
    plan = {
        "enabled": True, "fields": date_plan["fields"], "joins": date_plan["joins"],
        "temporal_policies": date_plan["temporal_policies"],
        "date_key_policies": date_plan.get("date_key_policies") or [],
    }
    attach_period_row_policies(plan, model)
    return plan


_REQUIRED_ANCHOR = re.compile(r"REQUIRED ANCHOR \(copy this exact subquery[^)]*\): (\(SELECT MAX\((.+?)\) FROM .+\))$", re.M)


def _anchor_and_value(plan: dict) -> tuple[str, str]:
    found = _REQUIRED_ANCHOR.search(format_semantic_field_plan(plan))
    assert found, format_semantic_field_plan(plan)
    return found.group(1), found.group(2)


def _validate(sql: str, plan: dict):
    return validate_sql_detailed(
        sql, KNOWN_TABLES, "azure_sql", None, TABLE_COLUMNS, {"semantic_plan": plan},
    )


class TestTheAnchorKeepsMonthRows:

    def test_the_anchor_a_model_copies_passes_the_year_row_rule(self, model):
        binding = _resolve(STOCK, PERIOD_FACT, _roles(model, BAL_TS="approved"))["binding"]
        plan = _plan_for(model, binding)
        anchor, value = _anchor_and_value(plan)
        sql = (
            f"SELECT SUM(f.CUR_ON_HND_QTY) AS STOCK FROM {PERIOD_FACT} f "
            f"WHERE f.PRD_DMS_KEY % 100 BETWEEN 1 AND 12 AND CAST(f.{value} AS date) = CAST({anchor} AS date)"
        )
        result = _validate(sql, plan)
        assert result.ok, (result.code, result.reason)

    def test_a_calendar_date_on_a_period_fact_keeps_month_rows_too(self, model):
        receipt = next(role for role in model["date_roles"] if role["fact_column"] == "LST_RCT_DT_DMS_KEY")
        binding = _resolve(STOCK, "MART.ITM_RCT_PRD_FCT", _roles(model), confirmed_date_role=receipt)["binding"]
        plan = _plan_for(model, binding)
        anchor, _ = _anchor_and_value(plan)
        sql = (
            "SELECT SUM(f.RCT_QTY) AS RECEIVED FROM MART.ITM_RCT_PRD_FCT f "
            "JOIN MART.DT_DMS d ON f.LST_RCT_DT_DMS_KEY = d.DT_DMS_KEY "
            f"WHERE f.PRD_DMS_KEY % 100 BETWEEN 1 AND 12 AND d.DMS_DT = {anchor}"
        )
        result = _validate(sql, plan)
        assert result.ok, (result.code, result.reason)

    def test_the_repair_note_repeats_the_same_anchor(self, model):
        """A field-plan repair re-states the anchor for a calendar date; a
        repair that dropped the month rule would refuse the SQL it asks for."""
        from core.semantic_model import build_field_plan_repair_note

        receipt = next(role for role in model["date_roles"] if role["fact_column"] == "LST_RCT_DT_DMS_KEY")
        binding = _resolve(STOCK, "MART.ITM_RCT_PRD_FCT", _roles(model), confirmed_date_role=receipt)["binding"]
        plan = _plan_for(model, binding)
        found = re.search(
            r"REQUIRED ANCHOR \(copy this exact subquery as the RIGHT SIDE[^)]*\): (\(SELECT .+\))$",
            build_field_plan_repair_note(plan), re.M,
        )
        assert found, build_field_plan_repair_note(plan)
        sql = (
            "SELECT SUM(f.RCT_QTY) AS RECEIVED FROM MART.ITM_RCT_PRD_FCT f "
            "JOIN MART.DT_DMS d ON f.LST_RCT_DT_DMS_KEY = d.DT_DMS_KEY "
            f"WHERE f.PRD_DMS_KEY % 100 BETWEEN 1 AND 12 AND d.DMS_DT = {found.group(1)}"
        )
        result = _validate(sql, plan)
        assert result.ok, (result.code, result.reason)

    def test_the_period_keys_own_anchor_is_the_decoded_key_alone(self, model):
        binding = _resolve(STOCK, PERIOD_FACT, _roles(model))["binding"]
        plan = _plan_for(model, binding)
        anchor, value = _anchor_and_value(plan)
        assert anchor == f"(SELECT MAX({value}) FROM {PERIOD_FACT})"
        assert "PRD_DMS_KEY" in value

    def test_the_probe_reads_the_newest_month_row_not_a_year_row(self, model):
        """The year row was written after December's rows; the anchor is December's."""
        binding = _resolve(STOCK, PERIOD_FACT, _roles(model, BAL_TS="approved"))["binding"]
        policy = _plan_for(model, binding)["temporal_policies"][0]
        conn = sqlite3.connect(":memory:")
        conn.execute("ATTACH DATABASE ':memory:' AS MART")
        conn.execute("CREATE TABLE MART.ITM_BAL_PRD_FCT (PRD_DMS_KEY INTEGER, BAL_TS TEXT)")
        conn.executemany("INSERT INTO MART.ITM_BAL_PRD_FCT VALUES (?, ?)", [
            (202211, "2022-11-30 09:00:00"),
            (202212, "2022-12-14 10:17:36"),
            (202200, "2022-12-20 08:00:00"),
        ])
        (newest,) = conn.execute(build_anchor_probe_sql(policy, "azure_sql")).fetchone()
        assert newest == "2022-12-14 10:17:36"


class TestADateIsNeverTheMeasure:

    def test_the_bound_date_on_the_chosen_fact_is_not_the_measure(self, model):
        binding = _resolve(STOCK, PERIOD_FACT, _roles(model, BAL_TS="approved"))["binding"]
        plan = _plan_for(model, binding)
        plan["source_scope"] = {
            "status": "selected", "selected_fact": PERIOD_FACT,
            "reason": "user-confirmed governed source",
        }
        compiled = compile_analytical_request_plan(
            STOCK, plan, analytical_intent_plan={"intent": "metric_query"},
        )
        assert compiled["measures"] == []
        assert "measure" in compiled["missing_slots"]


class TestTheLiveQuestion:
    """_handle_query_impl end to end: the reader's source choice, a metric on
    the monthly fact, and a model that copies the anchor its prompt requires.
    The model and the warehouse are the boundaries."""

    def _ask(self) -> tuple[list[str], list[str], int]:
        import core.query_pipeline as qp
        from core.graph_autopopulate import auto_populate_from_schema
        from core.semantic_contract import write_contract
        from core.semantic_model import write_semantic_model
        from gateway.base import PlatformEvent

        store.init_db()
        account = f"acct{os.urandom(4).hex()}"
        store.upsert_client(account, "Test Ltd")
        schema = {fqn: meta for fqn, meta in SCHEMA.items()
                  if fqn.split(".")[-1] in {"ITM_BAL_PRD_FCT", "PRD_DMS", "WHS_DMS"}}
        tmp = tempfile.mkdtemp()
        with open(os.path.join(tmp, "_schema.json"), "w", encoding="utf-8") as handle:
            json.dump(schema, handle)
        store.update_client_state(account, "READY", {"schema_dir": tmp, "kb_dir": tmp})
        auto_populate_from_schema(account, tmp)
        write_semantic_model(schema_dir=tmp, kb_dir=tmp, account_id=account)
        store.save_metric(account, {
            "name": "Stock on hand", "formula_type": "expression", "sql_template": "SUM(CUR_ON_HND_QTY)",
            "base_table": PERIOD_FACT, "synonyms": "stock on hand, on hand", "description": "Units in stock",
        })
        write_contract(account, tmp)
        known = {fqn.split(".", 1)[1] for fqn in schema}
        columns = {name: TABLE_COLUMNS[name] for name in known}

        calls: list[str] = []
        executed: list[str] = []
        sent: list[str] = []

        async def model(system, user, *args, **kwargs):
            calls.append(system)
            anchor, value = _REQUIRED_ANCHOR.search(system).group(1, 2)
            return (
                f"SELECT SUM(CUR_ON_HND_QTY) AS STOCK_ON_HAND FROM {PERIOD_FACT} "
                f"WHERE PRD_DMS_KEY % 100 BETWEEN 1 AND 12 AND {value} = {anchor}"
            ), 10, 10

        class Result:
            def __init__(self, sql, rows):
                self.sql, self.rows, self.truncated = sql, rows, False
                self.row_obligations, self.decision, self.analysis = [], None, None

        def warehouse(credentials, db_type, sql, *args, **kwargs):
            executed.append(sql)
            if "max_business_date" in sql:
                return Result(sql, [{"max_business_date": 202212 if "[PRD_DMS_KEY]) AS" in sql
                                     else "2022-12-14 10:17:36"}])
            return Result(sql, [{"STOCK_ON_HAND": 1250.0}])

        class Retriever:
            last_retrieval_weak = False
            last_retrieval_unscored = False

            def retrieve(self, question, n=8, allowed_tables=None):
                return []

            def retrieve_fact_patterns(self, question, n=2, allowed_tables=None):
                return []

            def _is_global(self, doc):
                return False

        class Adapter:
            platform, session_id, thread_id, last_result_id = "portal", f"{account}:portal:u1", "t1", None

            async def send_message(self, event, text, **kwargs):
                sent.append(str(text))

            async def send_typing(self, *args, **kwargs):
                return None

            def add_to_history(self, **kwargs):
                return None

            def cache_result(self, *args, **kwargs):
                return None

        chosen = {"_clarification_selected_source": "source_scope",
                  "_clarification_selected_option": {"id": "source_1", "label": "Item Balance by Period",
                                                     "value": PERIOD_FACT}}
        with contextlib.ExitStack() as stack:
            for mock in (
                patch.object(qp, "get_state", return_value={"state": "READY", "schema_dir": tmp, "kb_dir": tmp}),
                patch.object(qp, "get_client_db", return_value={
                    "db_type": "azure_sql", "credentials": {}, "name": "db", "id": 1}),
                patch.object(qp.store, "get_client", return_value={
                    "account_id": account, "state": "READY", "name": "T"}),
                patch.object(qp, "load_known_tables", return_value=known),
                patch.object(qp, "load_schema_columns", return_value=columns),
                patch.object(qp, "load_retriever", return_value=Retriever()),
                patch.object(qp, "llm_complete", model),
                patch.object(qp, "resolve_provider", return_value=("azure_openai", "gpt-4o", "key", {})),
                patch.object(qp, "_log_q", lambda *args, **kwargs: None),
                patch.object(qp, "retrieve_similar_examples", return_value=[]),
                patch.object(qp, "execute_governed_query", warehouse),
            ):
                stack.enter_context(mock)
            asyncio.run(qp._handle_query_impl(
                account, PlatformEvent(account, "u1", "c1", STOCK, "portal", raw=chosen), Adapter(), STOCK,
                {"id": 1, "role": "admin", "email": "u@x.com", "name": "U", "group_name": None, "lang": "en"},
            ))
        return sent, executed, len(calls)

    def test_a_model_that_copies_the_anchor_gets_an_answer(self):
        sent, executed, model_calls = self._ask()
        assert model_calls == 1
        assert not any("could not build a trusted query" in text for text in sent), sent
        answer = executed[-1]
        assert answer.startswith("SELECT SUM(CUR_ON_HND_QTY) AS STOCK_ON_HAND")
        assert "PRD_DMS_KEY" in answer.split(" = ", 1)[1] and "BAL_TS" not in answer
        assert any("What is our total stock on hand?" in text for text in sent), sent
