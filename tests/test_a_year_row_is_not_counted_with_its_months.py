"""
A period fact keyed yyyymm keeps a row for each whole year beside its months.

The monthly balance fact's period key is a yyyymm integer, and beside the
twelve month rows of a year (202201 .. 202212) the warehouse stores a row for
the year itself: 202200, month part 00, holding its own total of the year. Any
query that reads both counts every year twice -- "purchases by warehouse" with
no period at all is the plainest case -- and the SQL looks entirely ordinary.

The chosen rule: month rows only, always. A year is the sum of its twelve
months and a balance the last month of the period, so the answer is right
whether or not a year row exists, including for a year still in progress.

These drive the real pieces: the policy read from a semantic model built from
a schema file, the predicate executed against rows in SQLite, the validator's
public entry point, the SQL prompt, and the answer card. Synthetic tables in a
mart's naming convention; no customer data.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest

from core.failure_messages import translate_failure
from core.i18n import activate_language, deactivate_language
from core.llm import build_sql_system_prompt
from core.period_rows import (
    attach_period_row_policies,
    month_rows_predicate,
    period_row_policies,
)
from core.result_renderer import _send_results
from core.semantic_model import MODEL_JSON, build_semantic_model
from core.validator import REPAIRABLE_REASON_CODES, validate_sql_detailed


def _table(own_key: str, *columns: tuple[str, str]) -> dict:
    return {
        "columns": [
            {"name": name, "type": dtype, "nullable": True, "comment": ""}
            for name, dtype in columns
        ],
        "pk_columns": [own_key],
        "row_count": 1000,
        "comment": "",
        "schema": "MART",
        "database": "WH",
    }


SCHEMA = {
    "WH.MART.ITM_BAL_PRD_FCT": _table(
        "ITM_BAL_PRD_FCT_KEY",
        ("ITM_BAL_PRD_FCT_KEY", "bigint"), ("ITM_DMS_KEY", "int"),
        ("WHS_DMS_KEY", "int"), ("PRD_DMS_KEY", "int"),
        ("PCH_QTY", "decimal(18,4)"), ("CUR_ON_HND_QTY", "decimal(18,4)"),
    ),
    "WH.MART.ITM_BAL_DLY_FCT": _table(
        "ITM_BAL_DLY_FCT_KEY",
        ("ITM_BAL_DLY_FCT_KEY", "bigint"), ("ITM_DMS_KEY", "int"),
        ("WHS_DMS_KEY", "int"), ("ITM_BAL_EFC_DT_DMS_KEY", "int"),
        ("ON_HND_QTY", "decimal(18,4)"),
    ),
    "WH.MART.PRD_DMS": _table(
        "PRD_DMS_KEY",
        ("PRD_DMS_KEY", "int"), ("PRD_DSC", "varchar(40)"),
        ("PRD_YR", "int"), ("PRD_MTH", "int"),
    ),
    "WH.MART.DT_DMS": _table(
        "DT_DMS_KEY",
        ("DT_DMS_KEY", "int"), ("DMS_DT", "date"), ("YR", "int"), ("MTH", "int"),
    ),
    "WH.MART.WHS_DMS": _table(
        "WHS_DMS_KEY", ("WHS_DMS_KEY", "int"), ("WHS_DSC", "varchar(60)"),
    ),
    # A dimension with a yyyymm column of its own: a date role, never a policy.
    "WH.MART.ITM_DMS": _table(
        "ITM_DMS_KEY", ("ITM_DMS_KEY", "int"), ("ITM_DSC", "varchar(60)"),
        ("LST_PCH_PRD_DMS_KEY", "int"),
    ),
}

KNOWN_TABLES = {fqn.split(".", 1)[1] for fqn in SCHEMA}
TABLE_COLUMNS = {
    fqn.split(".", 1)[1]: {column["name"]: column["type"] for column in meta["columns"]}
    for fqn, meta in SCHEMA.items()
}


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    directory = tmp_path_factory.mktemp("period_rows_model")
    (directory / "_schema.json").write_text(json.dumps(SCHEMA), encoding="utf-8")
    return build_semantic_model(str(directory))


@pytest.fixture(scope="module")
def policies(model):
    return period_row_policies(model, db_type="azure_sql")


def _validate(sql: str, policies: list[dict], db_type: str = "azure_sql"):
    return validate_sql_detailed(
        sql, KNOWN_TABLES, db_type, None, TABLE_COLUMNS,
        {"semantic_plan": {"period_row_policies": policies}},
    )


class TestThePolicyComesFromTheModel:

    def test_the_period_fact_carries_one_policy(self, policies):
        assert [(p["fact_table"].split(".")[-1], p["fact_column"]) for p in policies] == [
            ("ITM_BAL_PRD_FCT", "PRD_DMS_KEY"),
        ]

    def test_the_period_table_and_a_day_keyed_fact_carry_none(self, policies):
        tables = {p["fact_table"].split(".")[-1] for p in policies}
        assert "PRD_DMS" not in tables
        assert "ITM_BAL_DLY_FCT" not in tables

    def test_a_period_column_on_a_dimension_carries_none(self, model, policies):
        # The premise: the model does give that column a yyyymm date role.
        assert any(
            role["fact_column"] == "LST_PCH_PRD_DMS_KEY"
            and role["date_key_type"] == "yyyymm_integer"
            for role in model["date_roles"]
        )
        assert "LST_PCH_PRD_DMS_KEY" not in {p["fact_column"] for p in policies}

    def test_attaching_puts_the_policy_on_the_plan(self, model, policies):
        plan: dict = {}
        attach_period_row_policies(plan, model)
        assert plan["period_row_policies"] == policies

    def test_an_account_without_a_compiled_model_reads_its_kb(self, model, tmp_path, policies):
        (tmp_path / MODEL_JSON).write_text(json.dumps(model), encoding="utf-8")
        plan: dict = {}
        attach_period_row_policies(plan, {}, kb_dir=str(tmp_path))
        assert plan["period_row_policies"] == policies

    def test_nothing_is_attached_when_no_fact_has_a_period_key(self):
        plan: dict = {}
        attach_period_row_policies(plan, {"tables": [], "date_roles": []})
        assert "period_row_policies" not in plan


class TestThePredicateKeepsTheMonthsOnly:
    """The predicate text the prompt and the repair hand out, executed."""

    @staticmethod
    def _rows():
        connection = sqlite3.connect(":memory:")
        connection.execute("CREATE TABLE F (PRD_DMS_KEY INTEGER, PCH_QTY REAL)")
        rows = [(202200, 120.0)] + [(202200 + month, 10.0) for month in range(1, 13)]
        rows += [(202300, 15.0)] + [(202300 + month, 5.0) for month in range(1, 4)]
        connection.executemany("INSERT INTO F VALUES (?, ?)", rows)
        return connection

    def test_without_it_every_year_counts_twice(self):
        (total,) = self._rows().execute("SELECT SUM(PCH_QTY) FROM F").fetchone()
        assert total == 270.0

    def test_with_it_the_total_is_the_months(self):
        predicate = month_rows_predicate("PRD_DMS_KEY", "azure_sql")
        (total,) = self._rows().execute(
            f"SELECT SUM(PCH_QTY) FROM F WHERE {predicate}").fetchone()
        assert total == 135.0

    def test_a_year_read_from_its_months_equals_the_year_row(self):
        connection = self._rows()
        (months,) = connection.execute(
            "SELECT SUM(PCH_QTY) FROM F WHERE PRD_DMS_KEY BETWEEN 202201 AND 202212"
        ).fetchone()
        (year_row,) = connection.execute(
            "SELECT PCH_QTY FROM F WHERE PRD_DMS_KEY = 202200").fetchone()
        assert months == year_row

    def test_oracle_spells_it_with_mod(self):
        assert month_rows_predicate("f.PRD_DMS_KEY", "oracle") == (
            "MOD(f.PRD_DMS_KEY, 100) BETWEEN 1 AND 12")


FACT = "MART.ITM_BAL_PRD_FCT"
DECODED = "TRY_CONVERT(date, CONVERT(varchar(6), f.PRD_DMS_KEY) + '01', 112)"

ACCEPTED = {
    "the month-row predicate": (
        f"SELECT w.WHS_DSC, SUM(f.PCH_QTY) AS PCH FROM {FACT} f "
        "JOIN MART.WHS_DMS w ON f.WHS_DMS_KEY = w.WHS_DMS_KEY "
        "WHERE f.PRD_DMS_KEY % 100 BETWEEN 1 AND 12 GROUP BY w.WHS_DSC"),
    "a filter on the decoded period date": (
        f"SELECT SUM(f.PCH_QTY) AS PCH FROM {FACT} f WHERE {DECODED} >= '2022-01-01'"),
    "the months of one year": (
        f"SELECT SUM(f.PCH_QTY) AS PCH FROM {FACT} f "
        "WHERE f.PRD_DMS_KEY BETWEEN 202201 AND 202212"),
    "the predicate in the fact's own join": (
        "SELECT w.WHS_DSC, SUM(f.PCH_QTY) AS PCH FROM MART.WHS_DMS w "
        f"JOIN {FACT} f ON f.WHS_DMS_KEY = w.WHS_DMS_KEY "
        "AND f.PRD_DMS_KEY % 100 BETWEEN 1 AND 12 GROUP BY w.WHS_DSC"),
    "a CTE that keeps month rows": (
        f"WITH m AS (SELECT f.WHS_DMS_KEY, SUM(f.PCH_QTY) AS PCH FROM {FACT} f "
        "WHERE f.PRD_DMS_KEY % 100 > 0 GROUP BY f.WHS_DMS_KEY) "
        "SELECT SUM(m.PCH) AS PCH FROM m"),
    "two named periods, each on the decoded date": (
        f"SELECT SUM(CASE WHEN {DECODED} >= '2022-03-01' AND {DECODED} < '2022-04-01' "
        "THEN f.PCH_QTY ELSE 0 END) AS PCH_MAR, "
        f"SUM(CASE WHEN {DECODED} >= '2022-04-01' AND {DECODED} < '2022-05-01' "
        f"THEN f.PCH_QTY ELSE 0 END) AS PCH_APR FROM {FACT} f "
        f"WHERE ({DECODED} >= '2022-03-01' AND {DECODED} < '2022-04-01') "
        f"OR ({DECODED} >= '2022-04-01' AND {DECODED} < '2022-05-01')"),
    "the latest month anchored on the decoded date": (
        f"SELECT SUM(f.CUR_ON_HND_QTY) AS OH FROM {FACT} f WHERE {DECODED} = "
        "(SELECT MAX(TRY_CONVERT(date, CONVERT(varchar(6), PRD_DMS_KEY) + '01', 112)) "
        f"FROM {FACT})"),
}

REFUSED = {
    "no period at all": f"SELECT SUM(f.PCH_QTY) AS PCH FROM {FACT} f",
    "the predicate on one side of an OR": (
        f"SELECT SUM(f.PCH_QTY) AS PCH FROM {FACT} f "
        "WHERE f.PRD_DMS_KEY % 100 BETWEEN 1 AND 12 OR f.WHS_DMS_KEY = 3"),
    "an OR whose second period is the year row": (
        f"SELECT SUM(f.PCH_QTY) AS PCH FROM {FACT} f "
        f"WHERE ({DECODED} >= '2022-03-01') OR (f.PRD_DMS_KEY = 202200)"),
    "a range across a year boundary": (
        f"SELECT SUM(f.PCH_QTY) AS PCH FROM {FACT} f "
        "WHERE f.PRD_DMS_KEY BETWEEN 202112 AND 202203"),
    "the year row itself": (
        f"SELECT SUM(f.PCH_QTY) AS PCH FROM {FACT} f WHERE f.PRD_DMS_KEY = 202200"),
    "a decoded date put back with ISNULL": (
        f"SELECT SUM(f.PCH_QTY) AS PCH FROM {FACT} f "
        f"WHERE ISNULL({DECODED}, '2022-06-01') >= '2022-01-01'"),
    "a filter only outside the CTE that summed": (
        f"WITH m AS (SELECT f.PRD_DMS_KEY, SUM(f.PCH_QTY) AS PCH FROM {FACT} f "
        "GROUP BY f.PRD_DMS_KEY) SELECT SUM(m.PCH) AS PCH FROM m "
        "WHERE m.PRD_DMS_KEY % 100 BETWEEN 1 AND 12"),
    "an anchor on the raw key, which a year row can win": (
        f"SELECT SUM(f.CUR_ON_HND_QTY) AS OH FROM {FACT} f "
        "WHERE f.PRD_DMS_KEY % 100 BETWEEN 1 AND 12 AND f.PRD_DMS_KEY = "
        f"(SELECT MAX(PRD_DMS_KEY) FROM {FACT})"),
    "the period table's key filtered instead of the fact's": (
        f"SELECT SUM(f.PCH_QTY) AS PCH FROM {FACT} f "
        "JOIN MART.PRD_DMS p ON f.PRD_DMS_KEY = p.PRD_DMS_KEY "
        "WHERE p.PRD_DMS_KEY % 100 BETWEEN 1 AND 12"),
}


class TestTheValidatorRefusesAYearRowRead:

    @pytest.mark.parametrize("shape", sorted(ACCEPTED))
    def test_a_month_row_read_is_accepted(self, policies, shape):
        result = _validate(ACCEPTED[shape], policies)
        assert result.ok, (shape, result.code, result.reason)

    @pytest.mark.parametrize("shape", sorted(REFUSED))
    def test_a_read_that_can_include_the_year_row_is_refused(self, policies, shape):
        result = _validate(REFUSED[shape], policies)
        assert not result.ok and result.code == "period_rows_mixed", (shape, result.reason)

    def test_the_refusal_names_the_exact_predicate(self, policies):
        result = _validate(REFUSED["no period at all"], policies)
        assert result.errors[0]["required_predicate"] == "f.PRD_DMS_KEY % 100 BETWEEN 1 AND 12"

    def test_a_fact_without_a_period_key_is_untouched(self, policies):
        result = _validate(
            "SELECT SUM(d.ON_HND_QTY) AS OH FROM MART.ITM_BAL_DLY_FCT d", policies)
        assert result.ok, (result.code, result.reason)

    def test_without_a_policy_nothing_is_enforced(self):
        result = _validate(REFUSED["no period at all"], [])
        assert result.ok, (result.code, result.reason)

    def test_oracle_is_refused_and_told_to_use_mod(self, model):
        oracle = period_row_policies(model, db_type="oracle")
        refused = _validate(REFUSED["no period at all"], oracle, "oracle")
        assert refused.code == "period_rows_mixed"
        assert refused.errors[0]["required_predicate"] == (
            "MOD(f.PRD_DMS_KEY, 100) BETWEEN 1 AND 12")
        accepted = _validate(
            f"SELECT SUM(f.PCH_QTY) AS PCH FROM {FACT} f "
            "WHERE MOD(f.PRD_DMS_KEY, 100) BETWEEN 1 AND 12", oracle, "oracle")
        assert accepted.ok, (accepted.code, accepted.reason)

    def test_the_refusal_is_repaired_not_final(self):
        assert "period_rows_mixed" in REPAIRABLE_REASON_CODES


class TestThePromptStatesTheRule:

    def _prompt(self, policies, context):
        return build_sql_system_prompt(
            "azure_sql", context,
            semantic_plan={"period_row_policies": policies},
            question="purchases by warehouse",
        )

    def test_it_is_stated_when_the_fact_is_in_context_even_without_a_field_plan(self, policies):
        prompt = self._prompt(policies, f"## Table: {FACT}\n| `PCH_QTY` | decimal |")
        block = prompt[prompt.index("## Period rows"):]
        assert "PRD_DMS_KEY % 100 BETWEEN 1 AND 12" in block
        assert FACT in block

    def test_it_is_left_out_when_the_fact_is_not(self, policies):
        prompt = self._prompt(policies, "## Table: MART.ITM_BAL_DLY_FCT")
        assert "## Period rows" not in prompt


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
            {}, adapter, "purchases by warehouse", [{"WHS_DSC": "North", "PCH": 135.0}],
            sql, 900, None, 1, {"id": 1, "db_type": "azure_sql"},
            question_id=None,
            confidence_context={"semantic_plan": {"period_row_policies": policies}},
        ))
    finally:
        deactivate_language(token)
    return "\n".join(adapter.sent)


class TestTheReaderIsTold:

    def test_the_card_says_the_year_rows_were_left_out(self, policies):
        text = _card(policies, ACCEPTED["the month-row predicate"])
        assert "monthly rows only" in text

    def test_in_french_too(self, policies):
        text = _card(policies, ACCEPTED["the month-row predicate"], "fr")
        assert "seules lignes mensuelles" in text

    def test_an_answer_from_another_fact_says_nothing(self, policies):
        text = _card(policies, "SELECT SUM(d.ON_HND_QTY) AS OH FROM MART.ITM_BAL_DLY_FCT d")
        assert "monthly rows only" not in text

    @pytest.mark.parametrize("lang, phrase", [("en", "twice"), ("fr", "deux fois")])
    def test_a_refusal_that_survived_every_repair_explains_itself(self, lang, phrase):
        token = activate_language(lang)
        try:
            card = translate_failure(kind="validation", code="period_rows_mixed")
        finally:
            deactivate_language(token)
        assert phrase in card["most_likely_reason"]
        assert card["suggested_next_step"]


class TestTheRealPipeline:
    """_handle_query_impl, end to end with the model and the warehouse stubbed.

    The pieces above can all be right and the rule still never applied: the
    policy is attached in one line of a very long function, on the path every
    generated query takes. This drives that function for a question that names
    no period at all -- the one that doubled -- and proves the policy was
    attached, the model's first query was refused and repaired, and only the
    month-row query ever reached the warehouse.
    """

    FIRST = (f"SELECT w.WHS_DSC, SUM(f.PCH_QTY) AS PCH FROM {FACT} f "
             "JOIN MART.WHS_DMS w ON f.WHS_DMS_KEY = w.WHS_DMS_KEY GROUP BY w.WHS_DSC")
    REPAIRED = (f"SELECT w.WHS_DSC, SUM(f.PCH_QTY) AS PCH FROM {FACT} f "
                "JOIN MART.WHS_DMS w ON f.WHS_DMS_KEY = w.WHS_DMS_KEY "
                "WHERE f.PRD_DMS_KEY % 100 BETWEEN 1 AND 12 GROUP BY w.WHS_DSC")

    def test_a_question_with_no_period_reads_month_rows_only(self, model, tmp_path):
        import contextlib
        from unittest.mock import patch

        import core.query_pipeline as qp
        import store.db as _db
        from gateway.base import PlatformEvent

        (tmp_path / MODEL_JSON).write_text(json.dumps(model), encoding="utf-8")
        seen: dict = {"prompts": [], "executed": [], "card": None}

        class Retriever:
            last_retrieval_weak = False
            last_retrieval_unscored = False

            def retrieve(self, question, n=8, allowed_tables=None):
                return [f"# {FACT}\n| `PRD_DMS_KEY` | int |\n| `PCH_QTY` | decimal |"]

            def retrieve_fact_patterns(self, question, n=2, allowed_tables=None):
                return []

            def _is_global(self, doc):
                return False

        async def model_reply(system, user, *args, **kwargs):
            seen["prompts"].append((system, user))
            if "PERIOD-ROW REPAIR REQUIRED" in user:
                return self.REPAIRED, 1, 1
            return self.FIRST, 1, 1

        class Governed:
            def __init__(self, sql):
                self.sql, self.rows = sql, [{"WHS_DSC": "North", "PCH": 135.0}]

        def execute(credentials, db_type, sql, **kwargs):
            seen["executed"].append(sql)
            return Governed(sql)

        async def card(event, adapter, question, rows, sql, *args, **kwargs):
            seen["card"] = (sql, (kwargs.get("confidence_context") or {}).get("semantic_plan"))

        class Adapter:
            platform, session_id, thread_id, last_result_id = "portal", "acct-pr:portal:u1", "t1", None

            async def send_message(self, event, text, **kwargs):
                return None

            async def send_typing(self, *args, **kwargs):
                return None

            def add_to_history(self, **kwargs):
                return None

            def cache_result(self, *args, **kwargs):
                return None

        _db.init_db()
        question = "purchases by warehouse"
        with contextlib.ExitStack() as stack:
            for mock in (
                patch.object(qp, "get_state", return_value={
                    "state": "READY", "schema_dir": str(tmp_path), "kb_dir": str(tmp_path)}),
                patch.object(qp, "get_client_db", return_value={
                    "db_type": "azure_sql", "credentials": {}, "name": "db", "id": 1}),
                patch.object(qp.store, "get_client", return_value={
                    "account_id": "acct-pr", "state": "READY", "name": "T"}),
                patch.object(qp, "load_known_tables", return_value=set(KNOWN_TABLES)),
                patch.object(qp, "load_schema_columns", return_value=TABLE_COLUMNS),
                patch.object(qp, "load_retriever", return_value=Retriever()),
                patch.object(qp, "llm_complete", model_reply),
                patch.object(qp, "_send_results", card),
                patch.object(qp, "execute_governed_query", execute),
                patch.object(qp, "resolve_provider",
                             return_value=("anthropic", "claude", "key", {})),
                patch.object(qp, "_log_q", lambda *args, **kwargs: None),
                # Few-shot retrieval reaches a vector store; none is needed here.
                patch.object(qp, "retrieve_similar_examples", return_value=[]),
            ):
                stack.enter_context(mock)
            asyncio.run(qp._handle_query_impl(
                "acct-pr", PlatformEvent("acct-pr", "u1", "c1", question, "portal"),
                Adapter(), question,
                {"id": 1, "role": "admin", "email": "u@x.com", "name": "U", "group_name": None},
            ))

        assert any("## Period rows" in system for system, _user in seen["prompts"])
        assert any("PERIOD-ROW REPAIR REQUIRED" in user for _system, user in seen["prompts"])
        assert seen["executed"] == [self.REPAIRED]
        sql, plan = seen["card"]
        assert sql == self.REPAIRED
        assert [p["fact_column"] for p in plan["period_row_policies"]] == ["PRD_DMS_KEY"]
