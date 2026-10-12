"""The rules every database's SQL prompt shares show their examples in that database's SQL.

The conditional-aggregation and ranking rules showed `FROM [schema].[table]`, and the null-aware aggregate
rule COUNT_BIG(*) and bracketed aliases, in the Snowflake, Oracle and PostgreSQL prompts too: SQL those
databases refuse, set before the model as the pattern to follow. Azure SQL's prompt is as it was.
"""

from __future__ import annotations

import re

import pytest

from core.llm import build_sql_system_prompt


def _prompt(db_type: str) -> str:
    return build_sql_system_prompt(db_type, "TABLE CONTEXT")


def _rule(prompt: str, heading: str) -> str:
    return prompt.split(heading, 1)[1].split("\n- ", 1)[0]


@pytest.mark.parametrize("db_type", ["snowflake", "oracle", "postgres"])
def test_the_shared_rules_show_no_azure_sql_only_syntax(db_type):
    prompt = _prompt(db_type)
    for heading in ("- CONDITIONAL AGGREGATION RULE:", "- RANKING RULE:", "- NULL-AWARE FILTERED AGGREGATE RULE"):
        rule = _rule(prompt, heading)
        assert not re.search(r"\[\w+\]", rule), (heading, rule)
        assert "COUNT_BIG" not in rule, heading
    assert "FROM schema.table" in _rule(prompt, "- RANKING RULE:")


def test_postgres_is_told_a_week_back_in_its_own_sql_and_named_for_the_year_pattern():
    prompt = _prompt("postgres")
    assert "INTERVAL '7 days'" in _rule(prompt, "- NULL-AWARE FILTERED AGGREGATE RULE")
    assert "PostgreSQL: same pattern with dialect date functions." in prompt


def test_azure_sql_keeps_its_own_examples():
    prompt = _prompt("azure_sql")
    assert "FROM [schema].[table]" in _rule(prompt, "- RANKING RULE:")
    assert "COUNT_BIG(*) AS [MatchedRows] on Azure SQL" in prompt
    assert "Snowflake / Oracle: same pattern with dialect date functions." in prompt
