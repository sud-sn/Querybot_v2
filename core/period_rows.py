"""A period key whose month part is 00 holds the year's total, not a month.

A monthly period fact keyed yyyymm often carries, beside the twelve month rows
of a year, a row for the year itself: key 202200, month part 00, holding the
warehouse's own total of that year. Read together with the months, every year
is counted twice. "Purchases by warehouse" doubles, "units sold in 2022"
doubles, and nothing in the SQL looks wrong.

One grain per question, and the grain is the month. Year, month, trend and
all-time questions all read month rows: a year is the sum of its months, and a
balance is the last month of the period. Month rows are exact whether or not a
year row exists, including for a year still in progress, whose year row the
warehouse may not have written yet.

A decoded period date already leaves month 00 out -- 20220001 is not a date,
so it decodes to NULL -- which is why questions filtered on the decoded date
were safe. A question with no period at all, or SQL that read the raw key,
was not. The rule is therefore enforced where every generated query passes:
the prompt states the predicate, the validator refuses a read of the fact that
lacks it, and the answer card says the year rows were left out.

Tenant-neutral: the policies come from the semantic model's date roles, for
any fact whose period key is a yyyymm integer.
"""

from __future__ import annotations

import re
from typing import Any

from core.date_roles import normalize_date_key_type

PERIOD_ROW_KEY_TYPES = frozenset({"yyyymm_integer"})


def month_rows_predicate(column_ref: str, db_type: str = "azure_sql") -> str:
    """The predicate that keeps month rows (01-12) and drops the year row (00)."""
    if str(db_type or "").lower() == "oracle":
        return f"MOD({column_ref}, 100) BETWEEN 1 AND 12"
    return f"{column_ref} % 100 BETWEEN 1 AND 12"


def is_month_period_value(value: Any) -> bool:
    """True for a yyyymm integer whose month part is 01-12."""
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return False
    return number >= 100 and 1 <= number % 100 <= 12


def _bare(name: str) -> str:
    return str(name or "").split(".")[-1].strip('[]"`').upper()


def _fact_tables(model: dict) -> tuple[set[str], set[str]]:
    qualified: set[str] = set()
    bare: set[str] = set()
    for table in model.get("tables") or []:
        if not isinstance(table, dict) or str(table.get("type") or "").lower() != "fact":
            continue
        for key in ("qualified_name", "fqn"):
            value = str(table.get(key) or "").upper()
            if value:
                qualified.add(value)
        name = _bare(str(table.get("table") or table.get("qualified_name") or ""))
        if name:
            bare.add(name)
    return qualified, bare


def period_row_policies(model: dict | None, *, db_type: str = "azure_sql") -> list[dict]:
    """One policy per fact column that is a yyyymm period key.

    Read from the compiled semantic model: its date roles say which columns
    are yyyymm periods, and its tables say which tables are facts. A period
    key on a dimension (the period table itself) carries no policy -- nothing
    there is summed.
    """
    model = model or {}
    qualified, bare = _fact_tables(model)
    policies: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for role in model.get("date_roles") or []:
        if not isinstance(role, dict):
            continue
        if normalize_date_key_type(str(role.get("date_key_type") or "")) not in PERIOD_ROW_KEY_TYPES:
            continue
        if str(role.get("status") or "").lower() == "rejected":
            continue
        table = str(role.get("fact_table") or "")
        column = str(role.get("fact_column") or "")
        if not table or not column:
            continue
        if table.upper() not in qualified and _bare(table) not in bare:
            continue
        key = (table.upper(), column.upper())
        if key in seen:
            continue
        seen.add(key)
        policies.append({
            "kind": "period_rows",
            "fact_table": table,
            "fact_column": column,
            "business_role": str(role.get("name") or ""),
            "predicate": month_rows_predicate(f"{table}.{column}", db_type),
        })
    return policies


def attach_period_row_policies(
    semantic_plan: dict | None,
    model: dict | None,
    *,
    kb_dir: str = "",
    db_type: str = "azure_sql",
) -> list[dict]:
    """Put the policies on the plan that the prompt and the validator both read.

    One plan object travels from generation through every repair to the answer
    card, so attaching here is what makes the rule hold for all of them. The
    compiled contract's model is preferred; an account whose contract has no
    model yet is read from its knowledge-base directory instead.
    """
    if not isinstance(semantic_plan, dict):
        return []
    if not model and kb_dir:
        from core.semantic_model import load_semantic_model

        model = load_semantic_model(kb_dir)
    policies = period_row_policies(model, db_type=db_type)
    if policies:
        semantic_plan["period_row_policies"] = policies
    return policies


def policies_in_scope(policies: list[dict] | None, *texts: str) -> list[dict]:
    """The policies whose fact table is named anywhere in ``texts``."""
    haystack = " ".join(str(text or "") for text in texts).upper()
    return [
        policy for policy in policies or []
        if isinstance(policy, dict) and _bare(str(policy.get("fact_table") or ""))
        and _bare(str(policy.get("fact_table") or "")) in haystack
    ]


def format_period_row_rules(policies: list[dict] | None, db_type: str = "azure_sql") -> str:
    """The prompt block that states the rule for each fact in scope."""
    lines: list[str] = []
    for policy in policies or []:
        table = str(policy.get("fact_table") or "")
        column = str(policy.get("fact_column") or "")
        if not table or not column:
            continue
        lines.append(
            f"- {table}: {column} is a yyyymm period. A value ending in 00 is the "
            "warehouse's total for the whole year, stored beside that year's twelve "
            "month rows, so reading both counts every year twice. Every SELECT that "
            f"reads {table} -- including each CTE and subquery -- MUST keep month "
            f"rows only: WHERE {month_rows_predicate(f'<alias>.{column}', db_type)} "
            "(with the alias you gave the table). A year is the sum of its twelve "
            "months; a stock balance is the last month of the period."
        )
    if not lines:
        return ""
    return "\n".join(["## Period rows — REQUIRED", *lines])


def sql_reads_policy_table(sql: str, policy: dict) -> bool:
    """Whether ``sql`` names the policy's fact table (bare name, any quoting)."""
    name = _bare(str(policy.get("fact_table") or ""))
    if not name:
        return False
    return bool(re.search(
        rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", str(sql or ""), re.I,
    ))
