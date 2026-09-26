"""
A date key is read the way the semantic plan declares it: joined, or decoded.

The SQL prompt's DATE-KEY RULE said every _DT_DMS_KEY / _DATE_DMS_KEY column
IS a YYYYMMDD integer, to be decoded with TRY_CONVERT, and that this
OVERRIDES the rules around it -- while the semantic plan for the same
question declared the key a surrogate into the date dimension, and the join
path said to join it. On a mart whose keys are surrogates (4067, not
20260315) the decode is NULL on every row: an empty answer, and no error.
The prompt could not have told the two apart anyway: it read a key's
declared type only from the plan's fields, where a date bound through the
dimension names the dimension's date column, so a declared surrogate looked
undeclared.

Now the rule is written per key, from the plan's date-key policies: a
surrogate is joined and never decoded, a YYYYMMDD integer is decoded, and a
key the plan does not declare joins its date dimension when the schema has
one -- right whatever its encoding -- and is decoded only when there is none.

The real prompt builder, with plans built by the real date-plan builder.
"""

from __future__ import annotations

from core.contextual_dates import build_contextual_date_plan
from core.llm import build_sql_system_prompt
from core.pipeline_context import _merge_semantic_plans

DATES = "Table MART.DT_DMS: DT_DMS_KEY int, CAL_DT date, YR int"
ONE_KEY = f"Table MART.SLS_FCT: ORD_DT_DMS_KEY int, NET_AMT decimal\n{DATES}"
TWO_KEYS = f"Table MART.SLS_FCT: ORD_DT_DMS_KEY int, SHP_DT_DMS_KEY int, NET_AMT decimal\n{DATES}"


def _binding(column, key_type, role):
    joined = key_type == "surrogate_fk"
    return {"fact_table": "MART.SLS_FCT", "fact_column": column,
            "dimension_table": "MART.DT_DMS" if joined else "",
            "dimension_key": "DT_DMS_KEY" if joined else "",
            "date_value_column": "CAL_DT" if joined else column, "date_key_type": key_type,
            "date_role": role, "context_name": role, "governance_status": "approved",
            "resolution_source": "metric_default"}


ORDER_JOINED = _binding("ORD_DT_DMS_KEY", "surrogate_fk", "Order Date")
ORDER_ENCODED = _binding("ORD_DT_DMS_KEY", "yyyymmdd_integer", "Order Date")
SHIP_ENCODED = _binding("SHP_DT_DMS_KEY", "yyyymmdd_integer", "Ship Date")
QUESTION = "net sales by month for 2025"


def _date_key_rule(*bindings, kb=ONE_KEY):
    """The DATE-KEY RULE the model is given, up to the next rule."""
    plan = {}
    for binding in bindings:
        plan = _merge_semantic_plans(plan, build_contextual_date_plan(binding, QUESTION))
    prompt = build_sql_system_prompt("azure_sql", kb, semantic_plan=plan or None, question=QUESTION)
    start = prompt.index("- AZURE SQL DATE-KEY RULE:")
    end = prompt.find("\n- ", start + 1)
    return prompt[start:end if end > 0 else None]


class TestADeclaredSurrogateKey:

    def test_is_joined_to_the_date_dimension(self):
        rule = _date_key_rule(ORDER_JOINED)
        assert "Surrogate keys into the date dimension (ORD_DT_DMS_KEY)" in rule
        assert "JOIN the date dimension on the key" in rule

    def test_is_never_decoded(self):
        rule = _date_key_rule(ORDER_JOINED)
        assert "TRY_CONVERT" not in rule
        assert "never decode the key" in rule


class TestADeclaredYyyymmddKey:

    def test_is_decoded(self):
        rule = _date_key_rule(ORDER_ENCODED)
        assert ("YYYYMMDD integers (ORD_DT_DMS_KEY): decode it with "
                "TRY_CONVERT(date, CONVERT(varchar(8), alias.KEY), 112)") in rule

    def test_is_not_sent_to_the_dimension(self):
        assert "Surrogate keys into the date dimension" not in _date_key_rule(ORDER_ENCODED)


class TestTwoKeysDeclaredDifferently:

    def test_each_is_read_its_own_way(self):
        rule = _date_key_rule(ORDER_JOINED, SHIP_ENCODED, kb=TWO_KEYS)
        assert "Surrogate keys into the date dimension (ORD_DT_DMS_KEY)" in rule
        assert "YYYYMMDD integers (SHP_DT_DMS_KEY)" in rule


class TestAKeyThePlanDoesNotDeclare:

    def test_joins_its_dimension_when_there_is_one_and_is_decoded_only_otherwise(self):
        rule = _date_key_rule(kb=TWO_KEYS)
        assert "Not declared (ORD_DT_DMS_KEY, SHP_DT_DMS_KEY): when the schema has a date dimension" in rule
        assert "JOIN it" in rule
        assert "only when there is none, decode it" in rule

    def test_beside_a_declared_one_is_named_apart(self):
        rule = _date_key_rule(ORDER_JOINED, kb=TWO_KEYS)
        assert "Not declared (SHP_DT_DMS_KEY)" in rule
        assert "Not declared (ORD_DT_DMS_KEY" not in rule
