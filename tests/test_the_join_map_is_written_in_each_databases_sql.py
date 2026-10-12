"""The join map's SQL is written the way the connected database writes it.

The knowledge base's _join_map.md shows the SQL model each join as SQL to copy. It wrote Azure SQL's brackets
for every database -- `LEFT JOIN [DIM_STORE] ON [FCT_SALES].STORE_KEY = ...` -- which Snowflake, Oracle and
PostgreSQL refuse, and which the validator then had to take out again. Each database's joins are now written
in its own SQL: a table by its own name, quoted only where the database needs it (a PostgreSQL name with
capitals). Azure SQL's are as they were.

Synthetic tables; every join line the document teaches is parsed in the database's dialect.
"""

from __future__ import annotations

import re

import pytest
import sqlglot

from core.schema import _build_join_map
from tests.test_the_join_map_teaches_the_graphs_joins import MASTER


def _sql_lines(doc: str) -> list[str]:
    blocks = re.findall(r"```sql\n(.*?)```", doc, re.S)
    return [line.strip() for block in blocks for line in block.splitlines() if "JOIN" in line]


def _postgres_master() -> dict:
    """The retail shape as PostgreSQL keeps it when it was created with capitals."""
    def table(key, *cols):
        return {"columns": [{"name": c, "type": "integer", "nullable": False, "comment": ""} for c in cols],
                "pk_columns": [key], "row_count": 100, "comment": "", "schema": "sales", "database": "shop"}
    return {
        "SHOP.SALES.FactOrderLines": table("OrderLineKey", "OrderLineKey", "StoreKey", "OrderDateKey", "NetAmount"),
        "SHOP.SALES.DimStores": table("StoreKey", "StoreKey", "StoreName"),
        "SHOP.SALES.DimCalendar": table("DateKey", "DateKey", "CalendarDate"),
        "__db_fk_constraints__": [
            {"constraint_name": "fk_store", "parent_schema": "sales", "parent_table": "FactOrderLines",
             "parent_col": "StoreKey", "ref_schema": "sales", "ref_table": "DimStores", "ref_col": "StoreKey",
             "ordinal": 1, "enforced": True, "source": "postgres"},
        ],
    }


@pytest.mark.parametrize("db_type, dialect", [("snowflake", "snowflake"), ("oracle", "oracle"),
                                              ("postgres", "postgres")])
def test_every_join_parses_in_the_databases_own_sql_and_carries_no_bracket(db_type, dialect):
    lines = _sql_lines(_build_join_map(MASTER, db_type))
    assert lines
    for line in lines:
        assert "[" not in line, line
        sqlglot.parse_one(f"SELECT 1 FROM t {line}", read=dialect)


def test_azure_sql_keeps_its_brackets():
    assert _build_join_map(MASTER) == _build_join_map(MASTER, "azure_sql")
    lines = _sql_lines(_build_join_map(MASTER))
    assert lines and all("[" in line for line in lines)


def test_a_postgres_name_with_capitals_is_quoted_and_one_in_small_letters_is_not():
    lines = _sql_lines(_build_join_map(_postgres_master(), "postgres"))
    key = next(line for line in lines if '"DimStores"' in line and "StoreKey" in line)
    assert '"FactOrderLines"."StoreKey" = "DimStores"."StoreKey"' in key
    assert "SALES" not in key and "SHOP" not in key          # a table by its own name, not the key's parts
    for line in lines:
        sqlglot.parse_one(f"SELECT 1 FROM t {line}", read="postgres")
