import sqlglot

from core.dashboard_filters import compile_dashboard_filters


BASE_SQL = "SELECT region, month, SUM(revenue) AS revenue FROM sales GROUP BY region, month"


def test_filter_is_ast_wrapped_and_literal_is_escaped():
    attack = "East' OR 1=1 --"
    result = compile_dashboard_filters(
        BASE_SQL,
        "azure_sql",
        [{"field": "region", "type": "text", "operator": "equals"}],
        {"region": attack},
    )

    assert result.applied == ("region",)
    assert result.ignored == ()
    assert "qb_dashboard_source" in result.sql
    assert "East'' OR 1=1 --" in result.sql
    # The compiler output remains a single valid SELECT; the attempted SQL is
    # a literal value rather than a second predicate.
    assert len(sqlglot.parse(result.sql, read="tsql")) == 1


def test_filters_only_known_output_fields_and_supports_operators():
    result = compile_dashboard_filters(
        BASE_SQL,
        "azure_sql",
        [
            {"field": "region", "operator": "contains"},
            {"field": "revenue", "type": "number", "operator": "gte"},
            {"field": "invented_field", "operator": "equals"},
        ],
        {"region": "North", "revenue": "1000", "invented_field": "secret"},
    )

    assert result.applied == ("region", "revenue")
    assert result.ignored == ("invented_field",)
    assert "LIKE '%North%'" in result.sql
    assert ">= 1000" in result.sql
    assert "invented_field" not in result.sql


def test_no_active_values_returns_original_sql_unchanged():
    result = compile_dashboard_filters(
        BASE_SQL, "azure_sql", [{"field": "region"}], {"region": ""}
    )
    assert result.sql == BASE_SQL
    assert result.applied == ()
