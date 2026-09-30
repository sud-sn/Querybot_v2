"""
A member the question names filters the answer.

"Sales of Climbing products in 2025", "sales of Summit Tent in 2025", "how
many products are in Climbing?" -- a question that names a member was left
to the model, every time: the governed compilers wrote no filter on a value,
so they stood aside for the planner, which is told the value the index
found. Where the planner is a stand-in, or strays, the question named the
one thing it was about and was answered by guesswork.

A member the value index finds as the value of one column -- or of a label
and its twin in another language, "Camping" in a category's English and
French names -- is now written as a filter by the grouped compiler, its
table joined along the governed path as a breakdown's is. A single total
filtered so is told apart from no rows: beside it, how many rows matched
and how many held a value, and a total of none is 0. A breakdown by the
member's own column is the member ("sales by product for the Climbing
category"), and a population the member narrows is counted on its own
table: "how many products are in Climbing" counts the category's products,
not the products sold there. A member named with the period beside it --
"Summit Tent 2025" -- is the member. Two members of one column, a value the
index found in two columns, or one it could not find whole are still the
planner's.

tests/star_harness.py keeps eight products in six subcategories of three
categories, named in English and French, and two years of sales.
"""

from __future__ import annotations

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("member-filter")) as built:
        yield built


def _category(product: tuple, lang: int = 0) -> str:
    return star.CATEGORIES[star.SUBCATEGORIES[product[3]][2]][lang]


def _sales_2025(keep, name: int = 1) -> dict[str, float]:
    """2025 sales of the products ``keep`` keeps, by name."""
    sales: dict[str, float] = {}
    for line in star.orders():
        product = star.PRODUCTS[line[4]]
        if line[2].year == 2025 and keep(product):
            sales[product[name]] = sales.get(product[name], 0) + line[7] * product[4]
    return sales


def _by_product(answer: dict) -> dict[str, float]:
    assert answer["model_wrote_sql"] is False
    return {row["PRODUCT"]: row["SALES_AMOUNT"] for row in answer["rows"]}


class TestTheProductAnswers:

    def test_one_products_sales(self, warehouse):
        answer = star.ask(warehouse, "Sales of Summit Tent in 2025")
        assert answer["model_wrote_sql"] is False
        assert [row["SALES_AMOUNT"] for row in answer["rows"]] == list(
            _sales_2025(lambda product: product[1] == "Summit Tent").values())

    def test_one_product_beside_its_larger_sibling(self, warehouse):
        # The index holds a "Summit Tent XL" as well: the question names one.
        import sqlite3

        from core.value_index import _index_path, normalize_value

        path = _index_path(star.ACCOUNT)
        with sqlite3.connect(path) as conn:
            (table, column, business_name) = conn.execute(
                "SELECT table_fqn, column_name, business_name FROM column_value WHERE value = 'Summit Tent'").fetchone()
            conn.execute("INSERT INTO column_value (table_fqn, column_name, business_name, value, value_norm) "
                         "VALUES (?, ?, ?, 'Summit Tent XL', ?)",
                         (table, column, business_name, normalize_value("Summit Tent XL")))
        try:
            # Not the question another test asks: a plan is reused for the same words.
            answer = star.ask(warehouse, "Sales for Summit Tent in 2025")
        finally:
            with sqlite3.connect(path) as conn:
                conn.execute("DELETE FROM column_value WHERE value = 'Summit Tent XL'")
        assert answer["model_wrote_sql"] is False
        assert [row["SALES_AMOUNT"] for row in answer["rows"]] == list(
            _sales_2025(lambda product: product[1] == "Summit Tent").values())

    @pytest.mark.parametrize("question", [
        "Sales of Climbing products in 2025",
        "Sales by product for the Climbing category in 2025",
    ])
    def test_a_categorys_products(self, warehouse, question):
        assert _by_product(star.ask(warehouse, question)) == _sales_2025(
            lambda product: _category(product) == "Climbing")

    def test_a_member_its_french_label_keeps_too(self, warehouse):
        assert _by_product(star.ask(warehouse, "Sales of Camping products in 2025")) == _sales_2025(
            lambda product: _category(product) == "Camping")

    def test_in_french(self, warehouse):
        assert _by_product(star.ask(warehouse, "Ventes des produits Escalade en 2025", "fr")) == _sales_2025(
            lambda product: _category(product, 1) == "Escalade", name=2)

    @pytest.mark.parametrize("question", [
        "How many products are in Climbing?",
        "How many products are in the Climbing category?",
    ])
    def test_the_products_a_category_keeps(self, warehouse, question):
        answer = star.ask(warehouse, question)
        assert answer["model_wrote_sql"] is False
        assert "FACTINTERNETSALES" not in answer["sql"].upper()
        assert [row["PRODUCT_COUNT"] for row in answer["rows"]] == [
            sum(1 for product in star.PRODUCTS.values() if _category(product) == "Climbing")]

    @pytest.mark.parametrize("question", [
        # A country the territory's region and country both keep.
        "Sales in France in 2025",
        # Two products, either or both.
        "Sales of Summit Tent and Ridge Tent in 2025",
    ])
    def test_still_the_planners(self, warehouse, question):
        assert star.ask(warehouse, question)["model_wrote_sql"] is True


def _context(named: list[str], verified: list[dict]) -> dict:
    return {"named_members": named, "verified_members": verified}


_CLIMBING = {"phrase": "Climbing", "table": "dbo.DimProductCategory", "column": "EnglishProductCategoryName",
             "value": "Climbing"}
_TENT = {"phrase": "Summit Tent", "table": "dbo.DimProduct", "column": "EnglishProductName", "value": "Summit Tent"}
_RIDGE = {"phrase": "Ridge Tent", "table": "dbo.DimProduct", "column": "EnglishProductName", "value": "Ridge Tent"}


class TestTheResolverAndThePeriod:
    """The value index finds each size of a fitting; the words beside the
    period decide which of them the question names."""

    _SIZES = ["Copper Tube 20", "Copper Tee 20", "Copper Cap 20", "Copper Elbow 20", "Copper Union 20",
              "Copper Plug 20", "Copper Nipple 20"]

    @pytest.fixture
    def fittings(self, warehouse):
        import sqlite3

        from core.value_index import _index_path, normalize_value

        path = _index_path(star.ACCOUNT)
        with sqlite3.connect(path) as conn:
            (table, business_name) = conn.execute(
                "SELECT DISTINCT table_fqn, business_name FROM column_value WHERE column_name='EnglishProductName'"
            ).fetchone()
            conn.executemany(
                "INSERT INTO column_value (table_fqn, column_name, business_name, value, value_norm) "
                "VALUES (?, 'EnglishProductName', ?, ?, ?)",
                [(table, business_name, name, normalize_value(name)) for name in self._SIZES])
        try:
            yield
        finally:
            with sqlite3.connect(path) as conn:
                conn.executemany("DELETE FROM column_value WHERE value = ?", [(name,) for name in self._SIZES])

    def test_only_the_size_the_question_names(self, fittings):
        from core import value_resolver as vr
        from core.query_pipeline import _without_the_stated_period

        question = "Sales of copper tube in 2025"
        resolved = vr.resolve_literals(
            star.ACCOUNT, question, known_terms=vr.build_known_terms(star.ACCOUNT, None),
            vocabulary=vr.build_vocabulary_words(star.ACCOUNT, None),
            **({"measure_forms": vr.build_measure_forms(star.ACCOUNT)} if hasattr(vr, "build_measure_forms") else {}))
        (item,) = [item for item in resolved["narrowed"] if "copper tube" in item["phrase"].lower()]
        by_value = {candidate["value"]: candidate["dropped"] for candidate in item["candidates"]}
        assert by_value["Copper Tube 20"] == ["2025"] and "tube" in by_value["Copper Tee 20"]
        kept = _without_the_stated_period(resolved, question)
        assert [member["value"] for member in kept["verified"]] == ["Copper Tube 20"]
        assert not [listed for listed in kept["in_lists"] if "Copper Tee 20" in listed["values"]]


class TestTheRule:

    @pytest.mark.parametrize("context,filters", [
        (_context([], []), []),
        (_context(["Climbing"], [_CLIMBING]), [_CLIMBING]),
        (_context(["Climbing", "Summit Tent"], [_CLIMBING, _TENT]), [_CLIMBING, _TENT]),
        # Named, but not found as one column's value.
        (_context(["France"], []), None),
        (_context(["Climbing", "France"], [_CLIMBING]), None),
        # Two members of one column.
        (_context(["Summit Tent", "Ridge Tent"], [_TENT, _RIDGE]), None),
    ])
    def test_the_filters(self, context, filters):
        from core.pipeline_helpers import verified_member_filters

        assert verified_member_filters(context) == filters

    def test_a_member_a_label_and_its_twin_keep(self):
        from core.query_pipeline import _verified_members

        resolved = {"several": [
            {"phrase": "Camping", "value": "Camping", "columns": [
                {"table_fqn": "dbo.DimProductCategory", "column": "EnglishProductCategoryName"},
                {"table_fqn": "dbo.DimProductCategory", "column": "FrenchProductCategoryName"}]},
            {"phrase": "France", "value": "France", "columns": [
                {"table_fqn": "dbo.DimSalesTerritory", "column": "SalesTerritoryRegion"},
                {"table_fqn": "dbo.DimSalesTerritory", "column": "SalesTerritoryCountry"}]},
        ]}
        assert _verified_members(resolved) == [{"phrase": "Camping", "table": "dbo.DimProductCategory",
                                                "column": "EnglishProductCategoryName", "value": "Camping"}]

    @pytest.mark.parametrize("question,population", [
        ("How many products are in Climbing?", "product"),
        ("How many products are in the Climbing category?", "product"),
        ("how many products do we have in Climbing", "product"),
        # The member narrows something else.
        ("How many products are in Climbing stores?", ""),
        ("Sales in Climbing", ""),
    ])
    def test_a_population_its_members_narrow(self, question, population):
        from core.analytical_intent import population_in_members

        assert population_in_members(question, [_CLIMBING]) == population

    def test_a_member_named_beside_the_period(self):
        from core.query_pipeline import _without_the_stated_period

        resolved = {"verified": [{"phrase": "Summit Tent"}],
                    "narrowed": [{"phrase": "Summit Tent 2025", "dropped": ["2025"]},
                                 {"phrase": "Summit Tents Large", "dropped": ["large"]}]}
        kept = _without_the_stated_period(resolved, "sales of summit tent in 2025")
        assert [item["phrase"] for item in kept["narrowed"]] == ["Summit Tents Large"]

    def test_the_family_of_a_member_is_not_promoted_beside_it(self):
        """"sales of Summit Tent in 2025": the run of words "Summit Tent 2025"
        is the product the question names and the year. Its candidates are
        the product and its larger sibling; promoted, they were an IN list of
        two, beside the verified product -- and the governed answer, which
        filters on one member of a column, stood aside for the SQL writer."""
        from core.query_pipeline import _without_the_stated_period

        tent = {"table_fqn": "MART.ITM_DMS", "column": "ITM_NM", "business_name": "Item Name"}
        item = {"phrase": "Summit Tent 2025", "dropped": ["2025"], **tent, "value": "SUMMIT TENT",
                "candidates": [{"value": "SUMMIT TENT", "dropped": ["2025"]},
                               {"value": "SUMMIT TENT XL", "dropped": ["2025"]}]}
        resolved = {"verified": [{"phrase": "Summit Tent", **tent, "value": "SUMMIT TENT"}],
                    "in_lists": [], "narrowed": [item]}
        kept = _without_the_stated_period(resolved, "sales of summit tent in 2025")
        assert kept["in_lists"] == [] and kept["narrowed"] == []
        assert [member["value"] for member in kept["verified"]] == ["SUMMIT TENT"]

    def test_a_product_the_index_lacks_is_not_given_as_its_sibling(self):
        """"sales of Summit Tent XL in 2025", the XL not in the index: the run
        of words is "Summit Tent 2025" (a word of under three letters is no
        part of one), and promoted it named "Summit Tent" a verified member
        beside the phrase that said it is none."""
        from core.query_pipeline import _without_the_stated_period

        tent = {"table_fqn": "MART.ITM_DMS", "column": "ITM_NM", "business_name": "Item Name"}
        lacking = {"phrase": "Summit Tent XL", "dropped": ["xl"], **tent, "value": "SUMMIT TENT"}
        beside_the_year = {"phrase": "Summit Tent 2025", "dropped": ["2025"], **tent, "value": "SUMMIT TENT",
                           "candidates": [{"value": "SUMMIT TENT", "dropped": ["2025"]}]}
        kept = _without_the_stated_period(
            {"verified": [], "in_lists": [], "narrowed": [lacking, beside_the_year]},
            "sales of summit tent xl in 2025")
        assert kept["verified"] == [] and kept["in_lists"] == []
        assert [item["phrase"] for item in kept["narrowed"]] == ["Summit Tent XL"]

    def test_a_word_another_phrase_shares_is_not_all_of_its_words(self):
        """"copper tube in 2025" beside "copper fittings": the two phrases
        share a word, not the words the first is made of."""
        from core.query_pipeline import _without_the_stated_period

        item = {"phrase": "copper tube 2025", "dropped": ["2025"], "table_fqn": "MART.ITM_DMS",
                "column": "ITM_NM", "business_name": "Item Name", "value": "COPPER TUBE 20",
                "candidates": [{"value": "COPPER TUBE 20", "dropped": ["2025"]}]}
        fittings = {"phrase": "copper fittings", "table_fqn": "MART.ITM_DMS", "column": "ITM_NM",
                    "business_name": "Item Name", "value": "COPPER FITTINGS"}
        kept = _without_the_stated_period({"verified": [fittings], "in_lists": [], "narrowed": [item]},
                                          "units sold of copper tube and copper fittings in 2025")
        assert [member["value"] for member in kept["verified"]] == ["COPPER FITTINGS", "COPPER TUBE 20"]

    @pytest.mark.parametrize("bucket,other", [
        ("verified", {"phrase": "Summit Tent", "value": "SUMMIT TENT"}),
        ("several", {"phrase": "Summit Tent", "value": "SUMMIT TENT",
                     "columns": [{"table_fqn": "MART.ITM_DMS", "column": "ITM_NM"},
                                 {"table_fqn": "MART.ITM_DMS", "column": "ITM_FR_NM"}]}),
        ("in_lists", {"phrase": "Summit Tent", "table_fqn": "MART.ITM_DMS", "column": "ITM_NM",
                      "values": ["SUMMIT TENT", "SUMMIT TENT XL"]}),
        ("narrowed", {"phrase": "Summit Tent XL", "dropped": ["xl"], "table_fqn": "MART.ITM_DMS",
                      "column": "ITM_NM", "value": "SUMMIT TENT"}),
    ])
    def test_the_words_another_phrase_read_are_its_whichever_bucket_holds_it(self, bucket, other):
        from core.query_pipeline import _without_the_stated_period

        item = {"phrase": "Summit Tent 2025", "dropped": ["2025"], "table_fqn": "MART.ITM_DMS",
                "column": "ITM_NM", "business_name": "Item Name", "value": "SUMMIT TENT",
                "candidates": [{"value": "SUMMIT TENT", "dropped": ["2025"]}]}
        resolved = {"verified": [], "in_lists": [], "several": [], "narrowed": [item]}
        resolved[bucket] = resolved[bucket] + [other]
        kept = _without_the_stated_period(resolved, "sales of summit tent in 2025")
        assert [member["phrase"] for member in kept["verified"]] == (
            ["Summit Tent"] if bucket == "verified" else [])
        assert "Summit Tent 2025" not in [member.get("phrase") for member in kept["narrowed"]]

    def test_a_member_found_only_beside_the_period(self):
        """"the selling floor in 2025": the one run of words tried was
        "selling floor 2025", and the warehouse was dropped with the year."""
        from core.query_pipeline import _without_the_stated_period

        member = {"table_fqn": "MART.WHS_DMS", "column": "WHS_DSC", "value": "SELLING FLOOR"}
        resolved = {"verified": [], "narrowed": [{"phrase": "selling floor 2025", "dropped": ["2025"], **member}]}
        kept = _without_the_stated_period(resolved, "units sold on the selling floor in 2025")
        assert kept["narrowed"] == []
        assert kept["verified"] == [{"phrase": "selling floor 2025", **member}]
        # Found on its own as well, it is one member.
        resolved["verified"] = [{"phrase": "selling floor", **member}]
        kept = _without_the_stated_period(resolved, "units sold on the selling floor in 2025")
        assert [item["phrase"] for item in kept["verified"]] == ["selling floor"]

    def test_several_members_beside_the_period_are_a_list(self):
        """"copper pipe in March 2025": the words beside the month name a 1/2
        and a 3/4 pipe. The first of them alone would answer for one size; the
        question names both."""
        from core.query_pipeline import _without_the_stated_period

        pipes = ["COPPER PIPE 1/2", "COPPER PIPE 3/4"]
        item = {"phrase": "copper pipe March", "dropped": ["march"], "table_fqn": "MART.ITM_DMS",
                "column": "ITM_NM", "business_name": "Item Name", "value": pipes[1],
                "candidates": [{"value": pipe, "dropped": ["march"]} for pipe in pipes]}
        kept = _without_the_stated_period({"verified": [], "in_lists": [], "narrowed": [item, dict(item)]},
                                          "units sold of copper pipe in March 2025")
        assert kept["verified"] == [] and kept["narrowed"] == []
        assert kept["in_lists"] == [{"phrase": "copper pipe March", "table_fqn": "MART.ITM_DMS", "column": "ITM_NM",
                                     "business_name": "Item Name", "values": pipes}]

    def test_a_candidate_that_leaves_another_word_is_not_one_of_them(self):
        """"copper tube 2025": COPPER TEE 20 leaves "tube" over as well as the
        year, and the question names no tee."""
        from core.query_pipeline import _without_the_stated_period

        item = {"phrase": "copper tube 2025", "dropped": ["2025"], "table_fqn": "MART.ITM_DMS",
                "column": "ITM_NM", "business_name": "Item Name", "value": "COPPER TUBE 20",
                "candidates": [{"value": "COPPER TUBE 20", "dropped": ["2025"]},
                               {"value": "COPPER TEE 20", "dropped": ["tube", "2025"]}]}
        kept = _without_the_stated_period({"verified": [], "in_lists": [], "narrowed": [item]},
                                          "units sold of copper tube in 2025")
        assert kept["in_lists"] == []
        assert [(member["value"], "candidates" in member) for member in kept["verified"]] == [
            ("COPPER TUBE 20", False)]

    def test_a_promoted_candidate_is_the_phrases_answer(self):
        """"brass cap 2025": BRASS CLAMP 2025 scores above BRASS CAP 20 but
        leaves "cap" over, and BRASS CAP 20 leaves only the year. Promoted, it
        answers the phrase -- which stayed beside it as one with no verified
        match, and the SQL writer was told both."""
        from core.query_pipeline import _without_the_stated_period

        item = {"phrase": "brass cap 2025", "dropped": ["cap"], "table_fqn": "MART.ITM_DMS", "column": "ITM_NM",
                "business_name": "Item Name", "value": "BRASS CLAMP 2025",
                "candidates": [{"value": "BRASS CLAMP 2025", "dropped": ["cap"]},
                               {"value": "BRASS CAP 20", "dropped": ["2025"]}]}
        kept = _without_the_stated_period({"verified": [], "in_lists": [], "narrowed": [item]},
                                          "units sold of brass cap in 2025")
        assert ([member["value"] for member in kept["verified"]], kept["narrowed"]) == (["BRASS CAP 20"], [])

    @pytest.mark.parametrize("expression,safe", [
        ("SUM(fact_rows.SalesAmount)", "COALESCE(SUM(fact_rows.SalesAmount), 0)"),
        ("SUM(a) - SUM(b * (1 - c))", "COALESCE(SUM(a), 0) - COALESCE(SUM(b * (1 - c)), 0)"),
        ("COUNT(DISTINCT k)", "COUNT(DISTINCT k)"),
    ])
    def test_a_total_of_none_is_zero(self, expression, safe):
        from core.pipeline_helpers import _null_safe_sums

        assert _null_safe_sums(expression) == safe

    @pytest.mark.parametrize("db_type,literal", [("azure_sql", "N'Men''s Jacket'"), ("snowflake", "'Men''s Jacket'")])
    def test_a_members_literal(self, db_type, literal):
        from core.pipeline_helpers import _member_literal

        assert _member_literal("Men's Jacket", db_type) == literal
