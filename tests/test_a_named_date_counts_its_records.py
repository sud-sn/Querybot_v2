"""
A date the question names counts the records it dates.

"How many items were created in the last 12 months?", "items created by month
in 2025", "how many item-warehouse records were created each year?" and their
French twins were never answered on the sample tenant. The creation date is kept
on the daily snapshot only, but both snapshots matched "items" equally and the
reader was asked which dataset to use; told, the planner found no measure --
nothing in the question is summed -- and refused.

Now:

  * a date the question names chooses the table that keeps it, unless a
    measure the question asks for is kept only on another;
  * "how many <records> were <created> ..." and "<records> <created> by
    <grain>" count the records that date covers: distinct items by the table's
    own item key, and a table's rows when the question names them as records.
    A noun the table keeps no key of is not counted, and "items created in
    2025" alone asks for the items, not how many;
  * a periodic snapshot repeats each record under every snapshot, so only its
    latest snapshot is counted -- unless the date named is the snapshot's own
    -- and the validator holds a query to that;
  * the counted records are not a breakdown: "items created by month" is not
    grouped by item, nor joined to the item table;
  * "créé(e)s", "ont été", "a été" and "fiches" are read in French.

A synthetic tenant (tests/answer_harness.py) whose daily snapshot keeps two
snapshots of each item-warehouse record with the date each was created; the
warehouse and the model are the only stand-ins.
"""

from __future__ import annotations

import datetime as dt
import re

import pytest

from tests import answer_harness as harness


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("named-date-counts")) as built:
        yield built


def _created() -> list[tuple[int, dt.date]]:
    """(item, creation date) of each record of one snapshot."""
    return [(row[1], dt.datetime.strptime(str(row[3]), "%Y%m%d").date()) for row in harness.STOCK]


def _count(bucket, distinct_items: bool) -> dict:
    groups: dict = {}
    for item, day in _created():
        groups.setdefault(bucket(day), []).append(item)
    return {key: len(set(items)) if distinct_items else len(items) for key, items in groups.items()}


def _month(day: dt.date) -> dt.date:
    return day.replace(day=1)


def _quarter(day: dt.date) -> dt.date:
    return day.replace(month=(day.month - 1) // 3 * 3 + 1, day=1)


def _year(day: dt.date) -> int:
    return day.year


def _answer(answer: dict) -> tuple[str, dict]:
    """The count the answer ran, by name, and its value in each period (None
    when it has no period)."""
    assert answer["model_wrote_sql"] is False
    (run,) = [run for run in answer["executed"] if re.search(r"\bAS \w+_COUNT\b", run["sql"])]
    (name,) = {key for row in run["rows"] for key in row if key.endswith("_COUNT")}
    return name, {row.get("PERIOD"): row[name] for row in run["rows"]}


class TestTheProductAnswers:

    def test_items_created_by_month(self, warehouse):
        answer = harness.ask(warehouse, "Items created by month in 2025")
        assert _answer(answer) == ("ITEM_COUNT", _count(_month, distinct_items=True))

    def test_items_created_by_quarter(self, warehouse):
        # Two items were created in the first quarter: a count of items, not
        # a breakdown by item.
        answer = harness.ask(warehouse, "Items created by quarter in 2025")
        assert _answer(answer) == ("ITEM_COUNT", _count(_quarter, distinct_items=True))

    @pytest.mark.parametrize("question", [
        "How many items were created in 2025?",
        # Anchored on the newest creation date, every record of the tenant is
        # inside the window.
        "How many items were created in the last 12 months?",
    ])
    def test_how_many_items(self, warehouse, question):
        answer = harness.ask(warehouse, question)
        assert _answer(answer) == ("ITEM_COUNT", {None: len({item for item, _day in _created()})})

    def test_records_each_year(self, warehouse):
        answer = harness.ask(warehouse, "How many item-warehouse records were created each year?")
        assert _answer(answer) == ("ITEM_WAREHOUSE_RECORD_COUNT", _count(_year, distinct_items=False))

    @pytest.mark.parametrize("question", [
        "How many item-warehouse records were created in 2025?",
        "How many item-warehouse records were created in the last 12 months?",
    ])
    def test_how_many_records(self, warehouse, question):
        answer = harness.ask(warehouse, question)
        assert _answer(answer) == ("ITEM_WAREHOUSE_RECORD_COUNT", {None: len(_created())})

    @pytest.mark.parametrize("question,bucket", [
        ("Combien de fiches ont été créées par mois en 2025 ?", _month),
        # "records item-warehouse": the rows, in French word order.
        ("Combien de fiches article-entrepôt ont été créées par année ?", _year),
    ])
    def test_in_french(self, warehouse, question, bucket):
        answer = harness.ask(warehouse, question, "fr")
        assert _answer(answer)[1] == _count(bucket, distinct_items=False)

    def test_not_over_a_measure_the_date_does_not_date(self, warehouse):
        # Units sold are kept on the monthly snapshot, which keeps no creation
        # date: the reader is told, and asked for one of its dates.
        answer = harness.ask(warehouse, "Units sold for items created in 2025")
        (asked,) = [reply["question"] for kind, reply in answer["replies"] if kind == "clarify"]
        assert "Creation Date" in asked and "Units sold" in asked
        assert answer["executed"] == []


class TestTheReading:

    @pytest.mark.parametrize("question,expected", [
        ("How many items were created in 2025?", ("item", "created")),
        ("Items created by month", ("item", "created")),
        ("How many item-warehouse records were created each year?", ("item-warehouse record", "created")),
        ("How many distinct items were created in 2025?", ("item", "created")),
        # A qualified noun is kept whole: "active items" are not all the items.
        ("How many active items were created in 2025?", ("active item", "created")),
        # Asks for the items, not how many.
        ("Items created in 2025", ("", "")),
        ("Which items were created by month?", ("", "")),
        ("Stock on hand by warehouse", ("", "")),
    ])
    def test_what_is_counted(self, question, expected):
        from core.analytical_intent import detect_record_count

        assert detect_record_count(question) == expected

    @pytest.mark.parametrize("question,expected", [
        ("Combien de fiches ont été créées par mois en 2025 ?", ("record", "created")),
        ("Combien de comptes ont été créés en 2025 ?", ("compte", "created")),
    ])
    def test_french(self, question, expected):
        from core.analytical_intent import detect_record_count
        from core.question_normalizer import canonical_question

        assert detect_record_count(canonical_question(question, "fr")) == expected

    @pytest.mark.parametrize("question,expected", [
        ("Quand la fiche a été créée ?", "the record has been created"),
        ("Quand le compte a été créé ?", "has been created"),
    ])
    def test_french_in_the_singular(self, question, expected):
        from core.question_normalizer import canonical_question

        assert expected in canonical_question(question, "fr")


_NAMED = {"kind": "named_period", "fact_table": "MART.STOCK_DAILY", "fact_column": "CRN_DT_KEY",
          "resolution_source": "explicit_date_role", "business_role": "Creation Date"}
# "Items" as the planner binds them: the item's name, through the key the
# daily snapshot keeps.
_ITEM = {"term": "items", "role": "display_dimension", "table": "MART.ITEMS", "column": "ITEM_NAME",
         "source_key_table": "MART.STOCK_DAILY", "source_key_column": "ITM_KEY"}
_WAREHOUSE = {"term": "warehouse", "role": "display_dimension", "table": "MART.WAREHOUSES", "column": "WHS_NAME",
              "source_key_table": "MART.STOCK_DAILY", "source_key_column": "WHS_KEY"}
_UNITS = {"term": "units sold", "role": "measure", "table": "MART.STOCK_DAILY", "column": "UNITS_SOLD"}


def _plan(policy: dict = _NAMED, fact: str = "MART.STOCK_DAILY", fields: tuple = (_ITEM,)) -> dict:
    return {"source_scope": {"selected_fact": fact}, "fields": [dict(field) for field in fields],
            "temporal_policies": [policy]}


class TestTheDecision:

    @pytest.mark.parametrize("noun,key", [
        ("item", "ITM_KEY"),
        ("item-warehouse record", ""),
    ], ids=["an entity by its key", "the rows"])
    def test_counts(self, noun, key):
        from core.analytical_request_plan import record_count_by_named_date

        decision = record_count_by_named_date(_plan(), {"record_count": noun})
        assert (decision["key"], decision["dated_column"]) == (key, "CRN_DT_KEY")

    @pytest.mark.parametrize("plan,noun,metrics", [
        (_plan({**_NAMED, "resolution_source": "fact_default_date_role"}), "item", []),
        (_plan(fact="MART.STOCK_MONTHLY", fields=({**_ITEM, "source_key_table": "MART.STOCK_MONTHLY"},)), "item", []),
        (_plan(), "item", [{"name": "Stock on hand"}]),
        (_plan(fields=(_ITEM, _UNITS)), "item", []),
        (_plan(), "sale", []),
        (_plan(), "active item", []),
    ], ids=["a date the question did not name", "a date on another table", "a metric", "a measure",
            "a noun the table keeps no key of", "a noun no field is named for whole"])
    def test_not(self, plan, noun, metrics):
        from core.analytical_request_plan import record_count_by_named_date

        assert record_count_by_named_date(plan, {"record_count": noun}, metrics) == {}

    def test_the_counted_are_not_a_breakdown(self):
        # "Items created by month" counts items; it is not grouped by item.
        from core.analytical_request_plan import compile_analytical_request_plan, demote_counted_records

        plan = _plan(fields=({**_ITEM, "enforcement": "required"},))
        demote_counted_records(plan, {"record_count": "item"})
        request = compile_analytical_request_plan(
            "Items created by month in 2025", plan, analytical_intent_plan={"record_count": "item"},
        )
        assert (request["status"], request["dimensions"]) == ("compiled", [])
        assert request["derived_measure"]["target_column"] == "ITM_KEY"

    def test_a_grouping_is_not_counted(self):
        # "warehouse" in "item-warehouse records ... by warehouse" is both a
        # word of what is counted and what the count is grouped by.
        from core.analytical_request_plan import compile_analytical_request_plan, demote_counted_records

        plan = _plan(fields=(_ITEM, _WAREHOUSE))
        intent = {"record_count": "item-warehouse record", "dimensions": ["warehouse"]}
        demote_counted_records(plan, intent)
        assert [field.get("enforcement") for field in plan["fields"]] == ["optional", None]
        request = compile_analytical_request_plan(
            "How many item-warehouse records were created in 2025 by warehouse?", plan,
            analytical_intent_plan=intent,
        )
        assert [dimension["term"] for dimension in request["dimensions"]] == ["warehouse"]

    @staticmethod
    def _scope(question: str):
        from core.source_resolution import resolve_source_scope

        def fact(name: str, measures: tuple[str, ...] = ()) -> dict:
            return {"type": "fact", "qualified_name": name, "entity": "Stock",
                    "fields": [{"column": column, "role": "measure"} for column in measures]}

        model = {
            "tables": [fact("MART.STOCK_DAILY", ("ON_HAND_QTY",)),
                       fact("MART.STOCK_MONTHLY", ("ON_HAND_QTY", "UNITS_SOLD"))],
            # As discovery names a creation date: "created" is read through
            # its synonym "created date".
            "date_roles": [{"name": "Creation Date", "business_role": "creation_date",
                            "fact_table": "MART.STOCK_DAILY", "fact_column": "CRN_DT_KEY",
                            "status": "approved", "synonyms": ["creation date", "created date"]}],
        }
        scope = resolve_source_scope(question, model)
        return scope["status"], scope["selected_fact"]

    @pytest.mark.parametrize("question", [
        "How many stock records were created in 2025?",
        # The date is all the question says of its source.
        "How many records were created in 2025?",
        "On hand qty of the stock created in 2025",
    ])
    def test_the_named_date_chooses_its_table(self, question):
        assert self._scope(question) == ("selected", "MART.STOCK_DAILY")

    def test_not_over_a_measure_only_another_table_keeps(self):
        assert self._scope("Units sold of the stock created in 2025") == ("selected", "MART.STOCK_MONTHLY")


class TestTheSnapshot:
    """The snapshot a count of a periodic snapshot's records is read at."""

    @staticmethod
    def _key(dated_column: str, fact_type: str = "periodic_snapshot") -> str:
        from core.analytical_request_plan import counted_snapshot_key

        model = {
            "tables": [{"qualified_name": "MART.STOCK_DAILY", "fact_type": fact_type}],
            "date_roles": [
                {"fact_table": "MART.STOCK_DAILY", "fact_column": "CRN_DT_KEY", "is_default": False},
                {"fact_table": "MART.STOCK_DAILY", "fact_column": "SNAP_DT_KEY", "is_default": True},
            ],
        }
        return counted_snapshot_key(model, "MART.STOCK_DAILY", dated_column)

    def test_its_latest_snapshot(self):
        assert self._key("CRN_DT_KEY") == "SNAP_DT_KEY"

    def test_each_snapshot_when_the_question_names_the_snapshots_date(self):
        assert self._key("SNAP_DT_KEY") == ""

    def test_not_a_snapshot(self):
        assert self._key("CRN_DT_KEY", fact_type="transaction") == ""


class TestTheValidator:
    """A count of a snapshot's records reads its latest snapshot, and counts rows."""

    _COLS = {"MART.STOCK_DAILY": {"SNAP_DT_KEY": "int", "CRN_DT_KEY": "int", "ITM_KEY": "int"}}
    _LATEST = "WHERE f.SNAP_DT_KEY = (SELECT MAX(s.SNAP_DT_KEY) FROM MART.STOCK_DAILY s)"

    def _validate(self, sql: str, semantics: str = "count_records"):
        from core.validator import validate_sql_detailed

        context = {"analytical_request_plan": {"derived_measure": {
            "semantics": semantics, "business_entity": "record", "target_table": "MART.STOCK_DAILY",
            "target_column": "" if semantics == "count_records" else "ITM_KEY",
            "snapshot_column": "SNAP_DT_KEY",
        }}}
        return validate_sql_detailed(sql, set(self._COLS), "azure_sql", None, self._COLS, context)

    def test_the_latest_snapshots_rows(self):
        assert self._validate(f"SELECT COUNT(*) AS N FROM MART.STOCK_DAILY f {self._LATEST}").ok

    def test_every_snapshot_is_refused(self):
        result = self._validate("SELECT COUNT(*) AS N FROM MART.STOCK_DAILY f")
        assert result.code == "derived_measure_mismatch"

    def test_every_snapshot_of_a_distinct_count_is_refused(self):
        result = self._validate(
            "SELECT COUNT(DISTINCT f.ITM_KEY) AS N FROM MART.STOCK_DAILY f", "count_distinct_business_identifier")
        assert result.code == "derived_measure_mismatch"

    def test_a_distinct_count_for_the_rows_is_refused(self):
        result = self._validate(f"SELECT COUNT(DISTINCT f.ITM_KEY) AS N FROM MART.STOCK_DAILY f {self._LATEST}")
        assert result.code == "derived_measure_mismatch"
