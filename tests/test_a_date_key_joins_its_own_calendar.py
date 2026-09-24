"""
A date key joins the calendar its own name and grain point at -- never the
first date-like table discovery happened to list.

A warehouse with a day table (DT_DMS, keyed yyyymmdd), a period table (PRD_DMS,
keyed yyyymm) and a monthly balance fact that carries PRD_DMS_KEY as a foreign
key beside an audit timestamp used to produce three "date dimensions": the day
table, the period table, and the balance FACT. Every date role in the warehouse
was then bound to whichever of the three came first. On one ordering that was
the balance fact -- a fact-to-fact join filed in the graph under a dimension's
name, where the no-fact-to-fact rule could not see it -- and on another it was
the period table, a yyyymmdd key compared with a yyyymm one, which matches
nothing.

Every builder that binds a date key is driven here from a real schema file, in
several table orders, and all four must agree: the entity graph, the KB join
map the SQL model reads, the semantic model, and the Date Roles form's
suggestions. Synthetic tables in a mart's naming convention; no customer data.
"""

from __future__ import annotations

import json
import re

import pytest

from core.date_roles import (
    DateDimensionCandidate,
    choose_date_dimension,
    find_date_value_column,
    is_date_dimension_table,
)
from core.schema import _build_join_map, build_entity_graph_from_schema
from core.semantic_model import build_semantic_model, suggest_date_key_bindings


def _table(schema: str, own_key: str, *columns: tuple[str, str]) -> dict:
    return {
        "columns": [
            {"name": name, "type": dtype, "nullable": False, "comment": ""}
            for name, dtype in columns
        ],
        "pk_columns": [own_key] if own_key else [],
        "row_count": 100,
        "comment": "",
        "schema": schema,
        "database": "WH",
    }


DAY = "WH.MART.DT_DMS"
PERIOD = "WH.MART.PRD_DMS"
WAREHOUSE = "WH.MART.WHS_DMS"
DAILY = "WH.MART.ITM_BAL_DLY_FCT"
MONTHLY = "WH.MART.ITM_BAL_PRD_FCT"

TABLES = {
    DAY: _table(
        "MART", "DT_DMS_KEY",
        ("DT_DMS_KEY", "int"), ("DMS_DT", "date"), ("YR", "int"), ("MTH", "int"),
        ("MTH_NM", "nvarchar"), ("MTH_FR_NM", "nvarchar"), ("AZ_LST_UPD_TS", "datetime2"),
    ),
    PERIOD: _table(
        "MART", "PRD_DMS_KEY",
        ("PRD_DMS_KEY", "int"), ("PRD_CD", "nvarchar"), ("PRD_YR", "int"),
        ("PRD_MTH", "int"), ("AZ_LST_UPD_TS", "datetime2"),
    ),
    WAREHOUSE: _table(
        "MART", "WHS_DMS_KEY",
        ("WHS_DMS_KEY", "int"), ("WHS_CD", "nvarchar"), ("WHS_DSC", "nvarchar"),
    ),
    DAILY: _table(
        "MART", "ITM_BAL_DLY_FCT_KEY",
        ("ITM_BAL_DLY_FCT_KEY", "int"), ("WHS_DMS_KEY", "int"),
        ("ITM_BAL_EFC_DT_DMS_KEY", "int"), ("ITM_WHS_CRN_DT_DMS_KEY", "int"),
        ("ON_HND_QTY", "decimal"), ("AZ_LST_UPD_TS", "datetime2"),
    ),
    MONTHLY: _table(
        "MART", "ITM_BAL_PRD_FCT_KEY",
        ("ITM_BAL_PRD_FCT_KEY", "int"), ("WHS_DMS_KEY", "int"), ("PRD_DMS_KEY", "int"),
        ("CUR_ON_HND_QTY", "decimal"), ("BAL_TS", "datetime2"),
        ("AZ_LST_UPD_TS", "datetime2"),
    ),
}

DAY_KEYS = {"ITM_BAL_EFC_DT_DMS_KEY", "ITM_WHS_CRN_DT_DMS_KEY"}

# The order a catalogue lists tables in is (database, schema, table name), so
# which date table comes "first" is an accident of naming. These include the
# two orders that broke: the balance fact ahead of the day table, and the
# period table ahead of it.
ORDERS = {
    "facts first": [DAILY, MONTHLY, WAREHOUSE, DAY, PERIOD],
    "alphabetical": sorted(TABLES),
    "reverse": sorted(TABLES, reverse=True),
    "period table first": [PERIOD, MONTHLY, DAY, DAILY, WAREHOUSE],
}


def _write(tmp_path, order: list[str], tables: dict | None = None) -> str:
    tables = tables or TABLES
    (tmp_path / "_schema.json").write_text(
        json.dumps({name: tables[name] for name in order}), encoding="utf-8",
    )
    return str(tmp_path)


def _columns(fqn: str) -> list[dict]:
    return TABLES[fqn]["columns"]


class TestOnlyACalendarIsACalendar:

    def test_a_balance_fact_carrying_a_period_key_is_not_a_date_table(self):
        assert is_date_dimension_table(MONTHLY, _columns(MONTHLY)) is False

    def test_a_table_with_its_own_key_is_not_a_date_table_whatever_its_name(self):
        # No _FCT suffix to go on: its own surrogate key is what gives it away.
        columns = [
            {"name": "STOCK_SNAPSHOT_KEY", "type": "int"},
            {"name": "DATE_KEY", "type": "int"},
            {"name": "SNAPSHOT_TS", "type": "datetime2"},
        ]
        assert is_date_dimension_table("WH.MART.STOCK_SNAPSHOT", columns) is False

    def test_a_fact_with_no_key_of_its_own_is_still_not_a_date_table(self):
        # Plenty of snapshot facts have no surrogate key, only their grain's
        # foreign keys; the name is then the only thing that says "fact".
        columns = [
            {"name": "WHS_DMS_KEY", "type": "int"},
            {"name": "PRD_DMS_KEY", "type": "int"},
            {"name": "BAL_TS", "type": "datetime2"},
            {"name": "ON_HND_QTY", "type": "decimal"},
        ]
        assert is_date_dimension_table("WH.MART.INV_SNAP_FCT", columns) is False

    def test_the_day_and_period_tables_still_are(self):
        assert is_date_dimension_table(DAY, _columns(DAY)) is True
        assert is_date_dimension_table(PERIOD, _columns(PERIOD)) is True

    def test_a_calendar_with_a_surrogate_key_of_its_own_is_still_a_calendar(self):
        # Its own key beside the date key would otherwise read as "an entity
        # that merely carries a date"; its year and month say it is the calendar.
        columns = [
            {"name": "DIM_DAY_KEY", "type": "int"},
            {"name": "DATE_KEY", "type": "int"},
            {"name": "CAL_DT", "type": "date"},
            {"name": "YR", "type": "int"},
            {"name": "MTH", "type": "int"},
        ]
        assert is_date_dimension_table("WH.MART.DIM_DAY", columns) is True

    def test_an_audit_stamp_is_never_a_calendars_date(self):
        assert find_date_value_column(_columns(DAY)) == "DMS_DT"
        # The period table has no calendar date at all; its load stamp is not one.
        assert find_date_value_column(_columns(PERIOD)) == ""
        # A key and a load stamp alone do not make a table a calendar.
        columns = [
            {"name": "DATE_KEY", "type": "int"},
            {"name": "AZ_LST_UPD_TS", "type": "datetime2"},
        ]
        assert is_date_dimension_table("WH.MART.CAL_TABLE", columns) is False


def _candidate(table: str, key: str, schema: str = "MART", key_type: str = "int"):
    return DateDimensionCandidate(table=table, key=key, key_type=key_type, schema=schema)


class TestTheChoiceIsMadeOnEvidence:

    def test_grain_decides_between_a_day_and_a_period_table_in_either_order(self):
        day = _candidate("DT_DMS", "DT_DMS_KEY")
        period = _candidate("PRD_DMS", "PRD_DMS_KEY")
        for candidates in ([day, period], [period, day]):
            chosen, confidence = choose_date_dimension("ITM_BAL_EFC_DT_DMS_KEY", "int", candidates)
            assert chosen is day and confidence >= 85
            chosen, _ = choose_date_dimension("ORD_PRD_DMS_KEY", "int", candidates)
            assert chosen is period

    def test_grain_alone_decides_when_the_names_do_not(self):
        # Neither table's key is spelled inside ORD_DT_DMS_KEY; only the
        # yyyymmdd grain its name declares tells the day table from the period.
        period = _candidate("PRD_DMS", "PRD_DMS_KEY")
        day = _candidate("CAL_DMS", "DATE_DMS_KEY")
        for candidates in ([period, day], [day, period]):
            assert choose_date_dimension("ORD_DT_DMS_KEY", "int", candidates) == (day, 92)

    def test_the_most_specific_key_name_wins(self):
        calendar = _candidate("DT_DMS", "DT_DMS_KEY")
        fiscal = _candidate("FSC_DT_DMS", "FSC_DT_DMS_KEY")
        for candidates in ([calendar, fiscal], [fiscal, calendar]):
            assert choose_date_dimension("ORD_FSC_DT_DMS_KEY", "int", candidates)[0] is fiscal
            assert choose_date_dimension("ORD_DT_DMS_KEY", "int", candidates)[0] is calendar

    def test_a_native_date_needs_no_calendar(self):
        day = _candidate("DT_DMS", "DT_DMS_KEY")
        assert choose_date_dimension("ORDER_DT", "datetime2", [day]) == (None, 0)

    def test_an_undecidable_key_is_left_undecided_not_guessed(self):
        first = _candidate("DIM_CALENDAR", "CALENDAR_KEY", schema="CAL_A")
        second = _candidate("D_DATE", "DAY_KEY", schema="CAL_B")
        for candidates in ([first, second], [second, first]):
            assert choose_date_dimension("ORDER_DATE_KEY", "int", candidates, schema="SALES") == (None, 0)
            # The key's own schema settles it when only one calendar lives there.
            chosen, confidence = choose_date_dimension(
                "ORDER_DATE_KEY", "int", candidates, schema="CAL_B",
            )
            assert chosen is second and confidence == 85


class TestEveryBuilderAgreesInEveryOrder:

    @pytest.mark.parametrize("order", ORDERS.values(), ids=list(ORDERS))
    def test_the_entity_graph(self, tmp_path, order):
        graph = build_entity_graph_from_schema(_write(tmp_path, order))
        entities = {e["entity_name"]: e for e in graph["entities"]}
        edges = [r for r in graph["relationships"] if r.get("generated_by") == "date_role"]

        assert {e["from_column"] for e in edges} == DAY_KEYS
        for edge in edges:
            alias = entities[edge["to_entity"]]
            assert (alias["table_name"], edge["to_column"]) == ("DT_DMS", "DT_DMS_KEY")
        # No alias of any fact, however the date roles came out.
        fact_tables = {"ITM_BAL_DLY_FCT", "ITM_BAL_PRD_FCT"}
        aliases = {entities[e["to_entity"]]["table_name"] for e in edges}
        assert not aliases & fact_tables

    @pytest.mark.parametrize("order", ORDERS.values(), ids=list(ORDERS))
    def test_the_kb_join_map(self, order):
        join_map = _build_join_map({name: TABLES[name] for name in order})
        section = join_map.split("## Role-Playing Date Dimension Joins", 1)[1].split("\n## ", 1)[0]
        joins = re.findall(r"\*\*Join:\*\* `([^`]+)` = `([^`]+)`", section)

        assert sorted(left for left, _ in joins) == sorted(
            f"{DAILY}.{key}" for key in DAY_KEYS
        )
        assert {right for _, right in joins} == {f"{DAY}.DT_DMS_KEY"}

    @pytest.mark.parametrize("order", ORDERS.values(), ids=list(ORDERS))
    def test_the_semantic_model(self, tmp_path, order):
        model = build_semantic_model(_write(tmp_path, order))
        roles = {r["fact_column"]: r for r in model["date_roles"]}

        for key in DAY_KEYS:
            assert roles[key]["dimension_table"].endswith("DT_DMS")
            assert roles[key]["dimension_key"] == "DT_DMS_KEY"
            assert roles[key]["status"] == "generated"
        # A monthly key stays a direct period encoding, as it always was.
        assert roles["PRD_DMS_KEY"]["date_key_type"] == "yyyymm_integer"
        assert roles["PRD_DMS_KEY"]["dimension_table"] == ""

    def test_the_semantic_model_does_not_depend_on_order_at_all(self, tmp_path):
        def roles_for(order):
            directory = tmp_path / str(len(list(tmp_path.iterdir())))
            directory.mkdir()
            model = build_semantic_model(_write(directory, order))
            return sorted(
                (r["fact_table"], r["fact_column"], r.get("dimension_table"),
                 r.get("date_key_type"), r.get("confidence"))
                for r in model["date_roles"]
            )

        results = [roles_for(order) for order in ORDERS.values()]
        assert all(result == results[0] for result in results)

    @pytest.mark.parametrize("order", ORDERS.values(), ids=list(ORDERS))
    def test_the_date_role_form_suggests_the_same_calendar(self, tmp_path, order):
        model = build_semantic_model(_write(tmp_path, order))
        bindings = suggest_date_key_bindings(model)["bindings"]
        daily = next(t["qualified_name"] for t in model["tables"] if t["table"] == "ITM_BAL_DLY_FCT")

        for key in DAY_KEYS:
            binding = bindings[f"{daily}|{key}"]
            assert binding["dimension_table"].endswith("DT_DMS")
            assert binding["dimension_key"] == "DT_DMS_KEY"


class TestTwoCalendarsOfTheSameGrain:
    """Two day tables; each key's own name says which one it joins."""

    TABLES = {
        "WH.MART.DT_DMS": _table(
            "MART", "DT_DMS_KEY", ("DT_DMS_KEY", "int"), ("DMS_DT", "date"),
        ),
        "WH.MART.DATE_DMS": _table(
            "MART", "DATE_DMS_KEY", ("DATE_DMS_KEY", "int"), ("CAL_DT", "date"),
        ),
        "WH.MART.SLS_FCT": _table(
            "MART", "SLS_FCT_KEY",
            ("SLS_FCT_KEY", "int"), ("ORD_DT_DMS_KEY", "int"),
            ("IVC_DATE_DMS_KEY", "int"), ("NET_AMT", "decimal"),
        ),
    }
    EXPECTED = {"ORD_DT_DMS_KEY": "DT_DMS", "IVC_DATE_DMS_KEY": "DATE_DMS"}

    @pytest.mark.parametrize("reverse", [False, True])
    def test_every_builder_joins_each_key_to_the_calendar_it_names(self, tmp_path, reverse):
        order = sorted(self.TABLES, reverse=reverse)
        directory = _write(tmp_path, order, self.TABLES)

        graph = build_entity_graph_from_schema(directory)
        entities = {e["entity_name"]: e for e in graph["entities"]}
        bound = {
            r["from_column"]: entities[r["to_entity"]]["table_name"]
            for r in graph["relationships"] if r.get("generated_by") == "date_role"
        }
        assert bound == self.EXPECTED

        join_map = _build_join_map({name: self.TABLES[name] for name in order})
        section = join_map.split("## Role-Playing Date Dimension Joins", 1)[1].split("\n## ", 1)[0]
        joins = {
            left.rsplit(".", 1)[1]: right.rsplit(".", 2)[1]
            for left, right in re.findall(r"\*\*Join:\*\* `([^`]+)` = `([^`]+)`", section)
        }
        assert joins == self.EXPECTED

        model = build_semantic_model(directory)
        roles = {
            r["fact_column"]: r["dimension_table"].rsplit(".", 1)[-1]
            for r in model["date_roles"] if r["fact_column"] in self.EXPECTED
        }
        assert roles == self.EXPECTED

        fact = next(t["qualified_name"] for t in model["tables"] if t["table"] == "SLS_FCT")
        bindings = suggest_date_key_bindings(model)["bindings"]
        suggested = {
            key: bindings[f"{fact}|{key}"]["dimension_table"].rsplit(".", 1)[-1]
            for key in self.EXPECTED
        }
        assert suggested == self.EXPECTED


class TestACalendarCanPointAtAnother:
    """A period table's start date reaches the day table; no calendar reaches itself."""

    TABLES = {
        # PRV_YR_DT_DMS_KEY is the same day last year: a key back into this
        # very calendar, which must not be sent to the other day table.
        "WH.MART.DT_DMS": _table(
            "MART", "DT_DMS_KEY",
            ("DT_DMS_KEY", "int"), ("DMS_DT", "date"), ("PRV_YR_DT_DMS_KEY", "int"),
        ),
        "WH.MART.DATE_DMS": _table(
            "MART", "DATE_DMS_KEY", ("DATE_DMS_KEY", "int"), ("CAL_DT", "date"),
        ),
        "WH.MART.PRD_DMS": _table(
            "MART", "PRD_DMS_KEY",
            ("PRD_DMS_KEY", "int"), ("PRD_STR_DT_DMS_KEY", "int"), ("PRD_YR", "int"),
        ),
    }

    @pytest.mark.parametrize("reverse", [False, True])
    def test_the_join_map_links_calendars_only_through_their_date_roles(self, reverse):
        order = sorted(self.TABLES, reverse=reverse)
        join_map = _build_join_map({name: self.TABLES[name] for name in order})
        section = join_map.split("## Role-Playing Date Dimension Joins", 1)[1].split("\n## ", 1)[0]
        joins = re.findall(r"\*\*Join:\*\* `([^`]+)` = `([^`]+)`", section)

        assert joins == [("WH.MART.PRD_DMS.PRD_STR_DT_DMS_KEY", "WH.MART.DT_DMS.DT_DMS_KEY")]


class TestAnUndecidableKeyIsReportedNotGuessed:
    """Two calendars that nothing distinguishes: an admin decides, not the order."""

    TABLES = {
        "WH.CAL_A.DIM_CALENDAR": _table(
            "CAL_A", "CALENDAR_KEY", ("CALENDAR_KEY", "int"), ("CALENDAR_DATE", "date"),
        ),
        "WH.CAL_B.D_DATE": _table(
            "CAL_B", "DAY_KEY", ("DAY_KEY", "int"), ("FULL_DATE", "date"),
        ),
        "WH.SALES.ORD_FCT": _table(
            "SALES", "ORD_FCT_KEY",
            ("ORD_FCT_KEY", "int"), ("ORDER_DATE_KEY", "int"), ("NET_AMT", "decimal"),
        ),
    }

    @pytest.mark.parametrize("reverse", [False, True])
    def test_no_role_is_invented_and_the_column_is_queued_for_review(self, tmp_path, reverse):
        order = sorted(self.TABLES, reverse=reverse)
        directory = _write(tmp_path, order, self.TABLES)

        model = build_semantic_model(directory)
        assert not [r for r in model["date_roles"] if r["fact_column"] == "ORDER_DATE_KEY"]
        queued = model["date_role_coverage"]["review_candidates"]
        assert any(item["column"] == "ORDER_DATE_KEY" for item in queued)

        graph = build_entity_graph_from_schema(directory)
        assert not [
            r for r in graph["relationships"]
            if r.get("generated_by") == "date_role" and r["from_column"] == "ORDER_DATE_KEY"
        ]
