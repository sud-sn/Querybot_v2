# -*- coding: utf-8 -*-
"""A join is checked against the data on a bounded slice of its fact.

Every join discovery suggests is profiled against the warehouse (see
tests/test_every_discovered_join_is_checked_against_the_data.py). The probe
read the whole fact table eight times over -- EXISTS, NOT EXISTS and a full
join among them -- so on a warehouse-sized fact it ran past its 20-second
timeout, three timeouts ended the run, and every join stayed "untested": the
resolver had no evidence to rank joins on, and the admin none to approve on.

The probe now reads the first rows of the fact once, joined to the
dimension's keys grouped once. A slice that matches nothing is not taken as
the verdict on the join -- "matches nothing" keeps the resolver off a join for
good -- so the whole table is asked first, by a query that stops at its first
match.

The warehouse is SQLite behind the probe boundary. Synthetic tables; no
customer data.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

import core.relationship_validator as rv

REL = {"id": 7, "from_column": "ITM_DMS_KEY", "to_column": "ITM_DMS_KEY", "join_type": "INNER"}
FACT = {"schema_name": "MART", "table_name": "ITM_BAL_DLY_FCT"}
DIM = {"schema_name": "MART", "table_name": "ITM_DMS"}


def _warehouse(fact_keys: list[int | None], dim_keys: list[int]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("ATTACH DATABASE ':memory:' AS MART")
    conn.execute("CREATE TABLE MART.ITM_BAL_DLY_FCT (ITM_DMS_KEY INT, ON_HND_QTY REAL)")
    conn.execute("CREATE TABLE MART.ITM_DMS (ITM_DMS_KEY INT, ITM_CD TEXT)")
    conn.executemany("INSERT INTO MART.ITM_BAL_DLY_FCT VALUES (?, 1.0)", [(k,) for k in fact_keys])
    conn.executemany("INSERT INTO MART.ITM_DMS VALUES (?, 'x')", [(k,) for k in dim_keys])
    return conn


def _probe(conn, *, sample_rows: int):
    """The real _execute_probe, with the warehouse and the slice size swapped in."""
    asked: list[str] = []

    def run_probe(db_type, raw_cfg, sql, **kwargs):
        asked.append(sql)
        return [tuple(row) for row in conn.execute(sql).fetchmany(kwargs.get("max_rows") or 200)]

    with patch.object(rv, "run_probe", run_probe), patch.object(rv, "PROFILE_SAMPLE_ROWS", sample_rows):
        result = rv._execute_probe("acct", REL, FACT, DIM, db_type="snowflake", raw_cfg={"credentials": {}})
    return result, asked


class TestTheProbeReadsABoundedSlice:

    def test_it_reads_no_more_of_the_fact_than_the_slice(self):
        conn = _warehouse([1, 1, 2, None, 99, 1, 2, 2], [1, 2])
        row = conn.execute(rv.build_profile_sql("snowflake", REL, FACT, DIM, sample_rows=5)).fetchone()
        # The first five rows: 1, 1, 2, NULL, 99.
        assert row == (5, 4, 3, 1, 3, 2, 2, 0)

    def test_by_default_the_slice_is_a_hundred_thousand_rows(self):
        assert rv.PROFILE_SAMPLE_ROWS == 100_000
        assert "SELECT TOP (100000) [ITM_DMS_KEY] AS k0 FROM [MART].[ITM_BAL_DLY_FCT]" in \
            rv.build_profile_sql("azure_sql", REL, FACT, DIM)

    def test_a_join_is_judged_on_the_slice(self):
        result, asked = _probe(_warehouse([1, 2, 2, 1, None, 3], [1, 2, 3]), sample_rows=4)
        assert result.status == rv.STATUS_VALID and result.checked_by == "db"
        assert result.match_rate == 100.0 and "first 4 source rows" in result.message
        assert len(asked) == 1


class TestASliceThatMatchesNothing:

    def test_the_whole_table_is_asked_before_the_join_is_marked_down(self):
        """The first rows are unknown members, the rest match."""
        result, asked = _probe(_warehouse([99, 99, 99, 99, 1, 2], [1, 2]), sample_rows=4)
        assert len(asked) == 2
        assert result.status == rv.STATUS_UNTESTED and result.checked_by == "db"
        assert result.join_multiplicity == "" and result.match_rate == -1.0
        assert "not representative" in result.message

    def test_a_join_that_matches_nowhere_is_still_marked_down(self):
        result, asked = _probe(_warehouse([99, 98, 97, 96, 95, 94], [1, 2]), sample_rows=4)
        assert len(asked) == 2
        assert result.status == rv.STATUS_WARNING and result.join_multiplicity == "zero_match"

    def test_a_table_read_to_its_end_needs_no_second_question(self):
        result, asked = _probe(_warehouse([99, 98], [1, 2]), sample_rows=4)
        assert len(asked) == 1
        assert result.join_multiplicity == "zero_match"


def test_an_unrepresentative_slice_neither_stops_the_run_nor_changes_the_join():
    """Three probes that cannot run end a profiling run, and a profiled INNER
    join with orphans becomes LEFT. A slice that proved nothing is neither."""
    unrepresentative = rv.RelationshipValidationResult(
        0, rv.STATUS_UNTESTED, "None of the first rows matched...", checked_by="db")
    suggested = [{"id": n, "status": "suggested", "generated_by": "heuristic",
                  "validation_status": "untested", "join_type": "INNER"} for n in range(1, 6)]
    with patch("store.list_relationships", return_value=suggested), \
            patch.object(rv, "validate_relationship", return_value=unrepresentative) as validate, \
            patch("store.update_relationship_validation") as update, \
            patch("store.set_suggested_join_type") as make_left:
        summary = rv.profile_suggested_relationships("acct")
    assert validate.call_count == 5
    assert summary["untested"] == 5 and summary["not_profiled"] == 0
    assert [call.args[2] for call in update.call_args_list] == [rv.STATUS_UNTESTED] * 5
    make_left.assert_not_called()
