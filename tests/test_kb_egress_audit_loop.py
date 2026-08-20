"""
tests/test_kb_egress_audit_loop.py

Found in the live build log on the test server, not in the audit:

    WARNING querybot.admin — KB egress log (kb_build) write failed for
    Emco_test: 'list' object has no attribute 'get'

_schema.json holds the discovered tables AND a handful of non-table entries
alongside them — "__db_fk_constraints__" is a LIST of the database's declared
foreign keys. core.schema._normalize_schema already knows this and passes any
"__"-prefixed key through untouched; the two egress-audit loops in admin/routes
did not, so one of them reached a list, called .get() on it, and aborted.

The exception was caught a level up and logged as a warning, so the build
reported success. What was lost is the compliance record of which tables had
real rows sent to the LLM — for every table after that key in iteration order.
On Azure the FK entry is appended last and all 14 tables happened to be written
first; the same code on a schema where it lands earlier loses the rest silently.
"""

from __future__ import annotations

import inspect
import pathlib
import re

import admin.routes as routes
from core.schema import _normalize_schema


def _audit_loop_bodies() -> list[str]:
    """The two egress-audit loops, taken from the module source."""
    source = inspect.getsource(routes)
    return re.findall(
        r"for _tkey, _tmeta in _schema\.items\(\):(.*?)store\.log_kb_egress\(",
        source,
        re.S,
    )


def test_both_audit_loops_exist():
    assert len(_audit_loop_bodies()) == 2, (
        "expected the discovery and kb_build egress-audit loops"
    )


def test_neither_loop_calls_get_on_a_non_table_entry():
    """Both loops now narrow to real tables up front, via the shared helper —
    as does schema-drift detection, the third site with the same defect."""
    source = inspect.getsource(routes)
    assert source.count("discovered_tables as _tables_only") == 3
    for body in _audit_loop_bodies():
        assert 'str(_tkey).startswith("__")' in body
        assert "isinstance(_tmeta, dict)" in body


def test_schema_drift_only_diffs_real_tables():
    """Same list, third site: the drift report crashed on it and the failure
    was swallowed into a warning, so a discovery that DID change the schema
    reported no drift at all."""
    source = inspect.getsource(routes._compute_schema_drift)
    assert "discovered_tables as _tables_only" in source

    drift = routes._compute_schema_drift(
        {"DB.S.T1": {"columns": [{"name": "A", "type": "int"}]},
         "__db_fk_constraints__": [{"from": "x"}]},
        {"DB.S.T1": {"columns": [{"name": "A", "type": "int"},
                                 {"name": "B", "type": "int"}]},
         "DB.S.T2": {"columns": []},
         "__db_fk_constraints__": [{"from": "x"}, {"from": "y"}]},
    )
    assert drift["added_tables"] == ["DB.S.T2"]
    assert drift["removed_tables"] == []
    assert "__db_fk_constraints__" not in str(drift["added_tables"])
    assert drift["column_changes"]["DB.S.T1"]["added"] == ["B"]


def test_the_discovery_counter_counts_tables_only():
    """Discovery reported "15/14 selected tables written" — the FK entry was
    being counted as a table."""
    from core.schema import discovered_tables

    master = {
        "DB.S.T1": {"columns": []},
        "DB.S.T2": {"columns": []},
        "__db_fk_constraints__": [{"from": "x"}],
    }
    assert len(discovered_tables(master)) == 2

    source = pathlib.Path("core/schema.py").read_text(encoding="utf-8")
    assert source.count("return len(discovered_tables(master))") == 3
    assert "    return len(master)" not in source


def test_the_fk_entry_really_is_a_list_after_normalisation():
    """The guard has to hold against what _normalize_schema actually returns —
    it deliberately preserves "__"-prefixed values as they are."""
    normalised = _normalize_schema({
        "DB.SCHEMA.TABLE": {"columns": [{"name": "A"}]},
        "__db_fk_constraints__": [{"from": "A", "to": "B"}],
    })
    assert isinstance(normalised["__db_fk_constraints__"], list)
    assert isinstance(normalised["DB.SCHEMA.TABLE"], dict)


def test_a_legacy_bare_column_list_is_still_wrapped_not_skipped():
    """A table whose value is a plain list is the OLD storage format and
    _normalize_schema wraps it into {"columns": [...]}. That must keep being
    audited — the skip is for non-table entries, not for legacy tables."""
    normalised = _normalize_schema({"DB.SCHEMA.OLD": [{"name": "A"}]})
    assert normalised["DB.SCHEMA.OLD"] == {"columns": [{"name": "A"}]}
    assert not str("DB.SCHEMA.OLD").startswith("__")


def test_the_audit_failure_is_reported_at_error_level():
    """It is the record of what was sent to the LLM. A warning that the build
    then reports as successful is how this went unnoticed."""
    source = inspect.getsource(routes)
    assert source.count("KB egress audit log") == 2
    assert 'log.warning("KB egress log (' not in source
