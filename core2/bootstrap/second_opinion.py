"""The AI's second opinion on columns that read like links but that names and values could not place.

A support rep kept as an employee's number (SUPPORT_REP_ID on the customers), a technician who is a pharmacist
(TCHN_KEY on the batches): nothing in the names says which table, and a handful of small numbers fits several
tables' keys, so Learn links neither. The AI reads the names, as a person would, and says which of the tables whose
keys could hold the column's values it refers to. The data decides whether it does: every filled row must find a
member, and every value one, or the link is not kept. A link kept this way is listed for the admin to confirm, with
what the AI read and what the data showed; it is used at once only where no other table's numbers could hold the
values, and otherwise waits for the admin (a reference number that fits an order line's numbers fits by chance).

What leaves for the AI: table and column names, never a value.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any

from core2.bootstrap import names
from core2.bootstrap.calendar import CalendarFinding
from core2.bootstrap.inventory import Inventory
from core2.bootstrap.joins import (_HIERARCHY, VERIFIED, JoinFinding, _compatible, _measure_target, _name_roles,
                                   _names_its_table, _of_its_own_table, _test, counts_things)
from core2.bootstrap.keys import TableKeys
from core2.bootstrap.profiler import TableProfile
from core2.model.schema import ColumnProfile, Evidence
from core2.warehouse.runner import Warehouse

log = logging.getLogger(__name__)

Complete = Callable[[str, str], str]

SYSTEM = """You read the names in a database. Each column listed holds the key of a row in another table, or of \
another row in its own table, but its name does not say which table: SUPPORT_REP_ID on customers may hold an \
employee's number, TECHNICIAN_ID on batches a pharmacist's. For each column, pick the one candidate table whose \
rows it refers to, judging by the names alone, or null when none fits or the column is not a reference at all (a \
code of its own, a count, a score). Reply with JSON only: {"links": {"<table>.<column>": "<candidate table>" or \
null}}."""

MAX_CANDIDATES = 12          # tables offered for one column; more says the values decide nothing
MAX_COLUMNS = 60             # columns asked about in one Learn
_PERSONAL = {"email", "phone", "national_id", "street"}


def _pointer_like(words: list[str]) -> bool:
    """Named as a reference: ending in a key word (TCHN_KEY, SUPPORT_REP_ID), or a step up (REPORTS_TO)."""
    return bool(words) and (words[-1] in names.KEY_SUFFIXES or bool(_HIERARCHY & set(words)))


def _fits(a: ColumnProfile, k: ColumnProfile) -> bool:
    """Could every value of ``a`` be one of ``k``'s, judging by the profiles alone?"""
    if a.distinct < 2 or a.distinct > k.distinct * 1.05 + 5:
        return False
    if a.min_num is not None and a.max_num is not None and k.min_num is not None and k.max_num is not None:
        return k.min_num <= a.min_num and a.max_num <= k.max_num
    if a.max_len is not None and k.max_len is not None:
        return a.max_len <= k.max_len + 2 and (a.min_len or 0) + 2 >= (k.min_len or 0)
    return False


def unplaced(inventory: Inventory, profiles: dict[str, TableProfile], keys: dict[str, TableKeys],
             calendars: dict[str, CalendarFinding], joins: list[JoinFinding]) -> list[tuple[str, str, list[tuple[str, str]]]]:
    """(table, column, candidate targets) of every column named like a reference that no link starts from."""
    linked = {(f.from_table, c) for f in joins if f.trust != "rejected" for c in f.from_columns}
    targets = [(key, keys[key].primary_key[0]) for key in inventory.tables
               if key in profiles and len(keys[key].primary_key) == 1 and key not in calendars]
    out = []
    for key, table in inventory.tables.items():
        if key not in profiles or key in calendars:
            continue
        tkeys = keys[key]
        numbering = set(tkeys.line_numbers) | {c for alt in tkeys.alternate_keys for c in alt[1:]}
        for column in table.columns:
            a = profiles[key].columns.get(column.name)
            words = names.tokens(column.name)
            if a is None or (key, column.name) in linked or column.name in numbering \
                    or column.data_type not in ("integer", "text") or not _pointer_like(words) \
                    or a.pattern in _PERSONAL or (a.pattern or "").startswith("flag") or counts_things(column.name) \
                    or _names_its_table(column.name, table.name) or _of_its_own_table(column.name, table.name) \
                    or column.name in tkeys.unique_columns:
                continue      # the table's own number or alternate key (an order header's SalesOrderID): its own
            fits = [(to, k) for to, k in targets if (to, k) != (key, column.name)
                    and _compatible(column.data_type, inventory.tables[to].type_of(k))
                    and _fits(a, profiles[to].columns[k])]
            if 0 < len(fits) <= MAX_CANDIDATES:
                out.append((key, column.name, fits))
    return out[:MAX_COLUMNS]


def _brief(inventory: Inventory, asked: list[tuple[str, str, list[tuple[str, str]]]]) -> str:
    def other_columns(table_key: str, but: str, limit: int) -> list[str]:
        return [c.name for c in inventory.tables[table_key].columns if c.name != but][:limit]

    return "Columns:\n" + json.dumps([{
        "column": f"{inventory.tables[key].name}.{column}",
        "its table's other columns": other_columns(key, column, 12),
        "candidates": [{"table": inventory.tables[to].name, "key": k, "columns": other_columns(to, k, 8)}
                       for to, k in fits],
    } for key, column, fits in asked], ensure_ascii=False)


def _parse(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        value = json.loads(cleaned)
    except ValueError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        try:
            value = json.loads(cleaned[start:end + 1]) if 0 <= start < end else {}
        except ValueError:
            value = {}
    return value if isinstance(value, dict) else {}


def second_opinion(warehouse: Warehouse, inventory: Inventory, profiles: dict[str, TableProfile],
                   keys: dict[str, TableKeys], calendars: dict[str, CalendarFinding], joins: list[JoinFinding],
                   complete: Complete) -> list[JoinFinding]:
    """The links the AI reads into columns no name or value placed, each kept only where the data holds it."""
    asked = unplaced(inventory, profiles, keys, calendars, joins)
    if not asked:
        return []
    try:
        reply = _parse(complete(SYSTEM, _brief(inventory, asked)))
    except Exception as exc:  # noqa: BLE001 - the links stay as names and values found them
        log.warning("core2 joins: the AI's second opinion failed: %s", exc)
        return []
    picks = reply.get("links")
    picks = picks if isinstance(picks, dict) else {}
    kept: list[JoinFinding] = []
    for key, column, fits in asked:
        table = inventory.tables[key]
        chosen = picks.get(f"{table.name}.{column}")
        if not isinstance(chosen, str):
            continue
        target = next(((to, k) for to, k in fits if inventory.tables[to].name.casefold() == chosen.casefold()), None)
        if target is None:
            continue                 # a table the values could not fit, or none offered: not a guess to test
        to, k = target
        a, t = profiles[key].columns[column], profiles[to].columns[k]
        # Its name is the role it plays for the target: a support rep, among the employees.
        found = JoinFinding(key, column, to, k, coverage=min(1.0, a.distinct / t.distinct) if t.distinct else 0.0,
                            from_unique=column in keys[key].unique_columns,
                            role_tokens=[] if names.opaque(column) else names.core_column(column))
        try:
            _test(warehouse, inventory, found, profiles, [])
        except Exception as exc:  # noqa: BLE001 - untested, so not kept
            log.warning("core2 joins: could not test %s.%s -> %s: %s", table.name, column, chosen, exc)
            continue
        if found.match_rate < VERIFIED or found.unmatched_values:
            continue                 # the AI's reading, and the data says otherwise
        target_name = inventory.tables[to].name
        found.trust = "proposed"
        found.evidence += [
            Evidence(kind="ai_opinion", weight=0.5, detail=f"the AI reads {column} as a {target_name} ({k})"),
            Evidence(kind="test_run", weight=found.match_rate, data={
                "rows": found.rows, "non_null": found.non_null, "unmatched": 0, "unmatched_values": 0,
                "placeholders": found.placeholder_rows}, detail=f"{found.match_rate:.1%} of {found.non_null:,} "
                "filled rows match"),
        ]
        if len(fits) > 1:
            # Its values would fit other tables' numbers as well (an order line's number holds every small
            # reference number): the AI's reading is the only thing choosing, so it waits for the admin.
            found.evidence.append(Evidence(kind="ambiguous", weight=-0.5, detail=(
                f"{column}'s values would fit {len(fits) - 1} other table(s) too; the AI's reading waits for an admin")))
        _measure_target(warehouse, inventory, keys, calendars, found)
        kept.append(found)
    _name_roles([*joins, *kept], inventory)
    return kept
