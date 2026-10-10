"""Links read from the warehouse's own query history: the joins its people and tools already write.

Analysts, reports and loading jobs join the warehouse's tables every day, and the warehouse keeps
their queries for a while (Snowflake's QUERY_HISTORY, Azure SQL's Query Store, Oracle's V$SQL).
Learn reads the SELECT statements its sign-in may see and keeps only which column each one compares
with which, table to table, and in how many queries. Nothing else of a query is kept or sent anywhere:
not its text, its values, nor who ran it. Each pair becomes a candidate link, tested on the data like
any other; the count is its evidence, and a link seen in two queries or more is one people use, whatever
its names say (a support rep who is an employee, a manager who reports to one).

A join to a subquery is not counted: Learn's own tests join a table to the target's keys read in a
subquery, so they never vouch for themselves. Where the warehouse says who ran a query (Snowflake, Oracle),
QueryBot's own sign-in and sessions are left out too. Azure SQL's Query Store does not say, so QueryBot's own
answers are among its queries: that is why history only adds a link to test and settles a tie between two
that test alike, and never excuses a link the data does not bear out. A warehouse that keeps no history, or
a sign-in not allowed to read it, leaves the step out, named in the journal.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable

import sqlglot
from sqlglot import exp

from core2.bootstrap.inventory import Inventory
from core2.bootstrap.journal import journal_of
from core2.bootstrap.keys import TableKeys
from core2.warehouse.runner import Warehouse

log = logging.getLogger(__name__)

MIN_QUERIES = 2           # a join one query writes may be a mistake; two queries make it a habit
MAX_QUERIES = 20_000
MAX_LENGTH = 200_000      # characters: a generated query longer than this is not read

# The latest SELECT statements, newest first, QueryBot's own sign-in left out where the warehouse says who ran each.
_READS = {
    "snowflake": [
        "SELECT query_text FROM snowflake.account_usage.query_history "
        "WHERE start_time > DATEADD(day, -90, CURRENT_TIMESTAMP()) AND query_type = 'SELECT' "
        "AND execution_status = 'SUCCESS' AND user_name <> CURRENT_USER() AND COALESCE(query_tag, '') <> 'QueryBot' "
        "ORDER BY start_time DESC LIMIT 20000",
        "SELECT query_text FROM TABLE(information_schema.query_history(result_limit => 10000)) "
        "WHERE query_type = 'SELECT' AND execution_status = 'SUCCESS' AND user_name <> CURRENT_USER() "
        "AND COALESCE(query_tag, '') <> 'QueryBot'",
    ],
    "tsql": [
        "SELECT TOP 20000 qt.query_sql_text FROM sys.query_store_query_text AS qt "
        "JOIN sys.query_store_query AS q ON q.query_text_id = qt.query_text_id "
        "ORDER BY q.last_execution_time DESC",
    ],
    "oracle": [
        "SELECT DBMS_LOB.SUBSTR(sql_fulltext, 4000, 1) FROM v$sql WHERE command_type = 3 "
        "AND parsing_user_id <> UID AND parsing_schema_name NOT IN ('SYS', 'SYSTEM') AND ROWNUM <= 20000",
    ],
}

# A link as history shows it: (from table, from columns, to table, to columns), table keys of the inventory.
Seen = tuple[str, tuple[str, ...], str, tuple[str, ...]]


def read_query_log(warehouse: Warehouse) -> list[str]:
    """The SELECT statements the warehouse keeps, as far as the sign-in may read them; none where it may not."""
    journal = journal_of(warehouse)
    reads = _READS.get(warehouse.dialect, [])
    if reads:
        journal.step("Reading which columns the warehouse's own queries join (only that: no query text or value "
                     "is kept)")
    reasons = []
    for sql in reads:
        try:
            rows = warehouse.query(sql, max_rows=MAX_QUERIES).rows
        except Exception as error:  # noqa: BLE001 - no grant to read the history: the step is left out
            reasons.append(str(error).splitlines()[0][:200] if str(error) else type(error).__name__)
            continue
        return [r[0] for r in rows if isinstance(r[0], str) and len(r[0]) <= MAX_LENGTH]
    if reasons:
        # Not data left out: a help the sign-in has no grant for. Links still come from names and values.
        log.info("core2 learn: the query history was not read: %s", reasons[-1])
        journal.step(f"The warehouse's query history could not be read ({reasons[-1]}): links come from names "
                     "and values alone")
    return []


def links_in(texts: Iterable[str], inventory: Inventory, keys: dict[str, TableKeys], dialect: str) -> Counter:
    """How many queries join each pair of columns, read as links: the side holding a unique key is pointed at."""
    tables = _table_index(inventory)
    seen: Counter = Counter()
    unread = 0
    for text in texts:
        found: set[Seen] = set()
        try:
            for statement in sqlglot.parse(text, read=dialect, error_level=sqlglot.ErrorLevel.IGNORE):
                if statement is not None:
                    for select in statement.find_all(exp.Select):
                        found |= _links_of(select, tables, inventory, keys)
        except Exception:  # noqa: BLE001 - a query sqlglot cannot read is not counted; the others still are
            unread += 1
            continue
        seen.update(found)
    if unread:
        log.info("core2 learn: %d of %d history queries could not be read", unread, len(texts))
    return seen


def _table_index(inventory: Inventory) -> dict[str, list[str]]:
    by_name: dict[str, list[str]] = {}
    for key, table in inventory.tables.items():
        by_name.setdefault(table.name.casefold(), []).append(key)
    return by_name


def _resolve(node: exp.Table, tables: dict[str, list[str]], inventory: Inventory) -> str | None:
    keys = tables.get(node.name.casefold(), [])
    if node.db:
        keys = [k for k in keys if inventory.tables[k].schema.casefold() == node.db.casefold()]
    return keys[0] if len(keys) == 1 else None


def _links_of(select: exp.Select, tables: dict[str, list[str]], inventory: Inventory,
              keys: dict[str, TableKeys]) -> set[Seen]:
    aliases: dict[str, str] = {}       # alias -> table key, for the tables this SELECT reads directly
    sources = [select.args.get("from_") or select.args.get("from"), *(select.args.get("joins") or [])]
    for source in sources:
        node = source.this if source is not None else None
        if isinstance(node, exp.Table) and node.name:
            key = _resolve(node, tables, inventory)
            if key:
                aliases[node.alias_or_name.casefold()] = key
    if len(aliases) < 2:
        return set()
    conditions = [j.args.get("on") for j in select.args.get("joins") or []] + [select.args.get("where")]
    pairs: dict[tuple[str, str], list[tuple[str, str]]] = {}     # (alias, alias) -> [(column, column)]
    for condition in conditions:
        if condition is None:
            continue
        for eq in condition.find_all(exp.EQ):
            if eq.find_ancestor(exp.Select) is not select:
                continue       # a subquery's own comparison, with its own tables
            left, right = _column(eq.this, aliases, inventory), _column(eq.expression, aliases, inventory)
            if left and right and left[0] != right[0]:
                if left[0] > right[0]:
                    left, right = right, left      # a.x = b.y and b.z = a.w compare the same two tables
                pairs.setdefault((left[0], right[0]), []).append((left[1], right[1]))
    out: set[Seen] = set()
    for (a, b), columns in pairs.items():
        link = _as_link(aliases[a], aliases[b], columns, keys, inventory)
        if link:
            out.add(link)
    return out


def _column(node: exp.Expression, aliases: dict[str, str], inventory: Inventory) -> tuple[str, str] | None:
    """(alias, column as the table spells it) of a plain column, or of one wrapped in a cast or a trim."""
    while isinstance(node, (exp.Cast, exp.TryCast, exp.Trim, exp.Paren)):
        node = node.this
    if not isinstance(node, exp.Column) or not node.name:
        return None
    if node.table:
        alias = node.table.casefold()
        if alias not in aliases:
            return None
        candidates = [alias]
    else:
        candidates = [a for a, k in aliases.items() if inventory.tables[k].column(node.name) is not None]
        if len(candidates) != 1:
            return None        # an unqualified column two of the tables hold: not read
    column = inventory.tables[aliases[candidates[0]]].column(node.name)
    return (candidates[0], column.name) if column is not None else None


def _as_link(a: str, b: str, columns: list[tuple[str, str]], keys: dict[str, TableKeys],
             inventory: Inventory) -> Seen | None:
    """The two tables' compared columns as a link: towards the side whose columns are its key."""
    left, right = tuple(c for c, _ in columns), tuple(c for _, c in columns)

    def keyed(table: str, cols: tuple[str, ...]) -> int:
        """2: the table's primary key; 1: another unique key of it; 0: neither."""
        k = keys.get(table)
        if k is None:
            return 0
        if sorted(cols) == sorted(k.primary_key):
            return 2
        if len(cols) == 1 and cols[0] in k.unique_columns or any(sorted(cols) == sorted(alt) for alt in k.alternate_keys):
            return 1
        return 0

    on_left, on_right = keyed(a, left), keyed(b, right)
    if not on_left and not on_right:
        return None            # two lists joined on a shared value (a region): no link
    if on_right > on_left or on_right == on_left and _names(inventory, b, right):
        return a, left, b, right
    if on_left > on_right or _names(inventory, a, left):
        return b, right, a, left
    return None


def _names(inventory: Inventory, table: str, columns: tuple[str, ...]) -> bool:
    from core2.bootstrap.joins import _names_its_table

    return len(columns) == 1 and _names_its_table(columns[0], inventory.tables[table].name)
