"""Where a row rule reaches: the tables that point at the table it is written on.

A rule written on a field of one table ("a store's region is West") is a rule on that table's rows and on every
row that points at one of them. A question on sales alone never names the stores table; written only where its
table is named, the rule left that question every region's sales. Now the sales rows count only where their store
is one of the reader's: ``sales.store_id IN (SELECT store_id FROM stores WHERE region = 'West')``.

The links are the learned model's (core2): a link from many rows to one ("each sale has one store"), followed
from the table a question reads towards the table the rule is on, through at most two tables between ("sales ->
stores -> regions"). A link the admin rejected, or one to a calendar, is never followed. Never the other way:
a rule on sales does not hide a store from the list of stores. A workspace without a learned model has no links,
and its rules apply where their table is named, as they always did.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

MOST_HOPS = 3


@dataclass(frozen=True)
class Link:
    """One step from a table to the table its column points at, named as the warehouse spells them."""
    table: str                       # "SCHEMA.SALES", upper-cased, as a query's tables are compared
    column: str                      # "STORE_ID", as the warehouse spells it
    to_table: str                    # "SCHEMA.STORES"
    to_column: str                   # "STORE_ID"
    to_parts: tuple[str, str, str]   # (database, schema, name) of to_table, as the warehouse spells them


def _named(table) -> str:
    return ".".join(p for p in (table.schema_name, table.name) if p).upper()


def from_model(model) -> list[Link]:
    """The links a rule may follow, read off a learned model."""
    out: list[Link] = []
    for join in model.joins.values():
        if join.trust == "rejected" or join.to_calendar or len(join.from_columns) != 1 or len(join.to_columns) != 1:
            continue
        if join.cardinality not in ("many_to_one", "one_to_one"):
            continue
        source, target = model.tables.get(join.from_table), model.tables.get(join.to_table)
        column, key = model.columns.get(join.from_columns[0]), model.columns.get(join.to_columns[0])
        if source is None or target is None or column is None or key is None or source.key == target.key:
            continue
        out.append(Link(_named(source), column.name, _named(target), key.name,
                        (target.database, target.schema_name, target.name)))
    return out


def links(account_id: str) -> list[Link]:
    """The workspace's links, from the model its questions are answered with; none when it has no model."""
    try:
        import store
        from core2.bootstrap.service import answering_model

        client = store.get_client(account_id) or {}
        if not client.get("db_config_id"):
            return []
        model = answering_model(account_id, client)
    except Exception as exc:  # noqa: BLE001 - rules still apply where their table is named
        log.warning("row rules: the links of %s could not be read: %s", account_id, exc)
        return []
    return from_model(model) if model is not None else []


def same_table(name: str, other: str) -> bool:
    """One table under two spellings: with or without its schema or database in front."""
    name, other = name.upper(), other.upper()
    return name == other or name.endswith("." + other) or other.endswith("." + name)


def path(start: str, goal: str, found: list[Link]) -> list[Link] | None:
    """The shortest way from the table a query reads to the table a rule is on, following links from many rows
    to one; None when there is none within MOST_HOPS."""
    if same_table(start, goal):
        return []
    frontier: list[tuple[str, list[Link]]] = [(start, [])]
    seen = {start.upper()}
    for _ in range(MOST_HOPS):
        nxt: list[tuple[str, list[Link]]] = []
        for here, steps in frontier:
            for link in found:
                if not same_table(link.table, here) or link.to_table in seen:
                    continue
                way = [*steps, link]
                if same_table(link.to_table, goal):
                    return way
                seen.add(link.to_table)
                nxt.append((link.to_table, way))
        frontier = nxt
    return None
