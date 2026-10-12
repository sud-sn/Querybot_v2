"""How the old question pipeline writes a PostgreSQL table or column name.

PostgreSQL folds a name written without quotes to small letters, and a name
created with capitals ("OrderLines") is found again only quoted, in its exact
case. The old pipeline carries most names upper-cased -- the table and column
maps it plans from (core.schema.load_known_tables, load_schema_columns) key
everything that way -- so the case a quoted PostgreSQL name needs is lost
before any SQL is written. Quoting what arrives, as Snowflake and Oracle are
written, would ask for "ORDER_LINES" where the table is order_lines.

A name is therefore written:

  * as it arrives when it has small letters in it: it came from _schema.json,
    the entity graph or a date role, which keep names as they were created;
  * else in the spelling discovery recorded for it, when the workspace's
    spellings are active (handle_query activates them for a question, and the
    admin checks for theirs);
  * else in small letters, which is what PostgreSQL made of any name created
    without quotes -- by far the usual case;

and is quoted only when it has to be (core2.warehouse.dialect.quote). A table
keeps its last two parts, schema and table: a PostgreSQL query cannot name
another database, and the database part the old pipeline carries is upper-cased
anyway.
"""

from __future__ import annotations

import json
import logging
import threading
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

log = logging.getLogger("querybot.postgres_names")

DB_TYPES = frozenset({"postgres", "postgresql"})


def is_postgres(db_type: object) -> bool:
    return str(db_type or "").strip().lower() in DB_TYPES


@dataclass(frozen=True)
class Spellings:
    """A workspace's names as its PostgreSQL database spells them."""

    names: dict[str, str] = field(default_factory=dict)      # NAME upper-cased -> as created
    schema_of: dict[str, str] = field(default_factory=dict)  # TABLE upper-cased -> its one schema


_ACTIVE: ContextVar[Spellings | None] = ContextVar("querybot_postgres_spellings", default=None)


def spellings_from_master(master: dict | None) -> Spellings:
    """The spellings of a PostgreSQL _schema.json: every schema, table and column
    name, keyed upper-cased; a name spelled two ways keeps the first. A table
    name found in one schema only also records that schema, so a bare table
    name can be qualified."""
    names: dict[str, str] = {}
    schemas: dict[str, set[str]] = {}
    for key, info in (master or {}).items():
        if str(key).startswith("__") or not isinstance(info, dict):
            continue
        table = str(key).split(".")[-1]
        schema = str(info.get("schema") or "")
        names.setdefault(table.upper(), table)
        if schema:
            names.setdefault(schema.upper(), schema)
            schemas.setdefault(table.upper(), set()).add(schema)
        for column in info.get("columns") or []:
            name = str((column.get("name") if isinstance(column, dict) else column) or "")
            if name:
                names.setdefault(name.upper(), name)
    return Spellings(
        names=names,
        schema_of={table: next(iter(found)) for table, found in schemas.items() if len(found) == 1},
    )


# Read once per version of a _schema.json: a join check activates them for each relationship.
_READ: dict[tuple[str, int, int], Spellings] = {}
_READ_LOCK = threading.Lock()


def spellings_from_schema(schema_dir: str) -> Spellings:
    path = Path(schema_dir or "") / "_schema.json"
    if not schema_dir or not path.exists():
        return Spellings()
    try:
        stat = path.stat()
        key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
        with _READ_LOCK:
            found = _READ.get(key)
        if found is None:
            found = spellings_from_master(json.loads(path.read_text(encoding="utf-8")))
            with _READ_LOCK:
                if len(_READ) >= 32:
                    _READ.clear()
                _READ[key] = found
        return found
    except Exception as exc:
        log.warning(
            "Could not read the PostgreSQL names of %s: %s -- names the old pipeline carries "
            "upper-cased are written in small letters", path, exc,
        )
        return Spellings()


def workspace_spellings(db_cfg: dict | None, state: dict | None) -> Spellings | None:
    """The spellings of a workspace's discovered schema when its database is PostgreSQL, else None."""
    if not is_postgres((db_cfg or {}).get("db_type")):
        return None
    return spellings_from_schema(str((state or {}).get("schema_dir") or ""))


def account_spellings(account_id: str) -> Spellings | None:
    """workspace_spellings for an admin check that has only the account: its saved connection and
    its discovered schema."""
    try:
        import store

        client = store.get_client(account_id) or {}
        cfg = store.get_db_config(client["db_config_id"]) if client.get("db_config_id") else None
        if not is_postgres((cfg or {}).get("db_type")):
            return None
        return workspace_spellings(cfg, store.get_client_state(account_id) or {})
    except Exception as exc:
        log.warning("Could not read the PostgreSQL names of account %s: %s", account_id, exc)
        return None


def activate(spellings: Spellings | None) -> Token:
    """Make ``spellings`` the ones names are written in, for this context; returns a token for deactivate()."""
    return _ACTIVE.set(spellings)


def activate_for_workspace(db_cfg: dict | None, state: dict | None) -> Token:
    """Activate the workspace's spellings when its database is PostgreSQL, none otherwise."""
    return activate(workspace_spellings(db_cfg, state))


def deactivate(token: Token | None) -> None:
    if token is None:
        return
    try:
        _ACTIVE.reset(token)
    except Exception:
        pass


@contextmanager
def using(spellings: Spellings | None) -> Iterator[None]:
    token = activate(spellings)
    try:
        yield
    finally:
        deactivate(token)


def _clean(part: object) -> str:
    text = str(part or "").strip()
    if len(text) >= 2 and text[0] == text[-1] == '"':
        return text[1:-1].replace('""', '"')
    return text.strip('[]"`')


def spelling(name: object) -> str:
    """The name as the database spells it (see the module docstring), unquoted."""
    clean = _clean(name)
    if not clean or any(ch.islower() for ch in clean):
        return clean
    active = _ACTIVE.get()
    found = active.names.get(clean.upper()) if active else None
    return found or clean.lower()


def ident(name: object) -> str:
    """One table, schema or column name as PostgreSQL SQL."""
    from core2.warehouse.dialect import quote

    clean = spelling(name)
    return quote(clean, "postgres") if clean else ""


def table(ref: object, schema: object = "") -> str:
    """A table as PostgreSQL SQL, from any reference the old pipeline carries:
    DATABASE.SCHEMA.TABLE, SCHEMA.TABLE or TABLE, quoted or not; ``schema``
    qualifies a bare name. Written schema.table, the database left out."""
    parts = [part for part in (_clean(piece) for piece in _split(str(ref or ""))) if part]
    if not parts:
        return ""
    if len(parts) == 1 and _clean(schema):
        parts = [_clean(schema), parts[0]]
    if len(parts) == 1:
        active = _ACTIVE.get()
        known = active.schema_of.get(spelling(parts[0]).upper()) if active else None
        if known:
            parts = [known, parts[0]]
    return ".".join(ident(part) for part in parts[-2:])


def _split(ref: str) -> list[str]:
    """A dotted name's parts; a dot inside double quotes or brackets is part of a name."""
    parts: list[str] = []
    current: list[str] = []
    closer = ""
    for ch in ref:
        if closer:
            current.append(ch)
            if ch == closer:
                closer = ""
            continue
        if ch == '"':
            closer = '"'
        elif ch == "[":
            closer = "]"
        if ch == ".":
            parts.append("".join(current))
            current = []
            continue
        current.append(ch)
    parts.append("".join(current))
    return parts
