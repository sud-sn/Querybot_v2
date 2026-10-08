"""What the database declares about itself: tables, columns, types and keys.

This is the bootstrap's starting point and the only part that trusts declarations.
In production it is read from what discovery already wrote (``_schema.json``);
for DuckDB it is read from the catalog. Everything after this step is checked
against the data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from core2 import ids
from core2.warehouse.runner import Warehouse


@dataclass
class InvColumn:
    name: str
    raw_type: str
    data_type: str          # core2.model.schema.DataType
    nullable: bool = True
    comment: str = ""


@dataclass
class InvTable:
    database: str
    schema: str
    name: str
    columns: list[InvColumn]
    primary_key: list[str] = field(default_factory=list)
    row_count: int | None = None
    comment: str = ""
    source_key: str = ""                    # the table's key in discovery's _schema.json
    masked: set[str] = field(default_factory=set)   # columns discovery masks
    mask_all: bool = False

    @property
    def key(self) -> str:
        return ids.table_key(self.database, self.schema, self.name)

    def column(self, name: str) -> InvColumn | None:
        wanted = ids.norm(name)
        return next((c for c in self.columns if ids.norm(c.name) == wanted), None)

    def type_of(self, name: str) -> str:
        """The column's core2 data type, or "" when the table has no such column."""
        column = self.column(name)
        return column.data_type if column else ""


@dataclass
class InvForeignKey:
    table: str              # table key
    columns: list[str]
    ref_table: str          # table key
    ref_columns: list[str]
    name: str = ""
    enforced: bool = False


@dataclass
class Inventory:
    tables: dict[str, InvTable]
    foreign_keys: list[InvForeignKey] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)   # what was corrected in what discovery wrote


_INTEGER = re.compile(r"^(BIG|SMALL|TINY|MEDIUM)?INT(EGER)?\d*$|^(U?INT\d+|HUGEINT|UBIGINT|USMALLINT|UTINYINT|LONG|BYTEINT)$")


def normalize_type(raw: str) -> str:
    """A warehouse type name as one of core2's data types."""
    text = (raw or "").strip().upper()
    base = re.sub(r"\(.*", "", text).strip()
    args = re.findall(r"\d+", text[len(base):]) if "(" in text else []
    if _INTEGER.match(base.replace(" ", "")):
        return "integer"
    if base in ("NUMBER", "NUMERIC", "DECIMAL", "DEC"):
        if len(args) >= 2 and int(args[1]) == 0:
            return "integer"
        return "decimal"
    if base in ("MONEY", "SMALLMONEY"):
        return "decimal"
    if base in ("FLOAT", "DOUBLE", "DOUBLE PRECISION", "REAL", "FLOAT4", "FLOAT8", "BINARY_FLOAT", "BINARY_DOUBLE"):
        return "float"
    if base in ("BOOLEAN", "BOOL", "BIT"):
        return "boolean"
    if base == "DATE":
        return "date"
    if base.startswith("TIMESTAMP") or base.startswith("DATETIME") or base in ("SMALLDATETIME", "DATETIMEOFFSET"):
        return "timestamp"
    if base in ("VARCHAR", "CHAR", "NVARCHAR", "NCHAR", "TEXT", "NTEXT", "STRING", "VARCHAR2", "NVARCHAR2",
                "CHARACTER", "CHARACTER VARYING", "CLOB", "NCLOB", "UUID", "UNIQUEIDENTIFIER"):
        return "text"
    return "other"


def from_duckdb(warehouse: Warehouse, *, schema: str = "main",
                declared_fks: list[dict[str, Any]] | None = None) -> Inventory:
    """The inventory of one DuckDB schema; ``declared_fks`` as discovery reports them."""
    q = warehouse.query
    database = q("SELECT current_database()").rows[0][0]
    tables: dict[str, InvTable] = {}
    for (name,) in q(f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{schema}' "
                     "AND table_type = 'BASE TABLE' ORDER BY table_name").rows:
        cols = q(f"SELECT column_name, data_type, is_nullable FROM information_schema.columns "
                 f"WHERE table_schema = '{schema}' AND table_name = '{name}' ORDER BY ordinal_position").rows
        table = InvTable(database=database, schema=schema, name=name,
                         columns=[InvColumn(c, t, normalize_type(t), n == "YES") for c, t, n in cols])
        tables[table.key] = table
    for table_name, constraint, columns in q(
            f"SELECT table_name, constraint_type, constraint_column_names FROM duckdb_constraints() "
            f"WHERE schema_name = '{schema}' AND constraint_type = 'PRIMARY KEY'").rows:
        key = ids.table_key(database, schema, table_name)
        if key in tables:
            tables[key].primary_key = list(columns)
    fks: list[InvForeignKey] = []
    for fk in declared_fks or []:
        parent = ids.table_key(database, fk.get("parent_schema") or schema, fk["parent_table"])
        ref = ids.table_key(database, fk.get("ref_schema") or schema, fk["ref_table"])
        if parent in tables and ref in tables:
            fks.append(InvForeignKey(parent, [fk["parent_col"]], ref, [fk["ref_col"]],
                                     fk.get("constraint_name", ""), bool(fk.get("enforced"))))
    return Inventory(tables=tables, foreign_keys=fks)


def from_schema_json(schema: dict[str, Any], db_type: str) -> Inventory:
    """The inventory discovery already wrote (``clients/<account>/schema/_schema.json``).

    The file's own keys differ by warehouse (Azure ``DB.SCHEMA.Table``, Snowflake
    and Oracle the bare name unless it repeats), so each table's full name is
    rebuilt from the entry's database / schema / owner fields. Metadata keys
    (``__...``) and non-table entries are skipped.
    """
    tables: dict[str, InvTable] = {}
    notes: list[str] = []
    by_schema_and_name: dict[tuple[str, str], str] = {}
    for file_key, entry in schema.items():
        if file_key.startswith("__") or not isinstance(entry, dict):
            continue
        parts = file_key.split(".")
        name = parts[-1]
        if db_type == "oracle":
            database, owner = "", entry.get("owner") or (parts[-2] if len(parts) > 1 else "")
        else:
            database = entry.get("database") or (parts[-3] if len(parts) > 2 else "")
            owner = entry.get("schema") or (parts[-2] if len(parts) > 1 else "")
        raw_columns = entry.get("columns") or []
        columns: list[InvColumn] = []
        for column in raw_columns:
            if isinstance(column, str):
                found = InvColumn(column, "", "other")
            else:
                raw = str(column.get("type") or "")
                nullable = column.get("nullable")
                found = InvColumn(str(column.get("name")), raw, normalize_type(raw),
                                  nullable is not False and str(nullable).upper() not in ("NO", "N", "FALSE"),
                                  str(column.get("comment") or ""))
            # A column listed twice (discovery merging two sources) is one column: named twice in
            # one query it is refused (Azure SQL 8156, "specified multiple times").
            if not any(c.name.casefold() == found.name.casefold() for c in columns):
                columns.append(found)
        rows = entry.get("row_count")
        table = InvTable(database=database, schema=owner, name=name, columns=columns,
                         primary_key=_declared_key(name, entry.get("pk_columns") or [], columns, notes),
                         row_count=int(rows) if isinstance(rows, (int, float)) and rows > 0 else None,
                         comment=str(entry.get("comment") or ""), source_key=file_key,
                         masked={str(c) for c in (entry.get("masked_fields") or [])},
                         mask_all=str(entry.get("mask_mode") or "").lower() == "all")
        tables[table.key] = table
        by_schema_and_name[(ids.norm(owner), ids.norm(name))] = table.key
        by_schema_and_name.setdefault(("", ids.norm(name)), table.key)

    fks: list[InvForeignKey] = []
    for fk in schema.get("__db_fk_constraints__") or []:
        if not isinstance(fk, dict):
            continue
        parent = by_schema_and_name.get((ids.norm(fk.get("parent_schema")), ids.norm(fk.get("parent_table")))) \
            or by_schema_and_name.get(("", ids.norm(fk.get("parent_table"))))
        ref = by_schema_and_name.get((ids.norm(fk.get("ref_schema")), ids.norm(fk.get("ref_table")))) \
            or by_schema_and_name.get(("", ids.norm(fk.get("ref_table"))))
        if parent and ref and fk.get("parent_col") and fk.get("ref_col") \
                and tables[parent].column(str(fk["parent_col"])) and tables[ref].column(str(fk["ref_col"])):
            fks.append(InvForeignKey(parent, [str(fk["parent_col"])], ref, [str(fk["ref_col"])],
                                     str(fk.get("constraint_name") or ""), bool(fk.get("enforced"))))
    return Inventory(tables=tables, foreign_keys=fks, notes=notes)


def _declared_key(table: str, listed: list[Any], columns: list[InvColumn], notes: list[str]) -> list[str]:
    """The primary key discovery recorded, as this table's own columns, each once.

    Discovery reads keys by table name across schemas: a table of the same name in
    another schema adds its key columns to this one's, so a key can name a column
    twice (CUS_DMS_KEY, CUS_DMS_KEY) or one this table does not have.
    """
    key: list[str] = []
    for name in listed:
        column = next((c.name for c in columns if c.name.casefold() == str(name).casefold()), None)
        if column is None:
            notes.append(f"The key recorded for {table} names {name}, which {table} does not have "
                         "(a table of the same name in another schema?): the key is found from the data instead.")
            return []
        if column not in key:
            key.append(column)
    if len(key) < len(listed):
        notes.append(f"The key recorded for {table} lists {', '.join(key)} more than once: each column is used once.")
    return key
