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
