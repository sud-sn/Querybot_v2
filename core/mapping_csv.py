"""The source mapping document: joins and column terms as a spreadsheet.

Two things a data team already has on paper and could not get into the
product. Joins were editable one at a time on a canvas, or in bulk through a
JSON endpoint driven by a UI. Column terms were a free-text box an admin had
to think of unprompted. Neither could be handed over as a file, and the one
import that did exist was a whole-graph JSON REPLACE keyed on internal ids --
a clone tool, not a mapping document.

Three properties decide whether this is usable:

**Tables, not entity names.** A mapping document from a data team is written
in warehouse terms: ``SALES.F_ORDER.CUSTOMER_ID``. The graph calls that table
"Orders" because somebody typed a name on a canvas. Making the file speak the
warehouse's language and resolving to entities on the way in is the whole
difference between a file somebody can produce and a file only this product
could have written.

**Additive, and a dry run first.** Every row is parsed, resolved and checked
before anything is written, and the caller decides whether to apply. An import
that half-succeeds leaves a graph nobody can reason about; one that refuses
everything because row 14 is wrong makes the admin find row 14 themselves.

**Round trip.** The same shape comes back out. "Editing the joins" in
practice is download, edit in a spreadsheet, upload -- and a format you can
only write is a format nobody maintains.

Nothing here touches the store or the database. It parses, it renders, and it
reports; the caller resolves entities and applies.
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass, field

log = logging.getLogger("querybot.mapping_csv")

# Rows above this are refused as a whole rather than half-applied. A mapping
# document is written by a person; a hundred thousand rows is a mistake, and
# finding out halfway through is worse than being told up front.
MAX_ROWS = 5000

JOIN_COLUMNS = ("from_table", "from_column", "to_table", "to_column",
                "relationship", "join_type", "label", "group")
TERM_COLUMNS = ("table", "column", "terms")

_RELATIONSHIPS = {"many_to_one", "one_to_one", "many_to_many"}
_JOIN_TYPES = {"INNER", "LEFT", "RIGHT", "FULL"}


@dataclass
class RowProblem:
    line: int
    reason: str
    detail: str = ""

    def as_dict(self) -> dict:
        return {"line": self.line, "reason": self.reason, "detail": self.detail}


@dataclass
class ParsedJoin:
    """One join as the file states it, before any entity is resolved."""

    line: int
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    relationship: str = "many_to_one"
    join_type: str = "LEFT"
    label: str = ""
    group: str = ""

    @property
    def key(self) -> tuple:
        """What makes two rows the same join.

        The GROUP is part of it: two rows sharing a group are two column pairs
        of ONE composite join, not two joins.
        """
        return (self.from_table.upper(), self.to_table.upper(),
                self.group.strip().upper())

    def as_dict(self) -> dict:
        return {
            "line": self.line, "from_table": self.from_table,
            "from_column": self.from_column, "to_table": self.to_table,
            "to_column": self.to_column, "relationship": self.relationship,
            "join_type": self.join_type, "label": self.label,
            "group": self.group,
        }


@dataclass
class ParsedTerms:
    line: int
    table: str
    column: str
    terms: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {"line": self.line, "table": self.table, "column": self.column,
                "terms": list(self.terms)}


@dataclass
class ParseResult:
    rows: list = field(default_factory=list)
    problems: list[RowProblem] = field(default_factory=list)
    fatal: str = ""

    @property
    def ok(self) -> bool:
        return not self.fatal

    def as_dict(self) -> dict:
        return {
            "rows": [r.as_dict() for r in self.rows],
            "problems": [p.as_dict() for p in self.problems],
            "fatal": self.fatal,
        }


def _reader(text: str, required: tuple[str, ...]) -> tuple[csv.DictReader | None, str]:
    """A DictReader over `text`, or a reason it cannot be read.

    Tolerates a UTF-8 BOM (every spreadsheet writes one), header case and
    surrounding spaces. A file that has been through Excel and back should
    still load; a file missing a column the importer needs should not.
    """
    if not str(text or "").strip():
        return None, "The file is empty."
    stream = io.StringIO(str(text).lstrip("﻿"))
    try:
        reader = csv.DictReader(stream)
        header = [str(h or "").strip().lower() for h in (reader.fieldnames or [])]
    except csv.Error as exc:
        return None, f"The file could not be read as CSV: {exc}"
    if not header:
        return None, "The file has no header row."
    reader.fieldnames = header
    missing = [column for column in required if column not in header]
    if missing:
        return None, ("The header is missing: " + ", ".join(missing)
                      + ". Expected: " + ", ".join(required) + ".")
    return reader, ""


def _cell(row: dict, name: str) -> str:
    return " ".join(str(row.get(name) or "").split())


def parse_joins(text: str) -> ParseResult:
    """Read a join mapping file. Never raises."""
    required = ("from_table", "from_column", "to_table", "to_column")
    reader, fatal = _reader(text, required)
    if reader is None:
        return ParseResult(fatal=fatal)

    result = ParseResult()
    for offset, raw in enumerate(reader):
        line = offset + 2      # +1 for the header, +1 because people count from 1
        if len(result.rows) >= MAX_ROWS:
            return ParseResult(
                fatal=f"More than {MAX_ROWS} rows. A mapping document is "
                      f"written by a person; this looks like an export.")
        if not any(str(value or "").strip() for value in raw.values()):
            continue           # a blank line between blocks is not an error

        values = {name: _cell(raw, name) for name in JOIN_COLUMNS}
        missing = [name for name in required if not values[name]]
        if missing:
            result.problems.append(RowProblem(
                line, "missing_value",
                "These are required and were blank: " + ", ".join(missing)))
            continue

        relationship = (values["relationship"] or "many_to_one").lower()
        if relationship not in _RELATIONSHIPS:
            result.problems.append(RowProblem(
                line, "bad_relationship",
                f"'{values['relationship']}' is not one of: "
                + ", ".join(sorted(_RELATIONSHIPS))))
            continue

        join_type = (values["join_type"] or "LEFT").upper()
        if join_type not in _JOIN_TYPES:
            result.problems.append(RowProblem(
                line, "bad_join_type",
                f"'{values['join_type']}' is not one of: "
                + ", ".join(sorted(_JOIN_TYPES))))
            continue

        result.rows.append(ParsedJoin(
            line=line,
            from_table=values["from_table"], from_column=values["from_column"],
            to_table=values["to_table"], to_column=values["to_column"],
            relationship=relationship, join_type=join_type,
            label=values["label"], group=values["group"],
        ))
    return result


def group_composites(rows: list[ParsedJoin]) -> list[tuple[ParsedJoin, list[dict]]]:
    """Fold rows sharing a group into one join with several column pairs.

    A composite key cannot be one row of a flat file, so the file says so with
    a shared group name. Rows with no group are each their own join -- two
    blank groups are not "the same composite", which is the mistake a
    ``group or ''`` key would make.
    """
    ordered: list[tuple[ParsedJoin, list[dict]]] = []
    index: dict[tuple, int] = {}
    for row in rows:
        pair = {"from_col": row.from_column, "to_col": row.to_column}
        if not row.group.strip():
            ordered.append((row, [pair]))
            continue
        position = index.get(row.key)
        if position is None:
            index[row.key] = len(ordered)
            ordered.append((row, [pair]))
        else:
            ordered[position][1].append(pair)
    return ordered


def parse_terms(text: str) -> ParseResult:
    """Read a column-terms mapping file. Never raises.

    ``terms`` is a comma-separated list inside one cell, which is how a person
    writes it and how the product already stores it.
    """
    reader, fatal = _reader(text, TERM_COLUMNS)
    if reader is None:
        return ParseResult(fatal=fatal)

    result = ParseResult()
    for offset, raw in enumerate(reader):
        line = offset + 2
        if len(result.rows) >= MAX_ROWS:
            return ParseResult(
                fatal=f"More than {MAX_ROWS} rows. A mapping document is "
                      f"written by a person; this looks like an export.")
        if not any(str(value or "").strip() for value in raw.values()):
            continue

        table, column = _cell(raw, "table"), _cell(raw, "column")
        if not table or not column:
            result.problems.append(RowProblem(
                line, "missing_value", "table and column are both required."))
            continue

        terms: list[str] = []
        seen: set[str] = set()
        for piece in str(raw.get("terms") or "").split(","):
            term = " ".join(piece.split())
            key = term.casefold()
            if term and key not in seen:
                seen.add(key)
                terms.append(term)
        if not terms:
            # An empty terms cell is how somebody CLEARS a column's terms, and
            # that is a legitimate edit -- but it is worth reporting, because
            # it is also what a mis-split file looks like.
            result.problems.append(RowProblem(
                line, "no_terms",
                f"{table}.{column} has no terms. Applying this row would "
                f"clear any it already has."))
        result.rows.append(ParsedTerms(line=line, table=table, column=column,
                                       terms=tuple(terms)))
    return result


# ── Rendering, so the file can be edited and sent back ────────────────────────

def _write(header: tuple[str, ...], rows: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(header),
                            lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def render_joins(relationships: list[dict], table_for_entity) -> str:
    """The join map as the file the importer reads back.

    ``table_for_entity`` maps an entity name to its qualified table, because
    the file speaks the warehouse's language rather than the graph's.

    A composite join comes out as several rows sharing a group named after the
    join, so a round trip through a spreadsheet reproduces it rather than
    flattening it to its first pair.
    """
    out: list[dict] = []
    for relationship in relationships or []:
        from_table = table_for_entity(relationship.get("from_entity") or "")
        to_table = table_for_entity(relationship.get("to_entity") or "")
        if not from_table or not to_table:
            continue

        from core.join_governance import join_pairs

        pairs = join_pairs(dict(relationship))
        group = ""
        if len(pairs) > 1:
            group = (str(relationship.get("relationship_key") or "").strip()
                     or f"{from_table}-{to_table}-{relationship.get('id') or ''}")
        for from_column, to_column in pairs:
            out.append({
                "from_table": from_table, "from_column": from_column,
                "to_table": to_table, "to_column": to_column,
                "relationship": str(relationship.get("relationship_type")
                                    or "many_to_one"),
                "join_type": str(relationship.get("join_type") or "LEFT").upper(),
                "label": str(relationship.get("label") or ""),
                "group": group,
            })
    return _write(JOIN_COLUMNS, out)


def render_terms(descriptions: dict) -> str:
    """Column terms as the file the importer reads back.

    ``descriptions`` is what store.list_table_descriptions returns: a table
    name mapped to an entry carrying ``column_synonym_map``.
    """
    out: list[dict] = []
    for table, entry in sorted((descriptions or {}).items()):
        for column, terms in sorted(((entry or {}).get("column_synonym_map") or {}).items()):
            out.append({"table": table, "column": column,
                        "terms": ", ".join(terms or [])})
    return _write(TERM_COLUMNS, out)
