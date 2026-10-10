"""The semantic model v2, as data.

Everything QueryBot believes about a connected database is one of these objects,
and every belief carries the evidence it rests on, how sure the builder is, where
it came from and whether anyone has confirmed it. Dictionaries are keyed by the
identity keys of :mod:`core2.ids`; nothing here is keyed by a display name.

The model is plain data: no method on it queries a warehouse or calls a model.
Builders (core2.bootstrap) write it, overrides (core2.model.overrides) amend it,
and the planner, resolver and compiler read it.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

Provenance = Literal["declared", "profile", "heuristic", "ai", "metric", "import", "admin"]
Status = Literal["verified", "proposed", "approved", "rejected", "needs_review"]

DataType = Literal["integer", "decimal", "float", "text", "date", "timestamp", "boolean", "other"]
ColumnRole = Literal[
    "measure",      # a number that is aggregated
    "attribute",    # a descriptive value to group or filter by
    "label",        # the name shown for an entity (one per key)
    "code",         # a short business code for an entity (one per key)
    "key",          # this table's own key, or part of it
    "foreign_key",  # points at another table's key
    "date",         # a date or timestamp that is a business date
    "date_key",     # an integer or text key into a calendar, or yyyymmdd
    "period_key",   # yyyymm / yyyy style period numbers
    "status",       # a low-cardinality state code (open, cancelled, ...)
    "flag",         # 0/1, Y/N, true/false
    "audit",        # when or by whom a row was written; never a business date
    "identifier",   # a business document number (invoice number); counted, not summed
    "text",         # free text
    "unknown",
]
Format = Literal["currency", "percent", "count", "number", "integer", "text", "date", "datetime"]
TableKind = Literal["fact", "snapshot", "dimension", "bridge", "calendar", "other"]
Cardinality = Literal["many_to_one", "one_to_one", "one_to_many", "many_to_many"]
JoinTrust = Literal["admin", "declared", "verified", "proposed", "rejected"]
DateKind = Literal["event", "snapshot", "planned", "due", "validity", "audit"]
Granularity = Literal["day", "month", "year", "timestamp"]
Additivity = Literal["additive", "semi_additive", "non_additive"]
TimeAggregation = Literal["last", "average", "first"]
FilterOp = Literal["in", "not_in", "eq", "ne", "gt", "gte", "lt", "lte", "between",
                   "contains", "starts_with", "is_null", "not_null"]
Scalar = Union[str, int, float, bool, None]


class _Data(BaseModel):
    # Stored versions are read back by newer code: unknown fields are ignored,
    # never an error.
    model_config = ConfigDict(extra="ignore")


class Evidence(_Data):
    """One fact a belief rests on, readable by an admin as it stands."""

    kind: str               # "declared_fk", "containment", "uniqueness", "load_clustering", ...
    detail: str             # "99.7% of 20,000 rows match; the target key is unique"
    weight: float = 0.0     # its contribution to the belief's score, where scored
    data: dict[str, Any] = Field(default_factory=dict)


class Belief(_Data):
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = 0.0
    provenance: Provenance = "profile"
    status: Status = "proposed"


class TopValue(_Data):
    value: str | None
    count: int


class ColumnProfile(_Data):
    """Aggregates computed inside the warehouse. No row ever leaves it."""

    rows: int = 0
    non_null: int = 0
    distinct: int = 0
    distinct_is_approx: bool = False
    min: str | None = None          # numbers and dates as text, ISO for dates
    max: str | None = None
    min_num: float | None = None
    max_num: float | None = None
    negatives: int | None = None
    zeros: int | None = None
    integer_share: float | None = None   # whole numbers among non-null numeric values
    min_len: float | None = None
    avg_len: float | None = None
    max_len: float | None = None
    top: list[TopValue] | None = None    # only for low-cardinality columns, only where allowed
    pattern: str | None = None           # yyyymmdd | yyyymm | yyyy | flag01 | flag_yn | code | name | free_text | ...
    time_share: float | None = None      # timestamps: share of values with a time of day
    date_min: str | None = None          # date-shaped numbers: the first and last real dates,
    date_max: str | None = None          # placeholders (-1, 0, 19000101) left out
    whole_year_rows: int | None = None   # yyyymm keys with month 00: a row for a whole year, not a month
    placeholder_rows: int | None = None  # date columns: 1900-01-01, 9999-12-31 and the like, not dates
    sampled: bool = False

    @property
    def null_share(self) -> float:
        return 0.0 if not self.rows else 1.0 - self.non_null / self.rows

    @property
    def distinct_ratio(self) -> float:
        return 0.0 if not self.non_null else self.distinct / self.non_null


class Column(Belief):
    key: str
    table: str
    name: str                       # as the warehouse spells it
    data_type: DataType = "other"
    raw_type: str = ""
    nullable: bool = True
    comment: str = ""
    role: ColumnRole = "unknown"
    business_name: str = ""
    description: str = ""
    synonyms: dict[str, list[str]] = Field(default_factory=dict)   # language -> words
    unit: str | None = None
    format: Format | None = None
    label_of: str | None = None     # the key column this column names
    sensitivity: Literal["none", "pii", "confidential"] = "none"
    # Learn's reading of people's data (core2/bootstrap/personal.py): a person's name, or a personal detail
    # (contact, birth date, national ID). Never sent to the AI; where the workspace is under compliance,
    # shown masked to a reader who has not signed the confidentiality attestation.
    personal: Literal["none", "name", "detail"] = "none"
    values_allowed: bool = False    # may common values be shown to the AI and in the catalog
    profile: ColumnProfile | None = None
    hidden: bool = False            # an admin hid it from questions
    value_names: dict[str, str] = Field(default_factory=dict)   # stored value -> what readers see ("C" -> "Cancelled")
    # A name the row holds in parts ("first name" and "last name"): column keys of the same table, read joined by
    # a space. Such a column is not in the warehouse; it is never in its table's column list.
    parts: list[str] = Field(default_factory=list)


_FLAG_TAIL = (" flag", " indicator", " ind", " yn")
_FLAG_LEADS = (("is ", "", "Not "), ("was ", "", "Not "), ("are ", "", "Not "), ("has ", "With ", "Without "),
               ("have ", "With ", "Without "), ("can ", "Can ", "Cannot "))


def value_names(column: Column) -> dict[str, str]:
    """What readers see for a column's stored values: the names an admin gave its codes ("C" -> "Cancelled"), or
    for a yes/no flag Learn found (1 and 0), words made of its own name: "Is sterile" -> Sterile and Not sterile,
    "Cancelled flag" -> Cancelled and Not cancelled, "Has sterile cleanroom" -> With and Without sterile cleanroom.
    Never 0 and 1 in an answer."""
    if column.value_names or column.role != "flag" or column.data_type not in ("integer", "boolean"):
        return column.value_names
    seen = {str(t.value).lower() for t in (column.profile.top if column.profile else []) if t.value is not None}
    if not seen <= {"0", "1", "true", "false"}:
        return {}          # a "flag" holding other values: shown as stored, never some of them as no value
    words = (column.business_name or column.name).strip()
    for tail in _FLAG_TAIL:
        if words.lower().endswith(tail) and len(words) > len(tail):
            words = words[:-len(tail)].strip()
    yes, no = "", "Not "
    for lead, with_, without in _FLAG_LEADS:
        if words.lower().startswith(lead) and len(words) > len(lead):
            words, yes, no = words[len(lead):].strip(), with_, without
            break
    if not words:
        return {}
    first = words.split()[0]
    lower = words if first.isupper() and len(first) > 1 else words[:1].lower() + words[1:]
    return {"1": f"{yes}{lower}" if yes else words[:1].upper() + words[1:], "0": f"{no}{lower}"}


class ColumnFilter(_Data):
    """A row filter on one column of the model (keys, not slugs)."""

    column: str
    op: FilterOp
    values: list[Scalar] = Field(default_factory=list)
    shown: list[str] = Field(default_factory=list)      # what readers see for the values (status 8: "Cancelled")


class Table(Belief):
    key: str
    database: str = ""
    schema_name: str = ""
    name: str = ""
    business_name: str = ""
    description: str = ""
    kind: TableKind = "other"
    grain: list[str] = Field(default_factory=list)      # column keys whose combination is unique
    grain_text: str = ""                                # "one row per invoice line"
    row_count: int = 0
    primary_key: list[str] = Field(default_factory=list)
    default_date: str | None = None                     # date role key
    default_filters: list[ColumnFilter] = Field(default_factory=list)
    readers_may_include: bool = True    # a question may ask for the rows default_filters leave out
    columns: list[str] = Field(default_factory=list)    # column keys, in warehouse order
    slug: str = ""
    profiled_at: datetime | None = None
    hidden: bool = False


class Join(Belief):
    key: str
    from_table: str
    from_columns: list[str]         # column keys
    to_table: str
    to_columns: list[str]
    cardinality: Cardinality = "many_to_one"
    match_rate: float = 0.0         # non-null, non-placeholder from-rows that find a match
    null_rate: float = 0.0
    orphan_rows: int = 0
    to_unique: bool = False
    max_fanout: float = 1.0
    # to_unique and max_fanout were measured on the data (by Learn, or the admin's link check). A link no one
    # measured -- declared by the database or brought over from today's setup, in a model learned before
    # the measure was taken -- is checked the first time a question follows it (core2/resolve/paths.unchecked).
    target_checked: bool = False
    role: str | None = None         # business name when the same tables join more than one way
    trust: JoinTrust = "proposed"
    to_calendar: bool = False       # the target is a calendar: used only as a date role
    # Rows of the target the link keeps, an admin's: "only the customer's current row".
    # Written into the join itself, so a row with no kept match shows as Unknown, never dropped.
    conditions: list[ColumnFilter] = Field(default_factory=list)
    # A row with no match is kept (as Unknown); an admin may keep matched rows only.
    keep_unmatched: bool = True


class Calendar(_Data):
    table: str
    key_column: str | None          # column key; None when the date itself is the key
    date_column: str | None         # None for a period table (one row per month, keyed yyyymm)
    grain: Literal["day", "month"] = "day"
    year_rows: bool = False         # a period table also holds whole-year rows (month 00)
    attributes: dict[str, str] = Field(default_factory=dict)   # canonical attribute -> column key
    first_date: date | None = None
    last_date: date | None = None
    contiguous: bool = False
    week_start: Literal["monday", "sunday", "saturday"] | None = None
    fiscal_year_start_month: int | None = None
    fiscal_year_named_by: Literal["start", "end"] | None = None
    placeholders: list[Scalar] = Field(default_factory=list)   # key values that mean "no date"
    evidence: list[Evidence] = Field(default_factory=list)


class DateRole(Belief):
    key: str
    table: str
    column: str                     # column key on `table`
    name: str = ""                  # "Invoice date"
    slug: str = ""
    calendar: str | None = None     # calendar table key when the column is a key into it
    calendar_join: str | None = None
    granularity: Granularity = "day"
    kind: DateKind = "event"
    is_default: bool = False
    score: float = 0.0
    coverage: float = 0.0
    placeholder_share: float = 0.0
    whole_year_share: float = 0.0   # rows keyed to month 00 (a whole year): never counted as a month
    first: date | None = None
    last: date | None = None
    synonyms: dict[str, list[str]] = Field(default_factory=dict)


class AggExpr(_Data):
    agg: Literal["sum", "count", "count_distinct", "avg", "min", "max"]
    column: str | None = None       # None with count = count rows
    filters: list[ColumnFilter] = Field(default_factory=list)
    table: str | None = None        # whose rows a count with no column counts; None = the measure's own table


class OpExpr(_Data):
    op: Literal["ratio", "subtract", "add", "multiply"]
    args: list[MeasureExpr]
    scale: float = 1.0              # 100 for a percentage


class RefExpr(_Data):
    measure: str                    # another measure's key


class SqlExpr(_Data):
    """An imported definition with no structural form: parsed and checked, never pasted."""

    sql: str
    columns: list[str] = Field(default_factory=list)
    filters: list[ColumnFilter] = Field(default_factory=list)   # rows each of its aggregates counts


MeasureExpr = Annotated[Union[AggExpr, OpExpr, RefExpr, SqlExpr], Field(union_mode="left_to_right")]
OpExpr.model_rebuild()


class Measure(Belief):
    key: str
    slug: str = ""
    business_name: str = ""
    description: str = ""
    synonyms: dict[str, list[str]] = Field(default_factory=dict)
    table: str = ""                 # base table key
    expr: MeasureExpr
    additivity: Additivity = "additive"
    time_aggregation: TimeAggregation | None = None    # for semi-additive measures
    format: Format = "number"
    unit: str | None = None
    unit_column: str | None = None
    default_date: str | None = None
    filters: list[ColumnFilter] = Field(default_factory=list)
    kind: Literal["metric", "import", "model", "proposed", "column", "count"] = "column"
    tested: bool = False
    hidden: bool = False


class Entity(Belief):
    """Something the business counts and groups by: a customer, an item, a warehouse."""

    slug: str
    business_name: str = ""
    synonyms: dict[str, list[str]] = Field(default_factory=dict)
    table: str
    key_columns: list[str] = Field(default_factory=list)
    label_column: str | None = None
    code_column: str | None = None
    members: int = 0
    parent: str | None = None       # entity slug one level up a hierarchy
    role: str | None = None         # for role-playing variants: "Bill-to"


class Attribute(Belief):
    """A column people group and filter by, reached from facts through joins."""

    slug: str                       # "customer.city", "invoice.status"
    column: str                     # column key
    entity: str | None = None       # entity slug, None for an attribute of a fact itself
    business_name: str = ""
    synonyms: dict[str, list[str]] = Field(default_factory=dict)
    members: int = 0
    indexed: bool = False           # member values are in the lookup index
    # How a question reads it: "group" (a category to group and filter by), "number" (an amount each member
    # or row has: a price, a weight, days in transit; compared, sorted and shown), "identifier" (names or
    # numbers each member or row by itself: an invoice number, an NPI; looked up, listed and ranked by,
    # shown as written) or "text" (free text: searched, never grouped).
    kind: Literal["group", "number", "identifier", "text"] = "group"


class QualityFlag(_Data):
    key: str
    object: str                     # table, column, measure or join key
    kind: Literal["outlier_period", "negative_values", "constant", "low_match_rate", "status_column",
                  "placeholder_dates", "listed_vs_active", "load_timestamp", "unit_mix", "same_values", "other"]
    message: str
    severity: Literal["info", "warning", "serious"] = "info"
    data: dict[str, Any] = Field(default_factory=dict)


class ReviewItem(_Data):
    """A call the builder made on thin evidence, offered to an admin (never waited on)."""

    key: str
    object: str
    question: str                   # "Which date should invoice lines use by default?"
    choice_made: str
    alternatives: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)


class ModelSettings(_Data):
    week_start: Literal["monday"] = "monday"
    fiscal_year_start_month: int | None = Field(default=None, ge=1, le=12)
    currency: str | None = None
    languages: list[str] = Field(default_factory=lambda: ["en"])


class SemanticModel(_Data):
    client_id: str = ""
    db_id: int | None = None
    db_type: str = ""               # snowflake | azure_sql | oracle | duckdb
    version: int = 0
    built_at: datetime | None = None
    source_hash: str = ""
    partial: bool = False           # profiling stopped at its budget
    settings: ModelSettings = Field(default_factory=ModelSettings)
    tables: dict[str, Table] = Field(default_factory=dict)
    columns: dict[str, Column] = Field(default_factory=dict)
    joins: dict[str, Join] = Field(default_factory=dict)
    calendars: dict[str, Calendar] = Field(default_factory=dict)   # by table key
    date_roles: dict[str, DateRole] = Field(default_factory=dict)
    measures: dict[str, Measure] = Field(default_factory=dict)
    entities: dict[str, Entity] = Field(default_factory=dict)      # by slug
    attributes: dict[str, Attribute] = Field(default_factory=dict) # by slug
    quality: list[QualityFlag] = Field(default_factory=list)
    review: list[ReviewItem] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def table_columns(self, table_key: str) -> list[Column]:
        table = self.tables.get(table_key)
        return [self.columns[c] for c in (table.columns if table else []) if c in self.columns]

    def joins_from(self, table_key: str) -> list[Join]:
        return sorted((j for j in self.joins.values() if j.from_table == table_key), key=lambda j: j.key)

    def date_roles_of(self, table_key: str) -> list[DateRole]:
        return sorted((r for r in self.date_roles.values() if r.table == table_key), key=lambda r: r.key)
