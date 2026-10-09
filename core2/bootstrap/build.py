"""Bootstrap, end to end: from a connected database to a semantic model version.

Each step reads only what earlier steps found, and every belief it records
carries its evidence. Business names here are the fallback readings of names
(``CUST_ORD_DT_KEY`` -> "Customer order date key"); the AI labelling step and
admin overrides improve them without changing anything the data decided.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from core2 import ids
from core2.bootstrap import names
from core2.bootstrap.calendar import CalendarFinding, find_calendars
from core2.bootstrap.dates import AUDIT_WORDS, DateCandidate, close_call, find_date_roles
from core2.bootstrap.inventory import InvColumn, Inventory, InvTable
from core2.bootstrap.joins import JoinFinding, discover_joins
from core2.bootstrap.journal import Journal, Watched, attempt, journal_of
from core2.bootstrap.labels import label
from core2.bootstrap.keys import TableKeys, infer_keys
from core2.bootstrap.measures import MeasureFinding, classify_tables, find_measures
from core2.bootstrap.profiler import ProfileOptions, TableProfile, profile_table
from core2.bootstrap.quality import find_quality
from core2.model.schema import (
    AggExpr,
    Attribute,
    Calendar,
    Column,
    DateRole,
    Entity,
    Evidence,
    Join,
    Measure,
    ReviewItem,
    SemanticModel,
    Table,
)
from core2.warehouse.runner import Warehouse


@dataclass
class BuildOptions:
    profile: ProfileOptions = field(default_factory=ProfileOptions)
    max_join_tests: int = 400
    workers: int = 4
    outliers: bool = True
    today: Callable[[], dt.datetime] = field(default=lambda: dt.datetime.now(dt.timezone.utc))
    labeler: Callable[[str, str], str] | None = None   # the AI that names things; None keeps names' readings
    progress: Callable[[str], None] | None = None      # each step as it happens, for the learned page


@dataclass
class Findings:
    """Everything the steps found, kept for the learned-data page and for tests."""

    inventory: Inventory
    profiles: dict[str, TableProfile]
    keys: dict[str, TableKeys]
    calendars: dict[str, CalendarFinding]
    joins: list[JoinFinding]
    dates: dict[str, list[DateCandidate]]
    kinds: dict[str, str]
    measures: list[MeasureFinding]
    # May a column's member values be read and shown (governance), whatever their number.
    values_allowed: Callable[[str, str], bool] = field(default=lambda table_key, column: True)


_PERIOD_NAMES = {"period", "month", "year", "week", "quarter", "fiscal period", "fiscal month"}

def _watched(warehouse: Warehouse, options: BuildOptions) -> Watched:
    return warehouse if isinstance(warehouse, Watched) else Watched(warehouse, Journal(options.progress))


def learn(warehouse: Warehouse, inventory: Inventory, options: BuildOptions | None = None) -> Findings:
    options = options or BuildOptions()
    warehouse = _watched(warehouse, options)
    journal = journal_of(warehouse)
    tables = list(inventory.tables.items())
    journal.step(f"Reading {len(tables)} tables ({sum(len(t.columns) for _, t in tables):,} columns): "
                 "counts, ranges and short code values, never rows")
    done = iter(range(1, len(tables) + 1))

    def read(item: tuple[str, InvTable]) -> TableProfile | None:
        _key, table = item
        profile = attempt(warehouse, f"the table {table.name}",
                          lambda: profile_table(warehouse, table, options.profile), None)
        if profile is not None:
            journal.step(f"Read {table.name} ({next(done)} of {len(tables)}): {profile.rows:,} rows"
                         + (", from a sample" if profile.sampled else ""))
        return profile

    with ThreadPoolExecutor(max_workers=max(1, options.workers)) as pool:
        profiled = list(pool.map(read, tables))
    profiles = {key: p for (key, _), p in zip(tables, profiled) if p is not None}
    unread = {key for (key, _), p in zip(tables, profiled) if p is None}
    if len(unread) == len(tables) and tables:
        raise RuntimeError("The database refused every table: " + "; ".join(
            f"{what} ({why})" for what, why in journal.left_out[:3]))
    if unread:
        # A table the database will not read is left out of the model, with every key that points at it.
        inventory = Inventory(tables={k: t for k, t in inventory.tables.items() if k not in unread},
                              foreign_keys=[fk for fk in inventory.foreign_keys
                                            if fk.table not in unread and fk.ref_table not in unread])
        tables = list(inventory.tables.items())
    for key, table in tables:
        for column in table.columns:
            # A generic NUMBER (no scale declared) holding only whole numbers is a whole-number column.
            if column.data_type == "decimal" and "(" not in column.raw_type \
                    and profiles[key].columns[column.name].integer_share == 1.0:
                column.data_type = "integer"
    journal.step("Finding each table's key")
    keys = {key: infer_keys(warehouse, table, profiles[key]) for key, table in tables}
    journal.step("Looking for calendar and period tables")
    calendars = find_calendars(warehouse, inventory, profiles, keys)
    joins = discover_joins(warehouse, inventory, profiles, keys, calendars, max_tests=options.max_join_tests,
                           workers=options.workers)
    journal.step("Finding the dates each table is about")
    dates = find_date_roles(warehouse, inventory, profiles, keys, calendars, joins)
    journal.step("Sorting tables into events, snapshots and lists; finding the measures")
    classified = classify_tables(inventory, profiles, keys, calendars, joins, dates)
    kinds = {key: kind for key, (kind, _) in classified.items()}
    measures = find_measures(inventory, profiles, keys, joins, dates, classified)
    for key, kind in list(kinds.items()):
        # A periodic table whose numbers are all amounts for the period (targets,
        # budgets) adds up over time like any fact.
        if kind == "snapshot" and not any(m.table == key and m.additivity == "semi_additive" for m in measures):
            kinds[key] = "fact"
    return Findings(inventory, profiles, keys, calendars, joins, dates, kinds, measures,
                    values_allowed=options.profile.values_allowed)


def build_model(warehouse: Warehouse, inventory: Inventory, *, client_id: str = "", db_id: int | None = None,
                db_type: str | None = None, options: BuildOptions | None = None,
                findings: Findings | None = None) -> SemanticModel:
    options = options or BuildOptions()
    warehouse = _watched(warehouse, options)
    journal = journal_of(warehouse)
    for note in inventory.notes:
        journal.step(note)
    f = findings or learn(warehouse, inventory, options)
    journal.step("Checking the data for things worth a look")
    flags = find_quality(warehouse, f.inventory, f.profiles, f.calendars, f.joins, f.dates, f.measures, f.kinds,
                         outliers=options.outliers)
    model = assemble(f, flags=flags, client_id=client_id, db_id=db_id, db_type=db_type or warehouse.db_type,
                     built_at=options.today())
    model.notes += inventory.notes
    for what, why in journal.left_out:
        model.notes.append(f"{what[:1].upper()}{what[1:]} was left out: the database refused it ({why}).")
    if options.labeler is not None:
        journal.step("Naming tables, columns and measures with the AI")
        model.notes += label(model, options.labeler)
    assign_slugs(model)
    return model


def assign_slugs(model: SemanticModel) -> None:
    """Planner-facing names from the final business names, unique within the model.

    Measures and entities come first (they are what questions name most), then
    dates, attributes and tables; a clash is resolved by prefixing the table.
    """
    taken: set[str] = set()
    for m in sorted(model.measures.values(), key=lambda m: (m.kind == "count", m.key)):
        base = ids.slug(m.business_name)
        m.slug = ids.unique_slug(base if base not in taken else
                                 ids.slug(f"{model.tables[m.table].business_name} {m.business_name}"), taken)
    entities = {}
    renamed: dict[str, str] = {}
    for old_slug, e in sorted(model.entities.items()):
        new_slug = ids.unique_slug(ids.slug(e.business_name), taken)
        renamed[old_slug] = new_slug
        e.slug = new_slug
        entities[new_slug] = e
    for e in entities.values():
        if e.parent:
            e.parent = renamed.get(e.parent, e.parent)
    model.entities = entities
    for r in sorted(model.date_roles.values(), key=lambda r: (not r.is_default, r.key)):
        base = ids.slug(r.name)
        r.slug = ids.unique_slug(base if base not in taken else
                                 ids.slug(f"{model.tables[r.table].business_name} {r.name}"), taken)
    for t in sorted(model.tables.values(), key=lambda t: t.key):
        t.slug = ids.unique_slug(ids.slug(t.business_name), taken)
    owner_slug = {e.table: e.slug for e in model.entities.values()}
    attributes: dict[str, Attribute] = {}
    for a in sorted(model.attributes.values(), key=lambda a: a.slug):
        column = model.columns[a.column]
        owner = owner_slug.get(column.table) or model.tables[column.table].slug
        name = _attribute_name(column.business_name, model.entities.get(owner))
        slug = ids.unique_slug(f"{owner}.{ids.slug(name)}", set(attributes))
        a.slug, a.entity = slug, owner if column.table in owner_slug else None
        a.business_name = column.business_name
        attributes[slug] = a
    model.attributes = attributes


def assemble(f: Findings, *, flags: list, client_id: str, db_id: int | None, db_type: str,
             built_at: dt.datetime) -> SemanticModel:
    model = SemanticModel(client_id=client_id, db_id=db_id, db_type=db_type, built_at=built_at)
    inv = f.inventory
    ck = {(key, c.name): ids.column_key(key, c.name) for key, t in inv.tables.items() for c in t.columns}
    taken: set[str] = set()

    # Joins first: column roles and entities depend on them.
    join_key_of: dict[tuple[str, str], str] = {}
    for j in f.joins:
        key = ids.join_key(j.from_table, j.from_columns, j.to_table, [t for _, t in j.pairs])
        if not j.also:     # one column's own link (its date's calendar, its review question)
            join_key_of[(j.from_table, j.from_column)] = key
        model.joins[key] = Join(
            key=key, from_table=j.from_table, from_columns=[ck[(j.from_table, c)] for c in j.from_columns],
            to_table=j.to_table, to_columns=[ck[(j.to_table, t)] for _, t in j.pairs],
            cardinality="one_to_one" if j.from_unique else "many_to_one", match_rate=j.match_rate,
            null_rate=j.null_rate, orphan_rows=j.unmatched, to_unique=True, role=j.role,
            trust="verified" if j.trust == "verified" else ("declared" if j.trust == "declared" else "proposed"),
            to_calendar=j.to_calendar, evidence=j.evidence, confidence=j.match_rate,
            provenance="declared" if j.declared else "profile",
            status="verified" if j.trust in ("verified", "declared") else "proposed")

    # Calendars.
    for key, cal in f.calendars.items():
        model.calendars[key] = Calendar(
            table=key, key_column=ck[(key, cal.key_column)] if cal.key_column else None,
            date_column=ck[(key, cal.date_column)] if cal.date_column else None,
            grain=cal.grain,  # type: ignore[arg-type]
            year_rows=cal.year_rows,
            attributes={a: ck[(key, c)] for a, c in cal.attributes.items()},
            first_date=cal.first, last_date=cal.last, contiguous=cal.contiguous,
            fiscal_year_start_month=cal.fiscal_year_start_month,
            fiscal_year_named_by=cal.fiscal_year_named_by,  # type: ignore[arg-type]
            placeholders=list(cal.placeholders), evidence=cal.evidence)
        if cal.fiscal_year_start_month and not model.settings.fiscal_year_start_month:
            model.settings.fiscal_year_start_month = cal.fiscal_year_start_month

    # Date roles.
    date_slugs: set[str] = set()
    for key, roles in f.dates.items():
        table_name = inv.tables[key].name
        for c in roles:
            if c.via_calendar and c.via_calendar.role:
                name = c.via_calendar.role
            elif c.via_calendar and not names.opaque(c.column):
                # A key into a calendar is named by what it keys: PRD_DMS_KEY is the period.
                text = names.read_tokens(names.core_column(c.column))
                name = text[:1].upper() + text[1:]
            elif not names.opaque(c.column):
                text = names.read_tokens(names.core_column(c.column))
                name = text[:1].upper() + text[1:]
            else:
                name = names.readable(c.column)
            if c.via_calendar is None and not names.opaque(c.column) and "date" not in name.lower() \
                    and c.granularity != "timestamp" and name.lower() not in _PERIOD_NAMES:
                name = f"{name} date"
            base = ids.slug(name)
            slug = ids.unique_slug(base if base not in date_slugs else ids.slug(f"{table_name} {name}"), date_slugs)
            key_c = ck[(key, c.column)]
            model.date_roles[key_c] = DateRole(
                key=key_c, table=key, column=key_c, name=name, slug=slug,
                calendar=c.via_calendar.to_table if c.via_calendar else None,
                calendar_join=join_key_of.get((key, c.column)) if c.via_calendar else None,
                granularity=c.granularity,  # type: ignore[arg-type]
                kind=c.kind,  # type: ignore[arg-type]
                is_default=c.is_default, score=round(c.score, 3), coverage=round(c.coverage, 4),
                placeholder_share=round(c.placeholder_share, 4), whole_year_share=round(c.whole_year_share, 4),
                first=c.first, last=c.last,
                evidence=c.evidence, confidence=max(0.0, min(1.0, c.score)), provenance="profile",
                status="verified" if c.is_default and not close_call(roles) else "proposed")
            taken.add(slug)
        runner = close_call(roles)
        default = next((c for c in roles if c.is_default), None)
        if runner and default:
            model.review.append(ReviewItem(
                key=f"default_date:{key}", object=key,
                question=f"Which date should {table_name} use when a question names none?",
                choice_made=names.readable(default.column), alternatives=[names.readable(runner.column)],
                evidence=default.evidence + runner.evidence))

    # Tables and columns.
    measure_columns = {(m.table, m.column) for m in f.measures if m.column}
    fk_columns = {(j.from_table, c): j for j in f.joins for c in j.from_columns}
    status_columns = {fl.object for fl in flags if fl.kind == "status_column"}
    for key, table in inv.tables.items():
        kind = f.kinds[key]
        tkeys = f.keys[key]
        business = _table_name(table.name)
        default_role = next((ck[(key, c.column)] for c in f.dates.get(key, []) if c.is_default), None)
        model.tables[key] = Table(
            key=key, database=table.database, schema_name=table.schema, name=table.name, business_name=business,
            kind=kind,  # type: ignore[arg-type]
            grain=[ck[(key, c)] for c in tkeys.primary_key], grain_text=_grain_text(table.name, kind, tkeys),
            row_count=f.profiles[key].rows, primary_key=[ck[(key, c)] for c in tkeys.primary_key],
            default_date=default_role, columns=[ck[(key, c.name)] for c in table.columns],
            slug=ids.unique_slug(ids.slug(business), taken), evidence=tkeys.evidence, confidence=1.0,
            provenance="profile", status="verified")
        by_column = {c.column: c for c in f.dates.get(key, [])}
        for column in table.columns:
            p = f.profiles[key].columns[column.name]
            k = ck[(key, column.name)]
            role = "attribute"
            fmt: str | None = None
            if column.name in tkeys.primary_key:
                role = "key"
            elif (key, column.name) in fk_columns:
                role = "date_key" if fk_columns[(key, column.name)].to_calendar else "foreign_key"
            elif column.name in by_column:
                c = by_column[column.name]
                role = "audit" if c.kind == "audit" else ("period_key" if c.granularity == "month" else "date")
            elif (key, column.name) in measure_columns:
                role = "measure"
            elif k in status_columns:
                role = "status"
            elif p.pattern in ("flag01", "flag", "flag_yn"):
                role = "flag"
            elif p.pattern == "free_text":
                role = "text"
            if column.data_type in ("date",):
                fmt = "date"
            elif column.data_type == "timestamp":
                fmt = "datetime"
            elif column.data_type == "text":
                fmt = "text"
            model.columns[k] = Column(
                key=k, table=key, name=column.name, data_type=column.data_type,  # type: ignore[arg-type]
                raw_type=column.raw_type, nullable=column.nullable, comment=column.comment,
                role=role,  # type: ignore[arg-type]
                business_name=names.readable(column.name, column.data_type), format=fmt,  # type: ignore[arg-type]
                values_allowed=f.values_allowed(key, column.name),
                profile=p, provenance="profile", status="verified", confidence=1.0)

    # Measures.
    for m in f.measures:
        table_name = inv.tables[m.table].name
        if m.column is None:
            key_m = f"m:{m.table}.rows"
            expr = AggExpr(agg="count")
        elif m.agg == "count_distinct":
            key_m = f"m:{m.table}.{m.column.casefold()}.distinct"
            expr = AggExpr(agg="count_distinct", column=ck[(m.table, m.column)])
        else:
            key_m = f"m:{m.table}.{m.column.casefold()}"
            expr = AggExpr(agg=m.agg, column=ck[(m.table, m.column)])  # type: ignore[arg-type]
        base = ids.slug(m.name)
        slug = ids.unique_slug(base if base not in taken else ids.slug(f"{_table_name(table_name)} {m.name}"), taken)
        unit = m.unit_values[0] if m.unit_column and len(m.unit_values) == 1 else None
        model.measures[key_m] = Measure(
            key=key_m, slug=slug, business_name=m.name, table=m.table, expr=expr,
            additivity=m.additivity,  # type: ignore[arg-type]
            time_aggregation=m.time_aggregation,  # type: ignore[arg-type]
            format=m.format,  # type: ignore[arg-type]
            unit=unit, unit_column=ck[m.unit_column] if m.unit_column else None,
            default_date=model.tables[m.table].default_date,
            kind="count" if m.agg in ("count", "count_distinct") else "column",
            evidence=m.evidence, provenance="profile", status="needs_review" if m.review else "verified",
            confidence=0.6 if m.review else 0.9)
        if m.review:
            model.review.append(ReviewItem(key=f"additivity:{key_m}", object=key_m, question=m.review,
                                           choice_made="taken at the end of each period",
                                           alternatives=["added up over time"], evidence=m.evidence))

    _entities_and_attributes(model, f, ck, taken)

    model.quality = flags
    for fl in flags:
        if fl.kind == "status_column":
            model.review.append(ReviewItem(key=f"default_filter:{fl.object}", object=fl.object,
                                           question=fl.message, choice_made="all rows are counted",
                                           alternatives=[f"leave out {v}" for v in fl.data.get("cancel_like", [])]))
    for j in f.joins:
        if j.trust == "proposed" and any(e.kind == "ambiguous" for e in j.evidence):
            model.review.append(ReviewItem(
                key=f"join:{j.from_table}.{j.from_column}", object=join_key_of[(j.from_table, j.from_column)],
                question=f"Which table does {inv.tables[j.from_table].name}.{j.from_column} point at?",
                choice_made="neither, until confirmed", alternatives=[inv.tables[j.to_table].name],
                evidence=j.evidence))
    return model


def _table_name(name: str) -> str:
    if names.opaque(name):
        return name
    # Read as a column's name is, each word with its neighbours: PRD_DMS is the period table,
    # ITM_BAL_PRD_FCT item balances by period, ABC_CLS_DMS the ABC classes.
    core = names.core_table(name)
    kept = [p for p in names.tokens(name) if p not in names.TABLE_AFFIXES and not p.isdigit()]
    if len(kept) == len(core):
        core = kept[:-1] + core[-1:]       # only the last word is made singular: sales order lines
    text = names.read_tokens(core)
    return text[:1].upper() + text[1:]


def _grain_text(name: str, kind: str, keys: TableKeys) -> str:
    noun = _table_name(name).lower()
    if kind == "snapshot" and len(keys.primary_key) > 1:
        return "one row per " + " and ".join(names.readable(c).lower() for c in keys.primary_key)
    if kind == "calendar":
        return "one row per day"
    return f"one row per {noun}"


def _entities_and_attributes(model: SemanticModel, f: Findings, ck: dict[tuple[str, str], str],
                             taken: set[str]) -> None:
    inv = f.inventory
    entity_of: dict[str, str] = {}
    for key, table in inv.tables.items():
        if f.kinds[key] != "dimension":
            continue
        tkeys = f.keys[key]
        profile = f.profiles[key]
        business = _table_name(table.name)
        slug = ids.unique_slug(ids.slug(business), taken)
        entity_of[key] = slug
        texts = [c for c in tkeys.unique_columns if table.type_of(c) == "text" and c not in tkeys.primary_key]
        label = _label_column(table, profile, tkeys)
        codes = [c for c in texts if c != label and (profile.columns[c].max_len or 99) <= 16]
        model.entities[slug] = Entity(
            slug=slug, business_name=business, table=key, key_columns=[ck[(key, c)] for c in tkeys.primary_key],
            label_column=ck[(key, label)] if label else None, code_column=ck[(key, codes[0])] if codes else None,
            members=profile.rows, provenance="profile", status="verified", confidence=0.9)
        if label:
            model.columns[ck[(key, label)]].role = "label"
            model.columns[ck[(key, label)]].label_of = model.tables[key].primary_key[0] if model.tables[key].primary_key else None
        if codes:
            model.columns[ck[(key, codes[0])]].role = "code"
        if len(tkeys.primary_key) == 1:
            # "How many customers do we have?": the members listed, each counted once.
            key_m = f"m:{key}.{tkeys.primary_key[0].casefold()}.distinct"
            counted = names.plural(business.lower())
            model.measures[key_m] = Measure(
                key=key_m, slug=ids.unique_slug(ids.slug(f"number of {counted}"), taken),
                business_name=f"Number of {counted}", table=key,
                expr=AggExpr(agg="count_distinct", column=ck[(key, tkeys.primary_key[0])]),
                additivity="non_additive", format="count", kind="count",
                default_date=model.tables[key].default_date,
                evidence=[Evidence(kind="entity_count", weight=1,
                                   detail=f"each {business.lower()} listed in {table.name}, counted once")],
                provenance="profile", status="verified", confidence=0.9)

    # Hierarchies: a dimension pointing at another dimension.
    for j in f.joins:
        if j.trust in ("verified", "declared") and j.from_table in entity_of and j.to_table in entity_of \
                and j.from_table != j.to_table:
            child = model.entities[entity_of[j.from_table]]
            if child.parent is None:
                child.parent = entity_of[j.to_table]

    # Attributes: what people group and filter by.
    for key, table in inv.tables.items():
        kind = f.kinds[key]
        if kind not in ("dimension", "fact", "snapshot", "bridge"):
            continue
        owner = entity_of.get(key) or model.tables[key].slug
        for column in table.columns:
            col = model.columns[ck[(key, column.name)]]
            p = col.profile
            if col.role in ("key", "foreign_key", "date_key", "date", "audit", "period_key", "measure", "text"):
                continue
            if not _attribute_worthy(column, col, kind):
                continue
            if kind != "dimension" and col.role not in ("status", "flag") and not (
                    p and 1 < p.distinct <= 1000 and column.data_type == "text"):
                continue
            name = _attribute_name(col.business_name, model.entities.get(owner))
            slug = f"{owner}.{ids.slug(name)}"
            if slug in model.attributes:
                slug = ids.unique_slug(slug, set(model.attributes))
            model.attributes[slug] = Attribute(
                slug=slug, column=col.key, entity=owner if key in entity_of else None,
                business_name=col.business_name, members=p.distinct if p else 0,
                provenance="profile", status="verified", confidence=0.8)


_NAME_WORDS = {"name", "nm", "title", "label"}
_DESCRIPTION_WORDS = {"desc", "dsc", "description", "descr"}


def _label_column(table: InvTable, profile: TableProfile, keys: TableKeys) -> str | None:
    """The column that names each member: a name, else a short description, else the longest unique text.

    Names need not be unique (two customers can both be "J. Smith"): a name-like
    column filled on nine rows in ten with nine distinct values in ten is a name.
    Grouping by it is made safe by also grouping by the member's code or key.
    """
    rows = profile.rows or 0
    best: list[tuple[int, float, str]] = []
    for column in table.columns:
        name = column.name
        p = profile.columns[name]
        if column.data_type != "text" or name in keys.primary_key or not rows:
            continue
        words = set(names.tokens(name))
        unique = name in keys.unique_columns
        named = bool(words & _NAME_WORDS)
        described = bool(words & _DESCRIPTION_WORDS) and (p.avg_len or 0) <= 60
        near_unique = p.non_null >= 0.9 * rows and p.distinct >= 0.9 * p.non_null
        if not (unique or (near_unique and (named or described))):
            continue
        rank = 0 if named else 1 if described else 2
        best.append((rank, -(p.avg_len or 0), name))
    return min(best)[2] if best else None


# Columns that describe the load, not the business: never offered as groupings.
_TECHNICAL = {"etl", "ext", "src", "sys", "dw", "dwh", "hash", "checksum", "rowid", "guid", "uuid", "az"}


def _attribute_worthy(column: InvColumn, col: Column, kind: str) -> bool:
    p = col.profile
    if p is None or not p.distinct:
        return False   # never filled
    words = set(names.tokens(column.name))
    if words & _TECHNICAL or words & AUDIT_WORDS:
        return False
    last = names.tokens(column.name)[-1:] or [""]
    if column.data_type in ("integer", "decimal", "float") and last[0] in names.KEY_SUFFIXES - {"code", "cd", "no"}:
        return False   # a key to a table this warehouse does not hold
    if kind == "dimension" and column.data_type in ("integer", "decimal", "float") and (
            p.distinct > 20 or p.distinct > 0.5 * p.non_null):
        return False   # a weight or a size, one per member: a number to read, not a group (pack sizes repeat)
    return True


def _attribute_name(business: str, entity: Entity | None) -> str:
    if entity is None:
        return business
    words = business.split()
    entity_words = entity.business_name.lower().split()
    if len(words) > 1 and words[0].lower() in entity_words:
        return " ".join(words[1:])
    return business
