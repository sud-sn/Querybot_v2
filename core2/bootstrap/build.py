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
from core2.bootstrap import names, personal
from core2.bootstrap.arithmetic import check_arithmetic, find_counters, find_parent_figures
from core2.bootstrap.calendar import CalendarFinding, find_calendars
from core2.bootstrap.dates import AUDIT_WORDS, DateCandidate, close_call, find_date_roles
from core2.bootstrap.history import MIN_QUERIES, links_in, read_query_log
from core2.bootstrap.inventory import InvColumn, Inventory, InvTable
from core2.bootstrap.joins import JoinFinding, discover_joins
from core2.bootstrap.journal import Journal, Watched, attempt, journal_of
from core2.bootstrap.labels import label
from core2.bootstrap.keys import TableKeys, infer_keys, series_key
from core2.bootstrap.measures import MeasureFinding, classify_tables, find_measures
from core2.bootstrap.profiler import ProfileOptions, TableProfile, profile_table
from core2.bootstrap.quality import find_quality
from core2.model.schema import (
    AggExpr,
    Attribute,
    Calendar,
    Column,
    ColumnFilter,
    ColumnProfile,
    DateRole,
    Entity,
    Evidence,
    Join,
    Measure,
    QualityFlag,
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
    # The SELECT statements the warehouse keeps (core2/bootstrap/history.py); only the columns they join are read.
    query_history: Callable[[Warehouse], list[str]] | None = read_query_log


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
    # Pairs of measure columns holding the same values: (table, column, other column, share of rows).
    same_values: list[tuple[str, str, str, float]] = field(default_factory=list)


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
    texts = attempt(warehouse, "the warehouse's query history", lambda: options.query_history(warehouse), []) \
        if options.query_history else []
    seen = links_in(texts, inventory, keys, warehouse.dialect) if texts else {}
    if texts:
        journal.step(f"Read {len(texts):,} of the warehouse's own queries: {sum(n >= MIN_QUERIES for n in seen.values())}"
                     f" joins between tables written in {MIN_QUERIES} or more")
    joins = discover_joins(warehouse, inventory, profiles, keys, calendars, max_tests=options.max_join_tests,
                           workers=options.workers, seen=seen)
    journal.step("Finding the dates each table is about")
    dates = find_date_roles(warehouse, inventory, profiles, keys, calendars, joins)
    for key, candidates in dates.items():
        # Readings taken again and again (an interface's counters every 15 minutes): the column that, with the
        # time, identifies each one, whatever links say (with no name to match, they may be wrong). It counts
        # nothing, and the counters run within it.
        stamp = next((c.column for c in candidates if c.is_default and c.granularity == "timestamp"), None)
        if stamp:
            series = series_key(warehouse, inventory.tables[key], profiles[key], stamp)
            if series:
                keys[key].alternate_keys.append([series, stamp])
    journal.step("Sorting tables into events, snapshots and lists; finding the measures")
    classified = classify_tables(inventory, profiles, keys, calendars, joins, dates)
    kinds = {key: kind for key, (kind, _) in classified.items()}
    measures = find_measures(inventory, profiles, keys, joins, dates, classified)
    journal.step("Checking how each number adds up against the rows")
    pointing: dict[str, list[str]] = {}
    for j in sorted(joins, key=lambda j: -profiles[j.to_table].rows):     # the biggest thing first: products
        if j.trust != "rejected" and not j.to_calendar:
            pointing.setdefault(j.from_table, []).append(j.from_column)
    same = check_arithmetic(warehouse, inventory, profiles, measures, pointing)
    stamped = {key: c.column for key, cs in dates.items() for c in cs
               if c.is_default and c.granularity == "timestamp" and kinds.get(key) == "fact"}
    # What each reading is of: the reading series first (the counters run within it), then every link.
    about = {key: [alt[0] for alt in keys[key].alternate_keys if alt[1:] == [when]][:1] for key, when in stamped.items()}
    for j in joins:
        if j.trust != "rejected" and not j.to_calendar and j.from_table in stamped \
                and j.from_column not in about[j.from_table]:
            about[j.from_table].append(j.from_column)
    find_counters(warehouse, inventory, profiles, {k: v.primary_key for k, v in keys.items()}, about, stamped,
                  measures)
    # A document's figure written on each of its lines (refills per prescription, freight per order): averaged.
    documents = {}
    for k, v in keys.items():
        dated = next((c.column for c in dates.get(k, []) if c.is_default), None)
        dated_columns = {c.column for c in dates.get(k, [])}
        doc = next((alt[0] for alt in v.alternate_keys if len(alt) == 2 and alt[1] not in dated_columns), None) or (
            v.primary_key[0] if len(v.primary_key) == 2 and inventory.tables[k].type_of(v.primary_key[1]) == "integer"
            else None)
        if doc and dated:
            documents[k] = (doc, dated)
    find_parent_figures(warehouse, inventory, profiles, documents, measures)
    for key, kind in list(kinds.items()):
        # A periodic table whose numbers are all amounts for the period (targets,
        # budgets) adds up over time like any fact, and its period dates events, not balances.
        if kind == "snapshot" and not any(m.table == key and m.additivity == "semi_additive" for m in measures):
            kinds[key] = "fact"
            for c in dates.get(key, []):
                if c.kind == "snapshot":
                    c.kind = "event"
                    c.add("flows", 0.0, "its numbers are amounts for each period, added up over time: the period "
                          "dates what happened in it, not a balance")
    return Findings(inventory, profiles, keys, calendars, joins, dates, kinds, measures,
                    values_allowed=options.profile.values_allowed, same_values=same)


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
                         outliers=options.outliers, values_allowed=options.profile.values_allowed)
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

    # People's data: names never sent to the AI; contact details, birth dates and ID numbers never shown.
    people = {key: personal.read_table(table, f.profiles[key]) for key, table in inv.tables.items()}

    # Date roles.
    date_slugs: set[str] = set()
    for key, roles in f.dates.items():
        table_name = inv.tables[key].name
        for c in roles:
            if people[key].get(c.column) == personal.PII:
                continue     # a birth date is never a date questions count by
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
        default_role = next((ck[(key, c.column)] for c in f.dates.get(key, []) if c.is_default
                             and ck[(key, c.column)] in model.date_roles), None)
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
            held = people[key].get(column.name)
            model.columns[k] = Column(
                key=k, table=key, name=column.name, data_type=column.data_type,  # type: ignore[arg-type]
                raw_type=column.raw_type, nullable=column.nullable, comment=column.comment,
                role=role,  # type: ignore[arg-type]
                business_name=names.readable(column.name, column.data_type), format=fmt,  # type: ignore[arg-type]
                values_allowed=f.values_allowed(key, column.name) and held is None,
                sensitivity="pii" if held == personal.PII else "none",
                personal="detail" if held == personal.PII else "name" if held == personal.NAME else "none",
                # People's values are not kept in the model: the catalog and the member list never see them,
                # nor its first and last (two people's birth dates, two emails).
                profile=p.model_copy(update={"top": None, "min": None, "max": None, "min_num": None, "max_num": None,
                                             "date_min": None, "date_max": None}) if held and p is not None else p,
                provenance="profile", status="verified", confidence=1.0)

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
            level = m.additivity == "semi_additive"
            model.review.append(ReviewItem(key=f"additivity:{key_m}", object=key_m, question=m.review,
                                           choice_made="taken at the end of each period" if level else "averaged",
                                           alternatives=["added up over time" if level else "added up"],
                                           evidence=m.evidence))

    _entities_and_attributes(model, f, ck, taken, people)

    model.quality = flags
    for table, a, b, share in f.same_values:
        # One figure twice (an allocated quantity that is the on-hand quantity again): an admin says which is meant.
        key_a, key_b = f"m:{table}.{a.casefold()}", f"m:{table}.{b.casefold()}"
        if key_a not in model.measures or key_b not in model.measures:
            continue
        name_a, name_b = model.measures[key_a].business_name, model.measures[key_b].business_name
        message = (f"{name_a} equals {name_b} on {share:.0%} of rows: one of the two may read the wrong column, "
                   "and answers about either would show the same figure.")
        model.quality.append(QualityFlag(key=f"same_values:{key_a}|{key_b}", object=key_a, kind="same_values",
                                         message=message, severity="warning", data={"other": key_b, "share": share}))
        model.review.append(ReviewItem(key=f"same_values:{key_a}|{key_b}", object=key_a,
                                       question=f"{name_a} equals {name_b} on {share:.0%} of rows. Are both defined "
                                                "right?", choice_made="both kept as they are",
                                       alternatives=[f"{name_a} reads the wrong column", f"{name_b} reads the wrong column"]))
    for key, held in people.items():
        if not held or key not in model.tables:
            continue
        said = {kind: [names.readable(c) for c, k in held.items() if k == kind] for kind in (personal.NAME, personal.PII)}
        parts = []
        if said[personal.NAME]:
            parts.append(f"{', '.join(said[personal.NAME])}: people's names, shown in answers but never sent to the AI")
        if said[personal.PII]:
            parts.append(f"{', '.join(said[personal.PII])}: personal details, never shown, listed or filtered on")
        model.review.append(ReviewItem(
            key=f"personal:{key}", object=key,
            question=f"{model.tables[key].business_name} holds people's data. " + "; ".join(parts) + ".",
            choice_made="kept from the AI", alternatives=["release a column on the Knowledge base page"]))
    for fl in flags:
        if fl.kind == "status_column":
            leave_out, shown = list(fl.data.get("leave_out") or []), list(fl.data.get("leave_out_shown") or [])
            if leave_out and fl.object in model.columns:
                # Rows a status says in full were cancelled, voided, reversed or duplicated are left out by
                # default: every answer says so, and the admin can count them again.
                table = model.tables[model.columns[fl.object].table]
                table.default_filters.append(ColumnFilter(column=fl.object, op="not_in", values=leave_out,
                                                          shown=[str(v) for v in shown]))
            others = [v for v in fl.data.get("cancel_like", []) if v not in shown]
            owner = model.tables[model.columns[fl.object].table] if fl.object in model.columns else None
            asked = f"{owner.business_name}: {fl.message}" if owner is not None and fl.data.get("lookup") else fl.message
            model.review.append(ReviewItem(
                key=f"default_filter:{fl.object}", object=fl.object, question=asked,
                choice_made=f"left out: {', '.join(map(str, shown))}" if leave_out else "all rows are counted",
                alternatives=(["count all rows"] if leave_out else []) + [f"leave out {v}" for v in others]))
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
                             taken: set[str], people: dict[str, dict[str, str]]) -> None:
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
        held = people.get(key, {})
        # A code shown beside each member's name is never a person's data: a unique last name or email is not one.
        texts = [c for c in tkeys.unique_columns if table.type_of(c) == "text" and c not in tkeys.primary_key
                 and c not in held]
        found = _label_column(table, profile, tkeys, held)
        label_key = _two_part_name(model, key, business, found, profile) if isinstance(found, tuple) else (
            ck[(key, found)] if found else None)
        codes = [c for c in texts if c != found and (profile.columns[c].max_len or 99) <= 16]
        model.entities[slug] = Entity(
            slug=slug, business_name=business, table=key, key_columns=[ck[(key, c)] for c in tkeys.primary_key],
            label_column=label_key, code_column=ck[(key, codes[0])] if codes else None,
            members=profile.rows, provenance="profile", status="verified", confidence=0.9)
        if label_key:
            model.columns[label_key].role = "label"
            model.columns[label_key].label_of = model.tables[key].primary_key[0] if model.tables[key].primary_key else None
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
            if col.role in ("key", "foreign_key", "date_key", "date", "audit", "period_key", "measure"):
                continue
            if not _attribute_worthy(column, col, kind):
                continue
            how = _attribute_kind(column, col, kind)
            if how is None:
                continue
            name = _attribute_name(col.business_name, model.entities.get(owner))
            slug = f"{owner}.{ids.slug(name)}"
            if slug in model.attributes:
                slug = ids.unique_slug(slug, set(model.attributes))
            model.attributes[slug] = Attribute(
                slug=slug, column=col.key, entity=owner if key in entity_of else None,
                business_name=col.business_name, members=p.distinct if p else 0, kind=how,
                provenance="profile", status="verified", confidence=0.8)
        # A name read from two columns (first and last) is grouped and filtered by like any other.
        entity = model.entities.get(owner) if key in entity_of else None
        label = model.columns.get(entity.label_column or "") if entity is not None else None
        if label is not None and label.parts and not any(a.column == label.key for a in model.attributes.values()):
            slug = ids.unique_slug(f"{owner}.name", set(model.attributes))
            model.attributes[slug] = Attribute(
                slug=slug, column=label.key, entity=owner, business_name=label.business_name,
                members=label.profile.distinct if label.profile else 0, provenance="profile", status="verified",
                confidence=0.8)


_NAME_WORDS = {"name", "nm", "title", "label"}
_DESCRIPTION_WORDS = {"desc", "dsc", "description", "descr"}


def _label_column(table: InvTable, profile: TableProfile, keys: TableKeys,
                  people: dict[str, str] | None = None) -> str | tuple[str, str] | None:
    """The column that names each member: its own name ("doctor name" in a doctor table, a full name), else a
    name, else a person's first and last name together, else a short description, else the longest unique text.
    Never a contact detail (an address, a phone number, an email), whatever it is called.

    Names need not be unique (two customers can both be "J. Smith"): a name-like
    column filled on nine rows in ten with nine distinct values in ten is a name.
    Grouping by it is made safe by also grouping by the member's code or key.
    """
    rows = profile.rows or 0
    people = people or {}
    subject = [w for w in names.core_table(table.name) if len(w) > 2] if not names.opaque(table.name) else []
    best: list[tuple[int, float, str]] = []
    for column in table.columns:
        name = column.name
        p = profile.columns[name]
        if column.data_type != "text" or name in keys.primary_key or not rows or people.get(name) == personal.PII:
            continue
        if personal.is_first_name(name) or personal.is_last_name(name):
            continue     # half a person's name, however unique: both together name them (below)
        words = set(names.tokens(name))
        unique = name in keys.unique_columns
        named = bool(words & _NAME_WORDS)
        described = bool(words & _DESCRIPTION_WORDS) and (p.avg_len or 0) <= 60
        near_unique = p.non_null >= 0.9 * rows and p.distinct >= 0.9 * p.non_null
        if not (unique or (near_unique and (named or described))):
            continue
        own = named and ("full" in words or any(names.same_word(w, t) for w in words for t in subject))
        rank = -1 if own else 0 if named else 2 if described else 3
        best.append((rank, -(p.avg_len or 0), name))
    found = min(best) if best else None
    if found is not None and found[0] <= 0:
        return found[2]
    # A person named in two parts: both together, before a description or an id-like text.
    texts = [c.name for c in table.columns if c.data_type == "text"]
    first = next((c for c in texts if personal.is_first_name(c)), None)
    last = next((c for c in texts if personal.is_last_name(c)), None)
    if first and last and profile.columns[first].non_null >= 0.9 * rows and profile.columns[last].non_null >= 0.9 * rows:
        return first, last
    return found[2] if found else None


def _two_part_name(model: SemanticModel, key: str, business: str, parts: tuple[str, str],
                   profile: TableProfile) -> str:
    """A name the table holds as a first and a last name: one column of the model, read as both joined (never a
    column of the warehouse). A person's name: shown in answers, never sent to the AI."""
    first, last = parts
    k = f"{key}.{first.casefold()}+{last.casefold()}"
    a, b = profile.columns[first], profile.columns[last]
    model.columns[k] = Column(
        key=k, table=key, name=f"{first}+{last}", data_type="text", role="label", personal="name",
        business_name=f"{business} name", format="text", values_allowed=False,
        parts=[f"{key}.{first.casefold()}", f"{key}.{last.casefold()}"],
        # Two people can share a name: never read as unique, so a grouping keeps them apart by their key.
        profile=ColumnProfile(rows=a.rows, non_null=min(a.non_null, b.non_null),
                              distinct=min(max(a.distinct, b.distinct), max(0, (profile.rows or 0) - 1)),
                              distinct_is_approx=True, avg_len=(a.avg_len or 0) + (b.avg_len or 0) + 1,
                              pattern="name"),
        provenance="profile", status="verified", confidence=0.8)
    return k


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
    if column.data_type in _NUMBERS and last[0] in names.KEY_SUFFIXES - {"code", "cd", "no"}:
        return False   # a key to a table this warehouse does not hold
    return True


_NUMBERS = ("integer", "decimal", "float")
# Words that make a whole number name something rather than count it: an NPI, a ZIP code, a GL account.
_IDENTIFYING = {"code", "cd", "no", "nbr", "num", "number", "npi", "zip", "zipcode", "postal", "postcode", "bin",
                "sku", "upc", "ean", "gtin", "isbn", "account", "acct", "acc"}
# Words of a row's place in its document (line 3 of an invoice, the second parcel of a shipment).
_ORDINAL = {"line", "lin", "ln", "seq", "sequence", "position", "pos"}
_FEW_NUMBERS = 50      # a number with more values than this on a fact row is a figure the measures add up


def _attribute_kind(column: InvColumn, col: Column, kind: str) -> str | None:
    """How a question reads ``col`` (see :class:`Attribute`), or None when it is not one to read.

    A number each member has (a list price, a weight, a ship method's typical transit days) is kept as a
    number to compare, sort and show: dropped, it could not be asked about at all ("ship methods with a
    transit time of 3 days" read as no such data). A whole number that names (an NPI, a GL account code)
    is an identifier, shown as written; a fact's own small numbers (refills authorized, fill number) are
    numbers; its position in a document (line 3) is nothing. A fact's text with more values than a
    category has (an invoice number, a tracking number) identifies its rows; free text is searched.
    """
    p = col.profile
    if p is None:
        return None
    if col.role in ("status", "flag"):
        return "group"
    people = col.personal != "none" or col.sensitivity != "none"
    if col.role == "text":
        return "text" if kind != "bridge" and not people else None
    few = p.distinct <= 20 and p.distinct <= 0.5 * p.non_null     # numbers members share (pack sizes)
    if people:      # read as before: governed where it is shown, never offered in a new way
        if kind == "dimension":
            return "group" if column.data_type not in _NUMBERS or few else None
        return "group" if column.data_type == "text" and 1 < p.distinct <= 1000 else None
    words = set(names.tokens(column.name))
    if column.data_type in _NUMBERS:
        whole = column.data_type == "integer" or p.integer_share == 1
        if whole and words & _IDENTIFYING and (p.min_num or 0) >= 100:
            return "identifier"
        if kind == "dimension":
            return "group" if few else "number"
        if words & _ORDINAL or not 1 < p.distinct <= _FEW_NUMBERS:
            return None
        return "number"
    if kind == "dimension":
        return "group"
    if column.data_type != "text" or p.distinct <= 1:
        return None
    return "group" if p.distinct <= 1000 else "identifier"


def _attribute_name(business: str, entity: Entity | None) -> str:
    if entity is None:
        return business
    words = business.split()
    entity_words = entity.business_name.lower().split()
    if len(words) > 1 and words[0].lower() in entity_words:
        return " ".join(words[1:])
    return business
