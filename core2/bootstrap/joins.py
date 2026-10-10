"""Finding the joins, testing each one on the data, and deciding which can run.

Candidates come from declared foreign keys, from names in any style and from
values (a column whose values sit inside another table's unique key). Every
candidate is test-run: how many rows find a match, how many keys are empty, how
many distinct values match nothing. A join the data verifies (99% of rows match
a unique key) can run at once; one that matches partly is proposed (used only
when nothing better exists, and named in the answer); one that fails is dropped.

Small whole-number keys (1, 2, 3 ...) sit inside many tables' keys at once, so a
value-only candidate must also cover most of its target (a column with 12 values
pointing at a 12-row table, not at a 240-row one) or spread over its range. Counts,
minutes and line numbers are runs of whole numbers too: a run of every number in a
stretch is a reference only when it reaches the target's newest key or uses nearly
all of them, and a small table (under 30 members) must be mostly used. A backup copy
of a table (DEVICES_BAK beside DEVICES) is never what a column points at.
When one column still verifies against two tables equally well, neither is
trusted: both are proposed and the choice goes to the review queue.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from sqlglot import exp

from core2.bootstrap import names
from core2.bootstrap.calendar import CalendarFinding
from core2.bootstrap.inventory import InvColumn, Inventory, InvTable
from core2.bootstrap.keys import TableKeys
from core2.bootstrap.profiler import TableProfile
from core2.model.schema import ColumnProfile, Evidence
from core2.bootstrap.journal import attempt, journal_of
from core2.warehouse import dialect as D
from core2.warehouse.runner import Warehouse

VERIFIED = 0.99
PROPOSED = 0.5
SAMPLE_ABOVE = 5_000_000


@dataclass
class JoinFinding:
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    declared: bool = False
    name_score: float = 0.0
    role_tokens: list[str] = field(default_factory=list)
    to_calendar: bool = False
    rows: int = 0
    non_null: int = 0
    unmatched: int = 0
    unmatched_values: int = 0
    placeholder_rows: int = 0
    match_rate: float = 0.0
    null_rate: float = 0.0
    coverage: float = 0.0
    from_unique: bool = False
    trust: str = "proposed"           # declared | verified | proposed | rejected
    role: str | None = None
    evidence: list[Evidence] = field(default_factory=list)
    also: list[tuple[str, str]] = field(default_factory=list)   # a multi-column key's other (from, to) pairs
    to_alternate: bool = False        # the target is another unique column of its table, not its key

    @property
    def ident(self) -> tuple[str, str, str, str]:
        return (self.from_table, self.from_column, self.to_table, self.to_column)

    @property
    def pairs(self) -> list[tuple[str, str]]:
        """Every (from column, to column) of the key, the first one first."""
        return [(self.from_column, self.to_column), *self.also]

    @property
    def from_columns(self) -> list[str]:
        return [f for f, _ in self.pairs]


def _compatible(a: str, b: str) -> bool:
    numeric = {"integer", "decimal"}
    return (a in numeric and b in numeric) or a == b == "text" or a == b == "date"


def text_and_number(a: str, b: str) -> bool:
    """A code kept as text ('0007') on one side, a whole-number key (7) on the other."""
    return {a, b} == {"text", "integer"}


_HIERARCHY = {"parent", "prnt", "par", "mgr", "manager", "reports", "rpt", "supervisor", "head", "rollup",
              "roll", "up", "sup", "lead", "owner", "uplink", "upln", "upstream", "upstrm"}
# Words that make a column a figure about the thing it names, not a pointer at it: QTY_PRESCRIBED,
# TOTAL_CUSTOMERS, AVG_ORDER.
_FIGURE_WORDS = {"qty", "quantity", "cnt", "count", "total", "tot", "sum", "avg", "average", "amt", "amount", "pct",
                 "percent"}
# A table's own number beside its key: BOOKING_NUMBER beside BOOKING_ID.
_NUMBER_WORDS = {"number", "nr"}
# A copy of a table, kept beside it: DEVICES_BAK, ORDERS_OLD, CUSTOMERS_20240101.
_COPY_WORDS = {"bak", "backup", "bkp", "bck", "old", "copy", "cpy", "tmp", "archive", "archived", "arch",
               "prev", "previous", "save", "saved"}
SMALL_TARGET = 30          # members: below this, values alone must use most of them
RUN = 0.9                  # share of a stretch's whole numbers a column holds for it to be a run of numbers


def name_score(from_column: str, to_table: str, to_column: str) -> tuple[float, list[str]]:
    """How strongly the names say ``from_column`` points at ``to_table.to_column``, and the role words left over."""
    if names.opaque(from_column) or names.opaque(to_column) and names.opaque(to_table):
        return 0.0, []
    if names.tokens(from_column) == names.tokens(to_column) and not names.opaque(to_column):
        return 1.0, []
    a = names.core_column(from_column)
    k = names.core_column(to_column)
    u = names.core_table(to_table)
    for target, score in ((k, 0.95), (u, 0.9)):
        if target and len(a) == len(target) and all(names.same_word(x, y) for x, y in zip(a, target)):
            return score, []
    for target, score in ((k, 0.85), (u, 0.8)):
        if target and names.ends_with(a, target) and len(a) > len(target):
            role = a[: len(a) - len(target)]
            if set(role) & _FIGURE_WORDS:
                return 0.0, []      # QTY_PRESCRIBED is how much was prescribed, not the prescriber
            return score, role
    words = names.tokens(from_column)
    keyed = bool(words) and words[-1] in names.KEY_SUFFIXES
    for target, score in ((k, 0.8), (u, 0.75)):
        # The role after the name: ABC_CLASS_VOLUME_KEY points at ABC_CLASS as its "volume" role. Only a key
        # says so: RENTAL_DURATION and VISITS_INCLUDED are figures about rentals and visits.
        if keyed and target and len(a) > len(target) and all(names.same_word(x, y) for x, y in zip(a, target)):
            return score, a[len(target):]
    return 0.0, []


def run_of_numbers(a: ColumnProfile) -> bool:
    """Every whole number of a stretch, or nearly: 30, 31 ... 299 minutes; lines 1, 2, 3, 4."""
    if a.min_num is None or a.max_num is None or a.distinct < 3:
        return False
    return a.distinct >= RUN * (a.max_num - a.min_num + 1)


def _value_plausible(a: ColumnProfile, k: ColumnProfile, *, a_type: str, k_is_dimension: bool) -> tuple[bool, float]:
    """Could ``a``'s values sit inside key ``k``, judging by values alone? Returns (plausible, coverage).

    Dense surrogate keys (1..N) contain every whole number in their range, so
    sitting inside the range proves little. A value-only candidate must use most
    of its target's members, or a good share of a dimension's members when it is
    a whole-number column; most of a small table's members (any column of small
    numbers fits inside 1..6); and when it is a run of numbers (minutes 30..299,
    lines 1..4), reach the target's newest key or use nearly every one: counts and
    measures stop wherever they stop, references reach the members that exist.
    """
    if a.distinct < 2 or k.distinct < 2:
        return False, 0.0
    coverage = min(1.0, a.distinct / k.distinct)
    if a.distinct > k.distinct * 1.05 + 5:
        return False, coverage
    enough = coverage >= 0.5 or (coverage >= 0.2 and a_type == "integer" and k_is_dimension)
    if k.distinct < SMALL_TARGET and coverage < 0.7:
        enough = False
    if a.min_num is not None and k.min_num is not None and a.max_num is not None and k.max_num is not None:
        if a.max_num < k.min_num or a.min_num > k.max_num:
            return False, coverage
        if run_of_numbers(a) and a.max_num < k.max_num and coverage < 0.9:
            return False, coverage
        return enough, coverage
    if a.max_len is not None and k.max_len is not None:
        if a.max_len > k.max_len + 2 or (a.min_len or 0) + 2 < (k.min_len or 0):
            return False, coverage
        return enough, coverage
    return False, coverage


_NO_COLUMN = InvColumn("", "", "other")


def _test(warehouse: Warehouse, inventory: Inventory, finding: JoinFinding, profiles: dict[str, TableProfile],
          placeholders: list) -> JoinFinding:
    d = warehouse.dialect
    src: InvTable = inventory.tables[finding.from_table]
    tgt: InvTable = inventory.tables[finding.to_table]
    a = exp.column(D.ident(finding.from_column, d), table="f")
    k = exp.column(D.ident(finding.to_column, d), table="t")
    one = exp.Literal.number(1)

    def equal(from_column: str, to_column: str) -> exp.Expression:
        fa = exp.column(D.ident(from_column, d), table="f")
        tk = exp.column(D.ident(to_column, d), table="t")
        raws = ((src.column(from_column) or _NO_COLUMN).raw_type, (tgt.column(to_column) or _NO_COLUMN).raw_type)
        kinds = (src.type_of(from_column), tgt.type_of(to_column))
        if text_and_number(*kinds):
            return D.same_number(fa, tk, d) if kinds[0] == "text" else D.same_number(tk, fa, d)
        return D.same_text(fa, tk, d, raws) if kinds[0] == "text" else exp.EQ(this=fa, expression=tk)

    def count_if(condition: exp.Expr) -> exp.Expression:
        return exp.Sum(this=exp.Case(ifs=[exp.If(this=condition, true=one.copy())], default=exp.Literal.number(0)))

    unmatched = exp.and_(exp.not_(exp.Is(this=a.copy(), expression=exp.Null())),
                         exp.Is(this=k.copy(), expression=exp.Null()))
    selects = [exp.Count(this=one.copy()), exp.Count(this=a.copy()), count_if(unmatched),
               exp.Count(this=exp.Distinct(expressions=[exp.Case(ifs=[exp.If(this=unmatched.copy(), true=a.copy())])]))]
    if placeholders:
        selects.append(count_if(exp.In(this=a.copy(), expressions=[
            exp.Literal.number(p) if isinstance(p, (int, float)) else exp.Literal.string(str(p)) for p in placeholders])))
    target = exp.select(*[exp.column(D.ident(t, d)) for _, t in finding.pairs]).distinct().from_(
        exp.to_table("__TGT__"))
    on = exp.and_(*[equal(f, t) for f, t in finding.pairs])
    query = (exp.select(*[s.as_(f"s{i}") for i, s in enumerate(selects)])
             .from_(exp.to_table("__SRC__").as_("f"))
             .join(target.subquery("t"), on=on, join_type="left"))
    rows = profiles[finding.from_table].rows
    name = D.table_sql(src.database, src.schema, src.name, d)
    sql = query.sql(dialect=d).replace("__TGT__", D.table_sql(tgt.database, tgt.schema, tgt.name, d), 1)
    if rows > SAMPLE_ABOVE:
        # The sample in its own query: where an alias goes around a sampling clause
        # differs by warehouse (Azure SQL wants "t AS f TABLESAMPLE", Oracle "t SAMPLE f").
        sample = f"(SELECT * FROM {name} {D.sample_clause(d, rows=1_000_000, total_rows=rows)})"
        try:
            values = warehouse.query(sql.replace("__SRC__", sample, 1)).rows[0]
        except Exception:  # noqa: BLE001 - a view is not sampled on Azure SQL: its first rows then
            values = warehouse.query(sql.replace("__SRC__", D.first_rows(name, d, rows=1_000_000), 1)).rows[0]
    else:
        values = warehouse.query(sql.replace("__SRC__", name, 1)).rows[0]
    finding.rows, finding.non_null = int(values[0] or 0), int(values[1] or 0)
    finding.unmatched, finding.unmatched_values = int(values[2] or 0), int(values[3] or 0)
    finding.placeholder_rows = int(values[4] or 0) if placeholders else 0
    finding.null_rate = 1.0 - finding.non_null / finding.rows if finding.rows else 0.0
    finding.match_rate = (finding.non_null - finding.unmatched) / finding.non_null if finding.non_null else 0.0
    return finding


def discover_joins(warehouse: Warehouse, inventory: Inventory, profiles: dict[str, TableProfile],
                   keys: dict[str, TableKeys], calendars: dict[str, CalendarFinding], *,
                   max_tests: int = 400, workers: int = 4) -> list[JoinFinding]:
    declared = {(fk.table, names_norm(fk.columns[0]), fk.ref_table, names_norm(fk.ref_columns[0]))
                for fk in inventory.foreign_keys if len(fk.columns) == 1}

    # Targets: single-column unique keys; for calendars only their key (or date).
    targets: list[tuple[str, str]] = []
    # A table's other unique keys (a circuit code beside the service id): pointed at only by a column
    # of the same name, or one the database declares; values alone would find them by accident.
    alternates: set[tuple[str, str]] = set()
    copies = _copies(inventory)
    for key, table in inventory.tables.items():
        if key in calendars:
            cal = calendars[key]
            targets.append((key, cal.key_column or cal.date_column))
            continue
        if key in copies:
            continue    # links point at the table itself, not at its copy; a declared one still stands
        primary = keys[key].primary_key
        for unique in keys[key].unique_columns:
            kind = table.type_of(unique)
            p = profiles[key].columns[unique]
            if kind not in ("integer", "text", "decimal"):
                continue
            if [unique] != primary:
                # Besides the key itself, only the table's own code (WHS_CD on the
                # warehouses), in a table big enough that its uniqueness is not an
                # accident: a warehouse's profit-centre code is unique in a small
                # warehouse table, and still the profit centre's, not the warehouse's.
                words = names.tokens(unique)
                if not (words and words[-1] in names.KEY_SUFFIXES | _NUMBER_WORDS):
                    continue   # a unique name or title is what a member is called, not what points at it
                if profiles[key].rows < 20 or not _names_its_table(unique, table.name):
                    if any(other != key and _names_its_table(unique, t.name) for other, t in inventory.tables.items()):
                        continue   # FILM_ID unique on a film's one category row is still the film's key
                    alternates.add((key, unique))
            if kind == "text" and (p.avg_len or 0) > 24:
                continue   # names and descriptions are not what keys point at
            if kind == "decimal" and p.integer_share != 1.0:
                continue
            targets.append((key, unique))

    # A table with a unique text column that is not its key reads like a dimension
    # (it names its members); value-only candidates may point at it more freely.
    dimension_like = {
        key for key, table in inventory.tables.items()
        if any(table.type_of(c) == "text" and c not in keys[key].primary_key for c in keys[key].unique_columns)}

    candidates: dict[tuple[str, str, str, str], JoinFinding] = {}
    # What the database declares is a candidate whatever the values look like: one value in every row
    # (every film in one language), a key of two columns, a target that is not the table's own key.
    for fk in inventory.foreign_keys:
        src, tgt = inventory.tables.get(fk.table), inventory.tables.get(fk.ref_table)
        if src is None or tgt is None or len(fk.columns) != len(fk.ref_columns):
            continue
        pairs = [(_named(src, a), _named(tgt, b)) for a, b in zip(fk.columns, fk.ref_columns)]
        if any(a is None or b is None for a, b in pairs) or not all(
                _compatible(src.type_of(a), tgt.type_of(b)) or text_and_number(src.type_of(a), tgt.type_of(b))
                for a, b in pairs):  # type: ignore[arg-type]
            continue
        (first, to_first), *rest = pairs
        score, role = name_score(first, tgt.name, to_first)  # type: ignore[arg-type]
        a, k = profiles[fk.table].columns[first], profiles[fk.ref_table].columns[to_first]  # type: ignore[index]
        candidates[(fk.table, first, fk.ref_table, to_first)] = JoinFinding(  # type: ignore[index]
            fk.table, first, fk.ref_table, to_first, declared=True, name_score=score,  # type: ignore[arg-type]
            role_tokens=role, to_calendar=fk.ref_table in calendars,
            coverage=min(1.0, a.distinct / k.distinct) if k.distinct else 0.0,
            from_unique=len(pairs) == 1 and first in keys[fk.table].unique_columns,
            also=rest)  # type: ignore[arg-type]

    for key, table in inventory.tables.items():
        tkeys = keys[key]
        is_calendar = key in calendars
        # A key's later columns that restart inside each of its first column's values (order number +
        # line 1, 2, 3) number lines, and point nowhere; a later column with more values than the first
        # (interface name + device) is an ordinary column of the key, free to point at its own table.
        sequence_columns = {c for alt in tkeys.alternate_keys for c in alt[1:]
                            if profiles[key].columns[c].distinct < profiles[key].columns[alt[0]].distinct}
        sequence_columns |= set(tkeys.line_numbers)
        for column in table.columns:
            a = profiles[key].columns[column.name]
            own_key = column.name in tkeys.primary_key and len(tkeys.primary_key) == 1
            if own_key and _names_its_table(column.name, table.name):
                continue   # CUSTOMER_ID on CUSTOMERS identifies the row; it points nowhere
            if column.name in sequence_columns or a.distinct < 2 or column.data_type not in ("integer", "text", "decimal", "date"):
                continue
            if a.pattern in ("flag01", "flag", "flag_yn"):
                continue
            if column.data_type == "decimal" and a.integer_share != 1.0:
                continue
            for to_table, to_column in targets:
                if to_table == key and to_column == column.name:
                    continue
                if (key, column.name, to_table, to_column) in candidates:
                    continue          # declared, and already a candidate
                to_type = inventory.tables[to_table].type_of(to_column)
                mixed = text_and_number(column.data_type, to_type)
                if not (_compatible(column.data_type, to_type) or mixed):
                    continue
                ident = (key, column.name, to_table, to_column)
                is_declared = (key, names_norm(column.name), to_table, names_norm(to_column)) in declared
                score, role = name_score(column.name, inventory.tables[to_table].name, to_column)
                if mixed and not (is_declared or score >= 0.9):
                    continue          # a code kept as text ('0007') points at a number key when the names say so
                k = profiles[to_table].columns[to_column]
                plausible, coverage = _value_plausible(a, k, a_type=column.data_type,
                                                       k_is_dimension=to_table in dimension_like)
                if (to_table, to_column) in alternates and not (
                        is_declared or score >= 0.95 or plausible and column.data_type == "text"):
                    continue          # another unique column of the table: named the same, declared, or a code
                                      # whose values are its codes (tested: every row must find one)
                to_calendar = to_table in calendars
                if to_calendar:
                    # Date keys are recognised by shape, whatever their names: a day key
                    # points at a day calendar, a yyyymm period at a month calendar.
                    grain = calendars[to_table].grain
                    plausible = (a.pattern == "yyyymmdd" and grain == "day") or (a.pattern == "yyyymm" and grain == "month") \
                        or (column.data_type == "date" and grain == "day") or is_declared or score > 0
                words = names.tokens(column.name)
                # MGR_KEY on the employees: a key named for a step up a hierarchy. Few rows are managers,
                # so the share of members it uses proves nothing; the test run decides.
                up_the_hierarchy = to_table == key and bool(_HIERARCHY & set(words)) and bool(words) \
                    and words[-1] in names.KEY_SUFFIXES
                if to_table == key and not (is_declared or score >= 0.8 and up_the_hierarchy
                                            or up_the_hierarchy and column.data_type == inventory.tables[key].type_of(
                                                to_column)):
                    continue   # a table points at itself only up a hierarchy (a parent, a manager)
                if is_calendar and not (is_declared or score >= 0.9):
                    continue   # a calendar's own period numbers are not foreign keys
                if own_key and not (is_declared or score >= 0.9):
                    continue   # a table's own key points elsewhere only when it says so (1:1 extensions)
                if plausible and not (is_declared or score) and not to_calendar \
                        and not names.opaque(column.name) and not _shares_a_word(column.name, inventory.tables[to_table].name):
                    plausible = False   # whole-number keys collide by chance: a meaningful name must agree
                plausible = plausible or up_the_hierarchy
                if not (is_declared or score >= 0.75 or plausible):
                    continue
                candidates[ident] = JoinFinding(*ident, declared=is_declared, name_score=score, role_tokens=role,
                                                to_calendar=to_calendar, coverage=coverage,
                                                from_unique=column.name in tkeys.unique_columns,
                                                to_alternate=(to_table, to_column) in alternates)

    # A document and its line, named alike in another table (a claim's prescription and fill number): one
    # link of two columns, never two links to half a key each. Only a document's lines: a day and an item
    # key a balance, and a movement on that day of that item is not one of its lines.
    for to_key, target in inventory.tables.items():
        lines = set(keys[to_key].line_numbers)
        for key_columns in [keys[to_key].primary_key, *keys[to_key].alternate_keys]:
            if len(key_columns) != 2 or key_columns[1] not in lines or to_key in copies:
                continue
            for key, table in inventory.tables.items():
                pairs = [(next((c.name for c in table.columns if not names.opaque(c.name)
                                and names.tokens(c.name) == names.tokens(column)
                                and (_compatible(c.data_type, target.type_of(column))
                                     or text_and_number(c.data_type, target.type_of(column)))), None), column)
                         for column in key_columns]
                (first, to_first), rest = pairs[0], pairs[1:]
                if key == to_key or first is None or any(f is None for f, _ in rest):
                    continue
                if (key, first, to_key, to_first) in candidates:
                    continue          # declared
                a, k = profiles[key].columns[first], profiles[to_key].columns[to_first]
                candidates[(key, "+".join(f for f, _ in pairs), to_key, "+".join(t for _, t in pairs))] = JoinFinding(
                    key, first, to_key, to_first, name_score=1.0, to_alternate=True,
                    coverage=min(1.0, a.distinct / k.distinct) if k.distinct else 0.0, also=rest)  # type: ignore[arg-type]

    ranked = sorted(candidates.values(), key=lambda f: (-f.declared, -f.name_score, -f.coverage, f.ident))
    tested = ranked[:max_tests]
    placeholder_of = {key: cal.placeholders for key, cal in calendars.items()}
    journal_of(warehouse).step(f"Testing {len(tested)} possible joins between tables against the data")

    def test(f: JoinFinding) -> JoinFinding:
        # A test the database refuses leaves the join untested: it then has no match to show and is not trusted.
        what = (f"the join test {inventory.tables[f.from_table].name}.{f.from_column} -> "
                f"{inventory.tables[f.to_table].name}.{f.to_column}")
        return attempt(warehouse, what, lambda: _test(warehouse, inventory, f, profiles,
                                                      placeholder_of.get(f.to_table, [])), f)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(test, tested))

    for f in tested:
        unmatched_note = (f"; {f.unmatched:,} rows hold {f.unmatched_values:,} values found nowhere"
                          if f.unmatched else "")
        f.evidence.append(Evidence(kind="test_run", weight=f.match_rate,
                                   data={"rows": f.rows, "non_null": f.non_null, "unmatched": f.unmatched,
                                         "unmatched_values": f.unmatched_values, "placeholders": f.placeholder_rows},
                                   detail=f"{f.match_rate:.1%} of {f.non_null:,} filled rows match{unmatched_note}"))
        if f.declared:
            f.evidence.append(Evidence(kind="declared_fk", detail="declared as a foreign key", weight=1.0))
        if f.name_score:
            f.evidence.append(Evidence(kind="name_match", weight=f.name_score,
                                       detail=f"the names match ({f.from_column} -> {f.to_column})"))
        distinct = profiles[f.from_table].columns[f.from_column].distinct
        members = profiles[f.to_table].columns[f.to_column].distinct
        value_only = not (f.declared or f.name_score)
        if value_only and f.unmatched_values > 0.2 * distinct:
            # Values alone, and a fifth of them found nowhere (seats 10, 15, 20 and 25 against six plans):
            # the numbers only happen to overlap. A real link's strays are a few deleted members.
            f.trust = "rejected"
            f.evidence.append(Evidence(kind="strays", weight=-1.0, detail=(
                f"{f.unmatched_values:,} of its {distinct:,} values are found nowhere: the numbers only overlap")))
        elif f.also and not f.declared and f.match_rate < VERIFIED:
            # A key of two columns named alike in both tables: only where every row finds its line. A fill
            # with no claim is no fault of the fill; the claim is what points at the fill.
            f.trust = "rejected"
            f.evidence.append(Evidence(kind="strays", weight=-1.0, detail=(
                "not every row finds its line, and no one declared the link")))
        elif value_only and f.to_alternate and f.match_rate < VERIFIED:
            # Another table's own code, by its values alone: two tables' postal codes share a few.
            f.trust = "rejected"
            f.evidence.append(Evidence(kind="strays", weight=-1.0, detail=(
                "not every row finds one, and nothing but the values says it is a link")))
        elif value_only and not f.to_calendar and f.match_rate < VERIFIED \
                and distinct - f.unmatched_values < 0.9 * members:
            # Values alone, and some rows find nothing: only a column using nearly every member is the
            # members' own, with a few deleted or not yet loaded. Refills allowed 0..11 use five of 18 pharmacists.
            f.trust = "rejected"
            f.evidence.append(Evidence(kind="strays", weight=-1.0, detail=(
                f"{f.non_null - round(f.match_rate * f.non_null):,} rows find nothing, and nothing but the values "
                "says it is a link")))
        elif f.match_rate >= VERIFIED:
            f.trust = "verified"
        elif f.declared and f.match_rate >= PROPOSED:
            f.trust = "declared"   # usable, as the database says; the unmatched rows are reported
        elif f.match_rate >= PROPOSED or (
                (f.declared or f.name_score >= 0.95 and not f.to_alternate) and f.match_rate > 0):
            # A join its names (or the database) vouch for stays usable when most keys
            # find nothing: the answer reports the unmatched rows rather than losing
            # the grouping altogether. Not so for another table's same-named unique column:
            # two tables' postal codes share a name, and a few values, and neither points at the other.
            f.trust = "proposed"
        else:
            f.trust = "rejected"

    # One target per column: the best-supported one. A tie between value-only
    # candidates is not decided here.
    by_column: dict[tuple[str, tuple[str, ...]], list[JoinFinding]] = {}
    for f in tested:
        if f.trust != "rejected":
            # A key of two columns is its own choice: the order number in it may also point at the orders.
            by_column.setdefault((f.from_table, tuple(f.from_columns)), []).append(f)
    kept: list[JoinFinding] = []
    for group in by_column.values():
        order = {"verified": 0, "declared": 1, "proposed": 2}
        group.sort(key=lambda f: (order[f.trust], -f.declared, -f.name_score, -f.coverage, -f.match_rate, f.ident))
        best = group[0]
        if len(group) > 1 and not best.declared and best.name_score == 0:
            rival = group[1]
            if rival.trust == best.trust and rival.name_score == 0 and abs(rival.coverage - best.coverage) < 0.05:
                for f in (best, rival):
                    f.trust = "proposed"
                    f.evidence.append(Evidence(kind="ambiguous", weight=-0.5,
                                               detail=f"{f.from_column} matches two tables equally well"))
                kept += [best, rival]
                continue
        kept.append(best)

    _name_roles(kept, inventory)
    return sorted(kept, key=lambda f: f.ident)


def _copies(inventory: Inventory) -> set[str]:
    """Tables kept as a copy of another: named as it is, with a word or a date saying it is a copy."""
    by_name: dict[tuple[str, ...], str] = {}
    for key, table in inventory.tables.items():
        by_name.setdefault(tuple(names.singular(t) for t in names.tokens(table.name)), key)
    out = set()
    for key, table in inventory.tables.items():
        words = [names.singular(t) for t in names.tokens(table.name)]
        marks = [t for t in words if t in _COPY_WORDS or t.isdigit()]
        rest = tuple(t for t in words if t not in _COPY_WORDS and not t.isdigit())
        if marks and rest and by_name.get(rest, key) != key:
            out.add(key)
    return out


def _named(table: InvTable, name: str) -> str | None:
    """The table's column of that name, as the table spells it."""
    wanted = names_norm(name)
    return next((c.name for c in table.columns if names_norm(c.name) == wanted), None)


def names_norm(name: str) -> str:
    return name.casefold()


def _names_its_table(column: str, table: str) -> bool:
    """CUSTOMER_ID, or BOOKING_NUMBER, on the table it is named for."""
    core, own = names.core_column(column), names.core_table(table)
    while len(core) > 1 and core[-1] in _NUMBER_WORDS:
        core = core[:-1]
    return bool(core) and len(core) == len(own) and all(names.same_word(a, b) for a, b in zip(core, own))


# Words too general to tie a column to a table on their own ("seller status" is not "item stock status").
_GENERIC_WORDS = {"sts", "status", "typ", "type", "cd", "code", "grp", "group", "cls", "class", "cat", "category",
                  "lvl", "level", "flg", "flag", "dms", "dim", "nm", "name", "desc", "dsc", "id", "key", "no", "nbr"}


def _shares_a_word(column: str, table: str) -> bool:
    left = [t for t in names.tokens(column) if t not in names.KEY_SUFFIXES and not t.isdigit() and t not in _GENERIC_WORDS]
    right = [t for t in names.core_table(table) if t not in _GENERIC_WORDS]
    return any(names.same_word(a, b) for a in left for b in right)


def _name_roles(joins: list[JoinFinding], inventory: Inventory) -> None:
    """A role name when a table joins the same target more than one way, or under another name."""
    by_pair: dict[tuple[str, str], list[JoinFinding]] = {}
    for f in joins:
        by_pair.setdefault((f.from_table, f.to_table), []).append(f)
    for group in by_pair.values():
        for f in group:
            words = list(f.role_tokens)
            own = False         # the column names the role itself (MANAGER_ID on employees: the manager)
            if not words and (len(group) > 1 or f.from_table == f.to_table) and not names.opaque(f.from_column):
                words = names.core_column(f.from_column)
                own = f.from_table == f.to_table
            if not words:
                continue
            # Read in the company of the noun: "CFM DLY" before a date is a confirmed delivery.
            text = names.read_tokens([*words, "dt"]).rsplit(" ", 1)[0] if f.to_calendar else names.read_tokens(words)
            if f.to_calendar:
                noun = "date"
            else:
                target = inventory.tables[f.to_table].name
                noun = "" if names.opaque(target) else " ".join(
                    names.EXPANSIONS.get(w, w) for w in names.core_table(target))
            if noun and not own and not text.endswith(noun):
                text = f"{text} {noun}"
            f.role = text[:1].upper() + text[1:]
