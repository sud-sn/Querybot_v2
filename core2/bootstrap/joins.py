"""Finding the joins, testing each one on the data, and deciding which can run.

Candidates come from declared foreign keys, from names in any style and from
values (a column whose values sit inside another table's unique key). Every
candidate is test-run: how many rows find a match, how many keys are empty, how
many distinct values match nothing. A join the data verifies (99% of rows match
a unique key) can run at once; one that matches partly is proposed (used only
when nothing better exists, and named in the answer); one that fails is dropped.

Small whole-number keys (1, 2, 3 ...) sit inside many tables' keys at once, so a
value-only candidate must also cover most of its target (a column with 12 values
pointing at a 12-row table, not at a 240-row one) or spread over its range.
When one column still verifies against two tables equally well, neither is
trusted: both are proposed and the choice goes to the review queue.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from sqlglot import exp

from core2.bootstrap import names
from core2.bootstrap.calendar import CalendarFinding
from core2.bootstrap.inventory import Inventory, InvTable
from core2.bootstrap.keys import TableKeys
from core2.bootstrap.profiler import TableProfile
from core2.model.schema import ColumnProfile, Evidence
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

    @property
    def ident(self) -> tuple[str, str, str, str]:
        return (self.from_table, self.from_column, self.to_table, self.to_column)


def _compatible(a: str, b: str) -> bool:
    numeric = {"integer", "decimal"}
    return (a in numeric and b in numeric) or a == b == "text" or a == b == "date"


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
            return score, a[: len(a) - len(target)]
    for target, score in ((k, 0.8), (u, 0.75)):
        # The role after the name: ABC_CLASS_VOLUME_KEY points at ABC_CLASS as its "volume" role.
        if target and len(a) > len(target) and all(names.same_word(x, y) for x, y in zip(a, target)):
            return score, a[len(target):]
    return 0.0, []


def _value_plausible(a: ColumnProfile, k: ColumnProfile, *, a_type: str, k_is_dimension: bool) -> tuple[bool, float]:
    """Could ``a``'s values sit inside key ``k``, judging by values alone? Returns (plausible, coverage).

    Dense surrogate keys (1..N) contain every whole number in their range, so
    sitting inside the range proves little. A value-only candidate must use most
    of its target's members, or a good share of a dimension's members when it is
    a whole-number column.
    """
    if a.distinct < 2 or k.distinct < 2:
        return False, 0.0
    coverage = min(1.0, a.distinct / k.distinct)
    if a.distinct > k.distinct * 1.05 + 5:
        return False, coverage
    enough = coverage >= 0.5 or (coverage >= 0.2 and a_type == "integer" and k_is_dimension)
    if a.min_num is not None and k.min_num is not None and a.max_num is not None and k.max_num is not None:
        if a.max_num < k.min_num or a.min_num > k.max_num:
            return False, coverage
        return enough, coverage
    if a.max_len is not None and k.max_len is not None:
        if a.max_len > k.max_len + 2 or (a.min_len or 0) + 2 < (k.min_len or 0):
            return False, coverage
        return enough, coverage
    return False, coverage


def _test(warehouse: Warehouse, inventory: Inventory, finding: JoinFinding, profiles: dict[str, TableProfile],
          placeholders: list) -> JoinFinding:
    d = warehouse.dialect
    src: InvTable = inventory.tables[finding.from_table]
    tgt: InvTable = inventory.tables[finding.to_table]
    a = exp.column(D.ident(finding.from_column, d), table="f")
    k = exp.column(D.ident(finding.to_column, d), table="t")
    one = exp.Literal.number(1)

    def count_if(condition: exp.Expr) -> exp.Expression:
        return exp.Sum(this=exp.Case(ifs=[exp.If(this=condition, true=one.copy())], default=exp.Literal.number(0)))

    unmatched = exp.and_(exp.not_(exp.Is(this=a.copy(), expression=exp.Null())),
                         exp.Is(this=k.copy(), expression=exp.Null()))
    selects = [exp.Count(this=one.copy()), exp.Count(this=a.copy()), count_if(unmatched),
               exp.Count(this=exp.Distinct(expressions=[exp.Case(ifs=[exp.If(this=unmatched.copy(), true=a.copy())])]))]
    if placeholders:
        selects.append(count_if(exp.In(this=a.copy(), expressions=[
            exp.Literal.number(p) if isinstance(p, (int, float)) else exp.Literal.string(str(p)) for p in placeholders])))
    target = exp.select(exp.column(D.ident(finding.to_column, d))).distinct().from_(exp.to_table("__TGT__"))
    query = (exp.select(*[s.as_(f"s{i}") for i, s in enumerate(selects)])
             .from_(exp.to_table("__SRC__").as_("f"))
             .join(target.subquery("t"), on=exp.EQ(this=a.copy(), expression=k.copy()), join_type="left"))
    rows = profiles[finding.from_table].rows
    source = D.table_sql(src.database, src.schema, src.name, d)
    if rows > SAMPLE_ABOVE:
        source = f"{source} {D.sample_clause(d, rows=1_000_000, total_rows=rows)}"
    sql = (query.sql(dialect=d).replace("__TGT__", D.table_sql(tgt.database, tgt.schema, tgt.name, d), 1)
           .replace("__SRC__", source, 1))
    values = warehouse.query(sql).rows[0]
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
    for key, table in inventory.tables.items():
        if key in calendars:
            cal = calendars[key]
            targets.append((key, cal.key_column or cal.date_column))
            continue
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
                if not (words and words[-1] in names.KEY_SUFFIXES) or profiles[key].rows < 20 \
                        or not _names_its_table(unique, table.name):
                    continue
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
    for key, table in inventory.tables.items():
        tkeys = keys[key]
        is_calendar = key in calendars
        sequence_columns = {c for alt in tkeys.alternate_keys for c in alt[1:]}
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
                if not _compatible(column.data_type, inventory.tables[to_table].type_of(to_column)):
                    continue
                ident = (key, column.name, to_table, to_column)
                is_declared = (key, names_norm(column.name), to_table, names_norm(to_column)) in declared
                score, role = name_score(column.name, inventory.tables[to_table].name, to_column)
                k = profiles[to_table].columns[to_column]
                plausible, coverage = _value_plausible(a, k, a_type=column.data_type,
                                                       k_is_dimension=to_table in dimension_like)
                to_calendar = to_table in calendars
                if to_calendar:
                    # Date keys are recognised by shape, whatever their names: a day key
                    # points at a day calendar, a yyyymm period at a month calendar.
                    grain = calendars[to_table].grain
                    plausible = (a.pattern == "yyyymmdd" and grain == "day") or (a.pattern == "yyyymm" and grain == "month") \
                        or (column.data_type == "date" and grain == "day") or is_declared or score > 0
                if to_table == key and not (is_declared or score >= 0.8):
                    continue   # a table pointing at itself needs more than values
                if is_calendar and not (is_declared or score >= 0.9):
                    continue   # a calendar's own period numbers are not foreign keys
                if own_key and not (is_declared or score >= 0.9):
                    continue   # a table's own key points elsewhere only when it says so (1:1 extensions)
                if plausible and not (is_declared or score) and not to_calendar \
                        and not names.opaque(column.name) and not _shares_a_word(column.name, inventory.tables[to_table].name):
                    plausible = False   # whole-number keys collide by chance: a meaningful name must agree
                if not (is_declared or score >= 0.75 or plausible):
                    continue
                candidates[ident] = JoinFinding(*ident, declared=is_declared, name_score=score, role_tokens=role,
                                                to_calendar=to_calendar, coverage=coverage,
                                                from_unique=column.name in tkeys.unique_columns)

    ranked = sorted(candidates.values(), key=lambda f: (-f.declared, -f.name_score, -f.coverage, f.ident))
    tested = ranked[:max_tests]
    placeholder_of = {key: cal.placeholders for key, cal in calendars.items()}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(lambda f: _test(warehouse, inventory, f, profiles, placeholder_of.get(f.to_table, [])), tested))

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
        if f.match_rate >= VERIFIED:
            f.trust = "verified"
        elif f.declared and f.match_rate >= PROPOSED:
            f.trust = "declared"   # usable, as the database says; the unmatched rows are reported
        elif f.match_rate >= PROPOSED or ((f.declared or f.name_score >= 0.95) and f.match_rate > 0):
            # A join its names (or the database) vouch for stays usable when most keys
            # find nothing: the answer reports the unmatched rows rather than losing
            # the grouping altogether.
            f.trust = "proposed"
        else:
            f.trust = "rejected"

    # One target per column: the best-supported one. A tie between value-only
    # candidates is not decided here.
    by_column: dict[tuple[str, str], list[JoinFinding]] = {}
    for f in tested:
        if f.trust != "rejected":
            by_column.setdefault((f.from_table, f.from_column), []).append(f)
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


def names_norm(name: str) -> str:
    return name.casefold()


def _names_its_table(column: str, table: str) -> bool:
    core, own = names.core_column(column), names.core_table(table)
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
            if not words and len(group) > 1 and not names.opaque(f.from_column):
                words = names.core_column(f.from_column)
            if not words:
                continue
            text = " ".join(names.EXPANSIONS.get(w, w) for w in words)
            if f.to_calendar:
                noun = "date"
            else:
                target = inventory.tables[f.to_table].name
                noun = "" if names.opaque(target) else " ".join(
                    names.EXPANSIONS.get(w, w) for w in names.core_table(target))
            if noun and not text.endswith(noun):
                text = f"{text} {noun}"
            f.role = text[:1].upper() + text[1:]
