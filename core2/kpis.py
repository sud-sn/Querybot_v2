"""Business metrics each kind of business is run by, proposed to the admin where the workspace's data holds them.

Learn makes a sum or a count of what each table holds. A business is read by its ratios and shares: a bank by its
approval rate, its delinquent loans and its loan-to-value; a pharmacy by its margin and its rejected claims; a shop
by its average order and its returns; a ledger by its profit margin. Each is written here once, as a formula over
the fields its words name, in the metric editor's own language (core2/model/authoring.py): what an admin is shown
is what they would have written, and they read, check and edit it the same way.

Nothing here becomes a metric by itself. A metric the data holds is checked on the data and proposed to the admin
as a new metric on the Requests page (core2/kpi_proposals.py); accepted, it is an admin's metric everyone's answers
use, kept with the business metric it was (``Measure.kpi``), so an executive dashboard shows it in its area.

Fields are found by the words of their names, never by their values. A share of rows by a status is the share of
the values whose words say it ("Approved"; "REJ" for rejected), read from the column's profile where its values may
be read, and never from a person's data. Shares of notable statuses (cancelled orders, failed tests) are proposed
for any business.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from core2.bootstrap import names
from core2.model.schema import Column, DateRole, Measure, SemanticModel, Table, value_names

May = Callable[[str], bool]
log = logging.getLogger("querybot.core2")


def _all(table: str) -> bool:
    return True


class Missing(Exception):
    """The data has no field or table the metric needs."""


def _words(text: str) -> set[str]:
    out: set[str] = set()
    for w in names.tokens(text or ""):
        for part in names.EXPANSIONS.get(w, w).split():
            out.add(names.singular(part))
    return out


# What names an id rather than the thing it is the id of ("Order number": the order).
_ID_WORDS = frozenset({"id", "number", "no", "num", "nbr", "key", "code", "ref", "reference"})
# A value holding these would be read as part of the formula around it.
_FORMULA_MARKS = frozenset("[]×÷−")

# Codes statuses are stored as, read as the words they stand for.
_CODES = {"rej": "rejected", "rjt": "rejected", "rejd": "rejected", "rjct": "rejected", "canc": "cancelled",
          "cxl": "cancelled", "cncl": "cancelled", "cnl": "cancelled", "decl": "declined", "dcl": "declined",
          "den": "denied", "appr": "approved", "aprv": "approved", "apvd": "approved", "rtn": "returned",
          "fail": "failed", "flr": "failed", "pend": "pending", "dlq": "delinquent", "y": "yes", "t": "true"}


def _value_words(value: str, shown: str = "") -> list[str]:
    """A stored value's words, its codes read ("REJ": rejected), with the name readers see it by."""
    out = []
    for text in (value, shown):
        for w in names.tokens(text):
            out.extend(_CODES.get(w, w).split())
    return out


def _says(words: list[str], stems: set[str]) -> bool:
    """Every stem begins one of the words ("approv" in "Approved", "past" and "due" in "30-89 days past due")."""
    return all(any(w.startswith(s) for w in words) for s in stems)


@dataclass(frozen=True)
class Formula:
    text: str                       # in the metric editor's language
    date: str | None = None         # the date role it is counted by, when not its table's own


@dataclass(frozen=True)
class Kpi:
    key: str                        # "bank.approval_rate"
    name: tuple[str, str]           # in English, in French
    says: tuple[str, str]           # what it counts, as its description
    area: tuple[str, str]           # the area of the business it is read in: its dashboard tab
    rank: int                       # 1: a headline of its business; 2: an area's lead; 3: supporting
    fmt: str                        # percent | currency | number
    better: str                     # higher | lower | neither
    kinds: frozenset[str]           # the kinds of business it is written for (core2/recipes.py)
    write: Callable[[Bind], Formula]
    share: bool = False             # a share of rows: always between 0 and 100


@dataclass
class Proposal:
    """A business metric the data holds: its formula, ready to be read, checked and proposed."""
    key: str
    name: str
    says: str
    area: str
    rank: int
    fmt: str
    better: str
    share: bool
    formula: Formula


class Bind:
    """The workspace's tables and fields, found by the words of their names."""

    def __init__(self, model: SemanticModel, may: May = _all):
        self.model, self.may = model, may
        self.used: set[str] = set()          # the status columns a metric already reads
        # Fields Learn adds up: a line's amount, never a unit price it averages.
        self.summed = {m.expr.column for m in model.measures.values()
                       if getattr(m.expr, "agg", "") == "sum" and getattr(m.expr, "column", None)}
        # Amounts kept in several currencies or units: a ratio of them would mix them (core2/recipes._mixed).
        self.mixed = {q.object for q in model.quality if q.kind == "unit_mix"} | {
            m.expr.column for m in model.measures.values()
            if m.unit_column and not m.unit and getattr(m.expr, "column", None)}

    # ── finding things ──────────────────────────────────────────────────────

    def table(self, *alternatives: set[str], avoid: set[str] = frozenset(), facts: bool = True) -> Table:
        """The table named with all the words of one alternative and none to avoid: the fullest match, the one
        with fewest other words, rows that happen (a fact) before a list, the busiest."""
        found = self.tables(*alternatives, avoid=avoid, facts=facts)
        if not found:
            raise Missing("table")
        return found[0]

    def tables(self, *alternatives: set[str], avoid: set[str] = frozenset(), facts: bool = True) -> list[Table]:
        """Every table :meth:`table` could name, the best first."""
        found = []
        for t in self.model.tables.values():
            if t.hidden or t.kind in ("calendar", "bridge") or not self.may(t.key) \
                    or (facts and t.kind not in ("fact", "snapshot")):
                continue
            words = _words(t.business_name or t.name)
            if words & avoid:
                continue
            hit = max((len(a) for a in alternatives if a <= words), default=0)
            if hit:
                found.append((hit, -len(words), t.kind == "fact", t.row_count or 0, t.key, t))
        return [f[5] for f in sorted(found, key=lambda f: f[:5], reverse=True)]

    def _fields(self, table: Table) -> list[Column]:
        return [c for c in self.model.table_columns(table.key) if not c.hidden and c.personal == "none"
                and not c.record_number and c.sensitivity == "none" and not c.parts]

    def number(self, table: Table, *alternatives: set[str], avoid: set[str] = frozenset()) -> Column:
        """A number field of the table named with all the words of one alternative and none to avoid."""
        from core2.model.authoring import _kind

        found = []
        for c in self._fields(table):
            if _kind(c) != "number" or c.key in self.mixed:
                continue
            words = _words(c.business_name or c.name)
            if words & avoid:
                continue
            hit = max((len(a) for a in alternatives if a <= words), default=0)
            if hit:
                found.append((hit, c.key in self.summed, -len(words), c.role == "measure", c.key, c))
        if not found:
            raise Missing("field")
        return max(found, key=lambda f: f[:5])[5]

    def maybe(self, table: Table, *alternatives: set[str], avoid: set[str] = frozenset()) -> Column | None:
        try:
            return self.number(table, *alternatives, avoid=avoid)
        except Missing:
            return None

    def status(self, table: Table, *alternatives: set[str], avoid: set[str] = frozenset()) -> tuple[Column, list[str]]:
        """A column of few values, some of which say one alternative's stems ("approv": Approved, APPR) and none
        a stem to avoid ("pend": Pending approval): the column and those values, as stored. Only a column whose
        values may be read, never a person's."""
        found = []
        for c in self._fields(table):
            p = c.profile
            if not c.values_allowed or p is None or not p.top or (p.distinct or len(p.top)) > 30 \
                    or c.role not in ("status", "attribute", "flag") or c.data_type not in ("text", "boolean"):
                continue
            shown = value_names(c)
            stored = [str(v.value) for v in p.top if v.value is not None]
            hits = []
            for value in stored:
                words = _value_words(value, shown.get(value, ""))
                if any(w.startswith(s) for w in words for s in avoid):
                    continue
                if any(_says(words, alt) for alt in alternatives):
                    hits.append(value)
            if any(_FORMULA_MARKS & set(h) for h in hits):
                continue              # "Approved [Amount]" would be read as a field inside the formula
            if hits and len(hits) < len(stored):
                named = _words(c.business_name or c.name)
                found.append((c.role == "status", bool(named & {"status", "decision", "result", "outcome", "state"}),
                              -len(hits), c.key, c, hits))
        if not found:
            raise Missing("status")
        best = max(found, key=lambda f: f[:4])
        return best[4], best[5]

    def values(self, column: Column, *alternatives: set[str], avoid: set[str] = frozenset()) -> list[str]:
        """The values of a column (as stored) whose words say one alternative's stems and none to avoid."""
        p, shown = column.profile, value_names(column)
        out = []
        for v in (p.top or []) if p is not None else []:
            if v.value is None or _FORMULA_MARKS & set(str(v.value)):
                continue
            words = _value_words(str(v.value), shown.get(str(v.value), ""))
            if not any(w.startswith(s) for w in words for s in avoid) and any(_says(words, a) for a in alternatives):
                out.append(str(v.value))
        return out

    def flag(self, table: Table, *alternatives: set[str]) -> tuple[Column, list[str]]:
        """A yes/no field named with one alternative's words, and the values that mean yes."""
        for c in self._fields(table):
            p = c.profile
            if c.role != "flag" and c.data_type != "boolean" or p is None or not p.top:
                continue
            words = _words(c.business_name or c.name)
            if not any(a <= words for a in alternatives):
                continue
            yes = [str(v.value) for v in p.top if v.value is not None
                   and str(v.value).strip().casefold() in ("true", "1", "y", "yes", "t")]
            if yes:
                return c, yes
        raise Missing("flag")

    def flag_named(self, table: Table, stem: str) -> tuple[Column, list[str]]:
        """A yes/no field one of whose words begins ``stem`` ("Cancellation flag", "Is returned"), and the values
        that mean yes."""
        for c in self._fields(table):
            p = c.profile
            if c.role != "flag" and c.data_type != "boolean" or p is None or not p.top:
                continue
            if not any(w.startswith(stem) for w in names.tokens(c.business_name or c.name)):
                continue
            yes = [str(v.value) for v in p.top if v.value is not None
                   and str(v.value).strip().casefold() in ("true", "1", "y", "yes", "t")]
            if yes and c.key not in self.used:
                return c, yes
        raise Missing("flag")

    def date(self, table: Table, *alternatives: set[str]) -> str | None:
        """The table's date named with one alternative's words, or None (its own date counts it)."""
        for role in self.model.date_roles_of(table.key):
            if role.kind != "audit" and any(a <= _words(role.name) for a in alternatives):
                return role.key
        return None

    def metric(self, *alternatives: set[str], avoid: set[str] = frozenset(), table: Table | None = None) -> Measure:
        """A metric of the model named with exactly one alternative's words (Revenue, Profit, Expenses), of the
        table's rows when one is named."""
        from core2.recipes import _mixed

        found = []
        for m in self.model.measures.values():
            if m.hidden or not self.may(m.table) or (table is not None and m.table != table.key) \
                    or _mixed(self.model, m) or self._reads_mixed(m):
                continue
            words = _words(m.business_name)
            if words & avoid:
                continue
            if any(a == words for a in alternatives):
                found.append((m.kind == "metric", -len(m.business_name), m.key, m))
        if not found:
            raise Missing("metric")
        return max(found, key=lambda f: f[:3])[3]

    def _reads_mixed(self, m: Measure, depth: int = 0) -> bool:
        """A metric made of amounts in several currencies or units, however deep (Profit = Revenue - Expenses)."""
        if depth > 8:
            return True

        def mixed(e) -> bool:
            if getattr(e, "column", None) in self.mixed or set(getattr(e, "columns", None) or []) & self.mixed:
                return True
            ref = getattr(e, "measure", None)
            if ref:
                return ref not in self.model.measures or self._reads_mixed(self.model.measures[ref], depth + 1)
            return any(mixed(a) for a in getattr(e, "args", None) or [])

        return mixed(m.expr)

    def distinct_of(self, table: Table, *alternatives: set[str]) -> str:
        """The field that counts the documents or members of the table's rows (an order number, a customer id), as
        written in a formula: one Learn counts distinct values of, or a key or link to them; on the table, else on
        the table its rows belong to (an invoice line's invoice)."""
        for m in self.model.measures.values():
            column = self.model.columns.get(getattr(m.expr, "column", None) or "")
            if m.table == table.key and not m.hidden and getattr(m.expr, "agg", "") == "count_distinct" \
                    and column is not None and column.table == table.key and self._countable(column) \
                    and any(a <= _words(m.business_name) for a in alternatives):
                return self.ref(column)
        parents = [self.model.tables[j.to_table] for j in self.model.joins_from(table.key)
                   if j.trust != "rejected" and not j.to_calendar and j.cardinality in ("many_to_one", "one_to_one")
                   and j.to_table in self.model.tables and j.to_table != table.key and self.may(j.to_table)]
        for owner in [table, *parents]:
            columns = self.model.table_columns(owner.key)
            keys = [c for c in columns if c.role == "key"]
            dated = {f for j in self.model.joins_from(owner.key) if j.to_calendar for f in j.from_columns}
            found = []
            for c in columns:
                if c.role not in ("foreign_key", "identifier", "key") or not self._countable(c) or c.key in dated:
                    continue
                if c.role == "key" and owner is table and len(keys) < 2:
                    continue          # the line's own id, one per row: its lines, not its documents
                words = _words(c.business_name or c.name)
                if not any(a <= words for a in alternatives):
                    continue
                bare = words - _ID_WORDS
                found.append((any(a == bare for a in alternatives), c.role in ("foreign_key", "identifier"),
                              -len(bare), c.key, c))
            if found:
                return self.ref(max(found, key=lambda f: f[:4])[4])
        raise Missing("documents")

    @staticmethod
    def _countable(column: Column) -> bool:
        """A column a metric may count the values of: never hidden, a person's data or a record about one."""
        return not column.hidden and column.personal == "none" and not column.record_number \
            and column.sensitivity == "none"

    # ── writing formulas ────────────────────────────────────────────────────

    def ref(self, column: Column) -> str:
        from core2.model.authoring import field_ref

        return f"[{field_ref(self.model, column.key)}]"

    def rows(self, table: Table) -> str:
        from core2.model.authoring import table_name

        return f"[{table_name(self.model, table.key)}]"

    def named(self, m: Measure) -> str:
        """A metric as a formula reads it: the field it adds up, or its name where no other metric has it."""
        column = getattr(m.expr, "column", None)
        if getattr(m.expr, "agg", "") == "sum" and column in self.model.columns and not m.filters \
                and not getattr(m.expr, "filters", None):
            return f"SUM({self.ref(self.model.columns[column])})"
        same = [o for o in self.model.measures.values() if o.business_name.casefold() == m.business_name.casefold()]
        if len(same) != 1 or "[" in m.business_name or "]" in m.business_name:
            raise Missing("metric name")
        return f"[{m.business_name}]"

    def is_one_of(self, column: Column, values: list[str]) -> str:
        """The condition that a row's value is one of ``values``, as stored."""
        self.used.add(column.key)
        if column.data_type == "boolean":
            return f"{self.ref(column)} = TRUE"
        if column.data_type == "integer":
            return f"{self.ref(column)} IN ({', '.join(str(int(v)) for v in values)})"
        quoted = [f"""'{v.replace("'", "''")}'""" for v in values]
        return f"{self.ref(column)} = {quoted[0]}" if len(quoted) == 1 else f"{self.ref(column)} IN ({', '.join(quoted)})"

    @staticmethod
    def share(rows: str, condition: str) -> str:
        return f"SUM(CASE WHEN {condition} THEN 1 ELSE 0 END) / COUNT({rows}) * 100"


# ── the metrics of each kind of business ─────────────────────────────────────

_LOAN_AVOID = {"application", "payment", "repayment", "officer", "target", "rate", "product", "type"}
_PRINCIPAL = ({"principal", "amount"}, {"principal"}, {"loan", "amount"}, {"funded", "amount"})
_PRINCIPAL_AVOID = {"outstanding", "paid", "remaining", "balance", "requested"}


def _bank_approval(b: Bind) -> Formula:
    """Approved over decided: an application still pending (or withdrawn) is no refusal; counting it would show
    the last months, whose applications are still being decided, as a collapse."""
    t = b.table({"loan", "application"}, {"credit", "application"}, {"application"})
    column, approved = b.status(t, {"approv"}, {"accept"}, {"grant"}, avoid={"pend", "not", "await", "pre", "un"})
    declined = b.values(column, {"declin"}, {"reject"}, {"denied"}, {"refus"}, {"turned", "down"},
                        avoid={"not", "un", "pend"})
    if not declined:
        raise Missing("decided")
    decided = b.is_one_of(column, [*approved, *declined])
    return Formula(f"SUM(CASE WHEN {b.is_one_of(column, approved)} THEN 1 ELSE 0 END) / "
                   f"SUM(CASE WHEN {decided} THEN 1 ELSE 0 END) * 100")


def _bank_loan_size(b: Bind) -> Formula:
    t = b.table({"loan"}, {"mortgage"}, avoid=_LOAN_AVOID)
    principal = b.number(t, *_PRINCIPAL, avoid=_PRINCIPAL_AVOID)
    return Formula(f"SUM({b.ref(principal)}) / COUNT({b.rows(t)})")


def _bank_ltv(b: Bind) -> Formula:
    t = b.table({"loan"}, {"mortgage"}, avoid=_LOAN_AVOID)
    principal = b.number(t, *_PRINCIPAL, avoid=_PRINCIPAL_AVOID)
    collateral = b.number(t, {"collateral"}, {"property", "value"}, {"appraised", "value"})
    return Formula(f"SUM({b.ref(principal)}) / SUM({b.ref(collateral)}) * 100")


def _bank_delinquent(b: Bind) -> Formula:
    t = b.table({"loan"}, {"mortgage"}, avoid=_LOAN_AVOID)
    column, values = b.status(t, {"past", "due"}, {"delinquen"}, {"overdue"}, {"arrear"}, {"default"},
                              avoid={"not", "non", "current"})
    return Formula(b.share(b.rows(t), b.is_one_of(column, values)))


def _bank_charge_off(b: Bind) -> Formula:
    t = b.table({"loan"}, {"mortgage"}, avoid=_LOAN_AVOID)
    principal = b.number(t, *_PRINCIPAL, avoid=_PRINCIPAL_AVOID)
    written = b.number(t, {"charge", "off"}, {"chargeoff"}, {"write", "off"}, {"writeoff"})
    return Formula(f"SUM({b.ref(written)}) / SUM({b.ref(principal)}) * 100")


def _payments(b: Bind) -> Table:
    return b.table({"loan", "payment"}, {"repayment"}, {"installment"}, {"loan", "instalment"},
                   avoid={"card", "target"})


def _bank_late(b: Bind) -> Formula:
    t = _payments(b)
    days = b.number(t, {"days", "past", "due"}, {"dpd"}, {"day", "past", "due"}, {"days", "late"}, {"day", "late"},
                    {"days", "overdue"}, {"day", "overdue"}, {"days", "arrear"}, {"day", "arrear"})
    return Formula(b.share(b.rows(t), f"{b.ref(days)} > 30"), date=b.date(t, {"due"}))


def _bank_collection(b: Bind) -> Formula:
    t = _payments(b)
    paid = b.number(t, {"amount", "paid"}, {"paid", "amount"}, {"payment", "amount"},
                    avoid={"principal", "interest", "fee", "due"})
    due = b.number(t, {"amount", "due"}, {"due", "amount"}, {"scheduled", "amount"}, {"installment", "amount"},
                   avoid={"paid"})
    return Formula(f"SUM({b.ref(paid)}) / SUM({b.ref(due)}) * 100", date=b.date(t, {"due"}))


def _cards(b: Bind) -> Table:
    return b.table({"card"}, avoid={"transaction", "payment", "statement"})


def _bank_credit_limit(b: Bind) -> Formula:
    t = _cards(b)
    limit = b.number(t, {"credit", "limit"}, {"limit"})
    return Formula(f"SUM({b.ref(limit)}) / COUNT({b.rows(t)})")


def _bank_active_cards(b: Bind) -> Formula:
    t = _cards(b)
    column, values = b.status(t, {"active"}, {"open"}, avoid={"in", "not", "un"})
    return Formula(b.share(b.rows(t), b.is_one_of(column, values)))


def _bank_compensation(b: Bind) -> Formula:
    t = b.table({"complaint"}, {"dispute"})
    paid = b.number(t, {"compensation"}, {"goodwill"}, {"redress"}, {"refund"})
    return Formula(f"SUM({b.ref(paid)}) / COUNT({b.rows(t)})")


def _bank_transaction_value(b: Bind) -> Formula:
    t = b.table({"transaction"}, avoid={"type", "code", "fee"})
    amount = b.number(t, {"amount"}, {"value"}, avoid={"fee", "balance", "tax"})
    return Formula(f"SUM({b.ref(amount)}) / COUNT({b.rows(t)})")


# Pharmacy
_FILLS_AVOID = {"status", "type", "code", "reason"}
_PRICE = ({"price", "charged"}, {"price"}, {"revenue"}, {"sale", "amount"}, {"charged", "amount"}, {"charge"})
_PRICE_AVOID = {"cash", "list", "unit", "cost", "copay", "patient"}
_COST = ({"ingredient", "cost"}, {"drug", "cost"}, {"acquisition", "cost"}, {"cost"})
_COST_AVOID = {"labor", "labour", "shipping", "unit", "list", "price", "per", "rate"}


def _fills(b: Bind) -> Table:
    return b.table({"fill"}, {"dispense"}, {"dispensing"}, avoid=_FILLS_AVOID)


def _pharmacy_costs(b: Bind, t: Table) -> str:
    cost = b.number(t, *_COST, avoid=_COST_AVOID)
    labor = b.maybe(t, {"labor", "cost"}, {"labour", "cost"})
    return f"SUM({b.ref(cost)})" + (f" - SUM({b.ref(labor)})" if labor is not None else "")


def _pharmacy_margin_pct(b: Bind) -> Formula:
    t = _fills(b)
    price = b.number(t, *_PRICE, avoid=_PRICE_AVOID)
    return Formula(f"(SUM({b.ref(price)}) - {_pharmacy_costs(b, t)}) / SUM({b.ref(price)}) * 100")


def _pharmacy_margin(b: Bind) -> Formula:
    t = _fills(b)
    price = b.number(t, *_PRICE, avoid=_PRICE_AVOID)
    return Formula(f"SUM({b.ref(price)}) - {_pharmacy_costs(b, t)}")


def _pharmacy_price_per_fill(b: Bind) -> Formula:
    t = _fills(b)
    price = b.number(t, *_PRICE, avoid=_PRICE_AVOID)
    return Formula(f"SUM({b.ref(price)}) / COUNT({b.rows(t)})")


def _pharmacy_refills(b: Bind) -> Formula:
    t = _fills(b)
    number = b.number(t, {"fill", "number"}, {"refill", "number"}, {"fill", "sequence"})
    return Formula(b.share(b.rows(t), f"{b.ref(number)} > 0"))


def _claims(b: Bind) -> Table:
    return b.table({"claim"}, avoid={"status", "type", "code", "reason", "line"})


def _pharmacy_claim_rejections(b: Bind) -> Formula:
    t = _claims(b)
    column, values = b.status(t, {"reject"}, {"denied"}, {"declin"}, avoid={"not", "un"})
    return Formula(b.share(b.rows(t), b.is_one_of(column, values)))


def _pharmacy_paid_to_billed(b: Bind) -> Formula:
    t = _claims(b)
    billed = b.number(t, {"billed"}, {"billed", "amount"}, {"submitted", "amount"}, {"claimed", "amount"})
    paid = b.number(t, {"paid", "amount"}, {"plan", "paid"}, {"payer", "paid"}, {"reimbursed"},
                    avoid={"patient", "copay", "billed"})
    return Formula(f"SUM({b.ref(paid)}) / SUM({b.ref(billed)}) * 100")


def _pharmacy_batch_rejections(b: Bind) -> Formula:
    t = b.table({"batch"}, {"compounding"}, avoid={"lot", "usage", "ingredient", "status"})
    column, values = b.status(t, {"reject"}, {"fail"}, {"recall"}, avoid={"not", "un"})
    return Formula(b.share(b.rows(t), b.is_one_of(column, values)))


def _pharmacy_quality_failures(b: Bind) -> Formula:
    t = b.table({"quality", "test"}, {"quality", "check"}, {"qc"}, {"test"}, avoid={"type"})
    column, values = b.status(t, {"fail"}, {"out", "of", "spec"}, {"oos"}, avoid={"not", "non"})
    return Formula(b.share(b.rows(t), b.is_one_of(column, values)))


def _shipments(b: Bind) -> Table:
    return b.table({"shipment"}, {"delivery"}, {"shipping"}, avoid={"status", "type", "method"})


def _pharmacy_excursions(b: Bind) -> Formula:
    t = _shipments(b)
    column, values = b.flag(t, {"temperature", "excursion"}, {"excursion"}, {"temperature", "breach"})
    return Formula(b.share(b.rows(t), b.is_one_of(column, values)))


def _pharmacy_count_accuracy(b: Bind) -> Formula:
    t = b.table({"inventory", "count"}, {"cycle", "count"}, {"stock", "count"}, {"stocktake"})
    counted = b.number(t, {"counted"}, {"counted", "quantity"}, {"physical"})
    system = b.number(t, {"system"}, {"system", "quantity"}, {"expected"}, {"book"}, avoid={"counted"})
    return Formula(f"SUM({b.ref(counted)}) / SUM({b.ref(system)}) * 100")


def _pharmacy_shipping_cost(b: Bind) -> Formula:
    t = _shipments(b)
    cost = b.number(t, {"shipping", "cost"}, {"freight"}, {"postage"}, {"cost"})
    return Formula(f"SUM({b.ref(cost)}) / COUNT({b.rows(t)})")


# Sales and retail
_LINES = ({"order", "line"}, {"sales", "line"}, {"sale", "line"}, {"invoice", "line"}, {"order", "item"},
          {"line", "item"}, {"lineitem"}, {"order", "detail"}, {"sales", "order"}, {"sale", "order"}, {"order"},
          {"sale"}, {"invoice"})
_LINES_AVOID = {"target", "return", "refund", "status", "type", "quota", "plan", "budget", "forecast"}
_REVENUE = ({"net", "amount"}, {"net", "sale"}, {"net", "revenue"}, {"net", "paid"}, {"revenue"}, {"sale", "amount"},
            {"line", "total"}, {"extended", "amount"}, {"line", "amount"}, {"invoice", "total"}, {"order", "total"},
            {"amount"}, {"total"})
_REVENUE_AVOID = {"discount", "tax", "cost", "freight", "shipping", "refund", "unit", "list", "target", "price",
                  "percent", "margin", "commission", "coupon", "rebate", "fee", "deposit", "credit", "profit",
                  "wholesale", "budget", "quota", "forecast"}
_ORDERS = ({"order"}, {"invoice"}, {"receipt"}, {"ticket"}, {"basket"}, {"transaction"})


def _sales(b: Bind) -> tuple[Table, str]:
    """The rows of sales and their net sales, written as a formula: a field added up, or the metric Learn made of
    them when no field holds it (quantity times unit price); on order lines, else on the orders' own total."""
    for t in b.tables(*_LINES, avoid=_LINES_AVOID):
        revenue = b.maybe(t, *_REVENUE, avoid=_REVENUE_AVOID)
        if revenue is not None:
            return t, f"SUM({b.ref(revenue)})"
        try:
            return t, b.named(b.metric(*_REVENUE, avoid=_REVENUE_AVOID, table=t))
        except Missing:
            continue
    raise Missing("revenue")


def _sales_aov(b: Bind) -> Formula:
    t, revenue = _sales(b)
    return Formula(f"{revenue} / COUNT(DISTINCT {b.distinct_of(t, *_ORDERS)})")


def _sales_cost(b: Bind, t: Table) -> Column:
    return b.number(t, {"cost", "amount"}, {"cost"}, {"cogs"}, {"cost", "of", "goods"},
                    avoid={"shipping", "freight", "unit", "labor", "labour", "standard", "price", "per", "rate"})


def _sales_margin_pct(b: Bind) -> Formula:
    t, revenue = _sales(b)
    cost = _sales_cost(b, t)
    return Formula(f"({revenue} - SUM({b.ref(cost)})) / {revenue} * 100")


def _sales_margin(b: Bind) -> Formula:
    t, revenue = _sales(b)
    return Formula(f"{revenue} - SUM({b.ref(_sales_cost(b, t))})")


def _money(column: Column) -> bool:
    """Not a fraction: a discount of 0.05 is a rate, never an amount."""
    p = column.profile
    return p is None or p.max_num is None or p.max_num > 1


def _sales_discount_rate(b: Bind) -> Formula:
    t, _ = _sales(b)
    try:
        discount = b.named(b.metric({"discount", "amount"}, {"discount", "value"}, table=t))
    except Missing:
        column = b.number(t, {"discount", "amount"}, {"discount"}, avoid={"percent", "rate", "pct"})
        if not _money(column):
            raise Missing("discount amount") from None
        discount = f"SUM({b.ref(column)})"
    gross = b.number(t, {"gross", "amount"}, {"gross", "sale"}, {"list", "amount"}, {"gross"},
                     {"extended", "list", "price"}, {"extended", "price"})
    return Formula(f"{discount} / SUM({b.ref(gross)}) * 100")


def _sales_quantity(b: Bind, t: Table) -> Column:
    return b.number(t, {"quantity"}, {"qty"}, {"units"}, {"unit", "sold"},
                    avoid={"return", "price", "cost", "measure", "backorder", "allocated"})


def _sales_units_per_order(b: Bind) -> Formula:
    t, _ = _sales(b)
    return Formula(f"SUM({b.ref(_sales_quantity(b, t))}) / COUNT(DISTINCT {b.distinct_of(t, *_ORDERS)})")


def _sales_selling_price(b: Bind) -> Formula:
    t, revenue = _sales(b)
    return Formula(f"{revenue} / SUM({b.ref(_sales_quantity(b, t))})")


def _sales_return_rate(b: Bind) -> Formula:
    t, revenue = _sales(b)
    returns = b.table({"return"}, {"refund"}, avoid={"reason", "type", "code", "status"})
    refund = b.number(returns, {"refund", "amount"}, {"refund"}, {"return", "amount"}, {"credit", "amount"},
                      avoid={"quantity", "tax"})
    return Formula(f"SUM({b.ref(refund)}) / {revenue} * 100")


def _sales_per_customer(b: Bind) -> Formula:
    t, revenue = _sales(b)
    return Formula(f"{revenue} / COUNT(DISTINCT {b.distinct_of(t, {'customer'}, {'client'})})")


# Finance
def _finance_margin(b: Bind) -> Formula:
    return Formula(f"{b.named(b.metric({'profit'}, {'net', 'profit'}, {'net', 'income'}))} / "
                   f"{b.named(b.metric({'revenue'}, {'sale'}, {'income'}, avoid={'net'}))} * 100")


def _finance_expense_ratio(b: Bind) -> Formula:
    return Formula(f"{b.named(b.metric({'expense'}, {'operating', 'expense'}, {'cost'}))} / "
                   f"{b.named(b.metric({'revenue'}, {'sale'}, {'income'}, avoid={'net'}))} * 100")


_BANK, _PHARMACY, _SALES, _FINANCE = (frozenset({"bank"}), frozenset({"pharmacy"}), frozenset({"sales"}),
                                      frozenset({"finance"}))

PACKS: list[Kpi] = [
    # A bank
    Kpi("bank.approval_rate", ("Approval rate", "Taux d'approbation"),
        ("Loan applications approved, as a share of those decided (approved or declined).",
         "Demandes de prêt approuvées, en part des demandes tranchées (approuvées ou refusées)."),
        ("Lending", "Prêts"), 1, "percent", "higher", _BANK, _bank_approval, share=True),
    Kpi("bank.delinquent_loans", ("Delinquent loans", "Prêts en retard"),
        ("Loans past due or in arrears, as a share of all loans.",
         "Prêts en retard de paiement, en part de tous les prêts."),
        ("Risk", "Risque"), 1, "percent", "lower", _BANK, _bank_delinquent, share=True),
    Kpi("bank.average_loan_size", ("Average loan size", "Montant moyen des prêts"),
        ("Principal lent, per loan.", "Capital prêté, par prêt."),
        ("Lending", "Prêts"), 2, "currency", "neither", _BANK, _bank_loan_size),
    Kpi("bank.loan_to_value", ("Loan-to-value", "Quotité de financement"),
        ("Principal lent as a share of the collateral's value.", "Capital prêté en part de la valeur des garanties."),
        ("Risk", "Risque"), 2, "percent", "lower", _BANK, _bank_ltv),
    Kpi("bank.charge_off_rate", ("Charge-off rate", "Taux de pertes sur prêts"),
        ("Amounts charged off as a share of the principal lent.", "Montants passés en perte, en part du capital prêté."),
        ("Risk", "Risque"), 2, "percent", "lower", _BANK, _bank_charge_off),
    Kpi("bank.late_payments", ("Late payment rate", "Taux de paiements en retard"),
        ("Loan payments more than 30 days past due, as a share of all payments due.",
         "Échéances payées avec plus de 30 jours de retard, en part de toutes les échéances."),
        ("Repayments", "Remboursements"), 2, "percent", "lower", _BANK, _bank_late, share=True),
    Kpi("bank.collection_rate", ("Collection rate", "Taux de recouvrement"),
        ("Amounts paid as a share of the amounts due.", "Montants payés en part des montants dus."),
        ("Repayments", "Remboursements"), 2, "percent", "higher", _BANK, _bank_collection),
    Kpi("bank.average_credit_limit", ("Average credit limit", "Plafond moyen"),
        ("Credit limit per card.", "Plafond de crédit par carte."),
        ("Cards", "Cartes"), 3, "currency", "neither", _BANK, _bank_credit_limit),
    Kpi("bank.active_cards", ("Active cards", "Cartes actives"),
        ("Cards active, as a share of all cards.", "Cartes actives, en part de toutes les cartes."),
        ("Cards", "Cartes"), 3, "percent", "higher", _BANK, _bank_active_cards, share=True),
    Kpi("bank.compensation_per_complaint", ("Compensation per complaint", "Indemnisation par réclamation"),
        ("Compensation paid, per complaint.", "Indemnisations versées, par réclamation."),
        ("Service", "Service"), 3, "currency", "lower", _BANK, _bank_compensation),
    Kpi("bank.average_transaction", ("Average transaction", "Transaction moyenne"),
        ("Amount per transaction.", "Montant par transaction."),
        ("Transactions", "Transactions"), 3, "currency", "neither", _BANK, _bank_transaction_value),
    # A pharmacy
    Kpi("pharmacy.gross_margin_pct", ("Gross margin %", "Marge brute %"),
        ("What fills were charged, less what they cost, as a share of what they were charged.",
         "Prix facturé des délivrances, moins leur coût, en part du prix facturé."),
        ("Profitability", "Rentabilité"), 1, "percent", "higher", _PHARMACY, _pharmacy_margin_pct),
    Kpi("pharmacy.gross_margin", ("Gross margin", "Marge brute"),
        ("What fills were charged, less what they cost.", "Prix facturé des délivrances, moins leur coût."),
        ("Profitability", "Rentabilité"), 2, "currency", "higher", _PHARMACY, _pharmacy_margin),
    Kpi("pharmacy.claim_rejection_rate", ("Claim rejection rate", "Taux de rejet des demandes"),
        ("Claims rejected or denied, as a share of all claims.",
         "Demandes de remboursement rejetées, en part de toutes les demandes."),
        ("Claims", "Remboursements"), 1, "percent", "lower", _PHARMACY, _pharmacy_claim_rejections, share=True),
    Kpi("pharmacy.paid_to_billed", ("Paid to billed", "Payé sur facturé"),
        ("What payers paid as a share of what was billed.", "Montant payé par les payeurs en part du montant facturé."),
        ("Claims", "Remboursements"), 2, "percent", "higher", _PHARMACY, _pharmacy_paid_to_billed),
    Kpi("pharmacy.price_per_fill", ("Average price per fill", "Prix moyen par délivrance"),
        ("What was charged, per fill.", "Prix facturé, par délivrance."),
        ("Fills", "Délivrances"), 2, "currency", "higher", _PHARMACY, _pharmacy_price_per_fill),
    Kpi("pharmacy.refill_share", ("Refill share", "Part des renouvellements"),
        ("Fills that are refills, as a share of all fills.", "Délivrances qui sont des renouvellements, en part du total."),
        ("Fills", "Délivrances"), 3, "percent", "higher", _PHARMACY, _pharmacy_refills, share=True),
    Kpi("pharmacy.batch_rejection_rate", ("Batch rejection rate", "Taux de lots rejetés"),
        ("Batches rejected or failed, as a share of all batches.", "Lots rejetés, en part de tous les lots."),
        ("Quality", "Qualité"), 2, "percent", "lower", _PHARMACY, _pharmacy_batch_rejections, share=True),
    Kpi("pharmacy.quality_failure_rate", ("Quality test failure rate", "Taux d'échec aux contrôles"),
        ("Quality tests failed, as a share of all tests.", "Contrôles qualité échoués, en part de tous les contrôles."),
        ("Quality", "Qualité"), 2, "percent", "lower", _PHARMACY, _pharmacy_quality_failures, share=True),
    Kpi("pharmacy.temperature_excursions", ("Temperature excursion rate", "Taux d'écarts de température"),
        ("Shipments with a temperature excursion, as a share of all shipments.",
         "Expéditions avec un écart de température, en part de toutes les expéditions."),
        ("Shipping", "Expéditions"), 3, "percent", "lower", _PHARMACY, _pharmacy_excursions, share=True),
    Kpi("pharmacy.count_accuracy", ("Inventory count accuracy", "Exactitude des inventaires"),
        ("Quantity counted as a share of the quantity the system expected.",
         "Quantité comptée en part de la quantité attendue par le système."),
        ("Inventory", "Stocks"), 3, "percent", "neither", _PHARMACY, _pharmacy_count_accuracy),
    Kpi("pharmacy.shipping_cost_per_shipment", ("Shipping cost per shipment", "Coût d'expédition moyen"),
        ("Shipping cost, per shipment.", "Coût d'expédition, par expédition."),
        ("Shipping", "Expéditions"), 3, "currency", "lower", _PHARMACY, _pharmacy_shipping_cost),
    # A shop
    Kpi("sales.average_order_value", ("Average order value", "Panier moyen"),
        ("Net sales per order.", "Ventes nettes par commande."),
        ("Sales", "Ventes"), 1, "currency", "higher", _SALES, _sales_aov),
    Kpi("sales.gross_margin_pct", ("Gross margin %", "Marge brute %"),
        ("Net sales less their cost, as a share of net sales.", "Ventes nettes moins leur coût, en part des ventes nettes."),
        ("Profitability", "Rentabilité"), 1, "percent", "higher", _SALES, _sales_margin_pct),
    Kpi("sales.gross_margin", ("Gross margin", "Marge brute"),
        ("Net sales less their cost.", "Ventes nettes moins leur coût."),
        ("Profitability", "Rentabilité"), 2, "currency", "higher", _SALES, _sales_margin),
    Kpi("sales.return_rate", ("Return rate", "Taux de retour"),
        ("Refunds as a share of net sales.", "Remboursements en part des ventes nettes."),
        ("Returns", "Retours"), 2, "percent", "lower", _SALES, _sales_return_rate),
    Kpi("sales.discount_rate", ("Discount rate", "Taux de remise"),
        ("Discounts as a share of gross sales.", "Remises en part des ventes brutes."),
        ("Pricing", "Prix"), 2, "percent", "lower", _SALES, _sales_discount_rate),
    Kpi("sales.sales_per_customer", ("Sales per customer", "Ventes par client"),
        ("Net sales per customer who bought.", "Ventes nettes par client acheteur."),
        ("Customers", "Clients"), 2, "currency", "higher", _SALES, _sales_per_customer),
    Kpi("sales.units_per_order", ("Units per order", "Articles par commande"),
        ("Quantity sold per order.", "Quantité vendue par commande."),
        ("Sales", "Ventes"), 3, "number", "higher", _SALES, _sales_units_per_order),
    Kpi("sales.average_selling_price", ("Average selling price", "Prix de vente moyen"),
        ("Net sales per unit sold.", "Ventes nettes par unité vendue."),
        ("Pricing", "Prix"), 3, "currency", "neither", _SALES, _sales_selling_price),
    # A ledger
    Kpi("finance.profit_margin", ("Profit margin", "Marge bénéficiaire"),
        ("Profit as a share of revenue.", "Bénéfice en part du chiffre d'affaires."),
        ("Profitability", "Rentabilité"), 1, "percent", "higher", _FINANCE, _finance_margin),
    Kpi("finance.expense_ratio", ("Expense ratio", "Ratio de charges"),
        ("Expenses as a share of revenue.", "Charges en part du chiffre d'affaires."),
        ("Costs", "Charges"), 2, "percent", "lower", _FINANCE, _finance_expense_ratio),
]
BY_KEY = {k.key: k for k in PACKS}

# Statuses worth a share of their own in any business: what went wrong, or did not happen.
NOTABLE = {"cancel": ("cancelled", "annulées"), "reject": ("rejected", "rejetées"), "return": ("returned", "retournées"),
           "fail": ("failed", "en échec"), "declin": ("declined", "refusées"), "refund": ("refunded", "remboursées"),
           "late": ("late", "en retard"), "overdue": ("overdue", "en retard"), "block": ("blocked", "bloquées"),
           "churn": ("churned", "perdues"), "void": ("voided", "annulées"), "revers": ("reversed", "contrepassées"),
           "defect": ("defective", "défectueuses"), "damag": ("damaged", "endommagées"),
           "expir": ("expired", "expirées"), "denied": ("denied", "refusées"), "disput": ("disputed", "contestées"),
           "abandon": ("abandoned", "abandonnées"), "lost": ("lost", "perdues")}
MOST_SHARES = 4
GENERIC = "any.share."


def _generic(b: Bind, lang: str) -> list[Proposal]:
    """A share of notable statuses on each table of rows that happen (cancelled orders, failed tests), one per
    table, the busiest tables first, never on a status a business metric already reads."""
    from core2.recipes import area_name

    out: list[Proposal] = []
    tables = sorted((t for t in b.model.tables.values() if t.kind in ("fact", "snapshot") and not t.hidden
                     and b.may(t.key)), key=lambda t: (-(t.row_count or 0), t.key))
    for t in tables:
        if len(out) >= MOST_SHARES:
            break
        for stem, (word, mot) in NOTABLE.items():
            try:
                column, values = b.status(t, {stem}, avoid={"not", "non", "un", "in"})
            except Missing:
                try:
                    column, values = b.flag_named(t, stem)
                except Missing:
                    continue
            if column.key in b.used or not (column.role in ("status", "flag") or _words(
                    column.business_name or column.name) & {"status", "state", "result", "outcome", "decision"}):
                continue
            area = area_name(b.model, t.key)
            rows = area.lower()
            name = f"Share of {word} {rows}" if lang != "fr" else f"Part des {rows} {mot}"
            says = (f"{word.capitalize()} {rows}, as a share of all {rows}." if lang != "fr"
                    else f"{area} {mot}, en part de tous les {rows}.")
            b.used.add(column.key)
            out.append(Proposal(f"{GENERIC}{column.key}.{stem}", name, says, area, 3, "percent", "lower",
                                True, Formula(b.share(b.rows(t), b.is_one_of(column, values)))))
            break
    return out


def proposals(model: SemanticModel, kind: str, *, may: May = _all, lang: str = "en") -> list[Proposal]:
    """The business metrics the data holds for a business of ``kind``, its own first, then shares of notable
    statuses; each with its formula, not yet read or checked. A metric the model already has (an admin's of the
    same business metric, or of the same name) is not proposed again."""
    b = Bind(model, may)
    i = 1 if lang == "fr" else 0
    had = {m.kpi for m in model.measures.values() if m.kpi}
    named = {m.business_name.casefold() for m in model.measures.values() if not m.hidden}
    out: list[Proposal] = []
    for kpi in PACKS:
        if kind not in kpi.kinds:
            continue
        try:
            formula = kpi.write(b)    # written even when kept already: the status it reads is then not shared again
        except Missing:
            continue
        except Exception as exc:  # noqa: BLE001 - one metric the data trips up is left out, never the rest
            log.warning("core2: the business metric %s could not be written: %s", kpi.key, exc)
            continue
        if kpi.key in had or kpi.name[i].casefold() in named:
            continue
        out.append(Proposal(kpi.key, kpi.name[i], kpi.says[i], kpi.area[i], kpi.rank, kpi.fmt, kpi.better,
                            kpi.share, formula))
    out.sort(key=lambda p: p.rank)
    out += [p for p in _generic(b, lang) if p.key not in had and p.name.casefold() not in named]
    return out


def measure_of(model: SemanticModel, proposal: Proposal) -> Measure:
    """The proposal as a metric, read as the editor reads one: an AuthoringError when the workspace's fields
    cannot be read that way (two tables of one name, a field gone)."""
    from core2 import ids
    from core2.model import authoring
    from core2.model.imports import _behaviour

    read = authoring.read(model, proposal.formula.text)
    additivity, time_aggregation = _behaviour(model, read.table, read.expr)
    slug = ids.unique_slug(ids.slug(proposal.name, fallback="metric"), {m.slug for m in model.measures.values()})
    date = proposal.formula.date
    if date and (date not in model.date_roles or model.date_roles[date].table != read.table):
        date = None
    return Measure(key=f"kpi:{slug}", slug=slug, business_name=proposal.name, description=proposal.says,
                   table=read.table, expr=read.expr, additivity=additivity, time_aggregation=time_aggregation,
                   format=proposal.fmt if proposal.fmt in authoring.FORMAT_WORDS else "number", default_date=date,
                   kind="metric", provenance="heuristic", status="proposed", better=proposal.better,
                   kpi=proposal.key, tested=False)


@dataclass
class Approved:
    """A business metric an admin accepted: where an executive dashboard shows it."""
    measure: Measure
    area: str
    rank: int
    order: int = field(default=0)
    area_key: str = ""              # the area in English, for telling it from a table's tab of the same subject


def approved(model: SemanticModel, *, may: May = _all, lang: str = "en") -> list[Approved]:
    """The business metrics an admin accepted, the most important first: their area and rank from their
    business metric, a share of a status in its table's area."""
    from core2.recipes import area_name

    i = 1 if lang == "fr" else 0
    order = {k.key: n for n, k in enumerate(PACKS)}
    out = []
    for m in model.measures.values():
        if not m.kpi or m.hidden or not may(m.table) or m.table not in model.tables:
            continue
        kpi = BY_KEY.get(m.kpi)
        if kpi is not None:
            out.append(Approved(m, kpi.area[i], kpi.rank, order[kpi.key], kpi.area[0]))
        elif m.kpi.startswith(GENERIC):
            area = area_name(model, m.table)
            out.append(Approved(m, area, 3, len(PACKS), area))
    out.sort(key=lambda a: (a.rank, a.order, a.measure.slug))
    return out


def date_role(model: SemanticModel, key: str | None) -> DateRole | None:
    return model.date_roles.get(key) if key else None
