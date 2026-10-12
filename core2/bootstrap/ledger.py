"""A ledger's own figures: revenue, expenses and profit, from its debits and credits and each account's type.

A general ledger keeps every amount twice, as a debit on one account and a credit on another, so its lines add
up to nothing: "what was our revenue?" answered with the lines' net amount, or with their debits, gives $0 or
the whole ledger's turnover. A finance reader means the accounts' own figures: revenue is the credits less the
debits on revenue accounts, expenses the debits less the credits on expense accounts, and profit the one less
the other. Learn finds a lines table with a debit and a credit amount that reaches an account type naming
revenue and expense accounts, and proposes the three figures with the words finance readers use ("revenue",
"spend", "net income"; "chiffre d'affaires", "dépenses", "bénéfice"). Each reads only the accounts of its type,
through the lines' link to their account, so a breakdown by cost centre or a month still adds up.

Account types are read from the column's own values (Revenue, Income, Sales; Expense, Cost of Sales, Operating
Expenses): a value naming a cost is an expense even where it also says "sales" ("Cost of Sales").
"""

from __future__ import annotations

from sqlglot import exp

from core2.bootstrap import names
from core2.model.schema import ColumnFilter, Evidence, Measure, OpExpr, RefExpr, SemanticModel, SqlExpr

_DEBIT = {"debit", "dr", "debits"}
_CREDIT = {"credit", "cr", "credits"}
_REVENUE = {"revenue", "revenues", "income", "sales", "turnover", "sale"}
_EXPENSE = {"expense", "expenses", "cost", "costs", "cogs", "opex", "expenditure", "expenditures"}
_KINDS = {"asset", "assets", "liability", "liabilities", "equity", "capital", *_REVENUE, *_EXPENSE}

REVENUE_WORDS = {"en": ["revenue", "sales", "income", "turnover", "revenues"],
                 "fr": ["chiffre d'affaires", "revenus", "ventes", "produits"]}
EXPENSE_WORDS = {"en": ["expenses", "spend", "spending", "costs", "expenditure", "opex"],
                 "fr": ["dépenses", "charges", "coûts"]}
PROFIT_WORDS = {"en": ["profit", "net income", "earnings", "net profit", "result"],
                "fr": ["bénéfice", "résultat", "profit"]}


def _kind(value: str) -> str | None:
    words = set(names.tokens(value))
    if words & _EXPENSE:
        return "expense"
    if words & _REVENUE:
        return "revenue"
    return None


def _amount(model: SemanticModel, table: str, words: set[str]) -> Measure | None:
    for m in model.measures.values():
        column = getattr(m.expr, "column", None)
        if m.table == table and column and getattr(m.expr, "agg", "") == "sum" and m.format == "currency" \
                and set(names.tokens(model.columns[column].name)) & words:
            return m
    return None


def _account_type(model: SemanticModel, table: str):
    """The column naming each account's type that the lines reach, with its revenue and expense values."""
    from core2.resolve.resolver import _reaches

    best = None
    for column in model.columns.values():
        if column.data_type != "text" or column.profile is None or not column.profile.top:
            continue
        values = [str(t.value) for t in column.profile.top if t.value not in (None, "")]
        kinds = [v for v in values if set(names.tokens(v)) & _KINDS]
        revenue = [v for v in values if _kind(v) == "revenue"]
        expense = [v for v in values if _kind(v) == "expense"]
        if not revenue or not expense or len(kinds) < 0.6 * len(values):
            continue      # the values are not account types (a customer's name that says "sales")
        if column.table != table and not _reaches(model, table, column.table):
            continue
        rank = (len(kinds) / len(values), "type" in names.tokens(column.name))
        if best is None or rank > best[0]:
            best = (rank, column, revenue, expense)
    return best[1:] if best else None


def _formula(first: str, second: str) -> str:
    """SUM(first - second), its columns written by their keys as an admin's formula is."""
    return exp.Sum(this=exp.Sub(this=exp.column(exp.to_identifier(first, quoted=True)),
                                expression=exp.column(exp.to_identifier(second, quoted=True)))).sql()


def ledger_measures(model: SemanticModel) -> list[str]:
    """Adds revenue, expenses and profit to a ledger's lines; returns a note for each table given them."""
    notes: list[str] = []
    taken = {w.casefold() for m in model.measures.values() for w in [m.business_name] if w}
    for table in list(model.tables):
        debit, credit = _amount(model, table, _DEBIT), _amount(model, table, _CREDIT)
        if debit is None or credit is None:
            continue
        found = _account_type(model, table)
        if found is None:
            continue
        column, revenue, expense = found
        d, c = debit.expr.column, credit.expr.column    # type: ignore[union-attr]
        typed = model.columns[column.key].business_name.lower()
        names_ = {"revenue": "Revenue", "expenses": "Expenses", "profit": "Profit"}
        if any(n.casefold() in taken for n in names_.values()):
            names_ = {k: f"Ledger {v.lower()}" for k, v in names_.items()}
        made: dict[str, str] = {}
        for role, first, second, values, words, better in (
                ("revenue", c, d, revenue, REVENUE_WORDS, "higher"),
                ("expenses", d, c, expense, EXPENSE_WORDS, "lower")):
            key = f"m:{table}.ledger.{role}"
            side = "credits less debits" if role == "revenue" else "debits less credits"
            model.measures[key] = Measure(
                key=key, business_name=names_[role], table=table, kind="model", format="currency",
                description=(f"The {side} on accounts whose {typed} is {' or '.join(values)}: what the ledger "
                             f"{'earned' if role == 'revenue' else 'spent'}."),
                expr=SqlExpr(sql=_formula(first, second), columns=[first, second],
                             filters=[ColumnFilter(column=column.key, op="in", values=list(values))]),
                additivity="additive", default_date=debit.default_date, better=better,  # type: ignore[arg-type]
                unit=debit.unit, unit_column=debit.unit_column,      # never two currencies added together
                synonyms={k: list(v) for k, v in words.items()},
                evidence=[Evidence(kind="ledger", weight=1, detail=(
                    f"{model.tables[table].business_name} holds a debit and a credit on each line, and each "
                    f"line's account has a {typed}: {names_[role].lower()} is the {side} on "
                    f"{' and '.join(values)} accounts"))],
                provenance="profile", status="verified", confidence=0.85)
            made[role] = key
        key = f"m:{table}.ledger.profit"
        model.measures[key] = Measure(
            key=key, business_name=names_["profit"], table=table, kind="model", format="currency",
            description=f"{names_['revenue']} less {names_['expenses'].lower()}.",
            expr=OpExpr(op="subtract", args=[RefExpr(measure=made["revenue"]), RefExpr(measure=made["expenses"])]),
            additivity="additive", default_date=debit.default_date, synonyms={k: list(v) for k, v in PROFIT_WORDS.items()},
            unit=debit.unit, unit_column=debit.unit_column,
            evidence=[Evidence(kind="ledger", weight=1, detail=f"{names_['revenue']} less {names_['expenses'].lower()}")],
            provenance="profile", status="verified", confidence=0.85)
        # The words now belong to the ledger's figures, not to the lines' plain debits and credits.
        for m in model.measures.values():
            if m.key in (made["revenue"], made["expenses"], key):
                continue
            for lang, words in m.synonyms.items():
                owned = {w.casefold() for w in [*REVENUE_WORDS.get(lang, []), *EXPENSE_WORDS.get(lang, []),
                                                *PROFIT_WORDS.get(lang, [])]}
                m.synonyms[lang] = [w for w in words if w.casefold() not in owned]
        notes.append(f"{model.tables[table].business_name}: revenue, expenses and profit are worked out from the "
                     f"debits and credits on each {typed}.")
    return notes
