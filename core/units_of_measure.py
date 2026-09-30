"""
A quantity is counted in its item's unit, and units do not add up.

A stock fact holds each item's quantity in that item's unit of measure: most
items counted in eaches (EA), some in feet (FT), metres (ME), rolls (RL).
Summed across items, "total stock on hand" adds eaches to feet to metres -- a
number that measures nothing, from SQL that looks entirely ordinary.

So a total of a quantity is read per unit: the query groups by the unit of
measure, keeps to one unit, or stays within one item (an item has one unit).
The rule rides on the semantic plan like the period-row rule: the prompt states
it, the validator refuses a total that mixes units, and the answer card says
the quantities are per unit. A value -- a quantity times a cost -- is currency
and still adds up across units.

Tenant-neutral: the policy comes from the discovered schema and the graph --
which facts hold quantities, which columns name a unit of measure (the fact's
own, or that of the item dimension the fact joins), and which identify an item.
"""

from __future__ import annotations

import logging
import re
import unicodedata

log = logging.getLogger("querybot.units_of_measure")

_FLAG_SUFFIXES = ("_FLG", "_FLAG", "_IND")
_ITEM_NAME_SUFFIXES = ("CD", "CODE", "NM", "NAME", "DSC", "DESC", "NUM", "NO", "ID")
_DIMENSION_AFFIXES = re.compile(r"^(?:DIM_|D_)|(?:_DMS|_DIM|_D)$")

_ASKED_FOR = re.compile(
    r"\b(?:all|any|regardless\s+of|across|whatever)\s+(?:the\s+)?units?\b"
    r"|\btoutes?\s+(?:les\s+)?unit[ée]s\b|quelle\s+que\s+soit\s+l['’]unit[ée]",
    re.IGNORECASE,
)


def _tokens(column: str) -> list[str]:
    return [t for t in str(column or "").upper().split("_") if t]


def is_unit_column(column: str) -> bool:
    """UNT_OF_MSR, UOM, BASE_UOM, UNIT_OF_MEASURE -- not UNIT_PRICE."""
    tokens = _tokens(column)
    return "UOM" in tokens or (
        bool({"UNT", "UNIT"} & set(tokens)) and bool({"MSR", "MEASURE"} & set(tokens))
    )


def is_quantity_column(column: str) -> bool:
    """ON_HND_QTY, QTY_ON_HAND, ORDER_QUANTITY -- not a flag about one."""
    name = str(column or "").upper()
    tokens = _tokens(name)
    return (
        bool({"QTY", "QUANTITY"} & set(tokens))
        and not name.endswith(_FLAG_SUFFIXES)
        and not is_unit_column(name)
    )


def question_asks_across_units(*texts: str) -> bool:
    return any(_ASKED_FOR.search(str(text or "")) for text in texts)


# A result's measure is a quantity of goods by its name: STOCK_ON_HAND,
# UNITS_SOLD, TOTAL_QTY, "Quantité disponible". A value, a count, a rate or an
# average beside a unit column is not: money adds up across units, 3 items
# counted in eaches are not 3 eaches, and a fill rate is no length.
_QUANTITY_WORDS = frozenset({
    "QTY", "QUANTITY", "QUANTITIES", "UNITS", "STOCK", "STOCKS", "STK", "OH", "QOH", "SOH",
    "ONHAND", "INVENTORY", "QTE", "QUANTITE", "QUANTITES", "UNITES",
})
# A movement's participle names a quantity where nothing else says what is
# counted -- TOTAL_SOLD -- and not beside a count noun: ORDERS_SHIPPED is a
# number of orders, LINES_RECEIVED one of lines.
_PARTICIPLES = frozenset({"SOLD", "ORDERED", "PURCHASED", "RECEIVED", "SHIPPED"})
_COUNT_NOUNS = frozenset({
    "ORDERS", "LINES", "ITEMS", "CUSTOMERS", "SHIPMENTS", "RECEIPTS", "INVOICES", "DELIVERIES",
    "TRANSACTIONS", "DOCUMENTS",
})
# Only as "on hand": HAND_TOOLS_SALES is the sales of hand tools.
_ON_HAND = frozenset({"ON", "HAND"})
_NOT_QUANTITY_WORDS = frozenset({
    "VALUE", "VAL", "AMT", "AMOUNT", "COST", "CST", "PRICE", "PRC", "REVENUE", "MARGIN",
    "PROFIT", "PFT", "SPEND", "COUNT", "CNT", "NUMBER", "NBR", "RATE", "RATIO", "PCT",
    "PERCENT", "PERCENTAGE", "SHARE", "AVG", "AVERAGE", "MEAN", "TURNS", "TURNOVER", "COVERAGE",
    "USD", "CAD", "EUR", "GBP", "DOLLAR", "DOLLARS",
    "VALEUR", "MONTANT", "COUT", "PRIX", "NOMBRE", "TAUX", "MOYENNE", "POURCENTAGE",
})
# A length of time: WEEKS_ON_HAND, STOCK_COVER_WKS, DAYS_OF_SUPPLY, "jours de
# stock". A period is not one -- MONTH_END_STOCK_ON_HAND, UNITS_SOLD_THIS_MONTH
# and "stock fin de mois" are stock and units -- nor is the window a quantity
# is counted over: UNITS_SOLD_LAST_4_WKS, QTY_SOLD_13_WKS, "ventes 4 dernières
# semaines".
_LENGTHS_OF_TIME = frozenset({"DAYS", "WEEKS", "WKS", "MONTHS", "MTHS", "JOURS", "SEMAINES"})
_A_WINDOW = frozenset({
    "LAST", "PAST", "PRIOR", "PREVIOUS", "PREV", "ROLLING", "TRAILING", "NEXT",
    "DERNIERS", "DERNIERES", "PROCHAINS", "PROCHAINES"})
# Stock cover is a length of time -- STOCK_COVER, "couverture de stock" --
# unless the name says it is a quantity: COVER_STOCK, COVER_STOCK_QTY.
_COVER = frozenset({"COVER", "COUVERTURE"})
# A date, a key or a code, whatever it is named after, where it ends the name:
# SHIPPED_DATE, SHIPPED_ON_DATE, RECEIVED_DT_KEY, SOLD_TO_ID. Not "to date" or
# "on time" -- UNITS_SOLD_TO_DATE and QTY_SHIPPED_ON_TIME are units -- nor a
# word further in: NUM_UNITS, QTY_NO_CHARGE.
_IDENTIFIER_HEADS = frozenset({
    "DATE", "DT", "DTE", "KEY", "SK", "ID", "NO", "NUM", "CD", "CODE", "TIME", "TIMESTAMP", "TS",
})
_COUNTED_SO_FAR = frozenset({("TO", "DATE"), ("TO", "DT"), ("ON", "TIME")})
# A quantity for each of something is a ratio: UNITS_PER_CASE, QTY_PER_ORDER,
# PER_CASE_QTY. PER_END_QTY is the period's end, and QTY_PER_WAREHOUSE one per
# warehouse.
_PER_WHAT = frozenset({
    "CASE", "CS", "PACK", "PK", "PALLET", "PLT", "BOX", "CARTON", "CTN", "BAG", "INNER", "ORDER",
    "LINE", "DAY", "WEEK", "MONTH", "YEAR", "HOUR", "UNIT", "EACH", "PERSON", "EMPLOYEE", "CUSTOMER",
    "TRANSACTION", "INVOICE", "SHIPMENT", "SKU", "STORE",
})
# The formats a quantity is shown in; a currency, a percentage or a count is
# not one.
_QUANTITY_FORMATS = frozenset({"", "number", "quantity", "decimal", "integer"})
# A reader who ranks the units themselves: "which unit of measure do we sell
# the most", "the unit of measure with the most stock", "largest unit of
# measure by stock", "top 2 units of measure", "which unit of measure leads",
# "order the units of measure by stock", "quelle unité de mesure a le plus de
# stock", "l'unité de mesure avec le plus de stock", or rows asked for in an
# order -- "sorted descending", "from highest to lowest", "du plus grand au
# plus petit", "classement des unités de mesure". Read without accents. The
# superlative follows the units and what they do, with no preposition between:
# one after "at", "for" or "pour" is about something else -- "which units of
# measure have at least 100", "what units do we hold at our largest
# warehouse", "quelle unité de mesure pour l'entrepôt le plus grand" -- as is
# one about a time, "the most recent stock by unit of measure", and "units"
# alone is a quantity: "for the items with the most units sold", "for units
# with the highest turnover". Nor "ordered by" -- "quantity ordered by unit of
# measure" is the quantity ordered -- nor the ERP's "order UOM", the unit an
# item is ordered in: "stock on hand by order UOM". Nor a rank of something
# else: "for our top-ranked suppliers".
_UNIT_WORDS = r"(?:units?(?:\s+of\s+measures?)?|uoms?|unites?(?:\s+de\s+mesures?)?)"
_UOM_WORDS = r"(?:units?\s+of\s+measures?|uoms?|unites?\s+de\s+mesures?)"
_SUPERLATIVE = r"(?:most|least|largest|smallest|biggest|highest|lowest|greatest|fewest)"
_NOT_A_RANK = r"(?!\s+(?:recent|recently|up|current|latest|a\s+jour|recente?s?|actuel))"
_ENDS = r"(?:highest|largest|biggest|greatest|most|lowest|smallest|least|fewest)"
_FRENCH_ENDS = r"(?:grande?s?|petite?s?|elevee?s?|haute?s?|bas(?:se)?s?|faibles?)"
# What the units do before the superlative: "has", "do we currently hold",
# "accounts for", "a le stock en main".
_THEIR_VERB = (
    r"(?:accounts?\s+for\s+|(?!(?:at|in|for|from|of|by|on|to|across|within|among|per)\b)[\w'-]+\s+){0,4}")
_LEUR_VERBE = (
    r"(?:en\s+(?:main|stock)\s+"
    r"|(?!(?:pour|dans|au|aux|du|des|de|en|chez|sur|par|entre|parmi)\b)(?![ld]')[\w'-]+\s+){0,4}")
_RANKS_THE_UNITS = re.compile(
    rf"\b(?:which|what)\s+(?:of\s+(?:our|the)\s+)?{_UNIT_WORDS}\s+{_THEIR_VERB}(?:the\s+)?{_SUPERLATIVE}\b"
    rf"{_NOT_A_RANK}"
    rf"|\b(?:which|what)\s+(?:of\s+(?:our|the)\s+)?{_UNIT_WORDS}\s+(?:leads?|comes?\s+first)\b"
    rf"|\b{_UOM_WORDS}\s+(?:with|having|holding|that\s+(?:has|have|holds?))\s+(?:the\s+)?{_SUPERLATIVE}\b"
    rf"{_NOT_A_RANK}"
    rf"|\b(?:{_SUPERLATIVE}|top|bottom|leading)\s+(?:\d+\s+)?{_UOM_WORDS}\b"
    rf"|\brank(?:s|ed|ing)?\s+(?:of\s+)?(?:the\s+|our\s+)?{_UNIT_WORDS}\b"
    rf"|(?:^\s*|[,;:]\s*|\b(?:please|and|then|also|you)\s+)(?:order|sort)\s+(?:the\s+|our\s+)?{_UOM_WORDS}\b"
    rf"|\bquel(?:le)?s?\s+{_UNIT_WORDS}\s+{_LEUR_VERBE}(?:le|la|les)\s+(?:plus|moins)\b{_NOT_A_RANK}"
    rf"|\b{_UNIT_WORDS}\s+(?:avec|qui\s+a|ayant)\s+(?:le|la|les)\s+(?:plus|moins)\b{_NOT_A_RANK}"
    rf"|\b(?:plus|moins)\s+(?:grande?s?|petite?s?)\s+{_UNIT_WORDS}\b"
    rf"|\b(?:{_ENDS}|high|low)[\s-]+to[\s-]+(?:the\s+)?(?:{_ENDS}|high|low)\b"
    rf"|\bd(?:u|e\s+la|es)\s+plus\s+{_FRENCH_ENDS}\s+a(?:u|ux|\s+la)\s+plus\s+{_FRENCH_ENDS}\b"
    r"|(?<!-)\b(?:sorted|ranked)(?=\s+(?:by|from|in|descending|ascending|highest|largest|biggest|lowest|smallest)\b"
    r"|\s*(?:[,.;:!?]|$))"
    r"|\b(?:descending|ascending|(?:highest|largest|biggest|lowest|smallest)\s+first"
    r"|classement|classer|classees?|triees?|decroissant|croissant)\b",
    re.IGNORECASE,
)


def _word_list(column: str) -> list[str]:
    """A column's name as its words, in order, without accents."""
    folded = "".join(
        ch for ch in unicodedata.normalize("NFD", str(column or "")) if unicodedata.category(ch) != "Mn")
    return [word.upper() for word in re.split(r"[^A-Za-z0-9]+|(?<=[a-z0-9])(?=[A-Z])", folded) if word]


def _not_goods(words: list[str]) -> bool:
    """A name that says its figure is not a quantity of goods: a value, a
    count, a rate, a length of time, an identifier, a ratio."""
    names = set(words)
    if names & _NOT_QUANTITY_WORDS:
        return True
    if any(word in _LENGTHS_OF_TIME and not (index and (words[index - 1].isdigit() or words[index - 1] in _A_WINDOW))
           for index, word in enumerate(words)):
        return True
    if (
        names & _COVER and not names & {"QTY", "QUANTITY", "QTE", "QUANTITE"}
        and not ("COVER" in words and set(words[words.index("COVER") + 1:]) & _QUANTITY_WORDS)
    ):
        return True
    pairs = list(zip(words, words[1:]))
    # "Mois de stock" is months of it; "stock fin de mois" and "stock du mois
    # de mars" are stock.
    if words[:2] == ["MOIS", "DE"]:
        return True
    if words and words[-1] in _IDENTIFIER_HEADS and tuple(words[-2:]) not in _COUNTED_SO_FAR:
        return True
    return any(first == "PER" and second in _PER_WHAT for first, second in pairs)


def is_quantity_measure(column: str, measure_format: str = "") -> bool:
    """A result's measure read as a quantity of goods: STOCK_ON_HAND, UNITS_SOLD,
    TOTAL_QTY, MONTH_END_STOCK_ON_HAND, UNITS_SOLD_TO_DATE -- not STOCK_VALUE,
    ITEM_COUNT, FILL_RATE, AVERAGE_DAYS_ON_HAND, STOCK_COVER, UNITS_PER_CASE,
    SHIPPED_DATE or ORDERS_SHIPPED."""
    if str(measure_format or "") not in _QUANTITY_FORMATS:
        return False
    words = _word_list(column)
    if _not_goods(words):
        return False
    names = set(words)
    return bool(names & _QUANTITY_WORDS or _ON_HAND <= names
                or (names & _PARTICIPLES and not names & _COUNT_NOUNS))


def unit_of(value) -> str:
    """A row's unit of measure as the rows are told apart by it: "ea" is "EA",
    and a blank unit is one unit, however it is blank."""
    return "" if value is None else str(value).strip().upper()


def kept_in_several_units(rows: list[dict], column: str, *, measure_format: str = "") -> bool:
    """A quantity the rows keep in more than one unit of measure.

    Its sum across them adds eaches to feet, and so do its mean, its spread,
    its outliers and every share of it: none is a figure of anything.
    """
    if not rows or not is_quantity_measure(column, measure_format):
        return False
    return len({unit_of(row.get(key)) for row in rows for key in row if is_unit_column(key)}) > 1


def per_unit_totals(
    rows: list[dict], unit_column: str, measure_column: str,
) -> tuple[list[tuple[str, float]], list[str]] | None:
    """A quantity's total in each unit of measure, largest first, and the units
    the rows give no total for, by name.

    Largest first for the reader, and in a set order: a query grouped by unit
    returns them in none, and a headline names three and folds the rest into
    "and 9 more" -- by name those three could be 0 BG, 3 BX and 1 CD, with
    32,402 FT among the rest. The order ranks nothing: the card says each total
    is in its own unit, and none is ahead of another. Equal totals go by name,
    and the rows with no unit last.

    Rows of one unit are merged where the measure adds up -- units sold by unit
    and month -- and None where it does not: a balance kept per month is not
    summed (core.analysis_contract.collapse_rows_by_label).
    """
    from core.analysis_contract import collapse_rows_by_label

    keyed = [{**row, unit_column: unit_of(row.get(unit_column))} for row in rows or []]
    totals = collapse_rows_by_label(keyed, unit_column, measure_column)
    if totals is None:
        return None
    valued = {unit for unit, _ in totals}
    missing = {row[unit_column] for row in keyed if row[unit_column] not in valued}
    return (sorted(totals, key=lambda pair: (pair[0] == "", -pair[1], pair[0])),
            sorted(missing, key=lambda unit: (unit == "", unit)))


def rows_per_unit(
    rows: list[dict], label_column: str, measure_column: str, question: str = "", *, measure_format: str = "",
) -> bool:
    """Rows that are a quantity's totals, one per unit of measure, which the
    reader did not ask to have ranked.

    A total of a quantity is read per unit, so "what is our total stock on
    hand?" comes back as a row of feet, one of eaches, one of metres. Each is a
    total in its own unit and none is ahead of another: 32,402 feet do not lead
    13,151 eaches, are not 19,251 above them, and are no share of a total of
    both. A reader who asks which unit holds the most is answered as asked, and
    money -- a quantity times a cost -- adds up across units, so is ranked too.
    """
    if not is_unit_column(label_column):
        return False
    if not kept_in_several_units(rows, measure_column, measure_format=measure_format):
        return False
    folded = "".join(
        ch for ch in unicodedata.normalize("NFD", question or "") if unicodedata.category(ch) != "Mn")
    return not _RANKS_THE_UNITS.search(folded)


def _qualified(entity: dict) -> str:
    schema = str(entity.get("schema_name") or "").strip()
    table = str(entity.get("table_name") or "").strip()
    return f"{schema}.{table}" if schema else table


def _item_columns(entity: dict, columns: list[tuple[str, str]], key: str) -> list[str]:
    """The columns that name one item: its key, and its own code, name or
    description (ITM_CD, ITM_NM) -- not ITM_TYP_CD, which names a type."""
    stem = _DIMENSION_AFFIXES.sub("", str(entity.get("table_name") or "").upper())
    own = {f"{stem}_{suffix}" for suffix in _ITEM_NAME_SUFFIXES}
    return [key] + [name for name, _ in columns if name.upper() in own and name.upper() != key.upper()]


def unit_policies(account_id: str) -> list[dict]:
    """One policy per fact that holds quantities and has a unit of measure --
    its own column, or the one on a dimension it joins."""
    import store
    from core.unknown_members import _columns_of, _discovered_columns

    state = store.get_client_state(account_id) or {}
    tables = _discovered_columns(state.get("schema_dir", ""))
    if not tables:
        return []
    entities = {e["entity_name"]: e for e in store.list_entities(account_id, active_only=True)}
    relationships = store.list_relationships(account_id, active_only=True)
    policies: list[dict] = []
    for entity in entities.values():
        if (entity.get("entity_type") or "") != "fact":
            continue
        columns = _columns_of(entity, tables)
        quantities = [name for name, _ in columns if is_quantity_column(name)]
        if not quantities:
            continue
        fact = _qualified(entity)
        units = [{"table": fact, "column": name} for name, _ in columns if is_unit_column(name)]
        items: list[dict] = []
        joins: list[dict] = []
        for rel in relationships:
            if rel.get("from_entity") != entity["entity_name"]:
                continue
            target = entities.get(rel.get("to_entity"))
            if target is None:
                continue
            target_columns = _columns_of(target, tables)
            target_units = [name for name, _ in target_columns if is_unit_column(name)]
            if not target_units:
                continue
            table = _qualified(target)
            units.extend({"table": table, "column": name} for name in target_units)
            joins.append({
                "table": table,
                "fact_column": str(rel.get("from_column") or ""),
                "key": str(rel.get("to_column") or ""),
            })
            items.append({"table": fact, "column": str(rel.get("from_column") or "")})
            items.extend(
                {"table": table, "column": name}
                for name in _item_columns(target, target_columns, str(rel.get("to_column") or ""))
            )
        if not units:
            continue
        policies.append({
            "kind": "units_of_measure",
            "fact_table": fact,
            "quantities": quantities,
            "unit_columns": _unique(units),
            "item_columns": _unique([item for item in items if item["column"]]),
            "unit_joins": _unique([join for join in joins if join["fact_column"] and join["key"]]),
        })
    return sorted(policies, key=lambda p: p["fact_table"])


def _unique(refs: list[dict]) -> list[dict]:
    seen: list[dict] = []
    for ref in refs:
        if ref not in seen:
            seen.append(ref)
    return seen


def attach_unit_policies(semantic_plan: dict | None, account_id: str, *questions: str) -> list[dict]:
    """Put the policies on the plan -- unless the question asks for a total
    across units, when it gets one."""
    if not isinstance(semantic_plan, dict) or question_asks_across_units(*questions):
        return []
    policies = unit_policies(account_id)
    if policies:
        semantic_plan["unit_policies"] = policies
    return policies


def _bare(name: str) -> str:
    return str(name or "").split(".")[-1].strip('[]"`').upper()


def policies_in_scope(policies: list[dict] | None, *texts: str) -> list[dict]:
    haystack = " ".join(str(text or "") for text in texts).upper()
    return [
        policy for policy in policies or []
        if isinstance(policy, dict) and re.search(
            rf"(?<![A-Z0-9_]){re.escape(_bare(policy.get('fact_table', '')))}(?![A-Z0-9_])", haystack,
        )
    ]


def _preferred_unit(policy: dict) -> dict:
    """The item's unit before the fact's own: the item always has one, and a
    fact's copy can be blank on most rows."""
    fact = _bare(policy["fact_table"])
    return next(
        (ref for ref in policy["unit_columns"] if _bare(ref["table"]) != fact),
        policy["unit_columns"][0],
    )


def format_unit_rules(policies: list[dict] | None) -> str:
    lines: list[str] = []
    for policy in policies or []:
        unit = _preferred_unit(policy)
        others = [
            f"{_bare(ref['table'])}.{ref['column']}" for ref in policy["unit_columns"] if ref != unit
        ]
        lines.append(
            f"- {policy['fact_table']}: {', '.join(policy['quantities'][:8])}"
            + (", ..." if len(policy["quantities"]) > 8 else "")
            + f" are in each item's unit ({_bare(unit['table'])}.{unit['column']}"
            + (f"; also {', '.join(others)}" if others else "") + ")."
        )
    if not lines:
        return ""
    return "\n".join([
        "## Units of measure — REQUIRED",
        "A quantity is counted in its item's unit of measure (each, feet, metres, ...), and "
        "different units do not add up. A SUM or AVG of a quantity across items MUST keep one "
        "total per unit: put the unit column in the SELECT and the GROUP BY (join the item "
        "table for it if needed) -- or filter to one unit, or group by item. A value (quantity "
        "times a cost) is currency and adds up across units.",
        *lines,
    ])


# ── Reading a query for a total that mixes units ────────────────────────────


def _quantity_aggregates(select, quantities: set[str]) -> list:
    """The SUM/AVG in this SELECT whose argument is made of quantities alone
    (ON_HND_QTY, ON_HND_QTY - ALC_QTY) -- not a value (ON_HND_QTY * ITM_CST)."""
    from sqlglot import exp

    found = []
    for node in select.find_all(exp.Sum, exp.Avg):
        if node.find_ancestor(exp.Select) is not select:
            continue
        columns = [str(c.name).upper() for c in node.this.find_all(exp.Column)]
        if columns and all(name in quantities for name in columns):
            found.append(node)
    return found


def _parse(sql, db_type: str):
    """The query's tree -- read the way the validator reads it -- or None."""
    import sqlglot

    from core.validator import _DIALECT, normalize_generated_sql

    if not isinstance(sql, str):
        return sql
    try:
        return sqlglot.parse_one(
            normalize_generated_sql(sql, db_type), read=_DIALECT.get(db_type, "snowflake"),
        )
    except Exception:
        return None


def _grouped_names(select) -> set[str]:
    from core.unknown_members import _grouped_columns

    if select.args.get("group") is None:
        return set()
    return {str(column.name).upper() for column in _grouped_columns(select)}


def unit_mixes(sql, policies: list[dict] | None, db_type: str = "azure_sql") -> list[dict]:
    """Each SELECT that totals a quantity across items without keeping units
    apart, with the fix: ``[{"policy", "required"}]``. Empty when every total
    is per unit, per item, or of one unit."""
    from sqlglot import exp

    from core.unknown_members import _conjuncts

    governed = [p for p in policies or [] if isinstance(p, dict) and p.get("quantities")]
    tree = _parse(sql, db_type) if governed else None
    if tree is None:
        return []

    mixes: list[dict] = []
    for select in tree.find_all(exp.Select):
        for policy in governed:
            if not _quantity_aggregates(select, {q.upper() for q in policy["quantities"]}):
                continue
            unit_names = {str(ref["column"]).upper() for ref in policy["unit_columns"]}
            item_names = {str(ref["column"]).upper() for ref in policy["item_columns"]}
            if _grouped_names(select) & (unit_names | item_names):
                continue
            if any(_keeps_one(conjunct, unit_names | item_names) for conjunct in _conjuncts(
                (select.args.get("where") or exp.Where()).this,
            )):
                continue
            mixes.append({"policy": policy, "required": _required(select, policy)})
    return mixes


def unit_for_total(formula: str, fact_table: str, policies: list[dict] | None,
                   db_type: str = "azure_sql") -> dict | None:
    """The unit a total of ``formula`` over ``fact_table`` is kept apart by,
    read by the same test the validator applies: ``{"table", "column",
    "join"}`` where ``join`` reaches the unit's table from the fact (None when
    the unit is the fact's own column). None when the formula totals no
    quantity of that fact -- a value (quantity times cost) adds up."""
    for policy in policies or []:
        if not isinstance(policy, dict) or not policy.get("quantities"):
            continue
        if _bare(policy.get("fact_table", "")) != _bare(fact_table):
            continue
        tree = _parse(f"SELECT {formula} FROM {_bare(fact_table)}", db_type)
        select = tree if tree is not None and tree.key == "select" else None
        if select is None or not _quantity_aggregates(select, {q.upper() for q in policy["quantities"]}):
            continue
        unit = _preferred_unit(policy)
        join = None
        if _bare(unit["table"]) != _bare(policy["fact_table"]):
            join = next((j for j in policy.get("unit_joins") or [] if j["table"] == unit["table"]), None)
            if join is None:
                continue
        return {"table": unit["table"], "column": unit["column"], "join": join}
    return None


def _keeps_one(conjunct, names: set[str]) -> bool:
    """column = 'EA', or column IN ('EA'): one unit, or one item."""
    from sqlglot import exp

    if isinstance(conjunct, exp.EQ):
        sides = (conjunct.this, conjunct.expression)
        return any(
            isinstance(a, exp.Column) and str(a.name).upper() in names and isinstance(b, exp.Literal)
            for a, b in (sides, sides[::-1])
        )
    if isinstance(conjunct, exp.In) and isinstance(conjunct.this, exp.Column):
        return str(conjunct.this.name).upper() in names and len(conjunct.expressions) == 1
    return False


def _required(select, policy: dict) -> str:
    """The unit column to group by, as this query can reach it. The item's
    unit is named before the fact's own copy, which can be blank on most rows
    -- with the join to reach it when the query does not read the item yet."""
    from core.unknown_members import _same_table, _sources

    sources = _sources(select)
    unit = _preferred_unit(policy)
    for table, _on, _inner in sources:
        if _same_table(table, unit["table"]):
            return f"{table.alias_or_name}.{unit['column']}"
    join = next((j for j in policy.get("unit_joins") or [] if j["table"] == unit["table"]), None)
    fact = next((t for t, _on, _inner in sources if _same_table(t, policy["fact_table"])), None)
    if join and fact is not None:
        item = _bare(unit["table"])
        return (
            f"{item}.{unit['column']} (JOIN {unit['table']} AS {item} "
            f"ON {fact.alias_or_name}.{join['fact_column']} = {item}.{join['key']})"
        )
    for ref in policy["unit_columns"]:
        for table, _on, _inner in sources:
            if _same_table(table, ref["table"]):
                return f"{table.alias_or_name}.{ref['column']}"
    return f"{unit['table']}.{unit['column']}"


def totals_per_unit(sql, policies: list[dict] | None, db_type: str = "azure_sql") -> bool:
    """Whether the query totals a quantity grouped by its unit of measure --
    an answer the card should say is per unit."""
    from sqlglot import exp

    governed = [p for p in policies or [] if isinstance(p, dict) and p.get("quantities")]
    tree = _parse(sql, db_type) if governed else None
    if tree is None:
        return False
    return any(
        _quantity_aggregates(select, {q.upper() for q in policy["quantities"]})
        and _grouped_names(select) & {str(ref["column"]).upper() for ref in policy["unit_columns"]}
        for select in tree.find_all(exp.Select)
        for policy in governed
    )
