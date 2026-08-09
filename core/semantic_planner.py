"""
Lightweight semantic field planner.

This module builds a deterministic field-source plan from the discovered schema
before the LLM writes SQL. It is intentionally conservative: it only emits a
plan when a user phrase maps to exact known columns and, when needed, a
reasonable join path can be inferred.
"""

from __future__ import annotations

import logging
import re
from collections import deque


log = logging.getLogger(__name__)


_ABBREVIATIONS = {
    # ── Generic dimensional abbreviations ─────────────────────────────────
    "AMT": "amount",
    "BAL": "balance",
    "BUS": "business",
    "CST": "cost",
    "CUS": "customer",
    "DLV": "delivery",
    "DLVD": "delivered",
    "DMS": "dimension",
    "DT": "date",
    "DVN": "division",
    "FCT": "fact",
    "GRP": "group",
    "IVC": "invoice",
    "IVCD": "invoiced",
    "ITM": "item",
    "LIN": "line",
    "NUM": "number",
    "ORD": "order",
    "PCH": "purchase",
    "PO": "purchase order",
    "PFT": "profit",
    "PRD": "period",
    "QTY": "quantity",
    "RCT": "receipt",
    "RPL": "replacement",
    "SFX": "suffix",
    "WHS": "warehouse",
    # ── M3/ERP raw short codes ─────────────────────────────────────────────
    "ACDT": "accounting date",
    "ALQT": "allocated quantity",
    "CONO": "company",
    "CSCD": "country",
    "CUAM": "customer amount",
    "CUCD": "currency",
    "CUNO": "customer",
    "DCOS": "delivery cost",
    "DIVI": "division",
    "DLIX": "delivery",
    "DLDT": "delivery date",
    "DLQT": "delivered quantity",
    "DWDT": "requested delivery date",
    "FACI": "facility",
    "ITDS": "item description",
    "ITGR": "item group",
    "ITNO": "item number",
    "ITTY": "item type",
    "IVDT": "invoice date",
    "IVNO": "invoice number",
    "IVQT": "invoiced quantity",
    "MFAM": "manufacturing amount",
    "ORDT": "order date",
    "ORNO": "order number",
    "ORQT": "ordered quantity",
    "ORST": "order status",
    "ORTP": "order type",
    "PCLA": "profit class",
    "PONR": "order line number",
    "POSX": "order line suffix",
    "SAAM": "sales amount",
    "SAPR": "sales price",
    "SDST": "sales district",
    "SMCD": "salesman",
    "SUNO": "supplier",
    "TRQT": "transaction quantity",
    "UCOS": "unit cost",
    "WHLO": "warehouse",
    "YEA4": "fiscal year",
}

_DIRECT_ALIASES = {
    # ── Dimensional fact columns ───────────────────────────────────────────
    "DIVI": {"division"},
    "ITM_GRP_DMS_KEY": {"item group", "item group key", "product group"},
    "CUS_IVC_LIN_AMT": {
        "invoice line amount",
        "total invoice line amount",
        "invoice amount",
        "sales amount",
    },
    "SOP_CUS_LIN_GRS_PFT_AMT": {
        "gross profit",
        "sales gross profit",
        "customer line gross profit",
    },
    "CUR_ON_HND_QTY": {"current on hand quantity", "on hand quantity"},
    "RCT_BUM_QTY": {"purchase receipt quantity", "receipt quantity"},
    "CUR_RPL_CST_AMT": {"current replacement cost", "replacement cost"},
    # ── M3/ERP raw column aliases ──────────────────────────────────────────
    # RULE: use only QUALIFIED multi-word phrases here — never bare generic words
    # like "quantity", "item", "supplier", "amount" which match too broadly and
    # pull the wrong table into the semantic plan.
    "TRQT": {"transaction quantity", "transaction qty"},           # NOT "quantity"/"volume"/"units"
    "PCLA": {"profit class", "fifo profit", "fifo margin", "margin tier", "fifo layer", "pcla"},
    "SUNO": {"supplier number", "vendor number"},                  # NOT bare "supplier"/"vendor"
    "CUNO": {"customer number"},                                   # NOT bare "customer"/"client"
    "SMCD": {"salesman code", "salesperson code", "sales rep code", "smcd"},
    "CUAM": {"customer amount", "billed amount"},                  # NOT "sales amount"/"revenue"
    "SAAM": {"net sales amount", "gross sales amount"},            # NOT bare "sales"
    "UCOS": {"unit cost", "cost per unit", "cogs per unit"},
    "WHLO": {"warehouse location", "whs"},                         # NOT bare "warehouse" (too broad)
    "ORNO": {"order number", "sales order number"},
    "IVNO": {"invoice number"},
    "IVQT": {"invoiced quantity", "invoiced qty", "billed quantity"},
    "ORQT": {"ordered quantity", "ordered qty", "order quantity"},
    "DLQT": {"delivered quantity", "delivered qty", "shipped quantity"},
    "SAPR": {"sales price", "list price"},                         # NOT bare "price"
    "ITNO": {"item number", "part number"},                        # NOT bare "item"/"product"/"sku"
    "ITGR": {"item group", "product group"},                       # NOT bare "category"
    # ── M3 MITBAL (item balance) ───────────────────────────────────────────
    # MITBAL columns carry the file's ML prefix, so they miss the bare M3
    # codes above: MLAVAL is not WHLO/ITNO-shaped and matched nothing at all.
    # Live case 16 named this table outright ("using the M3 balance data") and
    # still resolved against the daily snapshot fact, because no MITBAL
    # candidate was ever produced for the planner to choose between.
    # Same rule as the block above: qualified phrases only. "inventory value"
    # is two words and already the business term for this quantity; bare
    # "value" or "balance" would pull this ERP table into unrelated questions.
    "MLAVAL": {"m3 inventory value", "m3 balance value", "item balance value"},
    "MLSTQT": {"m3 stock quantity", "item balance stock quantity"},
    "MLALQT": {"m3 allocated quantity", "item balance allocated quantity"},
    "MLPERY": {"m3 period", "item balance period", "m3 balance period"},
    "MLWHLO": {"m3 warehouse", "item balance warehouse"},          # NOT bare "warehouse"
    "MLITNO": {"m3 item number", "item balance item number"},
    "MLLMDT": {"m3 last modified date"},
}

_JOIN_SYNONYMS = {
    "CUS_ORD_NUM": {"ORNO"},
    "CUS_ORD_LIN_NUM": {"PONR"},
    "CUS_ORD_LIN_SFX": {"POSX"},
}

_MEASURE_HINTS = ("amount", "profit", "quantity", "cost", "margin", "sales", "invoice")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _table_bare(table: str) -> str:
    return (table or "").upper().split(".")[-1]


def _table_schema(table: str) -> str:
    parts = (table or "").upper().split(".")
    return parts[-2] if len(parts) >= 2 else ""


def _table_variants(table: str) -> set[str]:
    table_u = (table or "").upper()
    parts = table_u.split(".")
    variants = {table_u}
    if parts:
        variants.add(parts[-1])
    if len(parts) >= 2:
        variants.add(".".join(parts[-2:]))
    return {v for v in variants if v}


def _planner_vocab(vocab=None):
    """Resolve the terminology vocab; defaults preserve legacy constants."""
    if vocab is not None:
        return vocab
    from core.vocab_packs import get_active_vocab
    return get_active_vocab()


def _column_words(column: str, vocab=None) -> list[str]:
    abbreviations = _planner_vocab(vocab).planner_abbreviations
    words: list[str] = []
    from core.identifier_intelligence import tokenize_identifier
    for token in tokenize_identifier(column, vocab=_planner_vocab(vocab)):
        if not token:
            continue
        word = abbreviations.get(token, token.lower())
        if word not in {"key", "dimension"}:
            words.append(word)
    return words


# Suffixes that mark a column as the human-readable label for its entity.
# Deliberately excludes key suffixes: WAREHOUSE_NAME answers to "warehouse",
# WAREHOUSE_SK does not. Aliasing both to the bare noun would make every such
# pair ambiguous, and ambiguous_source demotes both to optional -- weakening
# bindings that work today to fix ones that do not.
_DISPLAY_COLUMN_SUFFIXES = ("_NAME", "_DESCRIPTION", "_DESC", "_DSC", "_NM")


def _entity_noun_alias(column: str) -> str:
    """Return the entity a display column labels, or "" if it is not one.

    _aliases_for_column only ever produced the full column name as a phrase,
    so WAREHOUSE_NAME was addressable as "warehouse name" and nothing else.
    Nobody asks that way: dimensions are named by their entity noun ("revenue
    by warehouse", "sales per customer"), while measures happen to match
    because their business term usually *is* the whole column name
    (INVENTORY_VALUE / "inventory value"). The planner could therefore express
    measures but not the dimensions they are grouped by -- live case 10 built
    no plan at all, and case 7 had no dimension field for the display-name
    upgrade to attach to.
    """
    col = (column or "").upper()
    for suffix in _DISPLAY_COLUMN_SUFFIXES:
        if col.endswith(suffix) and len(col) > len(suffix):
            return col[: -len(suffix)]
    return ""


def _aliases_for_column(column: str, vocab=None) -> set[str]:
    v = _planner_vocab(vocab)
    col = (column or "").upper()
    aliases = {_norm(col), _norm(" ".join(_column_words(col, vocab=v)))}
    aliases.update(_norm(a) for a in v.direct_aliases.get(col, set()))
    entity = _entity_noun_alias(col)
    if entity:
        aliases.add(_norm(entity))
        aliases.add(_norm(" ".join(_column_words(entity, vocab=v))))
    # Users commonly omit a generic physical measure suffix. Derive that
    # business alias for every dataset (INVENTORY_VALUE -> "inventory")
    # instead of adding product/client-specific entries to Python.
    if any(
        col.endswith(suffix)
        for suffix in (
            "_AMOUNT", "_AMT", "_VALUE", "_QUANTITY", "_QTY", "_COUNT",
            "_TOTAL", "_COST", "_CST", "_PROFIT", "_PFT", "_MARGIN",
        )
    ):
        generic_suffixes = {
            "amount", "value", "quantity", "count", "number", "total",
            "metric", "measure",
        }
        # Derive only from the physical column expansion. A curated alias such
        # as "invoice amount" may intentionally be qualified; stripping it to
        # bare "invoice" would make a transaction noun look like a measure and
        # hard-require the wrong field.
        words = _norm(" ".join(_column_words(col, vocab=v))).split()
        while len(words) > 1 and words[-1] in generic_suffixes:
            words = words[:-1]
            if words:
                aliases.add(" ".join(words))
    return {a for a in aliases if a}


def _role_for_column(column: str, col_type: str = "") -> str:
    col = (column or "").upper()
    ctype = (col_type or "").upper()
    if col.endswith("_DT_DMS_KEY") or col.endswith("_DATE_DMS_KEY"):
        return "date_key"
    if col in {"DIVI"} or col.endswith("_DMS_KEY") or col in {"WHLO", "ORNO", "PONR", "POSX"}:
        return "dimension"
    if any(suffix in col for suffix in ("_AMT", "_QTY", "_CST", "_PFT")):
        return "measure"
    if any(token in ctype for token in ("INT", "DECIMAL", "NUMBER", "NUMERIC", "FLOAT")):
        return "measure"
    return "attribute"


def _table_context_score(table: str, question: str) -> int:
    t = _table_bare(table)
    q = _norm(question)
    score = 0
    if any(w in q for w in ("invoice", "sales", "customer order")) and any(x in t for x in ("CUS_ORD_IVC", "OOLINE", "OSBSTD")):
        score += 4
    if any(w in q for w in ("inventory", "on hand", "stock")) and "ITM_BAL" in t:
        score += 4
    if "purchase" in q and "PCH_ORD_RCT" in t:
        score += 4
    if "replacement" in q and "RPL_CST" in t:
        score += 4
    if any(w in q for w in ("fifo", "pcla", "margin")) and ("FIFO" in t or "OOLINE" in t):
        score += 4
    return score


def _score_candidate(
    table: str,
    column: str,
    role: str,
    question: str,
    base_tables: set[str],
    preferred_fact_tables: set[str] | None = None,
) -> int:
    score = _table_context_score(table, question)
    if table in base_tables:
        score += 6
    if role == "measure":
        score += 3
    if role == "dimension" and any(w in _norm(question) for w in ("by", "per", "each", "for each")):
        score += 2
    if _table_bare(table).startswith("DIM_"):
        score += 1
    if column == "DIVI" and "division" in _norm(question):
        score += 3
    if preferred_fact_tables and role == "measure" and any(
        _same_physical_table(table, preferred)
        for preferred in preferred_fact_tables
    ):
        score += 20
    return score


def _alias_forms(alias_norm: str) -> set[str]:
    """Singular + plural surface forms an alias can take in a question."""
    forms = {alias_norm}
    if not alias_norm.endswith("s"):
        forms.add(alias_norm + "s")
        if alias_norm.endswith("y"):
            forms.add(alias_norm[:-1] + "ies")
        if alias_norm.endswith(("x", "z", "ch", "sh")):
            forms.add(alias_norm + "es")
    if alias_norm.endswith("se"):
        forms.add(alias_norm[:-1] + "es")
    if alias_norm == "warehouse":
        forms.add("warehouses")
    return forms


def _alias_occurrence_spans(alias: str, question_norm: str) -> list[tuple[int, int]]:
    """Word-boundary spans where the alias (or a plural form) occurs."""
    alias_norm = _norm(alias)
    if not alias_norm:
        return []
    spans: list[tuple[int, int]] = []
    for form in _alias_forms(alias_norm):
        for m in re.finditer(rf"(?<![a-z0-9]){re.escape(form)}(?![a-z0-9])", question_norm):
            spans.append(m.span())
    return spans


def _contains_alias(alias: str, question_norm: str, question_compact: str) -> bool:
    alias_norm = _norm(alias)
    if not alias_norm:
        return False
    for form in _alias_forms(alias_norm):
        if re.search(rf"(?<![a-z0-9]){re.escape(form)}(?![a-z0-9])", question_norm):
            return True
    alias_compact = _compact(alias_norm)
    # Compact matching is only for long technical forms such as ITMGRPDMSKEY.
    # Short terms must not match inside larger words, e.g. AGE in percentAGE.
    return len(alias_compact) >= 6 and alias_compact in question_compact


_DURATION_IDIOM_RE = re.compile(
    r"\bdays?\s+(?:to|between|since|until)\b|\b(?:number|count)\s+of\s+days?\b"
)


def _column_matches_question(column: str, aliases: set[str], question_norm: str, question_compact: str) -> tuple[bool, str]:
    if column == "ITM_DMS_KEY" and "item group" in question_norm:
        return False, ""
    # A bare "DAY" column's alias pluralizes to "days", which matches inside
    # duration idioms — "avg days to pay", "days to ship", "days between
    # due and payment date", "days since last order", "number of days
    # present between X" — all ask for a calculated duration metric, not a
    # calendar-day grouping. Guard the same way the ITM_DMS_KEY case above
    # does. Trade-off: "revenue by day between Jan and Mar" (a date-range
    # filter, not a duration) also matches "day between" and is suppressed
    # too — accepted since a false suppression only loses a hint, while a
    # false requirement here hard-blocks the query.
    if column == "DAY" and _DURATION_IDIOM_RE.search(question_norm):
        return False, ""
    for alias in sorted(aliases, key=len, reverse=True):
        if not alias:
            continue
        if _contains_alias(alias, question_norm, question_compact):
            return True, alias
    return False, ""


def _find_candidates(
    question: str,
    table_columns: dict[str, dict[str, str]],
    allowed_tables: set[str] | None,
    selected_schema: str = "",
    vocab=None,
) -> list[dict]:
    qn = _norm(question)
    qc = _compact(question)
    selected_schema = (selected_schema or "").upper().strip()
    allowed_expanded: set[str] = set()
    for table in allowed_tables or set():
        allowed_expanded.update(_table_variants(str(table)))
    candidates: list[dict] = []
    for table, cols in table_columns.items():
        table_u = str(table).upper()
        if selected_schema:
            schema_name = _table_schema(table_u)
            if schema_name and schema_name != selected_schema:
                continue
        if allowed_tables is not None:
            variants = _table_variants(table_u)
            if not variants & allowed_expanded:
                continue
        for col, col_type in (cols or {}).items():
            col_u = str(col).upper()
            # Preserve the original casing for camel/Pascal tokenization while
            # the emitted physical column remains canonical uppercase.
            aliases = _aliases_for_column(str(col), vocab=vocab)
            matched, term = _column_matches_question(col_u, aliases, qn, qc)
            if not matched:
                continue
            candidates.append({
                "term": term,
                "table": table_u,
                "column": col_u,
                "role": _role_for_column(col_u, str(col_type)),
                "aliases": sorted(aliases),
            })
    return _drop_compound_modifier_matches(candidates, qn)


def _drop_compound_modifier_matches(candidates: list[dict], normalized_question: str) -> list[dict]:
    """Drop a term that is only the modifier half of a compound noun.

    English noun compounds are head-final: "product category" names a kind of
    category, not a kind of product. Once display columns answer to their
    entity noun, both halves match independently -- PRODUCT_NAME on "product"
    and CATEGORY_NAME on "category" -- and the generator groups by both.
    Live case 12 asked to allocate revenue "by product category" and came back
    split by product as well, a grain nobody requested.

    A single-word term is dropped when it sits immediately before another
    matched term in the question, i.e. it is modifying that term rather than
    naming a field of its own. Multi-word terms are left alone: they are
    already the specific phrase, not a fragment of one.
    """
    words = normalized_question.split()
    terms = {str(c.get("term") or "").strip() for c in candidates}
    terms.discard("")
    if len(terms) < 2:
        return candidates

    modifiers: set[str] = set()
    for term in terms:
        if " " in term:
            continue  # already a specific phrase
        for i, word in enumerate(words[:-1]):
            if word != term:
                continue
            # Head of the compound is whatever matched term follows it.
            for other in terms:
                if other == term:
                    continue
                head = other.split()
                if words[i + 1 : i + 1 + len(head)] == head:
                    modifiers.add(term)
                    break
            if term in modifiers:
                break

    if not modifiers:
        return candidates
    kept = [c for c in candidates if str(c.get("term") or "").strip() not in modifiers]
    if not kept:
        return candidates
    if len(kept) != len(candidates):
        log.info(
            "Dropped compound-modifier term(s) %s: they modify a following "
            "matched term rather than naming a field", sorted(modifiers),
        )
    return kept


def _choose_fields(
    question: str,
    candidates: list[dict],
    preferred_fact_tables: set[str] | None = None,
) -> list[dict]:
    if not candidates:
        return []
    base_tables = {
        c["table"]
        for c in candidates
        if c["role"] == "measure" and any(h in c["term"] for h in _MEASURE_HINTS)
    }
    chosen_by_term: dict[str, dict] = {}
    for c in candidates:
        key = c["term"]
        score = _score_candidate(
            c["table"], c["column"], c["role"], question, base_tables,
            preferred_fact_tables,
        )
        current = chosen_by_term.get(key)
        if not current or score > current["_score"]:
            c = dict(c)
            c["_score"] = score
            c["_tied_sources"] = {(_table_bare(c["table"]), c["column"])}
            chosen_by_term[key] = c
        elif score == current["_score"]:
            # Another source matched this term equally well — the pick below
            # is arbitrary (dict iteration order), so record the tie. The
            # plan will demote tied fields to enforcement="optional": a
            # wrong guess as a hard requirement rejects CORRECT SQL (e.g.
            # "doctors state" resolving to DIM_PATIENT.STATE when the SQL
            # rightly used DIM_PRESCRIBER.STATE), while a hint costs little.
            # Tie identity uses the BARE table name — schema catalogs list
            # the same table under multiple qualified forms (DB.SCHEMA.T,
            # SCHEMA.T, T), and those are one source, not an ambiguity.
            current["_tied_sources"].add((_table_bare(c["table"]), c["column"]))

    # Longest-match-wins: a term whose every occurrence in the question sits
    # inside a longer chosen term is not an independent mention — "insurance
    # coverage type" mentions "coverage type", not the bare "type" column of
    # some unrelated table. Only drop when ALL occurrences are covered:
    # "revenue by item and item group" keeps both "item" and "item group"
    # because the first "item" stands alone.
    qn = _norm(question)
    term_spans = {term: _alias_occurrence_spans(term, qn) for term in chosen_by_term}
    for term in list(chosen_by_term):
        spans = term_spans.get(term) or []
        if not spans:
            continue  # compact/technical match — can't locate it, keep it
        longer_spans = [
            span
            for other, other_spans in term_spans.items()
            if other != term and len(other) > len(term)
            for span in other_spans
        ]
        if longer_spans and all(
            any(ls[0] <= s[0] and s[1] <= ls[1] for ls in longer_spans)
            for s in spans
        ):
            chosen_by_term.pop(term)

    fields = list(chosen_by_term.values())
    fields.sort(key=lambda f: (f["role"] != "measure", f["term"], f["table"], f["column"]))
    for f in fields:
        f.pop("_score", None)
        tied = f.pop("_tied_sources", set())
        if len(tied) > 1:
            f["ambiguous_source"] = True
    return fields[:8]


def _question_asks_for_key(term: str, question: str) -> bool:
    q = _norm(question)
    term_norm = _norm(term)
    if not term_norm:
        return False
    # "sk"/"fk" belong here now that those suffixes are recognised as keys:
    # a user who asks for "warehouse sk" wants the key, not the display name.
    key_words = ("key", "id", "identifier", "number", "sk", "fk")
    return any(
        re.search(rf"(?<![a-z0-9]){re.escape(term_norm)}\s+{word}(?![a-z0-9])", q)
        or re.search(rf"(?<![a-z0-9]){word}\s+{re.escape(term_norm)}(?![a-z0-9])", q)
        for word in key_words
    )


def _question_asks_for_code(term: str, question: str) -> bool:
    q = _norm(question)
    term_norm = _norm(term)
    if not term_norm:
        return False
    return bool(
        re.search(rf"(?<![a-z0-9]){re.escape(term_norm)}\s+code(?![a-z0-9])", q)
        or re.search(rf"(?<![a-z0-9])code\s+{re.escape(term_norm)}(?![a-z0-9])", q)
    )


# Surrogate/foreign-key suffixes, longest first so "_DMS_KEY" wins over "_KEY".
# Deliberately excludes _CODE/_CD/_NO/_NUM: those are display or degenerate
# columns, and treating them as keys would make a column its own display field.
_KEY_SUFFIXES = ("_DMS_KEY", "_KEY", "_SK", "_FK", "_ID")


def _is_key_column(column: str) -> bool:
    col = (column or "").upper()
    return any(col.endswith(suffix) and len(col) > len(suffix) for suffix in _KEY_SUFFIXES)


def _key_prefix(column: str) -> str:
    """Strip a surrogate-key suffix to get the entity name it points at.

    Recognises the Kimball convention (_SK), the generic one (_ID/_FK/_KEY)
    and the Infor M3 one (_DMS_KEY). Previously only the last two were
    handled, so WAREHOUSE_SK yielded no prefix and the display-name upgrade
    below could never fire on a standard star schema -- warehouses and
    categories were reported to users as raw surrogate keys (10, 20, 30).
    """
    col = (column or "").upper()
    for suffix in _KEY_SUFFIXES:
        if col.endswith(suffix) and len(col) > len(suffix):
            return col[: -len(suffix)]
    return ""


def _table_allowed_for_display(table: str, allowed_tables: set[str] | None, selected_schema: str = "") -> bool:
    table_u = (table or "").upper()
    if selected_schema:
        schema_name = _table_schema(table_u)
        if schema_name and schema_name != selected_schema.upper():
            return False
    if allowed_tables is None:
        return True
    allowed_expanded: set[str] = set()
    for allowed in allowed_tables:
        allowed_expanded.update(_table_variants(str(allowed)))
    return bool(_table_variants(table_u) & allowed_expanded)


def _display_table_score(table: str, key_prefix: str) -> int:
    bare = _table_bare(table)
    score = 0
    if bare == f"{key_prefix}_DMS":
        score += 12
    # An exact dimension-table match under the common naming conventions
    # deserves the same weight as the M3 one above: D_WAREHOUSE for
    # WAREHOUSE_SK is as strong a signal as WAREHOUSE_DMS for WAREHOUSE_DMS_KEY.
    if key_prefix and bare in {f"D_{key_prefix}", f"DIM_{key_prefix}", key_prefix}:
        score += 12
    if bare.startswith(("DIM_", "D_")):
        score += 8
    if "FCT" not in bare and "FACT" not in bare:
        score += 4
    if key_prefix and key_prefix in bare:
        score += 3
    return score


def _find_display_field_for_key(
    key_column: str,
    term: str,
    question: str,
    table_columns: dict[str, dict[str, str]],
    allowed_tables: set[str] | None,
    selected_schema: str = "",
) -> dict | None:
    key_col = (key_column or "").upper()
    prefix = _key_prefix(key_col)
    if not prefix or not _is_key_column(key_col):
        return None
    if _question_asks_for_key(term, question):
        return None

    wants_code = _question_asks_for_code(term, question)
    display_candidates = [f"{prefix}_DSC", f"{prefix}_DESC", f"{prefix}_DESCRIPTION", f"{prefix}_NAME", f"{prefix}_NM"]
    code_candidates = [f"{prefix}_CD", f"{prefix}_CODE"]
    preferred = code_candidates + display_candidates if wants_code else display_candidates + code_candidates

    matches: list[dict] = []
    for table, cols in table_columns.items():
        cols_u = {str(c).upper() for c in (cols or {})}
        if key_col not in cols_u:
            continue
        if not _table_allowed_for_display(table, allowed_tables, selected_schema):
            continue
        for idx, display_col in enumerate(preferred):
            if display_col not in cols_u:
                continue
            matches.append({
                "table": table,
                "column": display_col,
                "source_key_column": key_col,
                "_score": _display_table_score(table, prefix) + (100 - idx),
            })
            break

    if not matches:
        return None
    matches.sort(key=lambda m: m["_score"], reverse=True)
    winner = dict(matches[0])
    winner.pop("_score", None)
    return winner


def _apply_display_dimension_fields(
    fields: list[dict],
    question: str,
    table_columns: dict[str, dict[str, str]],
    allowed_tables: set[str] | None,
    selected_schema: str = "",
) -> list[dict]:
    out: list[dict] = []
    for field in fields:
        col = (field.get("column") or "").upper()
        if field.get("role") == "dimension" and _is_key_column(col):
            display = _find_display_field_for_key(
                col,
                field.get("term") or "",
                question,
                table_columns,
                allowed_tables,
                selected_schema,
            )
            if display:
                upgraded = dict(field)
                upgraded.update({
                    "table": display["table"],
                    "column": display["column"],
                    "role": "display_dimension",
                    "source_key_column": display["source_key_column"],
                    "source_key_table": field.get("table", ""),
                    "display_required": True,
                })
                out.append(upgraded)
                continue
        out.append(field)
    return out


def _join_edges(table_columns: dict[str, dict[str, str]], vocab=None) -> dict[str, list[dict]]:
    join_synonyms = _planner_vocab(vocab).join_synonyms
    tables = {str(t).upper(): {str(c).upper() for c in (cols or {})} for t, cols in table_columns.items()}
    graph: dict[str, list[dict]] = {t: [] for t in tables}
    table_list = list(tables)
    for i, left in enumerate(table_list):
        for right in table_list[i + 1:]:
            conditions: list[tuple[str, str]] = []
            common = sorted(tables[left] & tables[right])
            # DIVI is a grouping/filter dimension, not a relational key — exclude it
            # from join conditions so it doesn't create false graph edges.
            conditions.extend((c, c) for c in common if c.endswith("_DMS_KEY") or c in {"CONO", "ORNO", "PONR", "POSX", "DLIX"})
            for lcol, rcols in join_synonyms.items():
                if lcol in tables[left]:
                    conditions.extend((lcol, rc) for rc in rcols if rc in tables[right])
                if lcol in tables[right]:
                    conditions.extend((rc, lcol) for rc in rcols if rc in tables[left])
            seen = set()
            deduped = []
            for cond in conditions:
                if cond not in seen:
                    seen.add(cond)
                    deduped.append(cond)
            if deduped:
                graph[left].append({"to": right, "conditions": deduped[:5]})
                graph[right].append({"to": left, "conditions": [(r, l) for l, r in deduped[:5]]})
    return graph


def _shortest_join_path(source: str, target: str, graph: dict[str, list[dict]]) -> list[dict]:
    if source == target:
        return []
    queue = deque([(source, [])])
    seen = {source}
    while queue:
        table, path = queue.popleft()
        for edge in graph.get(table, []):
            nxt = edge["to"]
            if nxt in seen:
                continue
            next_path = path + [{"from": table, "to": nxt, "conditions": edge["conditions"]}]
            if nxt == target:
                return next_path
            seen.add(nxt)
            queue.append((nxt, next_path))
    return []


def _build_required_joins(fields: list[dict], table_columns: dict[str, dict[str, str]], vocab=None) -> list[dict]:
    tables = []
    for f in fields:
        if f["table"] not in tables:
            tables.append(f["table"])
    if len(tables) <= 1:
        return []
    graph = _join_edges(table_columns, vocab=vocab)
    anchor = next((f["table"] for f in fields if f["role"] == "measure"), tables[0])

    # For display_dimension fields we already know the exact join key (source_key_column).
    # _join_edges uses a shared-column heuristic and can pick up OTHER _DMS_KEY columns
    # that the dimension table itself holds as FKs (e.g. WHS_DMS also has FCY_DMS_KEY),
    # creating multi-condition join plans the LLM won't reproduce.  Pin the condition to
    # just the authoritative key so the validator gets exactly one equality to enforce.
    display_key_map: dict[str, str] = {
        f["table"].upper(): f["source_key_column"].upper()
        for f in fields
        if f.get("display_required") and f.get("source_key_column")
    }

    joins: list[dict] = []
    seen_edges: set[tuple[str, str]] = set()
    for table in tables:
        if table == anchor:
            continue
        for edge in _shortest_join_path(anchor, table, graph):
            key = tuple(sorted([edge["from"], edge["to"]]))
            if key in seen_edges:
                continue
            seen_edges.add(key)
            target_u = edge["to"].upper()
            if target_u in display_key_map:
                # Override: use only the single authoritative FK, not all shared keys.
                edge = dict(edge)
                key_col = display_key_map[target_u]
                edge["conditions"] = [(key_col, key_col)]
            joins.append(edge)
    return joins


# Generic date-part columns on a date dimension. Matching "by month" / "in
# year 2024" to DT_DMS.MONTH / DT_DMS.YEAR is a useful hint, but must NOT be
# hard-required: the system prompt's DATE-KEY RULE teaches the LLM to bucket
# periods directly from the fact table's YYYYMMDD key (FORMAT(TRY_CONVERT(...))),
# which is equally valid SQL that never touches the dimension's date-part
# column — enforcing it as required hard-blocks those correct queries.
_DATE_PART_COLUMNS = {"DAY", "MONTH", "YEAR", "QUARTER", "WEEK", "SEMESTER", "HALF"}


def _anchor_fields_to_measure_fact(
    fields: list[dict],
    fact_tables: set[str] | None,
) -> str:
    """Phase 3 — the resolved measure's fact table is authoritative.

    Terms are scored per-term across every allowed table, so a generic word
    like "warehouse" or "date" can win on whichever table happens to score
    highest — including a fact table that has nothing to do with the measure.
    Live failure: "What is total revenue by warehouse?" bound

        Revenue   -> F_SALES_INVOICE.NET_REVENUE_AMOUNT      (correct)
        warehouse -> ERP_ITM_BAL_PRD_FCT.WHS_DMS_KEY         (wrong fact)

    Requiring both forces a fact-to-fact join, which duplicates every invoice
    row once per inventory row for the same warehouse.

    Rule: once a measure resolves, its fact is locked. A non-measure field
    sitting on a *different* fact is not evidence about this question — it is
    demoted to a hint so it can never hard-require the second fact. Dimension
    tables are untouched: joining the measure's fact to a dimension is the
    normal star path.

    Returns the locked fact (upper-cased), or "" when nothing was locked.
    """
    if not fields:
        return ""
    known_facts = {str(t).upper() for t in (fact_tables or set()) if str(t or "").strip()}
    if not known_facts:
        return ""

    def _is_fact(table_name: str) -> bool:
        name = str(table_name or "").upper()
        if not name:
            return False
        parts = name.split(".")
        for fact in known_facts:
            fact_parts = fact.split(".")
            if len(parts) >= 2 and len(fact_parts) >= 2:
                if parts[-2:] == fact_parts[-2:]:
                    return True
            elif parts[-1:] == fact_parts[-1:]:
                return True
        return False

    anchor = next(
        (
            str(f.get("table") or "").upper()
            for f in fields
            if f.get("role") == "measure" and _is_fact(f.get("table"))
        ),
        "",
    )
    if not anchor:
        return ""

    demoted: list[str] = []
    for field in fields:
        table_u = str(field.get("table") or "").upper()
        if (
            field.get("role") != "measure"
            and field.get("enforcement") != "optional"
            and _is_fact(table_u)
            and not _same_physical_table(table_u, anchor)
        ):
            field["enforcement"] = "optional"
            field["demoted_reason"] = (
                f"measure-first anchoring: fact is {anchor}"
            )
            demoted.append(f"{table_u}.{str(field.get('column') or '').upper()}")

    if demoted:
        log.info(
            "Measure-first anchoring locked fact %s; demoted rival-fact "
            "field(s) to optional: %s", anchor, demoted,
        )
    return anchor


def _same_physical_table(left: str, right: str) -> bool:
    left_parts = (left or "").upper().split(".")
    right_parts = (right or "").upper().split(".")
    if len(left_parts) >= 2 and len(right_parts) >= 2:
        return left_parts[-2:] == right_parts[-2:]
    return left_parts[-1:] == right_parts[-1:]


def _demote_measures_with_a_rival_at_the_requested_grain(
    fields: list[dict],
    question: str,
    table_columns: dict[str, dict[str, str]],
) -> None:
    """Stop hard-requiring a measure when the question asks for another grain.

    Live case 6 -- "month-end inventory value for February 2026" -- bound
    "inventory value" to F_INVENTORY_DAILY.INVENTORY_VALUE and required it.
    Nothing here consults the grain the question asked for, so a month-end
    question was pinned to the daily snapshot fact. The generator correctly
    wrote F_INVENTORY_MONTHLY.ENDING_INVENTORY_VALUE and was refused.

    The rival is not found by name equality: the monthly column is
    ENDING_INVENTORY_VALUE, which is why plain ambiguity detection missed it
    and the exact match won outright. Match on containment instead -- another
    fact carrying a column that *contains* this measure's name is a plausible
    alternative source for the same quantity.

    Demote rather than re-point: choosing the rival would mean guessing which
    of its columns is the right one. Demoting lets generation use either,
    which is all that is needed -- a wrong hard requirement rejects correct
    SQL, a wrong hint merely fails to help.

    Only fires when the question names a grain. Grain-free questions keep the
    exact binding hard-required exactly as before.
    """
    # Imported here: core.contextual_dates imports from this module, so a
    # top-level import would be circular.
    from core.contextual_dates import requested_temporal_grain

    grain = requested_temporal_grain(question)
    if not grain:
        return
    for field in fields:
        if field.get("enforcement") == "optional":
            continue
        if str(field.get("role") or "").strip().lower() not in {"measure", "measure_candidate"}:
            continue
        column = str(field.get("column") or "").upper()
        home = str(field.get("table") or "").upper()
        if not column or not home:
            continue
        rivals = [
            table for table, cols in (table_columns or {}).items()
            if str(table).upper() != home
            and any(column in str(c).upper() and str(c).upper() != column for c in (cols or {}))
        ]
        if rivals:
            field["enforcement"] = "optional"
            field["demoted_reason"] = "rival_measure_at_requested_grain"
            log.info(
                "Demoted required measure %s.%s to optional: the question asks "
                "for %s grain and %s also carries a column containing %s",
                home, column, grain, sorted(rivals)[:3], column,
            )


def build_semantic_field_plan(
    question: str,
    table_columns: dict[str, dict[str, str]] | None,
    allowed_tables: set[str] | None = None,
    selected_schema: str = "",
    vocab=None,
    fact_tables: set[str] | None = None,
    preferred_fact_tables: set[str] | None = None,
) -> dict:
    """Build a conservative field-source plan from exact known schema columns.

    `fact_tables` carries the model's fact classifications so the resolved
    measure's fact can be locked (Phase 3). Omitting it preserves the previous
    behaviour exactly — no anchoring is attempted without table roles.
    """
    vocab = _planner_vocab(vocab)
    normalized_columns = {
        str(t).upper(): {str(c).upper(): str(v) for c, v in (cols or {}).items()}
        for t, cols in (table_columns or {}).items()
    }
    candidates = _find_candidates(question, normalized_columns, allowed_tables, selected_schema, vocab=vocab)
    fields = _choose_fields(question, candidates, preferred_fact_tables)
    fields = _apply_display_dimension_fields(fields, question, normalized_columns, allowed_tables, selected_schema)
    if not fields:
        return {"enabled": False, "fields": [], "joins": [], "reason": "no matching semantic fields"}
    for field in fields:
        if (field.get("column") or "").upper() in _DATE_PART_COLUMNS:
            field["enforcement"] = "optional"
        # A term that tied across multiple source tables/columns was resolved
        # by arbitrary order, not evidence — keep it as a hint, never a hard
        # requirement (a wrong hard requirement rejects correct SQL).
        if field.get("ambiguous_source"):
            field["enforcement"] = "optional"
    _demote_measures_with_a_rival_at_the_requested_grain(
        fields, question, normalized_columns
    )
    # Phase 3: lock the fact to the resolved measure before joins are built,
    # so a rival fact's field can never contribute a required join edge.
    _locked_fact = _anchor_fields_to_measure_fact(fields, fact_tables)

    joins = _build_required_joins(fields, normalized_columns, vocab=vocab)
    # Join edges that exist only to reach optional (date-part) fields must be
    # optional too — otherwise the field is skippable but its join still blocks.
    _required_fields = [f for f in fields if f.get("enforcement") != "optional"]
    _required_edge_keys = {
        (e["from"], e["to"])
        for e in _build_required_joins(_required_fields, normalized_columns, vocab=vocab)
    }
    for edge in joins:
        if (edge["from"], edge["to"]) not in _required_edge_keys:
            edge["enforcement"] = "optional"
    # Required tables follow enforcement: a demoted rival fact is a hint, and
    # listing it as required would drag it back into retrieval and the prompt.
    required_tables = sorted({
        f["table"] for f in fields if f.get("enforcement") != "optional"
    }) or sorted({f["table"] for f in fields})
    return {
        "enabled": True,
        "fields": fields,
        "joins": joins,
        "required_tables": required_tables,
        "fact_anchor": _locked_fact,
        "reason": "schema-derived semantic field plan",
    }


def format_semantic_field_plan(plan: dict, db_type: str = "azure_sql") -> str:
    from core.contextual_dates import (
        format_date_value_expression,
        format_period_bucket_expression,
        format_required_anchor,
    )

    if not plan:
        return ""
    fields = list(plan.get("fields") or [])
    source_scope = plan.get("source_scope") or {}
    analytical_request_plan = plan.get("analytical_request_plan") or {}
    if not fields and not source_scope.get("selected_fact") and not analytical_request_plan:
        return ""
    lines = [
        "## Semantic field-source plan",
        "Use these exact source fields when the question mentions the mapped business terms.",
        "Do not move a mapped column to another table and do not remove underscores from column names.",
    ]
    if fields:
        lines.extend(["", "Resolved fields:"])
    if source_scope.get("selected_fact"):
        lines.extend([
            "",
            "Authoritative measure source:",
            f"- {source_scope['selected_fact']} is the single fact for measures in this request.",
            "- Do not substitute or directly join another fact table. Reach descriptive attributes only through governed dimensions.",
        ])
    if analytical_request_plan:
        try:
            from core.analytical_request_plan import format_analytical_request_plan
            request_text = format_analytical_request_plan(analytical_request_plan)
        except Exception:
            request_text = ""
        if request_text:
            lines.extend(["", request_text])
    for field in fields:
        role_alias = str(field.get("role_alias") or "").strip()
        if role_alias and field.get("date_key_type") == "surrogate_fk":
            # The physical dimension table and its role-playing alias are not
            # interchangeable once the table is joined with AS. Keep every
            # instruction internally consistent so generation can copy it.
            expr = f"{role_alias}.{field['column']}"
        else:
            expr = f"{field['table']}.{field['column']}"
        if field.get("date_key_type") in {"yyyymmdd_integer", "yyyymm_integer"}:
            expr = format_date_value_expression(
                str(field.get("table") or ""),
                str(field.get("column") or ""),
                str(field.get("date_key_type") or ""),
                db_type,
            )
        # Show a non-binding hint for measures — the LLM should aggregate only when
        # the query is aggregating (not for row-level queries like "show all invoices").
        role_hint = " [measure — apply SUM/COUNT only if aggregating]" if field.get("role") == "measure" else ""
        if field.get("display_required"):
            role_hint = (
                " [business display field - use this in SELECT and GROUP BY; "
                f"use {field.get('source_key_column')} only for JOINs unless the user asks for key/id]"
            )
        if role_alias and field.get("date_key_type") == "surrogate_fk":
            role_hint += (
                f" [role-playing date alias: {role_alias}; "
                "use the real date value through this alias, never parse the fact FK as a date]"
            )
        elif field.get("date_key_type") in {"yyyymmdd_integer", "yyyymm_integer"}:
            role_hint += (
                f" [decoded {field.get('temporal_grain') or 'calendar'} date; "
                "use this exact nullable expression and exclude invalid encoded values]"
            )
        lines.append(f"- {field['term']}: {expr}{role_hint}")
        requested_grain = str(field.get("requested_grain") or "").strip()
        if requested_grain and field.get("role") == "contextual_date":
            bucket = format_period_bucket_expression(
                expr,
                requested_grain,
                db_type,
                role_alias=role_alias,
                calendar_attributes=field.get("calendar_attributes") or {},
            )
            lines.append(
                f"  REQUIRED {requested_grain.upper()} BUCKET: {bucket}. "
                "Use it in SELECT/GROUP BY for the requested period; use the native date value above for filtering and MAX anchor scope."
            )
    avoid = plan.get("avoid_columns") or []
    if avoid:
        lines.append("")
        lines.append("Superseded columns (admin-approved mappings replace these):")
        for entry in avoid:
            term = str(entry.get("term") or "this term").strip()
            lines.append(
                f"- Do NOT use {entry['table']}.{entry['column']} for \"{term}\" — "
                f"the admin-approved source is {entry.get('use_instead_table')}.{entry.get('use_instead_column')}."
            )
    joins = plan.get("joins") or []
    if joins:
        lines.append("")
        lines.append("Required join path:")
        for edge in joins:
            role_alias = str(edge.get("role_alias") or "").strip()
            right_ref = role_alias or edge["to"]
            conds = " AND ".join(
                f"{edge['from']}.{left_col} = {right_ref}.{right_col}"
                for left_col, right_col in edge.get("conditions", [])
            )
            alias_clause = f" AS {role_alias}" if role_alias else ""
            role_clause = f" for {edge.get('business_role')}" if edge.get("business_role") else ""
            join_keyword = "LEFT JOIN" if edge.get("preserve_all") else "JOIN"
            lines.append(
                f"- {edge['from']} {join_keyword} {edge['to']}{alias_clause} ON {conds}{role_clause}"
            )
    temporal_policies = plan.get("temporal_policies") or []
    if temporal_policies:
        lines.append("")
        lines.append("Relative-date anchor policy:")
        lines.append(
            "Relative words such as today, yesterday, last N days, this month, "
            "and this year are relative to the latest available governed business "
            "date in the data, not the application server clock."
        )
        for policy in temporal_policies:
            date_ref = (
                f"{policy.get('role_alias')}.{policy.get('date_column')}"
                if (
                    policy.get("role_alias")
                    and str(policy.get("date_key_type") or "") == "surrogate_fk"
                )
                else f"{policy.get('date_table')}.{policy.get('date_column')}"
            )
            if policy.get("date_key_type") in {"yyyymmdd_integer", "yyyymm_integer"}:
                date_ref = format_date_value_expression(
                    str(policy.get("date_table") or policy.get("fact_table") or ""),
                    str(policy.get("date_column") or policy.get("fact_column") or ""),
                    str(policy.get("date_key_type") or ""),
                    db_type,
                )
            if policy.get("kind") == "latest_n_observed":
                lines.append(
                    f"- {policy.get('business_role') or 'Business date'}: select exactly the latest "
                    f"{int(policy.get('amount') or 1)} DISTINCT observed {policy.get('unit') or 'period'} "
                    f"value(s) from the governed fact rows using {date_ref}, ordered descending. "
                    "Filter the answer to that selected period set and group by the real business date; "
                    "do not turn this into MAX(date)-N calendar arithmetic and do not return one scalar total."
                )
            else:
                lines.append(
                    f"- {policy.get('business_role') or 'Business date'}: derive the anchor "
                    f"with MAX({date_ref}) over the same governed source rows, then apply "
                    f"the requested {policy.get('kind')} boundary from that anchor."
                )
            if policy.get("date_key_type") in {"yyyymmdd_integer", "yyyymm_integer"}:
                encoded_expr = format_date_value_expression(
                    str(policy.get("date_table") or policy.get("fact_table") or ""),
                    str(policy.get("date_column") or policy.get("fact_column") or ""),
                    str(policy.get("date_key_type") or ""),
                    db_type,
                )
                lines.append(
                    f"  PHYSICAL ENCODING: {policy.get('date_key_type')} at "
                    f"{policy.get('temporal_grain') or 'calendar'} grain. Use "
                    f"{encoded_expr}; invalid/sentinel integer values must resolve "
                    "to NULL and be excluded."
                )
            if policy.get("kind") == "latest_snapshot":
                lines.append(
                    "  SNAPSHOT RULE: filter the decoded business date equal to the "
                    "required MAX anchor; do not sum inventory/balance values across periods."
                )
            _anchor = "" if policy.get("kind") == "latest_n_observed" else format_required_anchor(policy, db_type)
            if _anchor:
                lines.append(
                    f"  REQUIRED ANCHOR (copy this exact subquery as the anchor; do not "
                    f"build your own): {_anchor}"
                )
        lines.append(
            "- Do not use GETDATE(), CURRENT_DATE, CURRENT_TIMESTAMP, SYSDATE, or NOW() "
            "for these relative periods."
        )
        lines.append(
            "- Never anchor on MAX(CALENDAR_DATE) over an unrestricted date dimension: "
            "calendar tables include future rows with no matching fact records, which "
            "silently yields zero results."
        )
    return "\n".join(lines)
