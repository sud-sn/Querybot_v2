"""Metric selection helpers for multi-schema questions.

The metric registry is account-wide, but SQL generation needs a narrower view:
when a question mentions a schema-specific dimension/entity, only metrics from
that same schema should be enforced.  This module keeps that scoping logic
small and deterministic so generic names like "Revenue" do not override a more
specific schema-local metric such as "Total Revenue USD".
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from core.source_resolution import GENERIC_MEASURE_WORDS
from core.word_forms import (
    FRENCH_FUNCTION_WORDS, FRENCH_GENERIC_MEASURE_WORDS, base_form, with_one_quantity_word, without_grain_or_window,
)


log = logging.getLogger(__name__)

_GENERIC_WORDS = GENERIC_MEASURE_WORDS | FRENCH_GENERIC_MEASURE_WORDS

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "by", "each", "find", "for", "from",
    "give", "how", "in", "is", "me", "my", "of", "on", "or", "per", "show",
    "the", "to", "total", "what", "with",
}


@dataclass
class MetricScopeResult:
    metrics: list[dict[str, Any]]
    ambiguous: bool = False
    options: list[str] | None = None
    reason: str = ""
    context_schemas: set[str] | None = None


def _norm(text: str) -> str:
    """The text's words, lower-case and without accents: "unités vendues" is
    "unites vendues". Shredded at the accent instead, it was "unit s
    vendues", and its "unit" matched every "unit price" asked for."""
    import unicodedata

    folded = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", folded.lower()).strip()


def _tokens(text: str) -> set[str]:
    return {
        base_form(tok)
        for tok in re.findall(r"[a-z0-9]+", _norm(text))
        if len(tok) > 1 and tok not in _STOP_WORDS and tok not in FRENCH_FUNCTION_WORDS
    }


def _table_schema(table: str) -> str:
    parts = re.split(r"[.\[\]\"]+", (table or "").upper())
    parts = [p for p in parts if p]
    return parts[-2] if len(parts) >= 2 else ""



def _metric_phrases(metric: dict[str, Any]) -> list[str]:
    phrases: list[str] = []
    for key in ("name", "synonyms", "example_questions"):
        value = str(metric.get(key) or "")
        for part in re.split(r"[,;\n]+", value):
            part = _norm(part)
            if part:
                phrases.append(part)
    return list(dict.fromkeys(phrases))


# "Stock" and "inventory" said alone: the stock on hand's own words
# (core/starter_metrics.py), and part of every other stock measure's name.
_BARE_STOCK_WORDS = ("stock", "stocks", "inventory", "inventories", "inventaire", "inventaires")


def _is_bare(phrase: str) -> bool:
    """A metric phrase that is nothing but a bare word for stock."""
    words = _tokens(phrase)
    return bool(words) and words <= _tokens(" ".join(_BARE_STOCK_WORDS))


def _bare_word_stands_alone(metric: dict[str, Any], metrics: list[dict[str, Any]], *texts: str) -> bool:
    """Whether a bare "stock" the question says is this metric's to claim:
    the question says no word of another metric's that this one does not --
    "value of our inventory", "available inventory", "inventory sold" each
    name a measure of their own, and a bare word answered them with the
    stock on hand. Grains and windows are not such words."""
    asked: set[str] = set()
    for text in texts:
        if text:
            asked |= _tokens(without_grain_or_window(text))
    own = {word for phrase in _metric_phrases(metric) for word in _tokens(phrase)}
    others = {word for other in metrics or [] if other is not metric
              for phrase in _metric_phrases(other) for word in _tokens(phrase)}
    return not ((asked & others) - own - _tokens(" ".join(_BARE_STOCK_WORDS)))


def _named_spans(metric: dict[str, Any], question: str, *, bare_ok: bool = True) -> list[tuple[int, int]]:
    """Where the question says one of the metric's own phrases, whole."""
    q = _norm(question)
    return [
        found.span()
        for phrase in _metric_phrases(metric)
        if bare_ok or not _is_bare(phrase)
        for found in re.finditer(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", q)
    ]


def _phrase_score(metric: dict[str, Any], question: str, *, reader_question: str = "",
                  bare_ok: bool = True) -> int:
    """How strongly this metric's wording matches the question.

    Scored against the canonical English AND, when they differ, the reader's own
    words -- taking the better of the two. Both are needed and neither is
    sufficient. _norm keeps only [a-z0-9], so an accented French word is SHREDDED
    rather than merely unmatched ("annee" survives, "année" becomes "ann e"), and
    measured on a French reader this function returned no metric at all where its
    English twin returned one. Scoring the canonical form alone would then have
    made a French-authored synonym unreachable in turn, which is the opposite
    failure on the same axis.
    """
    best = _phrase_score_one(metric, question, bare_ok=bare_ok)
    if reader_question and reader_question != question:
        best = max(best, _phrase_score_one(metric, reader_question, bare_ok=bare_ok))
    return best


def _phrase_score_one(metric: dict[str, Any], question: str, *, bare_ok: bool = True) -> int:
    q = _norm(question)
    # A grain or a window is not evidence of which measure: the exact phrase
    # below still reads the whole question ("revenue this month" authored for a
    # month-to-date metric is a phrase), the word overlap does not.
    q_tokens = _tokens(without_grain_or_window(question))
    if not q_tokens:
        return 0
    best = 0
    for phrase in _metric_phrases(metric):
        phrase_tokens = _tokens(phrase)
        if not phrase_tokens or (not bare_ok and _is_bare(phrase)):
            continue
        overlap = q_tokens & phrase_tokens
        # Overlap on nothing but generic quantity words is not evidence about
        # WHICH metric was meant. "show me the inventory value by warehouse"
        # shared exactly one token with "purchase order value" -- "value" --
        # and scored 10, enough to be the only matched metric, because the
        # only floor anywhere in this matcher is score > 0. That pinned
        # PCH_ORD_RCT_FCT as the measure fact and the validator then rejected
        # the model's correct ITM_BAL_PRD_FCT SQL with source_fact_mismatch.
        #
        # Real matches are never this thin: the same metric scores 166 on
        # "total amount of confirmed purchase orders by profit center" and
        # Revenue scores 122 on "show total revenue by profit centre", both on
        # subject words. Dropping generic-only overlap cannot reach them.
        #
        # The exact-phrase bonus below is deliberately still allowed: a metric
        # literally named "Total Value" appearing verbatim in the question IS
        # evidence, and that path requires the whole phrase, not one token.
        if overlap and overlap <= _GENERIC_WORDS:
            overlap = set()
        # Nor is a bare "stock" that another metric's word beside it claims:
        # "inventory sold" shares "inventory" with every stock measure, and
        # "sold" with the units sold alone.
        if overlap and not bare_ok and overlap <= _GENERIC_WORDS | _tokens(" ".join(_BARE_STOCK_WORDS)):
            overlap = set()
        score = len(overlap) * 10
        # The whole phrase, whichever word for a quantity either says it in.
        if re.search(rf"(?<![a-z0-9]){re.escape(with_one_quantity_word(phrase))}(?![a-z0-9])",
                     with_one_quantity_word(q)):
            score += 100 + len(phrase_tokens) * 12
        elif len(phrase_tokens) > 1 and overlap == phrase_tokens:
            # Every word of the phrase, in another order, is one more word of
            # evidence: "la valeur de notre stock" has all of "valeur du stock"
            # and half of "valeur du stock en fin de mois", whose other words
            # name a narrower measure nobody asked for. Both scored 20, and the
            # tie was broken by the metrics' names.
            score += 10
        best = max(best, score)
    metadata_tokens = _tokens(" ".join([
        str(metric.get("required_columns") or ""),
        str(metric.get("allowed_dimensions") or ""),
        str(metric.get("grain") or ""),
        str(metric.get("category") or ""),
    ]))
    return best + len(q_tokens & metadata_tokens)


def _narrower_than_asked(metric: dict[str, Any], question: str, *, reader_question: str = "") -> set[str]:
    """The words this metric's matching phrases say and the question does not.

    A phrase that shares a word with the question but says more names a
    narrower measure than the one asked: "back orders" to "orders by
    warehouse", "valeur du stock en fin de mois" to "valeur du stock". Empty
    when any phrase the question shares a word with is said whole -- the
    metric is then named, not merely neighboured. Generic words for a quantity
    or a value narrow nothing and are not counted as unsaid.
    """
    unsaid_words: set[str] = set()
    for text in dict.fromkeys(t for t in (question, reader_question) if t):
        said = _tokens(with_one_quantity_word(text))
        asked = _tokens(without_grain_or_window(text))
        for phrase in _metric_phrases(metric):
            words = _tokens(with_one_quantity_word(phrase))
            overlap = asked & words
            if not overlap or overlap <= _GENERIC_WORDS:
                continue
            unsaid = words - said - _GENERIC_WORDS
            if not unsaid:
                return set()
            unsaid_words |= unsaid
    return unsaid_words


# The states a quantity of stock or of an order is kept in apart from the
# quantity itself -- back-ordered, reserved, allocated -- which the starter
# metrics keep as measures of their own (core/starter_metrics.py). As _tokens
# reads them: folded, and a verb at its base ("allocated" is "allocate");
# French "commandes en souffrance" and "reliquats" are back orders, "réservé"
# and "alloué" their participles.
_STATE_WORDS = frozenset({
    "back", "backorder", "backorders", "backordered", "souffrance", "reliquat", "reliquats",
    "reserved", "reserve", "reservee", "reserves", "reservees",
    "allocate", "alloue", "allouee", "alloues", "allouees", "commit", "committed",
})


def _measure_columns_in_plan(semantic_plan: dict[str, Any] | None) -> set[str]:
    """The measure columns the field plan found in the question's own words."""
    return {
        str(field.get("column") or "").upper()
        for field in (semantic_plan or {}).get("fields") or []
        if str(field.get("role") or "") == "measure" and field.get("column")
    }


def _columns_read_by(metric: dict[str, Any]) -> set[str]:
    return (
        _split_required_columns(str(metric.get("required_columns") or ""))
        | _split_required_columns(str(metric.get("sql_template") or ""))
    )


def _split_required_columns(raw: str) -> set[str]:
    ignore = {
        "AS", "CASE", "CAST", "COALESCE", "ELSE", "END", "ISNULL", "NULLIF",
        "SUM", "THEN", "WHEN",
    }
    cols: set[str] = set()
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", raw or ""):
        token_u = token.upper()
        if token_u not in ignore:
            cols.add(token_u)
    return cols


def _sql_tables(sql: str) -> set[str]:
    tables: set[str] = set()
    pattern = re.compile(
        r"\b(?:FROM|JOIN)\s+((?:\[[^\]]+\]|\w+)(?:\s*\.\s*(?:\[[^\]]+\]|\w+)){0,2})",
        re.IGNORECASE,
    )
    for match in pattern.finditer(sql or ""):
        raw = re.sub(r"\s+", "", match.group(1))
        raw = raw.replace("[", "").replace("]", "").replace('"', "")
        if raw:
            tables.add(raw.upper())
    return tables


def _canonical_table_references(tables: set[str]) -> set[str]:
    """Collapse bare, schema, and database-qualified aliases of one table.

    Schema-qualified names are preferred because they are portable across
    database/catalog deployments and avoid injecting the same KB document
    several times under equivalent spellings.
    """
    cleaned = {
        str(table or "").replace("[", "").replace("]", "").replace('"', "").upper()
        for table in tables
        if str(table or "").strip()
    }
    qualified_by_bare: dict[str, list[str]] = {}
    for table in cleaned:
        parts = [part for part in table.split(".") if part]
        if len(parts) >= 2:
            qualified_by_bare.setdefault(parts[-1], []).append(table)

    result: set[str] = set()
    handled_bare: set[str] = set()
    for bare, variants in qualified_by_bare.items():
        identities = {".".join(value.split(".")[-2:]) for value in variants}
        if len(identities) == 1:
            # Prefer the stable schema.table representation when available.
            result.add(next(iter(identities)))
            handled_bare.add(bare)
        else:
            result.update(variants)
    for table in cleaned:
        parts = [part for part in table.split(".") if part]
        if len(parts) == 1 and parts[0] in handled_bare:
            continue
        if len(parts) >= 2 and parts[-1] in handled_bare:
            continue
        result.add(table)
    return result


def metric_source_tables(metric: dict[str, Any], table_columns: dict[str, dict[str, str]] | None) -> set[str]:
    """Infer metric source tables from base_table, SQL, and required columns."""
    tables: set[str] = set()
    base_table = str(metric.get("base_table") or "").strip()
    if base_table:
        tables.add(base_table.upper())
    tables.update(_sql_tables(str(metric.get("sql_template") or "")))

    required = _split_required_columns(str(metric.get("required_columns") or ""))
    if table_columns and required:
        # A column the metric's own tables hold is read there. ITM_CST on the
        # base table was reason enough to add every other fact with an
        # ITM_CST, so a daily inventory value also claimed the monthly fact.
        # The own tables are still added as the catalog names them, which is
        # what qualifies a bare base table.
        own = {_bare_table(t) for t in tables}
        held = {
            str(c).upper()
            for table, cols in table_columns.items() if _bare_table(table) in own
            for c in (cols or {})
        }
        for table, cols in table_columns.items():
            col_names = {str(c).upper() for c in (cols or {})}
            wanted = required if _bare_table(table) in own else required - held
            if wanted & col_names:
                tables.add(str(table).upper())
    return _canonical_table_references(tables)


def _bare_table(name: object) -> str:
    return str(name or "").replace("[", "").replace("]", "").replace('"', "").split(".")[-1].strip().upper()


def metric_source_schemas(
    metric: dict[str, Any],
    table_columns: dict[str, dict[str, str]] | None,
    entity_schema_map: dict[str, str] | None = None,
) -> set[str]:
    """Return the set of schema names this metric's source tables belong to.

    Resolution order (first match wins):
    1. base_entity → entity_schema_map lookup (most reliable: the entity graph
       always has schema_name; metrics created through the UI have base_entity)
    2. base_table / sql_template table parsing (works when the admin set an FQN
       like PHARMACY.FACT_PRESCRIPTION_FILL in base_table)
    3. required_columns matched against all_columns from _schema.json (fallback
       when base_table is a bare name and entity_schema_map is not available)
    """
    # ── Path 1: base_entity → entity graph schema (preferred) ─────────────────
    if entity_schema_map:
        base_entity = str(metric.get("base_entity") or "").strip()
        if base_entity:
            schema = entity_schema_map.get(base_entity, "")
            if schema:
                return {schema.upper()}

    # ── Path 2 & 3: infer from table references / column matching ─────────────
    return {
        schema
        for schema in (_table_schema(t) for t in metric_source_tables(metric, table_columns))
        if schema
    }


def _schemas_from_graph(graph_context: dict[str, Any] | None, graph: dict[str, Any] | None) -> set[str]:
    if not graph_context or not graph_context.get("enabled") or not graph:
        return set()
    detected = {str(e) for e in graph_context.get("detected") or []}
    if graph_context.get("anchor"):
        detected.add(str(graph_context.get("anchor")))
    schemas: set[str] = set()
    for entity in graph.get("entities") or []:
        if entity.get("entity_name") not in detected:
            continue
        schema = str(entity.get("schema_name") or "").upper().strip()
        if schema:
            schemas.add(schema)
    return schemas


def _schemas_from_semantic_plan(semantic_plan: dict[str, Any] | None) -> set[str]:
    schemas: set[str] = set()
    for field in (semantic_plan or {}).get("fields") or []:
        schema = _table_schema(str(field.get("table") or ""))
        if schema:
            schemas.add(schema)
    return schemas


def ambiguous_metric_question(reader_question: str, lang: str = "en") -> tuple[str, str]:
    """The question asked when a bare measure word has more than one
    definition, in the reader's language, and the term it asks about: the
    reader's own word -- "sales", "ventes", "revenue" -- never a word the
    reader did not use. Only a bare measure question is ambiguous
    (_is_generic_metric_question), so its words are the term."""
    from core.i18n import t

    term = " ".join(re.sub(r"[?!.\u00bf\u00a1:;]+", " ", str(reader_question or "")).split())[:60]
    return t("clar.metric_scope", lang=lang, term=term), term


def _is_generic_metric_question(question: str) -> bool:
    q_tokens = _tokens(question)
    return bool(q_tokens) and q_tokens <= {"revenue", "sales", "amount", "charge"}


def resolve_metric_scope(
    metrics: list[dict[str, Any]],
    question: str,
    table_columns: dict[str, dict[str, str]] | None,
    *,
    selected_schema: str = "",
    graph_context: dict[str, Any] | None = None,
    graph: dict[str, Any] | None = None,
    semantic_plan: dict[str, Any] | None = None,
    entity_schema_map: dict[str, str] | None = None,
    reader_question: str = "",
    limit: int = 6,
) -> MetricScopeResult:
    """Return the metrics that should be visible/enforced for this question.

    Parameters
    ----------
    reader_question : The reader's own words, when ``question`` is the
        canonicalised English form and the two differ. Scored alongside it and
        the better score wins, so a metric whose synonyms an admin wrote in the
        reader's language is still matched. Omit for an English reader, where the
        two are the same string.
    entity_schema_map : Optional dict mapping entity_name → schema_name (UPPER).
        Built from the full entity graph.  When provided, ``base_entity`` on each
        metric is used as the primary schema lookup — bypassing fragile
        ``base_table`` bare-name parsing and ``_schema.json`` column matching.
        Pass ``{e["entity_name"]: e["schema_name"].upper() for e in graph entities}``.
    """
    selected_schema = (selected_schema or "").upper().strip()
    context_schemas = {selected_schema} if selected_schema else set()
    context_schemas.update(_schemas_from_graph(graph_context, graph))
    context_schemas.update(_schemas_from_semantic_plan(semantic_plan))

    scored: list[tuple[int, dict[str, Any], set[str]]] = []
    # By name: a metric is copied below once its schemas are known.
    bare_ok = {str(metric.get("name") or ""): _bare_word_stands_alone(metric, metrics, question, reader_question)
               for metric in metrics or []}
    for metric in metrics or []:
        score = _phrase_score(metric, question, reader_question=reader_question,
                              bare_ok=bare_ok[str(metric.get("name") or "")])
        if score <= 0:
            continue
        schemas = metric_source_schemas(metric, table_columns, entity_schema_map)
        if context_schemas and schemas and not (schemas & context_schemas):
            continue
        if context_schemas and schemas:
            score += 80
        if schemas:
            metric = dict(metric)
            metric["_source_schemas"] = ",".join(sorted(schemas))
        scored.append((score, metric, schemas))

    # A metric for a state the question does not name yields to the measure it
    # does. "Orders by warehouse" shares only "orders" with "back orders"; the
    # field plan read "orders" as the order quantity itself, yet the metric
    # matched, the registry outranked the field, and every orders question --
    # "ordered quantity" in English, "commandes" in French -- was answered with
    # the back-ordered quantity. Only a state is a reason: "Net Sales" or
    # "Total Revenue USD" says more than "sales" or "revenue" too, but it is
    # the governed reading of what was asked and never yields to a column of
    # the bare word. Without a measure of the question's own the metric stays
    # in scope, as it did before.
    named_measures = _measure_columns_in_plan(semantic_plan)
    if named_measures:
        kept: list[tuple[int, dict[str, Any], set[str]]] = []
        for score, metric, schemas in scored:
            states = _narrower_than_asked(metric, question, reader_question=reader_question) & _STATE_WORDS
            reads = _columns_read_by(metric)
            rivals = sorted(named_measures - reads)
            # Its own column named too, the question asks for it as well.
            if states and rivals and not named_measures & reads:
                log.info(
                    "Metric %r left out: it names a state the question does not (%s); the question names %s",
                    metric.get("name"), ", ".join(sorted(states)), ", ".join(rivals),
                )
                continue
            kept.append((score, metric, schemas))
        scored = kept

    scored.sort(key=lambda item: (-item[0], item[1].get("name", "")))
    if not scored:
        return MetricScopeResult(metrics=[], context_schemas=context_schemas)

    schema_sets = [schemas for _, _, schemas in scored if schemas]
    has_cross_schema_choices = len({s for schemas in schema_sets for s in schemas}) > 1
    if not context_schemas and has_cross_schema_choices and _is_generic_metric_question(question):
        return MetricScopeResult(
            metrics=[],
            ambiguous=True,
            options=[m.get("name", "") for _, m, _ in scored[:4] if m.get("name")],
            reason="Multiple revenue-like metrics exist across schemas.",
            context_schemas=context_schemas,
        )

    # Keep only the best metric when it is clearly ahead or schema context exists.
    # The window of 8 points (one token overlap = 10pts, exact-phrase bonus ≥100pts)
    # is intentionally tight: it admits a close second candidate so the LLM can
    # pick between them, but does NOT admit a clearly weaker metric.
    best_score = scored[0][0]
    _TIE_WINDOW = 8
    chosen = [m for score, m, _ in scored if score >= best_score - _TIE_WINDOW]
    # Each measure the question names in its own words is one it asks for,
    # however far its score is from the first: "deliveries and physical
    # inventory counts by month" names two, and the window above -- which
    # keeps a close second reading of ONE measure -- kept the longer name
    # alone. A name said where another's already is, is not a second measure.
    #
    # Read in each wording the question has -- the canonical English and the
    # reader's own -- and a name said inside another's in either is not a
    # second measure: "valeur du stock par entrepôt" is "value of inventory
    # by warehouse" in English, where "inventory" (the stock on hand) stood
    # alone, but in the reader's words its "stock" is inside "valeur du
    # stock", and the stock on hand was read beside the value asked for.
    texts = [text for text in dict.fromkeys((question, reader_question)) if text]

    def _spans(metric: dict[str, Any], text: str) -> list[tuple[int, int]]:
        return _named_spans(metric, text, bare_ok=bare_ok.get(str(metric.get("name") or ""), True))

    said = {text: [span for metric in chosen for span in _spans(metric, text)] for text in texts}

    def _overlaps(spans: list[tuple[int, int]], others: list[tuple[int, int]]) -> bool:
        return any(start < other_end and other_start < end
                   for start, end in spans for other_start, other_end in others)

    for _score, metric, _schemas in scored:
        if any(metric is kept for kept in chosen):
            continue
        spans = {text: _spans(metric, text) for text in texts}
        named_alone = any(spans[text] and not _overlaps(spans[text], said[text]) for text in texts)
        inside_another = any(spans[text] and _overlaps(spans[text], said[text]) for text in texts)
        if named_alone and not inside_another:
            chosen.append(metric)
            for text in texts:
                said[text].extend(spans[text])
    return MetricScopeResult(
        metrics=chosen[: max(1, int(limit or 6))],
        context_schemas=context_schemas,
    )


# The figures a business is asked about in money, and the column codes a
# warehouse keeps any of them under. A question asking for one is asking for
# money a quantity-only warehouse does not have: "what is my revenue trend",
# asked of item balances, was asked which date to use.
_MONEY_WORDS = {
    "revenue": "revenue", "revenues": "revenue", "turnover": "turnover", "sales": "sales",
    "profit": "profit", "profits": "profit", "margin": "margin", "margins": "margin",
    "income": "income", "earnings": "earnings",
}
_MONEY_WORD_RE = re.compile(
    r"\b(" + "|".join(sorted(_MONEY_WORDS, key=len, reverse=True)) + r")\b(?!\s+(?:cent(?:er|re)s?|cent(?:er|re)\b))",
    re.I,
)
_MONEY_COLUMN_TOKENS = frozenset({
    # Not NET or GRS alone: ITM_NET_WT is a weight. NET_SLS_AMT says SLS.
    "REV", "RVN", "REVENUE", "SLS", "SALES", "SAL", "SALE", "AMT", "AMOUNT", "IVC",
    "INVOICE", "TURNOVER", "INCOME", "PRC", "PRICE", "SELL", "PROFIT", "PRFT", "MRG", "MARGIN", "MGN",
    "EARN", "EARNINGS", "GMV",
})
_FLAG_SUFFIXES = ("_IND", "_FLG", "_FLAG", "_KEY", "_ID", "_CD")


def measure_the_data_lacks(
    question: str,
    metrics: list[dict[str, Any]] | None,
    table_columns: dict[str, dict[str, str]] | None,
    fact_tables: set[str] | None = None,
) -> str:
    """The money word a question asks for that this workspace has no figure
    of -- no metric names it and no column of its facts (``fact_tables``,
    every table when none is known) holds any money a sale is kept in -- or
    "". ``question`` is read in its canonical English; a profit centre is a
    place, not a profit."""
    found = _MONEY_WORD_RE.search(_norm(question))
    if not found or not table_columns:
        return ""
    facts = {_bare_table(table) for table in fact_tables or ()}
    if facts:
        table_columns = {table: columns for table, columns in table_columns.items()
                         if _bare_table(table) in facts}
    word = _MONEY_WORDS[found.group(1).lower()]
    for metric in metrics or []:
        if any(re.search(rf"(?<![a-z0-9]){found.group(1).lower()}(?![a-z0-9])", phrase) or word in phrase
               for phrase in _metric_phrases(metric)):
            return ""
    for columns in table_columns.values():
        for column in (columns or {}):
            name = str(column).upper()
            if name.endswith(_FLAG_SUFFIXES):
                continue
            tokens = set(re.split(r"[^A-Z0-9]+", name))
            if tokens & _MONEY_COLUMN_TOKENS:
                return ""
            # A profit centre's key is a place: PFT_CTR.
            if "PFT" in tokens and "CTR" not in tokens:
                return ""
    return word
