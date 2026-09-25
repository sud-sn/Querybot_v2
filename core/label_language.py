"""
core/label_language.py

A label is shown in the reader's language.

A warehouse built for two languages keeps a label twice: ITM_GRP_DSC and
ITM_GRP_FR_DSC, MTH_NM and MTH_FR_NM. Nothing knew the second was the first
in French, so a French reader was shown the English label -- and an English
reader could be shown the French one, since both read as a description.

And the French one is not always filled. On a real inventory warehouse's
schema and sample rows, read offline, every business label's French twin --
items, item groups, product groups, ABC classes, regions, divisions -- is
blank on every row, while the calendar's (day, month, holiday and period
names) is filled. Showing the French column as it is would give a French
reader a column of blanks.

So a label with a twin in another language follows the reader, value by
value: a French reader is shown the French label where it is filled and the
English one where it is blank -- COALESCE(NULLIF(TRIM(fr), ''), en) -- and an
English reader the English one. A question that names a language ("in
French", "en anglais") is shown that one. The rule rides on the semantic plan
like the unit rule: the prompt states it, the validator refuses a label in the
wrong language, and the answer card tells the reader where the other language
stands in.

A twin belongs to its table: a label is held to the rule only where the table
it is read from keeps the twin, so a table with ITM_NM and no French copy is
left alone however many others have one.

Tenant-neutral: the twins are read from the discovered schema's names.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger("querybot.label_language")

# A token that marks a column as another language's copy, and the language.
# Only unambiguous ones: FRE and FRN also read as freight and foreign.
_LANGUAGE_TOKENS = {
    "FR": "fr", "FRA": "fr", "FRENCH": "fr", "FRANCAIS": "fr",
    "EN": "en", "ENG": "en", "ENGLISH": "en", "ANGLAIS": "en",
}
_OTHER_LANGUAGE = {"fr": "en", "en": "fr"}
# Only a label has a twin worth following: FR in PRC_FR_DT is "from".
_LABEL_SUFFIXES = frozenset({
    "NM", "NAME", "DSC", "DESC", "DESCRIPTION", "LBL", "LABEL", "TXT", "TEXT", "SHRT", "SHORT",
})

_ASKED = (
    ("fr", re.compile(r"\b(?:in|en)\s+(?:french|fran[cç]ais)\b|\bfrench\s+(?:name|label|description)s?\b"
                      r"|\b(?:nom|libell[ée]|description)s?\s+(?:en\s+)?fran[cç]ais\b", re.IGNORECASE)),
    ("en", re.compile(r"\b(?:in|en)\s+(?:english|anglais)\b|\benglish\s+(?:name|label|description)s?\b"
                      r"|\b(?:nom|libell[ée]|description)s?\s+(?:en\s+)?anglais\b", re.IGNORECASE)),
)


def language_twins(columns) -> list[dict]:
    """The label columns among `columns` that repeat another of them in a
    second language: ``[{"base": "ITM_GRP_DSC", "twin": "ITM_GRP_FR_DSC",
    "language": "fr"}]``, the names as given.

    The language token may stand anywhere in the name (ITM_FR_NM, ITM_NM_FR);
    what is left must be a label that exists. Where both copies carry a token
    (ITM_EN_NM, ITM_FR_NM) the English one stands as the base.
    """
    names = {str(c).upper(): str(c) for c in columns or []}
    twins = []
    for upper, name in names.items():
        tokens = upper.split("_")
        for index, token in enumerate(tokens):
            language = _LANGUAGE_TOKENS.get(token)
            rest = tokens[:index] + tokens[index + 1:]
            if not language or not rest or rest[-1] not in _LABEL_SUFFIXES:
                continue
            base = "_".join(rest)
            if base not in names and language == "fr":
                base = next((
                    candidate for candidate in (
                        "_".join(tokens[:index] + [other] + tokens[index + 1:])
                        for other, lang in _LANGUAGE_TOKENS.items() if lang == "en"
                    ) if candidate in names
                ), "")
            if base in names:
                twins.append({"base": names[base], "twin": name, "language": language})
                break
    return sorted(twins, key=lambda t: t["twin"])


def question_names_a_language(*texts: str) -> str:
    """"in French", "en anglais", "French names": the language the question
    asks labels in, or ""."""
    text = " ".join(str(t or "") for t in texts)
    for language, pattern in _ASKED:
        if pattern.search(text):
            return language
    return ""


def label_policies(account_id: str) -> list[dict]:
    """One policy per discovered table whose labels have a twin."""
    import store
    from core.business_meaning import _discovered_tables

    state = store.get_client_state(account_id) or {}
    policies = []
    for table, columns in sorted(_discovered_tables(str(state.get("schema_dir") or "")).items()):
        twins = language_twins([name for name, _ in columns])
        if twins:
            policies.append({"kind": "label_language", "table": table, "twins": twins})
    return policies


def attach_label_policies(
    semantic_plan: dict | None, account_id: str, reader_language: str, *questions: str,
) -> list[dict]:
    """Put the policies on the plan, with the language labels are shown in:
    the one the question names, or else the reader's."""
    if not isinstance(semantic_plan, dict):
        return []
    policies = label_policies(account_id)
    if policies:
        semantic_plan["label_policies"] = policies
        semantic_plan["label_language"] = (
            question_names_a_language(*questions) or str(reader_language or "en").lower()[:2]
        )
    return policies


def _bare(name: str) -> str:
    return str(name or "").split(".")[-1].strip('[]"`').upper()


def policies_in_scope(policies: list[dict] | None, *texts: str) -> list[dict]:
    haystack = " ".join(str(text or "") for text in texts).upper()
    return [
        policy for policy in policies or []
        if isinstance(policy, dict) and re.search(
            rf"(?<![A-Z0-9_]){re.escape(_bare(policy.get('table', '')))}(?![A-Z0-9_])", haystack,
        )
    ]


def _quote(column: str, db_type: str) -> str:
    from core.relationship_validator import _quote_col

    return _quote_col(column, db_type)


def label_expression(alias: str, twin: dict, db_type: str = "azure_sql") -> str:
    """The label in the twin's language, or the base's where it is blank."""
    prefix = f"{alias}." if alias else ""
    fr, en = _quote(twin["twin"], db_type), _quote(twin["base"], db_type)
    return f"COALESCE(NULLIF(TRIM({prefix}{fr}), ''), {prefix}{en})"


def _shown(language: str, twin: dict) -> bool:
    """Whether the twin is the reader's language (else the base is)."""
    return twin["language"] == language


def _language_name(language: str) -> str:
    return {"fr": "French", "en": "English"}.get(language, language)


def format_label_rules(policies: list[dict] | None, language: str, db_type: str = "azure_sql") -> str:
    lines: list[str] = []
    for policy in policies or []:
        table = _bare(policy["table"])
        for twin in policy["twins"]:
            if _shown(language, twin):
                lines.append(f"- {table}: {label_expression(table, twin, db_type)}")
            else:
                lines.append(f"- {table}: {twin['base']}, not {twin['twin']}")
    if not lines:
        return ""
    if any(_shown(language, t) for p in policies or [] for t in p["twins"]):
        head = (
            f"The reader reads labels in {_language_name(language)}. A label below has a twin in "
            "that language, which is blank on many rows: show the twin where it is filled and "
            "the other where it is blank, as written with the table's alias -- select AND group "
            "by that expression, never either column alone."
        )
    else:
        head = (
            f"The reader reads labels in {_language_name(language)}. These label columns have a "
            "twin in another language: show the one named, never its twin."
        )
    return "\n".join(["## Labels in the reader's language — REQUIRED", head, *lines])


# ── Reading a query for a label in the wrong language ───────────────────────


def _twins_by_table(policies: list[dict] | None) -> dict[str, list[dict]]:
    by_table: dict[str, list[dict]] = {}
    for policy in policies or []:
        if isinstance(policy, dict):
            by_table.setdefault(_bare(policy.get("table", "")), []).extend(
                twin for twin in policy.get("twins") or [] if isinstance(twin, dict)
            )
    return by_table


def _source_twins(select, by_table: dict[str, list[dict]], ctes: dict) -> dict[str, list[dict]]:
    """{how this SELECT names each source: the twins its columns can read} --
    a table's own twins; for a CTE or a derived table, the twins it passes up
    whole, both columns, since what it combined was read where it read them."""
    from sqlglot import exp

    everywhere: list[dict] = []
    for twins in by_table.values():
        everywhere.extend(t for t in twins if t not in everywhere)

    def passed_up(query) -> list[dict]:
        outputs = {str(name).upper() for name in getattr(query, "named_selects", None) or []}
        return [t for t in everywhere if {t["base"].upper(), t["twin"].upper()} <= outputs]

    source = select.args.get("from_") or select.args.get("from")
    nodes = ([source.this] if source is not None else []) + [
        join.this for join in select.args.get("joins") or []
    ]
    sources: dict[str, list[dict]] = {}
    for node in nodes:
        key = str(node.alias_or_name or "").upper()
        if isinstance(node, exp.Table):
            name = str(node.name).upper()
            sources[key] = passed_up(ctes[name]) if name in ctes and not node.db else by_table.get(name, [])
        elif isinstance(node, exp.Subquery):
            sources[key] = passed_up(node.this)
    return sources


def _shown_columns(select) -> list[tuple]:
    """(column, aggregated) for each column the SELECT list shows -- its own,
    not one only counted, compared, or partitioning or ordering a window."""
    from sqlglot import exp

    shown = []
    for projection in select.expressions:
        for column in projection.find_all(exp.Column):
            if column.find_ancestor(exp.Select) is not select:
                continue
            aggregated, hidden, child, node = False, False, column, column.parent
            while node is not None and node is not select:
                if isinstance(node, (exp.Count, exp.Predicate)) or (
                    isinstance(node, exp.Window) and child.arg_key in {"partition_by", "order"}
                ):
                    hidden = True
                    break
                aggregated = aggregated or isinstance(node, exp.AggFunc)
                child, node = node, node.parent
            if not hidden:
                shown.append((column, aggregated))
    return shown


def label_mismatches(sql, policies: list[dict] | None, language: str, db_type: str = "azure_sql") -> list[dict]:
    """Each label a SELECT shows in the wrong language, with the fix:
    ``[{"column", "twin", "required", "clause"}]``.

    The reader's-language twin must be shown wherever its other is: a SELECT
    that shows the English label to a French reader must show the French one
    beside it -- in one COALESCE, or both passed up to the query that
    combines them -- and a grouped SELECT groups by what it shows. A SELECT
    that shows the twin in the other language to a reader of this one is
    wrong wherever it stands.
    """
    from sqlglot import exp

    from core.units_of_measure import _parse
    from core.unknown_members import _grouped_columns

    by_table = _twins_by_table(policies)
    tree = _parse(sql, db_type) if by_table and language else None
    if tree is None:
        return []
    ctes = {str(cte.alias).upper(): cte.this for cte in tree.find_all(exp.CTE)}
    found: list[dict] = []
    for select in tree.find_all(exp.Select):
        sources = _source_twins(select, by_table, ctes)
        # (source, COLUMN) -> (the qualifier as written, shown outside an aggregate)
        shown: dict[tuple[str, str], tuple[str, bool]] = {}
        for column, aggregated in _shown_columns(select):
            name, qualifier = str(column.name).upper(), str(column.table or "")
            for key in [qualifier.upper()] if qualifier else list(sources):
                if any(name in (t["base"].upper(), t["twin"].upper()) for t in sources.get(key, [])):
                    written, bare = shown.get((key, name), (qualifier, False))
                    shown[(key, name)] = (written, bare or not aggregated)
                    break
        grouped = {
            (str(column.table or "").upper(), str(column.name).upper())
            for column in _grouped_columns(select)
        } if select.args.get("group") is not None else None
        for (key, name), (qualifier, bare) in shown.items():
            for twin in sources[key]:
                base, other = twin["base"].upper(), twin["twin"].upper()
                if not _shown(language, twin):
                    if name == other:
                        found.append({
                            "column": twin["twin"], "twin": twin["base"], "clause": "SELECT",
                            "required": f"{qualifier + '.' if qualifier else ''}{_quote(twin['base'], db_type)}",
                        })
                    continue
                if name == base and (key, other) not in shown:
                    found.append({
                        "column": twin["base"], "twin": twin["twin"], "clause": "SELECT",
                        "required": label_expression(qualifier, twin, db_type),
                    })
                elif name in (base, other) and bare and grouped is not None and not any(
                    column == name and (not table or table == key) for table, column in grouped
                ):
                    found.append({
                        "column": twin["base"] if name == base else twin["twin"],
                        "twin": twin["twin"] if name == base else twin["base"], "clause": "GROUP BY",
                        "required": label_expression(qualifier, twin, db_type),
                    })
    unique: list[dict] = []
    for item in found:
        if item not in unique:
            unique.append(item)
    return unique


def labels_fall_back(sql, policies: list[dict] | None, language: str, db_type: str = "azure_sql") -> bool:
    """Whether the query shows a label through its twin in the reader's
    language with the other in its blanks -- one COALESCE of the two, which
    the answer card tells the reader of."""
    from sqlglot import exp

    from core.units_of_measure import _parse

    pairs = {
        (str(twin["twin"]).upper(), str(twin["base"]).upper())
        for policy in policies or [] if isinstance(policy, dict)
        for twin in policy.get("twins") or [] if _shown(language, twin)
    }
    tree = _parse(sql, db_type) if pairs else None
    if tree is None:
        return False
    for coalesce in tree.find_all(exp.Coalesce):
        names = {str(column.name).upper() for column in coalesce.find_all(exp.Column)}
        if any({twin, base} <= names for twin, base in pairs):
            return True
    return False
