"""
core/multi_period.py
─────────────────────
Question-time detection of named calendar periods to be compared.

  "Compare 2025 against 2024 by revenue category"
  "Compare Q1 2024, Q1 2023, and Q1 2022"
  "Revenue in 2023, 2024 and 2025 by product"

Scope, and why it is narrower than it looks
───────────────────────────────────────────
This module OWNS question-time named-period detection. It does not execute
anything, and it must not grow an executor.

It was originally written around a different design: one LLM rewrite of the
base SQL per period, N governed executions in parallel, then a merge. That
design was built (merge_multi_period_results, build_multi_period_chart_payload,
build_multi_period_rewrite_prompt) and never connected to anything — and when
it was finally examined, two independent objections landed. The rewrite prompt
would have sent the row-policy-INJECTED SQL to the model, carrying user id,
group membership and policy literals outside the compliance boundary; and N
separately-validated statements are each legal without being commensurable,
so the arithmetic across them can be confidently wrong. The whole N-execution
half was deleted rather than wired.

What replaced it is one governed query that pivots the named periods into
side-by-side columns, which needs no second execution and no rewrite. This
module supplies the detection and the period vocabulary for that; the SQL is
compiled elsewhere.

core/period_comparison.py stays separate and stays narrow: it derives its
second period by shifting a window backwards from an existing result, has no
parameter for a period the user named, and runs from the compare_prior chip
after an answer. It should not be dragged to question time.

Entry points
────────────
  detect_multi_period_intent(question) → MultiPeriodIntent | None
  extract_period_specs(question) → list[PeriodSpec]
  question_names_comparable_periods(question) → bool
  build_period_plan(intent, question, semantic_plan, db_type) → PeriodPlan | None
"""

from __future__ import annotations

import calendar as _calendar
import re
from dataclasses import dataclass, field
from datetime import date


# ══════════════════════════════════════════════════════════════════════════════
# Data models
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PeriodSpec:
    label: str          # human-readable: "Q1 2024", "2023", "Jan 2024"
    raw_text: str       # original matched text from question


@dataclass
class MultiPeriodIntent:
    period_specs: list[PeriodSpec]
    grain: str          # "monthly" | "quarterly" | "yearly"
    compare_count: int  # number of periods to compare
    raw_match: str = ""
    # Where the labels came from. "named" means every label was written by the
    # user; "relative" means they were derived from the server clock by
    # _generate_relative_specs.
    #
    # This has to be a flag rather than a substring test. "compare revenue for
    # the last 2 years for warehouse 2025 and warehouse 2024" produces
    # clock-derived labels that DO appear in the question -- as warehouse IDs --
    # so asking "is this label in the question?" fails open on exactly the case
    # the check exists to catch. Anything that compiles a period into a SQL
    # predicate must require source == "named".
    source: str = "named"


# ══════════════════════════════════════════════════════════════════════════════
# Detection & period extraction
# ══════════════════════════════════════════════════════════════════════════════

# Named year references: "2022", "2023", "2024"
#
# A bare four-digit integer is the weakest period signal in the language, and
# on an ERP schema it is usually not a period at all. Executed against the
# unguarded version, "list SKUs 2001 2002 2003" yielded three periods and
# "compare warehouse 2024 stock to warehouse 2025 stock" yielded two. Both are
# part numbers. Two gates below narrow it: a plausible-calendar range, and a
# cue word immediately before the number.
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")

# A literal range, deliberately NOT derived from date.today(). Choosing the
# vocabulary of what looks like a year must not make period detection depend on
# the server clock -- that dependency is what `source` exists to track.
_YEAR_MIN, _YEAR_MAX = 1990, 2039

# A word that can plausibly precede a period reference. "in 2024", "against
# 2023", "between 2020 and 2024" are periods; "warehouse 2024", "SKU 2001",
# "priced 2020" are not.
_PERIOD_CUE_WORDS = frozenset({
    "in", "for", "during", "since", "until", "through", "of", "from", "to",
    "between", "and", "vs", "vs.", "versus", "against", "compare", "compared",
    "comparing", "year", "years", "fy", "cy",
})

# Trailing punctuation that can separate items in a list of periods.
_PERIOD_LIST_PUNCT = frozenset({",", ";", ":", "(", "[", "-", "/", "&"})

# Quarter references: "Q1 2024", "2024 Q1", "Q1/2024"
_QUARTER_RE = re.compile(r"\bQ([1-4])[\s\-/]?(20\d{2})\b|\b(20\d{2})[\s\-/]?Q([1-4])\b", re.I)

# Month references: "Jan 2024", "January 2024"
_MONTH_RE = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\s*(20\d{2})\b",
    re.I,
)

# "last N years / quarters / months"
_LAST_N_PERIODS = re.compile(r"\blast\s+(\d+)\s+(year|quarter|month)s?\b", re.I)

# Multi-period signals: "compare X, Y, and Z" or "across the last N"
_MULTI_SIGNAL = re.compile(
    r"\b(compare|contrast|side.by.side|vs\.?\s|versus|across\s+(?:the\s+)?last\s+\d+|"
    r"(\d+)[- ]year|year.over.year.for.the.last)\b",
    re.I,
)


def detect_multi_period_intent(question: str) -> MultiPeriodIntent | None:
    """
    Return a MultiPeriodIntent if the question names two or more periods to be
    compared, else None.

    (The docstring used to say "3+ periods ... None if it's a 2-period
    question". The code below has had a two-period branch since the initial
    commit, so the docstring was describing a rule the function never had.)
    """
    q = question.strip()

    # "last N years/quarters/months" — explicit multi-period
    m = _LAST_N_PERIODS.search(q)
    if m:
        n    = int(m.group(1))
        unit = m.group(2).lower()
        if n >= 2:
            grain = {"year": "yearly", "quarter": "quarterly", "month": "monthly"}.get(unit, "yearly")
            specs = _generate_relative_specs(n, unit)
            return MultiPeriodIntent(
                period_specs=specs,
                grain=grain,
                compare_count=n,
                raw_match=m.group(0),
                source="relative",   # labels came from date.today(), not the user
            )

    # Named period extraction
    specs = extract_period_specs(q)
    if len(specs) >= 3:
        grain = _infer_grain(specs)
        return MultiPeriodIntent(
            period_specs=specs,
            grain=grain,
            compare_count=len(specs),
            raw_match=q[:80],
        )

    # 2 named periods — only if a multi-signal word is present and the
    # existing period_comparison engine can't handle it (e.g. non-calendar dates)
    if len(specs) == 2 and _MULTI_SIGNAL.search(q):
        grain = _infer_grain(specs)
        return MultiPeriodIntent(
            period_specs=specs,
            grain=grain,
            compare_count=2,
            raw_match=q[:80],
        )

    return None


def extract_period_specs(question: str) -> list[PeriodSpec]:
    """
    Extract all named period references from a question, in order of appearance.
    Deduplicates overlapping matches (quarter takes priority over bare year).
    """
    found: list[tuple[int, int, PeriodSpec]] = []  # (start, end, spec)

    # Quarter matches — highest priority
    for m in _QUARTER_RE.finditer(question):
        if m.group(1):  # Qn YYYY
            label = f"Q{m.group(1)} {m.group(2)}"
        else:           # YYYY Qn
            label = f"Q{m.group(4)} {m.group(3)}"
        found.append((m.start(), m.end(), PeriodSpec(label=label, raw_text=m.group(0))))

    # Month + year matches
    for m in _MONTH_RE.finditer(question):
        # Check not overlapping with a quarter match
        if not any(s <= m.start() < e for s, e, _ in found):
            label = f"{m.group(1).capitalize()} {m.group(2)}"
            found.append((m.start(), m.end(), PeriodSpec(label=label, raw_text=m.group(0))))

    # Bare year matches — only where no quarter/month match covers them, and
    # only where the number is actually being used as a period. Left to right,
    # because a comma is only a period separator once a period has been named.
    bare_year_accepted = bool(found)
    for m in _YEAR_RE.finditer(question):
        if any(s <= m.start() < e for s, e, _ in found):
            continue
        if not _bare_year_is_a_period(question, m, bare_year_accepted):
            continue
        bare_year_accepted = True
        found.append((m.start(), m.end(),
                      PeriodSpec(label=m.group(1), raw_text=m.group(0))))

    # Sort by position in text and deduplicate
    found.sort(key=lambda x: x[0])
    return [spec for _, _, spec in found]


def _bare_year_is_a_period(question: str, match: re.Match, seen_a_period: bool) -> bool:
    """Is this four-digit integer being used as a calendar period?

    Two independent gates, both required:

    1. It falls inside a plausible calendar range. A part number like 2050 or
       an SKU like 1974 can pass this; the point is only to reject the obvious
       non-years cheaply.
    2. The token immediately before it can introduce a period. This is the gate
       that does the real work: "warehouse 2024" and "SKU 2001" are rejected
       because "warehouse" and "SKU" are not period cues, while "against 2024"
       and "in 2023" are accepted.

    A separating comma counts only once some period has already been named, so
    "in 2023, 2024 and 2025" reads as three periods while "items priced 2020,
    2030" reads as none -- the comma in the second is separating part numbers.
    """
    if not (_YEAR_MIN <= int(match.group(1)) <= _YEAR_MAX):
        return False

    before = question[: match.start()].rstrip()
    if not before:
        return True                      # the question opens with the year
    if before[-1] in _PERIOD_LIST_PUNCT:
        return seen_a_period
    if before[-1] in {".", "?", "!"}:
        return True                      # start of a new sentence
    return before.split()[-1].lower().strip("\"'([-") in _PERIOD_CUE_WORDS


def _infer_grain(specs: list[PeriodSpec]) -> str:
    if not specs:
        return "yearly"
    first = specs[0].label
    if re.match(r"Q\d", first, re.I):
        return "quarterly"
    if re.match(r"[A-Za-z]{3,9}\s+\d{4}", first):
        return "monthly"
    return "yearly"


def _generate_relative_specs(n: int, unit: str) -> list[PeriodSpec]:
    """
    Generate PeriodSpec list for "last N years/quarters/months".
    Labels are relative strings that the SQL rewrite prompt will interpret.
    """
    from datetime import date

    today = date.today()
    specs = []

    if unit == "year":
        for i in range(n):
            year = today.year - i - 1  # last year = today.year-1
            specs.append(PeriodSpec(label=str(year), raw_text=str(year)))

    elif unit == "quarter":
        q_now = (today.month - 1) // 3 + 1
        y_now = today.year
        for i in range(n):
            # Shift backward by i quarters from "last quarter"
            total_q = (y_now * 4 + q_now - 1) - 1 - i  # -1 for "last"
            year  = total_q // 4
            qtr   = (total_q % 4) + 1
            specs.append(PeriodSpec(label=f"Q{qtr} {year}", raw_text=f"Q{qtr} {year}"))

    elif unit == "month":
        import calendar
        y, mo = today.year, today.month
        for i in range(n):
            total_m = (y * 12 + mo - 1) - 1 - i
            year  = total_m // 12
            month = (total_m % 12) + 1
            abbr  = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][month]
            specs.append(PeriodSpec(label=f"{abbr} {year}", raw_text=f"{abbr} {year}"))

    return specs[::-1]  # chronological order (oldest first)


# ══════════════════════════════════════════════════════════════════════════════
# Period vocabulary — parsing one label into calendar parts
# ══════════════════════════════════════════════════════════════════════════════

_MONTH_ABBR = {name.lower(): n for n, name in enumerate(_calendar.month_abbr) if name}
_MONTH_FULL = {name.lower(): n for n, name in enumerate(_calendar.month_name) if name}

# Above this, portal_chat.html silently drops y-keys past the palette length,
# so a 12-period request would render fewer series than were asked for --
# the same silent incompleteness this whole change exists to remove. Capping
# in SQL narrows the request, but it does so visibly.
_MAX_PERIODS = 6


@dataclass(frozen=True)
class PeriodPlan:
    """A compiled, governed instruction for comparing named periods.

    Every field is derived from a date role that actually resolved. If no
    governed date field is available this object is never built, because a
    hint that forbids YEAR()/DATEPART() while naming no sanctioned alternative
    is an instruction nothing can follow.
    """
    labels: list[str]        # chronological, oldest first
    aliases: list[str]       # column-name suffixes, parallel to labels
    predicates: list[str]    # SQL boolean expressions, parallel to labels
    grain: str               # "yearly" | "quarterly" | "monthly"
    date_field: dict         # the date_key_policy this was compiled against
    warnings: list[str] = field(default_factory=list)

    @property
    def oldest(self) -> str:
        return self.labels[0]

    @property
    def newest(self) -> str:
        return self.labels[-1]


def period_parts(label: str) -> tuple[int, int, int] | None:
    """Parse a period label into (year, kind, ordinal).

    kind is 0 for a year, 1 for a quarter, 2 for a month, which is also the
    sort order within a year, so the tuple sorts chronologically as-is.
    Returns None for anything unparseable -- the caller must refuse rather
    than guess.
    """
    text = str(label or "").strip()
    if not text:
        return None

    m = re.fullmatch(r"(\d{4})", text)
    if m:
        return (int(m.group(1)), 0, 0)

    m = re.fullmatch(r"Q([1-4])\s+(\d{4})", text, re.I)
    if m:
        return (int(m.group(2)), 1, int(m.group(1)))

    m = re.fullmatch(r"([A-Za-z]{3,9})\s+(\d{4})", text)
    if m:
        name = m.group(1).lower()
        month = _MONTH_ABBR.get(name[:3]) if name[:3] in _MONTH_ABBR else None
        if _MONTH_FULL.get(name):
            month = _MONTH_FULL[name]
        if month:
            return (int(m.group(2)), 2, month)
    return None


def period_sort_key(label: str) -> tuple[int, int, int]:
    """Chronological ordering. The detector returns question order, and the
    target question asks for 2025 before 2024, so every consumer must sort."""
    return period_parts(label) or (0, 0, 0)


def period_alias_suffix(label: str) -> str:
    """A column-name-safe suffix: "Q1 2024" -> "Q1_2024", "Jan 2024" -> "JAN_2024"."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(label or "").strip()).strip("_").upper()


def period_bounds(label: str) -> tuple[date, date] | None:
    """The half-open [start, end) span of a period.

    Half-open deliberately: a closed upper bound on a datetime column silently
    drops the last day's rows, and this is the fallback used when no calendar
    attribute is available, so it is the path with the least other checking.
    """
    parts = period_parts(label)
    if not parts:
        return None
    year, kind, ordinal = parts
    if kind == 0:
        return date(year, 1, 1), date(year + 1, 1, 1)
    if kind == 1:
        start_month = (ordinal - 1) * 3 + 1
        end_year, end_month = (year, start_month + 3) if start_month + 3 <= 12 else (year + 1, 1)
        return date(year, start_month, 1), date(end_year, end_month, 1)
    end_year, end_month = (year, ordinal + 1) if ordinal < 12 else (year + 1, 1)
    return date(year, ordinal, 1), date(end_year, end_month, 1)


# ══════════════════════════════════════════════════════════════════════════════
# Plan compilation
# ══════════════════════════════════════════════════════════════════════════════

def question_names_comparable_periods(question: str) -> bool:
    """Does this question name two or more periods and ask them to be compared?

    Used by core/contextual_dates to widen the date-binding gate. Deliberately
    narrower than question_has_explicit_date_filter, whose bare four-digit-year
    pattern would widen the gate on any question containing a number.
    """
    intent = detect_multi_period_intent(question)
    if intent is None or intent.source != "named":
        return False
    if not (2 <= len(intent.period_specs) <= _MAX_PERIODS):
        return False
    if not _wants_comparison(question):
        return False
    return all(period_parts(spec.label) for spec in intent.period_specs)


def _wants_comparison(question: str) -> bool:
    """Reuse the one live comparison vocabulary rather than adding a sixth.

    core/query_semantics.analyze_query_intent already lists against, relative
    to, contrast, delta, variance and benchmark. This product has a documented
    habit of growing parallel detectors for the same concept and letting them
    drift.
    """
    try:
        from core.query_semantics import analyze_query_intent
        return bool(analyze_query_intent(question).get("wants_comparison"))
    except Exception:
        return False


def _date_policy(semantic_plan: dict | None) -> dict:
    """The governed date field the plan bound, or {}."""
    for policy in (semantic_plan or {}).get("date_key_policies") or []:
        if policy.get("column") and (policy.get("table") or policy.get("role_alias")):
            return dict(policy)
    return {}


def _period_predicate(
    label: str,
    policy: dict,
    grain: str,
    db_type: str,
) -> str:
    """Compile one period into a governed SQL boolean expression.

    Preference order is the calendar dimension's own validated attributes,
    then a half-open range on the approved date value. There is deliberately
    no third option: wrapping the date column in YEAR()/DATEPART()/CONVERT()
    is what the surrogate_date_conversion rules in core/validator.py refuse,
    and emitting it here would produce SQL the product then rejects.
    """
    from core.contextual_dates import (
        format_calendar_attribute_ref,
        format_date_value_expression,
    )

    parts = period_parts(label)
    if not parts:
        return ""
    year, kind, ordinal = parts
    alias = str(policy.get("role_alias") or "")
    attrs = dict(policy.get("calendar_attributes") or {})

    def attr(name: str) -> str:
        return format_calendar_attribute_ref(alias, attrs, name, db_type)

    year_ref = attr("year")
    if kind == 0 and year_ref:
        return f"{year_ref} = {year}"
    if kind == 1 and year_ref and attr("quarter"):
        return f"{year_ref} = {year} AND {attr('quarter')} = {ordinal}"
    if kind == 2 and year_ref and attr("month_number"):
        return f"{year_ref} = {year} AND {attr('month_number')} = {ordinal}"
    if kind == 2 and attr("year_month"):
        return f"{attr('year_month')} = {year}{ordinal:02d}"

    bounds = period_bounds(label)
    if not bounds:
        return ""

    # The calendar dimension is joined under role_alias, so the reference must
    # be alias-qualified exactly as core/pipeline_helpers builds it -- naming
    # the physical table here produces SQL that will not resolve.
    value_column = str(policy.get("date_value_column") or "")
    if alias and value_column:
        qualified = f"{alias}.{_quote_identifier(value_column, db_type)}"
    elif policy.get("table") and policy.get("column"):
        qualified = (f"{policy['table']}."
                     f"{_quote_identifier(str(policy['column']), db_type)}")
    else:
        return ""

    # Carries the yyyymmdd/yyyymm integer conversion when the role needs one,
    # and returns the reference untouched otherwise.
    date_ref = format_date_value_expression(
        "", qualified, str(policy.get("date_key_type") or "surrogate_fk"), db_type)

    start, end = bounds
    return f"{date_ref} >= '{start.isoformat()}' AND {date_ref} < '{end.isoformat()}'"


def _quote_identifier(column: str, db_type: str) -> str:
    """Dialect quoting for one bare column name."""
    name = str(column or "").strip().strip("[]\"`")
    dialect = str(db_type or "azure_sql").lower()
    if dialect in {"azure_sql", "sqlserver", "mssql"}:
        return f"[{name}]"
    if dialect in {"snowflake", "oracle"}:
        return f'"{name}"'
    return name


def build_period_plan(
    intent: MultiPeriodIntent | None,
    question: str,
    semantic_plan: dict | None,
    db_type: str = "azure_sql",
) -> PeriodPlan | None:
    """Compile a detected intent into governed period predicates, or None.

    Returns None -- meaning "behave exactly as before" -- unless every gate
    holds. There is no partial mode: a plan that names some periods and
    guesses at others produces arithmetic across incomparable columns.
    """
    if intent is None or intent.source != "named":
        return None

    labels = [spec.label for spec in intent.period_specs]
    if not (2 <= len(labels) <= _MAX_PERIODS):
        return None

    parsed = [period_parts(label) for label in labels]
    if not all(parsed):
        return None
    if len({p[1] for p in parsed}) != 1:          # mixed grains are not comparable
        return None
    if len(set(labels)) != len(labels):
        return None

    # Every label must be the user's own words. This is belt-and-braces over
    # intent.source, and it is cheap.
    lowered = (question or "").lower()
    if not all(str(spec.raw_text or "").lower() in lowered for spec in intent.period_specs):
        return None
    if not _wants_comparison(question):
        return None

    policy = _date_policy(semantic_plan)
    if not policy:
        return None

    ordered = sorted(labels, key=period_sort_key)
    predicates = [_period_predicate(label, policy, intent.grain, db_type)
                  for label in ordered]
    if not all(predicates):
        return None

    return PeriodPlan(
        labels=ordered,
        aliases=[period_alias_suffix(label) for label in ordered],
        predicates=predicates,
        grain=intent.grain,
        date_field=policy,
    )
