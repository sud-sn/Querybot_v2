from __future__ import annotations

import re
from dataclasses import asdict, dataclass


_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "twenty": 20,
    "fifty": 50,
    "hundred": 100,
}


@dataclass(frozen=True)
class TopNIntent:
    """Structured, schema-independent interpretation of a Top-N request."""

    limit: int
    direction: str = "descending"
    tie_policy: str = "exactly_n"
    per_group: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def detect_top_n_intent(question: str) -> TopNIntent | None:
    """
    Detect explicit Top/Bottom-N language without guessing business entities.

    A numeric limit is mandatory. Generic words such as "highest" on their
    own remain ordinary ranking intent and are not forced into a row limit.
    """
    q = re.sub(r"\s+", " ", (question or "").strip().lower())
    if not q:
        return None

    number = r"(?P<n>\d{1,3}|" + "|".join(_NUMBER_WORDS) + r")"
    patterns = (
        rf"\b(?:top|bottom|best|worst|highest|lowest|leading)\s+{number}\b",
        rf"\b{number}\s+(?:top|bottom|best|worst|highest|lowest|leading)\b",
    )
    match = next((m for p in patterns if (m := re.search(p, q))), None)
    if match is None:
        return None

    raw_limit = match.group("n")
    limit = int(raw_limit) if raw_limit.isdigit() else _NUMBER_WORDS[raw_limit]
    if limit < 1:
        return None

    direction = "ascending" if re.search(r"\b(bottom|worst|lowest)\b", match.group(0)) else "descending"
    tie_policy = (
        "include_ties"
        if re.search(r"\b(with|include|including|keep)\s+(all\s+)?ties\b", q)
        else "exactly_n"
    )
    per_group = bool(re.search(
        r"\b(per|for|in|within|from)\s+(each|every)\b|\bper\s+[a-z][\w-]*\b",
        q,
    ))
    return TopNIntent(limit, direction, tie_policy, per_group)


def _has_any(text: str, phrases: tuple[str, ...]) -> bool:
    lowered = (text or "").lower()
    return any(phrase in lowered for phrase in phrases)


def analyze_query_intent(question: str) -> dict[str, bool]:
    """
    Classify broad analytics intent from plain-English phrasing.

    This is generic language understanding only. It deliberately avoids any
    client-specific business rules so it is safe to reuse in SQL grounding and
    clarification prompts.
    """
    q = (question or "").strip().lower()
    top_n = detect_top_n_intent(question)
    return {
        # ── People / employee scope ───────────────────────────────────────────
        # Covers direct HR terms, role titles, and generic "people" language.
        # Intentionally broad so downstream rules can narrow with conjunction.
        "has_employee_scope": bool(re.search(
            r"\b(employee|employees|staff|workforce|headcount"
            r"|worker|workers|personnel|team\s+member|team\s+members"
            r"|associate|associates|agent|agents|rep|reps"
            r"|hire|hires|joiner|joiners|new\s+hire"
            r"|hr\b|human\s+resource|human\s+resources"
            r"|person|people|colleague|colleagues"
            r"|contractor|contractors|consultant|consultants"
            r"|earner|earners|performer|performers|hire|hires)\b",
            q,
        )),

        # ── Distinct / deduplicated count ────────────────────────────────────
        # Catches "unique employees", "how many different products", etc.
        "wants_distinct_count": bool(re.search(
            r"\b(unique|distinct|deduplicated|how\s+many\s+different"
            r"|individual|separate|non.duplicate|no\s+duplicate|unduplicated)\b",
            q,
        )),

        # ── Grouping / breakdown ─────────────────────────────────────────────
        # Used only in conjunction with has_employee_scope so broad coverage
        # is acceptable — a false positive here doesn't fire a hint on its own.
        "wants_grouping": bool(re.search(
            r"\b(grouped?\s+by|group\s+by|breakdown|broken\s+down\s+by"
            r"|split\s+by|by\s+(department|team|region|category|type|location"
            r"|division|grade|level|unit|branch|office|gender|status|role)"
            r"|per\s+(department|team|region|category|type|location|division)"
            r"|for\s+each|each\s+(department|team|region|category|group|type|unit)"
            r"|based\s+on|segmented\s+by|segmented\s+into|categorised\s+by|categorized\s+by)\b",
            q,
        )),

        # ── Categorical / status filter ──────────────────────────────────────
        # Removed bare "where" and "with" — they are stop words and fire on
        # almost every sentence.  Kept and extended meaningful status phrases.
        "wants_status_filter": bool(re.search(
            r"\b(status|marked\s+as|labelled\s+as|labeled\s+as|tagged\s+as"
            r"|classified\s+as|categoris[e]?d\s+as|flagged\s+as"
            r"|who\s+are|whose\s+status|filter\s+by\s+(status|type|category)"
            r"|with\s+status|with\s+type|where\s+status|where\s+type"
            r"|type\s+is|status\s+is|category\s+is|whose\s+type)\b",
            q,
        )),

        # ── Name lookup ──────────────────────────────────────────────────────
        # Added "who" phrases, split name terms, and "list" in entity context.
        "wants_names": bool(re.search(
            r"\b(name|names|full\s*name|fullname"
            r"|first\s+name|last\s+name|surname|forename|given\s+name|family\s+name"
            r"|who\s+(is|are|has|have|was|were|did|does)\b"
            r"|list\s+(all|the|of)\s+\w+"
            r"|show\s+(me\s+)?(all\s+)?(the\s+)?\w+\s+(names?|who))\b",
            q,
        )),

        # ── Time series / trend ───────────────────────────────────────────────
        # Added quarterly, annually, hourly, "each month/quarter", historical.
        "wants_time_series": bool(re.search(
            r"\b(trend|over\s+time|time\s+series|timeline"
            r"|by\s+(month|week|year|quarter|day|date|hour)"
            r"|monthly|weekly|daily|quarterly|annually|yearly|hourly"
            r"|month\s+by\s+month|week\s+by\s+week|day\s+by\s+day|quarter\s+by\s+quarter"
            r"|each\s+(month|week|quarter|year|day)"
            r"|historical|history|progression"
            r"|over\s+the\s+(months|weeks|quarters|years|period|last\s+\d+))\b",
            q,
        )),

        # ── Comparison framing ───────────────────────────────────────────────
        # Added delta, variance, contrast, relative to, benchmark, against.
        "wants_comparison": bool(re.search(
            r"\b(compare|comparison|versus|vs\.?|difference|gap"
            r"|contrast|against|relative\s+to|in\s+relation\s+to"
            r"|better\s+than|worse\s+than|higher\s+than|lower\s+than"
            r"|delta|variance|benchmark)\b",
            q,
        )),

        # ── Year-over-year ───────────────────────────────────────────────────
        # Kept original (it was well-written). Added SPLY and annual variants.
        "wants_yoy": bool(re.search(
            r"\b(last\s+year.{0,20}(compared|vs|versus|against|than).{0,20}"
            r"(year\s+before|prior\s+year|previous\s+year)"
            r"|year.{0,10}over.{0,10}year"
            r"|yoy"
            r"|annual\s+(growth|change|comparison)"
            r"|same\s+(period|time)\s+last\s+year"
            r"|sply"
            r"|(compared|vs|versus).{0,20}(last\s+year|prior\s+year|previous\s+year|year\s+before)"
            r"|(last\s+year|this\s+year|prior\s+year|previous\s+year)"
            r".{0,20}(compared|vs|versus|difference|change|growth))\b",
            q,
        )),

        # ── Aggregate threshold → HAVING clause ──────────────────────────────
        # Added "having" itself, "minimum of", "no fewer than", "only show/include".
        # Also catches "more than N <entity>" patterns like "more than 10 staff".
        "wants_having_filter": bool(re.search(
            r"\bhaving\b"
            r"|\b(only\s+(show|include|display|return|list).{0,40}"
            r"(more\s+than|less\s+than|at\s+least|at\s+most|greater\s+than|fewer\s+than))"
            r"|\b(minimum\s+of|maximum\s+of|at\s+(minimum|maximum)"
            r"|no\s+less\s+than|no\s+more\s+than|no\s+fewer\s+than"
            r"|with\s+a\s+(count|total|sum|average|minimum|maximum)"
            r"|with\s+(total|count|sum).{0,20}(over|above|exceeding|more\s+than|less\s+than))\b"
            r"|\b(more\s+than|greater\s+than|at\s+least|exceeds?|threshold"
            r"|less\s+than|fewer\s+than).{0,30}"
            r"\b(count|total|sum|average|avg|number\s+of|amount|records?|entries"
            r"|staff|employees?|people|workers?|orders?|products?|items?|customers?"
            r"|transactions?|visits?|members?|users?|sales|purchases?)\b"
            r"|\b(count|total|sum|average|avg|number\s+of|amount).{0,30}"
            r"(more\s+than|greater\s+than|at\s+least|over|above"
            r"|less\s+than|fewer\s+than|under|below)\b"
            r"|\b(more\s+than|greater\s+than|at\s+least|fewer\s+than|less\s+than)"
            r"\s+\d+\b",
            q,
        )),

        # ── Top-N per group → window ROW_NUMBER ──────────────────────────────
        # Tightened: removed bare "by" on the right side, added "in every / from each".
        "wants_top_per_group": bool(re.search(
            r"\b(top|best|highest|lowest|worst|bottom|leading|top.ranked)"
            r".{0,25}(per\b|in\s+each|for\s+each|within\s+(each|every)"
            r"|across\s+each|in\s+every|from\s+each)\b"
            r"|\b(per\b|in\s+each|for\s+each|within\s+(each|every)|in\s+every|from\s+each)"
            r".{0,25}(top|best|highest|lowest|worst|bottom)\b",
            q,
        )),

        # ── Percentage / share of total ──────────────────────────────────────
        # Added distribution, ratio, makeup, out of total, as a percentage of.
        "wants_share": bool(re.search(
            r"\b(percent(age)?|proportion|share|contribution|distribution"
            r"|ratio|weighting?\b|makeup"
            r"|what\s+.{0,15}(percent|share|part|fraction|portion)\b"
            r"|how\s+much\s+.{0,15}(contribut|make\s+up|account|of\s+total)"
            r"|out\s+of\s+(the\s+)?total"
            r"|as\s+a\s+percent(age)?\s+of"
            r"|relative\s+to\s+(the\s+)?total"
            r"|breakdown\s+of)\b",
            q,
        )),

        # ── Anti-join / missing records ───────────────────────────────────────
        # This flag HARD-enforces a LEFT JOIN + IS NULL shape in the validator,
        # so a false positive here blocks otherwise-valid SQL. Bare "without" /
        # "missing" / "never" / "absent" / contractions are NOT enough on their
        # own — "total sales without tax" and "don't include cancelled orders"
        # are not anti-join questions. Each trigger needs record/entity context:
        #   • contraction preceded by a relative pronoun ("customers who haven't…")
        #   • "never" followed by a verb ("products never sold")
        #   • "without"/"missing" followed by a record noun ("items without receipts")
        "wants_missing_records": bool(re.search(
            r"\b(?:who|that|which)\s+(?:haven'?t|hasn'?t|didn'?t|don'?t|doesn'?t|weren'?t|wasn'?t)\b"
            r"|\bnever\s+(?:been\s+)?(?:had|has|have|sold|bought|paid|made|placed|shipped|\w{2,}ed)\b"
            r"|\babsent\s+from\b"
            r"|\b(unmatched|unlinked|orphan(?:ed)?"
            r"|no\s+activity|no\s+transactions?|no\s+orders?|no\s+purchases?"
            r"|no\s+matching\s+(shipment|invoice|record|receipt|order|transaction|row|entry)s?"
            r"|zero\s+(orders?|sales?|records?|transactions?|visits?)"
            r"|not\s+placed|not\s+made|not\s+submitted|not\s+attended|skipped|missed)\b"
            r"|\b(not\s+in|have\s+no|has\s+no|with\s+no|without|missing|lacks?|lacking"
            r"|do\s+not\s+have|does\s+not\s+have|never\s+had)\b"
            r".{0,40}\b(records?|orders?|transactions?|sales?|attendances?|entries|matches?"
            r"|results?|absences?|purchases?|receipts?|shipments?|invoices?|visits?|activities)\b"
            # Noun-first form needs a strong trailing phrase — bare "never" /
            # "without" / "missing" after a noun is usually exclusion phrasing
            # ("sales without tax", "amounts with missing due dates"), not an
            # anti-join ask.
            r"|\b(records?|orders?|transactions?|sales?|attendances?|entries|matches?"
            r"|results?|absences?|purchases?|receipts?|shipments?|invoices?|visits?|activities)\b"
            r".{0,40}\b(not\s+in|never\s+had|have\s+no|has\s+no|(?:are|is|were|was)\s+missing|(?:were|was)\s+never)\b"
            r"|\b(employees?|customers?|products?|items?|users?|patients?)\b"
            r".{0,50}\b(no|never|without|not)\b.{0,30}"
            r"\b(absences?|orders?|sales?|records?|transactions?|visits?|attendances?|purchases?|receipts?|shipments?|invoices?)\b",
            q,
        )),

        # ── Conditional aggregation / pivot ───────────────────────────────────
        # Major expansion: added "count by status", "how many X and how many Y",
        # "X count vs Y count", employment/contract type splits.
        "wants_conditional_split": bool(re.search(
            r"\b(active\s*.{0,10}(vs\.?|versus|and|compare).{0,10}(inactive|terminated|former)"
            r"|male\s*.{0,10}(vs\.?|versus|and).{0,10}female"
            r"|split\s+by|side\s+by\s+side|pivot"
            r"|count\s+(by|for\s+each|per)\s+(status|type|category|group|gender|department)"
            r"|breakdown\s+by\s+(status|type|category|gender|contract|employment)"
            r"|how\s+many\s+.{0,30}and\s+how\s+many"
            r"|(count|number)\s+of\s+.{0,25}(vs\.?|versus|and)\s+.{0,25}(count|number)\s+of"
            r"|by\s+(employment\s+type|contract\s+type|gender|marital\s+status"
            r"|grade|band|pay\s+type))\b",
            q,
        )),

        # ── Month-over-month / quarter-over-quarter ───────────────────────────
        # Fixed: removed bare "mom" (too many false positives on unrelated text).
        # Added "period over period", "month-on-month", "weekly change".
        "wants_mom_qoq": bool(re.search(
            r"\b(month.{0,5}over.{0,5}month|month.on.month"
            r"|quarter.{0,5}over.{0,5}quarter|quarter.on.quarter"
            r"|qoq\b|period.{0,5}over.{0,5}period|\bpop\b"
            r"|monthly\s+(change|growth|trend|comparison|variation|progression)"
            r"|quarterly\s+(change|growth|trend|comparison|variation|progression)"
            r"|weekly\s+(change|growth|trend|comparison)"
            r"|how\s+.{0,20}changed\s+.{0,20}(each\s+month|monthly|each\s+quarter|quarterly)"
            r"|(each\s+month|each\s+quarter|by\s+month|by\s+quarter)"
            r".{0,30}(change|growth|trend|compare|comparison))\b",
            q,
        )),

        # ── Cumulative / running total ────────────────────────────────────────
        # Added MTD, QTD, "to date", "so far this year/month".
        "wants_cumulative": bool(re.search(
            r"\b(cumulative|running\s+total|cumulative\s+(sum|count|total)"
            r"|year.{0,5}to.{0,5}date|\bytd\b|\bmtd\b|\bqtd\b"
            r"|total\s+so\s+far|running\s+sum|accumulated|progressive\s+total"
            r"|to\s+date\b|so\s+far\s+this\s+(year|month|quarter)"
            r"|aggregate\s+total|rolling\s+total)\b",
            q,
        )),

        # ── Rolling / moving average ─────────────────────────────────────────
        # Added 7-day/30-day shorthand, sliding window, EMA.
        "wants_rolling": bool(re.search(
            r"\b(rolling\s+(average|avg|mean|total)"
            r"|moving\s+(average|avg|mean)"
            r"|trailing\s+(average|avg)"
            r"|smooth(?:ed|ing)?\s*(trend|curve|line|out)?"
            r"|sliding\s+window|window\s+(average|avg)"
            r"|\d+.?(day|week|month|period).?(rolling|moving|trailing|average|avg)"
            r"|7.day|30.day|90.day|\bema\b)\b",
            q,
        )),

        # ── Named period filter (Q/H/month/fiscal) ───────────────────────────
        # Added month abbreviations, fiscal year/quarter, this/last month|quarter,
        # MTD, QTD, YTD (shared with cumulative — both hints are useful).
        # "quarterly/annually" also signals a named period granularity.
        "wants_named_period": bool(re.search(
            r"\b(q[1-4]\b|quarter\s+[1-4]"
            r"|first\s+quarter|second\s+quarter|third\s+quarter|fourth\s+quarter"
            r"|quarterly|annually|yearly"
            r"|h[12]\b|first\s+half|second\s+half"
            r"|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?"
            r"|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
            r"|\bfy\b|financial\s+year|fiscal\s+(year|quarter|half)"
            r"|calendar\s+(year|quarter)"
            r"|this\s+(month|quarter|week|year)\b"
            r"|last\s+(month|quarter|week|year)\b"
            r"|last\s+\d+\s+(month|week|day)s?"
            r"|\bytd\b|\bmtd\b|\bqtd\b)\b",
            q,
        )),

        # ── Ranking / leaderboard ─────────────────────────────────────────────
        # Added "who has the most/highest", standings, "top N by", positions.
        "wants_ranking": bool(re.search(
            r"\b(rank(?:ed|ing)?|leaderboard|top\s+performer|scoreboard|score\s+board"
            r"|ordered\s+by\s+performance|best\s+performing|worst\s+performing"
            r"|by\s+performance|ranked\s+list|standings?"
            r"|league\s+table|position(?:s|ing)?"
            r"|who\s+(is|are)\s+the\s+(best|top|highest|lowest|worst|leading)"
            r"|who\s+has\s+the\s+(most|highest|lowest|best|worst|fewest)"
            r"|most\s+productive|least\s+productive"
            r"|top\s+\d+\s+(by|based\s+on|for|in)\b)\b",
            q,
        )),
        "wants_top_n": top_n is not None,
    }


def summarize_query_intent(question: str) -> str:
    intent = analyze_query_intent(question)
    labels: list[str] = []
    if intent["wants_distinct_count"] and intent["has_employee_scope"]:
        labels.append("distinct employee counting")
    elif intent["has_employee_scope"]:
        labels.append("employee-focused query")
    if intent["wants_grouping"]:
        labels.append("grouped breakdown")
    if intent["wants_status_filter"]:
        labels.append("categorical status/value filtering")
    if intent["wants_names"]:
        labels.append("name lookup")
    if intent["wants_time_series"]:
        labels.append("time-series analysis")
    if intent["wants_comparison"]:
        labels.append("comparison framing")
    if intent["wants_yoy"]:
        labels.append("year-over-year comparison")
    if intent["wants_having_filter"]:
        labels.append("aggregate threshold filter (HAVING)")
    if intent["wants_top_per_group"]:
        labels.append("top-N per group (window function)")
    if intent["wants_share"]:
        labels.append("percentage / share of total")
    if intent["wants_missing_records"]:
        labels.append("anti-join / missing records")
    if intent["wants_conditional_split"]:
        labels.append("conditional aggregation / pivot")
    if intent["wants_mom_qoq"]:
        labels.append("month-over-month / quarter-over-quarter trend")
    if intent["wants_cumulative"]:
        labels.append("cumulative / running total")
    if intent["wants_rolling"]:
        labels.append("rolling / moving average")
    if intent["wants_named_period"]:
        labels.append("named period filter (Q/H/month)")
    if intent["wants_ranking"]:
        labels.append("ranking / leaderboard")
    if intent["wants_top_n"]:
        top_n = detect_top_n_intent(question)
        if top_n:
            labels.append(f"top-{top_n.limit} result limit")
    return ", ".join(labels)


def build_generic_query_hints(question: str) -> str:
    """
    Return safe, cross-client guidance for common analytics phrasing.

    This is intentionally generic language understanding, not a client-specific
    semantic registry. It helps the SQL model interpret ordinary requests such
    as "unique employee count" and lightly misspelled filter values.
    """
    q = (question or "").strip().lower()
    if not q:
        return ""

    intent = analyze_query_intent(question)
    top_n = detect_top_n_intent(question)
    hints: list[str] = [
        "GENERIC QUERY INTERPRETATION RULES:",
        "- Exact schema-backed categorical values from the provided context are authoritative and must be preserved exactly, even when they look misspelled. Only normalize a value when that exact literal is absent from schema or business context.",
    ]

    if intent["wants_distinct_count"] and intent["has_employee_scope"]:
        hints.append(
            "- When the user asks for a unique employee count or distinct employee total, use COUNT(DISTINCT stable employee key). Prefer EMPLOYEE_ID, EMPLOYEE_NUMBER, PERSON_ID, PERSON_NUMBER, STAFF_ID, or USER_ID over employee names when such keys exist."
        )

    if intent["has_employee_scope"] and intent["wants_grouping"]:
        hints.append(
            "- When the user asks for employees by a category such as department, group by that category and count distinct employees rather than counting raw attendance or event rows unless the question explicitly asks for record volume."
        )

    if intent["wants_status_filter"]:
        hints.append(
            "- Phrases like 'marked as', 'who are', or 'with status' usually mean a filter on a categorical status or value column, not a different metric."
        )

    if intent["wants_names"] and intent["has_employee_scope"]:
        hints.append(
            "- If the user asks for employee names, return names after applying the requested filters; do not convert the request into an aggregate unless they explicitly ask for a count or ranking."
        )

    if intent["wants_yoy"]:
        hints.append(
            "- YEAR-OVER-YEAR COMPARISON DETECTED: The user wants to compare a metric across two "
            "consecutive years. Follow the YEAR-OVER-YEAR / PERIOD COMPARISON RULE in the system "
            "prompt exactly:\n"
            "  • Use a CTE to compute per-year aggregates; CAST the year column to INT.\n"
            "  • Derive the anchor year as MAX(CAST(year_col AS INT)) from the data — do NOT "
            "use GETDATE()/CURRENT_DATE/SYSDATE as the anchor and do NOT write MAX(col)-1 in a "
            "WHERE clause.\n"
            "  • LEFT JOIN the CTE to itself on prev_year = curr_year - 1.\n"
            "  • Always output: current year value, previous year value, absolute difference, "
            "and percentage change rounded to 2 decimal places.\n"
            "  • If approved metric formulas are present, use them inside the CTE aggregation — "
            "never substitute KB column names for approved formulas in a YoY query."
        )

    if intent["wants_having_filter"]:
        hints.append(
            "- HAVING FILTER DETECTED: The user wants to filter groups by an aggregate threshold "
            "(e.g. 'more than N', 'at least N'). Apply the HAVING RULE: filter on the aggregate "
            "in HAVING, not in WHERE. Use COUNT(*)/SUM(col) etc. directly in HAVING — do not "
            "reference an alias."
        )

    if intent["wants_top_per_group"]:
        hints.append(
            "- TOP-N PER GROUP DETECTED: The user wants the best/top/highest N records within "
            "each group. Apply the TOP-N PER GROUP RULE: use ROW_NUMBER() OVER (PARTITION BY "
            "group_col ORDER BY metric DESC) in a CTE, then filter WHERE rn <= N."
        )

    if intent["wants_share"]:
        hints.append(
            "- PERCENTAGE/SHARE DETECTED: The user wants to see proportions or contributions. "
            "Apply the PERCENTAGE OF TOTAL RULE: use SUM(metric)*100.0/SUM(SUM(metric)) OVER () "
            "in a single query with GROUP BY — do not use a separate subquery for the denominator."
        )

    if intent["wants_missing_records"]:
        hints.append(
            "- ANTI-JOIN / MISSING RECORDS DETECTED: The user wants records with no matching "
            "rows in another table. Apply the ANTI-JOIN RULE: use LEFT JOIN … WHERE right_key "
            "IS NULL. The FROM table must be the source/parent table containing the records "
            "to list; the missing-side table belongs on the RIGHT side of the LEFT JOIN. "
            "Do NOT answer by querying only the missing-side table or only checking a measure "
            "column for NULL. Do NOT use NOT IN (fails with NULLs) or NOT EXISTS."
        )

    if intent["wants_conditional_split"]:
        hints.append(
            "- CONDITIONAL SPLIT DETECTED: The user wants side-by-side counts for different "
            "categories (e.g. active vs inactive, male vs female). Apply the CONDITIONAL "
            "AGGREGATION RULE: use SUM(CASE WHEN status = 'X' THEN 1 ELSE 0 END) in a single "
            "query — do not use multiple subqueries or UNION."
        )

    if intent["wants_mom_qoq"]:
        hints.append(
            "- MONTH-OVER-MONTH / QUARTER-OVER-QUARTER DETECTED: The user wants period-over-period "
            "change. Apply the staged MoM/QoQ RULE: (1) aggregate the approved metric by the "
            "resolved business-date period in period_totals, (2) compute LAG(METRIC) from that "
            "alias in period_comparison, and (3) calculate difference and PCT_CHANGE in the final "
            "SELECT. Never use LAG(SUM(...)). Use a native date-dimension value directly; only "
            "convert a numeric key when schema metadata identifies it as YYYYMMDD. Date-role words "
            "such as booked/paid/dispensed select a date JOIN and do not imply a status filter. "
            "Always output: period, current value, prior value, difference, and PCT_CHANGE rounded "
            "to 2 decimal places."
        )

    if intent["wants_cumulative"]:
        hints.append(
            "- CUMULATIVE / RUNNING TOTAL DETECTED: The user wants a running/cumulative sum. "
            "Apply the RUNNING TOTAL RULE: use SUM(SUM(metric)) OVER (ORDER BY date_col ROWS "
            "UNBOUNDED PRECEDING) — nested aggregate window. GROUP BY the period first, then "
            "accumulate with the window function."
        )

    if intent["wants_rolling"]:
        hints.append(
            "- ROLLING / MOVING AVERAGE DETECTED: The user wants a smoothed average over a "
            "sliding window. Apply the MOVING AVERAGE RULE: AVG(metric) OVER (ORDER BY date_col "
            "ROWS BETWEEN N-1 PRECEDING AND CURRENT ROW). Default window = 3 periods if the "
            "user didn't specify. Cast integer columns to FLOAT for Azure SQL."
        )

    if intent["wants_named_period"]:
        hints.append(
            "- NAMED PERIOD FILTER DETECTED: The user referred to a specific quarter (Q1-Q4), "
            "half (H1/H2), month name, or 'last N months/weeks'. Apply the NAMED PERIOD FILTERS "
            "from the SQL syntax rules — use DATEPART/EXTRACT/QUARTER/MONTH with the correct "
            "integer mapping. Do NOT use GETDATE()/SYSDATE — anchor to MAX(date_col) in the data."
        )

    if intent["wants_ranking"] and not top_n:
        hints.append(
            "- RANKING DETECTED: The user wants entities ordered/ranked by a metric. Apply the "
            "RANKING RULE: include RANK() OVER (ORDER BY SUM(metric) DESC) AS RANK alongside the "
            "aggregate in the SELECT list. Use DENSE_RANK() only if the user mentions 'no gaps'. "
            "Always ORDER BY the rank column."
        )

    if top_n:
        direction = "ASC" if top_n.direction == "ascending" else "DESC"
        tie_rule = (
            "Include all rows tied at the Nth position using TOP (N) WITH TIES or RANK/DENSE_RANK <= N."
            if top_n.tie_policy == "include_ties"
            else "Return exactly N rows; use TOP (N), or ROW_NUMBER() followed by rn <= N. Do not use RANK/DENSE_RANK because ties can return more than N rows."
        )
        scope_rule = (
            "This is Top-N per group: use ROW_NUMBER() OVER (PARTITION BY the group dimension ORDER BY the metric) and filter rn <= N."
            if top_n.per_group
            else "This is a global Top-N: apply the limit to the final result set."
        )
        hints.append(
            f"- TOP-{top_n.limit} CONTRACT: {scope_rule} {tie_rule} "
            f"Order the requested metric {direction} and add a stable dimension as a secondary ordering key. "
            "Never reinterpret Top-N as a threshold such as value > MIN(value FROM TopN); that can return zero rows when the boundary is tied."
        )

    summary = summarize_query_intent(question)
    if summary:
        hints.append(f"- Query-intent summary: {summary}.")

    if len(hints) == 1:
        return ""
    return "\n".join(hints) + "\n"
