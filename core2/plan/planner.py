"""From a question to a typed plan: the one step where an AI decides about a question (DESIGN §7).

The prompt is a stable prefix (rules, the plan's schema, the workspace catalog and
a few examples written in the workspace's own words), byte-identical for a model
version so providers cache it, then a short tail (today, the conversation, the
member names found in the question, the question). The answer must be one JSON
object that validates as a :class:`Plan`; the plan is then checked against the
model by resolving it. A plan that fails gets one repair round with the problems
and the allowed options; one that still fails becomes a clarification, never a
guess.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from core2.bootstrap import names
from core2.model.schema import DateRole, Measure, SemanticModel
from core2.plan.catalog import catalog_text
from core2.plan.followup import asks_series, keeps_period, points_at_period
from core2.plan.ir import Clarify, Plan, Sort, Window, schema_text
from core2.plan.values import Masked, ValueMatch, mask, placeholder, unmask
from core2.resolve import paths as P
from core2.resolve.resolver import Context, ResolveError, resolve

Complete = Callable[[str, str], str]       # (stable system prompt, question tail) -> the model's text

RULES = """You turn a business question into a query plan for QueryBot. You never write SQL.
Answer with exactly one JSON object that matches the PLAN SCHEMA. No prose, no markdown.

How to plan:
1. kind: "query" for questions about the data. "clarify" only when the question is genuinely ambiguous
   (two measures fit equally, a date that could mean two dates, a name that fits several members): put the
   question and 2-5 options from the catalog in clarify. "describe_data" for questions about the data
   itself rather than its numbers: "what data do you have", "what can you tell me about my data", "what
   can I ask", "what does X mean", "how is X calculated", "what dates does the data cover", "how far back
   does it go", "what do you know about customers", "what can I ask about purchasing". Put what it asks
   about in about: measure, group or date slugs, or a subject's name as written after ## in the catalog;
   leave about empty for the data as a whole. "smalltalk" for greetings and thanks. "unsupported" when
   nothing in the catalog answers it; say why in notes.
2. measures: the measure slugs the question names. Never swap in a different measure because it is close
   (sales is not margin, quantity is not value, count of rows is not count of orders). When the question
   needs a ratio, difference, sum or product of two measures (a margin %, revenue per order), put it in
   derived with a short name, the op and the two slugs (scale 100 for a percentage). Days between two
   dates of the same rows ("days from order to invoice", "how long to invoice", "delivery delay", "lead
   time") go in durations: a short name, start and end (two date slugs on one table, the earlier event as
   start) and agg (avg; max for "longest", min for "shortest", sum for "total days"). To keep only rows by
   it ("invoiced more than 14 days after ordering"), add a filter whose field is the duration's name (op
   gt, gte, lt, lte or between; values in days), with measure false when the rows are only counted. To
   keep groups by their figure instead ("carriers whose average delivery takes over 4 days", "ship methods
   with a transit time of 3 days" when no field holds it), keep the duration as a measure and set total
   true on its filter. Sort by a duration with its name.
3. group_by: attribute or entity slugs for "by X", "per X", "each X", "which X"; an entity slug groups by
   its name. A time: attribute for "by weekday", "by month of the year", "weekends". When the catalog lists
   roles for an entity and the question names one ("by ship-to customer", "the customer's home store"),
   put it in via: {"<the group_by slug>": "<role>"}; with no role named, leave via empty.
4. time.grain makes a series: "monthly", "by month", "weekly", "per quarter", "trend", "over time" (month
   when unsaid). No grain for one total.
5. time.window, written as the user said it; never compute relative dates yourself:
   "last 6 months" -> {"kind": "last", "unit": "month", "n": 6} (complete months before the current one);
   "this year", "year to date", "so far this year" -> {"kind": "this", "unit": "year"};
   "last month" -> {"kind": "previous", "unit": "month"}; "last year" -> {"kind": "previous", "unit": "year"};
   "in March 2026" -> {"kind": "between", "start": "2026-03-01", "end": "2026-03-31"};
   "since January" -> {"kind": "since", "start": "<this year>-01-01"};
   fiscal periods set "fiscal": true: "this fiscal year" -> {"kind": "this", "unit": "year", "fiscal": true};
   "last fiscal year" -> {"kind": "previous", "unit": "year", "fiscal": true}; a fiscal year, half or quarter
   named by its number ("FY2026", "fiscal year 2026", "H1 FY26", "Q3 FY26") -> kind "between" with the dates
   FISCAL YEARS gives for it. A quarter, half or year the question does not call fiscal ("Q2 2026", "in 2025")
   is the calendar's: "Q2 2026" -> 2026-04-01 to 2026-06-30.
   No window when the user gave none: everything is the default. "Latest", "current", "now", "on the latest
   snapshot" for a stock, balance or other snapshot figure is no window either: the latest snapshot is the
   default.
6. time.date: a date slug only when the question names which date ("by ship date", "invoiced in May",
   "expiring in 2028"); otherwise null and the measure's default date is used. A second date in the
   question ("ordered in March and delivered in April") is a filter on that date slug with ISO dates.
7. time.compare: "vs last year", "year over year" -> {"kind": "same_period_last_year"};
   "vs last month", "month over month", "vs the previous period" -> {"kind": "previous_period"};
   "from February to March", "March vs February" -> window March and
   {"kind": "window", "window": February}; the intent is then "compare".
8. filters: a member goes in a filter on its attribute with op "eq" (or "in" for several), using the exact
   stored value from VALUE MATCHES or a previous plan; "excluding X" -> "ne"/"not_in". VALUE MATCHES holds
   only the names the reader put in quotes ("OEM") and the members of the answer on screen they point at:
   a word of the question that is not quoted is never a member, even when one is called that ("available
   stock" is not a stock status AVAILABLE, "for northline" is no customer): answer without that filter.
   A code the catalog lists, chosen for what the question means ("units received" -> movement type RCV),
   may filter. A condition on a total ("customers with more than 10,000 in sales") is a filter on the
   measure slug. A name in VALUE MATCHES can also be what a measure measures ("cost of goods sold" is a
   ledger account and the cost on sales lines): read it as the member only when that member's table can
   be broken down and filtered as the question asks; otherwise use the measure.
   A field the catalog calls a number (a price, a weight, days in transit) is compared on its value: op eq,
   gt, gte, lt, lte or between with numbers ("a transit time of exactly 3 days" -> eq [3]); sort by its
   slug ("the cheapest"), and group by it only when the catalog lists its values. An identifier (an
   invoice or tracking number) is filtered with eq on the value the reader wrote, and listed or ranked by
   ("the 10 largest invoices"). Free text is filtered with contains on the words the reader looks for.
9. sort and limit: "top 5" -> sort by the measure (desc) with limit 5; "bottom 5", "least" -> desc false.
   "grew the most" -> sort by "change" desc; "biggest drop" -> "change" with desc false.
10. intent: value | breakdown | trend | compare | rank | share | list | count | drivers | forecast. "List the
   X", "what are the X" with no measure -> intent list with group_by [X] and no measures. "How many X" ->
   intent count with the "Number of X" measure; "how many X bought / ordered / were invoiced in a period"
   also sets time.date to the date of that event (the invoice date for "bought"), never a date of X's own.
   "Share of", "% of total", "contribution" -> intent share.
   "Why did X drop/rise/change", "what drove X", "explain the change in X" -> intent drivers with the
   measure; time.window is the period asked about ("in March" -> March 2026), time.compare only when the
   question names the other period (else the period before is used); drivers.dimensions only when the
   question names where to look ("why ... by region"). A "why" about the answer on screen keeps that
   answer's time, grain included.
   "Forecast X", "predict X", "X for the next N months", "where will X be" -> intent forecast with
   forecast.periods N (3 when unsaid) and time.grain (month when unsaid); a window only when the question
   limits the history to use.
11. Follow-ups: PREVIOUS PLAN is the plan behind the last answer. When the new question changes it ("by
   week instead", "only for X", "what about returns", "top 5 of those"), return the whole new plan with
   follow_up "refine"; a question about something else is follow_up "new". "Those", "them", "the same
   customers" are the members of the answer on screen, whose names you are not given: never filter on a
   name that is not in VALUE MATCHES or a previous plan, and never a filter with no values: to answer for
   them, keep the previous plan's group_by, sort and limit. "The first one", "the lowest one", "that
   division" are in VALUE MATCHES when the member they point at is known: filter on its value. Never put
   such words themselves ("lowest", "that division") in a filter; when they are not matched, ask (clarify). To follow them into another period ("how did those
   same customers do in 2025?"), keep the previous plan's group_by, sort, limit and time.window, and put the
   other period in time.compare, with intent "compare".
   When it could be either (a short, complete question after an answer narrowed to a member or a period,
   as "how many orders?" after net sales for North in March), plan it as a new question with follow_up
   "unsure": the reader is asked which they meant. When READING says how the question was read, follow it.
12. notes: one short line for each assumption the user did not state.
13. chart: only when the question says how to show it ("as a pie", "as a donut" or "ring" -> "donut",
   "bar chart", "stacked bars" -> "stacked", "as a heatmap" or "grid" -> "heatmap", "as a scatter" or
   "plotted against each other" -> "scatter", "on a map" -> "map", "just the table" -> "table");
   on a follow-up ("show that as a pie") return the previous plan with chart set and follow_up "refine".
14. include_left_out: true only when the question asks for rows a table LEAVES OUT by default (listed in the
   catalog), as "including cancelled orders" or "with test orders too", or asks about those rows themselves
   ("revenue on cancelled lines", "which customer type cancels the most"); otherwise leave it out.
"""


@dataclass
class Turn:
    question: str
    plan: Plan | None = None
    masked: Masked | None = None        # what the AI was shown, when member values are withheld
    shown: list[tuple] = field(default_factory=list)   # the answer's members in order: (attribute, value, number)


@dataclass
class Outcome:
    plan: Plan
    repaired: bool = False
    problems: list[str] = field(default_factory=list)   # what the last attempt got wrong, if anything
    raw: list[str] = field(default_factory=list)        # the model's answers, for the record
    masked: Masked | None = None                        # the placeholders used, when values are withheld
    tried: Plan | None = None                           # the last plan that failed its check, if any


def _examples(model: SemanticModel) -> str:
    """Three worked examples in this workspace's own words (deterministic for a model version).

    They use the main fact's leading measure (money first) and an entity it reaches,
    so they never teach a combination the model cannot answer.
    """
    measures = sorted((m for m in model.measures.values() if not m.hidden and m.kind != "count"
                       and m.additivity == "additive" and m.default_date),
                      key=lambda m: (-model.tables[m.table].row_count, m.format != "currency", m.slug))
    if not measures:
        return ""
    m: Measure = measures[0]
    entities = sorted((e for e in model.entities.values()
                       if e.table != m.table and P.all_paths(model, m.table, e.table)), key=lambda e: e.slug)
    lines = ["EXAMPLES (in this workspace's words)"]
    plan: dict[str, Any] = {"kind": "query", "intent": "trend", "measures": [m.slug],
                            "time": {"grain": "month", "window": {"kind": "last", "unit": "month", "n": 6}}}
    lines.append(f"Q: {m.business_name.lower()} by month for the last 6 months\nA: {json.dumps(plan)}")
    if entities:
        e = entities[0]
        plan = {"kind": "query", "intent": "rank", "measures": [m.slug], "group_by": [e.slug],
                "sort": [{"by": m.slug, "desc": True}], "limit": 5,
                "time": {"window": {"kind": "this", "unit": "year"}}}
        lines.append(f"Q: top 5 {names.plural(e.business_name.lower())} by {m.business_name.lower()} this year\n"
                     f"A: {json.dumps(plan)}")
        plan = {"kind": "query", "intent": "compare", "measures": [m.slug], "group_by": [e.slug],
                "time": {"window": {"kind": "previous", "unit": "month"},
                         "compare": {"kind": "previous_period"}},
                "sort": [{"by": "change", "desc": True}]}
        lines.append(f"Q: which {names.plural(e.business_name.lower())} grew the most last month\n"
                     f"A: {json.dumps(plan)}")
    days = _two_dates(model, m.table)
    if days is not None:
        start, end = days
        name = f"Days from {start.name.lower()} to {end.name.lower()}"
        plan = {"kind": "query", "intent": "value", "measures": [],
                "durations": [{"name": name, "start": start.slug, "end": end.slug, "agg": "avg"}],
                "time": {"window": {"kind": "previous", "unit": "year"}}}
        lines.append(f"Q: average days from {start.name.lower()} to {end.name.lower()} last year\n"
                     f"A: {json.dumps(plan)}")
    plan = {"kind": "query", "intent": "drivers", "measures": [m.slug],
            "time": {"window": {"kind": "previous", "unit": "month"}}}
    lines.append(f"Q: why did {m.business_name.lower()} drop last month?\nA: {json.dumps(plan)}")
    plan = {"kind": "query", "intent": "forecast", "measures": [m.slug], "time": {"grain": "month"},
            "forecast": {"periods": 6}}
    lines.append(f"Q: forecast {m.business_name.lower()} for the next 6 months\nA: {json.dumps(plan)}")
    plan = {"kind": "describe_data", "about": [m.slug]}
    lines.append(f"Q: how is {m.business_name.lower()} calculated?\nA: {json.dumps(plan)}")
    return "\n".join(lines)


def _analyses(model: SemanticModel) -> str:
    """How to ask for a figure worked out over the rows of a plain answer, one rule and one example for each
    kind this workspace's data can give (per member, ranges, running figures, ranks in groups, who had none)."""
    measures = sorted((m for m in model.measures.values() if not m.hidden and m.kind != "count"
                       and m.additivity == "additive" and m.default_date),
                      key=lambda m: (-model.tables[m.table].row_count, m.format != "currency", m.slug))
    if not measures:
        return ""
    m: Measure = measures[0]
    what = m.business_name.lower()
    # The thing with the most members first (customers before regions): what a figure per member is asked of.
    entities = sorted((e for e in model.entities.values()
                       if e.table != m.table and P.all_paths(model, m.table, e.table)),
                      key=lambda e: (-(e.members or 0), e.slug))
    # What the main table's rows count: its orders (a distinct count), or its rows when each is one thing (fills).
    keyed = len(model.tables[m.table].primary_key or model.tables[m.table].grain) == 1
    counts = sorted((c for c in model.measures.values() if not c.hidden and c.table == m.table and (
                     getattr(c.expr, "agg", "") == "count_distinct" and getattr(c.expr, "column", None) or
                     getattr(c.expr, "agg", "") == "count" and not getattr(c.expr, "column", None) and keyed)),
                    key=lambda c: (getattr(c.expr, "agg", "") != "count_distinct", c.slug))
    numbers = sorted((a for a in model.attributes.values() if a.kind == "number" and a.column in model.columns
                      and P.all_paths(model, m.table, model.columns[a.column].table) or
                      a.kind == "number" and a.column in model.columns and model.columns[a.column].table == m.table),
                     key=lambda a: a.slug)
    lines = ["ANALYSES: a figure worked out over the rows of a plain answer. At most one per plan, never with "
             "time.compare; the plan's other parts (measures, group_by, filters, time) say what the rows are."]
    count = counts[0] if counts else None
    things = count.business_name.lower().removeprefix("number of ") if count is not None else "rows"
    if entities or counts:
        e = entities[0] if entities else None
        lines.append(
            '- per: each member\'s figure first, then one figure over the members. of: an entity ("per customer"), '
            'the count measure of what is counted ("per order" -> its "number of" measure), or none for each '
            "row's own value (days between two dates of each row). summary avg (default) | median | percentile "
            '(p: "90th percentile" -> 90) | min | max | count or share with op and values ("customers with 3 or more '
            'orders" -> count, gte [3]; "what share of customers ..." -> share). everyone true only when members '
            "with none count too (\"including those with no orders\").")
        if e is not None:
            plan = {"kind": "query", "intent": "value", "measures": [m.slug], "per": {"of": e.slug, "summary": "median"},
                    "time": {"window": {"kind": "this", "unit": "year"}}}
            lines.append(f"Q: median {what} per {e.business_name.lower()} this year\nA: {json.dumps(plan)}")
        if count is not None:
            plan = {"kind": "query", "intent": "value", "measures": [m.slug], "per": {"of": count.slug}}
            lines.append(f"Q: average {what} per {names.singular(things)}\nA: {json.dumps(plan)}")
    days = _two_dates(model, m.table)
    if numbers or days is not None or entities:
        lines.append(
            "- bands: rows or members put in ranges of a value, counted per range. of: a number field or a "
            "duration's name for each row (\"orders by delivery time\"); a measure with per for each member's "
            "figure (\"customers by spend band\"; \"customers with 1, 2 to 4 and 5 or more orders\": of the count of "
            "orders, per customer, edges [1, 2, 5]). edges: the reader's own lower edges (\"under 3, 3 to 7, 8 or "
            "more\" -> [3, 8]); [] lets the data choose round ones.")
        if days is not None:
            start, end = days
            name = f"Days from {start.name.lower()} to {end.name.lower()}"
            plan = {"kind": "query", "intent": "breakdown", "measures": [(count or m).slug],
                    "durations": [{"name": name, "start": start.slug, "end": end.slug, "measure": False}],
                    "bands": {"of": name, "edges": [3, 8]}}
            asked = f"how many {things} took" if count is not None else f"{what} by how long it took,"
            lines.append(f"Q: {asked} under 3, 3 to 7 and 8 or more days from {start.name.lower()} to "
                         f"{end.name.lower()}\nA: {json.dumps(plan)}")
    lines.append(
        '- rolling: "running total", "cumulative" -> kind running; "year to date by month" -> running with reset '
        '"year" ("fiscal_year" for the fiscal year); "3-month moving average" -> moving_average, periods 3; "rolling '
        '12-month total" -> moving_sum, periods 12. Always with time.grain; time.window is the periods shown.')
    plan = {"kind": "query", "intent": "trend", "measures": [m.slug], "rolling": {"kind": "moving_average", "periods": 3},
            "time": {"grain": "month", "window": {"kind": "last", "unit": "month", "n": 12}}}
    lines.append(f"Q: 3-month moving average of {what} over the last 12 months\nA: {json.dumps(plan)}")
    if len(entities) >= 2:
        a, b = entities[0], entities[-1]          # the most members ranked in each of the fewest
        lines.append(
            "- within: a rank that starts again in each member of a grouping: group_by holds both, within.group "
            "the one each rank starts again in, within.top the first n of each.")
        plan = {"kind": "query", "intent": "rank", "measures": [m.slug], "group_by": [b.slug, a.slug],
                "within": {"group": b.slug, "top": 3}, "sort": [{"by": m.slug, "desc": True}]}
        lines.append(f"Q: top 3 {names.plural(a.business_name.lower())} in each {b.business_name.lower()}\n"
                     f"A: {json.dumps(plan)}")
    if entities:
        e = entities[0]
        lines.append(
            "- absent: the members with no matching rows, intent list. of: the entity; measure: whose rows are the "
            "activity; time.window: when (none: never). without: filters on which rows count (\"never bought "
            "X\"); had: filters on rows they did have (\"bought A but never B\" -> had A, without B); "
            "active_before true for \"stopped\", \"lapsed\", \"no longer\".")
        plan = {"kind": "query", "intent": "list", "absent": {"of": e.slug, "measure": (count or m).slug},
                "time": {"window": {"kind": "last", "unit": "day", "n": 90}}}
        lines.append(f"Q: which {names.plural(e.business_name.lower())} had no {things if count else what} in the "
                     f"last 90 days\nA: {json.dumps(plan)}")
    return "\n".join(lines)


def _two_dates(model: SemanticModel, table: str) -> tuple[DateRole, DateRole] | None:
    """Two daily event dates of ``table`` for the worked example, the earlier event first.

    The table's default and a date whose data starts no later (orders come before their
    invoices); when none does, the default and the date starting soonest after it.
    """
    dates = [r for r in model.date_roles.values()
             if r.table == table and r.kind == "event" and r.granularity != "month" and r.first is not None]
    default = next((r for r in dates if r.is_default), None)
    if default is None or default.first is None:
        return None
    day = default.first
    others = [r for r in dates if r.key != default.key and r.first is not None]
    earlier = sorted((r for r in others if (r.first or day) <= day), key=lambda r: (r.first or day, r.slug))
    if earlier:
        return earlier[0], default
    later = sorted(others, key=lambda r: (r.first or day, r.slug))
    return (default, later[0]) if later else None


def stable_prompt(model: SemanticModel, *, values_allowed: bool = True, personal_shown: bool = False) -> str:
    """Rules, schema, catalog and examples: identical for every question on a model version."""
    return "\n\n".join(part for part in (
        RULES, "PLAN SCHEMA (JSON Schema)\n" + schema_text(),
        catalog_text(model, values_allowed=values_allowed, personal_shown=personal_shown), _examples(model),
        _analyses(model)) if part)


def fiscal_years_text(model: SemanticModel, today: dt.date) -> str:
    """The workspace's fiscal years as dates, for a question that names one by its number.

    "Fiscal year 2026" was planned as this fiscal year: told only that fiscal periods are
    fiscal, the AI could not know that FY2026 ended months ago (a year starting in April,
    named by the year it ends). Nothing is said where the fiscal year is the calendar year.
    """
    start = model.settings.fiscal_year_start_month or next(
        (c.fiscal_year_start_month for c in model.calendars.values() if c.fiscal_year_start_month), None)
    if not start or start == 1:
        return ""
    named_by = next((c.fiscal_year_named_by for c in model.calendars.values()
                     if c.fiscal_year_start_month == start and c.fiscal_year_named_by), None) or "end"
    begins = today.year if today.month >= start else today.year - 1      # the year this fiscal year starts in

    def span(year: int) -> str:
        first = dt.date(year, start, 1)
        return f"FY{year + 1 if named_by == 'end' else year} = {first.isoformat()} to " \
               f"{(dt.date(year + 1, start, 1) - dt.timedelta(days=1)).isoformat()}"

    return (f"FISCAL YEARS (only for a period the question calls fiscal: \"FY2026\", \"fiscal year 2026\", "
            f"\"Q3 FY26\"; one it does not, as \"Q2 2026\", is the calendar's): a fiscal year starts on 1 "
            f"{dt.date(2000, start, 1):%B} and is named by the year it {'ends' if named_by == 'end' else 'starts'} "
            f"in. This fiscal year: {span(begins)}; last fiscal year: {span(begins - 1)}; the one before: "
            f"{span(begins - 2)}.")


def question_tail(question: str, *, today: dt.date, history: list[Turn], matches: list[ValueMatch],
                  masked: Masked | None = None, scrub: Callable[[str], str] | None = None,
                  model: SemanticModel | None = None, reading: str | None = None) -> str:
    """Today, the conversation, the member matches and the question (each question scrubbed when asked to).

    With member values withheld (``masked``), the previous turn is shown as the AI saw
    it: its question with placeholders, and its plan's member values as placeholders too.
    ``reading`` "refine": the words of the question already say it changes the answer above.
    """
    clean = scrub or (lambda text: text)
    lines = [f"TODAY: {today.isoformat()} ({today:%A})"]
    fiscal = fiscal_years_text(model, today) if model is not None else ""
    if fiscal:
        lines.append(fiscal)
    if history:
        last = history[-1]
        if masked is None or last.masked is not None:
            asked = last.masked.question if masked is not None and last.masked is not None else last.question
            lines.append(f"PREVIOUS QUESTION: {clean(asked)}")
        if last.plan is not None:
            lines.append("PREVIOUS PLAN: " + _plan_shown(last.plan, model, masked))
        if reading == "refine" and last.plan is not None:
            lines.append('READING: this question changes the answer above: return the previous plan with what '
                         'the question changes, and follow_up "refine". Keep its time, durations, derived measures '
                         'and filters unless the question changes them.')
    if matches:
        lines.append("VALUE MATCHES (member names found in the question; use these exact values):")
        for m in matches:
            shown = next((t for t, v in masked.values.items() if v == m.value), m.text) if masked else m.text
            stored = shown if masked else m.value
            lines.append(f'- "{shown}" -> {m.attribute} = "{stored}"')
    if model is not None:
        # Read from the question as the AI is shown it: with member values withheld, a name never shows through.
        from core2.plan.words import readings_text

        said = readings_text(model, clean(masked.question if masked else question))
        if said:
            lines.append(said)
    lines.append(f"QUESTION: {clean(masked.question if masked else question)}")
    return "\n".join(lines)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse(text: str) -> Plan:
    """The one JSON object in the model's answer, as a Plan (fences and stray prose tolerated)."""
    body = _FENCE.sub("", text.strip())
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("the answer holds no JSON object")
    data = json.loads(body[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("the answer is not a JSON object")
    _forgive(data)
    if data.get("kind") == "clarify" and "clarify" not in data and isinstance(data.get("question"), str):
        # The question back written beside "kind" rather than in "clarify": asked twice so, it was refused twice,
        # and the reader got "could not work out how to answer" instead of the question.
        about = data.get("about") if data.get("about") in ("measure", "date", "member", "path", "follow_up") else "other"
        options = [str(o) for o in data.get("options") or [] if isinstance(o, (str, int, float))]
        data = {"kind": "clarify", "clarify": {"about": about, "question": data["question"], "options": options}}
    return Plan.model_validate(data)


def _forgive(data: dict[str, Any]) -> None:
    """Slips of form that leave the meaning plain, put right where they would refuse the plan (a real model wrote
    them, and wrote them again when told): a duration counted ("agg": "count", where its rows are counted by the
    plan's measure), "measure": false written on the filter on a duration rather than on the duration, and "what
    share of the total is that" written as a share of members with no condition for them to meet (it is the
    share each member has of the total: intent share)."""
    durations = [d for d in data.get("durations") or [] if isinstance(d, dict)]
    for d in durations:
        if d.get("agg") not in (None, "avg", "min", "max", "sum"):
            d.pop("agg")
    per = data.get("per")
    if isinstance(per, dict) and per.get("summary") == "share" and per.get("op") is None and not per.get("values"):
        data.pop("per")
        if data.get("intent") in (None, "rank", "breakdown", "value"):
            data["intent"] = "share"
    by_name = {str(d.get("name", "")).casefold(): d for d in durations}
    for f in data.get("filters") or []:
        if isinstance(f, dict) and "measure" in f:
            said = f.pop("measure")
            named = by_name.get(str(f.get("field", "")).casefold())
            if named is not None and isinstance(said, bool) and "measure" not in named:
                named["measure"] = said


# A value the AI wrote for a member it was not told the name of: "first_one", "warehouse_with_max_increase",
# "first_supplier". Words joined by underscores, one of them pointing; never a name the reader wrote.
_STANDS_FOR = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")
_POINTING = {"first", "second", "third", "last", "top", "bottom", "max", "min", "maximum", "minimum", "highest",
             "lowest", "biggest", "largest", "smallest", "most", "least", "increase", "decrease", "previous",
             "selected", "one", "that", "this", "those", "these", "same", "specific", "given", "chosen", "above"}


# "<top_warehouse_from_previous_answer>", "{first_one}": a stand-in in brackets, whatever is inside.
_BRACKETED = re.compile(r"^\s*(?:<[^<>]+>|\{[^{}]+\})\s*$")


def stands_for(value: Any) -> bool:
    """Is this filter value a stand-in for a member the question points at, rather than a name?"""
    if not isinstance(value, str):
        return False
    return bool(_BRACKETED.match(value)) or bool(_STANDS_FOR.match(value)) and bool(set(value.split("_")) & _POINTING)


def _shape(plan: Plan, model: SemanticModel, question: str) -> list[str]:
    """What a plan asks that cannot be meant, whatever the model holds: each a sentence the repair round acts on."""
    problems: list[str] = []
    said = question.casefold()
    absent = [*plan.absent.without, *plan.absent.had] if plan.absent is not None else []
    for f in [*plan.filters, *absent]:
        if f.op not in ("is_null", "not_null") and not f.values:
            problems.append(f"The filter on {f.field} has no value. To answer for the members of the answer above, "
                            "keep the previous plan's group_by, sort and limit instead of filtering; filter only on "
                            "values in VALUE MATCHES or a previous plan, or leave the filter out.")
            continue
        for value in f.values:
            if stands_for(value) and str(value).casefold() not in said:
                problems.append(f'"{value}" is not a value of {f.field}: it stands for a member the question points '
                                "at. Filter only on values in VALUE MATCHES or a previous plan; when the question "
                                "does not say which member it means, return a clarify plan asking which one.")
    time = plan.time
    other = time.compare.window if time.compare is not None and time.compare.kind == "window" else None
    if other is not None and time.window.kind != "all" and (other == time.window or _overlap(time.window, other)):
        how = "is the same period as" if other == time.window else "overlaps"
        dates = ""
        if time.window.kind == "between" and time.window.start and time.window.end:
            first, last = time.window.start.isoformat(), time.window.end.isoformat()
            dates = (f' For two dates: time.window {{"kind": "between", "start": "{last}", "end": "{last}"}} and '
                     f'time.compare.window {{"kind": "between", "start": "{first}", "end": "{first}"}}.')
        problems.append(f"time.compare.window {how} time.window, so the change cannot be read. A change between two "
                        'dates or periods ("between 27 May and 17 August", "from Q1 to Q2") is time.window = the later '
                        f"one alone and time.compare.window = the earlier one alone.{dates}")
    dates = {r.slug for r in model.date_roles.values()}
    grouped = [g for g in plan.group_by if g in dates]
    if grouped:
        problems.append(f"{', '.join(grouped)} {'is a date' if len(grouped) == 1 else 'are dates'}: figures for "
                        "each day, snapshot, week or month are intent trend with time.date set to the date and "
                        "time.grain (day, week, month, quarter or year), never a group_by on a date.")
    return problems


def _overlap(a: Window, b: Window) -> bool:
    """Do two written-out periods share a day?"""
    if a.kind != "between" or b.kind != "between" or not (a.start and a.end and b.start and b.end):
        return False
    return a.start <= b.end and b.start <= a.end


def _snapshot(plan: Plan, model: SemanticModel) -> bool:
    """Is every measure of the plan a stock read at a snapshot (on hand, headcount)?"""
    found = [m for m in model.measures.values() if m.slug in plan.measures]
    return bool(found) and len(found) == len(set(plan.measures)) and all(m.additivity == "semi_additive" for m in found)


def fill_in(plan: Plan, model: SemanticModel, previous: Plan | None, said_follow_up: bool) -> Plan:
    """The plan with what it plainly means filled in, before it is checked:

    * a ranking sorted by a measure the plan does not show shows it ("rank the items by back-orders" sorted by
      a measure it left out, and was refused);
    * a stock compared between two dates written as one span ("on hand between 27 May and 17 August", as
      27 May-17 August against itself, or against 27 May) is the later date against the earlier one;
    * a plan that says nothing of the answer above, for the same measure right after it, follows it
      ("show the returns by warehouse" after the returns of Q1).
    """
    if plan.kind != "query":
        return plan
    update: dict[str, Any] = {}
    named = {*plan.measures, *(d.name for d in plan.derived), *(d.name for d in plan.durations)}
    slugs = {m.slug for m in model.measures.values()}
    extra = [s.by for s in plan.sort if s.by in slugs and s.by not in named and s.by not in plan.group_by]
    if extra:
        update["measures"] = [*plan.measures, *dict.fromkeys(extra)]
    time, other = plan.time, plan.time.compare.window if plan.time.compare is not None else None
    if other is not None and time.window.kind == "between" and other.kind == "between" and _snapshot(plan, model) \
            and time.window.start and time.window.end and other.start and other.end and other.end < time.window.end \
            and (other == time.window or _overlap(time.window, other)):
        last, first = time.window.end, (other.start if other != time.window else time.window.start)
        update["time"] = time.model_copy(update={
            "window": Window(kind="between", start=last, end=last, fiscal=time.window.fiscal),
            "compare": time.compare.model_copy(update={"window": Window(kind="between", start=first, end=first)})})
    elif other is not None and other == time.window and time.window.kind == "between" and _snapshot(plan, model) \
            and time.window.start and time.window.end and time.window.start < time.window.end:
        last, first = time.window.end, time.window.start
        update["time"] = time.model_copy(update={
            "window": Window(kind="between", start=last, end=last, fiscal=time.window.fiscal),
            "compare": time.compare.model_copy(update={"window": Window(kind="between", start=first, end=first)})})
    if not said_follow_up and previous is not None and previous.kind == "query" \
            and set(plan.measures) & set(previous.measures):
        update["follow_up"] = "refine"
    return plan.model_copy(update=update) if update else plan


def check(plan: Plan, model: SemanticModel, today: dt.date, question: str = "") -> tuple[list[str], str]:
    """What stops the plan from being answered, in words the planner can act on; and what it could use instead.

    The problems are plain sentences (a reader may see them); the hint is for
    the repair round only.
    """
    if plan.kind == "clarify":
        return ([] if plan.clarify and plan.clarify.question else ["a clarify plan needs clarify.question"]), ""
    if plan.kind != "query":
        return [], ""
    shape = _shape(plan, model, question)
    if shape:
        return shape, ""
    try:
        resolve(plan, model, Context(today=today))
    except ResolveError as exc:
        if exc.kind in ("unconfirmed", "denied", "ambiguous"):
            return [], ""          # not the planner's to fix: the service answers or asks the reader
        options = f" Allowed: {', '.join(exc.options[:12])}." if exc.options else ""
        return [f"{exc.message}.{options}"], _reaching(model, exc.unlinked, question) if exc.unlinked else ""
    except ValueError as exc:
        return [str(exc)], ""
    return [], ""


_COMMON = {"a", "an", "the", "of", "by", "in", "for", "per", "and", "or", "to", "on", "at", "is", "are", "was",
           "what", "which", "how", "each", "our", "my", "me", "show", "give", "total", "q1", "q2", "q3", "q4"}


def _words(text: str) -> set[str]:
    return {names.singular(w) for w in re.findall(r"[a-z0-9]+", text.casefold())
            if w not in _COMMON and not w.isdigit()}


def _reaching(model: SemanticModel, table: str, question: str) -> str:
    """The measures that can be broken down by what one could not, those sharing the question's words first.

    "Cost of goods sold by item group": the words found a ledger account of that
    name, and the ledger has no link to item groups -- but the cost on the
    invoice lines does. The repair round is shown the measures that reach the
    breakdown, so it can read the words as the measure they also name.
    """
    asked = _words(question)
    reaches: dict[str, bool] = {}
    found: list[tuple[int, str, Measure]] = []
    for m in model.measures.values():
        if m.hidden or not m.slug or m.table not in model.tables:
            continue
        if m.table not in reaches:
            reaches[m.table] = m.table == table or bool(P.all_paths(model, m.table, table))
        if reaches[m.table]:
            said = " ".join([m.business_name, *(w for words in m.synonyms.values() for w in words)])
            found.append((-len(asked & _words(said)), m.business_name.casefold(), m))
    if not found:
        return ""
    found.sort(key=lambda f: (f[0], f[1]))
    listed = ", ".join(f"{m.slug} ({m.business_name})" for _, _, m in found[:8])
    return (f"Measures that can be broken down that way: {listed}. If the question's words name one of them"
            " (a name in VALUE MATCHES can also be what a measure measures), use that measure instead.")


def _plan_shown(plan: Plan, model: SemanticModel | None, masked: Masked | None) -> str:
    """A plan as the AI may see it: member values as placeholders when values are withheld."""
    if masked is None:
        return plan.model_dump_json(exclude_defaults=True)
    data = plan.model_dump(mode="json", exclude_defaults=True)
    durations = {str(d.get("name", "")).casefold() for d in data.get("durations", [])}
    absent = data.get("absent") or {}
    for f in [*data.get("filters", []), *absent.get("without", []), *absent.get("had", [])]:
        if model is not None and (f.get("field") in model.measures or f.get("field") in model.date_roles
                                  or getattr(model.attributes.get(f.get("field")), "kind", "") == "number"):
            continue      # amounts, dates and numbers ("3 days in transit"), not member values
        if str(f.get("field", "")).casefold() in durations:
            continue      # numbers of days
        f["values"] = [placeholder(v, masked.values) for v in f.get("values", [])]
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _unmask_plan(plan: Plan, masked: Masked | None) -> Plan:
    if masked is None or not masked.values:
        return plan
    data = plan.model_dump()
    absent = data.get("absent") or {}
    for f in [*data.get("filters", []), *absent.get("without", []), *absent.get("had", [])]:
        f["values"] = [unmask(v, masked) for v in f.get("values", [])]
    return Plan.model_validate(data)


def plan_question(model: SemanticModel, question: str, complete: Complete, *, today: dt.date,
                  history: list[Turn] | None = None, matches: list[ValueMatch] | None = None,
                  values_allowed: bool = True, scrub: Callable[[str], str] | None = None,
                  hidden: list[ValueMatch] | None = None,
                  reading: str | None = None) -> Outcome:
    """The question's plan, checked against the model; at most two calls to the AI.

    ``reading``: "refine" or "new" when the question's words already say which (core2/plan/followup.py);
    the plan is marked so, whatever the AI said. None: the AI reads it, and may say "unsure".
    """
    matches = matches or []
    history = history or []
    known = history[-1].masked.values if history and history[-1].masked is not None else None
    # Every member name the question writes is withheld from the AI, quoted or not; only the quoted ones
    # (``matches``) are offered to it as names to filter on.
    masked = None if values_allowed else mask(question, [*matches, *(hidden or [])], known=known)
    # A workspace under compliance (its questions scrubbed) shows people's details masked, never refuses them.
    stable = stable_prompt(model, values_allowed=values_allowed, personal_shown=scrub is not None)
    tail = question_tail(question, today=today, history=history, matches=matches, masked=masked,
                         scrub=scrub, model=model, reading=reading)
    raw: list[str] = []
    problems: list[str] = []
    hint = ""
    # The hint ranks measures by the question's words: the question as the AI saw it, placeholders and all.
    shown = (scrub or (lambda text: text))(masked.question if masked else question)
    tried: Plan | None = None
    text = ""
    for attempt in range(2):
        prompt = tail if attempt == 0 else (
            f"{tail}\n\nYOUR PREVIOUS ANSWER:\n{text}\n\nPROBLEMS:\n" + "\n".join(f"- {p}" for p in problems)
            + (f"\n{hint}" if hint else "") + "\nReturn the corrected plan as one JSON object.")
        text = complete(stable, prompt)
        raw.append(text)
        try:
            plan = _unmask_plan(parse(text), masked)
        except (ValueError, ValidationError) as exc:
            problems, hint = [_short(exc)], ""
            continue
        plan = fill_in(plan, model, history[-1].plan if history and reading is None else None,
                        '"follow_up"' in text)
        problems, hint = check(plan, model, today, shown)
        if not problems:
            if reading in ("refine", "new") and plan.kind == "query":
                plan = plan.model_copy(update={"follow_up": reading})
            settled = settle(plan, model, question, history[-1].plan if history else None)
            if settled is not plan and not check(settled, model, today, shown)[0]:
                plan = settled
            return Outcome(plan, repaired=attempt > 0, raw=raw, masked=masked, tried=tried)
        tried = plan
    return Outcome(Plan(kind="clarify", clarify=Clarify(
        about="other", question="I could not work out how to answer that from this data. Could you say it "
        "another way, naming what to measure and for when?")), repaired=True, problems=problems, raw=raw,
        masked=masked, tried=tried)


# "The most", "the top 10": highest first. "The least", "the bottom 5": lowest first ("at least 5" is neither).
_MOST = re.compile(r"\b(?:most|highest|top|biggest|largest|greatest|maximum)\b", re.IGNORECASE)
_LEAST = re.compile(r"\b(?:(?<!at\s)least|lowest|bottom|fewest|smallest|minimum)\b", re.IGNORECASE)


def settle(plan: Plan, model: SemanticModel, question: str, previous: Plan | None) -> Plan:
    """The plan with what the question's own words settle, whatever the AI wrote. Found by asking the
    workbooks' questions of a real model:

    * a follow-up that names no period keeps the one of the answer above where the AI left it out ("show
      the returns by warehouse" after a Q1 question was answered for all time) or the question points at it
      ("due that month" was this month); a "why" keeps the period it is compared with; and the route a
      reader already picked for a field (which ABC class) is not asked again;
    * the latest snapshot of a stock is no window ("on the latest snapshot" was read as the last day,
      a partial load);
    * a comparison or a ranking of members the question asks no series of drops a grain kept from a trend
      before it ("which customer type cancels the most" after a monthly trend was ranked month by month)
      ("which channel grew the most" came back month by month: "Auto-Refill rose most; Auto-Refill fell
      most");
    * "the most" sorts highest first and "the least" lowest first ("lost the most" was sorted lowest first).
    """
    if plan.kind != "query":
        return plan
    update: dict[str, Any] = {}
    time = plan.time
    if previous is not None and previous.kind == "query" and plan.follow_up == "refine":
        if keeps_period(question):
            kept: dict[str, Any] = {}
            # A period the AI chose stands ("why did it grow?" after "which month grew the most" is that month),
            # unless the question points at the one above ("due that month").
            if time.window != previous.time.window and (time.window.kind == "all" or points_at_period(question)):
                kept["window"] = previous.time.window
            if time.date is None and previous.time.date:
                kept["date"] = previous.time.date
            if plan.intent in ("compare", "drivers") and previous.time.compare is not None \
                    and time.compare != previous.time.compare:
                kept["compare"] = previous.time.compare
            if kept:
                time = time.model_copy(update=kept)
        if previous.via and any(k not in plan.via for k in previous.via):
            update["via"] = {**previous.via, **plan.via}
    window = time.window
    if window.kind == "last" and window.unit == "day" and (window.n or 1) == 1 and plan.measures and all(
            m.slug not in plan.measures or m.additivity == "semi_additive" for m in model.measures.values()) \
            and any(m.slug in plan.measures for m in model.measures.values()):
        time = time.model_copy(update={"window": Window()})
    ranks_members = plan.intent == "rank" and bool(plan.group_by)      # a ranking of periods is its grain
    if (plan.intent == "compare" or ranks_members) and time.grain is not None and not asks_series(question) \
            and not re.search(rf"\b(?:{time.grain.replace('fiscal_', '')}|daily|annual)", question, re.IGNORECASE):
        time = time.model_copy(update={"grain": None})
    if time != plan.time:
        update["time"] = time
    if plan.sort and plan.sort[0].by not in ("change", "pct_change", "share", "period"):
        most, least = bool(_MOST.search(question)), bool(_LEAST.search(question))
        if most != least and plan.sort[0].desc != most:
            update["sort"] = [Sort(by=plan.sort[0].by, desc=most), *plan.sort[1:]]
    return plan.model_copy(update=update) if update else plan


def _short(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:6])
    return str(exc)[:300]
