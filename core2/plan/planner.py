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
from core2.model.schema import AggExpr, DateRole, Entity, Measure, SemanticModel
from core2.plan.catalog import catalog_text
from core2.plan.followup import (asks_series, keeps_period, measures_named, names_own_period, points_at_period,
                                 the_period)
from core2.plan.analyses import basket, due_and_done, funnel_dates, leading, ranked_entity, start_and_end, target_pair
from core2.plan.ir import Clarify, Compare, Derived, Filter, PerMember, Plan, Sort, Window, schema_text
from core2.plan.thread import kept_from_above, member_of
from core2.plan.values import Masked, ValueMatch, about_people, mask, placeholder, unmask
from core2.resolve import paths as P
from core2.resolve.resolver import Context, ResolveError, resolve

Complete = Callable[[str, str], str]       # (stable system prompt, question tail) -> the model's text

RULES = """You turn a business question into a query plan for Solviz. You never write SQL.
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
   nothing in the catalog answers it; say why in notes. A question that names no figure at all ("how are we
   doing?", "show me the numbers") is no refusal: clarify, about "other", with 2-4 of the main measures.
2. measures: the measure slugs the question names. Never swap in a different measure because it is close
   (sales is not margin, quantity is not value, count of rows is not count of orders). When the question
   needs a ratio, difference, sum or product of two measures (a margin %, revenue per order), put it in
   derived with a short name, the op and the two slugs (scale 100 for a percentage); a part may be an
   earlier derived's name: a margin % is the ratio of a derived difference (sales less cost) to the sales,
   scale 100, never sales over cost. A figure times a number (ARR as 12 times MRR) is a product of the slug
   with itself and the number as scale; a change or "% increase" from one period to another is never a derived
   of a measure with itself: it is time.compare, sorted by change or pct_change. A rate of a condition ("the
   cancellation rate by region", "% of orders returned", "share of lines late", "reversal rate") goes in rates:
   a short name, the measure counted (number_of_orders) and where, the condition on that measure's own rows
   (status eq ["C"]); each group's rate is of its own whole, and the condition is no filter. A rate of one
   quantity over another of the same rows ("return rate" as units returned over units sold) has the whole as
   its measure (units_sold) and the part as its condition (units_returned gt [0]); but a share OF ROWS ("what
   share of orders had any return", "% of lines with a return") is their count (number_of_orders) with that
   condition. Days between two
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
   put it in via: {"<the group_by slug>": "<role>"}; with no role named, leave via empty. When it lists
   routes ("through (name one in via): customer, supplier") and the question says whose ("French
   customers", "by supplier nation"), put that entity's slug in via for the field it reaches.
4. time.grain makes a series: "monthly", "by month", "weekly", "per quarter", "trend", "over time" (month
   when unsaid). No grain for one total.
5. time.window, written as the user said it; never compute relative dates yourself:
   "last 6 months" -> {"kind": "last", "unit": "month", "n": 6} (complete months before the current one);
   "this year", "year to date", "so far this year" -> {"kind": "this", "unit": "year"}; "so far", "to date" or
   "ever" with no period named is everything up to now: no window;
   "last month" -> {"kind": "previous", "unit": "month"}; "last year" -> {"kind": "previous", "unit": "year"};
   "in March 2026" -> {"kind": "between", "start": "2026-03-01", "end": "2026-03-31"};
   "since January" -> {"kind": "since", "start": "<this year>-01-01"};
   fiscal periods set "fiscal": true: "this fiscal year" -> {"kind": "this", "unit": "year", "fiscal": true};
   "last fiscal year" -> {"kind": "previous", "unit": "year", "fiscal": true}; a fiscal year, half or quarter
   named by its number ("FY2026", "fiscal year 2026", "H1 FY26", "Q3 FY26") -> kind "between" with the dates
   FISCAL YEARS gives for it. A quarter, half or year the question does not call fiscal ("Q2 2026", "in 2025")
   is the calendar's: "Q2 2026" -> 2026-04-01 to 2026-06-30.
   In French, a quarter is a "trimestre" written T1 to T4 ("T1 2025" is Q1 2025), a half a "semestre" written
   S1 or S2, and an "exercice" a fiscal year ("l'exercice 2026" is FY2026).
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
11. Follow-ups: PREVIOUS PLAN is the plan behind the last answer. Read the question as a person reading the
   conversation would, and decide; never ask which was meant.
   It is a follow-up (follow_up "refine") when it points at the answer ("it", "they", "those", "the first one"),
   changes it ("by week instead", "only for X", "top 5 of those", "what about returns"), or goes on with the same
   subject without a scope of its own: another figure of the same records or a new split of them, with no period
   or member named ("how many orders?" after net sales for North in March is the orders of North in March).
   Return the whole new plan with follow_up "refine", and keep the previous plan's filters and time (window,
   date) unless the question names its own.
   It is new (follow_up "new") when it is about something else: other records ("which managers have the most
   direct reports?" after payroll by month), a list of another kind of thing, a question about the data itself;
   or it states its whole scope itself ("net sales by region in 2025", "across all stores", "overall"). A new
   question keeps nothing of the previous plan. When it could be either, a question about the same records that
   names no period and no member of its own is a follow-up.
   "Those", "them", "the same customers" are the members of the answer on screen, whose names you are not given:
   never filter on a name that is not in VALUE MATCHES or a previous plan, and never a filter with no values: to
   answer for them, keep the previous plan's group_by, sort and limit. "The first one", "the lowest one", "that
   division" are in VALUE MATCHES when the member they point at is known: filter on its value. Never put such
   words themselves ("lowest", "that division") in a filter; when they are not matched, ask (clarify). To follow
   them into another period ("how did those same customers do in 2025?"), keep the previous plan's group_by, sort,
   limit and time.window, and put the other period in time.compare, with intent "compare". When READING says how
   the question was read, follow it. A question in French follows these rules the same way.
12. notes: one short line for each assumption the user did not state.
13. chart: only when the question says how to show it ("as a pie", "as a donut" or "ring" -> "donut",
   "bar chart", "stacked bars" -> "stacked", "as a heatmap" or "grid" -> "heatmap", "as a scatter" or
   "plotted against each other" -> "scatter", "on a map" -> "map", "just the table" -> "table");
   on a follow-up ("show that as a pie") return the previous plan with chart set and follow_up "refine".
14. include_left_out: true only when the question asks for rows a table LEAVES OUT by default (listed in the
   catalog), as "including cancelled orders" or "with test orders too", or asks about those rows themselves
   ("revenue on cancelled lines", "which customer type cancels the most"); otherwise leave it out.
15. A measure the catalog says "is X less Y" (net = debit less credit, revenue = gross less discount) is the
   total of an account, a category or any member asked about ("operating expense by profit centre", "sales of
   valves"): use it, never X or Y alone, unless the question names X or Y ("debits", "gross amount").
"""


@dataclass
class Turn:
    question: str
    plan: Plan | None = None
    masked: Masked | None = None        # what the AI was shown, when member values are withheld
    shown: list[tuple] = field(default_factory=list)   # the answer's members in order: (attribute, value, number)
    periods: list[str] = field(default_factory=list)   # the answer's periods in order, by their first day (ISO)
    title: str = ""                     # what the answer shows ("Net amount by region"): "this" in "add this ..."
    # Why the turn got no answer, in a few fixed words ("it was refused", "no answer came back"); "" for an answered
    # turn. A turn that went unanswered stays in the conversation by its words alone, with no plan: "and 2025?"
    # right after it still means its question, and nothing it tried is carried.
    unanswered: str = ""


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
    if entities and count is not None:
        e = entities[0]
        lines.append(
            "- gap: time between one member's events, in date order; no measures. of: the entity; measure: whose "
            "rows are the events; summary avg (default) | median | min | max of the days from each event back to "
            "the one before; nth: days from the first event to the nth instead (\"time to second purchase\" -> 2; "
            "time.window then holds the first events); within_days: the share of events followed by the next "
            "within n days instead (\"reordered within 30 days\", \"readmitted within 30 days\"). time.grain for a "
            "series; group_by only the member's own fields.")
        plan = {"kind": "query", "intent": "value", "gap": {"of": e.slug, "measure": count.slug},
                "time": {"window": {"kind": "this", "unit": "year"}}}
        lines.append(f"Q: average days between {things} per {e.business_name.lower()} this year\n"
                     f"A: {json.dumps(plan)}")
        lines.append(
            "- lifecycle: members by their activity in each period (time.grain; a month by default); no measures. "
            "status all (active, new, returning, won back and lapsed side by side) | active | new | returning | "
            "won_back | lapsed | churn_rate (\"churn\", \"attrition of customers\", \"retention rate\" is 100 minus "
            "it: ask churn_rate). \"New vs returning\" -> all.")
        plan = {"kind": "query", "intent": "trend", "lifecycle": {"of": e.slug, "measure": count.slug,
                                                                  "status": "churn_rate"},
                "time": {"grain": "month", "window": {"kind": "last", "unit": "month", "n": 12}}}
        lines.append(f"Q: monthly {e.business_name.lower()} churn rate over the last year\nA: {json.dumps(plan)}")
        lines.append(
            "- cohort: members grouped by the period of their first activity (time.grain; a month by default), and "
            "the share (value share) or number (value count) of each still active n periods later; no measures, no "
            "group_by. time.window holds the first periods; periods: how many after the first (12 by default).")
        plan = {"kind": "query", "intent": "breakdown", "cohort": {"of": e.slug, "measure": count.slug, "periods": 6},
                "time": {"grain": "month", "window": {"kind": "last", "unit": "month", "n": 12}}}
        lines.append(f"Q: retention by month of first {names.singular(things)} for the last year\n"
                     f"A: {json.dumps(plan)}")
    pair = start_and_end(model)
    if pair is not None:
        start, end = pair
        rows = names.plural(model.tables[start.table].business_name.lower())
        lines.append(
            "- active: rows in force on each date, from each row's start and end dates (an empty or far-off end is "
            "none); no measures. status active (in force at each period's end: \"headcount\", \"active "
            "subscriptions\", \"open at month end\") | starts | ends | all (in force, starts and ends) | end_rate (\"attrition "
            "rate\", \"turnover\"). time.grain: the periods (a month by default).")
        plan = {"kind": "query", "intent": "trend", "active": {"start": start.slug, "end": end.slug},
                "time": {"grain": "month", "window": {"kind": "this", "unit": "year"}}}
        lines.append(f"Q: how many {rows} were there at each month end this year\nA: {json.dumps(plan)}")
    pair = due_and_done(model)
    if pair is not None:
        due, done = pair
        rows = names.plural(model.tables[done.table].business_name.lower())
        lines.append(
            "- on_time: rows against the date they were due by; no measures. done: when it happened; due: when it "
            "was expected; grace_days: days late still counted on time; late true for the share late (\"later than "
            "requested\", \"delayed\"). ageing true for the rows not done yet, by "
            "days past due today (\"AR aging\", \"overdue\", \"open past due\"), with a measure to total (amount "
            "open) when asked; time.window then only says the day to age on (its end).")
        plan = {"kind": "query", "intent": "value", "on_time": {"done": done.slug, "due": due.slug},
                "time": {"window": {"kind": "this", "unit": "year"}}}
        lines.append(f"Q: what share of {rows} were on time this year\nA: {json.dumps(plan)}")
    lead = leading(model)
    who = ranked_entity(model, lead) if lead is not None else None
    if lead is not None and who is not None:
        many = names.plural(who.business_name.lower())
        lines.append(
            "- pareto: who makes most of a figure, each member's figure ranked from the largest; no measures, no "
            "group_by. of: the entity; measure: the figure. mode abc (\"ABC analysis\", \"A/B/C classes\"; cuts "
            "[80, 95] by default) | pareto (\"which customers make 80% of revenue\", \"the top 20%\": cuts [80]) "
            "| ntile (\"deciles\": groups 10, \"quartiles\": 4). members true to list each member with its class "
            "or decile.")
        plan = {"kind": "query", "intent": "breakdown", "pareto": {"of": who.slug, "measure": lead.slug},
                "time": {"window": {"kind": "this", "unit": "year"}}}
        lines.append(f"Q: ABC analysis of {many} by {lead.business_name.lower()} this year\nA: {json.dumps(plan)}")
        plan = {"kind": "query", "intent": "rank", "pareto": {"of": who.slug, "measure": lead.slug, "mode": "pareto",
                                                              "cuts": [80]}}
        lines.append(f"Q: which {many} make 80% of {lead.business_name.lower()}\nA: {json.dumps(plan)}")
    stages = funnel_dates(model)
    if len(stages) >= 2:
        rows = names.plural(model.tables[stages[0].table].business_name.lower())
        lines.append(
            "- funnel: rows through dates of their own, in order; no measures, no group_by. stages: the date slugs "
            "in the order they happen (\"order to ship to delivery\"); time.window holds the first stage.")
        plan = {"kind": "query", "intent": "breakdown", "funnel": {"stages": [r.slug for r in stages]},
                "time": {"window": {"kind": "this", "unit": "year"}}}
        lines.append(f"Q: funnel of {rows} from {stages[0].name.lower()} to {stages[-1].name.lower()} this year\n"
                     f"A: {json.dumps(plan)}")
    found = basket(model)
    if found is not None:
        item, count = found
        many = names.plural(item.business_name.lower())
        lines.append(
            "- together: items on the same basket, in pairs; no measures, no group_by. of: the items' entity; "
            "basket: the count measure of the baskets (\"number of orders\"). A quoted item, as a filter on the "
            "items' name, keeps the pairs that hold it (\"what is bought with \\\"X\\\"\").")
        plan = {"kind": "query", "intent": "rank", "together": {"of": item.slug, "basket": count.slug}, "limit": 10}
        lines.append(f"Q: which {many} are bought together\nA: {json.dumps(plan)}")
    pair_t = target_pair(model)
    if pair_t is not None:
        actual, target = pair_t
        lines.append(
            "- target: a figure against its target, budget or quota (\"vs budget\", \"against quota\", \"did we hit "
            "target\"): actual and target measures, no measures list; group_by and time as asked. The answer adds "
            "the difference and the % of target.")
        plan = {"kind": "query", "intent": "trend", "target": {"actual": actual.slug, "target": target.slug},
                "time": {"grain": "month", "window": {"kind": "this", "unit": "year"}}}
        lines.append(f"Q: {actual.business_name.lower()} against {target.business_name.lower()} by month this year\n"
                     f"A: {json.dumps(plan)}")
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


def _fiscal(model: SemanticModel) -> tuple[int, str]:
    """The month the workspace's fiscal year starts in (1: the calendar year), and whether a fiscal year is named
    by the year it starts or ends in."""
    start = model.settings.fiscal_year_start_month or next(
        (c.fiscal_year_start_month for c in model.calendars.values() if c.fiscal_year_start_month), None) or 1
    named_by = next((c.fiscal_year_named_by for c in model.calendars.values()
                     if c.fiscal_year_start_month == start and c.fiscal_year_named_by), None) or "end"
    return start, named_by


def fiscal_years_text(model: SemanticModel, today: dt.date) -> str:
    """The workspace's fiscal years as dates, for a question that names one by its number.

    "Fiscal year 2026" was planned as this fiscal year: told only that fiscal periods are
    fiscal, the AI could not know that FY2026 ended months ago (a year starting in April,
    named by the year it ends). Nothing is said where the fiscal year is the calendar year.
    """
    start, named_by = _fiscal(model)
    if start == 1:
        return ""
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
        if last.unanswered:
            # The question before was never planned (the AI was not reached): the answer before it is what was on
            # screen, and the unanswered question says what the reader went on to ask.
            shown = next((t for t in reversed(history[:-1]) if t.plan is not None and not t.unanswered), None)
            if shown is not None and shown.plan is not None and (masked is None or shown.masked is not None):
                earlier = shown.masked.question if masked is not None and shown.masked is not None else shown.question
                lines.append(f"EARLIER QUESTION: {clean(earlier)}")
                lines.append(_plan_line("EARLIER PLAN", shown.plan, model, masked))
        if masked is None or last.masked is not None:
            asked = last.masked.question if masked is not None and last.masked is not None else last.question
            said = f" (not answered: {last.unanswered})" if last.unanswered else ""
            lines.append(f"PREVIOUS QUESTION{said}: {clean(asked)}")
        if last.plan is not None:
            lines.append(_plan_line("PREVIOUS PLAN (not answered)" if last.unanswered else "PREVIOUS PLAN",
                                    last.plan, model, masked))
        if reading == "refine" and last.plan is not None:
            lines.append('READING: this question changes the answer above: return the previous plan with what '
                         'the question changes, and follow_up "refine". Keep its time, durations, derived measures '
                         'and filters unless the question changes them.')
    if matches:
        # The heading says which: the AI egress log reads the stored values' one as values sent (core/llm_audit.py).
        lines.append("VALUE MATCHES (member names found in the question, as placeholders; use these exact "
                     "placeholders):" if masked else
                     "VALUE MATCHES (member names found in the question; use these exact values):")
        for m in matches:
            shown = placeholder(m.value, masked.values) if masked else m.text     # never the name, masked
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
    them, and wrote them again when told): a period written as null ("window": null, not set); a duration counted
    ("agg": "count", where its rows are counted by the
    plan's measure), "measure": false written on the filter on a duration rather than on the duration, and "what
    share of the total is that" written as a share of members with no condition for them to meet (it is the
    share each member has of the total: intent share)."""
    time = data.get("time")
    if isinstance(time, dict):
        # "window": null, "grain": null: not set (a real model wrote them so, and the plan was refused).
        for key in [k for k, v in time.items() if v is None]:
            time.pop(key)
    for key in [k for k in ("per", "rolling", "within", "bands", "absent", "gap", "lifecycle", "cohort", "active",
                            "on_time", "pareto", "funnel", "together", "target", "drivers", "chart")
                if data.get(k) is None and k in data]:
        data.pop(key)
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
# A row's place among the rows: "top", "first", "the lowest one".
_ROW_PLACES = _POINTING - {"previous", "above", "selected", "given", "chosen", "specific", "same", "that", "this",
                           "those", "these", "increase", "decrease"}
# "top 1 from previous answer", "the first one in the list above": a description of a row, in words.
_DESCRIBED = re.compile(r"\b(?:previous|prior|last|above)\s+(?:answer|result|list|table|ranking|question)\b|"
                        r"\b(?:from|in|on)\s+the\s+(?:answer|list|table|ranking)\b|\bon\s+screen\b", re.IGNORECASE)


def stands_for(value: Any) -> bool:
    """Is this filter value a stand-in for a member the question points at, rather than a name?"""
    if not isinstance(value, str):
        return False
    return bool(_BRACKETED.match(value)) or bool(_STANDS_FOR.match(value)) and bool(set(value.split("_")) & _POINTING) \
        or bool(_DESCRIBED.search(value)) and bool(set(value.lower().split()) & _ROW_PLACES)


# "The first supplier on that list", "the 2nd one": a place in the answer on screen.
_PLACE_WORD = re.compile(r"\b(first|1st|top|second|2nd|third|3rd|fourth|4th|fifth|5th|last|bottom)\b", re.IGNORECASE)
_PLACES = {"first": 0, "1st": 0, "top": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2, "fourth": 3, "4th": 3,
           "fifth": 4, "5th": 4, "last": -1, "bottom": -1}


# "From the first to the second half of fiscal year 2026", "H1 against H2 2025": a year's two halves.
_HALVES = re.compile(r"\b(?:first|1st)\s+half\b.*\b(?:second|2nd)\s+half\b|\b(?:second|2nd)\s+half\b.*\b(?:first|1st)\s+"
                     r"half\b|\b(?:first|1st)\s+(?:to|and|vs\.?|versus|against)\s+(?:the\s+)?(?:second|2nd)\s+half\b|"
                     r"\bh1\b.*\bh2\b|\bh2\b.*\bh1\b", re.IGNORECASE)
_FISCAL_YEAR = re.compile(r"\b(?:fiscal\s+(?:year\s+)?|fy\s?)'?(\d{4}|\d{2})\b", re.IGNORECASE)
_YEAR = re.compile(r"\b((?:19|20)\d{2})\b")


def _halves(question: str, model: SemanticModel) -> tuple[Window, Window] | None:
    """A year's first and second halves, as the question names them ("the first to the second half of fiscal year
    2026"): the second, and the first it is compared with."""
    if not _HALVES.search(question):
        return None
    fiscal = _FISCAL_YEAR.search(question)
    calendar = _YEAR.search(question)
    if fiscal:
        number = int(fiscal.group(1))
        number += 2000 if number < 100 else 0
        start_month, named_by = _fiscal(model)
        begins = number - 1 if named_by == "end" and start_month != 1 else number
        first = dt.date(begins, start_month, 1)
    elif calendar:
        first = dt.date(int(calendar.group(1)), 1, 1)
    else:
        return None
    middle = dt.date(first.year + (first.month + 5) // 12, (first.month + 5) % 12 + 1, 1)
    end = dt.date(first.year + 1, first.month, 1)
    day = dt.timedelta(days=1)
    return (Window(kind="between", start=middle, end=end - day, fiscal=bool(fiscal)),
            Window(kind="between", start=first, end=middle - day, fiscal=bool(fiscal)))


_GRAIN_MONTHS = {"month": 1, "quarter": 3, "year": 12, "fiscal_month": 1, "fiscal_quarter": 3, "fiscal_year": 12}


def _months_on(day: dt.date, months: int) -> dt.date:
    at = day.month - 1 + months
    return dt.date(day.year + at // 12, at % 12 + 1, 1) if day.day == 1 else day.replace(
        year=day.year + at // 12, month=at % 12 + 1)


def why_periods(plan: Plan, model: SemanticModel, question: str, history: list[Turn]) -> Plan:
    """The two periods a "why" is about, from the question's words or the answers on screen, never the AI's guess.

    Found asking a real model (gpt-4o):

    * "Why did it change from the first to the second half of fiscal year 2026?" kept the months of the trend on
      screen, and was answered as February against March: the two halves are worked out here;
    * "Why did it grow?" after "Which month grew the most?" showed May against May a year before, and was answered
      as May against June: the one period on screen is the one explained, against the period it was compared with;
    * "Pourquoi cette différence ?" after the same figure on 17 August, then on 27 May, was answered as 27 May
      against the day before: the two dates on screen are the two compared.
    """
    if plan.kind != "query" or plan.intent not in ("drivers", "compare"):
        return plan
    time = plan.time
    halves = _halves(question, model)
    if halves is not None:
        window, prior = halves
        return plan.model_copy(update={"time": time.model_copy(update={
            "grain": None, "window": window, "compare": Compare(kind="window", window=prior)})})
    if plan.intent != "drivers" or not history or not keeps_period(question) or history[-1].plan is None or history[-1].plan.kind != "query":
        return plan
    last = history[-1]
    before = last.plan
    assert before is not None
    months = _GRAIN_MONTHS.get(before.time.grain or "")
    if before.time.compare is not None and months and len(last.periods) == 1:
        try:
            start = dt.date.fromisoformat(last.periods[0])
        except ValueError:
            return plan
        window = Window(kind="between", start=start, end=_months_on(start, months) - dt.timedelta(days=1),
                        fiscal=before.time.window.fiscal)
        compare = before.time.compare
        if compare.kind == "window" and compare.window is not None and compare.window.start \
                and before.time.window.start:
            # The comparison on screen was a stretch against another as long: the month's own match in it.
            shift = (compare.window.start.year - before.time.window.start.year) * 12 + \
                (compare.window.start.month - before.time.window.start.month)
            compare = Compare(kind="window", window=Window(
                kind="between", start=_months_on(start, shift),
                end=_months_on(_months_on(start, shift), months) - dt.timedelta(days=1), fiscal=window.fiscal))
        return plan.model_copy(update={"time": time.model_copy(update={
            "grain": None, "window": window, "compare": compare, "date": time.date or before.time.date})})
    earlier = history[-2].plan if len(history) >= 2 else None
    if earlier is not None and earlier.kind == "query" and time.compare is None \
            and set(earlier.measures) == set(before.measures) == set(plan.measures) \
            and before.time.compare is None and earlier.time.compare is None:
        a, b = earlier.time.window, before.time.window
        if a.kind == b.kind == "between" and a.start and b.start and (a.start, a.end) != (b.start, b.end):
            later, sooner = (a, b) if a.start > b.start else (b, a)
            return plan.model_copy(update={"time": time.model_copy(update={
                "grain": None, "window": later, "compare": Compare(kind="window", window=sooner),
                "date": time.date or before.time.date})})
    return plan


_SNAPSHOT = re.compile(r"\b(?:snapshots?|instantan[ée]s?|photos?)\b", re.IGNORECASE)


# "Where do we have back-ordered quantity?": the members that have some.
_WHERE_WE_HAVE = re.compile(r"\bwhere\s+(?:do|does|did)\s+(?:we|you|they|it)\s+(?:have|hold|keep|carry|stock)\b|"
                            r"\bo[uù]\s+(?:avons|a-t-on)[\s-]+(?:nous\s+)?\b", re.IGNORECASE)


def where_we_have(plan: Plan, model: SemanticModel, question: str) -> Plan:
    """ "Where do we have back-ordered quantity on the 17 August snapshot?" listed every warehouse, most with none:
    where there is some is where it is above 0."""
    if plan.kind != "query" or not plan.group_by or len(plan.measures) != 1 or not _WHERE_WE_HAVE.search(question or ""):
        return plan
    measure = plan.measures[0]
    if any(f.field == measure for f in plan.filters):
        return plan
    return plan.model_copy(update={"filters": [*plan.filters, Filter(field=measure, op="gt", values=[0])]})


def counted_each(plan: Plan, model: SemanticModel, question: str) -> Plan:
    """ "How many items in the top group cost more than $1,000 each?" was planned as the number of items, kept
    only if the group's own average cost was above $1,000: 0. A count of a list's members with a condition on
    another figure is the members whose own figure meets it, counted."""
    if plan.kind != "query" or plan.group_by or plan.per is not None or plan.shaped():
        return plan
    by_slug = {m.slug: m for m in model.measures.values()}
    counts = [by_slug[s] for s in plan.measures if s in by_slug and getattr(by_slug[s].expr, "agg", None)
              in ("count", "count_distinct")]
    conditions = [f for f in plan.filters if f.field in by_slug and f.op in ("gt", "gte", "lt", "lte", "eq", "between")
                  and by_slug[f.field] not in counts]
    if len(counts) != 1 or len(conditions) != 1:
        return plan
    entity = next((e for e in model.entities.values() if e.table == counts[0].table), None)
    if entity is None:
        return plan
    f = conditions[0]
    try:
        per = PerMember(of=entity.slug, summary="count", op=f.op, values=[float(v) for v in f.values])
    except (TypeError, ValueError, ValidationError):
        return plan
    return plan.model_copy(update={"measures": [f.field], "filters": [x for x in plan.filters if x is not f],
                                   "per": per, "intent": "count", "sort": [], "limit": None})


# "Which techs had fewer than 3 visits?", "who placed more than 5 orders?", "list the stores with no ...": the
# members themselves are asked for.
_WHICH_MEMBERS = re.compile(r"^\s*(?:which|who|whom|list|name)\b", re.IGNORECASE)


def members_asked(plan: Plan, model: SemanticModel, question: str) -> Plan:
    """ "Which techs had fewer than 3 visits in December 2025?" was answered "1 rows": the members whose own figure
    meets a condition were counted, where the question asks which they are. Asked with "which" or "who", they are
    listed by name, each with its figure; "how many" still counts them."""
    from core2.resolve.resolver import find_slug

    per = plan.per
    if plan.kind != "query" or per is None or per.summary != "count" or per.op is None or not per.of \
            or per.everyone or plan.group_by or len(plan.measures) != 1 or not _WHICH_MEMBERS.match(question or "") \
            or re.search(r"\bhow\s+many\b", question, re.IGNORECASE):
        return plan
    found = find_slug(model, per.of)
    if found is None or found[0] not in ("entity", "attribute"):
        return plan
    measure = plan.measures[0]
    return plan.model_copy(update={
        "per": None, "group_by": [per.of], "intent": "list", "limit": None,
        "filters": [*plan.filters, Filter(field=measure, op=per.op, values=list(per.values))],
        "sort": [Sort(by=measure, desc=per.op in ("gt", "gte"))]})


def counted_per_itself(plan: Plan, model: SemanticModel, question: str) -> Plan:
    """ "How many patients had more than 10 fills in 2025?" was planned as each patient's number of patients (one,
    always) above 10: none. A member's own count is never its figure per member; the count the question names
    ("fills") is."""
    per = plan.per
    if plan.kind != "query" or per is None or not per.of or len(plan.measures) != 1:
        return plan
    from core2.resolve.resolver import find_slug

    by_slug = {m.slug: m for m in model.measures.values()}
    figure = by_slug.get(plan.measures[0])
    found = find_slug(model, per.of)
    table = found[1].table if found is not None and isinstance(found[1], Entity) else None
    if figure is None or table is None or figure.table != table \
            or getattr(figure.expr, "agg", None) not in ("count", "count_distinct"):
        return plan
    named = [s for s in measures_named(model, question) if s in by_slug and by_slug[s].table != table
             and getattr(by_slug[s].expr, "agg", None) in ("count", "count_distinct")]
    if len(named) != 1:
        return plan
    return plan.model_copy(update={"measures": named, "sort": [
        s.model_copy(update={"by": named[0]}) if s.by == figure.slug else s for s in plan.sort]})


# "How many movement documents were voided?", "issue docs", "combien de documents": documents, not their lines.
_DOCUMENTS = re.compile(r"\b(?:documents?|docs?|vouchers?|pièces?)\b", re.IGNORECASE)


def documents_counted(plan: Plan, model: SemanticModel, question: str) -> Plan:
    """ "How many movement documents were voided in 2025?" counted the movement lines (one document has several):
    a question that counts documents counts the distinct document numbers of the same rows, where the data keeps
    that count beside the count of rows."""
    if plan.kind != "query" or not _DOCUMENTS.search(question or ""):
        return plan
    by_slug = {m.slug: m for m in model.measures.values()}
    swap: dict[str, str] = {}
    for slug in plan.measures:
        rows = by_slug.get(slug)
        if rows is None or getattr(rows.expr, "agg", None) != "count" or getattr(rows.expr, "column", None):
            continue
        docs = [m for m in by_slug.values() if m.table == rows.table and not m.hidden
                and getattr(m.expr, "agg", None) == "count_distinct" and getattr(m.expr, "column", None)
                and not getattr(m.expr, "filters", None)
                and any(a.column == m.expr.column and a.kind == "identifier" for a in model.attributes.values())]
        if len(docs) == 1:
            swap[slug] = docs[0].slug
    if not swap:
        return plan
    return plan.model_copy(update={
        "measures": [swap.get(m, m) for m in plan.measures],
        "sort": [s.model_copy(update={"by": swap[s.by]}) if s.by in swap else s for s in plan.sort],
        "filters": [f.model_copy(update={"field": swap[f.field]}) if f.field in swap else f for f in plan.filters]})


_HOW_MANY_ROLE = re.compile(r"^\s*(?:how\s+many|combien\s+(?:de|d'))\s*([a-zà-ÿ'-]+(?:\s+[a-zà-ÿ'-]+)?)",
                            re.IGNORECASE)


def role_counted(plan: Plan, model: SemanticModel, question: str) -> Plan:
    """ "How many managers do we have?" was planned as the employees counted under each manager, and answered with
    the manager who has most ("Quinn Lopez leads with 19 across 40"). The things counted are the members of the
    role the grouping follows: those with at least one row, counted."""
    hit = _HOW_MANY_ROLE.match(question or "")
    if plan.kind != "query" or hit is None or len(plan.group_by) != 1 or plan.shaped() or len(plan.measures) != 1 \
            or plan.derived or plan.time.grain or plan.limit:
        return plan
    slug = plan.group_by[0]
    role = str(plan.via.get(slug) or "").removeprefix("role:").strip()
    said = [names.singular(w) for w in re.findall(r"[a-zà-ÿ]+", hit.group(1).casefold())]
    wanted = [names.singular(w) for w in re.findall(r"[a-zà-ÿ]+", role.casefold())]
    if not role or said[:len(wanted)] != wanted:
        return plan
    return plan.model_copy(update={"group_by": [], "intent": "count", "sort": [], "limit": None,
                                   "per": PerMember(of=slug, summary="count", op="gte", values=[1.0])})


# "How many prescriptions did we fill", "were invoiced", "we shipped": the verb of what was done to them.
_DONE = re.compile(r"\b(?:did|do|does|have|has|had)\s+(?:we|you|they|i)\s+(\w+)|\b(?:were|was|been|got|get|being)\s+"
                   r"(\w+)|\bwe\s+(\w+ed)\b", re.IGNORECASE)


def _done_words(question: str) -> set[str]:
    """The verbs the question says were done to what it counts, as written ("fill", "filled", "invoiced")."""
    return {w.lower() for hit in _DONE.finditer(question) for w in hit.groups() if w}


def _row_word(model: SemanticModel, table: str) -> str:
    """One row of ``table`` in a word: "fill" for Fills, "invoice" for Invoice lines' "Invoices"."""
    words = re.findall(r"[a-z]+", model.tables[table].business_name.casefold()) if table in model.tables else []
    return names.singular(words[-1]) if words else ""


def counted_where_done(plan: Plan, model: SemanticModel, question: str) -> Plan:
    """ "How many prescriptions did we fill in 2025?" was counted by the prescriptions' own written date: 4,258
    written, where the fills of 2025 were of 4,211. The verb is the name of the rows that point at those counted (a
    fill is of a prescription, an invoice of a customer): those counted are the ones with such a row in the
    period, by that row's own date ("how many customers did we invoice in Q2")."""
    if plan.kind != "query" or plan.group_by or plan.shaped() or len(plan.measures) != 1 or plan.derived \
            or plan.time.window.kind == "all" or plan.time.compare is not None or not question:
        return plan
    by_slug = {m.slug: m for m in model.measures.values()}
    counted = by_slug.get(plan.measures[0])
    done = _done_words(question)
    if counted is None or getattr(counted.expr, "agg", None) not in ("count", "count_distinct") or not done:
        return plan
    for m in sorted(by_slug.values(), key=lambda m: m.slug):
        if m.table == counted.table or m.hidden or getattr(m.expr, "agg", None) != "count":
            continue
        word = _row_word(model, m.table)
        if len(word) < 3 or not done & {word, f"{word}ed", f"{word}d", f"{word}ing", f"{word}s"}:
            continue
        try:
            reached = P.best_path(model, m.table, counted.table) is not None   # a fill's prescription's patient
        except P.Ambiguous:
            reached = False
        if not reached:
            continue
        entity = next((e for e in model.entities.values() if e.table == counted.table and not e.role), None)
        role = next((r for r in model.date_roles.values() if r.slug == plan.time.date), None)
        date = plan.time.date if role is not None and role.table == m.table else None
        return plan.model_copy(update={
            "measures": [m.slug], "intent": "count", "sort": [], "limit": None,
            "per": PerMember(of=entity.slug if entity is not None else counted.slug, summary="count", op="gte",
                             values=[1.0]),
            "time": plan.time.model_copy(update={"date": date})})
    return plan


_MARGIN = re.compile(r"\bmargin|\bmarge\b", re.IGNORECASE)
_COSTLY = re.compile(r"\b(?:cost|costs|cogs|expense|expenses|co[uû]t|co[uû]ts|charges)\b", re.IGNORECASE)
_SHARE_WORDS = re.compile(r"\s*(?:%|percent(?:age)?|pct|rate|ratio|en\s+%|pourcentage|taux)\s*$", re.IGNORECASE)


def margin_over_sales(plan: Plan, model: SemanticModel, question: str) -> Plan:
    """ "Whats the gross margin % on that" was planned as the price charged over the ingredient cost: 239.8%, a
    markup. A margin is what is left of the sales once the cost is taken off, over the sales: (sales - cost) /
    sales, worked out as their difference first."""
    if plan.kind != "query" or not plan.derived:
        return plan
    named = {m.slug: m.business_name for m in model.measures.values()}
    derived: list[Derived] = []
    for d in plan.derived:
        sales, cost = d.measures
        if d.op != "ratio" or not _MARGIN.search(d.name) or sales not in named or cost not in named \
                or not _COSTLY.search(named[cost]) or _COSTLY.search(named[sales]):
            derived.append(d)
            continue
        left = _SHARE_WORDS.sub("", d.name).strip() or "Margin"
        if left.casefold() == d.name.casefold() or any(x.name.casefold() == left.casefold() for x in plan.derived):
            left = f"{left} amount"
        derived += [Derived(name=left, op="difference", measures=[sales, cost]),
                    d.model_copy(update={"measures": [left, sales]})]
    if len(derived) == len(plan.derived):
        return plan
    return plan.model_copy(update={"derived": derived})


def in_time_order(plan: Plan, model: SemanticModel, question: str) -> Plan:
    """Two periods compared, read from the earlier to the later: "How did those items do in Q1 2026?" after Q2 was
    planned as Q1 against Q2, and answered "rose most" for what fell from Q1 to Q2."""
    time = plan.time
    other = time.compare.window if time.compare is not None and time.compare.kind == "window" else None
    if plan.kind != "query" or other is None or time.window.kind != "between" or other.kind != "between" \
            or not (time.window.start and other.start) or other.start <= time.window.start:
        return plan
    return plan.model_copy(update={"time": time.model_copy(update={
        "window": other, "compare": time.compare.model_copy(update={"window": time.window})})})


def snapshot_date(plan: Plan, model: SemanticModel, question: str) -> Plan:
    """ "How many warehouses and items are in each snapshot?" was counted by the item list's own date (when its
    rows last changed), and refused: a question about snapshots is about the date of a snapshot table. The one
    snapshot date of the plan's grain is taken ("each snapshot" by day: the daily table's), or the only one."""
    if plan.kind != "query" or not _SNAPSHOT.search(question or ""):
        return plan
    dates = [r for r in model.date_roles.values() if r.kind == "snapshot" and r.is_default
             and model.tables[r.table].kind == "snapshot"]
    if not dates or any(r.slug == plan.time.date for r in dates):
        return plan
    grain = plan.time.grain
    fitting = [r for r in dates if r.granularity == grain] if grain else []
    chosen = fitting if len(fitting) == 1 else dates if len(dates) == 1 else []
    if len(chosen) != 1:
        return plan
    return plan.model_copy(update={"time": plan.time.model_copy(update={"date": chosen[0].slug})})


# "Average selling price per unit", "what percentage ... did they reject", "the return rate": a ratio asked for.
_RATIO_WORDS = re.compile(r"\bper\b|%|\bpercent(?:age)?\b|\brate\b|\bratio\b|\bshare of\b|\bproportion\b|"
                          r"\baverage\b.*\bper\b", re.IGNORECASE)
# "Revenue, quantity and price per unit": the parts asked for too.
_AND_ALSO = re.compile(r"\b(?:and|as well as|plus|along with|with)\b|,", re.IGNORECASE)


def _words_name(name: str) -> str:
    """A derived measure's name as words: "average_selling_price_per_unit" -> "Average selling price per unit"."""
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name.strip()).replace("_", " ")
    text = " ".join(text.split())
    return text[:1].upper() + text[1:] if text else name


_AVERAGE = re.compile(r"\b(?:average|avg|mean)\b", re.IGNORECASE)
_PER_SOMETHING = re.compile(r"\b(?:per|each|every)\b", re.IGNORECASE)


_CODE_PART = re.compile(r"(.*?)(_?)(?:code|cd)", re.IGNORECASE)


# "Combined", "in total": the members named are added up, not set side by side.
_TOGETHER = re.compile(r"\b(?:combined|together|total|totals|altogether|in\s+all|overall|sum\s+of)\b", re.IGNORECASE)


def each_named(plan: Plan, model: SemanticModel, question: str) -> Plan:
    """ "Revenue in Northfield Industrial and Eastgate Industrial in 2025" was one figure for the two added up: two to
    six members of one field named for a single figure are shown each, side by side (the answer still says their
    total), unless the question adds them up ("combined", "in total")."""
    if plan.kind != "query" or plan.intent != "value" or plan.group_by or plan.time.grain or plan.time.compare \
            or len(plan.filters) != 1 or _TOGETHER.search(question or ""):
        return plan
    f = plan.filters[0]
    if f.op not in ("eq", "in") or not 2 <= len(f.values) <= 6 \
            or not (f.field in model.attributes or f.field in model.entities):
        return plan
    # Each name written in the question, joined by "and" ("Organic or Vegan" is one figure for either).
    said = (question or "").casefold()
    at = [said.find(str(v).casefold()) for v in f.values]
    if min(at) < 0 or re.search(r"\bor\b", said[min(at):max(at)]) or not re.search(r"\band\b|,", said[min(at):max(at)]):
        return plan
    return plan.model_copy(update={"intent": "breakdown", "group_by": [f.field]})


def named_not_coded(plan: Plan, model: SemanticModel, question: str) -> Plan:
    """ "Top 5 medications by total cost" was grouped by the medication code and answered "medication 2563431
    leads": the rows keep each medication's description beside its code. A grouping by a code whose row keeps a
    description (or name) of the same thing is by that description, unless the question asks for the code."""
    if plan.kind != "query" or not plan.group_by or re.search(r"\bcodes?\b", question or "", re.IGNORECASE):
        return plan
    swap: dict[str, str] = {}
    for slug in plan.group_by:
        attribute = model.attributes.get(slug)
        head, _, part = slug.rpartition(".")
        # Rows of events keeping a code and its description side by side; a list's own code ("by tag code") is
        # the reader's or the AI's choice, and stands.
        found = _CODE_PART.fullmatch(part) if attribute is not None and head and attribute.entity is None else None
        if found is None or attribute.column not in model.columns:
            continue
        table = model.columns[attribute.column].table
        for word in ("description", "name", "desc"):
            other = model.attributes.get(f"{head}.{found.group(1)}{found.group(2)}{word}")
            if other is not None and other.column in model.columns and model.columns[other.column].table == table \
                    and other.slug not in plan.group_by:
                swap[slug] = other.slug
                break
    if not swap:
        return plan
    return plan.model_copy(update={"group_by": [swap.get(g, g) for g in plan.group_by],
                                   "sort": [x.model_copy(update={"by": swap[x.by]}) if x.by in swap else x
                                            for x in plan.sort]})


# A period's own name within the year: the same axis as a series of those periods inside one year.
_SAME_AXIS = {"month": "time:month_of_year", "quarter": "time:quarter_of_year"}


def one_time_axis(plan: Plan, model: SemanticModel, question: str) -> Plan:
    """ "Encounters by month" on a dashboard was planned month by month and also grouped by the month's name:
    each month came out as a series of its own, one point each. Inside one year, the month's name is the series'
    own axis: the grouping is dropped. Over several years it compares the same month of each year, and stays."""
    axis = _SAME_AXIS.get(plan.time.grain or "") if plan.kind == "query" else None
    window = plan.time.window
    if axis is None or axis not in plan.group_by or window.kind != "between" or not (window.start and window.end) \
            or window.start.year != window.end.year:
        return plan
    return plan.model_copy(update={"group_by": [g for g in plan.group_by if g != axis]})


def dates_as_window(plan: Plan, model: SemanticModel, question: str) -> Plan:
    """ "How many customers placed an order in 2024?" was written with the year as a filter on the order date
    (between 1 January and 31 December) and refused: the customers are counted by what they did in a period only
    through the plan's time. A date filter between two days, with no period of its own, is the plan's period; so
    is "since 1 January 2025" with "up to 31 December 2025" on the same date ("top prescribers in 2025" was
    written so, and the follow-ups kept both filters into another year)."""
    if plan.kind != "query" or plan.time.window.kind != "all":
        return plan
    roles = {r.slug for r in model.date_roles.values()}
    found = [f for f in plan.filters if f.field in roles and f.op == "between" and len(f.values) == 2]
    if not found:
        found = _since_and_until(plan, roles)
    if len(found) != 1 or plan.time.date not in (None, found[0].field):
        return plan
    f = found[0]
    try:
        start, end = (dt.date.fromisoformat(str(v)[:10]) for v in f.values)
    except ValueError:
        return plan
    used = [x for x in plan.filters if x.field == f.field and x.op in ("between", "gte", "gt", "lte", "lt")]
    return plan.model_copy(update={
        "filters": [x for x in plan.filters if x is not f and x not in used],
        "time": plan.time.model_copy(update={"date": f.field, "window": Window(kind="between", start=start, end=end)})})


def _since_and_until(plan: Plan, roles: set[str]) -> list[Filter]:
    """One "since" and one "up to" filter on the same date, as the one span they make (empty when there are not
    exactly those two)."""
    on = [f for f in plan.filters if f.field in roles and f.op in ("gte", "gt", "lte", "lt") and len(f.values) == 1]
    lower = [f for f in on if f.op in ("gte", "gt")]
    upper = [f for f in on if f.op in ("lte", "lt")]
    if len(lower) != 1 or len(upper) != 1 or lower[0].field != upper[0].field:
        return []
    try:
        start = dt.date.fromisoformat(str(lower[0].values[0])[:10])
        end = dt.date.fromisoformat(str(upper[0].values[0])[:10])
    except ValueError:
        return []
    start += dt.timedelta(days=1) if lower[0].op == "gt" else dt.timedelta(0)
    end -= dt.timedelta(days=1) if upper[0].op == "lt" else dt.timedelta(0)
    if end < start:
        return []
    return [Filter(field=lower[0].field, op="between", values=[start.isoformat(), end.isoformat()])]


_DATE_WORDS = {"date", "dates", "time", "day", "on", "at", "of", "the", "when", "dt"}


def date_in_a_name(plan: Plan, model: SemanticModel, question: str) -> Plan:
    """ "Order subtotal by ship method in 2024" was dated by the ship date: the word "ship" is the method's, not a
    date's. A date the question names only inside the name of a field it groups or filters by is no date it
    asked for: the measure's own date is taken."""
    role = next((r for r in model.date_roles.values() if r.slug == plan.time.date), None) \
        if plan.kind == "query" and plan.time.date else None
    if role is None or role.is_default or not question:
        return plan
    own = [w for w in re.findall(r"[a-z0-9]+", role.name.casefold()) if w not in _DATE_WORDS]
    if not own:
        return plan
    from core2.resolve.resolver import find_slug

    left = question.casefold()
    for slug in [*plan.group_by, *(f.field for f in plan.filters)]:
        found = find_slug(model, slug)
        if found is None or not hasattr(found[1], "business_name"):
            continue
        names = [found[1].business_name]
        if getattr(found[1], "entity", None) in model.entities:
            names.append(model.entities[found[1].entity].business_name)
        for name in names:
            words = [w for w in re.findall(r"[a-z0-9]+", name.casefold()) if w not in ("name", "id", "code")]
            if len(words) >= 2:
                left = re.sub(r"\b" + r"\s+".join(re.escape(w) + "s?" for w in words) + r"\b", " ", left)
    if left == question.casefold() or any(re.search(rf"\b{re.escape(w)}", left) for w in own):
        return plan
    return plan.model_copy(update={"time": plan.time.model_copy(update={"date": None})})


def list_date_unasked(plan: Plan, model: SemanticModel, question: str) -> Plan:
    """ "How many customers bought from us in Q2 2026?" was dated by the customer's first invoice date, a date on
    the customer list, and counted 0 (the profitability stress test). Members of a period are those its rows name;
    a date on the member list itself counts them only when the question names it ("customers created in 2025",
    "by first invoice date")."""
    role = next((r for r in model.date_roles.values() if r.slug == plan.time.date), None) \
        if plan.kind == "query" and plan.time.date and plan.time.window.kind != "all" else None
    if role is None or role.table not in model.tables or model.tables[role.table].kind != "dimension" or not question:
        return plan
    own = [w for w in re.findall(r"[a-z]+", role.name.casefold()) if w not in _DATE_WORDS and len(w) > 2]
    said = re.findall(r"[a-z]+", question.casefold())
    if not own or any(names.same_word(w, s) for w in own for s in said):
        return plan
    return plan.model_copy(update={"time": plan.time.model_copy(update={"date": None})})


def average_asked(plan: Plan, model: SemanticModel, question: str) -> Plan:
    """ "Which territory had the highest average order subtotal?" was planned as the subtotal added up, and named
    the territory with the most sales. A figure that adds up, written after "average", is each row's figure
    averaged (per, of none): the average order's subtotal in each territory. "Average sales per customer" says
    what to average over, and is the AI's to plan."""
    if plan.kind != "query" or plan.per is not None or plan.rolling is not None or plan.derived or plan.durations \
            or plan.bands is not None or plan.time.compare is not None or len(plan.measures) != 1:
        return plan
    measure = next((m for m in model.measures.values() if m.slug == plan.measures[0]), None)
    hit = _AVERAGE.search(question or "")
    if measure is None or hit is None:
        return plan
    if getattr(measure.expr, "agg", "") in ("max", "min"):
        # "Average list price by product line" took the highest price of a price history: the number each
        # product has, of that name, averaged over the products.
        name = measure.business_name.casefold()
        own = [a for a in model.attributes.values() if a.kind == "number" and a.business_name.casefold() == name]
        if len(own) != 1:
            return plan
        return plan.model_copy(update={"measures": [own[0].slug], "sort": [
            s.model_copy(update={"by": own[0].slug}) if s.by == measure.slug else s for s in plan.sort]})
    if getattr(measure.expr, "agg", "") != "sum":
        return plan
    rest = re.split(r"[?.;!]", question[hit.end():], maxsplit=1)[0]
    if _PER_SOMETHING.search(rest) or measure.slug not in measures_named(model, rest):
        return plan
    return plan.model_copy(update={"per": PerMember(of=None, summary="avg")})


def _plain_words(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def ratio_alone(plan: Plan, model: SemanticModel, question: str) -> Plan:
    """A ratio the question asks for is the answer, under a name in words.

    "What is our average selling price per unit by item group?" was planned as the amount, the quantity and their
    ratio, and answered with the amount ("Electrical leads with $4.39M"), the ratio last and named
    "average_selling_price_per_unit". The ratio's own parts are left out of the answer unless the question asks for
    them too ("revenue, quantity and price per unit"), and a sort on a part sorts on the ratio. A figure per row of
    a count ("average days supply per fill", each fill's days supply averaged) is never also split by those rows.
    """
    if plan.kind != "query":
        return plan
    # "What was our gross profit in 2013?" was written with the derived gross_profit among the measures too, and
    # refused ("no measure called gross_profit"): a derived figure is shown as it is, and named among the measures
    # only when another derived figure is made of it (the reader asked for both).
    parts_of = {_plain_words(m) for d in plan.derived for m in d.measures}
    made = {_plain_words(d.name) for d in plan.derived}
    slugs = {m.slug for m in model.measures.values()} if model is not None else set()
    listed = [m for m in plan.measures if m in slugs or _plain_words(m) not in made or _plain_words(m) in parts_of]
    if listed != plan.measures:
        plan = plan.model_copy(update={"measures": listed})
    update: dict[str, Any] = {}
    renamed = {d.name: _words_name(d.name) for d in plan.derived if _words_name(d.name) != d.name}
    derived = [d.model_copy(update={"name": renamed.get(d.name, d.name),
                                    "measures": [renamed.get(m, m) for m in d.measures]}) for d in plan.derived]
    sort = [s.model_copy(update={"by": renamed[s.by]}) if s.by in renamed else s for s in plan.sort]
    if any(m in renamed for m in plan.measures):
        update["measures"] = [renamed.get(m, m) for m in plan.measures]
    parts = {m for d in derived for m in d.measures}
    if derived and plan.measures and set(plan.measures) <= parts and _RATIO_WORDS.search(question or "") \
            and not _AND_ALSO.search(question or ""):
        update["measures"] = []
        sort = [s.model_copy(update={"by": derived[0].name}) if s.by in parts else s for s in sort]
    if renamed or derived != plan.derived:
        update["derived"] = derived
    # A duration's name too ("days_between_order_and_ship" -> "Days between order and ship"), wherever the plan
    # refers to it: a sort, a condition, ranges of it.
    spans = {d.name: _words_name(d.name) for d in plan.durations if _words_name(d.name) != d.name}
    if spans:
        update["durations"] = [d.model_copy(update={"name": spans.get(d.name, d.name)}) for d in plan.durations]
        sort = [s.model_copy(update={"by": spans[s.by]}) if s.by in spans else s for s in sort]
        update["filters"] = [f.model_copy(update={"field": spans[f.field]}) if f.field in spans else f
                             for f in plan.filters]
        if plan.bands is not None and plan.bands.of in spans:
            update["bands"] = plan.bands.model_copy(update={"of": spans[plan.bands.of]})
    if sort != plan.sort:
        update["sort"] = sort
    if plan.per is not None and plan.per.of and plan.group_by:
        counted = next((m for m in model.measures.values() if m.slug == plan.per.of), None)
        if counted is not None:
            rows = counted.table
            own = [g for g in plan.group_by
                   if g in model.entities and any(model.columns[c].table == rows
                                                  for c in model.entities[g].key_columns if c in model.columns)
                   or g not in model.entities and g not in model.attributes and g.replace(".", "_") in counted.slug]
            if own:
                update["group_by"] = [g for g in plan.group_by if g not in own]
    return plan.model_copy(update=update) if update else plan


_PLAIN_WORDS = {"of", "the", "a", "an", "and", "or", "per", "by", "on", "in", "for", "to", "number", "fact", "fct",
                "table", "data", "n2", "n3"}


def _word_set(text: str) -> set[str]:
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+", text.casefold())}


def named_twin(plan: Plan, model: SemanticModel, question: str, previous: Plan | None = None) -> Plan:
    """Of two measures with one name, the one the question's words point at, else the one that holds the name.

    "Which item groups have the highest average item cost?" names a figure kept on a daily balance and again on a
    monthly one, and the AI took the monthly one. The words that tell the two apart (each one's table, and the
    words its slug adds to the name: "on hand", "period") pick one when the question says them; when it says none,
    the one on the table the conversation was already measuring is kept ("Split that by promotion" after the
    resellers' discount; "average freight per order" after sales orders), else the one core2.plan.catalog.taken_of
    names (of tables of events, the largest; else the plain name; never a staging copy): the Metrics page says
    so, and flags the pair.
    """
    from core2.plan.catalog import same_named, taken_of

    if plan.kind != "query":
        return plan
    twins = same_named(model)
    if not twins:
        return plan
    by_slug = {m.slug: m for m in model.measures.values()}
    asked = _word_set(question)
    data = plan.model_dump()

    def slugs(value: Any) -> set[str]:
        if isinstance(value, str):
            return {value} if value in by_slug else set()
        if isinstance(value, dict):
            return set().union(*(slugs(v) for k, v in value.items() if k != "notes"))
        if isinstance(value, list):
            return set().union(*(slugs(v) for v in value)) if value else set()
        return set()

    swap: dict[str, str] = {}
    for slug in slugs(data):
        chosen = by_slug[slug]
        if chosen.key not in twins:
            continue
        options = [chosen, *twins[chosen.key]]
        name = _word_set(chosen.business_name)

        def told_apart(m: Measure) -> set[str]:
            own = _word_set(m.slug.replace("_", " ")) | _word_set(model.tables[m.table].business_name) | {
                w for ws in m.synonyms.values() for word in ws for w in _word_set(word)}
            return own - name - _PLAIN_WORDS

        said = {m.slug: len(told_apart(m) & asked) for m in options}
        most = max(said.values())
        pick = [m for m in options if said[m.slug] == most]
        if most == 0 and not (name - _PLAIN_WORDS) & asked:
            pick = [chosen]            # the question does not name it: the conversation's own figure stands
        elif len(pick) > 1:
            # Named, but by words both say ("the average exchange rate", each one's synonyms saying "exchange"):
            # the one the conversation was measuring, else the one that holds the plain name.
            before = {m.table for m in pick if previous is not None and m.slug in slugs(previous.model_dump())}
            same_table = [m for m in pick if m.table in {by_slug[x].table for x in slugs(previous.model_dump())}] \
                if previous is not None else []
            if before or len(same_table) == 1:
                pick = [m for m in pick if m.table in before] or same_table
            else:
                pick = [taken_of(model, pick) or chosen]
        if len(pick) == 1 and pick[0].slug != slug and pick[0].slug not in slugs(data):
            swap[slug] = pick[0].slug
    if not swap:
        return plan

    def put(value: Any) -> Any:
        if isinstance(value, str):
            return swap.get(value, value)
        if isinstance(value, dict):
            return {k: (v if k == "notes" else put(v)) for k, v in value.items()}
        if isinstance(value, list):
            return [put(v) for v in value]
        return value

    return Plan.model_validate(put(data))


# "Later than the requested date", "delayed": the share late. "On time" said alone: the share on time.
_LATE = re.compile(r"\b(?:late|later\s+than|delayed|delays?|missed|behind\s+schedule)\b", re.IGNORECASE)
_ON_TIME = re.compile(r"\bon[\s-]time\b", re.IGNORECASE)


# "So far", "to date", "ever": everything up to now, when no period is named with it.
_ALL_TIME = re.compile(r"\b(?:so far|to date|to this day|ever|all[- ]time|of all time|jusqu'(?:à|a) (?:pr(?:é|e)sent|"
                       r"maintenant)|(?:à|a) ce jour|depuis (?:toujours|le d(?:é|e)but))\b", re.I)
# A period named in the question: a year, a quarter, a month, "this", "last", "since".
_PERIOD_SAID = re.compile(r"\b(?:year|years|month|months|quarter|quarters|week|weeks|today|yesterday|this|last|past|"
                          r"previous|since|ytd|mtd|qtd|fy\s?\d{2,4}|(?:19|20)\d\d|q[1-4]|h[12]|t[1-4]|s[12]|"
                          r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
                          r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|ann(?:é|e)e|mois|trimestre|"
                          r"semaine|semestre|exercice|cette|dernier|derni(?:è|e)re|depuis)\b", re.I)


# Why a question was refused, when the reason is that it names nothing to measure.
_VAGUE = re.compile(r"\b(?:vague|ambiguous|unclear|too (?:broad|general)|not specific|no specific|does not specify|"
                    r"doesn't specify|unspecified|open[- ]ended)\b", re.I)
_WHICH_FIGURES = "Which figures would you like to see? These I can answer"
# A question about the data itself, not its figures: "what data do you have", "what can I ask", "how far back".
_ABOUT_THE_DATA = re.compile(r"\b(?:data|dataset|database|tables?|fields?|columns?|ask|know|cover|covers|covered|"
                             r"mean|means|meaning|calculated|defined|definition|available|go back|far back|"
                             r"donn(?:é|e)es|tables?|champs?|demander|signifie|couvre)\b", re.I)


def route_named(plan: Plan, model: SemanticModel, question: str) -> Plan:
    """"The 5 best French customers", "revenue by supplier nation": a field the rows reach by two routes (an order
    line's nation is its customer's or its supplier's) is reached through the one the question names (the sector
    stress test: the shorter route, the supplier's, was taken for the customers' nation)."""
    from core2.resolve.resolver import find_slug

    if plan.kind != "query":
        return plan
    said = {names.singular(w) for w in re.findall(r"[a-z]+", question.casefold())}
    tables = {m.table for m in model.measures.values() if m.slug in plan.measures}
    via = dict(plan.via)
    for slug in [*plan.group_by, *(f.field for f in plan.filters)]:
        found = find_slug(model, slug)
        if slug in via or not found or found[0] not in ("attribute", "entity"):
            continue
        goal = model.columns[found[1].column].table if found[0] == "attribute" else found[1].table
        routes = {r for t in tables for r in P.through_entities(model, t, goal)}
        named = [e for e in model.entities.values() if e.table in routes and any(
            {names.singular(w) for w in re.findall(r"[a-z]+", word.casefold())} <= said
            for word in [e.business_name, *(w for ws in e.synonyms.values() for w in ws)] if word)]
        if len(named) == 1:
            via[slug] = named[0].slug
    return plan if via == plan.via else plan.model_copy(update={"via": via})


def all_time_asked(plan: Plan, model: SemanticModel, question: str) -> Plan:
    """"How many loans have we charged off so far, and for how much?" was answered for this year alone (the sector
    stress test): with no period named, "so far", "to date" and "ever" are everything up to now."""
    window = plan.time.window if plan.kind == "query" else None
    if window is None or window.kind == "all" or not _ALL_TIME.search(question) \
            or _PERIOD_SAID.search(_ALL_TIME.sub(" ", question)):
        return plan
    return plan.model_copy(update={"time": plan.time.model_copy(update={"window": Window()})})


def late_asked(plan: Plan, question: str) -> Plan:
    """The on-time share or the late share, as the question words it: "What share of order lines had a confirmed
    delivery date later than the requested one?" was answered with the share on time (the profitability stress test)."""
    ot = plan.on_time if plan.kind == "query" else None
    if ot is None or ot.ageing:
        return plan
    late = bool(_LATE.search(question)) and not _ON_TIME.search(question)
    on_time = bool(_ON_TIME.search(question)) and not _LATE.search(question)
    if late and not ot.late or on_time and ot.late:
        return plan.model_copy(update={"on_time": ot.model_copy(update={"late": late})})
    return plan


def pointed_in(plan: Plan, model: SemanticModel, matches: list[ValueMatch], question: str = "",
               on_screen: list[tuple] | None = None) -> Plan:
    """A filter value written as a stand-in ("first_warehouse_id", "<top warehouse from previous answer>") for the
    one member of that field the question's words found ("the first one" -> warehouse "301"): that member, with no
    second call to the AI. When the words found none ("the first supplier on that list", the field being called
    party) but the AI put the stand-in on the field the answer on screen is grouped by, the question's one place
    word ("first", "top", "last") picks that row. A stand-in with neither is left for the repair round to ask
    about."""
    if plan.kind != "query" or not plan.filters or not (matches or on_screen):
        return plan
    places = {_PLACES[m.group(1).lower()] for m in _PLACE_WORD.finditer(question or "")}

    def attribute_of(slug: str) -> str | None:
        if slug in model.attributes:
            return slug
        entity = model.entities.get(slug)
        if entity is None:
            return None
        column = entity.label_column or entity.code_column
        return next((a.slug for a in model.attributes.values() if a.column == column), None)

    def thing(slug: str | None) -> str | None:
        attribute = model.attributes.get(slug or "")
        return attribute.entity if attribute is not None else None

    def stand_in(value: object) -> bool:
        # "TOP" for "the top group": the question's own place word, written as a value no member has.
        return stands_for(value) or str(value).casefold() in _PLACES and not any(
            str(m.value).casefold() == str(value).casefold() for m in matches)

    filters = []
    for f in plan.filters:
        if f.op in ("eq", "in") and f.values and all(stand_in(v) for v in f.values):
            field = attribute_of(f.field)
            found = {m.value for m in matches if field is not None and m.attribute == field}
            shown = str(on_screen[0][0]) if on_screen else None
            # The group's code for the group shown by its name: the same thing, pointed at by its row on screen.
            if not found and field is not None and field != shown and thing(field) and thing(field) == thing(shown):
                field = shown
            if not found and on_screen and field is not None and len(places) == 1 \
                    and shown == field and next(iter(places)) < len(on_screen):
                row = on_screen[next(iter(places))]
                found = {row[1]} if row[1] is not None else set()
            one = {row[1] for row in on_screen or [] if str(row[0]) == shown and row[1] is not None}
            if not found and not places and field is not None and shown == field and len(one) == 1:
                # "What was their sales quota?" after the one sales rep who sold most: the one member on screen.
                found = one
            if len(found) == 1:
                f = f.model_copy(update={"field": field, "op": "eq", "values": [next(iter(found))]})
        filters.append(f)
    return plan.model_copy(update={"filters": filters}) if filters != plan.filters else plan


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
    problems += _money_on_other(plan, model, question)
    problems += _counts_what_it_asks(plan, model, question)
    problems += _counted_by_itself(plan, model)
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


# "$1,000", "€2.5k", "1,000 dollars": an amount of money the question sets a condition on.
_MONEY = re.compile(r"[$€£¥]\s?(\d[\d,]*(?:\.\d+)?)\s*([km])?\b|\b(\d[\d,]*(?:\.\d+)?)\s*([km])?\s*(?:dollars|euros|"
                    r"pounds|usd|eur|gbp|cad|aud)\b", re.IGNORECASE)
# A number field named for something that is not money: a weight, a count, days, a share.
_NOT_MONEY = _word_set("weight wt qty quantity count days percent pct rate length height width volume age units number "
                       "size score hours minutes")
_MONEY_WORDS = _word_set("cost price amount amt value revenue sales fee charge spend margin profit paid pay payment "
                         "balance income expense discount tax budget")


def _money_on_other(plan: Plan, model: SemanticModel, question: str) -> list[str]:
    """ "How many items cost more than $1,000 each?" with the condition set on the items' net weight (a real model
    wrote it so, after first setting it on a date): an amount of money set on a figure that is not money."""
    amounts: dict[float, str] = {}
    for m in _MONEY.finditer(question or ""):
        digits, scale = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        amount = float(digits.replace(",", "")) * {"k": 1e3, "m": 1e6}.get((scale or "").lower(), 1)
        amounts[amount] = m.group(0).strip()
    if not amounts:
        return []
    measures = {m.slug: m for m in model.measures.values()}
    dates = {r.slug for r in model.date_roles.values()}
    problems = []
    for f in plan.filters:
        said = next((amounts[float(v)] for v in f.values if isinstance(v, (int, float)) and float(v) in amounts), None)
        if said is None or f.op not in ("gt", "gte", "lt", "lte", "between", "eq", "ne"):
            continue
        if f.field in measures:
            words, other = _word_set(f"{f.field} {measures[f.field].business_name}"), measures[f.field].format not in (
                "currency", None)
        elif f.field in model.attributes:
            column = model.columns[model.attributes[f.field].column]
            words = _word_set(f"{f.field.replace('.', ' ')} {column.business_name} {column.name.replace('_', ' ')}")
            other = column.format not in ("currency", None) or bool(words & _NOT_MONEY)
        else:
            words, other = set(), f.field in dates
        if not other or words & _MONEY_WORDS:
            continue
        asked = _word_set(question)
        money = sorted((m for m in measures.values() if m.format == "currency" and not m.hidden),
                       key=lambda m: (-len(_word_set(m.business_name) & asked), m.slug))
        offered = ", ".join(m.slug for m in money[:4] if _word_set(m.business_name) & asked) or ", ".join(
            m.slug for m in money[:4])
        problems.append(f'"{said}" is money, and {f.field} is not. A condition on money is set on a money figure'
                        f"{(' (' + offered + ')') if offered else ''}: a filter on that measure, with the members "
                        'it is for ("each item") in group_by.')
    return problems


_HOW_MANY = re.compile(r"\bhow\s+many\s+(?:different\s+|distinct\s+|unique\s+|separate\s+)?([a-z]+(?:\s+[a-z]+)?)",
                       re.IGNORECASE)


def _counts_what_it_asks(plan: Plan, model: SemanticModel, question: str) -> list[str]:
    """ "How many items does the top one have on hand?" was answered with the stock on hand: "how many" of a thing
    the data lists asks for a count of that thing, or a list of them."""
    if plan.kind != "query" or plan.per is not None or plan.absent is not None or plan.bands is not None:
        return []
    problems = []
    for m in _HOW_MANY.finditer(question or ""):
        said = m.group(1).lower()
        for e in model.entities.values():
            name = (e.business_name or e.slug).lower()
            if not (said.startswith(names.plural(name).lower()) or said == name or said.startswith(name + " ")):
                continue
            plural = names.plural(name).lower()
            counts = [x for x in model.measures.values() if not x.hidden and x.table == e.table
                      and getattr(x.expr, "agg", None) in ("count", "count_distinct")]
            asked = [x for x in model.measures.values() if x.slug in plan.measures]
            counted = any(x in counts or plural in (x.business_name or "").lower() for x in asked)
            listed = e.slug in plan.group_by or any(model.attributes[g].entity == e.slug for g in plan.group_by
                                                    if g in model.attributes)
            if not counted and not listed:
                offered = ", ".join(x.slug for x in counts[:2])
                problems.append(f'"how many {plural}" asks for a count of {plural}'
                                + (f" ({offered})" if offered else "") + f", or the {plural} listed (group_by "
                                f"{e.slug}) with the conditions the question sets; not another figure of them.")
            break
    return problems


def _figure_against_figure(plan: Plan, model: SemanticModel) -> list[str]:
    """A filter that compares a figure with another figure ("stores whose sales were below target") written as
    the other figure's name in quotes: the database compared the sales with the text 'target_amount' and refused."""
    from core2.resolve.resolver import find_slug

    out = []
    for f in plan.filters:
        named = [v for v in f.values if isinstance(v, str) and (found := find_slug(model, v)) is not None
                 and found[0] == "measure"]
        if named:
            out.append(f"the filter on {f.field} compares it with {named[0]}, another figure, as if it were a value: "
                       "a figure against its target is the plan's target; any other comparison of two figures is a "
                       "derived difference (the first less the second), filtered on its name (lt 0 for below).")
    return out


def _total_against_average(plan: Plan, model: SemanticModel) -> list[str]:
    """A figure worked out from a total and an average of the same rows ("the discount rate, discount over list
    price": the discounts added up over the list price per unit, averaged) is no rate nor any figure at all: 71
    million percent (the sector stress test). Both are totals: the total the average is the unit of, when the rows
    keep one (the extended list price)."""
    if plan.kind != "query":
        return []
    by_slug = {m.slug: m for m in model.measures.values() if m.slug}
    out = []
    for d in plan.derived:
        pair = [by_slug.get(s) for s in d.measures]
        if None in pair or pair[0].table != pair[1].table:
            continue
        aggs = [getattr(m.expr, "agg", "sum" if m.additivity == "additive" else "") for m in pair]
        if sorted(aggs) != ["avg", "sum"]:
            continue
        average, total = (pair[0], pair[1]) if aggs[0] == "avg" else (pair[1], pair[0])
        words = set(_words(average.business_name)) - {"unit", "per", "average", "avg"}
        twins = [m.slug for m in model.measures.values() if m.table == average.table and m is not total
                 and getattr(m.expr, "agg", "sum" if m.additivity == "additive" else "") == "sum"
                 and words and words <= set(_words(m.business_name))]
        out.append(f"{d.name} sets {total.slug}, a total, against {average.slug}, an average per row: worked out from "
                   "the two it is no figure. Use totals for both" + (f" ({', '.join(twins[:2])})." if twins else "."))
    return out


def _counted_by_itself(plan: Plan, model: SemanticModel) -> list[str]:
    """ "What are the most common sales reasons on internet orders?" was planned as the number of sales reasons
    by sales reason: 1 each. A count of a list's own members grouped by those members says nothing; how often
    each occurs is a count of the rows of a table that records them."""
    if plan.kind != "query" or plan.per is not None or not plan.group_by:
        return []
    problems = []
    for m in model.measures.values():
        if m.slug not in plan.measures or getattr(m.expr, "agg", None) not in ("count", "count_distinct"):
            continue
        entity = next((e for e in model.entities.values() if e.table == m.table), None)
        if entity is None or m.expr.agg == "count_distinct" and m.expr.column not in (
                *entity.key_columns, entity.code_column, entity.label_column):
            continue
        own = {entity.label_column, entity.code_column, *entity.key_columns} - {None}
        # Grouped through a link of the list to itself ("by manager": each manager's employees), it is no
        # longer each member counting itself.
        if not any((g == entity.slug or g in model.attributes and model.attributes[g].column in own)
                   and not plan.via.get(g) for g in plan.group_by):
            continue
        plural = names.plural((entity.business_name or entity.slug).lower())
        others = [x.slug for x in model.measures.values() if not x.hidden and x.table != m.table and x.slug
                  and getattr(x.expr, "agg", None) == "count" and P.all_paths(model, x.table, m.table)][:4]
        problems.append(f"{m.slug} counts the {plural} themselves: grouped by {entity.slug}, each one is 1. How "
                        f"often each occurs is a count of the rows that record them, grouped by {entity.slug}"
                        + (f" ({', '.join(others)})" if others else "") + ".")
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


def fill_in(plan: Plan, model: SemanticModel, previous: Plan | None, said_follow_up: bool,
            question: str = "") -> Plan:
    """The plan with what it plainly means filled in, before it is checked:

    * a ranking sorted by a measure the plan does not show shows it ("rank the items by back-orders" sorted by
      a measure it left out, and was refused);
    * a stock compared between two dates written as one span ("on hand between 27 May and 17 August", as
      27 May-17 August against itself, or against 27 May) is the later date against the earlier one;
    * a plan that says nothing of the answer above, for the same measure, a filter or the period right after it,
      follows it ("show the returns by warehouse" after the returns of Q1).
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
    # A plan that says nothing of the answer above but keeps its figure, a filter of it or its period follows it
    # ("how many orders?" planned with North and March and no word of follow-up): rule 11 reads it so.
    window = plan.time.window
    if not said_follow_up and previous is not None and previous.kind == "query" \
            and (set(plan.measures) & set(previous.measures)
                 or not names_own_period(question) and (any(f in previous.filters for f in plan.filters)
                                                        or window.kind != "all" and window == previous.time.window)):
        update["follow_up"] = "refine"
    return plan.model_copy(update=update) if update else plan


def _named_in_full(model: SemanticModel, question: str):
    """The one measure the question names by its whole name ("scrapped quantity"), or None: none, or several of the
    same name (an order quantity of sales, of purchases and of work orders)."""
    text = " ".join(re.findall(r"[a-z0-9]+", question.casefold()))
    said = [m for m in model.measures.values() if not m.hidden and m.business_name and len(m.business_name) > 3
            and re.search(rf"\b{re.escape(' '.join(re.findall(r'[a-z0-9]+', m.business_name.casefold())))}\b", text)]
    # A longer name holds a shorter one ("net sales amount" holds "sales amount"): the longest is the one named.
    longest = max((len(m.business_name) for m in said), default=0)
    said = [m for m in said if len(m.business_name) == longest]
    return said[0] if len({m.business_name.casefold() for m in said}) == 1 and len(said) == 1 else None


def _measures_named(model: SemanticModel, question: str) -> set[str]:
    from core2.plan.followup import measures_named

    return measures_named(model, question)


def carried(plan: Plan, previous: Plan | None, reading: str | None) -> Plan:
    """A turn that measures nothing (the AI left the figure out of "break it down by profit centre", or read "which
    division grew the most?" as new and found no figure) measures what the answer on screen did: its measures and
    the figures it worked out. A list or a count of things measures nothing on purpose."""
    # Planned with the conversation in view (a turn its words did not settle): a question that measures nothing
    # measures the answer on screen, whatever the AI called it; one planned afresh never has a previous.
    following = reading != "new"
    if plan.kind != "query" or not following or previous is None or previous.kind != "query" \
            or plan.measures or plan.derived or plan.durations or plan.intent in ("list", "count") \
            or not (previous.measures or previous.derived):
        return plan
    return plan.model_copy(update={"measures": list(previous.measures), "derived": list(previous.derived),
                                   "follow_up": "refine"})


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
    compared = _figure_against_figure(plan, model) or _total_against_average(plan, model)
    if compared:
        return compared, ""
    try:
        resolve(plan, model, Context(today=today))
    except ResolveError as exc:
        if exc.kind in ("unconfirmed", "denied", "ambiguous"):
            return [], ""          # not the planner's to fix: the service answers or asks the reader
        options = f" Allowed: {', '.join(exc.options[:12])}." if exc.options else ""
        if exc.message.endswith(" has no date"):
            return [f"{exc.message}.{options}"], _dated(model, plan, question)
        return [f"{exc.message}.{options}"], _reaching(model, exc.unlinked, question) if exc.unlinked else ""
    except ValueError as exc:
        return [str(exc)], ""
    return [], ""


_COMMON = {"a", "an", "the", "of", "by", "in", "for", "per", "and", "or", "to", "on", "at", "is", "are", "was",
           "what", "which", "how", "each", "our", "my", "me", "show", "give", "total", "q1", "q2", "q3", "q4"}


def _words(text: str) -> set[str]:
    return {names.singular(w) for w in re.findall(r"[a-z0-9]+", text.casefold())
            if w not in _COMMON and not w.isdigit()}


def _dated(model: SemanticModel, plan: Plan, question: str) -> str:
    """The measures with a date that reach the plan's groupings, those sharing the question's words first.

    "Which payer covered the most cost in 2025?" took the amount each payer covered over its lifetime, kept on
    the payer list with no date, and gave up; the encounters' payer coverage is dated and reaches the payers."""
    from core2.resolve.resolver import find_slug

    targets = set()
    for slug in plan.group_by:
        found = find_slug(model, slug)
        column = getattr(found[1], "column", None) if found else None
        table = model.columns[column].table if column in model.columns else getattr(found[1], "table", None) \
            if found else None
        if table:
            targets.add(table)
    asked = _words(question)
    found_: list[tuple[int, str, Measure]] = []
    for m in model.measures.values():
        if m.hidden or not m.slug or m.table not in model.tables or not model.tables[m.table].default_date:
            continue
        if all(t == m.table or P.all_paths(model, m.table, t) for t in targets):
            said = " ".join([m.business_name, *(w for words in m.synonyms.values() for w in words)])
            found_.append((-len(asked & _words(said)), m.business_name.casefold(), m))
    if not found_:
        return ""
    found_.sort(key=lambda f: (f[0], f[1]))
    listed = ", ".join(f"{m.slug} ({m.business_name}, on {model.tables[m.table].business_name})"
                       for _, _, m in found_[:8])
    return (f"Measures with a date that can be broken down that way: {listed}. A figure kept with no date (a total "
            "on a list) cannot be limited to a period: if the question's words name one of these, use it instead.")


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


def _member_filters(data: dict[str, Any], model: SemanticModel | None) -> list[dict[str, Any]]:
    """A dumped plan's conditions on members (a customer, a region): not amounts, dates or numbers."""
    durations = {str(d.get("name", "")).casefold() for d in data.get("durations", [])}
    absent = data.get("absent") or {}
    out = []
    for f in [*data.get("filters", []), *absent.get("without", []), *absent.get("had", [])]:
        if model is not None and not about_people(model, f.get("field")) and (
                f.get("field") in model.measures or f.get("field") in model.date_roles
                or getattr(model.attributes.get(f.get("field")), "kind", "") == "number"):
            continue      # amounts, dates and numbers ("3 days in transit"), not member values; never a person's
        if str(f.get("field", "")).casefold() in durations:
            continue      # numbers of days
        out.append(f)
    return out


def _plan_shown(plan: Plan, model: SemanticModel | None, masked: Masked | None) -> str:
    """A plan as the AI may see it: member values as placeholders when values are withheld."""
    if masked is None:
        return plan.model_dump_json(exclude_defaults=True)
    data = plan.model_dump(mode="json", exclude_defaults=True)
    for f in _member_filters(data, model):
        f["values"] = [placeholder(v, masked.values) for v in f.get("values", [])]
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _plan_line(label: str, plan: Plan, model: SemanticModel | None, masked: Masked | None) -> str:
    """``label: plan`` for the tail; a plan that shows member values as stored says so in its heading, which the AI
    egress log reads as values sent (core/llm_audit.py)."""
    stored = masked is None and any(f.get("values") for f in _member_filters(
        plan.model_dump(mode="json", exclude_defaults=True), model))
    return f"{label}{' (member values as stored)' if stored else ''}: {_plan_shown(plan, model, masked)}"


def _unmask_plan(plan: Plan, masked: Masked | None) -> Plan:
    if masked is None or not masked.values:
        return plan
    data = plan.model_dump()
    absent = data.get("absent") or {}
    for f in [*data.get("filters", []), *absent.get("without", []), *absent.get("had", [])]:
        f["values"] = [unmask(v, masked) for v in f.get("values", [])]
    return Plan.model_validate(data)


def _derived_named(question: str, defined: list) -> list:
    """The figures the conversation worked out (gross profit) that the question names again, latest first."""
    asked = _word_set(question)
    out, seen = [], set()
    for d in reversed(defined):
        words = _word_set(_words_name(d.name)) - _PLAIN_WORDS
        if words and words <= asked and d.name.casefold() not in seen:
            seen.add(d.name.casefold())
            out.append(d)
    return out


def _members_compared(question: str, hidden: list[ValueMatch], model: SemanticModel) -> tuple[str, list[str]] | None:
    """Two or more members of one field the question writes ("actual and budget"), and that field's slug."""
    by_field: dict[str, list[str]] = {}
    for m in hidden:
        by_field.setdefault(m.attribute, [])
        if m.text not in by_field[m.attribute]:
            by_field[m.attribute].append(m.text)
    found = [(slug, said) for slug, said in by_field.items() if len(said) >= 2 and slug in model.attributes]
    return found[0] if len(found) == 1 else None


def last_plan(history: list[Turn]) -> Plan | None:
    """The plan a question follows: the last answered turn's (or the question asked back); a turn that got no answer
    has none, and the one before it is what was on screen."""
    return next((turn.plan for turn in reversed(history) if turn.plan is not None and not turn.unanswered), None)


# A question back about the conversation itself: "What about the answer above: change the time period, ...?"
_WHICH_WAY = re.compile(r"\b(?:answer\s+above|previous\s+answer|previous\s+question|r[ée]ponse\s+ci-dessus|"
                        r"follow[-\s]?up|new\s+question)\b", re.IGNORECASE)


def _asks_which_way(clarify: Clarify) -> bool:
    """A question back about the conversation itself, not which member, link, date or figure ("which store from the
    answer above do you mean?" is a fair question)."""
    return clarify.about == "follow_up" or clarify.about == "other" and bool(_WHICH_WAY.search(clarify.question or ""))


def plan_question(model: SemanticModel, question: str, complete: Complete, *, today: dt.date,
                  history: list[Turn] | None = None, matches: list[ValueMatch] | None = None,
                  values_allowed: bool = True, scrub: Callable[[str], str] | None = None,
                  hidden: list[ValueMatch] | None = None,
                  reading: str | None = None, on_screen: list[tuple] | None = None,
                  written: list[ValueMatch] | None = None, defined: list | None = None) -> Outcome:
    """The question's plan, checked against the model; at most two calls to the AI.

    ``reading``: "refine" or "new" when the question's words already say which (core2/plan/followup.py);
    the plan is marked so, whatever the AI said. None: the AI reads it with the conversation in view and decides
    (never asked back); a follow-up then keeps the filters and the period from above (core2/plan/thread.py).
    ``on_screen``: the members of the answer the question follows, in the order shown (attribute slug, value, ...).
    ``written``: every member name the question writes, quoted or not (two compared: "actual and budget").
    ``defined``: the figures the conversation's plans worked out (gross profit), oldest first: definitions only,
    never a figure of an answer, so a new question may use them too.
    """
    matches = matches or []
    history = history or []
    before = last_plan(history)
    # The placeholders the conversation already used for member values, from the last turn that had any: a turn that
    # got no answer may have none, and "<v1>" must still mean the same member.
    known = next((t.masked.values for t in reversed(history) if t.masked is not None), None)
    # Every member name the question writes is withheld from the AI, quoted or not; only the quoted ones
    # (``matches``) are offered to it as names to filter on. A person's name or record number is withheld in every
    # workspace: one the question points at on the answer on screen ("the first one"), or one a plan of the
    # conversation filters on, makes the whole question go to the AI with placeholders.
    people = any(about_people(model, m.attribute) for m in [*matches, *(hidden or [])]) or any(
        t.plan is not None and any(about_people(model, f.get("field")) and f.get("values") for f in _member_filters(
            t.plan.model_dump(mode="json", exclude_defaults=True), model)) for t in history)
    masked = None if values_allowed and not people else mask(question, [*matches, *(hidden or [])], known=known)
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
        if attempt == 0 and plan.kind == "clarify" and plan.clarify is not None and plan.clarify.about == "member" \
                and matches and len({m.text.casefold() for m in matches}) == len(matches):
            # "How many items do we hold from the first supplier on that list?" was asked back ("which supplier
            # do you mean?") with the row it points at in VALUE MATCHES: the question is planned for it.
            # Named as VALUE MATCHES names it: by its placeholder where values are withheld.
            problems, hint = ["The question names the member it means: " + "; ".join(
                f'"{placeholder(m.value, masked.values) if masked else m.text}" is in VALUE MATCHES ({m.attribute})'
                for m in matches) + ". Plan the question for it, filtering on that value, rather than asking which "
                "one."], ""
            continue
        previous = before
        if plan.kind == "clarify" and plan.clarify is not None and previous is not None \
                and previous.kind == "query" and _asks_which_way(plan.clarify):
            # "What about the answer above: change the time period, add a breakdown ...?" (asked of a French reader,
            # in English): whether a question follows the answer above is decided, never asked.
            problems, hint = ["Do not ask whether or how the question is about the answer above: decide as rule 11 "
                              "says (a follow-up keeps the previous plan's filters and time; a question about "
                              "something else is new) and return the plan."], ""
            continue
        if attempt == 0 and plan.kind == "clarify" and plan.clarify is not None and plan.clarify.about == "member" \
                and on_screen and previous is not None and previous.kind == "query" and previous.limit \
                and _THOSE.search(question) and str(on_screen[0][0]) in model.attributes:
            # "What was their sales quota for 2013?" after the one sales rep who sold most was asked back ("which
            # employee?"): the members on screen are the ones meant, and the plan is kept to them in code.
            problems, hint = [f"The question means the members of the answer above (by {on_screen[0][0]}): Solviz "
                              'keeps the plan to them. Plan the question with follow_up "refine" and no filter on '
                              "them, rather than asking which one."], ""
            continue
        if attempt == 0 and plan.kind == "clarify" and plan.clarify is not None and plan.clarify.about == "measure" \
                and reading != "new" and previous is not None and previous.kind == "query" and previous.measures \
                and not _measures_named(model, question):
            # "Which division grew the most compared with H1 2025?" right after revenue was asked back "which
            # measure?": a question that names no figure is about the one on screen.
            problems, hint = [f"The question names no figure: it is about the figure of the answer above "
                              f"({', '.join(previous.measures)}). Plan it with those measures and follow_up "
                              '"refine", rather than asking which measure.'], ""
            continue
        spelled = _named_in_full(model, question) if attempt == 0 and plan.kind == "clarify" and \
            plan.clarify is not None and plan.clarify.about == "measure" else None
        if spelled is not None:
            # "Scrapped quantity by month" was asked back, scrapped or rejected quantity? (the sector stress test):
            # a question that names one figure by its whole name means it.
            problems, hint = [f'The question names the measure {spelled.slug} by its own name '
                              f'("{spelled.business_name}"): plan it with that measure rather than asking which one.'], ""
            continue
        refused = plan.kind in ("unsupported", "clarify") and (plan.clarify is None or plan.clarify.about != "member")
        reuse = _derived_named(question, defined or []) if attempt == 0 and refused else []
        if reuse:
            # "Show gross profit by month for 2013" after the answer that worked gross profit out: refused as not in
            # the catalog. The figure the answer above made is made again.
            problems, hint = ["Earlier in this conversation, Solviz worked out " + "; ".join(
                f'"{d.name}" as {json.dumps(d.model_dump(exclude_defaults=True))}' for d in reuse) +
                ". Use it again as a derived measure (derived) for this question."], ""
            continue
        compared = _members_compared(question, written or [], model) if attempt == 0 and refused else None
        if compared is not None:
            # "Compare actual and budget for 2011" was refused as having no measure for budget: two members of one
            # field, compared by grouping by it. Their names are said only where values may reach the AI.
            slug, said = compared
            what = " and ".join(f'"{w}"' for w in said) if values_allowed else f"{len(said)} words of the question"
            problems, hint = [f"{what} are members of the field {slug}, not measures: compare them by grouping the "
                              f"measure by {slug}."], ""
            continue
        if (plan.kind == "unsupported" and _VAGUE.search(" ".join(plan.notes)) or plan.kind == "describe_data"
                and not plan.about and not _ABOUT_THE_DATA.search(question)) and main_figures(model, question):
            # "How are we doing?" was refused as too vague (the sector stress test): a question that names no
            # figure is asked back, with the data's main figures to pick from.
            plan = Plan(kind="clarify", clarify=Clarify(about="other", question=_WHICH_FIGURES,
                                                        options=main_figures(model, question)))
        plan = fill_in(plan, model, before if history and reading is None else None,
                        '"follow_up"' in text, question)
        plan = carried(plan, before, reading)
        plan = late_asked(plan, question)
        plan = pointed_in(plan, model, matches, question, on_screen)
        plan = ratio_alone(plan, model, question)
        halved = why_periods(plan, model, question, [])       # "the first to the second half of FY2026"
        if halved is not plan and not check(halved, model, today, shown)[0]:
            plan = halved
        snapped = snapshot_date(plan, model, question)         # "in each snapshot"
        if snapped is not plan and not check(snapped, model, today, shown)[0]:
            plan = snapped
        for rule in (route_named, all_time_asked, dates_as_window, where_we_have, counted_each, counted_per_itself, members_asked,
                     counted_where_done, role_counted, documents_counted, in_time_order, average_asked, date_in_a_name,
                     list_date_unasked, each_named, named_not_coded, one_time_axis, restated_rates, margin_over_sales):
            # 2024 as a date filter, "where do we have", "$1,000 each", "patients with more than 10 fills", "which
            # techs had fewer than 3", "did we fill", "how many managers", "movement documents", Q1 vs Q2, "average
            # subtotal", "by ship method" (no ship date), "bought in Q2" (not by the customer's own first invoice
            # date), "in A and B" (each), a code's description, "by month" planned twice over, a margin % as sales
            # over cost
            ruled = rule(plan, model, question)
            if ruled is not plan and not check(ruled, model, today, shown)[0]:
                plan = ruled
        problems, hint = check(plan, model, today, shown)
        if not problems:
            twin = named_twin(plan, model, question, before)
            if twin is not plan and not check(twin, model, today, shown)[0]:
                plan = twin
            if reading in ("refine", "new") and plan.kind == "query":
                plan = plan.model_copy(update={"follow_up": reading})
            if plan.kind == "query" and plan.follow_up == "unsure":
                # Could be either: a question about the same records with no scope of its own goes on with the
                # answer above (rule 11), its scope kept and said on the answer, a new question one click away.
                follows = before is not None and before.kind == "query"
                plan = plan.model_copy(update={"follow_up": "refine" if follows else "new"})
            settled = settle(plan, model, question, before)
            if settled is not plan and not check(settled, model, today, shown)[0]:
                plan = settled
            explained = why_periods(plan, model, question, history)
            if explained is not plan and not check(explained, model, today, shown)[0]:
                plan = explained
            # The thread rule: a follow-up keeps the filters from above that it does not replace, a new question
            # none it does not write (core2/plan/thread.py).
            threaded = kept_from_above(plan, model, question, before,
                                       lambda tried_plan: not check(tried_plan, model, today, shown)[0])
            if threaded is not plan and not check(threaded, model, today, shown)[0]:
                plan = threaded
            if plan.follow_up == "refine":
                kept = those_members(plan, model, question, before, on_screen)
                if kept is not plan and not check(kept, model, today, shown)[0]:
                    plan = kept
            alone = one_member_series(plan, model)
            if alone is not plan and not check(alone, model, today, shown)[0]:
                plan = alone
            return Outcome(plan, repaired=attempt > 0, raw=raw, masked=masked, tried=tried)
        tried = plan
    ways = ways_on(tried, model) or main_figures(model, question)
    return Outcome(Plan(kind="clarify", clarify=Clarify(
        about="other", options=ways,
        question=("I could not work out how to answer that as asked. These I can answer" if ways else
                  "I could not work out how to answer that from this data. Could you say it another way, naming "
                  "what to measure and for when?"))), repaired=True, problems=problems, raw=raw, masked=masked,
        tried=tried)


def period_words(window: Window) -> str:
    """A window as a question writes it: "in 2025", "from 1 Mar 2025 to 31 May 2025", "in the last 3 months"."""
    if window.kind == "between" and window.start and window.end:
        if (window.start.month, window.start.day, window.end.month, window.end.day) == (1, 1, 12, 31) \
                and window.start.year == window.end.year:
            return f"in {window.start.year}"
        return f"from {window.start:%-d %b %Y} to {window.end:%-d %b %Y}"
    if window.kind == "last" and window.unit:
        n = window.n or 1
        return f"in the last {n} {window.unit}s" if n > 1 else f"in the last {window.unit}"
    if window.kind == "this" and window.unit:
        return f"this {window.unit}"
    return ""


def main_figures(model: SemanticModel, question: str) -> list[str]:
    """The main figures of the data ("show me the 2025 numbers"): the money, then the count, of its largest
    tables of events, for the year the question names."""
    year = re.search(r"\b(?:19|20)\d{2}\b", question or "")
    period = f" in {year.group()}" if year else ""
    facts = sorted((t for t in model.tables.values() if t.kind == "fact"), key=lambda t: -(t.row_count or 0))
    out: list[str] = []
    for table in facts[:2]:
        own = [m for m in model.measures.values() if m.table == table.key and m.status != "rejected"]
        # What a business reads first: the net or the revenue before the gross, a discount or a tax; the documents
        # (orders) before their lines.
        def first(m) -> tuple:
            words = set(re.findall(r"[a-z]+", m.business_name.lower()))
            return (not words & {"net", "revenue", "sales", "turnover", "income"},
                    bool(words & {"gross", "discount", "tax", "cost", "fee", "refund", "freight", "shipping"}),
                    m.business_name)

        money = sorted((m for m in own if m.format == "currency" and m.additivity == "additive"), key=first)
        counts = sorted((m for m in own if m.format == "count" and m.kind != "model"),
                        key=lambda m: (not (isinstance(m.expr, AggExpr) and m.expr.agg == "count_distinct"),
                                       m.business_name))
        for m in (money[:1] + counts[:1]):
            out.append(f"{m.business_name}{period}")
    return list(dict.fromkeys(out))[:4]


def ways_on(tried: Plan | None, model: SemanticModel) -> list[str]:
    """Questions the data answers that are nearest the one that could not be planned: each figure the planner
    tried, for the period it read, split as it read it -- one click each, never a dead end."""
    if tried is None or tried.kind != "query":
        return []
    from core2.resolve.resolver import find_slug

    period = period_words(tried.time.window)
    split = ""
    for slug in tried.group_by[:1]:
        found = find_slug(model, slug)
        if found is not None and found[0] in ("attribute", "entity"):
            # A field that names a list's members ("Store name") is said as the list ("by store").
            entity = model.entities.get(getattr(found[1], "entity", None) or "") if found[0] == "attribute" else None
            split = f" by {(entity or found[1]).business_name.lower()}"
    out: list[str] = []
    for slug in tried.measures[:3]:
        found = find_slug(model, slug)
        if found is None or found[0] != "measure":
            continue
        name = getattr(found[1], "business_name", "") or slug.replace("_", " ")
        for text in (f"{name}{split} {period}".strip(), f"{name} {period}".strip()):
            if text not in out:
                out.append(text)
    return out[:4]


# What a member of a list did that its own dates record: "customers who signed up in 2025" are counted by the
# customer's signup date, "subscriptions cancelled in 2025" by the subscription's end date -- never by what they
# did on the invoices of that year (a customer invoiced in 2025 may have signed up in 2022).
_STARTED = re.compile(r"\b(?:sign(?:ed|s|ing)?[\s-]*ups?|join(?:ed|s|ing)?|start(?:ed|s|ing)?|began|begin|"
                      r"add(?:ed|s|ing)?|acquir(?:ed|es|e|ing)|gain(?:ed|s)?|won|ajout(?:é|ée|és|ées)|acquis|"
                      r"open(?:ed|s)?|creat(?:ed|es|ion)|regist(?:ered|ers|ration)|enroll?(?:ed|s|ment)?|hired?|"
                      r"onboard(?:ed)?|inscri(?:ts?|tes?|ption)|embauch(?:é|ée|és|ées))\b", re.IGNORECASE)
_ENDED = re.compile(r"\b(?:cancel(?:l?ed|s|l?ation)?|churn(?:ed|s)?|terminat(?:ed|es|ion)|end(?:ed|s)?|clos(?:ed|es)|"
                    r"left|leav(?:e|es|ing)|quit|expir(?:ed|es|y|ation)|lapsed|résili(?:é|ée|és|ées|ation)|"
                    r"annul(?:é|ée|és|ées|ation)|partis?)\b", re.IGNORECASE)
_START_WORDS = frozenset({"start", "signup", "sign", "join", "joined", "created", "creation", "open", "opened",
                          "registration", "registered", "hire", "hired", "enrolled", "enrollment", "begin", "onboard"})
_END_WORDS = frozenset({"end", "ended", "cancel", "cancelled", "canceled", "cancellation", "termination", "terminated",
                        "churn", "churned", "close", "closed", "expiry", "expiration", "leave", "left", "exit"})


def restated_rates(plan: Plan, model: SemanticModel, question: str) -> Plan:
    """ "The return rate for Frozen, as refunds over net sales": a rate whose condition only says again what the
    question filters on is no condition (every row meets it): it is the figure over the other one the plan names."""
    if not plan.rates or not any(_restated(rate, plan) for rate in plan.rates):
        return plan
    kept, made = [], []
    for rate in plan.rates:
        others = [m for m in plan.measures if m != rate.measure]
        if not _restated(rate, plan):
            kept.append(rate)
        elif len(others) == 1:
            made.append(Derived(name=rate.name, op="ratio", measures=[rate.measure, others[0]], scale=100))
    update: dict[str, Any] = {"rates": kept, "derived": [*plan.derived, *made]}
    if made:
        update["measures"] = [m for m in plan.measures if all(m not in d.measures for d in made)]
    return plan.model_copy(update=update)


def _restated(rate: Any, plan: Plan) -> bool:
    """Does every condition of the rate say again a filter of the whole question?"""
    def same(a: Filter, b: Filter) -> bool:
        return a.field == b.field and a.op == b.op and {str(v) for v in a.values} == {str(v) for v in b.values}

    return all(any(same(w, f) for f in plan.filters) for w in rate.where)


def _rate_on_left_out(plan: Plan, model: SemanticModel) -> bool:
    """Does a rate's condition name a column whose rows a table leaves out by default?"""
    from core2.resolve.resolver import find_slug

    left_out = {df.column for t in model.tables.values() for df in t.default_filters}
    for rate in plan.rates:
        for f in rate.where:
            found = find_slug(model, f.field)
            if found is not None and getattr(found[1], "column", None) in left_out:
                return True
    return False


_SPANS = {"month": 1, "quarter": 3, "half": 6, "year": 12}


def _one_period(window: Window, unit: str | None) -> bool:
    """Is ``window`` exactly one ``unit`` (a quarter: Q1 2026, "last quarter"; a month, a half, a year)?"""
    if unit is None:
        return False
    if window.kind in ("this", "last", "previous") and window.unit == unit and (window.n or 1) == 1:
        return True
    if window.kind != "between" or window.start is None or window.end is None or unit not in _SPANS:
        return False
    months = (window.end.year - window.start.year) * 12 + window.end.month - window.start.month + 1
    ends = (window.end + dt.timedelta(days=1)).day == 1
    return window.start.day == 1 and ends and months == _SPANS[unit] and (window.start.month - 1) % months == 0


def _lists_own_date(plan: Plan, model: SemanticModel, question: str) -> str | None:
    """The date of the list a count of its members is about, by the question's words (``_STARTED``,
    ``_ENDED``), when the list has exactly one date of that kind; else None."""
    started, ended = bool(_STARTED.search(question or "")), bool(_ENDED.search(question or ""))
    if started == ended or plan.kind != "query":
        return None
    from core2.resolve.resolver import find_slug

    words = _START_WORDS if started else _END_WORDS
    for slug in plan.measures:
        found = find_slug(model, slug)
        if found is None or found[0] not in ("measure", "count"):
            continue
        table = model.tables.get(getattr(found[1], "table", ""))
        if table is None or table.kind != "dimension":
            continue
        roles = [r for r in model.date_roles_of(table.key) if r.kind != "audit"
                 and set(re.findall(r"[a-z]+", r.name.lower())) & words]
        if len(roles) == 1:
            return roles[0].slug
    return None


# "The most", "the top 10": highest first. "The least", "the bottom 5": lowest first ("at least 5" is neither).
_MOST = re.compile(r"\b(?:most|highest|top|biggest|largest|greatest|maximum)\b", re.IGNORECASE)
_LEAST = re.compile(r"\b(?:(?<!at\s)least|lowest|bottom|fewest|smallest|minimum)\b", re.IGNORECASE)


# "Compare it with 2024", "compare that to last year": the answer on screen, set against another period.
_COMPARE_IT = re.compile(r"\b(?:compare|comparer?)\s+(?:it|that|this|these|them|those|ça|cela)\b", re.IGNORECASE)


def _dated_alike(model: SemanticModel, plan: Plan, previous: Plan) -> bool:
    """Does the date the answer above was counted by fit this plan: the same figures, or a date on the table of
    each of its figures?"""
    from core2.resolve.resolver import find_slug

    if set(plan.measures) <= set(previous.measures) and plan.measures:
        return True
    found = find_slug(model, previous.time.date or "")
    tables = {model.measures[m].table for m in plan.measures if m in model.measures}
    return found is not None and found[0] == "date" and bool(tables) and tables == {found[1].table}


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
    * "the most" sorts highest first and "the least" lowest first ("lost the most" was sorted lowest first);
    * "and by referral source?" splits by that field instead of the one on screen (it was split by gender and
      referral source); "break that down by channel" after a monthly trend splits the whole period by channel,
      unless it asks for each month; "the cost for those" keeps the members on screen and adds the cost (it
      ranked the formulas again by cost); "mostly debits or credits?" sets the two totals side by side, with no
      months kept from a trend before it.
    """
    if plan.kind != "query":
        return plan
    update: dict[str, Any] = {}
    if previous is not None and previous.kind == "query" and plan.follow_up == "refine":
        update.update(_reshaped(plan, previous, question, model))
        plan = plan.model_copy(update=update) if update else plan
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
        elif time.window != previous.time.window and _one_period(previous.time.window, the_period(question)):
            # "Did we hit our target for the quarter?" after Q1 2026: Q1 2026, not the quarter we are in.
            time = time.model_copy(update={"window": previous.time.window})
        if time.date is None and previous.time.date and not keeps_period(question) \
                and _dated_alike(model, plan, previous):
            # "2026 so far" after the customers who signed up each month: those who signed up in 2026. A new period
            # keeps the date the answer above was counted by, unless the AI chose one of its own.
            time = time.model_copy(update={"date": previous.time.date})
        if previous.via and any(k not in plan.via for k in previous.via):
            update["via"] = {**previous.via, **plan.via}
    window = time.window
    # "Stock today", "on the latest day": the latest snapshot, which may be a few days old; never an empty answer
    # because today's has not been loaded yet.
    latest_day = window.kind == "last" and window.unit == "day" and (window.n or 1) == 1 \
        or window.kind == "this" and window.unit == "day"
    if latest_day and plan.measures and all(
            m.slug not in plan.measures or m.additivity == "semi_additive" for m in model.measures.values()) \
            and any(m.slug in plan.measures for m in model.measures.values()):
        time = time.model_copy(update={"window": Window()})
    ranks_members = plan.intent == "rank" and bool(plan.group_by)      # a ranking of periods is its grain
    # "Compare it with 2024" after a monthly series: the series on screen, month by month against 2024's.
    the_series = previous is not None and previous.kind == "query" and previous.time.grain == time.grain \
        and not plan.group_by and _COMPARE_IT.search(question or "") is not None
    if (plan.intent == "compare" or ranks_members) and time.grain is not None and not asks_series(question) \
            and not the_series \
            and not re.search(rf"\b(?:{time.grain.replace('fiscal_', '')}|daily|annual)", question, re.IGNORECASE):
        time = time.model_copy(update={"grain": None})
    if plan.rates and not plan.include_left_out and _rate_on_left_out(plan, model):
        # "Cancellation rate": the rows a table leaves out by default (cancelled ones) are what it counts.
        update["include_left_out"] = True
    own = _lists_own_date(plan, model, question) if not time.date and time.window.kind != "all" else None
    if own is not None and plan.lifecycle is not None and plan.lifecycle.status == "new":
        # "How many new customers did we add in 2023?": the customers created then, by their own date -- not
        # those whose first order fell then (every customer is "new" in the data's first month).
        update["lifecycle"] = None
        if not asks_series(question):
            time = time.model_copy(update={"grain": None})
    if own is not None:
        time = time.model_copy(update={"date": own})
    if time != plan.time:
        update["time"] = time
    if plan.sort and plan.sort[0].by not in ("change", "pct_change", "share", "period"):
        most, least = bool(_MOST.search(question)), bool(_LEAST.search(question))
        if most != least and plan.sort[0].desc != most:
            update["sort"] = [Sort(by=plan.sort[0].by, desc=most), *plan.sort[1:]]
    return plan.model_copy(update=update) if update else plan


# "And by referral source?", "what about by region", "now by store": the same answer split another way.
_AND_BY = re.compile(r"^\s*(?:and|what about|how about|now|then|ok|okay)?[\s,]*(?:split|broken down|broken)?\s*"
                     r"(?:by|per)\s+[^?]+\??\s*$", re.IGNORECASE)
# "Break that down by channel", "split it by region": the same total split by a field.
_BREAK_DOWN = re.compile(r"\b(?:break|split|divide)\s+(?:that|it|this|them|those)\s+(?:down\s+)?(?:by|per|into)\b"
                         r"|\bbreakdown\s+by\b", re.IGNORECASE)
# "The cost for those", "and their margin", "same for them": the members on screen, another figure.
_FOR_THOSE = re.compile(r"\b(?:for|of|on)\s+(?:those|them|these)(?:\s+ones)?\b|\btheir\b", re.IGNORECASE)
# "Mostly debits or credits?", "more online or in store?": two totals set side by side.
_EITHER = re.compile(r"\b(?:mostly|mainly|largely|more|rather)\b[^?]*\bor\b", re.IGNORECASE)
# "What is the average days supply per fill for each?": for each member of the answer on screen.
_FOR_EACH = re.compile(r"\b(?:for|of|in)\s+each(?:\s+(?:one|of\s+them|of\s+those))?\s*\??\s*$|\beach\s+of\s+(?:them|those)\b",
                       re.IGNORECASE)
_SERIES_WORDS = re.compile(r"\b(?:each|every|per|by)\s+(?:day|week|month|quarter|year)\b|\b(?:daily|weekly|monthly|"
                           r"quarterly|yearly|annually|over time|trend)\b", re.IGNORECASE)


def _reshaped(plan: Plan, previous: Plan, question: str, model: SemanticModel | None = None) -> dict[str, Any]:
    """What a follow-up's own words say about the shape of the answer it changes, whatever the AI kept."""
    update: dict[str, Any] = {}
    series = asks_series(question) or bool(_SERIES_WORDS.search(question))
    added = [g for g in plan.group_by if g not in previous.group_by]
    if _AND_BY.match(question) and previous.group_by and added and len(plan.group_by) > len(added):
        update["group_by"] = added                                   # instead of the field on screen
    if _BREAK_DOWN.search(question) and previous.time.grain and not previous.group_by and plan.group_by \
            and plan.time.grain == previous.time.grain and not series:
        update["time"] = plan.time.model_copy(update={"grain": None})
        if plan.intent == "trend":
            update["intent"] = "breakdown"
    if _FOR_THOSE.search(question) and previous.group_by and (previous.limit or previous.sort) \
            and set(plan.group_by) <= set(previous.group_by):
        if series or not keeps_period(question):
            # "Show their monthly quantity sold in 2026": the same members, the figure and the shape it asks
            # (the figure they were ranked by before stays behind, unless the question names it too).
            named = measures_named(model, question) if model is not None else set()
            measures = [m for m in plan.measures if m not in previous.measures or m in named] or plan.measures
            shown = {*measures, *(d.name for d in plan.derived)}
            sort = [s for s in plan.sort if s.by in shown or s.by in ("change", "pct_change", "share", "period")]
            if not sort and measures and not series:
                sort = [Sort(by=measures[0], desc=True)]
            # A series of those members ("their spend by month") runs month by month for each of them: the
            # ranking's limit counted members, never rows of months (it showed 5 months of one supplier); the
            # members themselves are found by name (those_members) or were named by the question.
            limit = previous.limit if not series else None
            if series:
                sort = [s for s in sort if s.by == "period"]
            update.update({"group_by": [*previous.group_by, *(g for g in plan.group_by if g not in previous.group_by)],
                           "sort": sort, "limit": limit, "measures": measures})
        else:
            measures = [*previous.measures, *(m for m in plan.measures if m not in previous.measures)]
            update.update({"group_by": list(previous.group_by), "sort": list(previous.sort), "limit": previous.limit,
                           "measures": measures, "intent": previous.intent})
    if _FOR_EACH.search(question) and previous.group_by and not plan.group_by:
        update["group_by"] = list(previous.group_by)                 # "for each" of the members on screen
    if _EITHER.search(question) and len(plan.measures) + len(plan.derived) >= 2 and plan.time.grain and not series:
        update["time"] = update.get("time", plan.time).model_copy(update={"grain": None})
        if plan.intent == "trend":
            update["intent"] = "value"
    swapped = _another_member(plan, previous, question)
    if swapped is not None:
        update["filters"] = swapped
    return update


# "And contractors?", "what about North?", "et les CDD ?": the same answer for another member of a field the answer
# on screen was narrowed by -- in its place. "Both", "too", "together" ask for the two at once.
_AND_ANOTHER = re.compile(r"^\s*(?:and|what about|how about|same for|et|et pour|et les|et la|et le|pareil pour)\s+"
                          r"(?!by\b|per\b|par\b)[^?]*\??\s*$", re.IGNORECASE)
_BOTH = re.compile(r"\b(?:both|together|combined|as well|too|also|including|plus|in all|in total|added|ensemble|"
                   r"aussi|les deux|cumul\w*|compris|and|et|or|ou)\b|,|&|\+", re.IGNORECASE)


def _another_member(plan: Plan, previous: Plan, question: str) -> list[Filter] | None:
    """The filters of a follow-up that names another member ("and contractors?" after the part-time employees):
    each filter whose members are the ones on screen and more keeps only the new ones; None when there is none.
    The AI had counted part-time and contractors together (101), where the reader asked for contractors (42)."""
    if not _AND_ANOTHER.match(question or "") or _BOTH.search(re.sub(r"^\s*\w+", "", question)) \
            or len(re.findall(r"\w+", question)) > 8:
        return None
    before = {f.field: {str(v) for v in f.values} for f in previous.filters if f.op in ("eq", "in")}
    changed = False
    out: list[Filter] = []
    for f in plan.filters:
        now = [v for v in f.values if str(v) not in before.get(f.field, set())]
        if f.op in ("eq", "in") and f.field in before and before[f.field] < {str(v) for v in f.values} and now:
            out.append(f.model_copy(update={"op": "eq" if len(now) == 1 else "in", "values": now}))
            changed = True
        else:
            out.append(f)
    return out if changed else None


# "Those items", "these stores", "their", "them", "the same customers": the members of the answer on screen.
_THOSE = re.compile(r"\b(?:those|these|them|their|the\s+same)\b", re.IGNORECASE)


def those_members(plan: Plan, model: SemanticModel, question: str, previous: Plan | None,
                  on_screen: list[tuple] | None) -> Plan:
    """The members of the ranking on screen, by name, when a follow-up asks about them in another period or as a
    series: "How did those items do in Q1 2026?" after the top 5 items of Q2 was ranked again in Q1, five other
    items. In the same period and shape, its order and its limit find them again, and are kept as they are. Once
    named, they stay named for the next "their" ("show their monthly quantity")."""
    if plan.kind != "query" or previous is None or previous.kind != "query" or not on_screen \
            or not _THOSE.search(question or ""):
        return plan
    field = str(on_screen[0][0])
    attribute = model.attributes.get(field)
    if attribute is None:
        return plan
    same = {field, *([attribute.entity] if attribute.entity else [])}     # "item.name", or the entity "item"
    if not same & set(previous.group_by) or any(f.field in same for f in plan.filters):
        return plan
    group_by = plan.group_by if same & set(plan.group_by) else [field, *plan.group_by]
    named = next((f for f in previous.filters if f.field in same and f.op in ("in", "eq") and f.values), None)
    if named is not None:                       # named on the answer before: still those
        return plan.model_copy(update={"filters": [*plan.filters, named], "group_by": group_by, "limit": None})
    # In the same period and shape, the same figure's order and limit find them again; another figure ("their
    # sales quota" after the rep who sold most) would rank others.
    if not previous.limit or (plan.time.window, plan.time.grain, plan.time.compare) == (
            previous.time.window, previous.time.grain, previous.time.compare) and (
            set(plan.measures), plan.derived) == (set(previous.measures), previous.derived):
        return plan
    members = list(dict.fromkeys(str(row[1]) for row in on_screen[:previous.limit] if row[1] is not None))
    if not members:
        return plan
    return plan.model_copy(update={"filters": [*plan.filters, Filter(field=field, op="in", values=members)],
                                   "group_by": group_by, "limit": None})


def one_member_series(plan: Plan, model: SemanticModel) -> Plan:
    """A series split by a field the plan keeps to one member of is that member's series. "Booked revenue by month
    for "ARI AURON"", clicked on revenue by doctor, kept the split by doctor: a line per doctor for one doctor,
    headed "ARI AURON led in every month". A ranking, a share or a comparison of members keeps its split."""
    if plan.kind != "query" or plan.time.grain is None or not plan.group_by or plan.shaped() or plan.limit \
            or plan.intent not in (None, "trend", "breakdown", "value", "count", "forecast"):
        return plan
    one = {member_of(model, f.field) for f in plan.filters if f.op in ("eq", "in") and len(f.values) == 1}
    group_by = [g for g in plan.group_by if member_of(model, g) not in one]
    if group_by == plan.group_by:
        return plan
    update: dict[str, Any] = {"group_by": group_by}
    if not group_by and plan.intent == "breakdown":
        update["intent"] = "trend"
    return plan.model_copy(update=update)


def _short(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:6])
    return str(exc)[:300]
