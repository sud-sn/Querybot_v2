"""The typed plan: the only thing the AI writes when answering a question.

A plan names what to compute in the model's own words (slugs from the catalog)
and says nothing about tables, joins or SQL: those are the resolver's and the
compiler's job. It is strict: an unknown field is an error, so a plan that does
not fit this shape is repaired or becomes a clarification, never half-read.
Relative time ("last 6 months") is written relatively and resolved by code.
"""

from __future__ import annotations

import datetime as dt
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field

Grain = Literal["day", "week", "month", "quarter", "year", "fiscal_month", "fiscal_quarter", "fiscal_year"]
Unit = Literal["day", "week", "month", "quarter", "year"]
FilterOp = Literal["in", "not_in", "eq", "ne", "gt", "gte", "lt", "lte", "between", "contains", "starts_with",
                   "is_null", "not_null"]
Intent = Literal["value", "breakdown", "trend", "compare", "rank", "share", "drivers", "list", "count", "forecast"]
TIME_ATTRIBUTES = ("time:day_of_week", "time:month_of_year", "time:quarter_of_year", "time:is_weekend")
# A calendar table's own facts (where Learn found them) and the hour of a date with a time: the workspace's own.
CALENDAR_TIMES = ("time:business_day", "time:holiday", "time:holiday_name", "time:fiscal_week")
HOUR = "time:hour_of_day"
ALL_TIMES = (*TIME_ATTRIBUTES, *CALENDAR_TIMES, HOUR)
YES_NO_TIMES = ("is_weekend", "business_day", "holiday")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Window(_Strict):
    """When. ``between`` dates are inclusive, as the user means them. ``last`` is n complete units before the current
    one ("the last 2 months"); ``rolling`` is n units up to and including today ("the last 2 months from today",
    "the past 60 days")."""

    kind: Literal["all", "between", "since", "until", "last", "rolling", "this", "to_date", "previous"] = "all"
    start: dt.date | None = None
    end: dt.date | None = None
    unit: Unit | None = None
    n: int | None = Field(default=None, ge=1, le=1000)
    fiscal: bool = False
    include_current: bool = False


class Compare(_Strict):
    kind: Literal["previous_period", "same_period_last_year", "window"]
    window: Window | None = None


class TimeSpec(_Strict):
    date: str | None = None          # a date slug; None = the measure's (or its table's) default
    grain: Grain | None = None       # None = one total over the window
    window: Window = Field(default_factory=Window)
    compare: Compare | None = None


Scalar = Union[str, int, float, bool]


class Filter(_Strict):
    field: str                       # an attribute slug, a date slug (a second date), or a measure slug (on totals)
    op: FilterOp
    values: list[Scalar] = Field(default_factory=list)
    total: bool = False              # on a duration: each group's figure (its average), not each row's days


class Sort(_Strict):
    by: str                          # a measure or attribute slug, or change | pct_change | share | period
    desc: bool = True


class Derived(_Strict):
    """A measure the model does not have, made of ones it has (shown as proposed)."""

    name: str
    op: Literal["ratio", "difference", "sum", "product"]
    measures: list[str] = Field(min_length=2, max_length=2)
    scale: float = 1.0               # 100 for a percentage


class Rate(_Strict):
    """The share of a figure whose rows meet a condition, for each group: "cancellation rate by region" is the
    orders with status Cancelled over all orders, in each region (never the share of all cancellations each
    region holds)."""

    name: str                        # "Cancellation rate"
    measure: str                     # the figure: number_of_orders, net_amount
    where: list[Filter] = Field(min_length=1)    # the condition on its rows: status eq ["CANCELLED"]


class Figure(_Strict):
    """A figure no metric holds, worked out for this question: a formula in the admin's formula language
    (core2/model/authoring.py), read and checked in code like a metric an admin writes. "The lowest yield" is
    MIN([Batch · Yield percent]); "average margin per fill" AVG([Fill · Price charged] - [Fill · Ingredient cost]);
    "fills over 100" COUNT(CASE WHEN [Fill · Price charged] > 100 THEN 1 END). Its name is used like a measure's."""

    name: str = Field(min_length=1, max_length=80)
    formula: str = Field(min_length=1, max_length=600)
    format: Literal["number", "integer", "currency", "percent", "count"] | None = None


class Duration(_Strict):
    """Days from one date of the same rows to another: "days from order to invoice"."""

    name: str                        # "Days from order to invoice"
    start: str                       # date slug: the earlier event
    end: str                         # date slug: the later event
    agg: Literal["avg", "min", "max", "sum"] = "avg"
    measure: bool = True             # false when it only limits the rows ("invoiced more than 14 days after")


class Clarify(_Strict):
    about: Literal["measure", "date", "member", "path", "follow_up", "other"]
    question: str
    options: list[str] = Field(default_factory=list)


class Forecast(_Strict):
    periods: int = Field(ge=1, le=36)


class Drivers(_Strict):
    dimensions: list[str] | None = None


class PerMember(_Strict):
    """Each member's figure first, then summarised across the members: "average orders per customer", "median
    spend per customer", "how many customers had 3 or more orders", "90th-percentile days to deliver"."""

    of: str | None = None            # an entity or attribute slug ("customer"), or a count measure's slug for the
    #                                  thing it counts ("number_of_orders": each order); None: each row of the measure
    summary: Literal["avg", "median", "percentile", "min", "max", "count", "share"] = "avg"
    p: float | None = Field(default=None, gt=0, lt=100)    # summary "percentile": 90 for the 90th
    op: Literal["eq", "ne", "gt", "gte", "lt", "lte", "between"] | None = None   # count/share: each member's figure
    values: list[float] = Field(default_factory=list)                            # "3 or more" -> gte [3]
    everyone: bool = False           # every listed member counts, one with no rows as 0 (default: those with rows)


class Rolling(_Strict):
    """A series accumulated or smoothed: "running total", "year to date", "3-month moving average", "last 12
    months" (each point the 12 months up to it)."""

    kind: Literal["running", "moving_average", "moving_sum"]
    periods: int = Field(default=3, ge=2, le=36)               # moving: how many periods each point covers
    reset: Literal["year", "fiscal_year"] | None = None        # running: start again each (fiscal) year


class Within(_Strict):
    """A rank that starts again in each member of a grouping: "rank stores within each region", "top 3 products
    per store"."""

    group: str                       # the grouping slug (one of group_by) each ranking is inside
    top: int | None = Field(default=None, ge=1, le=100)        # keep the first n of each


class Bands(_Strict):
    """Rows or members put into ranges of a value, counted (and totalled) per range: "sales by order size band",
    "patients by age band", "orders by delivery time: under 3 days, 3-7, over a week"."""

    of: str                          # a number attribute slug, a duration's name, or a measure slug (with per)
    edges: list[float] = Field(default_factory=list, max_length=12)  # the reader's own, ascending; [] = from data
    per: str | None = None           # with a measure: whose figure is banded (an entity, or a count measure's slug)


class Absent(_Strict):
    """Members with no matching rows: "customers with no order in the last 90 days", "products never sold",
    "customers who bought A but never B". The window is the plan's time.window (none: never)."""

    of: str                          # an entity slug: whose list
    measure: str                     # a measure slug whose rows are the activity (orders, fills)
    without: list[Filter] = Field(default_factory=list)   # only rows like these count as activity ("never B")
    had: list[Filter] = Field(default_factory=list)       # and they did have rows like these, any time ("bought A")
    active_before: bool = False      # only members with activity before the window (lapsed, not never active)


class Gap(_Strict):
    """Time between one member's events, in date order: "average days between orders per customer", "refill gap",
    "days from a customer's first order to their second", "share of admissions followed by another within 30
    days". Each member's days with an event count once each; the window holds the later event of each gap (the
    first event, with nth)."""

    of: str                          # an entity slug: whose events (customer, patient, machine)
    measure: str                     # a measure slug whose rows are the events (orders, fills, admissions)
    summary: Literal["avg", "median", "min", "max"] = "avg"
    nth: int | None = Field(default=None, ge=2, le=20)     # days from the first event to the nth ("second order": 2)
    within_days: int | None = Field(default=None, ge=1, le=3650)   # share of events with the next within n days


class Lifecycle(_Strict):
    """Members by their activity over a period: the window as one figure ("returning customers in the last 2
    months"), or each week, month, quarter or year of it with time.grain ("new vs returning customers by month").
    New: first activity ever in it; returning: active in it and before it (new and returning make up the active);
    won back: returning, with none in the period before; lapsed: active the period before, not in it; repeat:
    active on 2 or more days in it; one-time: on one day only; loyal: active in every month (week, quarter) of it;
    churn rate: the lapsed over those active the period before."""

    of: str                          # an entity slug: whose activity
    measure: str                     # a measure slug whose rows are the activity
    status: Literal["all", "active", "new", "returning", "won_back", "lapsed", "churn_rate", "repeat", "one_time",
                    "loyal"] = "all"
    name: str = ""                   # a saved segment's name ("Regular shoppers"), said on the answer for them


class Segment(_Strict):
    """Only the members of a segment, or a split by it, for any other figure: "net sales from returning customers by
    store" (status returning), "net sales: new vs returning customers" (status new, split). A member's own activity
    decides it -- all of it, not only the rows the figure counts: new when their first ever was in the period (each
    row's own period in a series), returning when they had some before it; won back, repeat and one-time over the
    question's whole period."""

    of: str                          # an entity slug: whose segment (customer)
    measure: str                     # a measure slug whose rows are their activity (number of orders)
    status: Literal["new", "returning", "won_back", "repeat", "one_time"] = "returning"
    split: bool = False              # a grouping by it (new against returning, repeat against one-time)
    name: str = ""                   # a saved segment's name ("Regular shoppers"), said on the answer for them


class Cohort(_Strict):
    """Members grouped by the period of their first activity (time.grain, a month by default), and how many are
    active in each period after it: "retention by month of first purchase", "subscription retention by start
    month". The window holds the cohorts' first periods."""

    of: str                          # an entity slug
    measure: str                     # a measure slug whose rows are the activity
    periods: int = Field(default=12, ge=1, le=36)          # periods after the first shown
    value: Literal["share", "count"] = "share"


class Active(_Strict):
    """Rows in force on each date, from each row's start and end: "headcount by month", "active subscriptions at
    each month end", "open tickets at month end"; with the starts, the ends and the end rate ("hires and leavers",
    "attrition rate"). An end that is empty or a placeholder far ahead means the row is still in force."""

    start: str                       # date slug: when each row starts (hire date, start date, opened)
    end: str                         # date slug: when it ends (termination, end date, closed)
    status: Literal["all", "active", "starts", "ends", "end_rate"] = "active"


class OnTime(_Strict):
    """Rows against the date they were expected by: "on time %", "deliveries more than 3 days late", "AR aging"
    (rows not done yet, in ranges of days past due today). On time: done on or before the expected date, plus
    grace_days."""

    done: str                        # date slug: when it happened (delivered, paid, received)
    due: str                         # date slug: when it was expected (promised, due)
    grace_days: int = Field(default=0, ge=0, le=365)
    ageing: bool = False             # the rows not done yet, by days past due today (current, 1-30, 31-60, ...)
    late: bool = False               # the share late instead ("later than requested", "delayed", "missed")


class Pareto(_Strict):
    """Who makes most of a figure, each member's figure ranked from the largest: ABC classes ("ABC analysis of
    products by sales": A the members making the first 80% of it, B the next ones up to 95%, C the rest), the
    members making up a share ("which customers make 80% of revenue"), or groups of equal size by the figure
    ("customers by decile of spend", quartiles). A member's class is the one its share starts in: the member that
    takes the total past 80% is still an A."""

    of: str                          # an entity slug: whose figure (product, customer)
    measure: str                     # a measure slug: the figure (net amount)
    mode: Literal["abc", "pareto", "ntile"] = "abc"
    cuts: list[float] = Field(default_factory=lambda: [80.0, 95.0], min_length=1, max_length=3)  # cumulative %
    groups: int = Field(default=10, ge=2, le=20)               # ntile: 10 deciles, 4 quartiles, 5 quintiles
    members: bool = False            # abc / ntile: each member with its class, not one row a class


class Funnel(_Strict):
    """Rows through dates of their own in order ("orders through ship and delivery", "claims submitted, approved,
    paid"): how many reached each stage (a date filled, and every stage before it), the share of the first stage
    and of the stage before, and the average days from the stage before. The window holds the first stage."""

    stages: list[str] = Field(min_length=2, max_length=6)   # date slugs of the same rows, in order


class Together(_Strict):
    """Items bought in the same basket: "products bought together", "what is bought with \"Road Bike\"" (a quoted
    item, as a filter on the items, keeps the pairs that hold it). Each pair: the baskets holding both, their share
    of all baskets, how often a basket with the first holds the second, and the lift (how much more often than by
    chance)."""

    of: str                          # an entity slug: the items (product)
    basket: str                      # a count measure whose distinct value is a basket (number of orders)


class Target(_Strict):
    """A figure against its target, budget or quota ("sales against target by store", "spend vs budget by
    month"): the two figures, the difference and the actual as a % of the target, with the plan's groupings and
    periods."""

    actual: str                      # a measure slug: what happened (net amount, spend)
    target: str                      # a measure slug: what was planned (target amount, budget, quota)


class Plan(_Strict):
    kind: Literal["query", "clarify", "describe_data", "unsupported", "smalltalk"] = "query"
    intent: Intent | None = None
    measures: list[str] = Field(default_factory=list)
    derived: list[Derived] = Field(default_factory=list)
    figures: list[Figure] = Field(default_factory=list)     # worked out for this question, by formula
    rates: list[Rate] = Field(default_factory=list)
    durations: list[Duration] = Field(default_factory=list)
    group_by: list[str] = Field(default_factory=list)       # attribute slugs and time:... attributes
    via: dict[str, str] = Field(default_factory=dict)       # attribute slug -> the entity it is reached through
    filters: list[Filter] = Field(default_factory=list)
    time: TimeSpec = Field(default_factory=TimeSpec)
    sort: list[Sort] = Field(default_factory=list)
    limit: int | None = Field(default=None, ge=1, le=10000)
    forecast: Forecast | None = None
    drivers: Drivers | None = None
    clarify: Clarify | None = None
    about: list[str] = Field(default_factory=list)          # describe_data: what the question asks about
    # "as a pie", "a donut", "stacked", "as a heatmap", "as a scatter", "on a map", "just the table"
    chart: Literal["bar", "line", "area", "pie", "donut", "stacked", "heatmap", "scatter", "map", "table"] | None = None
    # "unsure": the AI could not tell; decided in code (a follow-up when an answer precedes it), never asked of the reader
    follow_up: Literal["new", "refine", "unsure"] = "new"
    include_left_out: bool = False      # "including cancelled orders": the rows tables leave out by default
    segment: Segment | None = None      # only (or split by) new, returning, repeat... members: core2/resolve/segments.py
    # Analytical shapes (core2/resolve/shapes.py), each optional so every stored plan still reads.
    per: PerMember | None = None
    rolling: Rolling | None = None
    within: Within | None = None
    bands: Bands | None = None
    absent: Absent | None = None
    gap: Gap | None = None
    lifecycle: Lifecycle | None = None
    cohort: Cohort | None = None
    active: Active | None = None
    on_time: OnTime | None = None
    pareto: Pareto | None = None
    funnel: Funnel | None = None
    together: Together | None = None
    target: Target | None = None
    notes: list[str] = Field(default_factory=list)

    def shaped(self) -> bool:
        """Does the plan ask for an analytical shape (worked out over the plain answer's rows)?"""
        return any(x is not None for x in (self.per, self.rolling, self.within, self.bands, self.absent, self.gap,
                                           self.lifecycle, self.cohort, self.active, self.on_time, self.pareto,
                                           self.funnel, self.together))


def schema_text() -> str:
    """The plan's JSON schema, for the planner prompt (stable: part of the cached prefix)."""
    import json

    return json.dumps(Plan.model_json_schema(), sort_keys=True, separators=(",", ":"))
