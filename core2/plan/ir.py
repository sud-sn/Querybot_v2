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
    """When. ``between`` dates are inclusive, as the user means them."""

    kind: Literal["all", "between", "since", "until", "last", "this", "to_date", "previous"] = "all"
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


class Plan(_Strict):
    kind: Literal["query", "clarify", "describe_data", "unsupported", "smalltalk"] = "query"
    intent: Intent | None = None
    measures: list[str] = Field(default_factory=list)
    derived: list[Derived] = Field(default_factory=list)
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
    # "unsure": it could change the answer on screen or ask afresh; the reader is asked which (core2/plan/followup.py)
    follow_up: Literal["new", "refine", "unsure"] = "new"
    include_left_out: bool = False      # "including cancelled orders": the rows tables leave out by default
    # Analytical shapes (core2/resolve/shapes.py), each optional so every stored plan still reads.
    per: PerMember | None = None
    rolling: Rolling | None = None
    within: Within | None = None
    bands: Bands | None = None
    absent: Absent | None = None
    notes: list[str] = Field(default_factory=list)

    def shaped(self) -> bool:
        """Does the plan ask for an analytical shape (worked out over the plain answer's rows)?"""
        return any(x is not None for x in (self.per, self.rolling, self.within, self.bands, self.absent))


def schema_text() -> str:
    """The plan's JSON schema, for the planner prompt (stable: part of the cached prefix)."""
    import json

    return json.dumps(Plan.model_json_schema(), sort_keys=True, separators=(",", ":"))
