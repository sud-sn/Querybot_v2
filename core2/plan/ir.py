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
    about: Literal["measure", "date", "member", "path", "other"]
    question: str
    options: list[str] = Field(default_factory=list)


class Forecast(_Strict):
    periods: int = Field(ge=1, le=36)


class Drivers(_Strict):
    dimensions: list[str] | None = None


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
    chart: Literal["bar", "line", "area", "pie", "donut", "table"] | None = None   # "as a pie", "a donut", "just the table"
    follow_up: Literal["new", "refine"] = "new"
    include_left_out: bool = False      # "including cancelled orders": the rows tables leave out by default
    notes: list[str] = Field(default_factory=list)


def schema_text() -> str:
    """The plan's JSON schema, for the planner prompt (stable: part of the cached prefix)."""
    import json

    return json.dumps(Plan.model_json_schema(), sort_keys=True, separators=(",", ":"))
