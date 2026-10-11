"""The analyses a workspace's data can give, read off its model: what Learn found that each one needs.

* Members with dated activity (customers and their orders, patients and their fills): time between one member's
  events, new / returning / lapsed members and churn, cohorts by first period.
* Rows with a start and an end (employees hired and terminated, subscriptions started and ended): how many are in
  force on each date, how many started and ended.
* Rows with a date they were expected by and a date they happened (an order line's promised date and its
  receipts): the share on time, and the rows not done yet by days past due.

The planner's worked examples (core2/plan/planner.py) and the questions offered to readers (core2/answer/describe.py,
the learned-data page) are built from the same findings, so a question offered is one the data can answer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from core2.bootstrap import names
from core2.model.schema import DateRole, Entity, Measure, SemanticModel
from core2.resolve import paths as P

May = Callable[[str], bool]


def _all(table: str) -> bool:
    return True


@dataclass
class Activity:
    entity: Entity                 # whose activity (customers)
    measure: Measure               # whose rows are the activity (the number of orders)
    things: str                    # "orders"
    member: str                    # "customer"
    date: DateRole | None          # the date the activity is counted on


# Who comes back, or does not: the parties retention, churn and the time between visits are asked of. An item,
# a store or an employee's payslips have activity too, and "item retention" is no question anyone asks.
# Not "account" (a ledger's accounts) nor "buyer" (a purchasing department's staff): both are as often not parties.
PARTIES = {"customer", "client", "patient", "member", "subscriber", "user", "guest", "student",
           "donor", "policyholder", "shopper", "consumer", "borrower", "tenant", "player", "visitor", "household",
           "cardholder", "passenger", "resident", "citizen", "beneficiary", "insured", "contact", "lead"}


def party(entity: Entity) -> bool:
    """Named as a party: its own word, or a known short form of one (CUST); never a word that only starts like one
    (a cost centre is no customer)."""
    words = names.tokens(entity.business_name or entity.slug)
    return any(names.singular(names.EXPANSIONS.get(w, w)) in PARTIES for w in words)


def activity(model: SemanticModel, may: May = _all) -> Activity | None:
    """The main activity: the largest table with a dated count of its own (orders, fills), and the party with the
    most members its rows reach (customers, patients)."""
    counts = [m for m in model.measures.values() if not m.hidden and m.table in model.tables and may(m.table)
              and m.default_date and (getattr(m.expr, "agg", "") == "count_distinct" and getattr(m.expr, "column", None)
                                      or getattr(m.expr, "agg", "") == "count" and not getattr(m.expr, "column", None)
                                      and len(model.tables[m.table].primary_key or model.tables[m.table].grain) == 1)]
    counts.sort(key=lambda m: (-(model.tables[m.table].row_count or 0), getattr(m.expr, "agg", "") != "count_distinct",
                               m.slug))
    for m in counts:
        entities = sorted((e for e in model.entities.values() if e.table != m.table and may(e.table) and party(e)
                           and len(e.key_columns) == 1 and P.all_paths(model, m.table, e.table)),
                          key=lambda e: (-(e.members or 0), e.slug))
        if entities:
            e = entities[0]
            things = m.business_name.lower().removeprefix("number of ").removeprefix("count of ")
            date = next((r for r in model.date_roles.values() if r.key == m.default_date), None)
            return Activity(e, m, things, e.business_name.lower(), date)
    return None


# Words of a date that ends what another date of the same rows started.
END_WORDS = {"end", "ended", "ending", "termination", "terminated", "terminate", "close", "closed", "closing",
             "leave", "left", "leaving", "exit", "expiry", "expired", "expires", "cancel", "cancelled", "canceled",
             "cancellation", "stop", "stopped", "resolved", "discharge", "discharged", "until", "churn", "churned"}


def start_and_end(model: SemanticModel, may: May = _all) -> tuple[DateRole, DateRole] | None:
    """A list's own start date and the date that ends each member (hire and termination, start and end): the
    default date of a table things are listed on, and a daily date of the same rows named as an end. Not a
    movement's expiry date: a row of activity does not stay in force."""
    found = []
    listed = {e.table for e in model.entities.values()}
    for role in model.date_roles.values():
        if not role.is_default or role.granularity == "month" or role.table not in model.tables or not may(role.table) \
                or role.table not in listed or role.kind not in ("event", "validity"):
            continue
        ends = [r for r in model.date_roles.values() if r.table == role.table and r.key != role.key
                and r.granularity != "month" and set(names.tokens(r.name or r.slug)) & END_WORDS]
        if ends:
            found.append((role, sorted(ends, key=lambda r: r.slug)[0]))
    found.sort(key=lambda pair: (-(model.tables[pair[0].table].row_count or 0), pair[0].slug))
    return found[0] if found else None


def due_and_done(model: SemanticModel, may: May = _all) -> tuple[DateRole, DateRole] | None:
    """A date rows were expected by (promised, due) and the date they happened, on the same rows or on rows that
    point at them (a receipt of an order line)."""
    joined = {(j.from_table, j.to_table) for j in model.joins.values() if P.usable(j)}
    for due in sorted(model.date_roles.values(), key=lambda r: r.slug):
        if due.kind not in ("planned", "due") or due.granularity == "month" or not may(due.table):
            continue
        done = sorted((r for r in model.date_roles.values() if r.kind == "event" and r.key != due.key
                       and r.granularity != "month" and may(r.table)
                       and (r.table == due.table or (r.table, due.table) in joined)),
                      key=lambda r: (r.table == due.table and r.is_default, not r.is_default, r.slug))
        if done:
            return due, done[0]
    return None


def _year(role: DateRole | None, this_year: int) -> int | None:
    return min(role.last.year, this_year) if role is not None and role.last is not None else None


def suggestions(model: SemanticModel, *, this_year: int, may: May = _all) -> list[tuple[str, dict]]:
    """One question for each analysis the data can give, in the reader's words, with the plan that answers it."""
    out: list[tuple[str, dict]] = []

    def year(role: DateRole | None) -> tuple[str, dict]:
        y = _year(role, this_year)
        return (f" in {y}", {"kind": "between", "start": f"{y}-01-01", "end": f"{y}-12-31"}) if y else ("", {})

    act = activity(model, may)
    if act is not None:
        when, window = year(act.date)
        members = names.plural(act.member)
        single = names.singular(act.things)
        of = {"of": act.entity.slug, "measure": act.measure.slug}
        out.append((f"Average days between {act.things} per {act.member}{when}",
                    {"kind": "query", "intent": "value", "gap": of, "time": {"window": window} if window else {}}))
        out.append((f"New, returning and lapsed {members} by month{when}",
                    {"kind": "query", "intent": "trend", "lifecycle": of,
                     "time": {"grain": "month", **({"window": window} if window else {})}}))
        # Every month's newcomers: a year's may hold none (customers all signed up when the shop opened).
        out.append((f"{act.member[:1].upper()}{act.member[1:]} retention by month of first {single}",
                    {"kind": "query", "intent": "breakdown", "cohort": {**of, "periods": 6},
                     "time": {"grain": "month"}}))
    pair = start_and_end(model, may)
    if pair is not None:
        start, end = pair
        rows = names.plural(model.tables[start.table].business_name.lower())
        when, window = year(start)
        out.append((f"{rows[:1].upper()}{rows[1:]} at each month end{when}",
                    {"kind": "query", "intent": "trend", "active": {"start": start.slug, "end": end.slug},
                     "time": {"grain": "month", **({"window": window} if window else {})}}))
    pair = due_and_done(model, may)
    if pair is not None:
        due, done = pair
        rows = names.plural(model.tables[done.table].business_name.lower())
        when, window = year(done)
        out.append((f"Share of {rows} on time{when}",
                    {"kind": "query", "intent": "value", "on_time": {"done": done.slug, "due": due.slug},
                     "time": {"window": window} if window else {}}))
    return out


def questions(model: SemanticModel, *, this_year: int, may: May = _all) -> list[str]:
    """One question for each analysis the data can give, in the reader's words."""
    return [q for q, _ in suggestions(model, this_year=this_year, may=may)]
