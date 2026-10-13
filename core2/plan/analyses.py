"""The analyses a workspace's data can give, read off its model: what Learn found that each one needs.

* Members with dated activity (customers and their orders, patients and their fills): time between one member's
  events, new / returning / lapsed members and churn, cohorts by first period.
* Rows with a start and an end (employees hired and terminated, subscriptions started and ended): how many are in
  force on each date, how many started and ended.
* Rows with a date they were expected by and a date they happened (an order line's promised date and its
  receipts): the share on time, and the rows not done yet by days past due.
* A figure many members share (net amount over products and customers): ABC classes, who makes the first 80%,
  deciles.
* Rows with dates of their own in order (ordered, shipped, delivered): a funnel through them.
* Items counted on baskets (products on orders): what is bought together.
* A target, budget or quota beside what happened: the figure against it.

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


# A date that says a row was finished or arrived (its end, completion, receipt), and one that says it began.
_ENDED = {"end", "ended", "completed", "completion", "closed", "finished", "done", "received", "receipt", "delivered",
          "delivery", "fulfilled", "resolved", "shipped", "ship", "arrived", "arrival"}
_STARTED = {"start", "started", "begin", "began", "opened", "created"}


def due_and_done(model: SemanticModel, may: May = _all) -> tuple[DateRole, DateRole] | None:
    """A date rows were expected by (promised, due) and the date they happened, on the same rows or on rows that
    point at them (a receipt of an order line)."""
    joined = {(j.from_table, j.to_table) for j in model.joins.values() if P.usable(j)}

    def said(role: DateRole, words: set[str]) -> bool:
        """Does the role's name or column say one of the words, glued or not (ACTUALSTARTDATE, end_date)?"""
        own = set(names.tokens(role.slug)) | set(names.tokens(role.key.rsplit(".", 1)[-1]))
        own |= {w[:-len(d)] for w in list(own) for d in ("date", "dt", "time") if w.endswith(d) and len(w) > len(d)}
        return any(w in words or any(len(x) >= 4 and w.endswith(x) for x in words) for w in own)

    for due in sorted(model.date_roles.values(), key=lambda r: r.slug):
        if due.kind not in ("planned", "due") or due.granularity == "month" or not may(due.table):
            continue
        # A work order due by its due date is done when it ends, never when a step of it starts: a date that says
        # it ended or arrived first (its own rows' before a step's), one that says it started last.
        done = sorted((r for r in model.date_roles.values() if r.kind == "event" and r.key != due.key
                       and r.granularity != "month" and may(r.table)
                       and (r.table == due.table or (r.table, due.table) in joined)),
                      key=lambda r: (said(r, _STARTED), not said(r, _ENDED), r.table != due.table and said(r, _ENDED),
                                     r.table == due.table and r.is_default, not r.is_default, r.slug))
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
    lead = leading(model, may)
    who = ranked_entity(model, lead, may) if lead is not None else None
    if lead is not None and who is not None:
        role = next((r for r in model.date_roles.values() if r.key == lead.default_date), None)
        when, window = year(role)
        many = names.plural(who.business_name.lower())
        out.append((f"{many[:1].upper()}{many[1:]} making 80% of {lead.business_name.lower()}{when}",
                    {"kind": "query", "intent": "rank", "pareto": {"of": who.slug, "measure": lead.slug,
                                                                 "mode": "pareto", "cuts": [80]},
                     "time": {"window": window} if window else {}}))
    stages = funnel_dates(model, may)
    if len(stages) >= 2:
        rows = names.plural(model.tables[stages[0].table].business_name.lower())
        when, window = year(stages[0])
        listed = [r.name.lower() for r in stages]
        said = listed[0] if len(listed) == 1 else ", ".join(listed[:-1]) + f" and {listed[-1]}"
        out.append((f"{rows[:1].upper()}{rows[1:]} through {said}{when}",
                    {"kind": "query", "intent": "breakdown", "funnel": {"stages": [r.slug for r in stages]},
                     "time": {"window": window} if window else {}}))
    found = basket(model, may)
    if found is not None:
        item, count = found
        role = next((r for r in model.date_roles.values() if r.key == count.default_date), None)
        when, window = year(role)
        out.append((f"{names.plural(item.business_name)} bought together{when}",
                    {"kind": "query", "intent": "rank", "together": {"of": item.slug, "basket": count.slug},
                     "limit": 10, "time": {"window": window} if window else {}}))
    pair_t = target_pair(model, may)
    if pair_t is not None:
        actual, target = pair_t
        role = next((r for r in model.date_roles.values() if r.key == actual.default_date), None)
        # The last whole year: targets are set ahead, so this year's run past the figures there are yet.
        y = _year(role, this_year)
        y = y - 1 if y is not None and role is not None and role.last is not None and role.last.year == y \
            and role.last.month < 12 else y
        when, window = (f" in {y}", {"kind": "between", "start": f"{y}-01-01", "end": f"{y}-12-31"}) if y else ("", {})
        out.append((f"{actual.business_name} against {target.business_name.lower()} by month{when}",
                    {"kind": "query", "intent": "trend", "target": {"actual": actual.slug, "target": target.slug},
                     "time": {"grain": "month", **({"window": window} if window else {})}}))
    return out


def questions(model: SemanticModel, *, this_year: int, may: May = _all) -> list[str]:
    """One question for each analysis the data can give, in the reader's words."""
    return [q for q, _ in suggestions(model, this_year=this_year, may=may)]


def leading(model: SemanticModel, may: May = _all) -> Measure | None:
    """The figure the business is measured by most: its recipe's first amount that adds up (net amount before cost
    amount), else the busiest table's."""
    from core2 import recipes

    leads = [m for m in recipes.headlines(model, recipes.kind_of(model, may), may)
             if getattr(m.expr, "agg", "") == "sum" and m.additivity == "additive"]
    return leads[0] if leads else None


def ranked_entity(model: SemanticModel, m: Measure, may: May = _all) -> Entity | None:
    """Whose share of ``m`` is asked: a party first (customers), else the listed thing with most members; never one
    named by a person's name (employees by their pay, patients by their copay)."""
    found = []
    for e in model.entities.values():
        label = model.columns.get(e.label_column or "")
        if e.table == m.table or not may(e.table) or e.role or (e.members or 0) <= 10 or label is None \
                or label.personal != "none" or not P.all_paths(model, m.table, e.table):
            continue
        found.append(e)
    found.sort(key=lambda e: (not party(e), -(e.members or 0), e.slug))
    return found[0] if found else None


# The order a document's dates come in, by the words they are named with; a date named otherwise sits by when
# its data starts.
STAGE_WORDS = [{"created", "opened", "ordered", "order", "booked", "booking", "submitted", "placed", "requested",
                "raised", "logged", "written", "prescribed"},
               {"acknowledged", "approved", "accepted", "confirmed", "assigned", "scheduled", "picked", "packed"},
               {"shipped", "ship", "dispatched", "sent", "filled", "fill", "service", "started", "invoiced"},
               {"delivered", "received", "resolved", "completed", "closed", "paid", "settled", "posted"}]


def _stage(role: DateRole) -> int | None:
    words = set(names.tokens(role.name or role.slug))
    return next((i for i, group in enumerate(STAGE_WORDS) if words & group), None)


# A row's end, never a step of a funnel (a ticket resolved or an order closed is).
_ENDINGS = END_WORDS - {"resolved", "close", "closed", "closing", "stop", "stopped"}


def funnel_dates(model: SemanticModel, may: May = _all) -> list[DateRole]:
    """Up to three daily dates of the same rows that are steps of one process (ordered, shipped, delivered), in
    the order they come: the busiest table with two such steps. Never a row's end (a termination, a contract's
    end: in force, not a funnel), nor a date the rows were loaded."""
    by_table: dict[str, list[DateRole]] = {}
    for r in model.date_roles.values():
        words = set(names.tokens(r.name or r.slug))
        if r.kind == "event" and r.granularity != "month" and r.first is not None and r.table in model.tables \
                and may(r.table) and _stage(r) is not None and not words & _ENDINGS:
            by_table.setdefault(r.table, []).append(r)
    tables = sorted((t for t, rs in by_table.items() if len({_stage(r) for r in rs}) >= 2),
                    key=lambda t: (-(model.tables[t].row_count or 0), t))
    if not tables:
        return []
    dates = sorted(by_table[tables[0]], key=lambda r: (_stage(r), r.first, r.slug))
    kept: list[DateRole] = []
    for r in dates:                              # one date a step
        if not kept or _stage(r) != _stage(kept[-1]):
            kept.append(r)
    return kept[:3]


# What a basket is (an order, an invoice, a visit), and what is bought on one (products, items, drugs).
BASKET_WORDS = {"order", "invoice", "receipt", "basket", "cart", "transaction", "sale", "booking", "visit", "po",
                "purchase", "prescription", "ticket", "check", "bill"}
ITEM_WORDS = {"product", "item", "sku", "article", "part", "drug", "medication", "formula", "ingredient",
              "service", "menu", "dish", "book", "track", "album", "film", "course", "material", "variant", "style"}


def basket(model: SemanticModel, may: May = _all) -> tuple[Entity, Measure] | None:
    """Items counted on baskets: a distinct count of a document's number on the lines (orders on order lines), and
    the things bought on them (products, items, drugs)."""
    for m in sorted(model.measures.values(), key=lambda m: (-(model.tables[m.table].row_count or 0)
                                                           if m.table in model.tables else 0, m.slug)):
        table = model.tables.get(m.table)
        if m.hidden or table is None or not may(m.table) or getattr(m.expr, "agg", "") != "count_distinct" \
                or not getattr(m.expr, "column", None) or not set(_words(m.business_name)) & BASKET_WORDS:
            continue
        items = [e for e in model.entities.values() if e.table != m.table and may(e.table) and not e.role
                 and set(_words(e.business_name or e.slug)) & ITEM_WORDS and (e.members or 0) > 10
                 and e.label_column and len(P.all_paths(model, m.table, e.table)) == 1]
        items.sort(key=lambda e: (-(e.members or 0), e.slug))
        if items:
            return items[0], m
    return None


def _words(name: str) -> list[str]:
    return [names.singular(names.EXPANSIONS.get(w, w)) for w in names.tokens(name or "")]


TARGET_WORDS = {"target", "budget", "quota", "planned", "forecast", "goal"}


def target_pair(model: SemanticModel, may: May = _all) -> tuple[Measure, Measure] | None:
    """A target, budget or quota, and the figure it is set for: an amount that adds up, on another table, in one
    unit, sharing the most words with it (sales amount for a sales amount quota), the business's own lead first."""
    from core2 import recipes

    leads = [m.slug for m in recipes.headlines(model, recipes.kind_of(model, may), may)]
    visible = [m for m in model.measures.values() if not m.hidden and may(m.table) and m.table in model.tables
               and getattr(m.expr, "agg", "") == "sum" and m.additivity == "additive"
               and not (m.unit_column and not m.unit)]
    for t in sorted(visible, key=lambda m: m.slug):
        words = set(_words(t.business_name))
        if not words & TARGET_WORDS:
            continue
        rest = words - TARGET_WORDS
        actuals = [a for a in visible if a.table != t.table and a.format == t.format and a.default_date
                   and not set(_words(a.business_name)) & TARGET_WORDS]
        actuals.sort(key=lambda a: (-len(set(_words(a.business_name)) & rest),
                                    leads.index(a.slug) if a.slug in leads else len(leads),
                                    -(model.tables[a.table].row_count or 0), a.slug))
        if actuals:
            return actuals[0], t
    return None
