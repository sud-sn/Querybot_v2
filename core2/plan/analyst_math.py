"""The arithmetic an analyst reaches for, read from the question's own words where the AI's plan missed it.

Each rule takes a checked plan and returns it, or a better one the planner checks again before keeping
(core2/plan/planner.py): never a plan the resolver would refuse.

* A share of each group's own total ("what % of each region's sales came from each category"): ``share_of`` the
  group the question names, never the grand total. A "rate" whose only condition is on a field the answer is
  grouped by is such a share: every row of each group meets it, and each group's rate was 100%.
* Only the groups that fell or rose ("which products sold less in Q3 than in Q2", "the stores whose sales fell",
  "formulas with more than 5% fewer fills"): a filter on each group's change between the compared periods, or on
  its % change, beside the figures -- never every group, the ones that rose among them.
* A condition on the total of a document ("orders over $200", "baskets above $75", "invoices of more than
  $5,000"): each document's own total, so the answer is grouped by its number -- never the total of everything a
  customer or a product had, kept above the amount.
"""

from __future__ import annotations

import functools
import logging
import re
from collections.abc import Callable

from core2.bootstrap import names
from core2.model.schema import SemanticModel
from core2.plan.ir import Filter, Plan

log = logging.getLogger(__name__)


def _kept_safe(rule: Callable[[Plan, SemanticModel, str], Plan]) -> Callable[[Plan, SemanticModel, str], Plan]:
    """A rule that cannot read the question leaves the plan as the AI wrote it, and says so in the log."""

    @functools.wraps(rule)
    def safe(plan: Plan, model: SemanticModel, question: str) -> Plan:
        try:
            return rule(plan, model, question)
        except Exception as exc:  # noqa: BLE001 - a rule is an improvement; the plan stands without it
            log.warning("core2 rule %s left the plan as written: %s", rule.__name__, exc)
            return plan

    return safe


# "of each region's", "within each therapeutic class", "of its store's", "of that payer's", "of the region total".
_WITHIN = re.compile(r"\b(?:of|within|in)\s+(?:each|every|its|their|that|the|this|a)\s+"
                     r"([a-z][a-z'’-]*(?:\s+[a-z][a-z'’-]*){0,2})", re.IGNORECASE)
# A share said outright: "% of each region's sales", "share of", "what proportion of", "percentage of".
_SHARE_OF = re.compile(r"(?:%|\bper\s?cent(?:age)?|\bshare|\bproportion|\bfraction)\s+(?:of|from)\b",
                       re.IGNORECASE)
_SHARE_WORDS = re.compile(r"%|\bper\s?cent|\bpercentage\b|\bshare\b|\bproportion\b|\bpart\b|\bmix\b|\bsplit\b",
                          re.IGNORECASE)


def _words(text: str) -> list[str]:
    return [names.singular(w) for w in re.findall(r"[a-z0-9]+", (text or "").casefold())]


def _group_names(model: SemanticModel, slug: str) -> list[list[str]]:
    """The ways a reader names a grouping: its entity's or attribute's name, each as singular words."""
    found: list[str] = []
    if slug in model.entities:
        found.append(model.entities[slug].business_name)
    attribute = model.attributes.get(slug)
    if attribute is not None:
        found.append(attribute.business_name)
        owner = next((e for e in model.entities.values() if e.table == model.columns[attribute.column].table), None)
        if owner is not None and attribute.business_name.casefold().endswith(" name"):
            found.append(owner.business_name)
    found.append(slug.split(".")[-1].replace("_", " "))
    if "." in slug:
        found.append(slug.split(".")[0].replace("_", " "))
    return [w for w in (_words(n) for n in found) if w]


def _named_group(plan: Plan, model: SemanticModel, said: str) -> str | None:
    """The grouping of the plan the first words name ("region's revenue" -> region, "therapeutic class"), or
    None. A possessive ends the name ("store's front-store sales" is the store)."""
    words: list[str] = []
    for raw in said.split():
        words += _words(re.sub(r"['’]s?$", "", raw))
        if re.search(r"['’]s?$", raw):
            break
    for n in range(len(words), 0, -1):
        lead = words[:n]
        for slug in plan.group_by:
            for name in _group_names(model, slug):
                if lead == name or lead[-1:] == name[-1:] and (n == 1 or lead[:1] == name[:1]):
                    return slug
    return None


@_kept_safe
def share_in_group(plan: Plan, model: SemanticModel, question: str) -> Plan:
    """ "What percentage of each region's revenue came from each category" divided each region and category by the
    grand total, or was planned as a "rate" whose condition is every category (100% each): each share is of the
    region's own total, the grouping the question names."""
    if plan.kind != "query" or plan.shaped() or len(plan.group_by) < 1 or plan.time.compare is not None:
        return plan
    grouped = set(plan.group_by)
    fake = [r for r in plan.rates if r.where and all(f.field in grouped for f in r.where)]
    if not fake and (plan.intent != "share" and not _SHARE_OF.search(question or "")):
        return plan                # "net sales split by store in each region", "margin % by region": no share
    within = None
    if len(plan.group_by) >= 2:
        for hit in _WITHIN.finditer(question or ""):
            within = _named_group(plan, model, hit.group(1))
            if within is not None:
                break
        if within is None and fake:
            # The rate's condition is the grouping each share is OF ("by category" in "% of each region's
            # revenue by category"): the share is within the other one.
            on = {f.field for r in fake for f in r.where}
            rest = [g for g in plan.group_by if g not in on]
            within = rest[0] if len(rest) == 1 else None
    if within is None and not fake:
        return plan
    if within is not None and within == plan.share_of and plan.intent == "share" and not fake:
        return plan
    ratio = _two_figures(model, question, fake[0].measure) if len(fake) == 1 else None
    if ratio is not None:
        # "Refunds as a percentage of net sales": the first figure over the second, each counted by its own
        # date, the "rate" on the grouping read as the ratio of the two figures it named.
        part, whole = ratio
        from core2.plan.ir import Derived

        named = fake[0].name
        derived = [*plan.derived, Derived(name=named, op="ratio", measures=[part, whole], scale=100.0)]
        return plan.model_copy(update={
            "rates": [r for r in plan.rates if r not in fake], "derived": derived,
            "measures": list(dict.fromkeys([*plan.measures, part, whole])),
            "intent": "breakdown" if plan.intent in (None, "share", "value") and plan.group_by else plan.intent,
            "asked": [a.model_copy(update={"plan": named}) if a.plan and a.plan.casefold() == named.casefold() else a
                      for a in plan.asked]})
    rates = [r for r in plan.rates if r not in fake]
    # The figure each share is of comes first: a share is worked out on the first measure shown.
    measures = list(dict.fromkeys([*(r.measure for r in fake), *plan.measures]))
    if not measures or plan.derived and not plan.measures:
        return plan
    names_of_fake = {r.name.casefold() for r in fake}
    asked = [a.model_copy(update={"plan": "share"}) if a.plan and a.plan.casefold() in names_of_fake else a
             for a in plan.asked]
    sort = [s for s in plan.sort if s.by.casefold() not in names_of_fake]
    return plan.model_copy(update={"intent": "share", "rates": rates, "measures": measures, "asked": asked,
                                   "share_of": within, "sort": sort})


_OF_WHOLE = re.compile(r"(?:%|\bper\s?cent(?:age)?\b|\bproportion\b|\bshare\b|\bratio\b|\brate\b|\bfraction\b)"
                       r"\s+(?:of|to|against|over)\s+", re.IGNORECASE)


def _two_figures(model: SemanticModel, question: str, whole_hint: str) -> tuple[str, str] | None:
    """The two figures a question divides ("refunds as a percentage of net sales", "shipping cost as a % of
    revenue"): (part, whole), the whole named right after "% of", the part the figure named last before it;
    None unless it names both."""
    from core2.plan.followup import measures_in_order

    text = question or ""
    hit = _OF_WHOLE.search(text)
    if hit is None:
        return None
    found = measures_in_order(model, text)          # word positions, as the question's words are counted
    at = len(re.findall(r"[A-Za-z][A-Za-z0-9]*", text[:hit.start()]))
    after = len(re.findall(r"[A-Za-z][A-Za-z0-9]*", text[:hit.end()]))
    whole = next((slug for i, slug in found if i >= after), None)
    before = [slug for i, slug in found if i < at and slug != whole]
    if whole is None or not before:
        return None
    return before[-1], whole


# Groups kept by their change: the verb of a clause about them ("that fell", "whose sales dropped"), or a
# comparison of their figure across the periods ("sold less in Q3 than in Q2", "spent more in 2025 than 2024").
_DOWN = r"(?:fell|dropped|declined|decreased|shrank|slipped|slumped|went\s+down|got\s+worse|lost\s+ground)"
_UP = r"(?:grew|rose|increased|went\s+up|improved|gained)"
_ACT = (r"(?:sold|spent|bought|ordered|earned|made|filled|wrote|originated|generated|brought\s+in|did|had|"
        r"received|shipped|dispensed|booked|deposited|lent|billed)")
_CLAUSE = r"\b(?:that|which|who|whose(?:\s+[a-z-]+){1,3})\s+(?:(?:have|has|had|was|were)\s+)?"
_FELL = re.compile(_CLAUSE + _DOWN + r"\b|\b" + _ACT + r"\s+(?:less|fewer|lower)\b(?!\s+than\s+[$€£]?\d)|"
                   r"\b(?:less|fewer|lower)\b[^.?!]{0,60}\bthan\s+(?:in\s+|the\s+|during\s+|for\s+)?"
                   r"(?:q[1-4]|h[12]|20\d\d|last|previous|prior|before|[a-z]+\s+20\d\d)", re.IGNORECASE)
_ROSE = re.compile(_CLAUSE + _UP + r"\b|\b" + _ACT + r"\s+more\b(?!\s+than\s+[$€£]?\d)", re.IGNORECASE)
_RANKED = re.compile(r"\b(?:the\s+)?(?:most|least|fastest|slowest|biggest|largest|smallest|steepest|sharpest)\b",
                     re.IGNORECASE)
# "more than 5% fewer fills", "down more than 10%", "fell by over 20 percent", "grew by more than 15%".
_PCT_DOWN = re.compile(r"\b(?:more|over)\s+than\s+(\d+(?:\.\d+)?)\s*(?:%|percent|per\s?cent)\s+(?:fewer|less|lower|"
                       r"down|below)\b|\b(?:down|fell|dropped|declined|decreased)\s+(?:by\s+)?(?:more\s+than|over)\s+"
                       r"(\d+(?:\.\d+)?)\s*(?:%|percent|per\s?cent)", re.IGNORECASE)
_PCT_UP = re.compile(r"\b(?:more|over)\s+than\s+(\d+(?:\.\d+)?)\s*(?:%|percent|per\s?cent)\s+(?:more|higher|up|"
                     r"above)\b|\b(?:up|grew|rose|increased)\s+(?:by\s+)?(?:more\s+than|over)\s+(\d+(?:\.\d+)?)\s*"
                     r"(?:%|percent|per\s?cent)", re.IGNORECASE)


@_kept_safe
def kept_by_change(plan: Plan, model: SemanticModel, question: str) -> Plan:
    """ "Which products sold less in Q3 2026 than in Q2 2026, and by how much?" listed every product, the ones that
    sold more with them: the groups the question keeps are those whose change has that sign (or whose % change
    passes the number it names)."""
    text = question or ""
    # One figure shown: the change kept is its change ("which stores had more returns", with revenue beside it,
    # kept the stores whose revenue rose).
    if plan.kind != "query" or plan.time.compare is None or not plan.group_by or plan.shaped() \
            or any(f.field in ("change", "pct_change") for f in plan.filters) \
            or len(plan.measures) + len(plan.derived) + len(plan.rates) + len(plan.figures) != 1:
        return plan
    keep: Filter | None = None
    down, up = _PCT_DOWN.search(text), _PCT_UP.search(text)
    if down and not up:
        keep = Filter(field="pct_change", op="lt", values=[-float(down.group(1) or down.group(2))])
    elif up and not down:
        keep = Filter(field="pct_change", op="gt", values=[float(up.group(1) or up.group(2))])
    else:
        fell, rose = _FELL.search(text), _ROSE.search(text)
        if bool(fell) == bool(rose):
            return plan
        hit = fell or rose
        if _RANKED.match(text, hit.end()) or _RANKED.search(text[hit.end():hit.end() + 12]):
            return plan                  # "which store grew the most": a ranking, every store in it
        keep = Filter(field="change", op="lt" if fell else "gt", values=[0])
    sort = list(plan.sort)
    if not sort:
        from core2.plan.ir import Sort

        sort = [Sort(by="change", desc=keep.op == "gt")]
    return plan.model_copy(update={"filters": [*plan.filters, keep], "sort": sort,
                                   "intent": plan.intent or "compare"})


_MOVE = r"(?:change|drop|growth|increase|decline|decrease|rise|fall|gain|loss|jump|swing|shortfall|uplift)"
# "how much of the drop came from each product family", "how much of the chain's change came from each region",
# "each region's % of the total change", "share of the increase", "contribution to the growth", "break the change
# down by fee type".
_OF_A_CHANGE = re.compile(
    r"\bhow\s+much\s+of\s+(?:[a-z'’$-]+\s+){0,4}?" + _MOVE + r"\b.{0,90}?\b(?:came|come|comes|coming|was|were|is|"
    r"are|due|from)\b|(?:%|\bper\s?cent(?:age)?|\bshare|\bproportion|\bpart)\s+of\s+(?:the\s+|that\s+|this\s+)?"
    r"(?:total\s+|overall\s+|whole\s+|[a-z'’-]+'s\s+)?" + _MOVE + r"\b|\bcontribut\w*\s+to\s+(?:the\s+|that\s+)?"
    r"(?:total\s+|overall\s+)?" + _MOVE + r"\b|\bbreak\s+(?:the|that|this)\s+" + _MOVE + r"\s+down\b|"
    r"\bbreak\s+down\s+(?:the|that|this)\s+" + _MOVE + r"\b", re.IGNORECASE)
# A question about why: its drivers answer looks for where the change sits, among the groupings a reader checks first.
_WHY = re.compile(r"\bwhy\b|\bwhat\s+(?:drove|drives|is\s+driving|was\s+driving|caused|explains?)\b|\bexplain\b|"
                  r"\breasons?\s+for\b", re.IGNORECASE)


@_kept_safe
def share_of_change(plan: Plan, model: SemanticModel, question: str) -> Plan:
    """ "Total originations fell from 2024 to 2025 - how much of the drop came from each product family?" was
    answered with each family's own % change: the question asks for each family's part of the drop. A comparison
    by the grouping the question names, with each group's share of the whole change beside its change; planned as
    a "why" looking only at that grouping, it is that comparison. A question about why keeps its drivers answer."""
    text = question or ""
    if plan.kind != "query" or plan.share_of_change or plan.time.compare is None or plan.shaped() \
            or not _OF_A_CHANGE.search(text) or _WHY.search(text):
        return plan
    if plan.intent == "drivers":
        looked = [d for d in ((plan.drivers.dimensions if plan.drivers else None) or []) if d not in plan.group_by]
        if not looked and not plan.group_by:
            return plan
        return plan.model_copy(update={"intent": "compare", "drivers": None, "group_by": [*plan.group_by, *looked],
                                       "share_of_change": True})
    if not plan.group_by:
        return plan
    return plan.model_copy(update={"share_of_change": True, "intent": plan.intent or "compare"})


# "orders over $200", "baskets above $75 in July", "which orders came to more than $1,500", "invoices of more than
# 5,000", "transactions worth over $1,000".
_DOCUMENT_TOTAL = re.compile(
    r"\b(orders?|invoices?|baskets?|transactions?|tickets?|receipts?|claims?|bills?|shipments?|deals?|bookings?|"
    r"purchases?|sales\s+orders?|purchase\s+orders?|statements?)\b(?:\s+(?:in|from|during|for)\s+[^,.?!]{1,30}?)?"
    r"\s+(?:that\s+|which\s+)?(?:came\s+to\s+|totall?ed\s+|totall?ing\s+|were\s+worth\s+|was\s+worth\s+|worth\s+|"
    r"(?:with\s+)?(?:a\s+)?(?:total|value|amount)\s+(?:of\s+)?|of\s+)?(?:over|above|more\s+than|greater\s+than|"
    r"bigger\s+than|larger\s+than|at\s+least|under|below|less\s+than)\s+[$€£]?\s?\d", re.IGNORECASE)


@_kept_safe
def total_at_its_level(plan: Plan, model: SemanticModel, question: str) -> Plan:
    """ "List every order over $200 in September 2026 with the customer, store and amount" kept the customers whose
    total passed $200 (none did, at that grain, and the answer said so): a total of a document is that document's,
    so the answer is grouped by the document's number, the condition on each one's total."""
    hit = _DOCUMENT_TOTAL.search(question or "")
    if plan.kind != "query" or hit is None or plan.shaped() or plan.time.compare is not None:
        return plan
    from core2.model.schema import AggExpr

    # An amount of money each document came to: never a count ("stores with orders above 100" is the stores'
    # number of orders), and never the members the question asks about ("which customers had orders totalling
    # more than $2,000" is each customer's own total).
    measured = [model.measures[m] for m in (_slug_of(model, f.field) for f in plan.filters) if m]
    measured = [m for m in measured if m.format == "currency" and isinstance(m.expr, AggExpr) and m.expr.agg == "sum"]
    if not measured or any(_members_named(question, model.entities[g]) for g in plan.group_by
                           if g in model.entities):
        return plan
    word = names.singular(hit.group(1).split()[-1].casefold())
    tables = {m.table for m in measured if m.table}
    numbers = [a for a in model.attributes.values() if a.kind == "identifier" and a.column in model.columns
               and model.columns[a.column].table in tables and word in _words(a.business_name)]
    if len(numbers) != 1 or numbers[0].slug in plan.group_by:
        return plan
    return plan.model_copy(update={"group_by": [numbers[0].slug, *plan.group_by],
                                   "intent": plan.intent if plan.intent in ("list", "rank") else "list"})


def _slug_of(model: SemanticModel, slug: str) -> str | None:
    """The key of the measure a filter's field names, or None (a filter on a member, a date)."""
    return next((key for key, m in model.measures.items() if m.slug == slug), None)


# Periods named outright, as the calendar's: "Q3 2025", "2025 Q3", "H1 2026", "first half of 2025", "March 2026",
# "Sep 2025", "2024" (a year alone, not one of the others' years).
_MONTHS = {m: i for i, ms in enumerate(("jan january", "feb february", "mar march", "apr april", "may", "jun june",
                                        "jul july", "aug august", "sep sept september", "oct october",
                                        "nov november", "dec december"), start=1) for m in ms.split()}
_NAMED = re.compile(
    r"(?<![\w-])(?:q([1-4])\s*(?:of\s+)?((?:19|20)\d\d)|((?:19|20)\d\d)\s*[-\s]?q([1-4])|"
    r"h([12])\s*(?:of\s+)?((?:19|20)\d\d)|(first|second|1st|2nd)\s+half\s+(?:of\s+)?((?:19|20)\d\d)|"
    r"(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?\s+((?:19|20)\d\d)|((?:19|20)\d\d))(?![\w-])",
    re.IGNORECASE)
_FISCAL_WORDS = re.compile(r"\b(?:fiscal|fy\s?'?\d{0,4}|fytd|exercice|financial\s+year)\b", re.IGNORECASE)


# Part of a period ("Jan-May 2026", "year to date", "through June"): the two named are not the periods compared.
_PART_OF_PERIOD = re.compile(
    r"\b(?:to\s+date|so\s+far|ytd|mtd|qtd|through|thru|until|till|up\s+to|first\s+\d+|last\s+\d+)\b|"
    r"\b[a-z]{3,9}\.?\s*(?:-|–|to)\s*[a-z]{3,9}\.?\s+(?:19|20)\d\d", re.IGNORECASE)
# What may stand between the two periods compared: "H1 2025 vs H1 2026", "from 2024 to 2025", "Q3 2025 and Q3 2026".
_BETWEEN_PERIODS = re.compile(r"^\s*,?\s*(?:vs\.?|versus|against|compared\s+(?:to|with)|to|and|or|over|than)"
                              r"\s+(?:in\s+|the\s+)?$", re.IGNORECASE)


def _periods_named(question: str, *, side_by_side: bool = False) -> list[tuple[str, "object", "object"]]:
    """Each period the question names, in order: (kind, first day, last day), kind year | half | quarter | month.
    ``side_by_side``: only two named one after the other, joined as a comparison ("H1 2025 vs H1 2026"), else
    none -- never a period a condition holds ("customers who joined in Q3 2025")."""
    import datetime as dt

    found = list(_NAMED.finditer(question or ""))
    if side_by_side:
        pairs = [(a, b) for a, b in zip(found, found[1:])
                 if _BETWEEN_PERIODS.match(question[a.end():b.start()])]
        found = list(pairs[0]) if len(pairs) == 1 else []
    out = []
    for m in found:
        g = m.groups()
        if g[0] or g[3]:
            q, year = (int(g[0]), int(g[1])) if g[0] else (int(g[3]), int(g[2]))
            start = dt.date(year, 3 * q - 2, 1)
            end = (dt.date(year + 1, 1, 1) if q == 4 else dt.date(year, 3 * q + 1, 1)) - dt.timedelta(days=1)
            out.append(("quarter", start, end))
        elif g[4] or g[6]:
            half = int(g[4]) if g[4] else (1 if g[6].casefold() in ("first", "1st") else 2)
            year = int(g[5] or g[7])
            start = dt.date(year, 1 if half == 1 else 7, 1)
            end = dt.date(year, 6, 30) if half == 1 else dt.date(year, 12, 31)
            out.append(("half", start, end))
        elif g[8]:
            month, year = _MONTHS[g[8].casefold()], int(g[9])
            start = dt.date(year, month, 1)
            end = (dt.date(year + 1, 1, 1) if month == 12 else dt.date(year, month + 1, 1)) - dt.timedelta(days=1)
            out.append(("month", start, end))
        else:
            year = int(g[10])
            out.append(("year", dt.date(year, 1, 1), dt.date(year, 12, 31)))
    return out


@_kept_safe
def two_periods_named(plan: Plan, model: SemanticModel, question: str) -> Plan:
    """ "Total orig $ fell from 2024 to 2025 -- how much came from each product family?" was planned as 2024 and
    2025 together against 2022 and 2023, and "H1 2026 against H1 2025" as the bank's fiscal halves. Two periods of
    one kind named outright, with no "fiscal", are the calendar's, and a comparison sets the later one against the
    earlier: the change runs from the first to the second."""
    from core2.plan.ir import Compare, Window

    if plan.kind != "query" or _FISCAL_WORDS.search(question or "") or plan.time.grain or plan.shaped():
        return plan
    if plan.time.compare is None and plan.intent not in ("compare", "drivers"):
        return plan
    if _PART_OF_PERIOD.search(question or ""):
        return plan
    named = list(dict.fromkeys(_periods_named(question, side_by_side=True)))
    kinds = {k for k, _, _ in named}
    if len(named) != 2 or len(kinds) != 1:
        return plan
    earlier, later = sorted(named, key=lambda p: p[1])
    window, compare = plan.time.window, plan.time.compare
    other = compare.window if compare is not None and compare.kind == "window" else None

    def inside(w, period) -> bool:
        return w is not None and w.kind == "between" and not w.fiscal and w.start is not None \
            and w.end is not None and period[1] <= w.start <= w.end <= period[2]

    if inside(window, later) and inside(other, earlier):
        return plan                 # the two periods already, or a like-for-like part of each
    return plan.model_copy(update={"time": plan.time.model_copy(update={
        "window": Window(kind="between", start=later[1], end=later[2]),
        "compare": Compare(kind="window", window=Window(kind="between", start=earlier[1], end=earlier[2]))})})


# "... and what did they spend in total?", "... and how much did they deposit", "with their total".
_AND_THE_AMOUNT = re.compile(r"\band\s+(?:what|how\s+much)\b|\b(?:their|the)\s+total\b|\bin\s+total\b",
                             re.IGNORECASE)


@_kept_safe
def members_over_a_total(plan: Plan, model: SemanticModel, question: str) -> Plan:
    """ "How many customers deposited more than $50,000 with us in Q2 2026? Split it by segment" was planned as the
    deposits kept above $50,000 by segment, and refused (a filter on a total needs it in the answer): it counts the
    customers whose own total passes the amount, in each segment -- each member's figure first, then counted."""
    from core2.plan.ir import PerMember
    from core2.plan.planner import _how_many_of

    if plan.kind != "query" or plan.shaped() or plan.time.compare is not None or plan.derived or plan.rates:
        return plan
    by_slug = {m.slug: m for m in model.measures.values()}
    totals = [f for f in plan.filters if f.field in by_slug and f.op in ("gt", "gte", "lt", "lte", "between")
              and f.values and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in f.values)]
    if len(totals) != 1:
        return plan
    counted = [e for slug, e in model.entities.items() if slug not in plan.group_by and _how_many_of(question, e)]
    if len(counted) != 1:
        return plan
    entity = next(slug for slug, e in model.entities.items() if e is counted[0])
    f = totals[0]
    figure = by_slug[f.field]
    if figure.table == counted[0].table:
        return plan                       # a figure of the members' own rows (their credit limit): a filter
    if any(m != f.field for m in plan.measures) or _AND_THE_AMOUNT.search(question or ""):
        return plan                       # "... and what did they spend in total": the figures asked stay
    return plan.model_copy(update={
        "intent": "count", "measures": [f.field], "filters": [x for x in plan.filters if x is not f],
        "per": PerMember(of=entity, summary="count", op=f.op, values=[float(v) for v in f.values])})


_HOW_MANY = re.compile(r"^\s*(?:how\s+many|the\s+number\s+of|number\s+of|count\s+of|combien)\b", re.IGNORECASE)
_MONEY_WORDS = re.compile(r"\$|€|£|\bdollars?\b|\beuros?\b|\bpounds?\b|\bamount\b|\bvalue\b|\bworth\b|\bbalance",
                          re.IGNORECASE)


@_kept_safe
def how_many_is_a_count(plan: Plan, model: SemanticModel, question: str) -> Plan:
    """ "Branches that originated a 7/1 ARM in 2024 but none in 2025 -- and how many they did in 2024" was answered
    with the dollars they lent: "how many" is never an amount of money. The part of the request asking how many,
    planned as a money total, is the count of the same rows."""
    from core2.model.schema import AggExpr

    if plan.kind != "query":
        return plan
    by_slug = {m.slug: m for m in model.measures.values()}
    swap: dict[str, str] = {}
    for part in plan.asked:
        m = by_slug.get(part.plan or "")
        if m is None or part.kind != "figure" or m.format != "currency" or m.filters or not m.table \
                or getattr(m.expr, "filters", None) or not _HOW_MANY.search(part.words) \
                or _MONEY_WORDS.search(part.words):
            continue
        counts = [c for c in model.measures.values() if c.table == m.table and not c.hidden and not c.filters
                  and isinstance(c.expr, AggExpr) and c.expr.agg in ("count", "count_distinct")
                  and not c.expr.filters]
        # The count the words name ("how many orders": each order once, not each line), else the rows themselves.
        said = set(_words(part.words))
        named = [c for c in counts if set(_words(c.business_name.casefold().removeprefix("number of "))) <= said]
        rows = [c for c in counts if c.expr.agg == "count" and not c.expr.column]
        chosen = named if len(named) == 1 else rows if not named and len(rows) == 1 else []
        if chosen:
            swap[m.slug] = chosen[0].slug
    if not swap:
        return plan
    update: dict = {
        "measures": list(dict.fromkeys(swap.get(s, s) for s in plan.measures)),
        "asked": [a.model_copy(update={"plan": swap[a.plan]}) if a.plan in swap else a for a in plan.asked],
        "sort": [s.model_copy(update={"by": swap[s.by]}) if s.by in swap else s for s in plan.sort]}
    if plan.absent is not None and plan.absent.measure in swap:
        update["absent"] = plan.absent.model_copy(update={"measure": swap[plan.absent.measure]})
    return plan.model_copy(update=update)


@_kept_safe
def absent_of_the_counted(plan: Plan, model: SemanticModel, question: str) -> Plan:
    """ "How many patients filled a statin in Q1 2026 but didn't fill one in Q2 2026? by store" was planned as the
    drugs with no fills in Q2: the data keeps its patients only as a number on the prescriptions, with no list of
    them, and the AI named the closest list it had. Those who had none are the ones the question counts -- the
    count of them the data keeps ("Number of patients"), each told apart by that number."""
    from core2.model.schema import AggExpr
    from core2.resolve import paths as P

    a = plan.absent
    if plan.kind != "query" or a is None or not question:
        return plan
    planned = model.entities.get(a.of)
    if planned is None or _members_named(question, planned):
        return plan
    activity = next((m for m in model.measures.values() if m.slug == a.measure), None)
    found = []
    for m in model.measures.values():
        expr = m.expr
        if m.hidden or m.slug == a.of or not isinstance(expr, AggExpr) or expr.agg != "count_distinct" \
                or not expr.column or expr.filters or m.filters or expr.column not in model.columns:
            continue
        if any(expr.column in e.key_columns or any(j.to_table == e.table and expr.column in j.from_columns
                                                   for j in model.joins.values())
               for e in model.entities.values()):
            continue                               # a list the data has: the entity is named, not its count
        word = m.business_name.casefold().removeprefix("number of ").strip()
        if not word or word == m.business_name.casefold():
            continue
        forms = "|".join(re.escape(w) for w in {word, names.plural(word), names.singular(word)})
        if not re.search(rf"\b(?:how\s+many|number\s+of)\s+(?:\w+\s+){{0,2}}(?:{forms})\b|^\s*(?:\w+\s+){{0,2}}"
                         rf"(?:{forms})\s+(?:who|that|with)\b", question, re.IGNORECASE):
            continue
        holder = model.columns[expr.column].table
        if activity is not None and activity.table and activity.table != holder:
            try:
                if P.best_path(model, activity.table, holder) is None:
                    continue
            except P.Ambiguous:
                continue
        found.append(m)
    if len({m.business_name.casefold() for m in found}) != 1:
        return plan
    # Of several counts of the same members (on the prescriptions, on the immunizations), the one the activity's
    # rows reach.
    counter = next((m for m in found if activity is not None and m.table == activity.table), found[0])
    return plan.model_copy(update={"intent": "count", "absent": a.model_copy(update={"of": counter.slug})})


def _members_named(question: str, entity) -> bool:
    """Does the question name ``entity`` as the ones counted or listed ("how many customers", "customers who",
    "which branches") -- not only as the grouping ("how many by region")?"""
    said = {entity.business_name.casefold(), *(w.casefold() for ws in entity.synonyms.values() for w in ws)}
    said |= {names.plural(w) for w in said}
    words = "|".join(re.escape(w) for w in sorted(said, key=len, reverse=True) if w)
    if not words:
        return False
    return bool(re.search(rf"\b(?:how\s+many|number\s+of|which|what|list(?:\s+the)?|combien\s+(?:de|d')\s*)\s*"
                          rf"(?:(?!by\b|per\b|in\b|each\b|for\b)\w+\s+)?(?:{words})\b|^\s*(?:the\s+)?(?:\w+\s+)?"
                          rf"(?:{words})\s+(?:who|that|which|with|where)\b", question, re.IGNORECASE))


@_kept_safe
def ratio_with_its_parts(plan: Plan, model: SemanticModel, question: str) -> Plan:
    """ "Fee reversals as a % of fees charged, by branch" showed the % alone: a reader checks a ratio against the two
    figures it divides. Each group's ratio the question names as one figure of another is shown beside them, and
    the answer stays ordered by the ratio."""
    from core2.plan.followup import measures_in_order
    from core2.plan.ir import Sort

    if plan.kind != "query" or not plan.group_by or plan.shaped() or plan.time.compare is not None \
            or not _OF_WHOLE.search(question or ""):
        return plan
    named = {slug for _, slug in measures_in_order(model, question)}
    ratios = [d for d in plan.derived if d.op == "ratio" and set(d.measures) <= named]
    if len(ratios) != 1:
        return plan
    parts = [m for m in ratios[0].measures if m not in plan.measures]
    if not parts and plan.sort:
        return plan
    sort = list(plan.sort) or [Sort(by=ratios[0].name, desc=True)]
    return plan.model_copy(update={"measures": [*plan.measures, *parts], "sort": sort})


# "What share of each store's sales came from Beauty", "what % of each region's scripts were Medicaid", "within
# each department, what share of 2025 sales came from private label items".
_CAME_FROM = re.compile(r"\b(?:came|come|comes|coming|was|were|is|are)\s+(?:from\s+|in\s+|on\s+|for\s+)?", re.I)


@_kept_safe
def share_from_a_member(plan: Plan, model: SemanticModel, question: str) -> Plan:
    """ "For the Nebraska stores, what share of each store's front store sales came from Beauty in Q2 2026?" kept
    the sales to Beauty and gave each store's part of all of Beauty: the question asks what part of each store's
    own sales Beauty was -- Beauty's sales over each store's total, the condition on the part only."""
    from core2.plan.ir import Rate

    text = question or ""
    share = _SHARE_OF.search(text)
    if plan.kind != "query" or plan.shaped() or plan.rates or plan.derived or plan.time.compare is not None \
            or not plan.group_by or len(plan.measures) != 1 or share is None:
        return plan
    came = _CAME_FROM.search(text, share.end())
    if came is None:
        return plan
    after = set(_words(text[came.end():]))

    def said(f: Filter) -> bool:
        if any(set(_words(str(v))) and set(_words(str(v))) <= after for v in f.values):
            return True
        attribute = model.attributes.get(f.field)
        own = set(_words(attribute.business_name)) - {"flag", "code", "name", "indicator", "type"} \
            if attribute is not None else set()
        return bool(own) and own <= after

    members = [f for f in plan.filters if f.op in ("eq", "in") and f.field not in plan.group_by and f.values
               and said(f)]
    if len(members) != 1:
        return plan
    f = members[0]
    group = plan.share_of if plan.share_of in plan.group_by else None
    if group is None:
        for hit in _WITHIN.finditer(text):
            group = _named_group(plan, model, hit.group(1))
            if group is not None:
                break
    if group is None:
        if len(plan.group_by) != 1:
            return plan
        group = plan.group_by[0]
    measure = next((m for m in model.measures.values() if m.slug == plan.measures[0]), None)
    if measure is None:
        return plan
    what = " / ".join(str(v) for v in f.values) if all(set(_words(str(v))) <= after for v in f.values) \
        else model.attributes[f.field].business_name
    rate = Rate(name=f"{what} % of {measure.business_name.lower()}", measure=measure.slug, where=[f])
    return plan.model_copy(update={
        "filters": [x for x in plan.filters if x is not f], "rates": [rate], "measures": [], "group_by": [group],
        "share_of": None, "intent": "breakdown",
        "sort": [s for s in plan.sort if s.by in (group, rate.name)]})


_PER_N = re.compile(r"\bper\s+(100|1,?000|10,?000|100,?000)\s+[a-z]", re.IGNORECASE)


@_kept_safe
def per_hundred(plan: Plan, model: SemanticModel, question: str) -> Plan:
    """ "Flu shots per 100 scripts by region" was planned as the flu shots times the scripts: so many of one figure
    for each 100 of another is their ratio, times 100."""
    hit = _PER_N.search(question or "")
    if plan.kind != "query" or hit is None or not plan.derived:
        return plan
    n = float(hit.group(1).replace(",", ""))
    fixed = [d.model_copy(update={"op": "ratio", "scale": n})
             if d.op == "product" and d.scale == n or d.op == "ratio" and d.scale == 1.0 else d for d in plan.derived]
    return plan if fixed == list(plan.derived) else plan.model_copy(update={"derived": fixed})
