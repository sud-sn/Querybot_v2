"""Executive dashboards by kind of business: one shape everywhere, the figures each kind of business leads with.

"An executive dashboard", "an overview", "a CFO dashboard": the whole business. Designed afresh by the AI each
time, two such requests on the same data gave two dashboards of different shapes under names the AI chose. Built
from a recipe, the same data always gives the same dashboard: the kind of business is read off the workspace's own
names (orders, stores and returns are sales; prescriptions and fills a pharmacy; payroll and headcount a
workforce), and its recipe says which figures it leads with, which groupings matter and whose top 10 is shown.

The shape is the same for every business:

* Overview: up to four headline figures over the last 12 months (each card says how it moved against the 12
  months before, and draws its trend line); the analysis
  the recipe names, where the data gives it (the share of receipts on time; new, returning and lapsed customers;
  employees at each month end); the first figure by month (its total is checked against that trend once built);
  the first two figures by their main grouping.
* A tab for each area (the tables things are counted on, the first figure's first, at most three): a figure of
  its own by month, by a grouping the overview has not shown it by, and a top 10.

Every tile's plan is written here, from the model: no AI call designs the dashboard or plans its tiles, and a
tile the data cannot give is left out. The name is the business's ("Sales executive dashboard"). A request naming
a year ("an executive dashboard for 2025") counts that year, against the year before.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core2.bootstrap import names
from core2.model.schema import Attribute, Entity, Measure, SemanticModel
from core2.plan import analyses
from core2.plan.ir import Plan
from core2.resolve.resolver import Context, ResolveError, _date_role_for, resolve

May = Callable[[str], bool]

MOST_HEADLINES = 4
MOST_AREAS = 5                   # the tables a business is told by, a tab each
MOST_GROUP_MEMBERS = 25          # a breakdown's bars; more is a top 10's job
MOST_TILES = 48                  # tiles of the whole business's dashboard; each is a plan run as it is (no AI)
MOST_TABS = 8                    # its tabs, the overview first
MOST_NUMBERS = 6                 # numbers in a row: the overview's, and each business metric area's


def _all(table: str) -> bool:
    return True


@dataclass
class Recipe:
    kind: str
    title: str                       # "Sales": its dashboard is the "Sales executive dashboard"
    titre: str                       # the same, in French
    signs: set[str]                  # words of its tables' and measures' names that say a business is this kind
    leads: list[set[str]]            # its headline figures, in order: words each one's measure is named with
    groupings: list[str]             # words of the groupings its figures are broken down by, the first first
    tops: list[str] = field(default_factory=list)   # whose top 10 it shows: never people by their pay
    analysis: str | None = None      # "lifecycle" | "on_time" | "active": on its overview where the data gives it


RECIPES = [
    Recipe("sales", "Sales", "Ventes",
           {"sale", "order", "revenue", "customer", "store", "shop", "product", "return", "refund", "invoice",
            "channel", "basket", "checkout", "retail", "turnover"},
           [{"net", "revenue", "sale", "turnover"}, {"order"}, {"margin", "profit"}, {"return", "refund"},
            {"quantity", "unit"}, {"gross"}],
           ["category", "territory", "region", "store", "channel", "segment", "country", "brand", "city"],
           tops=["customer", "product", "store", "reseller"], analysis="lifecycle"),
    Recipe("purchasing", "Purchasing", "Achats",
           {"purchase", "supplier", "vendor", "receipt", "procurement", "po", "buyer", "requisition"},
           [{"spend", "amount", "cost", "value"}, {"po", "purchase"}, {"received", "receipt"}, {"rejected"}],
           ["category", "supplier", "vendor", "country", "buyer"],
           tops=["supplier", "vendor", "item"], analysis="on_time"),
    Recipe("workforce", "Workforce", "Effectifs",
           {"employee", "payroll", "headcount", "salary", "hire", "termination", "staff", "fte", "pay", "overtime",
            "department", "absence"},
           [{"headcount"}, {"fte"}, {"gross", "pay", "payroll", "salary"}, {"overtime"}],
           ["department", "location", "level", "type", "country", "grade", "job"],
           tops=["department", "location"], analysis="active"),
    Recipe("subscriptions", "Subscriptions", "Abonnements",
           {"subscription", "mrr", "arr", "plan", "seat", "churn", "renewal", "recurring", "subscriber", "trial"},
           [{"mrr", "arr", "recurring"}, {"seat"}, {"amount", "billed", "billing"}, {"subscription"}],
           ["plan", "segment", "country", "billing", "channel"],
           tops=["customer", "plan"], analysis="active"),
    Recipe("pharmacy", "Pharmacy", "Pharmacie",
           {"prescription", "rx", "fill", "drug", "pharmacy", "prescriber", "formula", "compound", "dispense",
            "refill", "ndc", "pharmacist", "batch", "dosage"},
           [{"fill", "prescription", "rx", "dispense", "dispensed"}, {"price", "charged", "revenue", "sale", "paid"},
            {"cost", "ingredient"}, {"claim", "billed"}],
           ["payer", "formula", "category", "pharmacy", "store", "drug", "plan", "type"],
           tops=["formula", "drug", "prescriber", "product"], analysis="lifecycle"),
    Recipe("healthcare", "Healthcare", "Santé",
           {"encounter", "patient", "claim", "condition", "procedure", "provider", "payer", "admission",
            "diagnosis", "immunization", "observation", "careplan", "allergy"},
           [{"encounter", "visit", "admission"}, {"claim", "cost", "paid", "charge"}, {"procedure"},
            {"medication"}],
           ["class", "type", "payer", "organization", "provider", "state", "city"],
           tops=["organization", "provider", "payer"], analysis="lifecycle"),
    Recipe("inventory", "Inventory", "Stocks",
           {"stock", "inventory", "warehouse", "hand", "onhand", "allocated", "movement", "bin", "lot",
            "replenishment", "reorder", "balance"},
           [{"hand", "onhand", "stock", "available"}, {"value"}, {"allocated", "reserved"},
            {"movement", "moved", "extended"}],
           ["warehouse", "group", "category", "location", "type", "class", "supplier"],
           tops=["item", "product", "warehouse"]),
    Recipe("finance", "Finance", "Finance",
           {"ledger", "gl", "journal", "account", "budget", "debit", "credit", "posting", "centre", "center",
            "entry", "actual", "variance"},
           [{"actual", "net", "balance"}, {"budget"}, {"debit"}, {"credit"}],
           ["division", "type", "account", "centre", "center", "department", "category", "class"],
           tops=["account", "centre", "center"]),
    Recipe("network", "Network operations", "Exploitation réseau",
           {"device", "alarm", "incident", "outage", "ticket", "site", "node", "interface", "uptime", "latency",
            "router", "circuit", "bandwidth"},
           [{"alarm", "incident"}, {"ticket"}, {"billed", "revenue"}, {"usage", "outage", "downtime"}],
           ["severity", "site", "region", "type", "vendor", "priority", "category"],
           tops=["site", "device", "customer"]),
    Recipe("field_service", "Field service", "Interventions",
           {"job", "technician", "appointment", "visit", "dispatch", "callout", "crew", "booking", "estimate"},
           [{"booking", "job", "visit", "appointment"}, {"revenue", "amount", "sale"}, {"hour", "labor", "labour"},
            {"visit"}],
           ["service", "type", "region", "branch", "partner", "category", "status"],
           tops=["customer", "technician"], analysis="lifecycle"),
    Recipe("manufacturing", "Manufacturing", "Production",
           {"production", "scrap", "assembly", "machine", "yield", "routing", "material", "bom", "workcenter",
            "operation", "downtime"},
           [{"produced", "production", "output"}, {"scrap", "waste"}, {"cost"}, {"downtime"}],
           ["line", "plant", "machine", "product", "location", "category"],
           tops=["product", "machine"]),
    # A bank's words are its own: loans, deposits, principal and collateral say a bank, where customers, products
    # and transactions alone would say a shop.
    Recipe("bank", "Banking", "Banque",
           {"loan", "deposit", "principal", "collateral", "interest", "mortgage", "overdraft", "teller", "atm",
            "delinquency", "borrower", "lending", "arrears", "repayment", "withdrawal", "branch", "chargeoff"},
           [{"principal", "loan", "lending"}, {"balance", "deposit"}, {"interest"}, {"transaction"},
            {"payment", "repayment"}],
           ["product", "branch", "region", "segment", "type", "channel", "category", "state", "family"],
           tops=["branch", "customer", "product", "officer"], analysis="lifecycle"),
]
# The industry an admin set on the Compliance page says the kind of business when the names alone leave it close.
# The kinds of business an industry is (healthcare and pharmacy is a clinic's or a pharmacy's: never one over the
# other, only over every other kind).
INDUSTRY_KIND = {"banking": frozenset({"bank"}), "healthcare_pharmacy": frozenset({"pharmacy", "healthcare"})}
INDUSTRY_WEIGHT = 3.0
GENERIC = Recipe("business", "", "", set(), [],
                 ["category", "region", "type", "segment", "country", "group", "class", "status"],
                 tops=["customer", "client", "product", "item", "account", "site", "store"])
FEWEST_SIGNS = 3                 # what says a business's kind: a sign on a table things are counted on is two

# Copies kept beside the real table ("Devices bak"): never an area of the business.
_COPIES = {"bak", "backup", "copy", "old", "tmp", "temp", "archive", "staging", "stg"}
# Words of a table's name that say how it is kept, not what it holds ("Headcount monthly", "Order line").
_KEPT = {"monthly", "daily", "weekly", "yearly", "snapshot", "snapshots", "fact", "facts", "history", "line",
         "lines", "detail", "details", "header", "headers", "rows", "row"}

# Words of a request that ask for the whole business, not a subject: "an executive dashboard", "a CFO dashboard".
_WHOLE = {"executive", "executives", "exec", "overview", "kpi", "kpis", "ceo", "cfo", "coo", "cxo", "management",
          "leadership", "board", "business", "company", "summary", "headline", "headlines", "general", "global",
          "overall", "whole", "main", "key", "figures", "high", "level", "direction", "dirigeants", "pilotage",
          "entreprise", "synthèse", "synthese", "ensemble", "vue", "d'ensemble", "général", "generale", "générale",
          "globale", "société", "activité", "indicateurs", "clés"}
_FILLER = {"a", "an", "the", "of", "our", "my", "for", "on", "in", "all", "everything", "de", "la", "le", "les", "du",
           "des", "d", "l", "en", "pour", "sur", "mon", "notre", "nos", "un", "une", "et", "and", "with", "avec"}
_YEAR = re.compile(r"^(19|20)\d\d$")


def whole_business(about: str) -> tuple[bool, int | None]:
    """Does the request ask for the whole business ("" or "executive", "CFO", "an overview for 2025")? And the
    year it names, if any."""
    words = re.findall(r"[\w'’-]+", (about or "").casefold().replace("’", "'"))
    years = [int(w) for w in words if _YEAR.match(w)]
    rest = [w for w in words if not _YEAR.match(w) and w not in _FILLER and w not in _WHOLE]
    return not rest and len(years) <= 1, (years[0] if years else None)


def _words(name: str) -> set[str]:
    out: set[str] = set()
    for w in names.tokens(name or ""):
        for part in names.EXPANSIONS.get(w, w).split():
            out.add(names.singular(part))
    return out


def _stems(name: str) -> set[str]:
    """A name's words to their stems, for telling two names of one subject apart from two subjects: Shipping and
    Shipment are one, Quality and Quality tests one, Repayments and Payments two."""
    out = set()
    for w in _words(name):
        stem = re.sub(r"(ments?|ings?)$", "", w) if len(w) > 6 else w
        out.add(stem[:-1] if len(stem) > 3 and stem[-1] == stem[-2] else stem)
    return out


def _last(name: str) -> str:
    words = names.tokens(name or "")
    return names.singular(names.EXPANSIONS.get(words[-1], words[-1]).split()[-1]) if words else ""


def _figures(model: SemanticModel, may: May) -> list[Measure]:
    """The figures a business is told by: on rows that happen on a date, summed or counted (an average of each
    row's price, or a count of a table's rows, says nothing of the business), or a metric an admin defined. Not
    a count of a list's members (customers are counted by what they did), nor a figure in several units or
    currencies (one number of a tile would be four)."""
    out = []
    for m in model.measures.values():
        table = model.tables.get(m.table)
        if m.hidden or m.kpi or table is None or not may(m.table) or not dated(model, m) \
                or table.kind == "dimension" or _mixed(model, m) or _words(table.business_name) & _COPIES:
            continue
        agg = getattr(m.expr, "agg", "")
        if m.kind != "metric" and (agg not in ("sum", "count", "count_distinct") or m.format == "percent"):
            continue
        if names.tokens(m.business_name)[-1:] == ["rows"]:
            continue
        out.append(m)
    return out


def dated(model: SemanticModel, m: Measure) -> bool:
    """Counted on rows that happen on a date: its own, or, for an order line with no date of its own, its order's."""
    return bool(m.default_date) or m.table in model.tables and _date_role_for(
        model, Plan.model_validate({"kind": "query"}), m, m.table) is not None


def _mixed(model: SemanticModel, m: Measure) -> bool:
    """Kept in several units or currencies: a unit on each row and none for the whole, or Learn saw the mix."""
    column = getattr(m.expr, "column", "") or ""
    return bool(m.unit_column and not m.unit) or any(q.kind == "unit_mix" and q.object == column for q in model.quality)


def kind_of(model: SemanticModel, may: May = _all, industry: str = "") -> Recipe:
    """The recipe whose signs the workspace's names carry most (a sign on a table things are counted on counts
    twice), or the generic one where none carries enough. The industry an admin set on the Compliance page
    (``industry``: "banking", "healthcare_pharmacy") weighs for its own kind, once its names show it at all."""
    counted = {m.table for m in _figures(model, may)}
    seen: dict[str, float] = {}
    for t in model.tables.values():
        if may(t.key) and t.kind != "calendar":
            for w in _words(t.business_name or t.name):
                seen[w] = max(seen.get(w, 0.0), 2.0 if t.key in counted else 1.0)
    for m in model.measures.values():
        if not m.hidden and may(m.table):
            for w in _words(m.business_name):
                seen[w] = max(seen.get(w, 0.0), 1.0)
    hinted = INDUSTRY_KIND.get(industry or "", frozenset())

    def score(r: Recipe) -> float:
        found = sum(seen.get(s, 0.0) for s in r.signs)
        return found + INDUSTRY_WEIGHT if found and r.kind in hinted else found

    scored = [(score(r), -i, r) for i, r in enumerate(RECIPES)]
    best_score, _, best = max(scored, key=lambda s: s[:2])
    return best if best_score >= FEWEST_SIGNS else GENERIC


# What a lead is measured in: money words want an amount ("sales": the sales amount, never the number of sales);
# words of things counted want a count ("orders": the number of orders, never the order quantity).
_MONEY = {"revenue", "sale", "net", "gross", "turnover", "amount", "spend", "cost", "value", "price", "pay", "payroll",
          "salary", "mrr", "arr", "billed", "billing", "budget", "charged", "paid", "refund", "margin", "profit"}
_THINGS = {"order", "return", "fill", "prescription", "rx", "visit", "booking", "job", "appointment", "encounter",
           "admission", "alarm", "incident", "ticket", "receipt", "claim", "po", "purchase", "subscription"}


def _wants(lead: set[str]) -> str | None:
    return "currency" if lead & _MONEY else "count" if lead & _THINGS else None


def headlines(model: SemanticModel, recipe: Recipe, may: May = _all) -> list[Measure]:
    """Up to four figures the business leads with: the recipe's, in its order, then the rest by kind (an admin's
    metric first, a figure named like a lead, money before counts, the busiest table first), at most two of one
    table."""
    figures = _figures(model, may)
    rows = {k: model.tables[k].row_count or 0 for k in {m.table for m in figures}}
    rank = {"currency": 0, "count": 1}
    picked: list[Measure] = []
    for lead in recipe.leads:
        if len(picked) == MOST_HEADLINES:
            break
        tables = {m.table for m in picked}
        fits = [m for m in figures if m not in picked and _words(m.business_name) & lead]
        want = _wants(lead)
        if fits:
            # Measured as the lead is (an amount for sales, a count for orders); the one its words say most of
            # ("gross pay" for pay, before a monthly salary), named for it ("number of visits", never "visits
            # included" in a plan), on a table already leading ("on hand value" beside "on hand quantity"), named
            # most plainly.
            picked.append(min(fits, key=lambda m: (m.kind != "metric", want is not None and m.format != want,
                                                   -len(_words(m.business_name) & lead),
                                                   _last(m.business_name) not in lead,
                                                   m.table not in tables if tables else False,
                                                   len(names.tokens(m.business_name)), rank.get(m.format, 2),
                                                   -rows[m.table], m.slug)))
    led = set().union(*recipe.leads) if recipe.leads else set()
    rest = sorted((m for m in figures if m not in picked),
                  key=lambda m: (m.kind != "metric", -len(_words(m.business_name) & led), rank.get(m.format, 2),
                                 -rows[m.table], m.slug))
    for m in rest:
        if len(picked) == MOST_HEADLINES:
            break
        if sum(p.table == m.table for p in picked) < 2:
            picked.append(m)
    return picked


def _named_by(model: SemanticModel, a: Attribute) -> str:
    e = model.entities.get(a.entity) if a.entity else None
    if e is not None and a.column == e.label_column:
        return e.business_name or a.business_name
    return a.business_name or a.slug


def groupings(model: SemanticModel, m: Measure, recipe: Recipe, may: May = _all) -> list[Attribute]:
    """The groupings of a few members (two to 25) the figure may be broken down by, the recipe's first, a
    member's name before its code. Never a person's detail. (Whether the figure reaches it, the resolver says.)"""
    found = []
    for a in model.attributes.values():
        column = model.columns.get(a.column)
        if a.kind != "group" or not 2 <= (a.members or 0) <= MOST_GROUP_MEMBERS or column is None \
                or not may(column.table) or column.personal != "none":
            continue
        e = model.entities.get(a.entity) if a.entity else None
        if e is not None and e.role or e is None and column.table != m.table:
            continue
        found.append((a, e))

    def rank(item: tuple[Attribute, Entity | None]) -> tuple:
        a, e = item
        words = _words(_named_by(model, a)) | (_words(e.business_name) if e is not None else set())
        hit = next((i for i, g in enumerate(recipe.groupings) if g in words), len(recipe.groupings))
        named = e is not None and a.column == e.label_column
        return hit, not named, a.members, a.slug

    return [a for a, _ in sorted(found, key=rank)]


def tops(model: SemanticModel, m: Measure, recipe: Recipe, may: May = _all) -> list[Entity]:
    """Whose top 10 the figure may rank, the recipe's first: of more members than a breakdown shows, with a name
    that is no person's detail."""
    found = []
    for e in model.entities.values():
        if e.role or not may(e.table) or e.table == m.table or (e.members or 0) <= 10 or not e.label_column:
            continue
        label = model.columns.get(e.label_column)
        if label is not None and label.personal == "detail":
            continue
        words = _words(e.business_name or e.slug)
        hit = next((i for i, t in enumerate(recipe.tops) if t in words), None)
        if hit is not None:
            found.append((hit, -(e.members or 0), e.slug, e))
    return [f[3] for f in sorted(found, key=lambda f: f[:3])]


def area_name(model: SemanticModel, table_key: str) -> str:
    """A tab's name for the rows of a table: "Orders" for order lines, "Payroll", "Headcount" for a monthly
    headcount, "Receipts"."""
    table = model.tables[table_key]
    words = (table.business_name or table.name).split()
    kept = [w for w in words if w.casefold() not in _KEPT] or words
    counted = next((m for m in model.measures.values() if m.table == table_key and getattr(m.expr, "agg", "") == "count"
                    and m.business_name.casefold().startswith("number of ")), None)
    countable = table.kind != "snapshot" and (counted is None or not counted.business_name.casefold().endswith(" rows"))
    name = " ".join(kept[:-1] + [names.plural(kept[-1]) if countable else kept[-1]])
    return name[:1].upper() + name[1:]


@dataclass
class Made:
    """A tile of the recipe: what it shows, the question it answers, and the plan that answers it."""
    title: str
    kind: str                        # number | trend | breakdown | table
    question: str
    plan: dict[str, Any]
    tab: str


_SAY = {
    "en": {"overview": "Overview", "period": "over the last 12 months", "in": "in {year}", "by_month": "{m} by month",
           "by": "{m} by {g}", "top": "Top 10 {e}", "top_q": "Top 10 {e} by {m}", "name": "{title} executive dashboard",
           "plain": "Executive dashboard",
           "lifecycle": "New, returning and lapsed {members} by month", "on_time": "Share of {rows} on time",
           "active": "{Rows} at each month end", "against": "{a} against {t} by month"},
    "fr": {"overview": "Vue d'ensemble", "period": "sur les 12 derniers mois", "in": "en {year}",
           "by_month": "{m} par mois", "by": "{m} par {g}", "top": "Top 10 des {e}", "top_q": "Top 10 des {e} par {m}",
           "name": "Tableau de bord de direction — {title}", "plain": "Tableau de bord de direction",
           "lifecycle": "Nouveaux, fidèles et perdus par mois ({members})", "on_time": "Part des {rows} à l'heure",
           "active": "{Rows} en cours à chaque fin de mois", "against": "{a} par rapport à {t} par mois"},
}


def executive(model: SemanticModel, *, today: dt.date, year: int | None = None, language: str = "en",
              allowed: set[str] | None = None, industry: str = "") -> tuple[str, list[Made]] | None:
    """The whole business's dashboard, by the recipe of its kind: its name and its tiles, each with its plan, for
    a reader allowed ``allowed`` tables (None: all). A tile is kept only where the resolver can answer its plan:
    a grouping the figure reaches one way, a top 10 of members it is counted by. None when the data has no
    figure to lead with."""
    def may(table: str) -> bool:
        return allowed is None or table in allowed

    ctx = Context(today=today, allowed_tables=allowed)

    def answers(plan: dict[str, Any]) -> bool:
        try:
            resolve(Plan.model_validate(plan), model, ctx)
        except (ResolveError, ValueError):
            return False
        return True

    say = _SAY.get(language, _SAY["en"])
    recipe = kind_of(model, may, industry)
    leads = [m for m in headlines(model, recipe, may) if answers(_value(m, {"kind": "all"}))]
    if not leads:
        return None
    if year is not None:
        window: dict[str, Any] = {"kind": "between", "start": f"{year}-01-01", "end": f"{year}-12-31"}
        when = say["in"].format(year=year)
    else:
        window, when = {"kind": "last", "unit": "month", "n": 12}, say["period"]
    overview = say["overview"]
    made: list[Made] = []
    # Two figures of one name on two tables (the sales amount of resellers and of the web): each says whose.
    by_name: dict[str, set[str]] = {}
    for f in _figures(model, may):
        by_name.setdefault(f.business_name.casefold(), set()).add(f.table)

    def called(m: Measure) -> str:
        if len(by_name.get(m.business_name.casefold(), ())) > 1:
            return f"{m.business_name} ({area_name(model, m.table)})"
        return m.business_name

    shown: set[tuple[str, str]] = set()          # (figure, grouping, "month" or "top") already on the dashboard

    def add(tile: Made, *marks: tuple[str, str]) -> bool:
        if not answers(tile.plan):
            return False
        made.append(tile)
        shown.update(marks)
        return True

    def trend(m: Measure, tab: str) -> None:
        title = say["by_month"].format(m=called(m))
        add(Made(title, "trend", f"{title} {when}", {"kind": "query", "intent": "trend", "measures": [m.slug],
                                                     "time": {"grain": "month", "window": window}}, tab),
            (m.slug, "month"))

    def breakdown(m: Measure, tab: str) -> None:
        for a in groupings(model, m, recipe, may):
            if (m.slug, a.slug) in shown or ("by", a.slug) in shown and tab == overview:
                continue
            title = say["by"].format(m=called(m), g=_named_by(model, a).lower())
            if add(Made(title, "breakdown", f"{title} {when}",
                        {"kind": "query", "intent": "breakdown", "measures": [m.slug], "group_by": [a.slug],
                         "sort": [{"by": m.slug, "desc": True}], "time": {"window": window}}, tab),
                   (m.slug, a.slug), ("by", a.slug)):
                return

    def top(m: Measure, tab: str) -> None:
        for e in tops(model, m, recipe, may):
            if ("top", e.slug) in shown:
                continue
            many = names.plural((e.business_name or e.slug).lower())
            if add(Made(say["top"].format(e=many), "table",
                        f"{say['top_q'].format(e=many, m=called(m).lower())} {when}",
                        {"kind": "query", "intent": "rank", "measures": [m.slug], "group_by": [e.slug],
                         "limit": 10, "sort": [{"by": m.slug, "desc": True}], "time": {"window": window}}, tab),
                   ("top", e.slug)):
                return

    # The business metrics an admin accepted (core2/kpis.py), the most important first, where this reader's
    # tables answer them.
    from core2 import kpis

    accepted = [a for a in kpis.approved(model, may=may, lang=language) if answers(_value(a.measure, window))]
    for m in leads:
        add(Made(called(m), "number", f"{called(m)} {when}", _value(m, window), overview))
    extra = _analysis(model, recipe, window, when, say, may)
    if extra is not None:
        extra.tab = overview
        add(extra)
    # The business's headline ratios beside its totals, as far as the row of numbers goes.
    room = MOST_NUMBERS - sum(t.kind == "number" for t in made)
    for a in [a for a in accepted if a.rank == 1][:max(room, 0)]:
        add(Made(a.measure.business_name, "number", f"{a.measure.business_name} {when}", _value(a.measure, window),
                 overview))
    trend(leads[0], overview)
    for m in leads[:2]:
        breakdown(m, overview)

    # The areas of the business its accepted metrics read (Lending, Risk, Repayments), and the tables it is told
    # by (Loans, Transactions): a table's tab named for an area (Quality tests, Quality) goes on the area's.
    areas_of: dict[str, list[Any]] = {}
    for a in accepted:
        areas_of.setdefault(a.area, []).append(a)
    english = {area: those[0].area_key or area for area, those in areas_of.items()}

    def area_for(tab: str) -> str:
        return next((area for area in areas_of if _stems(english[area]) and _stems(english[area]) <= _stems(tab)),
                    tab)

    figures = _figures(model, may)
    rows = {m.table: model.tables[m.table].row_count or 0 for m in figures}
    order = list(dict.fromkeys(m.table for m in leads))
    areas = [*order, *sorted({m.table for m in figures} - set(order), key=lambda t: (-rows[t], t))][:MOST_AREAS]
    by_table: list[tuple[str, str, list[Measure]]] = []
    for table_key in areas:
        own = [m for m in leads if m.table == table_key] + [
            m for m in headlines(model, recipe, lambda t, k=table_key: t == k and may(t))
            if m not in leads and answers(_value(m, window))]
        if own:
            by_table.append((table_key, area_for(area_name(model, table_key)), own))
    # Which tabs there is room for: the areas of the business's leading metrics, the tables its headline totals
    # are counted on, the other areas, the other tables.
    ranked = [area for area, those in areas_of.items() if min(a.rank for a in those) <= 2]
    ranked += [tab for table_key, tab, _ in by_table if table_key in order]
    ranked += list(areas_of)
    ranked += [tab for _, tab, _ in by_table]
    room = [overview.casefold()]
    for tab in ranked:
        if tab.casefold() not in room and len(room) < MOST_TABS:
            room.append(tab.casefold())

    # Each area: its numbers first, the first two by month, the first by the business's grouping.
    for area, those in areas_of.items():
        if area.casefold() not in room:
            continue
        for a in those[:MOST_NUMBERS]:
            add(Made(a.measure.business_name, "number", f"{a.measure.business_name} {when}",
                     _value(a.measure, window), area))
        for a in those[:2]:
            trend(a.measure, area)
        breakdown(those[0].measure, area)

    paired = analyses.target_pair(model, may)
    against: dict[str, Made] = {}          # a target's table: its figure against the target, by month
    if paired is not None:
        actual, target = paired
        title = say["against"].format(a=actual.business_name, t=target.business_name.lower())
        made_target = Made(title, "trend", f"{title} {when}",
                           {"kind": "query", "intent": "trend", "target": {"actual": actual.slug, "target": target.slug},
                            "time": {"grain": "month", "window": window}}, "")
        if answers(made_target.plan):
            against[target.table] = made_target
            shown.add((target.slug, "month"))

    tabs: set[str] = {overview.casefold()}
    for table_key, tab, own in by_table:
        if tab.casefold() in tabs or tab.casefold() not in room:
            continue
        tabs.add(tab.casefold())
        if table_key in against:
            against[table_key].tab = tab
            made.append(against[table_key])
        trended = next((m for m in own if (m.slug, "month") not in shown), None)
        if trended is not None:
            trend(trended, tab)
        breakdown(own[0], tab)
        top(own[0], tab)
    made = made[:MOST_TILES]
    title = recipe.titre if language == "fr" else recipe.title
    name = say["name"].format(title=title) if title else say["plain"]
    return (f"{name} {year}" if year is not None else name), made


def _value(m: Measure, window: dict[str, Any]) -> dict[str, Any]:
    """A headline figure over the period: its card says how it moved against the period before, and draws its
    trend line, as any one number's does (a plan asking for the comparison itself would be drawn as one)."""
    return {"kind": "query", "intent": "value", "measures": [m.slug], "time": {"window": window}}


def _analysis(model: SemanticModel, recipe: Recipe, window: dict[str, Any], when: str, say: dict[str, str],
              may: May) -> Made | None:
    """The analysis the recipe shows on its overview, where the data gives it (core2/plan/analyses.py)."""
    if recipe.analysis == "lifecycle":
        act = analyses.activity(model, may)
        if act is not None:
            title = say["lifecycle"].format(members=names.plural(act.member))
            return Made(title, "trend", f"{title} {when}",
                        {"kind": "query", "intent": "trend",
                         "lifecycle": {"of": act.entity.slug, "measure": act.measure.slug},
                         "time": {"grain": "month", "window": window}}, "")
    if recipe.analysis == "on_time":
        pair = analyses.due_and_done(model, may)
        if pair is not None:
            due, done = pair
            title = say["on_time"].format(rows=names.plural(model.tables[done.table].business_name.lower()))
            return Made(title, "number", f"{title} {when}",
                        {"kind": "query", "intent": "value", "on_time": {"done": done.slug, "due": due.slug},
                         "time": {"window": window}}, "")
    if recipe.analysis == "active":
        pair = analyses.start_and_end(model, may)
        if pair is not None:
            start, end = pair
            rows = names.plural(model.tables[start.table].business_name.lower())
            title = say["active"].format(Rows=rows[:1].upper() + rows[1:])
            return Made(title, "trend", f"{title} {when}",
                        {"kind": "query", "intent": "trend", "active": {"start": start.slug, "end": end.slug},
                         "time": {"grain": "month", "window": window}}, "")
    return None
