"""Dashboards by subject and by reader: a catalogue of recipes, each built for the person who reads it.

"A dashboard for the sales team", "customer retention dashboard", "something for my store managers": a request
that names a subject or a reader is built by its recipe, planned in code from the workspace's model, so the same
data always gives the same dashboard and no AI call designs it. Seven groups: sales and customers, inventory and
supply, finance, people, subscriptions, health and pharmacy, operations.

A recipe finds what it shows by the words things are named with: its figures (net sales, orders, refunds, stock on
hand), the groupings it splits them by (regions, channels, categories), the lists it ranks (stores, products,
suppliers), and the analyses Learn found the data gives (a customer's orders, a due and a done date, a start and
an end, a target). Every tile's plan is checked by the resolver before it is kept: a tile the data cannot give is
left out, and the recipe says what was missing ("no targets in this data"). A figure the recipe measures against a
target the data does not have is measured against the year before in its place, and said. A recipe whose main
figure the data does not have is not for this data, and says what it needs.

Readers never see the catalogue: they ask in their own words, and the recipe the words name is built. ``catalogue``
says, for a workspace and a reader's tables, which recipes are ready, which only in part, and which are not for
this data; ``find`` the recipes a request names; ``build`` one recipe's name and tiles; ``around`` a dashboard
around the answer on screen ("create a dashboard using this").
"""

from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core2 import recipes
from core2.bootstrap import names
from core2.model.schema import Attribute, DateRole, Entity, Measure, SemanticModel
from core2.plan import analyses
from core2.plan.ir import Plan
from core2.recipes import Made
from core2.resolve.resolver import Context, ResolveError, _documents, resolve

May = Callable[[str], bool]

YEAR = {"kind": "last", "unit": "month", "n": 12}
QUARTER = {"kind": "last", "unit": "month", "n": 3}
MONTH = {"kind": "last", "unit": "month", "n": 1}
TO_DATE = {"kind": "to_date", "unit": "year"}
MOST_TILES = 16
FEWEST = 3                      # tiles a recipe gives at least, or it is not for this data
QUIET_DAYS = 90                 # a member not seen for this long has lapsed


def _words(name: str) -> set[str]:
    return recipes._words(name)


# ── words the pieces of a recipe are found by ───────────────────────────────

SALES = {"net", "revenue", "sale", "turnover", "sold", "income"}
SOLD = {"revenue", "sale", "turnover", "sold", "income"}             # words that say sales on any table
SALES_TABLES = {"order", "sale", "invoice", "transaction", "receipt", "basket", "checkout", "pos", "booking",
                "shipment", "ticket", "billing", "fill"}
NOT_SALES = {"refund", "return", "cost", "tax", "discount", "budget", "target", "quota", "fee", "travel", "copay",
             "withheld", "debit", "credit", "expense", "spend", "purchase", "ordered", "freight", "shipping", "pay",
             "salary", "wage", "payroll"}
LEDGER = {"ledger", "journal", "gl", "posting", "budget", "entry", "trial"}
DOCUMENTS = {"order", "invoice", "transaction", "receipt", "basket", "ticket", "booking", "sale", "check", "bill",
             "job", "visit", "fill", "prescription", "encounter", "claim", "po", "purchase", "subscription"}
# Words of a profit and loss's own lines, on a table that is no sales table.
PNL = {"revenue", "income", "expense", "expenses", "opex", "ebitda", "profit", "operating"}
# Words of a table of purchases: never an order line of sales.
PURCHASE_TABLES = {"purchase", "po", "requisition", "procurement", "buy", "buying", "supplier", "vendor"}
UNITS = {"quantity", "unit", "qty", "volume", "piece", "item"}
COSTS = {"cost", "cogs"}
MARGINS = {"margin", "profit"}
REFUNDS = {"refund", "refunded", "credit"}
RETURNS = {"return", "returned"}
REGION = ["region", "territory", "area", "zone", "district", "state", "province", "country", "city"]
CHANNEL = ["channel", "source", "medium", "segment", "type"]
CATEGORY = ["category", "department", "family", "group", "class", "subcategory", "line", "brand", "genre"]
STORE = {"store", "branch", "shop", "outlet", "location", "site", "pharmacy", "restaurant", "dealer", "clinic",
         "reseller", "warehouse"}
SELLER = {"rep", "representative", "salesperson", "seller", "agent", "account manager"}
PRODUCT = {"product", "item", "sku", "article", "part", "drug", "formula", "service", "menu", "dish", "book",
           "track", "album", "film", "course", "material", "variant", "style", "medication"}
CUSTOMER = analyses.PARTIES
SUPPLIER = {"supplier", "vendor", "manufacturer"}
REASON = ["reason", "cause", "code", "type", "category"]


# ── what each recipe and tile is called ──────────────────────────────────────

_SAY: dict[str, dict[str, str]] = {
    "en": {
        "overview": "Overview", "when": "over the last 12 months", "last_month": "last month",
        "ytd": "{f} year to date", "this_month": "{f} last month", "quarter": "{f} over the last 3 months",
        "by_month": "{f} by month", "against_ly": "{f} by month against the year before",
        "by": "{f} by {g}", "by_ly": "{f} by {g} against the year before", "heat": "{f} by {g} and month", "series": "{f} by {g}, month by month",
        "top": "Top 10 {e} by {f}", "ranked": "{E} ranked by {f}, with change", "risers": "Biggest risers: {e}",
        "fallers": "Biggest fallers: {e}", "lowest": "Lowest {e} by {f}",
        "target": "{a} against {t} by month", "target_by": "{a} against {t} by {g}",
        "behind": "{E} furthest behind {t}",
        "lifecycle": "New, returning and lapsed {members} by month", "churn": "Churn rate of {members} by month",
        "active_members": "Active {members}",
        "cohort": "{Member} retention by month of first {thing}", "deciles": "{Members} by decile of {f}",
        "absent": "{Members} not seen for 90 days", "gap": "Average days between {things} per {member}",
        "abc": "ABC classes of {e} by {f}", "pareto": "{E} making 80% of {f}", "together": "{E} bought together",
        "slow": "Slowest-moving {e} by {f}", "on_time": "Share of {rows} on time",
        "on_time_by": "Share of {rows} on time by {g}", "late": "Share of {rows} late",
        "ageing": "Open {rows} by days past due", "active": "{Rows} at each month end",
        "active_by": "{Rows} by {g}", "starts": "New {rows} by month", "ends": "{Rows} ended by month",
        "end_rate": "{Rows} ended: rate by month", "hires": "Hires by month", "leavers": "Leavers by month",
        "attrition": "Attrition rate by month", "funnel": "{Rows} through {stages}",
        "duration": "Average days from {a} to {b}", "duration_by": "Average days from {a} to {b} by {g}",
        "name": "{name}",
    },
    "fr": {
        "overview": "Vue d'ensemble", "when": "sur les 12 derniers mois", "last_month": "le mois dernier",
        "ytd": "{f} depuis le début de l'année", "this_month": "{f} le mois dernier",
        "quarter": "{f} sur les 3 derniers mois",
        "by_month": "{f} par mois", "against_ly": "{f} par mois par rapport à l'année précédente",
        "by": "{f} par {g}", "by_ly": "{f} par {g} par rapport à l'année précédente",
        "heat": "{f} par {g} et par mois", "series": "{f} par {g}, mois par mois",
        "top": "Top 10 des {e} par {f}", "ranked": "Classement des {e} par {f}, avec l'évolution",
        "risers": "Plus fortes hausses : {e}", "fallers": "Plus fortes baisses : {e}",
        "lowest": "{E} les plus bas en {f}",
        "target": "{a} par rapport à {t} par mois", "target_by": "{a} par rapport à {t} par {g}",
        "behind": "{E} les plus en retard sur {t}",
        "lifecycle": "Nouveaux, fidèles et perdus par mois ({members})", "churn": "Taux d'attrition par mois ({members})",
        "active_members": "{Members} actifs",
        "cohort": "Rétention par mois de premier achat ({members})", "deciles": "{Members} par décile de {f}",
        "absent": "{Members} sans activité depuis 90 jours", "gap": "Jours moyens entre deux {things} par {member}",
        "abc": "Classes ABC des {e} par {f}", "pareto": "{E} qui font 80 % de {f}",
        "together": "{E} achetés ensemble", "slow": "{E} les moins vendus par {f}",
        "on_time": "Part des {rows} à l'heure", "on_time_by": "Part des {rows} à l'heure par {g}",
        "late": "Part des {rows} en retard", "ageing": "{Rows} ouverts par jours de retard",
        "active": "{Rows} en cours à chaque fin de mois", "active_by": "{Rows} par {g}",
        "starts": "Nouveaux {rows} par mois", "ends": "{Rows} terminés par mois",
        "end_rate": "Taux de fin des {rows} par mois", "hires": "Entrées par mois", "leavers": "Sorties par mois",
        "attrition": "Taux de départ par mois",
        "funnel": "{Rows} par étape : {stages}", "duration": "Jours moyens de {a} à {b}",
        "duration_by": "Jours moyens de {a} à {b} par {g}", "name": "{name}",
    },
}

# Figures worked out for a recipe, by name in each language.
_NAMED = {
    "average_order": ("Average order", "Panier moyen"),
    "margin": ("Gross margin", "Marge brute"),
    "margin_pct": ("Gross margin %", "Marge brute %"),
    "return_rate": ("Return rate", "Taux de retour"),
    "refund_share": ("Refunds as % of sales", "Remboursements en % des ventes"),
    "rejected_share": ("Rejected share", "Part rejetée"),
    "overtime_share": ("Overtime share of hours", "Part des heures supplémentaires"),
    "pay_per_head": ("Pay per head", "Rémunération par personne"),
    "profit": ("Gross profit", "Marge brute"),
    "profit_per": ("Gross profit per {thing}", "Marge brute par {thing}"),
    "revenue_per": ("Revenue per {thing}", "Chiffre d'affaires par {thing}"),
    "cost_per": ("Cost per {thing}", "Coût par {thing}"),
    "arr": ("ARR", "ARR"),
    "scrap_rate": ("Scrap rate", "Taux de rebut"),
    "paid_share": ("Paid share of billed", "Part payée du facturé"),
    "variance": ("Variance to {t}", "Écart à {t}"),
}

# What a missing piece is called, in "no ... in this data".
_PIECES = {
    "sales": ("sales with an amount and a date", "des ventes avec un montant et une date"),
    "orders": ("orders counted", "des commandes comptées"),
    "customers": ("customers on the sales", "des clients sur les ventes"),
    "products": ("products on the sales", "des produits sur les ventes"),
    "stores": ("stores or branches", "des magasins ou agences"),
    "regions": ("regions", "des régions"),
    "channels": ("channels", "des canaux"),
    "categories": ("categories", "des catégories"),
    "targets": ("targets", "des objectifs"),
    "costs": ("costs", "des coûts"),
    "returns": ("returns linked to sales", "des retours liés aux ventes"),
    "reasons": ("return reasons", "des motifs de retour"),
    "stock": ("stock on hand by item", "du stock disponible par article"),
    "reorder": ("reorder points or allocations", "des seuils de réapprovisionnement ou des allocations"),
    "warehouses": ("warehouses", "des entrepôts"),
    "movements": ("stock movements", "des mouvements de stock"),
    "purchases": ("purchase orders with due and receipt dates",
                  "des commandes d'achat avec des dates prévues et de réception"),
    "purchase_lines": ("purchase lines with an amount", "des lignes d'achat avec un montant"),
    "suppliers": ("suppliers", "des fournisseurs"),
    "rejections": ("rejected quantities", "des quantités rejetées"),
    "ledger": ("a ledger, or sales with cost", "un grand livre, ou des ventes avec leur coût"),
    "operating_costs": ("operating costs", "des charges d'exploitation"),
    "budget": ("actuals and budget", "un réel et un budget"),
    "invoices": ("invoices with due and paid dates", "des factures avec des dates d'échéance et de paiement"),
    "customer_cost": ("sales with cost by customer", "des ventes avec leur coût par client"),
    "employees": ("employees with start and end dates", "des salariés avec dates d'entrée et de sortie"),
    "headcount": ("a headcount", "des effectifs"),
    "payroll": ("payroll lines", "des lignes de paie"),
    "overtime": ("overtime", "des heures supplémentaires"),
    "departments": ("departments", "des services"),
    "subscriptions": ("subscriptions with start, end and amount",
                      "des abonnements avec début, fin et montant"),
    "plans": ("plans", "des formules"),
    "fills": ("fills or prescriptions with amounts", "des délivrances ou ordonnances avec montants"),
    "payers": ("payers", "des payeurs"),
    "prescribers": ("prescribers", "des prescripteurs"),
    "encounters": ("encounters with a date", "des consultations datées"),
    "incidents": ("incidents or tickets with open and close dates",
                  "des incidents ou tickets avec dates d'ouverture et de clôture"),
    "severity": ("a severity or priority", "une gravité ou priorité"),
    "sites": ("sites", "des sites"),
    "jobs": ("jobs with dates", "des interventions datées"),
    "technicians": ("technicians", "des techniciens"),
    "on_time": ("a due date to be on time against", "une date prévue pour mesurer la ponctualité"),
    "production": ("production records", "des relevés de production"),
    "scrap": ("scrap or waste", "des rebuts"),
    "downtime": ("downtime", "des arrêts"),
    "lines": ("production lines or plants", "des lignes ou usines"),
    "one_currency": ("figures in one currency (these are kept in several)",
                     "des montants dans une seule devise (ils sont tenus en plusieurs)"),
}

# Words of the rows a start and an end date belong to: people employed, or things subscribed to.
STAFF = {"employee", "staff", "worker", "personnel", "hire", "headcount", "contractor", "associate", "people"}
SUBSCRIBED = {"subscription", "contract", "service", "membership", "plan", "policy", "licence", "license",
              "agreement", "enrollment", "enrolment", "account", "subscriber"}
CARE = {"patient", "encounter", "admission", "clinic", "care", "diagnosis", "condition", "procedure", "provider",
        "ward", "discharge"}


def figures(model: SemanticModel, may: May) -> list[Measure]:
    """The figures a recipe may show: those the executive recipe leads with (summed or counted on rows that
    happen on a date, in one unit), and the ones Learn worked out as formulas (a ledger's revenue from its debits,
    credits and account types)."""
    out = recipes._figures(model, may)
    seen = {m.key for m in out}
    for m in model.measures.values():
        table = model.tables.get(m.table)
        if m.key in seen or m.hidden or table is None or not may(m.table) or not recipes.dated(model, m) \
                or table.kind == "dimension" or getattr(m.expr, "agg", None) is not None \
                or m.format not in ("currency", "number", "integer", "count") or recipes._mixed(model, m):
            continue
        out.append(m)
    return out


class Missing(Exception):
    """The recipe's main piece is not in this data."""

    def __init__(self, piece: str):
        super().__init__(piece)
        self.piece = piece


@dataclass(frozen=True)
class Fig:
    """A figure a tile shows: a measure of the model, or one worked out from two (an average order)."""
    name: str
    base: Measure                                   # the measure its groupings and lists are reached from
    measures: tuple[str, ...] = ()
    derived: tuple[tuple[tuple[str, Any], ...], ...] = ()
    lower_better: bool = False

    @property
    def sort(self) -> str:
        return self.measures[0] if self.measures else dict(self.derived[-1])["name"]

    def part(self) -> dict[str, Any]:
        out: dict[str, Any] = {"measures": list(self.measures)}
        if self.derived:
            out["derived"] = [dict(d) for d in self.derived]
        return out

    @property
    def tables(self) -> set[str]:
        return {self.base.table}


_STAMP = {"timestamp", "date", "time", "datetime", "ts", "dt", "at", "on", "day"}


def _event(role: DateRole) -> str:
    """A date by the event it marks: "opened" for an opened timestamp, "order" for an order date."""
    words = [w for w in (role.name or role.slug).lower().split() if w not in _STAMP]
    return " ".join(words) or (role.name or role.slug).lower()


def low(name: str) -> str:
    """A figure's name inside a sentence: lower case, but an acronym kept (MRR, FTE)."""
    return " ".join(w if w.isupper() and len(w) > 1 else w.lower() for w in name.split())


def _derived(name: str, op: str, a: str, b: str, scale: float = 1.0) -> tuple[tuple[str, Any], ...]:
    return tuple({"name": name, "op": op, "measures": [a, b], "scale": scale}.items())


class Kit:
    """What a recipe builds its tiles with: the model's figures, groupings and lists found by their words, and a
    tile kept only when the resolver answers its plan."""

    def __init__(self, model: SemanticModel, *, today: dt.date, lang: str, allowed: set[str] | None):
        self.model, self.today, self.lang = model, today, lang if lang in _SAY else "en"
        self.allowed = allowed
        self.say = _SAY[self.lang]
        self.ctx = Context(today=today, allowed_tables=allowed)
        self.figures = figures(model, self.may)
        self.rows = {m.table: model.tables[m.table].row_count or 0 for m in self.figures}
        self.made: list[Made] = []
        self.left_out: list[str] = []          # pieces a tile wanted and the data did not have
        self.instead: list[str] = []           # tiles drawn in the place of one the data could not give
        self.used: list[str] = []              # tables the tiles read
        self.tab = self.say["overview"]
        self._answers: dict[str, bool] = {}
        self._shown: set[str] = set()

    # ── reading the model ───────────────────────────────────────────────────

    def may(self, table: str) -> bool:
        return self.allowed is None or table in self.allowed

    def t(self, en: str, fr: str) -> str:
        return fr if self.lang == "fr" else en

    def named(self, key: str, **kw: str) -> str:
        en, fr = _NAMED[key]
        return (fr if self.lang == "fr" else en).format(**kw)

    def piece(self, key: str) -> str:
        en, fr = _PIECES.get(key, (key, key))
        return fr if self.lang == "fr" else en

    def measure(self, words: set[str], *, want: str | None = None, table: str | None = None,
                avoid: set[str] = frozenset(), counts: bool = False, near: str | None = None) -> Measure | None:
        """The figure named with ``words`` (most of them, named most plainly), measured as ``want`` says
        (currency or count); on ``table`` when given, else on ``near``'s table first."""
        fits = []
        for m in self.figures:
            w = _words(m.business_name)
            agg = getattr(m.expr, "agg", "")
            if not w & words or w & avoid or table is not None and m.table != table:
                continue
            if counts != (agg in ("count", "count_distinct")) and m.kind != "metric":
                continue
            if want == "currency" and m.format != "currency" or want == "number" and m.format == "currency":
                continue
            fits.append(m)
        if not fits:
            return None
        return min(fits, key=lambda m: (near is not None and m.table != near, m.kind != "metric",
                                        -len(_words(m.business_name) & words),
                                        getattr(m.expr, "agg", "") != "count_distinct",
                                        len(names.tokens(m.business_name)), -self.rows.get(m.table, 0), m.slug))

    def fig(self, m: Measure | None, *, lower_better: bool = False) -> Fig | None:
        return None if m is None else Fig(m.business_name, m, (m.slug,), lower_better=lower_better)

    def worked(self, key: str, a: Fig | None, b: Fig | None, op: str, scale: float = 1.0, *,
               inner: Fig | None = None, lower_better: bool = False, **kw: str) -> Fig | None:
        """A figure worked out from two (``inner``: one worked out first, kept with it)."""
        if a is None or b is None:
            return None
        name = self.named(key, **kw)
        derived = (*(inner.derived if inner is not None else ()),
                   _derived(name, op, a.sort, b.sort, scale))
        return Fig(name, a.base, (), derived, lower_better)

    def table_words(self, table: str) -> set[str]:
        t = self.model.tables.get(table)
        return _words(t.business_name or t.name) if t is not None else set()

    def entities(self, words: set[str], *, people: bool = True, fewest: int = 3) -> list[Entity]:
        """Lists named with ``words`` a reader may see, the most members first; ``people``: also one named by a
        person's name (technicians, prescribers), never by a contact detail."""
        found = []
        for e in self.model.entities.values():
            if e.role or not self.may(e.table) or (e.members or 0) < fewest or not e.label_column:
                continue
            label = self.model.columns.get(e.label_column)
            if label is None or label.personal == "detail" or not people and label.personal != "none":
                continue
            ew = _words(e.business_name or e.slug)
            if ew & words:
                found.append((-(e.members or 0), e.slug, e))
        return [f[2] for f in sorted(found, key=lambda f: f[:2])]

    def groupings(self, fig: Fig, words: list[str]) -> list[Attribute]:
        """Groupings of a few members the figure may be split by, named with one of ``words`` (the first first)."""
        out = []
        for a in recipes.groupings(self.model, fig.base, recipes.GENERIC, self.may):
            e = self.model.entities.get(a.entity) if a.entity else None
            w = _words(recipes._named_by(self.model, a)) | (_words(e.business_name) if e is not None else set())
            hit = next((i for i, g in enumerate(words) if g in w), None)
            if hit is not None:
                out.append((hit, a))
        return [a for _, a in sorted(out, key=lambda x: x[0])]

    def date(self, words: set[str], table: str | None = None) -> DateRole | None:
        found = [r for r in self.model.date_roles.values() if self.may(r.table) and r.kind in ("event", "planned", "due",
                                                                                              "validity")
                 and set(names.tokens(r.name or r.slug)) & words and (table is None or r.table == table)
                 and r.granularity != "month"]
        found.sort(key=lambda r: (not r.is_default, -(self.model.tables[r.table].row_count or 0), r.slug))
        return found[0] if found else None

    # ── tiles ───────────────────────────────────────────────────────────────

    def answers(self, plan: dict[str, Any]) -> bool:
        key = json.dumps(plan, sort_keys=True, default=str)
        if key not in self._answers:
            try:
                resolve(Plan.model_validate(plan), self.model, self.ctx)
                self._answers[key] = True
            except (ResolveError, ValueError):
                self._answers[key] = False
        return self._answers[key]

    def add(self, title: str, kind: str, plan: dict[str, Any], *, when: str | None = None) -> bool:
        if len(self.made) >= MOST_TILES or title.casefold() in self._shown:
            return False
        plan = {"kind": "query", **plan}
        if not self.answers(plan):
            return False
        self._shown.add(title.casefold())
        question = f"{title} {when if when is not None else self.say['when']}".strip()
        self.made.append(Made(title, kind, question, plan, self.tab))
        for slug in plan.get("measures") or []:
            m = self.model.measures.get(slug) or next((x for x in self.model.measures.values() if x.slug == slug), None)
            if m is not None and m.table not in self.used:
                self.used.append(m.table)
        return True

    def lack(self, piece: str) -> bool:
        if piece and piece not in self.left_out:
            self.left_out.append(piece)
        return False

    def number(self, fig: Fig | None, window: dict | None = None, *, title: str | None = None,
               when: str | None = None, piece: str = "") -> bool:
        if fig is None:
            return self.lack(piece) if piece else False
        return self.add(title or fig.name, "number", {"intent": "value", **fig.part(),
                                                      "time": {"window": window or YEAR}}, when=when)

    def trend(self, fig: Fig | None, *, last_year: bool = False, window: dict | None = None, title: str | None = None,
              piece: str = "") -> bool:
        if fig is None:
            return self.lack(piece) if piece else False
        time: dict[str, Any] = {"grain": "month", "window": window or YEAR}
        if last_year and self.add(title or self.say["against_ly"].format(f=fig.name), "trend",
                                  {"intent": "trend", **fig.part(),
                                   "time": {**time, "compare": {"kind": "same_period_last_year"}}}):
            return True
        return self.add(title or self.say["by_month"].format(f=fig.name), "trend",
                        {"intent": "trend", **fig.part(), "time": time})

    def by(self, fig: Fig | None, words: list[str], piece: str, *, window: dict | None = None,
           title: str | None = None, desc: bool = True, lower_first: bool = False, limit: int | None = None,
           compare: str | None = None) -> bool:
        if fig is None:
            return False
        for a in self.groupings(fig, words):
            g = low(recipes._named_by(self.model, a))
            plan: dict[str, Any] = {"intent": "breakdown", **fig.part(), "group_by": [a.slug],
                                    "sort": [{"by": fig.sort, "desc": desc and not lower_first}],
                                    "time": {"window": window or YEAR}}
            if limit:
                plan["limit"] = limit
            if compare:
                plan["time"]["compare"] = {"kind": compare}
            if self.add((title or self.say["by"]).format(f=fig.name, g=g, G=g.capitalize()), "breakdown", plan):
                return True
        return self.lack(piece)

    def heat(self, fig: Fig | None, words: list[str], piece: str) -> bool:
        if fig is None:
            return False
        for a in self.groupings(fig, words):
            g = low(recipes._named_by(self.model, a))
            if self.add(self.say["heat"].format(f=fig.name, g=g), "breakdown",
                        {"intent": "trend", **fig.part(), "group_by": [a.slug], "chart": "heatmap",
                         "time": {"grain": "month", "window": YEAR}}):
                return True
        return self.lack(piece)

    def series(self, fig: Fig | None, words: list[str], piece: str) -> bool:
        if fig is None:
            return False
        for a in self.groupings(fig, words):
            g = low(recipes._named_by(self.model, a))
            if self.add(self.say["series"].format(f=fig.name, g=g), "trend",
                        {"intent": "trend", **fig.part(), "group_by": [a.slug],
                         "time": {"grain": "month", "window": YEAR}}):
                return True
        return self.lack(piece)

    def _many(self, e: Entity) -> str:
        return names.plural(low(e.business_name or e.slug))

    def top(self, fig: Fig | None, words: set[str], piece: str, *, n: int = 10, people: bool = True,
            lowest: bool = False, window: dict | None = None) -> bool:
        if fig is None:
            return False
        for e in self.entities(words, people=people):
            many = self._many(e)
            title = (self.say["lowest"] if lowest else self.say["top"]).format(
                e=many, E=many.capitalize(), f=low(fig.name) if self.lang == "en" else fig.name)
            if self.add(title, "table", {"intent": "rank", **fig.part(), "group_by": [e.slug], "limit": n,
                                         "sort": [{"by": fig.sort, "desc": not lowest}],
                                         "time": {"window": window or YEAR}}):
                return True
        return self.lack(piece)

    def ranked(self, fig: Fig | None, words: set[str], piece: str) -> bool:
        """Every member ranked by the figure, with its change on the 12 months before."""
        if fig is None:
            return False
        for e in self.entities(words):
            many = self._many(e)
            if self.add(self.say["ranked"].format(e=many, E=many.capitalize(), f=low(fig.name)), "table",
                        {"intent": "rank", **fig.part(), "group_by": [e.slug], "limit": 50,
                         "sort": [{"by": fig.sort, "desc": True}],
                         "time": {"window": YEAR, "compare": {"kind": "previous_period"}}}):
                return True
        return self.lack(piece)

    def movers(self, fig: Fig | None, words: set[str], piece: str, *, rising: bool) -> bool:
        """The members whose figure moved most over the last 3 months against the 3 before, up or down."""
        if fig is None or not fig.measures:
            return False
        for e in self.entities(words):
            many = self._many(e)
            title = self.say["risers" if rising else "fallers"].format(e=many)
            if self.add(title, "table", {"intent": "rank", **fig.part(), "group_by": [e.slug], "limit": 5,
                                         "sort": [{"by": "change", "desc": rising}],
                                         "time": {"window": QUARTER, "compare": {"kind": "previous_period"}}},
                        when=self.t("over the last 3 months", "sur les 3 derniers mois")):
                return True
        return self.lack(piece)

    def target(self, *, words: set[str] | None = None, groups: list[str] | None = None, behind: bool = False,
               of: Fig | None = None) -> bool:
        """A figure against its target: by month, by a list's members (the furthest behind first), or by a
        grouping."""
        pair = analyses.target_pair(self.model, self.may)
        if pair is None or of is not None and pair[0].slug not in of.measures:
            self.lack("targets")
            return self._against_last_year(of, words=words, groups=groups, behind=behind)
        actual, target = pair
        a, t = actual.business_name, low(target.business_name)
        part = {"intent": "trend" if words is None and groups is None else "breakdown",
                "target": {"actual": actual.slug, "target": target.slug}}
        if words is None and groups is None:
            return self.add(self.say["target"].format(a=a, t=t), "trend",
                            {**part, "time": {"grain": "month", "window": YEAR}})
        if words is not None:
            for e in self.entities(words):
                many = self._many(e)
                title = (self.say["behind"] if behind else self.say["target_by"]).format(
                    a=a, t=t, g=names.singular(many), E=many.capitalize())
                plan = {**part, "group_by": [e.slug], "time": {"window": TO_DATE},
                        "sort": [{"by": "of_target", "desc": not behind}]}
                if behind:
                    plan["limit"] = 5
                if self.add(title, "table", plan, when=self.t("this year to date", "depuis le début de l'année")):
                    return True
        for g in self.groupings(self.fig(actual), groups or []):
            name = low(recipes._named_by(self.model, g))
            if self.add(self.say["target_by"].format(a=a, t=t, g=name), "breakdown",
                        {**part, "group_by": [g.slug], "time": {"window": TO_DATE}},
                        when=self.t("this year to date", "depuis le début de l'année")):
                return True
        return False

    def _against_last_year(self, of: Fig | None, *, words: set[str] | None, groups: list[str] | None,
                           behind: bool) -> bool:
        """No target to measure ``of`` against: the year before is, in the same view (by month; each store ranked
        with its change; the biggest fallers in the place of those furthest behind; a grouping against the year
        before). The tile drawn in its place is noted in ``instead``; nothing when it is on the dashboard
        already."""
        if of is None:
            return False
        made = len(self.made)
        if words is None and groups is None:
            # Against the year before only: a plain "by month" is no stand-in for a target, and is often there.
            self.add(self.say["against_ly"].format(f=of.name), "trend",
                     {"intent": "trend", **of.part(), "time": {"grain": "month", "window": YEAR,
                                                               "compare": {"kind": "same_period_last_year"}}})
        elif words is not None and behind:
            self.movers(of, words, "", rising=False)
        elif words is not None:
            self.ranked(of, words, "")
        else:
            self.by(of, groups or [], "", compare="same_period_last_year", title=self.say["by_ly"])
        if len(self.made) == made:
            return False
        self.instead.append(self.made[-1].title)
        return True

    # ── analyses Learn found ────────────────────────────────────────────────

    def activity(self) -> analyses.Activity | None:
        return analyses.activity(self.model, self.may)

    def _who(self, act: analyses.Activity) -> dict[str, str]:
        members = names.plural(act.member)
        return {"members": members, "Members": members.capitalize(), "member": act.member,
                "Member": act.member.capitalize(), "things": act.things, "thing": names.singular(act.things)}

    def lifecycle(self, status: str = "all", *, number: bool = False) -> bool:
        """Members by their activity each month (new, returning, won back, lapsed), the churn rate, or (``number``)
        how many were active over the period."""
        act = self.activity()
        if act is None:
            return self.lack("customers")
        w = self._who(act)
        of = {"of": act.entity.slug, "measure": act.measure.slug}
        if number:
            # Customers with orders in the period: their count, counted by what they did.
            counted = next((m for m in self.model.measures.values() if m.table == act.entity.table and not m.hidden
                            and getattr(m.expr, "agg", "") == "count_distinct"), None)
            return counted is not None and self.add(self.say["active_members"].format(**w), "number",
                                                    {"intent": "value", "measures": [counted.slug],
                                                     "time": {"window": YEAR}})
        if status == "churn_rate":
            return self.add(self.say["churn"].format(**w), "trend",
                            {"intent": "trend", "lifecycle": {**of, "status": "churn_rate"},
                             "time": {"grain": "month", "window": YEAR}})
        return self.add(self.say["lifecycle"].format(**w), "trend",
                        {"intent": "trend", "lifecycle": of, "time": {"grain": "month", "window": YEAR}})

    def cohort(self) -> bool:
        act = self.activity()
        if act is None:
            return self.lack("customers")
        # Every month's newcomers, the grid's own window: a year's may hold none (all signed up at the start).
        return self.add(self.say["cohort"].format(**self._who(act)), "breakdown",
                        {"intent": "breakdown", "cohort": {"of": act.entity.slug, "measure": act.measure.slug,
                                                           "periods": 6}, "time": {"grain": "month"}}, when="")

    def gap(self) -> bool:
        act = self.activity()
        if act is None:
            return self.lack("customers")
        return self.add(self.say["gap"].format(**self._who(act)), "number",
                        {"intent": "value", "gap": {"of": act.entity.slug, "measure": act.measure.slug},
                         "time": {"window": YEAR}})

    def absent(self) -> bool:
        act = self.activity()
        if act is None:
            return self.lack("customers")
        return self.add(self.say["absent"].format(**self._who(act)), "table",
                        {"intent": "list", "absent": {"of": act.entity.slug, "measure": act.measure.slug,
                                                      "active_before": True},
                         "time": {"window": {"kind": "last", "unit": "day", "n": QUIET_DAYS}}}, when="")

    def deciles(self, fig: Fig | None) -> bool:
        act = self.activity()
        if act is None or fig is None or not fig.measures:
            return self.lack("customers")
        return self.add(self.say["deciles"].format(f=low(fig.name), **self._who(act)), "breakdown",
                        {"intent": "rank", "pareto": {"of": act.entity.slug, "measure": fig.sort, "mode": "ntile",
                                                      "cuts": [10]}, "time": {"window": YEAR}})

    def abc(self, fig: Fig | None, words: set[str], piece: str, *, pareto: bool = False) -> bool:
        if fig is None or not fig.measures:
            return False
        for e in self.entities(words, fewest=11):
            many = self._many(e)
            key = "pareto" if pareto else "abc"
            mode = {"mode": "pareto", "cuts": [80]} if pareto else {"mode": "abc"}
            if self.add(self.say[key].format(e=many, E=many.capitalize(), f=low(fig.name)),
                        "table" if pareto else "breakdown",
                        {"intent": "rank", "pareto": {"of": e.slug, "measure": fig.sort, **mode},
                         "time": {"window": YEAR}}):
                return True
        return self.lack(piece)

    def together(self) -> bool:
        found = analyses.basket(self.model, self.may)
        if found is None:
            return False
        item, count = found
        many = names.plural(item.business_name)
        return self.add(self.say["together"].format(E=many), "table",
                        {"intent": "rank", "together": {"of": item.slug, "basket": count.slug}, "limit": 10,
                         "time": {"window": YEAR}})

    def slow(self, fig: Fig | None, words: set[str], piece: str) -> bool:
        """The slowest movers: the members with the least of the figure over the last 90 days."""
        if fig is None:
            return False
        for e in self.entities(words, fewest=11):
            many = self._many(e)
            if self.add(self.say["slow"].format(E=many.capitalize(), e=many, f=low(fig.name)), "table",
                        {"intent": "rank", **fig.part(), "group_by": [e.slug], "limit": 10,
                         "sort": [{"by": fig.sort, "desc": False}],
                         "time": {"window": {"kind": "last", "unit": "day", "n": QUIET_DAYS}}},
                        when=self.t("over the last 90 days", "sur les 90 derniers jours")):
                return True
        return self.lack(piece)

    def due(self) -> tuple[DateRole, DateRole] | None:
        return analyses.due_and_done(self.model, self.may)

    def _rows(self, role: DateRole) -> str:
        t = self.model.tables[role.table]
        return names.plural((t.business_name or t.name).lower())

    def on_time(self, *, groups: list[str] | None = None, entity: set[str] | None = None, late: bool = False,
                ageing: bool = False) -> bool:
        pair = self.due()
        if pair is None:
            return self.lack("on_time")
        due, done = pair
        rows = self._rows(done)
        on = {"done": done.slug, "due": due.slug}
        if ageing:
            rows_due = self._rows(due)
            return self.add(self.say["ageing"].format(Rows=rows_due.capitalize(), rows=rows_due), "breakdown",
                            {"intent": "breakdown", "on_time": {**on, "ageing": True}, "time": {}}, when="")
        if groups is None and entity is None:
            key = "late" if late else "on_time"
            return self.add(self.say[key].format(rows=rows), "number",
                            {"intent": "value", "on_time": {**on, "late": late}, "time": {"window": YEAR}})
        if entity is not None:
            for e in self.entities(entity):
                g = low(e.business_name or e.slug)
                if self.add(self.say["on_time_by"].format(rows=rows, g=g), "breakdown",
                            {"intent": "breakdown", "on_time": on, "group_by": [e.slug], "time": {"window": YEAR}}):
                    return True
        home = self._date_measure(done)
        for a in recipes.groupings(self.model, home, recipes.GENERIC, self.may) if groups and home else []:
            g = low(recipes._named_by(self.model, a))
            if not any(w in _words(g) for w in groups or []):
                continue
            if self.add(self.say["on_time_by"].format(rows=rows, g=g), "breakdown",
                        {"intent": "breakdown", "on_time": on, "group_by": [a.slug], "time": {"window": YEAR}}):
                return True
        return False

    def _date_measure(self, role: DateRole) -> Measure | None:
        return next((m for m in self.figures if m.table == role.table), None)

    def active(self, status: str = "active", *, groups: list[str] | None = None, number: bool = False,
               of: set[str] = STAFF | SUBSCRIBED) -> bool:
        pair = self.life(of)
        if pair is None:
            return False
        start, end = pair
        rows = self._rows(start)
        part = {"start": start.slug, "end": end.slug, "status": status}
        if groups:
            m = self._date_measure(start)
            for a in recipes.groupings(self.model, m, recipes.GENERIC, self.may) if m is not None else []:
                g = low(recipes._named_by(self.model, a))
                if not any(w in _words(g) for w in groups):
                    continue
                if self.add(self.say["active_by"].format(Rows=rows.capitalize(), g=g), "breakdown",
                            {"intent": "breakdown", "active": part, "group_by": [a.slug],
                             "time": {"window": MONTH}}, when=self.t("at the last month end",
                                                                      "à la dernière fin de mois")):
                    return True
            return False
        staff = bool(self.table_words(start.table) & STAFF)
        words = {"rows": rows, "Rows": rows.capitalize()}
        title = {"active": self.say["active"].format(**words),
                 "starts": self.say["hires" if staff else "starts"].format(**words),
                 "ends": self.say["leavers" if staff else "ends"].format(**words),
                 "end_rate": self.say["attrition" if staff else "end_rate"].format(**words)}[status]
        if number:
            return self.add(rows.capitalize() if status == "active" else title, "number",
                            {"intent": "value", "active": part, "time": {"window": MONTH}},
                            when=self.t("at the last month end", "à la dernière fin de mois"))
        return self.add(title, "trend", {"intent": "trend", "active": part,
                                         "time": {"grain": "month", "window": YEAR}})

    def life(self, words: set[str]) -> tuple[DateRole, DateRole] | None:
        """The start and end dates of rows named with ``words`` (employees' hire and termination, subscriptions'
        start and end)."""
        pair = analyses.start_and_end(self.model, self.may)
        return pair if pair is not None and self.table_words(pair[0].table) & words else None

    def mixed(self, words: set[str]) -> bool:
        """Is there a figure named with ``words`` the recipes leave out only because it is kept in several
        currencies or units?"""
        return any(not m.hidden and self.may(m.table) and _words(m.business_name) & words
                   and recipes._mixed(self.model, m) for m in self.model.measures.values())

    def funnel(self) -> bool:
        stages = analyses.funnel_dates(self.model, self.may)
        if len(stages) < 2:
            return False
        rows = self._rows(stages[0])
        listed = [(r.name or r.slug).lower() for r in stages]
        said = ", ".join(listed[:-1]) + self.t(" and ", " et ") + listed[-1]
        return self.add(self.say["funnel"].format(Rows=rows.capitalize(), stages=said), "breakdown",
                        {"intent": "breakdown", "funnel": {"stages": [r.slug for r in stages]},
                         "time": {"window": YEAR}})

    def duration(self, start_words: set[str], end_words: set[str], *, groups: list[str] | None = None,
                 entity: set[str] | None = None) -> bool:
        """Average days between two dates of the same rows (opened to resolved, ordered to received)."""
        for start in sorted((r for r in self.model.date_roles.values() if self.may(r.table)
                             and set(names.tokens(r.name or r.slug)) & start_words and r.granularity != "month"),
                            key=lambda r: (not r.is_default, -(self.model.tables[r.table].row_count or 0), r.slug)):
            end = self.date(end_words, start.table)
            if end is None or end.key == start.key:
                continue
            a, b = _event(start), _event(end)
            name = self.say["duration"].format(a=a, b=b)
            d = {"name": name[:1].upper() + name[1:], "start": start.slug, "end": end.slug}
            plan: dict[str, Any] = {"intent": "value", "durations": [d], "time": {"window": YEAR}}
            if entity is not None or groups is not None:
                splits: list[tuple[str, str]] = []
                for e in self.entities(entity or set()):
                    splits.append((e.slug, low(e.business_name or e.slug)))
                m = self._date_measure(start)
                if groups and m is not None:
                    for at in self.groupings(self.fig(m), groups):    # type: ignore[arg-type]
                        splits.append((at.slug, low(recipes._named_by(self.model, at))))
                for slug, g in splits:
                    if self.add(self.say["duration_by"].format(a=a, b=b, g=g), "breakdown",
                                {**plan, "intent": "breakdown", "group_by": [slug],
                                 "sort": [{"by": d["name"], "desc": True}]}):
                        return True
                return False
            return self.add(d["name"], "number", plan)
        return False

    def tab_named(self, en: str, fr: str) -> None:
        self.tab = self.t(en, fr)


# ── the recipes ─────────────────────────────────────────────────────────────

def _sales_like(k: Kit, m: Measure) -> bool:
    """Named as sales ("revenue", "sales amount"), or money ("net amount") on a table of sales or of customers'
    payments (a rental shop's); never on a ledger's or a purchase's rows."""
    table = k.table_words(m.table)
    if table & (LEDGER | PURCHASE_TABLES):
        return False
    return bool(_words(m.business_name) & SOLD or _of_sales(k, m))


def _of_sales(k: Kit, m: Measure) -> bool:
    """On a table of sales, or of customers' payments (a rental shop's takings)."""
    table = k.table_words(m.table)
    return bool(table & SALES_TABLES or table & {"payment"} and _customers_pay(k, m))


def _customers_pay(k: Kit, m: Measure) -> bool:
    """Each of the table's rows names one customer: payments received, not paid out."""
    homes = {e.table for e in k.entities(CUSTOMER)}
    return any(j.from_table == m.table and j.to_table in homes and j.cardinality == "many_to_one"
               and j.trust != "rejected" for j in k.model.joins.values())


def _on_lines(k: Kit, fits: list[Measure]) -> list[Measure]:
    """Of an order's total and its lines' values, the lines': they reach all the order reaches, and its products."""
    lines = [m for m in fits if any(other.table in _documents(k.model, m.table) for other in fits)]
    headers = {d for m in lines for d in _documents(k.model, m.table)}
    return [m for m in fits if m.table not in headers] or fits


def _sales(k: Kit) -> Fig:
    """The sales figure: money named as sales (net before gross) on rows of sales, never a ledger's, a purchase's
    or a payslip's."""
    fits = [m for m in k.figures if m.format == "currency" and _words(m.business_name) & SALES
            and not _words(m.business_name) & NOT_SALES and _sales_like(k, m)]
    if not fits:
        fits = [m for m in k.figures if m.format == "currency" and _of_sales(k, m)
                and _sales_like(k, m) and not _words(m.business_name) & NOT_SALES
                and _words(m.business_name) & {"amount", "total", "value", "price"}]
    if not fits:
        raise Missing("sales")
    fits = _on_lines(k, fits)
    best = k.measure(set().union(*(_words(m.business_name) for m in fits)) & (SALES | {"amount", "total", "value",
                                                                                     "price"}),
                     want="currency", avoid=NOT_SALES)
    if best is None or best not in fits:
        best = fits[0]
    if not _words(best.business_name) & (SALES | SOLD):          # "Amount" on payments, "Value at unit price"
        return Fig(k.t("Sales", "Ventes"), best, (best.slug,))
    return k.fig(best)  # type: ignore[return-value]


def _sales_parts(k: Kit) -> dict[str, Fig | None]:
    sales = _sales(k)
    home = sales.base.table
    # Orders are counted on the orders, not on their lines: an order line's order is the document it belongs to.
    docs = _documents(k.model, home)
    orders = k.fig(k.measure(DOCUMENTS, counts=True, table=docs[0])) if len(docs) == 1 else None
    orders = orders or k.fig(k.measure(DOCUMENTS, counts=True, table=home))    # never another table's tickets
    units = k.fig(k.measure(UNITS, want="number", table=home, avoid=RETURNS | {"rejected", "received"}))
    cost = k.fig(k.measure(COSTS, want="currency", table=home, avoid={"unit", "travel", "shipping", "freight"}))
    margin_m = k.measure(MARGINS, want="currency", table=home)
    margin = k.fig(margin_m) if margin_m is not None else k.worked("margin", sales, cost, "difference")
    margin_pct = k.worked("margin_pct", margin, sales, "ratio", 100, inner=margin if margin is not None and margin.derived
                          else None)
    avg = k.worked("average_order", sales, orders, "ratio") \
        if orders is not None and orders.base.table in (home, *docs) else None
    refunds = k.fig(k.measure(REFUNDS | RETURNS, want="currency", avoid={"tax"}), lower_better=True)
    returns = k.fig(k.measure(RETURNS, counts=True), lower_better=True)
    return_units = k.fig(k.measure(RETURNS, want="number", avoid={"amount", "value"}), lower_better=True)
    if return_units is not None and units is not None:
        return_rate = k.worked("return_rate", return_units, units, "ratio", 100, lower_better=True)
    elif returns is not None and orders is not None:
        return_rate = k.worked("return_rate", returns, orders, "ratio", 100, lower_better=True)
    else:
        return_rate = None
    return {"sales": sales, "orders": orders, "units": units, "cost": cost, "margin": margin,
            "margin_pct": margin_pct, "average_order": avg, "refunds": refunds, "returns": returns,
            "return_units": return_units, "return_rate": return_rate}


def sales_overview(k: Kit) -> None:
    p = _sales_parts(k)
    s = p["sales"]
    k.number(s)
    k.number(p["orders"], piece="orders")
    k.number(p["average_order"])
    k.number(p["margin_pct"], piece="costs")
    k.trend(s, last_year=True)
    k.lifecycle()
    k.by(s, REGION, "regions")
    k.by(s, CHANNEL, "channels")
    k.tab_named("Customers and products", "Clients et produits")
    k.top(s, CUSTOMER, "customers")
    k.top(s, PRODUCT, "products", people=False)
    k.by(s, CATEGORY, "categories")


def sales_performance(k: Kit) -> None:
    p = _sales_parts(k)
    s = p["sales"]
    k.number(s)
    k.number(p["orders"], piece="orders")
    k.number(p["units"])
    k.number(p["average_order"])
    k.number(s, TO_DATE, title=k.say["ytd"].format(f=s.name), when="")
    k.trend(s, last_year=True)
    k.heat(s, REGION, "regions")
    k.by(s, CHANNEL, "channels")
    k.tab_named("Stores and products", "Magasins et produits")
    k.ranked(s, STORE | SELLER, "stores")
    k.target(words=STORE | SELLER, of=s) or k.target(of=s)
    k.movers(s, PRODUCT, "products", rising=True)
    k.movers(s, PRODUCT, "products", rising=False)


def customer_health(k: Kit) -> None:
    p = _sales_parts(k)
    if k.activity() is None:
        raise Missing("customers")
    k.lifecycle("active", number=True)
    k.gap()
    k.lifecycle()
    k.lifecycle("churn_rate")
    k.cohort()
    k.deciles(p["sales"])
    k.absent()
    k.top(p["sales"], CUSTOMER, "customers")


def product_and_category(k: Kit) -> None:
    p = _sales_parts(k)
    s = p["sales"]
    if not k.entities(PRODUCT, people=False, fewest=11):
        raise Missing("products")
    k.number(s)
    k.number(p["margin_pct"], piece="costs")
    k.number(p["return_rate"], piece="returns")
    k.series(s, CATEGORY, "categories")
    k.by(s, CATEGORY, "categories", compare="same_period_last_year")
    k.abc(s, PRODUCT, "products")
    k.by(p["margin_pct"], CATEGORY, "costs") if p["margin_pct"] is not None else k.lack("costs")
    k.together()
    k.top(p["return_rate"], PRODUCT, "returns", people=False) if p["return_rate"] is not None else k.lack("returns")
    k.slow(p["units"] or s, PRODUCT, "products")
    k.top(s, PRODUCT, "products", people=False)


def store_scorecard(k: Kit) -> None:
    p = _sales_parts(k)
    s = p["sales"]
    stores = k.entities(STORE)
    if not stores:
        raise Missing("stores")
    k.ranked(s, STORE, "stores")
    k.target(words=STORE, of=s)
    k.top(p["average_order"], STORE, "orders") if p["average_order"] is not None else k.lack("orders")
    k.top(p["return_rate"], STORE, "returns") if p["return_rate"] is not None else k.lack("returns")
    k.movers(s, STORE, "stores", rising=True)
    k.movers(s, STORE, "stores", rising=False)
    k.target(words=STORE, behind=True, of=s)
    # Each store beside the others of its region: its region's stores side by side.
    for g in k.groupings(s, REGION):
        for e in stores:
            name = low(recipes._named_by(k.model, g))
            title = k.t(f"{s.name} by {name} and {low(e.business_name or e.slug)}",
                        f"{s.name} par {name} et {low(e.business_name or e.slug)}")
            if k.add(title, "breakdown", {"intent": "breakdown", **s.part(), "group_by": [g.slug, e.slug],
                                          "sort": [{"by": s.sort, "desc": True}], "time": {"window": YEAR}}):
                return
    k.lack("regions")


def returns_and_quality(k: Kit) -> None:
    p = _sales_parts(k)
    if p["returns"] is None and p["refunds"] is None and p["return_units"] is None:
        raise Missing("returns")
    k.number(p["refunds"])
    k.number(p["returns"])
    k.number(p["return_rate"])
    k.trend(p["return_rate"]) or k.trend(p["returns"] or p["refunds"])
    k.trend(p["refunds"])
    base = p["returns"] or p["refunds"] or p["return_units"]
    k.by(base, REASON, "reasons")
    k.by(p["return_rate"] or base, CATEGORY, "categories")
    k.top(p["return_rate"] or base, PRODUCT, "products", people=False)
    k.top(p["return_rate"] or base, STORE, "stores")
    k.duration({"order", "sale", "sold", "purchase", "bought", "invoice"}, {"return", "returned", "refund"})


def monthly_review(k: Kit) -> None:
    p = _sales_parts(k)
    s = p["sales"]
    last = k.say["last_month"]
    k.number(s, MONTH, when=last)
    k.number(p["orders"], MONTH, when=last)
    k.number(p["average_order"], MONTH, when=last)
    k.number(p["margin_pct"], MONTH, when=last)
    k.trend(s, last_year=True)
    k.by(s, REGION, "regions", window=MONTH, compare="same_period_last_year")
    k.by(s, CATEGORY, "categories", window=MONTH, compare="same_period_last_year")
    k.movers(s, PRODUCT, "products", rising=False)
    k.target(words=STORE | SELLER, behind=True, of=s)
    k.absent()


# Inventory and supply

STOCK = {"hand", "onhand", "stock", "available", "closing", "inventory", "balance"}
VALUE = {"value", "valuation", "cost", "amount"}


def _stock(k: Kit) -> tuple[Fig | None, Fig | None]:
    """Stock on hand: its quantity (where it is kept in one unit) and its value; one of them at least."""
    qty = k.measure(STOCK, want="number", avoid={"allocated", "reserved", "movement", "reorder"})
    value = k.measure(STOCK, want="currency", near=qty.table if qty is not None else None)
    if value is not None and not _words(value.business_name) & VALUE:
        value = None
    if qty is None and value is None:
        raise Missing("stock")
    return k.fig(qty), k.fig(value)


def inventory_health(k: Kit) -> None:
    qty, value = _stock(k)
    k.number(value)
    k.number(qty)
    k.trend(value or qty)
    k.by(value or qty, ["warehouse", "location", "site", "store", "region"], "warehouses")
    k.by(value or qty, CATEGORY, "categories")
    k.abc(value or qty, PRODUCT, "products")
    k.top(value or qty, PRODUCT, "products", people=False)
    moved = k.fig(k.measure({"movement", "moved", "quantity", "issued", "sold"}, want="number",
                            avoid=STOCK | {"allocated"}) or k.measure({"movement", "transaction", "issue"},
                                                                      counts=True))
    k.slow(moved, PRODUCT, "movements") if moved is not None else k.lack("movements")
    k.trend(moved) if moved is not None else None


def availability(k: Kit) -> None:
    qty, value = _stock(k)
    allocated = k.fig(k.measure({"allocated", "reserved", "committed"}, want="number"))
    if allocated is None:
        raise Missing("reorder")
    qty = qty or value
    k.number(qty)
    k.number(allocated)
    k.trend(allocated)
    k.by(allocated, ["warehouse", "location", "site", "store"], "warehouses")
    k.by(qty, ["warehouse", "location", "site", "store"], "warehouses")
    k.top(allocated, PRODUCT, "products", people=False)
    k.top(qty, PRODUCT, "products", people=False, lowest=True)
    k.by(allocated, CATEGORY, "categories")


def _purchases(k: Kit) -> Fig:
    m = k.measure({"line", "amount", "spend", "value", "purchase", "po", "cost", "total"}, want="currency",
                  avoid={"unit", "budget", "price"})
    if m is None or not k.table_words(m.table) & PURCHASE_TABLES:
        raise Missing("purchase_lines")
    return k.fig(m)  # type: ignore[return-value]


def supplier_performance(k: Kit) -> None:
    pair = k.due()
    if pair is None or not (k.table_words(pair[0].table) | k.table_words(pair[1].table)) & PURCHASE_TABLES:
        raise Missing("purchases")                      # a customer's order due on a date is no purchase
    k.on_time()
    k.on_time(late=True)
    k.duration({"order", "ordered", "po", "placed", "created"}, {"received", "receipt", "delivered", "delivery"})
    received = k.fig(k.measure({"received", "receipt"}, want="number"))
    rejected = k.fig(k.measure({"rejected", "reject", "returned", "damaged"}, want="number"), lower_better=True)
    k.number(rejected, piece="rejections")
    k.number(k.worked("rejected_share", rejected, received, "ratio", 100, lower_better=True))
    k.on_time(entity=SUPPLIER)
    k.duration({"order", "ordered", "po", "placed", "created"}, {"received", "receipt", "delivered", "delivery"},
               entity=SUPPLIER)
    k.top(rejected, SUPPLIER, "suppliers") if rejected is not None else None
    k.on_time(ageing=True)


def spend_analysis(k: Kit) -> None:
    spend = _purchases(k)
    k.number(spend)
    k.number(k.fig(k.measure({"po", "purchase", "order"}, counts=True, near=spend.base.table)))
    k.trend(spend, last_year=True)
    k.by(spend, CATEGORY, "categories")
    k.top(spend, SUPPLIER, "suppliers")
    k.abc(spend, SUPPLIER, "suppliers", pareto=True)
    k.by(spend, ["country", "region"], "regions")
    k.top(spend, PRODUCT, "products", people=False)
    price = next((m for m in k.model.measures.values() if not m.hidden and k.may(m.table) and m.table == spend.base.table
                  and getattr(m.expr, "agg", "") == "avg" and m.format == "currency"), None)
    if price is not None:
        f = k.fig(price)
        k.movers(f, PRODUCT, "products", rising=True)


# Finance

def _ledger(k: Kit) -> dict[str, Fig | None]:
    """A ledger's revenue, expenses and profit: figures on a ledger's rows, or named as a ledger's lines on a table
    that is no sales table (a sales amount alone is no profit and loss)."""
    def ledger(m: Measure | None) -> Measure | None:
        if m is None:
            return None
        table = k.table_words(m.table)
        return m if table & LEDGER or not table & SALES_TABLES and _words(m.business_name) & PNL else None

    revenue = k.fig(ledger(k.measure({"revenue", "income", "sale", "turnover"}, want="currency", avoid={"budget"})))
    expenses = k.fig(ledger(k.measure({"expense", "expenses", "cost", "opex", "spend"}, want="currency",
                                      avoid={"budget", "unit"})), lower_better=True)
    profit = k.fig(ledger(k.measure({"profit", "net", "margin", "result", "ebitda"}, want="currency",
                                    avoid={"budget"})))
    return {"revenue": revenue, "expenses": expenses, "profit": profit}


def profit_and_loss(k: Kit) -> None:
    led = _ledger(k)
    revenue = led["revenue"]
    if revenue is None:
        try:
            p = _sales_parts(k)
        except Missing:
            raise Missing("one_currency" if k.mixed({"revenue", "income", "debit", "credit"}) else "ledger") from None
        if p["cost"] is None:
            raise Missing("ledger")
        revenue, led["expenses"], led["profit"] = p["sales"], p["cost"], p["margin"]
        k.lack("operating_costs")
    k.number(revenue)
    k.number(led["expenses"])
    k.number(led["profit"])
    k.number(k.worked("margin_pct", led["profit"], revenue, "ratio", 100,
                      inner=led["profit"] if led["profit"] is not None and led["profit"].derived else None))
    k.trend(revenue, last_year=True)
    k.trend(led["profit"])
    k.trend(led["expenses"])
    k.by(revenue, ["division", "segment", "business", "region", "department", "category"], "categories")
    k.by(led["expenses"], ["division", "department", "centre", "center", "segment", "type", "category"],
         "categories")
    k.by(led["expenses"], ["type", "account", "class", "category"], "categories")


def budget_against_actual(k: Kit) -> None:
    pair = analyses.target_pair(k.model, k.may)
    if pair is None:
        raise Missing("one_currency" if k.mixed({"revenue", "actual", "net", "debit", "credit"}) else "budget")
    actual, budget = pair
    a, b = k.fig(actual), k.fig(budget)
    k.number(a, TO_DATE, title=k.say["ytd"].format(f=actual.business_name), when="")
    k.number(b, TO_DATE, title=k.say["ytd"].format(f=budget.business_name), when="")
    k.number(k.worked("variance", a, b, "difference", t=low(budget.business_name)), TO_DATE, when=k.t("this year to date",
                                                                       "depuis le début de l'année"))
    k.target()
    k.target(groups=["centre", "center", "department", "division", "unit"])
    k.target(groups=["account", "type", "category", "class"])
    k.target(words={"centre", "center", "department", "account", "division", "store", "region"}, behind=True)
    k.trend(a, last_year=True)


def receivables(k: Kit) -> None:
    pair = k.due()
    if pair is None or not k.table_words(pair[0].table) & {"invoice", "bill", "receivable", "ar", "payment",
                                                            "statement", "claim"}:
        raise Missing("invoices")
    due, paid = pair
    k.on_time(ageing=True)
    k.on_time()
    k.duration({"invoice", "invoiced", "issued", "billed", "submitted", "document"},
               {"paid", "payment", "settled", "received"})
    amount = k.fig(k.measure({"amount", "billed", "total", "invoice", "open", "outstanding"}, want="currency",
                             table=due.table))
    k.number(amount)
    k.trend(amount)
    k.on_time(entity=CUSTOMER | {"payer"})
    k.top(amount, CUSTOMER | {"payer"}, "customers")


def customer_profitability(k: Kit) -> None:
    p = _sales_parts(k)
    if p["margin"] is None or k.activity() is None:
        raise Missing("customer_cost")
    s = p["sales"]
    k.number(s)
    k.number(p["cost"])
    k.number(p["margin"])
    k.number(p["margin_pct"])
    k.top(p["margin"], CUSTOMER, "customers")
    k.top(p["margin"], CUSTOMER, "customers", lowest=True)
    k.by(p["margin_pct"], ["segment", "type", "channel", "region", "category"], "categories")
    k.trend(p["margin_pct"])
    k.abc(p["margin"], CUSTOMER, "customers", pareto=True)


# People

def workforce_overview(k: Kit) -> None:
    pair = k.life(STAFF)
    head = k.fig(k.measure({"headcount", "fte", "employee", "staff"}, want="number", avoid={"pay", "salary"}))
    if pair is None and head is None:
        raise Missing("employees")
    k.active(number=True, of=STAFF)
    k.number(head)
    k.active("end_rate", of=STAFF) or k.lack("employees")
    k.active(of=STAFF)
    k.active("starts", of=STAFF)
    k.active("ends", of=STAFF)
    k.active(groups=["department", "division", "team", "unit"], of=STAFF) or k.by(
        head, ["department", "division", "team"], "departments")
    k.active(groups=["location", "site", "office", "country", "region"], of=STAFF) or k.by(
        head, ["location", "site", "office", "country"], "regions")
    k.by(head, ["type", "level", "grade", "job"], "departments")


PAY = {"pay", "payroll", "salary", "wage", "earnings", "compensation", "remuneration"}


def workforce_cost(k: Kit) -> None:
    found = k.measure(PAY | {"gross"}, want="currency", avoid={"tax", "withheld", "net", "deduction"})
    pay = k.fig(found if found is not None and _words(found.business_name) & PAY else None)
    if pay is None:
        raise Missing("payroll")
    overtime = k.fig(k.measure({"overtime"}, want="number"))
    hours = k.fig(k.measure({"hour", "hours", "worked"}, want="number", avoid={"overtime"}))
    head = k.fig(k.measure({"headcount", "fte"}, want="number"))
    k.number(pay)
    k.number(overtime, piece="overtime")
    k.number(k.worked("overtime_share", overtime, hours, "ratio", 100)) if hours is not None else None
    k.trend(pay, last_year=True)
    k.trend(overtime) if overtime is not None else None
    k.by(pay, ["department", "division", "team", "unit"], "departments")
    k.by(overtime, ["department", "division", "team", "unit"], "departments") if overtime is not None else None
    k.by(pay, ["location", "site", "country", "region"], "regions")
    if head is not None:
        k.by(k.worked("pay_per_head", pay, head, "ratio"), ["department", "division", "team"], "departments")


# Subscriptions

def recurring_revenue(k: Kit) -> None:
    mrr = k.fig(k.measure({"mrr", "recurring", "arr"}, want="currency"))
    if mrr is None:
        raise Missing("subscriptions")
    k.number(mrr, MONTH, when=k.t("at the last month end", "à la dernière fin de mois"))
    arr = Fig(k.named("arr"), mrr.base, (), (_derived(k.named("arr"), "product", mrr.sort, mrr.sort, 12),))
    k.number(arr, MONTH, when=k.t("at the last month end", "à la dernière fin de mois"))
    k.active(number=True, of=SUBSCRIBED)
    k.trend(mrr, last_year=True)
    k.by(mrr, ["plan", "tier", "product", "edition"], "plans", window=MONTH)
    k.by(mrr, ["segment", "country", "region", "channel"], "regions", window=MONTH)
    k.active("starts", of=SUBSCRIBED)
    k.active("ends", of=SUBSCRIBED)
    k.top(mrr, CUSTOMER, "customers", window=MONTH)
    invoiced = k.fig(k.measure({"amount", "billed", "invoice", "revenue"}, want="currency", avoid={"tax", "mrr"}))
    k.trend(invoiced) if invoiced is not None else None


def subscriber_retention(k: Kit) -> None:
    if k.life(SUBSCRIBED) is None:
        raise Missing("subscriptions")
    k.active(number=True, of=SUBSCRIBED)
    k.active("end_rate", of=SUBSCRIBED)
    k.active("ends", of=SUBSCRIBED)
    k.cohort()
    k.lifecycle("churn_rate")
    k.active(groups=["plan", "tier", "product", "status"], of=SUBSCRIBED)
    k.active(groups=["segment", "country", "region", "channel"], of=SUBSCRIBED)
    k.lifecycle()
    k.absent()


# Health and pharmacy

def pharmacy_operations(k: Kit) -> None:
    fills = k.fig(k.measure({"fill", "prescription", "rx", "dispense", "dispensed", "script"}, counts=True))
    if fills is None:
        raise Missing("fills")
    home = fills.base.table
    revenue = k.fig(k.measure({"price", "charged", "revenue", "sale", "paid", "amount"}, want="currency", table=home,
                              avoid={"copay", "cost", "tax"}))
    cost = k.fig(k.measure({"cost", "ingredient", "acquisition"}, want="currency", table=home))
    profit = k.worked("profit", revenue, cost, "difference")
    thing = names.singular(fills.name.lower().removeprefix("number of ").removeprefix("nombre de "))
    k.number(fills)
    k.number(revenue)
    k.number(k.worked("profit_per", profit, fills, "ratio", inner=profit, thing=thing))
    k.trend(fills, last_year=True)
    k.by(fills, ["payer", "plan", "insurance", "type"], "payers")
    k.top(profit or revenue or fills, {"formula", "drug", "product", "medication", "item"}, "products", people=False)
    k.top(fills, {"prescriber", "doctor", "physician"}, "prescribers")
    k.gap()
    k.lifecycle()


def care_utilisation(k: Kit) -> None:
    visits = k.fig(k.measure({"encounter", "visit", "admission", "appointment"}, counts=True))
    act = k.activity()
    if visits is None or not (k.table_words(visits.base.table) & CARE
                              or act is not None and _words(act.entity.business_name) & CARE):
        raise Missing("encounters")
    cost = k.fig(k.measure({"cost", "charge", "claim", "paid", "billed"}, want="currency", near=visits.base.table))
    thing = names.singular(visits.name.lower().removeprefix("number of "))
    k.number(visits)
    k.number(cost)
    k.number(k.worked("cost_per", cost, visits, "ratio", thing=thing))
    k.series(visits, ["class", "type", "category"], "categories")
    k.by(visits, ["payer", "insurance", "plan"], "payers")
    k.top(visits, {"condition", "diagnosis", "procedure"}, "categories", people=False)
    k.lifecycle()
    k.top(visits, {"organization", "provider", "clinic", "hospital"}, "sites")


# Operations

def network_operations(k: Kit) -> None:
    alarms = k.fig(k.measure({"alarm", "incident", "event", "outage", "fault"}, counts=True))
    tickets = k.fig(k.measure({"ticket", "case", "incident", "request"}, counts=True))
    if alarms is None and tickets is None:
        raise Missing("incidents")
    k.number(alarms)
    k.number(tickets)
    k.duration({"opened", "raised", "created", "logged", "reported"}, {"resolved", "closed", "cleared", "fixed"})
    k.trend(alarms or tickets)
    k.by(alarms or tickets, ["severity", "priority", "level"], "severity")
    k.by(tickets, ["priority", "severity", "category"], "severity") if tickets is not None else None
    k.top(alarms or tickets, {"site", "location", "node", "pop", "exchange"}, "sites")
    k.duration({"opened", "raised", "created", "logged"}, {"resolved", "closed", "cleared"},
               groups=["site", "region", "priority", "severity"])
    k.active(number=True)
    k.top(alarms or tickets, {"device", "router", "switch", "node"}, "sites", people=False)


def field_service(k: Kit) -> None:
    jobs = k.fig(k.measure({"booking", "job", "visit", "appointment", "callout", "work"}, counts=True))
    if jobs is None:
        raise Missing("jobs")
    revenue = k.fig(k.measure({"revenue", "amount", "sale", "fee", "price", "charged", "billed", "line"},
                              want="currency", avoid={"travel", "tax", "cost"}))
    thing = names.singular(jobs.name.lower().removeprefix("number of "))
    k.number(jobs)
    k.number(revenue)
    k.number(k.worked("revenue_per", revenue, jobs, "ratio", thing=thing))
    k.on_time() or k.lack("on_time")
    k.trend(jobs, last_year=True)
    k.by(revenue or jobs, ["service", "type", "category", "trade"], "categories")
    k.top(revenue or jobs, {"technician", "engineer", "crew", "worker"}, "technicians")
    k.funnel()
    k.by(jobs, ["region", "branch", "area", "city", "state", "partner"], "regions")
    k.lifecycle()


def manufacturing(k: Kit) -> None:
    output = k.fig(k.measure({"produced", "production", "output", "batch", "good", "yield"}, want="number",
                             avoid={"scrap", "waste", "percent"}))
    if output is None:
        raise Missing("production")
    scrap = k.fig(k.measure({"scrap", "waste", "reject", "defect"}, want="number"), lower_better=True)
    downtime = k.fig(k.measure({"downtime", "stoppage", "outage"}, want="number"), lower_better=True)
    k.number(output)
    k.number(scrap, piece="scrap")
    k.number(k.worked("scrap_rate", scrap, output, "ratio", 100, lower_better=True))
    k.number(downtime, piece="downtime")
    k.trend(output, last_year=True)
    k.trend(scrap) if scrap is not None else None
    k.by(output, ["line", "plant", "machine", "workcenter", "site", "location"], "lines")
    k.by(scrap, ["line", "plant", "machine", "workcenter", "site"], "lines") if scrap is not None else None
    k.top(scrap or output, PRODUCT, "products", people=False)


# ── the catalogue ───────────────────────────────────────────────────────────

@dataclass
class Book:
    key: str
    group: str
    name: tuple[str, str]                 # (English, French)
    reader: tuple[str, str]
    needs: str                            # the piece it cannot be without (a key of _PIECES)
    asks: str                             # a pattern of the requests that name it
    build: Callable[[Kit], None]


GROUPS = {"sales": ("Sales and customers", "Ventes et clients"),
          "supply": ("Inventory and supply", "Stocks et approvisionnement"),
          "finance": ("Finance", "Finance"), "people": ("People", "Ressources humaines"),
          "subscriptions": ("Subscriptions", "Abonnements"), "health": ("Health and pharmacy", "Santé et pharmacie"),
          "operations": ("Operations", "Opérations")}

BOOKS = [
    Book("sales_overview", "sales", ("Sales executive overview", "Vue d'ensemble des ventes"),
         ("Owner, CEO", "Dirigeant"), "sales",
         r"sales (?:executive|exec|leadership|owner'?s?) (?:overview|dashboard)|owner'?s? (?:overview|dashboard)|"
         r"sales overview|vue d'ensemble des ventes|synth[eè]se des ventes", sales_overview),
    Book("sales_performance", "sales", ("Sales performance", "Performance commerciale"),
         ("Head of sales", "Direction commerciale"), "sales",
         r"sales (?:performance|team|teams|manager|managers|director|directors|head|leaders?|reps?|force)|"
         r"head of sales|(?:for|of) (?:the|my|our) sales|sales vs (?:target|plan|ly)|"
         r"performance commerciale|[ée]quipe commerciale|direct(?:eur|rice|ion) commercial(?:e)?|force de vente",
         sales_performance),
    Book("customer_health", "sales", ("Customer health", "Santé clients"), ("Marketing, CRM", "Marketing, CRM"),
         "customers",
         r"customer (?:health|retention|loyalty|churn|base|lifecycle|insights?)|retention|churn|crm|marketing|"
         r"loyalty|repeat (?:customers|purchase)|lapsed|fid[ée]lisation|r[ée]tention|attrition client|"
         r"sant[ée] client", customer_health),
    Book("product_and_category", "sales", ("Product and category", "Produits et catégories"),
         ("Merchandising", "Marchandisage"), "products",
         r"product (?:and category|performance|mix)|category (?:performance|management|manager)|merchandis\w+|"
         r"assortment|range review|abc analysis|produits? et cat[ée]gories?|assortiment", product_and_category),
    Book("store_scorecard", "sales", ("Store scorecard", "Tableau de bord magasins"),
         ("Regional managers", "Directeurs régionaux"), "stores",
         r"(?:store|branch|shop|outlet|location|site)s? (?:scorecard|managers?|performance|comparison|league)|"
         r"regional managers?|area managers?|scorecard|magasins?|agences?|directeurs? r[ée]gionaux?",
         store_scorecard),
    Book("returns_and_quality", "sales", ("Returns and quality", "Retours et qualité"), ("Operations", "Opérations"),
         "returns", r"returns?(?: and (?:quality|refunds))?(?: dashboard)?|refunds?|quality|retours?|remboursements?",
         returns_and_quality),
    Book("monthly_review", "sales", ("Monthly business review", "Revue mensuelle d'activité"),
         ("Leadership meeting", "Comité de direction"), "sales",
         r"monthly (?:business )?review|mbr|board pack|month(?:ly)? report|leadership (?:meeting|pack)|"
         r"revue mensuelle|comit[ée] de direction|rapport mensuel", monthly_review),
    Book("inventory_health", "supply", ("Inventory health", "Santé des stocks"), ("Supply chain", "Supply chain"),
         "stock", r"inventory|stock (?:health|levels?|value|position)|warehouse|supply chain|stocks?|entrep[oô]ts?",
         inventory_health),
    Book("availability", "supply", ("Availability and replenishment", "Disponibilité et réapprovisionnement"),
         ("Planners", "Planificateurs"), "reorder",
         r"availability|replenish\w*|reorder|stock[- ]?outs?|planners?|allocat\w+|disponibilit[ée]|"
         r"r[ée]approvisionnement|ruptures?", availability),
    Book("supplier_performance", "supply", ("Supplier performance", "Performance fournisseurs"),
         ("Procurement", "Achats"), "purchases",
         r"suppliers?(?: performance| scorecard)?|vendors?(?: performance)?|on[- ]time deliver\w*|lead times?|"
         r"fournisseurs?|d[ée]lais? de livraison", supplier_performance),
    Book("spend_analysis", "supply", ("Spend analysis", "Analyse des dépenses"),
         ("Finance, procurement", "Finance, achats"), "purchase_lines",
         r"spend(?: analysis| cube)?|procurement|purchasing|purchases|buyers?|d[ée]penses|achats",
         spend_analysis),
    Book("profit_and_loss", "finance", ("Profit and loss overview", "Compte de résultat"), ("CFO", "Directeur financier"),
         "ledger", r"p ?& ?l|p and l|profit and loss|income statement|cfo|finance(?: team)?|financial|"
         r"compte de r[ée]sultat|daf|directeur financier|financier", profit_and_loss),
    Book("budget_against_actual", "finance", ("Budget against actual", "Budget et réel"),
         ("FP&A, cost-centre owners", "Contrôle de gestion"), "budget",
         r"budget|actual(?:s)? (?:vs|against|versus)|variance|fp ?& ?a|cost[- ]cent(?:re|er)s?|"
         r"contr[oô]le de gestion|[ée]carts?|r[ée]el", budget_against_actual),
    Book("receivables", "finance", ("Receivables and ageing", "Créances et balance âgée"),
         ("Credit control", "Recouvrement"), "invoices",
         r"receivables?|a/?r\b|ageing|aging|overdue|credit control|collections?|days to pay|dso|cr[ée]ances?|"
         r"balance [âa]g[ée]e|recouvrement|impay[ée]s", receivables),
    Book("customer_profitability", "finance", ("Customer profitability", "Rentabilité clients"),
         ("Commercial finance", "Finance commerciale"), "customer_cost",
         r"customer profitab\w+|profitab\w+|margin by customer|loss[- ]making|rentabilit[ée]", customer_profitability),
    Book("workforce_overview", "people", ("Workforce overview", "Effectifs"), ("HR leadership", "Direction RH"),
         "employees", r"workforce|headcount|hr\b|human resources|people (?:team|dashboard)|attrition|turnover rate|"
         r"hires|leavers|effectifs?|rh\b|ressources humaines|recrutements?|d[ée]parts", workforce_overview),
    Book("workforce_cost", "people", ("Workforce cost", "Coût du personnel"), ("HR, finance", "RH, finance"), "payroll",
         r"payroll|workforce cost|labou?r cost|staff cost|people cost|overtime|wages?|salar(?:y|ies)|paie|"
         r"masse salariale|co[uû]t du personnel|heures suppl[ée]mentaires", workforce_cost),
    Book("recurring_revenue", "subscriptions", ("Recurring revenue", "Revenus récurrents"),
         ("Founders, finance", "Fondateurs, finance"), "subscriptions",
         r"mrr|arr|recurring revenue|saas|subscriptions?(?: revenue)?|net revenue retention|nrr|"
         r"revenus? r[ée]currents?|abonnements?", recurring_revenue),
    Book("subscriber_retention", "subscriptions", ("Subscriber retention", "Fidélisation des abonnés"),
         ("Customer success", "Customer success"), "subscriptions",
         r"subscriber retention|customer success|accounts? at risk|trial to paid|renewals?|cancellations?|"
         r"fid[ée]lisation des abonn[ée]s|r[ée]siliations?", subscriber_retention),
    Book("pharmacy_operations", "health", ("Pharmacy operations", "Activité de la pharmacie"),
         ("Owner, pharmacist in charge", "Pharmacien titulaire"), "fills",
         r"pharmacy|pharmacist|fills?|prescriptions?|scripts?|dispens\w+|pharmacie|pharmacien|ordonnances?|"
         r"d[ée]livrances?", pharmacy_operations),
    Book("care_utilisation", "health", ("Care utilisation", "Activité de soins"),
         ("Clinic operations", "Direction de clinique"), "encounters",
         r"care utili[sz]ation|clinic(?:al)?|encounters?|patients?|utili[sz]ation|admissions?|consultations?|"
         r"activit[ée] de soins", care_utilisation),
    Book("network_operations", "operations", ("Network operations", "Exploitation réseau"),
         ("NOC manager", "Responsable NOC"), "incidents",
         r"network|noc|incidents?|alarms?|outages?|tickets?|sla|service desk|helpdesk|r[ée]seau|supervision",
         network_operations),
    Book("field_service", "operations", ("Field service", "Interventions terrain"),
         ("Operations manager", "Responsable des opérations"), "jobs",
         r"field service|technicians?|dispatch\w*|jobs?|bookings?|appointments?|engineers?|interventions?|"
         r"techniciens?|rendez-vous", field_service),
    Book("manufacturing", "operations", ("Manufacturing", "Production"), ("Plant manager", "Directeur d'usine"),
         "production", r"manufactur\w+|plant|factory|production|scrap|downtime|oee|usine|rebuts?|fabrication",
         manufacturing),
]
BY_KEY = {b.key: b for b in BOOKS}


def _text(pair: tuple[str, str], lang: str) -> str:
    return pair[1] if lang == "fr" else pair[0]


@dataclass
class Built:
    book: Book
    name: str
    tiles: list[Made]
    state: str                         # ready | partly | no
    missing: list[str] = field(default_factory=list)   # what was left out, in the reader's words
    instead: list[str] = field(default_factory=list)   # the tiles drawn in the place of those left out
    uses: list[str] = field(default_factory=list)      # the tables its tiles read


def build(model: SemanticModel, key: str, *, today: dt.date, lang: str = "en",
          allowed: set[str] | None = None) -> Built:
    """One recipe, for a reader allowed ``allowed`` tables: its name, its tiles (each plan checked), whether the
    data gives it all, in part, or not at all, and what was missing."""
    book = BY_KEY[key]
    k = Kit(model, today=today, lang=lang, allowed=allowed)
    name = _text(book.name, k.lang)
    try:
        book.build(k)
    except Missing as gone:
        return Built(book, name, [], "no", [k.piece(gone.piece)])
    if len(k.made) < FEWEST:
        return Built(book, name, [], "no", [k.piece(book.needs)])
    tiles = _ordered(k.made)
    missing = [k.piece(p) for p in k.left_out]
    uses = [(model.tables[t].business_name or model.tables[t].name).lower() for t in k.used if t in model.tables]
    instead = [t for t in k.instead if any(m.title == t for m in tiles)]
    return Built(book, name, tiles, "partly" if missing else "ready", missing=missing, uses=uses, instead=instead)


def kind_of(plan: dict[str, Any]) -> str:
    """What a plan draws as a dashboard tile: a trend over time, a ranking, a breakdown, or one number."""
    if (plan.get("time") or {}).get("grain") and not plan.get("group_by"):
        return "trend"
    if plan.get("group_by"):
        return "table" if plan.get("limit") else "breakdown"
    return "number"


_SHORT = {"day", "week"}


def _short(window: dict[str, Any]) -> bool:
    """Is the period too short for a trend by month to say anything (a month, a week, a few days)?"""
    kind, unit = window.get("kind"), window.get("unit")
    if kind == "between" and window.get("start") and window.get("end"):
        try:
            return (dt.date.fromisoformat(str(window["end"])) - dt.date.fromisoformat(str(window["start"]))).days < 80
        except ValueError:
            return False
    if kind in ("last", "this", "previous", "to_date"):
        return unit in _SHORT or unit == "month" and int(window.get("n") or 1) < 3
    return False


MOST_BREAKDOWNS = 3
MOST_TOPS = 2


def around(model: SemanticModel, plan: dict[str, Any], title: str, question: str, *, today: dt.date,
           lang: str = "en", allowed: set[str] | None = None) -> tuple[str, list[Made]] | None:
    """A dashboard around the answer the reader has on screen ("create a dashboard using this"): the answer itself,
    with its figure's headline number, its trend (against the year before where the data has it), its breakdowns by
    the groupings it may be split by, and its top lists, each over the answer's period, by the answer's date and
    under the answer's filters. A tile the answer already is, or a breakdown by a grouping the answer keeps to one
    member, is left out. None when the answer's figure is no measure of the model (a metric of the reader's own,
    a shape with no figure): the request is then designed from the answer's name."""
    k = Kit(model, today=today, lang=lang, allowed=allowed)
    by_slug = {m.slug: m for m in model.measures.values()}
    m = next((by_slug[s] for s in plan.get("measures") or [] if isinstance(s, str) and s in by_slug), None)
    if m is None or not k.may(m.table):
        return None
    fig = k.fig(m)
    time = plan.get("time") or {}
    window = dict(time.get("window") or {"kind": "all"})
    filters = [f for f in plan.get("filters") or [] if isinstance(f, dict)]
    kept_to_one = {f.get("field") for f in filters if f.get("op") in ("eq", "in") and len(f.get("values") or []) == 1}
    k._shown.add(title.casefold())
    k.number(fig, window)
    k.trend(fig, last_year=True, window=YEAR if _short(window) else window)
    breakdowns = 0
    for a in recipes.groupings(model, m, recipes.GENERIC, k.may):
        if breakdowns >= MOST_BREAKDOWNS:
            break
        if a.slug in kept_to_one or [a.slug] == plan.get("group_by"):
            continue
        g = low(recipes._named_by(model, a))
        if k.add(k.say["by"].format(f=fig.name, g=g, G=g.capitalize()), "breakdown",
                 {"intent": "breakdown", **fig.part(), "group_by": [a.slug], "sort": [{"by": fig.sort, "desc": True}],
                  "time": {"window": window}}):
            breakdowns += 1
    tops = 0
    for e in recipes.tops(model, m, recipes.GENERIC, k.may):
        if tops >= MOST_TOPS:
            break
        if e.slug in kept_to_one or [e.slug] == plan.get("group_by"):
            continue
        many = k._many(e)
        if k.add(k.say["top"].format(e=many, E=many.capitalize(), f=low(fig.name) if k.lang == "en" else fig.name),
                 "table", {"intent": "rank", **fig.part(), "group_by": [e.slug], "limit": 10,
                           "sort": [{"by": fig.sort, "desc": True}], "time": {"window": window}}):
            tops += 1
    shape = (kind_of(plan), tuple(plan.get("group_by") or ()), time.get("grain"))
    tiles = [Made(title, kind_of(plan), question, plan, k.tab)]
    for made in k.made:
        under = dict(made.plan)
        if filters:
            under["filters"] = [*filters, *(under.get("filters") or [])]
        if time.get("date"):
            under["time"] = {**under.get("time", {}), "date": time["date"]}
        if plan.get("include_left_out"):
            under["include_left_out"] = True
        if (made.kind, tuple(under.get("group_by") or ()), (under.get("time") or {}).get("grain")) == shape:
            continue
        if not k.answers(under):
            continue
        tiles.append(Made(made.title, made.kind, made.question, under, made.tab))
    if len(tiles) < 2:
        return None
    named = [str(v) for f in filters if f.get("op") in ("eq", "in") for v in f.get("values") or []][:3]
    name = k.t(f"{fig.name} overview", f"Vue d'ensemble : {fig.name}")
    return (f"{name} — {', '.join(named)}" if named else name), _ordered(tiles)


_ORDER = {"number": 0, "trend": 1, "breakdown": 2, "table": 3}


def _ordered(made: list[Made]) -> list[Made]:
    """The numbers first on each tab, the tabs in the order they came."""
    tabs = list(dict.fromkeys(m.tab for m in made))
    return [m for tab in tabs for m in sorted((x for x in made if x.tab == tab), key=lambda x: _ORDER[x.kind])]


def catalogue(model: SemanticModel, *, today: dt.date, lang: str = "en",
              allowed: set[str] | None = None) -> list[dict[str, Any]]:
    """Every recipe as this workspace's data gives it: ready, partly (and what is missing), or not for this data
    (and what it needs). For checking the catalogue against a workspace: readers never see it, they ask in their
    own words."""
    out = []
    for book in BOOKS:
        built = build(model, book.key, today=today, lang=lang, allowed=allowed)
        out.append({"key": book.key, "group": _text(GROUPS[book.group], lang), "name": built.name,
                    "reader": _text(book.reader, lang), "state": built.state, "tiles": len(built.tiles),
                    "tabs": list(dict.fromkeys(t.tab for t in built.tiles)), "missing": built.missing,
                    "uses": built.uses,
                    "needs": Kit(model, today=today, lang=lang, allowed=allowed).piece(book.needs)})
    return out


def find(about: str, *, lists: bool = False) -> list[str]:
    """The recipes a request names, the best first: its own name ("a sales performance dashboard"), else the
    longest phrase of a reader or subject it holds ("for my store managers", "customer retention"). A request that
    lists its tiles (``lists``: "returns by courier and weekday") is built as it lists them, unless it names a
    recipe outright."""
    text = " ".join((about or "").casefold().replace("’", "'").split())
    if not text:
        return []
    scored = []
    for i, book in enumerate(BOOKS):
        if any(n.casefold() in text for n in book.name):
            scored.append((1000, -i, book.key))
            continue
        if lists:
            continue
        hits = [m.group(0) for m in re.finditer(rf"\b(?:{book.asks})\b", text)]
        if hits:
            scored.append((max(len(h) for h in hits), -i, book.key))
    return [key for _, _, key in sorted(scored, reverse=True)]
