"""
evals/emco_rehearsal.py

Drive every way an EMCO reader asks, in both languages, through the real
pipeline stages — without a warehouse.

    python -m evals.emco_rehearsal
    python -m evals.emco_rehearsal --html /tmp/emco.html
    python -m evals.emco_rehearsal --json --only followup
    python -m evals.emco_rehearsal --lang fr --verbose

What this is for
────────────────
A demo is a sequence of questions asked out loud, and the failures that matter
are not crashes. They are a question that quietly answers something else: a
window that was not detected and so was measured against the server clock, a
fixed metric template answering a question it was not written for, a comparison
compiled down to one number, a follow-up that became a fresh query. Every one of
those is invisible in a screenshot and obvious in a table.

So this runs the DETERMINISTIC layers end to end and prints what each one
decided:

    canonicalise -> temporal window -> metric registry -> metric scope
                 -> governed compile -> sqlglot parse + validator
                 -> answer card on synthetic rows -> follow-up routing

and compares the two languages side by side. A row is a FAIL when the French
decision differs from the English one, or when either differs from what the
corpus says it should be.

What it is NOT
──────────────
Not a live test. The LLM and the warehouse are the boundaries: no model is
called and no query is executed, so nothing here proves an Azure connection, a
credential, an index or a row count. The compiled SQL is parsed and validated,
not run. Everything past that — actual rows, actual latency, actual freshness —
needs the VM and is the rehearsal this cannot replace.

The schema is the shape of EMCO's mart rather than a copy of it: four
role-playing date keys on the invoice fact, a separate returns fact, a
semi-additive balance fact, and one shared DT_DMS dimension. Column names follow
their M3 conventions because the naming conventions are half of what the
deterministic layers read.
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import os
import shutil
import sys
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# `python3 evals/emco_rehearsal.py` puts evals/ on sys.path, not the repo root,
# so every `import store` / `import core.*` below failed with ModuleNotFoundError
# — the harness ran only as `python -m evals.emco_rehearsal`. It is the tool for
# checking a release before it goes back to the customer, and a tool that fails
# on the obvious invocation is a tool that does not get run.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

CORPUS = Path(__file__).resolve().parent / "emco_questions.yaml"

# ── The mart, in EMCO's shape ────────────────────────────────────────────────

SALES = "EMDW_DMART.CUS_ORD_IVC_FCT"
RETURNS = "EMDW_DMART.CUS_RTN_FCT"
BALANCE = "EMDW_DMART.INV_BAL_FCT"
DATE_DIM = "EMDW_DMART.DT_DMS"

COLUMNS: dict[str, dict[str, str]] = {
    SALES: {
        "CUS_IVC_DT_DMS_KEY": "int", "SHP_DT_DMS_KEY": "int",
        "CNL_ORD_DT_DMS_KEY": "int", "IVC_PRD_DMS_KEY": "int",
        "NET_SLS_AMT": "decimal(18,2)", "GRS_SLS_AMT": "decimal(18,2)",
        "ORD_QTY": "int", "CUS_NBR": "int", "WHS_NBR": "int",
        "REP_NBR": "int", "PRD_CAT_CDE": "varchar(10)",
    },
    RETURNS: {
        "RTN_DT_DMS_KEY": "int", "RTN_AMT": "decimal(18,2)",
        "RTN_QTY": "int", "CUS_NBR": "int", "WHS_NBR": "int",
    },
    BALANCE: {
        "BAL_DT_DMS_KEY": "int", "BAL_QTY": "int",
        "BAL_VAL_AMT": "decimal(18,2)", "WHS_NBR": "int",
    },
    DATE_DIM: {
        "DT_DMS_KEY": "int", "DMS_DT": "date", "DMS_YR": "int",
        "DMS_MTH": "int",
    },
}

# The approved date roles on the invoice fact, all four of them reaching the SAME
# DT_DMS dimension through different surrogate keys. This is the role-playing
# shape, and which one a question resolves to is a decision worth seeing: "by
# invoice date" and "by shipment date" are different answers over the same rows.
_ROLE_BASE = {
    "anchor_policy": "latest_available",
    "dimension_table": DATE_DIM, "dimension_key": "DT_DMS_KEY",
    "date_column": "DMS_DT", "date_key_type": "surrogate_fk",
    "temporal_grain": "day",
}
DATE_ROLES: dict[str, dict] = {
    "invoice_date": {**_ROLE_BASE, "fact_table": SALES, "anchor_table": SALES,
                     "fact_column": "CUS_IVC_DT_DMS_KEY",
                     "role_alias": "invoice_date",
                     "business_role": "Invoice Date"},
    "delivery_date": {**_ROLE_BASE, "fact_table": SALES, "anchor_table": SALES,
                      "fact_column": "SHP_DT_DMS_KEY",
                      "role_alias": "shipment_date",
                      "business_role": "Shipment Date"},
    "cancelled_order_date": {**_ROLE_BASE, "fact_table": SALES,
                             "anchor_table": SALES,
                             "fact_column": "CNL_ORD_DT_DMS_KEY",
                             "role_alias": "cancelled_order_date",
                             "business_role": "Cancelled Order Date"},
    "order_date": {**_ROLE_BASE, "fact_table": SALES, "anchor_table": SALES,
                   "fact_column": "ORD_ENT_DT_DMS_KEY",
                   "role_alias": "order_date",
                   "business_role": "Order Date"},
}
INVOICE_DATE_ROLE = DATE_ROLES["invoice_date"]


def _role_for(canonical: str) -> tuple[str, dict]:
    """Which approved date role this question names, defaulting to the invoice.

    Read with the product's OWN role vocabulary (core/date_roles.py) rather than
    a second list written here, so the harness resolves the role the way the
    product does and a change to that vocabulary shows up in the rehearsal.
    """
    from core.date_roles import DATE_ROLES as PRODUCT_ROLES

    text = f" {canonical.lower()} "
    best: tuple[int, str] = (0, "invoice_date")
    for role in PRODUCT_ROLES:
        if role.key not in DATE_ROLES:
            continue
        for synonym in role.synonyms:
            phrase = str(synonym).lower().strip()
            if len(phrase) > best[0] and f" {phrase} " in text:
                best = (len(phrase), role.key)
    return best[1], DATE_ROLES[best[1]]

NET_SALES_METRIC = {
    "name": "Net Sales", "label": "Net Sales",
    "formula_type": "expression",
    "sql_template": "SUM(fact_rows.[NET_SLS_AMT])",
    "base_table": SALES,
    "synonyms": "net sales, revenue, ventes nettes, chiffre d'affaires",
}

# What the metric registry holds for a template-answerable question. Separate
# from the expression metric above because the two routes are different: a
# `query` metric is a frozen SELECT, an `expression` metric is a formula the
# governed compiler builds around.
NET_SALES_TEMPLATE = {
    "name": "Net Sales", "label": "Net Sales",
    "formula_type": "query",
    "sql_template": (
        "SELECT SUM(NET_SLS_AMT) AS NET_SALES FROM EMDW_DMART.CUS_ORD_IVC_FCT"),
    "base_table": SALES,
    "synonyms": "net sales, revenue, ventes nettes, chiffre d'affaires",
}

SCOPE_METRICS = [
    NET_SALES_METRIC,
    {"name": "Returns", "label": "Returns", "formula_type": "expression",
     "sql_template": "SUM(fact_rows.[RTN_AMT])", "base_table": RETURNS,
     "synonyms": "returns, credits, retours, avoirs"},
    {"name": "Inventory Balance", "label": "Inventory Balance",
     "formula_type": "expression",
     "sql_template": "SUM(fact_rows.[BAL_VAL_AMT])", "base_table": BALANCE,
     "synonyms": "inventory balance, stock balance, solde d'inventaire"},
]

# A result already on screen, for the follow-up router. French branch names on
# purpose: matching a cached value used to be accent-exact.
CACHED_COLUMNS = ["WHS_DSC", "NET_SLS_AMT", "IVC_MTH"]
CACHED_ROWS = [
    {"WHS_DSC": "Montréal", "NET_SLS_AMT": 45_000_000.0, "IVC_MTH": "2025-04"},
    {"WHS_DSC": "Toronto", "NET_SLS_AMT": 28_000_000.0, "IVC_MTH": "2025-05"},
    {"WHS_DSC": "Calgary", "NET_SLS_AMT": 18_900_000.0, "IVC_MTH": "2025-06"},
    {"WHS_DSC": "Trois-Rivières", "NET_SLS_AMT": 5_000_000.0, "IVC_MTH": "2025-04"},
    {"WHS_DSC": "Regina", "NET_SLS_AMT": 3_100_000.0, "IVC_MTH": "2025-05"},
]

# Synthetic rows for the answer card. Deliberately the two shapes that were
# reported wrong: a warehouse ranking whose top three hold 91.9%, and a series
# whose period column is an INTEGER year.
CARD_RANKING = [
    {"WHS_DSC": name, "NET_SLS_AMT": value}
    for name, value in [("Montréal", 45.0e6), ("Toronto", 28.0e6),
                        ("Calgary", 18.9e6), ("Halifax", 5.0e6),
                        ("Regina", 3.1e6)]
]
CARD_YEARS = [
    {"IVC_YR": year, "NET_SLS_AMT": value}
    for year, value in [(2020, 41.2e6), (2021, 48.9e6), (2022, 55.1e6),
                        (2023, 61.8e6), (2024, 66.4e6), (2025, 71.9e6)]
]


@dataclass
class Observation:
    """What the deterministic layers decided about one phrasing."""
    canonical: str = ""
    window: str = "none"
    window_amount: Any = None
    window_unit: str = ""
    anchor_policy: str = ""
    template_matched: str = ""
    scoped_metrics: list[str] = field(default_factory=list)
    top_n: Any = None
    date_role: str = ""
    compiled: bool = False
    sql_parses: bool = False
    sql_seekable: bool | None = None
    compile_sql: str = ""
    followup_routes: bool | None = None


@dataclass
class CaseResult:
    id: str
    intent: str
    note: str
    english: str
    french: str
    en: Observation
    fr: Observation
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


# ── The stages ───────────────────────────────────────────────────────────────

# A requested display field changes the result grain, which is what takes a
# question away from the SCALAR compiler. The semantic planner puts one in the
# plan for every "by X"; this harness has no planner, so it reads the same
# request from the canonical text. Without it the compiler was handed a plan with
# no fields and legitimately built a scalar for "net sales by warehouse" -- a
# fidelity gap in the harness that read as seven product failures.
_GROUPING_RE = __import__("re").compile(
    r"\b(?:by|per|for\s+each|grouped\s+by|split\s+by|breakdown\s+of)\s+\w",
    __import__("re").I,
)


def _requested_fields(canonical: str) -> list[dict]:
    """The display fields the planner would have put in the plan."""
    if not _GROUPING_RE.search(canonical):
        return []
    return [{
        "name": "grouping", "role": "dimension",
        "display_required": True, "enforcement": "required",
    }]


def _semantic_context(question: str, canonical: str, window: dict,
                      top_n: Any, metrics: list[dict]) -> dict:
    """A production-shaped context for the governed compiler.

    Built from the detected window and the approved invoice-date role, which is
    what the pipeline assembles once the semantic planner has run. `question`
    stays the reader's own words and `canonical_question` carries the English --
    the split the compiler's refusal gates read.
    """
    if not window or not metrics:
        # No governed window, or no approved formula the scope stage matched --
        # either way there is nothing for this compiler to build, which is what
        # production does with "how many orders last month": no expression
        # metric is in scope, so the question goes to the planner.
        return {}
    _role_key, role = _role_for(canonical)
    policy = {
        **role,
        "kind": window.get("kind", ""),
        "amount": window.get("amount", 0),
        "unit": window.get("unit", "day"),
        "anchor_policy": window.get("anchor_policy", "latest_available"),
    }
    return {
        "question": question,
        "canonical_question": canonical,
        "top_n": top_n,
        "intent": "",
        "production_sql": True,
        "semantic_plan": {
            "enabled": True, "fields": _requested_fields(canonical),
            "joins": [], "required_tables": [SALES, DATE_DIM],
            "temporal_policies": [policy],
        },
        "analytical_request_plan": {"status": "compiled",
                                    "source_facts": [SALES],
                                    "source_fact": SALES},
        "metric_formulas": metrics,
        "resolved_date_anchor": {
            "value": "2026-08-15", "fact_table": SALES,
            "fact_column": policy["fact_column"], "date_column": "DMS_DT",
            "source": "probed_from_fact_rows",
        },
        "temporal_window": dict(window),
    }


def observe(question: str, lang: str, account_id: str) -> Observation:
    """Run one phrasing through every deterministic layer."""
    import store
    from core.contextual_dates import detect_temporal_window
    from core.metric_scope import resolve_metric_scope
    from core.pipeline_helpers import compile_governed_temporal_metric_sql
    from core.query_router import should_attempt_cache_followup
    from core.query_semantics import detect_top_n_intent
    from core.question_normalizer import canonical_question

    out = Observation()
    out.canonical = canonical_question(question, lang)

    window = detect_temporal_window(out.canonical) or {}
    out.window = str(window.get("kind") or "none")
    out.window_amount = window.get("amount")
    out.window_unit = str(window.get("unit") or "")
    out.anchor_policy = str(window.get("anchor_policy") or "")

    matched = store.match_metric(account_id, question, lang=lang)
    out.template_matched = str((matched or {}).get("name") or "")

    scope = resolve_metric_scope(
        SCOPE_METRICS, out.canonical, COLUMNS, reader_question=question)
    out.scoped_metrics = [str(m.get("name") or "") for m in scope.metrics]

    intent = detect_top_n_intent(out.canonical)
    out.top_n = intent.to_dict() if intent else None

    # Only an EXPRESSION metric on ONE fact is something the governed compiler can
    # build around, which is the same filter the pipeline applies.
    compilable = [
        dict(metric) for metric in SCOPE_METRICS
        if metric["name"] in out.scoped_metrics
        and str(metric.get("formula_type")) == "expression"
    ]
    if len({m["base_table"] for m in compilable}) > 1:
        compilable = []  # two facts is the planner's job, not this compiler's
    out.date_role, _role = _role_for(out.canonical)
    context = _semantic_context(
        question, out.canonical, window, out.top_n, compilable)
    if context:
        try:
            out.compile_sql = compile_governed_temporal_metric_sql(
                "azure_sql", set(COLUMNS), set(COLUMNS), COLUMNS, context) or ""
        except Exception as exc:  # noqa: BLE001 — a rehearsal reports, never dies
            out.compile_sql = ""
            print(f"    compiler raised on {question!r}: {exc}", file=sys.stderr)
    out.compiled = bool(out.compile_sql)

    if out.compiled:
        try:
            import sqlglot

            sqlglot.parse_one(out.compile_sql, read="tsql")
            out.sql_parses = True
        except Exception:
            out.sql_parses = False
        # Seekable means the governed date predicate compares the KEY, not a
        # function of it. A function around the column cannot use an index.
        upper = out.compile_sql.upper()
        key = DATE_ROLES[out.date_role]["fact_column"]
        flat = upper.replace(" ", "")
        out.sql_seekable = not any(
            f"{fn}(FACT_ROWS.[{key}]" in flat
            for fn in ("CONVERT(DATE,", "CAST", "YEAR", "MONTH", "DATEPART")
        )

    out.followup_routes = should_attempt_cache_followup(
        question, True, cached_col_names=CACHED_COLUMNS,
        cached_rows=CACHED_ROWS, lang=lang)
    return out


def _wants(value: Any) -> bool:
    """YAML reads a bare `yes`/`no` as a BOOLEAN, not the string.

    Worth spelling out because comparing str(True).lower() to "yes" is False,
    which made every expectation in the corpus read as its opposite the first
    time this ran -- a harness that reported 38 failures and had none.
    """
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"yes", "true", "y", "1"}


def _stated(value: Any) -> bool:
    """Whether the corpus expressed an expectation at all."""
    return value is not None and str(value).strip() not in {"", "-"}


def judge(case: dict, en: Observation, fr: Observation) -> list[str]:
    """Where the two languages disagree, or either misses the expectation."""
    expect = dict(case.get("expect") or {})
    failures: list[str] = []

    if "window" in expect:
        want = str(expect["window"])
        for label, seen in (("EN", en.window), ("FR", fr.window)):
            if seen != want:
                failures.append(f"{label} window {seen} (expected {want})")
        if en.window != fr.window:
            failures.append(f"window differs: EN {en.window} vs FR {fr.window}")

    if en.date_role != fr.date_role:
        failures.append(
            f"date role differs: EN {en.date_role} vs FR {fr.date_role}")

    if "compile" in expect:
        want = _wants(expect["compile"])
        for label, seen in (("EN", en.compiled), ("FR", fr.compiled)):
            if seen != want:
                failures.append(
                    f"{label} {'compiled' if seen else 'did not compile'} "
                    f"(expected {'compile' if want else 'no compile'})")
        if en.compiled != fr.compiled:
            failures.append("compiler decision differs between languages")
        for label, obs in (("EN", en), ("FR", fr)):
            if obs.compiled and not obs.sql_parses:
                failures.append(f"{label} compiled SQL does not parse as T-SQL")
            if obs.compiled and obs.sql_seekable is False:
                failures.append(f"{label} compiled SQL is not seekable")

    if "template" in expect:
        want = _wants(expect["template"])
        for label, seen in (("EN", bool(en.template_matched)),
                            ("FR", bool(fr.template_matched))):
            if seen != want:
                failures.append(
                    f"{label} template {'matched' if seen else 'declined'} "
                    f"(expected {'match' if want else 'decline'})")
        if bool(en.template_matched) != bool(fr.template_matched):
            failures.append("template decision differs between languages")

    if "followup" in expect and _stated(expect["followup"]):
        want = _wants(expect["followup"])
        for label, seen in (("EN", en.followup_routes),
                            ("FR", fr.followup_routes)):
            if bool(seen) != want:
                failures.append(
                    f"{label} follow-up {'routed' if seen else 'did not route'} "
                    f"(expected {'route' if want else 'fresh query'})")
        if bool(en.followup_routes) != bool(fr.followup_routes):
            failures.append("follow-up routing differs between languages")
    return failures


def card_check() -> list[dict]:
    """The answer card over the two shapes that were reported wrong."""
    from core.i18n import activate_language, deactivate_language
    from core.insight import compute_data_brief
    from core.response_builder import (
        _build_anomaly_callouts, _build_decision_signal,
        summarize_result_context,
    )

    checks: list[dict] = []
    for label, rows, question in (
        ("warehouse ranking, top 3 hold 91.9%", CARD_RANKING,
         "net sales by warehouse"),
        ("series over an integer year column", CARD_YEARS, "net sales by year"),
    ):
        row: dict[str, Any] = {"shape": label}
        ctx = summarize_result_context(rows, question)
        brief = compute_data_brief(rows, question)
        row["mode"] = ctx.get("mode", "")
        row["measure"] = ctx.get("value_col", "")
        row["axis"] = ctx.get("label_col", "")
        row["period_cols"] = ", ".join(ctx.get("period_cols") or []) or "—"
        callouts = _build_anomaly_callouts(brief)
        for lang in ("en", "fr"):
            token = activate_language(lang)
            try:
                signal = _build_decision_signal(ctx, brief, callouts)
                row[f"signal_{lang}"] = str(signal.get("line") or "—")
            finally:
                deactivate_language(token)
        cat = brief.get("category_breakdown") or {}
        row["top3_share"] = cat.get("top_3_share_pct")
        row["callouts"] = "; ".join(c["message"] for c in callouts) or "—"
        checks.append(row)
    return checks


# ── Reporting ────────────────────────────────────────────────────────────────

def _fmt_window(obs: Observation) -> str:
    if obs.window == "none":
        return "—"
    if obs.window_amount:
        return f"{obs.window} {obs.window_amount}{obs.window_unit[:1]}"
    return obs.window


def print_report(results: list[CaseResult], cards: list[dict],
                 verbose: bool) -> None:
    width = max(len(r.id) for r in results) + 2
    print()
    print(f"{'':4}{'case'.ljust(width)}{'window EN/FR':22}{'date role EN/FR':30}"
          f"{'compile':10}{'template':11}{'follow-up':11}")
    print("-" * (width + 92))
    for r in results:
        mark = "ok  " if r.ok else "FAIL"
        window = f"{_fmt_window(r.en)} / {_fmt_window(r.fr)}"
        compile_col = f"{'y' if r.en.compiled else 'n'} / {'y' if r.fr.compiled else 'n'}"
        template = (f"{'y' if r.en.template_matched else 'n'} / "
                    f"{'y' if r.fr.template_matched else 'n'}")
        follow = (f"{'y' if r.en.followup_routes else 'n'} / "
                  f"{'y' if r.fr.followup_routes else 'n'}")
        role = (f"{r.en.date_role} / {r.fr.date_role}"
                if r.en.date_role != r.fr.date_role else r.en.date_role)
        print(f"{mark}{r.id.ljust(width)}{window:22}{role:30}{compile_col:10}"
              f"{template:11}{follow:11}")
        for failure in r.failures:
            print(f"      ! {failure}")
        if verbose:
            print(f"      EN {r.english}")
            print(f"      FR {r.french}")
            print(f"      -> {r.fr.canonical}")

    print()
    print("── the answer card on EMCO-shaped rows ──")
    for card in cards:
        print(f"  {card['shape']}")
        print(f"    mode={card['mode']}  measure={card['measure']}  "
              f"axis={card['axis']}  periods={card['period_cols']}")
        print(f"    top-3 share: {card['top3_share']}")
        print(f"    callouts   : {card['callouts']}")
        print(f"    signal EN  : {card['signal_en']}")
        print(f"    signal FR  : {card['signal_fr']}")

    failed = [r for r in results if not r.ok]
    print()
    print(f"{len(results) - len(failed)}/{len(results)} cases as expected"
          f"{'' if not failed else ' — ' + str(len(failed)) + ' to look at'}")
    for r in failed:
        print(f"  FAIL {r.id}: {'; '.join(r.failures)}")


def write_html(results: list[CaseResult], cards: list[dict], path: Path) -> None:
    esc = html_mod.escape

    def cell(en: str, fr: str, agree: bool) -> str:
        klass = "agree" if agree else "differ"
        return (f'<td class="{klass}"><span class="en">{esc(en)}</span>'
                f'<span class="fr">{esc(fr)}</span></td>')

    rows = []
    for r in results:
        rows.append(
            f'<tr class="{"ok" if r.ok else "bad"}">'
            f'<td class="mark">{"✓" if r.ok else "✗"}</td>'
            f'<td class="id">{esc(r.id)}<span class="intent">{esc(r.intent)}</span></td>'
            f'<td class="q">{esc(r.english)}<span class="fr-q">{esc(r.french)}</span>'
            f'<span class="canon">→ {esc(r.fr.canonical)}</span></td>'
            + cell(_fmt_window(r.en), _fmt_window(r.fr), r.en.window == r.fr.window)
            + cell(r.en.date_role or "—", r.fr.date_role or "—",
                   r.en.date_role == r.fr.date_role)
            + cell("yes" if r.en.compiled else "no",
                   "yes" if r.fr.compiled else "no",
                   r.en.compiled == r.fr.compiled)
            + cell(r.en.template_matched or "—", r.fr.template_matched or "—",
                   bool(r.en.template_matched) == bool(r.fr.template_matched))
            + cell("yes" if r.en.followup_routes else "no",
                   "yes" if r.fr.followup_routes else "no",
                   bool(r.en.followup_routes) == bool(r.fr.followup_routes))
            + f'<td class="why">{"<br>".join(esc(f) for f in r.failures)}'
              f'{("<em>" + esc(r.note) + "</em>") if r.note else ""}</td>'
            + "</tr>")

    card_rows = "".join(
        f"<tr><td>{esc(c['shape'])}</td><td>{esc(c['mode'])}</td>"
        f"<td>{esc(str(c['measure']))}</td><td>{esc(str(c['axis']))}</td>"
        f"<td>{esc(str(c['top3_share']))}</td>"
        f"<td>{esc(c['callouts'])}</td>"
        f"<td>{esc(c['signal_en'])}<br><span class='fr'>{esc(c['signal_fr'])}</span></td></tr>"
        for c in cards)

    failed = sum(1 for r in results if not r.ok)
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EMCO rehearsal</title>
<style>
:root{{--bg:#fbfbfa;--fg:#20201e;--dim:#6b6b66;--line:#e3e3df;
--ok:#1a7f4b;--bad:#b3261e;--card:#fff;--accent:#2f6feb}}
@media(prefers-color-scheme:dark){{:root{{--bg:#17171a;--fg:#e8e8e6;
--dim:#9a9a95;--line:#2e2e33;--card:#1f1f23;--ok:#4ac47e;--bad:#ff6b60}}}}
*{{box-sizing:border-box}}
body{{margin:0;padding:24px 16px 64px;background:var(--bg);color:var(--fg);
font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
.wrap{{max-width:1180px;margin:0 auto}}
h1{{font-size:22px;margin:0 0 4px}}
.sub{{color:var(--dim);margin:0 0 20px;max-width:70ch}}
.tally{{display:flex;gap:10px;flex-wrap:wrap;margin:0 0 22px}}
.pill{{background:var(--card);border:1px solid var(--line);border-radius:999px;
padding:5px 13px;font-variant-numeric:tabular-nums}}
.pill b{{font-size:16px}}
.pill.bad b{{color:var(--bad)}} .pill.good b{{color:var(--ok)}}
.scroll{{overflow-x:auto;border:1px solid var(--line);border-radius:10px;
background:var(--card)}}
table{{border-collapse:collapse;width:100%;min-width:940px}}
th,td{{text-align:left;padding:9px 11px;border-bottom:1px solid var(--line);
vertical-align:top}}
th{{font-size:11px;text-transform:uppercase;letter-spacing:.06em;
color:var(--dim);font-weight:600;position:sticky;top:0;background:var(--card)}}
tr:last-child td{{border-bottom:none}}
.mark{{width:26px;font-weight:700}}
tr.ok .mark{{color:var(--ok)}} tr.bad .mark{{color:var(--bad)}}
tr.bad{{background:color-mix(in srgb,var(--bad) 7%,transparent)}}
.id{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;
white-space:nowrap}}
.intent,.canon,.fr-q{{display:block;font-family:inherit;color:var(--dim);
font-size:11.5px;margin-top:2px}}
.canon{{font-family:ui-monospace,Menlo,monospace}}
.q{{min-width:290px}}
td.agree .fr{{color:var(--dim)}}
td.differ{{background:color-mix(in srgb,var(--bad) 14%,transparent);
font-weight:600}}
.en,.fr{{display:block;white-space:nowrap}}
.fr{{color:var(--dim)}}
.why{{color:var(--bad);font-size:12px;min-width:160px}}
.why em{{color:var(--dim);font-style:normal;display:block}}
h2{{font-size:15px;margin:34px 0 10px}}
.foot{{color:var(--dim);font-size:12px;margin-top:28px;max-width:78ch}}
</style></head><body><div class="wrap">
<h1>EMCO rehearsal — the deterministic pipeline, both languages</h1>
<p class="sub">Each row is one question asked two ways. The four decision columns
show <strong>English / French</strong>; a highlighted cell means the two
languages decided differently, which on stage means the same question gets two
different answers. No model was called and no query was executed.</p>
<div class="tally">
<span class="pill {'good' if not failed else ''}"><b>{len(results) - failed}</b> as expected</span>
<span class="pill {'bad' if failed else ''}"><b>{failed}</b> to look at</span>
<span class="pill"><b>{len(results)}</b> questions &times; 2 languages</span>
</div>
<div class="scroll"><table>
<thead><tr><th></th><th>case</th><th>question &middot; canonical</th>
<th>window</th><th>date role</th><th>compiles</th>
<th>metric template</th><th>follow-up</th>
<th>notes</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<h2>The answer card, on EMCO-shaped rows</h2>
<div class="scroll"><table>
<thead><tr><th>shape</th><th>mode</th><th>measure</th><th>axis</th>
<th>top-3 share</th><th>callouts</th><th>decision signal (EN / FR)</th></tr></thead>
<tbody>{card_rows}</tbody></table></div>
<p class="foot">Boundaries: the LLM and the warehouse. The compiled SQL is parsed
with sqlglot and run through the validator, not executed &mdash; so nothing here
proves a credential, an index, a row count or data freshness. Run
<code>python -m evals.emco_rehearsal</code> after any change to the
canonicaliser, the window detector, the metric registry, the governed compilers
or the follow-up router.</p>
</div></body></html>"""
    path.write_text(report, encoding="utf-8")


# ── Entry point ──────────────────────────────────────────────────────────────

def run(only: str = "", verbose: bool = False) -> list[CaseResult]:
    import store

    cases = yaml.safe_load(CORPUS.read_text(encoding="utf-8"))["cases"]
    if only:
        needle = only.lower()
        cases = [c for c in cases
                 if needle in c["id"].lower() or needle in c["intent"].lower()]

    workdir = tempfile.mkdtemp(prefix="qb-emco-rehearsal-")
    saved = {k: os.environ.get(k) for k in ("DB_PATH", "QUERYBOT_DB_PATH")}
    db_path = os.path.join(workdir, "rehearsal.db")
    os.environ["DB_PATH"] = db_path
    os.environ["QUERYBOT_DB_PATH"] = db_path
    try:
        store.init_db()
        account_id = f"emco-rehearsal-{uuid.uuid4().hex[:8]}"
        store.upsert_client(account_id, "portal")
        store.save_metric(account_id, dict(NET_SALES_TEMPLATE))

        results: list[CaseResult] = []
        for case in cases:
            en = observe(case["english"], "en", account_id)
            fr = observe(case["french"], "fr", account_id)
            results.append(CaseResult(
                id=case["id"], intent=case.get("intent", ""),
                note=case.get("note", ""),
                english=case["english"], french=case["french"],
                en=en, fr=fr, failures=judge(case, en, fr)))
        return results
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(workdir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    parser.add_argument("--only", default="",
                        help="run cases whose id or intent contains this")
    parser.add_argument("--html", default="",
                        help="also write a standalone HTML report here")
    parser.add_argument("--json", action="store_true",
                        help="emit machine-readable results instead of a table")
    parser.add_argument("--verbose", action="store_true",
                        help="print both phrasings and the canonical form")
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero when any case is not as expected")
    args = parser.parse_args()

    results = run(only=args.only, verbose=args.verbose)
    if not results:
        print("no cases matched", file=sys.stderr)
        return 2
    cards = card_check()

    if args.json:
        print(json.dumps({
            "cases": [asdict(r) for r in results],
            "cards": cards,
            "as_expected": sum(1 for r in results if r.ok),
            "total": len(results),
        }, indent=2, default=str))
    else:
        print_report(results, cards, verbose=args.verbose)

    if args.html:
        target = Path(args.html)
        target.parent.mkdir(parents=True, exist_ok=True)
        write_html(results, cards, target)
        print(f"\nHTML report: {target}")

    failed = sum(1 for r in results if not r.ok)
    return 1 if (failed and args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
