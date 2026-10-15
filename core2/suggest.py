"""A reader's start page: the questions a new conversation offers them, each one checked for them on the data.

Every card is a question the data answers (core2/question_map.py), chosen for the reader who sees it and run for
them -- on their tables, under their row rules and personal-data clearance (core2.service.portal_replay) -- once a
day, and again once the data is relearned or an admin changes a metric, a link or a date. A card that fails or
comes back empty is dropped and logged, never shown.

* For you: the figures they ask about most (core2_answer, read here by code: nothing is sent to the AI), each from
  an angle they have not looked at -- last month against the month before, over time, split another way, its top
  ten -- an analysis of their tables they have never run, and metrics an admin accepted lately. A reader with no
  questions yet gets the main figures of their tables. Three sets ("Show others" steps through them); each card
  carries its answer now: the number, its change and its trend line.
* Explore: by subject, what each of their tables answers (a trend, a split, a ranking, an analysis).
* New: the metrics an admin accepted, and the data relearned, since their last visit.

No question they asked in the last 14 days is offered again, and no person's name is ever in one: cards are made of
the names of figures and groupings, never of a member's. Kinds of card a reader asks rise, kinds they leave fall
(core2_suggest_kind). Asked as the first question of a new conversation, a card's question is answered from the
card's own plan with no AI (core2.service.portal_answer: card_plan), so the answer is the one that was checked.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from typing import Any

log = logging.getLogger(__name__)

HISTORY_DAYS = 90          # the questions a reader's suggestions are read from
RECENT_DAYS = 14           # one asked in the last 14 days is not offered again
SETS, PER_SET = 3, 3       # For you: three sets of three
SUBJECTS, PER_SUBJECT = 6, 4
MOST_MEMBERS = 30          # a split worth offering has 2 to 30 members
BUDGET = 40.0              # seconds the cards of one page are checked for on the data
WORKERS = 4
KINDS = ("number", "trend", "breakdown", "rank", "segment", "analysis")
_SEGMENTS = ("lifecycle", "cohort", "gap", "active")
_RANKS = ("pareto", "together")


@dataclass
class Card:
    kind: str                  # number | trend | breakdown | rank | segment | analysis
    question: str
    question_fr: str
    plan: dict[str, Any]
    subject: str = ""
    measure: str = ""
    group: str = ""
    analysis: str = ""
    why: str = ""
    why_fr: str = ""
    value: str = ""            # its answer now (For you)
    change: dict[str, Any] | None = None
    spark: dict[str, Any] | None = None
    better: str = "higher"     # which way the figure is good news

    @property
    def key(self) -> str:
        return hashlib.sha1(json.dumps(self.plan, sort_keys=True).encode()).hexdigest()[:12]


def _signature(plan: dict[str, Any]) -> tuple:
    """What a plan asks, whatever its period: its figures, groupings and analysis."""
    analysis = next((k for k in ("lifecycle", "cohort", "gap", "active", "pareto", "funnel", "together", "target",
                                 "on_time") if plan.get(k)), "")
    grain = bool((plan.get("time") or {}).get("grain"))
    return (tuple(sorted(str(m) for m in plan.get("measures") or [])), tuple(sorted(plan.get("group_by") or [])),
            analysis, grain and not plan.get("group_by"))


def _plain(text: str) -> str:
    return " ".join(str(text or "").casefold().split()).strip(" ?.!")


# ── what the reader asked ──────────────────────────────────────────────────────


@dataclass
class History:
    weights: dict[str, float]          # measure slug -> how much they ask about it (recent weighs more)
    grouped: dict[str, set[str]]       # measure slug -> the groupings they split it by
    trended: set[str]                  # measures they asked over time
    analyses: set[str]                 # analyses they ran
    recent: set[tuple]                 # what they asked in the last 14 days (_signature)
    recent_words: set[str]             # and in their words
    asked: int = 0


def history(account_id: str, user_id: str, today: dt.date) -> History:
    import store

    since = (today - dt.timedelta(days=HISTORY_DAYS)).isoformat()
    try:
        rows = store.list_core2_answers_of(account_id, user_id, since)
    except Exception as exc:  # noqa: BLE001 - a reader with no readable history gets the main figures
        log.warning("core2: the questions of reader %s of %s could not be read: %s", user_id, account_id, exc)
        rows = []
    out = History(weights={}, grouped={}, trended=set(), analyses=set(), recent=set(), recent_words=set())
    recent_since = today - dt.timedelta(days=RECENT_DAYS)
    for row in rows:
        plan = row.get("plan") or {}
        try:
            day = dt.date.fromisoformat(str(row.get("created_at") or "")[:10])
        except ValueError:
            day = today
        if day >= recent_since:
            out.recent_words.add(_plain(row.get("question") or ""))
            if plan:
                out.recent.add(_signature(plan))
        if not plan:
            continue
        out.asked += 1
        weight = 1.0 / (1.0 + max(0, (today - day).days) / 14.0)
        measures = [str(m) for m in plan.get("measures") or []]
        measures += [str(m) for d in plan.get("derived") or [] if isinstance(d, dict) for m in d.get("measures") or []]
        for key in ("lifecycle", "cohort", "gap", "pareto", "together", "target", "on_time", "active", "funnel"):
            part = plan.get(key)
            if part:
                out.analyses.add(key)
                if isinstance(part, dict) and part.get("measure"):
                    measures.append(str(part["measure"]))
        for slug in dict.fromkeys(measures):
            out.weights[slug] = out.weights.get(slug, 0.0) + weight
            out.grouped.setdefault(slug, set()).update(str(g) for g in plan.get("group_by") or [])
            if (plan.get("time") or {}).get("grain"):
                out.trended.add(slug)
    return out


# ── the cards ──────────────────────────────────────────────────────────────────


def _kind_of(entry: Any) -> str:
    if entry.kind == "figure":
        return "number"
    if entry.kind == "split":
        return "breakdown"
    if entry.analysis in _SEGMENTS:
        return "segment"
    return "rank" if entry.analysis in _RANKS else "analysis"


def _from_entry(entry: Any) -> Card:
    return Card(kind=_kind_of(entry), question=entry.question, question_fr=entry.question_fr or entry.question,
                plan=dict(entry.plan), subject=entry.subject, measure=entry.measure, group=entry.group,
                analysis=entry.analysis)


def _number(measure: Any, dated: bool) -> Card:
    """The figure last month (against the month before, with its line): a whole period, so its answer carries
    both. A figure with no date is one number."""
    name = measure.business_name
    if not dated:
        return Card(kind="number", question=name, question_fr=name,
                    plan={"intent": "value", "measures": [measure.slug]}, measure=measure.slug)
    return Card(kind="number", question=f"{name} last month", question_fr=f"{name} le mois dernier",
                plan={"intent": "value", "measures": [measure.slug],
                      "time": {"window": {"kind": "previous", "unit": "month"}}}, measure=measure.slug)


def _trend(measure: Any) -> Card:
    name = measure.business_name
    return Card(kind="trend", question=f"{name} by month over the last 12 months",
                question_fr=f"{name} par mois sur les 12 derniers mois",
                plan={"intent": "trend", "measures": [measure.slug],
                      "time": {"grain": "month", "window": {"kind": "last", "unit": "month", "n": 12}}},
                measure=measure.slug)


def _rank(measure: Any, entity: Any) -> Card:
    from core2.bootstrap import names

    many = names.plural(entity.business_name.lower())
    return Card(kind="rank", question=f"Top 10 {many} by {measure.business_name.lower()}",
                question_fr=f"Top 10 des {many} par {measure.business_name.lower()}",
                plan={"intent": "rank", "measures": [measure.slug], "group_by": [entity.slug],
                      "sort": [{"by": measure.slug, "desc": True}], "limit": 10},
                measure=measure.slug, group=entity.slug)


def _why(card: Card, en: str, fr: str) -> Card:
    card.why, card.why_fr = en, fr
    return card


def _members(model: Any, entry: Any) -> int:
    """How many members a split shows: as checked on the data (up to 50), else as Learn counted them."""
    if entry.rows is not None:
        return int(entry.rows)
    attribute = model.attributes.get(entry.group)
    if attribute is not None:
        column = model.columns.get(attribute.column)
        return int(attribute.members or (column.profile.distinct if column is not None and column.profile else 0) or 0)
    entity = model.entities.get(entry.group)
    table = model.tables.get(entity.table) if entity is not None else None
    return int(table.row_count or 0) if table is not None else 0


def _counts_its_own_list(model: Any, entry: Any) -> bool:
    """"Number of customers by customer", "number of product categories by product category": a list's count split
    by the list itself is one each."""
    measure = next((m for m in model.measures.values() if m.slug == entry.measure), None)
    if measure is None or not _a_count(measure):
        return False
    entity = model.entities.get(entry.group)
    if entity is not None:
        return entity.table == measure.table
    attribute = model.attributes.get(entry.group)
    column = model.columns.get(attribute.column) if attribute is not None else None
    table = model.tables.get(measure.table)
    return column is not None and column.table == measure.table and table is not None \
        and _members(model, entry) >= 0.9 * (table.row_count or 0) > 0


def _good_split(model: Any, entry: Any) -> bool:
    """A split worth offering: a few members to compare, not thousands of accounts."""
    return entry.kind == "split" and 2 <= _members(model, entry) <= MOST_MEMBERS \
        and not _counts_its_own_list(model, entry)


def _ranked(model: Any, entry: Any) -> bool:
    """A list long enough for its top ten to say something."""
    return entry.kind == "split" and entry.group in model.entities and _members(model, entry) > 10 \
        and not _counts_its_own_list(model, entry)


def _a_sum(measure: Any) -> bool:
    """A figure that adds up (an amount, a quantity): the lead of a table before an average or a rate of it."""
    from core2.model.schema import AggExpr

    return isinstance(measure.expr, AggExpr) and measure.expr.agg == "sum"


def _a_count(measure: Any) -> bool:
    from core2.model.schema import AggExpr

    return isinstance(measure.expr, AggExpr) and measure.expr.agg in ("count", "count_distinct")


def _main_figures(model: Any, entries: list[Any]) -> list[Any]:
    """The figures a reader's tables are mostly about: an admin's metrics first, then those answered by the most
    groupings (the figure a table is kept for is split every way)."""
    splits: dict[str, int] = {}
    for e in entries:
        if e.kind == "split":
            splits[e.measure] = splits.get(e.measure, 0) + 1
    figures = {e.measure for e in entries if e.kind == "figure"}
    found = [m for m in model.measures.values() if m.slug in figures]
    listed = {e.table for e in model.entities.values()}
    # What happened (amounts, counts of activity) before how many there are of a list (ingredients, formulas), and
    # before a bare count of a table's rows ("number of sales target rows").
    found.sort(key=lambda m: (m.kind not in ("metric", "import", "model"), _a_count(m) and m.table in listed,
                              _a_count(m) and m.business_name.lower().endswith(" rows"), -splits.get(m.slug, 0),
                              not _a_sum(m), m.format != "currency", m.slug))
    return found


def _for_you(model: Any, entries: list[Any], seen: History, accepted: list[Any]) -> list[Card]:
    """Every card For you could hold, the likeliest first: a round of each figure's first angle, then of each
    one's next, so a set holds several figures rather than one four ways."""
    by_slug = {m.slug: m for m in model.measures.values()}
    figures = {e.measure for e in entries if e.kind == "figure"}
    first: list[Card] = []
    for measure in accepted:
        if measure.slug in figures:
            first.append(_why(_number(measure, bool(measure.default_date)), "A metric an admin accepted lately",
                              "Une mesure acceptée récemment par un administrateur"))
    asked = [by_slug[s] for s, _ in sorted(seen.weights.items(), key=lambda kv: -kv[1])
             if s in by_slug and s in figures]
    mine = {m.slug for m in asked}
    main = asked[:4] or _main_figures(model, entries)[:4]
    angles: list[list[Card]] = []
    for measure in main:
        name = measure.business_name
        why = (f"From your questions on {name.lower()}", f"D'après vos questions sur {name.lower()}") \
            if measure.slug in mine else (f"A main figure of {_subject(model, measure).lower()}",
                                          f"Un chiffre clé : {_subject(model, measure).lower()}")
        cards = [_number(measure, bool(measure.default_date))]
        used = seen.grouped.get(measure.slug, set())
        split = next((e for e in entries if e.measure == measure.slug and _good_split(model, e)
                      and e.group not in used), None)
        if split is not None:
            cards.append(_from_entry(split))
        if measure.default_date and measure.slug not in seen.trended:
            cards.append(_trend(measure))
        listed = next((e for e in entries if e.measure == measure.slug and _ranked(model, e) and e.group not in used),
                      None)
        if listed is not None:
            cards.append(_rank(measure, model.entities[listed.group]))
        for card in cards:
            card.better = measure.better
        angles.append([_why(c, *why) for c in cards])
    tables = {m.table for m in main}
    analyses = [_why(_from_entry(e), "An analysis you have not run yet", "Une analyse que vous n'avez pas encore faite")
                for e in entries if e.kind == "analysis" and e.analysis not in seen.analyses
                and (not e.measure or (e.measure in by_slug and by_slug[e.measure].table in tables))]
    out = list(first)
    for depth in range(max((len(a) for a in angles), default=0)):
        out += [a[depth] for a in angles if depth < len(a)]
        if depth < len(analyses):
            out.append(analyses[depth])
    return out + analyses[max((len(a) for a in angles), default=0):]


def _subject(model: Any, measure: Any) -> str:
    table = model.tables.get(measure.table)
    return (table.business_name or table.name) if table is not None else measure.business_name


def _explore(model: Any, entries: list[Any], seen: History) -> list[tuple[str, list[Card]]]:
    """By subject, what the reader's tables answer: an analysis, a trend, a split, a ranking."""
    by_subject: dict[str, list[Any]] = {}
    for e in entries:
        if e.subject:
            by_subject.setdefault(e.subject, []).append(e)
    asked_tables = {model.measures[k].table for k in model.measures if model.measures[k].slug in seen.weights}
    by_slug = {m.slug: m for m in model.measures.values()}

    listed = {e.table for e in model.entities.values()}

    def rank(subject: str) -> tuple:
        measures = [by_slug[e.measure] for e in by_subject[subject] if e.measure in by_slug]
        mine = any(m.table in asked_tables for m in measures)
        # A table of what happened (orders, fills) before a list of things (customers, products) counted.
        a_list = all(_a_count(m) and m.table in listed for m in measures)
        return (not mine, a_list, -sum(e.kind == "figure" for e in by_subject[subject]), subject)

    out = []
    for subject in sorted(by_subject, key=rank):
        found = by_subject[subject]
        mains = _main_figures(model, found)
        cards: list[Card] = []
        cards += [_from_entry(e) for e in found if e.kind == "analysis"][:1]
        if mains and mains[0].default_date:
            cards.append(_trend(mains[0]))
        for measure in mains[:2]:
            split = next((e for e in found if e.measure == measure.slug and _good_split(model, e)), None)
            if split is not None:
                cards.append(_from_entry(split))
            listed = next((e for e in found if e.measure == measure.slug and _ranked(model, e)), None)
            if listed is not None:
                cards.append(_rank(measure, model.entities[listed.group]))
        if len(mains) > 1:
            cards.append(_number(mains[1], bool(mains[1].default_date)))
        for card in cards:
            card.subject = subject
        out.append((subject, cards))
        if len(out) >= SUBJECTS * 2:
            break
    return out


# ── checked for the reader ────────────────────────────────────────────────────


def _rows(payload: dict[str, Any]) -> list:
    return list((payload.get("data") or {}).get("rows") or [])


def answered(payload: dict[str, Any] | None) -> bool:
    """Did the plan give this reader an answer with a value in it?"""
    if not isinstance(payload, dict) or payload.get("unsupported") or payload.get("kind") in ("refused", "denied"):
        return False
    for row in _rows(payload):
        values = row.values() if isinstance(row, dict) else row
        if any(v is not None and v != "" for v in values):
            return True
    return False


def _preview(card: Card, payload: dict[str, Any]) -> dict[str, Any]:
    """What a For you card shows of its answer: the number with its change and line, or a trend's last period."""
    from core2.answer.builder import sparkline

    kpi = payload.get("kpi") if isinstance(payload.get("kpi"), dict) else None
    answer = payload.get("answer") or {}
    if kpi is not None and kpi.get("state") == "ready" and card.kind != "trend":
        change = kpi.get("change") if isinstance(kpi.get("change"), dict) else None
        return {"value": str(answer.get("short_value") or ""), "spark": kpi.get("trend"),
                "change": {k: change.get(k) for k in ("direction", "pct", "change", "good", "vs")} if change else None}
    data = payload.get("data") or {}
    headers = list(data.get("headers") or [])
    if card.kind == "trend" and len(headers) == 2 and headers[0] == "period":
        # The last month and the one before it, and the year's line.
        from core2.answer.builder import against, fmt

        measure, format_ = headers[1], str((data.get("column_formats") or {}).get(headers[1]) or "number")
        values = [_amount(row.get(measure)) if isinstance(row, dict) else None for row in _rows(payload)]
        known = [v for v in values if v is not None]
        if not known:
            return {"value": "", "spark": None, "change": None}
        moved = against(values[-1], values[-2], format_, "", better=card.better) \
            if len(values) >= 2 and values[-1] is not None and values[-2] is not None else None
        return {"value": fmt(values[-1], format_) if values[-1] is not None else "", "spark": sparkline(values),
                "change": {k: moved.get(k) for k in ("direction", "pct", "change", "good")} if moved else None}
    return {"value": "", "spark": None, "change": None}


def _amount(value: Any) -> float | None:
    try:
        return None if value is None or isinstance(value, bool) else float(value)
    except (TypeError, ValueError):
        return None


def _check(cards: list[Card], run: Callable[[dict, str], dict], *, budget: float, done: dict[str, dict],
           previews: set[str]) -> None:
    """Run each card not yet checked today (``done``: key -> verdict), several at once, within ``budget``
    seconds; a card not reached is checked next time, not shown now."""
    todo = [c for c in dict((c.key, c) for c in cards).values() if c.key not in done]
    if not todo:
        return
    began = time.monotonic()
    pool = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="start-page")
    try:
        waiting = {pool.submit(run, c.plan, c.question): c for c in todo}
        while waiting:
            left = budget - (time.monotonic() - began)
            if left <= 0:
                break
            finished, _ = wait(list(waiting), timeout=left, return_when=FIRST_COMPLETED)
            for future in finished:
                card = waiting.pop(future)
                try:
                    payload = future.result()
                except Exception as exc:  # noqa: BLE001 - a card that does not run is not offered
                    log.info("core2: suggestion %r did not run: %s", card.question, exc)
                    done[card.key] = {"ok": False}
                    continue
                ok = answered(payload)
                if not ok:
                    log.info("core2: suggestion %r gave this reader no answer; it is not offered", card.question)
                verdict: dict[str, Any] = {"ok": ok}
                if ok and card.key in previews:
                    verdict.update(_preview(card, payload))
                done[card.key] = verdict
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


# ── the page ───────────────────────────────────────────────────────────────────


def _weights(account_id: str, user_id: str) -> dict[str, float]:
    import store

    try:
        counts = store.core2_suggest_kinds(account_id, user_id)
    except Exception:  # noqa: BLE001 - every kind weighs the same
        counts = {}
    return {k: (counts.get(k, (0, 0))[1] + 1) / (counts.get(k, (0, 0))[0] + 3) for k in KINDS}


def _sets(cards: list[Card], weights: dict[str, float]) -> list[list[Card]]:
    """For you's sets: the likeliest first, each of different kinds where it can be; kinds the reader asks
    come before kinds they leave."""
    order = {id(c): i for i, c in enumerate(cards)}
    ranked = sorted(cards, key=lambda c: (order[id(c)] // 6, -weights.get(c.kind, 0.0), order[id(c)]))
    left = list(ranked)
    out: list[list[Card]] = []
    while left and len(out) < SETS:
        chosen: list[Card] = []
        for card in list(left):
            if len(chosen) < PER_SET and card.kind not in {c.kind for c in chosen}:
                chosen.append(card)
                left.remove(card)
        for card in list(left):
            if len(chosen) < PER_SET:
                chosen.append(card)
                left.remove(card)
        out.append(chosen)
    return out


def _fresh(card: Card, seen: History) -> bool:
    return _signature(card.plan) not in seen.recent and _plain(card.question) not in seen.recent_words \
        and _plain(card.question_fr) not in seen.recent_words


def build(account_id: str, portal_user: dict[str, Any], *, client: dict[str, Any] | None = None,
          today: dt.date | None = None, run: Callable[[dict, str], dict] | None = None,
          budget: float = BUDGET, kept: dict[str, Any] | None = None, since: str = "") -> dict[str, Any]:
    """A reader's start page, language-neutral (``view`` says it in theirs). ``kept``: their page kept before,
    whose cards checked today are not run again; ``since``: their last visit, for what is new."""
    import store
    from core2 import question_map as Q
    from core2.bootstrap.service import answering_model

    today = today or dt.date.today()
    client = client or store.get_client(account_id) or {}
    user_id = str(portal_user.get("id") or "")
    model = answering_model(account_id, client)
    if model is None:
        return {"state": "learning", "day": today.isoformat()}
    from core2.service import _allowed_model_tables

    allowed = store.get_allowed_tables(portal_user)
    tables = _allowed_model_tables(model, set(allowed) if allowed is not None else None)
    visible = [k for k, t in model.tables.items() if not getattr(t, "hidden", False) and (tables is None or k in tables)]
    if tables is not None and not tables:
        return {"state": "no_access", "day": today.isoformat()}
    db_id = client.get("db_config_id")
    # Read only: a map made before the latest change still holds what was answerable (every card is run again
    # below); the chat's own questions start its rebuild.
    qmap = Q.current(account_id, client, rebuild=False) or Q.load(account_id, db_id)
    data = _data_line(model, visible, account_id, db_id)
    if qmap is None:
        return {"state": "mapping", "day": today.isoformat(), "data": data}
    entries = Q.answerable(qmap, tables)
    seen = history(account_id, user_id, today)
    accepted_keys = []
    if since:
        try:
            accepted_keys = store.core2_metrics_accepted_since(account_id, db_id, since)
        except Exception as exc:  # noqa: BLE001 - nothing said new
            log.warning("core2: the metrics accepted for %s could not be read: %s", account_id, exc)
    accepted = [model.measures[k] for k in accepted_keys if k in model.measures and not model.measures[k].hidden
                and (tables is None or model.measures[k].table in tables)]
    candidates = [c for c in _dedupe(_for_you(model, entries, seen, accepted)) if _fresh(c, seen)]
    explore = [(s, [c for c in _dedupe(cards) if _fresh(c, seen)]) for s, cards in _explore(model, entries, seen)]
    done: dict[str, dict] = dict(((kept or {}).get("page") or {}).get("runs") or {}) \
        if (kept or {}).get("stamp") == qmap.stamp and (kept or {}).get("day") == today.isoformat() else {}
    if run is None:
        from core2.service import portal_replay

        def run(plan: dict, question: str) -> dict:
            return portal_replay(account_id, plan, portal_user, question=question)
    began = time.monotonic()
    _check(candidates[:SETS * PER_SET * 2], run, budget=budget, done=done,
           previews={c.key for c in candidates})
    left = max(5.0, budget - (time.monotonic() - began))
    _check([c for _, cards in explore[:SUBJECTS] for c in cards[:PER_SUBJECT + 1]], run, budget=left, done=done,
           previews=set())
    for card in candidates:
        verdict = done.get(card.key) or {}
        card.value, card.change, card.spark = verdict.get("value") or "", verdict.get("change"), verdict.get("spark")
    today_kept = (kept or {}).get("day") == today.isoformat()
    weights = dict(((kept or {}).get("page") or {}).get("weights") or {}) if today_kept else {}
    weights = weights or _weights(account_id, user_id)      # the day's: a page made again today keeps its order
    sets = _sets([c for c in candidates if (done.get(c.key) or {}).get("ok")], weights)
    subjects = []
    for subject, cards in explore:
        kept_cards = [c for c in cards if (done.get(c.key) or {}).get("ok")][:PER_SUBJECT]
        if len(kept_cards) >= 2:
            subjects.append({"subject": subject, "cards": [asdict(c) for c in kept_cards]})
        if len(subjects) >= SUBJECTS:
            break
    if not today_kept:
        # Offered once a day, however often the page is made again: a kind's weight holds for the day, so the
        # same reader sees the same sets until they ask something new.
        try:
            for kind in {c.kind for s in sets for c in s}:
                store.count_core2_suggest_kind(account_id, user_id, kind, shown=1)
        except Exception as exc:  # noqa: BLE001 - the page stands without the count
            log.warning("core2: suggestions shown to %s could not be counted: %s", user_id, exc)
    new = None
    learned = data.get("learned") or ""
    if accepted or (since and learned and learned > since[:10]):
        new = {"metrics": [m.business_name for m in accepted][:8], "count": len(accepted),
               "relearned": learned if since and learned and learned > since[:10] else ""}
    return {"state": "ready", "day": today.isoformat(), "stamp": qmap.stamp, "data": data, "new": new,
            "for_you": [[asdict(c) for c in s] for s in sets], "explore": subjects, "runs": done,
            "asked": seen.asked, "weights": weights}


def _dedupe(cards: list[Card]) -> list[Card]:
    out, keys = [], set()
    for card in cards:
        if card.key not in keys:
            keys.add(card.key)
            out.append(card)
    return out


def _data_line(model: Any, visible: list[str], account_id: str, db_id: Any) -> dict[str, Any]:
    """How fresh the data is and how much of it the reader may use: the last day of their tables' activity (never
    one ahead of today: a due date is not data), how many tables, and when it was last learned."""
    import store

    today = dt.date.today()
    lasts = [r.last for r in model.date_roles.values() if r.table in visible and r.last is not None
             and r.kind == "event" and r.is_default and r.last <= today]
    learned = ""
    try:
        build = store.latest_core2_build(account_id, db_id)
        if build and build.get("status") in ("done", "ok", "finished", "success"):
            learned = str(build.get("finished_at") or "")[:10]
    except Exception:  # noqa: BLE001 - the model's own time below
        build = None
    if not learned and model.built_at is not None:
        learned = model.built_at.date().isoformat()
    return {"until": max(lasts).isoformat() if lasts else "", "tables": len(visible), "learned": learned}


# ── kept, served, asked ───────────────────────────────────────────────────────

_BUILDING: set[tuple[str, str]] = set()
_LOCK = threading.Lock()


def page(account_id: str, portal_user: dict[str, Any], *, lang: str = "en", wait_for: bool = True,
         wait_seconds: float = 8.0, today: dt.date | None = None) -> dict[str, Any]:
    """The reader's start page in their language: the one kept today while it is still true, else built in the
    background and waited for up to ``wait_seconds``; "building" until then (the page asks again)."""
    import store

    today = today or dt.date.today()
    user_id = str(portal_user.get("id") or "")
    client = store.get_client(account_id) or {}
    kept = store.load_core2_start_page(account_id, user_id)
    if _current(kept, account_id, user_id, client, today):
        return view(kept["page"], lang)
    done = _in_background(account_id, portal_user, today)
    if wait_for and done.wait(wait_seconds):
        kept = store.load_core2_start_page(account_id, user_id)
        if kept and kept["day"] == today.isoformat():
            return view(kept["page"], lang)
    return {"state": "building"}


def _current(kept: dict[str, Any] | None, account_id: str, user_id: str, client: dict[str, Any],
             today: dt.date) -> bool:
    """Is the kept page still true: today's, on the map as it is now, after the reader's last question?"""
    from core2 import question_map as Q

    if not kept or kept["day"] != today.isoformat() or not kept["stamp"]:
        return False
    qmap = Q.current(account_id, client, rebuild=True)
    return qmap is not None and kept["stamp"] == qmap.stamp \
        and (kept["page"] or {}).get("mark") == _mark(account_id, user_id, today)


def _mark(account_id: str, user_id: str, today: dt.date) -> str:
    """What the reader has asked lately, cheaply: a new question makes For you new again."""
    import store

    try:
        rows = store.list_core2_answers_of(account_id, user_id, (today - dt.timedelta(days=RECENT_DAYS)).isoformat(),
                                           limit=1)
    except Exception:  # noqa: BLE001
        return ""
    return str(rows[0]["created_at"]) if rows else ""


def _build_and_keep(account_id: str, portal_user: dict[str, Any], today: dt.date, kept: dict | None,
                    marked: str, **kw: Any) -> dict[str, Any]:
    import store

    user_id = str(portal_user.get("id") or "")
    # Their last visit, for what is new since: the page kept before today was made then; a page made again later
    # today keeps the same visit, so the whole day says the same.
    since = str((kept or {}).get("seen_at") or "") if (kept or {}).get("day") == today.isoformat() else \
        str((kept or {}).get("updated_at") or "")
    built = build(account_id, portal_user, today=today, kept=kept, since=since, **kw)
    built["mark"] = marked
    if built.get("state") in ("ready", "no_access", "learning", "mapping"):
        try:
            store.save_core2_start_page(account_id, user_id, day=today.isoformat(), stamp=str(built.get("stamp") or ""),
                                        page=built, seen_at=since)
        except Exception as exc:  # noqa: BLE001 - shown now, built again next time
            log.warning("core2: the start page of %s could not be kept: %s", user_id, exc)
    return built


_DONE: dict[tuple[str, str], threading.Event] = {}


def _in_background(account_id: str, portal_user: dict[str, Any], today: dt.date) -> threading.Event:
    """Build the reader's page on a thread of its own, once at a time; the event is set when it is kept."""
    import store

    key = (account_id, str(portal_user.get("id") or ""))
    with _LOCK:
        if key in _BUILDING:
            return _DONE[key]
        _BUILDING.add(key)
        done = _DONE[key] = threading.Event()

    def work() -> None:
        try:
            kept = store.load_core2_start_page(*key)
            _build_and_keep(account_id, portal_user, today, kept, _mark(*key, today))
        except Exception as exc:  # noqa: BLE001 - asked again on the next new conversation
            log.warning("core2: the start page of %s could not be built: %s", key[1], exc, exc_info=True)
        finally:
            with _LOCK:
                _BUILDING.discard(key)
            done.set()

    threading.Thread(target=work, daemon=True, name=f"start-page-{key[1]}").start()
    return done


_KIND_WORDS = {"number": ("Number", "Nombre"), "trend": ("Trend", "Tendance"), "breakdown": ("Breakdown", "Répartition"),
               "rank": ("Ranking", "Classement"), "segment": ("Segment", "Segment"),
               "analysis": ("Analysis", "Analyse")}


def view(built: dict[str, Any], lang: str = "en") -> dict[str, Any]:
    """What the page is sent as: the reader's language, no plans (a card is asked by its words)."""
    fr = lang == "fr"

    def card(c: dict[str, Any]) -> dict[str, Any]:
        value = str(c.get("value") or "")
        if fr:      # a time between events: "4.7 days" is "4.7 jours"
            value = re.sub(r"\b(\d[\d.,]*) days?\b", lambda m: f"{m.group(1)} jour{'s' if m.group(0).endswith('s') else ''}",
                           value)
        return {"id": Card(kind=c["kind"], question=c["question"], question_fr=c["question_fr"], plan=c["plan"]).key,
                "kind": c["kind"], "label": _KIND_WORDS.get(c["kind"], ("", ""))[1 if fr else 0],
                "question": c["question_fr"] if fr else c["question"],
                "why": (c.get("why_fr") if fr else c.get("why")) or "", "value": value,
                "change": c.get("change"), "spark": c.get("spark")}

    out: dict[str, Any] = {"state": built.get("state") or "ready", "data": built.get("data") or {},
                           "new": built.get("new")}
    if out["state"] != "ready":
        return out
    out["for_you"] = [[card(c) for c in s] for s in built.get("for_you") or []]
    out["explore"] = [{"subject": s["subject"], "cards": [card(c) for c in s["cards"]]}
                      for s in built.get("explore") or []]
    out["asked"] = int(built.get("asked") or 0)
    return out


def card_plan(account_id: str, user_id: str, question: str) -> tuple[dict[str, Any], str] | None:
    """The plan of today's card the reader asked by its words, and its kind; None when the words are no card's.
    Counted as asked: kinds a reader asks rise."""
    import store

    try:
        kept = store.load_core2_start_page(account_id, str(user_id))
    except Exception:  # noqa: BLE001 - planned as any question
        return None
    if not kept:
        return None
    words = _plain(question)
    built = kept["page"] or {}
    for c in [c for s in built.get("for_you") or [] for c in s] + \
             [c for s in built.get("explore") or [] for c in s.get("cards") or []]:
        if words in (_plain(c.get("question") or ""), _plain(c.get("question_fr") or "")):
            try:
                store.count_core2_suggest_kind(account_id, str(user_id), str(c.get("kind") or ""), clicked=1)
            except Exception:  # noqa: BLE001
                pass
            return dict(c.get("plan") or {}), str(c.get("kind") or "")
    return None


__all__ = ["Card", "build", "page", "view", "card_plan", "answered", "history"]
