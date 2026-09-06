"""Draft the modelling work, so the human only has to confirm it.

``core.model_readiness`` answers *what to do next*. It hands an admin a list of
gaps ordered by how many failing questions each one unblocks — which is the
right list, and still leaves fourteen paragraphs and forty synonym fields to be
typed by somebody who has an actual job. A backlog that names the work is
better than no backlog and is still work.

This module drafts the entries. Same discipline as
``core.table_description_author`` and ``core.graph_autopopulate``:

  **Nothing here writes.** Every drafter returns ``Draft`` objects carrying a
  before, an after, the evidence they were derived from, and a confidence. A
  proposal becomes a change when a person accepts it, and never before. That is
  not politeness. A column's role decides whether a filter reads as a date or a
  category; a synonym decides which measure a question resolves to. Applying
  either automatically moves answers silently, which is the failure class every
  bug on this branch belongs to.

  **A confirmed row is never proposed over.** ``entity_properties`` carries
  ``status``; a row a human has confirmed is the answer, and a drafter that
  proposes a change to it is asking the admin to re-decide something they
  already decided. Suggested rows and missing rows are fair game.

  **Every draft carries where it came from.** ``reason`` says it in words and
  ``evidence`` lists the specific inputs. A reviewer approving forty items in a
  sitting is only as good as what they can see, and "the model said so" is not
  reviewable.

Three sources, in descending order of how sure they are:

1. **Date roles** — deterministic. ``core.date_roles`` already knows that
   INVOICE_DT_DMS_KEY is the invoice date; nothing was writing that knowledge
   into the workspace where the resolver reads it.
2. **Column vocabulary from the value index** — evidence from two places at
   once: a column whose stored values appear verbatim in questions people
   asked. That is a column readers filter on without ever naming, so it needs
   the words they *do* use.
3. **Metric shapes** — question shapes asked repeatedly that never bound to a
   metric. Repetition is the evidence; one person asking once is a question,
   the same shape eight times is a missing measure.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger("querybot.model_drafts")

# A draft below this is noise. The date-role drafter clears it comfortably;
# the mined ones have to earn it.
MIN_CONFIDENCE = 40

# How many questions are read when harvesting column vocabulary. Drafting is an
# admin-time report, not a query-time cost, but an unbounded scan of a busy
# account's log is still a bad neighbour.
MAX_QUESTIONS = 600

# A shape asked fewer times than this is a question, not a missing metric.
MIN_SHAPE_OCCURRENCES = 3

# The most drafts of one kind returned at once. A review queue longer than an
# afternoon is a review queue nobody starts -- the same reasoning as
# model_readiness.MAX_CURATION_ITEMS, and the count is reported separately so
# the admin knows the list was cut rather than exhausted.
MAX_DRAFTS_PER_KIND = 25

_WORD = re.compile(r"[a-z0-9]+")

# The parts of a question that change between two askings of it. Months and
# relative periods belong here for the same reason years do: "gross margin for
# March" and "gross margin for April" are one missing measure asked twice, and
# a shape that keeps the month counts them as two shapes of one occurrence
# each -- which is exactly the threshold that then refuses to propose either.
_PERIOD_WORDS = frozenset("""
january february march april may june july august september october november
december jan feb mar apr jun jul aug sep sept oct nov dec
janvier fevrier février mars avril mai juin juillet aout août septembre octobre
novembre decembre décembre
q1 q2 q3 q4 h1 h2 fy ytd mtd qtd
today yesterday tomorrow week weeks month months quarter quarters year years
day days last previous prior next current this ago since until between
hier aujourd hui semaine semaines mois trimestre trimestres annee année annees
années jour jours dernier derniere dernière derniers precedent précédent
prochain courant depuis
""".split())

# Words that name no column and measure nothing. Deliberately short: an
# aggressive list drops the domain words that are the whole point, and every
# candidate is gated on repetition anyway.
_STOPWORDS = frozenset("""
a an and are as at be by can could de des du et for from give had has have how in
into is it its la le les me most much my of on or our per please que qui show
that the their there these they this those to top total un une versus vs was
were what when where which who why will with would you your
combien comment dans est et il ils je la le les leur ma mes moi mon ne nos notre
nous ou par pas plus pour quand que quel quelle qui sa se ses son sont sur ta tes
toi ton tous tout un une vos votre vous y
""".split())


@dataclass(frozen=True)
class Draft:
    """One proposed change to one column's declared meaning.

    ``before`` is the entity_properties row as it stands (empty when there is
    none). ``payload`` is what the drafter proposes it become. Only the keys
    present in ``payload`` are being proposed -- a drafter that has an opinion
    about the role and none about synonyms says so by omitting the key, rather
    than by proposing the current value back.
    """

    kind: str
    entity: str
    column: str
    payload: dict
    reason: str
    confidence: int
    before: dict = field(default_factory=dict)
    evidence: tuple[str, ...] = ()

    @property
    def target(self) -> str:
        return f"{self.entity}.{self.column}" if self.column else self.entity

    @property
    def changes(self) -> dict[str, tuple[str, str]]:
        """``{field: (before, after)}`` for the fields this draft would change.

        The diff a reviewer reads. Fields whose proposed value equals what is
        already there are absent: showing somebody an unchanged row and asking
        them to approve it wastes the only thing a review queue spends, which
        is their attention.
        """
        out: dict[str, tuple[str, str]] = {}
        for key, proposed in (self.payload or {}).items():
            current = str((self.before or {}).get(key) or "")
            after = str(proposed or "")
            if current != after:
                out[key] = (current, after)
        return out

    @property
    def is_change(self) -> bool:
        return bool(self.changes)

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "entity": self.entity,
            "column": self.column,
            "target": self.target,
            "payload": dict(self.payload or {}),
            "before": dict(self.before or {}),
            "changes": {k: list(v) for k, v in self.changes.items()},
            "reason": self.reason,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
        }


def _property_key(entity: str, column: str) -> tuple[str, str]:
    return (str(entity or "").strip().upper(), str(column or "").strip().upper())


def index_properties(properties: list[dict] | None) -> dict[tuple[str, str], dict]:
    """``entity_properties`` rows keyed for lookup, upper-cased on both parts.

    Case is the quiet one. The graph stores entity names as the admin typed
    them and column names as the warehouse spells them, and a drafter that
    looks up "Orders"/"order_date" against a row stored as "ORDERS"/"ORDER_DATE"
    finds nothing, proposes over a confirmed row, and the admin is asked to
    re-confirm something they already did.
    """
    return {
        _property_key(row.get("entity_name"), row.get("column_name")): dict(row)
        for row in (properties or [])
    }


def _is_confirmed(row: dict | None) -> bool:
    # A row with no status at all is confirmed: the column defaults to
    # 'confirmed', so a NULL means an old row written before the column
    # existed, and those are admin-authored.
    if not row:
        return False
    return str(row.get("status") or "confirmed") == "confirmed"


# ── 1 · Date roles ────────────────────────────────────────────────────────────

def date_role_drafts(
    columns: list[dict] | None,
    properties: list[dict] | None = None,
    *,
    limit: int = MAX_DRAFTS_PER_KIND,
) -> list[Draft]:
    """Propose the business date role for every date-role column.

    ``core.date_roles`` has known since it was written that INVOICE_DT_DMS_KEY
    is the invoice date. Nothing wrote that into ``entity_properties``, so the
    resolver -- which reads the workspace, not the vocabulary -- saw an
    unclassified key and a question about invoice dates had to guess which of
    six date columns to filter on.

    Deterministic and cheap: no model call, no sampled values, nothing that
    needs an egress budget.
    """
    from core.date_roles import DATE_ROLES, date_role_terms, detect_date_role

    by_key = {role.key: role for role in DATE_ROLES}

    known = index_properties(properties)
    drafts: list[Draft] = []
    for entry in (columns or []):
        entity = str(entry.get("entity") or entry.get("entity_name") or "").strip()
        column = str(entry.get("column") or entry.get("column_name")
                     or entry.get("name") or "").strip()
        if not entity or not column:
            continue

        current = known.get(_property_key(entity, column))
        if _is_confirmed(current):
            continue

        role = detect_date_role(column)
        if role is None:
            continue
        # A column the detector could not match by name is DERIVED: the role
        # is built from the column's own words, so it answers with exactly one
        # term -- its own label -- and a priority of 60. When that derived key
        # happens to name a role the dictionary already holds, take the
        # dictionary's: same role, six synonyms instead of one, and a priority
        # that says how sure the vocabulary is rather than how sure the
        # fallback is.
        role = by_key.get(role.key, role)

        terms = date_role_terms(role)
        payload = {
            "role": "date",
            "display_name": role.label,
            "synonyms": ", ".join(terms),
        }
        # priority is the vocabulary's own confidence in the match: a named
        # role (invoice_date, 95) is a dictionary hit, while a role derived
        # from an unfamiliar column name carries 60 and reads as a guess a
        # human should look at. Passing it through rather than inventing a
        # number keeps one source of truth for how sure we are.
        draft = Draft(
            kind="date_role",
            entity=entity,
            column=column,
            payload=payload,
            before=dict(current or {}),
            reason=(f"{column} matches the {role.label} vocabulary; "
                    f"a question about {terms[0]} should filter on this column"),
            confidence=max(MIN_CONFIDENCE, min(100, int(role.priority))),
            evidence=(f"date_roles:{role.key}",),
        )
        if draft.is_change:
            drafts.append(draft)

    drafts.sort(key=lambda d: (-d.confidence, d.target))
    return drafts[:limit]


# ── 2 · Column vocabulary, from the value index and the question log ──────────

def _phrases(text: str, max_words: int = 3) -> list[str]:
    words = [w for w in _WORD.findall(str(text or "").casefold())]
    out: list[str] = []
    for size in range(1, max_words + 1):
        for i in range(len(words) - size + 1):
            run = words[i:i + size]
            if all(w in _STOPWORDS for w in run):
                continue
            out.append(" ".join(run))
    return out


def _value_forms(value: str) -> tuple[tuple[str, ...], ...]:
    """One stored value as the word runs a reader might type it as."""
    text = " ".join(str(value or "").split()).casefold()
    if not text:
        return ()
    forms = {tuple(_WORD.findall(text))}
    forms.add(tuple(text.split()))
    return tuple(f for f in forms if f)


# How many words after the value still count as naming its column.
#
# After, and not before. A value used the way readers use one is a modifier,
# and a modifier precedes its head noun: "shipped ORDERS", "pending INVOICES",
# "cancelled SHIPMENTS". What stands BEFORE the value is the measure the
# question is about -- "REVENUE for shipped orders" -- and counting it proposes
# "revenue" as vocabulary for a status column, which then resolves every
# question mentioning revenue onto it. A two-word window either side did
# exactly that; the fixture it was written against was too small for the
# threshold to be crossed, so it looked correct.
VALUE_WINDOW = 2

# How many questions must use a phrase beside a value before it is proposed.
# One reader is a coincidence; the same phrasing from a third is vocabulary.
MIN_PHRASE_OCCURRENCES = 3


def _neighbourhood(words: list[str], form: tuple[str, ...],
                   window: int = VALUE_WINDOW) -> list[str]:
    """Phrases within `window` words of an occurrence of `form`.

    Returns [] when the value is not in the question at all, which is what
    keeps a question that merely shares a topic from voting.
    """
    size = len(form)
    if not size:
        return []
    out: list[str] = []
    for i in range(len(words) - size + 1):
        if tuple(words[i:i + size]) != form:
            continue
        after = words[i + size:i + size + window]
        # The head noun starts immediately after the value, so only runs
        # anchored there are considered: "shipped orders last year" offers
        # "orders", not "last" and not "orders last". Every word must carry
        # meaning -- a run containing a stopword or a period word is a piece
        # of sentence, not a name for a column.
        for length in range(1, len(after) + 1):
            piece = after[:length]
            if any(w in _STOPWORDS or w in _PERIOD_WORDS for w in piece):
                break
            out.append(" ".join(piece))
    return out


def column_vocabulary_drafts(
    indexed_values: list[dict] | None,
    questions: list[str] | None,
    properties: list[dict] | None = None,
    *,
    limit: int = MAX_DRAFTS_PER_KIND,
) -> list[Draft]:
    """Propose the words readers use for a column they filter on by value.

    ``indexed_values`` are value-index rows: entity, column, and the distinct
    values sampled from it. ``questions`` are what people actually asked.

    The signal is a reader naming a *value* without naming its column --
    "revenue for shipped orders" mentions SHIPPED but never ORDER_STATUS. The
    resolver has to reach the column from the value, and it can only do that if
    the column carries the words the reader used instead. Those words are the
    ones standing next to the value.

    Next to it, specifically, and not merely somewhere in the same question.
    Counting every phrase in the question proposes whatever the question was
    *about* -- "revenue" -- as vocabulary for a status column, which would send
    every question mentioning revenue to it. The first version of this did
    exactly that and only looked correct because the fixture was too small for
    the threshold to be crossed.

    Evidence from two independent places, which is what makes this worth
    proposing at all: the value came from the warehouse and the phrasing came
    from a person. Either alone is a guess.
    """
    known = index_properties(properties)
    questions = [q for q in (questions or []) if q][:MAX_QUESTIONS]
    if not questions:
        return []

    tokenised = [_WORD.findall(str(q).casefold()) for q in questions]
    drafts: list[Draft] = []

    for entry in (indexed_values or []):
        entity = str(entry.get("entity") or entry.get("entity_name") or "").strip()
        column = str(entry.get("column") or entry.get("column_name") or "").strip()
        values = entry.get("values") or []
        if not entity or not column or not values:
            continue

        current = known.get(_property_key(entity, column))
        if _is_confirmed(current):
            continue

        # Words the matcher would already find. A phrase that is the column
        # name back again, in any of its forms, teaches the resolver nothing.
        already = set(_phrases(column.replace("_", " ")))
        already |= set(_phrases(str((current or {}).get("display_name") or "")))
        for term in str((current or {}).get("synonyms") or "").split(","):
            already |= set(_phrases(term))
        # ORDER_STATUS is not taught anything by "orders". Fold the trivial
        # plural so a morphological variant of the column's own name does not
        # read as a reader's word for it.
        already |= {w + "s" for w in list(already)}
        already |= {w[:-1] for w in list(already) if w.endswith("s")}

        prepared = [(str(value), forms)
                    for value, forms in ((v, _value_forms(v)) for v in values)
                    if forms]
        # Every value of this column, not just the one being scanned. A phrase
        # standing beside SHIPPED can contain PENDING -- "shipped pending
        # orders" -- and filtering against only the current value lets the
        # other one through as vocabulary for the column it is a value of.
        all_forms = {" ".join(form) for _, forms in prepared for form in forms}

        hits: dict[str, int] = {}
        matched_values: set[str] = set()
        # Questions outer, values inner. The count this builds is "how many
        # QUESTIONS used this phrase beside a value", and the threshold on it
        # means "more than one person". Counting per value instead lets one
        # question naming three values -- "shipped orders, pending orders,
        # cancelled orders" -- vote three times and clear a threshold that
        # exists precisely to stop it.
        for words in tokenised:
            seen_here: set[str] = set()
            for value, forms in prepared:
                for form in forms:
                    for phrase in _neighbourhood(words, form):
                        matched_values.add(value)
                        if phrase in already:
                            continue
                        # A value is vocabulary for the ROW, not the column.
                        # Substring in both directions against every value, so
                        # neither a value itself nor a phrase containing one
                        # survives: "shipped" as a synonym for a status column
                        # would resolve every question about a shipped
                        # anything onto it.
                        if any(phrase in f or f in phrase for f in all_forms):
                            continue
                        seen_here.add(phrase)
            for phrase in seen_here:
                hits[phrase] = hits.get(phrase, 0) + 1

        candidates = [(phrase, count) for phrase, count in hits.items()
                      if count >= MIN_PHRASE_OCCURRENCES]
        if not candidates or not matched_values:
            continue
        candidates.sort(key=lambda pair: (-pair[1], -len(pair[0].split()), pair[0]))
        terms = [phrase for phrase, _ in candidates[:6]]

        existing = str((current or {}).get("synonyms") or "").strip()
        merged = ", ".join(
            list(dict.fromkeys(
                [t.strip() for t in existing.split(",") if t.strip()] + terms))
        )
        draft = Draft(
            kind="column_synonyms",
            entity=entity,
            column=column,
            payload={"synonyms": merged},
            before=dict(current or {}),
            reason=(f"readers named values of {column} "
                    f"({', '.join(sorted(matched_values)[:3])}) without naming "
                    f"the column; these are the words they used beside them"),
            confidence=min(95, MIN_CONFIDENCE + 10 * candidates[0][1]),
            evidence=tuple(f"{phrase!r} beside a value in {count} questions"
                           for phrase, count in candidates[:6]),
        )
        if draft.is_change:
            drafts.append(draft)

    drafts.sort(key=lambda d: (-d.confidence, d.target))
    return drafts[:limit]


# ── 3 · Metric shapes ─────────────────────────────────────────────────────────

_NUMBER = re.compile(r"\b\d[\d,.]*\b")
_QUOTED = re.compile(r"['\"][^'\"]{1,40}['\"]")



def question_shape(question: str) -> str:
    """One question reduced to the shape it shares with its rephrasings.

    "revenue for March 2024" and "revenue for April 2025" are the same missing
    measure asked twice. Dates, numbers and quoted literals are the parts that
    vary between two askings of one question, so they are what has to go --
    what is left is the measure and the cut, which is the thing worth counting.
    """
    text = " ".join(str(question or "").split()).casefold()
    if not text:
        return ""
    text = _QUOTED.sub(" ", text)
    # One number pattern, not a year pattern beside it: \b\d[\d,.]*\b already
    # matches 2024, and a second regex that can only ever match a subset of
    # the first is a line a reader has to work out is unreachable.
    text = _NUMBER.sub(" ", text)
    words = [w for w in _WORD.findall(text)
             if w not in _STOPWORDS and w not in _PERIOD_WORDS]
    return " ".join(words)


def metric_shape_drafts(
    rows: list[dict] | None,
    *,
    min_occurrences: int = MIN_SHAPE_OCCURRENCES,
    limit: int = MAX_DRAFTS_PER_KIND,
) -> list[Draft]:
    """Question shapes asked repeatedly that never bound to a metric.

    ``rows`` are query_log rows: question, success, and whether a metric was
    resolved. A shape asked once is a question. The same shape asked eight
    times, never once resolving to a governed measure, is a measure the
    workspace does not have -- and the eight askings are the evidence for
    building it, which is exactly what an admin has no way to see today.

    A metric is not drafted with SQL. Proposing executable logic from question
    text alone would be guessing at a definition, and a wrong governed measure
    is worse than a missing one. What this drafts is the *case*: the shape, how
    often, and the phrasings -- so a human writes the definition knowing it is
    wanted.
    """
    counts: dict[str, int] = {}
    examples: dict[str, list[str]] = {}
    resolved: set[str] = set()

    for row in (rows or []):
        question = str(row.get("question") or "").strip()
        shape = question_shape(question)
        if not shape:
            continue
        if row.get("metric_id") or row.get("metric_name"):
            # This shape has a measure. Whatever else it did, it is not a gap.
            resolved.add(shape)
            continue
        counts[shape] = counts.get(shape, 0) + 1
        examples.setdefault(shape, [])
        if question not in examples[shape] and len(examples[shape]) < 4:
            examples[shape].append(question)

    drafts: list[Draft] = []
    for shape, count in counts.items():
        if shape in resolved or count < min_occurrences:
            continue
        drafts.append(Draft(
            kind="metric_shape",
            entity="",
            column="",
            payload={"shape": shape, "occurrences": count},
            before={},
            reason=(f"asked {count} times and never answered by a governed "
                    f"measure"),
            confidence=min(95, MIN_CONFIDENCE + 10 * count),
            evidence=tuple(examples.get(shape, ())),
        ))

    drafts.sort(key=lambda d: (-d.confidence, -int(d.payload.get("occurrences") or 0),
                               d.payload.get("shape") or ""))
    return drafts[:limit]
