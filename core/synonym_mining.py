"""The words users actually type, learned from the questions that failed.

A metric carries the vocabulary somebody thought of when they wrote it down.
Readers use a different one, and the gap only shows up as a failure: a question
that should have bound to a metric did not, the reader rephrased, and the
second attempt worked. Nobody records the first attempt as evidence, so the
same word fails again next week for the next person.

The signal is that pair. One reader, one short window, a question that failed
followed by one that succeeded and bound to a metric. Whatever the reader said
the first time and stopped saying the second is a word that means that metric
to them and to nobody in the model.

Three properties keep this from being noise dressed as insight:

**Only the difference counts.** A phrase is a candidate only if it appears in
the failed question, is absent from the successful one, and is not already
somewhere the matcher would have found it — the metric's name, its synonyms,
or the table and column names it is built from.

**One occurrence is a typo.** Nothing is proposed below ``MIN_OCCURRENCES``,
because a single reader saying "turnover" once is a coincidence and three
readers saying it is a vocabulary gap. The count is carried on the proposal so
an admin sees the evidence rather than a verdict.

**A proposal is never applied.** This produces a queue for one-click review.
Writing a synonym straight onto a metric would let anyone who mistypes a
question rename a governed measure, which is the shape of a supply-chain
attack rather than a feature.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger("querybot.synonym_mining")

# How long after a failure a rephrase still counts as the same attempt. Long
# enough for someone to read an error and try again, short enough that the
# next question of the morning is not attributed to it.
REPHRASE_WINDOW_MINUTES = 10

# Below this a phrase is a coincidence rather than a vocabulary gap.
MIN_OCCURRENCES = 2

# The longest phrase considered. Beyond three words a "synonym" is a sentence,
# and the matcher looks for whole phrases, so a long one would never fire.
MAX_PHRASE_WORDS = 3

# How many pairs are examined. Mining is an admin-time report, not a query-time
# cost, but an unbounded scan of a busy account's log is still a bad neighbour.
MAX_PAIRS = 400

_WORD = re.compile(r"[a-z0-9]+")

# Words that carry no metric meaning. Deliberately small: an aggressive list
# would drop the domain words that are the entire point ("marge", "uplift"),
# and a phrase that survives is still gated on MIN_OCCURRENCES.
_STOPWORDS = frozenset("""
a an the and or of for in on at by to from with without is are was were be been
what which who whom whose how many much show me give tell list find get
this that these those it its as per each every all any some
last first next previous current total sum count average avg min max top bottom
please can could would do does did
le la les un une des du de et ou pour dans sur par avec sans est sont
quel quelle quels quelles combien comment montre donne liste tous toutes
dernier derniere premier premiere prochain courant chaque tout
""".split())


@dataclass(frozen=True)
class Rephrase:
    """A question that failed and the one the reader asked instead."""

    failed: str
    succeeded: str
    user_key: str = ""
    at: str = ""


@dataclass(frozen=True)
class SynonymProposal:
    """A word readers use for a metric that the metric does not know."""

    metric_id: int
    metric_name: str
    phrase: str
    occurrences: int = 0
    examples: tuple[str, ...] = ()
    detail: str = ""

    @property
    def strong(self) -> bool:
        return self.occurrences >= MIN_OCCURRENCES + 1


def _tokens(text: str) -> list[str]:
    return _WORD.findall(str(text or "").casefold())


def _phrases(text: str) -> set[str]:
    """Every 1..MAX_PHRASE_WORDS run of meaningful words."""
    words = [w for w in _tokens(text) if w not in _STOPWORDS and len(w) > 2]
    out: set[str] = set()
    for size in range(1, MAX_PHRASE_WORDS + 1):
        for i in range(len(words) - size + 1):
            out.add(" ".join(words[i:i + size]))
    return out


def known_vocabulary(metric: dict) -> set[str]:
    """Everything the matcher would already have found this metric by.

    Name, synonyms, and the identifiers its SQL is built from — a reader who
    types a column name is not proposing a synonym, they are naming the thing.
    """
    parts = [str(metric.get("name") or ""), str(metric.get("synonyms") or "")]
    sql = str(metric.get("sql_template") or "")
    # Every word-like run in the SQL: table and column identifiers, and the
    # literals inside a WHERE clause too. Both are excluded from proposals,
    # for different reasons. A reader who types a column name is naming the
    # thing rather than renaming it. A reader who types a filter VALUE —
    # "bretagne numbers" for a metric scoped to that region — is naming a
    # slice, and accepting it as a synonym would bind a governed measure to
    # one region's name for everybody.
    parts.extend(re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", sql))
    vocabulary: set[str] = set()
    for part in parts:
        for chunk in re.split(r"[,\n;|]+", part):
            chunk = chunk.replace("_", " ").replace(".", " ").strip()
            if chunk:
                vocabulary |= _phrases(chunk) | {chunk.casefold()}
    return vocabulary


def rephrase_pairs(rows: list[dict], *, window_minutes: int = REPHRASE_WINDOW_MINUTES,
                   limit: int = MAX_PAIRS) -> list[Rephrase]:
    """Failure-then-success pairs from one account's query log, oldest first.

    ``rows`` are query_log rows ordered by time: question, success,
    created_at, and a user key. Pairs are strictly adjacent per user — a
    failure followed by that same reader's next question — because a failure
    two questions back was abandoned, not rephrased.
    """
    from datetime import datetime, timedelta

    def _at(value):
        try:
            return datetime.fromisoformat(str(value or "").replace("Z", ""))
        except (TypeError, ValueError):
            return None

    latest_failure: dict[str, dict] = {}
    pairs: list[Rephrase] = []
    for row in rows:
        user_key = str(row.get("user_key") or row.get("portal_user_id") or "")
        question = str(row.get("question") or "").strip()
        if not question:
            continue
        if not row.get("success"):
            latest_failure[user_key] = row
            continue

        previous = latest_failure.pop(user_key, None)
        if not previous:
            continue
        before, after = _at(previous.get("created_at")), _at(row.get("created_at"))
        if before and after and after - before > timedelta(minutes=window_minutes):
            continue
        failed = str(previous.get("question") or "").strip()
        if not failed or failed.casefold() == question.casefold():
            continue
        pairs.append(Rephrase(failed=failed, succeeded=question,
                              user_key=user_key, at=str(row.get("created_at") or "")))
        if len(pairs) >= limit:
            break
    return pairs


def candidates_from(pair: Rephrase, metric: dict) -> set[str]:
    """Phrases the reader dropped when they rephrased into this metric.

    The difference, not the failed question: a word that survives into the
    successful question is one the metric already understands, so proposing it
    would add a synonym that changes nothing.
    """
    dropped = _phrases(pair.failed) - _phrases(pair.succeeded)
    known = known_vocabulary(metric)
    return {phrase for phrase in dropped if phrase not in known}


def propose(pairs: list[Rephrase], match_metric) -> list[SynonymProposal]:
    """Rank the vocabulary gaps these pairs reveal.

    ``match_metric(question) -> metric | None`` is the store's own matcher, so
    a proposal is only ever made for a metric the product would really have
    bound the successful question to.
    """
    seen: dict[tuple[int, str], dict] = {}
    for pair in pairs:
        try:
            metric = match_metric(pair.succeeded)
        except Exception as exc:  # noqa: BLE001
            log.warning("Metric match failed while mining synonyms: %s", exc)
            continue
        if not metric or not metric.get("id"):
            continue
        for phrase in candidates_from(pair, metric):
            key = (int(metric["id"]), phrase)
            entry = seen.setdefault(key, {
                "metric": metric, "phrase": phrase, "count": 0, "examples": [],
            })
            entry["count"] += 1
            if len(entry["examples"]) < 3:
                entry["examples"].append(pair.failed)

    proposals = [
        SynonymProposal(
            metric_id=int(entry["metric"]["id"]),
            metric_name=str(entry["metric"].get("name") or ""),
            phrase=entry["phrase"],
            occurrences=entry["count"],
            examples=tuple(entry["examples"]),
            detail=(f'{entry["count"]} readers asked for '
                    f'"{entry["metric"].get("name") or "this metric"}" '
                    f'as "{entry["phrase"]}" and got nothing.'),
        )
        for entry in seen.values()
        if entry["count"] >= MIN_OCCURRENCES
    ]
    # Most evidence first, then the longer phrase: "net margin" is a better
    # synonym than "margin", and a tie on count should not be alphabetical.
    proposals.sort(key=lambda p: (-p.occurrences, -len(p.phrase.split()), p.phrase))
    return proposals


def mine(account_id: str, *, window_minutes: int = REPHRASE_WINDOW_MINUTES,
         limit: int = MAX_PAIRS) -> list[SynonymProposal]:
    """The whole pass for one account. Never raises: this is a report.

    Reads the query log directly rather than through a store helper, because
    nothing else needs failure-then-success adjacency and a helper for one
    caller is a helper that drifts.
    """
    import store

    try:
        with store.get_db() as conn:
            rows = [dict(r) for r in conn.execute(
                """
                SELECT question, success, created_at,
                       COALESCE(portal_user_id, zoom_user_id) AS user_key
                FROM query_log
                WHERE account_id = ? AND question IS NOT NULL AND question != ''
                ORDER BY created_at ASC, id ASC
                LIMIT ?
                """,
                (account_id, int(limit) * 4),
            ).fetchall()]
    except Exception as exc:  # noqa: BLE001
        log.warning("Synonym mining could not read the query log for %s: %s",
                    account_id, exc)
        return []

    pairs = rephrase_pairs(rows, window_minutes=window_minutes, limit=limit)
    if not pairs:
        return []
    return propose(pairs, lambda question: store.match_metric(account_id, question))
