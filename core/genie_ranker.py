"""
core/genie_ranker.py

Behavioral signal scoring engine for the Genie suggestion panel.

Scores each suggestion using a blend of:
  • Behavioral signals (CTR, execution rate, success rate, dismissal penalty)
    sourced from the recommendation_event table via
    store.learning_store.get_suggestion_stats
  • Source quality boost (governed > learned > static)
  • Cold-start guard: <10 impressions → source boost only (no behavioral noise)
  • Confidence blending: more impressions → trust behavioral score more

Score weights (warm-start path, impressions ≥ threshold):
  CTR:           30%  (clicked / displayed)
  Execution:     40%  (executed / displayed)
  Success:       25%  (successful / displayed)
  Dismissal:     -5%  (dismissed / displayed)

Source boosts (applied as cold-start score or blended with behavioral):
  governed:      +0.10  (admin-approved learning candidate)
  learned:       +0.05  (auto-harvested from query log)
  static:         0.00  (metric registry / Stage-2 cache default)

Confidence blending:
  confidence = min(impressions / 100, 1.0)
  score = confidence × behavioral + (1 − confidence) × source_boost

Cold-start (impressions < impression_threshold):
  score = source_boost  (no behavioural data yet — avoid noise)

Usage:
  from core.genie_ranker import rank_suggestions
  ranked = rank_suggestions(account_id, suggestions, source_map={"What were...": "governed"})
"""

from __future__ import annotations

import logging

log = logging.getLogger("querybot.genie_ranker")

# ── Constants ─────────────────────────────────────────────────────────────────

_IMPRESSION_THRESHOLD = 10    # cold-start guard: fewer impressions than this
_CONFIDENCE_SCALE     = 100   # impressions needed to reach full confidence (1.0)

# Source tier → quality boost added to score.
# Legacy source strings from governed_store payload are mapped to their tier.
_SOURCE_BOOST: dict[str, float] = {
    "governed":         0.10,
    "admin_correction": 0.10,  # admin corrected SQL — governed tier
    "pre_governed":     0.10,
    "learned":          0.05,
    "auto":             0.05,  # auto-harvested == learned tier
    "static":           0.00,
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_source_boost(source: str) -> float:
    """Return the quality boost for a source tier. Unknown tiers → 0.0."""
    return _SOURCE_BOOST.get(source, 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def score_suggestion(
    stats: dict[str, int],
    source: str = "static",
    impression_threshold: int = _IMPRESSION_THRESHOLD,
) -> float:
    """
    Compute a ranking score for one suggestion given its behavioural stats.

    Parameters
    ----------
    stats : dict returned by get_suggestion_stats().
            Keys: "displayed", "clicked", "executed", "successful", "dismissed".
            Missing keys default to 0.
    source : source tier of the suggestion.
             One of "governed", "learned", "static", or legacy names such as
             "auto", "admin_correction", "pre_governed".
    impression_threshold : minimum displayed-impressions before behavioural
                           signals are incorporated into the score.

    Returns
    -------
    float
        Ranking score.  Higher is better.
        Range: [−0.05, ~1.10] (dismissal can push below 0; source boost can
        exceed 1.0 only when behavioral=1.0 and boost>0; in practice stays
        well below 1.1 for real data).

    Algorithm
    ---------
    impressions = stats.get("displayed", 0)

    Cold-start  (impressions < impression_threshold):
        score = source_boost

    Warm-start:
        ctr          = clicked    / impressions
        exec_rate    = executed   / impressions
        success_rate = successful / impressions
        dismiss_rate = dismissed  / impressions

        behavioral   = 0.30 × ctr
                     + 0.40 × exec_rate
                     + 0.25 × success_rate
                     − 0.05 × dismiss_rate

        confidence   = min(impressions / 100, 1.0)
        score        = confidence × behavioral + (1 − confidence) × source_boost
    """
    boost       = _get_source_boost(source)
    impressions = int(stats.get("displayed", 0))

    # ── Cold-start ────────────────────────────────────────────────────────────
    if impressions < impression_threshold:
        return boost

    # ── Warm-start ────────────────────────────────────────────────────────────
    clicks     = int(stats.get("clicked",    0))
    executed   = int(stats.get("executed",   0))
    successful = int(stats.get("successful", 0))
    dismissed  = int(stats.get("dismissed",  0))

    ctr          = clicks    / impressions
    exec_rate    = executed  / impressions
    success_rate = successful / impressions
    dismiss_rate = dismissed / impressions

    behavioral = (
        0.30 * ctr
        + 0.40 * exec_rate
        + 0.25 * success_rate
        - 0.05 * dismiss_rate
    )

    confidence = min(impressions / _CONFIDENCE_SCALE, 1.0)
    return confidence * behavioral + (1.0 - confidence) * boost


def rank_suggestions(
    account_id: str,
    suggestions: list[dict],
    *,
    source_map: dict[str, str] | None = None,
) -> list[dict]:
    """
    Return suggestions sorted by behavioural score (descending).

    Fetches event stats from the recommendation_event table and scores each
    suggestion using :func:`score_suggestion`.  Fails gracefully — a DB error
    on any individual suggestion is logged and that suggestion is scored as 0.0
    (cold-start static tier).  A complete failure in stats lookup is similarly
    absorbed so the caller always receives a list.

    Parameters
    ----------
    account_id : Tenant key used for event lookups (tenant-isolated).
    suggestions : list of dicts with at least a "question" key.
                  Typically ``{"question": str, "fqn": str}`` from
                  :func:`core.suggestions.get_suggestions`.
    source_map : optional mapping of question text → source tier string.
                 When provided, overrides any "source" field already on the
                 suggestion dict.  Useful when the caller knows which
                 suggestions came from the governed collection.

    Returns
    -------
    list[dict]
        Same dicts, sorted by score descending, with a ``"_score"`` key
        injected for observability (stripped by the caller if desired).
    """
    from store.learning_store import get_suggestion_stats

    if not suggestions:
        return []

    scored: list[tuple[float, dict]] = []
    for sug in suggestions:
        question = sug.get("question", "")

        # Resolve source: source_map wins, then existing field, then "static"
        source = (
            (source_map or {}).get(question)
            or sug.get("source", "static")
        )

        try:
            stats = get_suggestion_stats(account_id, question)
        except Exception as exc:
            log.debug(
                "rank_suggestions: stats lookup failed for %r — %s",
                question[:40], exc,
            )
            stats = {}

        s = score_suggestion(stats, source=source)
        sug_copy = dict(sug)
        sug_copy["_score"] = round(s, 6)
        scored.append((s, sug_copy))

    scored.sort(key=lambda t: t[0], reverse=True)
    return [sug for _, sug in scored]
