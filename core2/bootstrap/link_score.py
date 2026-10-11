"""How sure Learn is of a link: the chance it is right, from what Learn found about it.

The share of rows that find a match says how well a link holds, not whether it is one: a column of small numbers
finds a match in any table numbered 1..N. So each thing Learn found moves the odds, by a weight fitted on every
link Learn tested on the synthetic warehouses, in every naming style, against their truth
(evals/core2/link_calibration.py); the public schemas, kept out of the fit, check that a link scored 90% is right
about 9 times in 10. What the AI read into a column (core2/bootstrap/second_opinion.py) was never in a fit: it adds
a fixed, modest amount, and its links go to the admin to confirm.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core2.bootstrap.joins import JoinFinding

FEATURES = ("declared", "name", "same_name", "no_name", "match", "missed", "coverage", "alternate", "history",
            "self")

# Fitted by evals/core2/link_calibration.py on 420 tested links; rerun it when Learn's tests change. The
# benchmark's warehouses keep no query history, so "history" fitted to nothing: HISTORY stands in for it.
WEIGHTS = {"bias": 0.932, "declared": 2.03, "name": 0.235, "same_name": 0.491, "no_name": -0.242, "match": 0.622,
           "missed": -1.795, "coverage": 0.511, "alternate": -0.733, "history": 0.0, "self": 0.38}
HISTORY = 1.0                # log-odds the warehouse's own queries add; not fitted (no history to fit it on)
AI_READ = 1.0                # log-odds the AI's reading adds; not fitted (no truth to fit it on)


def features(f: "JoinFinding") -> dict[str, float]:
    named = f.name_score > 0
    return {
        "declared": float(f.declared),
        "name": f.name_score,
        "same_name": float(f.name_score >= 1.0),
        "no_name": float(not f.declared and not named),
        "match": f.match_rate,
        # Rows finding nothing, on a scale where the first few count most: 1% missed is far from 0%.
        "missed": math.log10(1.0 + 1000.0 * (1.0 - f.match_rate)) if f.non_null else 3.0,
        "coverage": f.coverage,
        "alternate": float(f.to_alternate),
        "history": float(f.history > 0),
        "self": float(f.from_table == f.to_table),
    }


def link_score(f: "JoinFinding", weights: dict[str, float] | None = None) -> float:
    """The chance the link is right, 0..1."""
    w = weights or WEIGHTS
    x = features(f)
    z = w["bias"] + sum(w[k] * x[k] for k in FEATURES)
    if any(e.kind == "ai_opinion" for e in f.evidence):
        z += AI_READ
    if f.history and not w["history"]:
        z += HISTORY
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
