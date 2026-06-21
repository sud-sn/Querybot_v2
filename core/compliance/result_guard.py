from __future__ import annotations

import hashlib
from typing import Any

from core.compliance.models import PolicyDecision


def _mask(value: Any, strategy: str, seed: str) -> Any:
    if value is None:
        return None
    strategy = (strategy or "redact").lower()
    text = str(value)
    if strategy == "partial":
        return ("*" * max(4, len(text) - 4)) + text[-4:]
    if strategy in {"hash", "tokenize"}:
        digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()[:16]
        return f"TKN-{digest}" if strategy == "tokenize" else digest
    if strategy == "null":
        return None
    return "[REDACTED]"


def protect_rows(
    rows: list[dict],
    decision: PolicyDecision,
    lineage: dict[str, list[str]],
    *,
    account_id: str,
) -> list[dict]:
    if not rows or not decision.masking:
        return [dict(row) for row in rows]
    protected = []
    for row in rows:
        item = dict(row)
        for output_column, sources in lineage.items():
            strategies = [
                decision.masking[source]
                for source in sources
                if source in decision.masking
            ]
            if strategies and output_column in item:
                item[output_column] = _mask(item[output_column], strategies[0], account_id)
        protected.append(item)
    return protected
