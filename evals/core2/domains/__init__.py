"""The synthetic business domains, each a module with ``build() -> Domain``."""

from __future__ import annotations

import importlib

from evals.core2.framework import Domain

DOMAINS = ("retail", "inventory", "finance", "purchasing", "hr", "subscriptions")


def available() -> list[str]:
    """The domains whose module exists (all of them once the set is complete)."""
    out = []
    for name in DOMAINS:
        try:
            importlib.import_module(f"evals.core2.domains.{name}")
        except ModuleNotFoundError as exc:
            if exc.name != f"evals.core2.domains.{name}":
                raise
            continue
        out.append(name)
    return out


def build(name: str, seed: int = 7) -> Domain:
    return importlib.import_module(f"evals.core2.domains.{name}").build(seed)
