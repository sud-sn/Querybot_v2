"""
Question sets: what a business asks of its warehouse, in English and French,
each question with its answer computed by a reference query written by hand
against the warehouse's own tables -- never taken from the SQL the product
writes.

Three warehouses, each spelled the way a different kind of client spells
theirs:

    distribution  tests/answer_harness.py: a distribution mart in one ERP
                  family's abbreviated uppercase (ITM_BAL_DLY_FCT, WHS_DMS)
    outfitters    tests/star_harness.py: a Kimball star in PascalCase
                  (FactInternetSales, DimProduct)
    ledger        tests/ledger_harness.py: a dbt-style mart in lowercase
                  snake_case whose facts carry dates, not date keys
                  (fct_purchase_order_lines, dim_supplier)

A question is answered right when every figure its reference query returns is
in the product's answer, and the answer has as many rows as the reference.

The model is a stand-in here, as it is in the harnesses: a question the product
cannot answer without the model writing its SQL is the release runner's to
check, against the real model. A question marked ``governed`` is one the
product answers without the model today; if it comes to need the model, that is
a regression, and it fails here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal


@dataclass(frozen=True)
class Question:
    id: str
    area: str
    en: str
    fr: str
    # DuckDB SQL over the warehouse's tables. It selects the figures the answer
    # must hold -- the measures, not the labels -- one row per row the answer
    # should have.
    reference: str
    # Answered without the model today: needing it is a regression.
    governed: bool = True
    # "answer", or "no data": a period the warehouse holds nothing for, where
    # the right answer is to say so rather than to show a zero.
    kind: str = "answer"
    # The option to pick when the product asks which one was meant.
    choose: str | None = None
    # "Which one had the lowest ...": the answer may rank them all, the one
    # asked for first; only its first row is held to the reference.
    first_row: bool = False
    # Where the product falls short today, by language: "model" when it needs
    # the model to write the SQL, which the release runner checks against the
    # real one, or why it answers wrong or not at all. The day one of these
    # passes, the test says so, and the note goes.
    today: dict = field(default_factory=dict)


def _number(value) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float, Decimal)):
        return float(value)
    # The card carries its cells as text, formatted for the reader.
    text = str(value).strip().replace(",", "").replace("\u202f", "").replace("\xa0", "") if isinstance(
        value, str) else ""
    text = text.strip("$€ ")
    if re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        return float(text)
    return None


def shown_rows(answer: dict) -> list | None:
    """The rows of the answer card the reader was shown, or None when no card
    was: what the reader saw, not the last statement the warehouse ran -- an
    empty answer is followed by the product's own row counts."""
    for kind, reply in reversed(answer.get("replies") or []):
        if kind == "answer" and isinstance(reply, dict):
            return list((reply.get("data") or {}).get("rows") or [])
    return None


def figures(rows: list) -> list[list[float]]:
    """The numbers of each row, in order: dict rows or tuples."""
    out = []
    for row in rows:
        values = row.values() if isinstance(row, dict) else row
        out.append([n for n in (_number(v) for v in values) if n is not None])
    return out


def answer_rows(answer: dict) -> list | None:
    """The rows of the answer the reader was shown, at full precision: the
    statement the card was built from -- its columns and its row count -- not
    the card's rounded text. None when no card was shown."""
    card = shown_rows(answer)
    if not card:
        return card
    headers = {str(name).upper() for name in card[0]}
    for statement in reversed(answer.get("executed") or []):
        rows = statement.get("rows") or []
        if len(rows) == len(card) and headers <= {str(name).upper() for name in rows[0]}:
            return rows
    return card


def reference_rows(warehouse, sql: str) -> list[tuple]:
    """The reference query's rows, run on the warehouse itself."""
    return warehouse.con.execute(sql).fetchall()


def _same(left: float, right: float) -> bool:
    return abs(left - right) <= max(0.011, abs(right) * 1e-6)


def verdict(question: Question, answer: dict, expected: list[tuple], marker: str) -> tuple[str, str]:
    """PASS, WRONG, NO ANSWER or NEEDS THE MODEL, and why. ``marker`` is what
    the stand-in model writes as SQL."""
    rows = answer_rows(answer) or []
    if answer.get("model_wrote_sql") and (not rows or marker in str(rows)):
        return "NEEDS THE MODEL", "the SQL writer was asked"
    if question.kind == "no data":
        if not any(figures(rows)):
            return "PASS", "no figure, as the warehouse holds none"
        return "WRONG", f"figures where the warehouse holds none: {rows[:3]}"
    if not rows:
        replies = [str(reply)[:160] for _kind, reply in answer.get("replies") or []]
        return "NO ANSWER", f"no rows; replies: {replies[:2]}"
    wanted = [n for figs in figures(expected) for n in figs]
    pool = [n for figs in figures(rows[:1] if question.first_row else rows) for n in figs]
    missing = []
    for number in wanted:
        match = next((i for i, have in enumerate(pool) if _same(have, number)), None)
        if match is None:
            missing.append(number)
        else:
            pool.pop(match)
    if missing:
        return "WRONG", f"missing {missing[:6]} of {wanted[:12]}; the answer held {figures(rows)[:6]}"
    if len(rows) != len(expected) and not (question.first_row and len(expected) == 1):
        return "WRONG", f"{len(rows)} rows where the reference has {len(expected)}: {rows[:6]}"
    return "PASS", ""
