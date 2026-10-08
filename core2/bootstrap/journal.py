"""What Learn is doing, and what it had to leave out, as it happens.

Learn reads a real warehouse, and one query in hundreds can be refused: a table
the service account may not read, a column of a type the warehouse will not
compare, a view it will not sample, two tables whose text keys are in different
collations. Each such query is left out and said -- in the service log, in the
progress the learned page shows while Learn runs, and in the model's notes --
and Learn goes on with everything else. Only a warehouse that no longer answers
at all stops it: then nothing more could be learned, and saying so is the result.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any, TypeVar

from core2.warehouse.runner import QueryResult, Warehouse

log = logging.getLogger("querybot.core2")

T = TypeVar("T")


def reason(error: BaseException) -> str:
    """The warehouse's own words for a refusal, on one line."""
    return " ".join(str(error).split())[:240] or type(error).__name__


class Journal:
    def __init__(self, write: Callable[[str], None] | None = None):
        self._write = write
        self._lock = threading.Lock()
        self.left_out: list[tuple[str, str]] = []

    def step(self, text: str) -> None:
        log.info("core2 learn: %s", text)
        if self._write is None:
            return
        try:
            self._write(text)
        except Exception as exc:  # noqa: BLE001 - the progress line is a help; Learn goes on without it
            log.warning("core2 learn: progress line not saved: %s", exc)

    def leave_out(self, what: str, error: BaseException) -> None:
        why = reason(error)
        with self._lock:
            self.left_out.append((what, why))
        log.warning("core2 learn: left out %s: %s", what, why)
        self.step(f"Left out {what}: the database refused it ({why})")


class Watched:
    """A warehouse that carries the journal of the Learn reading it."""

    def __init__(self, warehouse: Warehouse, journal: Journal):
        self.warehouse = warehouse
        self.journal = journal

    @property
    def dialect(self) -> str:
        return self.warehouse.dialect

    @property
    def db_type(self) -> str:
        return self.warehouse.db_type

    def query(self, sql: str, *, max_rows: int | None = None) -> QueryResult:
        return self.warehouse.query(sql, max_rows=max_rows)


def journal_of(warehouse: Any) -> Journal:
    journal = getattr(warehouse, "journal", None)
    return journal if isinstance(journal, Journal) else Journal()


def answers(warehouse: Warehouse) -> bool:
    """Does the warehouse still answer at all? A refusal of one query is then about that query."""
    try:
        warehouse.query("SELECT 1 FROM DUAL" if warehouse.dialect == "oracle" else "SELECT 1")
        return True
    except Exception:  # noqa: BLE001 - not answering is the answer
        return False


def attempt(warehouse: Warehouse, what: str, run: Callable[[], T], default: T) -> T:
    """``run()``; when the warehouse refuses it but still answers, ``default``, with ``what`` left out and said."""
    try:
        return run()
    except Exception as error:
        if not answers(warehouse):
            raise
        journal_of(warehouse).leave_out(what, error)
        return default
