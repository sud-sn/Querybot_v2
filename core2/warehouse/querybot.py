"""Running core2's bootstrap SQL on a workspace's configured warehouse.

One connection per worker thread for the length of a build (a Snowflake key-pair
sign-in costs a second or two, and a build sends a few hundred small queries),
opened with today's connection helpers so sign-in, query tagging and retries are
exactly what the rest of QueryBot uses. Every statement gets a timeout, and every
statement is checked read-only before it is sent.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from core2.warehouse.dialect import for_db_type
from core2.warehouse.runner import QueryResult, assert_read_only


@dataclass
class QueryBotWarehouse:
    db_type: str                     # snowflake | azure_sql | oracle
    credentials: dict[str, Any]
    timeout_seconds: int = 120
    # Azure SQL answers 40613 while a paused database resumes (a minute or so) and during a failover:
    # four tries over about a minute, as today's pipeline makes. One try failed every question asked
    # while the database was waking up.
    connect_retries: int = 4
    dialect: str = ""
    _local: threading.local = field(default_factory=threading.local, repr=False)
    _all: list[Any] = field(default_factory=list, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self.dialect = for_db_type(self.db_type)

    def _connection(self) -> Any:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            return conn
        from core.schema import _az_connect, _ora_connect, _sf_connect

        if self.db_type == "snowflake":
            conn = _sf_connect(self.credentials, max_retries=1)
            cur = conn.cursor()
            try:
                cur.execute(f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {int(self.timeout_seconds)}")
            finally:
                cur.close()
        elif self.db_type == "azure_sql":
            conn = _az_connect({**self.credentials, "login_timeout": 20}, max_retries=self.connect_retries)
            conn.timeout = int(self.timeout_seconds)
        elif self.db_type == "oracle":
            conn = _ora_connect(self.credentials, max_retries=1)
            conn.call_timeout = int(self.timeout_seconds) * 1000
        else:
            raise ValueError(f"core2 cannot profile a {self.db_type!r} database")
        self._local.conn = conn
        with self._lock:
            self._all.append(conn)
        return conn

    def query(self, sql: str, *, max_rows: int | None = None) -> QueryResult:
        assert_read_only(sql, self.dialect)
        start = time.perf_counter()
        cur = self._connection().cursor()
        try:
            cur.execute(sql)
            columns = [d[0] for d in (cur.description or [])]
            if max_rows is None:
                rows = cur.fetchall()
                truncated = False
            else:
                rows = cur.fetchmany(max_rows + 1)
                truncated = len(rows) > max_rows
                rows = rows[:max_rows]
        finally:
            try:
                cur.close()
            except Exception:  # noqa: BLE001 - a cursor that will not close must not hide the result
                pass
        return QueryResult(columns, [tuple(r) for r in rows], truncated, (time.perf_counter() - start) * 1000)

    def close(self) -> None:
        with self._lock:
            connections, self._all = self._all, []
        for conn in connections:
            try:
                conn.close()
            except Exception:  # noqa: BLE001 - closing is best effort
                pass

    def __enter__(self) -> QueryBotWarehouse:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
