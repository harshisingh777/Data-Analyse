"""A sandboxed, read-only SQL workbench over a single DataFrame (SQLite in memory)."""

from __future__ import annotations

import sqlite3
import threading
import time

import pandas as pd

from .config import Settings
from .utils import DataLensError

TABLE_NAME = "dataset"

_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    getattr(sqlite3, "SQLITE_RECURSIVE", 33),
}


def _authorizer(action, arg1, arg2, db_name, trigger):
    """Deny every statement that is not a pure read (no INSERT, DROP, ATTACH, PRAGMA ...)."""
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


class SqlWorkbench:
    def __init__(self, df: pd.DataFrame, row_limit: int | None = None, timeout_seconds: float | None = None):
        self.row_limit = row_limit or Settings.SQL_ROW_LIMIT
        self.timeout = timeout_seconds or Settings.SQL_TIMEOUT_SECONDS
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._load(df)
        # Read the schema before locking the connection down; PRAGMA is denied afterwards and
        # Python < 3.11 cannot remove an authorizer once set.
        self._schema = [
            {"name": r[1], "type": r[2] or "TEXT"}
            for r in self._conn.execute(f'PRAGMA table_info("{TABLE_NAME}")').fetchall()
        ]
        self._conn.set_authorizer(_authorizer)

    def _load(self, df: pd.DataFrame) -> None:
        frame = df.copy()
        for col in frame.columns:
            series = frame[col]
            if pd.api.types.is_datetime64_any_dtype(series):
                frame[col] = series.dt.strftime("%Y-%m-%d %H:%M:%S").where(series.notna(), None)
            elif pd.api.types.is_bool_dtype(series):
                frame[col] = series.astype("Int64")
            elif series.dtype == object:
                frame[col] = series.map(lambda v: v if v is None or isinstance(v, (str, int, float)) else str(v))
        frame.to_sql(TABLE_NAME, self._conn, index=False)

    def schema(self) -> list[dict]:
        return list(self._schema)

    def execute(self, query: str) -> dict:
        query = (query or "").strip().rstrip(";").strip()
        if not query:
            raise DataLensError("Enter a SQL query to run.")
        if len(query) > 10_000:
            raise DataLensError("Query is too long.")

        deadline = time.perf_counter() + self.timeout

        def watchdog() -> int:
            return 1 if time.perf_counter() > deadline else 0

        with self._lock:
            self._conn.set_progress_handler(watchdog, 10_000)
            start = time.perf_counter()
            try:
                cursor = self._conn.execute(query)
                if cursor.description is None:
                    raise DataLensError("Only SELECT queries are supported.")
                columns = [d[0] for d in cursor.description]
                rows = cursor.fetchmany(self.row_limit + 1)
            except sqlite3.DatabaseError as exc:
                message = str(exc)
                if "interrupted" in message:
                    message = f"Query exceeded the {self.timeout:g}s time limit."
                elif "not authorized" in message or "prohibited" in message:
                    message = "Only read-only SELECT queries are allowed."
                raise DataLensError(f"SQL error: {message}") from exc
            except sqlite3.Warning as exc:  # e.g. multiple statements
                raise DataLensError(f"SQL error: {exc}") from exc
            finally:
                self._conn.set_progress_handler(None, 0)
            elapsed_ms = (time.perf_counter() - start) * 1000

        truncated = len(rows) > self.row_limit
        rows = rows[: self.row_limit]
        return {
            "columns": columns,
            "rows": [list(r) for r in rows],
            "row_count": len(rows),
            "truncated": truncated,
            "elapsed_ms": round(elapsed_ms, 2),
        }

    def close(self) -> None:
        self._conn.close()


def example_queries(profile: dict) -> list[dict]:
    """Starter queries tailored to the dataset's own columns."""
    cols = profile["columns"]
    numeric = [c["name"] for c in cols if c["semantic_type"] == "numeric"]
    categorical = [c["name"] for c in cols if c["semantic_type"] in ("categorical", "boolean")]
    dates = [c["name"] for c in cols if c["semantic_type"] == "datetime"]
    q = [{"title": "Preview rows", "sql": f"SELECT *\nFROM {TABLE_NAME}\nLIMIT 20"}]
    if categorical:
        c = categorical[0]
        agg = f',\n       ROUND(AVG("{numeric[0]}"), 2) AS avg_{_slug(numeric[0])}' if numeric else ""
        q.append(
            {
                "title": f"Breakdown by {c}",
                "sql": f'SELECT "{c}",\n       COUNT(*) AS rows{agg}\nFROM {TABLE_NAME}\nGROUP BY "{c}"\nORDER BY rows DESC',
            }
        )
    if numeric:
        n = numeric[0]
        q.append(
            {
                "title": f"Summary of {n}",
                "sql": (
                    f'SELECT COUNT("{n}") AS n,\n       MIN("{n}") AS min,\n       ROUND(AVG("{n}"), 2) AS mean,\n'
                    f'       MAX("{n}") AS max\nFROM {TABLE_NAME}'
                ),
            }
        )
    if dates:
        d = dates[0]
        q.append(
            {
                "title": f"Monthly trend by {d}",
                "sql": f"SELECT strftime('%Y-%m', \"{d}\") AS month,\n       COUNT(*) AS rows\nFROM {TABLE_NAME}\n"
                "GROUP BY month\nORDER BY month",
            }
        )
    if len(categorical) >= 2 and numeric:
        a, b, n = categorical[0], categorical[1], numeric[0]
        q.append(
            {
                "title": "Window function: rank within group",
                "sql": (
                    f'SELECT "{a}", "{b}", avg_value,\n'
                    f'       RANK() OVER (PARTITION BY "{a}" ORDER BY avg_value DESC) AS rank_in_group\n'
                    f'FROM (\n  SELECT "{a}", "{b}", ROUND(AVG("{n}"), 2) AS avg_value\n'
                    f'  FROM {TABLE_NAME}\n  GROUP BY "{a}", "{b}"\n)\nORDER BY "{a}", rank_in_group'
                ),
            }
        )
    return q


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_") or "value"
