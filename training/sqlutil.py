"""Shared SQLite execution + result-comparison helpers.

Used by prepare_data.py (to drop examples whose gold SQL doesn't actually run)
and by evaluate.py (to score predictions by *execution accuracy*, not string match).
"""
from __future__ import annotations

import re
import sqlite3
from typing import Any, Iterable

# Gretel contexts are CREATE TABLE + INSERT scripts. A few use dialect-specific
# syntax that SQLite rejects; those examples get filtered out during prep.
_ORDER_BY = re.compile(r"\border\s+by\b", re.IGNORECASE)


def build_db(context: str) -> sqlite3.Connection:
    """Materialise an in-memory DB from a CREATE/INSERT script. Raises on bad SQL."""
    con = sqlite3.connect(":memory:")
    con.text_factory = str
    con.executescript(context)
    return con


def run_sql(con: sqlite3.Connection, sql: str) -> list[tuple]:
    """Execute a single query and return all rows. Raises on bad SQL."""
    return con.execute(sql).fetchall()


def _norm_cell(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, str):
        return v.strip()
    return v


def _norm_rows(rows: Iterable[tuple]) -> list[tuple]:
    return [tuple(_norm_cell(c) for c in row) for row in rows]


def results_match(gold_rows: list[tuple], pred_rows: list[tuple], gold_sql: str) -> bool:
    """Compare result sets.

    Order matters only when the gold query asked for an order; otherwise two
    queries returning the same rows in a different order are both correct.
    """
    g, p = _norm_rows(gold_rows), _norm_rows(pred_rows)
    if _ORDER_BY.search(gold_sql):
        return g == p
    return sorted(g, key=repr) == sorted(p, key=repr)


def normalize_sql(sql: str) -> str:
    """Loose normalisation for the (secondary) exact-match metric."""
    sql = sql.strip().rstrip(";")
    sql = re.sub(r"\s+", " ", sql)
    sql = re.sub(r"\s*([(),])\s*", r"\1", sql)
    return sql.lower()


def gold_is_runnable(context: str, sql: str) -> bool:
    try:
        con = build_db(context)
        run_sql(con, sql)
        con.close()
        return True
    except Exception:
        return False
