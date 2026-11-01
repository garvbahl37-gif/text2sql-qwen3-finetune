"""Enrich a bare DDL schema with the signals strong text-to-SQL systems provide.

Schema linking -- deciding which table a column belongs to -- is the dominant
failure mode in text-to-SQL, reported at over 60% of errors in the 2025
literature (LinkAlign, JOLT-SQL, LitE-SQL). Our prompt sent raw CREATE TABLE and
nothing else, so the model had to infer column ownership from naming alone. It
frequently guessed wrong: given an e-commerce schema it wrote `f.country` where
country lives on `customers`, producing a perfectly structured query that would
not run.

This adds, from the schema itself with no extra data source:

  * a column index naming the one table each column belongs to
  * inferred join paths from <table>_id columns to their primary keys
  * an explicit warning for column names that appear in more than one table
  * optional sample rows, which the literature finds helps value grounding

Everything here is derived by querying SQLite's own catalog, so it stays correct
for any schema rather than relying on a naming convention.
"""
from __future__ import annotations

import sqlite3


def _catalog(schema: str) -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.executescript(schema)
    return con


def enrich_schema(schema: str, sample_rows: int = 0) -> str:
    """Return the schema plus a comment block describing its structure."""
    try:
        con = _catalog(schema)
    except Exception:
        return schema  # unparseable: leave it exactly as given

    try:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        if not tables:
            return schema

        info = {t: list(con.execute(f"PRAGMA table_info({t})")) for t in tables}
        lines = ["-- Column index. Each column belongs to exactly the table listed here;",
                 "-- qualify every column with its own table and do not move one between tables."]

        owner: dict[str, list[str]] = {}
        for t in tables:
            cols = [(r[1], r[2] or "") for r in info[t]]
            pks = [r[1] for r in info[t] if r[5]]
            desc = ", ".join(f"{c}{'(' + ty + ')' if ty else ''}" for c, ty in cols)
            lines.append(f"--   {t}: {desc}" + (f"   [primary key: {', '.join(pks)}]" if pks else ""))
            for c, _ in cols:
                owner.setdefault(c, []).append(t)

        # Join paths: a <name>_id column that is another table's primary key.
        pk_of = {t: [r[1] for r in info[t] if r[5]] for t in tables}
        joins = set()
        for t in tables:
            for c in [r[1] for r in info[t]]:
                for other in tables:
                    if other != t and c in pk_of.get(other, []):
                        joins.add(f"--   {t}.{c} = {other}.{c}")
        if joins:
            lines.append("-- Join paths:")
            lines += sorted(joins)

        shared = {c: ts for c, ts in owner.items() if len(ts) > 1}
        if shared:
            lines.append("-- These column names appear in several tables, so always qualify them:")
            for c, ts in sorted(shared.items()):
                lines.append(f"--   {c}: {', '.join(ts)}")

        if sample_rows > 0:
            lines.append("-- Sample values:")
            for t in tables:
                try:
                    rows = con.execute(f"SELECT * FROM {t} LIMIT {sample_rows}").fetchall()
                except Exception:
                    continue
                if rows:
                    cols = [r[1] for r in info[t]]
                    lines.append(f"--   {t}: " + " | ".join(
                        ", ".join(f"{c}={v!r}" for c, v in zip(cols, row)) for row in rows))

        return schema.rstrip() + "\n\n" + "\n".join(lines)
    finally:
        con.close()
