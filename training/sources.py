"""Training sources for text-to-SQL, and how to validate each one.

Two datasets with complementary weaknesses:

  gretel        synthetic, but ships INSERT rows, so gold SQL can be verified by
                actually running it and the eval can compare result sets.
  createcontext derived from Spider and WikiSQL, so real-world phrasing and
                schemas, but CREATE TABLE only. Gold SQL is validated by
                executing it against the empty schema, which still proves the
                syntax parses and every table and column reference resolves.

Mixing them trades a little validation strength for a lot of linguistic and
schema diversity, which is what the model was short of.
"""
from __future__ import annotations

import re

from sqlutil import build_db, run_sql

# Gretel's own complexity labels, inferred for datasets that lack them so the
# headroom-based balancing in prepare_data.py can span both sources.
_AGG = re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX|GROUP\s+BY|HAVING)\b", re.I)
_JOIN = re.compile(r"\bJOIN\b", re.I)
_WINDOW = re.compile(r"\bOVER\s*\(", re.I)
_SETOP = re.compile(r"\b(UNION|INTERSECT|EXCEPT)\b", re.I)
_SUBQ = re.compile(r"\(\s*SELECT\b", re.I)


def classify_sql(sql: str) -> str:
    """Best-effort complexity label matching Gretel's vocabulary."""
    if _WINDOW.search(sql):
        return "window functions"
    if _SETOP.search(sql):
        return "set operations"
    if _SUBQ.search(sql):
        return "subqueries"
    joins = len(_JOIN.findall(sql))
    if joins > 1:
        return "multiple_joins"
    if joins == 1:
        return "single join"
    if _AGG.search(sql):
        return "aggregation"
    return "basic SQL"


def _executes(context: str, sql: str) -> bool:
    try:
        con = build_db(context)
    except Exception:
        return False
    try:
        run_sql(con, sql)
        return True
    except Exception:
        return False
    finally:
        con.close()


# --- gretelai/synthetic_text_to_sql -----------------------------------------

def gretel_fields(row: dict) -> dict | None:
    ctx = row.get("sql_context") or ""
    if "CREATE TABLE" not in ctx.upper():
        return None
    return {
        "sql_context": ctx,
        "sql_prompt": row["sql_prompt"],
        "gold_sql": row["sql"].strip().rstrip(";"),
        "domain": row.get("domain", ""),
        "complexity": row.get("sql_complexity") or classify_sql(row["sql"]),
        "id": row.get("id"),
        "source": "gretel",
    }


# --- b-mc2/sql-create-context -----------------------------------------------

def createcontext_fields(row: dict) -> dict | None:
    ctx = row.get("context") or ""
    if "CREATE TABLE" not in ctx.upper():
        return None
    sql = (row.get("answer") or "").strip().rstrip(";")
    if not sql:
        return None
    return {
        "sql_context": ctx,
        "sql_prompt": row["question"],
        "gold_sql": sql,
        "domain": "",
        "complexity": classify_sql(sql),
        "id": None,
        "source": "createcontext",
    }


SOURCES = {
    "gretel": {
        "repo": "gretelai/synthetic_text_to_sql",
        "split": "train",
        "fields": gretel_fields,
        # Has INSERT rows, so this is a genuine execution check.
        "validate": lambda f: _executes(f["sql_context"], f["gold_sql"]),
        "note": "synthetic, executable with data",
    },
    "createcontext": {
        "repo": "b-mc2/sql-create-context",
        "split": "train",
        "fields": createcontext_fields,
        # Empty tables, so this proves syntax and schema references only.
        "validate": lambda f: _executes(f["sql_context"], f["gold_sql"]),
        "note": "Spider/WikiSQL derived, schema-only validation",
    },
}
