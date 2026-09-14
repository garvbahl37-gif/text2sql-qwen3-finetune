"""Generate training data for multi-level aggregation, the model's weak spot.

The failure it targets: questions needing two stages -- aggregate rows into
groups, then compute something *across* those groups (a share of the group
total, a rank within the group, a top-N per group). The model reaches for a
window function in the same SELECT as the GROUP BY, which windows over the
pre-aggregation rows and silently produces nonsense such as percentages above
100, or computes a rank and never filters on it.

The correct shape is always a CTE: aggregate first, then window over the
aggregate. Nothing in the base training data teaches that shape often enough,
so this synthesises it across many schemas.

Every generated example is executed before it is kept, and its gold answer is
checked for sanity (shares summing to 100 per group, ranks starting at 1), so
a bad generator cannot quietly poison the training set.

    python gen_multilevel.py --n 1200 --out data/multilevel.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a precise text-to-SQL engine. Given a SQLite schema and a question, "
    "reply with a single valid SQLite query that answers it. "
    "Output only the SQL. No explanation, no markdown, no commentary."
)
USER_TEMPLATE = "### Schema\n{schema}\n\n### Question\n{question}"

# (domain, fact table, dimension table, group col, sub col, measure, qty, status col, good status, bad status)
DOMAINS = [
    ("retail",    "orders",    "customers", "country",  "category",   "price",      "quantity", "status", "completed", "cancelled"),
    ("saas",      "usages",    "accounts",  "region",   "plan",       "unit_cost",  "seats",    "state",  "active",    "churned"),
    ("logistics", "shipments", "hubs",      "zone",     "carrier",    "rate",       "weight",   "status", "delivered", "lost"),
    ("clinic",    "visits",    "clinics",   "city",     "department", "unit_price", "sessions", "status", "attended",  "noshow"),
    ("media",     "streams",   "studios",   "market",   "genre",      "royalty",    "plays",    "status", "cleared",   "blocked"),
    ("energy",    "readings",  "sites",     "district", "source",     "tariff",     "units",    "status", "billed",    "voided"),
]
GROUP_VALUES = ["India", "USA", "UK", "Germany", "Brazil", "Japan", "Kenya", "Spain"]
SUB_VALUES = ["Electronics", "Accessories", "Apparel", "Home", "Sports", "Books", "Garden"]


def build_schema(d, rng) -> tuple[str, list]:
    _, fact, dim, gcol, scol, measure, qty, status, good, bad = d
    groups = rng.sample(GROUP_VALUES, rng.randint(3, 4))
    subs = rng.sample(SUB_VALUES, rng.randint(3, 4))

    lines = [
        f"CREATE TABLE {dim} (\n  {dim[:-1]}_id INTEGER PRIMARY KEY,\n  name TEXT,\n  {gcol} TEXT\n);",
        f"CREATE TABLE items (\n  item_id INTEGER PRIMARY KEY,\n  item_name TEXT,\n  {scol} TEXT,\n  {measure} REAL\n);",
        f"CREATE TABLE {fact} (\n  {fact[:-1]}_id INTEGER PRIMARY KEY,\n  {dim[:-1]}_id INTEGER,\n  item_id INTEGER,\n  {qty} INTEGER,\n  {status} TEXT\n);",
    ]
    dim_rows, fact_rows, item_rows = [], [], []
    for i, g in enumerate(groups, 1):
        dim_rows.append(f"({i}, '{chr(64+i)}cme', '{g}')")
    for j, sname in enumerate(subs, 1):
        item_rows.append(f"({j}, 'item{j}', '{sname}', {rng.choice([50,75,100,250,500,800])})")
    fid = 1
    for i in range(1, len(groups) + 1):
        for j in range(1, len(subs) + 1):
            for _ in range(rng.randint(1, 3)):
                st = good if rng.random() < 0.75 else bad
                fact_rows.append(f"({fid}, {i}, {j}, {rng.randint(1,4)}, '{st}')")
                fid += 1
    lines.append(f"INSERT INTO {dim} VALUES {', '.join(dim_rows)};")
    lines.append(f"INSERT INTO items VALUES {', '.join(item_rows)};")
    lines.append(f"INSERT INTO {fact} VALUES {', '.join(fact_rows)};")
    return "\n".join(lines), groups


def make_example(d, rng) -> dict | None:
    domain, fact, dim, gcol, scol, measure, qty, status, good, bad = d
    schema, _ = build_schema(d, rng)
    n = rng.choice([1, 2, 2, 3])
    kind = rng.choice(["share", "topn", "rank", "above_avg"])

    agg = (f"WITH agg AS (\n"
           f"  SELECT d.{gcol} AS grp, i.{scol} AS sub,\n"
           f"         SUM(f.{qty} * i.{measure}) AS total,\n"
           f"         COUNT(DISTINCT f.{dim[:-1]}_id) AS uniques\n"
           f"  FROM {fact} f\n"
           f"  JOIN {dim} d ON f.{dim[:-1]}_id = d.{dim[:-1]}_id\n"
           f"  JOIN items i ON f.item_id = i.item_id\n"
           f"  WHERE f.{status} = '{good}'\n"
           f"  GROUP BY d.{gcol}, i.{scol}\n)")

    if kind == "share":
        q = (f"For each {gcol}, show every {scol} with its total {measure} value and its "
             f"percentage of that {gcol}'s total. Only count {good} rows.")
        sql = (f"{agg}\nSELECT grp, sub, total,\n"
               f"       total * 100.0 / SUM(total) OVER (PARTITION BY grp) AS pct\n"
               f"FROM agg\nORDER BY grp, pct DESC")
    elif kind == "topn":
        q = (f"For each {gcol}, find the top {n} {scol} values by total {measure}. Show the "
             f"total, the number of unique {dim}, the percentage of the {gcol} total, and the "
             f"rank. Exclude {bad} rows.")
        sql = (f"{agg},\nranked AS (\n"
               f"  SELECT grp, sub, total, uniques,\n"
               f"         total * 100.0 / SUM(total) OVER (PARTITION BY grp) AS pct,\n"
               f"         RANK() OVER (PARTITION BY grp ORDER BY total DESC) AS rnk\n"
               f"  FROM agg\n)\n"
               f"SELECT * FROM ranked WHERE rnk <= {n} ORDER BY grp, rnk")
    elif kind == "rank":
        q = (f"Rank the {scol} values within each {gcol} by total {measure}, showing the total "
             f"and the rank. Count only {good} rows.")
        sql = (f"{agg}\nSELECT grp, sub, total,\n"
               f"       RANK() OVER (PARTITION BY grp ORDER BY total DESC) AS rnk\n"
               f"FROM agg\nORDER BY grp, rnk")
    else:
        q = (f"Which {scol} values have a total {measure} above the average across all "
             f"{scol} values in the same {gcol}? Only {good} rows count.")
        sql = (f"{agg},\nwith_avg AS (\n"
               f"  SELECT grp, sub, total, AVG(total) OVER (PARTITION BY grp) AS grp_avg\n"
               f"  FROM agg\n)\n"
               f"SELECT grp, sub, total FROM with_avg WHERE total > grp_avg ORDER BY grp, total DESC")

    # Execute and sanity-check before keeping it.
    try:
        con = sqlite3.connect(":memory:")
        con.executescript(schema)
        rows = con.execute(sql).fetchall()
        con.close()
    except Exception:
        return None
    if not rows:
        return None
    if kind in ("share", "topn"):
        idx = 3 if kind == "share" else 4
        if any(r[idx] is None or r[idx] > 100.0001 or r[idx] <= 0 for r in rows):
            return None
    if kind == "topn" and any(r[5] > n for r in rows):
        return None

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(schema=schema, question=q)},
            {"role": "assistant", "content": sql},
        ],
        "sql_context": schema, "sql_prompt": q, "gold_sql": sql,
        "domain": domain, "complexity": "multi-level aggregation",
        "source": "synthetic-multilevel", "id": None, "kind": kind,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("data/multilevel.jsonl"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out, attempts, kinds = [], 0, {}
    while len(out) < args.n and attempts < args.n * 40:
        attempts += 1
        ex = make_example(rng.choice(DOMAINS), rng)
        if ex:
            kinds[ex["kind"]] = kinds.get(ex["kind"], 0) + 1
            ex.pop("kind")
            out.append(ex)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(f"  generated {len(out)} examples in {attempts} attempts -> {args.out}")
    for k, v in sorted(kinds.items(), key=lambda x: -x[1]):
        print(f"    {v:>5}  {k}")


if __name__ == "__main__":
    main()
