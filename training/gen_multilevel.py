"""Generate diverse multi-level aggregation training data.

Multi-level means: aggregate rows into groups, then compute something *across*
those groups -- a share of the group total, a rank within the group, a top-N per
group, a comparison against the group's own average. The base model reaches for
a window function in the same SELECT as the GROUP BY, which windows over the
pre-aggregation rows and silently returns nonsense.

The first version of this generator produced one query shape per kind, with the
same column aliases every time (grp, sub, pct, rnk). A model trained on it
scored 100% on a held-out set drawn from the same generator and 1/6 on
hand-written questions: it had memorised the template, not the skill. Everything
below varies deliberately -- aliases, phrasing, schema shape, join count, and
the SQL formulation itself -- so that the only thing consistently learnable is
the two-stage structure.

    python gen_multilevel.py --n 4000 --out data/multilevel.jsonl
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

# Each domain names its own tables and columns, so nothing is a fixed keyword.
DOMAINS = [
    dict(fact="orders", dim="customers", dim_key="customer_id", item="products", item_key="product_id",
         gcol="country", scol="category", price="price", qty="quantity",
         status="status", good="completed", bad="cancelled", gname="country", sname="category"),
    dict(fact="shipments", dim="warehouses", dim_key="warehouse_id", item="parcels", item_key="parcel_id",
         gcol="zone", scol="carrier", price="rate", qty="weight",
         status="state", good="delivered", bad="lost", gname="zone", sname="carrier"),
    dict(fact="visits", dim="clinics", dim_key="clinic_id", item="services", item_key="service_id",
         gcol="city", scol="department", price="unit_price", qty="sessions",
         status="outcome", good="attended", bad="cancelled", gname="city", sname="department"),
    dict(fact="plays", dim="stations", dim_key="station_id", item="tracks", item_key="track_id",
         gcol="market", scol="genre", price="royalty", qty="spins",
         status="clearance", good="cleared", bad="blocked", gname="market", sname="genre"),
    dict(fact="enrolments", dim="campuses", dim_key="campus_id", item="courses", item_key="course_id",
         gcol="district", scol="subject", price="fee", qty="credits",
         status="state", good="enrolled", bad="withdrawn", gname="district", sname="subject"),
    dict(fact="tickets", dim="venues", dim_key="venue_id", item="events", item_key="event_id",
         gcol="region", scol="event_type", price="face_value", qty="seats",
         status="status", good="issued", bad="refunded", gname="region", sname="event type"),
]
GROUPS = ["India", "USA", "UK", "Germany", "Brazil", "Japan", "Kenya", "Spain", "Canada", "Italy"]
SUBS = ["Electronics", "Accessories", "Apparel", "Home", "Sports", "Books", "Garden", "Toys"]

# Alias pools. Fixed aliases are what made the first attempt memorisable.
A_GRP = ["g", "grp", "k", "cat", "bucket", "seg"]
A_TOT = ["total", "amount", "revenue", "value", "sum_val", "measure"]
A_PCT = ["pct", "percentage", "share", "pct_of_total", "share_pct", "portion"]
A_RNK = ["rnk", "rank_in_group", "position", "rn", "place", "ordinal"]
A_CNT = ["uniques", "distinct_count", "n_customers", "how_many", "party_count"]
A_CTE1 = ["agg", "totals", "per_group", "base", "rolled", "summed"]
A_CTE2 = ["ranked", "scored", "with_rank", "ordered", "final"]


def build_schema(d, rng, three_tables: bool) -> str:
    groups = rng.sample(GROUPS, rng.choice([3, 3, 4]))
    subs = rng.sample(SUBS, rng.choice([3, 3, 4]))
    L = []
    if three_tables:
        L.append(f"CREATE TABLE {d['dim']} ({d['dim_key']} INTEGER PRIMARY KEY, name TEXT, {d['gcol']} TEXT);")
        L.append(f"CREATE TABLE {d['item']} ({d['item_key']} INTEGER PRIMARY KEY, label TEXT, "
                 f"{d['scol']} TEXT, {d['price']} REAL);")
        L.append(f"CREATE TABLE {d['fact']} (id INTEGER PRIMARY KEY, {d['dim_key']} INTEGER, "
                 f"{d['item_key']} INTEGER, {d['qty']} INTEGER, {d['status']} TEXT);")
        L.append("INSERT INTO %s VALUES %s;" % (d["dim"], ",".join(
            f"({i},'n{i}','{g}')" for i, g in enumerate(groups, 1))))
        L.append("INSERT INTO %s VALUES %s;" % (d["item"], ",".join(
            f"({j},'l{j}','{s}',{rng.choice([50,75,100,250,500,800])})" for j, s in enumerate(subs, 1))))
        rows, fid = [], 1
        for i in range(1, len(groups) + 1):
            for j in range(1, len(subs) + 1):
                for _ in range(1 + (1 if rng.random() < .35 else 0)):
                    st = d["good"] if rng.random() < .75 else d["bad"]
                    rows.append(f"({fid},{i},{j},{rng.randint(1,4)},'{st}')")
                    fid += 1
        L.append(f"INSERT INTO {d['fact']} VALUES {','.join(rows)};")
    else:
        # flat single table -- no joins at all
        L.append(f"CREATE TABLE {d['fact']} (id INTEGER PRIMARY KEY, {d['gcol']} TEXT, {d['scol']} TEXT, "
                 f"{d['qty']} INTEGER, {d['price']} REAL, {d['status']} TEXT);")
        rows, fid = [], 1
        for g in groups:
            for s in subs:
                for _ in range(1 + (1 if rng.random() < .35 else 0)):
                    st = d["good"] if rng.random() < .75 else d["bad"]
                    rows.append(f"({fid},'{g}','{s}',{rng.randint(1,4)},"
                                f"{rng.choice([50,75,100,250,500,800])},'{st}')")
                    fid += 1
        L.append(f"INSERT INTO {d['fact']} VALUES {','.join(rows)};")
    return "\n".join(L)


def from_clause(d, three: bool) -> tuple[str, str, str, str]:
    """Returns (FROM+JOINs, group expr, sub expr, measure expr, id expr)."""
    if three:
        f = (f"FROM {d['fact']} f\n  JOIN {d['dim']} dd ON f.{d['dim_key']} = dd.{d['dim_key']}\n"
             f"  JOIN {d['item']} ii ON f.{d['item_key']} = ii.{d['item_key']}")
        return f, f"dd.{d['gcol']}", f"ii.{d['scol']}", f"SUM(f.{d['qty']} * ii.{d['price']})", f"f.{d['dim_key']}"
    f = f"FROM {d['fact']} f"
    return f, f"f.{d['gcol']}", f"f.{d['scol']}", f"SUM(f.{d['qty']} * f.{d['price']})", "f.id"


PHRASES = {
    "share": [
        "For each {g}, show every {s} with its total {m} and what percentage of that {g}'s total it represents. Only count {good} rows.",
        "What fraction of each {g}'s total {m} does each {s} account for? Give it as a percentage, counting only {good} records.",
        "Break down {m} by {g} and {s}, including each {s}'s share of its {g} as a percent. Ignore anything not {good}.",
        "Per {g}, list {s} totals alongside their percentage contribution to that {g}. Exclude {bad} rows.",
    ],
    "topn": [
        "For each {g}, find the top {n} {s} by total {m}. Show the total, the percentage of the {g} total, and the rank. Exclude {bad} rows.",
        "Which {n} {s} lead each {g} by {m}? Include the total, its share of the {g}, and where it ranks. Only {good} rows count.",
        "Give me the {n} highest-earning {s} within every {g}, with totals, percentage of {g} revenue, and rank. Skip {bad} entries.",
    ],
    "top1": [
        "In each {g}, which single {s} has the highest total {m}? Count only {good} rows.",
        "For every {g}, identify the leading {s} by {m}, ignoring {bad} rows.",
        "Show the best-performing {s} per {g} measured by total {m}, excluding {bad}.",
    ],
    "rank": [
        "Rank the {s} within each {g} by total {m}, showing the total and the rank. Count only {good} rows.",
        "Order {s} inside every {g} by their {m} total and number them. Exclude {bad} rows.",
        "For each {g}, number the {s} from highest to lowest {m}. Only {good} rows.",
    ],
    "above_avg": [
        "Which {s} have a total {m} above the average across all {s} in the same {g}? Only {good} rows count.",
        "List the {s} that beat their own {g}'s typical {s} total for {m}, excluding {bad} rows.",
        "For every {g}, show the {s} whose {m} total exceeds that {g}'s mean. Ignore {bad} records.",
    ],
}


def make(d, rng) -> dict | None:
    three = rng.random() < 0.6
    schema = build_schema(d, rng, three)
    frm, gexpr, sexpr, mexpr, idexpr = from_clause(d, three)
    kind = rng.choice(["share", "topn", "top1", "rank", "above_avg"])
    n = rng.choice([2, 2, 3])

    g, tot, pct, rnk, cnt = (rng.choice(A_GRP), rng.choice(A_TOT), rng.choice(A_PCT),
                            rng.choice(A_RNK), rng.choice(A_CNT))
    c1, c2 = rng.choice(A_CTE1), rng.choice(A_CTE2)
    sub = rng.choice(["s", "item", "kind", "label", "b"])

    q = rng.choice(PHRASES[kind]).format(g=d["gname"], s=d["sname"], m=d["price"],
                                         good=d["good"], bad=d["bad"], n=n)

    base_cte = (f"WITH {c1} AS (\n  SELECT {gexpr} AS {g}, {sexpr} AS {sub},\n"
                f"         {mexpr} AS {tot},\n"
                f"         COUNT(DISTINCT {idexpr}) AS {cnt}\n  {frm}\n"
                f"  WHERE f.{d['status']} = '{d['good']}'\n"
                f"  GROUP BY {gexpr}, {sexpr}\n)")

    if kind == "share":
        sql = (f"{base_cte}\nSELECT {g}, {sub}, {tot},\n"
               f"       {tot} * 100.0 / SUM({tot}) OVER (PARTITION BY {g}) AS {pct}\n"
               f"FROM {c1}\nORDER BY {g}, {pct} DESC")
    elif kind == "topn":
        sql = (f"{base_cte},\n{c2} AS (\n  SELECT {g}, {sub}, {tot}, {cnt},\n"
               f"         {tot} * 100.0 / SUM({tot}) OVER (PARTITION BY {g}) AS {pct},\n"
               f"         RANK() OVER (PARTITION BY {g} ORDER BY {tot} DESC) AS {rnk}\n"
               f"  FROM {c1}\n)\nSELECT * FROM {c2} WHERE {rnk} <= {n} ORDER BY {g}, {rnk}")
    elif kind == "top1":
        # deliberately a different formulation from the ranked CTE above
        if rng.random() < 0.5:
            sql = (f"{base_cte},\n{c2} AS (\n  SELECT {g}, {sub}, {tot},\n"
                   f"         RANK() OVER (PARTITION BY {g} ORDER BY {tot} DESC) AS {rnk}\n"
                   f"  FROM {c1}\n)\nSELECT {g}, {sub}, {tot} FROM {c2} WHERE {rnk} = 1 ORDER BY {g}")
        else:
            sql = (f"{base_cte}\nSELECT {g}, {sub}, {tot} FROM {c1} t\n"
                   f"WHERE {tot} = (SELECT MAX({tot}) FROM {c1} u WHERE u.{g} = t.{g})\n"
                   f"ORDER BY {g}")
    elif kind == "rank":
        sql = (f"{base_cte}\nSELECT {g}, {sub}, {tot},\n"
               f"       RANK() OVER (PARTITION BY {g} ORDER BY {tot} DESC) AS {rnk}\n"
               f"FROM {c1}\nORDER BY {g}, {rnk}")
    else:
        if rng.random() < 0.5:
            sql = (f"{base_cte},\n{c2} AS (\n  SELECT {g}, {sub}, {tot}, AVG({tot}) OVER (PARTITION BY {g}) AS avg_val\n"
                   f"  FROM {c1}\n)\nSELECT {g}, {sub}, {tot} FROM {c2} WHERE {tot} > avg_val ORDER BY {g}, {tot} DESC")
        else:
            sql = (f"{base_cte}\nSELECT {g}, {sub}, {tot} FROM {c1} t\n"
                   f"WHERE {tot} > (SELECT AVG({tot}) FROM {c1} u WHERE u.{g} = t.{g})\n"
                   f"ORDER BY {g}, {tot} DESC")

    try:
        con = sqlite3.connect(":memory:")
        con.executescript(schema)
        cur = con.execute(sql)
        cols = [c[0] for c in cur.description]
        rows = cur.fetchall()
        con.close()
    except Exception:
        return None
    if not rows:
        return None
    if pct in cols:
        i = cols.index(pct)
        if any(r[i] is None or r[i] <= 0 or r[i] > 100.0001 for r in rows):
            return None
    if rnk in cols:
        i, gi = cols.index(rnk), cols.index(g)
        seen: dict = {}
        for r in rows:
            seen.setdefault(r[gi], []).append(r[i])
        if any(min(v) != 1 for v in seen.values()):
            return None
        if kind == "topn" and any(r[i] > n for r in rows):
            return None

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(schema=schema, question=q)},
            {"role": "assistant", "content": sql},
        ],
        "sql_context": schema, "sql_prompt": q, "gold_sql": sql,
        "domain": d["fact"], "complexity": "multi-level aggregation",
        "source": "synthetic-multilevel", "id": None, "kind": kind,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("data/multilevel.jsonl"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out, attempts, kinds = [], 0, {}
    while len(out) < args.n and attempts < args.n * 40:
        attempts += 1
        ex = make(rng.choice(DOMAINS), rng)
        if ex:
            kinds[ex["kind"]] = kinds.get(ex["kind"], 0) + 1
            ex.pop("kind")
            out.append(ex)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(f"  generated {len(out)} in {attempts} attempts -> {args.out}")
    for k, v in sorted(kinds.items(), key=lambda x: -x[1]):
        print(f"    {v:>5}  {k}")


if __name__ == "__main__":
    main()
