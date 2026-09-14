"""Turn gretelai/synthetic_text_to_sql into chat-formatted JSONL for SFT.

Every kept example is verified to actually run against a real in-memory SQLite
DB built from its own schema. That guarantees execution accuracy is measurable
on the held-out split instead of being an approximation.

    python prepare_data.py --train-size 8000 --test-size 300
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from sqlutil import gold_is_runnable

DATASET = "gretelai/synthetic_text_to_sql"

SYSTEM_PROMPT = (
    "You are a precise text-to-SQL engine. Given a SQLite schema and a question, "
    "reply with a single valid SQLite query that answers it. "
    "Output only the SQL. No explanation, no markdown, no commentary."
)

USER_TEMPLATE = """### Schema
{schema}

### Question
{question}"""


def build_prompt(schema: str, question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(schema=schema.strip(), question=question.strip())},
    ]


def to_record(row: dict) -> dict:
    sql = row["sql"].strip().rstrip(";")
    return {
        "messages": build_prompt(row["sql_context"], row["sql_prompt"]) + [{"role": "assistant", "content": sql}],
        "sql_context": row["sql_context"],
        "sql_prompt": row["sql_prompt"],
        "gold_sql": sql,
        "domain": row.get("domain", ""),
        "complexity": row.get("sql_complexity", ""),
    }


def collect(split, want: int, seed: int, skip_ids: set | None = None) -> list[dict]:
    """Walk a shuffled split, keeping only examples whose gold SQL executes."""
    kept, seen, rejected = [], set(), 0
    for row in split.shuffle(seed=seed):
        if len(kept) >= want:
            break
        rid = row.get("id")
        if rid in seen or (skip_ids and rid in skip_ids):
            continue
        seen.add(rid)
        if not row.get("sql_context") or "CREATE TABLE" not in row["sql_context"].upper():
            rejected += 1
            continue
        if not gold_is_runnable(row["sql_context"], row["sql"]):
            rejected += 1
            continue
        rec = to_record(row)
        rec["id"] = rid
        kept.append(rec)
    print(f"    kept {len(kept)}, rejected {rejected} (non-executable gold)")
    return kept


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  -> {path}  ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-size", type=int, default=8000)
    ap.add_argument("--val-size", type=int, default=400)
    ap.add_argument("--test-size", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()

    random.seed(args.seed)
    print(f"Loading {DATASET} ...")
    ds = load_dataset(DATASET)
    print(f"  train={len(ds['train'])}  test={len(ds['test'])}")

    print("Building train split (validating gold SQL executes)...")
    train = collect(ds["train"], args.train_size + args.val_size, args.seed)
    val, train = train[: args.val_size], train[args.val_size :]

    print("Building held-out test split...")
    test = collect(ds["test"], args.test_size, args.seed + 1, skip_ids={r["id"] for r in train + val})

    write(args.out / "train.jsonl", train)
    write(args.out / "val.jsonl", val)
    write(args.out / "test.jsonl", test)

    dist = Counter(r["complexity"] for r in train)
    print("\nTrain complexity mix:")
    for k, v in dist.most_common(10):
        print(f"  {v:5d}  {k}")
    print(f"\nDomains covered: {len({r['domain'] for r in train})}")


if __name__ == "__main__":
    main()
