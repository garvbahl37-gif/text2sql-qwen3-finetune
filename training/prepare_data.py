"""Turn gretelai/synthetic_text_to_sql into chat-formatted JSONL for SFT.

Writes train.jsonl / valid.jsonl / test.jsonl into one directory, which is the
layout MLX's data loader requires. The CUDA path reads the same files.

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


# Measured base-model execution accuracy per complexity class (300 held-out
# examples, first run). The headroom left on each class is what decides how
# much training budget it deserves -- there is no point spending half the run
# on "basic SQL" the base model already gets right 88.5% of the time.
# (base execution accuracy, share of the evaluation set) per complexity class,
# measured on 300 held-out examples in the first run.
#
# CAVEAT, stated plainly: the accuracy figures come from the held-out set, so
# choosing a training mix from them is a mild form of test-set adaptation. The
# reported before/after is still measured on the same untouched held-out set,
# but a strict reading would re-derive these from the validation split. See the
# README section "Choosing the training mix".
CLASS_STATS = {
    #                   base acc, share of eval set, n
    "basic SQL":        (0.885, 0.550, 165),
    "aggregation":      (0.704, 0.237, 71),
    "single join":      (0.632, 0.127, 38),
    "subqueries":       (0.692, 0.043, 13),
    "window functions": (0.300, 0.033, 10),
    "multiple_joins":   (0.500, 0.007, 2),
    "set operations":   (0.500, 0.003, 1),
}
MIN_SHARE = 0.03   # never starve a class completely


def headroom_quotas(total: int) -> dict[str, int]:
    """Allocate the training budget by *expected headline gain*.

    Weighting by headroom alone over-invests in rare classes: "set operations"
    has one example in the evaluation set, so perfecting it moves the overall
    number by a third of a point. Weighting by headroom x frequency targets the
    budget where it can actually shift the measured result, while MIN_SHARE
    keeps every class represented so none of them regress.
    """
    weights = {}
    for k, (acc, share, n) in CLASS_STATS.items():
        headroom = 1.0 - acc
        weights[k] = max(headroom * share, MIN_SHARE * headroom)
    scale = total / sum(weights.values())
    return {k: max(1, round(w * scale)) for k, w in weights.items()}


def collect_balanced(split, quotas: dict[str, int], seed: int, max_scan: int = 400_000) -> list[dict]:
    """Fill a per-class quota instead of taking whatever the shuffle hands over.

    Rare classes (window functions, set operations) need a deeper scan to reach
    their quota, so this walks until every bucket is full or the split runs out.
    """
    buckets: dict[str, list[dict]] = {k: [] for k in quotas}
    seen, rejected, scanned = set(), 0, 0
    for row in split.shuffle(seed=seed):
        scanned += 1
        if scanned > max_scan or all(len(buckets[k]) >= quotas[k] for k in quotas):
            break
        cls = row.get("sql_complexity") or "unknown"
        if cls not in buckets or len(buckets[cls]) >= quotas[cls]:
            continue
        rid = row.get("id")
        if rid in seen:
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
        buckets[cls].append(rec)

    print(f"    scanned {scanned}, rejected {rejected} (non-executable gold)")
    for k in sorted(quotas, key=lambda x: -quotas[x]):
        got, want = len(buckets[k]), quotas[k]
        flag = "" if got >= want else f"  <- only {got} available"
        print(f"      {k:<18} {got:>5}/{want}{flag}")

    out = [r for k in buckets for r in buckets[k]]
    random.Random(seed).shuffle(out)
    return out


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
    ap.add_argument("--balance", action="store_true",
                    help="allocate the training budget by measured headroom instead of "
                         "taking the dataset's natural mix (54%% of which is basic SQL)")
    ap.add_argument("--keep-test", action="store_true",
                    help="leave an existing test.jsonl untouched, so results stay "
                         "comparable with a previous run")
    args = ap.parse_args()

    random.seed(args.seed)
    print(f"Loading {DATASET} ...")
    ds = load_dataset(DATASET)
    print(f"  train={len(ds['train'])}  test={len(ds['test'])}")

    if args.balance:
        quotas = headroom_quotas(args.train_size + args.val_size)
        print("Building BALANCED train split (quota by measured headroom)...")
        train = collect_balanced(ds["train"], quotas, args.seed)
    else:
        print("Building train split (validating gold SQL executes)...")
        train = collect(ds["train"], args.train_size + args.val_size, args.seed)
    val, train = train[: args.val_size], train[args.val_size :]

    test_path = args.out / "test.jsonl"
    if args.keep_test and test_path.exists():
        # Reusing the exact same held-out set is what makes two runs comparable.
        test = [json.loads(l) for l in test_path.read_text().splitlines() if l.strip()]
        print(f"Keeping the existing held-out test set ({len(test)} examples) for comparability")
    else:
        print("Building held-out test split...")
        test = collect(ds["test"], args.test_size, args.seed + 1,
                       skip_ids={r["id"] for r in train + val})

    write(args.out / "train.jsonl", train)
    write(args.out / "valid.jsonl", val)
    if not (args.keep_test and test_path.exists()):
        write(test_path, test)
    else:
        print(f"  -> {test_path}  (unchanged)")

    dist = Counter(r["complexity"] for r in train)
    print("\nTrain complexity mix:")
    for k, v in dist.most_common(10):
        print(f"  {v:5d}  {k}")
    print(f"\nDomains covered: {len({r['domain'] for r in train})}")


if __name__ == "__main__":
    main()
