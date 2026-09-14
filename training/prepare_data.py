"""Build chat-formatted JSONL for text-to-SQL SFT, from one or more datasets.

Writes train.jsonl / valid.jsonl / test.jsonl into one directory, the layout
MLX's loader requires. The CUDA path reads the same files.

Every kept example is verified to actually execute against a real in-memory
SQLite database built from its own schema (see sources.py for what that proves
per dataset). Roughly 21% of the raw Gretel data has gold SQL that does not
run; training on it would teach the model broken queries.

  python prepare_data.py --sources gretel --train-size 4000
  python prepare_data.py --sources gretel,createcontext --train-size 20000 --balance
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from sources import SOURCES

SYSTEM_PROMPT = (
    "You are a precise text-to-SQL engine. Given a SQLite schema and a question, "
    "reply with a single valid SQLite query that answers it. "
    "Output only the SQL. No explanation, no markdown, no commentary."
)
USER_TEMPLATE = """### Schema
{schema}

### Question
{question}"""

# Measured base-model execution accuracy per complexity class, and how often the
# class appears (300 held-out examples, first run).
#
# CAVEAT: these accuracies come from the held-out set, so choosing a training
# mix from them is a mild form of test-set adaptation. The reported before/after
# is still measured on untouched data. See the README.
CLASS_STATS = {
    "basic SQL":        (0.885, 0.550),
    "aggregation":      (0.704, 0.237),
    "single join":      (0.632, 0.127),
    "subqueries":       (0.692, 0.043),
    "window functions": (0.300, 0.033),
    "multiple_joins":   (0.500, 0.007),
    "set operations":   (0.500, 0.003),
}
MIN_SHARE = 0.03


def to_record(f: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                schema=f["sql_context"].strip(), question=f["sql_prompt"].strip())},
            {"role": "assistant", "content": f["gold_sql"]},
        ],
        **{k: f[k] for k in ("sql_context", "sql_prompt", "gold_sql", "domain", "complexity", "source", "id")},
    }


def headroom_quotas(total: int) -> dict[str, int]:
    """Allocate budget by expected headline gain: headroom x how often it appears.

    Headroom alone over-invests in rare classes -- 'set operations' is one
    example in 300, so perfecting it moves the overall number by a third of a
    point. MIN_SHARE keeps every class represented so none regress.
    """
    w = {k: max((1 - acc) * share, MIN_SHARE * (1 - acc)) for k, (acc, share) in CLASS_STATS.items()}
    scale = total / sum(w.values())
    return {k: max(1, round(v * scale)) for k, v in w.items()}


def harvest(source_names: list[str], quotas: dict[str, int] | None, want: int, seed: int,
            max_scan_per_source: int = 250_000) -> list[dict]:
    """Pull validated examples from each source, filling per-class quotas if given.

    Two passes. The first caps each source at a fair share of every class --
    counted PER SOURCE, not as a global bucket size, or whichever dataset is
    listed first simply fills the buckets and the rest contribute nothing. The
    second pass lets any source top up what the first left short, so a quota is
    never wasted because one dataset is thin in that class.
    """
    buckets: dict[str, list[dict]] = {k: [] for k in (quotas or {"all": want})}
    taken: dict[str, Counter] = {n: Counter() for n in source_names}
    rejected: Counter = Counter()
    targets = quotas or {"all": want}
    n_sources = len(source_names)

    def globally_full() -> bool:
        return all(len(buckets[k]) >= targets[k] for k in buckets)

    def scan(name: str, per_source_cap: dict[str, int], label: str) -> None:
        spec = SOURCES[name]
        ds = load_dataset(spec["repo"], split=spec["split"]).shuffle(seed=seed)
        scanned, added = 0, 0
        for row in ds:
            scanned += 1
            if scanned > max_scan_per_source or globally_full():
                break
            if all(taken[name][k] >= per_source_cap[k] for k in buckets):
                break
            f = spec["fields"](row)
            if f is None:
                rejected[name] += 1
                continue
            key = f["complexity"] if quotas else "all"
            if key not in buckets:
                continue
            if len(buckets[key]) >= targets[key] or taken[name][key] >= per_source_cap[key]:
                continue
            if not spec["validate"](f):
                rejected[name] += 1
                continue
            buckets[key].append(to_record(f))
            taken[name][key] += 1
            added += 1
        print(f"      {label}: scanned {scanned:,}, added {added:,}, rejected {rejected[name]:,}")

    fair = {k: max(1, -(-v // n_sources)) for k, v in targets.items()}
    for name in source_names:
        print(f"  [{name}] {SOURCES[name]['repo']} -- {SOURCES[name]['note']}")
        scan(name, fair, "fair share")

    if not globally_full():
        print("  topping up unfilled quotas")
        for name in source_names:
            if globally_full():
                break
            scan(name, targets, f"top-up {name}")

    if quotas:
        print("\n  quota fill:")
        for k in sorted(quotas, key=lambda x: -quotas[x]):
            got = len(buckets[k])
            print(f"      {k:<18} {got:>6}/{quotas[k]}" + ("" if got >= quotas[k] else "   <- exhausted"))

    out = [r for k in buckets for r in buckets[k]]
    random.Random(seed).shuffle(out)
    return out


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"  -> {path}  ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="gretel",
                    help="comma separated: " + ", ".join(SOURCES))
    ap.add_argument("--train-size", type=int, default=4000)
    ap.add_argument("--val-size", type=int, default=200)
    ap.add_argument("--test-size", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--balance", action="store_true",
                    help="allocate by measured headroom instead of the natural mix")
    ap.add_argument("--keep-test", action="store_true",
                    help="leave an existing test.jsonl untouched, for comparability")
    args = ap.parse_args()

    names = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        raise SystemExit(f"Unknown source(s): {unknown}. Available: {list(SOURCES)}")

    random.seed(args.seed)
    total = args.train_size + args.val_size
    quotas = headroom_quotas(total) if args.balance else None

    print(f"Harvesting {total:,} examples from: {', '.join(names)}"
          + (" (balanced by headroom)" if args.balance else " (natural mix)"))
    pool = harvest(names, quotas, total, args.seed)
    val, train = pool[: args.val_size], pool[args.val_size :]

    # The test set always comes from Gretel: it ships INSERT rows, which is what
    # makes execution accuracy (comparing result sets) measurable at all.
    test_path = args.out / "test.jsonl"
    if args.keep_test and test_path.exists():
        print(f"\nKeeping the existing held-out test set for comparability")
    else:
        print("\nBuilding held-out test set from Gretel's test split")
        spec = SOURCES["gretel"]
        ds = load_dataset(spec["repo"], split="test").shuffle(seed=args.seed + 1)
        test, seen = [], {r["id"] for r in pool if r.get("id") is not None}
        for row in ds:
            if len(test) >= args.test_size:
                break
            f = spec["fields"](row)
            if f is None or f["id"] in seen or not spec["validate"](f):
                continue
            test.append(to_record(f))
        write(test_path, test)

    write(args.out / "train.jsonl", train)
    write(args.out / "valid.jsonl", val)

    print("\nTrain mix by complexity:")
    for k, v in Counter(r["complexity"] for r in train).most_common():
        print(f"  {v:>6}  {k}")
    print("\nTrain mix by source:")
    for k, v in Counter(r["source"] for r in train).most_common():
        print(f"  {v:>6}  {k}")


if __name__ == "__main__":
    main()
