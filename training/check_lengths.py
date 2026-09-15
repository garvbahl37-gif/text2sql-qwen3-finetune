"""Fail loudly if training examples exceed the configured sequence cap.

MLX truncates over-long sequences and only prints a warning, which is easy to
miss in a long log. Truncating a gold SQL answer mid-query teaches the model to
emit unfinished statements, so this refuses to start instead.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/train.jsonl"))
    ap.add_argument("--tokenizer", default="mlx-community/Qwen3-4B-Instruct-2507-4bit")
    ap.add_argument("--max-seq", type=int, required=True)
    ap.add_argument("--tolerate", type=float, default=0.0,
                    help="fraction of examples allowed to exceed the cap")
    ap.add_argument("--drop", action="store_true",
                    help="remove over-long examples and rewrite the file, instead of failing. "
                         "Dropping a handful is better than truncating them mid-answer, and "
                         "better than blocking a 16,000-example run over one outlier.")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = [json.loads(l) for l in args.data.open()]
    lengths = [
        len(tok(tok.apply_chat_template(r["messages"], tokenize=False),
                add_special_tokens=False)["input_ids"])
        for r in rows
    ]
    raw_lengths = list(lengths)
    over = [n for n in lengths if n > args.max_seq]
    lengths.sort()
    pct = lambda p: lengths[max(0, int(len(lengths) * p) - 1)]
    print(f"  {args.data}: {len(rows)} examples")
    print(f"  median {pct(.5)}  p90 {pct(.9)}  p99 {pct(.99)}  max {lengths[-1]}  cap {args.max_seq}")
    print(f"  over cap: {len(over)} ({len(over)/len(rows):.1%})")

    if not over:
        print("  ok")
        return

    if args.drop:
        keep = [r for r, n in zip(rows, raw_lengths) if n <= args.max_seq]
        with args.data.open("w") as fh:
            for r in keep:
                fh.write(json.dumps(r) + "\n")
        print(f"  dropped {len(rows) - len(keep)} over-long example(s); {len(keep)} remain")
        print("  ok")
        return

    if len(over) / len(rows) > args.tolerate:
        print(f"\n  REFUSING TO TRAIN: {len(over)} examples would be truncated mid-answer.")
        print(f"  Raise max_seq_length to at least {lengths[-1]}, pass --drop, or shorten the data.")
        sys.exit(1)
    print("  ok")


if __name__ == "__main__":
    main()
