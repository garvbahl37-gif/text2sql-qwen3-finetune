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
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = [json.loads(l) for l in args.data.open()]
    lengths = [
        len(tok(tok.apply_chat_template(r["messages"], tokenize=False),
                add_special_tokens=False)["input_ids"])
        for r in rows
    ]
    over = [n for n in lengths if n > args.max_seq]
    lengths.sort()
    pct = lambda p: lengths[max(0, int(len(lengths) * p) - 1)]
    print(f"  {args.data}: {len(rows)} examples")
    print(f"  median {pct(.5)}  p90 {pct(.9)}  p99 {pct(.99)}  max {lengths[-1]}  cap {args.max_seq}")
    print(f"  over cap: {len(over)} ({len(over)/len(rows):.1%})")

    if len(over) / len(rows) > args.tolerate:
        print(f"\n  REFUSING TO TRAIN: {len(over)} examples would be truncated mid-answer.")
        print(f"  Raise max_seq_length to at least {lengths[-1]}, or shorten the data.")
        sys.exit(1)
    print("  ok")


if __name__ == "__main__":
    main()
