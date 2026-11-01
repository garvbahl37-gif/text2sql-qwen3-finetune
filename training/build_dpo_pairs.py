"""Build execution-graded preference pairs for DPO.

Supervised fine-tuning only ever shows the model correct SQL, so it never learns
what its own wrong answers look like. Preference training does: it shows a
correct query beside a specific incorrect one and pushes probability toward the
former. The 2025 literature reports this as the largest single gain in
text-to-SQL -- ExCoT-DPO moves BIRD from 57.4 to 68.5 and Spider from 78.8 to
86.6 after SFT.

The grading here is execution, not similarity: run the model's query and the
reference against the same database and compare result sets. A pair is kept only
when the model is genuinely wrong, so the signal is real failures rather than
stylistic difference.

    python build_dpo_pairs.py --adapter outputs/run4-mlx --limit 2500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

from evalcore import extract_sql, score_one


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="mlx-community/Qwen3-4B-Instruct-2507-4bit")
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/train.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/dpo_pairs.jsonl"))
    ap.add_argument("--limit", type=int, default=2500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--temp", type=float, default=0.8,
                    help="sample rather than decode greedily, so the failures collected are "
                         "the ones the model actually produces in use")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.data.open()][: args.limit]
    print(f"Generating on {len(rows)} training questions with {args.adapter}")

    model, tok = load(args.base, adapter_path=str(args.adapter))
    sampler = make_sampler(temp=args.temp)
    prompts = [
        tok.encode(tok.apply_chat_template(r["messages"][:-1], tokenize=False,
                                           add_generation_prompt=True))
        for r in rows
    ]

    preds, t0 = [], time.time()
    for i in range(0, len(prompts), args.batch_size):
        resp = batch_generate(model, tok, prompts[i : i + args.batch_size],
                              max_tokens=args.max_tokens, sampler=sampler, verbose=False)
        preds.extend(resp.texts)
        done = min(i + args.batch_size, len(prompts))
        print(f"    {done}/{len(prompts)}  ({done / max(time.time() - t0, 1e-6):.1f}/s)", flush=True)
    del model, tok
    mx.clear_cache()

    pairs, correct, unusable = [], 0, 0
    for r, raw in zip(rows, preds):
        sql = extract_sql(raw)
        if not sql:
            unusable += 1
            continue
        runs, ok = score_one(r["sql_context"], r["gold_sql"], sql)
        if ok:
            correct += 1
            continue
        # The model is wrong here: prefer the reference over this exact mistake.
        user = r["messages"][1]["content"]
        pairs.append({
            "prompt": user,
            "chosen": r["gold_sql"],
            "rejected": sql,
            "failure": "did_not_run" if not runs else "ran_but_wrong",
            "complexity": r.get("complexity", ""),
            "source": r.get("source", ""),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")

    from collections import Counter
    print(f"\n  questions           : {len(rows)}")
    print(f"  model already right : {correct} ({correct/len(rows):.1%})")
    print(f"  no SQL produced     : {unusable}")
    print(f"  preference pairs    : {len(pairs)} -> {args.out}")
    print("\n  pairs by failure mode:")
    for k, v in Counter(p["failure"] for p in pairs).most_common():
        print(f"    {v:>5}  {k}")
    print("\n  pairs by complexity:")
    for k, v in Counter(p["complexity"] for p in pairs).most_common(6):
        print(f"    {v:>5}  {k or '(none)'}")


if __name__ == "__main__":
    main()
