"""Score base Qwen3-4B against the fine-tuned adapter, on Apple Silicon via MLX.

Same scoring as the CUDA path (see evalcore.py): build the real SQLite database
from each example's own schema, run both the generated and the reference query,
compare result sets. Writes the eval_report.json the web frontend renders.

    python evaluate_mlx.py --adapter outputs/qwen3-4b-text2sql-mlx --limit 300
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

from evalcore import build_report, print_table
from sqlutil import gold_is_runnable

DEFAULT_BASE = "mlx-community/Qwen3-4B-Instruct-2507-4bit"


def generate_all(
    tag: str,
    base: str,
    adapter: str | None,
    conversations: list[list[dict]],
    batch_size: int,
    max_tokens: int,
) -> list[str]:
    print(f"\n=== {tag}: {base}{f' + {adapter}' if adapter else ''} ===")
    model, tokenizer = load(base, adapter_path=adapter)
    sampler = make_sampler(temp=0.0)  # greedy, so scores are reproducible

    prompts = [
        tokenizer.encode(
            tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
        )
        for c in conversations
    ]

    outputs: list[str] = []
    started = time.time()
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        resp = batch_generate(
            model, tokenizer, chunk, max_tokens=max_tokens, sampler=sampler, verbose=False
        )
        outputs.extend(resp.texts)
        done = min(i + batch_size, len(prompts))
        rate = done / max(time.time() - started, 1e-6)
        print(f"    {done}/{len(prompts)}  ({rate:.1f} examples/s)", flush=True)

    del model, tokenizer
    gc.collect()
    mx.clear_cache()
    return outputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--adapter", type=Path, default=Path("outputs/qwen3-4b-text2sql-mlx"))
    ap.add_argument("--test", type=Path, default=Path("data/test.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("outputs/eval_report.json"))
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--skip-base", action="store_true", help="reuse cached base predictions")
    args = ap.parse_args()

    examples = [json.loads(l) for l in args.test.read_text().splitlines() if l.strip()][: args.limit]
    # Only score examples whose reference query runs, so a broken gold answer
    # can't be counted against either model.
    examples = [e for e in examples if gold_is_runnable(e["sql_context"], e["gold_sql"])]
    print(f"Scoring {len(examples)} examples with executable gold SQL")

    conversations = [e["messages"][:-1] for e in examples]

    # Record how much data the adapter actually saw, so the site can state it
    # rather than hardcoding a number that drifts when TRAIN_SIZE changes.
    train_file = args.test.parent / "train.jsonl"
    train_examples = sum(1 for _ in train_file.open()) if train_file.exists() else None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    base_cache = args.out.parent / "base_preds.json"
    if args.skip_base and base_cache.exists():
        base_raw = json.loads(base_cache.read_text())
        print(f"Reusing cached base predictions ({len(base_raw)})")
    else:
        base_raw = generate_all("BASE", args.base, None, conversations, args.batch_size, args.max_tokens)
        base_cache.write_text(json.dumps(base_raw))

    tuned_raw = generate_all(
        "FINE-TUNED", args.base, str(args.adapter), conversations, args.batch_size, args.max_tokens
    )

    print("\nExecuting predictions against real SQLite databases ...")
    report, rows = build_report(
        examples, base_raw, tuned_raw,
        base_model=args.base,
        tuned_model=str(args.adapter),
        backend="mlx-apple-silicon",
        train_examples=train_examples,
    )

    args.out.write_text(json.dumps(report, indent=2))
    (args.out.parent / "eval_rows.json").write_text(json.dumps(rows, indent=2))
    print_table(report["metrics"], report["counts"])
    print(f"\nReport -> {args.out}")


if __name__ == "__main__":
    main()
