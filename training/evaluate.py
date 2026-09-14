"""Score base Qwen3-4B against the fine-tuned adapter on a CUDA GPU (Unsloth).

Scoring lives in evalcore.py and is shared with the Apple Silicon path, so both
backends report identical numbers. Writes the eval_report.json the frontend
renders as its Benchmark tab.

    python evaluate.py --adapter outputs/qwen3-4b-text2sql-lora --limit 300
"""
from __future__ import annotations

from unsloth import FastLanguageModel  # noqa: I001  (must precede transformers)

import argparse
import gc
import json
from pathlib import Path

import torch

from evalcore import build_report, print_table
from sqlutil import gold_is_runnable


@torch.inference_mode()
def generate(model, tokenizer, conversations, batch_size: int, max_new_tokens: int) -> list[str]:
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    outputs: list[str] = []
    for i in range(0, len(conversations), batch_size):
        chunk = conversations[i : i + batch_size]
        texts = [
            tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in chunk
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=3072).to("cuda")
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,               # greedy: reproducible scores
            pad_token_id=tokenizer.pad_token_id,
        )
        prompt_len = enc["input_ids"].shape[1]
        for row in gen:
            outputs.append(tokenizer.decode(row[prompt_len:], skip_special_tokens=True))
        print(f"    generated {min(i + batch_size, len(conversations))}/{len(conversations)}", flush=True)
    return outputs


def run_model(tag: str, model_path: str, conversations, max_seq: int, batch_size: int, max_new: int) -> list[str]:
    print(f"\n=== {tag}: {model_path} ===")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path, max_seq_length=max_seq, dtype=None, load_in_4bit=True
    )
    preds = generate(model, tokenizer, conversations, batch_size, max_new)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="unsloth/Qwen3-4B-Instruct-2507")
    ap.add_argument("--adapter", type=Path, default=Path("outputs/qwen3-4b-text2sql-lora"))
    ap.add_argument("--test", type=Path, default=Path("data/test.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("outputs/eval_report.json"))
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=192)
    ap.add_argument("--max-seq", type=int, default=3072)
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
        base_raw = run_model("BASE", args.base, conversations, args.max_seq, args.batch_size, args.max_new)
        base_cache.write_text(json.dumps(base_raw))

    tuned_raw = run_model(
        "FINE-TUNED", str(args.adapter), conversations, args.max_seq, args.batch_size, args.max_new
    )

    print("\nExecuting predictions against real SQLite databases ...")
    report, rows = build_report(
        examples, base_raw, tuned_raw,
        base_model=args.base, tuned_model=str(args.adapter),
        backend="cuda-unsloth", train_examples=train_examples,
    )

    args.out.write_text(json.dumps(report, indent=2))
    (args.out.parent / "eval_rows.json").write_text(json.dumps(rows, indent=2))
    print_table(report["metrics"], report["counts"])
    print(f"\nReport -> {args.out}")


if __name__ == "__main__":
    main()
