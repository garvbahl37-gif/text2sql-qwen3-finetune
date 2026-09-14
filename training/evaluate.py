"""Score base Qwen3-4B against the fine-tuned adapter on the held-out test set.

Primary metric is *execution accuracy*: build the real SQLite DB from each
example's schema, run both the gold and the predicted query, compare result
sets. String similarity is reported too, but never used as the headline number
(many correct queries are written differently from the gold).

Writes eval_report.json, which the web frontend renders as its Benchmark tab.

    python evaluate.py --adapter outputs/qwen3-4b-text2sql-lora --limit 300
"""
from __future__ import annotations

from unsloth import FastLanguageModel  # noqa: I001  (must precede transformers)

import argparse
import gc
import json
import re
from collections import defaultdict
from pathlib import Path

import torch

from sqlutil import build_db, gold_is_runnable, normalize_sql, results_match, run_sql

FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
SQL_KEYWORD = r"(?:WITH|SELECT|INSERT|UPDATE|DELETE)"
LINE_START = re.compile(rf"^\s*{SQL_KEYWORD}\b", re.IGNORECASE | re.MULTILINE)
ANYWHERE = re.compile(rf"\b{SQL_KEYWORD}\b", re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Pull a query out of a raw completion.

    Deliberately generous: the base model wraps SQL in markdown and prose, and
    penalising it for formatting instead of correctness would inflate the
    fine-tuned model's apparent win. Returns "" when there is no query at all
    (e.g. the model refused), so it scores as invalid rather than as garbage.
    """
    if m := FENCE.search(text):
        text = m.group(1)

    # Prefer a keyword that begins a line — that's where real SQL lives.
    if m := LINE_START.search(text):
        text = text[m.start() :]
    else:
        # Otherwise take the *last* keyword: models put prose first, SQL last,
        # so this avoids latching onto an English "select" in the preamble.
        matches = list(ANYWHERE.finditer(text))
        if not matches:
            return ""
        text = text[matches[-1].start() :]

    return text.split(";")[0].strip()


def score_one(context: str, gold_sql: str, pred_sql: str) -> tuple[bool, bool]:
    """Return (executes_cleanly, result_set_matches_gold)."""
    if not pred_sql:
        return False, False
    try:
        con = build_db(context)
    except Exception:
        return False, False
    try:
        pred_rows = run_sql(con, pred_sql)
    except Exception:
        con.close()
        return False, False
    try:
        gold_rows = run_sql(con, gold_sql)
    except Exception:
        con.close()
        return True, False
    ok = results_match(gold_rows, pred_rows, gold_sql)
    con.close()
    return True, ok


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


def summarise(rows: list[dict], key: str) -> dict:
    n = len(rows)
    return {
        "execution_accuracy": round(sum(r[f"{key}_exec_ok"] for r in rows) / n, 4),
        "valid_sql_rate": round(sum(r[f"{key}_runs"] for r in rows) / n, 4),
        "exact_match": round(sum(r[f"{key}_exact"] for r in rows) / n, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="unsloth/Qwen3-4B-Instruct-2507")
    ap.add_argument("--adapter", type=Path, default=Path("outputs/qwen3-4b-text2sql-lora"))
    ap.add_argument("--test", type=Path, default=Path("data/test.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("outputs/eval_report.json"))
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--max-seq", type=int, default=3072)
    ap.add_argument("--skip-base", action="store_true", help="reuse a previous base run to save GPU time")
    args = ap.parse_args()

    examples = [json.loads(l) for l in args.test.read_text().splitlines() if l.strip()][: args.limit]
    # Only score examples whose gold query runs, so a broken reference can't
    # be counted against either model.
    examples = [e for e in examples if gold_is_runnable(e["sql_context"], e["gold_sql"])]
    print(f"Scoring {len(examples)} examples with executable gold SQL")

    conversations = [e["messages"][:-1] for e in examples]  # drop the gold assistant turn

    base_cache = args.out.parent / "base_preds.json"
    if args.skip_base and base_cache.exists():
        base_raw = json.loads(base_cache.read_text())
        print(f"Reusing cached base predictions ({len(base_raw)})")
    else:
        base_raw = run_model("BASE", args.base, conversations, args.max_seq, args.batch_size, args.max_new)
        base_cache.parent.mkdir(parents=True, exist_ok=True)
        base_cache.write_text(json.dumps(base_raw))

    tuned_raw = run_model("FINE-TUNED", str(args.adapter), conversations, args.max_seq, args.batch_size, args.max_new)

    print("\nExecuting predictions against real SQLite databases ...")
    rows = []
    for ex, b_raw, t_raw in zip(examples, base_raw, tuned_raw):
        b_sql, t_sql = extract_sql(b_raw), extract_sql(t_raw)
        b_runs, b_ok = score_one(ex["sql_context"], ex["gold_sql"], b_sql)
        t_runs, t_ok = score_one(ex["sql_context"], ex["gold_sql"], t_sql)
        gold_norm = normalize_sql(ex["gold_sql"])
        rows.append({
            "id": ex.get("id"),
            "domain": ex.get("domain", ""),
            "complexity": ex.get("complexity", ""),
            "question": ex["sql_prompt"],
            "schema": ex["sql_context"],
            "gold_sql": ex["gold_sql"],
            "base_sql": b_sql,
            "base_raw_len": len(b_raw),
            "base_runs": b_runs,
            "base_exec_ok": b_ok,
            "base_exact": normalize_sql(b_sql) == gold_norm,
            "tuned_sql": t_sql,
            "tuned_raw_len": len(t_raw),
            "tuned_runs": t_runs,
            "tuned_exec_ok": t_ok,
            "tuned_exact": normalize_sql(t_sql) == gold_norm,
        })

    metrics = {"base": summarise(rows, "base"), "tuned": summarise(rows, "tuned")}

    by_complexity: dict[str, dict] = {}
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["complexity"] or "unknown"].append(r)
    for name, bucket in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        by_complexity[name] = {
            "n": len(bucket),
            "base": round(sum(r["base_exec_ok"] for r in bucket) / len(bucket), 4),
            "tuned": round(sum(r["tuned_exec_ok"] for r in bucket) / len(bucket), 4),
        }

    # Curate showcase samples: mostly wins, but keep honest failures in.
    wins = [r for r in rows if r["tuned_exec_ok"] and not r["base_exec_ok"]]
    both = [r for r in rows if r["tuned_exec_ok"] and r["base_exec_ok"]]
    losses = [r for r in rows if not r["tuned_exec_ok"] and r["base_exec_ok"]]
    neither = [r for r in rows if not r["tuned_exec_ok"] and not r["base_exec_ok"]]
    samples = wins[:14] + both[:4] + losses[:3] + neither[:3]

    report = {
        "base_model": args.base,
        "tuned_model": str(args.adapter),
        "dataset": "gretelai/synthetic_text_to_sql",
        "n_examples": len(rows),
        "decoding": "greedy",
        "metrics": metrics,
        "deltas": {
            k: round(metrics["tuned"][k] - metrics["base"][k], 4) for k in metrics["base"]
        },
        "by_complexity": by_complexity,
        "counts": {
            "tuned_win": len(wins), "both_correct": len(both),
            "tuned_regression": len(losses), "both_wrong": len(neither),
        },
        "samples": samples,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    (args.out.parent / "eval_rows.json").write_text(json.dumps(rows, indent=2))

    print("\n" + "=" * 62)
    print(f"{'metric':<22}{'base':>12}{'fine-tuned':>14}{'delta':>12}")
    print("-" * 62)
    for k in metrics["base"]:
        b, t = metrics["base"][k], metrics["tuned"][k]
        print(f"{k:<22}{b:>11.1%}{t:>13.1%}{t - b:>+11.1%}")
    print("=" * 62)
    print(f"wins {len(wins)}  |  both right {len(both)}  |  regressions {len(losses)}  |  both wrong {len(neither)}")
    print(f"\nReport -> {args.out}")


if __name__ == "__main__":
    main()
