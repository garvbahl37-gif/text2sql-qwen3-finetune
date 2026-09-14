"""QLoRA SFT of Qwen3-4B-Instruct on text-to-SQL, via Unsloth.

Tuned for *minimum GPU-hours*: 4-bit base, LoRA on all projections, loss masked
to assistant tokens only, one epoch over verified examples, and a sequence
length measured from the data rather than left at the usual 2048.

Runs on any CUDA GPU: a rented A40/L40S, or a free Kaggle T4 (fp16 is selected
automatically there, since Turing has no bf16).

    python train.py --data data --out outputs/qwen3-4b-text2sql-lora
"""
from __future__ import annotations

# Unsloth patches transformers/trl on import, so it must come first.
from unsloth import FastLanguageModel, is_bfloat16_supported  # noqa: I001

import argparse
import dataclasses
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import train_on_responses_only

# Qwen3 ChatML turn markers — used to mask the loss onto assistant tokens only.
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"


def supported_config(**kwargs) -> SFTConfig:
    """Build an SFTConfig, dropping keys this TRL version doesn't know.

    TRL renamed several fields across versions (max_seq_length -> max_length,
    evaluation_strategy -> eval_strategy). Filtering keeps the run from dying
    on a rented GPU because of a library bump.
    """
    valid = {f.name for f in dataclasses.fields(SFTConfig)}
    aliases = {"max_seq_length": "max_length", "evaluation_strategy": "eval_strategy"}
    cleaned = {}
    for k, v in kwargs.items():
        if k in valid:
            cleaned[k] = v
        elif aliases.get(k) in valid:
            cleaned[aliases[k]] = v
        else:
            print(f"  [config] dropping unsupported option: {k}")
    return SFTConfig(**cleaned)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen3-4B-Instruct-2507")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("outputs/qwen3-4b-text2sql-lora"))
    # Measured over the prepared data: median 194 tokens, p99 396, longest 581.
    # The usual 2048 default is ~3x oversized and costs memory for nothing.
    ap.add_argument("--max-seq", type=int, default=640)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-eval", action="store_true", help="skip periodic val loss to save GPU time")
    args = ap.parse_args()

    print(f"Loading {args.model} in 4-bit ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq,
        dtype=None,          # auto: bf16 on Ampere+, fp16 otherwise
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        lora_alpha=args.rank,          # alpha == r is a stable default for SFT
        lora_dropout=0.0,              # 0 lets Unsloth use its fused fast path
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
    )

    ds = load_dataset(
        "json",
        data_files={"train": str(args.data / "train.jsonl"), "val": str(args.data / "valid.jsonl")},
    )

    def render(batch):
        return {
            "text": [
                tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                for m in batch["messages"]
            ]
        }

    keep = ["text"]
    train_ds = ds["train"].map(render, batched=True, remove_columns=[c for c in ds["train"].column_names if c not in keep])
    val_ds = ds["val"].map(render, batched=True, remove_columns=[c for c in ds["val"].column_names if c not in keep])
    print(f"train={len(train_ds)}  val={len(val_ds)}")
    print("\n--- sample rendered example ---")
    print(train_ds[0]["text"][:900])
    print("--- end sample ---\n")

    cfg = supported_config(
        output_dir=str(args.out),
        dataset_text_field="text",
        max_seq_length=args.max_seq,
        packing=False,                 # required: response-only masking needs real turn boundaries
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        optim="adamw_8bit",
        weight_decay=0.01,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        logging_steps=10,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,
        evaluation_strategy="no" if args.no_eval else "steps",
        eval_steps=100,
        per_device_eval_batch_size=args.batch_size,
        seed=args.seed,
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
        run_name="qwen3-4b-text2sql",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=None if args.no_eval else val_ds,
        args=cfg,
    )

    # Mask the loss so the model is graded on the SQL it writes, not on
    # re-predicting the schema it was handed. Meaningful quality win here.
    trainer = train_on_responses_only(
        trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART
    )

    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU: {gpu.name}  ({gpu.total_memory / 1e9:.1f} GB)")
    stats = trainer.train()

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out))
    tokenizer.save_pretrained(str(args.out))

    summary = {
        "base_model": args.model,
        "train_examples": len(train_ds),
        "lora_rank": args.rank,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "effective_batch": args.batch_size * args.grad_accum,
        "train_runtime_sec": round(stats.metrics.get("train_runtime", 0), 1),
        "train_loss": round(stats.metrics.get("train_loss", 0), 4),
        "peak_gpu_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
    }
    (args.out / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nLoRA adapter saved to {args.out}")


if __name__ == "__main__":
    main()
