"""Merge the LoRA adapter into the base weights and publish to the HF Hub.

The Space serves the merged 16-bit model. The optional GGUF export gives the
Space a free CPU fallback for when ZeroGPU quota runs out.

    python merge_and_push.py --repo yourname/qwen3-4b-text2sql --gguf
"""
from __future__ import annotations

from unsloth import FastLanguageModel  # noqa: I001

import argparse
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, default=Path("outputs/qwen3-4b-text2sql-lora"))
    ap.add_argument("--repo", required=True, help="HF repo id, e.g. yourname/qwen3-4b-text2sql")
    ap.add_argument("--local-only", action="store_true", help="merge to disk without uploading")
    ap.add_argument("--gguf", action="store_true", help="also upload a q4_k_m GGUF for CPU inference")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--max-seq", type=int, default=4096)
    args = ap.parse_args()

    token = os.getenv("HF_TOKEN")
    if not args.local_only and not token:
        raise SystemExit("HF_TOKEN is not set. export HF_TOKEN=hf_... (write scope)")

    print(f"Loading adapter {args.adapter} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.adapter), max_seq_length=args.max_seq, dtype=None, load_in_4bit=False
    )

    merged_dir = args.adapter.parent / "merged-16bit"
    print(f"Merging to 16-bit -> {merged_dir}")
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

    if args.local_only:
        print(f"Done (local only): {merged_dir}")
        return

    print(f"Pushing merged weights -> https://huggingface.co/{args.repo}")
    model.push_to_hub_merged(
        args.repo, tokenizer, save_method="merged_16bit", token=token, private=args.private
    )

    if args.gguf:
        gguf_repo = f"{args.repo}-gguf"
        print(f"Exporting q4_k_m GGUF -> https://huggingface.co/{gguf_repo}")
        model.push_to_hub_gguf(
            gguf_repo, tokenizer, quantization_method=["q4_k_m"], token=token, private=args.private
        )

    print("\nPublished. Point the Space at:")
    print(f"  MODEL_ID={args.repo}")


if __name__ == "__main__":
    main()
