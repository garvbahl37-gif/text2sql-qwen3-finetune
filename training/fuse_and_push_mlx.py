"""Fuse the MLX LoRA adapter into full-precision weights and publish to the Hub.

The adapter is trained against a 4-bit base to fit in unified memory, but it is
fused into the *full-precision* base here rather than into a dequantised copy.
That is standard QLoRA practice: it keeps 4-bit quantisation error out of the
published weights, and produces safetensors that transformers -- and therefore
the Hugging Face Space -- can load directly.

    python fuse_and_push_mlx.py --repo bharatverse11/qwen3-4b-text2sql
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

FULL_PRECISION_BASE = "Qwen/Qwen3-4B-Instruct-2507"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=FULL_PRECISION_BASE,
                    help="full-precision base to fuse into (NOT the 4-bit training base)")
    ap.add_argument("--adapter", type=Path, default=Path("outputs/qwen3-4b-text2sql-mlx"))
    ap.add_argument("--save-path", type=Path, default=Path("outputs/merged-16bit"))
    ap.add_argument("--repo", help="HF repo id, e.g. bharatverse11/qwen3-4b-text2sql")
    ap.add_argument("--local-only", action="store_true", help="fuse to disk without uploading")
    args = ap.parse_args()

    if not args.local_only and not args.repo:
        raise SystemExit("Pass --repo <user>/<name>, or --local-only to skip uploading.")
    if not args.local_only and not os.getenv("HF_TOKEN"):
        raise SystemExit("HF_TOKEN is not set. Run:  set -a && . ../.env && set +a")

    cfg = args.adapter / "adapter_config.json"
    if not cfg.exists():
        raise SystemExit(f"No adapter at {args.adapter}. Train first.")
    trained_on = json.loads(cfg.read_text()).get("model", "unknown")
    print(f"Adapter trained against : {trained_on}")
    print(f"Fusing into             : {args.base}")
    if "4bit" in str(trained_on) and "4bit" in args.base:
        print("  WARNING: fusing into a 4-bit base bakes quantisation error into the\n"
              "           published weights. Pass the full-precision repo instead.")

    # mlx_lm.load() fetches only model files, but mlx_lm's save() then calls
    # snapshot_download(local_files_only=True), which demands a COMPLETE
    # snapshot -- including .gitattributes, LICENSE and README.md. Without
    # them the fuse dies at the very last step with IncompleteSnapshotError.
    # Pre-fetch the whole repo (the missing files are a few KB).
    if "/" in args.base and not Path(args.base).exists():
        from huggingface_hub import snapshot_download

        print("Completing the base model snapshot ...")
        snapshot_download(args.base)

    cmd = [sys.executable, "-m", "mlx_lm", "fuse",
           "--model", args.base,
           "--adapter-path", str(args.adapter),
           "--save-path", str(args.save_path)]
    if args.repo and not args.local_only:
        cmd += ["--upload-repo", args.repo]

    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    print(f"\nFused weights -> {args.save_path}")
    if args.repo and not args.local_only:
        print(f"Published     -> https://huggingface.co/{args.repo}")
        print(f"\nSet this in your Hugging Face Space:\n  MODEL_ID={args.repo}")


if __name__ == "__main__":
    main()
