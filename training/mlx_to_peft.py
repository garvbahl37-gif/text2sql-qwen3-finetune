"""Convert an MLX LoRA adapter back to PEFT format.

The inverse of peft_to_mlx.py. PEFT stores lora_A as (r, in) and lora_B as
(out, r); MLX stores lora_a as (in, r) and lora_b as (r, out). The mapping is a
transpose in both directions, so a round trip is lossless and can be checked.

Needed because TRL's DPOTrainer reads PEFT adapters, while training and
evaluation here happen in MLX.

    python mlx_to_peft.py --src outputs/run4-mlx --dst outputs/run4-peft \
        --base unsloth/Qwen3-4B-Instruct-2507
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--base", default="unsloth/Qwen3-4B-Instruct-2507")
    args = ap.parse_args()

    cfg = json.loads((args.src / "adapter_config.json").read_text())
    lp = cfg.get("lora_parameters", {})
    r = int(lp.get("rank", 32))
    scale = float(lp.get("scale", 1.0))
    alpha = int(round(scale * r))

    w = load_file(args.src / "adapters.safetensors")
    out, targets = {}, set()
    for name, arr in w.items():
        for suffix, peft in ((".lora_a", ".lora_A.weight"), (".lora_b", ".lora_B.weight")):
            if name.endswith(suffix):
                stem = name[: -len(suffix)]
                out["base_model.model." + stem + peft] = np.ascontiguousarray(arr.T)
                targets.add(stem.rsplit(".", 1)[-1])
                break

    args.dst.mkdir(parents=True, exist_ok=True)
    save_file(out, str(args.dst / "adapter_model.safetensors"))
    (args.dst / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM",
        "base_model_name_or_path": args.base,
        "r": r, "lora_alpha": alpha, "lora_dropout": 0.0, "bias": "none",
        "target_modules": sorted(targets), "fan_in_fan_out": False,
        "inference_mode": False,
    }, indent=2))
    for extra in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if (args.src / extra).exists():
            shutil.copy(args.src / extra, args.dst / extra)

    print(f"  {len(out)} tensors -> {args.dst}")
    print(f"  rank {r}, alpha {alpha} (scale {scale}), targets {sorted(targets)}")


if __name__ == "__main__":
    main()
