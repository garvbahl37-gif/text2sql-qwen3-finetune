"""Convert a PEFT/Unsloth LoRA adapter to MLX format.

Lets an adapter trained on CUDA be evaluated with the Apple Silicon harness, so
runs from different machines can be scored on exactly the same held-out set.

The two libraries store the same maths transposed:

  PEFT  lora_A.weight (r, in)    lora_B.weight (out, r)   y + (alpha/r)·(x @ Aᵀ @ Bᵀ)
  MLX   lora_a       (in, r)     lora_b        (r, out)   y + scale·((x @ a) @ b)

so a = Aᵀ, b = Bᵀ, and scale = alpha / r.

    python peft_to_mlx.py --src <peft dir> --dst <mlx dir>
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
from safetensors.numpy import load_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    args = ap.parse_args()

    cfg = json.loads((args.src / "adapter_config.json").read_text())
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scale = alpha / r

    weights = load_file(args.src / "adapter_model.safetensors")
    out, pairs = {}, 0
    for name, arr in weights.items():
        if ".lora_A" not in name and ".lora_B" not in name:
            continue
        # base_model.model.model.layers.N.self_attn.q_proj.lora_A.weight
        clean = name.replace("base_model.model.", "")
        for suffix, target in ((".lora_A.weight", ".lora_a"), (".lora_B.weight", ".lora_b")):
            if clean.endswith(suffix):
                out[clean[: -len(suffix)] + target] = mx.array(arr.T)   # transpose
                pairs += 1
                break

    args.dst.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.dst / "adapters.safetensors"), out)
    (args.dst / "adapter_config.json").write_text(json.dumps({
        "fine_tune_type": "lora",
        "num_layers": -1,
        "lora_parameters": {
            "keys": sorted({k.rsplit(".", 1)[0].split(".", 3)[-1] for k in out}),
            "rank": r, "scale": scale, "dropout": 0.0,
        },
    }, indent=2))
    for extra in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if (args.src / extra).exists():
            shutil.copy(args.src / extra, args.dst / extra)

    print(f"  converted {pairs} tensors ({pairs//2} LoRA pairs) -> {args.dst}")
    print(f"  rank {r}, alpha {alpha}, scale {scale}")
    k = next(iter(out))
    print(f"  example: {k}  shape {out[k].shape}")


if __name__ == "__main__":
    main()
