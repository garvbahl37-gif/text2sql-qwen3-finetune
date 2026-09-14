#!/usr/bin/env bash
# One-shot pipeline for a fresh RunPod pod.
#
#   Template : RunPod PyTorch 2.4+ (CUDA 12.1+)
#   GPU      : A40 / L40S / A6000 (48GB)  -- ~45 min end to end
#   Disk     : 60GB container + 20GB volume
#
#   export HF_TOKEN=hf_...        # write scope
#   export HF_REPO=yourname/qwen3-4b-text2sql
#   bash runpod_bootstrap.sh
#
# Every stage is idempotent: re-running skips work that already succeeded.

set -euo pipefail
cd "$(dirname "$0")"

: "${HF_TOKEN:?export HF_TOKEN=hf_... (write scope) first}"
: "${HF_REPO:?export HF_REPO=yourname/qwen3-4b-text2sql first}"

TRAIN_SIZE="${TRAIN_SIZE:-8000}"
TEST_SIZE="${TEST_SIZE:-300}"
ADAPTER="outputs/qwen3-4b-text2sql-lora"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

if [ ! -f .bootstrap_deps_done ]; then
  log "Installing dependencies (~4 min)"
  pip install -q --upgrade pip
  # Unsloth first: it pins a consistent torch/xformers/triton set.
  pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  pip install -q --no-deps trl peft accelerate bitsandbytes
  pip install -q datasets huggingface_hub sentencepiece protobuf
  touch .bootstrap_deps_done
else
  log "Dependencies already installed, skipping"
fi

if [ ! -f data/train.jsonl ]; then
  log "Preparing data (validating gold SQL executes)"
  python prepare_data.py --train-size "$TRAIN_SIZE" --test-size "$TEST_SIZE"
else
  log "Data already prepared, skipping"
fi

if [ ! -f "$ADAPTER/adapter_model.safetensors" ]; then
  log "Training QLoRA adapter (the long stage, ~45 min on an A40)"
  python train.py --data data --out "$ADAPTER"
else
  log "Adapter already trained, skipping"
fi

log "Evaluating base vs fine-tuned (execution accuracy)"
python evaluate.py --adapter "$ADAPTER" --limit "$TEST_SIZE"

log "Merging and publishing to the Hub"
python merge_and_push.py --adapter "$ADAPTER" --repo "$HF_REPO" --gguf

log "Done"
cat <<EOF

Next steps
  1. Copy outputs/eval_report.json into the repo at web/data/eval_report.json
       runpodctl send outputs/eval_report.json
     (or just scp / download it from the pod's file browser)
  2. Set MODEL_ID=$HF_REPO in the Hugging Face Space
  3. TERMINATE THIS POD -- it bills per second while it exists

EOF
