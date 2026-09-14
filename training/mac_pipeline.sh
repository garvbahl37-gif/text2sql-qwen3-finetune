#!/usr/bin/env bash
# End-to-end fine-tune on Apple Silicon using MLX. No rented GPU, no CUDA.
#
#   Requires : Apple Silicon, macOS 13.5+, ~16GB free unified memory, ~25GB disk
#   Measured : M5 Pro (24GB, 16-core GPU)
#
#   set -a && . ../.env && set +a     # loads HF_TOKEN and HF_REPO
#   bash mac_pipeline.sh
#
# Every stage is idempotent: re-running skips work that already succeeded.

set -euo pipefail
cd "$(dirname "$0")"

: "${HF_TOKEN:?Run: set -a && . ../.env && set +a}"
: "${HF_REPO:?Run: set -a && . ../.env && set +a}"

PY="../.venv/bin/python"
ADAPTER="outputs/qwen3-4b-text2sql-mlx"
TRAIN_SIZE="${TRAIN_SIZE:-4000}"
TEST_SIZE="${TEST_SIZE:-300}"
FULL_BASE="Qwen/Qwen3-4B-Instruct-2507"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

if [ ! -x "$PY" ]; then
  log "Creating the virtualenv"
  python3 -m venv ../.venv
  ../.venv/bin/pip install -q --upgrade pip
  ../.venv/bin/pip install -q "mlx-lm>=0.21.0" datasets huggingface_hub transformers
fi

if [ ! -f data/train.jsonl ]; then
  log "Preparing data (validating every gold query executes)"
  $PY prepare_data.py --train-size "$TRAIN_SIZE" --val-size 200 --test-size "$TEST_SIZE"
else
  log "Data already prepared, skipping"
fi

if [ ! -f "$ADAPTER/adapters.safetensors" ]; then
  # iters, the LR decay horizon and the dataset size have to agree, or you
  # silently train for the wrong number of epochs. Check before spending an hour.
  ROWS=$(wc -l < data/train.jsonl | tr -d ' ')
  BS=$(awk '/^batch_size:/{print $2}' mlx_lora_config.yaml)
  ACC=$(awk '/^grad_accumulation_steps:/{print $2}' mlx_lora_config.yaml)
  ITERS=$(awk '/^iters:/{print $2}' mlx_lora_config.yaml)
  DECAY=$(awk '/^  arguments:/{gsub(/[],[]/,""); print $3}' mlx_lora_config.yaml)
  EXPECT_ITERS=$((ROWS / BS))
  EXPECT_DECAY=$((ITERS / ACC))
  if [ "$ITERS" != "$EXPECT_ITERS" ]; then
    echo "MISMATCH: $ROWS examples at batch_size $BS is $EXPECT_ITERS iters/epoch, but"
    echo "  mlx_lora_config.yaml says iters: $ITERS. Fix it, or set TRAIN_SIZE=$((ITERS * BS))." >&2
    exit 1
  fi
  if [ "$DECAY" != "$EXPECT_DECAY" ]; then
    echo "MISMATCH: the LR schedule advances once per optimizer update, so its decay" >&2
    echo "  horizon should be iters/grad_accumulation_steps = $EXPECT_DECAY, not $DECAY." >&2
    exit 1
  fi
  log "Training the LoRA adapter ($ROWS examples, $ITERS iters, ~21 min on an M5 Pro)"
  log "Close memory-hungry apps first -- this needs most of the GPU's memory budget."
  $PY -m mlx_lm lora -c mlx_lora_config.yaml
else
  log "Adapter already trained, skipping"
fi

log "Evaluating base vs fine-tuned (execution accuracy)"
$PY evaluate_mlx.py --adapter "$ADAPTER" --limit "$TEST_SIZE"

log "Fusing into full-precision weights and publishing"
# Fuse into the FULL-PRECISION base, not the 4-bit one the adapter trained
# against, so no quantisation error is baked into the published weights.
$PY fuse_and_push_mlx.py --base "$FULL_BASE" --adapter "$ADAPTER" --repo "$HF_REPO"

log "Done"
cat <<EOF

Next steps
  1. cp outputs/eval_report.json ../web/data/eval_report.json
  2. Set MODEL_ID=$HF_REPO in your Hugging Face Space
  3. Deploy the frontend:  cd ../web && vercel --prod

EOF
